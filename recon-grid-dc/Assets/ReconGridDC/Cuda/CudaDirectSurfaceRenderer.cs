using System;
using System.Runtime.InteropServices;
using UnityEngine;
using UnityEngine.Rendering;

namespace ReconGridDC.Cuda
{
    /// <summary>
    /// Owns the Unity D3D11 buffers used by the phase-4 CUDA direct surface path. CUDA copies
    /// the already-expanded Dual Contouring soup into these buffers on Unity's render thread.
    /// No surface vertices or normals are read back to managed memory during normal operation.
    /// </summary>
    [DisallowMultipleComponent]
    public sealed class CudaDirectSurfaceRenderer : MonoBehaviour
    {
        const string Dll = "LiverCudaSim";

        [StructLayout(LayoutKind.Sequential)]
        struct DirectSurfaceStatsNative
        {
            public int supported, configured, registered, capacity;
            public ulong gpuCopyBytes, dispatchCount;
            public int lastError, cudaDeviceOrdinal, registrationStage, cudaErrorCode;
            public int positionByteWidth, positionUsage, positionBindFlags, positionMiscFlags, positionStructureByteStride;
        }

        [DllImport(Dll)] static extern int LCS_DirectSurfaceSetBuffers(
            IntPtr positionBuffer, IntPtr normalBuffer, IntPtr auxBuffer,
            IntPtr vertexCountBuffer, int capacity);
        [DllImport(Dll)] static extern void LCS_DirectSurfaceRelease();
        [DllImport(Dll)] static extern IntPtr LCS_GetDirectSurfaceRenderEventFunc();
        [DllImport(Dll)] static extern int LCS_DirectSurfaceGetStats(out DirectSurfaceStatsNative stats);

        [Header("CUDA Migration Phase 4 - GPU Direct Surface")]
        [Tooltip("Draws the CUDA Dual Contouring output directly from D3D11 GPU buffers. This requires Direct3D11 and falls back to the CPU Mesh path when unavailable.")]
        public bool enabledForManager = true;
        [Tooltip("Adds this much grid extent around the fitted bounds so moving or cut pieces are not prematurely culled.")]
        [Min(1f)] public float boundsPaddingMultiplier = 3f;

        [Header("Diagnostics (runtime)")]
        [SerializeField] bool directSurfaceActive;
        [SerializeField] string directSurfaceStatus = "Not configured.";
        [SerializeField] int surfaceCapacity;
        [SerializeField] bool d3d11Compatible;
        [SerializeField] int nativeRegistered;
        [SerializeField] ulong gpuCopyBytes;
        [SerializeField] ulong dispatchCount;
        [SerializeField] int lastNativeError;
        [SerializeField] int cudaDeviceOrdinal = -1;
        [SerializeField] int registrationStage;
        [SerializeField] int cudaRegistrationError;
        [SerializeField] int positionBufferByteWidth;
        [SerializeField] int positionBufferUsage;
        [SerializeField] int positionBufferBindFlags;
        [SerializeField] int positionBufferMiscFlags;
        [SerializeField] int positionBufferStructureStride;

        GraphicsBuffer _positions;
        GraphicsBuffer _normals;
        GraphicsBuffer _aux;
        GraphicsBuffer _vertexCount;
        GraphicsBuffer _indirectArgs;
        ComputeShader _indirectArgsShader;
        int _indirectArgsKernel = -1;
        CommandBuffer _drawCommands;
        Camera _drawCamera;
        Material _material;
        Bounds _bounds;
        IntPtr _renderEvent;
        bool _visible = true;
        bool _nativeConfigured;

        public bool IsActive => directSurfaceActive;
        public string Status => directSurfaceStatus;
        public ulong GpuCopyBytes => gpuCopyBytes;
        public ulong DispatchCount => dispatchCount;
        public int LastNativeError => lastNativeError;

        public bool Configure(int capacity, Bounds bounds, Shader shader, Color color)
        {
            Release();
            surfaceCapacity = Mathf.Max(0, capacity);
            _bounds = bounds;
            _bounds.extents *= Mathf.Max(1f, boundsPaddingMultiplier);
            d3d11Compatible = SystemInfo.graphicsDeviceType == GraphicsDeviceType.Direct3D11;
            if (!enabledForManager)
            {
                directSurfaceStatus = "Disabled by Inspector.";
                return false;
            }
            if (!d3d11Compatible)
            {
                directSurfaceStatus = $"GPU direct surface requires Direct3D11; active API is {SystemInfo.graphicsDeviceType}.";
                return false;
            }
            if (surfaceCapacity <= 0 || shader == null)
            {
                directSurfaceStatus = "Invalid surface capacity or direct-surface shader.";
                return false;
            }

            try
            {
                // On D3D11 Unity's Structured GraphicsBuffer is not accepted by
                // cudaGraphicsD3D11RegisterResource. Raw buffers keep the same contiguous
                // float layout for CUDA and are decoded by ByteAddressBuffer in the shader.
                _positions = new GraphicsBuffer(GraphicsBuffer.Target.Raw, surfaceCapacity * 3, sizeof(float));
                _normals = new GraphicsBuffer(GraphicsBuffer.Target.Raw, surfaceCapacity * 3, sizeof(float));
                _aux = new GraphicsBuffer(GraphicsBuffer.Target.Raw, surfaceCapacity * 4, sizeof(float));
                // CUDA writes the live vertex count to this Raw buffer. A Unity compute dispatch
                // then transfers it to indirect args entirely on the GPU.
                _vertexCount = new GraphicsBuffer(GraphicsBuffer.Target.Raw, 1, sizeof(uint));
                _indirectArgs = new GraphicsBuffer(GraphicsBuffer.Target.IndirectArguments, 4, sizeof(uint));
                _indirectArgsShader = Resources.Load<ComputeShader>("CudaDirectSurfaceIndirectArgs");
                if (_indirectArgsShader == null)
                {
                    directSurfaceStatus = "CudaDirectSurfaceIndirectArgs.compute is missing from Assets/Resources.";
                    Release();
                    return false;
                }
                _indirectArgsKernel = _indirectArgsShader.FindKernel("BuildIndirectArgs");
                _indirectArgsShader.SetBuffer(_indirectArgsKernel, "_CudaSurfaceVertexCount", _vertexCount);
                _indirectArgsShader.SetBuffer(_indirectArgsKernel, "_IndirectArgs", _indirectArgs);
                _drawCamera = Camera.main;
                if (_drawCamera == null)
                {
                    directSurfaceStatus = "GPU direct surface requires a tagged Main Camera.";
                    Release();
                    return false;
                }
                _drawCommands = new CommandBuffer { name = "CUDA Direct Surface" };
                _drawCamera.AddCommandBuffer(CameraEvent.BeforeForwardAlpha, _drawCommands);
                _material = new Material(shader) { name = "Runtime_CudaDirectSurface" };
                _material.SetColor("_Color", color);
                _material.SetBuffer("_CudaSurfacePositions", _positions);
                _material.SetBuffer("_CudaSurfaceNormals", _normals);
                _material.SetBuffer("_CudaSurfaceAux", _aux);
                _material.SetBuffer("_CudaSurfaceVertexCount", _vertexCount);

                int rc = LCS_DirectSurfaceSetBuffers(_positions.GetNativeBufferPtr(), _normals.GetNativeBufferPtr(),
                    _aux.GetNativeBufferPtr(), _vertexCount.GetNativeBufferPtr(), surfaceCapacity);
                if (rc != 0)
                {
                    lastNativeError = rc;
                    directSurfaceStatus = $"LCS_DirectSurfaceSetBuffers failed ({rc}). CPU Mesh fallback remains active.";
                    Release();
                    return false;
                }

                _nativeConfigured = true;
                _renderEvent = LCS_GetDirectSurfaceRenderEventFunc();
                if (_renderEvent == IntPtr.Zero)
                {
                    directSurfaceStatus = "The deployed DLL did not return a direct-surface render event.";
                    Release();
                    return false;
                }

                directSurfaceActive = true;
                directSurfaceStatus = "CUDA-D3D11 direct surface configured. Waiting for first render event.";
                return true;
            }
            catch (Exception exception)
            {
                directSurfaceStatus = $"GPU direct surface setup failed: {exception.Message}";
                Release();
                return false;
            }
        }

        public bool RequestDraw()
        {
            if (!directSurfaceActive || !_visible || _material == null || _vertexCount == null || _renderEvent == IntPtr.Zero)
                return false;

            // Keep the CUDA copy, GPU count conversion, and indirect draw in one render-thread
            // command stream. Issuing them separately lets Unity run the compute pass before
            // the native CUDA callback has written the current frame's count.
            _drawCommands.Clear();
            _drawCommands.IssuePluginEvent(_renderEvent, 0);
            _drawCommands.DispatchCompute(_indirectArgsShader, _indirectArgsKernel, 1, 1, 1);
            _drawCommands.DrawProceduralIndirect(Matrix4x4.identity, _material, 0, MeshTopology.Triangles, _indirectArgs);
            ReadStats();
            if (lastNativeError != 0)
            {
                directSurfaceStatus = $"CUDA-D3D11 copy failed ({lastNativeError}); use CPU Mesh fallback.";
                directSurfaceActive = false;
                return false;
            }
            return true;
        }

        public void SetVisible(bool visible) => _visible = visible;

        public void ReadStats()
        {
            if (!_nativeConfigured) return;
            try
            {
                if (LCS_DirectSurfaceGetStats(out DirectSurfaceStatsNative stats) != 0) return;
                nativeRegistered = stats.registered;
                gpuCopyBytes = stats.gpuCopyBytes;
                dispatchCount = stats.dispatchCount;
                lastNativeError = stats.lastError;
                cudaDeviceOrdinal = stats.cudaDeviceOrdinal;
                registrationStage = stats.registrationStage;
                cudaRegistrationError = stats.cudaErrorCode;
                positionBufferByteWidth = stats.positionByteWidth;
                positionBufferUsage = stats.positionUsage;
                positionBufferBindFlags = stats.positionBindFlags;
                positionBufferMiscFlags = stats.positionMiscFlags;
                positionBufferStructureStride = stats.positionStructureByteStride;
                if (lastNativeError == -4101)
                    directSurfaceStatus = $"CUDA-D3D11 registration failed at {RegistrationStageName(registrationStage)} " +
                                          $"(CUDA error {cudaRegistrationError}, device {cudaDeviceOrdinal}).";
            }
            catch (EntryPointNotFoundException)
            {
                lastNativeError = -1;
                directSurfaceStatus = "The deployed DLL lacks the phase-4 direct-surface API.";
                directSurfaceActive = false;
            }
        }

        static string RegistrationStageName(int stage)
        {
            switch (stage)
            {
                case 1: return "D3D11 device lookup";
                case 2: return "CUDA device selection";
                case 3: return "position buffer registration";
                case 4: return "normal buffer registration";
                case 5: return "aux buffer registration";
                case 6: return "vertex-count buffer registration";
                default: return "unknown registration stage";
            }
        }

        public void Release()
        {
            directSurfaceActive = false;
            if (_nativeConfigured)
            {
                try { LCS_DirectSurfaceRelease(); }
                catch (EntryPointNotFoundException) { }
            }
            _nativeConfigured = false;
            _renderEvent = IntPtr.Zero;
            _positions?.Release(); _positions = null;
            _normals?.Release(); _normals = null;
            _aux?.Release(); _aux = null;
            _vertexCount?.Release(); _vertexCount = null;
            _indirectArgs?.Release(); _indirectArgs = null;
            if (_drawCamera != null && _drawCommands != null)
                _drawCamera.RemoveCommandBuffer(CameraEvent.BeforeForwardAlpha, _drawCommands);
            _drawCommands?.Release(); _drawCommands = null;
            _drawCamera = null;
            _indirectArgsShader = null;
            _indirectArgsKernel = -1;
            if (_material != null)
            {
                if (Application.isPlaying) Destroy(_material); else DestroyImmediate(_material);
                _material = null;
            }
        }

        void OnDisable() => Release();
        void OnDestroy() => Release();
    }
}
