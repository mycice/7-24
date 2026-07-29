using System;
using System.Runtime.InteropServices;
using UnityEngine;
using UnityEngine.Rendering;

namespace ReconGridDC.Cuda
{
    public enum CudaPluginInstance
    {
        Primary,
        Secondary
    }

    /// <summary>
    /// D3D11 buffers for the CUDA-resident Dual Contouring renderer. The native plugin writes
    /// separate outer-surface and cut-wall batches, so cut walls cannot corrupt outer smoothing.
    /// </summary>
    [DisallowMultipleComponent]
    public sealed class CudaDirectSurfaceRenderer : MonoBehaviour
    {
        const string Dll = "LiverCudaSim";
        const string Dll2 = "LiverCudaSim2";
        [HideInInspector] public CudaPluginInstance pluginInstance;

        [StructLayout(LayoutKind.Sequential)]
        struct DirectSurfaceStatsNative
        {
            public int supported, configured, registered, capacity;
            public ulong gpuCopyBytes, dispatchCount;
            public int lastError, cudaDeviceOrdinal, registrationStage, cudaErrorCode;
            public int positionByteWidth, positionUsage, positionBindFlags, positionMiscFlags, positionStructureByteStride;
            public int rawVertexCount, sourceVertexCountClamped, rawTriangleCount, validTriangleCount;
            public int rejectedInvalidTriangleCount, rejectedDegenerateTriangleCount;
            public int outerTriangleCount, cutWallTriangleCount, writtenVertexCount, capacityOverflow;
            public int indexedVertexCount, indexedIndexCount, mergeEnabled;
            public int outerIndexedVertexCount, outerIndexedIndexCount, cutWallIndexedVertexCount, cutWallIndexedIndexCount;
            public int topologyDiagnosticReady, topologyDiagnosticOuterVertexSamples;
            public int topologyDiagnosticUniqueOuterVertexEstimate, topologyDiagnosticError;
            public ulong topologyDiagnosticSequence;
            public int rawNoWallTriangleCount, rawAnyWallTriangleCount, rawMixedWallTriangleCount;
        }

        [DllImport(Dll)] static extern int LCS_DirectSurfaceSetBuffers(
            IntPtr outerPositionBuffer, IntPtr outerNormalBuffer, IntPtr outerAuxBuffer, IntPtr outerIndexBuffer, IntPtr outerIndexCountBuffer,
            IntPtr cutPositionBuffer, IntPtr cutNormalBuffer, IntPtr cutAuxBuffer, IntPtr cutIndexBuffer, IntPtr cutIndexCountBuffer, int capacity);
        [DllImport(Dll)] static extern void LCS_DirectSurfaceRelease();
        [DllImport(Dll)] static extern IntPtr LCS_GetDirectSurfaceRenderEventFunc();
        [DllImport(Dll)] static extern int LCS_DirectSurfaceGetStats(out DirectSurfaceStatsNative stats);
        [DllImport(Dll)] static extern int LCS_DirectSurfaceRequestTopologyDiagnostic();
        [DllImport(Dll2, EntryPoint = "LCS_DirectSurfaceSetBuffers")] static extern int LCS2_DirectSurfaceSetBuffers(
            IntPtr outerPositionBuffer, IntPtr outerNormalBuffer, IntPtr outerAuxBuffer, IntPtr outerIndexBuffer, IntPtr outerIndexCountBuffer,
            IntPtr cutPositionBuffer, IntPtr cutNormalBuffer, IntPtr cutAuxBuffer, IntPtr cutIndexBuffer, IntPtr cutIndexCountBuffer, int capacity);
        [DllImport(Dll2, EntryPoint = "LCS_DirectSurfaceRelease")] static extern void LCS2_DirectSurfaceRelease();
        [DllImport(Dll2, EntryPoint = "LCS_GetDirectSurfaceRenderEventFunc")] static extern IntPtr LCS2_GetDirectSurfaceRenderEventFunc();
        [DllImport(Dll2, EntryPoint = "LCS_DirectSurfaceGetStats")] static extern int LCS2_DirectSurfaceGetStats(out DirectSurfaceStatsNative stats);
        [DllImport(Dll2, EntryPoint = "LCS_DirectSurfaceRequestTopologyDiagnostic")] static extern int LCS2_DirectSurfaceRequestTopologyDiagnostic();

        bool SecondaryPlugin => pluginInstance == CudaPluginInstance.Secondary;
        int NativeSetBuffers(IntPtr op,IntPtr on,IntPtr oa,IntPtr oi,IntPtr oc,IntPtr cp,IntPtr cn,IntPtr ca,IntPtr ci,IntPtr cc,int capacity) => SecondaryPlugin ? LCS2_DirectSurfaceSetBuffers(op,on,oa,oi,oc,cp,cn,ca,ci,cc,capacity) : LCS_DirectSurfaceSetBuffers(op,on,oa,oi,oc,cp,cn,ca,ci,cc,capacity);
        void NativeRelease() { if (SecondaryPlugin) LCS2_DirectSurfaceRelease(); else LCS_DirectSurfaceRelease(); }
        IntPtr NativeRenderEvent() => SecondaryPlugin ? LCS2_GetDirectSurfaceRenderEventFunc() : LCS_GetDirectSurfaceRenderEventFunc();
        int NativeGetStats(out DirectSurfaceStatsNative s) { return SecondaryPlugin ? LCS2_DirectSurfaceGetStats(out s) : LCS_DirectSurfaceGetStats(out s); }
        int NativeRequestTopologyDiagnostic() => SecondaryPlugin ? LCS2_DirectSurfaceRequestTopologyDiagnostic() : LCS_DirectSurfaceRequestTopologyDiagnostic();

        [Header("CUDA Migration Phase 4 - GPU Direct Surface")]
        [Tooltip("Draws CUDA Dual Contouring output from D3D11 GPU buffers. Direct3D11 only.")]
        public bool enabledForManager = true;
        [Min(1f)] public float boundsPaddingMultiplier = 3f;

        [Header("Diagnostics (runtime)")]
        [SerializeField] bool directSurfaceActive;
        [SerializeField] string directSurfaceStatus = "Not configured.";
        [SerializeField] int surfaceCapacity;
        [SerializeField] bool d3d11Compatible;
        [SerializeField] int nativeRegistered;
        [SerializeField] ulong gpuCopyBytes, dispatchCount;
        [SerializeField] int lastNativeError, cudaDeviceOrdinal = -1, registrationStage, cudaRegistrationError;
        [SerializeField] int positionBufferByteWidth, positionBufferUsage, positionBufferBindFlags, positionBufferMiscFlags, positionBufferStructureStride;
        [Header("GPU Surface Diagnostics (runtime)")]
        [SerializeField] int rawVertexCount, sourceVertexCountClamped, rawTriangleCount, validTriangleCount;
        [SerializeField] int rejectedInvalidTriangleCount, rejectedDegenerateTriangleCount, outerTriangleCount, cutWallTriangleCount;
        [SerializeField] int outerIndexCount, cutWallIndexCount, writtenVertexCount;
        [SerializeField] bool capacityOverflow;
        [SerializeField] int indexedVertexCount, indexedIndexCount;
        [SerializeField] int outerIndexedVertexCount, outerIndexedIndexCount, cutWallIndexedVertexCount, cutWallIndexedIndexCount;
        [SerializeField] bool gpuVertexMergeActive;
        [Tooltip("Set true once in Play Mode. Only scalar GPU diagnostic counters are read back.")]
        [SerializeField] bool requestTopologyDiagnostic;
        [SerializeField] bool topologyDiagnosticReady;
        [SerializeField] ulong topologyDiagnosticSequence;
        [SerializeField] int topologyDiagnosticOuterVertexSamples, topologyDiagnosticUniqueOuterVertexEstimate, topologyDiagnosticError;
        [Header("Raw Wall-Tag Diagnostics (runtime)")]
        [SerializeField] int rawNoWallTriangleCount, rawAnyWallTriangleCount, rawMixedWallTriangleCount;

        sealed class DrawBatch
        {
            public GraphicsBuffer positions, normals, aux, indices, indexCount, indirectArgs;
            public Material material;
        }

        DrawBatch _outer, _cut;
        ComputeShader _indirectArgsShader;
        int _indirectArgsKernel = -1;
        CommandBuffer _drawCommands;
        Camera _drawCamera;
        Bounds _bounds;
        IntPtr _renderEvent;
        bool _visible = true, _nativeConfigured;

        public bool IsActive => directSurfaceActive;
        public string Status => directSurfaceStatus;
        public ulong GpuCopyBytes => gpuCopyBytes;
        public ulong DispatchCount => dispatchCount;
        public int LastNativeError => lastNativeError;
        public int RawVertexCount => rawVertexCount;
        public int SourceVertexCountClamped => sourceVertexCountClamped;
        public int RawTriangleCount => rawTriangleCount;
        public int ValidTriangleCount => validTriangleCount;
        public int RejectedInvalidTriangleCount => rejectedInvalidTriangleCount;
        public int RejectedDegenerateTriangleCount => rejectedDegenerateTriangleCount;
        public int OuterTriangleCount => outerTriangleCount;
        public int CutWallTriangleCount => cutWallTriangleCount;
        public int OuterIndexCount => outerIndexCount;
        public int CutWallIndexCount => cutWallIndexCount;
        public int WrittenVertexCount => writtenVertexCount;
        public bool CapacityOverflow => capacityOverflow;
        public int IndexedVertexCount => indexedVertexCount;
        public int IndexedIndexCount => indexedIndexCount;
        public bool GpuVertexMergeActive => gpuVertexMergeActive;
        public bool TopologyDiagnosticReady => topologyDiagnosticReady;
        public ulong TopologyDiagnosticSequence => topologyDiagnosticSequence;
        public int TopologyDiagnosticOuterVertexSamples => topologyDiagnosticOuterVertexSamples;
        public int TopologyDiagnosticUniqueOuterVertexEstimate => topologyDiagnosticUniqueOuterVertexEstimate;
        public int TopologyDiagnosticError => topologyDiagnosticError;
        public int RawNoWallTriangleCount => rawNoWallTriangleCount;
        public int RawAnyWallTriangleCount => rawAnyWallTriangleCount;
        public int RawMixedWallTriangleCount => rawMixedWallTriangleCount;

        public bool Configure(int capacity, Bounds bounds, Shader shader, Material sourceMaterial,
            Vector3 projectUvMin, Vector3 projectUvSize)
        {
            Release();
            surfaceCapacity = Mathf.Max(0, capacity);
            _bounds = bounds; _bounds.extents *= Mathf.Max(1f, boundsPaddingMultiplier);
            d3d11Compatible = SystemInfo.graphicsDeviceType == GraphicsDeviceType.Direct3D11;
            if (!enabledForManager) { directSurfaceStatus = "Disabled by Inspector."; return false; }
            if (!d3d11Compatible) { directSurfaceStatus = $"GPU direct surface requires Direct3D11; active API is {SystemInfo.graphicsDeviceType}."; return false; }
            if (surfaceCapacity <= 0 || shader == null) { directSurfaceStatus = "Invalid surface capacity or direct-surface shader."; return false; }
            try
            {
                _outer = CreateBatch(shader, sourceMaterial, projectUvMin, projectUvSize, "Outer");
                _cut = CreateBatch(shader, sourceMaterial, projectUvMin, projectUvSize, "CutWall");
                _indirectArgsShader = Resources.Load<ComputeShader>("CudaDirectSurfaceIndirectArgs");
                if (_indirectArgsShader == null) { directSurfaceStatus = "CudaDirectSurfaceIndirectArgs.compute is missing from Assets/Resources."; Release(); return false; }
                _indirectArgsKernel = _indirectArgsShader.FindKernel("BuildIndirectArgs");
                BindIndirectArguments(_outer); BindIndirectArguments(_cut);
                _drawCamera = Camera.main;
                if (_drawCamera == null) { directSurfaceStatus = "GPU direct surface requires a tagged Main Camera."; Release(); return false; }
                _drawCommands = new CommandBuffer { name = "CUDA Direct Surface Batches" };
                _drawCamera.AddCommandBuffer(CameraEvent.BeforeForwardAlpha, _drawCommands);
                int rc = NativeSetBuffers(
                    Native(_outer.positions), Native(_outer.normals), Native(_outer.aux), Native(_outer.indices), Native(_outer.indexCount),
                    Native(_cut.positions), Native(_cut.normals), Native(_cut.aux), Native(_cut.indices), Native(_cut.indexCount), surfaceCapacity);
                if (rc != 0) { lastNativeError = rc; directSurfaceStatus = $"LCS_DirectSurfaceSetBuffers failed ({rc})."; Release(); return false; }
                _nativeConfigured = true; _renderEvent = NativeRenderEvent();
                if (_renderEvent == IntPtr.Zero) { directSurfaceStatus = "The deployed DLL did not return a direct-surface render event."; Release(); return false; }
                directSurfaceActive = true;
                directSurfaceStatus = "CUDA-D3D11 direct surface configured with isolated outer and cut-wall batches.";
                return true;
            }
            catch (Exception e) { directSurfaceStatus = $"GPU direct surface setup failed: {e.Message}"; Release(); return false; }
        }

        DrawBatch CreateBatch(Shader shader, Material sourceMaterial, Vector3 projectUvMin,
            Vector3 projectUvSize, string label)
        {
            var batch = new DrawBatch
            {
                positions = new GraphicsBuffer(GraphicsBuffer.Target.Raw, surfaceCapacity * 3, sizeof(float)),
                normals = new GraphicsBuffer(GraphicsBuffer.Target.Raw, surfaceCapacity * 3, sizeof(float)),
                aux = new GraphicsBuffer(GraphicsBuffer.Target.Raw, surfaceCapacity * 4, sizeof(float)),
                indices = new GraphicsBuffer(GraphicsBuffer.Target.Raw, surfaceCapacity, sizeof(uint)),
                indexCount = new GraphicsBuffer(GraphicsBuffer.Target.Raw, 1, sizeof(uint)),
                indirectArgs = new GraphicsBuffer(GraphicsBuffer.Target.IndirectArguments, 4, sizeof(uint)),
                material = new Material(shader) { name = $"Runtime_CudaDirectSurface_{label}" }
            };
            if (sourceMaterial != null)
                batch.material.CopyPropertiesFromMaterial(sourceMaterial);
            batch.material.SetVector("_ProjectUvMin", projectUvMin);
            batch.material.SetVector("_ProjectUvSize", projectUvSize);
            batch.material.SetBuffer("_CudaSurfacePositions", batch.positions);
            batch.material.SetBuffer("_CudaSurfaceNormals", batch.normals);
            batch.material.SetBuffer("_CudaSurfaceAux", batch.aux);
            batch.material.SetBuffer("_CudaSurfaceIndices", batch.indices);
            batch.material.SetBuffer("_CudaSurfaceVertexCount", batch.indexCount);
            return batch;
        }

        void BindIndirectArguments(DrawBatch batch)
        {
            _indirectArgsShader.SetBuffer(_indirectArgsKernel, "_CudaSurfaceVertexCount", batch.indexCount);
            _indirectArgsShader.SetBuffer(_indirectArgsKernel, "_IndirectArgs", batch.indirectArgs);
        }

        static IntPtr Native(GraphicsBuffer buffer) => buffer.GetNativeBufferPtr();

        public bool RequestDraw() => RequestDraw(true);

        public bool RequestDraw(bool refreshNativeSurface)
        {
            if (!directSurfaceActive || !_visible || _outer == null || _cut == null || _renderEvent == IntPtr.Zero) return false;
            _drawCommands.Clear();
            if (refreshNativeSurface)
                _drawCommands.IssuePluginEvent(_renderEvent, 0);
            QueueDrawBatch(_outer); QueueDrawBatch(_cut);
            if (refreshNativeSurface)
                ReadStats();
            if (lastNativeError == 0) return true;
            directSurfaceStatus = $"CUDA-D3D11 copy failed ({lastNativeError}).";
            directSurfaceActive = false; return false;
        }

        void QueueDrawBatch(DrawBatch batch)
        {
            _drawCommands.SetComputeBufferParam(_indirectArgsShader, _indirectArgsKernel, "_CudaSurfaceVertexCount", batch.indexCount);
            _drawCommands.SetComputeBufferParam(_indirectArgsShader, _indirectArgsKernel, "_IndirectArgs", batch.indirectArgs);
            _drawCommands.DispatchCompute(_indirectArgsShader, _indirectArgsKernel, 1, 1, 1);
            _drawCommands.DrawProceduralIndirect(Matrix4x4.identity, batch.material, 0, MeshTopology.Triangles, batch.indirectArgs);
        }

        public void SetVisible(bool visible) => _visible = visible;

        public void ReadStats()
        {
            if (!_nativeConfigured) return;
            try
            {
                if (NativeGetStats(out var s) != 0) return;
                nativeRegistered=s.registered; gpuCopyBytes=s.gpuCopyBytes; dispatchCount=s.dispatchCount; lastNativeError=s.lastError;
                cudaDeviceOrdinal=s.cudaDeviceOrdinal; registrationStage=s.registrationStage; cudaRegistrationError=s.cudaErrorCode;
                positionBufferByteWidth=s.positionByteWidth; positionBufferUsage=s.positionUsage; positionBufferBindFlags=s.positionBindFlags; positionBufferMiscFlags=s.positionMiscFlags; positionBufferStructureStride=s.positionStructureByteStride;
                rawVertexCount=s.rawVertexCount; sourceVertexCountClamped=s.sourceVertexCountClamped; rawTriangleCount=s.rawTriangleCount; validTriangleCount=s.validTriangleCount;
                rejectedInvalidTriangleCount=s.rejectedInvalidTriangleCount; rejectedDegenerateTriangleCount=s.rejectedDegenerateTriangleCount; outerTriangleCount=s.outerTriangleCount; cutWallTriangleCount=s.cutWallTriangleCount;
                outerIndexCount=s.outerIndexedIndexCount; cutWallIndexCount=s.cutWallIndexedIndexCount; writtenVertexCount=s.writtenVertexCount; capacityOverflow=s.capacityOverflow!=0;
                indexedVertexCount=s.indexedVertexCount; indexedIndexCount=s.indexedIndexCount; outerIndexedVertexCount=s.outerIndexedVertexCount; outerIndexedIndexCount=s.outerIndexedIndexCount;
                cutWallIndexedVertexCount=s.cutWallIndexedVertexCount; cutWallIndexedIndexCount=s.cutWallIndexedIndexCount; gpuVertexMergeActive=s.mergeEnabled!=0;
                topologyDiagnosticReady=s.topologyDiagnosticReady!=0; topologyDiagnosticSequence=s.topologyDiagnosticSequence; topologyDiagnosticOuterVertexSamples=s.topologyDiagnosticOuterVertexSamples;
                topologyDiagnosticUniqueOuterVertexEstimate=s.topologyDiagnosticUniqueOuterVertexEstimate; topologyDiagnosticError=s.topologyDiagnosticError;
                rawNoWallTriangleCount=s.rawNoWallTriangleCount; rawAnyWallTriangleCount=s.rawAnyWallTriangleCount; rawMixedWallTriangleCount=s.rawMixedWallTriangleCount;
                if (lastNativeError == -4101) directSurfaceStatus = $"CUDA-D3D11 registration failed at {RegistrationStageName(registrationStage)} (CUDA error {cudaRegistrationError}, device {cudaDeviceOrdinal}).";
            }
            catch (EntryPointNotFoundException) { lastNativeError=-1; directSurfaceStatus="The deployed DLL lacks the phase-4 dual-batch API."; directSurfaceActive=false; }
        }

        void Update()
        {
            if (!requestTopologyDiagnostic) return;
            requestTopologyDiagnostic=false;
            if (!_nativeConfigured || !directSurfaceActive) { directSurfaceStatus="GPU topology diagnostic was requested before CUDA direct surface became active."; return; }
            try
            {
                int rc=NativeRequestTopologyDiagnostic();
                if(rc!=0) { topologyDiagnosticError=rc; directSurfaceStatus=$"LCS_DirectSurfaceRequestTopologyDiagnostic failed ({rc})."; }
                else { topologyDiagnosticReady=false; directSurfaceStatus="GPU topology diagnostic requested; scalar results arrive after the next CUDA render event."; }
            }
            catch (EntryPointNotFoundException) { topologyDiagnosticError=-1; directSurfaceStatus="The deployed DLL lacks the GPU surface diagnostics API. Rebuild and deploy LiverCudaSim.dll manually."; }
        }

        static string RegistrationStageName(int stage)
        {
            switch(stage)
            {
                case 1:return "D3D11 device lookup"; case 2:return "CUDA device selection"; case 3:return "outer position registration";
                case 4:return "outer normal registration"; case 5:return "outer aux registration"; case 6:return "outer index registration";
                case 7:return "outer index-count registration"; case 8:return "cut position registration"; case 9:return "cut normal registration";
                case 10:return "cut aux registration"; case 11:return "cut index registration"; case 12:return "cut index-count registration"; default:return "unknown registration stage";
            }
        }

        public void Release()
        {
            directSurfaceActive=false;
            if(_nativeConfigured) try { NativeRelease(); } catch (EntryPointNotFoundException) { }
            _nativeConfigured=false; _renderEvent=IntPtr.Zero;
            ReleaseBatch(_outer); _outer=null; ReleaseBatch(_cut); _cut=null;
            if(_drawCamera!=null && _drawCommands!=null) _drawCamera.RemoveCommandBuffer(CameraEvent.BeforeForwardAlpha,_drawCommands);
            _drawCommands?.Release(); _drawCommands=null; _drawCamera=null; _indirectArgsShader=null; _indirectArgsKernel=-1;
        }

        static void ReleaseBatch(DrawBatch batch)
        {
            if(batch==null) return;
            batch.positions?.Release(); batch.normals?.Release(); batch.aux?.Release(); batch.indices?.Release(); batch.indexCount?.Release(); batch.indirectArgs?.Release();
            if(batch.material!=null) { if(Application.isPlaying) UnityEngine.Object.Destroy(batch.material); else UnityEngine.Object.DestroyImmediate(batch.material); }
        }

        void OnDisable() => Release();
        void OnDestroy() => Release();
    }
}
