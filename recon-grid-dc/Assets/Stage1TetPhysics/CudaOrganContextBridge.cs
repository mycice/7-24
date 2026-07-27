using System;
using System.Collections;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using ReconGridDC.Stage1TetPhysics.Core;
using UnityEngine;

namespace ReconGridDC.Stage1TetPhysics
{
    /// <summary>
    /// CUDA migration phase 1 only: owns a native, static-data mirror of this tetrahedral organ.
    /// The existing Unity XPBD solver remains the sole runtime driver until phase 2 explicitly
    /// replaces it. No per-frame vertex data is uploaded by this component.
    /// </summary>
    [DefaultExecutionOrder(250)]
    [DisallowMultipleComponent]
    [RequireComponent(typeof(Stage1TetSoftBodyController))]
    public sealed class CudaOrganContextBridge : MonoBehaviour
    {
        const string Dll = "LiverCudaSim";

        [StructLayout(LayoutKind.Sequential)]
        struct OrganContextInitDesc
        {
            public int particleCount;
            public int tetCount;
            public int surfaceTriangleCount;
            public int edgeConstraintCount;
            public float density;
            public float youngsModulus;
            public float poissonsRatio;
            public float damping;
        }

        [StructLayout(LayoutKind.Sequential)]
        struct OrganContextStatsNative
        {
            public uint handle;
            public int initialized;
            public int lastError;
            public int particleCount;
            public int tetCount;
            public int surfaceTriangleCount;
            public int edgeConstraintCount;
            public ulong hostFingerprint;
            public ulong deviceBytes;
            public ulong uploadBytes;
            public ulong uploadOperations;
            public float uploadMilliseconds;
            public float validationKernelMilliseconds;
            public ulong restPositionHash;
            public ulong topologyHash;
        }

        [DllImport(Dll)] static extern int LCS_OrganCreate(out uint handle);
        [DllImport(Dll)] static extern int LCS_OrganDestroy(uint handle);
        [DllImport(Dll)] static extern int LCS_OrganInitialize(
            uint handle, ref OrganContextInitDesc desc,
            float[] restPositions3, int[] tetIds4, float[] inverseMass,
            float[] restVolumes, int[] tetActive, int[] surfaceTriangleIds3,
            int[] edgeConstraintIds2, float[] edgeRestLengths);
        [DllImport(Dll)] static extern int LCS_OrganGetStats(uint handle, out OrganContextStatsNative stats);

        [Header("CUDA Migration Phase 1")]
        [Tooltip("Creates an isolated native CUDA organ context and uploads immutable tetrahedral data once. It does not drive XPBD, gripper contact, Tet-to-Grid, cutting, or rendering.")]
        public bool enableCudaOrganContext = true;
        [Tooltip("Logs context creation and validation details once per Play session.")]
        public bool logContextDiagnostics = true;
        [Tooltip("Requests a low-frequency native stats snapshot for Inspector diagnostics. This does not read back organ vertex arrays.")]
        public bool sampleNativeStats = true;
        [Min(0.1f)] public float statsSampleIntervalSeconds = 1f;

        [Header("Diagnostics (runtime)")]
        [SerializeField] uint contextHandle;
        [SerializeField] bool contextReady;
        [SerializeField] int lastNativeError;
        [SerializeField] string status = "Waiting for tetrahedral organ initialization.";
        [SerializeField] int uploadedParticles;
        [SerializeField] int uploadedTetrahedra;
        [SerializeField] int uploadedSurfaceTriangles;
        [SerializeField] int uploadedEdgeConstraints;
        [SerializeField] ulong uploadedDeviceBytes;
        [SerializeField] ulong initializationUploadBytes;
        [SerializeField] ulong initializationUploadOperations;
        [SerializeField] float initializationUploadMilliseconds;
        [SerializeField] float validationKernelMilliseconds;
        [SerializeField] string restPositionHash;
        [SerializeField] string topologyHash;
        [SerializeField] int runtimeStatsSamples;

        Stage1TetSoftBodyController _softBody;
        float _nextStatsSampleTime;
        bool _createAttempted;

        void Awake()
        {
            _softBody = GetComponent<Stage1TetSoftBodyController>();
        }

        void OnEnable()
        {
            if (Application.isPlaying)
                StartCoroutine(CreateAfterTetMeshIsReady());
        }

        IEnumerator CreateAfterTetMeshIsReady()
        {
            while (enabled && enableCudaOrganContext && _softBody != null && !_softBody.IsReady)
                yield return null;

            if (enabled && enableCudaOrganContext && !_createAttempted)
                CreateAndUpload();
        }

        void Update()
        {
            if (!contextReady || !sampleNativeStats || Time.unscaledTime < _nextStatsSampleTime)
                return;

            _nextStatsSampleTime = Time.unscaledTime + Mathf.Max(0.1f, statsSampleIntervalSeconds);
            ReadStats(false);
        }

        void OnDisable() => DestroyContext();
        void OnDestroy() => DestroyContext();

        void CreateAndUpload()
        {
            _createAttempted = true;
            TetMeshData data = _softBody != null ? _softBody.MeshData : null;
            if (data == null)
            {
                status = "TetMeshData is unavailable; native context was not created.";
                return;
            }

            try
            {
                lastNativeError = LCS_OrganCreate(out contextHandle);
            }
            catch (EntryPointNotFoundException)
            {
                status = "The deployed LiverCudaSim.dll predates the phase-1 context API. Rebuild and deploy the new DLL.";
                return;
            }
            catch (DllNotFoundException)
            {
                status = "LiverCudaSim.dll is not available to Unity.";
                return;
            }
            if (lastNativeError != 0)
            {
                status = $"LCS_OrganCreate failed ({lastNativeError}). Check the deployed LiverCudaSim.dll.";
                contextHandle = 0;
                return;
            }

            try
            {
                int[] tetActive = ConvertActiveFlags(data.TetActive, data.NumTets);
                BuildUniqueRestEdges(data, out int[] edgeIds, out float[] edgeLengths);
                OrganContextInitDesc desc = new OrganContextInitDesc
                {
                    particleCount = data.NumParticles,
                    tetCount = data.NumTets,
                    surfaceTriangleCount = data.NumSurfaceTris,
                    edgeConstraintCount = edgeLengths.Length,
                    density = _softBody.density,
                    youngsModulus = _softBody.youngsModulus,
                    poissonsRatio = _softBody.poissonsRatio,
                    damping = _softBody.damping
                };
                lastNativeError = LCS_OrganInitialize(contextHandle, ref desc,
                    Flatten(data.RestPositions, data.NumParticles), data.TetIds, data.InvMass,
                    data.RestVolumes, tetActive, data.SurfaceTriIds, edgeIds, edgeLengths);
                if (lastNativeError != 0)
                {
                    status = $"LCS_OrganInitialize failed ({lastNativeError}). Existing simulation remains unchanged.";
                    DestroyContext();
                    return;
                }

                contextReady = true;
                status = "Static tetrahedral data uploaded. Legacy Unity XPBD remains the active runtime driver.";
                ReadStats(true);
            }
            catch (Exception exception)
            {
                status = $"Context upload preparation failed: {exception.Message}";
                Debug.LogError($"[CudaOrganContext] {status}", this);
                DestroyContext();
            }
        }

        void ReadStats(bool log)
        {
            if (contextHandle == 0)
                return;
            lastNativeError = LCS_OrganGetStats(contextHandle, out OrganContextStatsNative stats);
            if (lastNativeError != 0)
            {
                contextReady = false;
                status = $"LCS_OrganGetStats failed ({lastNativeError}).";
                return;
            }

            uploadedParticles = stats.particleCount;
            uploadedTetrahedra = stats.tetCount;
            uploadedSurfaceTriangles = stats.surfaceTriangleCount;
            uploadedEdgeConstraints = stats.edgeConstraintCount;
            uploadedDeviceBytes = stats.deviceBytes;
            initializationUploadBytes = stats.uploadBytes;
            initializationUploadOperations = stats.uploadOperations;
            initializationUploadMilliseconds = stats.uploadMilliseconds;
            validationKernelMilliseconds = stats.validationKernelMilliseconds;
            restPositionHash = ToHex(stats.restPositionHash);
            topologyHash = ToHex(stats.topologyHash);
            runtimeStatsSamples++;

            if (log && logContextDiagnostics)
            {
                Debug.Log($"[CudaOrganContext] Handle {contextHandle} initialized: " +
                          $"particles={uploadedParticles}, tets={uploadedTetrahedra}, " +
                          $"surfaceTris={uploadedSurfaceTriangles}, edges={uploadedEdgeConstraints}, " +
                          $"upload={initializationUploadBytes} B in {initializationUploadMilliseconds:F3} ms, " +
                          $"device={uploadedDeviceBytes} B, restHash={restPositionHash}, topologyHash={topologyHash}.", this);
            }
        }

        void DestroyContext()
        {
            if (contextHandle == 0)
                return;
            int rc = LCS_OrganDestroy(contextHandle);
            if (rc != 0 && logContextDiagnostics)
                Debug.LogWarning($"[CudaOrganContext] LCS_OrganDestroy({contextHandle}) returned {rc}.", this);
            contextHandle = 0;
            contextReady = false;
        }

        static int[] ConvertActiveFlags(bool[] flags, int count)
        {
            int[] result = new int[count];
            for (int i = 0; i < count; ++i)
                result[i] = flags != null && i < flags.Length && flags[i] ? 1 : 0;
            return result;
        }

        static float[] Flatten(Vector3[] values, int count)
        {
            float[] result = new float[count * 3];
            for (int i = 0; i < count; ++i)
            {
                Vector3 value = values[i];
                result[i * 3] = value.x;
                result[i * 3 + 1] = value.y;
                result[i * 3 + 2] = value.z;
            }
            return result;
        }

        static void BuildUniqueRestEdges(TetMeshData data, out int[] edgeIds, out float[] edgeLengths)
        {
            int[,] pairs = { { 0, 1 }, { 0, 2 }, { 0, 3 }, { 1, 2 }, { 1, 3 }, { 2, 3 } };
            var seen = new HashSet<ulong>();
            var ids = new List<int>();
            var lengths = new List<float>();
            for (int tet = 0; tet < data.NumTets; ++tet)
            {
                int baseIndex = tet * 4;
                for (int pair = 0; pair < 6; ++pair)
                {
                    int a = data.TetIds[baseIndex + pairs[pair, 0]];
                    int b = data.TetIds[baseIndex + pairs[pair, 1]];
                    int lo = Mathf.Min(a, b);
                    int hi = Mathf.Max(a, b);
                    ulong key = ((ulong)(uint)lo << 32) | (uint)hi;
                    if (!seen.Add(key))
                        continue;
                    ids.Add(lo);
                    ids.Add(hi);
                    lengths.Add(Vector3.Distance(data.RestPositions[lo], data.RestPositions[hi]));
                }
            }
            edgeIds = ids.ToArray();
            edgeLengths = lengths.ToArray();
        }

        static string ToHex(ulong value) => $"0x{value:X16}";
    }
}
