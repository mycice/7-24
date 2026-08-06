using System;
using System.Collections;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using ReconGridDC.Stage1TetPhysics.Core;
using ReconGridDC.Stage2Grasping.Grasping;
using ReconGridDC.Stage2Grasping.Modules;
using ReconGridDC.Cuda;
using UnityEngine;

namespace ReconGridDC.Stage1TetPhysics
{
    [StructLayout(LayoutKind.Sequential)]
    public struct CudaToolCapsule
    {
        public float ax, ay, az, radius;
        public float bx, by, bz, friction;
        public float prevAx, prevAy, prevAz, unused0;
        public float prevBx, prevBy, prevBz, unused1;
    }

    /// <summary>
    /// CUDA migration phase 1 only: owns a native, static-data mirror of this tetrahedral organ.
    /// The existing Unity XPBD solver remains the sole runtime driver until phase 2 explicitly
    /// replaces it. No per-frame vertex data is uploaded by this component.
    /// </summary>
    [DefaultExecutionOrder(250)]
    [DisallowMultipleComponent]
    [RequireComponent(typeof(Stage1TetSoftBodyController))]
    public sealed class CudaOrganContextBridge : MonoBehaviour, ICudaOrganActivitySource
    {
        const string Dll = "LiverCudaSim";
        const string Dll2 = "LiverCudaSim2";

        [Header("Multi-organ instance")]
        [Tooltip("Secondary routes this organ context and its Tet-to-Grid writes to LiverCudaSim2.dll.")]
        public CudaPluginInstance pluginInstance;

        public enum CudaXpbdMode
        {
            Disabled,
            ReadOnlyComparison,
            CudaDriverIsolated,
            CudaDriverWithTemporaryPublish
        }

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
            public ulong dynamicDeviceBytes;
            public ulong xpbdStepCount;
            public float lastXpbdMilliseconds;
            public float totalXpbdMilliseconds;
            public int lastNanCount;
        }

        [StructLayout(LayoutKind.Sequential)]
        struct OrganContextXpbdParams
        {
            public int numSubSteps;
            public int constraintIterations;
            public float edgeCompliance;
            public float youngsModulus;
            public float poissonsRatio;
            public float damping;
            public float gravityX;
            public float gravityY;
            public float gravityZ;
            public float groundY;
        }

        [StructLayout(LayoutKind.Sequential)]
        struct OrganContextComparisonStatsNative
        {
            public int particleCount;
            public int cudaNanCount;
            public float maxError;
            public float rmsError;
            public float centroidError;
            public float bboxMinError;
            public float bboxMaxError;
            public float bboxExtentError;
        }

        [StructLayout(LayoutKind.Sequential)]
        struct OrganContextToolContactParamsNative
        {
            public int capsuleCount, contactEnabled, keepContactActiveWhenIdle, useCandidateCulling;
            public int contactIterations, couplingPasses, graspRequest, releaseRequest;
            public float contactDistance, contactCompliance, tangentialFriction, tangentialDamping;
            public float candidatePadding, graspHeight, graspCoreRadius, graspFormDuration;
            public float graspBoundsMinX, graspBoundsMinY, graspBoundsMinZ;
            public float graspBoundsMaxX, graspBoundsMaxY, graspBoundsMaxZ;
            public float frameCenterX, frameCenterY, frameCenterZ;
            public float axisUX, axisUY, axisUZ;
            public float axisVX, axisVY, axisVZ;
            public float axisWX, axisWY, axisWZ;
            public float graspGaussianWidth, graspInfluenceRadius, graspSoftFollowRate;
        }

        [StructLayout(LayoutKind.Sequential)]
        struct OrganContextToolContactStatsNative
        {
            public int activeCandidates, contactCount;
            public float maxContactDepth;
            public int surfaceCandidateTriangles, surfaceContactTriangles;
            public float surfaceMaxContactDepth;
            public int graspedParticleCount;
            public ulong dispatchCount, toolUploadBytes, toolUploadOperations;
            public float lastToolMilliseconds, totalToolMilliseconds;
            public int lastToolError;
        }

        [StructLayout(LayoutKind.Sequential)]
        struct OrganContextTetToGridDescNative
        {
            public int cornerCount;
            public float restCentroidX, restCentroidY, restCentroidZ;
        }

        [StructLayout(LayoutKind.Sequential)]
        struct OrganContextTetToGridStatsNative
        {
            public int configured, cornerCount, mappedCorners, fallbackCorners;
            public ulong uploadBytes, updateCount;
            public float lastUpdateMilliseconds, totalUpdateMilliseconds;
            public int lastError;
        }

        [DllImport(Dll)] static extern int LCS_OrganCreate(out uint handle);
        [DllImport(Dll)] static extern int LCS_OrganDestroy(uint handle);
        [DllImport(Dll)] static extern int LCS_OrganInitialize(
            uint handle, ref OrganContextInitDesc desc,
            float[] restPositions3, int[] tetIds4, float[] inverseMass,
            float[] restVolumes, int[] tetActive, int[] surfaceTriangleIds3,
            int[] edgeConstraintIds2, float[] edgeRestLengths);
        [DllImport(Dll)] static extern int LCS_OrganGetStats(uint handle, out OrganContextStatsNative stats);
        [DllImport(Dll)] static extern int LCS_OrganXpbdInitialize(
            uint handle, float[] initialPositions3,
            int edgeColorCount, int[] edgeColorOffsets, int[] edgeColorCounts, int[] edgeColorFlat,
            int tetColorCount, int[] tetColorOffsets, int[] tetColorCounts, int[] tetColorFlat,
            int surfaceColorCount, int[] surfaceColorOffsets, int[] surfaceColorCounts, int[] surfaceColorFlat);
        [DllImport(Dll)] static extern int LCS_OrganXpbdStep(uint handle, float dt, ref OrganContextXpbdParams parameters);
        [DllImport(Dll)] static extern int LCS_OrganXpbdComparePositions(
            uint handle, float[] unityPositions3, int particleCount,
            out OrganContextComparisonStatsNative comparison);
        [DllImport(Dll)] static extern int LCS_OrganXpbdGetPositions(uint handle, float[] positions3, int particleCount);
        [DllImport(Dll)] static extern int LCS_OrganToolStep(uint handle, float dt, CudaToolCapsule[] capsules,
            ref OrganContextToolContactParamsNative parameters);
        [DllImport(Dll)] static extern int LCS_OrganToolGetStats(uint handle, out OrganContextToolContactStatsNative stats);
        [DllImport(Dll)] static extern int LCS_OrganTetToGridConfigure(uint handle,
            ref OrganContextTetToGridDescNative desc, int[] hostTetByCorner, float[] barycentricWeights4,
            float[] alignedRestCorners3, byte[] activeMask);
        [DllImport(Dll)] static extern int LCS_OrganTetToGridUpdate(uint handle, int applyLocalDeformation);
        [DllImport(Dll)] static extern int LCS_OrganTetToGridGetStats(uint handle, out OrganContextTetToGridStatsNative stats);

        [DllImport(Dll2, EntryPoint="LCS_OrganCreate")] static extern int LCS2_OrganCreate(out uint handle);
        [DllImport(Dll2, EntryPoint="LCS_OrganDestroy")] static extern int LCS2_OrganDestroy(uint handle);
        [DllImport(Dll2, EntryPoint="LCS_OrganInitialize")] static extern int LCS2_OrganInitialize(uint handle, ref OrganContextInitDesc desc, float[] restPositions3, int[] tetIds4, float[] inverseMass, float[] restVolumes, int[] tetActive, int[] surfaceTriangleIds3, int[] edgeConstraintIds2, float[] edgeRestLengths);
        [DllImport(Dll2, EntryPoint="LCS_OrganGetStats")] static extern int LCS2_OrganGetStats(uint handle, out OrganContextStatsNative stats);
        [DllImport(Dll2, EntryPoint="LCS_OrganXpbdInitialize")] static extern int LCS2_OrganXpbdInitialize(uint handle, float[] initialPositions3, int edgeColorCount, int[] edgeColorOffsets, int[] edgeColorCounts, int[] edgeColorFlat, int tetColorCount, int[] tetColorOffsets, int[] tetColorCounts, int[] tetColorFlat, int surfaceColorCount, int[] surfaceColorOffsets, int[] surfaceColorCounts, int[] surfaceColorFlat);
        [DllImport(Dll2, EntryPoint="LCS_OrganXpbdStep")] static extern int LCS2_OrganXpbdStep(uint handle, float dt, ref OrganContextXpbdParams parameters);
        [DllImport(Dll2, EntryPoint="LCS_OrganXpbdComparePositions")] static extern int LCS2_OrganXpbdComparePositions(uint handle, float[] positions, int count, out OrganContextComparisonStatsNative stats);
        [DllImport(Dll2, EntryPoint="LCS_OrganXpbdGetPositions")] static extern int LCS2_OrganXpbdGetPositions(uint handle, float[] positions, int count);
        [DllImport(Dll2, EntryPoint="LCS_OrganToolStep")] static extern int LCS2_OrganToolStep(uint handle, float dt, CudaToolCapsule[] capsules, ref OrganContextToolContactParamsNative parameters);
        [DllImport(Dll2, EntryPoint="LCS_OrganToolGetStats")] static extern int LCS2_OrganToolGetStats(uint handle, out OrganContextToolContactStatsNative stats);
        [DllImport(Dll2, EntryPoint="LCS_OrganTetToGridConfigure")] static extern int LCS2_OrganTetToGridConfigure(uint handle, ref OrganContextTetToGridDescNative desc, int[] hostTetByCorner, float[] weights, float[] corners, byte[] mask);
        [DllImport(Dll2, EntryPoint="LCS_OrganTetToGridUpdate")] static extern int LCS2_OrganTetToGridUpdate(uint handle, int applyLocalDeformation);
        [DllImport(Dll2, EntryPoint="LCS_OrganTetToGridGetStats")] static extern int LCS2_OrganTetToGridGetStats(uint handle, out OrganContextTetToGridStatsNative stats);

        bool SecondaryPlugin => pluginInstance == CudaPluginInstance.Secondary;
        int NativeOrganCreate(out uint h) { return SecondaryPlugin ? LCS2_OrganCreate(out h) : LCS_OrganCreate(out h); }
        int NativeOrganDestroy(uint h) => SecondaryPlugin ? LCS2_OrganDestroy(h) : LCS_OrganDestroy(h);
        int NativeOrganInitialize(uint h,ref OrganContextInitDesc d,float[] r,int[] t,float[] m,float[] v,int[] a,int[] s,int[] e,float[] l) => SecondaryPlugin ? LCS2_OrganInitialize(h,ref d,r,t,m,v,a,s,e,l) : LCS_OrganInitialize(h,ref d,r,t,m,v,a,s,e,l);
        int NativeOrganGetStats(uint h,out OrganContextStatsNative s) { return SecondaryPlugin ? LCS2_OrganGetStats(h,out s) : LCS_OrganGetStats(h,out s); }
        int NativeXpbdInitialize(uint h,float[] p,int ec,int[] eo,int[] en,int[] ef,int tc,int[] to,int[] tn,int[] tf,int sc,int[] so,int[] sn,int[] sf) => SecondaryPlugin ? LCS2_OrganXpbdInitialize(h,p,ec,eo,en,ef,tc,to,tn,tf,sc,so,sn,sf) : LCS_OrganXpbdInitialize(h,p,ec,eo,en,ef,tc,to,tn,tf,sc,so,sn,sf);
        int NativeXpbdStep(uint h,float dt,ref OrganContextXpbdParams p) => SecondaryPlugin ? LCS2_OrganXpbdStep(h,dt,ref p) : LCS_OrganXpbdStep(h,dt,ref p);
        int NativeXpbdCompare(uint h,float[] p,int c,out OrganContextComparisonStatsNative s) { return SecondaryPlugin ? LCS2_OrganXpbdComparePositions(h,p,c,out s) : LCS_OrganXpbdComparePositions(h,p,c,out s); }
        int NativeXpbdGetPositions(uint h,float[] p,int c) => SecondaryPlugin ? LCS2_OrganXpbdGetPositions(h,p,c) : LCS_OrganXpbdGetPositions(h,p,c);
        int NativeToolStep(uint h,float dt,CudaToolCapsule[] c,ref OrganContextToolContactParamsNative p) => SecondaryPlugin ? LCS2_OrganToolStep(h,dt,c,ref p) : LCS_OrganToolStep(h,dt,c,ref p);
        int NativeToolGetStats(uint h,out OrganContextToolContactStatsNative s) { return SecondaryPlugin ? LCS2_OrganToolGetStats(h,out s) : LCS_OrganToolGetStats(h,out s); }
        int NativeTetToGridConfigure(uint h,ref OrganContextTetToGridDescNative d,int[] t,float[] w,float[] c,byte[] m) => SecondaryPlugin ? LCS2_OrganTetToGridConfigure(h,ref d,t,w,c,m) : LCS_OrganTetToGridConfigure(h,ref d,t,w,c,m);
        int NativeTetToGridUpdate(uint h,int a) => SecondaryPlugin ? LCS2_OrganTetToGridUpdate(h,a) : LCS_OrganTetToGridUpdate(h,a);
        int NativeTetToGridGetStats(uint h,out OrganContextTetToGridStatsNative s) { return SecondaryPlugin ? LCS2_OrganTetToGridGetStats(h,out s) : LCS_OrganTetToGridGetStats(h,out s); }

        [Header("CUDA Migration Phase 1")]
        [Tooltip("Creates an isolated native CUDA organ context and uploads immutable tetrahedral data once. It does not drive XPBD, gripper contact, Tet-to-Grid, cutting, or rendering.")]
        public bool enableCudaOrganContext = true;
        [Tooltip("Logs context creation and validation details once per Play session.")]
        public bool logContextDiagnostics = true;
        [Tooltip("Requests a low-frequency native stats snapshot for Inspector diagnostics. This does not read back organ vertex arrays.")]
        public bool sampleNativeStats = true;
        [Min(0.1f)] public float statsSampleIntervalSeconds = 1f;

        [Header("CUDA Migration Phase 2 - XPBD")]
        [Tooltip("Enables the independent CUDA tetrahedral XPBD state inside this organ context. Disabled preserves the exact phase-1 behavior.")]
        public bool enableCudaXpbd;
        [Tooltip("Read Only Comparison: Unity XPBD remains the only writer. CUDA Driver Isolated: CUDA does not publish positions. CUDA Driver With Temporary Publish is the phase-3 visual verification path and reads positions back once per physics step for legacy rendering and Stage 3.")]
        public CudaXpbdMode cudaXpbdMode = CudaXpbdMode.Disabled;
        [Min(1)] [Tooltip("Constraint passes per CUDA XPBD substep. Keep at 1 to match the current Unity solver.")]
        public int cudaConstraintIterations = 1;
        [Min(0.05f)] [Tooltip("Minimum time between explicit CUDA-versus-Unity position comparisons. Comparison mode alone performs a complete CUDA position readback.")]
        public float comparisonIntervalSeconds = 1f;

        [Header("CUDA Migration Phase 3 - Tool Contact")]
        [Tooltip("Disabled preserves the existing CPU grasp and Unity Compute contact route. CUDA Driver modes with this enabled use CUDA-only contact and grasp constraints.")]
        public bool enableCudaToolContact;
        [Tooltip("CUDA contact requires a CUDA driver mode. The temporary publish mode is intended only for visual phase-3 verification and is not the final zero-readback path.")]
        public bool temporaryPublishCudaPositions = true;

        [Header("CUDA Migration Phase 4 - Tet To Grid")]
        [Tooltip("Uploads the already-built Stage 3 host-tet/barycentric embedding once and evaluates it on CUDA every fixed step. This removes the full tet-position readback and full grid-position upload from the CUDA driver path.")]
        public bool enableCudaTetToGrid = true;

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
        [SerializeField] bool cudaXpbdReady;
        [SerializeField] string cudaXpbdStatus = "CUDA XPBD is disabled.";
        [SerializeField] int cudaEdgeColorCount;
        [SerializeField] int cudaTetColorCount;
        [SerializeField] ulong cudaDynamicDeviceBytes;
        [SerializeField] ulong cudaXpbdStepCount;
        [SerializeField] float cudaLastXpbdMilliseconds;
        [SerializeField] float cudaTotalXpbdMilliseconds;
        [SerializeField] int cudaNanCount;
        [SerializeField] int comparisonSamples;
        [SerializeField] int comparisonParticleCount;
        [SerializeField] float positionMaxError;
        [SerializeField] float positionRmsError;
        [SerializeField] float centroidError;
        [SerializeField] float bboxMinError;
        [SerializeField] float bboxMaxError;
        [SerializeField] float bboxExtentError;
        [SerializeField] bool cudaToolContactActive;
        [SerializeField] string cudaToolContactStatus = "CUDA tool contact is disabled.";
        [SerializeField] int cudaToolActiveCandidates;
        [SerializeField] int cudaToolContactCount;
        [SerializeField] float cudaToolMaxContactDepth;
        [SerializeField] int cudaToolSurfaceCandidateTriangles;
        [SerializeField] int cudaToolSurfaceContactTriangles;
        [SerializeField] float cudaToolSurfaceMaxContactDepth;
        [SerializeField] int cudaToolGraspedParticleCount;
        [SerializeField] ulong cudaToolDispatchCount;
        [SerializeField] ulong cudaToolUploadBytes;
        [SerializeField] ulong cudaToolUploadOperations;
        [SerializeField] float cudaToolLastMilliseconds;
        [SerializeField] float cudaToolTotalMilliseconds;
        [SerializeField] int cudaToolLastError;
        [SerializeField] bool cudaToolBroadphaseOverlapping;
        [SerializeField] bool cudaToolBroadphaseDisabled;
        [SerializeField] bool cudaToolGraspLocked;
        [SerializeField] bool enableOrganToolBroadphase;
        [SerializeField] bool cudaTetToGridReady;
        [SerializeField] string cudaTetToGridStatus = "CUDA Tet-to-Grid is disabled.";
        [SerializeField] int cudaTetToGridMappedCorners;
        [SerializeField] int cudaTetToGridFallbackCorners;
        [SerializeField] ulong cudaTetToGridUploadBytes;
        [SerializeField] ulong cudaTetToGridUpdateCount;
        [SerializeField] float cudaTetToGridLastMilliseconds;
        [SerializeField] float cudaTetToGridTotalMilliseconds;
        [SerializeField] int cudaTetToGridLastError;
        [SerializeField] bool gpuResidentTetToGridFrameActive;
        [Header("Adaptive CUDA XPBD (runtime)")]
        [SerializeField] int cudaXpbdEffectiveStepInterval = 1;
        [SerializeField] int cudaXpbdSkippedFixedSteps;
        [SerializeField] float cudaXpbdAccumulatedDeltaTime;

        Stage1TetSoftBodyController _softBody;
        float _nextStatsSampleTime;
        float _nextToolStatsSampleTime;
        float _nextComparisonTime;
        bool _createAttempted;
        readonly CudaToolCapsule[] _toolCapsules = new CudaToolCapsule[12];
        float[] _temporaryPositionReadback;
        Bounds _organRestBounds;
        bool _organRestBoundsReady;
        bool _cudaGraspMayBeActive;
        LiverCudaManager _gpuWorkScheduler;
        int _idleCudaXpbdStepInterval = 3;
        int _idleCudaXpbdFixedStepCounter;

        public bool IsCudaDriverActive =>
            contextReady && cudaXpbdReady && enableCudaXpbd &&
            (cudaXpbdMode == CudaXpbdMode.CudaDriverIsolated ||
             cudaXpbdMode == CudaXpbdMode.CudaDriverWithTemporaryPublish);

        public bool IsCudaToolContactDriverActive =>
            IsCudaDriverActive && enableCudaToolContact;

        public bool CudaToolContactActive => cudaToolContactActive;
        public bool CudaToolBroadphaseOverlapping => cudaToolBroadphaseOverlapping;
        public bool CudaToolGraspLocked => cudaToolGraspLocked;
        public bool CanUseCudaTetToGrid => contextReady && cudaXpbdReady && enableCudaTetToGrid && IsCudaDriverActive;
        // Set by the Stage 3 bridge only after its CUDA Tet-to-Grid mapping is configured for
        // the detailed CUDA organ. This keeps the compatibility readback path available for
        // every other presentation and fallback mode.
        public bool ShouldSkipCpuPositionPublish =>
            IsCudaDriverActive && cudaTetToGridReady && gpuResidentTetToGridFrameActive;

        public void ConfigureAdaptiveCudaXpbdScheduling(LiverCudaManager scheduler, int idleStepInterval)
        {
            _gpuWorkScheduler = scheduler;
            _idleCudaXpbdStepInterval = Mathf.Clamp(idleStepInterval, 2, 12);
            _idleCudaXpbdFixedStepCounter = 0;
            cudaXpbdAccumulatedDeltaTime = 0f;
        }

        public void ConfigureOrganToolBroadphase(bool enabled)
        {
            enableOrganToolBroadphase = enabled;
            if (!enabled)
            {
                cudaToolBroadphaseOverlapping = true;
                cudaToolBroadphaseDisabled = false;
            }
        }

        public bool TryGetScheduledCudaXpbdDeltaTime(float fixedDeltaTime, out float scheduledDeltaTime)
        {
            scheduledDeltaTime = fixedDeltaTime;
            if (_gpuWorkScheduler == null)
            {
                cudaXpbdEffectiveStepInterval = 1;
                return true;
            }

            cudaXpbdAccumulatedDeltaTime += fixedDeltaTime;
            if (_gpuWorkScheduler.IsAdaptiveGpuWorkActiveThisFrame())
            {
                cudaXpbdEffectiveStepInterval = 1;
                _idleCudaXpbdFixedStepCounter = 0;
                scheduledDeltaTime = cudaXpbdAccumulatedDeltaTime;
                cudaXpbdAccumulatedDeltaTime = 0f;
                return true;
            }

            cudaXpbdEffectiveStepInterval = _idleCudaXpbdStepInterval;
            _idleCudaXpbdFixedStepCounter++;
            if (_idleCudaXpbdFixedStepCounter < _idleCudaXpbdStepInterval)
            {
                cudaXpbdSkippedFixedSteps++;
                return false;
            }

            _idleCudaXpbdFixedStepCounter = 0;
            scheduledDeltaTime = cudaXpbdAccumulatedDeltaTime;
            cudaXpbdAccumulatedDeltaTime = 0f;
            return true;
        }

        // In phase 3, CUDA may become the writer only after SoftBodyCollisionModule has
        // successfully submitted the gripper packet for this fixed step. This prevents a
        // failed packet from silently replacing the established Unity collision route.
        public bool ShouldUseCudaDriverThisFixedStep =>
            IsCudaDriverActive && (!IsCudaToolContactDriverActive || cudaToolContactActive);

        public bool RequiresTemporaryCudaPositionPublish =>
            IsCudaToolContactDriverActive && temporaryPublishCudaPositions &&
            cudaXpbdMode == CudaXpbdMode.CudaDriverWithTemporaryPublish;

        // A comparison is meaningful only when Unity positions belong to the same fixed
        // step as the CUDA state. This is deliberately limited to the diagnostic mode.
        public bool RequiresSynchronousUnityReadback =>
            contextReady && cudaXpbdReady && enableCudaXpbd &&
            cudaXpbdMode == CudaXpbdMode.ReadOnlyComparison;

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
            if (cudaTetToGridReady)
                ReadTetToGridStats();
            if (IsCudaToolContactDriverActive && Time.unscaledTime >= _nextToolStatsSampleTime)
            {
                _nextToolStatsSampleTime = Time.unscaledTime + Mathf.Max(0.1f, statsSampleIntervalSeconds);
                ReadToolStats();
            }
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
                lastNativeError = NativeOrganCreate(out contextHandle);
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
                lastNativeError = NativeOrganInitialize(contextHandle, ref desc,
                    Flatten(data.RestPositions, data.NumParticles), data.TetIds,
                    BuildUnityEquivalentInverseMass(data, _softBody.density),
                    data.RestVolumes, tetActive, data.SurfaceTriIds, edgeIds, edgeLengths);
                if (lastNativeError != 0)
                {
                    status = $"LCS_OrganInitialize failed ({lastNativeError}). Existing simulation remains unchanged.";
                    DestroyContext();
                    return;
                }

                contextReady = true;
                CacheOrganRestBounds(data);
                InitializeCudaXpbd(data, edgeIds);
                status = cudaXpbdReady
                    ? "Static and CUDA XPBD data uploaded. Unity XPBD remains the active runtime driver unless CUDA Driver Isolated is selected."
                    : "Static tetrahedral data uploaded. Legacy Unity XPBD remains the active runtime driver.";
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
            lastNativeError = NativeOrganGetStats(contextHandle, out OrganContextStatsNative stats);
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
            cudaDynamicDeviceBytes = stats.dynamicDeviceBytes;
            cudaXpbdStepCount = stats.xpbdStepCount;
            cudaLastXpbdMilliseconds = stats.lastXpbdMilliseconds;
            cudaTotalXpbdMilliseconds = stats.totalXpbdMilliseconds;
            cudaNanCount = stats.lastNanCount;
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
            int rc = NativeOrganDestroy(contextHandle);
            if (rc != 0 && logContextDiagnostics)
                Debug.LogWarning($"[CudaOrganContext] LCS_OrganDestroy({contextHandle}) returned {rc}.", this);
            contextHandle = 0;
            contextReady = false;
            cudaXpbdReady = false;
        }

        void InitializeCudaXpbd(TetMeshData data, int[] edgeIds)
        {
            cudaXpbdReady = false;
            if (!enableCudaXpbd || cudaXpbdMode == CudaXpbdMode.Disabled)
            {
                cudaXpbdStatus = "CUDA XPBD is disabled; phase-1 static context remains active.";
                return;
            }

            try
            {
                BuildEdgeColorGroups(edgeIds, data.NumParticles,
                    out int[] edgeOffsets, out int[] edgeCounts, out int[] edgeFlat);
                BuildTetColorGroups(data,
                    out int[] tetOffsets, out int[] tetCounts, out int[] tetFlat);
                BuildSurfaceTriangleColorGroups(data,
                    out int[] surfaceOffsets, out int[] surfaceCounts, out int[] surfaceFlat);
                cudaEdgeColorCount = edgeCounts.Length;
                cudaTetColorCount = tetCounts.Length;
                lastNativeError = NativeXpbdInitialize(contextHandle,
                    Flatten(data.Positions, data.NumParticles),
                    edgeCounts.Length, edgeOffsets, edgeCounts, edgeFlat,
                    tetCounts.Length, tetOffsets, tetCounts, tetFlat,
                    surfaceCounts.Length, surfaceOffsets, surfaceCounts, surfaceFlat);
                cudaXpbdReady = lastNativeError == 0;
                cudaXpbdStatus = cudaXpbdReady
                    ? "CUDA XPBD initialized. Read Only Comparison does not write the Unity organ. CUDA Driver Isolated does not publish positions; Temporary Publish is reserved for phase-3 visual verification."
                    : $"LCS_OrganXpbdInitialize failed ({lastNativeError}). Unity XPBD remains active.";
            }
            catch (EntryPointNotFoundException)
            {
                cudaXpbdStatus = "The deployed LiverCudaSim.dll does not contain the phase-2 CUDA XPBD API. Rebuild and deploy the DLL.";
            }
            catch (Exception exception)
            {
                cudaXpbdStatus = $"CUDA XPBD setup failed: {exception.Message}";
                Debug.LogError($"[CudaOrganContext] {cudaXpbdStatus}", this);
            }
        }

        public void StepCudaXpbd(float dt)
        {
            if (!contextReady || !cudaXpbdReady || !enableCudaXpbd || cudaXpbdMode == CudaXpbdMode.Disabled)
                return;

            Vector3 gravity = _softBody.enableGravity
                ? new Vector3(0f, _softBody.gravityY, 0f)
                : Vector3.zero;
            OrganContextXpbdParams parameters = new OrganContextXpbdParams
            {
                numSubSteps = Mathf.Max(1, _softBody.numSubSteps),
                constraintIterations = Mathf.Max(1, cudaConstraintIterations),
                edgeCompliance = Mathf.Max(0f, _softBody.edgeCompliance),
                youngsModulus = Mathf.Max(1f, _softBody.youngsModulus),
                poissonsRatio = Mathf.Clamp(_softBody.poissonsRatio, 0.01f, 0.499f),
                damping = Mathf.Clamp01(_softBody.damping),
                gravityX = gravity.x,
                gravityY = gravity.y,
                gravityZ = gravity.z,
                groundY = _softBody.groundY
            };
            lastNativeError = NativeXpbdStep(contextHandle, dt, ref parameters);
            if (lastNativeError != 0)
            {
                cudaXpbdStatus = $"LCS_OrganXpbdStep failed ({lastNativeError}). Unity XPBD remains available as the fallback.";
                return;
            }

            if (cudaXpbdMode == CudaXpbdMode.CudaDriverIsolated)
                cudaXpbdStatus = "CUDA Driver Isolated is active. No complete CUDA-to-CPU tet position readback is performed; legacy rendering, Stage 3, cutter, and gripper are intentionally not driven.";
        }

        // Returns true only when this fixed step has a valid CUDA contact packet. Callers
        // must retain the legacy contact path when this is false.
        public bool SubmitCudaToolInput(GripperTool gripper, SoftBodyCollisionModule settings, float dt)
        {
            cudaToolContactActive = false;
            if (!IsCudaToolContactDriverActive || gripper == null || settings == null)
            {
                cudaToolContactStatus = "CUDA tool contact is unavailable because the driver, gripper, or collision settings are missing.";
                return false;
            }

            if (!gripper.TryBuildCudaToolInput(_toolCapsules, out int capsuleCount, out Bounds bounds,
                    out Vector3 center, out Vector3 axisU, out Vector3 axisV, out Vector3 axisW))
            {
                cudaToolContactStatus = $"CUDA tool input was not submitted: {gripper.CudaToolInputStatus} Legacy Unity contact remains active for this step.";
                return false;
            }

            OrganContextToolContactParamsNative parameters = new OrganContextToolContactParamsNative
            {
                capsuleCount = capsuleCount,
                contactEnabled = (settings.keepContactActiveWhenIdle || gripper.WantsToolContact) ? 1 : 0,
                keepContactActiveWhenIdle = settings.keepContactActiveWhenIdle ? 1 : 0,
                useCandidateCulling = settings.useToolContactCandidateCulling ? 1 : 0,
                contactIterations = Mathf.Max(1, settings.toolContactIterations),
                couplingPasses = Mathf.Max(1, settings.toolContactCouplingPasses),
                graspRequest = gripper.WantsClosedGrasp ? 1 : 0,
                releaseRequest = gripper.WantsClosedGrasp ? 0 : 1,
                contactDistance = Mathf.Max(0f, settings.toolContactDistance),
                contactCompliance = Mathf.Max(0f, settings.toolContactCompliance),
                tangentialFriction = Mathf.Clamp01(settings.toolContactTangentialFriction),
                tangentialDamping = Mathf.Clamp01(settings.toolContactTangentialDamping),
                candidatePadding = Mathf.Max(0f, settings.toolContactCandidatePadding),
                graspHeight = Mathf.Max(0f, gripper.gaussianHeight),
                graspCoreRadius = Mathf.Clamp(gripper.gaussianHardCoreRadius, 0.1f, 1f),
                graspFormDuration = Mathf.Max(0f, gripper.gaussianFormDuration),
                graspGaussianWidth = Mathf.Clamp(gripper.gaussianWidth, 0.05f, 1f),
                graspInfluenceRadius = Mathf.Clamp(
                    gripper.gaussianInfluenceRadius,
                    Mathf.Clamp(gripper.gaussianHardCoreRadius, 0.1f, 1f), 1f),
                graspSoftFollowRate = Mathf.Max(0.1f, gripper.gaussianSoftFollowRate),
                graspBoundsMinX = bounds.min.x, graspBoundsMinY = bounds.min.y, graspBoundsMinZ = bounds.min.z,
                graspBoundsMaxX = bounds.max.x, graspBoundsMaxY = bounds.max.y, graspBoundsMaxZ = bounds.max.z,
                frameCenterX = center.x, frameCenterY = center.y, frameCenterZ = center.z,
                axisUX = axisU.x, axisUY = axisU.y, axisUZ = axisU.z,
                axisVX = axisV.x, axisVY = axisV.y, axisVZ = axisV.z,
                axisWX = axisW.x, axisWY = axisW.y, axisWZ = axisW.z
            };

            bool toolOverlapsOrgan = !enableOrganToolBroadphase || ToolBoundsOverlapOrgan(bounds, settings);
            bool keepLockedGraspActive = _cudaGraspMayBeActive && gripper.WantsClosedGrasp;
            bool releaseLockedGrasp = _cudaGraspMayBeActive && !gripper.WantsClosedGrasp;
            bool useDisabledPacket = !toolOverlapsOrgan && !keepLockedGraspActive;
            cudaToolBroadphaseOverlapping = toolOverlapsOrgan;
            cudaToolBroadphaseDisabled = useDisabledPacket;
            if (useDisabledPacket)
            {
                capsuleCount = 0;
                parameters.capsuleCount = 0;
                parameters.contactEnabled = 0;
                parameters.graspRequest = 0;
                parameters.releaseRequest = releaseLockedGrasp ? 1 : 0;
            }

            lastNativeError = NativeToolStep(contextHandle, dt, _toolCapsules, ref parameters);
            if (lastNativeError != 0)
            {
                cudaToolContactStatus = $"LCS_OrganToolStep failed ({lastNativeError}). Legacy Unity contact remains active for this step.";
                return false;
            }
            cudaToolContactActive = true;
            if (!gripper.WantsClosedGrasp)
                _cudaGraspMayBeActive = false;
            else if (toolOverlapsOrgan)
                _cudaGraspMayBeActive = true;
            cudaToolGraspLocked = _cudaGraspMayBeActive;
            cudaToolContactStatus = releaseLockedGrasp
                ? "CUDA grasp released after the tool left this organ; contact is now disabled by the organ bounds broad phase."
                : keepLockedGraspActive && !toolOverlapsOrgan
                ? "CUDA grasp lock keeps this organ active outside its rest bounds until the gripper opens."
                : useDisabledPacket
                ? "CUDA tool contact disabled by the organ bounds broad phase; no capsule upload or contact iteration is needed for this organ."
                : $"CUDA tool packet uploaded ({capsuleCount} capsules). Contact and grasp constraints execute inside the following CUDA XPBD step.";
            return true;
        }

        void CacheOrganRestBounds(TetMeshData data)
        {
            _organRestBoundsReady = false;
            if (data == null || data.RestPositions == null || data.NumParticles <= 0)
                return;

            Bounds bounds = new Bounds(data.RestPositions[0], Vector3.zero);
            for (int i = 1; i < data.NumParticles; ++i)
                bounds.Encapsulate(data.RestPositions[i]);
            _organRestBounds = bounds;
            _organRestBoundsReady = true;
        }

        bool ToolBoundsOverlapOrgan(Bounds toolBounds, SoftBodyCollisionModule settings)
        {
            if (!_organRestBoundsReady)
                return true;

            Bounds expandedOrganBounds = _organRestBounds;
            float padding = Mathf.Max(0f, settings.organBroadphasePadding) +
                            Mathf.Max(0f, settings.toolContactCandidatePadding) +
                            Mathf.Max(0f, settings.toolContactDistance);
            expandedOrganBounds.Expand(2f * padding);
            return expandedOrganBounds.Intersects(toolBounds);
        }

        public bool IsWorldBoundsNearOrgan(Bounds worldBounds, float padding)
        {
            if (!_organRestBoundsReady)
                return true;

            Bounds expandedOrganBounds = _organRestBounds;
            expandedOrganBounds.Expand(2f * Mathf.Max(0f, padding));
            return expandedOrganBounds.Intersects(worldBounds);
        }

        public bool PublishCudaPositions(TetMeshData data)
        {
            if (!RequiresTemporaryCudaPositionPublish || data == null)
                return false;
            int count = data.NumParticles;
            if (_temporaryPositionReadback == null || _temporaryPositionReadback.Length != count * 3)
                _temporaryPositionReadback = new float[count * 3];
            lastNativeError = NativeXpbdGetPositions(contextHandle, _temporaryPositionReadback, count);
            if (lastNativeError != 0)
            {
                cudaXpbdStatus = $"LCS_OrganXpbdGetPositions failed ({lastNativeError}).";
                return false;
            }
            for (int i = 0; i < count; ++i)
            {
                Vector3 position = new Vector3(_temporaryPositionReadback[i * 3], _temporaryPositionReadback[i * 3 + 1], _temporaryPositionReadback[i * 3 + 2]);
                data.Positions[i] = position;
                data.PrevPositions[i] = position;
                data.Velocities[i] = Vector3.zero;
            }
            cudaXpbdStatus = "CUDA XPBD and CUDA tool contact are active. Temporary Publish performs one full position readback per fixed step for phase-3 visual verification.";
            return true;
        }

        public bool ConfigureCudaTetToGrid(int[] hostTetByCorner, Vector4[] weights, Vector3[] alignedRestCorners,
                                           byte[] activeMask, Vector3 restCentroid)
        {
            cudaTetToGridReady = false;
            gpuResidentTetToGridFrameActive = false;
            if (!CanUseCudaTetToGrid || hostTetByCorner == null || weights == null || alignedRestCorners == null || activeMask == null ||
                hostTetByCorner.Length == 0 || weights.Length != hostTetByCorner.Length ||
                alignedRestCorners.Length != hostTetByCorner.Length || activeMask.Length != hostTetByCorner.Length)
            {
                cudaTetToGridStatus = "CUDA Tet-to-Grid requires a ready CUDA driver context and matching Stage 3 mapping arrays.";
                return false;
            }

            int count = hostTetByCorner.Length;
            float[] flatWeights = new float[count * 4];
            float[] flatRest = new float[count * 3];
            for (int i = 0; i < count; ++i)
            {
                Vector4 w = weights[i];
                flatWeights[4 * i] = w.x; flatWeights[4 * i + 1] = w.y; flatWeights[4 * i + 2] = w.z; flatWeights[4 * i + 3] = w.w;
                Vector3 p = alignedRestCorners[i];
                flatRest[3 * i] = p.x; flatRest[3 * i + 1] = p.y; flatRest[3 * i + 2] = p.z;
            }
            OrganContextTetToGridDescNative desc = new OrganContextTetToGridDescNative
            {
                cornerCount = count,
                restCentroidX = restCentroid.x, restCentroidY = restCentroid.y, restCentroidZ = restCentroid.z
            };
            try
            {
                lastNativeError = NativeTetToGridConfigure(contextHandle, ref desc, hostTetByCorner, flatWeights, flatRest, activeMask);
            }
            catch (EntryPointNotFoundException)
            {
                cudaTetToGridStatus = "The deployed LiverCudaSim.dll predates the phase-4 Tet-to-Grid API.";
                return false;
            }
            if (lastNativeError != 0)
            {
                cudaTetToGridStatus = $"LCS_OrganTetToGridConfigure failed ({lastNativeError}); Stage 3 CPU synchronization remains available.";
                return false;
            }
            cudaTetToGridReady = true;
            cudaTetToGridStatus = "CUDA Tet-to-Grid mapping uploaded once; dynamic corner positions remain on GPU.";
            ReadTetToGridStats();
            return true;
        }

        public void SetGpuResidentTetToGridFrameActive(bool active)
        {
            gpuResidentTetToGridFrameActive = active && cudaTetToGridReady && CanUseCudaTetToGrid;
        }

        public bool UpdateCudaTetToGrid(bool applyLocalDeformation)
        {
            if (!cudaTetToGridReady || !CanUseCudaTetToGrid)
                return false;
            lastNativeError = NativeTetToGridUpdate(contextHandle, applyLocalDeformation ? 1 : 0);
            if (lastNativeError != 0)
            {
                cudaTetToGridStatus = $"LCS_OrganTetToGridUpdate failed ({lastNativeError}); Stage 3 CPU synchronization remains available.";
                cudaTetToGridReady = false;
                return false;
            }
            return true;
        }

        public void ReadTetToGridStats()
        {
            if (!contextReady || !enableCudaTetToGrid) return;
            OrganContextTetToGridStatsNative stats;
            try { lastNativeError = NativeTetToGridGetStats(contextHandle, out stats); }
            catch (EntryPointNotFoundException) { return; }
            if (lastNativeError != 0) { cudaTetToGridLastError = lastNativeError; return; }
            cudaTetToGridMappedCorners = stats.mappedCorners;
            cudaTetToGridFallbackCorners = stats.fallbackCorners;
            cudaTetToGridUploadBytes = stats.uploadBytes;
            cudaTetToGridUpdateCount = stats.updateCount;
            cudaTetToGridLastMilliseconds = stats.lastUpdateMilliseconds;
            cudaTetToGridTotalMilliseconds = stats.totalUpdateMilliseconds;
            cudaTetToGridLastError = stats.lastError;
        }

        void ReadToolStats()
        {
            lastNativeError = NativeToolGetStats(contextHandle, out OrganContextToolContactStatsNative tool);
            if (lastNativeError != 0)
            {
                cudaToolContactStatus = $"LCS_OrganToolGetStats failed ({lastNativeError}).";
                return;
            }
            cudaToolActiveCandidates = tool.activeCandidates;
            cudaToolContactCount = tool.contactCount;
            cudaToolMaxContactDepth = tool.maxContactDepth;
            cudaToolSurfaceCandidateTriangles = tool.surfaceCandidateTriangles;
            cudaToolSurfaceContactTriangles = tool.surfaceContactTriangles;
            cudaToolSurfaceMaxContactDepth = tool.surfaceMaxContactDepth;
            cudaToolGraspedParticleCount = tool.graspedParticleCount;
            cudaToolDispatchCount = tool.dispatchCount;
            cudaToolUploadBytes = tool.toolUploadBytes;
            cudaToolUploadOperations = tool.toolUploadOperations;
            cudaToolLastMilliseconds = tool.lastToolMilliseconds;
            cudaToolTotalMilliseconds = tool.totalToolMilliseconds;
            cudaToolLastError = tool.lastToolError;
            cudaToolContactStatus = "CUDA tool diagnostics sampled. Contact and grasp constraints are resident in CudaOrganContext.";
        }

        public void CompareCudaWithUnity(TetMeshData data)
        {
            if (!contextReady || !cudaXpbdReady || !enableCudaXpbd ||
                cudaXpbdMode != CudaXpbdMode.ReadOnlyComparison || data == null ||
                Time.unscaledTime < _nextComparisonTime)
                return;

            _nextComparisonTime = Time.unscaledTime + Mathf.Max(0.05f, comparisonIntervalSeconds);
            lastNativeError = NativeXpbdCompare(contextHandle,
                Flatten(data.Positions, data.NumParticles), data.NumParticles,
                out OrganContextComparisonStatsNative comparison);
            if (lastNativeError != 0)
            {
                cudaXpbdStatus = $"LCS_OrganXpbdComparePositions failed ({lastNativeError}).";
                return;
            }

            comparisonSamples++;
            comparisonParticleCount = comparison.particleCount;
            cudaNanCount = comparison.cudaNanCount;
            positionMaxError = comparison.maxError;
            positionRmsError = comparison.rmsError;
            centroidError = comparison.centroidError;
            bboxMinError = comparison.bboxMinError;
            bboxMaxError = comparison.bboxMaxError;
            bboxExtentError = comparison.bboxExtentError;
            cudaXpbdStatus = comparison.cudaNanCount == 0
                ? "Read Only Comparison is active. Unity XPBD is the sole writer; CUDA values are diagnostic only."
                : $"Read Only Comparison detected {comparison.cudaNanCount} CUDA NaN particle(s).";
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

        static float[] BuildUnityEquivalentInverseMass(TetMeshData data, float density)
        {
            float[] mass = new float[data.NumParticles];
            for (int tet = 0; tet < data.NumTets; tet++)
            {
                if (!data.TetActive[tet])
                    continue;
                float contribution = Mathf.Max(0f, density) * data.RestVolumes[tet] * 0.25f;
                int baseIndex = tet * 4;
                mass[data.TetIds[baseIndex]] += contribution;
                mass[data.TetIds[baseIndex + 1]] += contribution;
                mass[data.TetIds[baseIndex + 2]] += contribution;
                mass[data.TetIds[baseIndex + 3]] += contribution;
            }

            float sum = 0f;
            int count = 0;
            for (int i = 0; i < data.NumParticles; i++)
            {
                if (data.InvMass[i] != 0f && mass[i] > 1e-12f)
                {
                    sum += mass[i];
                    count++;
                }
            }
            float massFloor = (count > 0 ? sum / count : 1f) * 0.1f;
            float[] result = new float[data.NumParticles];
            for (int i = 0; i < data.NumParticles; i++)
                result[i] = data.InvMass[i] == 0f ? 0f : 1f / Mathf.Max(mass[i], massFloor);
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

        static void BuildTetColorGroups(TetMeshData data,
            out int[] offsets, out int[] counts, out int[] flat)
        {
            List<int[]> groups = GraphColoring.Compute(
                data.TetIds, data.NumTets, data.NumParticles, data.TetActive);
            FlattenGroups(groups, out offsets, out counts, out flat);
        }

        static void BuildEdgeColorGroups(int[] edgeIds, int particleCount,
            out int[] offsets, out int[] counts, out int[] flat)
        {
            int edgeCount = edgeIds.Length / 2;
            int[] color = new int[edgeCount];
            Array.Fill(color, -1);
            var incident = new List<int>[particleCount];
            for (int particle = 0; particle < particleCount; particle++)
                incident[particle] = new List<int>(8);
            for (int edge = 0; edge < edgeCount; edge++)
            {
                incident[edgeIds[edge * 2]].Add(edge);
                incident[edgeIds[edge * 2 + 1]].Add(edge);
            }

            int maxColor = -1;
            for (int edge = 0; edge < edgeCount; edge++)
            {
                var used = new HashSet<int>();
                foreach (int neighbour in incident[edgeIds[edge * 2]])
                    if (color[neighbour] >= 0) used.Add(color[neighbour]);
                foreach (int neighbour in incident[edgeIds[edge * 2 + 1]])
                    if (color[neighbour] >= 0) used.Add(color[neighbour]);
                int assigned = 0;
                while (used.Contains(assigned)) assigned++;
                color[edge] = assigned;
                maxColor = Mathf.Max(maxColor, assigned);
            }

            var groups = new List<int[]>(maxColor + 1);
            for (int group = 0; group <= maxColor; group++)
            {
                var members = new List<int>();
                for (int edge = 0; edge < edgeCount; edge++)
                    if (color[edge] == group) members.Add(edge);
                groups.Add(members.ToArray());
            }
            FlattenGroups(groups, out offsets, out counts, out flat);
        }

        // Triangles in one group never share a particle, so the CUDA contact kernel can
        // update its three vertices without atomics or write races.
        static void BuildSurfaceTriangleColorGroups(TetMeshData data,
            out int[] offsets, out int[] counts, out int[] flat)
        {
            int triangleCount = data != null ? data.NumSurfaceTris : 0;
            if (triangleCount <= 0 || data.SurfaceTriIds == null)
            {
                offsets = Array.Empty<int>();
                counts = Array.Empty<int>();
                flat = Array.Empty<int>();
                return;
            }

            int[] color = new int[triangleCount];
            Array.Fill(color, -1);
            var incident = new List<int>[data.NumParticles];
            for (int particle = 0; particle < incident.Length; particle++)
                incident[particle] = new List<int>(8);
            for (int triangle = 0; triangle < triangleCount; triangle++)
            {
                int baseIndex = triangle * 3;
                incident[data.SurfaceTriIds[baseIndex]].Add(triangle);
                incident[data.SurfaceTriIds[baseIndex + 1]].Add(triangle);
                incident[data.SurfaceTriIds[baseIndex + 2]].Add(triangle);
            }

            int maxColor = -1;
            for (int triangle = 0; triangle < triangleCount; triangle++)
            {
                var used = new HashSet<int>();
                int baseIndex = triangle * 3;
                for (int corner = 0; corner < 3; corner++)
                    foreach (int neighbour in incident[data.SurfaceTriIds[baseIndex + corner]])
                        if (color[neighbour] >= 0) used.Add(color[neighbour]);
                int assigned = 0;
                while (used.Contains(assigned)) assigned++;
                color[triangle] = assigned;
                maxColor = Mathf.Max(maxColor, assigned);
            }

            var groups = new List<int[]>(maxColor + 1);
            for (int group = 0; group <= maxColor; group++)
            {
                var members = new List<int>();
                for (int triangle = 0; triangle < triangleCount; triangle++)
                    if (color[triangle] == group) members.Add(triangle);
                groups.Add(members.ToArray());
            }
            FlattenGroups(groups, out offsets, out counts, out flat);
        }

        static void FlattenGroups(List<int[]> groups, out int[] offsets, out int[] counts, out int[] flat)
        {
            offsets = new int[groups.Count];
            counts = new int[groups.Count];
            int total = 0;
            for (int group = 0; group < groups.Count; group++) total += groups[group].Length;
            flat = new int[total];
            int offset = 0;
            for (int group = 0; group < groups.Count; group++)
            {
                offsets[group] = offset;
                counts[group] = groups[group].Length;
                Array.Copy(groups[group], 0, flat, offset, groups[group].Length);
                offset += groups[group].Length;
            }
        }

        static string ToHex(ulong value) => $"0x{value:X16}";
    }
}
