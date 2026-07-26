// LiverCudaManager.cs - liver loading, physics, reconstruction, and rendering orchestration.
//
// Cutting setup and execution live in LiverCudaCutter. This manager only creates/fetches that sibling
// component and lets it participate in the startup and per-frame simulation loop.
//
// Build cuda_plugin -> LiverCudaSim.dll -> Assets/Plugins/x86_64/. Attach to an empty GameObject; Play.
//   Cutting is owned by the sibling LiverCudaCutter component. Assign its LiverCuttingRod target to
//   inverseAPI/TargetObject.targetObject so the Haply cursor drives the line.

using System.IO;
using System.Runtime.InteropServices;
using UnityEngine;
using UnityEngine.Rendering;
using Unity.Mathematics;
using ReconGridDC.Preprocess;
using ReconGridDC.Demo;      // LiverGrid, RodMath, OrbitCamera

namespace ReconGridDC.Cuda
{
    [RequireComponent(typeof(LiverCudaCutter))]
    public sealed class LiverCudaManager : MonoBehaviour
    {
        [Header("Liver model (under StreamingAssets)")]
        public string mshFileName = "liver3_refined_1.msh";

        [Header("Grid fitting + smoothing")]
        public int targetLongAxisVoxels = 48;
        public int marginVoxels         = 2;
        public int qefIters             = 10;
        public int smoothIters          = 3;
        public int bakeOversample       = 1;

        [Header("Physics — paper SS3.4 liver values")]
        public float gravity    = -9.81f;
        public float physicsDt  = 0.02f;
        public float ks         = 7.5e4f;   // paper SS3.4 structural stiffness
        public float kb         = 2e4f;     // paper SS3.4 bending stiffness
        public float cs_damp    = 0.92f;    // paper SS3.4 structural damping (VERBATIM)
        public float cb         = 0.9f;     // paper SS3.4 bending damping    (VERBATIM)
        // PAPER-SILENT global mass-proportional (Rayleigh) damping (1/s). The ONLY term that damps the
        // rigid pendulum/slosh mode (Kelvin-Voigt cs/cb damp only RELATIVE velocity). 0 = paper-exact;
        // ~2-3 lets the soft liver sag and SETTLE instead of swinging forever (free-fall terminal v = g/alpha).
        public float globalDamp = 2.0f;
        public float massScale  = 1f;       // PAPER-SILENT; keep near 1 (holds shape; deformation from cutting)
        // XPBD GO-LIVE (design rev-B §9-(4)): the native solver is now XPBD small-steps. solverHInit /
        // solverHMax are DEAD (the RK45 adaptive-h controller is retired; kept only for the
        // PhysicsInitDesc ABI). maxSubstepsPerFrame is REPURPOSED as the fixed substep count N
        // (h = physicsDt / N); 20 is the steps-1-3-validated configuration. §9-(6) re-derives
        // N = max(N_cut, N_elastic) when the cut tick lands.
        public float solverHInit = 1e-4f;   // DEAD under XPBD (ABI placeholder)
        public float solverHMax  = 1.5e-3f; // DEAD under XPBD (ABI placeholder)
        public int   maxSubstepsPerFrame = 20;

        [Header("Anchor (deformable + pinned)")]
        [Tooltip("Off: preserve the current fixed cap at one end. On: remove that cap and pin a sparse set of points throughout the liver to prevent large rigid-body motion while preserving local deformation.")]
        public bool pinDistributedAnchors = false;
        [Tooltip("Number of interior simulation points fixed when Pin Distributed Anchors is enabled. Increase gradually if the organ still moves too much; high values reduce deformation.")]
        [Min(4)] public int distributedAnchorCount = 12;
        public Vector3 anchorAxis = new Vector3(1, 0, 0);
        public float   anchorBand = 1.5f;

        [Header("Liver surface rendering (Project2 Built-in approximation)")]
        public bool useLiverSurfaceShader = true;
        // Serialized Shader reference guarantees inclusion in player builds (design §5 / review
        // A3-m4: Shader.Find returns null in builds unless the shader is referenced by an asset).
        public Shader    liverSurfaceShader;
        public Texture2D liverAlbedoTex;    // liver2.png        (sRGB ON)  -> _MainTex
        // Kept serialized for scene compatibility. Project2 uses liver2_spec.png as _BumpMap.
        public Texture2D liverNormalTex;    // liver-texture-square_bump.png (optional legacy slot)
        public Texture2D liverSpecTex;      // liver2_spec.png   (sRGB off) -> _BumpMap
        public Texture2D liverHeightTex;    // liver2_height.png (sRGB ON, reference meta) -> _ParallaxMap
        // Reference _TriplanarScale 5.5 x their 2.6735u rest span = 14.70 texture tiles across
        // the liver; our UV1 rest anchors span ~17.8u -> 14.70/17.8 = 0.83 (P3 conversion).
        public float triplanarScale = 0.83f;
        // Reference scene runs its directional light at intensity 2.0 (their SampleScene.unity)
        // vs our 1.0 — a faithful material port renders ~half as bright without this (P3-m5).
        public bool matchReferenceLighting = true;
        public bool matchReferenceSkybox = true;
        public bool matchReferenceFog = true;
        public bool createReferenceGround = true;
        public Color referenceGroundColor = new Color(0.22f, 0.72f, 0.22f, 1f);

        [Header("Rendering - smooth visual mesh")]
        public bool useSmoothVisualRenderer = true;
        public bool hideRawCudaSurfaceWhenVisualActive = true;

        [Header("Surface membrane visualization")]
        [Tooltip("Draw a thin translucent layer on the live reconstructed outer surface. This is visualization only and does not change physics or collision behavior.")]
        public bool showSurfaceMembrane = true;
        public Shader surfaceMembraneShader;
        public Color surfaceMembraneColor = new Color(1f, 0.97f, 0.72f, 0.05f);
        [Tooltip("Overall membrane opacity. 0 is invisible and 1 is fully opaque before the rim adjustment.")]
        [Range(0f, 1f)] public float surfaceMembraneOpacity = 0.05f;
        [Tooltip("Small outward offset along the current surface normal to avoid z-fighting with the liver material.")]
        [Min(0f)] public float surfaceMembraneOffset = 0.015f;
        [Range(0f, 2f)] public float surfaceMembraneRimStrength = 0.65f;

        // ── SoftBody.cs rendering panel mirror (defaults = SoftBody EFFECTIVE runtime values:
        // field defaults passed through the SetupTwoSidedMaterial clamps, P1 exact table).
        // CreateLiverMaterial re-applies the same clamps to whatever the user types, so the
        // material can never leave the reference's sanity envelope.
        [Header("渲染 - 基础颜色")]
        [Tooltip("切面內部顏色（our UV1.w cut-wall fold-in; = reference EffectiveCutColor）")]
        public Color interiorColor = new Color(0.30f, 0.025f, 0.018f, 1f);
        [Tooltip("肝臟表面顏色（reference EffectiveLiverColor result）")]
        public Color liverColor = new Color(0.62f, 0.18f, 0.10f, 1f);
        [Range(0f, 1f)]      public float liverTextureStrength  = 0.98f;  // clamp max(x, 0.95)
        [Range(0.25f, 2.5f)] public float liverTextureContrast  = 1.35f;  // clamp max(x, 1.35)
        [Range(0f, 1f)]      public float liverTextureColorBlend = 1.0f;  // SoftBody hardcodes 1.0
        [Tooltip("0 = rest-triplanar LiverTissueGPU path; 1 = SofaUnity/Project2-style planar UV texture path.")]
        [Range(0f, 1f)]      public float liverUvTextureWeight  = 1.0f;
        public Vector2 liverTextureTiling = new Vector2(2f, 2f);
        public Vector2 liverTextureOffset = new Vector2(0.15f, 0f);
        [Range(0.1f, 16f)]   public float liverTriplanarBlend   = 4f;     // runtime 4.0, not prop default 5.0

        [Header("渲染 - 法线贴图 (Project2 _BumpMap path)")]
        [Range(0f, 3f)] public float liverNormalStrength           = 0.12f;  // clamp [0.04, 0.12]
        [Range(0f, 2f)] public float liverProceduralNormalStrength = 0.05f;  // clamp min(x, 0.05)

        [Header("渲染 - PBR 高光")]
        [Range(0f, 1f)] public float liverRoughness        = 0.70f;          // clamp max(x, 0.70)
        [Range(0f, 3f)] public float liverSpecularStrength = 0.18f;          // clamp min(x, 0.18)
        public Color liverSpecularColor = new Color(0.95f, 0.90f, 0.85f, 1f);

        [Header("渲染 - 次表面散射 SSS")]
        public Color liverSSSColor = new Color(0.76f, 0.13f, 0.055f, 1f);
        [Range(0f, 3f)]  public float liverSSSStrength  = 0.06f;             // clamp min(x, 0.06)
        [Range(0f, 2f)]  public float liverSSSBacklight = 0.04f;             // clamp min(x, 0.04)
        [Range(1f, 24f)] public float liverSSSPower     = 7f;
        [Range(0f, 1f)]  public float liverSSSWrap      = 0.32f;

        [Header("渲染 - Fresnel 边缘")]
        [Range(0f, 3f)] public float liverFresnelStrength = 0.08f;           // clamp min(x, 0.08)
        [Range(1f, 8f)] public float liverFresnelPow      = 4.2f;

        [Header("Rendering - GPU Tissue Detail")]
        [Range(0f, 1f)] public float liverWetness             = 0.18f;       // clamp min(x, 0.18)
        [Range(0f, 1f)] public float liverMicroMottleStrength = 0.08f;       // clamp min(x, 0.08)
        // Rest-space noise frequencies CONVERTED x0.1502 (their 2.6735u rest span / our 17.8u):
        // reference 20 -> 3.0, 22 -> 3.3; copying 20/22 verbatim renders mottle/veins ~6.7x too fine.
        [Range(0.1f, 80f)]  public float liverMicroMottleScale = 3.0f;
        [Range(0.2f, 1.5f)] public float liverAlbedoBrightness = 0.82f;      // clamp min(x, 0.82)
        [Range(0f, 1f)]     public float liverVeinStrength     = 0.10f;
        [Range(0.1f, 80f)]  public float liverVeinScale        = 3.3f;
        [Tooltip("调试用：1 时直接显示 liver2 triplanar，绕过光照（贴图链路诊断）")]
        [Range(0f, 1f)]     public float liverDebugTextureOnly = 0f;
        // v5.1 D10 tear law (the 藕断丝连 fix): sever any intact edge stretched beyond this ratio x
        // rest length. Rim bridge edges the sweep never covered are dragged 10-50L by the falling
        // piece — tissue past its elastic limit TEARS instead of stretching forever. MUST exceed
        // the max legit gravity strain (review wf_c416ae77 R2-m1: ~2.5-3x observed in this rig,
        // so 4.0 keeps margin; bridges blow through any tau within a frame or two). <1 disables.
        public float tearStretchRatio    = 4.0f;

        [Header("Game-mode camera")]
        public bool setupOrbitCamera = true;

        // ── native plugin P/Invoke ────────────────────────────────────────────────────────────
        const string DLL = "LiverCudaSim";

        [StructLayout(LayoutKind.Sequential)]
        struct ReconInitDesc { public int voxelCount, cornerCount, isectCount, surfaceEdgeCount, triCapacity, dimsX, dimsY, dimsZ; public float L; public int qefIters; }
        [StructLayout(LayoutKind.Sequential)]
        struct IsectInterop { public float gx, gy, gz, nx, ny, nz; public int cornerA, cornerB; public float t; public int insideA; }
        [StructLayout(LayoutKind.Sequential)]
        struct GridEdgeInterop { public int axis, bcX, bcY, bcZ, insideA, p0, p1, p2; }   // 32B (surface edges)
        [StructLayout(LayoutKind.Sequential)]
        struct PhysicsInitDesc { public int cornerCount, bendPairCount; public float ks, cs, kb, cb, hInit, hMax; public int maxSubsteps; public float alpha; }
        [StructLayout(LayoutKind.Sequential)]
        struct BendPairInterop { public int i, j, k; public float theta0; public int alive, p0, p1, p2; }   // 32B

        [DllImport(DLL)] static extern int  LCS_Init(ref ReconInitDesc desc, float[] cornerPos, int[] voxelCorner, int[] voxelIsectOffset, int[] voxelIsectCount, IsectInterop[] isect, GridEdgeInterop[] surfaceEdges);
        [DllImport(DLL)] static extern int  LCS_InitPhysics(ref PhysicsInitDesc desc, float[] mass, int[] pinned, int[] active, float[] extForce, int[] nbrIdx, float[] restNbr, BendPairInterop[] bendPairs);
        [DllImport(DLL)] static extern int  LCS_Step(float dt);
        [DllImport(DLL)] static extern int  LCS_SetCornerPos(float[] positions);
        [DllImport(DLL)] static extern int  LCS_ComputeRotations();
        [DllImport(DLL)] static extern int  LCS_Finalize();
        [DllImport(DLL)] static extern int  LCS_GetSurfaceCounts(out int vertexCount, out int indexCount);
        [DllImport(DLL)] static extern int  LCS_GetSurface([Out] float[] outPos3, [Out] float[] outNrm3, [Out] int[] outIdx);
        [DllImport(DLL)] static extern int  LCS_GetNbrIdx([Out] int[] outNbr);       // diagnostic: live post-sever nbrIdx
        [DllImport(DLL)] static extern int  LCS_GetCornerPos([Out] float[] outPos);  // diagnostic: live deformed cornerPos
        [DllImport(DLL)] static extern int  LCS_GetCutDebug([Out] uint[] out18, out float alpha);  // [DEBUG-CUT] v4.1 P3
        [DllImport(DLL)] static extern int  LCS_GetSurfaceAux([Out] float[] outAux4);  // UV1: rest anchor + wall flag (render design §4.4)
        [DllImport(DLL)] static extern void LCS_Shutdown();

        // ── runtime ───────────────────────────────────────────────────────────────────────────
        Mesh      mesh;
        Material  mat;
        Material  cutMat;
        Material  membraneMat;
        Material  groundMat;
        Material  skyboxMat;
        MeshRenderer rawMeshRenderer;
        DcLiverVisualRenderer visualRenderer;
        GameObject referenceGround;
        bool      referenceGroundCreated;
        bool      initialized, cutReady;
        bool      presentationVisible = true;
        bool      _liverMatActive;   // LiverSurface material in use -> read the aux channel
        bool      _auxUnsupported;   // stale DLL without LCS_GetSurfaceAux -> degrade gracefully
        int       maxIdx;            // = 3 * triCapacity (the non-indexed soup upper bound)
        float[]   pos, nrm;
        float[]   aux;               // [4*maxIdx] rest anchor + wall flag readback (design §4.4)
        Vector4[] auxV;              // pre-allocated UV1 upload array (design §4.4: array overload)
        Vector2[] uv0V;              // Project2/SofaUnity-style planar UV0 from rest anchors
        Vector4[] tanV;              // tangent basis for Built-in Project2 normal/height detail
        int[]     idxSeq;            // [0,1,2,...] sequential triangle indices
        Vector3[] verts, norms;
        int3      dimsRt; float Lrt; float3 originRt, gridCenterRt;
        Vector3   projectUvMinRt, projectUvSizeRt;
        int       _gridEdgeCountRt;          // for the [DEBUG-CUT] raw-counter capacity readout
        LiverCudaCutter _cutter;
        int       _stepErrThrottle;
        Vector3[] _gridRestCorners;
        byte[]    _gridActiveMask;
        bool      _externalGridDeformationActive;
        bool      _externalUploadUnsupported;

        public bool IsInitialized => initialized && cutReady;
        public int GridCornerCount => _gridRestCorners != null ? _gridRestCorners.Length : 0;
        public bool ExternalGridDeformationActive => _externalGridDeformationActive;

        // Arrays are immutable after Start and are exposed only for Stage 3 embedding setup.
        public bool TryGetGridRestData(out Vector3[] corners, out byte[] activeMask)
        {
            corners = _gridRestCorners;
            activeMask = _gridActiveMask;
            return initialized && corners != null && activeMask != null;
        }

        public void SetExternalGridDeformationActive(bool active)
        {
            if (active && _externalUploadUnsupported)
                return;
            _externalGridDeformationActive = active;
        }

        /// <summary>
        /// Shows or hides only the CUDA/Dual-Contouring presentation. The CUDA simulation,
        /// cutter, and buffers remain initialized so Stage 3 can switch presentation modes
        /// without recreating them.
        /// </summary>
        public void SetPresentationVisible(bool visible)
        {
            presentationVisible = visible;
            if (rawMeshRenderer != null)
                rawMeshRenderer.enabled = visible;
            if (visualRenderer != null)
                visualRenderer.SetVisible(visible);
        }

        public bool UploadExternalCornerPositions(float[] positions)
        {
            if (!initialized || _externalUploadUnsupported || positions == null ||
                positions.Length != GridCornerCount * 3)
                return false;

            try
            {
                int rc = LCS_SetCornerPos(positions);
                if (rc == 0)
                    return true;
                Debug.LogError($"[Stage3Coupling] LCS_SetCornerPos failed (rc={rc}).", this);
            }
            catch (System.EntryPointNotFoundException)
            {
                _externalUploadUnsupported = true;
                _externalGridDeformationActive = false;
                Debug.LogError("[Stage3Coupling] The deployed LiverCudaSim.dll does not contain LCS_SetCornerPos. " +
                               "Close Unity, rebuild cuda_plugin, then deploy the new DLL to Assets/Plugins/x86_64.", this);
            }
            return false;
        }

        // Exact live reconstructed exterior, without the membrane shader's visual normal offset.
        public Mesh LiveOuterSurfaceMesh => visualRenderer != null ? visualRenderer.LiveOuterSurfaceMesh : null;

        // ── Diagnostics (press P in Play to dump split/gravity ground truth) ──────────────────────
        int       diagCornerCount;
        byte[]    diagActive, diagPinned;
        int[]     diagNbr;          // readback buffer [6*cc]
        int[]     diagNbrInit;      // initial nbrIdx [6*cc] (to count CUT-severed edges by axis)
        float[]   diagPos;          // readback buffer [3*cc]
        int[]     diagComp;         // union-find parent [cc]
        float     diagInitMeanY = float.NaN;
        int       _diagFrame;
        string _hudCoverage = "";

        // [DEBUG-CUT] (v4.1 P3): read the 16 GPU counters EVERY frame (the device window is one
        // chain — 1 Hz sampling would discard 59/60 frames and miss exactly the transients being
        // hunted, plan review R1-m5/R2-m2), aggregate host-side, print at 1 Hz or immediately when
        // an anomaly counter fires. Slots: 0=emaPairs 1=jump>L 2=jump>3L 3=maxDisp(f) 4=slotNew
        // 5=epochInval 6=dangleRefs 7=giantWall 8=giantMixed 9=giantSkin 10=maxEdge(f) 11=rim<3
        // 12=pinch1 13=all6severed 14=frozenNow 15=spare 16=rawCutPts 17=rawTriIdx.
        uint[]  _dbgFrame = new uint[18];
        ulong[] _dbgSum   = new ulong[18];   // event counters summed over the print window
        uint[]  _dbgMax   = new uint[18];    // gauges max'd over the print window
        float   _dbgAlpha;
        float   _dbgNextPrint, _dbgNextAnomaly;
        // dangleRefs(6) is a per-frame gauge like the giant-tri counts (the scan re-visits persistent
        // state every frame), so it is MAXed, not summed (final review V3-m1). tearMarks(15) is
        // CUMULATIVE on the device (tick-phase writer, excluded from the per-chain zeroing) -> max.
        static readonly int[] _dbgSumSlots = { 0, 1, 2, 4, 5, 14 };
        static readonly int[] _dbgMaxSlots = { 3, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17 };

        void Start()
        {
            // 1. Reused liver preprocess.
            string path = Path.Combine(Application.streamingAssetsPath, mshFileName);
            if (!File.Exists(path)) { Debug.LogError($"[LiverCuda] .msh not found: {path}"); return; }
            MshMesh mshMesh = MshLoader.LoadMsh(path);
            MshSurface surf = MshLoader.ExtractBoundary(mshMesh);
            MshLoader.Bounds(mshMesh, out float3 bmin, out float3 bmax);
            projectUvMinRt = (Vector3)bmin;
            projectUvSizeRt = (Vector3)(bmax - bmin);
            var meshLs = new MeshLevelSet(surf);
            GridFit.FitToBounds(bmin, bmax, targetLongAxisVoxels, marginVoxels, out dimsRt, out Lrt, out originRt);
            var ls = new SmoothedSdfGrid(meshLs, originRt, dimsRt, Lrt, bakeOversample, smoothIters);
            BackgroundGrid g = BackgroundGrid.Build(ls, dimsRt, Lrt, originRt);
            gridCenterRt = originRt + 0.5f * (float3)dimsRt * Lrt;

            _gridRestCorners = new Vector3[g.cornerCount];
            for (int c = 0; c < g.cornerCount; c++) _gridRestCorners[c] = (Vector3)g.cornerPos[c];
            _gridActiveMask = (byte[])g.cornerActive.Clone();

            if (g.surfaceEdges == null || g.surfaceEdges.Length == 0) { Debug.LogError("[LiverCuda] no surface edges. Aborting."); return; }
            LiverGrid.CountOccupancy(g.voxelOccupied, out bool anyIn, out bool anyOut);
            if (!anyIn)  { Debug.LogError("[LiverCuda] grid has NO tissue. Aborting."); return; }
            if (!anyOut) { Debug.LogError("[LiverCuda] grid has NO empty shell (raise marginVoxels). Aborting."); return; }

            int pinnedCount = pinDistributedAnchors
                ? LiverGrid.PinDistributed(g.cornerPos, g.cornerActive, g.cornerInside, g.pinned,
                                           Mathf.Max(4, distributedAnchorCount))
                : LiverGrid.PinCap(g.cornerPos, g.cornerActive, g.pinned, (float3)anchorAxis, anchorBand);
            if (pinnedCount == 0) { Debug.LogError("[LiverCuda] anchor pinned 0 corners. Aborting."); return; }
            if (massScale != 1f) for (int c = 0; c < g.cornerCount; c++) g.mass[c] *= massScale;

            int triCapacity = 6 * g.surfaceEdges.Length + 1024 + 2 * g.gridEdges.Length;
            maxIdx = 3 * triCapacity;

            // 2. Marshal + LCS_Init (static DC).
            var cornerPos = new float[3 * g.cornerCount];
            for (int c = 0; c < g.cornerCount; c++) { cornerPos[3*c+0]=g.cornerPos[c].x; cornerPos[3*c+1]=g.cornerPos[c].y; cornerPos[3*c+2]=g.cornerPos[c].z; }
            var isectI = new IsectInterop[Mathf.Max(1, g.isect.Length)];
            for (int i = 0; i < g.isect.Length; i++) { var e = g.isect[i]; isectI[i] = new IsectInterop { gx=e.globalRest.x, gy=e.globalRest.y, gz=e.globalRest.z, nx=e.normal.x, ny=e.normal.y, nz=e.normal.z, cornerA=e.cornerA, cornerB=e.cornerB, t=e.t, insideA=e.insideA }; }
            var seI = new GridEdgeInterop[Mathf.Max(1, g.surfaceEdges.Length)];
            for (int i = 0; i < g.surfaceEdges.Length; i++) { var se = g.surfaceEdges[i]; seI[i] = new GridEdgeInterop { axis=se.axis, bcX=se.baseCorner.x, bcY=se.baseCorner.y, bcZ=se.baseCorner.z, insideA=se.insideA }; }
            var rdesc = new ReconInitDesc { voxelCount=g.voxelCount, cornerCount=g.cornerCount, isectCount=g.isect.Length, surfaceEdgeCount=g.surfaceEdges.Length, triCapacity=triCapacity, dimsX=dimsRt.x, dimsY=dimsRt.y, dimsZ=dimsRt.z, L=Lrt, qefIters=qefIters };
            int rc = LCS_Init(ref rdesc, cornerPos, g.voxelCorner, g.voxelIsectOffset, g.voxelIsectCount, isectI, seI);
            if (rc != 0) { Debug.LogError($"[LiverCuda] LCS_Init failed (rc={rc}). Check the DLL / CUDA runtime."); return; }
            initialized = true;

            // 3. Physics.
            int cc = g.cornerCount;
            var pinnedInt = new int[cc]; var activeInt = new int[cc];
            for (int c = 0; c < cc; c++) { pinnedInt[c]=g.pinned[c]; activeInt[c]=g.cornerActive[c]; }
            var extForce = new float[3 * cc];
            for (int c = 0; c < cc; c++) extForce[3*c+1] = gravity * g.mass[c];
            var restNbr = new float[3 * 6 * cc];
            for (int s = 0; s < 6 * cc; s++) { restNbr[3*s+0]=g.restNbr[s].x; restNbr[3*s+1]=g.restNbr[s].y; restNbr[3*s+2]=g.restNbr[s].z; }
            var bendI = new BendPairInterop[Mathf.Max(1, g.bendPairs.Length)];
            for (int i = 0; i < g.bendPairs.Length; i++) { var bp = g.bendPairs[i]; bendI[i] = new BendPairInterop { i=bp.i, j=bp.j, k=bp.k, theta0=bp.theta0, alive=bp.alive }; }
            var pdesc = new PhysicsInitDesc { cornerCount=cc, bendPairCount=g.bendPairs.Length, ks=ks, cs=cs_damp, kb=kb, cb=cb, hInit=solverHInit, hMax=solverHMax, maxSubsteps=maxSubstepsPerFrame, alpha=globalDamp };
            rc = LCS_InitPhysics(ref pdesc, g.mass, pinnedInt, activeInt, extForce, g.nbrIdx, restNbr, bendI);
            if (rc != 0) { Debug.LogError($"[LiverCuda] LCS_InitPhysics failed (rc={rc})."); return; }

            // Diagnostic buffers (press P in Play to dump connected-components + gravity-motion ground truth).
            diagCornerCount = cc;
            diagActive = g.cornerActive; diagPinned = g.pinned;
            diagNbr = new int[6 * cc]; diagPos = new float[3 * cc]; diagComp = new int[cc];
            diagNbrInit = (int[])g.nbrIdx.Clone();   // snapshot the pre-cut graph for per-axis sever counting
            // 4. Cutter runtime.
            EnsureCutterComponent();
            _gridEdgeCountRt = g.gridEdges.Length;
            rc = _cutter.InitializeNative(g, dimsRt, Lrt, originRt, gridCenterRt, triCapacity, tearStretchRatio, anchorAxis);
            if (rc != 0) { Debug.LogError($"[LiverCuda] LiverCudaCutter.InitializeNative failed (rc={rc})."); return; }
            cutReady = true;

            // 6. Initial cut-aware rebuild + Mesh. The material is created FIRST (stage-2 review
            // S2-m4) so _liverMatActive is known before the first BuildOrUpdateMesh — the very
            // first uploaded mesh then already carries UV1.
            mat = CreateLiverMaterial();
            cutMat = CreateCutMaterial();
            SetupSmoothVisualRenderer();
            LCS_Finalize();
            AllocRenderArrays();
            BuildOrUpdateMesh(firstTime: true);

            // 7. Renderer + camera. LiverSurface (PhotorealisticLiver port) when enabled with all
            // textures assigned; else the legacy flat CudaSurface path (design §5: never a
            // black/pink screen).
            rawMeshRenderer = GetComponent<MeshRenderer>();
            if (rawMeshRenderer == null) rawMeshRenderer = gameObject.AddComponent<MeshRenderer>();
            rawMeshRenderer.material = mat;
            UpdateRawRendererVisibility();
            ApplyReferenceLighting();
            ApplyReferenceSceneRendering();
            SetupCamera();

            Debug.Log($"[LiverCuda] Stage 3 OK — grid {dimsRt} L={Lrt:F3} | pinned={pinnedCount}.");
        }

        void EnsureCutterComponent()
        {
            if (_cutter == null) _cutter = GetComponent<LiverCudaCutter>();
            if (_cutter == null) _cutter = gameObject.AddComponent<LiverCudaCutter>();
        }

        void Update()
        {
            if (!initialized || !cutReady || mesh == null) return;

            // a. Either retain the original CUDA physics path or let Stage 3 provide the current
            // lattice positions. Running both would double-drive the same corner buffer.
            int rc;
            if (_externalGridDeformationActive)
            {
                try { rc = LCS_ComputeRotations(); }
                catch (System.EntryPointNotFoundException)
                {
                    _externalUploadUnsupported = true;
                    _externalGridDeformationActive = false;
                    Debug.LogError("[Stage3Coupling] The deployed LiverCudaSim.dll is stale and lacks LCS_ComputeRotations.", this);
                    return;
                }
                if (rc != 0) { if ((_stepErrThrottle++ % 120) == 0) Debug.LogError($"[Stage3Coupling] LCS_ComputeRotations failed (rc={rc})."); return; }
            }
            else
            {
                rc = LCS_Step(physicsDt);
                if (rc != 0) { if ((_stepErrThrottle++ % 120) == 0) Debug.LogError($"[LiverCuda] LCS_Step failed (rc={rc})."); return; }
            }

            // b. Let the cutter advance its tool state and native pass.
            if (_cutter != null) _cutter.Step(RodSpanLength());

            // c. Rotations + cut-aware rebuild, then render. rc check (final review V1-m2): the
            // v4.1 chain can fail with -3300/-3301 (dbg memset / epoch roll) — surface it.
            rc = LCS_Finalize();
            if (rc != 0 && (_stepErrThrottle++ % 120) == 0) Debug.LogError($"[LiverCuda] LCS_Finalize failed (rc={rc}).");
            BuildOrUpdateMesh(firstTime: false);
            SyncSurfaceMembraneSettings();
            // d. Diagnostics. Auto gravity-motion log every ~1s; press P for a full split/gravity dump.
            PollCutDebug();
            if ((_diagFrame++ % 60) == 0) LogGravityMotion();
            if (Input.GetKeyDown(KeyCode.P)) DumpDiagnostics();
        }

        // [DEBUG-CUT] per-frame poll + host aggregation + gated print (v4.1 P3). One line answers:
        // EMA aliasing (jump>L / epoch), EMA lag (jumps==0 but maxDisp~L during fall), dangling FP
        // refs & giant tris (the strand/fan direct measurement), rim/pinch (0-g roughness), debris
        // freeze (all6/frozen), and counter starvation (raw vs capacity) — with the RUNTIME alpha.
        void PollCutDebug()
        {
            if (LCS_GetCutDebug(_dbgFrame, out _dbgAlpha) != 0) return;
            foreach (int s in _dbgSumSlots) _dbgSum[s] += _dbgFrame[s];
            foreach (int s in _dbgMaxSlots) if (_dbgFrame[s] > _dbgMax[s]) _dbgMax[s] = _dbgFrame[s];

            // Immediate print (throttled to 4 Hz) when an anomaly fires THIS frame; else 1 Hz summary.
            bool anomaly = _dbgFrame[1] > 0 || _dbgFrame[2] > 0 || _dbgFrame[6] > 0 ||
                           _dbgFrame[7] > 0 || _dbgFrame[8] > 0;
            float now = Time.unscaledTime;
            if (anomaly && now >= _dbgNextAnomaly)
            {
                _dbgNextAnomaly = now + 0.25f;
                Debug.LogWarning($"[DEBUG-CUT][ANOMALY] {FormatCutDebug(_dbgFrame, frameLocal: true)}");
            }
            if (now >= _dbgNextPrint)
            {
                _dbgNextPrint = now + 1f;
                var agg = new uint[18];
                foreach (int s in _dbgSumSlots) agg[s] = (uint)System.Math.Min(_dbgSum[s], uint.MaxValue);
                foreach (int s in _dbgMaxSlots) agg[s] = _dbgMax[s];
                Debug.Log($"[DEBUG-CUT] {FormatCutDebug(agg, frameLocal: false)}");
                System.Array.Clear(_dbgSum, 0, _dbgSum.Length);
                System.Array.Clear(_dbgMax, 0, _dbgMax.Length);
            }
        }

        string FormatCutDebug(uint[] d, bool frameLocal)
        {
            float maxDispL = ToF(d[3]) / Lrt, maxEdgeL = ToF(d[10]) / Lrt;
            int cutCap = Mathf.Max(1, 4 * _gridEdgeCountRt);
            return $"a={_dbgAlpha:F2} {(frameLocal ? "frame" : "1s-window")} | " +
                   $"ema(pairs={d[0]} jump>L={d[1]} jump>3L={d[2]} maxDisp={maxDispL:F2}L new={d[4]} epoch={d[5]}) | " +
                   $"tris(dangle={d[6]} giantWall={d[7]} giantMixed={d[8]} giantSkin={d[9]} maxEdge={maxEdgeL:F2}L) | " +
                   $"rim<3={d[11]} pinch1={d[12]} | debris(all6={d[13]} frozen+={d[14]}) tears={d[15]} | " +
                   $"raw(cutPts={d[16]}/{cutCap} triIdx={d[17]}/{maxIdx})";
        }
        static float ToF(uint u) { return System.BitConverter.ToSingle(System.BitConverter.GetBytes(u), 0); }

        // Mean Y of active corners over time — if gravity moves the tissue this DECREASES; if flat, the
        // liver is effectively in a vacuum (no net fall). Answers Bug 3.
        void LogGravityMotion()
        {
            if (diagPos == null || LCS_GetCornerPos(diagPos) != 0) return;
            // ANCHORED-COMPONENT extent (Bug-B fix): the rod span must track the tissue still hanging
            // from the anchor. Percentile trimming cannot exclude a SEVERED BIG piece (35% of corners
            // free-falling inflated the extent to 131.9 → an auto-sized rod would become absurd).
            // Union-find over the LIVE post-sever graph (~1 Hz, reuses the DumpDiagnostics buffers);
            // extent is measured over corners connected to ANY pinned corner only.
            bool haveComp = diagNbr != null && LCS_GetNbrIdx(diagNbr) == 0;
            if (haveComp)
            {
                for (int c = 0; c < diagCornerCount; c++) diagComp[c] = c;
                for (int c = 0; c < diagCornerCount; c++)
                {
                    if (diagActive[c] == 0) continue;
                    for (int s = 1; s < 6; s += 2)                        // +x,+y,+z: each edge once
                    {
                        int nb = diagNbr[6 * c + s];
                        if (nb >= 0 && diagActive[nb] != 0) Union(c, nb);
                    }
                }
            }
            var anchoredRoot = new System.Collections.Generic.HashSet<int>();
            if (haveComp)
                for (int c = 0; c < diagCornerCount; c++)
                    if (diagActive[c] != 0 && diagPinned[c] != 0) anchoredRoot.Add(Find(c));

            double sum = 0; int n = 0; float lo = float.PositiveInfinity, hi = float.NegativeInfinity;
            Vector3 mn = Vector3.positiveInfinity, mx = Vector3.negativeInfinity;
            for (int c = 0; c < diagCornerCount; c++)
            {
                if (diagActive[c] == 0) continue;
                var p = new Vector3(diagPos[3 * c + 0], diagPos[3 * c + 1], diagPos[3 * c + 2]);
                if (!haveComp || anchoredRoot.Contains(Find(c)))          // anchored tissue only
                {
                    mn = Vector3.Min(mn, p); mx = Vector3.Max(mx, p);
                }
                if (diagPinned[c] != 0) continue;                          // free-active for meanY only
                sum += p.y; n++;
                if (p.y < lo) lo = p.y; if (p.y > hi) hi = p.y;
            }
            if (mx.x >= mn.x)
            {
                Vector3 ext = mx - mn;
                _deformedMaxExt = Mathf.Max(ext.x, Mathf.Max(ext.y, ext.z));
            }
            if (n == 0) return;
            float mean = (float)(sum / n);
            if (float.IsNaN(diagInitMeanY)) diagInitMeanY = mean;
            Debug.Log($"[Diag/Gravity] free-active meanY={mean:F4} (Δ from start={mean - diagInitMeanY:+0.000;-0.000}) range[{lo:F3},{hi:F3}] n={n}");
            // Coverage is unconditional now (effRodLen = max(cutter rod length, anchored span)); keep a 1 Hz
            // info line + live HUD instead of the old stale warning (review wf_4a45e51c R3-m1/m2).
            _hudCoverage = _cutter != null
                ? _cutter.FormatCoverageHud(RodSpanLength(), _deformedMaxExt)
                : $"CUT rod={RodSpanLength():F1}u | anchored extent={_deformedMaxExt:F1}u | press P for cut diagnostics";
        }

        // Connected-components over the LIVE post-sever structural graph (active corners only).
        // Answers Bug 1: components==1 => cut did NOT disconnect; a component with 0 pinned => it should fall.
        void DumpDiagnostics()
        {
            if (diagNbr == null || LCS_GetNbrIdx(diagNbr) != 0) { Debug.LogWarning("[Diag] nbrIdx readback failed."); return; }
            if (diagPos == null || LCS_GetCornerPos(diagPos) != 0) { Debug.LogWarning("[Diag] cornerPos readback failed."); return; }
            int cc = diagCornerCount;

            // Union-find over edges (c -> nbr) where BOTH endpoints are active and nbr>=0 (severed = -1).
            for (int c = 0; c < cc; c++) diagComp[c] = c;
            int severed = 0, activeActiveEdges = 0;
            var cutAxis = new int[3];   // CUT-severed active-active slots by axis (0=X,1=Y,2=Z)
            for (int c = 0; c < cc; c++)
            {
                if (diagActive[c] == 0) continue;
                for (int s = 0; s < 6; s++)
                {
                    // Count edges that were a LIVE active-active link pre-cut and are now severed, by axis.
                    int initNb = (diagNbrInit != null) ? diagNbrInit[6 * c + s] : -1;
                    if (initNb >= 0 && diagActive[initNb] != 0 && diagNbr[6 * c + s] < 0) cutAxis[s / 2]++;

                    int nb = diagNbr[6 * c + s];
                    if (nb < 0) { severed++; continue; }
                    if (diagActive[nb] == 0) continue;
                    if (s == 1 || s == 3 || s == 5) activeActiveEdges++;   // count each edge once (+axis slots)
                    Union(c, nb);
                }
            }

            // Tally components: size, #pinned, meanY.
            var size = new System.Collections.Generic.Dictionary<int, int>();
            var pin  = new System.Collections.Generic.Dictionary<int, int>();
            var ysum = new System.Collections.Generic.Dictionary<int, double>();
            int activeN = 0;
            for (int c = 0; c < cc; c++)
            {
                if (diagActive[c] == 0) continue;
                activeN++;
                int r = Find(c);
                size.TryGetValue(r, out int sv); size[r] = sv + 1;
                pin.TryGetValue(r, out int pv); pin[r] = pv + (diagPinned[c] != 0 ? 1 : 0);
                ysum.TryGetValue(r, out double yv); ysum[r] = yv + diagPos[3 * c + 1];
            }

            int nComp = size.Count, freeComps = 0;
            var sb = new System.Text.StringBuilder();
            sb.Append($"[Diag/Split] ACTIVE corners={activeN} | COMPONENTS={nComp} | active-active edges={activeActiveEdges} | severed nbr-slots={severed}\n");
            // Axis histogram note (wf_65901910 B1): a PLANAR cut legitimately severs mostly the family
            // along its normal — dominance is EXPECTED, not a defect. The real incompleteness signal is
            // the dominant count vs the anchored cross-section's column count (~(ext/L)^2).
            int domCut = Mathf.Max(cutAxis[0], Mathf.Max(cutAxis[1], cutAxis[2])) / 2;
            float colNeed = (_deformedMaxExt > 0f) ? (_deformedMaxExt / Lrt) * (_deformedMaxExt / Lrt) * 0.5f : 0f;
            sb.Append($"   CUT-severed edges by axis: X={cutAxis[0] / 2} Y={cutAxis[1] / 2} Z={cutAxis[2] / 2}  " +
                      $"(dominant family = cut-plane normal, expected; dominant={domCut} vs ~cross-section need≈{colNeed:F0} => " +
                      $"{(colNeed > 0 && domCut < 0.7f * colNeed ? "cut looks INCOMPLETE — sweep further" : "count plausible for a full section")})\n");

            // BRIDGE CLASSIFIER v2 (wf_65901910 B3): classify against the PATH-WIDE swept-voxel ribbon,
            // not a single plane (single-plane was a false-negative on partial slits and a false-positive
            // on fallen pieces — forensics B1). interior-hole = un-severed pre-cut edge whose deformed
            // midpoint lies INSIDE the painted ribbon (the blade passed there but the edge survived =
            // real detection miss). plane-straddle-outside-ribbon = the legacy directional cue only.
            if (_cutter != null && _cutter.SweptVoxelCount > 0)
            {
                Vector3 n = _cutter.LastCutNormal.sqrMagnitude > 1e-8f ? _cutter.LastCutNormal.normalized : Vector3.up;
                Vector3 p0 = _cutter.LastCutPoint;
                int brBoundary = 0, brHole = 0;
                int[] fwd = { 1, 3, 5 };   // +X,+Y,+Z neighbor slots (each undirected edge once)
                for (int c = 0; c < cc; c++)
                {
                    if (diagActive[c] == 0) continue;
                    Vector3 rc = DefPos(c);
                    float sc = Vector3.Dot(rc - p0, n);
                    foreach (int s in fwd)
                    {
                        int initNb = (diagNbrInit != null) ? diagNbrInit[6 * c + s] : -1;
                        if (initNb < 0 || diagActive[initNb] == 0) continue;   // not a pre-cut active-active link
                        if (diagNbr[6 * c + s] < 0) continue;                  // already severed => not a bridge
                        Vector3 rn = DefPos(initNb);
                        float sn = Vector3.Dot(rn - p0, n);
                        if ((sc >= 0f) == (sn >= 0f)) continue;                // must STRADDLE the cut plane
                        Vector3 mid = 0.5f * (rc + rn);
                        if (_cutter.SweptContains(mid)) brHole++;              // swept + straddling + survived
                        else brBoundary++;                                     // straddling but never swept
                    }
                }
                sb.Append($"   BRIDGES v2 (ribbon-based): interior-hole={brHole} (blade passed, edge survived — real detection miss)  " +
                          $"unswept-straddle={brBoundary} (coverage/gesture gap — sweep deeper/wider)\n");
                _hudCoverage = _cutter.FormatBridgeHud(brHole, brBoundary, RodSpanLength(), _deformedMaxExt);
            }
            foreach (var kv in size)
            {
                int r = kv.Key; int sz = kv.Value; int np = pin[r]; float my = (float)(ysum[r] / sz);
                bool free = (np == 0);
                if (free) freeComps++;
                sb.Append($"   comp size={sz} pinned={np} meanY={my:F3} {(free ? "<-- FREE (should fall)" : "(anchored)")}\n");
            }
            sb.Append(nComp == 1
                ? "   => 1 component: the cut did NOT disconnect the graph (no piece can fall). Detection/sever coverage gap."
                : (freeComps == 0
                    ? "   => split, but EVERY component is pinned (both halves anchored). Geometry: pin spans both sides."
                    : $"   => split with {freeComps} FREE component(s): a piece SHOULD free-fall. If it doesn't, gravity/integration is the issue."));

            Debug.Log(sb.ToString());
        }

        int Find(int x) { while (diagComp[x] != x) { diagComp[x] = diagComp[diagComp[x]]; x = diagComp[x]; } return x; }
        void Union(int a, int b) { int ra = Find(a), rb = Find(b); if (ra != rb) diagComp[ra] = rb; }

        // DEFORMED (current world) position of a corner id — from the live cornerPos readback (diagPos).
        // The cut plane + swept region are world-space, so bridge straddle/coverage must be tested in this frame.
        Vector3 DefPos(int id) => new Vector3(diagPos[3 * id + 0], diagPos[3 * id + 1], diagPos[3 * id + 2]);

        // Live coverage HUD (wf_65901910 B3 minor): the 1 Hz console warning was consistently missed.
        void OnGUI()
        {
            if (!string.IsNullOrEmpty(_hudCoverage))
                GUI.Label(new Rect(10, 10, 900, 22), _hudCoverage);
            if (_cutter != null && !string.IsNullOrEmpty(_cutter.LastSweepStatus))
                GUI.Label(new Rect(10, 32, 900, 22), _cutter.LastSweepStatus);
        }

        // Auto-size the rod to span the whole tissue cross-section so the swept ribbon can mark a complete
        // separating band. Diagnosis wf_19bb05de G2 (major): sizing from the REST grid extent under-covers when
        // gravity stretches the tissue (observed 2.3x — the material image of a world sweep shrinks 1/lambda
        // along the stretched direction). Use the larger of rest extent and the cached DEFORMED extent.
        float _deformedMaxExt = -1f;   // deformed active-corner extent, cached by LogGravityMotion (~1 Hz)
        float RodSpanLength()
        {
            float maxExt = Mathf.Max(dimsRt.x, Mathf.Max(dimsRt.y, dimsRt.z)) * Lrt;
            if (_deformedMaxExt > maxExt) maxExt = _deformedMaxExt;
            return maxExt + 2f * Lrt;
        }

        // ── Render (non-indexed expanded soup) ──────────────────────────────────────────────────

        // Liver surface material (design §5): serialized shader reference first (build-safe),
        // Shader.Find fallback with a warning, legacy CudaSurface if unavailable/disabled.
        Material CreateLiverMaterial()
        {
            ResolveLiverRenderAssets();

            // Serialized-scene migration (render corrections wf_413cc887 + wf_7e66fcc2): scenes
            // saved with a RETRACTED default keep it serialized — migrate known-bad values only;
            // any other user value is kept. Current target 0.112 = the reference Project2_Liver2
            // tiling (2,2) over planar [0,1] UVs => ~2 tiles across / 17.8u.
            if (Mathf.Abs(triplanarScale - 0.07f) < 1e-4f || Mathf.Abs(triplanarScale - 0.55f) < 1e-4f
                || Mathf.Abs(triplanarScale - 0.112f) < 1e-4f)
            {
                Debug.Log($"[LiverCuda] triplanarScale {triplanarScale:F3} (retracted default) migrated to 0.83 (LiverTissueGPU convention: their 2.67u rest span x 5.5 / our 17.8u anchors).");
                triplanarScale = 0.83f;
            }
            if (useLiverSurfaceShader)
            {
                Shader lsh = liverSurfaceShader != null
                    ? liverSurfaceShader
                    : (Shader.Find("ReconGridDC/Project2LiverSurface") ?? Shader.Find("ReconGridDC/LiverSurface"));
                if (lsh == null)
                {
                    Debug.LogWarning("[LiverCuda] Project2LiverSurface/LiverSurface shader not found (assign the serialized field for player builds) — falling back to CudaSurface.");
                }
                else if (liverAlbedoTex == null || liverSpecTex == null || liverHeightTex == null)
                {
                    // Design §5/§8 (STRICT, stage-2 review S2-major): shader OR textures missing
                    // -> the CudaSurface fallback path, never a partially-textured LiverSurface.
                    Debug.LogWarning("[LiverCuda] LiverSurface textures unassigned — falling back to CudaSurface. " +
                                     "Assign liver2 / liver2_spec / liver2_height from Assets/ReconGridDC/Textures on the LiverCudaManager inspector.");
                }
                else
                {
                    if (liverSurfaceShader == null)
                        Debug.LogWarning("[LiverCuda] liver surface shader found via Shader.Find — assign the serialized 'liverSurfaceShader' field so player builds include it.");
                    // AUTO-CORRECT swapped texture slots by NAME (mirrors SoftBody.cs's by-path
                    // auto-load; a swapped inspector assignment rendered bump-as-albedo pink +
                    // liver2-as-normal garbage): bump/normal name -> _NormalMap, spec -> _SpecMap,
                    // remaining (liver2) -> _MainTex.
                    Texture2D texA = liverAlbedoTex, texN = liverNormalTex, texS = liverSpecTex, texH = liverHeightTex;
                    foreach (var t in new[] { liverAlbedoTex, liverNormalTex, liverSpecTex, liverHeightTex })
                    {
                        if (t == null) continue;
                        string n = t.name.ToLowerInvariant();
                        if (n.Contains("height"))                       texH = t;
                        else if (n.Contains("bump") || n.Contains("normal")) texN = t;
                        else if (n.Contains("spec"))                    texS = t;
                        else                                            texA = t;
                    }
                    if (texA != liverAlbedoTex || texN != liverNormalTex || texS != liverSpecTex || texH != liverHeightTex)
                        Debug.LogWarning("[LiverCuda] liver texture slots were swapped in the inspector; auto-corrected by name.");
                    liverAlbedoTex = texA; liverNormalTex = texN; liverSpecTex = texS; liverHeightTex = texH;
                    var m = new Material(lsh);
                    // Trilinear (review C1-m3): ~10 tiles on a curved surface + mips — bilinear
                    // mip transitions would band.
                    liverAlbedoTex.wrapMode = TextureWrapMode.Repeat; liverAlbedoTex.filterMode = FilterMode.Bilinear; m.SetTexture("_MainTex", liverAlbedoTex); m.SetTexture("_BaseMap", liverAlbedoTex);
                    if (liverNormalTex != null)
                    {
                        liverNormalTex.wrapMode = TextureWrapMode.Repeat;
                        liverNormalTex.filterMode = FilterMode.Bilinear;
                        m.SetTexture("_NormalMap", liverNormalTex);
                    }
                    liverSpecTex.wrapMode   = TextureWrapMode.Repeat; liverSpecTex.filterMode   = FilterMode.Bilinear; m.SetTexture("_SpecMap", liverSpecTex); m.SetTexture("_BumpMap", liverSpecTex); m.SetTexture("_SpecGlossMap", liverSpecTex);
                    liverHeightTex.wrapMode = TextureWrapMode.Repeat; liverHeightTex.filterMode = FilterMode.Bilinear; m.SetTexture("_HeightMap", liverHeightTex); m.SetTexture("_ParallaxMap", liverHeightTex);
                    foreach (string prop in new[] { "_MainTex", "_BaseMap", "_NormalMap", "_SpecMap", "_BumpMap", "_SpecGlossMap", "_HeightMap", "_ParallaxMap" })
                    {
                        m.SetTextureScale(prop, liverTextureTiling);
                        m.SetTextureOffset(prop, liverTextureOffset);
                    }
                    m.SetVector("_TextureTilingOffset", new Vector4(liverTextureTiling.x, liverTextureTiling.y, liverTextureOffset.x, liverTextureOffset.y));
                    m.SetFloat("_TriplanarScale", triplanarScale);   // DC-scale density (see field comment)
                    m.SetColor("_InteriorColor", interiorColor);
                    m.SetFloat("_TextureContrast", Mathf.Max(liverTextureContrast, 1.35f));
                    m.SetFloat("_TextureColorBlend", Mathf.Clamp01(liverTextureColorBlend));
                    m.SetFloat("_UvTextureWeight", Mathf.Clamp01(liverUvTextureWeight));
                    m.SetFloat("_AlbedoBrightness", Mathf.Min(liverAlbedoBrightness, 0.82f));
                    m.SetFloat("_TriplanarBlend", liverTriplanarBlend > 0f ? liverTriplanarBlend : 5f);
                    m.SetFloat("_ProceduralNormalStrength", Mathf.Min(liverProceduralNormalStrength, 0.05f));
                    m.SetFloat("_Wetness", Mathf.Min(liverWetness, 0.18f));
                    m.SetFloat("_MicroMottleStrength", Mathf.Min(liverMicroMottleStrength, 0.08f));
                    m.SetFloat("_MicroMottleScale", liverMicroMottleScale);
                    m.SetFloat("_VeinStrength", liverVeinStrength);
                    m.SetFloat("_VeinScale", liverVeinScale);
                    m.SetFloat("_DebugTextureOnly", liverDebugTextureOnly);
                    // LiverTissueGPU law: _Color is the TISSUE BASE (enters as +base*0.08 and the
                    // texture-strength lerp base), NOT Project2's 0.72 gray multiplier — SoftBody
                    // sets surfaceColor (0.62,0.18,0.10) here (SoftBody.cs:140/763).
                    Color tissueBase = new Color(0.62f, 0.18f, 0.10f, 1f);
                    m.SetColor("_Color", tissueBase);
                    m.SetColor("_BaseColor", tissueBase);
                    m.SetFloat("_TextureStrength", Mathf.Max(liverTextureStrength, 0.95f));
                    m.SetFloat("_Roughness", Mathf.Max(liverRoughness, 0.70f));
                    m.SetFloat("_SpecularStrength", Mathf.Min(liverSpecularStrength, 0.18f));
                    // F0 target (metallic 0.184 over the liver2 LINEAR mean) = (0.059,0.038,0.035);
                    // _SpecularColor is itself sRGB-converted at bind — exact-hit value solved by
                    // inverting lerp(0.04, srgb2lin(SC), 0.35) == target (verify wf_35f93c51 m2).
                    m.SetColor("_SpecularColor", liverSpecularColor);
                    m.SetColor("_WetSpecColor", liverSpecularColor);
                    m.SetFloat("_SpecMapStrength", 0.65f);
                    m.SetFloat("_NormalStrength", Mathf.Min(Mathf.Max(liverNormalStrength, 0.04f), 0.12f));
                    m.SetFloat("_FresnelStrength", Mathf.Min(liverFresnelStrength, 0.08f));
                    m.SetFloat("_FresnelPow", liverFresnelPow);
                    m.SetColor("_SSSColor", liverSSSColor);
                    m.SetFloat("_SSSStrength", Mathf.Min(liverSSSStrength, 0.06f));
                    m.SetFloat("_SSSDirect", Mathf.Min(liverSSSBacklight, 0.04f));
                    m.SetFloat("_SSSPower", liverSSSPower);
                    m.SetFloat("_SSSWrap", liverSSSWrap);
                    m.SetFloat("_Metallic", 0.184f);
                    m.SetFloat("_Smoothness", 0.939f);
                    m.SetFloat("_Glossiness", 0.939f);
                    m.SetFloat("_BumpScale", 1.03f);
                    m.SetFloat("_Parallax", 0.08f);
                    m.SetFloat("_CutSmoothness", 0.22f);
                    m.SetFloat("_Cull", 0f);
                    m.doubleSidedGI = true;
                    _liverMatActive = true;
                    return m;
                }
            }
            _liverMatActive = false;
            var sh = Shader.Find("ReconGridDC/CudaSurface") ?? Shader.Find("Standard") ?? Shader.Find("Diffuse");
            return new Material(sh);
        }

        Material CreateCutMaterial()
        {
            Shader sh = Shader.Find("ReconGridDC/LiverCutInterior") ?? Shader.Find("Standard") ?? Shader.Find("Diffuse");
            var m = new Material(sh) { name = "Runtime_DcLiverCutInterior" };
            SetMatColor(m, "_Color", interiorColor);
            SetMatColor(m, "_BaseColor", interiorColor);
            SetMatColor(m, "_WetSpecColor", new Color(0.95f, 0.78f, 0.68f, 1f));
            SetMatColor(m, "_SSSColor", liverSSSColor);
            SetMatFloat(m, "_Metallic", 0f);
            SetMatFloat(m, "_Smoothness", 0.58f);
            SetMatFloat(m, "_Glossiness", 0.58f);
            SetMatFloat(m, "_SpecularStrength", 0.65f);
            SetMatFloat(m, "_FresnelStrength", 0.22f);
            SetMatFloat(m, "_FresnelPow", 4.0f);
            SetMatFloat(m, "_Wetness", 0.75f);
            m.doubleSidedGI = true;
            return m;
        }

        void SetupSmoothVisualRenderer()
        {
            if (!useSmoothVisualRenderer || !_liverMatActive) return;
            visualRenderer = GetComponent<DcLiverVisualRenderer>();
            if (visualRenderer == null) visualRenderer = gameObject.AddComponent<DcLiverVisualRenderer>();
            membraneMat = CreateSurfaceMembraneMaterial();
            visualRenderer.Initialize(mat, cutMat, showSurfaceMembrane ? membraneMat : null);
        }

        Material CreateSurfaceMembraneMaterial()
        {
            Shader sh = surfaceMembraneShader != null
                ? surfaceMembraneShader
                : Shader.Find("ReconGridDC/LiverSurfaceMembrane");
            if (sh == null)
            {
                Debug.LogWarning("[LiverCuda] Surface membrane shader not found; membrane visualization is disabled.");
                return null;
            }

            var result = new Material(sh) { name = "Runtime_LiverSurfaceMembrane" };
            result.SetColor("_Color", surfaceMembraneColor);
            result.SetFloat("_Opacity", Mathf.Clamp01(surfaceMembraneOpacity));
            result.SetFloat("_NormalOffset", Mathf.Max(0f, surfaceMembraneOffset));
            result.SetFloat("_RimStrength", surfaceMembraneRimStrength);
            return result;
        }

        void SyncSurfaceMembraneSettings()
        {
            if (visualRenderer == null || membraneMat == null) return;

            membraneMat.SetColor("_Color", surfaceMembraneColor);
            membraneMat.SetFloat("_Opacity", Mathf.Clamp01(surfaceMembraneOpacity));
            membraneMat.SetFloat("_NormalOffset", Mathf.Max(0f, surfaceMembraneOffset));
            membraneMat.SetFloat("_RimStrength", surfaceMembraneRimStrength);
            visualRenderer.ConfigureMembrane(showSurfaceMembrane ? membraneMat : null);
        }

        void UpdateRawRendererVisibility()
        {
            if (rawMeshRenderer == null) return;
            bool showVisual = !presentationVisible ||
                              (useSmoothVisualRenderer && hideRawCudaSurfaceWhenVisualActive &&
                               visualRenderer != null && visualRenderer.HasVisibleGeometry);
            rawMeshRenderer.enabled = !showVisual;
        }

        static void SetMatColor(Material m, string prop, Color value)
        {
            if (m != null && m.HasProperty(prop)) m.SetColor(prop, value);
        }

        static void SetMatFloat(Material m, string prop, float value)
        {
            if (m != null && m.HasProperty(prop)) m.SetFloat(prop, value);
        }

        void ResolveLiverRenderAssets()
        {
#if UNITY_EDITOR
            Shader project2Shader = UnityEditor.AssetDatabase.LoadAssetAtPath<Shader>("Assets/ReconGridDC/Cuda/Project2LiverSurface.shader");
            if (surfaceMembraneShader == null)
                surfaceMembraneShader = UnityEditor.AssetDatabase.LoadAssetAtPath<Shader>("Assets/ReconGridDC/Cuda/LiverSurfaceMembrane.shader");
            if (useSmoothVisualRenderer && project2Shader != null)
                liverSurfaceShader = project2Shader;
            if (liverSurfaceShader == null)
                liverSurfaceShader = project2Shader;
            if (liverSurfaceShader == null)
                liverSurfaceShader = UnityEditor.AssetDatabase.LoadAssetAtPath<Shader>("Assets/ReconGridDC/Cuda/LiverSurface.shader");
            if (liverAlbedoTex == null)
                liverAlbedoTex = UnityEditor.AssetDatabase.LoadAssetAtPath<Texture2D>("Assets/ReconGridDC/Textures/liver2.png");
            if (liverNormalTex == null)
                liverNormalTex = UnityEditor.AssetDatabase.LoadAssetAtPath<Texture2D>("Assets/ReconGridDC/Textures/liver-texture-square_bump.png");
            if (liverSpecTex == null)
                liverSpecTex = UnityEditor.AssetDatabase.LoadAssetAtPath<Texture2D>("Assets/ReconGridDC/Textures/liver2_spec.png");
            if (liverHeightTex == null)
                liverHeightTex = UnityEditor.AssetDatabase.LoadAssetAtPath<Texture2D>("Assets/ReconGridDC/Textures/liver2_height.png");
#endif
        }

        Color EffectiveLiverColor()
        {
            Color c = liverColor;
            float maxChannel = Mathf.Max(c.r, Mathf.Max(c.g, c.b));
            float minChannel = Mathf.Min(c.r, Mathf.Min(c.g, c.b));
            float saturation = maxChannel <= 1e-5f ? 0f : (maxChannel - minChannel) / maxChannel;
            if (maxChannel > 0.82f || saturation < 0.35f || c.g > c.r * 0.40f || c.b > c.r * 0.32f)
                c = new Color(0.62f, 0.18f, 0.10f, liverColor.a);
            else
                c.a = liverColor.a;
            return c;
        }

        void AllocRenderArrays()
        {
            pos    = new float[3 * maxIdx];
            nrm    = new float[3 * maxIdx];
            verts  = new Vector3[maxIdx];
            norms  = new Vector3[maxIdx];
            uv0V   = new Vector2[maxIdx];
            tanV   = new Vector4[maxIdx];
            idxSeq = new int[maxIdx];
            for (int i = 0; i < maxIdx; i++) idxSeq[i] = i;
            // aux arrays are allocated in BuildOrUpdateMesh: the material is created BEFORE the
            // first Alloc+Build (S2-m4), so the alloc fires on the very first build — the lazy
            // check remains as a defensive no-op.
        }

        void BuildOrUpdateMesh(bool firstTime)
        {
            if (LCS_GetSurfaceCounts(out int N, out _) != 0) return;
            if (N < 0) N = 0;
            if (N > maxIdx) N = maxIdx;
            if (LCS_GetSurface(pos, nrm, null) != 0) return;

            // UV1 aux channel (render design §4.4): rest anchor + wall flag for the rest-position
            // triplanar shader. The material is created BEFORE the first build (S2-m4), so this
            // alloc fires on the first call; the null check is defensive only.
            // Stale-DLL guard: an old LiverCudaSim.dll without the export must degrade to the
            // no-UV1 look (uniform texel, f=0) instead of throwing every frame.
            bool haveAux = _liverMatActive && !_auxUnsupported;
            if (haveAux && (aux == null || auxV == null || uv0V == null || tanV == null))
            {
                aux = new float[4 * maxIdx];
                auxV = new Vector4[maxIdx];
                uv0V = new Vector2[maxIdx];
                tanV = new Vector4[maxIdx];
            }
            if (haveAux)
            {
                try { if (LCS_GetSurfaceAux(aux) != 0) haveAux = false; }
                catch (System.EntryPointNotFoundException)
                {
                    _auxUnsupported = true; haveAux = false;
                    Debug.LogError("[LiverCuda] LCS_GetSurfaceAux missing — the deployed LiverCudaSim.dll is STALE. " +
                                   "Close Unity and copy cuda_plugin/build/Release/LiverCudaSim.dll to Assets/Plugins/x86_64.");
                }
            }

            // Live bounds (design §4.4 / review A3-m1): fixed grid-sized bounds would frustum-cull
            // the whole mesh once a severed piece falls past ~2x the grid extent (shadow first).
            Vector3 bmn = new Vector3(float.MaxValue, float.MaxValue, float.MaxValue);
            Vector3 bmx = new Vector3(float.MinValue, float.MinValue, float.MinValue);
            for (int i = 0; i < N; i++)
            {
                var vtx = new Vector3(pos[3*i+0], pos[3*i+1], pos[3*i+2]);
                verts[i] = vtx;
                norms[i] = new Vector3(nrm[3*i+0], nrm[3*i+1], nrm[3*i+2]);
                if (haveAux)
                {
                    Vector3 rest = new Vector3(aux[4*i+0], aux[4*i+1], aux[4*i+2]);
                    auxV[i] = new Vector4(rest.x, rest.y, rest.z, aux[4*i+3]);
                    uv0V[i] = ProjectLiverUv(rest);
                }
                bmn = Vector3.Min(bmn, vtx); bmx = Vector3.Max(bmx, vtx);
            }
            if (firstTime)
            {
                mesh = new Mesh { indexFormat = IndexFormat.UInt32 };
                mesh.MarkDynamic();
                var mf = GetComponent<MeshFilter>();
                if (mf == null) mf = gameObject.AddComponent<MeshFilter>();
                mf.mesh = mesh;
            }
            mesh.Clear();
            mesh.SetVertices(verts, 0, N);
            mesh.SetNormals(norms, 0, N);
            if (haveAux)
            {
                BuildTangents(N);
                mesh.SetUVs(0, uv0V, 0, N);            // TEXCOORD0 planar UV, SofaUnity/Project2-style visible liver
                mesh.SetUVs(1, auxV, 0, N);            // TEXCOORD1 float4 rest anchor + cut wall flag
                mesh.SetTangents(tanV, 0, N);
            }
            mesh.SetIndices(idxSeq, 0, N, MeshTopology.Triangles, 0);   // non-indexed: triangles = [0..N)
            Bounds liveBounds;
            if (N > 0)
            {
                liveBounds = new Bounds(0.5f * (bmn + bmx), (bmx - bmn) + Vector3.one * Lrt);
                mesh.bounds = liveBounds;
            }
            else
            {
                liveBounds = new Bounds((Vector3)gridCenterRt, (Vector3)((float3)dimsRt * Lrt * 2f));
                mesh.bounds = liveBounds;
            }

            if (haveAux && useSmoothVisualRenderer && visualRenderer != null && _liverMatActive)
                visualRenderer.UpdateFromSoup(verts, norms, uv0V, auxV, N, liveBounds);
            else if (visualRenderer != null)
                visualRenderer.ClearVisuals();

            UpdateRawRendererVisibility();
        }

        void BuildTangents(int count)
        {
            for (int i = 0; i + 2 < count; i += 3)
            {
                Vector3 p0 = verts[i + 0], p1 = verts[i + 1], p2 = verts[i + 2];
                Vector2 w0 = uv0V[i + 0], w1 = uv0V[i + 1], w2 = uv0V[i + 2];

                Vector3 e1 = p1 - p0;
                Vector3 e2 = p2 - p0;
                Vector2 d1 = w1 - w0;
                Vector2 d2 = w2 - w0;
                float det = d1.x * d2.y - d1.y * d2.x;

                Vector3 t;
                if (Mathf.Abs(det) > 1e-8f)
                {
                    float r = 1f / det;
                    t = (e1 * d2.y - e2 * d1.y) * r;
                }
                else
                {
                    t = Vector3.zero;
                }

                Vector3 n = (norms[i + 0] + norms[i + 1] + norms[i + 2]) / 3f;
                if (n.sqrMagnitude < 1e-10f) n = Vector3.Cross(e1, e2);
                if (n.sqrMagnitude < 1e-10f) n = Vector3.up;
                n.Normalize();

                t = t - n * Vector3.Dot(n, t);
                if (!IsFinite(t) || t.sqrMagnitude < 1e-10f)
                    t = FallbackTangent(n);
                else
                    t.Normalize();

                Vector4 tangent = new Vector4(t.x, t.y, t.z, 1f);
                tanV[i + 0] = tangent;
                tanV[i + 1] = tangent;
                tanV[i + 2] = tangent;
            }

            for (int i = count - (count % 3); i < count; i++)
                tanV[i] = new Vector4(1, 0, 0, 1);
        }

        static Vector3 FallbackTangent(Vector3 n)
        {
            Vector3 axis = Mathf.Abs(n.y) < 0.9f ? Vector3.up : Vector3.right;
            Vector3 t = Vector3.Cross(axis, n);
            if (t.sqrMagnitude < 1e-10f) t = Vector3.right;
            else t.Normalize();
            return t;
        }

        static bool IsFinite(Vector3 v)
        {
            return !(float.IsNaN(v.x) || float.IsInfinity(v.x) ||
                     float.IsNaN(v.y) || float.IsInfinity(v.y) ||
                     float.IsNaN(v.z) || float.IsInfinity(v.z));
        }

        Vector2 ProjectLiverUv(Vector3 rest)
        {
            Vector3 min = projectUvMinRt;
            Vector3 size = projectUvSizeRt;
            if (size.sqrMagnitude < 1e-10f)
            {
                min = (Vector3)originRt;
                size = (Vector3)((float3)dimsRt * Lrt);
            }

            int uAxis, vAxis;
            if (size.x <= size.y && size.x <= size.z)
            {
                uAxis = 1;
                vAxis = 2;
            }
            else if (size.y <= size.x && size.y <= size.z)
            {
                uAxis = 0;
                vAxis = 2;
            }
            else
            {
                uAxis = 0;
                vAxis = 1;
            }

            float u = (Axis(rest, uAxis) - Axis(min, uAxis)) / Mathf.Max(Axis(size, uAxis), 1e-6f);
            float v = (Axis(rest, vAxis) - Axis(min, vAxis)) / Mathf.Max(Axis(size, vAxis), 1e-6f);
            return new Vector2(u, v);
        }

        static float Axis(Vector3 value, int axis)
        {
            return axis == 0 ? value.x : (axis == 1 ? value.y : value.z);
        }

        // Render correction v3 (P3-m5): the reference scene's directional light runs at
        // intensity 2.0 with the same color; ours ships 1.0 — align it (idempotent, logged).
        void ApplyReferenceLighting()
        {
            if (!matchReferenceLighting) return;
            foreach (var l in FindObjectsOfType<Light>())
            {
                if (l.type != LightType.Directional) continue;
                l.intensity = 2.0f;
                l.color = new Color(1f, 0.9568628f, 0.8392157f, 1f);
                l.shadows = LightShadows.Soft;
                l.shadowStrength = 0.82f;
                l.shadowBias = 0.02f;
                l.shadowNormalBias = 0.12f;
                Debug.Log("[LiverCuda] directional light -> intensity 2.0 + soft shadows (Project2 reference scene).");
                break;
            }
        }

        void ApplyReferenceSceneRendering()
        {
            if (matchReferenceSkybox)
            {
                Shader sky = Shader.Find("Skybox/Procedural");
                if (sky != null)
                {
                    if (skyboxMat == null) skyboxMat = new Material(sky) { name = "Runtime_Project2_Skybox" };
                    SetMatColor(skyboxMat, "_SkyTint", new Color(0.50f, 0.62f, 0.78f, 1f));
                    SetMatColor(skyboxMat, "_GroundColor", new Color(0.58f, 0.64f, 0.64f, 1f));
                    SetMatFloat(skyboxMat, "_Exposure", 1.0f);
                    SetMatFloat(skyboxMat, "_AtmosphereThickness", 0.75f);
                    RenderSettings.skybox = skyboxMat;
                    DynamicGI.UpdateEnvironment();
                }
            }

            if (matchReferenceFog)
            {
                RenderSettings.fog = true;
                RenderSettings.fogMode = FogMode.ExponentialSquared;
                RenderSettings.fogDensity = 0.012f;
                RenderSettings.fogColor = new Color(0.62f, 0.68f, 0.68f, 1f);
            }

            if (createReferenceGround) EnsureReferenceGround();
        }

        void EnsureReferenceGround()
        {
            if (referenceGround == null)
            {
                referenceGround = GameObject.Find("Project2ReferenceGround");
                if (referenceGround == null)
                {
                    referenceGround = GameObject.CreatePrimitive(PrimitiveType.Plane);
                    referenceGround.name = "Project2ReferenceGround";
                    referenceGroundCreated = true;
                    var col = referenceGround.GetComponent<Collider>();
                    if (col != null) Destroy(col);
                }
            }

            var mr = referenceGround.GetComponent<MeshRenderer>();
            if (mr == null) mr = referenceGround.AddComponent<MeshRenderer>();
            if (groundMat == null)
            {
                Shader sh = Shader.Find("Standard") ?? Shader.Find("Diffuse");
                groundMat = new Material(sh) { name = "Runtime_Project2_Ground" };
                SetMatColor(groundMat, "_Color", referenceGroundColor);
                SetMatFloat(groundMat, "_Metallic", 0f);
                SetMatFloat(groundMat, "_Smoothness", 0.15f);
                SetMatFloat(groundMat, "_Glossiness", 0.15f);
            }
            mr.sharedMaterial = groundMat;
            mr.shadowCastingMode = ShadowCastingMode.Off;
            mr.receiveShadows = true;

            Vector3 gridSize = (Vector3)((float3)dimsRt * Lrt);
            float planeSize = Mathf.Max(8f, Mathf.Max(gridSize.x, Mathf.Max(gridSize.y, gridSize.z)) * 1.9f);
            referenceGround.transform.position = new Vector3(gridCenterRt.x, originRt.y - Lrt * 0.35f, gridCenterRt.z);
            referenceGround.transform.rotation = Quaternion.identity;
            referenceGround.transform.localScale = new Vector3(planeSize / 10f, 1f, planeSize / 10f);
        }

        void SetupCamera()
        {
            if (!setupOrbitCamera) return;
            Camera cam = Camera.main;
            if (cam == null) { var go = new GameObject("Main Camera") { tag = "MainCamera" }; cam = go.AddComponent<Camera>(); go.AddComponent<AudioListener>(); }
            float radius = math.length((float3)dimsRt * Lrt) * 0.5f;
            float camDist = Mathf.Max(radius * 2.2f, 1f);
            cam.transform.position = (Vector3)gridCenterRt + new Vector3(0f, 0f, -camDist);
            cam.transform.LookAt((Vector3)gridCenterRt);
            cam.farClipPlane = Mathf.Max(cam.farClipPlane, camDist + radius * 4f);
            var orbit = cam.GetComponent<OrbitCamera>() ?? cam.gameObject.AddComponent<OrbitCamera>();
            orbit.target = (Vector3)gridCenterRt; orbit.distance = camDist;
        }

        void OnDestroy()
        {
            if (initialized) LCS_Shutdown();
            if (mesh != null) Destroy(mesh);
            if (mat  != null) Destroy(mat);
            if (cutMat != null) Destroy(cutMat);
            if (membraneMat != null) Destroy(membraneMat);
            if (groundMat != null) Destroy(groundMat);
            if (skyboxMat != null) Destroy(skyboxMat);
            if (referenceGroundCreated && referenceGround != null) Destroy(referenceGround);
        }
    }
}
