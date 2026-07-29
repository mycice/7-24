using System;
using System.Collections.Generic;
using ReconGridDC.Cuda;
using ReconGridDC.Stage1TetPhysics;
using ReconGridDC.Stage1TetPhysics.Core;
using ReconGridDC.Stage1TetPhysics.Physics;
using UnityEngine;

namespace ReconGridDC.Stage3Coupling
{
    /// <summary>
    /// One-way Stage 3 prototype: current XPBD tetrahedron positions drive the existing
    /// ReconGridDC lattice. CUDA cutting and Dual Contouring remain the grid-side systems.
    /// </summary>
    [DefaultExecutionOrder(200)]
    [DisallowMultipleComponent]
    public sealed class Stage3TetToGridEmbeddingBridge : MonoBehaviour
    {
        public enum OrganPresentationMode
        {
            DetailedCudaCuttableOrgan,
            CoarseTetPhysicsOrgan
        }

        [Header("Stage 3 Switch")]
        [Tooltip("When enabled, tetrahedral XPBD positions drive the CUDA regular grid. Disabling restores the original CUDA grid physics on the next frame.")]
        public bool synchronizationEnabled = true;
        public Stage1TetSoftBodyController tetSoftBody;
        public LiverCudaManager cudaLiver;

        [Header("Embedding")]
        [Tooltip("Fits the active CUDA grid bounds into the tetrahedral rest bounds before creating host-tetrahedron mappings and aligning the CUDA visual organ to the physical organ.")]
        public bool fitRestBoundsAutomatically = true;
        [Tooltip("Spatial-hash cell size in tetrahedral rest coordinates. Zero selects an automatic value.")]
        [Min(0f)] public float hostSearchCellSize = 0f;
        [Range(0f, 0.05f)] public float insideTolerance = 0.001f;
        [Tooltip("Active CUDA corners outside the tetrahedral volume are projected to their nearest tetrahedron surface during initialization. Only a point that cannot obtain a valid tetrahedron uses global body displacement.")]
        [HideInInspector] public string unmappedPointStrategy = "Project active corners to nearest tetrahedron surface";
        [Tooltip("Applies cached tetrahedral local deformation to the CUDA grid. Exact interior embeddings and projected surface fallbacks use the same continuous displacement field.")]
        public bool enableLocalTetDeformation = true;
        [Tooltip("Legacy diagnostic threshold retained for scene compatibility. Local deformation is now governed by valid exact or projected mappings, not this percentage.")]
        [Range(0f, 100f)] public float minimumMappingCoverageForLocalDeformation = 95f;

        [Header("Presentation")]
        [Tooltip("Detailed CUDA mode preserves cutting and the existing Dual Contouring surface. Coarse tetrahedral mode shows the direct collision mesh for faster, clearer grasp/contact evaluation, but does not display CUDA cutting.")]
        public OrganPresentationMode organPresentation = OrganPresentationMode.DetailedCudaCuttableOrgan;
        [Tooltip("Shows the coarse tetrahedral mesh used for collision and grasping. Disable this to present the aligned CUDA organ as the single visible organ.")]
        public bool showPhysicsProxy = false;

        [Header("Cutting Scale Adaptation")]
        [Tooltip("Makes CUDA cut-point reconstruction and cutting tolerances use the regular grid after it has been fitted to the tetrahedral organ. Choose this before entering Play; do not toggle after a cut has already been created.")]
        public bool adaptCudaCuttingMetric = true;

        [Header("Gravity-Safe Cut Surface")]
        [Tooltip("Adds a small render-only separation to CUDA cut feature points when the cut normal is aligned with gravity. This does not move tetrahedral physics vertices or change gripper collision and grasping.")]
        public bool stabilizeCutSurfaceAgainstGravity = true;
        [Tooltip("Minimum additional cut-surface gap as a fraction of the fitted CUDA voxel length. Horizontal cuts receive the full value; vertical cuts receive none.")]
        [Range(0f, 0.5f)] public float gravitySafeGapInVoxels = 0.12f;
        [Tooltip("Exponent applied to abs(dot(cutNormal, gravityDirection)). Values above 1 restrict stabilization more strongly to near-horizontal cut planes.")]
        [Range(0.5f, 4f)] public float gravityAlignmentExponent = 1.5f;

        [Header("Readback")]
        [Tooltip("Stage 3 forces synchronous tetrahedron position readback while active. This makes the update order deterministic, but can reduce frame rate.")]
        public bool forceSynchronousTetReadback = true;
        [Tooltip("Default phase-4 path: evaluate the existing host-tet/barycentric map in CudaOrganContext. Falls back to the original CPU mapping when disabled or unavailable.")]
        public bool preferCudaTetToGrid = true;
        [Tooltip("Logs mapping diagnostics whenever a new embedding is built.")]
        public bool logDiagnostics = true;

        [Header("Diagnostics (runtime)")]
        [SerializeField] int totalGridCorners;
        [SerializeField] int activeGridCorners;
        [SerializeField] int mappedActiveCorners;
        [SerializeField] int projectedActiveCorners;
        [SerializeField] int fallbackActiveCorners;
        [SerializeField] int inactiveRestCorners;
        [SerializeField] int globalFallbackCorners;
        [SerializeField] float activeMappingSuccessPercent;
        [SerializeField] float effectiveHostSearchCellSize;
        [SerializeField] float maxGridDisplacement;
        [SerializeField] float maxMappedLocalDisplacement;
        [SerializeField] bool localTetDeformationApplied;
        [SerializeField] float originalCutVoxelLength;
        [SerializeField] Vector3 alignedRestEdgeLengths;
        [SerializeField] float effectiveCutVoxelLength;
        [SerializeField] float conservativeCutStepLength;
        [SerializeField] float cutMetricScale = 1f;
        [SerializeField] string cutMetricStatus = "Waiting for embedding.";
        [SerializeField] float gravitySafeGapWorld;
        [SerializeField] string gravitySafeCutStatus = "Waiting for embedding.";
        [SerializeField] int lastUploadFrame = -1;
        [SerializeField] string lastStatus = "Waiting for Stage 1 tetra body and CUDA liver.";
        [SerializeField] bool cudaTetToGridActive;

        readonly Dictionary<Vector3Int, List<int>> _tetHash = new Dictionary<Vector3Int, List<int>>();
        readonly List<int> _candidateScratch = new List<int>();
        readonly HashSet<int> _candidateSet = new HashSet<int>();
        int[] _hostTetByCorner;
        Vector4[] _weightsByCorner;
        Vector3[] _gridRestCorners;
        Vector3[] _alignedGridRestCorners;
        byte[] _gridActiveMask;
        float[] _uploadPositions;
        Vector3 _gridCenter;
        Vector3 _tetCenter;
        Vector3 _tetRestCentroid;
        Vector3 _gridToTetScale = Vector3.one;
        bool _embeddingReady;
        bool _savedAsyncReadback;
        bool _readbackOverrideApplied;
        Stage1TetSoftBodyController _subscribedController;

        void OnEnable()
        {
            TrySubscribe();
        }

        void Start()
        {
            TrySubscribe();
        }

        void FixedUpdate()
        {
            TrySubscribe();
            if (!synchronizationEnabled)
            {
                DisableExternalMode();
                return;
            }

            if (!_embeddingReady)
                TryBuildEmbedding();
        }

        void OnDisable()
        {
            Unsubscribe();
            RestoreReadbackMode();
            DisableExternalMode();
        }

        void OnDestroy()
        {
            Unsubscribe();
            RestoreReadbackMode();
            DisableExternalMode();
        }

        void OnValidate()
        {
            insideTolerance = Mathf.Max(0f, insideTolerance);
            minimumMappingCoverageForLocalDeformation = Mathf.Clamp(minimumMappingCoverageForLocalDeformation, 0f, 100f);
            gravitySafeGapInVoxels = Mathf.Clamp(gravitySafeGapInVoxels, 0f, 0.5f);
            gravityAlignmentExponent = Mathf.Clamp(gravityAlignmentExponent, 0.5f, 4f);
            if (!Application.isPlaying)
            {
                ApplyPresentation();
                return;
            }
            _embeddingReady = false;
            ApplyPresentation();
        }

        void TrySubscribe()
        {
            if (_subscribedController == tetSoftBody || tetSoftBody == null)
                return;

            Unsubscribe();
            _subscribedController = tetSoftBody;
            _subscribedController.AfterSolverStep += OnTetSolverStep;
            _subscribedController.AfterCudaSolverStep += OnCudaSolverStep;
        }

        void Unsubscribe()
        {
            if (_subscribedController != null)
                _subscribedController.AfterSolverStep -= OnTetSolverStep;
            if (_subscribedController != null)
                _subscribedController.AfterCudaSolverStep -= OnCudaSolverStep;
            _subscribedController = null;
        }

        void OnTetSolverStep(TetMeshData data, XPBDSolverGPU solver, TetMeshVisualizer visualizer)
        {
            if (!synchronizationEnabled || !_embeddingReady || cudaLiver == null || data == null)
                return;

            ApplyPresentation();
            if (organPresentation == OrganPresentationMode.CoarseTetPhysicsOrgan)
            {
                RestoreReadbackMode();
                cudaLiver.SetExternalGridDeformationActive(false);
                lastStatus = "Showing the direct coarse tetrahedral physics organ; CUDA cutting display is paused.";
                return;
            }
            if (cudaTetToGridActive)
                return;
            ApplyReadbackMode();
            UpdateGridPositions(data);
            if (cudaLiver.UploadExternalCornerPositions(_uploadPositions))
            {
                cudaLiver.SetExternalGridDeformationActive(true);
                lastUploadFrame = Time.frameCount;
                lastStatus = localTetDeformationApplied
                    ? "Synchronized aligned local tetrahedral deformation to the CUDA grid."
                    : "Synchronized stable tetrahedral-body displacement to the CUDA grid.";
            }
            else
            {
                cudaLiver.SetExternalGridDeformationActive(false);
                lastStatus = "CUDA position upload failed; original CUDA grid physics remains active.";
            }
        }

        void OnCudaSolverStep()
        {
            if (!synchronizationEnabled || !_embeddingReady || !cudaTetToGridActive || cudaLiver == null)
                return;
            if (organPresentation != OrganPresentationMode.DetailedCudaCuttableOrgan)
            {
                tetSoftBody.GetComponent<CudaOrganContextBridge>().SetGpuResidentTetToGridFrameActive(false);
                return;
            }
            if (!cudaLiver.ShouldPublishExpensiveGpuWorkThisFrame())
            {
                lastStatus = "Idle GPU schedule: XPBD advanced while Tet-to-Grid and surface publication retained the previous GPU frame.";
                return;
            }
            if (tetSoftBody.GetComponent<CudaOrganContextBridge>().UpdateCudaTetToGrid(enableLocalTetDeformation))
            {
                tetSoftBody.GetComponent<CudaOrganContextBridge>().SetGpuResidentTetToGridFrameActive(true);
                cudaLiver.SetExternalGridDeformationActive(true);
                lastUploadFrame = Time.frameCount;
                lastStatus = "CUDA Tet-to-Grid updated the shared CUDA grid without tetrahedron or grid position readback.";
            }
            else
            {
                cudaTetToGridActive = false;
                tetSoftBody.GetComponent<CudaOrganContextBridge>().SetGpuResidentTetToGridFrameActive(false);
                lastStatus = "CUDA Tet-to-Grid update failed; CPU Stage 3 synchronization will be used on the next solver step.";
            }
        }

        void TryBuildEmbedding()
        {
            if (tetSoftBody == null || cudaLiver == null || !tetSoftBody.IsReady || !cudaLiver.IsInitialized)
            {
                lastStatus = "Waiting for both Stage 1 tetra physics and the CUDA liver to initialize.";
                return;
            }

            if (!cudaLiver.TryGetGridRestData(out _gridRestCorners, out _gridActiveMask) ||
                _gridRestCorners == null || _gridActiveMask == null)
            {
                lastStatus = "CUDA grid rest data is unavailable.";
                return;
            }

            TetMeshData data = tetSoftBody.MeshData;
            if (data == null || data.RestPositions == null || data.TetIds == null || data.NumTets == 0)
            {
                lastStatus = "Tetrahedral rest data is unavailable.";
                return;
            }

            if (organPresentation == OrganPresentationMode.DetailedCudaCuttableOrgan)
                ApplyReadbackMode();
            else
                RestoreReadbackMode();
            ConfigureRestAlignment(data);
            ApplyPresentation();
            BuildSpatialHash(data);
            BuildCornerMappings(data);
            ConfigureCudaTetToGrid();
            _embeddingReady = true;
            lastStatus = "Embedding ready. Waiting for the next tetrahedral physics step.";

            if (logDiagnostics)
            {
                Debug.Log($"[Stage3Coupling] Tet-to-Grid embedding ready: exact active " +
                          $"{mappedActiveCorners}/{activeGridCorners} ({activeMappingSuccessPercent:F1}%), " +
                          $"projected active={projectedActiveCorners}, global fallback={fallbackActiveCorners}, inactive rest={inactiveRestCorners}, " +
                          $"hash cell={effectiveHostSearchCellSize:F4}.", this);
            }
        }

        void ConfigureCudaTetToGrid()
        {
            cudaTetToGridActive = false;
            CudaOrganContextBridge bridge = tetSoftBody != null ? tetSoftBody.GetComponent<CudaOrganContextBridge>() : null;
            if (!preferCudaTetToGrid || bridge == null || !bridge.CanUseCudaTetToGrid)
                return;
            cudaTetToGridActive = bridge.ConfigureCudaTetToGrid(_hostTetByCorner, _weightsByCorner,
                _alignedGridRestCorners, _gridActiveMask, _tetRestCentroid);
            bridge.SetGpuResidentTetToGridFrameActive(cudaTetToGridActive &&
                                                       organPresentation == OrganPresentationMode.DetailedCudaCuttableOrgan);
            if (cudaTetToGridActive)
                RestoreReadbackMode();
        }

        void ApplyReadbackMode()
        {
            if (!forceSynchronousTetReadback || _readbackOverrideApplied || tetSoftBody == null)
                return;
            _savedAsyncReadback = tetSoftBody.useAsyncPositionReadback;
            tetSoftBody.useAsyncPositionReadback = false;
            _readbackOverrideApplied = true;
        }

        void RestoreReadbackMode()
        {
            if (!_readbackOverrideApplied || tetSoftBody == null)
                return;
            tetSoftBody.useAsyncPositionReadback = _savedAsyncReadback;
            _readbackOverrideApplied = false;
        }

        void DisableExternalMode()
        {
            if (tetSoftBody != null)
            {
                CudaOrganContextBridge bridge = tetSoftBody.GetComponent<CudaOrganContextBridge>();
                if (bridge != null)
                    bridge.SetGpuResidentTetToGridFrameActive(false);
            }
            if (cudaLiver != null)
            {
                cudaLiver.SetExternalGridDeformationActive(false);
                cudaLiver.ClearStage3CutRestMetric();
                cudaLiver.ConfigureGravitySafeCutSurface(false, Vector3.zero, 0f, gravityAlignmentExponent);
            }
        }

        void ConfigureRestAlignment(TetMeshData data)
        {
            Bounds gridBounds = ComputeGridActiveBounds(_gridRestCorners, _gridActiveMask);
            Bounds tetBounds = ComputeBounds(data.RestPositions);
            _gridCenter = gridBounds.center;
            _tetCenter = tetBounds.center;
            _tetRestCentroid = ComputeCentroid(data.RestPositions);
            _gridToTetScale = Vector3.one;
            if (fitRestBoundsAutomatically)
            {
                _gridToTetScale = new Vector3(
                    SafeScale(tetBounds.size.x, gridBounds.size.x),
                    SafeScale(tetBounds.size.y, gridBounds.size.y),
                    SafeScale(tetBounds.size.z, gridBounds.size.z));
            }

            _alignedGridRestCorners = new Vector3[_gridRestCorners.Length];
            for (int c = 0; c < _gridRestCorners.Length; c++)
                _alignedGridRestCorners[c] = GridToTetRest(_gridRestCorners[c]);

            ConfigureCuttingMetric();
        }

        void ConfigureCuttingMetric()
        {
            originalCutVoxelLength = cudaLiver != null ? cudaLiver.GridVoxelLength : 0f;
            alignedRestEdgeLengths = new Vector3(
                originalCutVoxelLength * Mathf.Abs(_gridToTetScale.x),
                originalCutVoxelLength * Mathf.Abs(_gridToTetScale.y),
                originalCutVoxelLength * Mathf.Abs(_gridToTetScale.z));
            effectiveCutVoxelLength = (alignedRestEdgeLengths.x + alignedRestEdgeLengths.y + alignedRestEdgeLengths.z) / 3f;
            conservativeCutStepLength = Mathf.Min(alignedRestEdgeLengths.x,
                Mathf.Min(alignedRestEdgeLengths.y, alignedRestEdgeLengths.z));
            cutMetricScale = originalCutVoxelLength > 1e-6f ? effectiveCutVoxelLength / originalCutVoxelLength : 1f;

            if (!adaptCudaCuttingMetric)
            {
                cudaLiver.ClearStage3CutRestMetric();
                cutMetricStatus = "Disabled; original CUDA cutting metric is active.";
                ConfigureGravitySafeCutSurface();
                return;
            }

            var alignedRest = new float[_alignedGridRestCorners.Length * 3];
            for (int c = 0; c < _alignedGridRestCorners.Length; c++)
            {
                Vector3 p = _alignedGridRestCorners[c];
                alignedRest[3 * c] = p.x;
                alignedRest[3 * c + 1] = p.y;
                alignedRest[3 * c + 2] = p.z;
            }

            bool configured = cudaLiver.ConfigureStage3CutRestMetric(alignedRest, alignedRestEdgeLengths);
            cutMetricStatus = configured
                ? "Active: CUDA cutting uses anisotropic Stage 3 voxel extents and aligned rest edges."
                : "Failed: original CUDA cutting metric remains active.";
            ConfigureGravitySafeCutSurface();
        }

        void ConfigureGravitySafeCutSurface()
        {
            gravitySafeGapWorld = Mathf.Max(0f, gravitySafeGapInVoxels * effectiveCutVoxelLength);
            Vector3 gravity = tetSoftBody != null && tetSoftBody.enableGravity
                ? new Vector3(0f, tetSoftBody.gravityY, 0f)
                : Vector3.zero;
            bool active = stabilizeCutSurfaceAgainstGravity && gravity.sqrMagnitude > 1e-8f && gravitySafeGapWorld > 0f;
            bool configured = cudaLiver != null && cudaLiver.ConfigureGravitySafeCutSurface(
                active, gravity, gravitySafeGapWorld, gravityAlignmentExponent);
            gravitySafeCutStatus = configured
                ? (active
                    ? "Active: render-only cut gap scales with cut-normal/gravity alignment."
                    : "Inactive: gravity is disabled or stabilization is switched off.")
                : "Failed: deployed CUDA plugin does not support gravity-safe cut surfaces.";
        }

        void ApplyPresentation()
        {
            if (tetSoftBody != null && tetSoftBody.Visualizer != null)
                tetSoftBody.Visualizer.SetMainSurfaceVisible(
                    organPresentation == OrganPresentationMode.CoarseTetPhysicsOrgan || showPhysicsProxy);
            if (cudaLiver != null)
                cudaLiver.SetPresentationVisible(organPresentation == OrganPresentationMode.DetailedCudaCuttableOrgan);
        }

        void BuildSpatialHash(TetMeshData data)
        {
            Bounds tetBounds = ComputeBounds(data.RestPositions);
            float automatic = Mathf.Max(0.001f, Mathf.Max(tetBounds.size.x, Mathf.Max(tetBounds.size.y, tetBounds.size.z)) / 20f);
            effectiveHostSearchCellSize = hostSearchCellSize > 0f ? hostSearchCellSize : automatic;
            _tetHash.Clear();
            for (int t = 0; t < data.NumTets; t++)
            {
                if (data.TetActive != null && !data.TetActive[t])
                    continue;
                GetTetBounds(data, t, out Vector3 min, out Vector3 max);
                Vector3Int minCell = ToCell(min);
                Vector3Int maxCell = ToCell(max);
                for (int z = minCell.z; z <= maxCell.z; z++)
                for (int y = minCell.y; y <= maxCell.y; y++)
                for (int x = minCell.x; x <= maxCell.x; x++)
                {
                    Vector3Int cell = new Vector3Int(x, y, z);
                    if (!_tetHash.TryGetValue(cell, out List<int> list))
                    {
                        list = new List<int>();
                        _tetHash.Add(cell, list);
                    }
                    list.Add(t);
                }
            }
        }

        void BuildCornerMappings(TetMeshData data)
        {
            totalGridCorners = _gridRestCorners.Length;
            activeGridCorners = 0;
            mappedActiveCorners = 0;
            projectedActiveCorners = 0;
            fallbackActiveCorners = 0;
            inactiveRestCorners = 0;
            globalFallbackCorners = 0;
            _hostTetByCorner = new int[totalGridCorners];
            _weightsByCorner = new Vector4[totalGridCorners];
            _uploadPositions = new float[totalGridCorners * 3];

            for (int c = 0; c < totalGridCorners; c++)
            {
                _hostTetByCorner[c] = -1;
                WriteUploadPosition(c, _alignedGridRestCorners[c]);
                if (_gridActiveMask[c] == 0)
                {
                    inactiveRestCorners++;
                    continue;
                }

                activeGridCorners++;
                Vector3 tetPoint = GridToTetRest(_gridRestCorners[c]);
                if (TryFindHostTet(data, tetPoint, out int tetIndex, out Vector4 weights))
                {
                    _hostTetByCorner[c] = tetIndex;
                    _weightsByCorner[c] = weights;
                    mappedActiveCorners++;
                }
                else if (TryFindProjectedHostTet(data, tetPoint, out tetIndex, out weights))
                {
                    _hostTetByCorner[c] = tetIndex;
                    _weightsByCorner[c] = weights;
                    projectedActiveCorners++;
                }
                else
                {
                    fallbackActiveCorners++;
                }
            }
            activeMappingSuccessPercent = activeGridCorners > 0
                ? 100f * mappedActiveCorners / activeGridCorners
                : 0f;
        }

        bool TryFindHostTet(TetMeshData data, Vector3 point, out int hostTet, out Vector4 weights)
        {
            Vector3Int cell = ToCell(point);
            for (int radius = 0; radius <= 1; radius++)
            {
                _candidateScratch.Clear();
                for (int z = cell.z - radius; z <= cell.z + radius; z++)
                for (int y = cell.y - radius; y <= cell.y + radius; y++)
                for (int x = cell.x - radius; x <= cell.x + radius; x++)
                {
                    if (_tetHash.TryGetValue(new Vector3Int(x, y, z), out List<int> candidates))
                        _candidateScratch.AddRange(candidates);
                }

                for (int i = 0; i < _candidateScratch.Count; i++)
                {
                    int t = _candidateScratch[i];
                    if (TryBarycentric(data, t, point, out weights) && IsInside(weights))
                    {
                        hostTet = t;
                        return true;
                    }
                }
            }
            hostTet = -1;
            weights = default;
            return false;
        }

        bool TryFindProjectedHostTet(TetMeshData data, Vector3 point, out int hostTet, out Vector4 weights)
        {
            hostTet = -1;
            weights = default;
            float bestDistanceSq = float.PositiveInfinity;
            Vector3Int cell = ToCell(point);

            // Most grid points outside the proxy lie immediately next to its surface. Search
            // locally first; only rare outliers require the complete active-tetrahedron pass.
            for (int radius = 0; radius <= 4; radius++)
            {
                CollectCandidates(cell, radius);
                EvaluateProjectedCandidates(data, point, ref hostTet, ref weights, ref bestDistanceSq);
                if (hostTet >= 0)
                    return true;
            }

            for (int t = 0; t < data.NumTets; t++)
            {
                if (data.TetActive != null && !data.TetActive[t])
                    continue;
                EvaluateProjectedTet(data, t, point, ref hostTet, ref weights, ref bestDistanceSq);
            }
            return hostTet >= 0;
        }

        void CollectCandidates(Vector3Int cell, int radius)
        {
            _candidateScratch.Clear();
            _candidateSet.Clear();
            for (int z = cell.z - radius; z <= cell.z + radius; z++)
            for (int y = cell.y - radius; y <= cell.y + radius; y++)
            for (int x = cell.x - radius; x <= cell.x + radius; x++)
            {
                if (!_tetHash.TryGetValue(new Vector3Int(x, y, z), out List<int> candidates))
                    continue;
                for (int i = 0; i < candidates.Count; i++)
                {
                    if (_candidateSet.Add(candidates[i]))
                        _candidateScratch.Add(candidates[i]);
                }
            }
        }

        void EvaluateProjectedCandidates(TetMeshData data, Vector3 point, ref int bestTet, ref Vector4 bestWeights, ref float bestDistanceSq)
        {
            for (int i = 0; i < _candidateScratch.Count; i++)
                EvaluateProjectedTet(data, _candidateScratch[i], point, ref bestTet, ref bestWeights, ref bestDistanceSq);
        }

        void EvaluateProjectedTet(TetMeshData data, int tet, Vector3 point, ref int bestTet, ref Vector4 bestWeights, ref float bestDistanceSq)
        {
            if (!TryBarycentric(data, tet, point, out Vector4 pointWeights))
                return;
            if (IsInside(pointWeights))
            {
                bestTet = tet;
                bestWeights = pointWeights;
                bestDistanceSq = 0f;
                return;
            }

            int b = tet * 4;
            Vector3 p0 = data.RestPositions[data.TetIds[b]];
            Vector3 p1 = data.RestPositions[data.TetIds[b + 1]];
            Vector3 p2 = data.RestPositions[data.TetIds[b + 2]];
            Vector3 p3 = data.RestPositions[data.TetIds[b + 3]];
            TestProjectedFace(point, p0, p1, p2, new Vector4(1f, 0f, 0f, 0f), new Vector4(0f, 1f, 0f, 0f), new Vector4(0f, 0f, 1f, 0f), tet, ref bestTet, ref bestWeights, ref bestDistanceSq);
            TestProjectedFace(point, p0, p1, p3, new Vector4(1f, 0f, 0f, 0f), new Vector4(0f, 1f, 0f, 0f), new Vector4(0f, 0f, 0f, 1f), tet, ref bestTet, ref bestWeights, ref bestDistanceSq);
            TestProjectedFace(point, p0, p2, p3, new Vector4(1f, 0f, 0f, 0f), new Vector4(0f, 0f, 1f, 0f), new Vector4(0f, 0f, 0f, 1f), tet, ref bestTet, ref bestWeights, ref bestDistanceSq);
            TestProjectedFace(point, p1, p2, p3, new Vector4(0f, 1f, 0f, 0f), new Vector4(0f, 0f, 1f, 0f), new Vector4(0f, 0f, 0f, 1f), tet, ref bestTet, ref bestWeights, ref bestDistanceSq);
        }

        static void TestProjectedFace(Vector3 point, Vector3 a, Vector3 b, Vector3 c, Vector4 wa, Vector4 wb, Vector4 wc, int tet, ref int bestTet, ref Vector4 bestWeights, ref float bestDistanceSq)
        {
            ClosestPointOnTriangle(point, a, b, c, out Vector3 closest, out Vector3 barycentric);
            float distanceSq = (point - closest).sqrMagnitude;
            if (distanceSq >= bestDistanceSq)
                return;
            bestDistanceSq = distanceSq;
            bestTet = tet;
            bestWeights = wa * barycentric.x + wb * barycentric.y + wc * barycentric.z;
        }

        static void ClosestPointOnTriangle(Vector3 p, Vector3 a, Vector3 b, Vector3 c, out Vector3 closest, out Vector3 barycentric)
        {
            Vector3 ab = b - a;
            Vector3 ac = c - a;
            Vector3 ap = p - a;
            float d1 = Vector3.Dot(ab, ap);
            float d2 = Vector3.Dot(ac, ap);
            if (d1 <= 0f && d2 <= 0f) { closest = a; barycentric = new Vector3(1f, 0f, 0f); return; }

            Vector3 bp = p - b;
            float d3 = Vector3.Dot(ab, bp);
            float d4 = Vector3.Dot(ac, bp);
            if (d3 >= 0f && d4 <= d3) { closest = b; barycentric = new Vector3(0f, 1f, 0f); return; }

            float vc = d1 * d4 - d3 * d2;
            if (vc <= 0f && d1 >= 0f && d3 <= 0f)
            {
                float v = d1 / (d1 - d3);
                closest = a + v * ab;
                barycentric = new Vector3(1f - v, v, 0f);
                return;
            }

            Vector3 cp = p - c;
            float d5 = Vector3.Dot(ab, cp);
            float d6 = Vector3.Dot(ac, cp);
            if (d6 >= 0f && d5 <= d6) { closest = c; barycentric = new Vector3(0f, 0f, 1f); return; }

            float vb = d5 * d2 - d1 * d6;
            if (vb <= 0f && d2 >= 0f && d6 <= 0f)
            {
                float w = d2 / (d2 - d6);
                closest = a + w * ac;
                barycentric = new Vector3(1f - w, 0f, w);
                return;
            }

            float va = d3 * d6 - d5 * d4;
            if (va <= 0f && (d4 - d3) >= 0f && (d5 - d6) >= 0f)
            {
                float w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
                closest = b + w * (c - b);
                barycentric = new Vector3(0f, 1f - w, w);
                return;
            }

            float denominator = 1f / (va + vb + vc);
            float faceV = vb * denominator;
            float faceW = vc * denominator;
            closest = a + ab * faceV + ac * faceW;
            barycentric = new Vector3(1f - faceV - faceW, faceV, faceW);
        }

        void UpdateGridPositions(TetMeshData data)
        {
            Vector3 globalTetDisplacement = ComputeCentroid(data.Positions) - _tetRestCentroid;
            localTetDeformationApplied = enableLocalTetDeformation;
            maxGridDisplacement = 0f;
            maxMappedLocalDisplacement = 0f;
            globalFallbackCorners = 0;

            for (int c = 0; c < totalGridCorners; c++)
            {
                int tet = _hostTetByCorner[c];
                Vector3 targetPosition = _alignedGridRestCorners[c] + globalTetDisplacement;
                if (tet < 0)
                {
                    globalFallbackCorners++;
                }
                else
                {
                    int b = tet * 4;
                    Vector4 w = _weightsByCorner[c];
                    Vector3 currentTetPosition = data.Positions[data.TetIds[b]] * w.x +
                                                 data.Positions[data.TetIds[b + 1]] * w.y +
                                                 data.Positions[data.TetIds[b + 2]] * w.z +
                                                 data.Positions[data.TetIds[b + 3]] * w.w;
                    Vector3 restTetPosition = data.RestPositions[data.TetIds[b]] * w.x +
                                              data.RestPositions[data.TetIds[b + 1]] * w.y +
                                              data.RestPositions[data.TetIds[b + 2]] * w.z +
                                              data.RestPositions[data.TetIds[b + 3]] * w.w;
                    Vector3 localTetDisplacement = currentTetPosition - restTetPosition;
                    maxMappedLocalDisplacement = Mathf.Max(maxMappedLocalDisplacement, localTetDisplacement.magnitude);
                    if (localTetDeformationApplied)
                        targetPosition = _alignedGridRestCorners[c] + localTetDisplacement;
                }

                WriteUploadPosition(c, targetPosition);
                maxGridDisplacement = Mathf.Max(maxGridDisplacement, (targetPosition - _alignedGridRestCorners[c]).magnitude);
            }
        }

        bool TryBarycentric(TetMeshData data, int tet, Vector3 p, out Vector4 w)
        {
            int b = tet * 4;
            Vector3 a = data.RestPositions[data.TetIds[b]];
            Vector3 b1 = data.RestPositions[data.TetIds[b + 1]];
            Vector3 c = data.RestPositions[data.TetIds[b + 2]];
            Vector3 d = data.RestPositions[data.TetIds[b + 3]];
            float det = Vector3.Dot(Vector3.Cross(b1 - a, c - a), d - a);
            if (Mathf.Abs(det) < 1e-10f)
            {
                w = default;
                return false;
            }
            Vector3 q = p - a;
            float w1 = Vector3.Dot(Vector3.Cross(q, c - a), d - a) / det;
            float w2 = Vector3.Dot(Vector3.Cross(b1 - a, q), d - a) / det;
            float w3 = Vector3.Dot(Vector3.Cross(b1 - a, c - a), q) / det;
            w = new Vector4(1f - w1 - w2 - w3, w1, w2, w3);
            return true;
        }

        bool IsInside(Vector4 w)
        {
            return w.x >= -insideTolerance && w.y >= -insideTolerance &&
                   w.z >= -insideTolerance && w.w >= -insideTolerance &&
                   w.x <= 1f + insideTolerance && w.y <= 1f + insideTolerance &&
                   w.z <= 1f + insideTolerance && w.w <= 1f + insideTolerance;
        }

        Vector3 GridToTetRest(Vector3 p)
        {
            Vector3 delta = p - _gridCenter;
            return _tetCenter + Vector3.Scale(delta, _gridToTetScale);
        }

        Vector3Int ToCell(Vector3 p)
        {
            return new Vector3Int(
                Mathf.FloorToInt(p.x / effectiveHostSearchCellSize),
                Mathf.FloorToInt(p.y / effectiveHostSearchCellSize),
                Mathf.FloorToInt(p.z / effectiveHostSearchCellSize));
        }

        static float SafeScale(float numerator, float denominator)
        {
            return denominator > 1e-5f ? numerator / denominator : 1f;
        }

        static Bounds ComputeBounds(Vector3[] points)
        {
            Bounds bounds = new Bounds(points[0], Vector3.zero);
            for (int i = 1; i < points.Length; i++) bounds.Encapsulate(points[i]);
            return bounds;
        }

        static Vector3 ComputeCentroid(Vector3[] points)
        {
            Vector3 sum = Vector3.zero;
            for (int i = 0; i < points.Length; i++) sum += points[i];
            return sum / Mathf.Max(1, points.Length);
        }

        static Bounds ComputeGridActiveBounds(Vector3[] points, byte[] active)
        {
            bool found = false;
            Bounds bounds = new Bounds(Vector3.zero, Vector3.zero);
            for (int i = 0; i < points.Length; i++)
            {
                if (active[i] == 0) continue;
                if (!found) { bounds = new Bounds(points[i], Vector3.zero); found = true; }
                else bounds.Encapsulate(points[i]);
            }
            return found ? bounds : ComputeBounds(points);
        }

        static void GetTetBounds(TetMeshData data, int tet, out Vector3 min, out Vector3 max)
        {
            int b = tet * 4;
            min = max = data.RestPositions[data.TetIds[b]];
            for (int i = 1; i < 4; i++)
            {
                Vector3 p = data.RestPositions[data.TetIds[b + i]];
                min = Vector3.Min(min, p);
                max = Vector3.Max(max, p);
            }
        }

        void WriteUploadPosition(int corner, Vector3 p)
        {
            int b = corner * 3;
            _uploadPositions[b] = p.x;
            _uploadPositions[b + 1] = p.y;
            _uploadPositions[b + 2] = p.z;
        }
    }
}
