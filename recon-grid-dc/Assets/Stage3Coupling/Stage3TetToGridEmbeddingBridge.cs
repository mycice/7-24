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
        [Header("Stage 3 Switch")]
        [Tooltip("When enabled, tetrahedral XPBD positions drive the CUDA regular grid. Disabling restores the original CUDA grid physics on the next frame.")]
        public bool synchronizationEnabled = true;
        public Stage1TetSoftBodyController tetSoftBody;
        public LiverCudaManager cudaLiver;

        [Header("Embedding")]
        [Tooltip("Fits the active CUDA grid bounds into the tetrahedral rest bounds before creating host-tetrahedron mappings. Use this for the Phase 3 prototype because the two projects use different initial transforms.")]
        public bool fitRestBoundsAutomatically = true;
        [Tooltip("Spatial-hash cell size in tetrahedral rest coordinates. Zero selects an automatic value.")]
        [Min(0f)] public float hostSearchCellSize = 0f;
        [Range(0f, 0.05f)] public float insideTolerance = 0.001f;
        [Tooltip("Corners without a host tetrahedron retain their original CUDA grid rest position.")]
        [HideInInspector] public string unmappedPointStrategy = "Keep CUDA grid rest position";

        [Header("Readback")]
        [Tooltip("Stage 3 forces synchronous tetrahedron position readback while active. This makes the update order deterministic, but can reduce frame rate.")]
        public bool forceSynchronousTetReadback = true;
        [Tooltip("Logs mapping diagnostics whenever a new embedding is built.")]
        public bool logDiagnostics = true;

        [Header("Diagnostics (runtime)")]
        [SerializeField] int totalGridCorners;
        [SerializeField] int activeGridCorners;
        [SerializeField] int mappedActiveCorners;
        [SerializeField] int fallbackActiveCorners;
        [SerializeField] int inactiveRestCorners;
        [SerializeField] int globalFallbackCorners;
        [SerializeField] float activeMappingSuccessPercent;
        [SerializeField] float effectiveHostSearchCellSize;
        [SerializeField] float maxGridDisplacement;
        [SerializeField] float maxMappedLocalDisplacement;
        [SerializeField] int lastUploadFrame = -1;
        [SerializeField] string lastStatus = "Waiting for Stage 1 tetra body and CUDA liver.";

        readonly Dictionary<Vector3Int, List<int>> _tetHash = new Dictionary<Vector3Int, List<int>>();
        readonly List<int> _candidateScratch = new List<int>();
        int[] _hostTetByCorner;
        Vector4[] _weightsByCorner;
        Vector3[] _gridRestCorners;
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
            if (!Application.isPlaying)
                return;
            _embeddingReady = false;
        }

        void TrySubscribe()
        {
            if (_subscribedController == tetSoftBody || tetSoftBody == null)
                return;

            Unsubscribe();
            _subscribedController = tetSoftBody;
            _subscribedController.AfterSolverStep += OnTetSolverStep;
        }

        void Unsubscribe()
        {
            if (_subscribedController != null)
                _subscribedController.AfterSolverStep -= OnTetSolverStep;
            _subscribedController = null;
        }

        void OnTetSolverStep(TetMeshData data, XPBDSolverGPU solver, TetMeshVisualizer visualizer)
        {
            if (!synchronizationEnabled || !_embeddingReady || cudaLiver == null || data == null)
                return;

            UpdateGridPositions(data);
            if (cudaLiver.UploadExternalCornerPositions(_uploadPositions))
            {
                cudaLiver.SetExternalGridDeformationActive(true);
                lastUploadFrame = Time.frameCount;
                lastStatus = "Synchronized tetrahedral deformation to the CUDA grid.";
            }
            else
            {
                cudaLiver.SetExternalGridDeformationActive(false);
                lastStatus = "CUDA position upload failed; original CUDA grid physics remains active.";
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

            ApplyReadbackMode();
            ConfigureRestAlignment(data);
            BuildSpatialHash(data);
            BuildCornerMappings(data);
            _embeddingReady = true;
            lastStatus = "Embedding ready. Waiting for the next tetrahedral physics step.";

            if (logDiagnostics)
            {
                Debug.Log($"[Stage3Coupling] Tet-to-Grid embedding ready: active mapped " +
                          $"{mappedActiveCorners}/{activeGridCorners} ({activeMappingSuccessPercent:F1}%), " +
                          $"active fallback={fallbackActiveCorners}, inactive rest={inactiveRestCorners}, " +
                          $"hash cell={effectiveHostSearchCellSize:F4}.", this);
            }
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
            if (cudaLiver != null)
                cudaLiver.SetExternalGridDeformationActive(false);
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
            fallbackActiveCorners = 0;
            inactiveRestCorners = 0;
            globalFallbackCorners = 0;
            _hostTetByCorner = new int[totalGridCorners];
            _weightsByCorner = new Vector4[totalGridCorners];
            _uploadPositions = new float[totalGridCorners * 3];

            for (int c = 0; c < totalGridCorners; c++)
            {
                _hostTetByCorner[c] = -1;
                WriteUploadPosition(c, _gridRestCorners[c]);
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

        void UpdateGridPositions(TetMeshData data)
        {
            Vector3 globalGridDisplacement = TetDisplacementToGrid(
                ComputeCentroid(data.Positions) - _tetRestCentroid);
            maxGridDisplacement = 0f;
            maxMappedLocalDisplacement = 0f;
            globalFallbackCorners = 0;

            for (int c = 0; c < totalGridCorners; c++)
            {
                int tet = _hostTetByCorner[c];
                if (tet < 0)
                {
                    // These corners can be shared by Dual Contouring cells. Keeping them
                    // at rest while neighbouring corners move tears the reconstructed surface.
                    Vector3 fallback = _gridRestCorners[c] + globalGridDisplacement;
                    WriteUploadPosition(c, fallback);
                    globalFallbackCorners++;
                    maxGridDisplacement = Mathf.Max(maxGridDisplacement, globalGridDisplacement.magnitude);
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
                    Vector3 localGridDisplacement = TetDisplacementToGrid(currentTetPosition - restTetPosition);
                    WriteUploadPosition(c, _gridRestCorners[c] + localGridDisplacement);
                    maxMappedLocalDisplacement = Mathf.Max(maxMappedLocalDisplacement, localGridDisplacement.magnitude);
                    maxGridDisplacement = Mathf.Max(maxGridDisplacement, localGridDisplacement.magnitude);
                }
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

        Vector3 TetDisplacementToGrid(Vector3 tetDisplacement)
        {
            return new Vector3(
                tetDisplacement.x / _gridToTetScale.x,
                tetDisplacement.y / _gridToTetScale.y,
                tetDisplacement.z / _gridToTetScale.z);
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
