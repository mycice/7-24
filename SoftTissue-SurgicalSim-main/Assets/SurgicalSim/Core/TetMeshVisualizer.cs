using System.Collections.Generic;
using UnityEngine;

namespace SurgicalSim.Core
{
    [RequireComponent(typeof(MeshFilter))]
    [RequireComponent(typeof(MeshRenderer))]
    public class TetMeshVisualizer : MonoBehaviour
    {
        [Header("References")]
        public TetMeshLoader meshLoader;

        [Header("Visual Settings")]
        public Material surfaceMaterial;
        public bool showWireframe = false;

        [Header("Normal Smoothing")]
        public bool smoothSharedPositionNormals = true;
        public float normalMergeTolerance = 0.00075f;

        MeshFilter _meshFilter;
        MeshRenderer _meshRenderer;
        Mesh _surfaceMesh;
        Material _mainMaterial;
        HashSet<int> _normalMergeExcludedVertices;
        readonly Dictionary<Vector3Int, List<int>> _normalGroups = new Dictionary<Vector3Int, List<int>>();
        readonly List<List<int>> _normalGroupPool = new List<List<int>>();

        Vector3[] _surfaceVertices;
        Vector2[] _surfaceUvs;
        Vector3[] _surfaceRestPositions;
        TetMeshData _data;
        bool _initialized;
        bool _mainSurfaceVisible = true;

        void Awake()
        {
            _meshFilter = GetComponent<MeshFilter>();
            _meshRenderer = GetComponent<MeshRenderer>();
            if (meshLoader == null)
                meshLoader = GetComponent<TetMeshLoader>();
        }

        void Start()
        {
            if (meshLoader == null)
            {
                Debug.LogError("[TetMeshVisualizer] TetMeshLoader is required.");
                return;
            }

            meshLoader.OnMeshLoaded += Init;
            if (meshLoader.IsLoaded)
                Init(meshLoader.MeshData);
        }

        void OnDestroy()
        {
            if (meshLoader != null)
                meshLoader.OnMeshLoaded -= Init;
        }

        void Init(TetMeshData data)
        {
            _data = data;
            if (data.NumSurfaceTris == 0)
            {
                Debug.LogWarning("[TetMeshVisualizer] No surface triangles to render.");
                return;
            }

            _surfaceMesh = new Mesh
            {
                name = "TetSurface",
                indexFormat = data.NumParticles > 65535
                    ? UnityEngine.Rendering.IndexFormat.UInt32
                    : UnityEngine.Rendering.IndexFormat.UInt16
            };
            _surfaceMesh.MarkDynamic();

            _surfaceVertices = new Vector3[data.NumParticles];
            System.Array.Copy(data.Positions, _surfaceVertices, data.NumParticles);
            _surfaceMesh.vertices = _surfaceVertices;
            _surfaceMesh.triangles = data.SurfaceTriIds;
            _surfaceUvs = BuildPlanarUVs(data.Positions, data.NumParticles);
            _surfaceMesh.uv = _surfaceUvs;
            EnsureRestPositionUvs();
            RecalculateNormals();
            _surfaceMesh.RecalculateBounds();

            _meshFilter.mesh = _surfaceMesh;
            _surfaceMesh = _meshFilter.mesh;
            _surfaceMesh.MarkDynamic();

            if (surfaceMaterial != null)
                _meshRenderer.material = surfaceMaterial;
            else
            {
                Shader shader = Shader.Find("Universal Render Pipeline/Lit") ?? Shader.Find("Standard");
                _meshRenderer.material = new Material(shader) { color = new Color(0.85f, 0.25f, 0.15f) };
            }

            _mainMaterial = surfaceMaterial != null ? surfaceMaterial : _meshRenderer.sharedMaterial;
            ApplySurfaceMaterial();
            _initialized = true;
        }

        public void SetNormalMergeExcludedVertices(HashSet<int> excludedVertices)
        {
            _normalMergeExcludedVertices = excludedVertices;
        }

        public void SetSurfaceMaterial(Material material)
        {
            if (material == null) return;
            surfaceMaterial = material;
            _mainMaterial = material;
            ApplySurfaceMaterial();
        }

        public void SetMainSurfaceVisible(bool visible)
        {
            _mainSurfaceVisible = visible;
            ApplySurfaceMaterial();
        }

        public void Refresh()
        {
            if (!_initialized || _data == null || _surfaceMesh == null) return;
            EnsureVertexCapacity();
            System.Array.Copy(_data.Positions, _surfaceVertices, _data.NumParticles);
            _surfaceMesh.SetVertices(_surfaceVertices, 0, _data.NumParticles);
            RecalculateNormals();
            _surfaceMesh.RecalculateBounds();
        }

        public void RebuildTopology(int[] triangles)
        {
            if (!_initialized || _data == null || _surfaceMesh == null) return;
            EnsureVertexCapacity();
            System.Array.Copy(_data.Positions, _surfaceVertices, _data.NumParticles);
            _surfaceMesh.Clear();
            _surfaceMesh.indexFormat = _data.NumParticles > 65535
                ? UnityEngine.Rendering.IndexFormat.UInt32
                : UnityEngine.Rendering.IndexFormat.UInt16;
            _surfaceMesh.SetVertices(_surfaceVertices, 0, _data.NumParticles);
            _surfaceMesh.uv = _surfaceUvs;
            EnsureRestPositionUvs();
            _surfaceMesh.triangles = triangles;
            RecalculateNormals();
            _surfaceMesh.RecalculateBounds();
        }

        void EnsureVertexCapacity()
        {
            if (_surfaceVertices != null && _surfaceVertices.Length >= _data.NumParticles) return;
            _surfaceVertices = new Vector3[_data.NumParticles];
            _surfaceUvs = BuildPlanarUVs(_data.Positions, _data.NumParticles);
            _surfaceMesh.indexFormat = _data.NumParticles > 65535
                ? UnityEngine.Rendering.IndexFormat.UInt32
                : UnityEngine.Rendering.IndexFormat.UInt16;
        }

        void ApplySurfaceMaterial()
        {
            if (_meshRenderer == null) return;
            _meshRenderer.enabled = _mainSurfaceVisible;
            if (_mainSurfaceVisible && _mainMaterial != null)
                _meshRenderer.sharedMaterial = _mainMaterial;
        }

        void RecalculateNormals()
        {
            _surfaceMesh.RecalculateNormals();
            if (!smoothSharedPositionNormals || normalMergeTolerance <= 0f) return;

            Vector3[] normals = _surfaceMesh.normals;
            int count = Mathf.Min(_data.NumParticles, normals.Length);
            float inverseTolerance = 1f / Mathf.Max(1e-7f, normalMergeTolerance);
            ClearNormalGroups();
            for (int i = 0; i < count; i++)
            {
                if (_normalMergeExcludedVertices != null && _normalMergeExcludedVertices.Contains(i))
                    continue;
                Vector3Int key = NormalMergeKey(_surfaceVertices[i], inverseTolerance);
                if (!_normalGroups.TryGetValue(key, out List<int> group))
                {
                    group = RentNormalGroup();
                    _normalGroups.Add(key, group);
                }
                group.Add(i);
            }

            foreach (KeyValuePair<Vector3Int, List<int>> pair in _normalGroups)
            {
                Vector3 merged = Vector3.zero;
                foreach (int index in pair.Value)
                    merged += normals[index];
                if (merged.sqrMagnitude < 1e-10f) continue;
                merged.Normalize();
                foreach (int index in pair.Value)
                    normals[index] = merged;
            }
            ClearNormalGroups();
            _surfaceMesh.normals = normals;
        }

        List<int> RentNormalGroup()
        {
            int last = _normalGroupPool.Count - 1;
            if (last < 0) return new List<int>(2);
            List<int> group = _normalGroupPool[last];
            _normalGroupPool.RemoveAt(last);
            return group;
        }

        void ClearNormalGroups()
        {
            foreach (KeyValuePair<Vector3Int, List<int>> pair in _normalGroups)
            {
                pair.Value.Clear();
                _normalGroupPool.Add(pair.Value);
            }
            _normalGroups.Clear();
        }

        void EnsureRestPositionUvs()
        {
            if (_surfaceMesh == null || _data == null) return;
            int count = _surfaceMesh.vertexCount;
            if (_surfaceRestPositions == null || _surfaceRestPositions.Length != count)
            {
                _surfaceRestPositions = new Vector3[count];
                for (int i = 0; i < count; i++)
                    _surfaceRestPositions[i] = _data.RestPositions[i];
            }
            _surfaceMesh.SetUVs(1, new List<Vector3>(_surfaceRestPositions));
        }

        static Vector2[] BuildPlanarUVs(Vector3[] positions, int count)
        {
            Vector2[] uvs = new Vector2[count];
            if (positions == null || count == 0) return uvs;
            for (int i = 0; i < count; i++)
                uvs[i] = new Vector2(positions[i].x / 0.12f, positions[i].z / 0.12f);
            return uvs;
        }

        static Vector3Int NormalMergeKey(Vector3 position, float inverseTolerance)
        {
            return new Vector3Int(
                Mathf.RoundToInt(position.x * inverseTolerance),
                Mathf.RoundToInt(position.y * inverseTolerance),
                Mathf.RoundToInt(position.z * inverseTolerance));
        }

        void OnDrawGizmos()
        {
            if (!showWireframe || _data == null || !_initialized) return;
            Gizmos.color = new Color(0f, 1f, 0f, 0.3f);
            int drawn = 0;
            for (int t = 0; t < _data.NumTets && drawn < 500; t++)
            {
                if (!_data.TetActive[t]) continue;
                int offset = t * 4;
                Vector3 p0 = _data.Positions[_data.TetIds[offset]];
                Vector3 p1 = _data.Positions[_data.TetIds[offset + 1]];
                Vector3 p2 = _data.Positions[_data.TetIds[offset + 2]];
                Vector3 p3 = _data.Positions[_data.TetIds[offset + 3]];
                Gizmos.DrawLine(p0, p1); Gizmos.DrawLine(p0, p2); Gizmos.DrawLine(p0, p3);
                Gizmos.DrawLine(p1, p2); Gizmos.DrawLine(p1, p3); Gizmos.DrawLine(p2, p3);
                drawn++;
            }
        }
    }
}
