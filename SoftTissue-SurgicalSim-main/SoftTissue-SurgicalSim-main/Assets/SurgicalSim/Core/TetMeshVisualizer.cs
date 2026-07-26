// TetMeshVisualizer.cs
// 把四面體網格的邊界表面渲染成 Unity Mesh
// 網格頂點每幀跟隨 TetMeshData.Positions 更新（XPBD 求解後調用 Refresh）
// Phase 1: 靜態顯示（無物理）
// Phase 2+ : 動態更新（物理求解後調用 Refresh()）

using UnityEngine;
using System.Collections.Generic;
using SurgicalSim.Core;

namespace SurgicalSim.Core
{
    [RequireComponent(typeof(MeshFilter))]
    [RequireComponent(typeof(MeshRenderer))]
    public class TetMeshVisualizer : MonoBehaviour
    {
        // ── Inspector 配置 ──────────────────────────────────
        [Header("引用")]
        [Tooltip("四面體網格加載器（同 GameObject 或手動指定）")]
        public TetMeshLoader meshLoader;

        [Header("視覺配置")]
        [Tooltip("表面材質（留空則使用默認 URP Lit）")]
        public Material surfaceMaterial;

        [Tooltip("Material used by the cut-surface submesh. Leave empty to create a simple two-sided pink material.")]
        public Material cutSurfaceMaterial;

        public Color cutSurfaceColor = new Color(0.85f, 0.45f, 0.45f, 1f);

        [Tooltip("是否顯示四面體邊框（Debug 用，性能消耗大）")]
        public bool showWireframe = false;

        [Header("Visual Smoothing")]
        [Tooltip("Average normals for vertices that share the same position, including split cut-surface pairs.")]
        public bool smoothSharedPositionNormals = true;

        [Tooltip("Position tolerance for visual normal merging.")]
        public float normalMergeTolerance = 0.00075f;

        [Range(0f, 90f)]
        [Tooltip("Only merge same-position normals whose angle is below this value. Keeps cut rims sharp instead of smoothing across the split.")]
        public float normalMergeMaxAngleDeg = 35f;

        [Tooltip("Smooth only the interior vertices of the cut submesh with cut-triangle area-weighted normals.")]
        public bool smoothCutSurfaceNormals = true;

        [Tooltip("Position tolerance for cut-only normal merging.")]
        public float cutNormalMergeTolerance = 0.0015f;

        [Range(0f, 90f)]
        [Tooltip("Max angle for cut-only normal merging. Rim and exterior vertices stay hard.")]
        public float cutNormalMergeMaxAngleDeg = 75f;

        // ── 私有狀態 ─────────────────────────────────────────
        MeshFilter   _meshFilter;
        MeshRenderer _meshRenderer;
        Mesh         _surfaceMesh;
        Material     _mainMaterial;
        Material     _runtimeCutSurfaceMaterial;
        Material     _invisibleSurfaceMaterial;
        HashSet<int> _normalMergeExcludedVertices;
        bool         _mainSurfaceVisible = true;

        // 頂點緩存（避免每幀 GC）
        Vector3[] _surfaceVerts;
        Vector3[] _surfaceNormals;
        Vector2[] _surfaceUVs;
        Vector3[] _surfaceRestPositions;
        int[] _cachedSurfaceTriangles;
        int[] _cachedCutTriangles;
        bool[] _cutSurfaceVertexScratch;
        bool[] _cutVertexScratch;
        bool[] _cutBoundaryVertexScratch;
        Vector3[] _cutAccumScratch;
        readonly Dictionary<Vector3Int, List<int>> _sharedNormalGroups = new Dictionary<Vector3Int, List<int>>();
        readonly List<List<int>> _sharedNormalGroupPool = new List<List<int>>();
        readonly Dictionary<long, int> _cutEdgeCounts = new Dictionary<long, int>();
        readonly Dictionary<Vector3Int, List<int>> _cutNormalGroups = new Dictionary<Vector3Int, List<int>>();
        readonly List<List<int>> _cutNormalGroupPool = new List<List<int>>();
        static readonly int[] s_emptyTriangleArray = new int[0];

        TetMeshData _data;
        bool        _initialized = false;

        // ── 生命週期 ─────────────────────────────────────────
        void Awake()
        {
            _meshFilter   = GetComponent<MeshFilter>();
            _meshRenderer = GetComponent<MeshRenderer>();

            if (meshLoader == null)
                meshLoader = GetComponent<TetMeshLoader>();
        }

        void Start()
        {
            if (meshLoader == null)
            {
                Debug.LogError("[TetMeshVisualizer] 未找到 TetMeshLoader，請掛載到同一個 GameObject");
                return;
            }

            // 等待加載完成後初始化
            meshLoader.OnMeshLoaded += Init;

            // 如果已經加載完成（同幀），直接初始化
            if (meshLoader.IsLoaded)
                Init(meshLoader.MeshData);
        }

        void OnDestroy()
        {
            if (meshLoader != null)
                meshLoader.OnMeshLoaded -= Init;
        }

        // ── 初始化 ──────────────────────────────────────────
        void Init(TetMeshData data)
        {
            _data = data;

            if (data.NumSurfaceTris == 0)
            {
                Debug.LogWarning("[TetMeshVisualizer] 沒有表面三角形數據，無法渲染");
                return;
            }

            // 構建 Unity Mesh
            _surfaceMesh = new Mesh
            {
                name = "TetSurface",
                // 超過 65535 頂點時需要 32-bit 索引
                indexFormat = data.NumParticles > 65535
                            ? UnityEngine.Rendering.IndexFormat.UInt32
                            : UnityEngine.Rendering.IndexFormat.UInt16
            };
            _surfaceMesh.MarkDynamic();

            // 頂點緩存
            _surfaceVerts   = new Vector3[data.NumParticles];
            _surfaceNormals = new Vector3[data.NumParticles];

            // 複製初始頂點位置
            System.Array.Copy(data.Positions, _surfaceVerts, data.NumParticles);

            _surfaceMesh.vertices  = _surfaceVerts;
            _surfaceMesh.triangles = data.SurfaceTriIds;
            _cachedSurfaceTriangles = CopyTriangles(data.SurfaceTriIds);
            _cachedCutTriangles = null;
            _surfaceUVs = BuildPlanarUVs(data.Positions, data.NumParticles);
            _surfaceMesh.uv = _surfaceUVs;
            EnsureRestPositionUVs();
            RecalculateSurfaceNormals();
            _surfaceMesh.RecalculateBounds();

            // ★ 关键修复: mesh setter 会创建实例副本,
            // 必须读回 getter 拿到渲染器实际使用的那个实例
            _meshFilter.mesh = _surfaceMesh;
            _surfaceMesh = _meshFilter.mesh;  // 捕获实际渲染实例
            _surfaceMesh.MarkDynamic();

            // 設置材質
            if (surfaceMaterial != null)
                _meshRenderer.material = surfaceMaterial;
            else
            {
                // 使用 URP Lit 默認材質（紅色，用於快速識別）
                var mat = new Material(Shader.Find("Universal Render Pipeline/Lit"));
                mat.color = new Color(0.85f, 0.25f, 0.15f); // 肝臟紅
                _meshRenderer.material = mat;
            }

            // 如果 surfaceMaterial 已由外部(SoftBody)预先设置，确保 _mainMaterial 正确
            if (surfaceMaterial != null)
                _mainMaterial = surfaceMaterial;
            else
                _mainMaterial = _meshRenderer.sharedMaterial != null
                    ? _meshRenderer.sharedMaterial
                    : _meshRenderer.material;

            _initialized = true;
            // Init完成后立即重新应用正确材质
            ApplySurfaceMaterials(_surfaceMesh != null && _surfaceMesh.subMeshCount > 1);
            Debug.Log($"[TetMeshVisualizer] ✅ 表面網格初始化完成 | " +
                      $"頂點: {data.NumParticles} | 三角形: {data.NumSurfaceTris}");
        }

        // ── 公開 API ─────────────────────────────────────────

        public void SetNormalMergeExcludedVertices(HashSet<int> excludedVertices)
        {
            _normalMergeExcludedVertices = excludedVertices;
        }

        public void SetSurfaceMaterial(Material material)
        {
            if (material == null) return;

            surfaceMaterial = material;
            _mainMaterial = material;

            if (_meshRenderer == null) return;
            bool hasCutSubmesh = _surfaceMesh != null && _surfaceMesh.subMeshCount > 1;
            ApplySurfaceMaterials(hasCutSubmesh);
        }

        public void SetCutSurfaceMaterial(Material material)
        {
            if (material == null) return;

            cutSurfaceMaterial = material;
            if (_meshRenderer == null || _surfaceMesh == null) return;
            if (_surfaceMesh.subMeshCount > 1)
                ApplySurfaceMaterials(hasCutSubmesh: true);
        }

        public void SetMainSurfaceVisible(bool visible)
        {
            _mainSurfaceVisible = visible;
            if (_meshRenderer == null || _surfaceMesh == null) return;
            ApplySurfaceMaterials(_surfaceMesh.subMeshCount > 1);
        }

        public bool HasCutSurfaceSubmesh =>
            _surfaceMesh != null &&
            _surfaceMesh.subMeshCount > 1 &&
            _surfaceMesh.GetIndexCount(1) > 0;

        /// <summary>
        /// 每幀物理求解後調用此方法刷新視覺網格
        /// 直接讀取 TetMeshData.Positions，無 GC 分配
        /// </summary>
        int _refreshCount = 0;

        public void Refresh()
        {
            if (!_initialized || _data == null) return;
            bool vertexCountChanged = _surfaceMesh.vertexCount != _data.NumParticles;

            // 检查顶点数是否变化（vertex splitting 会增加顶点）
            if (_surfaceVerts == null || _surfaceVerts.Length < _data.NumParticles)
            {
                _surfaceVerts   = new Vector3[_data.NumParticles];
                _surfaceNormals = new Vector3[_data.NumParticles];
                _surfaceUVs     = BuildPlanarUVs(_data.Positions, _data.NumParticles);
                _surfaceMesh.indexFormat = _data.NumParticles > 65535
                    ? UnityEngine.Rendering.IndexFormat.UInt32
                    : UnityEngine.Rendering.IndexFormat.UInt16;
            }

            // 只拷贝有效的粒子数
            System.Array.Copy(_data.Positions, _surfaceVerts, _data.NumParticles);

            _surfaceMesh.SetVertices(_surfaceVerts, 0, _data.NumParticles);
            if (vertexCountChanged || _surfaceUVs == null || _surfaceUVs.Length != _data.NumParticles)
            {
                _surfaceUVs = BuildPlanarUVs(_data.Positions, _data.NumParticles);
                _surfaceMesh.uv = _surfaceUVs;
                EnsureRestPositionUVs();
            }
            RecalculateSurfaceNormals();
            _surfaceMesh.RecalculateBounds();
        }

        /// <summary>
        /// 切割后调用: 重建拓扑 (三角形索引 + 顶点数组)
        /// 直接操作 _surfaceMesh, 避免 mf.mesh 副本问题
        /// </summary>
        public void RebuildTopology(int[] newTriangles)
        {
            if (!_initialized || _data == null || _surfaceMesh == null) return;

            // 扩容顶点缓存
            if (_surfaceVerts == null || _surfaceVerts.Length < _data.NumParticles)
            {
                _surfaceVerts   = new Vector3[_data.NumParticles];
                _surfaceNormals = new Vector3[_data.NumParticles];
                _surfaceUVs     = BuildPlanarUVs(_data.Positions, _data.NumParticles);
            }

            // 复制所有粒子位置
            System.Array.Copy(_data.Positions, _surfaceVerts, _data.NumParticles);

            // 重建 mesh
            _surfaceMesh.Clear();
            _surfaceMesh.indexFormat = _data.NumParticles > 65535
                ? UnityEngine.Rendering.IndexFormat.UInt32
                : UnityEngine.Rendering.IndexFormat.UInt16;
            _surfaceMesh.SetVertices(_surfaceVerts, 0, _data.NumParticles);
            if (_surfaceUVs == null || _surfaceUVs.Length != _data.NumParticles)
                _surfaceUVs = BuildPlanarUVs(_data.Positions, _data.NumParticles);
            _surfaceMesh.uv = _surfaceUVs;
            EnsureRestPositionUVs();
            _surfaceMesh.subMeshCount = 1;
            _surfaceMesh.SetTriangles(newTriangles, 0);
            _cachedSurfaceTriangles = CopyTriangles(newTriangles);
            _cachedCutTriangles = null;
            ApplySurfaceMaterials(hasCutSubmesh: false);
            RecalculateSurfaceNormals();
            _surfaceMesh.RecalculateBounds();

            // Avoid per-frame console spam during continuous cutting.
        }

        public void RebuildTopology(List<int> surfaceTriangles, List<int> cutTriangles)
        {
            if (!_initialized || _data == null || _surfaceMesh == null) return;

            if (_surfaceVerts == null || _surfaceVerts.Length < _data.NumParticles)
            {
                _surfaceVerts   = new Vector3[_data.NumParticles];
                _surfaceNormals = new Vector3[_data.NumParticles];
                _surfaceUVs     = BuildPlanarUVs(_data.Positions, _data.NumParticles);
            }

            System.Array.Copy(_data.Positions, _surfaceVerts, _data.NumParticles);

            _surfaceMesh.Clear();
            _surfaceMesh.indexFormat = _data.NumParticles > 65535
                ? UnityEngine.Rendering.IndexFormat.UInt32
                : UnityEngine.Rendering.IndexFormat.UInt16;
            _surfaceMesh.SetVertices(_surfaceVerts, 0, _data.NumParticles);
            if (_surfaceUVs == null || _surfaceUVs.Length != _data.NumParticles)
                _surfaceUVs = BuildPlanarUVs(_data.Positions, _data.NumParticles);
            _surfaceMesh.uv = _surfaceUVs;
            EnsureRestPositionUVs();

            bool hasCutSubmesh = cutTriangles != null && cutTriangles.Count >= 3;
            int[] surfaceTriangleCache = CopyTriangles(surfaceTriangles);
            int[] cutTriangleCache = hasCutSubmesh ? CopyTriangles(cutTriangles) : null;
            _surfaceMesh.subMeshCount = hasCutSubmesh ? 2 : 1;
            _surfaceMesh.SetTriangles(surfaceTriangleCache, 0);
            if (hasCutSubmesh)
                _surfaceMesh.SetTriangles(cutTriangleCache, 1);
            _cachedSurfaceTriangles = surfaceTriangleCache;
            _cachedCutTriangles = cutTriangleCache;

            ApplySurfaceMaterials(hasCutSubmesh);
            RecalculateSurfaceNormals();
            _surfaceMesh.RecalculateBounds();
        }

        void ApplySurfaceMaterials(bool hasCutSubmesh)
        {
            if (_meshRenderer == null) return;

            if (_mainMaterial == null)
                _mainMaterial = _meshRenderer.sharedMaterial != null
                    ? _meshRenderer.sharedMaterial
                    : _meshRenderer.material;

            if (!hasCutSubmesh)
            {
                Material surfaceMat = _mainSurfaceVisible ? _mainMaterial : EnsureInvisibleSurfaceMaterial();
                if (surfaceMat != null)
                    _meshRenderer.sharedMaterials = new[] { surfaceMat };
                return;
            }

            Material cutMat = cutSurfaceMaterial != null
                ? cutSurfaceMaterial
                : EnsureRuntimeCutSurfaceMaterial();

            Material mainMat = _mainSurfaceVisible ? _mainMaterial : EnsureInvisibleSurfaceMaterial();
            _meshRenderer.sharedMaterials = new[] { mainMat, cutMat };
        }

        Material EnsureInvisibleSurfaceMaterial()
        {
            if (_invisibleSurfaceMaterial != null) return _invisibleSurfaceMaterial;

            Shader shader = Shader.Find("Universal Render Pipeline/Unlit");
            if (shader == null) shader = Shader.Find("Unlit/Color");
            if (shader == null) shader = Shader.Find("Sprites/Default");
            if (shader == null) return _mainMaterial;

            _invisibleSurfaceMaterial = new Material(shader) { name = "Runtime_InvisibleTetSurface" };
            if (_invisibleSurfaceMaterial.HasProperty("_BaseColor"))
                _invisibleSurfaceMaterial.SetColor("_BaseColor", new Color(0f, 0f, 0f, 0f));
            if (_invisibleSurfaceMaterial.HasProperty("_Color"))
                _invisibleSurfaceMaterial.SetColor("_Color", new Color(0f, 0f, 0f, 0f));
            _invisibleSurfaceMaterial.SetOverrideTag("RenderType", "Transparent");
            _invisibleSurfaceMaterial.renderQueue = (int)UnityEngine.Rendering.RenderQueue.Transparent;
            if (_invisibleSurfaceMaterial.HasProperty("_Surface"))
                _invisibleSurfaceMaterial.SetFloat("_Surface", 1f);
            if (_invisibleSurfaceMaterial.HasProperty("_Blend"))
                _invisibleSurfaceMaterial.SetFloat("_Blend", 0f);
            if (_invisibleSurfaceMaterial.HasProperty("_SrcBlend"))
                _invisibleSurfaceMaterial.SetFloat("_SrcBlend", (float)UnityEngine.Rendering.BlendMode.SrcAlpha);
            if (_invisibleSurfaceMaterial.HasProperty("_DstBlend"))
                _invisibleSurfaceMaterial.SetFloat("_DstBlend", (float)UnityEngine.Rendering.BlendMode.OneMinusSrcAlpha);
            if (_invisibleSurfaceMaterial.HasProperty("_ZWrite"))
                _invisibleSurfaceMaterial.SetFloat("_ZWrite", 0f);
            if (_invisibleSurfaceMaterial.HasProperty("_Cull"))
                _invisibleSurfaceMaterial.SetFloat("_Cull", 0f);
            _invisibleSurfaceMaterial.EnableKeyword("_SURFACE_TYPE_TRANSPARENT");
            return _invisibleSurfaceMaterial;
        }

        Material EnsureRuntimeCutSurfaceMaterial()
        {
            if (_runtimeCutSurfaceMaterial != null) return _runtimeCutSurfaceMaterial;

            Shader shader = Shader.Find("SurgicalSim/LiverCutSurfaceGPU");
            if (shader == null) shader = Shader.Find("Universal Render Pipeline/Lit");
            if (shader == null) shader = Shader.Find("Universal Render Pipeline/Unlit");
            if (shader == null) shader = Shader.Find("Unlit/Color");
            if (shader == null) shader = Shader.Find("Sprites/Default");
            if (shader == null) shader = Shader.Find("Standard");
            if (shader == null) return _mainMaterial;

            _runtimeCutSurfaceMaterial = new Material(shader);
            _runtimeCutSurfaceMaterial.name = "Runtime_LiverCutSubmesh";
            if (_runtimeCutSurfaceMaterial.HasProperty("_CutColor"))
                _runtimeCutSurfaceMaterial.SetColor("_CutColor", cutSurfaceColor);
            if (_runtimeCutSurfaceMaterial.HasProperty("_InteriorColor"))
                _runtimeCutSurfaceMaterial.SetColor("_InteriorColor", cutSurfaceColor);
            if (_runtimeCutSurfaceMaterial.HasProperty("_BaseColor"))
                _runtimeCutSurfaceMaterial.SetColor("_BaseColor", cutSurfaceColor);
            if (_runtimeCutSurfaceMaterial.HasProperty("_Color"))
                _runtimeCutSurfaceMaterial.SetColor("_Color", cutSurfaceColor);
            if (_runtimeCutSurfaceMaterial.HasProperty("_TextureStrength"))
                _runtimeCutSurfaceMaterial.SetFloat("_TextureStrength", 0f);
            if (_runtimeCutSurfaceMaterial.HasProperty("_Roughness"))
                _runtimeCutSurfaceMaterial.SetFloat("_Roughness", 0.74f);
            if (_runtimeCutSurfaceMaterial.HasProperty("_Smoothness"))
                _runtimeCutSurfaceMaterial.SetFloat("_Smoothness", 0.18f);
            if (_runtimeCutSurfaceMaterial.HasProperty("_Metallic"))
                _runtimeCutSurfaceMaterial.SetFloat("_Metallic", 0f);
            if (_runtimeCutSurfaceMaterial.HasProperty("_Wetness"))
                _runtimeCutSurfaceMaterial.SetFloat("_Wetness", 0.18f);
            if (_runtimeCutSurfaceMaterial.HasProperty("_SpecularStrength"))
                _runtimeCutSurfaceMaterial.SetFloat("_SpecularStrength", 0.22f);
            if (_runtimeCutSurfaceMaterial.HasProperty("_FiberIntensity"))
                _runtimeCutSurfaceMaterial.SetFloat("_FiberIntensity", 0.08f);
            if (_runtimeCutSurfaceMaterial.HasProperty("_FiberScale"))
                _runtimeCutSurfaceMaterial.SetFloat("_FiberScale", 18f);
            ConfigureOpaqueTwoSided(_runtimeCutSurfaceMaterial);
            _runtimeCutSurfaceMaterial.doubleSidedGI = true;
            return _runtimeCutSurfaceMaterial;
        }

        static void ConfigureOpaqueTwoSided(Material mat)
        {
            if (mat == null) return;
            mat.SetOverrideTag("RenderType", "Opaque");
            mat.renderQueue = (int)UnityEngine.Rendering.RenderQueue.Geometry;

            if (mat.HasProperty("_Surface"))
                mat.SetFloat("_Surface", 0f);
            if (mat.HasProperty("_Blend"))
                mat.SetFloat("_Blend", 0f);
            if (mat.HasProperty("_SrcBlend"))
                mat.SetFloat("_SrcBlend", (float)UnityEngine.Rendering.BlendMode.One);
            if (mat.HasProperty("_DstBlend"))
                mat.SetFloat("_DstBlend", (float)UnityEngine.Rendering.BlendMode.Zero);
            if (mat.HasProperty("_ZWrite"))
                mat.SetFloat("_ZWrite", 1f);
            if (mat.HasProperty("_AlphaClip"))
                mat.SetFloat("_AlphaClip", 0f);
            if (mat.HasProperty("_Cull"))
                mat.SetFloat("_Cull", 0f);

            mat.DisableKeyword("_SURFACE_TYPE_TRANSPARENT");
            mat.DisableKeyword("_ALPHATEST_ON");
            mat.DisableKeyword("_ALPHAPREMULTIPLY_ON");
        }

        void RecalculateSurfaceNormals()
        {
            if (_surfaceMesh == null) return;

            _surfaceMesh.RecalculateNormals();
            Vector3[] normals = _surfaceMesh.normals;
            if (normals == null || normals.Length == 0) return;

            int count = Mathf.Min(_data != null ? _data.NumParticles : normals.Length, normals.Length);
            if (_surfaceVerts == null || _surfaceVerts.Length < count) return;

            bool needsSharedSmoothing = smoothSharedPositionNormals && normalMergeTolerance > 0f;
            bool cutMasksReady = (needsSharedSmoothing || smoothCutSurfaceNormals) &&
                                 PrepareCutNormalScratchAndMasks(count);

            if (needsSharedSmoothing)
                SmoothSharedPositionNormals(normals, count, skipCutVertices: cutMasksReady);

            if (smoothCutSurfaceNormals && cutMasksReady)
                ApplyCutSurfaceNormals(normals, count);

            _surfaceMesh.normals = normals;
        }

        void SmoothSharedPositionNormals(Vector3[] normals, int count, bool skipCutVertices)
        {
            float invTol = 1f / Mathf.Max(1e-7f, normalMergeTolerance);
            float minDot = Mathf.Cos(Mathf.Clamp(normalMergeMaxAngleDeg, 0f, 90f) * Mathf.Deg2Rad);
            ClearSharedNormalGroups();
            Dictionary<Vector3Int, List<int>> groups = _sharedNormalGroups;

            for (int i = 0; i < count; i++)
            {
                if (ShouldSkipSharedNormalVertex(i, skipCutVertices))
                    continue;

                Vector3Int key = NormalMergeKey(_surfaceVerts[i], invTol);
                if (!groups.TryGetValue(key, out var group))
                {
                    group = RentSharedNormalGroup();
                    groups[key] = group;
                }
                group.Add(i);
            }

            for (int i = 0; i < count; i++)
            {
                if (ShouldSkipSharedNormalVertex(i, skipCutVertices))
                    continue;

                Vector3 baseNormal = normals[i];
                if (baseNormal.sqrMagnitude < 1e-10f) continue;
                baseNormal.Normalize();

                Vector3 merged = Vector3.zero;
                if (!groups.TryGetValue(NormalMergeKey(_surfaceVerts[i], invTol), out var group))
                    continue;
                for (int j = 0; j < group.Count; j++)
                {
                    int idx = group[j];
                    Vector3 candidate = normals[idx];
                    if (candidate.sqrMagnitude < 1e-10f) continue;
                    candidate.Normalize();
                    if (Vector3.Dot(baseNormal, candidate) >= minDot)
                        merged += candidate;
                }

                if (merged.sqrMagnitude > 1e-10f)
                    normals[i] = merged.normalized;
            }

            ClearSharedNormalGroups();
        }

        bool ShouldSkipSharedNormalVertex(int vertex, bool skipCutVertices)
        {
            if (_normalMergeExcludedVertices != null &&
                _normalMergeExcludedVertices.Contains(vertex))
            {
                return true;
            }

            return skipCutVertices &&
                   _cutVertexScratch != null &&
                   vertex < _cutVertexScratch.Length &&
                   _cutVertexScratch[vertex];
        }

        List<int> RentSharedNormalGroup()
        {
            int last = _sharedNormalGroupPool.Count - 1;
            if (last >= 0)
            {
                List<int> group = _sharedNormalGroupPool[last];
                _sharedNormalGroupPool.RemoveAt(last);
                return group;
            }

            return new List<int>(2);
        }

        void ClearSharedNormalGroups()
        {
            if (_sharedNormalGroups.Count == 0)
                return;

            foreach (var pair in _sharedNormalGroups)
            {
                pair.Value.Clear();
                _sharedNormalGroupPool.Add(pair.Value);
            }
            _sharedNormalGroups.Clear();
        }

        bool PrepareCutNormalScratchAndMasks(int count)
        {
            if (_surfaceMesh.subMeshCount <= 1)
                return false;

            int[] cutTris = _cachedCutTriangles;
            if (cutTris == null || cutTris.Length < 3)
                return false;

            EnsureCutNormalScratch(count);
            System.Array.Clear(_cutSurfaceVertexScratch, 0, count);
            System.Array.Clear(_cutVertexScratch, 0, count);
            System.Array.Clear(_cutBoundaryVertexScratch, 0, count);
            System.Array.Clear(_cutAccumScratch, 0, count);
            _cutEdgeCounts.Clear();

            bool[] surfaceVertex = _cutSurfaceVertexScratch;
            int[] surfaceTris = _cachedSurfaceTriangles;
            if (surfaceTris != null && surfaceTris.Length >= 3)
            {
                for (int i = 0; i < surfaceTris.Length; i++)
                {
                    int v = surfaceTris[i];
                    if (v >= 0 && v < count)
                        surfaceVertex[v] = true;
                }
            }

            bool[] cutVertex = _cutVertexScratch;
            bool[] cutBoundaryVertex = _cutBoundaryVertexScratch;
            Vector3[] cutAccum = _cutAccumScratch;
            Dictionary<long, int> edgeCounts = _cutEdgeCounts;

            for (int i = 0; i + 2 < cutTris.Length; i += 3)
            {
                int a = cutTris[i];
                int b = cutTris[i + 1];
                int c = cutTris[i + 2];
                if (!ValidTriangle(a, b, c, count))
                    continue;

                Vector3 n = Vector3.Cross(_surfaceVerts[b] - _surfaceVerts[a], _surfaceVerts[c] - _surfaceVerts[a]);
                if (n.sqrMagnitude < 1e-14f)
                    continue;

                cutVertex[a] = true;
                cutVertex[b] = true;
                cutVertex[c] = true;
                cutAccum[a] += n;
                cutAccum[b] += n;
                cutAccum[c] += n;
                AddEdgeCount(edgeCounts, a, b);
                AddEdgeCount(edgeCounts, b, c);
                AddEdgeCount(edgeCounts, c, a);
            }

            MarkBoundaryEdges(edgeCounts, cutBoundaryVertex);
            return true;
        }

        void ApplyCutSurfaceNormals(Vector3[] normals, int count)
        {
            bool[] surfaceVertex = _cutSurfaceVertexScratch;
            bool[] cutVertex = _cutVertexScratch;
            bool[] cutBoundaryVertex = _cutBoundaryVertexScratch;
            Vector3[] cutAccum = _cutAccumScratch;

            float invTol = 1f / Mathf.Max(1e-7f, Mathf.Max(cutNormalMergeTolerance, normalMergeTolerance));
            ClearCutNormalGroups();
            Dictionary<Vector3Int, List<int>> groups = _cutNormalGroups;
            for (int i = 0; i < count; i++)
            {
                if (!IsCutInteriorVertex(i, cutVertex, surfaceVertex, cutBoundaryVertex, cutAccum))
                    continue;

                Vector3Int key = NormalMergeKey(_surfaceVerts[i], invTol);
                if (!groups.TryGetValue(key, out var group))
                {
                    group = RentCutNormalGroup();
                    groups[key] = group;
                }
                group.Add(i);
            }

            float minDot = Mathf.Cos(Mathf.Clamp(cutNormalMergeMaxAngleDeg, 0f, 90f) * Mathf.Deg2Rad);
            foreach (var pair in groups)
            {
                List<int> group = pair.Value;
                for (int i = 0; i < group.Count; i++)
                {
                    int idx = group[i];
                    Vector3 baseNormal = cutAccum[idx];
                    if (baseNormal.sqrMagnitude < 1e-10f)
                        continue;
                    baseNormal.Normalize();

                    Vector3 merged = Vector3.zero;
                    for (int j = 0; j < group.Count; j++)
                    {
                        int other = group[j];
                        Vector3 candidate = cutAccum[other];
                        if (candidate.sqrMagnitude < 1e-10f)
                            continue;
                        candidate.Normalize();
                        if (Vector3.Dot(baseNormal, candidate) >= minDot)
                            merged += cutAccum[other];
                    }

                    if (merged.sqrMagnitude > 1e-10f)
                        normals[idx] = merged.normalized;
                }
            }

            ClearCutNormalGroups();
        }

        void EnsureCutNormalScratch(int count)
        {
            if (_cutSurfaceVertexScratch == null || _cutSurfaceVertexScratch.Length < count)
                _cutSurfaceVertexScratch = new bool[count];
            if (_cutVertexScratch == null || _cutVertexScratch.Length < count)
                _cutVertexScratch = new bool[count];
            if (_cutBoundaryVertexScratch == null || _cutBoundaryVertexScratch.Length < count)
                _cutBoundaryVertexScratch = new bool[count];
            if (_cutAccumScratch == null || _cutAccumScratch.Length < count)
                _cutAccumScratch = new Vector3[count];
        }

        List<int> RentCutNormalGroup()
        {
            int last = _cutNormalGroupPool.Count - 1;
            if (last >= 0)
            {
                List<int> group = _cutNormalGroupPool[last];
                _cutNormalGroupPool.RemoveAt(last);
                return group;
            }

            return new List<int>(2);
        }

        void ClearCutNormalGroups()
        {
            if (_cutNormalGroups.Count == 0)
                return;

            foreach (var pair in _cutNormalGroups)
            {
                pair.Value.Clear();
                _cutNormalGroupPool.Add(pair.Value);
            }
            _cutNormalGroups.Clear();
        }

        static bool IsCutInteriorVertex(
            int vertex,
            bool[] cutVertex,
            bool[] surfaceVertex,
            bool[] cutBoundaryVertex,
            Vector3[] cutAccum)
        {
            return cutVertex[vertex] &&
                   !surfaceVertex[vertex] &&
                   !cutBoundaryVertex[vertex] &&
                   cutAccum[vertex].sqrMagnitude > 1e-14f;
        }

        static bool ValidTriangle(int a, int b, int c, int count)
        {
            return a >= 0 && a < count &&
                   b >= 0 && b < count &&
                   c >= 0 && c < count &&
                   a != b && b != c && c != a;
        }

        static void AddEdgeCount(Dictionary<long, int> edgeCounts, int a, int b)
        {
            long key = EdgeKey(a, b);
            edgeCounts.TryGetValue(key, out int count);
            edgeCounts[key] = count + 1;
        }

        static void MarkBoundaryEdges(Dictionary<long, int> edgeCounts, bool[] cutBoundaryVertex)
        {
            foreach (var pair in edgeCounts)
            {
                if (pair.Value != 1)
                    continue;

                ulong key = unchecked((ulong)pair.Key);
                int a = (int)(key >> 32);
                int b = (int)(key & 0xffffffffUL);
                cutBoundaryVertex[a] = true;
                cutBoundaryVertex[b] = true;
            }
        }

        static long EdgeKey(int a, int b)
        {
            uint lo = (uint)Mathf.Min(a, b);
            uint hi = (uint)Mathf.Max(a, b);
            return ((long)lo << 32) | (long)hi;
        }

        static int[] CopyTriangles(int[] triangles)
        {
            if (triangles == null || triangles.Length == 0)
                return s_emptyTriangleArray;

            int[] copy = new int[triangles.Length];
            System.Array.Copy(triangles, copy, triangles.Length);
            return copy;
        }

        static int[] CopyTriangles(List<int> triangles)
        {
            if (triangles == null || triangles.Count == 0)
                return s_emptyTriangleArray;

            int[] copy = new int[triangles.Count];
            triangles.CopyTo(copy);
            return copy;
        }

        static Vector3Int NormalMergeKey(Vector3 p, float invTol)
        {
            return new Vector3Int(
                Mathf.RoundToInt(p.x * invTol),
                Mathf.RoundToInt(p.y * invTol),
                Mathf.RoundToInt(p.z * invTol));
        }

        void EnsureRestPositionUVs()
        {
            if (_data == null || _surfaceMesh == null) return;

            int count = _surfaceMesh.vertexCount;
            if (count <= 0) count = _data.NumParticles;
            if (_surfaceRestPositions == null || _surfaceRestPositions.Length != count)
            {
                Vector3[] old = _surfaceRestPositions;
                _surfaceRestPositions = new Vector3[count];

                int copied = old == null ? 0 : Mathf.Min(old.Length, count);
                for (int i = 0; i < copied; i++)
                    _surfaceRestPositions[i] = old[i];
                for (int i = copied; i < count; i++)
                    _surfaceRestPositions[i] = i < _data.RestPositions.Length
                        ? _data.RestPositions[i]
                        : _data.Positions[i];
            }

            _surfaceMesh.SetUVs(1, new List<Vector3>(_surfaceRestPositions));
        }

        // UV 生成: 自动选择最宽的两个轴做平面投影 (XZ / XY / YZ 自适应)
        // SofaUnity 使用 .msh 内嵌 UV (EMBEDDED);
        // 我们用世界空间坐标投影与标准 Tiling(2,2) 配合实现类似效果。
        static Vector2[] BuildPlanarUVs(Vector3[] positions, int count)
        {
            var uvs = new Vector2[count];
            if (positions == null || count <= 0) return uvs;

            // 计算包围盒和质心
            Vector3 bmin = new Vector3(float.MaxValue,  float.MaxValue,  float.MaxValue);
            Vector3 bmax = new Vector3(float.MinValue, float.MinValue, float.MinValue);
            Vector3 center = Vector3.zero;
            for (int i = 0; i < count; i++)
            {
                bmin = Vector3.Min(bmin, positions[i]);
                bmax = Vector3.Max(bmax, positions[i]);
                center += positions[i];
            }
            center /= Mathf.Max(count, 1);

            Vector3 size = bmax - bmin;

            // 找到最短轴 (normal 方向), 另两个就是 UV 平面
            // 竖挂肝脏: Y 最长, XZ 个展 -> UV 用 X+Y
            // 平放肝脏: X/Z 最长, Y 最短 -> UV 用 X+Z
            float sx = size.x, sy = size.y, sz = size.z;
            int uAxis, vAxis;
            // 最短轴 = normal
            if (sx <= sy && sx <= sz)       { uAxis = 1; vAxis = 2; } // Y、Z 最宽
            else if (sy <= sx && sy <= sz)  { uAxis = 0; vAxis = 2; } // X、Z 最宽
            else                             { uAxis = 0; vAxis = 1; } // X、Y 最宽

            // uvScale: 肝脏尺寸约 0.12m。
            // UV = (pos - center) / uvScale -> 配合材质 Tiling(2,2) 后约重复 2 次
            const float uvScale = 0.12f;
            for (int i = 0; i < count; i++)
            {
                float u = (Axis(positions[i], uAxis) - Axis(center, uAxis)) / uvScale;
                float v = (Axis(positions[i], vAxis) - Axis(center, vAxis)) / uvScale;
                uvs[i] = new Vector2(u, v);
            }
            return uvs;
        }

        static float Axis(Vector3 v, int axis)
        {
            return axis == 0 ? v.x : (axis == 1 ? v.y : v.z);
        }

        // ── Gizmos（Debug 用）────────────────────────────────
        void OnDrawGizmos()
        {
            if (!showWireframe || _data == null || !_initialized) return;

            Gizmos.color = new Color(0f, 1f, 0f, 0.3f);

            // 畫出四面體邊框（只畫 active 的）
            int drawnCount = 0;
            for (int t = 0; t < _data.NumTets && drawnCount < 500; t++)
            {
                if (!_data.TetActive[t]) continue;

                Vector3 p0 = _data.Positions[_data.TetIds[t*4+0]];
                Vector3 p1 = _data.Positions[_data.TetIds[t*4+1]];
                Vector3 p2 = _data.Positions[_data.TetIds[t*4+2]];
                Vector3 p3 = _data.Positions[_data.TetIds[t*4+3]];

                Gizmos.DrawLine(p0, p1); Gizmos.DrawLine(p0, p2);
                Gizmos.DrawLine(p0, p3); Gizmos.DrawLine(p1, p2);
                Gizmos.DrawLine(p1, p3); Gizmos.DrawLine(p2, p3);
                drawnCount++;
            }
        }
    }
}
