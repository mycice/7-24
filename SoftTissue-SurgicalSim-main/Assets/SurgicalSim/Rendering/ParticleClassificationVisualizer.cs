using System.Collections.Generic;
using SurgicalSim.Core;
using UnityEngine;
using UnityEngine.Rendering;

namespace SurgicalSim.Rendering
{
    /// <summary>
    /// 在运行时用不同颜色显示四面体网格的表面粒子和内部粒子。
    /// 该组件只读取粒子位置，不会修改软体物理数据。
    /// </summary>
    [DisallowMultipleComponent]
    [RequireComponent(typeof(TetMeshLoader))]
    public sealed class ParticleClassificationVisualizer : MonoBehaviour
    {
        const int MaxInstancesPerBatch = 1023;

        [Header("粒子分类显示")]
        [InspectorName("显示粒子")]
        [Tooltip("总开关。关闭后不绘制任何粒子，也不会产生额外的每帧绘制开销。")]
        public bool showParticles = false;

        [InspectorName("显示表面粒子")]
        [Tooltip("显示参与构成器官外表面三角形的粒子。")]
        public bool showSurfaceParticles = true;

        [InspectorName("表面粒子颜色")]
        [Tooltip("器官外表面粒子的显示颜色。颜色的 A 值控制透明度。")]
        public Color surfaceParticleColor = new Color(1f, 0.78f, 0.08f, 0.9f);

        [InspectorName("显示内部粒子")]
        [Tooltip("显示位于器官体积内部、通常被外表面遮挡的粒子。")]
        public bool showInteriorParticles = true;

        [InspectorName("内部粒子颜色")]
        [Tooltip("器官内部粒子的显示颜色。颜色的 A 值控制透明度。")]
        public Color interiorParticleColor = new Color(0.05f, 0.85f, 1f, 0.65f);

        [InspectorName("粒子大小")]
        [Tooltip("粒子标记在模型局部坐标中的直径。")]
        [Range(0.001f, 0.03f)]
        public float particleSize = 0.006f;

        [InspectorName("穿透器官显示")]
        [Tooltip("开启后，内部粒子不会被肝脏表面遮挡；关闭后遵循正常的深度遮挡。")]
        public bool showThroughOrgan = true;

        [Header("四面体线框显示")]
        [InspectorName("显示四面体")]
        [Tooltip("显示当前活动四面体的六条边。")]
        public bool showTetrahedra = false;

        [InspectorName("四面体颜色")]
        [Tooltip("四面体线框颜色。颜色的 A 值会与四面体透明度相乘。")]
        public Color tetrahedronColor = new Color(0.25f, 1f, 0.35f, 1f);

        [InspectorName("四面体透明度")]
        [Tooltip("单独控制四面体线框的透明度。")]
        [Range(0f, 1f)]
        public float tetrahedronOpacity = 0.75f;

        [InspectorName("四面体穿透显示")]
        [Tooltip("开启后，四面体线框会显示在器官表面之上。")]
        public bool tetrahedronShowThroughOrgan = true;

        [Header("表面三角形显示")]
        [InspectorName("显示表面三角形")]
        [Tooltip("显示四面体网格的外表面三角形。")]
        public bool showSurfaceTriangles = false;

        [InspectorName("表面三角形颜色")]
        [Tooltip("表面三角形的填充颜色。颜色的 A 值会与透明度相乘。")]
        public Color surfaceTriangleColor = new Color(1f, 0.2f, 0.1f, 1f);

        [InspectorName("表面三角形透明度")]
        [Tooltip("单独控制表面三角形填充的透明度。")]
        [Range(0f, 1f)]
        public float surfaceTriangleOpacity = 0.2f;

        [InspectorName("表面三角形穿透显示")]
        [Tooltip("开启后，表面三角形会显示在原肝脏表面之上。")]
        public bool surfaceTriangleShowThroughOrgan = true;

        [HideInInspector]
        public Shader particleShader;

        TetMeshLoader _loader;
        TetMeshData _data;
        Mesh _particleMesh;
        Material _surfaceMaterial;
        Material _interiorMaterial;
        Mesh _tetraMesh;
        Mesh _surfaceTriangleMesh;
        Material _tetraMaterial;
        Material _surfaceTriangleMaterial;
        bool[] _surfaceParticleMask;
        int[] _classifiedSurfaceTriangles;
        int _classifiedParticleCount = -1;
        int[] _tetraLineIndices;
        int _tetraTopologySignature;
        int _tetraTopologyCount = -1;
        int _tetraTopologyParticleCount = -1;
        int _surfaceTriangleSignature;
        int _surfaceTriangleCount = -1;

        readonly Matrix4x4[] _surfaceMatrices = new Matrix4x4[MaxInstancesPerBatch];
        readonly Matrix4x4[] _interiorMatrices = new Matrix4x4[MaxInstancesPerBatch];

        void Awake()
        {
            _loader = GetComponent<TetMeshLoader>();
        }

        void OnEnable()
        {
            if (_loader == null)
                _loader = GetComponent<TetMeshLoader>();

            if (_loader != null)
            {
                _loader.OnMeshLoaded += HandleMeshLoaded;
                if (_loader.IsLoaded)
                    HandleMeshLoaded(_loader.MeshData);
            }
        }

        void OnDisable()
        {
            if (_loader != null)
                _loader.OnMeshLoaded -= HandleMeshLoaded;
        }

        void OnDestroy()
        {
            DestroyRuntimeObject(_particleMesh);
            DestroyRuntimeObject(_surfaceMaterial);
            DestroyRuntimeObject(_interiorMaterial);
            DestroyRuntimeObject(_tetraMesh);
            DestroyRuntimeObject(_surfaceTriangleMesh);
            DestroyRuntimeObject(_tetraMaterial);
            DestroyRuntimeObject(_surfaceTriangleMaterial);
        }

        void LateUpdate()
        {
            if (_data == null || _data.Positions == null)
                return;

            if (showParticles && (showSurfaceParticles || showInteriorParticles))
            {
                EnsureResources();
                if (_particleMesh != null && _surfaceMaterial != null && _interiorMaterial != null)
                {
                    EnsureParticleClassification();
                    UpdateMaterialSettings();
                    DrawParticles();
                }
            }

            if (showTetrahedra || showSurfaceTriangles)
            {
                EnsureOverlayResources();
                if (_tetraMesh != null && _surfaceTriangleMesh != null &&
                    _tetraMaterial != null && _surfaceTriangleMaterial != null)
                {
                    EnsureOverlayTopology();
                    UpdateOverlayMeshes();
                    UpdateOverlayMaterialSettings();
                    DrawOverlays();
                }
            }
        }

        void HandleMeshLoaded(TetMeshData data)
        {
            _data = data;
            _classifiedSurfaceTriangles = null;
            _classifiedParticleCount = -1;
            _tetraTopologyCount = -1;
            _tetraTopologyParticleCount = -1;
            _surfaceTriangleCount = -1;
            EnsureParticleClassification();
        }

        void EnsureParticleClassification()
        {
            if (_data == null)
                return;

            int[] surfaceTriangles = _data.SurfaceTriIds;
            if (_surfaceParticleMask != null &&
                _classifiedParticleCount == _data.NumParticles &&
                ReferenceEquals(_classifiedSurfaceTriangles, surfaceTriangles))
            {
                return;
            }

            _surfaceParticleMask = new bool[_data.NumParticles];
            if (surfaceTriangles != null)
            {
                for (int i = 0; i < surfaceTriangles.Length; i++)
                {
                    int particleIndex = surfaceTriangles[i];
                    if ((uint)particleIndex < (uint)_surfaceParticleMask.Length)
                        _surfaceParticleMask[particleIndex] = true;
                }
            }

            _classifiedSurfaceTriangles = surfaceTriangles;
            _classifiedParticleCount = _data.NumParticles;
        }

        void EnsureResources()
        {
            if (_particleMesh == null)
                _particleMesh = CreateOctahedronMesh();

            if (particleShader == null)
                particleShader = Shader.Find("SurgicalSim/ParticleClassification");

            if (particleShader == null)
            {
                Debug.LogError("[粒子分类显示] 找不到 SurgicalSim/ParticleClassification Shader。", this);
                enabled = false;
                return;
            }

            if (_surfaceMaterial == null)
                _surfaceMaterial = CreateRuntimeMaterial("表面粒子材质");
            if (_interiorMaterial == null)
                _interiorMaterial = CreateRuntimeMaterial("内部粒子材质");
        }

        void EnsureOverlayResources()
        {
            if (particleShader == null)
                particleShader = Shader.Find("SurgicalSim/ParticleClassification");

            if (particleShader == null)
            {
                Debug.LogError("[网格结构显示] 找不到 SurgicalSim/ParticleClassification Shader。", this);
                enabled = false;
                return;
            }

            IndexFormat indexFormat = _data.NumParticles > ushort.MaxValue
                ? IndexFormat.UInt32
                : IndexFormat.UInt16;

            if (_tetraMesh == null)
            {
                _tetraMesh = new Mesh
                {
                    name = "四面体线框网格",
                    indexFormat = indexFormat,
                    hideFlags = HideFlags.HideAndDontSave
                };
                _tetraMesh.MarkDynamic();
            }
            else if (_tetraMesh.indexFormat != indexFormat)
            {
                _tetraMesh.Clear();
                _tetraMesh.indexFormat = indexFormat;
                _tetraTopologyCount = -1;
            }

            if (_surfaceTriangleMesh == null)
            {
                _surfaceTriangleMesh = new Mesh
                {
                    name = "表面三角形网格",
                    indexFormat = indexFormat,
                    hideFlags = HideFlags.HideAndDontSave
                };
                _surfaceTriangleMesh.MarkDynamic();
            }
            else if (_surfaceTriangleMesh.indexFormat != indexFormat)
            {
                _surfaceTriangleMesh.Clear();
                _surfaceTriangleMesh.indexFormat = indexFormat;
                _surfaceTriangleCount = -1;
            }

            if (_tetraMaterial == null)
                _tetraMaterial = CreateRuntimeMaterial("四面体线框材质");
            if (_surfaceTriangleMaterial == null)
                _surfaceTriangleMaterial = CreateRuntimeMaterial("表面三角形材质");
        }

        void EnsureOverlayTopology()
        {
            if (showTetrahedra)
            {
                int signature = ComputeTetrahedronTopologySignature();
                if (_tetraTopologyCount != _data.NumTets ||
                    _tetraTopologyParticleCount != _data.NumParticles ||
                    _tetraTopologySignature != signature)
                {
                    RebuildTetrahedronLines();
                    _tetraTopologyCount = _data.NumTets;
                    _tetraTopologyParticleCount = _data.NumParticles;
                    _tetraTopologySignature = signature;
                }
            }

            if (showSurfaceTriangles)
            {
                int triangleCount = _data.SurfaceTriIds == null ? 0 : _data.SurfaceTriIds.Length;
                int signature = ComputeSurfaceTriangleSignature();
                if (_surfaceTriangleCount != triangleCount ||
                    _surfaceTriangleSignature != signature)
                {
                    RebuildSurfaceTriangles();
                    _surfaceTriangleCount = triangleCount;
                    _surfaceTriangleSignature = signature;
                }
            }
        }

        int ComputeTetrahedronTopologySignature()
        {
            unchecked
            {
                int hash = 17;
                int tetCount = Mathf.Min(_data.NumTets, _data.TetActive == null ? 0 : _data.TetActive.Length);
                tetCount = Mathf.Min(tetCount, _data.TetIds == null ? 0 : _data.TetIds.Length / 4);
                for (int t = 0; t < tetCount; t++)
                {
                    hash = hash * 31 + (_data.TetActive[t] ? 1 : 0);
                    int b = t * 4;
                    hash = hash * 31 + _data.TetIds[b];
                    hash = hash * 31 + _data.TetIds[b + 1];
                    hash = hash * 31 + _data.TetIds[b + 2];
                    hash = hash * 31 + _data.TetIds[b + 3];
                }
                return hash;
            }
        }

        int ComputeSurfaceTriangleSignature()
        {
            unchecked
            {
                int hash = 17;
                int[] triangles = _data.SurfaceTriIds;
                if (triangles == null)
                    return hash;

                for (int i = 0; i < triangles.Length; i++)
                    hash = hash * 31 + triangles[i];
                return hash;
            }
        }

        void RebuildTetrahedronLines()
        {
            var edgeKeys = new HashSet<ulong>();
            var lineIndices = new List<int>();
            int particleCount = Mathf.Min(_data.NumParticles, _data.Positions.Length);
            int tetCount = Mathf.Min(_data.NumTets, _data.TetActive == null ? 0 : _data.TetActive.Length);
            tetCount = Mathf.Min(tetCount, _data.TetIds == null ? 0 : _data.TetIds.Length / 4);

            for (int t = 0; t < tetCount; t++)
            {
                if (!_data.TetActive[t])
                    continue;

                int b = t * 4;
                int v0 = _data.TetIds[b];
                int v1 = _data.TetIds[b + 1];
                int v2 = _data.TetIds[b + 2];
                int v3 = _data.TetIds[b + 3];
                AddUniqueEdge(v0, v1, particleCount, edgeKeys, lineIndices);
                AddUniqueEdge(v0, v2, particleCount, edgeKeys, lineIndices);
                AddUniqueEdge(v0, v3, particleCount, edgeKeys, lineIndices);
                AddUniqueEdge(v1, v2, particleCount, edgeKeys, lineIndices);
                AddUniqueEdge(v1, v3, particleCount, edgeKeys, lineIndices);
                AddUniqueEdge(v2, v3, particleCount, edgeKeys, lineIndices);
            }

            _tetraLineIndices = lineIndices.ToArray();
            _tetraMesh.Clear(false);
            _tetraMesh.SetVertices(_data.Positions, 0, particleCount);
            _tetraMesh.SetIndices(_tetraLineIndices, MeshTopology.Lines, 0, false);
            _tetraMesh.RecalculateBounds();
        }

        static void AddUniqueEdge(
            int a,
            int b,
            int particleCount,
            HashSet<ulong> edgeKeys,
            List<int> lineIndices)
        {
            if ((uint)a >= (uint)particleCount || (uint)b >= (uint)particleCount || a == b)
                return;

            int min = Mathf.Min(a, b);
            int max = Mathf.Max(a, b);
            ulong key = ((ulong)(uint)min << 32) | (uint)max;
            if (!edgeKeys.Add(key))
                return;

            lineIndices.Add(min);
            lineIndices.Add(max);
        }

        void RebuildSurfaceTriangles()
        {
            int particleCount = Mathf.Min(_data.NumParticles, _data.Positions.Length);
            int[] source = _data.SurfaceTriIds;
            var validTriangles = new List<int>(source == null ? 0 : source.Length);
            if (source != null)
            {
                int completeIndexCount = source.Length - source.Length % 3;
                for (int i = 0; i < completeIndexCount; i += 3)
                {
                    int a = source[i];
                    int b = source[i + 1];
                    int c = source[i + 2];
                    if ((uint)a >= (uint)particleCount ||
                        (uint)b >= (uint)particleCount ||
                        (uint)c >= (uint)particleCount)
                    {
                        continue;
                    }

                    validTriangles.Add(a);
                    validTriangles.Add(b);
                    validTriangles.Add(c);
                }
            }

            _surfaceTriangleMesh.Clear(false);
            _surfaceTriangleMesh.SetVertices(_data.Positions, 0, particleCount);
            _surfaceTriangleMesh.SetTriangles(validTriangles, 0, false);
            _surfaceTriangleMesh.RecalculateBounds();
        }

        void UpdateOverlayMeshes()
        {
            int particleCount = Mathf.Min(_data.NumParticles, _data.Positions.Length);
            if (showTetrahedra && _tetraLineIndices != null)
            {
                _tetraMesh.SetVertices(_data.Positions, 0, particleCount);
                _tetraMesh.RecalculateBounds();
            }

            if (showSurfaceTriangles && _surfaceTriangleMesh.subMeshCount > 0)
            {
                _surfaceTriangleMesh.SetVertices(_data.Positions, 0, particleCount);
                _surfaceTriangleMesh.RecalculateBounds();
            }
        }

        void UpdateOverlayMaterialSettings()
        {
            Color tetraColor = tetrahedronColor;
            tetraColor.a *= Mathf.Clamp01(tetrahedronOpacity);
            _tetraMaterial.SetColor("_Color", tetraColor);
            _tetraMaterial.SetFloat(
                "_ZTest",
                tetrahedronShowThroughOrgan
                    ? (float)CompareFunction.Always
                    : (float)CompareFunction.LessEqual);

            Color triangleColor = surfaceTriangleColor;
            triangleColor.a *= Mathf.Clamp01(surfaceTriangleOpacity);
            _surfaceTriangleMaterial.SetColor("_Color", triangleColor);
            _surfaceTriangleMaterial.SetFloat(
                "_ZTest",
                surfaceTriangleShowThroughOrgan
                    ? (float)CompareFunction.Always
                    : (float)CompareFunction.LessEqual);
        }

        void DrawOverlays()
        {
            Matrix4x4 localToWorld = transform.localToWorldMatrix;
            if (showSurfaceTriangles && _surfaceTriangleMesh.GetIndexCount(0) > 0)
            {
                Graphics.DrawMesh(
                    _surfaceTriangleMesh,
                    localToWorld,
                    _surfaceTriangleMaterial,
                    gameObject.layer,
                    null,
                    0,
                    null,
                    ShadowCastingMode.Off,
                    false,
                    null,
                    LightProbeUsage.Off,
                    null);
            }

            if (showTetrahedra && _tetraMesh.GetIndexCount(0) > 0)
            {
                Graphics.DrawMesh(
                    _tetraMesh,
                    localToWorld,
                    _tetraMaterial,
                    gameObject.layer,
                    null,
                    0,
                    null,
                    ShadowCastingMode.Off,
                    false,
                    null,
                    LightProbeUsage.Off,
                    null);
            }
        }

        Material CreateRuntimeMaterial(string materialName)
        {
            var material = new Material(particleShader)
            {
                name = materialName,
                enableInstancing = true,
                hideFlags = HideFlags.HideAndDontSave
            };
            return material;
        }

        void UpdateMaterialSettings()
        {
            _surfaceMaterial.SetColor("_Color", surfaceParticleColor);
            _interiorMaterial.SetColor("_Color", interiorParticleColor);

            float depthTest = showThroughOrgan
                ? (float)CompareFunction.Always
                : (float)CompareFunction.LessEqual;
            _surfaceMaterial.SetFloat("_ZTest", depthTest);
            _interiorMaterial.SetFloat("_ZTest", depthTest);
        }

        void DrawParticles()
        {
            int surfaceCount = 0;
            int interiorCount = 0;
            float size = Mathf.Max(0.0001f, particleSize);
            Vector3 scale = Vector3.one * size;
            Matrix4x4 localToWorld = transform.localToWorldMatrix;

            int count = Mathf.Min(_data.NumParticles, _data.Positions.Length);
            for (int i = 0; i < count; i++)
            {
                bool isSurface = _surfaceParticleMask != null &&
                                 i < _surfaceParticleMask.Length &&
                                 _surfaceParticleMask[i];

                if (isSurface)
                {
                    if (!showSurfaceParticles)
                        continue;

                    _surfaceMatrices[surfaceCount++] =
                        localToWorld * Matrix4x4.TRS(_data.Positions[i], Quaternion.identity, scale);
                    if (surfaceCount == MaxInstancesPerBatch)
                    {
                        DrawBatch(_surfaceMaterial, _surfaceMatrices, surfaceCount);
                        surfaceCount = 0;
                    }
                }
                else
                {
                    if (!showInteriorParticles)
                        continue;

                    _interiorMatrices[interiorCount++] =
                        localToWorld * Matrix4x4.TRS(_data.Positions[i], Quaternion.identity, scale);
                    if (interiorCount == MaxInstancesPerBatch)
                    {
                        DrawBatch(_interiorMaterial, _interiorMatrices, interiorCount);
                        interiorCount = 0;
                    }
                }
            }

            if (surfaceCount > 0)
                DrawBatch(_surfaceMaterial, _surfaceMatrices, surfaceCount);
            if (interiorCount > 0)
                DrawBatch(_interiorMaterial, _interiorMatrices, interiorCount);
        }

        void DrawBatch(Material material, Matrix4x4[] matrices, int count)
        {
            Graphics.DrawMeshInstanced(
                _particleMesh,
                0,
                material,
                matrices,
                count,
                null,
                ShadowCastingMode.Off,
                false,
                gameObject.layer,
                null,
                LightProbeUsage.Off,
                null);
        }

        static Mesh CreateOctahedronMesh()
        {
            var mesh = new Mesh
            {
                name = "粒子标记八面体",
                hideFlags = HideFlags.HideAndDontSave
            };

            mesh.vertices = new[]
            {
                new Vector3(0f, 0.5f, 0f),
                new Vector3(0f, -0.5f, 0f),
                new Vector3(-0.5f, 0f, 0f),
                new Vector3(0.5f, 0f, 0f),
                new Vector3(0f, 0f, -0.5f),
                new Vector3(0f, 0f, 0.5f)
            };
            mesh.triangles = new[]
            {
                0, 5, 3,
                0, 3, 4,
                0, 4, 2,
                0, 2, 5,
                1, 3, 5,
                1, 4, 3,
                1, 2, 4,
                1, 5, 2
            };
            mesh.RecalculateNormals();
            mesh.RecalculateBounds();
            return mesh;
        }

        static void DestroyRuntimeObject(Object obj)
        {
            if (obj == null)
                return;

            if (Application.isPlaying)
                Destroy(obj);
            else
                DestroyImmediate(obj);
        }
    }
}
