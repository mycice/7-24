using System;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Rendering;

namespace ReconGridDC.Cuda
{
    public sealed class DcLiverVisualRenderer : MonoBehaviour
    {
        struct VertexKey : IEquatable<VertexKey>
        {
            public int rx, ry, rz;
            public int px, py, pz;

            public bool Equals(VertexKey other)
            {
                return rx == other.rx && ry == other.ry && rz == other.rz &&
                       px == other.px && py == other.py && pz == other.pz;
            }

            public override bool Equals(object obj)
            {
                return obj is VertexKey other && Equals(other);
            }

            public override int GetHashCode()
            {
                unchecked
                {
                    int h = 17;
                    h = h * 31 + rx; h = h * 31 + ry; h = h * 31 + rz;
                    h = h * 31 + px; h = h * 31 + py; h = h * 31 + pz;
                    return h;
                }
            }
        }

        const float Quant = 10000f;

        MeshFilter _outerFilter, _cutFilter, _membraneFilter;
        MeshRenderer _outerRenderer, _cutRenderer, _membraneRenderer;
        Mesh _outerMesh, _cutMesh;

        readonly Dictionary<VertexKey, int> _outerMap = new Dictionary<VertexKey, int>(32768);
        readonly List<Vector3> _outerVerts = new List<Vector3>(32768);
        readonly List<Vector2> _outerUvs = new List<Vector2>(32768);
        readonly List<Vector4> _outerAux = new List<Vector4>(32768);
        readonly List<Vector3> _outerNormalSums = new List<Vector3>(32768);
        readonly List<Vector3> _outerNormals = new List<Vector3>(32768);
        readonly List<Vector4> _outerTangents = new List<Vector4>(32768);
        readonly List<Vector3> _tanSums = new List<Vector3>(32768);
        readonly List<int> _outerIndices = new List<int>(65536);

        readonly List<Vector3> _cutVerts = new List<Vector3>(8192);
        readonly List<Vector3> _cutNormals = new List<Vector3>(8192);
        readonly List<int> _cutIndices = new List<int>(8192);
        bool _presentationVisible = true;

        public bool IsInitialized => _outerMesh != null && _cutMesh != null;
        public bool HasVisibleGeometry { get; private set; }
        public Mesh LiveOuterSurfaceMesh => _outerMesh;

        public void Initialize(Material exteriorMaterial, Material cutMaterial, Material membraneMaterial = null)
        {
            EnsureObjects();
            if (exteriorMaterial != null) _outerRenderer.sharedMaterial = exteriorMaterial;
            if (cutMaterial != null) _cutRenderer.sharedMaterial = cutMaterial;
            ConfigureMembrane(membraneMaterial);
        }

        public void ConfigureMembrane(Material membraneMaterial)
        {
            EnsureObjects();
            _membraneRenderer.sharedMaterial = membraneMaterial;
            _membraneRenderer.enabled = _presentationVisible && membraneMaterial != null && _outerIndices.Count > 0;
        }

        public void SetVisible(bool visible)
        {
            _presentationVisible = visible;
            if (_outerRenderer != null) _outerRenderer.enabled = visible && _outerIndices.Count > 0;
            if (_cutRenderer != null) _cutRenderer.enabled = visible && _cutIndices.Count > 0;
            if (_membraneRenderer != null)
                _membraneRenderer.enabled = visible && _membraneRenderer.sharedMaterial != null && _outerIndices.Count > 0;
        }

        public void ClearVisuals()
        {
            HasVisibleGeometry = false;
            if (_outerMesh != null) _outerMesh.Clear();
            if (_cutMesh != null) _cutMesh.Clear();
            if (_outerRenderer != null) _outerRenderer.enabled = false;
            if (_cutRenderer != null) _cutRenderer.enabled = false;
            if (_membraneRenderer != null) _membraneRenderer.enabled = false;
        }

        public void UpdateFromSoup(
            Vector3[] soupVerts,
            Vector3[] soupNormals,
            Vector2[] soupUvs,
            Vector4[] soupAux,
            int count,
            Bounds liveBounds)
        {
            EnsureObjects();
            ClearWorkingData();

            if (soupVerts == null || soupAux == null || count < 3)
            {
                ClearVisuals();
                return;
            }

            int triCount = count - count % 3;
            for (int i = 0; i < triCount; i += 3)
            {
                bool cutWall = soupAux[i].w >= 0.5f || soupAux[i + 1].w >= 0.5f || soupAux[i + 2].w >= 0.5f;
                if (cutWall) AddCutTriangle(soupVerts, i);
                else AddOuterTriangle(soupVerts, soupUvs, soupAux, i);
            }

            CommitOuterMesh(liveBounds);
            CommitCutMesh(liveBounds);
            HasVisibleGeometry = _outerIndices.Count > 0 || _cutIndices.Count > 0;
        }

        void EnsureObjects()
        {
            if (_outerFilter == null || _outerRenderer == null)
                EnsureChild("DcLiverExterior", out _outerFilter, out _outerRenderer);
            if (_cutFilter == null || _cutRenderer == null)
                EnsureChild("DcLiverCutInterior", out _cutFilter, out _cutRenderer);
            if (_membraneFilter == null || _membraneRenderer == null)
                EnsureChild("DcLiverSurfaceMembrane", out _membraneFilter, out _membraneRenderer);

            if (_outerMesh == null)
            {
                _outerMesh = new Mesh { name = "DcLiverExteriorMesh", indexFormat = IndexFormat.UInt32 };
                _outerMesh.MarkDynamic();
                _outerFilter.sharedMesh = _outerMesh;
                _membraneFilter.sharedMesh = _outerMesh;
            }
            if (_cutMesh == null)
            {
                _cutMesh = new Mesh { name = "DcLiverCutInteriorMesh", indexFormat = IndexFormat.UInt32 };
                _cutMesh.MarkDynamic();
                _cutFilter.sharedMesh = _cutMesh;
            }
        }

        void EnsureChild(string childName, out MeshFilter filter, out MeshRenderer renderer)
        {
            Transform child = transform.Find(childName);
            GameObject go = child != null ? child.gameObject : new GameObject(childName);
            go.transform.SetParent(transform, false);
            go.transform.localPosition = Vector3.zero;
            go.transform.localRotation = Quaternion.identity;
            go.transform.localScale = Vector3.one;

            filter = go.GetComponent<MeshFilter>();
            if (filter == null) filter = go.AddComponent<MeshFilter>();
            renderer = go.GetComponent<MeshRenderer>();
            if (renderer == null) renderer = go.AddComponent<MeshRenderer>();
            renderer.shadowCastingMode = ShadowCastingMode.On;
            renderer.receiveShadows = true;
            renderer.enabled = false;
        }

        void ClearWorkingData()
        {
            _outerMap.Clear();
            _outerVerts.Clear();
            _outerUvs.Clear();
            _outerAux.Clear();
            _outerNormalSums.Clear();
            _outerNormals.Clear();
            _outerTangents.Clear();
            _tanSums.Clear();
            _outerIndices.Clear();

            _cutVerts.Clear();
            _cutNormals.Clear();
            _cutIndices.Clear();
        }

        void AddOuterTriangle(Vector3[] soupVerts, Vector2[] soupUvs, Vector4[] soupAux, int i)
        {
            Vector3 p0 = soupVerts[i];
            Vector3 p1 = soupVerts[i + 1];
            Vector3 p2 = soupVerts[i + 2];
            Vector3 face = Vector3.Cross(p1 - p0, p2 - p0);
            if (!IsFinite(face) || face.sqrMagnitude < 1e-12f) return;

            int a = AddOuterVertex(p0, soupUvs != null ? soupUvs[i] : Vector2.zero, soupAux[i]);
            int b = AddOuterVertex(p1, soupUvs != null ? soupUvs[i + 1] : Vector2.zero, soupAux[i + 1]);
            int c = AddOuterVertex(p2, soupUvs != null ? soupUvs[i + 2] : Vector2.zero, soupAux[i + 2]);
            if (a == b || b == c || c == a) return;

            AccumulateNormal(a, face);
            AccumulateNormal(b, face);
            AccumulateNormal(c, face);
            _outerIndices.Add(a);
            _outerIndices.Add(b);
            _outerIndices.Add(c);
        }

        int AddOuterVertex(Vector3 pos, Vector2 uv, Vector4 aux)
        {
            var key = new VertexKey
            {
                rx = Q(aux.x), ry = Q(aux.y), rz = Q(aux.z),
                px = Q(pos.x), py = Q(pos.y), pz = Q(pos.z)
            };
            if (_outerMap.TryGetValue(key, out int index)) return index;

            index = _outerVerts.Count;
            _outerMap.Add(key, index);
            _outerVerts.Add(pos);
            _outerUvs.Add(uv);
            _outerAux.Add(aux);
            _outerNormalSums.Add(Vector3.zero);
            return index;
        }

        void AddCutTriangle(Vector3[] soupVerts, int i)
        {
            Vector3 p0 = soupVerts[i];
            Vector3 p1 = soupVerts[i + 1];
            Vector3 p2 = soupVerts[i + 2];
            Vector3 n = Vector3.Cross(p1 - p0, p2 - p0);
            if (!IsFinite(n) || n.sqrMagnitude < 1e-12f) return;
            n.Normalize();

            int start = _cutVerts.Count;
            _cutVerts.Add(p0); _cutVerts.Add(p1); _cutVerts.Add(p2);
            _cutNormals.Add(n); _cutNormals.Add(n); _cutNormals.Add(n);
            _cutIndices.Add(start); _cutIndices.Add(start + 1); _cutIndices.Add(start + 2);
        }

        void CommitOuterMesh(Bounds liveBounds)
        {
            _outerMesh.Clear();
            if (_outerIndices.Count == 0)
            {
                _outerRenderer.enabled = false;
                _membraneRenderer.enabled = false;
                return;
            }

            EnsureUsableOuterUvs(liveBounds);
            NormalizeOuterNormals();
            BuildOuterTangents();

            _outerMesh.indexFormat = _outerVerts.Count > 65535 ? IndexFormat.UInt32 : IndexFormat.UInt16;
            _outerMesh.SetVertices(_outerVerts);
            _outerMesh.SetUVs(0, _outerUvs);
            _outerMesh.SetUVs(1, _outerAux);
            _outerMesh.SetNormals(_outerNormals);
            _outerMesh.SetTangents(_outerTangents);
            _outerMesh.SetTriangles(_outerIndices, 0);
            _outerMesh.bounds = liveBounds;
            _outerRenderer.enabled = _presentationVisible;
            _membraneRenderer.enabled = _presentationVisible && _membraneRenderer.sharedMaterial != null;
        }

        void CommitCutMesh(Bounds liveBounds)
        {
            _cutMesh.Clear();
            if (_cutIndices.Count == 0)
            {
                _cutRenderer.enabled = false;
                return;
            }

            _cutMesh.indexFormat = _cutVerts.Count > 65535 ? IndexFormat.UInt32 : IndexFormat.UInt16;
            _cutMesh.SetVertices(_cutVerts);
            _cutMesh.SetNormals(_cutNormals);
            _cutMesh.SetTriangles(_cutIndices, 0);
            _cutMesh.bounds = liveBounds;
            _cutRenderer.enabled = _presentationVisible;
        }

        void EnsureUsableOuterUvs(Bounds liveBounds)
        {
            if (_outerUvs.Count == 0) return;

            Vector2 mn = new Vector2(float.MaxValue, float.MaxValue);
            Vector2 mx = new Vector2(float.MinValue, float.MinValue);
            for (int i = 0; i < _outerUvs.Count; i++)
            {
                Vector2 uv = _outerUvs[i];
                if (float.IsNaN(uv.x) || float.IsInfinity(uv.x) ||
                    float.IsNaN(uv.y) || float.IsInfinity(uv.y))
                {
                    mn = mx = Vector2.zero;
                    break;
                }
                mn = Vector2.Min(mn, uv);
                mx = Vector2.Max(mx, uv);
            }

            if ((mx - mn).sqrMagnitude > 1e-8f) return;

            Vector3 min = liveBounds.min;
            Vector3 size = liveBounds.size;
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

            for (int i = 0; i < _outerUvs.Count; i++)
            {
                Vector3 p = _outerVerts[i];
                float u = (Axis(p, uAxis) - Axis(min, uAxis)) / Mathf.Max(Axis(size, uAxis), 1e-6f);
                float v = (Axis(p, vAxis) - Axis(min, vAxis)) / Mathf.Max(Axis(size, vAxis), 1e-6f);
                _outerUvs[i] = new Vector2(u, v);
            }
        }

        void NormalizeOuterNormals()
        {
            _outerNormals.Clear();
            for (int i = 0; i < _outerNormalSums.Count; i++)
            {
                Vector3 n = _outerNormalSums[i];
                if (!IsFinite(n) || n.sqrMagnitude < 1e-12f) n = Vector3.up;
                else n.Normalize();
                _outerNormals.Add(n);
            }
        }

        void BuildOuterTangents()
        {
            _outerTangents.Clear();
            _tanSums.Clear();
            for (int i = 0; i < _outerVerts.Count; i++)
                _tanSums.Add(Vector3.zero);

            for (int i = 0; i + 2 < _outerIndices.Count; i += 3)
            {
                int i0 = _outerIndices[i];
                int i1 = _outerIndices[i + 1];
                int i2 = _outerIndices[i + 2];
                Vector3 p0 = _outerVerts[i0], p1 = _outerVerts[i1], p2 = _outerVerts[i2];
                Vector2 w0 = _outerUvs[i0], w1 = _outerUvs[i1], w2 = _outerUvs[i2];
                Vector3 e1 = p1 - p0;
                Vector3 e2 = p2 - p0;
                Vector2 d1 = w1 - w0;
                Vector2 d2 = w2 - w0;
                float det = d1.x * d2.y - d1.y * d2.x;
                if (Mathf.Abs(det) < 1e-8f) continue;

                Vector3 t = (e1 * d2.y - e2 * d1.y) / det;
                if (!IsFinite(t) || t.sqrMagnitude < 1e-12f) continue;
                AddTangent(i0, t);
                AddTangent(i1, t);
                AddTangent(i2, t);
            }

            for (int i = 0; i < _outerVerts.Count; i++)
            {
                Vector3 n = i < _outerNormals.Count ? _outerNormals[i] : Vector3.up;
                Vector3 t = _tanSums[i] - n * Vector3.Dot(n, _tanSums[i]);
                if (!IsFinite(t) || t.sqrMagnitude < 1e-12f) t = FallbackTangent(n);
                else t.Normalize();
                _outerTangents.Add(new Vector4(t.x, t.y, t.z, 1f));
            }
        }

        void AccumulateNormal(int index, Vector3 n)
        {
            _outerNormalSums[index] = _outerNormalSums[index] + n;
        }

        void AddTangent(int index, Vector3 t)
        {
            _tanSums[index] = _tanSums[index] + t;
        }

        static int Q(float v)
        {
            return Mathf.RoundToInt(v * Quant);
        }

        static Vector3 FallbackTangent(Vector3 n)
        {
            Vector3 axis = Mathf.Abs(n.y) < 0.9f ? Vector3.up : Vector3.right;
            Vector3 t = Vector3.Cross(axis, n);
            if (t.sqrMagnitude < 1e-12f) return Vector3.right;
            t.Normalize();
            return t;
        }

        static float Axis(Vector3 value, int axis)
        {
            return axis == 0 ? value.x : (axis == 1 ? value.y : value.z);
        }

        static bool IsFinite(Vector3 v)
        {
            return !(float.IsNaN(v.x) || float.IsInfinity(v.x) ||
                     float.IsNaN(v.y) || float.IsInfinity(v.y) ||
                     float.IsNaN(v.z) || float.IsInfinity(v.z));
        }

        void OnDestroy()
        {
            if (_outerMesh != null) Destroy(_outerMesh);
            if (_cutMesh != null) Destroy(_cutMesh);
        }
    }
}
