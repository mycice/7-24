// MeshLevelSet.cs — liver-import pipeline, design §2.
// ILevelSetProvider built from a boundary triangulation (MshSurface). Feeds BackgroundGrid.Build,
// which samples Sample()/Gradient() at every corner ONCE at preprocess.
//
//   Unsigned distance : min closest-point-on-triangle over all boundary tris (Ericson), reporting the
//                        closest FEATURE (face / edge / vertex).
//   Sign (watertight)  : angle-weighted pseudonormal (Bærentzen & Aanæs 2005) at the closest feature —
//                          face   → face normal
//                          edge   → mean of its 2 incident face normals
//                          vertex → incident-ANGLE-weighted sum of incident face normals  (the crux)
//                        sign = sign(dot(p − closest, n_pseudo)).
//   Sign (fallback)    : if the boundary is NOT watertight, generalized winding number (Jacobson 2013 /
//                        libigl) — robust to holes/non-manifold defects. inside ⇔ |w| > 0.5.
//   Sample(p)          = sign * unsignedDistance.
//   Gradient(p)        = sign * normalize(p − closest)  (outward both sides — matches SphereLevelSet);
//                        epsilon fallback to the feature pseudonormal when p is ON the surface (NaN guard;
//                        BackgroundGrid.cs:146 calls Gradient EXACTLY at the crossing point).
//
// PERFORMANCE (design R3): closest-point is BRUTE-FORCE (O(tris) per query). For a one-time preprocess
// over ~tens of thousands of corners this is borderline-but-tolerable (seconds). A spatial-hash / BVH
// broadphase is the recommended optimization (a drop-in replacement for FindClosest); not implemented
// here to keep the correctness baseline simple. design §2 / R3.

using System;
using System.Collections.Generic;
using Unity.Mathematics;
using UnityEngine;

namespace ReconGridDC.Preprocess
{
    public sealed class MeshLevelSet : ILevelSetProvider
    {
        readonly float3[] _verts;
        readonly int3[]   _tris;
        readonly float3[] _faceNormal;          // outward unit normal per boundary tri
        readonly float3[] _vertPseudoNormal;    // angle-weighted vertex pseudonormal (normalized), per vert
        readonly Dictionary<long, int2> _edgeFaces;  // sorted-edge-key → (faceA, faceB); -1 if absent
        readonly bool _watertight;

        const long KEY_MULT = 4194304L;         // 2^22 > any vertex index (matches MshLoader)
        const float ON_SURFACE_EPS = 1e-8f;

        public MeshLevelSet(MshSurface surf)
        {
            _verts = surf.verts;
            _tris  = surf.tris;

            // Face normals (outward by construction — MshLoader winds away from the tet's opposite vertex).
            _faceNormal = new float3[_tris.Length];
            for (int i = 0; i < _tris.Length; i++)
            {
                int3 t = _tris[i];
                float3 n = math.cross(_verts[t.y] - _verts[t.x], _verts[t.z] - _verts[t.x]);
                float len = math.length(n);
                _faceNormal[i] = len > 1e-20f ? n / len : new float3(0, 1, 0);
            }

            // Edge → incident faces (≤2). Reused for the watertight check AND the edge pseudonormal.
            _edgeFaces = new Dictionary<long, int2>(_tris.Length * 2);
            for (int i = 0; i < _tris.Length; i++)
            {
                int3 t = _tris[i];
                AddEdgeFace(t.x, t.y, i);
                AddEdgeFace(t.y, t.z, i);
                AddEdgeFace(t.z, t.x, i);
            }
            _watertight = CheckWatertight();
            if (!_watertight)
                Debug.LogWarning("[MeshLevelSet] boundary is NOT watertight → using generalized-winding-number sign (defect-tolerant fallback).");

            // Vertex pseudonormal = normalize(Σ angle_f · normal_f) over incident faces (angle = interior
            // angle of face f at this vertex). The angle weighting is what makes the vertex case correct.
            var acc = new float3[_verts.Length];
            for (int i = 0; i < _tris.Length; i++)
            {
                int3 t = _tris[i];
                acc[t.x] += AngleAt(_verts[t.x], _verts[t.y], _verts[t.z]) * _faceNormal[i];
                acc[t.y] += AngleAt(_verts[t.y], _verts[t.z], _verts[t.x]) * _faceNormal[i];
                acc[t.z] += AngleAt(_verts[t.z], _verts[t.x], _verts[t.y]) * _faceNormal[i];
            }
            _vertPseudoNormal = new float3[_verts.Length];
            for (int v = 0; v < _verts.Length; v++)
            {
                float len = math.length(acc[v]);
                _vertPseudoNormal[v] = len > 1e-20f ? acc[v] / len : new float3(0, 1, 0);
            }
        }

        // ── ILevelSetProvider ─────────────────────────────────────────────────────────────────
        public float Sample(float3 p)
        {
            FindClosest(p, out float3 closest, out float3 nPseudo);
            float dist = math.distance(p, closest);
            float sign = SignAt(p, closest, nPseudo);
            return sign * dist;
        }

        public float3 Gradient(float3 p)
        {
            FindClosest(p, out float3 closest, out float3 nPseudo);
            float3 d = p - closest;
            float len = math.length(d);
            if (len < ON_SURFACE_EPS)
                return nPseudo;                      // p ON the surface → use the feature pseudonormal (outward)
            float sign = SignAt(p, closest, nPseudo);
            return sign * (d / len);                 // outward SDF gradient (both inside & outside)
        }

        float SignAt(float3 p, float3 closest, float3 nPseudo)
        {
            if (_watertight)
                return math.dot(p - closest, nPseudo) >= 0f ? 1f : -1f;
            return WindingNumberInside(p) ? -1f : 1f;     // winding-number fallback
        }

        // ── Closest point (brute force) ─────────────────────────────────────────────────────
        // Reports the closest point + its pseudonormal (feature-dependent). O(tris) per call (R3).
        void FindClosest(float3 p, out float3 closest, out float3 nPseudo)
        {
            float bestD2 = float.PositiveInfinity;
            float3 bestPt = default;
            int bestTri = 0, bestFeat = 0, bestSub = 0;
            for (int i = 0; i < _tris.Length; i++)
            {
                int3 t = _tris[i];
                float3 cp = ClosestOnTri(p, _verts[t.x], _verts[t.y], _verts[t.z], out int feat, out int sub);
                float d2 = math.distancesq(p, cp);
                if (d2 < bestD2) { bestD2 = d2; bestPt = cp; bestTri = i; bestFeat = feat; bestSub = sub; }
            }
            closest = bestPt;
            nPseudo = PseudoNormal(bestTri, bestFeat, bestSub);
        }

        float3 PseudoNormal(int tri, int feat, int sub)
        {
            if (feat == 0) return _faceNormal[tri];                         // FACE
            int3 t = _tris[tri];
            if (feat == 2)                                                  // VERTEX
                return _vertPseudoNormal[sub == 0 ? t.x : sub == 1 ? t.y : t.z];
            // EDGE: sub 0=(x,y) 1=(y,z) 2=(z,x). Mean of the 2 incident face normals.
            int u = sub == 0 ? t.x : sub == 1 ? t.y : t.z;
            int w = sub == 0 ? t.y : sub == 1 ? t.z : t.x;
            if (_edgeFaces.TryGetValue(EdgeKey(u, w), out int2 fs))
            {
                float3 n = _faceNormal[fs.x];
                if (fs.y >= 0) n += _faceNormal[fs.y];
                float len = math.length(n);
                if (len > 1e-20f) return n / len;
            }
            return _faceNormal[tri];                                        // degenerate fallback
        }

        // Ericson, Real-Time Collision Detection §5.1.5. Returns the closest point and the closest
        // feature: feat 0=face, 1=edge, 2=vertex; sub = vertex 0/1/2 (a/b/c) or edge 0(ab)/1(bc)/2(ca).
        static float3 ClosestOnTri(float3 p, float3 a, float3 b, float3 c, out int feat, out int sub)
        {
            float3 ab = b - a, ac = c - a, ap = p - a;
            float d1 = math.dot(ab, ap), d2 = math.dot(ac, ap);
            if (d1 <= 0f && d2 <= 0f) { feat = 2; sub = 0; return a; }                 // vertex A

            float3 bp = p - b;
            float d3 = math.dot(ab, bp), d4 = math.dot(ac, bp);
            if (d3 >= 0f && d4 <= d3) { feat = 2; sub = 1; return b; }                 // vertex B

            float vc = d1 * d4 - d3 * d2;
            if (vc <= 0f && d1 >= 0f && d3 <= 0f) { feat = 1; sub = 0; return a + (d1 / (d1 - d3)) * ab; }  // edge AB

            float3 cp = p - c;
            float d5 = math.dot(ab, cp), d6 = math.dot(ac, cp);
            if (d6 >= 0f && d5 <= d6) { feat = 2; sub = 2; return c; }                 // vertex C

            float vb = d5 * d2 - d1 * d6;
            if (vb <= 0f && d2 >= 0f && d6 <= 0f) { feat = 1; sub = 2; return a + (d2 / (d2 - d6)) * ac; }  // edge AC (CA)

            float va = d3 * d6 - d5 * d4;
            if (va <= 0f && (d4 - d3) >= 0f && (d5 - d6) >= 0f)                        // edge BC
            {
                feat = 1; sub = 1;
                float w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
                return b + w * (c - b);
            }

            feat = 0; sub = 0;                                                         // face interior
            float denom = 1f / (va + vb + vc);
            return a + ab * (vb * denom) + ac * (vc * denom);
        }

        // ── Winding-number fallback (Van Oosterom–Strackee solid angle) ───────────────────────
        bool WindingNumberInside(float3 p)
        {
            double sum = 0.0;
            for (int i = 0; i < _tris.Length; i++)
            {
                int3 t = _tris[i];
                float3 av = _verts[t.x] - p, bv = _verts[t.y] - p, cv = _verts[t.z] - p;
                float la = math.length(av), lb = math.length(bv), lc = math.length(cv);
                float num = math.determinant(new float3x3(av, bv, cv));   // av · (bv × cv)
                float den = la * lb * lc + math.dot(av, bv) * lc + math.dot(bv, cv) * la + math.dot(cv, av) * lb;
                sum += 2.0 * math.atan2(num, den);
            }
            double w = sum / (4.0 * System.Math.PI);
            return System.Math.Abs(w) > 0.5;
        }

        // ── Adjacency build helpers ────────────────────────────────────────────────────────
        void AddEdgeFace(int u, int v, int face)
        {
            long key = EdgeKey(u, v);
            if (_edgeFaces.TryGetValue(key, out int2 fs))
            {
                if (fs.y < 0) fs.y = face;     // second incident face (watertight ⇒ exactly 2)
                _edgeFaces[key] = fs;
            }
            else _edgeFaces[key] = new int2(face, -1);
        }

        bool CheckWatertight()
        {
            foreach (var kv in _edgeFaces)
                if (kv.Value.y < 0) return false;   // an edge with <2 incident faces ⇒ open/non-manifold
            return true;
        }

        static long EdgeKey(int u, int v)
        {
            int lo = math.min(u, v), hi = math.max(u, v);
            return lo * KEY_MULT + hi;
        }

        // Interior angle of a triangle at vertex `at`, given its two other vertices.
        static float AngleAt(float3 at, float3 o1, float3 o2)
        {
            float3 e1 = math.normalizesafe(o1 - at), e2 = math.normalizesafe(o2 - at);
            return math.acos(math.clamp(math.dot(e1, e2), -1f, 1f));
        }
    }
}
