// SmoothedSdfGrid.cs — liver tuning, design Fix 2 (surface smoothness) + R3 (fast startup).
// Wraps an expensive ILevelSetProvider (MeshLevelSet) by baking its signed field ONCE onto a dense
// scalar grid covering the full reconstruction domain, smoothing the field (Taubin λ|μ to limit
// shrinkage), then serving cheap trilinear samples + normalized central-difference gradients.
//   - Removes the high-frequency roughness of the source tet-boundary → a smoother DC surface.
//   - BackgroundGrid.Build then samples cheap trilinear lookups; its per-CROSSING Gradient calls (which
//     were brute-force closest-tri) become O(1). NOTE (honesty): the bake itself still pays the
//     brute-force baseLs.Sample ONCE per bake node (≈ the corner count), so startup is only MODESTLY
//     improved, NOT eliminated — a spatial-accel/BVH on MeshLevelSet (liver-import design R3) is the
//     real startup fix and remains a follow-up.
// The bake grid spans [origin, origin + reconDims·L] (the reconstruction domain incl. the margin shell)
// so every corner sample AND every isosurface-crossing gradient BackgroundGrid requests has a valid
// stencil; queries are clamped to the grid border.

using Unity.Mathematics;
using UnityEngine;

namespace ReconGridDC.Preprocess
{
    public sealed class SmoothedSdfGrid : ILevelSetProvider
    {
        readonly float3 _origin;
        readonly float  _h;          // node spacing (= reconL / oversample)
        readonly int3   _nd;         // node count per axis
        readonly float[] _phi;       // flat [i + nd.x*(j + nd.y*k)]

        // Taubin pass factors (low shrinkage): one iteration = a +λ pass then a −μ pass.
        const float Lambda = 0.5f;
        const float Mu     = -0.53f;

        /// <param name="baseLs">expensive source SDF (e.g. MeshLevelSet)</param>
        /// <param name="origin">reconstruction grid origin (GridFit)</param>
        /// <param name="reconDims">reconstruction voxel dims (GridFit)</param>
        /// <param name="reconL">reconstruction voxel size (GridFit)</param>
        /// <param name="oversample">bake nodes per recon voxel (≥1; 1 = recon-corner-aligned)</param>
        /// <param name="smoothIters">Taubin iterations (0 = bake only, no smoothing)</param>
        public SmoothedSdfGrid(ILevelSetProvider baseLs, float3 origin, int3 reconDims, float reconL,
                               int oversample, int smoothIters)
        {
            oversample = math.max(1, oversample);
            _origin = origin;
            _h = reconL / oversample;
            _nd = reconDims * oversample + new int3(1, 1, 1);   // nodes span [origin, origin+reconDims*L]

            // Memory guard (design Fix 2): cap the bake grid (~96^3 floats ≈ 3.5 MB).
            long nodes = (long)_nd.x * _nd.y * _nd.z;
            if (nodes > 96L * 96 * 96)
                Debug.LogWarning($"[SmoothedSdfGrid] bake grid {_nd} = {nodes} nodes is large (>96^3); lower oversample/targetLongAxisVoxels.");

            _phi = new float[nodes];
            for (int k = 0; k < _nd.z; k++)
            for (int j = 0; j < _nd.y; j++)
            for (int i = 0; i < _nd.x; i++)
                _phi[Idx(i, j, k)] = baseLs.Sample(_origin + new float3(i, j, k) * _h);

            for (int it = 0; it < smoothIters; it++)
            {
                LaplacianPass(Lambda);   // shrink
                LaplacianPass(Mu);       // un-shrink (Taubin)
            }
        }

        int Idx(int i, int j, int k) => i + _nd.x * (j + _nd.y * k);

        // ── ILevelSetProvider ─────────────────────────────────────────────────────────────────
        public float Sample(float3 p) => Trilinear(p);

        public float3 Gradient(float3 p)
        {
            float e = 0.5f * _h;
            float gx = Trilinear(p + new float3(e, 0, 0)) - Trilinear(p - new float3(e, 0, 0));
            float gy = Trilinear(p + new float3(0, e, 0)) - Trilinear(p - new float3(0, e, 0));
            float gz = Trilinear(p + new float3(0, 0, e)) - Trilinear(p - new float3(0, 0, e));
            float3 g = new float3(gx, gy, gz);
            float len = math.length(g);
            if (len < 1e-8f) return new float3(0, 1, 0);   // degenerate (flat region) → safe unit
            return g / len;                                 // unit outward (ILevelSetProvider contract)
        }

        // ── Trilinear sample with border clamp ────────────────────────────────────────────────
        float Trilinear(float3 p)
        {
            float3 g = (p - _origin) / _h;
            // clamp to [0, nd-1) so the +1 stencil stays in-bounds
            g = math.clamp(g, float3.zero, (float3)(_nd - 1) - 1e-4f);
            int3 c0 = (int3)math.floor(g);
            float3 f = g - (float3)c0;
            int i0 = c0.x, j0 = c0.y, k0 = c0.z;
            int i1 = i0 + 1, j1 = j0 + 1, k1 = k0 + 1;

            float c000 = _phi[Idx(i0, j0, k0)], c100 = _phi[Idx(i1, j0, k0)];
            float c010 = _phi[Idx(i0, j1, k0)], c110 = _phi[Idx(i1, j1, k0)];
            float c001 = _phi[Idx(i0, j0, k1)], c101 = _phi[Idx(i1, j0, k1)];
            float c011 = _phi[Idx(i0, j1, k1)], c111 = _phi[Idx(i1, j1, k1)];

            float c00 = math.lerp(c000, c100, f.x), c10 = math.lerp(c010, c110, f.x);
            float c01 = math.lerp(c001, c101, f.x), c11 = math.lerp(c011, c111, f.x);
            float c0_ = math.lerp(c00, c10, f.y),   c1_ = math.lerp(c01, c11, f.y);
            return math.lerp(c0_, c1_, f.z);
        }

        // ── Smoothness metric (test/diagnostic): variance of the discrete Laplacian Δφ = avg6 − φ over
        //    interior nodes. Smoothing should reduce this monotonically with smoothIters. design Testing #3.
        public float LaplacianVariance()
        {
            double sum = 0, sum2 = 0; long n = 0;
            for (int k = 1; k < _nd.z - 1; k++)
            for (int j = 1; j < _nd.y - 1; j++)
            for (int i = 1; i < _nd.x - 1; i++)
            {
                float c = _phi[Idx(i, j, k)];
                float avg = (_phi[Idx(i - 1, j, k)] + _phi[Idx(i + 1, j, k)] +
                             _phi[Idx(i, j - 1, k)] + _phi[Idx(i, j + 1, k)] +
                             _phi[Idx(i, j, k - 1)] + _phi[Idx(i, j, k + 1)]) * (1.0f / 6.0f);
                double lap = avg - c;
                sum += lap; sum2 += lap * lap; n++;
            }
            if (n == 0) return 0f;
            double mean = sum / n;
            return (float)(sum2 / n - mean * mean);
        }

        // ── Laplacian smoothing pass (interior nodes only; border = baked, it's the empty shell) ─
        void LaplacianPass(float factor)
        {
            var dst = new float[_phi.Length];
            System.Array.Copy(_phi, dst, _phi.Length);
            for (int k = 1; k < _nd.z - 1; k++)
            for (int j = 1; j < _nd.y - 1; j++)
            for (int i = 1; i < _nd.x - 1; i++)
            {
                float c = _phi[Idx(i, j, k)];
                float avg = (_phi[Idx(i - 1, j, k)] + _phi[Idx(i + 1, j, k)] +
                             _phi[Idx(i, j - 1, k)] + _phi[Idx(i, j + 1, k)] +
                             _phi[Idx(i, j, k - 1)] + _phi[Idx(i, j, k + 1)]) * (1.0f / 6.0f);
                dst[Idx(i, j, k)] = c + factor * (avg - c);
            }
            System.Array.Copy(dst, _phi, _phi.Length);
        }
    }
}
