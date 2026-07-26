// DeformCoupling_GpuOracle_Tests.cs — Stage 1B-ii Task 3
// Integration smoke oracle: runs the full per-frame coupling pipeline (solver.Step → dc.Build)
// under gravity + pinned-top conditions and asserts:
//   (a) No NaN or Infinity in any valid VoxelExternalFP.
//   (b) The minimum Y among valid feature points decreased (surface sagged).
//   (c) All valid feature points stay within 10× the grid extent of the center (no explosion).
//
// [Category("GPU")] — requires a real GPU and Unity player loop context; skip in CI without one.

using NUnit.Framework;
using Unity.Mathematics;
using UnityEngine;
using ReconGridDC.Preprocess;
using ReconGridDC.Recon;
using ReconGridDC.Physics;
using ReconGridDC.Core;

#if UNITY_EDITOR
using UnityEditor;
#endif

[Category("GPU")]
public class DeformCoupling_GpuOracle_Tests
{
    // ── Shader loaders ──────────────────────────────────────────────────────────

    static ComputeShader LoadReconShader()
    {
#if UNITY_EDITOR
        foreach (var guid in AssetDatabase.FindAssets("Recon t:ComputeShader"))
        {
            var path = AssetDatabase.GUIDToAssetPath(guid);
            if (path.EndsWith("/Recon.compute"))
            {
                var cs = AssetDatabase.LoadAssetAtPath<ComputeShader>(path);
                if (cs != null) return cs;
            }
        }
#endif
        return Resources.Load<ComputeShader>("Recon");
    }

    static ComputeShader LoadPhysicsShader()
    {
#if UNITY_EDITOR
        foreach (var guid in AssetDatabase.FindAssets("Physics t:ComputeShader"))
        {
            var path = AssetDatabase.GUIDToAssetPath(guid);
            if (path.EndsWith("/Physics.compute"))
            {
                var cs = AssetDatabase.LoadAssetAtPath<ComputeShader>(path);
                if (cs != null) return cs;
            }
        }
#endif
        return Resources.Load<ComputeShader>("Physics");
    }

    // ── Helper: minimum Y among valid feature points (w > 0.5) ─────────────────

    static float MinValidY(ReconBuffers rb)
    {
        var fp = new float4[rb.VoxelCount];
        rb.VoxelExternalFP.GetData(fp);

        float minY = float.MaxValue;
        foreach (var v in fp)
        {
            if (v.w > 0.5f && v.y < minY)
                minY = v.y;
        }
        // If no valid FP found, return +MaxValue (test will catch it via sag assertion).
        return minY;
    }

    // ── Integration smoke: gravity + pinned-top → surface sags, all FPs finite ──

    /// <summary>
    /// Builds a sphere grid (dims=12, L=0.25, center=(1.5,1.5,1.5), r=1.0),
    /// pins the top y-layer, sets gravity, runs 10 frames of solver.Step + dc.Build,
    /// then asserts all valid FPs are finite and the surface has sagged.
    /// </summary>
    [Test]
    public void GravityPin_SurfaceSagsAndStaysFinite()
    {
        var recon   = LoadReconShader();
        var physics = LoadPhysicsShader();

        Assert.IsNotNull(recon,   "Recon.compute must be loadable via AssetDatabase or Resources");
        Assert.IsNotNull(physics, "Physics.compute must be loadable via AssetDatabase or Resources");

        // ── Build grid ───────────────────────────────────────────────────────────
        var center = new float3(1.5f, 1.5f, 1.5f);
        var dimsi  = new int3(12, 12, 12);
        const float L = 0.25f;

        var g = BackgroundGrid.Build(
            new SphereLevelSet(center, 1.0f),
            dimsi,
            L,
            float3.zero);

        // ── Pin the top corner layer (j = dims.y = 12) ──────────────────────────
        for (int k = 0; k <= g.dims.z; k++)
        for (int i = 0; i <= g.dims.x; i++)
        {
            int c = GridConventions.CornerId(i, g.dims.y, k, g.dims);
            g.pinned[c] = 1;
        }

        int triCap = 6 * g.surfaceEdges.Length + 1024;
        using (var rb = new ReconBuffers(g, triCap))
        {
            rb.Upload(g);

            // ── Set constant gravity ExtForce: (0, -9.81*mass, 0) per corner ─────
            var ext = new float3[g.cornerCount];
            for (int c = 0; c < g.cornerCount; c++)
                ext[c] = new float3(0f, -9.81f * g.mass[c], 0f);
            rb.ExtForce.SetData(ext);

            // ── Create solver (PHYSICS shader) + DC (RECON shader) ───────────────
            // Soft spring (Ks=500, Kb=100): gravity produces clearly-visible sag within the
            // 40-frame budget so the strict 1e-3 threshold is reliably exceeded.
            // A stiff spring (e.g. Ks=7.5e4) sags sub-threshold and cannot distinguish a
            // live coupling from a completely static surface.
            var dc = new DualContouring(recon);
            var solver = new MassSpringSolver(physics)
            {
                Ks = 500f,
                Cs = 0.92f,
                Kb = 100f,
                Cb = 0.9f,
            };

            // ── Initial DC build (at rest) ────────────────────────────────────────
            dc.Build(rb, g.dims, L, 20);
            float minY0 = MinValidY(rb);
            Assert.AreNotEqual(float.MaxValue, minY0,
                "Initial DC build must produce at least one valid feature point");

            // ── Run 40 per-frame coupling steps ──────────────────────────────────
            // ORDER: solver.Step THEN dc.Build (physics moves corners, DC reads moved corners).
            // 40 frames × dt=0.02 s = 0.8 s of simulated time — enough for soft-spring sag
            // to comfortably exceed the 1e-3 m strict threshold used in Assert (b) below.
            for (int f = 0; f < 40; f++)
            {
                solver.Step(rb, 0.02f);
                dc.Build(rb, g.dims, L, 20);
            }

            // ── Assert (a): no NaN or Inf in any valid feature point ─────────────
            var fp = new float4[rb.VoxelCount];
            rb.VoxelExternalFP.GetData(fp);
            foreach (var v in fp)
            {
                if (v.w > 0.5f)
                {
                    Assert.IsFalse(float.IsNaN(v.x) || float.IsInfinity(v.x),
                        $"Valid FP x is NaN/Inf: {v}");
                    Assert.IsFalse(float.IsNaN(v.y) || float.IsInfinity(v.y),
                        $"Valid FP y is NaN/Inf: {v}");
                    Assert.IsFalse(float.IsNaN(v.z) || float.IsInfinity(v.z),
                        $"Valid FP z is NaN/Inf: {v}");
                }
            }

            // ── Assert (b): surface sagged — min Y strictly decreased by > 1e-3 ──
            // The soft spring ensures gravity produces observable sag well above noise.
            // A static (broken) coupling would leave minY1 == minY0, failing this check.
            float minY1 = MinValidY(rb);
            Assert.Less(minY1, minY0 - 1e-3f,
                $"surface must visibly sag under gravity (min Y strictly decreases). " +
                $"Initial minY={minY0:F6}, after 40 frames minY={minY1:F6}");

            // ── Assert (c): no explosion — all valid FPs within 10× grid extent ──
            float3 gridExtent = (float3)g.dims * L;
            float  bound      = math.cmax(gridExtent) * 10f;  // 10× the largest grid dimension
            foreach (var v in fp)
            {
                if (v.w > 0.5f)
                {
                    float3 delta = new float3(v.x, v.y, v.z) - center;
                    float  dist  = math.length(delta);
                    Assert.Less(dist, bound,
                        $"Valid FP {new float3(v.x, v.y, v.z)} is more than {bound:F2} units from grid center (explosion?)");
                }
            }
        }
    }
}
