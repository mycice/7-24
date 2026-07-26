// Stitch_GpuOracle_Tests.cs — Tasks 7 Step 3 (deferred) + 8 Step 2 (deferred)
//
// These tests were deferred from C2 because they depend on DualContouring (Task 9).
// They exercise DC_Stitch, DC_ClearNormals, DC_NormalsScatter, DC_NormalsNormalize,
// and DC_WriteIndirectArgs via the DualContouring dispatcher.
//
// Category [GPU]: requires a GPU machine + Unity Test Runner (EditMode).
// Load cs via AssetDatabase.LoadAssetAtPath (mirrors Qef_GpuOracle_Tests pattern).
//
// Tests:
//   1. DC_Stitch_Sphere_ProducesClosedTriangleSet
//      Sphere-in-padded-grid smoke test: TriCounter>0, count%3==0, all indices in [0,VoxelCount).
//
//   2. DC_Normals_Sphere_PointOutward
//      Key winding/outward-normal validation (deferred from C2 Task 8 Step 2).
//      For each valid voxel: dot(normalize(fp.xyz - sphereCenter), normalize(normal)) > 0.3.
//      Checks > 10 voxels.

using NUnit.Framework;
using Unity.Mathematics;
using UnityEngine;
using ReconGridDC.Preprocess;
using ReconGridDC.Recon;

#if UNITY_EDITOR
using UnityEditor;
#endif

[Category("GPU")]
public class Stitch_GpuOracle_Tests
{
    // ── Shared helpers ────────────────────────────────────────────────────────

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

    /// <summary>
    /// Builds a sphere in a 10x10x10 grid (L=0.25, origin=0) with the sphere centered at
    /// (1.25,1.25,1.25) r=0.8 — strictly inside, ≥1 voxel of padding on each side.
    /// </summary>
    static (BackgroundGrid g, float3 center) BuildSphereGrid()
    {
        var center = new float3(1.25f, 1.25f, 1.25f);
        var ls     = new SphereLevelSet(center, 0.8f);
        var g      = BackgroundGrid.Build(ls, new int3(10, 10, 10), 0.25f, float3.zero);
        return (g, center);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Test 1: DC_Stitch smoke test (Task 7 Step 3 — deferred from C2)
    //
    // Verifies:
    //   • TriCounter > 0  (surface was stitched)
    //   • TriCounter % 3 == 0  (only whole triangles)
    //   • Every (idx & 0x3FFFFFFF) is in [0, VoxelCount)  (no out-of-range voxel refs; bit31=cut FP, no high bit=external)
    // ─────────────────────────────────────────────────────────────────────────
    [Test]
    public void DC_Stitch_Sphere_ProducesClosedTriangleSet()
    {
        var cs = LoadReconShader();
        Assert.IsNotNull(cs, "Recon.compute must be loadable via AssetDatabase or Resources");

        var (g, _) = BuildSphereGrid();

        using (var rb = new ReconBuffers(g, 200000))
        {
            rb.Upload(g);

            // Run the full DualContouring pipeline (FP + stitch + normals + indirect args)
            new DualContouring(cs).Build(rb, g.dims, 0.25f, 20);

            // Read back triangle counter
            var cntBuf = new uint[1];
            rb.TriCounter.GetData(cntBuf);
            uint triIndexCount = cntBuf[0];

            Assert.Greater(triIndexCount, 0u,
                "DC_Stitch must produce at least one triangle for a sphere");

            Assert.AreEqual(0u, triIndexCount % 3u,
                "TriCounter must be a multiple of 3 (whole triangles only)");

            // Read back indices and range-check them
            int readCount = (int)Mathf.Min(triIndexCount, (uint)(3 * rb.TriCapacity));
            var idx = new int[readCount];
            rb.Tri.GetData(idx, 0, 0, readCount);

            foreach (int raw in idx)
            {
                // TWO-WAY tag scheme: mask 0x3FFFFFFF strips bit31 (cut FP, skin+wall shared); the masked
                // low bits are the slot (8*v+comp) for a cut tag or the voxelId for an external tag. In this
                // ZERO-CUT path no high bit is ever set, so the masked value is the voxelId.
                int v = raw & 0x3FFFFFFF;
                Assert.GreaterOrEqual(v, 0,
                    $"Decoded voxel index {v} (raw={raw}) must be >= 0");
                Assert.Less(v, rb.VoxelCount,
                    $"Decoded voxel index {v} (raw={raw}) must be < VoxelCount={rb.VoxelCount}");

                // 2C-fix (plan §7 zero-cut regression): with NO cut (VoxelCutMask all-zero), DC_Stitch MUST
                // take the else branch → EVERY vertex is an EXTERNAL tag (bit31 clear). A future branch-
                // predicate regression that selected a cut FP (_CutFP) here would flip bit31.
                bool internalTag = (raw & unchecked((int)0x80000000)) != 0;
                Assert.IsFalse(internalTag,
                    $"Zero-cut DC_Stitch must emit only external tags (bit31 clear); raw={raw} had bit31 set");
            }
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Test 2: DC_Normals_Sphere_PointOutward (Task 8 Step 2 — deferred from C2)
    //
    // Key winding / outward-normal validation.
    // For every valid voxel (fp.w > 0.5 and |normal| > 0.5):
    //   dot(normalize(fp.xyz - sphereCenter), normalize(normal)) > 0.3  (outward)
    // Asserts that > 10 voxels were checked.
    // ─────────────────────────────────────────────────────────────────────────
    [Test]
    public void DC_Normals_Sphere_PointOutward()
    {
        var cs = LoadReconShader();
        Assert.IsNotNull(cs, "Recon.compute must be loadable via AssetDatabase or Resources");

        var (g, sphereCenter) = BuildSphereGrid();

        using (var rb = new ReconBuffers(g, 200000))
        {
            rb.Upload(g);

            // Run the full DualContouring pipeline (FP + stitch + normals + indirect args)
            new DualContouring(cs).Build(rb, g.dims, 0.25f, 20);

            // Read back feature points and normalized normals
            var fp = new float4[rb.VoxelCount];
            rb.VoxelExternalFP.GetData(fp);

            var nf = new float3[rb.VoxelCount];
            rb.VoxelExternalFPNormalF.GetData(nf);

            int checked_ = 0;
            for (int v = 0; v < rb.VoxelCount; v++)
            {
                // Skip voxels with no feature point or no normal
                if (fp[v].w < 0.5f) continue;
                float nLen = math.length(nf[v]);
                if (nLen < 0.5f) continue;

                float3 radial = math.normalize(fp[v].xyz - sphereCenter);
                float  d      = math.dot(radial, math.normalize(nf[v]));

                Assert.Greater(d, 0.3f,
                    $"voxel {v}: normal not outward — dot(radial, n) = {d:F4} " +
                    $"fp=({fp[v].x:F3},{fp[v].y:F3},{fp[v].z:F3}) " +
                    $"n=({nf[v].x:F3},{nf[v].y:F3},{nf[v].z:F3})");

                checked_++;
            }

            Assert.Greater(checked_, 10,
                "Expected > 10 valid surface voxels on the sphere to verify normal direction; " +
                $"only {checked_} found — check sphere placement / grid size");
        }
    }
}
