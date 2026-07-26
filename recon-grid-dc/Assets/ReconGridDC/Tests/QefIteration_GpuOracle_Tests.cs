// QefIteration_GpuOracle_Tests.cs — Task 6 Step 2 (non-stationary oracle)
// GPU oracle test: dispatches DC_FeaturePoints with a TILTED plane level set so the
// QEF gradient F is non-zero at x0, exercising the D5 last-interior-point iteration.
//
// A tilted plane φ(p) = dot(p - p0, n) with n = normalize(float3(1,1,0)) through
// the center of a single voxel [0,1]^3.  Four edges cross the plane at non-coincident
// points, so x0 = mean(p_i) ≠ feature point and the QEF must iterate.
//
// Asserts:
//   fp.w == 1.0               (valid)
//   |dot(fp.xyz - p0, n)| < tol  (feature point lies on plane)
//   fp has no NaN
//   fp.xyz ∈ [0,1]^3 within tol  (inside voxel — D5 held it in)

using NUnit.Framework;
using Unity.Mathematics;
using UnityEngine;
using ReconGridDC.Preprocess;
using ReconGridDC.Recon;

#if UNITY_EDITOR
using UnityEditor;
#endif

[Category("GPU")]
public class QefIteration_GpuOracle_Tests
{
    // Tilted plane: φ(p) = dot(p - p0, n), n = normalize(1,1,0)
    // p0 = center of voxel (0.5, 0.5, 0.5) so the plane passes through the voxel.
    sealed class TiltedPlane : ILevelSetProvider
    {
        static readonly float3 n  = math.normalize(new float3(1f, 1f, 0f));
        static readonly float3 p0 = new float3(0.5f, 0.5f, 0.5f);

        public float  Sample(float3 p)   => math.dot(p - p0, n);
        public float3 Gradient(float3 p) => n;
    }

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

    [Test]
    public void DC_FeaturePoints_TiltedPlane_FpOnPlaneAndInsideVoxel()
    {
        var cs = LoadReconShader();
        Assert.IsNotNull(cs, "Recon.compute must be loadable via AssetDatabase or Resources");

        // Build a 1-voxel grid [0,1]^3 with the tilted plane
        var g = BackgroundGrid.Build(new TiltedPlane(), new int3(1, 1, 1), 1f, float3.zero);
        Assert.AreEqual(1, g.voxelCount, "Expected exactly 1 voxel");
        // The tilted plane crosses at least 4 edges (x- and y-direction edges all cross it)
        Assert.GreaterOrEqual(g.voxelIsectCount[0], 4,
            "Tilted plane must produce >= 4 intersections so QEF must iterate");

        using (var rb = new ReconBuffers(g, 64))
        {
            rb.Upload(g);

            // RebuildIsectWorld must run BEFORE DC_FeaturePoints so _IsectWorld is populated.
            // At rest, CornerPos == rest positions => IsectWorld == globalRest (static 1A path).
            int kRebuild = cs.FindKernel("RebuildIsectWorld");
            Assert.GreaterOrEqual(kRebuild, 0, "Kernel RebuildIsectWorld not found in Recon.compute");
            cs.SetInt("_IsectCount", rb.IsectCount);
            cs.SetBuffer(kRebuild, "_Isect",      rb.Isect);
            cs.SetBuffer(kRebuild, "_CornerPos",  rb.CornerPos);
            cs.SetBuffer(kRebuild, "_IsectWorld", rb.IsectWorld);
            cs.Dispatch(kRebuild, Mathf.CeilToInt(rb.IsectCount / 64f), 1, 1);

            int k = cs.FindKernel("DC_FeaturePoints");
            Assert.GreaterOrEqual(k, 0, "Kernel DC_FeaturePoints not found in Recon.compute");

            // Set uniforms
            cs.SetInt("_VoxelCount", rb.VoxelCount);
            cs.SetFloat("_L", 1f);
            cs.SetInt("_QefIters", 20);

            // Bind the 7 required buffers (includes _IsectWorld populated by RebuildIsectWorld above)
            cs.SetBuffer(k, "_CornerPos",        rb.CornerPos);
            cs.SetBuffer(k, "_VoxelCorner",      rb.VoxelCorner);
            cs.SetBuffer(k, "_VoxelIsectOffset", rb.VoxelIsectOffset);
            cs.SetBuffer(k, "_VoxelIsectCount",  rb.VoxelIsectCount);
            cs.SetBuffer(k, "_Isect",            rb.Isect);
            cs.SetBuffer(k, "_IsectWorld",       rb.IsectWorld);
            cs.SetBuffer(k, "_VoxelExternalFP",  rb.VoxelExternalFP);

            // Dispatch: 1 group of 64 threads; only thread 0 does work
            cs.Dispatch(k, Mathf.CeilToInt(rb.VoxelCount / 64f), 1, 1);

            // Readback
            var fp = new float4[rb.VoxelCount];
            rb.VoxelExternalFP.GetData(fp);

            float4 p = fp[0];
            float3 n  = math.normalize(new float3(1f, 1f, 0f));
            float3 p0 = new float3(0.5f, 0.5f, 0.5f);
            const float tol = 1e-2f;

            // fp.w == 1 (valid)
            Assert.AreEqual(1f, p.w, tol, "Feature point must be marked valid (w=1)");

            // No NaN
            Assert.IsFalse(float.IsNaN(p.x), "fp.x must not be NaN");
            Assert.IsFalse(float.IsNaN(p.y), "fp.y must not be NaN");
            Assert.IsFalse(float.IsNaN(p.z), "fp.z must not be NaN");

            // Feature point lies ON the tilted plane (|φ(fp.xyz)| < tol)
            float onPlane = math.abs(math.dot(new float3(p.x, p.y, p.z) - p0, n));
            Assert.Less(onPlane, tol,
                $"Feature point must lie on plane dot(fp-p0,n)~0, got {onPlane}");

            // Feature point is INSIDE the voxel [0,1]^3 (D5: BoxSdf <= 0)
            Assert.GreaterOrEqual(p.x, -tol, "fp.x must be >= 0 (inside voxel)");
            Assert.LessOrEqual(p.x, 1f + tol, "fp.x must be <= 1 (inside voxel)");
            Assert.GreaterOrEqual(p.y, -tol, "fp.y must be >= 0 (inside voxel)");
            Assert.LessOrEqual(p.y, 1f + tol, "fp.y must be <= 1 (inside voxel)");
            Assert.GreaterOrEqual(p.z, -tol, "fp.z must be >= 0 (inside voxel)");
            Assert.LessOrEqual(p.z, 1f + tol, "fp.z must be <= 1 (inside voxel)");
        }
    }
}
