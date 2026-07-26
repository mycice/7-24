// Qef_GpuOracle_Tests.cs — Task 6 Step 2
// GPU oracle test: dispatches DC_FeaturePoints directly (self-contained, no DualContouring helper).
// Verified by running in Unity EditMode on a GPU machine.
// A plane at x=0.5 in a single-voxel grid [0,1]^3:
//   - 4 x-direction isect points, all at x=0.5
//   - QEF should converge to x≈0.5 (feature point pinned on plane)
//   - valid flag (w) == 1.0

using NUnit.Framework;
using Unity.Mathematics;
using UnityEngine;
using ReconGridDC.Preprocess;
using ReconGridDC.Recon;

#if UNITY_EDITOR
using UnityEditor;
#endif

public class Qef_GpuOracle_Tests
{
    sealed class PlaneX : ILevelSetProvider
    {
        public float  Sample(float3 p)   => p.x - 0.5f;
        public float3 Gradient(float3 p) => new float3(1, 0, 0);
    }

    static ComputeShader LoadReconShader()
    {
#if UNITY_EDITOR
        // Robust: locate Recon.compute anywhere in the project (folder-location independent).
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
        // Fallback: Resources.Load (requires shader to be in a Resources folder)
        return Resources.Load<ComputeShader>("Recon");
    }

    [Test]
    public void DC_FeaturePoints_PlaneVoxel_FpOnPlane()
    {
        var cs = LoadReconShader();
        Assert.IsNotNull(cs, "Recon.compute must be loadable via AssetDatabase or Resources");

        // Build a 1-voxel grid with a plane at x=0.5
        var g = BackgroundGrid.Build(new PlaneX(), new int3(1, 1, 1), 1f, float3.zero);
        Assert.AreEqual(1, g.voxelCount, "Expected exactly 1 voxel");
        Assert.AreEqual(4, g.voxelIsectCount[0], "Expected 4 x-direction intersections in voxel 0");

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

            // Bind buffers
            cs.SetBuffer(k, "_CornerPos",         rb.CornerPos);
            cs.SetBuffer(k, "_VoxelCorner",       rb.VoxelCorner);
            cs.SetBuffer(k, "_VoxelIsectOffset",  rb.VoxelIsectOffset);
            cs.SetBuffer(k, "_VoxelIsectCount",   rb.VoxelIsectCount);
            cs.SetBuffer(k, "_Isect",             rb.Isect);
            cs.SetBuffer(k, "_IsectWorld",        rb.IsectWorld);
            cs.SetBuffer(k, "_VoxelExternalFP",   rb.VoxelExternalFP);

            // Dispatch: ceil(1/64) = 1 group of 64 threads; only thread 0 does work
            cs.Dispatch(k, Mathf.CeilToInt(rb.VoxelCount / 64f), 1, 1);

            // Readback and assert
            var fp = new float4[rb.VoxelCount];
            rb.VoxelExternalFP.GetData(fp);

            Assert.AreEqual(1f,   fp[0].w, 1e-5f,  "Feature point must be marked valid (w=1)");
            Assert.AreEqual(0.5f, fp[0].x, 1e-3f,  "Feature point x must be on the plane x=0.5");
            Assert.IsFalse(float.IsNaN(fp[0].y),    "Feature point y must not be NaN");
            Assert.IsFalse(float.IsNaN(fp[0].z),    "Feature point z must not be NaN");
        }
    }
}
