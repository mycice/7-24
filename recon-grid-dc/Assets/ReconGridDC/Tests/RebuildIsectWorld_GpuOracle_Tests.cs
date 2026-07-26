// RebuildIsectWorld_GpuOracle_Tests.cs — Stage 1B-ii Task 1 Step 2
// GPU oracle test: dispatches RebuildIsectWorld directly and verifies the lerp
// along a deformed edge produces the correct deformed intersection position.
//
// Setup: plane at x=0.5 in a 1-voxel grid [0,1]^3.
//   - Corner (0,0,0) is at x=0 (inside, phi=-0.5).
//   - Corner (1,0,0) is at x=1 (outside, phi=+0.5). t = 0.5 along this edge.
//   - Displace corner (1,0,0) by +1 in x: new position = (2,0,0).
//   - Deformed lerp: lerp((0,0,0), (2,0,0), 0.5) => x = 1.0.
//
// This also verifies the STATIC 1A path is preserved:
//   at rest CornerPos == rest positions => IsectWorld == globalRest (unchanged).
//
// [Category("GPU")] — requires a real GPU; skip in CI without one.

using NUnit.Framework;
using Unity.Mathematics;
using UnityEngine;
using ReconGridDC.Preprocess;
using ReconGridDC.Recon;

#if UNITY_EDITOR
using UnityEditor;
#endif

[Category("GPU")]
public class RebuildIsectWorld_GpuOracle_Tests
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

    /// <summary>
    /// Dispatch RebuildIsectWorld only; verify that a deformed x-edge isect lerps to x=1.0.
    /// </summary>
    [Test]
    public void RebuildIsectWorld_LerpsAlongDeformedEdge()
    {
        var cs = LoadReconShader();
        Assert.IsNotNull(cs, "Recon.compute must be loadable via AssetDatabase or Resources");

        // Build a 1-voxel grid with a plane at x=0.5
        var g = BackgroundGrid.Build(new PlaneX(), new int3(1, 1, 1), 1f, float3.zero);
        Assert.AreEqual(1, g.voxelCount, "Expected exactly 1 voxel");
        Assert.AreEqual(4, g.voxelIsectCount[0], "Expected 4 x-direction intersections in voxel 0 (plane x=0.5)");

        using (var rb = new ReconBuffers(g, 64))
        {
            rb.Upload(g);

            // Displace corner (1,0,0) outward by +1 in x: rest pos (1,0,0) -> (2,0,0).
            // For dims=(1,1,1), CornerId(1,0,0) = 1 + (1+1)*(0 + (1+1)*0) = 1.
            int cidx = ReconGridDC.Core.GridConventions.CornerId(1, 0, 0, g.dims);
            var pos = new float3[rb.CornerPos.count];
            rb.CornerPos.GetData(pos);
            pos[cidx] += new float3(1, 0, 0);   // (1,0,0) => (2,0,0)
            rb.CornerPos.SetData(pos);

            // Dispatch RebuildIsectWorld only (not the full DC pipeline)
            int k = cs.FindKernel("RebuildIsectWorld");
            Assert.GreaterOrEqual(k, 0, "Kernel RebuildIsectWorld not found in Recon.compute");

            cs.SetInt("_IsectCount", rb.IsectCount);
            cs.SetBuffer(k, "_Isect",      rb.Isect);
            cs.SetBuffer(k, "_CornerPos",  rb.CornerPos);
            cs.SetBuffer(k, "_IsectWorld", rb.IsectWorld);

            cs.Dispatch(k, Mathf.CeilToInt(rb.IsectCount / 64f), 1, 1);

            // Readback IsectWorld and verify
            var iw = new float3[rb.IsectCount];
            rb.IsectWorld.GetData(iw);

            // The isect on the deformed x-edge (cornerA=(0,0,0)->x=0, cornerB=(1,0,0)->now x=2, t=0.5)
            // => lerp(0, 2, 0.5) = 1.0. At least one isect must have x≈1.0.
            bool found = false;
            for (int i = 0; i < rb.IsectCount; i++)
            {
                if (Mathf.Abs(iw[i].x - 1.0f) < 1e-3f)
                {
                    found = true;
                    break;
                }
            }
            Assert.IsTrue(found,
                $"RebuildIsectWorld: expected at least one isect to lerp to x=1.0 on the deformed edge. " +
                $"Got x values: {string.Join(", ", System.Array.ConvertAll(iw, v => v.x.ToString("F4")))}");
        }
    }

    /// <summary>
    /// At rest (no displacement), RebuildIsectWorld must produce IsectWorld == globalRest.
    /// This confirms the static 1A path is preserved.
    /// </summary>
    [Test]
    public void RebuildIsectWorld_AtRest_MatchesGlobalRest()
    {
        var cs = LoadReconShader();
        Assert.IsNotNull(cs, "Recon.compute must be loadable via AssetDatabase or Resources");

        var g = BackgroundGrid.Build(new PlaneX(), new int3(1, 1, 1), 1f, float3.zero);

        using (var rb = new ReconBuffers(g, 64))
        {
            rb.Upload(g);

            // Dispatch RebuildIsectWorld with unmodified (rest) CornerPos
            int k = cs.FindKernel("RebuildIsectWorld");
            cs.SetInt("_IsectCount", rb.IsectCount);
            cs.SetBuffer(k, "_Isect",      rb.Isect);
            cs.SetBuffer(k, "_CornerPos",  rb.CornerPos);
            cs.SetBuffer(k, "_IsectWorld", rb.IsectWorld);
            cs.Dispatch(k, Mathf.CeilToInt(rb.IsectCount / 64f), 1, 1);

            // Readback
            var iw = new float3[rb.IsectCount];
            rb.IsectWorld.GetData(iw);

            // Compare against globalRest from the original isect data
            for (int i = 0; i < g.isect.Length; i++)
            {
                var gr = g.isect[i].globalRest;
                Assert.AreEqual(gr.x, iw[i].x, 1e-4f, $"IsectWorld[{i}].x should equal globalRest.x at rest");
                Assert.AreEqual(gr.y, iw[i].y, 1e-4f, $"IsectWorld[{i}].y should equal globalRest.y at rest");
                Assert.AreEqual(gr.z, iw[i].z, 1e-4f, $"IsectWorld[{i}].z should equal globalRest.z at rest");
            }
        }
    }
}
