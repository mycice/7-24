using NUnit.Framework;
using Unity.Mathematics;
using ReconGridDC.Preprocess;

public class BackgroundGrid_Tests
{
    // Plane at x = 0.5 inside a 1x1x1-voxel grid spanning [0,1]^3, L=1, origin=0.
    sealed class PlaneX : ILevelSetProvider {
        public float Sample(float3 p)=>p.x-0.5f;
        public float3 Gradient(float3 p)=>new float3(1,0,0);
    }

    [Test] public void SingleVoxel_PlaneX_FourXEdgesIntersect()
    {
        var g = BackgroundGrid.Build(new PlaneX(), new int3(1,1,1), 1f, float3.zero);
        Assert.AreEqual(1, g.voxelCount);
        // 4 x-direction local edges (e0,e2,e4,e6) cross x=0.5 -> 4 isect entries in voxel 0
        Assert.AreEqual(4, g.voxelIsectCount[0]);
        foreach (var idx in System.Linq.Enumerable.Range(g.voxelIsectOffset[0], g.voxelIsectCount[0]))
        {
            Assert.AreEqual(0.5f, g.isect[idx].globalRest.x, 1e-5f);   // crossing at x=0.5
            Assert.AreEqual(1f, g.isect[idx].normal.x, 1e-4f);         // outward +x
        }
    }

    [Test] public void CornerInside_FollowsSign()
    {
        var g = BackgroundGrid.Build(new PlaneX(), new int3(1,1,1), 1f, float3.zero);
        // corner (0,0,0) at x=0 -> inside (φ=-0.5); corner (1,0,0) at x=1 -> outside
        Assert.AreEqual(1, g.cornerInside[ReconGridDC.Core.GridConventions.CornerId(0,0,0,g.dims)]);
        Assert.AreEqual(0, g.cornerInside[ReconGridDC.Core.GridConventions.CornerId(1,0,0,g.dims)]);
    }

    [Test, Category("GPU")] public void ReconBuffers_AllocateUploadDispose()
    {
        var g = BackgroundGrid.Build(new PlaneX(), new int3(2,2,2), 1f, Unity.Mathematics.float3.zero);
        using (var rb = new ReconGridDC.Recon.ReconBuffers(g, triCapacity: 4096))
        {
            rb.Upload(g);
            Assert.AreEqual(g.voxelCount, rb.VoxelCount);
            Assert.AreEqual(4096, rb.TriCapacity);
            Assert.IsNotNull(rb.IndirectArgs);
        }
    }
}
