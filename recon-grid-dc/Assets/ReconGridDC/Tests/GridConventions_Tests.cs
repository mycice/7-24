using NUnit.Framework;
using Unity.Mathematics;     // if math pkg absent, use UnityEngine.Vector3Int / Vector3 and adjust signatures
using ReconGridDC.Core;

public class GridConventions_Tests
{
    [Test] public void EdgeCorners_MatchFig24()
    {
        // bottom loop
        Assert.AreEqual((0,1),(GridConventions.EdgeCorners[0,0],GridConventions.EdgeCorners[0,1]));
        Assert.AreEqual((1,2),(GridConventions.EdgeCorners[1,0],GridConventions.EdgeCorners[1,1]));
        Assert.AreEqual((2,3),(GridConventions.EdgeCorners[2,0],GridConventions.EdgeCorners[2,1]));
        Assert.AreEqual((3,0),(GridConventions.EdgeCorners[3,0],GridConventions.EdgeCorners[3,1]));
        // verticals
        Assert.AreEqual((0,4),(GridConventions.EdgeCorners[8,0],GridConventions.EdgeCorners[8,1]));
        Assert.AreEqual((3,7),(GridConventions.EdgeCorners[11,0],GridConventions.EdgeCorners[11,1]));
    }

    [Test] public void VoxelId_RoundTrips()
    {
        int3 dims = new int3(4,5,6);
        int id = GridConventions.VoxelId(2,3,4,dims);
        Assert.AreEqual(2 + 4*(3 + 5*4), id);
    }

    [Test] public void XEdgeStencil_OrderedE0E2E6E4()
    {
        var s = GridConventions.EdgeStencil[0]; // x axis, CCW about +x
        Assert.AreEqual(0,  s[0].localEdge); Assert.AreEqual(new int3(0,0,0),   s[0].voxelOffset);
        Assert.AreEqual(2,  s[1].localEdge); Assert.AreEqual(new int3(0,-1,0),  s[1].voxelOffset);
        Assert.AreEqual(6,  s[2].localEdge); Assert.AreEqual(new int3(0,-1,-1), s[2].voxelOffset);
        Assert.AreEqual(4,  s[3].localEdge); Assert.AreEqual(new int3(0,0,-1),  s[3].voxelOffset);
    }

    [Test] public void BoxSdf_NegativeInsideZeroOnFace()
    {
        float3 c = new float3(0,0,0); float L = 2f; // half=1
        Assert.Less(GridConventions.BoxSdf(new float3(0,0,0),c,L), 0f);
        Assert.AreEqual(0f, GridConventions.BoxSdf(new float3(1,0,0),c,L), 1e-6f);
        Assert.Greater(GridConventions.BoxSdf(new float3(2,0,0),c,L), 0f);
    }
}
