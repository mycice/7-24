// MeshLevelSet_Tests.cs — liver-import pipeline, design §Testing #3/#4/#5.
// CPU tests (no GPU). Validate the mesh SDF against an analytic BoxLevelSet on an inline cube,
// and the GridFit helper.

using NUnit.Framework;
using Unity.Mathematics;
using ReconGridDC.Preprocess;

public class MeshLevelSet_Tests
{
    // Outward-wound unit cube (half-extent h): 8 verts, 12 tris. Matches BoxLevelSet(0, (h,h,h)).
    static MshSurface Cube(float h)
    {
        var v = new float3[]
        {
            new float3(-h,-h,-h), new float3( h,-h,-h), new float3( h, h,-h), new float3(-h, h,-h),
            new float3(-h,-h, h), new float3( h,-h, h), new float3( h, h, h), new float3(-h, h, h),
        };
        var t = new int3[]
        {
            new int3(1,2,6), new int3(1,6,5),   // +x
            new int3(0,4,7), new int3(0,7,3),   // -x
            new int3(3,7,6), new int3(3,6,2),   // +y
            new int3(0,1,5), new int3(0,5,4),   // -y
            new int3(4,5,6), new int3(4,6,7),   // +z
            new int3(0,3,2), new int3(0,2,1),   // -z
        };
        return new MshSurface { verts = v, tris = t };
    }

    [Test]
    public void Sign_MatchesBox_AndGradientOutward()
    {
        var ls  = new MeshLevelSet(Cube(1f));
        var box = new BoxLevelSet(new float3(0,0,0), new float3(1,1,1));

        // Center: inside, distance to nearest face = 1 → −1.
        Assert.Less(ls.Sample(new float3(0,0,0)), 0f);
        Assert.AreEqual(box.Sample(new float3(0,0,0)), ls.Sample(new float3(0,0,0)), 1e-4f);

        // Interior off-center point.
        var pi = new float3(0.5f, 0f, 0f);
        Assert.AreEqual(box.Sample(pi), ls.Sample(pi), 1e-4f);   // −0.5

        // Exterior point.
        var po = new float3(2f, 0f, 0f);
        Assert.Greater(ls.Sample(po), 0f);
        Assert.AreEqual(box.Sample(po), ls.Sample(po), 1e-4f);   // +1

        // Gradient is ~unit and points OUTWARD (+x) both inside and outside the +x face.
        var gi = ls.Gradient(pi);
        Assert.AreEqual(1f, math.length(gi), 1e-3f);
        Assert.Greater(gi.x, 0.99f);
        var go = ls.Gradient(po);
        Assert.AreEqual(1f, math.length(go), 1e-3f);
        Assert.Greater(go.x, 0.99f);
    }

    [Test]
    public void UnsignedDistance_MatchesAnalyticBox()
    {
        var ls  = new MeshLevelSet(Cube(1f));
        var box = new BoxLevelSet(new float3(0,0,0), new float3(1,1,1));

        // Exterior point off the +x face: distance 0.5 to the closest face point (1, 0.3, 0.2).
        var p = new float3(1.5f, 0.3f, 0.2f);
        Assert.AreEqual(0.5f, ls.Sample(p), 1e-4f);
        Assert.AreEqual(box.Sample(p), ls.Sample(p), 1e-4f);
    }

    [Test]
    public void OnSurface_Gradient_IsFiniteUnit()
    {
        var ls = new MeshLevelSet(Cube(1f));
        // Exactly on the +x face → normalize(p−closest) is 0/0; epsilon fallback must return a finite unit.
        var g = ls.Gradient(new float3(1f, 0f, 0f));
        Assert.IsFalse(math.any(math.isnan(g)), "on-surface gradient must not be NaN");
        Assert.AreEqual(1f, math.length(g), 1e-3f);
    }

    [Test]
    public void GridFit_CoversBoundsWithMargin()
    {
        var min = new float3(-11f, -3.35f, -3.29f);
        var max = new float3(6.82f, 6.55f, 6.76f);
        GridFit.FitToBounds(min, max, 48, 2, out int3 dims, out float L, out float3 origin);

        Assert.Greater(L, 0f);
        Assert.IsTrue(math.all(dims >= new int3(1)), "dims ≥ 1 per axis");
        float3 hi = origin + (float3)dims * L;
        Assert.IsTrue(math.all(origin <= min), "origin covers below the min corner");
        Assert.IsTrue(math.all(hi >= max), "origin + dims*L covers above the max corner");
        // Longest axis (x, extent 17.82) resolved at ~target voxels: L ≈ 17.82/48.
        Assert.AreEqual(17.82f / 48f, L, 1e-2f);
    }
}
