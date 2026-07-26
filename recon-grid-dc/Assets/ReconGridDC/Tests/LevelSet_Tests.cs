using NUnit.Framework;
using Unity.Mathematics;
using ReconGridDC.Preprocess;

public class LevelSet_Tests
{
    [Test] public void Sphere_SignAndGradient()
    {
        var s = new SphereLevelSet(new float3(0,0,0), 1f);
        Assert.Less(s.Sample(new float3(0,0,0)), 0f);          // inside
        Assert.AreEqual(0f, s.Sample(new float3(1,0,0)), 1e-5f); // on surface
        Assert.Greater(s.Sample(new float3(2,0,0)), 0f);        // outside
        var g = s.Gradient(new float3(0.5f,0,0));
        Assert.AreEqual(1f, math.length(g), 1e-4f);            // unit gradient
        Assert.Greater(g.x, 0.99f);                            // points outward (+x)
    }

    [Test] public void Box_SignedDistance()
    {
        var b = new BoxLevelSet(new float3(0,0,0), new float3(1,1,1));
        Assert.AreEqual(-1f, b.Sample(new float3(0,0,0)), 1e-5f); // center, dist to nearest face = 1 inside => -1
        Assert.AreEqual(0f, b.Sample(new float3(1,0,0)), 1e-5f);
        Assert.AreEqual(1f, b.Sample(new float3(2,0,0)), 1e-5f);
    }
}
