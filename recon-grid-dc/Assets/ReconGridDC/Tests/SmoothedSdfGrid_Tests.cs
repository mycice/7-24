// SmoothedSdfGrid_Tests.cs — liver tuning, design Fix 2 Testing #2/#3.
// CPU tests: bake/interp correctness vs analytic SDFs; smoothing preserves sign + low shrinkage.

using NUnit.Framework;
using Unity.Mathematics;
using ReconGridDC.Preprocess;

public class SmoothedSdfGrid_Tests
{
    [Test]
    public void Bake_NoSmooth_MatchesBaseInFlatRegion_GradientUnit()
    {
        var box = new BoxLevelSet(new float3(0, 0, 0), new float3(1, 1, 1));
        // domain [-2,2]^3 (dims 16 × L 0.25), recon-aligned bake, no smoothing.
        var ls = new SmoothedSdfGrid(box, new float3(-2, -2, -2), new int3(16, 16, 16), 0.25f, 1, 0);

        // Flat +x region: box distance is linear (x−1) ⇒ trilinear bake is ~exact.
        var p = new float3(1.5f, 0f, 0f);
        Assert.AreEqual(box.Sample(p), ls.Sample(p), 0.02f);

        var g = ls.Gradient(p);
        Assert.AreEqual(1f, math.length(g), 1e-3f);   // unit
        Assert.Greater(g.x, 0.9f);                     // outward +x

        var gon = ls.Gradient(new float3(1f, 0f, 0f)); // on the face
        Assert.IsFalse(math.any(math.isnan(gon)));
        Assert.AreEqual(1f, math.length(gon), 1e-3f);
    }

    [Test]
    public void Smoothing_PreservesSign_AndLowShrinkage_OnSphere()
    {
        var sphere = new SphereLevelSet(new float3(0, 0, 0), 1f);     // smooth surface, radius 1
        var sm = new SmoothedSdfGrid(sphere, new float3(-2, -2, -2), new int3(16, 16, 16), 0.25f, 1, 4);

        // Sign preserved: clearly inside < 0, clearly outside > 0.
        Assert.Less(sm.Sample(new float3(0, 0, 0)), 0f);
        Assert.Greater(sm.Sample(new float3(1.5f, 0, 0)), 0f);

        // Low shrinkage (Taubin): the zero crossing stays near the true surface (radius 1).
        Assert.Less(sm.Sample(new float3(0.85f, 0, 0)), 0f);   // just inside
        Assert.Greater(sm.Sample(new float3(1.15f, 0, 0)), 0f); // just outside
        Assert.Less(math.abs(sm.Sample(new float3(1f, 0, 0))), 0.1f, "surface stays near r=1 (low shrinkage)");

        // Gradient finite + unit.
        var g = sm.Gradient(new float3(0.7f, 0, 0));
        Assert.IsFalse(math.any(math.isnan(g)));
        Assert.AreEqual(1f, math.length(g), 1e-2f);
    }

    // A deliberately bumpy field: sphere r≈1 with a high-frequency sinusoidal perturbation.
    sealed class BumpySphere : ReconGridDC.Preprocess.ILevelSetProvider
    {
        public float Sample(float3 p)
            => math.length(p) - (1f + 0.15f * math.sin(8f * p.x) * math.sin(8f * p.y) * math.sin(8f * p.z));
        public float3 Gradient(float3 p)
        { float l = math.length(p); return l > 1e-8f ? p / l : new float3(1, 0, 0); }
    }

    [Test]
    public void Smoothing_MonotonicallyReducesRoughness()   // design Testing #3
    {
        var bumpy = new BumpySphere();
        var dims = new int3(24, 24, 24); var origin = new float3(-2, -2, -2); float L = 4f / 24f;
        float v0 = new SmoothedSdfGrid(bumpy, origin, dims, L, 1, 0).LaplacianVariance();
        float v2 = new SmoothedSdfGrid(bumpy, origin, dims, L, 1, 2).LaplacianVariance();
        float v4 = new SmoothedSdfGrid(bumpy, origin, dims, L, 1, 4).LaplacianVariance();

        Assert.Greater(v0, v2, "2 smoothing iters must reduce field roughness vs 0");
        Assert.Greater(v2, v4, "4 smoothing iters must reduce roughness further");

        // Surface sign still preserved after smoothing (low shrinkage, not a no-op the other way).
        var sm = new SmoothedSdfGrid(bumpy, origin, dims, L, 1, 4);
        Assert.Less(sm.Sample(float3.zero), 0f);
        Assert.Greater(sm.Sample(new float3(1.8f, 0, 0)), 0f);
    }
}
