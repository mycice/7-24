using NUnit.Framework;
using Unity.Mathematics;
using ReconGridDC.Preprocess;
using ReconGridDC.Core;

public class PhysicsTopology_Tests
{
    // Everything inside => full uniform lattice (all corners inside, every edge exists)
    sealed class AllInside : ILevelSetProvider
    {
        public float Sample(float3 p) => -1f;
        public float3 Gradient(float3 p) => new float3(0, 1, 0);
    }

    // Stage 2D-1b: a sphere small enough to leave a ring of fully-OUTSIDE corners around the lattice
    // (so cornerActive has both 1s and 0s) — exercises the tissue-only spring/bend gating (paper §2.1.4).
    sealed class Sphere : ILevelSetProvider
    {
        readonly float3 c; readonly float r;
        public Sphere(float3 center, float radius) { c = center; r = radius; }
        public float Sample(float3 p) => math.length(p - c) - r;
        public float3 Gradient(float3 p)
        {
            float3 d = p - c; float len = math.length(d);
            return len < 1e-8f ? new float3(0, 1, 0) : d / len;
        }
    }

    // ── Task 1 CPU tests ─────────────────────────────────────────────────────────────────────

    [Test]
    public void Springs_OnePerGridEdge_RestLengthEqualsVoxel()
    {
        var g = BackgroundGrid.Build(new AllInside(), new int3(2, 2, 2), 0.25f, float3.zero);
        // axis edges in a 2x2x2-voxel (3x3x3-corner) lattice:
        //   x: 2*3*3=18 ; y: 3*2*3=18 ; z: 3*3*2=18  => 54 springs
        Assert.AreEqual(54, g.springs.Length);
        foreach (var s in g.springs)
            Assert.AreEqual(0.25f, s.L0, 1e-6f);
    }

    [Test]
    public void BendPairs_15PerCorner_BoundaryArmsDead()
    {
        var g = BackgroundGrid.Build(new AllInside(), new int3(2, 2, 2), 0.25f, float3.zero);
        Assert.AreEqual(15 * g.cornerCount, g.bendPairs.Length);
        // a corner of the cube (3 neighbors: +x,+y,+z) has C(3,2)=3 alive pairs
        // an interior corner (6 neighbors) has all 15 alive
        int interior  = GridConventions.CornerId(1, 1, 1, g.dims);
        int aliveInt  = CountAlive(g, interior);
        Assert.AreEqual(15, aliveInt);
        int cornerId  = GridConventions.CornerId(0, 0, 0, g.dims);
        Assert.AreEqual(3, CountAlive(g, cornerId)); // 3 neighbors (+x,+y,+z) -> C(3,2)=3
    }

    [Test]
    public void RestBendAngle_CollinearPairsArePi()
    {
        var g = BackgroundGrid.Build(new AllInside(), new int3(2, 2, 2), 0.25f, float3.zero);
        int interior = GridConventions.CornerId(1, 1, 1, g.dims);
        // the (-x,+x),(-y,+y),(-z,+z) pairs are collinear -> rest angle == pi
        // PairSlots: slot 0=(0,1)=(-x,+x), slot 9=(2,3)=(-y,+y), slot 14=(4,5)=(-z,+z)
        int[] collinearSlots = { 0, 9, 14 };
        foreach (int slot in collinearSlots)
        {
            var bp = g.bendPairs[15 * interior + slot];
            Assert.AreEqual(1, bp.alive, $"collinear pair slot {slot} should be alive");
            Assert.AreEqual(math.PI, bp.theta0, 1e-5f, $"collinear pair slot {slot} should have theta0=pi");
        }
        Assert.Pass();
    }

    [Test]
    public void NbrIdx_SymmetryCheck()
    {
        var g = BackgroundGrid.Build(new AllInside(), new int3(2, 2, 2), 0.25f, float3.zero);
        int interior = GridConventions.CornerId(1, 1, 1, g.dims);
        // Interior corner (1,1,1) in a 3x3x3 corner grid should have all 6 neighbors
        for (int s = 0; s < 6; s++)
            Assert.AreNotEqual(-1, g.nbrIdx[6 * interior + s], $"slot {s} should have a neighbor");
        // Corner (0,0,0) should only have +x(1), +y(3), +z(5) neighbors
        int corner = GridConventions.CornerId(0, 0, 0, g.dims);
        Assert.AreEqual(-1, g.nbrIdx[6 * corner + 0]); // -x missing
        Assert.AreNotEqual(-1, g.nbrIdx[6 * corner + 1]); // +x present
        Assert.AreEqual(-1, g.nbrIdx[6 * corner + 2]); // -y missing
        Assert.AreNotEqual(-1, g.nbrIdx[6 * corner + 3]); // +y present
        Assert.AreEqual(-1, g.nbrIdx[6 * corner + 4]); // -z missing
        Assert.AreNotEqual(-1, g.nbrIdx[6 * corner + 5]); // +z present
    }

    [Test]
    public void Mass_UniformOne()
    {
        var g = BackgroundGrid.Build(new AllInside(), new int3(2, 2, 2), 0.25f, float3.zero);
        foreach (var m in g.mass)
            Assert.AreEqual(1.0f, m, 1e-7f);
    }

    [Test]
    public void Pinned_DefaultZero()
    {
        var g = BackgroundGrid.Build(new AllInside(), new int3(2, 2, 2), 0.25f, float3.zero);
        foreach (var p in g.pinned)
            Assert.AreEqual(0, p);
    }

    // ── Stage 2D-1b: tissue-only physics oracle (paper §2.1.4, paper:485-499) ──────────────────
    // A sphere of radius 1 centered in a 8^3-voxel grid leaves the corner lattice partly outside,
    // so cornerActive has both 1s and 0s. The gating must guarantee: (1) AllInside ⇒ all active;
    // (2) NO structural spring touches an inactive corner; (3) NO alive bending pair touches an
    // inactive corner. These three together prove the empty-space spring/bend "cage" is removed.

    [Test]
    public void CornerActive_AllInside_AllActive()
    {
        var g = BackgroundGrid.Build(new AllInside(), new int3(2, 2, 2), 0.25f, float3.zero);
        Assert.AreEqual(g.cornerCount, g.cornerActive.Length);
        foreach (var a in g.cornerActive)
            Assert.AreEqual(1, a, "every corner is incident to an occupied voxel ⇒ active");
    }

    [Test]
    public void CornerActive_Sphere_HasInactiveCornersAndActiveShell()
    {
        var g = BackgroundGrid.Build(new Sphere(new float3(1f, 1f, 1f), 1f),
                                     new int3(8, 8, 8), 0.25f, float3.zero);
        int active = 0, inactive = 0;
        for (int c = 0; c < g.cornerCount; c++)
            if (g.cornerActive[c] == 1) active++; else inactive++;
        // Must have BOTH (otherwise the gating is untested): some tissue + some empty-space corners.
        Assert.Greater(inactive, 0, "a radius-1 sphere in an 8^3 grid must leave outside corners");
        Assert.Greater(active,   0, "the sphere interior + shell must produce active corners");
        // Every INSIDE corner must be active (inside ⇒ all 8 incident voxels... at least one ⇒ occupied).
        for (int c = 0; c < g.cornerCount; c++)
            if (g.cornerInside[c] == 1)
                Assert.AreEqual(1, g.cornerActive[c], "an inside corner must be active");
    }

    [Test]
    public void Springs_NeverTouchInactiveCorner()
    {
        var g = BackgroundGrid.Build(new Sphere(new float3(1f, 1f, 1f), 1f),
                                     new int3(8, 8, 8), 0.25f, float3.zero);
        foreach (var s in g.springs)
        {
            Assert.AreEqual(1, g.cornerActive[s.i], $"spring endpoint i={s.i} must be active");
            Assert.AreEqual(1, g.cornerActive[s.j], $"spring endpoint j={s.j} must be active");
        }
    }

    [Test]
    public void BendPairs_AliveNeverTouchInactiveCorner()
    {
        var g = BackgroundGrid.Build(new Sphere(new float3(1f, 1f, 1f), 1f),
                                     new int3(8, 8, 8), 0.25f, float3.zero);
        foreach (var bp in g.bendPairs)
        {
            if (bp.alive == 0) continue;
            Assert.AreEqual(1, g.cornerActive[bp.i], $"alive bend center i={bp.i} must be active");
            Assert.AreEqual(1, g.cornerActive[bp.j], $"alive bend arm j={bp.j} must be active");
            Assert.AreEqual(1, g.cornerActive[bp.k], $"alive bend arm k={bp.k} must be active");
        }
    }

    // ── Task 2 GPU buffer test ────────────────────────────────────────────────────────────────

    [Test, Category("GPU")]
    public void PhysicsBuffers_AllocateUploadDispose()
    {
        var g = BackgroundGrid.Build(new AllInside(), new int3(2, 2, 2), 0.25f, Unity.Mathematics.float3.zero);
        using (var rb = new ReconGridDC.Recon.ReconBuffers(g, 4096))
        {
            rb.Upload(g);
            Assert.AreEqual(g.cornerCount, rb.CornerCount);
            Assert.AreEqual(g.springs.Length, rb.SpringCount);
            Assert.AreEqual(g.bendPairs.Length, rb.BendPairCount);
            Assert.IsNotNull(rb.KSlope);
            Assert.IsNotNull(rb.Vel);
            Assert.IsNotNull(rb.Mass);
            Assert.IsNotNull(rb.Pinned);
            Assert.IsNotNull(rb.ForceInt);
            Assert.IsNotNull(rb.ExtForce);
            Assert.IsNotNull(rb.YTrialPos);
            Assert.IsNotNull(rb.YTrialVel);
            Assert.IsNotNull(rb.Springs);
            Assert.IsNotNull(rb.BendPairs);
            Assert.IsNotNull(rb.NbrIdx);
            Assert.IsNotNull(rb.RestNbr);
            Assert.IsNotNull(rb.ErrMax);
        }
    }

    // ── Helpers ──────────────────────────────────────────────────────────────────────────────

    static int CountAlive(BackgroundGrid g, int corner)
    {
        int c = 0;
        for (int p = 15 * corner; p < 15 * corner + 15; p++)
            if (g.bendPairs[p].alive == 1) c++;
        return c;
    }
}
