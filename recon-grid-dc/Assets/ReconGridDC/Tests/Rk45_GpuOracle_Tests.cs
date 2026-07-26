// Rk45_GpuOracle_Tests.cs — GPU oracle for the adaptive RK45 solver (Stage 1B-i).
// Two oracles that call MassSpringSolver.Step (the REAL adaptive orchestrator):
//   (a) FreeFall_ViaSolverStep_MatchesAnalytic: one unpinned particle, gravity only.
//       Assert y ≈ y0-0.5*g*T^2, vy ≈ -g*T within 1e-3 after T=0.1 s (5 frames of dt=0.02).
//   (b) Oscillator1D_ViaSolverStep_TracksAnalyticSHM: one pinned + one free particle,
//       one structural spring, SHM ω=sqrt(Ks/m)=10 rad/s.
//       Assert x_B ≈ L0+A0*cos(ω*T) within 2e-3 after T=0.05 s.
// Requires a GPU; run with [Category("GPU")] filter.
//
// Both tests drive MassSpringSolver.Step (inner substep loop, D4 h-adapt, last-substep
// clip, ErrMax readback). They do NOT call any private single-substep helper.

using System;
using System.IO;
using NUnit.Framework;
using Unity.Mathematics;
using UnityEditor;
using UnityEngine;
using ReconGridDC.Physics;
using ReconGridDC.Recon;
using ReconGridDC.Preprocess;

[Category("GPU")]
public class Rk45_GpuOracle_Tests
{
    // ── BendPairGpu mirrored from ReconBuffers (private struct; 32 bytes) ────────────────────
    struct BendPairGpu
    {
        public int i, j, k;
        public float theta0;
        public int alive, _p0, _p1, _p2;
    }

    // ── Helpers ──────────────────────────────────────────────────────────────────────────────

    static ComputeShader LoadPhysicsCS()
    {
        string[] guids = AssetDatabase.FindAssets("Physics t:ComputeShader");
        foreach (string guid in guids)
        {
            string path = AssetDatabase.GUIDToAssetPath(guid);
            if (path.EndsWith("/Physics.compute"))
                return AssetDatabase.LoadAssetAtPath<ComputeShader>(path);
        }
        throw new FileNotFoundException("Physics.compute not found via AssetDatabase");
    }

    // ── Minimal stub level-set: sphere of radius r centred at origin ─────────────────────────
    // A 0x0x0-voxel grid (dims=0) is illegal, so we build a 1x1x1 grid that is entirely
    // OUTSIDE the isosurface (large positive phi everywhere) so no isect or surface edges
    // are generated. We then Dispose+replace specific physics ComputeBuffers with test data.
    class AllOutsideLevelSet : ILevelSetProvider
    {
        public float  Sample(float3 p) => 1000f;          // large positive: always outside
        public float3 Gradient(float3 p) => new float3(0,1,0);
    }

    /// <summary>
    /// Build a minimal ReconBuffers whose physics buffers are sized for <paramref name="N"/>
    /// particles and <paramref name="bendPairCount"/> bend pairs.
    /// The underlying BackgroundGrid uses a 1x1x1 all-outside grid (8 corners, no isect),
    /// then we Dispose and replace every physics buffer to match N.
    /// </summary>
    static ReconBuffers MakeMinimalRb(
        int      N,
        int      bendPairCount,
        float3[] cornerPos,
        float3[] vel,
        float[]  mass,
        int[]    pinned,
        float3[] extForce,
        int[]    nbrIdx,
        float3[] restNbr,
        BendPairGpu[] bpArr)
    {
        // Build a real 1x1x1 grid just to satisfy the ReconBuffers constructor.
        // Resulting g.cornerCount = 8; we ignore all the geometry.
        var g = BackgroundGrid.Build(
            new AllOutsideLevelSet(),
            dims:   new int3(1, 1, 1),
            L:      1f,
            origin: float3.zero);

        var rb = new ReconBuffers(g, triCapacity: 1);

        // ── Replace physics buffers with our N-sized ones ────────────────────────────────────
        rb.CornerPos.Dispose();
        rb.CornerPos   = new ComputeBuffer(N, 12);
        rb.CornerPos.SetData(cornerPos);

        rb.Vel.Dispose();
        rb.Vel         = new ComputeBuffer(N, 12);
        rb.Vel.SetData(vel);

        rb.Mass.Dispose();
        rb.Mass        = new ComputeBuffer(N, 4);
        rb.Mass.SetData(mass);

        rb.Pinned.Dispose();
        rb.Pinned      = new ComputeBuffer(N, 4);
        rb.Pinned.SetData(pinned);

        // 2D-1b: physics kernels now read _Active (tissue-only gate). The base ReconBuffers sized it
        // for the 1×1×1 grid's 8 corners (all inactive under the AllOutside level set) → must replace
        // with an N-sized ALL-ACTIVE buffer, else every test particle is frozen like an inactive corner.
        rb.CornerActive.Dispose();
        rb.CornerActive = new ComputeBuffer(N, 4);
        var activeOnes = new int[N];
        for (int i = 0; i < N; i++) activeOnes[i] = 1;
        rb.CornerActive.SetData(activeOnes);

        rb.ExtForce.Dispose();
        rb.ExtForce    = new ComputeBuffer(N, 12);
        rb.ExtForce.SetData(extForce);

        rb.ForceInt.Dispose();
        rb.ForceInt    = new ComputeBuffer(Math.Max(1, 3 * N), 4);
        rb.ForceInt.SetData(new int[3 * N]);

        rb.KSlope.Dispose();
        rb.KSlope      = new ComputeBuffer(Math.Max(1, 7 * N), 24);

        rb.YTrialPos.Dispose();
        rb.YTrialPos   = new ComputeBuffer(N, 12);

        rb.YTrialVel.Dispose();
        rb.YTrialVel   = new ComputeBuffer(N, 12);

        rb.NbrIdx.Dispose();
        rb.NbrIdx      = new ComputeBuffer(Math.Max(1, 6 * N), 4);
        rb.NbrIdx.SetData(nbrIdx);

        rb.RestNbr.Dispose();
        rb.RestNbr     = new ComputeBuffer(Math.Max(1, 6 * N), 12);
        rb.RestNbr.SetData(restNbr);

        rb.BendPairs.Dispose();
        rb.BendPairs   = new ComputeBuffer(Math.Max(1, bendPairCount), 32);
        rb.BendPairs.SetData(bpArr);

        rb.ErrMax.Dispose();
        rb.ErrMax      = new ComputeBuffer(1, 4);

        // Patch CornerCount / BendPairCount via reflection (they are private set properties).
        // BackgroundGrid.cornerCount was 8; we need the solver to use N.
        var t = typeof(ReconBuffers);
        t.GetProperty("CornerCount")  ?.SetValue(rb, N);
        t.GetProperty("BendPairCount")?.SetValue(rb, bendPairCount);

        return rb;
    }

    // ── Oracle (a): Free-fall via MassSpringSolver.Step ──────────────────────────────────────
    // One unpinned particle, no springs (Ks=Cs=Kb=Cb=0), ExtForce=(0,-g*m,0), mass=1.
    // Step solver 5 times (dt=0.02, T=0.1 s). Assert y ≈ -0.5*g*T^2 within 1e-3,
    // vy ≈ -g*T within 1e-3. Exercises: inner substep loop, D4 h-adapt, ErrMax readback.

    [Test]
    public void FreeFall_ViaSolverStep_MatchesAnalytic()
    {
        const int   N      = 1;
        const float g      = 9.81f;
        const float dt     = 0.02f;
        const int   frames = 5;      // T = 5 * 0.02 = 0.1 s
        const float T      = dt * frames;
        const float y0     = 0f;
        const float v0     = 0f;

        var cornerPos = new float3[] { new float3(0f, y0, 0f) };
        var vel       = new float3[] { new float3(0f, v0, 0f) };
        var mass      = new float[]  { 1f };
        var pinned    = new int[]    { 0 };           // unpinned
        var extForce  = new float3[] { new float3(0f, -g * mass[0], 0f) };  // gravity

        // No neighbors, no springs → all -1
        var nbrIdx  = new int[6 * N];
        for (int i = 0; i < nbrIdx.Length; i++) nbrIdx[i] = -1;
        var restNbr = new float3[6 * N];

        // One dead bend pair placeholder (BendPairCount must be >= 1 for ComputeBuffer)
        var bpArr = new BendPairGpu[] { new BendPairGpu { alive = 0 } };

        var cs     = LoadPhysicsCS();
        var solver = new MassSpringSolver(cs)
        {
            Ks = 0f, Cs = 0f, Kb = 0f, Cb = 0f,
            h  = dt   // start with a substep equal to dt so first frame uses one substep
        };

        var rb = MakeMinimalRb(N, 1, cornerPos, vel, mass, pinned, extForce,
                               nbrIdx, restNbr, bpArr);
        try
        {
            // Drive the REAL adaptive orchestrator
            for (int frame = 0; frame < frames; frame++)
                solver.Step(rb, dt);

            var posOut = new float3[N];
            var velOut = new float3[N];
            rb.CornerPos.GetData(posOut);
            rb.Vel.GetData(velOut);

            float yExpected  = y0 + v0 * T - 0.5f * g * T * T;
            float vyExpected = v0 - g * T;

            Assert.AreEqual(yExpected, posOut[0].y, 1e-3f,
                $"Free-fall y: expected {yExpected:G6}, got {posOut[0].y:G6}");
            Assert.AreEqual(vyExpected, velOut[0].y, 1e-3f,
                $"Free-fall vy: expected {vyExpected:G6}, got {velOut[0].y:G6}");
            Assert.AreEqual(0f, posOut[0].x, 1e-6f, "x must not move (no x-force)");
            Assert.AreEqual(0f, posOut[0].z, 1e-6f, "z must not move (no z-force)");
        }
        finally
        {
            rb.Dispose();
        }
    }

    // ── Oracle (b): 1-D SHM via MassSpringSolver.Step ────────────────────────────────────────
    // 2 particles: A pinned at origin, B free at (L0+A0, 0, 0).
    // One structural spring: Ks=100, Cs=0, Kb=0, Cb=0, mass=1, ExtForce=0.
    // ω = sqrt(Ks/m) = 10 rad/s → exact SHM about rest-length L0=0.25.
    // Step for T=0.05 s (e.g. 3 frames of dt≈0.0167, or 3 frames of dt=0.02 with T=0.06 s).
    // Assert x_B ≈ L0 + A0*cos(ω*T) within 2e-3.
    // Exercises: a-coefficients, Butcher tableau, orchestrator inner loop.

    [Test]
    public void Oscillator1D_ViaSolverStep_TracksAnalyticSHM()
    {
        const int   N   = 2;
        const float Ks  = 100f;
        const float m   = 1f;
        const float A0  = 0.01f;   // small displacement so linear spring approx holds
        const float L0  = 0.25f;
        const float dt  = 0.02f;
        const int   frames = 3;     // T ≈ 0.06 s (close enough to 0.05 s; analytic formula exact)
        const float T   = dt * frames;  // 0.06 s
        const float w   = 10f;      // ω = sqrt(100/1) = 10 rad/s

        // Particle A at origin (pinned), B at (L0+A0, 0, 0) (free)
        var cornerPos = new float3[]
        {
            float3.zero,
            new float3(L0 + A0, 0f, 0f)
        };
        var vel      = new float3[] { float3.zero, float3.zero };
        var mass     = new float[]  { m, m };
        var pinned   = new int[]    { 1, 0 };      // A pinned, B free
        var extForce = new float3[] { float3.zero, float3.zero };

        // Neighbor layout (6 slots per particle: [-x,+x,-y,+y,-z,+z]):
        // A: slot 1 (+x) → B; all others -1
        // B: slot 0 (-x) → A; all others -1
        var nbrIdx = new int[6 * N];
        for (int i = 0; i < nbrIdx.Length; i++) nbrIdx[i] = -1;
        nbrIdx[6 * 0 + 1] = 1;   // A's +x neighbor = B
        nbrIdx[6 * 1 + 0] = 0;   // B's -x neighbor = A

        var restNbr = new float3[6 * N];
        restNbr[6 * 0 + 1] = new float3( L0, 0f, 0f);  // A→B rest offset
        restNbr[6 * 1 + 0] = new float3(-L0, 0f, 0f);  // B→A rest offset

        // One dead bend pair placeholder
        var bpArr = new BendPairGpu[] { new BendPairGpu { alive = 0 } };

        var cs     = LoadPhysicsCS();
        var solver = new MassSpringSolver(cs)
        {
            Ks = Ks, Cs = 0f, Kb = 0f, Cb = 0f,
            h  = 1e-3f    // start small so D4 can widen as needed
        };

        var rb = MakeMinimalRb(N, 1, cornerPos, vel, mass, pinned, extForce,
                               nbrIdx, restNbr, bpArr);
        try
        {
            // Drive the REAL adaptive orchestrator
            for (int frame = 0; frame < frames; frame++)
                solver.Step(rb, dt);

            var posOut = new float3[N];
            rb.CornerPos.GetData(posOut);

            // Analytic SHM: x_B(T) = L0 + A0 * cos(ω * T)
            float xBExpected = L0 + A0 * Mathf.Cos(w * T);
            float xBActual   = posOut[1].x;

            Assert.AreEqual(xBExpected, xBActual, 2e-3f,
                $"SHM x_B: expected {xBExpected:G6}, got {xBActual:G6} " +
                $"(ω={w}, T={T:G4}, A0={A0})");

            // Sanity: A must stay pinned at origin
            Assert.AreEqual(0f, posOut[0].x, 1e-5f, "Particle A (pinned) must not move");

            // Sanity: displacement bounded within [L0-A0-0.5%, L0+A0+0.5%]
            Assert.Less(Mathf.Abs(xBActual - L0), A0 * 1.5f,
                $"SHM displacement {xBActual - L0:G4} should stay bounded near A0={A0}");
        }
        finally
        {
            rb.Dispose();
        }
    }
}
