// BendingForce_GpuOracle_Tests.cs — GPU oracle for bending force kernel (Task 4).
// Three cases: (a) collinear at rest -> guard skips, no NaN; (b) right-angle perturbed -> restoring
// force; (c) momentum conservation: sum of forces on i, j, k ≈ 0.
// Requires a GPU; run with [Category("GPU")] filter on a GPU machine.

using System.IO;
using NUnit.Framework;
using Unity.Mathematics;
using UnityEditor;
using UnityEngine;

[Category("GPU")]
public class BendingForce_GpuOracle_Tests
{
    // ── GPU struct matching BendPairGpu (32 bytes) ───────────────────────────────────────────────
    struct BendPairGpu
    {
        public int   i;
        public int   j;
        public int   k;
        public float theta0;
        public int   alive;
        public int   _p0, _p1, _p2;
    }

    // ── Helpers ──────────────────────────────────────────────────────────────────────────────────

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

    // Upload 3-particle bending setup and run AccumulateBending (no structural force).
    // Returns decoded forces on [p0, p1, p2] in order: center=0, arm1=1, arm2=2.
    static (float3 fi, float3 fj, float3 fk) RunBendKernel(
        float3 xi, float3 xj, float3 xk,
        float3 vi, float3 vj, float3 vk,
        float theta0, float Kb, float Cb)
    {
        const int N = 3;
        var positions  = new float3[] { xi, xj, xk };
        var velocities = new float3[] { vi, vj, vk };
        var forceInt   = new int[3 * N];

        // One bending pair: center=0 (i), arm j=1, arm k=2
        var bp = new BendPairGpu
        {
            i = 0, j = 1, k = 2,
            theta0 = theta0,
            alive  = 1,
            _p0 = 0, _p1 = 0, _p2 = 0
        };
        var bpArr = new BendPairGpu[] { bp };

        var bufPos   = new ComputeBuffer(N, 12);
        var bufVel   = new ComputeBuffer(N, 12);
        var bufBP    = new ComputeBuffer(1, 32);
        var bufForce = new ComputeBuffer(3 * N, 4);

        try
        {
            bufPos.SetData(positions);
            bufVel.SetData(velocities);
            bufBP.SetData(bpArr);
            bufForce.SetData(forceInt);

            var cs = LoadPhysicsCS();
            cs.SetInt("_CornerCount",   N);
            cs.SetInt("_BendPairCount", 1);
            cs.SetFloat("_Kb", Kb);
            cs.SetFloat("_Cb", Cb);

            // Clear forces first
            int kClear = cs.FindKernel("ClearForces");
            cs.SetBuffer(kClear, "_ForceInt", bufForce);
            cs.Dispatch(kClear, 1, 1, 1);

            // Accumulate bending
            int kBend = cs.FindKernel("AccumulateBending");
            cs.SetBuffer(kBend, "_ForceInt",  bufForce);
            cs.SetBuffer(kBend, "_YTrialPos", bufPos);
            cs.SetBuffer(kBend, "_YTrialVel", bufVel);
            cs.SetBuffer(kBend, "_BendPairs", bufBP);
            cs.Dispatch(kBend, 1, 1, 1);

            bufForce.GetData(forceInt);

            const float FS = 1024f;
            float3 fi = new float3(forceInt[0] / FS, forceInt[1] / FS, forceInt[2] / FS);
            float3 fj = new float3(forceInt[3] / FS, forceInt[4] / FS, forceInt[5] / FS);
            float3 fk = new float3(forceInt[6] / FS, forceInt[7] / FS, forceInt[8] / FS);
            return (fi, fj, fk);
        }
        finally
        {
            bufPos.Dispose(); bufVel.Dispose(); bufBP.Dispose(); bufForce.Dispose();
        }
    }

    // ── Test (a): collinear triple — singularity guard should produce near-zero / no NaN ─────────
    // i at origin, j=-x, k=+x. Nij = +x, Nik = -x => c = dot(+x,-x) = -1 => s2=0 => skip pair.

    [Test]
    public void CollinearTriple_GuardSkips_NearZeroNoNaN()
    {
        float3 xi = float3.zero;
        float3 xj = new float3(-1f, 0f, 0f);
        float3 xk = new float3( 1f, 0f, 0f);

        // theta0 = pi (collinear at rest); no velocity
        float theta0 = math.PI;
        var (fi, fj, fk) = RunBendKernel(xi, xj, xk, float3.zero, float3.zero, float3.zero,
                                          theta0, Kb: 2e4f, Cb: 0.9f);

        // Guard should skip: all forces zero (no NaN, no Inf)
        Assert.IsFalse(float.IsNaN(fi.x) || float.IsNaN(fi.y) || float.IsNaN(fi.z), "fi must not be NaN");
        Assert.IsFalse(float.IsNaN(fj.x) || float.IsNaN(fj.y) || float.IsNaN(fj.z), "fj must not be NaN");
        Assert.IsFalse(float.IsNaN(fk.x) || float.IsNaN(fk.y) || float.IsNaN(fk.z), "fk must not be NaN");

        float mag = math.length(fi) + math.length(fj) + math.length(fk);
        Assert.AreEqual(0f, mag, 1f, "Collinear pair: guard should produce near-zero total force");
    }

    // ── Test (b): right-angle triple perturbed from rest — restoring force ────────────────────────
    // i at origin, j=+x, k=+y. theta_rest = pi/2.
    // Perturb: displace j to (1, 0.1, 0) => theta slightly < pi/2 => Kb*(theta - theta0) < 0 =>
    // force should act to restore (push j away from k direction, increase angle back).

    [Test]
    public void RightAngle_Perturbed_RestoringForce()
    {
        float3 xi = float3.zero;
        float3 xj = new float3(1f, 0.1f, 0f);   // perturbed: slightly toward k
        float3 xk = new float3(0f, 1f, 0f);

        float theta0 = math.PI * 0.5f;  // 90 degrees rest

        var (fi, fj, fk) = RunBendKernel(xi, xj, xk, float3.zero, float3.zero, float3.zero,
                                          theta0, Kb: 2e4f, Cb: 0f);

        // j is perturbed TOWARD k (k is at +y), so the current angle (~84°) < theta0 (90°).
        // The bending restoring force must push j AWAY from k, i.e. in the -y direction
        // (drive j.y from 0.1 back toward 0 so the angle reopens to 90°).
        // NOTE: the kernel computes Nij = normalize(x_i - x_j) (pointing from j TOWARD i), so the
        // original "+y" expectation (which assumed Nij = x_j - x_i) was inverted. With the kernel's
        // convention the restoring force on j is correctly -y. (Magnitude ~1972 with Kb=2e4.)
        Assert.Less(fj.y, 0f,
            $"Restoring force on j should have NEGATIVE y (push j away from k at +y); got {fj.y}");

        // Total force magnitude on j should be nonzero
        Assert.Greater(math.length(fj), 0.1f, "Restoring force on j should be nonzero");
    }

    // ── Test (c): momentum conservation — Σ(fi + fj + fk) ≈ 0 ──────────────────────────────────
    // For any bending pair the center receives -(F_ij + F_ik), so total force = 0.

    [Test]
    public void BendingForce_MomentumConservation()
    {
        float3 xi = float3.zero;
        float3 xj = new float3(1f, 0f, 0f);
        float3 xk = new float3(0f, 1f, 0f);
        float theta0 = math.PI * 0.5f;

        // Perturb so there IS a nonzero force, but sum should still be ~0
        float3 xjPerturbed = new float3(1f, 0.2f, 0f);

        var (fi, fj, fk) = RunBendKernel(xi, xjPerturbed, xk, float3.zero, float3.zero, float3.zero,
                                          theta0, Kb: 2e4f, Cb: 0f);

        float3 total = fi + fj + fk;
        Assert.AreEqual(0f, total.x, 2f, $"Total force x should be ~0 (got {total.x})");
        Assert.AreEqual(0f, total.y, 2f, $"Total force y should be ~0 (got {total.y})");
        Assert.AreEqual(0f, total.z, 2f, $"Total force z should be ~0 (got {total.z})");
    }

    // ── Test (d): alive=0 pair produces no force ──────────────────────────────────────────────────

    [Test]
    public void DeadPair_ZeroForce()
    {
        const int N = 3;
        float3 xi = float3.zero;
        float3 xj = new float3(1f, 0f, 0f);
        float3 xk = new float3(0f, 1f, 0f);

        var positions  = new float3[] { xi, xj, xk };
        var velocities = new float3[] { float3.zero, float3.zero, float3.zero };
        var forceInt   = new int[3 * N];

        // Mark pair as dead
        var bp = new BendPairGpu { i = 0, j = 1, k = 2, theta0 = math.PI * 0.5f, alive = 0 };
        var bpArr = new BendPairGpu[] { bp };

        var bufPos   = new ComputeBuffer(N, 12);
        var bufVel   = new ComputeBuffer(N, 12);
        var bufBP    = new ComputeBuffer(1, 32);
        var bufForce = new ComputeBuffer(3 * N, 4);

        try
        {
            bufPos.SetData(positions);
            bufVel.SetData(velocities);
            bufBP.SetData(bpArr);
            bufForce.SetData(forceInt);

            var cs = LoadPhysicsCS();
            cs.SetInt("_CornerCount",   N);
            cs.SetInt("_BendPairCount", 1);
            cs.SetFloat("_Kb", 2e4f);
            cs.SetFloat("_Cb", 0f);

            int kClear = cs.FindKernel("ClearForces");
            cs.SetBuffer(kClear, "_ForceInt", bufForce);
            cs.Dispatch(kClear, 1, 1, 1);

            int kBend = cs.FindKernel("AccumulateBending");
            cs.SetBuffer(kBend, "_ForceInt",  bufForce);
            cs.SetBuffer(kBend, "_YTrialPos", bufPos);
            cs.SetBuffer(kBend, "_YTrialVel", bufVel);
            cs.SetBuffer(kBend, "_BendPairs", bufBP);
            cs.Dispatch(kBend, 1, 1, 1);

            bufForce.GetData(forceInt);

            for (int c = 0; c < 3 * N; c++)
                Assert.AreEqual(0, forceInt[c], $"Dead pair: force[{c}] should be zero");
        }
        finally
        {
            bufPos.Dispose(); bufVel.Dispose(); bufBP.Dispose(); bufForce.Dispose();
        }
    }
}
