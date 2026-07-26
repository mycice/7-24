// StructuralForce_GpuOracle_Tests.cs — GPU oracle for structural force kernel (Task 3).
// Verifies ClearForces + AccumulateStructural against an analytic hand-computed case.
// Requires a GPU; run with [Category("GPU")] filter on a GPU machine.

using System.IO;
using NUnit.Framework;
using Unity.Mathematics;
using UnityEditor;
using UnityEngine;

[Category("GPU")]
public class StructuralForce_GpuOracle_Tests
{
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

    // ── Test: stretched spring produces correct force ─────────────────────────────────────────────
    // Setup: 2 particles.
    //   p0 at (0,0,0), p1 at (0.3, 0, 0).  L0 = 0.25, Ks = 7.5e4, Cs = 0 (zero velocities).
    // Expected structural force on p0:
    //   dir = normalize(p0 - p1) = (-1,0,0)
    //   F = -Ks*(|d|-L0)*dir = -7.5e4*(0.3-0.25)*(-1,0,0) = (3750, 0, 0)
    // p1 is not pinned but its force from p0's perspective is the mirror; we read p0's force.

    [Test]
    public void StretchedSpring_CorrectForceMagnitude()
    {
        const int   N    = 2;
        const float Ks   = 7.5e4f;
        const float Cs   = 0f;
        const float L0   = 0.25f;
        const float dist = 0.3f;

        // Positions: p0=(0,0,0), p1=(dist,0,0)
        var positions = new float3[] { new float3(0, 0, 0), new float3(dist, 0, 0) };
        var velocities = new float3[] { float3.zero, float3.zero };

        // Neighbor index: p0 has p1 as +x neighbor (slot 1); p1 has p0 as -x neighbor (slot 0).
        // Layout: int[6*N], slot order: -x,+x,-y,+y,-z,+z
        var nbrIdx = new int[] {
            -1, 1, -1, -1, -1, -1,   // p0: +x neighbor is p1
            0, -1, -1, -1, -1, -1    // p1: -x neighbor is p0
        };

        // Rest neighbor offsets: float3[6*N]
        // p0 slot 1 (+x): restNbr = p1 - p0 = (L0,0,0)  -> length = L0
        // p1 slot 0 (-x): restNbr = p0 - p1 = (-L0,0,0) -> length = L0
        var restNbr = new float3[] {
            float3.zero, new float3(L0, 0, 0), float3.zero, float3.zero, float3.zero, float3.zero,
            new float3(-L0, 0, 0), float3.zero, float3.zero, float3.zero, float3.zero, float3.zero
        };

        // Force accumulator: int[3*N]
        var forceInt = new int[3 * N];

        // Create GPU buffers
        var bufPos    = new ComputeBuffer(N, 12);
        var bufVel    = new ComputeBuffer(N, 12);
        var bufNbr    = new ComputeBuffer(6 * N, 4);
        var bufRest   = new ComputeBuffer(6 * N, 12);
        var bufForce  = new ComputeBuffer(3 * N, 4);
        var bufActive = new ComputeBuffer(N, 4);   // 2D-1b: AccumulateStructural now reads _Active[n]

        try
        {
            bufPos.SetData(positions);
            bufVel.SetData(velocities);
            bufNbr.SetData(nbrIdx);
            bufRest.SetData(restNbr);
            bufForce.SetData(forceInt);
            bufActive.SetData(new int[] { 1, 1 });   // both particles active (tissue)

            var cs = LoadPhysicsCS();
            cs.SetInt("_CornerCount", N);
            cs.SetFloat("_Ks", Ks);
            cs.SetFloat("_Cs", Cs);

            // ClearForces
            int kClear = cs.FindKernel("ClearForces");
            cs.SetBuffer(kClear, "_ForceInt", bufForce);
            cs.Dispatch(kClear, 1, 1, 1);

            // AccumulateStructural
            int kStruct = cs.FindKernel("AccumulateStructural");
            cs.SetBuffer(kStruct, "_ForceInt",   bufForce);
            cs.SetBuffer(kStruct, "_YTrialPos",  bufPos);
            cs.SetBuffer(kStruct, "_YTrialVel",  bufVel);
            cs.SetBuffer(kStruct, "_NbrIdx",     bufNbr);
            cs.SetBuffer(kStruct, "_RestNbr",    bufRest);
            cs.SetBuffer(kStruct, "_Active",     bufActive);   // 2D-1b: tissue-only gate
            cs.Dispatch(kStruct, 1, 1, 1);

            // Readback
            bufForce.GetData(forceInt);

            // Decode fixed-point force on p0
            const float FS = 1024f;  // FORCE_SCALE
            float3 f0 = new float3(forceInt[0] / FS, forceInt[1] / FS, forceInt[2] / FS);

            // p0 is displaced toward p1 (+x) so the spring pulls it in +x direction
            // dir from p0 to p1 = +x; spring pushes p0 toward p1 => F_on_p0 = +x direction
            // F = -Ks*(|d|-L0)*dir_p0_minus_p1  where dir = (p0-p1)/|p0-p1| = (-1,0,0)
            // => F_on_p0 = -7.5e4*(0.3-0.25)*(-1,0,0) = (3750, 0, 0)
            float expectedFx = Ks * (dist - L0);  // 3750
            Assert.AreEqual(expectedFx, f0.x, 5f,
                $"Force on p0 x-component expected ~{expectedFx}, got {f0.x}");
            Assert.AreEqual(0f, f0.y, 1f, "Force y should be near zero");
            Assert.AreEqual(0f, f0.z, 1f, "Force z should be near zero");

            // Also verify Newton's 3rd law: force on p1 should be approximately -(force on p0)
            float3 f1 = new float3(forceInt[3] / FS, forceInt[4] / FS, forceInt[5] / FS);
            Assert.AreEqual(-expectedFx, f1.x, 5f,
                $"Force on p1 x-component expected ~{-expectedFx}, got {f1.x}");
        }
        finally
        {
            bufPos.Dispose(); bufVel.Dispose(); bufNbr.Dispose();
            bufRest.Dispose(); bufForce.Dispose(); bufActive.Dispose();
        }
    }

    // ── Test: zero extension produces zero force ──────────────────────────────────────────────────

    [Test]
    public void ZeroExtension_ZeroForce()
    {
        const int   N  = 2;
        const float Ks = 7.5e4f;
        const float L0 = 0.25f;

        // Place particles exactly at rest length: no extension, no force
        var positions = new float3[] { new float3(0, 0, 0), new float3(L0, 0, 0) };
        var velocities = new float3[] { float3.zero, float3.zero };

        var nbrIdx = new int[] {
            -1, 1, -1, -1, -1, -1,
            0, -1, -1, -1, -1, -1
        };
        var restNbr = new float3[] {
            float3.zero, new float3(L0, 0, 0), float3.zero, float3.zero, float3.zero, float3.zero,
            new float3(-L0, 0, 0), float3.zero, float3.zero, float3.zero, float3.zero, float3.zero
        };
        var forceInt = new int[3 * N];

        var bufPos   = new ComputeBuffer(N, 12);
        var bufVel   = new ComputeBuffer(N, 12);
        var bufNbr   = new ComputeBuffer(6 * N, 4);
        var bufRest  = new ComputeBuffer(6 * N, 12);
        var bufForce = new ComputeBuffer(3 * N, 4);
        var bufActive = new ComputeBuffer(N, 4);   // 2D-1b: AccumulateStructural now reads _Active[n]

        try
        {
            bufPos.SetData(positions);
            bufVel.SetData(velocities);
            bufNbr.SetData(nbrIdx);
            bufRest.SetData(restNbr);
            bufForce.SetData(forceInt);
            bufActive.SetData(new int[] { 1, 1 });   // both particles active (tissue)

            var cs = LoadPhysicsCS();
            cs.SetInt("_CornerCount", N);
            cs.SetFloat("_Ks", Ks);
            cs.SetFloat("_Cs", 0f);

            int kClear = cs.FindKernel("ClearForces");
            cs.SetBuffer(kClear, "_ForceInt", bufForce);
            cs.Dispatch(kClear, 1, 1, 1);

            int kStruct = cs.FindKernel("AccumulateStructural");
            cs.SetBuffer(kStruct, "_ForceInt",  bufForce);
            cs.SetBuffer(kStruct, "_YTrialPos", bufPos);
            cs.SetBuffer(kStruct, "_YTrialVel", bufVel);
            cs.SetBuffer(kStruct, "_NbrIdx",    bufNbr);
            cs.SetBuffer(kStruct, "_RestNbr",   bufRest);
            cs.SetBuffer(kStruct, "_Active",    bufActive);   // 2D-1b: tissue-only gate
            cs.Dispatch(kStruct, 1, 1, 1);

            bufForce.GetData(forceInt);

            const float FS = 1024f;
            float3 f0 = new float3(forceInt[0] / FS, forceInt[1] / FS, forceInt[2] / FS);
            Assert.AreEqual(0f, f0.x, 1f, "No extension: Fx should be zero");
            Assert.AreEqual(0f, f0.y, 1f);
            Assert.AreEqual(0f, f0.z, 1f);
        }
        finally
        {
            bufPos.Dispose(); bufVel.Dispose(); bufNbr.Dispose();
            bufRest.Dispose(); bufForce.Dispose(); bufActive.Dispose();
        }
    }
}
