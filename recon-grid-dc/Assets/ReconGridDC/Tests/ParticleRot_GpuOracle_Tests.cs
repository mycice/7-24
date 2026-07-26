// ParticleRot_GpuOracle_Tests.cs — Stage 2D-2 GPU oracle for ComputeParticleRot
// (paper §2.1.2 / Berndt [22] rotated axes; plan 2026-06-28-dc-2d2-berndt-rframe.md §6).
//
// ComputeParticleRot writes R = polar(F) into _ParticleRot[c], F = the local deformation gradient from
// corner c's INTACT incident edges. These tests drive a SINGLE center corner (index 0) with 6 axis
// neighbors (indices 1..6) and read back R for corner 0, exercising the math + all three identity guards.
//
// Cases (plan §6 / §2 verification additions):
//   (a) deformed = Rθ·rest (30° about an axis)      → R == Rθ (orthonormal, det≈1, ‖R−Rθ‖<1e-3)
//   (b) pure shear F                                 → R orthonormal, det≈1, R != F
//   (c) det F<0 (reflect one axis)                   → R == identity (M2 reflection guard)
//   (d) colinear survivors (only ±X intact)          → R == identity (M3 per-axis gate)
//   (e) ‖F−I‖<deadband (tiny stretch)                → R == identity (M4 deadband)
//   (f) LIVE-ish regression: small symmetric stretch ABOVE the deadband → ‖R−I‖<tol (near identity)
//
// Slot map (NbrDelta, mirrors Physics.compute / GridConventions): 0=-x,1=+x,2=-y,3=+y,4=-z,5=+z.
// [Category("GPU")] — requires a Unity GPU context. Dispatch pattern mirrors StructuralForce_GpuOracle.

using System.IO;
using NUnit.Framework;
using Unity.Mathematics;
using UnityEditor;
using UnityEngine;

[Category("GPU")]
public class ParticleRot_GpuOracle_Tests
{
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

    // ── Config: 1 center corner (index 0) + 6 axis neighbors (1..6) ───────────────────────────────
    const int N = 7;   // corner 0 = center, 1..6 = -x,+x,-y,+y,-z,+z neighbors

    // Rest neighbor offsets for the center corner (unit lattice spacing L=1). Slot order matches NbrDelta.
    static readonly float3[] RestDir =
    {
        new float3(-1, 0, 0),  // slot 0: -x
        new float3( 1, 0, 0),  // slot 1: +x
        new float3( 0,-1, 0),  // slot 2: -y
        new float3( 0, 1, 0),  // slot 3: +y
        new float3( 0, 0,-1),  // slot 4: -z
        new float3( 0, 0, 1),  // slot 5: +z
    };

    /// <summary>
    /// Run ComputeParticleRot for the center corner. <paramref name="def"/> applied to each rest
    /// direction gives the deformed neighbor position (center stays at origin). <paramref name="intact"/>
    /// (length 6) selects which slots are intact (true) vs severed (false → NbrIdx=-1). Returns R (rows).
    /// </summary>
    static float3x3 RunRot(System.Func<float3, float3> def, bool[] intact)
    {
        var positions = new float3[N];
        positions[0] = float3.zero;                 // center at origin
        for (int s = 0; s < 6; s++)
            positions[1 + s] = def(RestDir[s]);     // neighbor s = deformed rest dir

        // NbrIdx[6*N]: only the CENTER (corner 0) carries neighbors; neighbors have none (not read).
        var nbrIdx = new int[6 * N];
        for (int i = 0; i < nbrIdx.Length; i++) nbrIdx[i] = -1;
        for (int s = 0; s < 6; s++)
            nbrIdx[6 * 0 + s] = intact[s] ? (1 + s) : -1;

        // RestNbr[6*N]: center's rest offsets; neighbors' slots unused.
        var restNbr = new float3[6 * N];
        for (int s = 0; s < 6; s++) restNbr[6 * 0 + s] = RestDir[s];

        // All corners active + unpinned (center must compute; neighbors must be active to be included).
        var active = new int[N]; for (int i = 0; i < N; i++) active[i] = 1;
        var pinned = new int[N]; // all 0

        // _ParticleRot[N], stride 36 (RotGpu = 9 floats). Seed garbage to confirm the kernel writes it.
        var rotData = new float[9 * N];
        for (int i = 0; i < rotData.Length; i++) rotData[i] = 123.456f;

        var bufPos    = new ComputeBuffer(N, 12);
        var bufNbr    = new ComputeBuffer(6 * N, 4);
        var bufRest   = new ComputeBuffer(6 * N, 12);
        var bufActive = new ComputeBuffer(N, 4);
        var bufPinned = new ComputeBuffer(N, 4);
        var bufRot    = new ComputeBuffer(N, 36);

        try
        {
            bufPos.SetData(positions);
            bufNbr.SetData(nbrIdx);
            bufRest.SetData(restNbr);
            bufActive.SetData(active);
            bufPinned.SetData(pinned);
            bufRot.SetData(rotData);

            var cs = LoadPhysicsCS();
            int kRot = cs.FindKernel("ComputeParticleRot");
            cs.SetInt("_CornerCount", N);
            cs.SetBuffer(kRot, "_NbrIdx",      bufNbr);
            cs.SetBuffer(kRot, "_RestNbr",     bufRest);
            cs.SetBuffer(kRot, "_CornerPos",   bufPos);
            cs.SetBuffer(kRot, "_Active",      bufActive);
            cs.SetBuffer(kRot, "_Pinned",      bufPinned);
            cs.SetBuffer(kRot, "_ParticleRot", bufRot);
            cs.Dispatch(kRot, 1, 1, 1);

            var outRot = new float[9 * N];
            bufRot.GetData(outRot);

            // Corner 0's 9 floats are the HLSL ROWS of R: r0=(0,1,2), r1=(3,4,5), r2=(6,7,8). The kernel
            // uses HLSL row-major mul(R,v) = (dot(r0,v), dot(r1,v), dot(r2,v)). Unity.Mathematics float3x3
            // is COLUMN-major: its ctor args are COLUMNS and mul(M,v)=M.c0*v.x+M.c1*v.y+M.c2*v.z. To return
            // the SAME linear map, the j-th Unity column = (row0[j], row1[j], row2[j]). So pass the COLUMNS:
            float3 r0 = new float3(outRot[0], outRot[1], outRot[2]);
            float3 r1 = new float3(outRot[3], outRot[4], outRot[5]);
            float3 r2 = new float3(outRot[6], outRot[7], outRot[8]);
            return new float3x3(
                new float3(r0.x, r1.x, r2.x),   // Unity column 0 = (row0[0],row1[0],row2[0])
                new float3(r0.y, r1.y, r2.y),   // Unity column 1
                new float3(r0.z, r1.z, r2.z));  // Unity column 2
        }
        finally
        {
            bufPos.Dispose(); bufNbr.Dispose(); bufRest.Dispose();
            bufActive.Dispose(); bufPinned.Dispose(); bufRot.Dispose();
        }
    }

    static readonly bool[] AllIntact = { true, true, true, true, true, true };

    // ── 3x3 helpers (row-major float3x3; rows are r0,r1,r2 like the kernel) ────────────────────────
    static float Det(float3x3 m) => math.determinant(m);

    static float FrobDiff(float3x3 a, float3x3 b)
    {
        float3x3 d = a - b;
        return math.sqrt(math.dot(d.c0, d.c0) + math.dot(d.c1, d.c1) + math.dot(d.c2, d.c2));
    }

    // Orthonormality residual: ‖RᵀR − I‖_F.
    static float OrthoResidual(float3x3 R)
    {
        float3x3 g = math.mul(math.transpose(R), R);
        return FrobDiff(g, float3x3.identity);
    }

    // Build a row-major rotation about a unit axis by angle θ (Rodrigues). Returned so that mul(R, v)
    // rotates v (matching the kernel's row-major mul(R,v) convention).
    static float3x3 RotAxisAngle(float3 axis, float angleRad)
    {
        // Unity.Mathematics quaternion → float3x3 is the row-major rotation matrix with mul(m,v) = rotate.
        quaternion q = quaternion.AxisAngle(math.normalize(axis), angleRad);
        return new float3x3(q);
    }

    const float Tol = 1e-3f;

    // ── (a) deformed = Rθ·rest → R == Rθ ──────────────────────────────────────────────────────────
    [Test]
    public void Rotation30Deg_RecoversRotation()
    {
        float3x3 Rtheta = RotAxisAngle(new float3(0.3f, 0.8f, 0.5f), math.radians(30f));

        // deformed neighbor = Rθ · restDir (center at origin), so F = Rθ and polar(F) = Rθ.
        float3x3 R = RunRot(v => math.mul(Rtheta, v), AllIntact);

        Assert.Less(OrthoResidual(R), Tol, "R must be orthonormal (RᵀR≈I)");
        Assert.AreEqual(1f, Det(R), Tol, "det(R) must be +1 (proper rotation)");
        Assert.Less(FrobDiff(R, Rtheta), Tol, $"R must equal Rθ; ‖R−Rθ‖={FrobDiff(R, Rtheta)}");
    }

    // ── (b) pure shear F → R orthonormal, det≈1, R != F ──────────────────────────────────────────
    [Test]
    public void Shear_ReturnsOrthonormalPolarPart_NotF()
    {
        // F maps x-rest dir to (1, γ, 0): a pure shear (det=1, not orthonormal). Apply F to every rest dir.
        float gamma = 0.4f;
        // Row-major F with mul(F,v): row0=(1,γ,0), row1=(0,1,0), row2=(0,0,1) → F·(1,0,0)=(1,0,0)?? Build
        // explicitly so F·x picks up the shear in y: we want F·(1,0,0) = (1, γ, 0).
        float3x3 F = new float3x3(
            new float3(1, 0, 0),    // row 0
            new float3(gamma, 1, 0),// row 1 → (F·v).y = γ*v.x + v.y
            new float3(0, 0, 1));   // row 2

        float3x3 R = RunRot(v => math.mul(F, v), AllIntact);

        Assert.Less(OrthoResidual(R), Tol, "R must be orthonormal");
        Assert.AreEqual(1f, Det(R), Tol, "det(R) must be +1");
        Assert.Greater(FrobDiff(R, F), 0.05f, "R must be the rotation part, NOT the shear F itself");
    }

    // ── (c) det F<0 (reflect one axis) → R == identity (M2 reflection guard) ──────────────────────
    [Test]
    public void ReflectedF_NegDet_FallsBackToIdentity()
    {
        // Reflect x: F = diag(-1, 1, 1) → det = -1 < detEps → identity fallback (M2).
        float3x3 F = new float3x3(
            new float3(-1, 0, 0),
            new float3( 0, 1, 0),
            new float3( 0, 0, 1));

        float3x3 R = RunRot(v => math.mul(F, v), AllIntact);
        Assert.Less(FrobDiff(R, float3x3.identity), Tol, "det F<0 must fall back to identity (M2 guard)");
    }

    // ── (d) colinear survivors (only ±X intact) → R == identity (M3 per-axis gate) ────────────────
    [Test]
    public void ColinearSurvivors_OnlyX_FallsBackToIdentity()
    {
        // Sever -y,+y,-z,+z (slots 2,3,4,5): only the ±X axis survives → my=mz=0 → M3 per-axis gate.
        var intact = new[] { true, true, false, false, false, false };

        // Use a real rotation so a missing guard would yield a non-identity (NaN-poisoned) R.
        float3x3 Rtheta = RotAxisAngle(new float3(0, 0, 1), math.radians(20f));
        float3x3 R = RunRot(v => math.mul(Rtheta, v), intact);

        Assert.Less(FrobDiff(R, float3x3.identity), Tol,
            "colinear survivors (zero diagonal on y/z) must fall back to identity (M3 per-axis gate)");
    }

    // ── (e) ‖F−I‖<deadband (tiny stretch) → R == identity (M4 deadband) ───────────────────────────
    [Test]
    public void TinyStretch_BelowDeadband_FallsBackToIdentity()
    {
        // F = (1+ε)·I with ε tiny so ‖F−I‖_F = ε·√3 < deadband (1e-3). ε=1e-4 ⇒ ‖F−I‖≈1.7e-4.
        float eps = 1e-4f;
        float s = 1f + eps;
        float3x3 R = RunRot(v => v * s, AllIntact);
        Assert.Less(FrobDiff(R, float3x3.identity), Tol,
            "sub-deadband stretch must keep R≡I (M4 deadband protects the translation demo)");
    }

    // ── (f) LIVE-ish regression: small symmetric stretch ABOVE deadband → ‖R−I‖<tol ───────────────
    [Test]
    public void SmallSymmetricStretch_AboveDeadband_NearIdentity()
    {
        // A pure symmetric stretch (no rotation) just above the deadband: its polar rotation part is
        // EXACTLY identity (symmetric F ⇒ polar R = I). Choose ε so ‖F−I‖ > deadband (kernel computes R,
        // not the hand-bound identity) yet the recovered R stays within tol of I.
        float eps = 5e-3f;             // ‖F−I‖ ≈ 8.7e-3 > deadband(1e-3) → kernel runs the polar solve
        float s = 1f + eps;
        float3x3 R = RunRot(v => v * s, AllIntact);
        Assert.Less(FrobDiff(R, float3x3.identity), Tol,
            "a pure (rotation-free) stretch must yield R≈I from the live polar solve (LIVE regression)");
    }
}
