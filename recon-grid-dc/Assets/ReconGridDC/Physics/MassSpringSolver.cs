// MassSpringSolver.cs — C# orchestrator for the GPU mass-spring adaptive RK45 solver.
// Outer fixed dt=0.02s loop; inner adaptive-h substeps using Dormand-Prince RK45.
// Per-frame h adaptation via DEVIATION D4: h_new = h * 0.9 * (eps/delta)^(1/5).
// DEVIATION D6: per-frame retrospective h-adaptation (read cross-substep max error once/frame,
// update h via D4 formula for next frame) to avoid a GPU->CPU stall every substep;
// RK stage math + D4 formula unchanged.

using UnityEngine;
using ReconGridDC.Recon;

namespace ReconGridDC.Physics
{
    public sealed class MassSpringSolver
    {
        // ── Kernel handles ──────────────────────────────────────────────────────────────────────
        readonly ComputeShader cs;
        readonly int kSeed;
        readonly int kClear;
        readonly int kStruct;
        readonly int kBend;
        readonly int kBuild;
        readonly int kSlope;
        readonly int kIntegrate;
        readonly int kError;
        readonly int kRot;     // Stage 2D-2: ComputeParticleRot (paper §2.1.2 / Berndt [22])

        // ── Physics parameters ──────────────────────────────────────────────────────────────────
        public float Ks = 7.5e4f;   // structural stiffness
        public float Cs = 0.92f;    // structural damping
        public float Kb = 2e4f;     // bending stiffness
        public float Cb = 0.9f;     // bending damping

        // ── Adaptive step state ─────────────────────────────────────────────────────────────────
        /// <summary>Current adaptive substep size h; persists across frames.</summary>
        public float h = 1e-4f;

        /// <summary>Absolute upper bound on the substep h (explicit-RK stability safeguard).
        /// Because D6 adapts h only ONCE per frame, the per-step embedded error cannot reject a
        /// too-large h mid-frame; at the paper-stiff ks (7.5e4, omega~950 rad/s) the DoPri5 stability
        /// limit is h≲~3e-3. hMax keeps omega*h well inside the stable region (>=~40 substeps/frame
        /// at dt=0.02), preventing the per-frame controller from "hunting" past the boundary -> jitter.
        /// Tunable; lower it for stiffer ks. (Realization safeguard alongside D6.)</summary>
        public float hMax = 5e-4f;

        /// <summary>HARD cap on substeps per Step() call — the single-thread realization of the paper's
        /// §3.2 thread-decoupling. Bounds the per-frame physics GPU cost so a ringing/stiff lattice
        /// (whose adaptive h collapses toward the 1e-6 floor) cannot run thousands of substeps and peg the
        /// GPU / block the render thread. When the cap is hit before covering dt, the sim advances LESS than
        /// dt this frame (slow-motion under heavy stiffness — exactly the paper's accepted "thread delay"),
        /// but render FPS stays bounded. Replaces the old 200000 guard. Raise cs_damp/mass (faster settle →
        /// larger h → fewer substeps needed) toward real-time, or raise this to trade FPS for sim speed.</summary>
        public int maxSubstepsPerFrame = 16;
        int _capLogThrottle;

        // Tolerance and scale for DEVIATION D4 step-size adaptation
        const float Eps      = 1e-3f;   // error tolerance (D4)
        const float ErrScale = 1e6f;    // must match ERR_SCALE in PhysicsCommon.hlsl

        // ── Construction ────────────────────────────────────────────────────────────────────────
        public MassSpringSolver(ComputeShader physics)
        {
            cs         = physics;
            kSeed      = cs.FindKernel("SeedTrialFromState");
            kClear     = cs.FindKernel("ClearForces");
            kStruct    = cs.FindKernel("AccumulateStructural");
            kBend      = cs.FindKernel("AccumulateBending");
            kBuild     = cs.FindKernel("BuildTrialState");
            kSlope     = cs.FindKernel("ComputeSlope");
            kIntegrate = cs.FindKernel("IntegrateY5");
            kError     = cs.FindKernel("ComputeError");
            kRot       = cs.FindKernel("ComputeParticleRot");  // Stage 2D-2 (paper §2.1.2)
        }

        // ─────────────────────────────────────────────────────────────────────────────────────────
        /// <summary>
        /// Advance the simulation by one outer fixed dt (default 0.02 s) using inner adaptive substeps.
        /// Binds all buffers from <paramref name="rb"/> and dispatches all physics kernels.
        /// </summary>
        public void Step(ReconBuffers rb, float dt)
        {
            // Upload shared scalar uniforms once per call
            cs.SetInt("_CornerCount",   rb.CornerCount);
            cs.SetInt("_BendPairCount", rb.BendPairCount);
            cs.SetFloat("_Ks", Ks);
            cs.SetFloat("_Cs", Cs);
            cs.SetFloat("_Kb", Kb);
            cs.SetFloat("_Cb", Cb);

            // Reset the error accumulator for this frame
            rb.ErrMax.SetData(new uint[] { 0 });

            // ── Inner adaptive substep loop ──────────────────────────────────────────────────────
            float t     = 0f;
            int   guard = 0;

            while (t < dt - 1e-9f && guard++ < maxSubstepsPerFrame)
            {
                // PAPER-SILENT (last-substep clip): clip so substeps sum exactly to dt
                float hStep = Mathf.Min(h, dt - t);
                cs.SetFloat("_H", hStep);

                // Stage 0: seed trial state from committed y_n
                Dispatch(kSeed, rb);

                for (int s = 0; s < 7; s++)
                {
                    cs.SetInt("_Stage", s);

                    // For stages 1-6 build the trial state from accumulated k slopes
                    if (s > 0)
                        Dispatch(kBuild, rb);

                    // Accumulate forces at trial state
                    Dispatch(kClear,  rb);
                    Dispatch(kStruct, rb);
                    DispatchBend(rb);

                    // Compute slope k[s] = f(t, yTrial)
                    cs.SetInt("_Stage", s);  // re-set after DispatchBend potentially changed nothing
                    Dispatch(kSlope, rb);
                }

                // Error estimate for D4 step-size adaptation (InterlockedMax into ErrMax)
                Dispatch(kError, rb);

                // Integrate committed state: y_{n+1} = y_n + h * sum(B5 * k)
                // PAPER-SILENT (always-accept): never reject; adjust h for next substep
                Dispatch(kIntegrate, rb);

                t += hStep;
            }
            // Substep-budget cap (paper §3.2 decoupling, single-thread realization): if the loop hit the
            // cap before covering dt, the lattice is ringing (adaptive h collapsing) → the sim runs in
            // slow-motion THIS frame to keep render FPS bounded instead of pegging the GPU on thousands of
            // substeps. Raise cs_damp/massScale (settle faster → larger h) or maxSubstepsPerFrame.
            if (guard >= maxSubstepsPerFrame && t < dt - 1e-9f && (_capLogThrottle++ % 120) == 0)
                Debug.LogWarning($"[MassSpringSolver] substep cap {maxSubstepsPerFrame} hit (h={h:E2}, covered {t / dt:P0} of dt) — sim slow-motion; raise cs_damp/massScale (damping) or maxSubstepsPerFrame.");

            // ── Per-frame h adaptation (DEVIATION D4 + D6) ──────────────────────────────────────────
            // DEVIATION D6: per-frame retrospective h-adaptation (read cross-substep max error
            // once/frame, update h via D4 for next frame) to avoid a GPU->CPU stall every substep;
            // RK stage math + D4 formula unchanged.
            var em    = new uint[1];
            rb.ErrMax.GetData(em);

            float delta = Mathf.Max(em[0] / ErrScale, 1e-12f);

            // DEVIATION D4: h_new = h * 0.9 * (eps/delta)^(1/5), ratio is eps/delta (not delta/eps)
            float hn = h * 0.9f * Mathf.Pow(Eps / delta, 0.2f);  // DEVIATION D4

            // Clamp relative: must stay in [0.2*h, 5*h]
            hn = Mathf.Clamp(hn, 0.2f * h, 5f * h);

            // Clamp absolute: must stay in [1e-6, hMax]. hMax (<= dt) is the explicit-RK stability
            // safeguard for the per-frame D6 controller (see field doc); also never exceed dt.
            h = Mathf.Clamp(hn, 1e-6f, Mathf.Min(hMax, dt));
        }

        // ─────────────────────────────────────────────────────────────────────────────────────────
        /// <summary>
        /// Stage 2D-2 (paper §2.1.2 / Berndt [22]): compute each particle's rotation R = polar(F) into
        /// rb.ParticleRot. ORDER (M1): dispatch AFTER cutDetector.Dispatch (so SeverLinks' NbrIdx=-1 is
        /// visible → severed edges excluded from F) and BEFORE dc.Build (AccumulateCutFP reads frame N's
        /// R). One thread per corner. Binds the topology/state buffers ComputeParticleRot reads.
        /// </summary>
        public void ComputeRotations(ReconBuffers rb)
        {
            cs.SetInt("_CornerCount", rb.CornerCount);

            cs.SetBuffer(kRot, "_NbrIdx",      rb.NbrIdx);
            cs.SetBuffer(kRot, "_RestNbr",     rb.RestNbr);
            cs.SetBuffer(kRot, "_CornerPos",   rb.CornerPos);
            cs.SetBuffer(kRot, "_Active",      rb.CornerActive);
            cs.SetBuffer(kRot, "_Pinned",      rb.Pinned);
            cs.SetBuffer(kRot, "_ParticleRot", rb.ParticleRot);

            cs.Dispatch(kRot, Groups(rb.CornerCount), 1, 1);
        }

        // ─────────────────────────────────────────────────────────────────────────────────────────
        // Helpers

        static int Groups(int n) => Mathf.Max(1, Mathf.CeilToInt(n / 64f));

        void Dispatch(int k, ReconBuffers rb)
        {
            Bind(k, rb);
            cs.Dispatch(k, Groups(rb.CornerCount), 1, 1);
        }

        void DispatchBend(ReconBuffers rb)
        {
            Bind(kBend, rb);
            cs.Dispatch(kBend, Groups(rb.BendPairCount), 1, 1);
        }

        /// <summary>
        /// Bind all physics buffers for kernel <paramref name="k"/>.
        /// Both read (_YTrialPos/_YTrialVel) and RW (_YTrialPosRW/_YTrialVelRW) names
        /// are bound to the SAME underlying buffers (rb.YTrialPos / rb.YTrialVel).
        /// Force kernels use the read-only names; RK stage kernels use the RW names.
        /// </summary>
        void Bind(int k, ReconBuffers rb)
        {
            cs.SetBuffer(k, "_CornerPos",   rb.CornerPos);
            cs.SetBuffer(k, "_Vel",         rb.Vel);
            cs.SetBuffer(k, "_Mass",        rb.Mass);
            cs.SetBuffer(k, "_Pinned",      rb.Pinned);
            cs.SetBuffer(k, "_Active",      rb.CornerActive);  // Stage 2D-1b: tissue-only physics (paper §2.1.4)
            cs.SetBuffer(k, "_ExtForce",    rb.ExtForce);
            cs.SetBuffer(k, "_ForceInt",    rb.ForceInt);
            cs.SetBuffer(k, "_NbrIdx",      rb.NbrIdx);
            cs.SetBuffer(k, "_RestNbr",     rb.RestNbr);
            cs.SetBuffer(k, "_BendPairs",   rb.BendPairs);
            cs.SetBuffer(k, "_KSlope",      rb.KSlope);
            cs.SetBuffer(k, "_ErrMax",      rb.ErrMax);
            // Both names point at the same buffer; Unity tolerates same buffer for read and RW
            cs.SetBuffer(k, "_YTrialPos",   rb.YTrialPos);
            cs.SetBuffer(k, "_YTrialVel",   rb.YTrialVel);
            cs.SetBuffer(k, "_YTrialPosRW", rb.YTrialPos);
            cs.SetBuffer(k, "_YTrialVelRW", rb.YTrialVel);
        }
    }
}
