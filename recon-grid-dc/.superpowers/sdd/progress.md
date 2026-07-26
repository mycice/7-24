# Stage 1A SDD Progress Ledger

Plan: docs/superpowers/plans/2026-06-27-dc-stage1a-recon.md
Branch: feat/dc-stage1a-recon

- (none complete yet)
- C1 (Tasks 1-5, CPU C# core): complete (commits c935837..8420fbb, review clean after fixes 8420fbb)
- C2 (Tasks 6-8, compute kernels): complete (commits 8420fbb..e693036; 2 reviewers opus; fixed IsectGpu 32->40 stride [caught real bug], D5 off-by-one, +non-stationary QEF oracle)
- C3 (Tasks 9-10 + deferred oracles): complete (commits e693036..7738e1b; reviewer opus; Approved, dispatch order/binding-map/TriCounter-reset verified; outward-normal oracle added)
  Minor (for final review): C3-M1 fixed-point truncation (cosmetic); C3-M2 DC_NormalsScatter over-dispatch at G(TriCapacity) -> optimize in 1B
- FINAL whole-branch review: READY TO MERGE (opus; 0 blocking; full CPU->GPU->render trace consistent; non-blocking C3-M1/M2 + winding visual-check tracked for 1B)
- Stage 1A COMPLETE (commits c935837..7738e1b on feat/dc-stage1a-recon)

## Stage 1B-i (physics solver) — SDD
Plan: docs/superpowers/plans/2026-06-27-dc-stage1b-i-physics.md
Clusters: C1=Tasks1-2 (preprocess+buffers); C2=Tasks3-5 (kernels+solver); final review=Task6.
- (1B-i in progress)
- 1B-i C1 (Tasks 1-2): complete (commits 845c05a..ccea020; reviewer sonnet; Approved). Minor(final): rename shadowed local cornerCount in Build(); fix Upload() zero-init comment (runtime safe: solver clears ForceInt/ErrMax + seeds YTrial; ExtForce set by caller before Step).
- 1B-i C2 (Tasks 3-5): complete (commits 136929f..de34a57; 2 opus reviewers; Butcher tableau verified row-sums=c & b-sums=1, D2/D3/D4 + bending/theta_dot exact). Fix de34a57: RK45 oracles now drive MassSpringSolver.Step (free-fall + 1D SHM) + substep guard raised. Minor(final): fixed-point truncation (cosmetic, pre-existing); YTrial SRV+UAV dual-binding OK on DX11 (could ping-pong for strict backends); ComputeSlope reads _YTrialVelRW view.
- 1B-i FINAL whole-branch review: READY (opus; 0 blocking; tableau/deviations/pinned/oracles/fixture all verified; 1A intact).
- Stage 1B-i COMPLETE (commits 845c05a..de34a57). Track-for-1B-ii: commit new .meta after Unity import; structural-atomics->gather perf; cosmetic minors.

## Stage 1B-ii (deformation coupling) — SDD
Plan: docs/superpowers/plans/2026-06-27-dc-stage1b-ii-coupling.md
Clusters: C1=Task1 (RebuildIsectWorld+IsectWorld+DC wiring); C2=Tasks2-3 (per-frame manager + integration test); final review.
- (1B-ii in progress)
- 1B-ii C1 (Task 1): complete (commit f27c4ee..6c412a3; reviewer opus; Approved; 1A NO-REGRESSION preserved [rest==globalRest structurally]). Nit: stale Recon.compute header comment.
- 1B-ii C2 (Tasks 2-3): complete (commits 6c412a3..9ff737b; reviewer opus; Approved; milestone wired). Fix 9ff737b: DeformCoupling smoke strengthened (soft spring Ks=500/Kb=100, 40 frames, strict min-Y decrease) to truly prove surface-follows-deformation.
- 1B-ii FINAL whole-branch review: READY (opus; 0 blocking; full physics->rebuild->DC->draw chain consistent; no 1A/1B-i regression; oracles discriminating). STAGE 1B-ii COMPLETE (commits f27c4ee..9ff737b).
- *** STAGE 1 COMPLETE *** (1A static DC recon + 1B-i physics solver + 1B-ii deformation coupling). Next = Stage 2 (cutting).
- Track-forward: user must let Unity import + commit generated .meta (esp. Physics.compute); cosmetic stale header comments in DualContouring/Recon.compute/ReconCommon.
- STAGE 1 ALIGNMENT AUDIT (5 dims + skeptic): 0 critical, 2 important (both FIXED ccbe507), ~57 confirm-ok. Fixes: QEF oracles now dispatch RebuildIsectWorld (test regression from 1B-ii); D6 documented (per-frame h-adapt) + DeviationLedger.cs created. Stage 1 fully aligned to paper+design (deviations D1-D6 only).
- GPU RUN caught 3 red (28 green). ROOT CAUSES: (1) CRITICAL integrator sign bug — ComputeSlope used (Fext-Fint)/m but F_int=Eq11/Eq14 ACTUAL restoring force, so RHS must be (Fext+Fint)/m; minus made springs anti-restoring -> oscillator drift + gravity explosion. FIXED + D3 ledger reversed (orig v4 D3 was a misjudgment from gap C37). (2) bending test (b) asserted wrong sign (fj.y>0) — kernel correct (fj.y<0 restoring, j moves away from k@+y); FIXED test. Free-fall unaffected (Fint=0). StructuralForce oracle untouched (tests fs not RHS).
- SENIOR AUDIT (opus): math+architecture CLEAN (0 crit/imp); re-derived all 4 force/damping terms under corrected +F_int = restoring+dissipative; QEF/RK45/Hermite/winding exact; no 2nd sign bug; no undeclared deviation.
- JITTER DIAGNOSIS (opus): NOT an algorithm bug. (1) no visible sag = paper-stiff ks=7.5e4 -> sub-pixel sag + DemoSceneSetup gave no ks knob; (2) jitter = explicit RK45 near stability edge (omega~950, h_stab~3e-3) + per-frame D6 h-controller hunting past the edge + near-zero damping (zeta~0.0017). Double-manager RULED OUT (scene has only DemoSceneSetup; no persistent ReconGridManager).
- FIXES: (A) MassSpringSolver hMax=5e-4 cap on substep h (explicit-RK stability safeguard for per-frame D6; stable at any ks incl paper 7.5e4). (B) DemoSceneSetup exposes+forwards soft demo physics (ks=800,csDamp=20,kb=200,cb=5,gravity,pin) -> visible clean sag. Solver/test paper defaults unchanged.
