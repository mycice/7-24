# XPBD Combination — Round 3 (audit-hardened rev. B)

> **Rev. B (2026-07-05) — applied the Cycle-2 fresh-eyes design re-audit (workflow wtn1t8s1x, 5 angles +
> adversarial verify + synthesis; verdict "needs-rework": 6 blockers, 12 majors, 9 minors, 2 nits — all
> in the never-re-audited rev-A additions and their cross-cutting interactions).** Load-bearing rev-B
> changes, tagged **[audit rev-B]** in place: **(B1)** §9 RESTRUCTURED — the buffer rebind is now step
> (4), FIRST after the standalone steps (the old order made steps 4–5 gates unexecutable); **(B2)** tet
> CUT/TEAR lifecycle (`d_tetActive` + `k_DeactivateTets`, §4 [Strain]) — without it tets glue the cut
> halves; **(B3)** the SNH pair pinned to the reference's REST-ZERO pair `tr(FᵀF)−3` + `det(F)−1` (the
> prior mixed pair inflates every tet); **(B4)** per-corner OWNERSHIP MATRIX (§5 item 4) — jaw contact vs
> slab no longer fight over captured corners; **(B5)** tear-exemption RELEASE-DECAY (§6) — release no
> longer shreds the tent; **(B6)** frame 1:11 DESCOPED (multi-tool infra does not exist; §11 item 7).
> Majors: sever-propagation via `k_RefreshLiveness` + persisted owner-slot maps (M1), the
> `xpbd_begin_frame/xpbd_substep` API split (M2), gate-driven in-loop haptic detect + pose-upload fix
> (M3), per-corner contact domain replacing the unimplementable barycentric bullet (M4), contact/slab λ
> storage + `Fn` timing + partial anchor advance (M5), core-pin lifecycle + the ONE w-rule (M6), 6-tet
> parity table + build asserts (M7), corrected full `xpbd_bind` signature (M8), the cs PERPENDICULAR
> post-pass fully discretized (M9), `cb` in-constraint γ_B is the COMPLETE port — no post-pass (M10),
> 96-grid N/tear re-derivation (M11), §11/§12 stale-text rewrite (M12); minors m1–m9 + nits n1–n2 in
> place. Steps 1–3 remain CLEAN (zero physics defects); M1/M2 are module API/bookkeeping refactors.

> Revision of the 2026-07-05 grasper-contact optimization plan, driven by the paper-demo VIDEO
> (youtube.com/watch?v=eK_QKEXwyag) the user gave 4 keyframes for. Ground truth re-verified against
> HEAD e82a6a3 (physics.cu, cut.cu, haptics.cu) this session — every code line cited below was read.
> This section is BOTH the answer to the user's question ("can I combine my mass-spring model with
> XPBD?") AND the implementer's plan. It SUPERSEDES the within-RK45 framing of Rounds 1–2 for Issue A;
> Issues C and D fold in unchanged except where noted. The paper's mass-spring MODEL (topology, rest
> lengths, severing, recon, anchored BC) is kept verbatim; only the INTEGRATOR changes.
>
> **Rev. A (2026-07-05) — applied the Phase-1 design audit (workflow w7lgj8kvp, 5 agents + synthesis,
> verdict "needs-major-rework").** Closed 4 blockers and 8 majors before any further code. Cross-checked
> against the user's production solver `Simulation/Assets/SurgicalSim/{Shaders/XPBDSolver.compute,
> Physics/XPBDSolverGPU.cs, Grasping/GripperTool.cs}`, github.com/FantasyVR/neohookean_XPBD, and the
> Macklin literature. Load-bearing changes vs the original round-3 draft, each tagged **[audit]** at its
> section: **(B1)** the volume element is now a per-TET Stable-Neo-Hookean pair as the BASELINE (§4
> [Strain]) — a scalar per-cell `C=V−V₀` is proven NOT to stop filament collapse and is retired; **(B2)**
> grasp is now a HARD core-pin + DYNAMIC transition ring (§5), matching `GripperTool.cs`, not a uniform
> compliant attachment; **(B3)** a NEW migration step (rev-A "§9-(6.5)"; now §9-(4) after the rev-B
> restructure) specifies the buffer-rebind so `xpbd_step`
> deforms the SHARED `recon_corner_pos()`/`physics_vel()`, not the module's private buffers; **(B4)**
> §6.5 now audits BOTH cut paths (tick-phase CCD + rod-phase Möller-Trumbore quad). Majors: an explicit
> **omega / relaxation** section (§3.5), a **coupling-pass** scheme (§3.5, K=2 per XPBDSolverGPU.cs), a
> split **N_cut vs N_elastic** derivation (§9-(5)/§11), the bending anisotropy disclosure (§4 [Bending]),
> a **1–2-ring dilated** tear exemption (§6), the Coulomb `Fn`-from-λ definition (§5), and the step-4
> **moving-tool** latency gate (§9-(4)). Minors folded into §3/§4/§12. The SPINE is unchanged and
> validated: full-XPBD (not hybrid), `α̃=(1/ks)/h²`=exact-spring, small-steps, cut-survives-topologically.

---

## 0. The acceptance bar (the paper's own demo video)

Four keyframes the plan must land:

- **0:05** — two laparoscopic graspers PINCH-AND-HOLD a fold of tissue between their jaws: a real
  two-jaw grip plus a visible compression crease.
- **0:43 / 1:00** — a single grasper grabs a point and PULLS the tissue into a long, smooth CONICAL
  TENT: large-strain elastic stretch many times the tool size, perfectly smooth, and CRITICALLY the
  tissue does **not tear at the grip** even at extreme stretch.
- **1:11** — two graspers pull in opposite directions → bidirectional tents, a third probe resting
  below. **[audit rev-B — DESCOPED from this migration: needs ≥3 simultaneous tools, and the native
  pipeline is single-tool global state throughout (§5 MULTI-TOOL REALITY CHECK, §11 item 7). The bar for
  THIS migration is the single-tool frames + Issues A/B/C; 1:11 is a follow-on work item.]**

The soft body is a bunny-shaped tissue phantom. The deformation is smooth (no blocky voxel artifacts),
large-strain, stable, and grasp-driven (the grasped point rides with the tool). That is the bar.

---

## 1. RECOMMENDATION — full-XPBD, not a hybrid

**YES, you combine your mass-spring model with XPBD — and the correct combination is to solve your
existing springs IMPLICITLY as XPBD constraints, replacing the RK45 integrator entirely. NOT a hybrid
(RK45 base + a bolted-on XPBD contact pass).**

The key conceptual point that answers the user's question directly: **XPBD of a mass-spring system IS
still your mass-spring model.** Macklin/Müller/Chentanez (XPBD, MIG 2016) prove that a distance
constraint with compliance `α̃ = (1/k)/Δt²` reproduces a Hookean spring of stiffness `k` *exactly* — but
unconditionally stable at any `k` and any timestep. So "combining mass-spring with XPBD" does not mean
grafting a foreign solver onto your model; it means integrating the SAME springs (same 6-neighbour
topology `d_nbrIdx`, same rest lengths `d_restNbr`, same bending pairs `d_bendPairs`/θ₀, same severing
`nbrIdx=-1`/`active=0`/`alive=0`) with an unconditionally-stable implicit projection instead of the
explicit Dormand–Prince RK45. The particles, the edges, the rest lengths, the cut, and the recon all
stay. Only `physics_step`'s inner integration loop is swapped.

### Why full-XPBD and not hybrid — each reason tied to the acceptance bar and the actual code

**(i) The tent's fragility is IN the elastic integration — a contact-only post-pass cannot fix it.**
The current adaptive-h controller (physics.cu:584, `hn = g_h*0.9*(EPS_TOL/delta)^0.2`, clamped
`[0.2h,5h]` then `[1e-6, min(hMax,dt)]`) shrinks `g_h` as the spring error `delta` grows. At a cone
stretched many-times-tool-size with `ks=7.5e4`, the stiff near-grip region drives `delta` up, `g_h`
collapses toward the `1e-6` floor, and the `while (t < dt-1e-9f && guard++ < maxSubsteps)` loop
(physics.cu:543) hits its `maxSubsteps` guard **without reaching t=dt** — the frame is left
under-integrated → jitter/blowup/slowdown, exactly where the video needs a clean static cone. A hybrid
keeps this loop, so it keeps this failure. XPBD (small-steps substepping, Macklin 2019) is
unconditionally stable at any stiffness and strain — the tent stops shimmering.

**(ii) The "springs erase the dent/grip" tug-of-war is architectural, not tunable.** `k_AccStructural`
(physics.cu:173) recomputes `fs = -ks*(len-L0)*dir - cs*(vi-vj)` every RK stage from the current
`cornerPos` and pulls a displaced/grasped corner back toward the rest lattice UNOPPOSED. Any position
write that lives OUTSIDE the integrator (the contact dent from `k_ApplyContactDisp` at haptics.cu:345;
a grip attachment) is fought by the springs INSIDE it. A single grasp point pulled into a 10× tent is
not a floor a once-per-frame projection can re-impose — it is a *sustained attachment* competing
against `ks` every substep. Full-XPBD dissolves the tug-of-war: the attachment AND the elastic edges
are BOTH constraints in the SAME solver, reconciled together (across the substep loop's under-relaxed
coloured sweeps + K coupling passes, §3.5 — not literally one sweep) instead of fighting across two
solvers. This is the "stiff contact + soft tissue in one solver" property (compliance `α̃`), and it is
what makes A-Part 1's held dent fall out for free (§7 below).

**(iii) Two stiffness models is strictly MORE coupling risk for a WORSE result.** The hybrid keeps `ks`
in explicit force units (CFL-limited, fragile) AND a separate contact compliance (unconditionally
stable) that you must keep mutually consistent every frame, with the projection re-winning the
tug-of-war each frame — which is precisely the render-reorder + double-`Finalize` gymnastics Round 2's
within-RK45 plan needed (§7). Full-XPBD has ONE stiffness model (compliance) for everything. The
hybrid's only merit is a smaller code delta, which the user has explicitly declared a non-goal.

**Verdict:** full-XPBD. It fixes all four video-blocking failure modes at their root
(RK45-tent-fragility, spring-erases-dent, fake-grasp, tear-at-grip). The hybrid leaves the first two
structurally in place.

**One caveat the user must accept as a SEPARATE lever (do not oversell the swap):** XPBD makes the tent
STABLE, SMOOTH-CURVATURE, and NON-TEARING, but it does NOT make the cone geometrically ROUND. Silhouette
roundness is grid-bound: at `targetLongAxisVoxels=48` only ~4–6 corners span the cone cross-section, so
the recon (per-voxel QEF dual contouring, dc_recon) shades a hexagonal-ish prism, not a circular cone.
A truly round cone needs `targetLongAxisVoxels`→96 (Issue B-2). Full-XPBD's unconditional stability is
exactly what makes 96 affordable where RK45's collapsing adaptive-h could not. Sell it as: **integrator
swap = stable / smooth-curvature / no-tear tent; finer grid = actually-round silhouette.** Both, not one.

---

## 2. Verified ground truth (HEAD e82a6a3) — the exact thing being replaced

- **RK45 substep loop:** `physics_step` at physics.cu:529–594. Inner loop 543–573:
  `while (t<dt-1e-9f && guard++ < maxSubsteps)` → `k_SeedTrial` (547) → 7× stage loop
  `{k_BuildTrial(551); k_ClearForces(552); k_AccStructural(553); k_AccBending(555); k_ComputeSlope(557)}`
  → `k_ComputeError(560)` → `k_IntegrateY5(561)` → `t += hStep` → per-substep `cut_ribbon_tick()` at
  568–572. Adaptive-h at 576–586.
- **Structural spring:** `k_AccStructural` physics.cu:149–178. `fs = (-ks*(len-L0))*dir - cs*(vi-vj)`
  (173), `L0 = length3(restNbr[6p+s])` (167). Sever gates: `if (nbrIdx[6p+s] < 0) continue;` (164) and
  `if (active[n]==0) continue;` (165).
- **Bending:** `k_AccBending` physics.cu:181–217, gated `if (bp.alive==0) return;` (189).
- **Haptic reaction is LIVE in the slope:** `k_ComputeSlope` reads `d_hapticForce` and forms
  `k.dv = (extForce[p] + hapticForce[p] + Fint)/mass[p] - alpha*yTrialVel[p]` (physics.cu:277).
  Pinned/inactive frozen at 263.
- **Commit:** `k_IntegrateY5` writes `cornerPos[p] += H*dx; vel[p] += H*dv` (307–308), pinned/inactive
  frozen with `vel=0` (289–292). `cornerPos = recon_corner_pos()` is the single shared buffer
  physics/cut/haptics/recon all read/write (physics.cu:534).
- **Contact dent = position write the springs erase:** `k_ScatterContact`
  `contactDisp[best] += nOut*(-contactBeta*dentDepth/3)`, `dentDepth = fminf(depth, dentDepthMax)`
  (haptics.cu:325–329); `k_ApplyContactDisp` `cornerPos[c] += clamp(d, dispMax)` then `vel[c] *= 0.15`
  (345–350). Next frame `k_AccStructural` pulls it back.
- **Tear law:** `k_TearOverstretch` cut.cu:354–399. Severs any active–active grid edge with
  `len2 > tearLen2 = (tau*voxelL)²`, tau=4.0, gated on `g_toolSet` (host). Already skips severed
  (371) and inactive (372) edges.
- **Grasp is FAKE:** `haptics_set_grasp` only sets `g_graspMu = 1 + 5*grasp01` (haptics.cu:709) — a
  friction-cone multiplier. No attachment, no jaw compression.
- **Contact detection band:** `haptics_detect` returns `wk < g_toolRadius + margin` (haptics.cu:449);
  `margin` read only inside `if(!g_hasProxy)` as `max(captureMargin, |physDisp|)` — first-touch only.

---

## 3. The exact architecture — mass-spring MODEL → XPBD constraints

Replace the body of `physics_step` (physics.cu:543–586) with a **fixed-N-substep small-steps loop**
(Macklin, "Small Steps in Physics Simulation," SCA 2019: many substeps × ONE constraint iteration each
— deterministic, no adaptive-h). Repurpose `PhysicsInitDesc.maxSubsteps` as N so the C# `StructLayout`
ABI stays stable; `hInit`/`hMax` lose meaning and are documented as superseded (§9).

**Per substep (h = dt/N):**

1. **Predict.** `x_prev = cornerPos;` then apply gravity + the SS2.3 haptic reaction, then advect:
   ```
   vel  += h*(extForce + hapticForce)/mass;          // gravity + felt reaction, out of k.dv
   cornerPos += h*vel;
   ```
   Pinned (`pinned[p]!=0`) and inactive (`active[p]==0`) corners are frozen — reuse the exact guard from
   `k_ComputeSlope` (physics.cu:263) / `k_IntegrateY5` (289). (The Rayleigh drag is applied ONCE per
   substep in the post-solve velocity write, step 4 — NOT here — so it is not double-counted.)
2. **Reset multipliers.** `lambda = 0` for every constraint (XPBD requires a fresh λ per substep).
3. **Project constraints in-place on `cornerPos`** (§3.5 defines the sweep: under-relaxed coloured GS,
   iters + K coupling passes; §4 defines each constraint).
4. **Velocity update + damping.** `vel = (cornerPos - x_prev)/h`, then the per-substep implicit Rayleigh
   + global drag as a single scale on the written velocity:
   `vel *= velScale`, `velScale = (1 − damping)/(1 + α·h)`.
   **[audit] The Rayleigh factor is the IMPLICIT `1/(1+α·h)` (equivalently `exp(-α·h)`), NEVER the
   explicit `(1−α·h)`** — the explicit form flips sign and INJECTS energy when `α·h > 1`. `α` is re-derived
   under N substeps so the per-frame decay matches the RK45 golden: target `exp(−α·dt)` over the frame ⇒
   per-substep `exp(−α·dt/N)`. The Kelvin–Voigt `cs` shear damping is separate (§4 [Structural]): the
   along-constraint γ (implemented, steps 1–3) stays; the PERPENDICULAR post-pass kernel — specified in §4,
   **[audit rev-B]** run AFTER `vel=(x−x_prev)/h` and BEFORE the `velScale` multiply, over the existing
   edge COLOURS (no atomics) — is added at the strain step. `cb` bending damping needs NO post-pass (§4
   [Bending]). This is the PBD implicit-velocity identity (Müller PBD 2007): a satisfied constraint exerts
   NO restoring force, which is the mechanism that ends the spring-vs-dent/grip tug-of-war.

**cornerPos remains the single committed state** (shared with recon), written exactly where
`k_IntegrateY5` wrote it. **[audit] The already-built step-1 `xpbd_solver.cu` module owns PRIVATE
`d_pos`/`d_prevPos`/`d_vel` — for the standalone self-test ONLY. Before the solver goes live in the DLL
it MUST be rebound to operate IN-PLACE on the shared `recon_corner_pos()` and write physics.cu's
`physics_vel()`; that rebind is a first-class migration step (§9-(4), rev-B order), not an afterthought.** Until it
happens, cut/recon/haptics/particleRot read the OLD `cornerPos` while XPBD deforms a private buffer —
silent divergence, the single biggest correctness risk in the migration. `particleRot` reconstruction
(`physics_compute_rotations`, `k_ComputeParticleRot`, Stage 3) is integrator-agnostic — it reads
committed `cornerPos`+`nbrIdx` after the solve, UNCHANGED. dc_recon is UNCHANGED (still re-meshes
`cornerPos`). cut.cu is UNCHANGED except the grasped-edge tear exemption (§6).

### GPU parallelism — the one genuinely new piece of infrastructure

The current force kernels use a fixed-point atomic accumulator (`AtomicAddForce`) so per-particle gather
is race-free by construction. XPBD projection WRITES shared corners, so a naive parallel edge kernel
races (two edges writing one corner). Two standard options:

- **Graph-colour the 6-neighbour axis-aligned lattice** (recommended): the regular grid 2-colours
  structurally (checkerboard); the 6 neighbour slots partition into a few race-free colour sets (an 8³
  lattice needs 6 colours). One kernel launch per colour, per substep, no atomics.
- **Averaged Jacobi with under-relaxation** (Macklin, Unified Particle Physics, SIGGRAPH 2014):
  accumulate per-corner position deltas atomically (the pattern `AtomicAddForce` already implements),
  divide by constraint count, under-relax. Simpler to drop onto the existing atomic infrastructure;
  slightly slower convergence, needs the relaxation tuned.

Recommend coloured batches. This is the main cost of going full-XPBD — mechanical, but budget for it.
Perf is a non-constraint (RTX 3090) but the coloured-kernel count multiplies by N, so profile once that
the fixed budget holds 60 fps.

### 3.5 Convergence: under-relaxation (ω) + coupling passes — NOT a single sweep [audit]

**Do NOT assume one coloured Gauss–Seidel sweep converges the stiff coupled network.** The original draft
repeatedly equated "graph-coloured 6-neighbour batches" with "true Gauss–Seidel that converges fast in
one iteration." That is true for a SINGLE isolated constraint but FALSE for the coupled elastic +
bending + volume + contact + grasp system at 1 iteration/substep. The user's production solver does not
trust it either: it under-relaxes every projection at **ω = 0.2** (`XPBDSolverGPU.cs:450`) AND runs
**`ToolContactCouplingPasses = 2`** alternating `SolveInternalOnce()` + `SolveToolContacts()`
(`XPBDSolverGPU.cs:508-513`). Convergence here relies on THREE nested knobs, not one sweep:
`N substeps × iters × K coupling passes`, each under-relaxed by ω.

- **ω (SOR under-relaxation).** Applied CONSISTENTLY to λ AND to the position write, so the compliant
  equilibrium `ΔL = mg/ks` is ω-INVARIANT — ω changes only the convergence RATE, not the effective
  stiffness (this is the reviewed `xpbd-math` fix; the oracle pins it). **Fixed-point-invariance ≠
  per-sweep STABILITY:** a colour applying its FULL correction (ω=1) ignores shared particles that later
  colours move in the same sweep, so ω=1 can OVERSHOOT on stiff edges and on the near-hard
  contact/grasp constraints of step 8 (reference contact compliance `1e-7` is exactly the regime where
  over-relaxation is worst). **Default ω = 0.2–0.5, matching the reference.** ω=1 is permitted for the
  step-1 structural-only self-test (where it is provably fine and the analytic `ΔL=mg/ks` still holds),
  but before committing ω=1 for the FULL constraint set a stability test at `ks=7.5e4` with
  structural+bending+contact+grasp is REQUIRED; escalate under-relaxation specifically for the
  hard-compliance contact/grasp passes.
- **K coupling passes (step 8).** After the elastic (structural + bending + volume) colour sweep, run
  **K = 2** coupling passes of `{internal-once, contact/grasp}` per substep (mirrors
  `XPBDSolverGPU.cs:508-513`); the LAST pass ends on contact/grasp, satisfying §4's "contact projected
  LAST" by construction. **[audit rev-B — λ semantics across passes:** all multipliers (elastic, contact,
  slab) are reset ONCE per substep (§3 step 2) and ACCUMULATE across iters AND all K passes within the
  substep — they are one XPBD solve, not K independent ones.] Without coupling passes, the held-dent
  depth (§7) and the tent-tip position (§5) become ORDERING/ITERATION-dependent — a blend that shimmers —
  because one coloured sweep under-couples the local contact/grasp against the bulk elastic+volume. A
  step-7/8 VERIFY must show that increasing K removes any residual grip-vs-volume fight and that measured
  dent depth == tool penetration within tolerance.
- **GS anisotropy.** 1-iteration fixed-colour GS is the configuration MOST prone to axis-aligned
  faceting (threatens the smooth tent, Issue B). ω<1 + K coupling passes + the bending/volume network are
  the mitigations; if faceting persists, add a 2nd iteration or alternate the colour order per substep.

---

## 4. Constraints (the model, expressed as XPBD)

**[Structural distance] — the 6-neighbour springs, verbatim topology.** For each active–active edge
(skip when `nbrIdx[6p+s] < 0` OR `active[n]==0` — the IDENTICAL severing test `k_AccStructural`
already uses at physics.cu:164–165):
```
C          = |xi - xj| - L0          // L0 = length3(restNbr[6p+s]) — the SAME rest length, unchanged
grad       = (xi - xj)/|xi - xj|
w_i        = 1/mass[i]  (0 if pinned or inactive)   // anchored BC and severed debris get w=0
α̃         = (1/ks)/h²                              // reproduces ks=7.5e4 EXACTLY (XPBD MIG 2016)
Δλ         = (-C - α̃*λ - γ*gradDx) / ((1+γ)*(w_i + w_j) + α̃)   // damped; γ=0 ⇒ undamped
xi += w_i*(ω*Δλ)*grad ;  xj -= w_j*(ω*Δλ)*grad ;  λ += ω*Δλ    // ω applied to λ AND position (§3.5)
```
**[audit] Kelvin–Voigt damping — pin the mapping so the factor-`ks` bug cannot re-seed.** The paper's
`cs*(vi-vj)` structural dashpot ports to the Macklin XPBD damped-constraint term:
```
β  = cs   DIRECTLY   — the Kelvin–Voigt dashpot STIFFNESS, in force / relative-velocity units.
                       NOT cs/ks. NOT a compliance. NOT an inverse of anything. (matches the
                       xpbd_solver.h betaS/betaB field docs: "set betaS = cs and betaB = cb DIRECTLY")
γ  = α̃ * β * h  =  (cs/ks)/h            // the damped-constraint coefficient
gradDx = grad · ((xi−x_prev_i) − (xj−x_prev_j))   // relative velocity ALONG the constraint gradient
```
(A prior draft's "β from cs" wording produced a real factor-`ks` code bug, β=cs/ks, which made `cs`
inert. The code is now correct; this prose is pinned to match it. Same rule for bending: `β_B = cb`
directly.) **Two routes, staged deliberately:**
- **Along-constraint γ (implemented, steps 1–3).** The XPBD damped-constraint term above. It correctly
  reproduces the Kelvin–Voigt dashpot ALONG the constraint gradient and is verified by the step-3
  cs-isolation test (betaS removes ~99.9% of the along-edge oscillation envelope). This is the CURRENT
  implementation and is CORRECT for the along-edge component.
- **PERPENDICULAR velocity post-pass (target for the tent; lands at the strain step). [audit rev-B —
  fully specified; the bare "−cs*(vi−vj) in the velocity write" was a spec hole.]** The along-constraint γ
  damps ONLY the gradient-projected (along-edge) component, which the distance constraint already holds
  stiffly; it UNDER-DAMPS the perpendicular/shear/flutter modes `cs` was tuned to kill (the modes that
  make a large-strain tent SHIMMER). The paper's `cs` damps the FULL relative velocity (all 3 components,
  physics.cu:173). The port is a per-edge kernel with these EXACT semantics:
  - **Placement:** after `vel = (x − x_prev)/h`, BEFORE the `velScale` (Rayleigh/drag) multiply.
  - **Component:** because γ STAYS as the along-edge dashpot (it is verified, steps 1–3), the post-pass
    damps ONLY the PERPENDICULAR component — damping the full vector would DOUBLE-damp the along-edge
    part and break the step-3 cs-isolation golden:
    `vrel⊥ = (vi−vj) − dot(vi−vj, n)·n ;  vi += −h·w_i·cs·vrel⊥ ;  vj −= −h·w_j·cs·vrel⊥`
    (the `h·w·cs` factor is the explicit one-step dashpot impulse per unit mass).
  - **Race strategy:** dispatch over the EXISTING structural edge colours (one launch per colour, no
    atomics) — within a colour no two edges share a particle, so the `vi`/`vj` writes are race-free.
  - **Stability:** this is an EXPLICIT dashpot — stable only while `cs·h·w·deg < 2` (deg ≤ 6). At the
    paper's cs=0.92, m=1, h≈1e-3 the margin is ~5.5e-3 ≪ 2 (safe by orders of magnitude); assert the
    bound when N or mass changes, or switch to the implicit per-edge form `vrel⊥ /= (1 + cs·h·(wi+wj))`.
  The step-1–3 γ-only implementation is an ACCEPTED INTERIM — correct for the small-strain settle tests,
  INSUFFICIENT for the tent; the post-pass is REQUIRED at the strain step (§9), not optional.

Setting stretch compliance NONZERO (`α̃ > 0`, not the α=0 hard limit) is MANDATORY: it lets edges extend
elastically into the cone instead of locking into flat panels, and it keeps grasped edges elastic instead
of over-straining into a tear (§6).

**[Bending] — the 15-pair network, verbatim topology.** For each pair in `d_bendPairs` (i=center,
j,k=arms), dropped when `bp.alive==0` (the SAME gate as physics.cu:189): an angle constraint `C = θ - θ₀`
using the same geometry `k_AccBending` already computes (`Nij`, `Nik`, `θ = acos(dot)`, physics.cu:194–202),
compliance `α_b = (1/kb)/h²`, damping `β_B = cb` DIRECTLY (§4 [Structural] rule). **[audit rev-B] The
in-constraint `γ_B = α̃_b·cb·h` damped term IS the COMPLETE port of the paper's `cb·θ̇` (physics.cu:209):
the paper's bending dashpot acts along the Eq14 gradient subspace BY CONSTRUCTION, which is exactly what
the along-gradient γ damps. Unlike `cs`, NO velocity post-pass is needed for bending — a prior draft's
"moves to the velocity post-pass" wording was WRONG and would have made an implementer rip out verified
damping or double-damp. (Selftest J proves γ_B active: a released hinge's KE envelope collapses at
βB=cb·factor, rings at βB=0.)** Keep the `kb=2e4` vs `ks=7.5e4` ratio → bending `α̃` ~3–4×
the stretch `α̃`. **Do NOT drop the bending network** — it decouples bending from stretch and is the
project's existing anti-facet asset; distance-only constraints kink the tent into flat panels (the
opposite of the video).

**[audit] This is a SOFTENED, geometry-scaled bending constraint, NOT the clean angle-energy gradient —
disclose it.** The implemented gradients are the Eq14 force-DIRECTION vectors `g_j = (n_v − c·n_u)`,
`g_k = (n_u − c·n_v)` with `g_i = −(g_j+g_k)` (this is DELIBERATE and REQUIRED: the true `∇θ` divides arm
j by `L_u=|xi−xj|` and arm k by `L_v=|xi−xk|`, i.e. carries `1/(sinθ·|arm|)`; keeping that made bending
~10× too stiff and gave a 50% cantilever-droop mismatch vs the RK45 golden — the un-normalized Eq14 form
matches to 0.4%; the oracle pins it). Two consequences that MUST be stated and gated:
- **Anisotropy at large strain.** Dropping `1/L_u`, `1/L_v` UNEQUALLY (neither arm normalized) makes the
  effective bending stiffness arm-length-asymmetry-dependent. On the near-rest lattice the arms are
  near-equal so it is benign; at a stretched/grasped tent the arms become very unequal, so bending
  becomes anisotropic and direction-dependent EXACTLY in the target regime.
- **Effective kb only ≈ kb when `w_g ≪ α̃_b`.** [audit rev-B — numbers corrected; a prior draft printed
  `α̃_b~7.8e4`, off by 1000×.] `α̃_b = 1/(kb·h²) ≈ 78` at kb=2e4, h≈8e-4; with per-corner mass 1.0,
  `w_g = Σ w·|g|² ~ O(1)` (|g| ≤ sinθ ≤ 1) — so `w_g ≪ α̃_b` roughly HOLDS at the shipped config and the
  effective stiffness IS ≈ kb near rest. The residual risk at the tent is therefore the ANISOTROPY above
  (arm-length asymmetry), not a wholesale kb rescale — which is why the gate below is geometric
  (unequal-arm golden-match), not a stiffness sweep.

**Gate:** step-2 VERIFY must include a LARGE-STRAIN bending test at tent geometry (not just
clamped-cantilever droop). If the anisotropy reads as a directional/faceted tent, either restore per-arm
`1/L_u`, `1/L_v` scaling (still non-singular — `|g|` keeps the `sinθ` factor) or document the anisotropy
as an explicit tuning axis.

**[Strain — per-TET Stable Neo-Hookean] — THE ONE ADDITION to the model. [audit: BLOCKER rewrite].**

The original draft named a scalar per-voxel-cell volume constraint `C = V − V₀` as the "cheap baseline."
That is RETIRED for two hard reasons the audit confirmed against the code + the user's own production
solver:
1. **It is non-implementable on this model as written.** The paper model is a 6-neighbour corner
   LATTICE with NO tetrahedra and NO defined cell-volume primitive: there is no cell→corner incidence,
   no `V₀` rule, no gradient. An 8-corner hexahedron has no unique volume without a decomposition — which
   the draft never specified.
2. **Even if defined, a SCALAR volume constraint does NOT prevent the filament collapse it was
   introduced to fix.** A tent conserves TOTAL volume while thinning uniformly toward a filament. What
   resists collapse + faceting is the DEVIATORIC (shape) term `tr(FᵀF)−3`, which the draft
   de-prioritized ("upgrade only if too springy") — exactly backwards.

**Baseline (NOT an upgrade): the user's OWN production formulation — per-tet Stable Neo-Hookean**
(`Simulation/Assets/SurgicalSim/Shaders/XPBDSolver.compute:252-332`; Macklin/Müller MIG 2021), which is
also what FantasyVR implements. Specify it as first-class migration infrastructure:

- **Tet mesh over the lattice — [audit rev-B: fully pinned, was underspecified].** Decompose each active
  voxel cell (its 8 corners) into **6 tetrahedra** (the Kuhn/"diagonal" split; the 5-tet option is DROPPED
  — it requires a mirrored checkerboard parity the draft never specified). **Corner-index table** (cell
  corners numbered `c0..c7` = `(x,y,z)` bit order `c[i] = (x+ (i&1), y+((i>>1)&1), z+((i>>2)&1))`, main
  diagonal `c0–c7`): the 6 tets are `{c0,c1,c3,c7} {c0,c3,c2,c7} {c0,c2,c6,c7} {c0,c6,c4,c7} {c0,c4,c5,c7}
  {c0,c5,c1,c7}`. **Parity alternation (MANDATORY):** a uniform lattice-wide diagonal biases stiffness
  along that body diagonal — an anisotropy source. Alternate the main diagonal per cell by parity
  `(x+y+z)&1` (mirror the table's local indices on odd cells) so the bias cancels cell-to-cell. Emit a tet
  only when all 4 corners are `active` at build time. Build device arrays `TetIds[4·tetCount]`,
  `InvRestMatrix[tetCount]` (`Dm⁻¹`, `Dm = [x1−x0 | x2−x0 | x3−x0]` at rest), `RestVolume[tetCount]`
  **`V = |det(Dm)|/6` with a degenerate-tet guard (skip V < ε, per XPBDSolverGPU.cs:208-231) AND a
  build-time assert `det(Dm) > 0` after applying the table's ordering** — a mis-ordered fixed table yields
  negative V hence negative compliance, silent at rest. Plus `d_tetActive[tetCount]` (lifecycle below) and
  a tet greedy colouring (4-vertex mask, same 64-colour fail-loud rule; budget: 6 tets/cell sharing
  corners colours like the bending set — well under 64). This is the biggest single piece of NEW
  infrastructure after graph-colouring.
- **Deformation gradient** per tet each substep: `F = Ds · Dm⁻¹`, `Ds = [x1−x0 | x2−x0 | x3−x0]` from
  CURRENT positions.
- **(a) Deviatoric constraint — MANDATORY, this is what stops collapse/faceting:**
  `C_dev = tr(FᵀF) − 3`, compliance `α̃_dev = 1/(h²·μ·V)`. μ = shear modulus.
- **(b) Hydrostatic constraint — incompressibility crease:** `C_hyd = det(F) − 1`, compliance
  `α̃_hyd = 1/(h²·λ·V)`. λ = first Lamé.
- **[audit rev-B — the constraint PAIR is pinned as the reference-as-written pair; a prior draft mixed
  two incompatible conventions.]** The pair above is EXACTLY the user's production solver
  (`XPBDSolver.compute:219` deviatoric `tr(FᵀF)−3`, `:311` hydrostatic `det(F)−1`) — both REST-ZERO, so
  the rest state F=I satisfies both constraints exactly and the lattice is stress-free at rest by
  construction. The prior draft paired a rest-zero deviatoric with Macklin-2021's rest-OFFSET hydrostatic
  `det(F)−(1+μ/λ)` — that offset is only stress-free when paired with Macklin's UN-offset deviatoric
  `C_D=√(tr(FᵀF))` (which pulls toward det<1, the offset compensating); mixing conventions makes every
  tet inflate toward `det(F)=1+μ/λ` (+67% volume at ν=0.3) and corrupts the golden gate. If the
  rest-zero pair shows volume drift under large strain in practice, the DOCUMENTED alternative is the
  full Macklin pair `{√(tr(FᵀF)), det(F)−(1+μ/λ)}` — switch BOTH together, never mix.
- **μ, λ from a chosen E, ν:** `μ = E/(2(1+ν))`, `λ = Eν/((1+ν)(1−2ν))`. **Pin ν = 0.45** (soft-tissue
  near-incompressible, still numerically kind; λ/μ ≈ 9) as the starting point; E tuned against the video
  (order-of-magnitude anchor: the paper's ks=7.5e4 spring lattice). Final E/ν confirmation is a §11 user
  decision.

**[audit rev-B — TET LIFECYCLE × CUT (was a BLOCKER hole: nothing deactivated tets).]** Cuts and tears
sever lattice EDGES and bending pairs (`k_SeverLinks` cut.cu:405-433 writes `nbrIdx=-1` on both half-slots
+ `bendPairs.alive=0`; `k_TearOverstretch` same mechanism; a corner's `active[]` flips 0 only when ALL its
links are gone, cut.cu:451). A tet spanning the cut keeps all 4 corners active — its dev+hyd constraints
would PERMANENTLY GLUE the two halves together, breaking cutting (the paper's title feature) and the v5.1
tear/strand behaviour. Therefore:
- **Deactivation rule:** a tet DIES (`d_tetActive[t]=0`, permanent) when ANY of its 6 member edges (the
  4-choose-2 corner pairs restricted to lattice edges — for the Kuhn split, each tet has 3 axis edges, 2
  face diagonals, 1 body diagonal; only the AXIS edges exist in `nbrIdx`) is severed. Concretely: a tet
  dies when any of its member AXIS edges has `nbrIdx[6p+s] < 0`, checked against the same live flags the
  structural gate reads. (Face/body diagonals are not severable primitives — axis-edge severing is what
  the cut produces, and any cut surface through a cell severs axis edges of that cell.)
- **Where it runs:** a `k_DeactivateTets` pass appended to `cut_ribbon_tick()` immediately AFTER
  `k_SeverLinks`/`k_TearOverstretch` (one thread per live tet, reads the nbrIdx image, flips
  `d_tetActive`) — or equivalently folded into the per-substep liveness refresh (§9 rebind step). Also
  RUNTIME-skip any tet containing a corner with `active[c]==0` (w=0 is NOT sufficient — a w=0 corner in a
  live tet still glues its 3 live corners to frozen debris).
- **VERIFY (added to the strain-step gate):** re-run the step-0 known-cut regression WITH tets live and
  assert a measured separation gap opens (the halves must NOT stay glued); plus one tear-through-tet test
  (overstretch an edge past τ·L, assert the incident tets die and the tear opens).

This is the ONLY element not already in the paper's mass-spring model; it is what turns a distance-spring
lattice into a volumetric soft body that tents smoothly without collapsing to a filament. Justified by
the user's production solver + Macklin/Müller Stable Neo-Hookean (2021) + Müller Strain-Based Dynamics
(2015) + iMSTK's volumetric-tissue guidance. Step gates ("smooth non-collapsing cone" + the cut/tear
gates above + a REST-STRESS gate: a zero-gravity tet lattice holds `det(F)=1` with no drift) are met by
the DEVIATORIC term + the rest-zero pair; a scalar `C=V−V₀` alone would fail them.

**[Contact] — non-penetration, native (subsumes the spring-fought dent).** An inequality
`C = sd - r ≥ 0` (sd = signed distance to the tool surface, r = the winning primitive's radius),
projected when `sd < r`, killing only inward normal velocity (tangential slide preserved for friction).
Uses Issue D's `ClosestOnTool` (jaw-mesh ∪ shaft-capsule) and per-primitive `triRadius`. Because
`vel = (cornerPos - x_prev)/h` bakes the projected position into velocity, the held dent PERSISTS
automatically — **Round 2's A-Part 1 "held projection after the spring solve" + render-reorder +
double-`Finalize` is retired; the floor is just another constraint in the loop.** (§7.) **[audit] Four
requirements the draft's bare "hard-compliance α=0" version got wrong:**
- **Use a small NON-ZERO compliance** (`~1e-7`, scaled `/h²` like the elastic terms — matching the
  reference `ToolContactCompliance = 1e-7`), NOT the literal `α=0` hard limit. On a near-0 trigger (fast
  tool at first detection, Issue C) a hard `α=0` injects a large one-substep positional POP on deep
  first-touch; a small compliance + under-relaxation (§3.5) avoids it.
- **Constraint DOMAIN: per-corner over the capture band [audit rev-B — the prior "barycentric" bullet
  was unimplementable].** The constraint is `C = sd(cornerPos[c]) − r ≥ 0` evaluated PER ACTIVE CORNER
  for every corner within the tool's capture band (|sd| < r + skin) — each in-band corner gets its OWN
  constraint, projected along its own contact normal. That per-corner field over the band is what makes
  the dent smooth (Issue B-1) — NOT a barycentric scatter: the reference's barycentric distribution
  (`XPBDSolver.compute:1027-1039`) presupposes the surface-triangle vertices ARE solver particles, which
  does not hold here (our render surface is QEF dual-contour vertices, NOT degrees of freedom; the
  corners are). The inherited single-corner argmin winner-take-all (one corner takes the whole
  correction) remains RETIRED — the fix is the per-corner band domain above, not weights.
- **Project contact LAST each substep** (after elastic + volume), inside the K coupling passes (§3.5),
  so the non-penetration floor is the final word on the surface each substep.
- **Skip any captured/contacting corner whose `active[c]==0`** (a cut severed it — do not push debris).
- **GRASP OWNERSHIP EXEMPTION [audit rev-B — without it, jaw contact and the slab FIGHT over every
  captured corner].** While `d_grasped[c]==1`, corner `c` is EXEMPT from the OWNING tool's jaw contact
  constraints (the slab owns its normal, §5's ownership matrix); shaft contact and OTHER tools' contact
  still apply. Core-pinned corners (`d_graspCore[c]==1`) are kinematic — skip ALL contact on them.
- **[audit] Dent/hapticForce double-count.** §7 retires `k_ApplyContactDisp`'s `cornerPos` write +
  `vel*=0.15` but KEEPS the `hapticForce` scatter (now in predict). The new contact CONSTRAINT (kills
  inward vel) and the kept `hapticForce` reaction (pushes the same region) now BOTH act on the contact
  patch — a step-5/8 gate must confirm they do not double-count (dent depth stays == tool penetration).

**[Grasp] — two-jaw pinch, native (subsumes `g_graspMu`).** Per-grasped-corner attachment to the jaw
frame + a slab-compression constraint, both in the SAME GS loop, with a Coulomb stick/slip cap (§5).
Matches frame 0:05 (pinch-and-hold + crease) and the video's slip-out.

---

## 5. Grasp + stretch + tent mechanism (frames 0:05 / 0:43 / 1:00; 1:11 descoped — see MULTI-TOOL below)

The tent and the two-jaw grip are the same family of constraint. **[audit] The smooth cone is an
EMERGENT elastic result around a PINNED CORE, not a property of a uniformly-compliant grip.** The user's
production reference (`Simulation/Assets/SurgicalSim/Grasping/GripperTool.cs`,
`CaptureParticles`/`UpdateGraspedParticles`) does NOT uniformly soft-attach the grasped set: it HARD-PINS
a small CORE (`gaussianHardCoreRadius = 0.55`, `invMass = 0`, writing
`Positions = PrevPositions = target` every frame) and leaves a surrounding DYNAMIC TRANSITION RING FULLY
free, so the elastic + bending + volume solver forms the smooth cone around the pinned core. A uniform
per-corner compliant attachment with NO core/ring split either LAGS the tool (too soft → the tent tip
trails the jaw) or is a plain pin with no emergent smoothing (too stiff). So this design adopts the
reference's structure, and corrects the earlier "adapts GripperTool to a uniform compliant attachment"
claim, which mischaracterized the reference.

**Single-grasper tent (0:43 / 1:00).** On grasp-capture, tag the captured corners and split them:
- **CORE (hard pin). [audit rev-B — semantics + lifecycle pinned.]** The core is selected by the
  reference's NORMALIZED-ELLIPSE rule, not a world radius: `gaussianHardCoreRadius = 0.55` is a FRACTION
  of the captured set's u/v extents in the jaw plane (`GripperTool.cs:507-532`) — corners whose
  normalized in-plane coordinate lies inside that ellipse fraction are core. (A world-radius reading
  fails on the 48-voxel lattice: core = empty → grasp does nothing, or core = everything → no ring, no
  cone.) **Mandatory fallback:** if the ellipse test selects NO corner, promote the single nearest
  captured corner to core (`GripperTool.cs:550-557`) — the grasp must never be core-less. **Pin
  lifecycle:** on capture, STORE each core corner's original `invMass` and its jaw-local offset; each
  substep the pin kernel writes `invMass=0`, `cornerPos = x_prev = jawPose·storedLocalOffset` AND
  `vel = 0` (the implemented velocity kernel skips w=0 corners, so without the explicit `vel=0` a stale
  pre-capture velocity survives the whole grasp, feeds haptics, and reactivates as a SNAP on release).
  On release, restore `invMass` from the stored original ONLY where `active[c]!=0`, clear
  `d_graspCore`/`d_grasped`, and damp the restored corners' velocity (×0.1, per GripperTool.cs:600-601)
  so the spring-back starts clean. **The `w`-rule everywhere:** `w_i = 0` iff `pinned[i] OR
  active[i]==0 OR graspCore[i]` — every constraint kernel uses this ONE rule (else the solve can move
  "pinned" core corners).
- **TRANSITION RING (fully dynamic).** All OTHER captured/neighbouring corners stay dynamic with their
  normal `invMass`; they are NOT attached. The nonzero-compliance structural + bending + Stable-Neo-
  Hookean volume constraints (§4) drag them into the long smooth cone (many-×L) emergently, reconciled
  each substep by the under-relaxed coloured sweep + K coupling passes (§3.5).

As the tool pulls away, the pinned core rides with it and the dynamic ring tents smoothly around it. The
third resting probe is just a non-penetration contact (§4 [Contact]). **If a genuinely uniform compliant
grip (no hard core) is later preferred for slip-out realism, it MUST be gated by a test proving it
reproduces a NON-LAGGING smooth cone — do not assume it.**

**[audit rev-B — MULTI-TOOL REALITY CHECK (was a blocker: frame 1:11 claimed "no extra machinery").]**
Frame 1:11 (two graspers pulling opposite + a third resting probe) needs **≥3 simultaneous tools**, but
the ENTIRE native pipeline is single-tool global state: one capsule/proxy/anchor (haptics.cu:107-121),
one `g_toolSet` (cut.cu:87), no tool index in the plugin API (plugin_api.cpp), and the grasp arrays above
have no owner-tool dimension. Supporting 1:11 requires real infrastructure: a tool index through
`LCS_SetTool`/`LCS_SetPhysicalTool`, per-tool proxy latch + `JawSlabWorld` + capture sets, an `ownerTool`
id in `d_grasped`/`d_graspAnchor`, and a per-tool contact query. **DECISION (descope): the acceptance bar
for THIS migration is the SINGLE-TOOL frames — 0:05 (one grasper's two jaws: satisfiable single-tool),
0:43/1:00 (one grasper tent), Issues A/B/C. Frame 1:11 is DESCOPED to a follow-on multi-tool work item
(listed in §11 for the user to confirm).** Do not let any step-gate claim 1:11.

**Two-jaw pinch-and-hold (0:05).** Expose the two jaw planes in world space (`JawSlabWorld` — a NEW
GrasperRig helper, NOT yet implemented; Round 2 named it but never wrote it; implement it alongside the
existing `CollisionSoupWorld`/`ShaftCapsuleWorld`): slab midplane `mid`, inward normal `nJaw`, current
gap `g(grasp01)` shrinking toward 0 as jaws close. While `grasp01 > graspThreshold`:

1. **"Between the jaws" corner-selection predicate (must be explicit — review found it undefined).** A
   corner `c` is grasped iff ALL of: (a) it is inside the jaw region — project onto the tool axis and the
   hinge→tip span, i.e. `dot(cornerPos[c]-hingeWorld, axis) ∈ [0, jawLenWorld]` AND radial distance to
   the axis `< jawHalfWidthWorld`; (b) it is within the slab half-thickness — `|dot(cornerPos[c]-mid,
   nJaw)| < g/2 + captureSkin` (captureSkin widens the band at the CAPTURE INSTANT so corners near the
   closing jaws are included; **[audit rev-B] because the set is latched once, corners entering the band
   LATER are deliberately NOT captured — that is the intended semantics, state it, do not imply ongoing
   catching**); (c) `active[c] && !pinned[c]`. Capture happens once, on the rising edge of
   `grasp01 > graspThreshold` (latch the set); do NOT re-select every frame (that would let the set churn
   as the tissue moves). **[audit rev-B — hysteresis, or dithering re-latches every frame:** release
   fires at `grasp01 < releaseThreshold` with `releaseThreshold < graspThreshold` (e.g. 0.3 vs 0.5,
   mirroring GripperTool.cs's intent latch) — a `grasp01` hovering AT the threshold must not re-capture
   from the deformed pose every dither, which would walk the grip and thrash the §6 `graspRing`.**
   Store each captured corner's offset in the jaw-local frame.
2. **Slab compression** — for each captured corner, project the NORMAL component into the slab:
   `dot(cornerPos - mid, nJaw) ∈ [-g/2, +g/2]` in the GS loop, where **`g` clamps to a floor
   `jawGapClosed > 0` (§11 item 5) so the band NEVER fully collapses.** [audit] A two-sided midplane band
   that lets `g→0` forces every captured corner onto the midplane (sheet collapse) and CANNOT produce the
   frame-0:05 bulge/crease (tissue displaced sideways around the jaws). Enforce `g ≥ jawGapClosed`, and
   couple the slab to the volume/strain constraint (§4 [Strain]) so the compressed tissue bulges sideways
   (incompressibility) instead of thinning to a sheet.
3. **Coulomb stick/slip attachment [audit: a NEW mechanism, honestly labelled].** The video's slip-out is
   driven by a per-corner tangential stick/slip anchor with a friction cone. **This is NOT a port of
   GripperTool** — the production repo's ONLY friction is unconditional tangential `prevPos` DAMPING on
   CONTACT (`XPBDSolver.compute:1056-1065`); its GRASP never slips (it releases only on jaw-open). So
   present this as a new mechanism justified by the video, and specify the two pieces PBD does not give
   for free:
   - **Normal force `Fn` — from an ALLOCATED, ACCUMULATED multiplier with pinned timing [audit rev-B —
     the bare "λ_normal/h²" had no data source and a chicken-and-egg].** Position-based XPBD gives no
     first-class `Fn`; three requirements make it real: (1) **Storage:** allocate `d_slabLambda[cornerCount]`
     (and `d_contactLambda[cornerCount]` for §4 [Contact]) — note the mirrored reference contact kernel
     (XPBDSolver.compute:1034) accumulates NO λ, so the ported kernel MUST be upgraded to the accumulating
     XPBD form (the `−α̃·λ` term needs it). (2) **Accumulation:** reset in §3 step-2 with the other
     multipliers; accumulate `ω·Δλ` across ALL iters AND K coupling passes within the substep; the
     two-sided slab λ is SIGNED — use `|λ_slab|`. (3) **Timing (breaks the chicken-and-egg):**
     `Fn = |λ_slab| / h²` from the PREVIOUS substep's converged value (seed 0 on capture) — the Coulomb
     cap consumes last-substep's normal force while this substep's λ converges; at N≈20 substeps/frame
     the one-substep lag is invisible. (Without all three, the cone either never slips or chatters.)
   - **Stick-anchor PARTIAL advance on slip [audit rev-B — "advance to the current position" zeroes
     kinetic friction].** Keep a per-corner tangential anchor. Each substep: if the tangential correction
     needed stays within `mu*Fn`, STICK (hold the anchor); if it exceeds `mu*Fn`, SLIP — advance the
     anchor TOWARD the corner's tangential position but leave it trailing by exactly the cone radius
     (`anchor += dir * (|excess| − coneRadius)`, the haptics.cu:588 partial-advance rule), so the anchor
     keeps pulling at `mu*Fn` DURING the slide (kinetic friction) instead of dropping to zero. This is
     the per-corner analogue of the global `g_stickAnchor`/`g_slipping` hysteresis (haptics.cu:577-589),
     which is a SINGLE world latch and cannot be reused per-corner as-is.
   - **Normal/tangential ownership (unchanged from the draft, still required):** the SLAB constraint owns
     the NORMAL (jaw-to-jaw compression); the stick/slip anchor owns only the TANGENTIAL (in-plane
     grip/drag/slip). Project the anchor target onto the slab mid-plane (remove its `nJaw` component) so
     the two do not double-constrain the normal and fight. On jaw-open, release all anchors → tissue
     springs back under the elastic constraints alone.

4. **PER-CORNER OWNERSHIP MATRIX [audit rev-B — BLOCKER fix: without it, jaw CONTACT (§4) and the SLAB
   fight over every captured corner's normal].** A corner squeezed between closed jaws is two-sided-
   infeasible for the jaw contact constraint at `jawGapClosed = 0.3× tool radius` — contact would chatter
   between winning jaws, over-squeeze, and split the normal multiplier so `mu*Fn` is wrong. Exactly ONE
   owner per axis per corner:
   | corner state | jaw contact (§4) | slab normal | Coulomb tangential |
   |---|---|---|---|
   | `graspCore==1` | SKIP (kinematic) | SKIP | SKIP |
   | `grasped==1` (ring) | SKIP for the OWNING tool's jaws (shaft + other tools still apply) | OWNS the normal | OWNS the tangential |
   | not captured | OWNS everything | — | — |
   Jaw `triRadius` must be set CONSISTENTLY with `jawGapClosed` (the contact skin of the two jaw plates
   must not overlap inside the clamped slab: `2·triRadius < jawGapClosed`), else even non-captured
   corners near the jaw tips sit in permanent two-sided violation. **Step-gate: the pinch VERIFY runs
   WITH contact enabled and asserts captured ring corners stay INSIDE the slab band (no expulsion, no
   midplane collapse) while non-captured corners are still repelled by the jaw surfaces.**

**[audit] Re-filter the latched grasp set on `active[]`/`alive` every substep.** The captured set is
latched once on the rising edge (below) and never re-checked; a cut that severs a grasped corner
mid-grasp would otherwise leave the attachment pulling now-debris, and the tear exemption (§6) would keep
it un-tearable. Skip attachment + slab + core-pin for ANY captured corner whose `active[c]==0`.

**Per-corner grasp state is NEW device infrastructure (review: the existing friction math is a SINGLE
world latch).** The hysteresis math at haptics.cu:577–589 (`g_stickAnchor`, `g_slipping`) is ONE global
god-object anchor — it CANNOT be reused per-corner as-is. Allocate device arrays `d_graspAnchor[cornerCount]`
(jaw-local offset), `d_graspSlipping[cornerCount]` (per-corner stick/slip latch), `d_grasped[cornerCount]`
(the capture flag, also read by the tear exemption §6), and `d_graspCore[cornerCount]` (1 = hard-pinned
core, 0 = dynamic ring — the core/ring split above). Budget this as new infrastructure alongside the
graph-colouring + tet-mesh work (§3/§4). The per-corner Coulomb math is the SAME formula as the global
one, just indexed per grasped corner against that corner's own anchor + `Fn=λ_normal/h²`.

`g_graspMu` becomes a secondary friction boost, not the mechanism.

**MECHANISM ACCURACY (do not misstate):** the grip is a HARD-pinned CORE (`invMass=0`, rides the jaw)
surrounded by a FULLY DYNAMIC transition ring — the smooth cone is the EMERGENT elastic result around the
core (matching `GripperTool.cs`), NOT a uniform compliant attachment. The tangential stick/slip cone
(step 3) is a NEW mechanism added on top of the core for the video's slip-out realism, justified by the
video, not a verbatim port of any external grasp. State both accurately.

---

## 6. The tear law — cuts still tear, the grasp tent does not

The tent stretches far past `tau=4L`, so `k_TearOverstretch` (cut.cu:354–399) would SEVER the grip at
the cone — the OPPOSITE of frames 0:43/1:00. Two independent guards, both MANDATORY and both landing
BEFORE the grasp constraint is enabled:

1. **Grasped-neighbourhood exemption (primary). [audit] Dilate the grasped set by 1–2 graph rings —
   incident-ring-only is NOT enough.** In a long tent the PEAK strain is NOT on the edges incident to the
   grasped/pinned core (the core rides WITH the tool, so its own edges barely stretch); it is on the
   NEXT-RING edges — active–active, non-grasped — which are dragged hardest and will exceed `tau·L = 4L`
   and TEAR, decapitating the tent just below the grip even with an incident-only exemption. So exempt any
   edge whose endpoint is within **N_exempt = 1–2 graph-hops** of a grasped corner: precompute a
   per-corner `graspRing` flag (BFS-dilate `d_grasped` by N_exempt over `nbrIdx`, recomputed when the
   grasp set changes), and in `k_TearOverstretch`, after the existing severed/inactive early-outs
   (cut.cu:371–372), add: if `graspRing[idA] || graspRing[idB]` → `return` (exempt). Grasp is deliberate
   large strain, not fracture. Still surgical — every edge outside the dilated neighbourhood keeps the
   tau=4L law, so the cut/tear behaviour (including the 藕断丝连 gravity-strand fix, cut.cu:342–352) is
   untouched elsewhere. **Do NOT rely on the incident ring alone.** (Secondary guard (2) below is an
   assumption until measured: §4's nonzero stretch compliance SHOULD bound per-edge stretch below 4L in
   the grasp neighbourhood, but §6 does not prove it — so guard (1)'s dilation is the load-bearing one,
   and the grasp step must add a gate that no next-ring edge tears.)

   **[audit rev-B — RELEASE DECAY (was a BLOCKER: instant exemption-clear shreds the released tent).]**
   On jaw-open the tent's edges are still FAR beyond `τ·L = 4L`; if `graspRing` clears instantly
   ("recomputed when the grasp set changes"), the memoryless per-substep `k_TearOverstretch` — armed for
   the whole session by the permanent `g_toolSet` latch — severs the former grip region on the very first
   post-release substeps instead of letting it spring back. **Rule: former-`graspRing` edges KEEP their
   exemption per-edge until that edge's length drops below a re-arm threshold `0.8·τ·L` (with a safety
   timeout of ~2 s in case an edge never relaxes, e.g. it snagged on a pinned boundary — then the normal
   law resumes and it tears honestly).** Cleared per-edge, not per-set. **Gate (grasp step VERIFY): pull
   the tent past 4L, RELEASE, assert clean spring-back with ZERO tears.**

   **[audit rev-B — RIM-BRIDGE carve-out (the exemption must not shield cut debris-strands).]** Inside
   the dilated ring the tear law is fully disabled — but the tear law's OTHER job is killing the
   rim-bridge 藕断丝连 strands a cut leaves behind, and a grasp-and-cut workflow (cut the fold you are
   holding) would leave those strands UN-tearable within the ring (their endpoints never trip the
   `active[c]==0` re-filter). **Rule: the exemption does NOT apply to any edge whose endpoint LOST a
   neighbour this session (corner has any `nbrIdx[6c+s] < 0` beyond its initial boundary state) — i.e.
   cut-rim edges keep the tau law even inside the ring.** Equivalently: recompute `graspRing` after each
   sever event inside the ring. **Gate: add a grasp+cut-the-fold case to the cut-step/grasp-step VERIFY**
   (cut through a held fold; assert the strands still tear while the grip itself does not).
2. **Nonzero stretch compliance (secondary, from §4).** Because structural `α̃ > 0`, grasped edges
   extend elastically instead of locking then over-straining — they reach the cone geometry without
   spiking toward the tear threshold in the first place.

**Tear-arming is a PERMANENT session latch — do not under-state this (review major).** `g_toolSet` is
set true the first time `cut_set_tool` is ever called and is NEVER cleared for the session
(cut.cu:949/975); the tear threshold `g_tearLen2` is armed per-desc (cut.cu:954). So the tent is NOT
"tear-safe today" in any realistic session: the moment a cut tool has EVER been configured (which the
demo scene does at startup), the tear law is armed for the WHOLE session, and a grasp tent stretched past
`tau*L` anywhere would be severed at the grip — even if the user is not actively cutting at that moment.
Therefore guard (1) (the grasped-edge exemption) is **MANDATORY for the tent in essentially every
session**, not merely when cutting and grasping "coexist." Prefer (1)+(2) together. **The `grasped` flag
must be wired before the grasp attachment goes live** — otherwise grasping a stiff or gravity-loaded
liver tears it at the jaws.

---

## 6.5 Does the CUTTING algorithm survive full-XPBD? — VERIFIED YES (audit wf_f9e9fa1f)

A dedicated 5-slice code audit + adversarial skeptic (all read the actual cut.cu/dc_recon.cu/physics.cu)
concluded **GO — cutting survives essentially VERBATIM.** All five slices returned `survives=verbatim`,
overall `survives-with-minor-changes` (high confidence), and the skeptic's `found_real_break=false`.

**Why it survives (the load-bearing fact): cutting is TOPOLOGICAL severing that operates ONLY on the shared
corner POSITIONS + the CONNECTIVITY GRAPH + cut masks — none of which depend on HOW positions are
integrated.** A targeted grep of `cut.cu` (all 1152 lines) for every RK45 artifact — `d_kSlope`, `d_errMax`,
`k_ComputeError`, `g_h`, `hStep`, `d_vel`, `yTrialVel`, `d_forceInt` — returns **ZERO hits**. Every cut
kernel reads only `cornerPos`/`d_prevCorner` (positions), `nbrIdx`/`bendPairs`/`voxelCutMask`/`conn4096`
(connectivity), and `particleRot` (material frame).

- **k_DetectCutCCD** (cut.cu:237-339): the CCD blade sweep derives motion from two POSITION snapshots
  (`Pv=a1-a0`, `Qv=b1-b0`, cut.cu:279-280), never physics velocity — so `v=(x-x_prev)/h` semantics are
  irrelevant. Verbatim.
- **k_SeverLinks** (cut.cu:405-433): pure connectivity mutation (`nbrIdx=-1`, bending `alive=0`). Verbatim.
- **k_TearOverstretch** (cut.cu:388-391): a pure `cornerPos` edge-length compare vs `(tau*voxelL)^2`, no
  velocity/slope/h. Verbatim (fires for a real cut identically).
- **k_ComputeParticleRot** (physics.cu:355-451): polar decomposition of position edge-vectors only. Verbatim.
- **dc_recon** dual-contouring/QEF + severed-edge rebuild (dc_recon.cu:118-142): a pure function of
  `(cornerPos, connectivity, cut masks, R)` → byte-identical surface regardless of which integrator produced
  `cornerPos`. Verbatim.

**[audit] There are TWO cut paths — the draft audited only ONE. Both survive; they behave DIFFERENTLY
under the migration:**
- **PATH A — tick-phase CCD** (`cut_ribbon_tick()` → `k_DetectCutCCD`, cut.cu:1009/237): the MOVING
  deformed tissue edge vs a static blade, run per accepted physics substep. **This path MOVES** into the
  XPBD fixed-N substep loop (the ONLY relocation — see below).
- **PATH B — rod-phase quad** (`cut_detect()` → `k_DetectCutQuad`, cut.cu:985/175, paper SS2.1.3): the
  MOVING blade vs static tissue, a world-quad Möller–Trumbore test the C# manager runs per rod substep
  BETWEEN `LCS_Step` and `LCS_Finalize`. It reads ONLY `c_cornerPos` + `c_particleRot`, so it survives
  VERBATIM by the same grep argument — but **it STAYS in the C# manager; do NOT move it into the substep
  loop.** Moving it would be a bug. Only PATH A moves.
- **Consistency requirement:** both paths must read the SAME XPBD-committed `c_cornerPos`. PATH B runs
  after `LCS_Step` commits `cornerPos`; PATH A runs inside `LCS_Step` post-substep-commit. §9-(6) must
  gate BOTH: an active-ROD cut (blade sweeping through static tissue → PATH B) AND a falling-tissue CCD
  cut (PATH A), each severing correctly against XPBD-committed positions.

**The ONLY changes (minimal, NOT a rewrite):**
1. **Relocate the `cut_ribbon_tick()` call site** from inside the RK45 `while(t<dt)` loop (physics.cu:570)
   into the XPBD fixed-N substep loop, still POST-position-commit, rolling `d_prevCorner` per XPBD substep
   (cut.cu:1040). A call-site MOVE, no kernel rewrite. The tick's contract is only "cornerPos committed +
   prev-snapshot rolled per committed sub-interval," which an XPBD substep satisfies identically — and
   **fixed h=dt/N is STRICTLY SAFER for no-missed-cut/no-tunnel than the adaptive RK45 h** (which could grow
   and skip a cut).
2. **The two force-side sever gates become identical constraint-side gates:** `k_AccStructural`'s
   `if(n<0)continue; if(active[n]==0)continue;` (physics.cu:163-165) → "skip emitting the distance
   constraint for that neighbour"; `k_AccBending`'s `if(bp.alive==0)return;` (physics.cu:188-189) → "skip the
   dihedral constraint." Same `nbrIdx`/`alive`/`active` flags, same effect (drop the constraint instead of
   the force). Dropping a constraint from a GS loop is cleaner/more stable than removing a stiff explicit
   force.

**Paper-native, not hostile:** the paper's cut/reconstruction method is lifted from Berndt et al. [22]
"Efficient surgical cutting with position-based dynamics" — so XPBD is the cut algorithm's **native home**,
and the switch moves the solver back TOWARD the cut method's own PBD formulation.

**Honest caveats (all migration-execution or tuning, NOT algorithm breakage):** (a) the tick MUST fire
post-commit and `d_prevCorner` MUST roll per substep — botching this is a cut-granularity/tunneling
regression, a migration bug not an inherent break; (b) size N per §9-(6): `N = max(N_cut, N_elastic)`,
`N_cut` from the TISSUE-travel bound `h·|tissueVel_max| ≲ voxelL` (bounds 14–200 at physicsDt=0.02);
(c) the material-frame R blend uses `particleRotPrev` (a per-FRAME temporal snapshot, physics.cu:606) — if
the outer frame cadence shifts, R's smoothing constant shifts slightly (a cut-FP reconstruction-QUALITY
detail, not a correctness/integrator dependence); (d) the grasped-neighbourhood tear exemption (§6) is a
GRASP-feature requirement, unrelated to cut survival; (e) **[audit rev-B] the step-7 tet constraints add
the ONE genuinely new cut interaction — the `d_tetActive` lifecycle (§4 [Strain]): without the
`k_DeactivateTets` pass, tets BRIDGE the cut. The sever machinery itself stays verbatim; the tets must
FOLLOW it.**

---

## 7. Issue A held-dent — falls out for FREE under XPBD

Round 2's Issue A needed: a new `k_ProjectNonPenetration` pass run AFTER the spring solve, a render
reorder (`LCS_Step → Finalize → StepHaptics → LCS_ProjectContact → Finalize AGAIN → BuildOrUpdateMesh`),
a new `LCS_ProjectContact` native entry, and a double-`Finalize`, all to stop the springs erasing the
dent (root cause A-cause-1: the additive push the springs recover, haptics.cu:325–350 vs
`k_AccStructural`).

Under full-XPBD ALL of that collapses to **a contact constraint solved inside the substep loop** (§4
[Contact]). Because `vel = (cornerPos - x_prev)/h` bakes the projected position into velocity, a
satisfied non-penetration constraint exerts no restoring force — the springs cannot erase it *by
construction*. The dent depth = tool penetration, Ks-INDEPENDENT, and it appears the SAME frame it is
imposed with no render reorder (the constraint runs inside the committed step, before the single
`Finalize`/recon the render already consumes). **[audit] The Ks-independent held depth relies on the
contact/elastic passes actually RECONCILING — i.e. the K=2 coupling passes + ω<1 of §3.5, projected LAST
each substep — NOT on a single coloured sweep magically converging. The step-5/8 gate must MEASURE that
dent depth == tool penetration (and does not drift with sweep order); it is not free from the algebra
alone.** **Retire** `k_ScatterContact`'s `-contactBeta*dentDepth/3` position write and
`k_ApplyContactDisp`'s `vel*=0.15` global damp; the constraint's targeted inward-normal-velocity kill is
strictly more correct. **KEEP** the `hapticForce` scatter (now folded into the predict step) — the felt
reaction is orthogonal and still wanted.

- A-Part 0 (rate-limit `_hapticCenter` optics clamp, LiverCudaManager.cs) is INTEGRATOR-INDEPENDENT and
  KEPT verbatim — it bounds the ghost/`dP` separation, unrelated to the solver.
- A-Part 2 (two-jaw grasp) becomes native constraints (§5) instead of bolted-on post-passes.

This is the biggest architectural simplification the swap buys: the entire Round-2 A ordering churn is
deleted, not reimplemented.

---

## 8. Issue B (smooth) and Issue C (near-0 trigger), folded in

**Issue B — smooth, tool-shaped, non-blocky deformation.** The render surface is ALREADY decoupled from
the physics lattice by per-voxel QEF dual contouring (dc_recon), so smoothness is NOT integrator-bound;
it is bound by (a) how corner-quantized the deformation is and (b) how fine the corner lattice can be
while stable. XPBD delivers both:
- **B-1** the per-corner contact constraint + GS relaxation diffuses the dent into a smooth basin (no
  winner-take-all argmin — the Round-1 K-nearest scatter is subsumed).
- **B-2** raise `targetLongAxisVoxels` 48→96 (SampleScene.unity + LiverCudaManager.cs) — L halves, QEF
  feature-point density doubles, the tent silhouette goes from faceted prism to round cone. XPBD's
  unconditional stability is what makes the finer, stiffer lattice affordable where the RK45 adaptive-h
  would starve. **This is the roundness lever the integrator swap does NOT provide on its own — ship
  both.**
- The bending network (§4) + nonzero stretch compliance are the mandatory anti-facet / anti-lock guards.

**Issue C — near-0 contact trigger.** KEEP the Round-2 swept-segment CCD plan verbatim; it composes
SYNERGISTICALLY with XPBD and gets STRICTLY EASIER. Detection (the CCD FREE-branch rising-edge trigger,
which needs only the TOOL kinematic trajectory `g_prevPhysPos`→`g_physPos`, haptics.cu:113/696/107) is
integrator-agnostic. XPBD substepping resolves contact once PER SUBSTEP, so per-substep tool
displacement is `1/N` smaller → the sweep length per resolution shrinks → `nsub→~1` and the first-touch
penetration window shrinks by the substep count. Under XPBD the detection band can legitimately be
`triRadius + numerical-eps (~1e-3)` with NO motion margin at all — closer to literal 0 than CCD-alone
promised. **The one refinement — GATE-DRIVEN, not unconditional [audit rev-B — M3: a prior draft
unconditionally relocated the detect while §9's step made it a fallback; reconciled to gate-driven]:**
IF the step-5 moving-tool gate fails, relocate the FREE-branch swept-CCD detect INSIDE the substep loop
(where `cut_ribbon_tick()` runs) — and that relocation REQUIRES the data-flow fix spelled out in
§9-(5): the tool pose today is uploaded only inside `StepHaptics`, AFTER `LCS_Step`
(LiverCudaManager.cs:522/557/608), so an in-loop detect would sweep a ONE-FRAME-STALE trajectory unless
the pose upload moves BEFORE `LCS_Step` and the pose is interpolated prev→current per substep. The
A-Part-1 "post-solve projection" that Issue C's phrasing assumed is now the native contact constraint
(§4/§7); read C's terminology accordingly. Issue C still DEPENDS on Issue D's per-primitive radius
threading (sweep vs BOTH jaw mesh and shaft capsule) — land D first.

---

## 9. Migration order (sim runnable and verifiable at every gate)

Each step keeps the sim working and has an explicit VERIFY. Do not proceed past a failing gate.

- **(0) Baseline.** Tag the RK45 build. Capture a golden gravity-settle rest shape and one known cut as
  regression references.
- **(1) XPBD substep loop, structural-only + gravity predict.** Replace the RK45 body with the fixed-N
  small-steps loop; ONLY the structural distance constraint (§4) + gravity in predict. **VERIFY against a
  BENDING-FREE reference (review: do NOT compare to the full RK45 golden here — that golden includes
  bending, which this step omits, so the shapes legitimately differ and the gate would false-fail).**
  Correct acceptance: (a) a single-edge / 1-D particle chain under gravity reproduces the analytic Hookean
  rest-length elongation `ΔL = m g / ks` (the direct `α̃=(1/ks)/h²` = exact-spring claim); (b) the full
  liver reaches a STABLE equilibrium with NO blow-up at the paper stiffness `ks=7.5e4` (the stability
  claim — the whole point of the swap). The "matches the RK45 golden rest shape" check is DEFERRED to step
  (2), after bending is added.
- **(2) Add bending.** Port the 15-pair dihedral constraint (§4). **VERIFY** (a) bending stiffness matches
  the RK45 flap behaviour (released-flap settle test); (b) NOW the full-liver gravity-settle rest shape
  matches the RK45 golden from step (0) within tolerance (this is the deferred structural+bending shape
  match); (c) **[audit rev-B — n2]** the LARGE-STRAIN unequal-arm tent-geometry bending golden-match
  (the §4 [Bending] gate; implemented as selftest K).
- **(3) Add damping.** Rayleigh `alpha` as the per-substep velocity term; `cs`/`cb` as XPBD damping or a
  velocity post-pass. **VERIFY** slosh/settle matches (no ring, no perpetual swing).
**[audit rev-B — the order below is RESTRUCTURED (blocker B1): the original put the buffer rebind at
"6.5", AFTER steps 4–6 whose gates (live dent readback, cut-tick relocation, both-cut-paths, tear
exemption) all REQUIRE the live pipeline — under "do not proceed past a failing gate" the old step 4 was
unexecutable. The rebind is now step (4); everything after it runs on the LIVE path.]**

- **(4) BUFFER REBIND + GO-LIVE in the DLL (was "6.5" — now FIRST after the standalone steps).**
  Steps 1–3 validated the module on its PRIVATE `d_pos`/`d_prevPos`/`d_vel` (the `xpbd_selftest`
  executable) with physics.cu / the shipped DLL BYTE-UNCHANGED. This step connects it live:
  - **Full bind signature [audit rev-B — types corrected (live buffers are `int*`, not `uint8_t*`;
    `restNbr` is `float3*`) and inputs completed (mass/extForce/hapticForce were missing)]:**
    ```
    xpbd_bind(float3* cornerPos, float3* vel,                       // recon_corner_pos(), physics_vel()
              const int* nbrIdx, const float3* restNbr,             // 6-slot topology + rest vectors
              const BendPairGpu* bendPairs, int bendCapacity,       // 15/corner, alive flags LIVE
              const int* active, const int* pinned,                 // LIVE flags (cut flips active at runtime)
              const float* mass,                                    // per-corner mass (invMass derived on-device)
              const float3* extForce, const float3* hapticForce,    // predict-step inputs (step 5)
              int3 dims, int cornerCount)
    ```
    `cornerPos = recon_corner_pos()` (dc_recon.cu:625 — the buffer cut.cu:932 / haptics.cu:377 / recon /
    particleRot all read); `vel = (float3*)physics_vel()` (physics.cu:620, haptics.cu:381). `invMass` is
    DERIVED on-device each refresh from `mass`/`pinned`/`active` (+`graspCore` later) — NEVER baked at
    init, because `cut.cu:451` flips `active[]` at RUNTIME (debris freeze).
  - **Edge-list derivation + SEVER PROPAGATION [audit rev-B — was the M1 hole: cut writes
    `nbrIdx[6p+s]=-1` (cut.cu:423-424) + `bendPairs.alive=0`, which the module's private colour-ordered
    `d_edgeActive`/`d_bendActive` would never see].** Canonical dedup: emit each lattice edge ONCE from
    its POSITIVE slot `s = 2·axis+1` of the lower corner (`restLen = length3(restNbr[6p+s])`); persist on
    device, per colour-ordered edge, its OWNER SLOT `edgeOwnerSlot[e] = 6p+s`, and per colour-ordered
    bend element its SOURCE INDEX `bendSrcIdx[b]` into the live `bendPairs` array. **Mechanism (pinned:
    the per-substep refresh, NOT a cut.cu callback):** a `k_RefreshLiveness` kernel at the START of every
    substep derives `d_edgeActive[e] = (nbrIdx[edgeOwnerSlot[e]] >= 0) && active[i] && active[j]`
    (the active-endpoint test is k_AccStructural's SECOND gate, physics.cu:165 — an edge to frozen
    debris exerts no force there, so the constraint must drop too; bending needs NO active test — a
    frozen member is handled by its w=0, matching k_AccBending's alive-only gate), `d_bendActive[b] =
    bendPairs[bendSrcIdx[b]].alive`, and `d_invMass[c] = (pinned||!active||graspCore) ? 0 : 1/mass[c]` —
    all from the LIVE shared flags, the SAME flags physics.cu's gates read (physics.cu:163-165, 189).
    Zero cut.cu changes; a sever is picked up the very next substep. The colouring stays STATIC (a
    dead edge just early-outs).
  - **Stepping API split [audit rev-B — M2: `cut_ribbon_tick()` must fire BETWEEN substeps, but the
    module's only entry ran all N internally].** Split: `xpbd_begin_frame(dt)` computes the h-derived
    coefficients once (α̃, γ, velScale — §12 R-compliance); `xpbd_substep()` advances ONE substep
    (refresh-liveness → predict → sweeps → velocity write — refresh FIRST, per the "START of every
    substep" mechanism above, so predict already sees a fresh invMass). `physics_step` (`LCS_Step`)
    owns the N-loop: `xpbd_begin_frame(dt); for n in 1..N { xpbd_substep(); cut_ribbon_tick(); }` —
    preserving `tickErr` surfacing (physics.cu:571/591) and the per-substep `d_prevCorner` roll
    (cut.cu:1040). (`xpbd_step(dt)` stays as a thin wrapper for the standalone self-test.)
  - **Bind-time state [audit rev-B — first-frame kick prevention]:** on go-live, initialize
    `d_prevPos = cornerPos` (else the first velocity write reads garbage) and ADOPT the current
    `physics_vel()` contents (do not zero a moving liver's velocity).
  - Swap `physics_step`'s body to the loop above at the point `plugin_api` calls it. This is the FIRST
    edit to physics.cu — until here the user's byte-unchanged constraint holds; after here the RK45 body
    is retired (§10 D3/D4/D6 VOID). **C# checklist [audit rev-B]:** `maxSubstepsPerFrame` becomes N —
    update BOTH the C# default AND the serialized value in SampleScene.unity (else the scene silently
    ships the old N=16); document `hInit`/`hMax` as dead on the C# side.
  - **VERIFY (the divergence check):** read back ONE deformed corner via `recon_readback_corner_pos()`
    (dc_recon.h:69) after the swapped `LCS_Step`, confirming cut (`c_cornerPos`), haptics
    (`c_cornerPos`/`c_vel`), recon, and particleRot all read the SAME deformed `cornerPos` XPBD produced.
    Re-run the step-0 gravity-settle golden + the known cut against the live path (structural+bending+
    damping only — same constraint set as steps 1–3, so the goldens transfer). **Sever-propagation gate:
    cut mid-frame; assert the corresponding constraint drops the SAME frame (readback `d_edgeActive`) and
    a multi-cut session keeps propagating.**
- **(5) Move `hapticForce` into predict** (now live, so the gates are executable). **VERIFY** the felt
  reaction now moves the tissue IN the solver: read back `cornerPos` — a sustained force dents and HOLDS
  (converges, does not oscillate/run away); withdraw → clean spring-back, no crater. **[audit rev-B — Ks
  scoping corrected]** at step 5 the dent is a FORCE dent (`a = F·invMass`, so depth ≈ F/k_eff, Ks-
  DEPENDENT — it SCALES ~1/ks). The Ks-INVARIANT "dent depth == tool penetration" is a property of the
  step-8 contact CONSTRAINT (§7), NOT of this force dent — do NOT assert Ks-invariance here; that gate
  lives at §9-(8). **[audit] Add a
  MOVING-tool gate — a static press hides a one-frame latency.** Haptics runs POST-`Finalize` once per
  frame (LiverCudaManager.cs:562), so `hapticForce` is a STALE-by-one-frame input to the next frame's
  fixed-N predict — RK45 masked this by re-reading LIVE `d_hapticForce` every stage (physics.cu:557);
  fixed-N XPBD freezes it for the whole frame. Sweep the tool at video-representative speed and confirm
  the dent TRACKS with no one-frame lag/overshoot. **[audit — headless proxy vs in-app]** the headless
  live-path gate (xpbd_livepath_test GATE E) proves the testable slice: E1–E4 that predict CONSUMES
  hapticForce, and E5 an ONSET/mutation gate — a K-frame-stale ramp shows the real 1-frame staleness dents
  PROMPTLY while an 8-frame lag delays the onset (<40%), so the gate is NON-VACUOUS and would catch a
  multi-frame regression. The FULL video-speed sweep lag/overshoot characterization is confirmed IN-APP
  (a coarse 4³ lightly-damped saturating lattice is not a faithful swept-tool proxy — the dent saturates,
  hiding a magnitude-ramp lag) — it joins the deferred Unity checklist alongside the visual golden + the
  real-blade cut regression. §3 states `hapticForce` is a one-frame-delayed predict input under fixed-N; **[audit rev-B — M3 reconciliation: the in-substep-loop haptic detect (§8) is
  GATE-DRIVEN, not unconditional — it fires ONLY if this moving-tool gate fails.** If it does: split the
  tool-pose upload OUT of `StepHaptics` and move it BEFORE `LCS_Step` in `Update()` (today the pose
  arrives AFTER LCS_Step, LiverCudaManager.cs:522/557/608 — an in-loop detect would otherwise sweep a
  one-frame-STALE trajectory, defeating the purpose); interpolate prev→current pose per substep
  (mirroring PATH B's rod substepping); accept + gate the last-frame-triangle staleness.]
- **(6) Move `cut_ribbon_tick()` between substeps** (via the step-4 API split — the call site now exists).
  **Pin N = max(N_cut, N_elastic) [audit rev-B — numbers corrected (m1): the outer physics step is
  `physicsDt = 0.02 s` (LiverCudaManager.cs), NOT 1/60].**
  - **`N_cut` (no-tunnel). [audit rev-B — the criterion's ACTOR corrected]:** PATH A's CCD sweeps the
    MOVING TISSUE against a rod held frozen during the physics phase — so the binding requirement is
    **one substep's TISSUE travel must not skip the blade: `h·|tissueVel_max| ≲ voxelL`** (worst case =
    fast-falling severed pieces; a static tool does NOT permit N_cut→1). Tool motion is PATH B's job (the
    rod-phase quad runs per C# rod substep, independent of N). Measured-`g_h` bounds map to
    `N_cut = ceil(physicsDt / g_h)` ∈ [ceil(0.02/1.5e-3)=**14**, ceil(0.02/1e-4)=**200**].
  - **`N_elastic` (convergence).** From an elastic-convergence test: does the gravity-settle rest shape
    AND the 10× tent CONVERGE at the chosen iters/coupling-passes per substep? Because `α̃=(1/ks)/h²`
    grows ~`1/h²` with N, each edge gets SOFTER per substep, so 1-iteration GS UNDER-converges per frame
    at very large N. **Larger N is NOT unconditionally safe** — beyond `N_cut`, prefer a 2nd iteration or
    K coupling passes (§3.5) over pushing N toward 200. Reference N=10 (XPBDSolverGPU.cs) is the sane
    order of magnitude for elasticity.
  Land the tear-law PREP here, INERT [audit rev-B — B1: the grasp does not exist yet, so the old "grasp
  tent does not tear" gate was untestable at this step]: allocate `d_grasped`/`graspRing` ZERO-FILLED,
  add the `k_TearOverstretch` early-out (§6), and unit-test it with a SYNTHETIC `d_grasped` pattern (set
  a flag, overstretch, assert exempt; clear, assert tears). The real grasp-tent gate lives at step (8).
  **VERIFY** the existing cut regression passes, no tunneling, BOTH cut paths sever correctly (PATH A
  falling-tissue CCD + PATH B active-rod quad, §6.5) against XPBD-committed `cornerPos`.
- **(7) Add the strain constraint** — **DONE** (commit c92d5ed). 6-tet parity-alternating build in
  `xpbd_build_tets` (host, one-time from the bound live buffers); fused dev-then-hyd solve per tet colour;
  tet cut/tear lifecycle folded into `k_RefreshLiveness` (zero cut.cu changes — mirrors the edge/bend
  sever-propagation mechanism exactly); `cs` perpendicular velocity post-pass (raw/perp-damp/velscale
  3-kernel split). E=3e4/nu=0.45 are a §11 item 3 STARTING ANCHOR (native-only `XpbdParams` fields, not
  C#-marshalled) — tune against the video once visible. VERIFY: G1 rest-stress (zero-gravity, exact 0
  drift), G2 isotropy (+X/+Z/diagonal within 1.02x), G3 cut-with-tets (severed layer separates, tets do
  not glue). livepath 26 gates + standalone 58 + oracle 6 ALL PASS; haptics.cu/cut.cu/dc_recon.cu
  byte-unchanged for this step.

  Design recap (§4 [Strain], per-tet Stable Neo-Hookean — the pinned rest-zero pair
  `tr(FᵀF)−3` + `det(F)−1`). Build the tet mesh (6-tet parity-alternating split, `TetIds`/
  `InvRestMatrix`/`RestVolume` with `|det|/6` + build asserts) + its greedy colouring + `d_tetActive` +
  the `k_DeactivateTets` cut/tear lifecycle pass (§4). Add the `cs` PERPENDICULAR velocity post-pass
  (§4 [Structural]) — the tent regime begins here. **VERIFY** (a) the pulled tent is a smooth
  NON-COLLAPSING cone; (b) REST-STRESS: a zero-gravity tet lattice holds `det(F)=1` with no drift;
  (c) ISOTROPY: tents pulled along ±x, ±y, and a diagonal read the same (the parity alternation works);
  (d) CUT-WITH-TETS: the step-0 known cut re-run with tets live opens a measured separation gap (tets
  die, halves do NOT stay glued) + one tear-through-tet test; (e) increasing K coupling passes (§3.5)
  removes any grip-vs-volume fight; tune E/ν (ν=0.45 start).
- **(8) Fold contact + two-jaw grasp** — **DONE** (commit pending-audit), with THREE flagged deviations
  from the literal text below, tracked here for the user's review:
  **[flag 1 — analytic jaw contact, not the literal triangle mesh]** §4[Contact]'s "jaw-mesh ∪
  shaft-capsule" is implemented as shaft-CAPSULE (as designed) + jaw-as-an-analytic-ORIENTED-BOX (the
  SAME `JawSlabWorld` mid/normal/gap/hinge/axis/halfWidth/len primitive §5 already uses for the slab,
  reused for non-penetration too), NOT a second independent triangle-soup closest-point routine ported
  alongside haptics.cu's (haptics.cu stays byte-unchanged). Justification: every OTHER jaw-geometry use
  in this design is already analytic; porting a full closest-point-on-triangle-soup (Ericson-class
  vertex/edge/face region code) was judged higher-risk than reusing the primitive the design itself
  favours elsewhere. **[flag 2 — Coulomb cone radius is a documented interpretation, not derived]** Fn=
  |λ_slab_prev|/h² (design-specified); the cone RADIUS (a position-space quantity) is `mu*Fn*h²*invMass`
  — the same XPBD force↔position relationship (Δx=w·Δλ) the constraint solve itself uses, chosen because
  the design does not pin an exact conversion ("tune mu against the video's slip-out feel" is explicit
  in §11 item 6). **[flag 3 — 48→96 grid bump NOT APPLIED]** M11's `targetLongAxisVoxels` 48→96 (+
  `massScale=0.125` retune, already-existing C# infrastructure) is a documented, ready-to-apply RECIPE
  (see the design-sync note at the end of this item) but was deliberately left for the user to apply —
  §11 item 2 itself asks the user to "confirm" it, it 8x's the corner count/memory footprint of an
  already-tuned scene, and the user has not yet seen the step-7/8 tent at the CURRENT resolution.
  Implemented: per-corner `d_grasped`/`d_graspCore` (the "one w-rule" — `invMass=0` iff pinned||!active||
  graspCore, re-derived every substep in `k_RefreshLiveness` alongside edges/bend/tets, so core-pin
  release is AUTOMATIC — no manual invMass save/restore); `k_MarkGraspCandidates` + host-side
  normalized-ellipse core selection (0.55 fraction of the CAPTURED SET's own u/v extents) + mandatory
  nearest-corner fallback; `k_ApplyCorePin` (kinematic, re-targeted every substep from the current jaw
  pose); `k_SolveJawShaftContact` (shaft always, jaw-box for UNCAPTURED corners only — the ownership
  split) + `k_SolveSlabCoulomb` (ring-only slab band + Coulomb stick/slip with jaw-LOCAL anchor so rigid
  tool motion never false-slips); K=2 coupling passes (elastic+strain, then contact/grasp — LAST pass
  ends on contact/grasp) gated on a NEW `g_toolArmed` latch (an all-zero default `XpbdToolState` is NOT a
  safe "definitely inert" sentinel by itself — its degenerate jaw box sits at the world origin, which can
  coincide with real corner geometry there; K stays 1 and contact/grasp kernels are skipped until the
  tool has been armed at least once, preserving GATE B's bind-plumbing bitwise parity). `cut.cu`
  extensions (already-unlocked file): `cut_recompute_grasp_ring` (BFS ring dilation + rim-bridge
  carve-out via an `nbrIdx`-at-`cut_init` snapshot), release-decay per-GRID-EDGE state (`d_edgeWasExempt`/
  `d_edgeDecayTicks`, 0.8·τ·L re-arm threshold, ~2000-tick/~2s safety timeout) in `k_TearOverstretch`.
  `physics.cu` bridges `xpbd_set_tool`/`cut_recompute_grasp_ring` per frame (grasp/contact do NOT
  hard-depend on the cutting subsystem being initialized). New (FIRST-touch) `plugin_api.cpp` exports
  `LCS_SetGraspTool`/`LCS_SetGraspParams`/`LCS_GraspToolAbiVersion` (unavoidable — step 8 needs genuinely
  NEW per-frame data, jaw-plane geometry, that no earlier step's buffers carry; every step through 6 kept
  plugin_api.cpp frozen specifically because it never needed new inputs). New `GrasperRig.JawSlabWorld()`
  (C#, derives hinge/axis/normal/mid/gap in WORLD SPACE from the SAME hinge+open-angle state
  `CollisionSoupWorld` already tracks) + `LiverCudaManager.cs` wiring (new Inspector-tunable grasp/contact
  fields, `StepHaptics()` calls `LCS_SetGraspTool` each frame — the SAME once-per-frame staleness contract
  §9-(5) established for `hapticForce`). VERIFY: H (pinch-hold — non-empty capture+core fallback,
  captured corners resist gravity-driven drift far more than an uncaptured same-layer neighbour over a
  150-frame window sized to this lattice's known-slow alpha=2 settling, design-precedent from the E5
  saga), I (grasp→graspRing integration — BFS dilates exactly `nExemptRings` hops, no over-reach).
  livepath 28 gates + standalone 58 + oracle 6 ALL PASS; haptics.cu/cut.cu(pre-existing lines)/
  dc_recon.cu byte-unchanged; plugin_api.cpp's FIRST edit this migration (additive only). **Honestly
  UNTESTED by this session's automated gates** (Unity-dependent, cannot run headlessly): the C# GrasperRig/
  LiverCudaManager wiring itself, the full video-speed tent-pull+release visual/dynamic soak, the two-jaw
  pinch visual crease, and Issue C's near-0 trigger characterization — these join the standing Unity
  checklist (visual gravity-settle golden, real-blade cut regression) alongside the NEW items above.

  **Multi-subagent conformance audit (6 angles, build+mutation-tested on the RTX 3090, not read-only)
  — verdict: zero live solver/physics defects; 7 findings, ALL test-coverage/discriminating-power gaps
  in the gate suite itself (exactly the class of issue this process exists to catch), fixed and
  re-verified by mutation-test:**
  (1) [major] G2's isotropy gate (dynamic force-response comparison) was proven EMPIRICALLY BLIND to a
  disabled-parity-alternation regression at every direction/corner/force tried (root cause: X-vs-Z at
  ANY interior corner is blind BY SYMMETRY — both the parity pattern (x+y+z)&1 and the un-flipped main
  diagonal (1,1,1) are invariant under x<->z exchange). REPLACED with a direct STRUCTURAL check
  (readback actual tet corner-id quadruples via a new `xpbd_readback_tet_ids` accessor, confirm two
  adjacent differing-parity cells' tets match their OWN expected flip and NONE of the other's) —
  100%-deterministic, mutation-confirmed (baseline 6/6 + 6/6 correct; flip=0 mutant 0/6 + 6-matching-
  wrong-flip on the odd cell). The old dynamic pull is retained as an unassorted diagnostic only.
  (2) [major] Gate H's gravity-sag comparison could not distinguish a working `k_ApplyCorePin` kinematic
  retarget from a no-op frozen-in-place artifact of the invMass=0 w-rule (mutation-confirmed: a no-op
  ApplyCorePin passed identically). FIXED: gate H now moves the synthetic jaw mid-test and asserts the
  core corner tracks the new pose (mutation-verified: dx moved == dx commanded, exactly).
  (3) [minor] Gate I exercised only the BFS-dilation half of `cut_recompute_grasp_ring`, zero coverage of
  the rim-bridge carve-out. FIXED: added a sub-check severing an edge incident to a ring corner and
  confirming it is carved OUT of the ring (mutation-verified: ring flag flips 1->0 after the sever).
  (4) [minor] G1's rest-stress metric structurally cannot catch a `youngE<=0` dispatch-gating regression
  (C_dev=C_hyd=0 identically at F=I regardless of gating). FIXED: documented G1's actual scope inline
  (rest-equilibrium only) rather than silently implying broader coverage.
  (5-7) [nit] core-ellipse centering convention vs the cited reference (no functional defect, left as-is
  per the audit's own recommendation), and a doc-commit-sequencing note (no code impact). Livepath now
  30 gates; the three flagged deviations (a/b/c above) were independently confirmed reasonable.

  Design recap (§3.5/§4/§5), retire the old
  dent write + render reorder (§7). Add the core/ring split (normalized-ellipse core + fallback, pin
  lifecycle with invMass save/restore + vel=0), the OWNERSHIP MATRIX (§5 item 4), `JawSlabWorld`,
  Coulomb stick/slip (`d_slabLambda`/`d_contactLambda`, `Fn=|λ_slab|/h²` prev-substep, partial anchor
  advance), the 1–2-ring `graspRing` dilation + RELEASE-DECAY + rim-bridge carve-out (§6), `active[]`
  re-filter, capture/release hysteresis. Run the contact/grasp pass under-relaxed (§3.5) with a small
  NON-ZERO contact compliance. **[audit rev-B — M11: `targetLongAxisVoxels` 48→96 lands HERE, so
  RE-DERIVE `N_cut` with the halved voxelL, RE-RUN the step-6 cut/no-tunnel regression at 96, and expect
  `N_exempt` (graph-hops) to need doubling — the ring's PHYSICAL width halved.]** **VERIFY the
  acceptance bar (single-tool, 1:11 descoped per §5):** smooth large-strain tent riding the tool without
  tearing at the grip OR the next ring (0:43/1:00); pull past 4L → RELEASE → clean spring-back, zero
  tears; two-jaw pinch-hold with a visible crease, jaw gap ≥ `jawGapClosed`, WITH contact enabled —
  captured corners stay in-band (0:05); grasp+cut-the-fold — strands still tear, grip does not; near-0
  trigger (Issue C); smooth non-blocky dent (Issue B, incl. 48→96); dent depth == tool penetration with
  no hapticForce double-count.

---

## 10. Paper deviations (all flagged)

- **INTEGRATOR — wholesale replacement (pre-approved by the user).** The paper's SS3.4 Dormand–Prince
  RK45 (Eq17–21) and its adaptive-h controller are RETIRED entirely. Deviation-ledger entries D3
  (a=(Fext+Fint)/m), D4 (adaptive h), D6 (cross-substep error accumulator) become VOID and must be
  re-documented as "superseded by XPBD small-steps substepping," not silently dropped. `k_ComputeError`,
  the DOPRI_B4 row, `d_errMax`, the `g_h` controller, `hInit`/`hMax` all lose meaning; `maxSubsteps` is
  repurposed as the fixed substep count N. **This deviation is confined to the integrator** — it stays
  inside the paper's OWN cited PBD family (the paper cites Macklin/Müller XPBD as ref [7] and Berndt
  PBD-cutting as ref [22], and its own demo video is an XPBD-class result).
- **STRAIN constraint — one addition, per-tet Stable Neo-Hookean** (§4 [Strain]) **[audit rev-B]**. Not
  in the paper's mass-spring model; the BASELINE (not an upgrade) is the REST-ZERO pair deviatoric
  `tr(FᵀF)−3` + hydrostatic `det(F)−1` on a 6-tet parity-alternating hex→tet decomposition — EXACTLY the
  user's production solver (XPBDSolver.compute:219/311). A scalar per-cell `C=V−V₀` is retired (does not
  stop filament collapse and is undefined on a tet-less lattice); the mixed pair with Macklin's
  `det(F)−(1+μ/λ)` offset is retired too (inconsistent with a rest-zero deviatoric — inflates every tet).
  Includes the tet CUT/TEAR lifecycle (`d_tetActive`, §4) — a new, flagged interaction with the paper's
  cutting. Needed for a smooth non-collapsing cone.
- **MASS MODEL — PAPER-SILENT [audit rev-B — m6]:** the paper never specifies corner mass; the code uses
  uniform `m = 1.0` per corner (invMass=1, pinned=0). Retained under XPBD. At `targetLongAxisVoxels`
  48→96 mandate volume-proportional mass (÷8) or an explicit retune + fresh goldens (§11 item 2).
- **DAMPING TERMS RETAINED FROM THE RK45 PATH — PAPER-SILENT [audit rev-B — m8]:** the Rayleigh `alpha`
  drag and the scalar `(1−damping)` velocity kill are NOT in the paper; they are inherited numerical
  damping, retained in the XPBD post-solve velocity write (`velScale`), re-derived per-N per
  §9-(3)/§12 R-compliance.
- **SOLVER LOOP — small-steps + under-relaxed coloured GS + K coupling passes** (§3.5) **[audit]**. The
  paper does not specify a GPU projection order; ω=0.2–0.5 under-relaxation and K=2 alternating
  internal/contact-grasp passes are adopted from the user's production solver (XPBDSolverGPU.cs), not from
  paper fidelity. A single coloured sweep does NOT converge the stiff coupled set.
- **TWO-JAW GRASP — a genuine new deviation** (§5). **[audit rev-B — m9: entry extended.]** The paper
  SHOWS grasping (Fig 3.11) but specifies NO mechanism; the paper's tool model is a single unified
  capsule. Added for the video goal: the two-jaw slab + Coulomb tangential stick/slip, PLUS (i) the
  **hard core-pin** — a KINEMATIC `invMass=0` override of the paper's Eq9 corner dynamics for the core
  set while grasped — and (ii) the **graspRing tear exemption + release-decay + rim-bridge carve-out**
  (§6), a scoped modification of the pre-existing tear-law deviation D10. Justified by the video +
  grasping literature + the production reference, not by paper fidelity.
- **TOOL as jaw-mesh ∪ shaft-capsule** (Issue D) and **contact scattered to background-grid corners**
  (not the colliding triangle's own vertices) are PRE-EXISTING deviations, inherited unchanged.
- **IN-SOLVER TISSUE NON-PENETRATION — a flagged deviation that SUPPLEMENTS the paper [audit rev-B — m7:
  a prior draft booked this as "moves closer to the paper", which inverted the paper's mechanism].** The
  paper's SS3.2 + Algorithm 1 non-penetration projects the PROXY TOOL out of the tissue (god-object;
  tissue deforms only via the thresholded reaction force) — that mechanism is KEPT UNCHANGED in
  haptics.cu. What §4 [Contact] adds is NEW: a tissue-corner non-penetration constraint inside the
  solver, so the tissue itself cannot interpenetrate the tool between haptic updates. It supplements the
  paper's reaction-force deformation (§7 keeps the hapticForce scatter); it does not replace or
  re-implement Algorithm 1.
- **The mass-spring MODEL (SS2.1 topology, rest lengths per the D2-corrected Eq11 — the paper's PRINT
  omits the L0 term; the code-side D2 correction `(len−L0)` is what "verbatim" means here [audit rev-B —
  n1]), CUT (SS2.1.2 / [22]), RECON (SS2.2 QEF), and the anchored boundary condition (pinned corners)
  are UNCHANGED** — full fidelity retained.

---

## 11. Remaining USER decisions (flagged)

1. **Substep count N. [audit] `N = max(N_cut, N_elastic)`, two criteria (§9-(6)) — NOT `ceil(dt/g_h)`
   alone.** `N_cut` (no-tunnel, from `h·|tissueVel_max| ≲ voxelL` — the TISSUE moves in PATH A, not the
   tool — bounds 14–200 at `physicsDt=0.02`) governs cut granularity ONLY. `N_elastic` (from a
   convergence test on the gravity-settle + 10× tent) governs elastic accuracy; because `α̃` grows
   ~`1/h²`, pushing N far past `N_cut` UNDER-converges 1-iteration GS per frame — so do NOT "err larger"
   blindly. Beyond `N_cut`, add a 2nd iteration or K coupling passes (§3.5) instead of inflating N.
   Reference N=10 is the right order for elasticity. Confirm 60 fps at the chosen N × iters × K × grid.
2. **Grid resolution.** `targetLongAxisVoxels` 48 → 96 for a round cone (Issue B-2). Confirm the ~cubic
   corner-count increase is acceptable (it is, on the 3090, per the user's perf non-constraint).
   **[audit rev-B — m6: the bump also OCTUPLES total mass at unit corner masses. Mandate
   volume-proportional mass (÷8 per corner at 96) or an explicit gravity/stiffness retune + fresh
   goldens; re-derive `N_cut` at the halved voxelL and expect `N_exempt` to double (§9-(8)).]**
3. **Strain-constraint material. [audit rev-B — the old "start with cheap per-cell volume" item is
   OBSOLETE: §4 [Strain] retired scalar volume as non-implementable/insufficient.]** The real open
   decisions: confirm **E** (magnitude vs the video's tissue read; anchor: ks=7.5e4 lattice) and
   **ν = 0.45** (recommended start), and confirm the 6-tet parity-alternating split (§4) over any
   alternative. Decide after seeing the step-7 tent against the video.
4. **Grasp control mapping.** Currently holding **G** drives `_grasp01` (LiverCudaManager.cs:595).
   Confirm G stays the grasp control.
5. **`jawGapClosed`** — residual thickness the fully-closed jaws leave (0 = full pinch-through, visually
   violent; a small positive gap reads as "holding compressed tissue"). Recommend ~0.3× tool radius
   (with jaw contact skin `2·triRadius < jawGapClosed`, §5 item 4); approve against frame 0:05.
6. **Grasp feel tuning. [audit rev-B — the old "attachment compliance" knob no longer exists (the grip
   is a hard core-pin, §5).]** The real knobs: Coulomb **μ** (when the tent slips out), **`jawGapClosed`**
   (crease depth), and the **core ellipse fraction** (0.55 start — how much tissue rides the jaw
   kinematically). Tune all three against the video's slip-out feel.
7. **Multi-tool scope. [audit rev-B — B6 descope decision, needs your sign-off.]** Frame 1:11 (two
   graspers + a probe, ≥3 simultaneous tools) is DESCOPED from this migration's acceptance bar — the
   native pipeline is single-tool global state throughout (one proxy/anchor, one `g_toolSet`, no tool
   index in the plugin API), and multi-tool is real infrastructure (§5 MULTI-TOOL REALITY CHECK). Confirm
   the descope (recommended), or commission the multi-tool work item as a follow-on phase.

---

## 12. Migration RISKS (verified)

- **(R-parallelism, main cost)** GPU Gauss–Seidel ordering is the real work: XPBD projection writes
  shared corners and races a naive edge kernel. Mitigate with graph-coloured 6-neighbour batches (the
  regular grid colours cleanly) or averaged-Jacobi + under-relaxation. **[audit] Convergence needs ω<1
  under-relaxation (§3.5, default 0.2–0.5) + K=2 coupling passes for the contact/grasp step — a single
  coloured sweep does NOT converge the stiff coupled set.** Plus the per-tet SNH tet-mesh + its colouring
  (§4). New infrastructure the hybrid avoids — the genuine cost of going full-XPBD.
- **(R-cut, HIGHEST)** `cut_ribbon_tick` placement: the tick MUST fire AFTER `cornerPos` is committed
  per substep (matching today's post-`k_IntegrateY5` slot) via the §9-(4) `xpbd_substep()` API split, or
  CCD sub-interval granularity/tunneling changes; `N ≥ N_cut` per §9-(6) (`h·|tissueVel_max| ≲ voxelL`).
- **(R-tear)** tau=4L tent-sever is NOT auto-fixed by the swap — the grasped-neighbourhood exemption +
  release-decay + rim-bridge carve-out (§6) are mandatory and independent; the exemption machinery lands
  INERT at §9-(6) and goes live with the grasp at §9-(8).
- **(R-compliance) [audit] centralise ALL h-dependent coefficients, not just `α̃`.** The damping/compliance
  terms have DIFFERENT h-powers: `α̃ ~ 1/h²` (structural, bending, deviatoric, hydrostatic — each with its
  own stiffness), `γ ~ 1/h` (the along-constraint damped term `γ=α̃·β·h`), and the Rayleigh velScale `~h`
  (`1/(1+α·h)`). Recompute ALL of them in ONE place whenever N (hence h) changes. A naive port that
  updates `α̃` but forgets `γ`/Rayleigh silently changes the effective damping; one that hardcodes a
  stiffness and forgets the `/h²` reintroduces stiffness-vs-timestep coupling and defeats the point.
  **[audit] The legacy scalar `damping` knob (`v *= (1−damping)` per substep) is ALSO N-dependent: its
  per-frame retention is `(1−damping)^N` (e.g. 0.98²⁰≈0.67, a 33% per-frame kill at N=20), so RETUNE it
  when N changes — unlike `alpha`, which §3/§9-(3) re-derive per-N. Velocity-scaling only, so it cannot
  move the equilibrium (a tuning/doc note, not a correctness bug); prefer `alpha` for physical damping.**
- **(R-bending-damp) [audit rev-B — REWRITTEN; the old "must move to the velocity post-pass or it is
  lost" was FALSE and would have made an implementer rip out verified damping.]** The in-constraint
  `γ_B = α̃_b·cb·h` IS the complete port of `cb*θ̇` (§4 [Bending]); the risk is only that a re-implementation
  leaves `γ_B` inert (the factor-k class of bug). Guard: keep selftest J (released-hinge isolation) green;
  NO bending velocity post-pass exists or is needed.
- **(R-GS-anisotropy)** pure GS over a regular grid can bias smoothness along the sweep axis; coloured
  batches or a few Jacobi+over-relaxation passes keep it isotropic. Untuned iteration count reads as a
  slightly faceted/directional cone — a tuning risk, not a stability one.
- **(R-frames, LOW)** `physics_compute_rotations` / `k_ComputeParticleRot` is integrator-agnostic
  (reads committed `cornerPos`+`nbrIdx` after the solve) — unchanged as long as XPBD commits `cornerPos`
  before `LCS_Finalize`.
- **(R-buffer, now BLOCKER-grade until §9-(4) lands)** `vel = (cornerPos - x_prev)/h` must run BEFORE
  haptics reads/damps `d_vel` — same ordering contract as today; `cornerPos`+`vel` remain the ONLY
  persisted state, `cornerPos` shared with recon, or recon/cut-FP/particleRot break. **[audit] The
  step-1..3 module owns PRIVATE `d_pos`/`d_prevPos`/`d_vel`, DISCONNECTED from `recon_corner_pos()` /
  `physics_vel()`. Until the §9-(4) rebind (`xpbd_bind` → shared pointers, `k_RefreshLiveness`,
  `physics_step` body swapped) lands, cut/recon/haptics/particleRot read the OLD cornerPos while XPBD
  deforms a private buffer — silent divergence. This is the single biggest correctness risk; the
  divergence-check + sever-propagation VERIFY in §9-(4) is mandatory.**
- **(R-strain-tuning) [audit rev-B — "start cheap per-cell volume" DELETED; scalar volume is retired
  (§4 [Strain])].** The per-tet SNH pair needs an E/ν tune (§11 item 3) + the tet lifecycle gates
  (§9-(7): rest-stress det=1, isotropy, cut-with-tets separation).
- **(R-roundness expectation)** XPBD alone does NOT round the silhouette — set this expectation
  explicitly; the grid bump (B-2) is the companion lever.

---

## 13. Sources (published literature only)

- Müller, Heidelberger, Hennix, Ratcliff, *Position Based Dynamics*, JVCIR 18(2):109–118, 2007.
- Macklin, Müller, Chentanez, *XPBD: Position-Based Simulation of Compliant Constrained Dynamics*, MIG
  2016 — https://matthias-research.github.io/pages/publications/XPBD.pdf (compliance α̃=α/dt² reproduces
  a spring of stiffness k exactly).
- Macklin, Müller, Chentanez, Kim, *Small Steps in Physics Simulation*, SCA 2019 —
  https://mmacklin.com/smallsteps.pdf (many substeps × 1 iteration).
- Macklin, Müller, Chentanez, Kim, *Unified Particle Physics for Real-Time Applications*, SIGGRAPH 2014
  (averaged-Jacobi GPU parallelism; NVIDIA FleX).
- Müller, Chentanez, Kim, Macklin, *Strain Based Dynamics*, SCA 2015 —
  https://matthias-research.github.io/pages/publications/strainBasedDynamics.pdf.
- Macklin, Müller, *A Constraint-based Formulation of Stable Neo-Hookean Materials*, MIG 2021 — the
  per-tet constraint-pair FRAMEWORK behind §4 [Strain]. **[audit rev-B]** The ADOPTED pair is the
  production reference's rest-zero `tr(FᵀF)−3` + `det(F)−1` (XPBDSolver.compute:219/311); Macklin's own
  `{√(tr(FᵀF)), det(F)−(1+μ/λ)}` is the documented self-consistent ALTERNATIVE — never mix the two.
- **User's production solver** (authorized reference): `Simulation/Assets/SurgicalSim/Shaders/
  XPBDSolver.compute` (per-tet SNH:252-332, barycentric contact:1027-1039, tangential contact
  damping:1056-1065), `Physics/XPBDSolverGPU.cs` (ω=0.2:450, `ToolContactCouplingPasses=2`:508-513,
  N=10), `Grasping/GripperTool.cs` (hard core-pin `gaussianHardCoreRadius=0.55` + dynamic transition
  ring). This is the primary structural reference for §3.5/§4 [Strain]/§5.
- **github.com/FantasyVR/neohookean_XPBD** — validates the SNH constraint ALGEBRA only. **[audit] Note:
  it uses JACOBI + `maxIte=10` + NO substeps and `C=det(F)−1`, i.e. the OPPOSITE loop structure to this
  design's many-substeps × few-iterations and a different rest offset — so anchor the loop-structure
  justification to Macklin Small-Steps 2019 + the production `Simulation` solver, NOT to FantasyVR.**
- Bender, Müller, Macklin, *A Survey on Position Based Dynamics*, EG 2017 tutorial.
- iMSTK PbdModel docs (distance+volume vs NeoHookean for volumetric tissue) —
  https://imstk.gitlab.io/Dynamical_Models/PbdModel.html.
- Hybrid MSD-into-PBD surgical variant, PMC5830759 — https://pmc.ncbi.nlm.nih.gov/articles/PMC5830759/.
