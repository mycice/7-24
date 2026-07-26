# Collision/Contact Redesign — Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Eliminate the initial-penetration / liver-sticks-to-jaws / framerate-collapse bug by retiring the tissue-side position dent alongside the hard `k_SolveJawShaftContact` non-penetration constraint (both deliberate deviations from what the paper's SS2.3/SS3.2 sections describe, not fidelity improvements — see Task 1's audit-fix note) in favor of the paper's bounded Eq27 instrument-force feedback plus a narrow, velocity-gated CCD tunnelling gate, making the two-jaw grasp (core-pin + slab) compliant and breakable instead of kinematic/rigid, and gating the expensive K=2 elastic↔contact coupling pass to genuine grasp only.

**Architecture:** Six coordinated native (CUDA) changes plus two Unity C# changes, all inside the existing STEP 8 module boundary (`cuda_plugin/cuda/xpbd/xpbd_solver.{h,cu}`, `cuda_plugin/cuda/haptics.cu`, `Assets/ReconGridDC/Cuda/{LiverCudaManager.cs,GrasperRig.cs}`). No new files, no new modules — every change is a targeted edit inside the module that already owns the concern. `haptics.cu` receives its first-ever edit this migration (previously frozen byte-for-byte); every other file has already been edited in earlier steps.

**Tech Stack:** CUDA C++ (device kernels, `xpbd_solver.cu`/`haptics.cu`), C++ host glue (`physics.cu`, `plugin_api.cpp` — untouched this pass), C# Unity (`LiverCudaManager.cs`, `GrasperRig.cs`), headless CUDA self-test harness (`cuda_plugin/tests/xpbd_livepath_test.cu`, built via the existing `cuda-build-loop` CMake flow).

## Global Constraints

- Supersedes the step-8 contact design in `docs/superpowers/specs/2026-07-05-xpbd-combination-round3-design.md` §4[Contact]. The spec of record for THIS plan is `docs/superpowers/specs/2026-07-07-collision-contact-redesign.md` — every task below implements one numbered item from it; do not deviate from its Option A direction.
- Decisions already confirmed by the user (2026-07-07): (1) build the CCD anti-tunnelling gate now, not deferred; (2) auto-clearance init pose (not a hand-placed fixed offset); (3) compliant + breakable core-pin (not a kinematic weld with an explicit release key); (4) K=2 coupling gates on **grasp only**, not on "tool armed."
- `xp.omega = 1.0` stays fixed for the structural/bending/tet constraint set (unchanged, validated since steps 1-3). ALL new grasp/contact tunables (native `XpbdGraspParams` fields: `omegaGrasp`, `ccdOmega`, `ccdVelThresh`, `ccdMaxDisp`, `coreAlpha`, `coreBreakForce`, `graspMaxDisp`) live in `XpbdGraspParams`, never in `XpbdParams` — matches the existing precedent that `XpbdGraspParams` is the C#-tunable home for grasp/contact knobs (`mu`, `contactAlpha`, `contactSkin`, …). NOTE the naming distinction: the NATIVE struct fields above are flat, already-scaled values; the C# side additionally exposes `ccdMaxDispOverL`/`graspMaxDispOverL`/`graspContactSkinOverL` Inspector fields (this codebase's established `...OverL` convention for L-relative tunables), each multiplied by `Lrt` before being written into its native counterpart at the `gp`-construction call site — `graspContactSkinOverL` in particular feeds the PRE-EXISTING native `contactSkin` field, not a new native field of its own.
- Every new native tunable ships with a documented **STARTING ANCHOR** value and an explicit "tune against the video" note — this codebase's established convention (see `E=3e4`/`nu=0.45` in `physics.cu:557-564`), never a silently-invented "final" number.
- No destructive git operations, no force-push, no commits without being asked (standing constraint for this whole session).
- Real-scene scale is `L≈0.4` (voxel spacing); the headless self-test lattice uses `L` from its own fixture (currently ~0.2-1.0 depending on the gate) — every new gate must pass at the FIXTURE's own `L`, not assume the real scene's `L=0.4`.
- `haptics.cu`-only and Unity C#-only changes have **no** headless coverage (the `xpbd_livepath_test` harness links `dc_recon.cu + physics.cu + cut.cu + xpbd_solver.cu`, never `haptics.cu`, and never the C# `GrasperRig`/`LiverCudaManager` layer). Those tasks are verified by an explicit **Unity manual verification checklist** step instead of an automated test — do not invent a fake headless test for them.
- After all 9 tasks land: re-run the FULL existing suite (`xpbd_livepath_test` 30 gates + `xpbd_selftest` 58 checks + 6 oracle tests) and confirm zero regressions, per this project's audit-round convention for every prior step.
- **Stability Measure E ("fix ρ and mass=ρ·L³ at real L first; XPBD compliance and τ·L then follow") is explicitly OUT OF SCOPE for this plan** — noted here (audit fix, fourth re-audit) so it isn't a silent omission the way Measures A-D are each named and traced through Tasks 2/3/4/8. Per-corner mass is a flat, uniform `1.0` everywhere in this codebase (`BackgroundGrid.cs:302`, real scene AND every test fixture alike, "PAPER-SILENT: uniform mass," pre-dating this redesign by several migration steps) — rearchitecting that into a real `ρ·L³` model is a separate, substantially larger change outside this plan's stated Goal. This plan does NOT commit Measure E's specifically-warned-against error (a naive `1/L²` alpha rescale) — every new compliance/cap in Tasks 2-4 is a flat STARTING-ANCHOR constant (matching this codebase's existing convention for `contactAlpha`/`coreAlpha`) or an explicit `*L`-scaled displacement cap (dimensionally a length, not a compliance), never a rescaled stiffness. Revisiting the mass model, if ever undertaken, is future work outside this plan.

---

### Task 1: Retire the tissue-side position dent in `haptics.cu` (first-ever edit this migration)

**Files:**
- Modify: `cuda_plugin/cuda/haptics.cu:280-351` (kernels `k_ScatterContact`, `k_ApplyContactDisp`)
- Modify: `cuda_plugin/cuda/haptics.cu:358-379` (`haptics_init` — drop the `d_contactDisp` allocation)
- Modify: `cuda_plugin/cuda/haptics.cu` haptic-step dispatch block (the `k_ApplyContactDisp` launch — locate via `grep -n "k_ApplyContactDisp" cuda_plugin/cuda/haptics.cu`, currently the dispatch site around line 686-694 per the redesign spec's root-cause table)
- No test file — see the "Unity manual verification" step below (haptics.cu has zero headless coverage by design; see Global Constraints).

**Interfaces:**
- Consumes: nothing new — `k_ScatterContact`'s existing `hapticForce` parameter (a `float3*` already aliased to the shared `physics_haptic_force()` buffer, unchanged signature for that half).
- Produces: `k_ScatterContact`'s new signature (drops `contactDisp`, `dentDepthMax`, `contactBeta` params — see Step 3). Later tasks do not depend on anything from this task; it is a pure retirement.

- [ ] **Step 1: Read the current kernel bodies to confirm exact text before editing**

Run: `grep -n "k_ScatterContact\|k_ApplyContactDisp\|d_contactDisp" cuda_plugin/cuda/haptics.cu`

Expected: matches at (approximately) lines 280 (`k_ScatterContact` definition), 284 (`float3* contactDisp` param), 326-329 (the `contactDisp` atomicAdd triplet), 335 (`k_ApplyContactDisp` definition), 369/375 (`d_contactDisp` cudaMalloc/cudaMemset in `haptics_init`), and one dispatch call inside `haptics_step` (search separately — see Step 4).

- [ ] **Step 2: Delete the `contactDisp` accumulation from `k_ScatterContact`, keep only the force scatter**

The kernel currently ends with (from the file as read this session):

```cpp
    // (a) depth-weighted thresholded reaction force (paper's atomic-add rule).
    float w = (depthSum > 1e-9f) ? (depth / depthSum) : 0.f;
    float3 fShare = reaction * (w / 3.f);
    atomicAdd(&hapticForce[best].x, fShare.x);
    atomicAdd(&hapticForce[best].y, fShare.y);
    atomicAdd(&hapticForce[best].z, fShare.z);

    // (b) position update: the paper's SS3.2 BOUNDED non-penetration constraint (update particle
    // positions to PREVENT penetration), NOT an unbounded plastic crater (BUG-A, diagnosis wf_6703c058).
    // depth = radius - signedDist grows PAST the radius once the physical axis is inside, so scaling the
    // dent by the raw depth punched a single-frame hole that, with recovery killed + soft Ks, carved a
    // growing crater far deeper/wider than the thin jaw. Clamp the DENT depth to the tool radius so the
    // per-frame push is bounded and the recon-rebuild feedback converges the surface to the tool
    // penetration WITHOUT overshoot. (The FORCE weighting w above keeps the true depth — force is
    // separately capped by reactionMax.)
    float dentDepth = fminf(depth, dentDepthMax);
    float3 dShare = nOut * (-contactBeta * dentDepth / 3.f);
    atomicAdd(&contactDisp[best].x, dShare.x);
    atomicAdd(&contactDisp[best].y, dShare.y);
    atomicAdd(&contactDisp[best].z, dShare.z);
}
```

Replace the WHOLE kernel (signature + body) with:

```cpp
__global__ void k_ScatterContact(int triCount, const float4* triVms, const float3* soup,
                                 float3 reaction, float depthSum, float proximityMax,
                                 const float3* cornerPos, const int* active, const int* pinned,
                                 int cornerCount, float3* hapticForce)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;       // one thread per (triangle, vertex-lane)
    int t = i / 3, lane = i % 3;
    if (t >= triCount) return;
    float4 vt = triVms[t];
    if (vt.w >= proximityMax) return;                    // proximity gate (only near faces contribute)
    float3 vms = make_float3(vt.x, vt.y, vt.z);
    float depth = length3(vms);                           // depth = |vms| (the key is the selection dist)

    float3 v = soup[3 * t + lane];
    int best = -1; float bd = 1e30f;
    for (int cptr = 0; cptr < cornerCount; cptr++) {
        // Skip frozen/severed AND pinned corners (D11 "nearest active corner", unchanged from before —
        // this is the FORCE routing rule only; the position-dent half this comment used to also justify
        // is retired below, redesign 2026-07-07 Option A).
        if (active[cptr] == 0 || pinned[cptr] != 0) continue;
        float3 d = cornerPos[cptr] - v; float dd = dot3(d, d);
        if (dd < bd) { bd = dd; best = cptr; }
    }
    if (best < 0) return;

    // Depth-weighted thresholded reaction force (paper's Eq27 atomic-add rule). This is now the ONLY
    // tissue-side effect of contact — no position write. AUDIT FIX (BLOCKING finding, paper-fidelity
    // angle): an earlier draft of this comment claimed retiring the position dent makes this "FORCE-ONLY
    // per SS2.3/Eq27," implying full paper fidelity — that overstates it. Eq27 (F=Kp*dP-Dp*v) is the
    // paper's formula for "the force feedback applied to the SURGICAL INSTRUMENT" only (the haptic-device
    // reaction); separately, the paper's SS3.2 explicitly states a reaction force threshold "is... applied
    // to update the PARTICLE POSITIONS, ensuring smooth collision forces without penetration" — a
    // TISSUE-side position update the paper itself specifies, distinct from Eq27's instrument-force role.
    // This deletion is a DELIBERATE DEVIATION from that SS3.2 mechanic (the redesign spec's root-cause
    // item 4 identifies it as a bug amplifier stacked on top of k_SolveJawShaftContact, not something the
    // paper says is wrong) — flagged here explicitly, the same way the CCD gate/compliant core-pin/
    // K-regating are flagged as redesign-only additions elsewhere in this plan, rather than misrepresented
    // as a fidelity improvement. See docs/superpowers/specs/2026-07-07-collision-contact-redesign.md
    // root-cause item 4 for the full justification.
    float w = (depthSum > 1e-9f) ? (depth / depthSum) : 0.f;
    float3 fShare = reaction * (w / 3.f);
    atomicAdd(&hapticForce[best].x, fShare.x);
    atomicAdd(&hapticForce[best].y, fShare.y);
    atomicAdd(&hapticForce[best].z, fShare.z);
}
```

- [ ] **Step 3: Delete the `k_ApplyContactDisp` kernel entirely**

Delete the whole kernel (the comment block above it and the body):

```cpp
// Apply the accumulated contact displacement to the LIVE corner positions, clamped to
// contactDispMax per frame; damp the velocity component along the push so springs do not
// slingshot the patch back ("ensuring smooth collision forces", SS3.2 line 986-987).
__global__ void k_ApplyContactDisp(int cornerCount, const float3* contactDisp, float dispMax,
                                   const int* pinned, float3* cornerPos, float3* vel)
{
    ...
}
```

Nothing later in the file may reference `k_ApplyContactDisp` after this deletion — verify with `grep -n k_ApplyContactDisp cuda_plugin/cuda/haptics.cu` returning ONLY the dispatch call site (removed in Step 4 next) or nothing.

- [ ] **Step 4: Remove the `d_contactDisp` buffer (alloc/memset in `haptics_init`) and the `k_ApplyContactDisp` dispatch in `haptics_step`**

In `haptics_init` (around line 369/375), delete:

```cpp
    HK(cudaMalloc(&d_contactDisp, (size_t)cornerCount * sizeof(float3)));
```
and
```cpp
    HK(cudaMemset(d_contactDisp, 0, (size_t)cornerCount * sizeof(float3)));
```

Find the module-scope declaration (`grep -n "d_contactDisp" cuda_plugin/cuda/haptics.cu` will show it, likely near the top with the other `static float3* d_...` buffers) and delete that declaration line too.

Find the `haptics_step` dispatch block (`grep -n "k_ApplyContactDisp\|k_ScatterContact<<<" cuda_plugin/cuda/haptics.cu`). The `k_ScatterContact<<<...>>>` launch call must drop the `contactBeta`, `dentDepthMax`, `d_contactDisp` arguments to match the new signature from Step 2. Delete the separate `k_ApplyContactDisp<<<...>>>` launch line entirely (it has no successor kernel to call instead — its job is gone).

Also delete any host-side variable computing `dentDepthMax` (search `grep -n dentDepthMax cuda_plugin/cuda/haptics.cu` — likely `g_hp.hapticToolRadius`-derived or similar local in `haptics_step`) if it becomes unused after this edit.

**AUDIT FIX (third re-audit): update the two comment blocks that describe the retired dent as current, which the earlier draft of this task never mentioned.** `grep -n "k_ApplyContactDisp\|UPDATE THE PARTICLE POSITIONS" cuda_plugin/cuda/haptics.cu` will additionally surface two PURE-COMMENT locations this task's edits above do not touch:
- The file-header "Paper mapping" table (around line 11) that documents `k_ApplyContactDisp`/the position-dent as part of the module's current design.
- The Pass-1/Pass-2 explanation directly above `k_CountPenetrating` (around line 262) that describes "(b) POSITION: atomicAdd disp = contactBeta * depth_t * (-nOut_t) / 3 into a displacement accumulator (applied clamped by k_ApplyContactDisp — the visible dent)" as current behavior.

Update both to describe the RETIRED mechanism in the past tense (e.g. "Pass (b) POSITION was retired 2026-07-07 — see k_ScatterContact's own comment for why; this module now only ever does (a) FORCE") rather than leaving them describing deleted code as live. This is a documentation-accuracy fix only — Step 5's `grep`-based verification below should show ZERO hits for `k_ApplyContactDisp`/`contactDisp` describing CURRENT behavior once this is done (a hit describing the RETIRED mechanism in past tense, e.g. inside this very edit's own explanatory text, is fine and expected).

- [ ] **Step 5: Confirm it builds**

Run (from repo root, using the project's established build loop — see `[[cuda-build-loop]]` memory): rebuild `xpbd_livepath_test` (links `haptics.cu`? — confirm via `grep -n haptics cuda_plugin/tests/CMakeLists.txt` or the relevant `CMakeLists.txt`; if `haptics.cu` is NOT in that target's sources, instead do a syntax-only compile check via `nvcc -c cuda_plugin/cuda/haptics.cu -o /tmp/haptics_check.o <same include flags as the real target>` to catch typos without needing a full link).

Expected: clean compile, no undefined-reference errors, no leftover references to `contactDisp`/`dentDepthMax`/`k_ApplyContactDisp`.

- [ ] **Step 6: Unity manual verification (haptics.cu has no headless coverage)**

Open the scene in Play mode. Move the haptic tool (IJKL/UO) into light contact with the liver surface WITHOUT grasping (G not held). Confirm:
- The liver surface no longer visibly dents/craters at the contact point (the position-write path is gone).
- `_haptics.Force` (visible in the HUD, per `LiverCudaManager.cs`'s existing HUD text) still rises smoothly as you press harder, and falls back toward zero on release — the FORCE feedback path (Task 1 kept this) still works.
- No NaN/exploding positions in the console.

This step has no pass/fail automation — record the observation in the task's completion note (e.g. "confirmed: no dent, force still responds") before moving to Task 2.

- [ ] **Step 7: Commit**

```bash
git add cuda_plugin/cuda/haptics.cu
git commit -m "fix(haptics): retire tissue-side position dent, keep force-only Finger-Proxy contact"
```

---

### Task 2: CCD-gated speculative contact (replaces `k_SolveJawShaftContact`'s hard-ejection semantics)

**Files:**
- Modify: `cuda_plugin/cuda/xpbd/xpbd_solver.h` — add 7 fields to `XpbdGraspParams` (after `contactSkin`, before `nExemptRings`): the 6 CCD/omegaGrasp/core fields this task's Step 1 introduces, PLUS `graspMaxDisp` (the audit-fix Stability-Measure-D cap, threaded through Tasks 3/4's core-pin and slab-STICK kernels)
- Modify: `cuda_plugin/cuda/xpbd/xpbd_solver.cu:511-560` (rename+rewrite `k_SolveJawShaftContact` → `k_SolveCCDContact`)
- Modify: `cuda_plugin/cuda/xpbd/xpbd_solver.cu:110` (`g_gp` default initializer — append 7 new defaults)
- Modify: `cuda_plugin/cuda/xpbd/xpbd_solver.cu:1406-1410` (call site — updated fully in Task 5, this task only fixes the kernel name/args at the CURRENT call site so the file keeps compiling standalone)
- Modify: `Assets/ReconGridDC/Cuda/LiverCudaManager.cs:258-262` (`XpbdGraspParamsInterop` — mirror the 7 new fields) and `:127-142` (new `[Header]` Inspector fields) and `:461-467` (`gp` construction)
- Test: `cuda_plugin/tests/xpbd_livepath_test.cu` (new gates, added at end of this task)

**Interfaces:**
- Consumes: `d_vel` (already a module-scope static `float3*`, aliased at bind time — no new parameter plumbing needed at the module boundary, just pass it into the kernel call).
- Produces: `k_SolveCCDContact(int N, float ccdOmega, float ccdVelThresh, float ccdMaxDisp, float contactAlpha, float3 shaftA, float3 shaftB, float shaftR, float3 hinge, float3 axis, float3 normal, float3 w, float jawLen, float jawHalfWidth, float gapHalf, const int* grasped, const float* invMass, const float3* vel, float3* pos, float* jawLambda, float* shaftLambda)` — later tasks (5) reposition its ONE call site outside the K-loop; no other task calls it directly.

- [ ] **Step 1: Add the 7 new `XpbdGraspParams` fields**

In `cuda_plugin/cuda/xpbd/xpbd_solver.h`, the struct currently reads (per this session's read of the file):

```cpp
struct XpbdGraspParams
{
    float graspThreshold, releaseThreshold;  // capture/release hysteresis (design §5 item 1; e.g. 0.5/0.3)
    float jawGapClosed;                      // §11 item 5 floor: min slab half-gap, never fully collapses
    float coreEllipseFrac;                   // normalized-ellipse core fraction (0.55 start, §5)
    float mu;                                // Coulomb friction coefficient (ring tangential stick/slip)
    float contactAlpha;                      // small NON-ZERO contact/slab compliance (~1e-7, /h^2 like
                                              // the elastic terms -- design §4[Contact])
    float contactSkin;                       // capture-band widening for [Contact] (design §4[Contact])
    int   nExemptRings;                      // graspRing BFS dilation depth (1-2, design §6)
};
```

Add 7 new fields (float; keep `nExemptRings` LAST since it's the lone `int` — mixing int/float order in a `StructLayout(Sequential)`-marshalled struct is fine as long as C++ and C# list them in the SAME order, so append the new floats before `nExemptRings` to keep the int last, simplest to eyeball):

```cpp
struct XpbdGraspParams
{
    float graspThreshold, releaseThreshold;  // capture/release hysteresis (design §5 item 1; e.g. 0.5/0.3)
    float jawGapClosed;                      // §11 item 5 floor: min slab half-gap, never fully collapses
    float coreEllipseFrac;                   // normalized-ellipse core fraction (0.55 start, §5)
    float mu;                                // Coulomb friction coefficient (ring tangential stick/slip)
    float contactAlpha;                      // small NON-ZERO contact/slab compliance (~1e-7, /h^2 like
                                              // the elastic terms -- design §4[Contact])
    float contactSkin;                       // capture-band widening for [Contact] (design §4[Contact])
    // ── redesign 2026-07-07 (collision-contact-redesign.md, Option A) ──────────────────────────────
    float ccdVelThresh;   // corner speed (world units/s) TOWARD the tool above which the CCD gate fires;
                           // a statically-embedded corner (near-zero approach velocity) gets ZERO
                           // correction by design (rule a) -- STARTING ANCHOR 0.5, paper-silent, tune
                           // against the video's fast-plunge test.
    float ccdMaxDisp;      // absolute per-substep correction cap in world units (rule b) -- caller
                           // computes this as a fraction of L (design range 0.1-0.25*L); STARTING
                           // ANCHOR passed in already-scaled (native module has no L of its own).
    float ccdOmega;        // CCD gate's OWN under-relaxation (rule c), independent of xp.omega and
                            // omegaGrasp -- STARTING ANCHOR 0.25 (design's own explicit bound "<=0.25").
    float omegaGrasp;      // separate SOR under-relaxation for the core-pin + slab/Coulomb constraint
                            // set (redesign: under-relax the SOFT grasp block, not the stiff structural
                            // one) -- STARTING ANCHOR 0.25.
    float coreAlpha;       // core-pin XPBD compliance (small non-zero, same near-rigid magnitude as
                            // contactAlpha) -- STARTING ANCHOR 1.0e-7.
    float coreBreakForce;  // |coreLambda|/h^2 threshold (Newtons) above which a core corner detaches
                            // PERMANENTLY for this grasp session -- STARTING ANCHOR 400, tune against
                            // the video's hard-pull-releases feel.
    float graspMaxDisp;    // audit fix (spec Stability Measure D names BOTH grasp and CCD constraints,
                            // not CCD alone): absolute per-substep correction cap in world units for
                            // k_SolveCorePin AND k_SolveSlabCoulomb's STICK branch -- caller computes
                            // this as a fraction of L, same convention as ccdMaxDisp. STARTING ANCHOR
                            // passed in already-scaled; without this, a sudden large elastic imbalance
                            // on a grasped corner could move it more than 0.1-0.25*L in one frame before
                            // omegaGrasp/coreAlpha alone bring it back down, or before the core-pin's
                            // break threshold has a chance to fire.
    int   nExemptRings;    // graspRing BFS dilation depth (1-2, design §6)
};
```

- [ ] **Step 2: Update the `g_gp` default initializer in `xpbd_solver.cu`**

Current (line 110):
```cpp
static XpbdGraspParams g_gp = { 0.5f, 0.3f, 0.02f, 0.55f, 0.3f, 1.0e-7f, 0.02f, 1 };
```

Replace with (appending the 7 new defaults in struct order, `nExemptRings=1` stays last):
```cpp
static XpbdGraspParams g_gp = { 0.5f, 0.3f, 0.02f, 0.55f, 0.3f, 1.0e-7f, 0.02f,
                                /*ccdVelThresh*/0.5f, /*ccdMaxDisp*/0.045f, /*ccdOmega*/0.25f,
                                /*omegaGrasp*/0.25f, /*coreAlpha*/1.0e-7f, /*coreBreakForce*/400.f,
                                /*graspMaxDisp*/0.045f, /*nExemptRings*/1 };
```
(`0.045` — AUDIT FIX, fourth re-audit: was `0.06`, a coefficient of `0.30*L` at gate CCD1's fixture `L=0.2` — outside the spec's `0.1-0.25*L` range for rule (b)/Measure D, and CCD1 relies on this DEFAULT since it never calls `xpbd_set_grasp_params` explicitly. `0.045` sits in-range at both `L=0.2` (`0.225*L`) and the real scene's `L≈0.4` (`0.1125*L`), so no test's implicit reliance on this default ships an out-of-spec value regardless of which fixture scale exercises it.)

(`ccdMaxDisp` default of 0.06 assumes a standalone-test-scale `L≈0.2-0.4`; the C# caller overrides this every frame via `xpbd_set_grasp_params`, so this default only matters for tests that skip calling it — document that assumption inline if any new gate relies on the default rather than setting it explicitly.)

- [ ] **Step 3: Rename and rewrite `k_SolveJawShaftContact` → `k_SolveCCDContact` with the 4 CCD rules**

The current kernel (lines 505-560, this session's read):

```cpp
// CONTACT (design §4[Contact]): per-corner non-penetration vs the shaft CAPSULE (Issue D, always) and
// the jaw SLAB used as a contact surface (uncaptured corners only -- grasped corners are the SLAB
// constraint's job, k_SolveSlabCoulomb below; this is the ownership-matrix "OWNS everything" / "SKIP for
// the owning tool's jaws" split, design §5 item 4). Single-corner constraints -- no colouring needed
// (each corner only ever writes its OWN position). w=0 (pinned/inactive/core) corners are automatic
// no-ops via the wc<=0 early-out, matching every other constraint's w-rule.
__global__ void k_SolveJawShaftContact(int N, float omega, float contactAlpha,
                                       float3 shaftA, float3 shaftB, float shaftR,
                                       float3 hinge, float3 axis, float3 normal, float3 w,
                                       float jawLen, float jawHalfWidth, float gapHalf,
                                       const int* grasped, const float* invMass, float3* pos,
                                       float* jawLambda, float* shaftLambda)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= N) return;
    float wc = invMass[c];
    if (wc <= 0.f) return;

    // -- shaft capsule (Issue D; closest-point-on-segment, reimplemented independently of haptics.cu) --
    {
        float3 ab = shaftB - shaftA;
        float ab2 = dot3(ab, ab);
        float tt = ab2 > 1e-12f ? dot3(pos[c] - shaftA, ab) / ab2 : 0.f;
        tt = fmaxf(0.f, fminf(1.f, tt));
        float3 closest = shaftA + ab * tt;
        float3 dvec = pos[c] - closest;
        float dist = len3(dvec);
        float C = dist - shaftR;
        if (C < 0.f) {
            float3 n = dist > 1e-9f ? dvec * (1.f / dist) : mkf3(0.f, 1.f, 0.f);
            float denom = wc + contactAlpha;
            float dLambda = (-C - contactAlpha * shaftLambda[c]) / denom;
            float applied = omega * dLambda;
            shaftLambda[c] += applied;
            pos[c] = pos[c] + n * (wc * applied);
        }
    }

    // -- jaw slab as a contact BOX (uncaptured corners caught inside the closing jaws only) --
    if (grasped[c] == 0) {
        float3 rel = pos[c] - hinge;
        float u = dot3(rel, axis), v = dot3(rel, w), n = dot3(rel, normal);
        if (u >= 0.f && u <= jawLen && fabsf(v) <= jawHalfWidth) {
            float absN = fabsf(n);
            if (absN > gapHalf) {
                float Ctrue = gapHalf - absN;                  // true constraint: gapHalf - |n| >= 0
                float sgn = (n >= 0.f) ? 1.f : -1.f;            // grad(Ctrue) w.r.t. n = -sgn
                float denom = wc + contactAlpha;
                float dLambda = (-Ctrue - contactAlpha * jawLambda[c]) / denom;
                float applied = omega * dLambda;
                jawLambda[c] += applied;
                pos[c] = pos[c] - normal * (sgn * wc * applied);
            }
        }
    }
}
```

Replace with:

```cpp
// CCD ANTI-TUNNELLING GATE (redesign 2026-07-07, Option A — supersedes the old hard
// k_SolveJawShaftContact non-penetration constraint). The paper's Finger-Proxy (haptics.cu) is the
// ONLY tissue non-penetration authority now — this kernel exists SOLELY to stop a fast-moving corner
// from tunnelling clean through the thin shaft/jaw geometry between substeps, not to eject anything
// that is merely sitting inside the tool. Four mandatory rules (design spec's own words):
//   (a) fires ONLY when the corner's OWN velocity component toward the tool exceeds ccdVelThresh —
//       a statically-embedded corner (near-zero approach speed) gets ZERO correction;
//   (b) caps the applied per-substep correction at ccdMaxDisp regardless of nominal penetration depth;
//   (c) uses its OWN ccdOmega, independent of xp.omega (structural/tet) and omegaGrasp (core/slab);
//   (d) early-outs on wc<=0 (pinned/inactive) for BOTH branches, and on grasped[c]!=0 for the JAW-SLAB
//       branch ONLY (not the shaft-capsule branch) — "never fights the compliant core-pin or a pinned
//       neighbour" (design spec's own words) applies specifically to the jaw-slab box test, which
//       operates in the SAME jaw-local coordinate frame k_SolveSlabCoulomb/k_SolveCorePin already own
//       (a genuine double-solve risk); the shaft capsule is a completely separate world-space geometric
//       primitive neither grasp kernel ever tests against, so there is no double-solve risk there to
//       avoid.
//
// AUDIT FIX (collision re-review, regression-vs-retired-constraint angle): an EARLIER audit round
// (see the superseded reasoning this replaces) excluded grasped corners from the shaft branch TOO,
// worried about the shaft push and the core-pin's pull "fighting" on the same corner — but that's
// ordinary, expected multi-constraint reconciliation (exactly what the K-loop's iteration is FOR), not
// a double-solve. The real, concrete cost of that over-broad exclusion was only found by a LATER,
// dedicated regression-vs-old-design audit: the OLD (retired) k_SolveJawShaftContact protected EVERY
// corner against the shaft, grasped or not; neither k_SolveCorePin nor k_SolveSlabCoulomb has ANY
// shaft-capsule awareness (verified: neither kernel's signature or body references shaftA/shaftB/
// shaftR anywhere) — so excluding grasped corners from the shaft branch left them with ZERO
// shaft-tunnelling protection, a confirmed regression relative to the design this plan replaces, not a
// narrowing. Restored: only the jaw-slab branch excludes grasped corners now, matching the ORIGINAL
// (pre-redesign) kernel's own actual behavior for the shaft branch, while still keeping the redesign's
// genuine improvement (velocity-gated, capped, own-omega) for that branch.
//
// AUDIT FIX (fourth re-audit round 2, lambda-vs-capped-position-consistency angle): rule (b)'s cap
// MUST scale lambda by the SAME factor as the position correction, not cap position alone while
// accumulating an uncapped lambda. This codebase's own load-bearing invariant (k_solve_edges's comment:
// "omega applied CONSISTENTLY to both lambda and position, so the compliant fixed point C=-alpha~*lambda
// stays omega-INVARIANT") requires this; capping only the position silently breaks that identity and
// makes shaftLambda/jawLambda read HIGHER than what was actually applied whenever the cap engages
// (every other constraint kernel in this file — edges, bending, tet, and k_SolveSlabCoulomb's STICK/
// gap-closing branches after this same audit fix — honors this identity without exception).
__global__ void k_SolveCCDContact(int N, float ccdOmega, float ccdVelThresh, float ccdMaxDisp, float contactAlpha,
                                  float3 shaftA, float3 shaftB, float shaftR,
                                  float3 hinge, float3 axis, float3 normal, float3 w,
                                  float jawLen, float jawHalfWidth, float gapHalf,
                                  const int* grasped, const float* invMass, const float3* vel, float3* pos,
                                  float* jawLambda, float* shaftLambda)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= N) return;
    float wc = invMass[c];
    if (wc <= 0.f) return;                                       // rule (d): pinned/inactive (both branches)

    // -- shaft capsule (closest-point-on-segment, reimplemented independently of haptics.cu). Runs for
    // EVERY corner including grasped ones (see AUDIT FIX above) -- this is the ONLY shaft-tunnelling
    // protection in the whole solver; neither grasp kernel checks the shaft at all. --
    {
        float3 ab = shaftB - shaftA;
        float ab2 = dot3(ab, ab);
        float tt = ab2 > 1e-12f ? dot3(pos[c] - shaftA, ab) / ab2 : 0.f;
        tt = fmaxf(0.f, fminf(1.f, tt));
        float3 closest = shaftA + ab * tt;
        float3 dvec = pos[c] - closest;
        float dist = len3(dvec);
        float C = dist - shaftR;
        if (C < 0.f) {
            float3 n = dist > 1e-9f ? dvec * (1.f / dist) : mkf3(0.f, 1.f, 0.f);
            float vApproach = -dot3(vel[c], n);                  // rule (a): speed moving INTO the tool
            if (vApproach > ccdVelThresh) {
                float denom = wc + contactAlpha;
                float dLambda = (-C - contactAlpha * shaftLambda[c]) / denom;
                float applied = ccdOmega * dLambda;               // rule (c)
                float3 corr = n * (wc * applied);
                float corrLen = len3(corr);
                // AUDIT FIX: scale BOTH applied (before lambda accumulation) and corr by the SAME
                // factor when the cap engages, instead of capping corr alone (rule b) while leaving
                // shaftLambda uncapped — keeps shaftLambda consistent with what was ACTUALLY applied.
                if (corrLen > ccdMaxDisp) {
                    float capScale = ccdMaxDisp / corrLen;
                    applied *= capScale;
                    corr = corr * capScale;
                }
                shaftLambda[c] += applied;
                pos[c] = pos[c] + corr;
            }
        }
    }

    // -- jaw slab as a contact BOX (uncaptured corners caught inside the closing jaws only -- this
    // branch's OWN `grasped[c]==0` check, NOT a function-level early-return, is what excludes grasped
    // corners here; the shaft branch above deliberately has no such exclusion, see the AUDIT FIX in the
    // kernel header) --
    if (grasped[c] == 0) {
        float3 rel = pos[c] - hinge;
        float u = dot3(rel, axis), v = dot3(rel, w), n = dot3(rel, normal);
        if (u >= 0.f && u <= jawLen && fabsf(v) <= jawHalfWidth) {
            float absN = fabsf(n);
            if (absN > gapHalf) {
                float Ctrue = gapHalf - absN;
                float sgn = (n >= 0.f) ? 1.f : -1.f;
                float3 nOut = normal * sgn;                       // outward direction for this branch
                float vApproach = -dot3(vel[c], nOut);            // rule (a)
                if (vApproach > ccdVelThresh) {
                    float denom = wc + contactAlpha;
                    float dLambda = (-Ctrue - contactAlpha * jawLambda[c]) / denom;
                    float applied = ccdOmega * dLambda;           // rule (c)
                    float corrMag = wc * applied;
                    // AUDIT FIX: same consistent-scaling fix as the shaft branch above.
                    if (fabsf(corrMag) > ccdMaxDisp) {
                        float capScale = ccdMaxDisp / fabsf(corrMag);
                        applied *= capScale;
                        corrMag *= capScale;
                    }
                    jawLambda[c] += applied;
                    pos[c] = pos[c] - normal * (sgn * corrMag);
                }
            }
        }
    }
}
```

**AUDIT FIX (collision-focused re-review, 5-agent round): known limitations of this kernel, deliberately NOT fixed by this plan — documented here so they are acknowledged rather than silently assumed away.**

1. **This is a discrete, per-substep POSITION check, not a true swept/continuous (TOI) test.** Both branches test only the corner's CURRENT (post-`k_predict`) position against the tool geometry, never its path from `prevPos` to `pos`. A corner whose straight-line motion carries it from clearly-outside on one side to clearly-outside on the other side within a single substep (a genuine full tunnel-through) is invisible to this check — `C` reads `>0` at both ends, the gate never fires, and the crossing leaves no trace. Worse, the velocity-gate (rule a)'s sign convention is derived from the corner's post-motion position, so even in the "near miss, ended up just barely on the far side" case, `vApproach` reads as receding rather than approaching, further suppressing detection right at the boundary of the full-tunnel case.
   - **Shaft-capsule branch: structurally real, but NOT reachable by anything currently in this codebase.** Requires a corner to move more than `2*shaftR` in one substep (`h=0.001s` at real scene scale) — roughly 400-1800 world-units/second depending on `shaftR`. Every real velocity source in this sim (gravity, elastic/tet solve, damping, even a core-pin break's transient force) tops out 100-450x below this. The tool geometry itself is also frozen for an entire frame's ~20 substeps (only `xpbd_set_tool`, once per frame, changes it), so tool speed cannot contribute swept motion at the substep granularity this kernel operates at either. A defensive hardening fix exists if ever wanted (a segment-vs-capsule closest-distance test using `prevPos[c]` in place of the current point-vs-capsule test — a well-scoped, single-kernel addition, not an architectural change) but is not urgent given the margin.
   - **Jaw-slab-box branch: the SAME structural gap, but reachable via a narrower, already-anomalous path.** The velocity threshold for tunnelling through the jaw's gap (tens to ~160 u/s, since the gap is much thinner than the shaft's diameter) is far closer to what a violent transient event could produce — specifically, a corner's velocity spike DURING the tension buildup that precedes a core-pin break (before `Fcore` crosses `coreBreakForce` and the break/detach fires). Two things already in this plan narrow this window without fully closing it: (a) the corner is excluded from this very branch (`grasped[c]==0`) for the ENTIRE tension-buildup phase, since it's still core-pinned; the jaw-box branch only ever sees it AFTER the break clears `grasped[c]`; (b) the just-added break-path velocity damping (`vel[c] *= 0.1f` in `k_SolveCorePin`, this same audit round) directly reduces the residual velocity the corner carries into its first post-break substep. Neither eliminates the gap outright (a large tangential/lateral velocity component orthogonal to the core-pin's own pull direction could still be present at the moment of detach), so this remains a real, if narrow and already-mitigated, edge case — not fixed further here, but flagged explicitly rather than silently assumed safe. See Task 9's added checklist item.
   - **Test coverage: the jaw-slab-box branch has ZERO coverage in this plan's test suite.** CCD1, CCD2, and KGATE all deliberately park `jawHinge` far away with `jawLen=jawHalfWidth=jawGap=0` specifically to make this branch inert while isolating the shaft-capsule branch. No gate in this plan exercises the jaw-box branch's own correctness (tunnelling or otherwise) at all. A dedicated future gate is a reasonable follow-up but is not added here, to keep this already-large plan's scope bounded.

2. **This kernel tests the PHYSICAL tool position, not the PROXY position.** `shaftA`/`shaftB`/the jaw geometry are all sourced (via `xpbd_set_tool`) from the SAME raw, keyboard/device-driven pose (`_hapticCenter`/`hs`/`he` in `LiverCudaManager.cs`) that feeds `haptics.cu`'s Finger-Proxy computation — NOT from `_haptics.ProxyPos` (the Algorithm-1-projected, already surface-clamped, bounded-step-size position the paper's own model guarantees is well-behaved). This is a deliberate, considered choice, not an oversight, and it IS necessary: the paper's proxy model provably constrains only the DISPLAYED proxy and the FORCE feedback (Eq 22/27) — it says nothing about the underlying simulated MESH corners this XPBD solver actually manipulates, which only ever feel a soft, finite-stiffness reaction force (`k_ScatterContact`) that cannot categorically prevent a corner's predicted position from tunnelling through thin geometry between substeps. Something has to guard the mesh corners directly, and nothing else in this architecture does. However, testing against the physical (not proxy) tool DOES reintroduce, in a narrower and now velocity-gated form, exactly the "react to the raw tool's unsmoothed fast motion" exposure the proxy architecture's whole design intent is to avoid needing — a proxy-fed version of this check would have received an already-bounded-step-size input, likely shrinking or removing the need for rules (a)-(c)'s machinery entirely. This plan does NOT switch to proxy-based testing (it would require new C#-to-native plumbing for `ProxyPos`, and the existing tunables were derived assuming the current physical-tool input) — documented here as a considered-but-not-chosen alternative, not a defect to silently paper over. Relatedly, `ccdVelThresh`'s STARTING ANCHOR (0.5 u/s) sits BELOW ordinary keyboard drive speed (`hapticMoveSpeed=4` u/s) — meaning rule (a)'s gate is live for essentially any deliberate approach, not just rare fast plunges, which is worth knowing when reasoning about how "narrow" this gate actually is in practice (see Task 9's checklist).

3. **A corner that becomes deeply embedded via SLOW means (not a fast approach) has no mechanism to ever be pushed back out.** This is the design's own explicit, stated philosophy (rule a exists specifically so a merely-embedded, near-zero-velocity corner gets zero correction, unlike the OLD unconditional hard constraint) — not an oversight. But this plan's test suite (Gate CCD1) only verifies the STATIC case stays put; it never separately verifies/documents what happens to a corner that drifts into embedding gradually (e.g., via elastic settling, or a captured-then-released corner that ends up geometrically overlapping the shaft) and then simply stays there indefinitely with no recovery path. See Task 9's added checklist item.

4. **A moving tool dragging a corner caught in its geometry converges to a small, BOUNDED steady-state lag (not unbounded/compounding drag), and takes only 2-3 frames to settle from a deep initial embedding — both confirmed quantitatively, not just asserted.** At real-scene defaults (`hapticMoveSpeed=4`, `ccdOmega=0.25`, 20 substeps/frame), the steady-state lag is roughly 0.2-0.26·L (about half a shaft radius at most) and does not grow frame-over-frame; `ccdOmega`'s specific value barely matters here since 20 substeps is far more convergence budget than needed. This is NOT a meaningful recurrence of the original bug's severity (which was an unconditional, unbounded drag) — noted here only so this specific quantitative question, once raised, has a documented answer rather than being left as an open worry.

- [ ] **Step 4: Fix the ONE existing call site so the file keeps compiling (final position/gating fixed in Task 5)**

Current call (line 1406-1410):
```cpp
        if (g_bound && g_toolArmed) {   // STEP 8: contact + grasp, projected LAST in this coupling pass
            k_SolveJawShaftContact<<<Gr(g_N), 64>>>(g_N, g_p.omega, g_gp.contactAlpha,
                g_tool.shaftA, g_tool.shaftB, g_tool.shaftR,
                g_tool.jawHinge, jawAxis3, jawNormal3, g_jawW, g_tool.jawLen, g_tool.jawHalfWidth, gapHalf,
                d_grasped, d_invMass, d_pos, d_jawContactLambda, d_shaftContactLambda);
```

Replace ONLY this call (leave `k_SolveSlabCoulomb`'s call directly below it untouched — Task 4 handles that one) with:
```cpp
        if (g_bound && g_toolArmed) {   // CCD gate stays gated on toolArmed (fires regardless of grasp)
            k_SolveCCDContact<<<Gr(g_N), 64>>>(g_N, g_gp.ccdOmega, g_gp.ccdVelThresh, g_gp.ccdMaxDisp, g_gp.contactAlpha,
                g_tool.shaftA, g_tool.shaftB, g_tool.shaftR,
                g_tool.jawHinge, jawAxis3, jawNormal3, g_jawW, g_tool.jawLen, g_tool.jawHalfWidth, gapHalf,
                d_grasped, d_invMass, d_vel, d_pos, d_jawContactLambda, d_shaftContactLambda);
```

(This still runs INSIDE the K-loop for now — Task 5 moves it outside. Doing the rename/rewrite here keeps this task's diff reviewable in isolation; Task 5's diff is then a pure repositioning, not tangled with the rule rewrite.)

- [ ] **Step 5: Add the C# mirror fields**

First update the stale size comment directly above the struct (line 242-243):
```csharp
        // STEP 8 (design rev-B §9-(8)): field order MUST match cuda/xpbd/xpbd_solver.h XpbdToolState
        // (6 float3 + 4 float, flattened -- 92 bytes) / XpbdGraspParams (7 float + 1 int -- 32 bytes).
```
becomes:
```csharp
        // STEP 8 (design rev-B §9-(8)) + redesign 2026-07-07: field order MUST match cuda/xpbd/xpbd_solver.h
        // XpbdToolState (6 float3 + 4 float, flattened -- 92 bytes) / XpbdGraspParams (14 float + 1 int
        // -- 60 bytes, grown from 7 float + 1 int by the redesign's 7 new ccd*/omegaGrasp/core*/graspMaxDisp
        // fields). NOTE: unlike XpbdToolStateInterop, XpbdGraspParams has NO native-side ABI size guard (no
        // LCS_GraspParamsAbiVersion export) — an order mismatch here silently corrupts values rather
        // than failing loud. Double-check this order against xpbd_solver.h byte-for-byte.
```

Then, in `Assets/ReconGridDC/Cuda/LiverCudaManager.cs`, the interop struct (lines 257-262):
```csharp
        [StructLayout(LayoutKind.Sequential)]
        struct XpbdGraspParamsInterop
        {
            public float graspThreshold, releaseThreshold, jawGapClosed, coreEllipseFrac, mu, contactAlpha, contactSkin;
            public int nExemptRings;
        }
```
Replace with:
```csharp
        [StructLayout(LayoutKind.Sequential)]
        struct XpbdGraspParamsInterop
        {
            public float graspThreshold, releaseThreshold, jawGapClosed, coreEllipseFrac, mu, contactAlpha, contactSkin;
            public float ccdVelThresh, ccdMaxDisp, ccdOmega, omegaGrasp, coreAlpha, coreBreakForce, graspMaxDisp;
            public int nExemptRings;
        }
```

Add new Inspector fields near the existing STEP 8 header (after `graspNExemptRings`, line 142):
```csharp
        // redesign 2026-07-07 (collision-contact-redesign.md): CCD anti-tunnelling gate + compliant
        // grasp tunables. All STARTING ANCHORS — tune against the video, none are paper-specified.
        public float ccdVelThresh   = 0.5f;    // world units/s; corner approach speed floor (rule a)
        public float ccdMaxDispOverL = 0.15f;  // per-frame cap = this * L (rule b, design range 0.1-0.25)
        public float ccdOmega       = 0.25f;   // CCD gate's own SOR under-relaxation (rule c)
        public float omegaGrasp     = 0.25f;   // core-pin + slab/Coulomb under-relaxation (soft grasp block)
        public float coreAlpha      = 1.0e-7f; // core-pin XPBD compliance (near-rigid, same order as contactAlpha)
        // AUDIT FIX (fourth re-audit round 2, real-scene-scale angle): was 400, an UN-DERIVED guess made
        // before the test gate's rigorous empirical derivation existed. The real scene's h/coreAlpha/
        // omegaGrasp are IDENTICAL to the test fixture's (physicsDt=0.02, maxSubstepsPerFrame=20 -> same
        // h=0.001 as the test's DT/NSUB), so the SAME derivation transfers directly: ordinary hold
        // settles to Fcore~=9.8-13N, a genuine hard pull settles to Fcore roughly equal to the applied
        // load. 400 sits ~31x above ordinary hold (vs. the test-validated 150's ~11-15x margin) -- not
        // dangerously low, but un-validated and needlessly close to the "too rigid to ever feel
        // breakable" failure mode round 1 of this same audit already found at 50000. Retuned to match
        // the test-validated value directly.
        public float coreBreakForce = 150f;    // Newtons; sustained overload above this detaches the core corner
        // audit fix (spec Stability Measure D names BOTH grasp and CCD constraints, not CCD alone):
        // per-frame displacement cap for the core-pin AND slab STICK branch, same convention as
        // ccdMaxDispOverL. Without this, spec Measure D ("incremental correction, not total... 0.1-0.25*L")
        // was only half-implemented (CCD side only).
        public float graspMaxDispOverL = 0.15f;
        // audit fix (spec Grasp layer item B: "widen the degenerate quarter-voxel jaw box; /L-scale the
        // skins"): Task 7 fixes the jaw box; this fixes the sibling capture-band skin, which was a flat
        // 0.02 constant with no L-awareness (same degenerate-at-real-scale problem the jaw box had).
        // Replaces the old flat `graspContactSkin` field below at the gp-construction call site.
        public float graspContactSkinOverL = 0.05f;
```

Locate the existing flat `graspContactSkin` field (current source, line 140: `public float graspContactSkin = 0.02f;`) and mark it superseded rather than deleting it outright (avoids an unrelated scene-serialization break for existing scenes that already set a value on this field):
```csharp
        // SUPERSEDED by graspContactSkinOverL (redesign 2026-07-07, /L-scale fix) — kept for scene
        // compatibility only; the gp-construction call site below no longer reads this field directly.
        public float graspContactSkin      = 0.02f;
```

Update the `gp` construction (lines 461-467):
```csharp
            var gp = new XpbdGraspParamsInterop
            {
                graspThreshold = graspCaptureThreshold, releaseThreshold = graspReleaseThreshold,
                jawGapClosed = graspJawGapClosed, coreEllipseFrac = graspCoreEllipseFrac, mu = graspMu,
                contactAlpha = graspContactAlpha, contactSkin = graspContactSkin, nExemptRings = graspNExemptRings
            };
```
becomes:
```csharp
            var gp = new XpbdGraspParamsInterop
            {
                graspThreshold = graspCaptureThreshold, releaseThreshold = graspReleaseThreshold,
                jawGapClosed = graspJawGapClosed, coreEllipseFrac = graspCoreEllipseFrac, mu = graspMu,
                contactAlpha = graspContactAlpha, contactSkin = graspContactSkinOverL * Lrt,
                ccdVelThresh = ccdVelThresh, ccdMaxDisp = ccdMaxDispOverL * Lrt, ccdOmega = ccdOmega,
                omegaGrasp = omegaGrasp, coreAlpha = coreAlpha, coreBreakForce = coreBreakForce,
                graspMaxDisp = graspMaxDispOverL * Lrt,
                nExemptRings = graspNExemptRings
            };
```

- [ ] **Step 6: Write the failing headless gates for the CCD rules**

Append to `cuda_plugin/tests/xpbd_livepath_test.cu` (after the existing gate I, following the file's `g8_gate_*` naming convention):

```cpp
// ── STEP-8-REDESIGN gate CCD1 (2026-07-07, audit-fixed): a corner sitting STATICALLY embedded in the
// shaft (zero approach velocity) must receive ZERO correction from k_SolveCCDContact — this is the rule
// that replaces the old hard k_SolveJawShaftContact's unconditional ejection, and is the direct fix for
// the "liver sticks to the jaws at init" bug. Construct a shaft that already overlaps a corner, hold the
// lattice at rest (velocity settles near zero under damping) for a few frames, then confirm the
// overlapping corner's position barely moves once settled — NOT ejected outward. AUDIT FIX (major
// finding, test-discriminating-power angle): the settle window was 60 frames; this project's own
// established precedent for this exact fixture's alpha=2 Rayleigh damping (g8_gate_pinch_hold's comment,
// "this lattice's alpha=2 Rayleigh damping is SLOW... a short window catches a transient oscillation
// upswing rather than the genuine monotonic sag trend") uses 150 frames for the analogous settle
// requirement — bumped to match, so a still-oscillating residual velocity doesn't make this gate flaky
// or unable to distinguish "genuinely near-zero" from "still ringing down."
static int g8_gate_ccd_static_embed(const Lattice& g) {
    int fails = 0;
    if (initLivePath(g) != 0) return 1;

    // let the lattice settle under gravity first so residual velocity is small and legitimate — this
    // gate is about EMBEDDED-AND-STILL, not "never moved at all." 150 frames (not 60), matching this
    // fixture's own established slow-damping settle precedent (see comment above).
    for (int f = 0; f < 150; f++) if (physics_step(DT) != 0) { printf("  FAIL  CCD1 settle physics_step\n"); physics_shutdown(); recon_shutdown(); return ++fails; }

    int probeC = cid(CX/2, CY/2, CZ/2);   // an interior corner, away from pinned boundary
    std::vector<float> p0(3 * CC); recon_readback_corner_pos(p0.data());
    float3 probePos = make_float3(p0[3*probeC], p0[3*probeC+1], p0[3*probeC+2]);

    XpbdToolState ts = {};
    ts.shaftA = probePos - make_float3(0.f, 0.f, 5.f * L);     // shaft passes right through probeC
    ts.shaftB = probePos + make_float3(0.f, 0.f, 5.f * L);
    ts.shaftR = 2.f * L;                                        // generous radius -- probeC starts deep inside
    ts.jawAxis = make_float3(1.f, 0.f, 0.f); ts.jawNormal = make_float3(0.f, 1.f, 0.f);
    ts.jawHinge = make_float3(-1000.f, -1000.f, -1000.f);       // jaw box parked far away -- shaft-only gate
    ts.jawLen = 0.f; ts.jawHalfWidth = 0.f; ts.jawGap = 0.f;
    ts.grasp01 = 0.f;                                           // below capture threshold -- no grasp involved
    if (physics_set_tool(&ts) != 0) { printf("  FAIL  CCD1 physics_set_tool\n"); physics_shutdown(); recon_shutdown(); return ++fails; }

    for (int f = 0; f < 30; f++) if (physics_step(DT) != 0) { printf("  FAIL  CCD1 physics_step\n"); fails++; physics_shutdown(); recon_shutdown(); return fails; }
    std::vector<float> p1(3 * CC); recon_readback_corner_pos(p1.data());
    // AUDIT FIX (fourth re-audit, CUDA-symbol-correctness angle): was `length(...)` -- undefined here.
    // xpbd_livepath_test.cu (confirmed via its actual #include list: dc_recon.h, physics.h, cut.h,
    // xpbd_solver.h, cuda_runtime.h, cstdio, cmath, vector, algorithm) never includes common.cuh (the
    // only place a length3(float3) helper lives) and has NO local len3/length3/length helper of its own
    // either -- xpbd_solver.cu's `len3` is a SEPARATE translation unit's local function, not visible
    // here. Compute the magnitude manually with sqrtf (already available via <cmath>, already included)
    // instead of inventing a dependency this file doesn't have.
    float3 dMove = make_float3(p1[3*probeC]-p0[3*probeC], p1[3*probeC+1]-p0[3*probeC+1], p1[3*probeC+2]-p0[3*probeC+2]);
    float moved = sqrtf(dMove.x*dMove.x + dMove.y*dMove.y + dMove.z*dMove.z);
    printf("  info  CCD1 static-embed: probe corner moved %.5f over 30 frames (shaft R=%.3f, embedded)\n", moved, ts.shaftR);
    // A hard-ejection constraint would push this corner out by close to shaftR (>= 0.5*shaftR) within a
    // handful of frames. The CCD gate (near-zero approach velocity) must leave it near its settled rest
    // position -- some elastic/gravity drift is fine, ejection-scale motion is not.
    if (!(moved < 0.5f * ts.shaftR))
        { printf("  FAIL  CCD1 STEP8-REDESIGN: static-embedded corner moved %.5f, expected < 0.5*shaftR=%.5f (looks like a hard-ejection regression)\n", moved, 0.5f*ts.shaftR); fails++; }

    physics_shutdown(); recon_shutdown();
    return fails;
}

// ── STEP-8-REDESIGN gate CCD2 (2026-07-07, audit-fixed TWICE): a FAST-moving corner approaching the
// shaft DOES get a correction from k_SolveCCDContact, and it is capped at ccdMaxDisp — never a
// single-frame teleport regardless of how deep the (synthetic) penetration is set.
//
// AUDIT FIX ROUND 1 (BLOCKING, test-discriminating-power angle): the original version measured raw 3D
// displacement magnitude over one frame and asserted it landed in [1e-5, capBudget] -- satisfied by
// GRAVITY'S OWN FALL ALONE, independent of the CCD kernel. Round 1 offset the shaft axis 0.5L along X
// (perpendicular to gravity) so the correction direction was well-defined and isolated from gravity's
// own Y displacement.
//
// AUDIT FIX ROUND 2 (BLOCKING, second re-audit): round 1's X-offset broke the mechanism it tests.
// k_SolveCCDContact's velocity gate (rule a) checks vApproach=-dot3(vel[c],n) where n is the RADIAL
// direction -- with the round-1 X-offset, n=+X by construction. But this fixture's lattice is fully
// X/Z-symmetric (only the TOP LAYER is pinned; gravity acts purely in -Y) and the probe corner
// cid(CX/2,CY/2,CZ/2) sits at the exact X/Z center, so vel.x never grows above float noise in EITHER
// run -- vApproach never crosses ccdVelThresh regardless of its value, and k_SolveCCDContact's shaft
// branch never fires in EITHER run. Isolating the correction's DISPLACEMENT from gravity also isolated
// the velocity-gate's TRIGGER from gravity -- and gravity is the only real motion source this fixture
// has. Net effect: the gate would FAIL a CORRECT implementation (false negative), worse than the
// original vacuous pass.
//
// FIX (round 2): stop fighting gravity -- use it as the deliberate, physically-meaningful approach-
// velocity source (exactly how a real falling/pressing tissue corner triggers this gate in production).
// Offset the shaft axis 0.5L along -Y (not X), so the corner sits ABOVE the axis with a clean,
// deterministic radial direction n=+Y (an explicit offset, not CCD1's degenerate dist~=0 fallback, so a
// tiny float perturbation can't flip which branch computes n). As gravity pulls the corner down it
// falls FURTHER into the shaft -- vel.y grows increasingly negative, so vApproach=-vel.y grows positive
// and genuinely "approaches," crossing ccdVelThresh within 1-2 substeps. The CONTROL/CANDIDATE
// DIFFERENTIAL (not axis orthogonality) is what isolates the CCD kernel's own contribution: both runs
// experience the IDENTICAL gravity pull and elastic coupling (no RNG anywhere in this deterministic
// sim), so any DIFFERENCE in how far the corner falls between the two runs is attributable ONLY to
// whether the CCD kernel's correction (pushing back along +Y, opposing the fall) was live -- the same
// principle as this file's own e5_onset()-style "two pristine full re-inits, differenced" pattern
// (round-3 audit precedent), just packed into one function instead of two calls from main().
static int g8_gate_ccd_fast_approach(const Lattice& g) {
    int fails = 0;
    int probeC = cid(CX/2, CY/2, CZ/2);
    float3 axisOffset = make_float3(0.f, 0.5f * L, 0.f);   // shaft axis 0.5L BELOW probeC in Y
    float shaftR = 2.f * L;                                 // > 0.5L -- probeC starts embedded, radial dist=0.5L
    // AUDIT FIX (fourth re-audit, fresh spec-pass angle): was 0.05*L -- BELOW the spec's own stated
    // 0.1-0.25*L range for rule (b)'s cap (spec CCD rule b + Stability Measure D). 0.15*L matches the
    // C# production default's coefficient (ccdMaxDispOverL=0.15) and sits centered in-range regardless
    // of which fixture L this gate runs at.
    float ccdMaxDisp = 0.15f * L;
    int N = 20;   // fixture's numSubsteps -- see initLivePath / xpbd_set_params in this file

    // ── CONTROL: ccdVelThresh huge -- k_SolveCCDContact's position-writing branches are provably inert
    // this run, so the corner's Y-displacement is pure gravity+elastic free-fall (uncorrected baseline). ──
    if (initLivePath(g) != 0) return 1;
    std::vector<float> pc0(3 * CC); recon_readback_corner_pos(pc0.data());
    float3 probePos0 = make_float3(pc0[3*probeC], pc0[3*probeC+1], pc0[3*probeC+2]);
    XpbdToolState tsCtrl = {};
    tsCtrl.shaftA = probePos0 - axisOffset - make_float3(5.f * L, 0.f, 0.f);
    tsCtrl.shaftB = probePos0 - axisOffset + make_float3(5.f * L, 0.f, 0.f);
    tsCtrl.shaftR = shaftR;
    tsCtrl.jawAxis = make_float3(1.f, 0.f, 0.f); tsCtrl.jawNormal = make_float3(0.f, 1.f, 0.f);
    tsCtrl.jawHinge = make_float3(-1000.f, -1000.f, -1000.f);
    tsCtrl.jawLen = 0.f; tsCtrl.jawHalfWidth = 0.f; tsCtrl.jawGap = 0.f;
    tsCtrl.grasp01 = 0.f;
    XpbdGraspParams gpCtrl = { 0.5f, 0.3f, 0.02f, 0.55f, 0.3f, 1.0e-7f, 0.02f,
                              /*ccdVelThresh*/1.0e6f /* gate can never fire this run */,
                              /*ccdMaxDisp*/ccdMaxDisp, 0.25f, 0.25f, 1.0e-7f, 400.f, /*graspMaxDisp*/ccdMaxDisp, 1 };
    xpbd_set_grasp_params(&gpCtrl);
    if (physics_set_tool(&tsCtrl) != 0) { printf("  FAIL  CCD2 control physics_set_tool\n"); physics_shutdown(); recon_shutdown(); return ++fails; }
    if (physics_step(DT) != 0) { printf("  FAIL  CCD2 control physics_step\n"); physics_shutdown(); recon_shutdown(); return ++fails; }
    std::vector<float> pc1(3 * CC); recon_readback_corner_pos(pc1.data());
    float ctrlDy = pc1[3*probeC+1] - pc0[3*probeC+1];   // Y is where BOTH gravity and the correction act
    physics_shutdown(); recon_shutdown();

    // ── CANDIDATE: ccdVelThresh tiny (0.005 -- comfortably below the ~0.00981 u/s a single substep of
    // gravity already imparts at this fixture's h=DT/20, so the gate is live from substep 1 or 2 onward,
    // not waiting on a marginal threshold crossing). The correction opposes the fall (pushes +Y).
    // Identical setup to the control run above otherwise. ──
    if (initLivePath(g) != 0) return 1;
    std::vector<float> pd0(3 * CC); recon_readback_corner_pos(pd0.data());
    float3 probePos1 = make_float3(pd0[3*probeC], pd0[3*probeC+1], pd0[3*probeC+2]);
    XpbdToolState tsCand = {};
    tsCand.shaftA = probePos1 - axisOffset - make_float3(5.f * L, 0.f, 0.f);
    tsCand.shaftB = probePos1 - axisOffset + make_float3(5.f * L, 0.f, 0.f);
    tsCand.shaftR = shaftR;
    tsCand.jawAxis = make_float3(1.f, 0.f, 0.f); tsCand.jawNormal = make_float3(0.f, 1.f, 0.f);
    tsCand.jawHinge = make_float3(-1000.f, -1000.f, -1000.f);
    tsCand.jawLen = 0.f; tsCand.jawHalfWidth = 0.f; tsCand.jawGap = 0.f;
    tsCand.grasp01 = 0.f;
    XpbdGraspParams gpCand = { 0.5f, 0.3f, 0.02f, 0.55f, 0.3f, 1.0e-7f, 0.02f,
                              /*ccdVelThresh*/0.005f,
                              /*ccdMaxDisp*/ccdMaxDisp, 0.25f, 0.25f, 1.0e-7f, 400.f, /*graspMaxDisp*/ccdMaxDisp, 1 };
    xpbd_set_grasp_params(&gpCand);
    if (physics_set_tool(&tsCand) != 0) { printf("  FAIL  CCD2 candidate physics_set_tool\n"); physics_shutdown(); recon_shutdown(); return ++fails; }
    if (physics_step(DT) != 0) { printf("  FAIL  CCD2 candidate physics_step\n"); physics_shutdown(); recon_shutdown(); return ++fails; }
    std::vector<float> pd1(3 * CC); recon_readback_corner_pos(pd1.data());
    float candDy = pd1[3*probeC+1] - pd0[3*probeC+1];
    physics_shutdown(); recon_shutdown();

    // The CCD correction pushes the corner back along +Y (opposing the fall), so the candidate must end
    // up MEANINGFULLY HIGHER than the control's uncorrected free-fall -- diff = candDy - ctrlDy > 0 and
    // appreciable -- while the difference (the CCD-attributable part only, common gravity/elastic motion
    // cancels out of this subtraction) stays within the per-substep cap budget over the whole frame.
    float diff = candDy - ctrlDy;
    float capBudget = (float)N * ccdMaxDisp * 1.5f;   // generous multi-substep envelope, not a tight bound
    printf("  info  CCD2 fast-approach: control dY=%.6f, candidate dY=%.6f, diff=%.6f (ccdMaxDisp=%.5f, budget=%.5f)\n",
           ctrlDy, candDy, diff, ccdMaxDisp, capBudget);
    if (!(diff > 1e-5f))
        { printf("  FAIL  CCD2 STEP8-REDESIGN: candidate not meaningfully higher than control's free-fall (diff=%.6f) -- CCD gate looks dead (vacuous-pass regression)\n", diff); fails++; }
    // ... but still respects the per-substep cap (rule b) over the whole frame.
    if (!(diff < capBudget))
        { printf("  FAIL  CCD2 STEP8-REDESIGN: correction diff %.6f exceeded the per-substep cap budget %.6f (rule b broken)\n", diff, capBudget); fails++; }

    return fails;
}
```

Add both to `main()`, following the file's actual registration idiom exactly (there is NO macro — `main()` inlines each gate as a `{ int x = gate_fn(g); g_fail += x; printf(...); }` block, e.g. the existing STEP 8 gates at lines 759-769):
```cpp
    {
        int cf = g8_gate_ccd_static_embed(g);
        g_fail += cf;
        printf("  info  CCD1 STEP8-REDESIGN static-embed gate: %s (%d failure%s)\n", cf == 0 ? "PASS" : "FAIL", cf, cf == 1 ? "" : "s");
    }
    {
        int cf = g8_gate_ccd_fast_approach(g);
        g_fail += cf;
        printf("  info  CCD2 STEP8-REDESIGN fast-approach gate: %s (%d failure%s)\n", cf == 0 ? "PASS" : "FAIL", cf, cf == 1 ? "" : "s");
    }
```
Insert these right after the existing `g8_gate_grasp_ring_integration` block (before the final `printf("== %s ...")` summary line at the end of `main()`).

- [ ] **Step 7: Build and run — CCD1/CCD2 may not both fully settle until Task 5 repositions the K-gating (a BUILD failure is still blocking; a K-timing-related test failure is not, yet)**

Run: rebuild `xpbd_livepath_test` per `[[cuda-build-loop]]`; run it.

Expected at THIS point in the plan (before Task 5): CCD1 likely already passes (the velocity gate logic is live from Step 3-4), but treat any failure here as expected-until-Task-5 IF it stems from K-loop-vs-outside-loop timing, not from the rule logic itself — re-verify after Task 5 lands. Do not treat a Task-2-time failure as blocking if Task 5's repositioning is the documented fix; DO treat a build error as blocking. **This hedge carries forward through Task 3's checkpoint too — see Task 3 Step 8's note.**

- [ ] **Step 8: Commit**

```bash
git add cuda_plugin/cuda/xpbd/xpbd_solver.h cuda_plugin/cuda/xpbd/xpbd_solver.cu \
        Assets/ReconGridDC/Cuda/LiverCudaManager.cs cuda_plugin/tests/xpbd_livepath_test.cu
git commit -m "feat(xpbd): CCD-gated speculative contact replaces hard jaw/shaft ejection"
```

---

### Task 3: Compliant, breakable core-pin

**Files:**
- Modify: `cuda_plugin/cuda/xpbd/xpbd_solver.cu:185-210` (`k_RefreshLiveness` — drop `graspCore` from the zero-invMass rule)
- Modify: `cuda_plugin/cuda/xpbd/xpbd_solver.cu:488-503` (delete `k_ApplyCorePin`'s kinematic body, replace with the new compliant `k_SolveCorePin`)
- Modify: `cuda_plugin/cuda/xpbd/xpbd_solver.cu` (module-scope buffer declarations, ~line 100) — add `d_coreLambda`
- Modify: `cuda_plugin/cuda/xpbd/xpbd_solver.cu` (`xpbd_bind`'s tail, where `d_slabLambda` etc. are allocated) — allocate/zero `d_coreLambda`
- Modify: `cuda_plugin/cuda/xpbd/xpbd_solver.cu:1358-1365` (delete the `k_ApplyCorePin` pre-step dispatch)
- Modify: `cuda_plugin/cuda/xpbd/xpbd_solver.cu:1372-1376` (lambda-reset block — add `d_coreLambda` reset)
- Test: `cuda_plugin/tests/xpbd_livepath_test.cu` — update Gate H2's exact-tracking assertion + add a new break-on-overpull gate

**Interfaces:**
- Consumes: `d_coreOffU/N/W` (existing, unchanged — capture-time jaw-local anchor), `omegaGrasp`/`coreAlpha`/`coreBreakForce`/`graspMaxDisp` from `XpbdGraspParams` (added in Task 2 Step 1).
- Produces: `k_SolveCorePin(int N, float omegaGrasp, float coreAlpha, float coreBreakForce, float graspMaxDisp, float h, float3 hinge, float3 axis, float3 normal, float3 w, const float* offU, const float* offN, const float* offW, const float* invMass, float3* pos, float* coreLambda, int* graspCore, int* grasped)` — called from inside the K-loop (Task 5 finalizes exact placement, but this task's Step 5 already puts it in a reasonable temporary spot so the file compiles standalone).

- [ ] **Step 1: Remove `graspCore` from the zero-invMass rule in `k_RefreshLiveness`**

Current (line 204, this session's read):
```cpp
        invMass[t] = (pinned[t] != 0 || active[t] == 0 || graspCore[t] != 0) ? 0.f : (1.f / mass[t]);
```
Replace with:
```cpp
        // redesign 2026-07-07: a core-pinned corner is no longer kinematically frozen — it keeps its
        // NORMAL invMass and is pulled toward the jaw's target by the compliant k_SolveCorePin
        // constraint instead (so it can be overpowered and break, per the design's "compliant +
        // breakable core-pin" decision). Only pinned/inactive corners get invMass=0 now.
        invMass[t] = (pinned[t] != 0 || active[t] == 0) ? 0.f : (1.f / mass[t]);
```

- [ ] **Step 2: Add the `d_coreLambda` buffer**

Near the other STEP 8 buffer declarations (module scope, alongside `d_slabLambda` etc.):
```cpp
static float*  d_coreLambda       = nullptr;  // [N] core-pin XPBD multiplier (reset /substep, like d_slabLambda)
```

In `xpbd_bind`'s tail where the other grasp buffers are allocated/zeroed (find via `grep -n "d_slabLambda.*cudaMalloc\|d_slabLambda.*cudaMemset" cuda_plugin/cuda/xpbd/xpbd_solver.cu`), add the matching pair:
```cpp
    XK(cudaMalloc(&d_coreLambda, (size_t)cc * sizeof(float)));
    XK(cudaMemset(d_coreLambda, 0, (size_t)cc * sizeof(float)));
```

Also free it in `xpbd_shutdown` alongside the other grasp buffers (find `cudaFree(d_slabLambda)` and add `cudaFree(d_coreLambda); d_coreLambda = nullptr;` next to it).

- [ ] **Step 3: Replace `k_ApplyCorePin`'s kinematic body with the compliant `k_SolveCorePin`**

Current (lines 488-503):
```cpp
// CORE PIN (design §5 item core lifecycle): each substep, force core corners to the CURRENT jaw pose's
// stored local offset -- KINEMATIC, not solved (invMass=0 for these corners is already enforced by
// k_RefreshLiveness's "one w-rule", so the constraint sweeps below never move them; this kernel is what
// actually MAKES them ride the jaw, since predict skips w=0 corners entirely and would otherwise leave
// them frozen at last-substep's position forever). Runs ONCE per substep, right after predict/refresh,
// so the elastic/tet/contact sweeps that follow see the correct current-frame target.
__global__ void k_ApplyCorePin(int N, float3 hinge, float3 axis, float3 normal, float3 w,
                               const int* graspCore, const float* offU, const float* offN, const float* offW,
                               float3* pos, float3* prevPos, float3* vel)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= N) return;
    if (graspCore[c] == 0) return;
    float3 target = hinge + axis * offU[c] + normal * offN[c] + w * offW[c];
    pos[c] = target; prevPos[c] = target; vel[c] = mkf3(0.f, 0.f, 0.f);
}
```

Replace with:
```cpp
// CORE PIN (redesign 2026-07-07): compliant, BREAKABLE XPBD distance-to-target constraint — replaces
// the old kinematic weld (pos=target;vel=0;invMass=0). Runs INSIDE the K coupling sweep (alongside
// k_SolveSlabCoulomb, own omegaGrasp), so it reconciles with the elastic solve instead of overriding
// it outright. On sustained overload (|coreLambda|/h^2 > coreBreakForce) the corner detaches
// PERMANENTLY for this grasp session — "a hard pull cleanly detaches the corner instead of dragging
// the whole liver" (redesign spec). Detaching clears BOTH graspCore and grasped (a broken core corner
// leaves the grasp entirely, not merely demoted to a slipping ring corner). AUDIT FIX (major finding,
// spec-conformance angle): spec Stability Measure D ("incremental correction, not total... cap the
// total displacement applied to a corner within a frame at 0.1-0.25*L") names BOTH the grasp AND CCD
// constraints, but the first draft of this kernel only bounded magnitude via omegaGrasp/coreAlpha, with
// no explicit cap comparable to k_SolveCCDContact's ccdMaxDisp -- added graspMaxDisp, capped the SAME
// way k_SolveCCDContact caps its correction.
//
// AUDIT FIX (fourth re-audit round 2, lambda-vs-capped-position-consistency angle): the graspMaxDisp cap
// MUST scale coreLambda by the SAME factor as the position correction — capping corrMag alone while
// accumulating an uncapped `applied` into coreLambda breaks this codebase's own load-bearing invariant
// (k_solve_edges's comment: "omega applied CONSISTENTLY to both lambda and position, so the compliant
// fixed point C=-alpha~*lambda stays omega-INVARIANT"). Left uncapped-lambda, coreLambda would read
// HIGHER than what was actually applied whenever the cap engages (most likely right after capture, or
// during the initial substeps of a hard pull — exactly the transients coreBreakForce's own derivation
// needs to be accurate about), risking the break threshold firing earlier than the empirically-derived
// margin assumes. Scale both consistently instead.
//
// AUDIT FIX (collision re-review, grasp-transition-boundary angle): on break, damp `vel[c]` the SAME
// way the whole-grasp release path already does (`k_ReleaseDampVelocity`, x0.1 — "so the spring-back
// starts clean"). The original draft only cleared `graspCore`/`grasped` here with NO velocity handling
// at all — a real, confirmed asymmetry: a corner detaching via overload (this path) is arguably the
// HIGHER-energy case (it broke specifically because it was under enough tension to cross
// `coreBreakForce`), yet it was the ONE release path with no damping. Since `k_SolveCCDContact`
// dispatches once AFTER the K-loop in the SAME substep (Task 5's final order) and gates on
// `grasped[c]==0` (which this break just set), an undamped high-tension corner is IMMEDIATELY eligible
// for CCD's velocity-gated correction in that same substep — its stale (one-substep-old) high approach
// velocity could plausibly cross `ccdVelThresh` and trigger an extra, uncushioned CCD push right as it
// detaches, reading as a visible "snap -> immediately caught and shoved" micro-jerk instead of the clean
// "snap -> settle" the whole-grasp path was deliberately engineered to produce. Requires a new `vel`
// parameter (not previously needed by this kernel).
__global__ void k_SolveCorePin(int N, float omegaGrasp, float coreAlpha, float coreBreakForce, float graspMaxDisp, float h,
                               float3 hinge, float3 axis, float3 normal, float3 w,
                               const float* offU, const float* offN, const float* offW,
                               const float* invMass, float3* pos, float3* vel, float* coreLambda,
                               int* graspCore, int* grasped)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= N) return;
    if (graspCore[c] == 0) return;
    float wc = invMass[c];
    if (wc <= 0.f) return;
    float3 target = hinge + axis * offU[c] + normal * offN[c] + w * offW[c];
    float3 d = pos[c] - target;
    float dist = len3(d);
    if (dist > 1e-9f) {
        float3 n = d * (1.f / dist);
        float C = dist;                                         // constraint: pos == target (C=0 at rest)
        float denom = wc + coreAlpha;
        float dLambda = (-C - coreAlpha * coreLambda[c]) / denom;
        float applied = omegaGrasp * dLambda;
        float corrMag = wc * applied;
        // AUDIT FIX: scale BOTH applied (before lambda accumulation) and corrMag by the SAME factor
        // when the cap engages (see the kernel-header comment above) — never cap corrMag alone.
        if (fabsf(corrMag) > graspMaxDisp) {
            float capScale = graspMaxDisp / fabsf(corrMag);
            applied *= capScale;
            corrMag *= capScale;
        }
        coreLambda[c] += applied;
        pos[c] = pos[c] + n * corrMag;
    }

    float Fcore = fabsf(coreLambda[c]) / (h * h);
    if (Fcore > coreBreakForce) {
        graspCore[c] = 0;
        grasped[c]   = 0;
        vel[c] = vel[c] * 0.1f;   // AUDIT FIX: match k_ReleaseDampVelocity's damping so THIS release path
                                  // starts clean too, closing the asymmetry described above.
    }
}
```

- [ ] **Step 4: Delete the `k_ApplyCorePin` pre-step dispatch (it is now solved inside the K-loop, not applied before it)**

Current (lines 1358-1365):
```cpp
    // STEP 8: force core corners to the CURRENT jaw target every substep (design §5) -- w=0 already
    // stops the solve from moving them, but predict itself SKIPS w=0 corners, so without this they would
    // stay frozen at whatever position they last held instead of riding the tool.
    if (g_bound && g_toolArmed)
        k_ApplyCorePin<<<Gr(g_N), 64>>>(g_N, g_tool.jawHinge,
            (len3(g_tool.jawAxis) > 1e-9f ? g_tool.jawAxis * (1.f/len3(g_tool.jawAxis)) : mkf3(0,0,1)),
            (len3(g_tool.jawNormal) > 1e-9f ? g_tool.jawNormal * (1.f/len3(g_tool.jawNormal)) : mkf3(0,1,0)),
            g_jawW, d_graspCore, d_coreOffU, d_coreOffN, d_coreOffW, d_pos, d_prevPos, d_vel);
```
Delete this whole block. (Core corners now have normal invMass per Step 1, so `predict` no longer skips them — they need no separate pre-step target-force; `k_SolveCorePin`, dispatched in Step 5 below, is their only mover.)

- [ ] **Step 5: Reset `d_coreLambda` each substep and dispatch `k_SolveCorePin` inside the K-loop (temporary placement — Task 5 finalizes exact gating)**

Current lambda-reset block (lines 1372-1376):
```cpp
    if (g_bound && g_toolArmed) {   // STEP 8: multipliers reset ONCE per substep, accumulate across iters
        cudaMemsetAsync(d_slabLambda,         0, (size_t)g_N * sizeof(float));         // AND all K passes
        cudaMemsetAsync(d_jawContactLambda,   0, (size_t)g_N * sizeof(float));
        cudaMemsetAsync(d_shaftContactLambda, 0, (size_t)g_N * sizeof(float));
    }
```
Add `d_coreLambda` to the same reset:
```cpp
    if (g_bound && g_toolArmed) {   // STEP 8: multipliers reset ONCE per substep, accumulate across iters
        cudaMemsetAsync(d_slabLambda,         0, (size_t)g_N * sizeof(float));         // AND all K passes
        cudaMemsetAsync(d_jawContactLambda,   0, (size_t)g_N * sizeof(float));
        cudaMemsetAsync(d_shaftContactLambda, 0, (size_t)g_N * sizeof(float));
        cudaMemsetAsync(d_coreLambda,         0, (size_t)g_N * sizeof(float));
    }
```

Immediately after the EXISTING `k_SolveSlabCoulomb` call inside the K-loop (itself dispatched right after the renamed, Task-2 CCD contact call — so real order is CCD → Slab → CorePin), add the core-pin dispatch (this is a TEMPORARY position — Task 5 Step 2 moves the gating condition from `g_toolArmed` to `g_grasping`; for now, keep it compiling alongside the existing `k_SolveSlabCoulomb` call):
```cpp
            k_SolveSlabCoulomb<<<Gr(g_N), 64>>>(g_N, g_p.omega, h, g_gp.contactAlpha, g_gp.mu,
                g_tool.jawHinge, jawAxis3, jawNormal3, g_jawW, g_tool.jawLen, g_tool.jawHalfWidth, gapHalf,
                d_grasped, d_graspCore, d_invMass, d_slabLambdaPrev, d_slabLambda,
                d_coulombAnchorU, d_coulombAnchorW, d_pos);
            k_SolveCorePin<<<Gr(g_N), 64>>>(g_N, g_gp.omegaGrasp, g_gp.coreAlpha, g_gp.coreBreakForce, g_gp.graspMaxDisp, h,
                g_tool.jawHinge, jawAxis3, jawNormal3, g_jawW, d_coreOffU, d_coreOffN, d_coreOffW,
                d_invMass, d_pos, d_vel, d_coreLambda, d_graspCore, d_grasped);
```
(`d_vel` — audit fix, collision re-review: this kernel's signature gained a `float3* vel` parameter above, needed for the break-path damping fix; pass the module's existing `d_vel` here, matching every other kernel in this substep that already reads/writes it.)

- [ ] **Step 6: Update Gate H2's exact-tracking assertion (a compliant pin no longer tracks EXACTLY within 20 frames) AND its now-stale explanatory comment**

**AUDIT FIX (minor finding, code-correctness angle):** the original plan only replaced the two assertion lines, leaving the PRECEDING explanatory comment block (real file, lines 356-361) naming the now-deleted `k_ApplyCorePin` kernel and describing its kinematic semantics ("target = hinge + axis*offU..., re-read from g_tool EVERY substep") — both now false once this task lands. Replace that comment block too, not just the assertions.

Current (real file, lines 356-361):
```cpp
    // ── H2 (audit round-1 fix, MAJOR finding): the sag comparison above cannot distinguish a working
    // k_ApplyCorePin kinematic re-target from a no-op side effect of the w-rule freezing the core at its
    // CAPTURE-time position forever (mutation-tested: a no-op ApplyCorePin passed the sag check
    // identically). MOVE the jaw and confirm the core corner actually TRACKS the new pose -- this is the
    // only way to observe k_ApplyCorePin's real write path (target = hinge + axis*offU + normal*offN +
    // w*offW, re-read from g_tool EVERY substep) rather than the coincidental freeze-in-place artifact. ──
```
Replace with:
```cpp
    // ── H2 (audit round-1 fix, MAJOR finding; updated 2026-07-07 for the compliant core-pin redesign):
    // the sag comparison above cannot distinguish a working core-pin re-target from a no-op side effect
    // of the w-rule freezing the core at its CAPTURE-time position forever. MOVE the jaw and confirm the
    // core corner actually TRACKS the new pose -- this is the only way to observe k_SolveCorePin's real
    // pull (target = hinge + axis*offU + normal*offN + w*offW, re-read from g_tool EVERY substep) rather
    // than the coincidental freeze-in-place artifact. NOTE: since the redesign replaced the old KINEMATIC
    // weld (pos=target exactly, every substep) with a COMPLIANT XPBD pull, tracking is now asymptotic,
    // not exact -- see the two-part progress/no-overshoot check below instead of an exact-match bound. ──
```

The existing gate (`cuda_plugin/tests/xpbd_livepath_test.cu`, `g8_gate_pinch_hold`, sub-check H2, lines ~356-383 this session's read) asserts:

The existing gate (`cuda_plugin/tests/xpbd_livepath_test.cu`, `g8_gate_pinch_hold`, sub-check H2, lines ~356-383 this session's read) asserts:
```cpp
        if (!(dxMoved > 0.8f * MOVE_DX))
            { printf("  FAIL  H2 STEP8 core-pin: core corner did NOT track the moved jaw (dxMoved=%.4f, want>%.4f)\n", dxMoved, 0.8f*MOVE_DX); fails++; }
```
This assumed the OLD kinematic weld (exact, no-lag tracking). With the new compliant pin, tracking is asymptotic — 20 frames may not reach 0.8x, but it must be trending toward the target and not stuck near 0. Replace with a two-part check: (a) real progress was made, (b) it's converging (not oscillating/diverging):
```cpp
        // redesign 2026-07-07: the core-pin is now COMPLIANT, not a kinematic weld — it will not track
        // the jaw EXACTLY within 20 frames the way the old pos=target override did. Assert it makes
        // REAL, monotonically-improving progress toward the new target instead of an exact-match bound.
        if (!(dxMoved > 0.2f * MOVE_DX))
            { printf("  FAIL  H2 STEP8-REDESIGN core-pin: core corner made no meaningful progress toward the moved jaw (dxMoved=%.4f, want>%.4f)\n", dxMoved, 0.2f*MOVE_DX); fails++; }
        if (!(dxMoved < 1.05f * MOVE_DX))
            { printf("  FAIL  H2 STEP8-REDESIGN core-pin: core corner OVERSHOT the moved jaw target (dxMoved=%.4f, jaw moved=%.4f) -- looks unstable\n", dxMoved, MOVE_DX); fails++; }
```

- [ ] **Step 7: Add a new gate — core-pin breaks under sustained overload**

Append after the updated Gate H:
```cpp
// ── STEP-8-REDESIGN gate CORE-BREAK (2026-07-07, audit-fixed): a core corner held against a LARGE,
// sustained external pull must detach (graspCore AND grasped both clear) once |coreLambda|/h^2 exceeds
// coreBreakForce — "a hard pull cleanly detaches the corner instead of dragging the whole liver." A
// light/ordinary hold (no extra external force, just gravity + normal grasp dynamics) must NOT detach.
//
// AUDIT FIX ROUND 1 (BLOCKING, test-discriminating-power angle): the original version used
// coreBreakForce=5 with NO negative control -- ordinary settling error alone could plausibly cross it,
// so the gate could not distinguish "breaks only under genuine overload" from "breaks almost immediately
// regardless of load." Round 1 added a negative-control run and raised coreBreakForce to 50000, based on
// a hand-estimated (not simulated) "~2500" worst-case ordinary-hold lambda.
//
// AUDIT FIX ROUND 2 (BLOCKING, second re-audit): round 1's "50000" overcorrected in the OPPOSITE
// direction. A direct simulation of the actual k_SolveCorePin recursion (predict -> K=2 passes of
// applied=omegaGrasp*(-C) -> k_update_velocity overwrite -> Rayleigh velScale damping, per substep) shows
// the constraint's STEADY-STATE reaction force equals the sustained applied load, not the hand-estimated
// "~2500": ordinary hold (gravity only) settles to Fcore~=9.81N (transient spike ~13N); the deliberate
// 500N synthetic pull settles to Fcore~=500N (transient spike ~653N, a consistent ~1.31x overshoot ratio
// reproduced across multiple force scales). coreBreakForce=50000 is ~77x higher than even the 500N pull's
// transient peak -- the positive-pull assertion would FAIL for a CORRECT implementation (never crosses
// the threshold), the mirror-image failure mode to round 1's "too low" bug.
//
// FIX (round 2): coreBreakForce=150 -- roughly 11x above the ~13N ordinary-hold transient peak (safe
// margin against a false detach during normal grasping) and roughly 3-4x below the 500N pull's ~500-653N
// settle/transient range (safe margin for the positive case to reliably cross it). The negative-control
// run below is what actually validates this margin empirically (via its own PASS/FAIL, not a blind
// guess) -- if a future change to omegaGrasp/coreAlpha/the K-pass count shifts the real hold-force
// materially, this gate fails LOUDLY rather than silently passing either direction.
static int g8_gate_core_break(const Lattice& g) {
    int fails = 0;
    const float CORE_BREAK_FORCE = 150.f;   // see derivation in the comment above

    // ── NEGATIVE CONTROL: ordinary hold, zero extra external force. The core corner must NOT detach --
    // this is what the original version never checked, and is exactly what a "breaks almost immediately
    // regardless of load" mutant would fail. ──
    if (initLivePath(g) != 0) return 1;
    XpbdToolState tsHold = {};
    tsHold.jawAxis = make_float3(1.f, 0.f, 0.f); tsHold.jawNormal = make_float3(0.f, 1.f, 0.f);
    tsHold.jawHinge = make_float3(0.f, L, 2.f * L);
    tsHold.jawLen = 2.5f * L; tsHold.jawHalfWidth = 1.5f * L; tsHold.jawGap = 0.4f * L;
    tsHold.shaftA = make_float3(-100.f, -100.f, -100.f); tsHold.shaftB = make_float3(-100.f, -99.f, -100.f); tsHold.shaftR = 0.01f;
    tsHold.grasp01 = 0.8f;
    XpbdGraspParams gpHold = { 0.5f, 0.3f, 0.02f, 0.55f, 0.3f, 1.0e-7f, 0.02f,
                              0.5f, 0.06f, 0.25f, 0.25f, 1.0e-7f, CORE_BREAK_FORCE, /*graspMaxDisp*/0.06f, 1 };
    xpbd_set_grasp_params(&gpHold);
    if (physics_set_tool(&tsHold) != 0) { printf("  FAIL  CORE-BREAK hold physics_set_tool\n"); physics_shutdown(); recon_shutdown(); return ++fails; }
    int coreCHold = -1;
    { std::vector<int> hCore(CC); cudaMemcpy(hCore.data(), xpbd_grasp_core_flags(), (size_t)CC*sizeof(int), cudaMemcpyDeviceToHost);
      for (int c = 0; c < CC; c++) if (hCore[c]) { coreCHold = c; break; } }
    if (coreCHold < 0) { printf("  FAIL  CORE-BREAK no core corner found (hold run)\n"); physics_shutdown(); recon_shutdown(); return ++fails; }
    for (int f = 0; f < 40; f++) if (physics_step(DT) != 0) { printf("  FAIL  CORE-BREAK hold physics_step\n"); fails++; break; }
    std::vector<int> hCoreHoldAfter(CC);
    cudaMemcpy(hCoreHoldAfter.data(), xpbd_grasp_core_flags(), (size_t)CC*sizeof(int), cudaMemcpyDeviceToHost);
    printf("  info  CORE-BREAK negative control: after 40 ordinary-hold frames, core[%d]=%d (want 1)\n", coreCHold, hCoreHoldAfter[coreCHold]);
    if (!(hCoreHoldAfter[coreCHold] != 0))
        { printf("  FAIL  CORE-BREAK STEP8-REDESIGN: core corner detached under ORDINARY hold with no extra force -- coreBreakForce is too low, or the break fires unconditionally\n"); fails++; }
    physics_shutdown(); recon_shutdown();

    // ── POSITIVE CASE: a LARGE, sustained synthetic external pull must detach the core corner. ──
    if (initLivePath(g) != 0) return 1;
    XpbdToolState ts = tsHold;   // identical setup to the control run above
    xpbd_set_grasp_params(&gpHold);   // same coreBreakForce -- the ONLY difference is the injected force below
    if (physics_set_tool(&ts) != 0) { printf("  FAIL  CORE-BREAK physics_set_tool\n"); physics_shutdown(); recon_shutdown(); return ++fails; }

    int coreC = -1;
    { std::vector<int> hCore(CC); cudaMemcpy(hCore.data(), xpbd_grasp_core_flags(), (size_t)CC*sizeof(int), cudaMemcpyDeviceToHost);
      for (int c = 0; c < CC; c++) if (hCore[c]) { coreC = c; break; } }
    if (coreC < 0) { printf("  FAIL  CORE-BREAK no core corner found\n"); physics_shutdown(); recon_shutdown(); return ++fails; }

    // Apply a LARGE synthetic external pull directly on the core corner by writing into the shared
    // hapticForce buffer (predict sums (extForce+hapticForce)*invMass — writing hapticForce is the
    // EXACT injection technique GATE E1-E4 already established in this file, via physics_haptic_force();
    // no new accessor needed).
    std::vector<float3> hHf(CC, make_float3(0.f, 0.f, 0.f));
    hHf[coreC] = make_float3(500.f, 0.f, 0.f);   // large lateral pull
    cudaMemcpy((float3*)physics_haptic_force(), hHf.data(), (size_t)CC * sizeof(float3), cudaMemcpyHostToDevice);

    for (int f = 0; f < 40; f++) if (physics_step(DT) != 0) { printf("  FAIL  CORE-BREAK physics_step\n"); fails++; break; }

    std::vector<int> hCoreAfter(CC), hGraspedAfter(CC);
    cudaMemcpy(hCoreAfter.data(),    xpbd_grasp_core_flags(), (size_t)CC*sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(hGraspedAfter.data(), xpbd_grasp_flags(),      (size_t)CC*sizeof(int), cudaMemcpyDeviceToHost);
    printf("  info  CORE-BREAK: after 500N pull for 40 frames, core[%d]=%d grasped[%d]=%d (coreBreakForce=%.0f)\n", coreC, hCoreAfter[coreC], coreC, hGraspedAfter[coreC], CORE_BREAK_FORCE);
    if (!(hCoreAfter[coreC] == 0 && hGraspedAfter[coreC] == 0))
        { printf("  FAIL  CORE-BREAK STEP8-REDESIGN: core corner did not detach under a large sustained overload -- if the negative control above also failed, coreBreakForce's margin needs re-deriving at this fixture's h; if only this half failed, the 500N pull may need to be larger or held longer\n"); fails++; }

    physics_shutdown(); recon_shutdown();
    return fails;
}
```

Register it in `main()`, following the same inline idiom as the other STEP 8 gates (insert alongside the CCD1/CCD2 registrations added in Task 2 Step 6):
```cpp
    {
        int bf = g8_gate_core_break(g);
        g_fail += bf;
        printf("  info  CORE-BREAK STEP8-REDESIGN gate: %s (%d failure%s)\n", bf == 0 ? "PASS" : "FAIL", bf, bf == 1 ? "" : "s");
    }
```

- [ ] **Step 8: Build and run all STEP-8 gates**

Run: rebuild `xpbd_livepath_test`; run it; confirm CORE-BREAK and the updated H2 print PASS.

**AUDIT FIX (fourth re-audit, execution-order angle): the CCD1/CCD2 K-timing hedge from Task 2 Step 7 CARRIES FORWARD to this checkpoint, unqualified "all PASS" above does NOT include them.** At this point (Tasks 1-3 landed, Task 5 not yet), the K-loop gating condition is STILL `int K = (g_bound && g_toolArmed) ? 2 : 1` (Task 5 Step 1 is what changes this to `g_grasping`) — Task 3 does nothing to touch K-gating or the CCD dispatch position. CCD1/CCD2's fixtures arm a tool but never grasp (`grasp01=0.f`), so `g_toolArmed` is true and K=2 still fires for them at this checkpoint, an extra elastic+CCD pass relative to the final (Task-5-complete) design. Treat a CCD1/CCD2 failure here the SAME way Task 2 Step 7 does: expected-until-Task-5 IF it traces to this K-timing condition, not a regression to root-cause now. CORE-BREAK and H2 are NOT subject to this hedge — they are genuine-grasp gates where K=2-via-`g_toolArmed` is correct/intended behavior both before and after Task 5, so an unqualified PASS is the right expectation for those two specifically.

**AUDIT FIX (fourth re-audit, invMass-ripple angle):** a PASS on Gate H's ORIGINAL sag-comparison sub-check (`fabsf(capturedSag) < 0.5f * fabsf(ctrlSag)`, lines ~348-354, unchanged by this task) is not automatically sufficient evidence here — unlike every other empirical constant in this plan (`coreBreakForce`, the CCD gates' thresholds, Task 4's slab-omega settle window), this sub-check's underlying premise was never explicitly re-derived for the new reality Task 3 creates. Before this task, core corners contributed an EXACT zero to `capturedSag` (kinematically frozen, `p0==p1` trivially); after this task, they contribute a small but genuinely nonzero compliant sag (bounded by `coreAlpha`/`omegaGrasp`, no longer diluted by an exact-zero term), which will generally make `capturedSag`'s magnitude somewhat LARGER than before. This is very likely still comfortably under the `0.5x` margin given `coreAlpha=1.0e-7` is deliberately near-rigid — but do not just glance at PASS/FAIL here; read the gate's own printed `capturedSag`/`ctrlSag` values and confirm the margin is NOT uncomfortably close to the `0.5x` boundary (i.e. genuinely passing with headroom, not passing by a hair). If it fails or passes with little headroom, the fix is to widen the sag-check's settle window (same non-negotiable principle as Task 4 Step 3: never loosen `coreAlpha`/`omegaGrasp` back toward rigidity just to make an old threshold pass again).

- [ ] **Step 9: Commit**

```bash
git add cuda_plugin/cuda/xpbd/xpbd_solver.h cuda_plugin/cuda/xpbd/xpbd_solver.cu cuda_plugin/tests/xpbd_livepath_test.cu
git commit -m "feat(xpbd): compliant breakable core-pin replaces the kinematic weld"
```

---

### Task 4: Genuinely compliant slab STICK branch + `omegaGrasp` on the whole grasp set

**Files:**
- Modify: `cuda_plugin/cuda/xpbd/xpbd_solver.cu:568-617` (`k_SolveSlabCoulomb` — STICK branch + omega parameter)
- Modify: `cuda_plugin/cuda/xpbd/xpbd_solver.cu` (the call site — pass `g_gp.omegaGrasp` instead of `g_p.omega`)

**Interfaces:**
- Consumes: `omegaGrasp`, `graspMaxDisp` (both added Task 2 Step 1).
- Produces: `k_SolveSlabCoulomb` gains ONE new parameter (`graspMaxDisp`, inserted right after `mu` — see Step 1) and its existing `omega` parameter now receives `g_gp.omegaGrasp` from the call site instead of `g_p.omega`.

- [ ] **Step 1: Make the STICK branch a partial (omega-scaled), capped correction instead of a full rigid snap**

Current full kernel body (real file, lines 568-617, this session's read):
```cpp
__global__ void k_SolveSlabCoulomb(int N, float omega, float h, float contactAlpha, float mu,
                                   float3 hinge, float3 axis, float3 normal, float3 w,
                                   float jawLen, float jawHalfWidth, float gapHalf,
                                   const int* grasped, const int* graspCore, const float* invMass,
                                   const float* slabLambdaPrev, float* slabLambda,
                                   float* anchorU, float* anchorW, float3* pos)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= N) return;
    if (grasped[c] == 0 || graspCore[c] != 0) return;
    float wc = invMass[c];
    if (wc <= 0.f) return;                     // active[]==0 re-filter (cut severed it mid-grasp, §5)

    float3 rel = pos[c] - hinge;
    float u = dot3(rel, axis), v = dot3(rel, w), n = dot3(rel, normal);

    if (u >= 0.f && u <= jawLen && fabsf(v) <= jawHalfWidth) {
        float absN = fabsf(n);
        if (absN > gapHalf) {
            float Ctrue = gapHalf - absN;
            float sgn = (n >= 0.f) ? 1.f : -1.f;
            float denom = wc + contactAlpha;
            float dLambda = (-Ctrue - contactAlpha * slabLambda[c]) / denom;
            float applied = omega * dLambda;
            slabLambda[c] += applied;
            pos[c] = pos[c] - normal * (sgn * wc * applied);
            rel = pos[c] - hinge; n = dot3(rel, normal);   // refresh n for the anchor reconstruction below
        }
    }

    float Fn = fabsf(slabLambdaPrev[c]) / (h * h);
    float3 anchorWorld = hinge + axis * anchorU[c] + w * anchorW[c] + normal * n;  // normal = slab's job
    float3 tangential = pos[c] - anchorWorld;                                     // already in-plane
    float dist = len3(tangential);
    float coneRadius = mu * Fn * (h * h * wc);
    if (dist <= coneRadius + 1e-9f) {
        pos[c] = pos[c] - tangential;                       // STICK: rigid in-plane attachment
    } else {
        float3 dir = tangential * (1.f / dist);
        pos[c] = pos[c] - dir * coneRadius;                  // SLIP: cap the correction at the cone
        float3 newAnchor = anchorWorld + dir * (dist - coneRadius);   // partial-advance (haptics.cu:588
        float3 newRel = newAnchor - hinge;                             // pattern, reimplemented) so the
        anchorU[c] = dot3(newRel, axis);                               // anchor keeps pulling at mu*Fn
        anchorW[c] = dot3(newRel, w);                                  // DURING the slide.
    }
}
```
Replace with (new `graspMaxDisp` parameter added to the signature; BOTH the normal/gap-closing branch AND the tangential STICK branch capped — see the AUDIT FIX comments below for why both, not just STICK):
```cpp
__global__ void k_SolveSlabCoulomb(int N, float omega, float h, float contactAlpha, float mu, float graspMaxDisp,
                                   float3 hinge, float3 axis, float3 normal, float3 w,
                                   float jawLen, float jawHalfWidth, float gapHalf,
                                   const int* grasped, const int* graspCore, const float* invMass,
                                   const float* slabLambdaPrev, float* slabLambda,
                                   float* anchorU, float* anchorW, float3* pos)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= N) return;
    if (grasped[c] == 0 || graspCore[c] != 0) return;
    float wc = invMass[c];
    if (wc <= 0.f) return;                     // active[]==0 re-filter (cut severed it mid-grasp, §5)

    float3 rel = pos[c] - hinge;
    float u = dot3(rel, axis), v = dot3(rel, w), n = dot3(rel, normal);

    if (u >= 0.f && u <= jawLen && fabsf(v) <= jawHalfWidth) {
        float absN = fabsf(n);
        if (absN > gapHalf) {
            float Ctrue = gapHalf - absN;
            float sgn = (n >= 0.f) ? 1.f : -1.f;
            float denom = wc + contactAlpha;
            float dLambda = (-Ctrue - contactAlpha * slabLambda[c]) / denom;
            float applied = omega * dLambda;
            slabLambda[c] += applied;
            // AUDIT FIX (blocking-equivalent finding, third re-audit): this normal/gap-closing branch is
            // STRUCTURALLY IDENTICAL to k_SolveCCDContact's jaw-slab branch (same Ctrue/sgn/dLambda/
            // applied pattern) -- which Task 2 capped at ccdMaxDisp (rule b) -- but this branch governs
            // ALREADY-GRASPED ring corners, which k_SolveCCDContact's own jaw-slab branch explicitly
            // EXCLUDES (`if (grasped[c] == 0)`). It is the only remaining uncapped write Stability
            // Measure D's "cap the total displacement... 0.1-0.25*L" requirement was meant to close;
            // the first audit round only capped the STICK branch below, missing this one. Capped the
            // SAME way.
            float corrMag = wc * applied;
            if (fabsf(corrMag) > graspMaxDisp) corrMag = (corrMag > 0.f ? graspMaxDisp : -graspMaxDisp);
            pos[c] = pos[c] - normal * (sgn * corrMag);
            rel = pos[c] - hinge; n = dot3(rel, normal);   // refresh n for the anchor reconstruction below
        }
    }

    float Fn = fabsf(slabLambdaPrev[c]) / (h * h);
    float3 anchorWorld = hinge + axis * anchorU[c] + w * anchorW[c] + normal * n;  // normal = slab's job
    float3 tangential = pos[c] - anchorWorld;                                     // already in-plane
    float dist = len3(tangential);
    float coneRadius = mu * Fn * (h * h * wc);
    if (dist <= coneRadius + 1e-9f) {
        // redesign 2026-07-07: STICK is now a genuinely COMPLIANT pull, not a rigid full-snap — omega
        // (now omegaGrasp, passed in from the call site) under-relaxes this correction exactly like
        // every other XPBD constraint's dLambda*omega pattern, so the solve converges over the K
        // coupling passes instead of crawling a stiff instantaneous snap each substep. AUDIT FIX (major
        // finding, spec-conformance angle): spec Stability Measure D names BOTH grasp and CCD
        // constraints for the explicit 0.1-0.25*L displacement cap — added graspMaxDisp here too,
        // capped the SAME way k_SolveCCDContact/k_SolveCorePin cap theirs.
        float3 corr = tangential * omega;
        float corrLen = len3(corr);
        if (corrLen > graspMaxDisp) corr = corr * (graspMaxDisp / corrLen);
        pos[c] = pos[c] - corr;                              // STICK: compliant, capped in-plane attachment
    } else {
        float3 dir = tangential * (1.f / dist);
        pos[c] = pos[c] - dir * coneRadius;                  // SLIP: cap the correction at the cone
        float3 newAnchor = anchorWorld + dir * (dist - coneRadius);   // partial-advance (haptics.cu:588
        float3 newRel = newAnchor - hinge;                             // pattern, reimplemented) so the
        anchorU[c] = dot3(newRel, axis);                               // anchor keeps pulling at mu*Fn
        anchorW[c] = dot3(newRel, w);                                  // DURING the slide.
    }
}
```
(The SLIP branch's own `coneRadius` cap is unchanged — Stability Measure D's 0.1-0.25*L cap is a SEPARATE, additional bound on top of it, not a replacement; SLIP was already self-limiting via the friction cone and needs no further change.)

- [ ] **Step 2: Pass `g_gp.omegaGrasp`/`g_gp.graspMaxDisp` instead of `g_p.omega` at the call site**

Find the current call (already visited in Tasks 2-3, inside the K-loop):
```cpp
            k_SolveSlabCoulomb<<<Gr(g_N), 64>>>(g_N, g_p.omega, h, g_gp.contactAlpha, g_gp.mu,
```
Change the second argument and insert `g_gp.graspMaxDisp` after `g_gp.mu` (matching the new signature from Step 1):
```cpp
            k_SolveSlabCoulomb<<<Gr(g_N), 64>>>(g_N, g_gp.omegaGrasp, h, g_gp.contactAlpha, g_gp.mu, g_gp.graspMaxDisp,
```

(The `k_SolveCorePin` call added in Task 3 Step 5 already passes `g_gp.omegaGrasp` AND `g_gp.graspMaxDisp` — no change needed there.)

- [ ] **Step 3: Build and run the existing Gate H (pinch-hold) to confirm the slab under sustained grasp still resists gravity sag correctly with the new omega**

Run: rebuild + run `xpbd_livepath_test`.

Expected: Gate H's sag-comparison sub-check (`capturedSag < 0.5 * ctrlSag`) still passes. If it fails because `omegaGrasp=0.25` makes the slab converge too slowly within the gate's existing 150-frame window, that is a genuine finding — do NOT loosen the omega back toward 1.0 to make the test pass; instead widen the gate's settle window (e.g. 300 frames) since compliant convergence legitimately needs more iterations than a rigid snap, and note the change in the commit message.

- [ ] **Step 4: Commit**

```bash
git add cuda_plugin/cuda/xpbd/xpbd_solver.cu cuda_plugin/tests/xpbd_livepath_test.cu
git commit -m "fix(xpbd): slab STICK branch is now compliant, uses omegaGrasp not xp.omega"
```

---

### Task 5: Re-gate K coupling to grasp-only + reposition the CCD gate outside the K-loop + pre-allocate capture scratch buffers

**Files:**
- Modify: `cuda_plugin/cuda/xpbd/xpbd_solver.cu:1382` (the `K = ... ? 2 : 1` line)
- Modify: `cuda_plugin/cuda/xpbd/xpbd_solver.cu:1386-1416` (the K-loop body — split CCD contact OUT, keep slab/core-pin IN)
- Modify: `cuda_plugin/cuda/xpbd/xpbd_solver.cu:1420-1421` (the `slabLambdaPrev` snapshot — gate on `g_grasping`, not implicitly via the old `g_toolArmed` block)
- Modify: `cuda_plugin/cuda/xpbd/xpbd_solver.cu:1221-1234` (the capture-path `cudaMalloc`/`cudaFree` — replace with pre-allocated persistent scratch buffers)
- Modify: `cuda_plugin/cuda/xpbd/xpbd_solver.cu` (module-scope + `xpbd_bind` tail — declare/allocate the 4 persistent capture-scratch buffers)

**Interfaces:**
- Consumes: nothing new.
- Produces: no signature changes to any kernel — this task ONLY changes host-side dispatch order/gating and buffer lifetime management.

- [ ] **Step 1: Change the K-gating condition from "tool armed" to "actually grasping"**

Current (line 1382):
```cpp
    int K = (g_bound && g_toolArmed) ? 2 : 1;
```
Replace with:
```cpp
    // redesign 2026-07-07 (decision 4, user-confirmed): K=2 coupling gates on GENUINE GRASP, not merely
    // "tool armed" — the force-only CCD/contact path (dispatched separately, see below) needs no
    // reconciliation with the elastic solve; only the core-pin+slab set benefits from a second pass.
    // This is the SECOND framerate fix (alongside retiring the hard contact constraint): before this
    // change K=2 ran unconditionally from the first xpbd_set_tool call, doubling elastic iteration cost
    // even when the tool was merely nearby and not gripping anything.
    int K = (g_bound && g_grasping) ? 2 : 1;
```

- [ ] **Step 2: Split the K-loop body — CCD contact runs ONCE outside the loop (last), grasp constraints stay inside gated on `g_grasping`**

Current K-loop body (lines 1386-1416, after Tasks 2-4's edits are applied):
```cpp
    for (int k = 0; k < K; k++) {
        for (int it = 0; it < g_itersF; it++) {
            for (int c = 0; c < g_numColors; c++)               // structural distance constraints
                if (g_colorCnt[c] > 0)
                    k_solve_edges<<<Gr(g_colorCnt[c]), 64>>>(...);
            if (g_doBendF)
                for (int c = 0; c < g_bNumColors; c++)
                    if (g_bColorCnt[c] > 0)
                        k_solve_bending<<<Gr(g_bColorCnt[c]), 64>>>(...);
            if (g_doStrainF)
                for (int c = 0; c < g_tNumColors; c++)
                    if (g_tColorCnt[c] > 0)
                        k_solve_tet<<<Gr(g_tColorCnt[c]), 64>>>(...);
        }
        if (g_bound && g_toolArmed) {   // CCD gate stays gated on toolArmed (fires regardless of grasp)
            k_SolveCCDContact<<<Gr(g_N), 64>>>(...);
            k_SolveSlabCoulomb<<<Gr(g_N), 64>>>(...);
            k_SolveCorePin<<<Gr(g_N), 64>>>(...);
        }
    }
```

Replace the block AFTER the `for (int it ...)` inner loop closes (i.e. everything from `if (g_bound && g_toolArmed) {` down to its matching `}`) with a grasp-only gate, and add ONE new CCD dispatch AFTER the whole `for (int k = 0; k < K; k++)` loop closes:
```cpp
    for (int k = 0; k < K; k++) {
        for (int it = 0; it < g_itersF; it++) {
            for (int c = 0; c < g_numColors; c++)               // structural distance constraints
                if (g_colorCnt[c] > 0)
                    k_solve_edges<<<Gr(g_colorCnt[c]), 64>>>(...);
            if (g_doBendF)
                for (int c = 0; c < g_bNumColors; c++)
                    if (g_bColorCnt[c] > 0)
                        k_solve_bending<<<Gr(g_bColorCnt[c]), 64>>>(...);
            if (g_doStrainF)
                for (int c = 0; c < g_tNumColors; c++)
                    if (g_tColorCnt[c] > 0)
                        k_solve_tet<<<Gr(g_tColorCnt[c]), 64>>>(...);
        }
        if (g_bound && g_grasping) {   // grasp set ONLY reconciles with elastic across K passes now
            k_SolveSlabCoulomb<<<Gr(g_N), 64>>>(g_N, g_gp.omegaGrasp, h, g_gp.contactAlpha, g_gp.mu, g_gp.graspMaxDisp,
                g_tool.jawHinge, jawAxis3, jawNormal3, g_jawW, g_tool.jawLen, g_tool.jawHalfWidth, gapHalf,
                d_grasped, d_graspCore, d_invMass, d_slabLambdaPrev, d_slabLambda,
                d_coulombAnchorU, d_coulombAnchorW, d_pos);
            k_SolveCorePin<<<Gr(g_N), 64>>>(g_N, g_gp.omegaGrasp, g_gp.coreAlpha, g_gp.coreBreakForce, g_gp.graspMaxDisp, h,
                g_tool.jawHinge, jawAxis3, jawNormal3, g_jawW, d_coreOffU, d_coreOffN, d_coreOffW,
                d_invMass, d_pos, d_vel, d_coreLambda, d_graspCore, d_grasped);
        }
    }
    if (g_bound && g_toolArmed) {   // CCD contact: force/speculative-only, dispatched ONCE, LAST, regardless
                                    // of K or grasp state — "project contact last" (unchanged principle),
                                    // but no longer needs a second coupling pass (redesign 2026-07-07).
        k_SolveCCDContact<<<Gr(g_N), 64>>>(g_N, g_gp.ccdOmega, g_gp.ccdVelThresh, g_gp.ccdMaxDisp, g_gp.contactAlpha,
            g_tool.shaftA, g_tool.shaftB, g_tool.shaftR,
            g_tool.jawHinge, jawAxis3, jawNormal3, g_jawW, g_tool.jawLen, g_tool.jawHalfWidth, gapHalf,
            d_grasped, d_invMass, d_vel, d_pos, d_jawContactLambda, d_shaftContactLambda);
    }
```

- [ ] **Step 3: Gate the lambda-reset block and the `slabLambdaPrev` snapshot on `g_grasping` (not `g_toolArmed`) for the grasp-only buffers; keep shaft/jaw-contact lambda resets on `g_toolArmed`**

Current (from Task 3 Step 5's edit):
```cpp
    if (g_bound && g_toolArmed) {   // STEP 8: multipliers reset ONCE per substep, accumulate across iters
        cudaMemsetAsync(d_slabLambda,         0, (size_t)g_N * sizeof(float));         // AND all K passes
        cudaMemsetAsync(d_jawContactLambda,   0, (size_t)g_N * sizeof(float));
        cudaMemsetAsync(d_shaftContactLambda, 0, (size_t)g_N * sizeof(float));
        cudaMemsetAsync(d_coreLambda,         0, (size_t)g_N * sizeof(float));
    }
```
Replace with two separately-gated blocks (the CCD/shaft/jaw lambdas are consumed by the always-runs-when-armed CCD kernel; the slab/core lambdas are consumed only by the grasp-only kernels):
```cpp
    if (g_bound && g_toolArmed) {   // CCD gate's multipliers: reset whenever the tool is armed
        cudaMemsetAsync(d_jawContactLambda,   0, (size_t)g_N * sizeof(float));
        cudaMemsetAsync(d_shaftContactLambda, 0, (size_t)g_N * sizeof(float));
    }
    if (g_bound && g_grasping) {     // grasp-only multipliers: reset only while genuinely grasping
        cudaMemsetAsync(d_slabLambda, 0, (size_t)g_N * sizeof(float));
        cudaMemsetAsync(d_coreLambda, 0, (size_t)g_N * sizeof(float));
    }
```

Current `slabLambdaPrev` snapshot (line 1420-1421):
```cpp
    if (g_bound && g_toolArmed)   // STEP 8: snapshot for the NEXT substep's Fn (design §5 item 3 PREV-substep timing)
        cudaMemcpyAsync(d_slabLambdaPrev, d_slabLambda, (size_t)g_N * sizeof(float), cudaMemcpyDeviceToDevice);
```
Replace the gate:
```cpp
    if (g_bound && g_grasping)   // snapshot for the NEXT substep's Fn (design §5 item 3 PREV-substep timing)
        cudaMemcpyAsync(d_slabLambdaPrev, d_slabLambda, (size_t)g_N * sizeof(float), cudaMemcpyDeviceToDevice);
```

- [ ] **Step 4: Pre-allocate the 4 capture-scratch buffers once (bind time) instead of malloc/free every capture event**

Current capture path (lines 1221-1234, this session's read):
```cpp
        int *dCand = nullptr; float *dU = nullptr, *dV = nullptr, *dNn = nullptr;
        if (cudaMalloc(&dCand, (size_t)cc * sizeof(int))  != cudaSuccess) return XPBD_ERR_BAD_ARG;
        if (cudaMalloc(&dU,    (size_t)cc * sizeof(float)) != cudaSuccess) { cudaFree(dCand); return XPBD_ERR_BAD_ARG; }
        if (cudaMalloc(&dV,    (size_t)cc * sizeof(float)) != cudaSuccess) { cudaFree(dCand); cudaFree(dU); return XPBD_ERR_BAD_ARG; }
        if (cudaMalloc(&dNn,   (size_t)cc * sizeof(float)) != cudaSuccess) { cudaFree(dCand); cudaFree(dU); cudaFree(dV); return XPBD_ERR_BAD_ARG; }
        k_MarkGraspCandidates<<<Gr(cc), 64>>>(cc, g_tool.jawHinge, axis, normal, g_jawW,
            g_tool.jawLen, g_tool.jawHalfWidth, gapHalf, g_gp.contactSkin,
            g_liveActive, g_livePinned, d_pos, dCand, dU, dV, dNn);
        std::vector<int> hCand(cc); std::vector<float> hU(cc), hV(cc), hN(cc);
        cudaMemcpy(hCand.data(), dCand, (size_t)cc * sizeof(int),   cudaMemcpyDeviceToHost);
        cudaMemcpy(hU.data(),    dU,    (size_t)cc * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(hV.data(),    dV,    (size_t)cc * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(hN.data(),   dNn,    (size_t)cc * sizeof(float), cudaMemcpyDeviceToHost);
        cudaFree(dCand); cudaFree(dU); cudaFree(dV); cudaFree(dNn);
```

Add 4 persistent module-scope buffers (alongside the other STEP 8 declarations):
```cpp
static int*    d_candScratch = nullptr;   // [N] k_MarkGraspCandidates scratch — allocated ONCE at bind,
static float*  d_uScratch    = nullptr;   // reused every capture event (redesign 2026-07-07: avoids a
static float*  d_vScratch    = nullptr;   // cudaMalloc/cudaFree pair on every rising-edge capture, which
static float*  d_nScratch    = nullptr;   // could storm if grasp01 crosses threshold rapidly)
```
Allocate them in `xpbd_bind`'s tail, alongside the other grasp buffers:
```cpp
    XK(cudaMalloc(&d_candScratch, (size_t)cc * sizeof(int)));
    XK(cudaMalloc(&d_uScratch,    (size_t)cc * sizeof(float)));
    XK(cudaMalloc(&d_vScratch,    (size_t)cc * sizeof(float)));
    XK(cudaMalloc(&d_nScratch,    (size_t)cc * sizeof(float)));
```
Free them in `xpbd_shutdown` alongside the other grasp buffers.

Replace the capture-path body to reuse them (no per-event malloc/free):
```cpp
        k_MarkGraspCandidates<<<Gr(cc), 64>>>(cc, g_tool.jawHinge, axis, normal, g_jawW,
            g_tool.jawLen, g_tool.jawHalfWidth, gapHalf, g_gp.contactSkin,
            g_liveActive, g_livePinned, d_pos, d_candScratch, d_uScratch, d_vScratch, d_nScratch);
        std::vector<int> hCand(cc); std::vector<float> hU(cc), hV(cc), hN(cc);
        cudaMemcpy(hCand.data(), d_candScratch, (size_t)cc * sizeof(int),   cudaMemcpyDeviceToHost);
        cudaMemcpy(hU.data(),    d_uScratch,    (size_t)cc * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(hV.data(),    d_vScratch,    (size_t)cc * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(hN.data(),    d_nScratch,    (size_t)cc * sizeof(float), cudaMemcpyDeviceToHost);
```
(No `cudaFree` calls needed here anymore — the buffers persist for the module's lifetime. Also remove the now-dead `if (cudaMalloc(...) != cudaSuccess) return XPBD_ERR_BAD_ARG;` chain from this call site; allocation failures are now only possible at `xpbd_bind` time, which already fails loudly via the `XK(...)` macro.)

- [ ] **Step 5: Add a gate confirming K=1 (not 2) when armed-but-not-grasping**

Append to `cuda_plugin/tests/xpbd_livepath_test.cu`:
```cpp
// ── STEP-8-REDESIGN gate KGATE (2026-07-07): with the tool armed but grasp01 below the capture
// threshold (never grasping), the elastic solve must behave EXACTLY like the K=1 standalone reference
// (bitwise-comparable within float tolerance) — confirming K=2 no longer runs unconditionally just
// because a tool is present (the SECOND framerate fix, alongside the CCD retirement). Mirrors GATE B's
// bound==standalone parity pattern.
static int g8_gate_k_regate(const Lattice& g) {
    int fails = 0;

    // Reference: tool NEVER set (g_toolArmed stays false) — the historical K=1 behavior.
    if (initLivePath(g) != 0) return 1;
    for (int f = 0; f < 10; f++) if (physics_step(DT) != 0) { printf("  FAIL  KGATE ref physics_step\n"); physics_shutdown(); recon_shutdown(); return ++fails; }
    std::vector<float> pRef(3 * CC); recon_readback_corner_pos(pRef.data());
    physics_shutdown(); recon_shutdown();

    // Candidate: tool ARMED (parked far away, grasp01=0 -- never crosses the capture threshold).
    if (initLivePath(g) != 0) return 1;
    XpbdToolState ts = {};
    ts.shaftA = make_float3(-1000.f, -1000.f, -1000.f); ts.shaftB = make_float3(-1000.f, -999.f, -1000.f); ts.shaftR = 0.01f;
    ts.jawAxis = make_float3(1.f, 0.f, 0.f); ts.jawNormal = make_float3(0.f, 1.f, 0.f);
    ts.jawHinge = make_float3(-1000.f, -1000.f, -1000.f); ts.jawLen = 0.f; ts.jawHalfWidth = 0.f; ts.jawGap = 0.f;
    ts.grasp01 = 0.f;
    if (physics_set_tool(&ts) != 0) { printf("  FAIL  KGATE physics_set_tool\n"); physics_shutdown(); recon_shutdown(); return ++fails; }
    for (int f = 0; f < 10; f++) if (physics_step(DT) != 0) { printf("  FAIL  KGATE cand physics_step\n"); fails++; physics_shutdown(); recon_shutdown(); return fails; }
    std::vector<float> pCand(3 * CC); recon_readback_corner_pos(pCand.data());
    physics_shutdown(); recon_shutdown();

    float maxDiff = 0.f;
    for (int i = 0; i < 3 * CC; i++) maxDiff = fmaxf(maxDiff, fabsf(pRef[i] - pCand[i]));
    printf("  info  KGATE armed-not-grasping vs K=1 reference: max corner diff=%.8f\n", maxDiff);
    if (!(maxDiff < 1e-4f))
        { printf("  FAIL  KGATE STEP8-REDESIGN: armed-but-not-grasping diverged from the K=1 reference by %.8f (K=2 may still be running unconditionally)\n", maxDiff); fails++; }

    return fails;
}
```
Register it in `main()`, following the same inline idiom as the other STEP 8 gates:
```cpp
    {
        int gf = g8_gate_k_regate(g);
        g_fail += gf;
        printf("  info  KGATE STEP8-REDESIGN K-regate gate: %s (%d failure%s)\n", gf == 0 ? "PASS" : "FAIL", gf, gf == 1 ? "" : "s");
    }
```

- [ ] **Step 6: Build and run the FULL suite**

Run: rebuild `xpbd_livepath_test`; run all gates (should now be 30 + CCD1 + CCD2 + CORE-BREAK + KGATE = 34); also rebuild+run `xpbd_selftest` (58 checks) and the 6 oracle tests, per this project's established full-regression convention.

Expected: ALL PASS, zero regressions in the pre-existing 30+58+6.

- [ ] **Step 7: Commit**

```bash
git add cuda_plugin/cuda/xpbd/xpbd_solver.cu cuda_plugin/tests/xpbd_livepath_test.cu
git commit -m "perf(xpbd): K=2 gates on genuine grasp only; CCD contact runs once outside the loop; capture scratch buffers pre-allocated"
```

---

### Task 6: Auto-clearance init pose (Unity C#)

**Files:**
- Modify: `Assets/ReconGridDC/Cuda/LiverCudaManager.cs:544` (delete the fixed `-3L,+4L,0` heuristic) and the surrounding `Start()` block (~505-559) to reorder the clearance search to AFTER `_grasper.Build()`.

**Interfaces:**
- Consumes: `meshLs` (the `MeshLevelSet` local already constructed earlier in `Start()` at line 411 — `ILevelSetProvider.Sample(float3 p)`, signed distance, `<0` inside `>0` outside), `_grasper.JawSlabWorld(...)` (existing method, called with `grasp01=0` to get the OPEN-jaw assembly extents).
- Produces: `_hapticCenter`'s final value — no other task depends on this beyond what already consumed `_hapticCenter`.

- [ ] **Step 1: Confirm `meshLs` is in scope at the haptics block (same method, declared earlier)**

Run: `grep -n "var meshLs\|_hapticCenter =" Assets/ReconGridDC/Cuda/LiverCudaManager.cs`

Expected: `meshLs` declared around line 411, `_hapticCenter = ...` around line 544 — both inside the same `Start()` method, `meshLs` declared textually earlier, so it is in scope (C# local scoping is sequential-within-method, not block-restricted here — confirm no enclosing `{ }` block ends between the two lines before relying on this).

- [ ] **Step 2: Remove the fixed heuristic; move `_hapticCenter`'s assignment to AFTER the grasper build block**

Current (line 544, inside the `if (enableHaptics)` block, BEFORE `_grasper` is constructed at line 549):
```csharp
                _hapticCenter = (Vector3)gridCenterRt + new Vector3(-3f * Lrt, 4f * Lrt, 0f);   // start above/outside
                _hapticAxis = new Vector3(0, 0, 1);
                SetupProxyLine();
                // Assemble the 3-part grasper mesh (proxy + physical ghost); if it builds, hide the
                // diagnostic capsule lines and show the instrument instead (user: replace the two rods).
                _grasper = gameObject.AddComponent<GrasperRig>();
                _grasper.fitLength = Mathf.Max(0.5f, hapticProbeLength);   // match the short probe (user: too big)
                bool meshOk = false;
                try { meshOk = _grasper.Build(); }   // a mesh-build failure must NOT abort Start -> keep the lines
                catch (System.Exception ex) { Debug.LogWarning("[LiverCuda] GrasperRig.Build failed, keeping capsule lines: " + ex.Message); }
                if (meshOk) { _proxyLine.enabled = false; _physLine.enabled = false; _springLine.enabled = false; }
```

Replace with (moves the `_hapticCenter` assignment to AFTER `meshOk` is known, and runs the clearance search there):
```csharp
                _hapticAxis = new Vector3(0, 0, 1);
                SetupProxyLine();
                // Assemble the 3-part grasper mesh (proxy + physical ghost); if it builds, hide the
                // diagnostic capsule lines and show the instrument instead (user: replace the two rods).
                _grasper = gameObject.AddComponent<GrasperRig>();
                _grasper.fitLength = Mathf.Max(0.5f, hapticProbeLength);   // match the short probe (user: too big)
                bool meshOk = false;
                try { meshOk = _grasper.Build(); }   // a mesh-build failure must NOT abort Start -> keep the lines
                catch (System.Exception ex) { Debug.LogWarning("[LiverCuda] GrasperRig.Build failed, keeping capsule lines: " + ex.Message); }
                if (meshOk) { _proxyLine.enabled = false; _physLine.enabled = false; _springLine.enabled = false; }

                // redesign 2026-07-07 (decision 2, auto-clearance): replace the fixed -3L,+4L,0 heuristic
                // (which embedded the jaws inside the liver at real-scene scale, L~0.4 — root cause #5 of
                // the "liver sticks to jaws at init" bug) with a runtime search along the approach axis
                // until BOTH the shaft capsule and the CLOSED-pose jaw slab box clear the tissue SDF by a
                // skin margin. AUDIT FIX (fourth re-audit, fresh spec-pass angle): spec item A explicitly
                // requires the CLOSED pose ("SDF > skin for the union of the shaft capsule AND the
                // closed-pose jaw slab box... a shaft-only clearance can still leave the jaw straddling
                // the surface") — an earlier draft sampled the OPEN pose (grasp01=0) instead, reasoning
                // that the open pose's wider gap was "more conservative." That reasoning doesn't hold:
                // JawSlabWorld's open angle ROTATES the tip position in 3D (not merely widens a gap along
                // one axis), so clearing the open pose's extremes does NOT guarantee the closed pose's
                // (physically different) tip positions also clear — exactly the straddling failure mode
                // the spec's own rationale warns about. Sample grasp01=1 (closed), matching the spec
                // literally. Reuses `meshLs` (the exact MeshLevelSet built above at Start() Section 3,
                // still in scope) — no new SDF construction needed.
                {
                    Vector3 approachAxis = _hapticAxis.normalized;
                    Vector3 candidate = (Vector3)gridCenterRt + new Vector3(-3f * Lrt, 4f * Lrt, 0f);   // starting guess
                    float clearSkin = hapticSkinRadius + 0.5f * Lrt;
                    float step = 0.25f * Lrt;
                    bool cleared = false;
                    // audit fix (minor finding, spec-conformance angle: spec item A explicitly says "log
                    // both clearances") — these are captured OUTSIDE the loop so the final log line can
                    // report shaft and jaw clearance SEPARATELY, not just one aggregate OK/FAILED flag
                    // (without this, a non-convergence failure gives no way to tell which of the two
                    // sub-assemblies never cleared). meshOk-false also now surfaces explicitly: jawClear
                    // defaulting to true in that fallback path means only shaft clearance was checked, per
                    // the spec's own warning that "a shaft-only clearance can still leave the jaw
                    // straddling the surface" — this is logged so it is visible, not silent.
                    bool lastShaftClear = false, lastJawClear = false;
                    for (int iter = 0; iter < 64; iter++)
                    {
                        RodMath.Endpoints(candidate, approachAxis, hapticProbeLength, out Vector3 chs, out Vector3 che);
                        float sdfA = meshLs.Sample((Unity.Mathematics.float3)chs);
                        float sdfB = meshLs.Sample((Unity.Mathematics.float3)che);
                        float sdfC = meshLs.Sample((Unity.Mathematics.float3)candidate);
                        bool shaftClear = sdfA > clearSkin && sdfB > clearSkin && sdfC > clearSkin;

                        bool jawClear = true;
                        // audit fix (blocking-equivalent, verified via a minimal dotnet-build repro):
                        // `meshOk && _grasper.JawSlabWorld(..., out jH, ...)` short-circuits when meshOk
                        // is false, so the C# compiler's definite-assignment analysis cannot prove jH/jA/
                        // jN/jM/jG/jL/jHW were assigned inside `if (jawChecked)` below (CS0165 "use of
                        // unassigned local variable") -- it does NOT track that jawChecked==true implies
                        // JawSlabWorld ran. Pre-declare with `= default` so the `out` call ASSIGNS them
                        // (rather than inline-declaring them at the call site) and the compiler is happy
                        // regardless of which branch runs.
                        Vector3 jH = default, jA = default, jN = default, jM = default;
                        float jG = default, jL = default, jHW = default;
                        bool jawChecked = meshOk && _grasper.JawSlabWorld(candidate, approachAxis, 1f,
                            out jH, out jA, out jN, out jM, out jG, out jL, out jHW);
                        if (jawChecked)
                        {
                            // Sample the 4 extremes of the CLOSED (grasp01=1, spec-mandated) jaw box:
                            // both hinge-end and tip-end, both jaw-plane sides. An approximate, not exact,
                            // Minkowski clearance test — sufficient since the search only retreats along
                            // one axis and a further retreat monotonically increases clearance for the
                            // liver's roughly-convex near-surface region.
                            Vector3 halfGapN = jN * (0.5f * jG);
                            Vector3 tipU = jA * jL;
                            float sN0 = meshLs.Sample((Unity.Mathematics.float3)(jH + halfGapN));
                            float sN1 = meshLs.Sample((Unity.Mathematics.float3)(jH - halfGapN));
                            float sN2 = meshLs.Sample((Unity.Mathematics.float3)(jH + tipU + halfGapN));
                            float sN3 = meshLs.Sample((Unity.Mathematics.float3)(jH + tipU - halfGapN));
                            jawClear = sN0 > clearSkin && sN1 > clearSkin && sN2 > clearSkin && sN3 > clearSkin;
                        }
                        lastShaftClear = shaftClear; lastJawClear = jawClear;

                        if (shaftClear && jawClear) { cleared = true; break; }
                        candidate -= approachAxis * step;
                    }
                    if (!cleared)
                        Debug.LogWarning($"[LiverCuda] auto-clearance init did not converge in 64 steps; using last candidate {candidate}. Consider a larger starting offset.");
                    if (!meshOk)
                        Debug.LogWarning("[LiverCuda] auto-clearance: GrasperRig mesh unavailable, jaw clearance NOT checked (shaft-only clearance) — see spec item A's warning that this can still leave the jaw straddling the surface.");
                    _hapticCenter = candidate;
                    Debug.Log($"[LiverCuda] init clearance: hapticCenter={_hapticCenter} shaftClear={lastShaftClear} jawClear={lastJawClear} (auto-clearance {(cleared ? "OK" : "FAILED, see warning above")})");
                }
```

- [ ] **Step 2: Confirm `float3` conversion compiles (Unity.Mathematics interop)**

Run: `grep -n "using Unity.Mathematics" Assets/ReconGridDC/Cuda/LiverCudaManager.cs`

Expected: the `using Unity.Mathematics;` directive already exists (the file already calls `math`-namespace types elsewhere per the codebase's established SDF usage); if missing, add it to the file's `using` block. Confirm `MeshLevelSet.Sample(float3 p)` accepts `Unity.Mathematics.float3` (per `Assets/ReconGridDC/Preprocess/ILevelSetProvider.cs:6`) and that an implicit or explicit `Vector3 -> float3` conversion exists in this codebase (search `grep -n "float3)" Assets/ReconGridDC/Preprocess/*.cs` for the established cast idiom used elsewhere, and match it exactly rather than assuming Unity provides an implicit conversion).

- [ ] **Step 3: Unity manual verification**

Enter Play mode. Before touching any input, confirm via the new `[LiverCuda] init clearance: ...` log line that clearance converged (`OK`, not `FAILED`). Press P (existing `DumpDiagnostics` hotkey) immediately at frame 1 and confirm `grasped=0` (no accidental capture) via the `[DEBUG-grasp1]` log already in place. Visually confirm the grasper model spawns fully outside the liver mesh, not embedded.

- [ ] **Step 4: Commit**

```bash
git add Assets/ReconGridDC/Cuda/LiverCudaManager.cs
git commit -m "fix(unity): auto-clearance init pose replaces the fixed -3L,+4L,0 heuristic"
```

---

### Task 7: Widen the degenerate jaw half-width floor (Unity C#)

**Files:**
- Modify: `Assets/ReconGridDC/Cuda/GrasperRig.cs:77-81` (`Build()` — the `_jawHalfWidth` computation)
- Test: none new — verified via existing `[DEBUG-grasp1]` logging + manual checklist (Step 2)

**Interfaces:**
- Consumes: `fitLength` (existing public field, already set by `LiverCudaManager.cs:550`); `graspContactSkinOverL` (added in Task 2 Step 5 — used here only in the sense that this task's spec item B bullet is the same one that field fixes; no code dependency between them).
- Produces: `_jawHalfWidth` (existing private field, read by `JawSlabWorld` — no signature change).

- [ ] **Step 1: Add a minimum-width floor relative to `fitLength`**

Current (lines 77-81, this session's read — NOTE: line numbers corrected from an earlier draft's off-by-one; the block starts at the `_collSoup =` line, not one line earlier):
```csharp
                _collSoup = new float[3 * (_collJawUpV.Length + _collJawDnV.Length)];
                float hw = 0f;
                foreach (var v in _collJawUpV) hw = Mathf.Max(hw, Mathf.Abs(v.x));
                foreach (var v in _collJawDnV) hw = Mathf.Max(hw, Mathf.Abs(v.x));
                _jawHalfWidth = Mathf.Max(hw, 1e-3f);
```
Replace the last line:
```csharp
                // redesign 2026-07-07: the collision-mesh-derived half-width can be a "degenerate
                // quarter-voxel knife" at real-scene scale (the mesh's own thin-jaw geometry, not a bug
                // in this computation) — floor it to a fraction of the jaw's own length so the capture
                // band and jaw-box contact test never collapse to a sliver regardless of L. 0.15x is a
                // STARTING ANCHOR; tune against the video's pinch feel. This is spec Grasp-layer item B's
                // "widen the degenerate quarter-voxel jaw box" half; the sibling "/L-scale the skins"
                // half is fixed in Task 2 Step 5 (graspContactSkinOverL, a separate flat-constant-at-
                // real-scale gap in the SAME spec bullet, on the capture-band skin rather than the jaw
                // box itself). DELIBERATE choice (fourth re-audit considered and confirmed this, not an
                // inconsistency): floors against `fitLength` (this instrument's own jaw length), NOT
                // against `L`/`Lrt` (tissue voxel size) — the jaw's physical width is a property of the
                // INSTRUMENT model, not the tissue's grid resolution, so it should track the jaw's own
                // scale rather than an unrelated voxel size; `graspContactSkinOverL`'s `/L`-scaling is
                // correct for THAT field because a capture band genuinely should track tissue voxel size.
                _jawHalfWidth = Mathf.Max(hw, 0.15f * fitLength, 1e-3f);
```

- [ ] **Step 2: Unity manual verification**

Enter Play mode, hold G near the liver surface, confirm the captured-corner count (`[DEBUG-grasp1]` log's `grasped=` field) is non-trivial (more than 1-2 corners) on a normal-sized grasp gesture, not just the mandatory-fallback single corner.

- [ ] **Step 3: Commit**

```bash
git add Assets/ReconGridDC/Cuda/GrasperRig.cs
git commit -m "fix(unity): floor jaw half-width to a fraction of fitLength, not just the raw collision mesh"
```

---

### Task 8: Retune `hapticDp` starting value; mark `contactBeta`/`contactDispMax` as dead (Unity C#)

**Files:**
- Modify: `Assets/ReconGridDC/Cuda/LiverCudaManager.cs:104` (`hapticDp` default), `:109`/`:116` (`hapticContactBeta`, `hapticDispMaxOverL` — mark dead in comments, keep the fields for ABI stability), and `:528-539` (the `hp` construction — remove the now-dead `contactBeta`/`contactDispMax` initializers; audit fix, see Step 2)
- Modify: `Assets/ReconGridDC/Cuda/HapticTool.cs` (the C# `HapticParams` interop struct definition, distinct from `LiverCudaManager.cs`'s construction call site — audit fix: the original draft of this task never listed this file)
- Modify: `cuda_plugin/cuda/haptics.h` (if `HapticParams` documents these fields — mark them dead there too, matching the existing `hInit`/`hMax` dead-ABI-placeholder precedent in `xpbd_solver.h`)

**Interfaces:**
- Consumes: nothing new.
- Produces: nothing new — pure tuning-value + documentation change, no signature change (removing the fields from the marshalled struct would require touching the ABI guard and is NOT part of this task — YAGNI, keep them as harmless dead ABI placeholders exactly like `hInit`/`hMax`).

- [ ] **Step 1: Compute and document the `hapticDp` starting value**

`k_ScatterContact` (Task 1) can, in the worst case (a single localized contact patch where one corner is nearest for every triangle-vertex sample), deposit the FULL `reaction` force onto ONE corner — not `reaction/3` (the `/3` divisor only splits a single triangle's 3 vertex-lanes; if all lanes across MANY triangles resolve to the SAME nearest corner, their `w`-shares sum back to the full `reaction` via the partition-of-unity property of `w = depth/depthSum`). With per-corner mass = 1.0 (uniform, this codebase's established mass model) and `hapticKp = 2200` (current default), the critical-damping floor for this single-corner worst case is `Dp_min = 2*sqrt(Kp * m) = 2*sqrt(2200*1) ≈ 93.8`. The current default (`hapticDp = 10`) is roughly 9x under-damped against this worst case.

In `Assets/ReconGridDC/Cuda/LiverCudaManager.cs:104`, change:
```csharp
        public float hapticDp            = 10f;       // linear damping D_p (contact only; raised with Kp to stay stable)
```
to:
```csharp
        // redesign 2026-07-07: retuned from 10 -> 100. Derivation (worst case, single-corner contact
        // patch — k_ScatterContact's w=depth/depthSum partition-of-unity can concentrate the FULL
        // reaction force on one corner, not reaction/3): critical damping floor for the tool<->tissue
        // position-force loop is Dp_min = 2*sqrt(Kp*m) = 2*sqrt(2200*1) ~= 93.8 (m=1, this codebase's
        // uniform per-corner mass). STARTING ANCHOR only — verify via Task 8 Step 2's static-hold test,
        // paper-silent (Eq22/27 do not specify Dp numerically).
        public float hapticDp            = 100f;      // linear damping D_p (contact only; sized to over-damp
                                                        // the worst-case single-corner reaction, see derivation above)
```

- [ ] **Step 2: Mark `contactBeta`/`contactDispMax` as dead (Task 1 retired their only consumer)**

`Assets/ReconGridDC/Cuda/LiverCudaManager.cs:109` currently:
```csharp
        public float hapticContactBeta   = 0.85f;     // fraction of the physical penetration applied/frame
```
becomes:
```csharp
        // DEAD as of redesign 2026-07-07 (Task 1 retired k_ApplyContactDisp, the only consumer of
        // contactBeta/contactDispMax). Kept as harmless Inspector fields for scene-file compatibility —
        // do NOT wire these into any new code. Mirrors the hInit/hMax dead-ABI-placeholder precedent
        // (xpbd_solver.cu comment, migration STEP 4).
        public float hapticContactBeta   = 0.85f;     // DEAD — see note above
```
`:116` currently:
```csharp
        public float hapticDispMaxOverL  = 0.6f;      // per-frame dent step clamp = this * L
```
becomes:
```csharp
        public float hapticDispMaxOverL  = 0.6f;      // DEAD — see hapticContactBeta's note above
```

**AUDIT FIX (third re-audit): this step originally assumed Task 1 already removed the C# `hp`-construction usage of these two fields — that assumption is WRONG.** Task 1's Files list (see its header) scopes its edits exclusively to `cuda_plugin/cuda/haptics.cu` — it never touches `Assets/ReconGridDC/Cuda/LiverCudaManager.cs`. Task 1 only deletes the NATIVE kernels that *read* `g_hp.contactBeta`/`contactDispMax` inside `haptics.cu`; the C# `hp`-construction call site in a completely different file is untouched by anything in Task 1's scope. So the two lines below are certain (not merely possible) to still exist after Task 1 — remove them HERE, in this task, as the step that actually owns this cleanup:

Find the `hp` construction (real file, lines 528-539 as of this plan's writing — this happens to stay valid even after Task 6's ~50-line insertion further down the SAME method, since 528-539 sits textually BEFORE Task 6's insertion point at line 544, but verify with `grep -n "new HapticTool.HapticParams"` rather than trusting the number if Task 6 already landed and shifted things unexpectedly; the code below is a unique, unmodified text anchor either way):
```csharp
                var hp = new HapticTool.HapticParams {
                    kP = hapticKp, dP = hapticDp, kO = 20f, dO = 0.5f,
                    a = 0.1f, eps = 1e-3f,                       // Algorithm 1 constants (paper)
                    reactionMax = hapticReactionMax,
                    contactBeta = hapticContactBeta,
                    contactDispMax = hapticDispMaxOverL * Lrt,   // per-frame clamp in world units
                    captureMargin = hapticCaptureMarginOverL * Lrt,  // D18 detection band (world units)
                    proxyTrackStep = hapticProxyTrackOverL * Lrt,    // D19 proxy attraction (world units)
                    muS = hapticMuS, muK = hapticMuK,                // friction cone (Chai3D)
                    outerMax = 128
                };
```
Remove the two now-dead lines (`contactBeta = ...` and `contactDispMax = ...`):
```csharp
                var hp = new HapticTool.HapticParams {
                    kP = hapticKp, dP = hapticDp, kO = 20f, dO = 0.5f,
                    a = 0.1f, eps = 1e-3f,                       // Algorithm 1 constants (paper)
                    reactionMax = hapticReactionMax,
                    captureMargin = hapticCaptureMarginOverL * Lrt,  // D18 detection band (world units)
                    proxyTrackStep = hapticProxyTrackOverL * Lrt,    // D19 proxy attraction (world units)
                    muS = hapticMuS, muK = hapticMuK,                // friction cone (Chai3D)
                    outerMax = 128
                };
```
(Leaving `contactBeta`/`contactDispMax` UNSET in this initializer is safe — `HapticTool.HapticParams` is a C# struct, so the omitted fields simply default to 0, and `haptics_init`/`haptics_step` no longer read them after Task 1's kernel deletion; do NOT remove the fields from the `HapticTool.HapticParams` struct DEFINITION itself, to avoid unnecessary ABI churn — see the note below.)

Add `Assets/ReconGridDC/Cuda/HapticTool.cs` to this task's Files list — that is where the actual C# `HapticParams` interop struct (the ABI-marshalled mirror of the native `HapticParams`) is defined, distinct from `LiverCudaManager.cs`'s `hp`-construction call site above; confirm via `grep -n "contactBeta\|contactDispMax" cuda_plugin/cuda/haptics.h` and `Assets/ReconGridDC/Cuda/HapticTool.cs` whether the struct FIELDS themselves (native and/or C# interop) should also get a "DEAD" comment, matching the Inspector-field comments added earlier in this task — do not remove the fields, only mark them dead, to avoid unnecessary ABI churn.

- [ ] **Step 3: Unity manual static-hold verification (the honest empirical check this derivation calls for)**

Enter Play mode. Press the tool against the liver surface and HOLD it still (no further movement) for several seconds. Watch the HUD's `|F|` readout (existing, per `LiverCudaManager.cs`'s HUD text). Confirm it converges to a steady value smoothly — no sustained oscillation/buzzing. If it still oscillates, `Dp=100` is insufficient for THIS specific contact geometry — bump further (document the actual working value found, do not silently leave a value that visibly buzzes).

Also check `hapticReactionMax` (currently 600) against the SAME test: since Task 1 made the reaction force the ONLY tissue-side contact effect (no dent to visually compensate), a press that used to LOOK like it was "doing something" via the dent alone might now feel like a faint push with no dent to sell it, even though the force readout is climbing correctly (the redesign spec's own flagged risk: "faint push/pass-through" once dent is force-only). If pressing firmly feels weak/pass-through despite `|F|` climbing toward `reactionMax`, raise `hapticReactionMax` (document the new value and why, same as the `Dp` derivation above) rather than reintroducing any position-write mechanism.

- [ ] **Step 4: Commit**

```bash
git add Assets/ReconGridDC/Cuda/LiverCudaManager.cs cuda_plugin/cuda/haptics.h
git commit -m "tune(haptics): retune Dp for the force-only contact loop; mark contactBeta/contactDispMax dead"
```

---

### Task 9: Full regression pass + Unity end-to-end verification checklist

**Files:**
- None modified — this task is verification-only.

**Interfaces:**
- Consumes: everything from Tasks 1-8.
- Produces: a go/no-go verdict for the whole redesign.

- [ ] **Step 1: Rebuild and run the complete native suite**

Run (per `[[cuda-build-loop]]`):
```bash
# from the project's established build directory, per memory's documented recipe
cmake --build build --config Release --target xpbd_livepath_test
cmake --build build --config Release --target xpbd_selftest
./build/Release/xpbd_livepath_test.exe
./build/Release/xpbd_selftest.exe
```
Expected: `xpbd_livepath_test` reports 34/34 gates PASS (the original 30 + CCD1 + CCD2 + CORE-BREAK + KGATE); `xpbd_selftest` reports 58/58 PASS; the 6 oracle tests PASS. Any failure here is a genuine regression — return to the task that owns the failing area, do not proceed to Unity verification with a known-red native suite.

- [ ] **Step 2: Rebuild the Unity native plugin DLL and redeploy**

Run whatever this project's established DLL rebuild+redeploy step is (per `[[cuda-build-loop]]` memory — confirm the exact command via that memory or the repo's build scripts before running).

- [ ] **Step 3: Unity end-to-end verification checklist**

In Play mode, walk through EVERY item; do not report Task 9 complete until all are checked. AUDIT FIX (fourth re-audit, fresh spec-pass angle): the spec's own stage-1 verification criteria are more specific than the original checklist here — "tears stay at 0" and "`|F|` never reaches the original bug report's ~2460N blow-up figure" were previously only implicitly covered ("no NaN") rather than checked directly. Explicit items added below (marked NEW):
- [ ] Frame 1 (before any input): `[DEBUG-grasp1]`/`DumpDiagnostics` show `grasped=0`, no NaN, tool visibly outside the liver mesh (Task 6).
- [ ] **NEW:** Frame 1 through several seconds of idle (tool present, no input): press P (`DumpDiagnostics`) and confirm the tear-count field (`LCS_GetCutDebug`'s counter 15, same one `[DEBUG-grasp1]` already reads as `tearDeltaSincePrevLog`) stays at 0, and `|F|` (HUD) stays near 0 — the spec's stage-1 "idle substep" criteria, not just "no NaN."
- [ ] Move the tool near/through the liver WITHOUT holding G: liver does not follow/stick to the tool; framerate stays flat (no K=2 cost; Task 5).
- [ ] **NEW (fourth re-audit round 2, scenario-walkthrough angle):** approach the surface at a NORMAL, deliberate keyboard speed (not a deliberate fast-plunge tap) and watch closely as the tool nears contact — this is a real, previously-untested regime: `ccdVelThresh`'s STARTING ANCHOR (0.5 u/s) sits far below ordinary keyboard drive speed (`hapticMoveSpeed=4` u/s), so the CCD gate is essentially ALWAYS live for any deliberate approach, not just fast plunges, and neither CCD1 (static) nor CCD2 (slow gravity-only fall) exercises this. Watch specifically for jitter/chatter as the approach velocity crosses the threshold near the surface (a corner's approach velocity oscillating in/out of the gate frame-to-frame is a classic chatter failure mode) — a smooth, non-jittery approach at ordinary speed confirms this untested regime is fine in practice; visible stutter/chatter here means `ccdVelThresh`/`ccdOmega` need retuning, not a code defect.
- [ ] Press the tool into the surface: a smooth, converging `|F|` force response, NO visible dent/crater (Task 1), no oscillation (Task 8).
- [ ] **NEW:** During ordinary contact/pressing (not the deliberate hard-pull-release case below): tear-count stays at 0, and `|F|` never approaches the ORIGINAL bug report's ~2460N blow-up figure (this plan's whole reason for existing) — if it does, this is a regression, not a tuning nit.
- [ ] Fast plunge (quick keyboard tap through the surface): tool does not tunnel through; some correction visibly resists it but the liver does not explode/eject (Task 2).
- [ ] **NEW (collision re-review, 5-agent round, true-tunnel-through angle) — KNOWN, quantitatively bounded, not expected to manifest:** `k_SolveCCDContact` is a discrete per-substep position check, not a true swept/continuous test — a corner (or the tool) moving fast enough to fully cross the shaft capsule within one substep would not be caught. At real scene scale this requires ~400-1800 world-units/second, roughly 100-450x beyond anything gravity, the elastic solve, damping, or realistic tool speeds can produce (the tool geometry is also frozen for the whole frame's ~20 substeps, so tool speed cannot contribute swept motion at this granularity either) — this item exists only so a deliberately extreme stress-test (not ordinary play) has a documented expected outcome; do not expect to trigger it during normal use.
- [ ] **NEW (collision re-review, 5-agent round, jaw-box tunnelling angle):** perform a hard, sudden pull WHILE gripped (triggering a core-pin break, per the item below) several times, immediately after each break watch closely whether the just-released corner ever visibly passes cleanly through the closing jaw's gap without any resisting correction — the jaw-slab-box branch has the SAME discrete-position-check limitation as the shaft branch above, but its tunnelling threshold is much lower (tens to ~160 u/s, since the jaw gap is thinner than the shaft's diameter) and a violent break-triggering tension spike is the one scenario in this plan plausibly close to that range. This exact branch has ZERO automated test coverage (every CCD gate deliberately parks the jaw hinge far away to isolate the shaft branch) — this manual check is the ONLY verification this specific risk gets. If you see a corner cleanly pass through with no resistance, note it as a real, if narrow, gap — the fix (a swept segment-vs-box test) is scoped but deliberately deferred, not silently assumed safe.
- [ ] **NEW (collision re-review, 5-agent round, regression-vs-retired-constraint angle):** this is the intended, deliberate behavior, not a bug — but confirm it doesn't look wrong: let a corner drift into overlapping the shaft SLOWLY (e.g., release a grasp near the shaft and let elastic settling carry a corner into contact, rather than a fast plunge) and confirm it simply stays there, motionless relative to the shaft, rather than ever being pushed back out — the CCD gate deliberately does nothing for a near-zero-velocity embedded corner (that's what Gate CCD1 verifies natively), so a slowly-embedded corner has NO recovery path in this design. This should look like "the tissue rests gently against the tool," not an error — but if it instead looks like a corner is visibly stuck/deformed in a way that reads as wrong, that's worth flagging (it would mean the corner needs some OTHER mechanism, e.g. the elastic solve's own restoring force, to eventually work it free, which this checklist item is verifying actually happens in practice).
- [ ] **NEW (collision re-review, haptics-vs-CCD-consistency angle) — KNOWN, BOUNDED, cross-module lag distinct from the substep-lag item below:** the visual proxy (`haptics.cu`, rendered every frame with ZERO lag — `_haptics.Step()` sets the tool pose and steps in the same call, same frame) and the CCD gate (`k_SolveCCDContact`, consumed inside `LCS_Step`) are on DIFFERENT timing: `LCS_SetGraspTool` writes `g_tool` once per frame but it is only READ by the substep loop at the TOP OF THE NEXT FRAME'S `LCS_Step` — a full FRAME of staleness (not merely the one-SUBSTEP lag the item below describes), confirmed by tracing the actual `Update()` call order. On a fast plunge, this means the proxy the user SEES can already show contact at the tool's current position while CCD is still testing tunnelling against where the tool was one frame ago. Bounded by one frame's worth of tool displacement (not unbounded), and `ccdVelThresh`/`ccdMaxDisp` still gate/cap whatever correction results — but confirm this doesn't look like a visible "the tool visually punched through, then got shoved back a frame later" artifact during a genuinely fast plunge. If it does, this is a real architectural gap (no explicit fix scoped in this plan) worth a follow-up, not a code defect this plan is expected to have already closed.
- [ ] Hold G near the surface: jaws capture a reasonable-sized patch (not just 1 fallback corner, Task 7); tent forms smoothly, holds under mild movement (Task 3/4).
- [ ] **NEW:** with the jaws closed around tissue, look closely along the WHOLE length of the jaw (hinge to tip), not just the tip — `JawSlabWorld`'s analytic contact box uses a SINGLE gap measurement taken at the tip and applies it UNIFORMLY as `gapHalf` along the entire jaw length, while the real jaw mesh (what `haptics.cu`'s SDF actually sees, and what the user visually sees) may taper/curve rather than staying perfectly flat between hinge and tip. The box's WIDTH dimension is a proven, exact superset of the real mesh at any grasp angle (verified: the open/close rotation is about local X, which cannot change the width-bound sampled from raw mesh vertices) — but the gap/tip-vs-hinge profile is NOT similarly proven, only assumed flat. Watch for any visible mismatch between where the analytic contact/grasp logic treats tissue as "inside the closing jaws" and where the real jaw mesh visually appears to actually touch, especially near the hinge end.
- [ ] **NEW:** while gripping, drag the tool sideways a MODERATE distance at ordinary speed (a realistic surgical tug, not the deliberate hard-yank case below) — watch for any faint "ripple"/one-SUBSTEP-lag artifact at the tent's shoulder (uncaptured corners adjacent to the grasp are governed by the CCD kernel using velocity from the previous SUBSTEP, one step behind the grasp constraints' fresher writes WITHIN THE SAME FRAME — a smaller, native-internal timing gap, distinct from the cross-module FRAME-level lag noted above; likely imperceptible given both corrections are small/capped, but worth a visual check during a fast-ish drag).
- [ ] While gripping, pull HARD and fast: the grip cleanly releases (core corner detaches) rather than the whole liver tearing/shattering or sticking permanently (Task 3).
- [ ] **NEW (important, scenario-walkthrough angle):** the plan's two `coreBreakForce` reference points (native gate: ~9.8N ordinary hold vs. a deliberate 500N synthetic yank) are two extremes ~40-65x apart — they say nothing about a MODERATE, realistic surgical tug in between, which is where most real usage will land. Explicitly try a firm-but-not-violent pull (distinct from both the "mild movement" item above and the "HARD and fast" item below) and record which way it goes (holds vs. releases) — this is a genuine tuning checkpoint, not a pass/fail bug: if a moderate, reasonable tug releases too easily (feels fragile) or a clearly-hard pull still won't let go (feels stuck), retune `coreBreakForce` from its (audit-fix, real-scene-scale angle) 150 STARTING ANCHOR default accordingly, the same STARTING-ANCHOR-then-tune convention this whole plan already uses for every other empirical constant.
- [ ] **NEW:** a core-pin break clears `graspCore`/`grasped` for the OVERLOADED CORNER ONLY, not the whole grasped set — so a hard pull can plausibly produce a PARTIAL break (some core corners detach, others and all ring corners stay grasped), leaving a lopsided/asymmetric tent rather than a clean full release. This is a real, designed consequence of per-corner breakability (a gripped tissue tearing unevenly is physically plausible), not necessarily a bug — but confirm it doesn't look broken/glitchy (e.g. one side snapping free while the other stays rigidly anchored in a visually jarring way); if it looks wrong, this is a `coreAlpha`/`omegaGrasp`/geometry tuning question, not necessarily a sign the break logic itself is incorrect.
- [ ] **NEW (important, real-scene-scale angle) — KNOWN LIMITATION this plan does NOT fully close:** while gripping (core corner active), separately press/touch the tissue at or very near the SAME grasped patch with continued contact (not the grip itself — a second point of contact-force injection). `haptics.cu`'s `k_ScatterContact` (Task 1) selects the nearest-active-corner for its force share via a loop that excludes `active==0`/`pinned!=0` but does NOT exclude `graspCore!=0`/`grasped!=0` — since Task 3 gives core corners normal invMass, `hapticForce` now reaches them exactly like any deformable corner, and a worst-case concentrated reaction share (Task 8's own derivation: `k_ScatterContact`'s partition-of-unity weighting CAN route the FULL reaction, up to `hapticReactionMax=600`, onto a single nearest corner) landing on an ALREADY core-pinned corner inflates that corner's `C=dist(pos,target)` the SAME substep, and hence its `coreLambda`/`Fcore` reading — a real path to premature, unintended detachment during ordinary simultaneous grasp+touch play that neither the CORE-BREAK gate (grip-only force) nor any other gate in this plan exercises. If this check reveals spurious detachment under simultaneous contact, the durable fix is to exclude `grasped[c]!=0` corners from `k_ScatterContact`'s corner-selection loop (mirroring the SAME principle already applied to `k_SolveCCDContact`'s rule (d) — never let an independent force-routing mechanism fight the grasp constraints over the same corner) — this is explicitly OUT OF SCOPE for this plan (it would require a new cross-module bridge from `xpbd_solver.cu`'s grasp state into `haptics.cu`, which currently has no dependency on the XPBD module at all) and should be scoped as its own follow-up if this check fails.
- [ ] Release G: liver springs back elastically, no residual freeze/weld artifacts.
- [ ] No console errors/exceptions throughout the whole sequence.

- [ ] **Step 4: Record the verdict**

If ALL native gates pass and ALL Unity checklist items pass: the redesign is DONE — update the `[[xpbd-migration-status]]` memory file to reflect the collision/contact redesign's completion (this is a memory-write step, not a code step — do it as a normal conversation action after this plan finishes, not as a git commit).

If any Unity checklist item fails: that failure is a NEW bug report, not a plan defect — triage via `systematic-debugging`/`diagnose`, do not silently patch inside this plan's scope.
