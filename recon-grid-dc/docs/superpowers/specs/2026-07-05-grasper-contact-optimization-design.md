# Grasper Contact Optimization Plan — 4 Issues (2026-07-05)

> Research wf_7d51fa1e (20 research agents + synthesis) → combined plan → review wf_f181fb1a-equivalent
> (paper-fidelity + feasibility-perf, both found real majors/blocker) → **this doc = the corrected plan**
> incorporating both rounds of review feedback. Ground truth re-verified against HEAD 7825eed.

## Cross-issue dependency map (resolve conflicts up front)

- **D (shaft) must land before B (blocky dent) is re-tuned.** D adds a new collision primitive (shaft
  capsule) into the same `ClosestOnTool` used by every kernel in B's scatter path. If B's multi-corner
  weighting is tuned/verified first using jaw-only geometry, the shaft capsule's addition changes the SDF
  field and re-invalidates that tuning. Order: **D → B**.
- **A (single-spring/misalignment) and C (trigger distance) both touch `dP` optics, but from opposite
  ends.** A's fix clamps the *physical* pose's advance (shrinks worst-case `dP`); C's fix shrinks the
  *detection band* (how early contact starts). These compose cleanly with no conflict: A bounds how deep
  `dP` can get once in contact, C bounds how early "in contact" begins. Validate C's tightened margin using
  A's *already-clamped* physical trajectory — otherwise an unclamped fast key-hold could still blow through
  a tightened band and you'd misattribute a false "still tunnels" to C when it's actually A's gap.
- **B's multi-corner scatter changes the effective visual magnitude of `dentDepthMax`/`contactBeta`.** Once
  B lands, re-eyeball whether `hapticContactBeta=0.85` / `hapticDispMaxOverL=0.6` still look right — B
  redistributes the same clamped depth over more corners, which will look *shallower* per-corner even
  though total volume is conserved. Expected side effect, not a regression — don't re-inflate `contactBeta`
  reflexively.
- **None of A/B/C/D require touching `tearStretchRatio`, Ks/Kb, or the cut-detection stack** (Stage 6/D10) —
  fully orthogonal, no interaction with the cut architecture.
- **C amplifies an incomplete D.** Issue C shrinks `hapticSkinRadius` (the jaw's skin radius), which is also
  the ONLY radius `haptics_detect`/`haptics_phys_inside`/the friction cone currently know about. Until
  Issue D's radius threading reaches all three of those host-side gates (not just the dent/force path — see
  Issue D's "SECOND BLOCKER" note below), shrinking `hapticSkinRadius` makes the jaw/shaft radius mismatch
  WORSE. **Do not finalize Issue C's tightened values until Issue D's detection/release/friction threading
  is confirmed complete end-to-end.**

## Paper deviations this plan introduces or extends (disclosed per review requirement)

The paper models the surgical instrument as **one unified capsule SDF** (lines 778-780: "the surgical
instrument is first approximated as a slender capsule, with its volume represented by a directed distance
field") and states the reaction force is applied "to the particles of the nearby triangle faces" (line 919
— most naturally read as the colliding triangle's own vertices). This project **already deviates** from
both, pre-existing this plan:
- The tool is a **jaw-only triangle mesh** (not one capsule) — `GrasperRig.cs` builds `_collSoup` from jaw
  collision OBJs only; `ClosestOnTool` dispatches mesh-or-capsule, never both.
- The reaction/dent is scattered to the **nearest background-grid corner** (not the triangle's own particle
  vertices) — `k_ScatterContact` argmins over `cornerPos[]`, a separate coarse deformation grid
  (`targetLongAxisVoxels=48`, L≈0.37-0.4u), not the colliding triangle's 3 vertices.

**Issue D (below) extends deviation #1** (jaw-mesh + shaft-capsule, still not one unified capsule, but now
covering the whole tool via a union of two primitives instead of one). **Issue B (below) extends deviation
#2** (K-nearest weighted corners instead of 1, still background-grid corners, not triangle vertices). Both
extensions are justified by external prior art (cited per-issue below), but that prior art justifies the
*engineering choice*, not paper-fidelity — flagging this explicitly so nobody mistakes the corrected
contact stack for closer-to-Algorithm-1 than it is.

---

## Issue D — shaft穿模 (highest severity, do first)

**Adopted root cause:** `ClosestOnTool` (haptics.cu:134-138) is a mutually-exclusive branch:
```cpp
return (toolCount > 0) ? ClosestOnToolMesh(...) : ClosestOnSegment(s, e, A);
```
Since `g_toolTriCount>0` whenever the jaw mesh is uploaded (always, per `GrasperRig.Build()` success), the
capsule branch is permanently dead code. `GrasperRig.cs` builds `_collSoup` from jaw collision OBJs only
(lines 43-48, 70-78) — there is no shaft geometry in the soup at all — and even the dead capsule fallback
uses `hapticProbeLength=1.2` (a short jaw-region probe), not the true shaft span (measured OBJ bounds:
shaft z spans `-25.17` to `+307.0` local units — ~300 units of handle with zero collision representation at
any code path).

**Exact fix — CHEAP fold-in (revised after feasibility-perf review flagged the original "second independent
SDF pipeline" variant as roughly doubling reduction-kernel launch count per `haptics_step`; adopt this
single-pass version instead). REVISED AGAIN after the final review round found a real blocker in the first
draft of this fold-in — see "Pi/outRadius lag" below; the code shown here is the corrected version.**

1. **`cuda_plugin/cuda/haptics.cu`** — extend `ClosestOnTool` to internally evaluate BOTH the mesh candidate
   and a shaft-capsule candidate in the SAME call (no second reduction pass, O(1) marginal cost — an extra
   `ClosestOnSegment` is a handful of dot products, not a kernel launch):
   ```cpp
   // Returns the winning tool-surface point AND its associated radius (out-param), so the caller's
   // downstream depth = radius - signedDist uses the CORRECT radius for whichever primitive won — jaw
   // mesh (skin-inflated, small radius) or shaft capsule (thicker handle radius). Single pass, no
   // duplicate per-triangle pipeline: this is the fix for the feasibility review's flagged cost.
   __device__ __forceinline__ float3 ClosestOnTool(const float3* toolTri, int toolCount, float3 shift,
                                                    float3 s, float3 e,              // jaw capsule fallback (unused when mesh present)
                                                    float3 shaftS, float3 shaftE, float shaftR, float meshR,
                                                    float3 A, float* outRadius)
   {
       float3 meshP  = (toolCount > 0) ? ClosestOnToolMesh(toolTri, toolCount, shift, A)
                                        : ClosestOnSegment(s, e, A);
       float3 shaftP = ClosestOnSegment(shaftS + shift, shaftE + shift, A);
       float meshDepthish  = length3(meshP  - A) - meshR;    // smaller = more binding/closer to touching
       float shaftDepthish = length3(shaftP - A) - shaftR;
       if (shaftDepthish < meshDepthish) { *outRadius = shaftR; return shaftP; }
       *outRadius = meshR; return meshP;
   }
   ```
   **BLOCKER found and fixed (final review round): the `Pi`/`outRadius` one-iteration lag.** The alternating-
   projection loop in `k_ClosestPerTriangle`/`k_NearestSigned` (haptics.cu:168-180, 236-245) captures
   `Pi = P` (the ENTRY value of `P` for this iteration) **before** overwriting `P = Pn` — so the point that
   actually feeds `signedDist`/`depth` downstream (`Pi`) was produced by the **previous** iteration's
   `ClosestOnTool` call (or the pre-loop `P0` call), while the **current** iteration's `ClosestOnTool` call
   produces `Pn`, which is discarded whenever the loop breaks this same iteration. A naive `outRadius`
   write on every call therefore reports the radius from the call that produced the discarded `Pn`, not the
   call that produced `Pi` — exactly wrong at the jaw/shaft handoff (near the hinge, where both primitives
   are comparably close and the winning primitive can flip between consecutive iterations, which is
   precisely the scenario this fix exists to get right). **Fix: thread the radius through the SAME lag
   pattern as `P`/`Pi` itself** — a shadow variable updated in lockstep, not read fresh at loop-exit:
   ```cpp
   float rP0;
   float3 P = ClosestOnTool(toolTri, toolCount, toolShift, s, e, shaftS, shaftE, shaftR, meshR, ctr, &rP0);
   float radiusOfP = rP0;
   float3 A = a, Pi = P; float radiusOfPi = radiusOfP;   // capture radius alongside Pi from the START
   #pragma unroll 1
   for (int i = 0; i < 100; i++)
   {
       A = ClosestOnTriangle(a, b, c, P);
       float rPn;
       float3 Pn = ClosestOnTool(toolTri, toolCount, toolShift, s, e, shaftS, shaftE, shaftR, meshR, A, &rPn);
       float dPrev = length3(P - A);
       float dNew  = length3(Pn - A);
       Pi = P; radiusOfPi = radiusOfP;      // capture BOTH together — same lockstep lag as P/Pi already has
       P = Pn; radiusOfP = rPn;
       if (fabsf(dPrev - dNew) < eps) break;
   }
   // downstream: float depth = radiusOfPi - signedDist;   (was the constant `radius` parameter)
   ```
   This is still O(1) marginal cost (one extra float shadowing `P`/`Pi`, no new kernel/reduction) and now
   correctly pairs `Pi` with the radius of the primitive that actually produced it, at every iteration
   including the jaw/shaft handoff. Every call site (`k_ClosestPerTriangle`, `k_NearestSigned`,
   `haptics_detect`, `haptics_phys_inside`, `haptics_separate`, the physical-pose re-query) must thread
   through the new shaft segment + both radii AND this shadow-radius lockstep pattern — mechanical, ~6 call
   sites, budget for it, but each site's *marginal* cost is one extra `ClosestOnSegment` call plus one extra
   float local, not a duplicated pipeline.
   **Updated Verify requirement (the original Verify step only checked pure-jaw and pure-shaft regimes, not
   the handoff where this bug lived): also drive the tool so the CONTACT POINT crosses the jaw/shaft
   boundary near the hinge over several frames, and confirm depth/force/dent stay continuous (no snap/pop)
   as the winning primitive flips — this is the regression test for the lag bug specifically.**

   **SECOND BLOCKER found and fixed (a follow-up review round, after round 2's fix landed): the
   shadow-radius above only reaches `k_ClosestPerTriangle`'s own `depth = radiusOfPi - signedDist`
   computation — i.e. the DENT/FORCE path. It does NOT reach the two HOST-SIDE gates that decide contact
   detection and proxy release, which is where the shaft-tunneling complaint (Issue D's own reason for
   existing) actually lives:**
   - `haptics_detect` (`return wk < g_toolRadius + margin;`) compares the GPU-reduced winning distance `wk`
     against the single scalar `g_toolRadius` at the HOST — never per-primitive.
   - `haptics_phys_inside` (`return win.x < g_toolRadius;`) same defect, plus the friction cone's
     `cappedDepth = fminf(depth, g_toolRadius)` (the Coulomb-cone saturation radius) — none of these three
     were in the original "~6 call sites" list, because the shadow-radius pattern lives entirely inside
     `k_ClosestPerTriangle`'s per-triangle loop and never reaches a REDUCED (winning-triangle) host read.
     Since `g_toolRadius` is always `hapticSkinRadius` (~0.2, shrinking further under Issue C) whenever the
     jaw mesh is uploaded — which is always — shaft contact would still be **detected and released** using
     the jaw's tiny skin radius instead of the shaft's true ~`2.5*_scale` radius, even after the dent/force
     path is correctly primitive-aware. The shaft can still penetrate substantially before contact
     registers, and the persistent proxy can release prematurely while the shaft is still buried — the
     shaft-tunneling complaint would only be PARTIALLY fixed.
   - **Fix:** thread the winning primitive's radius all the way to every host-side scalar comparison:
     1. `k_NearestSigned`'s `out[t] = make_float4(signedDist, 0.f, 0.f, axisToTri)` currently leaves `.y`/
        `.z` unused (hardcoded 0) — pack `radiusOfPi` into `.y`. `haptics_phys_inside` already does
        `cudaMemcpy(&win, &d_triVms[minTri], ...)` (a full float4 readback of the winning triangle) — change
        `return win.x < g_toolRadius;` to `return win.x < win.y;`. Zero new buffers, zero new kernel calls.
     2. `k_ClosestPerTriangle`'s `triVms[t] = make_float4(vms.x, vms.y, vms.z, key)` uses all 4 float slots
        (push vector + key) — no free slot. Add a companion `float* d_triRadius` buffer (`[triCap]`, alongside
        the existing `d_triVms`), written by `k_ClosestPerTriangle` as `d_triRadius[t] = radiusOfPi;`.
        `haptics_detect` does not currently identify a winning TRIANGLE (only a winning key value via
        `k_ReduceVmsKey`) — add the existing `k_PickVmsTri` call (already used elsewhere for this exact
        purpose) to get `minTri`, then read `d_triRadius[minTri]` back to host and compare
        `wk < triRadius + margin` instead of `wk < g_toolRadius + margin`. One extra kernel launch + one
        extra `cudaMemcpy(sizeof(float))` per FREE-branch frame (haptics_detect only runs when not already
        in contact) — negligible, not inside the 128-iteration separation loop.
     3. The friction cone's `cappedDepth = fminf(depth, g_toolRadius)` runs in `haptics_step`'s IN-CONTACT
        branch, which has `g_minTri` already set by the preceding `haptics_separate` call (the winning
        triangle for the current proxy pose) — read `d_triRadius[g_minTri]` (one more small `cudaMemcpy`,
        already paying a handful of these per frame in this same function) and use it in place of the
        constant `g_toolRadius`.

   **THIRD BLOCKER found and fixed (a second follow-up review round — same underlying pattern as the
   second blocker, extended to 2 more sites the previous fix list still missed):** the physical-pose
   re-query inside the dent block (`haptics.cu:661-663`, `k_ClosestPerTriangle(..., g_toolRadius, 0.f, ...)`)
   feeds TWO MORE downstream host reads that still used the constant `g_toolRadius`, neither named in the
   fix list above:
     - `haptics.cu:679`: `float proximityMax = (minDist<1e29f?minDist:0.f) + g_toolRadius;` — the dent
       FOOTPRINT band gating `k_CountPenetrating`.
     - `haptics.cu:690`: `g_toolRadius` passed as `k_ScatterContact`'s `dentDepthMax` — the per-corner dent
       depth clamp (BUG-A's bound).
   Both are fed by the SAME re-query as the friction cone fix above, so the fix is the same pattern applied
   once more: this re-query does not currently call `k_PickVmsTri` — add that call (right after
   `k_ReduceVmsKey`, before computing `minDist`) to get the winning triangle, read
   `d_triRadius[minTriDent]` once, and use it in place of `g_toolRadius` at BOTH line 679 and line 690.
   One more small kernel call + `cudaMemcpy` per in-contact frame — same cost class as the other three
   fixes, not inside the 128-iteration separation loop.

   **Implementer trap to avoid:** `k_ClosestPerTriangle`'s existing `radius` PARAMETER (distinct from the
   module-global `g_toolRadius` passed as its argument) is used for TWO different purposes today — (i) the
   AABB-lite CULL bound at `haptics.cu:164` (`ctr - toolQueryCenter > boundR + radius + captureMargin + triR`
   — a conservative distance check deciding whether to even evaluate a tissue triangle at all), and (ii) the
   exact DEPTH calc at line 196 (`depth = radius - signedDist`, now superseded by `radiusOfPi` under this
   fix). Do NOT remove the `radius` parameter entirely — keep it (set to `fmaxf(meshR, shaftR)`, the LARGER
   of the two primitive radii) for the CULL check only, where a conservative over-approximation is safe
   (worst case: a few extra triangles evaluated, never a missed contact); the DEPTH calc is what must use
   the per-primitive `radiusOfPi`/`triRadius`, not the cull bound.

   **Closing statement (to stop this from becoming a 4th/5th/Nth round): every scalar use of `g_toolRadius`
   in `haptics.cu` downstream of a triangle-selection must become the winning triangle's own `triRadius`
   via this SAME lookup pattern (`k_PickVmsTri` → `d_triRadius[minTri]`). The complete, exhaustive list after
   this round: (1) `k_ClosestPerTriangle`'s internal per-triangle depth calc — covered by the round-2
   shadow-radius fold-in; (2) `haptics_detect`'s `wk < radius + margin` — covered; (3) `haptics_phys_inside`'s
   `win.x < radius` — covered; (4) the friction cone's `cappedDepth` — covered; (5) the dent's `proximityMax`
   — covered by this round; (6) the dent's `dentDepthMax` — covered by this round. The only REMAINING
   `g_toolRadius` reference in `haptics.cu` is the assignment site itself (`g_toolRadius = toolRadius;` in
   `haptics_set_physical_tool`, not a threshold comparison — N/A, nothing to fix). Before implementing,
   re-grep `haptics.cu` for `g_toolRadius` one more time and confirm every hit is either this assignment or
   already threaded through `d_triRadius`/the shadow-radius pattern — do not add a new scalar-radius
   comparison without going through this same lookup.**

   **Updated Verify (final form):** explicitly test SHAFT-ONLY contact (jaws clear of tissue, shaft driven
   into the liver) and confirm (a) `g_inContact` flips and `haptics_phys_inside` stays true at penetration
   depths BETWEEN `hapticSkinRadius` and the true shaft radius (detection/release), (b) the friction/grip
   feel saturates at the shaft's own radius, not the jaw's, (c) the visible DENT at a shaft contact point is
   not anomalously shallow/narrow compared to a jaw-tip dent of the same true penetration depth (footprint/
   depth-clamp) — this last check is the regression test for the 679/690 gap specifically, since a fix that
   only covered detection/release/friction would look "done" while still silently under-denting the shaft.

   **Cross-issue note (added because this compounds with Issue C, not called out in the original dependency
   map):** Issue C's fix SHRINKS `hapticSkinRadius` further (0.2 → ~0.13-0.14) — until the three consumers
   above are made primitive-aware, shrinking `hapticSkinRadius` makes the jaw/shaft radius mismatch WORSE,
   not neutral (the detection/release/friction gates would gate shaft contact even more incorrectly against
   an even-smaller jaw radius). **Do not finalize Issue C's tightened margin values until Issue D's
   detection+release+friction radius threading (all three consumers above) is confirmed complete
   end-to-end** — not just the dent/force path from round 2.

   Deviation disclosure: this shaft+jaw union (two primitives, min-depth-wins) is a further departure from
   the paper's single-unified-capsule tool model (paper lines 778-780), extending the pre-existing jaw-mesh
   deviation — justified by coverage necessity (the paper's own single-capsule model would also need this
   same union logic if it modeled a jointed jaw+shaft tool at all, which it does not).

2. **`Assets/ReconGridDC/Cuda/GrasperRig.cs`** — add a method exposing the shaft capsule in world space,
   mirroring `CollisionSoupWorld`:
   ```csharp
   // Shaft is a straight capsule from the hinge (z=-17.5) out to the handle end (z=+307), local-to-world
   // via the same rot/pos as the jaw mesh. Radius from the measured OBJ half-extent (x/y half-width ~2.5).
   public void ShaftCapsuleWorld(Vector3 pos, Vector3 axis, out Vector3 worldS, out Vector3 worldE, out float radius)
   {
       Quaternion rot = Quaternion.LookRotation(axis.sqrMagnitude > 1e-8f ? axis.normalized : Vector3.forward, Vector3.up)
                        * Quaternion.AngleAxis(axisRoll, Vector3.forward);
       Vector3 hingeLocal  = ToLocal(new Vector3(0f, Y_OFFSET, JAW_HINGE_Z));
       Vector3 handleLocal = ToLocal(new Vector3(0f, Y_OFFSET, 307f));   // shaft handle end, measured bbox
       worldS = pos + rot * hingeLocal;
       worldE = pos + rot * handleLocal;
       radius = 2.5f * _scale;   // measured shaft half-width (x[-2.5,2.5]), scaled like the mesh
   }
   ```
   Call this alongside `CollisionSoupWorld` in `LiverCudaManager.StepHaptics()`, and pass both segments +
   radii through a small extension to the existing `SetToolMesh` upload path (add 2 float3 + 1 float to the
   P/Invoke signature, or a new `LCS_SetShaftCapsule(s,e,r)` call — either is a small ABI addition, bump
   `LCS_HapticAbiVersion` accordingly per the existing ABI-guard pattern).

3. **Do NOT** use the orphaned `Assets/StreamingAssets/Haptic_grasper_shaft_collision.obj` as the primary
   fix — valid zero-authoring fallback if the analytic-capsule route hits schedule pressure, but a faceted
   mesh for a simple cylindrical shaft adds GPU cost and precision for no benefit over an analytic capsule.

**Verify:** drive the shaft (jaws well clear of tissue) directly into the liver and confirm (a) `g_inContact`
flips to CONTACT before the shaft visually passes the recon surface by more than the capture margin, (b) a
non-zero bounded dent appears at the shaft contact point. Regression: re-run jaw-only contact (tip pressed
in, shaft in open air) and confirm force/dent unchanged (the union must prefer the closer/more precise mesh
candidate there — meshDepthish should win when the jaw is actually the nearer primitive).

**Prior art:** Zilles & Salisbury 1995 (god-object surface-agnostic to primitive count); Ruspini/Kolarov/
Khatib SIGGRAPH 1997 (decompose complex tool geometry into simpler primitives, full mesh only where
precision matters); CHAI3D `cToolCursor`/`cToolGripper` (per-child min-reduction across multiple haptic
points/primitives — tip cursor distinguished from shaft cursors along the tool axis, matching this fix's
tip-mesh + shaft-capsule split); Basdogan/Ho/Srinivasan MMVR VI 1998 (5-DOF line-segment tool rendering for
laparoscopic instruments — the classical justification for a slender tool as an axis capsule).

---

## REVISION NOTE (round 2 — 2026-07-05, after user re-test + re-clarification)

The A/B/C sections below **REPLACE** the earlier round-1 A/B/C. Issue **D stays exactly as written above —
do not re-open it.** What changed and why:

- **User decisions (final, binding):** (1) keep the red ghost **visible** — do NOT fade/hide it (this
  retires round-1's "Open decision #1" — the answer is a hard NO, drop the fade suggestion entirely).
  (2) **Performance is a non-constraint** (RTX 3090) — every "measure baseline cost first / follow-up
  ticket / block on perf" hedge from round 1 is **withdrawn**; always pick the highest-QUALITY option.
  (3) Trigger distance must get **as close to 0 as possible** (like the reference video), which round-1's
  fixed-margin shrink provably cannot reach — Issue C is re-scoped to **swept-segment CCD**.
- **New root-cause finding that reframes Issue A (verified by direct read at HEAD 3d8cfec, this session):**
  the dent is a **soft additive push the Ks springs then RECOVER**, applied in the **wrong solver phase**.
  Round 1 treated A purely as a `dP`/ghost-alignment optics problem (rate-limit `_hapticCenter`). That
  optics fix is KEPT (it is still correct and still wanted), but it does **not** make the liver hold a
  dent. The actual "can't deform the liver" complaint is a constraint-vs-spring TOPOLOGY problem — see
  A-Part 1 below. The round-1 A section is therefore **subsumed**, not discarded: its `_hapticCenter`
  clamp becomes A-Part 0 (optics), and two new parts (held non-penetration projection; real two-jaw
  grasp) are added.
- **The reference video could not be decoded in this environment** (no ffmpeg/network; 117s, 1518×1080).
  A2's grasp design targets **standard laparoscopic grasper realism** (jaws close, tissue compressed and
  held BETWEEN the two jaws, friction resists slip-out, tissue springs back on release) as the goal, not a
  frame-by-frame match. **Flag to user:** if the video shows something materially different from that
  standard behavior, A2's slab+attachment parameters may need revisiting.

---

## Issue A — 器械无法让肝脏形变 (deformation does not HOLD) + real two-jaw grasping

**Adopted root cause (VERIFIED by direct read at HEAD 3d8cfec — three independent defects, not one):**

**A-cause-1 — the dent is a soft additive push the springs erase (the CORE of "can't deform the liver").**
End-to-end trace, all confirmed this session:
- Per contact frame `d_contactDisp` is `cudaMemset` to 0 (haptics.cu:686) — a FRESH per-frame push, not an
  accumulator.
- `k_ScatterContact` (haptics.cu:325-329) writes `contactDisp[best] += nOut*(-contactBeta*dentDepth/3)`,
  `dentDepth = fminf(depth, dentDepthMax=g_toolRadius)`, `contactBeta=0.85`, routed to the **single nearest**
  deformable corner (argmin, haptics.cu:298-307).
- `k_ApplyContactDisp` (haptics.cu:345-350) does `cornerPos[best] += clamp(disp, dispMax=contactDispMax)`
  then `vel *= 0.15` — it only **damps** velocity, it never **holds** position or sets a target.
- **Crucial ordering (confirmed in `LiverCudaManager.Update`):** `LCS_Step` (the 7-stage DOPRI RK45 spring
  solve, `k_IntegrateY5` at physics.cu:561) runs at the **TOP** of `Update()` (line 522); `LCS_Finalize`
  runs next (line 557); `BuildOrUpdateMesh` (the RENDER) at line 559; `StepHaptics()`→`haptics_step` (the
  dent write) runs LAST (line 562) — i.e. AFTER the surface was already reconstructed and rendered this
  frame (this is exactly why A-Part 1's projection must reorder the render, see the Hook section below).
  So NEXT frame's RK45 runs `k_AccStructural` (physics.cu:552-553, `fs = -ks*(len-L0)*dir` recomputed
  every stage from the current `cornerPos`) **UNOPPOSED** against the displaced corner and pulls it back
  toward the rest lattice. Net steady-state dent = equilibrium of (per-frame push `contactBeta*dentDepth`)
  vs (Ks spring recovery). **At a gravity-safe high Ks the springs recover ~as fast as the push applies →
  shallow-to-invisible net dent = exactly the user's complaint.** The `vel*=0.15` damp slows the rebound
  but a damped spring still returns to rest — it does not change the POSITION equilibrium. **No value of
  `Kp`/`Dp`/`reactionMax` touches this** (`Kp*dP` at haptics.cu:646 drives only the felt HANDLE force;
  `dP=proxy-physical` is pure-kinematic upstream — round-1 G5 confirmed). This is a topology problem, not a
  force-magnitude problem.
  - *Note on the RK reaction path:* `hapticForce` IS scattered into the RK slope (physics.cu:557,
    `k_ComputeSlope(..., d_hapticForce, ...)`) and IS consumed by next frame's solve — so a sustained
    `Fext` exists and competes with `Fint` inside the integrator. But it is capped by `reactionMax` and,
    being a bounded force fighting a stiff Ks, reaches only a shallow equilibrium. The dent's VISIBLE depth
    comes almost entirely from the `k_ApplyContactDisp` position write — the part the springs then erase.
    So the force path alone cannot rescue the dent. **KEEP the force scatter** (orthogonal, still wanted
    for feel); the fix is to stop letting the springs erase the *position*.

**A-cause-2 — grasping is FAKE.** `haptics_set_grasp` (haptics.cu:706-710) only sets `g_graspMu=1+5*grasp01`
(a friction-cone multiplier); `GrasperRig`/`StepHaptics` only lerps `_grasp01` and animates the jaw
open-angle (LiverCudaManager.cs:594-596). The collision is a **single god-object over the whole jaw soup** —
closing the jaws does **not** compress/hold tissue BETWEEN two opposing jaw surfaces. There is no
two-jaw squeeze anywhere. This is the architectural gap the reference video needs closed.

**A-cause-3 (round-1 finding, KEPT as A-Part 0) — the `_hapticCenter` open-loop integrator lets `dP`/ghost
separation grow unbounded.** `_hapticCenter += move*(hapticMoveSpeed*dt)` (LiverCudaManager.cs:589) has zero
rate limit; `dP = ProxyPos - _hapticCenter` then grows without bound in contact (the `|dP|=2.32u` screenshot).
This is an OPTICS defect (ghost drifts far from the steel proxy), independent of A-cause-1's deformation
defect.

---

### A-Part 0 (optics — KEPT verbatim from round 1): rate-limit `_hapticCenter` in contact

Unchanged from the round-1 Issue A fix. In `LiverCudaManager.StepHaptics()`, after computing the candidate
`_hapticCenter` (line 589), clamp its offset from the proxy while in contact:
```csharp
Vector3 candidate = _hapticCenter + move * (hapticMoveSpeed * Time.deltaTime);   // keyboard branch
if (_haptics.Ready && _haptics.InContact)
{
    Vector3 proxyPos = _haptics.ProxyPos;
    Vector3 off = candidate - proxyPos;
    float maxOff = 1.25f * hapticToolRadius;      // reuse the cappedDepth precedent (haptics.cu:585)
    float offLen = off.magnitude;
    if (offLen > maxOff) candidate = proxyPos + off * (maxOff / offLen);
}
_hapticCenter = candidate;
```
Bounds displayed/felt `dP`. **One-frame lag is accepted** (`InContact`/`ProxyPos` are last-frame values,
overwritten inside `_haptics.Step()` which runs after this clamp — worst case one extra `~0.08u` advance,
negligible vs the `1.25*hapticToolRadius` ceiling; same one-frame-lag class as `physDisp`/`g_prevPhysPos`,
ledger D18). **Do NOT raise `Kp`/`Dp`** for this (zero effect on `dP`; Colgate-Brown passivity risk). This
part only fixes ghost alignment — it does NOT hold a dent (that is Part 1). **The red ghost stays visible
per the user's binding decision — the round-1 "fade the ghost" open decision is CLOSED as NO; delete that
suggestion.**

---

### A-Part 1 (the CORE fix): held non-penetration PROJECTION run AFTER the spring solve

Replace the spring-fought additive push with a **held non-penetration position projection** that runs AFTER
the RK45 spring solve each frame, so the springs have already had their say and the projection has the last
word (standard PBD/XPBD collision ordering). This makes the dent depth = tool penetration and **Ks-INDEPENDENT
by construction** — the springs may slide a corner ALONG the tool surface but can never restore it THROUGH
the tool within a tick.

**New kernel `k_ProjectNonPenetration` (in `haptics.cu`), run once per COMMITTED step (never inside RK
stages — that would corrupt the `k_ComputeError` DOPRI estimate at physics.cu:562).** For every deformable
corner near the tool, using Issue-D's tool-shaped surface `ClosestOnTool` (jaw-mesh ∪ shaft-capsule) and the
winning primitive's radius `triRadius` (threaded through by Issue D):
```cpp
// c = corner index. MANDATORY GUARD (review blocker — every position-writing kernel in this codebase
// enforces it: k_ScatterContact:304, k_ApplyContactDisp:340, k_IntegrateY5/Slope/Error): skip anchored
// boundary-condition (pinned) and frozen/severed (inactive) corners, else the held projection would
// plastically drift the anchored BC and displace severed debris. This MUST be the first statement.
if (active[c] == 0 || pinned[c] != 0) return;
// Gate to corners within (triRadius + band) of the tool, same proximity test the scatter path already
// uses, so this is a bounded local pass, not a full-grid scan per corner.
float outRadius;
float3 P    = ClosestOnTool(d_toolTri, g_toolTriCount, /*shift=*/make_float3(0,0,0),
                            g_physS, g_physE, g_shaftS, g_shaftE, g_shaftR, g_toolRadius,
                            cornerPos[c], &outRadius);   // nearest tool-surface point (D's signature)
float3 rel  = cornerPos[c] - P;
float  d    = length3(rel);
float3 nOut = (d > 1e-9f) ? rel / d : lastGoodNormal;    // outward from the tool surface
float  sd   = dot3(cornerPos[c] - P, nOut);              // signed dist: >0 outside, <0 penetrating
float  target = outRadius;                               // stay >= one tool radius off the surface
if (sd < target) {                                       // corner is inside the held floor
    float alpha = g_hp.contactProjectAlpha;              // 1.0 = hard hold; <1 = XPBD-style compliant
    cornerPos[c] += nOut * (alpha * (target - sd));      // push out to the floor
    float vn = dot3(vel[c], nOut);                       // kill ONLY inward normal velocity
    if (vn < 0.f) vel[c] -= nOut * vn;                   // tangential slide preserved (friction handled elsewhere)
}
```
This is a **FLOOR, not a push**: it enforces that a corner cannot be closer than one radius to the tool.
Because it runs AFTER the springs, the projection wins; because it writes no rest-length change, withdrawal
(all `sd >= target`) leaves the constraint inactive and Ks alone recovers the surface elastically → clean
spring-back, **no plastic crater**.

**Gauss-Seidel diffusion sweeps (quality-first, perf is free on the 3090):** run **N=4-6 alternating sweeps**
per committed step of `{ k_ProjectNonPenetration ; local spring-length relaxation over the corner
neighborhood }`. This both stiffens the hold AND diffuses the dent into a smooth basin — **directly sharing
machinery with Issue B** (the diffusion sweep IS B's smoothness mechanism; see Issue B below, which
subsumes the round-1 K-nearest scatter). The local relaxation reuses the existing neighbor topology
(`d_nbrIdx`/`d_restNbr` already uploaded for `k_AccStructural`): a Jacobi/GS length constraint
`cornerPos += 0.5 * (restLen/curLen - 1) * (cornerPos - nbrPos)` per neighbor edge, mirroring the sibling
project's `SolveInternalOnce()` (Simulation/Assets/SurgicalSim/Physics/XPBDSolverGPU.cs:508-513, comment
"内部约束和接触交替求解，让局部压陷能扩散到体内，同时保持最终不穿模").

**Hook — where the projection runs (REVISED per review major: the projection must run BEFORE the mesh the
user SEES is built).** The current `Update()` order is `LCS_Step (springs, line 522) → LCS_Finalize (recon
isosurface rebuild from cornerPos, 557) → BuildOrUpdateMesh (RENDER, 559) → StepHaptics (562)`. If the held
projection is simply appended after `StepHaptics`, it moves `cornerPos` AFTER the surface was already
reconstructed and rendered this frame — so the surface the user sees is the spring-recovered (shallow) one,
and the held dent only appears a frame late AND partially (each rendered frame catches the springs mid-
recovery, before the projection re-deepens). Fix the ordering so the projection is applied, then the recon
+ render are redone from the projected corners. Required per-frame order:
```
LCS_Step (RK45 springs move cornerPos)
LCS_Finalize            # rebuild recon so StepHaptics can read recon_expand_pos for the proxy/detect
StepHaptics             # proxy + force scatter (dent WRITE retired — Part 1 owns the dent now)
LCS_ProjectContact      # held non-penetration floor: N GS sweeps, moves cornerPos (the dent)
LCS_Finalize  (AGAIN)   # rebuild recon from the PROJECTED corners  ← perf is free (3090)
BuildOrUpdateMesh       # render the held-dent surface (now reflects the projection this same frame)
```
- **New native entry `LCS_ProjectContact()`** called from `LiverCudaManager.Update()` after `StepHaptics()`,
  followed by a SECOND `LCS_Finalize()` + `BuildOrUpdateMesh()` (move the existing 557/559 calls to after the
  projection, or add a second pair — the extra recon rebuild is cheap and perf is a non-constraint here). It
  reads the tool pose `StepHaptics` already uploaded and runs the N sweeps against the current `cornerPos`.
  Add the `DllImport` next to the others (LiverCudaManager.cs:250-263), bump `LCS_HapticAbiVersion` per the
  existing ABI-guard pattern.
- **Alternative (b), avoids a second Finalize:** move the FIRST `LCS_Finalize`+`BuildOrUpdateMesh` to run
  AFTER the projection, and have `StepHaptics` read the recon from the PREVIOUS frame (one-frame-stale proxy
  detect — acceptable, the god-object is persistent frame-to-frame). Prefer (a) unless the double Finalize
  is measurably a problem (it will not be on the 3090).

Either way: the projection **runs exactly once per committed step, outside RK stages**, and is the LAST
writer of `cornerPos` BEFORE the recon rebuild the render consumes; next frame's `LCS_Step` re-solves the
springs and the projection re-imposes the floor — a stable fixed point, not a tug-of-war, and the RENDERED
surface shows the held dent the SAME frame it is imposed.

**Retire vs keep in `k_ScatterContact`:** RETIRE the `-contactBeta*dentDepth/3` position write (the dent is
now the projection's job) — or keep it at a tiny `contactBeta` as a **1-frame visual lead-in only**. **KEEP
the force scatter** into `hapticForce` (the RK reaction path is orthogonal and still wanted for feel).

**New params:** `contactProjectAlpha` (default 1.0; expose <1 for XPBD-style compliance), `contactSweeps`
(default 4-6). Both in `HapticParams`; ABI bump.

**Deviation flags (disclose):**
- **This MOVES CLOSER to the paper, not further.** Paper SS3.2 (lines 982-987): "the proxy surgical
  instrument's position is updated in real-time to prevent penetration" + "a reaction force threshold is
  set and applied to update the particle positions, ensuring smooth collision forces **without
  penetration**." That is a **held non-penetration constraint re-run every haptic frame**, not a
  spring-fought push. Algorithm 1 (lines 864-914) is a converging fixed-point loop (`while ||v_ms||>=eps:
  Pt += a*v_ms`) that iterates the proxy OUT of penetration each tick. So Part 1 is an **honest upgrade
  toward the paper** — the round-1 soft-push was the deviation.
- The N-sweep GS diffusion of the dent into the body is **not paper-mandated** (the paper does not specify
  a tissue-side relaxation loop) — it is a disclosed quality addition, justified by the in-repo sibling
  XPBD's own `couplingPasses` loop (same organ) and by projective-dynamics prior art.
- Projecting **background-grid corners** (not the colliding triangle's own vertices) is the **pre-existing
  deviation** already documented for the god-object scatter; Part 1 inherits it, does not newly diverge.

**Composition with D and the existing code:** Part 1 **depends on D** (uses `ClosestOnTool`'s jaw∪shaft
surface + per-primitive `triRadius`) and **reuses B's** neighborhood relaxation for diffusion. It composes
with A-Part 0 (Part 0 clamps the *physical* advance = optics; Part 1 holds the *tissue* dent = deformation;
orthogonal). It composes with C (a tighter/CCD trigger just engages the held floor earlier). It does NOT
touch `tearStretchRatio`/Ks/Kb/cut stack (orthogonal per the dependency map). The `vel*=0.15` global damp in
`k_ApplyContactDisp` can be **removed** once the projection's targeted `vn` kill replaces it (the projection
kills only inward normal velocity, which is more correct — the global damp was a workaround for the
spring-back the projection now prevents).

**Verify:** (a) with Ks at the gravity-safe (heavy-liver, no-sag) value, press the jaw into the tissue and
confirm a **deep, STABLE** dent whose depth ≈ tool penetration and does NOT decay/flicker frame-to-frame
(read back `cornerPos` — the dented corners hold, not oscillate); (b) sweep Ks from soft to stiff and
confirm dent depth is **invariant** (the Ks-independence claim); (c) withdraw the tool and confirm the
surface springs back elastically with **no residual crater** (`cornerPos` returns to rest lattice);
(d) confirm the DOPRI error/adaptive-h is unaffected (projection runs outside RK stages — log `k_ComputeError`
output before/after, must be byte-identical during the non-contact phase).

---

### A-Part 2 (real two-jaw grasp — a NEW, disclosed deviation beyond the paper, justified by the video goal)

Add a **two-jaw PINCH constraint** on top of Part 1 so closing the jaws actually compresses and holds tissue
BETWEEN them (A-cause-2 fix). This is a deliberate deviation — the paper models a single unified capsule and
has no grasp — justified by the reference-video goal and by the **in-repo sibling prior art** (GripperTool.cs,
same organ).

**Geometry (`GrasperRig.cs`):** expose the two jaw planes in world space (mirroring `CollisionSoupWorld` and
the D-added `ShaftCapsuleWorld`): a `JawSlabWorld(pos, axis, grasp01, out Vector3 mid, out Vector3 nJaw, out
float gap)` returning the slab midplane point, the inward jaw normal, and the current gap `g(grasp01)` that
**shrinks toward 0 as the jaws close**. Pass to native alongside the collision soup upload in `StepHaptics`
(ABI bump).

**While `grasp01 > graspThreshold` (e.g. 0.5):**
1. **Slab compression.** For each corner between the jaws, project it into the shrinking slab:
   `dot(cornerPos - mid, nJaw) ∈ [-g/2, +g/2]`. As `g→0` the tissue slab is COMPRESSED between the jaws.
   Run this in the same GS sweep loop as Part 1's projection (it is another position constraint).
2. **Per-corner jaw-frame attachment with Coulomb stick/slip.** On the rising edge of "fully closed" (mirror
   GripperTool's `isFullyClosed && !IsGrasping` at GripperTool.cs:283-294), CAPTURE the grasped corners:
   store each grasped corner's offset in the jaw local frame (an anchor). Each frame, the anchor's world
   target = jaw-frame offset transformed by the current jaw pose → the tissue rides with the tool (grip &
   drag). Apply **Coulomb stick/slip release** by reusing the existing hysteresis math already at
   haptics.cu:577-589 (`muS`/`muK`, `g_slipping` hysteresis, `frictionRadius = cappedDepth*mu`) — applied
   **per grasped corner** against `mu*Fn` instead of to the global proxy anchor. Slip when pulled past the
   cone → tissue slides out of the grip; stick otherwise → real drag. On jaw OPEN (`!wantClose`, mirror
   GripperTool.cs:291-294) RELEASE all anchors → tissue springs back under Ks alone (Part 1's clean
   spring-back).

   **MECHANISM ACCURACY (review major — do not misstate the port):** the in-repo `GripperTool.cs` grasp is
   NOT a Coulomb position constraint — it is a HARD KINEMATIC LOCK: `LockGraspParticle` sets the grasped
   particle's `invMass = 0` (infinite mass, fully pinned to the jaw frame, no slip). What is described above
   is therefore an **adaptation of GripperTool's grasp INTENT**, not a verbatim port — it deliberately uses
   A-Part 1's soft/compliant position-constraint machinery + a Coulomb cone SO THAT tissue can realistically
   SLIP OUT when over-pulled (which the reference video shows and a hard invMass=0 lock cannot do). **Pick
   ONE explicitly:** adopt the **compliant Coulomb attachment** (recommended — gives the slip-out realism;
   the friction cone math already exists) OR the **invMass=0 kinematic lock** (simpler, exact GripperTool
   parity, but NO slip-out — grip is unbreakable until jaw-open). The plan adopts the **compliant Coulomb
   attachment**.
   **TEAR-LAW / GRAVITY REGRESSION to guard (review major — either mechanism can trip the cut system):** a
   grasped corner dragged by the jaws (or held against gravity by a stiff attachment) can over-stretch its
   grid edges to its UN-grasped neighbours past `tearStretchRatio*L` (tau=4.0) → `k_TearOverstretch`
   (cut.cu, gated on `g_toolSet`) would spuriously SEVER those edges = the tissue tears at the grip instead
   of being held. Guard REQUIRED: either (i) exempt edges incident to a grasp-attached corner from the tear
   law (add a per-corner `grasped` flag the tear kernel checks), or (ii) make the attachment COMPLIANT
   enough (soft target, capped per-frame drag) that grasped-edge strain stays under tau*L, or (iii) cap the
   grasp drag velocity. Prefer (i)+(ii) together. This guard MUST be specified before A-Part 2 is
   implemented — without it, grasping a stiff or gravity-loaded liver tears it at the jaws.
3. Keep `g_graspMu` as a **secondary** effect (it still boosts the god-object friction cone); the real hold
   now comes from the slab + attachment constraints.

**New params:** `graspThreshold`, `jawGapClosed` (min gap at full close), grasp attachment `muS/muK` (can
reuse the existing global ones initially). ABI bump. New native entries `LCS_SetJawSlab(mid, nJaw, gap)` and
grasp capture/release triggered from the `_grasp01` transitions in `StepHaptics`.

**Deviation flag (disclose prominently):** two-jaw slab + per-corner attachment is a **NEW deviation** from
the paper's single-capsule, grasp-free model. It is an ADAPTATION (not a verbatim port) of the in-repo
sibling `Simulation/Assets/SurgicalSim/Grasping/GripperTool.cs` ("Phase 2 摩擦夹取 InvMass 锁定法",
CaptureParticles/UpdateGraspedParticles/ReleaseParticles, lines 283-294 & 448-551) — the sibling uses a hard
`invMass=0` kinematic lock; this plan instead uses a compliant Coulomb attachment so tissue can slip out
(see "MECHANISM ACCURACY" above). Ported into the ReconGridDC RK45 mass-spring path, which lacks any grasp.
Justified by the video goal + working same-organ prior art, NOT by paper fidelity.

**Needs USER decision:** (1) grasp is currently driven by holding **G** (LiverCudaManager.cs:595) —
confirm that stays the grasp control. (2) `jawGapClosed` — how much residual thickness the fully-closed jaws
leave (0 = full pinch-through, which can be visually violent; a small positive gap reads as "holding
compressed tissue"). Recommend a small positive default (~0.3×tool radius); flag the exact value for user
approval after they see it against the video.

**Verify:** (a) open jaws, push into tissue, close G — confirm tissue is visibly SQUEEZED between the two
jaw surfaces (not a single god-object dimple); (b) with jaws closed, drag the tool laterally — tissue rides
with the grip (sticks) until pulled past `mu*Fn`, then slips out; (c) open jaws — tissue springs back to
rest, no residual attachment; (d) confirm no interaction with the cut stack (grasp ≠ cut).

**Prior art:** IN-REPO (strongest): `XPBDSolverGPU.Step` elastic→`SolveToolContacts`(after
elastic)→`couplingPasses{SolveInternalOnce();SolveToolContacts();}`→`k_PostSolve` (lines 436-517), tool
contact via XPBD compliance `_ToolContactAlpha=ToolContactCompliance/sdt2` (line 457) — a stiff contact
pinning the surface while soft NeoHookean tissue relaxes around it in one GS loop (exactly Part 1's
stiff-floor-vs-soft-spring, no tug-of-war). `GripperTool.cs` Phase-2 friction grasp (grounds Part 2).
Müller et al. PBD 2007 (JVCIR 18(2):109-118): unilateral collision as a GS position constraint projected
AFTER stretch/bend; implicit `v=(x-x_prev)/dt` bakes the held position into velocity so no restoring force
fights a satisfied constraint. Macklin/Müller/Chentanez XPBD 2016 (MIG): compliance
`α̃=α/dt²` makes a stiff contact and soft tissue coexist in one GS loop. Bouaziz et al. Projective Dynamics
2014 / Tournier et al. Stable Constrained Dynamics 2015: contact as a projection reconciled with elastic
constraints. Paper SS3.2 lines 982-987 + Algorithm 1 lines 864-914 (held non-penetration re-run each tick).
IPC-GraspSim (arXiv:2111.01391): two-jaw pads with per-contact Coulomb stick/slip (grounds Part 2's slab +
attachment).

---

## Issue B — 碰撞不精细/大洞 (blocky dent → smooth tool-shaped dent)

**Adopted root cause (confirmed by direct read):** `k_ScatterContact` (haptics.cu:298-307) routes each
penetrating (triangle, lane) to the **single nearest** deformable corner via a hard argmin — no distance
weighting. Combined with the coarse background grid (`targetLongAxisVoxels=48` → L≈0.37-0.4u), the dent can
only move whole grid corners, winner-take-all per lane → the "few blocky point-pushes" the user calls a
"big hole." Independent of D (which changes WHERE contact is detected) and of A-Part 1 (which changes the
push into a held floor).

**Exact fix (quality-first, perf is a non-constraint):** attack blockiness on **two composing fronts**.

**B-1 — the dent is now DIFFUSED by A-Part 1's GS relaxation sweeps.** The single biggest smoothness win is
already in A-Part 1: alternating `{ non-penetration projection ; local spring-length relaxation }` for N=4-6
sweeps diffuses the held dent into a smooth basin over the neighborhood, exactly as the sibling XPBD's
`couplingPasses` loop does. This **subsumes** the round-1 K-nearest-scatter idea for the projection path:
the held floor is applied per-corner (every corner within the tool footprint is projected, not just one
argmin winner), so there is no winner-take-all anymore, and the relaxation spreads it. **Do NOT also do
K-nearest weighting on the projection** — it is redundant with per-corner projection + relaxation.

**B-2 — finer background grid (now unblocked by the perf non-constraint).** The dent resolution is bounded
by grid corner spacing L. On the 3090, **raise `targetLongAxisVoxels`** (48 → 96 or higher) so L halves and
the held dent conforms much more tightly to the tool shape. This was explicitly discouraged in round 1 for
perf reasons; that objection is **withdrawn** — the user chose quality and a finer grid is the most direct
lever on dent sharpness. (Cost scales ~cubically in corner count; acceptable on the 3090. Verify the RK45
solve still converges within `maxSubsteps` at the finer grid — if not, that is a separate Ks/h tuning
follow-up, not a blocker.)

**B-3 — (only if the FORCE scatter remains argmin-routed) apply K-nearest to the FORCE path.** A-Part 1
KEEPS the `hapticForce` scatter (for feel). If the retained force scatter still argmins to one corner, apply
the round-1 top-K=6 inverse-square weighting **to the force term only** so the felt reaction is also
smoothly distributed:
```cpp
const int K = 6; int idx[K]; float dist2[K]; int nFound = 0;
for (int cptr = 0; cptr < cornerCount; cptr++) {
    if (active[cptr]==0 || pinned[cptr]!=0) continue;
    float3 d = cornerPos[cptr]-v; float dd = dot3(d,d);
    if (nFound < K) { idx[nFound]=cptr; dist2[nFound]=dd; nFound++; }
    else { int worst=0; for(int k=1;k<K;k++) if(dist2[k]>dist2[worst]) worst=k;
           if (dd < dist2[worst]) { idx[worst]=cptr; dist2[worst]=dd; } }
}
if (nFound==0) return;
float wsum=0.f, wK[K];
for (int k=0;k<nFound;k++){ wK[k]=1.f/(dist2[k]+1e-6f); wsum+=wK[k]; }
for (int k=0;k<nFound;k++){ float w=wK[k]/wsum; int c=idx[k];
    float3 fShare = reaction*(depthWeight/3.f)*w;                 // force only
    atomicAdd(&hapticForce[c].x,fShare.x); /* .y .z */ }
```
Weights sum to 1 → total force mass preserved → `reactionMax` clamp still valid. (No perf caveat now — the
full-grid scan is fine on the 3090; if desired, a uniform-grid bucket lookup is a clean optional speedup but
NOT required.)

**Composition:** B-2 (finer grid) is independent and lands anytime. B-1 is literally A-Part 1's sweeps — no
separate code. B-3 is a small addition to the retained force scatter. All compose with D (D's tool surface
is what the projection conforms to) and with C.

**Deviation flag:** routing to background-grid corners (not the triangle's own vertices) is the pre-existing
deviation; neither B-1/B-2/B-3 newly diverges. Do not describe as "the paper's own scheme refined."

**Verify:** press the jaw tip in at a shallow angle; read back `cornerPos` and confirm the dent (i) conforms
to the jaw shape (tool-shaped, not a single spike), (ii) decays smoothly over ≥3-4 corners, (iii) is
visibly sharper at the finer `targetLongAxisVoxels`. Regression: total force magnitude unchanged (B-3
weights sum to 1).

**Prior art:** in-repo `XPBDSolverGPU` `couplingPasses` diffusion (same organ); projective-dynamics /
meshfree inverse-square weighting; Sagardia & Hulin VRST 2016 (multi-primitive contact aggregation); CHAI3D
`cHapticPoint` aggregation.

---

## Issue C — 触发距离尽量接近 0 (near-zero trigger via swept-segment CCD)

**Re-scope (round 2):** the user now wants the trigger distance **as close to 0 as possible** (reference
video). Round-1's fixed-margin shrink **provably cannot reach near-0 without tunneling** (a reactive margin
is either too wide at rest or too narrow on a sudden plunge). Replace it with **swept-segment CCD**, shrink
the skin radius to the watertightness minimum, and coincide the render gizmo with the collision surface.

**Adopted root cause (confirmed by direct read):** the resting trigger band is TWO additive world-unit terms.
- **TERM 1 = `g_toolRadius`** = `hapticSkinRadius=0.2` when the jaw mesh is active (always;
  LiverCudaManager.cs:90, passed at :608 `useMesh ? hapticSkinRadius : hapticToolRadius`). `haptics_detect`
  returns `wk < g_toolRadius + margin`. This is a **geometric watertightness** inflation (contact = tool
  axis within one radius of a face), not a safety buffer. It cannot go to literal 0 (a 0-radius tool slips
  between the coarse tissue triangles).
- **TERM 2 = `margin = fmaxf(captureMargin, physDisp)`**, read ONLY inside the FREE/rising-edge branch
  (`if(!g_hasProxy)`), so it affects **first-touch trigger distance only** — exactly the user's complaint.
  `captureMargin = hapticCaptureMarginOverL(0.25)*Lrt ≈ 0.093u` (a STATIC floor). `physDisp = ||g_physPos -
  g_prevPhysPos||` = LAST frame's displacement (reactive anti-tunnel). At rest `physDisp≈0`, so resting band
  ≈ 0.2 + 0.093 ≈ 0.29u — the 0.093 floor buys nothing at rest but a bigger visible gap.
- **A THIRD, perceived-only contributor:** the proxy gizmo is drawn at `hapticToolRadius=0.9u`
  (LiverCudaManager.cs:512, `Mathf.Max(0.04f, hapticToolRadius)`) while collision uses `hapticSkinRadius=0.2u`
  — a **4.5× render-vs-collision mismatch**. The user sees a fat 0.9u sphere whose surface is nowhere near
  the 0.2u collision surface, so even a perfect trigger looks like it fires "in the air."

**Why a fixed margin cannot reach near-0 without tunneling:** `physDisp` is measured from the PREVIOUS frame
(rear-facing). On a sudden key-down from rest, last frame's `physDisp=0` while this frame jumps
`hapticMoveSpeed*dt ≈ 0.08u` — a one-frame window where the band was not widened and the tool is ~0.08u
inside before detection. That is why round 1 refused margin=0.

**Exact fix — three composed changes:**

**(1) SWEPT-SEGMENT CCD detection (removes TERM 2 entirely; exactly 0 at rest, self-scaling under motion).**
The swept geometry ALREADY exists: `g_prevPhysPos` (updated haptics.cu:696) and `g_physPos` are both tracked;
`physDisp` is literally that segment's length, today thrown away as a scalar. Instead of testing the tool at
one static pose and padding, **sweep the tool over `[prevPhys → curPhys]`** and test whether the swept volume
comes within TERM 1 of the skin:
- Add `g_prevPhysS`, `g_prevPhysE` beside `g_prevPhysPos` (haptics.cu:113); update them in lockstep at :696.
- New device fn `ClosestOnSweptTool`: min over `nsub+1` lerp stations of Issue-D's `ClosestOnTool`, walking
  a shift from `shiftS = (prevPhys − curPhys)` to `shiftE = 0` (the mesh stays uploaded once at the current
  pose; each station just TRANSLATES the query by `shift` — no second upload, reusing the shift mechanism
  `haptics_separate` already uses at :483-486). **Swept vs BOTH primitives** (jaw mesh + shaft capsule),
  comparing against each primitive's own radius (Issue D's `triRadius` threading).
- In the FREE-branch detect, `nsub = ceil(sweepLen / (0.5*radius))`, cap ~32. `nsub=0` at rest →
  byte-identical to today's static query, ZERO extra cost. Station spacing `≤ 0.5*radius` guarantees
  consecutive sampled capsules of radius r centered ≤ r/2 apart OVERLAP → their union covers the continuous
  swept volume with no sub-radius corridor a triangle can hide in → a full-speed one-tick plunge is caught
  by **geometry**, not a padding term.
- Change the band from `wk < g_toolRadius + margin` to `wk < triRadius + eps` (`eps ~1e-3`, a numerical
  epsilon, NOT a motion margin) and **drop `fmaxf(captureMargin, physDisp)` from the FREE branch**
  (:538-539). Seed the rising-edge proxy at the FIRST station that trips (tightens the seed at :549).

**(2) SHRINK `hapticSkinRadius` toward the watertightness minimum** (0.2 → ~0.35*L ≈ 0.13-0.14; test lateral
slide across the jaw tip for leak, then push toward ~0.25*L ≈ 0.09). **Perf is free** → ALSO raise
`targetLongAxisVoxels` (Issue B-2's finer grid → smaller watertight radius allowed → even closer to 0). The
CCD now absorbs all the fast-motion safety the skin radius used to share, so shrinking it is safe.

**(3) DRAW THE PROXY GIZMO AT ITS TRUE COLLISION RADIUS.** Change LiverCudaManager.cs:512 from
`hapticToolRadius`(0.9) to the shrunk `hapticSkinRadius` (~0.13) so the rendered sphere surface COINCIDES
with the collision surface. This is the **single biggest lever on the PERCEIVED resting gap** — without it,
shrinking the collision radius still leaves a fat visual ghost that looks like it fires early. (Round 1
omitted this entirely.) **Note:** the RED PHYSICAL ghost stays visible per the user's binding decision — this
change is about the gizmo RADIUS coinciding with collision, NOT about hiding either gizmo.

**ORDERING / dependency (honor, do not re-derive):** **land Issue D FIRST.** `haptics_detect`'s band and the
swept test must compare `wk` against the **winning primitive's** `triRadius` (Issue D's threading), and the
sweep must be swept-vs-BOTH-primitives. Do NOT finalize the shrunk `hapticSkinRadius` until D's per-primitive
radius threading is complete end-to-end (else the ungated shaft mismatch gets WORSE — see D's cross-issue
note). Composes cleanly with A (A only rate-limits the IN-CONTACT physical advance and holds the dent; the
FREE-branch detect is orthogonal). This CCD **SUPERSEDES** round-1 Issue-C levers #2 (shrink
`hapticCaptureMarginOverL`) and #3 (validate-after-A) — under CCD there is no static floor to tune and no
`physDisp` to bound. Lever #1 (shrink `hapticSkinRadius`) STAYS and gains the render-radius companion (change
3). **Retire `hapticCaptureMarginOverL` from detection** (separation already passes margin=0 at :485).

**Deviation flag:** the paper's trigger is bare `distance < instrument radius` with NO margin (lines 855-857);
its only tolerance `eps=1e-3` is a convergence stopping criterion (Algorithm 1 lines 867/886/908), NOT a
spatial band. So **removing the margin moves BACK toward the paper.** The swept CCD term is a **disclosed
project addition** (same class as the D18 margin it replaces), justified by the IEEE CCD-proxy prior art, not
paper-mandated.

**Needs USER decision:** the final shrunk `hapticSkinRadius` value (and whether to also raise
`targetLongAxisVoxels`, and by how much) — "as small as the mesh watertightness allows" is an empirical
leak-test result, flag the chosen value once tested.

**Verify:** (a) slow press, measure `|dP|`/`proxyY−physY` at the `g_inContact` rising edge → shrinks from
~0.29u to skinRadius-only (~0.09-0.14u), fires the instant the skin touches; (b) force `nsub=0` and confirm
byte-identical to the pre-change static query at rest (degenerate no-op); (c) TUNNEL STRESS: slam the tool
through in one tick at max `hapticMoveSpeed`, then artificially DOUBLE `hapticMoveSpeed` — confirm no tunnel
and `nsub` auto-scales (safety is geometric, not tuned); (d) fast lateral slide across the jaw tip does not
leak at the lowest `hapticSkinRadius`; (e) visually confirm the proxy gizmo now TOUCHES the surface at the
moment of contact (change 3).

**Prior art:** Xu & Barbič ACM TOH 2016 (point/tool-vs-SDF continuous collision via conservative advancement;
frames the fixed margin as exactly what CCD eliminates); IEEE 8798712 "Continuous Collision Detection for
Virtual Proxy Haptic Rendering of Deformable Triangular Mesh Models" + "Proxy Pop-Out" (the direct field
solution — swept-volume proxy trigger replacing a capture-margin floor); Speculative Contacts (Catto); Zilles
& Salisbury 1995 (contact surface IS the constraint surface → standoff = tool radius alone); Ruspini/Kolarov/
Khatib SIGGRAPH 1997 (proxy on the constraint surface); CHAI3D `cAlgorithmFingerProxy`/`cToolCursor` (proxy
radius is BOTH collision AND rendered cursor radius — the precedent for change 3); paper lines 855-857 (bare
`distance < radius`, margin-free) + 867/886/908 (`eps` = convergence, not a band); DeviationLedger D18
(captureMargin = project-only anti-tunnel addition).

---

## Summary table for the implementer (rounds 1+2)

| Issue | File(s) | Function/Kernel | Change | New/changed params |
|---|---|---|---|---|
| D shaft (unchanged) | `haptics.cu`, `GrasperRig.cs`, `LiverCudaManager.cs` | `ClosestOnTool` shadow-radius fold-in + `d_triRadius[]` + all host-side gates radius-aware + `ShaftCapsuleWorld` | jaw-mesh ∪ shaft-capsule, per-primitive radius threaded to every gate | shaft radius `2.5*_scale`; ABI bump |
| A-Part 0 optics | `LiverCudaManager.cs` | `StepHaptics()` | clamp `_hapticCenter` to `ProxyPos + 1.25*toolRadius` in contact | clamp const ~1.25; no Kp/Dp change |
| A-Part 1 held dent | `haptics.cu`, `physics.cu`, `LiverCudaManager.cs` | NEW `k_ProjectNonPenetration` + N-sweep GS relax; NEW `LCS_ProjectContact()` after `StepHaptics`; retire dent write in `k_ScatterContact` (keep force) | held non-penetration floor AFTER spring solve → Ks-independent dent | `contactProjectAlpha`(1.0), `contactSweeps`(4-6); ABI bump |
| A-Part 2 real grasp | `haptics.cu`, `GrasperRig.cs`, `LiverCudaManager.cs` | NEW `JawSlabWorld` + slab compression + per-corner jaw attachment w/ Coulomb stick-slip (reuse haptics.cu:577-589) | two-jaw squeeze + grip/drag + spring-back on open | `graspThreshold`(0.5), `jawGapClosed`(~0.3r); ABI bump |
| B smooth dent | `haptics.cu`, recon grid config | A-Part 1 sweeps (B-1) + finer `targetLongAxisVoxels` (B-2) + optional K-nearest on FORCE only (B-3) | per-corner projection + diffusion + finer grid → tool-shaped dent | `targetLongAxisVoxels` 48→96+; K=6 (force only) |
| C near-0 trigger | `haptics.cu`, `LiverCudaManager.cs` | NEW `ClosestOnSweptTool` + swept-CCD FREE-branch detect; shrink `hapticSkinRadius`; gizmo radius = collision radius | swept-segment CCD (band→`radius+eps`, drop margin); render==collision radius | `hapticSkinRadius` 0.2→~0.09-0.14; gizmo radius→skinRadius; retire `hapticCaptureMarginOverL` from detect |

**Order of implementation:** **D** (per-primitive radius + shaft, prerequisite for A-Part 1 / B / C) →
**A-Part 1** (held projection — the core deformation fix; also provides B-1 diffusion) → **B-2** (finer grid)
→ **C** (swept CCD + shrunk radius + gizmo radius) → **A-Part 0** (optics clamp) → **A-Part 2** (two-jaw
grasp) → **B-3** (optional force-path K-nearest). A-Part 0 and A-Part 2 can slot in any time after their
deps. None touch `tearStretchRatio`/Ks/Kb/cut stack.

**Paper-deviation ledger (round 2 additions, disclosed):**
- A-Part 1 (held non-penetration projection) — **moves TOWARD the paper** (SS3.2 / Algorithm 1); the
  round-1 soft-push was the deviation being corrected. The N-sweep GS tissue-diffusion is a disclosed
  quality addition (not paper-mandated; grounded in the in-repo sibling XPBD).
- A-Part 2 (two-jaw slab + per-corner attachment grasp) — **NEW deviation** (paper has single capsule, no
  grasp); ported from the in-repo sibling `GripperTool.cs`; justified by the video goal.
- B (finer grid + per-corner/K-nearest smoothing over background-grid corners) — inherits the pre-existing
  background-grid deviation; no new divergence.
- C (swept-segment CCD) — **removing the margin moves TOWARD the paper's margin-free trigger**; the swept
  term is a disclosed project addition (replaces the D18 captureMargin).

**Open decisions for the user (round 2 — do not implement without confirmation):**
1. **A-Part 2 grasp control** — confirm holding **G** stays the grasp toggle (LiverCudaManager.cs:595).
2. **A-Part 2 `jawGapClosed`** — residual thickness at full jaw close (0 = full pinch-through, visually
   harsh; ~0.3×tool radius = "holding compressed tissue"). Confirm after seeing it against the video.
3. **C final `hapticSkinRadius`** (and whether/how much to raise `targetLongAxisVoxels`) — empirical
   leak-test result; confirm before shipping as new defaults.
4. **Reference video** — could not be decoded here; if its grasp/friction behavior differs materially from
   standard laparoscopic jaw-compression, A-Part 2's slab/attachment params may need revisiting.

**CLOSED from round 1 (no longer open):** "fade/hide the red ghost" — user decision is **NO, keep it
visible**. "Measure `k_ScatterContact` baseline / block B on a spatial accel structure" — **withdrawn**, perf
is a non-constraint, always pick quality.
