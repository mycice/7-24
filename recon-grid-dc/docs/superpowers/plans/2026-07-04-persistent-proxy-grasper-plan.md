# Persistent Finger-Proxy + Grasper Implementation Plan (rev 2, post plan-review wncvquq3j)

> Implement task-by-task; each task ends with a build + oracle checkpoint and a multi-subagent review
> gate. Do NOT advance a task until its review passes.

**Goal:** Make the haptic tool NEVER penetrate the liver (paper §2.3 god-object guarantee), feel a firm
force that grows with push depth, stop the CONTACT/free flicker, render the assembled grasper as the
on-surface PROXY + a ghost PHYSICAL tool + a spring line, and add open/close jaws with grasping.

**Root cause (diagnosis wf_de18f2f3, repro scratchpad/proxy_tunnel_repro.py):** TWO independent tunneling
causes — (1) the per-frame proxy re-seed to the physical pose (D13, haptics.cu:299-302) destroys the
persistent god-object state; (2) the FREE flag `key=(axisDist<radius)?depth:1e30` (haptics.cu:137) reads
"no penetration" once deeper than one radius, so a fast plunge free-steps through. Inner alternating
projection (Eq 26) is faithful and MUST be kept. Fix = {persistent proxy + captured detection} +
free-space seeding + per-frame re-anchor.

## Global Constraints
- KEEP unchanged: inner alternating projection (P0 centroid axis pt, A=SDF(tri,P), Pn=SDF(axis,A),
  |dPrev−dNew|<eps, i<100), min-depth argmin, a=0.1 outer step, eps=1e-3, outerMax cap (D10), skin-only
  wall skip (D12).
- `ΔP = Proxy − Physical` (Eq 22). `F = Kp·ΔP − Dp·v`, `τ = Ko·θ·k − Do·ω` (Eq 27), damping contact-only.
- NEVER store triangle indices/barycentric anchors for the proxy (soup rebuilt every frame). Re-anchor by
  re-projection of the carried world-space P_t.
- The capture margin is a PER-CALL kernel argument, NOT baked into the shared key — it must not leak into
  the separation loop or the D17 dent pass.
- Ledger every deviation from Algorithm 1.

**Measured OBJ geometry (Assets/StreamingAssets/):** jaws_up/jaws_down x∈[-2.17,2.17] y∈[7.25,10.75]
z∈[-43.00,-17.50]; shaft x∈[-2.50,2.50] y∈[6.40,11.60] z∈[-25.17,307.00]. Tool axis = Z, geometry
Y-offset ≈9. Jaw tip = z≈-43; jaw base/hinge ≈ z=-17.5; shaft handle runs to z=+307.

---

## Task A1: Persistent proxy + captured detection + free-space seeding (the never-penetrate fix)

**Files:** `cuda_plugin/cuda/haptics.cu` (kernel signature + haptics_step), `haptics.h` (HapticParams +=
`captureMargin`, `proxyTrackStep`; new export for contact state), `src/plugin_api.cpp` (contact-state
export), `Assets/ReconGridDC/Cuda/HapticTool.cs` + `LiverCudaManager.cs` (params + native InContact),
`Assets/ReconGridDC/Core/DeviationLedger.cs`. Test: `cuda_plugin/tests/haptics_oracle.py`.

**A1.1 — per-call capture margin (BLOCKER fix).** Change the kernel signature to
`k_ClosestPerTriangle(..., float radius, float captureMargin, float eps, float4* triVms)` and
`key = (axisDist < radius + captureMargin) ? depth : 1e30f`. Every existing call passes `captureMargin=0`
by default; ONLY the new detection call (A1.3) passes a nonzero margin. This keeps the outer separation
loop and the D17 dent pass at the true radius (margin 0) — the band is a detection-only widening.

**A1.2 — persistent state.** Keep `g_proxyS/E/g_proxyPos_dev/g_proxyQuat` across frames; add
`bool g_hasProxy` (init false in haptics_init) and `float3 g_prevPhysPos` (for per-frame physical
displacement). DELETE the D13 re-seed (haptics.cu:299-302).

**A1.3 — detection pass (fast-plunge-safe).** Run `k_ClosestPerTriangle` at the PHYSICAL pose with
`captureMargin = max(baseCaptureMargin, |g_physPos − g_prevPhysPos|)` (baseCaptureMargin ≈ 1.5·L; the
per-frame-displacement term makes a fast plunge always land inside the band, closing root-cause 2 without
a fixed guess). Reduce to `physInBand` (bool) + `physDepthTrue` (depth at margin 0, for reference).

**A1.4 — state machine.**
- **Free** (`!physInBand`): `g_proxy* = g_phys*`; `g_hasProxy=false`. ΔP=0 → F=0 (no phantom force).
- **Rising edge into contact** (`!g_hasProxy && physInBand`): SEED on the NEAR side — if a prior
  free-space proxy exists use it, else start at `g_phys*` — then run the outer separation loop TO
  CONVERGENCE (up to outerMax) and assert the final `‖v_ms‖<eps` (proxy actually reached the surface, not
  merely started moving). Set `g_hasProxy=true`.
- **In contact** (`g_hasProxy`):
  a. **Attraction BEFORE separation** (§2.3 prose line 787-789, NOT in Algorithm 1): `Δ=g_physPos−g_proxyPos;
     move g_proxyS/E/Pos by Δ·min(1, proxyTrackStep/max(|Δ|,eps))` (proxyTrackStep ≈ 0.5·L — bounded so
     the proxy can never leap through the tissue in one frame).
  b. **Separation** = the existing outer loop, querying `k_ClosestPerTriangle` at the PROXY pose with
     `captureMargin=0`; `P_t += a·v_ms` until `‖v_ms‖<eps` or outerMax. Because the proxy is on the near
     side, `Pi−A` is genuinely outward.
  c. **Lost-support safety** (cut deleted the supporting tri / surface receded): if the proxy best-tri key
     is 1e30 while `physInBand`, re-seed `g_proxy*=g_phys*` and run the FULL outer loop (up to outerMax),
     not a single step.
- **Contact predicate for FORCE/DAMPING** (decoupled from the band, paper "damping only during contact",
  line 929): `g_inContact = (g_hasProxy && |ΔP| > eps)`. The band gates ONLY seeding, never Eq-27 damping.

**A1.5 — outward-sign guard (narrow).** In `k_ClosestPerTriangle` keep `nOut = normalize(Pi−A)` as the
separation direction; ONLY when it is degenerate (`nlen<1e-6`) OR provably interior
(`dot(Pi−A, faceNormal) < 0`) fall back to the face normal `sign(dot(Pi−A,fn))·fn`. This preserves the
paper's minimum-separation semantics (D15) for the normal outside case and only guards the post-cut
interior case. Update the oracle `alt_project()` + D15 note in lockstep.

**A1.6 — native contact-state export.** Add `LCS_GetHapticContact()` (or extend GetProxyTool) returning
native `g_inContact`; `HapticTool.InContact` reads THIS, not `Force.sqrMagnitude>1e-6` (fixes the C#/native
1-frame disagreement that drives the orange tint + grasp-enable).

**A1.7 — force/dent.** `g_force = ΔP·Kp − g_linVel·Dp` (ΔP now large; keep reactionMax cap). `g_torque`
unchanged (θ=0 under q_B=q_A, D14). Keep the D17 dent pass at physical pose with `captureMargin=0`
(unaffected by the band; still ≈ physical penetration). `g_prevPhysPos = g_physPos` at the end.

**Ledger:** rewrite **D13** (re-seed → persistent proxy; the "identical steady-state ΔP" claim held only
for depth<r = the bug); re-map **D16** ("Move Proxy to the Physics Tool Position" = the bounded attraction
A1.4a; the loop fixed point is the tissue SURFACE ‖v_ms‖<eps, NOT the physical pose); restate **D14**
(q_B=q_A → restoring torque 0); add **D18** (captureMargin — detection-only widening sized to per-frame
physical displacement; does NOT change the Eq-27 contact predicate); add **D19** (proxyTrackStep — the
§2.3 PROSE coupling line 787-789, explicitly ABSENT from Algorithm 1, bounded < tissue thickness, applied
before the separation loop which remains the binding constraint); add **D15** interior-side guard note.

## Task A2: Regression test
`test_persistent_proxy_no_tunnel` in the oracle (promote proxy_tunnel_repro.py): closed-surface fixture +
deep firm push asserts (a) proxy never crosses interior; (b) |F| grows monotonically with depth; (c) zero
contact-flips across a straight push; (d) proxy re-anchors across a simulated soup rebuild / tissue
translation; (e) free-space motion → F=0; (f) fast plunge (physical delta > radius in one frame) is still
captured (band = max(base, |Δphys|)) and the proxy stays outside. Source-gate: D13 re-seed line gone,
`g_hasProxy` + per-call `captureMargin` present, separation call passes margin 0.

**CHECKPOINT + REVIEW GATE 1** (A1+A2 vs paper + verifier spec). Must PASS before B1.

## Task B1: Assemble grasper mesh + dual display + spring
**Files:** create `Assets/ReconGridDC/Cuda/GrasperRig.cs`; modify `LiverCudaManager.cs`.
1. Runtime-parse the 3 OBJs (v/vn/f). Build a root with children shaft/jaws_up/jaws_down.
2. **Recenter + scale (3-axis, from measured bboxes):** translate assembled mesh by pivot `(0, −9, +17.5)`
   so the Y-offset (≈9) → 0 and the jaw hinge (z=-17.5) → origin; the tool axis (Z) then coincides with the
   capsule axis. Uniform scale `= rodLength(12u) / jawSpan` where jawSpan = jaw length ≈25.5u (z -43→-17.5),
   so the jaw region maps onto the ~12u contact capsule; the shaft handle scales with it (extends back,
   off the contact zone). Rotate so local +Z aligns with `_hapticAxis`.
3. Two instances: **Proxy** (opaque, driven by `_haptics.ProxyPos/ProxyRot`, orange on native InContact)
   and **Physical** (semi-transparent/wireframe ghost, driven by the physical pose).
4. `LineRenderer` **spring** proxy-jaw-tip → physical-jaw-tip (Eq-22 ΔP made visible). Remove the old
   capsule LineRenderer.

**CHECKPOINT + REVIEW GATE 2.** Must PASS before C1.

## Task C1: Open/close jaws + grasping (paper EXTENSION — no paper algorithm exists)
**Files:** `GrasperRig.cs` (jaw animation), `LiverCudaManager.cs` (input+wiring), `haptics.cu`/`.h` +
`plugin_api.cpp` (grasp input + hold kernel).
1. Jaws open/close on a key (G hold = close): rotate jaws_up/jaws_down about the hinge pivot (local z=0
   after recenter = OBJ z=-17.5) between openAngle and 0.
2. **Grasp hold-constraint (fully specified):** native input `haptics_set_grasp(bool closed, float jawHalfLen,
   float jawRadius)`. Gripped corners = corners whose nearest point on the JAW capsule (proxy tip segment,
   half-length jawHalfLen) is within jawRadius AND active AND non-pinned. While closed: each gripped corner
   gets a target = its rigid offset from the proxy jaw at grip time, applied as a bounded position
   constraint (like k_ApplyContactDisp but toward the jaw, clamped per frame) + a hold force; store the
   grip offset on the rising edge of closure, release (clear) on open. Moving the closed grasper drags the
   gripped tissue; opening releases it.
3. **Ledger D20 (paper EXTENSION, honest):** the paper DEMONSTRATES grasping (Fig 3.11/3.12 captions) but
   specifies NO grasping algorithm; k_GraspHold is a NEW hold-constraint reusing the §2.3 pinned-skip /
   contact-patch INFRASTRUCTURE, NOT the §2.3 outward-separation reaction. Bounded per-frame, release on open.

**CHECKPOINT + REVIEW GATE 3.** Must PASS.

## Final: whole-feature review (A+B+C) vs paper §2.3 + Fig 2.11/3.11; progress.md entry; deploy note.
