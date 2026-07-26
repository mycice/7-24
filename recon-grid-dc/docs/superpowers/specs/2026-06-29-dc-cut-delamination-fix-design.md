# DC Cut Delamination Fix — Design Spec (final: differential temporal lag)

**Date:** 2026-06-29
**Status:** Implemented → Unity A/B verification
**Decision provenance:** five multi-agent workflows. The first three converged on an External-FP hypothesis (box-center / per-component split); a faithful **CPU-mirror** investigation (two independent mirrors, HIGH confidence) then **refuted** that hypothesis as a no-op and identified the true cause. This spec records the corrected diagnosis.

## Problem

After a horizontal blade cut of the demo sphere, the render **delaminates** (分层): a correct rounded upper dome + a dark gap + a thin nearly-flat sliver below, instead of a clean upper/lower split. Plus: the cut **seam** should be near-zero width.

## Root cause (CPU-mirror confirmed, HIGH confidence)

**Differential temporal lag between the outer skin and the cut wall.**

A cut boundary voxel renders two surfaces:
- **Outer skin** — `_CutFPExternal` (per-component isosurface QEF) → `DC_Stitch`. Recomputed fresh every frame; **no temporal smoothing** → tracks the falling/spinning piece **instantly**.
- **Cut wall** — `_CutFP` (per-component cut-point centroid) → `BuildCutTriangles`. Reconstructed in the owner's material frame, then **EMA-smoothed** by `InterpCutFP` with α = `cutFPInterp` = **0.5** (the 2D-2b feature).

Under the demo's sustained post-cut motion (free-fall + a one-shot spin ω=3 rad/s, ReconGridManager.cs:269-308), the un-smoothed skin tracks instantly while the EMA-smoothed wall **trails** it by a steady-state lag ≈ `v·(1-α)/α` (≈ one frame-displacement at α=0.5, accumulating as the piece falls). Result: **rounded skin dome on top, lagged flat wall sliver below, dark gap between** = the observed delamination.

The code **already documents this exact artifact** but shipped at the bottom of the safe band: `InterpCutFP` M5(b) (Cutting.compute:755-757) — *"with α<1 the displayed FP TRAILS the R-reconstructed curr, so on the rigid post-cut spin the wall floats off the spinning piece → recommend α near 1"*; and ReconGridManager.cs:97-98 — *"a LOW value lags the cut wall... keep α ~0.5–1.0"*. α=0.5 still lags visibly.

### Hypotheses ruled out (CPU mirror)
- **External-FP box center (per-component `compCenter`):** a **no-op**. Every planar-cut component is a rigid same-rest-y block (ConnectivityLUT splits the straddling voxel into lower-4 / upper-4, `vertToComp=[0,0,1,1,0,0,1,1]`), so the QEF seed already moves at the component rate and the D5 box clamp never engages a partial-rate seed. Max per-component-vs-whole-voxel FP difference **0.0086 world units** vs the ~0.5–1.0 a visible gap needs. (The earlier `compCenter` change was reverted; only a NOTE comment remains in `ComputeComponentFP`.)
- **Bridging quads:** none. `DC_Stitch` quads at the cut all resolve their 4 vertices to one piece (keyed on the shared inside corner); the mirror found **0 bridge quads** and a perfectly bimodal skin-displacement histogram (tracks fully or stays).
- **`_VoxelExternalFP` fallback:** never reached for the visible skin in steady state (the per-component External FP is structurally valid — every boundary voxel emits an isect on each sign-change edge).

## Fix

**`cutFPInterp` `0.5 → 1.0`** (ReconGridManager.cs). With α=1.0, `InterpCutFP`'s `lerp(prev, curr, 1) = curr` makes the cut wall `_CutFP` byte-identical to the current material-frame reconstruction every frame, so the wall tracks the fall/spin **exactly like the un-smoothed skin** → dome and sliver stay welded, the gap closes. The EMA's only benefit (smoother cut *opening* during the slow sweep) does not apply to the fast post-cut fall, so α=1.0 loses nothing the demo needs. The 2D-2b machinery is **preserved** (set α<1 to re-enable).

**Seam → near-zero (separate, valid request):** `bladeD` `0.04 → 0.02` (D/L=0.08, in the paper sub-voxel band; gap = D scales linearly; not 0 to avoid zero-area wall faces). The *large* gap the user saw was the delamination, not the seam; this just sharpens the genuine D-wide slit.

## Why this is paper-faithful

2D-2b is itself a paper feature (Discussion clean:1271-1276: interpolate post-cut feature points prev→current to smooth large-deformation separation). The implementation applied the EMA to **only** the wall, not the skin — that asymmetry is the bug. α=1.0 removes the asymmetry by turning the EMA off (both surfaces un-smoothed, tracking together). The proper way to *keep* the opening-smoothing without the lag is a **fall-aware α** (relax→1 for fast-moving owner particles) or applying the same EMA symmetrically to `_CutFPExternal` — deferred as an enhancement; α=1.0 is the correct, minimal fix for the reported bug.

## Verification

- **Decisive Unity A/B (the mirrors' recommended repro):** set `cutFPInterp = 1.0` → the flat sliver snaps up to the falling dome, the dark gap closes; at `0.5` the gap returns. This is a **temporal** effect the static CPU mirrors could not render but both pinpointed in the code.
- **GPU oracle (to add):** extend `CutFPInterp_GpuOracle_Tests` — seed `_CutFP=curr`, `_PrevCutFP=prev`, advance `curr` by a fixed per-frame delta `d`, chain `InterpCutFP` N frames; assert steady-state lag `(curr − displayed) → d·(1-α)/α` (= `d` at α=0.5, **= 0 at α=1.0**). This pins the lag mechanism and the fix.
- **Regression note:** the prior `CutRimExternalFP` Test 4 was removed (a rigorous CPU mirror proved it did not discriminate). Tests 1-3 (round External margin, interior-only w==0, Internal centroid unchanged) still hold.

## Out of scope

- 2D-3 Ablation (resumes after this is verified).
- Fall-aware α / symmetric-EMA enhancement (preserves opening-smoothing) — only if the demo later needs both.
