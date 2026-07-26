# Stage 2D + completion — full paper alignment (master roadmap)

> The COMPLETE remaining-work plan to make the cutting effect match the paper (Li/Zhou 2026 CMPB §2.1)
> EXACTLY. Grounded in the module-by-module audit (workflow w3dyi5tv3) + findings.md. Each item: paper spec
> (with line refs), current status, what to build, acceptance. Ordered by priority. Detailed sub-plans are
> separate docs; this is the authoritative ordering + scope. Discipline rules 1-3 (architecture fidelity,
> traceability, variable↔symbol) apply to every item.

## Where we are (post seam-fix, commit 137e13f)
DONE & paper-faithful: Stage 1 (DC recon + mass-spring Eq11/Eq13-16 + adaptive RK45); 2A 4096 connectivity LUT;
2B Möller-Trumbore detection + ±D/2 cut points; 2C cut-surface generation with PER-COMPONENT outer-surface split
(skin opens); seam now sub-voxel (bladeD=0.04, _CutPull=4.0 → gap==D, smooth). The cut is VISIBLE and tracks the
per-frame DC remesh.

The cut is still **visual-only** — the two halves do not physically separate. That + 3 more paper modules remain.

## Priority-ordered remaining work

### ▶ 2D-1 — §2.1.2 PHYSICAL SEVERING  (BLOCKER; detailed in 2026-06-28-dc-2d1-severing.md)
- **Paper**: line 367 "when the cut is complete, the physical connection of that edge is severed"; line 1162-1163
  (Fig 3.9) the lower half falls away / upper rises; **line 520-522 explicitly warns against** "the soft tissue
  appears visually severed while forces are still acting on the separated ends" — exactly today's state.
- **Status**: MISSING (no SeverLinks kernel). Force kernels ALREADY honor the guards: `AccumulateStructural`
  skips `n<0` (Physics.compute:94); `AccumulateBending` returns on `alive==0` (Physics.compute:128). Only the
  WRITE side is missing.
- **Build**: a `SeverLinks` compute kernel (one thread per grid edge): for each CUT edge (axis,baseCorner)→idA,idB,
  set `_NbrIdx[6*idA+sPlus]=-1` and `_NbrIdx[6*idB+sMinus]=-1` (both directions; slots 1/3/5 = +x/+y/+z,
  0/2/4 = −x/−y/−z), and zero `alive` for every bending pair centered at idA whose arm == idB (and centered at
  idB whose arm == idA). Idempotent → dispatch each frame after DetectCut. NO Physics.compute changes (guards
  already present). NbrIdx is the load-bearing link; Springs buffer is unread by force kernels (leave/optional).
- **Accept**: in Unity the sphere splits and the two halves separate under gravity; GPU oracle asserts zero
  structural + zero bending force is transmitted across a severed edge.

### ▶ 2D-2 — §2.1.2 DYNAMIC PARTICLE FRAMES (Berndt R)
- **Paper**: lines 355-368 + Fig 2.6/2.7. Under deformation the voxel grid distorts; each particle's local axes
  ROTATE (new coordinate system per Berndt[22]); the 6 axis ±directions map to the 6 incident edges; a cut
  point's LOCAL coords are recorded relative to the edge's two vertices in the UNDEFORMED/rest state; after the
  cut the world position is recomputed via the particle's CURRENT rotated frame (the cut point swings with the
  particle, Fig 2.7).
- **Status**: MISSING. Cut point stores a STATIC WORLD offset (`EmitCutPoint` Cutting.compute:230) added raw in
  `AccumulateCutFP` (Cutting.compute:436, `world = cornerPos[owner] + localOffset`, no rotation). The inputs
  ALREADY EXIST: `_NbrIdx` + `_RestNbr` (rest offsets) + live `_CornerPos` = the rest→current edge pairs.
- **Build**: a per-particle `_FrameBuf : float3x3` (or quaternion); a `ComputeFrames` kernel (1 thread/particle):
  deformation gradient `F = (Σ_e p_e ⊗ r_e)(Σ_e r_e ⊗ r_e)^-1` over the ≤6 incident edges (`r_e=_RestNbr`,
  `p_e=_CornerPos[_NbrIdx]-_CornerPos[p]`, skip `_NbrIdx==-1`); polar-decompose `F=R·S` (iterate
  `R_{k+1}=½(R_k+R_k^{-T})`, 3-5 iters) → store R. At cut instant store `localOffset = Rᵀ_owner·(P_side−pPos)`
  (rest-frame). Per frame `world = pos + R_owner·localOffset` (the placeholder at Cutting.compute:436). PAPER-SILENT:
  polar-decomposition for "method [22]".
- **Accept**: apply a known rigid rotation to all corners → recovered R == that rotation, and a recorded cut
  point rotates rigidly with the tissue (drift==0 under pure rotation). Visually the cut follows twisting/sagging
  with no shear/drift.
- **Synergy with severing**: after 2D-1 the two halves move independently; 2D-2 keeps each half's cut wall glued
  to its own deforming surface. Do 2D-1 first (separation), then 2D-2 (no drift).

### ▶ 2D-2b — TEMPORAL feature-point interpolation (paper:1270-1276) — was MISSING, added by audit
- **Paper** (lines 1270-1276, EXPLICIT): "due to feature point drift, our approach may cause a sudden separation
  of the two sides of the cutting surface over a large distance, rather than a smooth split, under large
  deformations. To address this issue, **we interpolate the positions of post-cut feature points from the
  previous frame to the current frame**." This is a SECOND, distinct anti-drift mechanism layered on top of the
  2D-2 R-frame: 2D-2 removes shear/rotation drift; this lerp removes the residual large-deformation *jump* so the
  cut opens smoothly rather than snapping.
- **Status**: MISSING (and was absent from the original roadmap — the central completeness gap the audit caught).
- **Build**: a `_CutFPPrev : float4[8*VoxelCount]` last-frame per-component FP buffer (parallel to `_CutFP`); each
  frame after `ComputeComponentFP`, lerp `_CutFP[slot] = lerp(_CutFPPrev[slot], _CutFP[slot], α)` for valid slots
  (α a smoothing factor; paper gives no value → PAPER-SILENT tune), then copy current→prev. Handle first-appearance
  (a newly-valid slot has no prev → use current, no lerp). Cite paper:1273-1276.
- **Accept**: under a fast large deformation the cut opens smoothly (no single-frame snap); visually the two cut
  walls track frame-to-frame without jitter.
- **Deviation note**: the paper itself flags this as not physically exact (a visual-smoothing pass) → declare as a
  PAPER-DOCUMENTED smoothing step in the DeviationLedger. Do AFTER 2D-2 (it smooths the R-frame-anchored FP).

### ▶ 2D-3 — §2.1.4 ABLATION (two-pass volume removal)
> Ordering note (audit): §2.1.4 ablation has NO paper dependency on §2.1.2 dynamic frames — it depends only on
> the §2.1.1 LUT/centroid/redraw pipeline (already DONE, paper:498-499). Its position after 2D-2 is by
> IMPLEMENTATION SURFACE AREA (new buffer + state guards across 3 .compute files), not a paper dependency; it can
> be promoted ahead of 2D-2/2b if a demo needs volume-removal sooner. (Severing-first IS a hard dependency, paper:520-522.)
- **Paper**: lines 491-499 + Fig 2.9. When the tool path passes through tissue, in-range particles change state
  in TWO ORDERED passes: PASS 1 — particles OUTSIDE the isosurface are deleted (no force, no surface, NO cut
  points); PASS 2 — particles INSIDE are "ablated": incident "ablated edges" marked (incl. toward SURVIVING
  neighbors), the particle deleted, cut points generated along ablated edges (bound to the SURVIVING endpoint),
  funneled into the SAME 4096-LUT + centroid + redraw pipeline; then surface redrawn.
- **Status**: MISSING entirely. No particleState buffer, no AblatePass1/Pass2, no tool-range/mode, no
  `state!=alive` early-out in any per-particle kernel.
- **Build**: `particleStateBuf : int[cornerCount]` (0=alive,1=ablated,2=deleted); a tool range/radius + ablation-
  mode flag on CuttingTool; `AblatePass1` then (barrier) `AblatePass2` kernels; retrofit `if(state!=ALIVE)return`
  into ALL per-particle/corner kernels (ClearForces, AccumulateStructural, AccumulateBending, SeedTrialFromState,
  BuildTrialState, ComputeSlope, IntegrateY5, ComputeError, DC_FeaturePoints, RebuildIsectWorld,
  ComputeComponentFP, BuildCutTriangles, AccumulateCutFP); single-endpoint EmitCutPoint variant for ablated edges.
  PAPER-SILENT (findings.md:39): orphan cut-point → surviving endpoint.
- **Accept**: GPU oracle — Pass-1 outside particles deleted with no cut points; Pass-2 inside particles produce
  surviving-endpoint cut points; a deleted particle exerts zero force; a volume is removed and the surface reseals.
- **Largest surface area** (new buffer + guards across all 3 .compute files) → after 2D-1/2D-2.

### ▶ 2D-4 — §2.1.3 INTERACTIVE TOOL + detection robustness
- **Paper**: line 36 + 932-938 — the headline is INTERACTIVE cutting; the experimental tool is driven by an Omni
  haptic device via OpenHaptics (the tool pose comes from the user's hand). lines 380-382 motivate the swept-plane
  by low-FPS/fast-motion robustness.
- **Status**: PARTIAL. Detection math is faithful + oracle-tested; tool is a hardcoded auto z-sweep
  (ReconGridManager.cs:164-179). No mouse/keyboard/haptic driver. `DegenerateEps=1e-8` (CuttingTool.cs:76) is a
  length-dependent magnitude threshold (not normalized). No oblique/fast-multi-voxel-sweep test.
- **Build**: a `CuttingToolInput` MonoBehaviour driving `tool.Advance(S,E)` from a mouse-ray/screen-drag (keep the
  auto-sweep as a fallback mode; optionally an OpenHaptics/Omni binding). Normalize the degenerate-sweep guard to
  swept AREA / displacement vs a world-scale ε independent of blade length. Add GPU oracles: a fast sweep crossing
  >1 voxel layer in one frame (proves the swept-quad prevents missed collisions) + an oblique (non-axis) plane.
- **Accept**: user can drive the blade by mouse and cut interactively; the fast-sweep oracle passes.
- Lower priority than physical correctness; closes the largest §2.1.3 fidelity gap.

### ▶ FIDELITY / HOUSEKEEPING (do alongside the above)
- **§2.1.1 cut-FP — INVERT to paper-strict pure centroid (audit R3)**: `ComputeComponentFP` fuses an isosurface
  QEF + cut-centroid pull (`_CutPull=4.0`) instead of paper:344's plain centroid ("the feature point is the
  centroid of all the cut points"). For COMPLETE paper alignment the recommendation is INVERTED vs the original:
  the exact pure centroid is **already accumulated** in `_CutFPAccumPos/_CutFPAccumCnt` (Cutting.compute centroid
  path), and interior cut voxels already degenerate to it. So **make the pure cut-centroid the DEFAULT** for the
  cut-wall FP (paper-strict), and keep the `_CutPull` QEF-fusion behind a flag as **DEVIATION D7** (used only when
  the boundary lip needs isosurface conformance). Either way ADD a `DeviationLedger.cs` D7 entry citing paper:344.
  NOTE: with `_CutPull=4.0` the current FP already ≈ the centroid (the verified-good seam), so this is mostly a
  fidelity/documentation refinement + an optional switch; defer the behavioral switch until after severing unless
  it changes the look (it should not, since 4.0 ≈ centroid).
- **DeviationLedger completeness**: add D7+ entries for the cut-FP fusion, ±D/2 offset, n_cut formula, AABB
  broadphase, degenerate-sweep gate, ablation orphan-cut-point rule — the central ledger currently stops at D6.
- **params §3.4**: bladeD fixed (0.04). OPTIONAL: raise dims 24³→38³ (≈54k particles, paper budget) for a finer
  cut surface — ~8× GPU cost; defer unless fidelity to the paper's particle count is required.

## Execution model (per item, the user's workflow)
For EACH item: detailed sub-plan → multi-subagent audit (paper-alignment + feasibility) → subagent implementation
→ multi-subagent code review → user Unity verify. 2D-1 (severing) is first and is detailed now.
