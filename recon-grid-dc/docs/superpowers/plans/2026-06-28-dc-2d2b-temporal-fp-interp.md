# Plan — 2D-2b temporal feature-point interpolation (paper Discussion, clean:1271-1276)

Date: 2026-06-28
Status: DRAFT → multi-subagent audit → implement → multi-subagent review

## 1. Goal / paper basis (VERBATIM)

newpaper_clean.txt:1270-1277 (Discussion / limitations): "Compared with traditional mesh cutting methods,
due to **feature point drift**, our approach may cause a **sudden separation of the two sides of the
cutting surface over a large distance, rather than a smooth split, under large deformations**. To address
this issue, **we interpolate the positions of post-cut feature points from the previous frame to the
current frame**, thereby achieving visually smooth transitions. **However, this strategy does not fully
adhere to physical principles** and requires further refinement."

⇒ 2D-2b = a TEMPORAL interpolation of the post-cut feature points: the displayed cut FP each frame eases
from its previous-frame value toward the current-frame computed value, so a large frame-to-frame drift of
the FP (under large deformation) renders as a smooth transition instead of a sudden jump. The paper gives
**NO formula** (interpolation factor, seeding, validity handling are all paper-silent → declare). The
paper itself flags this as a visual approximation that "does not fully adhere to physical principles."

## 2. Where it sits in the pipeline

DualContouring.Build cut-FP order (DualContouring.cs:157-174): ClearCutFP → LookupConnectivity →
AccumulateCutFP → **ComputeComponentFP (→ _CutFP, the current-frame FP)** → … → DC_Stitch (reads _CutFP)
→ BuildCutTriangles (reads _CutFP) → DC_NormalizeCutFP.

2D-2b inserts a new kernel **InterpCutFP** immediately AFTER ComputeComponentFP (DualContouring.cs:174)
and BEFORE the consumers (DC_Stitch / BuildCutTriangles), so they render the SMOOTHED _CutFP. It composes
cleanly with 2D-2: ComputeComponentFP already produces the R-frame-reconstructed current FP; 2D-2b smooths
that final position.

## 3. Design

### A. `_PrevCutFP` buffer
- `RWStructuredBuffer<float4> _PrevCutFP` — same shape as `_CutFP` (8*VoxelCount slots, float4 = xyz pos +
  w validity). 16-byte stride. C# `ComputeBuffer PrevCutFP = new ComputeBuffer(8*VoxelCount, 16)`.
- PERSISTS across frames — it is NOT touched by ClearCutFP (which zeros _CutFP.w each frame). Initialised
  to all-zero (w=0 ⇒ invalid) at allocation, so frame-0 cut FPs appear at their current value (no interp
  from garbage). Reset only on re-init / clear-cuts.

### B. `InterpCutFP` kernel (Cutting.compute) — one thread per slot (8*VoxelCount)
```hlsl
float4 curr = _CutFP[slot];        // current frame (ComputeComponentFP, R-frame reconstructed)
float4 prev = _PrevCutFP[slot];    // previous DISPLAYED FP
float4 outv;
if (curr.w < 0.5)        outv = float4(0,0,0,0);                                   // no cut FP now → invalid
else if (prev.w < 0.5)   outv = curr;                                             // NEWLY valid → appear at current
else                     outv = float4(lerp(prev.xyz, curr.xyz, _CutFPInterp), 1.0); // smooth the drift
_CutFP[slot]     = outv;   // consumers (DC_Stitch, BuildCutTriangles) use the smoothed FP
_PrevCutFP[slot] = outv;   // persist this frame's DISPLAYED FP for next frame (EMA)
```
- This is exponential smoothing (EMA): displayed_N = lerp(displayed_{N-1}, current_N, α). α = `_CutFPInterp`.
- SEEDING decision (paper-silent, declare): a NEWLY-valid slot (prev invalid) appears at `curr` (no interp
  on first appearance) — the literal reading of "interpolate post-cut FPs from the previous frame", since a
  just-appeared FP has no previous post-cut value. The smoothing then applies to the per-frame DRIFT (the
  paper's stated symptom: "feature point drift … under large deformations"). [Alternative considered: seed
  from the pre-cut continuous DC dual vertex so the split eases open from the uncut surface — rejected for
  this stage: it requires reading the Recon per-voxel FP into the cut slots and the paper's wording is
  about drift of post-cut points, not the open-from-uncut transition. Audit to confirm.]

### C. `_CutFPInterp` uniform (the interpolation factor α, PAPER-SILENT)
- α ∈ (0,1]; small = smoother/laggier, 1 = no smoothing (current behavior). Default ≈ 0.25 (eases over
  ~1/α ≈ 4 frames). Exposed/configurable. Set by DualContouring.Build (a public field on the manager).

### D. Wiring
- ReconBuffers: add `PrevCutFP` buffer (8*VoxelCount × 16), zero-init, Dispose.
- Cutting.compute: `#pragma kernel InterpCutFP` + `RWStructuredBuffer<float4> _PrevCutFP` + the kernel.
- DualContouring: `kInterpCutFP = cut.FindKernel("InterpCutFP")`; after ComputeComponentFP dispatch
  (line 174), `BindCut(kInterpCutFP, rb); cut.SetFloat("_CutFPInterp", interp); cut.Dispatch(kInterpCutFP,
  G(slotCount),1,1);`. Bind `_PrevCutFP` in BindCut (and to InterpCutFP). Pass α from the manager.
- Gated inside the existing `if (cuttingEnabled)` cut-FP block (null cutting ⇒ Stage-1 path unchanged).

## 4. No-regression / interactions
- With α = 1.0, `lerp(prev,curr,1)=curr` and newly-valid=curr ⇒ _CutFP is byte-identical to pre-2D-2b
  every frame ⇒ no behavior change (the off switch). Default α<1 only smooths.
- Composes with 2D-2 (smooths the R-frame-reconstructed FP) and with the rest-pose cut (the cut SET is
  unaffected — only the FP POSITION is smoothed; severing/topology untouched).
- The smoothing introduces intended LAG (displayed FP behind the true FP). The paper accepts this
  ("does not fully adhere to physical principles").
- Validity: BuildCutTriangles/DC_Stitch skip w<0.5; a newly-valid slot is immediately valid (outv=curr,
  w=1) ⇒ no dropped quads from the interp. A valid→invalid transition (rare with cumulative cuts) drops.

## 5. Verification
- GPU oracle `CutFPInterp_GpuOracle_Tests`: (1) α=1 ⇒ _CutFP==curr (no-op). (2) prev valid + curr valid,
  α=0.25 ⇒ _CutFP.xyz == lerp(prev,curr,0.25) and _PrevCutFP updated to the same. (3) prev invalid + curr
  valid ⇒ _CutFP==curr (appear). (4) curr invalid ⇒ _CutFP invalid. (5) EMA convergence: iterate the
  kernel N times with a fixed curr ⇒ _CutFP → curr geometrically (within tol after ~log(tol)/log(1-α)).
- REGRESSION: existing CutSurface oracle with α=1 (or _PrevCutFP seeded = _CutFP) ⇒ byte-identical.
- VISUAL (user): under the 2D-2 spin/large-deformation demo, the cut surface should open/track SMOOTHLY
  (no per-frame popping of the cut walls) with α<1 vs visibly jumpier with α=1.

## 6. Out of scope
- Seeding from the pre-cut continuous DC vertex (open-from-uncut easing). 2D-3 ablation.
- Adaptive α (drift-magnitude-dependent) + rotation-aware residual smoothing. The paper specifies a single
  prev→current interpolation; these are the principled FUTURE refinements (see §7 artifacts).

## 7. AUDIT REVISIONS (workflow ws89m9fpw, APPROVE-WITH-REVISIONS, paperFaithful=yes-with-declared-deviations) — BINDING

### M1 (paper framing) — EMA is ONE faithful realization
Add to §3.B: the paper's "interpolate ... from the previous frame to the current frame" is literal at the
STEP level (per-frame lerp of last-displayed→current); chaining it = EMA, which introduces a steady-state
lag under SUSTAINED drift that the paper accepts ("does not fully adhere to physical principles"). Declare
the alternatives the paper text does not exclude (paper-silent): (a) fixed-N-frame ease from the cut
instant; (b) per-frame-displacement-clamped drift.

### M2 (seeding rationale) — declare it targets DRIFT-SNAP, not the appearance pop
Strengthen §3.B to LEAD with the semantic argument: the paper attributes the artifact to "feature point
DRIFT" (clean:1270) = post-appearance per-frame movement of an ALREADY-EXISTING FP. So smoothing targets
the per-frame jump of an already-valid slot; seeding a newly-valid slot at `curr` (its first post-cut value)
is the literal reading. EXPLICITLY declare: 2D-2b targets DRIFT-SNAP, NOT the cut-instant appearance pop
(easing the appearance from the pre-cut continuous DC vertex is a DIFFERENT feature the paper never
describes → out of scope, pointer to 2D-3).

### M3 (alpha plumbing) — concrete mechanism
Add a 5th param to `DualContouring.Build(rb, dims, L, qefIters, float cutFPInterp = 1f)` (DEFAULT 1.0 ⇒
no-op ⇒ back-compat for any not-yet-updated caller). Inside the cutReady block, before the InterpCutFP
dispatch: `cut.SetFloat("_CutFPInterp", cutFPInterp);` (mirrors the existing `_QefIters`/`_L` SetInt/SetFloat
at DualContouring.cs:154-155). ReconGridManager exposes `public float cutFPInterp = 0.5f;` and passes it to
both Build call sites. (α=0.5 default: 2D-2b ON but gentle; smooths the cut-OPENING/FP-jump transients —
its benefit, visible during the sweep — with a bounded, solver-damped transient rotation lag on the
post-cut spin. Set to 1.0 to disable; lower for more smoothing.)

### M4 (_PrevCutFP zero-init) — explicit, not implicit
Project convention is EXPLICIT SetData-zero in Upload (ReconBuffers.cs:473, "clean init guarantees no NaN
on the very first dc.Build"), NOT implicit zero on ComputeBuffer creation. So: allocate `PrevCutFP = new
ComputeBuffer(Math.Max(1, slotCount), 16)` (mirror CutFP :358); `PrevCutFP.SetData(new float4[Math.Max(1,
slotCount)])` in Upload (alongside :473); `PrevCutFP?.Dispose()` in Dispose. Frame-0/first-cut seeding
(w=0 → invalid → newly-valid → outv=curr) DEPENDS on this.

### M5 (§4 declare 3 interaction artifacts — paper-silent, accepted)
(a) DIFFERENTIAL-LAG SHEAR (TRANSIENT): under large deformation two FPs sharing one cut quad at different
   radii from the spin axis lag by different amounts → quad shear; worst-case instantaneous lag ≈
   v·(1−α)/α. On the spin demo (one-SHOT impulse ω=3, solver-damped, NOT sustained) this is a bounded
   transient PEAK (~few·D at the impulse), not a sustained floor. Declare; recommend α not too small.
(b) EMA-vs-2D-2 OPPOSITION: with α<1 the displayed FP TRAILS the R-frame-reconstructed curr → the cut wall
   lags the tissue rotation that 2D-2 tracks (geometrically the wall floats off the spinning piece). On a
   RIGID rotation 2D-2 already makes curr smooth (no drift), so 2D-2b can only lag it — recommend α near 1
   (≥0.5) on the spin demo; the principled future fix is rotation-aware smoothing (smooth only the residual
   after removing the owner's rigid R). 2D-2b's REAL benefit is the cut-OPENING transients (cut/sever FP
   jumps), most visible DURING the sweep, not on the post-cut rigid spin.
(c) CUMULATIVE-RECUT SLOT-RENUMBER SMEAR: VoxelCutMask is cumulative (ReconBuffers.cs:457); a SECOND cut in
   an already-cut voxel can renumber components (ConnectivityLUT dense-renumber) → `_PrevCutFP[8v+comp]` may
   refer to a different physical component → a single-frame interp smear. Rare (single-blade sweeps rarely
   re-cut a voxel). Declare as bounded; optional guard deferred.
Also declare: `_PrevCutFP` is left untouched across a null-cut frame (cutReady false) and reset to zero only
on re-init, so a new cut session never seeds from a stale prior FP. And: a valid→invalid→valid slot re-pops
(EMA history discarded on the curr-invalid branch) — rare with the stable pure-centroid path.

### M6 (verification) — strengthen the visual check
§5 VISUAL must explicitly check the cut wall stays ATTACHED to the spinning freed piece at the chosen α
(the current "opens smoothly" check would PASS while the wall floats off). Keep the α=1 byte-identical
REGRESSION test (note _PrevCutFP is still written at α=1 — harmless; the off-switch is purely on consumed
_CutFP). The intended 2D-2b BENEFIT demo is during the SWEEP (smoother cut opening at α<1 vs jumpier at α=1).

### Realization details (confirmed)
- All FOUR _CutFP consumers run after the ComputeComponentFP dispatch (DualContouring.cs:174) and therefore
  see the smoothed FP: DC_Stitch (StitchVertId, Recon.compute:226), BuildCutTriangles (Cutting.compute:707),
  DC_NormalsScatter (FetchPos, Recon.compute:329), DC_NormalizeCutFP (step 14). Normals computed on the
  displayed geometry → position/normal stay coherent. (Insert InterpCutFP right after line 174, before the
  TriCounter reset at line 178.)
- InterpCutFP guard: `uint slot=id.x; if (slot >= 8u*(uint)_VoxelCount) return;` — mirror ClearCutFP:460
  EXACTLY (8u*_VoxelCount, not _CutSlotCount). Bind ONLY `_CutFP` + `_PrevCutFP` for kInterpCutFP. Declare
  `RWStructuredBuffer<float4> _PrevCutFP;` next to `_CutFP` + `float _CutFPInterp;` near the scalar uniforms;
  `#pragma kernel InterpCutFP` after the ComputeComponentFP pragma.
- The two paper "sides" are two INDEPENDENT per-component FPs (clean:344-346); smoothing each position
  independently smooths their separation as a derived quantity (no coupled model; generalizes to >2 comps).
