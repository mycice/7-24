# Plan — 2D-2 Berndt per-particle rotated frame (paper §2.1.2)

Date: 2026-06-28
Status: DRAFT → audit → implement → review

## 1. Goal / paper basis

Make cut-surface points follow tissue ROTATION, not just translation. Today the cut point is a STATIC
world offset from its owner particle:
- `EmitCutPoint` (Cutting.compute:227): `cp.localOffset = P_side - pPos` (frozen world vector).
- `AccumulateCutFP` (Cutting.compute:467): `world = _CornerPos[owner] + cp.localOffset` — comment already
  says "2C static; 2D: +mul(R_owner,localOffset)".
So under TRANSLATION the cut walls track (cornerPos moves), but under ROTATION they DRIFT (the world offset
does not rotate).

paper §2.1.2 (newpaper_clean.txt:355-368): under deformation the voxel grid distorts and "the coordinate
axes of the particles are rotated due to the influence of the surrounding connections. The calculation of
the new coordinate system is based on [22]" (Berndt). "When an edge is cut, the local coordinates of the
cut point ... are recorded ... based on the condition that the edge has not undergone any deformation ...
the new cut point is computed using the corresponding coordinate axes." clean:1165: "After the particle
position changes, the cutting point can adjust its world coordinates according to the position and
orientation of the corresponding particle."

⇒ **world = cornerPos[owner] + R_owner · localOffset_rest**, where R_owner is the particle's current
rotation and localOffset_rest is the cut point's offset recorded in the owner's UNDEFORMED (material) frame.

This is also the rotation-generalization of the just-shipped rest-pose detection (the rest M-T tees up
2D-2: §E below).

## 2. Design

### A. Per-particle rotation R (Berndt [22] = polar of the local deformation gradient)
For corner c, gather its INTACT incident edges s (those with `_NbrIdx[6c+s] >= 0` — severed edges EXCLUDED
so the freed other-half cannot corrupt the frame):
- rest edge `r_s = _RestNbr[6c+s]`; deformed edge `d_s = _CornerPos[nbr] - _CornerPos[c]`.
- deformation gradient `F = (Σ_s d_s ⊗ r_s) · M^{-1}`, `M = Σ_s r_s ⊗ r_s`. The rest lattice is axis-
  aligned, so M is DIAGONAL → `M^{-1} = diag(1/Σ r_{s,x}², 1/Σ r_{s,y}², 1/Σ r_{s,z}²)` (cheap, no general
  3×3 inverse).
- `R = polar(F)` (rotation part): Higham iteration `R_{k+1} = 0.5 (R_k + R_k^{-T})`, `R_0 = F`, ~6 iters
  (R_k^{-T} via the 3×3 cofactor/adjugate). Orthonormal, det=+1.
- FALLBACK `R = identity` when: pinned, inactive (`_Active==0`), fewer than 2 valid non-colinear edges, or
  `|det F|` below eps (degenerate). Identity keeps the point as a pure translation offset (paper-safe).

### B. `_ParticleRot` buffer
- `struct RotGpu { float3 r0, r1, r2; }` = the 3 ROWS of R (36 bytes). C# twin in ReconBuffers (stride 36,
  same explicit-components pattern as GridEdgeGpu2/CutPointGpu). `int[cornerCount]`-sized.
- HLSL: `float3x3 R = float3x3(rg.r0, rg.r1, rg.r2); mul(R, v)` rotates v (row-major); `transpose(R)=R^{-1}`.
- Written by `ComputeParticleRot` (RWStructuredBuffer); read by the cut kernels (StructuredBuffer) — cross-
  shader like `_NbrIdx` (Cutting writes via SeverLinks, Physics reads). Single writer = race-free.

### C. `ComputeParticleRot` kernel (Physics.compute — reads physics buffers)
- One thread per corner: build F from ≤6 intact edges, polar→R, write `_ParticleRot[c]`. Pinned/inactive/
  degenerate → identity.
- DISPATCH after the RK45 `solver.Step` completes (CornerPos final) and BEFORE `cutDetector.Dispatch`
  (EmitCutPoint needs R) and `dc.Build` (AccumulateCutFP needs R). Add one dispatch in ReconGridManager
  (or a `MassSpringSolver` post-step) binding `_NbrIdx,_RestNbr,_CornerPos,_Active,_Pinned,_ParticleRot`.

### D. Cut-point changes (Cutting.compute) — the two flagged lines
- `EmitCutPoint`:227 → `cp.localOffset = mul(transpose(R_owner), P_side - pPos)` (un-rotate into the
  owner's material frame at cut time = record on the undeformed edge).
- `AccumulateCutFP`:467 → `world = _CornerPos[owner] + mul(R_owner, cp.localOffset)` (re-rotate by the
  CURRENT R each frame = computed using the corresponding axes).
- `R_owner = float3x3(_ParticleRot[owner].r0,r1,r2)`. Bind `_ParticleRot` to BOTH the DetectCut kernel
  (EmitCutPoint) and the AccumulateCutFP kernel (via DualContouring/CutDetector binding).
- NO REGRESSION INVARIANT: with `R = identity` (no rotation — the current translation demo), `transpose(I)·x
  = x` and `I·x = x`, so the cut FP is BYTE-IDENTICAL to today. The horizontal-cut-falls demo is unchanged.

### E. Detection under rotation (SCOPE DECISION)
Rest-pose detection currently M-T's the rest edge vs the WORLD blade; under rotation that cuts the wrong
material. The full §2.1.2 generalization transforms the world blade into each particle's MATERIAL frame via
R before the rest M-T. **DECISION: out of scope for 2D-2 (this stage).** 2D-2 covers the cut-SURFACE
rotation (the visible drift), which is the user-facing gap. Detection-under-rotation is a follow-up
(the demo is pure translation, so current detection is exact). Leave a TODO referencing this plan §E.
(Audit may pull it in if cheap; default = defer.)

## 3. Verification
- GPU oracle test `ParticleRot_GpuOracle_Tests`:
  (1) deformed = Rθ·rest for a known rotation Rθ (e.g. 30° about an axis) → assert ComputeParticleRot returns
      Rθ (orthonormal, det=1, ‖R−Rθ‖<tol).
  (2) F with a shear/stretch → R = the polar rotation part (orthonormal, det=1; not equal to F).
  (3) cut-point round-trip: store localOffset = R_cut^T·(P_side−cornerPos); reconstruct world =
      cornerPos + R_owner·localOffset → equals P_side when R_owner=R_cut; tracks under a NEW R'.
  (4) fallback: pinned/inactive/1-neighbor → R=identity.
- REGRESSION: with R=identity the cut-FP path is byte-identical (assert on the existing CutSurface oracle by
  binding identity _ParticleRot).
- VISUAL (user): 2D-2's effect is visible ONLY under ROTATION. The current free-fall is pure translation
  (R≈identity → no visible change). To SEE it: an OBLIQUE cut, or a scenario where the cut piece rotates
  (e.g. pin one edge so the freed piece swings). Optional demo toggle; not required for the unit tests.

## 4. Risks
- Severed-aware F: a cut-owner uses same-side neighbors; if <2 valid → identity (point translates only —
  acceptable, rare, and matches "no frame" gracefully).
- Polar convergence/degeneracy → identity fallback; bound iters (6) for real-time.
- 36-byte RotGpu stride — confirm Unity ComputeBuffer + HLSL StructuredBuffer (matches existing patterns).
- Dispatch ordering (ComputeParticleRot after Step, before cut+build) — must be wired in the manager.
- Cross-shader `_ParticleRot` (Physics writes, Cutting reads) — single writer, barrier at dispatch boundary.

## 5. Out of scope
- §E detection-under-rotation (blade→material-frame). 2D-2b temporal FP interpolation. 2D-3 ablation.

## 6. AUDIT REVISIONS (workflow wsuqvg6w3, APPROVE-WITH-REVISIONS) — BINDING

### M1 (CRITICAL) Dispatch order + R persistence
`SeverLinks` (sole writer of `_NbrIdx=-1`, Cutting.compute:733-734) runs INSIDE `cutDetector.Dispatch`
AFTER `DetectCut`/`EmitCutPoint`. So computing R *before* cutDetector.Dispatch builds it from the not-yet-
severed across-cut edge → contaminates the recorded offset. FIX:
- `_ParticleRot` PERSISTS across frames; allocate INITIALISED TO IDENTITY (so frame-0 emits = today's
  static offset = no regression).
- Per-frame order in ReconGridManager.Update: `solver.Step` → `cutDetector.Dispatch` (DetectCut[EmitCutPoint
  reads the persisted R = frame N-1's, post-sever for all prior cuts] + SeverLinks) → **ComputeParticleRot**
  (writes R_N, post-sever) → `dc.Build` (AccumulateCutFP reads R_N).
- DOCUMENT explicitly: EmitCutPoint reads R_{N-1}, AccumulateCutFP reads R_N. The edge cut THIS frame was
  intact in R_{N-1}, so the cut-frame offset carries a one-edge ΔR (sub-voxel; the neighbour is near-rest at
  the cut instant, and it is reconstructed via the evolving post-sever R thereafter). For the translation
  demo R≈I so it is exact. (A 2-pass finalize that re-keys new cut points to the post-sever R is the clean
  refinement — deferred; declare the ΔR approximation.)

### M2 (CRITICAL) Reflection guard in polar
The Higham iteration preserves sign(det F); for an inverted/reflected F (det F<0) it converges to a det=-1
REFLECTION → `mul(R,offset)` MIRRORS the wall (worse than identity). FIX: gate `d = determinant(F)`; if
`d <= detEps` (covers BOTH degenerate |d|~0 AND reflected d<0) → `R = identity`. Compute `R_k^{-T}` via
adjugate/cofactor ÷ det(R_k); R_0=F under the det>eps gate keeps iterates non-singular. (Net-new HLSL: no
float3x3/determinant/transpose exists in any shader yet.)

### M3 (IMPORTANT) Per-axis zero-diagonal gate (not an edge count)
M = Σ r_s⊗r_s is PROVABLY diagonal (restNbr = ±L·axis, BackgroundGrid.cs:307-317), so the cheap diag
inverse is valid — BUT a corner with intact edges on fewer than all 3 axes (boundary, or a multi-cut owner
with colinear survivors) gives a ZERO diagonal → 1/0 → NaN poisons F/R and every cut point on it. FIX:
build `mx=Σ r_{s,x}², my, mz` over intact edges; if `min(mx,my,mz) < axisEps·L²` for ANY axis → identity.
(Strictly stronger than `<2 edges`.) Oracle must add a colinear-2-neighbour-survivor → identity case.

### M4 (IMPORTANT) Deadband — keep the verified demo provably unperturbed
With R=identity the cut FP is byte-identical ONLY when an explicit identity buffer is hand-bound (the §3
oracle regression). LIVE, ComputeParticleRot computes R=polar(F) from gravity-sag/bounce → near-identity,
re-introducing ~1e-3 sub-voxel jitter on points the wggai2v70 fix (Cutting.compute centroid) just made
static. FIX: gate `if ‖F − I‖_F < deadband → R = identity` so genuinely-translating tissue keeps R≡I
(protects the hole/jitter fix). Add a LIVE regression: run the REAL kernel on a translation config, assert
‖R−I‖<tol (not a hand-bound identity). Correct the plan prose: live path is "sub-voxel-unchanged", not
bit-exact.

### M5 (IMPORTANT) Declare 3 paper deviations (in plan + code)
1. INFERENCE: R=polar(F) realizes Berndt [22]; §2.1.2 (clean:361) delegates the axis update to [22] without
   stating polar/F. The six-edge deformation gradient is the right input class (clean:363-364); [22]'s exact
   axis-mapping rule was not cross-checked.
2. DEVIATION (paper-silent): F EXCLUDES severed edges (NbrIdx<0) so a freed component cannot corrupt the
   frame; §2.1.2 maps axes to "the six surrounding edges" but is silent on cut-edge exclusion.
3. PAPER-SILENT: identity fallback reduces the cut point to the translation-only offset = the §3.9-validated
   regime (clean:1164-1166); degrades gracefully.

### M6 (IMPORTANT) Reconcile the over-claiming deviation block
Cutting.compute:60-62 currently PROMISES "2D-2 ... transform the world blade into each particle's MATERIAL
frame ... so deformed-mesh detection works directly under rotation" — exactly what §E DEFERS. EDIT it to:
2D-2 ships the cut-SURFACE rotation (EmitCutPoint/AccumulateCutFP R-frame) ONLY; blade→material-frame
DETECTION remains a follow-up (ref this plan §E).

### Verification additions
- CUT-WHILE-ROTATE is OUT OF CONTRACT until §E: the 2D-2 visual scenario MUST finish ALL cutting BEFORE any
  rotation (detection M-T's the rest plane with no R; only cut-THEN-rotate is consistent). State as an invariant.
- TWO-OWNER divergence: the two cut points of one edge reconstruct via R_A vs R_B; under a bend they separate
  by ≈ L·sinθ (~0.043 at 10°/cell, ~2×D). Add a two-adjacent-owner oracle (R_A≠R_B shear) asserting wall
  separation within tol OR document accepted; note ComputeComponentFP centroid (Cutting.compute:562-564) may
  launder within a component but not across the slit.

### Realization details (confirmed)
- `struct RotGpu` = 9 EXPLICIT floats `r0x,r0y,r0z,r1x,r1y,r1z,r2x,r2y,r2z` (36 bytes; rows of R). C# twin
  `RotGpu[CornerCount]` (NOT int[]). `new ComputeBuffer(Max(1,CornerCount), 36)` — 36-byte stride PROVEN
  in-project (Conn4096 ReconBuffers.cs:309). HLSL `float3x3 R = float3x3(rg.r0,rg.r1,rg.r2)`; `mul(R,v)`
  rotates (row-major); `transpose(R)=R^{-1}`.
- Cross-shader: declare RotGpu in Physics (writer, `RWStructuredBuffer<RotGpu>`) AND a GUARDED duplicate in
  Cutting (`#ifndef ROTGPU_DEFINED`, BendPairGpu precedent Cutting.compute:701-704; `StructuredBuffer<RotGpu>`
  reader). Single-writer = race-free; Unity's per-Dispatch UAV barrier suffices.
- Bind `_ParticleRot` in TWO C# sites: CutDetector.Dispatch (DetectCut, ~:90-97) + DualContouring.BindCut
  (AccumulateCutFP, ~:254-274). Dispatch ComputeParticleRot from ReconGridManager.Update (new dispatcher or
  MassSpringSolver.ComputeRotations(rb) binding _NbrIdx,_RestNbr,_CornerPos,_Active,_Pinned,_ParticleRot).
- CuttingCommon.hlsl CutPointGpu.localOffset doc: note that as of 2D-2 it is the owner's MATERIAL-frame
  offset, recovered via `mul(R_owner, localOffset)`; AccumulateCutFP is the SOLE reader (grep-confirmed).
- Tighten §1 citation 355-368 → 359-368 for the rotated-axis claim.
