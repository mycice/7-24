# 2D-3 Ablation Cutting — Design Spec

**Date:** 2026-06-29
**Status:** Approved design (Approach B — DC re-contour, **cavity crossings via INLINE RECOMPUTE in `DC_FeaturePoints`**) → writing-plans next
**Paper basis:** §2.1.4 "Ablation cutting process" (paper_pdf_text.txt p5-6, lines 478-499; Fig 2.9) + ref [22] Berndt et al. "Efficient surgical cutting with position-based dynamics" (ref22_berndt_text.txt: collision capsules ref22:361; heat-driven particle removal; re-extract cells with ≥1 missing particle ref22:316-318).
**Provenance:** brainstorm (resumed) + multi-subagent design-audit w167aqkba + plan-audit widh1dtbb + adversarial re-audit (HIGH confidence). This revision applies the **Approach B (DC re-contour)** design change, the 7 earlier audit must-fixes (preserved), the user-approved **Inline-recompute (Option A)** mechanism that replaces the structurally-impossible dynamic-isect buffer, and the 6 re-audit REWORK bug fixes + minors.

## Problem & goal

The cutting algorithm implements blade cutting (§2.1.1-2.1.3, volume-**preserving**: the skin splits, particles are severed but kept). The paper's §2.1.4 **ablation** is the missing **volume-LOSS** mode (electrocautery/ultrasonic): the instrument sweeps a region, tissue particles in range are **deleted**, and the surface is **redrawn around the resulting cavity**. Goal: implement §2.1.4 strictly (math + architecture) as a **self-contained module** that does not regress the (now paper-aligned) blade cut.

## APPROVED DESIGN CHANGE — cavity wall is "Approach B (DC re-contour)", NOT the blade's `_CutFP`/`BuildCutTriangles`

The cavity wall is produced by the **regular outer Dual-Contouring surface re-contouring the receded boundary**, NOT by the blade's per-component `_CutFP` centroid + `BuildCutTriangles`.

When a corner is deleted it becomes **effectively-outside** (`survivingInside := _CornerInsideRT && !_Deleted` becomes false). The receded boundary — every grid edge with **exactly one** surviving-inside endpoint — is re-contoured by `DC_FeaturePoints` (the QEF feature point) + the **`DC_Stitch` runtime gate** (Approach B, this spec §"Cavity reconstruction (C)"). The capsule-SDF crossings on those receded-boundary edges are the NEW isosurface intersections — and (per the re-audit, below) they are computed **inline inside `DC_FeaturePoints`**, NOT appended to a separate buffer.

**`BuildCutTriangles` and the `_CutFP` per-component centroid are NOT used for ablation.** This matches paper §2.1.4 "all voxel grids are updated and the surface is redrawn" and [22] "re-extract cells with ≥1 missing particle" (ref22:316-318) — the surface is re-extracted by the SAME contouring machinery, not by a slit-opener wall.

Rationale for choosing Approach B over the blade wall:
- Paper §2.1.4 says the surface is *redrawn*, not that a separate cut wall is inserted. The blade `_CutFP`/`BuildCutTriangles` path is the §2.1.1-2.1.3 *slit opener* (two ±D/2 cut points widening a gap) — semantically wrong for a one-sided volume-loss cavity.
- A cavity has no two-sided gap to open; it has a receded one-sided wall that is exactly an isosurface of the post-deletion occupancy. The regular DC surface already produces that wall once the receded-boundary crossings are present.

## STRUCTURAL REWORK (re-audit) — cavity crossings via INLINE RECOMPUTE in `DC_FeaturePoints` (Option A)

The previous revision routed cavity crossings through a separate `GenerateFrontierIsect` kernel that **appended to a flat dynamic-isect buffer feeding `DC_FeaturePoints`**. The adversarial re-audit proved this is **structurally impossible**, on two independent grounds grounded in the real code:

1. **No kernel reads a flat append buffer.** `DC_FeaturePoints` (Recon.compute:99-151) reads ONLY the **per-voxel static window** `_Isect/_IsectWorld[_VoxelIsectOffset[v] .. +_VoxelIsectCount[v])`. Those index arrays (`_VoxelIsectOffset`, `_VoxelIsectCount`) are baked once by `BackgroundGrid` prefix sums (BackgroundGrid.cs:135-152), uploaded once (ReconBuffers.Upload), and **never recomputed by any kernel**. A flat dynamic-isect buffer with its own append counter has no per-voxel window and is therefore read by **no kernel** — the crossings would never reach the QEF.
2. **A cavity-interior receded-boundary voxel has baked `cnt==0`.** Such a voxel's edges were all inside↔inside at bake time (no φ sign change ⇒ no baked isect), so `_VoxelIsectCount[v]==0`. Today `DC_FeaturePoints` early-returns `(0,0,0,0)` when `cnt==0` (Recon.compute:108-112) — so even if crossings existed elsewhere, this voxel writes an **invalid** feature point and the cavity wall has a hole.

**Replacement (user-chosen "Inline recompute" / Option A):** the cavity crossings are computed **inside `DC_FeaturePoints`** under a new `_AblationMode` uniform. The `GenerateFrontierIsect` kernel and the dynamic-isect buffer + counter are **deleted entirely** (old Task 4 is replaced). Because there are **no appends into `_IsectWorld`**, this **dissolves** the re-audit's RebuildIsectWorld-collision / append-ordering concern: `_IsectWorld` keeps exactly its baked entries, `RebuildIsectWorld` (Recon.compute:84-91) is unchanged, and no kernel mutates the per-voxel windows.

### `DC_FeaturePoints` gains an `_AblationMode` branch (NO LONGER "reused verbatim")

`DC_FeaturePoints` is **no longer reused verbatim** — it gains an ablation branch. The static branch stays **byte-identical**.

- **`_AblationMode == 0` (today's path, byte-identical):** read the static per-voxel window `[off, off+cnt)` exactly as today (Eq2 mean of `_IsectWorld[off+s]`, Eq1-5 QEF with D1 gradient + D5 stop, `cnt==0 ⇒ (0,0,0,0)`). Not one instruction changes on this path.

- **`_AblationMode != 0` (ablation):** the voxel's QEF crossing set is the **UNION** of two contributions:

  **(1) Filtered baked window** — iterate the baked window entries `[off, off+cnt)`. For each `IsectGpu it`, the INSIDE corner is whichever of `it.cornerA`/`it.cornerB` has `_CornerInsideRT != 0` (the baked-inside endpoint; recorded as `it.insideA`, ReconCommon.hlsl:15). **KEEP** the entry only if that inside corner has `_Deleted == 0` (i.e. it is still surviving-inside). This **drops** crossings whose inside corner was ablated (a deleted corner no longer participates in surface generation, paper:493-495) and **keeps** the surviving original skin. Use `_IsectWorld[off+s]` (deformed pos) + `_Isect[off+s].normal` exactly as today.

  PLUS

  **(2) Inline capsule crossings on receded-boundary edges** — enumerate the voxel's 12 edges from its 8 corners (`_VoxelCorner[8*v + c]`, the per-voxel global corner ids; the 12 local edge endpoint pairs `(lc_a, lc_b)` mirror `GridConventions.EdgeCorners`). For each edge `(A,B)` that is a **receded-boundary** edge — defined as **exactly one endpoint surviving-inside AND the non-surviving endpoint DELETED** (NOT merely φ-outside):
    - `survivingInside(c) := _CornerInsideRT[c]!=0 && _Deleted[c]==0`
    - `deletedInside(c)   := _CornerInsideRT[c]!=0 && _Deleted[c]==1`
    - edge subtype = `(survivingInside(A) && deletedInside(B)) || (survivingInside(B) && deletedInside(A))`

    compute the **capsule-SDF root** `t*` along the **DEFORMED** edge `lerp(_CornerPos[A], _CornerPos[B], t)` (see §"Capsule-SDF root finder"), set `pos = lerp(_CornerPos[A], _CornerPos[B], t*)`, set `normal =` the capsule **outward** normal `normalize(pos - closestPointOnSegment(pos, _CapA, _CapB))`, and add `(pos, normal)` to the QEF set.

  **Accumulate** the Eq2 initial mean over the UNION of (1)+(2), then run the existing Eq1-5 QEF iteration (D1 gradient at `x_k`, Eq4 decaying step, D5 box-SDF stop) over the same combined `(p_i, n_i)` set. **If the union is empty** write `(0,0,0,0)`; **else** write the feature point with `w = 1`. This makes a fully-interior receded-boundary voxel (baked `cnt==0`) get a **finite** in-voxel feature point — the structural fix.

  **Binds the ablation branch needs (threaded via `DualContouring.Build → kFP`):** `_CornerInsideRT`, `_Deleted`, and the capsule uniforms `_CapA`, `_CapB`, `_CapR`. Also requires `CorId()` in Recon.compute (see §"`CorId` in Recon.compute"). `_VoxelCorner` is already bound.

  **Crossing subtype discipline (re-audit bug #6):** the inline capsule crossing is emitted **only** on the surviving-inside ↔ **deleted** edge subtype. A surviving-inside ↔ **φ-outside** (never-inside) edge keeps its **baked φ crossing** via mechanism (1) — it is already a real baked isect in the window. This split is what keeps the capsule-SDF root well-posed (one sign change; see the invariant below).

### Capsule-SDF root finder (re-audit bug #6 — make bisection valid)

The crossing on a surviving-inside ↔ deleted edge is the **capsule-SDF root**, NOT `fa/(fa-fb)` (there is no φ sign change on an inside↔inside edge). Bisection is valid because of this **invariant**, established by `DetectAblation`:

> `_Deleted[c] == 1` **IFF** `CapsuleSdf(c) <= 0` (deleted ⇔ in-capsule), evaluated in the SAME frame.

So on a surviving-inside ↔ deleted edge, the **surviving** endpoint is OUT of the capsule (`sdf > 0`) and the **deleted** endpoint is IN (`sdf < 0`). A capsule is **convex**, so the capsule SDF along the straight deformed edge has **exactly ONE sign change** ⇒ bisection on `t ∈ [0,1]` converges to the unique root. Specify: **bisection ~24 iters** (or the closed-form segment-to-capsule root) evaluated in the **DEFORMED frame** (the same frame as detection — see below). The GPU oracle validates `|CapsuleSdf(pos)| < eps`.

### DEFORMED frame for ablation (resolves the re-audit rest-frame contradiction)

**Both detection AND crossing run in the DEFORMED frame.** `DetectAblation` tests the capsule against the **DEFORMED** `_CornerPos`, and the inline crossing uses the **DEFORMED** edge. Rationale (state explicitly):
- **(a)** the cavity wall is a **deformed-space** DC surface — the QEF over `_IsectWorld` is already in deformed space (RebuildIsectWorld re-projects each baked isect onto the deformed edge), so the capsule crossings must be in the same space to weld to it.
- **(b)** the `deleted ⇔ in-capsule` invariant (needed for a single capsule-SDF sign change ⇒ valid bisection) only holds if **detection + crossing share ONE frame**; mixing rest-detection with deformed-crossing would let a corner be `_Deleted` while its deformed position is OUT of the capsule (or vice-versa), breaking the sign-change guarantee.
- **(c)** the blade's **rest-frame** discipline (CutDetector.cs:73-78) exists only to dodge a **thin-swept-PLANE vs sagging-surface** miss (a zero-thickness blade plane can slip between a sagged edge's endpoints). A **capsule is a RADIUS volume**, so that miss does not apply — a volumetric probe catches the corner regardless of sub-voxel sag.

This **REPLACES** the previous "DetectAblation mirrors DetectCut's rest discipline" claim. The choice is **deformed (NOT reconstruct-rest)**, with the rationale above.

## Mechanism (§2.1.4, run per frame in the detector slot)

Strictly per paper:491-499, grounded in [22]. **Ablation is instant delete with no transition gate → it RE-RUNS every frame** (must-fix #7); the per-corner `_Deleted` flag is **reset to 0 each frame** at the head of the pass before re-classification. There is **no dynamic-isect buffer** (deleted by the Option-A rework — crossings live inside `DC_FeaturePoints`).

0. **Range-select** (paper:492 "particles within a certain range"; the capsule **shape** and "within a certain range" are **paper-silent, grounded in [22]'s collision capsules** ref22:361 — the paper does not prescribe a capsule, [22] does): per-corner test `CapsuleSdf(deformedCornerPos, prevTip, curTip, r) <= 0` — a swept **capsule** SDF, evaluated on the **DEFORMED** lattice. A NEW selection primitive (a volumetric probe), distinct from §2.1.3's zero-thickness M-T swept plane.
1. **Classify (no flip)** — `DetectAblation` [1/corner]: for every in-range corner, write **only** `_Deleted[c]=1`. It does **NOT** flip `_CornerInsideRT`, and does **NOT** set `_CornerActive` (must-fix #1, #2). The baked `cornerInside` stays readable un-flipped via the read-only `_CornerInsideRT`. This covers BOTH paper:493-495 (outside-in-range particles deleted) and paper:495-496 (inside-in-range particles "ablated"): both become `_Deleted=1`; the post-delete state of any corner is `survivingInside := inside && !deleted`. This step establishes the `deleted ⇔ in-capsule` invariant in the deformed frame.
2. **Sever deleted** (paper:493-494 "no longer participate in force calculations") — `SeverDeleted` [1/grid-edge]: for each grid edge with `_Deleted[idA]==1 || _Deleted[idB]==1`, set `NbrIdx=-1` both ways and `alive=0` on all incident bend pairs, so a deleted corner loses ALL 6 arms + every bend pair (must-fix #3). A **separate final step** (`FreezeDeleted` [1/corner]) applies `_CornerActive[c]=0` to deleted corners (must-fix #2) so Physics freezes them.
3. **Update voxels + redraw** (paper:498-499 = [22] "re-extract cells with ≥1 missing particle", ref22:316-318) — `RecomputeOccupancy` re-ORs the 8 corners' post-delete state: `_VoxelOccupied[v] = OR over 8 corners of (_CornerInsideRT[c]!=0 && _Deleted[c]==0)`. A voxel whose surviving-inside corners are all deleted → occupancy 0 → **vanishes through every existing gate = the cavity (volume loss)**. `dc.Build(ablationMode:true)` re-contours: `DC_FeaturePoints` (ablation branch) computes the receded-boundary feature points **inline** (filtered baked window + inline capsule crossings), and the `DC_Stitch` runtime gate stitches the receded boundary into the cavity wall.

**Occupancy / post-delete state — the single canonical formula (must-fix #1, should-fix reconcile):**
> A corner is **surviving-inside** iff `_CornerInsideRT[c] != 0 && _Deleted[c] == 0`.
> `_VoxelOccupied[v] = OR over the voxel's 8 corners of (surviving-inside)`.
> A grid edge is **surface-bearing** iff **exactly one** of its two endpoints is surviving-inside (covers BOTH surviving↔deleted-inside AND surviving↔φ-outside).
> A surface-bearing edge is a **receded-boundary** edge iff its non-surviving endpoint is additionally **deleted-inside** (`_CornerInsideRT!=0 && _Deleted==1`) — the narrower subset (surviving↔deleted only).

**TWO DISTINCT PREDICATES (re-audit IMPORTANT — do NOT conflate):**
- The **`DC_Stitch` runtime gate** fires on **every surface-bearing edge** (the BROAD predicate: exactly one endpoint surviving-inside). This is mandatory — it stitches BOTH the new cavity wall (surviving↔deleted) AND the surviving original skin (surviving↔φ-outside). It is also what makes frame-0 predicate-equivalence hold (with nothing deleted, the surface-bearing set == the static `_SurfaceEdges` φ-sign-change set). If the stitch gate used the narrow receded-boundary predicate, the entire surviving skin would un-stitch → a gross hole.
- The **`DC_FeaturePoints` mechanism (2)** inline capsule crossing fires only on the **receded-boundary** subset (the NARROW predicate: surviving↔deleted). Surviving↔φ-outside edges are served by mechanism (1) (their baked φ crossing), not by a capsule crossing.

`surviving-inside` + `_VoxelOccupied` are used identically by `RecomputeOccupancy`, `DC_FeaturePoints`, and `DC_Stitch`; the two **edge** predicates differ by design as above. `_CornerInsideRT` is a GPU copy of the baked `cornerInside` and is **read-only / never flipped**; deletion is expressed solely through `_Deleted`.

**Ablated-edge clarification (adjudicated):** §2.1.4's verbatim "edges connecting these [ablated] particles" names edges with both endpoints ablated, but the surface-bearing geometry lives on the **receded-boundary** (surviving-inside ↔ deleted) edge. Spec resolution: the inline capsule crossing is emitted **only** on receded-boundary edges (one one-sided crossing each, inside `DC_FeaturePoints`). Both-deleted interior edges carry no surface (their voxels vanish via occupancy); surviving-inside ↔ φ-outside edges keep their baked φ crossing (filtered window, mechanism (1)).

## Architecture

`AblationTool` (CPU) + `AblationDetector` (CPU orchestrator, mirrors `CutDetector`) + 3 ablation kernels in `Ablation.compute` (`RecomputeOccupancy`, `DetectAblation`, `SeverDeleted`+`FreezeDeleted`) + the `_AblationMode` branches in `DC_FeaturePoints` and `DC_Stitch` (Recon.compute), wired into `ReconGridManager.Update` at the `cutDetector.Dispatch` slot under a **mode toggle** (separate from the blade).

- **`AblationTool`** — swept capsule `{PrevTip, CurTip, Radius}`, auto-advanced like the blade sweep (ReconGridManager.cs:262-280 driver reused). Membership math (point-to-segment SDF) is new. `Valid` is hardcoded `true` — a stationary probe still ablates (must-fix should-fix; the capsule is a volumetric probe, unlike the blade's zero-area degenerate-sweep guard).
- **Kernel `DetectAblation`** [1/corner]: capsule-SDF in-range test on the **DEFORMED** `_CornerPos` (deformed frame — §"DEFORMED frame"); write `_Deleted[c]=1` for in-range corners. **No flip of `_CornerInsideRT`, no `_CornerActive` write** here (must-fix #1, #2). Establishes the `deleted ⇔ in-capsule` invariant.
- **Kernel `SeverDeleted`** [1/grid-edge]: gate on `_Deleted[idA]==1 || _Deleted[idB]==1`; set `NbrIdx=-1` both ways + `alive=0` on ALL incident bend pairs → all 6 arms + bend pairs of a deleted corner die (must-fix #3). A separate final pass `FreezeDeleted` sets `_CornerActive[c]=0` on deleted corners (must-fix #2).
- **Kernel `RecomputeOccupancy`** [1/voxel]: `_VoxelOccupied[v] = OR over 8 corners of (surviving-inside)`. Mirrors BackgroundGrid.cs:131-134 on the runtime state. **The load-bearing change** (today `_VoxelOccupied` is upload-once).
- **Cavity reconstruction (C) — Approach B, inline recompute (must-fix #4 + re-audit rework):**
  - **Feature point (`DC_FeaturePoints` ablation branch):** under `_AblationMode!=0`, the voxel's QEF crossing set is the UNION of (1) the **filtered baked window** (keep an entry iff its baked-inside corner has `_Deleted==0`) and (2) **inline capsule crossings** on receded-boundary edges (capsule-SDF root on the DEFORMED edge + capsule outward normal). Empty union ⇒ `(0,0,0,0)`; else the QEF feature point (w=1). A cavity-interior voxel with baked `cnt==0` gets a finite in-voxel feature point from (2). NO `_CutFP` centroid. NO dynamic-isect buffer. `_AblationMode==0` is byte-identical to today.
  - **Stitch (the runtime gate):** `_GridEdges` is **GridEdgeGpu2 (16B, NO `insideA`)** with count **`GridEdgeCount`** — different from `_SurfaceEdges` (**GridEdgeGpu 32B, HAS `insideA`**, count **`SurfaceEdgeCount`**). In ablation mode, bind `_GridEdges`+`_CornerInsideRT`+`_Deleted` to `kStitch`, dispatch over `GridEdgeCount` under an `_AblationMode` guard, and **DERIVE** `insideA = survivingInside(CorId(baseCorner))` for BOTH the inside-corner key (`insideCornerCoord`) AND the winding swap (replacing every `ge.insideA` read). Gate each quad on "exactly one endpoint surviving-inside" + `_VoxelOccupied`. Keep the static `_SurfaceEdges` branch as a **verbatim early-out** when `_AblationMode==0` (byte-identical). Requires `CorId()` in Recon.compute (see below).

### `CorId` in Recon.compute (re-audit bug #4 — won't compile without it)

The `DC_Stitch` ablation branch AND the `DC_FeaturePoints` inline branch both map a grid-corner coord → global corner id, but `CorId()` lives **only** in `Cutting.compute:160`; `Recon.compute` has only `VoxId()` (Recon.compute:70-73). **Add to `Recon.compute`** (mirror `Cutting.compute:160`):
```hlsl
int CorId(int3 c) { return c.x + (_Dims.x + 1) * (c.y + (_Dims.y + 1) * c.z); }
```
Without it the ablation branches do not compile.

**Buffers (net-new + repurposed):**
- New RW `RWStructuredBuffer<int> _CornerInsideRT` — GPU copy of baked `cornerInside`, init = baked, **read-only / never flipped**. (Today `cornerInside` is CPU-only; not on the GPU at all.)
- New RW `RWStructuredBuffer<int> _Deleted` — per-corner deleted flag (`int[CornerCount]`, init 0). The SOLE expression of deletion; **reset to 0 each frame**.
- `_VoxelOccupied` changes upload-once → per-frame RW.
- `DC_FeaturePoints` gains capsule **binds** (`_CornerInsideRT`, `_Deleted`, `_CapA`, `_CapB`, `_CapR`) on its `kFP` bind in `DualContouring.Build` (ablation path).
- **NO dynamic-isect buffer** (deleted by the Option-A rework — was the old Task-4 append buffer + counter).
- Reuse `_NbrIdx`/`_BendPairs`/`_CornerActive` (write contract), `_GridEdges` as-is, `_VoxelCorner`/`_Isect`/`_IsectWorld`/`_VoxelIsectOffset`/`_VoxelIsectCount` as-is (read-only windows, never recomputed).
- **NOT reused for ablation:** `_CutPoint`/`_CutPointCounter`/`_CutFP`/`_VoxelCutMask`/`EmitCutPoint`/`AccumulateCutFP`/`ComputeComponentFP`/`BuildCutTriangles` (those are the blade slit-opener — must-fix #4, #7).

**`dc.Build` signature (re-audit bug #2):** the real signature is `Build(ReconBuffers rb, int3 dims, float L, int qefIters, float cutFPInterp=1f)` (DualContouring.cs:127). Add **`bool ablationMode=false`** as the last arg: `Build(ReconBuffers rb, int3 dims, float L, int qefIters, float cutFPInterp=1f, bool ablationMode=false)`. **Concrete wiring (the ONE chosen):** `AblationDetector` sets the capsule uniforms (`_CapA/_CapB/_CapR`) on the **recon** shader and binds `_CornerInsideRT`/`_Deleted` to `kFP`/`kStitch` (it already holds the recon `ComputeShader` reference via `DualContouring`, or `Build` does these binds when `ablationMode==true`). `Build(ablationMode:true)` binds the capsule context + `_CornerInsideRT`/`_Deleted` to `kFP` and `kStitch`, and sets `_GridEdgeCount`. (Implementer note: `Build` owns the capsule binds because it owns the `kFP`/`kStitch` dispatch; `AblationDetector` passes the capsule + buffers to `Build`, or `Build` reads them from `rb` + a small capsule struct argument.)

**`_AblationMode` must be set UNCONDITIONALLY on EVERY `Build` (re-audit IMPORTANT — stale-global hazard):** a `ComputeShader`'s scalar uniforms are **global and sticky** across dispatches on the same shader instance. If `Build(ablationMode:true)` ever ran on the same `DualContouring` `cs` instance, `_AblationMode` stays `1`, and a later `Build(ablationMode:false)` would silently run the ablation branch on the blade → divergence (this hits Task-6's in-test `true`-then-`false` comparison directly). **Therefore `Build` ALWAYS calls `cs.SetInt("_AblationMode", ablationMode ? 1 : 0)` on BOTH paths** — the `false` path explicitly resets it to `0` (and binds nothing else) → today's path, byte-identical. NEVER leave `_AblationMode` unset relying on zero-init. A regression test asserts `Build(false)` after `Build(true)` on the SAME instance is byte-identical.

**Loop (ablation mode, reconciled — should-fix):**
`solver.Step` → `ablationTool.Advance` → `ablationDetector.Dispatch` { reset `_Deleted` to 0 → `DetectAblation` (classify `_Deleted`, deformed frame) → `SeverDeleted` (NbrIdx/bend) → `FreezeDeleted` (`_CornerActive=0` on deleted) → `RecomputeOccupancy` } → `solver.ComputeRotations` → `dc.Build(ablationMode:true)` (the receded-boundary crossings are computed INSIDE `DC_FeaturePoints`; the cavity boundary is stitched by the `DC_Stitch` runtime gate).

Mode-switch note (should-fix): `_VoxelOccupied`/`_CornerInsideRT`/`_Deleted` are **persistent** GPU buffers. The module is **mode-fixed-at-build** (a `cuttingMode` chosen before `Upload`); switching mode at runtime requires a re-`Upload` to reset them. The blade's `SeverLinks` must **never** dispatch in ablation mode (asserted in tests), and `SeverDeleted` never dispatches in blade mode.

## ComputeParticleRot — actual behavior (must-fix #6, CORRECTED)

`ComputeParticleRot` (Physics.compute:401-506) **EXCLUDES** severed/inactive neighbors: the per-axis loop does `if (nbr < 0) continue;` and `if (_Active[nbr] == 0) continue;` (Physics.compute:428-429). A deleted neighbor (severed by `SeverDeleted` → `NbrIdx=-1`, and finally `_Active=0`) is therefore **simply omitted** from the deformation-gradient sum `F`. A corner that loses **both** arms on an axis (e.g. both neighbors on x deleted) then has a **zero diagonal** on that axis of `M = Σ r_s⊗r_s`; the **M3 per-axis gate** (Physics.compute:448-452, `min(mx,my,mz) < axisEps`) catches it and falls back to **identity R** via `WriteIdentityRot`.

**Important (re-audit bug #5):** the M3 zero-diagonal gate fires only when **BOTH arms of an axis are gone** (that axis's `m·` ≈ 0). Deleting **ONE** neighbor leaves the opposite arm intact, so `mx>0` (etc.) and the **polar path runs** (not M3). The test that exercises M3 must therefore delete **BOTH** neighbors on one axis (or use a grid-boundary corner that has only a single arm on that axis), AND deform the corner enough to **clear the M4 deadband** (Physics.compute:472-485, `‖F-I‖_F < 1e-3`) so the identity result is attributable to the **M3 gate**, not the deadband. See Task 5 test 4b.

**There is NO "[22] Fig4(d) cross-product missing-arm fallback" in this code.** (Berndt Fig 4(d) ref22:332-340 is about estimating a missing cell-edge *orientation* in their Marching-Cubes extraction, NOT about this project's polar-decomposition `ComputeParticleRot`.) The correct statement everywhere is: **a deleted neighbor is excluded → a corner that loses BOTH arms of an axis falls back to identity R.** Identity fallback is declared **paper-silent-acceptable** (do NOT claim Fig4(d)). A corner losing an entire axis writes identity R (not NaN) — verified by a test.

## Reused verbatim (ablation)
`ConnectivityLUT`/`Conn4096` are NOT needed for the cavity (Approach B uses no per-component split). Reused **verbatim**: `RebuildIsectWorld` (unchanged — no appends into `_IsectWorld`); normals (`DC_ClearNormals`/`DC_NormalsScatter`/`DC_NormalsNormalize`); `DualContouring.Build` outer-surface block (gains the `ablationMode` arg + capsule binds but the static path is unchanged); `Bind`; the occupancy gates (`StitchVertId` Recon.compute:221, the `cnt==0` early-out on the static `DC_FeaturePoints` path) — ablated chunks vanish for free once occupancy is recomputed; the `SeverLinks` *write contract* (NbrIdx=-1 + bend alive=0) replicated (not gated on EdgeIsCut) in `SeverDeleted`; Physics force-skip on `_CornerActive==0`/`NbrIdx<0`; `ComputeParticleRot` (with the corrected identity-fallback behavior, §"ComputeParticleRot"); the auto-sweep driver (ReconGridManager.cs:262-280).

**NOT reused verbatim (re-audit):** `DC_FeaturePoints` gains an `_AblationMode` branch (the static branch is byte-identical; the ablation branch does inline filtered-window + capsule-crossing recompute). `DC_Stitch` gains an `_AblationMode` runtime gate (static branch byte-identical). Both are documented as "verbatim static branch + new ablation branch", NOT "reused verbatim".

## Net-new (the only additions)
1. `AblationTool` (swept capsule) + `AblationDetector` (CPU orchestrator).
2. `DetectAblation` kernel (capsule range test on DEFORMED `_CornerPos` → `_Deleted=1`, no flip).
3. `SeverDeleted` kernel (full isolation of deleted corners) + `FreezeDeleted` final `_CornerActive=0` pass.
4. `RecomputeOccupancy` kernel (`_VoxelOccupied` per-frame from surviving-inside).
5. `_AblationMode` branch in **`DC_FeaturePoints`** (Recon.compute): filtered baked window + inline capsule crossings on receded-boundary edges (replaces the deleted `GenerateFrontierIsect` kernel + dynamic-isect buffer). Adds `_CornerInsideRT`/`_Deleted`/`_CapA`/`_CapB`/`_CapR` binds + `CorId()`.
6. `_AblationMode` runtime gate in **`DC_Stitch`** (Recon.compute): derives `insideA` from `_CornerInsideRT`+`_Deleted` over 16B `_GridEdges`/`GridEdgeCount`; static branch byte-identical when off. Uses `CorId()`.
7. `CorId(int3)` helper in `Recon.compute` (mirror Cutting.compute:160).
8. `_CornerInsideRT` (read-only RT copy) + `_Deleted` buffers; `_VoxelOccupied` upload-once → RW. **NO dynamic-isect buffer.**
9. `dc.Build(..., bool ablationMode=false)` arg threading the capsule + `_CornerInsideRT`/`_Deleted` to `kFP`/`kStitch`.
10. `cuttingMode` demo toggle (Blade | Ablation), self-contained.

## Critical invariant (byte-identical when inactive) — TOP REGRESSION GATE
Making `_VoxelOccupied` dynamic + adding the `DC_FeaturePoints` ablation branch + the `DC_Stitch` runtime gate touches kernels the blade output reads. **Requirement:** with ablation inactive (blade or zero-cut mode, `_AblationMode==0`), the pipeline output is **byte-identical** to today:
- `RecomputeOccupancy` reproduces the static OR **exactly on frame 0** (`_Deleted` all-0 ⇒ surviving-inside == baked `cornerInside`).
- The static branch of `DC_FeaturePoints` runs when `_AblationMode==0` (not one instruction changes).
- The static `_SurfaceEdges` path in `DC_Stitch` is used when `_AblationMode==0` (the runtime branch is a verbatim early-out).
- **Stronger gate (should-fix):** a full `dc.Build(ablationMode:false)` on the same sphere yields a **byte-identical `_Tri` buffer + counter** vs the in-test baseline (the static/today path snapshot — there is no committed baseline artifact). **The `_Tri` comparison MUST be order-INSENSITIVE** (re-audit bug #1): `DC_Stitch` appends via `InterlockedAdd(_TriCounter,6)` (Recon.compute:284), so the triangle index ORDER is nondeterministic across dispatches and an index-by-index compare false-fails even baseline-vs-baseline. Assert `TriCounter[0]` equal first, then compare the **SET** of triangles (canonicalize each tri = rotate so the min vertex is first, sort the tri list, assert equal lists) OR a **commutative checksum** (sum/XOR of per-tri hashes).
- **Frame-0 predicate equivalence (should-fix):** with nothing ablated, the runtime-`_GridEdges` gate (derived `insideA`, receded-boundary predicate) selects EXACTLY the same quad set as the static `_SurfaceEdges` list.

## Testing (GPU oracles, `[Category("GPU")]`)

**Test-helper convention (re-audit minor):** the existing GPU tests each define their OWN inline `ILevelSetProvider` + a `LoadShader`-via-`AssetDatabase` helper (e.g. `Qef_GpuOracle_Tests` defines `PlaneX` + an `AssetDatabase.FindAssets("Recon t:ComputeShader")` loader). There is **no** shared `BuildSphere`/`SphereCenter`/`LoadAblationShader`. Each new ablation test defines its helpers **inline per the existing convention** (a sphere `ILevelSetProvider`, an `AssetDatabase` loader for `Ablation.compute`/`Recon.compute`, and a local sphere-center constant) — OR the plan may add ONE shared test-helper file as an explicit deliverable. Do not reference nonexistent shared helpers.

1. **Range select:** only corners within the capsule SDF (on the **DEFORMED** `_CornerPos`) get `_Deleted=1`; the oracle computes expected membership from the **same** deformed positions the kernel uses; `_CornerInsideRT` is UNCHANGED (never flipped); `_CornerActive` UNCHANGED.
2. **Volume loss:** a voxel whose surviving-inside corners are all deleted → `_VoxelOccupied==0` → emits no surface; surviving voxels unchanged.
3. **Inline feature point (Approach B, Option A):** a **fully-interior** voxel (baked `cnt==0`) that loses ≥1 corner to deletion gets a **FINITE in-voxel feature point `w=1`** from the inline capsule crossings (mechanism (2)); a voxel whose surviving-inside corners are all deleted gets `w=0`; each inline crossing pos satisfies `|CapsuleSdf(pos, a, b, r)| < eps` (capsule root on the DEFORMED edge, NOT `fa/(fa-fb)`); the filtered baked window (mechanism (1)) drops crossings whose inside corner is deleted; with `_AblationMode==0` the feature point is byte-identical to today.
4. **Cavity wall stitch (Approach B):** the `DC_Stitch` runtime gate stitches a watertight cavity wall welded to the surviving skin (one shared DC vertex per receded-boundary voxel — NO rim hole), using the inline feature points from `DC_FeaturePoints`. No `_CutFP`/`BuildCutTriangles` involved.
5. **Occupancy byte-identity (critical):** `RecomputeOccupancy` on frame 0 (nothing deleted) == the uploaded static `_VoxelOccupied`; blade-cut + zero-cut output byte-identical to pre-ablation; full `dc.Build(ablationMode:false)` `_Tri`+counter byte-identical via the **order-INSENSITIVE** set/checksum oracle (bug #1), baseline captured **in-test**.
6. **Frame-0 predicate equivalence:** runtime gate quad set == static `_SurfaceEdges` quad set when nothing deleted.
7. **Sever isolation (must-fix #3):** all 6 `NbrIdx` slots of a deleted corner == -1; every incident bend pair `alive==0`; deleted corner frozen (`_CornerActive==0` applied as the final `FreezeDeleted` step).
8. **Physics rotation (must-fix #6 + bug #5):** a corner that loses **BOTH** arms on one axis (both neighbors deleted) AND is deformed enough to clear the M4 deadband writes **identity R** (the M3 zero-diagonal gate), not NaN, and the identity is attributable to M3 (not the deadband); deleted corners are frozen + links severed.
9. **Classify-then-recompute ordering (should-fix):** `DC_FeaturePoints`/`DC_Stitch` read the post-`DetectAblation` `_Deleted` state; running `Dispatch` twice is idempotent (same `_Deleted`, same occupancy, same inline feature points — instant re-run + per-frame `_Deleted` reset).
10. **Mode-switch regression (should-fix):** the blade `SeverLinks` never dispatches in ablation mode; `SeverDeleted` never dispatches in blade mode; mode is fixed at build (re-Upload resets the persistent buffers).

## Paper-alignment trace
"within a certain range" → §0 capsule (paper:492 + [22] capsules ref22:361; capsule shape **paper-silent, grounded in [22]**) · "outside particles directly deleted" → §1 `_Deleted=1` (paper:493-495) · "ablated particles / ablated edges" → §1 (also `_Deleted=1`) (paper:495-496) · "cut points generated along the ablated edges" → the **inline capsule crossings** computed inside `DC_FeaturePoints` mechanism (2) on receded-boundary edges (paper:497-498) — the isosurface intersections of the redrawn surface · "particles are deleted / no longer participate in **force** calculations" → §2 SeverDeleted + `FreezeDeleted` `_CornerActive=0` (paper:493-494,497) · "no longer participate in **surface** generation" (paper:493-495) → covered by **(a)** `RecomputeOccupancy` (a deleted corner drops from occupancy ⇒ its all-deleted voxels vanish) + **(b)** the `DC_FeaturePoints` ablation branch **FILTERING OUT** crossings whose inside corner is deleted (mechanism (1)) so the deleted corner contributes no surface · "all voxel grids updated + surface redrawn" → §3 RecomputeOccupancy + DC re-contour (paper:498-499 = [22] ref22:316-318). The cavity wall is the **re-extracted outer surface** (Approach B), matching "the surface is redrawn", NOT a per-component cut-point centroid wall.

## Decisions
- **Approach B (DC re-contour)** for the cavity wall — the regular outer DC surface re-contours the receded boundary; the blade `_CutFP`/`BuildCutTriangles` is NOT used (paper §2.1.4 "surface redrawn").
- **Inline recompute (Option A)** for the cavity crossings — computed **inside `DC_FeaturePoints`** under `_AblationMode`, NOT via a separate kernel + dynamic-isect buffer (the re-audit proved that buffer is read by no kernel and fails on baked-`cnt==0` interior voxels). This dissolves the RebuildIsectWorld append/ordering concern.
- **DEFORMED frame** for both ablation detection AND crossing — the cavity wall is a deformed-space DC surface; the `deleted ⇔ in-capsule` invariant requires one shared frame; a capsule's radius defeats the thin-plane miss the blade's rest-frame discipline guards against.
- **`_Deleted` is the SOLE deletion flag** — `_CornerInsideRT` stays read-only/un-flipped; `_CornerActive=0` is a final post-emission step (`FreezeDeleted`), not a deletion signal (must-fix #1, #2).
- **Separate demo mode** (`cuttingMode` Blade|Ablation), mode-fixed-at-build — no mixed semantics on a shared buffer.
- **Capsule radius `r`** — tunable public field, default ≈0.4 (≈3 voxels at L=0.125). The capsule **shape** + "within a certain range" radius are **paper-silent, grounded in [22]**.
- **Auto-swept** capsule probe (matches the blade auto-sweep harness); `Valid=true` always (a stationary probe still ablates — should-fix; `Valid` is just `true`, no tautology).
- **Freeze** deleted corners (`_CornerActive=0` final step), not array-removal (simplest, matches paper:493-494).
- **Instant deletion** (paper:497 "directly deleted") → the whole pass RE-RUNS every frame; `_Deleted` is reset each frame (must-fix #7). [22]'s gradual mass-loss is out of scope.
- **Capsule-SDF root** by bisection (~24 iters) or closed-form segment-capsule, in the DEFORMED frame; valid because `deleted ⇔ in-capsule` ⇒ exactly one sign change on a surviving↔deleted edge (re-audit bug #6).

## Risks
- **R1 (top):** dynamic `_VoxelOccupied` + the `DC_FeaturePoints`/`DC_Stitch` ablation branches touch kernels the blade reads → a wrong recompute/gate silently breaks the blade cut + outer skin. Mitigation: the byte-identical frame-0 + full `dc.Build` `_Tri` **order-insensitive** oracles (tests 5, 6) before any Unity check; the static branches are literal early-outs.
- **R2:** the runtime surface-edge gate changes `DC_Stitch`'s domain (all `_GridEdges` + a derived-`insideA` predicate vs the curated static `_SurfaceEdges`) → risk of double/missing quads at the cavity rim + a perf increase. Validate watertightness + frame-0 predicate equivalence; keep it ablation-mode-only.
- **R3:** inline-crossing correctness — the inline branch must read the POST-`DetectAblation` `_Deleted` state (classify-all-then-Build ordering); the capsule-SDF root (not φ lerp) is the only valid crossing on a surviving↔deleted inside↔inside edge (re-audit bug #6); the receded-boundary subtype (surviving ↔ deleted, NOT surviving ↔ φ-outside) is what keeps the root well-posed. **Snapshot consistency (re-audit MINOR):** the `deleted ⇔ in-capsule` invariant (hence the single capsule-SDF sign change ⇒ valid bisection) holds at `Build` time only if `DetectAblation` and the `DC_FeaturePoints` inline crossing read the **same deformed `_CornerPos` + same `_Deleted` snapshot**. The loop order guarantees this: no `solver.Step` runs between `ablationDetector.Dispatch` and `dc.Build`, and `solver.ComputeRotations` (which runs between) writes only `_ParticleRot`, never `_CornerPos` or `_Deleted`. So both reads see one frozen frame.
- **R4 (corrected):** `ComputeParticleRot` corners — a deleted neighbor is **excluded** and a corner that loses BOTH arms of an axis falls back to **identity R** (NOT a cross-product fallback); deleting ONE arm leaves the polar path running (bug #5). Verify it writes identity (M3 gate, deadband cleared), not NaN (test 8).
- **R5:** inline-crossing cost — a large radius makes many voxels evaluate the 12-edge capsule loop in `DC_FeaturePoints`. Bounded by `VoxelCount` (one thread per voxel, ≤12 edge tests + ~24 bisection iters each); ablation-mode-only, so the static path pays nothing.

## Out of scope
- [22]'s heat-diffusion thermal model (§2.1.4 uses simpler range deletion).
- Coexisting blade+ablation on a shared edge (separate mode chosen).
- Gradual mass-loss.
- Per-component cut-FP cavity walls (`BuildCutTriangles`) — superseded by Approach B.
- A separate dynamic-isect buffer / `GenerateFrontierIsect` kernel — superseded by inline recompute (Option A).
