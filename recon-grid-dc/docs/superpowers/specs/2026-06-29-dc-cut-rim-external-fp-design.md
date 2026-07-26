# Spec — Restore the paper's External / Internal feature-point split at cut boundary voxels

Date: 2026-06-29
Status: DESIGN (approved by user) → writing-plans → implement (multi-subagent review)
Diagnosis: workflow w0kkfnvm7 (high confidence, paper-grounded)

## 1. Problem

After a cut, the **cut margin / cross-section** (the rim where the cut plane meets the model's outer
surface) renders **jagged / faceted / "square"** instead of following the model's original outer contour
(a horizontal cut of a sphere should leave a circular rim; a liver should leave a liver-shaped margin).
Only the cut RIM is affected — the rest of the surface stays correct.

**This is NOT a request to "make the cut rounder."** The goal is to make the cut margin **preserve the
model's ORIGINAL mechanical form / contour** — whatever shape the isosurface is. For the sphere that
contour happens to be round; for the liver it is irregular and must stay liver-shaped. The fix restores
the surface to its original isosurface at the cut, it does NOT fit a circle or smooth anything toward a
regular shape.

## 2. Root cause (diagnosed against the paper — workflow w0kkfnvm7)

The paper defines **two distinct feature points per cut boundary voxel** (Table 3 glossary,
newpaper_clean.txt:1340-1344):
- **External feature point** — "A feature point located on the outer surface." This is the standard
  Dual-Contouring / QEF feature point (Eqs. 1-5, clean:232-296: minimise `Σ (n_i·(x−p_i))²` over the
  isosurface intersection points + normals, box-clamped by the voxel SDF). It lies **on the original
  isosurface** (the model's true contour). The OUTER SKIN connects to it.
- **Internal feature point** — "A feature point located on the inner surface." This is "the centroid of
  all the cut points" per connected component (clean:343-344). It lies on the **cut plane** (the flat cut
  face). The CUT WALL connects to it.

The paper is explicit (clean:320-321): *"The surface generation during the cutting process **differs from
that of the outer surface**."* So the centroid-of-cut-points rule is **only** the cut-wall (Internal) FP;
it is NOT the outer-skin vertex.

**Our divergence (single place):** the implementation collapsed these two into ONE `_CutFP`:
- The 2C-fix made the OUTER skin read the per-component `_CutFP` so the skin would split per component and
  reveal the cut (`Recon.compute` StitchVertId:215-227 tags the outer-skin vertex internal; FetchPos:325-330
  resolves it to `_CutFP[slot].xyz`, NOT the isosurface point `_VoxelExternalFP[v]`).
- The later holes fix (workflow wggai2v70) set `_CutFP[slot]` = the **pure centroid of cut points**
  (`ComputeComponentFP`, Cutting.compute:624-637), which sits on the **flat blade plane** (cut points are
  emitted at ±D/2 along `n_cut`, EmitCutPoint:262).
- Net: the OUTER skin at every boundary cut voxel is dragged from its round, isosurface-conforming External
  point onto the flat blade plane → the faceted/square rim. Non-cut voxels still read the round
  `_VoxelExternalFP` (`DC_FeaturePoints`) — which is exactly why **only the rim** is faceted.

The round/contour-preserving External point **still exists and is still correct** (`_VoxelExternalFP`,
written by `DC_FeaturePoints`); it is simply no longer used at boundary voxels.

## 3. Design — decouple the External (skin) and Internal (wall) feature points

Restore the paper's two-FP model. **Do NOT touch the holes/jitter fix or the cut wall** — the Internal FP
(pure centroid) stays exactly as is. Only the OUTER SKIN's vertex source changes.

### 3.1 Keep (unchanged)
- `_CutFP[slot]` = per-component **Internal feature point** = pure cut-point centroid (blade plane). The
  CUT WALL (`BuildCutTriangles`) keeps using it. (Preserves the holes/jitter fix — workflow wggai2v70.)
- Non-cut voxels' outer skin keeps using `_VoxelExternalFP` (`DC_FeaturePoints`). Unchanged.

### 3.2 Add — per-component External feature point for the skin
- New buffer `_CutFPExternal[slot]` (same shape as `_CutFP`: 8·VoxelCount float4 slots; xyz = position,
  w = validity).
- `ComputeComponentFP` computes it per component of a cut voxel: the **External feature point = the
  isosurface-conforming point over THAT component's isosurface-intersection planes** — the paper's Eqs. 1-5
  External FP, restricted to the component's isect planes (which `ComputeComponentFP` already gathers via
  the inside-endpoint assignment, the 2C-fix B1 mechanism).
  - **Form (sub-decision, resolved in the plan/audit):** primary = the **per-component isosurface QEF**
    (reuse the proven, stable `DC_FeaturePoints` QEF approach over the component's isect planes — most
    faithful to the paper's External FP and to the original contour). Deterministic fallback = the
    component's **`isectMean`** (already computed; no per-frame re-solve, but sits sub-voxel INSIDE the
    surface). The audit picks one by confirming the QEF is as jitter-free as `DC_FeaturePoints` (it should
    be — it is the same machinery, and a boundary component always has isect planes so its validity never
    flips, so it cannot re-introduce the holes).
  - **Validity:** `_CutFPExternal[slot].w` valid iff the component has ≥1 isosurface-intersection plane
    (i.e. that component is on the outer surface). An interior-only cut component (cut but not on the
    surface) has no External FP — correct, it has only a cut wall, no outer skin.

### 3.3 Change — the OUTER skin resolves to the External FP (one data-flow edit)
- `Recon.compute` StitchVertId / FetchPos: for a cut, occupied boundary voxel, resolve the OUTER-skin
  vertex to `_CutFPExternal[slot]` (on the isosurface — preserves the original contour) instead of
  `_CutFP[slot]` (the flat blade plane).
- The skin **still splits per component** (each component's External FP is a distinct isosurface point →
  the cut stays visible, the 2C-fix RC1 is preserved), but every skin vertex now lies on the model's
  original isosurface → the rim follows the original contour (round sphere → round; liver → liver-shaped).
- `BuildCutTriangles` (the cut WALL) keeps using `_CutFP` (the Internal centroid). The skin (External) and
  the wall (Internal) are now two separate surfaces meeting at the rim = isosurface ∩ cut-plane.

### 3.4 Why this preserves the original form (not "rounds" it)
The External FP is the isosurface QEF — it follows the model's actual surface normals/intersections. For a
sphere it lands on the sphere (round); for a liver it lands on the liver surface (irregular). It performs
NO circle-fitting and NO regular-shape smoothing. It is literally the DC feature point the outer surface
already uses everywhere else — we are just no longer overriding it with the flat cut centroid at the rim.

## 4. No-regression
- The cut WALL (`_CutFP` centroid) and the holes/jitter fix are untouched → the cut still opens with gap D,
  no holes, no jitter.
- Non-cut voxels (`_VoxelExternalFP`) untouched → the rest of the surface is byte-identical.
- The External FP is deterministic and always valid for a boundary component (has isect planes) → it cannot
  re-introduce the holes (which were `_CutFP.w` validity flips on a per-frame re-solve).
- Composes with 2D-2 (the External FP positions reconstruct via the deformed isect world positions, so the
  rim follows the deforming/rotating tissue) and 2D-2b (temporal smoothing applies to the cut wall FP;
  whether it also applies to the External skin FP is a plan detail — default: smooth both consistently, or
  leave the skin un-smoothed since it is already stable; audit to decide).

## 5. Testing
- GPU oracle `CutRimExternalFP_GpuOracle_Tests`: build a known boundary cut voxel (isosurface through it +
  a cut); assert (a) `_CutFPExternal[slot]` lies on the isosurface (within tol of the analytic surface),
  NOT on the blade plane; (b) `_CutFP[slot]` lies on the blade plane (unchanged); (c) the two are distinct;
  (d) per-component: the two components get distinct External FPs (skin splits).
- Regression: existing CutSurface oracle — the cut WALL output unchanged; non-cut surface unchanged.
- Visual (user): the cut rim follows the sphere contour (round) — and, when a liver/irregular model is
  used later, follows the liver margin (the real acceptance criterion).

## 6. Out of scope
- Any circle-fitting / shape-regularisation (explicitly NOT wanted).
- Changing the cut wall, the severing, detection, 2D-2, or 2D-2b.
- 2D-3 ablation (the next stage, separate spec/plan).

## 7. Open sub-decisions for the plan/audit
1. External-FP form: per-component isosurface QEF (primary, paper-faithful) vs `isectMean` (deterministic
   fallback). Resolve by confirming the QEF is jitter-free (same machinery as `DC_FeaturePoints`).
2. Whether 2D-2b temporal smoothing should also apply to `_CutFPExternal` (default: leave the skin FP
   un-smoothed — it is already stable; smoothing it would lag the rim behind the deforming surface).
3. Buffer packing: a parallel `_CutFPExternal` buffer (clean) vs packing into spare lanes (avoid — clarity).
