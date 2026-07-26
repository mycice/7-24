# Stage 2C-fix — Per-component surface split (make the cut VISIBLE)

> Fixes the two architectural gaps the runtime audit found (findings.md "2C no visible cut root cause"):
> RC1 the outer skin never splits per component → cut walls hidden inside a closed skin; RC2 cut geometry
> generated in empty non-tissue voxels → spurious "flap". Honors paper §2.1.1 model (A): a cut voxel's single
> feature point is REPLACED by per-component feature points that BOTH the outer-surface stitch AND the cut-wall
> stitch use. Coding-discipline rules 1-3 apply.
>
> USER DECISION: per-component feature point = **unified QEF over {component's isosurface intersection points}
> ∪ {component's cut points}** (most paper-faithful; the skin stays on the isosurface AND the cut wall sits at
> the blade, sharing one per-component dual vertex).

## 0. Goal
A swept blade opens a VISIBLE slit in the deforming sphere: the outer skin separates into two pieces along the
cut (gap ≈ D), the exposed cut walls fill the interior, and NO geometry appears outside the tissue. Physical
separation of the halves is still 2D.

## 1. Root causes recap (what we are fixing)
- **RC1** `DC_Stitch` (Recon.compute) stitches the outer surface using ONE `_VoxelExternalFP[v]` per voxel,
  component-blind → skin never opens. FIX: make the outer stitch select the per-component feature point for
  cut voxels (same `LocalCornerOf`/`Conn4096Comp` keying `BuildCutTriangles` uses).
- **RC2** cut kernels gate only on `voxelCutMask!=0`, never on "is this voxel tissue?" → cut walls in empty
  space. FIX: a per-voxel occupancy gate (≥1 corner inside the isosurface).
- **RC3 (minor)** static `localOffset` cut-point drift under deformation — 2D's R-frame job; out of scope here.
- **Robustness** demo blade at x=3.0 sits exactly on a grid corner plane → double cut-point emission; offset to mid-edge.

## 2. New/changed data

### 2.1 Per-voxel occupancy (RC2 gate)
- CPU: in `BackgroundGrid.Build`, compute `voxelOccupied[v] = (any of the 8 corners has cornerInside==true)`.
  `cornerInside` already exists on the CPU (BackgroundGrid). Expose `byte[] voxelOccupied` (or pack into existing).
- GPU: new `ReconBuffers.VoxelOccupied : uint[VoxelCount]` (stride 4), uploaded once in `Upload`.
- Bind to the cut kernels + DC_Stitch.

### 2.2 Unified per-component feature point
- Keep `_CutFP : float4[8*VoxelCount]` but its MEANING changes: for a cut voxel, `_CutFP[8*v+comp]` is the
  **unified per-component dual vertex** = QEF over {component c's isect planes} pulled to {component c's cut
  centroid}. (Was: centroid of cut points only.)
- Keep `_CutFPAccumPos/Cnt` (the cut centroid) as an INPUT to the unified solve (one constraint), not the final FP.
- `_VoxelExternalFP[v]` (DC_FeaturePoints) is unchanged and used ONLY by UNCUT voxels (component 0 of an uncut voxel).
- `_CutFPNormal/F` unchanged (per-slot normals; now also feed the skin via the tag-aware scatter — already wired).

## 3. Kernel changes (all in Cutting.compute unless noted)

### 3.1 `DetectCut` (Cutting.compute) — RC2 + blade robustness
- Gate cut-bit propagation: when marking a sharing voxel, skip if `_VoxelOccupied[vid]==0` (no tissue → no cut).
  Also skip its cut-point emission if neither sharing tissue voxel exists. (Cut points are owner-bound; only
  emit for edges with ≥1 occupied sharing voxel.)
- (Demo) offset the blade off the corner plane: ReconGridManager blade x = `sphereCenter.x + L*0.5`.

### 3.2 `LookupConnectivity` / `AccumulateCutFP` — RC2 gate
- `voxelFPCount[v] = (mask==0 || _VoxelOccupied[v]==0) ? 0 : compCount`.
- `AccumulateCutFP`: skip scatter into voxels with `_VoxelOccupied[v]==0` (alongside the existing `mask==0` skip).
  The cut centroid accumulators are now an INPUT to §3.3, not the final FP.

### 3.3 `ComputeComponentFP` (NEW, replaces `FinalizeCutFP`) — the unified QEF
One thread per voxel (loop its components). For voxel v with `mask!=0 && occupied`:
```
read mask → compCount, vertToComp (Conn4096)
for comp in 0..compCount-1:
    // gather this component's isosurface intersection planes (pull model, like DC_FeaturePoints)
    int off=_VoxelIsectOffset[v], cnt=_VoxelIsectCount[v];
    accumulate over isect entries i in [off,off+cnt): determine the isect edge's component via the
       INSIDE endpoint corner: lc = LocalCornerOf(insideCornerCoord_of_isect_i, v); if vertToComp[lc]==comp →
       include plane (p_i=_IsectWorld[i], n_i=_Isect[i].normal). [cut+surface edge: assign to inside endpoint's
       component — PAPER-SILENT, rare]
    bool hasIsect = (#included>0);
    // cut centroid for this component (from AccumulateCutFP)
    int slot=8*v+comp; bool hasCut=_CutFPAccumCnt[slot]>0;
    float3 cutC = hasCut ? sum/POS_SCALE/cnt : 0;
    // seed
    x = hasIsect ? mean(included p_i) : (hasCut ? cutC : voxelCenter);
    // QEF gradient descent (mirror DC_FeaturePoints) + cut-plane pull
    for it in 0.._QefIters:
        F = Σ_included n_i*dot(n_i, p_i - x);
        if hasCut: F += _CutPull * (cutC - x);   // isotropic pull to the cut centroid (PAPER-SILENT weight)
        a = 0.1*(1 - it/_QefIters); xn = x + a*F;
        if BoxSdf(xn, voxelCenter, L) >= 0: break;  // keep last interior (D5)
        x = xn;
    _CutFP[slot] = float4(x, 1.0);
```
- If a component has neither isect nor cut points → `_CutFP[slot].w=0` (invalid; skipped by stitch).
- `_CutPull` weight: start ~1.0 (tunable). The cut centroid keeps each component's dual vertex on its side of
  the blade so the two components' skin pieces separate by ≈ D.
- NOTE: `insideCornerCoord_of_isect_i` — `_Isect[i]` stores cornerA/cornerB (global indices); the inside one is
  identifiable by sign (reuse the existing convention `insideA`, or recompute from corner occupancy). Spell out
  in implementation.

### 3.4 `DC_Stitch` (Recon.compute) — RC1, the skin split
Per surface edge (axis, baseCorner, insideA), for each of the 4 sharing voxels:
```
if (_VoxelCutMask[v] != 0 && _VoxelOccupied[v]) {     // cut voxel → per-component FP
    int lc = LocalCornerOf(insideCornerCoord, v);     // the surface edge's INSIDE endpoint corner
    int comp = Conn4096Comp(_Conn4096[mask], lc);
    float4 f = _CutFP[8*v+comp];
    vertId = (f.w>0.5) ? (0x80000000 | (8*v+comp)) : voxelId_external_fallback;
} else {                                               // uncut → existing single FP
    vertId = voxelId;                                  // external tag (high bit clear)
}
```
- The inside endpoint corner is a corner of all 4 sharing voxels (an edge's endpoints are corners of its 4
  voxels) → `LocalCornerOf` succeeds. Same numbering chain as BuildCutTriangles (re-weld-proof).
- When the 4 voxels disagree on component (a cut runs between them), the quad's 4 vertices come from different
  per-component FPs on opposite sides → the outer quad SPLITS → skin opens. This is the core RC1 fix.
- Needs Conn4096 + CutFP + VoxelCutMask + VoxelOccupied bound to DC_Stitch (cross-buffer; add in DualContouring.Bind).
- Fallback to the external FP if the component FP is invalid (degenerate) — avoids holes.

### 3.5 `BuildCutTriangles` — unchanged logic, now consumes the unified `_CutFP`
Already uses `_CutFP[8*v+comp]`; with §3.3 these are the unified dual vertices, so cut walls connect seamlessly
to the split skin. Add the `_VoxelOccupied` gate (skip empty voxels) for symmetry.

### 3.6 Dispatch order (DualContouring.Build) — update
```
RebuildIsectWorld → DC_FeaturePoints (external FP, uncut voxels)
ClearCutFP → LookupConnectivity(occupancy-gated) → AccumulateCutFP(occupancy-gated, cut centroid)
ComputeComponentFP [replaces FinalizeCutFP] (unified per-component QEF)   // needs Isect/IsectWorld/VoxelIsect* bound
reset TriCounter
DC_Stitch [now component-aware]      // needs Conn4096/CutFP/VoxelCutMask/VoxelOccupied bound
BuildCutTriangles
DC_ClearNormals → DC_NormalsScatter(tag-aware) → DC_NormalsNormalize → DC_NormalizeCutFP
DC_WriteIndirectArgs
```

## 4. Why this opens the cut (geometric argument)
A cut boundary voxel now contributes TWO dual vertices (one per component), each on its own side of the blade
(pulled to that component's cut centroid, offset ≈ ±D/2). The outer surface quads incident to that voxel use the
component vertex matching their inside corner → the skin sheet that used to pass straight through now terminates
at two separated vertices → the skin opens by ≈ D. The cut walls (BuildCutTriangles) bridge the same two vertices
across the cut → the interior is shown. Empty voxels are gated out → no flap.

## 5. Verification
- **Reuse + extend `CutSurface_GpuOracle_Tests`** but with a NON-trivial level set (the demo `SphereLevelSet` or a
  half-inside slab) so the outer skin is NON-empty (the old `AllInsideLS` structurally hid RC1/RC2):
  1. **No empty-voxel geometry (RC2):** assert no `_CutFP` slot is valid and no cut/skin triangle references a
     voxel with `_VoxelOccupied==0`.
  2. **Skin splits (RC1):** for a surface edge straddling the cut, assert its quad uses DIFFERENT per-component
     `_CutFP` slots for the two sides (vertex ids differ across the cut), i.e. the outer quad is no longer welded.
  3. **Per-component FP sane:** each cut boundary component's `_CutFP` is valid, within the voxel box, and the two
     components' FPs are separated (≈ along n_cut).
  4. Keep the 2C re-weld / count-divisor / LocalCornerOf round-trip tests green.
- **Unity Play (user):** the slit is VISIBLE through the sphere (two skin flaps + cut walls, gap ≈ D); no flap
  above/below; outer surface elsewhere unchanged; FPS stable.

## 6. Risks / watch-items (for the audit)
- **R1** `ComputeComponentFP` isect→component assignment for edges that are BOTH cut and surface edges (assign to
  inside endpoint's component — PAPER-SILENT; verify rare + correct).
- **R2** `_CutPull` weight + QEF stability: the cut-plane pull must not push the FP outside the voxel/its component;
  reuse the D5 BoxSdf interior-keep. Tune; default 1.0.
- **R3** DC_Stitch fallback when component FP invalid → external FP (no holes); confirm the tag/normal path handles it.
- **R4** occupancy gate correctness: boundary voxels (VoxelIsectCount>0) AND interior tissue voxels (all-inside,
  IsectCount==0) must both count as occupied; only all-outside voxels excluded. Verify the CPU `voxelOccupied` OR.
- **R5** no Stage-1 regression: uncut voxels still use `_VoxelExternalFP` via the external tag; zero-cut scene byte-identical.
- **R6** normals for split skin: the per-component skin vertices now scatter into `_CutFPNormal` (internal tag) —
  confirm two-sided lighting still reads correctly for the skin (not just the walls).

## 7. REV 2 — audit resolutions (2 auditors: A1 paper/correctness opus, A2 GPU feasibility sonnet). All folded in; implement THIS.
Both: re-weld-proof YES; paper-faithful YES (one declared deviation); feasible WITH these changes.

- **[B1 CRITICAL — isect→component needs an inside flag] IsectGpu has no inside-endpoint info on the GPU.** FIX:
  repurpose the unused `IsectGpu._pad` (ReconBuffers.cs / ReconCommon.hlsl) → `int insideA` (1 ⇒ cornerA is the
  INSIDE endpoint, else cornerB). `BackgroundGrid.Build` already computes `ia = cornerInside[ca]==1` — store it on
  the `IsectEntry` and marshal into `IsectGpu.insideA` in `ReconBuffers.Upload`. **Stride stays 40 bytes.**
  Verify nothing else read `_pad`.
- **[IMPORTANT-3 — guarantee ≈D separation] Assign each isect plane to ONLY its inside-endpoint's component**
  (NOT to both — assigning to both collapses the two components onto the same isosurface patch and re-occludes the
  slit on the boundary ring). So in `ComputeComponentFP`, include isect entry `i` in component `c`'s QEF iff
  `Conn4096Comp(mask, LocalCornerOf(insideCornerCoord(i), v)) == c`, where `insideCornerCoord(i) =
  CornerCoordHLSL(it.insideA==1 ? it.cornerA : it.cornerB)`. The two components then use DIFFERENT planes + are
  seeded at their DIFFERENT cut centroids (±D/2 apart) → they separate. **Add a HARD test**: the two boundary
  components' `_CutFP` separation ≥ 0.5·D (numeric floor, not a sanity check). If real-run still collapses, add the
  cut-plane projection fallback (project each FP onto its component's half-space) — note as a reserved safeguard.
- **[Paper deviation — declare] fused QEF(isect)+cut-centroid into ONE per-component vertex for boundary cut
  voxels** is a generalization the paper is silent on (it defines the outer QEF FP and the cut centroid FP in
  separate contexts). Declare a new PAPER-SILENT D-item in findings.md. Interior cut voxel (no isect) → solve
  degenerates to the pure cut centroid = paper-exact (line 344).
- **[B3 + M2 — includes & cross-bindings]**
  - Add `#include "ReconCommon.hlsl"` to Cutting.compute BEFORE `#include "CuttingCommon.hlsl"` (gives `IsectGpu`,
    `BoxSdf`, and the canonical `NORMAL_SCALE`; CuttingCommon's `#ifndef NORMAL_SCALE` guard then no-ops). Declare
    in Cutting.compute: `_Isect, _IsectWorld, _VoxelIsectOffset, _VoxelIsectCount, _VoxelCorner, _VoxelOccupied`,
    uniforms `_QefIters, _L`.
  - Add `#include "CuttingCommon.hlsl"` to Recon.compute AFTER `ReconCommon.hlsl` (gives `Conn4096Gpu`,
    `LocalCornerOf`, `Conn4096Comp`; guards prevent NORMAL_SCALE double-def). Declare in Recon.compute:
    `StructuredBuffer<Conn4096Gpu> _Conn4096; StructuredBuffer<uint> _VoxelCutMask; StructuredBuffer<uint> _VoxelOccupied;`
    (`_CutFP` already declared there).
  - `DualContouring.BindCut`: add `_Isect, _IsectWorld, _VoxelIsectOffset, _VoxelIsectCount, _VoxelCorner,
    _VoxelOccupied`. `Build` sets `cut.SetInt("_QefIters",..)` + `cut.SetFloat("_L",..)`.
  - `DualContouring.Bind` (Recon kernels): add `_Conn4096, _VoxelCutMask, _VoxelOccupied` (harmless to kernels
    that ignore them; needed by DC_Stitch).
- **[IMPORTANT-1 — DC_Stitch keying] derive the inside corner from `ge.insideA`:** `insideCornerCoord =
  (ge.insideA!=0) ? baseCorner : baseCorner+axisDir;` then `comp = Conn4096Comp(_Conn4096[mask],
  LocalCornerOf(insideCornerCoord, v))`. Keying off `baseCorner` unconditionally re-welds every `insideA==0` edge.
  Fallback to external `_VoxelExternalFP` (voxelId tag) if `_CutFP[slot].w<0.5` (degenerate → no hole; local re-weld OK).
- **[VoxelOccupied buffer] (RC2 gate)** new `ReconBuffers.VoxelOccupied : uint[VoxelCount]` (stride 4) =
  CPU per-voxel OR of the 8 corners' `cornerInside`. Alloc/Upload/Dispose/bind (to BOTH Bind + BindCut). Gate:
  `LookupConnectivity`/`AccumulateCutFP`/`ComputeComponentFP`/`BuildCutTriangles` skip `occupied==0`; `DetectCut`
  skips cut-bit propagation into `occupied==0` voxels, and gates cut-point emission at EDGE granularity (emit iff
  ≥1 of the edge's stencil voxels is occupied — apply before the `firstCut` emit, not per-voxel).
- **[kernel rename] `FinalizeCutFP` → `ComputeComponentFP`** (`#pragma kernel`, `DualContouring` handle
  `kFinalizeCutFP`→`kComputeComponentFP` + FindKernel + dispatch). The unified QEF replaces the centroid-only finalize.
- **[M1 — oracle harness] existing Stitch/DeformCoupling GPU tests must bind `_VoxelCutMask`(zeros),
  `_VoxelOccupied`(ones), `_Conn4096` to `kStitch`** or Unity errors "buffer not set". Update those test setups.
- **[zero-cut regression] confirmed byte-identical** (mask all-zero → DC_Stitch `else` branch = current path);
  ADD a zero-cut assertion so a future branch-predicate regression can't silently change Stage-1 output.
- **[robustness] blade x = sphereCenter.x + L*0.5** (off the corner plane) — keep.
- **[dispatch] `ComputeComponentFP` after `AccumulateCutFP`, before `DC_Stitch`/`BuildCutTriangles`; needs
  RebuildIsectWorld (deformed `_IsectWorld`) already run (it is, step 1). `DC_FeaturePoints` still computes
  `_VoxelExternalFP` for uncut voxels.
