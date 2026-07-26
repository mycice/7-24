# Findings — Cutting algorithm (paper §2.1 + design §5)

> Traceability reference for Stage 2. Every implementation block must cite the relevant item here / the paper.
> Source: `newpaper_clean.txt` (paper §2.1, Eq6-8, Fig 2.3–2.9) + design spec §5 + gap-analysis `dc_gap_findings.md`.

## Conventions already frozen in Stage 1 (reuse, do NOT redefine)
- Fig 2.4 voxel numbering: corners V0..V7; edges e0=V0V1,e1=V1V2,e2=V2V3,e3=V3V0,e4=V4V5,e5=V5V6,e6=V6V7,e7=V7V4,e8=V0V4,e9=V1V5,e10=V2V6,e11=V3V7. (`GridConventions.cs`)
- Each vertex has exactly 3 incident edges (V0={e0,e3,e8}, etc.).
- Each grid edge shared by 4 voxels; per-axis (offset, localEdge) stencil in `GridConventions.EdgeStencil`.
- `voxelId = i + Nx*(j + Ny*k)`; corner = particle (mass-spring node); fixed-point int atomics (FORCE_SCALE 1<<10, NORMAL_SCALE 1<<20).

## §2.1.1 Static cutting — the core (design §5.4–5.6)
- A cut voxel = undirected graph (8 vertices, 12 edges). Each edge: cut(1)/not(0) = 1 bit → **12-bit index → 2^12 = 4096 connectivity LUT** ("stored as a 2D array"; bitwise-OR to update when edge i is cut). [paper §2.1.1; design §5.4]
- A cut edge **disconnects** the edge in the connectivity subgraph AND **contributes a cut point**. Components = connected components of the UNCUT-edge subgraph. A cut voxel → **up to 8 components → up to 8 feature points**. [§2.1.1; gap C12/C13]
- **vertToComp** (design D-decision, gap C4): LUT must give, per VERTEX, its component id (vertex-keyed, dense 0..compCount-1). NOT edge-keyed — a cut edge straddles two components. Fig 2.4 example: cut {e0,e2,e4,e6} → 2 comps {V0,V3,V4,V7}/{V1,V2,V5,V6}. All-12-cut → 8 singletons.
- **Cut feature point = centroid of the component's cut points** (plain mean, NOT QEF). [§2.1.1 "feature point is the centroid of all the cut points"]
- **Per-particle stitching** (Fig 2.5): traverse each particle's 6 axis edges + 8 octants (surrounding voxels). For each CUT edge, gather the feature point of **the component containing this particle's local vertex** in each of the ≤4 shared voxels → quad → 2 triangles. Wrong component selection = re-welds the seam (the MC2024 failure). [§2.1.1; gap C15/C54]
- **Two cut points per cut edge**, offset ±D/2 (D = tool thickness, mm-scale, sub-voxel), each bound to one endpoint particle → gap width = D. [§3.4 Fig 3.8; gap C16/C19/C25]

## §2.1.3 Cutting path + detection (design §5.1–5.2)
- Tool = a LINE segment (endpoints S top / E bottom, thickness D). **Cutting plane = swept quad of the tool line over prev+current frame**, split by diagonal S_k→E_{k+1}: T1=(S_k,E_k,E_{k+1}), T2=(S_k,E_{k+1},S_{k+1}). Persists each frame (no missed collisions at low FPS). [§2.1.3 Fig 2.8; gap C22]
- **Collision = Möller-Trumbore (Eq6-8): ray = VOXEL grid edge (O=endpoint, D=edge vector), triangle = swept-plane triangle.** [gap C21 — NOT reversed]
  - Eq7: `[D, -L1, -L2][t,u,v]^T = T`, `T = V0 - O`, `L1=V1-V0, L2=V2-V0`.
  - Eq8 Cramer (shared denom `det=(L2×D)·L1`): `t=((L2×T)·L1)/det`, `u=-((L2×D)·T)/det`, `v=((D×T)·L1)/det`. (u carries the leading minus.)
  - Hit iff `u≥0, v≥0, u+v≤1` AND **`0≤t≤1`** (finite edge; gap C24). Cut point = `O + t·D`.
- **Cut bit propagates to all 4 voxels** sharing the hit edge, each at its own local edge id. [gap C1/S2-I7]

## §2.1.2 Dynamic cutting (design §5.7)
- Deformed grid ≠ cube; each particle has a rotated local frame R. Paper only cites Berndt[22] (no formula) → **decision: recover R by polar decomposition / shape-matching from rest neighbor offsets → current neighbor offsets** (PAPER-SILENT, cite [22]). [gap C17]
- Cut point anchored to ONE particle's frame: `world = particle.pos + R·localOffset` (3-DOF; Fig 2.7 red point swings with V3's frame). Re-evaluated each frame so the cut follows deformation. [gap C18/C51]
- On cut: sever the structural spring on that edge AND **invalidate all bending pairs using that edge as an arm** (else ghost forces across the cut). [gap C20/C35]

## §2.1.4 Ablation (design §5.9)
- Ordered TWO passes (cannot merge): (1) particles OUTSIDE isosurface in tool range → deleted (no cut points); (2) particles INSIDE → "ablated" → deleted, mark "ablated edges" (incident to ablated particles incl. toward surviving neighbors), generate cut points → **funnel into the SAME 4096-LUT + centroid + redraw pipeline**. [§2.1.4 Fig 2.9; gap C26/C27/C28]

## Params (paper §3.4)
voxel 0.25cm; D = 0.1–0.4 **mm** (sub-voxel); ε (cut threshold) — but DC uses M-T t∈[0,1] not a distance ε.

## Deviations relevant to Stage 2 (none new yet; reuse D1-D6). New PAPER-SILENT to declare: Berndt frame = polar decomp; cut-point ±D/2 offset direction; vertToComp dense renumber; partial-quad <4-corner rule; ablation orphan cut-point → surviving endpoint.

## 2C "no visible cut" root cause (2026-06-28, 3-subagent runtime audit; tests green but demo shows no slit + spurious flap)
Two INDEPENDENT architectural gaps, both confirmed by paper + code:
- **RC1 (no slit) — the OUTER surface never splits per component.** Paper §2.1.1 (lines 314-354) model is **(A)**: a cut voxel's single feature point is REPLACED by up to 8 PER-COMPONENT feature points, and BOTH the outer-surface stitch (over isosurface-intersecting edges) AND the cut-wall stitch (over cut edges, Fig 2.5) use the component-appropriate FP → the dual skin physically separates. Our impl is **(B)**: `DC_Stitch` (Recon.compute) uses ONE `_VoxelExternalFP[v]`, component-blind; `BuildCutTriangles` adds cut walls as SEPARATE geometry. The closed skin HIDES the interior cut walls → no visible slit. **My 2C design §4.4 had this gap too** (external=single FP / internal=cut FP treated separately); the 5-dim plan audit missed it because it's a runtime-visibility gap, not a static-correctness one. Fix: make `DC_Stitch` component-aware — for a cut voxel select the per-component FP via the same `LocalCornerOf`/`Conn4096Comp` keying `BuildCutTriangles` uses; unify outer+cut FP per component.
- **RC2 (the flap) — cut geometry generated in EMPTY non-tissue voxels.** Demo blade spans y∈[0.2,5.8] but sphere is y∈[1,5]; `gridEdges` = ALL interior edges (no sign filter), and `DetectCut`/`AccumulateCutFP`/`BuildCutTriangles` gate ONLY on `voxelCutMask!=0` — never on "is this voxel tissue?". So edges the blade crosses ABOVE/BELOW the sphere get cut → cut walls in empty space, exposed (no skin to hide them) = the gray flap. Fix: gate cut generation to tissue voxels. `VoxelIsectCount[v]>0` is already on the GPU but NOT bound to cut kernels (boundary voxels only); for interior tissue voxels need a per-voxel occupancy (≥1 corner inside) — `BackgroundGrid.cornerInside` exists on CPU but isn't uploaded.
- **RC3 (secondary, seam drift)**: `AccumulateCutFP` world = `cornerPos[owner]+localOffset` with static world-offset captured at cut time → drifts as the sphere deforms. This is 2D's R-frame job; minor for 2C.
- **Robustness**: demo blade at x=3.0 sits EXACTLY on a grid corner plane (L=0.25 → corner i=12) → M-T hits at t=0/t=1 (handled by closed interval, but double-emits cut points for the shared corner). Offset blade to mid-edge (x=3.125) for cleanliness.
- **Why tests passed**: `CutSurface_GpuOracle_Tests` uses `AllInsideLS` (all corners inside) → zero sign-change edges → `surfaceEdges` empty → `DC_Stitch` emits NO skin → tests exercise `BuildCutTriangles` in isolation, never the skin-vs-cutwall occlusion or the empty-voxel case. **Lesson reinforced: real-run + multi-subagent runtime audit catches what static green can't.**

## 2C-fix implemented (2026-06-28, plan 2026-06-28-dc-2c-fix-per-component-surface §7 REV 2)
- **D-item (NEW, PAPER-SILENT) — fused per-component dual vertex.** For a BOUNDARY cut voxel,
  `ComputeComponentFP` (Cutting.compute, replaces `FinalizeCutFP`) solves ONE per-component feature
  point as a unified QEF over {that component's isosurface-intersection planes} pulled toward {that
  component's cut centroid} (weight `_CutPull=1.0`). The paper (§2.1.1) defines the outer-surface QEF FP
  and the cut-centroid FP in SEPARATE contexts and is silent on fusing them; this generalization is what
  lets the SAME per-component vertex serve both the outer skin (DC_Stitch) and the cut wall
  (BuildCutTriangles) so they share one dual vertex and the slit opens. **Interior cut voxel (no isect)
  → the solve degenerates to the pure cut centroid = paper-exact.** Reserved safeguard (NOT yet needed):
  if a real run shows the two components collapsing, project each FP onto its component's cut half-space.
- **RC1 fix**: `DC_Stitch` (Recon.compute) now resolves each of a surface edge's 4 sharing voxels to a
  per-component cut FP via the surface edge's INSIDE-endpoint corner (`insideA`→`LocalCornerOf`→
  `Conn4096Comp`) — the SAME keying `BuildCutTriangles` uses (re-weld-proof). Uncut/empty/degenerate →
  external `_VoxelExternalFP` fallback (no holes). Zero-cut → mask all-zero → external branch → Stage-1
  byte-identical (locked by a new zero-cut assertion in `Stitch_GpuOracle_Tests`).
- **RC2 fix**: new `BackgroundGrid.voxelOccupied[v]` = OR of the 8 corners' `cornerInside`, uploaded to
  `ReconBuffers.VoxelOccupied` (uint[VoxelCount]); gates `DetectCut` (bit propagation + EDGE-granularity
  cut-point emit), `LookupConnectivity`, `AccumulateCutFP`, `ComputeComponentFP`, `BuildCutTriangles`,
  and `DC_Stitch`. No cut geometry in all-outside voxels → no flap.
- **B1**: `IsectGpu._pad`→`int insideA` (1⇒cornerA inside; stride stays 40); set from
  `BackgroundGrid.IsectEntry.insideA`. Lets each isect plane be assigned to ONLY its inside-endpoint's
  component (plan §7 IMPORTANT-3) → the two halves use DIFFERENT planes → they separate (≥0.5·D HARD test floor).
- **Robustness**: demo blade x = `sphereCenter.x + L*0.5` (off the corner plane).
