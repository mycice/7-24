# Stage 2D-1 — Physical severing (make the cut REAL: halves separate under gravity)

> Detailed design for the BLOCKER from the audit: the cut is currently visual-only. Paper §2.1.2 line 367
> "when the cut is complete, the physical connection of that edge is severed"; line 1162-1163 (Fig 3.9) the
> halves separate; line 520-522 explicitly warns against the present state (visually severed but forces still
> act across the ends). Strictly paper-aligned. Discipline rules 1-3 apply.

## 0. Goal
When a grid edge is cut, break the mass-spring physical connection on that edge so the two halves move
independently and separate under gravity — matching paper Fig 3.9. No change to the deformation algorithm
itself; only the topology (which links are live) changes on a cut.

## 1. Mechanism (verified against the code)
The Stage-1 force kernels are gather-based and ALREADY honor the severing guards — only the WRITE side is missing:
- **Structural force** `AccumulateStructural` (Physics.compute:80-110): per particle p, gathers its 6 neighbor
  slots and `if (_NbrIdx[6*p+s] < 0) continue;` (line 94). Force on p only (gather → AtomicAddForce to p). So a
  spring between A and B vanishes ONLY when BOTH `_NbrIdx[6*A+slotToB]=-1` AND `_NbrIdx[6*B+slotToA]=-1` (each
  side gathers independently). Severing MUST write both directions.
- **Bending force** `AccumulateBending` (Physics.compute:121-128): one thread per bending pair, `if (bp.alive==0)
  return;` (line 128). So zeroing a pair's `alive` removes its contribution to all of i/j/k.

Topology layout (BackgroundGrid.cs):
- `nbrIdx[6*c + s]`, slot order **0=−x, 1=+x, 2=−y, 3=+y, 4=−z, 5=+z** (NbrDelta, BackgroundGrid.cs:236-240); −1 = none.
- `bendPairs[15*c + p]` = `{int i(center), j, k, float theta0, int alive}` (BackgroundGrid.cs:28-29, 309-327); a
  pair's two arms are the edges (i,j) and (i,k). The cut edge (A,B) is an arm of a pair iff the pair is centered
  at A with j or k == B, or centered at B with j or k == A (A,B adjacent → no third center can have both as
  opposite-axis arms). So scanning A's 15 pairs (arm==B) and B's 15 pairs (arm==A) is COMPLETE.
- Bending `alive` and the pair's stored `j/k` are independent of `nbrIdx` (captured at build), so the scan is
  correct AND idempotent regardless of nbrIdx state.
- `Springs` buffer is NOT read by any force kernel (ReconBuffers.cs:246-248) — the load-bearing structural link
  is `_NbrIdx`. Leave Springs untouched (optional `alive` field later for bookkeeping; out of scope).

## 2. The `SeverLinks` kernel (NEW, in Cutting.compute — has the GridEdges decode + EdgeIsCut + CorId)
One thread per interior grid edge (reuse `_GridEdges`, gridEdgeCount). Idempotent → dispatch every frame.
```hlsl
[numthreads(64,1,1)]
void SeverLinks(uint3 id : SV_DispatchThreadID)
{
    uint e = id.x; if (e >= (uint)_GridEdgeCount) return;
    GridEdgeGpu2 ge = _GridEdges[e];
    int axis = ge.axis; int3 baseC = int3(ge.bcX, ge.bcY, ge.bcZ);
    if (!EdgeValid(axis, baseC)) return;       // in-range (same predicate as the gridEdges build)
    if (!EdgeIsCut(axis, baseC)) return;       // only cut edges (cumulative VoxelCutMask) — paper:367
    int3 axisDir = int3(axis==0?1:0, axis==1?1:0, axis==2?1:0);
    int idA = CorId(baseC);                     // edge's A endpoint (lower-coord)
    int idB = CorId(baseC + axisDir);           // B endpoint (A's +axis neighbor)
    int sPlus  = 2*axis + 1;                     // +x=1,+y=3,+z=5 — idB is idA's +axis neighbor
    int sMinus = 2*axis;                         // -x=0,-y=2,-z=4 — idA is idB's -axis neighbor
    // (1) sever the structural link, BOTH directions (Physics.compute:94 skips -1)
    _NbrIdx[6*idA + sPlus]  = -1;
    _NbrIdx[6*idB + sMinus] = -1;
    // (2) sever bending pairs whose arm crosses the cut edge (Physics.compute:128 skips alive==0)
    for (int p = 0; p < 15; p++)
    {
        int ia = 15*idA + p; BendPairGpu a = _BendPairs[ia];
        if (a.alive != 0 && (a.j == idB || a.k == idB)) { a.alive = 0; _BendPairs[ia] = a; }
        int ib = 15*idB + p; BendPairGpu b = _BendPairs[ib];
        if (b.alive != 0 && (b.j == idA || b.k == idA)) { b.alive = 0; _BendPairs[ib] = b; }
    }
}
```
Traceability comment block cites paper:367 / line 520-522 and Physics.compute:94/:128. `EdgeValid`/`EdgeIsCut`/
`CorId`/`GridEdgeGpu2` are the existing Cutting.compute helpers (no new table).

## 3. Cross-shader struct + buffer declarations (Cutting.compute)
SeverLinks writes physics buffers. Declarations to add to Cutting.compute:
- `RWStructuredBuffer<int> _NbrIdx;` (mirror Physics.compute `_NbrIdx`, int[6*cornerCount]).
- `RWStructuredBuffer<BendPairGpu> _BendPairs;` (BendPairGpu = `{int i,j,k; float theta0; int alive; int _p0,_p1,_p2}`
  = 32 bytes, defined in PhysicsCommon.hlsl).
- **Struct source**: prefer `#include "PhysicsCommon.hlsl"` in Cutting.compute IF it does not double-define
  anything already pulled in by ReconCommon/CuttingCommon (check NORMAL_SCALE/FixedPoint guards). If it conflicts,
  declare a guarded local `BendPairGpu` struct in Cutting.compute matching the C# layout byte-for-byte. The audit
  must confirm the chosen path compiles with no redefinition.
`_GridEdgeCount`, `_Dims`, `_VoxelCutMask`, `_GridEdges` are already declared/bound in Cutting.compute.

## 4. Dispatch wiring (CutDetector.cs)
`SeverLinks` runs AFTER `DetectCut` (cut bits must be set first), every frame, over gridEdgeCount:
- Add `int _kSeverLinks = cs.FindKernel("SeverLinks");` in the CutDetector ctor.
- In `CutDetector.Dispatch`, after the DetectCut dispatch: bind `_GridEdges, _VoxelCutMask, _NbrIdx, _BendPairs`
  + set `_Dims/_GridEdgeCount`; dispatch `ceil(GridEdgeCount/64)` groups.
- Order in ReconGridManager.Update is `solver.Step → cutDetector.Dispatch(DetectCut+SeverLinks) → dc.Build`; the
  sever writes take effect on the NEXT frame's Step (1-frame latency, negligible). Idempotent each frame.
- Bind `rb.NbrIdx` / `rb.BendPairs` (already public ComputeBuffers in ReconBuffers) to SeverLinks.

## 5. Demo tuning so the halves FULLY separate (acceptance-relevant)
The blade plane is x=center (y-z plane); the sweep severs x-edges at x=center for each z it passes. To disconnect
the +x half from the −x half COMPLETELY, the sweep must traverse the sphere's FULL z-extent (sphere z∈[1,5]).
Current `bladeSweepRange=1.5` covers z∈[1.5,4.5], leaving z∈[1,1.5] & [4.5,5] uncut → the halves stay hinged at
the z-ends. Recommend `bladeSweepRange ≈ 2.2` (z∈[0.8,5.2], beyond the sphere) so the cut fully traverses → the
−x/+x halves are topologically disconnected → the unpinned half falls under gravity. (The top y-layer is pinned,
so the falling piece is the unpinned bulk; matches paper Fig 3.9.) This is a demo param, not an algorithm change.

## 6. Verification
- **GPU oracle `Sever_GpuOracle_Tests.cs`** (mirror BendingForce/PhysicsTopology oracle pattern), small grid:
  1. Cut a single interior edge (set its VoxelCutMask bits), dispatch SeverLinks; read back `_NbrIdx` — both
     endpoints' relevant slots == −1; read back `_BendPairs` — every pair through the edge has alive==0.
  2. Then run AccumulateStructural + AccumulateBending and assert ZERO force is transmitted across the severed
     edge (compare the severed-edge force contribution to 0; an unsevered control edge still transmits force).
  3. Idempotency: dispatch SeverLinks twice → identical NbrIdx/BendPairs (no double-harm).
- **Unity Play (user)**: blade sweeps fully through the sphere → the unpinned half visibly separates and falls
  under gravity (the upper pinned half stays); FPS stable; no residual force pulling the halves back together.

## 7. Risks / watch-items (for the audit)
- **R1 both-direction severing**: must write BOTH `_NbrIdx[6*A+sPlus]` and `_NbrIdx[6*B+sMinus]` — a one-sided
  sever leaves a half-spring still pulling. Verify slot math (sPlus=2axis+1, sMinus=2axis) matches NbrDelta.
- **R2 cross-shader BendPairGpu**: ensure the struct declaration in Cutting.compute is byte-identical to
  PhysicsCommon.hlsl / C# BendPairGpu (i,j,k,theta0,alive,3 pad = 32B) and the include/redeclare path compiles
  with no redefinition (same hazard class as the 2C cross-includes).
- **R3 RW concurrency**: `_NbrIdx`/`_BendPairs` are RW in BOTH SeverLinks (Cutting.compute) and the physics
  kernels (Physics.compute), but never dispatched simultaneously (separate Dispatch calls = barrier). Confirm no
  same-frame race: SeverLinks runs in cutDetector.Dispatch, physics in solver.Step (different dispatches).
- **R4 no regression**: when there are no cuts, SeverLinks early-outs on `!EdgeIsCut` for every edge → writes
  nothing → Stage-1 physics byte-identical. Confirm.
- **R5 RestNbr not touched**: severing only sets NbrIdx=-1; RestNbr is read only when n≥0 (Physics.compute:96),
  so a stale RestNbr in a severed slot is never read. Confirm (no need to clear RestNbr).
- **R6 pinned interaction**: the pinned top layer must remain pinned; severing must not un-pin. SeverLinks only
  touches NbrIdx/BendPairs, not _Pinned — confirm the falling half is the unpinned region.
- **R7 paper fidelity**: severing structural + bending (not just structural) is required — a live bending pair
  through the cut applies ghost torque across it (the paper's "physical connection" includes bending). Confirm
  the bending scan covers exactly the pairs whose arm is the cut edge, no more (don't kill unrelated pairs).

## 8. REV 2 — audit resolutions (workflow wabip73b7: R1 paper+physics APPROVED, R2 GPU feasibility, R3 roadmap). Implement THIS.
Design is paper-faithful (§2.1.2, paper:367; paper:524-528 = exactly two force couplings: structural + bending,
so NbrIdx=-1 + bending alive=0 is necessary AND sufficient) and physics-correct (both-direction sever, bending
scan complete + no over-kill, slot math verified). Fold in before coding:

- **[CRITICAL — buffer-type wording]** `_NbrIdx`/`_BendPairs` are **read-only `StructuredBuffer`** in
  Physics.compute (lines 30-32); §0/§1 above saying "RW in both" is WRONG. **SeverLinks is the SOLE writer**
  (binds them RW in Cutting.compute via `RWStructuredBuffer<int> _NbrIdx; RWStructuredBuffer<BendPairGpu> _BendPairs;`).
  **Do NOT change Physics.compute's declarations to RW** — the single-writer model is safer and keeps the SRV
  read path. Unity `ComputeBuffer` is UAV-capable by default, so binding rb.NbrIdx/rb.BendPairs as RW in one
  kernel and SRV in another needs NO allocation/flag change (ReconBuffers.cs:343/:344). The cross-dispatch barrier
  (SeverLinks in cutDetector.Dispatch, physics reads next frame's solver.Step) makes the single writer race-free.
- **[IMPORTANT — cross-shader BendPairGpu] use a guarded LOCAL struct** in Cutting.compute (NOT `#include
  "PhysicsCommon.hlsl"`, which would drag SpringGpu/AtomicAddForce/DOPRI tables into the cut shader). Byte-identical
  to C# `ReconBuffers.BendPairGpu` / PhysicsCommon.hlsl:20-30 (32B):
  `#ifndef BENDPAIRGPU_DEFINED` / `#define BENDPAIRGPU_DEFINED` / `struct BendPairGpu { int i,j,k; float theta0;
  int alive; int _p0,_p1,_p2; };` / `#endif`. (Audit verified both paths compile collision-free; local is cleaner.)
- **[IMPORTANT — CutDetector wiring]** SeverLinks gets its OWN kernel handle + bind block; it CANNOT reuse
  DetectCut's bindings. Ctor: `_kSeverLinks = cs.FindKernel("SeverLinks")`. In Dispatch, AFTER the DetectCut
  dispatch: bind `_GridEdges, _VoxelCutMask, _NbrIdx(=rb.NbrIdx), _BendPairs(=rb.BendPairs)` to `_kSeverLinks`
  (`_Dims`/`_GridEdgeCount` already set), dispatch `ceil(GridEdgeCount/64)`. **SeverLinks does NOT need
  `_VoxelOccupied`** — `EdgeIsCut` reads only `_VoxelCutMask` (DetectCut already gated occupancy when setting bits).
- **[IMPORTANT — demo sweep]** `bladeSweepRange` 1.5 → **2.2** (sphere z-extent [1,5], center 3, r 2 → range 1.5
  leaves z∈[1,1.5]&[4.5,5] hinged; 2.2 → z∈[0.8,5.2] ⊃ [1,5]). Verify per-frame z-step `bladeSpeed*physicsDt`
  = 0.024 < L = 0.25 (no x-edge z-layer skipped). Acceptance test should assert the cut x-edge count across the
  plane == the # of in-tissue x-edges there (else a residual hinge keeps the half attached).
- **[MINOR — test reference]** §6 mirror **`CutBitPropagation_GpuOracle_Tests`** (build grid+ReconBuffers,
  FindKernel, set `_Dims`/`_GridEdgeCount`, bind VoxelCutMask/NbrIdx/BendPairs, dispatch, GetData) for the
  SeverLinks dispatch + **`BendingForce_GpuOracle_Tests.DeadPair_ZeroForce`** + a structural oracle for the
  zero-transmitted-force proof. (Plan's "PhysicsTopology_GpuOracle_Tests" is a misnomer — that file is a CPU test.)
- **[MINOR — RestNbr comment]** at the sever write, comment that `_RestNbr` is intentionally left stale (read-gated
  by the `n<0` guard at Physics.compute:94 before :96 reads it) so a future maintainer doesn't "fix" it.
