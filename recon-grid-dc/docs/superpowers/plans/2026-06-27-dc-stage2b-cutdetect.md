# Stage 2B — Cut detection (Möller-Trumbore) + cut points — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development. Steps use `- [ ]`.
> Builds on Stage 1 + 2A (CONN4096). Authoritative: Stage-2 design `docs/superpowers/plans/2026-06-27-dc-stage2-cutting-design.md` §3 (2B) + §10; paper §2.1.3 + **Eq6-8** + Fig 2.8; `findings.md` (§2.1.3).

**Goal:** detect where a swept blade plane crosses the grid's voxel edges (Möller-Trumbore, ray = voxel edge), mark those edges cut (propagated to all 4 sharing voxels' 12-bit masks), and emit two cut points per cut edge (offset ±D/2 across the cut, bound to the two endpoint particles). No surface yet (that is 2C); no severing/dynamic yet (2D).

## Global Constraints
- Project `D:/Desktop/Tissue_Simulation/SurgicalSim_DC`, module `Assets/ReconGridDC/`. Do NOT regress Stage 1/2A.
- **Coding discipline (mandatory):** (1) architectural fidelity — exactly these files/kernels; (2) traceability — every core block cites `// paper Eq6 / Eq7 / Eq8 / Fig 2.8 / §2.1.3` or `// design §5.x` / `// PAPER-SILENT`; (3) variable↔symbol — `t_ray, u_bary, v_bary, det, L1, L2, T_vec, n_cut, P_hit`.
- Möller-Trumbore is **ray = VOXEL grid edge** (`O`=edge endpoint world pos, `Dvec`=edge vector), **triangle = a swept-plane triangle**. NOT reversed. [gap C21]
- Eq7 sign: `T_vec = V0 − O`. Eq8 shared denom `det = dot(cross(L2,Dvec), L1)`; `u_bary` carries the leading minus. Hit iff `u_bary≥0 && v_bary≥0 && u_bary+v_bary≤1 && 0≤t_ray≤1`. [paper Eq6-8; gap C23/C24]
- Reuse `GridConventions.EdgeStencil` for cut-bit propagation (do NOT make a 2nd table). [§10]
- Env can't run Unity: GPU oracles with hand-computed values (user runs); CPU where possible.

## Files
```
Assets/ReconGridDC/
  Shaders/CuttingCommon.hlsl     // NEW: MollerTrumbore (Eq6-8), Conn4096Gpu mirror, CutPointGpu, gridEdge decode
  Shaders/Cutting.compute        // NEW: DetectCut (M-T + mark + emit cut points)
  Cutting/CuttingTool.cs         // NEW: blade S/E + D, swept plane (2 tris + n_cut), prev-frame state, degenerate guard
  Cutting/CutDetector.cs         // NEW: C# orchestrator (build/refresh gridEdges; dispatch DetectCut)
  Preprocess/BackgroundGrid.cs   // EXTEND: gridEdges[] = all interior grid edges (axis, baseCorner) for cut testing
  Recon/ReconBuffers.cs          // EXTEND: VoxelCutMask (uint/voxel), GridEdges buffer, CutPoint(+counter)
  Tests/
    MollerTrumbore_GpuOracle_Tests.cs   // known hit/miss + t∈[0,1] boundary
    CutBitPropagation_GpuOracle_Tests.cs// 1 edge cut → 4 voxels' bits + cut points
```

## Buffers (design §3)
- `VoxelCutMask : uint` (one per voxel; 12-bit, bitwise-OR). Cleared once at init (0).
- `GridEdges : (int axis, int3 baseCorner)` — all interior grid edges (all 4 incident voxels in-range), built at preprocess (like surfaceEdges but NO sign-change filter). `GridEdgeCount`.
- `CutPoint : (float3 localOffset, int ownerParticle, int edgeId)` + `CutPointCounter` (append). For 2B `localOffset = P_side − owner.pos` (world offset; 2D upgrades to R-frame). `edgeId` = the GridEdges index.

---

### Task 1: BackgroundGrid.gridEdges + ReconBuffers (VoxelCutMask, GridEdges, CutPoint)

**Files:** `BackgroundGrid.cs` (extend), `ReconBuffers.cs` (extend).

- [ ] **Step 1: BackgroundGrid emits all interior grid edges.** Mirror the existing `surfaceEdges` builder but WITHOUT the corner-sign-change filter — every grid edge (3 axes) whose 4 incident voxels are all in-range:
```csharp
// design §5.2: cut detection casts each VOXEL grid edge as the M-T ray.
public struct GridEdgeRec { public int axis; public int3 baseCorner; }
public GridEdgeRec[] gridEdges;
// in Build(), after surfaceEdges: same axis/stencil loop, drop the sign-change test, keep the all-4-voxels-in-range test.
```
- [ ] **Step 2: ReconBuffers** add `VoxelCutMask` (ComputeBuffer uint, voxelCount; init to 0 via SetData), `GridEdges` (struct `GridEdgeGpu2{int axis;int3 baseCorner;}` stride 16 → pad to 16 or 20; use 16: int+int3=16) + `GridEdgeCount`, `CutPoint` (struct `CutPointGpu{float3 localOffset;int ownerParticle;int edgeId;float _pad;}`=32B) + `CutPointCounter`. Upload gridEdges in Upload; dispose all. Traceability comments.
- [ ] **Step 3: Commit** `feat(reconDC/2B): gridEdges + cut buffers (VoxelCutMask, GridEdges, CutPoint)`.

### Task 2: CuttingCommon.hlsl — Möller-Trumbore (Eq6-8)

**Files:** `CuttingCommon.hlsl` (new).
- [ ] **Step 1:** structs + the M-T function (verbatim Eq forms):
```hlsl
#ifndef CUTTING_COMMON_INCLUDED
#define CUTTING_COMMON_INCLUDED
struct Conn4096Gpu { int compCount; int v0,v1,v2,v3,v4,v5,v6,v7; };   // mirror C# Conn4096Gpu (36B)
struct CutPointGpu { float3 localOffset; int ownerParticle; int edgeId; float _pad; }; // 32B

// paper Eq6-8 Moller-Trumbore. ray = O + t*Dvec (a VOXEL edge); triangle = (V0,V1,V2) (swept blade plane).
// Returns true on a valid finite-edge hit; outputs t_ray (∈[0,1]), u_bary, v_bary.
bool MollerTrumbore(float3 O, float3 Dvec, float3 V0, float3 V1, float3 V2,
                    out float t_ray, out float u_bary, out float v_bary)
{
    float3 L1 = V1 - V0;                 // Eq6: L1 = V1 - V0
    float3 L2 = V2 - V0;                 // Eq6: L2 = V2 - V0
    float3 T_vec = V0 - O;               // Eq7: T = V0 - O
    float det = dot(cross(L2, Dvec), L1);// Eq8 shared denominator (L2 x D)·L1
    t_ray = 0; u_bary = 0; v_bary = 0;
    if (abs(det) < 1e-12) return false;  // ray parallel to triangle
    float inv = 1.0 / det;
    t_ray  =  dot(cross(L2, T_vec), L1) * inv;   // Eq8: ((L2 x T)·L1)/det
    u_bary = -dot(cross(L2, Dvec), T_vec) * inv; // Eq8: -((L2 x D)·T)/det   (leading minus)
    v_bary =  dot(cross(Dvec, T_vec), L1) * inv; // Eq8: ((D x T)·L1)/det
    // paper Eq6 conditions + finite-edge bound (gap C24):
    return (u_bary >= 0 && v_bary >= 0 && u_bary + v_bary <= 1 && t_ray >= 0 && t_ray <= 1);
}
#endif
```
- [ ] **Step 2: Commit** `feat(reconDC/2B): CuttingCommon.hlsl Moller-Trumbore (Eq6-8) + cut structs`.

### Task 3: CuttingTool.cs — swept plane + n_cut

**Files:** `CuttingTool.cs` (new).
- [ ] **Step 1:** blade = line `S` (top), `E` (bottom), thickness `D`. Hold previous-frame `Sprev,Eprev`. **Swept quad** (S_k=Sprev,E_k=Eprev,S_{k+1}=S,E_{k+1}=E), diagonal `S_k→E_{k+1}`:
  - `T1 = (Sprev, Eprev, E)`, `T2 = (Sprev, E, S)`. [paper Fig 2.8; gap C22]
  - `n_cut = normalize(cross(Eprev - Sprev, E - Sprev))` (cut-surface normal; the gap opens along ±n_cut). PAPER-SILENT.
  - **Degenerate guard** [§10]: if `‖cross(T1 edges)‖ < eps` (zero/near-zero blade motion), mark `valid=false` → DetectCut skipped this frame.
  - Expose to GPU as 6 float3 (Sprev,Eprev,S,E + the 2 triangles derived in-shader) + `D` + `n_cut` + `valid`.
- [ ] **Step 2: Commit** `feat(reconDC/2B): CuttingTool swept plane (2 tris, n_cut, degenerate guard)`.

### Task 4: Cutting.compute DetectCut + CutDetector + GPU oracles

**Files:** `Cutting.compute` (new), `CutDetector.cs` (new), the 2 test files.
- [ ] **Step 1: Write the failing M-T oracle** `MollerTrumbore_GpuOracle_Tests.cs` (`[Category("GPU")]`, load Cutting.compute via FindAssets): dispatch a tiny `MT_Test` kernel (or DetectCut on 1 known edge) — a voxel edge `O=(0,0,0), Dvec=(0,0,1)` (z-edge) vs triangle `(( -1,-1,0.5),(1,-1,0.5),(0,1,0.5))` crosses at z=0.5 → assert hit, `t_ray≈0.5`, `u,v` in range; a parallel edge → miss; an edge whose infinite line hits but `t>1` → miss (finite-edge bound).
- [ ] **Step 2: DetectCut kernel** (per GridEdges entry):
```hlsl
#pragma kernel DetectCut
#include "CuttingCommon.hlsl"
// per grid edge e: O = corner(baseCorner), Dvec = corner(baseCorner+axisDir) - O  (deformed corners)
// test vs T1 and T2 (swept plane). If either hits (take the nearer t):
//   P_hit = O + t_ray*Dvec
//   mark cut bit in all 4 sharing voxels: for each EdgeStencil[axis] -> voxelCutMask[voxel] |= 1<<localEdge   // §10 reuse EdgeStencil
//   emit TWO cut points (EmitCutPoints inline): for each endpoint particle p (the 2 corners of the edge):
//      side = sign(dot(cornerPos[p] - P_hit, n_cut))           // which side of the cut plane (PAPER-SILENT ±D/2 dir = n_cut)
//      P_side = P_hit + side*(D*0.5)*n_cut                      // offset ±D/2 across the cut
//      append CutPoint{ localOffset = P_side - cornerPos[p], ownerParticle = p, edgeId = e }   // 2D upgrades localOffset to R-frame
```
Guard: skip if `!toolValid`; AABB broadphase cull (skip edges whose endpoints are both far from the swept-plane AABB). Use `InterlockedOr` on voxelCutMask (4 voxels) and `InterlockedAdd` on CutPointCounter.
- [ ] **Step 3: CutDetector.cs** orchestrator: set tool uniforms (Sprev,Eprev,S,E,D,n_cut,valid), bind buffers, dispatch `DetectCut` over `G(GridEdgeCount)`; roll Sprev=S,Eprev=E each frame.
- [ ] **Step 4: CutBitPropagation oracle** `CutBitPropagation_GpuOracle_Tests.cs`: a small grid + a swept plane crossing ONE interior grid edge → assert that edge's bit is set in all 4 sharing voxels, the 4 voxels' CONN4096 configs agree on the split, and exactly 2 cut points emitted with the right owners and ±D/2 offset (gap = D between them).
- [ ] **Step 5:** run, fix, **Commit** `feat(reconDC/2B): DetectCut (M-T ray=voxel edge, cut-bit→4 voxels, ±D/2 cut points) + oracles`.

### Task 5: 2B verification gate
- [ ] Run all tests (M-T oracle, cut-bit propagation; 2A/1 still green). Multi-subagent review (M-T Eq6-8 exactness; ray=voxel-edge; cut-bit→4-voxel via EdgeStencil; ±D/2 sign-by-side; degenerate guard; no regression). Fix to clean. Update trackers.

## Verification
- M-T: hand-computed hit/miss + t∈[0,1] boundary (oracle).
- Cut-bit propagation: 1 edge → 4 voxel bits + 4 agreeing configs + 2 cut points gap=D.
- No visual yet (2C). Headline check deferred to 2C (cut opens).

## Notes for 2C/2D
- 2C consumes VoxelCutMask + Conn4096 + CutPoint to build the cut surface.
- 2D upgrades CutPoint.localOffset to the owner's R-frame (rest-frame capture) + severs springs.
