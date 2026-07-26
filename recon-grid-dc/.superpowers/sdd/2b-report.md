# Stage 2B Report

**Status:** COMPLETE — Tasks 1-4 implemented; Task 5 (review gate) is user's.

## Commits (4, branch master)
- `5936d48` feat(reconDC/2B): gridEdges + cut buffers (VoxelCutMask, GridEdges, CutPoint)
- `673ce10` feat(reconDC/2B): CuttingCommon.hlsl Moller-Trumbore (Eq6-8) + cut structs
- `d24ce5b` feat(reconDC/2B): CuttingTool swept plane (2 tris, n_cut, degenerate guard)
- `547ae94` feat(reconDC/2B): DetectCut (M-T ray=voxel edge, cut-bit→4 voxels, ±D/2 cut points) + oracles

## Files created/extended
- `Preprocess/BackgroundGrid.cs` — added `GridEdgeRec{axis,baseCorner}` + `gridEdges[]` (all interior edges, no sign-change filter, same EdgeStencil all-4-in-range check)
- `Recon/ReconBuffers.cs` — added `GridEdgeGpu2` (16B), `CutPointGpu` (32B) structs; `VoxelCutMask`/`GridEdges`/`CutPoint`/`CutPointCounter` buffers; Upload + Dispose
- `Shaders/CuttingCommon.hlsl` — NEW: `Conn4096Gpu`, `CutPointGpu`, `GridEdgeGpu2`, `MollerTrumbore`
- `Shaders/Cutting.compute` — NEW: `DetectCut`, `MT_Test` kernels
- `Cutting/CuttingTool.cs` — NEW: swept-plane blade (T1/T2, n_cut, degenerate guard)
- `Cutting/CutDetector.cs` — NEW: C# orchestrator (bind uniforms, buffers, dispatch DetectCut)
- `Tests/MollerTrumbore_GpuOracle_Tests.cs` — NEW: 3 cases (hit t≈0.5, parallel miss, t>1 miss)
- `Tests/CutBitPropagation_GpuOracle_Tests.cs` — NEW: 1 edge → 4 voxel bits + ≥2 cut points, gap=D

## Confirmed correctness checklist
(a) **M-T: T_vec=V0-O + u leading minus + t∈[0,1]**: yes — `T_vec = V0 - O` (Eq7 sign); `u_bary = -dot(cross(L2,Dvec),T_vec)*inv` (leading minus per Eq8/gap C23); return condition includes `t_ray >= 0 && t_ray <= 1` (finite-edge bound, gap C24).

(b) **Ray = voxel edge / tri = swept plane**: yes — `O=cornerPos[idA]`, `Dvec=cornerPos[idB]-O`; triangles T1=(Sprev,Eprev,E), T2=(Sprev,E,S) are the blade swept-quad triangles (paper Fig 2.8; gap C21).

(c) **Cut-bit → 4 voxels via EdgeStencil**: yes — `ES_VoxelOffset(axis,s)` / `ES_LocalEdge(axis,s)` in Cutting.compute mirror C# `GridConventions.EdgeStencil[axis][s]` exactly; `InterlockedOr(_VoxelCutMask[vid], 1u<<localE)` for all 4 slots. No second table introduced (design §10).

(d) **±D/2 along n_cut sign-by-side**: yes — `side = sign(dot(pPos - P_hit, _NCut))` determines which side; `P_side = P_hit + side*(D*0.5)*_NCut`; `localOffset = P_side - pPos`. Each of the 2 endpoint particles gets its own signed P_side.
