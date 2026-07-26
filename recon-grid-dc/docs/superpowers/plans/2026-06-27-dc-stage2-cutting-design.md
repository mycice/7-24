# Stage 2 — Cutting — Implementation Design (paper §2.1 / design §5)

> **This is the detailed Stage-2 design to be multi-subagent audited BEFORE any code.** It operationalizes design-spec §5 into sub-stages, GPU buffers, kernels, interfaces, and verification, with paper equation/figure citations. Per-sub-stage task plans (complete code) are written via writing-plans as each sub-stage is reached.
> Authoritative refs: design spec `docs/superpowers/specs/2026-06-27-dc-cutting-design.md` §5 (+§2 ledger, §3 buffers); paper `newpaper_clean.txt` §2.1 (Eq6-8, Fig 2.3–2.9); extraction `findings.md`.

## 0. Goal, scope, coding discipline

**Goal:** add the paper's progressive cutting on top of Stage 1 (deformable hexa grid + DC surface), high-fidelity to §2.1: a blade sweep severs the grid's edge-graph, generates a non-manifold cut surface directly at the blade (width = tool thickness D), with no new particles/voxels added, plus ablation. Strict alignment with paper algorithms; only documented deviations D1–D6 + new PAPER-SILENT items (listed in §7).

**Coding discipline (user mandate — every implementer + reviewer enforces):**
1. **Strict architectural fidelity** — files/classes/kernels exactly as this design defines (§1.2); no arbitrary split/merge; I/O interfaces match.
2. **Theoretical traceability** — every core block carries a comment `// paper Eq6 / Fig 2.4 / §2.1.1` (or `// design §5.x`, `// DEVIATION Dn`, `// PAPER-SILENT`) naming the exact source.
3. **Variable↔symbol mapping** — core variables bridge code + paper symbols (e.g. `t_ray`, `u_bary`, `v_bary` for Eq6-8; `compCount`, `vertToComp`; matrices get suffix + a declaration comment with physical meaning + dims, e.g. `float3x3 R_frame; // per-particle rotation (Berndt[22]), maps rest→current neighbor frame`).

**Out of scope:** haptics (Stage 3). Builds strictly on Stage 1 (do NOT regress recon/physics/coupling).

## 1. Architecture

### 1.1 Where cutting plugs into the per-frame loop (design §4.6/§5.10)
The Stage-1 per-frame loop is `solver.Step → DualContouring.Build (RebuildIsectWorld → DC_FeaturePoints → DC_Stitch → DC_Normals → CopyCount) → draw`. Cutting inserts a **Cutting stage between physics and DC**, and the DC stage gains a cut-surface branch:
```
solver.Step                                   # physics moves corners (Stage 1)
[Cutting stage] (only when blade active):
   DetectCut (M-T: voxel edges vs swept plane) → mark voxelCutMask (→4 voxels), emit cut points   # 2B
   SeverSprings (structural+bending alive=0; feedback to physics next frame)                       # 2D
   [if ablation] AblatePass1 (delete outside) → AblatePass2 (ablate inside, mark ablated edges)     # 2D
   LookupConnectivity (CONN4096 → compCount/vertToComp per cut voxel)                              # 2A used here
DualContouring.Build (UNIFIED geometry pass, single triCounter reset + CopyCount):
   RebuildIsectWorld → UpdateParticleFrames (R) → DC_FeaturePoints (outer, unchanged)              # 2D frames
   → ComputeCutFeaturePoints (vertex-keyed centroid)                                               # 2C
   → DC_Stitch (outer) + BuildCutTriangles (cut surface, per-particle, into SAME triBuf)           # 2C
   → DC_Normals[barrier] → CopyCount
draw
```
Key invariant (design §5, "topology fixed under deformation" extended): **cutting CHANGES topology (cut bits), but only when the blade acts**; between cuts the cut state persists and the surface is rebuilt from the (deforming) corners + persistent cut bits.

### 1.2 Files (strict — new + extended)
```
Assets/ReconGridDC/
  Core/GridConventions.cs            // EXTEND: (axis,slot)→localEdge map for cut-bit propagation; localCornerOf(particle,voxel)
  Recon/ConnectivityLUT.cs           // NEW (2A): CPU build of CONN4096 (compCount + vertToComp[8], dense ids)
  Recon/ReconBuffers.cs              // EXTEND: voxelCutMask, conn4096, cutPoint(+counter), featurePoint(internal)+valid, cutFPAccum, voxelFPCount, particleState
  Cutting/CuttingTool.cs             // NEW (2B): blade line S/E + thickness D; swept-plane (2 tris) from prev+cur frame
  Cutting/CutDetector.cs             // NEW (2B): C# orchestrator for DetectCut dispatch
  Cutting/CutTopology.cs             // NEW (2C): C# orchestrator for LookupConnectivity→ComputeCutFeaturePoints→BuildCutTriangles
  Cutting/AblationCutter.cs          // NEW (2D): two-pass ablation orchestrator
  Physics/ParticleFrames.cs          // NEW (2D): per-particle R (polar decomposition, Berndt[22])
  Physics/MassSpringSolver.cs        // EXTEND (2D): consume severed springs/bending (alive=0) — already gated by `alive`
  Shaders/Cutting.compute            // NEW: DetectCut, MarkCutBits, EmitCutPoints, LookupConnectivity, ComputeCutFeaturePoints, BuildCutTriangles, SeverSprings, UpdateParticleFrames, AblatePass1, AblatePass2
  Shaders/CuttingCommon.hlsl         // NEW: shared structs (CutPointGpu, Conn4096 access), Moller-Trumbore (Eq6-8), localCornerOf
  Recon/DualContouring.cs            // EXTEND: dispatch cut-FP + cut-stitch inside the unified geometry pass
  Shaders/Recon.compute / ReconSurface.shader  // EXTEND: triBuf vertex tag (external=voxelId / internal=0x80000000|(8*voxelId+slot)); shader fetches from voxelExternalFP vs featurePointBuf by tag
  Demo/ReconGridManager.cs           // EXTEND: drive a demo blade sweep; toggle cutting
  Tests/  (EditMode)
    ConnectivityLUT_Tests.cs (2A, CPU), MollerTrumbore_GpuOracle_Tests.cs (2B), CutBitPropagation_Tests (2B),
    CutSurface_GpuOracle_Tests.cs (2C), Sever_GpuOracle_Tests.cs (2D), Ablation_GpuOracle_Tests.cs (2D)
```

### 1.3 New/extended GPU buffers (design §3 — already specified; materialize now)
- `voxelCutMaskBuf : uint` — 12-bit cut state per voxel (bitwise-OR). [§3.4]
- `conn4096Buf : (int compCount, int[8] vertToComp)` per config, length 4096. [§3.8, §5.4]
- `cutPointBuf : (float3 localOffset, int ownerParticle, int edgeId)` + `cutPointCounter` (append; two per cut edge). world = `owner.pos + owner.R·localOffset`. [§3.6, §5.3]
- `featurePointBuf : float3 + uint valid` (internal cut FP; 8 slots per **global voxelId**, block `8*voxelId`) + `featurePointNormalBuf : int3` + `cutFPAccumBuf : (int3 sum, int count)` + `voxelFPCountBuf : uint`. [§3.4]
- `particleStateBuf : uint` (0=alive,1=ablated,2=deleted). [§3.1]
- `frameBuf : float3x3` per particle (R, Berndt). [§3.1]
- triBuf vertex tag (§3.4): external = `voxelId` (high bit clear); internal = `0x80000000 | (8*voxelId+slot)`.

## 2. Sub-stage 2A — CONN4096 connectivity LUT (design §5.4; paper §2.1.1)

**Goal:** the foundational combinatorial table. Pure CPU, fully testable without GPU.

**Algorithm** (`ConnectivityLUT.cs`): for each `config ∈ [0,4096)` (12-bit cut mask), run union-find over the **uncut-edge subgraph** of the 8-vertex/12-edge cube (an edge is a connectivity link iff its bit is 0; a cut/1 bit removes the link). Output:
- `compCount[config]` = number of connected components (≤8). [paper §2.1.1 "up to 8 feature points"]
- `vertToComp[config][8]` = each vertex's component id, **densely renumbered 0..compCount-1** (canonical union-find compression). [design §5.4 vertToComp + dense-id invariant; gap C4]
- Edge incidence from `GridConventions.EdgeCorners` (the frozen Fig 2.4 numbering). Comment each: `// paper §2.1.1: cut bit DISCONNECTS the edge in the connectivity subgraph`.

**Why vertex-keyed (not edge-keyed)** — comment in code: a cut edge straddles two components, so no single per-edge id names "the" component; the centroid + stitching are per-vertex. [findings.md; gap C4/C54]

**Verification (CPU EditMode — runs anywhere):** `ConnectivityLUT_Tests.cs`
- Fig 2.4 example: `config` with e0,e2,e4,e6 cut → `compCount==2`, components `{V0,V3,V4,V7}` and `{V1,V2,V5,V6}` (assert `vertToComp` puts those vertex sets in 2 distinct dense ids). [paper Fig 2.4]
- All 12 edges cut → `compCount==8` (8 singletons).
- 0 edges cut → `compCount==1` (all one component).
- **Dense-id invariant**: for every config, `max(vertToComp[config]) == compCount[config]-1`.
- High popcount (cross cuts): a config with ≥6 cut edges yields >2 components and a valid vertToComp.

## 3. Sub-stage 2B — Cut detection + cut points (design §5.1–5.3; paper §2.1.3, Eq6-8, Fig 2.8)

**3.1 Cutting tool + swept plane** (`CuttingTool.cs`): blade = line segment endpoints `S` (top) / `E` (bottom), thickness `D`. **Swept cutting plane** = quad of (prev-frame S_k,E_k) + (cur-frame S_{k+1},E_{k+1}), split by diagonal `S_k→E_{k+1}`: `T1=(S_k,E_k,E_{k+1})`, `T2=(S_k,E_{k+1},S_{k+1})`. [paper §2.1.3 Fig 2.8; gap C22] Persists per frame.

**3.2 Möller-Trumbore cut detection** (`Cutting.compute` `DetectCut`; `CuttingCommon.hlsl`): **ray = voxel grid edge** (`O`=one endpoint world pos, `Dvec`=edge vector), **triangle = a swept-plane triangle**. [gap C21 — ray is the voxel edge, NOT the tool] Comment block: `// paper Eq6-8 (Moller-Trumbore), Fig 2.8: ray=voxel edge, tri=swept blade plane`.
- `L1 = V1-V0; L2 = V2-V0; T_vec = V0 - O` (Eq7 sign: `T=V0-O`). [gap C23]
- shared `det = dot(cross(L2,Dvec), L1)`; near-zero → no hit (parallel).
- `t_ray = dot(cross(L2,T_vec), L1)/det`; `u_bary = -dot(cross(L2,Dvec), T_vec)/det`; `v_bary = dot(cross(Dvec,T_vec), L1)/det`. (u carries leading minus, paper Eq8.) [gap C23]
- Hit iff `u_bary≥0 && v_bary≥0 && u_bary+v_bary≤1 && 0≤t_ray≤1` (finite edge bound). [paper Eq6 conditions + gap C24]
- Variables named exactly `t_ray,u_bary,v_bary` per discipline rule 3.

**3.3 Cut-bit propagation to 4 voxels** (`MarkCutBits`): a hit grid edge belongs to ≤4 voxels; via `GridConventions` `(axis,slot)→(voxelOffset,localEdge)` set `voxelCutMaskBuf[voxel] |= 1<<localEdge` for each. [gap C1/S2-I7]

**3.4 Cut points** (`EmitCutPoints`): from the single hit point `P_hit = O + t_ray·Dvec`, offset `±D/2` along the in-plane thickness direction → **two cut points**, each bound to one endpoint particle, stored as `(localOffset, ownerParticle, edgeId)` (owner-frame local; world recomputed in 2D). [paper §3.4 Fig 3.8; gap C16/C19/C25; PAPER-SILENT offset direction]

**Verification (GPU oracle):** M-T hit/miss on a known voxel-edge vs known triangle (analytic t/u/v); `0≤t≤1` boundary cases; `CutBitPropagation_Tests`: cutting one interior edge sets the matching bit in all 4 sharing voxels and the 4 CONN4096 configs agree on the split.

## 4. Sub-stage 2C — Cut topology + surface (design §5.5–5.6; paper §2.1.1, Fig 2.4/2.5)

**4.1 LookupConnectivity** (`Cutting.compute`): each cut voxel reads `voxelCutMaskBuf` → `conn4096Buf[mask]` → `compCount`, `vertToComp`. Set `voxelFPCountBuf[voxelId]=compCount`. [paper §2.1.1; design §5.5]

**4.2 ComputeCutFeaturePoints** (vertex-keyed centroid): per cut edge, each of its two cut points (bound to endpoint Va / Vb) is accumulated into `cutFPAccumBuf[8*voxelId + vertToComp[config][Va]]` resp. `[...Vb]` (the component containing that endpoint). Then `featurePointBuf[8*voxelId+slot] = sum/count` with `valid=1` if `count>0` else `valid=0` (no NaN). [paper §2.1.1 "centroid of cut points"; design §5.5; gap C14 + S1-final valid] Comment: `// paper §2.1.1: feature point = centroid of the component's cut points (NOT QEF)`.

**4.3 BuildCutTriangles — per-particle stitching** (Fig 2.5): traverse each particle's 6 axis edges + 8 octant voxels. For each CUT edge, in each of the ≤4 shared voxels select the internal FP of **the component containing THIS particle's local vertex**: `comp = vertToComp[config_of_voxel][localCornerOf(thisParticle, voxel)]`, FP at `featurePointBuf[8*voxelId+comp]`. **NEVER edge-keyed** (re-welds the seam = MC2024 bug). Partial-quad: 4 valid→2 tris, 3→1 tri, <3→none. Append into the SAME `triBuf` with the internal tag `0x80000000|(8*voxelId+slot)`. [paper §2.1.1 "V1,V2 share two FPs, connected respectively to form two triangles", Fig 2.5; design §5.6; gap C15/C54/S2-I8]

**4.4 Render** (`ReconSurface.shader`): decode triBuf vertex tag — external (high bit clear) fetch `voxelExternalFP[idx]`; internal (high bit set) fetch `featurePointBuf[idx&0x7FFFFFFF]`. [design §3.4]

**Verification:** a single planar cut across a cube/sphere → the cut voxels split (compCount→2) and **two non-manifold faces** are generated at the blade; with D>0 the gap width ≈ D; cross/curved cuts (Fig 3.10) handled (high-popcount LUT). EditMode GPU oracle on a small grid: assert a V0-side particle selects the {V0,...} component FP in all shared voxels and a V1-side particle selects the {V1,...} FP (seam NOT re-welded).

## 5. Sub-stage 2D — Dynamic frames + severing + ablation (design §5.7–5.9; paper §2.1.2/2.1.4)

**5.1 Per-particle frames** (`ParticleFrames.cs` + `UpdateParticleFrames` kernel): recover `R_frame` (float3x3, rest→current rotation) by **polar decomposition of the deformation gradient from the 6 rest-neighbor offsets to the current offsets** (Berndt[22] style; PAPER-SILENT — paper gives no formula, cites [22]). Comment: `// paper §2.1.2 cites Berndt[22]; PAPER-SILENT: R via polar decomposition of rest→current neighbor frame`. Cut point world = `owner.pos + R_frame·localOffset`. [gap C17/C18/C51]

**5.2 SeverSprings** (`Cutting.compute` `SeverSprings`): when an edge is cut, set its structural `springEdgeBuf.alive=0` AND **invalidate every bending pair using that edge as an arm** (`bendPairBuf.alive=0`), and set the two endpoints' `nbrIdx` to −1. The Stage-1 force kernels already skip `alive==0` (no change needed there) → the two halves separate physically. [paper §2.1.2 "physical connection severed"; design §5.8; gap C20/C35] Comment: `// paper §2.1.2: sever structural spring AND bending pairs through the cut edge (else ghost forces)`.

**5.3 Ablation** (`AblationCutter.cs` + `AblatePass1/AblatePass2`): ordered two passes (separate dispatches). Pass1: tool-range particles OUTSIDE isosurface → `particleState=deleted` (no cut points). Pass2: INSIDE → `particleState=ablated`→`deleted`, mark ablated edges (incident to ablated particles incl. toward surviving neighbors), generate cut points (bound to the SURVIVING endpoint; deleted endpoint contributes none), funnel into the SAME CONN4096+centroid+redraw pipeline. All per-particle/corner kernels early-out on `state!=alive`. [paper §2.1.4 Fig 2.9; design §5.9; gap C26/C27/C28 + S1/S2-minor orphan-ownership]

**Verification:** sever → in Unity the two halves separate under gravity (physics two-component); ablation removes a volume and reseals. GPU oracles: after a cut, the severed edge's spring/bending `alive==0`; ablation pass ordering; orphan cut-point bound to surviving particle.

## 6. Unified geometry pass + dispatch order (design §3.7/§4.6/§5.10)
ONE `triCounter` reset (head) + ONE `CopyCount` (tail) per frame; outer-surface triangles (Stage 1) AND cut-surface triangles (2C) append into the SAME `triBuf` between them; single `DC_Normals[barrier]` normalizes BOTH `voxelExternalFPNormalBuf` and `featurePointNormalBuf`. [design §4.6 / §5.10] `ClearCounters` also zeros `cutFPAccumBuf` + `voxelFPCountBuf` + `cutPointCounter` each frame.

## 7. Deviation ledger additions for Stage 2 (declare; mark in code)
No new numeric corrections (D1–D6 unchanged). New **PAPER-SILENT** items (paper gives no formula):
- Berndt[22] per-particle frame = **polar decomposition** of rest→current neighbor offsets (§5.1).
- Cut-point **±D/2 offset direction** = in-plane blade-thickness axis (§3.4).
- vertToComp **dense renumber** 0..compCount-1 (§2A).
- **partial-quad** <4-corner stitch rule (4→2,3→1,<3→none) (§4.3).
- Ablation **orphan cut-point → surviving endpoint** (§5.3).
- Cut detection uses M-T `t∈[0,1]` on the finite voxel edge (paper's "ε" distance criterion is replaced by the geometric swept-plane intersection, which is the DC-appropriate equivalent) — note this is a structural consequence of the DC method, not a deviation in the cutting logic.

## 8. Verification strategy (env can't run Unity)
- CPU units (CONN4096) → directly-runnable EditMode tests (run anywhere).
- GPU kernels → EditMode GPU oracles with hand-computed expected values (user runs on GPU), PLUS the headline Unity Play check: **a blade sweep visibly severs the deforming sphere into two halves along the cut, gap ≈ D, no re-welded seam, FPS does not collapse as cuts accumulate** (paper's headline + design §5.11).
- Every sub-stage: multi-subagent code review vs this design + §5 + paper §2.1; skeptic-verify; fix to clean; then user verifies in Unity (this is where real bugs surfaced in Stage 1).

## 9. Sub-stage order + handoff
2A (CONN4096, CPU, foundational) → 2B (detection + cut points) → 2C (cut surface, the visible "cut opens" milestone) → 2D (dynamic frames + severing + ablation, physical separation). Each: writing-plans detailed task plan → multi-subagent audit → subagent-driven implementation → multi-subagent code review → user Unity verify.

## 10. Design-audit refinements (5-dim audit 2026-06-27: 0 critical / 0 important / 10 minor; fold into the per-sub-stage plans)
Design is paper-aligned; these are clarifications, each tagged with the sub-stage that must implement them.
- **[2A] CONN4096 test**: the headline LUT test asserts the WHOLE 4096-entry table equals an independent reference union-find (not the weaker "popcount≥6 ⇒ comps>2"). Keep Fig 2.4 / all-cut-8 / dense-id as targeted checks.
- **[2A/2C] `localCornerOf`**: add a CPU oracle that `localCornerOf(p,voxel)` round-trips `cornerPos == voxelOrigin + CornerOffset[localCornerOf]` so the vertex key feeding `vertToComp` is provably the SAME Fig 2.4 numbering the LUT was built on. Comment cites `GridConventions.CornerOffset`.
- **[2B] reuse `EdgeStencil`**: `MarkCutBits` consumes the EXISTING `GridConventions.EdgeStencil` 4-entry-per-axis `(voxelOffset,localEdge)` — do NOT introduce a second table. The GridConventions EXTEND is `localCornerOf` ONLY.
- **[2B] ±D/2 formula**: 2B plan must give the EXACT unit in-plane thickness-axis (and which sign binds to which endpoint particle) so `EmitCutPoints` is deterministic (PAPER-SILENT).
- **[2B] degenerate sweep early-out**: skip `DetectCut` when the swept triangle is degenerate (`‖cross(L1,L2)‖<eps`, i.e. zero/near-zero blade motion) — PAPER-SILENT robustness.
- **[2C] cut-quad winding**: the 2-tri diagonal + winding (and 3→1 winding) is inferred from Fig 2.5 (PAPER-SILENT, gap I26), matching the §4.3 outer-surface convention; Unity verify both cut faces render front-facing from their sides (no inverted/black cut face).
- **[2D] particleState retrofit (load-bearing)**: adding `particleStateBuf` requires retrofitting the EXISTING Stage-1 per-particle kernels (`AccumulateStructural`, `AccumulateBending`, `IntegrateY5`, `DC_FeaturePoints`, `RebuildIsectWorld`) with `state!=alive` early-outs — they currently have NO such guard, so without it deleted/ablated particles still exert force & generate surface.
- **[2D] boundary/severed frame**: `UpdateParticleFrames` skips `nbrIdx==-1` slots; if the gathered neighbor set is rank-deficient (<3 independent dirs) fall back to identity or previous R (no flip); decide whether a just-severed neighbor is excluded the same frame. Oracle: 3-neighbor corner → finite orthonormal R; just-cut particle → stable R.
- **[2D] rest-frame localOffset capture**: at cut instant, store `localOffset = R_owner_at_cut⁻¹ · (P_hit_side − owner.pos)` (owner's REST/current frame inverse), so per-frame `world = owner.pos + R_owner·localOffset` rotates rigidly with the owner (Fig 2.7) with no drift. Oracle: rotate owner frame by known R → cut point rotates rigidly.
- **[doc] path note**: file paths are `Assets/ReconGridDC/…` (as-built, correct in this plan). The design SPEC §1.2 still shows the stale `Assets/SurgicalSim/ReconGridDC/` and `Core/GpuBuffers.cs`; THIS plan is authoritative for Stage-2 file targets (buffers stay in `Recon/ReconBuffers.cs`; Cutting split into CuttingTool/CutDetector/CutTopology/AblationCutter).
