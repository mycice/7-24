# Stage 2C — Cut topology + cut-surface generation (the visible "cut opens" milestone)

> Authoritative detailed plan for sub-stage 2C. Grounds design §4 (`2026-06-27-dc-stage2-cutting-design.md`)
> + paper §2.1.1 (Fig 2.4 / Fig 2.5) against the as-built Stage-1 code. Honors the 3 coding-discipline
> rules: (1) strict architectural fidelity to design §4, (2) every core block cites a paper eq/figure,
> (3) variable↔symbol mapping (`compCount`, `vertToComp`/`comp`, `featurePointBuf`/`_CutFP`, `cutFPAccum`, POS_SCALE).
>
> **REV 2 (post-audit):** folds in auditor A1 (algorithm) + A2 (GPU-realization). Resolved blockers:
> A2-F1 (int4 InterlockedAdd invalid → flat int accumulators), A1-C1 (edge predicate = BackgroundGrid's
> 4-voxel-in-bounds EdgeStencil test), A1-C2 (Cull Off + two-sided fragment, winding-sign-agnostic),
> plus A1 I1/I2/I3 + A2 F2/F3/F4a/F4c/F5/F6/F8.

## 0. Goal & headline check
A blade sweep (2B already marks `voxelCutMask` + emits cut points) now produces **two non-manifold cut
faces** at the blade, separated by gap ≈ D, with the **correct component selection** so the seam is NOT
re-welded (the MC2024 failure). Static recon (Stage 1) and cut faces share ONE `triBuf` + ONE indirect draw.
After 2C the cut is *visible* but the two halves do not yet physically separate (that is 2D severing).

## 1. What exists vs what 2C adds (from reconnaissance)
- `GridConventions`: has `CornerOffset[8]`, `EdgeCorners[12,2]`, `EdgeStencil[3][4]{voxelOffset,localEdge}`,
  `CornerId`, `VoxelId`, `VoxelCoord`. **Missing: `LocalCornerOf`** — the ONLY GridConventions EXTEND for 2C.
- `conn4096Buf` (`Conn4096`) EXISTS (2A): `Conn4096Gpu{int compCount; int v0..v7}`, stride 36, length 4096.
  `Conn4096Gpu.v{lc}` = `vertToComp[cfg*8+lc]` (slot index = local-corner id `lc`) — verified in `UploadConn4096`.
- `voxelCutMask` (uint/voxel) + `cutPoint{localOffset,ownerParticle,edgeId}`/`cutPointCounter` EXIST (2B, cumulative).
- `_GridEdges` (2B) = ALL interior grid edges built by `BackgroundGrid` with the predicate **"all 4 EdgeStencil
  voxels in-bounds"** (NOT a mere corner-range test). 2B set cut bits only for these edges. **2C must use the
  SAME predicate** so the stitched edge-set ⊆ the FP-populated edge-set (A1-C1).
- `Recon.compute`: `DC_Stitch` per-edge (STENCIL_OFF + `insideA` swap — DATA-DEPENDENT winding, NOT a fixed rule);
  `DC_NormalsScatter` per-tri (`& 0x7FFFFFFF` UNCONDITIONAL today → must become tag-aware); `DC_NormalsNormalize` per-voxel.
- `ReconSurface.shader`: `idx=_Tri[vid]&0x7FFFFFFF; wp=_VoxelExternalFP[idx].xyz; n=_ExtNormalF[idx];` — `Cull Back`, NO tag branch yet.
- **triBuf tag scheme:** bit31=0 ⇒ external (low bits = voxelId); bit31=1 ⇒ internal cut FP (low 31 bits = `8*voxelId+slot`).

### 1.1 New buffers (design §3.4 / §1.3) — add to `ReconBuffers.cs`. **All atomic accumulators are FLAT `int` buffers** (A2-F1).
| buffer | C# field | element / ComputeBuffer | length | meaning |
|---|---|---|---|---|
| `featurePointBuf` | `CutFP` | float4 (xyz=FP world, w=valid), stride 16 | `8*VoxelCount` | internal cut feature point, slot = `8*voxelId+comp` |
| `cutFPAccumPos` | `CutFPAccumPos` | int, stride 4 | `3*8*VoxelCount` | fixed-pt centroid pos sum, indexed `3*slot+0/1/2` |
| `cutFPAccumCount` | `CutFPAccumCnt` | int, stride 4 | `8*VoxelCount` | centroid count, indexed `slot` |
| `voxelFPCountBuf` | `VoxelFPCount` | uint, stride 4 | `VoxelCount` | per-voxel compCount (0 if uncut) |
| `featurePointNormalBuf` | `CutFPNormal` | int, stride 4 | `3*8*VoxelCount` | cut-FP normal accumulator, indexed `3*slot+0/1/2` (mirrors `_ExtNormalFixed`) |
| `featurePointNormalFBuf` | `CutFPNormalF` | float3, stride 12 | `8*VoxelCount` | normalized cut-FP normal (mirrors `_ExtNormalF`) |

> **Why flat int, not int4 (A2-F1, CRITICAL):** HLSL SM5 `InterlockedAdd` requires a scalar `int`/`uint`
> lvalue. `RWStructuredBuffer<int4>` does NOT allow `InterlockedAdd(buf[i].x,…)`. This mirrors the proven
> Stage-1 `_ExtNormalFixed : RWStructuredBuffer<int>` indexed `3*v+k`. Do NOT use int4.

**`POS_SCALE` = `1<<14` (16384.0f)** — new fixed-point scale for centroid position sums. **Declare in
`FixedPointAtomic.cs`** (next to `ForceScale 1<<10`, `NormalScale 1<<20`) **and `#define POS_SCALE 16384.0`
in `CuttingCommon.hlsl`** (NOT ReconCommon.hlsl — the cut kernels include CuttingCommon) (A2-F2).
**Overflow bound (A2-F2):** max scatter events into one slot ≤ (≤12 incident cut edges of a voxel) × (each
edge owned-side contributes its 2 cut points… but only the owner-side ones land in a given component) — a safe
upper bound is 12 edges × 4 shared-voxel writes ≈ 48 events; `48 × 16384 × 10(world) ≈ 7.9e6 ≪ 2.1e9` int32 max. Safe.

**`_TriCapacity` budget (A1-I3 / A2-F5):** external + cut tris share ONE `triBuf`. Size
`triCapacity = maxExternalTris + maxCutTris`, with `maxCutTris ≤ 2*GridEdgeCount` (≤2 tris per cut point /
≤1 face per owner-component per cut edge). `ReconGridManager` MUST pass this enlarged `triCapacity` to
`ReconBuffers` from the first 2C frame, else the `DC_Stitch`/`BuildCutTriangles` overflow guard silently
drops cut faces as cuts accumulate. Add an `Assert`/log if `TriCounter[0]` approaches `3*TriCapacity`.

## 2. Module map (design §4 — strict fidelity). Kernels in `Shaders/Cutting.compute`; orchestrated by `DualContouring.cs`.
1. `ClearCutFP` — per (voxel·8 slot): zero `cutFPAccumPos`, `cutFPAccumCount`, `featurePointBuf.w`, `featurePointNormalBuf`; per voxel: zero `voxelFPCountBuf`. [design §6 unified clear]
2. `LookupConnectivity` — per voxel: `voxelFPCount = (mask==0)?0:conn4096[mask].compCount`. [design §4.1; paper §2.1.1]
3. `AccumulateCutFP` — per cut point: scatter world pos into the ≤4 shared voxels' owner-component slots. [design §4.2; paper §2.1.1 centroid]
4. `FinalizeCutFP` — per (voxel·8 slot): `_CutFP = (count>0)? sum/POS_SCALE/count : invalid`. [design §4.2]
5. `BuildCutTriangles` — per particle: for each incident CUT grid-edge, this-particle's-component FP in the 4 shared voxels → quad → ≤2 tris into the SAME `triBuf` (internal tag). [design §4.3; paper Fig 2.5]
6. `DC_NormalizeCutFP` — per (voxel·8 slot): normalize `featurePointNormalBuf → featurePointNormalFBuf` (GPU realization of the unified normalize pass for cut slots; mirrors `DC_NormalsNormalize`). [design §6; PAPER-SILENT]

Modules 3+4 are the fixed-point-atomic-then-normalize GPU realization of design's single "ComputeCutFeaturePoints"
(same pattern as normals; not a new architectural module — Rule 1). `DC_NormalsScatter`/`Normalize` (Recon.compute)
and `ReconSurface.shader` become **tag-aware** (design §6) — see §3.6.

## 3. Detailed kernel specs

### 3.1 `LocalCornerOf` (GridConventions EXTEND — the ONLY core-convention addition)
```csharp
// LocalCornerOf: which local corner (0..7, Fig 2.4) of voxel `voxelCoord` coincides with global corner
// `cornerCoord`. -1 if not a corner of that voxel. This is the vertex key feeding vertToComp — it MUST be
// the SAME Fig 2.4 numbering the CONN4096 LUT was built on (design §10). Uses GridConventions.CornerOffset.
public static int LocalCornerOf(int3 cornerCoord, int3 voxelCoord) {
    int3 d = cornerCoord - voxelCoord;
    for (int lc = 0; lc < 8; lc++)
        if (CornerOffset[lc].x==d.x && CornerOffset[lc].y==d.y && CornerOffset[lc].z==d.z) return lc;
    return -1;
}
```
HLSL mirror in `CuttingCommon.hlsl` using the function-encoded `CornerOffset` (same encode-as-function pattern
as 2B's `ES_VoxelOffset`, to avoid SM5 static-multidim-array unreliability):
```hlsl
int3 CornerOffsetHLSL(int lc){ // V0(0,0,0)V1(1,0,0)V2(1,1,0)V3(0,1,0)V4(0,0,1)V5(1,0,1)V6(1,1,1)V7(0,1,1)
    if(lc==0)return int3(0,0,0); if(lc==1)return int3(1,0,0); if(lc==2)return int3(1,1,0); if(lc==3)return int3(0,1,0);
    if(lc==4)return int3(0,0,1); if(lc==5)return int3(1,0,1); if(lc==6)return int3(1,1,1); return int3(0,1,1); }
int LocalCornerOf(int3 cornerCoord,int3 voxelCoord){ int3 d=cornerCoord-voxelCoord;
    [unroll] for(int lc=0;lc<8;lc++) if(all(CornerOffsetHLSL(lc)==d)) return lc; return -1; }
int Conn4096Comp(Conn4096Gpu e,int lc){ // vertToComp[mask][lc] = e.v{lc}
    if(lc==0)return e.v0; if(lc==1)return e.v1; if(lc==2)return e.v2; if(lc==3)return e.v3;
    if(lc==4)return e.v4; if(lc==5)return e.v5; if(lc==6)return e.v6; return e.v7; }
int3 CornerCoordHLSL(int idx){ int nx=_Dims.x+1, ny=_Dims.y+1;     // inverse of CorId (reuse 2B CorId)
    return int3(idx%nx,(idx/nx)%ny, idx/(nx*ny)); }
```
A **CPU round-trip oracle** (test 1, §6) asserts `CornerOffsetHLSL`≡`CornerOffset[]` and `LocalCornerOf` round-trips.

### 3.2 Shared edge predicate (A1-C1 / A1-M4) — single source of truth, matches `BackgroundGrid.gridEdges`
```hlsl
// An interior grid edge (axis, baseC) is VALID iff all 4 EdgeStencil voxels are in-bounds — the EXACT
// predicate BackgroundGrid used to build _GridEdges (so the stitched edge-set ⊆ the populated edge-set).
bool EdgeValid(int axis,int3 baseC){ [unroll] for(int s=0;s<4;s++) if(!VoxInBounds(baseC+ES_VoxelOffset(axis,s))) return false; return true; }
// CUT iff its localEdge bit is set in the FIRST in-bounds stencil voxel (2B set the bit in ALL 4, so any agrees).
bool EdgeIsCut(int axis,int3 baseC){ [unroll] for(int s=0;s<4;s++){ int3 vc=baseC+ES_VoxelOffset(axis,s);
    if(VoxInBounds(vc)) return (_VoxelCutMask[VoxId(vc)] >> ES_LocalEdge(axis,s)) & 1u; } return false; }
```

### 3.3 `LookupConnectivity` (design §4.1; paper §2.1.1)
```hlsl
[numthreads(64,1,1)] void LookupConnectivity(uint3 id){ uint v=id.x; if(v>=(uint)_VoxelCount) return;
    uint mask=_VoxelCutMask[v]&0xFFFu; _VoxelFPCount[v]=(mask==0u)?0u:(uint)_Conn4096[mask].compCount; }
```

### 3.4 `AccumulateCutFP` (design §4.2; paper §2.1.1 "FP = centroid of the component's cut points", plain mean not QEF)
One thread per cut point. World pos at 2C = `cornerPos[owner]+localOffset` (already the ±D/2-offset `P_side`
from 2B, so the two components' centroids end up offset by ≈ D). Scatter into the ≤4 shared voxels; component
is the one containing the OWNER's local corner (VERTEX-keyed; a cut edge straddles two components — gap C4/C54).
**Invariant (A1-I1):** each `cutPoint[i]` contributes to slot `8*v+comp(owner)` AT MOST ONCE per voxel `v`;
distinct cut points sharing a component sum (paper "cut edges calculated twice"). No single-point double-count.
```hlsl
[numthreads(64,1,1)] void AccumulateCutFP(uint3 id){ uint i=id.x; if(i>=_CutPointCounter[0]) return;
    CutPointGpu cp=_CutPoint[i];
    float3 world=_CornerPos[cp.ownerParticle]+cp.localOffset;      // 2C static; 2D: +mul(R_owner,localOffset)
    int3 ownerCoord=CornerCoordHLSL(cp.ownerParticle);
    GridEdgeGpu2 ge=_GridEdges[cp.edgeId]; int axis=ge.axis; int3 baseC=int3(ge.bcX,ge.bcY,ge.bcZ);
    for(int s=0;s<4;s++){ int3 vc=baseC+ES_VoxelOffset(axis,s); if(!VoxInBounds(vc)) continue;
        int v=VoxId(vc); uint mask=_VoxelCutMask[v]&0xFFFu; if(mask==0u) continue;
        int lc=LocalCornerOf(ownerCoord,vc); if(lc<0) continue;     // owner is always an endpoint → lc≥0
        int comp=Conn4096Comp(_Conn4096[mask],lc); int slot=8*v+comp;
        InterlockedAdd(_CutFPAccumPos[3*slot+0],(int)round(world.x*POS_SCALE)); // fixed-point centroid sum
        InterlockedAdd(_CutFPAccumPos[3*slot+1],(int)round(world.y*POS_SCALE));
        InterlockedAdd(_CutFPAccumPos[3*slot+2],(int)round(world.z*POS_SCALE));
        InterlockedAdd(_CutFPAccumCnt[slot],1); } }
```
> **PAPER-SILENT assumption (A1-I2), planar-only:** the owner-particle's connectivity component is assumed to
> coincide with its geometric blade side (true for a single planar cut, where the two endpoints are on opposite
> sides AND in opposite components). For high-popcount/cross cuts (Fig 3.10) the gap DIRECTION may pinch locally
> — this is a KNOWN 2C limitation (not a re-weld; component selection stays vertex-correct), revisited if needed.
> The 2C oracle is scoped to the planar case.

### 3.5 `FinalizeCutFP` (design §4.2)
```hlsl
[numthreads(64,1,1)] void FinalizeCutFP(uint3 id){ uint s=id.x; if(s>=8u*(uint)_VoxelCount) return;
    int c=_CutFPAccumCnt[s];
    if(c>0){ float3 sum=float3(_CutFPAccumPos[3*s+0],_CutFPAccumPos[3*s+1],_CutFPAccumPos[3*s+2])/POS_SCALE;
             _CutFP[s]=float4(sum/(float)c,1.0); }   // centroid = mean of the component's cut points
    else   { _CutFP[s]=float4(0,0,0,0); } }          // invalid → no NaN
```

### 3.6 `BuildCutTriangles` — per-particle stitching (design §4.3; paper Fig 2.5)
One thread per particle. Corner coord `c`; its ≤6 incident grid edges as `(axis,baseCorner)`:
`+x=(0,c) -x=(0,c-ex) +y=(1,c) -y=(1,c-ey) +z=(2,c) -z=(2,c-ez)`. Each undirected edge has one base corner, so
each cut edge is visited by exactly its two endpoint particles → each endpoint emits its OWN face = two parallel
faces (the gap); no dup, no miss. Use the §3.2 shared predicate (rejects negative/boundary bases automatically).
```hlsl
[numthreads(64,1,1)] void BuildCutTriangles(uint3 id){ uint p=id.x; if(p>=(uint)_CornerCount) return; int3 c=CornerCoordHLSL(p);
    int axes[6]={0,0,1,1,2,2}; int3 bs[6]={c,c-int3(1,0,0),c,c-int3(0,1,0),c,c-int3(0,0,1)};
    for(int e=0;e<6;e++){ int axis=axes[e]; int3 baseC=bs[e];
        if(!EdgeValid(axis,baseC)) continue; if(!EdgeIsCut(axis,baseC)) continue;
        float3 fp[4]; int tag[4]; bool ok[4]; int n=0;
        for(int s=0;s<4;s++){ ok[s]=false; int3 vc=baseC+ES_VoxelOffset(axis,s);
            int v=VoxId(vc); uint mask=_VoxelCutMask[v]&0xFFFu; if(mask==0u) continue;     // vc in-bounds by EdgeValid
            int lc=LocalCornerOf(c,vc); if(lc<0) continue;
            int comp=Conn4096Comp(_Conn4096[mask],lc); int slot=8*v+comp;
            float4 f=_CutFP[slot]; if(f.w<0.5) continue;                                   // invalid FP → missing corner
            fp[s]=f.xyz; tag[s]=(int)(0x80000000u | (uint)slot); ok[s]=true; n++; }
        EmitCutQuad(ok,fp,tag,n); } }   // partial-quad: 4→2 tris, 3→1, <3→none
```
**Winding (A1-C2 — resolved):** rendering is **two-sided** (`Cull Off` + fragment normal flip via `SV_IsFrontFace`,
§3.7), so face winding is **display-sign-agnostic** — no `insideA`-analog needed and no "single global flip"
gamble. `EmitCutQuad` emits the 4 valid slots in stencil order with the DC diagonal `(0,1,2)+(0,2,3)`; the
3-valid case emits one tri over the 3 present slots in order. `EmitCutQuad` appends via
`InterlockedAdd(_TriCounter[0], 3*ntris, base)` (SAME counter as DC_Stitch — unified triBuf) with the
overflow guard `base+3*ntris > _TriCapacity*3 → return`, and writes the internal-tagged ids `tag[s]`.

### 3.7 Unified normals + render (design §6) — tag-aware
`DC_NormalsScatter` (Recon.compute) gains a per-vertex `FetchPos` so **mixed** triangles (some external, some
internal vertices) compute the correct face normal (A2-F4a):
```hlsl
float3 FetchPos(int raw){ int idx=raw&0x7FFFFFFF; return (raw&0x80000000)? _CutFP[idx].xyz : _VoxelExternalFP[idx].xyz; }
// per tri: p0=FetchPos(_Tri[3t+0]); p1=FetchPos(_Tri[3t+1]); p2=FetchPos(_Tri[3t+2]); fn=cross(p1-p0,p2-p0);
// per vertex v in {0,1,2}: raw=_Tri[3t+v]; idx=raw&0x7FFFFFFF;
//   if(raw&0x80000000) AtomicAddNormal(_CutFPNormal, idx, fn);  else AtomicAddNormal(_ExtNormalFixed, idx, fn);
```
Bind `_CutFP`,`_VoxelExternalFP`,`_CutFPNormal`,`_ExtNormalFixed` to the scatter kernel. **Zero-cut path is
byte-identical** to Stage-1 (all tags external) — no regression (A2-F4b).
`DC_NormalizeCutFP` (NEW, in Cutting.compute; module 6) normalizes the cut slots, mirroring `DC_NormalsNormalize`:
```hlsl
[numthreads(64,1,1)] void DC_NormalizeCutFP(uint3 id){ uint s=id.x; if(s>=8u*(uint)_VoxelCount) return;
    float3 nv=float3(_CutFPNormal[3*s+0],_CutFPNormal[3*s+1],_CutFPNormal[3*s+2])/NORMAL_SCALE;
    float len=length(nv); _CutFPNormalF[s]=(len>1e-6)? nv/len : float3(0,0,0); } // domain G(8*VoxelCount), uniform _CutSlotCount optional
```
`ReconSurface.shader` (A1-C2 + A2-F6): set `Cull Off`; decode tag; two-sided normal:
```hlsl
// vert: int raw=_Tri[vid]; int idx=raw&0x7FFFFFFF; bool internal=(raw&0x80000000)!=0;
//   wp = internal? _CutFP[idx].xyz : _VoxelExternalFP[idx].xyz;
//   o.n = internal? _CutFPNormalF[idx] : _ExtNormalF[idx];
// frag(v2f i, bool front:SV_IsFrontFace): float3 n = front? i.n : -i.n;  // two-sided lit (cut walls + back faces)
```
**Graphics-shader binding site (A2-F6/F7):** in the C# render path that issues `DrawProceduralIndirect`
(`ReconGridManager`/`DemoSceneSetup`), call `mat.SetBuffer("_CutFP",rb.CutFP)` and
`mat.SetBuffer("_CutFPNormalF",rb.CutFPNormalF)` (NOT `cs.SetBuffer`) **unconditionally** from the first 2C
frame (even when zero cuts → buffers hold zeros, never sampled by external verts). Add alongside the existing
`_Tri`,`_VoxelExternalFP`,`_ExtNormalF` material binds.

## 4. Dispatch order (design §6 — ONE triCounter reset, ONE indirect-args write). Insert into `DualContouring.Build()`:
```
set scalars (+ _CornerCount, _CutSlotCount=8*VoxelCount, bind _CutPointCounter, POS_SCALE via #define)
RebuildIsectWorld
DC_FeaturePoints                       (external FP, Stage 1)
ClearCutFP        [NEW]   G(8*VC)+G(VC) zero cutFPAccumPos/Cnt + _CutFP.w + _CutFPNormal + voxelFPCount   ┐
LookupConnectivity[NEW]   G(VC)         voxelFPCount = compCount                                          │ 2C
AccumulateCutFP   [NEW]   G(cutPointCounter, dispatched on capacity) scatter cut points → component sums  │  FP
FinalizeCutFP     [NEW]   G(8*VC)       centroid → _CutFP, valid                                          ┘
reset TriCounter=0                      (single reset, head of geometry)
DC_Stitch                 G(SurfaceEdgeCount) external faces → triBuf
BuildCutTriangles [NEW]   G(CornerCount)      cut faces → SAME triBuf
DC_ClearNormals           G(VC)         zero _ExtNormalFixed   // NOTE: _CutFPNormal already zeroed in ClearCutFP (top), NOT here (A2-F3)
DC_NormalsScatter         G(TriCapacity) tag-aware (external + internal)                                  [MODIFIED]
DC_NormalsNormalize       G(VC)         external normals
DC_NormalizeCutFP [NEW]   G(8*VC)       cut-FP normals                                                    [NEW]
DC_WriteIndirectArgs      1             TriCounter[0] → indirect draw
```
`CutDetector.Dispatch` (2B) runs BEFORE this block (after physics). `AccumulateCutFP` is dispatched over the
cut-point capacity with the kernel's own `i>=_CutPointCounter[0]` guard (counter lives on GPU). `BuildCutTriangles`
reads `_CutFP` only after `FinalizeCutFP` wrote it (separate Dispatch = implicit barrier — A2-F3 CONFIRM-OK).

## 5. Coding-discipline compliance
- **Rule 1:** kernels map to design §4 modules; Accumulate/Finalize/NormalizeCutFP are the documented
  fixed-point-atomic-then-normalize GPU realization of "ComputeCutFeaturePoints" + the unified normalize pass
  (same pattern as Stage-1 normals), NOT new architecture.
- **Rule 2:** every kernel header cites paper §2.1.1 / Fig 2.4 / Fig 2.5 + design §4.x/§6.
- **Rule 3:** `compCount`, `vertToComp`/`comp`, `featurePointBuf`/`_CutFP`, `cutFPAccumPos/Cnt`, `POS_SCALE`, `DC_NormalizeCutFP`.

## 6. Verification (`CutSurface_GpuOracle_Tests.cs`, EditMode; 3×3×3 grid, planar z=1.5 cut — reuse 2B scenario)
1. **LocalCornerOf round-trip (CPU, runs anywhere):** ∀ voxel × 8 corners, `LocalCornerOf` returns 0..7 and
   `voxelCoord+CornerOffset[LocalCornerOf]==cornerCoord`; non-corner → -1. Plus `CornerOffsetHLSL`≡`CornerOffset`.
   (Pins the vertex key to the Fig 2.4 numbering the LUT was built on — design §10; A1 merge gate.)
2. **Connectivity (GPU):** after the planar cut, cut voxels `voxelFPCount==2`, uncut `==0`.
3. **Centroid + count (GPU, A1-I1):** the two slots `_CutFP[8*v+compA]`,`[..compB]` are valid; assert the
   **count divisor** (`_CutFPAccumCnt`) equals the hand-derived #cut-points-per-component (catches acc.w off-by-one),
   and the two centroids' separation ≈ D.
4. **No re-weld (headline, A1):** a V0-side particle selects the {V0,…} component FP in ALL shared voxels, a
   V1-side particle selects {V1,…}; assert the two faces' vertices come from DIFFERENT slots and `triCounter`
   grew by the expected cut-tri count.
5. **Tag decode (GPU):** ≥1 triBuf entry has bit31 set and its low bits index a valid `_CutFP` slot.

**Unity Play (user):** blade sweep across the deforming sphere → a visible slit opens (gap ≈ D); both cut walls
render (two-sided, not black); outer surface unaffected; FPS stable as cuts accumulate. (Physical separation = 2D.)

## 7. Risks / watch-items (post-audit residual)
- **R-A numbering mismatch** (highest): `LocalCornerOf`/`Conn4096Comp`/`CornerOffsetHLSL` must equal the 2A LUT
  numbering — locked by test 1 (merge gate).
- **R-B two-sided render** replaces the winding gamble; verify cut walls lit on both sides (no black).
- **R-C POS_SCALE** 1<<14, bound derived (≤~8e6 ≪ int32). `#define` in CuttingCommon.hlsl.
- **R-D tag-aware normals** with per-vertex `FetchPos` (mixed tris correct); zero-cut path byte-identical.
- **R-E _TriCapacity** enlarged by `2*GridEdgeCount` from frame 1 (ReconGridManager) — else silent drop.
- **R-F cross-cut gap direction** = known planar-only limitation (A1-I2), not a re-weld.
