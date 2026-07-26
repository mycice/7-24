# Plan — Rest-pose cut DETECTION (replaces the detection-band stopgap)

Date: 2026-06-28
Status: DRAFT → audit → implement → review
Supersedes: the `_CutEdgeBand` detection-band edits (workflow w4amk1gw7 rejected them: critical over-cut).

## 1. Goal / root cause

The horizontal sweep leaves ~24 residual HINGE edges (lower hemisphere stays attached). Root cause
(workflow wpm9fwlmt, 3/3 skeptics refuted=false): the swept cutting quad is a ZERO-thickness plane
FIXED at world `bladeY=3.125`, but `DetectCut` M-T-tests it against **deformed** corner positions; the
~1 s progressive sweep lets gravity sag a Y grid-edge >L/2 out of that thin plane before its Z-band is
tested → M-T `t_ray` leaves `[0,1]` → no hit → cut bit never set → `SeverLinks` (gated on `EdgeIsCut`)
skips it → permanent hinge. Rest pose: 257 occupied Y-edges, blade hits 257/257; runtime 233 → 24 hinges.

The detection-band fix (widen M-T's finite-edge t-clamp) was REJECTED in review w4amk1gw7: `bladeY`
sits exactly L/2 from BOTH corner layers, so a symmetric band over-cuts the upper layer once the tissue
pre-sags (~0.06–0.08 L under-damped), severing the y=3.25 layer from both neighbours → a free-floating
1-voxel disc + ~2L kerf. The reviewer's recommended durable fix (and the user's choice):

**Detect WHICH edges are cut against the UNDEFORMED (rest) edge positions** — the paper's own basis
(newpaper_clean.txt:364-366: the cut point's local coords are recorded "based on the condition that the
edge has **not** undergone any deformation"). The cut targets a fixed MATERIAL plane → complete,
deterministic, ZERO over-cut, independent of the gravity-deformation race. The cut POINT is still
emitted on the DEFORMED edge so the cut surface tracks the tissue (paper §2.1.2 reconstruct-on-deformed).

## 2. Paper alignment — DECLARED DEVIATION (audit w87ghv7f6 R1)

**Detect-on-rest is a DELIBERATE, JUSTIFIED DEVIATION, not a literal paper reading.** The paper M-T-tests
the DEFORMED mesh (§2.1.3; clean:357-358 "distorted and no longer a uniform hexahedron" at cut time); the
clause clean:363-366 ("edge has not undergone any deformation") governs the cut-point RECORDING frame
(material-frame local coords for later reconstruction via the [22] rotated axes), NOT the intersection-test
frame. The deviation is justified: (a) the paper's deformed-mesh detection works in its surgeon-driven setup, but
under our SCRIPTED fixed-world-Y blade + soft tissue it suffers the detection-vs-deformation race (the
hinge bug) — detecting on the rest lattice cuts a fixed material plane that **reproduces the paper's
claimed Fig 3.9 full-disconnection RESULT** (clean:1157-1164) without the race (it does NOT claim to equal
the paper's deformed-mesh detection edge-set; it gives the intended complete cut the paper's setup also
achieves); (b) it removes the gravity-vs-sweep race; (c) it is the correct foundation for the deferred 2D-2
R-frame, where the world blade is transformed into each particle's MATERIAL frame before this rest M-T =
exactly the paper's [22] rotated-axes mechanism (clean:359-361). Emit-on-deformed keeps the surface tracking
the tissue (§2.1.2 reconstruct-on-deformed).

- The set of cut edges is a material/topological fact, computed once on the static lattice.
- **Emission on deformed frame** = paper §2.1.2 the cut point follows the particle's current position.
- For the **pure-translation** demo (lower half falls, no rotation) the rest plane = the initial world
  plane = the intended cut, and the world-swept blade's (x,z) window = the rest cross-section's (x,z), so
  the rest M-T cuts exactly the intended material disc. COMPLETE.
- **Rotation (2D-2):** under rotation the world blade must be transformed into each particle's material
  frame via the Berndt R before the rest M-T — DEFERRED to 2D-2. Rest detection is its foundation; add a
  TODO. (Today's demo has no rotation, so the world blade ≈ material blade.)

## 3. Design

### 3.1 DetectCut — detection on REST, emission on DEFORMED
Current (Cutting.compute ~270-298): `O = _CornerPos[idA]` (deformed), `Dvec = _CornerPos[idB]-O`;
broadphase AABB cull; `MollerTrumbore(O,Dvec,T1/T2,...)`; `P_hit = O + t_ray*Dvec`.

New:
```hlsl
// REST edge (analytic; grid is the uniform lattice cornerPos = origin + coord*L, BackgroundGrid.cs:107).
// paper §2.1.2 (clean:364-366): the cut is recorded on the UNDEFORMED edge — detect on rest so the cut
// targets a fixed MATERIAL plane and is immune to the gravity-vs-sweep deformation race (the hinge bug).
float3 O_rest    = _GridOrigin + (float3)cornerA_coord * _VoxelL;
float3 Dvec_rest = (float3)axisDir * _VoxelL;            // = rest(B) - rest(A)

// broadphase: cull on the REST edge (static → the target layer always straddles → never wrongly culled)
float3 eMin = min(O_rest, O_rest + Dvec_rest);
float3 eMax = max(O_rest, O_rest + Dvec_rest);
if (any(eMax < _SweptAABBMin) || any(eMin > _SweptAABBMax)) return;

// M-T on the REST edge → which edges the blade crosses (EXACT paper Eq6-8, NO band)
float t1,u1,v1,t2,u2,v2;
bool hit1 = MollerTrumbore(O_rest, Dvec_rest, _T1V0,_T1V1,_T1V2, t1,u1,v1);
bool hit2 = MollerTrumbore(O_rest, Dvec_rest, _T2V0,_T2V1,_T2V2, t2,u2,v2);
... anyHit / nearer t_ray as today ...

// cut POINT on the DEFORMED edge at the rest-determined parametric position (surface tracks the tissue)
float3 O_def    = _CornerPos[idA];
float3 Dvec_def = _CornerPos[idB] - O_def;
float3 P_hit = O_def + t_ray * Dvec_def;                 // t_ray in [0,1] by construction (rest straddle)
```
Then mark cut bits + `EmitCutPoint(P_hit, idA/idB, e)` exactly as today (EmitCutPoint already uses the
deformed `_CornerPos[p]` for `side`/`localOffset`, so it stays consistent).

NOTE: with rest detection a hit means the rest edge straddles, so `t_ray ∈ [0,1]` naturally — no clamp
needed. (Keep a `saturate(t_ray)` only as a defensive guard against FP edge cases; audit to decide.)

### 3.2 New uniforms (Cutting.compute)
```hlsl
float  _VoxelL;      // voxel side length L (rest lattice spacing)
float3 _GridOrigin;  // rest lattice origin (cornerPos_rest = _GridOrigin + coord*_VoxelL)
```

### 3.3 Revert the detection band (workflow w4amk1gw7 rejected)
- `CuttingCommon.hlsl`: revert `MollerTrumbore` to the original signature (drop `float tBand`); restore
  the finite-edge clamp to `t ∈ [-EPS, 1+EPS]`. (Exact paper Eq6-8.)
- `Cutting.compute`: remove the `_CutEdgeBand` uniform; DetectCut M-T calls drop the band arg; MT_Test
  drops the `0.0` arg.
- `CutDetector.cs`: remove `CutEdgeBand` + `AabbInflate` fields, the `_CutEdgeBand` bind, the AABB
  inflation (restore `aabbMin`/`aabbMax` unmodified). Add `_VoxelL` + `_GridOrigin` binds.
- `ReconGridManager.cs`: remove the `cutDetector.AabbInflate = L` line (replaced by passing L/origin).

### 3.4 CutDetector — bind L + origin
`Dispatch(tool, rb, dims)` needs L + origin. Source: the manager's `L` and the grid `origin`. Add them
to `Dispatch` signature (or store on CutDetector at construction). Bind:
`_cs.SetFloat("_VoxelL", L); _cs.SetVector("_GridOrigin", origin);`

### 3.5 ReconGridManager
Pass `L` and the grid `origin` (the BackgroundGrid origin) into `cutDetector.Dispatch`. The manager has
`L` (field) and the grid (`g.origin`).

## 4. Test impact (MUST update — load-bearing)
The GPU oracle tests run in REST pose (`_CornerPos == rest`), so rest-vs-deformed give the SAME result —
BUT only if `_VoxelL`/`_GridOrigin` are bound. If left unset they default to 0 → `O_rest = 0` → all edges
collapse to the origin → BROKEN. So:
- `CutBitPropagation_GpuOracle_Tests` (L=1, origin=0) and `CutSurface_GpuOracle_Tests` (L=1, origin=0):
  add `cs.SetFloat("_VoxelL", 1f)` + `cs.SetVector("_GridOrigin", Vector3.zero)` to their DetectCut setup.
- `MollerTrumbore_GpuOracle_Tests` (MT_Test): unaffected — MT_Test reads `_T1V0/_T1V1`/`_T2*` directly,
  does not use `_VoxelL`/`_GridOrigin`; just drop the reverted `tBand` arg.
- Confirm rest detection reproduces the existing assertions (the tests' cornerPos == origin+coord*L, so
  O_rest == O_deformed → identical M-T → same cut bits / cut points / counts).

## 5. Risks / edge cases (for the audit)
1. **Rest plane = intended cut?** Yes — rest = initial world; the demo cuts the initial mid-plane.
2. **World-Z sweep vs rest-Z edges:** the blade sweeps world-Z; rest edges at rest-Z. For the demo the
   tissue falls in Y (negligible Z translation), so world-Z ≈ rest-Z and the sweep covers the rest
   cross-section. Document the assumption (breaks only under large Z translation, not present here).
3. **AABB cull on rest:** rest edges static → the target layer always straddles bladeY → never wrongly
   culled; no inflation needed. Verify the swept AABB (world) and rest edges align (they do: rest = initial
   world, blade sweeps the initial (x,z) extent).
4. **t_ray ∈ [0,1] guaranteed?** A rest hit means the rest edge straddles the plane between its endpoints
   → t_ray ∈ [0,1]. So `P_hit = O_def + t_ray*Dvec_def` lands on the deformed edge. (Defensive saturate?)
5. **Inactive/severed corners:** `_CornerPos` for a frozen corner is still its last position → P_hit_def
   well-defined. No issue.
6. **Backward-compat / zero-cut:** null cutting shader path unchanged; `_ToolValid==0` early-out unchanged.
7. **Over-cut:** rest detection cuts ONLY the rest target layer (the only layer straddling rest-y=3.125)
   → ZERO over-cut by construction (the band's failure mode is structurally absent).
8. **Completeness:** rest target layer = 257 edges, blade sweeps full rest (x,z) → all 257 cut → 0 hinges.

## 6. Acceptance (user Unity verify)
- `cutPoints` reaches ~514 (257 edges × 2), `severedNbr` ~ baseline + 514.
- Lower hemisphere fully detaches and falls MONOTONICALLY (no residual hinge, no floating disc, no 2L kerf).
- Cut surface still tracks the deforming tissue (cut points on deformed edges).

## 7. Out of scope / deferred
- 2D-2 R-frame: transform the world blade into each particle's material frame before the rest M-T (for
  rotation). Today's rest M-T assumes blade trajectory ≈ material trajectory (true for pure translation).
- 2D-2b temporal FP interpolation. 2D-3 ablation.
