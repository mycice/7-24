# Stage 2D-1b — Tissue-only physics + horizontal cut (make a severed piece FALL)

> Root-cause fix from debug workflow w40o668qp. Severing (2D-1) is correct, but NO piece falls because the
> PHYSICS runs on the FULL 25³ corner lattice (incl. empty corners around/above the sphere) while the cut is
> occupancy-gated to tissue — so the never-severed empty-space spring/bend "cage" rigidly hangs both halves
> from a pinned corner layer that sits in EMPTY space 4 layers ABOVE the sphere. This violates paper §2.1.4
> (paper:485-499): "particles outside the isosurface are directly deleted and no longer participate in force
> calculations or surface generation." Fix = exclude outside-isosurface corners from the physical model
> (Option A, paper-faithful) + re-orient to the horizontal Fig-3.9 cut so the lower hemisphere falls straight
> down (translation, not rotation → also sidesteps the missing 2D-2 R-frame, keeping the cut walls clean).

## Root cause (verified, file:line)
- `BackgroundGrid.BuildPhysics` builds springs (cs:295-303) + 15 bend pairs/corner (cs:310-329) for EVERY
  corner; `cornerInside` (cs:99) is never read there.
- `Physics.compute` integrates every corner, guarding ONLY `_Pinned` (ComputeSlope:221, IntegrateY5:250,
  ComputeError:284); `MassSpringSolver` dispatches over all `CornerCount`.
- Cut is occupancy-gated: DetectCut skips empty voxels (Cutting.compute:336); SeverLinks acts only on
  `EdgeIsCut` (cumulative `_VoxelCutMask`, set only in occupied voxels) → empty-cage edges never severed.
- Demo pins `j=dims.y=24` → world y=6.0; sphere top y=5.0 → pin layer is pure empty space above the sphere.
- Gravity is correct (`(0,-9.81*mass,0)`, mass=1) — NOT the problem (a fully-disconnected piece free-falls).

## The fix — Option A: tissue-only physics (paper §2.1.4)
A corner is **active** iff it belongs to ≥1 OCCUPIED voxel (= inside corners + the one-ring of boundary
corners, so the surface shell stays load-bearing). Inactive (fully-outside) corners are frozen like pinned and
have NO springs/bend-pairs → the empty cage ceases to exist → a cut tissue piece is truly free → it falls.

### Steps (ordered)
1. **`Preprocess/BackgroundGrid.cs`** — compute `byte[] cornerActive` (public field), `active[c]=1` iff any
   incident voxel is occupied. Cleanest: `cornerActive` all 0; for each voxel `v` with `voxelOccupied[v]!=0`,
   set `cornerActive[voxelCorner[8*v+0..7]]=1` (reuse the existing voxelCorner + voxelOccupied). In
   `BuildPhysics`: SKIP emitting a structural spring if EITHER endpoint is inactive (gate the `springList.Add`
   at cs:295-303); set `bendPairs[*].alive=0` when the center OR either arm corner is inactive (gate at
   cs:310-329). (NbrIdx can stay — the structural force already skips a neighbor whose link we want dead; but
   for cleanliness also set `nbrIdx[6*c+s]=-1` toward an inactive neighbor, OR rely on freezing. Prefer:
   leave nbrIdx, freeze inactive corners in-kernel — see step 3 — so an inactive neighbor contributes a force
   only if it moves, and it won't because it's frozen. Simpler + fewer edits. BUT a frozen inactive neighbor
   at REST still forms a valid spring with an active corner → it would ANCHOR the active corner to the rest
   empty lattice. So we MUST drop those springs: skip springs where either endpoint inactive — implement that.)
2. **`Recon/ReconBuffers.cs`** — add `public ComputeBuffer CornerActive` (`int[CornerCount]`, stride 4): alloc
   next to `Pinned`, upload (`byte→int`) in `Upload` next to `Pinned.SetData`, dispose next to `Pinned?.Dispose`.
3. **`Shaders/Physics.compute`** — add `StructuredBuffer<int> _Active;` next to `_Pinned`. In `ComputeSlope`
   and `IntegrateY5` treat inactive EXACTLY like pinned (zero slope / freeze pos+vel): guard becomes
   `if (_Pinned[p]!=0 || _Active[p]==0)`. In `ComputeError` early-return for inactive too (don't inflate the
   RK45 error norm with frozen corners).
4. **`Physics/MassSpringSolver.cs`** — bind `cs.SetBuffer(k,"_Active", rb.CornerActive)` alongside `_Pinned`
   in `Bind`. (Dispatch domain unchanged — inactive corners are frozen in-kernel.)
5. **`Demo/ReconGridManager.cs`** — (a) PIN only TISSUE corners in the sphere's TOP band: replace the
   empty-`j=dims.y` pin loop with: for each corner, if `cornerActive[c]==1` (or cornerInside) AND
   `cornerPos[c].y >= (sphereCenter.y + sphereRadius) - L`, pin it. (b) RE-ORIENT to a HORIZONTAL cut
   (Fig 3.9): blade = a line spanning X at fixed `bladeY = sphereCenter.y + L*0.5` (off the corner plane),
   swept in Z. `S=(xMin, bladeY, bladeZ)`, `E=(xMax, bladeY, bladeZ)` with `xMin/xMax = sphereCenter.x ∓
   sphereRadius*bladeSpan` (span≈1.4, overshoot the sphere); advance `bladeZ` over `[center.z-range,
   center.z+range]`. The swept-quad normal becomes ≈ ±Y → severs Y-axis edges → horizontal cut plane y=bladeY.
   NO DetectCut/SeverLinks change (n_cut + M-T tri are computed generically from the swept quad). Result: the
   LOWER hemisphere (y<bladeY) is unpinned + fully severed → free-falls straight down (Fig 3.9).

### Why no other change is needed
- The cut detection/severing is generic in the swept-quad geometry → horizontal works unchanged.
- Gravity is correct; a disconnected piece free-falls (no ks softening needed for the FALL; ks only affects
  pre-cut internal sag, which is fine).
- The horizontal cut makes the free piece TRANSLATE (not rotate) → the static-offset cut FP stays valid →
  clean walls WITHOUT the 2D-2 R-frame (which is deferred).

## Verification
- **Pre-flight sanity (optional ~5 min)**: swap `SphereLevelSet`→`BoxLevelSet` (already implemented) nearly
  filling the lattice + pin top → confirm the cut separates (proves the empty cage was the bridge).
- **GPU oracle** (extend Sever/Physics oracle): on a small grid with some outside corners, assert inactive
  corners have no live springs/bend-pairs and are frozen (zero slope), and that after a full tissue cut the
  free side receives zero force from the pinned side.
- **Unity Play (user)**: horizontal sweep through the sphere → the LOWER hemisphere detaches and falls under
  gravity; the pinned top cap stays; cut walls clean (no jagged cantilever artifacts); FPS stable.

## Deviations / notes
- Excluding outside-isosurface corners is PAPER-FAITHFUL (§2.1.4) — record it as the canonical behavior, not a
  deviation. It is the substrate for 2D-3 ablation (delete == set active=0).
- The 2D-2 co-rotational R-frame remains deferred (the horizontal translating demo doesn't expose it); still
  needed later for a ROTATING cut and for full deformation fidelity (paper:359-368, 1165-1166).
- Bounce (under-damped relative-only Kelvin-Voigt) is the paper's model; a global drag would be PAPER-SILENT —
  leave unless the user wants a calmer demo.
