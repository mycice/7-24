# Stage 1B-i Cluster C1 Report

## Files Modified / Created

- `Assets/ReconGridDC/Preprocess/BackgroundGrid.cs` — added `Spring`/`BendPair` structs, physics fields (`mass`, `nbrIdx`, `restNbr`, `pinned`, `springs`, `bendPairs`, `cornerCount`), `NbrDelta`/`PairSlots` static tables, `BuildPhysics()`, called at end of `Build`.
- `Assets/ReconGridDC/Tests/PhysicsTopology_Tests.cs` — NEW: CPU tests for spring count (54 in 2x2x2), bend pair count (15*cornerCount), alive counts (interior=15, corner=3), collinear theta0=pi, nbrIdx symmetry, mass=1, pinned=0. GPU test `PhysicsBuffers_AllocateUploadDispose` [Category("GPU")].
- `Assets/ReconGridDC/Recon/ReconBuffers.cs` — added `SpringGpu`(16B) and `BendPairGpu`(32B) private interop structs; `CornerCount`/`SpringCount`/`BendPairCount` properties; physics `ComputeBuffer` fields (`Vel`, `Mass`, `Pinned`, `ForceInt`, `ExtForce`, `KSlope`, `YTrialPos`, `YTrialVel`, `Springs`, `BendPairs`, `NbrIdx`, `RestNbr`, `ErrMax`); alloc in ctor, upload (marshal + SetData) in `Upload`, dispose in `Dispose`.

## Commit Hashes

- Task 1: `2a0624d` — feat(reconDC/1B): preprocess springs + bending pairs + neighbors + mass/pin
- Task 2: `ccea020` — feat(reconDC/1B): physics GPU buffers (vel/mass/forces/RK scratch/springs/bendpairs)

## Stride Confirmations

| Buffer | Stride | Verified |
|---|---|---|
| `SpringGpu` | 16 bytes (int+int+float+float) | YES |
| `BendPairGpu` | 32 bytes (int+int+int+float+int+int+int+int) | YES |
| `KSlope` element | 24 bytes (float3 dx + float3 dv) | YES |
| `Vel`, `YTrialPos`, `YTrialVel`, `ExtForce` | 12 bytes (float3) | YES |
| `Mass` | 4 bytes (float) | YES |
| `Pinned`, `ForceInt`, `NbrIdx`, `ErrMax` | 4 bytes (int/uint) | YES |
| `RestNbr` | 12 bytes (float3) | YES |

## PairSlots Verification

Slot 0=(0,1)=-x/+x, slot 9=(2,3)=-y/+y, slot 14=(4,5)=-z/+z (collinear). Test uses `{0, 9, 14}`.

## Concerns

1. `cornerCount` public field is set inside `BuildPhysics()` (called from `Build`). A local `int cornerCount` in `Build` also exists (pre-existing) but only shadows within Build's scope; the public `g.cornerCount` is correctly set by `BuildPhysics`. No conflict.
2. `BackgroundGrid` is an `AllInside` lattice in tests — all `cornerInside` values are 1, so `surfaceEdges.Length == 0` (no sign-change edges). That's fine; physics topology does not depend on surface edges.
3. Stage 1A tests (`BackgroundGrid_Tests`, `GridConventions_Tests`, etc.) are unaffected — no existing fields removed.
4. GPU test `[Category("GPU")]` requires Unity GPU to run; will pass on GPU machine per plan.
