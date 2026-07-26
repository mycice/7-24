# Stage 1B-ii Cluster C1 Report — Task 1: RebuildIsectWorld

## Files Modified / Created

- `Assets/ReconGridDC/Recon/ReconBuffers.cs`
  - Added `public int IsectCount { get; private set; }` property (= Math.Max(1, g.isect.Length))
  - Added `public ComputeBuffer IsectWorld;` (float3, stride 12, length = IsectCount)
  - Allocated `IsectWorld` in ctor alongside `Isect`; disposed in `Dispose()`

- `Assets/ReconGridDC/Shaders/Recon.compute`
  - Added `#pragma kernel RebuildIsectWorld` (first pragma)
  - Added `int _IsectCount;` uniform
  - Added `RWStructuredBuffer<float3> _IsectWorld;` shared buffer (RW; DC_FeaturePoints reads it without writing — safe in HLSL SM5, two separate dispatches)
  - Added `RebuildIsectWorld` kernel: `_IsectWorld[e] = lerp(_CornerPos[it.cornerA], _CornerPos[it.cornerB], it.t)`
  - Modified `DC_FeaturePoints`: initial x0 now reads `_IsectWorld[off+s]` (was `_Isect[off+s].globalRest`); QEF loop uses `p_i = _IsectWorld[off+s2]`, `n_i = _Isect[off+s2].normal` (normal unchanged — PAPER-SILENT frozen)

- `Assets/ReconGridDC/Recon/DualContouring.cs`
  - Added `kRebuild = cs.FindKernel("RebuildIsectWorld")` in ctor
  - `Build()` now sets `_IsectCount = rb.IsectCount` and dispatches `RebuildIsectWorld` as the **first** step (G(rb.IsectCount) groups) before `DC_FeaturePoints`
  - `Bind()` now includes `cs.SetBuffer(k, "_IsectWorld", rb.IsectWorld)` bound to all kernels

- `Assets/ReconGridDC/Tests/RebuildIsectWorld_GpuOracle_Tests.cs` (new)
  - `[Category("GPU")]` test class
  - `RebuildIsectWorld_LerpsAlongDeformedEdge`: displaces corner (1,0,0) by +1 in x; asserts IsectWorld has entry at x=1.0
  - `RebuildIsectWorld_AtRest_MatchesGlobalRest`: at rest corners, asserts IsectWorld == globalRest (confirms static 1A path)

## Static 1A Path Preservation Confirmed

At rest: `CornerPos[cornerA] = restA`, `CornerPos[cornerB] = restB`.
`RebuildIsectWorld`: `IsectWorld[e] = lerp(restA, restB, t)`.
This is exactly how `globalRest` was computed at preprocess (same lerp along the same rest edge).
Therefore DC_FeaturePoints receives identical `p_i` values as before, and `n_i` is unchanged.
The 1A oracles (`Qef_GpuOracle`, `Stitch_GpuOracle`, `DC_Normals_Sphere_PointOutward`) and the static sphere demo must still pass.
The second test (`RebuildIsectWorld_AtRest_MatchesGlobalRest`) explicitly verifies this.

## Concerns / Notes

- `_IsectWorld` is declared `RWStructuredBuffer<float3>` globally (needed by `RebuildIsectWorld`). `DC_FeaturePoints` reads it without writing — this is valid in HLSL SM5 (no data hazard: separate Dispatch calls have an implicit UAV barrier between them).
- `Bind()` binds `_IsectWorld` to all kernels. Kernels that don't declare it in their signature silently ignore the binding — Unity's ComputeShader API behavior is safe here (no error for an unused binding).
- The `IsectCount` property is clamped to `Math.Max(1, g.isect.Length)` consistent with all other buffer allocs. The GPU kernel guards with `if (e >= (uint)_IsectCount) return;` so the dummy entry at index 0 (when isect.Length == 0) is never written to a bad address.
- No `.meta` files committed (Unity generates them automatically on import).
