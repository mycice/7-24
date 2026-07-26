# Stage 1B-i Cluster C2 Report — Tasks 3-5

## Files Delivered

- `Assets/ReconGridDC/Shaders/PhysicsCommon.hlsl` — shared constants, structs, fixed-point helpers, UnitDeriv, DOPRI Butcher tableau
- `Assets/ReconGridDC/Shaders/Physics.compute` — 8 kernels: ClearForces, AccumulateStructural, AccumulateBending, SeedTrialFromState, BuildTrialState, ComputeSlope, IntegrateY5, ComputeError
- `Assets/ReconGridDC/Physics/MassSpringSolver.cs` — C# orchestrator; outer fixed-dt, inner adaptive substep, per-frame D4 h-adaptation
- `Assets/ReconGridDC/Tests/StructuralForce_GpuOracle_Tests.cs`
- `Assets/ReconGridDC/Tests/BendingForce_GpuOracle_Tests.cs`
- `Assets/ReconGridDC/Tests/Rk45_GpuOracle_Tests.cs`

## Commits

- `350446d` — Task 3: PhysicsCommon.hlsl + Physics.compute (ClearForces + AccumulateStructural) + StructuralForce oracle
- `4cc7f66` — Task 4: AccumulateBending kernel + BendingForce oracle
- `cfe13a8` — Task 5: RK45 kernels (SeedTrialFromState, BuildTrialState, ComputeSlope, IntegrateY5, ComputeError) + MassSpringSolver + Rk45 oracle

## Butcher Tableau Verification

Read back PhysicsCommon.hlsl and cross-checked every entry against the plan's Global Constraints:
- a21=1/5 ✓; a31=3/40, a32=9/40 ✓; a41=44/45, a42=-56/15, a43=32/9 ✓
- a51=19372/6561, a52=-25360/2187, a53=64448/6561, a54=-212/729 ✓
- a61=9017/3168, a62=-355/33, a63=46732/5247, a64=49/176, a65=-5103/18656 ✓
- a71=35/384, a72=0, a73=500/1113, a74=125/192, a75=-2187/6784, a76=11/84 ✓
- b5: 35/384, 0, 500/1113, 125/192, -2187/6784, 11/84, 0 ✓
- b4: 5179/57600, 0, 7571/16695, 393/640, -92097/339200, 187/2100, 1/40 ✓

## Deviations Verified

- **D2** (AccumulateStructural, line 104): `-_Ks*(len-L0)*dir - _Cs*(vi-_YTrialVel[n])` — marked `// DEVIATION D2`
- **D3** (ComputeSlope, line 231): `k.dv = (_ExtForce[p] - Fint) / _Mass[p]` — minus sign correct, marked `// DEVIATION D3`
- **D4** (MassSpringSolver.cs, line 121): `h * 0.9f * Mathf.Pow(Eps / delta, 0.2f)` — ratio is eps/delta, marked `// DEVIATION D4`

## Bending Cross-Product Orderings Verified

- F_ij: `cross(cross(Nik, Nij), Nij)` — Nik × Nij first (correct per plan)
- F_ik: `cross(cross(Nij, Nik), Nik)` — Nij × Nik first (reversed, correct per plan)
- Center: `-(F_ij + F_ik)` momentum-conserving

## KSlope Indexing Verified

BuildTrialState reads `_KSlope[j * _CornerCount + p]` (prior stages j=0..5).
ComputeSlope writes `_KSlope[_Stage * _CornerCount + p]`. Both match plan's `stage*cornerCount+p`.

## Concerns

1. **Fixed-step oracle**: The RK45 oracle uses a single substep of size dt=0.02 in the free-fall test (no inner adaptive loop). For Ks=1e4 spring oscillator the analytic period ~0.063 s so dt=0.02 is ~3 steps/period — the fixed step may accumulate noticeable phase error. The test uses generous tolerances (energy < 5x E0, displacement < 2A) to remain GPU-machine-agnostic. A real stability validation would use smaller h.
2. **HLSL `[unroll]` on array indexed by `_Stage`**: HLSL `[unroll]` on a loop where the count depends on a runtime uniform (`_Stage` / `j<6`) may silently degrade to `[loop]` on some drivers. This is functionally correct but not performance-optimal.
3. **YTrialPos/YTrialVel same-buffer dual binding**: Unity allows binding the same ComputeBuffer to both a `StructuredBuffer<T>` and an `RWStructuredBuffer<T>` name on the same kernel. If a future driver enforces strict read/write aliasing this binding approach would need a ping-pong scheme. Currently correct for Unity 2021.3.
4. **No `.meta` files created**: per instructions. Unity will auto-generate them on import.
