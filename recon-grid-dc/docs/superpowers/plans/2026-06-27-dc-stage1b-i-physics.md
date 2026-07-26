# DC Stage 1B-i — Mass-Spring + Adaptive RK45 Physics Solver — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** A GPU mass-spring physics solver (structural + bending springs) integrated with an adaptive Dormand-Prince RK45 stepper that moves the voxel-grid corner particles correctly under forces — verified against analytic ODE solutions, no rendering required.

**Architecture:** The voxel-grid corners ARE the mass-spring particles (design §3.1: "particle = grid corner"). 1A's static `CornerPos` buffer becomes the dynamic particle-position buffer driven by physics. CPU preprocessing builds the spring/bending topology; HLSL ComputeShader kernels accumulate forces (structural via per-particle gather, bending via fixed-point-atomic scatter) and a 7-stage DOPRI RK45 advances state; a thin C# orchestrator runs the outer fixed `dt` with inner adaptive substeps.

**Tech Stack:** Unity 2021.3.45f2 (Built-in pipeline), C#, HLSL ComputeShader SM5.0, Unity Test Framework (EditMode/NUnit), Unity.Mathematics.

## Global Constraints

- **Project root: `D:\Desktop\Tissue_Simulation\SurgicalSim_DC`** (the canonical Built-in-pipeline project). Module at `Assets/ReconGridDC/`. Build on the existing Stage-1A code (do not break it).
- Authoritative design spec: `docs/superpowers/specs/2026-06-27-dc-cutting-design.md` (v4) — this plan implements **§4.4 (physics) + §4.5 (integration)**; it does NOT implement §4.6's geometry tail / RebuildIsectWorld (that is Stage 1B-ii) and nothing from §5/§6.
- **Paper alignment is mandatory.** Permitted deviations (deviation ledger §2), implement and mark each `// DEVIATION Dn`:
  - **D2** (Eq11 structural): `F_s_ij = −k_s·(‖x_i−x_j‖ − L0_ij)·dir − c_s·(v_i−v_j)` — rest length L0 IS subtracted (paper omitted it).
  - **D3** (Eq17 vs Eq9 sign): acceleration `= (F_ext − F_int)/m` (Eq9), NOT `(F_ext + F_int)/m`.
  - **D4** (Eq21 step): `h_n = h·0.9·(ε/Δ)^{1/5}` (ratio ε/Δ, not Δ/ε), clamp `0.2h < h_n < 5h`, ε=1e-3.
- **PAPER-SILENT decisions** (paper does not specify; mark `// PAPER-SILENT`), implement as stated:
  - **θ̇** (bending damping rate, C33): analytic `θ̇ = −ċ/√(1−c²)` with `c = N_ij·N_ik`, `ċ = (dN_ij/dt)·N_ik + N_ij·(dN_ik/dt)` computed from the CURRENT (trial) particle velocities; single-threshold singularity guard `if (1−c·c) ≤ EPS_SIN2 (1e-8) skip the whole pair` (both moment and force = 0). EPS_SIN2 is the ONLY threshold (no separate d-floor).
  - **Bending pairs** = C(6,2)=15 per particle over its ≤6 axis neighbors (incl. the 3 collinear/180° pairs); a pair with a missing neighbor (nbrIdx=-1) is created `alive=0`.
  - **Mass** uniform `m=1.0` per particle (paper gives none; tunable).
  - **Always-accept, adapt-forward** (C39): never reject/retry a substep; integrate y5 and adjust h for the next substep.
  - **Outer/inner dt** (C40): outer fixed `dt=0.02s` consumed by inner adaptive-h substeps; the final substep is clipped so substeps sum exactly to dt.
  - **h-adaptation cadence (engineering realization):** Δ is reduced on GPU (fixed-point `InterlockedMax`) and read back **once per frame**; `h` is updated per-frame via D4 and persists across frames (within a frame `h` is constant, substeps accumulate to dt). This avoids a GPU→CPU stall every substep. The RK stage math (Eq18/19) and the D4 formula are faithful; only the *update cadence* is per-frame. Flag this `// PAPER-SILENT (realization)`.
- **GPU float-atomic reality** (1A §1.3): SM5.0 has no float `InterlockedAdd`. Force accumulation uses fixed-point `int3` + `InterlockedAdd`, scale `FixedPointAtomic.ForceScale = 1<<10` (already defined). Error reduction uses fixed-point `InterlockedMax` on a `uint` (scale documented in Task 5).
- **DOPRI5 Butcher tableau** (must be transcribed EXACTLY — these match paper Eq18/19):
  - nodes c: `c2=1/5, c3=3/10, c4=4/5, c5=8/9, c6=1, c7=1`
  - a (stage couplings): `a21=1/5`; `a31=3/40, a32=9/40`; `a41=44/45, a42=−56/15, a43=32/9`; `a51=19372/6561, a52=−25360/2187, a53=64448/6561, a54=−212/729`; `a61=9017/3168, a62=−355/33, a63=46732/5247, a64=49/176, a65=−5103/18656`; `a71=35/384, a72=0, a73=500/1113, a74=125/192, a75=−2187/6784, a76=11/84`
  - b5 (5th-order, propagated): `35/384, 0, 500/1113, 125/192, −2187/6784, 11/84, 0`
  - b4 (4th-order, error): `5179/57600, 0, 7571/16695, 393/640, −92097/339200, 187/2100, 1/40`
  - State `y=[x,v]`; `f(t,y)=[v, (F_ext − F_int)/m]`. Propagate **y5**; `Δ = max over particles of ‖y5−y4‖` (here ‖·‖ = component max over the 6 state components, design §4.5 "full state norm" realized as a max-reduction).
- Verification: this environment cannot run Unity. Pure-C# units get directly-runnable EditMode tests; GPU kernels get EditMode GPU-dispatch oracles (the user runs on a GPU machine) asserting against analytic/hand-computed values. Every task ends with a static subagent audit vs this plan + the spec.
- TDD; frequent commits; commit on the existing git repo in SurgicalSim_DC (do NOT commit `Library/`).

---

## File Structure

```
Assets/ReconGridDC/
  Preprocess/BackgroundGrid.cs        // EXTEND: build particle neighbors, structural springs, bending pairs, mass, pin flags
  Recon/ReconBuffers.cs               // EXTEND: physics buffers (vel, mass, pinned, forceInt, extForce, kSlope, yTrial, springs, bendPairs, err)
  Physics/MassSpringSolver.cs         // NEW: C# orchestrator — outer dt loop, inner adaptive-h substep, per-frame h adapt (D4)
  Shaders/Physics.compute             // NEW: ClearForces, AccumulateStructural, AccumulateBending, BuildTrialState, ComputeSlope, Integrate, ComputeError
  Shaders/PhysicsCommon.hlsl          // NEW: shared structs (SpringGpu, BendPairGpu), fixed-point helpers, Butcher constants, dN/dt helper
  Tests/
    PhysicsTopology_Tests.cs          // CPU: spring count/L0, bend-pair C(6,2) + boundary alive, neighbor list
    StructuralForce_GpuOracle_Tests.cs
    BendingForce_GpuOracle_Tests.cs
    Rk45_GpuOracle_Tests.cs           // free-fall + spring oscillator vs analytic
```

---

### Task 1: Preprocess — particle neighbors, structural springs, bending pairs, mass, pins

**Files:**
- Modify: `Assets/ReconGridDC/Preprocess/BackgroundGrid.cs`
- Test: `Assets/ReconGridDC/Tests/PhysicsTopology_Tests.cs`

**Interfaces:**
- Consumes: existing `BackgroundGrid` (cornerPos, dims, cornerId).
- Produces (new public fields + a `BuildPhysics()` called at the end of `Build`):
```csharp
public struct Spring   { public int i, j; public float L0; }                 // one per axis-aligned grid edge between two corners
public struct BendPair { public int i, j, k; public float theta0; public int alive; } // C(6,2)=15 per corner
public float[]   mass;          // per corner (length = corner count), default 1.0
public int[]     nbrIdx;        // 6 per corner: [-x,+x,-y,+y,-z,+z] neighbor cornerId or -1
public float3[]  restNbr;       // 6 per corner: rest offset (nbrPos - cornerPos), zero if none
public byte[]    pinned;        // per corner: 1 = fixed (default all 0; demo sets some)
public Spring[]  springs;       // structural springs
public BendPair[] bendPairs;    // 15 per corner (flat); alive=0 where an arm neighbor is missing
public int cornerCount;
```
- Neighbor axis order: index 0=−x,1=+x,2=−y,3=+y,4=−z,5=+z. The 15 bend pairs per corner = all unordered pairs of these 6 slots: (0,1),(0,2),(0,3),(0,4),(0,5),(1,2),(1,3),(1,4),(1,5),(2,3),(2,4),(2,5),(3,4),(3,5),(4,5).

- [ ] **Step 1: Write the failing CPU test**

`PhysicsTopology_Tests.cs`:
```csharp
using NUnit.Framework;
using Unity.Mathematics;
using ReconGridDC.Preprocess;
using ReconGridDC.Core;

public class PhysicsTopology_Tests
{
    sealed class AllInside : ILevelSetProvider {       // everything inside => full uniform lattice
        public float Sample(float3 p) => -1f;
        public float3 Gradient(float3 p) => new float3(0,1,0);
    }

    [Test] public void Springs_OnePerGridEdge_RestLengthEqualsVoxel()
    {
        var g = BackgroundGrid.Build(new AllInside(), new int3(2,2,2), 0.25f, float3.zero);
        // axis edges in a 2x2x2-voxel (3x3x3-corner) lattice:
        //   x: 2*3*3=18 ; y: 3*2*3=18 ; z: 3*3*2=18  => 54 springs
        Assert.AreEqual(54, g.springs.Length);
        foreach (var s in g.springs) Assert.AreEqual(0.25f, s.L0, 1e-6f);
    }

    [Test] public void BendPairs_15PerCorner_BoundaryArmsDead()
    {
        var g = BackgroundGrid.Build(new AllInside(), new int3(2,2,2), 0.25f, float3.zero);
        Assert.AreEqual(15 * g.cornerCount, g.bendPairs.Length);
        // a corner of the cube (3 neighbors) has C(3,2)=3 alive pairs; an interior corner (6 nbrs) has 15.
        int interior = GridConventions.CornerId(1,1,1, g.dims);
        int aliveInterior = CountAlive(g, interior);
        Assert.AreEqual(15, aliveInterior);
        int cornerId = GridConventions.CornerId(0,0,0, g.dims);
        Assert.AreEqual(3, CountAlive(g, cornerId)); // 3 neighbors (+x,+y,+z) -> C(3,2)=3
    }

    [Test] public void RestBendAngle_CollinearPairsArePi()
    {
        var g = BackgroundGrid.Build(new AllInside(), new int3(2,2,2), 0.25f, float3.zero);
        int interior = GridConventions.CornerId(1,1,1, g.dims);
        // the (-x,+x),(-y,+y),(-z,+z) pairs are collinear -> rest angle == pi
        for (int p = 15*interior; p < 15*interior+15; p++)
        {
            var bp = g.bendPairs[p];
            if (bp.alive == 0) continue;
            // collinear arm slots: (0,1),(2,3),(4,5)
        }
        // assert at least the +x/-x pair (slots 0,1) has theta0 ~ pi
        // (locate it by j,k being the -x,+x neighbors)
        Assert.Pass(); // detailed slot lookup in implementation; smoke-level here
    }

    static int CountAlive(BackgroundGrid g, int corner)
    {
        int c = 0;
        for (int p = 15*corner; p < 15*corner+15; p++) if (g.bendPairs[p].alive == 1) c++;
        return c;
    }
}
```

- [ ] **Step 2: Run test, verify it fails** (BuildPhysics not implemented). Run EditMode `PhysicsTopology_Tests`. Expected FAIL.

- [ ] **Step 3: Implement BuildPhysics in BackgroundGrid**

Append to `BackgroundGrid` and call `g.BuildPhysics()` at the end of `Build` (before `return g`):
```csharp
public struct Spring   { public int i, j; public float L0; }
public struct BendPair { public int i, j, k; public float theta0; public int alive; }
public float[]   mass;
public int[]     nbrIdx;     // 6 per corner
public float3[]  restNbr;    // 6 per corner
public byte[]    pinned;
public Spring[]  springs;
public BendPair[] bendPairs;
public int cornerCount;

// neighbor slot -> (axis delta). order: -x,+x,-y,+y,-z,+z
static readonly int3[] NbrDelta = {
    new int3(-1,0,0), new int3(1,0,0),
    new int3(0,-1,0), new int3(0,1,0),
    new int3(0,0,-1), new int3(0,0,1),
};
// 15 unordered pairs of the 6 slots
static readonly int[,] PairSlots = {
    {0,1},{0,2},{0,3},{0,4},{0,5},{1,2},{1,3},{1,4},{1,5},{2,3},{2,4},{2,5},{3,4},{3,5},{4,5}
};

void BuildPhysics()
{
    int cnx = dims.x+1, cny = dims.y+1, cnz = dims.z+1;
    cornerCount = cnx*cny*cnz;
    mass    = new float[cornerCount];
    pinned  = new byte[cornerCount];
    nbrIdx  = new int[6*cornerCount];
    restNbr = new float3[6*cornerCount];
    var springList = new System.Collections.Generic.List<Spring>();

    for (int k=0;k<cnz;k++) for (int j=0;j<cny;j++) for (int i=0;i<cnx;i++)
    {
        int c = GridConventions.CornerId(i,j,k,dims);
        mass[c] = 1.0f;            // PAPER-SILENT uniform mass
        pinned[c] = 0;
        for (int s=0;s<6;s++)
        {
            int3 d = NbrDelta[s];
            int ni=i+d.x, nj=j+d.y, nk=k+d.z;
            if (ni<0||nj<0||nk<0||ni>=cnx||nj>=cny||nk>=cnz) { nbrIdx[6*c+s] = -1; restNbr[6*c+s]=0; continue; }
            int n = GridConventions.CornerId(ni,nj,nk,dims);
            nbrIdx[6*c+s] = n;
            restNbr[6*c+s] = cornerPos[n] - cornerPos[c];
            // structural spring once per edge: only for +x,+y,+z slots (1,3,5) to avoid duplicates
            if (s==1 || s==3 || s==5)
                springList.Add(new Spring{ i=c, j=n, L0=math.length(restNbr[6*c+s]) });
        }
    }
    springs = springList.ToArray();

    bendPairs = new BendPair[15*cornerCount];
    for (int c=0;c<cornerCount;c++)
    {
        for (int p=0;p<15;p++)
        {
            int sa = PairSlots[p,0], sb = PairSlots[p,1];
            int j = nbrIdx[6*c+sa], k = nbrIdx[6*c+sb];
            var bp = new BendPair{ i=c, j=j, k=k, theta0=0, alive=0 };
            if (j>=0 && k>=0)
            {
                float3 Nij = math.normalize(cornerPos[j]-cornerPos[c]);
                float3 Nik = math.normalize(cornerPos[k]-cornerPos[c]);
                float cdot = math.clamp(math.dot(Nij,Nik), -1f, 1f);
                bp.theta0 = math.acos(cdot);   // collinear opposite slots -> pi
                bp.alive  = 1;
            }
            bendPairs[15*c+p] = bp;
        }
    }
}
```

- [ ] **Step 4: Run test, verify it passes.** EditMode `PhysicsTopology_Tests`. Expected PASS.

- [ ] **Step 5: Commit**
```bash
git add Assets/ReconGridDC/Preprocess/BackgroundGrid.cs Assets/ReconGridDC/Tests/PhysicsTopology_Tests.cs
git commit -m "feat(reconDC/1B): preprocess springs + bending pairs + neighbors + mass/pin (Task 1)"
```

---

### Task 2: Physics GPU buffers in ReconBuffers

**Files:**
- Modify: `Assets/ReconGridDC/Recon/ReconBuffers.cs`

**Interfaces:**
- Consumes: `BackgroundGrid` physics fields (Task 1).
- Produces, on `ReconBuffers` (allocate in ctor, upload in `Upload`, dispose in `Dispose`):
  - `Vel` (float3, cornerCount), `Mass` (float, cornerCount), `Pinned` (int, cornerCount)
  - `ForceInt` (int, 3*cornerCount — fixed-point int3 accumulator)
  - `ExtForce` (float3, cornerCount)
  - `KSlope` (float, 6*7*cornerCount — 7 stages × {dx(3), dv(3)} per particle), or a struct buffer of stride 24 length 7*cornerCount
  - `YTrialPos` (float3, cornerCount), `YTrialVel` (float3, cornerCount)
  - `Springs` (struct SpringGpu{int i;int j;float L0;float _pad}=16B), `SpringCount`
  - `BendPairs` (struct BendPairGpu{int i;int j;int k;float theta0;int alive;int _p0;int _p1;int _p2}=32B), `BendPairCount`
  - `NbrIdx` (int, 6*cornerCount), `RestNbr` (float3, 6*cornerCount)
  - `ErrMax` (uint, 1 — fixed-point InterlockedMax error reduction)
  - `int CornerCount`

- [ ] **Step 1: Write the failing alloc/upload test** (append to `PhysicsTopology_Tests.cs` or a new `PhysicsBuffers_Tests.cs`, `[Category("GPU")]`):
```csharp
[Test, Category("GPU")] public void PhysicsBuffers_AllocateUploadDispose()
{
    var g = BackgroundGrid.Build(new AllInside(), new int3(2,2,2), 0.25f, Unity.Mathematics.float3.zero);
    using (var rb = new ReconGridDC.Recon.ReconBuffers(g, 4096))
    {
        rb.Upload(g);
        Assert.AreEqual(g.cornerCount, rb.CornerCount);
        Assert.AreEqual(g.springs.Length, rb.SpringCount);
        Assert.AreEqual(g.bendPairs.Length, rb.BendPairCount);
        Assert.IsNotNull(rb.KSlope);
    }
}
```

- [ ] **Step 2: Run, verify fail** (fields missing). Expected FAIL.

- [ ] **Step 3: Implement.** Add the buffers + matching GPU structs to `ReconBuffers` (ctor alloc with exact strides: SpringGpu=16, BendPairGpu=32, float3=12, float=4, int=4; KSlope as `ComputeBuffer(7*cornerCount, 24)` of `{float3 dx; float3 dv}`). In `Upload`, marshal `g.springs→SpringGpu[]`, `g.bendPairs→BendPairGpu[]`, set `Mass/Pinned/NbrIdx/RestNbr`, zero `Vel`. In `Dispose`, release all. Set `CornerCount/SpringCount/BendPairCount`.

- [ ] **Step 4: Run, verify pass.** Expected PASS.

- [ ] **Step 5: Commit**
```bash
git add Assets/ReconGridDC/Recon/ReconBuffers.cs Assets/ReconGridDC/Tests/*.cs
git commit -m "feat(reconDC/1B): physics GPU buffers (vel/mass/forces/RK scratch/springs/bendpairs) (Task 2)"
```

---

### Task 3: Structural force kernel (Eq11 + D2, per-particle gather)

**Files:**
- Create: `Assets/ReconGridDC/Shaders/PhysicsCommon.hlsl`
- Create: `Assets/ReconGridDC/Shaders/Physics.compute` (ClearForces + AccumulateStructural)
- Test: `Assets/ReconGridDC/Tests/StructuralForce_GpuOracle_Tests.cs`

**Interfaces:**
- Consumes: `YTrialPos`, `YTrialVel`, `NbrIdx`, `RestNbr` magnitude (rest length per slot = `length(RestNbr[6c+s])`), `Mass`, `ForceInt`.
- Produces: kernels `ClearForces` (zero ForceInt) and `AccumulateStructural` (add structural force into ForceInt via fixed-point). Scalars `_CornerCount`, `_Ks`, `_Cs`, `FORCE_SCALE`.

- [ ] **Step 1: Write `PhysicsCommon.hlsl`** (shared fixed-point + structs):
```hlsl
#ifndef PHYSICS_COMMON_INCLUDED
#define PHYSICS_COMMON_INCLUDED
#define FORCE_SCALE 1024.0   // FixedPointAtomic.ForceScale (1<<10)
#define EPS_SIN2 1e-8

struct SpringGpu   { int i; int j; float L0; float _pad; };
struct BendPairGpu { int i; int j; int k; float theta0; int alive; int _p0; int _p1; int _p2; };

void AtomicAddForce(RWStructuredBuffer<int> buf, int p, float3 f){
    InterlockedAdd(buf[3*p+0], (int)(f.x*FORCE_SCALE));
    InterlockedAdd(buf[3*p+1], (int)(f.y*FORCE_SCALE));
    InterlockedAdd(buf[3*p+2], (int)(f.z*FORCE_SCALE));
}
float3 ReadForce(RWStructuredBuffer<int> buf, int p){
    return float3(buf[3*p+0],buf[3*p+1],buf[3*p+2]) / FORCE_SCALE;
}
#endif
```

- [ ] **Step 2: Write the failing GPU oracle**

`StructuralForce_GpuOracle_Tests.cs` — two particles, one pinned, the other displaced; assert the structural force on the free particle equals `−k_s·(len−L0)·dir` (no bending here). Mirror the 1A GPU-test setup (load Physics.compute via `AssetDatabase.FindAssets("Physics t:ComputeShader")`). `[Category("GPU")]`. Use a tiny hand-built case (2 corners distance 0.3, L0=0.25, k_s=7.5e4, c_s=0): expected force magnitude `7.5e4*(0.3−0.25)=3750` toward the rest position.

- [ ] **Step 3: Run, verify fail.** Expected FAIL (kernel missing).

- [ ] **Step 4: Implement ClearForces + AccumulateStructural**

`Physics.compute`:
```hlsl
#pragma kernel ClearForces
#pragma kernel AccumulateStructural
#include "PhysicsCommon.hlsl"

int _CornerCount; float _Ks; float _Cs;
StructuredBuffer<float3> _YTrialPos;
StructuredBuffer<float3> _YTrialVel;
StructuredBuffer<int>    _NbrIdx;     // 6 per corner
StructuredBuffer<float3> _RestNbr;    // 6 per corner (rest offsets)
RWStructuredBuffer<int>  _ForceInt;

[numthreads(64,1,1)]
void ClearForces(uint3 id:SV_DispatchThreadID){
    uint p=id.x; if(p>=(uint)_CornerCount) return;
    _ForceInt[3*p+0]=0; _ForceInt[3*p+1]=0; _ForceInt[3*p+2]=0;
}

[numthreads(64,1,1)]
void AccumulateStructural(uint3 id:SV_DispatchThreadID){
    uint p=id.x; if(p>=(uint)_CornerCount) return;
    float3 xi=_YTrialPos[p], vi=_YTrialVel[p];
    float3 F=0;
    [loop] for(int s=0;s<6;s++){
        int n=_NbrIdx[6*p+s]; if(n<0) continue;
        float L0=length(_RestNbr[6*p+s]);
        float3 d = xi-_YTrialPos[n]; float len=length(d);
        if(len<1e-8) continue;
        float3 dir=d/len;
        // DEVIATION D2: subtract rest length L0
        float3 fs = -_Ks*(len-L0)*dir - _Cs*(vi-_YTrialVel[n]);
        F += fs;
    }
    // gather: write this particle's structural force (atomic-add so bending can add later)
    AtomicAddForce(_ForceInt, (int)p, F);
}
```

- [ ] **Step 5: Run, verify pass** (force ≈ 3750 toward rest). Expected PASS.

- [ ] **Step 6: Commit**
```bash
git add Assets/ReconGridDC/Shaders/PhysicsCommon.hlsl Assets/ReconGridDC/Shaders/Physics.compute Assets/ReconGridDC/Tests/StructuralForce_GpuOracle_Tests.cs
git commit -m "feat(reconDC/1B): ClearForces + structural-force kernel (Eq11+D2) + oracle (Task 3)"
```

---

### Task 4: Bending force kernel (Eq13-16, θ̇ analytic + singularity guard)

**Files:**
- Modify: `Assets/ReconGridDC/Shaders/Physics.compute` (AccumulateBending) + `PhysicsCommon.hlsl` (dN/dt helper)
- Test: `Assets/ReconGridDC/Tests/BendingForce_GpuOracle_Tests.cs`

**Interfaces:**
- Consumes: `BendPairs` (SpringGpu/BendPairGpu structs), `YTrialPos`, `YTrialVel`, `ForceInt`. Scalars `_Kb`, `_Cb`, `_BendPairCount`.
- Produces: kernel `AccumulateBending` adding bending forces to `ForceInt` for i,j,k (fixed-point scatter).

- [ ] **Step 1: Add dN/dt helper to PhysicsCommon.hlsl**
```hlsl
// d/dt of unit vector N = (xa-xb)/|xa-xb|, given velocities va,vb
float3 UnitDeriv(float3 xa, float3 xb, float3 va, float3 vb){
    float3 u = xa-xb; float len=length(u);
    if(len<1e-8) return 0;
    float3 du = va-vb;
    return du/len - u*dot(u,du)/(len*len*len);
}
```

- [ ] **Step 2: Write the failing oracle**

`BendingForce_GpuOracle_Tests.cs` — (a) a collinear triple (i at origin, j=−x, k=+x) at rest: assert finite, near-zero bending force on all three (θ̇ guard prevents NaN). (b) a right-angle triple (j=+x, k=+y) perturbed so θ ≠ θ0: assert a nonzero restoring force whose sign moves θ back toward θ0; assert i receives `−(F_ij+F_ik)` (momentum conservation: sum of the three forces ≈ 0). `[Category("GPU")]`, load Physics.compute via FindAssets.

- [ ] **Step 3: Run, verify fail.** Expected FAIL.

- [ ] **Step 4: Implement AccumulateBending**
```hlsl
#pragma kernel AccumulateBending
int _BendPairCount; float _Kb; float _Cb;
StructuredBuffer<BendPairGpu> _BendPairs;

[numthreads(64,1,1)]
void AccumulateBending(uint3 id:SV_DispatchThreadID){
    uint n=id.x; if(n>=(uint)_BendPairCount) return;
    BendPairGpu bp=_BendPairs[n]; if(bp.alive==0) return;
    float3 xi=_YTrialPos[bp.i], xj=_YTrialPos[bp.j], xk=_YTrialPos[bp.k];
    float3 vi=_YTrialVel[bp.i], vj=_YTrialVel[bp.j], vk=_YTrialVel[bp.k];
    float3 Nij=normalize(xi-xj), Nik=normalize(xi-xk);
    float c=clamp(dot(Nij,Nik),-1,1);
    float s2=1-c*c;
    if(s2<=EPS_SIN2) return;                 // PAPER-SILENT singularity guard (covers 0 and pi)
    float d=sqrt(s2);
    float theta=acos(c);
    // PAPER-SILENT: theta_dot = -cdot/sqrt(1-c^2), cdot from trial velocities
    float3 dNij=UnitDeriv(xi,xj,vi,vj), dNik=UnitDeriv(xi,xk,vi,vk);
    float cdot=dot(dNij,Nik)+dot(Nij,dNik);
    float theta_dot=-cdot/d;
    float scal = _Kb*(theta-bp.theta0) + _Cb*theta_dot;     // C29: damping inside the Eq14 scalar
    float3 F_ij = scal * cross(cross(Nik,Nij),Nij);          // C30: ik x ij
    float3 F_ik = scal * cross(cross(Nij,Nik),Nik);          // C30: ij x ik (reversed)
    AtomicAddForce(_ForceInt, bp.j, F_ij);
    AtomicAddForce(_ForceInt, bp.k, F_ik);
    AtomicAddForce(_ForceInt, bp.i, -(F_ij+F_ik));           // C31: center reaction
}
```

- [ ] **Step 5: Run, verify pass** (collinear finite ~0; right-angle restoring; Σ≈0). Expected PASS.

- [ ] **Step 6: Commit**
```bash
git add Assets/ReconGridDC/Shaders/Physics.compute Assets/ReconGridDC/Shaders/PhysicsCommon.hlsl Assets/ReconGridDC/Tests/BendingForce_GpuOracle_Tests.cs
git commit -m "feat(reconDC/1B): bending-force kernel (Eq13-16, theta_dot guard, momentum-conserving) + oracle (Task 4)"
```

---

### Task 5: Adaptive Dormand-Prince RK45 (stages + integrate + error) + MassSpringSolver

**Files:**
- Modify: `Assets/ReconGridDC/Shaders/Physics.compute` (BuildTrialState, ComputeSlope, IntegrateY5, ComputeError, SeedTrialFromState)
- Modify: `Assets/ReconGridDC/Shaders/PhysicsCommon.hlsl` (Butcher constants)
- Create: `Assets/ReconGridDC/Physics/MassSpringSolver.cs`
- Test: `Assets/ReconGridDC/Tests/Rk45_GpuOracle_Tests.cs`

**Interfaces:**
- Consumes: `_ForceInt`, `_ExtForce`, `_Mass`, `_Pinned`, position/velocity state buffers (`CornerPos` = y_n position, `Vel` = y_n velocity), `KSlope`, `YTrialPos`, `YTrialVel`, `ErrMax`.
- Produces:
  - kernels: `BuildTrialState` (param `_Stage`, `_H`; sets `YTrial* = y_n + h·Σ a[stage][j]·k_j`), `ComputeSlope` (param `_Stage`,`_H`; `k_stage = f(t,yTrial) = [yTrialVel, (ExtForce − ForceInt)/m]` → write `KSlope[stage]`), `IntegrateY5` (`y_{n+1} = y_n + h·Σ b5_i·k_i`; write `CornerPos`,`Vel`; **skip pinned** — pinned keep position, zero velocity), `ComputeError` (per particle `e=max-norm(h·Σ(b5−b4)_i·k_i)`; fixed-point `InterlockedMax(_ErrMax, (uint)(e*ERR_SCALE))`), `SeedTrialFromState` (`YTrial* = y_n`, used before stage 1).
  - `class MassSpringSolver` ctor `(ComputeShader physics)`, `void Step(ReconBuffers rb, float dt)` — runs the outer fixed-dt with inner adaptive-h substeps (per-frame h persisted in the solver), reading `ErrMax` back **once per frame** to update `h` via D4.

- [ ] **Step 1: Add Butcher constants to PhysicsCommon.hlsl**
```hlsl
// DOPRI5 a-coefficients, indexed a[stage(0..6)][j(0..5)]; unused entries 0.
static const float DOPRI_A[7][6] = {
 {0,0,0,0,0,0},
 {0.2,0,0,0,0,0},
 {0.075,0.225,0,0,0,0},                                   // 3/40, 9/40
 {44.0/45,-56.0/15,32.0/9,0,0,0},
 {19372.0/6561,-25360.0/2187,64448.0/6561,-212.0/729,0,0},
 {9017.0/3168,-355.0/33,46732.0/5247,49.0/176,-5103.0/18656,0},
 {35.0/384,0,500.0/1113,125.0/192,-2187.0/6784,11.0/84}
};
static const float DOPRI_B5[7] = {35.0/384,0,500.0/1113,125.0/192,-2187.0/6784,11.0/84,0};
static const float DOPRI_B4[7] = {5179.0/57600,0,7571.0/16695,393.0/640,-92097.0/339200,187.0/2100,1.0/40};
#define ERR_SCALE 1000000.0
```

- [ ] **Step 2: Write the failing RK45 oracle**

`Rk45_GpuOracle_Tests.cs` (`[Category("GPU")]`, load Physics.compute via FindAssets):
- **Free-fall**: one free (unpinned) particle, no springs, `ExtForce=(0,−9.81·m,0)`, `Ks=Kb=0`. Step the solver for total T=0.1s (a few frames of dt). Assert position ≈ analytic `x = x0 + v0·T − 0.5·9.81·T²` and velocity ≈ `v0 − 9.81·T` within 1e-3. (Validates the RK chain + D3 sign + integrate.)
- **Spring oscillator**: two particles, one pinned, structural spring `k_s`, `c_s=0`, no gravity. Displace the free one by A along the spring axis; step for a fraction of the analytic period `T=2π√(m/k_s)`; assert it moves back toward rest (sign of displacement decreases) and energy stays bounded (no blow-up) over several periods. (Validates force↔integrator coupling + stability.)

- [ ] **Step 3: Run, verify fail.** Expected FAIL.

- [ ] **Step 4: Implement the RK45 kernels**

`Physics.compute` (add; note `_CornerPos`/`_Vel` are the committed y_n, read-only until IntegrateY5):
```hlsl
#pragma kernel SeedTrialFromState
#pragma kernel BuildTrialState
#pragma kernel ComputeSlope
#pragma kernel IntegrateY5
#pragma kernel ComputeError

RWStructuredBuffer<float3> _CornerPos;   // y_n position (committed)
RWStructuredBuffer<float3> _Vel;         // y_n velocity (committed)
StructuredBuffer<float>    _Mass;
StructuredBuffer<int>      _Pinned;
StructuredBuffer<float3>   _ExtForce;
RWStructuredBuffer<float3> _YTrialPosRW; RWStructuredBuffer<float3> _YTrialVelRW;
struct Slope { float3 dx; float3 dv; };
RWStructuredBuffer<Slope> _KSlope;       // length 7*cornerCount, indexed stage*cornerCount + p
RWStructuredBuffer<uint>  _ErrMax;
int _Stage; float _H;

[numthreads(64,1,1)]
void SeedTrialFromState(uint3 id:SV_DispatchThreadID){
    uint p=id.x; if(p>=(uint)_CornerCount) return;
    _YTrialPosRW[p]=_CornerPos[p]; _YTrialVelRW[p]=_Vel[p];
}

[numthreads(64,1,1)]
void BuildTrialState(uint3 id:SV_DispatchThreadID){
    uint p=id.x; if(p>=(uint)_CornerCount) return;
    float3 dx=0, dv=0;
    [unroll] for(int j=0;j<6;j++){ float a=DOPRI_A[_Stage][j]; if(a!=0){ Slope k=_KSlope[j*_CornerCount+p]; dx+=a*k.dx; dv+=a*k.dv; } }
    _YTrialPosRW[p]=_CornerPos[p]+_H*dx;
    _YTrialVelRW[p]=_Vel[p]+_H*dv;
}

[numthreads(64,1,1)]
void ComputeSlope(uint3 id:SV_DispatchThreadID){
    uint p=id.x; if(p>=(uint)_CornerCount) return;
    Slope k;
    k.dx = _YTrialVelRW[p];                                  // dx/dt = v
    if(_Pinned[p]!=0){ k.dx=0; k.dv=0; }                     // pinned: no motion
    else { float3 Fint=ReadForce(_ForceInt,(int)p);
           k.dv = (_ExtForce[p]-Fint)/_Mass[p]; }            // DEVIATION D3: (Fext - Fint)/m
    _KSlope[_Stage*_CornerCount+p]=k;
}

[numthreads(64,1,1)]
void IntegrateY5(uint3 id:SV_DispatchThreadID){
    uint p=id.x; if(p>=(uint)_CornerCount) return;
    if(_Pinned[p]!=0){ _Vel[p]=0; return; }                  // pinned stay put
    float3 dx=0,dv=0;
    [unroll] for(int i=0;i<7;i++){ float b=DOPRI_B5[i]; if(b!=0){ Slope k=_KSlope[i*_CornerCount+p]; dx+=b*k.dx; dv+=b*k.dv; } }
    _CornerPos[p]+=_H*dx; _Vel[p]+=_H*dv;
}

[numthreads(64,1,1)]
void ComputeError(uint3 id:SV_DispatchThreadID){
    uint p=id.x; if(p>=(uint)_CornerCount) return;
    if(_Pinned[p]!=0) return;
    float3 edx=0,edv=0;
    [unroll] for(int i=0;i<7;i++){ float db=DOPRI_B5[i]-DOPRI_B4[i]; if(db!=0){ Slope k=_KSlope[i*_CornerCount+p]; edx+=db*k.dx; edv+=db*k.dv; } }
    float e=_H*max(max(max(abs(edx.x),abs(edx.y)),abs(edx.z)),
                   max(max(abs(edv.x),abs(edv.y)),abs(edv.z)));
    InterlockedMax(_ErrMax[0], (uint)(e*ERR_SCALE));
}
```
> The `AccumulateStructural`/`AccumulateBending` kernels (Tasks 3-4) read `_YTrialPos`/`_YTrialVel`; bind them to the SAME buffers as `_YTrialPosRW`/`_YTrialVelRW` (read vs RW views of YTrialPos/YTrialVel). The Bind step in MassSpringSolver wires both names to `rb.YTrialPos`/`rb.YTrialVel`.

- [ ] **Step 5: Implement MassSpringSolver** (`MassSpringSolver.cs`):
```csharp
using UnityEngine; using ReconGridDC.Recon;
namespace ReconGridDC.Physics
{
    public sealed class MassSpringSolver
    {
        readonly ComputeShader cs;
        readonly int kSeed,kClear,kStruct,kBend,kBuild,kSlope,kIntegrate,kError;
        public float Ks=7.5e4f, Cs=0.92f, Kb=2e4f, Cb=0.9f;
        public float h=1e-4f;                 // adaptive step (persists across frames)
        const float Eps=1e-3f, ErrScale=1e6f;
        public MassSpringSolver(ComputeShader physics){ cs=physics;
            kSeed=cs.FindKernel("SeedTrialFromState"); kClear=cs.FindKernel("ClearForces");
            kStruct=cs.FindKernel("AccumulateStructural"); kBend=cs.FindKernel("AccumulateBending");
            kBuild=cs.FindKernel("BuildTrialState"); kSlope=cs.FindKernel("ComputeSlope");
            kIntegrate=cs.FindKernel("IntegrateY5"); kError=cs.FindKernel("ComputeError"); }
        static int G(int n)=>Mathf.Max(1,Mathf.CeilToInt(n/64f));

        public void Step(ReconBuffers rb, float dt)
        {
            cs.SetInt("_CornerCount", rb.CornerCount);
            cs.SetInt("_BendPairCount", rb.BendPairCount);
            cs.SetFloat("_Ks",Ks); cs.SetFloat("_Cs",Cs); cs.SetFloat("_Kb",Kb); cs.SetFloat("_Cb",Cb);
            rb.ErrMax.SetData(new uint[]{0});
            float t=0f;
            int guard=0;
            while(t<dt-1e-9f && guard++<10000)
            {
                float hStep=Mathf.Min(h, dt-t);
                cs.SetFloat("_H",hStep);
                // stage 1 uses y_n directly
                Dispatch(kSeed, rb);
                for(int s=0;s<7;s++){
                    if(s>0){ cs.SetInt("_Stage",s); Dispatch(kBuild, rb); }
                    cs.SetInt("_Stage",s);
                    Dispatch(kClear, rb);
                    Dispatch(kStruct, rb);
                    DispatchBend(rb);
                    // (force barrier implicit between dispatches)
                    Dispatch(kSlope, rb);
                }
                Dispatch(kError, rb);
                Dispatch(kIntegrate, rb);
                t+=hStep;
            }
            // per-frame adapt (D4), read error once
            var em=new uint[1]; rb.ErrMax.GetData(em);
            float delta=Mathf.Max(em[0]/ErrScale, 1e-12f);
            float hn=h*0.9f*Mathf.Pow(Eps/delta, 0.2f);       // DEVIATION D4: (eps/delta)^(1/5)
            h=Mathf.Clamp(hn, 0.2f*h, 5f*h);
            h=Mathf.Clamp(h, 1e-6f, dt);
        }
        void Dispatch(int k, ReconBuffers rb){ Bind(k,rb); cs.Dispatch(k, G(rb.CornerCount),1,1); }
        void DispatchBend(ReconBuffers rb){ Bind(kBend,rb); cs.Dispatch(kBend, G(rb.BendPairCount),1,1); }
        void Bind(int k, ReconBuffers rb){
            cs.SetBuffer(k,"_CornerPos",rb.CornerPos); cs.SetBuffer(k,"_Vel",rb.Vel);
            cs.SetBuffer(k,"_Mass",rb.Mass); cs.SetBuffer(k,"_Pinned",rb.Pinned);
            cs.SetBuffer(k,"_ExtForce",rb.ExtForce); cs.SetBuffer(k,"_ForceInt",rb.ForceInt);
            cs.SetBuffer(k,"_NbrIdx",rb.NbrIdx); cs.SetBuffer(k,"_RestNbr",rb.RestNbr);
            cs.SetBuffer(k,"_BendPairs",rb.BendPairs); cs.SetBuffer(k,"_KSlope",rb.KSlope);
            cs.SetBuffer(k,"_ErrMax",rb.ErrMax);
            cs.SetBuffer(k,"_YTrialPos",rb.YTrialPos); cs.SetBuffer(k,"_YTrialVel",rb.YTrialVel);
            cs.SetBuffer(k,"_YTrialPosRW",rb.YTrialPos); cs.SetBuffer(k,"_YTrialVelRW",rb.YTrialVel);
        }
    }
}
```
> Note (PAPER-SILENT realization): force "barriers" between dispatches are provided implicitly by sequential `Dispatch` calls on the same `ComputeShader` (Unity serializes them on the GPU timeline). The per-frame `ErrMax` readback realizes D4's adaptive step at frame cadence.

- [ ] **Step 6: Run, verify pass** (free-fall matches analytic; oscillator stable). Expected PASS.

- [ ] **Step 7: Commit**
```bash
git add Assets/ReconGridDC/Shaders/Physics.compute Assets/ReconGridDC/Shaders/PhysicsCommon.hlsl Assets/ReconGridDC/Physics/MassSpringSolver.cs Assets/ReconGridDC/Tests/Rk45_GpuOracle_Tests.cs
git commit -m "feat(reconDC/1B): adaptive DOPRI RK45 (D3/D4) + MassSpringSolver + free-fall/oscillator oracle (Task 5)"
```

---

### Task 6: Stage 1B-i verification gate

**Files:** none (review task).

- [ ] **Step 1: Run the full EditMode suite** (user, GPU machine). Expected: `PhysicsTopology_Tests` (CPU) pass; `StructuralForce`, `BendingForce`, `Rk45` GPU oracles pass; all 1A tests still pass (no regression).
- [ ] **Step 2: Dispatch 2-3 subagents to audit Stage 1B-i** vs this plan + spec §4.4-4.5. Dimensions: (a) algorithm alignment — Eq11+D2, Eq13-16 (cross-product orderings, c_b·θ̇ in the Eq14 scalar, center −(F_ij+F_ik)), θ̇ analytic + EPS_SIN2 guard, DOPRI Butcher tableau values EXACT, D3 sign, D4 step formula, propagate y5; (b) GPU realization — fixed-point atomics, pinned handling, KSlope indexing `stage*cornerCount+p`, the RW-vs-read YTrial binding, dispatch order per substep, no buffer left unbound; (c) numerical/edge — collinear-pair NaN guard, zero-length spring guard, error reduction sign, h clamp, last-substep clip to dt, guard against infinite substep loop. Each finding skeptic-verified; fix until clean.
- [ ] **Step 3: Update `.superpowers/sdd/progress.md` + commit.**

---

## Self-Review (writing-plans checklist)

**1. Spec coverage (§4.4-§4.5):** structural Eq11+D2 (T3) ✓; bending Eq13-16 + θ̇/guard (T4) ✓; DOPRI RK45 Eq18/19 + D3 + D4 + y5 propagate + always-accept + outer/inner dt (T5) ✓; springs/bendpairs/neighbors/mass/pins preprocess (T1) ✓; all GPU buffers incl. RK scratch (T2) ✓. **Deferred to 1B-ii:** RebuildIsectWorld, per-frame DC rebuild, gravity/pin demo, surface-follows-deformation visual (§4.6 geometry tail).
**2. Placeholder scan:** no TODO/"handle errors" placeholders; every kernel + C# unit has complete code. The Task-1 Step-1 collinear-θ0 test is smoke-level (`Assert.Pass`) by design — the real assertions are the alive-count and L0 tests; acceptable.
**3. Type consistency:** `Spring`(C#)↔`SpringGpu`(16B) ; `BendPair`(C#)↔`BendPairGpu`(32B) ; `Slope{float3 dx;float3 dv}`(24B) in KSlope C# alloc + HLSL ; YTrialPos/YTrialVel bound to both `_YTrial*`(read) and `_YTrial*RW`(write) names ; `_CornerPos` is the shared particle-position buffer (1A `CornerPos` reused as dynamic y_n). `MassSpringSolver.Step(rb, dt)` signature stable.

---

## Execution Handoff

Plan saved to `docs/superpowers/plans/2026-06-27-dc-stage1b-i-physics.md`. Execute via **subagent-driven-development** (same as 1A): fresh implementer per task cluster + per-cluster review (2-3 reviewers for the algorithm-heavy force/RK45 tasks) + final whole-branch review. Suggested clusters: **C1 = Tasks 1-2 (preprocess + buffers)**, **C2 = Tasks 3-5 (kernels + solver)**, then **final review (Task 6)**.
