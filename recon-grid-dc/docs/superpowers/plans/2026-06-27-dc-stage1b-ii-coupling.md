# DC Stage 1B-ii — Deformation Coupling (surface follows physics) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Wire the Stage-1B-i physics solver to the Stage-1A DC reconstruction so the rendered surface visibly follows deformation: every frame the mass-spring solver moves the grid corners, `RebuildIsectWorld` re-projects the intersection points along the now-deformed edges, the DC surface is rebuilt, and a gravity + pinned-top demo shows the sphere sagging.

**Architecture:** The voxel-grid corners are the physics particles (1B-i). Intersection points are stored at preprocess as `(cornerA, cornerB, t)` (already in `IsectGpu`); their world position each frame is `lerp(cornerA.pos, cornerB.pos, t)` from the deformed corners. A new `RebuildIsectWorld` kernel writes those world positions into a new `IsectWorld` buffer that `DC_FeaturePoints` consumes (the rest `globalRest`/`normal` stay immutable). `DualContouring.Build` runs `RebuildIsectWorld` as its first step (so the static 1A path is unchanged — corners at rest → `IsectWorld == globalRest`). `ReconGridManager` switches from build-once to a per-frame loop: `solver.Step → DualContouring.Build (incl. RebuildIsectWorld) → DrawProceduralIndirect`.

**Tech Stack:** Unity 2021.3.45f2 (Built-in pipeline), C#, HLSL ComputeShader SM5.0, Unity Test Framework, Unity.Mathematics.

## Global Constraints

- **Project root: `D:\Desktop\Tissue_Simulation\SurgicalSim_DC`**, module `Assets/ReconGridDC/`. Build on Stage 1A + 1B-i (do not break either; the 1A and 1B-i tests must still pass).
- Authoritative design spec: `docs/superpowers/specs/2026-06-27-dc-cutting-design.md` (v4) — this plan implements **§4.6 (geometry tail: UpdateVoxelCorners→RebuildIsectWorld→DC, per-frame)** and the §4.2 note "outer-surface intersection points follow deformation (S1-I2)". It does NOT implement cutting (§5) or haptics (§6).
- **Topology is fixed under deformation (key correctness fact):** corner inside/outside classification (`cornerInside`) and therefore `surfaceEdges` are computed once at preprocess from REST positions and DO NOT change as the lattice deforms (the surface is embedded in the lattice and carried by it). So per-frame DC reuses the same `surfaceEdges` + stencils; only feature-point POSITIONS move. Deformation must NOT recompute corner signs or surface edges.
- **PAPER-SILENT decisions** (mark `// PAPER-SILENT`):
  - Intersection-point **normal is frozen at the rest normal** (design §4.2 note: freeze vs rotate is paper-silent; v1 freezes). `RebuildIsectWorld` updates POSITION only; `_Isect[].normal` stays the rest gradient normal used by the QEF.
  - The intersection parametric `t` is the REST zero-crossing, reused under deformation (affine carry along the deformed edge).
  - Gravity is a constant per-particle external force `(0, −g·m, 0)` set into `ExtForce` once (constant); pinned particles are excluded by the solver. The demo pins the top corner layer.
- **No new deviation ledger entries** (D1–D5 unchanged). This stage is wiring + one kernel; the physics/DC algorithms are already aligned.
- **Single geometry pass per frame** (design §3.7/§4.6): `DualContouring.Build` already does one `triCounter` reset + one `CopyCount`. Adding `RebuildIsectWorld` as its first dispatch keeps that invariant.
- Verification: env cannot run Unity. The `RebuildIsectWorld` kernel gets a GPU-dispatch oracle (lerp correctness). The integrated demo is verified by (a) a GPU integration smoke test (run a few solver+rebuild frames under gravity+pin; assert a surface feature point's Y decreased and all feature points are finite/bounded) and (b) the user's visual check (sphere sags, surface follows, no explosion). Every task ends with a subagent audit.
- TDD; commit on the SurgicalSim_DC git repo (do NOT commit `Library/`); use `git -C "D:/Desktop/Tissue_Simulation/SurgicalSim_DC"`.

---

## File Structure

```
Assets/ReconGridDC/
  Recon/ReconBuffers.cs        // EXTEND: add IsectWorld buffer (float3, isect.Length)
  Shaders/Recon.compute        // ADD: RebuildIsectWorld kernel; MODIFY: DC_FeaturePoints reads _IsectWorld
  Recon/DualContouring.cs      // MODIFY: Build() dispatches RebuildIsectWorld first; bind _IsectWorld + _Isect(RW for rebuild) 
  Demo/ReconGridManager.cs     // MODIFY: per-frame solver.Step → DualContouring.Build → draw; gravity + pin setup
  Tests/
    RebuildIsectWorld_GpuOracle_Tests.cs   // lerp correctness
    DeformCoupling_GpuOracle_Tests.cs      // integration smoke: gravity+pin → surface sags, finite
```

---

### Task 1: RebuildIsectWorld kernel + IsectWorld buffer + DC reads it (static path preserved)

**Files:**
- Modify: `Assets/ReconGridDC/Recon/ReconBuffers.cs` (add `IsectWorld`)
- Modify: `Assets/ReconGridDC/Shaders/Recon.compute` (add `RebuildIsectWorld`; `DC_FeaturePoints` reads `_IsectWorld`)
- Modify: `Assets/ReconGridDC/Recon/DualContouring.cs` (dispatch RebuildIsectWorld first; bind)
- Test: `Assets/ReconGridDC/Tests/RebuildIsectWorld_GpuOracle_Tests.cs`

**Interfaces:**
- Consumes: `_Isect` (cornerA, cornerB, t fields — already populated at preprocess), `_CornerPos` (dynamic particle positions), `voxelIsect*`.
- Produces: `ReconBuffers.IsectWorld` (`ComputeBuffer` float3, length = `Isect.count`); kernel `RebuildIsectWorld` (one thread per isect entry) writing `_IsectWorld[idx] = lerp(_CornerPos[cornerA], _CornerPos[cornerB], t)`; `DC_FeaturePoints` reads `_IsectWorld[off+s]` for `p_i` (normal still from `_Isect[off+s].normal`). `DualContouring.Build` dispatches `RebuildIsectWorld` (G over isect count) before `DC_FeaturePoints`.

- [ ] **Step 1: Add IsectWorld buffer to ReconBuffers**

In `ReconBuffers`: add `public ComputeBuffer IsectWorld;`. In ctor: `IsectWorld = new ComputeBuffer(Math.Max(1, isectLen), 12);` where `isectLen` = the same length used for `Isect` (capture it; if the ctor currently uses `Math.Max(1,g.isect.Length)`, reuse that). In `Dispose`: `IsectWorld?.Dispose();`. (No upload needed — written each frame by the kernel.)

- [ ] **Step 2: Write the failing GPU oracle**

`RebuildIsectWorld_GpuOracle_Tests.cs` (`[Category("GPU")]`, load Recon.compute via `AssetDatabase.FindAssets("Recon t:ComputeShader")`):
```csharp
// Plane x=0.5 in a 1-voxel grid => 4 isect points at x=0.5 (t along each x-edge).
// Move corner (1,0,0) outward (+x) by +1.0; the x-edge (0,0,0)->(1,0,0) isect should move to
// lerp(cornerA, cornerB', t). For phi=x-0.5, t = 0.5 along that edge => new x = lerp(0, 2, 0.5)=1.0.
[Test] public void RebuildIsectWorld_LerpsAlongDeformedEdge()
{
    var cs = LoadReconShader();
    var g = BackgroundGrid.Build(new PlaneX(), new int3(1,1,1), 1f, float3.zero);
    using (var rb = new ReconBuffers(g, 64))
    {
        rb.Upload(g);
        // displace corner (1,0,0): read CornerPos, add +1 in x, write back
        int cidx = ReconGridDC.Core.GridConventions.CornerId(1,0,0, g.dims);
        var pos = new float3[rb.CornerPos.count]; rb.CornerPos.GetData(pos);
        pos[cidx] += new float3(1,0,0); rb.CornerPos.SetData(pos);
        // dispatch RebuildIsectWorld only
        int k = cs.FindKernel("RebuildIsectWorld");
        cs.SetInt("_IsectCount", rb.IsectCount);  // add IsectCount accessor or pass isect.Length
        cs.SetBuffer(k,"_Isect", rb.Isect); cs.SetBuffer(k,"_CornerPos", rb.CornerPos);
        cs.SetBuffer(k,"_IsectWorld", rb.IsectWorld);
        cs.Dispatch(k, Mathf.CeilToInt(rb.IsectCount/64f),1,1);
        var iw = new float3[rb.IsectCount]; rb.IsectWorld.GetData(iw);
        // the isect on the moved x-edge (cornerA=(0,0,0), cornerB=(1,0,0), t=0.5) -> x = lerp(0,2,0.5)=1.0
        bool found=false;
        for(int i=0;i<rb.IsectCount;i++) if (Mathf.Abs(iw[i].x-1.0f)<1e-3f) found=true;
        Assert.IsTrue(found, "rebuilt isect should lerp to x=1.0 on the deformed edge");
    }
}
```
> `PlaneX` and `LoadReconShader` mirror the existing 1A `Qef_GpuOracle_Tests`. Add a `public int IsectCount` accessor to ReconBuffers (= the isect length) if not already exposed.

- [ ] **Step 3: Run, verify fail.** Expected FAIL (kernel/buffer missing).

- [ ] **Step 4: Implement RebuildIsectWorld + wire DC_FeaturePoints**

Add to `Recon.compute`:
```hlsl
#pragma kernel RebuildIsectWorld
int _IsectCount;
RWStructuredBuffer<float3> _IsectWorld;
// _Isect (StructuredBuffer<IsectGpu>) and _CornerPos already declared for DC_FeaturePoints
[numthreads(64,1,1)]
void RebuildIsectWorld(uint3 id:SV_DispatchThreadID){
    uint e=id.x; if(e>=(uint)_IsectCount) return;
    IsectGpu it=_Isect[e];
    _IsectWorld[e] = lerp(_CornerPos[it.cornerA], _CornerPos[it.cornerB], it.t);  // PAPER-SILENT: normal stays rest
}
```
Modify `DC_FeaturePoints`: it currently builds `x0` and `F` from `_Isect[off+s].globalRest`. Change the POSITION source to `_IsectWorld[off+s]` (keep `_Isect[off+s].normal` for `n_i`):
```hlsl
// inside DC_FeaturePoints, where it read e.globalRest:
//   float3 pos_i = _IsectWorld[off+s];   // was _Isect[off+s].globalRest
//   float3 n_i   = _Isect[off+s].normal; // unchanged (rest normal)
```
Add `RWStructuredBuffer<float3> _IsectWorld;` is for the rebuild kernel; in `DC_FeaturePoints` declare it as `StructuredBuffer<float3> _IsectWorld;` (read). (Two kernels, two views of the same buffer — fine; no kernel reads+writes it in one dispatch.)

- [ ] **Step 5: DualContouring.Build dispatches RebuildIsectWorld first + binds**

In `DualContouring`: find the `RebuildIsectWorld` kernel in the ctor; set `_IsectCount`; in `Build`, dispatch `RebuildIsectWorld` (groups = `G(isectCount)`) as the FIRST step (before `DC_FeaturePoints`). In `Bind`, add `cs.SetBuffer(k,"_IsectWorld", rb.IsectWorld);` (bind to all kernels that use it: RebuildIsectWorld + DC_FeaturePoints). Pass `isectCount` to `Build` (add a param or read `rb.IsectCount`). This makes the static 1A path identical (corners at rest → IsectWorld == globalRest), so the 1A `Qef`/`Stitch`/`Normals` oracles and the static sphere demo must still pass.

- [ ] **Step 6: Run, verify pass.** RebuildIsectWorld oracle passes; re-run 1A oracles (`Qef_GpuOracle`, `Stitch_GpuOracle`, `DC_Normals_Sphere_PointOutward`) — must STILL pass (static path unchanged).

- [ ] **Step 7: Commit**
```bash
git -C "D:/Desktop/Tissue_Simulation/SurgicalSim_DC" add -A
git -C "D:/Desktop/Tissue_Simulation/SurgicalSim_DC" -c user.name="xiaoyang.young" -c user.email="bowlandhurles@gmail.com" commit -m "feat(reconDC/1B-ii): RebuildIsectWorld kernel + IsectWorld buffer; DC reads deformed isect (Task 1)"
```

---

### Task 2: ReconGridManager per-frame coupling + gravity/pin demo

**Files:**
- Modify: `Assets/ReconGridDC/Demo/ReconGridManager.cs`

**Interfaces:**
- Consumes: `MassSpringSolver`, `DualContouring`, `ReconBuffers`, `BackgroundGrid` (pinned, mass, cornerPos).
- Produces: per-frame loop in `Update`: `solver.Step(rb, dt)` → `dc.Build(rb, dims, L, qefIters)` (now includes RebuildIsectWorld) → `DrawProceduralIndirect`. `Start` does: preprocess → set pins (top layer) → set gravity into ExtForce → upload → build solver+dc → initial Build. New inspector fields: `gravity` (float, default −9.81), `pinTopLayer` (bool, default true), `ks/cs/kb/cb` overrides, `physicsDt` (default 0.02).

- [ ] **Step 1: Implement the per-frame manager**

Rewrite `ReconGridManager` `Start`/`Update`:
```csharp
using ReconGridDC.Physics;
// fields:
public float gravity = -9.81f; public bool pinTopLayer = true; public float physicsDt = 0.02f;
public float ks=7.5e4f, cs=0.92f, kb=2e4f, cb=0.9f;
MassSpringSolver solver; DualContouring dc;

void Start(){
    var ls = new SphereLevelSet(sphereCenter, sphereRadius);
    var g = BackgroundGrid.Build(ls, dims, L, float3.zero);
    // pin the top corner layer (max j) so gravity sags the rest
    if (pinTopLayer){
        int cny = dims.y; // corners per y = dims.y+1; top index = dims.y
        for (int k=0;k<=dims.z;k++) for (int i=0;i<=dims.x;i++){
            int c = GridConventions.CornerId(i, dims.y, k, dims); g.pinned[c]=1;
        }
    }
    int triCap = 6*g.surfaceEdges.Length + 1024;
    rb = new ReconBuffers(g, triCap);
    rb.Upload(g);
    // gravity into ExtForce (constant): (0, gravity*mass, 0) per corner
    var ext = new float3[g.cornerCount];
    for (int c=0;c<g.cornerCount;c++) ext[c] = new float3(0f, gravity*g.mass[c], 0f);
    rb.ExtForce.SetData(ext);
    solver = new MassSpringSolver(recon){ Ks=ks, Cs=cs, Kb=kb, Cb=cb };
    dc = new DualContouring(recon);
    dc.Build(rb, dims, L, qefIters);     // initial surface
    mat = new Material(surfaceShader);
    mat.SetBuffer("_Tri", rb.Tri); mat.SetBuffer("_VoxelExternalFP", rb.VoxelExternalFP);
    mat.SetBuffer("_ExtNormalF", rb.VoxelExternalFPNormalF);
    float3 ext3 = (float3)dims*L; renderBounds = new Bounds((Vector3)(float3)(ext3*0.5f), (Vector3)(ext3*3f));
}

void Update(){
    if (rb==null) return;
    solver.Step(rb, physicsDt);          // physics moves CornerPos
    dc.Build(rb, dims, L, qefIters);     // RebuildIsectWorld + DC rebuild
    Graphics.DrawProceduralIndirect(mat, renderBounds, MeshTopology.Triangles, rb.IndirectArgs, 0);
}
```
> If `recon` (the ComputeShader) hosts only the Recon kernels and `Physics.compute` is a SEPARATE asset, `MassSpringSolver` needs the Physics ComputeShader, not `recon`. Add a `public ComputeShader physics;` field and assign Physics.compute; pass it to `new MassSpringSolver(physics)`. Update `DemoSceneSetup` to assign both `recon` (Recon.compute) and `physics` (Physics.compute).

- [ ] **Step 2: Update DemoSceneSetup to assign the physics shader**

Add `public ComputeShader physicsShader;` to `DemoSceneSetup`; in `CreateDemoScene`, `mgr.physics = physicsShader;` and `Debug.LogError` if null. (User drags Physics.compute into the new field.)

- [ ] **Step 3: Commit**
```bash
git -C "D:/Desktop/Tissue_Simulation/SurgicalSim_DC" add -A
git -C "D:/Desktop/Tissue_Simulation/SurgicalSim_DC" -c user.name="xiaoyang.young" -c user.email="bowlandhurles@gmail.com" commit -m "feat(reconDC/1B-ii): per-frame physics->rebuild->DC in ReconGridManager + gravity/pin demo (Task 2)"
```

---

### Task 3: Integration smoke oracle + verification gate

**Files:**
- Test: `Assets/ReconGridDC/Tests/DeformCoupling_GpuOracle_Tests.cs`

- [ ] **Step 1: Write the integration smoke oracle** (`[Category("GPU")]`)

Build the sphere grid (small, e.g. dims 12³), pin the top layer, set gravity, run `solver.Step + dc.Build` for ~10 frames, then read `VoxelExternalFP`: assert (a) NO NaN/Inf in any valid feature point; (b) the **minimum Y** among valid feature points has DECREASED vs the initial build (the unpinned bulk sagged); (c) feature points stay within a sane bounded box (no explosion — e.g. within 10× the grid extent of the center). This exercises the full per-frame coupling on GPU.
```csharp
[Test] public void GravityPin_SurfaceSagsAndStaysFinite()
{
    var recon = LoadReconShader();    // Recon.compute
    var physics = LoadPhysicsShader(); // Physics.compute (FindAssets "Physics t:ComputeShader")
    var g = BackgroundGrid.Build(new SphereLevelSet(new float3(1.5f,1.5f,1.5f),1.0f), new int3(12,12,12),0.25f,float3.zero);
    for(int k=0;k<=g.dims.z;k++) for(int i=0;i<=g.dims.x;i++) g.pinned[GridConventions.CornerId(i,g.dims.y,k,g.dims)]=1;
    using(var rb=new ReconBuffers(g, 6*g.surfaceEdges.Length+1024)){
        rb.Upload(g);
        var ext=new float3[g.cornerCount]; for(int c=0;c<g.cornerCount;c++) ext[c]=new float3(0,-9.81f*g.mass[c],0); rb.ExtForce.SetData(ext);
        var dc=new DualContouring(recon); var solver=new MassSpringSolver(physics){Ks=7.5e4f,Cs=0.92f,Kb=2e4f,Cb=0.9f};
        dc.Build(rb, g.dims, 0.25f, 20);
        float minY0 = MinValidY(rb);
        for(int f=0; f<10; f++){ solver.Step(rb,0.02f); dc.Build(rb,g.dims,0.25f,20); }
        var fp=new float4[rb.VoxelCount]; rb.VoxelExternalFP.GetData(fp);
        foreach(var v in fp) if(v.w>0.5f){ Assert.IsFalse(float.IsNaN(v.x)||float.IsInfinity(v.x),"finite"); }
        float minY1 = MinValidY(rb);
        Assert.Less(minY1, minY0 + 1e-4f, "surface should sag (min Y decreases) under gravity");
    }
}
```
(`MinValidY` reads VoxelExternalFP, returns min y among w>0.5.)

- [ ] **Step 2: Run, verify it passes** (after Tasks 1-2). Expected PASS on a GPU.

- [ ] **Step 3: Dispatch 2 subagents to audit Stage 1B-ii** vs this plan + spec §4.6/§4.2: (a) correctness — RebuildIsectWorld lerp + frozen normal, DC reads IsectWorld, topology NOT recomputed under deformation, single geometry pass per frame preserved, static 1A path unchanged; (b) integration/GPU — per-frame dispatch order (solver.Step → RebuildIsectWorld(first in Build) → DC → draw), gravity/pin setup correct, MassSpringSolver gets the Physics shader (not Recon), no buffer left unbound, no 1A/1B-i regression. Skeptic-verify; fix until clean.

- [ ] **Step 4: Final whole-branch review + update ledger + commit.**

---

## Self-Review (writing-plans checklist)

**1. Spec coverage (§4.6 geometry tail + §4.2 deform-follow):** RebuildIsectWorld (T1) ✓; DC reads deformed isect, static path preserved (T1) ✓; per-frame physics→rebuild→DC (T2) ✓; gravity/pin demo for visible deformation (T2) ✓; integration verification (T3) ✓. Deferred: cutting (Stage 2), haptics (Stage 3). Normal-rotation under deformation explicitly PAPER-SILENT-frozen.
**2. Placeholder scan:** no TODO/vague items; kernel + manager + tests have concrete code. The `MinValidY` helper is described (min y among valid FPs) — implement inline.
**3. Type consistency:** `IsectWorld` float3 (12B) read by DC_FeaturePoints, written by RebuildIsectWorld; `_IsectCount` int; `MassSpringSolver(physics)` takes the Physics ComputeShader (distinct from `recon`); `DualContouring.Build(rb, dims, L, qefIters)` signature unchanged (RebuildIsectWorld dispatched inside it using `rb.IsectCount`).

---

## Execution Handoff

Plan saved to `docs/superpowers/plans/2026-06-27-dc-stage1b-ii-coupling.md`. Execute via subagent-driven-development: **C1 = Task 1 (kernel+buffer+DC wiring)** with 2 reviewers (must confirm 1A static path unchanged), **C2 = Tasks 2-3 (manager + integration test)**, then final whole-branch review.
