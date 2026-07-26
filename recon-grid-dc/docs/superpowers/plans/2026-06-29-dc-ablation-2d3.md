# 2D-3 Ablation Cutting — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the paper's §2.1.4 volume-loss "ablation" cut — a swept-capsule probe deletes in-range tissue particles and the Dual-Contouring surface is **redrawn around the resulting cavity** — as a self-contained module that does not regress the (paper-aligned) blade cut.

**Architecture (Approach B — DC re-contour, INLINE RECOMPUTE):** A new `AblationTool` (swept capsule) + `AblationDetector` orchestrator drive a new `Ablation.compute` with three new kernels (`RecomputeOccupancy`, `DetectAblation`, `SeverDeleted` + a tiny `FreezeDeleted` pass). Deletion is expressed **only** through a per-corner `_Deleted` flag; the GPU copy of baked `cornerInside` (`_CornerInsideRT`) is **read-only / never flipped**. The cavity wall is produced by the **regular outer DC surface**: the receded-boundary capsule crossings are computed **INLINE inside `DC_FeaturePoints`** under a new `_AblationMode` uniform (the union of the filtered baked window + inline capsule-SDF roots on surviving↔deleted edges), and the cavity boundary is stitched by a **runtime `_AblationMode` gate in `DC_Stitch`**. There is **NO** separate `GenerateFrontierIsect` kernel and **NO** dynamic-isect buffer (the adversarial re-audit proved a flat append buffer is read by no kernel and fails on baked-`cnt==0` interior voxels). The blade's `_CutFP`/`BuildCutTriangles` and `EmitCutPoint` are **NOT used**. A `cuttingMode` toggle in `ReconGridManager` selects Blade *or* Ablation, **fixed at build**.

**Tech Stack:** Unity 2021.3 Built-in pipeline, HLSL compute shaders, C#, NUnit GPU oracle tests (`[Category("GPU")]`).

## Global Constraints

- **Strict paper alignment** (math + architecture): §2.1.4 (`paper_pdf_text.txt` p5-6, lines 478-499) + ref [22] Berndt (`ref22_berndt_text.txt`: capsules ref22:361, re-extract ref22:316-318). Every choice traces to a paper line; paper-silent choices are declared. The capsule **shape** + "within a certain range" radius are **paper-silent, grounded in [22]** (the paper does not prescribe a capsule; [22] does) — phrase citations that way, not as if the paper prescribes a capsule.
- **Spec is authoritative:** `docs/superpowers/specs/2026-06-29-dc-ablation-2d3-design.md` (Approach B + Inline-recompute revision). Follow it exactly.
- **CRITICAL INVARIANT — byte-identical when inactive (TOP GATE):** with `cuttingMode != Ablation` (blade or zero-cut, `_AblationMode==0`), the pipeline output (surface, normals, occupancy) MUST be byte-identical to today. `RecomputeOccupancy` reproduces the uploaded static `_VoxelOccupied` OR on frame 0; the static branch of `DC_FeaturePoints` runs unchanged; the static `_SurfaceEdges` branch in `DC_Stitch` is a **verbatim early-out** when `_AblationMode==0`. Test gate on Tasks 1, 4, 6, 8 — including a full `dc.Build(ablationMode:false)` byte-identical `_Tri`+counter assertion (**order-INSENSITIVE**, see below) and a frame-0 predicate-equivalence oracle.
- **ORDER-INSENSITIVE `_Tri` oracle (re-audit bug #1):** `DC_Stitch` appends via `InterlockedAdd(_TriCounter,6)` (Recon.compute:284), so the triangle index ORDER is nondeterministic across dispatches — an index-by-index `_Tri` compare false-fails even baseline-vs-baseline. Every byte-identity `_Tri` gate (Tasks 4, 5, 6, 8) MUST: assert `TriCounter[0]` equal first, then compare the **SET** of triangles — canonicalize each tri (rotate so the min vertex is first), sort the tri list, assert equal lists — OR a **commutative checksum** (sum/XOR of per-tri hashes). The byte-identity baseline is captured **IN-TEST** (run the static/today path, snapshot the `_Tri` set + counter, then run `dc.Build(ablationMode:false)` and compare) — there is **no committed baseline artifact**.
- **Canonical occupancy / post-delete formula (must-fix #1, used identically everywhere):**
  > surviving-inside(c) := `_CornerInsideRT[c] != 0 && _Deleted[c] == 0`
  > deleted-inside(c)   := `_CornerInsideRT[c] != 0 && _Deleted[c] == 1`
  > `_VoxelOccupied[v]` = OR over the 8 corners of surviving-inside
  > receded-boundary edge := exactly one endpoint surviving-inside AND the other endpoint deleted-inside
- **DEFORMED frame for ablation (re-audit bug #3, replaces "mirror DetectCut rest discipline"):** `DetectAblation` tests the capsule against the **DEFORMED** `_CornerPos`, and the inline crossing uses the **DEFORMED** edge. Rationale: (a) the cavity wall is a deformed-space DC surface (the QEF over `_IsectWorld` is deformed); (b) the `deleted ⇔ in-capsule` invariant (needed for a single capsule-SDF sign change ⇒ valid bisection) only holds if detection + crossing share ONE frame; (c) the blade's rest-frame discipline (CutDetector.cs:73-78) exists to dodge a thin-swept-PLANE vs sagging-surface miss — a capsule is a RADIUS volume, so that miss does not apply. **Do NOT** reconstruct rest; **do NOT** mirror DetectCut's rest discipline for ablation.
- **`CorId` in Recon.compute (re-audit bug #4 — required to compile):** the `DC_Stitch` ablation branch + the `DC_FeaturePoints` inline branch map a grid-corner coord → global id, but `CorId()` lives only in `Cutting.compute:160`; `Recon.compute` has only `VoxId()`. Add to `Recon.compute` (Tasks 4 and 6 Files/Implement): `int CorId(int3 c){ return c.x + (_Dims.x+1)*(c.y + (_Dims.y+1)*c.z); }` (mirror Cutting.compute:160).
- **Reuse verbatim** (do NOT reimplement): `RebuildIsectWorld` (unchanged — NO appends into `_IsectWorld`); `DualContouring.Build` outer-surface block (gains the `ablationMode` arg + capsule binds; the static path is unchanged); normals (`DC_ClearNormals`/`DC_NormalsScatter`/`DC_NormalsNormalize`); the occupancy gates (`StitchVertId` Recon.compute:221, the `cnt==0` early-out on the static `DC_FeaturePoints` path); the `SeverLinks` *write contract* (`NbrIdx=-1`, `BendPair.alive=0`) — replicated in `SeverDeleted` but gated on `_Deleted`, not `EdgeIsCut`; Physics force-skip on `_CornerActive==0`/`NbrIdx<0`; `ComputeParticleRot` (with the corrected identity-fallback behavior — see below); `AblationTool` mirrors `CuttingTool.Advance`.
- **NOT reused verbatim (re-audit):** `DC_FeaturePoints` gains an `_AblationMode` branch (static branch byte-identical; ablation branch does inline filtered-window + capsule-crossing recompute). `DC_Stitch` gains an `_AblationMode` runtime gate (static branch byte-identical). Fix every doc claim that calls either "reused verbatim".
- **NOT used for ablation (must-fix #4, #7):** `_CutFP`/`ComputeComponentFP`/`AccumulateCutFP`/`BuildCutTriangles` (the per-component cut-wall centroid path) and `EmitCutPoint` (hardcodes ±D/2 + needs `_NCut`/`_D`). The cavity wall is the re-contoured outer DC surface (Approach B). `Conn4096`/`ConnectivityLUT` are not needed for the cavity (no per-component split). **NO dynamic-isect buffer / `GenerateFrontierIsect` kernel** (superseded by inline recompute — Option A).
- **`ComputeParticleRot` actual behavior (must-fix #6 + bug #5):** it EXCLUDES severed (`NbrIdx<0`) + inactive (`_Active[nbr]==0`) neighbors via `continue` (Physics.compute:428-429), so a deleted neighbor is omitted from `F`; a corner that loses **BOTH** arms of an axis hits the **M3 zero-diagonal per-axis gate** (Physics.compute:448-452, `min(mx,my,mz) < axisEps`) and falls back to **identity R**. Deleting **ONE** arm leaves the opposite arm → that axis `m·>0` → the **polar path runs**, NOT M3. There is **NO [22] Fig4(d) cross-product fallback** in the code. Declare identity-fallback as paper-silent-acceptable.
- **`dc.Build` signature (re-audit bug #2):** the real signature is `Build(ReconBuffers rb, int3 dims, float L, int qefIters, float cutFPInterp=1f)` (DualContouring.cs:127). Add **`bool ablationMode=false`** as the LAST arg → `Build(ReconBuffers rb, int3 dims, float L, int qefIters, float cutFPInterp=1f, bool ablationMode=false)`. **Concrete wiring (the ONE chosen):** when `ablationMode==true`, `Build` sets `_AblationMode=1`, binds the capsule context (`_CapA`/`_CapB`/`_CapR` from a capsule passed in, or read from `rb`) + `_CornerInsideRT`/`_Deleted` to `kFP` AND `kStitch`, and sets `_GridEdgeCount`; when `false` (default), `Build` does NONE of that → today's path, byte-identical. ALL test/call sites use the real arg order `(rb, dims, L, qefIters)` + named `ablationMode:` — do NOT invent a `(dims, L, qefIters)`-only ordering.
- **Decisions (spec):** separate `cuttingMode` (Blade|Ablation), mode-fixed-at-build; capsule radius `r` a tunable public field (default 0.4); auto-swept probe with `Valid=true` always (a stationary probe still ablates); **freeze** deleted corners via a FINAL `FreezeDeleted` `_CornerActive=0` pass (not a deletion signal); **instant** deletion (paper:497) → the whole pass RE-RUNS every frame and `_Deleted` is reset each frame.
- **Test-helper convention (re-audit minor):** the existing GPU tests each define their OWN inline `ILevelSetProvider` + a `LoadShader`-via-`AssetDatabase` helper (e.g. `Qef_GpuOracle_Tests` defines `PlaneX` + `AssetDatabase.FindAssets("Recon t:ComputeShader")`). There is NO shared `BuildSphere`/`SphereCenter`/`LoadAblationShader`. Each new test defines its helpers **inline per the existing convention** (a sphere `ILevelSetProvider`, an `AssetDatabase` loader for `Ablation.compute`/`Recon.compute`, a local sphere-center constant) — OR add ONE shared test-helper file as an explicit deliverable (Task 2). Do not reference nonexistent shared helpers.
- **Per-task review (user standing requirement):** every task ends with a multi-subagent code review verifying the new code COMPLETELY matches (a) the paper's algorithm architecture + math derivation (§2.1.4 + [22]) and (b) this plan + the design spec — before moving to the next task.
- **No ±D/2, no `_CutFP` for ablation:** the cavity wall is one-sided and is produced by the regular DC contour, not a slit-opener.

---

### Task 1: Dynamic occupancy — `_CornerInsideRT` + `_Deleted` + `RecomputeOccupancy` (byte-identical foundation)

The load-bearing change: `_VoxelOccupied` becomes per-frame recomputed from the surviving-inside state. Do this FIRST and prove it is byte-identical to the static value when nothing is deleted — this de-risks R1 before any deletion logic exists.

**Files:**
- Create: `Assets/ReconGridDC/Shaders/Ablation.compute` (`RecomputeOccupancy` kernel)
- Modify: `Assets/ReconGridDC/Recon/ReconBuffers.cs` (add `CornerInsideRT` + `Deleted` buffers; init `CornerInsideRT` from `g.cornerInside`, `Deleted` to 0; dispose; keep `VoxelOccupied` alloc + initial upload, it stays the frame-0 source)
- Create: `Assets/ReconGridDC/Cutting/AblationDetector.cs` (minimal: load `Ablation.compute`, a `RecomputeOccupancy(rb, dims)` dispatch)
- Test: `Assets/ReconGridDC/Tests/Ablation_GpuOracle_Tests.cs`

**Interfaces:**
- Produces: `RWStructuredBuffer<uint> _VoxelOccupied` (now writable); `RWStructuredBuffer<int> _CornerInsideRT` (read-only RT copy of baked `cornerInside`, 1=inside, 0=outside); `RWStructuredBuffer<int> _Deleted` (per-corner, init 0); kernel `RecomputeOccupancy` [1/voxel]: `_VoxelOccupied[v] = OR_{c in 8 corners}(_CornerInsideRT[c]!=0 && _Deleted[c]==0)`. `AblationDetector.RecomputeOccupancy(ReconBuffers rb, Vector3Int dims)`.
- Consumes: `rb.VoxelCorner` (per-voxel 8 corner ids), `g.cornerInside` (baked).

- [ ] **Step 1: Write the failing GPU oracle test (byte-identical occupancy)** — defines its helpers inline (sphere `ILevelSetProvider` + `AssetDatabase` loader for `Ablation.compute`, per the existing convention).

```csharp
// Ablation_GpuOracle_Tests.cs — Test 1
[Test, Category("GPU")]
public void RecomputeOccupancy_FrameZero_MatchesStaticUpload()
{
    var ab = LoadAblationShader();                  // AssetDatabase find Ablation.compute (inline helper, per convention)
    var (g, rb) = BuildSphere();                    // inline sphere helper, rb.Upload(g)
    using (rb)
    {
        var staticOcc = new uint[rb.VoxelCount];
        rb.VoxelOccupied.GetData(staticOcc);        // the baked OR uploaded by Upload()
        new AblationDetector(ab).RecomputeOccupancy(rb, new Vector3Int(g.dims.x,g.dims.y,g.dims.z));
        var rtOcc = new uint[rb.VoxelCount];
        rb.VoxelOccupied.GetData(rtOcc);
        for (int v = 0; v < rb.VoxelCount; v++)
            Assert.AreEqual(staticOcc[v], rtOcc[v],
                $"voxel {v}: RecomputeOccupancy must reproduce the static OR exactly on frame 0 (_Deleted all-0 ⇒ surviving-inside == baked cornerInside)");
    }
}
```

- [ ] **Step 2: Run it — Expected: FAIL** (Ablation.compute / AblationDetector / `_CornerInsideRT` / `_Deleted` don't exist).

- [ ] **Step 3: Implement** — `Ablation.compute` `RecomputeOccupancy` + `_CornerInsideRT` + `_Deleted` buffers + `AblationDetector.RecomputeOccupancy`.

```hlsl
// Ablation.compute
#pragma kernel RecomputeOccupancy
int3 _Dims;
int  _VoxelCount;
StructuredBuffer<int>    _VoxelCorner;     // [8*v+c] -> global cornerId (mirror Recon.compute decl)
RWStructuredBuffer<int>  _CornerInsideRT;  // read-only RT copy of baked cornerInside (1=inside,0=outside) — NEVER flipped
RWStructuredBuffer<int>  _Deleted;         // 1 = deleted this run, 0 = alive
RWStructuredBuffer<uint> _VoxelOccupied;   // RW now
[numthreads(64,1,1)]
void RecomputeOccupancy(uint3 id : SV_DispatchThreadID)
{
    uint v = id.x; if (v >= (uint)_VoxelCount) return;
    uint occ = 0u;
    [unroll] for (int c = 0; c < 8; c++) {
        int cid = _VoxelCorner[8*(int)v + c];
        if (_CornerInsideRT[cid] != 0 && _Deleted[cid] == 0) { occ = 1u; break; }   // surviving-inside
    }
    _VoxelOccupied[v] = occ;   // == baked OR when _Deleted all-0 (frame 0)
}
```

```csharp
// ReconBuffers.cs — add near VoxelOccupied
public ComputeBuffer CornerInsideRT;   // int[CornerCount], read-only RT copy of baked cornerInside (1/0)
public ComputeBuffer Deleted;          // int[CornerCount], 1=deleted, init 0
// ctor: CornerInsideRT = new ComputeBuffer(Math.Max(1,CornerCount), 4); Deleted = new ComputeBuffer(Math.Max(1,CornerCount), 4);
// Upload(g): var ins=new int[CornerCount]; for(i) ins[i]=g.cornerInside[i]; CornerInsideRT.SetData(ins);
//            Deleted.SetData(new int[Math.Max(1,CornerCount)]);   // all 0
// Dispose: CornerInsideRT?.Dispose(); Deleted?.Dispose();
```

```csharp
// AblationDetector.cs (minimal for Task 1)
public sealed class AblationDetector {
  readonly ComputeShader cs; readonly int kOcc;
  public AblationDetector(ComputeShader s){ cs=s; kOcc=cs.FindKernel("RecomputeOccupancy"); }
  public void RecomputeOccupancy(ReconBuffers rb, Vector3Int dims){
    cs.SetInts("_Dims",dims.x,dims.y,dims.z); cs.SetInt("_VoxelCount",rb.VoxelCount);
    cs.SetBuffer(kOcc,"_VoxelCorner",rb.VoxelCorner);
    cs.SetBuffer(kOcc,"_CornerInsideRT",rb.CornerInsideRT);
    cs.SetBuffer(kOcc,"_Deleted",rb.Deleted);
    cs.SetBuffer(kOcc,"_VoxelOccupied",rb.VoxelOccupied);
    cs.Dispatch(kOcc, Mathf.Max(1,Mathf.CeilToInt(rb.VoxelCount/64f)),1,1);
  }
}
```

- [ ] **Step 4: Run it — Expected: PASS** (byte-identical occupancy).
- [ ] **Step 5: Multi-subagent review** — verify `_CornerInsideRT` init == baked `cornerInside` AND is never written by any kernel (read-only); `_Deleted` inits to 0; `RecomputeOccupancy` OR matches `BackgroundGrid.cs:131-134` with the `&& _Deleted==0` term; `_VoxelOccupied` still uploaded once (frame-0 source) AND now RW; no other reader of occupancy regressed. Confirm paper §2.1.4 "voxel grids updated" + spec §"Occupancy formula" / §3.
- [ ] **Step 6: Commit** — `feat(reconDC): RecomputeOccupancy + _CornerInsideRT + _Deleted (dynamic occupancy, byte-identical frame-0)`.

---

### Task 2: `AblationTool` (swept capsule) + capsule SDF

**Files:**
- Create: `Assets/ReconGridDC/Cutting/AblationTool.cs`
- Test: `Assets/ReconGridDC/Tests/AblationTool_Tests.cs` (pure CPU, no GPU)
- (Optional deliverable) Create ONE shared test-helper file `Assets/ReconGridDC/Tests/AblationTestHelpers.cs` (a sphere `ILevelSetProvider`, `LoadAblationShader`/`LoadReconShader` via `AssetDatabase`, `SphereCenter`) — OR keep helpers inline per the existing convention. Choose ONE and apply it consistently across Tasks 1/3/4/5/6/7/8.

**Interfaces:**
- Produces: `AblationTool { Vector3 PrevTip, CurTip; float Radius; void Advance(Vector3 newTip); bool Valid; }` + `static float CapsuleSdf(Vector3 p, Vector3 a, Vector3 b, float r)` = `dist(p, segment[a,b]) - r`. Mirrors `CuttingTool`'s roll-prev/cur + `Advance` pattern (CuttingTool.cs:105-112). **`Valid` is hardcoded `true`** (a stationary probe still ablates; just `Valid = true`, no `… || true` tautology).

- [ ] **Step 1: Failing test**

```csharp
[Test] public void CapsuleSdf_PointInsideAndOutside()
{
    var a = new Vector3(0,0,0); var b = new Vector3(2,0,0); float r = 0.5f;
    Assert.Less(AblationTool.CapsuleSdf(new Vector3(1,0,0), a,b,r), 0f);     // axis center inside
    Assert.Less(AblationTool.CapsuleSdf(new Vector3(1,0.4f,0), a,b,r), 0f);  // within radius
    Assert.Greater(AblationTool.CapsuleSdf(new Vector3(1,0.6f,0), a,b,r), 0f);// beyond radius
    Assert.Greater(AblationTool.CapsuleSdf(new Vector3(-0.6f,0,0), a,b,r), 0f);// past cap
    Assert.AreEqual(0f, AblationTool.CapsuleSdf(new Vector3(1,0.5f,0), a,b,r), 1e-5f);// on surface
}

[Test] public void Advance_StationaryTip_StaysValid()   // should-fix: stationary probe still ablates
{
    var t = new AblationTool(new Vector3(1,0,0), 0.5f);
    t.Advance(new Vector3(1,0,0));                        // no movement
    Assert.IsTrue(t.Valid, "a stationary capsule probe is still a valid volumetric probe");
}
```

- [ ] **Step 2: Run — FAIL.**
- [ ] **Step 3: Implement** `AblationTool` (point-to-segment distance: clamp `t = dot(p-a,b-a)/dot(b-a,b-a)` to [0,1], `closest = a + t*(b-a)`, `return length(p-closest) - r`). `Advance(newTip)` rolls `PrevTip=CurTip; CurTip=newTip; Valid = true` (NO degenerate guard — unlike the blade's zero-area swept plane, a capsule is a volume that ablates even when stationary).
- [ ] **Step 4: Run — PASS.**
- [ ] **Step 5: Multi-subagent review** — capsule SDF correct (segment distance + cap handling); matches ref [22] "collision capsules" (ref22:361 — capsule **shape is paper-silent, grounded in [22]**); `Valid` is unconditionally true (no tautology). Confirm spec §0 / §Architecture tool.
- [ ] **Step 6: Commit** — `feat(reconDC): AblationTool swept-capsule + CapsuleSdf (Valid always true)`.

---

### Task 3: `DetectAblation` kernel — capsule range-select → `_Deleted=1` (NO flip, NO active write, DEFORMED frame)

**Files:**
- Modify: `Assets/ReconGridDC/Shaders/Ablation.compute` (add `DetectAblation`)
- Modify: `Assets/ReconGridDC/Cutting/AblationDetector.cs` (add `DetectAblation` dispatch + capsule uniforms; reset `_Deleted` to 0 each frame before dispatch)
- Test: `Ablation_GpuOracle_Tests.cs` (Test 2)

**Interfaces:**
- Consumes: `_CornerPos` (**DEFORMED** positions for the range test — re-audit bug #3, NOT rest), capsule uniforms `_CapA,_CapB,_CapR`.
- Produces: kernel `DetectAblation` [1/corner] writes ONLY `_Deleted[c]=1` for in-range corners. **Does NOT flip `_CornerInsideRT` (must-fix #1). Does NOT set `_CornerActive` (must-fix #2 — `_CornerActive=0` is a FINAL `FreezeDeleted` step in Task 5).** Per-frame reset of `_Deleted` happens before this kernel (instant re-run every frame, must-fix #7). Establishes the `deleted ⇔ in-capsule` invariant in the deformed frame (needed for the Task-4 capsule-SDF root).

- [ ] **Step 1: Failing test** — build sphere + a capsule through it; dispatch `DetectAblation`; assert: corners with `CapsuleSdf(deformedPos) <= 0` have `_Deleted==1`; corners outside the capsule have `_Deleted==0`; `_CornerInsideRT` is UNCHANGED; `_CornerActive` is UNCHANGED. **The oracle computes expected membership from the SAME positions the kernel uses — the DEFORMED `_CornerPos` read back from the GPU (bug #3), NOT a separately-computed rest position.**

```csharp
[Test, Category("GPU")]
public void DetectAblation_SetsDeletedOnlyInRange_NoFlip_NoActiveWrite()
{
    var ab = LoadAblationShader(); var (g, rb) = BuildSphere(); using (rb) {
        var det = new AblationDetector(ab);
        var insideBefore = new int[rb.CornerCount]; rb.CornerInsideRT.GetData(insideBefore);
        var activeBefore = new int[rb.CornerCount]; rb.CornerActive.GetData(activeBefore);
        Vector3 a = SphereCenter, b = SphereCenter + new Vector3(1,0,0); float r = 0.5f;
        det.DetectAblation(rb, new Vector3Int(g.dims.x,g.dims.y,g.dims.z), a, b, r);
        var del = new int[rb.CornerCount]; rb.Deleted.GetData(del);
        var insideAfter = new int[rb.CornerCount]; rb.CornerInsideRT.GetData(insideAfter);
        var activeAfter = new int[rb.CornerCount]; rb.CornerActive.GetData(activeAfter);
        var cp  = new Vector3[rb.CornerCount]; rb.CornerPos.GetData(cp);   // the DEFORMED positions the kernel read (bug #3)
        int flagged=0;
        for (int c=0;c<rb.CornerCount;c++){
            bool inRange = AblationTool.CapsuleSdf(cp[c], a,b,r) <= 0f;     // expected = same deformed _CornerPos
            Assert.AreEqual(inRange ? 1 : 0, del[c], $"corner {c}: _Deleted must be 1 iff in range (deformed frame)");
            Assert.AreEqual(insideBefore[c], insideAfter[c], $"corner {c}: _CornerInsideRT must NOT be flipped");
            Assert.AreEqual(activeBefore[c], activeAfter[c], $"corner {c}: _CornerActive must NOT be written by DetectAblation");
            if (inRange) flagged++;
        }
        Assert.Greater(flagged,0,"capsule must catch >=1 corner");
    }
}
```

- [ ] **Step 2: Run — FAIL.**
- [ ] **Step 3: Implement `DetectAblation`** (classify ONLY, DEFORMED frame):

```hlsl
#pragma kernel DetectAblation
float3 _CapA, _CapB; float _CapR; int _CornerCount;
StructuredBuffer<float3> _CornerPos;        // DEFORMED positions (deformed-frame range test — bug #3)
// _CornerInsideRT, _Deleted declared in Task 1
float CapsuleSdf(float3 p, float3 a, float3 b, float r){
    float3 ab=b-a; float t=saturate(dot(p-a,ab)/max(dot(ab,ab),1e-8)); float3 q=a+t*ab; return length(p-q)-r; }
[numthreads(64,1,1)]
void DetectAblation(uint3 id:SV_DispatchThreadID){
    uint c=id.x; if(c>=(uint)_CornerCount) return;
    if(CapsuleSdf(_CornerPos[c], _CapA,_CapB,_CapR) <= 0.0)   // DEFORMED _CornerPos
        _Deleted[c] = 1;     // in range -> deleted (covers BOTH paper:493-495 outside + paper:495-496 inside-ablated)
    // NO flip of _CornerInsideRT (read-only). NO _CornerActive write (final step is Task 5).
}
```

`AblationDetector.DetectAblation(rb, dims, a, b, r)` first **resets `_Deleted` to 0** (per-frame, instant re-run — must-fix #7; via a tiny `ClearDeleted` kernel or `rb.Deleted.SetData`), then sets `_CapA/_CapB/_CapR`, binds `_CornerPos`,`_CornerInsideRT`,`_Deleted`, dispatches `G(CornerCount)`.

- [ ] **Step 4: Run — PASS.**
- [ ] **Step 5: Multi-subagent review** — **DEFORMED-frame** range test (bug #3 — the cavity wall is a deformed-space DC surface, the `deleted ⇔ in-capsule` invariant needs one frame, a capsule's radius defeats the thin-plane miss; the oracle uses the same deformed `_CornerPos`); the "mirror DetectCut rest discipline" claim is REMOVED; paper §2.1.4 pass-1 (outside-in-range deleted) + pass-2 (inside-in-range ablated→deleted) both realized as `_Deleted=1`; NO flip of `_CornerInsideRT`, NO `_CornerActive` write (must-fix #1, #2); per-frame `_Deleted` reset (must-fix #7). Confirm vs paper:493-497 + spec §1 + §"DEFORMED frame".
- [ ] **Step 6: Commit** — `feat(reconDC): DetectAblation capsule range-select → _Deleted (deformed frame, no flip, no active write)`.

---

### Task 4 (REPLACED): `DC_FeaturePoints` ablation-mode branch — inline recompute (filtered baked window + inline capsule crossings)

**Replaces the old Task 4 (`GenerateFrontierIsect` + dynamic-isect buffer), which the re-audit proved structurally impossible:** a flat append buffer with its own counter is read by NO kernel (`DC_FeaturePoints` reads only the per-voxel static window `_Isect/_IsectWorld[off..off+cnt)`; those index arrays are baked once and never recomputed), and a cavity-interior receded-boundary voxel has baked `cnt==0` (all edges inside↔inside at bake) so `DC_FeaturePoints` early-returns an invalid `(0,0,0,0)`. The crossings are therefore computed **INLINE inside `DC_FeaturePoints`** under a new `_AblationMode` uniform. This DISSOLVES the RebuildIsectWorld append/ordering concern (no appends into `_IsectWorld`; `RebuildIsectWorld` is unchanged).

**Files:**
- Modify: `Assets/ReconGridDC/Shaders/Recon.compute`:
  - Add `int CorId(int3 c){ return c.x + (_Dims.x+1)*(c.y + (_Dims.y+1)*c.z); }` (bug #4 — Recon.compute has only `VoxId`; mirror Cutting.compute:160).
  - Add the `_AblationMode` (int) uniform + the capsule uniforms `_CapA`,`_CapB`,`_CapR` + the `_CornerInsideRT`,`_Deleted` buffer decls.
  - **Add the literal 12-edge local table** (re-audit IMPORTANT #2 — the plan must WRITE it, not just say "mirror"). Verbatim from `GridConventions.EdgeCorners` (Core/GridConventions.cs:15-20), with local corner ids matching `CornerOffset` V0..V7:
    ```hlsl
    // 12 cube edges → local corner-id pairs (verbatim GridConventions.EdgeCorners, GridConventions.cs:15-20)
    static const int2 EDGE_CORNERS_HLSL[12] = {
        int2(0,1), int2(1,2), int2(2,3), int2(3,0),   // e0..e3 bottom loop
        int2(4,5), int2(5,6), int2(6,7), int2(7,4),   // e4..e7 top loop
        int2(0,4), int2(1,5), int2(2,6), int2(3,7),   // e8..e11 verticals
    };
    ```
    (Edge ORDER is irrelevant — mechanism (2) accumulates an unordered QEF union; only edge MEMBERSHIP matters, so a wrong pair = a face/space diagonal would bisect a non-edge and emit a garbage crossing with no oracle to catch it. The test below adds that oracle.)
  - Add the ablation branch to `DC_FeaturePoints` (see Implement).
- Modify: `Assets/ReconGridDC/Recon/DualContouring.cs` — `Build` gains `bool ablationMode=false` (LAST arg, bug #2). **`Build` ALWAYS calls `cs.SetInt("_AblationMode", ablationMode ? 1 : 0)` on BOTH paths** (re-audit IMPORTANT — `_AblationMode` is a sticky global uniform; leaving it unset on the `false` path lets a prior `Build(true)` leak the ablation branch onto the blade → divergence). When `ablationMode==true` ALSO binds `_CornerInsideRT`/`_Deleted` + the capsule (`_CapA/_CapB/_CapR`) to `kFP` (and `kStitch`, Task 6) + sets `_GridEdgeCount`; when `false`, sets `_AblationMode=0` and binds nothing else → byte-identical.
- Test: `Ablation_GpuOracle_Tests.cs` (Test 3 — inline FP oracle).

**Interfaces:**
- Consumes (ablation branch only): `_VoxelCorner` (8 corner ids/voxel), `_CornerPos` (DEFORMED), `_CornerInsideRT`, `_Deleted`, the baked window `_Isect`/`_IsectWorld`/`_VoxelIsectOffset`/`_VoxelIsectCount`, capsule uniforms `_CapA`/`_CapB`/`_CapR`, `_AblationMode`. `CornerOffsetHLSL(lc)` (CuttingCommon.hlsl:41, already included) + a literal 12-edge `(lc_a,lc_b)` table (mirror `GridConventions.EdgeCorners`) enumerate the voxel's edges.
- Produces: `_VoxelExternalFP[v]` = the QEF feature point (w=1) over the UNION of (1) filtered baked window + (2) inline capsule crossings; `(0,0,0,0)` if the union is empty. `_AblationMode==0` ⇒ byte-identical to today.

- [ ] **Step 1: Failing test** — build sphere; `DetectAblation` a capsule that fully engulfs ≥1 interior corner whose voxel had baked `cnt==0`; run `dc.Build(..., ablationMode:true)`; read `_VoxelExternalFP`. Assert:
  - a **fully-interior** voxel (baked `cnt==0`) that loses ≥1 of its surviving-inside corners to deletion gets a **FINITE in-voxel feature point with `w==1`** (`BoxSdf(fp, voxelCenter, L) <= 0`), sourced from the inline capsule crossings (mechanism (2));
  - a voxel whose surviving-inside corners are **all** deleted gets `w==0`;
  - each inline crossing position satisfies `|CapsuleSdf(pos, a, b, r)| < 1e-4` (capsule root on the DEFORMED edge, NOT `fa/(fa-fb)`);
  - **edge-membership oracle (re-audit IMPORTANT #2):** CPU-recompute the expected receded-boundary edges using `GridConventions.EdgeCorners` + the deformed corner positions + the capsule; assert each inline crossing lies on a **real cube edge** — i.e. its two corners are a `GridConventions.EdgeCorners` pair, NOT a face/space diagonal — and the kernel's inline-crossing set matches the CPU set 1:1. (This is the only guard against an `EDGE_CORNERS_HLSL` transcription error, since a diagonal would still pass the `|CapsuleSdf|<eps` check.)
  - the filtered baked window (mechanism (1)) drops a crossing whose baked-inside corner is deleted (a voxel that keeps a surviving-inside corner but loses the inside corner of one baked isect emits fewer window contributions);
  - with `ablationMode:false`, `_VoxelExternalFP` is byte-identical to the static run on the same sphere (per-voxel `float4` exact compare);
  - **mode-flip guard (re-audit IMPORTANT — `_AblationMode` sticky global):** run `dc.Build(ablationMode:true)` then `dc.Build(ablationMode:false)` on the SAME `DualContouring` instance; assert the second `_VoxelExternalFP` is byte-identical to the static baseline (proves `Build` resets `_AblationMode=0` unconditionally, not relying on zero-init).

```csharp
[Test, Category("GPU")]
public void DcFeaturePoints_AblationBranch_InteriorVoxelGetsFiniteFp_ByteIdenticalWhenOff()
{
    // build sphere; DetectAblation(capsule engulfing an interior corner of a cnt==0 voxel);
    // dc.Build(rb, dims, L, qefIters, ablationMode:true); read _VoxelExternalFP.
    // Assert: the cnt==0 receded-boundary voxel has w==1 and BoxSdf(fp,center,L)<=0 (finite, in-voxel).
    // Assert: a voxel with all surviving-inside corners deleted has w==0.
    // Assert: each inline crossing pos has |CapsuleSdf(pos,a,b,r)|<1e-4.
    // Then dc.Build(rb, dims, L, qefIters, ablationMode:false) → _VoxelExternalFP byte-identical to the static baseline.
}
```

- [ ] **Step 2: Run — FAIL.**
- [ ] **Step 3: Implement** the `_AblationMode` branch in `DC_FeaturePoints`:
  - `_AblationMode==0`: **today's path, byte-identical** (read the static window `[off,off+cnt)`, Eq2 mean of `_IsectWorld[off+s]`, Eq1-5 QEF, `cnt==0 ⇒ (0,0,0,0)`). Not one instruction changes.
  - `_AblationMode!=0`: build the crossing set as the UNION of —
    - **(1) Filtered baked window:** for `s` in `[0,cnt)`, let `it = _Isect[off+s]`; inside corner `= (it.insideA==1) ? it.cornerA : it.cornerB`; KEEP iff `_Deleted[insideCorner]==0`; contribute `(p_i = _IsectWorld[off+s], n_i = it.normal)`.
    - **(2) Inline capsule crossings:** for each of the voxel's 12 edges `e` in `[0,12)`, `(lc_a, lc_b) = EDGE_CORNERS_HLSL[e]`, `A = _VoxelCorner[8*v+lc_a]`, `B = _VoxelCorner[8*v+lc_b]`; if `(survivingInside(A) && deletedInside(B)) || (survivingInside(B) && deletedInside(A))` — a **receded-boundary** edge — bisect `t∈[0,1]` (~24 iters) on `CapsuleSdf(lerp(_CornerPos[A],_CornerPos[B],t), _CapA,_CapB,_CapR)` to the root `t*` (valid because `deleted ⇔ in-capsule` ⇒ surviving endpoint sdf>0, deleted endpoint sdf<0 ⇒ exactly one sign change); `p_i = lerp(_CornerPos[A],_CornerPos[B],t*)`; `n_i = normalize(p_i - closestPointOnSegment(p_i,_CapA,_CapB))` (capsule outward normal); contribute `(p_i,n_i)`.
  - Accumulate the Eq2 initial mean + run the existing Eq1-5 QEF (D1 gradient, Eq4 decay, D5 box-SDF stop) over the combined `(p_i,n_i)` set; empty union ⇒ `(0,0,0,0)`, else `(x, 1.0)`.
  - **Crossing subtype discipline (bug #6):** emit the inline capsule crossing ONLY on the surviving↔**deleted** edge subtype; a surviving↔**φ-outside** edge keeps its baked φ crossing via mechanism (1). Do NOT use `fa/(fa-fb)` for the inline crossing.
- [ ] **Step 4: Run — PASS** (especially the byte-identical `ablationMode:false` per-voxel FP compare + the finite-FP-on-`cnt==0` interior voxel).
- [ ] **Step 5: Multi-subagent review** — the static branch is a literal byte-identical early path (the critical invariant); the ablation branch's union = filtered baked window (drops crossings whose inside corner is deleted — paper:493-495 "no longer participate in surface generation") + inline capsule crossings (capsule-SDF root on the DEFORMED edge, NOT `fa/(fa-fb)`, bug #6); the receded-boundary predicate uses the canonical surviving/deleted-inside formula; `CorId()` added to Recon.compute (bug #4); the dynamic-isect buffer + `GenerateFrontierIsect` are GONE (no appends into `_IsectWorld` — RebuildIsectWorld unchanged); `dc.Build(..., ablationMode=false)` real signature (bug #2). Confirm vs spec §"STRUCTURAL REWORK" + §"Cavity reconstruction (C)" + paper:497-498.
- [ ] **Step 6: Commit** — `feat(reconDC): DC_FeaturePoints _AblationMode branch — inline recompute (filtered window + capsule crossings) + CorId`.

---

### Task 5: `SeverDeleted` — fully isolate deleted corners + final `FreezeDeleted` `_CornerActive=0`

**Files:**
- Modify: `Assets/ReconGridDC/Shaders/Ablation.compute` (`SeverDeleted` [1/grid-edge] gated on `_Deleted[idA]==1 || _Deleted[idB]==1`; + a `FreezeDeleted` [1/corner] final pass setting `_CornerActive[c]=0` for `_Deleted[c]==1`)
- Modify: `AblationDetector.cs` (bind `_NbrIdx`,`_BendPairs`,`_GridEdges`,`_Deleted`,`_CornerActive`; dispatch `SeverDeleted` then `FreezeDeleted` AFTER `DetectAblation`)
- Test: `Ablation_GpuOracle_Tests.cs` (Test 4 + the ComputeParticleRot identity test, must-fix #6 + bug #5)

**Interfaces:**
- Consumes: `_Deleted`, `_NbrIdx`, `_BendPairs`, `_GridEdges`, `_CornerActive`.
- Produces: a deleted corner with ALL 6 `NbrIdx` slots == -1, ALL incident bend pairs `alive==0`, and `_CornerActive==0` (applied as the FINAL `FreezeDeleted` step — must-fix #2).

- [ ] **Step 1: Failing tests**
  - (4a) after the full pass, **every** `_NbrIdx` slot (all 6) of a deleted corner == -1; every bend pair incident to a deleted corner (the ones centered at it + any neighbor pair referencing it as an arm) has `alive==0` (must-fix #3); deleted corner `_CornerActive==0` (final `FreezeDeleted`).
  - (4b — must-fix #6 + **bug #5**) a corner that loses **BOTH** arms on ONE axis (delete BOTH x-neighbors, or use a grid-boundary corner with a single arm on that axis) → after `solver.ComputeRotations`, its `_ParticleRot` is **identity** via the M3 zero-diagonal gate, NOT NaN. The corner must ALSO be **deformed enough to clear the M4 deadband** (`‖F-I‖_F >= 1e-3`, Physics.compute:472-485) so the identity is attributable to the **M3 gate** (`min(mx,my,mz)<axisEps`, Physics.compute:448-452), not the deadband. **Deleting only ONE neighbor leaves the opposite arm so `mx>0` and the polar path runs — that does NOT exercise M3 (the old test 4b was wrong).** Assert `rot[c] == RotGpu.Identity` (every component finite + equal to identity).

```csharp
[Test, Category("GPU")]
public void SeverDeleted_IsolatesDeletedCorner_All6Arms_AndBendPairs() {
    // DetectAblation + SeverDeleted + FreezeDeleted; pick a fully-interior deleted corner c.
    var nbr = new int[6*rb.CornerCount]; rb.NbrIdx.GetData(nbr);
    for (int s=0;s<6;s++) Assert.AreEqual(-1, nbr[6*c+s], $"deleted corner {c} arm {s} must be severed");
    // bend pairs centered at c all alive==0; + scan neighbor pairs referencing c.
    var act = new int[rb.CornerCount]; rb.CornerActive.GetData(act);
    Assert.AreEqual(0, act[c], "deleted corner frozen via FINAL FreezeDeleted _CornerActive=0 pass");
}

[Test, Category("GPU")]
public void Corner_LosingBothArmsOnAxis_WritesIdentityRot_NotNaN_ViaM3() {   // must-fix #6 + bug #5
    // Pick an interior corner c with both x-neighbors present. Delete BOTH x-neighbors (so mx≈0 → M3 fires).
    // Displace c (or its surviving y/z neighbors) so ‖F-I‖_F >= 1e-3 → clear the M4 deadband.
    // solver.ComputeRotations; assert rb.ParticleRot[c] == RotGpu.Identity (M3 zero-diagonal → identity), all finite.
    // Rationale comment: deleting ONE x-neighbor would leave the opposite arm (mx>0) → polar path, NOT M3.
}
```

- [ ] **Step 2: Run — FAIL.**
- [ ] **Step 3: Implement**
  - `SeverDeleted` reuses the `SeverLinks` WRITE CONTRACT (Cutting.compute: `_NbrIdx[6*idA+sPlus]=-1`, `_NbrIdx[6*idB+sMinus]=-1`, bend `alive=0` on the two endpoints' pairs referencing the other) but the GATE is `_Deleted[idA]==1 || _Deleted[idB]==1` (NOT `EdgeIsCut`). Because this fires on **every** edge incident to a deleted corner, all 6 of that corner's arms + all its bend pairs die (must-fix #3). Do NOT gate on `_CornerActive==0` (must-fix #2 — the baked exterior shell is active=0 and must not be touched).
  - `FreezeDeleted` [1/corner]: `if (_Deleted[c]==1) _CornerActive[c]=0;` — the FINAL step, after `DetectAblation` has classified (must-fix #2: active=0 applied to deleted corners only, as a separate final step).
- [ ] **Step 4: Run — PASS.**
- [ ] **Step 5: Multi-subagent review** — sever contract matches `SeverLinks` writes (NbrIdx=-1 both ways + bend alive=0) but gated on `_Deleted` so a deleted corner is FULLY isolated (all 6 arms — must-fix #3); `_CornerActive=0` is a FINAL `FreezeDeleted` pass on deleted corners only, never the baked exterior shell (must-fix #2); `ComputeParticleRot` excludes the deleted neighbor and a corner losing **BOTH** arms of an axis falls back to **identity** via the M3 gate (must-fix #6 — NO Fig4(d)); the test deletes BOTH arms + clears the M4 deadband so identity is attributable to M3, not the deadband (bug #5). Confirm vs paper:493-494 + spec §2 + §"ComputeParticleRot".
- [ ] **Step 6: Commit** — `feat(reconDC): SeverDeleted (full isolation, all 6 arms) + FreezeDeleted final _CornerActive=0`.

---

### Task 6: Runtime surface-edge gate for `DC_Stitch` (cavity outer skin, ablation mode only) — Approach B stitch

The cavity boundary creates new isosurface crossings (a surviving-inside corner now borders a deleted corner) absent from the static `_SurfaceEdges`. In ablation mode, `DC_Stitch` iterates `_GridEdges` with a runtime predicate + DERIVES `insideA`; otherwise it uses the static `_SurfaceEdges` path (byte-identical).

> **Predicate width (re-audit IMPORTANT #1 — do NOT narrow this gate):** the `DC_Stitch` gate is the **BROAD** "exactly one endpoint surviving-inside" (a *surface-bearing* edge), which fires on BOTH surviving↔deleted (new cavity wall) AND surviving↔φ-outside (the surviving original skin). This is REQUIRED — narrowing it to mechanism-(2)'s receded-boundary predicate (surviving↔deleted only) would un-stitch the entire surviving skin → a gross hole, and would break frame-0 predicate-equivalence (test 6b: with nothing deleted, the surface-bearing set == the static `_SurfaceEdges` φ-sign-change set). Only the `DC_FeaturePoints` mechanism-(2) capsule crossing (Task 4) uses the NARROW receded-boundary predicate. The two predicates are intentionally different — see spec §"Occupancy / post-delete state".

**Files:**
- Modify: `Assets/ReconGridDC/Shaders/Recon.compute` (`DC_Stitch`: add an `_AblationMode` uniform; when `==0` → the existing `_SurfaceEdges`/`_SurfaceEdgeCount` path UNCHANGED, verbatim early-out; when `!=0` → iterate `_GridEdges` (GridEdgeGpu2, 16B, NO `insideA`) over `GridEdgeCount`, DERIVE `insideA = (_CornerInsideRT[CorId(baseCorner)]!=0 && _Deleted[CorId(baseCorner)]==0)`, gate each quad on "exactly one endpoint surviving-inside" + `_VoxelOccupied`, and use the DERIVED `insideA` for BOTH `insideCornerCoord` AND the winding swap — must-fix #4). `CorId()` is already added in Task 4; if Task 6 is implemented first, add it here.
- Modify: `Assets/ReconGridDC/Recon/DualContouring.cs` (`Build`'s `bool ablationMode=false` arg → when true, binds `_GridEdges`/`_CornerInsideRT`/`_Deleted` to `kStitch` + sets `_GridEdgeCount`; default false = today's `_SurfaceEdges` path). Reuse the same `ablationMode` arg added in Task 4.
- Test: `Ablation_GpuOracle_Tests.cs` (Test 5 cavity wall + Test 6 byte-identity + Test 6b predicate-equivalence)

**Interfaces:**
- Consumes: `_GridEdges` (16B GridEdgeGpu2, count `GridEdgeCount`), `_CornerInsideRT`, `_Deleted`, `_VoxelOccupied`, `_AblationMode`.
- Produces: cavity-boundary outer-skin triangles welded to the receded DC feature points (from Task 4's inline `DC_FeaturePoints`); **byte-identical** stitch when `_AblationMode==0`.

- [ ] **Step 1: Failing tests** —
  - (5) in ablation mode, the cavity boundary (surviving-inside ↔ deleted) produces stitched skin triangles that weld to the receded DC feature points (one shared DC vertex per receded-boundary voxel — watertight, no rim hole).
  - (6 — CRITICAL) with `ablationMode:false`, a full `dc.Build(ablationMode:false)` `_Tri` buffer + counter is byte-identical to the static-`_SurfaceEdges` run on the same sphere — using the **ORDER-INSENSITIVE** oracle (bug #1: assert `TriCounter[0]` first, then compare the canonicalized tri SET or a commutative checksum; an index-by-index compare false-fails because `DC_Stitch` appends via `InterlockedAdd`). Baseline captured **in-test** (snapshot the static path's `_Tri` set + counter, then run `ablationMode:false` and compare). **The comparison run must be preceded by a `dc.Build(ablationMode:true)` on the SAME `DualContouring` instance** (re-audit IMPORTANT — `_AblationMode` sticky-global guard): the final `ablationMode:false` must still be byte-identical to the baseline, proving `Build` unconditionally resets `_AblationMode=0` (NOT relying on zero-init, which a prior `Build(true)` would have dirtied).
  - (6b — predicate equivalence, should-fix) with nothing deleted, the runtime-`_GridEdges` gate (derived `insideA` + "exactly one endpoint surviving-inside") selects EXACTLY the same quad set (same canonicalized `(voxel4, winding)` tuples) as the static `_SurfaceEdges` list.

```csharp
[Test, Category("GPU")]
public void DcBuild_AblationModeOff_ByteIdenticalTri_OrderInsensitive() {
    var (g, rb) = BuildSphere(); using (rb) {
        var dc = new DualContouring(reconShader);
        // baseline: run the static/today path IN-TEST (ablationMode:false), snapshot canonicalized set + counter
        dc.Build(rb, new int3(g.dims.x,g.dims.y,g.dims.z), L, qefIters, ablationMode:false);
        var cntA = new uint[1]; rb.TriCounter.GetData(cntA);
        var triA = new int[3*rb.TriCapacity]; rb.Tri.GetData(triA);
        var baseSet = CanonicalTriSet(triA, (int)cntA[0]);   // rotate each tri to min-vertex-first, then sort the list
        // DIRTY _AblationMode=1 on the SAME instance (sticky-global guard): nothing is deleted (_Deleted all-0),
        // so this trivial ablation pass is harmless, but it leaves _AblationMode=1 unless Build resets it.
        dc.Build(rb, new int3(g.dims.x,g.dims.y,g.dims.z), L, qefIters, ablationMode:true);
        // re-run ablationMode:false and compare order-insensitively (InterlockedAdd ⇒ index order is nondeterministic — bug #1).
        // MUST still equal the baseline ⇒ proves Build unconditionally SetInt("_AblationMode",0), not zero-init reliance.
        dc.Build(rb, new int3(g.dims.x,g.dims.y,g.dims.z), L, qefIters, ablationMode:false);
        var cntB = new uint[1]; rb.TriCounter.GetData(cntB);
        var triB = new int[3*rb.TriCapacity]; rb.Tri.GetData(triB);
        Assert.AreEqual(cntA[0], cntB[0], "TriCounter must match first");
        CollectionAssert.AreEqual(baseSet, CanonicalTriSet(triB, (int)cntB[0]), "tri SET must be byte-identical (order-insensitive)");
    }
}
```

- [ ] **Step 2: Run — FAIL.**
- [ ] **Step 3: Implement** the `_AblationMode` branch in `DC_Stitch` (the static `_AblationMode==0` branch is a literal early-out returning before any new code path so it stays byte-identical; the runtime branch reuses `StitchVertId`/`STENCIL_OFF`/the winding logic but sources `baseCorner` from `_GridEdges[e]` (GridEdgeGpu2 16B) and `insideA` from the DERIVED surviving-inside of `CorId(baseCorner)`, gating each quad on "exactly one endpoint surviving-inside" + `_VoxelOccupied`); thread `ablationMode` + `_GridEdgeCount` through `DualContouring.Build`; bind `_GridEdges`/`_CornerInsideRT`/`_Deleted` to `kStitch`.
- [ ] **Step 4: Run — PASS** (especially the order-insensitive byte-identity + predicate-equivalence gates).
- [ ] **Step 5: Multi-subagent review** — the runtime predicate expresses exactly the cavity-boundary crossings (no double/missing quads — R2); `insideA` DERIVED from `_CornerInsideRT`+`_Deleted` for BOTH the key and the winding swap (must-fix #4: `_GridEdges` is 16B with NO baked `insideA`, count `GridEdgeCount` — distinct from 32B `_SurfaceEdges`/`SurfaceEdgeCount`); `CorId()` present in Recon.compute (bug #4); ablation-mode-only with a byte-identical static early-out (the critical invariant); the `_Tri` byte-identity oracle is **order-insensitive** (bug #1); watertight cavity (skin welds to the Task-4 inline feature points). Confirm vs paper:498-499 ("surface redrawn") + spec §"Cavity reconstruction (C)" + the critical invariant.
- [ ] **Step 6: Commit** — `feat(reconDC): DC_Stitch _AblationMode runtime gate (derive insideA from _CornerInsideRT/_Deleted; order-insensitive byte-identical off)`.

---

### Task 7: `AblationDetector.Dispatch` orchestrator (full per-frame sequence)

Assemble the kernels into the §2.1.4 sequence, mirroring `CutDetector.Dispatch`. Instant deletion ⇒ the WHOLE pass re-runs every frame; the `_Deleted` flag is reset at the head (must-fix #7). **There is NO `GenerateFrontierIsect` step — the cavity crossings live inside `DC_FeaturePoints` (Task 4).**

**Files:**
- Modify: `Assets/ReconGridDC/Cutting/AblationDetector.cs` (`public void Dispatch(AblationTool tool, ReconBuffers rb, Vector3Int dims, float voxelL, Vector3 gridOrigin)` = reset `_Deleted` to 0 → `DetectAblation` (deformed frame) → `SeverDeleted` → `FreezeDeleted` → `RecomputeOccupancy`)
- Test: `Ablation_GpuOracle_Tests.cs` (Test 7 — end-to-end on a CPU mirror + ordering/idempotence oracle)

**Interfaces:**
- Produces: `AblationDetector.Dispatch(AblationTool, ReconBuffers, Vector3Int, float, Vector3)` — the complete ablation pass, consumed before `dc.Build(..., ablationMode:true)`.

- [ ] **Step 1: Failing tests** —
  - (7a) run `Dispatch` once; a CPU mirror of capsule membership (on the deformed `_CornerPos`) predicts the `_Deleted` / occupancy sets; assert: ablated voxels (all surviving-inside corners gone) → `_VoxelOccupied==0`; deleted corners severed (all 6 arms) + frozen (`_CornerActive==0`).
  - (7b — ordering/idempotence oracle, should-fix) `DC_FeaturePoints`/`DC_Stitch` read the post-`DetectAblation` `_Deleted` state (never an in-flight value); running `Dispatch` twice is idempotent (same `_Deleted`, same occupancy — instant re-run + per-frame `_Deleted` reset; a second `dc.Build(ablationMode:true)` yields the same inline feature points).

- [ ] **Step 2: Run — FAIL.**
- [ ] **Step 3: Implement** `Dispatch` (reset `_Deleted` to 0 at the head; dispatch the kernels in the canonical order: DetectAblation → SeverDeleted → FreezeDeleted → RecomputeOccupancy). Mirror `CutDetector.Dispatch`'s uniform/bind structure. **NO `GenerateFrontierIsect`** (crossings live in `DC_FeaturePoints`, Task 4).
- [ ] **Step 4: Run — PASS.**
- [ ] **Step 5: Multi-subagent review** — the dispatch order realizes paper:493-499 step-for-step (classify → sever → freeze → occupancy; the crossings/redraw happen in `dc.Build(ablationMode:true)`); ordering safe (classify ALL `_Deleted` before `dc.Build` reads it — must-fix #1/#7; `FreezeDeleted` is the final detector step); per-frame `_Deleted` reset (must-fix #7); no `GenerateFrontierIsect`/dynamic-isect (Option-A rework); reuses CutDetector patterns. Confirm vs spec §Mechanism + §Architecture loop.
- [ ] **Step 6: Commit** — `feat(reconDC): AblationDetector.Dispatch — full §2.1.4 ablation pass (reset/frame, instant delete, no dynamic-isect)`.

---

### Task 8: `ReconGridManager` integration — `cuttingMode` toggle + cavity demo

**Files:**
- Modify: `Assets/ReconGridDC/Demo/ReconGridManager.cs` (add `enum CuttingMode { Blade, Ablation }` + `public CuttingMode cuttingMode = CuttingMode.Blade`; `public float ablationRadius = 0.4f`; an `AblationTool ablationTool` + `AblationDetector ablationDetector` constructed when the ablation shader is assigned; in `Update`, when `cuttingMode==Ablation`: advance the capsule along an auto-sweep path (reuse the ReconGridManager.cs:262-280 driver) + `ablationDetector.Dispatch(ablationTool, rb, dims, L, Vector3.zero)` in the cutDetector slot, then `solver.ComputeRotations` + `dc.Build(rb, dims, L, qefIters, ablationMode:true)`; when `Blade`: today's path unchanged, `dc.Build(rb, dims, L, qefIters, ablationMode:false)`. **The blade `cutDetector.Dispatch` (incl. `SeverLinks`) must NOT run in ablation mode; `ablationDetector.Dispatch` must NOT run in blade mode** — should-fix mode guard)
- Modify: `DemoSceneSetup.cs` if it wires shaders (assign `Ablation.compute`)
- Test: manual Unity verification (the user) + `Ablation_GpuOracle_Tests.cs` (Test 8: full `dc.Build(ablationMode:true)` after Dispatch yields a watertight cavity; `ablationMode:false` byte-identical to the blade baseline via the **order-insensitive** `_Tri` oracle; assert `SeverLinks` is never dispatched in ablation mode)

**Interfaces:**
- Consumes: all prior tasks.
- Produces: a runnable ablation demo (auto-swept probe carves a cavity) + the preserved blade demo.

- [ ] **Step 1: Failing test (order-insensitive byte-identical blade mode + mode guard)** — with `cuttingMode==Blade`, the full frame output (`_Tri` set + counter, order-insensitive — bug #1) equals the pre-ablation baseline (regression gate) and `ablationDetector.Dispatch` is never called; with `Ablation`, a cavity (occupancy-0 region) + watertight wall appears and the blade `cutDetector.Dispatch`/`SeverLinks` is never called. Document mode-fixed-at-build (the persistent `_VoxelOccupied`/`_CornerInsideRT`/`_Deleted` require a re-`Upload` to switch — should-fix mode-switch note).
- [ ] **Step 2: Run — FAIL.**
- [ ] **Step 3: Implement** the `cuttingMode` toggle + the ablation auto-sweep + the loop integration + the mode guards + thread `ablationMode` to `dc.Build`.
- [ ] **Step 4: Run — PASS.**
- [ ] **Step 5: Multi-subagent review** — the loop order matches the spec (solver.Step → ablationTool.Advance → ablationDetector.Dispatch [DetectAblation→SeverDeleted→FreezeDeleted→RecomputeOccupancy] → solver.ComputeRotations → dc.Build(ablationMode:true) — should-fix loop reconcile; the crossings/redraw are inside dc.Build, Task 4); blade mode untouched (order-insensitive byte-identical) and the two detectors are mutually exclusive per mode (should-fix mode guard); mode is fixed at build (re-Upload to switch); radius tunable. Final whole-module paper-alignment audit (§2.1.4 + [22] + spec, end-to-end), confirming the cavity wall is the re-contoured outer surface (Approach B, inline recompute), NOT `_CutFP`/`BuildCutTriangles` and NOT a dynamic-isect buffer.
- [ ] **Step 6: Commit** — `feat(reconDC): cuttingMode Blade|Ablation toggle + auto-swept ablation probe demo (Approach B cavity, inline recompute)`.
- [ ] **Step 7: USER VERIFIES in Unity** — set `cuttingMode=Ablation`, tune `ablationRadius`; confirm the probe carves a volume-loss cavity with a watertight re-contoured wall, and `Blade` mode is unchanged.

---

## Self-Review

**Spec coverage:** §0 capsule range → Task 2+3; §1 classify `_Deleted` (no flip, deformed frame) → Task 3; §"STRUCTURAL REWORK" inline-recompute `DC_FeaturePoints` ablation branch (filtered window + capsule-SDF roots) → Task 4; §2 SeverDeleted (full isolation) + final `FreezeDeleted` `_CornerActive=0` → Task 5; §3 RecomputeOccupancy → Task 1; §"Cavity reconstruction (C)" Approach B (inline `DC_FeaturePoints` + `DC_Stitch` runtime gate) → Task 4+6; cuttingMode/radius/auto-sweep/Valid-always-true/instant-reset/freeze → Task 8/2/7; byte-identical invariant (order-insensitive) → Tasks 1,4,6,8 gates; ComputeParticleRot identity-fallback (must-fix #6 + bug #5) → Task 5 test 4b; risks R1(Tasks 1,4,6), R2(Task6), R3(Tasks 4,7), R4(Task5), R5(Task4 cost). All spec sections mapped.

**Task list — before (old, dynamic-isect) → after (Option A inline recompute + 6 bug fixes):**
| # | OLD (dynamic-isect buffer) | NEW (inline recompute + bug fixes) |
|---|---|---|
| 1 | RecomputeOccupancy + `_CornerInsideRT` + `_Deleted` | unchanged (formula `inside && !deleted`; byte-identical frame-0 foundation) |
| 2 | AblationTool + CapsuleSdf (`Valid` always true) | unchanged + capsule shape phrased **paper-silent, grounded in [22]**; optional shared test-helper deliverable |
| 3 | DetectAblation → `_Deleted` (no flip/no active), **rest frame "mirror DetectCut"** | DetectAblation → `_Deleted`, **DEFORMED frame** (bug #3); oracle uses the SAME deformed `_CornerPos`; "mirror DetectCut rest discipline" claim DELETED |
| 4 | **`GenerateFrontierIsect`** + **dynamic-isect buffer** → DC_FeaturePoints | **REPLACED**: `DC_FeaturePoints` `_AblationMode` branch = **inline recompute** (filtered baked window + inline capsule-SDF roots on surviving↔deleted edges); a `cnt==0` interior voxel gets a finite FP w=1; all-deleted → w=0; byte-identical when off; **+ `CorId` added to Recon.compute** (bug #4); `dc.Build(..., bool ablationMode=false)` real signature (bug #2); dynamic-isect buffer + kernel **DELETED** |
| 5 | SeverDeleted gated on `_Deleted` (all 6 arms) + `FreezeDeleted` + ComputeParticleRot identity test (**delete ONE arm**) | unchanged sever/freeze; **test 4b FIXED** — delete **BOTH** arms on an axis + clear the M4 deadband so M3 genuinely fires (bug #5) |
| 6 | DC_Stitch runtime gate deriving `insideA`; static branch byte-identical; index-by-index `_Tri` oracle | unchanged gate + **+ CorId in Recon.compute** (bug #4) + **ORDER-INSENSITIVE** `_Tri` byte-identity oracle (set/checksum, baseline in-test — bug #1) + predicate-equivalence |
| 7 | Dispatch: reset `_Deleted`+**isect counter** → DetectAblation → **GenerateFrontierIsect** → SeverDeleted → FreezeDeleted → RecomputeOccupancy | Dispatch: reset `_Deleted` → DetectAblation → SeverDeleted → FreezeDeleted → RecomputeOccupancy (**NO GenerateFrontierIsect** — crossings in DC_FeaturePoints); + ordering/idempotence oracle |
| 8 | cuttingMode toggle + demo + mode guard + mode-fixed note | unchanged + **order-insensitive** `_Tri` blade-baseline oracle (bug #1); real `dc.Build(..., ablationMode:true)` signature (bug #2) |

**Placeholder scan:** kernel + C# + test bodies are concrete. The only implementer choice is the capsule-SDF root method in Task 4 (bisection ~24 iters vs closed-form segment-capsule) — both specified, the GPU oracle validates `|CapsuleSdf(pos)|<eps`. No "TBD/handle edge cases" placeholders. The `fa/(fa-fb)` φ-lerp is explicitly REMOVED for the inline capsule crossing (bug #6); the surviving↔φ-outside subtype keeps its baked φ crossing (filtered window).

**Type consistency:** `_CornerInsideRT` (int, 1/0, read-only), `_Deleted` (int, 1/0), `_CornerActive` (int), `_VoxelOccupied` (uint), `_AblationMode` (int) consistent across Tasks 1/3/4/5/6; surviving-inside formula `_CornerInsideRT[c]!=0 && _Deleted[c]==0` identical in `RecomputeOccupancy`/`DC_FeaturePoints` branch/`DC_Stitch`; `_GridEdges` = GridEdgeGpu2 16B / `GridEdgeCount` (NOT `_SurfaceEdges` 32B / `SurfaceEdgeCount`) in Task 6; `CorId(int3)` added to Recon.compute (bug #4) used by Tasks 4+6; `AblationDetector.Dispatch(AblationTool, ReconBuffers, Vector3Int, float, Vector3)` and the REAL `DualContouring.Build(ReconBuffers, int3, float, int, float cutFPInterp=1f, bool ablationMode=false)` (bug #2) consistent across Tasks 4/6/7/8; `CapsuleSdf(p,a,b,r)` identical in C# (Task 2) and HLSL (Tasks 3/4).

**STRUCTURAL REWORK audit (Option A):** the dynamic-isect buffer + `GenerateFrontierIsect` kernel are DELETED from both docs (re-audit proof: a flat append buffer is read by NO kernel — `DC_FeaturePoints` reads only the per-voxel static window whose index arrays are baked once and never recomputed — and a cavity-interior voxel has baked `cnt==0` ⇒ invalid `(0,0,0,0)` early-out). The cavity crossings are computed INLINE in `DC_FeaturePoints` under `_AblationMode` (filtered baked window + inline capsule-SDF roots). This dissolves the RebuildIsectWorld append/ordering concern (no `_IsectWorld` appends; RebuildIsectWorld unchanged). `DC_FeaturePoints` + `DC_Stitch` are NO LONGER "reused verbatim" (verbatim static branch + new ablation branch). (spec §"STRUCTURAL REWORK", §"Net-new", §"Reused verbatim", §"Buffers", Task 4)

**6 REWORK-bug audit (all fixed):**
1. **Order-sensitive `_Tri` oracle** → ORDER-INSENSITIVE (assert TriCounter first, then canonicalized tri SET or commutative checksum; baseline captured in-test) — Global Constraints + Tasks 4/5/6/8.
2. **`dc.Build` signature** → real `(rb, dims, L, qefIters, cutFPInterp=1f)` + added `bool ablationMode=false`; ONE concrete wiring specified; all call sites use the real arg order — Global Constraints + Tasks 4/6/8.
3. **DetectAblation frame** → DEFORMED `_CornerPos` consistently; oracle computes expected membership from the same deformed positions; "mirror DetectCut rest discipline" claim deleted, deformed-frame rationale added — Global Constraints + Task 3 + spec §"DEFORMED frame".
4. **`CorId` missing in Recon.compute** → added `int CorId(int3 c){...}` (mirror Cutting.compute:160) in Tasks 4 + 6 Files/Implement (won't compile without it) — Global Constraints + Tasks 4/6.
5. **Task-5 test 4b wrong** → delete **BOTH** arms on one axis (M3 fires only when mx/my/mz≈0) AND clear the M4 deadband so identity is attributable to M3 — Task 5 test 4b + §"ComputeParticleRot".
6. **Capsule-SDF root under-specified** → state the `deleted ⇔ in-capsule` invariant ⇒ exactly one sign change on a surviving↔deleted edge ⇒ bisection valid (~24 iters, same frame as detection); tie the crossing to the surviving↔deleted subtype only (surviving↔φ-outside keeps its baked φ crossing) — Task 4 + spec §"Capsule-SDF root finder".

**Minors (all applied):** test helpers defined inline per the existing convention (no nonexistent shared `BuildSphere`/`SphereCenter`/`LoadAblationShader`; optional shared-helper deliverable in Task 2) — Global Constraints + Task 2; byte-identity baseline captured IN-TEST (no committed artifact) — Global Constraints + Task 6; paper-silent citations softened (capsule shape + "within a certain range" = paper-silent, grounded in [22]) — Global Constraints + Tasks 2 + spec; "no longer participate in SURFACE generation" (paper:493-495) explicitly traced to RecomputeOccupancy (deleted corner drops from occupancy) + the `DC_FeaturePoints` ablation branch FILTERING OUT crossings whose inside corner is deleted — spec §"Paper-alignment trace".

**Earlier must-fix audit (all 7 preserved):** #1 `_Deleted` + read-only `_CornerInsideRT` + occupancy/surviving formula (Tasks 1,3,4,6 + Global Constraints) · #2 NOT `_CornerActive==0` as deleted; `FreezeDeleted` final pass (Tasks 3,5) · #3 SeverDeleted full isolation, all 6 arms test (Task 5) · #4 `DC_Stitch` derives `insideA` from `_CornerInsideRT`+`_Deleted` over 16B `_GridEdges`, static branch byte-identical (Task 6) · #5 capsule-SDF root not `fa/(fa-fb)` (Task 4) · #6 ComputeParticleRot identity-fallback, no Fig4(d), corner-losing-axis identity test (Global Constraints + Task 5) · #7 per-frame reset of `_Deleted`, no `EmitCutPoint` reuse (Tasks 3,7).

**Should-fix audit (all reflected):** occupancy formula reconciled both docs · Task-4/6/8 byte-identity strengthened to full `dc.Build` `_Tri`+counter, **order-insensitive** · frame-0 predicate-equivalence oracle (Task 6) · mode-switch regression note + no-`SeverLinks`-in-ablation assert (Task 8) · classify-then-recompute ordering oracle (Task 7) · `Valid` hardcoded true, stationary-probe documented (Task 2) · loop order reconciled (Task 8 review).
