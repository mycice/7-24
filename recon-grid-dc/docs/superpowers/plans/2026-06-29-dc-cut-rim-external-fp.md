# Cut-Rim External Feature Point Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restore the paper's two feature-point split at cut boundary voxels so the OUTER skin at the cut margin follows the model's original isosurface contour (round sphere → round; liver → liver-shaped), while the cut WALL keeps the cut-point centroid — fixing the faceted/"square" cut cross-section.

**Architecture:** The paper (Table 3, clean:1340-1344) carries TWO feature points per cut boundary voxel — an **External** FP on the outer surface (the DC/QEF isosurface point, Eqs 1-5, clean:232-296) and an **Internal** FP on the cut/inner surface (the cut-point centroid, clean:343-344). The impl collapsed both into one `_CutFP` and pointed the outer skin at it; the later pure-centroid change put `_CutFP` on the flat blade plane → faceted rim. This plan adds a parallel `_CutFPExternal` buffer holding the per-component isosurface QEF, gives the outer skin a distinct tag bit so it resolves to `_CutFPExternal`, and leaves `_CutFP` (centroid) and the cut wall untouched.

**Tech Stack:** Unity 2021.3 Built-in pipeline, GPU ComputeShader (HLSL SM5), C# (ReconBuffers / DualContouring orchestration). Tests = GPU oracle tests under `Assets/ReconGridDC/Tests/` (run in the Unity Editor Test Runner; `[Category("GPU")]`).

## Global Constraints

- **STRICT paper alignment (math + architecture):** External FP = the Dual-Contouring/Hermite QEF of paper Eqs 1-5 (clean:232-296) — minimise `E(x)=Σ n_i·dot(n_i, p_i − x)` by the same D1 gradient descent / Eq4 decaying step / D5 box-SDF stop that `DC_FeaturePoints` (Recon.compute:126-150) already uses, run over THAT component's isosurface-intersection planes, seeded at the component's `isectMean`, with **NO** `_CutPull` term. Internal/cut-wall FP = the centroid of the component's cut points (clean:343-344) — UNCHANGED.
- **NOT a circle-fit / shape-regularisation.** The External FP follows the model's actual surface (normals + intersections). It must work for an irregular model (liver), not only the sphere. No smoothing toward a regular shape.
- **No regression of the holes/jitter fix (workflow wggai2v70):** do not modify `_CutFP` (the Internal centroid) or `BuildCutTriangles` (the cut wall). The External FP is always valid for a boundary component (it has isect planes) so its `.w` never flips → it cannot re-introduce the dropped-quad holes.
- **No regression of non-cut surface:** non-cut voxels keep using `_VoxelExternalFP`. Zero-cut path byte-identical.
- Mirror existing in-project patterns: buffer alloc/zero-init/dispose like `CutFP` (ReconBuffers.cs); bind sets like `_CutFP` (DualContouring.BindCut); slot domain = `8*VoxelCount`.

---

## File Structure

- `Assets/ReconGridDC/Recon/ReconBuffers.cs` — add `public ComputeBuffer CutFPExternal;` (8·VoxelCount × 16, zero-init in Upload, dispose). Mirrors `CutFP`.
- `Assets/ReconGridDC/Shaders/Cutting.compute` — re-declare `int _QefIters; float _L;` (removed in the pure-centroid cleanup; still bound by DualContouring as no-ops, now live again); declare `RWStructuredBuffer<float4> _CutFPExternal;`; in `ComputeComponentFP` add the per-component isosurface QEF → `_CutFPExternal[slot]`.
- `Assets/ReconGridDC/Shaders/Recon.compute` — declare `StructuredBuffer<float4> _CutFPExternal;`; `StitchVertId` checks `_CutFPExternal[slot].w` and emits the NEW external-cut tag `0x40000000|slot`; `FetchPos` resolves bit30 → `_CutFPExternal`.
- `Assets/ReconGridDC/Recon/DualContouring.cs` — bind `_CutFPExternal` to `ComputeComponentFP` (RW writer) and to `DC_Stitch` + `DC_NormalsScatter` (SRV readers, via BindCut/the recon binds).
- `Assets/ReconGridDC/Tests/CutRimExternalFP_GpuOracle_Tests.cs` (+ `.meta`) — new GPU oracle.

---

### Task 1: Add the `_CutFPExternal` buffer + bindings (no behaviour change yet)

**Files:**
- Modify: `Assets/ReconGridDC/Recon/ReconBuffers.cs` (mirror the `CutFP` field/alloc/upload/dispose)
- Modify: `Assets/ReconGridDC/Shaders/Cutting.compute` (declare `RWStructuredBuffer<float4> _CutFPExternal;` next to `_CutFP`)
- Modify: `Assets/ReconGridDC/Shaders/Recon.compute` (declare `StructuredBuffer<float4> _CutFPExternal;` next to the `_CutFP` declaration)
- Modify: `Assets/ReconGridDC/Recon/DualContouring.cs` (bind it everywhere `_CutFP` is bound for the cut-FP kernels + DC_Stitch/DC_NormalsScatter)

**Interfaces:**
- Produces: `ReconBuffers.CutFPExternal` (ComputeBuffer, `float4[8*VoxelCount]`, stride 16, zero-init). HLSL `_CutFPExternal` (RW in Cutting.compute, SRV in Recon.compute).

- [ ] **Step 1:** In ReconBuffers.cs, find the `CutFP` field declaration and add directly below it:
```csharp
public ComputeBuffer CutFPExternal; // 2D-rim fix: per-component EXTERNAL feature point (isosurface QEF) for the OUTER skin
```
- [ ] **Step 2:** Find the `CutFP = new ComputeBuffer(Math.Max(1, slotCount), 16);` allocation and add directly below it (same `slotCount = 8*VoxelCount`):
```csharp
CutFPExternal = new ComputeBuffer(Math.Max(1, slotCount), 16);
```
- [ ] **Step 3:** In `Upload(...)`, find the explicit `CutFP.SetData(new float4[Math.Max(1, slotCount)]);` zero-init and add directly below it:
```csharp
CutFPExternal.SetData(new float4[Math.Max(1, slotCount)]); // clean init: w=0 invalid until ComputeComponentFP writes it
```
- [ ] **Step 4:** In `Dispose()`, find `CutFP?.Dispose();` and add directly below it:
```csharp
CutFPExternal?.Dispose();
```
- [ ] **Step 5:** In Cutting.compute, find `RWStructuredBuffer<float4> _CutFP;` and add directly below it:
```hlsl
RWStructuredBuffer<float4> _CutFPExternal; // 2D-rim: per-component EXTERNAL feature point (isosurface QEF) — paper Table 3
```
  Also re-add (they were removed in the pure-centroid cleanup but are still SetInt/SetFloat-bound by DualContouring; needed again for the External-FP QEF box-clamp), near the other scalar uniforms:
```hlsl
int   _QefIters; // QEF iteration count (mirrors DC_FeaturePoints) — used by the External-FP QEF
float _L;        // voxel side length (D5 BoxSdf box extent) — used by the External-FP QEF
```
- [ ] **Step 6:** In Recon.compute, find the `_CutFP` declaration (the one read by FetchPos, near `_VoxelExternalFP`) and add directly below it:
```hlsl
StructuredBuffer<float4> _CutFPExternal; // 2D-rim: per-component EXTERNAL feature point read by the outer skin
```
  NOTE: Recon.compute reads `_CutFP` as `StructuredBuffer<float4>` (it does not write it); declare `_CutFPExternal` the same way.
- [ ] **Step 7:** In DualContouring.cs, find every `SetBuffer(..., "_CutFP", rb.CutFP)` and add an adjacent `"_CutFPExternal", rb.CutFPExternal` bind for the SAME kernel. Specifically: the `ComputeComponentFP` bind (writer), and the recon binds used by `DC_Stitch` + `DC_NormalsScatter` (readers — wherever `_CutFP` is bound for the stitch/normals path, e.g. BindCut and the DC_NormalsScatter bind block). The cut-FP kernel that WRITES it (ComputeComponentFP) needs it bound as the RW target.
- [ ] **Step 8 (verify, no test):** Re-read each edit; confirm `_CutFPExternal` is declared once per shader, allocated/zeroed/disposed once, and bound to ComputeComponentFP (write) + DC_Stitch/DC_NormalsScatter (read). No behaviour change yet (nothing reads a non-zero `_CutFPExternal`). Commit.
```bash
git add Assets/ReconGridDC/Recon/ReconBuffers.cs Assets/ReconGridDC/Shaders/Cutting.compute Assets/ReconGridDC/Shaders/Recon.compute Assets/ReconGridDC/Recon/DualContouring.cs
git commit -m "feat(reconDC): add _CutFPExternal buffer + bindings (2D-rim External FP scaffold)"
```

---

### Task 2: Compute the per-component External FP (isosurface QEF) in ComputeComponentFP

**Files:**
- Modify: `Assets/ReconGridDC/Shaders/Cutting.compute` — `ComputeComponentFP` (currently ~562-639)

**Interfaces:**
- Consumes: the existing per-component isect gather (`isectMean`, `nIncl`, `hasIsect`) + `conn`, `vCoord`, `off`, `cnt`, `_Isect`, `_IsectWorld`, `_VoxelCorner`, `_CornerPos`, `_QefIters`, `_L`, `BoxSdf` (from ReconCommon.hlsl, already included).
- Produces: `_CutFPExternal[slot]` = the isosurface-conforming External feature point per component (w=1 iff `hasIsect`, else w=0). `_CutFP[slot]` (centroid) UNCHANGED.

- [ ] **Step 1:** In `ComputeComponentFP`, after `Conn4096Gpu conn = _Conn4096[mask]; int compCount = conn.compCount;` and before the component loop, add the voxel center (needed for the D5 box-SDF; mirror DC_FeaturePoints:120-124):
```hlsl
    // Voxel center = mean of 8 deformed corners (D5 BoxSdf box center; mirror DC_FeaturePoints:120-124).
    float3 voxelCenter = (float3)0;
    [unroll] for (int cc = 0; cc < 8; cc++)
        voxelCenter += _CornerPos[_VoxelCorner[8 * (int)v + cc]];
    voxelCenter /= 8.0;
```
- [ ] **Step 2:** Inside the component loop, AFTER `bool hasIsect = (nIncl > 0); if (hasIsect) isectMean /= (float)nIncl;` and BEFORE the cut-centroid block, add the per-component EXTERNAL feature point (paper Eqs 1-5 — the DC/Hermite QEF over THIS component's isect planes; SAME machinery as DC_FeaturePoints, NO _CutPull, NO cut centroid):
```hlsl
        // ── EXTERNAL feature point (paper Table 3 "External feature point", Eqs 1-5 clean:232-296) ─────
        // The OUTER-skin vertex for this component: the Dual-Contouring/Hermite QEF over THIS component's
        // isosurface-intersection planes (the same D1 gradient / Eq4 decay / D5 box-SDF stop that
        // DC_FeaturePoints:126-150 uses), seeded at the component's isectMean. It conforms to the ORIGINAL
        // isosurface (round sphere → round; liver → liver-shaped) — it is NOT a circle fit. NO _CutPull and
        // NO cut centroid (that is the Internal/cut-wall FP below). Always valid when the component has
        // isect planes, so its .w never flips → it cannot re-introduce the wggai2v70 holes.
        if (hasIsect)
        {
            float3 xe = isectMean;                                  // Eq2 seed = component isect mean
            for (int ite = 0; ite < _QefIters; ite++)
            {
                float3 Fe = (float3)0;                              // Eq1/D1: F = Σ n_i·dot(n_i, p_i − x)
                for (int se = 0; se < cnt; se++)
                {
                    IsectGpu ie = _Isect[off + se];
                    int icE = (ie.insideA == 1) ? ie.cornerA : ie.cornerB;     // inside endpoint (B1)
                    int lcE = LocalCornerOf(CornerCoordHLSL(icE), vCoord);
                    if (lcE < 0) continue;
                    if (Conn4096Comp(conn, lcE) != comp) continue;            // only THIS component's planes
                    float3 p_i = _IsectWorld[off + se];                        // deformed isect position
                    float3 n_i = ie.normal;                                    // rest normal (PAPER-SILENT frozen, as DC_FeaturePoints)
                    Fe += n_i * dot(n_i, p_i - xe);
                }
                float ae  = 0.1 * (1.0 - (float)ite / (float)_QefIters);       // Eq4 decaying step
                float3 xn = xe + ae * Fe;
                if (BoxSdf(xn, voxelCenter, _L) >= 0.0) break;                 // D5: keep last interior iterate
                xe = xn;
            }
            _CutFPExternal[slot] = float4(xe, 1.0);                            // External FP valid
        }
        else
        {
            _CutFPExternal[slot] = float4(0, 0, 0, 0);                         // no outer surface for this component → no skin vertex
        }
```
- [ ] **Step 3:** Leave the existing `_CutFP[slot] = hasCut ? float4(cutC,1) : float4(isectMean,1);` (the Internal/cut-wall centroid) UNCHANGED. Re-read the function: the invalid-component early-out (`!hasIsect && !hasCut`) must ALSO write `_CutFPExternal[slot]=0` — confirm that branch sets both (add `_CutFPExternal[slot] = float4(0,0,0,0);` in the `if (!hasIsect && !hasCut)` block before `continue;`).
- [ ] **Step 4:** Commit.
```bash
git add Assets/ReconGridDC/Shaders/Cutting.compute
git commit -m "feat(reconDC): ComputeComponentFP writes per-component External FP (isosurface QEF, paper Eqs 1-5)"
```

---

### Task 3: Resolve the OUTER skin to the External FP (new tag bit)

**Files:**
- Modify: `Assets/ReconGridDC/Shaders/Recon.compute` — `StitchVertId` (215-227) + `FetchPos` (325-329)

**Interfaces:**
- Consumes: `_CutFPExternal` (Task 1/2), the existing tag scheme (bit31 `0x80000000` = Internal `_CutFP`; no high bit = `_VoxelExternalFP`).
- Produces: new tag bit `0x40000000` = External cut FP (`_CutFPExternal`). Outer skin at cut voxels resolves to it.

- [ ] **Step 1:** Change `StitchVertId` (Recon.compute:215-227) so the outer skin tags the EXTERNAL feature point (not the Internal `_CutFP`). Replace the validity check + return:
```hlsl
    int comp = Conn4096Comp(_Conn4096[mask], lc);
    int slot = 8 * voxelId + comp;
    // 2D-rim: the OUTER skin uses the per-component EXTERNAL feature point (isosurface QEF, round contour),
    // NOT the Internal cut centroid (_CutFP, flat blade plane). Tag bit30 (0x40000000) ⇒ _CutFPExternal.
    // Fall back to the per-voxel external FP if this component has no External FP (degenerate) → no hole.
    return (_CutFPExternal[slot].w > 0.5) ? (int)(0x40000000u | (uint)slot) : voxelId;
```
- [ ] **Step 2:** Update `FetchPos` (Recon.compute:325-329) to resolve the new bit30 tag. Replace the body:
```hlsl
float3 FetchPos(int raw)
{
    uint  u   = (uint)raw;
    bool  internalCut = (u & 0x80000000u) != 0u;   // cut WALL vertex → _CutFP (Internal centroid)
    bool  externalCut = (u & 0x40000000u) != 0u;   // cut SKIN vertex → _CutFPExternal (isosurface QEF)
    int   idx = (int)(u & 0x3FFFFFFFu);            // strip both high bits → slot (8v+comp) or voxelId
    if (internalCut) return _CutFP[idx].xyz;
    if (externalCut) return _CutFPExternal[idx].xyz;
    return _VoxelExternalFP[idx].xyz;              // normal surface vertex
}
```
  NOTE: the cut WALL (`BuildCutTriangles`) still emits `0x80000000|slot` → `_CutFP` (Internal centroid) UNCHANGED; only the SKIN (`StitchVertId`) now emits `0x40000000|slot` → `_CutFPExternal`. The two surfaces meet at the round isosurface∩cut-plane rim.
- [ ] **Step 3:** Confirm no other reader assumes bit31 is the only high tag bit. Grep Recon.compute + Cutting.compute for `0x80000000` / `0x7FFFFFFF` and verify: BuildCutTriangles' internal tag (Cutting.compute) stays `0x80000000` (wall); any mask that was `& 0x7FFFFFFF` to strip bit31 must become `& 0x3FFFFFFF` to also strip bit30 (FetchPos already updated; check DC_NormalsScatter's tag-decode if it strips bits independently).
- [ ] **Step 4:** Commit.
```bash
git add Assets/ReconGridDC/Shaders/Recon.compute
git commit -m "feat(reconDC): outer skin resolves to External FP at cut voxels (bit30 tag) — round cut rim"
```

---

### Task 4: GPU oracle test + regression

**Files:**
- Create: `Assets/ReconGridDC/Tests/CutRimExternalFP_GpuOracle_Tests.cs` (+ `.cs.meta`, fresh 32-hex guid)

**Interfaces:**
- Consumes: ComputeComponentFP (writes `_CutFPExternal`), the recon binds.

- [ ] **Step 1:** Mirror an existing GPU oracle (e.g. `CutSurface_GpuOracle_Tests.cs` / `CutBitPropagation_GpuOracle_Tests.cs`) for shader load + buffer setup + dispatch + GetData. Build a boundary cut voxel: a level set whose isosurface passes through a voxel + a planar cut through it (reuse the `BuildPlanarCut`/sphere helpers). Dispatch ClearCutFP → LookupConnectivity → AccumulateCutFP → ComputeComponentFP, then GetData `CutFP` + `CutFPExternal`.
- [ ] **Step 2:** Assert, for a boundary cut voxel with both isect planes and cut points:
  - `CutFPExternal[slot].w == 1` and its xyz lies on the analytic isosurface within tol (e.g. for a sphere of radius R centred c, `abs(length(fp-c)-R) < tol`) — i.e. it CONFORMS to the original contour, NOT the blade plane.
  - `CutFP[slot].w == 1` and its xyz lies on the blade plane (y == bladeY within tol) — UNCHANGED.
  - `CutFPExternal[slot]` != `CutFP[slot]` (the two FPs are distinct).
  - For a voxel split into 2 components, the two components' `CutFPExternal` are distinct (the skin splits).
  - A pure-interior cut component (cut, no isect planes) → `CutFPExternal[slot].w == 0` (no skin vertex).
- [ ] **Step 3:** Regression: keep an assertion (or a separate test) that `CutFP` (the Internal centroid) is byte-identical to before this change for the same input (the cut wall is unaffected). Confirm the existing `CutSurface_GpuOracle_Tests` still passes (it reads `_CutFP` for the wall + must now bind `_CutFPExternal`; if that oracle dispatches DC_Stitch/FetchPos, bind `rb.CutFPExternal` there too, zero-init → degenerate-fallback to `voxelId` keeps it valid).
- [ ] **Step 4:** Create the `.cs.meta` with a fresh 32-hex guid (mirror an existing test's `.meta`). Commit.
```bash
git add Assets/ReconGridDC/Tests/CutRimExternalFP_GpuOracle_Tests.cs Assets/ReconGridDC/Tests/CutRimExternalFP_GpuOracle_Tests.cs.meta
git commit -m "test(reconDC): CutRimExternalFP oracle — External FP on isosurface, Internal on blade plane"
```

---

## Self-Review

**Spec coverage:** §2 root cause → Tasks 1-3 (decouple External/Internal). §3.2 External FP per-component isosurface QEF → Task 2 (paper Eqs 1-5, no _CutPull). §3.3 skin resolves to External → Task 3 (bit30 tag). §3.4 preserve-original-contour (not circle-fit) → Task 2's QEF over the actual isosurface (validated by the oracle's "on the analytic isosurface" assert, which works for any model). §4 no-regression → `_CutFP`/wall untouched (Tasks 2-3 leave them) + Task 4 regression. §5 testing → Task 4. §7 sub-decision (QEF vs isectMean): RESOLVED to the QEF (paper-faithful), seeded at isectMean so it degrades gracefully to the stable mean when under-constrained (cannot jitter worse than DC_FeaturePoints, and never flips validity for a boundary component).

**Placeholder scan:** none — every step shows the actual HLSL/C#.

**Type consistency:** `_CutFPExternal` is `RWStructuredBuffer<float4>` (Cutting.compute, writer) / `StructuredBuffer<float4>` (Recon.compute, reader); C# `ComputeBuffer CutFPExternal` stride 16; tag bit30 `0x40000000`, mask `0x3FFFFFFF`. Consistent across tasks.

**Open item for the audit:** confirm the External-FP QEF is jitter-free in practice (same machinery as the stable DC_FeaturePoints; validity never flips for a boundary component) and that no other tag-decode site assumed bit31 was the only high bit (Task 3 Step 3).

---

## 8. REWORK REVISIONS (audit wnq0vyv8f — REWORK; core math APPROVED, scope was incomplete) — BINDING

The audit confirmed the External FP = the paper Eqs 1-5 QEF (byte-identical machinery to DC_FeaturePoints, restricted to the component's isect planes, seeded at isectMean, NO _CutPull) and that it **preserves the model's original contour for ANY model (liver included), not a circle-fit** — KEEP the per-component QEF, NOT isectMean. But the plan missed the render path + the normal source + several tag-decode sites. These additions are BINDING and override/extend the tasks above.

### R0 — Repo + uniforms
- ALL edits target **`D:\Desktop\Tissue_Simulation\SurgicalSim_DC`** (the cutting code lives here on `main`). The cwd `…\Simulation` is a Stage-1A-only tree with NO cutting code — edits there silently no-op.
- Task 1 Step 5: re-add ONLY `int _QefIters;` (still SetInt-bound by DualContouring.cs:158). Do **NOT** re-add `_L` — reuse the already-declared `float _VoxelL;` (Cutting.compute:69, == L) for the QEF box-clamp: `BoxSdf(xn, voxelCenter, _VoxelL)`. (Avoids a redundant L uniform; confirm _QefIters is non-zero else 0 iters → External FP == isectMean.)

### R1 — External-FP QEF: declarations + safety (Task 2)
- Add a PAPER-SILENT declaration comment: "Per-component restriction of the External FP is paper-silent — the paper (clean:314) computes the outer-surface FP once per voxel and scopes the per-connected-component split (clean:342-348) only to the INTERNAL/cut FP. We mirror the split onto the External FP so the outer skin OPENS at the cut (project RC1). Degrades to the single per-voxel DC_FeaturePoints result when compCount==1." Also note: "Fe = n_i·dot(n_i, p_i−x) is DC_FeaturePoints' D1 projected gradient (the exact −∇ of Eq1); paper Eq3's `n_i²` is the rank-1 outer-product projection, NOT a scalar square."
- Safety (explicit determinism): if `nIncl < 2` set `_CutFPExternal[slot] = float4(isectMean, 1.0)` and skip the QEF loop (a rank-deficient 1-plane component cannot constrain the QEF; the seed is already on the surface). nIncl≥2 → run the QEF.
- Do NOT extend InterpCutFP (2D-2b EMA) to `_CutFPExternal` — the External FP is the stable isosurface QEF; smoothing it would lag the rim behind the deforming surface (contradicts the track-the-contour requirement).

### R2 — The bit30 tag must be decoded at ALL FOUR sites (Task 3 expands)
Tag scheme: bit31 `0x80000000`=Internal cut wall (`_CutFP`); bit30 `0x40000000`=External cut skin (`_CutFPExternal`); no high bit=per-voxel external (`_VoxelExternalFP`); index mask `& 0x3FFFFFFF`. The SKIN vertex reuses the EXISTING per-voxel external NORMAL pipeline via `slot>>3` (= voxelId) — NO new normal buffers.

- **(a) FetchPos** (Recon.compute:325-329) — already in Task 3 Step 2 (three-way decode, mask 0x3FFFFFFF). Keep.
- **(b) ReconSurface.shader** (the on-screen vertex shader, ~:45-51) — CRITICAL, the actual draw path (ReconGridManager DrawProceduralIndirect). Currently `idx=raw&0x7FFFFFFF; isInternal=(raw&0x80000000); wp = isInternal?_CutFP[idx]:_VoxelExternalFP[idx]; n = isInternal?_CutFPNormalF[idx]:_ExtNormalF[idx];`. Change to:
```hlsl
uint u = (uint)raw;
int idx = (int)(u & 0x3FFFFFFFu);
if (u & 0x80000000u)      { wp = _CutFP[idx].xyz;          n = _CutFPNormalF[idx]; }   // internal cut WALL
else if (u & 0x40000000u) { wp = _CutFPExternal[idx].xyz;  n = _ExtNormalF[idx >> 3]; } // external cut SKIN — pos = isosurface QEF; normal = per-voxel external normal (voxel = slot>>3)
else                      { wp = _VoxelExternalFP[idx].xyz; n = _ExtNormalF[idx]; }      // normal surface
```
  Declare `StructuredBuffer<float4> _CutFPExternal;` in ReconSurface.shader. Bind `mat.SetBuffer("_CutFPExternal", rb.CutFPExternal)` in ReconGridManager where `_CutFP`/`_CutFPNormalF` are bound to the material (~ReconGridManager.cs:211-216).
- **(c) DC_NormalsScatter** (Recon.compute:372-377) — currently `i0=raw0&0x7FFFFFFF; if(raw0&0x80000000) AtomicAddCutNormal(i0,fn); else AtomicAddNormal(i0,fn);`. Change the mask to `& 0x3FFFFFFF` and add the bit30 branch routing the SKIN face normal into the per-voxel external accumulator at `slot>>3`:
```hlsl
uint u0 = (uint)raw0; int i0 = (int)(u0 & 0x3FFFFFFFu);
if      (u0 & 0x80000000u) AtomicAddCutNormal(i0, fn);        // internal cut WALL → _CutFPNormal[slot]
else if (u0 & 0x40000000u) AtomicAddNormal(i0 >> 3, fn);      // external cut SKIN → _ExtNormal[voxel]=slot>>3
else                       AtomicAddNormal(i0, fn);            // normal surface → _ExtNormal[voxel]
```
  Apply the SAME three-way decode to all three face vertices (raw0/raw1/raw2). Confirm DC_NormalsNormalize normalizes `_ExtNormal` for cut voxels too (it is per-voxel → it does).
- **StitchVertId** (Recon.compute:215-227) — Task 3 Step 1: emit `0x40000000|slot` gated on `_CutFPExternal[slot].w>0.5` (already specified). Keep.

### R3 — Bindings (Task 1 Step 7 made explicit)
Bind `_CutFPExternal` in BOTH:
- `DualContouring.Bind` (:253) — the SRV reader set used by DC_Stitch + DC_NormalsScatter (and a zero-cut Build dispatches these, so the SRV must resolve even with no cut).
- `DualContouring.BindCut` (:278) — the RW writer for ComputeComponentFP.
And the material bind in ReconGridManager (R2(b)).

### R4 — Tests (Task 4 expands — these are REQUIRED edits, not "should still pass")
- The new oracle `CutRimExternalFP_GpuOracle_Tests` (Task 4) — keep; bind `_CutFPExternal` (RW for ComputeComponentFP) in its setup.
- **CutSurface_GpuOracle_Tests**: (i) add `_CutFPExternal` to its `BindCut` (~:180) AND `BindReconAll` (~:263). (ii) UPDATE Test 5 (RC1 skin-splits, ~:475-481): the skin tag is now bit30 — decode `if ((raw & 0x40000000)==0) continue; int slot = raw & 0x3FFFFFFF;` and assert the slot's `rb.CutFPExternal[slot].w>0.5` (NOT `_CutFP`). Fix the `Assert.Less(slot, 8*VoxelCount)` (~:547) to mask `& 0x3FFFFFFF`.
- **Stitch_GpuOracle_Tests**: the range-invariant (~:68, "every (idx&0x7FFFFFFF) in [0,VoxelCount)") must mask `& 0x3FFFFFFF` and accept the bit30 (slot-domain) tag — update its decode (~:103-112). Bind `_CutFPExternal` if it dispatches DC_Stitch/DC_NormalsScatter (Stitch Test 1 runs a zero-cut Build).
- Re-run both oracles green after the edits.
