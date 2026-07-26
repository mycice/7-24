# DC Stage 1A — Static DC Outer-Surface Reconstruction — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reconstruct a watertight, correctly-oriented Dual-Contouring outer surface of a *static* procedural primitive (box/sphere) on a uniform hexahedral voxel grid, rendered via GPU, with every numeric piece verifiable.

**Architecture:** CPU C# preprocessing (grid build → corner inside/outside → intersecting edges → Hermite intersection points, per-voxel-compacted) uploads to GPU `ComputeBuffer`s; three HLSL ComputeShader kernels (QEF feature points → stitch triangles → per-vertex normals) build the surface; `Graphics.DrawProceduralIndirect` renders it. No physics (corners are fixed) — this is the standalone static milestone of design §4.

**Tech Stack:** Unity 2021.3.45f2, C#, HLSL ComputeShader (Shader Model 5.0), Unity Test Framework (EditMode, NUnit).

## Global Constraints

- Authoritative design spec: `docs/superpowers/specs/2026-06-27-dc-cutting-design.md` (v4). This plan implements **§4.1–§4.3 for the static case only** (no §4.4/§4.5 physics, no RebuildIsectWorld — corners do not move in 1A).
- New self-contained folder: `Assets/SurgicalSim/ReconGridDC/`. Do NOT reference or reuse `MC2024/`, `Cutting/`, `CuttingV2/`, `CuttingV3/`, `Physics/XPBD*`.
- Strict paper alignment; the only permitted deviations are **D1** (QEF gradient `F=Σ n_i·(n_i·(p_i−x_k))` at current `x_k`) and **D5** (Eq5 stop when `boxSDF(x_next)≥0`, keep last interior `x_prev`). Mark each in code with `// DEVIATION D1` / `// DEVIATION D5`.
- Frozen index conventions (design §1.3, Fig 2.4): corners V0..V7, edges e0..e11 exactly as tabulated in Task 2. The 4096 LUT is Stage 2 — not here.
- GPU float-atomic reality (design §1.3): SM5.0 has no float `InterlockedAdd`; normal accumulation uses **fixed-point int3 + InterlockedAdd**, scale `1<<20`, normalized in a post-barrier pass.
- Sign convention: inside `φ<0`, outside `φ>0`, isosurface `φ=0`.
- Voxel size `L = 0.25` cm (design §7). The primitive must sit strictly inside the grid with ≥1 voxel of padding on every side, so every intersecting grid edge has all 4 incident voxels present (no partial quads in 1A).
- Units `int3 gridDims = (Nx,Ny,Nz)` = voxel counts per axis; corners per axis = `Nx+1` etc.
- Verification model (this environment cannot compile/run Unity): pure-C# units get directly-runnable EditMode NUnit tests; HLSL kernels get EditMode tests that dispatch the kernel on a tiny known input and assert hand-computed values (the user runs these on a GPU machine). After each task, a subagent statically audits the code against this plan + the spec.
- TDD: write the failing test, see it fail, implement minimally, see it pass, commit. Frequent commits.

---

## File Structure

```
Assets/SurgicalSim/ReconGridDC/
  ReconGridDC.asmdef                       // runtime assembly
  Core/
    GridConventions.cs                     // Fig 2.4 numbering, voxelId/cornerId encode+decode, edge→corners, per-axis 4-voxel stencil, boxSDF
    FixedPointAtomic.cs                     // float↔int fixed-point scale constants (+ HLSL include twin)
  Preprocess/
    ILevelSetProvider.cs                   // Sample(float3)->float ; Gradient(float3)->float3
    PrimitiveLevelSets.cs                  // BoxLevelSet, SphereLevelSet (analytic SDF + gradient)
    BackgroundGrid.cs                      // CPU: build corners, classify, intersecting edges, Hermite, per-voxel isect compaction
  Recon/
    ReconBuffers.cs                        // allocates/owns the Stage-1A ComputeBuffers; uploads CPU preprocessing output
    DualContouring.cs                      // C# orchestrator: dispatch QEF→Stitch→Normals, CopyCount
  Shaders/
    ReconCommon.hlsl                       // shared structs, fixed-point helpers, boxSDF, decode helpers
    Recon.compute                          // kernels: DC_FeaturePoints, DC_Stitch, DC_NormalsScatter, DC_NormalsNormalize
    ReconSurface.shader                    // DrawProceduralIndirect vertex/fragment (fetch FP pos+normal by triBuf index)
  Demo/
    ReconGridManager.cs                    // MonoBehaviour: preprocess→upload→dispatch(once)→draw each frame
    DemoSceneSetup.cs                      // builds a box/sphere demo at runtime
  Tests/
    ReconGridDC.Tests.asmdef               // EditMode test assembly (refs ReconGridDC + nunit)
    GridConventions_Tests.cs
    LevelSet_Tests.cs
    BackgroundGrid_Tests.cs
    Qef_GpuOracle_Tests.cs                 // dispatches DC_FeaturePoints on a 1-voxel plane
    Stitch_GpuOracle_Tests.cs             // dispatches DC_Stitch on a known small grid
```

---

### Task 1: Folder scaffold + assembly definitions

**Files:**
- Create: `Assets/SurgicalSim/ReconGridDC/ReconGridDC.asmdef`
- Create: `Assets/SurgicalSim/ReconGridDC/Tests/ReconGridDC.Tests.asmdef`

**Interfaces:**
- Produces: assembly `ReconGridDC` (runtime) and `ReconGridDC.Tests` (EditMode) that all later tasks compile into.

- [ ] **Step 1: Create the runtime asmdef**

`ReconGridDC.asmdef`:
```json
{
  "name": "ReconGridDC",
  "rootNamespace": "ReconGridDC",
  "references": [],
  "includePlatforms": [],
  "excludePlatforms": [],
  "allowUnsafeCode": false,
  "autoReferenced": true
}
```

- [ ] **Step 2: Create the EditMode test asmdef**

`Tests/ReconGridDC.Tests.asmdef`:
```json
{
  "name": "ReconGridDC.Tests",
  "rootNamespace": "ReconGridDC.Tests",
  "references": ["ReconGridDC"],
  "includePlatforms": ["Editor"],
  "precompiledReferences": ["nunit.framework.dll"],
  "defineConstraints": ["UNITY_INCLUDE_TESTS"],
  "overrideReferences": true,
  "autoReferenced": false
}
```

- [ ] **Step 3: Commit**

```bash
git add Assets/SurgicalSim/ReconGridDC
git commit -m "feat(reconDC): scaffold ReconGridDC folder + asmdefs (Stage 1A)"
```

---

### Task 2: GridConventions — frozen Fig 2.4 numbering, indexing, stencils, boxSDF

**Files:**
- Create: `Assets/SurgicalSim/ReconGridDC/Core/GridConventions.cs`
- Test: `Assets/SurgicalSim/ReconGridDC/Tests/GridConventions_Tests.cs`

**Interfaces:**
- Produces:
  - `int3 GridConventions.GridDims` field-free static helpers taking dims as params:
  - `int CornerId(int i,int j,int k, int3 dims)` ; `int VoxelId(int i,int j,int k, int3 dims)`
  - `static readonly int3[] CornerOffset` (length 8) ; `static readonly int[,] EdgeCorners` (12×2)
  - `struct EdgeStencilEntry { public int3 voxelOffset; public int localEdge; }`
  - `static readonly EdgeStencilEntry[][] EdgeStencil` indexed `[axis][0..3]`, axis 0=x,1=y,2=z (the 4 voxels incident to a grid edge of that axis + their local edge id)
  - `static readonly int3[] QuadCornerOrder` per axis is folded into `EdgeStencil` order (the 4 entries are already in CCW order around +axis)
  - `float BoxSdf(float3 p, float3 center, float L)`

- [ ] **Step 1: Write the failing test**

`GridConventions_Tests.cs`:
```csharp
using NUnit.Framework;
using Unity.Mathematics;     // if math pkg absent, use UnityEngine.Vector3Int / Vector3 and adjust signatures
using ReconGridDC.Core;

public class GridConventions_Tests
{
    [Test] public void EdgeCorners_MatchFig24()
    {
        // bottom loop
        Assert.AreEqual((0,1),(GridConventions.EdgeCorners[0,0],GridConventions.EdgeCorners[0,1]));
        Assert.AreEqual((1,2),(GridConventions.EdgeCorners[1,0],GridConventions.EdgeCorners[1,1]));
        Assert.AreEqual((2,3),(GridConventions.EdgeCorners[2,0],GridConventions.EdgeCorners[2,1]));
        Assert.AreEqual((3,0),(GridConventions.EdgeCorners[3,0],GridConventions.EdgeCorners[3,1]));
        // verticals
        Assert.AreEqual((0,4),(GridConventions.EdgeCorners[8,0],GridConventions.EdgeCorners[8,1]));
        Assert.AreEqual((3,7),(GridConventions.EdgeCorners[11,0],GridConventions.EdgeCorners[11,1]));
    }

    [Test] public void VoxelId_RoundTrips()
    {
        int3 dims = new int3(4,5,6);
        int id = GridConventions.VoxelId(2,3,4,dims);
        Assert.AreEqual(2 + 4*(3 + 5*4), id);
    }

    [Test] public void XEdgeStencil_HasE0E2E4E6()
    {
        var s = GridConventions.EdgeStencil[0]; // x axis
        var locals = new int[]{s[0].localEdge,s[1].localEdge,s[2].localEdge,s[3].localEdge};
        CollectionAssert.AreEquivalent(new int[]{0,2,4,6}, locals);
    }

    [Test] public void BoxSdf_NegativeInsideZeroOnFace()
    {
        float3 c = new float3(0,0,0); float L = 2f; // half=1
        Assert.Less(GridConventions.BoxSdf(new float3(0,0,0),c,L), 0f);
        Assert.AreEqual(0f, GridConventions.BoxSdf(new float3(1,0,0),c,L), 1e-6f);
        Assert.Greater(GridConventions.BoxSdf(new float3(2,0,0),c,L), 0f);
    }
}
```

> If `Unity.Mathematics` is not in the project, replace `int3/float3` with `Vector3Int/Vector3` throughout this plan and drop the `using Unity.Mathematics;`. Decide once in this task and keep it consistent across all files.

- [ ] **Step 2: Run test to verify it fails**

Run (Unity Test Runner, EditMode) or: `Unity -runTests -testPlatform EditMode -testFilter GridConventions_Tests`
Expected: FAIL (GridConventions does not exist).

- [ ] **Step 3: Implement GridConventions**

`GridConventions.cs`:
```csharp
using Unity.Mathematics;

namespace ReconGridDC.Core
{
    public static class GridConventions
    {
        // Fig 2.4 local corner offsets (bottom z=0: V0..V3 CCW in xy; top z=1: V4..V7 above V0..V3)
        public static readonly int3[] CornerOffset =
        {
            new int3(0,0,0), new int3(1,0,0), new int3(1,1,0), new int3(0,1,0), // V0..V3
            new int3(0,0,1), new int3(1,0,1), new int3(1,1,1), new int3(0,1,1), // V4..V7
        };

        // 12 edges -> their two local corner ids (Fig 2.4)
        public static readonly int[,] EdgeCorners =
        {
            {0,1},{1,2},{2,3},{3,0},   // e0..e3 bottom loop
            {4,5},{5,6},{6,7},{7,4},   // e4..e7 top loop
            {0,4},{1,5},{2,6},{3,7},   // e8..e11 verticals
        };

        public struct EdgeStencilEntry { public int3 voxelOffset; public int localEdge; }

        // For a grid edge of a given axis at base corner C, the 4 incident voxels (offset from C's
        // voxel) and that edge's local id inside each. Entries are in CCW order viewed along +axis.
        public static readonly EdgeStencilEntry[][] EdgeStencil = new EdgeStencilEntry[][]
        {
            // axis 0 = x : edge C -> C+(1,0,0)
            new []{
                E(0,0,0, 0), E(0,-1,0, 2), E(0,-1,-1, 6), E(0,0,-1, 4),
            },
            // axis 1 = y : edge C -> C+(0,1,0)
            new []{
                E(0,0,0, 3), E(0,0,-1, 7), E(-1,0,-1, 5), E(-1,0,0, 1),
            },
            // axis 2 = z : edge C -> C+(0,0,1)
            new []{
                E(0,0,0, 8), E(-1,0,0, 9), E(-1,-1,0, 10), E(0,-1,0, 11),
            },
        };
        static EdgeStencilEntry E(int x,int y,int z,int le) =>
            new EdgeStencilEntry{ voxelOffset = new int3(x,y,z), localEdge = le };

        public static int CornerId(int i,int j,int k,int3 dims) =>
            i + (dims.x+1)*(j + (dims.y+1)*k);
        public static int VoxelId(int i,int j,int k,int3 dims) =>
            i + dims.x*(j + dims.y*k);
        public static int3 VoxelCoord(int voxelId,int3 dims) =>
            new int3(voxelId % dims.x, (voxelId/dims.x) % dims.y, voxelId/(dims.x*dims.y));

        // Eq5 box SDF, negative inside, zero on face. L = full edge length (half = L/2).
        public static float BoxSdf(float3 p, float3 center, float L)
        {
            float h = L*0.5f;
            float3 d = math.abs(p-center) - h;
            return math.max(math.max(d.x, d.y), d.z);
        }
    }
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: EditMode `GridConventions_Tests`. Expected: PASS (all 4 tests).

- [ ] **Step 5: Commit**

```bash
git add Assets/SurgicalSim/ReconGridDC/Core/GridConventions.cs Assets/SurgicalSim/ReconGridDC/Tests/GridConventions_Tests.cs
git commit -m "feat(reconDC): GridConventions (Fig2.4 numbering, voxel stencils, boxSDF) + tests"
```

---

### Task 3: Level-set providers (analytic SDF + gradient)

**Files:**
- Create: `Assets/SurgicalSim/ReconGridDC/Preprocess/ILevelSetProvider.cs`
- Create: `Assets/SurgicalSim/ReconGridDC/Preprocess/PrimitiveLevelSets.cs`
- Test: `Assets/SurgicalSim/ReconGridDC/Tests/LevelSet_Tests.cs`

**Interfaces:**
- Produces:
  - `interface ILevelSetProvider { float Sample(float3 p); float3 Gradient(float3 p); }`
  - `class SphereLevelSet : ILevelSetProvider` ctor `(float3 center, float radius)`
  - `class BoxLevelSet : ILevelSetProvider` ctor `(float3 center, float3 halfExtents)`
- Consumes: nothing.

- [ ] **Step 1: Write the failing test**

`LevelSet_Tests.cs`:
```csharp
using NUnit.Framework;
using Unity.Mathematics;
using ReconGridDC.Preprocess;

public class LevelSet_Tests
{
    [Test] public void Sphere_SignAndGradient()
    {
        var s = new SphereLevelSet(new float3(0,0,0), 1f);
        Assert.Less(s.Sample(new float3(0,0,0)), 0f);          // inside
        Assert.AreEqual(0f, s.Sample(new float3(1,0,0)), 1e-5f); // on surface
        Assert.Greater(s.Sample(new float3(2,0,0)), 0f);        // outside
        var g = s.Gradient(new float3(0.5f,0,0));
        Assert.AreEqual(1f, math.length(g), 1e-4f);            // unit gradient
        Assert.Greater(g.x, 0.99f);                            // points outward (+x)
    }

    [Test] public void Box_SignedDistance()
    {
        var b = new BoxLevelSet(new float3(0,0,0), new float3(1,1,1));
        Assert.AreEqual(-1f, b.Sample(new float3(0,0,0)), 1e-5f); // center, dist to nearest face = 1 inside => -1
        Assert.AreEqual(0f, b.Sample(new float3(1,0,0)), 1e-5f);
        Assert.AreEqual(1f, b.Sample(new float3(2,0,0)), 1e-5f);
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: EditMode `LevelSet_Tests`. Expected: FAIL (types missing).

- [ ] **Step 3: Implement providers**

`ILevelSetProvider.cs`:
```csharp
using Unity.Mathematics;
namespace ReconGridDC.Preprocess
{
    public interface ILevelSetProvider
    {
        float Sample(float3 p);     // signed: <0 inside, >0 outside
        float3 Gradient(float3 p);  // outward, ~unit length
    }
}
```

`PrimitiveLevelSets.cs`:
```csharp
using Unity.Mathematics;
namespace ReconGridDC.Preprocess
{
    public sealed class SphereLevelSet : ILevelSetProvider
    {
        readonly float3 c; readonly float r;
        public SphereLevelSet(float3 center, float radius){ c=center; r=radius; }
        public float Sample(float3 p) => math.length(p-c) - r;
        public float3 Gradient(float3 p)
        {
            float3 d = p-c; float len = math.length(d);
            return len > 1e-8f ? d/len : new float3(1,0,0);
        }
    }

    public sealed class BoxLevelSet : ILevelSetProvider
    {
        readonly float3 c, h;
        public BoxLevelSet(float3 center, float3 halfExtents){ c=center; h=halfExtents; }
        public float Sample(float3 p)
        {
            float3 q = math.abs(p-c) - h;
            float outside = math.length(math.max(q,0f));
            float inside  = math.min(math.max(q.x, math.max(q.y,q.z)), 0f);
            return outside + inside; // exact signed box distance
        }
        public float3 Gradient(float3 p)
        {
            // central difference (robust at edges/corners)
            const float e = 1e-3f;
            float dx = Sample(p+new float3(e,0,0)) - Sample(p-new float3(e,0,0));
            float dy = Sample(p+new float3(0,e,0)) - Sample(p-new float3(0,e,0));
            float dz = Sample(p+new float3(0,0,e)) - Sample(p-new float3(0,0,e));
            float3 g = new float3(dx,dy,dz);
            float len = math.length(g);
            return len > 1e-8f ? g/len : new float3(1,0,0);
        }
    }
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: EditMode `LevelSet_Tests`. Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add Assets/SurgicalSim/ReconGridDC/Preprocess/ILevelSetProvider.cs Assets/SurgicalSim/ReconGridDC/Preprocess/PrimitiveLevelSets.cs Assets/SurgicalSim/ReconGridDC/Tests/LevelSet_Tests.cs
git commit -m "feat(reconDC): ILevelSetProvider + Box/Sphere analytic level sets + tests"
```

---

### Task 4: BackgroundGrid — CPU preprocessing (corners, intersecting edges, Hermite, per-voxel isect)

**Files:**
- Create: `Assets/SurgicalSim/ReconGridDC/Preprocess/BackgroundGrid.cs`
- Test: `Assets/SurgicalSim/ReconGridDC/Tests/BackgroundGrid_Tests.cs`

**Interfaces:**
- Consumes: `ILevelSetProvider`, `GridConventions`.
- Produces (the CPU-side preprocessing output consumed by Task 5/6/7):
```csharp
public struct IsectEntry { public float3 globalRest; public float3 normal; public int cornerA; public int cornerB; public float t; }
public sealed class BackgroundGrid {
    public int3 dims; public float L; public float3 origin;          // origin = world pos of corner (0,0,0)
    public float3[] cornerPos;       // length (Nx+1)(Ny+1)(Nz+1)
    public byte[]  cornerInside;     // 1 if φ<0
    public int[]   voxelCorner;      // length 8*voxelCount : voxelCorner[8*v+k] = cornerId
    public int[]   voxelIsectOffset; // length voxelCount
    public int[]   voxelIsectCount;  // length voxelCount
    public IsectEntry[] isect;       // flat, per-voxel-compacted (shared edges duplicated into each voxel)
    public int voxelCount;           // dims.x*dims.y*dims.z
    public static BackgroundGrid Build(ILevelSetProvider ls, int3 dims, float L, float3 origin);
}
```

- [ ] **Step 1: Write the failing test**

`BackgroundGrid_Tests.cs`:
```csharp
using NUnit.Framework;
using Unity.Mathematics;
using ReconGridDC.Preprocess;

public class BackgroundGrid_Tests
{
    // Plane at x = 0.5 inside a 1x1x1-voxel grid spanning [0,1]^3, L=1, origin=0.
    sealed class PlaneX : ILevelSetProvider {
        public float Sample(float3 p)=>p.x-0.5f;
        public float3 Gradient(float3 p)=>new float3(1,0,0);
    }

    [Test] public void SingleVoxel_PlaneX_FourXEdgesIntersect()
    {
        var g = BackgroundGrid.Build(new PlaneX(), new int3(1,1,1), 1f, float3.zero);
        Assert.AreEqual(1, g.voxelCount);
        // 4 x-direction local edges (e0,e2,e4,e6) cross x=0.5 -> 4 isect entries in voxel 0
        Assert.AreEqual(4, g.voxelIsectCount[0]);
        foreach (var idx in System.Linq.Enumerable.Range(g.voxelIsectOffset[0], g.voxelIsectCount[0]))
        {
            Assert.AreEqual(0.5f, g.isect[idx].globalRest.x, 1e-5f);   // crossing at x=0.5
            Assert.AreEqual(1f, g.isect[idx].normal.x, 1e-4f);         // outward +x
        }
    }

    [Test] public void CornerInside_FollowsSign()
    {
        var g = BackgroundGrid.Build(new PlaneX(), new int3(1,1,1), 1f, float3.zero);
        // corner (0,0,0) at x=0 -> inside (φ=-0.5); corner (1,0,0) at x=1 -> outside
        Assert.AreEqual(1, g.cornerInside[ReconGridDC.Core.GridConventions.CornerId(0,0,0,g.dims)]);
        Assert.AreEqual(0, g.cornerInside[ReconGridDC.Core.GridConventions.CornerId(1,0,0,g.dims)]);
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: EditMode `BackgroundGrid_Tests`. Expected: FAIL (BackgroundGrid missing).

- [ ] **Step 3: Implement BackgroundGrid.Build**

`BackgroundGrid.cs`:
```csharp
using System.Collections.Generic;
using Unity.Mathematics;
using ReconGridDC.Core;

namespace ReconGridDC.Preprocess
{
    public struct IsectEntry { public float3 globalRest; public float3 normal; public int cornerA; public int cornerB; public float t; }

    public sealed class BackgroundGrid
    {
        public int3 dims; public float L; public float3 origin;
        public float3[] cornerPos; public byte[] cornerInside;
        public int[] voxelCorner; public int[] voxelIsectOffset; public int[] voxelIsectCount;
        public IsectEntry[] isect; public int voxelCount;

        public static BackgroundGrid Build(ILevelSetProvider ls, int3 dims, float L, float3 origin)
        {
            var g = new BackgroundGrid { dims=dims, L=L, origin=origin };
            int cnx=dims.x+1, cny=dims.y+1, cnz=dims.z+1;
            int cornerCount = cnx*cny*cnz;
            g.cornerPos = new float3[cornerCount];
            g.cornerInside = new byte[cornerCount];
            var phi = new float[cornerCount];
            for (int k=0;k<cnz;k++) for (int j=0;j<cny;j++) for (int i=0;i<cnx;i++)
            {
                int id = GridConventions.CornerId(i,j,k,dims);
                float3 p = origin + new float3(i,j,k)*L;
                g.cornerPos[id] = p;
                float v = ls.Sample(p);
                phi[id] = v;
                g.cornerInside[id] = (byte)(v < 0f ? 1 : 0);
            }

            g.voxelCount = dims.x*dims.y*dims.z;
            g.voxelCorner = new int[8*g.voxelCount];
            g.voxelIsectOffset = new int[g.voxelCount];
            g.voxelIsectCount  = new int[g.voxelCount];
            var flat = new List<IsectEntry>();

            for (int vk=0; vk<dims.z; vk++) for (int vj=0; vj<dims.y; vj++) for (int vi=0; vi<dims.x; vi++)
            {
                int v = GridConventions.VoxelId(vi,vj,vk,dims);
                // 8 corner ids
                for (int c=0;c<8;c++){
                    int3 o = GridConventions.CornerOffset[c];
                    g.voxelCorner[8*v+c] = GridConventions.CornerId(vi+o.x, vj+o.y, vk+o.z, dims);
                }
                g.voxelIsectOffset[v] = flat.Count;
                int cnt=0;
                for (int e=0;e<12;e++)
                {
                    int ca = g.voxelCorner[8*v + GridConventions.EdgeCorners[e,0]];
                    int cb = g.voxelCorner[8*v + GridConventions.EdgeCorners[e,1]];
                    bool ia = g.cornerInside[ca]==1, ib = g.cornerInside[cb]==1;
                    if (ia == ib) continue;                       // no sign change => no crossing (MC criterion)
                    float fa = phi[ca], fb = phi[cb];
                    float t = fa/(fa-fb);                          // zero-crossing param along A->B, in (0,1)
                    float3 pos = math.lerp(g.cornerPos[ca], g.cornerPos[cb], t);
                    float3 n = ls.Gradient(pos);
                    flat.Add(new IsectEntry{ globalRest=pos, normal=n, cornerA=ca, cornerB=cb, t=t });
                    cnt++;
                }
                g.voxelIsectCount[v] = cnt;
            }
            g.isect = flat.ToArray();
            return g;
        }
    }
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: EditMode `BackgroundGrid_Tests`. Expected: PASS (both tests).

- [ ] **Step 5: Commit**

```bash
git add Assets/SurgicalSim/ReconGridDC/Preprocess/BackgroundGrid.cs Assets/SurgicalSim/ReconGridDC/Tests/BackgroundGrid_Tests.cs
git commit -m "feat(reconDC): BackgroundGrid CPU preprocessing (corners/intersecting edges/Hermite) + tests"
```

---

### Task 5: FixedPointAtomic + ReconBuffers (GPU buffer allocation + upload)

**Files:**
- Create: `Assets/SurgicalSim/ReconGridDC/Core/FixedPointAtomic.cs`
- Create: `Assets/SurgicalSim/ReconGridDC/Recon/ReconBuffers.cs`

**Interfaces:**
- Consumes: `BackgroundGrid`.
- Produces: `class ReconBuffers : System.IDisposable` holding the Stage-1A `ComputeBuffer`s and an `Upload(BackgroundGrid g)` method, plus `int TriCapacity` and `ComputeBuffer IndirectArgs`.
  - Buffers (one struct/element layout each):
    - `cornerPos : ComputeBuffer<float3>` (StructuredBuffer, static corner world positions)
    - `voxelCorner : ComputeBuffer<int>` (8*voxelCount)
    - `voxelIsectOffset, voxelIsectCount : ComputeBuffer<int>`
    - `isect : ComputeBuffer<IsectGpu>` where `IsectGpu { float3 globalRest; float3 normal; int cornerA; int cornerB; float t; float _pad; }` (32 bytes, 16-byte friendly)
    - `voxelExternalFP : ComputeBuffer<float4>` (xyz=pos, w=valid)
    - `voxelExternalFPNormal : ComputeBuffer<int3>` represented as `ComputeBuffer<int>` length 3*voxelCount (fixed-point accumulator)
    - `tri : ComputeBuffer<int>` length `3*TriCapacity`, type `Append`? — use a plain RWStructuredBuffer with a separate `triCounter : ComputeBuffer<uint>` (1) we InterlockedAdd into (matches design §3.7), NOT Unity Append (so CopyCount is manual).
    - `indirectArgs : ComputeBuffer<uint>` length 4, `ComputeBufferType.IndirectArguments`
- `FixedPointAtomic`: `public const float NormalScale = 1<<20;` and matching HLSL define.

- [ ] **Step 1: Write FixedPointAtomic**

`FixedPointAtomic.cs`:
```csharp
namespace ReconGridDC.Core
{
    public static class FixedPointAtomic
    {
        public const float NormalScale = 1048576f; // 1<<20 ; HLSL twin in ReconCommon.hlsl
        public const float ForceScale  = 1024f;    // 1<<10 ; reserved for Stage 1B
    }
}
```

- [ ] **Step 2: Write the failing test (allocation + upload sizes)**

Add to `Tests/BackgroundGrid_Tests.cs` (or a new `ReconBuffers_Tests.cs`):
```csharp
[Test] public void ReconBuffers_AllocateUploadDispose()
{
    var g = BackgroundGrid.Build(new PlaneX(), new int3(2,2,2), 1f, Unity.Mathematics.float3.zero);
    using (var rb = new ReconGridDC.Recon.ReconBuffers(g, triCapacity: 4096))
    {
        rb.Upload(g);
        Assert.AreEqual(g.voxelCount, rb.VoxelCount);
        Assert.AreEqual(4096, rb.TriCapacity);
        Assert.IsNotNull(rb.IndirectArgs);
    }
}
```

- [ ] **Step 3: Run test to verify it fails**

Run: EditMode. Expected: FAIL (ReconBuffers missing).

- [ ] **Step 4: Implement ReconBuffers**

`ReconBuffers.cs`:
```csharp
using System;
using Unity.Mathematics;
using UnityEngine;
using ReconGridDC.Preprocess;

namespace ReconGridDC.Recon
{
    public sealed class ReconBuffers : IDisposable
    {
        public struct IsectGpu { public float3 globalRest; public float3 normal; public int cornerA; public int cornerB; public float t; public float _pad; }

        public int VoxelCount { get; private set; }
        public int TriCapacity { get; private set; }
        public ComputeBuffer CornerPos, VoxelCorner, VoxelIsectOffset, VoxelIsectCount, Isect;
        public ComputeBuffer VoxelExternalFP, VoxelExternalFPNormal, Tri, TriCounter, IndirectArgs;

        public ReconBuffers(BackgroundGrid g, int triCapacity)
        {
            VoxelCount = g.voxelCount; TriCapacity = triCapacity;
            int cc = g.cornerPos.Length;
            CornerPos = new ComputeBuffer(cc, 12);                       // float3
            VoxelCorner = new ComputeBuffer(8*VoxelCount, 4);
            VoxelIsectOffset = new ComputeBuffer(VoxelCount, 4);
            VoxelIsectCount  = new ComputeBuffer(VoxelCount, 4);
            Isect = new ComputeBuffer(Math.Max(1,g.isect.Length), 32);   // IsectGpu = 32 bytes
            VoxelExternalFP = new ComputeBuffer(VoxelCount, 16);         // float4
            VoxelExternalFPNormal = new ComputeBuffer(3*VoxelCount, 4);  // int fixed-point xyz
            Tri = new ComputeBuffer(3*TriCapacity, 4);                   // int indices
            TriCounter = new ComputeBuffer(1, 4);
            IndirectArgs = new ComputeBuffer(4, 4, ComputeBufferType.IndirectArguments);
        }

        public void Upload(BackgroundGrid g)
        {
            CornerPos.SetData(g.cornerPos);
            VoxelCorner.SetData(g.voxelCorner);
            VoxelIsectOffset.SetData(g.voxelIsectOffset);
            VoxelIsectCount.SetData(g.voxelIsectCount);
            var gpu = new IsectGpu[Math.Max(1,g.isect.Length)];
            for (int i=0;i<g.isect.Length;i++){ var e=g.isect[i];
                gpu[i]=new IsectGpu{ globalRest=e.globalRest, normal=e.normal, cornerA=e.cornerA, cornerB=e.cornerB, t=e.t, _pad=0 }; }
            Isect.SetData(gpu);
        }

        public void Dispose()
        {
            CornerPos?.Dispose(); VoxelCorner?.Dispose(); VoxelIsectOffset?.Dispose(); VoxelIsectCount?.Dispose();
            Isect?.Dispose(); VoxelExternalFP?.Dispose(); VoxelExternalFPNormal?.Dispose();
            Tri?.Dispose(); TriCounter?.Dispose(); IndirectArgs?.Dispose();
        }
    }
}
```

- [ ] **Step 5: Run test to verify it passes**

Run: EditMode. Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add Assets/SurgicalSim/ReconGridDC/Core/FixedPointAtomic.cs Assets/SurgicalSim/ReconGridDC/Recon/ReconBuffers.cs Assets/SurgicalSim/ReconGridDC/Tests/*.cs
git commit -m "feat(reconDC): FixedPointAtomic + ReconBuffers (GPU alloc/upload) + test"
```

---

### Task 6: Recon.compute — DC_FeaturePoints (QEF, D1 + D5) + GPU oracle test

**Files:**
- Create: `Assets/SurgicalSim/ReconGridDC/Shaders/ReconCommon.hlsl`
- Create: `Assets/SurgicalSim/ReconGridDC/Shaders/Recon.compute` (DC_FeaturePoints kernel)
- Create: `Assets/SurgicalSim/ReconGridDC/Tests/Qef_GpuOracle_Tests.cs`

**Interfaces:**
- Consumes: buffers from `ReconBuffers` (CornerPos, VoxelCorner, VoxelIsectOffset/Count, Isect).
- Produces: kernel `DC_FeaturePoints` writing `VoxelExternalFP` (`float4`: xyz feature point, w=valid). Constants `_VoxelCount`, `_L`, `_QefIters` (m).

- [ ] **Step 1: Write ReconCommon.hlsl (shared helpers)**

`ReconCommon.hlsl`:
```hlsl
#ifndef RECON_COMMON_INCLUDED
#define RECON_COMMON_INCLUDED
#define NORMAL_SCALE 1048576.0   // FixedPointAtomic.NormalScale (1<<20)

struct IsectGpu { float3 globalRest; float3 normal; int cornerA; int cornerB; float t; float _pad; };

float BoxSdf(float3 p, float3 c, float L){ float3 d = abs(p-c) - L*0.5; return max(max(d.x,d.y),d.z); }
#endif
```

- [ ] **Step 2: Write the failing GPU oracle test**

`Qef_GpuOracle_Tests.cs`:
```csharp
using NUnit.Framework;
using Unity.Mathematics;
using UnityEngine;
using ReconGridDC.Preprocess;
using ReconGridDC.Recon;

public class Qef_GpuOracle_Tests
{
    sealed class PlaneX : ILevelSetProvider {
        public float Sample(float3 p)=>p.x-0.5f; public float3 Gradient(float3 p)=>new float3(1,0,0);
    }

    [Test] public void DC_FeaturePoints_PlaneVoxel_FpOnPlane()
    {
        var cs = Resources.Load<ComputeShader>("Recon"); // or AssetDatabase.LoadAssetAtPath in editor
        Assert.IsNotNull(cs, "Recon.compute must be loadable");
        var g = BackgroundGrid.Build(new PlaneX(), new int3(1,1,1), 1f, float3.zero);
        using (var rb = new ReconBuffers(g, 64))
        {
            rb.Upload(g);
            int k = cs.FindKernel("DC_FeaturePoints");
            cs.SetInt("_VoxelCount", rb.VoxelCount);
            cs.SetFloat("_L", 1f);
            cs.SetInt("_QefIters", 20);
            cs.SetBuffer(k,"_CornerPos", rb.CornerPos);
            cs.SetBuffer(k,"_VoxelCorner", rb.VoxelCorner);
            cs.SetBuffer(k,"_VoxelIsectOffset", rb.VoxelIsectOffset);
            cs.SetBuffer(k,"_VoxelIsectCount", rb.VoxelIsectCount);
            cs.SetBuffer(k,"_Isect", rb.Isect);
            cs.SetBuffer(k,"_VoxelExternalFP", rb.VoxelExternalFP);
            cs.Dispatch(k, Mathf.CeilToInt(rb.VoxelCount/64f), 1, 1);

            var fp = new float4[rb.VoxelCount]; rb.VoxelExternalFP.GetData(fp);
            Assert.AreEqual(1f, fp[0].w, 1e-5f);              // valid
            Assert.AreEqual(0.5f, fp[0].x, 1e-3f);            // feature point pinned to plane x=0.5
            Assert.IsFalse(float.IsNaN(fp[0].y));
        }
    }
}
```

- [ ] **Step 3: Run test to verify it fails**

Run: EditMode `Qef_GpuOracle_Tests` (requires a GPU). Expected: FAIL (Recon.compute / kernel missing).

- [ ] **Step 4: Implement DC_FeaturePoints kernel**

`Recon.compute` (kernel block):
```hlsl
#pragma kernel DC_FeaturePoints
#include "ReconCommon.hlsl"

int _VoxelCount; float _L; int _QefIters;
StructuredBuffer<float3> _CornerPos;
StructuredBuffer<int> _VoxelCorner;            // 8 per voxel
StructuredBuffer<int> _VoxelIsectOffset;
StructuredBuffer<int> _VoxelIsectCount;
StructuredBuffer<IsectGpu> _Isect;
RWStructuredBuffer<float4> _VoxelExternalFP;    // xyz=fp, w=valid

[numthreads(64,1,1)]
void DC_FeaturePoints(uint3 id : SV_DispatchThreadID)
{
    uint v = id.x; if (v >= (uint)_VoxelCount) return;
    int off = _VoxelIsectOffset[v]; int cnt = _VoxelIsectCount[v];
    if (cnt == 0){ _VoxelExternalFP[v] = float4(0,0,0,0); return; }   // no surface in this voxel

    // Eq2 initial average point + voxel center from 8 deformed corners
    float3 x = 0; for (int s=0;s<cnt;s++) x += _Isect[off+s].globalRest; x /= cnt;   // x0 = mean p_i
    float3 c = 0; for (int cc=0;cc<8;cc++) c += _CornerPos[_VoxelCorner[8*v+cc]]; c /= 8.0;

    float3 xprev = x;
    for (int it=0; it<_QefIters; it++)
    {
        float3 F = 0;                                              // DEVIATION D1: QEF gradient at current x
        for (int s2=0;s2<cnt;s2++){ IsectGpu e=_Isect[off+s2]; F += e.normal * dot(e.normal, e.globalRest - x); }
        float a = 0.1 * (1.0 - (float)it/(float)_QefIters);       // Eq4 decaying step
        float3 xn = x + a*F;
        if (BoxSdf(xn, c, _L) >= 0.0) break;                      // DEVIATION D5: stop when next iterate exits voxel
        xprev = x; x = xn;
    }
    _VoxelExternalFP[v] = float4(xprev, 1.0);                     // D5: keep last interior point
}
```

- [ ] **Step 5: Run test to verify it passes**

Run: EditMode `Qef_GpuOracle_Tests`. Expected: PASS (fp.x≈0.5, valid=1, no NaN).

- [ ] **Step 6: Commit**

```bash
git add Assets/SurgicalSim/ReconGridDC/Shaders/ReconCommon.hlsl Assets/SurgicalSim/ReconGridDC/Shaders/Recon.compute Assets/SurgicalSim/ReconGridDC/Tests/Qef_GpuOracle_Tests.cs
git commit -m "feat(reconDC): DC_FeaturePoints QEF kernel (D1 gradient, D5 stop) + GPU oracle test"
```

---

### Task 7: Recon.compute — DC_Stitch (per-edge quad → 2 triangles, winding, append) + GPU oracle test

**Files:**
- Modify: `Assets/SurgicalSim/ReconGridDC/Shaders/Recon.compute` (add DC_Stitch kernel + edge-list inputs)
- Modify: `Assets/SurgicalSim/ReconGridDC/Recon/ReconBuffers.cs` (add an intersecting-grid-edge list buffer)
- Modify: `Assets/SurgicalSim/ReconGridDC/Preprocess/BackgroundGrid.cs` (emit the global intersecting-edge list)
- Test: `Assets/SurgicalSim/ReconGridDC/Tests/Stitch_GpuOracle_Tests.cs`

**Interfaces:**
- Consumes: `VoxelExternalFP` (Task 6), the 4-voxel stencil (`GridConventions.EdgeStencil`).
- Produces:
  - In `BackgroundGrid`: `public struct GridEdge { public int axis; public int3 baseCorner; public int insideA; }` and `public GridEdge[] surfaceEdges;` (interior intersecting grid edges only; `insideA` = 1 if the base corner is inside, used for winding).
  - In `ReconBuffers`: `ComputeBuffer SurfaceEdges` (struct `GridEdgeGpu{ int axis; int3 baseCorner; int insideA; int _pad; }`) and `int SurfaceEdgeCount`.
  - Kernel `DC_Stitch` appends triangle index triples into `Tri` via `InterlockedAdd(_TriCounter[0], 6, base)`.

- [ ] **Step 1: Extend BackgroundGrid to emit interior surface edges**

Add to `BackgroundGrid` (inside `Build`, after the voxel loop). Iterate global grid edges by axis; an edge is a surface edge iff its two corners differ in sign AND all 4 incident voxels exist (interior). Record axis, base corner, and `insideA`:
```csharp
// in BackgroundGrid: new field
public struct GridEdge { public int axis; public int3 baseCorner; public int insideA; }
public GridEdge[] surfaceEdges;

// after the voxel loop in Build():
var edges = new System.Collections.Generic.List<GridEdge>();
int3[] dirs = { new int3(1,0,0), new int3(0,1,0), new int3(0,0,1) };
for (int axis=0; axis<3; axis++)
{
    int3 d = dirs[axis];
    for (int k=0;k<cnz;k++) for (int j=0;j<cny;j++) for (int i=0;i<cnx;i++)
    {
        int3 a = new int3(i,j,k); int3 b = a + d;
        if (b.x>=cnx || b.y>=cny || b.z>=cnz) continue;   // edge must fit in corner grid
        int ca = GridConventions.CornerId(a.x,a.y,a.z,dims);
        int cb = GridConventions.CornerId(b.x,b.y,b.z,dims);
        if ((g.cornerInside[ca]==1) == (g.cornerInside[cb]==1)) continue; // no crossing
        // require all 4 incident voxels in-range (interior) so the quad is complete
        bool allIn = true;
        foreach (var se in GridConventions.EdgeStencil[axis])
        {
            int3 vc = a + se.voxelOffset;
            if (vc.x<0||vc.y<0||vc.z<0||vc.x>=dims.x||vc.y>=dims.y||vc.z>=dims.z){ allIn=false; break; }
        }
        if (!allIn) continue;
        edges.Add(new GridEdge{ axis=axis, baseCorner=a, insideA=g.cornerInside[ca] });
    }
}
g.surfaceEdges = edges.ToArray();
```

- [ ] **Step 2: Add SurfaceEdges buffer to ReconBuffers**

In `ReconBuffers`: add `public ComputeBuffer SurfaceEdges; public int SurfaceEdgeCount;` ; allocate `SurfaceEdges = new ComputeBuffer(Math.Max(1,g.surfaceEdges.Length), 20);` (GridEdgeGpu = int axis + int3 + int = 20 bytes → pad to 32: use 32 and a `_pad` int, set stride 32). In `Upload`, fill a `GridEdgeGpu[]` and `SetData`; set `SurfaceEdgeCount=g.surfaceEdges.Length`. Dispose it.

- [ ] **Step 3: Write the failing stitch oracle test**

`Stitch_GpuOracle_Tests.cs` — a sphere in a small padded grid; assert triangle count > 0, all indices in `[0,VoxelCount)`, and (for a single interior x-edge planar case) exactly 6 indices (2 triangles) appended:
```csharp
using NUnit.Framework; using Unity.Mathematics; using UnityEngine;
using ReconGridDC.Preprocess; using ReconGridDC.Recon;

public class Stitch_GpuOracle_Tests
{
    // plane x=1.5 in a 3x1x1-voxel grid [0,3]x[0,1]x[0,1]; the single interior x-edge column at i=... 
    // Simpler: use a 2x2x2 grid with plane x=1.0 so exactly the interior y-z grid edges at x=1 cross... 
    // Use sphere for a robust nonzero-count smoke check:
    [Test] public void DC_Stitch_Sphere_ProducesClosedTriangleSet()
    {
        var cs = Resources.Load<ComputeShader>("Recon");
        var ls = new SphereLevelSet(new float3(1.25f,1.25f,1.25f), 0.8f);
        var g = BackgroundGrid.Build(ls, new int3(10,10,10), 0.25f, float3.zero); // sphere strictly inside
        using (var rb = new ReconBuffers(g, 200000))
        {
            rb.Upload(g);
            ReconDispatch.FeaturePointsAndStitch(cs, rb, L:0.25f, qefIters:20); // helper added in Task 9
            uint[] cnt = new uint[1]; rb.TriCounter.GetData(cnt);
            Assert.Greater(cnt[0], 0u);
            Assert.AreEqual(0u, cnt[0] % 3u);                 // whole triangles
            int[] idx = new int[cnt[0]]; rb.Tri.GetData(idx, 0, 0, (int)cnt[0]);
            foreach (var ix in idx){ int v = ix & 0x7FFFFFFF; Assert.GreaterOrEqual(v,0); Assert.Less(v, rb.VoxelCount); }
        }
    }
}
```

> Note: this test depends on the `ReconDispatch` helper from Task 9. If executing strictly in order, write a minimal inline dispatch here first and replace with the helper in Task 9; the subagent reviewer should flag if the helper signature drifts.

- [ ] **Step 4: Run test to verify it fails**

Run: EditMode `Stitch_GpuOracle_Tests`. Expected: FAIL (DC_Stitch missing).

- [ ] **Step 5: Implement DC_Stitch kernel**

Add to `Recon.compute`:
```hlsl
#pragma kernel DC_Stitch
struct GridEdgeGpu { int axis; int3 baseCorner; int insideA; int _pad; };
int _SurfaceEdgeCount; int3 _Dims; int _TriCapacity;
StructuredBuffer<GridEdgeGpu> _SurfaceEdges;
RWStructuredBuffer<int> _Tri;
RWStructuredBuffer<uint> _TriCounter;
// 4-voxel stencil constants (mirror GridConventions.EdgeStencil), flattened [axis*4 + s]
static const int3 STENCIL_OFF[12] = {
  int3(0,0,0),int3(0,-1,0),int3(0,-1,-1),int3(0,0,-1),          // x: e0,e2,e6,e4 order (CCW about +x)
  int3(0,0,0),int3(0,0,-1),int3(-1,0,-1),int3(-1,0,0),          // y
  int3(0,0,0),int3(-1,0,0),int3(-1,-1,0),int3(0,-1,0) };        // z
int VoxId(int3 c){ return c.x + _Dims.x*(c.y + _Dims.y*c.z); }

[numthreads(64,1,1)]
void DC_Stitch(uint3 id : SV_DispatchThreadID)
{
    uint e = id.x; if (e >= (uint)_SurfaceEdgeCount) return;
    GridEdgeGpu ge = _SurfaceEdges[e];
    // gather the 4 incident voxel feature points (external FP index == voxelId, high bit clear)
    int v0 = VoxId(ge.baseCorner + STENCIL_OFF[ge.axis*4+0]);
    int v1 = VoxId(ge.baseCorner + STENCIL_OFF[ge.axis*4+1]);
    int v2 = VoxId(ge.baseCorner + STENCIL_OFF[ge.axis*4+2]);
    int v3 = VoxId(ge.baseCorner + STENCIL_OFF[ge.axis*4+3]);
    // winding: CCW order v0..v3 is correct when baseCorner is INSIDE; flip when outside (D-free, surface orientation)
    int a=v0,b=v1,c=v2,d=v3;
    if (ge.insideA == 0){ int t=b; b=d; d=t; }                  // reverse to keep outward normal
    uint baseIdx; InterlockedAdd(_TriCounter[0], 6u, baseIdx);
    if (baseIdx + 6u > (uint)(_TriCapacity*3)) return;          // overflow guard
    _Tri[baseIdx+0]=a; _Tri[baseIdx+1]=b; _Tri[baseIdx+2]=c;     // tri 1
    _Tri[baseIdx+3]=a; _Tri[baseIdx+4]=c; _Tri[baseIdx+5]=d;     // tri 2 (shared diagonal a-c)
}
```

> **Subagent audit focus:** confirm `STENCIL_OFF` rows exactly match `GridConventions.EdgeStencil` (same 4 voxels per axis, CCW about +axis) and that the winding flip on `insideA==0` yields outward normals (cross-check against Task 8's normal-direction oracle).

- [ ] **Step 6: Run test to verify it passes**

Run: EditMode `Stitch_GpuOracle_Tests`. Expected: PASS (nonzero whole-triangle count, indices in range).

- [ ] **Step 7: Commit**

```bash
git add Assets/SurgicalSim/ReconGridDC/Shaders/Recon.compute Assets/SurgicalSim/ReconGridDC/Recon/ReconBuffers.cs Assets/SurgicalSim/ReconGridDC/Preprocess/BackgroundGrid.cs Assets/SurgicalSim/ReconGridDC/Tests/Stitch_GpuOracle_Tests.cs
git commit -m "feat(reconDC): DC_Stitch kernel (4-voxel quad->2 tris, winding, append) + oracle test"
```

---

### Task 8: Recon.compute — normals (fixed-point scatter + normalize, barrier) + direction oracle

**Files:**
- Modify: `Assets/SurgicalSim/ReconGridDC/Shaders/Recon.compute` (DC_NormalsScatter, DC_NormalsNormalize, DC_ClearNormals)
- Test: add to `Assets/SurgicalSim/ReconGridDC/Tests/Stitch_GpuOracle_Tests.cs`

**Interfaces:**
- Consumes: `Tri`, `TriCounter`, `VoxelExternalFP`.
- Produces: `VoxelExternalFPNormal` (fixed-point int3) and a `VoxelExternalFPNormalF : ComputeBuffer<float3>` written by the normalize pass (add to `ReconBuffers`). The dispatch order is `DC_ClearNormals → DC_NormalsScatter → [barrier=separate dispatch] → DC_NormalsNormalize`.

- [ ] **Step 1: Add the normalized-normal output buffer**

In `ReconBuffers`: add `public ComputeBuffer VoxelExternalFPNormalF;` allocate `new ComputeBuffer(VoxelCount, 12);` (float3), dispose it.

- [ ] **Step 2: Write the failing normal-direction test**

Append to `Stitch_GpuOracle_Tests.cs`: after running normals on the sphere, assert each used voxel's normalized normal points roughly radially outward (dot with (fpPos - sphereCenter) > 0):
```csharp
[Test] public void DC_Normals_Sphere_PointOutward()
{
    var cs = Resources.Load<ComputeShader>("Recon");
    var center = new float3(1.25f,1.25f,1.25f);
    var g = BackgroundGrid.Build(new SphereLevelSet(center,0.8f), new int3(10,10,10), 0.25f, float3.zero);
    using (var rb = new ReconBuffers(g, 200000))
    {
        rb.Upload(g);
        ReconDispatch.Full(cs, rb, L:0.25f, qefIters:20);   // FP + stitch + normals (Task 9 helper)
        var fp = new float4[rb.VoxelCount]; rb.VoxelExternalFP.GetData(fp);
        var nf = new float3[rb.VoxelCount]; rb.VoxelExternalFPNormalF.GetData(nf);
        int checkd=0;
        for (int v=0; v<rb.VoxelCount; v++) if (fp[v].w>0.5f && math.length(nf[v])>0.5f)
        {
            float3 radial = math.normalize(fp[v].xyz - center);
            Assert.Greater(math.dot(radial, math.normalize(nf[v])), 0.3f, $"voxel {v} normal not outward");
            checkd++;
        }
        Assert.Greater(checkd, 10);
    }
}
```

- [ ] **Step 3: Run test to verify it fails**

Run: EditMode. Expected: FAIL (normal kernels missing).

- [ ] **Step 4: Implement the three normal kernels**

Add to `Recon.compute`:
```hlsl
#pragma kernel DC_ClearNormals
#pragma kernel DC_NormalsScatter
#pragma kernel DC_NormalsNormalize
RWStructuredBuffer<int> _ExtNormalFixed;     // 3*voxelCount int (xyz fixed-point)
RWStructuredBuffer<float3> _ExtNormalF;      // voxelCount float3 (output)

[numthreads(64,1,1)]
void DC_ClearNormals(uint3 id:SV_DispatchThreadID){
    uint v=id.x; if(v>=(uint)_VoxelCount) return;
    _ExtNormalFixed[3*v+0]=0; _ExtNormalFixed[3*v+1]=0; _ExtNormalFixed[3*v+2]=0;
}

void AtomicAddNormal(int v, float3 n){
    int xi=(int)(n.x*NORMAL_SCALE), yi=(int)(n.y*NORMAL_SCALE), zi=(int)(n.z*NORMAL_SCALE);
    InterlockedAdd(_ExtNormalFixed[3*v+0], xi);
    InterlockedAdd(_ExtNormalFixed[3*v+1], yi);
    InterlockedAdd(_ExtNormalFixed[3*v+2], zi);
}

[numthreads(64,1,1)]
void DC_NormalsScatter(uint3 id:SV_DispatchThreadID){
    uint tri=id.x; uint triCount=_TriCounter[0]/3u; if(tri>=triCount) return;
    int i0=_Tri[3*tri+0]&0x7FFFFFFF, i1=_Tri[3*tri+1]&0x7FFFFFFF, i2=_Tri[3*tri+2]&0x7FFFFFFF;
    float3 p0=_VoxelExternalFP[i0].xyz, p1=_VoxelExternalFP[i1].xyz, p2=_VoxelExternalFP[i2].xyz;
    float3 fn = cross(p1-p0, p2-p0);                   // un-normalized face normal (area-weighted)
    AtomicAddNormal(i0,fn); AtomicAddNormal(i1,fn); AtomicAddNormal(i2,fn);
}

[numthreads(64,1,1)]
void DC_NormalsNormalize(uint3 id:SV_DispatchThreadID){
    uint v=id.x; if(v>=(uint)_VoxelCount) return;
    float3 n=float3(_ExtNormalFixed[3*v+0],_ExtNormalFixed[3*v+1],_ExtNormalFixed[3*v+2])/NORMAL_SCALE;
    float len=length(n); _ExtNormalF[v] = len>1e-6 ? n/len : float3(0,0,0);
}
```

- [ ] **Step 5: Run test to verify it passes**

Run: EditMode `DC_Normals_Sphere_PointOutward`. Expected: PASS (normals outward; if many fail, the Task 7 winding flip is wrong — fix there).

- [ ] **Step 6: Commit**

```bash
git add Assets/SurgicalSim/ReconGridDC/Shaders/Recon.compute Assets/SurgicalSim/ReconGridDC/Recon/ReconBuffers.cs Assets/SurgicalSim/ReconGridDC/Tests/Stitch_GpuOracle_Tests.cs
git commit -m "feat(reconDC): per-vertex normals (fixed-point scatter+barrier+normalize) + outward oracle"
```

---

### Task 9: DualContouring dispatcher + ReconDispatch test helper + indirect args

**Files:**
- Create: `Assets/SurgicalSim/ReconGridDC/Recon/DualContouring.cs` (production dispatcher)
- Create: `Assets/SurgicalSim/ReconGridDC/Tests/ReconDispatch.cs` (test-only helper used by Tasks 7-8 oracles; in the Tests asmdef)

**Interfaces:**
- Consumes: `ComputeShader Recon`, `ReconBuffers`.
- Produces:
  - `class DualContouring` ctor `(ComputeShader recon)` with `void Build(ReconBuffers rb, float L, int qefIters)` that runs: set common params/buffers → `DC_FeaturePoints` → reset `TriCounter` to 0 → `DC_Stitch` → `DC_ClearNormals` → `DC_NormalsScatter` → `DC_NormalsNormalize` → write `IndirectArgs` from `TriCounter` (CopyCount-equivalent: `IndirectArgs[0]=triCounter; [1]=1; [2]=0; [3]=0`).
  - Test helper `ReconDispatch.Full(cs, rb, L, qefIters)` and `ReconDispatch.FeaturePointsAndStitch(...)` delegate to `DualContouring` (so Task 7/8 tests and production share one path).

- [ ] **Step 1: Implement DualContouring.Build**

`DualContouring.cs` — bind all buffers to each kernel, set `_Dims`, `_VoxelCount`, `_L`, `_QefIters`, `_SurfaceEdgeCount`, `_TriCapacity`; reset `TriCounter` via `SetData(new uint[]{0})` before `DC_Stitch`; after stitch run a tiny `DC_WriteIndirectArgs` kernel (add it: reads `_TriCounter[0]`, writes `_IndirectArgs[0]=count; [1]=1;[2]=0;[3]=0`). Dispatch group counts = `ceil(N/64)`.

```csharp
using UnityEngine; using ReconGridDC.Recon; using Unity.Mathematics;
namespace ReconGridDC.Recon
{
    public sealed class DualContouring
    {
        readonly ComputeShader cs;
        readonly int kFP,kStitch,kClearN,kScatterN,kNormN,kArgs;
        public DualContouring(ComputeShader recon){ cs=recon;
            kFP=cs.FindKernel("DC_FeaturePoints"); kStitch=cs.FindKernel("DC_Stitch");
            kClearN=cs.FindKernel("DC_ClearNormals"); kScatterN=cs.FindKernel("DC_NormalsScatter");
            kNormN=cs.FindKernel("DC_NormalsNormalize"); kArgs=cs.FindKernel("DC_WriteIndirectArgs"); }
        static int G(int n)=>Mathf.Max(1,Mathf.CeilToInt(n/64f));
        public void Build(ReconBuffers rb, int3 dims, float L, int qefIters)
        {
            cs.SetInt("_VoxelCount", rb.VoxelCount); cs.SetFloat("_L", L); cs.SetInt("_QefIters", qefIters);
            cs.SetInts("_Dims", dims.x,dims.y,dims.z);
            cs.SetInt("_SurfaceEdgeCount", rb.SurfaceEdgeCount); cs.SetInt("_TriCapacity", rb.TriCapacity);
            // FP
            Bind(kFP, rb); cs.Dispatch(kFP, G(rb.VoxelCount),1,1);
            // stitch (reset counter first)
            rb.TriCounter.SetData(new uint[]{0});
            Bind(kStitch, rb); cs.Dispatch(kStitch, G(rb.SurfaceEdgeCount),1,1);
            // normals
            Bind(kClearN, rb); cs.Dispatch(kClearN, G(rb.VoxelCount),1,1);
            Bind(kScatterN, rb); cs.Dispatch(kScatterN, G(rb.TriCapacity),1,1);   // upper-bound dispatch; kernel guards by triCount
            Bind(kNormN, rb); cs.Dispatch(kNormN, G(rb.VoxelCount),1,1);
            // indirect args
            Bind(kArgs, rb); cs.Dispatch(kArgs, 1,1,1);
        }
        void Bind(int k, ReconBuffers rb){
            cs.SetBuffer(k,"_CornerPos",rb.CornerPos); cs.SetBuffer(k,"_VoxelCorner",rb.VoxelCorner);
            cs.SetBuffer(k,"_VoxelIsectOffset",rb.VoxelIsectOffset); cs.SetBuffer(k,"_VoxelIsectCount",rb.VoxelIsectCount);
            cs.SetBuffer(k,"_Isect",rb.Isect); cs.SetBuffer(k,"_VoxelExternalFP",rb.VoxelExternalFP);
            cs.SetBuffer(k,"_SurfaceEdges",rb.SurfaceEdges); cs.SetBuffer(k,"_Tri",rb.Tri); cs.SetBuffer(k,"_TriCounter",rb.TriCounter);
            cs.SetBuffer(k,"_ExtNormalFixed",rb.VoxelExternalFPNormal); cs.SetBuffer(k,"_ExtNormalF",rb.VoxelExternalFPNormalF);
            cs.SetBuffer(k,"_IndirectArgs",rb.IndirectArgs);
        }
    }
}
```

Add `DC_WriteIndirectArgs` to `Recon.compute`:
```hlsl
#pragma kernel DC_WriteIndirectArgs
RWStructuredBuffer<uint> _IndirectArgs;
[numthreads(1,1,1)]
void DC_WriteIndirectArgs(uint3 id:SV_DispatchThreadID){
    _IndirectArgs[0]=_TriCounter[0]; _IndirectArgs[1]=1; _IndirectArgs[2]=0; _IndirectArgs[3]=0; // vertexCount, instanceCount,...
}
```

> Unity `DrawProceduralIndirect` args layout = {vertexCount, instanceCount, startVertex, startInstance}. `_TriCounter[0]` already counts indices (=vertices for non-indexed proc draw). Confirm in Task 10.

- [ ] **Step 2: Implement the test helper**

`Tests/ReconDispatch.cs`:
```csharp
using UnityEngine; using Unity.Mathematics; using ReconGridDC.Recon;
public static class ReconDispatch
{
    public static void Full(ComputeShader cs, ReconBuffers rb, float L, int qefIters)
        => new DualContouring(cs).Build(rb, GuessDims(rb), L, qefIters);
    public static void FeaturePointsAndStitch(ComputeShader cs, ReconBuffers rb, float L, int qefIters)
        => Full(cs, rb, L, qefIters); // 1A always runs the full chain
    static int3 GuessDims(ReconBuffers rb){ /* tests pass dims via a small wrapper; see note */ return default; }
}
```
> The helper needs `dims`; simplest is to have tests call `new DualContouring(cs).Build(rb, g.dims, L, iters)` directly. Replace the `ReconDispatch.Full(cs, rb, ...)` calls in Task 7/8 tests with `new DualContouring(cs).Build(rb, g.dims, 0.25f, 20)` and delete this helper. (Pick one; keep tests and production on the single `DualContouring.Build` path.)

- [ ] **Step 3: Update Task 7/8 tests to call `DualContouring.Build` directly**

Replace `ReconDispatch.*` calls with `new DualContouring(cs).Build(rb, g.dims, 0.25f, 20);`. Re-run `Stitch_GpuOracle_Tests` and `Qef_GpuOracle_Tests`: Expected PASS.

- [ ] **Step 4: Commit**

```bash
git add Assets/SurgicalSim/ReconGridDC/Recon/DualContouring.cs Assets/SurgicalSim/ReconGridDC/Shaders/Recon.compute Assets/SurgicalSim/ReconGridDC/Tests/*.cs
git commit -m "feat(reconDC): DualContouring dispatcher (FP->stitch->normals->indirect args) + wire tests"
```

---

### Task 10: ReconGridManager + ReconSurface.shader + Demo scene (visual milestone)

**Files:**
- Create: `Assets/SurgicalSim/ReconGridDC/Shaders/ReconSurface.shader`
- Create: `Assets/SurgicalSim/ReconGridDC/Demo/ReconGridManager.cs`
- Create: `Assets/SurgicalSim/ReconGridDC/Demo/DemoSceneSetup.cs`

**Interfaces:**
- Consumes: `BackgroundGrid`, `ReconBuffers`, `DualContouring`, `Recon.compute` (Resources or serialized field), `ReconSurface.shader`.
- Produces: a runnable MonoBehaviour that, in `Start`, preprocesses a sphere, uploads, builds the surface once, and renders it every frame via `Graphics.DrawProceduralIndirect`.

- [ ] **Step 1: Write the procedural surface shader**

`ReconSurface.shader` — vertex shader indexes `_Tri[SV_VertexID]`, decodes the tag (1A: all external → index = voxelId), fetches position from `_VoxelExternalFP` and normal from `_ExtNormalF`, simple Lambert shade:
```hlsl
Shader "ReconGridDC/ReconSurface"
{
  SubShader{ Pass{
    HLSLPROGRAM
    #pragma vertex vert
    #pragma fragment frag
    #include "UnityCG.cginc"
    StructuredBuffer<int> _Tri;
    StructuredBuffer<float4> _VoxelExternalFP;
    StructuredBuffer<float3> _ExtNormalF;
    struct v2f{ float4 pos:SV_POSITION; float3 n:TEXCOORD0; };
    v2f vert(uint vid:SV_VertexID){
      int idx = _Tri[vid] & 0x7FFFFFFF;       // 1A: external => voxelId
      float3 wp = _VoxelExternalFP[idx].xyz;
      v2f o; o.pos = UnityObjectToClipPos(float4(wp,1)); o.n = _ExtNormalF[idx]; return o;
    }
    fixed4 frag(v2f i):SV_Target{
      float ndl = saturate(dot(normalize(i.n), normalize(float3(0.3,1,0.2))))*0.8+0.2;
      return fixed4(ndl.xxx,1);
    }
    ENDHLSL
  }}
}
```

- [ ] **Step 2: Write ReconGridManager**

`ReconGridManager.cs`:
```csharp
using UnityEngine; using Unity.Mathematics;
using ReconGridDC.Preprocess; using ReconGridDC.Recon;
namespace ReconGridDC.Demo
{
    public sealed class ReconGridManager : MonoBehaviour
    {
        public ComputeShader recon; public Shader surfaceShader;
        public int3 dims = new int3(24,24,24); public float L=0.25f; public int qefIters=20;
        public float3 sphereCenter = new float3(3,3,3); public float sphereRadius=2.0f;
        ReconBuffers rb; Material mat; Bounds bounds;
        void Start(){
            var g = BackgroundGrid.Build(new SphereLevelSet(sphereCenter, sphereRadius), dims, L, float3.zero);
            rb = new ReconBuffers(g, triCapacity: 6*g.surfaceEdges.Length + 1024);
            rb.Upload(g);
            new DualContouring(recon).Build(rb, dims, L, qefIters);
            mat = new Material(surfaceShader);
            mat.SetBuffer("_Tri", rb.Tri); mat.SetBuffer("_VoxelExternalFP", rb.VoxelExternalFP);
            mat.SetBuffer("_ExtNormalF", rb.VoxelExternalFPNormalF);
            bounds = new Bounds((Vector3)(float3)(g.origin + (float3)dims*L*0.5f), (Vector3)(float3)dims*L*2f);
        }
        void Update(){
            if (rb!=null) Graphics.DrawProceduralIndirect(mat, bounds, MeshTopology.Triangles, rb.IndirectArgs, 0);
        }
        void OnDestroy(){ rb?.Dispose(); if(mat) Destroy(mat); }
    }
}
```

- [ ] **Step 3: Write DemoSceneSetup**

`DemoSceneSetup.cs` — a helper that creates a GameObject with `ReconGridManager`, assigns `recon` (Resources.Load or serialized) and `surfaceShader` (Shader.Find), and positions a camera. (Move `Recon.compute` under a `Resources/` subfolder or assign via inspector.)

- [ ] **Step 4: Verify visually (user, on a GPU machine)**

Run the demo scene in Play mode. Expected: a watertight, smoothly-shaded sphere whose silhouette matches radius `sphereRadius` centered at `sphereCenter`; no black/flickering faces; no holes. Rotate camera to confirm closed surface.

- [ ] **Step 5: Commit**

```bash
git add Assets/SurgicalSim/ReconGridDC/Shaders/ReconSurface.shader Assets/SurgicalSim/ReconGridDC/Demo/*.cs
git commit -m "feat(reconDC): ReconGridManager + procedural surface shader + demo (Stage 1A visual milestone)"
```

---

### Task 11: Stage 1A verification gate (subagent audit + checklist)

**Files:** none (review task).

- [ ] **Step 1: Run the full EditMode suite (user, GPU machine)**

Run all `ReconGridDC.Tests`. Expected: all PASS — `GridConventions`, `LevelSet`, `BackgroundGrid` (CPU, must pass anywhere); `Qef`, `Stitch`, `Normals` (GPU oracles).

- [ ] **Step 2: Dispatch 2-3 subagents to audit Stage 1A vs this plan + spec §4.1-§4.3**

Audit dimensions: (a) algorithm alignment — QEF D1/D5 exactly as ledger, Hermite `t=φA/(φA−φB)`, sign convention, MC sign-change == MC256 criterion; (b) index/stencil correctness — `STENCIL_OFF` (HLSL) ≡ `GridConventions.EdgeStencil` (C#), voxelId encoding consistent CPU↔GPU, winding→outward normals; (c) GPU realization — fixed-point atomic scale, the clear→scatter→normalize barrier ordering (separate dispatches), triCounter reset-before-stitch, indirect args layout, no buffer leaks (Dispose). Each finding skeptic-verified; fix until clean.

- [ ] **Step 3: Update tracking docs + commit**

Update `progress.md` (Stage 1A done + audit result) and `task_plan.md` (mark 1A). Commit.

```bash
git add progress.md task_plan.md
git commit -m "docs(reconDC): Stage 1A complete + audited; ready for Stage 1B (physics)"
```

---

## Self-Review (writing-plans checklist)

**1. Spec coverage (§4.1-§4.3 static subset):** preprocessing/level-set (Tasks 3-4) ✓; corner classify + intersecting edges + Hermite global+normal+cornerA/B+t (Task 4) ✓; per-voxel isect compaction for QEF addressability, design S1-I7 (Task 4/5) ✓; QEF D1+D5 (Task 6) ✓; 4-voxel stencil stitch + winding (Task 7) ✓; normals fixed-point scatter+barrier+normalize, S1-I3/C08 (Task 8) ✓; single triCounter reset + indirect args, S1-I6 (Task 9) ✓; render (Task 10) ✓. **Deferred to 1B (documented):** §4.4 mass-spring, §4.5 RK45, §4.6 RebuildIsectWorld + RK dispatch, physics buffers (forceInt/extForce/kSlope/yTrial), §4.7 physics oracles. **Not in scope (Stage 2/3):** 4096 LUT, cutting, haptics.

**2. Placeholder scan:** The only soft spots are the `ReconDispatch` test helper (Task 9 explicitly resolves it to direct `DualContouring.Build` calls) and the Demo wiring (Task 10 Step 3, mechanical Unity inspector setup). No "TODO/handle edge cases/add validation" placeholders; every kernel and C# unit has complete code.

**3. Type consistency:** `IsectEntry` (C#) ↔ `IsectGpu`(ReconBuffers, 32B) ↔ `IsectGpu`(HLSL) fields match (globalRest,normal,cornerA,cornerB,t,_pad). `GridEdge`(C#) ↔ `GridEdgeGpu`(HLSL, 32B). `voxelExternalFP` is `float4` (xyz+valid) in buffer, HLSL, and shader. `_Tri` index tag (high bit) consistent across DC_Stitch write, DC_NormalsScatter read, and ReconSurface.shader read. `DualContouring.Build(rb, dims, L, qefIters)` signature used identically in tests and ReconGridManager.

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-06-27-dc-stage1a-recon.md`. Two execution options:

1. **Subagent-Driven (recommended)** — dispatch a fresh subagent per task, review between tasks, fast iteration.
2. **Inline Execution** — execute tasks in this session using executing-plans, batch execution with checkpoints.

Which approach?
