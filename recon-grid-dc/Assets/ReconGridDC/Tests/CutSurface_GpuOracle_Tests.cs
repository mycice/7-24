// CutSurface_GpuOracle_Tests.cs — Stage 2C + 2C-FIX oracle tests (plan §5 / §7)
//
// 2C-FIX scenario (plan §7): the level set now has a REAL boundary so the OUTER skin is non-empty
// (the old AllInsideLS structurally hid RC1/RC2 — with everything inside there were no surface edges
// and no empty voxels). We use a half-inside slab phi = p.y - 1.5 on a 3×3×3 grid (4×4×4 corners,
// L=1, origin=0):
//   • corners at y∈{0,1} are INSIDE (phi<0), y∈{2,3} OUTSIDE → a planar tissue boundary at y=1.5.
//   • voxels vy=0 (all-inside) + vy=1 (boundary) are OCCUPIED; voxels vy=2 (all-outside) are EMPTY.
//   • a large swept plane at z=1.5 cuts every z-axis interior edge straddling z∈[1,2]; the centre
//     voxel (1,1,1) is common to all 4 cut z-edges → mask 0xF00 → 2 connectivity components
//     (bottom z-face V0..V3 vs top z-face V4..V7). It is also a boundary (occupied) voxel.
//
// Tests:
//   1. LocalCornerOf round-trip (CPU) — pins the vertex key to Fig 2.4 (the A1 merge gate).
//   2. Connectivity (GPU) — voxelFPCount[v]==(mask==0||!occupied?0:compCount); centre==2; empty==0.
//   3. ComputeComponentFP (GPU) — centre voxel's two component slots valid + their separation ≥ 0.5·D
//      (HARD numeric floor, plan §7 IMPORTANT-3 — guarantees the slit opens, not a sanity check).
//   4. RC2 (GPU) — a NON-occupied voxel has NO valid _CutFP slot and is referenced by NO emitted tri.
//   5. RC1 skin split (GPU, end-to-end) — DC_Stitch attaches the outer skin to per-component cut FPs;
//      the centre voxel's two component slots BOTH appear in the SKIN tris (different slots → split).
//   6. No re-weld + tag decode (GPU) — BuildCutTriangles emits internal-tagged tris from DIFFERENT
//      per-component slots for the centre voxel (the walls split; the MC2024 re-weld failure cannot recur).
//
// [Category("GPU")] on the GPU tests. Mirrors CutBitPropagation_GpuOracle_Tests structure.

using System.Collections.Generic;
using NUnit.Framework;
using Unity.Mathematics;
using UnityEngine;
using ReconGridDC.Preprocess;
using ReconGridDC.Recon;
using ReconGridDC.Cutting;
using ReconGridDC.Core;

#if UNITY_EDITOR
using UnityEditor;
#endif

public class CutSurface_GpuOracle_Tests
{
    // 2C-FIX level set (plan §7): a half-inside slab → a REAL planar tissue boundary at y=1.5, so the
    // outer skin is non-empty AND there are all-outside (empty) voxels to exercise the RC2 gate.
    // Gradient = +y (outward where phi increases) — analytic, matches a y-plane boundary.
    sealed class HalfInsideSlabLS : ILevelSetProvider
    {
        public float  Sample(float3 p)   => p.y - 1.5f;       // inside (phi<0) where y < 1.5
        public float3 Gradient(float3 p) => new float3(0, 1, 0);
    }

    static ComputeShader LoadCuttingShader()
    {
#if UNITY_EDITOR
        foreach (var guid in AssetDatabase.FindAssets("Cutting t:ComputeShader"))
        {
            var path = AssetDatabase.GUIDToAssetPath(guid);
            if (path.EndsWith("/Cutting.compute"))
            {
                var cs = AssetDatabase.LoadAssetAtPath<ComputeShader>(path);
                if (cs != null) return cs;
            }
        }
#endif
        return Resources.Load<ComputeShader>("Cutting");
    }

    static ComputeShader LoadReconShader()
    {
#if UNITY_EDITOR
        foreach (var guid in AssetDatabase.FindAssets("Recon t:ComputeShader"))
        {
            var path = AssetDatabase.GUIDToAssetPath(guid);
            if (path.EndsWith("/Recon.compute"))
            {
                var cs = AssetDatabase.LoadAssetAtPath<ComputeShader>(path);
                if (cs != null) return cs;
            }
        }
#endif
        return Resources.Load<ComputeShader>("Recon");
    }

    // ── Test 1: LocalCornerOf round-trip — CPU, runs anywhere (plan §6.1 / A1 merge gate) ─────
    [Test]
    public void LocalCornerOf_RoundTrips_AndMatchesFig24()
    {
        // The Fig 2.4 corner offsets the CONN4096 LUT was built on (must equal both the C#
        // GridConventions.CornerOffset[] AND the HLSL CornerOffsetHLSL()).
        int3[] fig24 =
        {
            new int3(0,0,0), new int3(1,0,0), new int3(1,1,0), new int3(0,1,0),
            new int3(0,0,1), new int3(1,0,1), new int3(1,1,1), new int3(0,1,1),
        };
        for (int lc = 0; lc < 8; lc++)
            Assert.AreEqual(fig24[lc], GridConventions.CornerOffset[lc],
                $"GridConventions.CornerOffset[{lc}] must match Fig 2.4 (and the HLSL CornerOffsetHLSL twin)");

        var dims = new int3(3, 3, 3);

        // ∀ voxel × 8 corners: LocalCornerOf returns 0..7 and round-trips.
        for (int vz = 0; vz < dims.z; vz++)
        for (int vy = 0; vy < dims.y; vy++)
        for (int vx = 0; vx < dims.x; vx++)
        {
            int3 voxelCoord = new int3(vx, vy, vz);
            for (int lc = 0; lc < 8; lc++)
            {
                int3 cornerCoord = voxelCoord + GridConventions.CornerOffset[lc];
                int got = GridConventions.LocalCornerOf(cornerCoord, voxelCoord);
                Assert.AreEqual(lc, got,
                    $"LocalCornerOf(voxel+CornerOffset[{lc}], voxel) must round-trip to {lc}");
                Assert.AreEqual(cornerCoord, voxelCoord + GridConventions.CornerOffset[got],
                    "voxelCoord + CornerOffset[LocalCornerOf] must equal the corner coord");
            }
        }

        // Non-corner → -1.
        Assert.AreEqual(-1, GridConventions.LocalCornerOf(new int3(2, 0, 0), new int3(0, 0, 0)),
            "A coord that is not a corner of the voxel must return -1");
        Assert.AreEqual(-1, GridConventions.LocalCornerOf(new int3(-1, 0, 0), new int3(0, 0, 0)),
            "A negative offset that is not a corner must return -1");
    }

    // ── Shared scenario builder: builds the slab grid, dispatches DetectCut at z=1.5 ────────────
    // Returns the populated ReconBuffers (caller disposes) + the grid. After this call the cut mask
    // and cut points reflect the planar z=1.5 cut over the half-inside slab.
    static void BuildPlanarCut(out BackgroundGrid g, out ReconBuffers rb, out ComputeShader cs,
                               int3 dims, float L, float D)
    {
        cs = LoadCuttingShader();
        Assert.IsNotNull(cs, "Cutting.compute must be loadable via AssetDatabase or Resources");

        g  = BackgroundGrid.Build(new HalfInsideSlabLS(), dims, L, float3.zero);
        rb = new ReconBuffers(g, 4096);
        rb.Upload(g);

        // Swept plane at z=1.5 spanning the full XY extent (n_cut=(0,0,1)).
        Vector3 sprev = new Vector3(-5, -5, 1.5f);
        Vector3 eprev = new Vector3( 5, -5, 1.5f);
        Vector3 ecur  = new Vector3( 5,  5, 1.5f);
        Vector3 scur  = new Vector3(-5,  5, 1.5f);
        var tool = new CuttingTool(sprev, eprev, D);
        tool.Advance(scur, ecur);
        Assert.IsTrue(tool.Valid, "Sweep must be non-degenerate");

        int kDetect = cs.FindKernel("DetectCut");
        cs.SetInts("_Dims", dims.x, dims.y, dims.z);
        cs.SetInt ("_GridEdgeCount", rb.GridEdgeCount);
        cs.SetInt ("_CutPointCapacity", rb.CutPointCapacity);
        cs.SetVector("_T1V0", tool.T1V0); cs.SetVector("_T1V1", tool.T1V1); cs.SetVector("_T1V2", tool.T1V2);
        cs.SetVector("_T2V0", tool.T2V0); cs.SetVector("_T2V1", tool.T2V1); cs.SetVector("_T2V2", tool.T2V2);
        cs.SetVector("_NCut", tool.NCut); cs.SetFloat("_D", tool.D); cs.SetInt("_ToolValid", 1);
        // Rest lattice (DetectCut tests the REST edge = _GridOrigin + coord*_VoxelL). L/origin match the
        // grid build above, so with no deformation rest == cornerPos and the cut set is unchanged.
        cs.SetFloat("_VoxelL", L); cs.SetVector("_GridOrigin", Vector3.zero);
        cs.SetVector("_SweptAABBMin", new Vector3(-6, -6, 1.4f));
        cs.SetVector("_SweptAABBMax", new Vector3( 6,  6, 1.6f));
        cs.SetBuffer(kDetect, "_CornerPos",       rb.CornerPos);
        cs.SetBuffer(kDetect, "_GridEdges",       rb.GridEdges);
        cs.SetBuffer(kDetect, "_VoxelCutMask",    rb.VoxelCutMask);
        cs.SetBuffer(kDetect, "_VoxelOccupied",   rb.VoxelOccupied);  // 2C-fix: RC2 gate in DetectCut
        cs.SetBuffer(kDetect, "_CutPoint",        rb.CutPoint);
        cs.SetBuffer(kDetect, "_CutPointCounter", rb.CutPointCounter);
        // 2D-2: EmitCutPoint reads _ParticleRot. rb.ParticleRot is identity-initialised by Upload, so
        // localOffset = transpose(I)·(P_side-pPos) = P_side-pPos → byte-identical to the pre-2D-2 path.
        cs.SetBuffer(kDetect, "_ParticleRot",     rb.ParticleRot);
        cs.Dispatch(kDetect, Mathf.CeilToInt(rb.GridEdgeCount / 64f), 1, 1);
    }

    // Bind every shared cut buffer to a Cutting.compute kernel (mirrors DualContouring.BindCut).
    static void BindCut(ComputeShader cs, int k, ReconBuffers rb)
    {
        cs.SetBuffer(k, "_CornerPos",      rb.CornerPos);
        cs.SetBuffer(k, "_GridEdges",      rb.GridEdges);
        cs.SetBuffer(k, "_VoxelCutMask",   rb.VoxelCutMask);
        cs.SetBuffer(k, "_CutPoint",       rb.CutPoint);
        cs.SetBuffer(k, "_CutPointCounter",rb.CutPointCounter);
        cs.SetBuffer(k, "_Conn4096",       rb.Conn4096);
        cs.SetBuffer(k, "_Tri",            rb.Tri);
        cs.SetBuffer(k, "_TriCounter",     rb.TriCounter);
        cs.SetBuffer(k, "_CutFP",          rb.CutFP);
        cs.SetBuffer(k, "_CutFPAccumPos",  rb.CutFPAccumPos);
        cs.SetBuffer(k, "_CutFPAccumCnt",  rb.CutFPAccumCnt);
        cs.SetBuffer(k, "_VoxelFPCount",   rb.VoxelFPCount);
        cs.SetBuffer(k, "_CutFPNormal",    rb.CutFPNormal);
        cs.SetBuffer(k, "_CutFPNormalF",   rb.CutFPNormalF);
        // 2C-fix (plan §7): ComputeComponentFP isect→component QEF inputs + RC2 gate.
        cs.SetBuffer(k, "_Isect",            rb.Isect);
        cs.SetBuffer(k, "_IsectWorld",       rb.IsectWorld);
        cs.SetBuffer(k, "_VoxelIsectOffset", rb.VoxelIsectOffset);
        cs.SetBuffer(k, "_VoxelIsectCount",  rb.VoxelIsectCount);
        cs.SetBuffer(k, "_VoxelCorner",      rb.VoxelCorner);
        cs.SetBuffer(k, "_VoxelOccupied",    rb.VoxelOccupied);
        // 2D-2: AccumulateCutFP reconstructs world = cornerPos[owner] + mul(R_owner, localOffset).
        // rb.ParticleRot is identity-initialised by Upload → mul(I, x) = x → byte-identical to pre-2D-2.
        cs.SetBuffer(k, "_ParticleRot",      rb.ParticleRot);
    }

    static void SetCutScalars(ComputeShader cs, int3 dims, ReconBuffers rb, float L, int qefIters)
    {
        cs.SetInts("_Dims", dims.x, dims.y, dims.z);
        cs.SetInt ("_VoxelCount",   rb.VoxelCount);
        cs.SetInt ("_CornerCount",  rb.CornerCount);
        cs.SetInt ("_TriCapacity",  rb.TriCapacity);
        cs.SetInt ("_CutSlotCount", 8 * rb.VoxelCount);
        cs.SetInt ("_QefIters",     qefIters);   // 2C-fix: ComputeComponentFP QEF
        cs.SetFloat("_L",           L);
    }

    // Run the 2C FP block (ClearCutFP → LookupConnectivity → AccumulateCutFP → ComputeComponentFP).
    // 2C-fix: ComputeComponentFP needs _IsectWorld to be the DEFORMED isect position. At rest (no
    // physics step) _IsectWorld must first be filled by RebuildIsectWorld; we run it via the Recon
    // shader before the FP block in the tests that read _CutFP positions.
    static void RunFpBlock(ComputeShader cs, int3 dims, ReconBuffers rb, float L, int qefIters)
    {
        SetCutScalars(cs, dims, rb, L, qefIters);
        int slotCount = 8 * rb.VoxelCount;
        int kClear   = cs.FindKernel("ClearCutFP");
        int kLookup  = cs.FindKernel("LookupConnectivity");
        int kAccum   = cs.FindKernel("AccumulateCutFP");
        int kCompFP  = cs.FindKernel("ComputeComponentFP");   // 2C-fix: replaces FinalizeCutFP
        BindCut(cs, kClear,  rb); cs.Dispatch(kClear,  Mathf.Max(1, Mathf.CeilToInt(slotCount / 64f)), 1, 1);
        BindCut(cs, kLookup, rb); cs.Dispatch(kLookup, Mathf.Max(1, Mathf.CeilToInt(rb.VoxelCount / 64f)), 1, 1);
        BindCut(cs, kAccum,  rb); cs.Dispatch(kAccum,  Mathf.Max(1, Mathf.CeilToInt(rb.CutPointCapacity / 64f)), 1, 1);
        BindCut(cs, kCompFP, rb); cs.Dispatch(kCompFP, Mathf.Max(1, Mathf.CeilToInt(rb.VoxelCount / 64f)), 1, 1);
    }

    // Fill _IsectWorld (deformed isect positions) + _VoxelExternalFP via the Recon shader at rest.
    // ComputeComponentFP reads _IsectWorld; at rest it equals globalRest. DC_Stitch reads _VoxelExternalFP.
    static void RunReconFP(ComputeShader recon, int3 dims, ReconBuffers rb, float L, int qefIters)
    {
        recon.SetInt("_VoxelCount",   rb.VoxelCount);
        recon.SetFloat("_L",          L);
        recon.SetInt("_QefIters",     qefIters);
        recon.SetInts("_Dims",        dims.x, dims.y, dims.z);
        recon.SetInt("_IsectCount",   rb.IsectCount);
        recon.SetInt("_SurfaceEdgeCount", rb.SurfaceEdgeCount);
        recon.SetInt("_TriCapacity",  rb.TriCapacity);

        int kRebuild = recon.FindKernel("RebuildIsectWorld");
        int kFP      = recon.FindKernel("DC_FeaturePoints");
        BindReconAll(recon, kRebuild, rb);
        recon.Dispatch(kRebuild, Mathf.Max(1, Mathf.CeilToInt(rb.IsectCount / 64f)), 1, 1);
        BindReconAll(recon, kFP, rb);
        recon.Dispatch(kFP, Mathf.Max(1, Mathf.CeilToInt(rb.VoxelCount / 64f)), 1, 1);
    }

    // Bind all Recon-path buffers (mirrors DualContouring.Bind, incl. the 2C-fix DC_Stitch buffers).
    static void BindReconAll(ComputeShader recon, int k, ReconBuffers rb)
    {
        recon.SetBuffer(k, "_CornerPos",        rb.CornerPos);
        recon.SetBuffer(k, "_VoxelCorner",      rb.VoxelCorner);
        recon.SetBuffer(k, "_VoxelIsectOffset", rb.VoxelIsectOffset);
        recon.SetBuffer(k, "_VoxelIsectCount",  rb.VoxelIsectCount);
        recon.SetBuffer(k, "_Isect",            rb.Isect);
        recon.SetBuffer(k, "_IsectWorld",       rb.IsectWorld);
        recon.SetBuffer(k, "_VoxelExternalFP",  rb.VoxelExternalFP);
        recon.SetBuffer(k, "_SurfaceEdges",     rb.SurfaceEdges);
        recon.SetBuffer(k, "_Tri",              rb.Tri);
        recon.SetBuffer(k, "_TriCounter",       rb.TriCounter);
        recon.SetBuffer(k, "_ExtNormalFixed",   rb.VoxelExternalFPNormal);
        recon.SetBuffer(k, "_ExtNormalF",       rb.VoxelExternalFPNormalF);
        recon.SetBuffer(k, "_IndirectArgs",     rb.IndirectArgs);
        recon.SetBuffer(k, "_CutFP",            rb.CutFP);
        recon.SetBuffer(k, "_CutFPNormal",      rb.CutFPNormal);
        recon.SetBuffer(k, "_Conn4096",         rb.Conn4096);      // 2C-fix RC1
        recon.SetBuffer(k, "_VoxelCutMask",     rb.VoxelCutMask);  // 2C-fix RC1
        recon.SetBuffer(k, "_VoxelOccupied",    rb.VoxelOccupied); // 2C-fix RC1/RC2
    }

    // ── Test 2: Connectivity (GPU, plan §6.2) ─────────────────────────────────────────────────
    [Test]
    [Category("GPU")]
    public void LookupConnectivity_PlanarCut_CentreVoxelHasTwoComponents()
    {
        var dims = new int3(3, 3, 3);
        BuildPlanarCut(out var g, out var rb, out var cs, dims, 1f, 0.2f);
        using (rb)
        {
            SetCutScalars(cs, dims, rb, 1f, 20);
            int kClear  = cs.FindKernel("ClearCutFP");
            int kLookup = cs.FindKernel("LookupConnectivity");
            int slotCount = 8 * rb.VoxelCount;
            BindCut(cs, kClear,  rb); cs.Dispatch(kClear,  Mathf.CeilToInt(slotCount / 64f), 1, 1);
            BindCut(cs, kLookup, rb); cs.Dispatch(kLookup, Mathf.CeilToInt(rb.VoxelCount / 64f), 1, 1);

            var fpCount = new uint[rb.VoxelCount];
            rb.VoxelFPCount.GetData(fpCount);
            var mask = new uint[rb.VoxelCount];
            rb.VoxelCutMask.GetData(mask);
            var occ = new uint[rb.VoxelCount];
            rb.VoxelOccupied.GetData(occ);

            // Rebuild the CONN4096 LUT on the CPU to cross-check compCount per mask.
            ConnectivityLUT.Build(out int[] compCount, out _);

            for (int v = 0; v < rb.VoxelCount; v++)
            {
                uint m = mask[v] & 0xFFFu;
                // 2C-fix: gate folds occupancy in — non-occupied voxels are forced to 0.
                uint expected = (m == 0u || occ[v] == 0u) ? 0u : (uint)compCount[m];
                Assert.AreEqual(expected, fpCount[v],
                    $"voxelFPCount[{v}] must equal (mask==0||!occ?0:compCount[mask]). mask=0x{m:X} occ={occ[v]}");
            }

            // Headline: the centre voxel (1,1,1) is cut on all 4 vertical edges → 2 components.
            int centre = GridConventions.VoxelId(1, 1, 1, dims);
            Assert.AreEqual(0xF00u, mask[centre] & 0xFFFu,
                "Centre voxel (1,1,1) must have all 4 vertical edges {8,9,10,11} cut = 0xF00");
            Assert.AreEqual(1u, occ[centre], "Centre voxel (1,1,1) must be occupied (boundary voxel)");
            Assert.AreEqual(2u, fpCount[centre],
                "Centre voxel (1,1,1) must split into exactly 2 connectivity components (top/bottom)");

            // RC2: a vy=2 voxel is all-outside the slab → NOT occupied → fpCount 0.
            int empty = GridConventions.VoxelId(0, 2, 0, dims);
            Assert.AreEqual(0u, occ[empty], "Voxel (0,2,0) is above the y=1.5 slab → must be non-occupied");
            Assert.AreEqual(0u, fpCount[empty], "Non-occupied voxel must have voxelFPCount==0");
        }
    }

    // ── Test 3: ComputeComponentFP centre voxel — two valid slots + separation ≥ 0.5·D (plan §7) ──
    [Test]
    [Category("GPU")]
    public void ComputeComponentFP_CentreVoxel_TwoValidSlots_SeparationAtLeastHalfD()
    {
        var dims = new int3(3, 3, 3);
        float D  = 0.2f;
        var recon = LoadReconShader();
        Assert.IsNotNull(recon, "Recon.compute must be loadable");
        BuildPlanarCut(out var g, out var rb, out var cs, dims, 1f, D);
        using (rb)
        {
            RunReconFP(recon, dims, rb, 1f, 20);   // fill _IsectWorld (deformed isect) at rest
            RunFpBlock(cs, dims, rb, 1f, 20);

            int centre = GridConventions.VoxelId(1, 1, 1, dims);

            // The centre voxel's mask is 0xF00 → 2 components (bottom z-face vs top z-face).
            ConnectivityLUT.Build(out int[] compCount, out int[] vertToComp);
            int maskCentre = 0xF00;
            int compBottom = vertToComp[maskCentre * 8 + 0]; // V0 (z=bottom)
            int compTop    = vertToComp[maskCentre * 8 + 4]; // V4 (z=top)
            Assert.AreNotEqual(compBottom, compTop,
                "Centre voxel: bottom-z-face and top-z-face corners must be in DIFFERENT components");

            int slotBottom = 8 * centre + compBottom;
            int slotTop    = 8 * centre + compTop;

            var cutFP = new float4[8 * rb.VoxelCount];
            rb.CutFP.GetData(cutFP);
            Assert.Greater(cutFP[slotBottom].w, 0.5f, "Centre bottom-component slot must be a valid FP");
            Assert.Greater(cutFP[slotTop].w,    0.5f, "Centre top-component slot must be a valid FP");

            // Both component dual vertices must stay inside the centre voxel box (D5 interior-keep).
            float3 boxC = new float3(1.5f, 1.5f, 1.5f);   // centre of voxel (1,1,1), L=1
            Assert.LessOrEqual(GridConventions.BoxSdf(cutFP[slotBottom].xyz, boxC, 1f), 1e-3f,
                "Bottom-component FP must stay within the voxel box (D5)");
            Assert.LessOrEqual(GridConventions.BoxSdf(cutFP[slotTop].xyz, boxC, 1f), 1e-3f,
                "Top-component FP must stay within the voxel box (D5)");

            // HARD floor (plan §7 IMPORTANT-3): the two boundary components' FPs must be separated by
            // ≥ 0.5·D. This is the numeric guarantee the slit OPENS (the two halves sit on opposite
            // sides of the blade) — not a sanity check. The separation is driven primarily along z
            // (the cut-plane normal) by the ±D/2-offset cut centroids.
            float3 pB = cutFP[slotBottom].xyz;
            float3 pT = cutFP[slotTop].xyz;
            float sep = math.length(pT - pB);
            Assert.GreaterOrEqual(sep, 0.5f * D,
                $"Centre voxel's two component FPs must be separated by ≥ 0.5·D={0.5f * D}. Actual={sep:F5}");
            Assert.Greater(math.abs(pT.z - pB.z), 0.0f,
                "Separation must have a non-zero component along the cut-plane normal (z)");
        }
    }

    // ── Test 4: RC2 — no geometry in non-occupied voxels (plan §7 test (a)) ─────────────────────
    [Test]
    [Category("GPU")]
    public void RC2_NonOccupiedVoxel_HasNoValidCutFP_AndNoTriReference()
    {
        var dims = new int3(3, 3, 3);
        float D  = 0.2f;
        var recon = LoadReconShader();
        Assert.IsNotNull(recon, "Recon.compute must be loadable");
        BuildPlanarCut(out var g, out var rb, out var cs, dims, 1f, D);
        using (rb)
        {
            // Full end-to-end build so BOTH the skin (DC_Stitch) and walls (BuildCutTriangles) run.
            new DualContouring(recon, cs).Build(rb, dims, 1f, 20);

            var occ = new uint[rb.VoxelCount];
            rb.VoxelOccupied.GetData(occ);
            var cutFP = new float4[8 * rb.VoxelCount];
            rb.CutFP.GetData(cutFP);

            var counter = new uint[1];
            rb.TriCounter.GetData(counter);
            int idxCount = (int)Mathf.Min(counter[0], (uint)(3 * rb.TriCapacity));
            var tri = new int[3 * rb.TriCapacity];
            rb.Tri.GetData(tri);

            // Decode every emitted vertex tag to the voxel it references (TWO-WAY tag scheme):
            //   bit31 cut FP (_CutFP, skin+wall shared) → slot = raw & 0x3FFFFFFF, voxel = slot/8.
            //   no high bit (per-voxel external)        → voxel = raw (voxelId).
            var referencedVoxels = new HashSet<int>();
            for (int i = 0; i < idxCount; i++)
            {
                int raw = tri[i];
                bool cutTag = (raw & unchecked((int)0x80000000)) != 0;
                int v = cutTag ? ((raw & 0x3FFFFFFF) / 8) : raw;
                referencedVoxels.Add(v);
            }

            // Find at least one non-occupied voxel and assert RC2 for it (and all of them).
            int nonOccupiedSeen = 0;
            for (int v = 0; v < rb.VoxelCount; v++)
            {
                if (occ[v] != 0u) continue;
                nonOccupiedSeen++;
                // (a1) No valid per-component cut FP in an empty voxel.
                for (int comp = 0; comp < 8; comp++)
                    Assert.LessOrEqual(cutFP[8 * v + comp].w, 0.5f,
                        $"Non-occupied voxel {v} slot {comp} must have an INVALID _CutFP (w<=0.5)");
                // (a2) No emitted triangle (skin or wall) may reference an empty voxel.
                Assert.IsFalse(referencedVoxels.Contains(v),
                    $"No triangle may reference non-occupied voxel {v} (RC2: no flap in empty space)");
            }
            Assert.Greater(nonOccupiedSeen, 0,
                "Scenario must contain ≥1 non-occupied (all-outside) voxel to exercise the RC2 gate");
        }
    }

    // ── Test 5: RC1 — the outer skin splits per component (plan §7 test (b), the headline) ───────
    [Test]
    [Category("GPU")]
    public void RC1_DC_Stitch_OuterSkin_SplitsPerComponent()
    {
        var dims = new int3(3, 3, 3);
        float D  = 0.2f;
        var recon = LoadReconShader();
        Assert.IsNotNull(recon, "Recon.compute must be loadable");
        BuildPlanarCut(out var g, out var rb, out var cs, dims, 1f, D);
        using (rb)
        {
            // Fill _IsectWorld + _VoxelExternalFP, then the 2C FP block → per-component cut FPs.
            RunReconFP(recon, dims, rb, 1f, 20);
            RunFpBlock(cs, dims, rb, 1f, 20);

            // Run ONLY DC_Stitch (NO BuildCutTriangles) so every INTERNAL tag in _Tri comes from the
            // SKIN — this is what proves RC1 (the outer surface, not the walls, now references cut FPs).
            rb.TriCounter.SetData(new uint[] { 0 });
            recon.SetInt("_VoxelCount",       rb.VoxelCount);
            recon.SetInts("_Dims",            dims.x, dims.y, dims.z);
            recon.SetInt("_SurfaceEdgeCount", rb.SurfaceEdgeCount);
            recon.SetInt("_TriCapacity",      rb.TriCapacity);
            int kStitch = recon.FindKernel("DC_Stitch");
            BindReconAll(recon, kStitch, rb);
            recon.Dispatch(kStitch, Mathf.Max(1, Mathf.CeilToInt(rb.SurfaceEdgeCount / 64f)), 1, 1);

            var counter = new uint[1];
            rb.TriCounter.GetData(counter);
            int idxCount = (int)Mathf.Min(counter[0], (uint)(3 * rb.TriCapacity));
            Assert.Greater(idxCount, 0, "DC_Stitch must emit surface triangles for the slab boundary");
            var tri = new int[3 * rb.TriCapacity];
            rb.Tri.GetData(tri);

            // STRICT-PAPER watertight fix (2026-06-29, workflow w18dv69wq): the cut-voxel SKIN now shares the
            // per-component cut-point centroid _CutFP (bit31) — the SAME vertex the cut WALL uses — so skin
            // and wall weld into one watertight surface (no rim hole). (A prior separate per-component
            // External-QEF skin vertex on bit30 was removed; it had left the red-wall/green-skin gap.)
            var cutFP = new float4[8 * rb.VoxelCount];
            rb.CutFP.GetData(cutFP);

            // This test runs ONLY DC_Stitch (no BuildCutTriangles), so every bit31 (0x80000000) tag in _Tri
            // comes from the SKIN — proving the outer surface references the per-component cut FP. With the
            // old component-blind DC_Stitch this set would be EMPTY (the skin only used voxelId external tags).
            var skinInternalSlots = new HashSet<int>();
            for (int i = 0; i < idxCount; i++)
            {
                uint u = (uint)tri[i];
                if ((u & 0x80000000u) == 0u) continue;                   // not a per-component cut-FP skin vertex
                int slot = (int)(u & 0x3FFFFFFFu);
                Assert.Greater(cutFP[slot].w, 0.5f, $"Skin cut-FP tag slot {slot} must be a valid per-component _CutFP");
                skinInternalSlots.Add(slot);
            }

            Assert.Greater(skinInternalSlots.Count, 0,
                "RC1: the outer skin must attach to ≥1 per-component cut FP (the skin no longer welds " +
                "straight through the cut). A component-blind DC_Stitch would produce ZERO cut tags.");

            // No re-weld: the centre voxel's TWO component slots are distinct; the skin on the two
            // z-sides of the blade attaches to DIFFERENT slots → the surface opens (does not weld).
            ConnectivityLUT.Build(out _, out int[] vertToComp);
            int centre = GridConventions.VoxelId(1, 1, 1, dims);
            int compBottom = vertToComp[0xF00 * 8 + 0];
            int compTop    = vertToComp[0xF00 * 8 + 4];
            int slotBottom = 8 * centre + compBottom;
            int slotTop    = 8 * centre + compTop;
            Assert.AreNotEqual(slotBottom, slotTop, "Centre voxel's two component slots must be distinct");
            Assert.IsTrue(skinInternalSlots.Contains(slotBottom) || skinInternalSlots.Contains(slotTop),
                "The skin must reference at least one of the centre voxel's per-component cut FPs " +
                "(the surface sheet terminates at the split cut vertex instead of welding through)");
        }
    }

    // ── Test 6: No re-weld + tag decode (GPU, plan §6.4/§6.5 — the walls split) ─────────────────
    [Test]
    [Category("GPU")]
    public void BuildCutTriangles_EmitsInternalTaggedTris_DifferentSlots_NoReweld()
    {
        var dims = new int3(3, 3, 3);
        float D  = 0.2f;
        var recon = LoadReconShader();
        Assert.IsNotNull(recon, "Recon.compute must be loadable");
        BuildPlanarCut(out var g, out var rb, out var cs, dims, 1f, D);
        using (rb)
        {
            RunReconFP(recon, dims, rb, 1f, 20);   // _IsectWorld for ComputeComponentFP
            RunFpBlock(cs, dims, rb, 1f, 20);

            // Reset triCounter (head of geometry) then run BuildCutTriangles (no DC_Stitch here →
            // every emitted tri is a CUT WALL tri, so the tag decode is unambiguous).
            rb.TriCounter.SetData(new uint[] { 0 });
            SetCutScalars(cs, dims, rb, 1f, 20);
            int kBuild = cs.FindKernel("BuildCutTriangles");
            BindCut(cs, kBuild, rb);
            cs.Dispatch(kBuild, Mathf.CeilToInt(rb.CornerCount / 64f), 1, 1);

            var counter = new uint[1];
            rb.TriCounter.GetData(counter);
            uint idxCount = counter[0];
            Assert.Greater(idxCount, 0u, "BuildCutTriangles must emit at least one cut-wall triangle");
            Assert.AreEqual(0u, idxCount % 3u, "Index count must be a multiple of 3 (whole triangles)");

            int triCount = (int)(idxCount / 3u);
            var tri = new int[3 * rb.TriCapacity];
            rb.Tri.GetData(tri);

            // Every emitted vertex must carry the internal tag (bit31 set) and index a VALID _CutFP slot.
            var cutFP = new float4[8 * rb.VoxelCount];
            rb.CutFP.GetData(cutFP);
            var occ = new uint[rb.VoxelCount];
            rb.VoxelOccupied.GetData(occ);
            int internalCount = 0;
            var slotsSeen = new HashSet<int>();
            for (int i = 0; i < triCount * 3; i++)
            {
                int raw = tri[i];
                bool internalTag = (raw & unchecked((int)0x80000000)) != 0;
                Assert.IsTrue(internalTag, $"Cut tri vertex {i} must carry the internal tag (bit31 set)");
                // mask 0x3FFFFFFF strips bit31 (cut FP, skin+wall shared) → slot domain (the bit30 External-cut tag was removed).
                int slot = raw & 0x3FFFFFFF;
                Assert.GreaterOrEqual(slot, 0);
                Assert.Less(slot, 8 * rb.VoxelCount, "Tag low bits must index a valid _CutFP slot");
                Assert.Greater(cutFP[slot].w, 0.5f, $"Tagged slot {slot} must be a valid FP");
                // RC2: a wall vertex may only live in an occupied voxel.
                Assert.AreEqual(1u, occ[slot / 8], $"Cut-wall vertex slot {slot} must be in an occupied voxel");
                slotsSeen.Add(slot);
                internalCount++;
            }
            Assert.Greater(internalCount, 0, "At least one internal-tagged cut vertex must exist (tag decode)");

            // No re-weld: the centre voxel's two component slots are DIFFERENT — a bottom-side particle
            // selects the bottom component FP, a top-side particle the top component FP. If they ever
            // collapsed to one slot the seam would re-weld (the MC2024 failure). Assert both centre
            // slots were used and are distinct.
            ConnectivityLUT.Build(out _, out int[] vertToComp);
            int centre = GridConventions.VoxelId(1, 1, 1, dims);
            int compBottom = vertToComp[0xF00 * 8 + 0];
            int compTop    = vertToComp[0xF00 * 8 + 4];
            int slotBottom = 8 * centre + compBottom;
            int slotTop    = 8 * centre + compTop;
            Assert.AreNotEqual(slotBottom, slotTop, "Centre voxel's two component slots must be distinct");
            Assert.IsTrue(slotsSeen.Contains(slotBottom) && slotsSeen.Contains(slotTop),
                "Both centre-voxel component slots must appear in the emitted cut tris (no re-weld: " +
                "the two faces come from DIFFERENT slots)");
        }
    }
}
