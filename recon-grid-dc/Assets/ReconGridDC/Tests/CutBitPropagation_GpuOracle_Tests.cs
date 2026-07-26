// CutBitPropagation_GpuOracle_Tests.cs — Stage 2B GPU oracle tests
// Verifies that DetectCut correctly:
//   1. Sets the cut bit in ALL 4 voxels sharing the cut edge (via EdgeStencil). design §3.3.
//   2. Emits exactly 2 cut points, each owned by the correct endpoint particle. design §3.6.
//   3. The gap between the two P_side positions = D (±D/2 in opposite directions). design §5.3.
//
// Test scenario:
//   A 3×3×3 voxel grid (4×4×4 corners), L=1, origin=(0,0,0).
//   A z-axis grid edge at baseCorner=(1,1,1) → A=(1,1,1), B=(1,1,2) (centre of the grid
//   → all 4 incident voxels are in-bounds: x-vox={0,1}, y-vox={0,1}).
//   A swept plane through z=1.5 (the midpoint of the edge) — T1/T2 form a large quad
//   whose triangles span z=1.5 across the full XY extent so the edge definitely hits.
//
// Expected:
//   - VoxelCutMask has bit 10 set (local edge 10 = the z-edge at local corner 2 of each voxel
//     — mapped via GridConventions.EdgeStencil[2][*].localEdge) in all 4 sharing voxels.
//   - CutPointCounter = 2.
//   - The two cut points: each P_side = P_hit ± (D/2)*n_cut from the SAME P_hit.
//     |P_side_A - P_side_B| = D (gap test).
//
// [Category("GPU")] — requires Unity GPU context.

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

[Category("GPU")]
public class CutBitPropagation_GpuOracle_Tests
{
    // Flat level-set: everything "inside" (phi < 0) so no surface edges
    // but ALL grid edges qualify as interior (no sign-change check in gridEdges).
    sealed class AllInsideLS : ILevelSetProvider
    {
        public float  Sample(float3 p)   => -1f; // inside everywhere
        public float3 Gradient(float3 p) => new float3(0, 0, 1);
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

    [Test]
    public void DetectCut_OneEdge_FourVoxelBits_TwoCutPoints_GapEqualsD()
    {
        var cs = LoadCuttingShader();
        Assert.IsNotNull(cs, "Cutting.compute must be loadable via AssetDatabase or Resources");

        // ── Build 3×3×3 grid ──────────────────────────────────────────────────────────────
        // dims=(3,3,3): 3 voxels per axis, 4 corners per axis.
        // All corners inside (AllInsideLS) → gridEdges has all interior edges (4×4×4 grid).
        var dims3  = new int3(3, 3, 3);
        float L    = 1f;
        var g      = BackgroundGrid.Build(new AllInsideLS(), dims3, L, float3.zero);

        Assert.IsNotNull(g.gridEdges, "gridEdges must be populated by BackgroundGrid.Build");
        Assert.Greater(g.gridEdges.Length, 0, "gridEdges must have at least one entry");

        // ── Identify the target edge: z-axis, baseCorner=(1,1,1) ─────────────────────────
        // A=(1,1,1), B=(1,1,2). Midpoint at z=1.5.
        // 4 incident voxels for a z-edge at (1,1,1) from EdgeStencil[2]:
        //   slot 0: voxelOffset=(0,0,0)   → voxel (1,1,1) → localEdge 8
        //   slot 1: voxelOffset=(-1,0,0)  → voxel (0,1,1) → localEdge 9
        //   slot 2: voxelOffset=(-1,-1,0) → voxel (0,0,1) → localEdge 10
        //   slot 3: voxelOffset=(0,-1,0)  → voxel (1,0,1) → localEdge 11
        // All 4 voxels: x∈[0,2], y∈[0,2], z∈[0,2] → in-bounds ✓.
        int targetEdgeIdx = -1;
        for (int i = 0; i < g.gridEdges.Length; i++)
        {
            var ge = g.gridEdges[i];
            if (ge.axis == 2
             && ge.baseCorner.x == 1 && ge.baseCorner.y == 1 && ge.baseCorner.z == 1)
            {
                targetEdgeIdx = i;
                break;
            }
        }
        Assert.GreaterOrEqual(targetEdgeIdx, 0,
            "Target z-edge at (1,1,1) must exist in gridEdges");

        // Corner world positions: A=(1,1,1), B=(1,1,2) at L=1.
        // cornerPos: straight grid, no deformation.
        int cnx = dims3.x + 1; // = 4
        int cny = dims3.y + 1;
        // Corner A index: CornerId(1,1,1,dims3)
        int idA = GridConventions.CornerId(1, 1, 1, dims3);
        int idB = GridConventions.CornerId(1, 1, 2, dims3);

        float3 posA = g.cornerPos[idA]; // (1,1,1) world
        float3 posB = g.cornerPos[idB]; // (1,1,2) world
        // P_hit at t=0.5: (1, 1, 1.5)

        // ── Swept plane geometry ───────────────────────────────────────────────────────────
        // A large quad at z=1.5 so ONLY our target edge (z∈[1,2]) is hit.
        // T1=(Sprev,Eprev,E_cur), T2=(Sprev,E_cur,S_cur).
        // Place Sprev=(-5,-5,1.5), Eprev=(5,-5,1.5), E_cur=(5,5,1.5), S_cur=(-5,5,1.5).
        // n_cut = normalize(cross(Eprev-Sprev, E_cur-Sprev))
        //       = normalize(cross((10,0,0),(10,10,0))) = normalize((0,0,100)) = (0,0,1).
        Vector3 sprev = new Vector3(-5, -5, 1.5f);
        Vector3 eprev = new Vector3( 5, -5, 1.5f);
        Vector3 ecur  = new Vector3( 5,  5, 1.5f);
        Vector3 scur  = new Vector3(-5,  5, 1.5f);

        float D = 0.2f;
        var tool = new CuttingTool(scur, ecur, D);
        tool.Advance(scur, ecur); // Advance triggers valid=true (but same pos — need a real advance)

        // Force meaningful Sprev/Eprev: re-create tool with Sprev/Eprev baked in.
        // CuttingTool ctor sets Sprev=S, Eprev=E, then Advance rolls them.
        // So: ctor(sprev, eprev, D) → tool.S=sprev, tool.Sprev=sprev; then Advance(scur,ecur):
        //   → Sprev=sprev, Eprev=eprev, S=scur, E=ecur, n_cut=normalize(cross(eprev-sprev, ecur-sprev)).
        tool = new CuttingTool(sprev, eprev, D);
        tool.Advance(scur, ecur);

        Assert.IsTrue(tool.Valid,
            "CuttingTool.Valid must be true for a non-degenerate sweep (large plane)");

        // n_cut should be (0,0,1): normal to the z=1.5 plane.
        Assert.AreEqual(0f, tool.NCut.x, 1e-4f, "n_cut.x should be 0");
        Assert.AreEqual(0f, tool.NCut.y, 1e-4f, "n_cut.y should be 0");
        Assert.AreEqual(1f, Mathf.Abs(tool.NCut.z), 1e-4f, "n_cut.z should be ±1");

        using (var rb = new ReconBuffers(g, 1024))
        {
            rb.Upload(g);

            // ── Dispatch DetectCut ─────────────────────────────────────────────────────
            // Set uniforms manually (matching CutDetector pattern but direct for oracle).
            int k = cs.FindKernel("DetectCut");
            Assert.GreaterOrEqual(k, 0, "DetectCut kernel not found in Cutting.compute");

            cs.SetInts("_Dims", dims3.x, dims3.y, dims3.z);
            cs.SetInt ("_GridEdgeCount", rb.GridEdgeCount);
            cs.SetInt ("_CutPointCapacity", rb.CutPointCapacity); // required by append guard (R2 #2)

            cs.SetVector("_T1V0", tool.T1V0);
            cs.SetVector("_T1V1", tool.T1V1);
            cs.SetVector("_T1V2", tool.T1V2);
            cs.SetVector("_T2V0", tool.T2V0);
            cs.SetVector("_T2V1", tool.T2V1);
            cs.SetVector("_T2V2", tool.T2V2);

            cs.SetVector("_NCut",  tool.NCut);
            cs.SetFloat ("_D",     tool.D);
            cs.SetInt   ("_ToolValid", tool.Valid ? 1 : 0);
            // Rest lattice (DetectCut tests the REST edge = _GridOrigin + coord*_VoxelL). With L=1,
            // origin=0 and no deformation, rest == cornerPos, so all assertions below are unchanged.
            cs.SetFloat ("_VoxelL",     L);
            cs.SetVector("_GridOrigin", Vector3.zero);

            // AABB: encompasses the whole plane
            cs.SetVector("_SweptAABBMin", new Vector3(-6, -6, 1.4f));
            cs.SetVector("_SweptAABBMax", new Vector3( 6,  6, 1.6f));

            cs.SetBuffer(k, "_CornerPos",        rb.CornerPos);
            cs.SetBuffer(k, "_GridEdges",        rb.GridEdges);
            cs.SetBuffer(k, "_VoxelCutMask",     rb.VoxelCutMask);
            cs.SetBuffer(k, "_VoxelOccupied",    rb.VoxelOccupied); // 2C-fix (plan §7): DetectCut now reads the RC2 gate (all-1 for AllInsideLS)
            cs.SetBuffer(k, "_CutPoint",         rb.CutPoint);
            cs.SetBuffer(k, "_CutPointCounter",  rb.CutPointCounter);
            // 2D-2: EmitCutPoint reads _ParticleRot. rb.ParticleRot is identity-initialised by Upload, so
            // localOffset = transpose(I)·(P_side-pPos) = P_side-pPos → the gap=D test below is unchanged.
            cs.SetBuffer(k, "_ParticleRot",      rb.ParticleRot);

            int groups = Mathf.CeilToInt(rb.GridEdgeCount / 64f);
            cs.Dispatch(k, groups, 1, 1);

            // ── Assert: VoxelCutMask in 4 sharing voxels ──────────────────────────────
            // EdgeStencil[2] (z-axis) → voxelOffsets from baseCorner=(1,1,1):
            //   slot 0: offset=(0,0,0)  → voxel(1,1,1), localEdge=8  → bit 8
            //   slot 1: offset=(-1,0,0) → voxel(0,1,1), localEdge=9  → bit 9
            //   slot 2: offset=(-1,-1,0)→ voxel(0,0,1), localEdge=10 → bit 10
            //   slot 3: offset=(0,-1,0) → voxel(1,0,1), localEdge=11 → bit 11
            var stencil = GridConventions.EdgeStencil[2]; // z-axis stencil
            var maskData = new uint[g.voxelCount];
            rb.VoxelCutMask.GetData(maskData);

            var baseC = new int3(1, 1, 1);
            for (int s = 0; s < stencil.Length; s++)
            {
                int3 vc = baseC + stencil[s].voxelOffset;
                int  vid = GridConventions.VoxelId(vc.x, vc.y, vc.z, dims3);
                int  le  = stencil[s].localEdge;
                uint bit = (uint)(1 << le);

                bool set = (maskData[vid] & bit) != 0;
                Assert.IsTrue(set,
                    $"VoxelCutMask[voxel({vc.x},{vc.y},{vc.z})].bit{le} must be set " +
                    $"(stencil slot {s}, EdgeStencil reuse — design §10)");
            }

            // ── Assert: exactly 2 cut points emitted ──────────────────────────────────
            var counterData = new uint[1];
            rb.CutPointCounter.GetData(counterData);
            uint cpCount = counterData[0];

            // Note: DetectCut processes ALL gridEdges; edges parallel to the z=1.5 plane
            // (x- and y-axis edges at z=1 or z=2) that also happen to be exactly AT z=1.5
            // won't be hit. However, other z-axis edges at different x,y (e.g. (0,1,1),(2,1,1)
            // etc.) whose mid-points at z=1.5 are also inside the large test plane WILL be hit.
            // Therefore we assert >= 2 (the target edge contributes at least 2) and that the
            // counter is even (each edge emits exactly 2).
            Assert.GreaterOrEqual((int)cpCount, 2, "Must emit at least 2 cut points (1 edge × 2)");
            Assert.AreEqual(0u, cpCount % 2, "CutPoint count must be even (2 per cut edge)");

            // ── Assert: gap = D for the target edge's cut points ──────────────────────
            // Identify the 2 cut points owned by idA and idB of the target edge.
            var cpData = new ReconBuffers.CutPointGpu[(int)cpCount];
            rb.CutPoint.GetData(cpData, 0, 0, (int)cpCount);

            Vector3? pSideA = null, pSideB = null;
            for (int i = 0; i < (int)cpCount; i++)
            {
                var cp = cpData[i];
                if (cp.edgeId == targetEdgeIdx)
                {
                    // Reconstruct P_side = cornerPos + localOffset
                    // localOffset stored as (localX, localY, localZ) in CutPointGpu
                    float3 ownerPos = g.cornerPos[cp.ownerParticle];
                    Vector3 pSide = new Vector3(
                        ownerPos.x + cp.localX,
                        ownerPos.y + cp.localY,
                        ownerPos.z + cp.localZ);

                    if (cp.ownerParticle == idA) pSideA = pSide;
                    if (cp.ownerParticle == idB) pSideB = pSide;
                }
            }

            Assert.IsTrue(pSideA.HasValue, "Cut point for edge endpoint A (idA) not found");
            Assert.IsTrue(pSideB.HasValue, "Cut point for edge endpoint B (idB) not found");

            // Gap between the two P_side positions must equal D (±D/2 in opposite directions
            // → total gap = D). PAPER-SILENT: ±D/2 offset along n_cut. design §5.3.
            float gap = Vector3.Distance(pSideA.Value, pSideB.Value);
            Assert.AreEqual(D, gap, D * 0.05f,
                $"Gap between cut points must equal D={D} (±D/2 in n_cut direction). " +
                $"Actual gap={gap:F5}. [design §5.3 / PAPER-SILENT ±D/2]");

            // ── Assert: idempotency — re-dispatching the SAME cut does NOT re-emit points ──
            // DetectCut runs every frame and VoxelCutMask persists, so a still-contacting blade
            // re-hits already-cut edges. Cut points must be emitted ONLY on the uncut→cut
            // transition (review R2 #1); otherwise CutPointCounter grows unbounded and 2C
            // centroids get duplicate-polluted. Dispatch again with identical state and verify
            // both the counter and the mask are unchanged.
            cs.Dispatch(k, groups, 1, 1);

            var counterData2 = new uint[1];
            rb.CutPointCounter.GetData(counterData2);
            Assert.AreEqual(cpCount, counterData2[0],
                "Re-dispatching an identical cut must NOT append new cut points " +
                "(emit-on-first-cut idempotency — review R2 #1)");

            var maskData2 = new uint[g.voxelCount];
            rb.VoxelCutMask.GetData(maskData2);
            for (int v = 0; v < g.voxelCount; v++)
                Assert.AreEqual(maskData[v], maskData2[v],
                    $"VoxelCutMask[{v}] must be unchanged on re-dispatch (cut bits are idempotent)");
        }
    }
}
