// Sever_GpuOracle_Tests.cs — Stage 2D-1 GPU oracle for SeverLinks (paper §2.1.2 line 367).
// Verifies that SeverLinks, given a cut edge, writes the physics topology to "dead":
//   1. Both endpoints' structural neighbor slots toward each other → -1 (NbrIdx).
//   2. Every bending pair whose arm crosses the cut edge → alive == 0.
//   3. Uncut control edges/pairs are UNTOUCHED (no over-kill).
//   4. Idempotent: a second dispatch changes nothing.
// The force kernels already SKIP these (AccumulateStructural _NbrIdx<0 @Physics.compute:94;
// AccumulateBending alive==0 @:128 — covered by StructuralForce/BendingForce oracles), so
// NbrIdx==-1 + alive==0 is exactly "zero force transmitted across the severed edge".
//
// [Category("GPU")] — requires Unity GPU context. Dispatch pattern mirrors CutBitPropagation_GpuOracle_Tests.

using NUnit.Framework;
using Unity.Mathematics;
using UnityEngine;
using ReconGridDC.Preprocess;
using ReconGridDC.Recon;
using ReconGridDC.Core;

#if UNITY_EDITOR
using UnityEditor;
#endif

[Category("GPU")]
public class Sever_GpuOracle_Tests
{
    // Flat level-set: everything inside → all interior grid edges + full physics topology present.
    sealed class AllInsideLS : ILevelSetProvider
    {
        public float  Sample(float3 p)   => -1f;
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
    public void SeverLinks_CutEdge_SeversBothNbrSlots_AndBridgingBendPairs_NoOverkill()
    {
        var cs = LoadCuttingShader();
        Assert.IsNotNull(cs, "Cutting.compute must be loadable");
        int kSever = cs.FindKernel("SeverLinks");
        Assert.GreaterOrEqual(kSever, 0, "SeverLinks kernel not found");

        // ── 3×3×3 grid (4×4×4 corners), L=1 ───────────────────────────────────────────────
        var dims3 = new int3(3, 3, 3);
        float L   = 1f;
        var g     = BackgroundGrid.Build(new AllInsideLS(), dims3, L, float3.zero);

        // Target: the z-axis grid edge at baseCorner=(1,1,1) → A=(1,1,1), B=(1,1,2).
        int idA = GridConventions.CornerId(1, 1, 1, dims3);
        int idB = GridConventions.CornerId(1, 1, 2, dims3);
        // Control (uncut) z-edge at (2,2,2)→(2,2,3).
        int idC = GridConventions.CornerId(2, 2, 2, dims3);
        int idCz = GridConventions.CornerId(2, 2, 3, dims3);

        // Slot map (NbrDelta): 0=-x,1=+x,2=-y,3=+y,4=-z,5=+z. z-edge: A→B is +z (slot 5); B→A is -z (slot 4).
        const int SP_Z = 5, SM_Z = 4;

        using (var rb = new ReconBuffers(g, 1024))
        {
            rb.Upload(g); // populates NbrIdx, BendPairs (alive=1), VoxelCutMask=0

            int cornerCount   = g.cornerCount;
            int bendPairCount = 15 * cornerCount;

            // ── Pre-state: read NbrIdx + BendPairs (BendPairGpu = 8 ints/32B: i,j,k,theta0,alive,3 pad) ──
            var nbr0 = new int[6 * cornerCount];   rb.NbrIdx.GetData(nbr0);
            var bp0  = new int[8 * bendPairCount]; rb.BendPairs.GetData(bp0);

            Assert.AreEqual(idB, nbr0[6 * idA + SP_Z], "pre: A's +z neighbor must be B");
            Assert.AreEqual(idA, nbr0[6 * idB + SM_Z], "pre: B's -z neighbor must be A");

            // Bending pairs centered at A whose arm == B (alive before sever) — these MUST die.
            var bridgeA = new System.Collections.Generic.List<int>();
            // A control pair centered at A that is alive and does NOT have B as an arm — must SURVIVE.
            int controlPair = -1;
            for (int p = 0; p < 15; p++)
            {
                int idx = 15 * idA + p;
                int bi = bp0[8 * idx + 0], bj = bp0[8 * idx + 1], bk = bp0[8 * idx + 2], al = bp0[8 * idx + 4];
                if (bi != idA) continue;
                if (al == 1 && (bj == idB || bk == idB)) bridgeA.Add(idx);
                else if (al == 1 && bj != idB && bk != idB && controlPair < 0) controlPair = idx;
            }
            Assert.Greater(bridgeA.Count, 0, "there must be ≥1 alive bending pair at A bridging the cut edge");
            Assert.GreaterOrEqual(controlPair, 0, "there must be an alive control pair at A NOT bridging the cut");

            // ── Set the cut bits for the target z-edge in its 4 stencil voxels (EdgeStencil[2]) ─────
            var mask = new uint[g.voxelCount];
            var baseC = new int3(1, 1, 1);
            var stencil = GridConventions.EdgeStencil[2]; // z-axis
            for (int s = 0; s < stencil.Length; s++)
            {
                int3 vc = baseC + stencil[s].voxelOffset;
                int vid = GridConventions.VoxelId(vc.x, vc.y, vc.z, dims3);
                mask[vid] |= (uint)(1 << stencil[s].localEdge);
            }
            rb.VoxelCutMask.SetData(mask);

            // ── Dispatch SeverLinks ────────────────────────────────────────────────────────────
            cs.SetInts("_Dims", dims3.x, dims3.y, dims3.z);
            cs.SetInt ("_GridEdgeCount", rb.GridEdgeCount);
            cs.SetBuffer(kSever, "_GridEdges",    rb.GridEdges);
            cs.SetBuffer(kSever, "_VoxelCutMask", rb.VoxelCutMask);
            cs.SetBuffer(kSever, "_NbrIdx",       rb.NbrIdx);
            cs.SetBuffer(kSever, "_BendPairs",    rb.BendPairs);
            int groups = Mathf.Max(1, Mathf.CeilToInt(rb.GridEdgeCount / 64f));
            cs.Dispatch(kSever, groups, 1, 1);

            // ── Assert: structural link severed BOTH directions ─────────────────────────────────
            var nbr1 = new int[6 * cornerCount]; rb.NbrIdx.GetData(nbr1);
            Assert.AreEqual(-1, nbr1[6 * idA + SP_Z], "A's +z slot toward B must be severed (-1)");
            Assert.AreEqual(-1, nbr1[6 * idB + SM_Z], "B's -z slot toward A must be severed (-1)");
            // Control edge untouched.
            Assert.AreEqual(idCz, nbr1[6 * idC + SP_Z], "uncut control edge must be UNTOUCHED");
            // A's other slots (e.g. +x=1) untouched.
            Assert.AreEqual(nbr0[6 * idA + 1], nbr1[6 * idA + 1], "A's +x neighbor must be untouched (no over-kill)");

            // ── Assert: bridging bending pairs dead; control pair alive ─────────────────────────
            var bp1 = new int[8 * bendPairCount]; rb.BendPairs.GetData(bp1);
            foreach (int idx in bridgeA)
                Assert.AreEqual(0, bp1[8 * idx + 4], $"bending pair {idx} (arm crosses cut) must be alive==0");
            Assert.AreEqual(1, bp1[8 * controlPair + 4], "control bending pair (no cut arm) must stay alive==1");

            // ── Idempotency: a second dispatch changes nothing ──────────────────────────────────
            cs.Dispatch(kSever, groups, 1, 1);
            var nbr2 = new int[6 * cornerCount]; rb.NbrIdx.GetData(nbr2);
            var bp2  = new int[8 * bendPairCount]; rb.BendPairs.GetData(bp2);
            for (int i = 0; i < nbr2.Length; i++)
                Assert.AreEqual(nbr1[i], nbr2[i], $"NbrIdx[{i}] must be stable under re-dispatch (idempotent)");
            Assert.AreEqual(1, bp2[8 * controlPair + 4], "control pair still alive after re-dispatch");
        }
    }
}
