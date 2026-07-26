// LiverIntegration_Tests.cs  - liver-import pipeline, design §Testing #6/#8/#9.
// CPU tests for the extracted pure helpers (RodMath, LiverGrid) + CuttingTool.Valid.

using NUnit.Framework;
using Unity.Mathematics;
using UnityEngine;
using ReconGridDC.Preprocess;
using ReconGridDC.Demo;
using ReconGridDC.Cutting;

public class LiverIntegration_Tests
{
    // ── §Testing #6: rod input math + sub-stepping + Valid ──────────────────────────────────
    [Test]
    public void RodMath_Endpoints_AndSubStepCount()
    {
        RodMath.Endpoints(new Vector3(0, 0, 0), new Vector3(1, 0, 0), 4f, out Vector3 s, out Vector3 e);
        Assert.AreEqual(new Vector3( 2, 0, 0), s);
        Assert.AreEqual(new Vector3(-2, 0, 0), e);

        // Move 1.0 with L=0.25  - ceil(1.0/0.25) = 4 sub-steps, each travel < L.
        int n = RodMath.SubStepCount(Vector3.zero, Vector3.zero, new Vector3(1, 0, 0), new Vector3(1, 0, 0), 0.25f);
        Assert.AreEqual(4, n);
        Assert.Less(1f / n, 0.25f + 1e-6f, "per-sub-step travel must be < L");

        // Zero motion  - exactly 1 sub-step (no spurious subdivision).
        Assert.AreEqual(1, RodMath.SubStepCount(s, e, s, e, 0.25f));
    }

    [Test]
    public void CuttingTool_Valid_StationaryVsMoving()
    {
        var tool = new CuttingTool(new Vector3(-2, 0, 0), new Vector3(2, 0, 0), 0.05f);
        tool.Advance(new Vector3(-2, 0, 0), new Vector3(2, 0, 0));   // no movement
        Assert.IsFalse(tool.Valid, "stationary rod  - zero swept area  - Valid==false (no spurious cut)");
        tool.Advance(new Vector3(-2, 0, 1), new Vector3(2, 0, 1));   // swept in z
        Assert.IsTrue(tool.Valid, "moved rod  - non-degenerate swept quad  - Valid==true");
    }

    // ── §Testing #8 + #9: empty-grid guard + anchor, on a CPU BoxLevelSet build ─────────────
    [Test]
    public void EmptyGridGuard_AndAnchorPinning_OnBoxGrid()
    {
        var box = new BoxLevelSet(new float3(0, 0, 0), new float3(1, 1, 1));   // box [-1,1]^3
        GridFit.FitToBounds(new float3(-1, -1, -1), new float3(1, 1, 1), 8, 2, out int3 dims, out float L, out float3 origin);
        var g = BackgroundGrid.Build(box, dims, L, origin);

        // #8: a correctly-fit grid has BOTH occupied and empty voxels + a non-empty surface.
        LiverGrid.CountOccupancy(g.voxelOccupied, out bool anyIn, out bool anyOut);
        Assert.IsTrue(anyIn,  "fit grid must contain tissue voxels");
        Assert.IsTrue(anyOut, "fit grid must contain an empty shell");
        Assert.Greater(g.surfaceEdges.Length, 0, "fit grid must produce surface edges");

        // #8 (trip case): a grid placed entirely off the box  - no tissue, no surface.
        var gOut = BackgroundGrid.Build(box, new int3(4, 4, 4), 0.1f, new float3(100, 100, 100));
        Assert.AreEqual(0, gOut.surfaceEdges.Length, "off-model grid must have zero surface edges (guard trips)");

        // #9: the +X anchor band pins  -  active corner; a negative band pins 0 (guard would trip).
        var pinned = new byte[g.cornerCount];
        int nPin = LiverGrid.PinCap(g.cornerPos, g.cornerActive, pinned, new float3(1, 0, 0), L * 1.5f);
        Assert.Greater(nPin, 0, "default anchor band must pin  -  corner");

        var pinnedNone = new byte[g.cornerCount];
        int nNone = LiverGrid.PinCap(g.cornerPos, g.cornerActive, pinnedNone, new float3(1, 0, 0), -1f);
        Assert.AreEqual(0, nNone, "negative band pins 0 (anchor guard trips)");

        // Whole-organ stabilization replaces the cap with a bounded number of distributed interior
        // anchors. The free majority remains deformable under external loading.
        int distributedTarget = 8;
        int nDistributed = LiverGrid.PinDistributed(
            g.cornerPos, g.cornerActive, g.cornerInside, pinned, distributedTarget);
        Assert.AreEqual(distributedTarget, nDistributed);

        int actualPinned = 0;
        float3 pinMin = new float3(float.PositiveInfinity);
        float3 pinMax = new float3(float.NegativeInfinity);
        for (int c = 0; c < g.cornerCount; c++)
        {
            if (pinned[c] == 0) continue;
            actualPinned++;
            Assert.AreEqual(1, g.cornerActive[c], "distributed anchors must be active particles");
            Assert.AreEqual(1, g.cornerInside[c], "distributed anchors should prefer interior particles");
            pinMin = math.min(pinMin, g.cornerPos[c]);
            pinMax = math.max(pinMax, g.cornerPos[c]);
        }
        Assert.AreEqual(distributedTarget, actualPinned, "old cap pins must be cleared before distributed selection");
        float3 covered = pinMax - pinMin;
        Assert.Greater(covered.x, L, "anchors should cover the model in X");
        Assert.Greater(covered.y, L, "anchors should cover the model in Y");
        Assert.Greater(covered.z, L, "anchors should cover the model in Z");
    }
}
