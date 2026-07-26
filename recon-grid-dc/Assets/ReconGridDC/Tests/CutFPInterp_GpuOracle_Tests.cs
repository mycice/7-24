// CutFPInterp_GpuOracle_Tests.cs — Stage 2D-2b GPU oracle tests (plan §5 / §7)
//
// Verifies InterpCutFP (Cutting.compute) — the temporal interpolation of the post-cut feature point
// (paper Discussion, newpaper_clean.txt:1271-1276). The kernel EMA-smooths _CutFP (curr) toward its
// previous DISPLAYED value (_PrevCutFP) with factor α = _CutFPInterp, and persists the displayed FP back
// to _PrevCutFP for the next frame:
//   curr.w < 0.5      → outv = 0          (no cut FP now → invalid)
//   prev.w < 0.5      → outv = curr       (newly valid → appear at current; no interp on first appearance)
//   else              → outv = lerp(prev.xyz, curr.xyz, α), w=1  (EMA smooth the drift)
//   _CutFP[slot] = _PrevCutFP[slot] = outv
//
// Tests (one thread per slot; the guard mirrors ClearCutFP = 8u*_VoxelCount):
//   1. α=1, prev valid, curr valid           ⇒ _CutFP == curr (the byte-identical OFF switch).
//   2. α=0.25, prev valid, curr valid        ⇒ _CutFP.xyz == lerp(prev,curr,0.25) AND _PrevCutFP == same.
//   3. prev invalid (w=0), curr valid        ⇒ _CutFP == curr (newly-valid appear).
//   4. curr invalid (w=0)                     ⇒ _CutFP.w == 0 (dropped).
//   5. EMA convergence: fixed curr, N dispatches (kernel rewrites _PrevCutFP each time; we re-seed
//      _CutFP=curr per iter) ⇒ _CutFP → curr geometrically (close after ~30 iters at α=0.25).
//
// Mirrors CutBitPropagation_GpuOracle_Tests / CutSurface_GpuOracle_Tests setup (LoadCuttingShader,
// BackgroundGrid.Build + ReconBuffers.Upload, direct SetData seeding, dispatch, GetData, assert).
//
// [Category("GPU")] — requires a Unity GPU context.

using NUnit.Framework;
using Unity.Mathematics;
using UnityEngine;
using ReconGridDC.Preprocess;
using ReconGridDC.Recon;

#if UNITY_EDITOR
using UnityEditor;
#endif

[Category("GPU")]
public class CutFPInterp_GpuOracle_Tests
{
    // Flat level-set: everything inside (phi<0) → a small fully-occupied grid (no surface edges needed;
    // we seed _CutFP / _PrevCutFP directly and only exercise InterpCutFP's slot arithmetic).
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

    // Build a small grid + buffers, seed _CutFP (curr) and _PrevCutFP (prev) directly, dispatch
    // InterpCutFP once at the given α. Returns the post-dispatch _CutFP and _PrevCutFP readbacks.
    static void RunInterp(float alpha, float4[] currSeed, float4[] prevSeed,
                          out float4[] cutOut, out float4[] prevOut)
    {
        var cs = LoadCuttingShader();
        Assert.IsNotNull(cs, "Cutting.compute must be loadable via AssetDatabase or Resources");

        var dims = new int3(3, 3, 3);
        var g = BackgroundGrid.Build(new AllInsideLS(), dims, 1f, float3.zero);
        var rb = new ReconBuffers(g, 1024);
        rb.Upload(g);                                 // zero-inits _CutFP / _PrevCutFP

        int slotCount = 8 * rb.VoxelCount;
        Assert.AreEqual(slotCount, currSeed.Length, "curr seed must cover all slots");
        Assert.AreEqual(slotCount, prevSeed.Length, "prev seed must cover all slots");

        using (rb)
        {
            rb.CutFP.SetData(currSeed);
            rb.PrevCutFP.SetData(prevSeed);

            int k = cs.FindKernel("InterpCutFP");
            Assert.GreaterOrEqual(k, 0, "InterpCutFP kernel must exist in Cutting.compute");
            cs.SetInt("_VoxelCount", rb.VoxelCount);   // guard: slot >= 8*_VoxelCount → return
            cs.SetFloat("_CutFPInterp", alpha);
            cs.SetBuffer(k, "_CutFP",     rb.CutFP);
            cs.SetBuffer(k, "_PrevCutFP", rb.PrevCutFP);
            cs.Dispatch(k, Mathf.Max(1, Mathf.CeilToInt(slotCount / 64f)), 1, 1);

            cutOut  = new float4[slotCount]; rb.CutFP.GetData(cutOut);
            prevOut = new float4[slotCount]; rb.PrevCutFP.GetData(prevOut);
        }
    }

    // Helper: a slot-filled array where slot `idx` carries `val` and every other slot is zero (invalid).
    static float4[] OneSlot(int slotCount, int idx, float4 val)
    {
        var a = new float4[slotCount];
        a[idx] = val;
        return a;
    }

    // ── Test 1: α=1, prev valid, curr valid ⇒ _CutFP == curr (OFF switch) ────────────────────────
    [Test]
    public void Alpha1_PrevValid_CurrValid_IsNoOp()
    {
        int slotCount = 8 * 27;     // 3×3×3 voxels
        int idx = 8 * 13 + 1;       // an arbitrary interior slot
        var curr = new float4(2.0f, 3.0f, 4.0f, 1.0f);
        var prev = new float4(9.0f, 8.0f, 7.0f, 1.0f);

        RunInterp(1.0f, OneSlot(slotCount, idx, curr), OneSlot(slotCount, idx, prev),
                  out var cutOut, out var prevOut);

        Assert.AreEqual(curr.x, cutOut[idx].x, 1e-5f, "α=1 ⇒ _CutFP.x == curr.x");
        Assert.AreEqual(curr.y, cutOut[idx].y, 1e-5f, "α=1 ⇒ _CutFP.y == curr.y");
        Assert.AreEqual(curr.z, cutOut[idx].z, 1e-5f, "α=1 ⇒ _CutFP.z == curr.z");
        Assert.AreEqual(1.0f,   cutOut[idx].w, 1e-5f, "α=1 ⇒ _CutFP.w == 1 (valid)");
        // _PrevCutFP must be updated to the displayed FP (== curr at α=1).
        Assert.AreEqual(curr.x, prevOut[idx].x, 1e-5f, "α=1 ⇒ _PrevCutFP.x == curr.x");
        Assert.AreEqual(curr.z, prevOut[idx].z, 1e-5f, "α=1 ⇒ _PrevCutFP.z == curr.z");
        Assert.AreEqual(1.0f,   prevOut[idx].w, 1e-5f, "α=1 ⇒ _PrevCutFP.w == 1");
    }

    // ── Test 2: α=0.25, prev valid, curr valid ⇒ _CutFP.xyz == lerp(prev,curr,0.25), _Prev == same ──
    [Test]
    public void Alpha025_PrevValid_CurrValid_LerpsAndPersists()
    {
        int slotCount = 8 * 27;
        int idx = 8 * 5 + 3;
        float a = 0.25f;
        var curr = new float4(10.0f, -4.0f, 6.0f, 1.0f);
        var prev = new float4(2.0f,   4.0f, 2.0f, 1.0f);
        float3 expected = math.lerp(prev.xyz, curr.xyz, a);

        RunInterp(a, OneSlot(slotCount, idx, curr), OneSlot(slotCount, idx, prev),
                  out var cutOut, out var prevOut);

        Assert.AreEqual(expected.x, cutOut[idx].x, 1e-5f, "α=0.25 ⇒ _CutFP.x == lerp.x");
        Assert.AreEqual(expected.y, cutOut[idx].y, 1e-5f, "α=0.25 ⇒ _CutFP.y == lerp.y");
        Assert.AreEqual(expected.z, cutOut[idx].z, 1e-5f, "α=0.25 ⇒ _CutFP.z == lerp.z");
        Assert.AreEqual(1.0f,       cutOut[idx].w, 1e-5f, "smoothed FP stays valid (w=1)");
        // _PrevCutFP must equal the same displayed FP (EMA persistence).
        Assert.AreEqual(expected.x, prevOut[idx].x, 1e-5f, "_PrevCutFP.x == displayed lerp.x");
        Assert.AreEqual(expected.y, prevOut[idx].y, 1e-5f, "_PrevCutFP.y == displayed lerp.y");
        Assert.AreEqual(expected.z, prevOut[idx].z, 1e-5f, "_PrevCutFP.z == displayed lerp.z");
        Assert.AreEqual(1.0f,       prevOut[idx].w, 1e-5f, "_PrevCutFP.w == 1");
    }

    // ── Test 3: prev invalid (w=0), curr valid ⇒ _CutFP == curr (newly-valid appear) ──────────────
    [Test]
    public void PrevInvalid_CurrValid_AppearsAtCurrent()
    {
        int slotCount = 8 * 27;
        int idx = 8 * 20 + 6;
        var curr = new float4(1.5f, 2.5f, 3.5f, 1.0f);
        var prev = new float4(99f, 99f, 99f, 0.0f);   // w=0 → invalid (must NOT interp from this)

        RunInterp(0.25f, OneSlot(slotCount, idx, curr), OneSlot(slotCount, idx, prev),
                  out var cutOut, out var prevOut);

        Assert.AreEqual(curr.x, cutOut[idx].x, 1e-5f, "newly-valid ⇒ _CutFP.x == curr.x");
        Assert.AreEqual(curr.y, cutOut[idx].y, 1e-5f, "newly-valid ⇒ _CutFP.y == curr.y");
        Assert.AreEqual(curr.z, cutOut[idx].z, 1e-5f, "newly-valid ⇒ _CutFP.z == curr.z");
        Assert.AreEqual(1.0f,   cutOut[idx].w, 1e-5f, "newly-valid ⇒ _CutFP.w == 1");
        Assert.AreEqual(1.0f,   prevOut[idx].w, 1e-5f, "newly-valid ⇒ _PrevCutFP.w == 1 (now seeded)");
        Assert.AreEqual(curr.x, prevOut[idx].x, 1e-5f, "newly-valid ⇒ _PrevCutFP.x == curr.x");
    }

    // ── Test 4: curr invalid (w=0) ⇒ _CutFP.w == 0 (dropped) ──────────────────────────────────────
    [Test]
    public void CurrInvalid_Drops()
    {
        int slotCount = 8 * 27;
        int idx = 8 * 9 + 2;
        var curr = new float4(5f, 5f, 5f, 0.0f);       // w=0 → no cut FP now
        var prev = new float4(1f, 2f, 3f, 1.0f);       // had a valid history (irrelevant — must drop)

        RunInterp(0.25f, OneSlot(slotCount, idx, curr), OneSlot(slotCount, idx, prev),
                  out var cutOut, out var prevOut);

        Assert.AreEqual(0.0f, cutOut[idx].w, 1e-5f, "curr invalid ⇒ _CutFP.w == 0 (dropped)");
        Assert.AreEqual(0.0f, cutOut[idx].x, 1e-5f, "dropped slot is zeroed (x)");
        Assert.AreEqual(0.0f, cutOut[idx].y, 1e-5f, "dropped slot is zeroed (y)");
        Assert.AreEqual(0.0f, cutOut[idx].z, 1e-5f, "dropped slot is zeroed (z)");
        Assert.AreEqual(0.0f, prevOut[idx].w, 1e-5f, "curr invalid ⇒ _PrevCutFP.w == 0 (EMA history reset)");
    }

    // ── Test 5: EMA convergence — fixed curr, N dispatches ⇒ _CutFP → curr geometrically ──────────
    [Test]
    public void EmaConverges_FixedCurr_ManyDispatches_ApproachesCurr()
    {
        var cs = LoadCuttingShader();
        Assert.IsNotNull(cs, "Cutting.compute must be loadable");

        var dims = new int3(3, 3, 3);
        var g = BackgroundGrid.Build(new AllInsideLS(), dims, 1f, float3.zero);
        var rb = new ReconBuffers(g, 1024);
        rb.Upload(g);

        int slotCount = 8 * rb.VoxelCount;
        int idx = 8 * 13 + 0;
        float a = 0.25f;
        var curr = new float4(10.0f, 20.0f, -5.0f, 1.0f);
        var prev = new float4(0.0f,   0.0f,  0.0f, 1.0f);   // valid history far from curr

        using (rb)
        {
            // Seed once: _PrevCutFP = prev (valid), _CutFP = curr.
            rb.PrevCutFP.SetData(OneSlot(slotCount, idx, prev));

            int k = cs.FindKernel("InterpCutFP");
            cs.SetInt("_VoxelCount", rb.VoxelCount);
            cs.SetFloat("_CutFPInterp", a);
            cs.SetBuffer(k, "_CutFP",     rb.CutFP);
            cs.SetBuffer(k, "_PrevCutFP", rb.PrevCutFP);
            int groups = Mathf.Max(1, Mathf.CeilToInt(slotCount / 64f));

            const int N = 30;
            for (int it = 0; it < N; it++)
            {
                // Re-seed _CutFP = curr each iteration (the prior dispatch overwrote _CutFP with the
                // smoothed value; ComputeComponentFP would refresh it to curr in the real pipeline).
                // _PrevCutFP PERSISTS on the GPU (the kernel wrote this iter's displayed FP into it).
                rb.CutFP.SetData(OneSlot(slotCount, idx, curr));
                cs.Dispatch(k, groups, 1, 1);
            }

            var cutOut = new float4[slotCount]; rb.CutFP.GetData(cutOut);

            // After N iterations the EMA error decays by (1-α)^N. (1-0.25)^30 ≈ 1.8e-4 ⇒ each axis is
            // within ≈ |curr-prev|*1.8e-4 of curr. Use a generous numeric tolerance.
            float decay = Mathf.Pow(1f - a, N);
            float tol = Mathf.Max(1e-3f, decay * math.length(curr.xyz - prev.xyz) * 2f);
            Assert.AreEqual(curr.x, cutOut[idx].x, tol, $"EMA must converge to curr.x within {tol}");
            Assert.AreEqual(curr.y, cutOut[idx].y, tol, $"EMA must converge to curr.y within {tol}");
            Assert.AreEqual(curr.z, cutOut[idx].z, tol, $"EMA must converge to curr.z within {tol}");
            Assert.AreEqual(1.0f,   cutOut[idx].w, 1e-5f, "converged slot stays valid");

            // Monotone-ish sanity: the converged value is much closer to curr than the seed prev was.
            float distFinal = math.length(cutOut[idx].xyz - curr.xyz);
            float distSeed  = math.length(prev.xyz - curr.xyz);
            Assert.Less(distFinal, distSeed * 0.01f,
                "After 30 EMA steps the displayed FP must be far closer to curr than the seed was");
        }
    }

    // Run N frames, each ADVANCING curr by `step` (a constant-velocity "fall"), at factor α; seed prev=curr0
    // so the lag starts at 0 and builds to its steady state. Returns the final displayed FP (read from _CutFP)
    // and the final curr. _PrevCutFP persists on the GPU across frames (the kernel rewrites it each dispatch).
    static void RunEmaConstantVelocity(float alpha, int idx, float4 curr0, float3 step, int frames,
                                       out float3 displayedFinal, out float3 currFinal)
    {
        var cs = LoadCuttingShader();
        Assert.IsNotNull(cs, "Cutting.compute must be loadable");
        var dims = new int3(3, 3, 3);
        var g = BackgroundGrid.Build(new AllInsideLS(), dims, 1f, float3.zero);
        var rb = new ReconBuffers(g, 1024);
        rb.Upload(g);
        int sc = 8 * rb.VoxelCount;
        using (rb)
        {
            rb.PrevCutFP.SetData(OneSlot(sc, idx, curr0));   // displayed_0 = curr0 ⇒ initial lag 0
            int k = cs.FindKernel("InterpCutFP");
            cs.SetInt("_VoxelCount", rb.VoxelCount);
            cs.SetFloat("_CutFPInterp", alpha);
            cs.SetBuffer(k, "_CutFP",     rb.CutFP);
            cs.SetBuffer(k, "_PrevCutFP", rb.PrevCutFP);
            int groups = Mathf.Max(1, Mathf.CeilToInt(sc / 64f));
            float3 curr = curr0.xyz;
            for (int f = 0; f < frames; f++)
            {
                curr += step;                                            // the falling piece advances each frame
                rb.CutFP.SetData(OneSlot(sc, idx, new float4(curr, 1f))); // ComputeComponentFP would refresh _CutFP=curr
                cs.Dispatch(k, groups, 1, 1);                            // _CutFP ← lerp(prev, curr, α); _PrevCutFP persists
            }
            var cutOut = new float4[sc]; rb.CutFP.GetData(cutOut);
            displayedFinal = cutOut[idx].xyz;
            currFinal = curr;
        }
    }

    // ── Test 6: the DELAMINATION mechanism — EMA steady-state lag under a constant fall ───────────────
    // _CutFP is EMA-smoothed by InterpCutFP. This test pins the resulting steady-state lag d·(1-α)/α (where
    // d is the per-frame fall step): at α=0.5 it equals d (one full frame-step behind); at α=1.0 (the FIX,
    // now the demo default) it is ZERO → the cut surface tracks the fall exactly → no delamination.
    // (HISTORICAL: the lag once delaminated against a SEPARATE un-smoothed external skin vertex,
    // _CutFPExternal — since removed; the cut skin now shares this same _CutFP, so that gap cannot recur.)
    [Test]
    public void EmaSteadyStateLag_IsZeroAtAlpha1_AndOneStepAtAlphaHalf()
    {
        int idx = 8 * 13 + 0;
        var curr0 = new float4(1f, 5f, 2f, 1f);
        float3 step = new float3(0f, -0.1f, 0f);   // fall 0.1 world-units/frame in −y
        const int frames = 80;                     // ≫ 1/α ⇒ fully converged to steady state

        // α = 1.0 (the FIX): lerp(prev,curr,1)=curr every frame ⇒ the wall == the falling piece, ZERO lag.
        RunEmaConstantVelocity(1.0f, idx, curr0, step, frames, out var disp1, out var curr1);
        float lag1 = math.abs(curr1.y - disp1.y);
        Assert.Less(lag1, 1e-3f,
            $"α=1.0 ⇒ the cut wall tracks the fall with ZERO lag (lag={lag1:F4}); this is the delamination fix.");

        // α = 0.5 (the OLD demo value = the BUG): steady-state lag = d·(1-α)/α = 0.1·0.5/0.5 = 0.1.
        RunEmaConstantVelocity(0.5f, idx, curr0, step, frames, out var dispH, out var currH);
        float lagH = math.abs(currH.y - dispH.y);
        float expected = 0.1f * (1f - 0.5f) / 0.5f;   // = 0.1
        Assert.AreEqual(expected, lagH, 5e-3f,
            $"α=0.5 ⇒ the cut WALL lags the fall by d(1-α)/α={expected:F3} (lag={lagH:F4}); the un-smoothed " +
            "skin does not, so the wall delaminates below the dome.");

        // Discrimination: the bug lags measurably; the fix does not.
        Assert.Greater(lagH, lag1 + 0.05f,
            $"The differential lag must be visible: α=0.5 lag {lagH:F4} ≫ α=1.0 lag {lag1:F4} = the delamination.");
    }
}
