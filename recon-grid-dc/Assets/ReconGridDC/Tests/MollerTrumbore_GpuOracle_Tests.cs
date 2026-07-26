// MollerTrumbore_GpuOracle_Tests.cs — Stage 2B GPU oracle tests
// Dispatches MT_Test kernel (Cutting.compute) with hand-computed inputs and verifies
// the M-T result matches the analytic expectation. User runs on a GPU machine.
//
// Three cases (paper Eq6-8; design §5.2):
//   1. Known hit:    ray O=(0,0,0), Dvec=(0,0,1)  vs  triangle with z=0.5 cross-section
//                   → hit=true, t_ray≈0.5, u/v in range.
//   2. Parallel miss: ray along triangle's plane → hit=false (parallel / det≈0).
//   3. t>1 miss (finite-edge bound, gap C24): infinite ray hits but t_ray>1 → hit=false.
//
// Oracle strategy: reuse _T1V0/_T1V1 uniforms for O/Dvec and _T2V0/V1/V2 for the triangle
// (same names as DetectCut uniforms — MT_Test kernel reads them this way per Cutting.compute).
//
// [Category("GPU")] — requires Unity GPU context; skips in CPU-only CI.

using NUnit.Framework;
using UnityEngine;

#if UNITY_EDITOR
using UnityEditor;
#endif

[Category("GPU")]
public class MollerTrumbore_GpuOracle_Tests
{
    // ── Shader loading ────────────────────────────────────────────────────────────────────
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

    static (ComputeShader cs, int k, ComputeBuffer results) SetupMTTest()
    {
        var cs = LoadCuttingShader();
        Assert.IsNotNull(cs, "Cutting.compute must be loadable via AssetDatabase or Resources");
        int k = cs.FindKernel("MT_Test");
        Assert.GreaterOrEqual(k, 0, "MT_Test kernel not found in Cutting.compute");
        // Results: [0]=t_ray [1]=u_bary [2]=v_bary [3]=hitFlag
        var results = new ComputeBuffer(4, 4); // float[4]
        cs.SetBuffer(k, "_MTResults", results);
        return (cs, k, results);
    }

    // Helper: set O and Dvec via the _T1V0/_T1V1 convention used by MT_Test
    static void SetRay(ComputeShader cs, Vector3 O, Vector3 Dvec)
    {
        cs.SetVector("_T1V0", O);
        cs.SetVector("_T1V1", Dvec);
    }

    // Helper: set triangle via _T2V0/V1/V2
    static void SetTri(ComputeShader cs, Vector3 V0, Vector3 V1, Vector3 V2)
    {
        cs.SetVector("_T2V0", V0);
        cs.SetVector("_T2V1", V1);
        cs.SetVector("_T2V2", V2);
    }

    // ── Test 1: Known hit at t≈0.5 ───────────────────────────────────────────────────────
    // Ray: O=(0,0,0), Dvec=(0,0,1) (unit z-edge of length 1).
    // Triangle: V0=(-1,-1,0.5), V1=(1,-1,0.5), V2=(0,1,0.5).
    //   Plane z=0.5; ray hits at t=0.5.
    //   L1=(2,0,0), L2=(1,2,0), T_vec=(-1,-1,0.5)-0=(-.1,-.1,.5).
    //   det = dot(cross(L2,Dvec), L1).
    //   cross(L2=(1,2,0), Dvec=(0,0,1)) = (2*1-0*0, 0*0-1*1, 1*0-2*0) = (2,-1,0).
    //   det = dot((2,-1,0), (2,0,0)) = 4.
    //   t = dot(cross(L2,T_vec), L1) / det.
    //   cross(L2=(1,2,0), T_vec=(-1,-1,0.5)) = (2*0.5-0*(-1), 0*(-1)-1*0.5, 1*(-1)-2*(-1))
    //                                         = (1, -0.5, 1).
    //   t = dot((1,-0.5,1), (2,0,0)) / 4 = 2/4 = 0.5. ✓
    //   u_bary = -dot(cross(L2,Dvec), T_vec)/det = -dot((2,-1,0),(-1,-1,0.5))/4
    //          = -(-2+1+0)/4 = -(-1)/4 = 0.25. ≥0. ✓
    //   v_bary = dot(cross(Dvec,T_vec), L1)/det.
    //   cross(Dvec=(0,0,1), T_vec=(-1,-1,0.5)) = (0*0.5-1*(-1), 1*(-1)-0*0.5, 0*(-1)-0*(-1))
    //                                           = (1, -1, 0).
    //   v_bary = dot((1,-1,0),(2,0,0))/4 = 2/4 = 0.5. ≥0. u+v=0.75≤1. ✓
    //   All conditions: hit=true, t=0.5, u=0.25, v=0.5.
    [Test]
    public void MT_KnownHit_tHalf_uv_InRange()
    {
        var (cs, k, results) = SetupMTTest();
        try
        {
            SetRay(cs, new Vector3(0, 0, 0), new Vector3(0, 0, 1));
            SetTri(cs, new Vector3(-1,-1, 0.5f), new Vector3(1,-1, 0.5f), new Vector3(0, 1, 0.5f));
            cs.Dispatch(k, 1, 1, 1);

            var data = new float[4];
            results.GetData(data);
            float t_ray  = data[0];
            float u_bary = data[1];
            float v_bary = data[2];
            float hit    = data[3];

            Assert.AreEqual(1f,   hit,    1e-5f, "Case 1: must be a hit (hitFlag=1)");
            Assert.AreEqual(0.5f, t_ray,  1e-4f, "Case 1: t_ray must be ≈0.5 (mid-edge)");
            Assert.GreaterOrEqual(u_bary, 0f,    "Case 1: u_bary ≥ 0");
            Assert.GreaterOrEqual(v_bary, 0f,    "Case 1: v_bary ≥ 0");
            Assert.LessOrEqual(u_bary + v_bary, 1f + 1e-4f, "Case 1: u+v ≤ 1");
        }
        finally { results.Dispose(); }
    }

    // ── Test 2: Parallel ray (det ≈ 0) → miss ────────────────────────────────────────────
    // Ray in the z=0.5 plane: O=(0,0,0.5), Dvec=(1,0,0).
    // Triangle in the same z=0.5 plane: V0=(-1,-1,0.5), V1=(1,-1,0.5), V2=(0,1,0.5).
    // det = dot(cross(L2,Dvec),L1). L2=(1,2,0), cross(L2,(1,0,0))=(0*0-0*0, 0*1-1*0, 1*0-2*1)=(0,0,-2).
    // L1=(2,0,0). det=dot((0,0,-2),(2,0,0))=0 → parallel → miss.
    [Test]
    public void MT_ParallelRay_Miss()
    {
        var (cs, k, results) = SetupMTTest();
        try
        {
            SetRay(cs, new Vector3(0, 0, 0.5f), new Vector3(1, 0, 0));
            SetTri(cs, new Vector3(-1,-1, 0.5f), new Vector3(1,-1, 0.5f), new Vector3(0, 1, 0.5f));
            cs.Dispatch(k, 1, 1, 1);

            var data = new float[4];
            results.GetData(data);
            Assert.AreEqual(0f, data[3], 1e-5f, "Case 2: parallel ray must be a miss (hitFlag=0)");
        }
        finally { results.Dispose(); }
    }

    // ── Test 3: Infinite ray hits but t_ray > 1 → finite-edge miss (gap C24) ─────────────
    // Ray: O=(0,0,0), Dvec=(0,0,0.3) (edge of length 0.3, does NOT reach z=0.5).
    // Triangle at z=0.5: same as case 1.
    // Infinite parametric t = 0.5 / 0.3 ≈ 1.667 > 1 → finite-edge bound: no hit.
    [Test]
    public void MT_tGreaterThanOne_FiniteEdgeMiss()
    {
        var (cs, k, results) = SetupMTTest();
        try
        {
            SetRay(cs, new Vector3(0, 0, 0), new Vector3(0, 0, 0.3f));
            SetTri(cs, new Vector3(-1,-1, 0.5f), new Vector3(1,-1, 0.5f), new Vector3(0, 1, 0.5f));
            cs.Dispatch(k, 1, 1, 1);

            var data = new float[4];
            results.GetData(data);
            float t_ray = data[0];
            float hit   = data[3];

            Assert.AreEqual(0f, hit, 1e-5f,
                $"Case 3: t_ray={t_ray:F3}>1 so finite-edge must miss (hitFlag=0) [gap C24]");
        }
        finally { results.Dispose(); }
    }
}
