// MshLoader_Tests.cs — liver-import pipeline, design §Testing #1/#2.
// CPU tests (no GPU). Reads the real liver asset from StreamingAssets (editor path).

using System.IO;
using NUnit.Framework;
using Unity.Mathematics;
using UnityEngine;
using ReconGridDC.Preprocess;

public class MshLoader_Tests
{
    static string LiverPath => Path.Combine(Application.streamingAssetsPath, "liver3_refined_1.msh");

    [Test]
    public void LoadMsh_Liver_ParsesCountsAndBounds()
    {
        Assert.IsTrue(File.Exists(LiverPath), $"liver asset missing: {LiverPath}");
        MshMesh m = MshLoader.LoadMsh(LiverPath);

        Assert.AreEqual(13412, m.nodes.Length, "node count");
        Assert.AreEqual(58776, m.tets.Length, "tet count");

        MshLoader.Bounds(m, out float3 min, out float3 max);
        // ±1e-2: ~16-sig-digit coords parsed into float32 → float rounding can exceed 1e-3 on large coords.
        Assert.AreEqual(-11.005f, min.x, 1e-2f); Assert.AreEqual(-3.350f, min.y, 1e-2f); Assert.AreEqual(-3.294f, min.z, 1e-2f);
        Assert.AreEqual(  6.818f, max.x, 1e-2f); Assert.AreEqual( 6.554f, max.y, 1e-2f); Assert.AreEqual( 6.758f, max.z, 1e-2f);
    }

    [Test]
    public void ExtractBoundary_IsWatertight_AndOutwardWound()
    {
        MshMesh m = MshLoader.LoadMsh(LiverPath);
        MshSurface s = MshLoader.ExtractBoundary(m);

        Assert.Greater(s.tris.Length, 0, "boundary must have triangles");
        Assert.Less(s.tris.Length, 4 * m.tets.Length, "boundary ≪ 4*tetCount");

        Assert.IsTrue(MshLoader.IsWatertight(s), "boundary must be a closed manifold (every edge shared by 2 faces)");

        // INDEPENDENT orientation check (NOT the pseudonormal sign method under test):
        // signed volume V = (1/6) Σ dot(v0, cross(v1,v2)) over outward-wound tris == +enclosed volume.
        // A consistently OUTWARD winding ⇒ V > 0 (and large); inward/inconsistent ⇒ V ≤ 0 / cancels.
        double v6 = 0.0;
        for (int i = 0; i < s.tris.Length; i++)
        {
            int3 t = s.tris[i];
            float3 a = s.verts[t.x], b = s.verts[t.y], c = s.verts[t.z];
            v6 += math.dot(a, math.cross(b, c));
        }
        double vol = v6 / 6.0;
        Assert.Greater(vol, 1.0, "signed volume > 0 (and non-trivial) ⇒ consistent OUTWARD winding");
    }

    [Test]
    public void LoadMsh_RemapsNonContiguousTags()
    {
        // gmsh 4.1 with NON-contiguous node tags (10,20,30,40) → the loader must remap tag→dense-index
        // (a tag==index+1 shortcut would corrupt the tet connectivity). One tet referencing all 4.
        string msh =
            "$MeshFormat\n4.1 0 8\n$EndMeshFormat\n" +
            "$Nodes\n1 4 10 40\n3 1 0 4\n10\n20\n30\n40\n0 0 0\n1 0 0\n0 1 0\n0 0 1\n$EndNodes\n" +
            "$Elements\n1 1 100 100\n3 1 4 1\n100 10 20 30 40\n$EndElements\n";
        string tmp = Path.Combine(Application.temporaryCachePath, "mshloader_gaptags.msh");
        File.WriteAllText(tmp, msh);
        try
        {
            MshMesh m = MshLoader.LoadMsh(tmp);
            Assert.AreEqual(4, m.nodes.Length);
            Assert.AreEqual(1, m.tets.Length);
            // tags 10,20,30,40 → dense indices 0,1,2,3 (positional within the block).
            var t = m.tets[0];
            Assert.AreEqual(new int4(0, 1, 2, 3), t);
            Assert.AreEqual(new float3(0, 0, 0), m.nodes[0]);  // tag 10
            Assert.AreEqual(new float3(0, 0, 1), m.nodes[3]);  // tag 40
        }
        finally { File.Delete(tmp); }
    }
}
