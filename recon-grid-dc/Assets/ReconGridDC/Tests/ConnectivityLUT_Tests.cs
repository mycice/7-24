// ConnectivityLUT_Tests.cs — Sub-stage 2A (design §5.4; paper §2.1.1)
// Pure-CPU EditMode tests. No GPU, no Category("GPU"). Runs anywhere.
// Verifies: Fig 2.4 example, all-cut singleton, no-cut single-component,
//           dense-id invariant for all 4096 configs, and an independent
//           reference union-find matching the whole table.

using NUnit.Framework;
using ReconGridDC.Core;
using ReconGridDC.Recon;

public class ConnectivityLUT_Tests
{
    // ── Lazy-build the table once for all tests in this fixture ───────────────────────
    static int[]  _compCount;
    static int[]  _vertToComp;

    static void EnsureBuilt()
    {
        if (_compCount == null)
            ConnectivityLUT.Build(out _compCount, out _vertToComp);
    }

    // ─────────────────────────────────────────────────────────────────────────────────
    // Test 1: Paper Fig 2.4 example
    // Cut e0, e2, e4, e6 → two components {V0,V3,V4,V7} and {V1,V2,V5,V6}.
    // paper Fig 2.4; design §5.4; findings.md §2.1.1.
    // ─────────────────────────────────────────────────────────────────────────────────
    [Test]
    public void Fig24Example_TwoComponents()
    {
        EnsureBuilt();

        // config = cut bits set for e0, e2, e4, e6
        // e0=(V0,V1), e2=(V2,V3), e4=(V4,V5), e6=(V6,V7)
        // paper Fig 2.4: these 4 cuts split the cube into {V0,V3,V4,V7} vs {V1,V2,V5,V6}
        int config = (1 << 0) | (1 << 2) | (1 << 4) | (1 << 6);
        int b = config * 8;

        Assert.AreEqual(2, _compCount[config],
            "Fig 2.4: cutting e0,e2,e4,e6 must yield exactly 2 components. // paper Fig 2.4");

        // Group A = {V0,V3,V4,V7}: all must share the same dense component id.
        int idA = _vertToComp[b + 0]; // V0's id is the reference for group A
        Assert.AreEqual(idA, _vertToComp[b + 3], "V3 must be in same component as V0 (group A)");
        Assert.AreEqual(idA, _vertToComp[b + 4], "V4 must be in same component as V0 (group A)");
        Assert.AreEqual(idA, _vertToComp[b + 7], "V7 must be in same component as V0 (group A)");

        // Group B = {V1,V2,V5,V6}: all must share a different dense component id.
        int idB = _vertToComp[b + 1]; // V1's id is the reference for group B
        Assert.AreNotEqual(idA, idB, "Group A and Group B must have distinct component ids");
        Assert.AreEqual(idB, _vertToComp[b + 2], "V2 must be in same component as V1 (group B)");
        Assert.AreEqual(idB, _vertToComp[b + 5], "V5 must be in same component as V1 (group B)");
        Assert.AreEqual(idB, _vertToComp[b + 6], "V6 must be in same component as V1 (group B)");

        // Dense ids must be 0 and 1 (exactly 2 distinct ids for compCount==2).
        Assert.IsTrue(idA == 0 || idA == 1, "Dense id for group A must be 0 or 1");
        Assert.IsTrue(idB == 0 || idB == 1, "Dense id for group B must be 0 or 1");
    }

    // ─────────────────────────────────────────────────────────────────────────────────
    // Test 2: All 12 edges cut → 8 singletons (each vertex is its own component).
    // paper §2.1.1 "up to 8 feature points"; design §5.4.
    // ─────────────────────────────────────────────────────────────────────────────────
    [Test]
    public void AllEdgesCut_EightSingletons()
    {
        EnsureBuilt();

        int config = 0xFFF; // all 12 bits set
        int b = config * 8;

        Assert.AreEqual(8, _compCount[config],
            "All edges cut: each vertex must be its own component. // paper §2.1.1");

        // All 8 vertToComp values must be distinct (dense 0..7).
        var seen = new bool[8];
        for (int v = 0; v < 8; v++)
        {
            int id = _vertToComp[b + v];
            Assert.IsFalse(seen[id], $"vertToComp[{v}]={id} is not unique (all-cut must yield 8 distinct ids)");
            seen[id] = true;
        }
    }

    // ─────────────────────────────────────────────────────────────────────────────────
    // Test 3: No edges cut → single connected component, all ids == 0.
    // design §5.4; paper §2.1.1 (fully connected subgraph → 1 component).
    // ─────────────────────────────────────────────────────────────────────────────────
    [Test]
    public void NoCut_OneComponent()
    {
        EnsureBuilt();

        int config = 0;
        int b = config * 8;

        Assert.AreEqual(1, _compCount[config],
            "No cuts: all vertices connected → 1 component. // design §5.4");

        for (int v = 0; v < 8; v++)
            Assert.AreEqual(0, _vertToComp[b + v],
                $"No-cut config: vertex {v} must have component id 0");
    }

    // ─────────────────────────────────────────────────────────────────────────────────
    // Test 4: Dense-id invariant over all 4096 configs.
    // For every config: ids are a contiguous set {0..compCount-1},
    // max id == compCount-1, and the number of distinct ids == compCount.
    // design §5.4 "dense-id invariant"; PAPER-SILENT (gap C4).
    // ─────────────────────────────────────────────────────────────────────────────────
    [Test]
    public void DenseIds_Invariant_AllConfigs()
    {
        EnsureBuilt();

        for (int config = 0; config < ConnectivityLUT.CONFIG_COUNT; config++)
        {
            int b    = config * 8;
            int nComp = _compCount[config];

            Assert.Greater(nComp, 0, $"config={config}: compCount must be >= 1");
            Assert.LessOrEqual(nComp, 8, $"config={config}: compCount must be <= 8");

            int maxId = -1;
            int distinctCount = 0;

            // Track distinct ids across vertices [0,7].
            bool[] seen = new bool[8];
            for (int v = 0; v < 8; v++)
            {
                int id = _vertToComp[b + v];
                Assert.GreaterOrEqual(id, 0,           $"config={config} v={v}: id must be >= 0");
                Assert.Less(id, nComp,                 $"config={config} v={v}: id must be < compCount={nComp}");
                if (id > maxId) maxId = id;
                if (!seen[id]) { seen[id] = true; distinctCount++; }
            }

            Assert.AreEqual(nComp - 1, maxId,
                $"config={config}: max vertToComp must == compCount-1 (dense ids). // design §5.4");
            Assert.AreEqual(nComp, distinctCount,
                $"config={config}: number of distinct ids must == compCount (dense ids).");
        }
    }

    // ─────────────────────────────────────────────────────────────────────────────────
    // Test 5: Whole-table match against an INDEPENDENT reference union-find.
    // Rewritten inline — completely separate from ConnectivityLUT.Build — to guard
    // against a bug that consistently applies to both implementations.
    // Checks: (a) compCount agrees, (b) induced vertex partition agrees (same equiv classes).
    // design §10 "[2A] CONN4096 test: whole 4096-entry table vs independent reference".
    // ─────────────────────────────────────────────────────────────────────────────────
    [Test]
    public void WholeTable_MatchesReferenceUnionFind()
    {
        EnsureBuilt();

        int[] refParent = new int[8]; // reused per config

        for (int config = 0; config < ConnectivityLUT.CONFIG_COUNT; config++)
        {
            // ── Reference implementation (independent) ────────────────────────────────
            // Initialise
            for (int v = 0; v < 8; v++) refParent[v] = v;

            // Unite on UNCUT (bit==0) edges using GridConventions.EdgeCorners.
            // paper §2.1.1: cut bit DISCONNECTS the edge in the connectivity subgraph.
            for (int e = 0; e < 12; e++)
            {
                if ((config >> e & 1) == 0) // bit==0 → uncut → live link
                {
                    int a = GridConventions.EdgeCorners[e, 0];
                    int b = GridConventions.EdgeCorners[e, 1];
                    RefUnion(refParent, a, b);
                }
            }

            // Densely renumber: first-appearance order (same canonical choice as LUT).
            int[] refDenseId = new int[8];
            for (int i = 0; i < 8; i++) refDenseId[i] = -1;
            int[] refVertToComp = new int[8];
            int refNextId = 0;
            for (int v = 0; v < 8; v++)
            {
                int root = RefFind(refParent, v);
                if (refDenseId[root] < 0) refDenseId[root] = refNextId++;
                refVertToComp[v] = refDenseId[root];
            }
            int refCompCount = refNextId;

            // ── Compare ───────────────────────────────────────────────────────────────
            int tableBase = config * 8;
            Assert.AreEqual(refCompCount, _compCount[config],
                $"config={config}: compCount mismatch vs reference union-find");

            // Partition equivalence: for every pair (u,v), LUT and reference must agree
            // whether they are in the same component (even if dense ids differ in labelling).
            // We compare by building a canonical form: sort vertices by their component id
            // in both, and check induced same-group/different-group relations.
            // Simpler: remap reference ids to match table's first-appearance labelling.
            int[] idMap    = new int[8]; // refId   -> tableId (-1 = unseen)
            int[] idMapRev = new int[8]; // tableId -> refId   (-1 = unseen) : injectivity guard
            for (int i = 0; i < 8; i++) { idMap[i] = -1; idMapRev[i] = -1; }

            for (int v = 0; v < 8; v++)
            {
                int tableId = _vertToComp[tableBase + v];
                int refId   = refVertToComp[v];

                // Forward: a reference component must not be SPLIT across two table ids.
                if (idMap[refId] < 0) idMap[refId] = tableId;
                else Assert.AreEqual(idMap[refId], tableId,
                        $"config={config} v={v}: reference component split across table ids");

                // Reverse (injectivity): two distinct reference components must not be MERGED into one table id.
                if (idMapRev[tableId] < 0) idMapRev[tableId] = refId;
                else Assert.AreEqual(idMapRev[tableId], refId,
                        $"config={config} v={v}: distinct reference components merged in table");
            }
        }
    }

    // ── Minimal independent union-find (reference implementation) ─────────────────────
    // Intentionally separate from the production path (no path halving, no shared state).

    static int RefFind(int[] parent, int x)
    {
        while (parent[x] != x) x = parent[x]; // naive (no compression)
        return x;
    }

    static void RefUnion(int[] parent, int a, int b)
    {
        a = RefFind(parent, a);
        b = RefFind(parent, b);
        if (a != b) parent[b] = a;
    }
}
