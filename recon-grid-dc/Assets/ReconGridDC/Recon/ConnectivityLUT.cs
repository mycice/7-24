// ConnectivityLUT.cs — Sub-stage 2A (design §5.4; paper §2.1.1)
// Pure-CPU build of the CONN4096 connectivity look-up table.
// For every 12-bit cut-mask config (2^12 = 4096 entries), runs union-find over
// the UNCUT-edge subgraph of the 8-vertex/12-edge cube:
//   - A 0-bit edge is a live connectivity link (vertices connected).
//   - A 1-bit (cut) edge REMOVES the link from the subgraph.  // paper §2.1.1
// Outputs compCount (# connected components) and vertToComp (dense 0..compCount-1
// component id per vertex), for use by LookupConnectivity in Stage 2C.
//
// Why vertex-keyed (not edge-keyed): a cut edge straddles two components; no single
// per-edge id names "the" component. Centroid + stitching are per vertex.
// (findings.md §2.1.1; gap C4/C54)
//
// PAPER-SILENT: dense renumbering 0..compCount-1 (gap C4; design §5.4 dense-id invariant).

using ReconGridDC.Core;

namespace ReconGridDC.Recon
{
    public static class ConnectivityLUT
    {
        // paper §2.1.1: 12-bit index → 2^12 = 4096 connectivity configurations.
        public const int CONFIG_COUNT = 4096;

        /// <summary>
        /// Build the CONN4096 table.
        /// <para>
        /// compCount[config]        = number of connected components for that cut-mask (1..8).
        /// vertToComp[config*8 + v] = dense component id (0..compCount[config]-1) for vertex v.
        /// </para>
        /// <para>Edge endpoints are read from GridConventions.EdgeCorners (Fig 2.4 numbering).</para>
        /// </summary>
        /// <param name="compCount">Output int[4096]: component count per config.</param>
        /// <param name="vertToComp">Output int[4096*8]: component id per (config, vertex).</param>
        public static void Build(out int[] compCount, out int[] vertToComp)
        {
            compCount  = new int[CONFIG_COUNT];       // compCount[config]
            vertToComp = new int[CONFIG_COUNT * 8];   // vertToComp[config*8 + v]

            // Reusable union-find parent array (8 vertices).
            int[] parent = new int[8];

            for (int config = 0; config < CONFIG_COUNT; config++)
            {
                // ── 1. Initialise union-find: each vertex is its own root ─────────────────
                for (int v = 0; v < 8; v++) parent[v] = v;

                // ── 2. Unite the two endpoints of each UNCUT edge ─────────────────────────
                // paper §2.1.1: cut bit DISCONNECTS the edge in the connectivity subgraph.
                // A 0-bit means the edge is LIVE → unite its endpoints.
                // A 1-bit means the edge is CUT → do NOT unite (link removed).
                for (int e = 0; e < 12; e++)
                {
                    if ((config >> e & 1) == 0) // bit e == 0 → uncut edge → live link
                    {
                        int a = GridConventions.EdgeCorners[e, 0]; // Fig 2.4 edge→corner table
                        int b = GridConventions.EdgeCorners[e, 1];
                        Union(parent, a, b);
                    }
                }

                // ── 3. Dense renumber: assign ids 0..compCount-1 in first-appearance order ─
                // PAPER-SILENT: dense ids (gap C4; design §5.4 dense-id invariant).
                int[] denseId = new int[8]; // maps raw root → dense id (-1 = unseen)
                for (int i = 0; i < 8; i++) denseId[i] = -1;

                int nextId = 0;
                int baseOffset = config * 8;

                for (int v = 0; v < 8; v++)
                {
                    int root = Find(parent, v); // canonical root of v's component
                    if (denseId[root] < 0)
                        denseId[root] = nextId++; // first appearance of this root → new id
                    vertToComp[baseOffset + v] = denseId[root];
                }

                compCount[config] = nextId; // total distinct components for this config
            }
        }

        // ── Union-Find helpers (path-compressed Find; union-by-rank omitted for simplicity;
        //    N=8 so depth is trivially bounded) ──────────────────────────────────────────────

        static int Find(int[] parent, int x)
        {
            while (parent[x] != x)
            {
                parent[x] = parent[parent[x]]; // path halving
                x = parent[x];
            }
            return x;
        }

        static void Union(int[] parent, int a, int b)
        {
            a = Find(parent, a);
            b = Find(parent, b);
            if (a != b) parent[b] = a; // merge b's tree under a
        }
    }
}
