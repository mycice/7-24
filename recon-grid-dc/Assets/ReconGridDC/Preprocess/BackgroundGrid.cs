using System.Collections.Generic;
using Unity.Mathematics;
using ReconGridDC.Core;

namespace ReconGridDC.Preprocess
{
    // B1 (plan §7): insideA tags WHICH endpoint of the edge is the INSIDE (phi<0) one — 1 ⇒ cornerA
    // is inside, 0 ⇒ cornerB is inside. Lets ComputeComponentFP assign each isect plane to the
    // connectivity component of its INSIDE endpoint (so the two halves use DIFFERENT planes → split).
    // paper §2.1.1 (Fig 2.4 connectivity); design §3.3.
    public struct IsectEntry { public float3 globalRest; public float3 normal; public int cornerA; public int cornerB; public float t; public int insideA; }

    /// <summary>
    /// A global grid edge that crosses the isosurface and has all 4 incident voxels in-range
    /// (interior edge). Used by DC_Stitch to produce the dual quad for each such edge.
    /// Task 7 Step 1.
    /// </summary>
    public struct GridEdge
    {
        public int  axis;        // 0=x, 1=y, 2=z
        public int3 baseCorner;  // integer grid corner coord of the "A" endpoint
        public int  insideA;     // 1 if baseCorner is inside (phi<0), 0 if outside
    }

    // ── Task 1 (Stage 1B-i): physics topology structs ───────────────────────────────────────
    /// <summary>One structural spring per axis-aligned grid edge (deduplicated: only +x/+y/+z edges).</summary>
    public struct Spring   { public int i, j; public float L0; }
    /// <summary>One bending pair out of C(6,2)=15 per corner. alive=0 when an arm neighbor is missing.</summary>
    public struct BendPair { public int i, j, k; public float theta0; public int alive; }

    // design §5.2: all interior grid edges used as M-T rays in cut detection (Stage 2B).
    // Unlike GridEdge (surface edges), no sign-change filter is applied.
    public struct GridEdgeRec
    {
        public int  axis;        // 0=x, 1=y, 2=z
        public int3 baseCorner;  // integer grid corner coord of the "A" endpoint
    }

    public sealed class BackgroundGrid
    {
        public int3 dims; public float L; public float3 origin;
        public float3[] cornerPos; public byte[] cornerInside;

        /// <summary>
        /// Stage 2D-1b (paper §2.1.4, paper:485-499): per-corner ACTIVE flag. cornerActive[c]=1 iff
        /// corner c is incident to ≥1 OCCUPIED voxel (= inside corners + the one-ring boundary shell);
        /// 0 ⇒ fully-outside (empty-space) corner. Paper §2.1.4: "particles outside the isosurface are
        /// directly deleted and no longer participate in force calculations or surface generation." So
        /// inactive corners are FROZEN in-kernel (treated exactly like pinned) and emit NO structural
        /// springs / bending pairs across an active↔inactive edge — this removes the empty-space spring/
        /// bend "cage" that otherwise hangs a severed tissue piece from the pinned empty lattice. plan §2.1.
        /// </summary>
        public byte[] cornerActive;

        public int[] voxelCorner; public int[] voxelIsectOffset; public int[] voxelIsectCount;
        public IsectEntry[] isect; public int voxelCount;

        /// <summary>
        /// RC2 occupancy gate (plan §2.1 / §7): voxelOccupied[v] = OR of its 8 corners' cornerInside.
        /// 1 ⇒ the voxel touches the tissue (≥1 corner inside the isosurface), 0 ⇒ fully-outside empty
        /// space. The cut kernels gate on this so no cut geometry is generated in non-tissue voxels
        /// (the spurious "flap"). Boundary voxels (IsectCount&gt;0) AND interior all-inside voxels both
        /// count as occupied; only all-outside voxels are excluded (audit R4). design §2.1.
        /// </summary>
        public byte[] voxelOccupied;

        /// <summary>
        /// Interior intersecting grid edges: each crosses the isosurface AND all 4 incident
        /// voxels exist (so the DC dual quad is complete). Built in Task 7 Step 1.
        /// </summary>
        public GridEdge[] surfaceEdges;

        /// <summary>
        /// ALL interior grid edges (no sign-change filter): every grid edge whose 4 incident
        /// voxels are all in-range. Used by Stage 2B DetectCut as M-T ray sources.
        /// design §5.2: cut detection casts each VOXEL grid edge as the M-T ray.
        /// </summary>
        public GridEdgeRec[] gridEdges;

        // ── Task 1 (Stage 1B-i): physics fields ─────────────────────────────────────────────
        /// <summary>Per-corner uniform mass (default 1.0). PAPER-SILENT: paper gives no mass value.</summary>
        public float[]    mass;
        /// <summary>6 per corner: [-x,+x,-y,+y,-z,+z] neighbor cornerId or -1.</summary>
        public int[]      nbrIdx;
        /// <summary>6 per corner: rest offset (nbrPos - cornerPos), zero float3 if no neighbor.</summary>
        public float3[]   restNbr;
        /// <summary>Per corner: 1 = fixed/pinned (default all 0).</summary>
        public byte[]     pinned;
        /// <summary>Structural springs, one per axis-aligned edge (+x/+y/+z deduplicated).</summary>
        public Spring[]   springs;
        /// <summary>Bending pairs: 15 per corner (flat array). Index: 15*cornerIdx + pairSlot.</summary>
        public BendPair[] bendPairs;
        /// <summary>Total number of corners = (dims.x+1)*(dims.y+1)*(dims.z+1).</summary>
        public int cornerCount;

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
            g.voxelOccupied    = new byte[g.voxelCount]; // RC2 gate (plan §2.1 / §7)
            var flat = new List<IsectEntry>();

            for (int vk=0; vk<dims.z; vk++) for (int vj=0; vj<dims.y; vj++) for (int vi=0; vi<dims.x; vi++)
            {
                int v = GridConventions.VoxelId(vi,vj,vk,dims);
                // 8 corner ids
                for (int c=0;c<8;c++){
                    int3 o = GridConventions.CornerOffset[c];
                    g.voxelCorner[8*v+c] = GridConventions.CornerId(vi+o.x, vj+o.y, vk+o.z, dims);
                }
                // RC2 occupancy (plan §7): voxelOccupied[v] = OR of the 8 corners' cornerInside.
                // ≥1 corner inside ⇒ this voxel touches tissue. Uses voxelCorner[8*v+0..7] (audit R4).
                byte occ = 0;
                for (int c=0;c<8;c++)
                    if (g.cornerInside[g.voxelCorner[8*v+c]] == 1) { occ = 1; break; }
                g.voxelOccupied[v] = occ;
                g.voxelIsectOffset[v] = flat.Count;
                int cnt=0;
                for (int e=0;e<12;e++)
                {
                    int ca = g.voxelCorner[8*v + GridConventions.EdgeCorners[e,0]];
                    int cb = g.voxelCorner[8*v + GridConventions.EdgeCorners[e,1]];
                    bool ia = g.cornerInside[ca]==1, ib = g.cornerInside[cb]==1;
                    if (ia == ib) continue;                       // no sign change => no crossing — direct corner sign-change test is the implemented, EQUIVALENT form of the design's MC256 criterion (no separate mc256 table is materialized in Stage 1)
                    float fa = phi[ca], fb = phi[cb];
                    float t = fa/(fa-fb);                          // zero-crossing param along A->B, in (0,1)
                    float3 pos = math.lerp(g.cornerPos[ca], g.cornerPos[cb], t);
                    float3 n = ls.Gradient(pos);
                    // B1 (plan §7): record which endpoint is INSIDE (phi<0). ia already computed above
                    // (cornerInside[ca]==1). Marshalled into IsectGpu.insideA in ReconBuffers.Upload.
                    flat.Add(new IsectEntry{ globalRest=pos, normal=n, cornerA=ca, cornerB=cb, t=t, insideA=(ia?1:0) });
                    cnt++;
                }
                g.voxelIsectCount[v] = cnt;
            }
            g.isect = flat.ToArray();

            // ── Task 7 Step 1: Build interior surface-edge list ──────────────────────────────
            // Iterate over every directed grid edge by axis. An edge qualifies as a "surface edge"
            // iff:  (a) both corner endpoints exist in the grid
            //       (b) the two endpoints have different inside/outside classification (sign change)
            //       (c) all 4 incident voxels are in-bounds (interior edge → complete DC quad)
            var edges = new List<GridEdge>();
            int3[] edgeDirs = { new int3(1,0,0), new int3(0,1,0), new int3(0,0,1) };
            for (int axis = 0; axis < 3; axis++)
            {
                int3 d = edgeDirs[axis];
                for (int k = 0; k < cnz; k++)
                for (int j = 0; j < cny; j++)
                for (int i = 0; i < cnx; i++)
                {
                    int3 a = new int3(i, j, k);
                    int3 b = a + d;
                    // Endpoint b must be within the corner grid
                    if (b.x >= cnx || b.y >= cny || b.z >= cnz) continue;

                    int ca = GridConventions.CornerId(a.x, a.y, a.z, dims);
                    int cb = GridConventions.CornerId(b.x, b.y, b.z, dims);

                    // Condition (b): sign change
                    if ((g.cornerInside[ca] == 1) == (g.cornerInside[cb] == 1)) continue;

                    // Condition (c): all 4 incident voxels in-bounds
                    bool allIn = true;
                    foreach (var se in GridConventions.EdgeStencil[axis])
                    {
                        int3 vc = a + se.voxelOffset;
                        if (vc.x < 0 || vc.y < 0 || vc.z < 0 ||
                            vc.x >= dims.x || vc.y >= dims.y || vc.z >= dims.z)
                        {
                            allIn = false;
                            break;
                        }
                    }
                    if (!allIn) continue;

                    edges.Add(new GridEdge
                    {
                        axis      = axis,
                        baseCorner = a,
                        insideA   = g.cornerInside[ca]  // 1 if a is inside, 0 if outside
                    });
                }
            }
            g.surfaceEdges = edges.ToArray();

            // ── Stage 2B Step 1: Build ALL interior grid edges (no sign-change filter) ────────
            // Mirror of surfaceEdges builder above but condition (b) [sign-change] is DROPPED.
            // Every grid edge (3 axes) whose 4 incident voxels are all in-range qualifies.
            // design §5.2: cut detection casts each VOXEL grid edge as the M-T ray.
            var gridEdgeList = new List<GridEdgeRec>();
            for (int axis = 0; axis < 3; axis++)
            {
                int3 d = edgeDirs[axis];
                for (int k = 0; k < cnz; k++)
                for (int j = 0; j < cny; j++)
                for (int i = 0; i < cnx; i++)
                {
                    int3 a = new int3(i, j, k);
                    int3 b = a + d;
                    // Endpoint b must be within the corner grid
                    if (b.x >= cnx || b.y >= cny || b.z >= cnz) continue;

                    // Condition: all 4 incident voxels in-bounds (interior edge → complete ring)
                    bool allIn = true;
                    foreach (var se in GridConventions.EdgeStencil[axis])
                    {
                        int3 vc = a + se.voxelOffset;
                        if (vc.x < 0 || vc.y < 0 || vc.z < 0 ||
                            vc.x >= dims.x || vc.y >= dims.y || vc.z >= dims.z)
                        {
                            allIn = false;
                            break;
                        }
                    }
                    if (!allIn) continue;

                    // No sign-change filter — all interior edges qualify as M-T ray candidates
                    gridEdgeList.Add(new GridEdgeRec { axis = axis, baseCorner = a });
                }
            }
            g.gridEdges = gridEdgeList.ToArray();

            // ── Task 1 (Stage 1B-i): build physics topology ──────────────────────────────────
            g.BuildPhysics();

            return g;
        }

        // ── Task 1 (Stage 1B-i): neighbor slot -> axis delta (index 0=-x,1=+x,2=-y,3=+y,4=-z,5=+z) ──
        static readonly int3[] NbrDelta = {
            new int3(-1, 0, 0), new int3(1, 0, 0),
            new int3( 0,-1, 0), new int3(0, 1, 0),
            new int3( 0, 0,-1), new int3(0, 0, 1),
        };

        // 15 unordered pairs of the 6 neighbor slots: all C(6,2)=15 combinations in the order
        // (0,1),(0,2),(0,3),(0,4),(0,5),(1,2),(1,3),(1,4),(1,5),(2,3),(2,4),(2,5),(3,4),(3,5),(4,5)
        static readonly int[,] PairSlots = {
            {0,1},{0,2},{0,3},{0,4},{0,5},
            {1,2},{1,3},{1,4},{1,5},
            {2,3},{2,4},{2,5},
            {3,4},{3,5},
            {4,5}
        };

        /// <summary>
        /// Builds per-corner neighbor lists, structural springs (one per +x/+y/+z edge),
        /// bending pairs C(6,2)=15 per corner, uniform mass, and zero pin flags.
        /// Called automatically at the end of Build().
        /// PAPER-SILENT: mass=1.0 (paper unspecified); pinned=0 (default; demo sets some).
        /// </summary>
        void BuildPhysics()
        {
            int cnx = dims.x + 1, cny = dims.y + 1, cnz = dims.z + 1;
            cornerCount = cnx * cny * cnz;

            // ── Stage 2D-1b: compute cornerActive (paper §2.1.4, paper:485-499) ──────────────────
            // active[c]=1 iff corner c is incident to ≥1 OCCUPIED voxel. Start all-0, then for every
            // voxel v with voxelOccupied[v]!=0 mark its 8 corners active. Reuses the already-built
            // voxelCorner (8/voxel) + voxelOccupied arrays. Result = inside corners + the one-ring of
            // boundary-shell corners (the surface stays load-bearing); fully-outside corners stay 0.
            // Inactive corners are frozen in-kernel + carry no springs/bend-pairs (gated below).
            cornerActive = new byte[cornerCount];
            for (int v = 0; v < voxelCount; v++)
            {
                if (voxelOccupied[v] == 0) continue;
                for (int c = 0; c < 8; c++)
                    cornerActive[voxelCorner[8 * v + c]] = 1;
            }

            mass    = new float[cornerCount];
            pinned  = new byte[cornerCount];
            nbrIdx  = new int[6 * cornerCount];
            restNbr = new float3[6 * cornerCount];

            var springList = new List<Spring>();

            for (int k = 0; k < cnz; k++)
            for (int j = 0; j < cny; j++)
            for (int i = 0; i < cnx; i++)
            {
                int c = GridConventions.CornerId(i, j, k, dims);
                mass[c]   = 1.0f;  // PAPER-SILENT: uniform mass
                pinned[c] = 0;

                for (int s = 0; s < 6; s++)
                {
                    int3 d  = NbrDelta[s];
                    int  ni = i + d.x, nj = j + d.y, nk = k + d.z;
                    if (ni < 0 || nj < 0 || nk < 0 || ni >= cnx || nj >= cny || nk >= cnz)
                    {
                        nbrIdx[6 * c + s]  = -1;
                        restNbr[6 * c + s] = float3.zero;
                        continue;
                    }
                    int n = GridConventions.CornerId(ni, nj, nk, dims);
                    nbrIdx[6 * c + s]  = n;
                    restNbr[6 * c + s] = cornerPos[n] - cornerPos[c];

                    // Structural spring once per edge: only emit for +x(slot 1), +y(slot 3), +z(slot 5)
                    // to avoid counting each edge twice (each edge has two endpoints).
                    // Stage 2D-1b (paper §2.1.4, paper:485-499): SKIP the spring when EITHER endpoint
                    // corner is inactive — an active corner must NOT be anchored to the frozen empty
                    // lattice (a frozen inactive corner at rest would otherwise pin the active corner
                    // via this spring, recreating the empty-space cage).
                    if ((s == 1 || s == 3 || s == 5) &&
                        cornerActive[c] != 0 && cornerActive[n] != 0)
                    {
                        springList.Add(new Spring
                        {
                            i  = c,
                            j  = n,
                            L0 = math.length(restNbr[6 * c + s])
                        });
                    }
                }
            }
            springs = springList.ToArray();

            // Build bending pairs: C(6,2)=15 per corner (flat, index 15*c + p)
            bendPairs = new BendPair[15 * cornerCount];
            for (int c = 0; c < cornerCount; c++)
            {
                for (int p = 0; p < 15; p++)
                {
                    int sa = PairSlots[p, 0], sb = PairSlots[p, 1];
                    int jj = nbrIdx[6 * c + sa];
                    int kk = nbrIdx[6 * c + sb];

                    var bp = new BendPair { i = c, j = jj, k = kk, theta0 = 0f, alive = 0 };
                    // Stage 2D-1b (paper §2.1.4, paper:485-499): a bending pair is alive only when both
                    // arms exist AND the center + BOTH arm corners are active. An inactive (frozen,
                    // empty-space) corner participating in a bend triple would couple an active corner
                    // to the frozen lattice → drop those pairs (same cage-removal as the springs above).
                    if (jj >= 0 && kk >= 0 &&
                        cornerActive[c] != 0 && cornerActive[jj] != 0 && cornerActive[kk] != 0)
                    {
                        float3 Nij  = math.normalize(cornerPos[jj] - cornerPos[c]);
                        float3 Nik  = math.normalize(cornerPos[kk] - cornerPos[c]);
                        float  cdot = math.clamp(math.dot(Nij, Nik), -1f, 1f);
                        bp.theta0 = math.acos(cdot);  // collinear opposite slots (e.g. -x,+x) -> pi
                        bp.alive  = 1;
                    }
                    bendPairs[15 * c + p] = bp;
                }
            }
        }
    }
}
