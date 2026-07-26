using Unity.Mathematics;

namespace ReconGridDC.Core
{
    public static class GridConventions
    {
        // Fig 2.4 local corner offsets (bottom z=0: V0..V3 CCW in xy; top z=1: V4..V7 above V0..V3)
        public static readonly int3[] CornerOffset =
        {
            new int3(0,0,0), new int3(1,0,0), new int3(1,1,0), new int3(0,1,0), // V0..V3
            new int3(0,0,1), new int3(1,0,1), new int3(1,1,1), new int3(0,1,1), // V4..V7
        };

        // 12 edges -> their two local corner ids (Fig 2.4)
        public static readonly int[,] EdgeCorners =
        {
            {0,1},{1,2},{2,3},{3,0},   // e0..e3 bottom loop
            {4,5},{5,6},{6,7},{7,4},   // e4..e7 top loop
            {0,4},{1,5},{2,6},{3,7},   // e8..e11 verticals
        };

        public struct EdgeStencilEntry { public int3 voxelOffset; public int localEdge; }

        // For a grid edge of a given axis at base corner C, the 4 incident voxels (offset from C's
        // voxel) and that edge's local id inside each. Entries are in CCW order viewed along +axis.
        public static readonly EdgeStencilEntry[][] EdgeStencil = new EdgeStencilEntry[][]
        {
            // axis 0 = x : edge C -> C+(1,0,0)
            new []{
                E(0,0,0, 0), E(0,-1,0, 2), E(0,-1,-1, 6), E(0,0,-1, 4),
            },
            // axis 1 = y : edge C -> C+(0,1,0)
            new []{
                E(0,0,0, 3), E(0,0,-1, 7), E(-1,0,-1, 5), E(-1,0,0, 1),
            },
            // axis 2 = z : edge C -> C+(0,0,1)
            new []{
                E(0,0,0, 8), E(-1,0,0, 9), E(-1,-1,0, 10), E(0,-1,0, 11),
            },
        };
        static EdgeStencilEntry E(int x,int y,int z,int le) =>
            new EdgeStencilEntry{ voxelOffset = new int3(x,y,z), localEdge = le };

        // ── Stage 2C (plan §3.1): LocalCornerOf — the ONLY GridConventions extend for 2C ──────
        // Which local corner (0..7, Fig 2.4) of voxel `voxelCoord` coincides with global corner
        // `cornerCoord`; -1 if `cornerCoord` is not a corner of that voxel. This is the vertex key
        // feeding vertToComp (CONN4096 LUT) — it MUST use the SAME Fig 2.4 numbering the LUT was
        // built on (design §10; locked by CutSurface oracle test 1). paper §2.1.1 / Fig 2.4.
        public static int LocalCornerOf(int3 cornerCoord, int3 voxelCoord)
        {
            int3 d = cornerCoord - voxelCoord;
            for (int lc = 0; lc < 8; lc++)
                if (CornerOffset[lc].x == d.x && CornerOffset[lc].y == d.y && CornerOffset[lc].z == d.z)
                    return lc;
            return -1;
        }

        public static int CornerId(int i,int j,int k,int3 dims) =>
            i + (dims.x+1)*(j + (dims.y+1)*k);
        public static int VoxelId(int i,int j,int k,int3 dims) =>
            i + dims.x*(j + dims.y*k);
        public static int3 VoxelCoord(int voxelId,int3 dims) =>
            new int3(voxelId % dims.x, (voxelId/dims.x) % dims.y, voxelId/(dims.x*dims.y));

        // Eq5 box SDF, negative inside, zero on face. L = full edge length (half = L/2).
        public static float BoxSdf(float3 p, float3 center, float L)
        {
            float h = L*0.5f;
            float3 d = math.abs(p-center) - h;
            return math.max(math.max(d.x, d.y), d.z);
        }
    }
}
