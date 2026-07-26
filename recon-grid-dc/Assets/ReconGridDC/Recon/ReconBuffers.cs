using System;
using Unity.Mathematics;
using UnityEngine;
using ReconGridDC.Preprocess;

namespace ReconGridDC.Recon
{
    public sealed class ReconBuffers : IDisposable
    {
        // ── GPU-side Hermite intersection entry (40 bytes) ───────────────────────────────
        // Matches HLSL IsectGpu in ReconCommon.hlsl and the field order in Recon.compute.
        // IsectGpu = 40 bytes (float3+float3+int+int+float+float); plan's "32" was wrong
        public struct IsectGpu
        {
            public float3 globalRest;  // 12 bytes
            public float3 normal;      // 12 bytes
            public int    cornerA;     //  4 bytes
            public int    cornerB;     //  4 bytes
            public float  t;           //  4 bytes
            public int    insideA;     //  4 bytes  -> total 40 bytes (B1: repurposed _pad; stride stays 40)
            //  insideA: 1  - cornerA is the INSIDE (phi<0) endpoint, else cornerB. ComputeComponentFP
            //  assigns this isect plane to the component of its INSIDE endpoint. plan §7 B1; design §3.3.
        }

        // ── GPU-side grid edge entry (Task 7 Step 2) ─────────────────────────────────────
        // Matches HLSL GridEdgeGpu in Recon.compute.
        // C# layout: int axis=4 + int3 baseCorner=12 + int insideA=4 + int _pad=4 = 24 bytes.
        // HLSL int3 is also 12 bytes (3 ints). Stride = 32 (rounded up for safety / alignment).
        public struct GridEdgeGpu
        {
            public int  axis;          //  4 bytes
            public int  bcX, bcY, bcZ; // 12 bytes (int3 as 3 ints for safe C# interop)
            public int  insideA;       //  4 bytes
            public int  _pad;          //  4 bytes -> total 24; stride set to 32 below
            public int  _pad2;         //  4 bytes
            public int  _pad3;         //  4 bytes -> total 32 bytes
        }

        // ── Stage 2B: GPU-side cut grid edge entry ───────────────────────────────────────
        // All interior grid edges (no sign-change filter), used as M-T ray sources.
        // design §5.2 / §3 (GridEdges buffer); 2B plan Task 1 Step 2.
        // Layout: int axis=4 + int3 as 3 ints=12 + padding=0  - total 16 bytes.
        // (HLSL struct GridEdgeGpu2: int axis; int3 baseCorner  - 4+12=16 bytes; no pad needed.)
        public struct GridEdgeGpu2
        {
            public int axis;           //  4 bytes: 0=x,1=y,2=z
            public int bcX, bcY, bcZ;  // 12 bytes: baseCorner.xyz as 3 ints (safe C# interop)
            //  total 16 bytes  - matches HLSL int+int3=16
        }

        // ── Stage 2B: GPU-side cut point entry ───────────────────────────────────────────
        // One cut point per edge endpoint  - two emitted per cut edge.
        // design §3.6 / §5.3; 2B plan Task 1 Step 2.
        // Layout: float3=12 + int=4 + int=4 + float=4 + int+int=8 = 32B.
        // Mirrors HLSL CutPointGpu (float3 localOffset; int ownerParticle; int edgeId;
        //   float _pad; int2 _pad2) = 32 bytes.
        public struct CutPointGpu
        {
            public float  localX, localY, localZ; // 12 bytes: 2D-2 owner MATERIAL-frame offset R_owner^T*(P_side-owner.pos)
            public int    ownerParticle;           //  4 bytes: global corner index of owner endpoint
            public int    edgeId;                  //  4 bytes: GridEdges index (matches GridEdgeGpu2)
            public float  _pad;                    //  4 bytes
            public int    _pad2a, _pad2b;          //  8 bytes: pad to 32 bytes total
            //  total 32 bytes  - matches HLSL CutPointGpu stride 32
        }

        // ── Task 2 (Stage 1B-i): GPU-side physics structs ────────────────────────────────
        // Field order MUST match HLSL declarations in PhysicsCommon.hlsl (Task 3).
        // These are private because callers use the ComputeBuffer fields directly.

        /// <summary>SpringGpu: 16 bytes. Matches HLSL: int i; int j; float L0; float _pad.</summary>
        struct SpringGpu
        {
            public int   i;    //  4 bytes
            public int   j;    //  4 bytes
            public float L0;   //  4 bytes
            public float _pad; //  4 bytes -> total 16 bytes
        }

        /// <summary>BendPairGpu: 32 bytes. Matches HLSL: int i; int j; int k; float theta0; int alive; int _p0; int _p1; int _p2.</summary>
        struct BendPairGpu
        {
            public int   i;      //  4 bytes
            public int   j;      //  4 bytes
            public int   k;      //  4 bytes
            public float theta0; //  4 bytes
            public int   alive;  //  4 bytes
            public int   _p0;    //  4 bytes
            public int   _p1;    //  4 bytes
            public int   _p2;    //  4 bytes -> total 32 bytes
        }

        // ── Stage 2D-2 (paper §2.1.2 / Berndt [22]): per-particle rotation frame ──────────────
        /// <summary>
        /// RotGpu: 36 bytes = the 3 ROWS of a 3x3 rotation R. NINE EXPLICIT floats (NOT 3 float3 fields)
        /// to mirror the CutPointGpu explicit-components pattern and avoid StructuredBuffer vector-
        /// alignment ambiguity. Matches HLSL `struct RotGpu { float3 r0, r1, r2; }` in Physics.compute
        /// (writer) + the guarded duplicate in Cutting.compute (reader). float3x3 R = float3x3(r0,r1,r2).
        /// </summary>
        public struct RotGpu
        {
            public float r0x, r0y, r0z; // 12 bytes: R row 0
            public float r1x, r1y, r1z; // 12 bytes: R row 1
            public float r2x, r2y, r2z; // 12 bytes: R row 2 -> total 36 bytes (rows of R)

            /// <summary>Identity rotation rows: r0=(1,0,0), r1=(0,1,0), r2=(0,0,1).</summary>
            public static RotGpu Identity => new RotGpu
            {
                r0x = 1f, r0y = 0f, r0z = 0f,
                r1x = 0f, r1y = 1f, r1z = 0f,
                r2x = 0f, r2y = 0f, r2z = 1f,
            };
        }

        // ── Properties ───────────────────────────────────────────────────────────────────────
        public int VoxelCount       { get; private set; }
        public int TriCapacity      { get; private set; }
        public int SurfaceEdgeCount { get; private set; }
        /// <summary>Number of Hermite intersection entries (= Isect.count = IsectWorld.count).</summary>
        public int IsectCount       { get; private set; }
        /// <summary>Number of all interior grid edges (Stage 2B M-T ray sources). design §5.2.</summary>
        public int GridEdgeCount    { get; private set; }
        /// <summary>Cut-point capacity (2 * GridEdgeCount maximum). design §3.6.</summary>
        public int CutPointCapacity { get; private set; }

        // ── Stage 2A: CONN4096 GPU struct ────────────────────────────────────────────────────
        // Mirrors the HLSL struct Conn4096Gpu in CuttingCommon.hlsl.
        // 9 ints = 36 bytes: compCount + one component-id per vertex (v0..v7).
        // paper §2.1.1: 4096-entry LUT; design §5.4; design §3.8.
        public struct Conn4096Gpu
        {
            public int compCount; //  4 bytes: number of connected components (1..8)
            public int v0;        //  4 bytes: dense component id for local vertex 0
            public int v1;        //  4 bytes: ... vertex 1
            public int v2;        //  4 bytes
            public int v3;        //  4 bytes
            public int v4;        //  4 bytes
            public int v5;        //  4 bytes
            public int v6;        //  4 bytes
            public int v7;        //  4 bytes  -> total 36 bytes
        }

        // Task 2 (Stage 1B-i): physics counts
        /// <summary>Total corner count (= particle count). Mirrors BackgroundGrid.cornerCount.</summary>
        public int CornerCount      { get; private set; }
        /// <summary>Number of structural springs.</summary>
        public int SpringCount      { get; private set; }
        /// <summary>Number of bending pairs (= 15 * CornerCount).</summary>
        public int BendPairCount    { get; private set; }

        // ── Core preprocessing buffers ────────────────────────────────────────────────────
        public ComputeBuffer CornerPos;           // float3[cornerCount]
        public ComputeBuffer VoxelCorner;         // int[8*voxelCount]
        public ComputeBuffer VoxelIsectOffset;    // int[voxelCount]
        public ComputeBuffer VoxelIsectCount;     // int[voxelCount]
        public ComputeBuffer Isect;               // IsectGpu[isect.Length]
        /// <summary>
        /// Deformed isect world positions. float3[IsectCount], stride 12.
        /// Written each frame by RebuildIsectWorld; read by DC_FeaturePoints for p_i.
        /// PAPER-SILENT: normal stays rest (frozen); only position is rebuilt per frame.
        /// </summary>
        public ComputeBuffer IsectWorld;          // float3[isect.Length], stride 12

        // ── Feature point buffers ─────────────────────────────────────────────────────────
        public ComputeBuffer VoxelExternalFP;     // float4[voxelCount] (xyz=pos, w=valid)

        // ── Normal buffers (Task 8 Step 1) ───────────────────────────────────────────────
        public ComputeBuffer VoxelExternalFPNormal;  // int[3*voxelCount] fixed-point accumulator
        public ComputeBuffer VoxelExternalFPNormalF; // float3[voxelCount] normalized output

        // ── Triangle output buffers ───────────────────────────────────────────────────────
        public ComputeBuffer Tri;           // int[3*TriCapacity] vertex indices
        public ComputeBuffer TriCounter;    // uint[1] running index count (InterlockedAdd target)
        public ComputeBuffer IndirectArgs;  // uint[4] for DrawProceduralIndirect

        // ── Surface edge buffer (Task 7 Step 2) ──────────────────────────────────────────
        public ComputeBuffer SurfaceEdges;  // GridEdgeGpu[surfaceEdges.Length], stride 32

        // ── Stage 2A: CONN4096 connectivity LUT buffer ───────────────────────────────────
        // Conn4096Gpu[4096], stride 36 (9 ints per entry).
        // Uploaded ONCE at construction; grid-independent (the LUT depends only on the
        // 12-bit cut-mask, not on voxel dimensions).
        // paper §2.1.1: 2^12 = 4096 configurations; design §5.4 / §3.8.
        public ComputeBuffer Conn4096;      // Conn4096Gpu[4096], stride 36

        // ── Stage 2B: cut-detection buffers ──────────────────────────────────────────────
        /// <summary>
        /// 12-bit cut-state bitmask per voxel. uint[voxelCount], stride 4, init 0.
        /// Bit e of voxel v is set when local edge e of v was cut by the blade.
        /// Marked via InterlockedOr in DetectCut for all 4 sharing voxels via EdgeStencil.
        /// design §3.4 / §5.2; paper §2.1.3.
        /// </summary>
        public ComputeBuffer VoxelCutMask;  // uint[voxelCount], stride 4

        /// <summary>
        /// RC2 occupancy gate (plan §2 / §7). uint[VoxelCount], stride 4. voxelOccupied[v] = OR of the
        /// 8 corners' cornerInside (1  - touches tissue, 0  - all-outside empty space). Uploaded once in
        /// Upload (grid-static); bound to BOTH the cut kernels (Cutting.compute) and DC_Stitch
        /// (Recon.compute) so neither emits geometry in non-tissue voxels (the spurious flap). design §2.1.
        /// </summary>
        public ComputeBuffer VoxelOccupied; // uint[voxelCount], stride 4

        /// <summary>
        /// All interior grid edges as M-T ray sources. GridEdgeGpu2[gridEdgeCount], stride 16.
        /// Uploaded once at init; re-uploaded if grid topology changes.
        /// design §3 (GridEdges buffer); 2B plan Task 1 Step 2.
        /// </summary>
        public ComputeBuffer GridEdges;     // GridEdgeGpu2[gridEdgeCount], stride 16

        /// <summary>
        /// Append buffer for cut points. CutPointGpu[0..cutPointCapacity], stride 32.
        /// CutPointCounter (uint[1]) is the running write index (InterlockedAdd).
        /// Two cut points are emitted per cut edge (one per endpoint particle).
        /// design §3.6 / §5.3; 2B plan Task 1 Step 2.
        /// </summary>
        public ComputeBuffer CutPoint;      // CutPointGpu[cutPointCapacity], stride 32
        /// <summary>
        /// Running cut-point count. uint[1], stride 4. Cumulative  - NOT reset per frame.
        /// DetectCut emits 2 points per edge only on its first cut (review R2 #1), so the
        /// counter accumulates monotonically; zeroed only at init (Upload) / re-init.
        /// </summary>
        public ComputeBuffer CutPointCounter; // uint[1], stride 4

        // ── Stage 2C: cut feature-point buffers (plan §1.1) ──────────────────────────────
        // ALL atomic accumulators are FLAT int buffers (plan A2-F1): HLSL SM5 InterlockedAdd
        // requires a scalar int/uint lvalue, so int4 is invalid. Slot index = 8*voxelId+comp.
        /// <summary>Internal cut feature point. float4[8*VoxelCount] (xyz=FP world, w=valid), stride 16.
        /// HLSL _CutFP; slot = 8*voxelId+comp. paper §2.1.1 centroid; design §3.4 / §4.2.</summary>
        public ComputeBuffer CutFP;          // float4[8*VoxelCount], stride 16
        /// <summary>Stage 2D-2b (paper Discussion, clean:1271-1276): previous-frame DISPLAYED cut FP.
        /// float4[8*VoxelCount] (xyz=FP world, w=valid), stride 16. HLSL _PrevCutFP; slot = 8*voxelId+comp.
        /// PERSISTS across frames (NOT cleared by ClearCutFP). InterpCutFP EMA-smooths _CutFP toward curr and
        /// writes the displayed FP back here for the next frame. Zero-init (w=0  - invalid) at Upload so frame-0
        /// / newly-valid slots appear at their current value (no interp from garbage). design §3.A / §7 M4.</summary>
        public ComputeBuffer PrevCutFP;      // float4[8*VoxelCount], stride 16
        /// <summary>Fixed-point centroid position sum. int[3*8*VoxelCount], stride 4. HLSL _CutFPAccumPos;
        /// indexed 3*slot+0/1/2. POS_SCALE fixed point (plan §3.4).</summary>
        public ComputeBuffer CutFPAccumPos;  // int[3*8*VoxelCount], stride 4
        /// <summary>Centroid count. int[8*VoxelCount], stride 4. HLSL _CutFPAccumCnt; indexed slot.</summary>
        public ComputeBuffer CutFPAccumCnt;  // int[8*VoxelCount], stride 4
        /// <summary>Per-voxel compCount (0 if uncut). uint[VoxelCount], stride 4. HLSL _VoxelFPCount.
        /// design §4.1; paper §2.1.1.</summary>
        public ComputeBuffer VoxelFPCount;   // uint[VoxelCount], stride 4
        /// <summary>Cut-FP normal accumulator. int[3*8*VoxelCount], stride 4. HLSL _CutFPNormal;
        /// indexed 3*slot+0/1/2 (mirrors _ExtNormalFixed). design §3.7.</summary>
        public ComputeBuffer CutFPNormal;    // int[3*8*VoxelCount], stride 4
        /// <summary>Normalized cut-FP normal. float3[8*VoxelCount], stride 12. HLSL _CutFPNormalF
        /// (mirrors _ExtNormalF). design §3.7.</summary>
        public ComputeBuffer CutFPNormalF;   // float3[8*VoxelCount], stride 12

        // ── Task 2 (Stage 1B-i): physics GPU buffers ─────────────────────────────────────
        // Strides: float3=12, float=4, int=4, SpringGpu=16, BendPairGpu=32, KSlope element=24.
        /// <summary>Particle velocities. float3[cornerCount], stride 12. Zeroed on Upload.</summary>
        public ComputeBuffer Vel;
        /// <summary>Per-particle mass. float[cornerCount], stride 4.</summary>
        public ComputeBuffer Mass;
        /// <summary>Pin flags. int[cornerCount], stride 4. 1=pinned (pinned particles do not move).</summary>
        public ComputeBuffer Pinned;
        /// <summary>
        /// Stage 2D-1b active flags (paper §2.1.4, paper:485-499). int[cornerCount], stride 4.
        /// 1=active (tissue: inside + boundary-shell corners), 0=inactive (fully-outside, FROZEN like
        /// pinned). Marshalled byte→int from g.cornerActive in Upload. Bound to Physics.compute as
        /// _Active; inactive corners are excluded from the physical model (no springs/bend-pairs +
        /// frozen pos/vel) so a severed tissue piece is free of the empty-space cage and can fall.
        /// </summary>
        public ComputeBuffer CornerActive;
        /// <summary>Fixed-point force accumulator (int3 per particle). int[3*cornerCount], stride 4.</summary>
        public ComputeBuffer ForceInt;
        /// <summary>External force per particle (e.g. gravity). float3[cornerCount], stride 12.</summary>
        public ComputeBuffer ExtForce;
        /// <summary>
        /// RK45 slope scratch: 7 stages x {float3 dx, float3 dv} per particle = stride 24.
        /// ComputeBuffer(7*cornerCount, 24). Indexed: stage*cornerCount + particleIdx.
        /// </summary>
        public ComputeBuffer KSlope;
        /// <summary>RK trial position. float3[cornerCount], stride 12.</summary>
        public ComputeBuffer YTrialPos;
        /// <summary>RK trial velocity. float3[cornerCount], stride 12.</summary>
        public ComputeBuffer YTrialVel;
        /// <summary>
        /// Structural springs. SpringGpu[springCount], stride 16.
        /// NOTE: Springs is built and uploaded here but is NOT read by Stage-1 force kernels;
        /// structural force is gather-based via NbrIdx/RestNbr. Springs is reserved for Stage-2
        /// cut bookkeeping (alive flag toggling on severed edges).
        /// </summary>
        public ComputeBuffer Springs;
        /// <summary>Bending pairs. BendPairGpu[bendPairCount], stride 32.</summary>
        public ComputeBuffer BendPairs;
        /// <summary>Per-corner neighbor corner indices. int[6*cornerCount], stride 4. -1 = no neighbor.</summary>
        public ComputeBuffer NbrIdx;
        /// <summary>Per-corner rest offset to each neighbor. float3[6*cornerCount], stride 12.</summary>
        public ComputeBuffer RestNbr;
        /// <summary>Fixed-point error max for adaptive RK45 D4 step. uint[1], stride 4.</summary>
        public ComputeBuffer ErrMax;
        /// <summary>
        /// Stage 2D-2 (paper §2.1.2 / Berndt [22]): per-particle rotation R. RotGpu[CornerCount], stride 36
        /// (3 rows of R). Written each frame by Physics.compute ComputeParticleRot (post-sever); read by the
        /// cut kernels (DetectCut/EmitCutPoint + AccumulateCutFP) so cut-surface points follow tissue
        /// ROTATION. PERSISTS across frames and is INITIALISED TO IDENTITY at Upload (so frame-0 emits =
        /// the pre-2D-2 static offset = no regression). Single writer (ComputeParticleRot) = race-free.
        /// </summary>
        public ComputeBuffer ParticleRot;

        // ─────────────────────────────────────────────────────────────────────────────────────
        public ReconBuffers(BackgroundGrid g, int triCapacity)
        {
            VoxelCount       = g.voxelCount;
            TriCapacity      = triCapacity;
            SurfaceEdgeCount = g.surfaceEdges != null ? g.surfaceEdges.Length : 0;
            GridEdgeCount    = g.gridEdges    != null ? g.gridEdges.Length    : 0;
            // 2 cut points per edge maximum; at least 1 to avoid zero-size ComputeBuffer
            CutPointCapacity = Math.Max(1, 2 * GridEdgeCount);

            int cc = g.cornerPos.Length;

            // Capture isect length once; used for both Isect and IsectWorld allocations.
            int isectLen = g.isect.Length;
            IsectCount = Math.Max(1, isectLen);

            // IsectGpu = 40 bytes (float3+float3+int+int+float+float); plan's "32" was wrong
            CornerPos        = new ComputeBuffer(cc, 12);
            VoxelCorner      = new ComputeBuffer(8 * VoxelCount, 4);
            VoxelIsectOffset = new ComputeBuffer(VoxelCount, 4);
            VoxelIsectCount  = new ComputeBuffer(VoxelCount, 4);
            Isect            = new ComputeBuffer(IsectCount, 40); // IsectGpu = 40 bytes
            // IsectWorld: float3 per isect entry (12 bytes). Written per-frame by RebuildIsectWorld.
            // At rest, RebuildIsectWorld(CornerPos=rest) -> IsectWorld == globalRest (static 1A path preserved).
            IsectWorld       = new ComputeBuffer(IsectCount, 12); // float3 = 12 bytes

            VoxelExternalFP  = new ComputeBuffer(VoxelCount, 16);       // float4 = 16 bytes

            VoxelExternalFPNormal  = new ComputeBuffer(3 * VoxelCount, 4);  // int x 3 per voxel
            VoxelExternalFPNormalF = new ComputeBuffer(VoxelCount, 12);     // float3 = 12 bytes (Task 8 Step 1)

            Tri          = new ComputeBuffer(3 * TriCapacity, 4);             // int
            TriCounter   = new ComputeBuffer(1, 4);                           // uint
            IndirectArgs = new ComputeBuffer(4, 4, ComputeBufferType.IndirectArguments);

            // Task 7 Step 2: stride 32 (GridEdgeGpu = 32 bytes as laid out above)
            SurfaceEdges = new ComputeBuffer(Math.Max(1, SurfaceEdgeCount), 32);

            // ── Stage 2A: allocate + upload CONN4096 (once, grid-independent) ───────────
            // paper §2.1.1: 2^12 = 4096 cut-mask configurations; design §5.4 / §3.8.
            // Conn4096Gpu = 9 ints = 36 bytes. Uploaded here; never re-uploaded unless rebuilt.
            Conn4096 = new ComputeBuffer(ConnectivityLUT.CONFIG_COUNT, 36); // Conn4096Gpu[4096]
            UploadConn4096();

            // ── Stage 2B: allocate cut-detection buffers ─────────────────────────────────
            // VoxelCutMask: uint per voxel; init to 0 via SetData below. design §3.4.
            VoxelCutMask = new ComputeBuffer(Math.Max(1, VoxelCount), 4);   // uint[voxelCount]
            // VoxelOccupied: uint per voxel (RC2 gate); uploaded once from g.voxelOccupied. plan §7.
            VoxelOccupied = new ComputeBuffer(Math.Max(1, VoxelCount), 4);  // uint[voxelCount]
            // GridEdges: GridEdgeGpu2[gridEdgeCount], stride 16 (int+int3 = 4+12 bytes).
            // design §3 / §5.2; 2B plan Task 1 Step 2.
            GridEdges    = new ComputeBuffer(Math.Max(1, GridEdgeCount), 16);
            // CutPoint: CutPointGpu[cutPointCapacity], stride 32 (float3+int+int+float pad = 24 - 2).
            // design §3.6 / §5.3; 2B plan Task 1 Step 2.
            CutPoint         = new ComputeBuffer(CutPointCapacity, 32);
            CutPointCounter  = new ComputeBuffer(1, 4);                     // uint[1]

            // ── Stage 2C: allocate cut feature-point buffers (plan §1.1) ─────────────────
            // Slot space = 8*VoxelCount. All accumulators FLAT int (plan A2-F1).
            int slotCount = 8 * VoxelCount;
            CutFP         = new ComputeBuffer(Math.Max(1, slotCount),     16); // float4[8*VC]
            // Stage 2D-2b: previous-frame displayed cut FP (EMA history). Same shape/stride as CutFP.
            PrevCutFP     = new ComputeBuffer(Math.Max(1, slotCount),     16); // float4[8*VC]
            CutFPAccumPos = new ComputeBuffer(Math.Max(1, 3 * slotCount),  4); // int[3*8*VC]
            CutFPAccumCnt = new ComputeBuffer(Math.Max(1, slotCount),      4); // int[8*VC]
            VoxelFPCount  = new ComputeBuffer(Math.Max(1, VoxelCount),     4); // uint[VC]
            CutFPNormal   = new ComputeBuffer(Math.Max(1, 3 * slotCount),  4); // int[3*8*VC]
            CutFPNormalF  = new ComputeBuffer(Math.Max(1, slotCount),     12); // float3[8*VC]

            // ── Task 2 (Stage 1B-i): allocate physics buffers ────────────────────────────
            CornerCount   = g.cornerCount;
            SpringCount   = g.springs   != null ? g.springs.Length   : 0;
            BendPairCount = g.bendPairs != null ? g.bendPairs.Length : 0;

            int cn = CornerCount;

            Vel       = new ComputeBuffer(Math.Max(1, cn),            12); // float3[cn]
            Mass      = new ComputeBuffer(Math.Max(1, cn),             4); // float[cn]
            Pinned    = new ComputeBuffer(Math.Max(1, cn),             4); // int[cn]
            CornerActive = new ComputeBuffer(Math.Max(1, cn),          4); // int[cn] (2D-1b active flags)
            ForceInt  = new ComputeBuffer(Math.Max(1, 3 * cn),         4); // int[3*cn]
            ExtForce  = new ComputeBuffer(Math.Max(1, cn),            12); // float3[cn]
            KSlope    = new ComputeBuffer(Math.Max(1, 7 * cn),        24); // {float3 dx; float3 dv}[7*cn], stride 24
            YTrialPos = new ComputeBuffer(Math.Max(1, cn),            12); // float3[cn]
            YTrialVel = new ComputeBuffer(Math.Max(1, cn),            12); // float3[cn]
            Springs   = new ComputeBuffer(Math.Max(1, SpringCount),   16); // SpringGpu[springCount], stride 16
            BendPairs = new ComputeBuffer(Math.Max(1, BendPairCount), 32); // BendPairGpu[bendPairCount], stride 32
            NbrIdx    = new ComputeBuffer(Math.Max(1, 6 * cn),         4); // int[6*cn]
            RestNbr   = new ComputeBuffer(Math.Max(1, 6 * cn),        12); // float3[6*cn]
            ErrMax    = new ComputeBuffer(1,                            4); // uint[1]
            // Stage 2D-2 (paper §2.1.2 / Berndt [22]): per-particle rotation. RotGpu = 36 bytes (3 rows of
            // R); 36-byte stride is proven in-project (Conn4096 above). Initialised to identity in Upload.
            ParticleRot = new ComputeBuffer(Math.Max(1, cn),          36); // RotGpu[cn], stride 36
        }

        public void Upload(BackgroundGrid g)
        {
            CornerPos.SetData(g.cornerPos);
            VoxelCorner.SetData(g.voxelCorner);
            VoxelIsectOffset.SetData(g.voxelIsectOffset);
            VoxelIsectCount.SetData(g.voxelIsectCount);

            // Upload Hermite intersections
            var gpuIsect = new IsectGpu[Math.Max(1, g.isect.Length)];
            for (int i = 0; i < g.isect.Length; i++)
            {
                var e = g.isect[i];
                gpuIsect[i] = new IsectGpu
                {
                    globalRest = e.globalRest,
                    normal     = e.normal,
                    cornerA    = e.cornerA,
                    cornerB    = e.cornerB,
                    t          = e.t,
                    insideA    = e.insideA   // B1 (plan §7): 1  - cornerA inside, else cornerB
                };
            }
            Isect.SetData(gpuIsect);

            // Task 7 Step 2: Upload surface edges
            if (g.surfaceEdges != null && g.surfaceEdges.Length > 0)
            {
                var gpuEdges = new GridEdgeGpu[g.surfaceEdges.Length];
                for (int i = 0; i < g.surfaceEdges.Length; i++)
                {
                    var se = g.surfaceEdges[i];
                    gpuEdges[i] = new GridEdgeGpu
                    {
                        axis    = se.axis,
                        bcX     = se.baseCorner.x,
                        bcY     = se.baseCorner.y,
                        bcZ     = se.baseCorner.z,
                        insideA = se.insideA,
                        _pad    = 0,
                        _pad2   = 0,
                        _pad3   = 0,
                    };
                }
                SurfaceEdges.SetData(gpuEdges);
            }

            // ── Stage 2B: upload all interior grid edges + init cut buffers ──────────────
            // GridEdgeGpu2: int axis + int3 baseCorner = 16 bytes. design §5.2 / §3.
            if (g.gridEdges != null && g.gridEdges.Length > 0)
            {
                var gpuGE = new GridEdgeGpu2[g.gridEdges.Length];
                for (int i = 0; i < g.gridEdges.Length; i++)
                {
                    var ge = g.gridEdges[i];
                    gpuGE[i] = new GridEdgeGpu2
                    {
                        axis = ge.axis,
                        bcX  = ge.baseCorner.x,
                        bcY  = ge.baseCorner.y,
                        bcZ  = ge.baseCorner.z,
                    };
                }
                GridEdges.SetData(gpuGE);
            }
            // VoxelCutMask: cleared to 0 at init (paper §2.1.3; design §3.4). Persists across
            // frames thereafter (cumulative cut state).
            VoxelCutMask.SetData(new uint[Math.Max(1, VoxelCount)]);
            // VoxelOccupied (RC2 gate, plan §7): grid-static OR of the 8 corners' cornerInside.
            // Uploaded once here (never changes within a build); marshalled byte→uint (1/0).
            var occ = new uint[Math.Max(1, VoxelCount)];
            if (g.voxelOccupied != null)
                for (int v = 0; v < VoxelCount; v++) occ[v] = g.voxelOccupied[v];
            VoxelOccupied.SetData(occ);
            // CutPointCounter: cleared to 0 at init ONLY. Cumulative thereafter  - cut points are
            // emitted once per edge (first-cut transition); not reset per frame (review R2 #1).
            CutPointCounter.SetData(new uint[1]);

            // ── Stage 2C: zero-init cut feature-point buffers (plan §1.1) ────────────────
            // ClearCutFP zeroes these every frame, but a clean init guarantees no NaN on the
            // very first dc.Build (which runs before any cut kernel). _CutFP.w=0 - invalid -
            // FetchPos/shader read of an external-only frame never touches a stale FP.
            int slotCount = 8 * VoxelCount;
            CutFP.SetData(new float4[Math.Max(1, slotCount)]);          // w=0  - invalid
            // Stage 2D-2b (§7 M4): EXPLICIT zero-init of the EMA history  - REQUIRED for frame-0 seeding
            // (w=0  - invalid  - InterpCutFP's newly-valid branch makes the first cut FP appear at curr, not
            // interp from garbage). Do NOT rely on implicit ComputeBuffer zero (project convention, :473).
            PrevCutFP.SetData(new float4[Math.Max(1, slotCount)]);      // w=0  - invalid (EMA seed)
            CutFPAccumPos.SetData(new int[Math.Max(1, 3 * slotCount)]);
            CutFPAccumCnt.SetData(new int[Math.Max(1, slotCount)]);
            VoxelFPCount.SetData(new uint[Math.Max(1, VoxelCount)]);
            CutFPNormal.SetData(new int[Math.Max(1, 3 * slotCount)]);
            CutFPNormalF.SetData(new float3[Math.Max(1, slotCount)]);

            // ── Task 2 (Stage 1B-i): upload physics data ─────────────────────────────────
            int cn = CornerCount;

            // Zero velocities (particles start at rest)
            Vel.SetData(new float3[Math.Max(1, cn)]);

            // Upload per-particle mass (float[cn])
            Mass.SetData(g.mass);

            // Upload pin flags as int[] (byte[] -> int[] conversion)
            var pinnedInt = new int[Math.Max(1, cn)];
            for (int i = 0; i < cn; i++) pinnedInt[i] = g.pinned[i];
            Pinned.SetData(pinnedInt);

            // Stage 2D-1b: upload active flags as int[] (byte[] -> int[] conversion). paper §2.1.4.
            // Inactive (0) corners are frozen in-kernel exactly like pinned (Physics.compute _Active).
            var activeInt = new int[Math.Max(1, cn)];
            for (int i = 0; i < cn; i++) activeInt[i] = g.cornerActive[i];
            CornerActive.SetData(activeInt);

            // Upload neighbor indices (int[6*cn])
            NbrIdx.SetData(g.nbrIdx);

            // Upload rest neighbor offsets (float3[6*cn])
            RestNbr.SetData(g.restNbr);

            // Stage 2D-2 (paper §2.1.2 / Berndt [22]): initialise the per-particle rotation to IDENTITY.
            // _ParticleRot PERSISTS across frames; a frame-0 (pre-first-ComputeParticleRot) cut emits with
            // R=identity  - the pre-2D-2 static world offset  - no regression. ComputeParticleRot overwrites
            // active/unpinned corners each frame thereafter.
            var rotIdentity = new RotGpu[Math.Max(1, cn)];
            for (int i = 0; i < rotIdentity.Length; i++) rotIdentity[i] = RotGpu.Identity;
            ParticleRot.SetData(rotIdentity);

            // Marshal and upload structural springs (SpringGpu[springCount])
            if (SpringCount > 0)
            {
                var gpuSprings = new SpringGpu[SpringCount];
                for (int i = 0; i < SpringCount; i++)
                {
                    var s = g.springs[i];
                    gpuSprings[i] = new SpringGpu { i = s.i, j = s.j, L0 = s.L0, _pad = 0f };
                }
                Springs.SetData(gpuSprings);
            }

            // Marshal and upload bending pairs (BendPairGpu[bendPairCount])
            if (BendPairCount > 0)
            {
                var gpuBP = new BendPairGpu[BendPairCount];
                for (int i = 0; i < BendPairCount; i++)
                {
                    var bp = g.bendPairs[i];
                    gpuBP[i] = new BendPairGpu
                    {
                        i      = bp.i,
                        j      = bp.j,
                        k      = bp.k,
                        theta0 = bp.theta0,
                        alive  = bp.alive,
                        _p0    = 0, _p1 = 0, _p2 = 0
                    };
                }
                BendPairs.SetData(gpuBP);
            }

            // ExtForce, ForceInt, KSlope, YTrialPos, YTrialVel, ErrMax are zero-initialized
            // by Unity ComputeBuffer on creation; no explicit upload needed here.
            // (Caller sets ExtForce per-frame for external forces.)
        }

        // ── Stage 2A: upload helper (called once from ctor) ──────────────────────────────
        // Builds ConnectivityLUT on the CPU and marshals it into Conn4096.
        // Grid-independent  - the LUT encodes only the 12-bit cut-mask topology.
        // paper §2.1.1; design §5.4 / §3.8; PAPER-SILENT: dense renumber.
        void UploadConn4096()
        {
            ConnectivityLUT.Build(out int[] compCount, out int[] vertToComp);

            var gpuData = new Conn4096Gpu[ConnectivityLUT.CONFIG_COUNT];
            for (int cfg = 0; cfg < ConnectivityLUT.CONFIG_COUNT; cfg++)
            {
                int b = cfg * 8; // base offset into vertToComp
                gpuData[cfg] = new Conn4096Gpu
                {
                    compCount = compCount[cfg],
                    v0 = vertToComp[b + 0],
                    v1 = vertToComp[b + 1],
                    v2 = vertToComp[b + 2],
                    v3 = vertToComp[b + 3],
                    v4 = vertToComp[b + 4],
                    v5 = vertToComp[b + 5],
                    v6 = vertToComp[b + 6],
                    v7 = vertToComp[b + 7],
                };
            }
            Conn4096.SetData(gpuData);
        }

        public void Dispose()
        {
            CornerPos?.Dispose();
            VoxelCorner?.Dispose();
            VoxelIsectOffset?.Dispose();
            VoxelIsectCount?.Dispose();
            Isect?.Dispose();
            IsectWorld?.Dispose();
            VoxelExternalFP?.Dispose();
            VoxelExternalFPNormal?.Dispose();
            VoxelExternalFPNormalF?.Dispose();   // Task 8 Step 1
            Tri?.Dispose();
            TriCounter?.Dispose();
            IndirectArgs?.Dispose();
            SurfaceEdges?.Dispose();             // Task 7 Step 2
            Conn4096?.Dispose();                 // Stage 2A: CONN4096 LUT
            VoxelCutMask?.Dispose();             // Stage 2B: 12-bit cut mask per voxel
            VoxelOccupied?.Dispose();            // Stage 2C fix: RC2 occupancy gate
            GridEdges?.Dispose();                // Stage 2B: all interior grid edges (M-T sources)
            CutPoint?.Dispose();                 // Stage 2B: emitted cut points (2 per cut edge)
            CutPointCounter?.Dispose();          // Stage 2B: cut point running counter

            // Stage 2C: dispose cut feature-point buffers
            CutFP?.Dispose();
            PrevCutFP?.Dispose();                // Stage 2D-2b: EMA history of the displayed cut FP
            CutFPAccumPos?.Dispose();
            CutFPAccumCnt?.Dispose();
            VoxelFPCount?.Dispose();
            CutFPNormal?.Dispose();
            CutFPNormalF?.Dispose();

            // Task 2 (Stage 1B-i): dispose physics buffers
            Vel?.Dispose();
            Mass?.Dispose();
            Pinned?.Dispose();
            CornerActive?.Dispose();   // Stage 2D-1b: tissue-only active flags
            ForceInt?.Dispose();
            ExtForce?.Dispose();
            KSlope?.Dispose();
            YTrialPos?.Dispose();
            YTrialVel?.Dispose();
            Springs?.Dispose();
            BendPairs?.Dispose();
            NbrIdx?.Dispose();
            RestNbr?.Dispose();
            ErrMax?.Dispose();
            ParticleRot?.Dispose();   // Stage 2D-2: per-particle rotation frame (Berndt R)
        }
    }
}
