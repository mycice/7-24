#ifndef CUTTING_COMMON_INCLUDED
#define CUTTING_COMMON_INCLUDED

// CuttingCommon.hlsl — Stage 2B + 2C shared definitions
// Provides: Conn4096Gpu mirror, CutPointGpu, GridEdgeGpu2, MollerTrumbore (Eq6-8),
//           and (2C) the pure grid-convention helpers CornerOffsetHLSL / LocalCornerOf /
//           Conn4096Comp + the fixed-point scales POS_SCALE / NORMAL_SCALE.
// All kernels in Cutting.compute include this header.
// Traceability: paper §2.1.3 (Eq6-8 M-T) / §2.1.1 (Fig 2.4/2.5 cut surface); design §5.2 / §3.4 / §3.6 / §5.3.
//
// NOTE (2C placement, HLSL declaration ordering): the uniform/buffer-dependent helpers
// CornerCoordHLSL (needs _Dims), EdgeValid / EdgeIsCut (need _VoxelCutMask + the ES_/Vox helpers)
// from plan §3.1/§3.2 live in Cutting.compute *after* those globals are declared — they cannot be
// defined here because this header is included before the globals. The PURE helpers (no global refs)
// stay here so both DetectCut(2B) and the 2C kernels share one definition.

// ── Stage 2C fixed-point scales ──────────────────────────────────────────────────────────
// POS_SCALE = 1<<14 — cut-FP centroid fixed-point sum scale (plan §3.4 / R-C). C# twin:
// FixedPointAtomic.PosScale. Declared HERE (NOT ReconCommon.hlsl) since the cut kernels include
// CuttingCommon. NORMAL_SCALE mirrors ReconCommon's (1<<20); declared here too for DC_NormalizeCutFP,
// guarded so a future double-include with ReconCommon does not redefine it.
#define POS_SCALE 16384.0
#ifndef NORMAL_SCALE
#define NORMAL_SCALE 1048576.0
#endif

// ── Connectivity LUT entry — mirrors C# Conn4096Gpu (9 ints = 36 bytes) ─────────────────
// paper §2.1.1: compCount + per-vertex dense component id (0..compCount-1).
// design §3.8 / §5.4.
struct Conn4096Gpu
{
    int compCount;  // number of connected components (1..8)
    int v0, v1, v2, v3, v4, v5, v6, v7; // dense comp id for local vertices 0..7
};

// ── Stage 2C pure grid-convention helpers (plan §3.1) — NO global references ──────────────
// CornerOffsetHLSL: the Fig 2.4 local-corner offsets, encoded as a function (same encode-as-
// function pattern as 2B's ES_VoxelOffset — SM5 static multidim arrays are unreliable). MUST equal
// C# GridConventions.CornerOffset[] (locked by CutSurface oracle test 1 = the A1 merge gate).
// paper §2.1.1 / Fig 2.4.
int3 CornerOffsetHLSL(int lc)
{   // V0(0,0,0) V1(1,0,0) V2(1,1,0) V3(0,1,0) V4(0,0,1) V5(1,0,1) V6(1,1,1) V7(0,1,1)
    if (lc == 0) return int3(0,0,0); if (lc == 1) return int3(1,0,0);
    if (lc == 2) return int3(1,1,0); if (lc == 3) return int3(0,1,0);
    if (lc == 4) return int3(0,0,1); if (lc == 5) return int3(1,0,1);
    if (lc == 6) return int3(1,1,1); return int3(0,1,1);
}

// LocalCornerOf: which local corner (0..7) of voxel `voxelCoord` coincides with global corner
// `cornerCoord`; -1 if not a corner of that voxel. Mirror of C# GridConventions.LocalCornerOf.
// This is the VERTEX KEY feeding vertToComp — same Fig 2.4 numbering the CONN4096 LUT was built on
// (design §10). paper §2.1.1 / Fig 2.4.
int LocalCornerOf(int3 cornerCoord, int3 voxelCoord)
{
    int3 d = cornerCoord - voxelCoord;
    [unroll] for (int lc = 0; lc < 8; lc++)
        if (all(CornerOffsetHLSL(lc) == d)) return lc;
    return -1;
}

// Conn4096Comp: vertToComp[mask][lc] = e.v{lc}. The component id of local corner `lc` in entry `e`.
// paper §2.1.1 (connectivity LUT); design §3.8 / §5.4.
int Conn4096Comp(Conn4096Gpu e, int lc)
{
    if (lc == 0) return e.v0; if (lc == 1) return e.v1; if (lc == 2) return e.v2; if (lc == 3) return e.v3;
    if (lc == 4) return e.v4; if (lc == 5) return e.v5; if (lc == 6) return e.v6; return e.v7;
}

// ── Cut-point GPU entry — mirrors C# CutPointGpu (32 bytes) ─────────────────────────────
// Two emitted per cut edge (one per endpoint particle).
// localOffset: as of Stage 2D-2 (paper §2.1.2 / Berndt [22]) this is the owner's MATERIAL-frame offset
//   localOffset = R_owner^T · (P_side - cornerPos[owner]) recorded at cut time (EmitCutPoint), recovered
//   each frame via world = cornerPos[owner] + mul(R_owner, localOffset). AccumulateCutFP (Cutting.compute)
//   is the SOLE reader (grep-confirmed). With R=identity this reduces to the pre-2D-2 static world offset
//   P_side - cornerPos[owner] (no regression on the translation demo).
// design §3.6 / §5.3; 2B plan Task 1 Step 2; 2D-2 plan 2026-06-28-dc-2d2-berndt-rframe.md.
// Stride = 32 bytes: float3(12) + int(4) + int(4) + float(4) + int2_pad(8) = 32.
struct CutPointGpu
{
    float3 localOffset;  // 12 bytes: 2D-2 owner MATERIAL-frame offset (R_owner^T·(P_side-cornerPos[owner]))
    int    ownerParticle;//  4 bytes: global corner index (= particle index)
    int    edgeId;       //  4 bytes: index into _GridEdges (matches GridEdgeGpu2 index)
    float  _pad;         //  4 bytes: pad
    int2   _pad2;        //  8 bytes: pad to 32 bytes total
    // total: 12+4+4+4+8 = 32 bytes — matches C# ComputeBuffer stride 32
};

// ── Interior grid edge entry — mirrors C# GridEdgeGpu2 (16 bytes) ────────────────────────
// Axis-aligned edge with baseCorner as the "A" endpoint.
// design §3 (GridEdges buffer); 2B plan Task 1 Step 2.
struct GridEdgeGpu2
{
    int  axis;        // 4 bytes  offset  0: 0=x, 1=y, 2=z
    int  bcX;         // 4 bytes  offset  4: baseCorner.x
    int  bcY;         // 4 bytes  offset  8: baseCorner.y
    int  bcZ;         // 4 bytes  offset 12: baseCorner.z
    // total: 16 bytes — three explicit ints (NOT int3) to mirror the proven Stage-1
    // GridEdgeGpu HLSL layout (Recon.compute) and remove all StructuredBuffer
    // vector-alignment ambiguity. Matches C# GridEdgeGpu2 field order exactly.
};

// ── Möller-Trumbore ray-triangle intersection (paper Eq6-8) ──────────────────────────────
// ray = VOXEL GRID EDGE  (O = A-endpoint world pos, Dvec = B - A, edge direction).
// triangle = (V0, V1, V2) = a SWEPT BLADE PLANE triangle.
// [paper §2.1.3; gap C21: ray is the voxel edge, NOT the blade]
//
// Variable↔symbol map (coding discipline rule 3):
//   t_ray   ∈ [0,1]  — parametric hit position along the voxel edge (paper Eq6)
//   u_bary, v_bary   — barycentric coords inside the triangle (paper Eq6)
//   det              — shared denominator (paper Eq8)
//   L1, L2           — triangle edge vectors from V0 (paper Eq6: L1=V1-V0, L2=V2-V0)
//   T_vec            — vector from ray origin to V0 (paper Eq7: T = V0 - O, leading sign)
//
// Returns true iff the voxel edge and the triangle actually intersect (all constraints met).
bool MollerTrumbore(float3 O, float3 Dvec, float3 V0, float3 V1, float3 V2,
                    out float t_ray, out float u_bary, out float v_bary)
{
    float3 L1 = V1 - V0;                  // paper Eq6: L1 = V1 - V0 (first triangle edge)
    float3 L2 = V2 - V0;                  // paper Eq6: L2 = V2 - V0 (second triangle edge)
    float3 T_vec = V0 - O;                // paper Eq7: T = V0 - O  (NOTE: V0 minus O, not O minus V0)

    // paper Eq8 shared denominator: det = (L2 × D) · L1
    float det = dot(cross(L2, Dvec), L1); // Eq8: det = dot(cross(L2,Dvec), L1)

    t_ray = 0; u_bary = 0; v_bary = 0;
    if (abs(det) < 1e-12) return false;   // ray parallel to triangle — no intersection

    float inv = 1.0 / det;

    // paper Eq8: t_ray = ((L2 × T) · L1) / det
    t_ray  =  dot(cross(L2, T_vec), L1) * inv;

    // paper Eq8: u_bary = -((L2 × D) · T) / det   [LEADING MINUS — gap C23]
    u_bary = -dot(cross(L2, Dvec), T_vec) * inv;

    // paper Eq8: v_bary = ((D × T) · L1) / det
    v_bary =  dot(cross(Dvec, T_vec), L1) * inv;

    // paper Eq6 conditions: u≥0, v≥0, u+v≤1 (inside triangle) + finite-edge bound 0≤t≤1.
    // EPS tolerance (audit w0o40rg9j): the swept cutting plane is split into TWO triangles on the
    // diagonal Sprev→E; a voxel edge whose hit lands EXACTLY on that shared diagonal or the quad rim
    // can be dropped by strict bounds, leaving an un-severed hinge edge that holds the cut piece
    // together. The paper designs the persisting swept quad specifically to NOT miss collisions
    // (clean:374-381), so a small inclusive tolerance is paper-faithful (closes the diagonal/rim gap;
    // over-inclusion at the rim only touches edges at the blade extent, outside the tissue — harmless).
    const float EPS = 1e-5;
    return (u_bary >= -EPS && v_bary >= -EPS && u_bary + v_bary <= 1.0 + EPS
            && t_ray >= -EPS && t_ray <= 1.0 + EPS);
}

#endif // CUTTING_COMMON_INCLUDED
