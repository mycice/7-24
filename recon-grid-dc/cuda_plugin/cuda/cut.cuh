// cut.cuh — shared CUDA structs + helpers for Stage 3 cutting.
// BYTE-IDENTICAL to the C# ReconBuffers structs + CuttingCommon.hlsl. Mirrors:
//   CutPointGpu (32B), Conn4096Gpu (36B), GridEdgeGpu2 (16B), POS_SCALE,
//   MollerTrumbore (paper Eq6-8), CornerOffset/LocalCornerOf/Conn4096Comp, EdgeStencil (ES_*),
//   VoxId/CorId/CornerCoord, EdgeValid/EdgeIsCut.
// Traceability: paper SS2.1.1 (Fig 2.4/2.5 cut surface), SS2.1.3 (Eq6-8 M-T).

#pragma once
#include "common.cuh"   // float3 ops, dot3/cross3/length3, int3 operator+

// ── Fixed-point scale for the cut-FP centroid sum (mirrors CuttingCommon POS_SCALE = 1<<14) ──
#define POS_SCALE 16384.0f

// ── Cut structs (strides MUST match C# ReconBuffers + CuttingCommon.hlsl) ─────────────────────
// CutPointGpu = 32 bytes: float3 localOffset(12) + int ownerParticle(4) + int edgeId(4) + float3 matNrm(12).
// (v4.1 P2: the former 12 pad bytes now carry the material-frame cut normal — device-only buffer,
// stride unchanged, no C# interop struct reads it.)
struct CutPointGpu
{
    float3 localOffset;   // 12  owner-frame offset on the UNDEFORMED edge: t*L*axisDir + R^T*(±D/2*n_cut) (SS2.1.2)
    int    ownerParticle; //  4  global corner index (= particle index)
    int    edgeId;        //  4  index into gridEdges
    float3 matNrm;        // 12  material-frame OUTWARD cut normal R_emit^T*(side*n_cut) (v4.1 P2 kerf fix):
                          //     reconstructed as R_now*matNrm so the per-component wall plane can pull the
                          //     FP back to ±D/2 (SS2.1.1 "cut edges calculated twice" = cut points are FULL
                          //     crossing points and join the Fig-2.2 refinement, not just the seed).
};
static_assert(sizeof(CutPointGpu) == 32, "CutPointGpu must be 32 bytes");

// Conn4096Gpu = 36 bytes: compCount + 8 dense component ids (v0..v7).
struct Conn4096Gpu
{
    int compCount;
    int v0, v1, v2, v3, v4, v5, v6, v7;   // -> 36 bytes
};
static_assert(sizeof(Conn4096Gpu) == 36, "Conn4096Gpu must be 36 bytes");

// GridEdgeGpu2 = 16 bytes: axis + baseCorner.xyz (all interior grid edges = M-T ray sources).
struct GridEdgeGpu2
{
    int axis;
    int bcX, bcY, bcZ;    // -> 16 bytes
};
static_assert(sizeof(GridEdgeGpu2) == 16, "GridEdgeGpu2 must be 16 bytes");

// RotGpu = 36 bytes (3 rows of R) — mirrors physics.cu RotGpu / C# ReconBuffers.RotGpu. The cut kernels
// read the per-particle rotation (written by ComputeParticleRot) to track cut walls under tissue rotation.
struct RotGpu { float3 r0, r1, r2; };
static_assert(sizeof(RotGpu) == 36, "RotGpu must be 36 bytes");
// mul(R, v): R is row-major (rows r0,r1,r2) -> (r0.v, r1.v, r2.v).
__device__ __forceinline__ float3 mulRv(const RotGpu& R, float3 v)
{
    return make_float3(dot3(R.r0, v), dot3(R.r1, v), dot3(R.r2, v));
}
// mul(transpose(R), v) = R^-1 v = v.x*r0 + v.y*r1 + v.z*r2 (R orthonormal -> transpose = inverse).
__device__ __forceinline__ float3 mulRtv(const RotGpu& R, float3 v)
{
    return v.x * R.r0 + v.y * R.r1 + v.z * R.r2;
}

// ── Conn4096 accessor (vertToComp): dense component id of local corner lc (0..7) ──────────────
__device__ __forceinline__ int Conn4096Comp(const Conn4096Gpu& e, int lc)
{
    switch (lc) {
        case 0: return e.v0; case 1: return e.v1; case 2: return e.v2; case 3: return e.v3;
        case 4: return e.v4; case 5: return e.v5; case 6: return e.v6; default: return e.v7;
    }
}

// ── Fig 2.4 local corner offsets — mirror GridConventions.CornerOffset / CornerOffsetHLSL ─────
__device__ __forceinline__ int3 CornerOffsetCuda(int lc)
{
    switch (lc) {
        case 0: return make_int3(0,0,0); case 1: return make_int3(1,0,0);
        case 2: return make_int3(1,1,0); case 3: return make_int3(0,1,0);
        case 4: return make_int3(0,0,1); case 5: return make_int3(1,0,1);
        case 6: return make_int3(1,1,1); default: return make_int3(0,1,1);
    }
}

// LocalCornerOf: which local corner (0..7) of voxelCoord coincides with cornerCoord; -1 if none.
__device__ __forceinline__ int LocalCornerOf(int3 cornerCoord, int3 voxelCoord)
{
    int dx = cornerCoord.x - voxelCoord.x, dy = cornerCoord.y - voxelCoord.y, dz = cornerCoord.z - voxelCoord.z;
    for (int lc = 0; lc < 8; lc++)
    {
        int3 o = CornerOffsetCuda(lc);
        if (o.x == dx && o.y == dy && o.z == dz) return lc;
    }
    return -1;
}

// VoxId / CorId / inverse(CornerCoord) — mirror GridConventions (VoxId is in common.cuh; re-add CorId here).
__device__ __forceinline__ int CorIdCuda(int3 c, int3 dims) { return c.x + (dims.x + 1) * (c.y + (dims.y + 1) * c.z); }
__device__ __forceinline__ int3 CornerCoordCuda(int idx, int3 dims)
{
    int nx = dims.x + 1, ny = dims.y + 1;
    return make_int3(idx % nx, (idx / nx) % ny, idx / (nx * ny));
}
__device__ __forceinline__ bool VoxInBounds(int3 c, int3 dims)
{
    return c.x >= 0 && c.y >= 0 && c.z >= 0 && c.x < dims.x && c.y < dims.y && c.z < dims.z;
}

// ── EdgeStencil (mirror GridConventions.EdgeStencil / ES_VoxelOffset + ES_LocalEdge) ──────────
// For grid edge (axis, baseCorner): 4 incident voxels (offset) + that edge's localEdge id in each.
__device__ __forceinline__ int3 ES_VoxelOffset(int axis, int slot)
{
    if (axis == 0) { const int3 a[4] = {{0,0,0},{0,-1,0},{0,-1,-1},{0,0,-1}}; return a[slot]; }
    if (axis == 1) { const int3 a[4] = {{0,0,0},{0,0,-1},{-1,0,-1},{-1,0,0}}; return a[slot]; }
                     const int3 a[4] = {{0,0,0},{-1,0,0},{-1,-1,0},{0,-1,0}}; return a[slot];
}
__device__ __forceinline__ int ES_LocalEdge(int axis, int slot)
{
    if (axis == 0) { const int a[4] = {0, 2, 6, 4}; return a[slot]; }
    if (axis == 1) { const int a[4] = {3, 7, 5, 1}; return a[slot]; }
                     const int a[4] = {8, 9, 10, 11}; return a[slot];
}

// EdgeValid: all 4 EdgeStencil voxels in-bounds (the predicate BackgroundGrid used to build gridEdges).
__device__ __forceinline__ bool EdgeValid(int axis, int3 baseC, int3 dims)
{
    for (int s = 0; s < 4; s++) if (!VoxInBounds(baseC + ES_VoxelOffset(axis, s), dims)) return false;
    return true;
}
// EdgeIsCut: localEdge bit set in ANY in-bounds stencil voxel (OR across all 4 — boundary-cut robustness).
__device__ __forceinline__ bool EdgeIsCut(int axis, int3 baseC, int3 dims, const unsigned int* voxelCutMask)
{
    bool cut = false;
    for (int s = 0; s < 4; s++)
    {
        int3 vc = baseC + ES_VoxelOffset(axis, s);
        if (VoxInBounds(vc, dims) && ((voxelCutMask[VoxId(vc, dims)] >> ES_LocalEdge(axis, s)) & 1u) != 0u) cut = true;
    }
    return cut;
}

// ── Möller-Trumbore (paper Eq6-8) ─────────────────────────────────────────────────────────────
// Stage 6 usage (k_DetectCutQuad): ray = the CURRENT DEFORMED grid edge in WORLD space, tri = the
// SS2.1.3 swept cutting plane (2 world triangles from the tool's prev+cur frames). Eq6-8 body
// unchanged/byte-identical. Undeformed-edge recording (alongRest + R^T gap) is done post-detection
// in EmitCutPoint (SS2.1.2). Returns hit + t_ray (along edge), u_bary, v_bary.
__device__ __forceinline__ bool MollerTrumbore(float3 O, float3 Dvec, float3 V0, float3 V1, float3 V2,
                                               float& t_ray, float& u_bary, float& v_bary)
{
    float3 L1 = V1 - V0;              // Eq6: L1 = V1 - V0
    float3 L2 = V2 - V0;              // Eq6: L2 = V2 - V0
    float3 T  = V0 - O;               // Eq7: T = V0 - O
    float  det = dot3(cross3(L2, Dvec), L1);   // Eq8: det = (L2 x D) . L1

    t_ray = 0.f; u_bary = 0.f; v_bary = 0.f;
    if (fabsf(det) < 1e-12f) return false;     // parallel
    float inv = 1.0f / det;

    t_ray  =  dot3(cross3(L2, T), L1) * inv;   // Eq8: t = (L2 x T).L1 / det
    u_bary = -dot3(cross3(L2, Dvec), T) * inv; // Eq8: u = -(L2 x D).T / det  (leading minus)
    v_bary =  dot3(cross3(Dvec, T), L1) * inv; // Eq8: v = (D x T).L1 / det

    const float EPS = 1e-5f;   // inclusive tolerance closes the swept-quad diagonal/rim gap (paper: quad must not miss)
    return (u_bary >= -EPS && v_bary >= -EPS && u_bary + v_bary <= 1.0f + EPS
            && t_ray >= -EPS && t_ray <= 1.0f + EPS);
}

// sign helper (HLSL sign(): -1/0/+1)
__device__ __forceinline__ float signf(float x) { return (x > 0.f) ? 1.0f : ((x < 0.f) ? -1.0f : 0.0f); }
