// dc_recon.cu  - native CUDA port of the static DC reconstruction kernels (Stage 1).
//
// Ported VERBATIM (HLSL -> CUDA is syntactic) from Assets/ReconGridDC/Shaders/Recon.compute:
//   RebuildIsectWorld, DC_FeaturePoints (QEF Eq 1-5, deviations D1/D5), DC_Stitch,
//   DC_ClearNormals, DC_NormalsScatter, DC_NormalsNormalize.
// Dispatch order mirrors DualContouring.Build's ZERO-CUT path exactly:
//   RebuildIsectWorld -> DC_FeaturePoints -> reset TriCounter -> DC_Stitch
//   -> DC_ClearNormals -> DC_NormalsScatter -> DC_NormalsNormalize.
//
// STAGE-1 SCOPE = STATIC, ZERO-CUT surface. In the C# zero-cut path StitchVertId collapses to
// `return voxelId` and FetchPos resolves every tag to _VoxelExternalFP, so the cut-aware branch and
// its buffers (_CutFP / _Conn4096 / _VoxelCutMask / _VoxelOccupied, bit31 tags) are NOT needed here
// and are DEFERRED to Stage 3  - this reproduces the C# zero-cut output exactly (the Stage-1 gate).

#include "common.cuh"
#include "dc_recon.h"
#include "cut.cuh"        // Conn4096Gpu, Conn4096Comp, LocalCornerOf, CorId/CornerCoord (Stage 3 cut-aware stitch)
#include "cut.h"          // cut_fp_chain / cut_build_triangles / cut_normalize_fp / cut_is_ready
#include "physics.h"      // physics_nbr_idx / physics_particle_rot (Stage 6 severed-edge isect rebuild)
#include <stdlib.h>
#include <string.h>

// ─────────────────────────────────────────────────────────────────────────────
// Device buffers (host holds the cudaMalloc'd addresses; passed to kernels as args).
// Mirror ReconBuffers fields used by the zero-cut path.
// ─────────────────────────────────────────────────────────────────────────────
static float3*       d_cornerPos        = nullptr;  // [cornerCount]
static int*          d_voxelCorner      = nullptr;  // [8*voxelCount]
static int*          d_voxelIsectOffset = nullptr;  // [voxelCount]
static int*          d_voxelIsectCount  = nullptr;  // [voxelCount]
static IsectGpu*     d_isect            = nullptr;  // [isectCount]
static GridEdgeGpu*  d_surfaceEdges     = nullptr;  // [surfaceEdgeCount]
static float3*       d_isectWorld       = nullptr;  // [isectCount]
static float4*       d_voxelExternalFP  = nullptr;  // [voxelCount] (xyz=FP, w=valid)
static int*          d_tri              = nullptr;  // [3*triCapacity]
static unsigned int* d_triCounter       = nullptr;  // [1]
static int*          d_extNormalFixed   = nullptr;  // [3*voxelCount]
static float3*       d_extNormalF       = nullptr;  // [voxelCount]
// Stage 3: non-indexed expanded vertex soup (one {pos,normal} per emitted _Tri index) for the copy render.
static float3*       d_expandPos        = nullptr;  // [3*triCapacity]
static float3*       d_expandNrm        = nullptr;  // [3*triCapacity]
// Liver surface rendering (design 2026-07-02-liver-surface-render-design.md §4): per-index
// UV1 = float4(REST anchor, wall flag) so the triplanar texture sticks to the material and the
// cut walls render interior color. Snapshots are taken ONCE at the end of recon_init (the grid is
// at rest there by construction); zero-initialized so the init-time first build (which runs
// BEFORE the snapshots exist) writes benign zeros that no mesh upload ever reads (the first C#
// BuildOrUpdateMesh happens after LCS_Finalize's second build).
static float4*       d_voxelExternalFPRest = nullptr;  // [voxelCount] rest-state skin FPs (§4.1)
static float3*       d_cornerPosRest       = nullptr;  // [cornerCount] rest lattice (§4.2 voxel rest center)
static float4*       d_expandAux           = nullptr;  // [3*triCapacity] xyz=rest anchor, w=wall flag

// Stage 3: cut buffers handed over by cut_init (recon_set_cut_buffers). g_cutActive gates the cut-aware path
//  - when 0 these are null and the stitch/normals/expand take the external-only (Stage-1/2) path byte-for-byte.
static unsigned int* d_cut_voxelCutMask = nullptr;
static unsigned int* d_cut_voxelOccupied= nullptr;
static Conn4096Gpu*  d_cut_conn4096     = nullptr;
static float4*       d_cut_cutFP        = nullptr;
static float3*       d_cut_cutFPNormalF = nullptr;
static int*          d_cut_cutFPNormal  = nullptr;
static int           g_cutActive        = 0;

static ReconInitDesc g_desc = {};
static int3          g_dims = {};
static bool          g_ready = false;

// ─────────────────────────────────────────────────────────────────────────────
// DC_Stitch 4-voxel stencil  - STENCIL_OFF[axis*4 + slot] = voxel offset from baseCorner
// (CCW about +axis). Mirrors Recon.compute STENCIL_OFF / Core/GridConventions.EdgeStencil exactly.
// ─────────────────────────────────────────────────────────────────────────────
__device__ const int3 STENCIL_OFF[12] =
{
    // axis 0 = x
    { 0, 0, 0}, { 0,-1, 0}, { 0,-1,-1}, { 0, 0,-1},
    // axis 1 = y
    { 0, 0, 0}, { 0, 0,-1}, {-1, 0,-1}, {-1, 0, 0},
    // axis 2 = z
    { 0, 0, 0}, {-1, 0, 0}, {-1,-1, 0}, { 0,-1, 0},
};

// ─────────────────────────────────────────────────────────────────────────────
// RebuildIsectWorld  - one thread per isect entry. _IsectWorld[e] = lerp(cornerPos[A], cornerPos[B], t).
// At rest (cornerPos == rest corners) _IsectWorld == globalRest (static 1A path). Normal stays frozen.
//
// Stage 6 ([22] Fig 3/4 + paper SS2.1.2/Fig 2.7): when the surface edge's physical connection has
// been SEVERED, the cell edge must be RECONSTRUCTED from the INSIDE particle's frame with the
// UNDEFORMED edge length  - never lerped across the separated actual endpoints. Lerping across a
// severed edge stretches the isosurface point over the widening gap: under gravity a horizontal cut
// gapes (Y springs are load-bearing)  - margin ridge bumps; a fallen fragment drags the skin into
// long strips. [22] p27: cells "reshape following the movement of the particles directly involved";
// removed neighbours are reconstructed, not referenced.
// ─────────────────────────────────────────────────────────────────────────────
// Uses RotGpu from cut.cuh (static_assert'd 36B there)  - same layout physics.cu writes.
__global__ void k_RebuildIsectWorld(int isectCount,
                                     const IsectGpu* isect, const float3* cornerPos,
                                     float3* isectWorld, int3 dims, float L,
                                     const int* nbrIdx, const RotGpu* particleRot)
{
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= isectCount) return;
    IsectGpu it = isect[e];
    int A = it.cornerA, B = it.cornerB;

    bool severed = false;
    int axis = 0, sgn = 1;
    if (nbrIdx != nullptr)
    {
        int nx = dims.x + 1, ny = dims.y + 1;
        int3 ca = make_int3(A % nx, (A / nx) % ny, A / (nx * ny));
        int3 cb = make_int3(B % nx, (B / nx) % ny, B / (nx * ny));
        int dxi = cb.x - ca.x, dyi = cb.y - ca.y, dzi = cb.z - ca.z;
        axis = (dxi != 0) ? 0 : ((dyi != 0) ? 1 : 2);
        sgn  = (dxi + dyi + dzi) > 0 ? 1 : -1;                    // B relative to A along axis
        // INVARIANT (review S2-m1): on an isect (in-lattice) edge, nbrIdx<0 <=> severed by
        // k_SeverLinks. BackgroundGrid builds nbrIdx for EVERY in-lattice pair (the active-active
        // gate applies to the spring list only); -1 initially occurs only at lattice bounds, which
        // isect edges never touch. If ablation (paper SS2.1.4) ever clears nbrIdx for other reasons,
        // switch this to a dedicated severed bit.
        severed = (nbrIdx[6 * A + 2 * axis + (sgn > 0 ? 1 : 0)] < 0);
    }
    if (!severed || particleRot == nullptr)
    {
        isectWorld[e] = lerp3(cornerPos[A], cornerPos[B], it.t);
        return;
    }

    // Severed: rebuild the missing endpoint as insidePos + R_inside * (±L * axisDir) (§2.1.2:
    // "the new cut point is computed using the corresponding coordinate axes").
    float3 dir = make_float3(axis == 0 ? (float)sgn : 0.f,
                             axis == 1 ? (float)sgn : 0.f,
                             axis == 2 ? (float)sgn : 0.f);       // A->B in rest coords
    if (it.insideA == 1)
    {
        RotGpu R = particleRot[A];
        float3 off = make_float3(dot3(R.r0, dir), dot3(R.r1, dir), dot3(R.r2, dir)) * L;
        isectWorld[e] = lerp3(cornerPos[A], cornerPos[A] + off, it.t);
    }
    else
    {
        RotGpu R = particleRot[B];
        float3 off = make_float3(dot3(R.r0, dir), dot3(R.r1, dir), dot3(R.r2, dir)) * L;
        isectWorld[e] = lerp3(cornerPos[B] - off, cornerPos[B], it.t);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// DC_FeaturePoints  - one thread per voxel. QEF gradient ascent (Eq 1-5).
// DEVIATION D1: gradient F = Σ n_i (n_i · (p_i  - x_k)) at current x_k.
// DEVIATION D5: stop when BoxSdf(x_next) >= 0; write last accepted interior x.
// ─────────────────────────────────────────────────────────────────────────────
__global__ void k_FeaturePoints(int voxelCount, float L, int qefIters,
                                const int* voxelIsectOffset, const int* voxelIsectCount,
                                const float3* isectWorld, const IsectGpu* isect,
                                const int* voxelCorner, const float3* cornerPos,
                                float4* voxelExternalFP)
{
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= voxelCount) return;

    int off = voxelIsectOffset[v];
    int cnt = voxelIsectCount[v];

    if (cnt == 0)
    {
        voxelExternalFP[v] = make_float4(0.f, 0.f, 0.f, 0.f);   // no surface crossing in this voxel
        return;
    }

    // Eq2: initial x = mean of intersection points (deformed IsectWorld).
    float3 x = make_float3(0.f, 0.f, 0.f);
    for (int s = 0; s < cnt; s++)
        x = x + isectWorld[off + s];
    x = x / (float)cnt;   // true divide  - matches HLSL `x /= cnt` source form exactly (Recon.compute:118)

    // Voxel center = mean of 8 deformed corners (Eq5 box center).
    float3 c = make_float3(0.f, 0.f, 0.f);
    for (int cc = 0; cc < 8; cc++)
        c = c + cornerPos[voxelCorner[8 * v + cc]];
    c = c / 8.0f;   // matches HLSL `c /= 8.0` source form (Recon.compute:124)

    for (int it = 0; it < qefIters; it++)
    {
        // DEVIATION D1: QEF negative gradient at current x_k.
        float3 F = make_float3(0.f, 0.f, 0.f);
        for (int s2 = 0; s2 < cnt; s2++)
        {
            float3 p_i = isectWorld[off + s2];     // deformed isect position
            float3 n_i = isect[off + s2].normal;   // rest normal (PAPER-SILENT frozen)
            F = F + n_i * dot3(n_i, p_i - x);      // DEVIATION D1
        }

        // Eq4: decaying step size.
        float a = 0.1f * (1.0f - (float)it / (float)qefIters);
        float3 xn = x + a * F;

        // DEVIATION D5: stop before leaving the voxel; keep last interior iterate.
        if (BoxSdf(xn, c, L) >= 0.0f)
            break;

        x = xn;
    }

    voxelExternalFP[v] = make_float4(x.x, x.y, x.z, 1.0f);   // D5: last interior point
}

// ── StitchVertId (cut-aware)  - mirror Recon.compute StitchVertId. Stage 7 (paper-exact single FP,
// audit wf_60781cea A2): a cut+occupied voxel resolves its SKIN stitch to the SAME per-component FP
// the wall uses (tag bit31)  - the paper's margin continuity comes from vertex sharing, not extra
// geometry. cutActive==0 -> always external (Stage-1/2 byte-identical).
__device__ __forceinline__ int StitchVertId(int voxelId, int3 voxelCoord, int3 insideCornerCoord, int cutActive,
                                            const unsigned int* voxelCutMask, const unsigned int* voxelOccupied,
                                            const Conn4096Gpu* conn4096, const float4* cutFP)
{
    if (!cutActive) return voxelId;
    unsigned int mask = voxelCutMask[voxelId] & 0xFFFu;
    if (mask == 0u || voxelOccupied[voxelId] == 0u) return voxelId;
    int lc = LocalCornerOf(insideCornerCoord, voxelCoord);
    if (lc < 0) return voxelId;
    int comp = Conn4096Comp(conn4096[mask], lc);
    int slot = 8 * voxelId + comp;
    return (cutFP[slot].w > 0.5f) ? (int)(0x80000000u | (unsigned int)slot) : voxelId;
}

// ── FetchPos (cut-aware)  - bit31 -> per-component FP; else _VoxelExternalFP[voxelId].
__device__ __forceinline__ float3 FetchPos(int raw, const float4* voxelExternalFP, const float4* cutFP)
{
    unsigned int u = (unsigned int)raw;
    int idx = (int)(u & 0x3FFFFFFFu);
    if (u & 0x80000000u) return make_float3(cutFP[idx].x, cutFP[idx].y, cutFP[idx].z);
    return make_float3(voxelExternalFP[idx].x, voxelExternalFP[idx].y, voxelExternalFP[idx].z);
}

// ─────────────────────────────────────────────────────────────────────────────
// DC_Stitch  - one thread per surface edge. 4 incident voxels' feature points form a quad  - 2 tris.
// Cut-aware (Stage 3): each vertex resolves to its per-component cut FP (bit31) for a cut occupied voxel,
// else the external FP (voxelId). The cut-voxel skin SPLITS per component -> the outer surface opens.
// ─────────────────────────────────────────────────────────────────────────────
__global__ void k_Stitch(int surfaceEdgeCount, int triCapacity, int3 dims,
                         const GridEdgeGpu* surfaceEdges, int* tri, unsigned int* triCounter,
                         int cutActive, const unsigned int* voxelCutMask, const unsigned int* voxelOccupied,
                         const Conn4096Gpu* conn4096, const float4* cutFP)
{
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= surfaceEdgeCount) return;

    GridEdgeGpu ge = surfaceEdges[e];
    int3 baseCorner = make_int3(ge.bcX, ge.bcY, ge.bcZ);

    // The surface edge's INSIDE-endpoint corner keys the per-component FP.
    int3 axisDir = make_int3(ge.axis == 0 ? 1 : 0, ge.axis == 1 ? 1 : 0, ge.axis == 2 ? 1 : 0);
    int3 insideCornerCoord = (ge.insideA != 0) ? baseCorner : baseCorner + axisDir;

    int base4 = ge.axis * 4;
    int3 c0 = baseCorner + STENCIL_OFF[base4 + 0];
    int3 c1 = baseCorner + STENCIL_OFF[base4 + 1];
    int3 c2 = baseCorner + STENCIL_OFF[base4 + 2];
    int3 c3 = baseCorner + STENCIL_OFF[base4 + 3];
    int v0 = StitchVertId(VoxId(c0, dims), c0, insideCornerCoord, cutActive, voxelCutMask, voxelOccupied, conn4096, cutFP);
    int v1 = StitchVertId(VoxId(c1, dims), c1, insideCornerCoord, cutActive, voxelCutMask, voxelOccupied, conn4096, cutFP);
    int v2 = StitchVertId(VoxId(c2, dims), c2, insideCornerCoord, cutActive, voxelCutMask, voxelOccupied, conn4096, cutFP);
    int v3 = StitchVertId(VoxId(c3, dims), c3, insideCornerCoord, cutActive, voxelCutMask, voxelOccupied, conn4096, cutFP);

    // Winding: STENCIL_OFF is CCW about +axis. insideA==1 keeps CCW; insideA==0 swaps b<->d (flip).
    int a = v0, b = v1, cc = v2, d = v3;
    if (ge.insideA == 0)
    {
        int tmp = b; b = d; d = tmp;
    }

    // Append 2 triangles (6 indices) via atomic counter.
    unsigned int baseIdx = atomicAdd(&triCounter[0], 6u);
    if (baseIdx + 6u > (unsigned int)(triCapacity * 3)) return;   // overflow guard

    tri[baseIdx + 0] = a;  tri[baseIdx + 1] = b;  tri[baseIdx + 2] = cc;   // tri 1: a,b,cc
    tri[baseIdx + 3] = a;  tri[baseIdx + 4] = cc; tri[baseIdx + 5] = d;    // tri 2: a,cc,d (shared diag a-cc)
}

// ─────────────────────────────────────────────────────────────────────────────
// DC_ClearNormals  - one thread per voxel. Zero the fixed-point normal accumulator.
// ─────────────────────────────────────────────────────────────────────────────
__global__ void k_ClearNormals(int voxelCount, int* extNormalFixed)
{
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= voxelCount) return;
    extNormalFixed[3 * v + 0] = 0;
    extNormalFixed[3 * v + 1] = 0;
    extNormalFixed[3 * v + 2] = 0;
}

// Fixed-point atomic normal accumulation (mirrors HLSL AtomicAddNormal).
__device__ __forceinline__ void AtomicAddNormal(int* extNormalFixed, int v, float3 n)
{
    atomicAdd(&extNormalFixed[3 * v + 0], (int)(n.x * NORMAL_SCALE));
    atomicAdd(&extNormalFixed[3 * v + 1], (int)(n.y * NORMAL_SCALE));
    atomicAdd(&extNormalFixed[3 * v + 2], (int)(n.z * NORMAL_SCALE));
}

// Cut-FP fixed-point atomic normal accumulation (mirrors HLSL AtomicAddCutNormal). slot = idx.
__device__ __forceinline__ void AtomicAddCutNormal(int* cutFPNormal, int slot, float3 n)
{
    atomicAdd(&cutFPNormal[3 * slot + 0], (int)(n.x * NORMAL_SCALE));
    atomicAdd(&cutFPNormal[3 * slot + 1], (int)(n.y * NORMAL_SCALE));
    atomicAdd(&cutFPNormal[3 * slot + 2], (int)(n.z * NORMAL_SCALE));
}

// ─────────────────────────────────────────────────────────────────────────────
// DC_NormalsScatter  - one thread per triangle. Area-weighted face normal (via tag-aware FetchPos so
// MIXED external/cut tris are correct) scattered per vertex to _ExtNormalFixed (external) or
// _CutFPNormal (bit31 cut FP). cutActive==0 -> external-only (Stage-1/2 byte-identical).
// ─────────────────────────────────────────────────────────────────────────────
__global__ void k_NormalsScatter(unsigned int triCounterUpper,
                                 const int* tri, const unsigned int* triCounter,
                                 const float4* voxelExternalFP, int* extNormalFixed,
                                 int cutActive, const float4* cutFP, int* cutFPNormal)
{
    unsigned int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= triCounterUpper) return;                 // upper-bound dispatch guard
    unsigned int triCount = triCounter[0] / 3u;
    if (t >= triCount) return;

    int raw0 = tri[3 * t + 0];
    int raw1 = tri[3 * t + 1];
    int raw2 = tri[3 * t + 2];

    // Tag-aware positions (external / skin FP / cut FP) so the face normal of a MIXED triangle is correct.
    float3 p0 = cutActive ? FetchPos(raw0, voxelExternalFP, cutFP) : make_float3(voxelExternalFP[raw0].x, voxelExternalFP[raw0].y, voxelExternalFP[raw0].z);
    float3 p1 = cutActive ? FetchPos(raw1, voxelExternalFP, cutFP) : make_float3(voxelExternalFP[raw1].x, voxelExternalFP[raw1].y, voxelExternalFP[raw1].z);
    float3 p2 = cutActive ? FetchPos(raw2, voxelExternalFP, cutFP) : make_float3(voxelExternalFP[raw2].x, voxelExternalFP[raw2].y, voxelExternalFP[raw2].z);

    float3 fn = cross3(p1 - p0, p2 - p0);             // area-weighted face normal

    #pragma unroll
    for (int k = 0; k < 3; k++)
    {
        unsigned int u = (unsigned int)((k == 0) ? raw0 : (k == 1) ? raw1 : raw2);
        int i = (int)(u & 0x3FFFFFFFu);
        if (cutActive && (u & 0x80000000u)) AtomicAddCutNormal(cutFPNormal, i, fn);   // component FP
        else                                AtomicAddNormal(extNormalFixed, i, fn);   // voxel FP
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// k_ExpandVertices  - one thread per emitted _Tri index. Resolve the tag -> {pos, normal} into the
// non-indexed soup (d_expandPos/d_expandNrm) so the copy render is a flat vertex array. bit31 -> cut FP.
// Liver-render design §4.2: ALSO emit aux = float4(REST anchor, wall flag):
//   wall flag  = bit30 (set ONLY by the wall emitter k_BuildCutTriangles/EmitCutQuad  - bit31 alone
//                would misclassify the cut-voxel SKIN stitches that share the component FP);
//   rest anchor: plain voxelId tag -> that voxel's rest skin FP (exact);
//                bit31 cut FP slot s -> owning voxel v=s>>3's rest skin FP when it has one
//                (surface voxel), else the voxel's REST CENTER = rest corner v000 + L/2
//                (uniform rest lattice; keeps anchors spatially coherent on deep interior walls).
// ─────────────────────────────────────────────────────────────────────────────
__global__ void k_ExpandVertices(unsigned int idxUpper, const int* tri, const unsigned int* triCounter,
                                 const float4* voxelExternalFP, const float3* extNormalF,
                                 int cutActive, const float4* cutFP, const float3* cutFPNormalF,
                                 float3* expandPos, float3* expandNrm,
                                 const float4* voxelExternalFPRest, const float3* cornerPosRest,
                                 const int* voxelCorner, float L, float4* expandAux)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= idxUpper) return;
    if (i >= triCounter[0]) return;
    int raw = tri[i];
    unsigned int u = (unsigned int)raw;
    int idx = (int)(u & 0x3FFFFFFFu);
    if (cutActive && (u & 0x80000000u))
    {
        expandPos[i] = make_float3(cutFP[idx].x, cutFP[idx].y, cutFP[idx].z);   // cut-wall FP
        expandNrm[i] = cutFPNormalF[idx];
        int v = idx >> 3;                                       // slot = 8*voxel + comp
        float4 rfp = voxelExternalFPRest[v];
        float3 anchor = (rfp.w > 0.f)
            ? make_float3(rfp.x, rfp.y, rfp.z)
            : cornerPosRest[voxelCorner[8 * v]] + make_float3(0.5f * L, 0.5f * L, 0.5f * L);
        expandAux[i] = make_float4(anchor.x, anchor.y, anchor.z, (float)((u >> 30) & 1u));
    }
    else
    {
        expandPos[i] = make_float3(voxelExternalFP[idx].x, voxelExternalFP[idx].y, voxelExternalFP[idx].z);
        expandNrm[i] = extNormalF[idx];
        float4 rfp = voxelExternalFPRest[idx];
        expandAux[i] = make_float4(rfp.x, rfp.y, rfp.z, 0.f);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// k_DebugScanTris ([DEBUG-CUT] v4.1 P3)  - one thread per emitted triangle: count dangling bit31
// refs (cutFP.w<0.5 renders at (0,0,0) => spike-to-origin filament) and giant triangles (max edge
// > 4L) classified by tag mix (all-cut=wall / mixed=stitch / none=skin). Direct measurement of the
// user's strand/fan complaint  - nonzero here with ema-jumps==0 reopens the stitch hypotheses.
// ─────────────────────────────────────────────────────────────────────────────
__global__ void k_DebugScanTris(unsigned int tcapTris, const int* tri, const unsigned int* triCounter,
                                const float4* voxelExternalFP, const float4* cutFP,
                                float voxelL, unsigned int* dbg)
{
    unsigned int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= tcapTris) return;
    unsigned int triCount = triCounter[0] / 3u;
    if (triCount > tcapTris) triCount = tcapTris;    // counter runs past capacity by design
    if (t >= triCount) return;

    int   nCut = 0;
    bool  dangle = false;
    float3 p[3];
    #pragma unroll
    for (int k = 0; k < 3; k++)
    {
        int raw = tri[3 * t + k];
        unsigned int u = (unsigned int)raw;
        if (u & 0x80000000u)
        {
            nCut++;
            if (cutFP[(int)(u & 0x3FFFFFFFu)].w < 0.5f) dangle = true;
        }
        p[k] = FetchPos(raw, voxelExternalFP, cutFP);
    }
    if (dangle) atomicAdd(&dbg[6], 1u);

    float m = fmaxf(length3(p[1] - p[0]), fmaxf(length3(p[2] - p[1]), length3(p[0] - p[2])));
    atomicMax(&dbg[10], __float_as_uint(m));         // monotone for non-negative floats
    if (m > 4.f * voxelL)
    {
        if (nCut == 3)      atomicAdd(&dbg[7], 1u);  // wall (EmitCutQuad)
        else if (nCut > 0)  atomicAdd(&dbg[8], 1u);  // mixed (k_Stitch skin/cut seam)
        else                atomicAdd(&dbg[9], 1u);  // pure skin
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// DC_NormalsNormalize  - one thread per voxel. fixed-point accum  - float  - normalize  - _ExtNormalF.
// ─────────────────────────────────────────────────────────────────────────────
__global__ void k_NormalsNormalize(int voxelCount, const int* extNormalFixed, float3* extNormalF)
{
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= voxelCount) return;

    float3 n = make_float3((float)extNormalFixed[3 * v + 0] / NORMAL_SCALE,
                           (float)extNormalFixed[3 * v + 1] / NORMAL_SCALE,
                           (float)extNormalFixed[3 * v + 2] / NORMAL_SCALE);
    float len = length3(n);
    extNormalF[v] = (len > 1e-6f) ? n / len : make_float3(0.f, 0.f, 0.f);   // true divide  - matches HLSL `n / len` (Recon.compute:418)
}

// ─────────────────────────────────────────────────────────────────────────────
// Host helpers
// ─────────────────────────────────────────────────────────────────────────────
static int G(int n) { int g = (n + 63) / 64; return g < 1 ? 1 : g; }

#define CK(call) do { cudaError_t _e = (call); if (_e != cudaSuccess) { recon_shutdown(); return -(int)_e; } } while (0)

// Full DC pipeline  - exactly DualContouring.Build's order on the default stream (separate launches on
// one stream are serialized = the HLSL per-Dispatch implicit barriers). The cut-FP chain + BuildCutTriangles
// + NormalizeCutFP run when cutting is active (cut_is_ready); when inactive they no-op and the cut-aware
// kernels take the external-only path (Stage-1/2 byte-identical).
static cudaError_t recon_build_static()
{
    int vc = g_desc.voxelCount;
    int ic = g_desc.isectCount;
    int sec = g_desc.surfaceEdgeCount;
    int tcap = g_desc.triCapacity;
    int cutActive = g_cutActive;

    // 2. RebuildIsectWorld (FIRST)  - Stage 6: severed surface edges rebuild their missing endpoint
    // from the inside particle's frame ([22] Fig 3/4); pointers null when cutting/physics inactive.
    const int*     rb_nbr = cutActive ? physics_nbr_idx() : nullptr;
    const RotGpu*  rb_rot = cutActive ? (const RotGpu*)physics_particle_rot() : nullptr;
    k_RebuildIsectWorld<<<G(ic), 64>>>(ic, d_isect, d_cornerPos, d_isectWorld,
                                       g_dims, g_desc.L, rb_nbr, rb_rot);
    // 3. DC_FeaturePoints
    k_FeaturePoints<<<G(vc), 64>>>(vc, g_desc.L, g_desc.qefIters,
                                   d_voxelIsectOffset, d_voxelIsectCount,
                                   d_isectWorld, d_isect, d_voxelCorner, d_cornerPos,
                                   d_voxelExternalFP);
    // 4-7b. cut-FP chain (ClearCutFP -> LookupConnectivity -> AccumulateCutFP -> ComputeComponentFP -> InterpCutFP)
    if (cutActive && cut_fp_chain() != 0) return cudaErrorLaunchFailure;
    // 8. reset TriCounter = 0 (head of geometry)
    cudaError_t e = cudaMemset(d_triCounter, 0, sizeof(unsigned int));
    if (e != cudaSuccess) return e;
    // 9. DC_Stitch (cut-aware external skin; Stage 7: cut voxels share the single component FP)
    k_Stitch<<<G(sec), 64>>>(sec, tcap, g_dims, d_surfaceEdges, d_tri, d_triCounter,
                             cutActive, d_cut_voxelCutMask, d_cut_voxelOccupied, d_cut_conn4096,
                             d_cut_cutFP);
    // 10. BuildCutTriangles (cut walls -> SAME _Tri, AFTER stitch)
    if (cutActive && cut_build_triangles() != 0) return cudaErrorLaunchFailure;
    // 11. DC_ClearNormals
    k_ClearNormals<<<G(vc), 64>>>(vc, d_extNormalFixed);
    // 12. DC_NormalsScatter (cut-aware; upper-bound dispatch over triCapacity; kernel guards on TriCounter)
    k_NormalsScatter<<<G(tcap), 64>>>((unsigned int)tcap, d_tri, d_triCounter,
                                      d_voxelExternalFP, d_extNormalFixed,
                                      cutActive, d_cut_cutFP, d_cut_cutFPNormal);
    // 13. DC_NormalsNormalize (external normals)
    k_NormalsNormalize<<<G(vc), 64>>>(vc, d_extNormalFixed, d_extNormalF);
    // 14. DC_NormalizeCutFP (component-FP normals, Stage-7 single family)
    if (cutActive && cut_normalize_fp() != 0) return cudaErrorLaunchFailure;
    // 15. Expand the tagged _Tri into a non-indexed {pos,normal,aux} soup for the copy render.
    k_ExpandVertices<<<G(3 * tcap), 64>>>((unsigned int)(3 * tcap), d_tri, d_triCounter,
                                          d_voxelExternalFP, d_extNormalF,
                                          cutActive, d_cut_cutFP, d_cut_cutFPNormalF,
                                          d_expandPos, d_expandNrm,
                                          d_voxelExternalFPRest, d_cornerPosRest,
                                          d_voxelCorner, g_desc.L, d_expandAux);

    cudaError_t le = cudaGetLastError();
    if (le != cudaSuccess) return le;
    return cudaDeviceSynchronize();   // "Wait for GPU Computation to Complete" (Fig 3.1 barrier)
}

int recon_init(const ReconInitDesc* desc,
               const float* cornerPos,
               const int*   voxelCorner,
               const int*   voxelIsectOffset,
               const int*   voxelIsectCount,
               const void*  isect,
               const void*  surfaceEdges)
{
    if (g_ready) recon_shutdown();
    if (!desc) return -1000;

    g_desc = *desc;
    g_dims = make_int3(desc->dimsX, desc->dimsY, desc->dimsZ);

    int vc   = desc->voxelCount;
    int cc   = desc->cornerCount;
    int ic   = desc->isectCount;
    int sec  = desc->surfaceEdgeCount;
    int tcap = desc->triCapacity;
    if (vc <= 0 || cc <= 0 || tcap <= 0) return -1001;

    // Allocate (Math.Max(1,n) mirrors ReconBuffers so a zero-count buffer never has size 0).
    int icA  = ic  > 0 ? ic  : 1;
    int secA = sec > 0 ? sec : 1;

    CK(cudaMalloc(&d_cornerPos,        (size_t)cc        * sizeof(float3)));
    CK(cudaMalloc(&d_voxelCorner,      (size_t)8 * vc    * sizeof(int)));
    CK(cudaMalloc(&d_voxelIsectOffset, (size_t)vc        * sizeof(int)));
    CK(cudaMalloc(&d_voxelIsectCount,  (size_t)vc        * sizeof(int)));
    CK(cudaMalloc(&d_isect,            (size_t)icA       * sizeof(IsectGpu)));
    CK(cudaMalloc(&d_surfaceEdges,     (size_t)secA      * sizeof(GridEdgeGpu)));
    CK(cudaMalloc(&d_isectWorld,       (size_t)icA       * sizeof(float3)));
    CK(cudaMalloc(&d_voxelExternalFP,  (size_t)vc        * sizeof(float4)));
    CK(cudaMalloc(&d_tri,              (size_t)3 * tcap  * sizeof(int)));
    CK(cudaMalloc(&d_triCounter,                          sizeof(unsigned int)));
    CK(cudaMalloc(&d_extNormalFixed,   (size_t)3 * vc    * sizeof(int)));
    CK(cudaMalloc(&d_extNormalF,       (size_t)vc        * sizeof(float3)));
    CK(cudaMalloc(&d_expandPos,        (size_t)3 * tcap  * sizeof(float3)));   // non-indexed soup (Stage 3 render)
    CK(cudaMalloc(&d_expandNrm,        (size_t)3 * tcap  * sizeof(float3)));
    // Liver-render design §4.1/§4.2: rest snapshots + per-index aux channel. Zero-init the
    // snapshots so the FIRST build below (which runs before the snapshots are taken) writes
    // benign zeros never read by any mesh upload (see the buffer decls note).
    CK(cudaMalloc(&d_voxelExternalFPRest, (size_t)vc      * sizeof(float4)));
    CK(cudaMalloc(&d_cornerPosRest,       (size_t)cc      * sizeof(float3)));
    CK(cudaMalloc(&d_expandAux,           (size_t)3 * tcap * sizeof(float4)));
    CK(cudaMemset(d_voxelExternalFPRest, 0, (size_t)vc * sizeof(float4)));
    CK(cudaMemset(d_cornerPosRest,       0, (size_t)cc * sizeof(float3)));

    // Upload the marshalled grid.
    CK(cudaMemcpy(d_cornerPos,        cornerPos,        (size_t)cc     * sizeof(float3),   cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_voxelCorner,      voxelCorner,      (size_t)8 * vc * sizeof(int),      cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_voxelIsectOffset, voxelIsectOffset, (size_t)vc     * sizeof(int),      cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_voxelIsectCount,  voxelIsectCount,  (size_t)vc     * sizeof(int),      cudaMemcpyHostToDevice));
    if (ic > 0)
        CK(cudaMemcpy(d_isect,        isect,            (size_t)ic     * sizeof(IsectGpu), cudaMemcpyHostToDevice));
    if (sec > 0)
        CK(cudaMemcpy(d_surfaceEdges, surfaceEdges,     (size_t)sec    * sizeof(GridEdgeGpu), cudaMemcpyHostToDevice));

    g_ready = true;

    cudaError_t e = recon_build_static();
    if (e != cudaSuccess) { recon_shutdown(); return -(int)e; }

    // Liver-render design §4.1: REST snapshots  - recon_init just built from the untouched rest
    // cornerPos upload, so d_voxelExternalFP right now IS the rest-state skin-FP field and
    // d_cornerPos IS the uniform rest lattice. Taken exactly once; physics starts later.
    CK(cudaMemcpy(d_voxelExternalFPRest, d_voxelExternalFP, (size_t)vc * sizeof(float4), cudaMemcpyDeviceToDevice));
    CK(cudaMemcpy(d_cornerPosRest,       d_cornerPos,       (size_t)cc * sizeof(float3), cudaMemcpyDeviceToDevice));
    return 0;
}

// Non-indexed expanded soup: vertexCount == indexCount == the emitted triangle index count.
int recon_get_counts(int* vertexCount, int* indexCount)
{
    if (!g_ready) return -1;
    unsigned int idx = 0;
    cudaError_t e = cudaMemcpy(&idx, d_triCounter, sizeof(unsigned int), cudaMemcpyDeviceToHost);
    if (e != cudaSuccess) return -(int)e;
    unsigned int cap = (unsigned int)(g_desc.triCapacity * 3);
    if (idx > cap) idx = cap;
    if (vertexCount) *vertexCount = (int)idx;
    if (indexCount)  *indexCount  = (int)idx;
    return 0;
}

// Copy the expanded vertex soup (positions + normals) back; outIdx (if given) gets the sequential indices.
int recon_get_surface(float* outPos3, float* outNrm3, int* outIdx)
{
    if (!g_ready) return -1;
    unsigned int idx = 0;
    CK(cudaMemcpy(&idx, d_triCounter, sizeof(unsigned int), cudaMemcpyDeviceToHost));
    unsigned int cap = (unsigned int)(g_desc.triCapacity * 3);
    if (idx > cap) idx = cap;
    int N = (int)idx;

    // d_expandPos/d_expandNrm are float3[N] (12 bytes contiguous) -> direct copy into float[3*N].
    if (outPos3 && N > 0) CK(cudaMemcpy(outPos3, d_expandPos, (size_t)N * sizeof(float3), cudaMemcpyDeviceToHost));
    if (outNrm3 && N > 0) CK(cudaMemcpy(outNrm3, d_expandNrm, (size_t)N * sizeof(float3), cudaMemcpyDeviceToHost));
    if (outIdx) for (int i = 0; i < N; i++) outIdx[i] = i;   // non-indexed triangle soup
    return 0;
}

// Liver-render design §4.4: copy the per-index aux channel (xyz = rest anchor, w = wall flag)
// back to host  - same clamped-count contract as recon_get_surface; separate export so the
// existing 3-arg LCS_GetSurface call sites stay ABI-untouched.
int recon_get_surface_aux(float* outAux4)
{
    if (!g_ready || !outAux4) return -1;
    unsigned int idx = 0;
    CK(cudaMemcpy(&idx, d_triCounter, sizeof(unsigned int), cudaMemcpyDeviceToHost));
    unsigned int cap = (unsigned int)(g_desc.triCapacity * 3);
    if (idx > cap) idx = cap;
    int N = (int)idx;
    if (N > 0) CK(cudaMemcpy(outAux4, d_expandAux, (size_t)N * sizeof(float4), cudaMemcpyDeviceToHost));
    return 0;
}

// ── Stage 2 shared accessors ────────────────────────────────────────────────────────────────
float3* recon_corner_pos()   { return d_cornerPos; }
int     recon_corner_count() { return g_desc.cornerCount; }
void    recon_dims(int* x, int* y, int* z) { *x = g_desc.dimsX; *y = g_desc.dimsY; *z = g_desc.dimsZ; }

// Diagnostic readback: copy the LIVE (deformed) cornerPos[3*cornerCount] floats to host. Returns 0 on success.
int recon_readback_corner_pos(float* hostOut)
{
    if (!d_cornerPos || !hostOut) return -1;
    int cc = g_desc.cornerCount;
    cudaError_t e = cudaMemcpy(hostOut, d_cornerPos, (size_t)cc * sizeof(float3), cudaMemcpyDeviceToHost);
    return (e == cudaSuccess) ? 0 : -(int)e;
}

int recon_upload_corner_pos(const float* hostPos)
{
    if (!d_cornerPos || !hostPos) return -1;
    int cc = g_desc.cornerCount;
    cudaError_t e = cudaMemcpy(d_cornerPos, hostPos, (size_t)cc * sizeof(float3), cudaMemcpyHostToDevice);
    return (e == cudaSuccess) ? 0 : -(int)e;
}

// ── Stage 3 shared accessors (read by cut.cu) + the cut-buffer handover (set by cut_init) ─────
void*         recon_isect()              { return d_isect; }
float3*       recon_isect_world()        { return d_isectWorld; }
int*          recon_voxel_isect_offset() { return d_voxelIsectOffset; }
int*          recon_voxel_isect_count()  { return d_voxelIsectCount; }
int*          recon_voxel_corner()       { return d_voxelCorner; }   // 8 corner ids/voxel (cut-FP Eq5 clamp box)
int*          recon_tri()                { return d_tri; }
unsigned int* recon_tri_counter()        { return d_triCounter; }
// Surface rendering: the reconstructed soft-tissue surface as a non-indexed soup (3 world verts/tri)
// + per-vertex aux (.w = bit30 wall flag) so the surface consumers can identify cut-wall triangles.
// Live index count comes from the existing recon_get_counts (NOT duplicated here).
float3* recon_expand_pos()   { return d_expandPos; }
float4* recon_expand_aux()   { return d_expandAux; }
float3* recon_expand_nrm()   { return d_expandNrm; }   // Surface rendering: per-vertex outward normal (winding guard)

void recon_set_cut_buffers(unsigned int* voxelCutMask, unsigned int* voxelOccupied, void* conn4096,
                           float4* cutFP, float3* cutFPNormalF, int* cutFPNormal)
{
    d_cut_voxelCutMask = voxelCutMask;
    d_cut_voxelOccupied = voxelOccupied;
    d_cut_conn4096 = (Conn4096Gpu*)conn4096;
    d_cut_cutFP = cutFP;
    d_cut_cutFPNormalF = cutFPNormalF;
    d_cut_cutFPNormal = cutFPNormal;
    g_cutActive = (voxelCutMask != nullptr) ? 1 : 0;
}

int recon_qef_iters() { return g_desc.qefIters; }

// [DEBUG-CUT] cross-TU hook (v4.1 P3, review R2-m4): cut_get_debug passes cut.cu's d_dbg; the scan
// runs here because d_tri / d_voxelExternalFP / FetchPos are private to dc_recon.
int recon_debug_scan_tris(unsigned int* dbgDev, float voxelL)
{
    if (!g_ready || !dbgDev) return -1;
    if (!g_cutActive || d_cut_cutFP == nullptr) return -2;   // only meaningful with the cut path live
    int tcap = g_desc.triCapacity;
    k_DebugScanTris<<<G(tcap), 64>>>((unsigned int)tcap, d_tri, d_triCounter,
                                     d_voxelExternalFP, d_cut_cutFP, voxelL, dbgDev);
    cudaError_t e = cudaGetLastError();
    return (e == cudaSuccess) ? 0 : -(int)e;
}

// Re-mesh from the (deformed) cornerPos. recon_build_static reads d_cornerPos for both the isect
// re-projection and the QEF, so the surface follows the physics. cudaDeviceSynchronize inside.
int recon_rebuild()
{
    if (!g_ready) return -1;
    cudaError_t e = recon_build_static();
    return (e == cudaSuccess) ? 0 : -(int)e;
}

void recon_shutdown()
{
    cudaFree(d_cornerPos);        d_cornerPos = nullptr;
    cudaFree(d_voxelCorner);      d_voxelCorner = nullptr;
    cudaFree(d_voxelIsectOffset); d_voxelIsectOffset = nullptr;
    cudaFree(d_voxelIsectCount);  d_voxelIsectCount = nullptr;
    cudaFree(d_isect);            d_isect = nullptr;
    cudaFree(d_surfaceEdges);     d_surfaceEdges = nullptr;
    cudaFree(d_isectWorld);       d_isectWorld = nullptr;
    cudaFree(d_voxelExternalFP);  d_voxelExternalFP = nullptr;
    cudaFree(d_tri);              d_tri = nullptr;
    cudaFree(d_triCounter);       d_triCounter = nullptr;
    cudaFree(d_extNormalFixed);   d_extNormalFixed = nullptr;
    cudaFree(d_extNormalF);       d_extNormalF = nullptr;
    cudaFree(d_expandPos);        d_expandPos = nullptr;
    cudaFree(d_expandNrm);        d_expandNrm = nullptr;
    cudaFree(d_voxelExternalFPRest); d_voxelExternalFPRest = nullptr;
    cudaFree(d_cornerPosRest);       d_cornerPosRest = nullptr;
    cudaFree(d_expandAux);           d_expandAux = nullptr;
    // Cut buffers are owned by cut.cu  - just drop the cached pointers (do not free here).
    d_cut_voxelCutMask = nullptr; d_cut_voxelOccupied = nullptr; d_cut_conn4096 = nullptr;
    d_cut_cutFP = nullptr; d_cut_cutFPNormalF = nullptr; d_cut_cutFPNormal = nullptr; g_cutActive = 0;
    g_ready = false;
}
