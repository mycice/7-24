// cut.cu  - native CUDA port of the Stage 3 cutting kernels (Cutting.compute).
//
// CUDA implementation of the Stage 3 cutting kernels.
// Dispatch order mirrors CutDetector.cs (DetectCut->SeverLinks) and DualContouring.cs (the cut-FP chain).
// Paper SS2.1.1 (Fig 2.4/2.5 cut surface) + SS2.1.3 (Fig 2.8 / Eq6-8 Moller-Trumbore).

#include "cut.cuh"
#include "cut.h"
#include "dc_recon.h"     // recon_* getters (cornerPos/isect/.../tri) + recon_set_cut_buffers
#include "physics.h"      // physics_* getters (particleRot/nbrIdx/bendPairs)
#include "common.cuh"     // IsectGpu, VoxId
#include <stdlib.h>
#include <string.h>

// Bending pair (32B)  - mirrors physics.cu BendPairGpu; SeverLinks writes alive=0.
struct BendPairGpuC { int i, j, k; float theta0; int alive, _p0, _p1, _p2; };
static_assert(sizeof(BendPairGpuC) == 32, "BendPairGpuC must be 32 bytes");

// Tool geometry passed by value to k_DetectCut (built per cut_detect from the host CutToolDesc).
struct ToolGpu
{
    float3 t1v0, t1v1, t1v2, t2v0, t2v1, t2v2;
    float3 ncut, aabbMin, aabbMax;
    float  d;
    int    valid;
};

// ── Stage 6 detection (paper SS2.1.3 + SS2.1.2; task_plan.md Stage 6) ─────────────────────────
// TWO exact world-space cadences replace the Stage-4 material-map (whose inverse tore at real folds).
// OPERATOR-SPLITTING NOTE (re-adjudication wf_dbf90148): the blade is frozen at frame granularity
// during the physics phase, so the cut sheet's PLACEMENT carries a splitting displacement bounded by
// one frame of relative blade-tissue motion (no holes, no separation failure; shrinks with frame
// rate  - the same splitting error accepted engine-wide).
//   rod phase  (cut_detect, tissue frozen): Moller-Trumbore of the DEFORMED edge vs the swept world
//              quad  - the paper's SS2.1.3 detection, proven in zero-g.
//   tick phase (cut_ribbon_tick, rod frozen): CONTINUOUS COLLISION of the MOVING deformed edge
//              against the static blade segment  - the exact space-time crossing (coplanarity is
//              QUADRATIC in t for a static blade), so falling/oscillating tissue cannot tunnel and
//              there is no global inverse map to tear. SS2.1.2 recording (alongRest) unchanged.

// ── Cut-owned device buffers ─────────────────────────────────────────────────────────────────
static GridEdgeGpu2*  d_gridEdges      = nullptr;  // [gridEdgeCount]
static float3*        d_prevCorner     = nullptr;  // [cornerCount] corner positions at the previous tick (CCD)
static float3*        d_cutRestCorner  = nullptr;  // [cornerCount] optional Stage 3 aligned rest positions
static Conn4096Gpu*   d_conn4096       = nullptr;  // [4096]
static unsigned int*  d_voxelOccupied  = nullptr;  // [voxelCount] (1/0)
static unsigned int*  d_voxelCutMask   = nullptr;  // [voxelCount] (cumulative, 12-bit)
static CutPointGpu*   d_cutPoint       = nullptr;  // [cutPointCapacity]
static unsigned int*  d_cutPointCounter= nullptr;  // [1] (cumulative)
static float4*        d_cutFP          = nullptr;  // [8*voxelCount]
static float4*        d_prevCutFP      = nullptr;  // [8*voxelCount] (EMA history)
static int*           d_cornerInside   = nullptr;  // [cornerCount] 1 = phi<0 (Stage 7 air-edge gate)
static int*           d_cutFPAccumPos  = nullptr;  // [3*8*voxelCount]
static int*           d_cutFPAccumCnt  = nullptr;  // [8*voxelCount]
static unsigned int*  d_voxelFPCount   = nullptr;  // [voxelCount]
static int*           d_cutFPNormal    = nullptr;  // [3*8*voxelCount]
static float3*        d_cutFPNormalF   = nullptr;  // [8*voxelCount]
// v4.1 buffers (plan review wf_b08c3fc2):
static unsigned int*  d_prevVoxelMask  = nullptr;  // [voxelCount] mask snapshot at the last cut_fp_chain (P1a EMA epoch)
static int*           d_cutNrmAccum    = nullptr;  // [3*8*voxelCount] per-slot world cut-normal accumulator (P2 kerf)
static unsigned int*  d_dbg            = nullptr;  // [16] [DEBUG-CUT] counters (P3): slots 0-14 zeroed each cut_fp_chain, 15 cumulative.
__device__ CutEventMeta g_cutEventMeta;
__device__ CutEventSummary g_cutEventSummary;
// d_dbg slot map: 0=emaPairs 1=jump>L 2=jump>3L 3=maxDisp(f-as-u) 4=slotNew 5=epochInval 6=dangleRefs
//                 7=giantWall 8=giantMixed 9=giantSkin 10=maxEdgeLen(f-as-u) 11=rim<3 12=pinch1
//                 13=all6severed 14=frozenThisFrame 15=tearMarks(CUMULATIVE  - incremented during
//                 LCS_Step's ribbon ticks, so the per-chain memset covers slots 0-14 ONLY; zeroing
//                 15 mid-frame would erase every tick-phase tear before readback, wf_c416ae77 R2-m3)

// ── Cached shared pointers (from dc_recon + physics; stable after their init) ─────────────────
static float3*       c_cornerPos       = nullptr;
static IsectGpu*     c_isect           = nullptr;
static float3*       c_isectWorld      = nullptr;
static int*          c_voxelIsectOffset= nullptr;
static int*          c_voxelIsectCount = nullptr;
static int*          c_voxelCorner     = nullptr;   // 8 corner ids/voxel (cut-FP Eq5 box-SDF clamp)
static int*          c_tri             = nullptr;
static unsigned int* c_triCounter      = nullptr;
static RotGpu*       c_particleRot     = nullptr;
static int*          c_nbrIdx          = nullptr;
static BendPairGpuC* c_bendPairs       = nullptr;
static int*          c_active          = nullptr;   // physics active flags (v4.1 P1b: freeze + geometry cull)

static CutInitDesc g_cdesc = {};
static int3        g_cdims = {};
static CutToolDesc g_tool  = {};
static bool        g_cut_ready = false;
static bool        g_cutRestMetricEnabled = false;
static float       g_cutVoxelL = 0.f;
static float3      g_cutVoxelSize = { 0.f, 0.f, 0.f };
static float       g_cutVoxelMaxL = 0.f;
static float       g_tearStretchRatio = 0.f;
static bool        g_gravityStabilizationEnabled = false;
static float3      g_gravityDirection = { 0.f, -1.f, 0.f };
static float       g_gravitySafeGap = 0.f;
static float       g_gravityAlignmentExponent = 1.f;

// ── Stage-6 host state ────────────────────────────────────────────────────────────────────────
static bool   g_toolSet        = false;          // at least one valid cutting sweep received
static bool   g_havePrevCorner = false;          // d_prevCorner holds the previous tick's positions
static float3 g_emitNcut  = { 0.f, 1.f, 0.f };   // last VALID world cut normal (for the ±D/2 side)
static float  g_tearLen2  = 0.f;                 // (tau*L)^2, 0 = tear law disabled (v5.1 D10)

static int Gc(int n) { int g = (n + 63) / 64; return g < 1 ? 1 : g; }
static float3 mk3(const float* a) { return make_float3(a[0], a[1], a[2]); }

__device__ __forceinline__ float BoxSdfAnisotropic(float3 p, float3 c, float3 size)
{
    float3 q = make_float3(fabsf(p.x - c.x) - 0.5f * size.x,
                           fabsf(p.y - c.y) - 0.5f * size.y,
                           fabsf(p.z - c.z) - 0.5f * size.z);
    float3 outside = make_float3(fmaxf(q.x, 0.f), fmaxf(q.y, 0.f), fmaxf(q.z, 0.f));
    float outsideLength = sqrtf(dot3(outside, outside));
    return outsideLength + fminf(fmaxf(q.x, fmaxf(q.y, q.z)), 0.f);
}

// ─────────────────────────────────────────────────────────────────────────────────────────────
// EmitCutPoint (device)  - record one cut point in the owner's MATERIAL frame (paper SS2.1.2).
//
// Per SS2.1.2 the cut-point local coordinate is recorded "relative to the two vertices of the edge
// ... based on the condition that the edge has NOT undergone any deformation." So we store the
// UNDEFORMED (rest) offset from the owner along the edge, `alongRest = t_owner * L * axisDir`
// (a scalar t along the particle's coordinate axis), PLUS the +-D/2 cut-wall gap expressed in the
// owner's material frame (R^T * side*D/2*n_cut). Reconstruction (AccumulateCutFP) applies the CURRENT
// rotation: world = cornerPos[owner] + R * localOffset  - "the new cut point is computed using the
// corresponding coordinate axes" (SS2.1.2). The +-D/2 gap is along the world cut normal AT EMISSION
// TIME and thereafter tracks the owner's frame (R_recon*R_emit^T*ncut)  - intended SS2.1.2 behavior;
// note mid-frame ribbon ticks record with the previous frame's R (sub-voxel cosmetic skew, D<<L).
// ─────────────────────────────────────────────────────────────────────────────────────────────
__device__ __forceinline__ void EmitCutPoint(float3 P_hit, float3 alongRest, int p, int edgeId,
                                             float3 ncut, float D,
                                             const float3* cornerPos, const RotGpu* particleRot,
                                             CutPointGpu* cutPoint, unsigned int* cutPointCounter,
                                             int cutPointCapacity)
{
    float3 pPos = cornerPos[p];
    float  side = signf(dot3(pPos - P_hit, ncut));        // which side of the WORLD cut plane the owner is on
    if (side == 0.0f) side = 1.0f;                        // on the plane -> +n_cut side

    unsigned int slot = atomicAdd(&cutPointCounter[0], 1u);
    if (slot >= (unsigned int)cutPointCapacity) return;   // capacity guard

    RotGpu R = particleRot[p];
    CutPointGpu cp;
    // UNDEFORMED-edge local coord + cut-gap in owner material frame (SS2.1.2). See header note.
    cp.localOffset   = alongRest + mulRtv(R, (side * (D * 0.5f)) * ncut);
    cp.ownerParticle = p;
    cp.edgeId        = edgeId;
    // v4.1 P2: material-frame OUTWARD cut normal (side-signed, unit). All cut points of one
    // (voxel,component) slot share the owner's side, so the per-slot world sum is coherent for a
    // single stroke; multi-stroke cancellation is guarded at the projection site.
    cp.matNrm        = mulRtv(R, side * ncut);
    cutPoint[slot]   = cp;
}

// ─────────────────────────────────────────────────────────────────────────────────────────────
// MarkAndEmit  - shared mark+record tail for BOTH detection cadences: set the cut bit in all 4
// OCCUPIED sharing voxels (RC2); on the FIRST cut of the edge, record the two SS2.1.2 cut points
// on the undeformed edge (alongRest = t*L*axis / (t-1)*L*axis, reconstructed via the current
// particle axes by AccumulateCutFP).
// ─────────────────────────────────────────────────────────────────────────────────────────────
__device__ __forceinline__ void MarkAndEmit(int e, int axis, int3 baseC, int3 axisDir, int idA, int idB,
                                            float t_ray, float3 P_hit, float3 ncut, float toolD,
                                            float voxelL, int3 dims, const float3* cornerPos,
                                            const float3* cutRestCorner,
                                            unsigned int* voxelCutMask, const unsigned int* voxelOccupied,
                                            CutPointGpu* cutPoint, unsigned int* cutPointCounter,
                                            int cutPointCapacity, const RotGpu* particleRot, bool recordToolEvent)
{
    bool firstCut = false, sawAnyVox = false, edgeOccupied = false;
    for (int s = 0; s < 4; s++)
    {
        int3 vc = baseC + ES_VoxelOffset(axis, s);
        if (!VoxInBounds(vc, dims)) continue;
        int vid = VoxId(vc, dims);
        if (voxelOccupied[vid] == 0u) continue;                 // RC2: no cut bits in empty voxels
        edgeOccupied = true;
        unsigned int cutBit = (unsigned int)(1 << ES_LocalEdge(axis, s));
        unsigned int old = atomicOr(&voxelCutMask[vid], cutBit);
        if (!sawAnyVox) { sawAnyVox = true; firstCut = (old & cutBit) == 0u; }
    }
    if (firstCut && edgeOccupied)
    {
        float3 restEdge;
        if (cutRestCorner != nullptr)
            restEdge = cutRestCorner[idB] - cutRestCorner[idA];
        else
        {
            float3 axisF = make_float3((float)axisDir.x, (float)axisDir.y, (float)axisDir.z);
            restEdge = voxelL * axisF;
        }
        float3 alongA = t_ray * restEdge;
        float3 alongB = (t_ray - 1.0f) * restEdge;
        EmitCutPoint(P_hit, alongA, idA, e, ncut, toolD, cornerPos, particleRot, cutPoint, cutPointCounter, cutPointCapacity);
        EmitCutPoint(P_hit, alongB, idB, e, ncut, toolD, cornerPos, particleRot, cutPoint, cutPointCounter, cutPointCapacity);
        unsigned int previousSequence = recordToolEvent && g_cutEventMeta.valid != 0
            ? atomicExch(&g_cutEventSummary.sequence, g_cutEventMeta.sequence)
            : g_cutEventMeta.sequence;
        if (recordToolEvent && g_cutEventMeta.valid != 0 && previousSequence != g_cutEventMeta.sequence)
        {
            // This is deliberately after the first-cut gate: repeated sweeps and air motion cannot
            // manufacture events. The latest edge is a compact representative for this event batch.
            g_cutEventSummary.start[0] = g_cutEventMeta.start[0]; g_cutEventSummary.start[1] = g_cutEventMeta.start[1]; g_cutEventSummary.start[2] = g_cutEventMeta.start[2];
            g_cutEventSummary.end[0] = g_cutEventMeta.end[0]; g_cutEventSummary.end[1] = g_cutEventMeta.end[1]; g_cutEventSummary.end[2] = g_cutEventMeta.end[2];
            g_cutEventSummary.normal[0] = g_cutEventMeta.normal[0]; g_cutEventSummary.normal[1] = g_cutEventMeta.normal[1]; g_cutEventSummary.normal[2] = g_cutEventMeta.normal[2];
            g_cutEventSummary.hitPoint[0] = P_hit.x; g_cutEventSummary.hitPoint[1] = P_hit.y; g_cutEventSummary.hitPoint[2] = P_hit.z;
            g_cutEventSummary.radius = g_cutEventMeta.radius;
            g_cutEventSummary.timestamp = g_cutEventMeta.timestamp;
            atomicAdd(&g_cutEventSummary.eventCount, 1u);
            g_cutEventSummary.rawCutPoints = cutPointCounter[0];
            g_cutEventSummary.valid = 1u;
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────────────────────
// DetectCutQuad  - ROD-SUBSTEP phase (tissue frozen; called via cut_detect). Paper SS2.1.3 verbatim:
// Moller-Trumbore (Eq6-8) of the swept cutting plane (2 world triangles, prev+cur tool frames)
// against the CURRENT DEFORMED grid edges. This is the architecture proven by the perfect zero-g
// cut. On hit t: alongRest = t*L*axis is the SS2.1.2 undeformed-edge record.
// ─────────────────────────────────────────────────────────────────────────────────────────────
__global__ void k_DetectCutQuad(int gridEdgeCount, ToolGpu tool, float voxelL, int3 dims,
                                int cutPointCapacity, const int* cornerInside,
                                const float3* cornerPos, const GridEdgeGpu2* gridEdges,
                                const float3* cutRestCorner,
                                unsigned int* voxelCutMask, const unsigned int* voxelOccupied,
                                CutPointGpu* cutPoint, unsigned int* cutPointCounter,
                                const RotGpu* particleRot)
{
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= gridEdgeCount) return;
    if (tool.valid == 0) return;

    GridEdgeGpu2 ge = gridEdges[e];
    int axis = ge.axis;
    int3 baseC = make_int3(ge.bcX, ge.bcY, ge.bcZ);
    int3 axisDir = make_int3(axis == 0 ? 1 : 0, axis == 1 ? 1 : 0, axis == 2 ? 1 : 0);

    int idA = CorIdCuda(baseC, dims);
    int idB = CorIdCuda(baseC + axisDir, dims);

    // AIR-EDGE RULE v3 (plan v3.1, diagnosis wf_2b56575e unanimous + plan review wf_038eabf9):
    // paper-literal  - MARK and EMIT for EVERY edge the plane crosses (SS2.1.1 "the cut edges will
    // be calculated twice", Table 3 no inside qualifier). Pure-air component FPs are bounded by the
    // per-component Eq5/D5 box + the D4 variant-(c) skin-side mean-plane projection in
    // k_ComputeComponentFP, which kills the A1 durian without starving the rim wall (the v2
    // emission gate starved pure-air rim comps => the row-of-pockets slit).

    // SS2.1.3 / Eq6: ray = the CURRENT DEFORMED grid edge, tris = the world swept quad.
    float3 O    = cornerPos[idA];
    float3 Dvec = cornerPos[idB] - O;

    // cull vs the tool AABB (pre-built by MakeToolDesc; inflated by L)
    float pad = voxelL;
    float3 eMin = make_float3(fminf(O.x, O.x + Dvec.x), fminf(O.y, O.y + Dvec.y), fminf(O.z, O.z + Dvec.z));
    float3 eMax = make_float3(fmaxf(O.x, O.x + Dvec.x), fmaxf(O.y, O.y + Dvec.y), fmaxf(O.z, O.z + Dvec.z));
    if (eMax.x < tool.aabbMin.x - pad || eMin.x > tool.aabbMax.x + pad ||
        eMax.y < tool.aabbMin.y - pad || eMin.y > tool.aabbMax.y + pad ||
        eMax.z < tool.aabbMin.z - pad || eMin.z > tool.aabbMax.z + pad) return;

    float t1, u1, v1, t2, u2, v2;
    bool hit1 = MollerTrumbore(O, Dvec, tool.t1v0, tool.t1v1, tool.t1v2, t1, u1, v1);
    bool hit2 = MollerTrumbore(O, Dvec, tool.t2v0, tool.t2v1, tool.t2v2, t2, u2, v2);
    if (!hit1 && !hit2) return;

    float t_ray = (hit1 && hit2) ? fminf(t1, t2) : (hit1 ? t1 : t2);
    t_ray = fminf(fmaxf(t_ray, 0.f), 1.f);
    float3 P_hit = O + t_ray * Dvec;          // world hit ON the deformed edge (no offset)

    MarkAndEmit(e, axis, baseC, axisDir, idA, idB, t_ray, P_hit, tool.ncut, tool.d, voxelL, dims,
                cornerPos, cutRestCorner, voxelCutMask, voxelOccupied, cutPoint, cutPointCounter, cutPointCapacity,
                particleRot, true);
}

// ─────────────────────────────────────────────────────────────────────────────────────────────
// DetectCutCCD  - PHYSICS-TICK phase (rod frozen; called via cut_ribbon_tick after every accepted
// RK45 substep). Continuous collision of the MOVING deformed edge against the STATIC blade segment
// [S,E]: with the blade static and edge endpoints linear in tick-time t, the coplanarity function
//   f(t) = ((a(t)-S) x (b(t)-S)) . (E-S)
// is QUADRATIC in t  - solved exactly, so falling/oscillating tissue cannot tunnel between ticks
// (the Stage-4 D2-blocker fix, now WITHOUT a global inverse map that tears at folds). At the root
// t*, validate the segment-segment intersection params (u along edge, v along blade) in the common
// plane. SS2.1.2 recording (alongRest = u*L*axis) unchanged.
// ─────────────────────────────────────────────────────────────────────────────────────────────
__global__ void k_DetectCutCCD(int gridEdgeCount, float3 S, float3 E, float3 emitNcut, float toolD,
                               float voxelL, int3 dims, int cutPointCapacity,
                               const float3* prevPos, const float3* curPos,
                               const GridEdgeGpu2* gridEdges, const float3* cutRestCorner,
                               const int* nbrIdx, const int* cornerInside,
                               unsigned int* voxelCutMask, const unsigned int* voxelOccupied,
                               CutPointGpu* cutPoint, unsigned int* cutPointCounter,
                               const RotGpu* particleRot)
{
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= gridEdgeCount) return;

    GridEdgeGpu2 ge = gridEdges[e];
    int axis = ge.axis;
    int3 baseC = make_int3(ge.bcX, ge.bcY, ge.bcZ);
    int3 axisDir = make_int3(axis == 0 ? 1 : 0, axis == 1 ? 1 : 0, axis == 2 ? 1 : 0);

    int idA = CorIdCuda(baseC, dims);
    int idB = CorIdCuda(baseC + axisDir, dims);

    // Already-severed edges cannot be cut again (wf_65901910 B2 minor): skipping them stops the CCD
    // from endlessly re-marking a fallen piece streaming past the blade (cut points/tris growth).
    if (nbrIdx[6 * idA + 2 * axis + 1] < 0) return;
    // AIR-EDGE RULE v3  - paper-literal mark AND emit (see k_DetectCutQuad).

    float3 a0 = prevPos[idA], a1 = curPos[idA];
    float3 b0 = prevPos[idB], b1 = curPos[idB];

    // cull: swept-edge AABB vs blade-segment AABB (+L pad)
    float pad = voxelL;
    float3 sMin = make_float3(fminf(S.x, E.x) - pad, fminf(S.y, E.y) - pad, fminf(S.z, E.z) - pad);
    float3 sMax = make_float3(fmaxf(S.x, E.x) + pad, fmaxf(S.y, E.y) + pad, fmaxf(S.z, E.z) + pad);
    float3 eMin = make_float3(fminf(fminf(a0.x, a1.x), fminf(b0.x, b1.x)),
                              fminf(fminf(a0.y, a1.y), fminf(b0.y, b1.y)),
                              fminf(fminf(a0.z, a1.z), fminf(b0.z, b1.z)));
    float3 eMax = make_float3(fmaxf(fmaxf(a0.x, a1.x), fmaxf(b0.x, b1.x)),
                              fmaxf(fmaxf(a0.y, a1.y), fmaxf(b0.y, b1.y)),
                              fmaxf(fmaxf(a0.z, a1.z), fmaxf(b0.z, b1.z)));
    if (eMax.x < sMin.x || eMin.x > sMax.x || eMax.y < sMin.y || eMin.y > sMax.y ||
        eMax.z < sMin.z || eMin.z > sMax.z) return;

    // f(t) = ((P0 + t*Pv) x (Q0 + t*Qv)) . D  with P=a(t)-S, Q=b(t)-S, D=E-S   - quadratic in t.
    float3 D  = E - S;
    float3 P0 = a0 - S, Pv = a1 - a0;
    float3 Q0 = b0 - S, Qv = b1 - b0;
    float  c0 = dot3(cross3(P0, Q0), D);
    float  c1 = dot3(cross3(P0, Qv) + cross3(Pv, Q0), D);
    float  c2 = dot3(cross3(Pv, Qv), D);

    // roots of c2 t^2 + c1 t + c0 = 0 in [0,1] (stable quadratic; degenerate -> linear).
    float roots[2]; int nr = 0;
    if (fabsf(c2) < 1e-14f)
    {
        if (fabsf(c1) > 1e-14f) { float r = -c0 / c1; if (r >= 0.f && r <= 1.f) roots[nr++] = r; }
        // c1,c2 ~ 0: f constant. Static edges are the rod phase's job. A PERSISTENTLY-COPLANAR edge
        // sliding through the blade within its own plane for a whole tick is skipped  - a measure-zero
        // configuration under generic motion (any transverse perturbation breaks coplanarity next
        // tick); documented limitation (review S1-m2).
    }
    else
    {
        float disc = c1 * c1 - 4.f * c2 * c0;
        if (disc >= 0.f)
        {
            float sq = sqrtf(disc);
            float q  = -0.5f * (c1 + (c1 >= 0.f ? sq : -sq));
            float r0 = q / c2, r1 = (fabsf(q) > 1e-20f) ? c0 / q : r0;
            if (r0 > r1) { float tmp = r0; r0 = r1; r1 = tmp; }
            if (r0 >= 0.f && r0 <= 1.f) roots[nr++] = r0;
            if (r1 >= 0.f && r1 <= 1.f && r1 != r0) roots[nr++] = r1;
        }
    }

    for (int ri = 0; ri < nr; ri++)
    {
        float tc = roots[ri];
        float3 a = a0 + tc * Pv;                 // edge endpoints at crossing time
        float3 b = b0 + tc * Qv;
        float3 e1 = b - a;
        float3 n2 = cross3(e1, D);
        float  n2len2 = dot3(n2, n2);
        if (n2len2 < 1e-16f) continue;           // edge parallel to blade at t*
        float3 w = S - a;
        float u = dot3(cross3(w, D),  n2) / n2len2;   // param along the edge a->b
        float v = dot3(cross3(w, e1), n2) / n2len2;   // param along the blade S->E
        const float EPS = 2e-2f;
        if (u < -EPS || u > 1.f + EPS || v < -EPS || v > 1.f + EPS) continue;

        float t_ray = fminf(fmaxf(u, 0.f), 1.f);
        // world hit for the ±D/2 side sign: the point on the CURRENT edge geometry.
        float3 P_hit = a1 + t_ray * (b1 - a1);
        // Emit normal (review S1-m1/S3-m3): the CCD cut wall's normal = blade direction x relative
        // tissue motion (the plane actually being cut)  - correct even before any VALID rod sweep;
        // degenerate motion falls back to the last valid sweep normal.
        float3 relV = 0.5f * (Pv + Qv);
        float3 nHit = cross3(D, relV);
        float  nLen = length3(nHit);
        float3 nEmit = (nLen > 1e-8f) ? nHit / nLen : emitNcut;
        MarkAndEmit(e, axis, baseC, axisDir, idA, idB, t_ray, P_hit, nEmit, toolD, voxelL, dims,
                    curPos, cutRestCorner, voxelCutMask, voxelOccupied, cutPoint, cutPointCounter, cutPointCapacity,
                    particleRot, true);
        return;                                   // first valid crossing cuts the edge
    }
}

// ─────────────────────────────────────────────────────────────────────────────────────────────
// TearOverstretch (v5.1, deviation D10 - plan review wf_c416ae77 R1 approve / R2 amendments) -
// one thread per grid edge, launched per ribbon tick AFTER k_DetectCutCCD and BEFORE k_SeverLinks.
// THE 藕断丝连 FIX: rim bridge edges the swept ribbon never crossed have NO fracture law  - the
// falling piece (restrained negligibly by 2-3 springs) drags their voxel chains 10-50L into the
// strand/fan artifacts (log: giantWall 0->45, maxEdge 2.15L->51L growing with fall). Tissue beyond
// its elastic limit TEARS: an intact active-active edge stretched past tau*L is marked+emitted via
// the NORMAL MarkAndEmit path (tear plane  - the edge, hit at the midpoint), so conn4096 splits,
// FPs recover to the lips, the torn face renders a real wall, and the same tick's k_SeverLinks
// severs the spring+bends. tau default 4.0 (R2-m1: must clear the max legit gravity strain with
// margin; bridges blow through ANY tau within a frame or two at terminal velocity). Gated on
// g_toolSet (host) so an uncut liver can never tear.
// ─────────────────────────────────────────────────────────────────────────────────────────────
__global__ void k_TearOverstretch(int gridEdgeCount, float tearStretchRatio, int3 dims, float voxelL,
                                  float toolD, const GridEdgeGpu2* gridEdges, const int* nbrIdx,
                                  const int* active, const float3* cornerPos,
                                  const float3* cutRestCorner,
                                  unsigned int* voxelCutMask, const unsigned int* voxelOccupied,
                                  CutPointGpu* cutPoint, unsigned int* cutPointCounter,
                                  int cutPointCapacity, const RotGpu* particleRot, unsigned int* dbg)
{
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= gridEdgeCount) return;

    GridEdgeGpu2 ge = gridEdges[e];
    int axis = ge.axis;
    int3 baseC = make_int3(ge.bcX, ge.bcY, ge.bcZ);
    int3 axisDir = make_int3(axis == 0 ? 1 : 0, axis == 1 ? 1 : 0, axis == 2 ? 1 : 0);
    int idA = CorIdCuda(baseC, dims);
    int idB = CorIdCuda(baseC + axisDir, dims);

    if (nbrIdx[6 * idA + 2 * axis + 1] < 0) return;            // already severed
    if (active[idA] == 0 || active[idB] == 0) return;          // frozen/inactive corners carry no spring
    // Already mask-marked (e.g. by k_DetectCutCCD earlier in this SAME tick, before k_SeverLinks
    // ran): the edge is already a cut  - MarkAndEmit would be a pure no-op (firstCut=false) and the
    // tears counter would over-count (final review W1-m2/W2-m2). Behavior-identical early-out.
    if (EdgeIsCut(axis, baseC, dims, voxelCutMask)) return;

    // R1-m3: an edge with NO occupied stencil voxel can never receive a mask bit, so k_SeverLinks
    // could never sever it and the tear would refire every tick  - skip it entirely.
    bool anyOcc = false;
    for (int s = 0; s < 4; s++)
    {
        int3 vc = baseC + ES_VoxelOffset(axis, s);
        if (VoxInBounds(vc, dims) && voxelOccupied[VoxId(vc, dims)] != 0u) { anyOcc = true; break; }
    }
    if (!anyOcc) return;

    float3 A = cornerPos[idA], B = cornerPos[idB];
    float3 d = B - A;
    float  len2 = dot3(d, d);
    float3 restEdge = cutRestCorner != nullptr
        ? cutRestCorner[idB] - cutRestCorner[idA]
        : voxelL * make_float3((float)axisDir.x, (float)axisDir.y, (float)axisDir.z);
    float restLen2 = dot3(restEdge, restEdge);
    float tearLen2 = tearStretchRatio * tearStretchRatio * restLen2;
    if (restLen2 <= 1e-12f || len2 <= tearLen2) return;

    float3 ncut  = d / sqrtf(len2);                            // tear plane  - the overstretched edge
    float3 P_hit = A + 0.5f * d;                               // midpoint: side=-1 for A, +1 for B (R1 verified)
    if (dbg) atomicAdd(&dbg[15], 1u);                          // CUMULATIVE tear counter (see slot map)
    MarkAndEmit(e, axis, baseC, axisDir, idA, idB, 0.5f, P_hit, ncut, toolD, voxelL, dims,
                cornerPos, cutRestCorner, voxelCutMask, voxelOccupied, cutPoint, cutPointCounter, cutPointCapacity,
                particleRot, false);
}

// SeverLinks  - one thread per grid edge. For cut edges: sever structural (NbrIdx=-1 both ways) +
// bending pairs whose arm crosses the edge (alive=0). Paper SS2.1.2 line 367.
// ─────────────────────────────────────────────────────────────────────────────────────────────
__global__ void k_SeverLinks(int gridEdgeCount, int3 dims, const GridEdgeGpu2* gridEdges,
                             const unsigned int* voxelCutMask, int* nbrIdx, BendPairGpuC* bendPairs)
{
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= gridEdgeCount) return;

    GridEdgeGpu2 ge = gridEdges[e];
    int axis = ge.axis;
    int3 baseC = make_int3(ge.bcX, ge.bcY, ge.bcZ);
    if (!EdgeValid(axis, baseC, dims)) return;
    if (!EdgeIsCut(axis, baseC, dims, voxelCutMask)) return;

    int3 axisDir = make_int3(axis == 0 ? 1 : 0, axis == 1 ? 1 : 0, axis == 2 ? 1 : 0);
    int idA = CorIdCuda(baseC, dims);
    int idB = CorIdCuda(baseC + axisDir, dims);
    int sPlus  = 2 * axis + 1;    // idA's slot toward idB
    int sMinus = 2 * axis;        // idB's slot toward idA

    nbrIdx[6 * idA + sPlus]  = -1;
    nbrIdx[6 * idB + sMinus] = -1;

    for (int p = 0; p < 15; p++)
    {
        int ia = 15 * idA + p; BendPairGpuC a = bendPairs[ia];
        if (a.alive != 0 && (a.j == idB || a.k == idB)) { a.alive = 0; bendPairs[ia] = a; }
        int ib = 15 * idB + p; BendPairGpuC b = bendPairs[ib];
        if (b.alive != 0 && (b.j == idA || b.k == idA)) { b.alive = 0; bendPairs[ib] = b; }
    }
}

// ── DeactivateSevered (v4.1 P1b)  - per corner: a particle whose 6 structural links are ALL severed
// is an isolated debris crumb; it integrates under gravity forever (the "dirty specks" runaways).
// Freeze it (active=0  - physics k_ComputeSlope/IntegrateY5 already honor active) and let the
// k_AccumulateCutFP inactive-owner skip drop its FP/geometry. No false positives: lattice-bound
// corners always keep >=3 in-lattice slots >=0 (BackgroundGrid builds nbrIdx for ALL in-lattice
// pairs), so all-6-severed can only be produced by k_SeverLinks.
// dbg[13]=all-6-severed count (per frame, regardless of active), dbg[14]=newly frozen this frame.
__global__ void k_DeactivateSevered(int cornerCount, const int* nbrIdx, int* active, unsigned int* dbg)
{
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= cornerCount) return;
    const int* nb = &nbrIdx[6 * p];
    if (nb[0] >= 0 || nb[1] >= 0 || nb[2] >= 0 || nb[3] >= 0 || nb[4] >= 0 || nb[5] >= 0) return;
    if (dbg) atomicAdd(&dbg[13], 1u);
    if (active[p] != 0)
    {
        active[p] = 0;
        if (dbg) atomicAdd(&dbg[14], 1u);
    }
}

// ── ClearCutFP  - per slot (8*VC): zero accum/cnt/CutFP.w/CutFPNormal (Stage-7 single FP) ──
__global__ void k_ClearCutFP(int voxelCount, int* cutFPAccumPos, int* cutFPAccumCnt, float4* cutFP,
                             int* cutFPNormal, unsigned int* voxelFPCount, int* cutNrmAccum)
{
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= 8 * voxelCount) return;
    cutFPAccumPos[3 * s + 0] = 0; cutFPAccumPos[3 * s + 1] = 0; cutFPAccumPos[3 * s + 2] = 0;
    cutFPAccumCnt[s] = 0;
    cutFP[s] = make_float4(0.f, 0.f, 0.f, 0.f);
    cutFPNormal[3 * s + 0] = 0; cutFPNormal[3 * s + 1] = 0; cutFPNormal[3 * s + 2] = 0;
    cutNrmAccum[3 * s + 0] = 0; cutNrmAccum[3 * s + 1] = 0; cutNrmAccum[3 * s + 2] = 0;   // v4.1 P2
    if (s < voxelCount) voxelFPCount[s] = 0u;
}

// ── LookupConnectivity  - per voxel: voxelFPCount = (cut & occupied) ? conn[mask].compCount : 0 ──
__global__ void k_LookupConnectivity(int voxelCount, const unsigned int* voxelCutMask,
                                     const unsigned int* voxelOccupied, const Conn4096Gpu* conn4096,
                                     unsigned int* voxelFPCount, unsigned int* dbg)
{
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= voxelCount) return;
    unsigned int mask = voxelCutMask[v] & 0xFFFu;
    if (mask == 0u || voxelOccupied[v] == 0u) { voxelFPCount[v] = 0u; return; }
    int compCount = conn4096[mask].compCount;
    voxelFPCount[v] = (unsigned int)compCount;
    // [DEBUG-CUT] pinch counter (P3): cut voxel whose corner graph is still ONE component  - the seam
    // pinches here (0-g roughness disambiguator).
    if (dbg && compCount == 1) atomicAdd(&dbg[12], 1u);
}

// ── AccumulateCutFP  - per cut point: scatter world pos into the owner-component slots (centroid sum) ──
__global__ void k_AccumulateCutFP(int cutPointCapacity,
                                  const unsigned int* cutPointCounter, const CutPointGpu* cutPoint,
                                  const RotGpu* particleRot, const float3* cornerPos, int3 dims,
                                  const GridEdgeGpu2* gridEdges, const unsigned int* voxelOccupied,
                                  const unsigned int* voxelCutMask, const Conn4096Gpu* conn4096,
                                  int* cutFPAccumPos, int* cutFPAccumCnt,
                                  const int* active, int* cutNrmAccum,
                                  int gravityStabilizationEnabled, float3 gravityDirection,
                                  float gravitySafeGap, float gravityAlignmentExponent)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    // OOB guard (diagnosis D1 minor): the emit counter increments BEFORE the capacity check, so
    // counter[0] can exceed capacity; without this clamp threads read d_cutPoint out of bounds.
    if (i >= (int)cutPointCounter[0] || i >= cutPointCapacity) return;

    CutPointGpu cp = cutPoint[i];
    // v4.1 P1b (review R1-m4/R2-m1): a cut point owned by a DEACTIVATED (fully-severed debris)
    // particle must stop seeding FPs. INTERIOR specks reach total==0 => w=0 => no wall/stitch tris
    // (geometry disappears); a SKIN-touching crumb keeps an iso-seeded FP (nIncl>0) => a frozen skin
    // patch with no wall remains  - the accepted R2-m1 limitation (the runaway integration itself is
    // stopped either way). Body-side walls are unaffected (each severed edge emits a separate
    // body-owned point).
    if (active[cp.ownerParticle] == 0) return;
    RotGpu R = particleRot[cp.ownerParticle];
    float3 world = cornerPos[cp.ownerParticle] + mulRv(R, cp.localOffset);   // re-rotate by current R
    float3 nrmW  = mulRv(R, cp.matNrm);                                      // v4.1 P2: world cut normal
    if (gravityStabilizationEnabled != 0 && gravitySafeGap > 0.f)
    {
        float alignment = fminf(fmaxf(fabsf(dot3(nrmW, gravityDirection)), 0.f), 1.f);
        float weight = powf(alignment, gravityAlignmentExponent);
        world = world + nrmW * (0.5f * gravitySafeGap * weight);
    }
    int3 ownerCoord = CornerCoordCuda(cp.ownerParticle, dims);

    GridEdgeGpu2 ge = gridEdges[cp.edgeId];
    int axis = ge.axis;
    int3 baseC = make_int3(ge.bcX, ge.bcY, ge.bcZ);

    for (int s = 0; s < 4; s++)
    {
        int3 vc = baseC + ES_VoxelOffset(axis, s);
        if (!VoxInBounds(vc, dims)) continue;
        int v = VoxId(vc, dims);
        if (voxelOccupied[v] == 0u) continue;
        unsigned int mask = voxelCutMask[v] & 0xFFFu;
        if (mask == 0u) continue;
        int lc = LocalCornerOf(ownerCoord, vc);
        if (lc < 0) continue;
        int comp = Conn4096Comp(conn4096[mask], lc);
        int slot = 8 * v + comp;
        atomicAdd(&cutFPAccumPos[3 * slot + 0], (int)rintf(world.x * POS_SCALE));
        atomicAdd(&cutFPAccumPos[3 * slot + 1], (int)rintf(world.y * POS_SCALE));
        atomicAdd(&cutFPAccumPos[3 * slot + 2], (int)rintf(world.z * POS_SCALE));
        atomicAdd(&cutFPAccumCnt[slot], 1);
        atomicAdd(&cutNrmAccum[3 * slot + 0], (int)rintf(nrmW.x * POS_SCALE));   // v4.1 P2
        atomicAdd(&cutNrmAccum[3 * slot + 1], (int)rintf(nrmW.y * POS_SCALE));
        atomicAdd(&cutNrmAccum[3 * slot + 2], (int)rintf(nrmW.z * POS_SCALE));
    }
}

// ── ComputeComponentFP  - per voxel (loops components). Stage 7 (paper-exact; audit wf_60781cea A2):
// ONE feature point per (voxel, connected component), seeded by ALL of the component's crossing
// points  - its ISOSURFACE INTERSECTIONS and its CUT POINTS together (SS2.1.1: "Each edge that
// contains a vertex will participate in the calculation, and the cut edges will be calculated
// twice"), refined by the Fig 2.2 particle-movement iteration over the component's iso
// intersections (Eq3/D1 gradient, Eq4 decaying step) and clamped by Eq5/D5. Skin quads (k_Stitch)
// and wall quads (k_BuildCutTriangles) SHARE this vertex, so the cut margin is watertight by
// construction. The paper has NO rim band and NO dual FP families  - Table 3's External/Internal
// feature points are POSITIONAL labels of this same single entity (Stage-5's dual-family split was
// a misreading and caused the skin-parallel-cut durian/shadow artifacts).
__global__ void k_ComputeComponentFP(int voxelCount, int3 dims, const unsigned int* voxelCutMask,
                                     const unsigned int* voxelOccupied, const Conn4096Gpu* conn4096,
                                     const int* voxelIsectOffset, const int* voxelIsectCount,
                                     const IsectGpu* isect, const float3* isectWorld,
                                     const int* cutFPAccumCnt, const int* cutFPAccumPos, float4* cutFP,
                                     const int* voxelCorner, const float3* cornerPos, float3 voxelSize,
                                     int qefIters, const int* cornerInside, const int* cutNrmAccum)
{
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= voxelCount) return;

    unsigned int mask = voxelCutMask[v] & 0xFFFu;
    if (mask == 0u || voxelOccupied[v] == 0u) return;

    int3 vCoord = make_int3(v % dims.x, (v / dims.x) % dims.y, v / (dims.x * dims.y));
    Conn4096Gpu conn = conn4096[mask];
    int compCount = conn.compCount;
    int off = voxelIsectOffset[v];
    int cnt = voxelIsectCount[v];

    float3 halfSize = 0.5f * voxelSize;

    for (int comp = 0; comp < compCount; comp++)
    {
        int slot = 8 * v + comp;

        // Per-COMPONENT Eq5/D5 box (plan v3.1 item 3, diagnosis D2 blocker): center = mean of the
        // COMPONENT'S OWN deformed corners. After a piece falls many*L, a voxel straddling both
        // fragments must clamp each side's FP toward ITS side, not toward the void between them
        // (the all-8 mean caused the post-separation striated strands). Reduces to the all-8 mean
        // pre-cut (single comp). Also decide pure-air here (Q1 correction: the criterion is "the
        // component contains NO inside (phi<0) corner"  - NOT nIncl==0, which also matches severed
        // all-inside groups with no incident iso edge).
        float3 bc = make_float3(0.f, 0.f, 0.f);
        int nCorn = 0;
        bool pureAir = true;
        for (int lc = 0; lc < 8; lc++)
        {
            if (Conn4096Comp(conn, lc) != comp) continue;
            int cid = voxelCorner[8 * v + lc];
            bc = bc + cornerPos[cid];
            nCorn++;
            if (cornerInside[cid] != 0) pureAir = false;
        }
        if (nCorn > 0) bc = bc / (float)nCorn;

        // Combined seed: sum of the COMPONENT's iso intersection points ...
        float3 sum = make_float3(0.f, 0.f, 0.f);
        int nIncl = 0;
        for (int s = 0; s < cnt; s++)
        {
            IsectGpu it = isect[off + s];
            int insideCorner = (it.insideA == 1) ? it.cornerA : it.cornerB;
            int3 icc = CornerCoordCuda(insideCorner, dims);
            int lc = LocalCornerOf(icc, vCoord);
            if (lc < 0) continue;
            if (Conn4096Comp(conn, lc) != comp) continue;
            sum = sum + isectWorld[off + s];
            nIncl++;
        }
        // ... plus its cut points ("the cut edges will be calculated twice" = both endpoint owners).
        int ccnt = cutFPAccumCnt[slot];
        if (ccnt > 0)
            sum = sum + make_float3((float)cutFPAccumPos[3 * slot + 0],
                                    (float)cutFPAccumPos[3 * slot + 1],
                                    (float)cutFPAccumPos[3 * slot + 2]) / POS_SCALE;
        int total = nIncl + ccnt;
        if (total == 0) { cutFP[slot] = make_float4(0.f, 0.f, 0.f, 0.f); continue; }
        float3 x = sum / (float)total;                       // Eq2 seed over ALL crossing points

        // Fig 2.2 particle-movement refinement over the component's iso intersections (Eq3/D1 + Eq4).
        if (nIncl > 0)
        {
            for (int it2 = 0; it2 < qefIters; it2++)
            {
                float3 F = make_float3(0.f, 0.f, 0.f);
                for (int s2 = 0; s2 < cnt; s2++)
                {
                    IsectGpu it = isect[off + s2];
                    int insideCorner = (it.insideA == 1) ? it.cornerA : it.cornerB;
                    int3 icc = CornerCoordCuda(insideCorner, dims);
                    int lc = LocalCornerOf(icc, vCoord);
                    if (lc < 0 || Conn4096Comp(conn, lc) != comp) continue;
                    float3 p_i = isectWorld[off + s2];
                    float3 n_i = it.normal;                   // rest normal (PAPER-SILENT frozen)
                    F = F + n_i * dot3(n_i, p_i - x);         // DEVIATION D1
                }
                float a = 0.1f * (1.0f - (float)it2 / (float)qefIters);   // Eq4
                float3 xn = x + a * F;
                if (BoxSdfAnisotropic(xn, bc, voxelSize) >= 0.0f) break;  // D5 stop
                x = xn;
            }
        }

        // DEVIATION D5 / Eq5 clamp: keep the FP inside its deformed voxel box.
        x.x = fminf(fmaxf(x.x, bc.x - halfSize.x), bc.x + halfSize.x);
        x.y = fminf(fmaxf(x.y, bc.y - halfSize.y), bc.y + halfSize.y);
        x.z = fminf(fmaxf(x.z, bc.z - halfSize.z), bc.z + halfSize.z);

        // D4 variant-(c) skin-side projection, AFTER the D5 clamp (Q1 correction on ordering; the
        // per-component box of a pure-air comp centers on its air corners, so projecting first
        // would be partially undone). PAPER-SILENT deviation (D9c): pure-air comp FPs are projected
        // onto the voxel's mean iso plane when above it  - moving only TOWARD tissue, which the Eq5
        // box never guarantees. Kills the A1 lip protrusion without starving the rim wall.
        if (pureAir && cnt > 0)
        {
            float3 nSum = make_float3(0.f, 0.f, 0.f);
            float  dSum = 0.f;
            for (int s3 = 0; s3 < cnt; s3++)
            {
                float3 n_i = isect[off + s3].normal;
                nSum = nSum + n_i;
                dSum = dSum + dot3(n_i, x - isectWorld[off + s3]);
            }
            float d = dSum / (float)cnt;
            float nLen = length3(nSum);
            if (d > 0.f && nLen > 1e-6f) x = x - (d / nLen) * nSum;   // project onto the mean skin plane
        }

        // v4.1 P2 (kerf fix, diagnosis S2 + plan review wf_b08c3fc2 R1-m1 ordering): wall-plane
        // projection LAST, with NO re-clamp after it  - the wall term has the final word on the
        // along-n_cut coordinate. Without this, iso intersections (~0.5L lateral) dominate the seed
        // and the Fig-2.2 refinement has no cut-plane term, so FPs settle ~0.22L from the plane and
        // the horizontal kerf reads ~0.44L x 1/sin(theta) instead of D=0.16L. The component's OWN
        // cut-point centroid already sits ON its ±D/2 wall (EmitCutPoint bakes the offset), so
        // projecting x onto the plane through that centroid  - the mean recorded cut normal pulls
        // the lip back to the paper's D-wide slit.
        // Degenerate guard (R1-m2/R2-m3): skip when the accumulated normals disagree badly (multi-
        // stroke opposing planes can cancel the sum -> normalize(~0) would NaN-poison the FP).
        if (ccnt > 0)
        {
            float3 nAcc = make_float3((float)cutNrmAccum[3 * slot + 0],
                                      (float)cutNrmAccum[3 * slot + 1],
                                      (float)cutNrmAccum[3 * slot + 2]) / POS_SCALE;
            float nLenW = length3(nAcc);
            if (nLenW > 1e-6f && nLenW > 0.25f * (float)ccnt)   // coherent sum of ccnt unit normals  - ccnt
            {
                float3 nbar = nAcc / nLenW;
                float3 cutC = make_float3((float)cutFPAccumPos[3 * slot + 0],
                                          (float)cutFPAccumPos[3 * slot + 1],
                                          (float)cutFPAccumPos[3 * slot + 2]) / POS_SCALE / (float)ccnt;
                x = x - dot3(nbar, x - cutC) * nbar;   // onto the comp's own ±D/2 wall plane
            }
        }

        cutFP[slot] = make_float4(x.x, x.y, x.z, 1.0f);
    }
}


// ── InterpCutFP  - per slot: EMA smooth the cut-FP drift (paper Discussion). alpha=1 -> OFF (=curr).
// v4.1 P1a (diagnosis S1, plan review wf_b08c3fc2): the slot key 8v+comp is RENUMBERED by conn4096
// whenever the voxel's cutMask grows (dense first-appearance renumbering), so EMA history under a
// stale key can belong to the OTHER lip  - lerping across lips paints filaments/fans. The mask
// mutates only by atomicOr growth, so "mask changed since the last chain" catches EVERY re-key:
// treat prev as invalid then (appear at curr). ALSO (R1-m3, the EMA LAG half): masks are static
// during free fall, so a piece falling >0.5L/frame at alpha<1 would trail its unsmoothed skin FPs -
// the jump gate snaps rigid motion instead of smoothing it. Together the artifacts are bounded at
// ANY alpha. [DEBUG-CUT]: dbg 0=pairs 1=jump>L 2=jump>3L 3=maxDisp 4=slotNew 5=epochInval.
__global__ void k_InterpCutFP(int voxelCount, float cutFPInterp, float voxelL,
                              const unsigned int* voxelCutMask, const unsigned int* prevVoxelMask,
                              float4* cutFP, float4* prevCutFP, unsigned int* dbg)
{
    int slot = blockIdx.x * blockDim.x + threadIdx.x;
    if (slot >= 8 * voxelCount) return;
    float4 curr = cutFP[slot];
    float4 prev = prevCutFP[slot];
    bool epoch = (voxelCutMask[slot >> 3] != prevVoxelMask[slot >> 3]);   // this voxel re-keyed
    float4 outv;
    if (curr.w < 0.5f)       outv = make_float4(0.f, 0.f, 0.f, 0.f);                 // no FP now
    else if (prev.w < 0.5f)
    {
        outv = curr;                                                                  // newly valid -> appear at curr
        if (dbg) atomicAdd(&dbg[4], 1u);
    }
    else if (epoch)
    {
        outv = curr;                                    // mask-epoch invalidation: history may be the other lip
        if (dbg) atomicAdd(&dbg[5], 1u);
    }
    else
    {
        float3 dv = make_float3(curr.x - prev.x, curr.y - prev.y, curr.z - prev.z);
        float  dl = length3(dv);
        if (dbg)
        {
            atomicAdd(&dbg[0], 1u);
            if (dl > voxelL)        atomicAdd(&dbg[1], 1u);
            if (dl > 3.f * voxelL)  atomicAdd(&dbg[2], 1u);
            atomicMax(&dbg[3], __float_as_uint(dl));    // monotone for non-negative floats
        }
        if (dl > 0.5f * voxelL)
            outv = curr;                                // jump gate: rigid motion (free fall), not jitter
        else
        {
            float3 l = lerp3(make_float3(prev.x, prev.y, prev.z), make_float3(curr.x, curr.y, curr.z), cutFPInterp);
            outv = make_float4(l.x, l.y, l.z, 1.0f);
        }
    }
    cutFP[slot] = outv;
    prevCutFP[slot] = outv;
}

// ── EmitCutQuad (device)  - append one cut edge's present component FPs as 1-2 tris (bit31 tags),
// ORIENTED outward (audit wf_60781cea A3 blocker: stencil-order winding left one full lip's wall
// normals inverted  - 224/224 in the harness). refDir points from the emitting particle THROUGH the
// severed edge toward the opposite endpoint = out of the fragment's cut face; flip when the
// geometric normal opposes it.
// PAPER-SILENT (review wf_4a45e51c R1-m2): the paper gives NO cut-wall winding (Fig 2.5 only says
// the four FPs "are connected"); the outward refDir convention is an implementation decision (D9b).
__device__ __forceinline__ void EmitCutQuad(bool ok[4], int tag[4], int n, float3 refDir,
                                            const float4* cutFP, int triCapacity,
                                            int* tri, unsigned int* triCounter, float voxelL)
{
    if (n < 3) return;
    int ord[4]; int m = 0;
    for (int s = 0; s < 4; s++) if (ok[s]) { ord[m] = s; m++; }
    int ntris = (n >= 4) ? 2 : 1;

    // v5.1 P5b giant-wall gate (review wf_c416ae77 R2-m2: 6L clears the legit sagged-tissue
    // envelope ~3.5L while runaway mid-gap FPs reach 10-50L): a wall quad whose FPs straddle the
    // widening separation gap is geometric noise  - skip it. Transient-only once the D10 tear law
    // splits the bridge comps (giantWall in [DEBUG-CUT] tracks the phenomenon either way).
    float3 q[4];
    for (int k = 0; k < m; k++)
    {
        float4 f = cutFP[(unsigned int)tag[ord[k]] & 0x3FFFFFFFu];
        q[k] = make_float3(f.x, f.y, f.z);
    }
    float gate2 = 36.0f * voxelL * voxelL;   // (6L)^2
    for (int a = 0; a < m; a++)
        for (int b = a + 1; b < m; b++)
        {
            float3 dq = q[a] - q[b];
            if (dot3(dq, dq) > gate2) return;
        }

    // outward orientation from the first triangle's geometric normal
    float3 nGeo = cross3(q[1] - q[0], q[2] - q[0]);
    bool flip = dot3(nGeo, refDir) < 0.0f;

    unsigned int baseIdx = atomicAdd(&triCounter[0], (unsigned int)(3 * ntris));
    if (baseIdx + (unsigned int)(3 * ntris) > (unsigned int)(triCapacity * 3)) return;

    int i1 = flip ? ord[2] : ord[1];
    int i2 = flip ? ord[1] : ord[2];
    tri[baseIdx + 0] = tag[ord[0]]; tri[baseIdx + 1] = tag[i1]; tri[baseIdx + 2] = tag[i2];
    if (ntris == 2)
    {
        int j1 = flip ? ord[3] : ord[2];
        int j2 = flip ? ord[2] : ord[3];
        tri[baseIdx + 3] = tag[ord[0]]; tri[baseIdx + 4] = tag[j1]; tri[baseIdx + 5] = tag[j2];
    }
}

// ── BuildCutTriangles  - one thread per particle: gather component FPs over incident cut edges ->
// cut-wall faces. Stage 7: NO rim band (the paper's margin is the SHARED single FP per component -
// skin quads end at the same vertex the wall starts from; audit wf_60781cea A2).
__global__ void k_BuildCutTriangles(int cornerCount, int triCapacity, int3 dims, float voxelL,
                                    const unsigned int* voxelCutMask, const unsigned int* voxelOccupied,
                                    const Conn4096Gpu* conn4096, const float4* cutFP,
                                    const float3* cornerPos,
                                    int* tri, unsigned int* triCounter, unsigned int* dbg)
{
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= cornerCount) return;
    int3 c = CornerCoordCuda(p, dims);

    int  axes[6] = {0, 0, 1, 1, 2, 2};
    int3 bs[6] = { c, c - make_int3(1,0,0), c, c - make_int3(0,1,0), c, c - make_int3(0,0,1) };

    for (int e = 0; e < 6; e++)
    {
        int axis = axes[e];
        int3 baseC = bs[e];
        if (!EdgeValid(axis, baseC, dims)) continue;
        if (!EdgeIsCut(axis, baseC, dims, voxelCutMask)) continue;

        // outward reference: from this particle toward the severed opposite endpoint
        int3 axisDir = make_int3(axis == 0 ? 1 : 0, axis == 1 ? 1 : 0, axis == 2 ? 1 : 0);
        bool basedHere = (baseC.x == c.x && baseC.y == c.y && baseC.z == c.z);
        int3 qc = basedHere ? c + axisDir : c - axisDir;
        float3 refDir = cornerPos[CorIdCuda(qc, dims)] - cornerPos[p];

        int tag[4]; bool ok[4]; int n = 0;
        for (int s = 0; s < 4; s++)
        {
            ok[s] = false;
            int3 vc = baseC + ES_VoxelOffset(axis, s);
            int v = VoxId(vc, dims);
            if (voxelOccupied[v] == 0u) continue;
            unsigned int mask = voxelCutMask[v] & 0xFFFu;
            if (mask == 0u) continue;
            int lc = LocalCornerOf(c, vc);
            if (lc < 0) continue;
            int comp = Conn4096Comp(conn4096[mask], lc);
            int slot = 8 * v + comp;
            if (cutFP[slot].w < 0.5f) continue;
            // bit31 = component cut FP; bit30 = WALL-EMITTER marker (liver-render design §3.3:
            // bit31 alone cannot distinguish wall from skin  - StitchVertId shares bit31 tags with
            // cut-voxel skin stitches; bit30 is transparent to every consumer: all idx decodes
            // mask & 0x3FFFFFFF, all cut-FP tests use bit31 only  - verified wf_9078a66a V1).
            tag[s] = (int)(0xC0000000u | (unsigned int)slot);
            ok[s] = true;
            n++;
        }
        // [DEBUG-CUT] rim<3 (P3): cut edge with 1-2 present FPs -> no wall quad emitted (hole
        // candidate). basedHere gates to ONE endpoint thread so each edge counts once (R2-m4).
        if (dbg && basedHere && n > 0 && n < 3) atomicAdd(&dbg[11], 1u);
        EmitCutQuad(ok, tag, n, refDir, cutFP, triCapacity, tri, triCounter, voxelL);
    }
}


// ── DC_NormalizeCutFP  - per slot: normalize the cut-FP normal accumulator ──
__global__ void k_NormalizeCutFP(int voxelCount, const int* cutFPNormal, float3* cutFPNormalF)
{
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= 8 * voxelCount) return;
    float3 nv = make_float3((float)cutFPNormal[3 * s + 0], (float)cutFPNormal[3 * s + 1], (float)cutFPNormal[3 * s + 2]) / NORMAL_SCALE;
    float len = length3(nv);
    cutFPNormalF[s] = (len > 1e-6f) ? nv / len : make_float3(0.f, 0.f, 0.f);
}

// ─────────────────────────────────────────────────────────────────────────────────────────────
// Host API
// ─────────────────────────────────────────────────────────────────────────────────────────────
#define CK(call) do { cudaError_t _e = (call); if (_e != cudaSuccess) { cut_shutdown(); return -(int)_e; } } while (0)

int cut_init(const CutInitDesc* desc, const void* conn4096, const void* gridEdges, const int* voxelOccupied,
             const int* cornerInside)
{
    if (g_cut_ready) cut_shutdown();
    if (!desc) return -3000;
    g_cdesc = *desc;
    g_cdims = make_int3(desc->dimsX, desc->dimsY, desc->dimsZ);

    int vc  = desc->voxelCount;
    int gec = desc->gridEdgeCount;
    int cpc = desc->cutPointCapacity > 0 ? desc->cutPointCapacity : 1;
    int slot = 8 * vc;
    if (vc <= 0 || desc->cornerCount <= 0) return -3001;
    int gecA = gec > 0 ? gec : 1;

    CK(cudaMalloc(&d_gridEdges,       (size_t)gecA   * sizeof(GridEdgeGpu2)));
    CK(cudaMalloc(&d_prevCorner,      (size_t)desc->cornerCount * sizeof(float3)));   // CCD tick snapshot
    CK(cudaMalloc(&d_cutRestCorner,   (size_t)desc->cornerCount * sizeof(float3)));
    CK(cudaMalloc(&d_conn4096,        (size_t)4096   * sizeof(Conn4096Gpu)));
    CK(cudaMalloc(&d_voxelOccupied,   (size_t)vc     * sizeof(unsigned int)));
    CK(cudaMalloc(&d_voxelCutMask,    (size_t)vc     * sizeof(unsigned int)));
    CK(cudaMalloc(&d_cutPoint,        (size_t)cpc    * sizeof(CutPointGpu)));
    CK(cudaMalloc(&d_cutPointCounter,                  sizeof(unsigned int)));
    CK(cudaMalloc(&d_cutFP,           (size_t)slot   * sizeof(float4)));
    CK(cudaMalloc(&d_prevCutFP,       (size_t)slot   * sizeof(float4)));
    CK(cudaMalloc(&d_cornerInside,    (size_t)desc->cornerCount * sizeof(int)));
    CK(cudaMalloc(&d_cutFPAccumPos,   (size_t)3*slot * sizeof(int)));
    CK(cudaMalloc(&d_cutFPAccumCnt,   (size_t)slot   * sizeof(int)));
    CK(cudaMalloc(&d_voxelFPCount,    (size_t)vc     * sizeof(unsigned int)));
    CK(cudaMalloc(&d_cutFPNormal,     (size_t)3*slot * sizeof(int)));
    CK(cudaMalloc(&d_cutFPNormalF,    (size_t)slot   * sizeof(float3)));
    CK(cudaMalloc(&d_prevVoxelMask,   (size_t)vc     * sizeof(unsigned int)));   // v4.1 P1a
    CK(cudaMalloc(&d_cutNrmAccum,     (size_t)3*slot * sizeof(int)));            // v4.1 P2
    CK(cudaMalloc(&d_dbg,             (size_t)16     * sizeof(unsigned int)));   // v4.1 P3

    CK(cudaMemcpy(d_conn4096, conn4096, (size_t)4096 * sizeof(Conn4096Gpu), cudaMemcpyHostToDevice));
    if (gec > 0) CK(cudaMemcpy(d_gridEdges, gridEdges, (size_t)gec * sizeof(GridEdgeGpu2), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_voxelOccupied, voxelOccupied, (size_t)vc * sizeof(unsigned int), cudaMemcpyHostToDevice));

    // Zero the cumulative + scratch buffers (cut state starts uncut).
    CK(cudaMemset(d_voxelCutMask,    0, (size_t)vc     * sizeof(unsigned int)));
    CK(cudaMemset(d_cutPointCounter, 0,                  sizeof(unsigned int)));
    CK(cudaMemset(d_cutFP,           0, (size_t)slot   * sizeof(float4)));     // w=0 invalid
    CK(cudaMemset(d_prevCutFP,       0, (size_t)slot   * sizeof(float4)));     // EMA seed (w=0)
    CK(cudaMemcpy(d_cornerInside, cornerInside, (size_t)desc->cornerCount * sizeof(int), cudaMemcpyHostToDevice));
    CK(cudaMemset(d_cutFPAccumPos,   0, (size_t)3*slot * sizeof(int)));
    CK(cudaMemset(d_cutFPAccumCnt,   0, (size_t)slot   * sizeof(int)));
    CK(cudaMemset(d_voxelFPCount,    0, (size_t)vc     * sizeof(unsigned int)));
    CK(cudaMemset(d_cutFPNormal,     0, (size_t)3*slot * sizeof(int)));
    CK(cudaMemset(d_cutFPNormalF,    0, (size_t)slot   * sizeof(float3)));
    CK(cudaMemset(d_prevVoxelMask,   0, (size_t)vc     * sizeof(unsigned int))); // matches the zeroed cut mask
    CK(cudaMemset(d_cutNrmAccum,     0, (size_t)3*slot * sizeof(int)));
    CK(cudaMemset(d_dbg,             0, (size_t)16     * sizeof(unsigned int)));
    CutEventMeta zeroEventMeta = {};
    CutEventSummary zeroEventSummary = {};
    CK(cudaMemcpyToSymbol(g_cutEventMeta, &zeroEventMeta, sizeof(zeroEventMeta)));
    CK(cudaMemcpyToSymbol(g_cutEventSummary, &zeroEventSummary, sizeof(zeroEventSummary)));

    // Cache the shared dc_recon + physics pointers (stable after their init).
    c_cornerPos        = recon_corner_pos();
    c_isect            = (IsectGpu*)recon_isect();
    c_isectWorld       = recon_isect_world();
    c_voxelIsectOffset = recon_voxel_isect_offset();
    c_voxelIsectCount  = recon_voxel_isect_count();
    c_voxelCorner      = recon_voxel_corner();
    c_tri              = recon_tri();
    c_triCounter       = recon_tri_counter();
    c_particleRot      = (RotGpu*)physics_particle_rot();
    c_nbrIdx           = physics_nbr_idx();
    c_bendPairs        = (BendPairGpuC*)physics_bend_pairs();
    c_active           = physics_active();   // v4.1 P1b (freeze debris + cull its geometry)
    if (!c_cornerPos || !c_particleRot || !c_nbrIdx || !c_active) { cut_shutdown(); return -3002; }

    // Hand the cut buffers to dc_recon so the cut-aware Stitch / Normals / Expand resolve cut FPs.
    recon_set_cut_buffers(d_voxelCutMask, d_voxelOccupied, d_conn4096, d_cutFP, d_cutFPNormalF, d_cutFPNormal);

    g_toolSet = false; g_havePrevCorner = false;
    g_cutRestMetricEnabled = false;
    g_cutVoxelL = desc->voxelL;
    g_cutVoxelSize = make_float3(desc->voxelL, desc->voxelL, desc->voxelL);
    g_cutVoxelMaxL = desc->voxelL;
    g_emitNcut = make_float3(0.f, 1.f, 0.f);
    // v5.1 D10: sanitize tau (R1-m5: a stale-DLL/struct mismatch must degrade to tear-OFF, never a
    // garbage threshold). tau < 1 would tear edges at rest length; NaN fails the >= compare.
    {
        g_tearStretchRatio = g_cdesc.tearStretchRatio;
        float tau = g_tearStretchRatio;
        g_tearLen2 = (tau >= 1.0f && tau == tau) ? (tau * g_cutVoxelL) * (tau * g_cutVoxelL) : 0.f;
    }

    g_cut_ready = true;
    return 0;
}

void cut_set_tool(const CutToolDesc* tool)
{
    if (!tool) return;
    g_tool = *tool;
    // Invalid descriptors mean there is no lateral tool sweep this frame. They must not arm the
    // post-cut tear law, otherwise gravity alone can turn an untouched liver into cut-wall debris.
    if (tool->valid == 0) return;
    g_emitNcut = mk3(tool->ncut);   // remember the last VALID world cut normal
    // Seed the CCD snapshot at ACTIVATION (review S1-m3): without this the first physics substep of
    // the next frame is consumed by the seed pass and its tissue motion is never CCD-checked.
    if (!g_toolSet && g_cut_ready && c_cornerPos != nullptr)
    {
        if (cudaMemcpy(d_prevCorner, c_cornerPos, (size_t)g_cdesc.cornerCount * sizeof(float3),
                       cudaMemcpyDeviceToDevice) == cudaSuccess)
            g_havePrevCorner = true;
    }
    g_toolSet = true;
}

int cut_set_rest_metric(const float* alignedRestPositions, float voxelSizeX, float voxelSizeY, float voxelSizeZ)
{
    if (!g_cut_ready || !alignedRestPositions ||
        !(voxelSizeX > 0.f) || !(voxelSizeY > 0.f) || !(voxelSizeZ > 0.f) ||
        voxelSizeX != voxelSizeX || voxelSizeY != voxelSizeY || voxelSizeZ != voxelSizeZ)
        return -3050;
    cudaError_t e = cudaMemcpy(d_cutRestCorner, alignedRestPositions,
                               (size_t)g_cdesc.cornerCount * sizeof(float3), cudaMemcpyHostToDevice);
    if (e != cudaSuccess) return -(int)e;
    g_cutVoxelSize = make_float3(voxelSizeX, voxelSizeY, voxelSizeZ);
    g_cutVoxelMaxL = fmaxf(voxelSizeX, fmaxf(voxelSizeY, voxelSizeZ));
    g_cutVoxelL = (voxelSizeX + voxelSizeY + voxelSizeZ) / 3.f;
    g_cutRestMetricEnabled = true;
    return 0;
}

void cut_clear_rest_metric()
{
    g_cutRestMetricEnabled = false;
    g_cutVoxelL = g_cdesc.voxelL;
    g_cutVoxelSize = make_float3(g_cdesc.voxelL, g_cdesc.voxelL, g_cdesc.voxelL);
    g_cutVoxelMaxL = g_cdesc.voxelL;
}

int cut_set_gravity_stabilization(int enabled, float gravityX, float gravityY, float gravityZ,
                                  float gapWorld, float alignmentExponent)
{
    if (!g_cut_ready || gapWorld != gapWorld || alignmentExponent != alignmentExponent ||
        gapWorld < 0.f || alignmentExponent < 0.5f || alignmentExponent > 4.f)
        return -3051;
    float3 gravity = make_float3(gravityX, gravityY, gravityZ);
    float gravityLength = length3(gravity);
    g_gravityStabilizationEnabled = enabled != 0 && gravityLength > 1e-6f && gapWorld > 0.f;
    g_gravityDirection = gravityLength > 1e-6f
        ? gravity / gravityLength
        : make_float3(0.f, -1.f, 0.f);
    g_gravitySafeGap = gapWorld;
    g_gravityAlignmentExponent = alignmentExponent;
    return 0;
}

// ─────────────────────────────────────────────────────────────────────────────────────────────
// Stage-6 detection host orchestration (task_plan.md Stage 6).
//   cut_detect()       - per C# rod substep (rod moved, tissue frozen): SS2.1.3 world quad M-T.
//   cut_ribbon_tick()  - per accepted RK45 substep (tissue moved, rod frozen): per-edge CCD of the
//                       moving deformed edge vs the static blade segment; then roll the corner
//                       snapshot so consecutive ticks tile the full space-time sweep.
// ─────────────────────────────────────────────────────────────────────────────────────────────
int cut_detect()
{
    if (!g_cut_ready || !g_toolSet || g_tool.valid == 0) return 0;
    int gec = g_cdesc.gridEdgeCount;
    if (gec <= 0) return 0;

    ToolGpu t;
    t.t1v0 = mk3(g_tool.t1v0); t.t1v1 = mk3(g_tool.t1v1); t.t1v2 = mk3(g_tool.t1v2);
    t.t2v0 = mk3(g_tool.t2v0); t.t2v1 = mk3(g_tool.t2v1); t.t2v2 = mk3(g_tool.t2v2);
    t.ncut = mk3(g_tool.ncut); t.aabbMin = mk3(g_tool.aabbMin); t.aabbMax = mk3(g_tool.aabbMax);
    t.d = g_tool.d; t.valid = g_tool.valid;

    const float3* cutRest = g_cutRestMetricEnabled ? d_cutRestCorner : nullptr;
    k_DetectCutQuad<<<Gc(gec), 64>>>(gec, t, g_cutVoxelMaxL, g_cdims, g_cdesc.cutPointCapacity,
                                     d_cornerInside, c_cornerPos, d_gridEdges, cutRest, d_voxelCutMask, d_voxelOccupied,
                                     d_cutPoint, d_cutPointCounter, c_particleRot);
    k_SeverLinks<<<Gc(gec), 64>>>(gec, g_cdims, d_gridEdges, d_voxelCutMask, c_nbrIdx, c_bendPairs);

    cudaError_t e = cudaGetLastError();
    return (e == cudaSuccess) ? 0 : -(int)e;
}

void cut_set_event_meta(const CutEventMeta* meta)
{
    if (!g_cut_ready || !meta) return;
    cudaMemcpyToSymbol(g_cutEventMeta, meta, sizeof(CutEventMeta));
}

int cut_get_event_summary(CutEventSummary* out)
{
    if (!g_cut_ready || !out) return -1;
    return cudaMemcpyFromSymbol(out, g_cutEventSummary, sizeof(CutEventSummary)) == cudaSuccess ? 0 : -3313;
}

// cut_ribbon_tick  - CCD pass per accepted RK45 substep. The blade segment is the CURRENT rod
// [S=t2v2, E=t1v2] (rod is frozen during the physics phase); prev corner snapshot rolls after
// every pass so each tick covers exactly one tissue sub-interval (linear endpoint motion).
int cut_ribbon_tick()
{
    if (!g_cut_ready || !g_toolSet) return 0;
    int gec = g_cdesc.gridEdgeCount;
    int cc  = g_cdesc.cornerCount;
    if (gec <= 0 || cc <= 0) return 0;

    if (!g_havePrevCorner)
    {
        if (cudaMemcpy(d_prevCorner, c_cornerPos, (size_t)cc * sizeof(float3), cudaMemcpyDeviceToDevice) != cudaSuccess)
            return -3200;
        g_havePrevCorner = true;
        return 0;                       // seed pass: nothing swept yet
    }

    float3 S = mk3(g_tool.t2v2);        // current rod endpoints (CuttingTool: T2V2=S, T1V2=E)
    float3 E = mk3(g_tool.t1v2);

    const float3* cutRest = g_cutRestMetricEnabled ? d_cutRestCorner : nullptr;
    k_DetectCutCCD<<<Gc(gec), 64>>>(gec, S, E, g_emitNcut, g_tool.d, g_cutVoxelMaxL, g_cdims,
                                    g_cdesc.cutPointCapacity, d_prevCorner, c_cornerPos,
                                    d_gridEdges, cutRest, c_nbrIdx, d_cornerInside, d_voxelCutMask, d_voxelOccupied,
                                    d_cutPoint, d_cutPointCounter, c_particleRot);
    // v5.1 D10 tear law  - BEFORE k_SeverLinks so the same tick severs the newly marked tears.
    if (g_tearLen2 > 0.f)
        k_TearOverstretch<<<Gc(gec), 64>>>(gec, g_tearStretchRatio, g_cdims, g_cutVoxelMaxL, g_tool.d,
                                           d_gridEdges, c_nbrIdx, c_active, c_cornerPos, cutRest,
                                           d_voxelCutMask, d_voxelOccupied, d_cutPoint,
                                           d_cutPointCounter, g_cdesc.cutPointCapacity,
                                           c_particleRot, d_dbg);
    k_SeverLinks<<<Gc(gec), 64>>>(gec, g_cdims, d_gridEdges, d_voxelCutMask, c_nbrIdx, c_bendPairs);

    if (cudaMemcpy(d_prevCorner, c_cornerPos, (size_t)cc * sizeof(float3), cudaMemcpyDeviceToDevice) != cudaSuccess)
        return -3201;

    cudaError_t e = cudaGetLastError();
    return (e == cudaSuccess) ? 0 : -(int)e;
}


int cut_fp_chain()
{
    if (!g_cut_ready) return 0;
    int vc = g_cdesc.voxelCount, slot = 8 * vc, cpc = g_cdesc.cutPointCapacity;
    int cc = g_cdesc.cornerCount;
    // [DEBUG-CUT] window = one chain (one frame); the C# side reads EVERY frame and aggregates
    // host-side (plan review R1-m5/R2-m2: 1 Hz sampling of a per-frame window misses transients).
    // Slots 0-14 ONLY: dbg[15] (tearMarks) is written during LCS_Step's ribbon ticks, which run
    // BEFORE this chain in the same frame  - zeroing it here would erase every tear before the
    // readback (wf_c416ae77 R2-m3). It stays cumulative.
    if (cudaMemset(d_dbg, 0, 15 * sizeof(unsigned int)) != cudaSuccess) return -3300;
    // v4.1 P1b: freeze fully-severed debris crumbs BEFORE the accumulate pass so their cut points
    // stop seeding FPs in the SAME frame.
    k_DeactivateSevered<<<Gc(cc), 64>>>(cc, c_nbrIdx, c_active, d_dbg);
    k_ClearCutFP<<<Gc(slot), 64>>>(vc, d_cutFPAccumPos, d_cutFPAccumCnt, d_cutFP, d_cutFPNormal, d_voxelFPCount,
                                   d_cutNrmAccum);
    k_LookupConnectivity<<<Gc(vc), 64>>>(vc, d_voxelCutMask, d_voxelOccupied, d_conn4096, d_voxelFPCount, d_dbg);
    k_AccumulateCutFP<<<Gc(cpc), 64>>>(cpc, d_cutPointCounter, d_cutPoint, c_particleRot, c_cornerPos, g_cdims,
                                       d_gridEdges, d_voxelOccupied, d_voxelCutMask, d_conn4096,
                                       d_cutFPAccumPos, d_cutFPAccumCnt, c_active, d_cutNrmAccum,
                                       g_gravityStabilizationEnabled ? 1 : 0, g_gravityDirection,
                                       g_gravitySafeGap, g_gravityAlignmentExponent);
    k_ComputeComponentFP<<<Gc(vc), 64>>>(vc, g_cdims, d_voxelCutMask, d_voxelOccupied, d_conn4096,
                                         c_voxelIsectOffset, c_voxelIsectCount, c_isect, c_isectWorld,
                                         d_cutFPAccumCnt, d_cutFPAccumPos, d_cutFP,
                                         c_voxelCorner, c_cornerPos, g_cutVoxelSize,
                                         recon_qef_iters(), d_cornerInside, d_cutNrmAccum);
    k_InterpCutFP<<<Gc(slot), 64>>>(vc, g_cdesc.cutFPInterp, g_cutVoxelL,
                                    d_voxelCutMask, d_prevVoxelMask, d_cutFP, d_prevCutFP, d_dbg);
    // v4.1 P1a: roll the mask epoch AFTER InterpCutFP consumed the comparison (default stream is
    // ordered, so this D2D copy runs after the kernel).
    if (cudaMemcpy(d_prevVoxelMask, d_voxelCutMask, (size_t)vc * sizeof(unsigned int),
                   cudaMemcpyDeviceToDevice) != cudaSuccess) return -3301;
    cudaError_t e = cudaGetLastError();
    return (e == cudaSuccess) ? 0 : -(int)e;
}

int cut_build_triangles()
{
    if (!g_cut_ready) return 0;
    k_BuildCutTriangles<<<Gc(g_cdesc.cornerCount), 64>>>(g_cdesc.cornerCount, g_cdesc.triCapacity, g_cdims,
                                                         g_cutVoxelL,
                                                         d_voxelCutMask, d_voxelOccupied, d_conn4096, d_cutFP,
                                                         c_cornerPos, c_tri, c_triCounter, d_dbg);
    cudaError_t e = cudaGetLastError();
    return (e == cudaSuccess) ? 0 : -(int)e;
}

// ── cut_get_debug (v4.1 P3)  - fill out18 with the 16 [DEBUG-CUT] counters + the RAW (unclamped)
// cutPoint/tri atomic counters (exact demand meters  - both increment past capacity by design), and
// report the RUNTIME cutFPInterp so the log line settles the alpha ambiguity (S1-f4). Runs the
// giant-triangle scan (dc_recon owns d_tri/FetchPos  - cross-TU hook per R2-m4) before the readback.
int cut_get_debug(unsigned int* out18, float* alphaOut)
{
    if (!g_cut_ready || !out18) return -1;
    int r = recon_debug_scan_tris(d_dbg, g_cutVoxelL);
    if (r != 0) return r;
    unsigned int dbg[16];
    if (cudaMemcpy(dbg, d_dbg, sizeof(dbg), cudaMemcpyDeviceToHost) != cudaSuccess) return -3310;
    unsigned int cutRaw = 0, triRaw = 0;
    if (cudaMemcpy(&cutRaw, d_cutPointCounter, sizeof(unsigned int), cudaMemcpyDeviceToHost) != cudaSuccess) return -3311;
    if (c_triCounter &&
        cudaMemcpy(&triRaw, c_triCounter, sizeof(unsigned int), cudaMemcpyDeviceToHost) != cudaSuccess) return -3312;
    memcpy(out18, dbg, sizeof(dbg));
    out18[16] = cutRaw;   // raw cut points emitted (cumulative) vs cutPointCapacity
    out18[17] = triRaw;   // raw tri indices this frame vs 3*triCapacity
    if (alphaOut) *alphaOut = g_cdesc.cutFPInterp;
    return 0;
}

int cut_normalize_fp()
{
    if (!g_cut_ready) return 0;
    int vc = g_cdesc.voxelCount;
    k_NormalizeCutFP<<<Gc(8 * vc), 64>>>(vc, d_cutFPNormal, d_cutFPNormalF);
    cudaError_t e = cudaGetLastError();
    return (e == cudaSuccess) ? 0 : -(int)e;
}

bool cut_is_ready() { return g_cut_ready; }

void cut_shutdown()
{
    cudaFree(d_gridEdges);       d_gridEdges = nullptr;
    cudaFree(d_prevCorner);      d_prevCorner = nullptr;
    cudaFree(d_cutRestCorner);   d_cutRestCorner = nullptr;
    g_toolSet = false; g_havePrevCorner = false; g_tearLen2 = 0.f;
    g_cutRestMetricEnabled = false; g_cutVoxelL = 0.f;
    g_cutVoxelSize = make_float3(0.f, 0.f, 0.f); g_cutVoxelMaxL = 0.f;
    g_tearStretchRatio = 0.f;
    g_gravityStabilizationEnabled = false;
    g_gravityDirection = make_float3(0.f, -1.f, 0.f);
    g_gravitySafeGap = 0.f;
    g_gravityAlignmentExponent = 1.f;
    cudaFree(d_conn4096);        d_conn4096 = nullptr;
    cudaFree(d_voxelOccupied);   d_voxelOccupied = nullptr;
    cudaFree(d_voxelCutMask);    d_voxelCutMask = nullptr;
    cudaFree(d_cutPoint);        d_cutPoint = nullptr;
    cudaFree(d_cutPointCounter); d_cutPointCounter = nullptr;
    cudaFree(d_cutFP);           d_cutFP = nullptr;
    cudaFree(d_prevCutFP);       d_prevCutFP = nullptr;
    cudaFree(d_cornerInside);    d_cornerInside = nullptr;
    cudaFree(d_cutFPAccumPos);   d_cutFPAccumPos = nullptr;
    cudaFree(d_cutFPAccumCnt);   d_cutFPAccumCnt = nullptr;
    cudaFree(d_voxelFPCount);    d_voxelFPCount = nullptr;
    cudaFree(d_cutFPNormal);     d_cutFPNormal = nullptr;
    cudaFree(d_cutFPNormalF);    d_cutFPNormalF = nullptr;
    cudaFree(d_prevVoxelMask);   d_prevVoxelMask = nullptr;
    cudaFree(d_cutNrmAccum);     d_cutNrmAccum = nullptr;
    cudaFree(d_dbg);             d_dbg = nullptr;
    c_active = nullptr;
    if (g_cut_ready) recon_set_cut_buffers(nullptr, nullptr, nullptr, nullptr, nullptr, nullptr);
    g_cut_ready = false;
}
