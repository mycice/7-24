// physics.cu  - host API for the tissue solver (Stage 2). SOLVER = XPBD small-steps since migration
// step 4 (design rev-B §9-(4), commit a170233): physics_init binds cuda/xpbd/xpbd_solver.cu to the
// SHARED buffers and physics_step runs its substep loop, interleaving the Stage-6 CCD cut tick.
//
// The RK45 machinery below (kernels k_SeedTrial..k_IntegrateY5 + the D3/D4/D6 deviations) is RETIRED -
// kept compiled for reference/rollback but NO LONGER DISPATCHED (deviation-ledger D3/D4/D6 VOID,
// superseded by XPBD small-steps; the solver stays inside the paper's own cited PBD family [7]/[22]).
//   Forces : structural Eq11 (+DEVIATION D2: subtract rest L0), bending Eq13-16 (Kelvin-Voigt)  - the
//            MODEL is unchanged; under XPBD the same springs/pairs solve as compliant constraints.
//   Frame  : ComputeParticleRot = per-particle coordinate frame from neighbour edges + prev-frame
//            average + cross-product fallback (Berndt [22] Fig 4 / paper SS2.1.2); used at Stage 3 -
//            integrator-agnostic, still dispatched.
//
// The committed state is (cornerPos, vel): cornerPos is SHARED with dc_recon (recon_corner_pos()) so the
// deformed corners re-mesh; vel is physics-owned (the XPBD module aliases both via xpbd_bind).

#include "common.cuh"     // float3 ops, length3/dot3/cross3, operators
#include "physics.h"
#include "dc_recon.h"     // recon_corner_pos(), recon_corner_count(), recon_dims()
#include "cut.h"          // cut_is_ready() + cut_ribbon_tick() (Stage 4 per-substep ribbon)
#include "xpbd/xpbd_solver.h"  // XPBD GO-LIVE (design rev-B 9-(4)): xpbd_bind + begin_frame/substep
#include <stdlib.h>
#include <math.h>

// ── Constants (mirror PhysicsCommon.hlsl + MassSpringSolver.cs) ──────────────────────────────
#define FORCE_SCALE 1024.0f     // FixedPointAtomic.ForceScale (1<<10)
#define EPS_SIN2    1e-8f       // PAPER-SILENT bending singularity guard (collinear / anti-collinear)
#define ERR_SCALE   1000000.0f  // ERR_SCALE (1e6); fixed-point error reduction
#define EPS_TOL     1e-3f       // D4 error tolerance (C# MassSpringSolver.Eps)

// ── GPU structs  - strides MUST match C# ReconBuffers (BendPairGpu 32, RotGpu 36) ─────────────
struct BendPairGpu { int i, j, k; float theta0; int alive, _p0, _p1, _p2; };           // 32 bytes
static_assert(sizeof(BendPairGpu) == 32, "BendPairGpu must be 32 bytes");
static_assert(sizeof(XpbdBendPairGpu) == sizeof(BendPairGpu),
              "xpbd module's layout-compatible BendPair copy drifted from physics.cu's");
struct RotGpu      { float3 r0, r1, r2; };                                              // 36 bytes (rows of R)
static_assert(sizeof(RotGpu) == 36, "RotGpu must be 36 bytes");
struct Slope       { float3 dx, dv; };                                                  // 24 bytes
static_assert(sizeof(Slope) == 24, "Slope must be 24 bytes");

// ── DOPRI5 Butcher tableau  - transcribed EXACTLY from PhysicsCommon.hlsl (Eq18/19) ───────────
__device__ const float DOPRI_A[7][6] = {
    { 0,             0,             0,             0,             0,             0 },
    { 0.2f,          0,             0,             0,             0,             0 },
    { 0.075f,        0.225f,        0,             0,             0,             0 },
    { 44.0f/45,     -56.0f/15,      32.0f/9,       0,             0,             0 },
    { 19372.0f/6561,-25360.0f/2187, 64448.0f/6561,-212.0f/729,    0,             0 },
    { 9017.0f/3168, -355.0f/33,     46732.0f/5247, 49.0f/176,    -5103.0f/18656, 0 },
    { 35.0f/384,     0,             500.0f/1113,   125.0f/192,   -2187.0f/6784,  11.0f/84 }
};
__device__ const float DOPRI_B5[7] = { 35.0f/384, 0, 500.0f/1113, 125.0f/192, -2187.0f/6784, 11.0f/84, 0 };
__device__ const float DOPRI_B4[7] = { 5179.0f/57600, 0, 7571.0f/16695, 393.0f/640, -92097.0f/339200, 187.0f/2100, 1.0f/40 };

// ── Device helpers ───────────────────────────────────────────────────────────────────────────
__device__ __forceinline__ float3 normalize3(float3 v) { return v / length3(v); }   // matches HLSL normalize
__device__ __host__ __forceinline__ float clampf(float x, float lo, float hi) { return fminf(fmaxf(x, lo), hi); }   // host (D4 h-clamp) + device (bending dot clamp)

// Fixed-point force atomics (SM has no native float atomicAdd; mirror PhysicsCommon AtomicAddForce/ReadForce).
__device__ __forceinline__ void AtomicAddForce(int* forceInt, int p, float3 f)
{
    atomicAdd(&forceInt[3 * p + 0], (int)(f.x * FORCE_SCALE));
    atomicAdd(&forceInt[3 * p + 1], (int)(f.y * FORCE_SCALE));
    atomicAdd(&forceInt[3 * p + 2], (int)(f.z * FORCE_SCALE));
}
__device__ __forceinline__ float3 ReadForce(const int* forceInt, int p)
{
    return make_float3((float)forceInt[3 * p + 0], (float)forceInt[3 * p + 1], (float)forceInt[3 * p + 2]) / FORCE_SCALE;
}

// UnitDeriv: d/dt normalize(xa-xb). dN/dt = du/|u| - u*(u.du)/|u|^3. Mirrors PhysicsCommon.hlsl.
__device__ __forceinline__ float3 UnitDeriv(float3 xa, float3 xb, float3 va, float3 vb)
{
    float3 u   = xa - xb;
    float  len = length3(u);
    if (len < 1e-8f) return make_float3(0.f, 0.f, 0.f);
    float3 du = va - vb;
    return du / len - (u * dot3(u, du)) / (len * len * len);
}

// ── 3x3 matrix (3 float3 ROWS) helpers  - mirror Physics.compute Det3x3/Inverse3x3/transpose ──
struct Mat3 { float3 r0, r1, r2; };

__device__ __forceinline__ float Det3x3(Mat3 m)
{
    return m.r0.x * (m.r1.y * m.r2.z - m.r1.z * m.r2.y)
         - m.r0.y * (m.r1.x * m.r2.z - m.r1.z * m.r2.x)
         + m.r0.z * (m.r1.x * m.r2.y - m.r1.y * m.r2.x);
}

__device__ __forceinline__ Mat3 Inverse3x3(Mat3 m)
{
    float invDet = 1.0f / Det3x3(m);
    Mat3 adj;   // adjugate = transpose(cofactors); inverse = adj/det
    adj.r0 = make_float3( (m.r1.y * m.r2.z - m.r1.z * m.r2.y),
                         -(m.r0.y * m.r2.z - m.r0.z * m.r2.y),
                          (m.r0.y * m.r1.z - m.r0.z * m.r1.y));
    adj.r1 = make_float3(-(m.r1.x * m.r2.z - m.r1.z * m.r2.x),
                          (m.r0.x * m.r2.z - m.r0.z * m.r2.x),
                         -(m.r0.x * m.r1.z - m.r0.z * m.r1.x));
    adj.r2 = make_float3( (m.r1.x * m.r2.y - m.r1.y * m.r2.x),
                         -(m.r0.x * m.r2.y - m.r0.y * m.r2.x),
                          (m.r0.x * m.r1.y - m.r0.y * m.r1.x));
    Mat3 r;
    r.r0 = adj.r0 * invDet;
    r.r1 = adj.r1 * invDet;
    r.r2 = adj.r2 * invDet;
    return r;
}

__device__ __forceinline__ Mat3 Transpose3(Mat3 m)
{
    Mat3 t;
    t.r0 = make_float3(m.r0.x, m.r1.x, m.r2.x);
    t.r1 = make_float3(m.r0.y, m.r1.y, m.r2.y);
    t.r2 = make_float3(m.r0.z, m.r1.z, m.r2.z);
    return t;
}

// ── Device buffers (physics-owned). cornerPos is SHARED from dc_recon (recon_corner_pos()). ──
static float3*       d_vel         = nullptr;  // [cornerCount]
static float*        d_mass        = nullptr;  // [cornerCount]
static int*          d_pinned      = nullptr;  // [cornerCount]
static int*          d_active      = nullptr;  // [cornerCount]
static float3*       d_extForce    = nullptr;  // [cornerCount]
static int*          d_forceInt    = nullptr;  // [3*cornerCount]
static int*          d_nbrIdx      = nullptr;  // [6*cornerCount]
static float3*       d_restNbr     = nullptr;  // [6*cornerCount]
static BendPairGpu*  d_bendPairs   = nullptr;  // [bendPairCount]
static Slope*        d_kSlope      = nullptr;  // [7*cornerCount]
static float3*       d_yTrialPos   = nullptr;  // [cornerCount]
static float3*       d_yTrialVel   = nullptr;  // [cornerCount]
static unsigned int* d_errMax      = nullptr;  // [1]
static RotGpu*       d_particleRot = nullptr;  // [cornerCount]
static RotGpu*       d_particleRotPrev = nullptr;  // [cornerCount] previous-frame frames (Berndt [22] Fig 4c)

static PhysicsInitDesc g_pdesc = {};
static float           g_h = 1e-4f;   // DEAD under XPBD (was the RK45 adaptive substep size; assigned
                                       // from the ABI-placeholder hInit at init, never read since 9-(4))
static bool            g_phys_ready = false;

// ─────────────────────────────────────────────────────────────────────────────────────────────
// Kernels  - one-to-one ports of Physics.compute
// ─────────────────────────────────────────────────────────────────────────────────────────────

// ClearForces  - zero the fixed-point force accumulator.
__global__ void k_ClearForces(int cc, int* forceInt)
{
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= cc) return;
    forceInt[3 * p + 0] = 0; forceInt[3 * p + 1] = 0; forceInt[3 * p + 2] = 0;
}

// AccStructural  - Eq11 + DEVIATION D2 (subtract rest L0). Per-particle gather over 6 neighbors.
__global__ void k_AccStructural(int cc, float ks, float cs,
                                const float3* yTrialPos, const float3* yTrialVel,
                                const int* nbrIdx, const float3* restNbr, const int* active,
                                int* forceInt)
{
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= cc) return;

    float3 xi = yTrialPos[p];
    float3 vi = yTrialVel[p];
    float3 F  = make_float3(0.f, 0.f, 0.f);

    for (int s = 0; s < 6; s++)
    {
        int n = nbrIdx[6 * p + s];
        if (n < 0) continue;
        if (active[n] == 0) continue;        // skip spring across an active<->inactive edge (paper SS2.1.4)

        float  L0  = length3(restNbr[6 * p + s]);
        float3 d   = xi - yTrialPos[n];
        float  len = length3(d);
        if (len < 1e-8f) continue;

        float3 dir = d / len;
        float3 fs  = (-ks * (len - L0)) * dir - cs * (vi - yTrialVel[n]);   // DEVIATION D2
        F = F + fs;
    }

    AtomicAddForce(forceInt, p, F);
}

// AccBending  - Eq13-16. One thread per bending pair (i=center, j,k=arms).
__global__ void k_AccBending(int bpc, float kb, float cb,
                             const float3* yTrialPos, const float3* yTrialVel,
                             const BendPairGpu* bendPairs, int* forceInt)
{
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= bpc) return;

    BendPairGpu bp = bendPairs[n];
    if (bp.alive == 0) return;

    float3 xi = yTrialPos[bp.i], xj = yTrialPos[bp.j], xk = yTrialPos[bp.k];
    float3 vi = yTrialVel[bp.i], vj = yTrialVel[bp.j], vk = yTrialVel[bp.k];

    float3 Nij = normalize3(xi - xj);
    float3 Nik = normalize3(xi - xk);

    float c  = clampf(dot3(Nij, Nik), -1.0f, 1.0f);
    float s2 = 1.0f - c * c;
    if (s2 <= EPS_SIN2) return;

    float d     = sqrtf(s2);
    float theta = acosf(c);

    float3 dNij = UnitDeriv(xi, xj, vi, vj);
    float3 dNik = UnitDeriv(xi, xk, vi, vk);
    float  cdot = dot3(dNij, Nik) + dot3(Nij, dNik);
    float  theta_dot = -cdot / d;

    float scal = kb * (theta - bp.theta0) + cb * theta_dot;            // Eq14 scalar: k_b*(theta-theta0)+c_b*thetadot

    float3 F_ij = scal * cross3(cross3(Nik, Nij), Nij);                // Eq14: F_M_ij = scalar x (N_ik x N_ij) x N_ij
    float3 F_ik = scal * cross3(cross3(Nij, Nik), Nik);                // Eq14: F_M_ik = scalar x (N_ij x N_ik) x N_ik

    AtomicAddForce(forceInt, bp.j, F_ij);
    AtomicAddForce(forceInt, bp.k, F_ik);
    AtomicAddForce(forceInt, bp.i, (-1.0f) * (F_ij + F_ik));           // Eq15: F_b_in = -(F_M_ij+F_M_ik); cross-pair atomic accumulation realizes Eq16
}

// SeedTrialFromState  - copy committed y_n into trial buffers (stage 0).
__global__ void k_SeedTrial(int cc, const float3* cornerPos, const float3* vel,
                            float3* yTrialPos, float3* yTrialVel)
{
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= cc) return;
    yTrialPos[p] = cornerPos[p];
    yTrialVel[p] = vel[p];
}

// BuildTrialState  - yTrial = y_n + H * sum_j(DOPRI_A[stage][j] * k_j). Stages 1-6.
__global__ void k_BuildTrial(int cc, int stage, float H,
                             const float3* cornerPos, const float3* vel, const Slope* kSlope,
                             float3* yTrialPos, float3* yTrialVel)
{
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= cc) return;

    float3 dx = make_float3(0.f, 0.f, 0.f);
    float3 dv = make_float3(0.f, 0.f, 0.f);
    for (int j = 0; j < 6; j++)
    {
        float a = DOPRI_A[stage][j];
        if (a != 0.0f)
        {
            Slope k = kSlope[j * cc + p];
            dx = dx + a * k.dx;
            dv = dv + a * k.dv;
        }
    }
    yTrialPos[p] = cornerPos[p] + H * dx;
    yTrialVel[p] = vel[p]       + H * dv;
}

// ComputeSlope  - k[stage] = f(t, yTrial) = [yTrialVel, (ExtForce+ForceInt)/m] (DEVIATION D3).
// Pinned OR inactive (paper SS2.1.4) corners: zero slope (frozen).
__global__ void k_ComputeSlope(int cc, int stage, const int* pinned, const int* active,
                               const float3* yTrialVel, const float3* extForce,
                               const int* forceInt, const float* mass, float alpha, Slope* kSlope)
{
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= cc) return;

    Slope k;
    if (pinned[p] != 0 || active[p] == 0)
    {
        k.dx = make_float3(0.f, 0.f, 0.f);
        k.dv = make_float3(0.f, 0.f, 0.f);
    }
    else
    {
        k.dx = yTrialVel[p];
        float3 Fint = ReadForce(forceInt, p);
        // DEVIATION D3: a=(F_ext+F_int)/m.  PAPER-SILENT global mass-proportional (Rayleigh) damping
        // f_d = -alpha*m*v -> a -= alpha*v: the ONLY term that damps RIGID-body / pendulum / slosh modes
        // (Kelvin-Voigt cs/cb damp only RELATIVE velocity, so a rigid rotation about the pin is undamped).
        // Lets the soft liver SAG and SETTLE instead of swinging forever. alpha=0 -> exact paper behavior.
        k.dv = (extForce[p] + Fint) / mass[p] - alpha * yTrialVel[p];
    }
    kSlope[stage * cc + p] = k;
}

// IntegrateY5  - committed y_{n+1} = y_n + H * sum_i(DOPRI_B5[i] * k_i). Pinned/inactive frozen.
__global__ void k_IntegrateY5(int cc, float H, const int* pinned, const int* active,
                              const Slope* kSlope, float3* cornerPos, float3* vel)
{
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= cc) return;

    if (pinned[p] != 0 || active[p] == 0)
    {
        vel[p] = make_float3(0.f, 0.f, 0.f);   // keep position; zero velocity to prevent drift
        return;
    }

    float3 dx = make_float3(0.f, 0.f, 0.f);
    float3 dv = make_float3(0.f, 0.f, 0.f);
    for (int i = 0; i < 7; i++)
    {
        float b = DOPRI_B5[i];
        if (b != 0.0f)
        {
            Slope k = kSlope[i * cc + p];
            dx = dx + b * k.dx;
            dv = dv + b * k.dv;
        }
    }
    cornerPos[p] = cornerPos[p] + H * dx;
    vel[p]       = vel[p]       + H * dv;
}

// ComputeError  - per-particle max-norm of H * sum_i((B5-B4)_i * k_i); InterlockedMax reduction (Eq20).
__global__ void k_ComputeError(int cc, float H, const int* pinned, const int* active,
                               const Slope* kSlope, unsigned int* errMax)
{
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= cc) return;
    if (pinned[p] != 0 || active[p] == 0) return;   // frozen corners must not inflate the error norm

    float3 edx = make_float3(0.f, 0.f, 0.f);
    float3 edv = make_float3(0.f, 0.f, 0.f);
    for (int i = 0; i < 7; i++)
    {
        float db = DOPRI_B5[i] - DOPRI_B4[i];
        if (db != 0.0f)
        {
            Slope k = kSlope[i * cc + p];
            edx = edx + db * k.dx;
            edv = edv + db * k.dv;
        }
    }
    float e = H * fmaxf(
        fmaxf(fmaxf(fabsf(edx.x), fabsf(edx.y)), fabsf(edx.z)),
        fmaxf(fmaxf(fabsf(edv.x), fabsf(edv.y)), fabsf(edv.z)));
    atomicMax(&errMax[0], (unsigned int)(e * ERR_SCALE));
}

// component accessor + world direction of a frame's local axis k (= column k of R, rows r0/r1/r2).
__device__ __forceinline__ float compF3(float3 v, int k) { return (k == 0) ? v.x : ((k == 1) ? v.y : v.z); }
__device__ __forceinline__ float3 RotColumn(const RotGpu& R, int k)
{
    return make_float3(compF3(R.r0, k), compF3(R.r1, k), compF3(R.r2, k));
}

// ComputeParticleRot  - the particle coordinate frame per Berndt [22] Fig 4 (paper SS2.1.2 "The calculation
// of the new coordinate system is based on the method described in [22]"). For each of the 3 axes:
//   (b) a direction from the CURRENT neighbour-edge vectors (the through-particle edge +k neighbour minus
//       -k neighbour; one-sided when only one neighbour survives);
//   (c) blended (averaged) with the neighbours' PREVIOUS-frame axis-k (temporal/spatial smoothing that
//       "avoids abrupt visual perturbation"  - the reason [22] keeps the iso-surface on the same cell edges);
//   (d) when BOTH neighbours of an axis were removed by cutting, that axis "cannot be computed accurately"
//       from edges, so it is recovered as the CROSS PRODUCT of the other two axes (Fig 4d).
// The three axes then form matrix M (columns = the axes); R = polar(M) is the nearest proper rotation, so
// the cut-point record/reconstruct round-trip (R^T then R, SS2.1.2) stays valid. pinned/inactive or a
// degenerate/reflected frame (>=2 axes lost, det<=eps) -> identity.
__global__ void k_ComputeParticleRot(int cc, const float3* cornerPos, const int* nbrIdx,
                                     const int* active, const int* pinned,
                                     const RotGpu* particleRotPrev, RotGpu* particleRot)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= cc) return;

    RotGpu identity;
    identity.r0 = make_float3(1, 0, 0); identity.r1 = make_float3(0, 1, 0); identity.r2 = make_float3(0, 0, 1);

    if (pinned[c] != 0 || active[c] == 0) { particleRot[c] = identity; return; }

    float3 cPos = cornerPos[c];
    float3 axis[3]; bool have[3] = { false, false, false };

    for (int k = 0; k < 3; k++)
    {
        int np = nbrIdx[6 * c + (2 * k + 1)];   // +k neighbour (slot 2k+1)
        int nm = nbrIdx[6 * c + (2 * k)];       // -k neighbour (slot 2k)
        bool okp = (np >= 0 && active[np] != 0);
        bool okm = (nm >= 0 && active[nm] != 0);

        float3 e;
        if (okp && okm)      e = cornerPos[np] - cornerPos[nm];   // (b) through-particle edge, points +k
        else if (okp)        e = cornerPos[np] - cPos;            // one-sided +k
        else if (okm)        e = cPos - cornerPos[nm];            // one-sided +k
        else                 continue;                            // both gone -> (d) cross-product later
        float len = length3(e);
        if (len < 1e-12f) continue;
        float3 dir = e / len;

        // (c) average with the neighbours' previous-frame axis-k (world column k), all surviving neighbours.
        float3 psum = make_float3(0.f, 0.f, 0.f); int pc = 0;
        for (int s = 0; s < 6; s++)
        {
            int n = nbrIdx[6 * c + s];
            if (n >= 0 && active[n] != 0) { psum = psum + RotColumn(particleRotPrev[n], k); pc++; }
        }
        if (pc > 0)
        {
            float pl = length3(psum);
            if (pl > 1e-6f)
            {
                float3 blended = dir + psum / pl;      // average of current edge dir + mean prev axis (both unit)
                float dl = length3(blended);
                // Robustness: only accept the blend when it is well-formed. Near-cancellation (a ~180deg
                // frame flip between frames) would normalize a tiny vector into a GARBAGE direction, so in
                // that case trust the current-edge direction instead.
                if (dl > 0.25f) dir = blended / dl;
            }
        }
        axis[k] = dir; have[k] = true;
    }

    // (d) recover a single fully-severed axis from the cross product of the other two (right-handed).
    int miss = -1, mc = 0;
    for (int k = 0; k < 3; k++) if (!have[k]) { miss = k; mc++; }
    if (mc == 1)
    {
        float3 a = axis[(miss + 1) % 3], b = axis[(miss + 2) % 3];
        float3 cx = cross3(a, b); float l = length3(cx);
        if (l > 1e-9f) { axis[miss] = cx / l; have[miss] = true; }
    }
    if (!(have[0] && have[1] && have[2])) { particleRot[c] = identity; return; }

    // Matrix M has the three axes as COLUMNS: column k = (M.r0[k], M.r1[k], M.r2[k]) = axis[k]. R = polar(M)
    // is then the nearest proper rotation whose column k is the world direction of local axis k (what
    // RotColumn / mulRv read). det(M)<=eps rejects a reflected/near-collapsed (axes ~parallel) frame.
    Mat3 M;
    M.r0 = make_float3(axis[0].x, axis[1].x, axis[2].x);
    M.r1 = make_float3(axis[0].y, axis[1].y, axis[2].y);
    M.r2 = make_float3(axis[0].z, axis[1].z, axis[2].z);
    if (Det3x3(M) <= 1e-4f) { particleRot[c] = identity; return; }

    // R = polar(M) via Higham R_{k+1}=0.5*(R+R^-T). 16 iters converges even for skewed M (det~0.02);
    // 6 was insufficient (||R^T R - I|| ~ 0.1 near det~0.03) and would break the R^T/R round-trip.
    Mat3 R = M;
    for (int it = 0; it < 16; it++)
    {
        Mat3 RinvT = Transpose3(Inverse3x3(R));
        R.r0 = 0.5f * (R.r0 + RinvT.r0);
        R.r1 = 0.5f * (R.r1 + RinvT.r1);
        R.r2 = 0.5f * (R.r2 + RinvT.r2);
    }

    // Orthonormality safety: if Higham did not fully converge (pathological M), fall back to identity so
    // EmitCutPoint/AccumulateCutFP keep an exact R^T==R^-1 round-trip (SS2.1.2). Columns of R = c0,c1,c2.
    float3 c0 = make_float3(R.r0.x, R.r1.x, R.r2.x);
    float3 c1 = make_float3(R.r0.y, R.r1.y, R.r2.y);
    float3 c2 = make_float3(R.r0.z, R.r1.z, R.r2.z);
    float ortho = fmaxf(fmaxf(fabsf(dot3(c0, c0) - 1.f), fabsf(dot3(c1, c1) - 1.f)), fabsf(dot3(c2, c2) - 1.f));
    ortho = fmaxf(ortho, fmaxf(fabsf(dot3(c0, c1)), fmaxf(fabsf(dot3(c0, c2)), fabsf(dot3(c1, c2)))));
    if (ortho > 1e-3f) { particleRot[c] = identity; return; }

    RotGpu rg; rg.r0 = R.r0; rg.r1 = R.r1; rg.r2 = R.r2;
    particleRot[c] = rg;
}

// ─────────────────────────────────────────────────────────────────────────────────────────────
// Host API
// ─────────────────────────────────────────────────────────────────────────────────────────────
static int Gp(int n) { int g = (n + 63) / 64; return g < 1 ? 1 : g; }

#define PK(call) do { cudaError_t _e = (call); if (_e != cudaSuccess) { physics_shutdown(); return -(int)_e; } } while (0)

int physics_init(const PhysicsInitDesc* desc,
                 const float* mass, const int* pinned, const int* active, const float* extForce,
                 const int* nbrIdx, const float* restNbr, const void* bendPairs)
{
    if (g_phys_ready) physics_shutdown();
    if (!desc) return -2000;

    g_pdesc = *desc;
    int cc  = desc->cornerCount;
    int bpc = desc->bendPairCount;
    if (cc <= 0) return -2001;
    if (cc != recon_corner_count()) return -2002;   // must share dc_recon's grid
    g_h = (desc->hInit > 0.f) ? desc->hInit : 1e-4f;

    int bpcA = bpc > 0 ? bpc : 1;

    PK(cudaMalloc(&d_vel,         (size_t)cc      * sizeof(float3)));
    PK(cudaMalloc(&d_mass,        (size_t)cc      * sizeof(float)));
    PK(cudaMalloc(&d_pinned,      (size_t)cc      * sizeof(int)));
    PK(cudaMalloc(&d_active,      (size_t)cc      * sizeof(int)));
    PK(cudaMalloc(&d_extForce,    (size_t)cc      * sizeof(float3)));
    PK(cudaMalloc(&d_forceInt,    (size_t)3 * cc  * sizeof(int)));
    PK(cudaMalloc(&d_nbrIdx,      (size_t)6 * cc  * sizeof(int)));
    PK(cudaMalloc(&d_restNbr,     (size_t)6 * cc  * sizeof(float3)));
    PK(cudaMalloc(&d_bendPairs,   (size_t)bpcA    * sizeof(BendPairGpu)));
    PK(cudaMalloc(&d_kSlope,      (size_t)7 * cc  * sizeof(Slope)));
    PK(cudaMalloc(&d_yTrialPos,   (size_t)cc      * sizeof(float3)));
    PK(cudaMalloc(&d_yTrialVel,   (size_t)cc      * sizeof(float3)));
    PK(cudaMalloc(&d_errMax,                        sizeof(unsigned int)));
    PK(cudaMalloc(&d_particleRot,     (size_t)cc  * sizeof(RotGpu)));
    PK(cudaMalloc(&d_particleRotPrev, (size_t)cc  * sizeof(RotGpu)));

    // Upload topology / properties.
    PK(cudaMemcpy(d_mass,     mass,     (size_t)cc     * sizeof(float),  cudaMemcpyHostToDevice));
    PK(cudaMemcpy(d_pinned,   pinned,   (size_t)cc     * sizeof(int),    cudaMemcpyHostToDevice));
    PK(cudaMemcpy(d_active,   active,   (size_t)cc     * sizeof(int),    cudaMemcpyHostToDevice));
    PK(cudaMemcpy(d_extForce, extForce, (size_t)cc     * sizeof(float3), cudaMemcpyHostToDevice));
    PK(cudaMemcpy(d_nbrIdx,   nbrIdx,   (size_t)6 * cc * sizeof(int),    cudaMemcpyHostToDevice));
    PK(cudaMemcpy(d_restNbr,  restNbr,  (size_t)6 * cc * sizeof(float3), cudaMemcpyHostToDevice));
    if (bpc > 0)
        PK(cudaMemcpy(d_bendPairs, bendPairs, (size_t)bpc * sizeof(BendPairGpu), cudaMemcpyHostToDevice));

    // Zero velocities + scratch (particles start at rest).
    PK(cudaMemset(d_vel,       0, (size_t)cc     * sizeof(float3)));
    PK(cudaMemset(d_forceInt,  0, (size_t)3 * cc * sizeof(int)));
    PK(cudaMemset(d_kSlope,    0, (size_t)7 * cc * sizeof(Slope)));
    PK(cudaMemset(d_yTrialPos, 0, (size_t)cc     * sizeof(float3)));
    PK(cudaMemset(d_yTrialVel, 0, (size_t)cc     * sizeof(float3)));
    PK(cudaMemset(d_errMax,    0,                  sizeof(unsigned int)));

    // ParticleRot = identity (persists; consumed at Stage 3).
    RotGpu* idn = (RotGpu*)malloc((size_t)cc * sizeof(RotGpu));
    if (!idn) { physics_shutdown(); return -2003; }
    for (int i = 0; i < cc; i++)
    {
        idn[i].r0 = make_float3(1, 0, 0); idn[i].r1 = make_float3(0, 1, 0); idn[i].r2 = make_float3(0, 0, 1);
    }
    cudaError_t e = cudaMemcpy(d_particleRot, idn, (size_t)cc * sizeof(RotGpu), cudaMemcpyHostToDevice);
    if (e == cudaSuccess)
        e = cudaMemcpy(d_particleRotPrev, idn, (size_t)cc * sizeof(RotGpu), cudaMemcpyHostToDevice);   // [22] Fig 4c seed = identity
    free(idn);
    if (e != cudaSuccess) { physics_shutdown(); return -(int)e; }

    // ── XPBD GO-LIVE (design rev-B §9-(4)): configure + bind the XPBD solver to the SHARED buffers. ──
    // Params map from the SAME PhysicsInitDesc the RK45 consumed (C# StructLayout ABI unchanged):
    //   numSubsteps = maxSubsteps REPURPOSED as the fixed small-steps count N (hInit/hMax are DEAD);
    //   betaS/betaB = the paper's cs/cb  - Kelvin-Voigt dashpot STIFFNESSES, used DIRECTLY (§4, NOT c/k);
    //   alpha       = the same PAPER-SILENT Rayleigh drag k_ComputeSlope applied (implicit 1/(1+a*h));
    //   damping     = 0 (no legacy scalar velocity kill on the live path  - alpha + cs/cb damp);
    //   iters=1, omega=1  - the steps-1-3-VALIDATED configuration for THIS constraint set (structural +
    //   bending + damping; §3.5 mandates re-evaluating omega before the full constraint set);
    //   gx/gy/gz = 0  - in BOUND mode gravity enters predict per-corner as extForce[c]/mass[c] (the
    //   same d_extForce the RK45 slope consumed).
    XpbdParams xp = {};
    xp.numSubsteps = (g_pdesc.maxSubsteps > 0) ? g_pdesc.maxSubsteps : 20;
    xp.iters   = 1;
    xp.omega   = 1.0f;
    xp.damping = 0.f;
    xp.ks      = g_pdesc.ks;
    xp.gx = 0.f; xp.gy = 0.f; xp.gz = 0.f;
    xp.kb      = g_pdesc.kb;
    xp.alpha   = g_pdesc.alpha;
    xp.betaS   = g_pdesc.cs;
    xp.betaB   = g_pdesc.cb;
    // STEP 7 (design rev-B §4[Strain]/§9-(7), §11 item 3): per-tet Stable Neo-Hookean material. E is a
    // STARTING ANCHOR (same order as the paper's ks=7.5e4 spring lattice, kept softer so the volumetric
    // term augments rather than double-stiffens the existing springs)  - a §11 USER DECISION to be tuned
    // against the video once the tent is visible; nu=0.45 is the design-pinned near-incompressible start.
    // Neither is C#-marshalled (XpbdParams is a native-only struct, unlike the ABI-frozen
    // PhysicsInitDesc)  - tune here, not via the C# side, until a real tuning UI is wanted.
    xp.youngE  = 3.0e4f;
    xp.poisson = 0.45f;
    xpbd_set_params(&xp);
    int dx = 0, dy = 0, dz = 0;
    recon_dims(&dx, &dy, &dz);
    int xrc = xpbd_bind(recon_corner_pos(), d_vel,
                        d_nbrIdx, d_restNbr, d_bendPairs, bpc,
                        d_active, d_pinned, d_mass,
                        d_extForce,
                        dx, dy, dz, cc);
    if (xrc != 0) { physics_shutdown(); return xrc; }

    // STEP 7 GO-LIVE: build the tet mesh from the just-bound live buffers (dims/cornerPos/active  - the
    // REST configuration, since no substep has run yet). Tet CUT/TEAR lifecycle (design §4[Strain]) is
    // derived every substep by k_RefreshLiveness from the SAME live nbrIdx/active flags edges/bending
    // already read  - zero cut.cu changes, exactly mirroring the sever-propagation mechanism §9-(4)
    // established for edges/bending.
    int trc = xpbd_build_tets();
    if (trc != 0) { physics_shutdown(); return trc; }

    g_phys_ready = true;
    return 0;
}

int physics_step(float dt)
{
    if (!g_phys_ready) return -1;
    if (!recon_corner_pos()) return -2;

    // ── XPBD small-steps body (design rev-B §9-(4) rebind + §9-(6) cut-tick interleave). The RK45
    // adaptive-h loop is RETIRED (deviation-ledger D3/D4/D6 VOID, superseded by XPBD small-steps
    // substepping  - the solver stays inside the paper's own cited PBD family, refs [7]/[22]).
    // physics_step OWNS the N-loop so the Stage-6 CCD cut tick fires BETWEEN substeps (§9-(6)) exactly
    // as it rode the RK45 substeps: one tick per committed sub-interval, post-position-commit, with the
    // d_prevCorner roll inside the tick (cut.cu). The solve projects the SHARED recon_corner_pos() in
    // place and writes d_vel via the bound module (xpbd_bind in physics_init)  - recon/cut/
    // particleRot all read the SAME deformed buffer.
    //   N = maxSubsteps, REPURPOSED as the fixed small-steps count (== the module's numSubsteps, from the
    // same field in physics_init). Design §9-(6): N = max(N_cut, N_elastic). N_cut (no-tunnel: one
    // substep's TISSUE travel < voxelL, worst case fast-falling severed debris)  - [14, 200] at
    // physicsDt=0.02 (bounds map from the old adaptive g_h∈[1e-4, 1.5e-3]); N_elastic (convergence) ~ 10.
    // The shipped N=20 (h=1e-3, near the old hMax) sits at N_cut's low end  - a sound first live value;
    // raise maxSubstepsPerFrame if fast debris tunnels past the per-substep CCD tick. PATH B (the
    // rod-phase quad, cut_detect between LCS_Step and LCS_Finalize) runs per C# rod substep, independent
    // of N.
    int rc = xpbd_begin_frame(dt);
    if (rc != 0) return rc;
    int N = (g_pdesc.maxSubsteps > 0) ? g_pdesc.maxSubsteps : 20;
    int tickErr = 0;   // first ribbon-tick failure (surfaced after the loop  - review R3-m4)
    for (int n = 0; n < N; n++)
    {
        rc = xpbd_substep();
        if (rc != 0) return rc;

        // Stage 6 CCD tick: continuous collision of every moving deformed edge vs the static blade
        // segment after EVERY committed substep  - exact space-time crossings, so fast tissue motion
        // under gravity cannot tunnel between per-frame snapshots. No-op until cutting is active.
        if (cut_is_ready())
        {
            int r = cut_ribbon_tick();
            if (r != 0 && tickErr == 0) tickErr = r;
        }
    }

    // NOTE: ComputeParticleRot lives in physics_compute_rotations(). Stage 3 calls it once after physics
    // for pre-cut material frames, and again after detect+sever so cut-FP reconstruction uses post-sever
    // connectivity.
    cudaError_t le = cudaGetLastError();
    if (le == cudaSuccess) le = cudaDeviceSynchronize();   // commit the frame (replaces the errMax sync)
    if (le != cudaSuccess) return -(int)le;
    if (tickErr != 0) return tickErr;   // surface ribbon-tick failures to C# (review R3-m4)
    return 0;
}

// physics_compute_rotations  - k_ComputeParticleRot: Berndt [22] Fig 4 particle frame (paper SS2.1.2).
// Snapshots the current frames into d_particleRotPrev first (Fig 4c previous-orientation average). Stage 3
// uses it both before DetectCut (material-frame hit recording) and after SeverLinks (cut-FP reconstruction).
int physics_compute_rotations()
{
    if (!g_phys_ready) return -1;
    int cc = g_pdesc.cornerCount;
    float3* cornerPos = recon_corner_pos();
    if (!cornerPos) return -2;
    // Snapshot the current frames as "previous" for [22] Fig 4c (neighbours' previous-orientation average).
    cudaError_t ce = cudaMemcpy(d_particleRotPrev, d_particleRot, (size_t)cc * sizeof(RotGpu), cudaMemcpyDeviceToDevice);
    if (ce != cudaSuccess) return -(int)ce;
    k_ComputeParticleRot<<<Gp(cc), 64>>>(cc, cornerPos, d_nbrIdx, d_active, d_pinned, d_particleRotPrev, d_particleRot);
    cudaError_t e = cudaGetLastError();
    return (e == cudaSuccess) ? 0 : -(int)e;
}

// Stage 3 shared accessors (read by cut.cu): particleRot (RotGpu*), nbrIdx (int*), bendPairs (BendPairGpu*).
void* physics_particle_rot() { return d_particleRot; }
int*  physics_nbr_idx()      { return d_nbrIdx; }
void* physics_bend_pairs()   { return d_bendPairs; }
int*  physics_active()       { return d_active; }   // v4.1 P1b: cut.cu freezes fully-severed debris
int*  physics_pinned()       { return d_pinned; }
void* physics_vel()          { return d_vel; }

// Diagnostic readback: copy the LIVE (post-sever) nbrIdx[6*cornerCount] to host. Returns 0 on success.
int physics_readback_nbr_idx(int* hostOut)
{
    if (!g_phys_ready || !hostOut) return -1;
    int cc = g_pdesc.cornerCount;
    cudaError_t e = cudaMemcpy(hostOut, d_nbrIdx, (size_t)6 * cc * sizeof(int), cudaMemcpyDeviceToHost);
    return (e == cudaSuccess) ? 0 : -(int)e;
}

void physics_shutdown()
{
    xpbd_shutdown();   // unbind FIRST: the XPBD module aliases cornerPos/d_vel and reads these buffers
    cudaFree(d_vel);         d_vel = nullptr;
    cudaFree(d_mass);        d_mass = nullptr;
    cudaFree(d_pinned);      d_pinned = nullptr;
    cudaFree(d_active);      d_active = nullptr;
    cudaFree(d_extForce);    d_extForce = nullptr;
    cudaFree(d_forceInt);    d_forceInt = nullptr;
    cudaFree(d_nbrIdx);      d_nbrIdx = nullptr;
    cudaFree(d_restNbr);     d_restNbr = nullptr;
    cudaFree(d_bendPairs);   d_bendPairs = nullptr;
    cudaFree(d_kSlope);      d_kSlope = nullptr;
    cudaFree(d_yTrialPos);   d_yTrialPos = nullptr;
    cudaFree(d_yTrialVel);   d_yTrialVel = nullptr;
    cudaFree(d_errMax);      d_errMax = nullptr;
    cudaFree(d_particleRot); d_particleRot = nullptr;
    cudaFree(d_particleRotPrev); d_particleRotPrev = nullptr;
    g_phys_ready = false;
}
