// xpbd_solver.cu  - XPBD soft-body solver, migration STEP 1 (structural distance + gravity + pinning).
// See xpbd_solver.h. Self-contained; does NOT touch physics.cu / the mass-spring solver.

#include "xpbd_solver.h"
#include <cuda_runtime.h>
#include <vector>
#include <cstring>
#include <cmath>

// ── minimal float3 helpers (kept local so this module is independent of common.cuh) ─────────────
__host__ __device__ __forceinline__ float3 mkf3(float x, float y, float z) { float3 r; r.x=x; r.y=y; r.z=z; return r; }
__host__ __device__ __forceinline__ float3 operator+(float3 a, float3 b) { return mkf3(a.x+b.x, a.y+b.y, a.z+b.z); }
__host__ __device__ __forceinline__ float3 operator-(float3 a, float3 b) { return mkf3(a.x-b.x, a.y-b.y, a.z-b.z); }
__host__ __device__ __forceinline__ float3 operator*(float3 a, float s)   { return mkf3(a.x*s, a.y*s, a.z*s); }
__host__ __device__ __forceinline__ float  dot3(float3 a, float3 b) { return a.x*b.x + a.y*b.y + a.z*b.z; }
__host__ __device__ __forceinline__ float  len3(float3 a)           { return sqrtf(dot3(a,a)); }
__host__ __device__ __forceinline__ float3 cross3(float3 a, float3 b) { return mkf3(a.y*b.z-a.z*b.y, a.z*b.x-a.x*b.z, a.x*b.y-a.y*b.x); }
__host__ __device__ __forceinline__ float  compF3(float3 v, int k) { return (k==0) ? v.x : ((k==1) ? v.y : v.z); }

// ── device buffers (SoA) ────────────────────────────────────────────────────────────────────────
static float3* d_pos      = nullptr;   // [N] current positions
static float3* d_prevPos  = nullptr;   // [N] positions at substep start (for v = (x-x_prev)/h)
static float3* d_vel      = nullptr;   // [N] velocities
static float*  d_invMass  = nullptr;   // [N] inverse mass (0 = pinned)
static int*    d_edgeI    = nullptr;   // [E] colour-ordered edge endpoint i
static int*    d_edgeJ    = nullptr;   // [E] colour-ordered edge endpoint j
static float*  d_restLen  = nullptr;   // [E] colour-ordered rest length
static float*  d_lambda   = nullptr;   // [E] colour-ordered per-edge XPBD multiplier (reset each substep)
static int*    d_edgeActive = nullptr; // [E] colour-ordered per-edge active flag (1=live, 0=severed).
                                       // STANDALONE mode: the notify API flips it. BOUND mode:
                                       // k_RefreshLiveness RE-DERIVES it every substep from the live
                                       // nbrIdx/active. The greedy COLOURING stays static either way.

static int     g_N = 0, g_E = 0;
static std::vector<int> g_colorOff, g_colorCnt;   // host: [numColors] flat-range per colour
static int     g_numColors = 0;
static bool    g_ready = false;

// M1 (design §9-(4)): the input-edge -> colour-slot PERMUTATION is PERSISTED, not discarded, so a
// runtime sever can be propagated to the colour-ordered d_edgeActive. The host inverse map serves the
// xpbd_set_*_active notify API (standalone self-test); in BOUND mode k_RefreshLiveness supersedes the
// notify path, deriving liveness from the LIVE shared nbrIdx/bendPairs every substep (d_bendSrcIdx then
// holds LIVE bendPairs indices, uploaded by xpbd_bind).
static int* d_edgeSrcIdx = nullptr;               // [E] colour-slot -> original input edge index
static std::vector<int> g_edgeSlotOfInput;        // host: input edge index -> colour slot

// M2 (design §9-(4)): frame-scope coefficients computed ONCE by xpbd_begin_frame so the caller
// (eventually physics_step) can own the N-loop and interleave cut_ribbon_tick between substeps.
static float g_hFrame = 0.f, g_alphaTildeF = 0.f, g_velScaleF = 1.f, g_betaGammaHSF = 0.f;
static float g_alphaBF = 0.f, g_betaGammaHBF = 0.f;
static int   g_itersF = 1;
static bool  g_doBendF = false, g_frameReady = false;
static bool  g_doStrainF = false;   // STEP 7: true iff tets exist AND youngE>0 (recomputed each begin_frame)

// STEP 2  - bending elements (paper Eq13-16 as XPBD angle constraints). Colour-ordered like the edges.
static int*    d_bendI = nullptr, *d_bendJ = nullptr, *d_bendK = nullptr;  // [B] apex + two arms
static float*  d_bendTheta0 = nullptr;   // [B] rest angle theta0
static int*    d_bendActive = nullptr;   // [B] active flag (1=live; a cut/tear flips via the notify API)
static float*  d_bendLambda = nullptr;   // [B] XPBD multiplier (reset per substep)
static int     g_B = 0;
static std::vector<int> g_bColorOff, g_bColorCnt;
static int     g_bNumColors = 0;
static int*    d_bendSrcIdx = nullptr;            // [B] colour-slot -> original input element index (M1)
static std::vector<int> g_bendSlotOfInput;        // host: input element index -> colour slot

// STEP 7  - per-tet Stable Neo-Hookean strain (design rev-B §4[Strain]/§9-(7)). Colour-ordered like
// edges/bending. BOUND MODE ONLY: built once by xpbd_build_tets from the live cornerPos/active/nbrIdx
// dims captured at xpbd_bind. d_tetActive is NOT written by a notify path  - it is RE-DERIVED every
// substep by k_RefreshLiveness from the SAME live nbrIdx/active flags (a tet's 3 member axis-edge owner
// slots + its 4 corners), so it is naturally permanent (nbrIdx never un-severs).
static int*    d_tetIds     = nullptr;   // [4*T] corner ids (p0,p1,p2,p3), colour-ordered
static float3* d_tetBr0     = nullptr;   // [T] Dm^-1 row 0 (rest inverse matrix, baked at build time)
static float3* d_tetBr1     = nullptr;   // [T] Dm^-1 row 1
static float3* d_tetBr2     = nullptr;   // [T] Dm^-1 row 2
static float*  d_tetMuV     = nullptr;   // [T] mu   * RestVolume  (material x rest-volume, h-independent)
static float*  d_tetLambdaV = nullptr;   // [T] lambda * RestVolume
static float*  d_tetAlphaDev= nullptr;   // [T] 1/(h^2*mu*V)      - recomputed once per xpbd_begin_frame
static float*  d_tetAlphaHyd= nullptr;   // [T] 1/(h^2*lambda*V)  - recomputed once per xpbd_begin_frame
static float*  d_tetLambdaDevL = nullptr; // [T] XPBD multiplier, deviatoric (reset each substep)
static float*  d_tetLambdaHydL = nullptr; // [T] XPBD multiplier, hydrostatic (reset each substep)
static int*    d_tetActive  = nullptr;   // [T] 1=live (re-derived every substep by k_RefreshLiveness)
static int*    d_tetEdgeOwnerSlot = nullptr; // [3*T] the tet's 3 member AXIS-edge owner slots (6p+s)
static int     g_T = 0;
static std::vector<int> g_tColorOff, g_tColorCnt;
static int     g_tNumColors = 0;
static bool    g_haveTets = false;

static XpbdParams g_p = { /*numSubsteps*/20, /*iters*/1, /*omega*/1.0f, /*damping*/0.02f,
                          /*ks*/7.5e4f, /*gx*/0.f, /*gy*/-9.81f, /*gz*/0.f, /*kb*/2.0e4f,
                          /*alpha*/0.f, /*betaS*/0.f, /*betaB*/0.f,
                          /*youngE*/0.f, /*poisson*/0.45f };

// ── REBIND state (design rev-B §9-(4)). In BOUND mode d_pos/d_vel ALIAS the shared live buffers
// (recon_corner_pos() / physics_vel()) and are NOT owned by this module; liveness (edgeActive/
// bendActive) and invMass are DERIVED at the start of every substep by k_RefreshLiveness from the
// LIVE shared nbrIdx / bendPairs.alive / active / pinned / mass  - the SAME flags physics.cu's force
// gates read  - so a cut.cu sever (nbrIdx=-1, alive=0, active=0) is picked up the very next substep
// with ZERO cut.cu changes. The colouring stays STATIC (a dead edge just early-outs). ─────────────────
static bool  g_bound         = false;
static int*  d_edgeOwnerSlot = nullptr;   // [E] owner slot (6p + s, s=2*axis+1) per colour-ordered edge
static const int*             g_liveNbrIdx      = nullptr;   // [6*N] live topology (cut writes -1)
static const XpbdBendPairGpu* g_liveBendPairs   = nullptr;   // live bend pairs (cut writes alive=0)
static const int*             g_liveActive      = nullptr;   // [N] live active flags (cut freezes debris)
static const int*             g_livePinned      = nullptr;   // [N] anchored BC
static const float*           g_liveMass        = nullptr;   // [N] invMass derived, NEVER baked
static const float3*          g_liveExtForce    = nullptr;   // [N] external force
static int g_dimsX = 0, g_dimsY = 0, g_dimsZ = 0;            // carried for the step-7 tet builder

static inline int Gr(int n) { int g = (n + 63) / 64; return g < 1 ? 1 : g; }

// ── kernels ───────────────────────────────────────────────────────────────────────────────────
// Predict: save prev, integrate the external body forces into velocity, advance position. Pinned
// (w=0) stay put.
// STANDALONE mode (extForce == nullptr): the acceleration is the uniform gravity g from XpbdParams.
// BOUND mode uses the live external body-force sum divided by mass. Internal elastic forces are handled
// by the constraint solve, and a debris-frozen corner (invMass 0) is skipped identically.
__global__ void k_predict(int N, float h, float3 g, const float3* extForce,
                          const float* invMass, float3* pos, float3* prevPos, float3* vel)
{
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= N) return;
    prevPos[p] = pos[p];
    if (invMass[p] > 0.f) {
        float3 a;
        if (extForce) {
            a = extForce[p] * invMass[p];
        } else {
            a = g;                                     // standalone self-test: uniform gravity
        }
        float3 v = vel[p] + a * h;          // v += h * a
        vel[p] = v;
        pos[p] = pos[p] + v * h;            // x += h * v
    }
}

// RefreshLiveness (design §9-(4), BOUND mode only)  - run at the START of every substep, BEFORE predict,
// so predict and the sweeps see fresh liveness/mass. Derives, from the LIVE shared buffers:
//   edgeActive[e] = nbrIdx[edgeOwnerSlot[e]] >= 0  AND  active[i] && active[j]
//                   (k_SeverLinks writes BOTH half-slots, so the owner slot alone captures a sever;
//                    the active[] test is k_AccStructural's second gate (`if (active[n] == 0) continue;`)  - an edge to a
//                    frozen debris corner exerts NO force there, so the constraint must drop too)
//   bendActive[b] = bendPairs[bendSrcIdx[b]].alive   (k_AccBending's only gate (`if (bp.alive == 0) return;`)  - a
//                    frozen member is handled by its w=0, matching the RK45 force-discard semantics)
//   invMass[c]    = (pinned[c] || active[c]==0) ? 0 : 1/mass[c]   (NEVER baked: cut.cu:451 flips
//                    active[] at runtime when it freezes fully-severed debris)
// One launch over max(N, E, B); no atomics (disjoint writes).
// STEP 7 addendum: a 4th domain T re-derives d_tetActive every substep from the SAME live nbrIdx/active
// flags  - a tet dies PERMANENTLY the substep any of its 3 member axis-edge owner slots is severed
// (tetEdgeOwnerSlot[3t+0..2], the 6p+s convention) or any of its 4 corners (tetIds[4t+0..3]) goes
// inactive. Zero cut.cu changes, exactly mirroring the edge/bend mechanism above.
__global__ void k_RefreshLiveness(int N, int E, int B, int T,
                                  const int* nbrIdx, const int* edgeOwnerSlot,
                                  const int* edgeI, const int* edgeJ,
                                  const XpbdBendPairGpu* bendPairs, const int* bendSrcIdx,
                                  const int* active, const int* pinned, const float* mass,
                                  const int* tetIds, const int* tetEdgeOwnerSlot,
                                  int* edgeActive, int* bendActive, float* invMass, int* tetActive)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t < E)
        edgeActive[t] = (nbrIdx[edgeOwnerSlot[t]] >= 0 &&
                         active[edgeI[t]] != 0 && active[edgeJ[t]] != 0) ? 1 : 0;
    if (t < B)
        bendActive[t] = (bendPairs[bendSrcIdx[t]].alive != 0) ? 1 : 0;
    if (t < N)
        invMass[t] = (pinned[t] != 0 || active[t] == 0) ? 0.f : (1.f / mass[t]);
    if (t < T) {
        bool live = nbrIdx[tetEdgeOwnerSlot[3*t+0]] >= 0 &&
                    nbrIdx[tetEdgeOwnerSlot[3*t+1]] >= 0 &&
                    nbrIdx[tetEdgeOwnerSlot[3*t+2]] >= 0 &&
                    active[tetIds[4*t+0]] != 0 && active[tetIds[4*t+1]] != 0 &&
                    active[tetIds[4*t+2]] != 0 && active[tetIds[4*t+3]] != 0;
        tetActive[t] = live ? 1 : 0;
    }
}

// Solve one COLOUR of structural distance constraints. Within a colour no two edges share a vertex,
// so each position is written by at most one edge -> true Gauss-Seidel, NO atomics.
// C = |xi-xj| - L0 ; a~ = (1/ks)/h^2 ; dLambda = (-C - a~*lambda)/(wi+wj+a~) ; xi += w*omega*dLambda*n.
__global__ void k_solve_edges(int colorOff, int colorCnt, float alphaTilde, float omega,
                              float betaGammaH, const float3* prevPos,
                              const int* edgeI, const int* edgeJ, const float* restLen,
                              const int* edgeActive, const float* invMass, float* lambda, float3* pos)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= colorCnt) return;
    int e  = colorOff + t;
    if (edgeActive[e] == 0) return;               // severed/inactive (step-5 cut): drop the constraint; colouring stays static
    int i  = edgeI[e], j = edgeJ[e];
    float wi = invMass[i], wj = invMass[j];
    float wsum = wi + wj;
    if (wsum <= 0.f) return;                      // both endpoints pinned
    float3 d = pos[i] - pos[j];
    float  L = len3(d);
    if (L < 1e-9f) return;                        // coincident: undefined gradient, skip
    float3 n = d * (1.f / L);
    float  C = L - restLen[e];
    // STEP 3 Kelvin-Voigt damping (Macklin XPBD 2016): gamma = a~ * beta * h (betaGammaH is a~*betaS*h,
    // precomputed on the host; beta = cs is the dashpot STIFFNESS, not a compliance). grad C . (x - x_prev)
    // is the relative displacement along the spring this substep; the damped constraint reproduces the
    // dashpot force f = cs*(relative velocity along the spring). betaGammaH=0 -> the undamped form exactly.
    float  gamma  = betaGammaH;
    float  gradDx = dot3(n, (pos[i] - prevPos[i]) - (pos[j] - prevPos[j]));
    float  dLambda = (-C - alphaTilde * lambda[e] - gamma * gradDx) / ((1.f + gamma) * wsum + alphaTilde);
    // SOR under-relaxation applied CONSISTENTLY to BOTH lambda and position, so the compliant fixed point
    // C = -a~*lambda (=> DL = mg/ks) is omega-INVARIANT (omega only changes convergence rate, not the
    // equilibrium stiffness). omega=1 is exact per-iteration; <1 trades speed for stability.
    float  applied = omega * dLambda;
    lambda[e] += applied;
    pos[i] = pos[i] + n * (wi * applied);
    pos[j] = pos[j] - n * (wj * applied);
}

// Solve one COLOUR of BENDING angle constraints  - the paper's Eq13-16 (k_AccBending) as XPBD. C = theta -
// theta0, theta = angle at apex i between (xi-xj) and (xi-xk). The gradient directions are the paper's Eq14
// FORCE directions (nv - c*nu) / (nu - c*nv)  - i.e. the angle gradient WITHOUT the 1/(sin*|arm|) factor.
// This reproduces the paper's SOFTER, geometry-scaled bending (effective force magnitude
// kb*(theta-theta0)*sin(theta)) so the XPBD rest shape MATCHES the RK45 golden (verified: a clamped
// cantilever's droop matches to <1% vs 50% for the clean 1/sin gradient, which is ~10x too stiff at these
// arm lengths and would over-resist the local deformation). |g|=sin(theta) so it is also NON-singular. Within a
// colour no two bend elements share a vertex -> no atomics.
__global__ void k_solve_bending(int colorOff, int colorCnt, float alphaB, float omega,
                                float betaGammaH, const float3* prevPos,
                                const int* bI, const int* bJ, const int* bK, const float* theta0,
                                const int* bActive, const float* invMass, float* lambda, float3* pos)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= colorCnt) return;
    int e = colorOff + t;
    if (bActive[e] == 0) return;
    int i = bI[e], j = bJ[e], k = bK[e];
    float wi = invMass[i], wj = invMass[j], wk = invMass[k];
    if (wi + wj + wk <= 0.f) return;                 // all three pinned
    float3 u = pos[i] - pos[j]; float Lu = len3(u);
    float3 v = pos[i] - pos[k]; float Lv = len3(v);
    if (Lu < 1e-9f || Lv < 1e-9f) return;
    float3 nu = u * (1.f / Lu), nv = v * (1.f / Lv);
    float c = fmaxf(-1.f, fminf(1.f, dot3(nu, nv)));
    float s = sqrtf(fmaxf(1.f - c*c, 0.f));
    if (s < 1e-4f) return;                            // collinear singularity (matches k_AccBending EPS_SIN2)
    float theta = acosf(c);
    float C = theta - theta0[e];
    // Eq14 force-direction gradients (NO 1/(sin*|arm|) normalization) -> paper-faithful soft bending,
    // matches the RK45 golden. gj = (nv - c*nu) ; gk = (nu - c*nv) ; gi = -(gj+gk)  (sums to zero).
    float3 gj = (nv - nu*c);
    float3 gk = (nu - nv*c);
    float3 gi = (gj + gk) * (-1.f);
    float wg = wi*dot3(gi,gi) + wj*dot3(gj,gj) + wk*dot3(gk,gk);
    // STEP 3 bending Kelvin-Voigt damping (cb*theta_dot): gamma = a~_b*betaB*h (betaGammaH precomputed);
    // grad C . (x - x_prev) summed over the 3 particles = the angle rate this substep.
    float gamma  = betaGammaH;
    float gradDx = dot3(gi, pos[i]-prevPos[i]) + dot3(gj, pos[j]-prevPos[j]) + dot3(gk, pos[k]-prevPos[k]);
    float denom = (1.f + gamma) * wg + alphaB;
    if (denom < 1e-12f) return;
    float dLambda = (-C - alphaB * lambda[e] - gamma * gradDx) / denom;
    float applied = omega * dLambda;
    lambda[e] += applied;
    pos[i] = pos[i] + gi * (wi * applied);
    pos[j] = pos[j] + gj * (wj * applied);
    pos[k] = pos[k] + gk * (wk * applied);
}

// STEP 7  - recompute the PER-TET alpha (design §12 R-compliance: alpha~1/h^2, recomputed together with
// every other h-derived coefficient whenever h changes). muV/lambdaV are h-INDEPENDENT (baked once at
// build time); this kernel runs ONCE per xpbd_begin_frame (not per substep  - h is fixed for the frame).
__global__ void k_ComputeTetAlpha(int T, float invH2, const float* muV, const float* lambdaV,
                                  float* alphaDev, float* alphaHyd)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= T) return;
    // muV/lambdaV > 0 guarded at build time (degenerate/zero-material tets are never emitted with
    // youngE<=0 handled by the caller skipping the tet dispatch entirely  - see xpbd_substep).
    alphaDev[t] = invH2 / muV[t];
    alphaHyd[t] = invH2 / lambdaV[t];
}

// STEP 7  - per-tet Stable Neo-Hookean constraint pair (design §4[Strain]): the REST-ZERO deviatoric
// C_dev=tr(F^tF)-3 and hydrostatic C_hyd=det(F)-1, solved back-to-back for the SAME tet (matching the
// production reference's fused SolveNeoHookeanDevTet+HydTet, Simulation/Assets/SurgicalSim/Shaders/
// XPBDSolver.compute:338-421) so the hydrostatic pass sees the deviatoric-corrected positions within
// the SAME colour dispatch (no extra colour-sync). F's three COLUMNS Fc0,Fc1,Fc2 are formed directly
// from the rest inverse rows Br0/Br1/Br2 (F = Ds*B; column c = e1*Br0[c]+e2*Br1[c]+e3*Br2[c]  - the same
// identity XPBDSolver.compute's mul(Ds,B) computes, re-derived to avoid a generic 3x3 type). Within a
// colour no two tets share a vertex -> true Gauss-Seidel, no atomics.
__global__ void k_solve_tet(int colorOff, int colorCnt, float omega,
                            const int* tetIds, const float3* Br0, const float3* Br1, const float3* Br2,
                            const int* tetActive, const float* alphaDev, const float* alphaHyd,
                            const float* invMass, float* lambdaDev, float* lambdaHyd, float3* pos)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= colorCnt) return;
    int e = colorOff + t;
    if (tetActive[e] == 0) return;                 // cut/tear severed a member axis edge, or a corner froze

    int i0 = tetIds[4*e+0], i1 = tetIds[4*e+1], i2 = tetIds[4*e+2], i3 = tetIds[4*e+3];
    float w0 = invMass[i0], w1 = invMass[i1], w2 = invMass[i2], w3 = invMass[i3];
    if (w0 + w1 + w2 + w3 <= 0.f) return;           // all four pinned/inactive

    float3 p0 = pos[i0], p1 = pos[i1], p2 = pos[i2], p3 = pos[i3];
    float3 e1v = p1 - p0, e2v = p2 - p0, e3v = p3 - p0;
    float3 br0 = Br0[e], br1 = Br1[e], br2 = Br2[e];

    // F columns: Fc_c = e1*Br0[c] + e2*Br1[c] + e3*Br2[c]  (F = Ds * B, B's rows = Br0/Br1/Br2)
    float3 Fc0 = e1v*compF3(br0,0) + e2v*compF3(br1,0) + e3v*compF3(br2,0);
    float3 Fc1 = e1v*compF3(br0,1) + e2v*compF3(br1,1) + e3v*compF3(br2,1);
    float3 Fc2 = e1v*compF3(br0,2) + e2v*compF3(br1,2) + e3v*compF3(br2,2);

    // ── (a) DEVIATORIC: C_dev = tr(F^tF) - 3 = |Fc0|^2+|Fc1|^2+|Fc2|^2 - 3. G = 2*F*B^T; grad columns
    // g1=Gcol0 (p1), g2=Gcol1 (p2), g3=Gcol2 (p3); g0 = -(g1+g2+g3) (p0). Gcol_c = 2*(Fc0*Br_c.x +
    // Fc1*Br_c.y + Fc2*Br_c.z)  - Br_c IS row c of B (Br0 for c=0, Br1 for c=1, Br2 for c=2).
    {
        float Cdev = dot3(Fc0,Fc0) + dot3(Fc1,Fc1) + dot3(Fc2,Fc2) - 3.f;
        float3 g1 = (Fc0*br0.x + Fc1*br0.y + Fc2*br0.z) * 2.f;
        float3 g2 = (Fc0*br1.x + Fc1*br1.y + Fc2*br1.z) * 2.f;
        float3 g3 = (Fc0*br2.x + Fc1*br2.y + Fc2*br2.z) * 2.f;
        float3 g0 = (g1 + g2 + g3) * (-1.f);
        float wg = w0*dot3(g0,g0) + w1*dot3(g1,g1) + w2*dot3(g2,g2) + w3*dot3(g3,g3);
        float denom = wg + alphaDev[e];
        if (denom > 1e-10f) {
            float dLambda = (-Cdev - alphaDev[e]*lambdaDev[e]) / denom;
            if (isfinite(dLambda)) {
                float applied = omega * dLambda;
                lambdaDev[e] += applied;
                p0 = p0 + g0*(w0*applied); p1 = p1 + g1*(w1*applied);
                p2 = p2 + g2*(w2*applied); p3 = p3 + g3*(w3*applied);
                pos[i0] = p0; pos[i1] = p1; pos[i2] = p2; pos[i3] = p3;
                // Recompute F from the deviatoric-corrected positions before the hydrostatic pass
                // (matches the reference's re-read of _Positions between kernels, XPBDSolver.compute:268).
                e1v = p1 - p0; e2v = p2 - p0; e3v = p3 - p0;
                Fc0 = e1v*compF3(br0,0) + e2v*compF3(br1,0) + e3v*compF3(br2,0);
                Fc1 = e1v*compF3(br0,1) + e2v*compF3(br1,1) + e3v*compF3(br2,1);
                Fc2 = e1v*compF3(br0,2) + e2v*compF3(br1,2) + e3v*compF3(br2,2);
            }
        }
    }

    // ── (b) HYDROSTATIC: C_hyd = det(F) - 1 = dot(Fc0, cross(Fc1,Fc2)) - 1. Cofactor columns
    // df0=cross(Fc1,Fc2), df1=cross(Fc2,Fc0), df2=cross(Fc0,Fc1); G_hyd = cofM*B^T (no factor of 2).
    {
        float3 df0 = cross3(Fc1,Fc2), df1 = cross3(Fc2,Fc0), df2 = cross3(Fc0,Fc1);
        float Chyd = dot3(Fc0, df0) - 1.f;
        float3 g1 = df0*br0.x + df1*br0.y + df2*br0.z;
        float3 g2 = df0*br1.x + df1*br1.y + df2*br1.z;
        float3 g3 = df0*br2.x + df1*br2.y + df2*br2.z;
        float3 g0 = (g1 + g2 + g3) * (-1.f);
        float wg = w0*dot3(g0,g0) + w1*dot3(g1,g1) + w2*dot3(g2,g2) + w3*dot3(g3,g3);
        float denom = wg + alphaHyd[e];
        if (denom > 1e-10f) {
            float dLambda = (-Chyd - alphaHyd[e]*lambdaHyd[e]) / denom;
            if (isfinite(dLambda)) {
                float applied = omega * dLambda;
                lambdaHyd[e] += applied;
                pos[i0] = p0 + g0*(w0*applied); pos[i1] = p1 + g1*(w1*applied);
                pos[i2] = p2 + g2*(w2*applied); pos[i3] = p3 + g3*(w3*applied);
            }
        }
    }
}

// Velocity update from the solved positions  - RAW, no damping yet. Pinned stay at zero. Split from the
// old fused (raw*velScale) form so the STEP 7 perpendicular cs post-pass (design §4[Structural]) can run
// strictly BETWEEN the raw velocity write and the velScale (Rayleigh/legacy-damping) multiply.
__global__ void k_update_velocity(int N, float invH, const float* invMass,
                                  const float3* pos, const float3* prevPos, float3* vel)
{
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= N) return;
    if (invMass[p] > 0.f)
        vel[p] = (pos[p] - prevPos[p]) * invH;
    else
        vel[p] = mkf3(0.f, 0.f, 0.f);
}

// STEP 7  - PERPENDICULAR velocity post-pass (design §4[Structural], the cs damping the along-constraint
// gamma term does NOT reach). Runs AFTER the raw velocity write, BEFORE velScale. Damps only the
// component of the relative velocity PERPENDICULAR to the (current, post-solve) edge direction  - the
// along-edge component is already damped by k_solve_edges' gamma term; damping it again here would
// double-damp and break the step-3 cs-isolation golden. Dispatched over the EXISTING structural edge
// colours (one launch per colour, no atomics  - within a colour no two edges share a vertex).
// Explicit one-step dashpot impulse per unit mass: vi += -h*wi*cs*vrelPerp (stability margin ~5.5e-3<<2
// at the paper's cs=0.92,m=1,h~1e-3  - design §4[Structural], safe by orders of magnitude).
__global__ void k_perp_damp_edges(int colorOff, int colorCnt, float cs, float h, const float3* pos,
                                  const int* edgeI, const int* edgeJ, const int* edgeActive,
                                  const float* invMass, float3* vel)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= colorCnt) return;
    int e = colorOff + t;
    if (edgeActive[e] == 0) return;
    int i = edgeI[e], j = edgeJ[e];
    float wi = invMass[i], wj = invMass[j];
    if (wi + wj <= 0.f) return;
    float3 d = pos[i] - pos[j];
    float L = len3(d);
    if (L < 1e-9f) return;
    float3 n = d * (1.f / L);
    float3 vrel = vel[i] - vel[j];
    float3 vrelPerp = vrel - n * dot3(vrel, n);
    float3 impulse = vrelPerp * (h * cs);
    if (wi > 0.f) vel[i] = vel[i] - impulse * wi;
    if (wj > 0.f) vel[j] = vel[j] + impulse * wj;
}

// Final post-solve damping multiply: velScale = Rayleigh/legacy-damping (design §12 R-compliance,
// ~h-power). Split out so k_perp_damp_edges (above) runs strictly before it, per design §4[Structural].
__global__ void k_apply_velscale(int N, float velScale, const float* invMass, float3* vel)
{
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= N) return;
    vel[p] = (invMass[p] > 0.f) ? vel[p] * velScale : mkf3(0.f, 0.f, 0.f);
}

#define XK(call) do { cudaError_t _e = (call); if (_e != cudaSuccess) { xpbd_shutdown(); return -(int)_e; } } while (0)

extern "C" int xpbd_init(int particleCount, const float* pos3, const float* invMass,
                         int edgeCount, const int* edges, const float* restLen)
{
    // Validate args + run the colouring BEFORE tearing down any existing instance (fail-preserving:
    // a bad re-init must not destroy a working sim; only a DEVICE failure below can  - see header).
    if (particleCount <= 0 || edgeCount < 0 || !pos3 || !invMass) return XPBD_ERR_BAD_ARG;
    if (edgeCount > 0 && (!edges || !restLen)) return XPBD_ERR_BAD_ARG;   // restLen REQUIRED with edges -
                                                   // a silent rest-0 default would collapse every spring (C2)

    // ── greedy edge colouring (host): no two edges in a colour share a vertex ─────────────────────
    // 64-bit used-colour bitmask per vertex; for a 6-neighbour lattice max degree 6 -> <=7 colours.
    std::vector<unsigned long long> vmask(particleCount, 0ull);
    std::vector<int> edgeColor(edgeCount, 0);
    int numColors = 0;
    for (int e = 0; e < edgeCount; e++) {
        int i = edges[2*e+0], j = edges[2*e+1];
        if (i < 0 || i >= particleCount || j < 0 || j >= particleCount) return XPBD_ERR_BAD_ARG;
        unsigned long long used = vmask[i] | vmask[j];
        int c = 0; while (c < 64 && (used & (1ull << c))) c++;
        if (c >= 64) return XPBD_ERR_COLOR_OVERFLOW; // colour budget exhausted (vertex degree >= 64): FAIL
                                                   // LOUDLY rather than aliasing to a used colour (which would
                                                   // race the no-atomics GS). Unreachable for a 6-neighbour
                                                   // lattice (deg<=6). Existing instance is PRESERVED.
        edgeColor[e] = c;
        vmask[i] |= (1ull << c); vmask[j] |= (1ull << c);
        if (c + 1 > numColors) numColors = c + 1;
    }
    if (g_ready) xpbd_shutdown();
    g_N = particleCount; g_E = edgeCount;
    g_numColors = numColors;
    // bucket edges by colour -> a flat colour-ordered permutation with per-colour [off,cnt).
    g_colorCnt.assign(numColors, 0);
    for (int e = 0; e < g_E; e++) g_colorCnt[edgeColor[e]]++;
    g_colorOff.assign(numColors, 0);
    for (int c = 1; c < numColors; c++) g_colorOff[c] = g_colorOff[c-1] + g_colorCnt[c-1];
    std::vector<int>   hEdgeI(g_E), hEdgeJ(g_E);
    std::vector<float> hRest(g_E);
    std::vector<int>   hActive(g_E, 1);            // every edge live; a cut/tear flips entries via the notify API
    std::vector<int>   hSrc(g_E);                  // M1: colour-slot -> input index (persisted, not discarded)
    std::vector<int>   cursor = g_colorOff;
    g_edgeSlotOfInput.assign(g_E, -1);
    for (int e = 0; e < g_E; e++) {
        int c = edgeColor[e]; int dst = cursor[c]++;
        hEdgeI[dst] = edges[2*e+0]; hEdgeJ[dst] = edges[2*e+1]; hRest[dst] = restLen ? restLen[e] : 0.f;
        hSrc[dst] = e; g_edgeSlotOfInput[e] = dst;
    }

    // ── allocate + upload ─────────────────────────────────────────────────────────────────────────
    XK(cudaMalloc(&d_pos,     (size_t)g_N * sizeof(float3)));
    XK(cudaMalloc(&d_prevPos, (size_t)g_N * sizeof(float3)));
    XK(cudaMalloc(&d_vel,     (size_t)g_N * sizeof(float3)));
    XK(cudaMalloc(&d_invMass, (size_t)g_N * sizeof(float)));
    if (g_E > 0) {
        XK(cudaMalloc(&d_edgeI,     (size_t)g_E * sizeof(int)));
        XK(cudaMalloc(&d_edgeJ,     (size_t)g_E * sizeof(int)));
        XK(cudaMalloc(&d_restLen,   (size_t)g_E * sizeof(float)));
        XK(cudaMalloc(&d_lambda,    (size_t)g_E * sizeof(float)));
        XK(cudaMalloc(&d_edgeActive,(size_t)g_E * sizeof(int)));
        XK(cudaMalloc(&d_edgeSrcIdx,(size_t)g_E * sizeof(int)));
    }
    XK(cudaMemcpy(d_pos,     pos3,    (size_t)g_N * sizeof(float3), cudaMemcpyHostToDevice));
    XK(cudaMemcpy(d_invMass, invMass, (size_t)g_N * sizeof(float),  cudaMemcpyHostToDevice));
    XK(cudaMemset(d_vel,     0, (size_t)g_N * sizeof(float3)));
    XK(cudaMemcpy(d_prevPos, pos3,    (size_t)g_N * sizeof(float3), cudaMemcpyHostToDevice));
    if (g_E > 0) {
        XK(cudaMemcpy(d_edgeI,     hEdgeI.data(),  (size_t)g_E * sizeof(int),   cudaMemcpyHostToDevice));
        XK(cudaMemcpy(d_edgeJ,     hEdgeJ.data(),  (size_t)g_E * sizeof(int),   cudaMemcpyHostToDevice));
        XK(cudaMemcpy(d_restLen,   hRest.data(),   (size_t)g_E * sizeof(float), cudaMemcpyHostToDevice));
        XK(cudaMemcpy(d_edgeActive,hActive.data(), (size_t)g_E * sizeof(int),   cudaMemcpyHostToDevice));
        XK(cudaMemcpy(d_edgeSrcIdx,hSrc.data(),    (size_t)g_E * sizeof(int),   cudaMemcpyHostToDevice));
        XK(cudaMemset(d_lambda, 0, (size_t)g_E * sizeof(float)));
    }
    g_frameReady = false;                          // params/topology changed: caller must begin a new frame
    g_ready = true;
    return 0;
}

// ── M1 notify path (design §9-(4)): propagate a runtime sever / mass change through the PERSISTED
// input->colour-slot permutation into the colour-ordered device flags. At the rebind step this is
// superseded by k_RefreshLiveness (deriving liveness from the LIVE shared nbrIdx/bendPairs/active/mass
// every substep); the notify API remains for the standalone self-test + targeted host-side control. ────
extern "C" int xpbd_set_edge_active(int inputEdgeIndex, int active)
{
    if (!g_ready) return XPBD_ERR_NOT_READY;
    if (inputEdgeIndex < 0 || inputEdgeIndex >= (int)g_edgeSlotOfInput.size()) return XPBD_ERR_BAD_ARG;
    int slot = g_edgeSlotOfInput[inputEdgeIndex];
    int v = active ? 1 : 0;
    return -(int)cudaMemcpy(d_edgeActive + slot, &v, sizeof(int), cudaMemcpyHostToDevice);
}

extern "C" int xpbd_set_bend_active(int inputBendIndex, int active)
{
    if (!g_ready) return XPBD_ERR_NOT_READY;
    if (inputBendIndex < 0 || inputBendIndex >= (int)g_bendSlotOfInput.size()) return XPBD_ERR_BAD_ARG;
    int slot = g_bendSlotOfInput[inputBendIndex];
    int v = active ? 1 : 0;
    return -(int)cudaMemcpy(d_bendActive + slot, &v, sizeof(int), cudaMemcpyHostToDevice);
}

extern "C" int xpbd_set_particle_inv_mass(int particle, float invMass)
{
    if (!g_ready) return XPBD_ERR_NOT_READY;
    if (particle < 0 || particle >= g_N) return XPBD_ERR_BAD_ARG;
    return -(int)cudaMemcpy(d_invMass + particle, &invMass, sizeof(float), cudaMemcpyHostToDevice);
}

extern "C" void xpbd_set_params(const XpbdParams* p) { if (p) { g_p = *p; g_frameReady = false; } }
                                                       // params changed -> h-derived coefficients stale;
                                                       // force a fresh xpbd_begin_frame (R-compliance)
extern "C" int  xpbd_color_count() { return g_numColors; }
extern "C" int  xpbd_bend_color_count() { return g_bNumColors; }

// §9-(4) sever-propagation VERIFY diagnostics ("readback d_edgeActive").
extern "C" int xpbd_get_edge_count() { return g_ready ? g_E : XPBD_ERR_NOT_READY; }
extern "C" int xpbd_readback_edge_active(int* out)
{
    if (!g_ready) return XPBD_ERR_NOT_READY;
    if (!out) return XPBD_ERR_BAD_ARG;
    if (g_E == 0) return 0;
    return -(int)cudaMemcpy(out, d_edgeActive, (size_t)g_E * sizeof(int), cudaMemcpyDeviceToHost);
}

extern "C" int xpbd_set_bending(int bendCount, const int* ijk, const float* theta0)
{
    if (!g_ready) return XPBD_ERR_NOT_READY;
    // ONLY an explicit bendCount==0 clears the set; a NEGATIVE count (e.g. an underflowed size) is a
    // caller bug and must FAIL LOUD, not silently wipe a working bend set (fail-preserving contract).
    // theta0 is REQUIRED alongside ijk  - a silent theta0=0 default would fold the rest shape (C2).
    if (bendCount < 0 || (bendCount > 0 && (!ijk || !theta0))) return XPBD_ERR_BAD_ARG;

    // greedy colouring over the THREE vertices of each element (share ANY vertex -> different colour),
    // run BEFORE the existing bend set is freed (fail-preserving: a colour-overflow / bad-index failure
    // leaves the OLD set intact and the module fully usable  - see the header failure semantics).
    std::vector<int> col;
    int numColors = 0;
    if (bendCount > 0) {
        std::vector<unsigned long long> vmask(g_N, 0ull);
        col.assign(bendCount, 0);
        for (int e = 0; e < bendCount; e++) {
            int i = ijk[3*e+0], j = ijk[3*e+1], k = ijk[3*e+2];
            if (i < 0 || i >= g_N || j < 0 || j >= g_N || k < 0 || k >= g_N) return XPBD_ERR_BAD_ARG;
            unsigned long long used = vmask[i] | vmask[j] | vmask[k];
            int c = 0; while (c < 64 && (used & (1ull << c))) c++;
            if (c >= 64) return XPBD_ERR_COLOR_OVERFLOW;   // fail loudly; OLD bend set preserved
            col[e] = c; vmask[i] |= (1ull<<c); vmask[j] |= (1ull<<c); vmask[k] |= (1ull<<c);
            if (c+1 > numColors) numColors = c+1;
        }
    }

    // colouring succeeded (or clearing requested): free any prior bending set + install the new one.
    cudaFree(d_bendI); cudaFree(d_bendJ); cudaFree(d_bendK);
    cudaFree(d_bendTheta0); cudaFree(d_bendActive); cudaFree(d_bendLambda); cudaFree(d_bendSrcIdx);
    d_bendI=d_bendJ=d_bendK=d_bendActive=d_bendSrcIdx=nullptr; d_bendTheta0=d_bendLambda=nullptr;
    g_B = 0; g_bColorOff.clear(); g_bColorCnt.clear(); g_bNumColors = 0; g_bendSlotOfInput.clear();
    g_frameReady = false;                          // topology changed: caller must begin a new frame
    if (bendCount <= 0) return 0;
    g_B = bendCount;
    g_bNumColors = numColors;
    g_bColorCnt.assign(numColors, 0);
    for (int e = 0; e < g_B; e++) g_bColorCnt[col[e]]++;
    g_bColorOff.assign(numColors, 0);
    for (int cc = 1; cc < numColors; cc++) g_bColorOff[cc] = g_bColorOff[cc-1] + g_bColorCnt[cc-1];
    std::vector<int> hI(g_B), hJ(g_B), hK(g_B), hAct(g_B, 1), hSrcB(g_B); std::vector<float> hT(g_B);
    std::vector<int> cur = g_bColorOff;
    g_bendSlotOfInput.assign(g_B, -1);
    for (int e = 0; e < g_B; e++) {
        int dst = cur[col[e]]++;
        hI[dst]=ijk[3*e+0]; hJ[dst]=ijk[3*e+1]; hK[dst]=ijk[3*e+2]; hT[dst]=theta0?theta0[e]:0.f;
        hSrcB[dst] = e; g_bendSlotOfInput[e] = dst;
    }
    // On any alloc/upload failure: free the partial bend buffers + reset bend state before returning
    // (mirror step-1's XK-calls-shutdown pattern, so a failed set leaves NO half-allocated set that a
    // subsequent xpbd_step could dispatch k_solve_bending against with a null buffer  - review minor).
    #define XB(call) do { cudaError_t _e=(call); if (_e!=cudaSuccess) { \
        cudaFree(d_bendI); cudaFree(d_bendJ); cudaFree(d_bendK); \
        cudaFree(d_bendTheta0); cudaFree(d_bendActive); cudaFree(d_bendLambda); cudaFree(d_bendSrcIdx); \
        d_bendI=d_bendJ=d_bendK=d_bendActive=d_bendSrcIdx=nullptr; d_bendTheta0=d_bendLambda=nullptr; \
        g_B=0; g_bNumColors=0; g_bColorOff.clear(); g_bColorCnt.clear(); g_bendSlotOfInput.clear(); \
        return -(int)_e; } } while(0)
    XB(cudaMalloc(&d_bendI,(size_t)g_B*sizeof(int)));      XB(cudaMalloc(&d_bendJ,(size_t)g_B*sizeof(int)));
    XB(cudaMalloc(&d_bendK,(size_t)g_B*sizeof(int)));      XB(cudaMalloc(&d_bendTheta0,(size_t)g_B*sizeof(float)));
    XB(cudaMalloc(&d_bendActive,(size_t)g_B*sizeof(int))); XB(cudaMalloc(&d_bendLambda,(size_t)g_B*sizeof(float)));
    XB(cudaMalloc(&d_bendSrcIdx,(size_t)g_B*sizeof(int)));
    XB(cudaMemcpy(d_bendI,hI.data(),(size_t)g_B*sizeof(int),cudaMemcpyHostToDevice));
    XB(cudaMemcpy(d_bendJ,hJ.data(),(size_t)g_B*sizeof(int),cudaMemcpyHostToDevice));
    XB(cudaMemcpy(d_bendK,hK.data(),(size_t)g_B*sizeof(int),cudaMemcpyHostToDevice));
    XB(cudaMemcpy(d_bendTheta0,hT.data(),(size_t)g_B*sizeof(float),cudaMemcpyHostToDevice));
    XB(cudaMemcpy(d_bendActive,hAct.data(),(size_t)g_B*sizeof(int),cudaMemcpyHostToDevice));
    XB(cudaMemcpy(d_bendSrcIdx,hSrcB.data(),(size_t)g_B*sizeof(int),cudaMemcpyHostToDevice));
    XB(cudaMemset(d_bendLambda,0,(size_t)g_B*sizeof(float)));
    #undef XB
    return 0;
}

// ── REBIND / GO-LIVE (design rev-B §9-(4)) ──────────────────────────────────────────────────────────
// Builds the module's topology FROM the live shared buffers (one-time device readback), reusing the
// audited xpbd_init/xpbd_set_bending machinery (validation -> colouring -> permutation -> install) on
// host lists, then CONVERTS to bound mode: the private pos/vel are freed and replaced by ALIASES of the
// shared cornerPos/vel; the owner-slot and live-source maps are uploaded for k_RefreshLiveness.
// Fail-preserving up through the colouring (inherited from xpbd_init); a DEVICE failure during install/
// conversion leaves the module shut down (header semantics).
extern "C" int xpbd_bind(float3* cornerPos, float3* vel,
                         const int* nbrIdx, const float3* restNbr,
                         const void* bendPairsRaw, int bendCapacity,
                         const int* active, const int* pinned,
                         const float* mass,
                         const float3* extForce,
                         int dimsX, int dimsY, int dimsZ, int cornerCount)
{
    if (!cornerPos || !vel || !nbrIdx || !restNbr || !active || !pinned || !mass || !extForce ||
        cornerCount <= 0) return XPBD_ERR_BAD_ARG;
    if (bendCapacity < 0 || (bendCapacity > 0 && !bendPairsRaw)) return XPBD_ERR_BAD_ARG;
    const int cc = cornerCount;

    // 1) One-time READBACK of the live topology + state (no module state touched yet  - a readback
    //    failure preserves any existing instance).
    std::vector<int>             hNbr((size_t)6 * cc);
    std::vector<float3>          hRestNbr((size_t)6 * cc);
    std::vector<float3>          hPos(cc);
    std::vector<int>             hActive(cc), hPinned(cc);
    std::vector<float>           hMass(cc);
    std::vector<XpbdBendPairGpu> hBp(bendCapacity > 0 ? bendCapacity : 1);
    #define XR(call) do { cudaError_t _e = (call); if (_e != cudaSuccess) return -(int)_e; } while (0)
    XR(cudaMemcpy(hNbr.data(),     nbrIdx,    (size_t)6 * cc * sizeof(int),    cudaMemcpyDeviceToHost));
    XR(cudaMemcpy(hRestNbr.data(), restNbr,   (size_t)6 * cc * sizeof(float3), cudaMemcpyDeviceToHost));
    XR(cudaMemcpy(hPos.data(),     cornerPos, (size_t)cc * sizeof(float3),     cudaMemcpyDeviceToHost));
    XR(cudaMemcpy(hActive.data(),  active,    (size_t)cc * sizeof(int),        cudaMemcpyDeviceToHost));
    XR(cudaMemcpy(hPinned.data(),  pinned,    (size_t)cc * sizeof(int),        cudaMemcpyDeviceToHost));
    XR(cudaMemcpy(hMass.data(),    mass,      (size_t)cc * sizeof(float),      cudaMemcpyDeviceToHost));
    if (bendCapacity > 0)
        XR(cudaMemcpy(hBp.data(), bendPairsRaw, (size_t)bendCapacity * sizeof(XpbdBendPairGpu),
                      cudaMemcpyDeviceToHost));
    #undef XR

    // 2) CANONICAL EDGE DEDUP (design §9-(4)): emit each lattice edge ONCE from its POSITIVE slot
    //    s = 2*axis+1 (slot order 0=-x,1=+x,2=-y,3=+y,4=-z,5=+z  - cut.cu k_SeverLinks writes exactly
    //    these). Emitted regardless of active[]  - liveness is PER-SUBSTEP (k_RefreshLiveness), the
    //    topology is static. ownerSlot[e] = 6p+s is what the refresh reads back through.
    std::vector<int>   hEdges;  hEdges.reserve((size_t)3 * cc * 2);
    std::vector<float> hRest;   hRest.reserve((size_t)3 * cc);
    std::vector<int>   hOwner;  hOwner.reserve((size_t)3 * cc);
    for (int p = 0; p < cc; p++)
        for (int axis = 0; axis < 3; axis++) {
            int s = 2 * axis + 1;
            int n = hNbr[(size_t)6 * p + s];
            if (n < 0) continue;
            if (n >= cc) return XPBD_ERR_BAD_ARG;
            float3 r = hRestNbr[(size_t)6 * p + s];
            hEdges.push_back(p); hEdges.push_back(n);
            hRest.push_back(sqrtf(r.x * r.x + r.y * r.y + r.z * r.z));   // L0 = length3(restNbr[6p+s])
            hOwner.push_back(6 * p + s);
        }
    const int E = (int)hRest.size();

    // 3) BEND ELEMENTS from the live pairs that are ALIVE at bind time (a sever is permanent  - dead
    //    pairs can never return, so they are not emitted). srcIdx = index into the LIVE bendPairs array.
    std::vector<int>   hIjk;  hIjk.reserve((size_t)bendCapacity * 3);
    std::vector<float> hTh;   hTh.reserve(bendCapacity);
    std::vector<int>   hSrcLive; hSrcLive.reserve(bendCapacity);
    for (int b = 0; b < bendCapacity; b++) {
        const XpbdBendPairGpu& bp = hBp[b];
        if (bp.alive == 0) continue;
        if (bp.i < 0 || bp.i >= cc || bp.j < 0 || bp.j >= cc || bp.k < 0 || bp.k >= cc)
            return XPBD_ERR_BAD_ARG;
        hIjk.push_back(bp.i); hIjk.push_back(bp.j); hIjk.push_back(bp.k);
        hTh.push_back(bp.theta0);
        hSrcLive.push_back(b);
    }
    const int B = (int)hTh.size();

    // 4) INSTALL via the audited standalone path (validates + colours BEFORE tearing down any previous
    //    instance, then allocates + uploads the colour-ordered arrays + permutation maps). invMass here
    //    is only the bind-time seed  - k_RefreshLiveness re-derives it every substep.
    std::vector<float> hInvMass(cc);
    for (int c = 0; c < cc; c++)
        hInvMass[c] = (hPinned[c] != 0 || hActive[c] == 0) ? 0.f : (1.f / hMass[c]);
    int rc = xpbd_init(cc, (const float*)hPos.data(), hInvMass.data(),
                       E, E > 0 ? hEdges.data() : nullptr, E > 0 ? hRest.data() : nullptr);
    if (rc != 0) return rc;
    rc = xpbd_set_bending(B, B > 0 ? hIjk.data() : nullptr, B > 0 ? hTh.data() : nullptr);
    if (rc != 0) { xpbd_shutdown(); return rc; }

    // 5) CONVERT to bound mode. Owner-slot map: device slot -> 6p+s (permuted through the colour map);
    //    bend source map: device slot -> LIVE bendPairs index (overwrites the identity srcIdx that
    //    xpbd_set_bending installed  - in bound mode d_bendSrcIdx indexes the LIVE array).
    if (E > 0) {
        std::vector<int> devOwner(E);
        for (int e = 0; e < E; e++) devOwner[g_edgeSlotOfInput[e]] = hOwner[e];
        XK(cudaMalloc(&d_edgeOwnerSlot, (size_t)E * sizeof(int)));
        XK(cudaMemcpy(d_edgeOwnerSlot, devOwner.data(), (size_t)E * sizeof(int), cudaMemcpyHostToDevice));
    }
    if (B > 0) {
        std::vector<int> devSrc(B);
        for (int b = 0; b < B; b++) devSrc[g_bendSlotOfInput[b]] = hSrcLive[b];
        XK(cudaMemcpy(d_bendSrcIdx, devSrc.data(), (size_t)B * sizeof(int), cudaMemcpyHostToDevice));
    }
    // Swap the private pos/vel for ALIASES of the shared live buffers. From g_bound=true on, shutdown
    // will not free them.
    cudaFree(d_pos); d_pos = cornerPos;
    cudaFree(d_vel); d_vel = vel;
    g_bound = true;
    g_liveNbrIdx = nbrIdx; g_liveBendPairs = (const XpbdBendPairGpu*)bendPairsRaw;
    g_liveActive = active; g_livePinned = pinned; g_liveMass = mass;
    g_liveExtForce = extForce;
    g_dimsX = dimsX; g_dimsY = dimsY; g_dimsZ = dimsZ;
    // 6) BIND-TIME STATE (§9-(4)): d_prevPos = cornerPos (no first-frame velocity kick); the CURRENT
    //    shared vel contents are ADOPTED (a moving liver is not zeroed  - d_vel now aliases it, untouched).
    XK(cudaMemcpy(d_prevPos, cornerPos, (size_t)cc * sizeof(float3), cudaMemcpyDeviceToDevice));

    g_frameReady = false;                          // caller must begin a fresh frame
    return 0;
}

// ── STEP 7: per-tet Stable Neo-Hookean strain BUILD (design rev-B §4[Strain]/§9-(7)) ───────────────
// Host-side generic 3x3 inverse (double precision  - build-time only, off the hot path).
static bool Inv3x3d(const double m[3][3], double out[3][3])
{
    double det = m[0][0]*(m[1][1]*m[2][2]-m[1][2]*m[2][1])
               - m[0][1]*(m[1][0]*m[2][2]-m[1][2]*m[2][0])
               + m[0][2]*(m[1][0]*m[2][1]-m[1][1]*m[2][0]);
    if (fabs(det) < 1e-20) return false;
    double invDet = 1.0 / det;
    out[0][0] =  (m[1][1]*m[2][2]-m[1][2]*m[2][1]) * invDet;
    out[0][1] = -(m[0][1]*m[2][2]-m[0][2]*m[2][1]) * invDet;
    out[0][2] =  (m[0][1]*m[1][2]-m[0][2]*m[1][1]) * invDet;
    out[1][0] = -(m[1][0]*m[2][2]-m[1][2]*m[2][0]) * invDet;
    out[1][1] =  (m[0][0]*m[2][2]-m[0][2]*m[2][0]) * invDet;
    out[1][2] = -(m[0][0]*m[1][2]-m[0][2]*m[1][0]) * invDet;
    out[2][0] =  (m[1][0]*m[2][1]-m[1][1]*m[2][0]) * invDet;
    out[2][1] = -(m[0][0]*m[2][1]-m[0][1]*m[2][0]) * invDet;
    out[2][2] =  (m[0][0]*m[1][1]-m[0][1]*m[1][0]) * invDet;
    return true;
}

extern "C" int xpbd_build_tets()
{
    if (!g_ready || !g_bound) return XPBD_ERR_NOT_READY;   // bound-mode only (needs dims + live topology)
    if (g_dimsX <= 0 || g_dimsY <= 0 || g_dimsZ <= 0) return XPBD_ERR_BAD_ARG;
    const int cc = g_N;
    const int nx = g_dimsX + 1, ny = g_dimsY + 1;   // corner-grid extents (VOXEL dims + 1 per axis -
                                                    // matches C# GridConventions.CornerId(i,j,k,dims))
    auto cid = [&](int x, int y, int z) { return x + nx * (y + ny * z); };

    // One-time readback of the buffers needed at BUILD time. d_pos aliases the shared cornerPos and is
    // still the REST configuration (xpbd_build_tets must run before the first xpbd_begin_frame, per the
    // header contract)  - so Dm/Dm^-1 baked here are the genuine rest-state inverse rest matrices.
    std::vector<float3> hPos(cc);
    std::vector<int>    hActive(cc);
    #define XT(call) do { cudaError_t _e = (call); if (_e != cudaSuccess) return -(int)_e; } while (0)
    XT(cudaMemcpy(hPos.data(),    d_pos,        (size_t)cc * sizeof(float3), cudaMemcpyDeviceToHost));
    XT(cudaMemcpy(hActive.data(), g_liveActive, (size_t)cc * sizeof(int),    cudaMemcpyDeviceToHost));
    #undef XT

    // Kuhn/"diagonal" 6-tet split, main diagonal c0-c7 (local bit-order corner index c[i] = (i&1,
    // (i>>1)&1, (i>>2)&1); design §4[Strain]).
    static const int TET_TABLE[6][4] = { {0,1,3,7}, {0,3,2,7}, {0,2,6,7}, {0,6,4,7}, {0,4,5,7}, {0,5,1,7} };

    std::vector<int>    hTetIds;     hTetIds.reserve(4096);
    std::vector<float3> hBr0, hBr1, hBr2; hBr0.reserve(1024); hBr1.reserve(1024); hBr2.reserve(1024);
    std::vector<float>  hMuV, hLambdaV;    hMuV.reserve(1024); hLambdaV.reserve(1024);
    std::vector<int>    hOwnerSlot;  hOwnerSlot.reserve(3072);

    // mu/lambda from E/nu (design §4[Strain]): mu=E/(2(1+nu)); lambda=E*nu/((1+nu)(1-2*nu)). youngE<=0
    // (strain OFF) still builds the tet mesh  - so the CUT lifecycle can be exercised independent of the
    // material tune  - with mu=lambda=1 placeholders that xpbd_begin_frame's g_doStrainF gate (youngE>0)
    // ensures are never dispatched as a real compliance.
    float nu = g_p.poisson, mu, lambda;
    if (g_p.youngE > 0.f && nu > -1.f && nu < 0.5f) {
        mu     = g_p.youngE / (2.f * (1.f + nu));
        lambda = g_p.youngE * nu / ((1.f + nu) * (1.f - 2.f * nu));
    } else {
        mu = lambda = 1.f;
    }
    if (!(mu > 0.f)) mu = 1.f;
    if (!(lambda > 0.f)) lambda = 1.f;

    for (int z = 0; z < g_dimsZ; z++)
    for (int y = 0; y < g_dimsY; y++)
    for (int x = 0; x < g_dimsX; x++) {
        bool allActive = true;
        for (int li = 0; li < 8 && allActive; li++) {
            int dx = li & 1, dy = (li >> 1) & 1, dz = (li >> 2) & 1;
            if (hActive[cid(x + dx, y + dy, z + dz)] == 0) allActive = false;
        }
        if (!allActive) continue;   // emit only when all 8 cell corners are active (design §4[Strain])

        int flip = ((x + y + z) & 1) ? 1 : 0;   // parity alternation: XOR local bit0 on odd cells so the
                                                 // main diagonal alternates c0-c7 / c1-c6 cell-to-cell
                                                 // (cancels the anisotropy bias a uniform diagonal would add)
        for (int ti = 0; ti < 6; ti++) {
            int gidTet[4]; float3 restP[4];
            for (int k = 0; k < 4; k++) {
                int li2 = TET_TABLE[ti][k] ^ flip;
                int dx = li2 & 1, dy = (li2 >> 1) & 1, dz = (li2 >> 2) & 1;
                gidTet[k] = cid(x + dx, y + dy, z + dz);
                restP[k] = hPos[gidTet[k]];
            }
            double e1[3] = { restP[1].x-restP[0].x, restP[1].y-restP[0].y, restP[1].z-restP[0].z };
            double e2[3] = { restP[2].x-restP[0].x, restP[2].y-restP[0].y, restP[2].z-restP[0].z };
            double e3[3] = { restP[3].x-restP[0].x, restP[3].y-restP[0].y, restP[3].z-restP[0].z };
            double Dm[3][3] = { {e1[0],e2[0],e3[0]}, {e1[1],e2[1],e3[1]}, {e1[2],e2[2],e3[2]} };
            double B[3][3];
            if (!Inv3x3d(Dm, B)) continue;   // singular rest tet: skip
            double det = Dm[0][0]*(Dm[1][1]*Dm[2][2]-Dm[1][2]*Dm[2][1])
                       - Dm[0][1]*(Dm[1][0]*Dm[2][2]-Dm[1][2]*Dm[2][0])
                       + Dm[0][2]*(Dm[1][0]*Dm[2][1]-Dm[1][1]*Dm[2][0]);
            double V = fabs(det) / 6.0;
            if (V < 1e-12) continue;         // degenerate-tet guard (XPBDSolverGPU.cs:208-231 pattern)

            // the tet's 3 member AXIS edges (popcount(local_a^local_b)==1). Axis identity is computed
            // from the ORIGINAL (pre-flip) local indices  - XOR-ing both endpoints by the SAME flip mask
            // preserves which bit differs  - then the SAME flip resolves the owning (lower-coordinate)
            // corner's global id/slot.
            int axisEdges = 0;
            for (int a = 0; a < 4 && axisEdges < 3; a++)
            for (int b = a + 1; b < 4 && axisEdges < 3; b++) {
                int lu = TET_TABLE[ti][a], lv = TET_TABLE[ti][b];
                int diff = lu ^ lv;
                if (diff != 1 && diff != 2 && diff != 4) continue;   // not a single-bit (axis) edge
                int axis = (diff == 1) ? 0 : (diff == 2 ? 1 : 2);
                int lu2 = lu ^ flip, lv2 = lv ^ flip;
                int lowerLocal = (((lu2 >> axis) & 1) == 0) ? lu2 : lv2;   // bit=0 side = lower along axis
                int dx = lowerLocal & 1, dy = (lowerLocal >> 1) & 1, dz = (lowerLocal >> 2) & 1;
                int lowerGid = cid(x + dx, y + dy, z + dz);
                hOwnerSlot.push_back(6 * lowerGid + (2 * axis + 1));
                axisEdges++;
            }
            if (axisEdges != 3) continue;   // invariant guard (never trips for the fixed Kuhn table)

            hTetIds.push_back(gidTet[0]); hTetIds.push_back(gidTet[1]);
            hTetIds.push_back(gidTet[2]); hTetIds.push_back(gidTet[3]);
            hBr0.push_back(mkf3((float)B[0][0], (float)B[0][1], (float)B[0][2]));
            hBr1.push_back(mkf3((float)B[1][0], (float)B[1][1], (float)B[1][2]));
            hBr2.push_back(mkf3((float)B[2][0], (float)B[2][1], (float)B[2][2]));
            hMuV.push_back(mu * (float)V);
            hLambdaV.push_back(lambda * (float)V);
        }
    }

    const int T = (int)hMuV.size();

    // greedy 4-vertex tet colouring (share ANY of the 4 corners -> different colour), same pattern as
    // the bending 3-vertex colouring.
    std::vector<unsigned long long> vmask(cc, 0ull);
    std::vector<int> tetColor(T, 0);
    int numColors = 0;
    for (int e = 0; e < T; e++) {
        int i0=hTetIds[4*e+0], i1=hTetIds[4*e+1], i2=hTetIds[4*e+2], i3=hTetIds[4*e+3];
        unsigned long long used = vmask[i0] | vmask[i1] | vmask[i2] | vmask[i3];
        int c = 0; while (c < 64 && (used & (1ull << c))) c++;
        if (c >= 64) return XPBD_ERR_COLOR_OVERFLOW;
        tetColor[e] = c;
        vmask[i0] |= (1ull<<c); vmask[i1] |= (1ull<<c); vmask[i2] |= (1ull<<c); vmask[i3] |= (1ull<<c);
        if (c + 1 > numColors) numColors = c + 1;
    }

    g_tColorCnt.assign(numColors, 0);
    for (int e = 0; e < T; e++) g_tColorCnt[tetColor[e]]++;
    g_tColorOff.assign(numColors, 0);
    for (int c = 1; c < numColors; c++) g_tColorOff[c] = g_tColorOff[c-1] + g_tColorCnt[c-1];
    std::vector<int> cursor = g_tColorOff;
    std::vector<int>    dIds((size_t)4*T);
    std::vector<float3> dBr0(T), dBr1(T), dBr2(T);
    std::vector<float>  dMuV(T), dLambdaV(T);
    std::vector<int>    dOwner((size_t)3*T);
    for (int e = 0; e < T; e++) {
        int dst = cursor[tetColor[e]]++;
        dIds[4*dst+0]=hTetIds[4*e+0]; dIds[4*dst+1]=hTetIds[4*e+1];
        dIds[4*dst+2]=hTetIds[4*e+2]; dIds[4*dst+3]=hTetIds[4*e+3];
        dBr0[dst]=hBr0[e]; dBr1[dst]=hBr1[e]; dBr2[dst]=hBr2[e];
        dMuV[dst]=hMuV[e]; dLambdaV[dst]=hLambdaV[e];
        dOwner[3*dst+0]=hOwnerSlot[3*e+0]; dOwner[3*dst+1]=hOwnerSlot[3*e+1]; dOwner[3*dst+2]=hOwnerSlot[3*e+2];
    }

    if (T > 0) {
        XK(cudaMalloc(&d_tetIds,        (size_t)4*T * sizeof(int)));
        XK(cudaMalloc(&d_tetBr0,        (size_t)T   * sizeof(float3)));
        XK(cudaMalloc(&d_tetBr1,        (size_t)T   * sizeof(float3)));
        XK(cudaMalloc(&d_tetBr2,        (size_t)T   * sizeof(float3)));
        XK(cudaMalloc(&d_tetMuV,        (size_t)T   * sizeof(float)));
        XK(cudaMalloc(&d_tetLambdaV,    (size_t)T   * sizeof(float)));
        XK(cudaMalloc(&d_tetAlphaDev,   (size_t)T   * sizeof(float)));
        XK(cudaMalloc(&d_tetAlphaHyd,   (size_t)T   * sizeof(float)));
        XK(cudaMalloc(&d_tetLambdaDevL, (size_t)T   * sizeof(float)));
        XK(cudaMalloc(&d_tetLambdaHydL, (size_t)T   * sizeof(float)));
        XK(cudaMalloc(&d_tetActive,     (size_t)T   * sizeof(int)));
        XK(cudaMalloc(&d_tetEdgeOwnerSlot, (size_t)3*T * sizeof(int)));

        XK(cudaMemcpy(d_tetIds,        dIds.data(),    (size_t)4*T*sizeof(int),   cudaMemcpyHostToDevice));
        XK(cudaMemcpy(d_tetBr0,        dBr0.data(),    (size_t)T*sizeof(float3), cudaMemcpyHostToDevice));
        XK(cudaMemcpy(d_tetBr1,        dBr1.data(),    (size_t)T*sizeof(float3), cudaMemcpyHostToDevice));
        XK(cudaMemcpy(d_tetBr2,        dBr2.data(),    (size_t)T*sizeof(float3), cudaMemcpyHostToDevice));
        XK(cudaMemcpy(d_tetMuV,        dMuV.data(),    (size_t)T*sizeof(float),  cudaMemcpyHostToDevice));
        XK(cudaMemcpy(d_tetLambdaV,    dLambdaV.data(),(size_t)T*sizeof(float),  cudaMemcpyHostToDevice));
        XK(cudaMemcpy(d_tetEdgeOwnerSlot, dOwner.data(),(size_t)3*T*sizeof(int), cudaMemcpyHostToDevice));
        XK(cudaMemset(d_tetAlphaDev,   0, (size_t)T*sizeof(float)));
        XK(cudaMemset(d_tetAlphaHyd,   0, (size_t)T*sizeof(float)));
        XK(cudaMemset(d_tetLambdaDevL, 0, (size_t)T*sizeof(float)));
        XK(cudaMemset(d_tetLambdaHydL, 0, (size_t)T*sizeof(float)));
        // Seed active=1: k_RefreshLiveness re-derives the true value before the FIRST predict of the
        // FIRST substep, so this seed is never read as a load-bearing liveness value.
        std::vector<int> ones(T, 1);
        XK(cudaMemcpy(d_tetActive, ones.data(), (size_t)T*sizeof(int), cudaMemcpyHostToDevice));
    }
    g_T = T; g_tNumColors = numColors; g_haveTets = true;
    g_frameReady = false;   // topology changed: caller must begin a new frame
    return 0;
}

extern "C" int xpbd_tet_count()       { return g_T; }
extern "C" int xpbd_tet_color_count() { return g_tNumColors; }
extern "C" int xpbd_readback_tet_ids(int* out)
{
    if (g_T <= 0) return 0;
    if (!out) return XPBD_ERR_BAD_ARG;
    return -(int)cudaMemcpy(out, d_tetIds, (size_t)4 * g_T * sizeof(int), cudaMemcpyDeviceToHost);
}

extern "C" int xpbd_begin_frame(float dt)
{
    if (!g_ready) return XPBD_ERR_NOT_READY;
    g_frameReady = false;                               // a FAILED begin must not leave the OLD frame armed
    int N = g_p.numSubsteps > 0 ? g_p.numSubsteps : 1;
    float h = dt / (float)N;
    if (!(h > 0.f) || !std::isfinite(h)) return XPBD_ERR_BAD_DT;  // rejects <=0, NaN (!(NaN>0)) AND +Inf
                                                        // (Inf would make gamma = 0*Inf = NaN downstream)
    float alphaTilde = 1.f / (g_p.ks * h * h);          // a~ = (1/ks)/h^2
    // STEP 3 damping: alpha (Rayleigh) as the implicit per-substep velocity term v *= 1/(1+alpha*h);
    // cs/cb (Kelvin-Voigt) as the Macklin XPBD damped-constraint term gamma = a~*beta*h.
    g_velScaleF    = (1.f - g_p.damping) / (1.f + g_p.alpha * h);
    g_betaGammaHSF = alphaTilde * g_p.betaS * h;
    g_alphaTildeF  = alphaTilde;
    g_itersF       = g_p.iters > 0 ? g_p.iters : 1;
    g_doBendF      = (g_B > 0 && g_p.kb > 0.f);
    float alphaB   = g_doBendF ? 1.f / (g_p.kb * h * h) : 0.f;     // a~_bend = (1/kb)/h^2
    g_betaGammaHBF = alphaB * g_p.betaB * h;
    g_alphaBF      = alphaB;
    g_hFrame       = h;
    // STEP 7: recompute the PER-TET alpha (1/h^2-scaled, §12 R-compliance) once per frame  - muV/lambdaV
    // are h-independent material x rest-volume bakes from xpbd_build_tets.
    g_doStrainF = (g_haveTets && g_T > 0 && g_p.youngE > 0.f);
    if (g_doStrainF)
        k_ComputeTetAlpha<<<Gr(g_T), 64>>>(g_T, 1.f / (h * h), d_tetMuV, d_tetLambdaV, d_tetAlphaDev, d_tetAlphaHyd);
    g_frameReady   = true;
    return 0;
}

extern "C" int xpbd_substep(void)
{
    if (!g_ready) return XPBD_ERR_NOT_READY;
    if (!g_frameReady) return XPBD_ERR_NO_FRAME;        // no begin_frame yet, or invalidated by
                                                        // set_params / re-init / set_bending
    float h = g_hFrame;
    float3 g = mkf3(g_p.gx, g_p.gy, g_p.gz);
    if (g_bound) {
        // §9-(4): refresh liveness + invMass from the LIVE shared flags at the START of every substep,
        // BEFORE predict  - a cut.cu sever/debris-freeze committed since the last substep takes effect now.
        // STEP 7: the same refresh re-derives d_tetActive (domain T) from the identical live flags.
        int refreshMax = g_N;
        if (g_E > refreshMax) refreshMax = g_E;
        if (g_B > refreshMax) refreshMax = g_B;
        if (g_T > refreshMax) refreshMax = g_T;
        k_RefreshLiveness<<<Gr(refreshMax), 64>>>(g_N, g_E, g_B, g_T,
            g_liveNbrIdx, d_edgeOwnerSlot, d_edgeI, d_edgeJ,
            g_liveBendPairs, d_bendSrcIdx,
            g_liveActive, g_livePinned, g_liveMass,
            d_tetIds, d_tetEdgeOwnerSlot,
            d_edgeActive, d_bendActive, d_invMass, d_tetActive);
    }
    k_predict<<<Gr(g_N), 64>>>(g_N, h, g,
                               g_bound ? g_liveExtForce : nullptr,
                               d_invMass, d_pos, d_prevPos, d_vel);
    if (g_E > 0)   cudaMemsetAsync(d_lambda,     0, (size_t)g_E * sizeof(float));   // reset multipliers /substep
    if (g_doBendF) cudaMemsetAsync(d_bendLambda, 0, (size_t)g_B * sizeof(float));
    if (g_doStrainF) {
        cudaMemsetAsync(d_tetLambdaDevL, 0, (size_t)g_T * sizeof(float));
        cudaMemsetAsync(d_tetLambdaHydL, 0, (size_t)g_T * sizeof(float));
    }
    for (int it = 0; it < g_itersF; it++) {
            for (int c = 0; c < g_numColors; c++)               // structural distance constraints
                if (g_colorCnt[c] > 0)
                    k_solve_edges<<<Gr(g_colorCnt[c]), 64>>>(g_colorOff[c], g_colorCnt[c],
                        g_alphaTildeF, g_p.omega, g_betaGammaHSF, d_prevPos,
                        d_edgeI, d_edgeJ, d_restLen, d_edgeActive, d_invMass, d_lambda, d_pos);
            if (g_doBendF)                                       // then bending angle constraints
                for (int c = 0; c < g_bNumColors; c++)
                    if (g_bColorCnt[c] > 0)
                        k_solve_bending<<<Gr(g_bColorCnt[c]), 64>>>(g_bColorOff[c], g_bColorCnt[c],
                            g_alphaBF, g_p.omega, g_betaGammaHBF, d_prevPos,
                            d_bendI, d_bendJ, d_bendK, d_bendTheta0, d_bendActive, d_invMass, d_bendLambda, d_pos);
            if (g_doStrainF)                                     // STEP 7: then per-tet SNH (dev+hyd fused)
                for (int c = 0; c < g_tNumColors; c++)
                    if (g_tColorCnt[c] > 0)
                        k_solve_tet<<<Gr(g_tColorCnt[c]), 64>>>(g_tColorOff[c], g_tColorCnt[c], g_p.omega,
                            d_tetIds, d_tetBr0, d_tetBr1, d_tetBr2, d_tetActive, d_tetAlphaDev, d_tetAlphaHyd,
                            d_invMass, d_tetLambdaDevL, d_tetLambdaHydL, d_pos);
        }
    // Velocity write, split so the STEP 7 perpendicular cs post-pass runs strictly between the raw write
    // and the velScale (Rayleigh/legacy-damping) multiply (design §4[Structural]).
    k_update_velocity<<<Gr(g_N), 64>>>(g_N, 1.f / h, d_invMass, d_pos, d_prevPos, d_vel);
    if (g_p.betaS > 0.f)
        for (int c = 0; c < g_numColors; c++)
            if (g_colorCnt[c] > 0)
                k_perp_damp_edges<<<Gr(g_colorCnt[c]), 64>>>(g_colorOff[c], g_colorCnt[c], g_p.betaS, h,
                    d_pos, d_edgeI, d_edgeJ, d_edgeActive, d_invMass, d_vel);
    k_apply_velscale<<<Gr(g_N), 64>>>(g_N, g_velScaleF, d_invMass, d_vel);
    return -(int)cudaGetLastError();                        // async launch check; caller syncs at frame end
}

extern "C" int xpbd_step(float dt)
{
    int rc = xpbd_begin_frame(dt);
    if (rc != 0) return rc;
    int N = g_p.numSubsteps > 0 ? g_p.numSubsteps : 1;
    for (int sub = 0; sub < N; sub++) {
        rc = xpbd_substep();
        if (rc != 0) return rc;
    }
    cudaError_t e = cudaGetLastError();
    if (e == cudaSuccess) e = cudaDeviceSynchronize();
    return (e == cudaSuccess) ? 0 : -(int)e;
}

extern "C" int xpbd_get_positions(float* out3)
{
    if (!g_ready) return XPBD_ERR_NOT_READY;
    if (!out3) return XPBD_ERR_BAD_ARG;
    return -(int)cudaMemcpy(out3, d_pos, (size_t)g_N * sizeof(float3), cudaMemcpyDeviceToHost);
}

extern "C" int xpbd_get_velocities(float* out3)
{
    if (!g_ready) return XPBD_ERR_NOT_READY;
    if (!out3) return XPBD_ERR_BAD_ARG;
    return -(int)cudaMemcpy(out3, d_vel, (size_t)g_N * sizeof(float3), cudaMemcpyDeviceToHost);
}

extern "C" void xpbd_shutdown()
{
    if (!g_bound) {                       // BOUND mode: d_pos/d_vel ALIAS the shared live buffers
        cudaFree(d_pos);                  // (recon_corner_pos()/physics_vel())  - owned by dc_recon/
        cudaFree(d_vel);                  // physics.cu, NEVER freed here.
    }
    d_pos = nullptr; d_vel = nullptr;
    cudaFree(d_prevPos); d_prevPos = nullptr;
    cudaFree(d_invMass); d_invMass = nullptr;
    cudaFree(d_edgeOwnerSlot); d_edgeOwnerSlot = nullptr;
    g_liveNbrIdx = nullptr; g_liveBendPairs = nullptr; g_liveActive = nullptr; g_livePinned = nullptr;
    g_liveMass = nullptr; g_liveExtForce = nullptr;
    g_dimsX = g_dimsY = g_dimsZ = 0; g_bound = false;
    cudaFree(d_edgeI);   d_edgeI = nullptr;
    cudaFree(d_edgeJ);   d_edgeJ = nullptr;
    cudaFree(d_restLen); d_restLen = nullptr;
    cudaFree(d_lambda);  d_lambda = nullptr;
    cudaFree(d_edgeActive); d_edgeActive = nullptr;
    cudaFree(d_edgeSrcIdx); d_edgeSrcIdx = nullptr;
    cudaFree(d_bendI); d_bendI = nullptr; cudaFree(d_bendJ); d_bendJ = nullptr; cudaFree(d_bendK); d_bendK = nullptr;
    cudaFree(d_bendTheta0); d_bendTheta0 = nullptr; cudaFree(d_bendActive); d_bendActive = nullptr;
    cudaFree(d_bendLambda); d_bendLambda = nullptr;
    cudaFree(d_bendSrcIdx); d_bendSrcIdx = nullptr;
    // STEP 7 tet buffers.
    cudaFree(d_tetIds); d_tetIds = nullptr;
    cudaFree(d_tetBr0); d_tetBr0 = nullptr; cudaFree(d_tetBr1); d_tetBr1 = nullptr; cudaFree(d_tetBr2); d_tetBr2 = nullptr;
    cudaFree(d_tetMuV); d_tetMuV = nullptr; cudaFree(d_tetLambdaV); d_tetLambdaV = nullptr;
    cudaFree(d_tetAlphaDev); d_tetAlphaDev = nullptr; cudaFree(d_tetAlphaHyd); d_tetAlphaHyd = nullptr;
    cudaFree(d_tetLambdaDevL); d_tetLambdaDevL = nullptr; cudaFree(d_tetLambdaHydL); d_tetLambdaHydL = nullptr;
    cudaFree(d_tetActive); d_tetActive = nullptr;
    cudaFree(d_tetEdgeOwnerSlot); d_tetEdgeOwnerSlot = nullptr;
    g_T = 0; g_tColorOff.clear(); g_tColorCnt.clear(); g_tNumColors = 0; g_haveTets = false; g_doStrainF = false;
    g_colorOff.clear(); g_colorCnt.clear(); g_numColors = 0;
    g_bColorOff.clear(); g_bColorCnt.clear(); g_bNumColors = 0;
    g_edgeSlotOfInput.clear(); g_bendSlotOfInput.clear();
    g_N = g_E = g_B = 0; g_ready = false; g_frameReady = false;
}
