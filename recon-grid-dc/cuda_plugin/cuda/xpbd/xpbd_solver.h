// xpbd_solver.h  - host-callable API for the XPBD soft-body solver (migration STEP 1).
//
// This is the round-3 design (docs/superpowers/specs/2026-07-05-xpbd-combination-round3-design.md):
// replace the mass-spring explicit RK45 integrator with FULL-XPBD, keeping the SAME model topology.
// A structural distance constraint with compliance a~ = (1/ks)/h^2 reproduces a Hookean spring of
// stiffness ks EXACTLY (Macklin/Muller/Chentanez, "XPBD", MIG 2016) but is unconditionally stable at
// any ks. Step 1 implements ONLY: the fixed-N small-steps substep loop, structural distance constraints
// (graph-coloured Gauss-Seidel, no atomics), gravity, and pinned (invMass=0) boundary conditions.
//
// SELF-CONTAINED MODULE: lives entirely under cuda/xpbd/ and does NOT touch physics.cu / the existing
// mass-spring solver. Later stages add bending and volume constraints + wire it to the
// shared cornerPos/nbrIdx buffers; step 1 is a standalone, analytically-verifiable core.
//
// Reference solver structure mirrored from the user's XPBD repo (Simulation/Assets/SurgicalSim/Physics/
// XPBDSolverGPU.cs + XPBDSolver.compute) and github.com/FantasyVR/neohookean_XPBD, reimplemented in CUDA.

#pragma once
#include <cstdint>
#include <vector_types.h>   // float3 (host-includable CUDA type header; same pattern as dc_recon.h)

// ── Status codes. STATE/VALIDATION errors live in a DISJOINT band (-1000..) so the live caller can
// distinguish a recoverable caller-state error from a fatal device fault: CUDA errors are returned as
// -(int)cudaError_t (small negatives, e.g. -1 = cudaErrorInvalidValue, -2 = cudaErrorMemoryAllocation),
// which the old hand-rolled -1/-2/-3 codes ALIASED. 0 = success.
#define XPBD_ERR_NOT_READY       (-1000)   // xpbd_init not (successfully) called
#define XPBD_ERR_NO_FRAME        (-1001)   // xpbd_substep without a valid xpbd_begin_frame (or invalidated
                                           // by xpbd_set_params / re-init / xpbd_set_bending)
#define XPBD_ERR_BAD_DT          (-1002)   // dt does not yield a positive finite substep h (incl. NaN/Inf)
#define XPBD_ERR_COLOR_OVERFLOW  (-1003)   // greedy colouring exhausted 64 colours (vertex degree >= 64):
                                           // FAIL LOUD rather than alias a used colour (no-atomics GS race)
#define XPBD_ERR_BAD_ARG         (-1004)   // null pointer / count / index out of range

extern "C" {

// Solver tunables. h = dt / numSubsteps.  compliance a = 1/ks ;  a~ = a / h^2  (per substep).
struct XpbdParams
{
    int   numSubsteps;   // N  - fixed small-steps count per outer dt (Macklin "Small Steps" SCA 2019)
    int   iters;         // constraint iterations per substep (1 = pure small-steps; >1 accumulates lambda)
    float omega;         // SOR under-relaxation, applied CONSISTENTLY to lambda AND position (so the
                         // compliant equilibrium DL=mg/ks is omega-invariant  - omega only changes the
                         // convergence RATE, not the effective stiffness). omega=1.0 is permitted for
                         // the STANDALONE step-1-3 self-test ONLY (design §3.5); the FULL-constraint-set
                         // default is 0.2-0.5 (production reference XPBDSolverGPU.cs:450 uses 0.2  - the
                         // low-compliance constraints are the worst over-relaxation regime).
                         // Committing omega=1 beyond steps 1-3 REQUIRES the §3.5 stability test at
                         // ks=7.5e4 with structural and bending constraints.
    float damping;       // legacy global post-solve velocity damping in [0,1): v *= (1 - damping) PER
                         // SUBSTEP. NOTE its per-FRAME retention is (1 - damping)^N, so its strength is
                         // N-DEPENDENT (unlike alpha, which is re-derived per-N for N-invariant decay).
                         // It cannot move the static equilibrium (DL=mg/ks), but RETUNE it if N changes;
                         // prefer alpha (Rayleigh) for physical damping. e.g. N=20, damping=0.02 -> 0.98^20
                         // ~= 0.67 per frame (a 33% per-frame velocity kill).
    float ks;            // structural stiffness (paper SS3.4 = 7.5e4) -> compliance 1/ks
    float gx, gy, gz;    // gravity acceleration (e.g. 0, -9.81, 0)
    float kb;            // bending stiffness (paper SS3.4 = 2e4) -> compliance 1/kb (step 2; 0 = no bending)
    // STEP 3 damping (paper SS3.4). alpha = global mass-proportional Rayleigh (1/s), applied as the
    // per-substep implicit velocity term v *= 1/(1+alpha*h). betaS/betaB = Kelvin-Voigt DAMPING STIFFNESSES
    // for the structural / bending constraints (Macklin XPBD 2016 damped-constraint term gamma = a~*beta*h;
    // beta is a damping STIFFNESS, NOT an inverse/compliance). The XPBD damped constraint reproduces a
    // dashpot force f = beta * (relative velocity along the constraint gradient), so to match the paper's
    // cs*(vi-vj) / cb*theta_dot set betaS = cs and betaB = cb DIRECTLY (do NOT divide by ks/kb). 0 = undamped.
    float alpha;         // Rayleigh (global velocity), 1/s
    float betaS;         // structural KV damping STIFFNESS = cs (the dashpot coefficient, NOT cs/ks)
    float betaB;         // bending KV damping STIFFNESS = cb
    // STEP 7 (design rev-B S4[Strain]/S9-(7)): per-tet Stable Neo-Hookean material. E/nu -> mu,lambda
    // (mu = E/(2(1+nu)), lambda = E*nu/((1+nu)(1-2*nu))); alpha_dev = 1/(h^2*mu*V), alpha_hyd =
    // 1/(h^2*lambda*V), V = REST volume (baked per-tet at build time in xpbd_build_tets). nu is PINNED
    // at 0.45 (near-incompressible, design S11 item 3); E is a tuning anchor (order of ks=7.5e4) -- both
    // are read fresh by xpbd_build_tets (call xpbd_set_params with the desired E/nu BEFORE building).
    float youngE;        // Young's modulus E. <=0 disables the strain constraint (tets still built/
                         // tracked for the cut lifecycle, but alpha->+inf so dev/hyd never correct).
    float poisson;       // Poisson ratio nu, design start 0.45.
};

// Allocate + upload. Particles: pos3[3*particleCount], invMass[particleCount] (0 = pinned/infinite mass).
// Edges: edges[2*edgeCount] (i,j vertex indices), restLen[edgeCount]. The host builds a greedy edge
// colouring (no two edges in a colour share a vertex) so each colour is one conflict-free GS dispatch.
// Returns 0 on success, XPBD_ERR_* or -(int)cudaError_t on error. FAILURE SEMANTICS: bad args / colour
// overflow are detected BEFORE any existing instance is torn down (a failed re-init with bad inputs
// preserves the running instance); a DEVICE alloc/upload failure during re-init, however, leaves the
// module shut down (no half-allocated state).
int  xpbd_init(int particleCount, const float* pos3, const float* invMass,
               int edgeCount, const int* edges, const float* restLen);

// Set/replace the solver tunables (safe to call any time after init).
void xpbd_set_params(const XpbdParams* p);

// STEP 2: register the bending elements (paper Eq13-16). Each element is a triple (apex i, arm j, arm k)
// with rest angle theta0 = angle at i between (xi-xj) and (xi-xk). Reproduces the k_AccBending model as an
// XPBD angle constraint C = theta - theta0 with compliance (1/kb)/h^2. ijk = 3*bendCount ints; theta0 =
// bendCount floats. Call AFTER xpbd_init; bendCount 0 (or never called) = structural-only (step 1). The
// bending elements get their own greedy colouring (no two sharing a vertex in a colour). Returns 0,
// XPBD_ERR_*, or -(int)cudaError_t. FAILURE SEMANTICS: the colouring runs BEFORE the existing bend set is
// freed (a colour-overflow failure preserves the old set); a DEVICE alloc/upload failure leaves the
// module bending-less (structural-only), never half-allocated. Invalidates the current frame (§9-(4)
// begin/substep contract)  - call xpbd_begin_frame again.
int  xpbd_set_bending(int bendCount, const int* ijk, const float* theta0);

// Number of bending colour groups (diagnostic / test).
int  xpbd_bend_color_count();

// Advance ONE outer frame `dt` by N internal XPBD substeps. Deforms the device position buffer in place.
// Returns 0 on success, negative CUDA error otherwise. (Thin wrapper over the M2 split below -
// byte-equivalent stepping, kept for the standalone self-test.)
int  xpbd_step(float dt);

// M2 API SPLIT (design rev-B §9-(4)): the live caller (physics_step) must OWN the substep loop so
// cut_ribbon_tick() can fire BETWEEN substeps (one CCD tick per committed sub-interval  - the R-cut
// contract). xpbd_begin_frame(dt) computes ALL h-derived coefficients once (a~ ~1/h^2, gamma ~1/h,
// velScale ~h  - §12 R-compliance: recomputed together); xpbd_substep() advances exactly ONE substep
// (predict -> lambda reset -> colour sweeps -> velocity write) and returns an ASYNC launch-error check
// (the caller syncs at frame end). Calling xpbd_set_params / re-init invalidates the frame: call
// xpbd_begin_frame again. Loop shape: begin_frame(dt); for n in 1..N { substep(); cut_ribbon_tick(); }
int  xpbd_begin_frame(float dt);
int  xpbd_substep(void);

// ── REBIND / GO-LIVE (design rev-B §9-(4)) ───────────────────────────────────────────────────────
// Layout-compatible copy of physics.cu's BendPairGpu  - the codebase convention (cut.cu declares its own
// BendPairGpuC the same way); keeps module isolation (no physics.h include). 32 bytes; layout MUST match.
struct XpbdBendPairGpu { int i, j, k; float theta0; int alive, _p0, _p1, _p2; };

// Bind the solver to the LIVE shared device buffers and derive its topology from them. After a
// successful bind the module projects IN-PLACE on `cornerPos` (= recon_corner_pos(), the buffer
// cut/recon/particleRot all read) and writes `vel` (= physics_vel())  - NO private pos/vel
// buffers on the live path (they remain only for the standalone self-test via xpbd_init).
//   - EDGE LIST: derived ON DEVICE-READBACK from nbrIdx by the canonical dedup  - each lattice edge is
//     emitted ONCE from its POSITIVE slot s = 2*axis+1 (slot order: 0=-x,1=+x,2=-y,3=+y,4=-z,5=+z),
//     restLen = length3(restNbr[6p+s]). Per colour-ordered edge the OWNER SLOT (6p+s) is persisted on
//     device; per colour-ordered bend element the SOURCE INDEX into the live bendPairs array.
//   - SEVER PROPAGATION: a k_RefreshLiveness kernel at the START of every substep derives
//     edgeActive[e] = (nbrIdx[edgeOwnerSlot[e]] >= 0) && active[i] && active[j]  (BOTH k_AccStructural
//     gates in physics.cu k_AccStructural  - an edge to debris-frozen corners must drop too), bendActive[b] =
//     bendPairs[bendSrcIdx[b]].alive (k_AccBending's only gate; a frozen member is handled by w=0),
//     and invMass[c] = (pinned||!active) ? 0 : 1/mass[c]  - from the LIVE shared flags (cut.cu writes
//     nbrIdx=-1 / alive=0 / active=0 at runtime). invMass is NEVER baked. Zero cut.cu changes; a sever
//     is picked up the very next substep. Colouring stays STATIC (a dead edge early-outs).
//   - BIND-TIME STATE: d_prevPos is initialized = cornerPos (no first-frame velocity kick) and the
//     CURRENT `vel` contents are adopted (a moving liver is not zeroed).
//   - PREDICT (bound mode): the acceleration is extForce[c] * invMass[c].
//   - Bend elements DEAD at bind time (alive==0) are not emitted (a sever is permanent).
// Only bendPairs entries [0, bendCapacity) are read. dims is carried for the step-7 tet builder.
// Replaces any prior xpbd_init/xpbd_bind state. Returns 0, XPBD_ERR_*, or -(int)cudaError_t.
int xpbd_bind(float3* cornerPos, float3* vel,
              const int* nbrIdx, const float3* restNbr,
              const void* bendPairs, int bendCapacity,
              const int* active, const int* pinned,
              const float* mass,
              const float3* extForce,
              int dimsX, int dimsY, int dimsZ, int cornerCount);

// M1 SEVER/LIVENESS NOTIFY PATH (design rev-B §9-(4)): the input-edge/element -> colour-slot permutation
// is PERSISTED (host inverse map + device src-index arrays), so a runtime sever propagates to the
// colour-ordered active flags  - the mechanism a cut needs to DROP a constraint (and a debris-freeze
// needs to zero a mass) after go-live. Indices are the CALLER's original input indices (xpbd_init edge
// order / xpbd_set_bending element order). At the rebind step these are superseded by k_RefreshLiveness
// (deriving liveness from the LIVE shared nbrIdx/bendPairs/active/mass every substep); they remain for
// the standalone self-test + targeted host control. Return 0 / negative on error.
int  xpbd_set_edge_active(int inputEdgeIndex, int active);      // 0 = severed (constraint dropped)
int  xpbd_set_bend_active(int inputBendIndex, int active);      // 0 = severed (element dropped)
int  xpbd_set_particle_inv_mass(int particle, float invMass);   // 0 = pinned/frozen (debris freeze)

// Read back the current particle positions to host (out3[3*particleCount]). Returns 0 on success.
int  xpbd_get_positions(float* out3);

// Read back the current particle velocities (out3[3*particleCount])  - for the step-3 damping/settle
// (residual kinetic-energy) verification. Returns 0 on success.
int  xpbd_get_velocities(float* out3);

// Number of colour groups the greedy colouring produced (diagnostic / test).
int  xpbd_color_count();

// Diagnostics for the §9-(4) sever-propagation VERIFY ("readback d_edgeActive"): the number of edges
// and a snapshot of the COLOUR-ORDERED per-edge active flags (out[edgeCount]). In bound mode these are
// re-derived by k_RefreshLiveness each substep, so a readback taken after a substep reflects any sever
// committed before it. Returns count / 0, or XPBD_ERR_* on error.
int  xpbd_get_edge_count();
int  xpbd_readback_edge_active(int* out);

// ── STEP 7: per-tet Stable Neo-Hookean strain (design rev-B §4[Strain]/§9-(7)) ────────────────────
// Builds a 6-tet parity-alternating Kuhn decomposition of every LIVE voxel cell (all 8 corners active
// at build time) from the buffers captured by a prior successful xpbd_bind (dims/cornerPos/active/
// nbrIdx). BOUND MODE ONLY (tets need the voxel-grid structure the standalone self-test has no use
// for). Computes rest Dm^-1 (3 rows) + RestVolume=|det(Dm)|/6 per tet (host, one-time; degenerate tets
// V<eps are skipped), a 4-vertex greedy tet colouring, and per-tet OWNER-EDGE SLOTS (the 3 member axis
// edges, i.e. the cell edges the tet actually spans  - face/body diagonals are not severable
// primitives) for LIVE deactivation: k_RefreshLiveness re-derives d_tetActive[t] every substep from the
// SAME live nbrIdx/active flags edges/bending already read  - a tet dies PERMANENTLY the substep any of
// its 3 member axis edges is severed (nbrIdx<0) or any of its 4 corners goes inactive, exactly
// mirroring the edge/bend liveness mechanism (zero cut.cu changes). Set XpbdParams.youngE/poisson
// BEFORE calling (xpbd_set_params)  - the material bake reads them at build time. Returns 0,
// XPBD_ERR_*, or -(int)cudaError_t. Call ONCE, after xpbd_bind, before the first xpbd_begin_frame.
int xpbd_build_tets();

// Diagnostics: tet count / tet colour-group count (0 before a successful xpbd_build_tets).
int xpbd_tet_count();
int xpbd_tet_color_count();

// Diagnostic (STEP 7 VERIFY, design §9-(7) isotropy gate  - audit round-1 fix): copy the tet corner-id
// quadruples to host, out4T[4*t+0..3] = the 4 corner ids of colour-ordered tet t (out sized
// 4*xpbd_tet_count() ints). A dynamic force-response isotropy test proved empirically INSENSITIVE to a
// disabled-parity-alternation regression on a coarse lattice (the pre-existing structural springs
// dominate the elastic response; the tet-orientation bias is a small secondary correction swamped at any
// force level tried)  - this accessor lets the test verify the MECHANISM directly instead: that two
// adjacent cells of differing parity actually received different main-diagonal corner sets. Returns 0 on
// success (0 tets is not an error), XPBD_ERR_* or -(int)cudaError_t otherwise.
int xpbd_readback_tet_ids(int* out4T);

// Free all device buffers.
void xpbd_shutdown();

} // extern "C"
