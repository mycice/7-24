// xpbd_livepath_test.cu  - migration STEP-4 VERIFY gates (design rev-B §9-(4)): the XPBD solver bound
// to the LIVE pipeline (recon_init -> physics_init[xpbd_bind] -> physics_step[begin_frame + N x
// {substep; cut tick}]). Headless: drives the REAL dc_recon + physics + xpbd sources on a synthetic
// 4x4x4-voxel grid (5^3 = 125 corners), no Unity, no C# marshalling.
//
//   GATE B  golden transfer   - the BOUND live path (physics_step) produces the SAME trajectory as the
//           STANDALONE module (xpbd_step on private buffers) for the identical lattice/params: same
//           kernels, same order, same coefficients (extForce/mass == g bitwise at mass=1).
//           E1 DENTS the tissue, E2 HOLDS under sustained force, E3 responds to a force reversal, E4
//           SPRINGS BACK when removed; E5 is the design-mandated MOVING-tool no-lag gate  - a K-frame-stale
//           ramp measures the ONSET dent (rising edge, saturation-immune): the real 1-frame staleness
//           dents PROMPTLY, an 8-frame lag DELAYS the onset (< 40%  - a built-in mutation/vacuity guard),
//           no overshoot. (The full video-speed sweep lag/overshoot characterization is an in-app Unity
//           gate  - a coarse 4^3 lightly-damped lattice is not a faithful swept-tool proxy.)
//   GATE A  divergence check  - after the swapped physics_step, recon_readback_corner_pos() (the buffer
//           BACK and carries the live sagging velocities, pinned exactly zero.
//   GATE C0 MID-FRAME sever (the spec's literal VERIFY shape)  - sever ONE edge BETWEEN substeps of an
//           OPEN frame (begin_frame + substeps, the physics_step loop shape) and READ BACK d_edgeActive:
//           the live count drops by exactly one at the very next substep  - discriminates the mandated
//           per-substep k_RefreshLiveness from a non-conformant per-frame refresh.
//   GATE C  sever propagation  - the EXACT k_SeverLinks write pattern (nbrIdx[6p+s]=-1 on BOTH half
//           slots, cut.cu:423-424) on the LIVE physics_nbr_idx() drops the constraints the SAME frame
//           (k_RefreshLiveness): the lower slab separates and free-falls, the pinned upper half holds.
//   GATE D  debris freeze + multi-sever session  - active[c]=0 (the cut.cu:451 debris pattern) freezes
//           a corner via the per-substep invMass re-derivation; a SECOND sever later in the session
//           still propagates (the refresh keeps working, not a one-shot).
//
// The remaining §9-(4) VERIFY items need the Unity side and stay on the user checklist: the visual
// gravity-settle golden in the app and the real-blade known-cut regression (needs the C#-built
// conn4096 LUT + cut masks). The severing MECHANISM those exercise is gated here with the identical
// device write patterns.
// Exit code = number of failed checks (0 = all pass).
#include "../cuda/dc_recon.h"
#include "../cuda/physics.h"
#include "../cuda/xpbd/xpbd_solver.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>
#include <vector>
#include <algorithm>
static int g_fail = 0;
static void check(bool ok, const char* name, double got, double want, double tol) {
    if (ok) { printf("  PASS  %-58s got=%.6g want=%.6g\n", name, got, want); }
    else    { printf("  FAIL  %-58s got=%.6g want=%.6g (tol=%.3g)\n", name, got, want, tol); g_fail++; }
}
// ── synthetic grid: DX x DY x DZ voxels, (DX+1)(DY+1)(DZ+1) corners, spacing L ──────────────────────
static const int   DX = 4, DY = 4, DZ = 4;
static const int   CX = DX + 1, CY = DY + 1, CZ = DZ + 1;
static const int   CC = CX * CY * CZ;
static const float L  = 0.2f;
static const float GRAV = -9.81f;
static const float KS = 7.5e4f, CS = 0.92f, KB = 2.0e4f, CB = 0.9f, ALPHA = 2.0f;
static const int   NSUB = 20;
static const float DT = 0.02f;            // the live physicsDt (LiverCudaManager.cs)
// STEP 7: matches the REAL production corner-index convention (C# GridConventions.CornerId(i,j,k,dims)
// = i + (dims.x+1)*(j + (dims.y+1)*k), confirmed against dc_recon.cu's corner-id decode)  - xpbd_build_tets
// assumes this EXACT formula when resolving a cell's 8 corners from (x,y,z), so the synthetic lattice must
// use it too (a different arithmetic-but-still-internally-consistent scheme would silently hand the tet
// builder the WRONG corners for each cell, since it reads back real device buffers by global id).
static inline int cid(int x, int y, int z) { return x + CX * (y + CY * z); }
// physics.cu's BendPairGpu layout (32 bytes)  - local copy, the codebase convention.
struct BendPairHost { int i, j, k; float theta0; int alive, _p0, _p1, _p2; };
// Build the lattice: positions, pinned (top layer y==CY-1), 6-slot nbrIdx/restNbr
// (slot order 0=-x,1=+x,2=-y,3=+y,4=-z,5=+z  - cut.cu k_SeverLinks writes s=2*axis+1 / 2*axis),
// and bending triples along X and Z ONLY (so no bend element spans a horizontal cut plane -
// the synthetic sever in GATE C then only needs edge writes, mirroring a clean planar cut).
struct Lattice {
    std::vector<float> pos, restNbr, mass, extForce;
    std::vector<int>   pinned, active, nbrIdx;
    std::vector<BendPairHost> bends;
};
static void buildLattice(Lattice& g) {
    g.pos.resize(3 * CC); g.mass.assign(CC, 1.f); g.pinned.assign(CC, 0); g.active.assign(CC, 1);
    g.extForce.resize(3 * CC); g.nbrIdx.assign(6 * CC, -1); g.restNbr.assign(3 * 6 * CC, 0.f);
    for (int x = 0; x < CX; x++) for (int y = 0; y < CY; y++) for (int z = 0; z < CZ; z++) {
        int c = cid(x, y, z);
        g.pos[3*c+0] = x * L; g.pos[3*c+1] = y * L; g.pos[3*c+2] = z * L;
        g.pinned[c] = (y == CY - 1) ? 1 : 0;                       // anchored top layer
        g.extForce[3*c+0] = 0.f; g.extForce[3*c+1] = GRAV * g.mass[c]; g.extForce[3*c+2] = 0.f;
        int coords[6][3] = { {x-1,y,z},{x+1,y,z},{x,y-1,z},{x,y+1,z},{x,y,z-1},{x,y,z+1} };
        for (int s = 0; s < 6; s++) {
            int nx = coords[s][0], ny = coords[s][1], nz = coords[s][2];
            if (nx < 0 || nx >= CX || ny < 0 || ny >= CY || nz < 0 || nz >= CZ) continue;
            int n = cid(nx, ny, nz);
            g.nbrIdx[6*c+s] = n;
            g.restNbr[3*(6*c+s)+0] = (nx - x) * L;
            g.restNbr[3*(6*c+s)+1] = (ny - y) * L;
            g.restNbr[3*(6*c+s)+2] = (nz - z) * L;
        }
    }
    // bending triples along X and Z (apex m, arms m-1/m+1 on the axis), theta0 = pi (straight).
    for (int y = 0; y < CY; y++) for (int z = 0; z < CZ; z++)
        for (int x = 1; x < CX - 1; x++)
            g.bends.push_back({ cid(x,y,z), cid(x-1,y,z), cid(x+1,y,z), 3.14159265f, 1, 0,0,0 });
    for (int y = 0; y < CY; y++) for (int x = 0; x < CX; x++)
        for (int z = 1; z < CZ - 1; z++)
            g.bends.push_back({ cid(x,y,z), cid(x,y,z-1), cid(x,y,z+1), 3.14159265f, 1, 0,0,0 });
}
// Init the LIVE pipeline: recon (synthetic grid, no isect  - physics only needs the shared cornerPos)
// then physics (which xpbd_binds and swaps the step body).
static int initLivePath(const Lattice& g) {
    ReconInitDesc rd = {};
    rd.voxelCount = DX * DY * DZ; rd.cornerCount = CC;
    rd.isectCount = 0; rd.surfaceEdgeCount = 0; rd.triCapacity = 16;
    rd.dimsX = DX; rd.dimsY = DY; rd.dimsZ = DZ; rd.L = L; rd.qefIters = 4;
    std::vector<int> voxelCorner(8 * rd.voxelCount), vOff(rd.voxelCount, 0), vCnt(rd.voxelCount, 0);
    int vi = 0;
    for (int x = 0; x < DX; x++) for (int y = 0; y < DY; y++) for (int z = 0; z < DZ; z++, vi++)
        for (int k = 0; k < 8; k++)
            voxelCorner[8*vi + k] = cid(x + (k & 1), y + ((k >> 1) & 1), z + ((k >> 2) & 1));
    unsigned char dummy[64] = {};                               // non-null for the 0-count uploads
    int rc = recon_init(&rd, g.pos.data(), voxelCorner.data(), vOff.data(), vCnt.data(), dummy, dummy);
    if (rc != 0) { printf("  FAIL  recon_init rc=%d\n", rc); return rc; }
    PhysicsInitDesc pd = {};
    pd.cornerCount = CC; pd.bendPairCount = (int)g.bends.size();
    pd.ks = KS; pd.cs = CS; pd.kb = KB; pd.cb = CB;
    pd.hInit = 1e-4f; pd.hMax = 1.5e-3f;                        // DEAD under XPBD (kept for ABI)
    pd.maxSubsteps = NSUB;                                       // REPURPOSED as the fixed N
    pd.alpha = ALPHA;
    rc = physics_init(&pd, g.mass.data(), g.pinned.data(), g.active.data(), g.extForce.data(),
                      g.nbrIdx.data(), g.restNbr.data(), g.bends.data());
    if (rc != 0) { printf("  FAIL  physics_init rc=%d\n", rc); return rc; }
    return 0;
}
// ── STEP 7 gate G1 (design §9-(7)): REST-STRESS  - a zero-gravity tet lattice holds det(F)=1 (and the
// rest-zero deviatoric C_dev=0) with NO drift. At rest F=I exactly satisfies BOTH constraints by
// construction; if the per-tet gradient/column algebra has a sign or indexing error, the rest
// configuration would spuriously drift even with no external force at all. Strain is ON by default
// (physics_init bakes youngE=3e4 unconditionally)  - no param override needed.
// SCOPE (audit round-1 finding, minor): this is a REST-EQUILIBRIUM check ONLY. It structurally cannot
// catch a "tets dispatch even when youngE<=0" gating regression, because C_dev=C_hyd=0 identically at
// F=I regardless of whether the dispatch gate is correct (dLambda=0 either way  - mutation-tested: a
// scratchpad build with g_doStrainF's youngE>0.f conjunct removed still passes G1 with got=0, want=0,
// bit-identical to baseline). That specific gating condition is exercised only incidentally by GATE B's
// (unrelated) bind-plumbing comparison, which happens to use a tet-less/youngE=0 standalone reference.
static float g1_rest_stress(const Lattice& g0) {
    Lattice g = g0;
    std::fill(g.extForce.begin(), g.extForce.end(), 0.f);   // zero gravity: nothing should move at all
    if (initLivePath(g) != 0) return -1e9f;
    std::vector<float> p0(3 * CC); recon_readback_corner_pos(p0.data());
    for (int f = 0; f < 60; f++) if (physics_step(DT) != 0) { physics_shutdown(); recon_shutdown(); return -1e9f; }
    std::vector<float> p1(3 * CC); recon_readback_corner_pos(p1.data());
    float maxMove = 0.f;
    for (int k = 0; k < 3 * CC; k++) maxMove = fmaxf(maxMove, fabsf(p1[k] - p0[k]));
    physics_shutdown(); recon_shutdown();
    return maxMove;
}
// ── STEP 7 gate G2 (design §9-(7)): ISOTROPY / parity-alternation  - STRUCTURAL verification. [audit
// round-1 fix, MAJOR finding, confirmed by build+mutation-test]: a DYNAMIC force-response comparison
// (pull one corner along +X/+Z/diagonal, compare displacement magnitude) was proven EMPIRICALLY
// INSENSITIVE to a disabled-alternation regression (forced flip=0 unconditionally) at every direction
// pair, pull corner, and force magnitude tried (isotropy ratio ~1.0-1.08 in BOTH the correct baseline
// and the mutated build, indistinguishable)  - the pre-existing structural+bending springs dominate the
// elastic response on this coarse 4-voxel lattice, swamping the tets' own small orientation bias at any
// deformation scale tested. Root cause is geometric, not a tuning miss: X-vs-Z at ANY interior corner is
// provably blind by symmetry (the parity pattern (x+y+z)&1 and the unflipped main diagonal (1,1,1) are
// BOTH invariant under x<->z exchange, so no X-vs-Z comparison can ever detect this class of bug,
// independent of tolerance).
// FIX: inspect the BUILT topology directly instead of its downstream force response  - for two adjacent
// cells of DIFFERING parity, confirm their tets' main-diagonal corner sets are actually DIFFERENT
// (flip=0 vs flip=1), which is the literal design §4[Strain] requirement. This is a 100%-deterministic
// check of the mechanism itself, not a force-response proxy, so no mutation can hide from it.
static int g2_parity_alternation_check(const Lattice& g0) {
    int fails = 0;
    Lattice g = g0;
    if (initLivePath(g) != 0) return 1;
    int T = xpbd_tet_count();
    if (T <= 0) { printf("  FAIL  G2 STEP7 parity: zero tets built\n"); physics_shutdown(); recon_shutdown(); return ++fails; }
    std::vector<int> tetIds(4 * T);
    if (xpbd_readback_tet_ids(tetIds.data()) != 0) { printf("  FAIL  G2 STEP7 parity: tet id readback failed\n"); physics_shutdown(); recon_shutdown(); return ++fails; }
    static const int TET_TABLE[6][4] = { {0,1,3,7}, {0,3,2,7}, {0,2,6,7}, {0,6,4,7}, {0,4,5,7}, {0,5,1,7} };
    auto expectedSet = [&](int x, int y, int z, int flip, int ti, int out4[4]) {
        for (int k = 0; k < 4; k++) {
            int li = TET_TABLE[ti][k] ^ flip;
            int dx = li & 1, dy = (li >> 1) & 1, dz = (li >> 2) & 1;
            out4[k] = cid(x + dx, y + dy, z + dz);
        }
    };
    auto sameSet = [&](const int a[4], const int* b) {   // unordered 4-element set equality
        int ac[4] = { a[0], a[1], a[2], a[3] }; int bc[4] = { b[0], b[1], b[2], b[3] };
        std::sort(ac, ac + 4); std::sort(bc, bc + 4);
        return ac[0] == bc[0] && ac[1] == bc[1] && ac[2] == bc[2] && ac[3] == bc[3];
    };
    auto countMatchesForFlip = [&](int x, int y, int z, int flip)->int {
        int matches = 0;
        for (int ti = 0; ti < 6; ti++) {
            int exp4[4]; expectedSet(x, y, z, flip, ti, exp4);
            for (int e = 0; e < T; e++) if (sameSet(exp4, &tetIds[4*e])) { matches++; break; }
        }
        return matches;
    };
    // Two INTERIOR-adjacent cells with DIFFERING parity: (1,1,1) sum=3 odd->flip=1, (2,1,1) sum=4
    // even->flip=0. Each cell's 6 tets must match ALL 6 of its OWN expected-flip quadruples and NONE of
    // the other flip's -- this directly proves adjacent cells alternate, the literal design requirement.
    struct { int x, y, z, expectFlip; } probes[2] = { {1,1,1,1}, {2,1,1,0} };
    for (auto& p : probes) {
        int matchOwn = countMatchesForFlip(p.x, p.y, p.z, p.expectFlip);
        int matchOther = countMatchesForFlip(p.x, p.y, p.z, 1 - p.expectFlip);
        printf("  info  G2 parity cell(%d,%d,%d) parity=%d: matches-expected-flip=%d/6 matches-OTHER-flip=%d/6\n",
               p.x, p.y, p.z, p.expectFlip, matchOwn, matchOther);
        if (matchOwn != 6)
            { printf("  FAIL  G2 STEP7 parity: cell(%d,%d,%d) tets do NOT match its expected flip=%d (got %d/6)\n", p.x, p.y, p.z, p.expectFlip, matchOwn); fails++; }
        if (matchOther != 0)
            { printf("  FAIL  G2 STEP7 parity: cell(%d,%d,%d) has %d tet(s) matching the OTHER (wrong) flip\n", p.x, p.y, p.z, matchOther); fails++; }
    }
    physics_shutdown(); recon_shutdown();
    return fails;
}
// ── STEP 7 gate G3 (design §9-(7)): CUT-WITH-TETS separation  - sever a STATIONARY hanging layer (y=3,
// held only by the pinned y=4 top) the SAME way GATE D's second cut does, but now with the tet strain
// constraint LIVE (default). If k_DeactivateTets (folded into k_RefreshLiveness, xpbd_solver.cu) failed
// to drop a tet spanning the cut plane, its 4 corners would stay glued across the severed edges and the
// layer would NOT fall (or fall far less than the spring-only GATE D case)  - this is the design's
// mandated "measured separation gap [the halves must NOT stay glued]" check. The identical nbrIdx=-1
// mechanism also covers the tear-through-tet case (k_TearOverstretch severs via the SAME write pattern,
// so the tet-liveness refresh cannot distinguish a cut sever from a tear sever).
static float g3_cut_with_tets(const Lattice& g0) {
    if (initLivePath(g0) != 0) return -1e9f;
    std::vector<float> pBefore(3 * CC); recon_readback_corner_pos(pBefore.data());
    float y3Before = 0.f; int y3Cnt = 0;
    for (int c = 0; c < CC; c++) if ((c / CX) % CY == 3) { y3Before += pBefore[3*c+1]; y3Cnt++; }
    int* liveNbr = physics_nbr_idx();
    int minusOne = -1;
    for (int x = 0; x < CX; x++) for (int z = 0; z < CZ; z++) {
        int a = cid(x, 3, z), b = cid(x, 4, z);
        cudaMemcpy(liveNbr + 6*a + 3, &minusOne, sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(liveNbr + 6*b + 2, &minusOne, sizeof(int), cudaMemcpyHostToDevice);
    }
    for (int f = 0; f < 60; f++) if (physics_step(DT) != 0) { physics_shutdown(); recon_shutdown(); return -1e9f; }
    std::vector<float> pAfter(3 * CC); recon_readback_corner_pos(pAfter.data());
    float y3After = 0.f;
    for (int c = 0; c < CC; c++) if ((c / CX) % CY == 3) y3After += pAfter[3*c+1];
    physics_shutdown(); recon_shutdown();
    return (y3Before - y3After) / (float)y3Cnt;
}
int main() {
    printf("== XPBD live-path (step-4 rebind) gates ==\n");
    Lattice g; buildLattice(g);
    // ── GATE B part 1: the STANDALONE module trajectory (run FIRST  - a later xpbd_init would unbind
    // the live path). Identical lattice, edges from the canonical positive-slot dedup, identical params.
    std::vector<float> invMass(CC);
    for (int c = 0; c < CC; c++) invMass[c] = g.pinned[c] ? 0.f : 1.f;
    std::vector<int> edges; std::vector<float> rest;
    for (int c = 0; c < CC; c++)
        for (int axis = 0; axis < 3; axis++) {
            int s = 2 * axis + 1, n = g.nbrIdx[6*c+s];
            if (n < 0) continue;
            edges.push_back(c); edges.push_back(n); rest.push_back(L);
        }
    std::vector<int> bijk; std::vector<float> bth;
    for (const BendPairHost& b : g.bends) { bijk.push_back(b.i); bijk.push_back(b.j); bijk.push_back(b.k); bth.push_back(b.theta0); }
    if (xpbd_init(CC, g.pos.data(), invMass.data(), (int)rest.size(), edges.data(), rest.data()) != 0)
        { printf("  FAIL standalone init\n"); return 1; }
    if (xpbd_set_bending((int)bth.size(), bijk.data(), bth.data()) != 0)
        { printf("  FAIL standalone set_bending\n"); return 1; }
    XpbdParams sp = { NSUB, 1, 1.0f, 0.f, KS, 0.f, GRAV, 0.f, KB, ALPHA, CS, CB };
    xpbd_set_params(&sp);
    std::vector<float> refPos(3 * CC);
    for (int f = 0; f < 60; f++) if (xpbd_step(DT) != 0) { printf("  FAIL standalone step\n"); return 1; }
    xpbd_get_positions(refPos.data());
    xpbd_shutdown();
    // ── LIVE PATH up: recon + physics (physics_init xpbd_binds; physics_step runs the XPBD body). ──
    if (initLivePath(g) != 0) return 1;
    // STEP 7: physics_init unconditionally builds tets (youngE=3e4, physics.cu). GATE A/B below compare
    // the bound path against the STANDALONE reference above, which has NO tets (xpbd_build_tets is
    // bound-mode only  - the standalone module has no voxel-grid structure to build them from). For that
    // comparison to stay apples-to-apples, disable the strain solve for this shared-state block (tets
    // stay BUILT, just inert  - g_doStrainF re-checks g_p.youngE fresh every begin_frame, so this does not
    // require rebuilding anything). The dedicated STEP 7 gates (G1-G3, run on fresh instances after F)
    // re-enable it to test the tet physics on its own terms.
    xpbd_set_params(&sp);
    // ── GATE A: divergence check  - the swapped physics_step deforms the SHARED recon cornerPos, and
    std::vector<float> p0(3 * CC), p1(3 * CC);
    recon_readback_corner_pos(p0.data());
    {   // velocity leg: after ONE frame the lattice is visibly sagging  - unpinned vy < 0, pinned == 0.
        int rc = physics_step(DT);
        if (rc != 0) { printf("  FAIL  physics_step rc=%d (frame 0)\n", rc); return ++g_fail; }
        std::vector<float> v(3 * CC);
        cudaMemcpy(v.data(), physics_vel(), (size_t)CC * 3 * sizeof(float), cudaMemcpyDeviceToHost);
        float minVy = 0.f, pinnedV = 0.f;
        for (int c = 0; c < CC; c++) {
            if (g.pinned[c]) pinnedV = fmaxf(pinnedV, fabsf(v[3*c]) + fabsf(v[3*c+1]) + fabsf(v[3*c+2]));
            else             minVy   = fminf(minVy, v[3*c+1]);
        }
        check(minVy < -1e-3f, "A divergence: physics_vel() carries LIVE velocities (sagging)", minVy, 0, 0);
        check(pinnedV == 0.f, "A divergence: pinned velocities EXACTLY zero in the shared vel", pinnedV, 0, 0);
    }
    for (int f = 1; f < 60; f++) {
        int rc = physics_step(DT);
        if (rc != 0) { printf("  FAIL  physics_step rc=%d (frame %d)\n", rc, f); return ++g_fail; }
    }
    recon_readback_corner_pos(p1.data());
    float maxMove = 0.f, pinnedMove = 0.f;
    for (int c = 0; c < CC; c++) {
        float dx = p1[3*c]-p0[3*c], dy = p1[3*c+1]-p0[3*c+1], dz = p1[3*c+2]-p0[3*c+2];
        float m = sqrtf(dx*dx + dy*dy + dz*dz);
        if (g.pinned[c]) pinnedMove = fmaxf(pinnedMove, m); else maxMove = fmaxf(maxMove, m);
    }
    printf("  info  live sag after %d frames: maxMove=%.5f pinnedMove=%.5g\n", 60, maxMove, pinnedMove);
    check(maxMove > 1e-3f, "A divergence: XPBD deformed the SHARED recon cornerPos", maxMove, 0, 0);
    check(pinnedMove == 0.f, "A divergence: pinned corners EXACT in the shared buffer", pinnedMove, 0, 0);
    // ── GATE B part 2: bound trajectory == standalone trajectory (same kernels/coeffs; mass=1 makes
    // extForce/mass == g bitwise, so the paths should agree to float-exactness). ──
    float maxDiff = 0.f;
    for (int k = 0; k < 3 * CC; k++) maxDiff = fmaxf(maxDiff, fabsf(p1[k] - refPos[k]));
    check(maxDiff < 1e-5f, "B golden transfer: BOUND live path == STANDALONE module", maxDiff, 0, 1e-5);
    // ── GATE C0: MID-FRAME sever, the design-mandated shape (§9-(4) VERIFY: "cut mid-frame; assert
    // the corresponding constraint drops the SAME frame (readback d_edgeActive)"). Drive the bound
    // module through the EXACT physics_step loop shape (begin_frame + substeps  - the public split API),
    // sever ONE edge with the k_SeverLinks pattern BETWEEN substeps of the OPEN frame, and assert the
    // colour-ordered d_edgeActive drops by EXACTLY one at the very next substep  - this discriminates
    // the per-substep k_RefreshLiveness from a (non-conformant) per-frame refresh. ──
    {
        int E = xpbd_get_edge_count();
        std::vector<int> ea0(E), ea1(E);
        if (xpbd_begin_frame(DT) != 0) { printf("  FAIL  begin_frame (C0)\n"); return ++g_fail; }
        for (int s = 0; s < NSUB / 2; s++)
            if (xpbd_substep() != 0) { printf("  FAIL  substep (C0 pre)\n"); return ++g_fail; }
        xpbd_readback_edge_active(ea0.data());                        // mid-frame baseline
        int a0 = cid(1, 0, 1), b0 = cid(2, 0, 1);                     // one interior +x edge, lower slab
        int* liveNbr0 = physics_nbr_idx();
        int minusOne0 = -1;
        cudaMemcpy(liveNbr0 + 6*a0 + 1, &minusOne0, sizeof(int), cudaMemcpyHostToDevice);  // +x slot
        cudaMemcpy(liveNbr0 + 6*b0 + 0, &minusOne0, sizeof(int), cudaMemcpyHostToDevice);  // -x slot
        if (xpbd_substep() != 0) { printf("  FAIL  substep (C0 post)\n"); return ++g_fail; }
        xpbd_readback_edge_active(ea1.data());
        for (int s = NSUB / 2 + 1; s < NSUB; s++)                     // finish the frame cleanly
            if (xpbd_substep() != 0) { printf("  FAIL  substep (C0 tail)\n"); return ++g_fail; }
        cudaDeviceSynchronize();
        int live0 = 0, live1 = 0;
        for (int e = 0; e < E; e++) { live0 += ea0[e]; live1 += ea1[e]; }
        printf("  info  mid-frame sever: live edges %d -> %d (of %d)\n", live0, live1, E);
        check(live0 - live1 == 1, "C0 MID-FRAME sever drops d_edgeActive at the NEXT SUBSTEP (same frame)",
              live0 - live1, 1, 0);
    }
    // ── GATE C: sever propagation  - k_SeverLinks' EXACT write pattern on the LIVE nbrIdx. Cut the
    // horizontal plane between y=2 and y=3: for every (x,2,z), sever its +y slot (s=3) and the
    // neighbour's -y slot (s=2). The lower slab (y<=2) must separate and fall; the pinned upper holds.
    int* liveNbr = physics_nbr_idx();
    int minusOne = -1;
    for (int x = 0; x < CX; x++) for (int z = 0; z < CZ; z++) {
        int a = cid(x, 2, z), b = cid(x, 3, z);
        cudaMemcpy(liveNbr + 6*a + 3, &minusOne, sizeof(int), cudaMemcpyHostToDevice);   // idA's +y slot
        cudaMemcpy(liveNbr + 6*b + 2, &minusOne, sizeof(int), cudaMemcpyHostToDevice);   // idB's -y slot
    }
    // same-frame pickup: one frame after the sever the freed slab is already moving.
    std::vector<float> p2(3 * CC), p3(3 * CC);
    recon_readback_corner_pos(p2.data());
    { int rc = physics_step(DT); if (rc != 0) { printf("  FAIL  physics_step rc=%d (C same-frame)\n", rc); return ++g_fail; } }
    recon_readback_corner_pos(p3.data());
    float slabDrop1 = 0.f;
    for (int c = 0; c < CC; c++) {
        int y = (c / CX) % CY;
        if (y <= 2) slabDrop1 = fmaxf(slabDrop1, p2[3*c+1] - p3[3*c+1]);
    }
    check(slabDrop1 > 1e-4f, "C sever: constraint dropped the SAME frame (slab moving)", slabDrop1, 0, 0);
    for (int f = 0; f < 60; f++) {                                // ~1.2 s of fall
        int rc = physics_step(DT);
        if (rc != 0) { printf("  FAIL  physics_step rc=%d (C fall, frame %d)\n", rc, f); g_fail++; break; }
    }
    recon_readback_corner_pos(p3.data());
    float slabDrop = 0.f, upperDrop = 0.f;
    for (int c = 0; c < CC; c++) {
        int y = (c / CX) % CY;
        float dy = p2[3*c+1] - p3[3*c+1];
        if (y <= 2) slabDrop = fmaxf(slabDrop, dy);
        else if (!g.pinned[c]) upperDrop = fmaxf(upperDrop, dy);
    }
    printf("  info  post-sever: lower-slab max drop=%.4f upper(unpinned) max drop=%.4f\n", slabDrop, upperDrop);
    check(slabDrop > 0.5f, "C sever: freed slab FREE-FALLS (constraints dropped)", slabDrop, 0, 0);
    check(upperDrop < 0.1f, "C sever: pinned upper half HOLDS", upperDrop, 0, 0.1);
    // ── GATE D: debris freeze (cut.cu:451 pattern) + multi-sever session. ──
    // Freeze one falling corner: sever all its remaining links + active=0 -> invMass re-derives to 0.
    int fc = cid(2, 1, 2);
    for (int s = 0; s < 6; s++) {
        int n = 0;
        cudaMemcpy(&n, liveNbr + 6*fc + s, sizeof(int), cudaMemcpyDeviceToHost);
        if (n < 0) continue;
        cudaMemcpy(liveNbr + 6*fc + s, &minusOne, sizeof(int), cudaMemcpyHostToDevice);
        int sBack = (s % 2 == 0) ? s + 1 : s - 1;                 // the neighbour's opposite half slot
        cudaMemcpy(liveNbr + 6*n + sBack, &minusOne, sizeof(int), cudaMemcpyHostToDevice);
    }
    // k_SeverLinks also kills every bend pair that uses a severed edge (cut.cu:427-432)  - mirror it,
    // else the still-alive bends through fc ANCHOR the falling slab to the frozen corner (kb=2e4).
    {
        unsigned char* bendBase = (unsigned char*)physics_bend_pairs();
        int aliveOff = (int)offsetof(BendPairHost, alive);
        int deadFlag = 0;
        for (int b = 0; b < (int)g.bends.size(); b++)
            if (g.bends[b].i == fc || g.bends[b].j == fc || g.bends[b].k == fc)
                cudaMemcpy(bendBase + (size_t)b * sizeof(BendPairHost) + aliveOff,
                           &deadFlag, sizeof(int), cudaMemcpyHostToDevice);
    }
    int zero = 0;
    cudaMemcpy(physics_active() + fc, &zero, sizeof(int), cudaMemcpyHostToDevice);
    std::vector<float> p4(3 * CC), p5(3 * CC);
    recon_readback_corner_pos(p4.data());
    for (int f = 0; f < 30; f++) {
        int rc = physics_step(DT);
        if (rc != 0) { printf("  FAIL  physics_step rc=%d (freeze loop, frame %d)\n", rc, f); g_fail++; break; }
    }
    recon_readback_corner_pos(p5.data());
    float frozenMove = fabsf(p5[3*fc+1] - p4[3*fc+1]) + fabsf(p5[3*fc] - p4[3*fc]) + fabsf(p5[3*fc+2] - p4[3*fc+2]);
    check(frozenMove == 0.f, "D freeze: active=0 debris STOPS (invMass re-derived live)", frozenMove, 0, 0);
    // positive control (a vacuity guard): a NON-frozen neighbour in the same falling slab kept moving
    // over the same window  - proves the freeze is selective, not a global stall.
    int nc = cid(2, 0, 2);
    float ctrlMove = fabsf(p5[3*nc+1] - p4[3*nc+1]);
    check(ctrlMove > 0.1f, "D freeze: non-frozen neighbour KEPT falling (freeze is selective)", ctrlMove, 0, 0);
    // multi-sever: a SECOND planar cut later in the session still propagates. DISCRIMINATING geometry:
    // cut the STATIONARY hanging layer (y=3, held by the pinned top y=4)  - before the cut it does not
    // move at all; it falls ONLY if the refresh picked the new sever up. (Cutting inside the already
    // free-falling slab would be unobservable  - both halves fall identically.)
    recon_readback_corner_pos(p4.data());
    float y3Before = 0.f; int y3Cnt = 0;
    for (int c = 0; c < CC; c++) if ((c / CX) % CY == 3) { y3Before += p4[3*c+1]; y3Cnt++; }
    for (int x = 0; x < CX; x++) for (int z = 0; z < CZ; z++) {
        int a = cid(x, 3, z), b = cid(x, 4, z);
        cudaMemcpy(liveNbr + 6*a + 3, &minusOne, sizeof(int), cudaMemcpyHostToDevice);   // idA's +y slot
        cudaMemcpy(liveNbr + 6*b + 2, &minusOne, sizeof(int), cudaMemcpyHostToDevice);   // idB's -y slot
    }
    for (int f = 0; f < 30; f++) {
        int rc = physics_step(DT);
        if (rc != 0) { printf("  FAIL  physics_step rc=%d (multi-sever loop, frame %d)\n", rc, f); g_fail++; break; }
    }
    recon_readback_corner_pos(p5.data());
    float y3After = 0.f;
    for (int c = 0; c < CC; c++) if ((c / CX) % CY == 3) y3After += p5[3*c+1];
    float layerDrop = (y3Before - y3After) / (float)y3Cnt;
    printf("  info  second cut: y=3 layer mean drop=%.4f (was stationary under the pinned top)\n", layerDrop);
    check(layerDrop > 0.2f, "D multi-sever: a SECOND cut in the session still propagates", layerDrop, 0, 0);
    physics_shutdown();
    recon_shutdown();
    // ── STEP 7 gates (design §9-(7)): per-tet Stable Neo-Hookean strain. Each runs on a FRESH sim
    // (pristine init/shutdown per measurement  - the same order-independence discipline the E5 rework
    // established) so none of them depend on the shared-state block above. ──
    {
        float drift = g1_rest_stress(g);
        if (drift < -1e8f) { printf("  FAIL  G1 rest-stress init/step\n"); g_fail++; }
        else check(drift < 1e-4f, "G1 STEP7 rest-stress: zero-gravity tet lattice holds det(F)=1 (no drift)", drift, 0, 1e-4);
    }
    {
        // PRIMARY gate (audit round-1 fix): direct structural verification that adjacent cells actually
        // alternate their tet main-diagonal (design §4[Strain]) -- see g2_parity_alternation_check's
        // header comment for why the PRIOR dynamic force-response comparison was replaced (empirically
        // proven insensitive by build+mutation-test, not merely under-toleranced).
        int pf = g2_parity_alternation_check(g);
        g_fail += pf;
        printf("  info  G2 STEP7 parity-alternation check: %s (%d failure%s)\n", pf == 0 ? "PASS" : "FAIL", pf, pf == 1 ? "" : "s");
    }
    {
        float layerDrop = g3_cut_with_tets(g);
        if (layerDrop < -1e8f) { printf("  FAIL  G3 cut-with-tets init/step\n"); g_fail++; }
        else {
            printf("  info  G3 cut-with-tets: y=3 layer mean drop=%.4f (stationary before the cut)\n", layerDrop);
            check(layerDrop > 0.1f, "G3 STEP7 cut-with-tets: severed layer SEPARATES (tets do not glue the cut)", layerDrop, 0, 0.1);
        }
    }
    printf("== %s (%d failure%s) ==\n", g_fail == 0 ? "ALL PASS" : "FAILURES", g_fail, g_fail == 1 ? "" : "s");
    return g_fail;
}
