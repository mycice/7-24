// xpbd_selftest.cu — standalone verification for XPBD migration STEPS 1+2+3.
// Build: part of the `xpbd_selftest` CMake executable target (links cuda/xpbd/xpbd_solver.cu).
// Verifies the round-3 design acceptance gates:
//   A  single spring: elongation DL == m*g/ks (the compliance a~=(1/ks)/h^2 = exact Hookean spring),
//      at BOTH a soft ks and the paper ks=7.5e4 (analytic + shipped N=20,it=1 config).
//   A2 vertical chain: total tip elongation == m*g/ks * n(n-1)/2 (multi-constraint equilibrium).
//   B  3D lattice at ks=7.5e4: stable — NO NaN/Inf, bounded, SETTLED — over many frames (stability claim).
//   C  pinned (invMass=0) boundary condition never moves.
//   M  omega-INVARIANCE on the kernel: DL=mg/ks is unchanged at omega in {1.0,0.5,0.2} (design §3.5 —
//      omega applied consistently to lambda AND position; catches a position-only-scaled-omega regression).
//   D  bending (step 2): XPBD rest shape matches the RK45 golden (physics.cu forces) at iters=5 AND the
//      shipped iters=1; E  bending stiffness monotone in kb.
//   K  bending (step 2): golden-match at a LARGE-STRAIN, UNEQUAL-ARM tent geometry (design §4 [Bending]
//      gate — the regime where the un-normalized Eq14 gradient is arm-length-anisotropic).
//   F/G/H  damping (step 3): undamped rings vs damped settles to golden; settle monotone in alpha.
//   I  cs Kelvin-Voigt ISOLATION (structural dashpot is ACTIVE, catches the betaS=cs/ks factor-k bug).
//   J  cb Kelvin-Voigt ISOLATION (bending dashpot is ACTIVE, catches the betaB=cb/kb factor-k twin).
//   L  greedy colour-budget overflow (>64) FAILS LOUD (XPBD_ERR_COLOR_OVERFLOW, -1003), never silently
//      aliases (GS-race guard); a failed set_bending leaves the module usable (fail-preserving).
//   N  M2 API split: (xpbd_begin_frame + N x xpbd_substep) steps BITWISE-identically to xpbd_step, with
//      foreign work interleaved between substeps (the live physics_step loop shape).
//   P  M2 frame contract: set_params invalidates the open frame; failed/NaN/Inf begin_frame refuses and
//      does not leave the old frame armed; recovery works.
//   Q  R-compliance: DL=mg/ks preserved across an N change (h-derived coefficients re-derived together).
//   O  M1 liveness notify: a runtime edge/bend sever through the PERSISTED permutation (geometry chosen
//      so the input->slot map is NON-identity) drops the RIGHT constraint; invMass=0 freezes debris.
// Exit code = number of failed checks (0 = all pass).

#include "../cuda/xpbd/xpbd_solver.h"
#include <cstdio>
#include <cmath>
#include <vector>

static int g_fail = 0;
static void check(bool ok, const char* name, double got, double want, double tol) {
    if (ok) { printf("  PASS  %-42s got=%.6g want=%.6g\n", name, got, want); }
    else    { printf("  FAIL  %-42s got=%.6g want=%.6g (tol=%.3g)\n", name, got, want, tol); g_fail++; }
}

// Run `frames` outer steps at dt, then read back positions.
static void settle(float dt, int frames, std::vector<float>& pos) {
    for (int f = 0; f < frames; f++) xpbd_step(dt);
    xpbd_get_positions(pos.data());
}

// ── A: single hanging spring, DL = m*g/ks ────────────────────────────────────────────────────────
static void testSingleSpring(float ks) {
    const float L0 = 1.0f, m = 1.0f, g = 9.81f;
    float pos[6]     = { 0,0,0,  0,-L0,0 };     // p0 pinned (top), p1 mass m (below)
    float invMass[2] = { 0.0f, 1.0f/m };
    int   edges[2]   = { 0, 1 };
    float rest[1]    = { L0 };
    if (xpbd_init(2, pos, invMass, 1, edges, rest) != 0) { printf("  FAIL  init(single ks=%.0f)\n", ks); g_fail++; return; }
    XpbdParams p = { /*N*/10, /*iters*/40, /*omega*/1.0f, /*damping*/0.15f, ks, 0.f, -g, 0.f };
    xpbd_set_params(&p);
    std::vector<float> out(6);
    settle(1.0f/60.0f, 3000, out);
    float drop = 0.f - out[4];                  // p1 world y is out[4]; drop from origin
    float DL   = drop - L0;                     // elongation beyond rest
    float want = m * g / ks;
    char nm[64]; snprintf(nm, sizeof nm, "A single-spring DL (ks=%.0f)", ks);
    check(fabs(DL - want) <= 0.05*want + 1e-6, nm, DL, want, 0.05*want);
    xpbd_shutdown();

    // Re-run at the SHIPPED config {N=20, iters=1} (small-steps) to prove DL=mg/ks is iters-invariant
    // — the analytic check and the stability config are then the same solver settings (review minor).
    float pos2[6] = { 0,0,0, 0,-L0,0 };
    if (xpbd_init(2, pos2, invMass, 1, edges, rest) != 0) { printf("  FAIL init(single-shipcfg)\n"); g_fail++; return; }
    XpbdParams q = { /*N*/20, /*iters*/1, 1.0f, /*damping*/0.05f, ks, 0.f, -g, 0.f };
    xpbd_set_params(&q);
    std::vector<float> out2(6);
    settle(1.0f/60.0f, 6000, out2);
    float DL2 = (0.f - out2[4]) - L0;
    snprintf(nm, sizeof nm, "A single-spring DL shipcfg N=20,it=1 (ks=%.0f)", ks);
    check(fabs(DL2 - want) <= 0.06*want + 1e-6, nm, DL2, want, 0.06*want);
    xpbd_shutdown();
}

// ── A2: vertical chain, total tip elongation = m*g/ks * n(n-1)/2 ──────────────────────────────────
static void testChain() {
    const int n = 8; const float L0 = 1.0f, m = 1.0f, g = 9.81f, ks = 1.0e4f;
    std::vector<float> pos(3*n); std::vector<float> invMass(n); std::vector<int> edges; std::vector<float> rest;
    for (int i = 0; i < n; i++) { pos[3*i]=0; pos[3*i+1]=-L0*i; pos[3*i+2]=0; invMass[i] = (i==0)?0.f:1.f/m; }
    for (int i = 0; i < n-1; i++) { edges.push_back(i); edges.push_back(i+1); rest.push_back(L0); }
    if (xpbd_init(n, pos.data(), invMass.data(), n-1, edges.data(), rest.data()) != 0) { printf("  FAIL init(chain)\n"); g_fail++; return; }
    XpbdParams p = { 10, 40, 1.0f, 0.2f, ks, 0.f, -g, 0.f };
    xpbd_set_params(&p);
    std::vector<float> out(3*n);
    settle(1.0f/60.0f, 4000, out);
    float tipDrop  = 0.f - out[3*(n-1)+1];      // last particle world y
    float restSpan = L0 * (n-1);
    float totalDL  = tipDrop - restSpan;
    float want     = (m*g/ks) * (float)(n*(n-1))/2.0f;   // sum of per-edge elongations
    check(fabs(totalDL - want) <= 0.08*want + 1e-6, "A2 chain total elongation", totalDL, want, 0.08*want);
    xpbd_shutdown();
}

// ── B: 3D lattice at ks=7.5e4, stable (no NaN/Inf, bounded) over many frames ──────────────────────
static void testStability() {
    const int D = 8; const float sp = 0.2f, ks = 7.5e4f, g = 9.81f, m = 1.0f;
    const int n = D*D*D;
    auto id = [&](int x,int y,int z){ return (x*D + y)*D + z; };
    std::vector<float> pos(3*n); std::vector<float> invMass(n); std::vector<int> edges; std::vector<float> rest;
    for (int x=0;x<D;x++) for (int y=0;y<D;y++) for (int z=0;z<D;z++) {
        int i = id(x,y,z); pos[3*i]=x*sp; pos[3*i+1]=y*sp; pos[3*i+2]=z*sp;
        invMass[i] = (y==D-1) ? 0.f : 1.f/m;    // top layer pinned, rest hang
    }
    // 6-neighbour axis-aligned structural edges (each edge once: +x,+y,+z from each corner).
    for (int x=0;x<D;x++) for (int y=0;y<D;y++) for (int z=0;z<D;z++) {
        int i = id(x,y,z);
        if (x+1<D){ edges.push_back(i); edges.push_back(id(x+1,y,z)); rest.push_back(sp); }
        if (y+1<D){ edges.push_back(i); edges.push_back(id(x,y+1,z)); rest.push_back(sp); }
        if (z+1<D){ edges.push_back(i); edges.push_back(id(x,y,z+1)); rest.push_back(sp); }
    }
    int E = (int)rest.size();
    if (xpbd_init(n, pos.data(), invMass.data(), E, edges.data(), rest.data()) != 0) { printf("  FAIL init(lattice)\n"); g_fail++; return; }
    XpbdParams p = { 20, 1, 1.0f, 0.02f, ks, 0.f, -g, 0.f };   // N=20 substeps x 1 iter (the design's real config)
    xpbd_set_params(&p);
    printf("  info  lattice n=%d E=%d colours=%d\n", n, E, xpbd_color_count());
    std::vector<float> out(3*n), out2(3*n);
    settle(1.0f/60.0f, 1000, out);              // settle to equilibrium
    settle(1.0f/60.0f, 1000, out2);             // 1000 MORE frames from the settled state
    bool finite = true; float maxAbs = 0.f, maxDelta = 0.f;
    for (int k = 0; k < 3*n; k++) {
        if (!std::isfinite(out2[k])) finite = false;
        maxAbs   = fmaxf(maxAbs, fabsf(out2[k]));
        maxDelta = fmaxf(maxDelta, fabsf(out2[k] - out[k]));   // frame-1000 -> frame-2000 drift
    }
    check(finite, "B lattice ks=7.5e4 all-finite (no blow-up)", finite?1:0, 1, 0);
    check(maxAbs < 10.0f, "B lattice bounded (|pos| < 10)", maxAbs, 0, 10);              // sags a little, never explodes
    // A slow-growing instability would keep drifting; a stable solve is SETTLED (near-zero further motion).
    check(maxDelta < 1e-2f, "B lattice SETTLED (1000->2000 drift < 1e-2)", maxDelta, 0, 1e-2);
    float topY = out2[3*id(0,D-1,0)+1];
    check(fabs(topY - (D-1)*sp) < 1e-5f, "C pinned BC unmoved", topY, (D-1)*sp, 1e-5);
    xpbd_shutdown();
}

// ── vec helpers for the CPU golden ───────────────────────────────────────────────────────────────
struct V3 { float x,y,z; };
static V3 add(V3 a,V3 b){return {a.x+b.x,a.y+b.y,a.z+b.z};}
static V3 sub(V3 a,V3 b){return {a.x-b.x,a.y-b.y,a.z-b.z};}
static V3 mul(V3 a,float s){return {a.x*s,a.y*s,a.z*s};}
static float dot(V3 a,V3 b){return a.x*b.x+a.y*b.y+a.z*b.z;}
static float nrm(V3 a){return sqrtf(dot(a,a));}
static V3 unit(V3 a){float L=nrm(a); return L>1e-12f?mul(a,1.f/L):V3{0,0,0};}
static V3 crs(V3 a,V3 b){return {a.y*b.z-a.z*b.y, a.z*b.x-a.x*b.z, a.x*b.y-a.y*b.x};}

// ── B) CPU "RK45 golden": the EXACT physics.cu forces (k_AccStructural fs=-ks*(len-L0)*dir + k_AccBending
// Eq14 elastic + gravity), damped-integrated to the integrator-INDEPENDENT rest shape. This is the
// step-(0) golden the design's step-2 gate compares against. ──────────────────────────────────────────
static void goldenSettle(int n, std::vector<V3>& x, const std::vector<float>& invMass,
                         const std::vector<int>& edges, const std::vector<float>& restLen,
                         const std::vector<int>& bend, const std::vector<float>& theta0,
                         float ks, float kb, V3 g, int frames) {
    const float h = 2.0e-4f, velDamp = 0.98f;         // heavy damping -> reach equilibrium (shape is integrator-free)
    int E = (int)restLen.size(), B = (int)theta0.size();
    std::vector<V3> vel(n, {0,0,0}), F(n);
    for (int f = 0; f < frames; f++) {
        for (int p = 0; p < n; p++) F[p] = (invMass[p]>0.f) ? V3{ g.x/invMass[p], g.y/invMass[p], g.z/invMass[p] } : V3{0,0,0};
        for (int e = 0; e < E; e++) {                 // structural: fs = -ks*(len-L0)*dir
            int i=edges[2*e], j=edges[2*e+1]; V3 d=sub(x[i],x[j]); float L=nrm(d); if(L<1e-9f) continue;
            V3 fs = mul(d, -ks*(L-restLen[e])/L); F[i]=add(F[i],fs); F[j]=sub(F[j],fs);
        }
        for (int b = 0; b < B; b++) {                 // bending: paper Eq14 (elastic term only)
            int i=bend[3*b], j=bend[3*b+1], k=bend[3*b+2];
            V3 Nij=unit(sub(x[i],x[j])), Nik=unit(sub(x[i],x[k]));
            float c=fmaxf(-1.f,fminf(1.f,dot(Nij,Nik))); float s2=1.f-c*c; if(s2<=1e-8f) continue;
            float th=acosf(c); float scal=kb*(th-theta0[b]);
            V3 Fij=mul(crs(crs(Nik,Nij),Nij), scal), Fik=mul(crs(crs(Nij,Nik),Nik), scal);
            F[j]=add(F[j],Fij); F[k]=add(F[k],Fik); F[i]=sub(F[i],add(Fij,Fik));
        }
        for (int p = 0; p < n; p++) if (invMass[p]>0.f) {
            vel[p] = mul(add(vel[p], mul(F[p], h*invMass[p])), velDamp);
            x[p]   = add(x[p], mul(vel[p], h));
        }
    }
}

// A CLAMPED cantilever (pin the FIRST TWO particles so the beam starts horizontal and gravity induces
// CURVATURE, which bending resists — a single-pin beam would just rigidly swing to vertical, kb-invariant).
static void buildClampedBeam(int n, float sp, std::vector<float>& pos, std::vector<float>& invMass,
                             std::vector<int>& edges, std::vector<float>& rest,
                             std::vector<int>& bend, std::vector<float>& th0) {
    pos.assign(3*n, 0.f); invMass.assign(n, 0.f); edges.clear(); rest.clear(); bend.clear(); th0.clear();
    for (int i=0;i<n;i++){ pos[3*i]=i*sp; pos[3*i+1]=0; pos[3*i+2]=0; invMass[i]=(i<2)?0.f:1.f; }  // clamp 0,1
    for (int i=0;i<n-1;i++){ edges.push_back(i); edges.push_back(i+1); rest.push_back(sp); }
    for (int m=1;m<n-1;m++){ bend.push_back(m); bend.push_back(m-1); bend.push_back(m+1); th0.push_back(3.14159265f); }
}

// tip sag (|y| of the free end) of a CLAMPED cantilever under gravity, via XPBD, at a given kb.
static float xpbdBeamTipSag(int n, float sp, float ks, float kb) {
    std::vector<float> pos, invMass, rest, th0; std::vector<int> edges, bend;
    buildClampedBeam(n, sp, pos, invMass, edges, rest, bend, th0);
    xpbd_init(n, pos.data(), invMass.data(), (int)rest.size(), edges.data(), rest.data());
    xpbd_set_bending((int)th0.size(), bend.data(), th0.data());
    XpbdParams p = { 20, 5, 1.0f, 0.05f, ks, 0.f, -9.81f, 0.f, kb };
    xpbd_set_params(&p);
    std::vector<float> out(3*n);
    for (int f=0;f<8000;f++) xpbd_step(1.0f/60.0f);
    xpbd_get_positions(out.data());
    float sag = fabsf(out[3*(n-1)+1]);
    xpbd_shutdown();
    return sag;
}

// ── step 2: bending — XPBD rest shape matches the RK45 golden; bending stiffness behaves correctly ─────
static void testBending() {
    const int n = 12; const float sp = 0.15f, ks = 7.5e4f, kb = 2.0e4f, g = 9.81f;
    std::vector<float> pos, invMass, rest, th0; std::vector<int> edges, bend;
    buildClampedBeam(n, sp, pos, invMass, edges, rest, bend, th0);

    // golden (code forces)
    std::vector<V3> xg(n); for (int i=0;i<n;i++) xg[i]={pos[3*i],pos[3*i+1],pos[3*i+2]};
    goldenSettle(n, xg, invMass, edges, rest, bend, th0, ks, kb, {0,-g,0}, 400000);

    // XPBD (structural + bending) — check the golden match at BOTH iters=5 AND the SHIPPED iters=1 config
    // (review minor: the production solver runs iters=1; assert the acceptance number IT produces, not just
    // a more-converged test setting).
    const float beamLen = sp*(n-1);
    for (int it : { 5, 1 }) {
        xpbd_init(n, pos.data(), invMass.data(), (int)rest.size(), edges.data(), rest.data());
        xpbd_set_bending((int)th0.size(), bend.data(), th0.data());
        XpbdParams p = { 20, it, 1.0f, 0.05f, ks, 0.f, -g, 0.f, kb };
        xpbd_set_params(&p);
        if (it == 5) printf("  info  beam bend-colours=%d\n", xpbd_bend_color_count());
        std::vector<float> out(3*n);
        for (int f=0;f<8000;f++) xpbd_step(1.0f/60.0f);
        xpbd_get_positions(out.data());
        xpbd_shutdown();
        float maxDiff = 0.f;
        for (int i=0;i<n;i++){ V3 xi={out[3*i],out[3*i+1],out[3*i+2]}; maxDiff = fmaxf(maxDiff, nrm(sub(xi, xg[i]))); }
        float relDiff = maxDiff / beamLen;
        char nm[80]; snprintf(nm, sizeof nm, "D bending rest-shape ~= RK45 golden (iters=%d, <5%% len)", it);
        printf("  info  beam(iters=%d) golden tipY=%.5f xpbd tipY=%.5f maxDiff=%.5f (%.2f%%)\n",
               it, xg[n-1].y, out[3*(n-1)+1], maxDiff, 100.f*relDiff);
        check(relDiff < 0.05f, nm, relDiff, 0, 0.05);
    }

    // bending stiffness behaves: a STIFFER beam (higher kb) sags LESS at the tip; and adding bending to a
    // structural-only beam reduces sag.
    float sag_noB = xpbdBeamTipSag(n, sp, ks, 0.0f);      // structural only (kb=0 disables bending)
    float sag_soft= xpbdBeamTipSag(n, sp, ks, 5.0e3f);
    float sag_stiff=xpbdBeamTipSag(n, sp, ks, 5.0e4f);
    printf("  info  tip sag: no-bend=%.4f soft-kb=%.4f stiff-kb=%.4f\n", sag_noB, sag_soft, sag_stiff);
    check(sag_stiff < sag_soft && sag_soft < sag_noB, "E bending stiffens the beam (sag monotone in kb)",
          sag_stiff, sag_noB, 0);
}

// ── step 3: damping — run a CLAMPED cantilever released from horizontal, measure residual kinetic
// energy + settled shape at the given damping. ────────────────────────────────────────────────────────
struct DampResult { float ke; std::vector<float> shape; };
static DampResult runBeam(int n, float sp, float ks, float kb,
                          float alpha, float betaS, float betaB, float damping, int frames) {
    std::vector<float> pos, invMass, rest, th0; std::vector<int> edges, bend;
    buildClampedBeam(n, sp, pos, invMass, edges, rest, bend, th0);
    xpbd_init(n, pos.data(), invMass.data(), (int)rest.size(), edges.data(), rest.data());
    xpbd_set_bending((int)th0.size(), bend.data(), th0.data());
    XpbdParams p = { 20, 1, 1.0f, damping, ks, 0.f, -9.81f, 0.f, kb, alpha, betaS, betaB };
    xpbd_set_params(&p);
    for (int f=0; f<frames; f++) xpbd_step(1.0f/60.0f);
    std::vector<float> outp(3*n), outv(3*n);
    xpbd_get_positions(outp.data()); xpbd_get_velocities(outv.data());
    float ke = 0.f;
    for (int i=0;i<n;i++) if (invMass[i]>0.f) { float vx=outv[3*i],vy=outv[3*i+1],vz=outv[3*i+2]; ke += 0.5f*(vx*vx+vy*vy+vz*vz); }
    xpbd_shutdown();
    return { ke, outp };
}
static void testDamping() {
    const int n=12; const float sp=0.15f, ks=7.5e4f, kb=2.0e4f, cs=0.92f, cb=0.9f, beamLen=sp*(n-1);
    // Macklin XPBD beta is a damping STIFFNESS = the Kelvin-Voigt dashpot coefficient c DIRECTLY (NOT c/k).
    // betaS=cs/ks was a factor-k bug that made cs/cb inert (review blocker ws1i08hz6).
    const float betaS = cs, betaB = cb;
    // reference equilibrium: heavy damping, long run -> the settled rest shape (damping-independent minimum)
    DampResult eq = runBeam(n, sp, ks, kb, 12.f, betaS, betaB, 0.20f, 12000);
    // undamped (alpha=cs=cb=0, damping=0): a released cantilever should keep RINGING (energy ~conserved)
    DampResult un = runBeam(n, sp, ks, kb, 0.f, 0.f, 0.f, 0.f, 300);      // 5 s
    // paper damping (alpha Rayleigh + cs/cb Kelvin-Voigt via XPBD): should SETTLE, no perpetual swing
    DampResult dp = runBeam(n, sp, ks, kb, 2.f, betaS, betaB, 0.f, 300);
    printf("  info  damping: KE undamped=%.5g  paper-damped=%.5g  ratio=%.4g\n", un.ke, dp.ke, dp.ke/(un.ke+1e-12f));
    check(un.ke > 1e-3f, "F undamped cantilever RINGS (residual KE high)", un.ke, 0, 0);
    check(dp.ke < 0.10f*un.ke, "F damping removes ringing energy (KE_damped << KE_undamped)", dp.ke/(un.ke+1e-12f), 0, 0.10);
    // damped run SETTLES to the SAME equilibrium as the heavy-damped golden (damping changes the transient,
    // not the rest shape) -> "slosh/settle matches".
    float maxDiff=0.f;
    for (int i=0;i<n;i++){ float dx=dp.shape[3*i]-eq.shape[3*i], dy=dp.shape[3*i+1]-eq.shape[3*i+1], dz=dp.shape[3*i+2]-eq.shape[3*i+2];
                          maxDiff=fmaxf(maxDiff, sqrtf(dx*dx+dy*dy+dz*dz)); }
    check(maxDiff/beamLen < 0.05f, "G damped settles to the golden equilibrium (<5% len)", maxDiff/beamLen, 0, 0.05);
    // more damping -> faster settle (lower residual KE at fixed time).
    DampResult light = runBeam(n, sp, ks, kb, 0.5f, betaS, betaB, 0.f, 120);
    DampResult heavy = runBeam(n, sp, ks, kb, 6.0f, betaS, betaB, 0.f, 120);
    printf("  info  settle KE: light-alpha=%.5g  heavy-alpha=%.5g\n", light.ke, heavy.ke);
    check(heavy.ke < light.ke, "H more damping -> faster settle (KE monotone in alpha)", heavy.ke, light.ke, 0);

    // cs/cb ISOLATION (review major: F/G/H are driven by the Rayleigh ALPHA term; the Kelvin-Voigt cs/cb
    // term must be exercised too — this is what caught the beta=c/k bug that made cs/cb inert). A cantilever
    // SWING is a near-rigid mode with tiny relative velocities, so cs/cb barely act there (and the paper's
    // cs=0.92/ks=7.5e4 is physically ultra-light, zeta~0.0017). Test the cs MECHANISM directly on a
    // STRETCHED SPRING (pinned + mass, NO gravity, NO alpha) whose oscillation IS along-spring relative
    // velocity, with a large betaS so the damped-constraint dashpot f=betaS*relvel is clearly visible.
    // Softer spring (ksS=1000, slower osc -> XPBD's own high-freq numerical damping stays mild over a few
    // periods) so the cs dashpot is cleanly separable; measure the mass's KE at an EARLY frame.
    // PHASE-INDEPENDENT metric: the PEAK kinetic energy the oscillation reaches over a late window [20,40]
    // frames = the oscillation ENVELOPE there (instantaneous KE at one frame is phase-dependent). More cs
    // damping -> lower envelope -> monotone.
    const float ksS = 1.0e3f;
    auto springEnvKE = [&](float bS)->float{
        float p2[6]={0,0,0, 0,-1.3f,0}; float im[2]={0.f,1.f}; int ed[2]={0,1}; float rl[1]={1.0f};  // rest 1.0, stretched 0.3
        xpbd_init(2,p2,im,1,ed,rl);
        XpbdParams q = { 20,1,1.0f,0.f, ksS, 0.f,0.f,0.f, 0.f, /*alpha*/0.f, bS, 0.f };   // no gravity, no alpha, only cs
        xpbd_set_params(&q);
        float envKE = 0.f; std::vector<float> ov(6);
        for (int f=0; f<40; f++){ xpbd_step(1.f/60.f);
            if (f>=20){ xpbd_get_velocities(ov.data()); envKE = fmaxf(envKE, 0.5f*(ov[3]*ov[3]+ov[4]*ov[4]+ov[5]*ov[5])); } }
        xpbd_shutdown(); return envKE;
    };
    float env0 = springEnvKE(0.f), envBig = springEnvKE(600.f);
    printf("  info  spring(ks=1e3) cs-damp envelope-KE[20,40]: none=%.5g cs-damped=%.5g ratio=%.4g\n",
           env0, envBig, envBig/(env0+1e-12f));
    check(env0 > 1e-3f, "I0 undamped spring RINGS (KE envelope persists)", env0, 0, 0);
    // The Kelvin-Voigt cs damped-constraint term ALONE (alpha=0) removes the along-spring oscillation
    // energy -> the term is ACTIVE (the beta=cs/ks bug that made it inert would leave the envelope ~= none).
    // The bending damper betaB=cb is a DISTINCT code path (k_solve_bending has its own 3-particle angle-rate
    // gradDx + its own denom (1+gamma)*wg+alphaB) and is isolated SEPARATELY in test J below — do NOT assume
    // test I or the geometric test D covers it (D runs with betaB=0; the historical betaS=cs/ks factor-k bug
    // has an exact betaB=cb/kb twin that only a bending-specific isolation would catch).
    check(envBig < 0.05f*env0, "I cs Kelvin-Voigt damped-constraint is ACTIVE (not inert): removes >95% envelope",
          envBig/(env0+1e-12f), 0, 0.05);
}

// ── step 3 (bending damper): J — betaB=cb Kelvin-Voigt ISOLATION (twin of test I for bending). A 3-particle
// hinge (two arms pinned, apex free) is driven off its bending rest angle and released with NO gravity, NO
// alpha, NO structural damping (betaS=0), so the ONLY dissipation channel is the bending damped-constraint
// betaB. betaB=0 -> the apex RINGS (angular KE envelope persists); large betaB -> the envelope collapses.
// A betaB=cb/kb factor-k regression (the bending twin of blocker ws1i08hz6) would leave betaB ~inert and
// FAIL this test. Arms soft (ks=1e3, like test I) so XPBD's own high-freq numerical damping stays mild;
// bending stiff (kb=2e4) so the motion is bending-dominated. Phase-independent metric = peak (envelope) KE
// over the late window [20,40] frames. ──────────────────────────────────────────────────────────────────
static void testBendDampIsolation() {
    const float ksA = 1.0e3f, kbB = 2.0e4f;
    // hinge geometry: j=(0,0,0), k=(2,0,0) pinned; apex i=(1,1,0) free. Arm rest = current |i-j|=|i-k|=sqrt2
    // (structural at rest at t=0 -> bending is the sole initial driver). theta0=120deg while the START angle
    // is 90deg, so bending swings the apex -> a clean angular oscillation, staying clear of the s<1e-4 guard.
    const float aL = sqrtf(2.0f);
    auto bendEnvKE = [&](float bB)->float{
        float p3[9] = { 0,0,0,  2,0,0,  1,1,0 };          // j, k, apex i
        float im[3] = { 0.f, 0.f, 1.f };                  // pin j,k; free apex
        int   ed[4] = { 2,0,  2,1 };                      // arm edges i-j, i-k
        float rl[2] = { aL, aL };
        int   bijk[3] = { 2, 0, 1 };                      // bend element: apex i(2), arms j(0), k(1)
        float th0[1]  = { 2.0943951f };                   // 120 deg rest angle (start is 90 deg)
        xpbd_init(3, p3, im, 2, ed, rl);
        xpbd_set_bending(1, bijk, th0);
        XpbdParams q = { 20,1,1.0f,0.f, ksA, 0.f,0.f,0.f, kbB, /*alpha*/0.f, /*betaS*/0.f, bB };
        xpbd_set_params(&q);
        float envKE=0.f; std::vector<float> ov(9);
        for (int f=0; f<40; f++){ xpbd_step(1.f/60.f);
            if (f>=20){ xpbd_get_velocities(ov.data()); envKE=fmaxf(envKE, 0.5f*(ov[6]*ov[6]+ov[7]*ov[7]+ov[8]*ov[8])); } }
        xpbd_shutdown(); return envKE;
    };
    float benv0 = bendEnvKE(0.f), benvBig = bendEnvKE(600.f);
    printf("  info  hinge bending cb-damp envelope-KE[20,40]: none=%.5g cb-damped=%.5g ratio=%.4g\n",
           benv0, benvBig, benvBig/(benv0+1e-12f));
    check(benv0 > 1e-3f, "J0 undamped hinge RINGS (bending KE envelope persists)", benv0, 0, 0);
    check(benvBig < 0.10f*benv0, "J betaB Kelvin-Voigt bending damper is ACTIVE (not inert): removes >90% envelope",
          benvBig/(benv0+1e-12f), 0, 0.10);
}

// ── step 2 (large strain): K — bending golden-match at a TENT / UNEQUAL-ARM geometry (design §4 [Bending]
// gate: "a LARGE-STRAIN bending test at tent geometry, not just clamped-cantilever droop"). The code's
// un-normalized Eq14 gradient drops 1/L_u,1/L_v so effective bending is arm-length-asymmetry-dependent
// EXACTLY at unequal-arm stretch; verify the XPBD solve still reproduces the RK45 golden there. A single
// bend element with widely-separated pinned bases and SHORT, UNEQUAL arm rest lengths forces both arms to
// STRETCH unequally (a held tent), and the settled apex must match goldenSettle within tolerance. ─────────
static void testLargeStrainBending() {
    const float ks = 7.5e4f, kb = 2.0e4f;
    // j=(0,0,0), k=(3,0,0) pinned (wide base); apex i free. Arm rest lengths per the rl initializer
    // below (UNEQUAL, both far shorter than the geometry demands) -> arms are forced to stretch a lot
    // and unequally. No gravity: a pure elastic (structural+bending) equilibrium held by the pinning.
    float pos[9]  = { 0,0,0,  3,0,0,  1.0f,1.2f,0 };
    float im[3]   = { 0.f, 0.f, 1.f };
    int   ed[4]   = { 2,0,  2,1 };
    float rl[2]   = { 0.8f, 1.35f };   // UNEQUAL + short -> both arms forced well past 30% stretch (a real tent)
    int   bijk[3] = { 2, 0, 1 };
    // theta0 = the apex angle at the START pose (so bending's rest is defined at a real bent geometry).
    V3 vi={pos[6],pos[7],pos[8]}, vj={pos[0],pos[1],pos[2]}, vk={pos[3],pos[4],pos[5]};
    V3 nu0=unit(sub(vi,vj)), nv0=unit(sub(vi,vk));
    float th0v = acosf(fmaxf(-1.f,fminf(1.f,dot(nu0,nv0))));
    float th0[1] = { th0v };

    // golden (physics.cu forces), no gravity
    std::vector<float> imv(im, im+3); std::vector<int> edv(ed, ed+4); std::vector<float> rlv(rl, rl+2);
    std::vector<int> bv(bijk, bijk+3); std::vector<float> tv(th0, th0+1);
    std::vector<V3> xg(3); for (int i=0;i<3;i++) xg[i]={pos[3*i],pos[3*i+1],pos[3*i+2]};
    goldenSettle(3, xg, imv, edv, rlv, bv, tv, ks, kb, {0,0,0}, 400000);

    // XPBD settle (structural + bending), shipped config
    xpbd_init(3, pos, im, 2, ed, rl);
    xpbd_set_bending(1, bijk, th0);
    XpbdParams p = { 20, 1, 1.0f, 0.05f, ks, 0.f, 0.f, 0.f, kb };
    xpbd_set_params(&p);
    std::vector<float> out(9);
    for (int f=0; f<12000; f++) xpbd_step(1.0f/60.0f);
    xpbd_get_positions(out.data());
    xpbd_shutdown();

    V3 xi = { out[6], out[7], out[8] };
    float armJ = nrm(sub(xi, vj)), armK = nrm(sub(xi, vk));         // realized (stretched) arm lengths
    float strainJ = (armJ - rl[0]) / rl[0], strainK = (armK - rl[1]) / rl[1];
    float apexDiff = nrm(sub(xi, xg[2]));
    float ref = nrm(sub(xg[2], vj));                               // golden arm-j length as the scale
    printf("  info  tent-bend: arms realized=%.3f/%.3f rest=%.2f/%.2f strain=%.0f%%/%.0f%% (unequal), apex diff=%.4f (%.2f%% of %.3f)\n",
           armJ, armK, rl[0], rl[1], 100.f*strainJ, 100.f*strainK, apexDiff, 100.f*apexDiff/ref, ref);
    check(strainJ > 0.30f && strainK > 0.30f, "K tent geometry IS large-strain (both arms > 30% stretch)",
          fminf(strainJ, strainK), 0.30, 0);
    check(apexDiff/ref < 0.05f, "K bending golden-match at UNEQUAL-arm tent geometry (<5%)", apexDiff/ref, 0, 0.05);
}

// ── step-1 infra: L — greedy colour-budget overflow must FAIL LOUD (XPBD_ERR_COLOR_OVERFLOW, -1003; a
// banded state error, NOT a negated CUDA code), never silently alias to a used colour (which would race
// the no-atomics GS). Drive a >64-incident-edge star (structural) and a >64-shared-apex bending set;
// both must return XPBD_ERR_COLOR_OVERFLOW — behavioral proof of the
// `if (c >= 64) return XPBD_ERR_COLOR_OVERFLOW;` guard, which no ≤6-degree lattice/beam ever exercises.
static void testColourBudget() {
    // structural: 70 edges all incident to vertex 0 -> 70 distinct colours needed -> COLOR_OVERFLOW.
    const int spokes = 70, n = spokes + 1;
    std::vector<float> pos(3*n, 0.f), invMass(n, 1.f), rest;
    std::vector<int> edges;
    for (int i=0;i<n;i++) pos[3*i] = (float)i;               // distinct positions (no coincident)
    invMass[0] = 0.f;
    for (int s=1;s<=spokes;s++){ edges.push_back(0); edges.push_back(s); rest.push_back((float)s); }
    int rc = xpbd_init(n, pos.data(), invMass.data(), spokes, edges.data(), rest.data());
    check(rc == XPBD_ERR_COLOR_OVERFLOW, "L structural colour-budget >64 FAILS LOUD (COLOR_OVERFLOW)",
          rc, XPBD_ERR_COLOR_OVERFLOW, 0);
    if (rc == 0) xpbd_shutdown();

    // bending: valid low-degree init, then 70 bend elements all sharing apex 0 -> COLOR_OVERFLOW.
    const int bn = 200;
    std::vector<float> bp(3*bn, 0.f), bim(bn, 1.f), br; std::vector<int> be;
    for (int i=0;i<bn;i++) bp[3*i] = (float)i;
    for (int i=0;i<bn-1;i++){ be.push_back(i); be.push_back(i+1); br.push_back(1.f); }   // chain, deg<=2
    if (xpbd_init(bn, bp.data(), bim.data(), bn-1, be.data(), br.data()) != 0) { printf("  FAIL init(bend-budget)\n"); g_fail++; return; }
    const int belems = 70; std::vector<int> bijk; std::vector<float> bth;
    for (int e=0;e<belems;e++){ bijk.push_back(0); bijk.push_back(1+2*e); bijk.push_back(2+2*e); bth.push_back(1.5708f); }
    int rb = xpbd_set_bending(belems, bijk.data(), bth.data());
    check(rb == XPBD_ERR_COLOR_OVERFLOW, "L bending colour-budget >64 FAILS LOUD (COLOR_OVERFLOW)",
          rb, XPBD_ERR_COLOR_OVERFLOW, 0);
    // F9 fail-preserving: the failed set_bending must leave the module fully usable (old set intact,
    // here: no set) — a subsequent step must still succeed, not dispatch into null bend buffers.
    check(xpbd_step(1.0f/60.0f) == 0, "L failed set_bending leaves the module usable (step ok)", 0, 0, 0);
    xpbd_shutdown();

    // C4: fail-preserving over LIVE state (the empty-state checks above cannot catch a reorder that
    // hoists the free/shutdown above the validation).
    // (a) a failed RE-INIT (bad edge index) over a LIVE instance must preserve it — it must still step
    //     and report its ORIGINAL colouring, not be shut down.
    {
        const int n = 12; const float sp = 0.15f;
        std::vector<float> bpos, bim, brest, bth0; std::vector<int> bedges, bbend;
        buildClampedBeam(n, sp, bpos, bim, bedges, brest, bbend, bth0);
        xpbd_init(n, bpos.data(), bim.data(), (int)brest.size(), bedges.data(), brest.data());
        xpbd_set_bending((int)bth0.size(), bbend.data(), bth0.data());
        int coloursBefore = xpbd_color_count(), bendColoursBefore = xpbd_bend_color_count();
        float posBad[6] = { 0,0,0, 0,-1,0 }; float imBad[2] = { 0.f, 1.f };
        int   edBad[2]  = { 0, 99 };                       // vertex 99 out of range for 2 particles
        float rlBad[1]  = { 1.f };
        int rcBad = xpbd_init(2, posBad, imBad, 1, edBad, rlBad);
        check(rcBad == XPBD_ERR_BAD_ARG, "L bad re-init REJECTED (BAD_ARG)", rcBad, XPBD_ERR_BAD_ARG, 0);
        check(xpbd_color_count() == coloursBefore && xpbd_bend_color_count() == bendColoursBefore,
              "L bad re-init PRESERVES the live instance (colourings intact)", xpbd_color_count(), coloursBefore, 0);
        check(xpbd_step(1.0f/60.0f) == 0, "L live instance still steps after rejected re-init", 0, 0, 0);
        // (b) a colour-overflow set_bending over an INSTALLED bend set must preserve the OLD set.
        std::vector<int> badB; std::vector<float> badT;
        for (int e=0; e<70; e++){ badB.push_back(0); badB.push_back(1); badB.push_back(2); badT.push_back(1.f); }
        int rcOv = xpbd_set_bending(70, badB.data(), badT.data());
        check(rcOv == XPBD_ERR_COLOR_OVERFLOW, "L overflow set_bending over a LIVE set REJECTED",
              rcOv, XPBD_ERR_COLOR_OVERFLOW, 0);
        check(xpbd_bend_color_count() == bendColoursBefore,
              "L overflow set_bending PRESERVES the old bend set", xpbd_bend_color_count(), bendColoursBefore, 0);
        // (c) a NEGATIVE bendCount is a caller bug: FAIL LOUD, do not silently wipe the set (C1).
        int rcNeg = xpbd_set_bending(-5, bbend.data(), bth0.data());
        check(rcNeg == XPBD_ERR_BAD_ARG, "L negative bendCount REJECTED (no silent wipe)",
              rcNeg, XPBD_ERR_BAD_ARG, 0);
        check(xpbd_bend_color_count() == bendColoursBefore,
              "L negative bendCount PRESERVES the bend set", xpbd_bend_color_count(), bendColoursBefore, 0);
        check(xpbd_step(1.0f/60.0f) == 0, "L module fully usable after all rejected calls", 0, 0, 0);
        xpbd_shutdown();
    }
}

// ── step-1 math: M — omega-INVARIANCE on the ACTUAL kernel. Design §3.5's load-bearing "reviewed
// xpbd-math fix": omega is applied CONSISTENTLY to BOTH lambda (`lambda[e] += applied`) AND position
// (`pos[i] + n * (wi * applied)`), so the
// compliant equilibrium DL=mg/ks is omega-INVARIANT (omega changes only the convergence RATE). Every OTHER
// device config runs omega=1.0, at which applied=1.0*dLambda so a "scale position but NOT lambda" kernel
// regression is byte-identical to correct — invisible. Only omega<1 exposes it; the Python oracle checks
// omega!=1 on its OWN CPU model, not the shipped kernel. Settle the single spring at omega in {1,0.5,0.2}
// and assert all three reach DL=mg/ks. ─────────────────────────────────────────────────────────────────
static void testOmegaInvariance() {
    const float L0=1.0f, m=1.0f, g=9.81f, ks=7.5e4f;
    float invMass[2] = { 0.f, 1.f/m };
    int   edges[2]   = { 0, 1 };
    float rest[1]    = { L0 };
    const float want = m*g/ks;
    const float oms[3] = { 1.0f, 0.5f, 0.2f };
    float DLs[3];
    for (int t=0; t<3; t++) {
        float pos[6] = { 0,0,0, 0,-L0,0 };
        if (xpbd_init(2, pos, invMass, 1, edges, rest) != 0) { printf("  FAIL init(omega)\n"); g_fail++; return; }
        XpbdParams p = { /*N*/20, /*iters*/40, oms[t], /*damping*/0.05f, ks, 0.f, -g, 0.f };
        xpbd_set_params(&p);
        std::vector<float> out(6);
        settle(1.0f/60.0f, 6000, out);
        DLs[t] = (0.f - out[4]) - L0;
        xpbd_shutdown();
    }
    printf("  info  omega-invariance DL: omega=1.0->%.6g 0.5->%.6g 0.2->%.6g (want mg/ks=%.6g)\n",
           DLs[0], DLs[1], DLs[2], want);
    for (int t=0; t<3; t++) {
        char nm[72]; snprintf(nm, sizeof nm, "M omega-invariant DL=mg/ks on the kernel (omega=%.1f)", oms[t]);
        check(fabs(DLs[t]-want) <= 0.06*want + 1e-9, nm, DLs[t], want, 0.06*want);
    }
}

// ── M2 (design §9-(4)): N — loop-shape lock for the split API. xpbd_step IS begin_frame + N x substep
// by construction, so what this check adds is: (a) FOREIGN WORK interleaved between substeps (sync
// readbacks + no-op notify writes — standing in for cut_ribbon_tick's device work) must NOT perturb the
// state: BITWISE-identical results (maxDiff == 0); (b) parity holds mid-transient (frame 10), not just
// at the settled end. (Refactor physics fidelity is carried by checks A..M + the oracle, not N alone.)
static void testSubstepApiParity() {
    const int n = 12; const float sp = 0.15f, ks = 7.5e4f, kb = 2.0e4f;
    std::vector<float> pos, invMass, rest, th0; std::vector<int> edges, bend;
    buildClampedBeam(n, sp, pos, invMass, edges, rest, bend, th0);
    XpbdParams p = { 20, 1, 1.0f, 0.05f, ks, 0.f, -9.81f, 0.f, kb, 2.f, 0.92f, 0.9f };
    // run A: monolithic wrapper (snapshot mid-transient at frame 10 AND settled at frame 200)
    std::vector<float> outA(3*n), outB(3*n), midA(3*n), midB(3*n), scratch(3*n);
    xpbd_init(n, pos.data(), invMass.data(), (int)rest.size(), edges.data(), rest.data());
    xpbd_set_bending((int)th0.size(), bend.data(), th0.data());
    xpbd_set_params(&p);
    for (int f=0; f<10;  f++) xpbd_step(1.0f/60.0f);
    xpbd_get_positions(midA.data());
    for (int f=10; f<200; f++) xpbd_step(1.0f/60.0f);
    xpbd_get_positions(outA.data());
    xpbd_shutdown();
    // run B: begin_frame + N x substep with FOREIGN WORK between substeps (the live loop shape)
    std::vector<float> pos2 = pos;
    xpbd_init(n, pos2.data(), invMass.data(), (int)rest.size(), edges.data(), rest.data());
    xpbd_set_bending((int)th0.size(), bend.data(), th0.data());
    xpbd_set_params(&p);
    for (int f=0; f<200; f++) {
        if (xpbd_begin_frame(1.0f/60.0f) != 0) { printf("  FAIL begin_frame\n"); g_fail++; xpbd_shutdown(); return; }
        for (int sub=0; sub<20; sub++) {
            if (xpbd_substep() != 0) { printf("  FAIL substep\n"); g_fail++; xpbd_shutdown(); return; }
            if ((sub & 3) == 0) {
                xpbd_get_positions(scratch.data());     // sync readback between substeps (foreign work)
                xpbd_set_edge_active(0, 1);             // no-op notify write (1 over 1) between substeps
            }
        }
        if (f == 9)  xpbd_get_positions(midB.data());
    }
    xpbd_get_positions(outB.data());
    xpbd_shutdown();
    float maxDiff = 0.f, midDiff = 0.f;
    for (int k=0; k<3*n; k++) {
        maxDiff = fmaxf(maxDiff, fabsf(outA[k]-outB[k]));
        midDiff = fmaxf(midDiff, fabsf(midA[k]-midB[k]));
    }
    check(midDiff == 0.f, "N split API + interleaved work: BITWISE parity mid-transient (frame 10)", midDiff, 0, 0);
    check(maxDiff == 0.f, "N split API + interleaved work: BITWISE parity settled (frame 200)", maxDiff, 0, 0);
}

// ── M2 contract (review F6a/F2): the frame-invalidation state machine. set_params/set_bending invalidate
// an OPEN frame (substep must refuse until a fresh begin_frame); a FAILED begin_frame must not leave the
// PREVIOUS frame armed; NaN/Inf dt is rejected. ────────────────────────────────────────────────────────
static void testFrameContract() {
    const float L0 = 1.0f;
    float pos[6]     = { 0,0,0,  0,-L0,0 };
    float invMass[2] = { 0.f, 1.f };
    int   edges[2]   = { 0, 1 };
    float rest[1]    = { L0 };
    xpbd_init(2, pos, invMass, 1, edges, rest);
    XpbdParams p = { 20, 1, 1.0f, 0.05f, 1.0e4f, 0.f, -9.81f, 0.f };
    xpbd_set_params(&p);
    check(xpbd_begin_frame(1.0f/60.0f) == 0, "P begin_frame ok", 0, 0, 0);
    check(xpbd_substep() == 0, "P substep ok inside a frame", 0, 0, 0);
    xpbd_set_params(&p);                                          // invalidates the open frame
    check(xpbd_substep() == XPBD_ERR_NO_FRAME, "P set_params INVALIDATES the open frame (substep refuses)",
          xpbd_substep(), XPBD_ERR_NO_FRAME, 0);
    check(xpbd_begin_frame(1.0f/60.0f) == 0 && xpbd_substep() == 0, "P fresh begin_frame re-arms", 0, 0, 0);
    check(xpbd_begin_frame(0.f) == XPBD_ERR_BAD_DT, "P dt=0 rejected (BAD_DT)", xpbd_begin_frame(0.f), XPBD_ERR_BAD_DT, 0);
    check(xpbd_substep() == XPBD_ERR_NO_FRAME, "P FAILED begin_frame does NOT leave the old frame armed",
          xpbd_substep(), XPBD_ERR_NO_FRAME, 0);
    const float qnan = nanf("");
    check(xpbd_begin_frame(qnan) == XPBD_ERR_BAD_DT, "P dt=NaN rejected (BAD_DT)", xpbd_begin_frame(qnan), XPBD_ERR_BAD_DT, 0);
    // dt=+Inf reaches ONLY the !isfinite clause (Inf>0 is true — the !(h>0) guard alone would ARM a
    // frame whose gamma = 0*Inf = NaN poisons every position). Behavioral proof of the isfinite half.
    const float pinf = INFINITY;
    check(xpbd_begin_frame(pinf) == XPBD_ERR_BAD_DT, "P dt=+Inf rejected (BAD_DT, isfinite clause)",
          xpbd_begin_frame(pinf), XPBD_ERR_BAD_DT, 0);
    check(xpbd_substep() == XPBD_ERR_NO_FRAME, "P rejected Inf leaves no armed frame",
          xpbd_substep(), XPBD_ERR_NO_FRAME, 0);
    check(xpbd_begin_frame(1.0f/60.0f) == 0 && xpbd_substep() == 0, "P recovers after rejected dt", 0, 0, 0);
    xpbd_shutdown();
}

// ── §12 R-compliance (review F7): ALL h-derived coefficients must be recomputed when N (hence h)
// changes. The XPBD equilibrium DL = mg/ks holds ONLY when alphaTilde uses the SAME h the substeps run
// at; a stale-h alphaTilde at N 20->40 makes the spring effectively 4x stiffer -> DL/4 -> FAILS. ───────
static void testHRederive() {
    const float L0 = 1.0f, m = 1.0f, g = 9.81f, ks = 7.5e4f;
    float pos[6]     = { 0,0,0,  0,-L0,0 };
    float invMass[2] = { 0.f, 1.f/m };
    int   edges[2]   = { 0, 1 };
    float rest[1]    = { L0 };
    const float want = m*g/ks;
    xpbd_init(2, pos, invMass, 1, edges, rest);
    XpbdParams p20 = { 20, 40, 1.0f, 0.05f, ks, 0.f, -g, 0.f };
    xpbd_set_params(&p20);
    std::vector<float> out(6);
    settle(1.0f/60.0f, 4000, out);
    float DL20 = (0.f - out[4]) - L0;
    XpbdParams p40 = { 40, 40, 1.0f, 0.05f, ks, 0.f, -g, 0.f };   // h halves -> alphaTilde x4
    xpbd_set_params(&p40);
    settle(1.0f/60.0f, 4000, out);
    float DL40 = (0.f - out[4]) - L0;
    printf("  info  h-rederive: DL @N=20 %.6g  @N=40 %.6g  (want mg/ks=%.6g)\n", DL20, DL40, want);
    check(fabs(DL20 - want) <= 0.06*want + 1e-9, "Q DL=mg/ks at N=20", DL20, want, 0.06*want);
    check(fabs(DL40 - want) <= 0.06*want + 1e-9, "Q DL=mg/ks after N 20->40 (h-coefficients re-derived)",
          DL40, want, 0.06*want);
    xpbd_shutdown();
}

// ── M1 (design §9-(4)): O — the persisted permutation + notify path actually DROPS constraints and
// FREEZES mass at runtime (the mechanism a live cut/debris-freeze needs after the rebind).
// PERMUTATION-DISCRIMINATING (review F1): the geometry is chosen so the input->colour-slot map is NOT
// the identity — a regression that ignores g_edgeSlotOfInput (raw-index write) severs the WRONG edge
// and fails loudly here, instead of passing by coincidence. ────────────────────────────────────────────
static void testLivenessNotify() {
    // (a) SEVER, permuted map: pinned 4-chain; edges (0,1),(1,2),(2,3) greedy-colour to [c0,c1,c0], so
    // slotOfInput = [0,2,1] — input edge 1 lives at colour-slot 2, input edge 2 at slot 1. Severing
    // INPUT edge 1 (p1-p2) must drop the (1,2) constraint: p2 AND p3 free-fall, p1 holds. A raw-index
    // bug writes slot 1 (= input edge 2 = (2,3)): then p2 stays HELD (its check fails) and only p3 falls.
    const float L0 = 1.0f, g = 9.81f;
    float pos[12]    = { 0,0,0,  0,-L0,0,  0,-2*L0,0,  0,-3*L0,0 };
    float invMass[4] = { 0.f, 1.f, 1.f, 1.f };
    int   edges[6]   = { 0,1,  1,2,  2,3 };
    float rest[3]    = { L0, L0, L0 };
    xpbd_init(4, pos, invMass, 3, edges, rest);
    // light drag (damping=0.01/substep) — heavier per-substep velScale drag imposes a low terminal
    // velocity that would mask the free-fall this check asserts (physics right, threshold wrong).
    XpbdParams p = { 20, 1, 1.0f, 0.01f, 1.0e4f, 0.f, -g, 0.f };
    xpbd_set_params(&p);
    std::vector<float> out(12);
    settle(1.0f/60.0f, 900, out);                       // settle the intact chain
    float y1s = out[4], y2s = out[7], y3s = out[10];
    // MID-FRAME sever (review F6b): notify BETWEEN substeps of an OPEN frame — the live cut's timing.
    if (xpbd_begin_frame(1.0f/60.0f) != 0) { printf("  FAIL begin_frame\n"); g_fail++; xpbd_shutdown(); return; }
    for (int s=0; s<10; s++) if (xpbd_substep() != 0) { printf("  FAIL substep\n"); g_fail++; xpbd_shutdown(); return; }
    if (xpbd_set_edge_active(1, 0) != 0) { printf("  FAIL set_edge_active\n"); g_fail++; xpbd_shutdown(); return; }
    for (int s=0; s<10; s++)
        if (xpbd_substep() != 0) { printf("  FAIL substep post-notify (frame wrongly invalidated?)\n"); g_fail++; xpbd_shutdown(); return; }
    settle(1.0f/60.0f, 60, out);                        // ~1 s after the sever
    float y1 = out[4], y2 = out[7], y3 = out[10];
    printf("  info  sever(permuted slot): p1 %.4f->%.4f (held)  p2 %.4f->%.4f  p3 %.4f->%.4f (both fall)\n",
           y1s, y1, y2s, y2, y3s, y3);
    check(fabsf(y1 - y1s) < 0.05f, "O sever: surviving edge HOLDS p1", y1, y1s, 0.05);
    check(y2 < y2s - 0.3f, "O sever: p2 FREE-FALLS (permuted slot hit the RIGHT edge)", y2, y2s, 0);
    check(y3 < y3s - 0.3f, "O sever: p3 falls with p2 (still chained below the cut)", y3, y3s, 0);
    // (b) FREEZE: zero p3's inv mass mid-fall -> it stops dead (debris-freeze mechanism).
    if (xpbd_set_particle_inv_mass(3, 0.f) != 0) { printf("  FAIL set_inv_mass\n"); g_fail++; xpbd_shutdown(); return; }
    xpbd_get_positions(out.data()); float y3f = out[10];
    settle(1.0f/60.0f, 30, out);
    check(fabsf(out[10] - y3f) < 1e-5f, "O freeze: invMass=0 stops the debris", out[10], y3f, 1e-5);
    xpbd_shutdown();

    // (c) BEND-SEVER, permuted map + LOCALIZED: on the 12-node beam the bend colouring interleaves
    // (colours cycle 0,1,2), so slotOfInput = [0,4,7,1,5,8,2,6,9,3] — input element 1 (apex vertex 2)
    // lives at colour-slot 4; a raw-index bug would instead kill slot 1 = input element 3 (apex vertex
    // 4). Deactivate ONLY input element 1 and assert the kink LOCALIZES at vertex 2, not vertex 4:
    // the freed angle theta(v2) sags while the still-constrained theta(v4) barely moves.
    {
        const int n = 12; const float sp = 0.15f;
        std::vector<float> bpos, bim, brest, bth0; std::vector<int> bedges, bbend;
        buildClampedBeam(n, sp, bpos, bim, bedges, brest, bbend, bth0);
        auto kinkAt = [&](const std::vector<float>& o, int v)->float{
            V3 xm={o[3*v],o[3*v+1],o[3*v+2]}, xa={o[3*(v-1)],o[3*(v-1)+1],o[3*(v-1)+2]},
               xb={o[3*(v+1)],o[3*(v+1)+1],o[3*(v+1)+2]};
            V3 u=unit(sub(xm,xa)), w=unit(sub(xm,xb));
            return fabsf(3.14159265f - acosf(fmaxf(-1.f,fminf(1.f,dot(u,w)))));
        };
        XpbdParams q = { 20, 5, 1.0f, 0.05f, 7.5e4f, 0.f, -g, 0.f, 5.0e4f };
        std::vector<float> bout(3*n);
        xpbd_init(n, bpos.data(), bim.data(), (int)brest.size(), bedges.data(), brest.data());
        xpbd_set_bending((int)bth0.size(), bbend.data(), bth0.data());
        xpbd_set_params(&q);
        for (int f=0; f<8000; f++) xpbd_step(1.0f/60.0f);
        xpbd_get_positions(bout.data());
        float k2b = kinkAt(bout, 2), k4b = kinkAt(bout, 4);       // all-alive baseline kinks
        if (xpbd_set_bend_active(1, 0) != 0) { printf("  FAIL set_bend_active\n"); g_fail++; xpbd_shutdown(); return; }
        for (int f=0; f<8000; f++) xpbd_step(1.0f/60.0f);
        xpbd_get_positions(bout.data());
        float k2a = kinkAt(bout, 2), k4a = kinkAt(bout, 4);
        float d2 = k2a - k2b, d4 = k4a - k4b;
        printf("  info  bend-sever(elem 1 -> slot 4): kink(v2) %.5f->%.5f (d=%.5f)  kink(v4) %.5f->%.5f (d=%.5f)\n",
               k2b, k2a, d2, k4b, k4a, d4);
        check(d2 > 5e-3f, "O bend-sever: freed angle at v2 SAGS (element dropped)", d2, 0, 0);
        check(d2 > 3.f*fmaxf(d4, 1e-6f), "O bend-sever: kink LOCALIZES at v2 not v4 (permuted slot correct)",
              d2, d4, 0);
        xpbd_shutdown();
    }

    // (c) BEND-SEVER: deactivate ALL bend elements on a stiff-kb clamped beam -> sags toward no-bend.
    const int n = 12; const float sp = 0.15f;
    std::vector<float> bpos, bim, brest, bth0; std::vector<int> bedges, bbend;
    buildClampedBeam(n, sp, bpos, bim, bedges, brest, bbend, bth0);
    xpbd_init(n, bpos.data(), bim.data(), (int)brest.size(), bedges.data(), brest.data());
    xpbd_set_bending((int)bth0.size(), bbend.data(), bth0.data());
    XpbdParams q = { 20, 5, 1.0f, 0.05f, 7.5e4f, 0.f, -g, 0.f, 5.0e4f };
    xpbd_set_params(&q);
    std::vector<float> bout(3*n);
    for (int f=0; f<8000; f++) xpbd_step(1.0f/60.0f);
    xpbd_get_positions(bout.data());
    float sagBefore = fabsf(bout[3*(n-1)+1]);
    for (int e=0; e<(int)bth0.size(); e++)
        if (xpbd_set_bend_active(e, 0) != 0) { printf("  FAIL set_bend_active\n"); g_fail++; xpbd_shutdown(); return; }
    for (int f=0; f<8000; f++) xpbd_step(1.0f/60.0f);
    xpbd_get_positions(bout.data());
    float sagAfter = fabsf(bout[3*(n-1)+1]);
    printf("  info  bend-sever: sag stiff-kb=%.4f -> bends-dropped=%.4f\n", sagBefore, sagAfter);
    check(sagAfter > 1.2f*sagBefore, "O bend-sever: dropping bend elements releases the beam", sagAfter, sagBefore, 0);
    xpbd_shutdown();
}

int main() {
    printf("== XPBD step-1+2+3 self-test ==\n");
    testSingleSpring(1.0e3f);
    testSingleSpring(7.5e4f);
    testChain();
    testStability();
    testOmegaInvariance();
    testBending();
    testLargeStrainBending();
    testDamping();
    testBendDampIsolation();
    testColourBudget();
    testSubstepApiParity();
    testFrameContract();
    testHRederive();
    testLivenessNotify();
    printf("== %s (%d failure%s) ==\n", g_fail==0?"ALL PASS":"FAILURES", g_fail, g_fail==1?"":"s");
    return g_fail;
}
