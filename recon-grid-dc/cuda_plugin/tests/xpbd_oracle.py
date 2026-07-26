#!/usr/bin/env python3
# xpbd_oracle.py  - CPU reference + source gates for the XPBD solver, migration STEP 1.
# CPU oracle for XPBD: a small CPU model that pins the load-bearing MATH, plus
# source-text gates that pin the design invariants in cuda/xpbd/. GPU-free (runs in CI); the real
# numerical proof is the CUDA xpbd_selftest executable (analytic elongation + ks=7.5e4 stability).
#
# Round-3 design: docs/superpowers/specs/2026-07-05-xpbd-combination-round3-design.md

from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent          # cuda_plugin/
XPBD = ROOT / "cuda" / "xpbd"

def test_compliance_reproduces_spring():
    # XPBD (Macklin 2016): a distance constraint with compliance a~=(1/ks)/h^2 reproduces a Hookean
    # spring of stiffness ks EXACTLY. A single mass m hanging from a pinned point under gravity g must
    # settle to elongation DL = m*g/ks. CPU-model the exact substep loop the CUDA kernels run.
    def sim(ks, m, g, h, substeps, frames, iters=40, omega=1.0, damping=0.15, L0=1.0):
        y = -L0; vy = 0.0; w = 1.0/m
        for _ in range(frames):
            for _s in range(substeps):
                yprev = y
                vy += g*h; y += vy*h                       # predict (gravity down: g negative)
                lam = 0.0
                aT = (1.0/ks)/(h*h)
                for _i in range(iters):                    # solve the single edge (top pinned at 0)
                    L = abs(0.0 - y)                        # |x_top - x_bot|
                    if L < 1e-12: break
                    n = (0.0 - y)/L                         # unit dir from bottom to top (+1)
                    C = L - L0
                    dl = (-C - aT*lam)/(w + aT)             # wtop=0 (pinned), wbot=w
                    applied = omega*dl                      # omega applied CONSISTENTLY to lambda AND position
                    lam += applied
                    y -= n*(w*applied)                      # x_bot += w*applied*grad_bot; grad_bot=-n
                vy = (y - yprev)/h * (1.0 - damping)
        return -y - L0                                     # elongation beyond rest
    for ks in (1.0e3, 7.5e4):
        DL = sim(ks, m=1.0, g=-9.81, h=(1/60)/10, substeps=10, frames=3000)
        want = 1.0*9.81/ks
        assert abs(DL - want) <= 0.05*want, f"ks={ks}: DL={DL:.6g} vs mg/ks={want:.6g}"
    # omega-INVARIANCE: with omega applied consistently to lambda+position, the equilibrium DL is unchanged
    # (only convergence rate differs). This is the review xpbd-math fix  - omega!=1 must NOT soften the spring.
    for om in (1.0, 0.5, 0.2):
        DL = sim(7.5e4, m=1.0, g=-9.81, h=(1/60)/10, substeps=10, frames=6000, omega=om)
        want = 9.81/7.5e4
        assert abs(DL - want) <= 0.05*want, f"omega={om}: DL={DL:.6g} vs mg/ks={want:.6g} (omega changed the equilibrium!)"

def test_damped_spring_dissipates():
    # CPU model of the DAMPED single spring  - the ONE piece of load-bearing math the GPU-free oracle
    # otherwise pins only by source-substring match (the Macklin damped denominator (1+gamma)*wsum + a~
    # and the gradDx sign). Mirror k_solve_edges' gamma/gradDx EXACTLY (i=top pinned, j=bottom mass):
    #   a~=(1/ks)/h^2 ; gamma=a~*betaS*h ; d=x_top-x_bot ; n=d/|d| ; C=|d|-L0 ;
    #   gradDx = n . ((x_top-xprev_top) - (x_bot-xprev_bot)) ;
    #   dl = (-C - a~*lam - gamma*gradDx)/((1+gamma)*w + a~) ; x_bot -= n*(w*dl).
    # Verify (a) betaS=0 (gamma=0) RINGS  - the along-spring oscillation envelope persists; (b) a large
    # betaS COLLAPSES it (dissipative). This test numerically catches the two mutations that flip the
    # envelope: a flipped gradDx SIGN (injects energy -> envBig > env0) and an INERT beta (gamma=0 ->
    # ratio ~ 1.0). It does NOT independently catch a missing (1+gamma) DENOMINATOR factor (at these
    # params the along-spring velocity is over-damped either way, so dropping (1+gamma) barely moves the
    # late-window peak)  - that factor is pinned instead by the exact-string asserts in
    # test_source_invariants ("(1.f + gamma) * wsum + alphaTilde" / "... * wg + alphaB").
    def envKE(ks, betaS, m=1.0, h=(1/60)/20, substeps=20, frames=40, iters=8, L0=1.0, y0=-1.3):
        y = y0; vy = 0.0; w = 1.0/m
        aT = (1.0/ks)/(h*h); gamma = aT*betaS*h
        peak = 0.0
        for f in range(frames):
            for _s in range(substeps):
                yprev = y
                y += vy*h                                    # predict: no gravity, no Rayleigh
                lam = 0.0
                for _i in range(iters):
                    L = abs(0.0 - y)                          # |x_top - x_bot| = |y|
                    if L < 1e-12: break
                    n = (0.0 - y)/L                           # d = x_top - x_bot = -y ; n = d/|d|
                    C = L - L0
                    gradDx = n * ((0.0 - 0.0) - (y - yprev))  # top pinned -> only the bottom moved this substep
                    dl = (-C - aT*lam - gamma*gradDx) / ((1.0 + gamma)*w + aT)
                    lam += dl
                    y = y - n*(w*dl)                          # x_bot -= n*(w*applied)  (omega=1)
                vy = (y - yprev)/h
            if f >= 20: peak = max(peak, 0.5*m*vy*vy)          # envelope KE over the late window [20,40]
        return peak
    env0   = envKE(1.0e3, 0.0)      # undamped: rings
    envBig = envKE(1.0e3, 600.0)    # strong Kelvin-Voigt dashpot: collapses
    assert env0 > 1e-3, f"undamped spring must ring (env0={env0:.4g})"
    assert envBig < 0.10*env0, f"betaS dashpot must dissipate (envBig={envBig:.4g} vs env0={env0:.4g}); " \
                               f"a flipped gradDx sign would INJECT energy (envBig>env0)"

def test_greedy_colouring_valid():
    # The greedy edge colouring must guarantee no two edges in a colour share a vertex (so a colour is
    # a conflict-free Gauss-Seidel dispatch with NO atomics). Model the exact host algorithm and verify.
    def colour(nEdges, edges):
        vmask = {}; ec = []
        for e in range(nEdges):
            i, j = edges[2*e], edges[2*e+1]
            used = vmask.get(i, 0) | vmask.get(j, 0)
            c = 0
            while used & (1 << c): c += 1
            ec.append(c); vmask[i] = vmask.get(i,0)|(1<<c); vmask[j] = vmask.get(j,0)|(1<<c)
        return ec
    # a 4x4x4 6-neighbour lattice: colouring must be VALID and use few colours (<=7, expect 6).
    D = 4; idx = lambda x,y,z: (x*D+y)*D+z
    edges = []
    for x in range(D):
        for y in range(D):
            for z in range(D):
                i = idx(x,y,z)
                if x+1<D: edges += [i, idx(x+1,y,z)]
                if y+1<D: edges += [i, idx(x,y+1,z)]
                if z+1<D: edges += [i, idx(x,y,z+1)]
    nE = len(edges)//2
    ec = colour(nE, edges)
    # validity: within any colour, all vertices are distinct
    from collections import defaultdict
    byc = defaultdict(list)
    for e in range(nE): byc[ec[e]].append((edges[2*e], edges[2*e+1]))
    for c, es in byc.items():
        verts = [v for pair in es for v in pair]
        assert len(verts) == len(set(verts)), f"colour {c} has a shared vertex (GS race)"
    assert max(ec)+1 <= 7, f"too many colours: {max(ec)+1}"

def test_bending_greedy_colouring_valid():
    # The BENDING greedy colouring is a DISTINCT 3-vertex algorithm (anchor in xpbd_solver.cu:
    # `used = vmask[i] | vmask[j] | vmask[k]`); a colour must not contain two elements sharing ANY of the three
    # vertices, else the no-atomics GS races (a lost update that can hide inside the shape tolerance). The
    # structural test above only models 2-endpoint edges, so model the 3-vertex version and verify it.
    def colour3(nElem, elems):                       # elems = flat [i,j,k, i,j,k, ...]
        vmask = {}; ec = []
        for e in range(nElem):
            i, j, k = elems[3*e], elems[3*e+1], elems[3*e+2]
            used = vmask.get(i,0) | vmask.get(j,0) | vmask.get(k,0)
            c = 0
            while c < 64 and (used & (1 << c)): c += 1
            assert c < 64, "bending colour budget exhausted (>64)"
            ec.append(c)
            for v in (i, j, k): vmask[v] = vmask.get(v,0) | (1 << c)
        return ec
    # buildClampedBeam's bend topology: for m in [1, n-2], triple (m, m-1, m+1). Consecutive triples share
    # TWO vertices, so the colouring is genuinely exercised (a naive same-colour placement would race).
    n = 12
    elems = []
    for m in range(1, n-1): elems += [m, m-1, m+1]
    nE = len(elems)//3
    ec = colour3(nE, elems)
    from collections import defaultdict
    byc = defaultdict(list)
    for e in range(nE): byc[ec[e]].append((elems[3*e], elems[3*e+1], elems[3*e+2]))
    for c, es in byc.items():                        # within a colour, ALL vertices distinct
        verts = [v for tri in es for v in tri]
        assert len(verts) == len(set(verts)), f"bend colour {c} has a shared vertex (GS race)"
    # this overlapping-triple chain needs exactly 3 colours  - the value the device reports
    # (selftest 'beam bend-colours=3'), so the oracle model and the shipped colouring agree.
    assert max(ec)+1 == 3, f"expected 3 bend colours for the clamped-beam topology, got {max(ec)+1}"

def test_source_invariants():
    cu  = (XPBD / "xpbd_solver.cu").read_text(encoding="utf-8", errors="ignore")
    hdr = (XPBD / "xpbd_solver.h").read_text(encoding="utf-8", errors="ignore")
    # compliance a~ = (1/ks)/h^2   - the exact-spring mapping
    assert "1.f / (g_p.ks * h * h)" in cu, "compliance a~=(1/ks)/h^2 not present"
    # XPBD multiplier formula with the lambda term + the step-3 Macklin damping (gamma). Reduces to the
    # undamped (-C - a~*lambda)/(wsum + a~) when gamma=0 (betaS=0).
    assert "(-C - alphaTilde * lambda[e] - gamma * gradDx) / ((1.f + gamma) * wsum + alphaTilde)" in cu, \
        "XPBD damped dLambda formula not present"
    # omega applied CONSISTENTLY to lambda AND position (review xpbd-math fix: omega must not soften the spring)
    assert "float  applied = omega * dLambda" in cu and "lambda[e] += applied" in cu \
        and "wi * applied" in cu and "wj * applied" in cu, "omega must scale lambda AND position consistently"
    # per-edge active/severed gate present now (step-5 cut = a flag flip, colouring stays static)
    assert "if (edgeActive[e] == 0) return;" in cu, "per-edge active gate missing (needed for step-5 cut)"
    # STEP 2 bending: paper Eq13-16 (k_AccBending) as an XPBD angle constraint. Must use the Eq14
    # force-DIRECTION gradients (nv-c*nu),(nu-c*nv) WITHOUT the 1/(sin*|arm|) normalization, else the
    # bending is ~10x too stiff and does NOT match the RK45 golden (verified: 0.4% vs 50% cantilever droop).
    assert "k_solve_bending" in cu and "k_solve_bending<<<" in cu, "bending kernel / per-colour dispatch missing"
    assert "float3 gj = (nv - nu*c);" in cu and "float3 gk = (nu - nv*c);" in cu, \
        "bending must use Eq14 force-direction gradients (no 1/sin normalization) to match the RK45 golden"
    assert "1.f / (g_p.kb * h * h)" in cu, "bending compliance a~=(1/kb)/h^2 missing"
    assert "vmask[i] | vmask[j] | vmask[k]" in cu, "bending greedy colouring over its 3 vertices missing"
    assert "xpbd_set_bending" in hdr, "bending C API missing"
    # STEP 3 damping: alpha (Rayleigh) as the per-substep implicit velocity term v *= 1/(1+alpha*h);
    # cs/cb (Kelvin-Voigt) as the Macklin XPBD damped-constraint term gamma = a~*beta*h with the
    # grad C . (x - x_prev) relative-velocity term. Both solve kernels carry it.
    assert "(1.f - g_p.damping) / (1.f + g_p.alpha * h)" in cu, "Rayleigh alpha velocity term missing"
    assert "alphaTilde * g_p.betaS * h" in cu and "alphaB * g_p.betaB * h" in cu, "gamma = a~*beta*h missing"
    assert "- gamma * gradDx" in cu, "Macklin XPBD damping term (grad C . (x-x_prev)) missing"
    assert "(1.f + gamma) * wsum + alphaTilde" in cu, "structural damped denom (1+gamma) missing"
    assert "(1.f + gamma) * wg + alphaB" in cu, "bending damped denom (1+gamma) missing"
    assert "xpbd_get_velocities" in hdr, "velocity readback (damping/settle verify) missing"
    # >=64-colour budget exhaustion FAILS LOUDLY (no silent aliasing race), in the DISJOINT error band
    # (state errors must not alias negated CUDA codes  - review F4)
    assert cu.count("if (c >= 64) return XPBD_ERR_COLOR_OVERFLOW;") >= 2, \
        "colour-budget overflow must fail loud (banded code) in the edge and bend colourings"
    assert "#define XPBD_ERR_COLOR_OVERFLOW  (-1003)" in hdr, "banded error codes missing from header"
    # M2 frame contract: set_params invalidates the open frame; substep refuses without a valid frame;
    # a FAILED begin_frame clears the previous frame and rejects non-finite dt (review F2/F6)
    assert "g_p = *p; g_frameReady = false;" in cu, "set_params must invalidate the open frame"
    assert "if (!g_frameReady) return XPBD_ERR_NO_FRAME;" in cu, "substep must refuse without a frame"
    assert "g_frameReady = false;                               // a FAILED begin must not leave the OLD frame armed" in cu, \
        "begin_frame must clear the old frame BEFORE validation"
    assert "!std::isfinite(h)" in cu, "begin_frame must reject NaN/Inf dt"
    assert "int  xpbd_begin_frame(float dt);" in hdr and "int  xpbd_substep(void);" in hdr, \
        "M2 split API missing from header"
    assert "xpbd_set_edge_active" in hdr and "xpbd_set_bend_active" in hdr \
        and "xpbd_set_particle_inv_mass" in hdr, "M1 notify API missing from header"
    # small-steps substep loop + predict/solve/update kernels
    assert "for (int sub = 0; sub < N; sub++)" in cu, "substep loop missing"
    assert "prevPos[p] = pos[p]" in cu and "pos[p] = pos[p] + v * h" in cu, "predict step missing"
    assert "vel[p] = (pos[p] - prevPos[p])" in cu, "velocity update v=(x-x_prev)/h missing"
    # graph-coloured Gauss-Seidel: one dispatch per colour, no atomics
    assert "k_solve_edges<<<" in cu and "g_colorOff[c]" in cu and "g_colorCnt[c]" in cu, "per-colour GS dispatch missing"
    assert "atomicAdd" not in cu, "GS colour pass must not use atomics"
    # pinned BC via invMass==0
    assert "invMass[p] > 0.f" in cu and "wsum <= 0.f" in cu, "pinned (invMass=0) BC handling missing"
    # ISOLATION: the module must NOT #include the mass-spring solver headers (comment mentions are fine).
    import re
    includes = re.findall(r'#\s*include\s*[<"]([^">]+)[">]', cu)
    assert not any("physics" in inc for inc in includes), f"xpbd module must not include physics: {includes}"
    assert "extern \"C\"" in hdr and "xpbd_step" in hdr and "xpbd_init" in hdr, "C API missing"

def test_rayleigh_alpha_decay():
    # alpha (Rayleigh) as the implicit per-substep velocity term v *= 1/(1+alpha*h): a free particle's speed
    # decays by that factor per substep. Verify it is a real, significant, monotone decay (settles), and
    # that a LARGER alpha decays faster (the monotonicity the CUDA test H asserts on the beam).
    def decay(alpha, dt=1/60, N=20, frames=120):
        h = dt/N; v = 1.0
        for _ in range(frames*N): v /= (1.0 + alpha*h)
        return v
    assert abs(decay(0.0) - 1.0) < 1e-12, "alpha=0 must not damp"
    d2, d6 = decay(2.0), decay(6.0)
    assert d2 < 0.05 and d6 < d2, f"alpha damping not significant/monotone: d2={d2:.4g} d6={d6:.4g}"

if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn(); print("PASS", fn.__name__)
    print("ALL PASS")
