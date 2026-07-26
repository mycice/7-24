// DeviationLedger.cs  - documentation-only; no runtime logic.
// Single authoritative list of all places this reimplementation departs from the Li/Zhou CMPB 2026 paper.
// Each deviation is labeled in the corresponding kernel/C# file with // DEVIATION Dn.
// Design reference: docs/superpowers/specs/2026-06-27-dc-cutting-design.md §2.

namespace ReconGridDC.Core
{
    /// Deviation ledger  - the ONLY places this reimplementation departs from the paper/design.
    /// D1: Eq3 QEF force = sum n_i*(n_i·(p_i-x)) at current x (paper's scalar reading gives F -  at x0).
    /// D2: Eq11 structural force subtracts rest length L0 (paper omitted it -> zero-length collapse).
    /// D3: acceleration = (F_ext + F_int)/m (Eq17's sign). F_int (Eq11/Eq14) is the ACTUAL restoring force on the particle, so Newton gives +F_int; Eq9's (F_ext-F_int) uses the opposite F_int convention and combined with Eq11 gives anti-restoring divergence (corrected after GPU oracles caught it).
    /// D4: RK45 step h_n = h*0.9*(eps/Delta)^(1/5) (paper Eq21 printed Delta/eps, inverted).
    /// D5: QEF Eq5 stop when boxSDF(x_next)>=0, keep last interior point (literal SDF<0 halts at x0).
    /// D6: per-frame retrospective RK45 h-adaptation (design §4.6 pseudocode shows per-substep) to avoid per-substep GPU readback stalls; stage math + D4 formula unchanged.
    public static class DeviationLedger {
    // D7 (Stage 4 - , task_plan.md): the SS2.1.3 swept plane is time-refined for MOVING tissue -
    //    rod-phase world quad M-T on DEFORMED edges + per-RK45-substep CCD of the moving edge vs the
    //    static blade (quadratic coplanarity, exact). Paper demos are zero-g; this is the documented
    //    completion (diagnosis wf_e3f28151; the Stage-4 material inverse-map variant was superseded -
    //    it tears at real folds, wf_19bb05de).
    // D8 (audit wf_6647e0ee C3): Eq17 gains a PAPER-SILENT global mass-proportional (Rayleigh)
    //    damping −alpha·v (globalDamp, default 2/s; 0 = paper-exact). Kelvin-Voigt cs/cb damp only
    //    RELATIVE velocity; without alpha the free-falling flap accelerates unboundedly (terminal
    //    velocity g/alpha). Stability decision.
    // D9 (audit wf_60781cea A1/A3, review wf_4a45e51c): (a) AIR-EDGE CUT GATE  - cutting edges are
    //    restricted to edges with >=1 inside (phi<0) endpoint; blade crossings in the AIR part of
    //    occupied surface voxels are not tissue cuts (SS2.1.1 edges-as-physical-interactions +
    //    SS2.1.4 outside particles deleted). (b) cut-wall WINDING is paper-silent; walls are oriented
    //    outward via refDir = toward the severed opposite endpoint.
    //
}
}
