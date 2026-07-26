// CuttingTool.cs — Stage 2B (design §5.1 / §3.1; paper §2.1.3, Fig 2.8)
// Represents the cutting blade as two 3-D endpoints S (top) and E (bottom)
// separated by thickness D. Maintains previous-frame state to form a swept quad.
//
// Swept-plane geometry (paper Fig 2.8; gap C22 / PAPER-SILENT diagonal convention):
//   T1 = (Sprev, Eprev, E)       — first triangle of the swept quad
//   T2 = (Sprev, E, S)           — second triangle of the swept quad (diagonal Sprev→E)
// Both triangles are tested per-edge by DetectCut.
//
// Cut-surface normal (PAPER-SILENT: paper gives no formula for n_cut direction):
//   n_cut = normalize(cross(Eprev - Sprev, E - Sprev))
// The two cut points are offset ±D/2 along ±n_cut — the ±D/2 SPLIT itself is paper §3.4 (p12:
// "two cutting points are generated on the cutting edge ... the gap gradually widens"); only the
// n_cut DIRECTION is paper-silent.
//
// Degenerate guard (design §10 / PAPER-SILENT robustness):
//   When ‖cross(L1,L2)‖ < eps the swept area is zero/near-zero (blade did not move).
//   Set Valid = false so DetectCut skips this frame.
//
// Traceability: paper §2.1.3 / Fig 2.8; design §5.1 / §10; PAPER-SILENT items annotated.

using Unity.Mathematics;
using UnityEngine;

namespace ReconGridDC.Cutting
{
    /// <summary>
    /// Cutting blade state: current/previous frame endpoints + swept-plane triangles.
    /// Call <see cref="Advance"/> each physics frame to update geometry.
    /// </summary>
    public sealed class CuttingTool
    {
        // ── Blade geometry (current frame) ───────────────────────────────────────────────
        /// <summary>Blade top endpoint, current frame. paper §2.1.3 S.</summary>
        public Vector3 S { get; private set; }
        /// <summary>Blade bottom endpoint, current frame. paper §2.1.3 E.</summary>
        public Vector3 E { get; private set; }
        /// <summary>Blade thickness (gap width, world units). design §3.1 / §5.3.</summary>
        public float D   { get; private set; }

        // ── Previous-frame blade state ────────────────────────────────────────────────────
        /// <summary>Blade top, previous frame (S_k). paper Fig 2.8.</summary>
        public Vector3 Sprev { get; private set; }
        /// <summary>Blade bottom, previous frame (E_k). paper Fig 2.8.</summary>
        public Vector3 Eprev { get; private set; }

        // ── Derived swept-plane geometry ──────────────────────────────────────────────────
        /// <summary>
        /// Swept-plane triangle 1: (Sprev, Eprev, E). paper Fig 2.8 / §2.1.3; gap C22.
        /// Diagonal is Sprev→E (splits quad Sprev,Eprev,E,S by the S_k→E_{k+1} diagonal).
        /// </summary>
        public Vector3 T1V0 => Sprev;  // triangle 1 vertex 0 = Sprev  (= S_k)
        public Vector3 T1V1 => Eprev;  // triangle 1 vertex 1 = Eprev  (= E_k)
        public Vector3 T1V2 => E;      // triangle 1 vertex 2 = E      (= E_{k+1})

        /// <summary>
        /// Swept-plane triangle 2: (Sprev, E, S). paper Fig 2.8 / §2.1.3; gap C22.
        /// </summary>
        public Vector3 T2V0 => Sprev;  // triangle 2 vertex 0 = Sprev  (= S_k)
        public Vector3 T2V1 => E;      // triangle 2 vertex 1 = E      (= E_{k+1})
        public Vector3 T2V2 => S;      // triangle 2 vertex 2 = S      (= S_{k+1})

        /// <summary>
        /// Cut-surface outward normal. Computed as normalize(cross(Eprev-Sprev, E-Sprev)).
        /// PAPER-SILENT: paper gives no formula for the n_cut DIRECTION; this is the geometric swept-plane
        /// normal. Cut points are offset ±(D/2)*n_cut from P_hit — the ±D/2 SPLIT is paper §3.4 (p12).
        /// </summary>
        public Vector3 NCut { get; private set; }

        /// <summary>
        /// True when the swept quad is non-degenerate (blade moved sufficiently this frame).
        /// PAPER-SILENT robustness (design §10): DetectCut must early-out when Valid = false.
        /// </summary>
        public bool Valid { get; private set; }

        // Degenerate threshold: cross-product magnitude below this → blade did not move.
        // PAPER-SILENT: chosen to reject sub-millimetre sweeps. design §10.
        const float DegenerateEps = 1e-8f;

        // ── Construction ──────────────────────────────────────────────────────────────────
        /// <summary>
        /// Initialise blade at rest pose (Sprev = S, Eprev = E → zero sweep, Valid = false).
        /// </summary>
        /// <param name="s">Top endpoint of blade.</param>
        /// <param name="e">Bottom endpoint of blade.</param>
        /// <param name="d">Blade thickness D (world units).</param>
        public CuttingTool(Vector3 s, Vector3 e, float d)
        {
            D     = d;
            S     = s;
            E     = e;
            Sprev = s;
            Eprev = e;
            RefreshDerived();
        }

        // ── Per-frame update ──────────────────────────────────────────────────────────────
        /// <summary>
        /// Advance to a new blade position for this frame.
        /// Rolls previous positions, then recomputes n_cut and validates the swept area.
        /// Call once per physics frame BEFORE dispatching DetectCut.
        /// </summary>
        /// <param name="newS">New top endpoint (S_{k+1}).</param>
        /// <param name="newE">New bottom endpoint (E_{k+1}).</param>
        public void Advance(Vector3 newS, Vector3 newE)
        {
            Sprev = S;  // roll: S_k becomes S_{k-1}
            Eprev = E;  // roll: E_k becomes E_{k-1}
            S     = newS;
            E     = newE;
            RefreshDerived();
        }

        /// <summary>
        /// Update the blade/stick thickness without disturbing the previous/current endpoint history.
        /// </summary>
        public void SetThickness(float d)
        {
            D = Mathf.Max(0.001f, d);
        }

        // ── Internal ──────────────────────────────────────────────────────────────────────
        void RefreshDerived()
        {
            // n_cut = normalize(cross(Eprev - Sprev, E - Sprev))
            // PAPER-SILENT: geometric normal of the swept quad (points perpendicular to blade plane).
            Vector3 arm1 = Eprev - Sprev;  // L1-like: along the blade in prev frame
            Vector3 arm2 = E     - Sprev;  // L2-like: from Sprev toward new E

            Vector3 cross = Vector3.Cross(arm1, arm2);
            float   mag   = cross.magnitude;

            // Degenerate guard: if the swept area is near-zero, the blade did not move.
            // design §10 / PAPER-SILENT robustness.
            if (mag < DegenerateEps)
            {
                Valid = false;
                NCut  = Vector3.zero;
            }
            else
            {
                Valid = true;
                NCut  = cross / mag;  // unit cut-plane normal
            }
        }
    }
}
