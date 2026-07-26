// ReconGridManager.cs — Stage 1B-ii Task 2
// MonoBehaviour: per-frame physics→DC coupling demo.
//   Start():  preprocess sphere → pin top y-layer → set gravity into ExtForce → Upload →
//             create MassSpringSolver(physics) + DualContouring(recon) → initial dc.Build.
//   Update(): solver.Step(rb, physicsDt) → dc.Build(rb,...) [incl. RebuildIsectWorld] → DrawProceduralIndirect.
// IMPORTANT: MassSpringSolver receives the PHYSICS shader (Physics.compute), NOT recon.
//            DualContouring receives the RECON shader (Recon.compute).

using UnityEngine;
using Unity.Mathematics;
using ReconGridDC.Preprocess;
using ReconGridDC.Recon;
using ReconGridDC.Physics;
using ReconGridDC.Cutting;
using ReconGridDC.Core;

namespace ReconGridDC.Demo
{
    public sealed class ReconGridManager : MonoBehaviour
    {
        [Header("GPU compute")]
        public ComputeShader recon;    // Recon.compute — assign in Inspector or via DemoSceneSetup
        public ComputeShader physics;  // Physics.compute — SEPARATE from recon; assign in Inspector or via DemoSceneSetup
        public ComputeShader cutting;  // Cutting.compute — Stage 2C cut kernels. UNASSIGNED (null) ⇒ Stage-1 behavior.

        [Header("Surface shader")]
        public Shader surfaceShader;   // ReconGridDC/ReconSurface — assign in Inspector or via Shader.Find

        [Header("Grid parameters")]
        // RESOLUTION (2026-06-29): bumped 24³/L=0.25 → 48³/L=0.125 (same 6×6×6 world; sphere r=2 @ (3,3,3)
        // unchanged). The strict-paper cut margin is the per-component cut-point CENTROID on the blade-plane
        // cross-section; its OUTLINE is a voxel-resolution polygon, so finer voxels render it as a smoother
        // circle (no algorithm change — the paper's own result sharpens with resolution). Higher compute
        // cost (≈8× voxels/particles); dial back here if the demo is too slow. triCapacity auto-scales.
        public int3  dims         = new int3(48, 48, 48);
        public float L            = 0.125f;
        public int   qefIters     = 20;

        [Header("Sphere parameters")]
        public float3 sphereCenter = new float3(3f, 3f, 3f);
        public float  sphereRadius = 2.0f;

        [Header("Physics parameters")]
        public float gravity      = -9.81f;
        public bool  pinTopLayer  = true;
        public float physicsDt    = 0.02f;
        public float ks           = 7.5e4f;
        public float cs_damp      = 0.92f;   // 'cs' conflicts with ComputeShader field name; use cs_damp
        public float kb           = 2e4f;
        public float cb           = 0.9f;

        [Header("Cutting (Stage 2C) — assign Cutting.compute above to enable")]
        // AUTO blade sweep so a cut is VISIBLE without input: a vertical blade line spanning the
        // sphere in y sweeps slowly along z through the sphere centre over a few seconds.
        public bool   enableCut  = true;     // master toggle (also requires 'cutting' assigned)
        // Blade thickness D (world units) → cut-gap width = D. Paper §3.4: D is SUB-VOXEL (0.1–0.4 mm
        // on a 2.5 mm voxel → D/L = 0.04–0.16). With L=0.125, the paper-band D ∈ [0.005, 0.02].
        // SEAM FIX (2026-06-28 audit): 0.12 (D/L=0.48, ~half a voxel — 3× over the paper band) → 0.04
        // (D/L=0.16, top of band, ~1/3 of the old width). Drop to 0.02–0.03 for a thinner hairline;
        // the gap scales linearly with D and (with _CutPull≥4 in Cutting.compute) renders as exactly D.
        // NEAR-ZERO SEAM (user request 2026-06-29): the real wall-to-wall seam == D (the two per-component
        // cut centroids sit ±D/2 apart, EmitCutPoint). The large "gap" the user saw was NOT this seam — it
        // was the DELAMINATION, root-caused (cpu-mirror-delamination) to the InterpCutFP EMA lag on the cut
        // WALL (cutFPInterp, fixed below), NOT a seam-width problem. Independent of that, drop D 0.04→0.02
        // (D/L=0.16 at L=0.125, top of the paper sub-voxel band 0.04–0.16) for the requested hairline slit.
        // The gap scales linearly with D; do NOT set 0 (the two wall faces collapse to zero area).
        public float  bladeD     = 0.02f;
        // Sweep speed (world units/s). z-step/frame = bladeSpeed*physicsDt must stay < L (0.125) so no
        // Y-edge layer is skipped: 4.0*0.02 = 0.08 < 0.125 ✓. Raised 1.2→4.0 so the blade crosses the
        // sphere z-extent in ~1 s (was ~3 s) — the lower half can only detach + fall AFTER the far cap
        // is severed, so a slow sweep made the fall look absent (audit w0o40rg9j: timing, not algorithm).
        public float  bladeSpeed = 4.0f;
        // Stage 2D-1b (paper Fig 3.9, paper:1153-1165): the blade is now a HORIZONTAL X-line at a fixed
        // bladeY (derived below), swept in z — so the swept-quad normal is ≈ ±Y and the cut plane is
        // y = bladeY. bladeSpan: the blade overshoots the sphere in x by ±(radius*bladeSpan) so the
        // full cross-section is traversed (≈1.4 ⇒ x∈[center.x-2.8, center.x+2.8] for r=2). Replaces the
        // old bladeYPad role (which padded the vertical blade in y).
        public float  bladeSpan  = 1.4f;     // blade extends ±(radius*bladeSpan) in x past the sphere
        // Sweep extent along z, relative to the sphere centre (start at -range, end at +range).
        // 2D-1 (severing): must cover the sphere's FULL z-extent so the cut plane fully traverses the
        // cross-section → the two halves topologically disconnect and the unpinned half falls. Sphere
        // z∈[1,5] (center 3, r 2) ⇒ range≥2 needed; 2.2 (z∈[0.8,5.2]) clears it. (1.5 left z∈[1,1.5]&[4.5,5]
        // hinged.) Per-frame z-step = bladeSpeed*physicsDt = 4.0*0.02 = 0.08 < L=0.125, so no rest-Z edge
        // layer is skipped (the coverage invariant asserted in Start()).
        public float  bladeSweepRange = 2.2f;

        [Header("Stage 2D-2 verify — A/B rotation frame")]
        // 2D-2 A/B: off = force R=identity so the cut surface does NOT follow rotation (drifts/distorts).
        // When true, ComputeRotations runs each frame → R=polar(F) per corner → the cut surface rotates
        // WITH the freed piece. When false, ParticleRot stays identity-initialised → the cut points keep a
        // static world offset → the cut walls DISTORT as the piece spins (the visible 2D-2 regression).
        public bool   enableRFrame      = true;
        // After the sweep COMPLETES (cut-then-rotate contract — cut-while-rotate is unsupported), give the
        // freed lower piece a one-time angular velocity so it tumbles (real rotation → R≠identity → 2D-2
        // becomes visible). false ⇒ exactly today's behaviour (free-fall, pure translation, R≈identity).
        public bool   demoSpinAfterCut  = true;
        public float  demoSpinRate      = 3.0f;   // rad/s, spin about world +Z (tumbles in the XY view)

        [Header("Stage 2D-2b — temporal FP interp")]
        // Temporal interpolation factor α for the post-cut feature points (paper Discussion, clean:1271-1276:
        // "interpolate the positions of post-cut feature points from the previous frame to the current frame").
        // 1.0 = OFF / byte-identical (no smoothing); lower = MORE smoothing of the per-frame cut-FP drift.
        //
        // DELAMINATION FIX (cpu-mirror-delamination, 2026-06-29): the EMA runs on the cut FP _CutFP
        // (InterpCutFP). The original delamination was a lag of the EMA-smoothed _CutFP against a SEPARATE
        // un-smoothed outer-skin vertex (the former _CutFPExternal, since removed — the skin now shares this
        // same _CutFP): under a sustained post-cut fall the un-smoothed skin tracked the falling piece while
        // the smoothed wall TRAILED → rounded skin dome on top, lagged flat wall sliver below, dark gap = the
        // "分层" the user saw. Two independent CPU mirrors confirmed this. α was 0.5 (bottom of the safe band
        // the comment itself flagged). Set α = 1.0 → lerp(prev,curr,1)=curr → _CutFP = current every frame →
        // tracks the fall EXACTLY → no lag. The EMA's only benefit (smoother cut OPENING during the slow
        // sweep) does not apply to the fast post-cut fall, so 1.0 loses nothing the demo needs. The 2D-2b
        // machinery is preserved (set <1 to re-enable); a FALL-AWARE α (relax→1 for fast-moving slots) would
        // keep opening-smoothing. (With _CutFPExternal removed, skin and wall share _CutFP so even α<1 can no
        // longer reopen the skin-vs-wall gap; α only affects the cut surface as a whole now.)
        public float  cutFPInterp = 1.0f;

        // ── Runtime state ─────────────────────────────────────────────────────────────────
        ReconBuffers     rb;
        Material         mat;
        Bounds           renderBounds;
        MassSpringSolver solver;
        DualContouring   dc;
        // Stage 2C cut runtime
        CuttingTool      tool;
        CutDetector      cutDetector;
        float            bladeZ;       // current sweep position along z (world)
        bool             cutActive;    // true when cutting shader assigned + enableCut
        // 2D-2 demo: one-shot guard so the freed lower piece gets its tumble angular velocity exactly once
        // (the frame the sweep finishes). Reset only on re-init (no re-init path here ⇒ one-shot per play).
        bool             _spinApplied = false;

        // ── DEBUG: runtime readback to settle the "does the lower half fall?" question ───────
        // Every ~30 frames logs: blade sweep progress, cut points emitted, # severed NbrIdx slots
        // (rising ⇒ severing IS happening), and the min-Y of the freed (active+unpinned, below-cut)
        // corners (decreasing ⇒ the piece IS falling). Set false to silence. (Temporary instrument.)
        public bool      debugFall = true;
        // DEBUG (delamination 2026-06-29): tag-color the render — RED = cut surface (skin+wall, the shared
        // per-component _CutFP), GRAY = body (_VoxelExternalFP). (The old GREEN cut-skin branch is gone with
        // _CutFPExternal; green never appears now.) Toggle in the Inspector to SEE what the bright sliver
        // actually is (cut surface vs fallen lower-half body). Drives ReconSurface _DebugTag.
        public bool      debugTagColor = false;
        int              _dbgFrame;
        int              _dbgSeveredBase = -1;   // baseline severed-slot count (grid-boundary -1s)

        void Start()
        {
            // 1. CPU preprocessing: build level set + background grid
            var ls = new SphereLevelSet(sphereCenter, sphereRadius);
            var g  = BackgroundGrid.Build(ls, dims, L, float3.zero);

            // 2. Pin the TOP CAP of the actual TISSUE (paper §2.1.4 + Fig 3.9, paper:485-499 / 1153-1165).
            //    Stage 2D-1b: pin ONLY tissue (active) corners within an L-thick band at the sphere's
            //    top (y ≥ (center.y + radius) − L). The old loop pinned the j=dims.y corner layer, which
            //    for the default params sits in EMPTY space ABOVE the sphere (sphere top y=5, pin layer
            //    y=6) — combined with the empty-cage springs this hung BOTH severed halves from empty
            //    space so nothing fell. Pinning the top tissue cap anchors the upper hemisphere; the
            //    lower hemisphere (y < bladeY) is unpinned + fully severed → it free-falls (Fig 3.9).
            //    Gate on cornerActive so no empty-space corner is pinned (a pinned inactive corner would
            //    be a no-op anyway since it is already frozen, but gating keeps the anchor on real tissue).
            if (pinTopLayer)
            {
                float yTopBand = (sphereCenter.y + sphereRadius) - L;
                for (int c = 0; c < g.cornerCount; c++)
                {
                    if (g.cornerActive[c] == 0) continue;          // tissue corners only
                    if (g.cornerPos[c].y >= yTopBand) g.pinned[c] = 1;
                }
            }

            // 3. Allocate GPU buffers. triCapacity = external + cut tris share ONE triBuf.
            //    maxCutTris ≤ 2*GridEdgeCount (plan §1 R-E): enlarge from frame 1 or BuildCutTriangles
            //    silently drops cut faces as cuts accumulate.
            int triCapacity = 6 * g.surfaceEdges.Length + 1024 + 2 * g.gridEdges.Length;
            rb = new ReconBuffers(g, triCapacity);

            // 4. Upload preprocessing data (CornerPos, Isect, physics topology, etc.)
            rb.Upload(g);

            // 5. Set gravity external force per corner: (0, gravity * mass, 0).
            //    Constant external force — set once here; physics kernels read it every Step.
            var ext = new float3[g.cornerCount];
            for (int c = 0; c < g.cornerCount; c++)
                ext[c] = new float3(0f, gravity * g.mass[c], 0f);
            rb.ExtForce.SetData(ext);

            // 6. Create physics solver (MUST use the Physics shader, not recon)
            solver = new MassSpringSolver(physics)
            {
                Ks = ks,
                Cs = cs_damp,
                Kb = kb,
                Cb = cb,
            };

            // 7. Create DC reconstructor (Recon shader) + wire the optional Cutting shader (Stage 2C).
            //    cutting==null ⇒ DualContouring skips all cut dispatches → Stage-1-identical.
            dc = new DualContouring(recon, cutting);

            // 7b. Stage 2C cut setup (Stage 2D-1b: HORIZONTAL cut, paper Fig 3.9, paper:1153-1165).
            //     The blade is a HORIZONTAL line spanning X at a fixed bladeY, swept along z through the
            //     sphere. Endpoints S=(xMin,bladeY,bladeZ), E=(xMax,bladeY,bladeZ). As bladeZ advances,
            //     the swept quad (S,E,Sprev,Eprev) lies in the y=bladeY plane → its normal n_cut ≈ ±Y →
            //     it severs Y-axis grid edges → a horizontal cut plane at y=bladeY. The lower hemisphere
            //     (y<bladeY) is unpinned + fully severed → falls straight down (translation, no rotation
            //     → cut walls stay clean without the deferred 2D-2 R-frame).
            cutActive = (cutting != null) && enableCut;
            if (cutActive)
            {
                cutDetector = new CutDetector(cutting);
                // Coverage invariant (workflow w87ghv7f6): the swept quad spans only the prev→current
                // blade-Z strip, so every rest-Z layer must be hit while in some frame's strip — i.e. the
                // per-frame z-step bladeSpeed*physicsDt must be < L, or a whole rest-Z edge layer falls
                // between two strips and is never tested (a fresh hinge ring). Rest detection does NOT make
                // this safer. Assert it loudly rather than silently miss a layer.
                if (bladeSpeed * physicsDt >= L)
                    Debug.LogWarning($"[ReconGridDC] blade z-step {bladeSpeed * physicsDt:F3} >= L {L:F3} — " +
                                     "rest-Z layers may be skipped (residual hinges). Lower bladeSpeed or physicsDt.");
                bladeZ      = sphereCenter.z - bladeSweepRange;     // start before the sphere
                // 2D-1b: bladeY is DERIVED — set just ABOVE the sphere centre and off the corner plane
                //   (+L*0.5) so the cut plane does not sit exactly on a y-corner layer (avoids double
                //   cut-point emission, mirrors the old +L*0.5 x-offset robustness fix).
                float bladeY = sphereCenter.y + L * 0.5f;
                float xMin   = sphereCenter.x - sphereRadius * bladeSpan;
                float xMax   = sphereCenter.x + sphereRadius * bladeSpan;
                var s0 = new Vector3(xMin, bladeY, bladeZ);
                var e0 = new Vector3(xMax, bladeY, bladeZ);
                tool   = new CuttingTool(s0, e0, bladeD);           // ctor: Sprev=S, Eprev=E → zero sweep
            }

            // 8. Run initial DC build (RebuildIsectWorld first, then full pipeline)
            //    cutFPInterp (2D-2b α) passed through; 1.0 = OFF, default field 0.5 smooths the cut-FP drift.
            dc.Build(rb, dims, L, qefIters, cutFPInterp);

            // 9. Build the draw material and bind GPU buffers.
            //    _CutFP / _CutFPNormalF bound UNCONDITIONALLY (plan §3.7 / A2-F6): even with zero cuts
            //    the buffers hold zeros and are never sampled by external verts.
            mat = new Material(surfaceShader);
            mat.SetBuffer("_Tri",             rb.Tri);
            mat.SetBuffer("_VoxelExternalFP", rb.VoxelExternalFP);
            mat.SetBuffer("_ExtNormalF",      rb.VoxelExternalFPNormalF);
            mat.SetBuffer("_CutFP",           rb.CutFP);
            mat.SetBuffer("_CutFPNormalF",    rb.CutFPNormalF);

            // 10. Compute render bounds (world-space AABB of the grid, 3× overestimate for sag)
            float3 gridExtent = (float3)dims * L;
            renderBounds = new Bounds(
                (Vector3)(float3)(gridExtent * 0.5f),
                (Vector3)(gridExtent * 3f));
        }

        void Update()
        {
            if (rb == null || mat == null) return;

            // Per-frame coupling: physics moves corners, then (2C) detect cuts, then DC rebuilds.
            // ORDER IS CRITICAL: solver.Step → cutDetector.Dispatch → dc.Build (RebuildIsectWorld inside).
            solver.Step(rb, physicsDt);          // mass-spring adaptive RK45 → updates CornerPos

            // Stage 2C: advance the AUTO blade sweep + detect cuts BEFORE dc.Build (plan §4 ordering).
            // CutDetector marks voxelCutMask + emits cut points; dc.Build then consumes them.
            if (cutActive && cutDetector != null)
            {
                // Sweep the blade slowly along z through the sphere; stop once past the far extent
                // (cuts are cumulative — the slit stays open after the sweep finishes).
                float zEnd = sphereCenter.z + bladeSweepRange;
                if (bladeZ < zEnd) bladeZ += bladeSpeed * physicsDt;

                // 2D-1b: HORIZONTAL blade — X-line at the derived bladeY, swept in z (matches Start()).
                float bladeY = sphereCenter.y + L * 0.5f;
                float xMin   = sphereCenter.x - sphereRadius * bladeSpan;
                float xMax   = sphereCenter.x + sphereRadius * bladeSpan;
                var newS = new Vector3(xMin, bladeY, bladeZ);
                var newE = new Vector3(xMax, bladeY, bladeZ);
                tool.Advance(newS, newE);        // rolls Sprev/Eprev, recomputes n_cut + Valid

                // L + grid origin (Vector3.zero — the BackgroundGrid is built with origin float3.zero in
                // Start()) feed DetectCut's REST-edge M-T (rest pos = origin + coord*L).
                cutDetector.Dispatch(tool, rb, new Vector3Int(dims.x, dims.y, dims.z), L, Vector3.zero);
            }

            // ── 2D-2 verify: one-time tumble of the freed lower piece AFTER the sweep completes ──────────
            // CONTRACT: cutting must COMPLETE before any rotation (detection M-Ts the rest plane with no R —
            // cut-while-rotate is unsupported). So this fires only once the sweep has finished (bladeZ>=zEnd).
            // We add a rigid angular velocity ω about a horizontal axis (world +Z) through the lower-piece
            // centroid: vel[c] += ω × (pos[c] − centroid). The piece then spins → F is a real rotation →
            // ComputeParticleRot yields R≈that rotation → the cut surface follows it (with enableRFrame on).
            if (demoSpinAfterCut && !_spinApplied && cutActive && cutDetector != null)
            {
                float zEnd   = sphereCenter.z + bladeSweepRange;   // sweep-end z (same as the sweep block)
                float bladeY = sphereCenter.y + L * 0.5f;          // cut plane y (same as the sweep block)
                if (bladeZ >= zEnd)
                {
                    int cn = rb.CornerCount;
                    var pos = new float3[cn]; rb.CornerPos.GetData(pos);      // float3[cornerCount], stride 12
                    var act = new int[cn];    rb.CornerActive.GetData(act);   // int[cornerCount], stride 4
                    var vel = new float3[cn]; rb.Vel.GetData(vel);            // float3[cornerCount], stride 12 (writable)

                    // Centroid of the LOWER piece = active corners below the cut plane.
                    float3 centroid = float3.zero; int nLow = 0;
                    for (int c = 0; c < cn; c++)
                    {
                        if (act[c] == 0) continue;                 // freed/simulated tissue corners only
                        if (pos[c].y < bladeY) { centroid += pos[c]; nLow++; }
                    }

                    if (nLow > 0)
                    {
                        centroid /= nLow;
                        // Spin about world +Z so the piece tumbles in the XY view. Vector3.Cross per Unity.
                        var omega = new Vector3(0f, 0f, demoSpinRate);
                        for (int c = 0; c < cn; c++)
                        {
                            if (act[c] == 0) continue;
                            if (pos[c].y >= bladeY) continue;
                            var r = new Vector3(pos[c].x - centroid.x,
                                                pos[c].y - centroid.y,
                                                pos[c].z - centroid.z);
                            var dv = Vector3.Cross(omega, r);      // ω × r = rigid rotational velocity
                            vel[c] += new float3(dv.x, dv.y, dv.z);
                        }
                        rb.Vel.SetData(vel);                       // write the tumble velocity back
                        Debug.Log($"[SPIN] applied ω=(0,0,{demoSpinRate:F2}) to {nLow} lower corners; " +
                                  $"centroid=({centroid.x:F2},{centroid.y:F2},{centroid.z:F2})");
                    }
                    _spinApplied = true;   // one-shot regardless of nLow (sweep is done; do not retry)
                }
            }

            // Stage 2D-2 (paper §2.1.2 / Berndt [22]): compute each particle's rotation R AFTER
            // cutDetector.Dispatch (so SeverLinks' severed edges are excluded from F — M1 ordering) and
            // BEFORE dc.Build (AccumulateCutFP reads frame N's R). With R≈identity (the pure-translation
            // demo, protected by the deadband) the cut surface is byte-/sub-voxel-identical to pre-2D-2.
            // 2D-2 A/B gate: when enableRFrame is false we SKIP ComputeRotations → ParticleRot stays
            // identity-initialised → the cut surface uses the static world offset and distorts under rotation.
            if (cutActive && enableRFrame) solver.ComputeRotations(rb);  // writes rb.ParticleRot = polar(F) per active corner (only cut kernels read it)

            dc.Build(rb, dims, L, qefIters, cutFPInterp);  // RebuildIsectWorld + DC rebuild + (2C) cut surface + (2D-2b) FP interp

            mat.SetFloat("_DebugTag", debugTagColor ? 1f : 0f);  // DEBUG: live tag-color toggle (wall/skin/body)

            // Render the DC surface via DrawProceduralIndirect using the arg buffer filled by DC_WriteIndirectArgs.
            Graphics.DrawProceduralIndirect(
                mat,
                renderBounds,
                MeshTopology.Triangles,
                rb.IndirectArgs,
                argsOffset: 0);

            // ── DEBUG readback (temporary instrument) ───────────────────────────────────────
            if (debugFall && (_dbgFrame++ % 30) == 0)
            {
                int cn  = rb.CornerCount;
                float zEnd   = sphereCenter.z + bladeSweepRange;
                float bladeY = sphereCenter.y + L * 0.5f;

                var nbr = new int[6 * cn];      rb.NbrIdx.GetData(nbr);
                int severed = 0; for (int i = 0; i < nbr.Length; i++) if (nbr[i] == -1) severed++;
                if (_dbgSeveredBase < 0) _dbgSeveredBase = severed;   // baseline = grid-boundary -1s at frame 0

                var pos = new float3[cn]; rb.CornerPos.GetData(pos);
                var act = new int[cn];    rb.CornerActive.GetData(act);
                var pin = new int[cn];    rb.Pinned.GetData(pin);
                float minY = float.PositiveInfinity, maxY = float.NegativeInfinity; int nLow = 0;
                for (int c = 0; c < cn; c++)
                {
                    if (act[c] == 0 || pin[c] != 0) continue;          // freed, simulated corners only
                    if (pos[c].y < bladeY) { nLow++; if (pos[c].y < minY) minY = pos[c].y; }
                    if (pos[c].y > maxY) maxY = pos[c].y;
                }
                var cpc = new uint[1]; rb.CutPointCounter.GetData(cpc);
                Debug.Log($"[FALL] bladeZ={bladeZ:F2}/{zEnd:F2}  cutPoints={cpc[0]}  " +
                          $"severedNbr={severed}(+{severed - _dbgSeveredBase})  " +
                          $"lowerActive={nLow}  lowerMinY={(nLow>0?minY:0f):F3}  maxY={maxY:F3}");
            }
        }

        void OnDestroy()
        {
            rb?.Dispose();
            if (mat != null) Destroy(mat);
        }
    }
}
