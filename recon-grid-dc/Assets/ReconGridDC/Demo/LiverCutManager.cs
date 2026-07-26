// LiverCutManager.cs — liver-import pipeline, design §5.
// Mirrors ReconGridManager but: (1) geometry from a liver .msh via MeshLevelSet + GridFit (negative
// origin), (2) anchor = pin a bbox-end band (deformable + anchored), (3) the auto-sweep blade is
// replaced by an INTERACTIVE keyboard-driven rod (CuttingTool) with per-frame sub-stepping, (4) the
// DC surface is drawn with the ReconSurface debug material + a LineRenderer rod gizmo. Zero core edits.
//
// Per frame: solver.Step → read keyboard → sub-step rod (Advance + Dispatch per sub-step) →
//            ComputeRotations → dc.Build → DrawProceduralIndirect + rod line.
//
// INPUT: legacy Input Manager (Input.GetKey) — Built-in pipeline default. If the project is configured
// for the new Input System only, swap ReadRodInput() accordingly.

using System.IO;
using UnityEngine;
using Unity.Mathematics;
using ReconGridDC.Preprocess;
using ReconGridDC.Recon;
using ReconGridDC.Physics;
using ReconGridDC.Cutting;

namespace ReconGridDC.Demo
{
    public sealed class LiverCutManager : MonoBehaviour
    {
        [Header("GPU compute (assign in Inspector / LiverSceneSetup)")]
        public ComputeShader recon;
        public ComputeShader physics;
        public ComputeShader cutting;        // null ⇒ Stage-1 (no cut), liver still simulates + draws
        public Shader        surfaceShader;  // ReconGridDC/ReconSurface (debug material)

        [Header("Liver model (under StreamingAssets)")]
        public string mshFileName = "liver3_refined_1.msh";

        [Header("Grid fitting + smoothing")]
        public int targetLongAxisVoxels = 48;   // longest axis voxel count → L = maxExtent/this (final FPS knob)
        public int marginVoxels         = 2;    // empty shell so the DC surface closes
        public int qefIters             = 10;   // Fix 1 (dc.Build lever): QEF iters / surface voxel (was 20)
        public int smoothIters          = 3;    // Fix 2: Taubin smoothing passes on the baked SDF (0 = off)
        public int bakeOversample       = 1;    // Fix 2: bake nodes per recon voxel (1 = recon-aligned)

        [Header("Physics — paper §3.4 liver values (TUNE; cs/mass = settling R5, hMax = stability R1)")]
        public float gravity    = -9.81f;
        public float physicsDt  = 0.02f;
        public float ks         = 7.5e4f;   // paper §3.4 structural stiffness (was demo-soft 800 → stretched)
        public float cs_damp    = 0.92f;    // paper §3.4 — NOTE ζ≈0.002 at mass=1 (nearly undamped); raise if it rings (R5)
        public float kb         = 2e4f;     // paper §3.4 bending
        public float cb         = 0.9f;     // paper §3.4 bending damping
        public float massScale  = 1f;       // per-corner mass × ; raise to damp ringing + allow larger hMax (R5)
        public float solverHMax = 1.5e-3f;  // Fix 1: RK45 substep size cap; lower if jitter
        public int   maxSubstepsPerFrame = 16; // PERF (paper §3.2 decoupling): hard cap on physics substeps/
                                               // frame → bounded FPS even when the stiff lattice rings. Lower
                                               // = faster FPS / more slow-motion; raise cs_damp/massScale to
                                               // settle faster (larger h) so fewer substeps are needed.

        [Header("Anchor (deformable + pinned) — pin the cap in +anchorAxis")]
        public Vector3 anchorAxis = new Vector3(1, 0, 0);  // tunable end/axis
        public float   anchorBand = 1.5f;                  // world-thickness of the pinned cap

        [Header("Interactive rod (keyboard)")]
        public float rodLength      = 12f;     // segment length (spans the cut cross-section)
        public float rodThicknessDOverL = 0.16f;  // D = this * L (sub-voxel cut gap, paper §3.4 band)
        public float moveSpeed      = 5f;      // units/s (WASD = X/Z, QE = Y)
        public float rotSpeed       = 60f;     // deg/s (arrows pitch/yaw)
        public float cutFPInterp    = 1.0f;    // 2D-2b α (1 = off)

        // ── runtime ──────────────────────────────────────────────────────────────────────────
        ReconBuffers     rb;
        Material         mat;
        Bounds           renderBounds;
        MassSpringSolver solver;
        DualContouring   dc;
        CuttingTool      tool;
        CutDetector      cutDetector;
        LineRenderer     rodLine;
        bool             cutActive;

        int3   dimsRt;
        float  Lrt;
        float3 originRt;
        float3 gridCenter;
        Vector3 rodCenter, rodAxis;

        public float3 GridCenter => gridCenter;
        public float3 GridExtent => (float3)dimsRt * Lrt;

        void Start()
        {
            // 1. Load liver + build the mesh SDF.
            string path = Path.Combine(Application.streamingAssetsPath, mshFileName);
            MshMesh mesh = MshLoader.LoadMsh(path);
            MshSurface surf = MshLoader.ExtractBoundary(mesh);
            MshLoader.Bounds(mesh, out float3 bmin, out float3 bmax);
            var meshLs = new MeshLevelSet(surf);

            // 2. Size the grid to the liver bbox (+ margin shell). Negative origin is fine.
            GridFit.FitToBounds(bmin, bmax, targetLongAxisVoxels, marginVoxels, out dimsRt, out Lrt, out originRt);

            // 2a. Fix 2: bake + Taubin-smooth the SDF over the full recon domain → smoother surface.
            //     (BackgroundGrid then samples cheap trilinear lookups; the brute-force SDF is paid once
            //     per bake node here, so startup is only modestly improved, not eliminated — see SmoothedSdfGrid.)
            var ls = new SmoothedSdfGrid(meshLs, originRt, dimsRt, Lrt, bakeOversample, smoothIters);
            var g = BackgroundGrid.Build(ls, dimsRt, Lrt, originRt);
            gridCenter = originRt + 0.5f * (float3)dimsRt * Lrt;

            // 2b. Empty/all-outside grid guard (design §5): voxelOccupied must have BOTH 0s and 1s.
            LiverGrid.CountOccupancy(g.voxelOccupied, out bool anyIn, out bool anyOut);
            if (!anyIn) { Debug.LogError("[LiverCutManager] grid contains NO tissue (liver outside the grid / mis-sized L,origin). Aborting."); return; }
            if (!anyOut) { Debug.LogError("[LiverCutManager] grid has NO empty shell (liver fills the grid). Increase marginVoxels. Aborting."); return; }

            // 3. Anchor: pin active (tissue) corners in the +anchorAxis cap band so the liver sags but
            //    does not free-fall. Guard: at least one corner must be pinned. design §5.
            int pinnedCount = LiverGrid.PinCap(g.cornerPos, g.cornerActive, g.pinned, (float3)anchorAxis, anchorBand);
            if (pinnedCount == 0) { Debug.LogError("[LiverCutManager] anchor pinned 0 corners (band too thin / wrong axis). Increase anchorBand. Aborting."); return; }
            Debug.Log($"[LiverCutManager] grid {dimsRt} L={Lrt:F3} origin={originRt} | boundary tris={surf.tris.Length} | pinned={pinnedCount}");

            // 3b. Mass scale (R5 settling knob): scale per-corner mass before upload + gravity force.
            if (massScale != 1f)
                for (int c = 0; c < g.cornerCount; c++) g.mass[c] *= massScale;

            // 4. GPU buffers + upload (same triCapacity formula as the demo).
            int triCapacity = 6 * g.surfaceEdges.Length + 1024 + 2 * g.gridEdges.Length;
            rb = new ReconBuffers(g, triCapacity);
            rb.Upload(g);

            // 5. Gravity external force.
            var ext = new float3[g.cornerCount];
            for (int c = 0; c < g.cornerCount; c++) ext[c] = new float3(0f, gravity * g.mass[c], 0f);
            rb.ExtForce.SetData(ext);

            // 6. Solver + DC + cut tool.
            solver = new MassSpringSolver(physics) { Ks = ks, Cs = cs_damp, Kb = kb, Cb = cb };
            solver.hMax = solverHMax;   // Fix 1: RK45 substep-cap stability/perf knob (R1)
            solver.maxSubstepsPerFrame = maxSubstepsPerFrame;   // PERF: bound per-frame physics cost
            dc = new DualContouring(recon, cutting);

            cutActive = cutting != null;
            // +L/2 nudge so the initial rod plane doesn't sit exactly on a rest-corner layer (avoids
            // double cut-point emission — mirrors the demo's bladeY +L*0.5 robustness). design §minor.
            rodCenter = (Vector3)gridCenter + new Vector3(0f, Lrt * 0.5f, 0f);
            rodAxis   = new Vector3(1, 0, 0);
            float rodD = rodThicknessDOverL * Lrt;
            GetRodEndpoints(rodCenter, rodAxis, out Vector3 s0, out Vector3 e0);
            if (cutActive)
            {
                cutDetector = new CutDetector(cutting);
                tool = new CuttingTool(s0, e0, rodD);   // Sprev=S, Eprev=E ⇒ zero initial sweep
            }

            // 7. Initial DC build.
            dc.Build(rb, dimsRt, Lrt, qefIters, cutFPInterp);

            // 8. Debug material + buffer binds (identical to the demo).
            mat = new Material(surfaceShader);
            mat.SetBuffer("_Tri",             rb.Tri);
            mat.SetBuffer("_VoxelExternalFP", rb.VoxelExternalFP);
            mat.SetBuffer("_ExtNormalF",      rb.VoxelExternalFPNormalF);
            mat.SetBuffer("_CutFP",           rb.CutFP);
            mat.SetBuffer("_CutFPNormalF",    rb.CutFPNormalF);

            // 9. Render bounds (grid AABB, padded for sag) + rod gizmo line.
            float3 ext3 = (float3)dimsRt * Lrt;
            renderBounds = new Bounds((Vector3)gridCenter, (Vector3)(ext3 * 2f));
            SetupRodLine(s0, e0);
        }

        void Update()
        {
            if (rb == null || mat == null) return;

            // Physics step (deform from last frame's forces).
            solver.Step(rb, physicsDt);

            // Interactive rod + sub-stepped cut.
            if (cutActive && cutDetector != null)
            {
                ReadRodInput(out Vector3 newCenter, out Vector3 newAxis);
                GetRodEndpoints(newCenter, newAxis, out Vector3 sNew, out Vector3 eNew);

                // Sub-step the delta so each sub-sweep travels < L (no skipped layers / no giant swath). §4.
                Vector3 s0 = tool.S, e0 = tool.E;
                int n = RodMath.SubStepCount(s0, e0, sNew, eNew, Lrt);
                var dimsVI = new Vector3Int(dimsRt.x, dimsRt.y, dimsRt.z);
                var originV = new Vector3(originRt.x, originRt.y, originRt.z);
                for (int i = 1; i <= n; i++)
                {
                    float ti = (float)i / n;
                    tool.Advance(Vector3.Lerp(s0, sNew, ti), Vector3.Lerp(e0, eNew, ti));
                    cutDetector.Dispatch(tool, rb, dimsVI, Lrt, originV);   // SAME origin as Build
                }
                rodCenter = newCenter; rodAxis = newAxis;
                if (rodLine != null) { rodLine.SetPosition(0, sNew); rodLine.SetPosition(1, eNew); }
            }

            // Rotation frame (cut walls follow rotation) + DC rebuild.
            if (cutActive) solver.ComputeRotations(rb);
            dc.Build(rb, dimsRt, Lrt, qefIters, cutFPInterp);

            mat.SetFloat("_DebugTag", 0f);
            Graphics.DrawProceduralIndirect(mat, renderBounds, MeshTopology.Triangles, rb.IndirectArgs, argsOffset: 0);
        }

        // ── Rod helpers ────────────────────────────────────────────────────────────────────
        void GetRodEndpoints(Vector3 center, Vector3 axis, out Vector3 s, out Vector3 e)
            => RodMath.Endpoints(center, axis, rodLength, out s, out e);

        void ReadRodInput(out Vector3 newCenter, out Vector3 newAxis)
        {
            float dt = Time.deltaTime;
            Vector3 move = Vector3.zero;
            if (Input.GetKey(KeyCode.A)) move.x -= 1f;
            if (Input.GetKey(KeyCode.D)) move.x += 1f;
            if (Input.GetKey(KeyCode.S)) move.z -= 1f;
            if (Input.GetKey(KeyCode.W)) move.z += 1f;
            if (Input.GetKey(KeyCode.Q)) move.y -= 1f;
            if (Input.GetKey(KeyCode.E)) move.y += 1f;
            newCenter = rodCenter + move * (moveSpeed * dt);

            float yaw = 0f, pitch = 0f;
            if (Input.GetKey(KeyCode.LeftArrow))  yaw   -= 1f;
            if (Input.GetKey(KeyCode.RightArrow)) yaw   += 1f;
            if (Input.GetKey(KeyCode.UpArrow))    pitch += 1f;
            if (Input.GetKey(KeyCode.DownArrow))  pitch -= 1f;
            if (Mathf.Abs(yaw) > 0f || Mathf.Abs(pitch) > 0f)
            {
                Quaternion rot = Quaternion.Euler(pitch * rotSpeed * dt, yaw * rotSpeed * dt, 0f);
                newAxis = (rot * rodAxis).normalized;
            }
            else newAxis = rodAxis;
        }

        void SetupRodLine(Vector3 s, Vector3 e)
        {
            var go = new GameObject("LiverRodGizmo");
            go.transform.SetParent(transform, false);
            rodLine = go.AddComponent<LineRenderer>();
            rodLine.positionCount = 2;
            rodLine.SetPosition(0, s);
            rodLine.SetPosition(1, e);
            rodLine.widthMultiplier = Mathf.Max(0.02f, rodThicknessDOverL * Lrt);
            rodLine.useWorldSpace = true;
            var sh = Shader.Find("Unlit/Color") ?? Shader.Find("Sprites/Default");
            if (sh != null) { rodLine.material = new Material(sh); rodLine.material.color = Color.cyan; }
        }

        void OnDestroy()
        {
            rb?.Dispose();
            if (mat != null) Destroy(mat);
        }
    }
}
