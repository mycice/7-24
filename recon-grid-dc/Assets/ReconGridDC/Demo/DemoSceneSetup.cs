// DemoSceneSetup.cs — Stage 1B-ii Task 2 Step 2
// Editor/runtime helper that creates the ReconGridManager GameObject, assigns:
//   - reconShader    (Recon.compute)    → mgr.recon
//   - physicsShader  (Physics.compute)  → mgr.physics   [NEW: Stage 1B-ii]
//   - surfaceShader  (ReconGridDC/ReconSurface) → mgr.surfaceShader
// and positions a camera.
//
// Usage in the editor:
//   1. Attach this script to any GameObject in the scene (or call CreateDemoScene() from a custom
//      menu item / Awake).
//   2. Drag Recon.compute into the reconShader field in the Inspector.
//   3. Drag Physics.compute into the physicsShader field in the Inspector.
//   4. Enter Play mode — the sphere surface will appear and sag under gravity.
//
// Alternatively, CreateDemoScene() can be called programmatically (e.g. from a custom editor menu).

using UnityEngine;
using Unity.Mathematics;

namespace ReconGridDC.Demo
{
    public sealed class DemoSceneSetup : MonoBehaviour
    {
        [Header("Required: drag Recon.compute here")]
        public ComputeShader reconShader;    // serialized — assigned in Inspector

        [Header("Required: drag Physics.compute here")]
        public ComputeShader physicsShader;  // serialized — assigned in Inspector; used by MassSpringSolver

        [Header("Optional: drag Cutting.compute here (Stage 2C). Leave empty for Stage-1 behavior.")]
        public ComputeShader cuttingShader;  // serialized — assigned in Inspector; enables the cut surface

        [Header("Grid/sphere override (optional)")]
        public int3   dims         = new int3(24, 24, 24);
        public float  L            = 0.25f;
        public int    qefIters     = 20;
        public float3 sphereCenter = new float3(3f, 3f, 3f);
        public float  sphereRadius = 2.0f;

        [Header("Physics (SOFT demo defaults so sag is VISIBLE)")]
        // The paper's ks=7.5e4 is very stiff -> equilibrium sag is sub-pixel (looks static) and the
        // explicit RK45 sits near its stability limit. For a clearly-visible, stable demo we use a
        // much softer spring here. (The solver's own defaults stay paper-faithful at 7.5e4; this only
        // overrides the DEMO instance. Raise ks toward 7.5e4 to approach paper stiffness.)
        public float  ks           = 800f;     // soft -> ~94x more sag than 7.5e4, clearly visible
        public float  csDamp       = 20f;      // higher damping -> clean sag-and-settle (not ringing)
        public float  kb           = 200f;
        public float  cb           = 5f;
        public float  gravity      = -9.81f;
        public bool   pinTopLayer  = true;

        void Awake()
        {
            CreateDemoScene();
        }

        /// <summary>
        /// Creates the ReconGridManager GameObject and a camera if none exists.
        /// Can be called from a custom editor menu or on Awake.
        /// </summary>
        public void CreateDemoScene()
        {
            // ── Surface shader (find by name; must be in the project) ────────────
            Shader surfShader = Shader.Find("ReconGridDC/ReconSurface");
            if (surfShader == null)
                Debug.LogError("[DemoSceneSetup] Could not find shader 'ReconGridDC/ReconSurface'. " +
                               "Ensure ReconSurface.shader is in the project.");

            if (reconShader == null)
                Debug.LogError("[DemoSceneSetup] reconShader (Recon.compute) is not assigned. " +
                               "Assign it in the Inspector.");

            // NEW (Stage 1B-ii): Physics.compute must be assigned for MassSpringSolver.
            // Without it, solver construction will fail at runtime.
            if (physicsShader == null)
                Debug.LogError("[DemoSceneSetup] physicsShader (Physics.compute) is not assigned. " +
                               "Drag Physics.compute into the physicsShader field in the Inspector.");

            // ── ReconGridManager GameObject ──────────────────────────────────────
            var managerGo = new GameObject("ReconGridDC_Manager");
            var mgr = managerGo.AddComponent<ReconGridManager>();

            // Wire serialized fields
            mgr.recon         = reconShader;
            mgr.physics       = physicsShader;  // CRITICAL: MassSpringSolver needs Physics shader, NOT recon
            mgr.cutting       = cuttingShader;  // Stage 2C: may be null (Stage-1 behavior preserved)
            mgr.surfaceShader = surfShader;
            mgr.dims          = dims;
            mgr.L             = L;
            mgr.qefIters      = qefIters;
            mgr.sphereCenter  = sphereCenter;
            mgr.sphereRadius  = sphereRadius;
            // Forward the soft demo physics so the sag is visible + stable (otherwise mgr keeps its
            // paper-stiff ks=7.5e4 default and the sphere only micro-jitters with no visible deformation).
            mgr.ks            = ks;
            mgr.cs_damp       = csDamp;
            mgr.kb            = kb;
            mgr.cb            = cb;
            mgr.gravity       = gravity;
            mgr.pinTopLayer   = pinTopLayer;

            // ── Camera (create one if the scene has none) ────────────────────────
            Camera cam = Camera.main;
            if (cam == null)
            {
                var camGo = new GameObject("Main Camera");
                camGo.tag = "MainCamera";
                cam = camGo.AddComponent<Camera>();
                camGo.AddComponent<AudioListener>();
            }

            // Position camera to frame the sphere: stand back along -z, look at sphere center
            float3 lookAt = sphereCenter;
            float  dist   = sphereRadius * 5f + dims.x * L * 0.5f;
            cam.transform.position = new Vector3(lookAt.x, lookAt.y, lookAt.z - dist);
            cam.transform.LookAt(new Vector3(lookAt.x, lookAt.y, lookAt.z));
            cam.backgroundColor = new Color(0.1f, 0.1f, 0.15f);
            cam.clearFlags      = CameraClearFlags.SolidColor;

            // Mouse-orbit so you can inspect the surface from all sides in Play mode
            // (hold LEFT mouse button to rotate, scroll wheel to zoom).
            var orbit = cam.gameObject.GetComponent<OrbitCamera>();
            if (orbit == null) orbit = cam.gameObject.AddComponent<OrbitCamera>();
            orbit.target   = new Vector3(lookAt.x, lookAt.y, lookAt.z);
            orbit.distance = dist;
        }
    }
}
