// LiverSceneSetup.cs — liver-import pipeline, design §5 (GPU asset acquisition).
// Mirrors DemoSceneSetup but targets LiverCutManager. Wires the three .compute assets (Inspector
// drag-drop — ComputeShaders cannot be Shader.Find'd) + the ReconSurface shader (Shader.Find), creates
// the manager + an orbit camera framed on the liver bbox.
//
// Usage: attach to a GameObject, drag Recon.compute / Physics.compute / Cutting.compute into the fields,
// ensure liver3_refined_1.msh is under Assets/StreamingAssets, press Play.

using UnityEngine;
using Unity.Mathematics;
using System.IO;
using ReconGridDC.Preprocess;

namespace ReconGridDC.Demo
{
    public sealed class LiverSceneSetup : MonoBehaviour
    {
        [Header("Required: drag Recon.compute here")]
        public ComputeShader reconShader;
        [Header("Required: drag Physics.compute here")]
        public ComputeShader physicsShader;
        [Header("Required for cutting: drag Cutting.compute here")]
        public ComputeShader cuttingShader;

        [Header("Liver model (under StreamingAssets)")]
        public string mshFileName = "liver3_refined_1.msh";

        [Header("Grid / physics overrides (forwarded) — paper §3.4 defaults")]
        public int   targetLongAxisVoxels = 48;
        public int   marginVoxels         = 2;
        public int   smoothIters          = 3;       // Fix 2 Taubin smoothing passes
        public float ks         = 7.5e4f;   // paper §3.4 (was 800 → liver stretched)
        public float kb         = 2e4f;     // paper §3.4
        public float cs         = 0.92f;    // paper §3.4 (raise if it rings — R5)
        public float cb         = 0.9f;     // paper §3.4
        public float massScale  = 1f;       // R5 settling knob
        public float gravity    = -9.81f;
        public float anchorBand = 1.5f;
        public Vector3 anchorAxis = new Vector3(1, 0, 0);

        void Awake() => CreateLiverScene();

        public void CreateLiverScene()
        {
            Shader surf = Shader.Find("ReconGridDC/ReconSurface");
            if (surf == null)
                Debug.LogError("[LiverSceneSetup] shader 'ReconGridDC/ReconSurface' not found.");
            if (reconShader == null)
                Debug.LogError("[LiverSceneSetup] reconShader (Recon.compute) not assigned (Inspector).");
            if (physicsShader == null)
                Debug.LogError("[LiverSceneSetup] physicsShader (Physics.compute) not assigned (Inspector).");
            if (cuttingShader == null)
                Debug.LogWarning("[LiverSceneSetup] cuttingShader (Cutting.compute) not assigned — liver will simulate but NOT cut.");

            // Manager.
            var go  = new GameObject("Liver_Manager");
            var mgr = go.AddComponent<LiverCutManager>();
            mgr.recon         = reconShader;
            mgr.physics       = physicsShader;
            mgr.cutting       = cuttingShader;
            mgr.surfaceShader = surf;
            mgr.mshFileName          = mshFileName;
            mgr.targetLongAxisVoxels = targetLongAxisVoxels;
            mgr.marginVoxels         = marginVoxels;
            mgr.smoothIters = smoothIters;
            mgr.ks         = ks;
            mgr.kb         = kb;
            mgr.cs_damp    = cs;
            mgr.cb         = cb;
            mgr.massScale  = massScale;
            mgr.gravity    = gravity;
            mgr.anchorBand = anchorBand;
            mgr.anchorAxis = anchorAxis;

            // Camera framed on the liver bbox (load once for accurate framing — editor-time cost only).
            float3 center = new float3(0, 0, 0);
            float  radius = 10f;
            string path = Path.Combine(Application.streamingAssetsPath, mshFileName);
            if (File.Exists(path))
            {
                MshMesh m = MshLoader.LoadMsh(path);
                MshLoader.Bounds(m, out float3 bmin, out float3 bmax);
                center = 0.5f * (bmin + bmax);
                radius = math.length(bmax - bmin) * 0.5f;
            }
            else Debug.LogError($"[LiverSceneSetup] liver asset not found: {path}");

            Camera cam = Camera.main;
            if (cam == null)
            {
                var camGo = new GameObject("Main Camera");
                camGo.tag = "MainCamera";
                cam = camGo.AddComponent<Camera>();
                camGo.AddComponent<AudioListener>();
            }
            float dist = radius * 3f;
            cam.transform.position = new Vector3(center.x, center.y, center.z - dist);
            cam.transform.LookAt(new Vector3(center.x, center.y, center.z));
            cam.backgroundColor = new Color(0.1f, 0.1f, 0.15f);
            cam.clearFlags      = CameraClearFlags.SolidColor;
            cam.farClipPlane    = Mathf.Max(cam.farClipPlane, dist + radius * 4f);

            var orbit = cam.gameObject.GetComponent<OrbitCamera>();
            if (orbit == null) orbit = cam.gameObject.AddComponent<OrbitCamera>();
            orbit.target   = new Vector3(center.x, center.y, center.z);
            orbit.distance = dist;
        }
    }
}
