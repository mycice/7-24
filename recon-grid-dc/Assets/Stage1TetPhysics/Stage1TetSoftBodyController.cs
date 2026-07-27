using System;
using System.Collections;
using ReconGridDC.Stage1TetPhysics.Core;
using ReconGridDC.Stage1TetPhysics.Physics;
using UnityEngine;

namespace ReconGridDC.Stage1TetPhysics
{
    [DisallowMultipleComponent]
    [RequireComponent(typeof(TetMeshLoader))]
    [RequireComponent(typeof(TetMeshVisualizer))]
    public sealed class Stage1TetSoftBodyController : MonoBehaviour
    {
        [Header("Stage 1 Switch")]
        [Tooltip("Enables only the isolated tetrahedral gravity and ground test. It does not connect to the cutter, Haply, or gripper.")]
        public bool simulationEnabled = true;

        [Header("Source Mesh")]
        [Tooltip("Relative path below Assets/StreamingAssets.")]
        public string tetMeshJson = "Stage1TetPhysics/liver3_refined_1.json";
        [Range(500f, 2000f)] public float density = 1050f;
        public float meshScale = 1.5f;
        public Vector3 initialOffset = new Vector3(0f, 2.5f, 0f);

        [Header("XPBD Material")]
        [Range(1, 30)] public int numSubSteps = 5;
        [Min(0f)] public float edgeCompliance = 0.00001f;
        [Min(1f)] public float youngsModulus = 3000f;
        [Range(0.01f, 0.499f)] public float poissonsRatio = 0.257f;
        [Range(0f, 0.5f)] public float damping = 0.044f;

        [Header("Gravity And Ground")]
        public bool enableGravity = true;
        public float gravityY = -9.81f;
        public float groundY = -3f;
        public bool showGroundPlane = true;
        public Color groundColor = new Color(0.24f, 0.34f, 0.28f, 1f);

        [Header("Rendering")]
        public ComputeShader xpbdComputeShader;
        public Material surfaceMaterial;
        public Color fallbackSurfaceColor = new Color(0.62f, 0.18f, 0.10f, 1f);
        public bool useAsyncPositionReadback = true;
        public bool pausePhysics;

        public bool IsReady => _ready;
        public TetMeshData MeshData => _data;
        public TetMeshVisualizer Visualizer => _visualizer;
        public XPBDSolverGPU Solver => _solver;

        // Optional modules prepare GPU-side contacts immediately before each solver step.
        public event Action<TetMeshData, XPBDSolverGPU, TetMeshVisualizer> BeforeSolverStep;
        // Consumers that require the current CPU particle positions, such as the Stage 3
        // Tet-to-Grid bridge, run after a completed solver step and readback.
        public event Action<TetMeshData, XPBDSolverGPU, TetMeshVisualizer> AfterSolverStep;

        TetMeshLoader _loader;
        TetMeshVisualizer _visualizer;
        XPBDSolverGPU _solver;
        TetMeshData _data;
        GameObject _ground;
        Material _runtimeGroundMaterial;
        Material _runtimeSurfaceMaterial;
        bool _ready;

        void Awake()
        {
            _loader = GetComponent<TetMeshLoader>();
            _visualizer = GetComponent<TetMeshVisualizer>();
            if (GetComponent<CudaOrganContextBridge>() == null)
                gameObject.AddComponent<CudaOrganContextBridge>();
            ConfigureLoader();
        }

        void Start()
        {
            if (xpbdComputeShader == null)
            {
                Debug.LogError("[Stage1TetPhysics] XPBDSolver.compute is not assigned.", this);
                return;
            }

            _loader.OnMeshLoaded += InitializeSimulation;
            if (_loader.IsLoaded)
                InitializeSimulation(_loader.MeshData);
            else
                StartCoroutine(LoadAfterComponentInitialization());
        }

        void OnDestroy()
        {
            if (_loader != null)
                _loader.OnMeshLoaded -= InitializeSimulation;
            _solver?.Dispose();
            if (_ground != null)
                Destroy(_ground);
            if (_runtimeGroundMaterial != null)
                Destroy(_runtimeGroundMaterial);
            if (_runtimeSurfaceMaterial != null)
                Destroy(_runtimeSurfaceMaterial);
        }

        void FixedUpdate()
        {
            if (!_ready || !simulationEnabled || pausePhysics || _solver == null)
                return;

            _solver.NumSubSteps = Mathf.Max(1, numSubSteps);
            _solver.EdgeCompliance = Mathf.Max(0f, edgeCompliance);
            _solver.Damping = Mathf.Clamp01(damping);
            _solver.Gravity = enableGravity ? new Vector3(0f, gravityY, 0f) : Vector3.zero;
            _solver.GroundY = groundY;
            BeforeSolverStep?.Invoke(_data, _solver, _visualizer);
            _solver.Step(Time.fixedDeltaTime);

            if (useAsyncPositionReadback)
                _solver.RequestPositionReadback(_data);
            else
                _solver.ReadbackPositions(_data);

            _visualizer.Refresh();
            AfterSolverStep?.Invoke(_data, _solver, _visualizer);
            UpdateGroundPlane();
        }

        void ConfigureLoader()
        {
            _loader.autoLoad = false;
            _loader.jsonFileName = tetMeshJson;
            _loader.massDensity = density;
            _loader.meshScale = meshScale;
        }

        IEnumerator LoadAfterComponentInitialization()
        {
            // TetMeshVisualizer subscribes in Start. Delay loading one frame so this
            // controller does not depend on Unity's unspecified Start order.
            yield return null;
            yield return StartCoroutine(_loader.LoadJsonAsync(tetMeshJson));
        }

        void InitializeSimulation(TetMeshData data)
        {
            if (_ready || data == null)
                return;

            _data = data;
            ApplyInitialOffset(_data, initialOffset);
            ConfigureVisualizer();

            _solver = new XPBDSolverGPU(xpbdComputeShader)
            {
                NumSubSteps = Mathf.Max(1, numSubSteps),
                EdgeCompliance = Mathf.Max(0f, edgeCompliance),
                YoungsModulus = Mathf.Max(1f, youngsModulus),
                PoissonsRatio = Mathf.Clamp(poissonsRatio, 0.01f, 0.499f),
                Damping = Mathf.Clamp01(damping),
                Density = density,
                Gravity = enableGravity ? new Vector3(0f, gravityY, 0f) : Vector3.zero,
                GroundY = groundY,
                ToolContactEnabled = false
            };
            _solver.Init(_data);
            EnsureGroundPlane();
            _ready = true;

            Debug.Log(
                "[Stage1TetPhysics] Ready. Gravity and solver ground are active; " +
                "Haply, cutter, tool contact, collision modules, and grasping modules are not connected.", this);
        }

        void ConfigureVisualizer()
        {
            _visualizer.meshLoader = _loader;
            if (surfaceMaterial != null)
            {
                _visualizer.surfaceMaterial = surfaceMaterial;
                return;
            }

            Shader shader = Shader.Find("Standard");
            if (shader == null)
                shader = Shader.Find("Universal Render Pipeline/Lit");
            if (shader == null)
                return;

            _runtimeSurfaceMaterial = new Material(shader)
            {
                name = "Stage1_TetLiver_RuntimeMaterial",
                color = fallbackSurfaceColor
            };
            _visualizer.surfaceMaterial = _runtimeSurfaceMaterial;
        }

        static void ApplyInitialOffset(TetMeshData data, Vector3 offset)
        {
            for (int i = 0; i < data.NumParticles; i++)
            {
                data.Positions[i] += offset;
                data.RestPositions[i] += offset;
                data.PrevPositions[i] += offset;
            }
        }

        void EnsureGroundPlane()
        {
            if (!showGroundPlane)
                return;

            _ground = GameObject.CreatePrimitive(PrimitiveType.Plane);
            _ground.name = "Stage1 XPBD Ground (visual only)";
            Collider collider = _ground.GetComponent<Collider>();
            if (collider != null)
                Destroy(collider);

            Shader shader = Shader.Find("Standard");
            if (shader == null)
                shader = Shader.Find("Universal Render Pipeline/Lit");
            if (shader != null)
            {
                _runtimeGroundMaterial = new Material(shader)
                {
                    name = "Stage1_XPBD_Ground_RuntimeMaterial",
                    color = groundColor
                };
                _ground.GetComponent<MeshRenderer>().sharedMaterial = _runtimeGroundMaterial;
            }
            UpdateGroundPlane();
        }

        void UpdateGroundPlane()
        {
            if (_ground == null)
                return;
            _ground.SetActive(showGroundPlane);
            _ground.transform.position = new Vector3(0f, groundY, 0f);
        }

        void OnValidate()
        {
            if (_loader != null)
                ConfigureLoader();
            UpdateGroundPlane();
        }
    }
}
