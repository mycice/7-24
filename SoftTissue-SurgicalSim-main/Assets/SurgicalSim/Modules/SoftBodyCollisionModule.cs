using SurgicalSim.Core;
using SurgicalSim.Physics;
using UnityEngine;

namespace SurgicalSim.Modules
{
    [DisallowMultipleComponent]
    public sealed class SoftBodyCollisionModule : MonoBehaviour
    {
        [Header("地面碰撞")]
        [InspectorName("地面高度")]
        public float groundY = 0f;

        [InspectorName("悬挂模式地面高度")]
        public float hangGroundY = -3f;

        [InspectorName("创建地面平面")]
        public bool createGroundPlane = true;

        [InspectorName("地面颜色")]
        public Color groundColor = new Color(0.3f, 0.7f, 0.3f, 0.8f);

        [Header("工具接触")]
        [InspectorName("接触距离")]
        [Range(0.001f, 0.08f)]
        public float toolContactDistance = 0.008f;

        [InspectorName("接触柔度")]
        [Range(0f, 1e-4f)]
        public float toolContactCompliance = 2e-7f;

        [InspectorName("接触迭代次数")]
        [Range(1, 16)]
        public int toolContactIterations = 10;

        [InspectorName("接触耦合次数")]
        [Range(1, 6)]
        public int toolContactCouplingPasses = 5;

        [InspectorName("切向摩擦")]
        [Range(0f, 1f)]
        public float toolContactTangentialFriction = 0.65f;

        [InspectorName("切向阻尼")]
        [Range(0f, 1f)]
        public float toolContactTangentialDamping = 0.5f;

        [InspectorName("启用接触候选裁剪")]
        public bool useToolContactCandidateCulling = true;

        [InspectorName("候选区域扩展")]
        [Range(0f, 0.25f)]
        public float toolContactCandidatePadding = 0.06f;

        [InspectorName("全表面压力三角形上限")]
        public int maxGpuFullSurfacePressureTris = 90000;

        [Header("模块引用")]
        [InspectorName("夹取模块")]
        public SoftBodyGraspingModule graspingModule;

        public float LastToolUploadMs { get; private set; }
        public float LastGripperStepMs { get; private set; }
        public GameObject GroundPlane => _groundPlane;

        TetMeshData _data;
        XPBDSolverGPU _solver;
        GameObject _groundPlane;
        Material _groundMaterial;

        void Awake()
        {
            if (graspingModule == null)
                graspingModule = GetComponent<SoftBodyGraspingModule>();
        }

        public void Initialize(TetMeshData data, XPBDSolverGPU solver)
        {
            _data = data;
            _solver = solver;
            ApplySolverSettings();
            EnsureGroundPlane();
        }

        public void ApplySolverSettings()
        {
            if (_solver == null)
                return;

            _solver.GroundY = groundY;
            _solver.ToolContactDistance = toolContactDistance;
            _solver.ToolContactCompliance = toolContactCompliance;
            _solver.ToolContactIterations = toolContactIterations;
            _solver.ToolContactCouplingPasses = toolContactCouplingPasses;
            _solver.ToolContactTangentialFriction = toolContactTangentialFriction;
            _solver.ToolContactTangentialDamping = toolContactTangentialDamping;
            _solver.ToolContactCandidatePadding = toolContactCandidatePadding;
        }

        public void UseHangingGroundHeight()
        {
            groundY = hangGroundY;
            UpdateGroundTransform();
        }

        public void PrepareContacts()
        {
            LastToolUploadMs = 0f;
            LastGripperStepMs = 0f;
            if (_solver == null || _data == null)
                return;

            _solver.UseToolContactCandidateCulling = useToolContactCandidateCulling;

            if (graspingModule != null && graspingModule.IsActive)
            {
                var gripperWatch = System.Diagnostics.Stopwatch.StartNew();
                graspingModule.PhysicsStep();
                gripperWatch.Stop();
                LastGripperStepMs = (float)gripperWatch.Elapsed.TotalMilliseconds;

                // Keep the tool's current and previous capsules resident every physics step.
                // CCD needs both states even while no control key is held.
                var uploadWatch = System.Diagnostics.Stopwatch.StartNew();
                graspingModule.UploadCollisionToGPU();
                _solver.UpdateToolContactCandidates(_data);
                uploadWatch.Stop();
                LastToolUploadMs = (float)uploadWatch.Elapsed.TotalMilliseconds;
                return;
            }

            _solver.ClearToolCollisionParams();
        }

        void EnsureGroundPlane()
        {
            if (!createGroundPlane)
            {
                if (_groundPlane != null)
                    _groundPlane.SetActive(false);
                return;
            }

            if (_groundPlane == null)
            {
                _groundPlane = GameObject.CreatePrimitive(PrimitiveType.Plane);
                _groundPlane.name = "Ground Plane";
                _groundPlane.transform.localScale = new Vector3(0.5f, 1f, 0.5f);
                var collider = _groundPlane.GetComponent<Collider>();
                if (collider != null)
                    Destroy(collider);

                _groundMaterial = new Material(
                    Shader.Find("Universal Render Pipeline/Lit") ?? Shader.Find("Standard"));
                _groundPlane.GetComponent<MeshRenderer>().material = _groundMaterial;
            }

            _groundPlane.SetActive(true);
            _groundMaterial.color = groundColor;
            UpdateGroundTransform();
        }

        void UpdateGroundTransform()
        {
            if (_groundPlane != null)
                _groundPlane.transform.position = new Vector3(0f, groundY, 0f);
        }

        void OnDestroy()
        {
            if (_groundPlane != null)
                Destroy(_groundPlane);
            if (_groundMaterial != null)
                Destroy(_groundMaterial);
        }
    }
}
