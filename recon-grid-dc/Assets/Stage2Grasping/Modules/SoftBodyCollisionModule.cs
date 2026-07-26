using ReconGridDC.Stage1TetPhysics.Core;
using ReconGridDC.Stage1TetPhysics.Physics;
using UnityEngine;

namespace ReconGridDC.Stage2Grasping.Modules
{
    [DisallowMultipleComponent]
    public sealed class SoftBodyCollisionModule : MonoBehaviour
    {
        [Header("GPU Tool Contact")]
        [Range(0.001f, 0.08f)] public float toolContactDistance = 0.008f;
        [Range(0f, 0.0001f)] public float toolContactCompliance = 0.0000002f;
        [Range(1, 16)] public int toolContactIterations = 8;
        [Range(1, 6)] public int toolContactCouplingPasses = 3;
        [Range(0f, 1f)] public float toolContactTangentialFriction = 0.65f;
        [Range(0f, 1f)] public float toolContactTangentialDamping = 0.5f;
        public bool useToolContactCandidateCulling = true;
        [Range(0f, 0.25f)] public float toolContactCandidatePadding = 0.06f;
        [Tooltip("Keep solving already-overlapping candidates after keyboard motion stops. This prevents the gripper from remaining inside the physical organ until elasticity slowly pulls it back out.")]
        public bool keepContactActiveWhenIdle = true;

        [Header("Module Reference")]
        public SoftBodyGraspingModule graspingModule;

        public float LastToolUploadMs { get; private set; }
        public float LastGripperStepMs { get; private set; }
        public int ActiveCandidateTriangles { get; private set; }
        public bool IsSolvingToolContact { get; private set; }

        TetMeshData _data;
        XPBDSolverGPU _solver;

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
        }

        public void ApplySolverSettings()
        {
            if (_solver == null)
                return;

            _solver.ToolContactDistance = toolContactDistance;
            _solver.ToolContactCompliance = toolContactCompliance;
            _solver.ToolContactIterations = toolContactIterations;
            _solver.ToolContactCouplingPasses = toolContactCouplingPasses;
            _solver.ToolContactTangentialFriction = toolContactTangentialFriction;
            _solver.ToolContactTangentialDamping = toolContactTangentialDamping;
            _solver.ToolContactCandidatePadding = toolContactCandidatePadding;
        }

        public void PrepareContacts()
        {
            LastToolUploadMs = 0f;
            LastGripperStepMs = 0f;
            ActiveCandidateTriangles = 0;
            IsSolvingToolContact = false;
            if (_solver == null || _data == null)
                return;

            _solver.UseToolContactCandidateCulling = useToolContactCandidateCulling;
            if (graspingModule == null || !graspingModule.IsActive)
            {
                _solver.ToolContactEnabled = false;
                _solver.ClearToolCollisionParams();
                return;
            }

            var gripperWatch = System.Diagnostics.Stopwatch.StartNew();
            graspingModule.PhysicsStep();
            gripperWatch.Stop();
            LastGripperStepMs = (float)gripperWatch.Elapsed.TotalMilliseconds;

            var uploadWatch = System.Diagnostics.Stopwatch.StartNew();
            graspingModule.UploadCollisionToGPU();
            _solver.UpdateToolContactCandidates(_data);
            uploadWatch.Stop();
            LastToolUploadMs = (float)uploadWatch.Elapsed.TotalMilliseconds;

            ActiveCandidateTriangles = _solver.ActiveToolContactCandidateCount;
            // A completed grasp pins selected particles directly to the jaw. Running the
            // surface contact solver at the same time would push that same region back out
            // of the jaws, fighting the grasp constraint and injecting large corrections.
            bool hasHardGrasp = graspingModule.gripperTool.IsGrasping;
            IsSolvingToolContact = !hasHardGrasp &&
                                  (keepContactActiveWhenIdle ||
                                   graspingModule.gripperTool.WantsToolContact);
            IsSolvingToolContact &= !useToolContactCandidateCulling ||
                                    _solver.HasToolContactCandidates;
            _solver.ToolContactEnabled = IsSolvingToolContact;
        }

        void OnDisable()
        {
            if (_solver != null)
            {
                _solver.ToolContactEnabled = false;
                _solver.ClearToolCollisionParams();
            }
        }
    }
}
