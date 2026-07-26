using ReconGridDC.Stage1TetPhysics;
using ReconGridDC.Stage1TetPhysics.Core;
using ReconGridDC.Stage1TetPhysics.Physics;
using ReconGridDC.Stage2Grasping.Modules;
using UnityEngine;

namespace ReconGridDC.Stage2Grasping
{
    [DisallowMultipleComponent]
    [RequireComponent(typeof(Stage1TetSoftBodyController))]
    [RequireComponent(typeof(SoftBodyCollisionModule))]
    [RequireComponent(typeof(SoftBodyGraspingModule))]
    public sealed class Stage2TetGraspingBridge : MonoBehaviour
    {
        [Header("Stage 2 Switch")]
        [Tooltip("Enables only the isolated gripper-to-tetrahedral-organ prototype.")]
        public bool stage2Enabled = true;

        public Stage1TetSoftBodyController softBodyController;
        public SoftBodyCollisionModule collisionModule;
        public SoftBodyGraspingModule graspingModule;

        bool _initialized;

        void Awake()
        {
            softBodyController = softBodyController ?? GetComponent<Stage1TetSoftBodyController>();
            collisionModule = collisionModule ?? GetComponent<SoftBodyCollisionModule>();
            graspingModule = graspingModule ?? GetComponent<SoftBodyGraspingModule>();
        }

        void Update()
        {
            if (!_initialized && softBodyController != null && softBodyController.IsReady)
                Initialize(softBodyController.MeshData, softBodyController.Solver, softBodyController.Visualizer);
        }

        void Initialize(TetMeshData data, XPBDSolverGPU solver, TetMeshVisualizer visualizer)
        {
            collisionModule.graspingModule = graspingModule;
            collisionModule.Initialize(data, solver);
            graspingModule.Initialize(data, solver, visualizer);
            softBodyController.BeforeSolverStep += PrepareSolverStep;
            _initialized = true;
        }

        void PrepareSolverStep(TetMeshData data, XPBDSolverGPU solver, TetMeshVisualizer visualizer)
        {
            if (!_initialized || !stage2Enabled)
            {
                solver.ToolContactEnabled = false;
                solver.ClearToolCollisionParams();
                return;
            }

            collisionModule.ApplySolverSettings();
            collisionModule.PrepareContacts();
        }

        void OnDestroy()
        {
            if (softBodyController != null)
                softBodyController.BeforeSolverStep -= PrepareSolverStep;
        }
    }
}
