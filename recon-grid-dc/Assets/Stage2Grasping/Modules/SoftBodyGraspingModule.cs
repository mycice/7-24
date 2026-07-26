using ReconGridDC.Stage1TetPhysics.Core;
using ReconGridDC.Stage1TetPhysics.Physics;
using ReconGridDC.Stage2Grasping.Grasping;
using UnityEngine;

namespace ReconGridDC.Stage2Grasping.Modules
{
    [DisallowMultipleComponent]
    public sealed class SoftBodyGraspingModule : MonoBehaviour
    {
        [Header("Stage 2 Grasping Switch")]
        [Tooltip("Enables only the isolated Stage 2 gripper. It does not connect to Haply or the cutting system.")]
        public bool enableGrasping;

        [Header("Scene Reference")]
        public GripperTool gripperTool;

        public bool IsActive => enableGrasping && gripperTool != null && gripperTool.enabled;

        public void Initialize(TetMeshData data, XPBDSolverGPU solver, TetMeshVisualizer visualizer)
        {
            if (gripperTool == null)
                gripperTool = GetComponentInChildren<GripperTool>(true);

            if (gripperTool == null)
            {
                if (enableGrasping)
                    Debug.LogError("[Stage2Grasping] GripperTool is not assigned.", this);
                return;
            }

            gripperTool.enabled = enableGrasping;
            if (enableGrasping)
                gripperTool.Init(data, solver, visualizer);
        }

        public void PhysicsStep()
        {
            if (IsActive)
                gripperTool.PhysicsStep();
        }

        public void UploadCollisionToGPU()
        {
            if (IsActive)
                gripperTool.UploadToolCollisionToGPU();
        }
    }
}
