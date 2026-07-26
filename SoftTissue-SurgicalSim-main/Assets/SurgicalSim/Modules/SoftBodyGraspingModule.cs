using SurgicalSim.Core;
using SurgicalSim.Grasping;
using SurgicalSim.Physics;
using UnityEngine;

namespace SurgicalSim.Modules
{
    [DisallowMultipleComponent]
    public sealed class SoftBodyGraspingModule : MonoBehaviour
    {
        [Header("夹取开关")]
        [InspectorName("启用夹取")]
        public bool enableGrasping = false;

        [Header("场景引用")]
        [InspectorName("夹爪物体")]
        public GripperTool gripperTool;

        public GripperTool Tool => gripperTool;
        public bool IsActive => enableGrasping && gripperTool != null && gripperTool.enabled;
        public bool IsContactActive => IsActive && gripperTool.WantsToolContact;

        public void Initialize(TetMeshData data, XPBDSolverGPU solver, TetMeshVisualizer visualizer)
        {
            if (gripperTool == null)
                gripperTool = FindObjectOfType<GripperTool>();

            if (gripperTool == null)
            {
                if (enableGrasping)
                    Debug.LogError("[夹取模块] 未指定 GripperTool。请把场景中的 Gripper Tool 拖入夹爪物体。", this);
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

        public void Reinitialize(TetMeshData data, XPBDSolverGPU solver, TetMeshVisualizer visualizer)
        {
            Initialize(data, solver, visualizer);
        }
    }
}
