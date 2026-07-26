# Real-Time GPU Soft-Tissue Collision and Grasping Simulation

This Unity project simulates deformable soft tissue, including collision with a surgical gripper and particle-based tissue grasping. The implementation uses GPU-accelerated Extended Position-Based Dynamics (XPBD) and tetrahedral volume meshes.

## Features

- GPU XPBD soft-body simulation for tetrahedral tissue meshes.
- Gravity, damping, and ground collision.
- Capsule-based gripper-to-tissue collision.
- Surface-contact candidate culling for efficient gripper interaction.
- Gripper closing, particle capture, release, and inverse-mass updates.
- Real-time liver surface visualization with Unity compute shaders.

## Main Components

- `Assets/SurgicalSim/SoftBody.cs`: connects mesh loading, the GPU solver, visualization, collision, and grasping modules.
- `Assets/SurgicalSim/Physics/XPBDSolverGPU.cs`: dispatches XPBD integration, constraints, ground collision, and gripper contact kernels.
- `Assets/SurgicalSim/Modules/SoftBodyCollisionModule.cs`: coordinates collision processing and gripper updates.
- `Assets/SurgicalSim/Grasping/GripperTool.cs`: handles gripper motion, collision capsule upload, tissue capture, and release.
- `Assets/SurgicalSim/Shaders/XPBDSolver.compute`: GPU physics kernels.
- `Assets/StreamingAssets/liver3-HD.msh`: tetrahedral liver mesh data.

## Getting Started

1. Open the project with Unity 2021.3 or later.
2. Open `Assets/Scenes/SampleScene.unity`.
3. Press Play and use the gripper controls configured in the scene to contact, close on, move, and release the liver tissue.

## License

MIT License
