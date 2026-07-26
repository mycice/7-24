#if UNITY_EDITOR
using ReconGridDC.Cuda;
using ReconGridDC.Stage1TetPhysics;
using ReconGridDC.Stage1TetPhysics.Core;
using ReconGridDC.Stage2Grasping;
using ReconGridDC.Stage2Grasping.Grasping;
using ReconGridDC.Stage2Grasping.Modules;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.SceneManagement;

namespace ReconGridDC.Stage3Coupling.Editor
{
    public static class Stage3CouplingSceneBuilder
    {
        const string ScenePath = "Assets/Stage3Coupling/Stage3CouplingTest.unity";
        const string ComputePath = "Assets/Stage1TetPhysics/Shaders/XPBDSolver.compute";

        [MenuItem("Tools/ReconGridDC/Stage 3/Create Or Replace Coupling Test Scene")]
        public static void CreateOrReplaceTestScene()
        {
            if (!EditorSceneManager.SaveCurrentModifiedScenesIfUserWantsTo()) return;
            Scene scene = EditorSceneManager.NewScene(NewSceneSetup.EmptyScene, NewSceneMode.Single);
            CreateCameraAndLight();
            Stage1TetSoftBodyController tet = CreateTetBody();
            CreateGripper(tet.gameObject);
            LiverCudaManager liver = CreateCudaLiver();
            GameObject coupling = new GameObject("Stage3 Tet-To-Grid Coupling");
            Stage3TetToGridEmbeddingBridge bridge = coupling.AddComponent<Stage3TetToGridEmbeddingBridge>();
            bridge.tetSoftBody = tet;
            bridge.cudaLiver = liver;
            bridge.synchronizationEnabled = true;
            bridge.forceSynchronousTetReadback = true;
            EditorSceneManager.SaveScene(scene, ScenePath);
            Debug.Log("[Stage3Coupling] Test scene created: " + ScenePath);
        }

        static Stage1TetSoftBodyController CreateTetBody()
        {
            GameObject body = new GameObject("Stage3 Tet Soft Body (Physics Source)");
            body.AddComponent<MeshFilter>(); body.AddComponent<MeshRenderer>();
            TetMeshLoader loader = body.AddComponent<TetMeshLoader>();
            TetMeshVisualizer visualizer = body.AddComponent<TetMeshVisualizer>();
            Stage1TetSoftBodyController controller = body.AddComponent<Stage1TetSoftBodyController>();
            SoftBodyGraspingModule grasping = body.AddComponent<SoftBodyGraspingModule>();
            SoftBodyCollisionModule collision = body.AddComponent<SoftBodyCollisionModule>();
            Stage2TetGraspingBridge stage2 = body.AddComponent<Stage2TetGraspingBridge>();
            controller.meshScale = 1.5f;
            controller.initialOffset = new Vector3(0f, 2.5f, 0f);
            controller.numSubSteps = 5;
            loader.autoLoad = false; loader.jsonFileName = controller.tetMeshJson;
            loader.massDensity = controller.density; loader.meshScale = controller.meshScale;
            visualizer.meshLoader = loader;
            controller.xpbdComputeShader = AssetDatabase.LoadAssetAtPath<ComputeShader>(ComputePath);
            grasping.enableGrasping = true; collision.graspingModule = grasping;
            collision.toolContactDistance = 0.008f;
            collision.toolContactCandidatePadding = 0.06f;
            stage2.softBodyController = controller; stage2.graspingModule = grasping; stage2.collisionModule = collision;
            return controller;
        }

        static void CreateGripper(GameObject tetBody)
        {
            GameObject root = new GameObject("Stage3 Gripper Tool");
            root.transform.position = new Vector3(0.15f, 0.5f, 0.7f);
            GripperTool tool = root.AddComponent<GripperTool>();
            tool.upperJawObject = CreateMeshPart("Upper Jaw", root.transform);
            tool.lowerJawObject = CreateMeshPart("Lower Jaw", root.transform);
            tool.shaftObject = CreateMeshPart("Shaft", root.transform);
            tetBody.GetComponent<SoftBodyGraspingModule>().gripperTool = tool;
        }

        static GameObject CreateMeshPart(string name, Transform parent)
        {
            GameObject part = new GameObject(name); part.transform.SetParent(parent, false);
            part.AddComponent<MeshFilter>(); part.AddComponent<MeshRenderer>(); return part;
        }

        static LiverCudaManager CreateCudaLiver()
        {
            GameObject go = new GameObject("Stage3 CUDA Grid Liver (Cutting and Rendering)");
            go.AddComponent<MeshFilter>(); go.AddComponent<MeshRenderer>();
            LiverCudaCutter cutter = go.AddComponent<LiverCudaCutter>();
            cutter.enableCutting = true; cutter.enableKeyboardControl = true;
            LiverCudaManager manager = go.AddComponent<LiverCudaManager>();
            manager.setupOrbitCamera = false;
            manager.gravity = 0f;
            manager.createReferenceGround = false;
            return manager;
        }

        static void CreateCameraAndLight()
        {
            GameObject cam = new GameObject("Main Camera") { tag = "MainCamera" };
            Camera camera = cam.AddComponent<Camera>(); camera.clearFlags = CameraClearFlags.SolidColor;
            camera.backgroundColor = new Color(0.075f, 0.09f, 0.11f); cam.transform.position = new Vector3(0f, 1.1f, -18f);
            cam.transform.LookAt(new Vector3(-0.25f, -1.2f, 0.3f));
            cam.AddComponent<Stage1TestCameraController>().enableKeyboardTranslation = false;
            GameObject light = new GameObject("Directional Light"); Light l = light.AddComponent<Light>();
            l.type = LightType.Directional; l.intensity = 1.25f; light.transform.rotation = Quaternion.Euler(45f, -25f, 0f);
        }
    }
}
#endif
