#if UNITY_EDITOR
using ReconGridDC.Stage1TetPhysics;
using ReconGridDC.Stage1TetPhysics.Core;
using ReconGridDC.Stage2Grasping.Grasping;
using ReconGridDC.Stage2Grasping.Modules;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.SceneManagement;

namespace ReconGridDC.Stage2Grasping.Editor
{
    public static class Stage2GraspingSceneBuilder
    {
        const string ScenePath = "Assets/Stage2Grasping/Stage2GraspingTest.unity";
        const string ComputePath = "Assets/Stage1TetPhysics/Shaders/XPBDSolver.compute";

        [MenuItem("Tools/ReconGridDC/Stage 2/Create Or Replace Grasping Test Scene")]
        public static void CreateOrReplaceTestScene()
        {
            if (!EditorSceneManager.SaveCurrentModifiedScenesIfUserWantsTo())
                return;

            Scene scene = EditorSceneManager.NewScene(NewSceneSetup.EmptyScene, NewSceneMode.Single);
            CreateCamera();
            CreateLight();
            CreateSoftBody();
            CreateGripper();

            EditorSceneManager.SaveScene(scene, ScenePath);
            Debug.Log("[Stage2Grasping] Test scene created: " + ScenePath);
        }

        static void CreateSoftBody()
        {
            GameObject body = new GameObject("Stage2 Tet Soft Body");
            body.AddComponent<MeshFilter>();
            body.AddComponent<MeshRenderer>();
            TetMeshLoader loader = body.AddComponent<TetMeshLoader>();
            TetMeshVisualizer visualizer = body.AddComponent<TetMeshVisualizer>();
            Stage1TetSoftBodyController controller = body.AddComponent<Stage1TetSoftBodyController>();
            SoftBodyGraspingModule grasping = body.AddComponent<SoftBodyGraspingModule>();
            SoftBodyCollisionModule collision = body.AddComponent<SoftBodyCollisionModule>();
            Stage2TetGraspingBridge bridge = body.AddComponent<Stage2TetGraspingBridge>();

            loader.autoLoad = false;
            loader.jsonFileName = controller.tetMeshJson;
            loader.massDensity = controller.density;
            loader.meshScale = controller.meshScale;
            visualizer.meshLoader = loader;
            controller.xpbdComputeShader = AssetDatabase.LoadAssetAtPath<ComputeShader>(ComputePath);
            grasping.enableGrasping = true;
            collision.graspingModule = grasping;
            bridge.softBodyController = controller;
            bridge.graspingModule = grasping;
            bridge.collisionModule = collision;
        }

        static void CreateGripper()
        {
            GameObject root = new GameObject("Stage2 Gripper Tool");
            root.transform.position = new Vector3(0.15f, 0.5f, 0.7f);
            GripperTool tool = root.AddComponent<GripperTool>();

            tool.upperJawObject = CreateMeshPart("Upper Jaw", root.transform);
            tool.lowerJawObject = CreateMeshPart("Lower Jaw", root.transform);
            tool.shaftObject = CreateMeshPart("Shaft", root.transform);

            SoftBodyGraspingModule grasping = Object.FindObjectOfType<SoftBodyGraspingModule>();
            grasping.gripperTool = tool;
        }

        static GameObject CreateMeshPart(string name, Transform parent)
        {
            GameObject part = new GameObject(name);
            part.transform.SetParent(parent, false);
            part.AddComponent<MeshFilter>();
            part.AddComponent<MeshRenderer>();
            return part;
        }

        static void CreateCamera()
        {
            GameObject cameraObject = new GameObject("Main Camera");
            cameraObject.tag = "MainCamera";
            Camera camera = cameraObject.AddComponent<Camera>();
            camera.clearFlags = CameraClearFlags.SolidColor;
            camera.backgroundColor = new Color(0.075f, 0.09f, 0.11f);
            cameraObject.transform.position = new Vector3(0f, 1.1f, -8.5f);
            cameraObject.transform.LookAt(new Vector3(-0.25f, -1.2f, 0.3f));
            camera.fieldOfView = 48f;
            cameraObject.AddComponent<Stage1TestCameraController>();
        }

        static void CreateLight()
        {
            GameObject lightObject = new GameObject("Directional Light");
            Light light = lightObject.AddComponent<Light>();
            light.type = LightType.Directional;
            light.intensity = 1.25f;
            lightObject.transform.rotation = Quaternion.Euler(45f, -25f, 0f);
        }
    }
}
#endif
