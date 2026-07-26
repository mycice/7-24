#if UNITY_EDITOR
using ReconGridDC.Stage1TetPhysics.Core;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.SceneManagement;

namespace ReconGridDC.Stage1TetPhysics.Editor
{
    public static class Stage1TetPhysicsSceneBuilder
    {
        const string ScenePath = "Assets/Stage1TetPhysics/Stage1TetPhysicsTest.unity";
        const string ComputePath = "Assets/Stage1TetPhysics/Shaders/XPBDSolver.compute";

        [MenuItem("Tools/ReconGridDC/Stage 1/Create Or Replace Tet Physics Test Scene")]
        public static void CreateOrReplaceTestScene()
        {
            if (!EditorSceneManager.SaveCurrentModifiedScenesIfUserWantsTo())
                return;

            Scene scene = EditorSceneManager.NewScene(NewSceneSetup.EmptyScene, NewSceneMode.Single);
            CreateCamera();
            CreateLight();

            GameObject body = new GameObject("Stage1 Tet Soft Body");
            body.AddComponent<MeshFilter>();
            body.AddComponent<MeshRenderer>();
            TetMeshLoader loader = body.AddComponent<TetMeshLoader>();
            TetMeshVisualizer visualizer = body.AddComponent<TetMeshVisualizer>();
            Stage1TetSoftBodyController controller = body.AddComponent<Stage1TetSoftBodyController>();
            loader.autoLoad = false;
            loader.jsonFileName = controller.tetMeshJson;
            loader.massDensity = controller.density;
            loader.meshScale = controller.meshScale;
            visualizer.meshLoader = loader;
            controller.xpbdComputeShader = AssetDatabase.LoadAssetAtPath<ComputeShader>(ComputePath);

            EditorSceneManager.SaveScene(scene, ScenePath);
            Selection.activeGameObject = body;
            Debug.Log("[Stage1TetPhysics] Test scene created: " + ScenePath);
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
