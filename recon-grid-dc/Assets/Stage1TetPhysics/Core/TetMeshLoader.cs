using System;
using System.Collections;
using System.IO;
using UnityEngine;

namespace ReconGridDC.Stage1TetPhysics.Core
{
    [DisallowMultipleComponent]
    public sealed class TetMeshLoader : MonoBehaviour
    {
        [Header("Tet Mesh")]
        [Tooltip("Path relative to Assets/StreamingAssets.")]
        public string jsonFileName = "Stage1TetPhysics/liver3_refined_1.json";

        [Tooltip("Automatically load the configured tetrahedral mesh at Start.")]
        public bool autoLoad = true;

        [Header("Physical Setup")]
        [Tooltip("Soft-body density in kg/m^3. Liver reference value: 1050.")]
        [Range(500f, 2000f)]
        public float massDensity = 1050f;

        [Tooltip("Uniform scale applied to mesh vertices before simulation.")]
        public float meshScale = 1f;

        public TetMeshData MeshData { get; private set; }
        public bool IsLoaded { get; private set; }
        public event Action<TetMeshData> OnMeshLoaded;

        void Start()
        {
            if (autoLoad)
                StartCoroutine(LoadJsonAsync(jsonFileName));
        }

        public IEnumerator LoadJsonAsync(string fileName)
        {
            string path = Path.Combine(Application.streamingAssetsPath, fileName);
            Debug.Log($"[TetMeshLoader] Loading JSON: {path}");

            if (!File.Exists(path))
            {
                Debug.LogError($"[TetMeshLoader] File not found: {path}\n" +
                               "Place the JSON file below Assets/StreamingAssets.", this);
                yield break;
            }

            string json = null;
            Exception readException = null;
            bool readDone = false;
            System.Threading.ThreadPool.QueueUserWorkItem(_ =>
            {
                try
                {
                    json = File.ReadAllText(path);
                }
                catch (Exception exception)
                {
                    readException = exception;
                }
                finally
                {
                    readDone = true;
                }
            });

            while (!readDone)
                yield return null;

            if (readException != null)
            {
                Debug.LogError($"[TetMeshLoader] Failed to read JSON: {readException.Message}", this);
                yield break;
            }

            TetMeshJson jsonData;
            try
            {
                jsonData = JsonUtility.FromJson<TetMeshJson>(json);
            }
            catch (Exception exception)
            {
                Debug.LogError($"[TetMeshLoader] Failed to parse JSON: {exception.Message}", this);
                yield break;
            }

            if (!ValidateJson(jsonData))
                yield break;

            if (!Mathf.Approximately(meshScale, 1f))
                ScaleVertices(jsonData.verts, meshScale);

            MeshData = new TetMeshData();
            MeshData.InitFromJson(jsonData, massDensity);
            IsLoaded = true;

            Debug.Log($"[TetMeshLoader] Loaded {MeshData.NumParticles} particles, " +
                      $"{MeshData.NumTets} tetrahedra, and {MeshData.NumSurfaceTris} surface triangles.", this);
            OnMeshLoaded?.Invoke(MeshData);
        }

        bool ValidateJson(TetMeshJson data)
        {
            if (data == null || data.verts == null || data.verts.Length == 0 ||
                data.tetIds == null || data.tetIds.Length == 0 ||
                data.tetSurfaceTriIds == null || data.tetSurfaceTriIds.Length == 0)
            {
                Debug.LogError("[TetMeshLoader] JSON mesh data is incomplete.", this);
                return false;
            }

            if (data.verts.Length % 3 != 0 || data.tetIds.Length % 4 != 0 ||
                data.tetSurfaceTriIds.Length % 3 != 0)
            {
                Debug.LogError("[TetMeshLoader] JSON mesh array lengths are invalid.", this);
                return false;
            }

            int particleCount = data.verts.Length / 3;
            for (int i = 0; i < data.tetIds.Length; i++)
            {
                if (data.tetIds[i] < 0 || data.tetIds[i] >= particleCount)
                {
                    Debug.LogError("[TetMeshLoader] A tetrahedron index is outside the vertex range.", this);
                    return false;
                }
            }

            for (int i = 0; i < data.tetSurfaceTriIds.Length; i++)
            {
                if (data.tetSurfaceTriIds[i] < 0 || data.tetSurfaceTriIds[i] >= particleCount)
                {
                    Debug.LogError("[TetMeshLoader] A surface-triangle index is outside the vertex range.", this);
                    return false;
                }
            }

            return true;
        }

        static void ScaleVertices(float[] vertices, float scale)
        {
            for (int i = 0; i < vertices.Length; i++)
                vertices[i] *= scale;
        }
    }
}
