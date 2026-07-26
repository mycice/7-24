using System.Collections.Generic;
using System.Globalization;
using System.IO;
using UnityEngine;
using UnityEngine.Rendering;

namespace ReconGridDC.Cuda
{
    /// <summary>
    /// Visual model adapter for a LiverCuttingRod target. The CUDA cutter still receives a center line,
    /// while the user sees and drives a real stick model such as StreamingAssets/xiaogun_v1.obj.
    /// </summary>
    [DisallowMultipleComponent]
    [RequireComponent(typeof(LiverCuttingRod))]
    public sealed class LiverCuttingStickModel : MonoBehaviour
    {
        [Header("Rod Target")]
        [SerializeField] LiverCuttingRod cuttingRod;
        [SerializeField] LiverCudaCutter liverCudaCutter;
        [SerializeField] bool findCutterWhenEmpty = true;
        [SerializeField] bool assignToCutterWhenEmpty = true;
        [SerializeField] bool disableCutterShapeDriver = true;

        [Header("Stick Model")]
        [Tooltip("Optional existing child model. Leave empty to load xiaogun_v1.obj from StreamingAssets at runtime.")]
        [SerializeField] Transform stickModel;
        [SerializeField] string streamingAssetsObjFileName = "xiaogun_v1.obj";
        [SerializeField] bool loadObjWhenModelMissing = true;
        [SerializeField] Material stickMaterial;

        [Header("Cut Shape")]
        [SerializeField] bool driveRodShapeFromThisComponent = true;
        [SerializeField] float stickLength = 1.0f;
        [SerializeField] float stickDiameter = 0.10f;
        [SerializeField] bool hideDebugLine = true;

        [Header("Model Alignment")]
        [Tooltip("xiaogun_v1.obj is authored along local +Z. Keep this at (0,0,0) unless your mesh points in another direction.")]
        [SerializeField] Vector3 modelEulerOffset;
        [SerializeField] Vector3 modelLocalOffset;

        Mesh _runtimeMesh;
        Material _runtimeMaterial;
        float _sourceLength = 1f;
        float _sourceDiameter = 0.1f;
        float _sourceCenterZ = 0.5f;

        public Transform StickModel => stickModel;

        public void Refresh()
        {
            ResolveReferences();
            EnsureStickModel();
            ApplyConfiguration();
        }

        void Reset()
        {
            ResolveReferences();
        }

        void Awake()
        {
            Refresh();
        }

        void Start()
        {
            Refresh();
        }

        void Update()
        {
            ApplyConfiguration();
        }

        void OnValidate()
        {
            stickLength = Mathf.Max(0.001f, stickLength);
            stickDiameter = Mathf.Max(0.001f, stickDiameter);

            if (Application.isPlaying)
            {
                Refresh();
            }
        }

        void ResolveReferences()
        {
            if (cuttingRod == null)
            {
                cuttingRod = GetComponent<LiverCuttingRod>();
            }

            if (liverCudaCutter == null && findCutterWhenEmpty)
            {
                liverCudaCutter = FindObjectOfType<LiverCudaCutter>();
            }

            if (liverCudaCutter != null)
            {
                if (assignToCutterWhenEmpty && liverCudaCutter.cuttingRodTarget == null && cuttingRod != null)
                {
                    liverCudaCutter.cuttingRodTarget = cuttingRod;
                }

                if (disableCutterShapeDriver)
                {
                    liverCudaCutter.driveCuttingRodShape = false;
                }
            }
        }

        void EnsureStickModel()
        {
            if (stickModel != null)
            {
                MeasureModelBounds();
                ApplyMaterialIfNeeded(stickModel);
                return;
            }

            if (!loadObjWhenModelMissing)
            {
                return;
            }

            string path = Path.Combine(Application.streamingAssetsPath, streamingAssetsObjFileName);
            if (!File.Exists(path))
            {
                Debug.LogError($"[LiverCuttingStickModel] OBJ not found: {path}");
                return;
            }

            _runtimeMesh = LoadObjMesh(path);
            if (_runtimeMesh == null)
            {
                return;
            }

            var go = new GameObject(Path.GetFileNameWithoutExtension(streamingAssetsObjFileName));
            go.transform.SetParent(transform, false);

            var mf = go.AddComponent<MeshFilter>();
            mf.sharedMesh = _runtimeMesh;

            var mr = go.AddComponent<MeshRenderer>();
            mr.sharedMaterial = ResolveMaterial();

            stickModel = go.transform;
            MeasureModelBounds();
        }

        void ApplyConfiguration()
        {
            if (cuttingRod == null)
            {
                return;
            }

            if (hideDebugLine)
            {
                cuttingRod.ShowLine = false;
            }

            if (driveRodShapeFromThisComponent)
            {
                cuttingRod.Length = stickLength;
                cuttingRod.Thickness = stickDiameter;
            }
            else
            {
                stickLength = cuttingRod.Length;
                stickDiameter = cuttingRod.Thickness;
            }

            ApplyModelPose();
        }

        void ApplyModelPose()
        {
            if (stickModel == null)
            {
                return;
            }

            float lengthScale = stickLength / Mathf.Max(0.0001f, _sourceLength);
            float diameterScale = stickDiameter / Mathf.Max(0.0001f, _sourceDiameter);

            stickModel.localRotation = Quaternion.Euler(modelEulerOffset);
            stickModel.localScale = new Vector3(diameterScale, diameterScale, lengthScale);
            stickModel.localPosition = modelLocalOffset + Vector3.back * (_sourceCenterZ * lengthScale);
        }

        void MeasureModelBounds()
        {
            if (stickModel == null)
            {
                return;
            }

            bool hasBounds = false;
            Bounds bounds = default;
            Matrix4x4 rootFromLocal;

            foreach (MeshFilter mf in stickModel.GetComponentsInChildren<MeshFilter>())
            {
                Mesh mesh = mf.sharedMesh;
                if (mesh == null)
                {
                    continue;
                }

                rootFromLocal = stickModel.worldToLocalMatrix * mf.transform.localToWorldMatrix;
                Vector3[] vertices = mesh.vertices;
                for (int i = 0; i < vertices.Length; i++)
                {
                    Vector3 p = rootFromLocal.MultiplyPoint3x4(vertices[i]);
                    if (!hasBounds)
                    {
                        bounds = new Bounds(p, Vector3.zero);
                        hasBounds = true;
                    }
                    else
                    {
                        bounds.Encapsulate(p);
                    }
                }
            }

            if (!hasBounds)
            {
                return;
            }

            _sourceLength = Mathf.Max(0.0001f, bounds.size.z);
            _sourceDiameter = Mathf.Max(0.0001f, Mathf.Max(bounds.size.x, bounds.size.y));
            _sourceCenterZ = bounds.center.z;
        }

        void ApplyMaterialIfNeeded(Transform root)
        {
            if (stickMaterial == null)
            {
                return;
            }

            foreach (MeshRenderer mr in root.GetComponentsInChildren<MeshRenderer>())
            {
                mr.sharedMaterial = stickMaterial;
            }
        }

        Material ResolveMaterial()
        {
            if (stickMaterial != null)
            {
                return stickMaterial;
            }

            if (_runtimeMaterial == null)
            {
                Shader shader = Shader.Find("Standard") ?? Shader.Find("Diffuse");
                _runtimeMaterial = new Material(shader) { name = "Runtime_Xiaogun_Material" };
                _runtimeMaterial.color = new Color(0.78f, 0.82f, 0.86f, 1f);
            }

            return _runtimeMaterial;
        }

        static Mesh LoadObjMesh(string path)
        {
            var positions = new List<Vector3>();
            var triangles = new List<int>();
            var meshVertices = new List<Vector3>();

            string[] lines = File.ReadAllLines(path);
            for (int lineIndex = 0; lineIndex < lines.Length; lineIndex++)
            {
                string line = lines[lineIndex].Trim();
                if (line.Length == 0 || line.StartsWith("#"))
                {
                    continue;
                }

                string[] parts = line.Split((char[])null, System.StringSplitOptions.RemoveEmptyEntries);
                if (parts.Length == 0)
                {
                    continue;
                }

                if (parts[0] == "v" && parts.Length >= 4)
                {
                    positions.Add(new Vector3(ParseFloat(parts[1]), ParseFloat(parts[2]), ParseFloat(parts[3])));
                }
                else if (parts[0] == "f" && parts.Length >= 4)
                {
                    var face = new List<int>(parts.Length - 1);
                    for (int i = 1; i < parts.Length; i++)
                    {
                        int positionIndex = ParseObjIndex(parts[i], positions.Count);
                        if (positionIndex < 0 || positionIndex >= positions.Count)
                        {
                            Debug.LogWarning($"[LiverCuttingStickModel] Skipped invalid OBJ face index at {Path.GetFileName(path)}:{lineIndex + 1}");
                            face.Clear();
                            break;
                        }

                        face.Add(meshVertices.Count);
                        meshVertices.Add(positions[positionIndex]);
                    }

                    for (int i = 1; i + 1 < face.Count; i++)
                    {
                        triangles.Add(face[0]);
                        triangles.Add(face[i]);
                        triangles.Add(face[i + 1]);
                    }
                }
            }

            if (meshVertices.Count == 0 || triangles.Count == 0)
            {
                Debug.LogError($"[LiverCuttingStickModel] OBJ has no usable triangles: {path}");
                return null;
            }

            var mesh = new Mesh { name = Path.GetFileNameWithoutExtension(path) };
            if (meshVertices.Count > 65535)
            {
                mesh.indexFormat = IndexFormat.UInt32;
            }

            mesh.SetVertices(meshVertices);
            mesh.SetTriangles(triangles, 0);
            mesh.RecalculateNormals();
            mesh.RecalculateBounds();
            return mesh;
        }

        static int ParseObjIndex(string token, int count)
        {
            string first = token.Split('/')[0];
            if (!int.TryParse(first, NumberStyles.Integer, CultureInfo.InvariantCulture, out int index))
            {
                return -1;
            }

            return index < 0 ? count + index : index - 1;
        }

        static float ParseFloat(string value)
        {
            return float.Parse(value, NumberStyles.Float, CultureInfo.InvariantCulture);
        }

        void OnDestroy()
        {
            if (_runtimeMesh != null)
            {
                Destroy(_runtimeMesh);
            }

            if (_runtimeMaterial != null)
            {
                Destroy(_runtimeMaterial);
            }
        }
    }
}
