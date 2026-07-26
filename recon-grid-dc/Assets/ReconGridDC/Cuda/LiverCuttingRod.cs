using UnityEngine;
using ReconGridDC.Demo;

namespace ReconGridDC.Cuda
{
    /// <summary>
    /// Scene-facing cutting rod target. Assign this GameObject to inverseAPI/TargetObject.targetObject
    /// so the Haply cursor drives the cut line through the existing Target Object API.
    /// </summary>
    [DisallowMultipleComponent]
    public sealed class LiverCuttingRod : MonoBehaviour
    {
        [Header("Rod Shape")]
        [SerializeField] float length = 12f;
        [SerializeField] float thickness = 0.05f;

        [Header("Visual")]
        [SerializeField] bool showLine = true;
        [SerializeField] Color lineColor = Color.cyan;

        LineRenderer _line;
        Material _lineMaterial;

        public float Length
        {
            get => length;
            set
            {
                length = Mathf.Max(0.001f, value);
                UpdateVisual();
            }
        }

        public float Thickness
        {
            get => thickness;
            set
            {
                thickness = Mathf.Max(0.001f, value);
                UpdateVisual();
            }
        }

        public bool ShowLine
        {
            get => showLine;
            set
            {
                showLine = value;
                UpdateVisual();
            }
        }

        public Vector3 Axis => transform.forward.sqrMagnitude > 1e-12f ? transform.forward.normalized : Vector3.forward;
        public Vector3 Center => transform.position;

        public void Configure(float rodLength, float rodThickness)
        {
            length = Mathf.Max(0.001f, rodLength);
            thickness = Mathf.Max(0.001f, rodThickness);
            EnsureVisual();
            UpdateVisual();
        }

        public void SetPose(Vector3 center, Vector3 axis)
        {
            transform.position = center;
            if (axis.sqrMagnitude > 1e-12f)
            {
                transform.rotation = Quaternion.LookRotation(axis.normalized, Vector3.up);
            }
            UpdateVisual();
        }

        public void GetEndpoints(out Vector3 s, out Vector3 e)
        {
            RodMath.Endpoints(Center, Axis, length, out s, out e);
        }

        void Awake()
        {
            EnsureVisual();
            UpdateVisual();
        }

        void Update()
        {
            UpdateVisual();
        }

        void OnValidate()
        {
            length = Mathf.Max(0.001f, length);
            thickness = Mathf.Max(0.001f, thickness);
            if (Application.isPlaying)
            {
                EnsureVisual();
                UpdateVisual();
            }
        }

        void EnsureVisual()
        {
            if (_line != null) return;

            _line = GetComponent<LineRenderer>();
            if (_line == null) _line = gameObject.AddComponent<LineRenderer>();
            _line.positionCount = 2;
            _line.useWorldSpace = false;

            Shader sh = Shader.Find("Unlit/Color") ?? Shader.Find("Sprites/Default");
            if (sh != null && _line.sharedMaterial == null)
            {
                _lineMaterial = new Material(sh);
                _line.sharedMaterial = _lineMaterial;
            }
        }

        void UpdateVisual()
        {
            if (_line == null) return;

            _line.enabled = showLine;
            _line.widthMultiplier = Mathf.Max(0.02f, thickness);
            if (_line.sharedMaterial != null) _line.sharedMaterial.color = lineColor;

            _line.SetPosition(0, Vector3.forward * (length * 0.5f));
            _line.SetPosition(1, Vector3.back * (length * 0.5f));
        }

        void OnDestroy()
        {
            if (_lineMaterial != null) Destroy(_lineMaterial);
        }
    }
}
