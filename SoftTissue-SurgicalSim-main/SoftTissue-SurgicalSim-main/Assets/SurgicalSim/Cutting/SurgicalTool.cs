// SurgicalTool.cs — 手术刀刃控制
// 刀刃表示为线段 (BladeA → BladeB)
// 移动基于世界坐标系, 不旋转
//   J/L → X, I/K → Y, U/O → Z, Shift → 加速

using System.Collections.Generic;
using System.Globalization;
using System.IO;
using UnityEngine;

namespace SurgicalSim.Cutting
{
    public class SurgicalTool : MonoBehaviour
    {
        [Header("Tool Model")]
        public string objFileName = "xiaogun_v1.obj";
        public float  modelScale  = 0.22f;

        [Header("Blade")]
        [Tooltip("刀刃物理长度 (世界单位, 不受 modelScale 影响)")]
        public float bladeLength = 0.44f;
        [Tooltip("Use loaded OBJ bounds.z * modelScale as the physical blade length.")]
        public bool useModelLengthForBlade = false;
        [Tooltip("Safety clamp for OBJ-derived blade length in world units.")]
        public float maxModelBladeLength = 0.60f;
        [Tooltip("Safety clamp for OBJ-derived blade radius in world units.")]
        public float maxModelBladeRadius = 0.02f;
        [Tooltip("Manual collision/contact radius when not deriving radius from OBJ bounds.")]
        public float bladeRadius = 0.0075f;

        [Header("Visual")]
        public Color toolColor = new Color(0.75f, 0.75f, 0.82f, 1f);

        [Header("Control")]
        public float moveSpeed       = 0.45f;
        public float boostMultiplier = 3.0f;

        [Header("Contact Proxy")]
        public bool useContactProxy = true;
        [Range(0f, 1f)] public float proxyNormalResistance = 0.85f;
        [Range(0f, 1f)] public float proxyPressingFriction = 0.70f;
        [Range(0f, 1f)] public float proxyCuttingFriction = 0.18f;
        public float proxyMaxTargetDistance = 0.08f;

        // ── 刀刃端点 (世界坐标) ─────────────────────────────
        /// <summary>刀刃顶端 (远离尖端的一端)</summary>
        public Vector3 BladeA     { get; private set; }
        /// <summary>刀刃尖端</summary>
        public Vector3 BladeB     { get; private set; }
        /// <summary>上一帧的 BladeA</summary>
        public Vector3 PrevBladeA { get; private set; }
        /// <summary>上一帧的 BladeB</summary>
        public Vector3 PrevBladeB { get; private set; }

        // 保留向后兼容
        public Vector3 TipPosition     => BladeB;
        public Vector3 PrevTipPosition => PrevBladeB;
        public Vector3 ToolDirection   => Vector3.down;
        public bool    IsCutting       { get; set; }
        public Vector3 LastMoveInputWorld { get; private set; }
        public bool    HasMoveInput       { get; private set; }
        public Vector3 ToolCenterWorld    => _toolPos;
        public Vector3 TargetCenterWorld  => _targetToolPos;
        public Vector3 PhysicalBladeA     => BladeA;
        public Vector3 PhysicalBladeB     => BladeB;
        public Vector3 ContactBladeA      => BladeA;
        public Vector3 ContactBladeB      => BladeB;
        public float   PhysicalBladeLength => EffectiveBladeLength;
        public float   PhysicalBladeRadius => EffectiveBladeRadius;
        public float   LastProxyLag { get; private set; }
        public float   LastProxyNormalCorrection { get; private set; }
        public float   LastProxyFrictionScale { get; private set; } = 1f;
        public float   EffectiveBladeLength
        {
            get
            {
                float len = useModelLengthForBlade
                    ? _meshHalfLen * 2f * Mathf.Max(0.0001f, modelScale)
                    : bladeLength;
                return useModelLengthForBlade
                    ? Mathf.Clamp(len, 0.02f, Mathf.Max(0.02f, maxModelBladeLength))
                    : Mathf.Max(0.02f, len);
            }
        }
        public float   EffectiveBladeRadius
        {
            get
            {
                float radius = useModelLengthForBlade
                    ? _meshRadius * Mathf.Max(0.0001f, modelScale)
                    : bladeRadius;
                return Mathf.Clamp(radius, 0.002f, Mathf.Max(0.002f, maxModelBladeRadius));
            }
        }

        // ── 内部 ─────────────────────────────────────────────
        GameObject   _toolObj;
        MeshFilter   _toolMF;
        MeshRenderer _toolMR;
        Vector3      _toolPos;
        Vector3      _targetToolPos;
        float        _meshHalfLen = 0.5f;
        float        _meshRadius = 0.05f;
        bool         _proxyContactActive;
        Vector3      _proxyContactNormalWorld = Vector3.up;
        float        _proxyPressure01;
        bool         _proxyCutting;
        bool         _proxyCutAdvanced;

        void Start()
        {
            _toolPos = new Vector3(0f, 0.2f, 0f);
            _targetToolPos = _toolPos;
            CreateToolFromOBJ();
            UpdateBladeEndpoints();
            PrevBladeA = BladeA;
            PrevBladeB = BladeB;
            ApplyTransform();
        }

        void Update()
        {
            // 1. 保存上帧刀刃
            PrevBladeA = BladeA;
            PrevBladeB = BladeB;

            // 2. 世界坐标移动
            Vector3 move = Vector3.zero;
            if (Input.GetKey(KeyCode.L)) move.x += 1f;
            if (Input.GetKey(KeyCode.J)) move.x -= 1f;
            if (Input.GetKey(KeyCode.I)) move.y += 1f;
            if (Input.GetKey(KeyCode.K)) move.y -= 1f;
            if (Input.GetKey(KeyCode.O)) move.z += 1f;
            if (Input.GetKey(KeyCode.U)) move.z -= 1f;
            HasMoveInput = move.sqrMagnitude > 1e-8f;
            LastMoveInputWorld = HasMoveInput ? move.normalized : Vector3.zero;

            bool boost = Input.GetKey(KeyCode.LeftShift) || Input.GetKey(KeyCode.RightShift);
            float speed = boost ? moveSpeed * boostMultiplier : moveSpeed;
            if (HasMoveInput)
                _targetToolPos += LastMoveInputWorld * speed * Time.deltaTime;

            AdvanceContactProxy();

            IsCutting = false; // 由 CuttingToolV3 碰触检测控制

            // 3. 更新刀刃端点
            UpdateBladeEndpoints();
            ApplyTransform();
        }

        public void SetContactProxyFeedback(bool active, Vector3 normalWorld,
                                            float pressure01, bool cutting,
                                            bool cutAdvanced)
        {
            _proxyContactActive = active && normalWorld.sqrMagnitude > 1e-8f;
            if (_proxyContactActive)
                _proxyContactNormalWorld = normalWorld.normalized;
            _proxyPressure01 = Mathf.Clamp01(pressure01);
            _proxyCutting = cutting;
            _proxyCutAdvanced = cutAdvanced;
        }

        void AdvanceContactProxy()
        {
            LastProxyNormalCorrection = 0f;
            LastProxyFrictionScale = 1f;

            if (!useContactProxy || !_proxyContactActive || _proxyPressure01 <= 1e-4f)
            {
                _toolPos = _targetToolPos;
                LastProxyLag = 0f;
                return;
            }

            Vector3 toTarget = _targetToolPos - _toolPos;
            float maxGap = Mathf.Max(0.001f, proxyMaxTargetDistance);
            if (toTarget.magnitude > maxGap)
            {
                toTarget = toTarget.normalized * maxGap;
                _targetToolPos = _toolPos + toTarget;
            }

            Vector3 n = _proxyContactNormalWorld.sqrMagnitude > 1e-8f
                ? _proxyContactNormalWorld.normalized
                : Vector3.up;
            float normalMove = Vector3.Dot(toTarget, n);
            Vector3 normalPart = n * normalMove;
            Vector3 tangentPart = toTarget - normalPart;

            if (normalMove > 0f)
            {
                float block = Mathf.Clamp01(proxyNormalResistance * _proxyPressure01);
                LastProxyNormalCorrection = normalMove * block;
                normalPart = n * (normalMove * (1f - block));
            }

            float friction = (_proxyCutting && _proxyCutAdvanced)
                ? proxyCuttingFriction
                : proxyPressingFriction;
            LastProxyFrictionScale = Mathf.Clamp01(1f - Mathf.Clamp01(friction) * _proxyPressure01);
            tangentPart *= LastProxyFrictionScale;

            _toolPos += normalPart + tangentPart;
            LastProxyLag = (_targetToolPos - _toolPos).magnitude;
        }

        void FixedUpdate()
        {
            UpdateBladeEndpoints();
            ApplyTransform();
        }

        void UpdateBladeEndpoints()
        {
            // BladeA/BladeB are the real contact segment; V3 owns any virtual cut length.
            float halfBlade = EffectiveBladeLength * 0.5f;
            // BladeA = 上端, BladeB = 下端(尖端)
            BladeA = _toolPos - ToolDirection * halfBlade;  // 上
            BladeB = _toolPos + ToolDirection * halfBlade;  // 下(尖端)
        }

        void ApplyTransform()
        {
            if (_toolObj == null) return;
            _toolObj.transform.position   = _toolPos;
            _toolObj.transform.rotation   = Quaternion.LookRotation(ToolDirection);
            _toolObj.transform.localScale = Vector3.one * modelScale;
        }

        // ═══════════════ OBJ 加载 ═══════════════════════════

        void CreateToolFromOBJ()
        {
            _toolObj = new GameObject("SurgicalTool_Model");
            _toolObj.transform.SetParent(transform);
            _toolMF = _toolObj.AddComponent<MeshFilter>();
            _toolMR = _toolObj.AddComponent<MeshRenderer>();

            string objPath = Path.Combine(Application.streamingAssetsPath, objFileName);
            if (File.Exists(objPath))
            {
                Mesh mesh = LoadOBJ(objPath);
                if (mesh != null)
                {
                    _toolMF.mesh = mesh;
                    _meshHalfLen = mesh.bounds.extents.z;
                    _meshRadius = Mathf.Max(mesh.bounds.extents.x, mesh.bounds.extents.y);
                    Debug.Log($"[SurgicalTool] OBJ V:{mesh.vertexCount} HalfLen:{_meshHalfLen:F3} Radius:{_meshRadius:F3}");
                }
            }
            else
            {
                Debug.LogWarning($"[SurgicalTool] OBJ not found: {objPath}");
                var cyl = GameObject.CreatePrimitive(PrimitiveType.Capsule);
                _toolMF.mesh = cyl.GetComponent<MeshFilter>().mesh;
                _meshHalfLen = 0.5f;
                _meshRadius = 0.05f;
                Destroy(cyl);
            }

            var mat = new Material(
                Shader.Find("Universal Render Pipeline/Lit") ??
                Shader.Find("Standard"));
            mat.color = toolColor;
            _toolMR.material = mat;
        }

        static Mesh LoadOBJ(string path)
        {
            var verts = new List<Vector3>();
            var tris  = new List<int>();
            foreach (string line in File.ReadAllLines(path))
            {
                string s = line.Trim();
                if (s.StartsWith("v "))
                {
                    var p = s.Split(new[]{' '}, System.StringSplitOptions.RemoveEmptyEntries);
                    if (p.Length >= 4)
                        verts.Add(new Vector3(
                            float.Parse(p[1], CultureInfo.InvariantCulture),
                            float.Parse(p[2], CultureInfo.InvariantCulture),
                            float.Parse(p[3], CultureInfo.InvariantCulture)));
                }
                else if (s.StartsWith("f "))
                {
                    var p  = s.Split(new[]{' '}, System.StringSplitOptions.RemoveEmptyEntries);
                    var fv = new List<int>();
                    for (int i = 1; i < p.Length; i++)
                        fv.Add(int.Parse(p[i].Split('/')[0], CultureInfo.InvariantCulture) - 1);
                    for (int i = 1; i < fv.Count - 1; i++)
                    { tris.Add(fv[0]); tris.Add(fv[i]); tris.Add(fv[i+1]); }
                }
            }
            if (verts.Count == 0) return null;
            var mesh = new Mesh { name = Path.GetFileNameWithoutExtension(path) };
            mesh.SetVertices(verts);
            mesh.SetTriangles(tris, 0);
            mesh.RecalculateNormals();
            mesh.RecalculateBounds();
            return mesh;
        }

        void OnDestroy() { if (_toolObj) Destroy(_toolObj); }
    }
}
