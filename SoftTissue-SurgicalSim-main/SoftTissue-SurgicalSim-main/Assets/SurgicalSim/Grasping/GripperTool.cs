// GripperTool.cs — Da Vinci 风格夹爪工具（真实 OBJ 模型版）
// 使用 3 个视觉 OBJ + 3 个碰撞 OBJ
// Phase 1: 戳压形变 (Poking) — 碰撞网格三角形 vs 软体粒子
// Phase 2: 摩擦夹取 (Grasping) — InvMass 锁定法
//
// 键盘控制：F/H→X, G/B→Y, V/N→Z, 1/3→旋转, 0→开合
// 与切割工具完全独立运行，可同时使用。

using System.Collections.Generic;
using System.Globalization;
using System.IO;
using UnityEngine;
using SurgicalSim.Core;
using SurgicalSim.Physics;

namespace SurgicalSim.Grasping
{
    public class GripperTool : MonoBehaviour
    {
        const float FourXModelScale = 0.01f;
        const float FourXCollisionMargin = FourXModelScale * 1.5f;
        const float FourXJawCapsuleRadius = FourXModelScale * 1.2f;
        const float FourXShaftCapsuleRadius = FourXModelScale * 2.8f;

        [Header("OBJ 模型文件")]
        public string jawUpVisual   = "Haptic_grasper_jaws_up.obj";
        public string jawDownVisual = "Haptic_grasper_jaws_down.obj";
        public string shaftVisual   = "Haptic_grasper_shaft.obj";
        public string jawUpCol      = "Haptic_grasper_jaws_up_collision.obj";
        public string jawDownCol    = "Haptic_grasper_jaws_down_collision.obj";
        public string shaftCol      = "Haptic_grasper_shaft_collision.obj";

        [Header("模型参数")]
        [Tooltip("OBJ unit to world scale. Current preset keeps the enlarged visual gripper at 0.01.")]
        public float modelScale = FourXModelScale;

        [Tooltip("Apply the calibrated enlarged visual/collision gripper preset at runtime.")]
        public bool useFourXGripperPreset = true;

        [Tooltip("两个夹爪相对杆身绕本地 Z 轴的安装旋转角度。")]
        public float jawRollAngle = 90f;

        [Tooltip("每个夹爪分别绕自身纵向中心轴的自转角度。")]
        public float jawSelfRollAngle = 90f;

        [Header("物理参数")]
        [Tooltip("碰撞 AABB 外扩边距 (m)")]
        public float collisionMargin = FourXCollisionMargin;

        [Tooltip("Jaw capsule contact radius (m). Kept close to the visual jaw thickness to avoid oversized dents.")]
        public float capsuleRadius = FourXJawCapsuleRadius;

        [Tooltip("杆身胶囊体半径 (m)，防止杆身穿模")]
        public float shaftRadius = FourXShaftCapsuleRadius;

        [Range(3, 10)]
        [Tooltip("用于拟合两个夹爪的胶囊体总数。奇数会把多出的 1 个分给上颚；杆身单独 1 个胶囊体。")]
        public int jawCapsuleCount = 10;

        [Range(0.25f, 1.5f)]
        [Tooltip("从夹爪 collision OBJ 分段拟合出的半径缩放系数。")]
        public float jawCapsuleFitRadiusScale = 0.75f;

        [Tooltip("显示碰撞胶囊体 Gizmo")]
        public bool showCapsuleGizmo = true;

        [Range(-0.5f, 0.5f)]
        [Tooltip("Move each jaw toward the center line along tool-local X (m).")]
        public float jawHorizontalCenterOffset = 0.012f;

        [Range(-0.5f, 0.5f)]
        [Tooltip("Move each jaw toward the center line along tool-local Y (m).")]
        public float jawVerticalCenterOffset = 0.012f;

        [Range(0f, 5f)]
        [Tooltip("Jaw-only contact friction multiplier. Shaft friction stays zero.")]
        public float jawContactFrictionMultiplier = 1000f;

        [Header("开合")]
        [Tooltip("最大张开角度 (度)")]
        public float maxOpenAngle = 50f;
        [Tooltip("开合角速度 (度/秒)")]
        public float openCloseSpeed = 60f;

        [Header("高斯硬约束抓取")]
        [Min(0f)]
        [Tooltip("硬约束核心沿虎口方向形成的高斯隆起高度 (m)。")]
        public float gaussianHeight = 0.02f;

        [Range(0.1f, 1f)]
        [Tooltip("高斯标准差相对抓取区域半径的比例；越大，曲面越平缓。")]
        public float gaussianWidth = 0.5f;

        [Range(0.1f, 1f)]
        [Tooltip("归一化椭圆内被硬锁定的核心半径；外围粒子保持动态，由弹性求解器自然过渡。")]
        public float gaussianHardCoreRadius = 0.55f;

        [Min(0f)]
        [Tooltip("从原始形状平滑过渡到高斯形状所需时间 (s)，设为 0 表示立即成形。")]
        public float gaussianFormDuration = 0.15f;

        [Min(1)]
        [Tooltip("Max FixedUpdate attempts after the jaw first reaches the grasp angle.")]
        public int captureRetryPhysicsSteps = 5;

        [Header("控制")]
        public float moveSpeed = 0.3f;
        public float rotateSpeed = 60f;
        public float boostMultiplier = 3.0f;

        [Header("可视化")]
        public Color toolColor = new Color(0.7f, 0.7f, 0.78f, 1f);

        // ── 公开状态 ─────────────────────────────────────────
        public bool IsGrasping { get; private set; }
        public float CurrentAngle => _currentAngle;

        // ── 私有状态 ─────────────────────────────────────────
        TetMeshData       _data;
        XPBDSolverGPU     _solver;
        TetMeshVisualizer _visualizer;

        Vector3 _toolPos;
        float   _toolRotY;       // 绕 Y 轴旋转角度
        float   _currentAngle;   // 当前开合角度 (0=闭合, maxOpenAngle=全开)
        bool    _wantClose;
        bool    _initialized;

        // 视觉 GameObjects
        GameObject _jawUpObj, _jawDownObj, _shaftObj;

        // 碰撞网格数据 (本地坐标，已缩放)
        Vector3[] _colVertsUp, _colVertsDown, _colVertsShaft;
        int[]     _colTrisUp,  _colTrisDown,  _colTrisShaft;

        // 铰接参数 (从模型推断)
        // 上下颚绕 Z 轴方向的铰接点旋转
        // 铰接点大约在 Y=9.0, Z=-25.5 (OBJ 坐标)
        Vector3 _pivotLocal; // 缩放后的铰接点 (本地坐标)
        Vector3 _tipLocal;   // 缩放后的尖端 (本地坐标)
        Vector3 _upperJawSelfAxisPointLocal;
        Vector3 _lowerJawSelfAxisPointLocal;

        // 胶囊体世界坐标 (用于 Gizmo 可视化)
        const int MaxGripperCapsules = 12;

        struct LocalCapsule
        {
            public Vector3 a, b;
            public float radius;
        }

        readonly List<LocalCapsule> _upperJawLocalCapsules = new List<LocalCapsule>(5);
        readonly List<LocalCapsule> _lowerJawLocalCapsules = new List<LocalCapsule>(5);
        readonly Vector3[] _capsuleA = new Vector3[MaxGripperCapsules];
        readonly Vector3[] _capsuleB = new Vector3[MaxGripperCapsules];
        readonly Vector3[] _prevCapsuleA = new Vector3[MaxGripperCapsules];
        readonly Vector3[] _prevCapsuleB = new Vector3[MaxGripperCapsules];
        readonly float[] _capsuleR = new float[MaxGripperCapsules];
        readonly float[] _capsuleFriction = new float[MaxGripperCapsules];
        readonly Vector3[] _lastCapsuleA = new Vector3[MaxGripperCapsules];
        readonly Vector3[] _lastCapsuleB = new Vector3[MaxGripperCapsules];
        readonly Vector3[] _dbgCapsuleA = new Vector3[MaxGripperCapsules];
        readonly Vector3[] _dbgCapsuleB = new Vector3[MaxGripperCapsules];
        readonly float[] _dbgCapsuleR = new float[MaxGripperCapsules];
        int _lastCapsuleCount;
        int _dbgCapsuleCount;
        int _dbgUpperCapsuleCount;
        int _dbgLowerCapsuleCount;
        Vector3 _dbgBBoxMin, _dbgBBoxMax;
        bool _hasPrevCapsules;

        // 夹取状态
        struct GraspedParticle
        {
            public int index;
            public float originalInvMass;
            public Vector3 frameCoordinates; // 抓取坐标系中的 (u, v, 厚度)
            public float gaussianWeight;
        }
        readonly List<GraspedParticle> _graspedParticles = new List<GraspedParticle>();
        readonly List<int> _graspedParticleIndices = new List<int>();
        Vector3 _graspCenterToolLocal;
        float _graspShapeBlend;
        int _transitionParticleCount;
        int _captureAttemptCount;
        bool _captureGaveUp;

        // ══════════════════════════════════════════════════════
        // 初始化
        // ══════════════════════════════════════════════════════
        public void Init(TetMeshData data, XPBDSolverGPU solver, TetMeshVisualizer visualizer)
        {
            _data = data;
            _solver = solver;
            _visualizer = visualizer;

            ApplyGripperScalePreset();

            _toolPos = new Vector3(0.15f, 0.5f, 0f);
            _toolRotY = 0f;
            _currentAngle = maxOpenAngle; // 初始张开
            _wantClose = false;
            IsGrasping = false;
            _graspedParticles.Clear();
            _graspedParticleIndices.Clear();
            _graspShapeBlend = 0f;
            _transitionParticleCount = 0;
            ResetCaptureRetryState();
            _uploadFrame = 0; // 重置诊断计数器

            // 铰接点 (OBJ 坐标 → 缩放后)
            _pivotLocal = new Vector3(0f, 9.0f, -25.5f) * modelScale;
            // 夹爪尖端 (OBJ 坐标 → 缩放后)
            _tipLocal = new Vector3(0f, 9.0f, -43.0f) * modelScale;
            // 杆身尾端 (手柄方向，延伸足够长以覆盖整个杆身)
            _shaftEndLocal = new Vector3(0f, 9.0f, 307.0f) * modelScale;
            _shaftTipLocal = _pivotLocal;
            _prevToolPos = _toolPos;
            _hasPrevCapsules = false;

            LoadModels();
            LoadCollisionMeshes();
            _initialized = true;

            Debug.Log($"[GripperTool] Init | scale={modelScale} | jawCapsules={Mathf.Clamp(jawCapsuleCount, 3, 10)} + shaft");
        }

        void ApplyGripperScalePreset()
        {
            if (!useFourXGripperPreset) return;

            modelScale = FourXModelScale;
            collisionMargin = FourXCollisionMargin;
            capsuleRadius = FourXJawCapsuleRadius;
            shaftRadius = FourXShaftCapsuleRadius;
        }

        // ══════════════════════════════════════════════════════
        // Update — 键盘输入 + 可视化
        // ══════════════════════════════════════════════════════
        void Update()
        {
            if (!_initialized) return;

            // ── 移动（FGHVBN）──────────────────────────────
            Vector3 move = Vector3.zero;
            if (Input.GetKey(KeyCode.H)) move.x += 1f;
            if (Input.GetKey(KeyCode.F)) move.x -= 1f;
            if (Input.GetKey(KeyCode.G)) move.y += 1f;
            if (Input.GetKey(KeyCode.B)) move.y -= 1f;
            if (Input.GetKey(KeyCode.N)) move.z += 1f;
            if (Input.GetKey(KeyCode.V)) move.z -= 1f;

            bool boost = Input.GetKey(KeyCode.LeftShift) || Input.GetKey(KeyCode.RightShift);
            float speed = boost ? moveSpeed * boostMultiplier : moveSpeed;
            if (move.sqrMagnitude > 1e-8f)
            {
                Vector3 desiredMove = move.normalized * speed * Time.deltaTime;
                // ★ 速度钒制: 每帧最大移动 capsuleRadius * 0.5，防止隧穿
                // 参考 SOFA alarmDistance 概念
                float maxMove = capsuleRadius * 0.5f;
                if (desiredMove.magnitude > maxMove)
                    desiredMove = desiredMove.normalized * maxMove;
                _toolPos += desiredMove;
            }

            // ── 旋转（Numpad 1/3）──────────────────────────
            if (Input.GetKey(KeyCode.Keypad1)) _toolRotY -= rotateSpeed * Time.deltaTime;
            if (Input.GetKey(KeyCode.Keypad3)) _toolRotY += rotateSpeed * Time.deltaTime;

            // Z/X continuous jaw control: hold to move, release to stop.
            bool closeHeld = Input.GetKey(KeyCode.Z);
            bool openHeld = Input.GetKey(KeyCode.X);
            float jawStep = openCloseSpeed * Time.deltaTime;
            if (closeHeld && !openHeld)
            {
                _wantClose = true;
                _currentAngle = Mathf.MoveTowards(_currentAngle, 0f, jawStep);
            }
            else if (openHeld && !closeHeld)
            {
                _wantClose = false;
                _currentAngle = Mathf.MoveTowards(_currentAngle, maxOpenAngle, jawStep);
            }

            UpdateVisual();
        }

        // ══════════════════════════════════════════════════════
        // PhysicsStep — 在 FixedUpdate 中被 SoftBody 调用
        // 碰撞已由 GPU CSToolCollision 处理，这里只做夹取
        // ══════════════════════════════════════════════════════
        public void PhysicsStep()
        {
            if (!_initialized || _data == null || _solver == null) return;

            Transform tf = _visualizer != null ? _visualizer.transform : transform;

            // 夹取逻辑（碰撞已由 GPU 处理）
            bool isFullyClosed = _wantClose && _currentAngle <= 25f;

            if (!isFullyClosed && !IsGrasping)
                ResetCaptureRetryState();

            if (isFullyClosed && !IsGrasping)
            {
                TryCaptureParticles(tf);
            }
            else if (IsGrasping && isFullyClosed)
            {
                UpdateGraspedParticles(tf);
            }
            else if (IsGrasping && !_wantClose)
            {
                ReleaseParticles();
                IsGrasping = false;
                ResetCaptureRetryState();
            }
        }

        void TryCaptureParticles(Transform meshTf)
        {
            if (_captureGaveUp)
                return;

            int maxAttempts = Mathf.Max(1, captureRetryPhysicsSteps);
            _captureAttemptCount++;
            IsGrasping = CaptureParticles(meshTf);

            if (IsGrasping)
                return;

            if (_captureAttemptCount >= maxAttempts)
            {
                _captureGaveUp = true;
                Debug.Log($"[GripperTool] Capture gave up after {_captureAttemptCount} physics steps.");
            }
        }

        void ResetCaptureRetryState()
        {
            _captureAttemptCount = 0;
            _captureGaveUp = false;
        }

        // ══════════════════════════════════════════════════════
        // GPU 碰撞: 夹爪 3~10 个拟合胶囊体 + 1 个杆身胶囊体
        // 参考 SOFA: proximity/contact pipeline + CCD
        // ══════════════════════════════════════════════════════
        int _uploadFrame = 0;
        Vector3 _prevToolPos;
        Vector3 _shaftEndLocal; // 杆身尾端(手柄方向)

        Vector3 _shaftTipLocal; // shaft tip/hinge end in scaled OBJ local space

        public void UploadToolCollisionToGPU()
        {
            if (!_initialized || _solver == null) return;

            Quaternion toolRot = Quaternion.Euler(0, _toolRotY, 0);
            Quaternion upperRot = Quaternion.AngleAxis(_currentAngle, Vector3.right);
            Quaternion lowerRot = Quaternion.AngleAxis(-_currentAngle, Vector3.right);

            int capsuleCount = 0;
            int upperAdded = AddJawCapsules(
                _upperJawLocalCapsules, true, upperRot, toolRot, ref capsuleCount);
            int lowerAdded = AddJawCapsules(
                _lowerJawLocalCapsules, false, lowerRot, toolRot, ref capsuleCount);

            Vector3 shaftA = toolRot * _shaftEndLocal + _toolPos;
            Vector3 shaftB = toolRot * _shaftTipLocal + _toolPos;
            AddCapsule(shaftA, shaftB, shaftRadius, ref capsuleCount);

            for (int i = 0; i < MaxGripperCapsules; i++)
                _capsuleFriction[i] = 0f;

            int jawCapsuleCount = upperAdded + lowerAdded;
            float jawFriction = IsGrasping ? 0f : Mathf.Max(0f, jawContactFrictionMultiplier);
            for (int i = 0; i < jawCapsuleCount && i < capsuleCount; i++)
                _capsuleFriction[i] = jawFriction;

            for (int i = 0; i < capsuleCount; i++)
            {
                if (_hasPrevCapsules && i < _lastCapsuleCount)
                {
                    _prevCapsuleA[i] = _lastCapsuleA[i];
                    _prevCapsuleB[i] = _lastCapsuleB[i];
                }
                else
                {
                    _prevCapsuleA[i] = _capsuleA[i];
                    _prevCapsuleB[i] = _capsuleB[i];
                }
            }

            _solver.SetCapsuleCollisionParams(
                _capsuleA,
                _capsuleB,
                _capsuleR,
                _prevCapsuleA,
                _prevCapsuleB,
                capsuleCount,
                _capsuleFriction);

            _dbgCapsuleCount = capsuleCount;
            _dbgUpperCapsuleCount = upperAdded;
            _dbgLowerCapsuleCount = lowerAdded;
            _dbgBBoxMin = Vector3.positiveInfinity;
            _dbgBBoxMax = Vector3.negativeInfinity;
            for (int i = 0; i < capsuleCount; i++)
            {
                _dbgCapsuleA[i] = _capsuleA[i];
                _dbgCapsuleB[i] = _capsuleB[i];
                _dbgCapsuleR[i] = _capsuleR[i];

                float margin = collisionMargin + _capsuleR[i];
                Vector3 pad = Vector3.one * margin;
                _dbgBBoxMin = Vector3.Min(_dbgBBoxMin, Vector3.Min(_capsuleA[i], _capsuleB[i]) - pad);
                _dbgBBoxMax = Vector3.Max(_dbgBBoxMax, Vector3.Max(_capsuleA[i], _capsuleB[i]) + pad);
            }

            if (capsuleCount == 0)
            {
                _dbgBBoxMin = Vector3.zero;
                _dbgBBoxMax = Vector3.zero;
            }

            if (_uploadFrame < 3)
            {
                Debug.Log($"[GripperTool] capsule collision F{_uploadFrame}: " +
                    $"jawCaps={upperAdded + lowerAdded} upper={upperAdded} lower={lowerAdded} " +
                    $"total={capsuleCount} shaftR={shaftRadius:F4}");
                _uploadFrame++;
            }

            for (int i = 0; i < capsuleCount; i++)
            {
                _lastCapsuleA[i] = _capsuleA[i];
                _lastCapsuleB[i] = _capsuleB[i];
            }
            _lastCapsuleCount = capsuleCount;
            _hasPrevCapsules = true;
            _prevToolPos = _toolPos;
        }

        int AddJawCapsules(
            List<LocalCapsule> localCapsules,
            bool isUpperJaw,
            Quaternion jawRot,
            Quaternion toolRot,
            ref int capsuleCount)
        {
            if (localCapsules == null || localCapsules.Count == 0)
                return 0;

            int added = 0;
            for (int i = 0; i < localCapsules.Count; i++)
            {
                LocalCapsule c = localCapsules[i];
                Vector3 a = TransformJawLocalPoint(c.a, isUpperJaw, jawRot, toolRot);
                Vector3 b = TransformJawLocalPoint(c.b, isUpperJaw, jawRot, toolRot);
                if (AddCapsule(a, b, c.radius, ref capsuleCount))
                    added++;
            }
            return added;
        }

        Vector3 TransformJawLocalPoint(
            Vector3 localPoint,
            bool isUpperJaw,
            Quaternion jawRot,
            Quaternion toolRot)
        {
            Vector3 selfRolledPoint = RotateAroundJawSelfAxis(localPoint, isUpperJaw);
            Vector3 v = selfRolledPoint - _pivotLocal;
            v = jawRot * v;
            v = Quaternion.AngleAxis(jawRollAngle, Vector3.forward) * v;
            v += _pivotLocal;
            v += GetJawCenteringOffsetLocal(isUpperJaw);
            return toolRot * v + _toolPos;
        }

        Vector3 GetJawCenteringOffsetLocal(bool isUpperJaw)
        {
            float sign = isUpperJaw ? -1f : 1f;
            return new Vector3(
                sign * jawHorizontalCenterOffset,
                sign * jawVerticalCenterOffset,
                0f);
        }

        Vector3 RotateAroundJawSelfAxis(Vector3 point, bool isUpperJaw)
        {
            Vector3 axisPoint = isUpperJaw
                ? _upperJawSelfAxisPointLocal
                : _lowerJawSelfAxisPointLocal;
            Quaternion selfRoll = Quaternion.AngleAxis(jawSelfRollAngle, Vector3.forward);
            return axisPoint + selfRoll * (point - axisPoint);
        }

        bool AddCapsule(Vector3 a, Vector3 b, float radius, ref int capsuleCount)
        {
            if (capsuleCount >= MaxGripperCapsules)
                return false;

            _capsuleA[capsuleCount] = a;
            _capsuleB[capsuleCount] = b;
            _capsuleR[capsuleCount] = Mathf.Max(0.0001f, radius);
            capsuleCount++;
            return true;
        }


        // ══════════════════════════════════════════════════════
        // Phase 2: 夹取
        // ══════════════════════════════════════════════════════
        bool CaptureParticles(Transform meshTf)
        {
            _graspedParticles.Clear();
            _graspedParticleIndices.Clear();
            _graspShapeBlend = 0f;
            _transitionParticleCount = 0;

            // 夹爪尖端区域 (OBJ Z ≈ -43 → 世界坐标)
            // 上下颚闭合时，夹住它们之间的粒子
            var worldVertsUp   = TransformCollisionVerts(_colVertsUp, true);
            var worldVertsDown = TransformCollisionVerts(_colVertsDown, false);

            // 用两组碰撞顶点的 AABB 交集作为夹取区域
            Vector3 bboxMin = new Vector3(float.MaxValue, float.MaxValue, float.MaxValue);
            Vector3 bboxMax = new Vector3(float.MinValue, float.MinValue, float.MinValue);
            foreach (var v in worldVertsUp)
            { bboxMin = Vector3.Min(bboxMin, v); bboxMax = Vector3.Max(bboxMax, v); }
            foreach (var v in worldVertsDown)
            { bboxMin = Vector3.Min(bboxMin, v); bboxMax = Vector3.Max(bboxMax, v); }

            // 缩小一点 bbox 只取核心区域
            Vector3 shrink = (bboxMax - bboxMin) * 0.1f;
            bboxMin += shrink;
            bboxMax -= shrink;

            // 转换到 mesh local space
            Vector3 localMin = meshTf.InverseTransformPoint(bboxMin);
            Vector3 localMax = meshTf.InverseTransformPoint(bboxMax);
            // 确保 min < max
            Vector3 realMin = Vector3.Min(localMin, localMax);
            Vector3 realMax = Vector3.Max(localMin, localMax);

            Vector3 jawCenterWorld = (bboxMin + bboxMax) * 0.5f;
            Quaternion toolRot = Quaternion.Euler(0f, _toolRotY, 0f);
            _graspCenterToolLocal = Quaternion.Inverse(toolRot) * (jawCenterWorld - _toolPos);
            GetGraspFrame(out Vector3 frameCenter, out Vector3 axisU,
                out Vector3 axisV, out Vector3 towardCrotch);

            var candidateIndices = new List<int>();
            var candidateCoordinates = new List<Vector3>();
            float halfU = 0f;
            float halfV = 0f;
            for (int i = 0; i < _data.NumParticles; i++)
            {
                if (_data.InvMass[i] == 0f) continue;
                Vector3 p = _data.Positions[i];
                if (p.x >= realMin.x && p.x <= realMax.x &&
                    p.y >= realMin.y && p.y <= realMax.y &&
                    p.z >= realMin.z && p.z <= realMax.z)
                {
                    Vector3 worldOffset = meshTf.TransformPoint(p) - frameCenter;
                    Vector3 coordinates = new Vector3(
                        Vector3.Dot(worldOffset, axisU),
                        Vector3.Dot(worldOffset, axisV),
                        Vector3.Dot(worldOffset, towardCrotch));
                    candidateIndices.Add(i);
                    candidateCoordinates.Add(coordinates);
                    halfU = Mathf.Max(halfU, Mathf.Abs(coordinates.x));
                    halfV = Mathf.Max(halfV, Mathf.Abs(coordinates.y));
                }
            }

            if (candidateIndices.Count == 0)
            {
                Debug.Log("[GripperTool] 抓取区域内没有可夹取粒子");
                return false;
            }

            halfU = Mathf.Max(halfU, 0.0001f);
            halfV = Mathf.Max(halfV, 0.0001f);
            float coreRadius = Mathf.Clamp(gaussianHardCoreRadius, 0.1f, 1f);
            float sigmaU = Mathf.Max(halfU * Mathf.Clamp(gaussianWidth, 0.1f, 1f), 0.0001f);
            float sigmaV = Mathf.Max(halfV * Mathf.Clamp(gaussianWidth, 0.1f, 1f), 0.0001f);
            int nearestCandidate = -1;
            float nearestRadius = float.MaxValue;

            for (int c = 0; c < candidateIndices.Count; c++)
            {
                Vector3 coordinates = candidateCoordinates[c];
                float normalizedU = coordinates.x / halfU;
                float normalizedV = coordinates.y / halfV;
                float normalizedRadius = Mathf.Sqrt(
                    normalizedU * normalizedU + normalizedV * normalizedV);

                if (normalizedRadius < nearestRadius)
                {
                    nearestRadius = normalizedRadius;
                    nearestCandidate = c;
                }

                if (normalizedRadius > coreRadius)
                {
                    _transitionParticleCount++;
                    continue;
                }

                float gaussianWeight = EvaluateGaussianWeight(coordinates, sigmaU, sigmaV);
                LockGraspParticle(candidateIndices[c], coordinates, gaussianWeight);
            }

            // 粗网格可能没有粒子落入核心，至少锁定最靠近中心的一个粒子。
            if (_graspedParticles.Count == 0 && nearestCandidate >= 0)
            {
                Vector3 coordinates = candidateCoordinates[nearestCandidate];
                float gaussianWeight = EvaluateGaussianWeight(coordinates, sigmaU, sigmaV);
                LockGraspParticle(candidateIndices[nearestCandidate], coordinates, gaussianWeight);
                _transitionParticleCount = Mathf.Max(0, _transitionParticleCount - 1);
            }

            _solver.UploadInvMass(_data);
            Debug.Log($"[GripperTool] 高斯硬核 {_graspedParticles.Count} 个粒子 | " +
                      $"动态过渡 {_transitionParticleCount} 个粒子");
            return _graspedParticles.Count > 0;
        }

        void UpdateGraspedParticles(Transform meshTf)
        {
            if (_graspedParticles.Count == 0) return;

            if (gaussianFormDuration <= 0f)
                _graspShapeBlend = 1f;
            else
                _graspShapeBlend = Mathf.MoveTowards(
                    _graspShapeBlend, 1f, Time.fixedDeltaTime / gaussianFormDuration);

            float smoothBlend = Mathf.SmoothStep(0f, 1f, _graspShapeBlend);
            GetGraspFrame(out Vector3 frameCenter, out Vector3 axisU,
                out Vector3 axisV, out Vector3 towardCrotch);

            foreach (var gp in _graspedParticles)
            {
                Vector3 coordinates = gp.frameCoordinates;
                float gaussianOffset = Mathf.Max(0f, gaussianHeight) * gp.gaussianWeight * smoothBlend;
                Vector3 targetWorld = frameCenter +
                    axisU * coordinates.x +
                    axisV * coordinates.y +
                    towardCrotch * (coordinates.z + gaussianOffset);
                Vector3 targetLocal = meshTf.InverseTransformPoint(targetWorld);

                _data.Positions[gp.index] = targetLocal;
                _data.PrevPositions[gp.index] = targetLocal;
                _data.Velocities[gp.index] = Vector3.zero;
            }
            _solver.UploadParticleStates(_data, _graspedParticleIndices);
        }

        void ReleaseParticles()
        {
            foreach (var gp in _graspedParticles)
            {
                _data.InvMass[gp.index] = gp.originalInvMass;
                _data.Velocities[gp.index] *= 0.1f;
            }
            int n = _graspedParticles.Count;
            _graspedParticles.Clear();
            _graspedParticleIndices.Clear();
            _graspShapeBlend = 0f;
            _transitionParticleCount = 0;
            _solver.UploadInvMass(_data);
            Debug.Log($"[GripperTool] 释放了 {n} 个粒子");
        }

        void LockGraspParticle(int index, Vector3 frameCoordinates, float gaussianWeight)
        {
            _graspedParticles.Add(new GraspedParticle
            {
                index = index,
                originalInvMass = _data.InvMass[index],
                frameCoordinates = frameCoordinates,
                gaussianWeight = gaussianWeight
            });
            _graspedParticleIndices.Add(index);
            _data.InvMass[index] = 0f;
        }

        static float EvaluateGaussianWeight(Vector3 coordinates, float sigmaU, float sigmaV)
        {
            float u = coordinates.x / sigmaU;
            float v = coordinates.y / sigmaV;
            return Mathf.Exp(-0.5f * (u * u + v * v));
        }

        void GetGraspFrame(out Vector3 center, out Vector3 axisU,
            out Vector3 axisV, out Vector3 towardCrotch)
        {
            Quaternion toolRot = Quaternion.Euler(0f, _toolRotY, 0f);
            Quaternion jawMountRot = Quaternion.AngleAxis(jawRollAngle, Vector3.forward);
            center = toolRot * _graspCenterToolLocal + _toolPos;
            towardCrotch = (toolRot * (_pivotLocal - _tipLocal)).normalized;

            axisU = toolRot * (jawMountRot * Vector3.right);
            axisU = Vector3.ProjectOnPlane(axisU, towardCrotch).normalized;
            if (axisU.sqrMagnitude < 1e-8f)
                axisU = Vector3.ProjectOnPlane(toolRot * Vector3.up, towardCrotch).normalized;
            axisV = Vector3.Cross(towardCrotch, axisU).normalized;
        }

        // ══════════════════════════════════════════════════════
        // 坐标变换
        // ══════════════════════════════════════════════════════
        /// <summary>将碰撞网格顶点变换到世界坐标（含开合旋转）</summary>
        Vector3[] TransformCollisionVerts(Vector3[] localVerts, bool isUpperJaw)
        {
            var result = new Vector3[localVerts.Length];
            // 开合旋转角度 (上颚向上旋转，下颚向下旋转)
            float jawAngle = isUpperJaw ? _currentAngle : -_currentAngle;
            Quaternion jawRot = Quaternion.AngleAxis(jawAngle, Vector3.right);
            Quaternion jawMountRot = Quaternion.AngleAxis(jawRollAngle, Vector3.forward);
            Quaternion toolRot = Quaternion.Euler(0, _toolRotY, 0);

            for (int i = 0; i < localVerts.Length; i++)
            {
                // 1. 每个夹爪先绕自己的纵向中心轴自转。
                Vector3 selfRolledPoint = RotateAroundJawSelfAxis(localVerts[i], isUpperJaw);

                // 2. 相对铰接点旋转（开合）
                Vector3 v = selfRolledPoint - _pivotLocal;
                v = jawRot * v;
                v = jawMountRot * v;
                v += _pivotLocal;
                v += GetJawCenteringOffsetLocal(isUpperJaw);

                // 3. 工具整体旋转
                v = toolRot * v;

                // 4. 平移到世界位置
                result[i] = v + _toolPos;
            }
            return result;
        }

        Vector3 GetJawTipCenter()
        {
            Quaternion toolRot = Quaternion.Euler(0, _toolRotY, 0);
            return toolRot * _tipLocal + _toolPos;
        }

        /// <summary>CPU 端 Point-to-Capsule 距离（用于诊断）</summary>
        static float PointCapsuleDist(Vector3 p, Vector3 a, Vector3 b)
        {
            Vector3 ab = b - a;
            float ab2 = Vector3.Dot(ab, ab);
            float t = ab2 < 1e-10f ? 0f : Mathf.Clamp01(Vector3.Dot(p - a, ab) / ab2);
            Vector3 closest = a + t * ab;
            return (p - closest).magnitude;
        }

        // ══════════════════════════════════════════════════════
        // OBJ 加载
        // ══════════════════════════════════════════════════════
        void LoadModels()
        {
            // 清除旧的
            if (_jawUpObj) Destroy(_jawUpObj);
            if (_jawDownObj) Destroy(_jawDownObj);
            if (_shaftObj) Destroy(_shaftObj);

            var mat = new Material(
                Shader.Find("Universal Render Pipeline/Lit") ??
                Shader.Find("Standard"));
            mat.color = toolColor;

            _jawUpObj   = CreateMeshObj("GripperJawUp",   jawUpVisual,   mat);
            _jawDownObj = CreateMeshObj("GripperJawDown", jawDownVisual, mat);
            _shaftObj   = CreateMeshObj("GripperShaft",   shaftVisual,   mat);
        }

        void LoadCollisionMeshes()
        {
            string dir = Application.streamingAssetsPath;

            LoadOBJData(Path.Combine(dir, jawUpCol),   out _colVertsUp,   out _colTrisUp);
            LoadOBJData(Path.Combine(dir, jawDownCol), out _colVertsDown, out _colTrisDown);
            LoadOBJData(Path.Combine(dir, shaftCol),   out _colVertsShaft, out _colTrisShaft);

            // 缩放碰撞顶点到世界单位
            if (_colVertsUp != null)
                for (int i = 0; i < _colVertsUp.Length; i++)
                    _colVertsUp[i] *= modelScale;
            if (_colVertsDown != null)
                for (int i = 0; i < _colVertsDown.Length; i++)
                    _colVertsDown[i] *= modelScale;
            if (_colVertsShaft != null)
                for (int i = 0; i < _colVertsShaft.Length; i++)
                    _colVertsShaft[i] *= modelScale;

            _upperJawSelfAxisPointLocal = CalculateJawSelfAxisPoint(_colVertsUp);
            _lowerJawSelfAxisPointLocal = CalculateJawSelfAxisPoint(_colVertsDown);

            Debug.Log($"[GripperTool] 碰撞网格: Up {_colVertsUp?.Length}V/{_colTrisUp?.Length/3}F " +
                      $"| Down {_colVertsDown?.Length}V/{_colTrisDown?.Length/3}F " +
                      $"| Shaft {_colVertsShaft?.Length}V/{_colTrisShaft?.Length/3}F");

            FitShaftCapsuleFromCollisionMesh();
            FitJawCapsules(_colVertsUp, _upperJawLocalCapsules, GetUpperJawCapsuleTargetCount(), "UpperJaw");
            FitJawCapsules(_colVertsDown, _lowerJawLocalCapsules, GetLowerJawCapsuleTargetCount(), "LowerJaw");
        }

        Vector3 CalculateJawSelfAxisPoint(Vector3[] verts)
        {
            if (verts == null || verts.Length == 0)
                return _pivotLocal;

            Vector2 center = Vector2.zero;
            int count = 0;
            float workingEndZ = Mathf.Max(_pivotLocal.z, _tipLocal.z);
            for (int i = 0; i < verts.Length; i++)
            {
                // 只用铰接点到尖端的工作段，避免后部连接结构拉偏中心轴。
                if (verts[i].z > workingEndZ)
                    continue;

                center += new Vector2(verts[i].x, verts[i].y);
                count++;
            }

            if (count == 0)
                return _pivotLocal;

            center /= count;
            return new Vector3(center.x, center.y, _pivotLocal.z);
        }

        int GetUpperJawCapsuleTargetCount()
        {
            int total = Mathf.Clamp(jawCapsuleCount, 3, 10);
            return Mathf.Clamp((total + 1) / 2, 1, 5);
        }

        int GetLowerJawCapsuleTargetCount()
        {
            int total = Mathf.Clamp(jawCapsuleCount, 3, 10);
            return Mathf.Clamp(total / 2, 1, 5);
        }

        void FitJawCapsules(Vector3[] verts, List<LocalCapsule> result, int count, string label)
        {
            result.Clear();
            count = Mathf.Clamp(count, 1, 5);

            if (verts == null || verts.Length == 0)
            {
                FitFallbackJawCapsules(result, count);
                Debug.LogWarning($"[GripperTool] {label} collision mesh missing; using fallback jaw capsules.");
                return;
            }

            Vector3 min = verts[0];
            Vector3 max = verts[0];
            for (int i = 1; i < verts.Length; i++)
            {
                min = Vector3.Min(min, verts[i]);
                max = Vector3.Max(max, verts[i]);
            }

            float zMin = min.z;
            float zMax = max.z;
            float zSpan = Mathf.Max(0.0001f, zMax - zMin);
            float zOverlap = Mathf.Max(0.0001f, zSpan * 0.025f);
            float minRadius = Mathf.Max(0.0005f, capsuleRadius * 0.35f);
            float radiusScale = Mathf.Clamp(
                jawCapsuleFitRadiusScale <= 0f ? 1f : jawCapsuleFitRadiusScale,
                0.25f,
                1.5f);

            for (int c = 0; c < count; c++)
            {
                float z0 = Mathf.Lerp(zMin, zMax, c / (float)count);
                float z1 = Mathf.Lerp(zMin, zMax, (c + 1) / (float)count);
                Vector3 segMin = Vector3.positiveInfinity;
                Vector3 segMax = Vector3.negativeInfinity;
                int segVerts = 0;

                for (int i = 0; i < verts.Length; i++)
                {
                    Vector3 v = verts[i];
                    if (v.z < z0 - zOverlap || v.z > z1 + zOverlap)
                        continue;

                    segMin = Vector3.Min(segMin, v);
                    segMax = Vector3.Max(segMax, v);
                    segVerts++;
                }

                if (segVerts == 0)
                {
                    segMin = min;
                    segMax = max;
                }

                float centerX = (segMin.x + segMax.x) * 0.5f;
                float centerY = (segMin.y + segMax.y) * 0.5f;
                float radius = 0f;

                for (int i = 0; i < verts.Length; i++)
                {
                    Vector3 v = verts[i];
                    if (segVerts > 0 && (v.z < z0 - zOverlap || v.z > z1 + zOverlap))
                        continue;

                    float dx = v.x - centerX;
                    float dy = v.y - centerY;
                    radius = Mathf.Max(radius, Mathf.Sqrt(dx * dx + dy * dy));
                }

                if (radius <= 0.00001f)
                {
                    radius = Mathf.Max(
                        Mathf.Abs(segMax.x - segMin.x) * 0.5f,
                        Mathf.Abs(segMax.y - segMin.y) * 0.5f);
                }

                result.Add(new LocalCapsule
                {
                    a = new Vector3(centerX, centerY, z0),
                    b = new Vector3(centerX, centerY, z1),
                    radius = Mathf.Max(minRadius, radius * radiusScale)
                });
            }

            Debug.Log($"[GripperTool] {label} fitted with {result.Count} capsules " +
                      $"from collision mesh z=[{zMin:F3}, {zMax:F3}]");
        }

        void FitFallbackJawCapsules(List<LocalCapsule> result, int count)
        {
            float zMin = Mathf.Min(_pivotLocal.z, _tipLocal.z);
            float zMax = Mathf.Max(_pivotLocal.z, _tipLocal.z);
            float centerX = (_pivotLocal.x + _tipLocal.x) * 0.5f;
            float centerY = (_pivotLocal.y + _tipLocal.y) * 0.5f;

            for (int c = 0; c < count; c++)
            {
                result.Add(new LocalCapsule
                {
                    a = new Vector3(centerX, centerY, Mathf.Lerp(zMin, zMax, c / (float)count)),
                    b = new Vector3(centerX, centerY, Mathf.Lerp(zMin, zMax, (c + 1) / (float)count)),
                    radius = capsuleRadius
                });
            }
        }

        void FitShaftCapsuleFromCollisionMesh()
        {
            if (_colVertsShaft == null || _colVertsShaft.Length == 0)
                return;

            Vector3 min = _colVertsShaft[0];
            Vector3 max = _colVertsShaft[0];
            for (int i = 1; i < _colVertsShaft.Length; i++)
            {
                min = Vector3.Min(min, _colVertsShaft[i]);
                max = Vector3.Max(max, _colVertsShaft[i]);
            }

            float centerX = (min.x + max.x) * 0.5f;
            float centerY = (min.y + max.y) * 0.5f;
            _shaftEndLocal = new Vector3(centerX, centerY, max.z);
            _shaftTipLocal = new Vector3(centerX, centerY, min.z);

            Debug.Log($"[GripperTool] Shaft capsule fitted from collision mesh: " +
                      $"localZ=[{min.z:F3}, {max.z:F3}] center=({centerX:F3}, {centerY:F3}) radius={shaftRadius:F3}");
        }

        GameObject CreateMeshObj(string name, string objFile, Material mat)
        {
            var go = new GameObject(name);
            go.transform.SetParent(transform);
            var mf = go.AddComponent<MeshFilter>();
            var mr = go.AddComponent<MeshRenderer>();
            mr.material = mat;

            string path = Path.Combine(Application.streamingAssetsPath, objFile);
            if (File.Exists(path))
            {
                Mesh mesh = LoadOBJMesh(path);
                if (mesh != null)
                {
                    mf.mesh = mesh;
                    Debug.Log($"[GripperTool] Loaded {name}: {mesh.vertexCount}V");
                }
            }
            else
            {
                Debug.LogWarning($"[GripperTool] OBJ not found: {path}");
            }
            return go;
        }

        // ── 可视化更新 ───────────────────────────────────────
        void UpdateVisual()
        {
            Quaternion toolRot = Quaternion.Euler(0, _toolRotY, 0);

            if (_shaftObj != null)
            {
                _shaftObj.transform.position = _toolPos;
                _shaftObj.transform.rotation = toolRot;
                _shaftObj.transform.localScale = Vector3.one * modelScale;
            }

            if (_jawUpObj != null)
            {
                // 上颚：先铰接旋转（开合），再整体旋转+平移
                // 在 localScale=modelScale 下，铰接点是在 OBJ 原始坐标
                _jawUpObj.transform.localScale = Vector3.one * modelScale;
                _jawUpObj.transform.position = _toolPos;
                _jawUpObj.transform.rotation = toolRot;

                // 用 pivot 做开合：先移到 pivot，旋转，再移回
                // 简化：对整个 jaw mesh 做变换
                ApplyJawRotation(_jawUpObj, true, _currentAngle, toolRot);
            }

            if (_jawDownObj != null)
            {
                _jawDownObj.transform.localScale = Vector3.one * modelScale;
                _jawDownObj.transform.position = _toolPos;
                _jawDownObj.transform.rotation = toolRot;

                ApplyJawRotation(_jawDownObj, false, -_currentAngle, toolRot);
            }
        }

        void ApplyJawRotation(
            GameObject jawObj,
            bool isUpperJaw,
            float angle,
            Quaternion toolRot)
        {
            // 铰接点的世界位置
            Vector3 pivotWorld = toolRot * _pivotLocal + _toolPos;
            Vector3 selfAxisPointLocal = isUpperJaw
                ? _upperJawSelfAxisPointLocal
                : _lowerJawSelfAxisPointLocal;
            Vector3 selfAxisPointWorld = toolRot * selfAxisPointLocal + _toolPos;
            Quaternion jawMountRot = Quaternion.AngleAxis(jawRollAngle, Vector3.forward);

            // 先设置到工具位置
            jawObj.transform.position = _toolPos;
            jawObj.transform.rotation = toolRot;
            jawObj.transform.localScale = Vector3.one * modelScale;

            // 先绕各自中心轴自转，再绕杆轴安装旋转，最后绕安装后的铰接轴开合。
            jawObj.transform.RotateAround(
                selfAxisPointWorld, toolRot * Vector3.forward, jawSelfRollAngle);
            jawObj.transform.RotateAround(pivotWorld, toolRot * Vector3.forward, jawRollAngle);
            Vector3 hingeAxisWorld = toolRot * (jawMountRot * Vector3.right);
            jawObj.transform.RotateAround(pivotWorld, hingeAxisWorld, angle);
            jawObj.transform.position += toolRot * GetJawCenteringOffsetLocal(isUpperJaw);
        }

        // ── OBJ 解析 ─────────────────────────────────────────
        static Mesh LoadOBJMesh(string path)
        {
            LoadOBJData(path, out Vector3[] verts, out int[] tris);
            if (verts == null || verts.Length == 0) return null;
            var mesh = new Mesh { name = Path.GetFileNameWithoutExtension(path) };
            mesh.SetVertices(new List<Vector3>(verts));
            mesh.SetTriangles(tris, 0);
            mesh.RecalculateNormals();
            mesh.RecalculateBounds();
            return mesh;
        }

        static void LoadOBJData(string path, out Vector3[] verts, out int[] tris)
        {
            verts = null; tris = null;
            if (!File.Exists(path)) return;

            var vertList = new List<Vector3>();
            var triList  = new List<int>();
            foreach (string line in File.ReadAllLines(path))
            {
                string s = line.Trim();
                if (s.StartsWith("v "))
                {
                    var p = s.Split(new[]{' '}, System.StringSplitOptions.RemoveEmptyEntries);
                    if (p.Length >= 4)
                        vertList.Add(new Vector3(
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
                    { triList.Add(fv[0]); triList.Add(fv[i]); triList.Add(fv[i+1]); }
                }
            }
            verts = vertList.ToArray();
            tris = triList.ToArray();
        }

        void OnDestroy()
        {
            if (_jawUpObj) Destroy(_jawUpObj);
            if (_jawDownObj) Destroy(_jawDownObj);
            if (_shaftObj) Destroy(_shaftObj);
        }

        // ── Gizmo: 碰撞胶囊体可视化 ─────────────────────────
        void OnDrawGizmos()
        {
            if (!_initialized || !showCapsuleGizmo) return;

            for (int i = 0; i < _dbgCapsuleCount; i++)
            {
                if (i < _dbgUpperCapsuleCount)
                    Gizmos.color = new Color(0f, 1f, 1f, 0.5f);
                else if (i < _dbgUpperCapsuleCount + _dbgLowerCapsuleCount)
                    Gizmos.color = new Color(1f, 0f, 1f, 0.5f);
                else
                    Gizmos.color = new Color(0f, 1f, 0f, 0.3f);

                DrawCapsuleGizmo(_dbgCapsuleA[i], _dbgCapsuleB[i], _dbgCapsuleR[i]);
            }

            // AABB — 黄色线框
            if (_dbgCapsuleCount > 0)
            {
                Gizmos.color = Color.yellow;
                Vector3 center = (_dbgBBoxMin + _dbgBBoxMax) * 0.5f;
                Vector3 size   = _dbgBBoxMax - _dbgBBoxMin;
                Gizmos.DrawWireCube(center, size);
            }
        }

        static void DrawCapsuleGizmo(Vector3 a, Vector3 b, float r)
        {
            Gizmos.DrawWireSphere(a, r);
            Gizmos.DrawWireSphere(b, r);
            Gizmos.DrawLine(a, b);
            // 画几条平行线表示胶囊体轮廓
            Vector3 dir = (b - a).normalized;
            Vector3 up = Vector3.up;
            if (Mathf.Abs(Vector3.Dot(dir, up)) > 0.9f) up = Vector3.right;
            Vector3 side = Vector3.Cross(dir, up).normalized * r;
            Vector3 topDir = Vector3.Cross(dir, side).normalized * r;
            Gizmos.DrawLine(a + side, b + side);
            Gizmos.DrawLine(a - side, b - side);
            Gizmos.DrawLine(a + topDir, b + topDir);
            Gizmos.DrawLine(a - topDir, b - topDir);
        }

        // ── GUI ──────────────────────────────────────────────
        void OnGUI()
        {
            if (!_initialized) return;
            var style = new GUIStyle(GUI.skin.box) { fontSize = 12 };
            style.normal.textColor = Color.white;
            style.alignment = TextAnchor.UpperLeft;
            style.richText = true;

            string state = IsGrasping ? "<color=#FF4444>夹取中</color>" :
                          (_wantClose ? "<color=#FFFF00>闭合中</color>" : "张开");

            string info =
                $"夹爪: {state}\n" +
                $"角度: {_currentAngle:F1}°\n" +
                $"粒子: 硬核={_graspedParticles.Count} 动态过渡={_transitionParticleCount}\n" +
                $"胶囊: jaw={_dbgUpperCapsuleCount + _dbgLowerCapsuleCount} total={_dbgCapsuleCount}\n" +
                $"FGHVBN移动 1/3旋转 Z/X开合";
            GUI.Box(new Rect(10, 360, 260, 110), info, style);
        }
    }
}
