// CuttingToolV3.cs 鈥?Phase 1: 鎵帬闈㈣繛缁垏鍓叉帶鍒跺櫒
//
// 宸ヤ綔娴?
//   1. SurgicalTool 鎻愪緵鍒€鍒冪嚎娈电鐐?(BladeA, BladeB)
//   2. 纰拌Е妫€娴? 鍒€鍒冮檮杩戞槸鍚︽湁娲昏穬 tet
//   3. 灏嗗垁鍒冪鐐硅浆鎹㈠埌 mesh local space
//   4. 浼犻€掓壂鎺犲洓杈瑰舰 (A0,B0,A1,B1) 缁?TetSubdivisionCutter
//   5. FixedUpdate 涓?flush 鍒?GPU

using System.Collections.Generic;
using UnityEngine;
using SurgicalSim.Core;
using SurgicalSim.Cutting;
using SurgicalSim.Physics;
using SurgicalSim.Rendering;

namespace SurgicalSim.CuttingV3
{
    public class CuttingToolV3 : MonoBehaviour
    {
        [Header("Cutting Parameters")]
        [Tooltip("Tets inside this swept-blade radius are considered for cutting.")]
        public float cutRadius = 0.05f;
        public bool useFiniteSweptBladeFilter = false;

        [Header("Contact Detection")]
        [Tooltip("Radius around the blade used to detect tissue contact.")]
        public float contactRadius = 0.03f;

        [Header("杈撳叆")]
        public bool useSurgicalTool = true;
        public bool requireCutInput = true;
        public KeyCode cutKey = KeyCode.Space;
        public float minCutMoveDistance = 0.001f;
        public float contactExitPadding = 0.01f;
        public int contactExitDebounceFrames = 3;

        [Header("Pressure Gate")]
        public bool usePressureCutGate = true;
        public float bladeCollisionRadius = 0.0075f;
        public float maxPressureBladeRadius = 0.012f;
        [Tooltip("Active topology-cut blade length. Keep close to the real/contact blade to avoid distant cuts.")]
        public float pressureBladeLength = 0.66f;
        [Range(1f, 2f)] public float maxPressureBladeLengthMultiplier = 1.5f;
        public float pressureContactBladeLength = 0.44f;
        public bool pressureUseFiniteSweptBladeFilter = true;
        public bool useGpuFullSurfacePressureContact = true;
        [Range(0f, 1f)] public float cutPressureThreshold = 0.14f;
        [Range(0f, 1f)] public float sustainPressureThreshold = 0.06f;
        [Range(0f, 1f)] public float singleContactPressureThreshold = 0.14f;
        [Range(1, 12)] public int pressureArmFrames = 1;
        [Range(1, 12)] public int pressureReleaseFrames = 3;
        [Range(0.01f, 1f)] public float pressureEmaAlpha = 0.75f;
        public float tangentialCutThreshold = 0.00025f;
        public float dynamicTangentialSustainThreshold = 0.00012f;
        [Range(1, 12)] public int tangentialReleaseFrames = 3;
        [Range(1, 16)] public int minPressureContactCount = 1;
        [Range(0, 8)] public int maxPressureMetricsAgeFrames = 3;
        [Range(0, 8)] public int pressureContactGraceFrames = 4;
        [Tooltip("Minimum GPU contact depth for sharp-contact cut startup.")]
        public float sharpCutMinDepth = 0.0002f;

        [Header("Cut Trajectory")]
        [Range(0.01f, 1f)] public float surgicalBladeCenterEmaAlpha = 0.65f;
        [Tooltip("Maximum movement per surgical-tool Cut microstep. <= 0 uses cut radius.")]
        public float maxSurgicalCutMicroStepLength = 0.006f;
        [Range(1, 16)] public int maxSurgicalCutMicroStepsPerFrame = 8;
        public bool stabilizeKeyboardCutTrajectory = true;

        [Header("Multi-Stroke Seam Continuity")]
        [Tooltip("刀片长度有限、需要沿边缘分多刀切割时：若新一刀的起点与上一刀的终点距离不超过此半径，" +
                 "视为同一条逻辑切缝的延续，跨刀保留切割点缓存(_strokeSplitCache)，避免接缝处因重新插值产生" +
                 "几何错位而漏切（残桥）。建议设为 cutRadius 的 1~2 倍。")]
        public float strokeContinuityRadius = 0.06f;
        [Tooltip("两刀之间允许的最大时间间隔（秒），超过则不再视为连续，按全新切割重置缓存。")]
        public float strokeContinuityTimeWindow = 2.0f;
        public bool LastStrokeWasContinuation { get; private set; }

        [Header("鎬ц兘")]
        [Range(1, 10)]
        public int flushInterval = 2;
        // ★ 需求④修复：默认开启。RemoveStretchedTets 只清理几何上已确认退化的 sliver
        // (近零 rest 体积 / 拉伸比超过 5x，或 >10x 时无条件清除——视为拓扑错误)，
        // 不是按"分量大小"批量删除小块，因此不会误删被切下的薄组织，只清理真正的碎屑。
        // 之前默认 false 导致这套已经写好的碎屑清理机制从未在默认配置下生效。
        public bool enablePostCutCleanup = true;

        [Header("璋冭瘯")]
        public bool showDebugRay = true;
        public float maxVisibleCutCapEdgeLength = 0.025f;

        // 鈹€鈹€ 绉佹湁 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
        TetSubdivisionCutter _cutter;
        TetMeshData          _data;
        TetMeshVisualizer    _visualizer;
        XPBDSolverGPU        _solver;
        SurgicalTool         _surgicalTool;
        SofaUnityVisualLiverRenderer _sofaVisualRenderer;
        Camera               _cam;

        bool    _isCutting;
        bool    _wasInsideMesh;
        int     _framesSinceFlush;
        int     _postCutCleanupFrames;
        int     _outsideContactFrames;
        int     _lastCleanupRemovedTetCount;

        const int POST_CUT_CLEANUP_WINDOW = 8;

        // 涓婂抚鍒€鍒冪鐐?(local space)
        Vector3 _prevBladeA_local;
        Vector3 _prevBladeB_local;
        bool    _hasPrevBlade;
        Vector3 _uploadedBladeA_local;
        Vector3 _uploadedBladeB_local;
        bool    _hasUploadedBlade;
        Vector3 _smoothedCutBladeCenterLocal;
        bool    _hasSmoothedCutBladeCenter;
        Vector3 _pressureMotionBladeA_local;
        Vector3 _pressureMotionBladeB_local;
        bool    _hasPressureMotionBlade;
        Vector3 _pendingPressureCutTargetA_local;
        Vector3 _pendingPressureCutTargetB_local;
        bool    _hasPendingPressureCutTarget;
        bool    _pendingPressureCutInvalidatedByNoContact;
        bool    _pressureCutInvalidatedByNoContact;

        // ★ 多刀接缝连续性：记录"上一刀真正结束时"刀片中心位置/时间，供下一刀开始时判断
        // 是否应当跨刀保留 _strokeSplitCache（preserveContinuity），从而让接缝处复用同一批切割点。
        Vector3 _lastStrokeEndCenterLocal;
        bool    _hasLastStrokeEndCenter;
        float   _lastStrokeEndTime = -1f;

        enum PressureCutState { NoContact, Pressing, Armed, Cutting, Exit }
        PressureCutState _pressureState = PressureCutState.NoContact;
        float _pressureRaw;
        float _pressureEma;
        float _lastTangentMove;
        float _lastEffectivePressureBladeLength;
        bool  _lastCutAdvanced;
        int   _pressureArmCounter;
        int   _pressureReleaseCounter;
        int   _tangentReleaseCounter;
        int   _pressureContactGraceCounter;
        int   _lastMetricsVersion;
        int   _lastProcessedMetricsVersion;
        bool  _pressureSourceActive;
        bool  _pressureHasContact;
        Vector3 _pressureNormalLocal = Vector3.up;

        // 榧犳爣妯″紡
        Vector3 _prevHitPoint;
        bool    _hasHit;

        LineRenderer _debugLine;
        readonly List<int> _renderOriginalSurfaceTris = new List<int>(8192);
        readonly List<int> _renderCutTris = new List<int>(4096);
        readonly HashSet<long> _renderOriginalSurfaceKeys = new HashSet<long>();

        // 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲
        public void Init(TetMeshData data, XPBDSolverGPU solver,
                         TetMeshVisualizer visualizer)
        {
            _data       = data;
            _solver     = solver;
            _visualizer = visualizer;
            _cam        = Camera.main;

            _cutter = new TetSubdivisionCutter();
            _cutter.Init(data, solver);
            _framesSinceFlush = 0;
            _postCutCleanupFrames = 0;
            _outsideContactFrames = 0;
            _lastCleanupRemovedTetCount = 0;
            _wasInsideMesh    = false;
            _hasPrevBlade     = false;

            if (useSurgicalTool) EnsureSurgicalTool();
            if (showDebugRay)    SetupDebugLine();

            Debug.Log($"[CuttingToolV3] Init P:{data.NumParticles} T:{data.NumTets}");
        }

        // 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲
        void Update()
        {
            if (_data == null) return;
            if (useSurgicalTool) AutoCutStep();
            else                 MouseCutStep();
        }

        // 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲
        // 鑷姩纰拌Е鍒囧壊 鈥?鎵帬闈㈢増
        // 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲
        void AutoCutStep()
        {
            if (usePressureCutGate)
            {
                PressureAutoCutStep();
                return;
            }

            LegacyAutoCutStep();
        }

        void LegacyAutoCutStep()
        {
            if (_surgicalTool == null)
            {
                EnsureSurgicalTool();
                return;
            }

            Transform tf = _visualizer != null ? _visualizer.transform : transform;
            Vector3 bladeA_local = tf.InverseTransformPoint(_surgicalTool.BladeA);
            Vector3 bladeB_local = tf.InverseTransformPoint(_surgicalTool.BladeB);

            bool gateActive = IsCutGateActive();
            float moveThreshold = Mathf.Max(0f, minCutMoveDistance);
            bool contactEnter = IsBladeNearMesh(bladeA_local, bladeB_local, contactRadius);
            bool contactStay = _wasInsideMesh &&
                               IsBladeNearMesh(bladeA_local, bladeB_local,
                                               contactRadius + Mathf.Max(0f, contactExitPadding));

            if (!_wasInsideMesh)
            {
                if (!contactEnter) return;

                _wasInsideMesh = true;
                _outsideContactFrames = 0;
            }
            else if (contactStay)
            {
                _outsideContactFrames = 0;
            }
            else
            {
                _outsideContactFrames++;
                if (_outsideContactFrames >= Mathf.Max(1, contactExitDebounceFrames))
                {
                    if (_isCutting && _hasPrevBlade && gateActive)
                    {
                        float exitMoveA = (bladeA_local - _prevBladeA_local).magnitude;
                        float exitMoveB = (bladeB_local - _prevBladeB_local).magnitude;
                        if (Mathf.Max(exitMoveA, exitMoveB) >= moveThreshold)
                            RecordCutResult(_cutter.Cut(_prevBladeA_local, _prevBladeB_local,
                                                        bladeA_local, bladeB_local,
                                                        cutRadius, useFiniteSweptBladeFilter));
                    }

                    if (_hasPrevBlade) MarkStrokeEnd(_prevBladeA_local, _prevBladeB_local);
                    _isCutting = false;
                    _hasPrevBlade = false;
                    _wasInsideMesh = false;
                    _outsideContactFrames = 0;
                    return;
                }
            }

            if (!gateActive)
            {
                if (_hasPrevBlade) MarkStrokeEnd(_prevBladeA_local, _prevBladeB_local);
                _isCutting = false;
                _hasPrevBlade = false;
                return;
            }

            if (!_isCutting || !_hasPrevBlade)
            {
                bool continuation = ShouldPreserveStrokeContinuity((bladeA_local + bladeB_local) * 0.5f);
                LastStrokeWasContinuation = continuation;
                _isCutting = true;
                _prevBladeA_local = bladeA_local;
                _prevBladeB_local = bladeB_local;
                _hasPrevBlade = true;
                _cutter.ResetStroke(continuation);
                return;
            }

            float moveA = (bladeA_local - _prevBladeA_local).magnitude;
            float moveB = (bladeB_local - _prevBladeB_local).magnitude;
            float maxMove = Mathf.Max(moveA, moveB);

            if (maxMove < moveThreshold) return;

            RecordCutResult(_cutter.Cut(_prevBladeA_local, _prevBladeB_local,
                                        bladeA_local, bladeB_local,
                                        cutRadius, useFiniteSweptBladeFilter));

            _prevBladeA_local = bladeA_local;
            _prevBladeB_local = bladeB_local;
        }

        void PressureAutoCutStep()
        {
            if (_surgicalTool == null) { EnsureSurgicalTool(); return; }
            _lastCutAdvanced = false;

            Transform tf = _visualizer != null ? _visualizer.transform : transform;
            GetPressureBladeSegmentWorld(out Vector3 bladeA_world, out Vector3 bladeB_world);
            Vector3 bladeA_local = tf.InverseTransformPoint(bladeA_world);
            Vector3 bladeB_local = tf.InverseTransformPoint(bladeB_world);
            StabilizeKeyboardCutSegment(tf, ref bladeA_local, ref bladeB_local);
            ApplySurgicalBladeCenterSmoothing(ref bladeA_local, ref bladeB_local);

            XPBDSolverGPU.ToolContactMetricsSnapshot metrics =
                _solver != null ? _solver.LastToolContactMetrics : default;
            bool newMetrics = metrics.Version != _lastProcessedMetricsVersion;
            if (newMetrics)
            {
                _lastProcessedMetricsVersion = metrics.Version;
                _lastMetricsVersion = metrics.Version;
                UpdatePressureSample(metrics);
                if (_pressureState == PressureCutState.Cutting && metrics.ContactCount <= 0)
                    InvalidatePressureCutForNoContact();
            }

            bool gateActive = IsCutGateActive();
            int metricsAge = _solver != null ? _solver.ToolContactMetricsAge : int.MaxValue;
            bool metricsFresh = metricsAge <= Mathf.Max(0, maxPressureMetricsAgeFrames);
            bool hasSharpContact = _pressureSourceActive &&
                                   metricsFresh &&
                                   metrics.ContactCount > 0 &&
                                   metrics.MaxDepth >= Mathf.Max(0f, sharpCutMinDepth);
            bool hasPressureSource = _pressureSourceActive &&
                                     metricsFresh &&
                                     (_pressureHasContact || hasSharpContact);
            bool maintainPressure = hasPressureSource && _pressureEma >= Mathf.Clamp01(sustainPressureThreshold);
            bool contactSustain = hasPressureSource && hasSharpContact;
            bool startPressure = hasPressureSource &&
                                 (_pressureEma >= Mathf.Clamp01(cutPressureThreshold) || hasSharpContact);
            UpdatePressureContactGrace(hasPressureSource || hasSharpContact);
            bool graceKeepsState =
                _pressureContactGraceCounter > 0 &&
                _pressureState != PressureCutState.NoContact &&
                _pressureState != PressureCutState.Exit;
            bool feedbackPressureSource = hasPressureSource || graceKeepsState;
            bool statePressureSource = hasPressureSource;
            bool stateMaintainPressure = maintainPressure || contactSustain;

            if (!gateActive || !_pressureSourceActive)
            {
                UpdateContactProxyFeedback(tf, false);
                ResetPressureCutState(PressureCutState.NoContact);
                return;
            }

            Vector3 center = (bladeA_local + bladeB_local) * 0.5f;
            Vector3 prevCenter = _hasPressureMotionBlade
                ? (_pressureMotionBladeA_local + _pressureMotionBladeB_local) * 0.5f
                : center;
            Vector3 n = _pressureNormalLocal.sqrMagnitude > 1e-8f
                ? _pressureNormalLocal.normalized
                : Vector3.up;
            Vector3 delta = center - prevCenter;
            Vector3 tangent = delta - Vector3.Dot(delta, n) * n;
            _lastTangentMove = tangent.magnitude;

            bool tangentReady = _lastTangentMove >= Mathf.Max(0f, tangentialCutThreshold);
            bool tangentSustainReady = _lastTangentMove >= Mathf.Max(0f, dynamicTangentialSustainThreshold);

            if (_pressureState == PressureCutState.Cutting && graceKeepsState && !hasPressureSource)
            {
                UpdateContactProxyFeedback(tf, true);
                return;
            }

            if (_pressureState == PressureCutState.Cutting &&
                _pressureCutInvalidatedByNoContact &&
                hasPressureSource)
            {
                ReanchorPressureCutAfterNoContact(bladeA_local, bladeB_local);
                UpdatePressureMotionBaseline(bladeA_local, bladeB_local);
                UpdateContactProxyFeedback(tf, true);
                return;
            }

            if (_pressureState == PressureCutState.Cutting && _hasPendingPressureCutTarget)
            {
                bool canDrainPending = hasPressureSource &&
                                       !_pendingPressureCutInvalidatedByNoContact &&
                                       (maintainPressure || contactSustain);
                if (canDrainPending)
                {
                    _pressureReleaseCounter = 0;
                    DrainPendingPressureCutSegment();
                    UpdateContactProxyFeedback(tf, true);
                    return;
                }
            }

            UpdatePressureMotionBaseline(bladeA_local, bladeB_local);

            if (newMetrics)
                AdvancePressureStateForSample(statePressureSource, stateMaintainPressure, hasSharpContact);

            switch (_pressureState)
            {
                case PressureCutState.NoContact:
                case PressureCutState.Pressing:
                case PressureCutState.Exit:
                    UpdateContactProxyFeedback(tf, feedbackPressureSource);
                    return;

                case PressureCutState.Armed:
                    if (startPressure && tangentReady)
                        StartPressureCut(bladeA_local, bladeB_local);
                    UpdateContactProxyFeedback(tf, feedbackPressureSource);
                    return;

                case PressureCutState.Cutting:
                    if (!tangentSustainReady)
                    {
                        _tangentReleaseCounter++;
                        if (_tangentReleaseCounter >= Mathf.Max(1, tangentialReleaseFrames))
                            ResetPressureCutState(PressureCutState.Exit);
                        UpdateContactProxyFeedback(tf, feedbackPressureSource);
                        return;
                    }

                    _tangentReleaseCounter = 0;
                    if (maintainPressure || contactSustain)
                        ConsumePressureCutSegment(bladeA_local, bladeB_local);
                    UpdateContactProxyFeedback(tf, feedbackPressureSource);
                    return;
            }
        }

        void UpdateContactProxyFeedback(Transform meshTransform, bool active)
        {
            if (_surgicalTool == null) return;
            if (!active)
            {
                _surgicalTool.SetContactProxyFeedback(false, Vector3.up, 0f, false, false);
                return;
            }

            Vector3 normalWorld = meshTransform != null
                ? meshTransform.TransformDirection(_pressureNormalLocal)
                : _pressureNormalLocal;
            _surgicalTool.SetContactProxyFeedback(
                true,
                normalWorld,
                _pressureEma,
                _pressureState == PressureCutState.Cutting,
                _lastCutAdvanced);
        }

        void UpdatePressureSample(XPBDSolverGPU.ToolContactMetricsSnapshot metrics)
        {
            float denom = _solver != null
                ? Mathf.Max(1e-6f, _solver.ToolContactDistance)
                : Mathf.Max(1e-6f, contactRadius);
            int minContacts = Mathf.Max(1, minPressureContactCount);
            float depthPressure = Mathf.Clamp01(metrics.MaxDepth / denom);
            bool hasAnyContact = metrics.ContactCount > 0 && metrics.MaxDepth > 0f;
            bool hasSharpDepth = hasAnyContact && metrics.MaxDepth >= Mathf.Max(0f, sharpCutMinDepth);
            bool strongSingleContact = hasAnyContact &&
                                       depthPressure >= Mathf.Clamp01(singleContactPressureThreshold);
            int effectiveMinContacts = strongSingleContact ? 1 : minContacts;
            float contactWeight = Mathf.Clamp01(metrics.ContactCount / (float)effectiveMinContacts);
            _pressureRaw = depthPressure * contactWeight;
            _pressureEma = Mathf.Lerp(_pressureEma, _pressureRaw, Mathf.Clamp01(pressureEmaAlpha));
            _pressureHasContact = hasAnyContact &&
                                  (metrics.ContactCount >= minContacts || strongSingleContact || hasSharpDepth);

            if (metrics.NormalReliability > 0.05f && metrics.NormalLocal.sqrMagnitude > 1e-8f)
                _pressureNormalLocal = metrics.NormalLocal.normalized;
        }

        void AdvancePressureStateForSample(bool hasPressureSource, bool maintainPressure, bool hasSharpContact)
        {
            bool cutPressure = hasPressureSource &&
                               (_pressureEma >= Mathf.Clamp01(cutPressureThreshold) || hasSharpContact);
            if (!maintainPressure)
                _pressureReleaseCounter++;
            else
                _pressureReleaseCounter = 0;

            switch (_pressureState)
            {
                case PressureCutState.NoContact:
                case PressureCutState.Exit:
                    _pressureArmCounter = 0;
                    if (hasPressureSource)
                    {
                        _pressureArmCounter = cutPressure ? 1 : 0;
                        _pressureState = _pressureArmCounter >= Mathf.Max(1, pressureArmFrames)
                            ? PressureCutState.Armed
                            : PressureCutState.Pressing;
                        _pressureReleaseCounter = 0;
                    }
                    break;

                case PressureCutState.Pressing:
                    if (_pressureReleaseCounter >= Mathf.Max(1, pressureReleaseFrames))
                    {
                        ResetPressureCutState(PressureCutState.NoContact);
                        break;
                    }

                    _pressureArmCounter = cutPressure ? _pressureArmCounter + 1 : 0;
                    if (_pressureArmCounter >= Mathf.Max(1, pressureArmFrames))
                        _pressureState = PressureCutState.Armed;
                    break;

                case PressureCutState.Armed:
                case PressureCutState.Cutting:
                    if (_pressureReleaseCounter >= Mathf.Max(1, pressureReleaseFrames))
                        ResetPressureCutState(PressureCutState.Exit);
                    break;
            }
        }

        // 是否应把"即将开始的这一刀"当作上一刀的延续（跨刀保留 _strokeSplitCache）。
        // 仅当：1) 上一刀确实留下过结束位置；2) 时间间隔在窗口内；3) 新起点与上一刀终点的距离在容差半径内。
        bool ShouldPreserveStrokeContinuity(Vector3 newCenterLocal)
        {
            if (!_hasLastStrokeEndCenter) return false;
            if (Time.time - _lastStrokeEndTime > Mathf.Max(0.01f, strokeContinuityTimeWindow)) return false;
            float r = Mathf.Max(0.0001f, strokeContinuityRadius);
            return (newCenterLocal - _lastStrokeEndCenterLocal).sqrMagnitude <= r * r;
        }

        // 记录"这一刀真正结束时"的刀片中心位置，供下一刀判断是否为同一条逻辑切缝的延续。
        void MarkStrokeEnd(Vector3 a, Vector3 b)
        {
            _lastStrokeEndCenterLocal = (a + b) * 0.5f;
            _hasLastStrokeEndCenter = true;
            _lastStrokeEndTime = Time.time;
        }

        void StartPressureCut(Vector3 bladeA_local, Vector3 bladeB_local)
        {
            bool continuation = ShouldPreserveStrokeContinuity((bladeA_local + bladeB_local) * 0.5f);
            LastStrokeWasContinuation = continuation;
            _cutter.ResetStroke(continuation);
            _isCutting = true;
            _pressureState = PressureCutState.Cutting;
            _tangentReleaseCounter = 0;
            _prevBladeA_local = bladeA_local;
            _prevBladeB_local = bladeB_local;
            _hasPrevBlade = true;
            _lastCutAdvanced = false;
            _pressureCutInvalidatedByNoContact = false;
            ClearPendingPressureCutTarget();
        }

        void ConsumePressureCutSegment(Vector3 bladeA_local, Vector3 bladeB_local)
        {
            if (!_hasPrevBlade)
            {
                _prevBladeA_local = bladeA_local;
                _prevBladeB_local = bladeB_local;
                _hasPrevBlade = true;
                ClearPendingPressureCutTarget();
                return;
            }

            TetSubdivisionCutter.CutResult result = CutSurgicalBladeSegment(
                _prevBladeA_local, _prevBladeB_local,
                bladeA_local, bladeB_local,
                EffectivePressureCutRadius,
                EffectivePressureUseFiniteSweptBladeFilter,
                out Vector3 consumedA,
                out Vector3 consumedB,
                out bool consumedTarget);
            RecordCutResult(result);
            _prevBladeA_local = consumedA;
            _prevBladeB_local = consumedB;
            _hasPrevBlade = true;

            if (consumedTarget)
                ClearPendingPressureCutTarget();
            else
                SetPendingPressureCutTarget(bladeA_local, bladeB_local);
        }

        void DrainPendingPressureCutSegment()
        {
            if (!_hasPendingPressureCutTarget)
                return;

            ConsumePressureCutSegment(
                _pendingPressureCutTargetA_local,
                _pendingPressureCutTargetB_local);
        }

        void ResetPressureCutState(PressureCutState nextState)
        {
            // ★ 在真正清掉 _hasPrevBlade 之前，记下这一刀结束的位置——
            // 如果下一刀很快在附近重新开始（刀片长度有限，需要分多刀沿边缘切割），
            // 就能识别为同一条逻辑切缝的延续，跨刀保留切割点缓存。
            if (_hasPrevBlade)
                MarkStrokeEnd(_prevBladeA_local, _prevBladeB_local);

            _pressureState = nextState;
            _pressureArmCounter = 0;
            _pressureReleaseCounter = 0;
            _tangentReleaseCounter = 0;
            _pressureContactGraceCounter = 0;
            _pressureCutInvalidatedByNoContact = false;
            _isCutting = false;
            _hasPrevBlade = false;
            ResetSurgicalCutSmoothing();
            ResetPressureMotionBaseline();
            ClearPendingPressureCutTarget();
        }

        public void SetPressureSourceActive(bool active)
        {
            _pressureSourceActive = active;
            if (active)
                return;

            if (_surgicalTool != null)
                _surgicalTool.SetContactProxyFeedback(false, Vector3.up, 0f, false, false);
            _pressureRaw = 0f;
            _pressureEma = 0f;
            _lastTangentMove = 0f;
            _pressureHasContact = false;
            _pressureNormalLocal = Vector3.up;
            _hasUploadedBlade = false;
            _uploadedBladeA_local = Vector3.zero;
            _uploadedBladeB_local = Vector3.zero;
            ResetSurgicalCutSmoothing();
            ResetPressureCutState(PressureCutState.NoContact);
        }

        public bool UploadBladeCollisionToGPU()
        {
            if (_solver == null)
            {
                SetPressureSourceActive(false);
                return false;
            }
            if (_surgicalTool == null) EnsureSurgicalTool();
            if (_surgicalTool == null)
            {
                SetPressureSourceActive(false);
                return false;
            }

            Transform tf = _visualizer != null ? _visualizer.transform : transform;
            GetPressureContactBladeSegmentWorld(out Vector3 bladeA_world, out Vector3 bladeB_world);
            Vector3 bladeA = tf.InverseTransformPoint(bladeA_world);
            Vector3 bladeB = tf.InverseTransformPoint(bladeB_world);
            Vector3 prevA = _hasUploadedBlade ? _uploadedBladeA_local : bladeA;
            Vector3 prevB = _hasUploadedBlade ? _uploadedBladeB_local : bladeB;
            float radius = EffectivePressureBladeRadius;

            _solver.SetCapsuleCollisionParams(
                bladeA, bladeB, radius,
                Vector3.zero, Vector3.zero, 0f,
                Vector3.zero, Vector3.zero, 0f,
                prevA, prevB,
                Vector3.zero, Vector3.zero,
                Vector3.zero, Vector3.zero,
                1);

            _uploadedBladeA_local = bladeA;
            _uploadedBladeB_local = bladeB;
            _hasUploadedBlade = true;
            _pressureSourceActive = true;
            return true;
        }

        void GetPressureBladeSegmentWorld(out Vector3 bladeA, out Vector3 bladeB)
        {
            GetBladeSegmentFromTipWorld(EffectivePressureBladeLength, out bladeA, out bladeB);
        }

        void GetPressureContactBladeSegmentWorld(out Vector3 bladeA, out Vector3 bladeB)
        {
            GetBladeSegmentFromTipWorld(EffectivePressureContactBladeLength, out bladeA, out bladeB);
        }

        void GetBladeSegmentFromTipWorld(float length, out Vector3 bladeA, out Vector3 bladeB)
        {
            bladeB = _surgicalTool != null ? _surgicalTool.PhysicalBladeB : transform.position;
            Vector3 fullA = _surgicalTool != null
                ? _surgicalTool.PhysicalBladeA
                : bladeB - Vector3.down * Mathf.Max(0.02f, length);
            Vector3 fullDir = bladeB - fullA;
            float fullLen = fullDir.magnitude;
            Vector3 dir = fullLen > 1e-6f ? fullDir / fullLen : Vector3.down;
            bladeA = bladeB - dir * Mathf.Max(0.02f, length);
        }

        void StabilizeKeyboardCutSegment(Transform meshTransform, ref Vector3 bladeA_local, ref Vector3 bladeB_local)
        {
            if (!stabilizeKeyboardCutTrajectory ||
                _surgicalTool == null ||
                !_surgicalTool.HasMoveInput ||
                !_hasPrevBlade)
                return;

            Vector3 intentLocal = meshTransform != null
                ? meshTransform.InverseTransformDirection(_surgicalTool.LastMoveInputWorld)
                : _surgicalTool.LastMoveInputWorld;
            if (intentLocal.sqrMagnitude < 1e-8f)
                return;
            intentLocal.Normalize();

            Vector3 prevCenter = (_prevBladeA_local + _prevBladeB_local) * 0.5f;
            Vector3 center = (bladeA_local + bladeB_local) * 0.5f;
            Vector3 delta = center - prevCenter;
            float along = Vector3.Dot(delta, intentLocal);
            if (along <= 1e-8f)
                return;

            Vector3 stabilizedCenter = prevCenter + intentLocal * along;
            Vector3 offset = stabilizedCenter - center;
            bladeA_local += offset;
            bladeB_local += offset;
        }

        void ApplySurgicalBladeCenterSmoothing(ref Vector3 bladeA_local, ref Vector3 bladeB_local)
        {
            Vector3 center = (bladeA_local + bladeB_local) * 0.5f;
            if (!_hasSmoothedCutBladeCenter)
            {
                _smoothedCutBladeCenterLocal = center;
                _hasSmoothedCutBladeCenter = true;
                return;
            }

            _smoothedCutBladeCenterLocal = Vector3.Lerp(
                _smoothedCutBladeCenterLocal,
                center,
                Mathf.Clamp01(surgicalBladeCenterEmaAlpha));
            Vector3 offset = _smoothedCutBladeCenterLocal - center;
            bladeA_local += offset;
            bladeB_local += offset;
        }

        void ResetSurgicalCutSmoothing()
        {
            _hasSmoothedCutBladeCenter = false;
            _smoothedCutBladeCenterLocal = Vector3.zero;
        }

        void UpdatePressureMotionBaseline(Vector3 bladeA_local, Vector3 bladeB_local)
        {
            _pressureMotionBladeA_local = bladeA_local;
            _pressureMotionBladeB_local = bladeB_local;
            _hasPressureMotionBlade = true;
        }

        void ResetPressureMotionBaseline()
        {
            _hasPressureMotionBlade = false;
            _pressureMotionBladeA_local = Vector3.zero;
            _pressureMotionBladeB_local = Vector3.zero;
            _lastTangentMove = 0f;
        }

        void UpdatePressureContactGrace(bool hasCurrentContact)
        {
            int graceFrames = Mathf.Max(0, pressureContactGraceFrames);
            if (hasCurrentContact)
                _pressureContactGraceCounter = graceFrames;
            else if (_pressureContactGraceCounter > 0)
                _pressureContactGraceCounter--;
        }

        void SetPendingPressureCutTarget(Vector3 bladeA_local, Vector3 bladeB_local)
        {
            _pendingPressureCutTargetA_local = bladeA_local;
            _pendingPressureCutTargetB_local = bladeB_local;
            _hasPendingPressureCutTarget = true;
            _pendingPressureCutInvalidatedByNoContact = false;
        }

        void ClearPendingPressureCutTarget()
        {
            _hasPendingPressureCutTarget = false;
            _pendingPressureCutInvalidatedByNoContact = false;
            _pendingPressureCutTargetA_local = Vector3.zero;
            _pendingPressureCutTargetB_local = Vector3.zero;
        }

        void InvalidatePressureCutForNoContact()
        {
            _pressureCutInvalidatedByNoContact = true;
            if (_hasPendingPressureCutTarget)
                _pendingPressureCutInvalidatedByNoContact = true;
        }

        void ReanchorPressureCutAfterNoContact(Vector3 bladeA_local, Vector3 bladeB_local)
        {
            // ★ 这里发生时 _pressureState 仍然是 Cutting——只是接触短暂丢失又恢复（噪声/抖动），
            // 按状态机自身的定义这从来都不是"新的一刀"，因此永远按 preserveContinuity=true 处理，
            // 绝不能在这里整体清空 _strokeSplitCache。用户截图里 HUD 显示的
            // "reason: pressure_no_contact" 正是这条路径被频繁触发的直接证据，
            // 之前这里无条件硬重置，几乎可以肯定是"刀缝看似接上却切不透"最常见的触发点。
            _cutter.ResetStroke(true);
            LastStrokeWasContinuation = true;
            _prevBladeA_local = bladeA_local;
            _prevBladeB_local = bladeB_local;
            _hasPrevBlade = true;
            _lastCutAdvanced = false;
            _pressureCutInvalidatedByNoContact = false;
            ClearPendingPressureCutTarget();
        }

        TetSubdivisionCutter.CutResult CutSurgicalBladeSegment(
            Vector3 fromA,
            Vector3 fromB,
            Vector3 toA,
            Vector3 toB,
            float bladeRadius,
            bool finiteSweptBladeFilter,
            out Vector3 consumedA,
            out Vector3 consumedB,
            out bool consumedTarget)
        {
            TetSubdivisionCutter.CutResult total = default;
            consumedA = fromA;
            consumedB = fromB;
            consumedTarget = false;
            if (_cutter == null)
                return total;

            float maxMove = Mathf.Max((toA - fromA).magnitude, (toB - fromB).magnitude);
            float stepLength = EffectiveSurgicalCutMicroStepLength(bladeRadius);
            if (maxMove <= 1e-6f)
            {
                consumedA = toA;
                consumedB = toB;
                consumedTarget = true;
                return total;
            }

            int maxSteps = Mathf.Max(1, maxSurgicalCutMicroStepsPerFrame);
            float maxConsumedMove = stepLength * maxSteps;
            float consumeT = Mathf.Min(1f, maxConsumedMove / maxMove);
            consumedTarget = consumeT >= 1f - 1e-5f;
            Vector3 targetA = Vector3.Lerp(fromA, toA, consumeT);
            Vector3 targetB = Vector3.Lerp(fromB, toB, consumeT);
            float consumedMove = Mathf.Max((targetA - fromA).magnitude, (targetB - fromB).magnitude);
            int steps = Mathf.Clamp(
                Mathf.CeilToInt(consumedMove / Mathf.Max(1e-6f, stepLength)),
                1,
                maxSteps);

            Vector3 prevA = fromA;
            Vector3 prevB = fromB;
            for (int step = 1; step <= steps; step++)
            {
                float t = step / (float)steps;
                Vector3 nextA = Vector3.Lerp(fromA, targetA, t);
                Vector3 nextB = Vector3.Lerp(fromB, targetB, t);
                AccumulateCutResult(
                    ref total,
                    _cutter.Cut(prevA, prevB, nextA, nextB, bladeRadius, finiteSweptBladeFilter));
                prevA = nextA;
                prevB = nextB;
            }

            consumedA = targetA;
            consumedB = targetB;
            return total;
        }

        float EffectiveSurgicalCutMicroStepLength(float bladeRadius)
        {
            if (maxSurgicalCutMicroStepLength > 0f)
                return Mathf.Max(0.0005f, maxSurgicalCutMicroStepLength);

            return Mathf.Max(0.0005f, Mathf.Max(bladeRadius, cutRadius) * 0.75f);
        }

        static void AccumulateCutResult(
            ref TetSubdivisionCutter.CutResult total,
            TetSubdivisionCutter.CutResult add)
        {
            total.newVerts += add.newVerts;
            total.newTets += add.newTets;
            total.cutTets += add.cutTets;
            total.elapsedMs += add.elapsedMs;
        }

        bool IsCutGateActive()
        {
            return !requireCutInput ||
                   Input.GetKey(cutKey) ||
                   (_surgicalTool != null && _surgicalTool.IsCutting);
        }

        bool IsBladeNearMesh(Vector3 bladeA, Vector3 bladeB, float radius)
        {
            // 妫€娴嬪垁鍒冪嚎娈甸檮杩戞槸鍚︽湁 tet 椤剁偣
            // 涓嶈浣跨敤 mf.mesh.bounds, 鍥犱负鍦ㄨ蒋浣撳舰鍙樻椂 Unity 鐨?Bounds 鍙兘浼氱紦瀛樿繃鏃舵暟鎹紝瀵艰嚧鍒囧埌涓€鍗婂仠姝紒
            radius = Mathf.Max(0f, radius);
            float r2 = radius * radius;
            Vector3 bladeMin = Vector3.Min(bladeA, bladeB) - Vector3.one * radius;
            Vector3 bladeMax = Vector3.Max(bladeA, bladeB) + Vector3.one * radius;
            for (int t = 0; t < _data.NumTets; t++)
            {
                if (!_data.TetActive[t]) continue;
                int b = t * 4;
                Vector3 v0 = _data.Positions[_data.TetIds[b]];
                Vector3 v1 = _data.Positions[_data.TetIds[b+1]];
                Vector3 v2 = _data.Positions[_data.TetIds[b+2]];
                Vector3 v3 = _data.Positions[_data.TetIds[b+3]];
                Vector3 tetMin = Vector3.Min(Vector3.Min(v0, v1), Vector3.Min(v2, v3));
                Vector3 tetMax = Vector3.Max(Vector3.Max(v0, v1), Vector3.Max(v2, v3));

                if (tetMin.x > bladeMax.x || tetMax.x < bladeMin.x ||
                    tetMin.y > bladeMax.y || tetMax.y < bladeMin.y ||
                    tetMin.z > bladeMax.z || tetMax.z < bladeMin.z)
                    continue;

                // 鍙鏈変换浣曚竴涓《鐐瑰埌绾挎璺濈灏忎簬 r2
                if (PointSegDistSq(v0, bladeA, bladeB) < r2 ||
                    PointSegDistSq(v1, bladeA, bladeB) < r2 ||
                    PointSegDistSq(v2, bladeA, bladeB) < r2 ||
                    PointSegDistSq(v3, bladeA, bladeB) < r2)
                    return true;
            }
            return false;
        }

        static float PointSegDistSq(Vector3 p, Vector3 a, Vector3 b)
        {
            Vector3 ab = b - a;
            float sq = ab.sqrMagnitude;
            if (sq < 1e-12f) return (p - a).sqrMagnitude;
            float t = Mathf.Clamp01(Vector3.Dot(p - a, ab) / sq);
            return (p - (a + ab * t)).sqrMagnitude;
        }

        // 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲
        // 榧犳爣鍒囧壊 (淇濈暀, 鏈敼涓烘壂鎺犻潰)
        // 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲
        void MouseCutStep()
        {
            // TODO: Phase 2 鈥?mouse鍒囧壊涔熸敼涓烘壂鎺犻潰
        }

        // 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲
        // FlushCutToGPU 鈥?SoftBody.FixedUpdate 璋冪敤
        // 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲
        void RecordCutResult(TetSubdivisionCutter.CutResult result)
        {
            _lastCutAdvanced =
                result.cutTets > 0 ||
                result.newVerts > 0 ||
                result.newTets > 0;

            if (enablePostCutCleanup && result.cutTets > 0)
                _postCutCleanupFrames = POST_CUT_CLEANUP_WINDOW;
        }

        public void FlushCutToGPU()
        {
            if (_data == null || _solver == null || _cutter == null) return;

            bool cleanupActive = enablePostCutCleanup && _postCutCleanupFrames > 0;
            if (!_cutter.IsDirty && !cleanupActive) return;

            if (cleanupActive)
                _postCutCleanupFrames--;

            if (_cutter.IsDirty)
            {
                _framesSinceFlush++;
                if (_framesSinceFlush < flushInterval)
                {
                    if (cleanupActive)
                        _lastCleanupRemovedTetCount = _cutter.RemoveStretchedTets(5.0f, 0.003f);
                    return;
                }
            }
            else
            {
                _framesSinceFlush = 0;
            }

            // Cleanup-only passes should not rebuild GPU buffers unless they delete tets.
            if (cleanupActive || _cutter.IsDirty)
                _lastCleanupRemovedTetCount = cleanupActive
                    ? _cutter.RemoveStretchedTets(5.0f, 0.003f)
                    : 0;

            if (!_cutter.IsDirty) return;

            _cutter.ReadbackFromGPU();
            SurfaceBuildResult surface = UpdateSurface();
            _cutter.ReinitializeGPU();
            ApplyVisibleSurface(surface);
            _framesSinceFlush = 0;
        }

        // 鈹€鈹€ 璇婃柇 GUI 灞炴€?鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
        public int    TotalSplitVerts         => _cutter?.TotalNewVerts         ?? 0;
        public int    TotalCutEdges           => _cutter?.TotalNewTets          ?? 0;
        public int    TotalCutTets            => _cutter?.TotalCutTets          ?? 0;
        public bool   ToolCutPressed          => _isCutting;
        public float  LastToolMoveDistance    => _cutter?.LastMoveDistance      ?? 0f;
        public int    LastCandidateTetCount   => _cutter?.LastCandidateTetCount ?? 0;
        public int    LastIntersectedTetCount => _cutter?.LastIntersectedTetCount ?? 0;
        public int    LastSeparatedVertexCount => _cutter?.LastSeparatedVertexCount ?? 0;
        public int    LastForcedSeparationCount => _cutter?.LastForcedSeparationCount ?? 0;
        public int    LastResidualSharedVertexCount => _cutter?.LastResidualSharedVertexCount ?? 0;
        public int    LastSeparationPassCount => _cutter?.LastSeparationPassCount ?? 0;
        public int    LastProtectedTetCount   => _cutter?.LastProtectedTetCount ?? 0;
        public int    LastSkippedCurrentPhasePivotCount => _cutter?.LastSkippedCurrentPhasePivotCount ?? 0;
        // 迭代2.1：全局连通分量审计（2024 AFCC + 2025 shared-vertex）
        public int    LastComponentCount          => _cutter?.LastComponentCount ?? 0;
        public int    LastSeparatedComponentCount => _cutter?.LastSeparatedComponentCount ?? 0;
        public int    LastIndependentBlockCount   => _cutter?.LastIndependentBlockCount ?? 0;
        public int    LastFragmentComponentCount  => _cutter?.LastFragmentComponentCount ?? 0;
        public int    LastResidualBridgeCount     => _cutter?.LastResidualBridgeCount ?? 0;
        public int    LastLargestComponentTetCount => _cutter?.LastLargestComponentTetCount ?? 0;
        public int    LastCleanupRemovedTetCount => _lastCleanupRemovedTetCount;
        public string LastCutRejectReason =>
            usePressureCutGate && _pressureSourceActive && _pressureState == PressureCutState.NoContact
                ? "pressure_no_contact"
                : (_cutter?.LastRejectReason ?? "no_cutter");
        public string PressureState           => _pressureState.ToString();
        public float  PressureRaw             => _pressureRaw;
        public float  PressureEma             => _pressureEma;
        public float  LastTangentMove         => _lastTangentMove;
        public float  LastProxyLag            => _surgicalTool != null ? _surgicalTool.LastProxyLag : 0f;
        public float  LastProxyNormalCorrection => _surgicalTool != null ? _surgicalTool.LastProxyNormalCorrection : 0f;
        public float  LastProxyFrictionScale  => _surgicalTool != null ? _surgicalTool.LastProxyFrictionScale : 1f;
        public bool   PressureSourceActive    => _pressureSourceActive;
        public int    LastPressureMetricsVersion => _lastMetricsVersion;
        public float  EffectivePressureBladeLength
        {
            get
            {
                float configured = Mathf.Max(0.02f, pressureBladeLength);
                float len = configured;
                if (_surgicalTool != null)
                {
                    float physical = Mathf.Max(0.02f, _surgicalTool.EffectiveBladeLength);
                    float contact = EffectivePressureContactBladeLength;
                    float reference = Mathf.Max(physical, contact);
                    float maxActiveLength = reference * Mathf.Max(1f, maxPressureBladeLengthMultiplier);
                    len = Mathf.Clamp(configured, Mathf.Min(physical, contact), maxActiveLength);
                }
                _lastEffectivePressureBladeLength = Mathf.Max(0.02f, len);
                return _lastEffectivePressureBladeLength;
            }
        }
        public float  EffectivePressureContactBladeLength
        {
            get
            {
                float configured = Mathf.Max(0.02f, pressureContactBladeLength);
                float physical = _surgicalTool != null
                    ? Mathf.Max(0.02f, _surgicalTool.PhysicalBladeLength)
                    : configured;
                return Mathf.Clamp(configured, 0.02f, physical);
            }
        }
        public float  EffectivePressureBladeRadius
        {
            get
            {
                float radius = Mathf.Max(0.0001f, bladeCollisionRadius);
                if (_surgicalTool != null)
                    radius = Mathf.Max(radius, _surgicalTool.PhysicalBladeRadius);
                return Mathf.Min(radius, Mathf.Max(0.0001f, maxPressureBladeRadius));
            }
        }
        public float  EffectivePressureCutRadius =>
            Mathf.Clamp(Mathf.Max(cutRadius, EffectivePressureBladeRadius, 0.012f), 0.006f, 0.015f);
        public float  EffectiveVisibleCutCapMaxEdgeLength
        {
            get
            {
                float configured = maxVisibleCutCapEdgeLength > 0f
                    ? maxVisibleCutCapEdgeLength
                    : float.PositiveInfinity;
                float adaptive = Mathf.Max(2.5f * EffectivePressureCutRadius, 0.018f);
                return Mathf.Min(configured, adaptive);
            }
        }
        public bool   EffectivePressureUseFiniteSweptBladeFilter =>
            usePressureCutGate ? pressureUseFiniteSweptBladeFilter : useFiniteSweptBladeFilter;

        // 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲
        SurfaceBuildResult UpdateSurface()
        {
            if (_data == null) return default;
            // SurfaceReconstructor 鑷姩妫€娴嬪垏鍙ｉ潰 (count==1 杈圭晫闈?
            int[] originalSurfaceTris = HasOriginalSurfaceSupportData()
                ? SurfaceReconstructor.RebuildOriginalSurfaceFromSupports(
                    _data,
                    _cutter.SurfaceSupport0,
                    _cutter.SurfaceSupport1,
                    _cutter.SurfaceSupport2,
                    _cutter.OriginalSurfaceFaceKeys)
                : SurfaceReconstructor.RebuildSurface(_data);

            if (originalSurfaceTris == null)
                originalSurfaceTris = System.Array.Empty<int>();

            int[] fullBoundaryTris = SurfaceReconstructor.RebuildSurface(_data);
            _renderOriginalSurfaceTris.Clear();
            _renderOriginalSurfaceTris.AddRange(originalSurfaceTris);
            BuildCutTriangleList(_data, fullBoundaryTris, originalSurfaceTris, _renderCutTris, _renderOriginalSurfaceKeys);

            _data.SetSurfaceTriIds(originalSurfaceTris);
            return new SurfaceBuildResult(originalSurfaceTris, _renderOriginalSurfaceTris, _renderCutTris);
        }

        bool HasOriginalSurfaceSupportData()
        {
            return _data != null &&
                   _cutter != null &&
                   _cutter.OriginalSurfaceFaceKeys != null &&
                   _cutter.OriginalSurfaceFaceKeys.Count > 0 &&
                   _cutter.SurfaceSupport0 != null &&
                   _cutter.SurfaceSupport1 != null &&
                   _cutter.SurfaceSupport2 != null;
        }

        static void BuildCutTriangleList(
            TetMeshData data,
            int[] fullBoundaryTris,
            int[] originalSurfaceTris,
            List<int> cutTris,
            HashSet<long> originalKeys)
        {
            cutTris.Clear();
            originalKeys.Clear();

            if (fullBoundaryTris == null || fullBoundaryTris.Length < 3)
                return;

            if (originalSurfaceTris != null)
            {
                for (int i = 0; i + 2 < originalSurfaceTris.Length; i += 3)
                    originalKeys.Add(SurfaceReconstructor.FaceKey(
                        originalSurfaceTris[i + 0],
                        originalSurfaceTris[i + 1],
                        originalSurfaceTris[i + 2]));
            }

            for (int i = 0; i + 2 < fullBoundaryTris.Length; i += 3)
            {
                int a = fullBoundaryTris[i + 0];
                int b = fullBoundaryTris[i + 1];
                int c = fullBoundaryTris[i + 2];
                if (originalKeys.Contains(SurfaceReconstructor.FaceKey(a, b, c)))
                    continue;
                if (!IsValidCutTriangle(data, a, b, c))
                    continue;

                cutTris.Add(a);
                cutTris.Add(b);
                cutTris.Add(c);
            }

            FilterTinyCutPatches(data, cutTris);
        }

        static void FilterTinyCutPatches(TetMeshData data, List<int> cutTris)
        {
            if (data == null || data.Positions == null || cutTris.Count < 3) return;

            int triCount = cutTris.Count / 3;
            bool[] visited = new bool[triCount];
            bool[] remove = new bool[triCount];
            List<int> stack = new List<int>(32);
            List<int> component = new List<int>(32);

            for (int seed = 0; seed < triCount; seed++)
            {
                if (visited[seed]) continue;

                stack.Clear();
                component.Clear();
                stack.Add(seed);
                visited[seed] = true;
                float area = 0f;

                while (stack.Count > 0)
                {
                    int tri = stack[stack.Count - 1];
                    stack.RemoveAt(stack.Count - 1);
                    component.Add(tri);
                    area += TriangleArea(data, cutTris[tri * 3 + 0], cutTris[tri * 3 + 1], cutTris[tri * 3 + 2]);

                    for (int other = 0; other < triCount; other++)
                    {
                        if (visited[other]) continue;
                        if (!TrianglesShareVertex(cutTris, tri, other)) continue;

                        visited[other] = true;
                        stack.Add(other);
                    }
                }

                if (area < 1e-8f)
                {
                    for (int i = 0; i < component.Count; i++)
                        remove[component[i]] = true;
                }
            }

            int write = 0;
            for (int tri = 0; tri < triCount; tri++)
            {
                if (remove[tri]) continue;
                cutTris[write++] = cutTris[tri * 3 + 0];
                cutTris[write++] = cutTris[tri * 3 + 1];
                cutTris[write++] = cutTris[tri * 3 + 2];
            }
            if (write < cutTris.Count)
                cutTris.RemoveRange(write, cutTris.Count - write);
        }

        static bool TrianglesShareVertex(List<int> tris, int a, int b)
        {
            int ai = a * 3;
            int bi = b * 3;
            return tris[ai + 0] == tris[bi + 0] || tris[ai + 0] == tris[bi + 1] || tris[ai + 0] == tris[bi + 2] ||
                   tris[ai + 1] == tris[bi + 0] || tris[ai + 1] == tris[bi + 1] || tris[ai + 1] == tris[bi + 2] ||
                   tris[ai + 2] == tris[bi + 0] || tris[ai + 2] == tris[bi + 1] || tris[ai + 2] == tris[bi + 2];
        }

        static float TriangleArea(TetMeshData data, int a, int b, int c)
        {
            return Vector3.Cross(data.Positions[b] - data.Positions[a],
                                 data.Positions[c] - data.Positions[a]).magnitude * 0.5f;
        }

        static bool IsValidCutTriangle(TetMeshData data, int a, int b, int c)
        {
            if (data == null || data.Positions == null) return false;
            if (a == b || a == c || b == c) return false;
            if (a < 0 || b < 0 || c < 0 ||
                a >= data.NumParticles || b >= data.NumParticles || c >= data.NumParticles)
                return false;

            Vector3 pa = data.Positions[a];
            Vector3 pb = data.Positions[b];
            Vector3 pc = data.Positions[c];
            float ab = (pa - pb).sqrMagnitude;
            float bc = (pb - pc).sqrMagnitude;
            float ca = (pc - pa).sqrMagnitude;
            const float minEdgeSqr = 1e-12f;
            if (ab < minEdgeSqr || bc < minEdgeSqr || ca < minEdgeSqr) return false;

            float area2 = Vector3.Cross(pb - pa, pc - pa).sqrMagnitude;
            const float minArea2 = 1e-12f;
            if (area2 < minArea2) return false;

            float maxEdge = Mathf.Max(ab, Mathf.Max(bc, ca));
            return area2 >= maxEdge * maxEdge * 1e-8f;
        }

        void ApplyVisibleSurface(SurfaceBuildResult surface)
        {
            if (_visualizer != null)
                _visualizer.RebuildTopology(surface.OriginalSurfaceList, surface.CutTriList);

            SofaUnityVisualLiverRenderer renderer = ResolveSofaVisualRenderer();
            if (renderer != null && renderer.IsInitialized)
                renderer.RebuildTetSurfaceTopology(surface.OriginalSurfaceTris ?? System.Array.Empty<int>());
        }

        SofaUnityVisualLiverRenderer ResolveSofaVisualRenderer()
        {
            if (_sofaVisualRenderer != null) return _sofaVisualRenderer;
            _sofaVisualRenderer = GetComponent<SofaUnityVisualLiverRenderer>();
            if (_sofaVisualRenderer == null && _visualizer != null)
                _sofaVisualRenderer = _visualizer.GetComponentInParent<SofaUnityVisualLiverRenderer>();
            return _sofaVisualRenderer;
        }

        readonly struct SurfaceBuildResult
        {
            public readonly int[] OriginalSurfaceTris;
            public readonly List<int> OriginalSurfaceList;
            public readonly List<int> CutTriList;

            public SurfaceBuildResult(int[] originalSurfaceTris, List<int> originalSurfaceList, List<int> cutTriList)
            {
                OriginalSurfaceTris = originalSurfaceTris ?? System.Array.Empty<int>();
                OriginalSurfaceList = originalSurfaceList ?? new List<int>();
                CutTriList = cutTriList ?? new List<int>();
            }
        }

        void EnsureSurgicalTool()
        {
            if (_surgicalTool != null) return;
            _surgicalTool = GetComponent<SurgicalTool>()
                         ?? gameObject.AddComponent<SurgicalTool>();
        }

        void SetupDebugLine()
        {
            if (GetComponent<LineRenderer>()) return;
            _debugLine = gameObject.AddComponent<LineRenderer>();
            _debugLine.startWidth = 0.003f; _debugLine.endWidth = 0.001f;
            _debugLine.material = new Material(Shader.Find("Sprites/Default"));
            _debugLine.startColor = Color.cyan; _debugLine.endColor = Color.yellow;
            _debugLine.positionCount = 2; _debugLine.enabled = false;
        }

        void OnDestroy()
        {
            if (_debugLine && _debugLine.material) Destroy(_debugLine.material);
        }
    }
}
