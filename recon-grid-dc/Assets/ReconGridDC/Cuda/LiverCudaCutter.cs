using System.Runtime.InteropServices;
using System.Reflection;
using Unity.Mathematics;
using UnityEngine;
using ReconGridDC.Cutting;
using ReconGridDC.Demo;
using ReconGridDC.Preprocess;
using ReconGridDC.Recon;

namespace ReconGridDC.Cuda
{
    /// <summary>
    /// Scene-facing CUDA cutter for LiverCudaManager. This component owns the cutting rod target,
    /// CuttingTool history, swept-ribbon diagnostics, and the native LCS_SetTool/LCS_DetectCut calls.
    /// </summary>
    [DisallowMultipleComponent]
    public sealed class LiverCudaCutter : MonoBehaviour
    {
        [Header("Interactive rod (TargetObject-driven cutting)")]
        public bool enableCutting = true;
        [Tooltip("Scene object that owns the visible cutting line. Drag this GameObject into inverseAPI/TargetObject.targetObject.")]
        public LiverCuttingRod cuttingRodTarget;
        public bool autoCreateCuttingRodTarget = true;
        [Tooltip("When true, this cutter writes rodLength/thickness to the target every frame. Disable this when a model-driven tool, such as xiaogun_v1, owns its visual length.")]
        public bool driveCuttingRodShape = true;
        public float rodLength = 12f;
        public bool autoSizeRod = false;
        public Vector3 rodAxis = new Vector3(0, 0, 1);
        public float rodThicknessDOverL = 0.16f;
        [Tooltip("Approximate a cylindrical cutting stick by testing multiple parallel swept planes across its diameter.")]
        public bool cutWithRodThicknessVolume = true;
        [Tooltip("Odd numbers include the center plane. Higher values cover thick sticks more smoothly but cost more CUDA cut passes.")]
        [Range(1, 15)] public int thicknessCutPlaneSamples = 5;
        public float cutFPInterp = 1.0f;

        [Header("Keyboard Control")]
        [Tooltip("Allow the keyboard to move and rotate the cutting rod. Disabled by default so the Haply TargetObject remains the sole controller.")]
        public bool enableKeyboardControl = false;
        [Tooltip("World-space translation speed of the cutting rod while a movement key is held.")]
        [Min(0f)] public float keyboardMoveSpeed = 5f;
        [Tooltip("Pitch/yaw speed of the cutting rod while a rotation key is held.")]
        [Min(0f)] public float keyboardRotationSpeed = 60f;
        public KeyCode moveLeftKey = KeyCode.A;
        public KeyCode moveRightKey = KeyCode.D;
        public KeyCode moveBackwardKey = KeyCode.S;
        public KeyCode moveForwardKey = KeyCode.W;
        public KeyCode moveDownKey = KeyCode.Q;
        public KeyCode moveUpKey = KeyCode.E;
        public KeyCode yawLeftKey = KeyCode.LeftArrow;
        public KeyCode yawRightKey = KeyCode.RightArrow;
        public KeyCode pitchUpKey = KeyCode.UpArrow;
        public KeyCode pitchDownKey = KeyCode.DownArrow;

        const string DLL = "LiverCudaSim";

        [StructLayout(LayoutKind.Sequential)]
        struct CutInitDesc
        {
            public int voxelCount, cornerCount, gridEdgeCount, isectCount, triCapacity;
            public int dimsX, dimsY, dimsZ;
            public float voxelL;
            public float originX, originY, originZ;
            public int cutPointCapacity;
            public float cutFPInterp;
            public float tearStretchRatio;
        }

        [StructLayout(LayoutKind.Sequential)]
        struct Conn4096Interop { public int compCount, v0, v1, v2, v3, v4, v5, v6, v7; }

        [StructLayout(LayoutKind.Sequential)]
        struct GridEdge2Interop { public int axis, bcX, bcY, bcZ; }

        [StructLayout(LayoutKind.Sequential)]
        struct CutToolDesc
        {
            public float t1v0x, t1v0y, t1v0z, t1v1x, t1v1y, t1v1z, t1v2x, t1v2y, t1v2z;
            public float t2v0x, t2v0y, t2v0z, t2v1x, t2v1y, t2v1z, t2v2x, t2v2y, t2v2z;
            public float ncutx, ncuty, ncutz;
            public float d;
            public int valid;
            public float aabbMinx, aabbMiny, aabbMinz, aabbMaxx, aabbMaxy, aabbMaxz;
        }

        [DllImport(DLL)] static extern int LCS_InitCut(ref CutInitDesc desc, Conn4096Interop[] conn4096, GridEdge2Interop[] gridEdges, int[] voxelOccupied, int[] cornerInside);
        [DllImport(DLL)] static extern void LCS_SetTool(ref CutToolDesc tool);
        [DllImport(DLL)] static extern int LCS_DetectCut();

        CuttingTool _tool;
        float _voxelL;
        float _cutMetricVoxelL;
        float _cutStepVoxelL;
        float3 _origin;
        int3 _dims;
        bool _nativeReady;
        MonoBehaviour _targetObjectController;
        PropertyInfo _targetObjectReferenceProperty;
        PropertyInfo _followPositionProperty;
        PropertyInfo _followRotationProperty;
        bool _inputModeInitialized;
        bool _savedFollowPosition;
        bool _savedFollowRotation;

        Bounds _sweptUnion;
        bool _sweptUnionSet;
        Vector3 _lastCutNormal = Vector3.up;
        Vector3 _lastCutPoint;
        readonly System.Collections.Generic.HashSet<int> _sweptVox = new System.Collections.Generic.HashSet<int>();

        public bool NativeReady => _nativeReady;
        public float OriginalVoxelLength => _voxelL;
        public float EffectiveCutVoxelLength => _cutMetricVoxelL > 0f ? _cutMetricVoxelL : _voxelL;
        public bool InteractiveReady => enableCutting && cuttingRodTarget != null && _tool != null;
        public int GridEdgeCount { get; private set; }
        public int SweptVoxelCount => _sweptVox.Count;
        public Vector3 LastCutNormal => _lastCutNormal;
        public Vector3 LastCutPoint => _lastCutPoint;
        public bool LastSweepWasValid { get; private set; }
        public string LastSweepStatus { get; private set; } = "Waiting for cutting rod movement.";

        public int InitializeNative(BackgroundGrid grid, int3 dims, float voxelL, float3 origin, float3 gridCenter,
            int triCapacity, float tearStretchRatio, Vector3 anchorAxis)
        {
            ConnectivityLUT.Build(out int[] compCount, out int[] vertToComp);
            var connI = new Conn4096Interop[ConnectivityLUT.CONFIG_COUNT];
            for (int cfg = 0; cfg < ConnectivityLUT.CONFIG_COUNT; cfg++)
            {
                int b = cfg * 8;
                connI[cfg] = new Conn4096Interop
                {
                    compCount = compCount[cfg],
                    v0 = vertToComp[b + 0], v1 = vertToComp[b + 1], v2 = vertToComp[b + 2], v3 = vertToComp[b + 3],
                    v4 = vertToComp[b + 4], v5 = vertToComp[b + 5], v6 = vertToComp[b + 6], v7 = vertToComp[b + 7]
                };
            }

            var geI = new GridEdge2Interop[Mathf.Max(1, grid.gridEdges.Length)];
            for (int i = 0; i < grid.gridEdges.Length; i++)
            {
                var ge = grid.gridEdges[i];
                geI[i] = new GridEdge2Interop { axis = ge.axis, bcX = ge.baseCorner.x, bcY = ge.baseCorner.y, bcZ = ge.baseCorner.z };
            }

            GridEdgeCount = grid.gridEdges.Length;
            var occI = new int[grid.voxelCount];
            for (int v = 0; v < grid.voxelCount; v++) occI[v] = grid.voxelOccupied[v];

            var cdesc = new CutInitDesc
            {
                voxelCount = grid.voxelCount,
                cornerCount = grid.cornerCount,
                gridEdgeCount = grid.gridEdges.Length,
                isectCount = grid.isect.Length,
                triCapacity = triCapacity,
                dimsX = dims.x,
                dimsY = dims.y,
                dimsZ = dims.z,
                voxelL = voxelL,
                originX = origin.x,
                originY = origin.y,
                originZ = origin.z,
                cutPointCapacity = Mathf.Max(1, 4 * grid.gridEdges.Length),
                cutFPInterp = cutFPInterp,
                tearStretchRatio = tearStretchRatio
            };

            var insideI = new int[grid.cornerCount];
            for (int ci = 0; ci < grid.cornerCount; ci++) insideI[ci] = grid.cornerInside[ci];

            int rc = LCS_InitCut(ref cdesc, connI, geI, occI, insideI);
            if (rc != 0)
            {
                _nativeReady = false;
                return rc;
            }

            _nativeReady = true;
            _dims = dims;
            _voxelL = voxelL;
            _cutMetricVoxelL = voxelL;
            _cutStepVoxelL = voxelL;
            _origin = origin;

            Vector3 initialCenter = (Vector3)gridCenter + new Vector3(0f, voxelL * 0.5f, 0f);
            rodAxis = rodAxis.sqrMagnitude < 1e-6f ? new Vector3(0, 0, 1) : rodAxis.normalized;
            if (anchorAxis.sqrMagnitude > 1e-6f && Mathf.Abs(Vector3.Dot(rodAxis, anchorAxis.normalized)) > 0.9f)
            {
                Debug.LogWarning("[LiverCuda] rodAxis is nearly parallel to anchorAxis - a cut cannot separate a free piece (it will only swing). Make rodAxis perpendicular to anchorAxis.", this);
            }

            float rodD = rodThicknessDOverL * voxelL;
            EnsureCuttingRodTarget(initialCenter, rodAxis, rodLength, rodD);
            if (enableCutting && cuttingRodTarget != null)
            {
                cuttingRodTarget.GetEndpoints(out Vector3 s0, out Vector3 e0);
                _tool = new CuttingTool(s0, e0, rodD);
            }

            return 0;
        }

        public void SetCutMetricVoxelLengths(float visualVoxelLength, float conservativeStepLength)
        {
            _cutMetricVoxelL = visualVoxelLength > 0f ? visualVoxelLength : _voxelL;
            _cutStepVoxelL = conservativeStepLength > 0f ? conservativeStepLength : _cutMetricVoxelL;
        }

        public void ResetCutMetricVoxelLength()
        {
            _cutMetricVoxelL = _voxelL;
            _cutStepVoxelL = _voxelL;
        }

        public void Step(float rodSpanLength)
        {
            if (!InteractiveReady) return;

            ApplyKeyboardControl();

            if (driveCuttingRodShape)
            {
                float effRodLen = autoSizeRod ? Mathf.Max(rodLength, rodSpanLength) : rodLength;
                cuttingRodTarget.Length = effRodLen;
                cuttingRodTarget.Thickness = Mathf.Max(0.002f, rodThicknessDOverL * EffectiveCutVoxelLength);
            }

            _tool.SetThickness(cuttingRodTarget.Thickness);
            cuttingRodTarget.GetEndpoints(out Vector3 sNew, out Vector3 eNew);
            Vector3 s0 = _tool.S, e0 = _tool.E;
            int n = RodMath.SubStepCount(s0, e0, sNew, eNew, _cutStepVoxelL);
            for (int i = 1; i <= n; i++)
            {
                float ti = (float)i / n;
                _tool.Advance(Vector3.Lerp(s0, sNew, ti), Vector3.Lerp(e0, eNew, ti));
                bool validCutTool = DispatchCutTool(_tool);
                LastSweepWasValid = validCutTool;
                if (validCutTool)
                {
                    AccumSwept(_tool.S);
                    AccumSwept(_tool.E);
                    _lastCutNormal = _tool.NCut;
                    _lastCutPoint = 0.5f * (_tool.S + _tool.E);
                    PaintSweptRibbon(_tool.S, _tool.E);
                    LastSweepStatus = "Valid lateral blade sweep dispatched to CUDA cut detection.";
                }
                else
                {
                    LastSweepStatus = "No cut dispatched: move the rod sideways across its own axis to create a swept cutting plane.";
                }
            }

            rodAxis = cuttingRodTarget.Axis;
        }

        /// <summary>
        /// Enables or disables the optional keyboard controller at runtime. When disabled, only
        /// externally driven transforms such as the Haply TargetObject affect the cutting rod.
        /// </summary>
        public void SetKeyboardControlEnabled(bool enabled)
        {
            enableKeyboardControl = enabled;
            SynchronizeInputMode();
        }

        void ApplyKeyboardControl()
        {
            SynchronizeInputMode();
            if (!enableKeyboardControl || cuttingRodTarget == null) return;

            Vector3 move = Vector3.zero;
            if (Input.GetKey(moveLeftKey)) move.x -= 1f;
            if (Input.GetKey(moveRightKey)) move.x += 1f;
            if (Input.GetKey(moveBackwardKey)) move.z -= 1f;
            if (Input.GetKey(moveForwardKey)) move.z += 1f;
            if (Input.GetKey(moveDownKey)) move.y -= 1f;
            if (Input.GetKey(moveUpKey)) move.y += 1f;

            if (move.sqrMagnitude > 1f) move.Normalize();
            cuttingRodTarget.transform.position += move * (keyboardMoveSpeed * Time.deltaTime);

            float yaw = 0f;
            float pitch = 0f;
            if (Input.GetKey(yawLeftKey)) yaw -= 1f;
            if (Input.GetKey(yawRightKey)) yaw += 1f;
            if (Input.GetKey(pitchUpKey)) pitch += 1f;
            if (Input.GetKey(pitchDownKey)) pitch -= 1f;

            if (yaw == 0f && pitch == 0f) return;

            Vector3 axis = cuttingRodTarget.Axis;
            Quaternion rotation = Quaternion.Euler(pitch * keyboardRotationSpeed * Time.deltaTime,
                                                    yaw * keyboardRotationSpeed * Time.deltaTime,
                                                    0f);
            cuttingRodTarget.SetPose(cuttingRodTarget.Center, rotation * axis);
        }

        void SynchronizeInputMode()
        {
            if (cuttingRodTarget == null) return;

            if (_targetObjectController == null ||
                GetTargetObjectReference(_targetObjectController) != cuttingRodTarget.gameObject)
            {
                _targetObjectController = FindTargetObjectController();
                _inputModeInitialized = false;
            }

            if (_targetObjectController == null) return;

            if (enableKeyboardControl)
            {
                if (!_inputModeInitialized)
                {
                    _savedFollowPosition = GetFollowPosition();
                    _savedFollowRotation = GetFollowRotation();
                    _inputModeInitialized = true;
                }

                SetFollowPosition(false);
                SetFollowRotation(false);
            }
            else if (_inputModeInitialized)
            {
                SetFollowPosition(_savedFollowPosition);
                SetFollowRotation(_savedFollowRotation);
                _inputModeInitialized = false;
            }
        }

        MonoBehaviour FindTargetObjectController()
        {
            foreach (var controller in FindObjectsOfType<MonoBehaviour>())
            {
                if (controller.GetType().Name == "TargetObject" &&
                    GetTargetObjectReference(controller) == cuttingRodTarget.gameObject)
                {
                    return controller;
                }
            }

            return null;
        }

        GameObject GetTargetObjectReference(MonoBehaviour controller)
        {
            CacheTargetObjectProperties(controller);
            return _targetObjectReferenceProperty?.GetValue(controller) as GameObject;
        }

        bool GetFollowPosition()
        {
            return _followPositionProperty != null && (bool)_followPositionProperty.GetValue(_targetObjectController);
        }

        bool GetFollowRotation()
        {
            return _followRotationProperty != null && (bool)_followRotationProperty.GetValue(_targetObjectController);
        }

        void SetFollowPosition(bool value)
        {
            _followPositionProperty?.SetValue(_targetObjectController, value);
        }

        void SetFollowRotation(bool value)
        {
            _followRotationProperty?.SetValue(_targetObjectController, value);
        }

        void CacheTargetObjectProperties(MonoBehaviour controller)
        {
            if (controller == _targetObjectController && _targetObjectReferenceProperty != null) return;

            var type = controller.GetType();
            _targetObjectReferenceProperty = type.GetProperty("TargetObjectReference");
            _followPositionProperty = type.GetProperty("FollowPosition");
            _followRotationProperty = type.GetProperty("FollowRotation");
        }

        public bool SweptContains(Vector3 p)
        {
            int cx = Mathf.FloorToInt((p.x - _origin.x) / _voxelL);
            int cy = Mathf.FloorToInt((p.y - _origin.y) / _voxelL);
            int cz = Mathf.FloorToInt((p.z - _origin.z) / _voxelL);
            return _sweptVox.Contains(SweptKey(cx, cy, cz));
        }

        public string FormatCoverageHud(float rodSpanLength, float anchoredExtent)
            => $"CUT rod={Mathf.Max(rodLength, rodSpanLength):F1}u | anchored extent={anchoredExtent:F1}u | press P for cut diagnostics";

        public string FormatBridgeHud(int interiorHoles, int unsweptStraddles, float rodSpanLength, float anchoredExtent)
            => $"CUT holes={interiorHoles} unswept={unsweptStraddles} | rod={Mathf.Max(rodLength, rodSpanLength):F1}u anchored={anchoredExtent:F1}u";

        void EnsureCuttingRodTarget(Vector3 initialCenter, Vector3 initialAxis, float initialLength, float initialThickness)
        {
            if (!enableCutting) return;

            if (cuttingRodTarget == null && autoCreateCuttingRodTarget)
            {
                var go = new GameObject("LiverCuttingRodTarget");
                go.transform.SetParent(transform, false);
                cuttingRodTarget = go.AddComponent<LiverCuttingRod>();
            }

            if (cuttingRodTarget == null)
            {
                Debug.LogError("[LiverCuda] enableCutting is true, but no LiverCuttingRod is assigned and autoCreateCuttingRodTarget is false.", this);
                enableCutting = false;
                return;
            }

            if (driveCuttingRodShape)
            {
                cuttingRodTarget.Configure(initialLength, initialThickness);
            }

            cuttingRodTarget.SetPose(initialCenter, initialAxis);
        }

        CutToolDesc MakeToolDesc(CuttingTool t)
        {
            return MakeToolDesc(t, Vector3.zero);
        }

        CutToolDesc MakeToolDesc(CuttingTool t, Vector3 offset)
        {
            Vector3 t1v0 = t.T1V0 + offset;
            Vector3 t1v1 = t.T1V1 + offset;
            Vector3 t1v2 = t.T1V2 + offset;
            Vector3 t2v0 = t.T2V0 + offset;
            Vector3 t2v1 = t.T2V1 + offset;
            Vector3 t2v2 = t.T2V2 + offset;

            Vector3 mn = Vector3.Min(Vector3.Min(t1v0, t1v1), Vector3.Min(t1v2, t2v2));
            Vector3 mx = Vector3.Max(Vector3.Max(t1v0, t1v1), Vector3.Max(t1v2, t2v2));
            Vector3 inflate = Vector3.one * Mathf.Max(0f, 0.5f * t.D);
            mn -= inflate;
            mx += inflate;

            return new CutToolDesc
            {
                t1v0x = t1v0.x, t1v0y = t1v0.y, t1v0z = t1v0.z,
                t1v1x = t1v1.x, t1v1y = t1v1.y, t1v1z = t1v1.z,
                t1v2x = t1v2.x, t1v2y = t1v2.y, t1v2z = t1v2.z,
                t2v0x = t2v0.x, t2v0y = t2v0.y, t2v0z = t2v0.z,
                t2v1x = t2v1.x, t2v1y = t2v1.y, t2v1z = t2v1.z,
                t2v2x = t2v2.x, t2v2y = t2v2.y, t2v2z = t2v2.z,
                ncutx = t.NCut.x, ncuty = t.NCut.y, ncutz = t.NCut.z,
                d = t.D,
                valid = t.Valid ? 1 : 0,
                aabbMinx = mn.x, aabbMiny = mn.y, aabbMinz = mn.z,
                aabbMaxx = mx.x, aabbMaxy = mx.y, aabbMaxz = mx.z
            };
        }

        bool DispatchCutTool(CuttingTool t)
        {
            var centerTool = MakeToolDesc(t);
            if (centerTool.valid == 0)
            {
                LCS_SetTool(ref centerTool);
                LCS_DetectCut();
                return false;
            }

            float radius = 0.5f * t.D;
            int samples = Mathf.Clamp(thicknessCutPlaneSamples, 1, 15);
            if ((samples & 1) == 0) samples += 1;

            if (cutWithRodThicknessVolume && radius > 0.0005f && samples > 1)
            {
                Vector3 offsetAxis = RodThicknessOffsetAxis(t);
                if (offsetAxis.sqrMagnitude > 1e-10f)
                {
                    offsetAxis.Normalize();
                    int mid = samples / 2;
                    for (int i = 0; i < samples; i++)
                    {
                        if (i == mid) continue;

                        float u = ((float)i / (samples - 1) - 0.5f) * 2f;
                        var offsetTool = MakeToolDesc(t, offsetAxis * (u * radius));
                        LCS_SetTool(ref offsetTool);
                        LCS_DetectCut();
                    }
                }
            }

            LCS_SetTool(ref centerTool);
            LCS_DetectCut();
            return true;
        }

        static Vector3 RodThicknessOffsetAxis(CuttingTool t)
        {
            Vector3 toolAxis = t.E - t.S;
            if (toolAxis.sqrMagnitude < 1e-10f) return Vector3.zero;
            toolAxis.Normalize();

            Vector3 sweep = 0.5f * ((t.S - t.Sprev) + (t.E - t.Eprev));
            Vector3 offsetAxis = Vector3.Cross(toolAxis, sweep);
            if (offsetAxis.sqrMagnitude > 1e-10f) return offsetAxis;

            offsetAxis = t.NCut;
            if (offsetAxis.sqrMagnitude > 1e-10f) return offsetAxis;

            offsetAxis = Vector3.Cross(toolAxis, Vector3.up);
            if (offsetAxis.sqrMagnitude > 1e-10f) return offsetAxis;

            return Vector3.Cross(toolAxis, Vector3.right);
        }

        void AccumSwept(Vector3 p)
        {
            if (!_sweptUnionSet)
            {
                _sweptUnion = new Bounds(p, Vector3.zero);
                _sweptUnionSet = true;
            }
            else
            {
                _sweptUnion.Encapsulate(p);
            }
        }

        void PaintSweptRibbon(Vector3 s, Vector3 e)
        {
            float step = 0.5f * _voxelL;
            int n = Mathf.Max(1, Mathf.CeilToInt(Vector3.Distance(s, e) / step));
            for (int i = 0; i <= n; i++)
            {
                Vector3 p = Vector3.Lerp(s, e, (float)i / n);
                int cx = Mathf.FloorToInt((p.x - _origin.x) / _voxelL);
                int cy = Mathf.FloorToInt((p.y - _origin.y) / _voxelL);
                int cz = Mathf.FloorToInt((p.z - _origin.z) / _voxelL);
                for (int dx = -1; dx <= 1; dx++)
                for (int dy = -1; dy <= 1; dy++)
                for (int dz = -1; dz <= 1; dz++)
                    _sweptVox.Add(SweptKey(cx + dx, cy + dy, cz + dz));
            }
        }

        static int SweptKey(int x, int y, int z) => (x & 0x3FF) | ((y & 0x3FF) << 10) | ((z & 0x3FF) << 20);
    }
}
