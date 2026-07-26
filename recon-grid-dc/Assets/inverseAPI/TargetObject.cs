using UnityEngine;

[DisallowMultipleComponent]
public class TargetObject : MonoBehaviour
{
    private enum PositionMappingMode
    {
        CameraViewport,
        DriverWorldPosition
    }

    private enum PositionInputSpace
    {
        DriverLocalPosition,
        DriverWorldPosition
    }

    [Header("Target Object API")]
    [SerializeField] private GameObject targetObject;

    [Header("Driver")]
    [SerializeField] private Transform driverTransform;

    [Header("Position Mapping")]
    [SerializeField] private PositionMappingMode positionMappingMode = PositionMappingMode.DriverWorldPosition;
    [SerializeField] private PositionInputSpace positionInputSpace = PositionInputSpace.DriverWorldPosition;

    [Header("Camera Viewport Mapping")]
    [SerializeField] private Camera viewCamera;
    [SerializeField] private HapticCursorWorldPositioner cursorWorldPositioner;
    [SerializeField] private bool useMainCameraWhenEmpty = true;
    [SerializeField] private bool useDesiredCursorWorldPositionAsCenter = true;
    [SerializeField] private bool captureInputCenterOnBind = false;
    [SerializeField] private Vector3 viewportWorldCenter = Vector3.zero;
    [SerializeField] private Vector3 inputCenter = Vector3.zero;
    [SerializeField] private Vector3 inputHalfRange = new Vector3(0.22f, 0.22f, 0.22f);
    [SerializeField, Range(0f, 0.45f)] private float viewportPadding = 0.02f;
    [SerializeField, Range(0f, 1f)] private float depthRangeScale = 0.35f;
    [SerializeField] private bool invertDepthInput = false;

    [Header("Follow Settings")]
    [SerializeField] private bool followPosition = true;
    [SerializeField] private bool followRotation = true;
    [SerializeField] private bool keepInitialOffset = false;
    [SerializeField] private bool bindOnEnable = true;

    [Header("Cursor Visual")]
    [SerializeField] private GameObject cursorModel;
    [SerializeField] private bool hideCursorModelOnPlay = true;
    [SerializeField] private bool disableCursorRenderers = true;

    private Transform _targetTransform;
    private Vector3 _positionOffset;
    private Quaternion _rotationOffset = Quaternion.identity;
    private bool _isBound;
    private Vector3 _runtimeInputCenter;
    private Vector3 _runtimeViewportWorldCenter;
    private float _cameraDepth;
    private Renderer[] _cursorRenderers;
    private Collider[] _cursorColliders;

    public GameObject TargetObjectReference => targetObject;
    public Transform DriverTransform => driverTransform != null ? driverTransform : transform;

    public bool FollowPosition
    {
        get => followPosition;
        set => followPosition = value;
    }

    public bool FollowRotation
    {
        get => followRotation;
        set => followRotation = value;
    }

    public bool KeepInitialOffset
    {
        get => keepInitialOffset;
        set => keepInitialOffset = value;
    }

    private void Awake()
    {
        CacheCursorVisualComponents();
    }

    private void OnEnable()
    {
        ApplyCursorVisualState();

        if (bindOnEnable)
        {
            Rebind();
        }
    }

    private void Start()
    {
        CacheCursorVisualComponents();
        ApplyCursorVisualState();
        if (!_isBound && bindOnEnable)
        {
            Rebind();
        }
    }

    private void LateUpdate()
    {
        ApplyCursorVisualState();

        if (!TryResolveTarget())
        {
            return;
        }

        if (!_isBound)
        {
            Rebind();
        }

        ApplyFollow();
    }

    public void SetTargetObject(GameObject newTargetObject)
    {
        targetObject = newTargetObject;
        Rebind();
    }

    public void SetDriverTransform(Transform newDriverTransform)
    {
        driverTransform = newDriverTransform;
        Rebind();
    }

    [ContextMenu("Rebind Target Object")]
    public void Rebind()
    {
        if (!TryResolveTarget())
        {
            _isBound = false;
            return;
        }

        var driver = DriverTransform;
        var driverPosition = driver.position;
        var driverRotation = driver.rotation;
        _runtimeInputCenter = captureInputCenterOnBind ? GetDriverInputPosition(driver) : inputCenter;
        CaptureViewportWorldCenter();
        CaptureCameraDepth();

        if (keepInitialOffset)
        {
            _positionOffset = Quaternion.Inverse(driverRotation) * (_targetTransform.position - driverPosition);
            _rotationOffset = Quaternion.Inverse(driverRotation) * _targetTransform.rotation;
        }
        else
        {
            _positionOffset = Vector3.zero;
            _rotationOffset = Quaternion.identity;
        }

        _isBound = true;
    }

    public void ClearTargetObject()
    {
        targetObject = null;
        _targetTransform = null;
        _isBound = false;
    }

    private bool TryResolveTarget()
    {
        if (targetObject == null)
        {
            _targetTransform = null;
            _isBound = false;
            return false;
        }

        _targetTransform = targetObject.transform;
        return true;
    }

    private void ApplyCursorVisualState()
    {
        if (!hideCursorModelOnPlay || cursorModel == null)
        {
            return;
        }

        if (disableCursorRenderers)
        {
            CacheCursorVisualComponents();

            if (_cursorRenderers != null)
            {
                foreach (var cursorRenderer in _cursorRenderers)
                {
                    if (cursorRenderer != null)
                    {
                        cursorRenderer.enabled = false;
                    }
                }
            }

            if (_cursorColliders != null)
            {
                foreach (var cursorCollider in _cursorColliders)
                {
                    if (cursorCollider != null)
                    {
                        cursorCollider.enabled = false;
                    }
                }
            }
        }

        if (cursorModel.activeSelf)
        {
            cursorModel.SetActive(false);
        }
    }

    private void CacheCursorVisualComponents()
    {
        if (cursorModel == null)
        {
            _cursorRenderers = null;
            _cursorColliders = null;
            return;
        }

        _cursorRenderers = cursorModel.GetComponentsInChildren<Renderer>(true);
        _cursorColliders = cursorModel.GetComponentsInChildren<Collider>(true);
    }


    [ContextMenu("Capture Current Input Center")]
    public void CaptureCurrentInputCenter()
    {
        inputCenter = GetDriverInputPosition(DriverTransform);
        _runtimeInputCenter = inputCenter;
    }

    [ContextMenu("Capture Current Viewport World Center")]
    public void CaptureCurrentViewportWorldCenter()
    {
        viewportWorldCenter = _targetTransform != null ? _targetTransform.position : transform.position;
        _runtimeViewportWorldCenter = viewportWorldCenter;
    }


    private void ApplyFollow()
    {
        var driver = DriverTransform;
        var driverPosition = driver.position;
        var driverRotation = driver.rotation;
        var nextPosition = GetNextPosition(driver, driverPosition, driverRotation);
        var nextRotation = driverRotation * _rotationOffset;

        if (followPosition)
        {
            _targetTransform.position = nextPosition;
        }

        if (followRotation)
        {
            _targetTransform.rotation = nextRotation;
        }
    }

    private Vector3 GetNextPosition(Transform driver, Vector3 driverPosition, Quaternion driverRotation)
    {
        if (positionMappingMode == PositionMappingMode.CameraViewport && TryGetViewCamera(out var cameraToUse))
        {
            return GetCameraViewportPosition(driver, cameraToUse);
        }

        return driverPosition + driverRotation * _positionOffset;
    }

    private Vector3 GetCameraViewportPosition(Transform driver, Camera cameraToUse)
    {
        var input = GetDriverInputPosition(driver);
        var safeHalfRange = GetSafeInputHalfRange();

        var normalizedX = Mathf.Clamp((input.x - _runtimeInputCenter.x) / safeHalfRange.x, -1f, 1f);
        var normalizedY = Mathf.Clamp((input.y - _runtimeInputCenter.y) / safeHalfRange.y, -1f, 1f);
        var normalizedZ = Mathf.Clamp((input.z - _runtimeInputCenter.z) / safeHalfRange.z, -1f, 1f);
        if (invertDepthInput)
        {
            normalizedZ = -normalizedZ;
        }

        var rangeScale = 1f - Mathf.Clamp(viewportPadding, 0f, 0.45f) * 2f;
        var cameraDepth = GetSafeCameraDepth(cameraToUse);

        GetCameraHalfExtents(cameraToUse, cameraDepth, out var halfWidth, out var halfHeight);
        var halfDepth = GetCameraDepthHalfRange(cameraToUse, cameraDepth, halfWidth, halfHeight) * rangeScale;

        return _runtimeViewportWorldCenter
            + cameraToUse.transform.right * (normalizedX * halfWidth * rangeScale)
            + cameraToUse.transform.up * (normalizedY * halfHeight * rangeScale)
            + cameraToUse.transform.forward * (normalizedZ * halfDepth);
    }

    private Vector3 GetDriverInputPosition(Transform driver)
    {
        return positionInputSpace == PositionInputSpace.DriverWorldPosition
            ? driver.position
            : driver.localPosition;
    }

    private Vector3 GetSafeInputHalfRange()
    {
        const float minimumRange = 0.0001f;
        return new Vector3(
            Mathf.Max(Mathf.Abs(inputHalfRange.x), minimumRange),
            Mathf.Max(Mathf.Abs(inputHalfRange.y), minimumRange),
            Mathf.Max(Mathf.Abs(inputHalfRange.z), minimumRange));
    }

    private bool TryGetViewCamera(out Camera cameraToUse)
    {
        cameraToUse = viewCamera;
        if (cameraToUse == null && useMainCameraWhenEmpty)
        {
            cameraToUse = Camera.main;
        }

        return cameraToUse != null;
    }

    private void CaptureViewportWorldCenter()
    {
        if (useDesiredCursorWorldPositionAsCenter && TryGetCursorWorldPositioner(out var positioner))
        {
            _runtimeViewportWorldCenter = positioner.DesiredCursorWorldPosition;
            return;
        }

        _runtimeViewportWorldCenter = viewportWorldCenter;
    }

    private bool TryGetCursorWorldPositioner(out HapticCursorWorldPositioner positioner)
    {
        positioner = cursorWorldPositioner;
        if (positioner == null)
        {
            positioner = FindObjectOfType<HapticCursorWorldPositioner>();
            cursorWorldPositioner = positioner;
        }

        return positioner != null;
    }

    private void CaptureCameraDepth()
    {
        if (!TryGetViewCamera(out var cameraToUse))
        {
            _cameraDepth = 0f;
            return;
        }

        _cameraDepth = Vector3.Dot(_runtimeViewportWorldCenter - cameraToUse.transform.position, cameraToUse.transform.forward);
    }

    private float GetSafeCameraDepth(Camera cameraToUse)
    {
        if (_cameraDepth > cameraToUse.nearClipPlane)
        {
            return _cameraDepth;
        }

        if (_targetTransform != null)
        {
            var currentDepth = Vector3.Dot(_targetTransform.position - cameraToUse.transform.position, cameraToUse.transform.forward);
            if (currentDepth > cameraToUse.nearClipPlane)
            {
                return currentDepth;
            }
        }

        return cameraToUse.nearClipPlane + 0.01f;
    }

    private static void GetCameraHalfExtents(Camera cameraToUse, float depth, out float halfWidth, out float halfHeight)
    {
        if (cameraToUse.orthographic)
        {
            halfHeight = cameraToUse.orthographicSize;
        }
        else
        {
            halfHeight = depth * Mathf.Tan(cameraToUse.fieldOfView * 0.5f * Mathf.Deg2Rad);
        }

        halfWidth = halfHeight * cameraToUse.aspect;
    }

    private float GetCameraDepthHalfRange(Camera cameraToUse, float centerDepth, float halfWidth, float halfHeight)
    {
        var depthFromViewSize = Mathf.Min(halfWidth, halfHeight) * depthRangeScale;
        var maxForwardDepth = Mathf.Max(0f, cameraToUse.farClipPlane - centerDepth);
        var maxBackwardDepth = Mathf.Max(0f, centerDepth - cameraToUse.nearClipPlane - 0.01f);
        var maxSymmetricDepth = Mathf.Min(maxForwardDepth, maxBackwardDepth);

        return Mathf.Min(depthFromViewSize, maxSymmetricDepth);
    }
}
