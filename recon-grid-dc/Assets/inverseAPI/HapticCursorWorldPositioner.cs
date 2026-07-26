using System.Collections;
using UnityEngine;

[DisallowMultipleComponent]
[DefaultExecutionOrder(10000)]
public class HapticCursorWorldPositioner : MonoBehaviour
{
    [Header("Required References")]
    [SerializeField] private Transform hapticOrigin;
    [SerializeField] private Transform cursor;

    [Header("World Bounds Mapping")]
    [SerializeField] private bool useWorldBoundsMapping = true;
    [SerializeField] private bool waitForStableCursorBeforeMapping = true;
    [SerializeField] private bool centerInputRangeAfterStartup = true;
    [SerializeField] private float worldMinX = -1f;
    [SerializeField] private float worldMaxX = 1f;
    [SerializeField] private float worldMinY = -1f;
    [SerializeField] private float worldMaxY = 1f;
    [SerializeField] private float worldMinZ = -1f;
    [SerializeField] private float worldMaxZ = 1f;
    [SerializeField] private bool clampToWorldBounds = true;

    [Header("Inverse3 Input Range")]
    [SerializeField] private Vector3 inputLocalMin = new Vector3(-0.22f, -0.22f, -0.22f);
    [SerializeField] private Vector3 inputLocalMax = new Vector3(0.22f, 0.22f, 0.22f);

    [Header("Target World Position")]
    [SerializeField] private Vector3 desiredCursorWorldPosition = Vector3.zero;
    [SerializeField] private bool alignOnStart = true;
    [SerializeField] private bool continuousAlignDuringStartup = true;
    [SerializeField] private bool createRuntimeCorrectionRoot = true;

    [Header("Settle Detection")]
    [SerializeField] private float initialDelaySeconds = 0.5f;
    [SerializeField] private float minimumAlignmentSeconds = 4f;
    [SerializeField] private int stableFrameCount = 15;
    [SerializeField] private float stablePositionEpsilon = 0.0005f;
    [SerializeField] private float timeoutSeconds = 5f;

    [Header("Debug")]
    [SerializeField] private bool logAlignment = true;
    [SerializeField] private Vector3 currentCursorLocalPosition;
    [SerializeField] private Vector3 currentCursorWorldPosition;
    [SerializeField] private Vector3 currentHapticOriginPosition;
    [SerializeField] private Vector3 requiredHapticOriginPosition;
    [SerializeField] private Vector3 currentNormalizedInput;
    [SerializeField] private Vector3 mappedCursorWorldPosition;
    [SerializeField] private Vector3 observedInputLocalMin;
    [SerializeField] private Vector3 observedInputLocalMax;

    private Coroutine _alignRoutine;
    private Transform _correctionFrame;
    private bool _hasObservedInput;
    private bool _worldBoundsMappingReady;
    private bool _hasCalibratedInputCenter;
    private Vector3 _calibratedInputLocalCenter;
    private Vector3 _activeCursorWorldTarget;

    public Transform HapticOrigin
    {
        get => hapticOrigin;
        set => hapticOrigin = value;
    }

    public Transform Cursor
    {
        get => cursor;
        set => cursor = value;
    }

    public Vector3 DesiredCursorWorldPosition
    {
        get => desiredCursorWorldPosition;
        set
        {
            desiredCursorWorldPosition = value;
            if (!useWorldBoundsMapping || !_worldBoundsMappingReady)
            {
                mappedCursorWorldPosition = value;
                _activeCursorWorldTarget = value;
            }
        }
    }

    public Vector3 MappedCursorWorldPosition => mappedCursorWorldPosition;

    public bool UseWorldBoundsMapping
    {
        get => useWorldBoundsMapping;
        set
        {
            useWorldBoundsMapping = value;
            _worldBoundsMappingReady = !useWorldBoundsMapping || !waitForStableCursorBeforeMapping;
        }
    }

    public float WorldMinX
    {
        get => worldMinX;
        set => worldMinX = value;
    }

    public float WorldMaxX
    {
        get => worldMaxX;
        set => worldMaxX = value;
    }

    public float WorldMinY
    {
        get => worldMinY;
        set => worldMinY = value;
    }

    public float WorldMaxY
    {
        get => worldMaxY;
        set => worldMaxY = value;
    }

    public float WorldMinZ
    {
        get => worldMinZ;
        set => worldMinZ = value;
    }

    public float WorldMaxZ
    {
        get => worldMaxZ;
        set => worldMaxZ = value;
    }

    public Vector3 InputLocalMin
    {
        get => inputLocalMin;
        set => inputLocalMin = value;
    }

    public Vector3 InputLocalMax
    {
        get => inputLocalMax;
        set => inputLocalMax = value;
    }

    public void SetWorldBounds(float minX, float maxX, float minY, float maxY, float minZ, float maxZ)
    {
        worldMinX = minX;
        worldMaxX = maxX;
        worldMinY = minY;
        worldMaxY = maxY;
        worldMinZ = minZ;
        worldMaxZ = maxZ;
    }

    private void Reset()
    {
        hapticOrigin = transform;
    }

    private void OnEnable()
    {
        EnsureCorrectionFrame();
        _worldBoundsMappingReady = !useWorldBoundsMapping || !waitForStableCursorBeforeMapping;
        _hasCalibratedInputCenter = false;
        mappedCursorWorldPosition = desiredCursorWorldPosition;
        _activeCursorWorldTarget = desiredCursorWorldPosition;

        if (useWorldBoundsMapping)
        {
            ResetObservedInputRange();
            if (alignOnStart && waitForStableCursorBeforeMapping)
            {
                StartAlignmentRoutine();
            }
        }
        else if (alignOnStart)
        {
            StartAlignmentRoutine();
        }
    }

    private void OnDisable()
    {
        if (_alignRoutine != null)
        {
            StopCoroutine(_alignRoutine);
            _alignRoutine = null;
        }
    }

    private void LateUpdate()
    {
        if (useWorldBoundsMapping && HasRequiredReferences() && _worldBoundsMappingReady)
        {
            ApplyWorldBoundsMapping();
        }
        else if (continuousAlignDuringStartup && _alignRoutine != null && HasRequiredReferences())
        {
            ApplyAlignment(false);
        }

        UpdateDebugValues();
    }

    [ContextMenu("Start Alignment Routine")]
    public void StartAlignmentRoutine()
    {
        if (_alignRoutine != null)
        {
            StopCoroutine(_alignRoutine);
        }

        _alignRoutine = StartCoroutine(AlignAfterCursorSettles());
    }

    [ContextMenu("Align Cursor Now")]
    public void AlignCursorNow()
    {
        if (!HasRequiredReferences())
        {
            Debug.LogWarning("HapticCursorWorldPositioner requires Haptic Origin and Cursor references.", this);
            return;
        }

        ApplyAlignment(true);
        UpdateDebugValues();
    }

    [ContextMenu("Use Observed Input Range")]
    public void UseObservedInputRange()
    {
        if (!_hasObservedInput)
        {
            Debug.LogWarning("No observed input range has been recorded yet. Move the inverse3 cursor first.", this);
            return;
        }

        inputLocalMin = observedInputLocalMin;
        inputLocalMax = observedInputLocalMax;
    }

    [ContextMenu("Reset Observed Input Range")]
    public void ResetObservedInputRange()
    {
        _hasObservedInput = false;
        observedInputLocalMin = Vector3.zero;
        observedInputLocalMax = Vector3.zero;
    }

    private void ApplyWorldBoundsMapping()
    {
        var input = cursor.localPosition;
        UpdateObservedInputRange(input);

        if (!_hasCalibratedInputCenter)
        {
            CalibrateInputCenter(input);
        }

        currentNormalizedInput = new Vector3(
            NormalizeInputDelta(input.x, _calibratedInputLocalCenter.x, inputLocalMin.x, inputLocalMax.x),
            NormalizeInputDelta(input.y, _calibratedInputLocalCenter.y, inputLocalMin.y, inputLocalMax.y),
            NormalizeInputDelta(input.z, _calibratedInputLocalCenter.z, inputLocalMin.z, inputLocalMax.z));

        mappedCursorWorldPosition = new Vector3(
            MapAxisAroundDesired(desiredCursorWorldPosition.x, worldMinX, worldMaxX, currentNormalizedInput.x),
            MapAxisAroundDesired(desiredCursorWorldPosition.y, worldMinY, worldMaxY, currentNormalizedInput.y),
            MapAxisAroundDesired(desiredCursorWorldPosition.z, worldMinZ, worldMaxZ, currentNormalizedInput.z));

        ApplyAlignment(mappedCursorWorldPosition, false);
    }

    private float NormalizeInputDelta(float value, float center, float min, float max)
    {
        float negativeRange = Mathf.Max(center - min, 0.000001f);
        float positiveRange = Mathf.Max(max - center, 0.000001f);
        float delta = value - center;
        float normalized = delta >= 0f ? delta / positiveRange : delta / negativeRange;
        return clampToWorldBounds ? Mathf.Clamp(normalized, -1f, 1f) : normalized;
    }

    private float MapAxisAroundDesired(float desired, float min, float max, float normalizedDelta)
    {
        if (normalizedDelta >= 0f)
        {
            return Mathf.Lerp(desired, max, normalizedDelta);
        }

        return Mathf.Lerp(desired, min, -normalizedDelta);
    }

    private void UpdateObservedInputRange(Vector3 input)
    {
        if (!_hasObservedInput)
        {
            observedInputLocalMin = input;
            observedInputLocalMax = input;
            _hasObservedInput = true;
            return;
        }

        observedInputLocalMin = Vector3.Min(observedInputLocalMin, input);
        observedInputLocalMax = Vector3.Max(observedInputLocalMax, input);
    }

    private void ApplyAlignment(bool shouldLog)
    {
        ApplyAlignment(desiredCursorWorldPosition, shouldLog);
    }

    private void ApplyAlignment(Vector3 targetWorldPosition, bool shouldLog)
    {
        var frameToMove = GetFrameToMove();
        _activeCursorWorldTarget = targetWorldPosition;
        var correction = targetWorldPosition - cursor.position;
        frameToMove.position += correction;

        if (shouldLog && logAlignment)
        {
            Debug.Log(
                $"Haptic cursor aligned. Cursor world: {cursor.position}, Frame: {frameToMove.position}",
                this);
        }
    }

    private IEnumerator AlignAfterCursorSettles()
    {
        if (!HasRequiredReferences())
        {
            Debug.LogWarning("HapticCursorWorldPositioner requires Haptic Origin and Cursor references.", this);
            _alignRoutine = null;
            yield break;
        }

        if (initialDelaySeconds > 0f)
        {
            yield return new WaitForSeconds(initialDelaySeconds);
        }

        var stableFrames = 0;
        var elapsed = 0f;
        var previousLocalPosition = cursor.localPosition;

        while (elapsed < timeoutSeconds)
        {
            yield return null;
            elapsed += Time.deltaTime;

            var currentLocalPosition = cursor.localPosition;
            if (Vector3.Distance(currentLocalPosition, previousLocalPosition) <= stablePositionEpsilon)
            {
                stableFrames++;
                if (elapsed >= minimumAlignmentSeconds && stableFrames >= stableFrameCount)
                {
                    break;
                }
            }
            else
            {
                stableFrames = 0;
            }

            previousLocalPosition = currentLocalPosition;
        }

        AlignCursorNow();
        if (useWorldBoundsMapping)
        {
            if (centerInputRangeAfterStartup)
            {
                CenterInputRangeOnCurrentCursor();
            }
            _hasCalibratedInputCenter = false;
            _worldBoundsMappingReady = true;
        }
        _alignRoutine = null;
    }

    private void CenterInputRangeOnCurrentCursor()
    {
        var current = cursor.localPosition;
        var halfRange = GetInputHalfRange();
        inputLocalMin = current - halfRange;
        inputLocalMax = current + halfRange;
        CalibrateInputCenter(current);
    }

    private void CalibrateInputCenter(Vector3 input)
    {
        _calibratedInputLocalCenter = input;
        _hasCalibratedInputCenter = true;
    }

    private Vector3 GetInputHalfRange()
    {
        return new Vector3(
            Mathf.Max(Mathf.Abs(inputLocalMax.x - inputLocalMin.x) * 0.5f, 0.000001f),
            Mathf.Max(Mathf.Abs(inputLocalMax.y - inputLocalMin.y) * 0.5f, 0.000001f),
            Mathf.Max(Mathf.Abs(inputLocalMax.z - inputLocalMin.z) * 0.5f, 0.000001f));
    }

    private bool HasRequiredReferences()
    {
        return hapticOrigin != null && cursor != null;
    }

    private void EnsureCorrectionFrame()
    {
        if (!createRuntimeCorrectionRoot || hapticOrigin == null || _correctionFrame != null)
        {
            return;
        }

        var originalParent = hapticOrigin.parent;
        var correctionRoot = new GameObject($"{hapticOrigin.name} Runtime World Offset").transform;
        correctionRoot.SetParent(originalParent, false);
        correctionRoot.localPosition = Vector3.zero;
        correctionRoot.localRotation = Quaternion.identity;
        correctionRoot.localScale = Vector3.one;

        hapticOrigin.SetParent(correctionRoot, true);
        _correctionFrame = correctionRoot;
    }

    private Transform GetFrameToMove()
    {
        return _correctionFrame != null ? _correctionFrame : hapticOrigin;
    }

    private void UpdateDebugValues()
    {
        if (cursor != null)
        {
            currentCursorLocalPosition = cursor.localPosition;
            currentCursorWorldPosition = cursor.position;
        }

        if (hapticOrigin != null)
        {
            currentHapticOriginPosition = hapticOrigin.position;
        }

        if (hapticOrigin != null && cursor != null)
        {
            var frameToMove = GetFrameToMove();
            requiredHapticOriginPosition = frameToMove.position + _activeCursorWorldTarget - cursor.position;
        }
    }
}
