using UnityEngine;

/// <summary>
/// Maps Haply input proxy transforms to a target object.
/// Position is driven by the Inverse3 cursor, rotation is driven by the stylus/VerseGrip cursor.
/// This script intentionally depends only on UnityEngine so it works across Haply Unity SDK versions.
/// </summary>
public class HaplyPoseTargetDriver : MonoBehaviour
{
    public enum UpdateMode
    {
        Update,
        FixedUpdate,
        LateUpdate
    }

    [Header("Input Sources")]
    [Tooltip("Transform driven by the Haply Inverse3 cursor. Used for target position.")]
    public Transform positionSource;

    [Tooltip("Transform driven by the Haply stylus / VerseGrip cursor. Used for target rotation.")]
    public Transform rotationSource;

    [Header("Target")]
    public Transform target;
    public Rigidbody targetRigidbody;
    public UpdateMode updateMode = UpdateMode.Update;

    [Header("Position Mapping")]
    [Min(0f)]
    public float positionGain = 1.0f;
    public Vector3 positionOffset = Vector3.zero;
    public bool lockX;
    public bool lockY;
    public bool lockZ;

    [Header("Rotation Mapping")]
    public bool useStylusRotation = true;
    public Vector3 modelRotationOffsetEuler = Vector3.zero;

    [Header("Smoothing")]
    [Tooltip("0 = no smoothing, larger values = smoother/slower.")]
    [Range(0f, 0.25f)]
    public float positionSmoothTime = 0.02f;

    [Tooltip("Rotation interpolation speed. 0 = snap directly.")]
    [Min(0f)]
    public float rotationLerpSpeed = 20f;

    [Header("Calibration")]
    public bool calibrateOnStart = true;
    public KeyCode recalibrateKey = KeyCode.C;

    private Vector3 _sourcePositionAtCalibration;
    private Vector3 _targetPositionAtCalibration;
    private Quaternion _sourceRotationAtCalibration = Quaternion.identity;
    private Quaternion _targetRotationAtCalibration = Quaternion.identity;
    private Vector3 _positionVelocity;
    private bool _hasCalibration;

    private void Reset()
    {
        target = transform;
        targetRigidbody = GetComponent<Rigidbody>();
    }

    private void Awake()
    {
        if (target == null)
            target = transform;

        if (targetRigidbody == null && target != null)
            targetRigidbody = target.GetComponent<Rigidbody>();
    }

    private void Start()
    {
        if (calibrateOnStart)
            Calibrate();
    }

    private void Update()
    {
        if (Input.GetKeyDown(recalibrateKey))
            Calibrate();

        if (updateMode == UpdateMode.Update)
            ApplyPose(Time.deltaTime);
    }

    private void FixedUpdate()
    {
        if (updateMode == UpdateMode.FixedUpdate)
            ApplyPose(Time.fixedDeltaTime);
    }

    private void LateUpdate()
    {
        if (updateMode == UpdateMode.LateUpdate)
            ApplyPose(Time.deltaTime);
    }

    public void Calibrate()
    {
        if (target == null)
            target = transform;

        if (positionSource != null)
            _sourcePositionAtCalibration = positionSource.position;

        _targetPositionAtCalibration = target.position;

        if (rotationSource != null)
            _sourceRotationAtCalibration = rotationSource.rotation;

        _targetRotationAtCalibration = target.rotation;
        _positionVelocity = Vector3.zero;
        _hasCalibration = true;
    }

    private void ApplyPose(float deltaTime)
    {
        if (target == null)
            return;

        if (!_hasCalibration)
            Calibrate();

        Vector3 desiredPosition = target.position;
        Quaternion desiredRotation = target.rotation;

        if (positionSource != null)
        {
            Vector3 sourceDelta = positionSource.position - _sourcePositionAtCalibration;
            desiredPosition = _targetPositionAtCalibration + sourceDelta * positionGain + positionOffset;

            if (lockX) desiredPosition.x = target.position.x;
            if (lockY) desiredPosition.y = target.position.y;
            if (lockZ) desiredPosition.z = target.position.z;
        }

        if (useStylusRotation && rotationSource != null)
        {
            Quaternion stylusDelta = rotationSource.rotation * Quaternion.Inverse(_sourceRotationAtCalibration);
            Quaternion modelAxisOffset = Quaternion.Euler(modelRotationOffsetEuler);
            desiredRotation = stylusDelta * _targetRotationAtCalibration * modelAxisOffset;
        }

        MoveTarget(desiredPosition, desiredRotation, deltaTime);
    }

    private void MoveTarget(Vector3 desiredPosition, Quaternion desiredRotation, float deltaTime)
    {
        Vector3 nextPosition = desiredPosition;
        Quaternion nextRotation = desiredRotation;

        if (positionSmoothTime > 0f)
        {
            nextPosition = Vector3.SmoothDamp(
                target.position,
                desiredPosition,
                ref _positionVelocity,
                positionSmoothTime,
                Mathf.Infinity,
                deltaTime);
        }

        if (rotationLerpSpeed > 0f)
        {
            float t = 1f - Mathf.Exp(-rotationLerpSpeed * deltaTime);
            nextRotation = Quaternion.Slerp(target.rotation, desiredRotation, t);
        }

        if (targetRigidbody != null && !targetRigidbody.isKinematic)
        {
            targetRigidbody.MovePosition(nextPosition);
            targetRigidbody.MoveRotation(nextRotation);
            return;
        }

        target.SetPositionAndRotation(nextPosition, nextRotation);
    }
}
