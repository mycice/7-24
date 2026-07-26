using System.Threading;
using Haply.Inverse.DeviceControllers;
using Haply.Inverse.DeviceData;
using UnityEngine;

/// <summary>
/// Haply Inverse3 force-feedback output gate.
///
/// This retained component intentionally contains no collision, contact, or grasp calculation.
/// A cutting workflow can provide a force through SetWorldForce; disabling the API immediately
/// sends zero force to the device.
/// </summary>
[DisallowMultipleComponent]
public sealed class CubeCollisionForceFeedback : MonoBehaviour
{
    [Header("Haply Force Feedback API")]
    [SerializeField] private bool enableForceFeedback = true;
    [SerializeField] private Inverse3Controller inverse3;

    private readonly ReaderWriterLockSlim _forceLock = new();
    private Vector3 _cachedWorldForce;

    public bool EnableForceFeedback
    {
        get => enableForceFeedback;
        set
        {
            enableForceFeedback = value;
            if (!enableForceFeedback)
            {
                SetWorldForce(Vector3.zero);
            }
        }
    }

    private void Awake()
    {
        inverse3 ??= FindObjectOfType<Inverse3Controller>();
    }

    private void OnEnable()
    {
        if (inverse3 == null)
        {
            inverse3 = FindObjectOfType<Inverse3Controller>();
        }

        if (inverse3 != null)
        {
            inverse3.DeviceStateChanged += OnDeviceStateChanged;
        }
    }

    private void OnDisable()
    {
        if (inverse3 != null)
        {
            inverse3.DeviceStateChanged -= OnDeviceStateChanged;
        }

        SetWorldForce(Vector3.zero);
    }

    /// <summary>Sets the force held by the Haply device callback until replaced or cleared.</summary>
    public void SetWorldForce(Vector3 worldForce)
    {
        _forceLock.EnterWriteLock();
        try
        {
            _cachedWorldForce = worldForce;
        }
        finally
        {
            _forceLock.ExitWriteLock();
        }
    }

    public void ClearForce()
    {
        SetWorldForce(Vector3.zero);
    }

    private void OnDeviceStateChanged(object sender, Inverse3EventArgs args)
    {
        args.DeviceController.SetCursorForce(enableForceFeedback ? GetWorldForce() : Vector3.zero);
    }

    private Vector3 GetWorldForce()
    {
        _forceLock.EnterReadLock();
        try
        {
            return _cachedWorldForce;
        }
        finally
        {
            _forceLock.ExitReadLock();
        }
    }
}
