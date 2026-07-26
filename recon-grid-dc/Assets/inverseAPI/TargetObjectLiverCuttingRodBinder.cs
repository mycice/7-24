using ReconGridDC.Cuda;
using UnityEngine;

[DisallowMultipleComponent]
[RequireComponent(typeof(TargetObject))]
public sealed class TargetObjectLiverCuttingRodBinder : MonoBehaviour
{
    [Header("Binding")]
    [SerializeField] private TargetObject targetObject;
    [SerializeField] private LiverCudaCutter liverCudaCutter;
    [SerializeField] private bool bindOnStart = true;
    [SerializeField] private bool keepBindingUpdated = true;
    [SerializeField] private bool findCutterWhenEmpty = true;
    [SerializeField] private bool logBinding = true;

    private GameObject _lastBoundObject;

    public bool BindNow()
    {
        ResolveReferences();

        if (targetObject == null || liverCudaCutter == null || liverCudaCutter.cuttingRodTarget == null)
        {
            return false;
        }

        var rodObject = liverCudaCutter.cuttingRodTarget.gameObject;
        if (targetObject.TargetObjectReference == rodObject && _lastBoundObject == rodObject)
        {
            return true;
        }

        targetObject.SetTargetObject(rodObject);
        _lastBoundObject = rodObject;

        if (logBinding)
        {
            Debug.Log($"[TargetObject] Bound cutting rod target: {rodObject.name}");
        }

        return true;
    }

    private void Reset()
    {
        ResolveReferences();
    }

    private void Awake()
    {
        ResolveReferences();
    }

    private void Start()
    {
        if (bindOnStart)
        {
            BindNow();
        }
    }

    private void Update()
    {
        if (keepBindingUpdated)
        {
            BindNow();
        }
    }

    private void ResolveReferences()
    {
        if (targetObject == null)
        {
            targetObject = GetComponent<TargetObject>();
        }

        if (liverCudaCutter == null && findCutterWhenEmpty)
        {
            liverCudaCutter = FindObjectOfType<LiverCudaCutter>();
        }
    }
}
