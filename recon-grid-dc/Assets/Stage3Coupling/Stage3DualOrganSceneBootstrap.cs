using System.Collections;
using ReconGridDC.Cuda;
using ReconGridDC.Stage1TetPhysics;
using ReconGridDC.Stage2Grasping;
using UnityEngine;

namespace ReconGridDC.Stage3Coupling
{
    [DefaultExecutionOrder(-1000)]
    [DisallowMultipleComponent]
    public sealed class Stage3DualOrganSceneBootstrap : MonoBehaviour
    {
        [Header("Dual Organ")]
        [Tooltip("Create the second organ when Play starts. Change this only outside Play mode, then enter Play again.")]
        public bool enableSecondOrgan = true;
        public Stage1TetSoftBodyController primaryTetSoftBody;
        public LiverCudaManager primaryCudaLiver;
        public Stage3TetToGridEmbeddingBridge primaryCoupling;
        public Vector3 secondaryOffset = new Vector3(5f, 5f, 5f);

        [Header("Dual Organ GPU Scheduling")]
        public bool reduceIdleOrganGpuWork = true;
        [Range(2, 12)] public int idleCudaXpbdStepInterval = 3;
        [Range(2, 12)] public int idleSurfaceUpdateInterval = 3;
        [Min(0f)] public float cuttingRodActivityPadding = 0.5f;
        [Range(0, 10)] public int activeHoldFrames = 2;

        void Awake()
        {
            if (!enableSecondOrgan)
            {
                Debug.Log("[Stage3DualOrgan] Second organ disabled; the original single-organ path is active.", this);
                return;
            }

            if (primaryTetSoftBody == null || primaryCudaLiver == null || primaryCoupling == null)
            {
                Debug.LogError("[Stage3DualOrgan] Primary organ references are incomplete.", this);
                return;
            }

            Stage1TetSoftBodyController secondaryTet = Instantiate(primaryTetSoftBody, Vector3.zero,
                Quaternion.identity, primaryTetSoftBody.transform.parent);
            secondaryTet.name = "Stage3 Tet Soft Body 2 (Physics Source)";
            secondaryTet.initialOffset = primaryTetSoftBody.initialOffset + secondaryOffset;
            secondaryTet.showGroundPlane = false;
            CudaOrganContextBridge primaryContext = primaryTetSoftBody.GetComponent<CudaOrganContextBridge>();
            CudaOrganContextBridge secondaryContext = secondaryTet.GetComponent<CudaOrganContextBridge>();
            secondaryContext.pluginInstance = CudaPluginInstance.Secondary;
            if (primaryContext != null)
                primaryContext.ConfigureOrganToolBroadphase(true);
            secondaryContext.ConfigureOrganToolBroadphase(true);
            Stage2TetGraspingBridge secondaryGrasping = secondaryTet.GetComponent<Stage2TetGraspingBridge>();
            if (secondaryGrasping != null)
                secondaryGrasping.initializeGripperTool = false;

            LiverCudaManager secondaryLiver = Instantiate(primaryCudaLiver, Vector3.zero,
                Quaternion.identity, primaryCudaLiver.transform.parent);
            secondaryLiver.gameObject.SetActive(false);
            secondaryLiver.name = "Stage3 CUDA Grid Liver 2 (Cutting and Rendering)";
            secondaryLiver.pluginInstance = CudaPluginInstance.Secondary;
            secondaryLiver.initialWorldOffset = primaryCudaLiver.initialWorldOffset + secondaryOffset;
            secondaryLiver.setupOrbitCamera = false;
            secondaryLiver.showPerformanceHud = false;
            secondaryLiver.matchReferenceLighting = false;
            secondaryLiver.matchReferenceSkybox = false;
            secondaryLiver.matchReferenceFog = false;
            secondaryLiver.createReferenceGround = false;

            LiverCudaCutter primaryCutter = primaryCudaLiver.GetComponent<LiverCudaCutter>();
            LiverCudaCutter secondaryCutter = secondaryLiver.GetComponent<LiverCudaCutter>();
            if (primaryCutter == null || secondaryCutter == null)
            {
                Debug.LogError("[Stage3DualOrgan] A LiverCudaCutter component is missing.", this);
                Destroy(secondaryLiver.gameObject);
                Destroy(secondaryTet.gameObject);
                return;
            }

            LiverCuttingRod sharedRod = primaryCutter.GetOrCreateCuttingRodTarget();
            if (sharedRod == null)
            {
                Debug.LogError("[Stage3DualOrgan] The primary cutting rod could not be created.", this);
                Destroy(secondaryLiver.gameObject);
                Destroy(secondaryTet.gameObject);
                return;
            }
            secondaryCutter.BindSharedCuttingRod(sharedRod);

            Stage3TetToGridEmbeddingBridge secondaryCoupling = Instantiate(primaryCoupling,
                Vector3.zero, Quaternion.identity, primaryCoupling.transform.parent);
            secondaryCoupling.gameObject.SetActive(false);
            secondaryCoupling.name = "Stage3 Tet-To-Grid Coupling 2";
            secondaryCoupling.tetSoftBody = secondaryTet;
            secondaryCoupling.cudaLiver = secondaryLiver;

            if (reduceIdleOrganGpuWork && primaryContext != null && secondaryContext != null)
            {
                primaryCudaLiver.ConfigureAdaptiveGpuWorkScheduling(
                    primaryContext,
                    idleSurfaceUpdateInterval,
                    cuttingRodActivityPadding,
                    activeHoldFrames);
                secondaryLiver.ConfigureAdaptiveGpuWorkScheduling(
                    secondaryContext,
                    idleSurfaceUpdateInterval,
                    cuttingRodActivityPadding,
                    activeHoldFrames);
                primaryContext.ConfigureAdaptiveCudaXpbdScheduling(
                    primaryCudaLiver, idleCudaXpbdStepInterval);
                secondaryContext.ConfigureAdaptiveCudaXpbdScheduling(
                    secondaryLiver, idleCudaXpbdStepInterval);
            }

            StartCoroutine(ActivateSecondaryAfterPrimaryInitialization(secondaryLiver, secondaryCoupling));
        }

        IEnumerator ActivateSecondaryAfterPrimaryInitialization(
            LiverCudaManager secondaryLiver,
            Stage3TetToGridEmbeddingBridge secondaryCoupling)
        {
            while (primaryCudaLiver != null && !primaryCudaLiver.IsInitialized)
                yield return null;

            if (primaryCudaLiver == null || secondaryLiver == null || secondaryCoupling == null)
                yield break;

            secondaryLiver.gameObject.SetActive(true);
            secondaryCoupling.gameObject.SetActive(true);
            Debug.Log($"[Stage3DualOrgan] Secondary organ activated with world offset {secondaryOffset}, one shared cutting rod, and isolated LiverCudaSim2 runtime.", this);
        }
    }
}
