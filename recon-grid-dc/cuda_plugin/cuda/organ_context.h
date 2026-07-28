// organ_context.h - isolated GPU-resident per-organ resource ownership.
// Migration phase 1 deliberately does not drive the existing solver or cutter.
#pragma once

#include <stdint.h>

enum OrganContextResult
{
    ORGAN_CONTEXT_OK = 0,
    ORGAN_CONTEXT_INVALID_HANDLE = -1001,
    ORGAN_CONTEXT_INVALID_ARGUMENT = -1002,
    ORGAN_CONTEXT_ALREADY_INITIALIZED = -1003,
    ORGAN_CONTEXT_CUDA_FAILURE = -1004,
    ORGAN_CONTEXT_NOT_INITIALIZED = -1005
};

// Field order mirrors CudaOrganContextBridge. All topology and rest-state arrays are immutable
// after this phase-1 upload; dynamic solver buffers are introduced in later migration phases.
struct OrganContextInitDesc
{
    int particleCount;
    int tetCount;
    int surfaceTriangleCount;
    int edgeConstraintCount;
    float density;
    float youngsModulus;
    float poissonsRatio;
    float damping;
};

struct OrganContextStats
{
    uint32_t handle;
    int initialized;
    int lastError;
    int particleCount;
    int tetCount;
    int surfaceTriangleCount;
    int edgeConstraintCount;
    uint64_t hostFingerprint;
    uint64_t deviceBytes;
    uint64_t uploadBytes;
    uint64_t uploadOperations;
    float uploadMilliseconds;
    float validationKernelMilliseconds;
    uint64_t restPositionHash;
    uint64_t topologyHash;
    uint64_t dynamicDeviceBytes;
    uint64_t xpbdStepCount;
    float lastXpbdMilliseconds;
    float totalXpbdMilliseconds;
    int lastNanCount;
};

// The layout is mirrored by CudaOrganContextBridge. Parameters intentionally keep the
// Stage1TetSoftBodyController names/semantics so the Unity and CUDA implementations can
// be compared without silently retuning the organ.
struct OrganContextXpbdParams
{
    int numSubSteps;
    int constraintIterations;
    float edgeCompliance;
    float youngsModulus;
    float poissonsRatio;
    float damping;
    float gravityX;
    float gravityY;
    float gravityZ;
    float groundY;
};

struct OrganContextComparisonStats
{
    int particleCount;
    int cudaNanCount;
    float maxError;
    float rmsError;
    float centroidError;
    float bboxMinError;
    float bboxMaxError;
    float bboxExtentError;
};

// Small per-fixed-step packet sent by Unity for the active gripper. Dynamic organ
// state never leaves CUDA for this API.
struct OrganContextToolCapsule
{
    float ax, ay, az, radius;
    float bx, by, bz, friction;
    float prevAx, prevAy, prevAz, unused0;
    float prevBx, prevBy, prevBz, unused1;
};

struct OrganContextToolContactParams
{
    int capsuleCount;
    int contactEnabled;
    int keepContactActiveWhenIdle;
    int useCandidateCulling;
    int contactIterations;
    int couplingPasses;
    int graspRequest;
    int releaseRequest;
    float contactDistance;
    float contactCompliance;
    float tangentialFriction;
    float tangentialDamping;
    float candidatePadding;
    float graspHeight;
    float graspCoreRadius;
    float graspFormDuration;
    float graspBoundsMinX, graspBoundsMinY, graspBoundsMinZ;
    float graspBoundsMaxX, graspBoundsMaxY, graspBoundsMaxZ;
    float frameCenterX, frameCenterY, frameCenterZ;
    float axisUX, axisUY, axisUZ;
    float axisVX, axisVY, axisVZ;
    float axisWX, axisWY, axisWZ;
};

struct OrganContextToolContactStats
{
    int activeCandidates;
    int contactCount;
    float maxContactDepth;
    int surfaceCandidateTriangles;
    int surfaceContactTriangles;
    float surfaceMaxContactDepth;
    int graspedParticleCount;
    uint64_t dispatchCount;
    uint64_t toolUploadBytes;
    uint64_t toolUploadOperations;
    float lastToolMilliseconds;
    float totalToolMilliseconds;
    int lastToolError;
};

// Immutable Stage-3 embedding data is prepared by C# once, then evaluated fully on CUDA.
// The grid itself remains owned by dc_recon/cut.cu; this context only writes its cornerPos.
struct OrganContextTetToGridDesc
{
    int cornerCount;
    float restCentroidX, restCentroidY, restCentroidZ;
};

struct OrganContextTetToGridStats
{
    int configured;
    int cornerCount;
    int mappedCorners;
    int fallbackCorners;
    uint64_t uploadBytes;
    uint64_t updateCount;
    float lastUpdateMilliseconds;
    float totalUpdateMilliseconds;
    int lastError;
};

int organ_context_create(uint32_t* outHandle);
int organ_context_destroy(uint32_t handle);
int organ_context_initialize(uint32_t handle,
                             const OrganContextInitDesc* desc,
                             const float* restPositions3,
                             const int* tetIds4,
                             const float* inverseMass,
                             const float* restVolumes,
                             const int* tetActive,
                             const int* surfaceTriangleIds3,
                             const int* edgeConstraintIds2,
                             const float* edgeRestLengths);
int organ_context_get_stats(uint32_t handle, OrganContextStats* outStats);
int organ_context_xpbd_initialize(uint32_t handle,
                                  const float* initialPositions3,
                                  int edgeColorCount, const int* edgeColorOffsets,
                                  const int* edgeColorCounts, const int* edgeColorFlat,
                                  int tetColorCount, const int* tetColorOffsets,
                                  const int* tetColorCounts, const int* tetColorFlat,
                                  int surfaceColorCount, const int* surfaceColorOffsets,
                                  const int* surfaceColorCounts, const int* surfaceColorFlat);
int organ_context_xpbd_step(uint32_t handle, float dt, const OrganContextXpbdParams* params);
int organ_context_xpbd_get_positions(uint32_t handle, float* outPositions3, int particleCount);
int organ_context_xpbd_compare_positions(uint32_t handle, const float* unityPositions3,
                                         int particleCount, OrganContextComparisonStats* outStats);
int organ_context_tool_step(uint32_t handle, float dt, const OrganContextToolCapsule* capsules,
                            const OrganContextToolContactParams* params);
int organ_context_tool_get_stats(uint32_t handle, OrganContextToolContactStats* outStats);
int organ_context_tet_to_grid_configure(uint32_t handle, const OrganContextTetToGridDesc* desc,
                                        const int* hostTetByCorner, const float* barycentricWeights4,
                                        const float* alignedRestCorners3, const unsigned char* activeMask);
int organ_context_tet_to_grid_update(uint32_t handle, int applyLocalDeformation);
int organ_context_tet_to_grid_get_stats(uint32_t handle, OrganContextTetToGridStats* outStats);
