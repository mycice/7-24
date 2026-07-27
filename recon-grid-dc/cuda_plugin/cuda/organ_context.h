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
