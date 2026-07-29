#pragma once

#include <stdint.h>

struct DirectSurfaceStats
{
    int supported;
    int configured;
    int registered;
    int capacity;
    uint64_t gpuCopyBytes;
    uint64_t dispatchCount;
    int lastError;
    int cudaDeviceOrdinal;
    int registrationStage;
    int cudaErrorCode;
    int positionByteWidth;
    int positionUsage;
    int positionBindFlags;
    int positionMiscFlags;
    int positionStructureByteStride;
    int rawVertexCount;
    int sourceVertexCountClamped;
    int rawTriangleCount;
    int validTriangleCount;
    int rejectedInvalidTriangleCount;
    int rejectedDegenerateTriangleCount;
    int outerTriangleCount;
    int cutWallTriangleCount;
    int writtenVertexCount;
    int capacityOverflow;
    int indexedVertexCount;
    int indexedIndexCount;
    int mergeEnabled;
    int outerIndexedVertexCount;
    int outerIndexedIndexCount;
    int cutWallIndexedVertexCount;
    int cutWallIndexedIndexCount;
    int topologyDiagnosticReady;
    int topologyDiagnosticOuterVertexSamples;
    int topologyDiagnosticUniqueOuterVertexEstimate;
    int topologyDiagnosticError;
    uint64_t topologyDiagnosticSequence;
    int rawNoWallTriangleCount;
    int rawAnyWallTriangleCount;
    int rawMixedWallTriangleCount;
};

int direct_surface_set_buffers(void* outerPositionBuffer, void* outerNormalBuffer, void* outerAuxBuffer,
                               void* outerIndexBuffer, void* outerIndexCountBuffer,
                               void* cutPositionBuffer, void* cutNormalBuffer, void* cutAuxBuffer,
                               void* cutIndexBuffer, void* cutIndexCountBuffer, int capacity);
void direct_surface_release();
void* direct_surface_get_render_event();
int direct_surface_get_stats(DirectSurfaceStats* outStats);
int direct_surface_request_topology_diagnostic();
