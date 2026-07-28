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
};

int direct_surface_set_buffers(void* positionBuffer, void* normalBuffer, void* auxBuffer,
                               void* vertexCountBuffer, int capacity);
void direct_surface_release();
void* direct_surface_get_render_event();
int direct_surface_get_stats(DirectSurfaceStats* outStats);
