#include <d3d11.h>
#include <cuda_d3d11_interop.h>
#include <cub/cub.cuh>

#include "direct_surface_interop.h"
#include "dc_recon.h"
#include <IUnityInterface.h>
#include <IUnityGraphics.h>
#include <IUnityGraphicsD3D11.h>

namespace
{
    constexpr int kBatchResourceCount = 5;
    constexpr int kResourceCount = kBatchResourceCount * 2;

    enum BufferSlot { Position = 0, Normal, Aux, Index, IndexCount };
    enum FrameCounter
    {
        CounterRawVertices = 0, CounterClampedVertices, CounterRawTriangles, CounterValidTriangles,
        CounterRejectedInvalid, CounterRejectedDegenerate, CounterOuterTriangles, CounterCutWallTriangles,
        CounterWrittenVertices, CounterCapacityOverflow, CounterRawNoWallTriangles,
        CounterRawAnyWallTriangles, CounterRawMixedWallTriangles, CounterCount
    };
    enum RegistrationStage
    {
        RegisterNone = 0, RegisterResolveD3DDevice = 1, RegisterSelectCudaDevice = 2,
        RegisterOuterPosition = 3, RegisterOuterNormal = 4, RegisterOuterAux = 5,
        RegisterOuterIndex = 6, RegisterOuterIndexCount = 7, RegisterCutPosition = 8,
        RegisterCutNormal = 9, RegisterCutAux = 10, RegisterCutIndex = 11, RegisterCutIndexCount = 12
    };

    cudaGraphicsResource* g_resources[kResourceCount]{};
    ID3D11Buffer* g_buffers[kResourceCount]{};
    DirectSurfaceStats g_stats{};

    unsigned int* g_frameCounters = nullptr;
    float3* g_outerCompactPositions = nullptr;
    float4* g_outerCompactAux = nullptr;
    unsigned int* g_outerCompactCount = nullptr;
    float3* g_cutCompactPositions = nullptr;
    float4* g_cutCompactAux = nullptr;
    unsigned int* g_cutCompactCount = nullptr;

    unsigned long long* g_outerKeysIn = nullptr;
    unsigned long long* g_outerKeysOut = nullptr;
    // Six quantized fields per compact outer vertex. This exactly mirrors the CPU VertexKey:
    // rest/aux xyz followed by position xyz, all quantized at 1e-4.
    int* g_outerKeyFields = nullptr;
    unsigned int* g_outerValuesIn = nullptr;
    unsigned int* g_outerValuesOut = nullptr;
    unsigned int* g_outerGroupStarts = nullptr;
    unsigned int* g_outerGroupIds = nullptr;
    unsigned int* g_outerOriginalToMerged = nullptr;
    unsigned int* g_outerMergedCount = nullptr;
    float3* g_outerNormalSums = nullptr;
    void* g_sortTempStorage = nullptr;
    void* g_scanTempStorage = nullptr;
    size_t g_sortTempStorageBytes = 0;
    size_t g_scanTempStorageBytes = 0;
    int g_capacity = 0;
    bool g_topologyDiagnosticRequested = false;

    bool RecordCudaFailure(RegistrationStage stage, cudaError_t error)
    {
        g_stats.registrationStage = static_cast<int>(stage);
        g_stats.cudaErrorCode = static_cast<int>(error);
        g_stats.lastError = -4101;
        return false;
    }

    void ReleaseResources()
    {
        for (int i = 0; i < kResourceCount; ++i)
        {
            if (g_resources[i]) cudaGraphicsUnregisterResource(g_resources[i]);
            g_resources[i] = nullptr;
        }
        g_stats.registered = 0;
    }

    void ReleaseGpuWorkBuffers()
    {
        cudaFree(g_frameCounters); g_frameCounters = nullptr;
        cudaFree(g_outerCompactPositions); g_outerCompactPositions = nullptr;
        cudaFree(g_outerCompactAux); g_outerCompactAux = nullptr;
        cudaFree(g_outerCompactCount); g_outerCompactCount = nullptr;
        cudaFree(g_cutCompactPositions); g_cutCompactPositions = nullptr;
        cudaFree(g_cutCompactAux); g_cutCompactAux = nullptr;
        cudaFree(g_cutCompactCount); g_cutCompactCount = nullptr;
        cudaFree(g_outerKeysIn); g_outerKeysIn = nullptr;
        cudaFree(g_outerKeysOut); g_outerKeysOut = nullptr;
        cudaFree(g_outerKeyFields); g_outerKeyFields = nullptr;
        cudaFree(g_outerValuesIn); g_outerValuesIn = nullptr;
        cudaFree(g_outerValuesOut); g_outerValuesOut = nullptr;
        cudaFree(g_outerGroupStarts); g_outerGroupStarts = nullptr;
        cudaFree(g_outerGroupIds); g_outerGroupIds = nullptr;
        cudaFree(g_outerOriginalToMerged); g_outerOriginalToMerged = nullptr;
        cudaFree(g_outerMergedCount); g_outerMergedCount = nullptr;
        cudaFree(g_outerNormalSums); g_outerNormalSums = nullptr;
        cudaFree(g_sortTempStorage); g_sortTempStorage = nullptr;
        cudaFree(g_scanTempStorage); g_scanTempStorage = nullptr;
        g_sortTempStorageBytes = 0;
        g_scanTempStorageBytes = 0;
        g_capacity = 0;
    }

    bool EnsureGpuWorkBuffers(int capacity)
    {
        if (capacity <= 0) return false;
        if (g_capacity == capacity && g_frameCounters && g_outerCompactPositions && g_cutCompactPositions &&
            g_outerKeysIn && g_outerKeysOut && g_outerKeyFields && g_outerValuesIn && g_outerValuesOut &&
            g_outerGroupStarts && g_outerGroupIds && g_outerOriginalToMerged && g_outerMergedCount && g_outerNormalSums &&
            g_sortTempStorage && g_scanTempStorage)
            return true;

        ReleaseGpuWorkBuffers();
        cudaError_t error = cudaMalloc(&g_frameCounters, CounterCount * sizeof(unsigned int));
        if (error == cudaSuccess) error = cudaMalloc(&g_outerCompactPositions, capacity * sizeof(float3));
        if (error == cudaSuccess) error = cudaMalloc(&g_outerCompactAux, capacity * sizeof(float4));
        if (error == cudaSuccess) error = cudaMalloc(&g_outerCompactCount, sizeof(unsigned int));
        if (error == cudaSuccess) error = cudaMalloc(&g_cutCompactPositions, capacity * sizeof(float3));
        if (error == cudaSuccess) error = cudaMalloc(&g_cutCompactAux, capacity * sizeof(float4));
        if (error == cudaSuccess) error = cudaMalloc(&g_cutCompactCount, sizeof(unsigned int));
        if (error == cudaSuccess) error = cudaMalloc(&g_outerKeysIn, capacity * sizeof(unsigned long long));
        if (error == cudaSuccess) error = cudaMalloc(&g_outerKeysOut, capacity * sizeof(unsigned long long));
        if (error == cudaSuccess) error = cudaMalloc(&g_outerKeyFields, capacity * 6u * sizeof(int));
        if (error == cudaSuccess) error = cudaMalloc(&g_outerValuesIn, capacity * sizeof(unsigned int));
        if (error == cudaSuccess) error = cudaMalloc(&g_outerValuesOut, capacity * sizeof(unsigned int));
        if (error == cudaSuccess) error = cudaMalloc(&g_outerGroupStarts, capacity * sizeof(unsigned int));
        if (error == cudaSuccess) error = cudaMalloc(&g_outerGroupIds, capacity * sizeof(unsigned int));
        if (error == cudaSuccess) error = cudaMalloc(&g_outerOriginalToMerged, capacity * sizeof(unsigned int));
        if (error == cudaSuccess) error = cudaMalloc(&g_outerMergedCount, sizeof(unsigned int));
        if (error == cudaSuccess) error = cudaMalloc(&g_outerNormalSums, capacity * sizeof(float3));
        if (error != cudaSuccess) { ReleaseGpuWorkBuffers(); return false; }

        cub::DeviceRadixSort::SortPairs(nullptr, g_sortTempStorageBytes, g_outerKeysIn, g_outerKeysOut,
                                        g_outerValuesIn, g_outerValuesOut, capacity);
        cub::DeviceScan::ExclusiveSum(nullptr, g_scanTempStorageBytes, g_outerGroupStarts, g_outerGroupIds, capacity);
        error = cudaMalloc(&g_sortTempStorage, g_sortTempStorageBytes);
        if (error == cudaSuccess) error = cudaMalloc(&g_scanTempStorage, g_scanTempStorageBytes);
        if (error != cudaSuccess) { ReleaseGpuWorkBuffers(); return false; }
        g_capacity = capacity;
        return true;
    }

    bool RegisterResources()
    {
        if (g_stats.registered) return true;
        for (int i = 0; i < kResourceCount; ++i)
            if (!g_buffers[i]) { g_stats.lastError = -4100; return false; }
        if (g_stats.capacity <= 0) { g_stats.lastError = -4100; return false; }

        ID3D11Device* d3dDevice = nullptr;
        g_buffers[Position]->GetDevice(&d3dDevice);
        if (!d3dDevice) return RecordCudaFailure(RegisterResolveD3DDevice, cudaErrorUnknown);
        IDXGIDevice* dxgiDevice = nullptr;
        HRESULT result = d3dDevice->QueryInterface(__uuidof(IDXGIDevice), reinterpret_cast<void**>(&dxgiDevice));
        if (FAILED(result) || !dxgiDevice) { d3dDevice->Release(); return RecordCudaFailure(RegisterResolveD3DDevice, cudaErrorUnknown); }
        IDXGIAdapter* adapter = nullptr;
        result = dxgiDevice->GetAdapter(&adapter);
        dxgiDevice->Release();
        if (FAILED(result) || !adapter) { d3dDevice->Release(); return RecordCudaFailure(RegisterResolveD3DDevice, cudaErrorUnknown); }
        int cudaDevice = -1;
        cudaError_t error = cudaD3D11GetDevice(&cudaDevice, adapter);
        adapter->Release(); d3dDevice->Release();
        if (error != cudaSuccess) return RecordCudaFailure(RegisterResolveD3DDevice, error);
        error = cudaSetDevice(cudaDevice);
        if (error != cudaSuccess) return RecordCudaFailure(RegisterSelectCudaDevice, error);
        g_stats.cudaDeviceOrdinal = cudaDevice;

        D3D11_BUFFER_DESC positionDesc{};
        g_buffers[Position]->GetDesc(&positionDesc);
        g_stats.positionByteWidth = static_cast<int>(positionDesc.ByteWidth);
        g_stats.positionUsage = static_cast<int>(positionDesc.Usage);
        g_stats.positionBindFlags = static_cast<int>(positionDesc.BindFlags);
        g_stats.positionMiscFlags = static_cast<int>(positionDesc.MiscFlags);
        g_stats.positionStructureByteStride = static_cast<int>(positionDesc.StructureByteStride);

        for (int i = 0; i < kResourceCount; ++i)
        {
            error = cudaGraphicsD3D11RegisterResource(&g_resources[i], g_buffers[i], cudaGraphicsRegisterFlagsNone);
            if (error != cudaSuccess)
            {
                ReleaseResources();
                return RecordCudaFailure(static_cast<RegistrationStage>(RegisterOuterPosition + i), error);
            }
        }
        g_stats.registered = 1;
        g_stats.registrationStage = RegisterNone;
        g_stats.cudaErrorCode = 0;
        return true;
    }

    __device__ bool IsFiniteFloat3(float3 v) { return isfinite(v.x) && isfinite(v.y) && isfinite(v.z); }

    __global__ void CompactByBatchKernel(const float3* sourcePositions, const float4* sourceAux,
                                         const unsigned int* sourceVertexCount,
                                         float3* outerPositions, float4* outerAux, unsigned int* outerCount,
                                         float3* cutPositions, float4* cutAux, unsigned int* cutCount,
                                         unsigned int* counters, unsigned int capacity)
    {
        unsigned int triangle = blockIdx.x * blockDim.x + threadIdx.x;
        unsigned int rawCount = *sourceVertexCount;
        unsigned int sourceCount = min(rawCount, capacity);
        if (triangle == 0u)
        {
            counters[CounterRawVertices] = rawCount;
            counters[CounterClampedVertices] = sourceCount - sourceCount % 3u;
            counters[CounterRawTriangles] = rawCount / 3u;
            if (rawCount > capacity) atomicExch(&counters[CounterCapacityOverflow], 1u);
        }
        unsigned int start = triangle * 3u;
        if (start + 2u >= sourceCount) return;
        float3 p0 = sourcePositions[start], p1 = sourcePositions[start + 1u], p2 = sourcePositions[start + 2u];
        float3 a = make_float3(p1.x - p0.x, p1.y - p0.y, p1.z - p0.z);
        float3 b = make_float3(p2.x - p0.x, p2.y - p0.y, p2.z - p0.z);
        float3 face = make_float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
        float area2 = face.x * face.x + face.y * face.y + face.z * face.z;
        if (!IsFiniteFloat3(p0) || !IsFiniteFloat3(p1) || !IsFiniteFloat3(p2) || !isfinite(area2))
        { atomicAdd(&counters[CounterRejectedInvalid], 1u); return; }
        if (area2 < 1e-12f) { atomicAdd(&counters[CounterRejectedDegenerate], 1u); return; }
        bool wall0 = sourceAux[start].w >= 0.5f;
        bool wall1 = sourceAux[start + 1u].w >= 0.5f;
        bool wall2 = sourceAux[start + 2u].w >= 0.5f;
        unsigned int wallVertexCount = static_cast<unsigned int>(wall0) + static_cast<unsigned int>(wall1) + static_cast<unsigned int>(wall2);
        bool isCut = wallVertexCount != 0u;
        atomicAdd(&counters[isCut ? CounterRawAnyWallTriangles : CounterRawNoWallTriangles], 1u);
        if (wallVertexCount != 0u && wallVertexCount != 3u) atomicAdd(&counters[CounterRawMixedWallTriangles], 1u);
        unsigned int target = atomicAdd(isCut ? cutCount : outerCount, 3u);
        if (target + 2u >= capacity) { atomicExch(&counters[CounterCapacityOverflow], 1u); return; }
        float3* targetPositions = isCut ? cutPositions : outerPositions;
        float4* targetAux = isCut ? cutAux : outerAux;
        targetPositions[target] = p0; targetPositions[target + 1u] = p1; targetPositions[target + 2u] = p2;
        targetAux[target] = sourceAux[start]; targetAux[target + 1u] = sourceAux[start + 1u]; targetAux[target + 2u] = sourceAux[start + 2u];
        atomicAdd(&counters[CounterValidTriangles], 1u);
        atomicAdd(&counters[isCut ? CounterCutWallTriangles : CounterOuterTriangles], 1u);
        atomicAdd(&counters[CounterWrittenVertices], 3u);
    }

    __device__ unsigned long long HashQuantizedFields(const int* values)
    {
        unsigned long long hash = 1469598103934665603ull;
        #pragma unroll
        for (int i = 0; i < 6; ++i) { hash ^= static_cast<unsigned int>(values[i]); hash *= 1099511628211ull; }
        return hash;
    }

    __global__ void BuildOuterMergeInputsKernel(const float3* positions, const float4* aux, const unsigned int* count,
                                                unsigned long long* keys, int* keyFields, unsigned int* values, unsigned int capacity)
    {
        unsigned int v = blockIdx.x * blockDim.x + threadIdx.x;
        unsigned int n = min(*count, capacity);
        if (v < n)
        {
            const float q = 10000.0f;
            int* fields = keyFields + v * 6u;
            // Keep the exact CPU key order from DcLiverVisualRenderer.VertexKey.
            fields[0] = __float2int_rn(aux[v].x * q);
            fields[1] = __float2int_rn(aux[v].y * q);
            fields[2] = __float2int_rn(aux[v].z * q);
            fields[3] = __float2int_rn(positions[v].x * q);
            fields[4] = __float2int_rn(positions[v].y * q);
            fields[5] = __float2int_rn(positions[v].z * q);
            keys[v] = HashQuantizedFields(fields);
            values[v] = v;
        }
        else if (v < capacity) { keys[v] = 0xffffffffffffffffull; values[v] = v; }
    }

    __device__ bool EqualOuterKey(const int* keyFields, unsigned int a, unsigned int b)
    {
        const int* left = keyFields + a * 6u;
        const int* right = keyFields + b * 6u;
        #pragma unroll
        for (int i = 0; i < 6; ++i)
            if (left[i] != right[i]) return false;
        return true;
    }

    __global__ void MarkGroupStartsKernel(const unsigned long long* keys, const unsigned int* sortedValues,
                                          const int* keyFields, const unsigned int* count,
                                          unsigned int* starts, unsigned int capacity)
    {
        unsigned int rank = blockIdx.x * blockDim.x + threadIdx.x;
        unsigned int n = min(*count, capacity);
        if (rank >= n) return;
        if (rank == 0u || keys[rank] != keys[rank - 1u]) { starts[rank] = 1u; return; }
        starts[rank] = EqualOuterKey(keyFields, sortedValues[rank], sortedValues[rank - 1u]) ? 0u : 1u;
    }

    __global__ void FinalizeGroupIdsKernel(const unsigned int* starts, unsigned int* ids,
                                           const unsigned int* count, unsigned int capacity)
    {
        unsigned int rank = blockIdx.x * blockDim.x + threadIdx.x;
        if (rank < min(*count, capacity)) ids[rank] += starts[rank] - 1u;
    }

    __global__ void ScatterOuterVerticesKernel(const unsigned int* sortedValues, const unsigned int* starts,
                                               const unsigned int* ids, const unsigned int* count,
                                               const float3* compactPositions, const float4* compactAux,
                                               float3* positions, float4* aux, float3* normalSums,
                                               unsigned int* remap, unsigned int* mergedCount,
                                               unsigned int capacity)
    {
        unsigned int rank = blockIdx.x * blockDim.x + threadIdx.x;
        unsigned int n = min(*count, capacity);
        if (rank >= n) return;
        unsigned int original = sortedValues[rank], merged = ids[rank];
        remap[original] = merged;
        if (starts[rank])
        {
            positions[merged] = compactPositions[original];
            aux[merged] = compactAux[original];
            normalSums[merged] = make_float3(0.0f, 0.0f, 0.0f);
        }
        if (rank == n - 1u) *mergedCount = merged + 1u;
    }

    __global__ void BuildOuterTrianglesKernel(const float3* positions, const unsigned int* count,
                                              const unsigned int* remap, const unsigned int* mergedCount, float3* sums,
                                              unsigned int* indices, unsigned int* indexCount,
                                              unsigned int capacity)
    {
        unsigned int tri = blockIdx.x * blockDim.x + threadIdx.x;
        unsigned int start = tri * 3u, n = min(*count, capacity);
        if (start + 2u >= n) return;
        unsigned int i0=remap[start], i1=remap[start+1u], i2=remap[start+2u];
        unsigned int mergedVertexCount = min(*mergedCount, capacity);
        if (i0 >= mergedVertexCount || i1 >= mergedVertexCount || i2 >= mergedVertexCount) return;
        // Match DcLiverVisualRenderer: welding may collapse a valid source triangle.
        if (i0 == i1 || i1 == i2 || i2 == i0) return;
        float3 p0 = positions[start], p1 = positions[start+1u], p2 = positions[start+2u];
        float3 a = make_float3(p1.x-p0.x,p1.y-p0.y,p1.z-p0.z), b = make_float3(p2.x-p0.x,p2.y-p0.y,p2.z-p0.z);
        float3 f = make_float3(a.y*b.z-a.z*b.y,a.z*b.x-a.x*b.z,a.x*b.y-a.y*b.x);
        float area2 = f.x*f.x + f.y*f.y + f.z*f.z;
        if (!isfinite(area2) || area2 < 1e-12f) return;
        unsigned int target = atomicAdd(indexCount, 3u);
        if (target + 2u >= capacity) return;
        indices[target] = i0; indices[target+1u] = i1; indices[target+2u] = i2;
        atomicAdd(&sums[i0].x,f.x); atomicAdd(&sums[i0].y,f.y); atomicAdd(&sums[i0].z,f.z);
        atomicAdd(&sums[i1].x,f.x); atomicAdd(&sums[i1].y,f.y); atomicAdd(&sums[i1].z,f.z);
        atomicAdd(&sums[i2].x,f.x); atomicAdd(&sums[i2].y,f.y); atomicAdd(&sums[i2].z,f.z);
    }

    __global__ void NormalizeNormalsKernel(const float3* sums, const unsigned int* count, float3* normals, unsigned int capacity)
    {
        unsigned int v = blockIdx.x * blockDim.x + threadIdx.x;
        if (v >= min(*count, capacity)) return;
        float3 n=sums[v]; float l=n.x*n.x+n.y*n.y+n.z*n.z;
        normals[v] = !isfinite(l) || l < 1e-12f ? make_float3(0,1,0) : make_float3(n.x*rsqrtf(l),n.y*rsqrtf(l),n.z*rsqrtf(l));
    }

    __global__ void BuildCutBatchKernel(const float3* compactPositions, const float4* compactAux, const unsigned int* count,
                                        float3* positions, float3* normals, float4* aux, unsigned int* indices, unsigned int* indexCount, unsigned int capacity)
    {
        unsigned int tri=blockIdx.x*blockDim.x+threadIdx.x, start=tri*3u, n=min(*count,capacity);
        if (start+2u>=n) return;
        float3 p0=compactPositions[start],p1=compactPositions[start+1u],p2=compactPositions[start+2u];
        float3 a=make_float3(p1.x-p0.x,p1.y-p0.y,p1.z-p0.z), b=make_float3(p2.x-p0.x,p2.y-p0.y,p2.z-p0.z);
        float3 f=make_float3(a.y*b.z-a.z*b.y,a.z*b.x-a.x*b.z,a.x*b.y-a.y*b.x);
        float l=f.x*f.x+f.y*f.y+f.z*f.z; f=l<1e-12f?make_float3(0,1,0):make_float3(f.x*rsqrtf(l),f.y*rsqrtf(l),f.z*rsqrtf(l));
        for(unsigned int i=0;i<3u;++i) { positions[start+i]=compactPositions[start+i]; normals[start+i]=f; aux[start+i]=compactAux[start+i]; indices[start+i]=start+i; }
        if(tri==0u) *indexCount=n;
    }

    void UNITY_INTERFACE_API DirectSurfaceRenderEvent(int)
    {
        if (!RegisterResources()) { if (g_stats.lastError == 0) g_stats.lastError = -4101; return; }
        if (cudaGraphicsMapResources(kResourceCount, g_resources, 0) != cudaSuccess) { g_stats.lastError=-4102; return; }
        void* mapped[kResourceCount]{}; size_t ignored=0; cudaError_t e=cudaSuccess;
        for(int i=0;i<kResourceCount && e==cudaSuccess;++i) e=cudaGraphicsResourceGetMappedPointer(&mapped[i],&ignored,g_resources[i]);
        if(e==cudaSuccess) e=cudaMemset(mapped[IndexCount],0,sizeof(unsigned int));
        if(e==cudaSuccess) e=cudaMemset(mapped[kBatchResourceCount+IndexCount],0,sizeof(unsigned int));
        if(e==cudaSuccess && !EnsureGpuWorkBuffers(g_stats.capacity)) e=cudaErrorMemoryAllocation;
        const unsigned int capacity=static_cast<unsigned int>(g_stats.capacity), threads=128u, vertexBlocks=(capacity+threads-1u)/threads, triangleBlocks=((capacity/3u)+threads-1u)/threads;
        if(e==cudaSuccess) e=cudaMemset(g_frameCounters,0,CounterCount*sizeof(unsigned int));
        if(e==cudaSuccess) e=cudaMemset(g_outerCompactCount,0,sizeof(unsigned int));
        if(e==cudaSuccess) e=cudaMemset(g_cutCompactCount,0,sizeof(unsigned int));
        if(e==cudaSuccess) e=cudaMemset(g_outerMergedCount,0,sizeof(unsigned int));
        if(e==cudaSuccess) e=cudaMemset(g_outerGroupStarts,0,capacity*sizeof(unsigned int));
        if(e==cudaSuccess) e=cudaMemset(g_outerGroupIds,0,capacity*sizeof(unsigned int));
        if(e==cudaSuccess) e=cudaMemset(g_outerOriginalToMerged,0xff,capacity*sizeof(unsigned int));
        if(e==cudaSuccess) e=cudaMemset(g_outerNormalSums,0,capacity*sizeof(float3));
        if(e==cudaSuccess)
        {
            CompactByBatchKernel<<<triangleBlocks,threads>>>(recon_expand_pos(),recon_expand_aux(),recon_tri_counter(),
                g_outerCompactPositions,g_outerCompactAux,g_outerCompactCount,g_cutCompactPositions,g_cutCompactAux,g_cutCompactCount,g_frameCounters,capacity);
            BuildOuterMergeInputsKernel<<<vertexBlocks,threads>>>(g_outerCompactPositions,g_outerCompactAux,g_outerCompactCount,g_outerKeysIn,g_outerKeyFields,g_outerValuesIn,capacity);
            e=cudaGetLastError();
            if(e==cudaSuccess) e=cub::DeviceRadixSort::SortPairs(g_sortTempStorage,g_sortTempStorageBytes,g_outerKeysIn,g_outerKeysOut,g_outerValuesIn,g_outerValuesOut,capacity);
            if(e==cudaSuccess) MarkGroupStartsKernel<<<vertexBlocks,threads>>>(g_outerKeysOut,g_outerValuesOut,g_outerKeyFields,g_outerCompactCount,g_outerGroupStarts,capacity);
            if(e==cudaSuccess) e=cub::DeviceScan::ExclusiveSum(g_scanTempStorage,g_scanTempStorageBytes,g_outerGroupStarts,g_outerGroupIds,capacity);
            if(e==cudaSuccess) FinalizeGroupIdsKernel<<<vertexBlocks,threads>>>(g_outerGroupStarts,g_outerGroupIds,g_outerCompactCount,capacity);
            if(e==cudaSuccess) ScatterOuterVerticesKernel<<<vertexBlocks,threads>>>(g_outerValuesOut,g_outerGroupStarts,g_outerGroupIds,g_outerCompactCount,
                g_outerCompactPositions,g_outerCompactAux,static_cast<float3*>(mapped[Position]),static_cast<float4*>(mapped[Aux]),g_outerNormalSums,
                g_outerOriginalToMerged,g_outerMergedCount,capacity);
            if(e==cudaSuccess) BuildOuterTrianglesKernel<<<triangleBlocks,threads>>>(g_outerCompactPositions,g_outerCompactCount,g_outerOriginalToMerged,g_outerMergedCount,g_outerNormalSums,
                static_cast<unsigned int*>(mapped[Index]),static_cast<unsigned int*>(mapped[IndexCount]),capacity);
            if(e==cudaSuccess) NormalizeNormalsKernel<<<vertexBlocks,threads>>>(g_outerNormalSums,g_outerMergedCount,static_cast<float3*>(mapped[Normal]),capacity);
            if(e==cudaSuccess) BuildCutBatchKernel<<<triangleBlocks,threads>>>(g_cutCompactPositions,g_cutCompactAux,g_cutCompactCount,
                static_cast<float3*>(mapped[kBatchResourceCount+Position]),static_cast<float3*>(mapped[kBatchResourceCount+Normal]),static_cast<float4*>(mapped[kBatchResourceCount+Aux]),
                static_cast<unsigned int*>(mapped[kBatchResourceCount+Index]),static_cast<unsigned int*>(mapped[kBatchResourceCount+IndexCount]),capacity);
        }
        if(e==cudaSuccess) e=cudaGetLastError();
        if(e==cudaSuccess && g_topologyDiagnosticRequested)
        {
            unsigned int counters[CounterCount]{}; unsigned int outerCount=0, cutCount=0, mergedCount=0, outerIndexCount=0;
            e=cudaMemcpy(counters,g_frameCounters,CounterCount*sizeof(unsigned int),cudaMemcpyDeviceToHost);
            if(e==cudaSuccess) e=cudaMemcpy(&outerCount,g_outerCompactCount,sizeof(unsigned int),cudaMemcpyDeviceToHost);
            if(e==cudaSuccess) e=cudaMemcpy(&cutCount,g_cutCompactCount,sizeof(unsigned int),cudaMemcpyDeviceToHost);
            if(e==cudaSuccess) e=cudaMemcpy(&mergedCount,g_outerMergedCount,sizeof(unsigned int),cudaMemcpyDeviceToHost);
            if(e==cudaSuccess) e=cudaMemcpy(&outerIndexCount,mapped[IndexCount],sizeof(unsigned int),cudaMemcpyDeviceToHost);
            if(e==cudaSuccess)
            {
                g_stats.rawVertexCount=counters[CounterRawVertices]; g_stats.sourceVertexCountClamped=counters[CounterClampedVertices];
                g_stats.rawTriangleCount=counters[CounterRawTriangles]; g_stats.validTriangleCount=counters[CounterValidTriangles];
                g_stats.rejectedInvalidTriangleCount=counters[CounterRejectedInvalid]; g_stats.rejectedDegenerateTriangleCount=counters[CounterRejectedDegenerate];
                g_stats.outerTriangleCount=counters[CounterOuterTriangles]; g_stats.cutWallTriangleCount=counters[CounterCutWallTriangles];
                g_stats.rawNoWallTriangleCount=counters[CounterRawNoWallTriangles];
                g_stats.rawAnyWallTriangleCount=counters[CounterRawAnyWallTriangles];
                g_stats.rawMixedWallTriangleCount=counters[CounterRawMixedWallTriangles];
                g_stats.writtenVertexCount=counters[CounterWrittenVertices]; g_stats.capacityOverflow=counters[CounterCapacityOverflow];
                g_stats.indexedVertexCount=mergedCount; g_stats.indexedIndexCount=outerIndexCount; g_stats.outerIndexedVertexCount=mergedCount; g_stats.outerIndexedIndexCount=outerIndexCount;
                g_stats.cutWallIndexedVertexCount=cutCount; g_stats.cutWallIndexedIndexCount=cutCount;
                g_stats.topologyDiagnosticReady=1; g_stats.topologyDiagnosticOuterVertexSamples=outerCount; g_stats.topologyDiagnosticUniqueOuterVertexEstimate=mergedCount;
                g_stats.topologyDiagnosticError=0; ++g_stats.topologyDiagnosticSequence;
            }
            else { g_stats.topologyDiagnosticReady=0; g_stats.topologyDiagnosticError=-static_cast<int>(e); }
            g_topologyDiagnosticRequested=false;
        }
        cudaError_t unmap=cudaGraphicsUnmapResources(kResourceCount,g_resources,0); if(e==cudaSuccess) e=unmap;
        if(e!=cudaSuccess) { g_stats.lastError=-static_cast<int>(e); return; }
        g_stats.mergeEnabled=1;
        g_stats.gpuCopyBytes+=static_cast<uint64_t>(g_stats.capacity)*(sizeof(float3)*4+sizeof(float4)*2+sizeof(unsigned int)*2)+sizeof(unsigned int)*2;
        ++g_stats.dispatchCount; g_stats.lastError=0;
    }
}

int direct_surface_set_buffers(void* outerPositionBuffer, void* outerNormalBuffer, void* outerAuxBuffer,
                               void* outerIndexBuffer, void* outerIndexCountBuffer,
                               void* cutPositionBuffer, void* cutNormalBuffer, void* cutAuxBuffer,
                               void* cutIndexBuffer, void* cutIndexCountBuffer, int capacity)
{
    ReleaseResources();
    void* inputs[kResourceCount] = { outerPositionBuffer, outerNormalBuffer, outerAuxBuffer, outerIndexBuffer, outerIndexCountBuffer,
                                     cutPositionBuffer, cutNormalBuffer, cutAuxBuffer, cutIndexBuffer, cutIndexCountBuffer };
    for(int i=0;i<kResourceCount;++i) g_buffers[i]=static_cast<ID3D11Buffer*>(inputs[i]);
    g_stats={}; g_stats.supported=1; g_stats.capacity=capacity; g_stats.cudaDeviceOrdinal=-1; g_stats.configured=capacity>0;
    for(int i=0;i<kResourceCount;++i) g_stats.configured=g_stats.configured && g_buffers[i]!=nullptr;
    return g_stats.configured?0:-4100;
}

void direct_surface_release()
{
    ReleaseResources(); ReleaseGpuWorkBuffers();
    for(int i=0;i<kResourceCount;++i) g_buffers[i]=nullptr;
    g_stats={};
}

void* direct_surface_get_render_event() { return reinterpret_cast<void*>(DirectSurfaceRenderEvent); }
int direct_surface_get_stats(DirectSurfaceStats* outStats) { if(!outStats) return -1; *outStats=g_stats; return 0; }
int direct_surface_request_topology_diagnostic()
{
    if(!g_stats.configured || g_stats.capacity<=0) return -4100;
    g_topologyDiagnosticRequested=true; g_stats.topologyDiagnosticReady=0; g_stats.topologyDiagnosticError=0; return 0;
}
