// organ_context.cu - phase-1 per-organ CUDA allocation and static upload.
#include "organ_context.h"

#include <cuda_runtime.h>
#include <unordered_map>
#include <memory>
#include <mutex>
#include <chrono>
#include <cstring>

namespace
{
    struct CudaOrganContext
    {
        OrganContextStats stats{};
        float* restPositions = nullptr;
        int* tetIds = nullptr;
        float* inverseMass = nullptr;
        float* restVolumes = nullptr;
        int* tetActive = nullptr;
        int* surfaceTriangles = nullptr;
        int* edgeIds = nullptr;
        float* edgeRestLengths = nullptr;
        float* material = nullptr;
        cudaEvent_t uploadStart = nullptr;
        cudaEvent_t uploadEnd = nullptr;
        cudaEvent_t kernelStart = nullptr;
        cudaEvent_t kernelEnd = nullptr;

        ~CudaOrganContext()
        {
            cudaFree(restPositions); cudaFree(tetIds); cudaFree(inverseMass);
            cudaFree(restVolumes); cudaFree(tetActive); cudaFree(surfaceTriangles);
            cudaFree(edgeIds); cudaFree(edgeRestLengths);
            cudaFree(material);
            cudaEventDestroy(uploadStart); cudaEventDestroy(uploadEnd);
            cudaEventDestroy(kernelStart); cudaEventDestroy(kernelEnd);
        }
    };

    std::mutex g_contextMutex;
    std::unordered_map<uint32_t, std::unique_ptr<CudaOrganContext>> g_contexts;
    uint32_t g_nextHandle = 1;

    uint64_t HashBytes(uint64_t hash, const void* bytes, size_t count)
    {
        const unsigned char* data = static_cast<const unsigned char*>(bytes);
        for (size_t i = 0; i < count; ++i)
        {
            hash ^= data[i];
            hash *= 1099511628211ull;
        }
        return hash;
    }

    template <typename T>
    bool Upload(T*& destination, const T* source, size_t count, CudaOrganContext& context)
    {
        if (count == 0) return true;
        const size_t bytes = sizeof(T) * count;
        if (cudaMalloc(&destination, bytes) != cudaSuccess) return false;
        if (cudaMemcpy(destination, source, bytes, cudaMemcpyHostToDevice) != cudaSuccess) return false;
        context.stats.uploadBytes += bytes;
        context.stats.uploadOperations += 1;
        context.stats.deviceBytes += bytes;
        return true;
    }

    __global__ void ValidateRestStateKernel(const float* restPositions, int count, unsigned long long* hashSink)
    {
        const int id = blockIdx.x * blockDim.x + threadIdx.x;
        if (id >= count) return;
        // A cheap read-only validation pass establishes a CUDA timing baseline without modifying state.
        const unsigned int bits = __float_as_uint(restPositions[id * 3]);
        atomicAdd(hashSink, static_cast<unsigned long long>(bits));
    }

    CudaOrganContext* FindContext(uint32_t handle)
    {
        const auto it = g_contexts.find(handle);
        return it == g_contexts.end() ? nullptr : it->second.get();
    }
}

int organ_context_create(uint32_t* outHandle)
{
    if (outHandle == nullptr) return ORGAN_CONTEXT_INVALID_ARGUMENT;
    std::lock_guard<std::mutex> lock(g_contextMutex);
    uint32_t handle = g_nextHandle++;
    if (handle == 0) handle = g_nextHandle++;
    auto context = std::make_unique<CudaOrganContext>();
    context->stats.handle = handle;
    context->stats.lastError = ORGAN_CONTEXT_OK;
    if (cudaEventCreate(&context->uploadStart) != cudaSuccess ||
        cudaEventCreate(&context->uploadEnd) != cudaSuccess ||
        cudaEventCreate(&context->kernelStart) != cudaSuccess ||
        cudaEventCreate(&context->kernelEnd) != cudaSuccess)
        return ORGAN_CONTEXT_CUDA_FAILURE;
    g_contexts.emplace(handle, std::move(context));
    *outHandle = handle;
    return ORGAN_CONTEXT_OK;
}

int organ_context_destroy(uint32_t handle)
{
    std::lock_guard<std::mutex> lock(g_contextMutex);
    const auto it = g_contexts.find(handle);
    if (it == g_contexts.end()) return ORGAN_CONTEXT_INVALID_HANDLE;
    g_contexts.erase(it);
    return ORGAN_CONTEXT_OK;
}

int organ_context_initialize(uint32_t handle, const OrganContextInitDesc* desc,
                             const float* restPositions3, const int* tetIds4,
                             const float* inverseMass, const float* restVolumes,
                             const int* tetActive, const int* surfaceTriangleIds3,
                             const int* edgeConstraintIds2, const float* edgeRestLengths)
{
    if (desc == nullptr || desc->particleCount <= 0 || desc->tetCount <= 0 ||
        restPositions3 == nullptr || tetIds4 == nullptr || inverseMass == nullptr ||
        restVolumes == nullptr || tetActive == nullptr)
        return ORGAN_CONTEXT_INVALID_ARGUMENT;

    std::lock_guard<std::mutex> lock(g_contextMutex);
    CudaOrganContext* context = FindContext(handle);
    if (context == nullptr) return ORGAN_CONTEXT_INVALID_HANDLE;
    if (context->stats.initialized) return ORGAN_CONTEXT_ALREADY_INITIALIZED;
    if ((desc->surfaceTriangleCount > 0 && surfaceTriangleIds3 == nullptr) ||
        (desc->edgeConstraintCount > 0 && (edgeConstraintIds2 == nullptr || edgeRestLengths == nullptr)))
        return ORGAN_CONTEXT_INVALID_ARGUMENT;

    context->stats.particleCount = desc->particleCount;
    context->stats.tetCount = desc->tetCount;
    context->stats.surfaceTriangleCount = desc->surfaceTriangleCount;
    context->stats.edgeConstraintCount = desc->edgeConstraintCount;
    cudaEventRecord(context->uploadStart);
    const float material[4] = { desc->density, desc->youngsModulus, desc->poissonsRatio, desc->damping };

    const bool uploaded =
        Upload(context->restPositions, restPositions3, static_cast<size_t>(desc->particleCount) * 3, *context) &&
        Upload(context->tetIds, tetIds4, static_cast<size_t>(desc->tetCount) * 4, *context) &&
        Upload(context->inverseMass, inverseMass, desc->particleCount, *context) &&
        Upload(context->restVolumes, restVolumes, desc->tetCount, *context) &&
        Upload(context->tetActive, tetActive, desc->tetCount, *context) &&
        Upload(context->surfaceTriangles, surfaceTriangleIds3, static_cast<size_t>(desc->surfaceTriangleCount) * 3, *context) &&
        Upload(context->edgeIds, edgeConstraintIds2, static_cast<size_t>(desc->edgeConstraintCount) * 2, *context) &&
        Upload(context->edgeRestLengths, edgeRestLengths, desc->edgeConstraintCount, *context) &&
        Upload(context->material, material, 4, *context);
    cudaEventRecord(context->uploadEnd);
    cudaEventSynchronize(context->uploadEnd);
    cudaEventElapsedTime(&context->stats.uploadMilliseconds, context->uploadStart, context->uploadEnd);
    if (!uploaded)
    {
        context->stats.lastError = ORGAN_CONTEXT_CUDA_FAILURE;
        return ORGAN_CONTEXT_CUDA_FAILURE;
    }

    unsigned long long* validationSink = nullptr;
    if (cudaMalloc(&validationSink, sizeof(unsigned long long)) != cudaSuccess ||
        cudaMemset(validationSink, 0, sizeof(unsigned long long)) != cudaSuccess)
    {
        cudaFree(validationSink);
        context->stats.lastError = ORGAN_CONTEXT_CUDA_FAILURE;
        return ORGAN_CONTEXT_CUDA_FAILURE;
    }
    cudaEventRecord(context->kernelStart);
    ValidateRestStateKernel<<<(desc->particleCount + 127) / 128, 128>>>(context->restPositions, desc->particleCount, validationSink);
    cudaEventRecord(context->kernelEnd);
    cudaEventSynchronize(context->kernelEnd);
    cudaEventElapsedTime(&context->stats.validationKernelMilliseconds, context->kernelStart, context->kernelEnd);
    cudaFree(validationSink);
    if (cudaGetLastError() != cudaSuccess)
    {
        context->stats.lastError = ORGAN_CONTEXT_CUDA_FAILURE;
        return ORGAN_CONTEXT_CUDA_FAILURE;
    }

    uint64_t restHash = 1469598103934665603ull;
    restHash = HashBytes(restHash, restPositions3, sizeof(float) * static_cast<size_t>(desc->particleCount) * 3);
    uint64_t topologyHash = 1469598103934665603ull;
    topologyHash = HashBytes(topologyHash, tetIds4, sizeof(int) * static_cast<size_t>(desc->tetCount) * 4);
    topologyHash = HashBytes(topologyHash, surfaceTriangleIds3, sizeof(int) * static_cast<size_t>(desc->surfaceTriangleCount) * 3);
    topologyHash = HashBytes(topologyHash, edgeConstraintIds2, sizeof(int) * static_cast<size_t>(desc->edgeConstraintCount) * 2);
    context->stats.restPositionHash = restHash;
    context->stats.topologyHash = topologyHash;
    context->stats.hostFingerprint = HashBytes(topologyHash, &context->stats.deviceBytes, sizeof(context->stats.deviceBytes));
    context->stats.initialized = 1;
    context->stats.lastError = ORGAN_CONTEXT_OK;
    return ORGAN_CONTEXT_OK;
}

int organ_context_get_stats(uint32_t handle, OrganContextStats* outStats)
{
    if (outStats == nullptr) return ORGAN_CONTEXT_INVALID_ARGUMENT;
    std::lock_guard<std::mutex> lock(g_contextMutex);
    CudaOrganContext* context = FindContext(handle);
    if (context == nullptr) return ORGAN_CONTEXT_INVALID_HANDLE;
    *outStats = context->stats;
    return ORGAN_CONTEXT_OK;
}
