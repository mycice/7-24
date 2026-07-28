#include <d3d11.h>
#include <cuda_d3d11_interop.h>

#include "direct_surface_interop.h"
#include "dc_recon.h"
#include <IUnityInterface.h>
#include <IUnityGraphics.h>
#include <IUnityGraphicsD3D11.h>

namespace
{
    cudaGraphicsResource* g_position = nullptr;
    cudaGraphicsResource* g_normal = nullptr;
    cudaGraphicsResource* g_aux = nullptr;
    cudaGraphicsResource* g_vertexCount = nullptr;
    ID3D11Buffer* g_positionBuffer = nullptr;
    ID3D11Buffer* g_normalBuffer = nullptr;
    ID3D11Buffer* g_auxBuffer = nullptr;
    ID3D11Buffer* g_vertexCountBuffer = nullptr;
    DirectSurfaceStats g_stats{};

    enum RegistrationStage
    {
        RegisterNone = 0,
        RegisterResolveD3DDevice = 1,
        RegisterSelectCudaDevice = 2,
        RegisterPosition = 3,
        RegisterNormal = 4,
        RegisterAux = 5,
        RegisterVertexCount = 6
    };

    bool RecordCudaFailure(RegistrationStage stage, cudaError_t error)
    {
        g_stats.registrationStage = static_cast<int>(stage);
        g_stats.cudaErrorCode = static_cast<int>(error);
        g_stats.lastError = -4101;
        return false;
    }

    void ReleaseResources()
    {
        if (g_position) cudaGraphicsUnregisterResource(g_position);
        if (g_normal) cudaGraphicsUnregisterResource(g_normal);
        if (g_aux) cudaGraphicsUnregisterResource(g_aux);
        if (g_vertexCount) cudaGraphicsUnregisterResource(g_vertexCount);
        g_position = g_normal = g_aux = g_vertexCount = nullptr;
        g_stats.registered = 0;
    }

    bool RegisterResources()
    {
        if (g_stats.registered) return true;
        if (!g_positionBuffer || !g_normalBuffer || !g_auxBuffer || !g_vertexCountBuffer || g_stats.capacity <= 0)
        {
            g_stats.lastError = -4100;
            return false;
        }

        ID3D11Device* d3dDevice = nullptr;
        g_positionBuffer->GetDevice(&d3dDevice);
        if (!d3dDevice)
        {
            g_stats.registrationStage = RegisterResolveD3DDevice;
            g_stats.cudaErrorCode = 0;
            g_stats.lastError = -4101;
            return false;
        }

        IDXGIDevice* dxgiDevice = nullptr;
        HRESULT adapterResult = d3dDevice->QueryInterface(__uuidof(IDXGIDevice), reinterpret_cast<void**>(&dxgiDevice));
        if (FAILED(adapterResult) || !dxgiDevice)
        {
            d3dDevice->Release();
            g_stats.registrationStage = RegisterResolveD3DDevice;
            g_stats.cudaErrorCode = static_cast<int>(adapterResult);
            g_stats.lastError = -4101;
            return false;
        }

        IDXGIAdapter* adapter = nullptr;
        adapterResult = dxgiDevice->GetAdapter(&adapter);
        dxgiDevice->Release();
        if (FAILED(adapterResult) || !adapter)
        {
            d3dDevice->Release();
            g_stats.registrationStage = RegisterResolveD3DDevice;
            g_stats.cudaErrorCode = static_cast<int>(adapterResult);
            g_stats.lastError = -4101;
            return false;
        }

        int cudaDevice = -1;
        cudaError_t error = cudaD3D11GetDevice(&cudaDevice, adapter);
        adapter->Release();
        d3dDevice->Release();
        if (error != cudaSuccess)
            return RecordCudaFailure(RegisterResolveD3DDevice, error);

        error = cudaSetDevice(cudaDevice);
        if (error != cudaSuccess)
            return RecordCudaFailure(RegisterSelectCudaDevice, error);
        g_stats.cudaDeviceOrdinal = cudaDevice;

        D3D11_BUFFER_DESC positionDesc{};
        g_positionBuffer->GetDesc(&positionDesc);
        g_stats.positionByteWidth = static_cast<int>(positionDesc.ByteWidth);
        g_stats.positionUsage = static_cast<int>(positionDesc.Usage);
        g_stats.positionBindFlags = static_cast<int>(positionDesc.BindFlags);
        g_stats.positionMiscFlags = static_cast<int>(positionDesc.MiscFlags);
        g_stats.positionStructureByteStride = static_cast<int>(positionDesc.StructureByteStride);

        // Unity owns these resources and CUDA writes them only inside the render-thread callback.
        // The neutral registration flag is accepted by more D3D11 buffer descriptions than
        // WriteDiscard while retaining the same map/copy/unmap flow.
        error = cudaGraphicsD3D11RegisterResource(&g_position, g_positionBuffer, cudaGraphicsRegisterFlagsNone);
        if (error != cudaSuccess) { ReleaseResources(); return RecordCudaFailure(RegisterPosition, error); }
        error = cudaGraphicsD3D11RegisterResource(&g_normal, g_normalBuffer, cudaGraphicsRegisterFlagsNone);
        if (error != cudaSuccess) { ReleaseResources(); return RecordCudaFailure(RegisterNormal, error); }
        error = cudaGraphicsD3D11RegisterResource(&g_aux, g_auxBuffer, cudaGraphicsRegisterFlagsNone);
        if (error != cudaSuccess) { ReleaseResources(); return RecordCudaFailure(RegisterAux, error); }
        error = cudaGraphicsD3D11RegisterResource(&g_vertexCount, g_vertexCountBuffer, cudaGraphicsRegisterFlagsNone);
        if (error != cudaSuccess) { ReleaseResources(); return RecordCudaFailure(RegisterVertexCount, error); }
        g_stats.registered = 1;
        g_stats.registrationStage = RegisterNone;
        g_stats.cudaErrorCode = 0;
        return true;
    }

    __global__ void ClampVertexCountKernel(unsigned int* vertexCount, unsigned int capacity)
    {
        if (threadIdx.x != 0 || blockIdx.x != 0) return;
        unsigned int count = *vertexCount;
        if (count > capacity) count = capacity;
        *vertexCount = count;
    }

    __device__ bool IsFiniteFloat3(float3 value)
    {
        return isfinite(value.x) && isfinite(value.y) && isfinite(value.z);
    }

    // The CPU visual renderer rejects degenerate triangle-soup entries before building its
    // Mesh. Replicate that essential validity check here so the zero-readback renderer does
    // not turn collapsed Dual Contouring triangles into the point-like debris seen on screen.
    __global__ void CompactValidSurfaceKernel(const float3* sourcePositions,
                                              const float3* sourceNormals,
                                              const float4* sourceAux,
                                              const unsigned int* sourceVertexCount,
                                              float3* targetPositions,
                                              float3* targetNormals,
                                              float4* targetAux,
                                              unsigned int* targetVertexCount,
                                              unsigned int capacity)
    {
        unsigned int triangleIndex = blockIdx.x * blockDim.x + threadIdx.x;
        unsigned int sourceCount = min(*sourceVertexCount, capacity);
        unsigned int sourceVertex = triangleIndex * 3u;
        if (sourceVertex + 2u >= sourceCount) return;

        float3 p0 = sourcePositions[sourceVertex + 0u];
        float3 p1 = sourcePositions[sourceVertex + 1u];
        float3 p2 = sourcePositions[sourceVertex + 2u];
        float3 edgeA = make_float3(p1.x - p0.x, p1.y - p0.y, p1.z - p0.z);
        float3 edgeB = make_float3(p2.x - p0.x, p2.y - p0.y, p2.z - p0.z);
        float3 face = make_float3(
            edgeA.y * edgeB.z - edgeA.z * edgeB.y,
            edgeA.z * edgeB.x - edgeA.x * edgeB.z,
            edgeA.x * edgeB.y - edgeA.y * edgeB.x);
        float areaSquared = face.x * face.x + face.y * face.y + face.z * face.z;
        if (!IsFiniteFloat3(p0) || !IsFiniteFloat3(p1) || !IsFiniteFloat3(p2) ||
            !isfinite(areaSquared) || areaSquared < 1e-12f)
            return;

        unsigned int targetVertex = atomicAdd(targetVertexCount, 3u);
        if (targetVertex + 2u >= capacity) return;
        targetPositions[targetVertex + 0u] = p0;
        targetPositions[targetVertex + 1u] = p1;
        targetPositions[targetVertex + 2u] = p2;
        targetNormals[targetVertex + 0u] = sourceNormals[sourceVertex + 0u];
        targetNormals[targetVertex + 1u] = sourceNormals[sourceVertex + 1u];
        targetNormals[targetVertex + 2u] = sourceNormals[sourceVertex + 2u];
        targetAux[targetVertex + 0u] = sourceAux[sourceVertex + 0u];
        targetAux[targetVertex + 1u] = sourceAux[sourceVertex + 1u];
        targetAux[targetVertex + 2u] = sourceAux[sourceVertex + 2u];
    }

    void UNITY_INTERFACE_API DirectSurfaceRenderEvent(int)
    {
        if (!RegisterResources())
        {
            if (g_stats.lastError == 0) g_stats.lastError = -4101;
            return;
        }
        cudaGraphicsResource* resources[] = { g_position, g_normal, g_aux, g_vertexCount };
        if (cudaGraphicsMapResources(4, resources, 0) != cudaSuccess) { g_stats.lastError = -4102; return; }
        void *pos=nullptr, *nrm=nullptr, *aux=nullptr, *vertexCount=nullptr;
        size_t ignored=0;
        cudaError_t e = cudaGraphicsResourceGetMappedPointer(&pos, &ignored, g_position);
        if (e == cudaSuccess) e = cudaGraphicsResourceGetMappedPointer(&nrm, &ignored, g_normal);
        if (e == cudaSuccess) e = cudaGraphicsResourceGetMappedPointer(&aux, &ignored, g_aux);
        if (e == cudaSuccess) e = cudaGraphicsResourceGetMappedPointer(&vertexCount, &ignored, g_vertexCount);
        const size_t count = static_cast<size_t>(g_stats.capacity);
        if (e == cudaSuccess) e = cudaMemset(vertexCount, 0, sizeof(unsigned int));
        if (e == cudaSuccess)
        {
            const unsigned int triangleCapacity = static_cast<unsigned int>(g_stats.capacity / 3);
            const unsigned int threads = 128u;
            const unsigned int blocks = (triangleCapacity + threads - 1u) / threads;
            CompactValidSurfaceKernel<<<blocks, threads>>>(
                recon_expand_pos(), recon_expand_nrm(), recon_expand_aux(), recon_tri_counter(),
                static_cast<float3*>(pos), static_cast<float3*>(nrm), static_cast<float4*>(aux),
                static_cast<unsigned int*>(vertexCount), static_cast<unsigned int>(g_stats.capacity));
            ClampVertexCountKernel<<<1, 1>>>(static_cast<unsigned int*>(vertexCount), static_cast<unsigned int>(g_stats.capacity));
        }
        if (e == cudaSuccess) e = cudaGetLastError();
        // The Unity draw follows this render callback immediately.  A kernel launch is
        // asynchronous, so explicitly finish the writes before giving the D3D11 buffers
        // back to Unity; otherwise the draw can observe a partially written triangle soup.
        if (e == cudaSuccess) e = cudaDeviceSynchronize();
        cudaError_t unmapError = cudaGraphicsUnmapResources(4, resources, 0);
        if (e == cudaSuccess) e = unmapError;
        if (e != cudaSuccess) { g_stats.lastError = -static_cast<int>(e); return; }
        g_stats.gpuCopyBytes += count * (sizeof(float3) * 2 + sizeof(float4)) + sizeof(unsigned int);
        ++g_stats.dispatchCount;
        g_stats.lastError = 0;
    }
}

int direct_surface_set_buffers(void* positionBuffer, void* normalBuffer, void* auxBuffer,
                               void* vertexCountBuffer, int capacity)
{
    ReleaseResources();
    g_positionBuffer = static_cast<ID3D11Buffer*>(positionBuffer);
    g_normalBuffer = static_cast<ID3D11Buffer*>(normalBuffer);
    g_auxBuffer = static_cast<ID3D11Buffer*>(auxBuffer);
    g_vertexCountBuffer = static_cast<ID3D11Buffer*>(vertexCountBuffer);
    g_stats = {};
    g_stats.supported = 1;
    g_stats.configured = g_positionBuffer && g_normalBuffer && g_auxBuffer && g_vertexCountBuffer && capacity > 0;
        g_stats.capacity = capacity;
        g_stats.cudaDeviceOrdinal = -1;
        return g_stats.configured ? 0 : -4100;
}

void direct_surface_release()
{
    ReleaseResources();
    g_positionBuffer = g_normalBuffer = g_auxBuffer = g_vertexCountBuffer = nullptr;
    g_stats = {};
}

void* direct_surface_get_render_event()
{
    return reinterpret_cast<void*>(DirectSurfaceRenderEvent);
}

int direct_surface_get_stats(DirectSurfaceStats* outStats)
{
    if (!outStats) return -1;
    *outStats = g_stats;
    return 0;
}
