// organ_context.cu - per-organ GPU-resident data and phase-2 tetrahedral XPBD.
#include "organ_context.h"
#include "dc_recon.h"
#include "cut.h"

#include <cuda_runtime.h>
#include <unordered_map>
#include <memory>
#include <mutex>
#include <cstring>
#include <cmath>
#include <cfloat>
#include <vector>
#include <algorithm>
#include <chrono>
#include <unordered_set>

namespace
{
    constexpr int kThreads = 128;

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

        float3* positions = nullptr;
        float3* previousPositions = nullptr;
        float3* velocities = nullptr;
        float3* invRestRows = nullptr;
        float* alphaDeviatoric = nullptr;
        float* alphaHydrostatic = nullptr;
        int* edgeColorOffsets = nullptr;
        int* edgeColorCounts = nullptr;
        int* edgeColorFlat = nullptr;
        int* tetColorOffsets = nullptr;
        int* tetColorCounts = nullptr;
        int* tetColorFlat = nullptr;
        int* surfaceColorOffsets = nullptr;
        int* surfaceColorCounts = nullptr;
        int* surfaceColorFlat = nullptr;
        int edgeColorCount = 0;
        int tetColorCount = 0;
        int surfaceColorCount = 0;
        std::vector<int> hostEdgeColorOffsets;
        std::vector<int> hostEdgeColorCounts;
        std::vector<int> hostTetColorOffsets;
        std::vector<int> hostTetColorCounts;
        std::vector<int> hostSurfaceColorOffsets;
        std::vector<int> hostSurfaceColorCounts;
        int xpbdInitialized = 0;
        OrganContextToolCapsule* toolCapsules = nullptr;
        OrganContextToolContactParams toolParams{};
        OrganContextToolContactParams graspCaptureParams{};
        int toolCapsuleCount = 0;
        int toolConfigured = 0;
        int toolGraspActive = 0;
        float toolGraspBlend = 0.0f;
        unsigned char* graspMask = nullptr;
        float3* graspFrameCoordinates = nullptr;
        float* graspWeights = nullptr;
        unsigned int* toolCounters = nullptr;
        OrganContextToolContactStats toolStats{};
        int* hostTetByCorner = nullptr;
        float4* barycentricWeights = nullptr;
        float3* alignedRestCorners = nullptr;
        unsigned char* gridActiveMask = nullptr;
        float3* currentCentroid = nullptr;
        OrganContextTetToGridStats tetToGridStats{};
        int4* internalFaces = nullptr;
        int internalFaceCount = 0;
        CutEventSummary* cutEventSummary = nullptr;
        unsigned int* cutToTetCounters = nullptr;
        unsigned int* candidateTetSequence = nullptr;
        unsigned int* candidateEdgeSequence = nullptr;
        unsigned int* candidateSharedFaceSequence = nullptr;
        unsigned int* candidateSurfaceFaceSequence = nullptr;
        OrganContextCutToTetStats cutToTetStats{};
        OrganContextTetFractureStats fractureStats{};
        unsigned int reportedClassificationCount = 0;
        float3 tetRestCentroid = make_float3(0, 0, 0);
        cudaEvent_t tetToGridStart = nullptr;
        cudaEvent_t tetToGridEnd = nullptr;
        cudaEvent_t cutToTetStart = nullptr;
        cudaEvent_t cutToTetEnd = nullptr;
        cudaEvent_t toolStart = nullptr;
        cudaEvent_t toolEnd = nullptr;
        cudaEvent_t uploadStart = nullptr;
        cudaEvent_t uploadEnd = nullptr;
        cudaEvent_t kernelStart = nullptr;
        cudaEvent_t kernelEnd = nullptr;

        ~CudaOrganContext()
        {
            cudaFree(restPositions); cudaFree(tetIds); cudaFree(inverseMass);
            cudaFree(restVolumes); cudaFree(tetActive); cudaFree(surfaceTriangles);
            cudaFree(edgeIds); cudaFree(edgeRestLengths); cudaFree(material);
            cudaFree(positions); cudaFree(previousPositions); cudaFree(velocities);
            cudaFree(invRestRows); cudaFree(alphaDeviatoric); cudaFree(alphaHydrostatic);
            cudaFree(edgeColorOffsets); cudaFree(edgeColorCounts); cudaFree(edgeColorFlat);
            cudaFree(tetColorOffsets); cudaFree(tetColorCounts); cudaFree(tetColorFlat);
            cudaFree(surfaceColorOffsets); cudaFree(surfaceColorCounts); cudaFree(surfaceColorFlat);
            cudaFree(toolCapsules); cudaFree(graspMask); cudaFree(graspFrameCoordinates);
            cudaFree(graspWeights); cudaFree(toolCounters);
            cudaFree(hostTetByCorner); cudaFree(barycentricWeights); cudaFree(alignedRestCorners);
            cudaFree(gridActiveMask); cudaFree(currentCentroid);
            cudaFree(internalFaces); cudaFree(cutEventSummary); cudaFree(cutToTetCounters);
            cudaFree(candidateTetSequence); cudaFree(candidateEdgeSequence);
            cudaFree(candidateSharedFaceSequence); cudaFree(candidateSurfaceFaceSequence);
            cudaEventDestroy(uploadStart); cudaEventDestroy(uploadEnd);
            cudaEventDestroy(kernelStart); cudaEventDestroy(kernelEnd);
            cudaEventDestroy(toolStart); cudaEventDestroy(toolEnd);
            cudaEventDestroy(tetToGridStart); cudaEventDestroy(tetToGridEnd);
            cudaEventDestroy(cutToTetStart); cudaEventDestroy(cutToTetEnd);
        }
    };

    std::mutex g_contextMutex;
    std::unordered_map<uint32_t, std::unique_ptr<CudaOrganContext>> g_contexts;
    uint32_t g_nextHandle = 1;

    __host__ __device__ inline float3 Add(float3 a, float3 b) { return make_float3(a.x + b.x, a.y + b.y, a.z + b.z); }
    __host__ __device__ inline float3 Sub(float3 a, float3 b) { return make_float3(a.x - b.x, a.y - b.y, a.z - b.z); }
    __host__ __device__ inline float3 Mul(float3 a, float s) { return make_float3(a.x * s, a.y * s, a.z * s); }
    __host__ __device__ inline float Dot(float3 a, float3 b) { return a.x*b.x + a.y*b.y + a.z*b.z; }
    __host__ __device__ inline float3 Cross(float3 a, float3 b)
    {
        return make_float3(a.y*b.z-a.z*b.y, a.z*b.x-a.x*b.z, a.x*b.y-a.y*b.x);
    }
    __host__ __device__ inline bool Finite3(float3 v)
    {
        return isfinite(v.x) && isfinite(v.y) && isfinite(v.z);
    }

    uint64_t HashBytes(uint64_t hash, const void* bytes, size_t count)
    {
        const unsigned char* data = static_cast<const unsigned char*>(bytes);
        for (size_t i = 0; i < count; ++i) { hash ^= data[i]; hash *= 1099511628211ull; }
        return hash;
    }

    template <typename T>
    bool Upload(T*& destination, const T* source, size_t count, CudaOrganContext& context, bool dynamic = false)
    {
        if (count == 0) return true;
        const size_t bytes = sizeof(T) * count;
        if (cudaMalloc(&destination, bytes) != cudaSuccess) return false;
        if (cudaMemcpy(destination, source, bytes, cudaMemcpyHostToDevice) != cudaSuccess) return false;
        context.stats.uploadBytes += bytes;
        context.stats.uploadOperations += 1;
        context.stats.deviceBytes += bytes;
        if (dynamic) context.stats.dynamicDeviceBytes += bytes;
        return true;
    }

    template <typename T>
    bool Allocate(T*& destination, size_t count, CudaOrganContext& context)
    {
        if (count == 0) return true;
        const size_t bytes = sizeof(T) * count;
        if (cudaMalloc(&destination, bytes) != cudaSuccess) return false;
        context.stats.deviceBytes += bytes;
        context.stats.dynamicDeviceBytes += bytes;
        return true;
    }

    __global__ void ValidateRestStateKernel(const float* restPositions, int count, unsigned long long* hashSink)
    {
        int id = blockIdx.x * blockDim.x + threadIdx.x;
        if (id < count) atomicAdd(hashSink, static_cast<unsigned long long>(__float_as_uint(restPositions[id * 3])));
    }

    __global__ void SumCentroidKernel(const float3* positions, int count, float3* sum)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= count) return;
        float3 p = positions[i];
        atomicAdd(&sum->x, p.x / count);
        atomicAdd(&sum->y, p.y / count);
        atomicAdd(&sum->z, p.z / count);
    }

    __global__ void TetToGridKernel(const float3* positions, const float3* restPositions, const int* tetIds,
                                    const int* hostTet, const float4* weights,
                                    const float3* alignedRest, const unsigned char* active,
                                    int cornerCount, float3 restCentroid, const float3* currentCentroid,
                                    int applyLocalDeformation, float3* cornerPos)
    {
        int c = blockIdx.x * blockDim.x + threadIdx.x;
        if (c >= cornerCount) return;
        const float3 rest = alignedRest[c];
        const int t = hostTet[c];
        if (active[c] && t >= 0 && applyLocalDeformation)
        {
            const int base = t * 4;
            const float4 w = weights[c];
            const float3 p0 = positions[tetIds[base]];
            const float3 p1 = positions[tetIds[base + 1]];
            const float3 p2 = positions[tetIds[base + 2]];
            const float3 p3 = positions[tetIds[base + 3]];
            // The C# mapping was created from the same tet rest state. This is the full
            // barycentric current position, expressed as a rest-relative displacement.
            const float3 local = Add(Add(Mul(p0, w.x), Mul(p1, w.y)), Add(Mul(p2, w.z), Mul(p3, w.w)));
            const float3 r0 = restPositions[tetIds[base]];
            const float3 r1 = restPositions[tetIds[base + 1]];
            const float3 r2 = restPositions[tetIds[base + 2]];
            const float3 r3 = restPositions[tetIds[base + 3]];
            const float3 restMapped = Add(Add(Mul(r0, w.x), Mul(r1, w.y)), Add(Mul(r2, w.z), Mul(r3, w.w)));
            cornerPos[c] = Add(rest, Sub(local, restMapped));
        }
        else
        {
            cornerPos[c] = Add(rest, Sub(*currentCentroid, restCentroid));
        }
    }

    __global__ void PrepareTetRestKernel(const float3* rest, const int* tetIds, const int* active,
                                         int tetCount, float3* rows)
    {
        int t = blockIdx.x * blockDim.x + threadIdx.x;
        if (t >= tetCount) return;
        if (!active[t]) { rows[t*3] = rows[t*3+1] = rows[t*3+2] = make_float3(0,0,0); return; }
        int i0=tetIds[t*4], i1=tetIds[t*4+1], i2=tetIds[t*4+2], i3=tetIds[t*4+3];
        float3 e1=Sub(rest[i1],rest[i0]), e2=Sub(rest[i2],rest[i0]), e3=Sub(rest[i3],rest[i0]);
        float det=Dot(Cross(e1,e2),e3);
        if (fabsf(det) < 1e-12f) { rows[t*3] = rows[t*3+1] = rows[t*3+2] = make_float3(0,0,0); return; }
        float inv=1.0f/det;
        rows[t*3] = Mul(Cross(e2,e3),inv);
        rows[t*3+1] = Mul(Cross(e3,e1),inv);
        rows[t*3+2] = Mul(Cross(e1,e2),inv);
    }

    __global__ void UpdateTetAlphaKernel(const float* volume, const int* active, int count,
                                         float mu, float lambda, float invH2, float* dev, float* hyd)
    {
        int t=blockIdx.x*blockDim.x+threadIdx.x;
        if (t>=count) return;
        float v=volume[t];
        dev[t]=(active[t] && v>1e-10f && mu>0.0f) ? invH2/(mu*v) : 0.0f;
        hyd[t]=(active[t] && v>1e-10f && lambda>0.0f) ? invH2/(lambda*v) : 0.0f;
    }

    __global__ void IntegrateKernel(float3* pos, float3* prev, float3* vel, const float* invMass,
                                    int count, float h, float3 gravity)
    {
        int i=blockIdx.x*blockDim.x+threadIdx.x;
        if (i>=count || invMass[i]==0.0f) return;
        float3 p=pos[i];
        if (!Finite3(p)) { p=prev[i]; vel[i]=make_float3(0,0,0); pos[i]=p; return; }
        float3 v=Add(vel[i],Mul(gravity,h));
        prev[i]=p; pos[i]=Add(p,Mul(v,h)); vel[i]=v;
    }

    __global__ void SolveEdgesKernel(float3* pos, const float* invMass, const int* edgeIds,
                                     const float* restLengths, const int* edgeFlat, int offset, int count,
                                     float alpha)
    {
        int local=blockIdx.x*blockDim.x+threadIdx.x;
        if (local>=count) return;
        int e=edgeFlat[offset+local], i0=edgeIds[e*2], i1=edgeIds[e*2+1];
        float w0=invMass[i0], w1=invMass[i1], total=w0+w1;
        if (total==0.0f) return;
        float3 p0=pos[i0], p1=pos[i1], d=Sub(p0,p1); float length=sqrtf(Dot(d,d));
        if (length<1e-12f) return;
        float dl=-(length-restLengths[e])/(total+alpha)*0.2f;
        float3 correction=Mul(d,dl/length);
        pos[i0]=Add(p0,Mul(correction,w0)); pos[i1]=Sub(p1,Mul(correction,w1));
    }

    __device__ inline float DetColumns(float3 c0, float3 c1, float3 c2) { return Dot(c0,Cross(c1,c2)); }

    __global__ void SolveNeoHookeanKernel(float3* pos, const float* invMass, const int* tetIds,
                                          const int* tetActive, const float3* invRestRows,
                                          const float* alphaDev, const float* alphaHyd,
                                          const int* tetFlat, int offset, int count)
    {
        int local=blockIdx.x*blockDim.x+threadIdx.x;
        if (local>=count) return;
        int t=tetFlat[offset+local]; if (!tetActive[t]) return;
        int i0=tetIds[t*4], i1=tetIds[t*4+1], i2=tetIds[t*4+2], i3=tetIds[t*4+3];
        float3 p0=pos[i0], p1=pos[i1], p2=pos[i2], p3=pos[i3];
        float w0=invMass[i0], w1=invMass[i1], w2=invMass[i2], w3=invMass[i3];
        float3 r0=invRestRows[t*3], r1=invRestRows[t*3+1], r2=invRestRows[t*3+2];
        if (Dot(r0,r0)+Dot(r1,r1)+Dot(r2,r2)<1e-20f) return;
        float3 e1=Sub(p1,p0), e2=Sub(p2,p0), e3=Sub(p3,p0);
        // F columns: Ds * B columns, with B stored as inverse rows.
        float3 f0=Add(Add(Mul(e1,r0.x),Mul(e2,r1.x)),Mul(e3,r2.x));
        float3 f1=Add(Add(Mul(e1,r0.y),Mul(e2,r1.y)),Mul(e3,r2.y));
        float3 f2=Add(Add(Mul(e1,r0.z),Mul(e2,r1.z)),Mul(e3,r2.z));
        float constraint=Dot(f0,f0)+Dot(f1,f1)+Dot(f2,f2)-3.0f;
        // G = 2 F B^T. Its columns are the position gradients g1/g2/g3.
        float3 g1=Mul(Add(Add(Mul(f0,r0.x),Mul(f1,r0.y)),Mul(f2,r0.z)),2.0f);
        float3 g2=Mul(Add(Add(Mul(f0,r1.x),Mul(f1,r1.y)),Mul(f2,r1.z)),2.0f);
        float3 g3=Mul(Add(Add(Mul(f0,r2.x),Mul(f1,r2.y)),Mul(f2,r2.z)),2.0f);
        float3 g0=Mul(Add(Add(g1,g2),g3),-1.0f);
        float denom=w0*Dot(g0,g0)+w1*Dot(g1,g1)+w2*Dot(g2,g2)+w3*Dot(g3,g3)+alphaDev[t];
        if (denom>=1e-10f) {
            float dl=-constraint/denom*0.2f;
            if (isfinite(dl)) { p0=Add(p0,Mul(g0,w0*dl)); p1=Add(p1,Mul(g1,w1*dl)); p2=Add(p2,Mul(g2,w2*dl)); p3=Add(p3,Mul(g3,w3*dl)); }
        }
        e1=Sub(p1,p0); e2=Sub(p2,p0); e3=Sub(p3,p0);
        f0=Add(Add(Mul(e1,r0.x),Mul(e2,r1.x)),Mul(e3,r2.x));
        f1=Add(Add(Mul(e1,r0.y),Mul(e2,r1.y)),Mul(e3,r2.y));
        f2=Add(Add(Mul(e1,r0.z),Mul(e2,r1.z)),Mul(e3,r2.z));
        float3 c0=Cross(f1,f2), c1=Cross(f2,f0), c2=Cross(f0,f1);
        g1=Add(Add(Mul(c0,r0.x),Mul(c1,r0.y)),Mul(c2,r0.z));
        g2=Add(Add(Mul(c0,r1.x),Mul(c1,r1.y)),Mul(c2,r1.z));
        g3=Add(Add(Mul(c0,r2.x),Mul(c1,r2.y)),Mul(c2,r2.z));
        g0=Mul(Add(Add(g1,g2),g3),-1.0f);
        constraint=DetColumns(f0,f1,f2)-1.0f;
        denom=w0*Dot(g0,g0)+w1*Dot(g1,g1)+w2*Dot(g2,g2)+w3*Dot(g3,g3)+alphaHyd[t];
        if (denom>=1e-10f) {
            float dl=-constraint/denom*0.2f;
            if (isfinite(dl)) { p0=Add(p0,Mul(g0,w0*dl)); p1=Add(p1,Mul(g1,w1*dl)); p2=Add(p2,Mul(g2,w2*dl)); p3=Add(p3,Mul(g3,w3*dl)); }
        }
        pos[i0]=p0; pos[i1]=p1; pos[i2]=p2; pos[i3]=p3;
    }

    __global__ void GroundKernel(float3* pos, float3* prev, int count, float groundY)
    {
        int i=blockIdx.x*blockDim.x+threadIdx.x; if (i>=count) return;
        if (pos[i].y<groundY) { float3 p=prev[i]; p.y=groundY; pos[i]=p; }
    }

    __global__ void PostSolveKernel(const float3* pos, const float3* prev, float3* vel,
                                    const float* invMass, int count, float invH, float damping)
    {
        int i=blockIdx.x*blockDim.x+threadIdx.x;
        if (i>=count || invMass[i]==0.0f) return;
        vel[i]=Mul(Sub(pos[i],prev[i]),invH*(1.0f-damping));
    }

    __device__ inline float3 Clamp3(float3 value, float minimum, float maximum)
    {
        return make_float3(fminf(fmaxf(value.x, minimum), maximum),
                           fminf(fmaxf(value.y, minimum), maximum),
                           fminf(fmaxf(value.z, minimum), maximum));
    }

    __device__ inline float3 ClosestPointOnSegment(float3 point, float3 a, float3 b, float* outT)
    {
        const float3 ab = Sub(b, a);
        const float denominator = Dot(ab, ab);
        const float t = denominator > 1e-12f ? fminf(1.0f, fmaxf(0.0f, Dot(Sub(point, a), ab) / denominator)) : 0.0f;
        if (outT) *outT = t;
        return Add(a, Mul(ab, t));
    }

    __device__ inline float3 ClosestPointOnTriangle(float3 point, float3 a, float3 b, float3 c, float3* bary)
    {
        const float3 ab = Sub(b, a), ac = Sub(c, a), ap = Sub(point, a);
        const float d1 = Dot(ab, ap), d2 = Dot(ac, ap);
        if (d1 <= 0.0f && d2 <= 0.0f) { *bary = make_float3(1,0,0); return a; }
        const float3 bp = Sub(point, b);
        const float d3 = Dot(ab, bp), d4 = Dot(ac, bp);
        if (d3 >= 0.0f && d4 <= d3) { *bary = make_float3(0,1,0); return b; }
        const float vc = d1 * d4 - d3 * d2;
        if (vc <= 0.0f && d1 >= 0.0f && d3 <= 0.0f)
        {
            const float v = d1 / (d1 - d3); *bary = make_float3(1.0f-v,v,0); return Add(a, Mul(ab,v));
        }
        const float3 cp = Sub(point, c);
        const float d5 = Dot(ab, cp), d6 = Dot(ac, cp);
        if (d6 >= 0.0f && d5 <= d6) { *bary = make_float3(0,0,1); return c; }
        const float vb = d5 * d2 - d1 * d6;
        if (vb <= 0.0f && d2 >= 0.0f && d6 <= 0.0f)
        {
            const float w = d2 / (d2 - d6); *bary = make_float3(1.0f-w,0,w); return Add(a, Mul(ac,w));
        }
        const float va = d3 * d6 - d5 * d4;
        if (va <= 0.0f && (d4-d3) >= 0.0f && (d5-d6) >= 0.0f)
        {
            const float w = (d4-d3) / ((d4-d3)+(d5-d6)); *bary=make_float3(0,1.0f-w,w); return Add(b,Mul(Sub(c,b),w));
        }
        const float denom = 1.0f / fmaxf(va+vb+vc, 1e-20f);
        const float v = vb * denom, w = vc * denom; *bary=make_float3(1.0f-v-w,v,w);
        return Add(Add(Mul(a,bary->x),Mul(b,bary->y)),Mul(c,bary->z));
    }

    __device__ inline void ClosestSegmentSegment(float3 a0, float3 a1, float3 b0, float3 b1,
                                                  float* sOut, float* tOut, float3* pa, float3* pb)
    {
        const float3 d1=Sub(a1,a0), d2=Sub(b1,b0), r=Sub(a0,b0);
        const float aa=Dot(d1,d1), ee=Dot(d2,d2), ff=Dot(d2,r);
        float s=0.0f, t=0.0f;
        if (aa <= 1e-12f && ee <= 1e-12f) { }
        else if (aa <= 1e-12f) t=fminf(1.0f,fmaxf(0.0f,ff/ee));
        else
        {
            const float cc=Dot(d1,r);
            if (ee <= 1e-12f) s=fminf(1.0f,fmaxf(0.0f,-cc/aa));
            else
            {
                const float bb=Dot(d1,d2), denom=aa*ee-bb*bb;
                s=fabsf(denom)>1e-12f ? fminf(1.0f,fmaxf(0.0f,(bb*ff-cc*ee)/denom)) : 0.0f;
                const float tNom=bb*s+ff;
                if (tNom < 0.0f) { t=0.0f; s=fminf(1.0f,fmaxf(0.0f,-cc/aa)); }
                else if (tNom > ee) { t=1.0f; s=fminf(1.0f,fmaxf(0.0f,(bb-cc)/aa)); }
                else t=tNom/ee;
            }
        }
        *sOut=s; *tOut=t; *pa=Add(a0,Mul(d1,s)); *pb=Add(b0,Mul(d2,t));
    }

    __device__ inline void ConsiderSurfaceContact(float3 triPoint, float3 capsulePoint, float3 bary,
                                                   float capsuleRadius, float3 fallbackNormal, int capsuleIndex,
                                                   float capsuleT, float* bestC, float3* bestNormal,
                                                   float3* bestBary, int* bestCapsule, float* bestCapsuleT)
    {
        const float3 delta=Sub(triPoint,capsulePoint);
        const float distance=sqrtf(fmaxf(Dot(delta,delta),0.0f));
        const float3 normal=distance>1e-7f ? Mul(delta,1.0f/distance) : fallbackNormal;
        const float constraint=distance-capsuleRadius;
        if (constraint < *bestC)
        {
            *bestC=constraint; *bestNormal=normal; *bestBary=bary;
            *bestCapsule=capsuleIndex; *bestCapsuleT=capsuleT;
        }
    }

    __global__ void SurfaceToolContactKernel(float3* positions, float3* previousPositions,
                                             const float* inverseMass, const int* surfaceTriangles,
                                             const int* surfaceFlat, int groupOffset, int groupCount,
                                             const OrganContextToolCapsule* capsules,
                                             OrganContextToolContactParams params, float alpha,
                                             unsigned int* counters)
    {
        const int local=blockIdx.x*blockDim.x+threadIdx.x;
        if (local>=groupCount || !params.contactEnabled || params.capsuleCount<=0) return;
        const int tri=surfaceFlat[groupOffset+local];
        const int i0=surfaceTriangles[tri*3], i1=surfaceTriangles[tri*3+1], i2=surfaceTriangles[tri*3+2];
        const float w0=inverseMass[i0], w1=inverseMass[i1], w2=inverseMass[i2];
        if (w0+w1+w2<=0.0f) return;
        float3 p0=positions[i0], p1=positions[i1], p2=positions[i2];
        float3 prev0=previousPositions[i0], prev1=previousPositions[i1], prev2=previousPositions[i2];
        bool candidate=false, contacted=false; float deepest=0.0f;

        // The color groups guarantee that no two active threads write a shared surface vertex.
        for (int solvePass=0; solvePass<2; ++solvePass)
        {
            const float3 triMin=make_float3(fminf(p0.x,fminf(p1.x,p2.x)),fminf(p0.y,fminf(p1.y,p2.y)),fminf(p0.z,fminf(p1.z,p2.z)));
            const float3 triMax=make_float3(fmaxf(p0.x,fmaxf(p1.x,p2.x)),fmaxf(p0.y,fmaxf(p1.y,p2.y)),fmaxf(p0.z,fmaxf(p1.z,p2.z)));
            float bestC=FLT_MAX, bestT=0.0f; int bestCapsule=-1;
            float3 bestN=make_float3(0,1,0), bestBary=make_float3(0,0,0);
            for (int cap=0; cap<params.capsuleCount; ++cap)
            {
                const OrganContextToolCapsule capsule=capsules[cap];
                const float3 a=make_float3(capsule.ax,capsule.ay,capsule.az), b=make_float3(capsule.bx,capsule.by,capsule.bz);
                const float3 prevA=make_float3(capsule.prevAx,capsule.prevAy,capsule.prevAz), prevB=make_float3(capsule.prevBx,capsule.prevBy,capsule.prevBz);
                const float shell=fmaxf(capsule.radius+params.contactDistance,1e-6f);
                const float3 capMin=make_float3(fminf(fminf(a.x,b.x),fminf(prevA.x,prevB.x))-shell-params.candidatePadding,
                                                fminf(fminf(a.y,b.y),fminf(prevA.y,prevB.y))-shell-params.candidatePadding,
                                                fminf(fminf(a.z,b.z),fminf(prevA.z,prevB.z))-shell-params.candidatePadding);
                const float3 capMax=make_float3(fmaxf(fmaxf(a.x,b.x),fmaxf(prevA.x,prevB.x))+shell+params.candidatePadding,
                                                fmaxf(fmaxf(a.y,b.y),fmaxf(prevA.y,prevB.y))+shell+params.candidatePadding,
                                                fmaxf(fmaxf(a.z,b.z),fmaxf(prevA.z,prevB.z))+shell+params.candidatePadding);
                if (triMax.x<capMin.x || triMax.y<capMin.y || triMax.z<capMin.z || triMin.x>capMax.x || triMin.y>capMax.y || triMin.z>capMax.z) continue;
                candidate=true;
                const float3 geometricNormalRaw=Cross(Sub(p1,p0),Sub(p2,p0));
                const float geometricLength=sqrtf(fmaxf(Dot(geometricNormalRaw,geometricNormalRaw),1e-16f));
                float3 fallback=Mul(geometricNormalRaw,1.0f/geometricLength);
                const float3 center=Mul(Add(Add(p0,p1),p2),1.0f/3.0f);
                if (Dot(fallback,Sub(center,Mul(Add(a,b),0.5f)))<0.0f) fallback=Mul(fallback,-1.0f);
                float3 bary; float3 point=ClosestPointOnTriangle(a,p0,p1,p2,&bary);
                ConsiderSurfaceContact(point,a,bary,shell,fallback,cap,0.0f,&bestC,&bestN,&bestBary,&bestCapsule,&bestT);
                point=ClosestPointOnTriangle(b,p0,p1,p2,&bary);
                ConsiderSurfaceContact(point,b,bary,shell,fallback,cap,1.0f,&bestC,&bestN,&bestBary,&bestCapsule,&bestT);
                float s,t; float3 capPoint,edgePoint;
                ClosestSegmentSegment(a,b,p0,p1,&s,&t,&capPoint,&edgePoint);
                ConsiderSurfaceContact(edgePoint,capPoint,make_float3(1.0f-t,t,0),shell,fallback,cap,s,&bestC,&bestN,&bestBary,&bestCapsule,&bestT);
                ClosestSegmentSegment(a,b,p1,p2,&s,&t,&capPoint,&edgePoint);
                ConsiderSurfaceContact(edgePoint,capPoint,make_float3(0,1.0f-t,t),shell,fallback,cap,s,&bestC,&bestN,&bestBary,&bestCapsule,&bestT);
                ClosestSegmentSegment(a,b,p2,p0,&s,&t,&capPoint,&edgePoint);
                ConsiderSurfaceContact(edgePoint,capPoint,make_float3(t,0,1.0f-t),shell,fallback,cap,s,&bestC,&bestN,&bestBary,&bestCapsule,&bestT);
            }
            if (bestC >= 0.0f || bestCapsule < 0) break;
            const float denom=w0*bestBary.x*bestBary.x+w1*bestBary.y*bestBary.y+w2*bestBary.z*bestBary.z;
            if (denom<=1e-12f) break;
            const float depth=-bestC, dl=depth/(denom+alpha);
            const float3 correction=Mul(bestN,dl);
            p0=Add(p0,Mul(correction,w0*bestBary.x)); p1=Add(p1,Mul(correction,w1*bestBary.y)); p2=Add(p2,Mul(correction,w2*bestBary.z));
            const OrganContextToolCapsule capsule=capsules[bestCapsule];
            const float3 toolNow=Add(Mul(make_float3(capsule.ax,capsule.ay,capsule.az),1.0f-bestT),Mul(make_float3(capsule.bx,capsule.by,capsule.bz),bestT));
            const float3 toolPrev=Add(Mul(make_float3(capsule.prevAx,capsule.prevAy,capsule.prevAz),1.0f-bestT),Mul(make_float3(capsule.prevBx,capsule.prevBy,capsule.prevBz),bestT));
            const float3 contactNow=Add(Add(Mul(p0,bestBary.x),Mul(p1,bestBary.y)),Mul(p2,bestBary.z));
            const float3 contactPrev=Add(Add(Mul(prev0,bestBary.x),Mul(prev1,bestBary.y)),Mul(prev2,bestBary.z));
            const float3 relative=Sub(Sub(contactNow,contactPrev),Sub(toolNow,toolPrev));
            const float normalVelocity=Dot(relative,bestN);
            if (normalVelocity<0.0f)
            {
                const float3 change=Mul(bestN,normalVelocity/denom);
                prev0=Add(prev0,Mul(change,w0*bestBary.x)); prev1=Add(prev1,Mul(change,w1*bestBary.y)); prev2=Add(prev2,Mul(change,w2*bestBary.z));
            }
            const float friction=fminf(fmaxf(params.tangentialFriction*capsule.friction*params.tangentialDamping,0.0f),1.0f);
            const float3 tangent=Sub(relative,Mul(bestN,Dot(relative,bestN)));
            if (friction>0.0f)
            {
                const float3 change=Mul(tangent,-friction/denom);
                p0=Add(p0,Mul(change,w0*bestBary.x)); p1=Add(p1,Mul(change,w1*bestBary.y)); p2=Add(p2,Mul(change,w2*bestBary.z));
            }
            contacted=true; deepest=fmaxf(deepest,depth);
        }
        if (candidate) { atomicAdd(&counters[0],1u); atomicAdd(&counters[4],1u); }
        if (contacted)
        {
            atomicAdd(&counters[1],1u); atomicMax(&counters[2],__float_as_uint(deepest));
            atomicAdd(&counters[5],1u); atomicMax(&counters[6],__float_as_uint(deepest));
            positions[i0]=p0; positions[i1]=p1; positions[i2]=p2;
            previousPositions[i0]=prev0; previousPositions[i1]=prev1; previousPositions[i2]=prev2;
        }
    }

    __global__ void CaptureGraspKernel(const float3* positions, int count,
                                       unsigned char* graspMask, float3* frameCoordinates,
                                       float* weights, OrganContextToolContactParams params)
    {
        const int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= count || graspMask[i]) return;
        const float3 p = positions[i];
        if (p.x < params.graspBoundsMinX || p.y < params.graspBoundsMinY || p.z < params.graspBoundsMinZ ||
            p.x > params.graspBoundsMaxX || p.y > params.graspBoundsMaxY || p.z > params.graspBoundsMaxZ)
            return;

        const float3 center = make_float3(params.frameCenterX, params.frameCenterY, params.frameCenterZ);
        const float3 u = make_float3(params.axisUX, params.axisUY, params.axisUZ);
        const float3 v = make_float3(params.axisVX, params.axisVY, params.axisVZ);
        const float3 w = make_float3(params.axisWX, params.axisWY, params.axisWZ);
        const float3 coordinate = make_float3(Dot(Sub(p, center), u), Dot(Sub(p, center), v), Dot(Sub(p, center), w));
        const float halfU = fmaxf((params.graspBoundsMaxX - params.graspBoundsMinX + params.graspBoundsMaxY - params.graspBoundsMinY) * 0.25f, 1e-4f);
        const float halfV = fmaxf((params.graspBoundsMaxZ - params.graspBoundsMinZ) * 0.5f, 1e-4f);
        const float radius = sqrtf((coordinate.x / halfU) * (coordinate.x / halfU) +
                                   (coordinate.y / halfV) * (coordinate.y / halfV));
        const float coreRadius = fminf(fmaxf(params.graspCoreRadius, 0.1f), 1.0f);
        const float influenceRadius = fminf(fmaxf(params.graspInfluenceRadius, coreRadius), 1.0f);
        if (radius > influenceRadius) return;

        float weight = 1.0f;
        if (radius > coreRadius && influenceRadius > coreRadius + 1e-5f)
        {
            const float sigma = fmaxf(params.graspGaussianWidth, 0.05f);
            const float distanceFromCore = radius - coreRadius;
            const float edgeDistance = influenceRadius - coreRadius;
            const float gaussian = expf(-0.5f * distanceFromCore * distanceFromCore / (sigma * sigma));
            const float edgeGaussian = expf(-0.5f * edgeDistance * edgeDistance / (sigma * sigma));
            weight = fminf(fmaxf((gaussian - edgeGaussian) /
                fmaxf(1.0f - edgeGaussian, 1e-5f), 0.0f), 1.0f);
        }
        graspMask[i] = 1;
        frameCoordinates[i] = coordinate;
        weights[i] = weight;
    }

    __global__ void ReleaseGraspKernel(float3* velocities, int count, unsigned char* graspMask)
    {
        const int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= count || !graspMask[i]) return;
        graspMask[i] = 0;
        velocities[i] = Mul(velocities[i], 0.1f);
    }

    __global__ void SolveGraspKernel(float3* positions, float3* previousPositions, float3* velocities,
                                     const float* inverseMass, int count, const unsigned char* graspMask,
                                     const float3* frameCoordinates, const float* weights,
                                     OrganContextToolContactParams params,
                                     OrganContextToolContactParams captureParams,
                                     float blend, float dt)
    {
        const int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= count || !graspMask[i] || inverseMass[i] == 0.0f) return;
        const float3 center = make_float3(params.frameCenterX, params.frameCenterY, params.frameCenterZ);
        const float3 u = make_float3(params.axisUX, params.axisUY, params.axisUZ);
        const float3 v = make_float3(params.axisVX, params.axisVY, params.axisVZ);
        const float3 w = make_float3(params.axisWX, params.axisWY, params.axisWZ);
        const float3 captureCenter = make_float3(
            captureParams.frameCenterX, captureParams.frameCenterY, captureParams.frameCenterZ);
        const float3 captureU = make_float3(
            captureParams.axisUX, captureParams.axisUY, captureParams.axisUZ);
        const float3 captureV = make_float3(
            captureParams.axisVX, captureParams.axisVY, captureParams.axisVZ);
        const float3 captureW = make_float3(
            captureParams.axisWX, captureParams.axisWY, captureParams.axisWZ);
        const float3 c = frameCoordinates[i];
        const float smooth = blend * blend * (3.0f - 2.0f * blend);
        const float weight = fminf(fmaxf(weights[i], 0.0f), 1.0f);
        const float3 captureTarget = Add(Add(Add(captureCenter, Mul(captureU, c.x)), Mul(captureV, c.y)),
                                         Mul(captureW, c.z));
        const float3 rigidTarget = Add(Add(Add(center, Mul(u, c.x)), Mul(v, c.y)), Mul(w, c.z));
        const float3 weightedMotionTarget = Add(captureTarget, Mul(Sub(rigidTarget, captureTarget), weight));
        const float3 target = Add(weightedMotionTarget,
            Mul(w, fmaxf(params.graspHeight, 0.0f) * weight * smooth));
        if (weight >= 0.9999f)
        {
            positions[i] = target;
            previousPositions[i] = target;
            velocities[i] = make_float3(0, 0, 0);
            return;
        }

        const float follow = weight * smooth *
            (1.0f - expf(-fmaxf(params.graspSoftFollowRate, 0.1f) * dt));
        const float3 correction = Mul(Sub(target, positions[i]), follow);
        positions[i] = Add(positions[i], correction);
        previousPositions[i] = Add(previousPositions[i], correction);
    }

    __global__ void ToolContactKernel(float3* positions, float3* previousPositions, const float* inverseMass,
                                      int particleCount, const OrganContextToolCapsule* capsules,
                                      OrganContextToolContactParams params, float alpha, unsigned int* counters)
    {
        const int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= particleCount || inverseMass[i] == 0.0f || !params.contactEnabled || params.capsuleCount <= 0) return;
        float3 p = positions[i];
        float3 prev = previousPositions[i];
        bool activeCandidate = false;
        bool contact = false;
        float deepest = 0.0f;
        for (int capsuleIndex = 0; capsuleIndex < params.capsuleCount; ++capsuleIndex)
        {
            const OrganContextToolCapsule capsule = capsules[capsuleIndex];
            float t = 0.0f;
            const float3 a = make_float3(capsule.ax, capsule.ay, capsule.az);
            const float3 b = make_float3(capsule.bx, capsule.by, capsule.bz);
            const float3 closest = ClosestPointOnSegment(p, a, b, &t);
            const float3 difference = Sub(p, closest);
            const float distanceSquared = Dot(difference, difference);
            const float radius = fmaxf(capsule.radius + params.contactDistance, 1e-6f);
            const float candidateRadius = radius + (params.useCandidateCulling ? fmaxf(params.candidatePadding, 0.0f) : 0.0f);
            if (distanceSquared <= candidateRadius * candidateRadius) activeCandidate = true;
            const float distance = sqrtf(fmaxf(distanceSquared, 1e-16f));
            if (distance >= radius) continue;

            float3 normal = distance > 1e-7f ? Mul(difference, 1.0f / distance) : make_float3(0, 1, 0);
            const float depth = radius - distance;
            const float correctionScale = depth / (1.0f + alpha * fmaxf(inverseMass[i], 1e-6f));
            p = Add(p, Mul(normal, correctionScale));

            const float3 prevA = make_float3(capsule.prevAx, capsule.prevAy, capsule.prevAz);
            const float3 prevB = make_float3(capsule.prevBx, capsule.prevBy, capsule.prevBz);
            const float3 toolDelta = Add(Mul(Sub(a, prevA), 1.0f - t), Mul(Sub(b, prevB), t));
            float3 relative = Sub(Sub(p, prev), toolDelta);
            const float normalVelocity = Dot(relative, normal);
            if (normalVelocity < 0.0f) prev = Add(prev, Mul(normal, normalVelocity));
            const float3 tangent = Sub(relative, Mul(normal, Dot(relative, normal)));
            const float friction = fminf(fmaxf(params.tangentialFriction * capsule.friction + params.tangentialDamping, 0.0f), 1.0f);
            prev = Add(prev, Mul(tangent, friction));
            contact = true;
            deepest = fmaxf(deepest, depth);
        }
        if (activeCandidate) atomicAdd(&counters[0], 1u);
        if (contact)
        {
            atomicAdd(&counters[1], 1u);
            atomicMax(&counters[2], __float_as_uint(deepest));
            positions[i] = p;
            previousPositions[i] = prev;
        }
    }

    __global__ void CountGraspKernel(const unsigned char* graspMask, int count, unsigned int* counters)
    {
        const int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < count && graspMask[i]) atomicAdd(&counters[3], 1u);
    }

    CudaOrganContext* FindContext(uint32_t handle)
    {
        const auto it=g_contexts.find(handle); return it==g_contexts.end()?nullptr:it->second.get();
    }

    bool Check(cudaError_t error, CudaOrganContext& context)
    {
        if (error==cudaSuccess) return true;
        context.stats.lastError=ORGAN_CONTEXT_CUDA_FAILURE; return false;
    }

    struct FaceKey
    {
        int a, b, c;
        bool operator==(const FaceKey& other) const { return a==other.a && b==other.b && c==other.c; }
    };

    struct FaceKeyHash
    {
        size_t operator()(const FaceKey& key) const
        {
            size_t h=static_cast<size_t>(key.a)*73856093u;
            h^=static_cast<size_t>(key.b)*19349663u;
            h^=static_cast<size_t>(key.c)*83492791u;
            return h;
        }
    };

    std::vector<int4> BuildInternalFaces(const int* tetIds, int tetCount)
    {
        struct FaceRecord { int4 vertices; int count; };
        std::unordered_map<FaceKey,FaceRecord,FaceKeyHash> faces;
        constexpr int corners[4][3]={{0,1,2},{0,1,3},{0,2,3},{1,2,3}};
        for (int t=0;t<tetCount;++t)
        {
            for (int f=0;f<4;++f)
            {
                int ids[3]={tetIds[t*4+corners[f][0]],tetIds[t*4+corners[f][1]],tetIds[t*4+corners[f][2]]};
                std::sort(ids,ids+3);
                FaceKey key{ids[0],ids[1],ids[2]};
                auto it=faces.find(key);
                if (it==faces.end()) faces.emplace(key,FaceRecord{make_int4(ids[0],ids[1],ids[2],0),1});
                else ++it->second.count;
            }
        }
        std::vector<int4> result;
        result.reserve(faces.size()/2);
        for (const auto& entry:faces) if (entry.second.count>1) result.push_back(entry.second.vertices);
        return result;
    }

    struct HostColorGroups
    {
        std::vector<int> offsets;
        std::vector<int> counts;
        std::vector<int> flat;
    };

    HostColorGroups BuildColorGroups(const std::vector<int>& ids, int elementCount,
                                     int stride, int particleCount)
    {
        HostColorGroups result;
        if (elementCount <= 0 || stride <= 0 || particleCount <= 0) return result;
        std::vector<std::vector<int>> incident(static_cast<size_t>(particleCount));
        for (int element=0;element<elementCount;++element)
            for (int corner=0;corner<stride;++corner)
                incident[ids[size_t(element)*stride+corner]].push_back(element);
        std::vector<int> colors(static_cast<size_t>(elementCount),-1);
        int maxColor=-1;
        for (int element=0;element<elementCount;++element)
        {
            std::unordered_set<int> used;
            for (int corner=0;corner<stride;++corner)
                for (int neighbour:incident[ids[size_t(element)*stride+corner]])
                    if (colors[neighbour]>=0) used.insert(colors[neighbour]);
            int color=0; while (used.count(color)) ++color;
            colors[element]=color; maxColor=std::max(maxColor,color);
        }
        result.offsets.resize(static_cast<size_t>(maxColor+1));
        result.counts.resize(static_cast<size_t>(maxColor+1));
        for (int color=0;color<=maxColor;++color)
        {
            result.offsets[color]=static_cast<int>(result.flat.size());
            for (int element=0;element<elementCount;++element)
                if (colors[element]==color) result.flat.push_back(element);
            result.counts[color]=static_cast<int>(result.flat.size())-result.offsets[color];
        }
        return result;
    }

    struct FaceOwnerRecord
    {
        int oriented[3]{};
        int count=0;
    };

    void BuildFaces(const std::vector<int>& tetIds, int tetCount,
                    std::vector<int>& surface, std::vector<int4>& internal)
    {
        std::unordered_map<FaceKey,FaceOwnerRecord,FaceKeyHash> faces;
        constexpr int corners[4][3]={{0,2,1},{0,1,3},{0,3,2},{1,2,3}};
        for (int tet=0;tet<tetCount;++tet)
            for (int face=0;face<4;++face)
            {
                int oriented[3]={tetIds[size_t(tet)*4+corners[face][0]],
                                 tetIds[size_t(tet)*4+corners[face][1]],
                                 tetIds[size_t(tet)*4+corners[face][2]]};
                int sorted[3]={oriented[0],oriented[1],oriented[2]};
                std::sort(sorted,sorted+3);
                FaceKey key{sorted[0],sorted[1],sorted[2]};
                auto it=faces.find(key);
                if (it==faces.end())
                {
                    FaceOwnerRecord record;
                    std::copy(oriented,oriented+3,record.oriented);
                    record.count=1;
                    faces.emplace(key,record);
                }
                else ++it->second.count;
            }
        surface.clear(); internal.clear();
        for (const auto& entry:faces)
        {
            const FaceOwnerRecord& record=entry.second;
            if (record.count==1)
                surface.insert(surface.end(),record.oriented,record.oriented+3);
            else if (record.count==2)
                internal.push_back(make_int4(entry.first.a,entry.first.b,entry.first.c,0));
        }
    }

    void BuildUniqueEdges(const std::vector<int>& tetIds, int tetCount,
                          const std::vector<float3>& rest, std::vector<int>& edges,
                          std::vector<float>& lengths)
    {
        constexpr int pairs[6][2]={{0,1},{0,2},{0,3},{1,2},{1,3},{2,3}};
        std::unordered_set<uint64_t> seen;
        edges.clear(); lengths.clear();
        for (int tet=0;tet<tetCount;++tet)
            for (int pair=0;pair<6;++pair)
            {
                int a=tetIds[size_t(tet)*4+pairs[pair][0]];
                int b=tetIds[size_t(tet)*4+pairs[pair][1]];
                if (a>b) std::swap(a,b);
                uint64_t key=(uint64_t(uint32_t(a))<<32)|uint32_t(b);
                if (!seen.insert(key).second) continue;
                edges.push_back(a); edges.push_back(b);
                float3 delta=Sub(rest[a],rest[b]);
                lengths.push_back(std::sqrt(Dot(delta,delta)));
            }
    }

    template <typename T>
    bool AllocateAndUploadRaw(T*& destination, const std::vector<T>& source)
    {
        destination=nullptr;
        if (source.empty()) return true;
        const size_t bytes=sizeof(T)*source.size();
        return cudaMalloc(&destination,bytes)==cudaSuccess &&
               cudaMemcpy(destination,source.data(),bytes,cudaMemcpyHostToDevice)==cudaSuccess;
    }

    bool AllocateAndUploadFloat3AsFloat(float*& destination, const std::vector<float3>& source)
    {
        destination=nullptr;
        if (source.empty()) return true;
        const size_t bytes=sizeof(float3)*source.size();
        return cudaMalloc(&destination,bytes)==cudaSuccess &&
               cudaMemcpy(destination,source.data(),bytes,cudaMemcpyHostToDevice)==cudaSuccess;
    }

    bool AllocateZeroedSequence(unsigned int*& destination, size_t count)
    {
        destination=nullptr;
        if (count==0) return true;
        const size_t bytes=sizeof(unsigned int)*count;
        return cudaMalloc(&destination,bytes)==cudaSuccess && cudaMemset(destination,0,bytes)==cudaSuccess;
    }

    template <typename T>
    void FreeAndReplace(T*& destination, T*& replacement)
    {
        cudaFree(destination); destination=replacement; replacement=nullptr;
    }

    bool ApplyTetFracture(CudaOrganContext& c, const CutEventSummary& event,
                          const unsigned int* counters)
    {
        c.fractureStats.processedEventSequence=event.sequence;
        if (c.fractureStats.applied)
        {
            c.fractureStats.rejectedReason=ORGAN_FRACTURE_REJECT_ALREADY_APPLIED;
            return false;
        }
        if (!c.fractureStats.runtimeCompatible || !c.tetToGridStats.configured)
        {
            c.fractureStats.rejectedReason=ORGAN_FRACTURE_REJECT_RUNTIME_PATH;
            return false;
        }
        if (!event.valid || event.rawCutPoints<4 || counters[1]==0 || counters[2]==0 || counters[4]==0 ||
            counters[6]==0 || counters[7]<2)
        {
            c.fractureStats.rejectedReason=ORGAN_FRACTURE_REJECT_NOT_FULLY_PENETRATING;
            return false;
        }

        const auto start=std::chrono::high_resolution_clock::now();
        const int oldParticleCount=c.stats.particleCount;
        const int tetCount=c.stats.tetCount;
        std::vector<float3> rest(oldParticleCount), positions(oldParticleCount), previous(oldParticleCount), velocities(oldParticleCount);
        std::vector<float> inverseMass(oldParticleCount), restVolumes(tetCount), graspWeights(oldParticleCount);
        std::vector<unsigned char> graspMask(oldParticleCount);
        std::vector<float3> graspFrame(oldParticleCount);
        std::vector<int> tetIds(size_t(tetCount)*4), tetActive(tetCount);
        if (cudaMemcpy(rest.data(),c.restPositions,sizeof(float3)*size_t(oldParticleCount),cudaMemcpyDeviceToHost)!=cudaSuccess ||
            cudaMemcpy(positions.data(),c.positions,sizeof(float3)*size_t(oldParticleCount),cudaMemcpyDeviceToHost)!=cudaSuccess ||
            cudaMemcpy(previous.data(),c.previousPositions,sizeof(float3)*size_t(oldParticleCount),cudaMemcpyDeviceToHost)!=cudaSuccess ||
            cudaMemcpy(velocities.data(),c.velocities,sizeof(float3)*size_t(oldParticleCount),cudaMemcpyDeviceToHost)!=cudaSuccess ||
            cudaMemcpy(inverseMass.data(),c.inverseMass,sizeof(float)*size_t(oldParticleCount),cudaMemcpyDeviceToHost)!=cudaSuccess ||
            cudaMemcpy(restVolumes.data(),c.restVolumes,sizeof(float)*size_t(tetCount),cudaMemcpyDeviceToHost)!=cudaSuccess ||
            cudaMemcpy(tetIds.data(),c.tetIds,sizeof(int)*tetIds.size(),cudaMemcpyDeviceToHost)!=cudaSuccess ||
            cudaMemcpy(tetActive.data(),c.tetActive,sizeof(int)*size_t(tetCount),cudaMemcpyDeviceToHost)!=cudaSuccess ||
            cudaMemcpy(graspMask.data(),c.graspMask,sizeof(unsigned char)*size_t(oldParticleCount),cudaMemcpyDeviceToHost)!=cudaSuccess ||
            cudaMemcpy(graspFrame.data(),c.graspFrameCoordinates,sizeof(float3)*size_t(oldParticleCount),cudaMemcpyDeviceToHost)!=cudaSuccess ||
            cudaMemcpy(graspWeights.data(),c.graspWeights,sizeof(float)*size_t(oldParticleCount),cudaMemcpyDeviceToHost)!=cudaSuccess)
        {
            c.fractureStats.rejectedReason=ORGAN_FRACTURE_REJECT_TOPOLOGY_REBUILD;
            c.fractureStats.lastError=ORGAN_CONTEXT_CUDA_FAILURE;
            return false;
        }

        const float3 n0=make_float3(event.normal[0],event.normal[1],event.normal[2]);
        const float nLength=std::sqrt(Dot(n0,n0));
        if (nLength<1e-8f)
        {
            c.fractureStats.rejectedReason=ORGAN_FRACTURE_REJECT_NOT_FULLY_PENETRATING;
            return false;
        }
        const float3 normal=Mul(n0,1.0f/nLength);
        const float3 origin=make_float3(event.hitPoint[0],event.hitPoint[1],event.hitPoint[2]);
        std::vector<unsigned char> tetPositive(tetCount,0), usedPositive(oldParticleCount,0), usedNegative(oldParticleCount,0);
        std::vector<float> positiveVolume(oldParticleCount,0.0f), negativeVolume(oldParticleCount,0.0f);
        for (int tet=0;tet<tetCount;++tet)
        {
            if (!tetActive[tet]) continue;
            float3 center=make_float3(0,0,0);
            for (int corner=0;corner<4;++corner) center=Add(center,Mul(positions[tetIds[size_t(tet)*4+corner]],0.25f));
            const bool positive=Dot(Sub(center,origin),normal)>=0.0f;
            tetPositive[tet]=positive?1:0;
            const float volume=std::max(std::fabs(restVolumes[tet]),1e-12f);
            for (int corner=0;corner<4;++corner)
            {
                const int vertex=tetIds[size_t(tet)*4+corner];
                if (positive) { usedPositive[vertex]=1; positiveVolume[vertex]+=volume; }
                else { usedNegative[vertex]=1; negativeVolume[vertex]+=volume; }
            }
        }

        std::vector<int> duplicate(oldParticleCount,-1);
        for (int vertex=0;vertex<oldParticleCount;++vertex)
            if (usedPositive[vertex] && usedNegative[vertex])
            {
                duplicate[vertex]=static_cast<int>(rest.size());
                rest.push_back(rest[vertex]); positions.push_back(positions[vertex]);
                previous.push_back(previous[vertex]); velocities.push_back(velocities[vertex]);
                inverseMass.push_back(inverseMass[vertex]); graspMask.push_back(graspMask[vertex]);
                graspFrame.push_back(graspFrame[vertex]); graspWeights.push_back(graspWeights[vertex]);
            }
        const int duplicateCount=static_cast<int>(rest.size())-oldParticleCount;
        if (duplicateCount<=0)
        {
            c.fractureStats.rejectedReason=ORGAN_FRACTURE_REJECT_NO_SHARED_NODES;
            return false;
        }

        const float halfGap=std::max(c.fractureStats.initialGap,0.0f)*0.5f;
        for (int vertex=0;vertex<oldParticleCount;++vertex)
        {
            const int copy=duplicate[vertex]; if (copy<0) continue;
            positions[vertex]=Sub(positions[vertex],Mul(normal,halfGap));
            previous[vertex]=Sub(previous[vertex],Mul(normal,halfGap));
            positions[copy]=Add(positions[copy],Mul(normal,halfGap));
            previous[copy]=Add(previous[copy],Mul(normal,halfGap));
            if (inverseMass[vertex]>0.0f)
            {
                const float originalMass=1.0f/inverseMass[vertex];
                const float total=positiveVolume[vertex]+negativeVolume[vertex];
                const float positiveFraction=total>1e-12f?positiveVolume[vertex]/total:0.5f;
                inverseMass[vertex]=1.0f/std::max(originalMass*(1.0f-positiveFraction),1e-12f);
                inverseMass[copy]=1.0f/std::max(originalMass*positiveFraction,1e-12f);
            }
            else inverseMass[copy]=0.0f;
        }
        for (int tet=0;tet<tetCount;++tet)
            if (tetPositive[tet])
                for (int corner=0;corner<4;++corner)
                {
                    int& vertex=tetIds[size_t(tet)*4+corner];
                    if (duplicate[vertex]>=0) vertex=duplicate[vertex];
                }

        std::vector<int> edges, surface;
        std::vector<float> edgeLengths;
        std::vector<int4> internal;
        BuildUniqueEdges(tetIds,tetCount,rest,edges,edgeLengths);
        BuildFaces(tetIds,tetCount,surface,internal);
        HostColorGroups edgeColors=BuildColorGroups(edges,static_cast<int>(edgeLengths.size()),2,static_cast<int>(rest.size()));
        HostColorGroups tetColors=BuildColorGroups(tetIds,tetCount,4,static_cast<int>(rest.size()));
        HostColorGroups surfaceColors=BuildColorGroups(surface,static_cast<int>(surface.size()/3),3,static_cast<int>(rest.size()));
        if (edges.empty() || surface.empty() || edgeColors.counts.empty() || tetColors.counts.empty() || surfaceColors.counts.empty())
        {
            c.fractureStats.rejectedReason=ORGAN_FRACTURE_REJECT_TOPOLOGY_REBUILD;
            return false;
        }

#define NEW_BUFFER(type,name) type* name=nullptr
        NEW_BUFFER(float,newRest); NEW_BUFFER(int,newTetIds); NEW_BUFFER(float,newInverseMass);
        NEW_BUFFER(int,newSurface); NEW_BUFFER(int,newEdges); NEW_BUFFER(float,newEdgeLengths);
        NEW_BUFFER(float3,newPositions); NEW_BUFFER(float3,newPrevious); NEW_BUFFER(float3,newVelocities);
        NEW_BUFFER(unsigned char,newGraspMask); NEW_BUFFER(float3,newGraspFrame); NEW_BUFFER(float,newGraspWeights);
        NEW_BUFFER(int4,newInternal); NEW_BUFFER(unsigned int,newCandidateTets); NEW_BUFFER(unsigned int,newCandidateEdges);
        NEW_BUFFER(unsigned int,newCandidateInternal); NEW_BUFFER(unsigned int,newCandidateSurface);
        NEW_BUFFER(int,newEdgeOffsets); NEW_BUFFER(int,newEdgeCounts); NEW_BUFFER(int,newEdgeFlat);
        NEW_BUFFER(int,newTetOffsets); NEW_BUFFER(int,newTetCounts); NEW_BUFFER(int,newTetFlat);
        NEW_BUFFER(int,newSurfaceOffsets); NEW_BUFFER(int,newSurfaceCounts); NEW_BUFFER(int,newSurfaceFlat);
#undef NEW_BUFFER
        bool ok=AllocateAndUploadFloat3AsFloat(newRest,rest);
        ok=ok && AllocateAndUploadRaw(newTetIds,tetIds) && AllocateAndUploadRaw(newInverseMass,inverseMass) &&
           AllocateAndUploadRaw(newSurface,surface) && AllocateAndUploadRaw(newEdges,edges) && AllocateAndUploadRaw(newEdgeLengths,edgeLengths) &&
           AllocateAndUploadRaw(newPositions,positions) && AllocateAndUploadRaw(newPrevious,previous) && AllocateAndUploadRaw(newVelocities,velocities) &&
           AllocateAndUploadRaw(newGraspMask,graspMask) && AllocateAndUploadRaw(newGraspFrame,graspFrame) && AllocateAndUploadRaw(newGraspWeights,graspWeights) &&
           AllocateAndUploadRaw(newInternal,internal) && AllocateAndUploadRaw(newEdgeOffsets,edgeColors.offsets) &&
           AllocateAndUploadRaw(newEdgeCounts,edgeColors.counts) && AllocateAndUploadRaw(newEdgeFlat,edgeColors.flat) &&
           AllocateAndUploadRaw(newTetOffsets,tetColors.offsets) && AllocateAndUploadRaw(newTetCounts,tetColors.counts) &&
           AllocateAndUploadRaw(newTetFlat,tetColors.flat) && AllocateAndUploadRaw(newSurfaceOffsets,surfaceColors.offsets) &&
           AllocateAndUploadRaw(newSurfaceCounts,surfaceColors.counts) && AllocateAndUploadRaw(newSurfaceFlat,surfaceColors.flat);
        if (ok) ok=AllocateZeroedSequence(newCandidateTets,size_t(tetCount)) &&
                           AllocateZeroedSequence(newCandidateEdges,edgeLengths.size()) &&
                           AllocateZeroedSequence(newCandidateInternal,internal.size()) &&
                           AllocateZeroedSequence(newCandidateSurface,surface.size()/3);
        if (!ok)
        {
#define FREE_NEW(name) cudaFree(name)
            FREE_NEW(newRest); FREE_NEW(newTetIds); FREE_NEW(newInverseMass); FREE_NEW(newSurface); FREE_NEW(newEdges); FREE_NEW(newEdgeLengths);
            FREE_NEW(newPositions); FREE_NEW(newPrevious); FREE_NEW(newVelocities); FREE_NEW(newGraspMask); FREE_NEW(newGraspFrame); FREE_NEW(newGraspWeights);
            FREE_NEW(newInternal); FREE_NEW(newCandidateTets); FREE_NEW(newCandidateEdges); FREE_NEW(newCandidateInternal); FREE_NEW(newCandidateSurface);
            FREE_NEW(newEdgeOffsets); FREE_NEW(newEdgeCounts); FREE_NEW(newEdgeFlat); FREE_NEW(newTetOffsets); FREE_NEW(newTetCounts); FREE_NEW(newTetFlat);
            FREE_NEW(newSurfaceOffsets); FREE_NEW(newSurfaceCounts); FREE_NEW(newSurfaceFlat);
#undef FREE_NEW
            c.fractureStats.rejectedReason=ORGAN_FRACTURE_REJECT_TOPOLOGY_REBUILD;
            c.fractureStats.lastError=ORGAN_CONTEXT_CUDA_FAILURE;
            return false;
        }

        FreeAndReplace(c.restPositions,newRest); FreeAndReplace(c.tetIds,newTetIds); FreeAndReplace(c.inverseMass,newInverseMass);
        FreeAndReplace(c.surfaceTriangles,newSurface); FreeAndReplace(c.edgeIds,newEdges); FreeAndReplace(c.edgeRestLengths,newEdgeLengths);
        FreeAndReplace(c.positions,newPositions); FreeAndReplace(c.previousPositions,newPrevious); FreeAndReplace(c.velocities,newVelocities);
        FreeAndReplace(c.graspMask,newGraspMask); FreeAndReplace(c.graspFrameCoordinates,newGraspFrame); FreeAndReplace(c.graspWeights,newGraspWeights);
        FreeAndReplace(c.internalFaces,newInternal); FreeAndReplace(c.candidateTetSequence,newCandidateTets);
        FreeAndReplace(c.candidateEdgeSequence,newCandidateEdges); FreeAndReplace(c.candidateSharedFaceSequence,newCandidateInternal);
        FreeAndReplace(c.candidateSurfaceFaceSequence,newCandidateSurface);
        FreeAndReplace(c.edgeColorOffsets,newEdgeOffsets); FreeAndReplace(c.edgeColorCounts,newEdgeCounts); FreeAndReplace(c.edgeColorFlat,newEdgeFlat);
        FreeAndReplace(c.tetColorOffsets,newTetOffsets); FreeAndReplace(c.tetColorCounts,newTetCounts); FreeAndReplace(c.tetColorFlat,newTetFlat);
        FreeAndReplace(c.surfaceColorOffsets,newSurfaceOffsets); FreeAndReplace(c.surfaceColorCounts,newSurfaceCounts); FreeAndReplace(c.surfaceColorFlat,newSurfaceFlat);
        c.hostEdgeColorOffsets=edgeColors.offsets; c.hostEdgeColorCounts=edgeColors.counts;
        c.hostTetColorOffsets=tetColors.offsets; c.hostTetColorCounts=tetColors.counts;
        c.hostSurfaceColorOffsets=surfaceColors.offsets; c.hostSurfaceColorCounts=surfaceColors.counts;
        c.edgeColorCount=static_cast<int>(edgeColors.counts.size()); c.tetColorCount=static_cast<int>(tetColors.counts.size());
        c.surfaceColorCount=static_cast<int>(surfaceColors.counts.size());
        c.stats.particleCount=static_cast<int>(rest.size()); c.stats.edgeConstraintCount=static_cast<int>(edgeLengths.size());
        c.stats.surfaceTriangleCount=static_cast<int>(surface.size()/3); c.internalFaceCount=static_cast<int>(internal.size());
        c.fractureStats.applied=1; c.fractureStats.rejectedReason=ORGAN_FRACTURE_REJECT_NONE;
        c.fractureStats.originalParticleCount=oldParticleCount; c.fractureStats.currentParticleCount=c.stats.particleCount;
        c.fractureStats.duplicatedNodeCount=duplicateCount; c.fractureStats.rebuiltEdgeConstraintCount=c.stats.edgeConstraintCount;
        c.fractureStats.rebuiltSurfaceTriangleCount=c.stats.surfaceTriangleCount; c.fractureStats.lastError=0;
        const auto end=std::chrono::high_resolution_clock::now();
        c.fractureStats.lastFractureMilliseconds=std::chrono::duration<float,std::milli>(end-start).count();
        return true;
    }

    __device__ __forceinline__ bool NearFiniteCut(float3 point, const CutEventSummary& event)
    {
        const float3 start=make_float3(event.start[0],event.start[1],event.start[2]);
        const float3 end=make_float3(event.end[0],event.end[1],event.end[2]);
        const float3 segment=Sub(end,start);
        const float lengthSquared=Dot(segment,segment);
        const float u=lengthSquared>1e-12f?fminf(1.0f,fmaxf(0.0f,Dot(Sub(point,start),segment)/lengthSquared)):0.0f;
        const float3 closest=Add(start,Mul(segment,u));
        const float extent=fmaxf(event.radius*2.0f,1e-4f);
        return Dot(Sub(point,closest),Sub(point,closest))<=extent*extent;
    }

    __global__ void PrepareCutToTetKernel(const CutEventSummary* event, unsigned int* counters)
    {
        if (blockIdx.x || threadIdx.x) return;
        counters[0]=0;
        if (!event->valid || event->sequence==0 || event->sequence==counters[8]) return;
        for (int i=1;i<=7;++i) counters[i]=0;
        counters[8]=event->sequence;
        counters[9]+=1;
        counters[10]=1;
        counters[0]=1;
    }

    __global__ void ClassifyTetsKernel(const float3* positions, const int* tetIds, const int* active,
                                       int count, const CutEventSummary* event, unsigned int* flags,
                                       unsigned int* counters)
    {
        const int t=blockIdx.x*blockDim.x+threadIdx.x;
        if (t>=count || !counters[0]) return;
        if (!active[t]) { flags[t]=0; return; }
        const float3 n0=make_float3(event->normal[0],event->normal[1],event->normal[2]);
        const float nLen=sqrtf(Dot(n0,n0));
        if (nLen<1e-8f) { flags[t]=0; return; }
        const float3 n=Mul(n0,1.0f/nLen);
        const float3 origin=make_float3(event->hitPoint[0],event->hitPoint[1],event->hitPoint[2]);
        float minD=FLT_MAX,maxD=-FLT_MAX;
        float3 center=make_float3(0,0,0);
        bool near=false;
        for (int k=0;k<4;++k)
        {
            const float3 p=positions[tetIds[t*4+k]];
            const float d=Dot(Sub(p,origin),n);
            minD=fminf(minD,d); maxD=fmaxf(maxD,d); center=Add(center,Mul(p,0.25f));
            near=near||NearFiniteCut(p,*event);
        }
        if (minD>0.0f) atomicAdd(&counters[1],1u);
        else if (maxD<0.0f) atomicAdd(&counters[2],1u);
        else atomicAdd(&counters[3],1u);
        const bool candidate=minD<=event->radius && maxD>=-event->radius && (near||NearFiniteCut(center,*event));
        flags[t]=candidate?event->sequence:0u;
        if (candidate) atomicAdd(&counters[4],1u);
    }

    __device__ __forceinline__ bool PrimitiveCrossesCut(const float3* positions, const int* ids, int vertexCount,
                                                         const CutEventSummary& event)
    {
        const float3 n0=make_float3(event.normal[0],event.normal[1],event.normal[2]);
        const float nLen=sqrtf(Dot(n0,n0));
        if (nLen<1e-8f) return false;
        const float3 n=Mul(n0,1.0f/nLen);
        const float3 origin=make_float3(event.hitPoint[0],event.hitPoint[1],event.hitPoint[2]);
        float minD=FLT_MAX,maxD=-FLT_MAX;
        float3 center=make_float3(0,0,0);
        bool near=false;
        for (int i=0;i<vertexCount;++i)
        {
            const float3 p=positions[ids[i]];
            const float d=Dot(Sub(p,origin),n);
            minD=fminf(minD,d); maxD=fmaxf(maxD,d);
            center=Add(center,Mul(p,1.0f/vertexCount));
            near=near||NearFiniteCut(p,event);
        }
        return minD<=event.radius && maxD>=-event.radius && (near||NearFiniteCut(center,event));
    }

    __global__ void ClassifyEdgesKernel(const float3* positions, const int* edges, int count,
                                        const CutEventSummary* event, unsigned int* flags,
                                        unsigned int* counters)
    {
        const int i=blockIdx.x*blockDim.x+threadIdx.x;
        if (i>=count || !counters[0]) return;
        const int ids[2]={edges[i*2],edges[i*2+1]};
        const bool candidate=PrimitiveCrossesCut(positions,ids,2,*event);
        flags[i]=candidate?event->sequence:0u;
        if (candidate) atomicAdd(&counters[5],1u);
    }

    __global__ void ClassifyFacesKernel(const float3* positions, const int* faces, int stride, int count,
                                        const CutEventSummary* event, unsigned int* flags,
                                        unsigned int* counters, int counterIndex)
    {
        const int i=blockIdx.x*blockDim.x+threadIdx.x;
        if (i>=count || !counters[0]) return;
        const int ids[3]={faces[i*stride],faces[i*stride+1],faces[i*stride+2]};
        const bool candidate=PrimitiveCrossesCut(positions,ids,3,*event);
        flags[i]=candidate?event->sequence:0u;
        if (candidate) atomicAdd(&counters[counterIndex],1u);
    }
}

int organ_context_create(uint32_t* outHandle)
{
    if (!outHandle) return ORGAN_CONTEXT_INVALID_ARGUMENT;
    std::lock_guard<std::mutex> lock(g_contextMutex);
    uint32_t handle=g_nextHandle++; if (!handle) handle=g_nextHandle++;
    auto context=std::make_unique<CudaOrganContext>(); context->stats.handle=handle;
    if (cudaEventCreate(&context->uploadStart)!=cudaSuccess || cudaEventCreate(&context->uploadEnd)!=cudaSuccess ||
        cudaEventCreate(&context->kernelStart)!=cudaSuccess || cudaEventCreate(&context->kernelEnd)!=cudaSuccess ||
        cudaEventCreate(&context->toolStart)!=cudaSuccess || cudaEventCreate(&context->toolEnd)!=cudaSuccess ||
        cudaEventCreate(&context->tetToGridStart)!=cudaSuccess || cudaEventCreate(&context->tetToGridEnd)!=cudaSuccess ||
        cudaEventCreate(&context->cutToTetStart)!=cudaSuccess || cudaEventCreate(&context->cutToTetEnd)!=cudaSuccess)
        return ORGAN_CONTEXT_CUDA_FAILURE;
    g_contexts.emplace(handle,std::move(context)); *outHandle=handle; return ORGAN_CONTEXT_OK;
}

int organ_context_destroy(uint32_t handle)
{
    std::lock_guard<std::mutex> lock(g_contextMutex); auto it=g_contexts.find(handle);
    if (it==g_contexts.end()) return ORGAN_CONTEXT_INVALID_HANDLE; g_contexts.erase(it); return ORGAN_CONTEXT_OK;
}

int organ_context_initialize(uint32_t handle, const OrganContextInitDesc* desc,
                             const float* restPositions3, const int* tetIds4, const float* inverseMass,
                             const float* restVolumes, const int* tetActive, const int* surfaceTriangleIds3,
                             const int* edgeConstraintIds2, const float* edgeRestLengths)
{
    if (!desc || desc->particleCount<=0 || desc->tetCount<=0 || !restPositions3 || !tetIds4 || !inverseMass || !restVolumes || !tetActive)
        return ORGAN_CONTEXT_INVALID_ARGUMENT;
    std::lock_guard<std::mutex> lock(g_contextMutex); CudaOrganContext* c=FindContext(handle);
    if (!c) return ORGAN_CONTEXT_INVALID_HANDLE; if (c->stats.initialized) return ORGAN_CONTEXT_ALREADY_INITIALIZED;
    if ((desc->surfaceTriangleCount>0 && !surfaceTriangleIds3) || (desc->edgeConstraintCount>0 && (!edgeConstraintIds2 || !edgeRestLengths))) return ORGAN_CONTEXT_INVALID_ARGUMENT;
    c->stats.particleCount=desc->particleCount; c->stats.tetCount=desc->tetCount;
    c->fractureStats.originalParticleCount=desc->particleCount;
    c->fractureStats.currentParticleCount=desc->particleCount;
    c->stats.surfaceTriangleCount=desc->surfaceTriangleCount; c->stats.edgeConstraintCount=desc->edgeConstraintCount;
    std::vector<int4> internalFaces=BuildInternalFaces(tetIds4,desc->tetCount);
    c->internalFaceCount=static_cast<int>(internalFaces.size());
    cudaEventRecord(c->uploadStart); float material[4]={desc->density,desc->youngsModulus,desc->poissonsRatio,desc->damping};
    bool ok=Upload(c->restPositions,restPositions3,size_t(desc->particleCount)*3,*c) &&
            Upload(c->tetIds,tetIds4,size_t(desc->tetCount)*4,*c) && Upload(c->inverseMass,inverseMass,desc->particleCount,*c) &&
            Upload(c->restVolumes,restVolumes,desc->tetCount,*c) && Upload(c->tetActive,tetActive,desc->tetCount,*c) &&
            Upload(c->surfaceTriangles,surfaceTriangleIds3,size_t(desc->surfaceTriangleCount)*3,*c) &&
            Upload(c->edgeIds,edgeConstraintIds2,size_t(desc->edgeConstraintCount)*2,*c) &&
            Upload(c->edgeRestLengths,edgeRestLengths,desc->edgeConstraintCount,*c) && Upload(c->material,material,4,*c) &&
            Upload(c->internalFaces,internalFaces.data(),internalFaces.size(),*c);
    cudaEventRecord(c->uploadEnd); cudaEventSynchronize(c->uploadEnd); cudaEventElapsedTime(&c->stats.uploadMilliseconds,c->uploadStart,c->uploadEnd);
    if (!ok) return ORGAN_CONTEXT_CUDA_FAILURE;
    unsigned long long* sink=nullptr; if (!Check(cudaMalloc(&sink,sizeof(unsigned long long)),*c) || !Check(cudaMemset(sink,0,sizeof(unsigned long long)),*c)) return ORGAN_CONTEXT_CUDA_FAILURE;
    cudaEventRecord(c->kernelStart); ValidateRestStateKernel<<<(desc->particleCount+kThreads-1)/kThreads,kThreads>>>(c->restPositions,desc->particleCount,sink);
    cudaEventRecord(c->kernelEnd); cudaEventSynchronize(c->kernelEnd); cudaEventElapsedTime(&c->stats.validationKernelMilliseconds,c->kernelStart,c->kernelEnd); cudaFree(sink);
    if (!Check(cudaGetLastError(),*c)) return ORGAN_CONTEXT_CUDA_FAILURE;
    uint64_t rest=1469598103934665603ull; rest=HashBytes(rest,restPositions3,sizeof(float)*size_t(desc->particleCount)*3);
    uint64_t top=1469598103934665603ull; top=HashBytes(top,tetIds4,sizeof(int)*size_t(desc->tetCount)*4); top=HashBytes(top,surfaceTriangleIds3,sizeof(int)*size_t(desc->surfaceTriangleCount)*3); top=HashBytes(top,edgeConstraintIds2,sizeof(int)*size_t(desc->edgeConstraintCount)*2);
    c->stats.restPositionHash=rest; c->stats.topologyHash=top; c->stats.hostFingerprint=HashBytes(top,&c->stats.deviceBytes,sizeof(c->stats.deviceBytes)); c->stats.initialized=1; c->stats.lastError=0;
    return ORGAN_CONTEXT_OK;
}

int organ_context_xpbd_initialize(uint32_t handle, const float* initialPositions3,
                                  int edgeColorCount, const int* edgeColorOffsets, const int* edgeColorCounts, const int* edgeColorFlat,
                                  int tetColorCount, const int* tetColorOffsets, const int* tetColorCounts, const int* tetColorFlat,
                                  int surfaceColorCount, const int* surfaceColorOffsets, const int* surfaceColorCounts, const int* surfaceColorFlat)
{
    if (!initialPositions3 || edgeColorCount<=0 || tetColorCount<=0 || surfaceColorCount<=0 ||
        !edgeColorOffsets || !edgeColorCounts || !edgeColorFlat || !tetColorOffsets || !tetColorCounts || !tetColorFlat ||
        !surfaceColorOffsets || !surfaceColorCounts || !surfaceColorFlat) return ORGAN_CONTEXT_INVALID_ARGUMENT;
    std::lock_guard<std::mutex> lock(g_contextMutex); CudaOrganContext* c=FindContext(handle);
    if (!c) return ORGAN_CONTEXT_INVALID_HANDLE; if (!c->stats.initialized) return ORGAN_CONTEXT_NOT_INITIALIZED; if (c->xpbdInitialized) return ORGAN_CONTEXT_ALREADY_INITIALIZED;
    int edgeFlatCount=edgeColorOffsets[edgeColorCount-1]+edgeColorCounts[edgeColorCount-1];
    int tetFlatCount=tetColorOffsets[tetColorCount-1]+tetColorCounts[tetColorCount-1];
    int surfaceFlatCount=surfaceColorOffsets[surfaceColorCount-1]+surfaceColorCounts[surfaceColorCount-1];
    if (edgeFlatCount!=c->stats.edgeConstraintCount || tetFlatCount!=c->stats.tetCount || surfaceFlatCount!=c->stats.surfaceTriangleCount) return ORGAN_CONTEXT_INVALID_ARGUMENT;
    bool ok=Allocate(c->positions,c->stats.particleCount,*c) && Allocate(c->previousPositions,c->stats.particleCount,*c) && Allocate(c->velocities,c->stats.particleCount,*c) &&
            Allocate(c->invRestRows,size_t(c->stats.tetCount)*3,*c) && Allocate(c->alphaDeviatoric,c->stats.tetCount,*c) && Allocate(c->alphaHydrostatic,c->stats.tetCount,*c) &&
            Allocate(c->toolCapsules,12,*c) && Allocate(c->graspMask,c->stats.particleCount,*c) &&
            Allocate(c->graspFrameCoordinates,c->stats.particleCount,*c) && Allocate(c->graspWeights,c->stats.particleCount,*c) &&
            Allocate(c->toolCounters,7,*c) &&
            Allocate(c->cutEventSummary,1,*c) && Allocate(c->cutToTetCounters,11,*c) &&
            Allocate(c->candidateTetSequence,c->stats.tetCount,*c) &&
            Allocate(c->candidateEdgeSequence,c->stats.edgeConstraintCount,*c) &&
            Allocate(c->candidateSharedFaceSequence,c->internalFaceCount,*c) &&
            Allocate(c->candidateSurfaceFaceSequence,c->stats.surfaceTriangleCount,*c) &&
            Upload(c->edgeColorOffsets,edgeColorOffsets,edgeColorCount,*c,true) && Upload(c->edgeColorCounts,edgeColorCounts,edgeColorCount,*c,true) && Upload(c->edgeColorFlat,edgeColorFlat,edgeFlatCount,*c,true) &&
            Upload(c->tetColorOffsets,tetColorOffsets,tetColorCount,*c,true) && Upload(c->tetColorCounts,tetColorCounts,tetColorCount,*c,true) && Upload(c->tetColorFlat,tetColorFlat,tetFlatCount,*c,true) &&
            Upload(c->surfaceColorOffsets,surfaceColorOffsets,surfaceColorCount,*c,true) && Upload(c->surfaceColorCounts,surfaceColorCounts,surfaceColorCount,*c,true) && Upload(c->surfaceColorFlat,surfaceColorFlat,surfaceFlatCount,*c,true);
    if (!ok || !Check(cudaMemcpy(c->positions,initialPositions3,sizeof(float3)*size_t(c->stats.particleCount),cudaMemcpyHostToDevice),*c) ||
        !Check(cudaMemcpy(c->previousPositions,initialPositions3,sizeof(float3)*size_t(c->stats.particleCount),cudaMemcpyHostToDevice),*c) ||
        !Check(cudaMemset(c->velocities,0,sizeof(float3)*size_t(c->stats.particleCount)),*c) ||
        !Check(cudaMemset(c->graspMask,0,c->stats.particleCount),*c) ||
        !Check(cudaMemset(c->toolCounters,0,sizeof(unsigned int)*7),*c) ||
        !Check(cudaMemset(c->cutEventSummary,0,sizeof(CutEventSummary)),*c) ||
        !Check(cudaMemset(c->cutToTetCounters,0,sizeof(unsigned int)*11),*c) ||
        !Check(cudaMemset(c->candidateTetSequence,0,sizeof(unsigned int)*size_t(c->stats.tetCount)),*c) ||
        (c->stats.edgeConstraintCount>0 && !Check(cudaMemset(c->candidateEdgeSequence,0,sizeof(unsigned int)*size_t(c->stats.edgeConstraintCount)),*c)) ||
        (c->internalFaceCount>0 && !Check(cudaMemset(c->candidateSharedFaceSequence,0,sizeof(unsigned int)*size_t(c->internalFaceCount)),*c)) ||
        (c->stats.surfaceTriangleCount>0 && !Check(cudaMemset(c->candidateSurfaceFaceSequence,0,sizeof(unsigned int)*size_t(c->stats.surfaceTriangleCount)),*c))) return ORGAN_CONTEXT_CUDA_FAILURE;
    PrepareTetRestKernel<<<(c->stats.tetCount+kThreads-1)/kThreads,kThreads>>>(reinterpret_cast<float3*>(c->restPositions),c->tetIds,c->tetActive,c->stats.tetCount,c->invRestRows);
    if (!Check(cudaDeviceSynchronize(),*c)) return ORGAN_CONTEXT_CUDA_FAILURE;
    c->hostEdgeColorOffsets.assign(edgeColorOffsets, edgeColorOffsets + edgeColorCount);
    c->hostEdgeColorCounts.assign(edgeColorCounts, edgeColorCounts + edgeColorCount);
    c->hostTetColorOffsets.assign(tetColorOffsets, tetColorOffsets + tetColorCount);
    c->hostTetColorCounts.assign(tetColorCounts, tetColorCounts + tetColorCount);
    c->hostSurfaceColorOffsets.assign(surfaceColorOffsets, surfaceColorOffsets + surfaceColorCount);
    c->hostSurfaceColorCounts.assign(surfaceColorCounts, surfaceColorCounts + surfaceColorCount);
    c->edgeColorCount=edgeColorCount; c->tetColorCount=tetColorCount; c->surfaceColorCount=surfaceColorCount;
    c->xpbdInitialized=1; c->stats.lastError=0; return ORGAN_CONTEXT_OK;
}

int organ_context_xpbd_step(uint32_t handle, float dt, const OrganContextXpbdParams* p)
{
    if (!p || !std::isfinite(dt) || dt<=0.0f || p->numSubSteps<=0) return ORGAN_CONTEXT_INVALID_ARGUMENT;
    std::lock_guard<std::mutex> lock(g_contextMutex); CudaOrganContext* c=FindContext(handle);
    if (!c) return ORGAN_CONTEXT_INVALID_HANDLE; if (!c->xpbdInitialized) return ORGAN_CONTEXT_NOT_INITIALIZED;
    float h=dt/p->numSubSteps; if (!std::isfinite(h) || h<=0.0f) return ORGAN_CONTEXT_INVALID_ARGUMENT;
    float nu=fminf(fmaxf(p->poissonsRatio,0.01f),0.499f), E=fmaxf(p->youngsModulus,1.0f);
    float mu=E/(2.0f*(1.0f+nu)), lambda=E*nu/((1.0f+nu)*(1.0f-2.0f*nu));
    UpdateTetAlphaKernel<<<(c->stats.tetCount+kThreads-1)/kThreads,kThreads>>>(c->restVolumes,c->tetActive,c->stats.tetCount,mu,lambda,1.0f/(h*h),c->alphaDeviatoric,c->alphaHydrostatic);
    if (c->toolConfigured && !Check(cudaMemset(c->toolCounters,0,sizeof(unsigned int)*7),*c))
        return ORGAN_CONTEXT_CUDA_FAILURE;
    cudaEventRecord(c->kernelStart);
    int iterations=p->constraintIterations>0?p->constraintIterations:1, particleBlocks=(c->stats.particleCount+kThreads-1)/kThreads;
    for (int sub=0;sub<p->numSubSteps;++sub)
    {
        IntegrateKernel<<<particleBlocks,kThreads>>>(c->positions,c->previousPositions,c->velocities,c->inverseMass,c->stats.particleCount,h,make_float3(p->gravityX,p->gravityY,p->gravityZ));
        auto solveInternal = [&]()
        {
            for (int iteration=0;iteration<iterations;++iteration)
            {
                for (int color=0;color<c->edgeColorCount;++color) { int count=c->hostEdgeColorCounts[color], offset=c->hostEdgeColorOffsets[color]; if (count>0) SolveEdgesKernel<<<(count+kThreads-1)/kThreads,kThreads>>>(c->positions,c->inverseMass,c->edgeIds,c->edgeRestLengths,c->edgeColorFlat,offset,count,p->edgeCompliance/(h*h)); }
                for (int color=0;color<c->tetColorCount;++color) { int count=c->hostTetColorCounts[color], offset=c->hostTetColorOffsets[color]; if (count>0) SolveNeoHookeanKernel<<<(count+kThreads-1)/kThreads,kThreads>>>(c->positions,c->inverseMass,c->tetIds,c->tetActive,c->invRestRows,c->alphaDeviatoric,c->alphaHydrostatic,c->tetColorFlat,offset,count); }
            }
        };
        auto solveSurfaceContact = [&]()
        {
            const int contactIterations = c->toolParams.contactIterations > 0 ? c->toolParams.contactIterations : 1;
            const float contactAlpha = fmaxf(c->toolParams.contactCompliance, 0.0f) / (h * h);
            for (int iteration=0; iteration<contactIterations; ++iteration)
                for (int color=0; color<c->surfaceColorCount; ++color)
                {
                    int count=c->hostSurfaceColorCounts[color], offset=c->hostSurfaceColorOffsets[color];
                    if (count>0) SurfaceToolContactKernel<<<(count+kThreads-1)/kThreads,kThreads>>>(
                        c->positions,c->previousPositions,c->inverseMass,c->surfaceTriangles,c->surfaceColorFlat,
                        offset,count,c->toolCapsules,c->toolParams,contactAlpha,c->toolCounters);
                }
        };
        solveInternal();
        if (c->toolConfigured && c->toolParams.contactEnabled)
        {
            const int couplingPasses = c->toolParams.couplingPasses > 0 ? c->toolParams.couplingPasses : 1;
            solveSurfaceContact();
            for (int pass=0; pass<couplingPasses; ++pass) { solveInternal(); solveSurfaceContact(); }
        }
        if (c->toolGraspActive)
        {
            c->toolGraspBlend = c->toolParams.graspFormDuration > 0.0f
                ? fminf(1.0f, c->toolGraspBlend + h / c->toolParams.graspFormDuration) : 1.0f;
            SolveGraspKernel<<<particleBlocks,kThreads>>>(c->positions,c->previousPositions,c->velocities,c->inverseMass,
                c->stats.particleCount,c->graspMask,c->graspFrameCoordinates,c->graspWeights,
                c->toolParams,c->graspCaptureParams,c->toolGraspBlend,h);
        }
        GroundKernel<<<particleBlocks,kThreads>>>(c->positions,c->previousPositions,c->stats.particleCount,p->groundY);
        PostSolveKernel<<<particleBlocks,kThreads>>>(c->positions,c->previousPositions,c->velocities,c->inverseMass,c->stats.particleCount,1.0f/h,fminf(fmaxf(p->damping,0.0f),1.0f));
    }
    if (c->toolConfigured)
        CountGraspKernel<<<particleBlocks,kThreads>>>(c->graspMask,c->stats.particleCount,c->toolCounters);
    cudaEventRecord(c->kernelEnd);
    cudaError_t err=cudaPeekAtLastError(); if (!Check(err,*c)) return ORGAN_CONTEXT_CUDA_FAILURE;
    if (cudaEventQuery(c->kernelEnd)==cudaSuccess) { cudaEventElapsedTime(&c->stats.lastXpbdMilliseconds,c->kernelStart,c->kernelEnd); c->stats.totalXpbdMilliseconds+=c->stats.lastXpbdMilliseconds; }
    ++c->stats.xpbdStepCount; c->stats.lastError=0; return ORGAN_CONTEXT_OK;
}

int organ_context_tool_step(uint32_t handle, float dt, const OrganContextToolCapsule* capsules,
                            const OrganContextToolContactParams* params)
{
    if (!params || !std::isfinite(dt) || dt <= 0.0f || params->capsuleCount < 0 || params->capsuleCount > 12 ||
        (params->capsuleCount > 0 && !capsules)) return ORGAN_CONTEXT_INVALID_ARGUMENT;
    std::lock_guard<std::mutex> lock(g_contextMutex); CudaOrganContext* c=FindContext(handle);
    if (!c) return ORGAN_CONTEXT_INVALID_HANDLE; if (!c->xpbdInitialized) return ORGAN_CONTEXT_NOT_INITIALIZED;
    cudaEventRecord(c->toolStart);
    c->toolParams=*params;
    c->toolCapsuleCount=params->capsuleCount;
    c->toolConfigured=1;
    c->toolStats.toolUploadBytes += sizeof(OrganContextToolContactParams) + sizeof(OrganContextToolCapsule)*size_t(params->capsuleCount);
    c->toolStats.toolUploadOperations += 1;
    if (params->capsuleCount > 0 && !Check(cudaMemcpy(c->toolCapsules,capsules,sizeof(OrganContextToolCapsule)*size_t(params->capsuleCount),cudaMemcpyHostToDevice),*c))
        return ORGAN_CONTEXT_CUDA_FAILURE;
    const int blocks=(c->stats.particleCount+kThreads-1)/kThreads;
    if (params->releaseRequest && c->toolGraspActive)
    {
        ReleaseGraspKernel<<<blocks,kThreads>>>(c->velocities,c->stats.particleCount,c->graspMask);
        c->toolGraspActive=0;
        c->toolGraspBlend=0.0f;
    }
    if (params->graspRequest && !c->toolGraspActive)
    {
        CaptureGraspKernel<<<blocks,kThreads>>>(c->positions,c->stats.particleCount,c->graspMask,
            c->graspFrameCoordinates,c->graspWeights,*params);
        c->graspCaptureParams=*params;
        c->toolGraspActive=1;
        c->toolGraspBlend=0.0f;
    }
    cudaEventRecord(c->toolEnd);
    cudaError_t error=cudaPeekAtLastError(); if (!Check(error,*c)) return ORGAN_CONTEXT_CUDA_FAILURE;
    if (cudaEventQuery(c->toolEnd)==cudaSuccess)
    {
        cudaEventElapsedTime(&c->toolStats.lastToolMilliseconds,c->toolStart,c->toolEnd);
        c->toolStats.totalToolMilliseconds+=c->toolStats.lastToolMilliseconds;
    }
    c->toolStats.dispatchCount += 1;
    c->toolStats.lastToolError=0;
    return ORGAN_CONTEXT_OK;
}

int organ_context_tool_get_stats(uint32_t handle, OrganContextToolContactStats* outStats)
{
    if (!outStats) return ORGAN_CONTEXT_INVALID_ARGUMENT;
    std::lock_guard<std::mutex> lock(g_contextMutex); CudaOrganContext* c=FindContext(handle);
    if (!c) return ORGAN_CONTEXT_INVALID_HANDLE; if (!c->xpbdInitialized) return ORGAN_CONTEXT_NOT_INITIALIZED;
    unsigned int counters[7]{};
    if (!Check(cudaMemcpy(counters,c->toolCounters,sizeof(counters),cudaMemcpyDeviceToHost),*c)) return ORGAN_CONTEXT_CUDA_FAILURE;
    c->toolStats.activeCandidates=static_cast<int>(counters[0]);
    c->toolStats.contactCount=static_cast<int>(counters[1]);
    float maxContactDepth = 0.0f;
    std::memcpy(&maxContactDepth, &counters[2], sizeof(float));
    c->toolStats.maxContactDepth=maxContactDepth;
    c->toolStats.surfaceCandidateTriangles=static_cast<int>(counters[4]);
    c->toolStats.surfaceContactTriangles=static_cast<int>(counters[5]);
    float surfaceMaxContactDepth = 0.0f;
    std::memcpy(&surfaceMaxContactDepth, &counters[6], sizeof(float));
    c->toolStats.surfaceMaxContactDepth=surfaceMaxContactDepth;
    c->toolStats.graspedParticleCount=static_cast<int>(counters[3]);
    *outStats=c->toolStats;
    return ORGAN_CONTEXT_OK;
}

int organ_context_tet_to_grid_configure(uint32_t handle, const OrganContextTetToGridDesc* desc,
                                        const int* hostTetByCorner, const float* barycentricWeights4,
                                        const float* alignedRestCorners3, const unsigned char* activeMask)
{
    if (!desc || desc->cornerCount <= 0 || !hostTetByCorner || !barycentricWeights4 || !alignedRestCorners3 || !activeMask)
        return ORGAN_CONTEXT_INVALID_ARGUMENT;
    std::lock_guard<std::mutex> lock(g_contextMutex); CudaOrganContext* c=FindContext(handle);
    if (!c) return ORGAN_CONTEXT_INVALID_HANDLE; if (!c->xpbdInitialized) return ORGAN_CONTEXT_NOT_INITIALIZED;
    if (desc->cornerCount != recon_corner_count()) return ORGAN_CONTEXT_INVALID_ARGUMENT;
    if (c->tetToGridStats.configured) return ORGAN_CONTEXT_ALREADY_INITIALIZED;
    const int count=desc->cornerCount;
    if (!Upload(c->hostTetByCorner, hostTetByCorner, count, *c, true) ||
        !Upload(c->barycentricWeights, reinterpret_cast<const float4*>(barycentricWeights4), count, *c, true) ||
        !Upload(c->alignedRestCorners, reinterpret_cast<const float3*>(alignedRestCorners3), count, *c, true) ||
        !Upload(c->gridActiveMask, activeMask, count, *c, true) || !Allocate(c->currentCentroid, 1, *c))
        return ORGAN_CONTEXT_CUDA_FAILURE;
    int mapped=0; for (int i=0;i<count;++i) if (activeMask[i] && hostTetByCorner[i]>=0) ++mapped;
    c->tetToGridStats.configured=1; c->tetToGridStats.cornerCount=count; c->tetToGridStats.mappedCorners=mapped;
    c->tetToGridStats.fallbackCorners=count-mapped;
    c->tetToGridStats.uploadBytes=sizeof(int)*size_t(count)+sizeof(float4)*size_t(count)+sizeof(float3)*size_t(count)+sizeof(unsigned char)*size_t(count);
    c->tetToGridStats.lastError=0;
    c->tetRestCentroid=make_float3(desc->restCentroidX,desc->restCentroidY,desc->restCentroidZ);
    return ORGAN_CONTEXT_OK;
}

int organ_context_tet_to_grid_update(uint32_t handle, int applyLocalDeformation)
{
    std::lock_guard<std::mutex> lock(g_contextMutex); CudaOrganContext* c=FindContext(handle);
    if (!c) return ORGAN_CONTEXT_INVALID_HANDLE;
    if (!c->xpbdInitialized || !c->tetToGridStats.configured) return ORGAN_CONTEXT_NOT_INITIALIZED;
    float3* cornerPos=recon_corner_pos();
    if (!cornerPos) return ORGAN_CONTEXT_NOT_INITIALIZED;
    cudaEventRecord(c->tetToGridStart);
    if (!Check(cudaMemset(c->currentCentroid,0,sizeof(float3)),*c)) return ORGAN_CONTEXT_CUDA_FAILURE;
    const int centroidParticleCount=c->fractureStats.applied
        ? c->fractureStats.originalParticleCount : c->stats.particleCount;
    const int particleBlocks=(centroidParticleCount+kThreads-1)/kThreads;
    SumCentroidKernel<<<particleBlocks,kThreads>>>(c->positions,centroidParticleCount,c->currentCentroid);
    const int cornerBlocks=(c->tetToGridStats.cornerCount+kThreads-1)/kThreads;
    TetToGridKernel<<<cornerBlocks,kThreads>>>(c->positions,reinterpret_cast<float3*>(c->restPositions),c->tetIds,
        c->hostTetByCorner,c->barycentricWeights,c->alignedRestCorners,c->gridActiveMask,c->tetToGridStats.cornerCount,
        c->tetRestCentroid,c->currentCentroid,applyLocalDeformation ? 1 : 0,cornerPos);
    cudaEventRecord(c->tetToGridEnd);
    if (!Check(cudaPeekAtLastError(),*c)) return ORGAN_CONTEXT_CUDA_FAILURE;
    if (cudaEventQuery(c->tetToGridEnd)==cudaSuccess)
    {
        cudaEventElapsedTime(&c->tetToGridStats.lastUpdateMilliseconds,c->tetToGridStart,c->tetToGridEnd);
        c->tetToGridStats.totalUpdateMilliseconds+=c->tetToGridStats.lastUpdateMilliseconds;
    }
    ++c->tetToGridStats.updateCount; c->tetToGridStats.lastError=0; return ORGAN_CONTEXT_OK;
}

int organ_context_tet_to_grid_get_stats(uint32_t handle, OrganContextTetToGridStats* outStats)
{
    if (!outStats) return ORGAN_CONTEXT_INVALID_ARGUMENT;
    std::lock_guard<std::mutex> lock(g_contextMutex); CudaOrganContext* c=FindContext(handle);
    if (!c) return ORGAN_CONTEXT_INVALID_HANDLE; *outStats=c->tetToGridStats; return ORGAN_CONTEXT_OK;
}

int organ_context_cut_to_tet_set_enabled(uint32_t handle, int enabled)
{
    std::lock_guard<std::mutex> lock(g_contextMutex); CudaOrganContext* c=FindContext(handle);
    if (!c) return ORGAN_CONTEXT_INVALID_HANDLE;
    if (!c->xpbdInitialized) return ORGAN_CONTEXT_NOT_INITIALIZED;
    c->cutToTetStats.enabled=enabled?1:0;
    c->cutToTetStats.lastError=0;
    return ORGAN_CONTEXT_OK;
}

int organ_context_cut_to_tet_classify_latest_all()
{
    std::lock_guard<std::mutex> lock(g_contextMutex);
    int overallResult=ORGAN_CONTEXT_OK;
    for (auto& entry:g_contexts)
    {
        CudaOrganContext& c=*entry.second;
        if (!c.xpbdInitialized || !c.cutToTetStats.enabled) continue;
        int copyResult=cut_copy_event_summary_to_device(c.cutEventSummary);
        if (copyResult!=0) { c.cutToTetStats.lastError=copyResult; overallResult=copyResult; continue; }
        cudaEventRecord(c.cutToTetStart);
        PrepareCutToTetKernel<<<1,1>>>(c.cutEventSummary,c.cutToTetCounters);
        ClassifyTetsKernel<<<(c.stats.tetCount+kThreads-1)/kThreads,kThreads>>>(
            c.positions,c.tetIds,c.tetActive,c.stats.tetCount,c.cutEventSummary,
            c.candidateTetSequence,c.cutToTetCounters);
        if (c.stats.edgeConstraintCount>0)
            ClassifyEdgesKernel<<<(c.stats.edgeConstraintCount+kThreads-1)/kThreads,kThreads>>>(
                c.positions,c.edgeIds,c.stats.edgeConstraintCount,c.cutEventSummary,
                c.candidateEdgeSequence,c.cutToTetCounters);
        if (c.internalFaceCount>0)
            ClassifyFacesKernel<<<(c.internalFaceCount+kThreads-1)/kThreads,kThreads>>>(
                c.positions,reinterpret_cast<const int*>(c.internalFaces),4,c.internalFaceCount,
                c.cutEventSummary,c.candidateSharedFaceSequence,c.cutToTetCounters,6);
        if (c.stats.surfaceTriangleCount>0)
            ClassifyFacesKernel<<<(c.stats.surfaceTriangleCount+kThreads-1)/kThreads,kThreads>>>(
                c.positions,c.surfaceTriangles,3,c.stats.surfaceTriangleCount,
                c.cutEventSummary,c.candidateSurfaceFaceSequence,c.cutToTetCounters,7);
        cudaEventRecord(c.cutToTetEnd);
        if (!Check(cudaPeekAtLastError(),c))
        {
            c.cutToTetStats.lastError=ORGAN_CONTEXT_CUDA_FAILURE;
            overallResult=ORGAN_CONTEXT_CUDA_FAILURE;
            continue;
        }
        if (c.fractureStats.enabled && !c.fractureStats.applied)
        {
            unsigned int counters[11]{};
            CutEventSummary event{};
            if (cudaMemcpy(counters,c.cutToTetCounters,sizeof(counters),cudaMemcpyDeviceToHost)!=cudaSuccess ||
                cudaMemcpy(&event,c.cutEventSummary,sizeof(event),cudaMemcpyDeviceToHost)!=cudaSuccess)
            {
                c.fractureStats.lastError=ORGAN_CONTEXT_CUDA_FAILURE;
                c.fractureStats.rejectedReason=ORGAN_FRACTURE_REJECT_TOPOLOGY_REBUILD;
                overallResult=ORGAN_CONTEXT_CUDA_FAILURE;
            }
            else if (counters[0] && event.sequence!=c.fractureStats.processedEventSequence)
                ApplyTetFracture(c,event,counters);
        }
        c.cutToTetStats.lastError=0;
    }
    return overallResult;
}

int organ_context_cut_to_tet_get_stats(uint32_t handle, OrganContextCutToTetStats* outStats)
{
    if (!outStats) return ORGAN_CONTEXT_INVALID_ARGUMENT;
    std::lock_guard<std::mutex> lock(g_contextMutex); CudaOrganContext* c=FindContext(handle);
    if (!c) return ORGAN_CONTEXT_INVALID_HANDLE;
    if (!c->xpbdInitialized) return ORGAN_CONTEXT_NOT_INITIALIZED;
    unsigned int counters[11]{};
    if (!Check(cudaMemcpy(counters,c->cutToTetCounters,sizeof(counters),cudaMemcpyDeviceToHost),*c))
    {
        c->cutToTetStats.lastError=ORGAN_CONTEXT_CUDA_FAILURE;
        return ORGAN_CONTEXT_CUDA_FAILURE;
    }
    c->cutToTetStats.classificationValid=static_cast<int>(counters[10]);
    c->cutToTetStats.processedEventSequence=counters[8];
    c->cutToTetStats.classificationCount=counters[9];
    c->cutToTetStats.positiveSideTetCount=static_cast<int>(counters[1]);
    c->cutToTetStats.negativeSideTetCount=static_cast<int>(counters[2]);
    c->cutToTetStats.straddlingTetCount=static_cast<int>(counters[3]);
    c->cutToTetStats.candidateTetCount=static_cast<int>(counters[4]);
    c->cutToTetStats.candidateEdgeConstraintCount=static_cast<int>(counters[5]);
    c->cutToTetStats.candidateSharedFaceCount=static_cast<int>(counters[6]);
    c->cutToTetStats.candidateSurfaceFaceCount=static_cast<int>(counters[7]);
    if (counters[9]!=c->reportedClassificationCount && cudaEventSynchronize(c->cutToTetEnd)==cudaSuccess)
    {
        float elapsed=0.0f;
        if (cudaEventElapsedTime(&elapsed,c->cutToTetStart,c->cutToTetEnd)==cudaSuccess)
        {
            c->cutToTetStats.lastClassificationMilliseconds=elapsed;
            c->cutToTetStats.totalClassificationMilliseconds+=elapsed;
        }
        c->reportedClassificationCount=counters[9];
    }
    *outStats=c->cutToTetStats;
    return ORGAN_CONTEXT_OK;
}

int organ_context_tet_fracture_configure(uint32_t handle, int enabled, int runtimeCompatible,
                                         float initialGap)
{
    if (!std::isfinite(initialGap) || initialGap<0.0f) return ORGAN_CONTEXT_INVALID_ARGUMENT;
    std::lock_guard<std::mutex> lock(g_contextMutex); CudaOrganContext* c=FindContext(handle);
    if (!c) return ORGAN_CONTEXT_INVALID_HANDLE;
    if (!c->xpbdInitialized) return ORGAN_CONTEXT_NOT_INITIALIZED;
    c->fractureStats.enabled=enabled?1:0;
    c->fractureStats.runtimeCompatible=runtimeCompatible?1:0;
    c->fractureStats.initialGap=initialGap;
    c->fractureStats.lastError=0;
    if (enabled && !runtimeCompatible)
        c->fractureStats.rejectedReason=ORGAN_FRACTURE_REJECT_RUNTIME_PATH;
    else if (!c->fractureStats.applied)
        c->fractureStats.rejectedReason=ORGAN_FRACTURE_REJECT_NONE;
    return ORGAN_CONTEXT_OK;
}

int organ_context_tet_fracture_get_stats(uint32_t handle, OrganContextTetFractureStats* outStats)
{
    if (!outStats) return ORGAN_CONTEXT_INVALID_ARGUMENT;
    std::lock_guard<std::mutex> lock(g_contextMutex); CudaOrganContext* c=FindContext(handle);
    if (!c) return ORGAN_CONTEXT_INVALID_HANDLE;
    if (!c->xpbdInitialized) return ORGAN_CONTEXT_NOT_INITIALIZED;
    *outStats=c->fractureStats;
    return ORGAN_CONTEXT_OK;
}

int organ_context_xpbd_get_positions(uint32_t handle, float* outPositions3, int particleCount)
{
    if (!outPositions3) return ORGAN_CONTEXT_INVALID_ARGUMENT;
    std::lock_guard<std::mutex> lock(g_contextMutex); CudaOrganContext* c=FindContext(handle);
    if (!c) return ORGAN_CONTEXT_INVALID_HANDLE; if (!c->xpbdInitialized) return ORGAN_CONTEXT_NOT_INITIALIZED; if (particleCount!=c->stats.particleCount) return ORGAN_CONTEXT_INVALID_ARGUMENT;
    return Check(cudaMemcpy(outPositions3,c->positions,sizeof(float3)*size_t(particleCount),cudaMemcpyDeviceToHost),*c)?ORGAN_CONTEXT_OK:ORGAN_CONTEXT_CUDA_FAILURE;
}

int organ_context_xpbd_compare_positions(uint32_t handle, const float* unityPositions3, int particleCount, OrganContextComparisonStats* out)
{
    if (!unityPositions3 || !out) return ORGAN_CONTEXT_INVALID_ARGUMENT;
    std::lock_guard<std::mutex> lock(g_contextMutex); CudaOrganContext* c=FindContext(handle);
    if (!c) return ORGAN_CONTEXT_INVALID_HANDLE; if (!c->xpbdInitialized || particleCount!=c->stats.particleCount) return ORGAN_CONTEXT_INVALID_ARGUMENT;
    // Readback is deliberately limited to the explicit, low-frequency compare mode. The CUDA
    // driver mode never calls this API and keeps its dynamic state on the device.
    std::vector<float3> cudaPositions(static_cast<size_t>(particleCount));
    if (!Check(cudaMemcpy(cudaPositions.data(), c->positions, sizeof(float3) * cudaPositions.size(), cudaMemcpyDeviceToHost), *c))
        return ORGAN_CONTEXT_CUDA_FAILURE;
    float3 cudaSum=make_float3(0,0,0), unitySum=make_float3(0,0,0);
    float3 cudaMin=make_float3(FLT_MAX,FLT_MAX,FLT_MAX), cudaMax=make_float3(-FLT_MAX,-FLT_MAX,-FLT_MAX);
    float3 unityMin=cudaMin, unityMax=cudaMax;
    float maxSquared=0.0f, sumSquared=0.0f; int nanCount=0;
    for (int i=0;i<particleCount;++i)
    {
        const float3 a=cudaPositions[i]; const float3 b=make_float3(unityPositions3[i*3],unityPositions3[i*3+1],unityPositions3[i*3+2]);
        if (!Finite3(a)) { ++nanCount; continue; }
        const float3 d=Sub(a,b); const float squared=Dot(d,d); maxSquared=fmaxf(maxSquared,squared); sumSquared+=squared;
        cudaSum=Add(cudaSum,a); unitySum=Add(unitySum,b);
        cudaMin.x=fminf(cudaMin.x,a.x); cudaMin.y=fminf(cudaMin.y,a.y); cudaMin.z=fminf(cudaMin.z,a.z);
        cudaMax.x=fmaxf(cudaMax.x,a.x); cudaMax.y=fmaxf(cudaMax.y,a.y); cudaMax.z=fmaxf(cudaMax.z,a.z);
        unityMin.x=fminf(unityMin.x,b.x); unityMin.y=fminf(unityMin.y,b.y); unityMin.z=fminf(unityMin.z,b.z);
        unityMax.x=fmaxf(unityMax.x,b.x); unityMax.y=fmaxf(unityMax.y,b.y); unityMax.z=fmaxf(unityMax.z,b.z);
    }
    out->particleCount=particleCount; out->cudaNanCount=nanCount; out->maxError=sqrtf(maxSquared); out->rmsError=sqrtf(sumSquared/particleCount);
    out->centroidError=sqrtf(Dot(Sub(Mul(cudaSum,1.0f/particleCount),Mul(unitySum,1.0f/particleCount)),Sub(Mul(cudaSum,1.0f/particleCount),Mul(unitySum,1.0f/particleCount))));
    out->bboxMinError=sqrtf(Dot(Sub(cudaMin,unityMin),Sub(cudaMin,unityMin))); out->bboxMaxError=sqrtf(Dot(Sub(cudaMax,unityMax),Sub(cudaMax,unityMax)));
    const float3 extentDiff=Sub(Sub(cudaMax,cudaMin),Sub(unityMax,unityMin)); out->bboxExtentError=sqrtf(Dot(extentDiff,extentDiff));
    c->stats.lastNanCount=nanCount; c->stats.lastError=0; return ORGAN_CONTEXT_OK;
}

int organ_context_get_stats(uint32_t handle, OrganContextStats* outStats)
{
    if (!outStats) return ORGAN_CONTEXT_INVALID_ARGUMENT; std::lock_guard<std::mutex> lock(g_contextMutex); CudaOrganContext* c=FindContext(handle);
    if (!c) return ORGAN_CONTEXT_INVALID_HANDLE; *outStats=c->stats; return ORGAN_CONTEXT_OK;
}
