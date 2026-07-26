// common.cuh — shared CUDA structs + math helpers for the native DC reconstruction.
// Stage 1 (Preprocess marshal + DC reconstruction, static).
//
// The structs here are BYTE-IDENTICAL to the C# ReconBuffers structs (ReconBuffers.cs) and the
// HLSL structs (ReconCommon.hlsl) — a stride mismatch silently corrupts the upload, so the layouts
// are pinned and asserted at compile time (static_assert below).
//
// Math helpers (BoxSdf, VoxId, NORMAL_SCALE) mirror ReconCommon.hlsl + Core/GridConventions.cs.

#pragma once
#include <cuda_runtime.h>
#include <vector_types.h>   // float3, float4, int3
#include <math.h>

// ─────────────────────────────────────────────────────────────────────────────
// GPU structs — strides MUST match C# ReconBuffers exactly.
// CUDA float3 has alignment 4 (NOT 16), so these pack with no padding — same as the
// C# Unity.Mathematics.float3 (3 sequential floats). Verified by static_assert.
// ─────────────────────────────────────────────────────────────────────────────

// Hermite intersection entry. Mirrors ReconBuffers.IsectGpu / HLSL IsectGpu (40 bytes):
//   float3 globalRest(12) + float3 normal(12) + int cornerA(4) + int cornerB(4) + float t(4) + int insideA(4).
struct IsectGpu
{
    float3 globalRest;  // 12  world pos of the intersection (rest); IsectWorld is rebuilt from this each frame
    float3 normal;      // 12  outward unit normal at the intersection (rest gradient — PAPER-SILENT frozen)
    int    cornerA;     //  4  global corner id of edge endpoint A
    int    cornerB;     //  4  global corner id of edge endpoint B
    float  t;           //  4  linear param along A->B where phi crosses 0
    int    insideA;     //  4  1 => cornerA is the INSIDE (phi<0) endpoint, else cornerB.  total = 40
};
static_assert(sizeof(IsectGpu) == 40, "IsectGpu must be 40 bytes (matches C# ReconBuffers.IsectGpu)");

// Surface grid edge. Mirrors ReconBuffers.GridEdgeGpu / HLSL GridEdgeGpu (32 bytes = 8 ints).
struct GridEdgeGpu
{
    int axis;                    //  axis 0=x,1=y,2=z
    int bcX, bcY, bcZ;           //  baseCorner.xyz (the "A" endpoint)
    int insideA;                 //  1 if baseCorner is inside (phi<0), else 0
    int _pad, _pad2, _pad3;      //  padding -> total 32 bytes
};
static_assert(sizeof(GridEdgeGpu) == 32, "GridEdgeGpu must be 32 bytes (matches C# ReconBuffers.GridEdgeGpu)");

// Fixed-point normal-accumulation scale. Mirrors HLSL NORMAL_SCALE / C# FixedPointAtomic.NormalScale (1<<20).
#define NORMAL_SCALE 1048576.0f

// ─────────────────────────────────────────────────────────────────────────────
// Minimal float3 vector helpers (CUDA does not define operators on float3).
// ─────────────────────────────────────────────────────────────────────────────
__device__ __host__ __forceinline__ int3   operator+(int3 a, int3 b)     { return make_int3(a.x + b.x, a.y + b.y, a.z + b.z); }
__device__ __host__ __forceinline__ int3   operator-(int3 a, int3 b)     { return make_int3(a.x - b.x, a.y - b.y, a.z - b.z); }
__device__ __host__ __forceinline__ float3 operator+(float3 a, float3 b) { return make_float3(a.x + b.x, a.y + b.y, a.z + b.z); }
__device__ __host__ __forceinline__ float3 operator-(float3 a, float3 b) { return make_float3(a.x - b.x, a.y - b.y, a.z - b.z); }
__device__ __host__ __forceinline__ float3 operator*(float3 a, float s)  { return make_float3(a.x * s, a.y * s, a.z * s); }
__device__ __host__ __forceinline__ float3 operator*(float s, float3 a)  { return make_float3(a.x * s, a.y * s, a.z * s); }
__device__ __host__ __forceinline__ float3 operator/(float3 a, float s)  { return make_float3(a.x / s, a.y / s, a.z / s); }

__device__ __host__ __forceinline__ float  dot3(float3 a, float3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
__device__ __host__ __forceinline__ float3 cross3(float3 a, float3 b)
{
    return make_float3(a.y * b.z - a.z * b.y,
                       a.z * b.x - a.x * b.z,
                       a.x * b.y - a.y * b.x);
}
__device__ __host__ __forceinline__ float  length3(float3 a) { return sqrtf(dot3(a, a)); }
// lerp(a, b, t) = a + (b-a)*t — matches HLSL lerp used by RebuildIsectWorld.
__device__ __host__ __forceinline__ float3 lerp3(float3 a, float3 b, float t)
{
    return make_float3(a.x + (b.x - a.x) * t,
                       a.y + (b.y - a.y) * t,
                       a.z + (b.z - a.z) * t);
}

// ─────────────────────────────────────────────────────────────────────────────
// Eq5 box SDF (max-form): negative inside, zero on face, positive outside.
// p = query, c = box center, L = full side length (half = L/2). Mirrors ReconCommon.hlsl BoxSdf
// and Core/GridConventions.BoxSdf exactly.
// ─────────────────────────────────────────────────────────────────────────────
__device__ __host__ __forceinline__ float BoxSdf(float3 p, float3 c, float L)
{
    float3 d = make_float3(fabsf(p.x - c.x) - L * 0.5f,
                           fabsf(p.y - c.y) - L * 0.5f,
                           fabsf(p.z - c.z) - L * 0.5f);
    return fmaxf(fmaxf(d.x, d.y), d.z);
}

// voxelId = i + Nx*(j + Ny*k) — matches Core/GridConventions.VoxelId + HLSL VoxId.
__device__ __host__ __forceinline__ int VoxId(int3 c, int3 dims)
{
    return c.x + dims.x * (c.y + dims.y * c.z);
}
