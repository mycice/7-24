#ifndef RECON_COMMON_INCLUDED
#define RECON_COMMON_INCLUDED

#define NORMAL_SCALE 1048576.0   // FixedPointAtomic.NormalScale (1<<20); HLSL twin of C# FixedPointAtomic.NormalScale

// Hermite intersection entry - matches C# ReconBuffers.IsectGpu (40 bytes)
// IsectGpu = 40 bytes (float3+float3+int+int+float+float); plan's "32" was wrong
struct IsectGpu
{
    float3 globalRest;  // world position of intersection (static; updated by RebuildIsectWorld in 1B)
    float3 normal;      // outward unit normal at intersection (analytic gradient)
    int    cornerA;     // global corner id of edge endpoint A (local-edge order; not enforced inside-first)
    int    cornerB;     // global corner id of edge endpoint B (local-edge order; not enforced outside-first)
    float  t;           // linear param along A->B where phi crosses 0
    int    insideA;     // B1 (plan §7): 1 ⇒ cornerA is the INSIDE (phi<0) endpoint, else cornerB. 40 bytes total
};

// Eq5 box SDF (max-form). Negative inside, zero on face, positive outside.
// p = query point, c = box center, L = full side length (half = L/2).
float BoxSdf(float3 p, float3 c, float L)
{
    float3 d = abs(p - c) - L * 0.5;
    return max(max(d.x, d.y), d.z);
}

#endif // RECON_COMMON_INCLUDED
