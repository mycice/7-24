#ifndef PHYSICS_COMMON_INCLUDED
#define PHYSICS_COMMON_INCLUDED

// ── Constants ────────────────────────────────────────────────────────────────────────────────────
#define FORCE_SCALE 1024.0   // FixedPointAtomic.ForceScale (1<<10)
#define EPS_SIN2    1e-8     // PAPER-SILENT: singularity guard for bending pairs (collinear / anti-collinear)
#define ERR_SCALE   1000000.0

// ── Structs ──────────────────────────────────────────────────────────────────────────────────────
// Field order MUST match C# ReconBuffers.SpringGpu / BendPairGpu (stride 16 / 32 respectively).

struct SpringGpu
{
    int   i;
    int   j;
    float L0;
    float _pad;
};

struct BendPairGpu
{
    int   i;
    int   j;
    int   k;
    float theta0;
    int   alive;
    int   _p0;
    int   _p1;
    int   _p2;
};

// ── Fixed-point force atomics ─────────────────────────────────────────────────────────────────────
// SM5.0 has no float InterlockedAdd; forces are accumulated as fixed-point ints (scale = FORCE_SCALE).
// ForceInt layout: int[3*cornerCount], packed as (x,y,z) per particle.

void AtomicAddForce(RWStructuredBuffer<int> buf, int p, float3 f)
{
    InterlockedAdd(buf[3 * p + 0], (int)(f.x * FORCE_SCALE));
    InterlockedAdd(buf[3 * p + 1], (int)(f.y * FORCE_SCALE));
    InterlockedAdd(buf[3 * p + 2], (int)(f.z * FORCE_SCALE));
}

float3 ReadForce(RWStructuredBuffer<int> buf, int p)
{
    return float3(buf[3 * p + 0], buf[3 * p + 1], buf[3 * p + 2]) / FORCE_SCALE;
}

// ── UnitDeriv: d/dt of normalize(xa - xb) given velocities va, vb ───────────────────────────────
// dN/dt = du/|u| - u*(u·du)/|u|^3   where u = xa-xb, du = va-vb
// PAPER-SILENT: used for the analytic theta_dot formula (C33).
float3 UnitDeriv(float3 xa, float3 xb, float3 va, float3 vb)
{
    float3 u   = xa - xb;
    float  len = length(u);
    if (len < 1e-8) return float3(0, 0, 0);
    float3 du = va - vb;
    return du / len - u * dot(u, du) / (len * len * len);
}

// ── DOPRI5 Butcher tableau ────────────────────────────────────────────────────────────────────────
// Values transcribed EXACTLY from plan Global Constraints (Eq18/19).
// a[stage][j]: stage coupling coefficients (7 rows × 6 cols; unused entries are 0).
static const float DOPRI_A[7][6] = {
    { 0,              0,              0,              0,              0,              0 },  // stage 0 (seed from y_n)
    { 0.2,            0,              0,              0,              0,              0 },  // stage 1: a21=1/5
    { 0.075,          0.225,          0,              0,              0,              0 },  // stage 2: 3/40, 9/40
    { 44.0/45,       -56.0/15,        32.0/9,         0,              0,              0 },  // stage 3
    { 19372.0/6561,  -25360.0/2187,   64448.0/6561,  -212.0/729,     0,              0 },  // stage 4
    { 9017.0/3168,   -355.0/33,       46732.0/5247,   49.0/176,      -5103.0/18656,  0 },  // stage 5
    { 35.0/384,       0,              500.0/1113,     125.0/192,     -2187.0/6784,    11.0/84 }  // stage 6 (= b5 weights)
};

// b5: 5th-order weights (propagated solution y5).
static const float DOPRI_B5[7] = {
    35.0/384,   0,   500.0/1113,   125.0/192,   -2187.0/6784,   11.0/84,   0
};

// b4: 4th-order weights (used only for error estimate y5 - y4).
static const float DOPRI_B4[7] = {
    5179.0/57600,   0,   7571.0/16695,   393.0/640,   -92097.0/339200,   187.0/2100,   1.0/40
};

#endif // PHYSICS_COMMON_INCLUDED
