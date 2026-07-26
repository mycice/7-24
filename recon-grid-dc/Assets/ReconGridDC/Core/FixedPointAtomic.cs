namespace ReconGridDC.Core
{
    public static class FixedPointAtomic
    {
        public const float NormalScale = 1048576f; // 1<<20 ; HLSL twin in ReconCommon.hlsl
        public const float ForceScale  = 1024f;    // 1<<10 ; reserved for Stage 1B
        // Stage 2C: fixed-point scale for cut-FP centroid position sums (paper §2.1.1 centroid).
        // HLSL twin "#define POS_SCALE 16384.0" lives in CuttingCommon.hlsl (the cut kernels include
        // CuttingCommon, NOT ReconCommon). Bound (plan §3.4 / R-C): max ~48 scatter events × 16384 ×
        // ~10 world ≈ 7.9e6 ≪ 2.1e9 int32 max — no overflow.
        public const float PosScale    = 16384f;   // 1<<14 ; cut-FP centroid fixed-point scale (2C)
    }
}
