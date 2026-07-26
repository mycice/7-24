// GridFit.cs — liver-import pipeline, design §3.
// Sizes the background voxel grid (dims, L, origin) to a model's bbox + a margin shell of empty
// voxels (so cornerActive/voxelOccupied have an outside ring → the DC surface closes).
//   L      = maxExtent / targetLongAxisVoxels
//   dims   = ceil(extent / L) + 2*marginVoxels   (per axis)
//   origin = min − marginVoxels*L                (negative origin is fine; passed identically to
//                                                 BackgroundGrid.Build AND CutDetector.Dispatch)
// Corner rest pos = origin + coord*L (coord ∈ [0,dims]) then covers [min − margin*L, max + margin*L].

using Unity.Mathematics;

namespace ReconGridDC.Preprocess
{
    public static class GridFit
    {
        public static void FitToBounds(float3 min, float3 max, int targetLongAxisVoxels, int marginVoxels,
                                       out int3 dims, out float L, out float3 origin)
        {
            float3 extent  = max - min;
            float maxExtent = math.max(math.cmax(extent), 1e-12f);  // guard degenerate (coincident) bounds → no /0
            int target = math.max(1, targetLongAxisVoxels);
            L = maxExtent / target;

            int3 inner = math.max((int3)math.ceil(extent / L), new int3(1, 1, 1));
            dims   = inner + 2 * marginVoxels;
            origin = min - marginVoxels * L;
        }
    }
}
