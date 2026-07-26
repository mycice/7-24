// LiverCutHelpers.cs — liver-import pipeline.
// Pure (MonoBehaviour-free) helpers extracted from LiverCutManager so the rod sub-stepping (design §4,
// R7), the empty-grid guard (§5 / Testing #8), and the anchor pinning (§5 / Testing #9) are CPU-testable.

using UnityEngine;
using Unity.Mathematics;

namespace ReconGridDC.Demo
{
    /// <summary>Interactive-rod geometry: segment endpoints + the fast-sweep sub-step count.</summary>
    public static class RodMath
    {
        /// <summary>Rod endpoints from center/axis/length: S = C + (Lr/2)a, E = C − (Lr/2)a.</summary>
        public static void Endpoints(Vector3 center, Vector3 axis, float length, out Vector3 s, out Vector3 e)
        {
            Vector3 a = axis.sqrMagnitude > 1e-12f ? axis.normalized : Vector3.right;
            s = center + a * (length * 0.5f);
            e = center - a * (length * 0.5f);
        }

        /// <summary>
        /// Number of sub-steps so each sub-step's max endpoint travel is &lt; L (design §4 / R7:
        /// prevents skipped rest-edge layers and a single giant swept quad on a fast sweep / teleport).
        /// N = max(1, ceil(maxEndpointDisplacement / L)).
        /// </summary>
        public static int SubStepCount(Vector3 s0, Vector3 e0, Vector3 sNew, Vector3 eNew, float L)
        {
            float maxDisp = Mathf.Max(Vector3.Distance(s0, sNew), Vector3.Distance(e0, eNew));
            return Mathf.Max(1, Mathf.CeilToInt(maxDisp / Mathf.Max(L, 1e-6f)));
        }
    }

    /// <summary>Pure CPU grid helpers: occupancy guard + anchor-cap pinning.</summary>
    public static class LiverGrid
    {
        /// <summary>Does voxelOccupied contain BOTH 0s and 1s? (empty/all-outside grid guard, §5/#8).</summary>
        public static void CountOccupancy(byte[] voxelOccupied, out bool anyInside, out bool anyOutside)
        {
            anyInside = false; anyOutside = false;
            for (int v = 0; v < voxelOccupied.Length; v++)
            {
                if (voxelOccupied[v] != 0) anyInside = true; else anyOutside = true;
                if (anyInside && anyOutside) return;
            }
        }

        /// <summary>
        /// Pin the cap in +axis: active corners with dot(pos, axis) ≥ maxProj − band are pinned.
        /// Writes into `pinned`, returns the count pinned (the §5/#9 anchor guard asserts ≥1).
        /// </summary>
        public static int PinCap(float3[] cornerPos, byte[] cornerActive, byte[] pinned, float3 axis, float band)
        {
            float3 a = math.normalizesafe(axis, new float3(1, 0, 0));
            float maxProj = float.NegativeInfinity;
            for (int c = 0; c < cornerPos.Length; c++)
                if (cornerActive[c] != 0) maxProj = math.max(maxProj, math.dot(cornerPos[c], a));

            int count = 0;
            for (int c = 0; c < cornerPos.Length; c++)
            {
                if (cornerActive[c] == 0) continue;
                if (math.dot(cornerPos[c], a) >= maxProj - band) { pinned[c] = 1; count++; }
            }
            return count;
        }

        /// <summary>
        /// Replace the cap anchor with a small set of spatially distributed interior anchors.
        /// The sparse fixed points suppress rigid-body translation/rotation while leaving the
        /// surrounding corners free to deform under contact.
        /// </summary>
        public static int PinDistributed(float3[] cornerPos, byte[] cornerActive, byte[] cornerInside,
                                         byte[] pinned, int targetCount)
        {
            if (cornerPos == null || cornerActive == null || pinned == null || targetCount <= 0)
                return 0;

            int count = math.min(cornerPos.Length, math.min(cornerActive.Length, pinned.Length));
            for (int c = 0; c < count; c++) pinned[c] = 0;

            var candidates = new System.Collections.Generic.List<int>();
            bool haveInsideMask = cornerInside != null && cornerInside.Length >= count;
            for (int c = 0; c < count; c++)
                if (cornerActive[c] != 0 && (!haveInsideMask || cornerInside[c] != 0)) candidates.Add(c);

            // Very thin/coarse models may have no strictly-inside corners. Active corners are still
            // valid simulation particles, so use them as a deterministic fallback.
            if (candidates.Count == 0 && haveInsideMask)
                for (int c = 0; c < count; c++)
                    if (cornerActive[c] != 0) candidates.Add(c);

            int wanted = math.min(targetCount, candidates.Count);
            if (wanted == 0) return 0;

            float3 bmin = new float3(float.PositiveInfinity);
            float3 bmax = new float3(float.NegativeInfinity);
            for (int i = 0; i < candidates.Count; i++)
            {
                float3 p = cornerPos[candidates[i]];
                bmin = math.min(bmin, p);
                bmax = math.max(bmax, p);
            }
            float3 span = math.max(bmax - bmin, new float3(1e-6f));

            // Deterministic farthest-point sampling in normalized model space. Normalization prevents
            // the liver's long axis from consuming every anchor before the shorter axes are covered.
            var selected = new System.Collections.Generic.List<int>(wanted);
            int seed = candidates[0];
            float seedScore = float.NegativeInfinity;
            float3 center = 0.5f * (bmin + bmax);
            for (int i = 0; i < candidates.Count; i++)
            {
                int c = candidates[i];
                float score = math.lengthsq((cornerPos[c] - center) / span);
                if (score > seedScore) { seedScore = score; seed = c; }
            }
            selected.Add(seed);
            pinned[seed] = 1;

            while (selected.Count < wanted)
            {
                int best = -1;
                float bestMinDist = float.NegativeInfinity;
                for (int i = 0; i < candidates.Count; i++)
                {
                    int c = candidates[i];
                    if (pinned[c] != 0) continue;

                    float minDist = float.PositiveInfinity;
                    float3 pc = (cornerPos[c] - bmin) / span;
                    for (int j = 0; j < selected.Count; j++)
                    {
                        float3 ps = (cornerPos[selected[j]] - bmin) / span;
                        minDist = math.min(minDist, math.lengthsq(pc - ps));
                    }
                    if (minDist > bestMinDist) { bestMinDist = minDist; best = c; }
                }
                if (best < 0) break;
                selected.Add(best);
                pinned[best] = 1;
            }
            return selected.Count;
        }
    }
}
