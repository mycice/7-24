// GraphColoring.cs
// 灏嶅洓闈㈤珨缍叉牸鍩疯鍦栬憲鑹诧紙Graph Coloring锛?// 浣滅敤锛氭妸 tet 鍒嗘垚鑻ュ共銆岄鑹茬祫銆嶏紝鍚屼竴绲勫収鐨?tet 涓嶅叡浜爞榛?//       鈫?鍚屼竴椤忚壊绲勫彲浠ュ畬鍏ㄤ甫琛屽煼琛岀磩鏉熸眰瑙ｏ紙鐒℃暩鎿氱鐖級
// 绠楁硶锛氳勃蹇冭憲鑹诧紙Greedy Coloring锛夛紝O(n路k) 瑜囬洔搴︼紝k = 椤忚壊鏁?// 鍙冭€冿細GPU-accelerated XPBD 鍦栬憲鑹叉柟妗堬紙妤晫妯欐簴鍋氭硶锛?
using System;
using System.Collections.Generic;
using UnityEngine;

namespace ReconGridDC.Stage1TetPhysics.Core
{
    public static class GraphColoring
    {
        /// <summary>
        /// 灏嶅洓闈㈤珨鍩疯璨績鍦栬憲鑹层€?        /// 杩斿洖 colorGroups锛氭瘡鍊嬪厓绱犳槸涓€鍊嬮鑹茬祫鍖呭惈鐨?tet 绱㈠紩鍒楄〃銆?        /// </summary>
        /// <param name="tetIds">鍥涢潰楂旈爞榛炵储寮曪紙骞冲潶鏁哥祫锛屾瘡4鍊嬩竴绲勶級</param>
        /// <param name="numTets">鍥涢潰楂旂附鏁?/param>
        /// <param name="numParticles">绮掑瓙绺芥暩</param>
        /// <returns>椤忚壊鍒嗙祫锛屾瘡绲勭偤 tet 绱㈠紩鏁哥祫</returns>
        public static List<int[]> Compute(int[] tetIds, int numTets, int numParticles,
            bool[] tetActive = null)
        {
            var startTime = DateTime.Now;

            // 姣忓€嬬矑瀛愯鍝簺 tet 浣跨敤锛堥劙鎺ラ棞淇傦級
            // vertToTets[v] = 鍖呭惈闋傞粸 v 鐨?tet 鍒楄〃
            var vertToTets = new List<int>[numParticles];
            for (int i = 0; i < numParticles; i++)
                vertToTets[i] = new List<int>(8); // 姣忓€嬮爞榛炲钩鍧囧爆鏂?~8 鍊?tet

            for (int t = 0; t < numTets; t++)
            {
                if (tetActive != null && !tetActive[t]) continue;
                int b = t * 4;
                vertToTets[tetIds[b + 0]].Add(t);
                vertToTets[tetIds[b + 1]].Add(t);
                vertToTets[tetIds[b + 2]].Add(t);
                vertToTets[tetIds[b + 3]].Add(t);
            }

            // 鈹€鈹€ 璨績钁楄壊 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
            int[] tetColor  = new int[numTets];
            Array.Fill(tetColor, -1); // -1 = 鏈垎閰?
            // 宸ヤ綔闆嗭細瑷橀寗閯板眳宸蹭娇鐢ㄧ殑椤忚壊
            var neighborColors = new HashSet<int>(32);

            for (int t = 0; t < numTets; t++)
            {
                if (tetActive != null && !tetActive[t]) { tetColor[t] = 0; continue; }
                int b = t * 4;
                neighborColors.Clear();

                // Collect colors used by tetrahedra that share a particle.
                for (int vi = 0; vi < 4; vi++)
                {
                    int v = tetIds[b + vi];
                    foreach (int neighbor in vertToTets[v])
                    {
                        if (neighbor < t && tetColor[neighbor] >= 0)
                            neighborColors.Add(tetColor[neighbor]);
                    }
                }

                // Choose the smallest available color.
                int color = 0;
                while (neighborColors.Contains(color))
                    color++;
                tetColor[t] = color;
            }

            // 鈹€鈹€ 鎸夐鑹插垎绲?鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
            int numColors = 0;
            foreach (int c in tetColor)
                if (c > numColors) numColors = c;
            numColors++;

            var groupLists = new List<int>[numColors];
            for (int i = 0; i < numColors; i++)
                groupLists[i] = new List<int>(numTets / numColors + 10);

            for (int t = 0; t < numTets; t++)
            {
                if (tetActive != null && !tetActive[t]) continue;
                groupLists[tetColor[t]].Add(t);
            }

            var groups = new List<int[]>(numColors);
            foreach (var g in groupLists)
                groups.Add(g.ToArray());

            // 鈹€鈹€ 绲辫▓鍫卞憡 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
            float elapsed = (float)(DateTime.Now - startTime).TotalMilliseconds;
            int   minSize = int.MaxValue, maxSize = 0;
            foreach (var g in groups)
            {
                if (g.Length < minSize) minSize = g.Length;
                if (g.Length > maxSize) maxSize = g.Length;
            }

            Debug.Log($"[GraphColoring] 钁楄壊瀹屾垚 | " +
                      $"椤忚壊鏁? {numColors} | " +
                      $"鏈€灏忕祫: {minSize} | 鏈€澶х祫: {maxSize} | " +
                      $"鑰楁檪: {elapsed:F1}ms");

            // Validate that tetrahedra in each color group do not share vertices.
#if UNITY_EDITOR
            ValidateColoring(groups, tetIds, numTets, tetActive);
#endif

            return groups;
        }

        /// <summary>
        /// 杩斿洖閬╁悎 GPU Dispatch 鐨勫钩鍧︽暩绲勬牸寮?        /// groupFlat: 鎵€鏈?tet 绱㈠紩骞冲潶鎺掑垪
        /// groupRanges: [numColors * 2] 鈫?(start, count) 灏?        /// </summary>
        public static void ComputeFlat(
            int[] tetIds, int numTets, int numParticles,
            out int[] groupFlat, out int[] groupRanges)
        {
            var groups = Compute(tetIds, numTets, numParticles);

            int totalCount = 0;
            foreach (var g in groups) totalCount += g.Length;

            groupFlat   = new int[totalCount];
            groupRanges = new int[groups.Count * 2];

            int offset = 0;
            for (int i = 0; i < groups.Count; i++)
            {
                var g = groups[i];
                groupRanges[i * 2 + 0] = offset;     // start
                groupRanges[i * 2 + 1] = g.Length;   // count
                Array.Copy(g, 0, groupFlat, offset, g.Length);
                offset += g.Length;
            }
        }

        // 鈹€鈹€ 椹楄瓑锛圗ditor 妯″紡涓嬩娇鐢級鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
        static void ValidateColoring(List<int[]> groups, int[] tetIds, int numTets,
            bool[] tetActive = null)
        {
            int conflicts = 0;
            for (int gi = 0; gi < groups.Count; gi++)
            {
                var group = groups[gi];
                var usedVerts = new HashSet<int>(group.Length * 4);
                foreach (int t in group)
                {
                    if (tetActive != null && !tetActive[t]) continue;
                    int b = t * 4;
                    for (int vi = 0; vi < 4; vi++)
                    {
                        int v = tetIds[b + vi];
                        if (!usedVerts.Add(v))
                        {
                            conflicts++;
                            break;
                        }
                    }
                }
            }
            if (conflicts > 0)
                Debug.LogWarning($"[GraphColoring] 鐫€鑹叉湁 {conflicts} 澶勮交寰啿绐侊紙涓嶅奖鍝嶇ǔ瀹氭€э級");
            else
                Debug.Log("[GraphColoring] 椹楄瓑閫氶亷锛氭墍鏈夐鑹茬祫鍏х劇闋傞粸琛濈獊");
        }
    }
}

