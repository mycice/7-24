// TetSubdivisionCutter.cs — Phase 2: 论文级切割质量
//
// 基于 PG2025 "Parallel Constraint Graph Partitioning and Coloring
//              for Realtime Soft-Body Cutting" (Peng Yu et al.)
//
// Phase 2 新增 (Section 4.3):
//   1. 轨迹修正 (Trajectory Correction): 交点接近顶点时吸附
//   2. 共享原始顶点判据 (Shared Original Vertex): 决定分离/连接
//   3. 切口内表面三角形: 使切口可见

using System.Collections.Generic;
using UnityEngine;
using SurgicalSim.Core;
using SurgicalSim.Cutting;
using SurgicalSim.Physics;

namespace SurgicalSim.CuttingV3
{
    public class TetSubdivisionCutter
    {
        TetMeshData   _data;
        XPBDSolverGPU _solver;

        int  _totalNewVerts, _totalNewTets, _totalCutTets;
        bool _dirty;

        readonly HashSet<int> _alreadyCutTets = new HashSet<int>();

        // ★ 双层缓存策略，彻底解决藕断丝连
        // _frameSplitCache : 帧级，每帧清除。同帧内同边复用相同 posV/negV，与当前帧平面精确匹配。
        // _strokeSplitCache: 划刀级，ResetStroke才清除。跨帧拓扑连续性，复用已切割原始边的顶点对。
        //   Only stable initial vertices (ID < _originalVertCount) can use stroke cache.
        struct SplitCacheValue
        {
            public int posV, negV;
            public float t;
            public Vector3 normal;
            public int snapVertex;

            public (int posV, int negV) Pair => (posV, negV);
        }

        readonly Dictionary<long, SplitCacheValue> _frameSplitCache
            = new Dictionary<long, SplitCacheValue>();
        readonly Dictionary<long, SplitCacheValue> _strokeSplitCache
            = new Dictionary<long, SplitCacheValue>();
        readonly Dictionary<int, int> _frameSnapCache = new Dictionary<int, int>();
        readonly HashSet<int> _dirtyCutVertices = new HashSet<int>();
        readonly Dictionary<int, List<int>> _dirtyIncidentMap = new Dictionary<int, List<int>>();
        readonly List<int> _componentScratch = new List<int>(64);
        readonly Queue<int> _componentQueue = new Queue<int>();
        readonly List<int> _vertexBirthPhase = new List<int>();

        readonly List<int> _surfaceSupport0 = new List<int>();
        readonly List<int> _surfaceSupport1 = new List<int>();
        readonly List<int> _surfaceSupport2 = new List<int>();
        readonly HashSet<long> _originalSurfaceFaceKeys = new HashSet<long>();
        readonly HashSet<long> _originalSurfaceEdgeKeys = new HashSet<long>();
        readonly int[] _sortVi = new int[4];
        readonly float[] _sortSd = new float[4];
        int _cutLogCounter;

        struct VertexSideLock { public int side; public Vector3 normal; }
        readonly Dictionary<int, VertexSideLock> _strokeVertexSide = new Dictionary<int, VertexSideLock>();

        // ★ 需求②：本次 Cut() 调用中新创建的"真插值"切割面顶点（不含 snap 克隆），
        // 以及它们在同一切口多边形内的邻接关系，供 Cut() 末尾做一次切向松弛平滑。
        readonly List<int> _newInterpVertsThisCut = new List<int>(64);
        readonly Dictionary<int, List<int>> _cutVertexAdjacency = new Dictionary<int, List<int>>();

        int _originalVertCount;
        int _cutPhaseId;

        const float MIN_TET_VOL = 1e-11f;
        const float SNAP_T = 1e-4f;
        const float SNAP_PLANE_DIST = 1.05e-4f;
        const float SIDE_EPS = 1e-5f;
        const float SLIVER_REST_VOLUME = 1e-10f;
        const int MAX_SEPARATION_PASSES = 6;
        // ★ 需求①修复：克隆体微偏移量。克隆顶点若与原顶点完全重合（位置/速度都相同），
        // 在没有额外外力的情况下 XPBD 求解器没有任何驱动力使两者分开，看起来就是"藕断丝连"。
        // 给克隆体一个微小的、朝其所属那一侧材料质心方向的位移，让分离在视觉/物理上立即可见。
        const float CLONE_SEPARATION_EPS = 3e-4f;
        // ★ 需求②修复：新切割面顶点的切向松弛系数（Zeng & Courtecuisse 2025 思路的轻量版：
        // 把"吸附阈值"重新表述为"切割网格质量优化"，但用一次性的邻居均值松弛代替完整的
        // 准静态弹性求解，只在切割平面内移动，绝不沿法线方向移动，不改变分割侧）。
        const float CUT_SURFACE_SMOOTH_FACTOR = 0.25f;
        private int _strokeStartTetCount = -1;
        private Vector3 _currentCutNormal = Vector3.zero;

        // ── 诊断 ─────────────────────────────────────────────
        public float  LastMoveDistance        { get; private set; }
        public int    LastCandidateTetCount   { get; private set; }
        public int    LastIntersectedTetCount { get; private set; }
        public string LastRejectReason        { get; private set; } = "idle";
        public int    LastSeparatedVertexCount { get; private set; }
        public int    LastForcedSeparationCount { get; private set; }
        public int    LastResidualSharedVertexCount { get; private set; }
        public int    LastSeparationPassCount { get; private set; }
        public int    LastProtectedTetCount { get; private set; }
        public int    LastSkippedCurrentPhasePivotCount { get; private set; }
        public int  TotalNewVerts => _totalNewVerts;
        public int  TotalNewTets  => _totalNewTets;
        public int  TotalCutTets  => _totalCutTets;
        public bool IsDirty       => _dirty;
        public IList<int> SurfaceSupport0 => _surfaceSupport0;
        public IList<int> SurfaceSupport1 => _surfaceSupport1;
        public IList<int> SurfaceSupport2 => _surfaceSupport2;
        public HashSet<long> OriginalSurfaceFaceKeys => _originalSurfaceFaceKeys;

        // ── 迭代2.1：全局材料连通分量 (2024 §2.2.1 AFCC 概念 + 2025 §4.3 shared-vertex 邻接) ──
        public int LastComponentCount { get; private set; }
        public int LastSeparatedComponentCount { get; private set; }
        public int LastIndependentBlockCount { get; private set; }
        public int LastFragmentComponentCount { get; private set; }
        public int LastResidualBridgeCount { get; private set; }
        public int LastLargestComponentTetCount { get; private set; }
        int[] _ufParent = System.Array.Empty<int>();
        int[] _ufSize = System.Array.Empty<int>();
        int[] _firstTetOfParticle = System.Array.Empty<int>();
        int[] _tetComponentRoot = System.Array.Empty<int>();
        readonly List<int> _strokeSepOrig = new List<int>();
        readonly List<int> _strokeSepClone = new List<int>();
        const int INDEPENDENT_BLOCK_MIN_TETS = 4;

        // ══════════════════════════════════════════════════════
        public void Init(TetMeshData data, XPBDSolverGPU solver)
        {
            _data = data; _solver = solver;
            _totalNewVerts = _totalNewTets = _totalCutTets = 0;
            _dirty = false;
            _alreadyCutTets.Clear();
            _frameSplitCache.Clear();
            _strokeSplitCache.Clear();
            _frameSnapCache.Clear();
            _strokeVertexSide.Clear();
            _dirtyCutVertices.Clear();
            _originalVertCount    = data.NumParticles;
            _cutPhaseId = 0;
            _vertexBirthPhase.Clear();
            for (int i = 0; i < data.NumParticles; i++)
                _vertexBirthPhase.Add(-1);
            _data.EnsureCapacity(data.NumParticles * 4);
            _data.EnsureTetCapacity(data.NumTets * 4);
            BuildOriginalSurfaceSupports();
            Debug.Log($"[SweptCutter] Init V:{data.NumParticles} T:{data.NumTets}");
        }

        private Vector3 _lastValidNormal = Vector3.zero;
        private Vector3 _strokeBaseNormal = Vector3.zero;

        // ★ 多刀接缝修复：preserveContinuity=true 表示"这一刀其实是上一刀的延续"
        // （刀片长度有限，沿肿瘤边缘需要分多刀切，或挥刀过程中接触短暂丢失又恢复）。
        // 此时绝不能清空 _strokeSplitCache / _strokeSepOrig / _strokeSepClone，也不推进 cutPhase：
        // _strokeSplitCache 记录了"原始边 -> 已生成的切割点对"，是跨帧/跨刀保持切口几何完全重合的唯一依据
        // （复用与否由 CanReuseStrokeSplit 按法线夹角/位置漂移逐条校验，不会因为复用了旧点而产生错误拓扑）。
        // 一旦清空，下一刀只能在该区域重新插值出一个略微偏移的新交点，
        // 和上一刀留下的切口产生几何错位——夹在中间的那个四面体，两刀的平面谁都判定不到它跨越平面（4 个角同侧），
        // 于是被漏切，材料看起来"接上了"实际仍由这一个四面体桥连——这正是用户反馈"两刀连上了却切不透"的根因。
        // _strokeSepOrig/_strokeSepClone 同理不清空，使残桥审计能继续跨刀追踪同一条逻辑缝合线。
        public void ResetStroke(bool preserveContinuity = false)
        {
            LastRejectReason = "not_cutting";
            _lastValidNormal = Vector3.zero;
            _strokeBaseNormal = Vector3.zero;
            // 帧级/同侧锁定缓存：无论是否连续都应清空，它们只在单帧内有效，不承担跨刀缝合职责。
            _frameSplitCache.Clear();
            _frameSnapCache.Clear();
            _strokeVertexSide.Clear();
            _dirtyCutVertices.Clear();
            _strokeStartTetCount  = -1;
            if (!preserveContinuity)
            {
                _strokeSplitCache.Clear();
                _strokeSepOrig.Clear();   // 迭代2.1：新 stroke 清空分离事件记录
                _strokeSepClone.Clear();
                AdvanceCutPhase();
            }
        }

        // ══════════════════════════════════════════════════════
        // 主入口: 平面切割
        // ══════════════════════════════════════════════════════
        public struct CutResult { public int newVerts, newTets, cutTets; public float elapsedMs; }

        public CutResult Cut(Vector3 A0, Vector3 B0, Vector3 A1, Vector3 B1,
                             float bladeRadius, bool useFiniteSweptBladeFilter = false)
        {
            var sw = System.Diagnostics.Stopwatch.StartNew();
            var result = new CutResult();
            LastCandidateTetCount = LastIntersectedTetCount = 0;
            LastSeparatedVertexCount = 0;
            LastForcedSeparationCount = 0;
            LastResidualSharedVertexCount = 0;
            LastSeparationPassCount = 0;
            LastProtectedTetCount = 0;
            LastSkippedCurrentPhasePivotCount = 0;
            _newInterpVertsThisCut.Clear();
            _cutVertexAdjacency.Clear();

            float moveA = (A1 - A0).magnitude;
            float moveB = (B1 - B0).magnitude;
            LastMoveDistance = Mathf.Max(moveA, moveB);
            bladeRadius = Mathf.Max(bladeRadius, 0.001f);

            if (LastMoveDistance < 1e-5f)
            { LastRejectReason = "move_too_small"; return result; }

            // ── 构建切割平面 ───────────────────────────────────
            Vector3 bladeDir = (B1 - A1);
            float bladeLen = bladeDir.magnitude;
            if (bladeLen < 1e-6f)
            { LastRejectReason = "degenerate_blade"; return result; }
            bladeDir /= bladeLen;

            Vector3 moveDir = ((A1 + B1) - (A0 + B0)) * 0.5f;
            float moveMag = moveDir.magnitude;
            if (moveMag < 1e-6f)
            { LastRejectReason = "no_movement"; return result; }
            moveDir /= moveMag;

            // ★ 修复：解锁法线与中心！允许切面随手腕甩动而弯曲！
            // 之前为了防止切口锯齿，锁定了第一帧的法线。但这导致如果玩家在收刀时“甩腕”或“旋转”刀锋，
            // 数学平面依然是直的，刀尖扫过的边缘区域即使在包围盒内，也因为不与这个固定的直平面相交而漏切（藕断丝连）！
            // 由于我们已经有了贯穿整个 Stroke 的 _edgeCache 和 _snappedCache 兜底，
            // 即使每帧平面发生微小偏转，缓存也能把相邻帧的四面体完美“缝合”在一起，绝不会出现锯齿或裂缝！
            Vector3 planeNormal = Vector3.Cross(bladeDir, moveDir);
            float planeNLen = planeNormal.magnitude;
            if (planeNLen < 1e-4f)
            {
                if (_lastValidNormal != Vector3.zero)
                {
                    planeNormal = _lastValidNormal;
                }
                else
                {
                    LastRejectReason = "parallel_move";
                    return result;
                }
            }
            else
            {
                planeNormal /= planeNLen;

                // ★ 智能分段：当刀片发生明显转弯（累计超过 25 度）时，解锁新生成的子四面体，允许二次切割。
                if (_strokeBaseNormal == Vector3.zero)
                {
                    _strokeBaseNormal = planeNormal;
                }
                else
                {
                    float angle = Vector3.Angle(_strokeBaseNormal, planeNormal);
                    if (angle > 25f && moveMag >= Mathf.Max(bladeRadius * 0.25f, 0.001f))
                    {
                        // ★ 修复：转弯只推进 cutPhase 并解锁子四面体（_strokeStartTetCount），
                        // 不再 Clear() _strokeSplitCache！沿肿瘤边缘走折线/曲线时，转弯点恰恰是最需要
                        // 切口几何保持连续的地方——CanReuseStrokeSplit 本身已经按法线夹角(<~20°)/
                        // 位置漂移(<3mm)/t 漂移逐条校验是否可复用，不需要在这里整体清空。
                        // 之前在转弯处整体清空，会导致转弯顶点处本可精确复用的切割点被强制作废，
                        // 下一段重新插值出一个轻微偏移的新点，与上一段的切口产生几何错位，
                        // 夹在中间的四面体两段平面都判定不到它跨越平面，于是被漏切——这正是
                        // "刀缝看似连上却切不透"在单笔连续挥刀转弯处的成因。
                        Debug.Log($"[TetSubdivisionCutter] Stroke normal turned {angle:F1} deg; advancing cut phase (keeping split-point continuity).");
                        AdvanceCutPhase();
                        _strokeStartTetCount = _data.NumTets;
                        _frameSplitCache.Clear();
                        _frameSnapCache.Clear();
                        _strokeVertexSide.Clear();
                        _strokeBaseNormal = planeNormal; // 重置基准法线
                    }
                }
                _lastValidNormal = planeNormal;
            }

            Vector3 planeCenter = (A1 + B1) * 0.5f;
            _currentCutNormal = planeNormal;

            // ★ 帧级缓存每帧必须清除：每帧平面法线都可能因刀的旋转而偏转，
            // 帧缓存若不清除，旧交点会错位到新平面之外，产生拓扑扭曲和藕断丝连！
            _frameSplitCache.Clear();
            _frameSnapCache.Clear();  // 同一帧内同顶点 snap 必须复用同一克隆——每帧重置

            // ── 遍历所有 tet ─────────────────────────────────
            _dirtyCutVertices.Clear();
            int vBefore = _data.NumParticles;
            int tBefore = _data.NumTets;

            // ★ 防止“切口粉碎机”效应的核心防线：
            // 因为这是一把“连续挥动”的刀，平面的数学方程被锁定了（_lockedNormal）。
            // 如果我们在这个循环里检查这一刀刚才（或者前几帧）切出来的子四面体，
            // 它们因为物理引擎的弹性会抖动，导致它们不可避免地再次跨越这个虚拟的数学平面。
            // 结果就是：切好的肉自己疯狂抖动着撞向留在原地的“幽灵刀刃”，被切成成千上万的粉末！
            // 所以，在同一个连续挥刀动作里，我们【绝对不能】切这一刀切出来的子四面体！
            if (_strokeStartTetCount == -1) _strokeStartTetCount = tBefore;
            // Cut-phase recut guard: child tets become cut candidates only after a phase unlock.
            int tetSnapshot = Mathf.Min(_strokeStartTetCount, tBefore);
            LastProtectedTetCount = Mathf.Max(0, tBefore - tetSnapshot);

            int candCount = 0, cutCount = 0;
            Vector3 minBounds = Vector3.Min(Vector3.Min(A0, B0), Vector3.Min(A1, B1));
            Vector3 maxBounds = Vector3.Max(Vector3.Max(A0, B0), Vector3.Max(A1, B1));
            float kMargin = Mathf.Max(bladeRadius, 0.015f);
            Vector3 lo = minBounds - Vector3.one * kMargin;
            Vector3 hi = maxBounds + Vector3.one * kMargin;

            for (int t = 0; t < tetSnapshot; t++)
            {
                if (!_data.TetActive[t]) continue;
                if (_alreadyCutTets.Contains(t)) continue;

                int b = t * 4;
                int i0 = _data.TetIds[b], i1 = _data.TetIds[b+1];
                int i2 = _data.TetIds[b+2], i3 = _data.TetIds[b+3];

                // 1. ★ AABB 测试提前，只处理在刀刃扫掠区域内的 tet
                Vector3 v0 = _data.Positions[i0], v1 = _data.Positions[i1];
                Vector3 v2 = _data.Positions[i2], v3 = _data.Positions[i3];
                Vector3 tMin = Vector3.Min(Vector3.Min(v0, v1), Vector3.Min(v2, v3));
                Vector3 tMax = Vector3.Max(Vector3.Max(v0, v1), Vector3.Max(v2, v3));
                // 如果两个 AABB 在任意一个轴上不重叠，则不可能相交
                if (tMin.x > hi.x || tMax.x < lo.x ||
                    tMin.y > hi.y || tMax.y < lo.y ||
                    tMin.z > hi.z || tMax.z < lo.z)
                    continue;
                if (useFiniteSweptBladeFilter &&
                    !TetTouchesSweptBlade(v0, v1, v2, v3, A0, B0, A1, B1, bladeRadius))
                    continue;

                // 2. 计算点到切割平面的有符号距离
                float d0 = Vector3.Dot(v0 - planeCenter, planeNormal);
                float d1 = Vector3.Dot(v1 - planeCenter, planeNormal);
                float d2 = Vector3.Dot(v2 - planeCenter, planeNormal);
                float d3 = Vector3.Dot(v3 - planeCenter, planeNormal);

                // 3. [Paper-faithful] Removed the plane-side "EnforceConsistentSide"
                // lock. It mutated d0..d3 BEFORE subdivision, bending the created
                // topology to a per-frame plane — re-injecting the very plane
                // dependence 2025 §4.3 abandons (and the audit flagged it as
                // corrupting subdivision itself). Subdivision now uses raw signed
                // distances + trajectory-correction snap (below) only; cross-frame
                // continuity comes from the shared split cache, not a side lock.

                // 4. Trajectory Correction (平面吸附)
                if (Mathf.Abs(d0) < SNAP_PLANE_DIST) d0 = 0f;
                if (Mathf.Abs(d1) < SNAP_PLANE_DIST) d1 = 0f;
                if (Mathf.Abs(d2) < SNAP_PLANE_DIST) d2 = 0f;
                if (Mathf.Abs(d3) < SNAP_PLANE_DIST) d3 = 0f;

                int pos = 0, neg = 0;
                if (d0 >= 0f) pos++; else neg++;
                if (d1 >= 0f) pos++; else neg++;
                if (d2 >= 0f) pos++; else neg++;
                if (d3 >= 0f) pos++; else neg++;

                if (pos == 4 || pos == 0) continue;

                candCount++;

                _sortVi[0] = i0; _sortVi[1] = i1; _sortVi[2] = i2; _sortVi[3] = i3;
                _sortSd[0] = d0; _sortSd[1] = d1; _sortSd[2] = d2; _sortSd[3] = d3;

                // ★ 拓扑一致性关键: 按照全局顶点ID排序！
                // 这保证了所有相邻四面体在共享面上切出的多边形，其三角剖分的对角线选择是完全一致的！
                // 否则共享面的对角线会交叉，导致内部面碎片和渲染错误！
                System.Array.Sort(_sortVi, _sortSd);

                _dirtyCutVertices.Add(i0);
                _dirtyCutVertices.Add(i1);
                _dirtyCutVertices.Add(i2);
                _dirtyCutVertices.Add(i3);

                _data.DeactivateTet(t);
                _alreadyCutTets.Add(t);

                if (pos == 1 || pos == 3)
                    Case1_Split(_sortVi, _sortSd, pos);
                else if (pos == 2)
                    Case2_Split(_sortVi, _sortSd);

                cutCount++;
            }

            if (cutCount > 0)
            {
                LastSeparatedVertexCount = SeparateDirtyVertices(planeCenter, planeNormal);
                AuditMaterialComponents(); // 迭代2.1：全局连通分量 + 闭合块独立判定 + 藕断丝连审计
                SmoothNewCutSurfaceVertices(planeNormal); // 需求②：新切割面顶点切向松弛，改善平滑度
            }

            LastCandidateTetCount = candCount;
            LastIntersectedTetCount = cutCount;
            result.cutTets  = cutCount;
            result.newVerts = _data.NumParticles - vBefore;
            result.newTets  = _data.NumTets - tBefore;
            result.elapsedMs = (float)sw.Elapsed.TotalMilliseconds;
            _totalCutTets  += result.cutTets;
            _totalNewVerts += result.newVerts;
            _totalNewTets  += result.newTets;
            if (cutCount > 0)
                _dirty = true;
            LastRejectReason = cutCount > 0 ? "cut_ok" : "no_intersection";

            if (cutCount > 0 && (++_cutLogCounter % 30) == 0)
                Debug.Log($"[SweptCutter] cut:{cutCount} new:{result.newVerts}v/{result.newTets}t " +
                          $"V:{_data.NumParticles} T:{_data.NumTets} {result.elapsedMs:F1}ms");
            return result;
        }

        // ═══════════════════════════════════════════════════════
        // Case 1: 1 isolated vertex → 4 sub-tets
        //
        // tet ABCD, A 孤立
        //   B₁ on edge(A,B), C₁ on edge(A,C), D₁ on edge(A,D)
        //
        // 孤立侧: {A, B₁, C₁, D₁}
        //
        // 多数侧 = 三棱柱 B-C-D (底面) / B₁-C₁-D₁ (切口面)
        //   ★ 保面分解 (pivot=B): 保留原始面 BCD
        //     {B, C, D, D₁}      ← 保留面 BCD ✓
        //     {B, C, C₁, D₁}     ← 中间连接
        //     {B, B₁, C₁, D₁}   ← 保留切口面 B₁C₁D₁ ✓
        // ═══════════════════════════════════════════════════════
        void Case1_Split(int[] v, float[] d, int posCount)
        {
            int isoSide = posCount == 1 ? 1 : -1;
            int isoIdx = -1;
            int[] oIdx = new int[3]; int oi = 0;
            for (int i = 0; i < 4; i++)
            {
                if ((d[i] >= 0 ? 1 : -1) == isoSide) isoIdx = i;
                else oIdx[oi++] = i;
            }

            int A = v[isoIdx];
            int B = v[oIdx[0]], C = v[oIdx[1]], D = v[oIdx[2]];
            bool isoIsPos = isoSide > 0;

            var (b1p, b1n) = EP(A, B, d[isoIdx], d[oIdx[0]]);
            var (c1p, c1n) = EP(A, C, d[isoIdx], d[oIdx[1]]);
            var (d1p, d1n) = EP(A, D, d[isoIdx], d[oIdx[2]]);

            int B1_iso = isoIsPos ? b1p : b1n;
            int C1_iso = isoIsPos ? c1p : c1n;
            int D1_iso = isoIsPos ? d1p : d1n;
            int B1_oth = isoIsPos ? b1n : b1p;
            int C1_oth = isoIsPos ? c1n : c1p;
            int D1_oth = isoIsPos ? d1n : d1p;

            // 需求②：记录切口三角形 {B1,C1,D1} 内部的邻接关系，供末尾切向松弛使用
            LinkCutAdjacency(B1_iso, C1_iso);
            LinkCutAdjacency(C1_iso, D1_iso);
            LinkCutAdjacency(D1_iso, B1_iso);
            LinkCutAdjacency(B1_oth, C1_oth);
            LinkCutAdjacency(C1_oth, D1_oth);
            LinkCutAdjacency(D1_oth, B1_oth);

            // 孤立侧
            AT(A, B1_iso, C1_iso, D1_iso);

            // 多数侧: 保面三棱柱分解
            AT(B, C,      D,      D1_oth);   // 保留面 BCD ✓
            AT(B, C,      C1_oth, D1_oth);    // 中间连接
            AT(B, B1_oth, C1_oth, D1_oth);    // 保留切口面 ✓
        }

        // ═══════════════════════════════════════════════════════
        // Case 2: 2+2 split → 6 sub-tets
        //
        // tet ABCD, {A,D} 正侧, {B,C} 负侧
        //   E₁ on edge(A,B), F₁ on edge(A,C)
        //   G₁ on edge(D,C), H₁ on edge(D,B)
        //
        // 正侧 = 四棱锥, 分解保留边 AD:
        //   {A, E₁, F₁, H₁}    ← A侧
        //   {A, D,  F₁, H₁}    ← 保留 AD ✓
        //   {D, F₁, G₁, H₁}    ← D侧
        //
        // 负侧同理, 保留边 BC:
        //   {B, E₁, F₁, H₁}    ← B侧
        //   {B, C,  F₁, H₁}    ← 保留 BC ✓
        //   {C, F₁, G₁, H₁}    ← C侧
        // ═══════════════════════════════════════════════════════
        void Case2_Split(int[] v, float[] d)
        {
            int A = -1, D_ = -1, B = -1, C = -1;
            float dA = 0, dD = 0, dB = 0, dC = 0;
            int pi = 0, ni = 0;
            for (int i = 0; i < 4; i++)
            {
                if (d[i] >= 0f)
                {
                    if (pi == 0) { A = v[i]; dA = d[i]; } else { D_ = v[i]; dD = d[i]; }
                    pi++;
                }
                else
                {
                    if (ni == 0) { B = v[i]; dB = d[i]; } else { C = v[i]; dC = d[i]; }
                    ni++;
                }
            }

            var (e1p, e1n) = EP(A,  B, dA, dB);
            var (f1p, f1n) = EP(A,  C, dA, dC);
            var (g1p, g1n) = EP(D_, C, dD, dC);
            var (h1p, h1n) = EP(D_, B, dD, dB);

            // 需求②：记录切口四边形 {e1,f1,g1,h1} 内部的邻接关系，供末尾切向松弛使用
            LinkCutAdjacency(e1p, f1p);
            LinkCutAdjacency(f1p, g1p);
            LinkCutAdjacency(g1p, h1p);
            LinkCutAdjacency(h1p, e1p);
            LinkCutAdjacency(e1n, f1n);
            LinkCutAdjacency(f1n, g1n);
            LinkCutAdjacency(g1n, h1n);
            LinkCutAdjacency(h1n, e1n);

            // 正侧 (A, D_)
            // ★ 全局一致性三角化：严格对齐 min-max ID
            AT(A, e1p, f1p, g1p);
            AT(A, e1p, g1p, h1p);
            AT(A, D_, g1p, h1p);

            // 负侧 (B, C)
            AT(B, e1n, f1n, g1n);
            AT(B, e1n, g1n, h1n);
            AT(B, C, f1n, g1n);
        }

        // ═══════════════════════════════════════════════════════
        // ★ Phase 2: Edge Pair with Trajectory Correction
        //
        // 论文 §4.3: "snapping intersection points within a threshold
        //             distance to their nearest vertex"
        //
        // 当 t < SNAP_T → 吸附到 va (正侧端点)
        // 当 t > 1-SNAP_T → 吸附到 vb (负侧端点)
        // 否则 → 在交点位置创建双顶点 (posV/negV)
        // ═══════════════════════════════════════════════════════
        (int posV, int negV) EP(int va, int vb, float da, float db)
        {
            long key = EK(va, vb);

            // 1. 帧级 split 缓存：同帧内同边只切一次
            if (_frameSplitCache.TryGetValue(key, out var frameCached)) return frameCached.Pair;

            float t = da / (da - db);
            int snapVertex = GetSnapVertex(va, vb, da, db, t);

            // 2. 划刀级缓存：原始边跨帧拓扑复用
            bool isOriginalEdge = (va < _originalVertCount) && (vb < _originalVertCount);
            if (isOriginalEdge && _strokeSplitCache.TryGetValue(key, out var strokeCached))
            {
                if (CanReuseStrokeSplit(strokeCached, va, vb, t, snapVertex, _currentCutNormal))
                {
                    _frameSplitCache[key] = strokeCached;
                    return strokeCached.Pair;
                }

                _strokeSplitCache.Remove(key);
            }

            (int posV, int negV) result;

            if (snapVertex == va)
            {
                // ★ 核心修复：snap 时同一帧内同一顶点的克隆必须复用！
                if (!_frameSnapCache.TryGetValue(va, out int va_dup))
                {
                    va_dup = Clone(va);
                    _frameSnapCache[va] = va_dup;
                    // 需求①：va_dup 代表 da<0 那一侧的复制体，朝该侧法线方向微偏移，避免与 va 完全重合
                    OffsetClone(va_dup, da >= 0f ? -_currentCutNormal : _currentCutNormal);
                }
                result = da >= 0f ? (va, va_dup) : (va_dup, va);
            }
            else if (snapVertex == vb)
            {
                // 同理，snap 到 vb 时也必须复用同一克隆
                if (!_frameSnapCache.TryGetValue(vb, out int vb_dup))
                {
                    vb_dup = Clone(vb);
                    _frameSnapCache[vb] = vb_dup;
                    // 需求①：vb_dup 代表 db<0 那一侧的复制体，朝该侧法线方向微偏移
                    OffsetClone(vb_dup, db >= 0f ? -_currentCutNormal : _currentCutNormal);
                }
                result = db >= 0f ? (vb, vb_dup) : (vb_dup, vb);
            }
            else
            {
                // 正常情况：在边的交点处创建两个重合的双顶点
                Vector3 ip  = Vector3.Lerp(_data.Positions[va],     _data.Positions[vb],     t);
                Vector3 ir  = Vector3.Lerp(_data.RestPositions[va], _data.RestPositions[vb], t);
                Vector3 iv  = Vector3.Lerp(_data.Velocities[va],    _data.Velocities[vb],    t);
                Vector3 ipv = Vector3.Lerp(_data.PrevPositions[va], _data.PrevPositions[vb], t);
                float imA = _data.InvMass[va], imB = _data.InvMass[vb];
                float im  = (imA == 0f || imB == 0f) ? 0f : Mathf.Lerp(imA, imB, t);

                int posV = _data.AddParticle(ip, iv, im);
                RecordVertexBirthPhase(posV);
                _data.RestPositions[posV] = ir; _data.PrevPositions[posV] = ipv;
                int negV = _data.AddParticle(ip, iv, im);
                RecordVertexBirthPhase(negV);
                _data.RestPositions[negV] = ir; _data.PrevPositions[negV] = ipv;
                ApplyEdgeSurfaceSupport(posV, va, vb);
                ApplyEdgeSurfaceSupport(negV, va, vb);
                result = (posV, negV);

                // 需求②：记录本次新建的真插值切割顶点，供 Cut() 末尾切向松弛
                _newInterpVertsThisCut.Add(posV);
                _newInterpVertsThisCut.Add(negV);
            }

            SplitCacheValue cacheValue = new SplitCacheValue
            {
                posV = result.posV,
                negV = result.negV,
                t = t,
                normal = _currentCutNormal,
                snapVertex = snapVertex
            };

            _frameSplitCache[key] = cacheValue;
            if (isOriginalEdge) _strokeSplitCache[key] = cacheValue;
            return result;
        }

        int GetSnapVertex(int va, int vb, float da, float db, float t)
        {
            if (t <= SNAP_T || Mathf.Abs(da) <= SNAP_PLANE_DIST) return va;
            if (t >= 1f - SNAP_T || Mathf.Abs(db) <= SNAP_PLANE_DIST) return vb;
            return -1;
        }

        bool CanReuseStrokeSplit(SplitCacheValue cached, int va, int vb, float t, int snapVertex, Vector3 normal)
        {
            const float maxTDrift = 0.08f;
            const float maxPosDrift = 0.003f;
            const float minNormalDot = 0.94f;

            if (cached.snapVertex >= 0 || snapVertex >= 0)
                return cached.snapVertex == snapVertex && snapVertex >= 0;

            if (Mathf.Abs(cached.t - t) > maxTDrift) return false;
            if (cached.normal != Vector3.zero && normal != Vector3.zero &&
                Vector3.Dot(cached.normal.normalized, normal.normalized) < minNormalDot)
                return false;

            Vector3 currentIp = Vector3.Lerp(_data.Positions[va], _data.Positions[vb], t);
            Vector3 cachedIp = (_data.Positions[cached.posV] + _data.Positions[cached.negV]) * 0.5f;
            return (currentIp - cachedIp).sqrMagnitude <= maxPosDrift * maxPosDrift;
        }

        int Clone(int s)
        {
            int i = _data.AddParticle(_data.Positions[s], _data.Velocities[s], _data.InvMass[s]);
            RecordVertexBirthPhase(i);
            _data.RestPositions[i]  = _data.RestPositions[s];
            _data.PrevPositions[i] = _data.PrevPositions[s];
            CopySurfaceSupport(i, s);
            return i;
        }

        // ════════════════════════════════════════════════════════════════════
        // 需求①：克隆体微偏移 — 解决"看起来仍连着"的藕断丝连假象
        //
        // Clone() 创建的新顶点与源顶点在 Positions/PrevPositions/Velocities 上完全重合。
        // 拓扑上两者已经是不同的、可独立运动的质点（分别归属不同的 tet），但若没有任何
        // 外力把它们朝不同方向拉开，XPBD 求解器没有梯度去把它们分开——视觉上仍然是一条
        // "缝合"得天衣无缝的接缝，给人藕断丝连的错觉，即使连通分量审计显示拓扑已经分离。
        //
        // 这里只移动 Positions 和 PrevPositions（同步偏移，不引入虚假速度），不改 RestPositions，
        // 不影响质量/刚度/体积积分，只是给"已经被判定为分离"的两个质点一个看得见的初始间隙。
        // ════════════════════════════════════════════════════════════════════
        void OffsetClone(int clone, Vector3 direction)
        {
            if (direction.sqrMagnitude < 1e-12f) return;
            Vector3 offset = direction.normalized * CLONE_SEPARATION_EPS;
            _data.Positions[clone]     += offset;
            _data.PrevPositions[clone] += offset;
        }

        // 分离阶段（SeparateDirtyVerticesOnePass）没有单一法线可用：一个 pivot 可能被拆成
        // 2 个以上的连通分量。这里改用几何方式——把 clone 推向"它自己那个分量"的质心方向，
        // 即推离原 pivot、靠近它实际所属的那块材料，方向语义上总是正确的。
        void OffsetCloneAwayFromComponent(int clone, int pivot, List<int> incidentTets, int[] compId, int n, int targetComp)
        {
            Vector3 centroidSum = Vector3.zero;
            int count = 0;
            for (int i = 0; i < n; i++)
            {
                if (compId[i] != targetComp) continue;
                int tet = incidentTets[i];
                int b = tet * 4;
                for (int k = 0; k < 4; k++)
                {
                    int v = _data.TetIds[b + k];
                    if (v == pivot) continue;
                    centroidSum += _data.Positions[v];
                    count++;
                }
            }
            if (count == 0) return;
            Vector3 centroid = centroidSum / count;
            OffsetClone(clone, centroid - _data.Positions[pivot]);
        }

        // ════════════════════════════════════════════════════════════════════
        // 需求②：新切割面顶点切向松弛 — Zeng & Courtecuisse 2025 思路的轻量版
        //
        // 细分法的切口走向严格跟随四面体棱边，分辨率受限于体网格密度，本身就比真实刀口
        // 粗糙。该方法把"吸附阈值"重新表述成一个小型质量优化问题：让每个新切割顶点朝它
        // 在同一切口多边形内的邻居均值靠拢，但被严格限制在切割平面内移动（法线方向分量
        // 被投影掉），因此不会改变任何顶点的分割侧、不会引入穿透，只是让切口线少一些
        // 因为逐 tet 独立插值而产生的局部锯齿。Positions 与 RestPositions 同步松弛同一增量，
        // 不会在新切割面上人为引入额外的弹性应变。
        // ════════════════════════════════════════════════════════════════════
        void SmoothNewCutSurfaceVertices(Vector3 planeNormal)
        {
            if (_newInterpVertsThisCut.Count == 0) return;
            if (planeNormal.sqrMagnitude < 1e-8f) return;
            Vector3 n = planeNormal.normalized;

            for (int idx = 0; idx < _newInterpVertsThisCut.Count; idx++)
            {
                int v = _newInterpVertsThisCut[idx];
                if (v < 0 || v >= _data.NumParticles) continue;
                if (_data.InvMass[v] <= 0f) continue; // 固定点不挪动
                if (!_cutVertexAdjacency.TryGetValue(v, out List<int> neighbors) || neighbors.Count == 0)
                    continue;

                Vector3 avgPos = Vector3.zero, avgRest = Vector3.zero;
                int cnt = 0;
                for (int i = 0; i < neighbors.Count; i++)
                {
                    int nb = neighbors[i];
                    if (nb < 0 || nb >= _data.NumParticles) continue;
                    avgPos  += _data.Positions[nb];
                    avgRest += _data.RestPositions[nb];
                    cnt++;
                }
                if (cnt == 0) continue;
                avgPos  /= cnt;
                avgRest /= cnt;

                Vector3 deltaPos = avgPos - _data.Positions[v];
                deltaPos -= Vector3.Dot(deltaPos, n) * n; // 只保留切向分量
                Vector3 deltaRest = avgRest - _data.RestPositions[v];
                deltaRest -= Vector3.Dot(deltaRest, n) * n;

                _data.Positions[v]     += deltaPos  * CUT_SURFACE_SMOOTH_FACTOR;
                _data.PrevPositions[v] += deltaPos  * CUT_SURFACE_SMOOTH_FACTOR;
                _data.RestPositions[v] += deltaRest * CUT_SURFACE_SMOOTH_FACTOR;
            }
        }

        void LinkCutAdjacency(int a, int b)
        {
            if (a == b || a < 0 || b < 0) return;
            AddAdjacency(a, b);
            AddAdjacency(b, a);
        }

        void AddAdjacency(int from, int to)
        {
            if (!_cutVertexAdjacency.TryGetValue(from, out List<int> list))
            {
                list = new List<int>(4);
                _cutVertexAdjacency[from] = list;
            }
            if (!list.Contains(to)) list.Add(to);
        }

        void AdvanceCutPhase()
        {
            _cutPhaseId++;
        }

        void RecordVertexBirthPhase(int vertex)
        {
            while (_vertexBirthPhase.Count <= vertex)
                _vertexBirthPhase.Add(_cutPhaseId);
            _vertexBirthPhase[vertex] = _cutPhaseId;
        }

        int VertexBirthPhase(int vertex)
        {
            if (vertex < 0 || vertex >= _vertexBirthPhase.Count)
                return _cutPhaseId;
            return _vertexBirthPhase[vertex];
        }

        /// <summary>添加一个 tet, 自动修正绕序, 始终创建</summary>
        void BuildOriginalSurfaceSupports()
        {
            ClearSurfaceSupports();
            if (_data == null) return;

            for (int v = 0; v < _data.NumParticles; v++)
                SetSurfaceSupport(v, v, -1, -1);

            int[] tris = _data.SurfaceTriIds;
            if (tris == null) return;

            for (int i = 0; i + 2 < tris.Length; i += 3)
            {
                int a = tris[i + 0];
                int b = tris[i + 1];
                int c = tris[i + 2];
                if (!ValidOriginalParticle(a) || !ValidOriginalParticle(b) || !ValidOriginalParticle(c))
                    continue;

                _originalSurfaceFaceKeys.Add(SurfaceReconstructor.FaceKey(a, b, c));
                _originalSurfaceEdgeKeys.Add(SurfaceReconstructor.EdgeKey(a, b));
                _originalSurfaceEdgeKeys.Add(SurfaceReconstructor.EdgeKey(b, c));
                _originalSurfaceEdgeKeys.Add(SurfaceReconstructor.EdgeKey(c, a));
            }
        }

        bool ValidOriginalParticle(int v) => v >= 0 && v < _originalVertCount;

        void ClearSurfaceSupports()
        {
            _surfaceSupport0.Clear();
            _surfaceSupport1.Clear();
            _surfaceSupport2.Clear();
            _originalSurfaceFaceKeys.Clear();
            _originalSurfaceEdgeKeys.Clear();
        }

        void EnsureSurfaceSupportSlot(int v)
        {
            while (_surfaceSupport0.Count <= v)
            {
                _surfaceSupport0.Add(-1);
                _surfaceSupport1.Add(-1);
                _surfaceSupport2.Add(-1);
            }
        }

        void SetSurfaceSupport(int v, int s0, int s1, int s2)
        {
            if (v < 0) return;
            EnsureSurfaceSupportSlot(v);
            _surfaceSupport0[v] = s0;
            _surfaceSupport1[v] = s1;
            _surfaceSupport2[v] = s2;
        }

        void CopySurfaceSupport(int dst, int src)
        {
            if (src < 0 || src >= _surfaceSupport0.Count)
            {
                SetSurfaceSupport(dst, -1, -1, -1);
                return;
            }

            SetSurfaceSupport(dst, _surfaceSupport0[src], _surfaceSupport1[src], _surfaceSupport2[src]);
        }

        void ApplyEdgeSurfaceSupport(int dst, int a, int b)
        {
            if (TryBuildEdgeSurfaceSupport(a, b, out int s0, out int s1, out int s2))
                SetSurfaceSupport(dst, s0, s1, s2);
            else
                SetSurfaceSupport(dst, -1, -1, -1);
        }

        bool TryBuildEdgeSurfaceSupport(int a, int b, out int s0, out int s1, out int s2)
        {
            s0 = -1;
            s1 = -1;
            s2 = -1;
            int count = 0;
            if (!AppendVertexSurfaceSupports(a, ref s0, ref s1, ref s2, ref count)) return false;
            if (!AppendVertexSurfaceSupports(b, ref s0, ref s1, ref s2, ref count)) return false;

            if (count == 1) return true;
            if (count == 2)
                return _originalSurfaceEdgeKeys.Contains(SurfaceReconstructor.EdgeKey(s0, s1));
            if (count == 3)
                return _originalSurfaceFaceKeys.Contains(SurfaceReconstructor.FaceKey(s0, s1, s2));
            return false;
        }

        bool AppendVertexSurfaceSupports(int vertex, ref int s0, ref int s1, ref int s2, ref int count)
        {
            if (!AppendSurfaceSupport(SupportOf(_surfaceSupport0, vertex), ref s0, ref s1, ref s2, ref count))
                return false;
            if (!AppendSurfaceSupport(SupportOf(_surfaceSupport1, vertex), ref s0, ref s1, ref s2, ref count))
                return false;
            if (!AppendSurfaceSupport(SupportOf(_surfaceSupport2, vertex), ref s0, ref s1, ref s2, ref count))
                return false;
            return true;
        }

        static int SupportOf(List<int> supports, int vertex)
        {
            return vertex >= 0 && vertex < supports.Count ? supports[vertex] : -1;
        }

        static bool AppendSurfaceSupport(int root, ref int s0, ref int s1, ref int s2, ref int count)
        {
            if (root < 0) return true;
            if ((count > 0 && s0 == root) ||
                (count > 1 && s1 == root) ||
                (count > 2 && s2 == root))
            {
                return true;
            }

            if (count >= 3) return false;
            if (count == 0) s0 = root;
            else if (count == 1) s1 = root;
            else s2 = root;
            count++;
            return true;
        }

        void AT(int a, int b, int c, int d)
        {
            // 退化/重复顶点检查
            if (a == b || a == c || a == d || b == c || b == d || c == d) return;

            // ★ 修复：必须使用 RestPositions 来判断绕序！
            // 如果使用变形后的 Positions，当网格在物理受力挤压时切开，可能会算出一个错误的相反绕序。
            // 等物理引擎将其恢复原状时，它就会永久翻转（Inside-out），导致法线朝内，渲染出黑色的背面！
            Vector3 p0 = _data.RestPositions[a], p1 = _data.RestPositions[b];
            Vector3 p2 = _data.RestPositions[c], p3 = _data.RestPositions[d];
            float sv = Vector3.Dot(Vector3.Cross(p1 - p0, p2 - p0), p3 - p0);

            // 体积过滤: 丢弃退化子tet → 减少碎屑
            if (Mathf.Abs(sv) < MIN_TET_VOL) return;

            if (sv < 0f) _data.AddTet(a, c, b, d);
            else         _data.AddTet(a, b, c, d);
        }

        // ══════════════════════════════════════════════════════
        static bool TetTouchesSweptBlade(
            Vector3 v0, Vector3 v1, Vector3 v2, Vector3 v3,
            Vector3 A0, Vector3 B0, Vector3 A1, Vector3 B1,
            float bladeRadius)
        {
            Vector3 center = (v0 + v1 + v2 + v3) * 0.25f;
            float tetRadius = Mathf.Sqrt(Mathf.Max(
                Mathf.Max((v0 - center).sqrMagnitude, (v1 - center).sqrMagnitude),
                Mathf.Max((v2 - center).sqrMagnitude, (v3 - center).sqrMagnitude)));
            float limit = tetRadius + bladeRadius;
            return DistanceToSweptBladeSq(center, A0, B0, A1, B1) <= limit * limit;
        }

        static float DistanceToSweptBladeSq(Vector3 p, Vector3 A0, Vector3 B0, Vector3 A1, Vector3 B1)
        {
            float d0 = PointTriangleDistanceSq(p, A0, A1, B1);
            float d1 = PointTriangleDistanceSq(p, A0, B1, B0);
            return Mathf.Min(d0, d1);
        }

        static float PointSegmentDistanceSq(Vector3 p, Vector3 a, Vector3 b)
        {
            Vector3 ab = b - a;
            float sq = ab.sqrMagnitude;
            if (sq < 1e-12f) return (p - a).sqrMagnitude;
            float t = Mathf.Clamp01(Vector3.Dot(p - a, ab) / sq);
            return (p - (a + ab * t)).sqrMagnitude;
        }

        static float PointTriangleDistanceSq(Vector3 p, Vector3 a, Vector3 b, Vector3 c)
        {
            Vector3 ab = b - a;
            Vector3 ac = c - a;
            Vector3 n = Vector3.Cross(ab, ac);
            float nSq = n.sqrMagnitude;
            if (nSq < 1e-12f)
            {
                return Mathf.Min(
                    PointSegmentDistanceSq(p, a, b),
                    Mathf.Min(PointSegmentDistanceSq(p, b, c), PointSegmentDistanceSq(p, c, a)));
            }

            float signed = Vector3.Dot(p - a, n) / Mathf.Sqrt(nSq);
            Vector3 projected = p - n * (Vector3.Dot(p - a, n) / nSq);
            if (PointInTriangle(projected, a, b, c, n))
                return signed * signed;

            return Mathf.Min(
                PointSegmentDistanceSq(p, a, b),
                Mathf.Min(PointSegmentDistanceSq(p, b, c), PointSegmentDistanceSq(p, c, a)));
        }

        static bool PointInTriangle(Vector3 p, Vector3 a, Vector3 b, Vector3 c, Vector3 n)
        {
            return Vector3.Dot(Vector3.Cross(b - a, p - a), n) >= 0f &&
                   Vector3.Dot(Vector3.Cross(c - b, p - b), n) >= 0f &&
                   Vector3.Dot(Vector3.Cross(a - c, p - c), n) >= 0f;
        }

        int SeparateDirtyVertices(Vector3 planeCenter, Vector3 planeNormal)
        {
            int separated = 0;
            LastForcedSeparationCount = 0;
            LastResidualSharedVertexCount = 0;
            LastSeparationPassCount = 0;

            for (int pass = 0; pass < MAX_SEPARATION_PASSES; pass++)
            {
                int passSeparated = SeparateDirtyVerticesOnePass(planeCenter, planeNormal);
                if (passSeparated == 0)
                    break;

                separated += passSeparated;
                LastSeparationPassCount++;
            }

            // 2025 paper §4.3: separation is decided SOLELY by the shared-vertex
            // connected-components pass above. The previous plane-side force-split
            // (ForceSeparateResidualSideSharing) overrode that connectivity by plane
            // side — the "spurious separation" the paper explicitly prevents — so it
            // is removed. If the shared-vertex CC still keeps tets together, they ARE
            // genuinely connected (real material / attached groove), not 藕断丝连.
            // The plane-side residual audit (CountResidualSideSharing) is likewise
            // dropped; a faithful global shared-vertex connected-components audit is
            // added separately (see AuditMaterialComponents).
            LastForcedSeparationCount = 0;
            LastResidualSharedVertexCount = 0;

            if (separated > 0)
                _dirty = true;

            return separated;
        }

        // ════════════════════════════════════════════════════════════════════
        // 迭代2.1：全局材料连通分量审计 (忠于 2024 §2.2.1 AFCC + 2025 §4.3 shared-vertex)
        // 2024 §2.2.1: "the disconnected clusters are connected components of graph G
        //   ... using the finding connected components algorithm."
        // 邻接关系 = 2025 §4.3 shared-vertex：两个 active tet 相连 ⟺ 共享 ≥1 顶点。
        // 用途：(1) 闭合/半闭合切口"是否成为独立 component"判定 (需求5)；
        //       (2) 藕断丝连审计：被分离的 (orig→clone) 若仍落在同一分量 = 残余桥 (需求1)；
        //       (3) 分量大小供后续 expansion/merge (需求6/7)。
        // 只读审计，不改拓扑；仅在真正发生切割时调用，O(active tets) 并查集。
        // ════════════════════════════════════════════════════════════════════
        void AuditMaterialComponents()
        {
            LastComponentCount = 0;
            LastSeparatedComponentCount = 0;
            LastIndependentBlockCount = 0;
            LastFragmentComponentCount = 0;
            LastResidualBridgeCount = 0;
            LastLargestComponentTetCount = 0;
            if (_data == null) return;

            int nt = _data.NumTets;
            int np = _data.NumParticles;
            if (nt <= 0) return;

            if (_ufParent.Length < nt)
            {
                int cap = Mathf.NextPowerOfTwo(nt);
                _ufParent = new int[cap];
                _ufSize = new int[cap];
                _tetComponentRoot = new int[cap];
            }
            if (_firstTetOfParticle.Length < np)
                _firstTetOfParticle = new int[Mathf.NextPowerOfTwo(np)];

            for (int t = 0; t < nt; t++) { _ufParent[t] = t; _ufSize[t] = 1; }
            for (int p = 0; p < np; p++) _firstTetOfParticle[p] = -1;

            // Union active tets that share a vertex (2025 §4.3 shared-vertex relation).
            for (int t = 0; t < nt; t++)
            {
                if (!_data.TetActive[t]) continue;
                int b = t * 4;
                for (int i = 0; i < 4; i++)
                {
                    int v = _data.TetIds[b + i];
                    if (v < 0 || v >= np) continue;
                    int first = _firstTetOfParticle[v];
                    if (first < 0) _firstTetOfParticle[v] = t;
                    else UfUnion(first, t);
                }
            }

            // Re-tally active-tet count per root (reuse _ufSize after union completes).
            for (int t = 0; t < nt; t++) _ufSize[t] = 0;
            for (int t = 0; t < nt; t++)
            {
                if (!_data.TetActive[t]) continue;
                int r = UfFind(t);
                _tetComponentRoot[t] = r;
                _ufSize[r]++;
            }

            int componentCount = 0, largestSize = 0, largestRoot = -1;
            for (int t = 0; t < nt; t++)
            {
                int s = _ufSize[t];
                if (s <= 0) continue;
                componentCount++;
                if (s > largestSize) { largestSize = s; largestRoot = t; }
            }

            int blocks = 0, fragments = 0;
            for (int t = 0; t < nt; t++)
            {
                int s = _ufSize[t];
                if (s <= 0 || t == largestRoot) continue;
                if (s >= INDEPENDENT_BLOCK_MIN_TETS) blocks++;
                else fragments++;
            }

            // 藕断丝连审计：本 stroke 每个被分离的 (orig→clone)，
            // 若两者经其它 shared-vertex 路径仍连回同一连通分量 = 残余桥(应为 0)。
            int residual = 0;
            for (int k = 0; k < _strokeSepOrig.Count; k++)
            {
                int ro = ParticleRoot(_strokeSepOrig[k]);
                int rc = ParticleRoot(_strokeSepClone[k]);
                if (ro >= 0 && rc >= 0 && ro == rc) residual++;
            }

            LastComponentCount = componentCount;
            LastLargestComponentTetCount = largestSize;
            LastSeparatedComponentCount = Mathf.Max(0, componentCount - 1);
            LastIndependentBlockCount = blocks;
            LastFragmentComponentCount = fragments;
            LastResidualBridgeCount = residual;
        }

        int UfFind(int x)
        {
            while (_ufParent[x] != x)
            {
                _ufParent[x] = _ufParent[_ufParent[x]]; // path halving
                x = _ufParent[x];
            }
            return x;
        }

        void UfUnion(int a, int b)
        {
            int ra = UfFind(a), rb = UfFind(b);
            if (ra == rb) return;
            if (_ufSize[ra] < _ufSize[rb]) { int tmp = ra; ra = rb; rb = tmp; }
            _ufParent[rb] = ra;
            _ufSize[ra] += _ufSize[rb];
        }

        int ParticleRoot(int p)
        {
            if (p < 0 || p >= _firstTetOfParticle.Length) return -1;
            int t = _firstTetOfParticle[p];
            return (t >= 0 && t < _ufParent.Length) ? UfFind(t) : -1;
        }

        // 供 HUD / 渲染按分量着色或标记"切下的块"使用。
        public int TetComponentRoot(int tet)
        {
            if (tet < 0 || tet >= _tetComponentRoot.Length) return -1;
            // 防过期：非 active tet 的 root 是上一帧残留值，返回 -1（审核 D 指出的潜在隐患）。
            if (_data != null && _data.TetActive != null && tet < _data.TetActive.Length && !_data.TetActive[tet])
                return -1;
            return _tetComponentRoot[tet];
        }

        int SeparateDirtyVerticesOnePass(Vector3 planeCenter, Vector3 planeNormal)
        {
            int separated = 0;
            BuildDirtyIncidentMap();

            foreach (int pivot in _dirtyCutVertices)
            {
                if (pivot < 0 || pivot >= _data.NumParticles) continue;
                if (pivot >= _originalVertCount && VertexBirthPhase(pivot) == _cutPhaseId)
                {
                    LastSkippedCurrentPhasePivotCount++;
                    continue;
                }

                if (!_dirtyIncidentMap.TryGetValue(pivot, out List<int> incidentTets))
                    continue;

                int n = incidentTets.Count;
                if (n < 2) continue;

                int[] compId = new int[n];
                for (int i = 0; i < n; i++) compId[i] = -1;

                _componentScratch.Clear();
                int compCount = 0;
                for (int i = 0; i < n; i++)
                {
                    if (compId[i] >= 0) continue;

                    int size = 0;
                    compId[i] = compCount;
                    _componentQueue.Clear();
                    _componentQueue.Enqueue(i);

                    while (_componentQueue.Count > 0)
                    {
                        int cur = _componentQueue.Dequeue();
                        size++;

                        for (int j = 0; j < n; j++)
                        {
                            if (compId[j] >= 0) continue;
                            if (!CanConnectThroughPivot(pivot, incidentTets[cur], incidentTets[j], planeCenter, planeNormal))
                                continue;

                            compId[j] = compCount;
                            _componentQueue.Enqueue(j);
                        }
                    }

                    _componentScratch.Add(size);
                    compCount++;
                }

                if (compCount <= 1) continue;

                int keepComp = 0;
                int keepSize = _componentScratch[0];
                for (int c = 1; c < compCount; c++)
                {
                    if (_componentScratch[c] > keepSize)
                    {
                        keepComp = c;
                        keepSize = _componentScratch[c];
                    }
                }

                for (int c = 0; c < compCount; c++)
                {
                    if (c == keepComp) continue;

                    int clone = Clone(pivot);
                    OffsetCloneAwayFromComponent(clone, pivot, incidentTets, compId, n, c); // 需求①：微偏移防藕断丝连
                    _strokeSepOrig.Add(pivot);   // 迭代2.1：记录分离事件，供全局连通分量审计藕断丝连
                    _strokeSepClone.Add(clone);
                    separated++;
                    for (int i = 0; i < n; i++)
                    {
                        if (compId[i] == c)
                            RewriteTetVertex(incidentTets[i], pivot, clone);
                    }
                }
            }

            return separated;
        }

        int ForceSeparateResidualSideSharing(Vector3 planeCenter, Vector3 planeNormal)
        {
            int forced = 0;
            BuildDirtyIncidentMap();

            foreach (int pivot in _dirtyCutVertices)
            {
                if (pivot < 0 || pivot >= _data.NumParticles) continue;
                if (pivot >= _originalVertCount && VertexBirthPhase(pivot) == _cutPhaseId)
                    continue;

                if (!_dirtyIncidentMap.TryGetValue(pivot, out List<int> incidentTets))
                    continue;

                int posCount = 0;
                int negCount = 0;
                for (int i = 0; i < incidentTets.Count; i++)
                {
                    int side = TetSideForSeparationAudit(pivot, incidentTets[i], planeCenter, planeNormal);
                    if (side > 0) posCount++;
                    else if (side < 0) negCount++;
                }

                if (posCount == 0 || negCount == 0)
                    continue;

                int splitSide = posCount <= negCount ? 1 : -1;
                int clone = Clone(pivot);
                forced++;

                for (int i = 0; i < incidentTets.Count; i++)
                {
                    int tet = incidentTets[i];
                    if (TetSideForSeparationAudit(pivot, tet, planeCenter, planeNormal) == splitSide)
                        RewriteTetVertex(tet, pivot, clone);
                }
            }

            return forced;
        }

        int CountResidualSideSharing(Vector3 planeCenter, Vector3 planeNormal)
        {
            int residual = 0;
            BuildDirtyIncidentMap();

            foreach (int pivot in _dirtyCutVertices)
            {
                if (pivot < 0 || pivot >= _data.NumParticles) continue;
                if (pivot >= _originalVertCount && VertexBirthPhase(pivot) == _cutPhaseId)
                    continue;

                if (!_dirtyIncidentMap.TryGetValue(pivot, out List<int> incidentTets))
                    continue;

                bool hasPos = false;
                bool hasNeg = false;
                for (int i = 0; i < incidentTets.Count; i++)
                {
                    int side = TetSideForSeparationAudit(pivot, incidentTets[i], planeCenter, planeNormal);
                    if (side > 0) hasPos = true;
                    else if (side < 0) hasNeg = true;

                    if (hasPos && hasNeg)
                    {
                        residual++;
                        break;
                    }
                }
            }

            return residual;
        }

        void BuildDirtyIncidentMap()
        {
            _dirtyIncidentMap.Clear();
            foreach (int pivot in _dirtyCutVertices)
                _dirtyIncidentMap[pivot] = new List<int>(32);

            for (int t = 0; t < _data.NumTets; t++)
            {
                if (!_data.TetActive[t]) continue;

                int b = t * 4;
                for (int i = 0; i < 4; i++)
                {
                    int v = _data.TetIds[b + i];
                    if (_dirtyIncidentMap.TryGetValue(v, out List<int> incident))
                        incident.Add(t);
                }
            }
        }

        bool CanConnectThroughPivot(int pivot, int tetA, int tetB, Vector3 planeCenter, Vector3 planeNormal)
        {
            // 2025 paper §4.3 (Fig.5) shared-vertex criterion ONLY.
            // Two tets incident to `pivot` (the shared cut intersection point) remain
            // connected iff they share at least one OTHER vertex. Subdivision already
            // duplicates intersection points per side (posV/negV) and the per-frame
            // split cache makes same-side neighbours reuse the SAME duplicated vertex,
            // so a shared non-pivot vertex == same material side. The previous
            // plane-side test (sideA==sideB) is exactly what the paper repudiates for
            // non-planar / trajectory-corrected cuts; it caused 藕断丝连 + 碎屑. Removed.
            return ShareVertexExcludingPivot(pivot, tetA, tetB);
        }

        bool ShareVertexExcludingPivot(int pivot, int tetA, int tetB)
        {
            int a = tetA * 4;
            int b = tetB * 4;
            for (int ia = 0; ia < 4; ia++)
            {
                int va = _data.TetIds[a + ia];
                if (va == pivot) continue;

                for (int ib = 0; ib < 4; ib++)
                {
                    int vb = _data.TetIds[b + ib];
                    if (vb != pivot && va == vb) return true;
                }
            }
            return false;
        }

        int TetSideExcludingPivot(int pivot, int tet, Vector3 planeCenter, Vector3 planeNormal)
        {
            int pos = 0, neg = 0;
            int b = tet * 4;
            for (int i = 0; i < 4; i++)
            {
                int v = _data.TetIds[b + i];
                if (v == pivot) continue;

                float d = Vector3.Dot(_data.Positions[v] - planeCenter, planeNormal);
                if (d > SIDE_EPS) pos++;
                else if (d < -SIDE_EPS) neg++;
            }

            if (pos > 0 && neg == 0) return 1;
            if (neg > 0 && pos == 0) return -1;
            return 0;
        }

        int TetSideForSeparationAudit(int pivot, int tet, Vector3 planeCenter, Vector3 planeNormal)
        {
            int pos = 0, neg = 0, count = 0;
            float sum = 0f;
            int b = tet * 4;
            for (int i = 0; i < 4; i++)
            {
                int v = _data.TetIds[b + i];
                if (v == pivot) continue;

                float d = Vector3.Dot(_data.Positions[v] - planeCenter, planeNormal);
                sum += d;
                count++;
                if (d > SIDE_EPS) pos++;
                else if (d < -SIDE_EPS) neg++;
            }

            if (pos > 0 && neg == 0) return 1;
            if (neg > 0 && pos == 0) return -1;

            if (count <= 0) return 0;
            float avg = sum / count;
            if (avg > SIDE_EPS) return 1;
            if (avg < -SIDE_EPS) return -1;
            return 0;
        }

        void RewriteTetVertex(int tet, int oldVertex, int newVertex)
        {
            int b = tet * 4;
            for (int i = 0; i < 4; i++)
            {
                if (_data.TetIds[b + i] == oldVertex)
                    _data.TetIds[b + i] = newVertex;
            }
        }

        public int RemoveStretchedTets(float maxStretchRatio = 5.0f, float minAbsoluteLength = 0.003f)
        {
            if (_data == null) return 0;
            int removedCount = 0;

            float sqrMaxRatio = maxStretchRatio * maxStretchRatio;
            float sqrMinLen = minAbsoluteLength * minAbsoluteLength;

            for (int t = 0; t < _data.NumTets; t++)
            {
                if (!_data.TetActive[t]) continue;

                if (_data.RestVolumes != null &&
                    t < _data.RestVolumes.Length &&
                    _data.RestVolumes[t] > 0f &&
                    _data.RestVolumes[t] < SLIVER_REST_VOLUME)
                {
                    _data.TetActive[t] = false;
                    removedCount++;
                    _dirty = true;
                    continue;
                }

                int b = t * 4;
                int v0 = _data.TetIds[b + 0];
                int v1 = _data.TetIds[b + 1];
                int v2 = _data.TetIds[b + 2];
                int v3 = _data.TetIds[b + 3];

                Vector3 p0 = _data.Positions[v0];
                Vector3 p1 = _data.Positions[v1];
                Vector3 p2 = _data.Positions[v2];
                Vector3 p3 = _data.Positions[v3];

                Vector3 r0 = _data.RestPositions[v0];
                Vector3 r1 = _data.RestPositions[v1];
                Vector3 r2 = _data.RestPositions[v2];
                Vector3 r3 = _data.RestPositions[v3];

                bool isStretched = false;

                bool CheckEdge(Vector3 cpA, Vector3 cpB, Vector3 crA, Vector3 crB)
                {
                    float currentSqr = (cpA - cpB).sqrMagnitude;
                    float restSqr = (crA - crB).sqrMagnitude;

                    if (restSqr < 1e-12f) return currentSqr > sqrMinLen; // Zero rest length but stretched

                    float ratio = currentSqr / restSqr;
                    // 如果拉伸超过 10 倍（平方100倍），不管它多短，绝对是拓扑错误（藕断丝连），必须斩断！
                    if (ratio > 100.0f) return true;

                    if (currentSqr < sqrMinLen) return false; // Ignore short edges
                    return ratio > sqrMaxRatio;
                }

                if (CheckEdge(p0, p1, r0, r1)) isStretched = true;
                else if (CheckEdge(p0, p2, r0, r2)) isStretched = true;
                else if (CheckEdge(p0, p3, r0, r3)) isStretched = true;
                else if (CheckEdge(p1, p2, r1, r2)) isStretched = true;
                else if (CheckEdge(p1, p3, r1, r3)) isStretched = true;
                else if (CheckEdge(p2, p3, r2, r3)) isStretched = true;

                if (isStretched)
                {
                    _data.TetActive[t] = false;
                    removedCount++;
                    _dirty = true;
                }
            }

            if (removedCount > 0)
            {
                Debug.Log($"[TetSubdivisionCutter] 清理藕断丝连: 已熔断 {removedCount} 个拉伸异形四面体!");
            }

            return removedCount;
        }

        public void FlushToGPU()
        {
            if (!_dirty) return;
            ReadbackFromGPU();
            ReinitializeGPU();
        }

        public void ReadbackFromGPU()
        {
            if (!_dirty) return;
            _solver.ReadbackAll(_data);
        }

        public void ReinitializeGPU()
        {
            if (!_dirty) return;
            _solver.Dispose();
            _solver.Init(_data);
            _dirty = false;
        }

        static long EK(int a, int b)
        { if (a > b) { int t = a; a = b; b = t; } return (long)a * 2000000L + b; }
        // ══════════════════════════════════════════════════════
        // 跨帧拓扑一致性锁定
        // ══════════════════════════════════════════════════════
        float EnforceConsistentSide(int vid, float d, Vector3 currentNormal)
        {
            if (vid >= _originalVertCount) return d; // Only stable original vertices can be side-locked.

            int currentSide = d >= 0f ? 1 : -1;

            if (_strokeVertexSide.TryGetValue(vid, out var locked))
            {
                // 如果平面发生了显著旋转（>25度，cos(25)≈0.9），则旧的锁定失效，更新为新平面的锁定
                if (Vector3.Dot(locked.normal, currentNormal) < 0.9f)
                {
                    _strokeVertexSide[vid] = new VertexSideLock { side = currentSide, normal = currentNormal };
                    return d;
                }

                if (currentSide != locked.side)
                {
                    // 发生了跨帧侧翻转！强制拉回锁定侧
                    // 锁为正侧(1) -> 设为 0f，这会被后续 Trajectory Correction 识别并吸附到 0f (保持正侧)
                    // 锁为负侧(-1) -> 设为 -1.01e-4f，恰好大于 SNAP_DIST，保证它严格是负的并且极靠近切面
                    return locked.side > 0 ? 0f : -SNAP_PLANE_DIST;
                }
            }
            else
            {
                // 首次在切面附近评估该顶点，记录其初始拓扑侧和法线
                _strokeVertexSide[vid] = new VertexSideLock { side = currentSide, normal = currentNormal };
            }
            return d;
        }
    }
}
