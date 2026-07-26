# Progressive Cutting Technical Spec v2 (Paper2025Graph Backend)

版本: v2（基于 v1 审阅后的优化重写）
定位: 用 2025 论文作为切割主干、2024 论文作为概念来源、其余作为自研工程胶水，替换当前 VNA 主切割路径
适用代码范围: `Assets/SurgicalSim/CuttingPaper`（新增）、`Assets/SurgicalSim/CuttingV3`、`Assets/SurgicalSim/Core`、`Assets/SurgicalSim/Physics`、`Assets/SurgicalSim/Cutting`

> 本文档相对 v1 的核心修订：
> 1. 修正论文分工。2025 是唯一可直接落地的切割主干；2024 只提供**目标与概念**，其 AFCC/cluster expansion/merging 绑死在「背景网格粒子 + 形状匹配 cluster」表示上，**不能照搬到 tet 网格**。
> 2. 明确「两篇论文都没给、必须自研」的清单，并把它们作为一等接口而非事后补丁：任意刀路相交几何 / 有限扫掠体 narrow phase、闭合 detach、cut-front graph、原子提交回滚、quality gate、阈值、quad 对角线确定性。
> 3. 全文围绕 9 个目标重组，每个目标都可追溯到「机制 → 来源 → 阶段 → 验收」。

---

## 0. 核心判断（先读这一段）

- **2025 论文（XPBD + 图着色，Beihang，Pacific Graphics 2025）= 切割主干。** 它给出了 tet 网格上的局部拓扑切分骨架：3 点交 → 4 子 tet、4 点交 → 6 子 tet（Fig.2）；trajectory correction（交点 snap 到最近顶点，Fig.3）；shared-original-vertex 分离判据（Fig.5）；SCGC 动态图着色 + constraint clustering。本项目主线就建立在这些局部机制之上。移动刀具的有限扫掠相交、闭合 detach、事务回滚和阈值仍是本项目自研部分。
- **2024 论文（shape matching + AFCC，Beihang，CMPB 2024）= 概念来源，不是第二套可用算法。** 它的物理表示是「背景网格上的粒子 + overlapping cluster + 形状匹配约束」，可视面用 marching cubes 变体；AFCC 是在**粒子连通图**上求连通分量，cluster expansion/merging 是**形状匹配 cluster** 的操作。这些都搬不到 tet 表示。我们只借它的**目标**：切开后视觉/物理一致、防 ghost force、碎片归并/附着、控制 component 增长。
- **大部分实现量与几乎全部工程风险是「自研胶水」，论文没给。** 真实切割器最难的部分——怎么对一把移动手术刀算出相交、有限扫掠体、共面/掠过/拐角、闭合路径 detach、原子提交回滚——两篇论文都几乎没写。本文档把这些显式列为一等模块。
- 因此本方案不是「照搬两篇论文」，而是：**以 2025 为核心切割器 + 自研一套完整切割系统外壳 + 借 2024 若干概念做一致性与碎片治理**。

主线 pipeline：

```text
Swept blade path
  -> Cut Stabilization Buffer        （自研：时间一致性）
  -> Patch Canonicalization          （自研：把刀路规范成有限 patch）
  -> Candidate tet filter            （现有 V3 / 空间索引）
  -> Finite swept-volume narrow phase（自研：论文未覆盖的相交几何）
  -> Trajectory correction           （2025）
  -> 2025 Tri/Quad tet subdivision   （2025，+自研对角线确定性）
  -> Tet Quality Gate                （自研：fail-safe）
  -> Atomic local commit / rollback  （自研）
  -> Cut-face registry               （自研：切面唯一来源）
  -> Shared-original-vertex 分离      （2025 机制）
  -> Material component graph 审计     （2024 概念，tet 上重实现）
  -> Fragment control                （2024 概念，tet 上重实现）
  -> Cut-front graph + proof-gated detach（自研：论文未覆盖）
  -> XPBD rebuild / hybrid graph coloring（2025）
  -> Registered smooth cut-surface rendering（现有渲染 + registry 绑定）
```

VNA 保留为实验对照，默认关闭，不再作为生产切割 backend 的优化目标。

---

## 1. 目标 → 机制 → 来源 → 阶段 映射总表

这张表是全文的锚。每个用户目标都必须在此可追溯，后续章节是它的展开。

| # | 用户目标 | 核心机制 | 来源 | 兑现阶段 | 关键验收 |
| --- | --- | --- | --- | --- | --- |
| G1 | 任意路径切割（直线/折线/S/直角/闭合/半闭合） | Patch canonicalization + 逐 patch 局部 tet 切分 | 2025 模板 + 自研 canonicalization | 直线=S1，折线/S/直角=S2，闭合=S3 | 各路径切面连续、无一串小洞、拐角是 crease |
| G2 | 切下局部组织块（非单平面） | 有限 swept patch（非无限平面）+ 闭合 detach | 自研 swept volume + 自研 cut-front graph | S3（detach），几何基础 S1/S2 | 闭合路径能产生独立 component |
| G3 | 只在 swept volume 内切（不全局切断） | 有限扫掠体 narrow phase + `(tetId,session,patch)` ledger | 自研（2025 假设切割面已存在） | S1 | 候选/相交局部化，无全局切断 |
| G4 | 切面实时生成/重建/跟随轨迹 | 显式 cut-face registry + 局部增量重建 | 2025 产生 cut cap + 自研 registry | S1 | 切面随刀路实时出现且来自 committed faces |
| G5 | 闭合/半闭合判断独立 component | Cut-front graph + residual bridge + solver audit，proof-gated | 自研（2024 AFCC 为概念来源） | S3 | 只有证明通过才 detach，假 detach=0 |
| G6 | 不产生大量碎屑 | Fragment controller：数值 sliver 附着 vs 真实薄片保留 | 2024 概念（cluster expansion/merging），tet 上重实现 | S2 | 无碎屑云；真实薄片不被吞 |
| G7 | tet 数量不爆炸 | Generation cap + same-patch ledger + trajectory correction + 预算门 | 2025 trajectory correction + 自研预算 | S1（门）→ S2 | newTets 有上界；无 sliver 级联 |
| G8 | FPS 不大幅下降 | Dirty-region 局部性 + incremental AFCC + hybrid coloring + 批量 solver rebuild | 2025 图着色 + 自研局部性 | S1（global 基线）→ S3（local） | cut 帧无大幅崩塌，全局 rebuild 仅 fallback |
| G9 | zero-width + 平滑切面 | 同位置 pos/neg clone 成对（zero-width）+ registry → smooth renderer | 2025 clone/shared-vertex + 自研 registry + 现有渲染 | S1 | 切面平滑、不显示外表面纹理、无黑刺 |

阶段缩写：S1=Stage 1，S2=Stage 2，S3=Stage 3（见 §12）。

---

## 2. 论文分工的正确边界

### 2.1 2025 论文提供什么（可直接落地）

精确到论文给出的内容：

- **Tet subdivision（Fig.2）。** 三点相交 → 4 个子 tet：`AB1C1D1, BB1C1C, B1C1D1C, B1CD1D`；四点相交 → 6 个子 tet：`AE1F1H1, ADF1H1, DF1G1H1, BE1F1H1, BCF1H1, CF1G1H1`。原 tet 从 cluster 移除、不再参与计算。这与本文 TriSection（4 子）/ QuadSection（6 子）的子 tet 数量完全一致。
- **Trajectory correction（Fig.3）。** 当交点离某 tet 顶点过近会生成体积极小的 tet；将该交点 snap 到最近顶点，避免 sliver。**注意：论文没有给具体阈值数字**，所有 snap 阈值都是本项目自定（见 §11）。
- **Shared-original-vertex 分离判据（Fig.5）。** tet 之间「除了共享切割交点外，还共享至少一个原始顶点」才算相连，否则应分离。论文用 2D 三角形演示。本文 §6 的分离逻辑直接采用此判据。
- **并行 XPBD 求解：SCGC 图着色 + Color Preemption shortcut + neighbor-based constraint clustering。** 支持拓扑变化后的动态重着色，这是本文 §9 的基础。

### 2.2 2025 论文**没**提供什么（必须自研）

- 怎么对一把**移动刀具的任意轨迹**算出「3 点 / 4 点相交」。论文直接假设切割面已经产生了这些交点，不讨论 swept volume、有限 footprint、共面/掠过/拐角分类。→ 本文 §5.2。
- Quad 切面**相邻 tet 之间对角线如何保持一致**（不一致会产生裂缝/bowtie）。→ 本文 §5.4 自研 deterministic diagonal。
- **闭合/半闭合路径如何 detach** 出一个独立 component。→ 本文 §7。
- 原子提交/回滚、quality gate、时间稳定、阈值的工程化。→ 本文 §5.5/§5.6/§11/§10。

### 2.3 2024 论文提供什么（仅概念，且表示不匹配）

- **AFCC（Aggregate Finding Connected Components）。** 在背景网格的粒子连通图 `G(V,E)` 上求连通分量，切割即删粒子 → 连通分量自然分裂成新 cluster；对 overlapping 粒子做复制以并行（受 ECL-CC 启发）。
- **Cluster expansion。** 孤立 cluster 吸收邻居 cluster 的部分粒子，避免自由漂浮的视觉碎片。
- **Cluster merging。** 用 CAS 原子操作把过小 cluster 合并，控制 cluster 数量与 readers-writers 竞争。
- **可视/物理一致性、防 ghost force、防碎片**，可视面用 marching cubes 变体。

**为什么不能照搬：** 以上全部建立在「背景网格 + 粒子 + overlapping 形状匹配 cluster」的表示上。本项目是 tet 网格 + XPBD，没有背景网格、没有 overlapping cluster、没有「删粒子让连通分量掉出来」的机制。因此我们**重新实现**一个 tet 表示下的同目的算法（`PaperMaterialComponentGraph` / `PaperFragmentController`），它们是「受 2024 启发」而非「使用 2024 算法」。文档与代码注释都应这样标注，避免有人去找 2024 的 GPU AFCC + background grid 往 tet 管线上硬套。

### 2.4 两篇在「分离」上的重叠如何收敛

2025 的 shared-vertex 判据与 2024 的 AFCC 解决的是同一件事：保证切开的块在物理上和视觉一致地分开。本方案**不把它们当两个独立算法叠加**，而是分工：

- **机制层（2025 shared-vertex）：** 决定每个被切影响的原始顶点要不要 clone、incident tet 如何重写——这是真正改拓扑的一步。
- **审计/治理层（2024 概念）：** 在 dirty region 上跑连通分量，**审计**机制层的结果是否与视觉一致、是否有残留 bridge、是否产生需要附着的数值碎片或需要保留的真实薄片。它不直接做分离，只做一致性检查 + 碎片策略。

### 2.5 两篇都没给、必须自研的总清单（最高风险）

按风险从高到低：

1. 任意刀路相交几何 / 有限扫掠体 narrow phase（§5.2）。
2. 闭合/半闭合 proof-gated detach + cut-front graph（§7）。
3. 原子提交/回滚契约（§5.6）。
4. Tet quality gate fail-safe（§5.5）。
5. 时间稳定层（§10.1）。
6. Quad 对角线确定性（§5.4）。
7. 全部数值阈值的 mesh-scale 化（§11）。

这些是工期与风险的真正所在，不是论文的「附带实现」。

---

## 3. 系统架构

### 3.1 命名空间与文件

新增独立命名空间，避免污染 V3 / VNA：

```text
Assets/SurgicalSim/CuttingPaper/
  PaperCutBackendAdapterV3.cs      接入 CuttingToolV3 的唯一适配器
  PaperProgressiveTetCutter.cs     主切割器（编排）
  PaperCutStabilizationBuffer.cs   时间一致性（自研）
  PaperCutCanonicalizer.cs         刀路 -> 有限 patch（自研）
  PaperCutSession.cs / PaperCutPatch.cs / PaperCutSheet.cs
  PaperCutPatchClassifier.cs       有限扫掠体 narrow phase（自研）
  PaperTrajectoryCorrection.cs     2025 snap
  PaperTetSubdivisionTemplates.cs  2025 Tri/Quad 模板 + 对角线确定性
  PaperTetQualityGate.cs           提交前 fail-safe（自研）
  PaperSplitRecordCache.cs         edge -> List<record>（一对多）
  PaperCutFaceRegistry.cs          切面唯一来源（自研）
  PaperMaterialComponentGraph.cs   2024 概念，tet 上重实现
  PaperSharedVertexSeparator.cs    2025 判据
  PaperResidualBridgeDetector.cs   残留连接检测（自研）
  PaperFragmentController.cs       2024 概念，tet 上重实现
  PaperCutFrontGraph.cs            闭合判定（自研）
  PaperDetachCommitter.cs          proof-gated detach（自研）
  PaperConstraintGraphUpdater.cs   2025 hybrid coloring
  PaperDirtyRegionTracker.cs       局部性（自研）
  PaperSolverAudit.cs              ghost-force 审计（自研）
  PaperCutDiagnostics.cs / PaperCutQualitySettings.cs
```

V3 侧只接入 adapter，不把逻辑塞进 `TetSubdivisionCutter.cs`：

```text
CuttingToolV3
  -> LegacySubdivision (TetSubdivisionCutter)
  -> ExperimentalVna   (TetVnaCutterAdapter，默认关闭)
  -> Paper2025Graph    (PaperCutBackendAdapterV3，新主线)
```

### 3.2 Backend enum

```csharp
public enum CuttingBackendV3
{
    LegacySubdivision,
    ExperimentalVna,
    Paper2025Graph
}
// 迁移期默认 LegacySubdivision；新主线稳定后切 Paper2025Graph。
// VNA 不得再默认开启。
```

`PaperCutBackendAdapterV3` 对外接口保持接近 `TetVnaCutterAdapter`，使 `CuttingToolV3` 的压力/输入/flush 逻辑无需大改：

```csharp
public sealed class PaperCutBackendAdapterV3
{
    void Init(TetMeshData data, XPBDSolverGPU solver, PaperCutQualitySettings settings);
    void ResetStroke();
    CutResult AddSweptBladeStep(Vector3 fromA, Vector3 fromB, Vector3 toA, Vector3 toB,
                                float bladeRadius, bool finiteSweptBladeFilter);
    CutResult CommitStroke();
    void RebuildSolverFromCurrentState();
    void BuildVisibleSurfaceLists(List<int> originalSurfaceTris, List<int> cutSurfaceTris);
    PaperCutDiagnostics Diagnostics { get; }
}
```

---

## 4. 核心表示与数据结构

唯一物理表示是 **tet 网格**。所有「分离/碎片/一致性」都在 tet + 其 incident 关系上完成，不引入 2024 的背景网格。

### 4.1 PaperCutSession / PaperCutPatch

`PaperCutSession` 管理一次连续刀路（pressure/contact 进入切割态时创建，刀离开/松开时结束）。生命周期：`BeginSession → AppendPatch* → FlushPatchBatch* → EndSession`。

`PaperCutPatch` 表示刀具实际扫过的**有限 bounded footprint**（不是无限平面）：

```csharp
struct PaperCutPatch
{
    int sessionId, patchId, cornerId;
    Vector3 prevA, prevB, currA, currB;   // swept quad 四角
    Vector3 planeNormal, moveDir;
    Bounds bounds;                         // 有限 AABB，narrow phase 用
    PaperPatchKind kind;                   // Straight / Curve / Corner
}
```

要点：patch 的有限 swept footprint 必须参与 narrow phase；**禁止用无限平面切全局**（这是 G3 的根本保证）。

### 4.2 PaperSplitRecordCache（一对多）

复杂路径中同一条原始边可能被不同 patch 在不同位置/方向切过，因此**不能是 `edge -> 单 record`**：

```csharp
Dictionary<OriginalEdgeKey, List<PaperSplitRecord>>
```

复用一条已有 record 需同时满足：同 session、同/相邻 patch、位置距离 < mesh-scale 阈值、法线夹角 < 阈值、cornerId 兼容、side label 兼容、record 未被 rollback。S 形回切或远处再次穿过同一 edge **必须新建 record**。key 必须基于原始顶点 id，而非当前 child vertex id。

所有 split / ledger / diagonal / cut-face registry 都必须使用同一套 canonical topology key，不能各模块各自拼 key：

```csharp
struct PaperTopologySupportKey
{
    OriginalSupportKind kind;      // Vertex / Edge / Face / TetInterior
    int originalA, originalB, originalC;
    int parentLineageId;
    int sessionId, patchId, cornerId;
    int sideLabel;
    int supportOrdinal;            // 同一 original support 多次穿越时递增
}
```

`same-patch ledger`、`PaperSplitRecordCache`、quad diagonal registry 和 cut-face registry 必须引用 `PaperTopologySupportKey` 或其稳定 hash。仅 `(tetId, sessionId, patchId)` 不够，因为 tail flush、corner patch、S 形回切、later patch re-cut prior children 都可能命中同一 tet lineage 的不同 material support。

### 4.3 PaperCutFaceRegistry（切面唯一来源）

```csharp
enum PaperFaceClass { OriginalSurface, CutSurface, HiddenInternal }
struct PaperCutFaceRecord { long faceKey; int sessionId, patchId, parentTet, side, a, b, c; Vector3 normal; PaperFaceClass faceClass; }
```

硬规则（G4/G9 的基础）：物理 split 模板**必须显式注册** cut faces；renderer 只渲染 `CutSurface`，主肝脏面只渲染 `OriginalSurface`；**禁止**靠「全是新顶点 / 法线方向 / shader backface」猜切面。

### 4.4 PaperMaterialComponentGraph / PaperCutFrontGraph

```csharp
// 2024 概念在 tet 上重实现：dirty region 内的物理连通分量
struct PaperMaterialComponentGraph {
    List<int> dirtyTets; List<Adjacency> materialAdjacency;
    Dictionary<int,int> tetToComponent;
    Dictionary<int,float> componentVolume, componentOriginalSurfaceArea;
    int residualSharedVertexBridges, residualCrossConstraintBridges;
}
// 自研：判断闭合/半闭合
struct PaperCutFrontGraph {
    /* nodes=cut-front vertices/boundary anchors; edges=registered cut-front edges */
    int regionId; List<EdgeKey> openInteriorEdges, boundaryAnchoredEdges, nonManifoldFrontEdges;
}
```

---

## 5. 切割主干（Stage 1 核心，2025 + 自研）

### 5.1 候选 tet 检测

Broad phase（现有 V3 候选 / `TetSpatialIndex`）只负责找出可能与 patch bounds 重叠的 tet，可漏远处 tet，但**不得对已重叠 tet 做最终 reject**。

### 5.2 有限扫掠体 narrow phase（自研，论文未覆盖）

这是 G3 的核心，也是论文给得最少的部分。每个 patch 执行：

1. patch AABB vs tet AABB；
2. 构造有限 swept volume：blade segment sweep prism + radius capsule + 前后端帽；
3. 把 bounded swept quad 拆成两个三角形只作为中心 sheet，不得替代 radius volume；
4. tet edge vs patch triangle 求交（带符号）；
5. patch/capsule/端帽 vs tet face 求交；
6. coplanar / grazing fallback；
7. vertex/edge/face degeneracy 分类；
8. self-intersection / fold / corner overlap 拆分或标记；
9. `PaperTopologySupportKey` ledger 去重。

所有 no-cut 必须给精确原因，禁止泛泛 `no_intersection`：`outside_patch_bounds` / `same_side` / `outside_finite_footprint` / `already_processed_same_patch` / `ambiguous_feature` / `quality_rejected`。`finiteSweptBladeFilter` 只能缩小候选范围，**不得把真实相交 tet 变成 generic no_intersection**。

输出 `PaperTetIntersection`，含 cutCase（Uncut/TriSection/QuadSection/VertexPass/EdgeGraze/FaceGraze/Ambiguous）、正负侧原始顶点计数、edgeHits、snappedOriginalVertices、finitePatchOverlap、rejectReason。

#### 5.2.1 有限扫掠体几何契约

`PaperCutPatchClassifier` 必须把刀具运动当成有厚度的局部 swept volume，而不是无限平面：

- **中心 sheet**：`prevA, prevB, currA, currB` 组成的 bounded quad；非平面 quad 先按最短对角线拆成两个 triangle，并记录对角线 key。
- **半径 volume**：对 blade segment sweep 做 capsule/prism expansion，半径 = `bladeRadius + finiteRadiusExpansion`。
- **端帽**：`prevA-prevB` 和 `currA-currB` 两端必须参与相交，避免高速移动时只切中间不切端点。
- **高速 substep**：当 `bladeTravel > maxSweptStepLength` 时强制拆成多个 patch；每个 subpatch 单独进 CSB 和 ledger。
- **corner patch**：折线/直角处必须生成 transition patch，不能靠相邻两个直 patch 的重叠自动补洞。
- **coplanar/grazing**：共面、贴边、过顶点不能 jitter；必须分类为 `FaceGraze`、`EdgeGraze`、`VertexPass` 或 `Ambiguous`。
- **self-intersection**：同 session 内局部 self-intersection 优先拆分为多个 non-folded patch；无法拆分时只允许 attached groove / no detach preview。

验收字段：

```text
sweptVolumeHits / sheetOnlyHits / capHits / radiusHits
substepCount / cornerPatchCount / grazingCases / selfIntersectRejected
outsideFiniteFootprintRejects
```

### 5.3 Trajectory correction（2025）

交点离原始顶点过近时 snap 到该顶点；snap 必须跨相邻 tet 复用同一 support；若 snap 后退化，转 `VertexPass`/`EdgeGraze`，**不允许 jitter**；snap 后仍无法生成合法 child 则该 parent rollback 并记录精确原因。阈值见 §11，均为 mesh-scale，论文未给数字。

### 5.4 Tri/Quad 模板（2025 模板 + 自研对角线确定性）

**TriSection**（一个顶点 A 独处一侧，B,C,D 另一侧；Pab/Pac/Pad 为交点）：4 个 child；cut face = 三角形 `(Pab,Pac,Pad)`。

**QuadSection**（A,B 一侧，C,D 另一侧；Pac/Pad/Pbc/Pbd 为交点）：6 个 child（两个三棱柱各拆 3 个 tet）；cut face 是四边形 `Pac,Pbc,Pbd,Pad`。

> 自研补充（论文未给，承重项）：四边形三角化必须 **deterministic 且相邻 tet 一致**，否则共享面处产生裂缝/bowtie。规则：先用 `PaperTopologySupportKey` 组 ordered pair，选 lexicographically smaller；key 退化时选更短几何对角线；**结果写入 registry，相邻 tet 必须复用 registry 的决定，不得各自重选**。

所有 child 提交前用 signed volume 自动修正 winding；cut face winding 由 patch normal 修正；pos/neg clone 同位置不同 particle id（→ zero-width，G9）。

### 5.5 Tet Quality Gate（自研 fail-safe，G7）

不能只检查体积。提交前对每个 child 检查：负体积 / 极小体积 / 长宽比 / 最小高度比 / 重复顶点；并对 cut-face 检查：零面积 / 法线翻转 / registry 重复面 / 局部非流形边 / 体积守恒误差；还要查预算：`MaxNewTetsPerPatch` / `MaxNewTetsPerStroke` / `MaxGenerationPerLineage`。

任一失败的 fallback 顺序（禁止只提交部分 child）：

1. trajectory correction snap 到最近合法支持；
2. 换 alternate diagonal / alternate prism decomposition；
3. 合并同 spatial key 相邻 patch 后重试；
4. rollback 该 parent，记录 `PaperTetQualityReject`。

**禁止为「看起来切了」提交质量不合格的 child。**

### 5.6 原子提交 / 回滚契约（自研）

> 粒度修订（以 §18 支柱 2 / §19.1 为准）：当多个 tet 共享同一条被切的 original face 时，回滚单元是 **cut-closure unit**，不是单个 parent。一个 unit 内要么全提交、要么全 defer/rollback，禁止「回滚一个 tet 而它的共切面邻居已提交」造成半切非流形 bridge。

禁止半提交。两级快照：

- **Parent 级**：classify → trajectory correction → 建/复用 split records → 建 clone → 建 child tets → 建 cut face records → 质量校验 → winding 修复 → 停用 parent → 追加 dirty records。任一步失败，按快照删除本 parent 新建的 particles/child tets/cut faces/split records，恢复 parent active，输出 `rollbackReason`。
- **Batch 级**：child 质量超阈、solver rebuild 失败、coloring 校验失败、registry 非流形、zero-width clone 丢失、fragment 无法区分 sliver/真实薄片、renderer 与 topology 不一致 → 回滚整个 batch，不留 inactive parent、孤立 clone、孤立 cut face 或未同步 solver buffer。

若后续阶段会改写已有 particle/clone 的 mutable state，回滚日志必须覆盖旧值（Positions/PrevPositions/RestPositions/Velocities/InvMasses/ownership/clone mapping/barrier 记录/solver buffer 注册态），不能只记 count 快照。

---

## 6. 分离与一致性（2025 机制 + 2024 概念审计）

### 6.1 为什么不用 plane-side

复杂路径下不存在全局正负侧：S 形、直角拐弯、多 stroke 闭合、trajectory correction 后交点不再严格共面、顶点/边退化穿越，都会让 plane-side 判定失败。因此分离判据是局部 graph + shared original vertex（2025，Fig.5）。

### 6.2 Shared-original-vertex 分离候选（2025 机制，Stage 2 只生成候选）

对每个被切割影响的原始顶点 `v`：

1. 收集 incident active tets；
2. 在这些 tet 上建局部 adjacency graph；
3. 只有满足以下条件才连边：共享 material face 或合法 material support；该 face 不是 registered cut face；不跨当前 patch barrier；shared original vertex support 合法；
4. 求连通分量；
5. 若 incident tets 分成多个 material components，只生成 `PaperSeparationCandidate`，不得立即 clone。

这是消除「藕断丝连」的候选发现步骤，不是提交步骤。真正 clone / reroute 只能由 §6.3 的 Lazy Separation 经过时间验证、质量门和 post-clone solver audit 后执行。任何实现如果在 §6.2 直接 clone，都视为违反本设计。

### 6.3 Lazy Separation（自研，防 vertex/constraint 爆炸，G6/G8）

「graph 说该分」不能直接触发物理 clone，否则会 vertex explosion + constraint explosion + XPBD stiffness spike。流程：

```text
帧 N         : 标记 bridge / shared-vertex 候选
帧 N+2~3 步后 : 验证该候选在当前形变下仍然分离
若持续且局部质量门通过 : 在可回滚 batch 内提交 clone/reroute，并立刻做 post-clone reroute 审计
否则               : 保持 attached groove 并输出 trace
```

clone 后必须立刻做局部 solver incidence 审计（`bridgeConstraints==0`、`rerouteFailedConstraints==0`、`constraintExplosionRatio<阈值`、`maxClonePerFrame` 未超限），失败则回滚本批 clone/reroute。

### 6.4 Material component graph 审计（2024 概念，tet 上重实现）

对 dirty region 构建 component graph，用于：检查视觉分离是否对应物理分离、防 ghost force、小碎片归并、薄片稳定。它**不**直接做分离，只审计 §6.2 的结果并喂给 fragment controller。

### 6.5 Incremental AFCC（自研局部性，G8）

2024 AFCC 不能每次 cut 全量 rebuild（会让每个 cut tick 变 O(N) 并触发 GPU sync spike）。规则：

```text
if dirtyRegion.Tets / activeTetCount < afccGlobalFallbackRatio (默认 0.05):
    从 dirty 边界跑 local BFS
else:
    full rebuild fallback 并记录原因
```

全量 rebuild 是 fallback，不是正常路径。

### 6.6 Residual Bridge Detector（自研）

每个 batch 后统计：shared vertex bridge、shared edge bridge、cross-component tet constraints、cross-component XPBD edge constraints、有 cut face 但无实际拓扑分裂、component 数与最大 component 占比。仍有 bridge 时：open groove 保持 attached 并记录 recut target；closed/boundary-anchored cut 不允许假 detach，必须阻止并输出 trace。

---

## 7. 闭合 / 半闭合 detach（自研，论文未覆盖，G2/G5）

> 这是两篇论文都没给的能力。2024 靠「删粒子让连通分量自然掉出」实现分离，tet + XPBD 世界没有这个机制，必须自建 detach gate。

### 7.1 形式化闭合判定

闭合/半闭合不能靠刀路外观判断，必须由 `PaperCutFrontGraph` 判断。Region 可进入 detach **preview** 当且仅当：

1. region 属于当前 live session；
2. 没有未解释的 interior open edge；
3. open edge 要么闭合回自身，要么锚定到原始外表面 boundary；
4. region 不跨 self-intersecting patches；
5. region 的 cut face records 属于同一 session version；
6. residual bridge detector 干净（无 shared vertex / shared edge / solver constraint bridge）。

#### 7.1.1 Boundary-anchored 半闭合定义

肝脏是封闭体表，没有天然“开放边界”。因此 `boundary-anchored` 不能简单理解为切线碰到外表面。半闭合 detach 必须满足以下拓扑定义：

- cut-front 的两个端点都落在 `OriginalSurface` 上，且端点 support 已稳定；
- 两个端点之间存在一条原始外表面路径，路径上的 surface triangles 未被当前 cut-front 穿越成多段冲突；
- `cut-front path + original-surface boundary path` 构成单一 simple loop；
- loop 内部 region 的 material component 可由 dirty component graph 唯一确定；
- loop 不跨 self-intersection patch、ambiguous face、未解释 open interior edge；
- solver audit 证明 loop 两侧没有 cross-component constraint。

如果上面任何条件失败，半闭合 region 只能保持 attached groove，并输出具体 trace：`boundary_anchor_missing`、`boundary_path_ambiguous`、`surface_loop_not_simple`、`solver_bridge_cross_boundary` 等。

### 7.2 Proof-gated detach（Stage 3 才允许真正切下）

detach 必须同时满足：cut-front closed 或 boundary-anchored；residual bridge clean；solver audit clean（无 cross-component particle/constraint/coloring 引用）；coloring conflict-free；topology epoch current；fragment controller 无 ambiguous 真实/sliver 判定；render registry 与 committed topology 一致。

任一 gate 失败：不 detach、保持 attached groove、输出 exact trace、**不允许视觉假装已分离**。Stage 2 只产出 preview，Stage 3 才允许 proof-gated detach。

### 7.3 版本一致性

每次成功 topology commit `SessionVersion++`；每次 cut-front graph rebuild `RegionVersion++`；detach preview 捕获 `(TopologyEpoch, SessionVersion, RegionVersion)`；live version 改变则旧 preview 立即失效、必须重算。跨 session 的 region 合并不在本期范围。

---

## 8. 碎屑控制与 tet 上界（2024 概念重实现，G6/G7）

### 8.1 数值 sliver vs 真实薄片

**数值 sliver**（需附着）：born in current patch/session；rest volume 极小；tet 数很少；无 meaningful original surface support；无闭合 cut boundary；邻接更大 component。

**真实薄片**（需保留）：cut boundary 闭合，或 original surface support 足够，或用户经闭合/半闭合路径明确切下。

### 8.2 Fragment attachment 语义（不是删除，不是隐藏 detach）

> 契合约束（以 §18 支柱 3 / §19.6 为准）：fragment 优先级低于分离意图，**只能改 ownership / velocity frame，不能改拓扑**；被分离锁定或落在 registered cut face 上的顶点永不可碰；stroke 未结束或 region 仍可能闭合时禁止 attach，只标 `fragment_deferred`，stroke-end 再判。这样 fragment 不会把刚分离或真实切下的薄片又粘回去（反向藕断丝连）。

数值 sliver attach 必须：保留 active tet / particle / mass / solver constraints；把 clone ownership 改写到邻近大 component；继承其 damping/velocity frame；输出 `fragmentAttached` + 目标 component id。

**禁止**：删 active tet、删质量、pin 住碎片掩盖不稳定、合并有闭合 cut boundary 的真实薄片、把 detach 失败伪装成 sliver attach。无法判定时默认保持 attached 并输出 `fragment_ambiguous`。

真实薄片：保留 mass/constraints；限制拓扑突变后的瞬时速度尖峰；可加短时 damping/shape stabilization；**禁止当 sliver 合并掉**。

### 8.3 tet 数量上界（G7）

same-patch ledger 防重复切；generation cap 防同一 lineage 无限 recut；trajectory correction 防 sliver 级联；§5.5 预算门（per-patch / per-stroke）硬上界；fragment control 附着数值 sliver 而非新增自由 tet。

---

## 9. XPBD / Graph Coloring 集成（2025，G8）

### 9.1 初版安全策略

第一版可沿用 `XPBDSolverGPU.RebuildTopologyFromCpuState(data)` 做全局 rebuild，但每次 topology commit 必须记录 dirty constraints：changed particles、removed/deactivated parent tets、new child tets、rewritten edge constraints、clone-created constraints、affected one-ring constraints。

即使 Stage 1 使用全局 rebuild，也必须受帧预算约束：

- 每帧最多提交 `maxTopologyBatchesPerFrame` 个 topology batch；
- 每帧最多新增 `maxNewTetsPerFrame` 和 `maxClonePerFrame`；
- 全局 rebuild 只能在 stroke end、batch 达阈值、或 solver 处于安全同步点时执行；
- 超预算 patch 进入 queue，不能被静默丢弃；如果 queue 超限，只能显示 debug preview 并记录 `topologyQueueOverflow`；
- solver upload 必须和 `TopologyEpoch` 绑定，禁止 CPU topology 已提交但 GPU buffers 仍旧。

### 9.2 Hybrid coloring（Stage 3）

```text
if dirtyConstraintRatio < scgcLocalRecolorRatio (默认 0.05):
    recolor local subgraph，并验证 dirty 边界无同色共享 particle
else:
    global recolor fallback
```

local recolor 失败必须 fallback 到 global rebuild，不得继续用有 conflict 的 coloring。每次 rebuild/recolor 后验证：无同色 shared vertex；tet/edge constraints 无 conflict；GPU buffers epoch-current；stale 预计算算子已重建。`solverTopologyEpoch` 每个被接受的 topology batch 恰好 +1。

2025 图着色的价值在 Stage 3：把 global recolor 逐步替换成 dirty graph recolor，减少 cut 帧 spike，同时保持并行 solver 正确性。

---

## 10. 时间一致性 / 局部性 / 可回滚 三原则（自研外壳）

### 10.1 Cut Stabilization Buffer（时间一致性，G1）

若每帧裸分类，同一空间区域会在相邻帧间跳变（`TriSection → EdgeGraze → QuadSection`），导致 tet split / component graph / XPBD constraints / 切面全部抖动。

`PaperCutStabilizationBuffer`：用 `PaperPatchSpatialKey`（含 StrokeId / SessionVersion / TopologyEpoch / ParentLineageHash / Sheet / Cell / DirectionBin）索引；规则：

```text
raw 与上一稳定态一致 -> stableCount++ , confidence 升
否则                 -> stableCount-- , 在 confidence 跌破阈值前保留旧稳定结果
stableCount < stablePatchFrameCount -> 用旧稳定结果或标记 pending
```

默认 `stablePatchFrameCount = 2`（不默认 3，手术刀快速移动时 3 帧会让切面明显滞后；3 帧仅作高抖动调试配置）。拓扑一旦提交，旧 stable classification **不得跨 epoch 复用**。`EndStroke()` 不得静默丢尾段：达稳定阈值的 pending 必须 flush；与上一 stable 连续且质量门通过的尾段以 `tailFlush` 提交；其余 drop 并记 `pendingTailDropped` + 空间 key 供诊断。

### 10.2 局部性优先

AFCC、component graph、constraint graph、graph coloring **默认只更新 dirty region**；全局 rebuild 只能作为 fallback，且必须记录原因与耗时。

### 10.3 一切可回滚

每个被接受的 cut batch 必须有完整 rollback log，覆盖 particles / tets / rest state / velocity / cut-face registry / split record cache / component graph / solver upload plan。

---

## 11. 阈值与配置（全部 mesh-scale，论文未给数字）

所有阈值来自 mesh-scale settings，禁止硬编码进算法。默认优先使用 **local mesh scale**，全局 `globalMedianRestEdgeLength` 只作为 fallback：

```text
localMeshScale(tet or region) =
    median original edge length over parent tet one-ring,
    clamped to [0.35 * globalMedianEdge, 2.5 * globalMedianEdge]
```

非均匀 tet 网格必须使用 local scale 计算 snap、plane epsilon、support reuse、volume 和 aspect 阈值，否则细网格区会过松、粗网格区会过严。

```csharp
struct PaperCutQualitySettings
{
    float globalMedianRestEdgeLength;
    float localScaleMinFactor;       // 默认 0.35
    float localScaleMaxFactor;       // 默认 2.5
    float snapEdgeT;                 // 建议初值 0.02（edge 参数）
    float planeSnapDistanceFactor;   // = 0.03 * localMeshScale
    float finiteRadiusExpansionFactor; // = bladeRadius + 0.5 * localMeshScale
    float minChildVolumeRatio;       // 建议 1e-4 * parent volume
    float minAbsoluteVolumeFactor;   // = localMeshScale^3 * factor
    float volumeConservationTolerance;
    float maxChildAspectRatio;       // 初值 80，后续 tune
    float minAltitudeRatio;
    float coplanarDistanceEpsilon;
    float supportReuseDistanceFactor; // = 0.02 * localMeshScale
    float supportReuseNormalAngleDeg;
    int   maxNewTetsPerPatch, maxNewTetsPerStroke;
    int   maxNewTetsPerFrame;
    int   maxGenerationPerLineage;
    int   maxClonePerFrame;
    int   maxTopologyBatchesPerFrame;
    int   maxPendingTopologyQueue;
    float maxSweptStepLengthFactor;  // = factor * localMeshScale
    float afccGlobalFallbackRatio;   // 默认 0.05
    float scgcLocalRecolorRatio;     // 默认 0.05
    int   stablePatchFrameCount;     // 默认 2
}
```

> 重要：上面这些数字（0.02 / 0.03 / 80 / 0.05 …）是**本项目的工程初值，不是论文给的**。2025 论文只说「snap 到最近顶点」，没给任何阈值。必须经 HUD/log 输出并实测调参。

任何阈值命中都必须记录使用的是 local scale 还是 global fallback：

```text
localScale=<float> globalScale=<float> scaleFallback=<0|1>
snapThreshold=<float> supportReuseThreshold=<float> minAbsVolume=<float>
```

---

## 12. 渲染（zero-width + 平滑切面，G9）

### 12.1 物理切面与视觉平滑面分离

物理层：tet subdivision 模板生成 cut cap faces 并注册到 `PaperCutFaceRegistry`。视觉层：`CutSurfaceRenderer` 只从 registry 读取 `CutSurface`，可对法线做 smoothing，可用 ordered cut sheet 生成更平滑 overlay，但 overlay 必须绑定实际 registered cut region。

**禁止**：从 VNA raw boundary 直接显示；从「新顶点构成的面」猜切面；从 normal/backface 猜内表面；物理失败却显示视觉成功。topology rollback 必须同步 rollback cut renderer。没有 committed `PaperCutFaceRecord` 时只能显示**不同材质**的 debug preview，且不计入验收指标。

### 12.2 纹理

主肝脏面：继续 liver material / triplanar，只渲染 `OriginalSurface`。切割内表面：单独 cut tissue material，可用 smooth normals，UV 用局部 plane projection 或 triplanar cut material。这样避免当前 VNA 的白/黑破碎纹理。

### 12.3 平滑、crease 和 zero-width depth policy

切割面必须平滑，但不能为了平滑跨越真实拓扑 crease：

- 同一 `cut sheet region` 内可做 normal smoothing；
- 不同 `cornerId`、不同 sheet side、不同 material component 之间禁止法线平均；
- 直角/折线 corner 必须保留 crease，不能被 smooth overlay 抹成斜面；
- cut surface ordering 必须来自 `PaperCutFaceRegistry` 的 patch order / support key order，不能按 triangle index 随机排序；
- zero-width pos/neg 面深度接近时，renderer 使用稳定 depth bias 或双面材质策略，禁止 z-fighting 闪烁；
- overlay 只允许覆盖已 committed cut-face region，不能跨到 original exterior surface；
- topology rollback 必须撤销 overlay mesh、normal group、material assignment 和 depth bias record。

### 12.4 zero-width

pos/neg clone 在**同一位置**、不同 particle id 成对创建，使切面零宽度——既保持 §G9 的 zero-width cutting，也让 smooth renderer 不产生缝隙黑刺。

---

## 13. 三阶段实施计划

前文模块收敛为 3 个大阶段。**不得跳过顺序**，每个目标在哪个阶段兑现见 §1 总表。

### Stage 0（前置）: 下线 VNA 主线

- `useVnaBackend` 默认 false；新增 backend enum；VNA 标记 experimental；HUD 显示当前 backend。
- 验收：不再出现 `vna:invalid_pending_tet_volume`；VNA 不再自动参与主切割。

### Stage 1: Cut Canonicalization + 局部 tet 切分基础

兑现目标：**G3、G4、G7（门）、G9**，以及 **G1 的直线部分**。

实现：stabilization buffer → canonicalizer → 有限扫掠体 narrow phase → trajectory correction → Tri/Quad 模板（含对角线确定性）→ tet quality gate → 原子提交/回滚 → cut-face registry → 受帧预算约束的全局 solver rebuild + coloring 校验 → registered smooth cut surface。

明确不做：closed detach、S/折线/直角、AFCC component detach、fragment attach、local recolor。

完成后效果：刀划过肝脏产生真实 cut groove；切面来自 committed cut faces（非 VNA raw boundary）、平滑、材质稳定；无黑刺；无 `vna:*`；open groove 仍 attached；solver 经全局 rebuild 同步新拓扑。

验收：

```text
backend=Paper2025Graph
triCases>0 or quadCases>0 ; cutFaces>0
minChildVolumeRatio>=阈值 ; maxChildAspect<=阈值
colorConflicts=0 ; vna:* 缺席
batchRollback=0（正常直线切）; parentRollback 允许但需解释
classificationFlipCount per stroke < 阈值
invalidChildTet=0 ; negativeVolumeChild=0 ; invalidCutFace=0 ; volumeConservationError=0
```

### Stage 2: 复杂路径 + 分离 + component/fragment 一致性

兑现目标：**G1 全部（折线/S/直角）、G6**，并为 **G5** 产出 detach preview。

实现：polyline / S-curve（有序短 patch）/ 直角 corner patch / 同 session 多 stroke closure preview；`PaperSplitRecordCache` 转 `edge->List`；later patch 可切 prior patch 残留；shared-vertex separator（lazy）；residual bridge detector；material component graph + incremental AFCC；cut-front graph 分类 open/closed/boundary-anchored；fragment controller。

关键规则：corner patch 不用 diagonal shortcut；same patch 不重复切同 lineage；clone 后立即局部 solver 审计。

完成后效果：S/折线不再一串小洞；直角是 crease；多数藕断丝连被 local separation 处理，仍连着则 HUD 给出 bridge reason；碎屑明显减少；clean closed/boundary region 进入 detach candidate（**仍不真正 detach**）。

验收：

```text
dirtyTetCount 局部 ; bridgeSharedVerts=0（clean through cut）
bridgeConstraints=0（detach candidate 前）
fragmentAttached / realThinPiecesPreserved 计数并解释
openFrontEdges 解释 attached open groove ; surfaceOnlyPreview=0（committed cut）
afccFullFallbackCount 低且解释 ; componentCount 在重复切下稳定
vertexExplosionRatio / postCloneBridgeConstraints=0 / rerouteFailedConstraints=0
```

### Stage 3: proof-gated detach + 高性能 XPBD

兑现目标：**G2、G5 完整、G8 完整**。

实现：capture epoch/version → solver audit → detach committer → XPBD dirty constraint graph → hybrid local/global recolor → upload buffers → GPU/CPU topology hash 校验 → bump epoch → 更新渲染面；完整回归测试。

帧预算契约：

- topology commit、solver rebuild、GPU upload 必须按 frame budget scheduler 执行；
- global rebuild fallback 每秒次数必须有上限，超过后进入 queued topology mode；
- queued topology mode 中视觉只能显示 debug preview 或已 committed groove，不能假装物理已提交；
- `averageCutFrameMs`、`p95CutFrameMs`、`solverUploadMs`、`globalRebuildFallbackPerSecond` 必须进 HUD。

完成后效果：闭合/半闭合路径能切下局部组织块且无 ghost force 拉回；cut 帧 FPS 不再因每帧全局 rebuild 大幅掉落；切面仍平滑；复杂路径无黑刺/碎片云。

验收：

```text
detachedComponents>=1（clean closed cut）; crossComponentConstraints=0（detach 后）
colorConflicts=0 ; topologyEpoch current ; gpuCpuHashMatch=1
averageCutFrameMs within target ; globalRebuildFallback rare and logged
localRecolorCount > globalRecolorCount（正常 stroke）
```

### 总体能力边界

| 阶段 | 完成后能力 | 不应期待 |
| --- | --- | --- |
| Stage 1 | 直线 open groove、真实 tet split、smooth cut face、无 VNA artifact | 不 detach、不支持复杂路径 |
| Stage 2 | 折线/S/直角、shared-vertex 分离、component preview、碎屑控制 | solver 高性能与最终 detach 仍未完全保证 |
| Stage 3 | proof/audit gated detach、XPBD coloring、性能稳定、完整复杂切割 | 跨 session 任意拓扑合并（未来扩展） |

---

## 14. 验收与测试

### 14.1 Golden tests（接入运行时前必过，§5.4 模板）

以小型 synthetic tet case 验证，不依赖 Unity 场景：child volume 总和=parent（容差内）；winding 修复后无负体积；无重复顶点；child 不重叠、不留洞；cut face winding 匹配 patch normal；**相邻 tet 共享同 support keys 时 quad 对角线决策一致**；pos/neg clone 用法匹配 side label；rollback 精确还原 particle/tet/registry 计数。

最低测试矩阵：

| 测试 | 目标 |
| --- | --- |
| 高速直线切割 | 验证 substep / cap / radius volume 不漏切 |
| S 形回切 | 验证 `edge -> List<record>` 和 parent lineage key |
| 直角 corner | 验证 transition patch 和 crease-preserving rendering |
| 共面贴边 / edge graze | 验证无 jitter、无黑刺、无漏洞 |
| vertex pass | 验证 snap / candidate / lazy separation 语义 |
| 非均匀 tet 网格 | 验证 local mesh scale 阈值 |
| 闭合 cut | 验证 detach preview / proof gate / no false detach |
| semi-closed boundary anchored cut | 验证 boundary path simple loop 定义 |
| rollback 后 renderer hash | 验证 topology 和 renderer 同步回滚 |

### 14.2 HUD / log 诊断（验收的一部分）

```text
backend=Paper2025Graph
csb stable/pending/flip=<i>/<i>/<i>
tqg pass/rollback/sliver/tiny/cutface/volume=<i>/<i>/<i>/<i>/<i>/<i>
patches / candidateTets / intersectedTets / triCases / quadCases / vertexPass / edgeGraze
snappedVerts / newParticles / newTets / cutFaces
lazy mark/valid/commit=<i>/<i>/<i>
afcc local/full/dirty=<i>/<i>/<ratio>
scgc local/full/conflict=<i>/<i>/<i>
bridgeSharedVerts / bridgeConstraints / colorConflicts
fragmentAttached / realThinPiecesPreserved / sliverRejected
rollbackReason / bridgeReason / surfaceOnlyPreview
maxChildAspect / minChildVolumeRatio / cutFaceRegistryHash / topologyEpoch
cutFrameMs canonical/classify/split/tqg/afcc/scgc/solver/render=<ms...>
p95CutFrameMs=<ms> globalRebuildFallbackPerSecond=<float> topologyQueue=<int>
localScale/globalScale/scaleFallback=<float>/<float>/<0|1>
arbitration conflict/defer/forcedConverge/unresolved=<i>/<i>/<i>/<i>
ownerViolation / stateSkipViolation / gateMutationViolation=<0|1>/<0|1>/<0|1>
pendingByState stable/staged/attached/separation/detach=<i>/<i>/<i>/<i>/<i>
```

「没有 diagnostics 的稳定不算稳定」。

### 14.3 视觉 / 物理验收

视觉：直线/S/折线切面连续；直角是 crease 不是错误斜切；切面材质稳定、不显示外表面纹理；无黑色竖刺。物理：open groove 不 detach；closed/boundary-anchored 经 component check 后才 detach；detached component 无 hidden constraint 拉回；真实小 component 不爆炸；数值 sliver 不变自由飞散 tet。

---

## 15. 风险与规避

| 风险 | 规避 |
| --- | --- |
| 把 2024 当 tet cutter | 2024 仅做 component 一致性/碎片治理（概念）；底层 cutter 必须是 2025 local subdivision（§2.3） |
| 相交几何（论文未给）做错 → G3 失效 | 有限扫掠体 narrow phase 一等模块 + 精确 reject reason + ledger（§5.2） |
| shared-vertex 分离过激 → 碎屑 | lazy separation + cut barrier + side label + original surface support；fragment controller 在 clone 阶段处理数值碎片（§6.3/§8） |
| smooth renderer 掩盖物理失败 | overlay 必须绑定 registered cut faces；无 cut face 只显示 debug 材质（§12.1） |
| tet 数量爆炸 | same-patch ledger + generation cap + trajectory correction + 预算门（§8.3） |
| 性能下降 | dirty region 局部性 + incremental AFCC + hybrid coloring + batch solver rebuild（§6.5/§9.2/§10.2） |
| 逐帧分类抖动 | cut stabilization buffer，commit 前必经稳定层（§10.1） |
| 假 detach | proof-gated detach，任一 gate 失败保持 attached（§7.2） |
| 防护机制互相抵消 | §18 owner/proposer/gate 表 + 状态机 + 仲裁诊断；新增 gate 必须声明 owner/input/output/failure action |
| 阈值硬编码 | 全部 mesh-scale 配置 + HUD 输出实测调参（§11） |

---

## 16. 修复后的稳定 pipeline 与 stage gate

```text
Blade Input
  -> Cut Stabilization Buffer
  -> Patch Canonicalization
  -> Candidate Tet Filter
  -> Finite Swept-Volume Narrow Phase
  -> 2025 Tet Subdivision Templates (+deterministic diagonal)
  -> Tet Quality Gate
  -> Atomic Local Commit / Rollback
  -> Dirty Region Tracker
  -> Incremental AFCC
  -> Residual Bridge Detector
  -> Lazy Vertex Separation Queue
  -> Post-Clone Reroute Audit
  -> Fragment Heuristic Attach / Merge
  -> Cut-Front Graph + Detach Candidate Builder
  -> Solver Audit / Proof Gate
  -> Detach Committer
  -> Hybrid SCGC Recolor
  -> XPBD Upload / Solve
  -> Registered Smooth Cut Surface
```

| Stage | pipeline 截止点 | 禁止提前运行 |
| --- | --- | --- |
| Stage 1 | `Tet Quality Gate -> Atomic Commit -> global solver rebuild -> smooth cut surface` | lazy separation、AFCC detach preview、hybrid SCGC、detach |
| Stage 2 | `Fragment Attach/Merge -> detach candidate preview` | detach commit、local SCGC 替换全局 |
| Stage 3 | 完整 pipeline | 无，但失败必须 rollback / 保持 attached groove |

每步必须输出 diagnostics；任一步失败只能回滚当前 topology batch，不留 partial tet split / partial clone / partial cut-face registry / partial component graph / partial solver buffer。

---

## 17. 最终判断

以 2025 论文为核心切割器是合理且可落地的：它给了 tet 切分模板、trajectory correction、shared-vertex 分离、并行图着色，正好对应本项目主线。2024 论文作为概念来源治理一致性与碎屑，但因表示不同（cluster + 背景网格 vs tet + XPBD）必须在 tet 上重实现，不能照搬。真正的工程量与风险在两篇都没给的部分——相交几何/有限扫掠体、闭合 detach、原子提交回滚、quality gate、时间稳定、阈值标定——本文档已把它们列为一等模块并给出阶段化落地。按 Stage 0 → 1 → 2 → 3 顺序推进，9 个目标可逐阶段兑现。

> 一句话：本方案 = **2025 为核心 + 自研完整切割系统外壳 + 借 2024 概念治理一致性与碎屑**，不是「照搬两篇论文」。

> 补充：单看每个机制都对，但它们必须彼此契合才不会互相抵消、反而制造藕断丝连。**§18（仲裁模型）、§19（完整性契约）、§20（验收）是让整套防呆机制收敛成一个不自相矛盾系统的核心，优先级高于前文任何单点规则。**

---

## 18. 机制契合与冲突仲裁模型（coherence layer）

> 核心要求：**预防藕断丝连不靠堆约束，而靠让已有约束彼此契合。** 前文每个机制（lazy separation、incremental AFCC、tet 预算、fragment attach、原子回滚、stabilization）单独都对，但两两之间会争夺同一个 tet/顶点/面，互相抵消——这会把藕断丝连从「漏防」变成「被防呆机制自己制造」。本节**不新增独立规则**，而是给出统一事务模型 + 优先级阶梯 + 冲突仲裁表，把它们收敛成一个不自相矛盾的系统。§10 三原则是它的子集。

### 18.0 主不变量：一个目标，一个权威

整个系统只允许一个最终判定：

```text
AcceptedDetach ⇔
    CutFrontClosedOrBoundaryAnchored
    AND CutClosureUnitWatertight
    AND NoMaterialPathAcrossAcceptedCutBarrier
    AND SolverHasNoCrossComponentReference
    AND RendererReadsCommittedSolverEpoch
```

任何条件不成立，结果只能是 `AttachedGrooveWithTrace`，不能是「看起来掉了但物理仍连着」，也不能是「为了避免风险而不显示切面」。这条主不变量把所有预防手段分成三类：

| 类型 | 例子 | 可以做 | 禁止做 |
| --- | --- | --- | --- |
| proposer | CSB、narrow phase、shared-vertex candidate、fragment controller | 提议 patch / split / clone / attach intent | 直接 detach、直接改 solver 约束 |
| gate | TQG、watertight check、solver audit、barrier registry | 接受 / defer / rollback / veto | 自己生成替代拓扑 |
| consumer | renderer、HUD、debug overlay | 读取 committed record | 反向影响物理结果 |

因此「约束少而契合」的实现准则是：**同一实体只被一个 owner 写入，其他模块只能通过 gate 请求它变更；每个 gate 必须声明失败动作和收敛路径。**

### 18.1 六根支柱

**支柱 1 — 单一 DirtyRegion 权威 + 闭包保证。**
每个 batch 只由 `PaperDirtyRegionTracker` 计算**一个** DirtyRegion，且必须对以下闭包：(a) 每个 cut face 的 face-邻居；(b) 每个被 clone 或 candidate 顶点的**完整 incident star（不止 dirty 内）**；(c) 每个新 child tet 的 one-ring。`{separation, residual bridge detector, AFCC, solver audit, coloring, renderer}` **全部消费同一个 region**，任何模块不得自算 halo。
→ 消除「模块 A 的 halo 比 B 小、于是漏掉 B 制造的 bridge」这一类最隐蔽的藕断丝连，是整个契合层的地基。

**支柱 2 — 单一事务单元 = cut-closure unit（不是 tet，不是机制）。**
原子提交/回滚的粒度是「保证切面 watertight 的最小 tet 集合」，不是单个 parent tet。snap、quality rollback、budget 全部在 unit 粒度上 all-or-nothing。unit 大小被 swept substep 约束，仍是局部的。
→ 直接解决「回滚一个 tet 却切了它邻居」的半切非流形 bridge。

**支柱 3 — proposer / gate 分离 + 优先级阶梯。**
每个机制要么是 proposer（提议改动），要么是 gate（只能「接受」或「拒绝→defer/rollback」，不能擅自改）。同一实体只有一个 owner。两个机制对同一实体分歧时按固定阶梯裁决：

| 优先级 | 层 | 权力 | 不能做 |
| --- | --- | --- | --- |
| 1 | 正确性不变量（watertight、no cross-component constraint） | 硬门：可强制 defer/rollback | 不能 "commit anyway" |
| 2 | 分离意图（shared-vertex 说"此处该断"） | 锁定相关顶点/面 | 不能被 fragment 覆盖 |
| 3 | fragment 策略（attach/merge） | 只能动 ownership / velocity frame | 不能改拓扑；不能碰被分离锁定或在 cut face 上的顶点 |
| 4 | 性能（budget / locality） | 只能 defer（versioned） | 不能 silently drop；不能越过 1/2 |
| 5 | 渲染 | 纯下游 | 不能反馈影响物理 |

→ 解决 separation 与 fragment attach 在同一边界顶点上对冲。

**支柱 4 — defer 必收敛，绝不静默丢弃。**
budget/lazy/queue 只能把工作**推迟**（带版本号），不能跳过。stroke 结束或 cut-front 闭合事件触发一次**强制收敛 pass**：它在**最终几何**上同步重跑分类/分离/校验（不等 wall-clock 帧），让 stabilization 和 lazy 验证在终态一次完成；仍无法解决的输出**显式 residual bridge + reason**，禁止留作沉默 attached。
→ 解决 stabilization(2 帧) / lazy(2~3 帧) / 快速 stroke 结束 三个计时器打架。

**支柱 5 — 单一版本线；barrier 与 pending 都带版本且被强制查询。**
no-merge barrier、pending separation candidate、detach preview 都携带 `(TopologyEpoch, SessionVersion, RegionVersion)`；任何 commit 立即作废与之重叠的 pending；任何 merge/attach/recut **必须先查 barrier**（像锁）；renderer 与 solver 读**同一 effective epoch**。
→ 解决 barrier 被 merge 误 weld、pending 候选过期、视觉 epoch 与 solver epoch 不一致（看起来切了其实没切）。

**支柱 6 — 一个连通性问题只归一个权威。**
日常 component 标注用 incremental local labeling（快、dirty-only）；「这条 front 是否已闭合成独立 component」只在 **front-closure 事件**触发一次 bounded **两侧同时 BFS**（代价=较小块）。detach gate 只采信后者，local labeling 永不下全局 detach 结论。
→ 解决「最后一根丝」是小 dirty、大全局影响，local AFCC 看不到。

### 18.2 冲突仲裁表（这就是"契合"的清单）

| 编号 | 冲突对 | 现象（藕断丝连 / 失效） | 由哪根支柱解决 |
| --- | --- | --- | --- |
| C1 | 原子 per-parent 回滚（§5.6）⨯ cut-closure（§19.1） | 回滚一个 tet、邻居已切 → 半切非流形 bridge | 支柱 2 |
| C2 | trajectory snap（§5.3）⨯ 共享交点 ⨯ quality | T1 snap 改了共享交点，T2 没跟 → 面不 watertight | 支柱 2：snap 是 support 级一次决策，须过所有 incident tet |
| C3 | lazy（§6.3）⨯ stabilization 2 帧（§10.1）⨯ stroke-end 收敛 | 快速 stroke 三计时器冲突 → 要么误提交要么留 attached | 支柱 4：终态同步收敛 |
| C4 | tet 预算（§5.5/§9.1）⨯ cut-closure 完整性 | 预算把一个 unit 切两帧 → 中间帧半切 bridge | 支柱 2+4：预算按 unit；stroke-end 越过帧预算排空 |
| C5 | fragment attach（§8）⨯ shared-vertex 分离（§6.2） | 同一顶点一个要 clone 断、一个要 glue 合 → 抖动 / 隐藏 bridge | 支柱 3：分离 > fragment，顶点被锁 |
| C6 | zero-width no-merge barrier（§19.4）⨯ fragment/cluster merge（§8.3）⨯ recut | barrier 对被 merge 误 weld → 数值 re-bridge | 支柱 5：barrier 强制查询 |
| C7 | local AFCC（§6.5）⨯ last-bridge 全局（§19.5）⨯ detach（§7.2） | 最后一根丝小 dirty，local 看不到全局分裂 → 不 detach | 支柱 6 |
| C8 | stabilization 跨帧复用 ⨯ epoch 不跨复用（§10.1）⨯ child recut | commit 后分类历史丢失 → re-cut 抖动 | 支柱 5：按 support/lineage 身份 rekey，历史随 child 传递 |
| C9 | renderer rollback 同步（§12）⨯ queued topology（§9.1） | CPU 已提交、GPU 未生效，渲染显示已切 → 视觉假分离 | 支柱 5：单一 effective epoch，双读点一致 |

### 18.3 一致性硬规则（实现必须遵守）

1. 任何 proposer 不得直接写它不 own 的实体；跨 owner 改动只能经 gate + 仲裁。
2. 任何「减少分离 / 推迟分离」的优化（lazy、budget、local、attach）都必须挂在支柱 4 的 defer 账上，并在 stroke-end 被强制清算。
3. 所有 dirty/halo 一律取自支柱 1 的同一 region；评审中出现第二处自算 halo 即判违规。
4. 所有 merge/attach/recut/clone 动手前先查支柱 5 的 barrier 与 pending 版本。

### 18.4 模块 owner / proposer / gate 表

| 模块 | 角色 | 拥有的数据 | 输出 | 失败动作 | 不得越权 |
| --- | --- | --- | --- | --- | --- |
| `PaperCutStabilizationBuffer` | proposer | temporal patch classification | `StablePatch` / `tailFlush` | freeze previous stable 或 drop+trace | 不写 tet、不 clone、不 detach |
| `PaperSweptVolumeClipper` | proposer | patch-tet intersection evidence | `IntersectionEvidence` | `no_intersection` / `ambiguous_evidence` | 不改 topology |
| `PaperTopologySupportLedger` | owner | support key、split record、diagonal choice | canonical support / face plan | reject duplicate/mismatched support | 不判定 component |
| `PaperLocalTetSubdivider` | proposer | child tet template candidate | staged child tets / cut faces | rollback unit | 不绕过 ledger 自建 support |
| `PaperTetQualityGate` | gate | child volume、aspect、lineage generation | pass / defer / rollback reason | rollback whole cut-closure unit | 不生成替代切法 |
| `PaperLazySeparationQueue` | proposer | shared-original-vertex candidate | versioned separation candidate | defer with reason | 不直接 clone solver particle |
| `PaperComponentLabeler` | gate/consumer | DirtyRegion component labels | local labels / warning | request last-bridge check | 不作 detach 最终结论 |
| `PaperLastBridgeChecker` | gate | closed front 两侧 BFS | detach candidate / residual bridge trace | attached+trace | 不改 fragment ownership |
| `PaperFragmentController` | proposer | sliver/tiny ownership intent | attach/merge proposal | defer if locked | 不碰 cut face、barrier、separation-locked vertex |
| `PaperSolverAudit` | gate | solver incidence graph | accept/veto detach | rollback detach + trace constraint ids | 不创建新 topology |
| `PaperCutSurfaceRenderer` | consumer | committed cut-face records | smooth zero-width surface | preview material only | 不从视觉结果反推物理 |

实现评审时按此表查：任何新函数如果既做 proposer 又做 gate，必须拆分；任何模块写了非 owner 数据，直接判设计违规。

### 18.5 状态机：防止防护手段跳步

所有切割数据必须按以下状态流动，禁止跳状态：

```text
RawBladeSegment
  -> StablePatch
  -> IntersectionEvidence
  -> StagedCutClosureUnit
  -> GeometryCommittedAttached
  -> SeparationCandidate
  -> SeparationValidated
  -> DetachCandidate
  -> DetachedCommitted
```

Phase / Stage 允许提前停止，但不能绕过。例如 open groove 在 `GeometryCommittedAttached` 停止是合法的；closed cut 若跳过 `SeparationValidated` 直接 detach 是非法的；性能预算不足只能停在 versioned pending state，不能丢失。每个状态必须记录 `TopologyEpoch`、`SessionVersion`、`DirtyRegionId` 和 `rollback token`。

### 18.6 禁止新增孤立 gate

后续不得随意加「再检查一次」式保护。新增任何 gate 前必须在文档中声明：

```text
owner:
input:
output:
failure action: defer | rollback | attached+trace | veto detach
upstream invariant:
downstream invariant:
diagnostic field:
```

没有这些字段的 gate 会制造隐性冲突，尤其容易与 lazy separation、fragment attach、budget scheduler 互相抵消。默认策略是复用 §18.4 的 owner，而不是新增 owner。

---

## 19. 藕断丝连完整性契约（按上轮 10 条修复落地）

> 本节是上一轮 10 条修复的落地，全部表达为支柱 1–6 下的具体契约，而非孤立新规则。

### 19.1 Cut-closure watertightness 不变量（修复 H1）

cut-closure unit = 被同一段 cut-front 穿过、且通过共享 original face 互为邻居的 tet 的传递闭包。不变量：

- 被 cut-front 穿过的**每条 original face**，其两个 incident tet 必须**都切**，且复用**同一组** split record（同一 `PaperTopologySupportKey`）；
- 一个 unit 内所有 parent 的切分要么全提交、要么全 defer/rollback（支柱 2）；
- 提交后 unit 的 cut faces 必须构成局部 2-manifold 分隔面，无单侧切面、无悬挂边。

违反则 `cut_front_not_watertight`，整 unit 回滚或 defer，禁止半切。

### 19.2 Graze 分离过程（修复 H2，补回 v1 删掉的解决路径）

分类（§5.2）之后必须有**解决**，否则 graze 处必留 bridge：

- **VertexPass：** 对该 original 顶点的 incident tet 做 fan 二分（按 cut surface 在其 star 内的延续方向），两侧各 clone 该顶点；二分不成 → attached + `vertexpass_fan_ambiguous`。
- **EdgeGraze：** 绕该 original 边做 ring 二分，两端点按侧 clone；ring 跨非流形 → attached + `edgegraze_ring_nonmanifold`。
- **FaceGraze：** 对该 original face 做 unzip：clone 其 3 顶点，两侧 incident tet 各归一侧。
- 三者都走支柱 2 unit 提交、支柱 4 defer 收敛。

### 19.3 Clone 语义分类（修复 H3）

两种 clone 必须在数据结构与阶段上分开，禁止混用一个 "clone" 概念：

| 类型 | 对象 | 何时 | 作用 | 单独是否造成分离 |
| --- | --- | --- | --- | --- |
| 交点复制（intersection duplication） | 新交点 Pab± 等 | Stage 1 subdivision | zero-width，单 tet 内部分开 | 否（全局仍靠原始顶点 attached） |
| 原始顶点 clone（shared-vertex clone） | 原始顶点 A/B/C/D… | Stage 2/3 分离 | 真正全局断开两侧 | 是 |

硬结论：**Stage 1 之后切口按设计仍 attached，这不是 bug**；全局分离完整性是 Stage 2/3 + §19.1 的契约。

### 19.4 zero-width no-collision / no-merge barrier（修复 H5）

subdivision 产生 zero-width 交点对时立即登记 barrier（no self-collision + no merge），随拓扑版本维护（支柱 5）。所有 collision/merge/attach/recut 先查 barrier；rollback 同步撤销。缺这条，self-collision 会把刚切开的零宽面又推合 / 粘住。

### 19.5 Last-bridge 全局检查（修复 H6，归支柱 6）

cut-front 闭合或 boundary-anchored 事件触发：从 front 两侧同时 BFS，先封闭一侧即较小 component，代价=较小块；恰好对半切本质 O(N/2)，进 budget scheduler 排帧、可在 stroke-end 容忍短同步停顿。detach gate 只采信本检查。

### 19.6 fragment 时间规则（修复 H7，归支柱 3+4）

stroke 未结束、或 region 仍可能闭合时**禁止 attach**，只标 `fragment_deferred`；stroke-end 再判 sliver / 真实薄片。被分离锁定或落在 registered cut face 上的顶点永不被 fragment 触碰（支柱 3）。

### 19.7 跨 stroke / 渐进切接缝（修复 H8、H9）

- 跨 stroke 闭合：cut-front 端点 join 用独立容差 `cutFrontJoinDistance`（≠ split reuse 距离），对不上 → 环不闭合，出 `cut_front_join_gap`。
- 渐进续切：later patch 必须按 `PaperTopologySupportKey` 复用老 groove 的交点/顶点 support，禁止「再切一刀」另起 support → 接缝 bridge。

### 19.8 queued topology 一致性（修复 H10，归支柱 5）

每个 batch 的 GPU 上传原子；solver 绝不在「部分 batch 已传、部分未传」的半拓扑上跑；renderer 的「已提交切面」以 solver effective epoch 为准，未生效只显示 preview 材质。

### 19.9 audit 范围按约束类型枚举（修复 H4）

solver audit 不只看 dirty 内 tet/edge 约束，必须**枚举所有约束类型**（tet/strain、distance、volume、collision/self-collision、任何全局或长程约束），逐类声明「是否可能跨 component」；可能跨的必须纳入审计或显式豁免登记。漏一类 = 潜在 ghost-force 藕断丝连。

---

## 20. 新增验收指标与既有章节修订

### 20.1 藕断丝连验收硬指标

```text
throughCutResidualBridges = 0 at stroke end   # 划穿后 stroke 末残留桥必须 0，否则带 explicit reason
cutFrontWatertight = 1                          # 每个提交 unit 的切面 2-manifold
lastBridgeGlobalCheckCount / lastBridgeGlobalCheckMs
deferredSeparations / resolvedAtStrokeEnd / unresolvedResidual(=0 期望)
barrierConsultMiss = 0                          # 任何 merge/recut 未查 barrier = 违规
dirtyRegionSingleSource = 1                     # 断言：无第二处自算 halo
fragmentDeferred / fragmentAttached / realThinPiecePreserved
effectiveEpochRenderSolverMatch = 1            # 渲染 epoch 与 solver effective epoch 一致
ownerViolation = 0                              # 任一模块写非 owner 数据 = 失败
stateSkipViolation = 0                          # 任一 cut batch 跳过 §18.5 状态 = 失败
gateMutationViolation = 0                       # gate 自行改拓扑 = 失败
arbitrationUnresolved = 0                       # 仲裁冲突不得留到 committed topology
```

总验收口：**划穿了但 stroke 末 `throughCutResidualBridges > 0` 且无 reason → 直接判失败；`ownerViolation/stateSkipViolation/gateMutationViolation/arbitrationUnresolved` 任一非 0 → 直接判失败。** 前者防 residual bridge，后者防「防护手段互相抵消」。

### 20.2 既有章节修订指针（保持不矛盾）

- §5.5 / §5.6：回滚粒度改为 **cut-closure unit**（§18 支柱 2、§19.1），不再 per-parent 独立回滚已与邻居共切面的 tet。
- §6.2 / §6.3：分离只产候选；lazy 提交受 §18 支柱 4 stroke-end 强制收敛约束。
- §8：fragment 受 §18 支柱 3 优先级与 §19.6 时间规则约束。
- §9.1：预算/queue 按 §18 支柱 2 unit 粒度，支柱 4 不得 silent drop。
- §10 三原则：被 §18 六支柱涵盖并扩展。
- §14.2 / §20.1：所有防护机制必须暴露 owner/state/gate 仲裁诊断；没有诊断的 gate 不允许进入主路径。
