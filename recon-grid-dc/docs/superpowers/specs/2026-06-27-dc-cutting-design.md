# 设计方案：基于 Dual Contouring 的软组织实时切割（Li/Zhou CMPB 2026 复刻）

> 论文：*A real-time reconstructed grid method for soft tissue cutting and haptic*，Li, Zhou, Zhou，Computer Methods and Programs in Biomedicine 273 (2026) 109123。视频 https://youtu.be/eK_QKEXwyag
> 全文：`newpaper_clean.txt`；理解摘要：`paper2_DC_summary.md`；查漏台账：`dc_gap_findings.md`（195 条）；设计审核台账：`dc_audit_confirmed.md`（44 条确认问题）。
>
> **版本 v4**：经 **3 轮多 subagent 对抗式审核 + skeptic 复核**收敛。v1→v2 修首轮 44 条（`dc_audit_confirmed.md`）；v2→v3 修次轮 11 important+17 partial+15 minor 内部一致性（`dc_audit_v2.md`，0 critical）；v3→v4 修末轮 5 important+5 minor（`dc_audit_v3.md`，0 critical，集中于触觉 finger-proxy 的 HLSL 编译期约束与碰撞标志载体）。**第 4 轮审核（`dc_audit_v4`）：0 critical / 0 important / 1 minor（首帧代理状态 seed），已修——设计收敛定稿。** 修订点在文中以 `[FIX Sx-yz]` 标注。

---

## 0. 目标、约束、硬性要求

| 项 | 内容 |
|---|---|
| **目标** | 从零实现论文的**完整管线**：可变形六面体网格 + mass-spring/RK45 物理 + DC 表面重建 + 切割（静态/动态/刀路/消融）+ 触觉（SDF 胶囊 + finger-proxy）。 |
| **硬性要求** | **算法思路 100% 对齐论文**，连最简单的算法也对齐。仅对论文 5 处自相矛盾的笔误做物理修正(D1-D5)，且逐条写入「偏离台账」(§2)。 |
| **计算载体** | Unity 2021.3.45f2 + **HLSL ComputeShader（GPU 直接实现，Shader Model 5.0）**，数据常驻 `ComputeBuffer`。论文用 C++/CUDA，本复刻用 ComputeShader——算法/数据结构等价，仅 API 不同。 |
| **编排** | 单 `ReconGridManager` MonoBehaviour，每帧按 Fig 3.1 顺序串行 Dispatch 三组 kernel（物理→几何/切割→触觉），CPU 仅编排。 |
| **输入** | 程序化基本体先行（立方体/球，解析 level set）；预处理抽象为 `ILevelSetProvider`，后续可接「网格→SDF 体素化」。 |
| **新文件夹** | `Assets/SurgicalSim/ReconGridDC/`，自包含，不依赖旧 XPBD/MC2024/CuttingV*。 |
| **分阶段** | 3 个 Stage：①基座（物理+DC 重建）②切割 ③触觉。每个 Stage 设计完成后**派多个 subagent 审核**，无问题才进入下一个 Stage 的代码撰写。 |
| **验证约束** | 本环境无法编译/运行 Unity → "对齐" = 静态代码对齐 + 多 subagent 审核 + EditMode oracle 设计。性能/视觉由用户在 Unity 内实测。 |

---

## 1. 总体架构

### 1.1 编排（方案 A：单编排器 + 模块化 Kernel）

论文是 3 个 CPU 线程各调度 CUDA。Unity 下 ComputeShader Dispatch 必须主线程发起，所以用**单编排器 + 串行 Dispatch** 复刻 Fig 3.1 的数据流（kernel 内部才是并行）。每帧 Dispatch 顺序见 §4.6（物理/RK45）、§5.10（切割）、§6.5（触觉）的权威清单。渲染走 `Graphics.DrawProceduralIndirect`，直接吃 GPU 三角/法线 buffer（不回 CPU）。

> **预处理（仅一次，Fig 3.1 Initialization）**：CPU 做 `Level set → Create Background Grid → 计算最外层网格交点`，GPU 做 `Calculate Feature Points → Construct Triangles`，CPU `Wait for GPU`（显式 barrier）后 `Draw Surface`。（来源：I05；CPU/GPU 分工与 barrier 是硬要求。）

### 1.2 文件夹结构

```
Assets/SurgicalSim/ReconGridDC/
  Core/
    GpuBuffers.cs            // 所有 ComputeBuffer 的集中分配/释放（SoA），含 §3 全部 buffer
    GridConventions.cs       // Fig 2.4 编号、gridDims/voxelId 编码、每轴 4-体素+localEdge stencil、octant 映射
    FixedPointAtomic.cs      // float↔定点 int 转换常量（scale）与 HLSL include 约定
    DeviationLedger.cs       // 偏离台账（注释 + 常量开关），与 §2 一一对应
  Preprocess/
    ILevelSetProvider.cs     // float Sample(float3); float3 Gradient(float3)
    PrimitiveLevelSets.cs    // Box / Sphere 解析 SDF
    BackgroundGrid.cs        // AABB→均匀网格→角点 inside/outside→MC256 交边→Hermite 交点(per-voxel 复制)
  Physics/
    MassSpringSolver.cs      // 结构+弯曲弹簧编排（定点原子累加）
    AdaptiveRK45.cs          // 外层 dt 循环 + 内层自适应子步 + 7 阶段三连编排
    ParticleFrames.cs        // 每质点旋转坐标系（极分解，Berndt[22] 风格）
  Recon/
    DualContouring.cs        // 外表面特征点(QEF) + 缝合 + 法线 + triBuf append 协议
    ConnectivityLUT.cs       // 4096 表 CPU 生成（compCount + vertToComp[8]，致密分量 id）
  Cutting/
    CuttingTool.cs           // 刀具线段 S/E、刀厚 D、扫掠面
    CutDetector.cs           // Möller-Trumbore，ray=体素边；cut 位传播到 4 体素
    CutTopology.cs           // 连通分量(顶点键) + 切点 + 切面特征点 + 部分四边形缝合
    AblationCutter.cs        // 消融两遍式 + particleState
  Haptics/
    ToolSDF.cs               // 胶囊 SDF + 三角面 SDF + 候选三角广相
    FingerProxy.cs           // Algorithm 1 交替投影（组协作 + 外层上限）
    HapticForce.cs           // Eq22-27 力/力矩(数值加固) + 反作用力散射到质点
    IHapticDevice.cs         // 力向量输出接口（OpenHaptics 适配留桩）
  Shaders/
    Physics.compute  Recon.compute  Cutting.compute  Haptics.compute
  Demo/
    ReconGridManager.cs      // 编排器（每帧 Dispatch 顺序 + barrier）
    DemoSceneSetup.cs        // 程序化立方体/球场景
  Tests/  (EditMode)
    ConnectivityLUT_Tests.cs LevelSet_Tests.cs FeaturePoint_Tests.cs
    SpringForce_Tests.cs CutBitPropagation_Tests.cs Quaternion_Tests.cs
```

### 1.3 全局约定（冻结，载力）

**坐标/索引约定（来源 C09/C11，照抄 Fig 2.4）：**
- 体素 8 角点 `V0..V7`：底面 `V0,V1,V2,V3`（逆时针），顶面 `V4,V5,V6,V7`（V4 在 V0 正上方）。
- 12 边编号（**4096 表按此生成，绝不可改**）：
  - 底环：`e0=V0-V1, e1=V1-V2, e2=V2-V3, e3=V3-V0`
  - 顶环：`e4=V4-V5, e5=V5-V6, e6=V6-V7, e7=V7-V4`
  - 竖边：`e8=V0-V4, e9=V1-V5, e10=V2-V6, e11=V3-V7`
- 每顶点恰 3 条入射边（连通分量用）：如 V0={e0,e3,e8}。
- **网格元数据 [FIX S1-M2]**：`GridConventions.cs` 冻结 `int3 gridDims (Nx,Ny,Nz)`、`voxelId = i + Nx*(j + Ny*k)` 及其逆。体素中心**不**由 world origin 推算，而取该体素 8 个**当前变形**角点(`voxelCornerBuf`)的均值（供 Eq5 box-SDF）。
- **每条网格边被 4 个体素共享**（DC 对偶）。每轴的 4-体素 stencil **同时给出每个体素槽位对应的本地边号 0..11**（`(axis, slot)→(voxelOffset, localEdge)` 映射）[FIX S2-C1/S2-I7]。例：x 轴边在角点 (i,j,k)：体素 (i,j-1,k-1),(i,j,k-1),(i,j-1,k),(i,j,k)，各自该边的本地边号不同。
- 每条相交边的 4-体素 stencil 同样用于外表面缝合（§4.3）。
- 两套查找表互不混淆（I04）：**MC256**（8 角点 inside/outside → 交边，预处理）≠ **CONN4096**（12 边 cut 位 → 连通分量，切割）。

**符号/方向约定：** inside = `φ<0`，outside = `φ>0`，等值面 = `φ=0`（I01/I02）。

**GPU 浮点原子约定 [FIX S1-I3/S1-I10]**：HLSL SM5.0 **无 float 原子加**（`InterlockedAdd` 仅 int/uint）。论文用 CUDA `atomicAdd(float*)` 做(a)受力累加、(b)法线平滑。本复刻用**定点整数原子散射**忠实复刻：累加 buffer 用 `int3`（或 `RWByteAddressBuffer`），`InterlockedAdd` 整数；写前 `float→int` 乘 `SCALE`，barrier 后 `int→float / SCALE` 还原。`SCALE`：力用 `1<<10`（量程 ±2e6、分辨率 1e-3，覆盖 k_s=7.5e4×位移）；法线用 `1<<20`。这是**平台实现细节非算法偏离**（同样的 scatter+atomic 数据流），在 `FixedPointAtomic.cs` 注释，不计入 §2 台账。

---

## 2. 偏离台账（Deviation Ledger）

> 以下 **8 处**偏离（D1–D5 修正论文笔误；D6-D8 为实现决策，用户决策：物理正确版 + 台账）。其余全部逐字对齐。每条在 `DeviationLedger.cs` 与对应 kernel 注释标 `// DEVIATION Dn`。

| ID | 论文原文 | 我们实现 | 原因 | 来源 |
|---|---|---|---|---|
| **D1** | Eq3 `F=Σ−(x0−p_i)·n_i²`，求值固定在 x0 | `F=Σ n_i·(n_i·(p_i−x_k))`（QEF 负梯度），在**当前 x_k** 求值 | 字面标量读法在 x0 处 F≡0，特征点永不动；`n_i²` 实为 Schmitz[31] 的法向投影 `n_i n_iᵀ` | C02/C03/C04/C49 |
| **D2** | Eq11 `F_s=−k_s·‖x_i−x_j‖·dir−c_s·Δẋ`（无静长） | `F_s=−k_s·(‖x_i−x_j‖−L0)·dir−c_s·Δẋ`，`L0`=初始边长(0.25cm) | 无 L0 = 零长弹簧 → 晶格塌缩 | C32/C52 |
| **D3** (修正于 GPU 验证) | Eq9 `M ẍ + F_int = F_ext` ⇒ `ẍ=(F_ext−F_int)/m`（把 F_int 当 LHS 内力=−实际力） | `ẍ=(F_ext+F_int)/m`（Eq17 的 + 号）；其中 F_int=Eq11/Eq14 累加的**实际回复力**（Eq11 前导负号已使其回复，StructuralForce oracle 已证） | Eq11/Eq14 定义的是**作用在质点上的实际力**（回复方向），牛顿 ⇒ +F_int。Eq9 用相反的 F_int 约定，与 Eq11 不一致；用 −F_int 配 Eq11 实际力 ⇒ 弹簧反回复 ⇒ 发散（GPU oracle 抓到：振子外漂、重力 demo 爆炸）。**原 v4 选 −F_int 是误判（来自 gap C37），GPU 跑测试后修正为 +F_int** | C37(修正) |
| **D4** | Eq21 `h_n=h·0.9·(Δ/ε)^{1/5}` | `h_n=h·0.9·(ε/Δ)^{1/5}` | 论文比值方向反了：误差大反而增步长 | C36/C53 |
| **D5** [FIX S1-I1/I13/M3] | Eq5 文字"SDF<0 时停止迭代" | 当**下一迭代点** `boxSDF(x_next) ≥ 0`（抵达/越出体素边界）时停，并**保留最后一个内部点 x_prev（即上一个 x_k）** | x0(质心)起步即在内部(SDF<0)，逐字会第 0 步即停、特征点冻在均值；论文真实意图是"约束特征点不越出体素" | C-/findings I14 |
| **D6** | §4.6 pseudocode places ComputeError→AdaptStep inside the inner substep loop (per-substep h adaptation) | adapt h ONCE per frame from the cross-substep max error (InterlockedMax ErrMax read back once/frame), constant h within a frame | per-substep GPU→CPU readback stalls the pipeline every substep; per-frame retrospective adaptation keeps the real-time budget; the DOPRI stage math (Eq18/19) and D4 step formula are unchanged | implementation decision |
| **D8** (audit wf_6647e0ee C3) | Eq17 `f=[ẋ, (F_ext+F_int)/m]` 无全局速度阻尼项 | Eq17 加质量比例（Rayleigh）阻尼 `−α·v`，α=globalDamp（C# 默认 2.0/s，激活） | 论文 demo 零重力；有重力+自由落体时 Kelvin-Voigt（cs/cb 只阻尼相对速度）无法耗散刚体摆动/坠落模态 → 无 α 则自由端无界加速/永振；α 给出终端速度 g/α。承载稳定性，故保留并入账（此前代码注释 PAPER-SILENT 但未入台账） | 稳定性决策；用户时代参数 |
| **D7** (Stage 4) | §2.1.3 "cutting plane ... consists of two triangles"（world 平面，每帧一次）；§2.1.2 切点记录于未变形边 | 检测射线 = **未变形 rest 边**（§2.1.2 字面），切割面 = 同一 §2.1.3 扫掠面**映射到材料坐标**（rod 采样 ~0.5L → 连续 inverse-trilinear 逆映射 → prev/cur 采样带 ribbon，2·(NS−1) 个三角形），并在**每个 RK45 physics substep 后**追加 ribbon tick；另加 3L world 邻近钳（防外推幻切）与 sliver 面积下限 | 论文所有 demo 零重力；组织在帧间运动时 world 检测在时空上欠采样（CPU 证明 tunnel_repro.py：稠密 300× 仍 comps=1），任何 world 采样密度都无法形成分离层；零变形时逆映射=恒等 ⇒ ribbon 与论文双三角逐位一致（0-g 行为不变） | Stage 4 (task_plan.md)；诊断 wf_e3f28151 3-agent 一致 |

> **论文未给、需补全的"非偏离"决策**（论文沉默，不算改动，标 `// PAPER-SILENT`）：每质点坐标系算法(C17)、θ̇ 解析式+奇异保护(C33/S1-I14/S2-C5)、自适应步接受策略(C39)、外层/内层 dt 关系(C40)、切点 ±D/2 偏移方向(C25)、三角缠绕+3角点缠绕(C07/S2-I8)、QEF 最大迭代 m(S1-M1)、特征点法线在变形下冻结 vs 旋转(S1-I2)、finger-proxy 外层上限/无解回退/碰撞门(S3-I2/I9/IMP6)、四元数 axis-angle 数值加固(S3-I8)、反作用力阈值(S3-M4)、**代理朝向 q_B 推进规则(S3-IMP3)**、**Algorithm 1 内层 A_i/P_{i+1} 每遍推进(S3-M1)**、**v_ms 每外层迭代重算非跨迭代累积(S3-minor)**、**消融切点归属存活端(S1/S2-minor)**、**接触标志 colliding 载体+阻尼门控(S3-final)**、**固定组+哨兵车道掩码(S3-final)**、**命中三角 contactTriId 载体+单/复数散射(S3-final)**、**空分量 valid 标志(S1-final)**。

---

## 3. 数据结构（GPU 常驻 SoA buffer）— 权威清单

> 所有 buffer 在 `GpuBuffers.cs` 集中管理。**预处理后容量固定**（切割不增质点/体素，只增"需渲染体素数"与三角数→append+counter）。`[FIX]` 标注审核新增项。

### 3.1 质点（particle = 网格角点，全局唯一，角点共享不重复）
```
posBuf       : float3    当前世界位（=积分提交态 y_n 的位置分量；RK 阶段内只读）
velBuf       : float3    速度 ẋ（=y_n 速度分量；RK 阶段内只读）
prevPosBuf   : float3    上一帧位（外层速度估计/调试）
massBuf      : float
insideBuf    : uint      inside/outside 等值面分类（预处理 φ 符号；消融判定输入，≠存活位）
particleStateBuf : uint  [FIX S2-I4/I6] 0=alive 1=ablated 2=deleted（所有 per-particle/角点 kernel 据此 early-out）
frameBuf     : float3x3  每质点旋转 R（动态切割切点用，§5.7）
restNbrBuf   : float3[6] 6 邻居静态偏移（极分解参考，PAPER-SILENT C17）
nbrIdxBuf    : int[6]    6 邻居索引（−1=无/已切断）
```

### 3.2 RK45 物理 scratch [FIX S1-I4/I5/I8/I9/S2-C2/S2-I2]
```
forceIntBuf  : int3      内力 F_int 定点累加器（每 RK 阶段 ClearForces 清零；结构/弯曲 InterlockedAdd 写入）
extForceBuf  : float3    外力，物理帧首 = 重力 + 上帧 toolReactionBuf；ClearForces **不**清它；F_ext 项读它
toolReactionBuf : int3   [FIX S3-I6] 触觉散射的刀具反力定点累加器（§6.4 写，物理帧首被读入 extForceBuf 后清零）
kSlopeBuf    : (float3 dx, float3 dv)[7]  7 个斜率 k1..k7（按 Eq18 Butcher 节点）
yTrialPosBuf : float3    当前阶段试探位 y_n + h·Σ a_ij k_j（受力 kernel 读它而非 posBuf）
yTrialVelBuf : float3    当前阶段试探速
```
**不变量**：`posBuf/velBuf` 在一个外层步内保持 `y_n` **只读**，直到末尾 `Integrate(y5)` 才写回；所有阶段受力在 `yTrial*` 上算。

### 3.3 交点 Hermite（预处理，per-voxel 复制以便寻址）[FIX S1-I7]
```
voxelIsectOffsetBuf : int   每体素交点段起址
voxelIsectCountBuf  : int   每体素交点数
isectBuf : (float3 globalRest, float3 normal, int cornerA, int cornerB, float t)
           // 每条相交边的 Hermite；共享边**复制进 4 个入射体素**各一份
           // t = 沿边参数；世界位每帧 = lerp(cornerA.pos, cornerB.pos, t)（§4.2 重建）
```
> 决策：用 per-voxel 紧凑 isect 列表（共享边复制到 4 体素），让 §4.2 每体素线程直接取自己的交点；`normal` 取自预处理 `normalize(Gradient)`，复制的 4 份法线一致。

### 3.4 体素 + 特征点 [FIX S1-I11/S2-C3/S2-IMP3/S2-I5]
```
voxelCutMaskBuf : uint     12 位 cut 状态（bitwise-OR 更新）
voxelCornerBuf  : int[8]   → particle 索引
voxelExternalFPBuf : float3 + uint valid   外表面 QEF 特征点（每体素 1 个，按 voxelId 索引）
voxelExternalFPNormalBuf : int3            外表面 FP 法线定点累加器（按 voxelId）
featurePointBuf       : float3 + uint valid   切面分量质心特征点（仅内部切面用）；valid=0 表示空分量(count==0)无 FP [FIX S1-final]
featurePointNormalBuf : int3     切面 FP 法线定点累加器
cutFPAccumBuf : (int3 sum, int count)  [FIX S2-I3] 每切面分量的切点定点和+计数（求质心，ClearCounters 清零）
voxelFPCountBuf : uint     每体素活跃切面 FP 数（0 未切；up to 8 切后），按 voxelId
```
> **固定 8 槽块寻址（按全局 voxelId）[FIX S1-I11/S2-C3/S2-IMP3]**：`featurePointBuf`/`featurePointNormalBuf`/`cutFPAccumBuf` 大小 = `8 × 全体素数 (Nx*Ny*Nz)`；每体素块起址 = `8*voxelId`（**全局** voxelId，§1.3 编码；预处理静态分配，**无需前缀和/原子分配/紧凑索引**）。这样**内部体素被深切**（8 角全在等值面内、MC256 非相交）也有槽位，不会越界/串块。`voxelFPCountBuf[voxelId]` 选活跃数。`voxelExternalFPBuf` 也按 voxelId 索引，与内部切面 FP **分开存**：一个既相交又被切的体素 → 1 外 + N 内，互不覆盖。
> **triBuf 顶点的 FP 来源编码 [FIX S3-IMP1/IMP7]**：取代单独的反查 buffer——triBuf 顶点索引用最高位打标：**外表面顶点 = `voxelId`（高位清，因外表面 FP 每体素 1 个，索引即 voxelId）**；**内部切面顶点 = `0x80000000 | (8*voxelId+slot)`（高位置）**。渲染顶点着色器与 §6.4 反查均据此位判别取 `voxelExternalFPBuf[idx]` 还是 `featurePointBuf[idx&0x7FFFFFFF]`，并解出 voxelId（外=idx；内=(idx&0x7FFFFFFF)/8）。

### 3.5 结构/弯曲约束
```
springEdgeBuf : (int i, int j, float L0, uint alive)              // 每网格边一根结构弹簧
bendPairBuf   : (int i, int j, int k, float theta0, uint alive)   // 每质点 C(6,2)=15 对（含 3 共线对）
```
> **边界初始化 [FIX S1-I15]**：init 时若 `nbrIdx[j]==-1 || nbrIdx[k]==-1` 则该弯曲对 `alive=0`（与 §5.8 切时失活同构）。角/边/面质点(3/4/5 邻居)→ 3/6/10 个活跃对。

### 3.6 切点（每条 cut 边两个）[FIX S1-I12]
```
cutPointBuf : (float3 localOffset, int ownerParticle, int edgeId)
              // localOffset 为**唯一权威**参数化：world = owner.pos + owner.R · localOffset（3-DOF）
              // 捕获 §5.3 的 ±D/2 横向间隙，等价论文"相对边两端的局部坐标"
              // （删除原 axisSign/signedDist 二义参数）
```

### 3.7 输出三角 [FIX S1-I6/S2-I3/S2-IMP1]
```
triBuf      : (int v0,v1,v2)  append；顶点索引按 §3.4 高位标编码(external=voxelId / internal=0x80000000|(8*voxelId+slot))
triCounter  : uint           **每帧 1 次** SetCounterValue(0)（统一几何 pass 头部，外+切共用，§5.10/§4.6）
TRI_CAPACITY: const          预分配上限；写入前 InterlockedAdd(triCounter,n,base) 预留，guard if(base+n>TRI_CAPACITY)
indirectArgsBuf : uint[4]    **每帧 1 次** CopyCount(triCounter)（外+切三角都 append 完后，统一几何 pass 尾部）
```
> **统一几何 pass [FIX S2-IMP1]**：外表面三角(§4.3)与切面三角(§5.6) **append 进同一 triBuf**，一帧内只 1 次 `SetCounterValue(0)`（头）、1 次 `DC_Normals[barrier]`、1 次 `CopyCount`（尾）。`indirectArgs = 外三角数 + 切三角数`。§4.6 几何尾**只 append 外表面**、不 reset/不 CopyCount；§5.10 **续 append 切面**、不再 reset。

### 3.8 查找表 + 触觉状态
```
mc256Buf   : CPU 生成标准 MC 角点→边表
conn4096Buf: §5.4（compCount + vertToComp[8]；edgeToComp 已删除，见 §5.4/§5.6）
toolStateBuf : 单元素 [FIX S3-I3/I4/I7/IMP3]
   { float3 P_A; float4 q_A;            // 物理工具（每帧由 IHapticDevice 写）
     float3 prevP_A; float4 prevq_A;    // 上一帧（求 v,ω 有限差，每帧尾滚动）
     float3 P_B; float4 q_B;            // 代理工具（跨帧持久；P_B 由 Algorithm 1 推位，q_B 由 §6.2 朝向规则推进，见下）
     float3 capSegA; float3 capSegB; float r;   // 胶囊线段+半径
     float3 linVel_v; float3 angVel_w;          // 工具自身线/角速度（≠质点 velBuf；由 UpdateToolVelocity 算）
     uint initialized }                          // [FIX S3-v4final] 0=未 seed；首帧置 1（见 §6.6 首帧初始化）
hapticOutBuf : 单元素 { float3 force; float3 torque }  [FIX S3-IMP5] Eq27 输出；ComputeForceTorque 写，IHapticDevice 读
candidateTriBuf : int(triBuf idx) + candidateTriCounter  [FIX S3-M3/IMP4] 胶囊-AABB 广相近邻三角(存 triBuf 下标)；**每帧 SetCounterValue(0)** 后再 append，guard CANDIDATE_CAPACITY
contactBuf : 单元素 { float3 vms; float dist; int contactTriId; uint colliding }  [FIX S3-final] ProxyRelax(tid0) 写：最小分离向量、最近距、命中三角 triBuf 下标、碰撞标志；ComputeForceTorque/ScatterReactionForce 读（取代原 vmsBuf）
```
> **碰撞标志 colliding [FIX S3-final]**：`colliding = (gs_done==1) || (maxOuter 耗尽且仍穿透)`；非碰撞(gs_done==2)=0。供 §6.4 阻尼门控与反力散射、§6.2/§6.3 q_B slerp 共用同一判定。`contactTriId` = ProxyRelax 归约出的 argmin 三角的 triBuf 下标（论文复数"nearby faces"→可选散射所有 `gs_len<r` 的候选；单/复记 PAPER-SILENT）。
> **q_B 推进规则 [FIX S3-IMP3]（PAPER-SILENT，朝向跟踪）**：Algorithm 1 仅解算位置 P_t；朝向 q_B 论文未给。每帧把 q_B 朝 q_A `slerp` 一固定角步 `a_o`（§7）；**不碰撞时 `q_B:=q_A`**（Δq→单位四元数）；首帧/无解回退时 `q_B:=q_A` 初始化（避免零四元数）。q_B 在 §6.3 与 P_B 一同写回。（更正 v2 "q_B 由 Algorithm 1 推进"的错误表述。）
> **首帧代理状态初始化 [FIX S3-v4final]（PAPER-SILENT）**：ComputeBuffer 零初始化会使 `P_B=(0,0,0)`、`prevP_A/prevq_A=0`，导致首帧巨大 `ΔP`（启动幻影力）与虚假首帧速度。故首帧（或 `toolStateBuf.initialized==0` 时）一次性 seed **`P_B:=P_A, q_B:=q_A, prevP_A:=P_A, prevq_A:=q_A`**（紧随 `IHapticDevice` 写入首个 P_A/q_A 之后，§6.6）。校验：首帧工具远离组织 ⇒ ΔP≈0、F≈0、速度无跳变（§6.7 oracle）。

---

## 4. Stage 1 — 基座（可变形网格 + 物理 + DC 外表面重建）

> 论文 2.2 + 2.1 开头 + Eq1-5,9-21 + Fig 2.1/2.2/2.10。范围：未切割管线——立方体/球受力变形、DC 外表面水密跟随。

### 4.1 预处理（CPU，`BackgroundGrid.cs` + `ILevelSetProvider`）
1. `ILevelSetProvider.Sample(p)`=解析 SDF（Box/Sphere）；`Gradient(p)`=法线。
2. AABB 包围 → **均匀**网格（voxel=0.25cm）。**不做八叉树**（决策见下框）。
3. 每**全局角点**算 φ 符号 → `insideBuf`（角点共享，只算一次，I02）；`particleStateBuf=alive`。
4. **MC256 表**（角点 8 位 mask）定位交边；交边上 φ 线性插值求交点 → 存 `globalRest+normal+cornerA/B+t`，并**复制进该边 4 个入射体素**的 isect 段（I05：只算最外层边界网格交点）。
5. 建结构弹簧（每边 L0=边长）；建弯曲对（15/质点，θ0=静态邻居夹角；缺邻居对 `alive=0` [FIX S1-I15]）；填 `restNbrBuf`。
6. 静态分配 `featurePointBuf`/`featurePointNormalBuf`/`cutFPAccumBuf`(8×**全体素数**，按 voxelId 索引)、`voxelExternalFPBuf`(按 voxelId)、`voxelFPCountBuf`。

> **决策（C01 八叉树）**：论文写"uniform grid + adaptive octree"，但其物理(6 邻居)、8 顶点/12 边图、固定 0.25cm 全假设均匀网格；八叉树会产生 T-junction/变邻居数，与 6 弹簧/15 弯曲拓扑冲突。**v1 采用纯均匀网格**，八叉树视为外表面预处理的可选细分、不进物理网格。标 `// PAPER-SILENT/D-grid`。

### 4.2 DC 外表面特征点（GPU，每体素 1 个，Eq1-5 + D1/D5）
每与等值面相交体素 1 线程，输入取自 `voxelIsectOffset/Count → isectBuf`（per-voxel 复制保证可寻址 [FIX S1-I7]）：
```
// [FIX S1-I2] 先把交点世界位按当前变形角点重建（外表面随变形跟随）
for each isect of voxel: p_i = lerp(cornerA.pos, cornerB.pos, isect.t)   // 由 RebuildIsectWorld kernel 完成
n_i = isect.normal                              // PAPER-SILENT：法线暂冻结于静态(可选随体素帧旋转)
x_k = mean(p_i)                                 // Eq2 初始平均点
voxelCenter = mean(8 个当前变形角点)             // Eq5 box 用变形体素中心
x_prev = x_k
for it in 0..m-1:                               // m=最大迭代，PAPER-SILENT 默认 20（§7）
    F = Σ_i  n_i * dot(n_i, p_i - x_k)          // DEVIATION D1：QEF 负梯度，当前 x_k
    a = 0.1 * (1 - it/m)                         // Eq4
    x_next = x_k + a * F                          // Eq4
    if boxSDF(x_next, voxelCenter, L) >= 0:       // DEVIATION D5：抵达/越界则停
        break
    x_prev = x_k = x_next
voxelExternalFPBuf[voxel] = x_prev               // D5：回写最后内部点（非夹回越界点）
```
- `boxSDF`=Eq5（max 形式，负=内部）。外表面 FP **用 QEF**；切面 FP **用质心**（C05/C10，绝不混用）。

### 4.3 外表面缝合 + 法线（GPU，per-particle）
- 对每条**相交边**：取其 4 共享体素的 `voxelExternalFPBuf` → 四边形 → 2 三角（C06 stencil）。
- **缠绕 [FIX C07]**：依该边 inside→outside / outside→inside 决定顶点序，保证法线朝外（PAPER-SILENT）。固定对角线避免缝隙。
- **append [FIX S1-I6]**：`InterlockedAdd(triCounter,2,base)`；`if(base+2>TRI_CAPACITY) return;` 写 base..base+2。
- **法线 [FIX C08/S1-I3]**：每三角面法线**定点 InterlockedAdd** 散射到 3 个 FP 顶点的 `*NormalBuf(int3)` → **barrier** → 归一化 kernel（`int3/SCALE` 后 normalize）。防闪烁/黑面。

### 4.4 物理：mass-spring（GPU，Eq9-16 + D2）
**结构力**（per-particle gather，无原子，Eq11+D2）：
```
F_s_i = Σ_{alive j}  [ -k_s*(length(xt_i-xt_j)-L0_ij)*normalize(xt_i-xt_j) - c_s*(vt_i-vt_j) ]
// xt/vt 取自 yTrial*；累加到 forceIntBuf（定点）
```
**弯曲力**（per-pair，定点原子散射，Eq13-16，C29/C30/C31）：
```
for each alive pair (i;j,k):                       // n=0..14
    N_ij = normalize(xt_i-xt_j); N_ik = normalize(xt_i-xt_k)
    c = clamp(dot(N_ij,N_ik), -1, 1)
    // [FIX S1-I14/S2-C5 + S1-minor] 奇异保护前置且单常量：先判原始量 s2，再开方
    float s2 = 1 - c*c;                             // sin²θ
    if s2 <= EPS_SIN2:  continue                    // EPS_SIN2≈1e-8 唯一阈值；共线对(c=±1)跳过整对，防 NaN
    d = sqrt(s2)                                    // 不再 max-floor，避免双常量失配死代码
    theta = acos(c)
    cdot = dot(d/dt N_ij, N_ik) + dot(N_ij, d/dt N_ik)   // dN/dt 由阶段速度解析(C33 PAPER-SILENT)
    theta_dot = -cdot / d
    s = k_b*(theta - theta0_jk) + c_b*theta_dot     // C29：阻尼在 Eq14 标量内
    F_M_ij = s * cross(cross(N_ik, N_ij), N_ij)     // C30：ik×ij
    F_M_ik = s * cross(cross(N_ij, N_ik), N_ik)     // C30：ij×ik（相反）
    atomicAddFixed(forceIntBuf[j], F_M_ij)
    atomicAddFixed(forceIntBuf[k], F_M_ik)
    atomicAddFixed(forceIntBuf[i], -(F_M_ij + F_M_ik))   // C31：中心反作用
F_int_i = F_s_i + F_b_i                             // Eq10
```
- **力累加 barrier**（C45a）：每 RK 阶段所有结构/弯曲写完才求该阶段斜率。

### 4.5 时间积分：自适应 Dormand-Prince RK45（GPU，Eq17-21 + D3/D4）
- 状态 `y=[x,ẋ]`；`f(t,y)=[ẋ, (F_ext + F_int)/m]`（**D3 修正**：F_int=实际回复力，牛顿 ⇒ +F_int；F_ext 读 `extForceBuf`）。
- **外层固定 dt=0.02s（帧预算），内层自适应 h 子步累加到 dt，末步裁剪正好落 dt**（C40，PAPER-SILENT）。
- 7 斜率 Eq18（DP Butcher tableau，逐字）；**`y(4)` 6 项(含 k7)、`y(5)` 5 项(k1,k3,k4,k5,k6；无 k2/k7，k7=FSAL 即 y(5) 下一步的 k1)**（Eq19）[FIX S1-minor]。
- **传播 5 阶解 y(5)**（局部外推，DOPRI5 标准，C38）。
- **误差 [FIX S1-minor]**：`Δ=‖y(5)−y(4)‖` 中 `y_n` 抵消，等价 `Δ=‖h·Σ(b5_i−b4_i)·k_i‖`，**直接从 `kSlopeBuf` 算（与 y_n 无关）**，故 `ComputeError` 不依赖 `posBuf/velBuf` 是否已被 `Integrate_y5` 覆盖（Eq20，全状态全质点范数，C53）。
- 步长：`h_n=h·0.9·(ε/Δ)^{1/5}`（**D4**），`ε=1e-3`，clamp `0.2h<h_n<5h`（Eq21）。
- **接受策略：always-accept，adapt-forward（不重试）**（C39，PAPER-SILENT，契合实时预算）。

### 4.6 Stage 1 Kernel + Dispatch 顺序（权威）[FIX S1-I4/I5/S2-C2/S2-IMP1/S3-I6]
```
物理帧首：extForceBuf = 重力 + toolReactionBuf(上帧)；toolReactionBuf = 0   # [FIX S3-I6] 1 帧延迟应用刀具力
内层自适应子步循环（h 累加到 dt）:
  for s in 1..7:                                   # DOPRI 7 阶段
     BuildTrialState_s   # yTrial* = y_n + h·Σ a_sj·k_j  （a 取 Butcher）
     ClearForces         # forceIntBuf=0（不动 extForceBuf）
     AccumulateStructural(yTrial)
     AccumulateBending(yTrial)
     [force barrier]                               # C45a
     ComputeSlope_s      # k_s = f(t_n+c_s·h, yTrial) 写 kSlopeBuf[s]
  ComputeError(Δ) → AdaptStep(h_n) [D4]            # 始终接受；Δ 由 kSlopeBuf 算
  # [D6] AdaptStep/ErrMax-readback 是 per-frame（内层循环结束后一次），非 per-substep
  Integrate_y5           # 由 k1,k3,k4,k5,k6 组 y(5) 写回 posBuf/velBuf
# === 统一几何 pass（每帧 1 次，外+切共用 triBuf，[FIX S2-IMP1]）===
ClearCounters           # triCounter.SetCounterValue(0) + 清零 FP/法线/cutFPAccum 累加器（仅此一处）
UpdateVoxelCorners → RebuildIsectWorld → DC_FeaturePoints(QEF, 写 voxelExternalFPBuf)
  → DC_Stitch(外表面三角 append, 顶点高位清=voxelId)
  # ↓ Stage 2 在此续接（§5.10），把切面三角 append 进同一 triBuf；无切割时跳过
  → DC_Normals[normal barrier]   # [FIX S2-final] 归一化 voxelExternalFPNormalBuf 与 featurePointNormalBuf **两套**
  → CopyCount(indirectArgs)       # 1 次，外+切都 append 完
```
> §8 据此：**每个 RK 子步 7 次力 barrier + 每帧 1 次法线 barrier**；几何 pass 每帧只 1 次 reset/CopyCount。

### 4.7 Stage 1 验证标准
- EditMode：解析 SDF/梯度对拍；单结构弹簧/单弯曲对受力手算对拍；**静止质点 3 共线弯曲对应得有限 ~0 力**（θ̇ 奇异保护，S1-I14）；边界质点(3/4/5 邻居→3/6/10 活跃对)受力有限；QEF 在解析球面上 FP 落面附近。
- Unity 内（用户跑）：立方体自由下落/受力变形回弹**能量稳定不发散**；DC 外表面**水密且随变形跟随**（含 RebuildIsectWorld）；无黑面/闪烁。

---

## 5. Stage 2 — 切割

> 论文 2.1 全 + Eq6-8 + Fig 2.3-2.9。依赖 Stage 1 变形网格与每质点坐标系。

### 5.1 刀路与扫掠面（`CuttingTool.cs`，2.1.3，C22）
- 刀具=**线段**，端点 `S`(上)/`E`(下)，刀厚 `D`。
- 扫掠面=上帧 `S_k,E_k` 与本帧 `S_{k+1},E_{k+1}` 四边形，**对角线 S_k→E_{k+1}**：`T1=(S_k,E_k,E_{k+1})`、`T2=(S_k,E_{k+1},S_{k+1})`（固定对角线避漏检）。任意帧持续存在（低帧率不漏检）。

### 5.2 碰撞检测 + cut 位传播（`CutDetector.cs`，Eq6-8，C21/C23/C24 + FIX S2-C1/I7）
- **ray=体素网格边**（`O`=端点，`D`=边向量）；**triangle=扫掠面两三角**（C21，不可反）。
- 逐字 Eq7-8 Cramer：`T=V0−O`；`det=(L2×D)·L1`（三式共用）；`t=((L2×T)·L1)/det`、`u=−((L2×D)·T)/det`、`v=((D×T)·L1)/det`（u 带负号）。`|det|<eps`→平行无命中。
- 命中：`u≥0,v≥0,u+v≤1` **且 `0≤t≤1`**（C24，限有限边内）。命中点 `P_hit=O+t·D`。
- **cut 位传播到 4 体素 [FIX S2-C1/I7]**：用 `(axis,slot)→(voxel,localEdge)` stencil 解析该全局边的 ≤4 个入射体素，对每个 `voxelCutMaskBuf[voxel] |= 1<<localEdge`。
- EditMode：切一条内部边 → 4 个共享体素对应位都置 + 4 个 CONN4096 config 一致分裂。

### 5.3 切点：每条 cut 边两个（C16/C19/C25/C47）
- 由 `P_hit` 沿**刀厚方向**偏移 `±D/2` 生成两切点，各绑边一端质点（偏移方向=切平面内法线/刀厚轴，PAPER-SILENT）。
- 存为 owner 局部量 `cutPointBuf.localOffset`；world=`owner.pos+owner.R·localOffset`（§5.7）。间隙=D（mm 级，亚体素；D→0 近细缝）。
- **owner 必须 alive [FIX S1/S2-minor]**：`ownerParticle` 须为 `state==alive` 的端点。消融边若内部端已 `deleted`，则把切点绑到**存活邻居**端（在其帧下重算 localOffset）；两端皆 deleted→不产切点（区域已消失）。

### 5.4 连通性表 CONN4096（`ConnectivityLUT.cs`，CPU 生成）[FIX S2-C4/I1]
对 config∈[0,4096)（12 位 cut mask），在 **uncut 边子图**（cut 位=断连）上并查集求连通分量，输出 2D 数组：
```
compCount[config]        : 分量数（≤8）
vertToComp[config][8]     : 每顶点所属分量 id（孤立顶点=自成一分量）   ← 唯一输出，驱动切面 FP 与缝合
```
> **顶点键是唯一键 [FIX S2-C4/IMP2/IMP4]**：切面 FP 与缝合**全部用 `vertToComp`**（每分量→其顶点→各顶点 3 边→把绑在该顶点端的切点累加进该分量质心）。被切边两端在不同分量，单"每边一个分量 id"无法编码；且"全 12 边皆切→8 个单点分量"(C48)只有 `vertToComp` 能表达。
> **删除 edgeToComp [FIX S2-IMP2]**：原拟的 `edgeToComp`（仅 uncut 边）在设计中**无任何消费者**（不存在独立的 uncut 缝合步），且对 cut 边语义不成立（cut 边跨两分量）。故从 LUT/§3.8/`ConnectivityLUT.cs` 全部删除。
> **分量 id 致密化 [FIX S2-minor]**：`vertToComp` 的分量 id **致密重编号 0..compCount-1**（并查集压缩），使 `featurePointBuf` 槽下标==分量 id、`voxelFPCountBuf=compCount` 恰好枚举占用槽。EditMode 断言 `max(vertToComp[config])==compCount[config]-1`（全 4096 项）。
- **EditMode 必测**：Fig 2.4 例（切 e0,e2,e4,e6→2 分量 {V0,V3,V4,V7}/{V1,V2,V5,V6}）；全切→8 单点；高 popcount 十字切。

### 5.5 连通分量 + 切面特征点（GPU，C05/C14 + FIX S2-C4/IMP3）
- 每被切体素查 CONN4096 → `compCount` → 用**固定 8 槽块按全局 voxelId**(`8*voxelId`，`voxelFPCountBuf[voxelId]=compCount`)放分量 FP（**内部深切体素也有槽**，§3.4）。
- 每分量 FP = **该分量切点算术质心**（不用 QEF）；用 `cutFPAccumBuf`(定点 sum+count) 原子累加。**空分量保护 [FIX S1-final]**：`for slot in 0..compCount-1: if count>0: featurePointBuf=sum/count, valid=1; else valid=0（不写、不产 NaN）`。（半消融边只绑存活端→删除端分量 count=0，会触发此分支。论文 L344：分量 FP 是其切点质心，零切点⇒无 FP。）
- **按顶点累加 [FIX S2-C4]**：每条 cut 边的两个切点，绑 V_a 端的累加进 `cutFPAccum[8*voxelId + vertToComp[config][V_a]]`、绑 V_b 端的累加进 `...[vertToComp[config][V_b]]`（C14：被切边切点经各自端点各贡献一次）。
- **消融归属 [FIX S1/S2-minor]**：消融半边（内部端已删）的切点只累加进**存活端**分量（删除端角点缺失、不贡献）。

### 5.6 缝合（GPU，per-particle，C15/C54 + FIX S2-IMP2/IMP4/I5/I8）
- 遍历每**质点**的 **6 轴向边 + 8 八分体素**（非单体素 12 边）。
- **cut 边分量选择按本质点角点（唯一正确键）[FIX S2-IMP2/IMP4]**：对每条本质点入射的 cut 边，在其 ≤4 共享体素各取内部切面 FP，**分量 = `vertToComp[该体素config][localCornerOf(本质点, 该体素)]`（即含本质点角点的分量）**，FP 地址 `featurePointBuf[8*voxelId + 分量]`。**绝不用 edgeToComp，绝不用"该边所属分量"**（cut 边跨两分量、会重新焊死接缝=MC2024 坑 C54）。这样 V1 与 V2 各取自己一侧两 FP（论文 L352-354 "分别连接"）。
- **部分四边形规则 [FIX S2-I8/S1-final]**：只收"存在且持有该分量切面 FP（`valid==1`）"的共享体素（`valid==0` 空分量槽排除）；**4 个有效→2 三角（§4.3 固定对角线/缠绕）；3 个→1 三角（按这 3 角点、缠绕与四边形版同向）；<3→不产**。（处理切区边缘/切口与外表面交界/消融删体素/AABB 边界。）
- **顶点编码 [FIX S2-I5/S3-IMP1]**：external 顶点写 `voxelId`（高位清）、internal 顶点写 `0x80000000|(8*voxelId+slot)`（高位置），即 §3.4 的来源编码（渲染与 §6.4 据此判别）。法线同 §4.3。
- **EditMode**：Fig 2.4 config 下，V0 侧质点在所有共享体素都选 {V0,...} 分量、V1 侧选 {V1,...} 分量（翻转遍历质点→翻转所选分量 id），证明接缝不重焊。

### 5.7 动态切割：每质点坐标系（`ParticleFrames.cs`，2.1.2，C17/C18/C51）
- 变形后网格非立方体，每质点旋转坐标系 `R`。论文只引 Berndt[22] 无公式 → **决策**：**极分解/shape-matching** 从 `restNbrBuf`→当前邻居偏移恢复正交 `R`（PAPER-SILENT，注引 [22]）。
- 切点锚定**单个**质点的帧（Fig 2.7：红切点随 V3 帧摆动，不依赖远端），world 每帧重算（C18 取 per-particle-frame 读法）。支持 Fig 3.9 重力张开 / Fig 3.12 边拖边切。

### 5.8 切断弹簧（GPU，C20/C35）
切某边 e：(1)结构弹簧 `alive=0`；(2)**作废所有以 e 为臂的弯曲对**（`alive=0`）；(3)两端 `nbrIdx` 互置 −1。否则跨切口仍传弯矩=幽灵力。静长/静角是 init 不变参考。

### 5.9 消融切割（`AblationCutter.cs`，2.1.4，C26/C27/C28 + FIX S2-I4/I6）
有序两遍（**独立 Dispatch、不可合并**）：
1. **Pass1**：刀路范围内 **等值面外** 质点 `particleState=deleted`（不产切点、不参与受力/表面）。
2. **Pass2**：范围内**内部**质点 `particleState=ablated`→`deleted`；标**消融边**（与 ablated 质点相连边，含通向存活邻居的边，C27）；按 §5.8 切断其弹簧/弯曲；在消融边产切点。
3. **汇入同一管线**：消融边写入 `voxelCutMaskBuf`(传 4 体素) → CONN4096 → 质心 FP → 重缝重绘（C28）。
- 所有 per-particle/角点 kernel 对 `state!=alive` early-out；体素的 deleted 角点视为缺失（§5.6 部分四边形）。

### 5.10 Stage 2 Kernel + Dispatch 顺序（接入 §4.6 统一几何 pass，**不重复 reset/CopyCount**）[FIX S2-IMP1]
```
# 物理子步后、§4.6 几何 pass 内：
DetectCut(刀路 vs 体素边, cut 位传 4 体素) → SeverSprings(反馈物理 §5.8) → [若消融] Ablate Pass1 → Pass2
  → LookupConnectivity(CONN4096)
  # ↓ 续接 §4.6 的统一几何 pass（ClearCounters 已在 pass 头执行过一次）：
  → ComputeCutFeaturePoints(顶点键质心, cutFPAccum) → BuildTriangles(部分四边形, 续 append 进同一 triBuf)
  # DC_Normals[barrier] 与 CopyCount 由 §4.6 几何 pass 尾统一执行（外+切都 append 完）
```
> **不再单独 ClearCounters/CopyCount [FIX S2-IMP1]**：`triCounter`/FP/法线/`cutFPAccum` 的清零统一在 §4.6 几何 pass 头的唯一 `ClearCounters`；切面三角 `InterlockedAdd(triCounter,…)` 接着外表面基址继续，末尾统一 1 次 `DC_Normals`+`CopyCount`。`indirectArgs = 外三角 + 切三角`。

### 5.11 Stage 2 验证标准
- EditMode：CONN4096 全 4096 项与并查集参考一致；Fig 2.4 例 + 全切 8 单点 + 高 popcount(C48)；cut 位传 4 体素一致；M-T 命中/`0≤t≤1` 边界例；边界 cut 边仅 3 共享体素→1 三角(S2-I8)。
- Unity 内：平面切**断图**、**两个非流形面**、宽=D 随 D 变宽(Fig 3.8)；十字/曲线/多刀(Fig 3.10)；**切割增多帧率仅缓慢下降不恒定**(C46)；重力/拖拽切口跟随(Fig 3.9/3.12)；消融体积损失重绘正确。

---

## 6. Stage 3 — 触觉力反馈

> 论文 2.3 全 + Eq22-27 + Algorithm 1 + Fig 2.11。依赖 Stage 2 三角面。

### 6.1 工具 SDF + 广相（`ToolSDF.cs`）
- 刀具≈**细长胶囊**（`capSegA/B`+半径 `r`）→ SDF；三角面 SDF（点到三角最近点）。两者皆凸。
- **候选三角广相 [FIX S3-M3]**：胶囊-AABB 扫描表面三角 → `candidateTriBuf`（限制 Algorithm 1 遍历集）。

### 6.2 代理 vs 物理工具（Eq22-25，C41/C42 + 数值加固 S3-I8）
- `P_A`=物理（隐藏跟手，`toolStateBuf` 每帧由设备写）；`P_B`=代理（渲染，留表面外，跨帧持久）。
- `ΔP=P_B−P_A`（C41）；`Δq=q_B⊗q_A⁻¹`（proxy⊗inv-physical，左乘，(w,x,y,z) 序，C42）。
- **加固 [FIX S3-I8]**（PAPER-SILENT）：先 `Δq` 规范化、`Δq_w<0` 取反；`Δq_w=clamp(Δq_w,-1,1)` 再 `θ=2·arccos(Δq_w)`（Eq24）；若 `sin(θ/2)<1e-6` 则 `k=0,τ_rot=0`，否则 `k=[Δq_x,Δq_y,Δq_z]/sin(θ/2)`（Eq25）。防 NaN 力矩。
- **q_B 朝向推进 [FIX S3-IMP3]**（PAPER-SILENT）：Algorithm 1 只解位置；朝向每帧 `q_B = slerp(q_B, q_A, a_o)`（§7 角步），**不碰撞时 `q_B:=q_A`**，首帧/无解回退 `q_B:=q_A`（防零四元数）。q_B 在 §6.3 与 P_B 一同写回 `toolStateBuf`。

### 6.3 finger-proxy 交替投影（`FingerProxy.cs`，Algorithm 1，C43 + FIX S3-I1/I2/I9/IMP2/IMP6/M1/M2）
**GPU 组协作分解 [FIX S3-I1/final]**：`Dispatch(1,1,1)`，**固定编译期组大小** `[numthreads(CANDIDATE_CAPACITY,1,1)]`，groupshared 也按 `CANDIDATE_CAPACITY` 声明（**绝不用运行时 K**——SM5.0 要求 numthreads 与 groupshared 维度为编译期常量）。活跃车道 `K=candidateTriCounter`，`tid≥K` 为非活跃车道。**关键 [FIX S3-IMP2]：所有线程（含 tid≥K）每轮都执行完整循环体与两个 barrier，break 读组一致 `gs_done`**（绝不让线程在 barrier 掉队 = UB/死锁）：
```
groupshared float3 gs_Dvec[CANDIDATE_CAPACITY]; groupshared float gs_len[CANDIDATE_CAPACITY];   // gs_len 是标量
groupshared float3 gs_vms, gs_Pt; groupshared int gs_done; groupshared int gs_triId;
if (tid==0) gs_Pt = P_B;  GroupMemoryBarrierWithGroupSync     // 取上帧持久代理位
for outer in 0..maxOuter-1:                        // [FIX S3-I2] PAPER-SILENT 外层上限 ~48，防 TDR
    Dvec = sentinel
    if (tid < K):                                  // 仅活跃车道跑内层；非活跃车道写 +INF 哨兵
        P = gs_Pt; i=0
        while i<100:                               // 内层交替投影
            A = SDF(triangle[tid], P); P2 = SDF(tool, A)   // Eq26（A_i,P_{i+1} 每遍推进，见注）
            Δ = |P - A| - |P2 - A|
            if |Δ|<ε: Dvec = P - A; break
            P = P2; i++
    gs_Dvec[tid]=Dvec; gs_len[tid]=(tid<K)? |Dvec| : +INF
    GroupMemoryBarrierWithGroupSync                // barrier#1：所有线程（含非活跃）都写完
    if (tid==0):                                   // 线程0 归约 + 唯一推进（K==0 时全 +INF→哨兵）
        j* = argmin_{j in 0..CAP-1} gs_len[j];  v_ms = gs_Dvec[j*]
        // [FIX S3-IMP6] 碰撞门：v_ms 即最近距方向
        if (gs_len[j*]==+INF) or (‖v_ms‖ ≥ r):    // 无候选/未穿透 → 朝物理工具走（Fig3.1 Colliding?N）
            gs_Pt += clamp_step(P_A - gs_Pt, maxStep); gs_done = 2; gs_triId = -1
        else:                                      // 穿透：沿最小分离推出
            gs_Pt += clamp_step(a * v_ms, maxStep); gs_done = (‖v_ms‖<ε) ? 1 : 0; gs_triId = candidateTriBuf[j*]
        gs_vms = v_ms
    GroupMemoryBarrierWithGroupSync                // barrier#2：结果发布；所有线程读同一 gs_done
    if (gs_done != 0): break                       // 组一致 break（1=收敛碰撞, 2=未碰/无解）
if (tid==0):
    bool colliding = (gs_done==1) || (gs_done==0 /*maxOuter 耗尽仍穿透*/)
    P_B = gs_Pt;  q_B = colliding ? slerp(q_B,q_A,a_o) : q_A    // [FIX S3-IMP3]
    contactBuf = { vms:gs_vms, dist:‖gs_vms‖, contactTriId:gs_triId, colliding:colliding?1:0 }  // [FIX S3-final]
```
- 穿透（最近距<r）沿最小分离推出，代理始终在外；未碰撞/空广相(K==0) 则代理追物理工具（ΔP→0，无幻影力 [FIX S3-IMP6/final]）。
- **PAPER-SILENT 注 [FIX S3-M1]**：Algorithm 1 第 11-12 行印的固定下标 A_0/P_1 会冻结循环；我们按 Eq26 每遍推进 `A_i=SDF(tri,P_i),P_{i+1}=SDF(tool,A_i)`，Δ/D 定义逐字保留。
- **PAPER-SILENT 注 [FIX S3-minor]**：`v_ms` 每外层迭代**重算**（P_t 已移动，累积旧 Dvec 失效），非论文 L5/16/17 跨迭代持久最小值。
- **PAPER-SILENT 注 [FIX S3-final]**：固定组+哨兵车道掩码、碰撞标志/命中三角载体(`contactBuf`)、`colliding` 含 maxOuter 耗尽仍穿透——均补全论文未给项。

### 6.4 力/力矩 + 反作用（`HapticForce.cs`，Eq27，C44/C50 + FIX S3-I5/I6/IMP1/IMP5/IMP7/M4）
- `ComputeForceTorque` 写 `hapticOutBuf`：`F=K_p·ΔP − colliding·D_p·v`；`τ=K_o·θ·k − colliding·D_o·ω`（**阻尼乘 `contactBuf.colliding` 门控，仅接触时生效** [FIX S3-final]；v,ω 来自 §6.5 `UpdateToolVelocity`）[FIX S3-IMP5]。`IHapticDevice` 每帧读回 `hapticOutBuf`。
- **反作用散射到质点 [FIX S3-IMP1/IMP7/final]**：`if contactBuf.colliding==0: 不散射`（无接触无反力）。否则取命中三角 `tri = triBuf[contactBuf.contactTriId]`，其 3 顶点据 §3.4 高位标分流求 voxelId——**external 顶点** `voxelId=idx`；**internal 顶点** `voxelId=(idx&0x7FFFFFFF)/8`——再 `voxelId→voxelCornerBuf[8]` 取 8 角点；设**阈值** `reactionForceThreshold`(§7) 后 `atomicAddFixed(F)` 进 `toolReactionBuf`（仅 alive 角点）。（外表面接触是主路径；论文复数"nearby faces"→可选对所有 `gs_len<r` 候选散射，记 PAPER-SILENT。）
- **力持久 + 时序 [FIX S3-I6/M4]**：散射写 `toolReactionBuf`（定点）；**物理帧首** `extForceBuf = 重力 + toolReactionBuf(上帧)` 后清 `toolReactionBuf`（§4.6）。`ClearForces` 只清 `forceIntBuf`。故刀具力以 **1 帧延迟**进入下帧全部 RK 阶段（契合论文多线程异步）。论文"update particle positions"在此实现为阈值力加(Fig 3.1 "Add Tool Force")，记 PAPER-SILENT。

### 6.5 跨组数据流 + Dispatch（Fig 3.1，C44/C50 + FIX S3-IMP4/minor）
四 hand-off：(1)物理位姿→几何 `Update Voxel Corner Points`；(2)几何 `Cut`→物理 `Update Links & Orientation`(断弹簧反馈)；(3)几何 `Construct Triangles`→触觉；(4)触觉 `Force(阈值)`→物理 `toolReactionBuf`(下帧生效)。触觉 `Colliding?` N 分支在 §6.3 内由碰撞门处理（代理追物理工具）。
**Dispatch 顺序**：
```
candidateTriBuf.SetCounterValue(0)              # [FIX S3-IMP4] 每帧重置广相 append 计数
→ UpdateToolVelocity   # [FIX S3-minor] linVel=(P_A−prevP_A)/dt; angVel 由 q_A⊗prevq_A⁻¹ 轴角/dt
→ UpdateToolSDF(胶囊-AABB 广相填 candidateTriBuf, guard CANDIDATE_CAPACITY)
→ ProxyRelax(Algorithm 1, §6.3)                # 写回 P_B/q_B
→ ComputeForceTorque(写 hapticOutBuf)
→ ScatterReactionForce(→ toolReactionBuf)
→ 帧尾滚动 prevP_A=P_A, prevq_A=q_A
```

### 6.6 设备适配（`IHapticDevice.cs`）
算法只产力/力矩向量；`IHapticDevice` 留接口，每帧写 `toolStateBuf.P_A/q_A`、读回 `hapticOutBuf` 力/力矩。OpenHaptics/Omni 绑定为可选适配桩（本环境无设备，不阻塞算法）。
- **首帧 seed [FIX S3-v4final]**：写入首个 `P_A/q_A` 后，若 `toolStateBuf.initialized==0` 则 `P_B:=P_A, q_B:=q_A, prevP_A:=P_A, prevq_A:=q_A, initialized:=1`（防启动幻影力/虚假速度）。

### 6.7 Stage 3 验证标准
- EditMode：胶囊/三角 SDF 与最近点解析对拍；Eq22-25 四元数偏差单测（含 `Δq_w≈1`、`>1`、`<0` 加固例 S3-I8）；Algorithm 1 收敛/无解回退例；**近而未穿透(‖v_ms‖≥r)时代理追物理工具、ΔP→0 无幻影力**(S3-IMP6)；**K=0/K=1 不卡死(全 barrier 组一致)且走 Move-Proxy、ΔP→0**(S3-final)；**移动但不碰撞⇒F==0 且 τ==0**(阻尼门控 S3-final)；external 顶点接触散射到正确体素 8 角点(S3-IMP7)。
- Unity 内：工具压下不穿透留表面；输出平滑力/力矩曲线（Fig 3.7：z 先升、~2s 后 x/y 升）；非收敛接触不卡死(maxOuter)。

---

## 7. 参数（论文 Section 3.4 + 补全）

| 参数 | 值 | 备注 |
|---|---|---|
| 体素尺寸 | 0.25 cm (=2.5mm) | 刀厚 D=0.1–0.4 **mm**，亚体素(C47) |
| 质点数(参考) | 54,301 | |
| 结构刚度 k_s | 7.5×10⁴ N/m | |
| 弯曲刚度 k_b | 2×10⁴ N/m | |
| 结构阻尼 c_s | 0.92 | |
| 弯曲阻尼 c_b | 0.9 | |
| 外层步 dt | 0.02 s (50Hz, 帧预算) | 内层自适应 h 子步(C40) |
| RK45 误差阈 ε | 1×10⁻³ | |
| QEF 最大迭代 m | 20 | `// PAPER-SILENT` [FIX S1-M1] |
| finger-proxy 外层上限 maxOuter | ~48 | `// PAPER-SILENT` TDR 安全(S3-I2) |
| 反作用力阈值 reactionForceThreshold | 调参 | `// PAPER-SILENT` [FIX S3-M4] |
| 代理朝向角步 a_o | ~0.2 | `// PAPER-SILENT` q_B slerp(S3-IMP3) |
| K_p,D_p,K_o,D_o,胶囊 r,a=0.1,maxStep | 论文未给→引擎内调参 | `// PAPER-SILENT` |
| 弯曲奇异阈 EPS_SIN2 | 1e-8 | sin²θ 单阈值(S1-I14) |
| 候选三角上限 CANDIDATE_CAPACITY | 预分配 | append guard(S3-IMP4) |
| 定点 SCALE(力/法线) | 1<<10 / 1<<20 | 平台实现(§1.3) |

---

## 8. 已知局限（诚实对待，设计为"绕过"而非偷偷"修好"）

- **体积损失**仍存在（DC 内部切面用质心，比 MC 低但非零）。不试图消除。
- **特征点漂移**：大变形下切口两侧可能**突然**大距离分离；论文缓解=切后 FP 上帧→当前帧插值（视觉 hack）。v1 可选实现，标视觉修正。
- **帧率非恒定**：随切割缓慢下降（仍渲染更多体素，C46）。校验目标=缓慢下降。
- **不支持无限切割/自适应网格未入物理**（与 §4.1 八叉树决策一致）。
- **同步点**：每 RK 子步 **7 次力 barrier** + 每帧 **1 次法线 barrier**（两类独立，C45）。
- **触觉非收敛**：grazing 接触靠 `maxOuter` + 无解回退 + step clamp 保证不卡死/不瞬移（S3-I2/I9）。

---

## 9. 验证与审核计划（每 Stage 多 subagent）

每个 Stage 的**设计/代码**完成后，派多个 subagent **对抗式审核**（4 视角，非冗余）：
1. **算法对齐审核员**：逐公式/逐图对照论文，核 §2 台账之外是否有未声明偏离。
2. **GPU 可行性/竞态审核员**：核 ComputeShader 落地、定点原子/barrier、buffer 读写竞态、Dispatch 顺序、RK 多阶段重算。
3. **数据结构完整性审核员**：核 §3 buffer 字段、索引约定(Fig 2.4)、CONN4096(vertToComp)、动态收缩、固定 8 槽寻址。
4. **边界/退化审核员**：180° 弯曲对、平行 ray、t∈[0,1]、十字切 >2 分量、消融两遍序、被切边切点归属、boundary 邻居<6、θ̇ 奇异、proxy 无解。

**对抗式确认**：每条"问题"再派独立 skeptic 复核（默认证伪、多数通过才采纳）。修复后复审，**该 Stage 全过才进入下一 Stage 代码撰写**。

EditMode oracle（本环境可静态设计、用户可跑）：4096 表(含 vertToComp)、解析 SDF/梯度、QEF 小网格、单弹簧/弯曲(含共线对有限性)、cut 位传 4 体素、M-T 边界、四元数偏差加固。

---

## 10. 实施顺序（进入 writing-plans 后）

1. 建 `ReconGridDC/` 骨架 + `GridConventions.cs`（冻结 Fig 2.4 编号 + gridDims + (axis,slot)→(voxel,localEdge) stencil）+ `FixedPointAtomic.cs` + `DeviationLedger.cs`。
2. **Stage 1**：Preprocess → Physics(结构 gather + 弯曲定点原子 + RK45 7 阶段三连) → DC 重建(RebuildIsectWorld+QEF+缝合+法线 barrier+append) → Demo 立方体。审核闭环。
3. **Stage 2**：CONN4096(vertToComp) → 切检测(传 4 体素) → 分量/切点/部分四边形缝合 → 动态坐标系 → 切断 → 消融(particleState 两遍)。审核闭环。
4. **Stage 3**：ToolSDF+广相 → finger-proxy(组协作+上限+回退) → 力/力矩(加固) → extForce 接线 → 设备桩。审核闭环。

> 每 Stage：实现 → grep 自检无悬挂引用 → 多 subagent 审核 → 修复至全过 → 更新 `task_plan.md`/`progress.md` → 下一 Stage。
