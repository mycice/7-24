# Recon Grid DC 代码-设计文档一致性审查

> 审查日期：2026-07-14  
> 审查范围：`recon-grid-dc` 当前工作区（包含未提交的 Unity 侧修改）  
> 设计基线：
> - `2026-07-05-xpbd-combination-round3-design.md`：XPBD 主体、应变、切割和迁移顺序
> - `2026-07-07-collision-contact-redesign.md`：覆盖 7 月 5 日文档中的 step-8 接触/抓取设计
> - `2026-07-07-collision-contact-redesign-fix.md`：已写出但当前尚未执行的修复计划

## 1. 执行摘要

当前代码不是“整体未实现设计”，而是两个完成度差异很大的子系统叠加：

1. **XPBD 主体实现质量较高。** 共享位置缓冲重绑定、固定子步、结构/弯曲约束、阻尼、Stable Neo-Hookean 四面体、切割 liveness、抓取区域 tear exemption 等，基本符合 7 月 5 日 rev-B 设计。
2. **碰撞/接触/抓取仍停留在已被 7 月 7 日文档废弃的 step-8 架构。** 当前 native 代码同时运行 Finger-Proxy 力、直接位置 dent、硬 tissue-side contact；core 仍是 kinematic weld；slab stick 仍是刚性位置 snap；K=2 仍由永久 `g_toolArmed` 触发。
3. **7 月 7 日的修复计划尚未落地。** 仓库中只存在计划文档，没有 `k_SolveCCDContact`、`omegaGrasp`、`coreAlpha`、`coreBreakForce`、`graspMaxDisp` 等实现或对应测试。
4. **Unity 侧新增了初始化外移和组件拆分，但没有满足“整个 shaft∪jaw assembly 的 SDF clearance”要求，且场景序列化值覆盖了代码默认值。** 当前 `0.01 m * 1 unit/m = 0.01 world unit` 的安全距离并不足以证明器械整体在组织外。

综合判断：**XPBD/切割主体可以继续作为可信基础；collision/contact/grasp 不能视为符合最新设计，当前仍具备文档中描述的爆炸、粘连、撕碎和帧率崩溃条件。**

## 2. 已正确实现的设计

### 2.1 完整 XPBD 已替换 RK45 主循环

- `physics_init` 调用 `xpbd_bind(recon_corner_pos(), d_vel, ...)`，XPBD 在共享 `cornerPos` 和 `physics_vel()` 上原地运行。
- `physics_step` 使用 `xpbd_begin_frame` + 固定 N 次 `xpbd_substep`，不再执行 RK45 adaptive-h 积分。
- 每个 XPBD 子步之后调用 `cut_ribbon_tick()`，符合 7 月 5 日 §6.5/§9-(4)/(6) 的 post-commit 切割时序。

定位：
- `cuda_plugin/cuda/physics.cu`：`physics_init`、`physics_step`
- `cuda_plugin/cuda/xpbd/xpbd_solver.cu`：`xpbd_bind`、`xpbd_begin_frame`、`xpbd_substep`

### 2.2 结构、弯曲与阻尼公式基本对齐

- `k_solve_edges` 使用 `alphaTilde=(1/ks)/h^2`，并同时对位置和 lambda 应用 `omega`。
- 结构 Kelvin-Voigt 沿边阻尼使用 `gamma=alphaTilde*betaS*h`。
- `k_perp_damp_edges` 位于 raw velocity write 与 Rayleigh `velScale` 之间，只处理垂直于边的相对速度。
- `k_solve_bending` 保留设计指定的 softened Eq.14 gradient，并使用 `betaB=cb` 的约束内阻尼。

定位：
- `cuda_plugin/cuda/xpbd/xpbd_solver.cu`：`k_solve_edges`、`k_solve_bending`、`k_perp_damp_edges`

### 2.3 Stable Neo-Hookean 与 parity tet 实现完整

- `xpbd_build_tets` 使用 6-tet Kuhn split，并按 `(x+y+z)&1` 交替 diagonal。
- `k_solve_tet` 实现 rest-zero `tr(F^T F)-3` 和 `det(F)-1` 两个约束。
- `k_RefreshLiveness` 每子步根据 axis edge 和 corner active 状态刷新 `d_tetActive`，切口不会被 tet 重新粘住。
- 测试对 parity 使用结构化 tet-id 检查，而不是依赖不敏感的动态 isotropy 指标。

定位：
- `cuda_plugin/cuda/xpbd/xpbd_solver.cu`：`xpbd_build_tets`、`k_solve_tet`、`k_RefreshLiveness`
- `cuda_plugin/tests/xpbd_livepath_test.cu`：G1/G2/G3 gates

### 2.4 切割和抓取区防撕裂机制已实现

- `cut_ribbon_tick` 保持在每个 XPBD 子步提交后运行。
- `cut_recompute_grasp_ring` 实现 BFS ring dilation。
- `k_TearOverstretch` 实现 per-edge release decay：边长降到 `0.8*tau*L` 后重新武装，另有超时保护。
- `k_CarveRimBridge` 使用初始化 topology snapshot 识别 cut rim，不让 grasp exemption 保护切割残丝。

定位：
- `cuda_plugin/cuda/cut.cu`：`k_TearOverstretch`、`cut_recompute_grasp_ring`、`k_CarveRimBridge`

## 3. 严重不符（违背核心设计）

### S1. 直接位置 dent 仍在运行，违反“唯一 tissue-push force path”

**代码：**
- `cuda_plugin/cuda/haptics.cu`：`k_ScatterContact`
- `cuda_plugin/cuda/haptics.cu`：`k_ApplyContactDisp`
- `cuda_plugin/cuda/haptics.cu`：`haptics_step` 中两个 kernel 的 dispatch

`k_ScatterContact` 不仅向 `hapticForce` 写反力，还累计 `contactDisp`；随后 `k_ApplyContactDisp` 直接执行：

```cpp
cornerPos[c] = cornerPos[c] + d;
vel[c] = vel[c] * 0.15f;
```

**违反：**7 月 7 日“Chosen architecture / Explicit edits”要求完整删除 `contactDisp`、`k_ApplyContactDisp` 和 `vel*=0.15`，普通接触只保留 Eq.27 force path。

**后果：**位置写与 XPBD 弹性/抓取解算不共享 multiplier，也不服从统一松弛；它会再次制造文档所述的多 authority 竞争、粘连和非物理速度杀死。

**结论：**这是当前最高优先级违约，修复计划 Task 1 尚未执行。

### S2. `k_SolveJawShaftContact` 仍是全深度硬推出，而非窄 CCD gate

**代码：**`cuda_plugin/cuda/xpbd/xpbd_solver.cu`：`k_SolveJawShaftContact`

kernel 对每个嵌入 shaft/jaw 的 corner 使用总 penetration `C` 求解，并直接应用完整 XPBD correction。它没有：

- approach velocity threshold；
- `0.1L` 级单子步位移上限；
- 独立 `omega_ccd<=0.25`；
- 与普通慢速接触 force path 的职责隔离。

**违反：**7 月 7 日“RETIRE k_SolveJawShaftContact”和 CCD rules a-d。

**后果：**初始嵌入或深穿透仍可能一次把 corner 推出较大距离，直接触发 `tau*L` tear 和 reconstruction 负载正反馈。

**结论：**修复计划 Task 2 的 `k_SolveCCDContact` 完全不存在。

### S3. 普通接触仍有三个并行组织推动通道

当前同时存在：

1. `haptics_step` 的 Eq.27 `hapticForce`；
2. `k_ApplyContactDisp` 的直接位置 dent；
3. `k_SolveJawShaftContact` 的 tissue-side 非穿透约束。

**违反：**7 月 7 日“one force path, four clean concerns”。

**后果：**三条路径在不同时间点、不同位置缓冲语义、不同松弛规则下竞争。这正是 redesign 文档定义的 “core disease”，当前代码没有消除它。

### S4. core grasp 仍是无限权威的 kinematic weld

**代码：**
- `cuda_plugin/cuda/xpbd/xpbd_solver.cu`：`k_ApplyCorePin`
- `cuda_plugin/cuda/xpbd/xpbd_solver.cu`：`k_RefreshLiveness` 的 `graspCore -> invMass=0`

每个子步直接写：

```cpp
pos[c] = target;
prevPos[c] = target;
vel[c] = 0;
```

并且 core 的 `invMass=0`。代码没有 core compliance、core lambda 或 break threshold。

**违反：**7 月 7 日 Grasp layer A：“Compliant, breakable core-pin (not a weld)”。

**后果：**被抓 corner 无法因过载自行脱离，器械仍可无限权威地拖动整个组织，保留“cannot be shaken off”的根因。

### S5. slab STICK 仍是刚性 snap，抓取 correction 没有位移上限

**代码：**`cuda_plugin/cuda/xpbd/xpbd_solver.cu`：`k_SolveSlabCoulomb`

STICK 分支执行：

```cpp
pos[c] = pos[c] - tangential;
```

normal correction 和 tangential correction 都没有 `graspMaxDisp`。调用处还传入全局 `g_p.omega`，当前为 1.0。

**违反：**7 月 7 日 Stability C/D 与 Grasp layer B；修复计划 Task 4 要求 `omegaGrasp` 和 `0.1-0.25L` correction cap。

**后果：**slab 可以在单次 projection 中刚性搬移 corner，仍可能造成局部拉伸尖峰、抖动与撕裂。

### S6. K=2 coupling 仍由永久 `g_toolArmed` 驱动

**代码：**
- `xpbd_set_tool`：每次调用后 `g_toolArmed=true`
- `xpbd_substep`：`int K = (g_bound && g_toolArmed) ? 2 : 1`

`g_toolArmed` 一旦置位，在 session 中不再清除。Unity 每帧调用 `LCS_SetGraspTool`，因此即便器械在体外且没有抓取，K=2 也永久开启。

**违反：**7 月 7 日 Grasp layer C；用户确认的决策是 K=2 **grasp-only**。

**后果：**内部约束 dispatch 数量永久翻倍，保留文档中的 standing framerate cost。

### S7. Unity 工具状态上传发生在 `LCS_Step` 之后

**代码：**
- `Assets/ReconGridDC/Cuda/LiverCudaManager.cs`：`Update`
- `Assets/ReconGridDC/Cuda/LiverCudaHaptics.cs`：`Step` / `PushGraspTool`
- `cuda_plugin/src/plugin_api.cpp`：`LCS_SetGraspTool` 注释要求 BEFORE `LCS_Step`

实际顺序是：

```text
LCS_Step -> cutter -> LCS_Finalize -> mesh rebuild -> LiverCudaHaptics.Step -> LCS_SetGraspTool
```

而 C API 合约明确要求 `LCS_SetGraspTool` 在 `LCS_Step` 之前调用。

**违反：**7 月 5 日 §9-(5) pose-upload 数据流要求，以及 7 月 7 日未来 CCD 的同帧工具状态要求。

**后果：**XPBD contact/grasp 使用上一帧工具姿态；显示 proxy 和 native CCD/contact 会错开整帧。快速运动时会表现为先视觉穿透、下一帧再被修正。

### S8. 初始化只移动 hinge，没有验证 shaft∪closed-jaw assembly clearance

**代码：**`Assets/ReconGridDC/Cuda/LiverCudaHaptics.cs`：`PlaceOutsideTissue`

当前逻辑把 hinge 放到 AABB 的 `min.z-safeDistance`，但没有调用 `TryGetWorldBounds`，也没有计算 shaft 与 closed jaw 的 union SDF/clearance，更没有循环外移直到整个 assembly 清出组织。

场景还将参数覆盖为：

```text
initialSafeDistanceMeters = 0.01
worldUnitsPerMeter = 1
```

即实际 safe distance 仅 `0.01` world unit。

**违反：**7 月 7 日 Initial-interference A：“Clear the whole tool assembly at the closed-jaw pose”。

**后果：**jaw tip、shaft 或 mesh bounds 仍可能与组织相交；只凭 hinge 在 AABB 外不能证明整体无初始干涉。

## 4. 逻辑瑕疵（未完全解决文档中的问题）

### L1. 场景参数覆盖了 redesign 的真实尺度起始值

`SampleScene.unity` 当前序列化值为：

- `targetLongAxisVoxels=48`，未执行 48→96；
- `massScale=1`，没有 96 网格对应的质量缩放；
- `gravity=-4`、`ks=10000`，不是文档验证锚点；
- `hapticDp=5`、`hapticReactionMax=50`，远低于代码默认的 10/600，也低于 force-only redesign 的调参方向。

**违反/未完成：**7 月 5 日 §8/§11 的 96-grid 决策与质量重标定；7 月 7 日 Stability A/B 的真实尺度 damping/force-cap 复核。

**影响：**当前 Unity 视觉结果不能直接作为设计文档 acceptance bar 的验证结果；force-only 改造后还可能出现“faint push/pass-through”。

### L2. `contactAlpha` 没有按 h 转成 alpha-tilde

`k_SolveJawShaftContact` 和 `k_SolveSlabCoulomb` 直接把 `g_gp.contactAlpha` 放进 XPBD denominator，没有在 `xpbd_begin_frame` 中按 `1/h^2` 统一转换。

**违反：**7 月 5 日 §4 Contact 和 §12 R-compliance 对所有 compliance h-power 的集中重算要求。

**影响：**接触/抓取有效刚度随 N 和 `physicsDt` 改变，破坏 XPBD 最重要的 timestep-independent material 语义。修复时应明确参数是物理 compliance 还是已经缩放的 alpha-tilde，并统一命名。

### L3. 抓取 capture 仍在 rising edge 做多次 malloc 和阻塞回读

**代码：**`xpbd_set_tool`

每次 capture 会：

- 4 次 `cudaMalloc`；
- 4 次 D2H `cudaMemcpy`；
- host 端遍历全部 corner；
- 多次 H2D 上传；
- 随后 free 临时缓冲。

当前 hysteresis 能降低重复触发概率，但没有预分配 scratch，也没有完全 GPU 化 core selection。

**对应：**7 月 7 日 Grasp layer C 已把 malloc/sync storm 明确列为帧率风险。

**影响：**真实 device input 在 threshold 附近抖动或频繁抓放时仍会产生主线程 stall。

### L4. haptic force 可能被路由到已抓取/core corner

`k_ScatterContact` 只排除 `active==0` 和 `pinned!=0`，不知道 `d_grasped`/`d_graspCore`。

**违反：**7 月 5 日 ownership matrix 的“每轴每 corner 只有一个 owner”原则；修复计划也把 simultaneous grasp+touch 列为已知风险。

**影响：**未来 core 改成 compliant-breakable 后，额外 hapticForce 可能虚增 core lambda，造成非预期 break。当前 kinematic core 则会吞掉该力，造成力路径与可见形变不一致。

### L5. jaw analytic box 与真实 collision mesh 不一致

Finger-Proxy 使用真实 jaw collision triangle soup；XPBD grasp/contact 使用 `JawSlabWorld` 的统一 gap oriented box。`jawGap` 只由 tip separation 推导，却沿整个 jaw length 使用。

**对应：**7 月 5 日 step-8 已将其标记为 deviation；7 月 7 日验证计划要求特别检查 hinge-to-tip 的几何错位。

**影响：**视觉 mesh 已接触的位置与 native slab 判定区域可能不一致，尤其是 tapered/curved jaw 靠近 hinge 的部分。

### L6. 测试仍验证旧架构，缺少 redesign gates

当前 `xpbd_livepath_test` 只有旧 step-8 Gate H/I；不存在：

- CCD1 static-embedded no-correction；
- CCD2 fast approach capped correction；
- CORE-BREAK；
- KGATE grasp-only；
- compliant slab settle/cap。

`haptics_oracle.py` 仍显式要求 `k_ApplyContactDisp` 存在，说明测试本身也尚未迁移到 7 月 7 日基线。

**影响：**即使当前 suite 全绿，也只能证明旧 step-8 行为没有回归，不能证明最新设计已实现。

### L7. `hapticDp` 没有按 redesign 的 closed-loop damping 重新计算

代码默认 `hapticDp=10`，场景值为 5；设计要求按 `Dp >= 2*sqrt(Kp_eff*m)` 起算并在真实 `L≈0.4` 下用静态深按验证收敛。

**影响：**删除直接 dent 后，force-only feedback loop 可能出现持续 buzzing；当前参数没有日志或测试证明满足过阻尼要求。

### L8. 96-grid 与质量模型决策仍悬空

当前 `targetLongAxisVoxels=48` 且 uniform mass 乘 `massScale=1`。如果直接升到 96，corner 数和总质量都会近似八倍。

**对应：**7 月 5 日 §10/§11 明确要求 volume-proportional mass 或重新调参/重做 golden。

**影响：**不能只改场景分辨率；必须同步处理质量、`N_cut`、tear ring 的物理宽度和性能预算。

## 5. 优化建议（可读性与性能改进）

### O1. 将 `k_ScatterContact` 的全量最近 corner 搜索替换为空间索引

当前每个 `(triangle, vertex lane)` 线程都执行：

```cpp
for (int cptr = 0; cptr < cornerCount; cptr++)
```

复杂度约为 `O(3 * contactTriangleCount * cornerCount)`。48→96 后 corner 数约八倍，最坏耗时同步放大。

建议利用背景规则网格直接计算邻近 cell/corner，或建立局部 hash/grid bucket；接触 patch 只需搜索有限邻域，不应扫描全体 corner。

### O2. 移除已退休的 RK45 scratch 和状态

`physics_init` 仍分配并清零：

- `d_forceInt`；
- `d_kSlope`；
- `d_yTrialPos`；
- `d_yTrialVel`；
- `d_errMax`；
- dead `g_h`。

这些缓冲在 XPBD live path 不再使用。删除后可降低初始化成本和显存占用，也能减少维护者误以为 RK45 仍参与运行。

### O3. 避免每帧 CPU 重建和同步上传完整 jaw triangle soup

`GrasperRig.CollisionSoupWorld` 每帧逐顶点做 CPU transform；`haptics_set_tool_mesh` 再在 CPU 计算 centroid/radius，并 H2D 上传整份 soup。

建议保留 tool-local collision mesh 在 GPU，只上传 rigid transform/jaw angle；在 kernel 中变换查询点或顶点。这样可以显著减少 C# 分配/循环、P/Invoke 数据复制和 PCIe 传输。

### O4. 减少 `haptics_step` 的同步 D2H 读回

当前一帧内多次读取 winner key、triangle id、`float4`、depth sum。每次同步 memcpy 都会打断 GPU pipeline。

建议将 reduction 和后续 scatter 合并，或使用一个小型 device result struct 最后一次性异步回读；不需要 host 决策的值应完全留在 device。

### O5. 预分配 capture scratch，避免 rising-edge allocator 抖动

将 `dCand/dU/dV/dNn` 在 `xpbd_bind` 时按 cornerCount 分配并复用。若 host 仍需 ellipse selection，也应使用 pinned staging buffer；更优方案是在 GPU 上做 reduction + core mark。

### O6. 将工具 pose upload 与 haptic solve 拆成明确的 pre-step/post-step 阶段

推荐 Unity 帧顺序：

```text
Read device/input pose
Upload physical tool + grasp state
LCS_Step (XPBD consumes current pose)
cut/finalize/reconstruct
Finger-Proxy solve against current surface
render/feedback
```

如果 Finger-Proxy 必须消费重建后的 triangle，可保留 post-step proxy solve，但应把“物理工具 pose 上传”从 `LiverCudaHaptics.Step` 拆出到 `PreparePhysicsStep`，避免 native grasp/CCD 一帧滞后。

### O7. 对设计版本建立可执行 feature-state 标记

当前代码注释大量写着 “STEP 8 design rev-B”，但该设计已被 7 月 7 日覆盖。建议增加明确状态：

- `LegacyHardContactEnabled` 临时编译开关，默认关闭；
- redesign ABI version；
- 测试名称包含 `REDESIGN`；
- 文档顶部维护“implemented commit / pending tasks”。

这样可避免“代码注释说已完成，但权威设计已变更”的认知错误。

## 6. 历史问题复盘

| 文档中的历史问题 | 当前状态 | 审查结论 |
|---|---|---|
| RK45 在高刚度/大应变下子步饥饿 | 已解决 | 固定 N XPBD 已上线 |
| XPBD private buffer 与 recon/cut 分叉 | 已解决 | `xpbd_bind` 使用共享缓冲 |
| 切断后 edge/bend/tet 仍施力 | 已解决 | `k_RefreshLiveness` 每子步刷新 |
| tet 跨切口粘连 | 已解决 | tet axis-edge liveness 已实现 |
| grasp tent 在释放瞬间被 tear | 已解决 | release decay 已实现 |
| grasp ring 保护切口残丝 | 已解决 | rim-bridge carve-out 已实现 |
| 初始工具嵌入触发硬推出 | 未解决 | 硬 contact 仍存在，clearance 不完整 |
| position dent 与 XPBD contact 双写 | 未解决 | 两者仍同时运行 |
| core 无法甩脱 | 未解决 | 仍是 kinematic weld |
| K=2 从 frame 1 永久运行 | 未解决 | `g_toolArmed` 永久 latch |
| 抓取阈值 malloc/sync storm | 部分解决 | 有 hysteresis，但 allocator/回读仍在 |
| 高速 tunnelling | 未按新设计解决 | 旧硬 contact 不是 velocity-gated CCD |
| 真实尺度 force-only damping/cap | 未验证 | 场景参数反而更低 |

## 7. 建议修复顺序

1. **执行 redesign Task 1：**删除 `contactDisp` 和 `k_ApplyContactDisp`，保留 Eq.27 force scatter。
2. **执行 redesign Task 2/5：**用 velocity-gated、capped、独立 omega 的 `k_SolveCCDContact` 替换硬 contact，并把 CCD 移出 K coupling loop。
3. **执行 redesign Task 3/4：**core 改为 compliant-breakable；slab normal/STICK 使用 `omegaGrasp` 和 `graspMaxDisp`。
4. **修正 coupling gate：**K=2 只在真实 `g_grasping` 时运行。
5. **重排 Unity 帧时序：**在 `LCS_Step` 前上传本帧 physical tool/grasp pose。
6. **实现真实 assembly auto-clearance：**至少基于 closed-pose shaft+jaw world bounds 迭代外移；理想实现为 union SDF clearance。
7. **增加 CCD1/CCD2/CORE-BREAK/KGATE 和 Unity 手工 checklist，再调 `Dp/reactionMax`。**
8. **最后决定 48→96，并同步质量、`N_cut`、`N_exempt` 与性能优化。**

## 8. 验证边界

本报告完成了静态代码、调用链、场景序列化值和现有测试内容审查。未在本次审查中启动 Unity Play Mode，也未获得现成 native test executable，因此以下结果仍需运行时验证：

- 初始帧 `|F|/tear_count/grasp_count`；
- 静态深按是否收敛或 buzzing；
- 快速 plunge；
- moderate/hard pull 的 break feel；
- 真实 jaw mesh 与 analytic slab 的视觉一致性；
- 48/96 网格下的实际 GPU 帧时间。

在完成 7 月 7 日 redesign 之前，不建议把当前 Unity 行为作为最终验收依据。
