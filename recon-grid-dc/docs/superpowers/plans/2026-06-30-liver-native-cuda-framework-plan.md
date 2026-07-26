# Liver Native-CUDA Concurrent Framework — Detailed Fix Plan (mirrors paper Fig 3.1)

> **For agentic workers:** implement stage-by-stage. Each stage ends with a **user DLL build + Unity test** AND a **2-subagent code review** (paper §3.1/§3.2 alignment + correctness) before the next stage.

**Goal:** Refactor the soft-body **solver** and **cutting** into a native **C++/CUDA** plugin that runs the paper's §3.2 real-time loop **concurrently** (3 threads + CUDA streams); Unity3D/C# only renders. Exact realization of paper Fig 3.1.

**Authoritative basis:** the paper (Fig 3.1 §3.1, §3.2 lines 949-993, mechanics §2.2 Eq 9-13, integration §2.3 Eq 17-21, QEF §2.1 Eq 1-5, params §3.4) + the audited design spec `docs/superpowers/specs/2026-06-30-liver-native-cuda-framework-design.md`. "Everything per the paper."

**Build (confirmed):** Win11, RTX 3090, CUDA 12.x + VS C++ + CMake — user builds `LiverCudaSim.dll` locally; the assistant cannot compile/test CUDA here → staged "write → you build+test → iterate" loop.

## Global constraints
- **Algorithm math is PORTED verbatim** from the paper-aligned C# kernels (`Physics.compute`, `Recon.compute`, `Cutting.compute`) — HLSL→CUDA is syntactic; the QEF (Eq 1-5), RK45 (Eq 18-21), forces (Eq 11-13), M-T cut, SeverLinks, Conn4096 LUT are unchanged.
- **Use the paper's §3.4 parameters VERBATIM** (`ks=7.5e4, kb=2e4, cs=0.92, cb=0.9, voxel L=0.25cm, dt=0.02`). **The threading IS the perf fix:** Thread 1 (physics) runs at its OWN rate (Fig 3.2 — the physics thread is the rate-limiter), so render is never blocked → the single-thread substep cap's render-protecting **slow-motion-throttle ROLE**, and the extra-damping "settling" workaround, are **NO LONGER NEEDED and are dropped** (strictly MORE paper-aligned). (The per-frame substep BUDGET `maxSubstepsPerFrame` itself REMAINS as an explicit-RK safety bound + Thread-1's own-rate control — see the Thread-1 note; only its slow-motion role is gone.) The liver **holds shape** because `ks=7.5e4` is the paper's stiff value — which ALSO resolves the user's "rod didn't touch but liver was cut" (when it holds shape, rest ≈ deformed, so the rest-frame cut appears where the rod touches).
- **The ONLY parameter deviation (user-requested, called out):** **higher resolution** than the paper's `L=0.25cm`, IF the user wants a more realistic surface (a user preference, not the paper). The default is the paper's `L=0.25cm`.
- **`DetectCut` is ported AS-IS — REST-lattice detection (do NOT switch to deformed-frame).** Deformed-frame detection is the **documented ROOT CAUSE** of the residual-hinge bug (a zero-thickness blade plane misses a gravity-sagged edge → `t_ray` leaves [0,1] → missed cut), closed by the shipped rest-pose fix (commit `43ad2ac`, memory `dc-incomplete-cut-detection-race`). The detection ray uses the rest lattice (`O_rest = gridOrigin + coord*L`); the cut POINT is emitted on the deformed edge. A deformation-aware detector, IF wanted later, is the **material-frame** approach (transform the world blade into each particle's material frame via R, THEN do the rest M-T — the deferred Berndt R-frame work), NOT a naive world-vs-deformed-edge revert.
- **Init in C# (reused), real-time loop in CUDA** (decision). KNOWN DEVIATION: the paper states "all computational processes are based on C++ and CUDA" (line 935); keeping the 3 Init boxes (Level set / Background Grid / Intersection Points) in C# is a pragmatic reuse of validated code (one-time, not the real-time loop). State it as a deviation; it is RESOLVED by the committed **Stage 6** (port the preprocess to native C++) → all computation in C++/CUDA per line 935. The C# reuse is the TEMPORARY init path for Stages 1-5.
- **Thread 3 haptic deferred** (decision). Inter-thread frame-rate relationship follows paper Fig 3.2.
- Existing C# ComputeShader demo kept as **reference/fallback**.

---

## Fig 3.1 → CUDA kernel / function mapping (the core requirement)

### Initialization (one-time) — C# host + a CUDA upload
| Fig 3.1 box | Realization |
|---|---|
| **Level set** | C# `MeshLevelSet`/`SmoothedSdfGrid` (reused) |
| **Create Background Grid** | C# `BackgroundGrid.Build` (reused) → corners, occupancy, physics topology |
| **Calculate Intersection Points of the Outermost Grid** | C# `BackgroundGrid` isect build (reused) |
| **Calculate Feature Points** (GPU) | plugin `Init()` uploads grid+topology to device, runs `k_FeaturePoints` once (QEF Eq 1-5) |
| **Construct Triangles** (GPU) | `k_Stitch` once → initial `M_render` |
| **Wait for GPU / Draw Surface** | `Init()` syncs; Unity draws `M_render` |

`Init(gridDesc*, topology*, isect*, params)` P/Invoke marshals the C#-built grid (corner pos/inside/active, voxelCorner, isect{globalRest,normal,cornerA,cornerB,t,insideA}, voxelIsectOffset/Count, surfaceEdges, gridEdges, nbrIdx/restNbr/bendPairs/mass/pinned, Conn4096) into `cudaMalloc`'d device buffers. **Match the C# `ReconBuffers` strides EXACTLY (audit):** `IsectGpu = 40 bytes` {globalRest 12, normal 12, cornerA 4, cornerB 4, t 4, insideA 4} (NOT 32); `GridEdgeGpu2 = 16 B`, `GridEdgeGpu = 32 B`; `bendPairs = 15/corner`; the per-component cut-FP buffers are `8·VoxelCount` slots. A stride mismatch silently corrupts the upload.

### Real-time loop — **Thread 1: Soft-Tissue Computation** (CUDA stream `s_phys`)
| Fig 3.1 box | CUDA kernel(s) — ported from | Paper |
|---|---|---|
| **Calculate Particle Position and Orientation** | `k_RK45_stage`×7 + `k_Integrate` + `k_Error` + `k_AdaptStep` (adaptive DoPri5) + `k_ParticleRot` | Eq 17-21; §2.1.2/[22] |
| **Update Links and Orientation of Particles** | `k_SeverLinks` (NbrIdx=-1 + bend alive=0) | §2.1.2 |
| **Update Force** | `k_ClearForces` + `k_AccStructural` (Eq 11) + `k_AccBending` (Eq 13-16; one-thread-per-pair scatter — **port as-is**) + barrier | §2.2 |
| **Add Tool Force** | `k_AddToolForce` (from Thread 3's proxy, or 0 when haptic stubbed) | §2.3 |
Writes particle state (`pos/vel/rot`) to the **double-buffered** `P_phys`.
> **Per-frame substep BUDGET (`maxSubstepsPerFrame`) retained — this is NOT the dropped slow-motion workaround:** it remains as an explicit-RK safety bound + Thread-1's own-rate control. Thread 1 runs at its own rate; if substeps lag, render is NOT blocked → the overrun now surfaces as the §3.2 "thread delay", **never as rendered slow-motion** (the slow-motion-throttle ROLE is the part dropped in Global Constraints). This is the whole point of concurrency.

### Real-time loop — **Thread 2: Geometry Reconstruction** (CUDA stream `s_recon`)
| Fig 3.1 box | CUDA kernel(s) — ported from | Paper |
|---|---|---|
| **Update Voxel Corner Points** | `k_RebuildIsectWorld` (re-project isect onto deformed edges, reads `P_phys`) | §2.1.1 |
| **Cut Voxel** | `k_DetectCut` (M-T swept-quad vs **REST lattice** `O_rest=gridOrigin+coord*L`; cut POINT on the deformed edge; **port AS-IS**) + cut-mask | §2.1.3 |
| **Look up in the Connectivity Table** | `k_LookupConnectivity` (Conn4096 → per-voxel per-component `vertToComp`) | §2.1.1 |
| **Calculate Feature Points** | external: `k_FeaturePoints` (QEF Eq 1-5). cut: `k_ClearCutFP` → `k_AccumulateCutFP` (per-component slot accumulation, **8·VoxelCount slots**, fixed-point `POS_SCALE` atomics) → `k_ComputeComponentFP` (per-component **CENTROID — NOT a QEF**; a QEF here re-introduces the jitter/holes bug, Cutting.compute) → `k_InterpCutFP` (2D-2b temporal EMA) | Eq 1-5 + §2.1.1 |
| **Construct Triangles** | external skin: `k_Stitch` (per-component `StitchVertId` split) · cut walls: `k_BuildCutTriangles` · normals: `k_ClearNormals`/`k_NormalsScatter`/`k_NormalsNormalize` + `k_NormalizeCutFP` → **tagged `_Tri`** + the FP/normal StructuredBuffers (NOT a flat vertex buffer — see Interop) | §2.1.1 |
Reads `P_phys[read-buffer]`; writes the surface mesh to `M_render`.

### Real-time loop — **Thread 3: Haptic** (DEFERRED — Stage 5, stubbed without device)
Update SDF · Move Proxy Tool · Colliding? · Separate Along Min Distance · Obtain Force Feedback (Eq 27). When stubbed: the proxy SDF still prevents tool penetration + feeds `Add Tool Force`=0; no device output.

### Unity render main thread
Reads the latest `M_render`; `DrawProceduralIndirect`.

---

## §3.1/§3.2 computing-framework data flow — EXACT realization of Fig 3.1 (the paper is the standard)

Paper §3.2 (lines 949-984): "each sub-thread's output to other threads is highlighted with **yellow boxes**." Those yellow boxes are **four cross-thread data edges**; the plugin reproduces them exactly, each as a **single-writer device buffer published to its reader via a `cudaEvent`** (never a bare CPU flag).

### Initialization (§3.1 / Fig 3.1 top — preprocessing stage), one-time, NOT threaded
Paper: "generate the background grid → calculate the intersection points between the grid and the isosurface → label the voxel-corner state → [GPU] compute feature points per voxel → connect the 4 surrounding voxels' feature points per isosurface-crossing edge into two triangles → continuous surface."
- **CPU (host):** `Level set` → `Create Background Grid` → `Calculate the Intersection Points of the Outermost Grid` → label corner state. (C# `MeshLevelSet`/`BackgroundGrid.Build` reused + uploaded — the stated init deviation.)
- **GPU (CUDA):** `Calculate Feature Points` (`k_FeaturePoints`, QEF Eq 1-5) → `Construct Triangles` (`k_Stitch`).
- **CPU:** `Wait for GPU Computation to Complete` = **`cudaDeviceSynchronize()`** (the explicit barrier drawn in Fig 3.1) → `Draw Surface` (Unity). Runs inside `Init()`; the real-time threads start only after it returns.

### Real-time update (§3.2 / Fig 3.1 bottom — 3 threads at their OWN rates, Fig 3.2)
Each thread loops independently (Fig 3.1 loop-backs); the render thread presents the latest surface. Per §3.2 + **Fig 3.2** the threads run at **independent frame rates** ("users can ignore any visual or haptic discomfort caused by thread delays at a certain frame rate") — this is the mechanism by which render is **never blocked** by Thread-1's RK substeps. **Per Fig 3.2 + Discussion (lines 1233-37) the soft-tissue (physics) thread is the RATE-LIMITER** and bounds the usable particle count; the geometry/render and haptic rates EXCEED the physics rate, and the inter-thread rate gap WIDENS with particle count — so Thread 1 runs at its own (lower) rate while render/haptic stay fast.

- **Thread 1 — Soft Tissue Computation** (`T_phys`/`s_phys`), loop: `Calculate Particle Position and Orientation` (RK45 + `k_ParticleRot`) → `Update Links and Orientation of Particles` (`k_SeverLinks` — applies the cut's connectivity change) → `Update Force` (`k_AccStructural`+`k_AccBending`) → `Add Tool Force` (`k_AddToolForce`) → loop.
- **Thread 2 — Geometry Reconstruction** (`T_recon`/`s_recon`), loop: `Update Voxel Corner Points` (`k_RebuildIsectWorld`) → `Cut Voxel` (`k_DetectCut`, rest-frame) → `Look up in the Connectivity Table` (`k_LookupConnectivity`, Conn4096) → `Calculate Feature Points` (`k_FeaturePoints` + the cut-FP chain) → `Construct Triangles` (`k_Stitch`+`k_BuildCutTriangles`+normals) → loop.
- **Thread 3 — Haptic** (`T_haptic`/`s_haptic`; DEFERRED/stubbed), loop: `Update SDF` → `Move Proxy Tool to the Physics Tool Position` → `Colliding?` (N→loop; Y→) → `Separate Along the Minimum Distance` → `Obtain Force Feedback Value` (Eq 27) → loop.

### The FOUR cross-thread edges (Fig 3.1 yellow outputs) — exact §3.2 mapping
| # | Paper §3.2 sentence | Edge (Fig 3.1) | Shared buffer (SINGLE writer) | Sync |
|---|---|---|---|---|
| **E1** | "the forces, positions, and orientations of particles … update the eight corner points of the background voxel grid in the geometry reconstruction thread" | **T1 `Calculate Particle Position+Orientation` → T2 `Update Voxel Corner Points`** | `P_phys` (pos/vel/rot) — writer T1 | T1 records `e_phys` on `s_phys`; T2 `cudaStreamWaitEvent(s_recon,e_phys)` before `k_RebuildIsectWorld`; 3-buffer ring (slow-reader safe) |
| **E2** | "the cutting operation … changes the connectivity between soft tissue particles while updating their forces and orientations" | **T2 `Cut Voxel` → T1 `Update Links and Orientation`** | `VoxelCutMask` + cut points — writer T2 (`k_DetectCut`) | T2 records `e_cut` on `s_recon`; T1 `cudaStreamWaitEvent(s_phys,e_cut)` before `k_SeverLinks`; double-buffered mask |
| **E2b** (CLOSES the cycle — audit) | the cut-FP chain needs the post-sever **rotation `R`** to re-rotate accumulated cut points ("updating their forces and orientations") | **T1 `Calculate Particle … Orientation` (`k_ComputeParticleRot`) → T2 `Calculate Feature Points` (`k_AccumulateCutFP`)** | **`ParticleRot`** (writer **T1**; double-buffered). NOTE `NbrIdx`/`BendPairs` are **T1-INTERNAL** (written by `k_SeverLinks`, read by T1's force kernels, NOT the cut-FP) | T1 records `e_phys2` on `s_phys`; T2's `k_AccumulateCutFP` consumes the **PREVIOUS published `R` (frame-(N-1))**. **NEW framework-induced thread-delay** — the single-thread reference reads same-frame `R_N` **EXACTLY**; the §3.2 split relaxes it to `R_{N-1}`, within §3.2's accepted "thread delay" but **DISTINCT from** §2.1.2's `EmitCutPoint` ΔR. Exact under translation (`R≈I`); ≤1-frame cut-wall rotation lag otherwise. Breaks the cycle WITHOUT deadlock |
| **E3** | "After the surface is reconstructed, the triangular face information is used in the force feedback thread for collision detection" | **T2 `Construct Triangles` → T3 `Update SDF`/collision** | `M_render` (triangles) — writer T2 | T2 records `e_surf`; T3 waits before collision (deferred) |
| **E4** | "a reaction force threshold is set and applied to update the particle positions" | **T3 `Obtain Force Feedback` → T1 `Add Tool Force`** | tool/reaction force — writer T3 | T3 records `e_force`; T1 reads in `k_AddToolForce` (deferred → 0 until Stage 5) |

**E2/E2b are the cut↔physics coupling the paper stresses** ("on the other hand, it changes the connectivity … while updating their forces and orientations"): the cut is **detected** in T2 (`Cut Voxel`) but the **link severing** that re-shapes the physics topology is in T1 (`Update Links`) — exactly Fig 3.1's box split. So the §3.2-faithful split is: `k_DetectCut`+cut-mask in T2 (E2 → T1); `k_SeverLinks`+`k_ComputeParticleRot` in T1, whose post-sever **`R` (`ParticleRot`)** the T2 cut-FP chain needs back (E2b → T2); the post-sever `NbrIdx`/`BendPairs` stay **T1-internal** (they feed T1's force kernels, NOT the cut-FP). The C# loop runs the whole serial chain `Step → DetectCut → SeverLinks → ComputeRotations → cut-FP` in one frame; the framework SPLITS it per Fig 3.1, making T1↔T2 a **per-frame CYCLE** (E1 pos → E2 mask → E2b sever+R). It is resolved WITHOUT deadlock by the **one-frame-staleness** model — every cross edge publishes a snapshot the reader consumes from the PREVIOUS iteration. For E2b this means `k_AccumulateCutFP` re-rotates accumulated cut points by the **previous-frame `R`** — a **NEW framework-induced thread-delay** (the single-thread reference uses same-frame `R` exactly), within §3.2's accepted "thread delay" and **DISTINCT from** §2.1.2's `EmitCutPoint` ΔR. Exact under translation; ≤1-frame cut-wall rotation lag otherwise (the §3.9 demo is translation-dominated → negligible). No edge blocks bidirectionally. **(If strict same-frame `R` is required, add a within-frame barrier — `k_AccumulateCutFP` waits on the CURRENT frame's `e_phys2` — trading some thread decoupling for exactness.)**

### §3.2 data-communication rules (VERBATIM — enforced)
Paper: "the same data should not be written to by two threads simultaneously on the CPU. If linear operations need to be performed on the data, atomic addition operations must be used in the GPU, and excessive atomic operations on the same thread are prohibited, otherwise parallel computation would degrade to serial."
- **Single-writer invariant:** each shared buffer (`P_phys`, `VoxelCutMask`, `M_render`, force) has exactly ONE writer thread; readers receive a published snapshot via the `cudaEvent` + ring. No buffer is concurrently written by two threads.
- **GPU atomic adds** for linear accumulation (force scatter, FP-slot accumulation, normal scatter): fixed-point `POS_SCALE`/`NORMAL_SCALE` `atomicAdd`.
- **No excessive atomics:** each kernel's atomic fan-out stays bounded; `k_AccBending`'s per-pair scatter runs on a single stream so it does not cross-thread-contend (the paper's "≤6" is the FP-frame update, not bending — see audit note below).

---

## Concurrency — per-thread sync mechanisms
- **3 host threads** (`std::thread`) in the plugin: `T_phys`, `T_recon`, `T_haptic`; each owns a CUDA stream.
- **Cross-thread handoff = double-buffer + cudaEvent (audit-mandated, NOT a bare CPU flag):**
  - `P_phys` is **double-buffered** (`A`/`B`). `T_phys` fills the write-buffer on `s_phys`, then **records `cudaEvent e_phys`** on `s_phys`. The swap of the write/read index happens only after `e_phys` completes; `T_recon` calls **`cudaStreamWaitEvent(s_recon, e_phys)`** before `k_RebuildIsectWorld` reads the read-buffer. A bare CPU atomic flag is INSUFFICIENT (kernel launches are async). Memory-order: release on publish, acquire on read.
  - **Slow-reader safety:** if `T_recon` is slower than `T_phys`, `T_phys` must NOT overwrite the buffer `T_recon` is reading — use a 3-buffer ring (writer never touches the in-flight reader's buffer) OR the writer blocks on swap until the reader releases. (3-buffer ring chosen — non-blocking.)
- **Mandatory intra-frame GPU barriers (audit):** within Thread 1, `k_AccStructural`+`k_AccBending` (atomic force accumulation) must ALL complete before `k_RK45_stage`/`k_Integrate` reads `_ForceInt` — enforce with `cudaStreamSynchronize(s_phys)` (single stream already serializes, but make the dependency explicit). Likewise `k_Normals` scatter must finish before normalize.
- **Cross-thread/cross-frame ordering (audit — corrected):** the CROSS-thread handoff buffers are `pos`/`ParticleRot` (T1→T2, E1/E2b) + `VoxelCutMask` (T2→T1, E2); each is single-writer + `cudaEvent`-published, the reader taking the PREVIOUS snapshot (the paper's thread-delay). `NbrIdx`/`BendPairs` (writer T1 `k_SeverLinks`) are **T1-INTERNAL** (read by T1's force kernels only — NOT cross-thread). **Within-thread ordering preserved per-thread:** T1 runs `k_SeverLinks` (severed `NbrIdx=-1` excluded from `F`) → `k_ComputeParticleRot`; T2 runs `k_DetectCut` → … → `k_AccumulateCutFP`. **Cross-thread `R` consumption (the load-bearing distinction):** in the SINGLE-THREAD reference `k_AccumulateCutFP` reads same-frame `R_N` **EXACTLY** and `EmitCutPoint` reads `R_{N-1}` (§2.1.2 ΔR); in the THREADED framework `k_AccumulateCutFP` reads the published `R_{N-1}` (the **E2b thread-delay** — a NEW framework approximation, exact under translation, ≤1-frame rotation lag), `EmitCutPoint` `R_{N-1}` as before. **RK substep micro-ordering (single `s_phys` stream serializes — state it explicitly):** `k_SeedTrial` → for s in 0..6 {`k_BuildTrial`(s>0) → `k_ClearForces` → `k_AccStructural` → `k_AccBending` → `k_ComputeSlope`(s)} → `k_Error` → `k_IntegrateY5`, ≤16 substeps (`maxSubstepsPerFrame`), embedded-error read once **per FRAME** (D6), not per substep. `_YTrialPos`/`_YTrialPosRW` ALIAS one buffer (C# relies on dispatch serialization) → do NOT overlap `k_BuildTrial`(s+1) with stage-s reads.
- **Atomics — port `k_AccBending` AS-IS, do NOT restructure to gather (audit — correctness trap):** it is one-thread-per-pair, scattering to center `i` (−(F_ij+F_ik)) and arms `j,k` via fixed-point atomic add. A particle receives atomic writes both as the center of its 15 pairs AND as an arm of neighbors' pairs (so >6). A naive per-particle gather of only its own 15 *centered* pairs would **DROP the arm-reaction forces** → silent physics error (no pair-incidence inverse index exists). The paper's "≤6 atomic adds/particle" (line 1072) is the **feature-point coordinate-frame** update (6-neighbor frame), NOT the bending scatter — it does not constrain `k_AccBending`. (If atomic contention is later measured as the bottleneck, build a per-particle pair-incidence list as a separate optimization.)
- **Threading is introduced ONLY at Stage 4** — Stages 1-3 run single-thread (one stream, correctness first).

---

## Unity ↔ CUDA interop (audit-mandated specifics)
- **Stages 0-3 — copy-based (simple, portable):** CUDA writes `M_render` to a device buffer → `cudaMemcpy` to a pinned host buffer → P/Invoke returns it → Unity uploads to a `GraphicsBuffer`/`Mesh` → `DrawProceduralIndirect`. A CPU roundtrip, fine for validation.
- **Stage 4 — zero-copy CUDA↔D3D11 (the perf path), with the audit fixes:**
  - **Buffer type:** the render vertex/index buffers MUST be Unity `GraphicsBuffer` with `Target = Vertex|Raw` and `Index|Raw` (a ByteAddressBuffer). **NOT Structured** — D3D11 forbids Vertex/Index+Structured, and registering a *structured* buffer with CUDA silently corrupts it. **The DC kernels do NOT emit per-vertex pos+normal (audit)** — the C# render path uses **tag-indexed indirection**: `_Tri[SV_VertexID]` is a TAGGED int the vertex shader dereferences into `_VoxelExternalFP`/`_ExtNormalF` (external) or `_CutFP`/`_CutFPNormalF` (bit31 cut). So zero-copy needs a NEW **`k_ExpandVertices`** kernel (one thread per `_Tri` index: decode the tag → fetch pos from `_VoxelExternalFP`/`_CutFP` + normal from `_ExtNormalF`/`_CutFPNormalF` → write interleaved `{pos, normal}` into the registered **Raw** vertex buffer at `SV_VertexID·stride`), the stride/layout matching the C# `VertexAttributeDescriptor`. (Stages 0-3's copy path can keep the tag-indirection rendered via Unity StructuredBuffers as today; the expansion is needed only for the Raw zero-copy buffer.)
  - **Register ONCE at init** (`cudaGraphicsD3D11RegisterResource` on the buffer's native ptr from `GraphicsBuffer.GetNativeBufferPtr()`); never per-frame (it syncs the render thread).
  - **Per-frame map/unmap on the RENDER thread only** (via `GL.IssuePluginEvent` → the plugin's render callback): `cudaGraphicsMapResources` → `cudaGraphicsResourceGetMappedPointer(&devPtr,&size,res)` → launch `k_Stitch` writing into `devPtr` → `cudaGraphicsUnmapResources` (the CUDA→D3D11 sync point) → Unity `DrawProceduralIndirect`. The D3D11 immediate context is single-threaded → the interop map/unmap + the interop kernel run on the render thread, NOT `T_recon`'s worker thread (which produces feature points into an intermediate device buffer that the render-thread kernel reads).
  - **Device match:** set CUDA's active device to the GPU backing Unity's D3D11 device (capture via `IUnityGraphicsD3D11::GetDevice()` in `UnityPluginLoad`; pick the matching `cudaD3D11GetDevice`).
  - **Unity Graphics API = Direct3D11** (Player settings; not D3D12).

---

## Build
`cuda_plugin/{CMakeLists.txt, src/{plugin_api.cpp, interop_d3d11.cpp, sim_threads.cpp}, cuda/{preprocess.cuh, physics.cu, dc_recon.cu, cut.cu, common.cuh}}`. CMake (`LANGUAGES CXX CUDA`, `CUDA_ARCHITECTURES 86`) → `LiverCudaSim.dll` linking `cudart` + Unity PluginAPI headers (`IUnityInterface.h`, `IUnityGraphicsD3D11.h`) + `d3d11`. User: `cmake -B build -A x64 && cmake --build build --config Release` → copy DLL to `Assets/Plugins/x86_64/`. C# side: `Assets/ReconGridDC/Cuda/LiverCudaManager.cs` (`[DllImport("LiverCudaSim")]`).

## Staged tasks (each = assistant writes → user builds+tests → 2-subagent review → next)

- **Stage 0 — Toolchain + render skeleton (de-risk).** `LiverCudaSim.dll`: `Init/GetTriangle/Shutdown`; one CUDA kernel writes a hard-coded triangle; copy-to-host; `LiverCudaManager` renders it. **Gate: a CUDA-computed green triangle in the Game view + Console `Init=0`.** Validates DLL build/load/P-Invoke/CUDA/render before any port.
- **Stage 1 — Preprocess marshal + DC reconstruction (static).** `Init()` uploads the C#-built grid; port `k_RebuildIsectWorld`/`k_FeaturePoints`(QEF)/`k_Stitch`/`k_Normals`; copy-based render. **Gate: the static liver renders from CUDA, matches the C# DC surface.**
- **Stage 2 — Mass-spring RK45 physics.** Port forces (Eq 11/13), 7 RK stages (Eq 18-19), error+adaptive step (Eq 20-21), `k_ParticleRot`; the loop in `Update()`. **Gate: liver deforms under gravity, anchored, HOLDS SHAPE at the paper's `ks=7.5e4` (the paper's stiff §3.4 params hold shape — no settling deviation; the threading absorbs the substep cost).**
- **Stage 3 — Cutting (rest-frame port) + cut-surface chain.** Port `k_DetectCut` (REST-lattice M-T, cut point on deformed edge — AS-IS) + `k_SeverLinks`, AND the full cut-surface chain `k_LookupConnectivity → k_ClearCutFP → k_AccumulateCutFP → k_ComputeComponentFP (centroid) → k_InterpCutFP → k_BuildCutTriangles → k_NormalizeCutFP` + the per-component `StitchVertId` split — EXACT C# ordering (these visually OPEN the cut). Interactive rod (keyboard, from C#). **Gate: the rod opens a visible cut; with the liver holding shape (Stage 2) the cut appears where the rod touches; no spurious cut on no-touch.**
- **Stage 4 — 3-thread concurrency + zero-copy interop.** Split `T_phys`/`T_recon` onto threads+streams+3-buffer ring + cudaEvents; switch to Raw-buffer CUDA↔D3D11 zero-copy (render-thread map/unmap). **Gate: concurrent, high FPS, render never blocked.**
- **Stage 5 — Haptic thread (optional).** SDF proxy collision (prevents penetration, feeds tool-force) + force feedback (Eq 27); stub device output.
- **Stage 6 — Port preprocess to native C++ (strict §3.1 alignment).** Move Level set / Create Background Grid / Calculate Intersection Points / label corner state from C# into the plugin (host C++), eliminating the init-in-C# deviation → ALL computation in C++/CUDA (line 935). **Gate: identical grid/surface vs the C# preprocess; the C# preprocess removed from the runtime path.**

## Data structures (device buffers — mirror C# `ReconBuffers`)
Particle: `pos/vel/mass/pinned/active[N]`, `nbrIdx[6N]/restNbr[6N]`, `bendPairs[15N]`, `particleRot[N]`. Grid: `voxelCorner[8V]`, `isect[]`, `voxelIsectOffset/Count[V]`, `surfaceEdges[]/gridEdges[]`, `voxelCutMask[V]/voxelOccupied[V]`, `Conn4096[4096]`. RK scratch: `yTrialPos/Vel`, `kSlope[7N]`, `errMax`. Render: `M_render` vertex+index (Raw at Stage 4). **Double-buffered (cross-thread handoff — the writer must NOT overwrite the snapshot the reader is mid-reading):** `pos`/`particleRot` (T1→T2, E1/E2b) + `voxelCutMask` (T2→T1, E2); `e_phys`/`e_phys2`/`e_cut` gate the ring swaps. (`nbrIdx`/`bendPairs` are T1-internal → no cross-thread buffer.)

## Testing (per stage, user-run in Unity + a host CUDA unit check where feasible)
0: triangle renders. 1: static liver == C# surface (visual + a host `test_qef.cpp` on a cube). 2: deforms+holds shape (+ a 1-spring RK oracle vs C#). 3: cut-where-touched (+ no spurious cut on no-touch). 4: concurrent FPS + zero-copy correctness + no race (stress: fast cut while deforming). 5: force feedback.
**Each stage ends with a 2-subagent code review:** (a) paper §3.1/§3.2 + algorithm-math alignment, (b) CUDA/interop/threading correctness.

## Risks
- **R1:** assistant can't compile/test CUDA → every stage gated on the user's build+test. Mitigation: Stage 0 validates the toolchain with ~50 lines; small stages.
- **R2:** zero-copy interop (Raw buffers, render-thread map/unmap, device match) — deferred to Stage 4; copy-based first.
- **R3:** threading races (double-buffer/cudaEvent/3-buffer ring, GPU barriers) — deferred to Stage 4; single-thread correct first.
- **R4:** explicit-RK substep cost unchanged — true threading hides it (Thread 1 at its own rate); the point of §3.2.
- **R5:** multi-week, multi-stage; the C# demo is the fallback throughout.

## Out of scope
D3D12/Vulkan interop; a real haptic loop without the Omni device (Stage 5 stubbed); re-deriving any algorithm.
