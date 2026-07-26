# Liver Native-CUDA Multi-Thread Framework — Design Spec

**Date:** 2026-06-30
**Status:** Design draft → multi-subagent audit next
**Goal:** Replace the Unity ComputeShader compute layer with a **native C++/CUDA plugin** that runs the soft-tissue physics, geometry reconstruction, and (optional) force feedback on **independent CPU threads + CUDA streams**, with Unity only rendering — a **literal realization of the paper's §3.1 (initialization) + §3.2 (real-time, three-thread) computing framework** (compute in C++/CUDA, rendering in Unity3D/C#).
**Why:** Unity's `ComputeShader.Dispatch` is main-thread-only, so the paper's true multi-threaded GPU concurrency is impossible in pure Unity C#. Native CUDA can launch kernels from multiple host threads on separate streams — the only path that matches the paper exactly. User-chosen.
**Build target (confirmed):** Windows 11, RTX 3090, **CUDA Toolkit 12.x + Visual Studio C++**; the user builds the `.dll` locally (the assistant cannot compile/test CUDA here → staged "write → you build+test → iterate" loop).

## Hard constraints + working mode
- The assistant writes all C++/CUDA + the Unity-side C# + the build scripts, but **cannot compile or run CUDA/the plugin here.** Each stage ends with the **user building the DLL and testing in Unity**, then reporting. The staging (below) front-loads toolchain validation so we never write thousands of un-tested lines blind.
- This **replaces** `Physics.compute` / `Recon.compute` / `Cutting.compute`. The current C# ComputeShader liver demo is kept as a **reference/fallback** (not deleted).
- Algorithm math is **ported, not re-derived** — the HLSL→CUDA port is mostly syntactic (the QEF Eq 1-5, RK45 Eq 17-21, M-T cut, Eq 11/13 forces are unchanged), so the paper-alignment already achieved is preserved. **Three correctness fixes are baked into the port:** (1) **deformed-frame cut detection** (M-T against the deformed `_CornerPos` edges, not the rest lattice — fixes "tool didn't touch but tissue was cut" + matches the paper's tool↔deformed-tissue interaction), (2) **settling** (damping/mass so the liver holds shape), (3) **higher resolution** for surface realism (the perf headroom from true threading allows it).

## Architecture (mirrors paper Fig 3.1)

### Initialization stage (one-time, on load)
CPU (host C++): level-set sample → background grid → outer-grid intersection points → label voxel-corner state → upload to CUDA device buffers. GPU (CUDA): calculate feature points, construct triangles → initial surface. (Mirrors the existing `BackgroundGrid.Build` + initial `dc.Build`, ported to host C++ + CUDA.)

### Real-time stage — THREE worker threads (paper §3.2) + Unity render thread
- **Thread 1 — Soft-Tissue Computation** (its own rate): calculate particle position + orientation (RK45), update links + orientation, update force, add tool force. Writes particle state to device buffer `P_phys`.
- **Thread 2 — Geometry Reconstruction** (its own rate): update voxel corner points (from `P_phys`), **cut voxel** (M-T deformed-frame + sever), look up the connectivity LUT, calculate feature points (QEF), construct triangles. Writes the surface mesh to the render-shared buffer `M_render`.
- **Thread 3 — Haptic** (optional, its own rate): update SDF, move proxy tool, collision test, separate along min distance, obtain force-feedback value. (Stubbed until a haptic device is wired; the proxy/SDF still informs `tool force` into Thread 1.)
- **Unity render main thread:** reads `M_render`, draws via `DrawProceduralIndirect`.

**Cross-thread data rule (paper §3.2):** no buffer is written by two threads at once; cross-thread handoff via **double-buffering + a lightweight lock/atomic flag** (Thread 1 writes `P_phys[A]` while Thread 2 reads `P_phys[B]`; swap on completion). GPU-side accumulation uses atomic adds **sparingly** ("excessive atomics on one thread degrade parallel→serial" — keep ≤6 atomic adds/particle, the paper's budget; revisit the bending kernel's scatter).

## Unity ↔ plugin interop
- **Plugin:** `LiverCudaSim.dll` → `Assets/Plugins/x86_64/`. C# side: `LiverCudaManager : MonoBehaviour` with `[DllImport("LiverCudaSim")]`.
- **P/Invoke API (host):** `Init(gridDesc, meshPath, params)`, `SetTool(S, E, D, cut)`, `GetSurfaceCounts(out vtx, out idx)`, `Shutdown()`, plus the render-thread entry (below).
- **Rendering — two-stage approach:**
  - **Copy-based (Stages 0-3, simple + portable):** CUDA writes the mesh to a device buffer; the plugin copies to a pinned host buffer; Unity reads it via P/Invoke each frame and uploads to a `GraphicsBuffer`/`Mesh`. A CPU roundtrip, fine for validation.
  - **CUDA↔D3D11 interop (Stage 4, the perf path):** the plugin obtains Unity's D3D11 device via `IUnityGraphicsD3D11::GetDevice()` (from `IUnityInterfaces` in `UnityPluginLoad`), registers Unity's vertex `GraphicsBuffer` with `cudaGraphicsD3D11RegisterResource`, and the reconstruction kernel writes vertices **directly** into the buffer Unity renders — no CPU roundtrip. Render-thread access via `GL.IssuePluginEvent(renderEventFunc, id)`. (Requires Unity Graphics API = Direct3D11.)
- **Lifecycle:** `UnityPluginLoad`/`UnityPluginUnload` capture the interfaces + device; worker threads start on `Init`, join on `Shutdown`.

## Build
- A `cuda_plugin/` CMake project (or `.vcxproj`) compiling `*.cu` + `*.cpp` → `LiverCudaSim.dll`, linking `cudart`, the Unity `PluginAPI` headers (`IUnityInterface.h`, `IUnityGraphicsD3D11.h`), and D3D11. The assistant provides the CMakeLists + source; the user runs `cmake --build` (their CUDA 12.x + VS) and copies the DLL to `Assets/Plugins/x86_64/`.
- Source tree: `cuda_plugin/{CMakeLists.txt, src/{plugin_api.cpp, interop_d3d11.cpp, sim_threads.cpp}, cuda/{preprocess.cu, physics.cu, dc_recon.cu, cut.cu, common.cuh}}`.

## Staged plan (each stage = a user build+test gate)
- **Stage 0 — Toolchain + interop skeleton (CRITICAL de-risk).** Minimal `LiverCudaSim.dll`: `Init/Update/Shutdown` P/Invoke; one trivial CUDA kernel writing a single hard-coded triangle's vertices; copy-to-host; `LiverCudaManager` reads it + `DrawProcedural`. **Success = a CUDA-computed triangle renders in Unity.** Proves the DLL builds, loads, P/Invoke + CUDA + the render path all work, before any algorithm port.
- **Stage 1 — Preprocess + DC reconstruction (static).** Port the level-set/grid/isect/connectivity-LUT (host C++) + `RebuildIsectWorld`/`DC_FeaturePoints`(QEF Eq 1-5)/`DC_Stitch`/normals (CUDA). Render the **static liver** from CUDA. Verify it matches the C# DC surface.
- **Stage 2 — Mass-spring RK45 physics.** Port force accumulation (structural Eq 11, bending Eq 13, Kelvin-Voigt), the 7 RK stages (Eq 18-19), error + adaptive step (Eq 20-21), `ComputeParticleRot`. Liver deforms under gravity, anchored; tune settling so it **holds shape**.
- **Stage 3 — Cutting (deformed-frame).** Port `DetectCut` (M-T against **deformed** edges) + `SeverLinks` + the connectivity-LUT split. Interactive rod cuts **where it touches** the deformed liver.
- **Stage 4 — Three-thread framework + D3D11 interop.** Split Thread 1/2 onto separate host threads + CUDA streams + double-buffered `P_phys`; switch rendering to zero-copy CUDA↔D3D11 interop. This is the literal §3.2 realization + the perf payoff (physics + reconstruction concurrent, render never blocked).
- **Stage 5 — Haptic thread (optional).** SDF proxy-tool collision + force feedback (Eq 27); stub if no Omni device.

Each stage: assistant writes → **user builds the DLL + tests in Unity** → multi-subagent code review (paper-alignment + correctness) → next stage.

## Data structures (device buffers, mirror the C# `ReconBuffers`)
Particle: pos/vel/mass/pinned/active, nbrIdx[6]/restNbr[6], bendPairs[15], particleRot. Grid: voxelCorner[8], isect (globalRest/normal/cornerA/cornerB/t/insideA), voxelIsectOffset/Count, surfaceEdges, gridEdges, voxelCutMask, voxelOccupied. Surface out: vertex/index buffers (`M_render`), feature points, normals. RK scratch: yTrial, kSlope[7], errMax. (Same layout as today; ported to `cudaMalloc`'d arrays.)

## Reused / preserved
The algorithm math (QEF, RK45, M-T, sever, Eq 11/13/17-21), the connectivity LUT (Conn4096), the preprocessing logic — all ported verbatim. The C# `MshLoader`/`MeshLevelSet`/`GridFit`/`SmoothedSdfGrid` can be reused for preprocessing (run in C#, hand the grid to the plugin via `Init`) OR ported to host C++ — decide at Stage 1 (reusing the C# preprocess + uploading is simpler).

## Testing
- Per stage, the **user builds + runs in Unity** (the only place CUDA executes). Stage gates: 0 = triangle renders; 1 = static liver matches C#; 2 = deforms + holds shape; 3 = cut-where-touched; 4 = concurrent + good FPS + zero-copy; 5 = force feedback.
- Where feasible, a host-side CUDA unit check (a `test_main.cpp` the user runs) for the kernel math (e.g., QEF on a known cube, RK step on a 1-spring oracle) vs the C# results.
- Multi-subagent review each stage: paper §3.1/§3.2 alignment + kernel-math correctness + interop/threading safety.

## Risks
- **R1 (top):** the assistant cannot compile/test CUDA here → every stage depends on the user's build+test loop. Mitigation: Stage 0 validates the whole toolchain with ~50 lines before any port; small stages with clear gates.
- **R2:** CUDA↔D3D11 interop is fiddly (device capture, resource registration, render-thread access, Unity must be D3D11). Mitigation: copy-based path first (Stages 0-3), interop only at Stage 4; document the D3D11 requirement.
- **R3:** threading data races (double-buffer handoff, stream sync). Mitigation: the §3.2 "no concurrent write + double-buffer" rule, explicit `cudaStreamSynchronize`/events at handoffs, introduced only at Stage 4 (single-thread correct first).
- **R4:** the explicit-RK substep cost is unchanged — true threading hides it (Thread 1 at its own rate, render not blocked), which is exactly the point; but if the physics thread can't keep real-time, the sim lags (paper's accepted "thread delay"), tuned via damping/mass.
- **R5:** scope — this is a multi-week, multi-stage effort; the C# demo remains the fallback throughout.

## Out of scope (for now)
- D3D12/Vulkan/Metal interop (D3D11 only).
- A production haptic loop without the Omni device (Stage 5 stubbed).
- Re-deriving any algorithm (all ported from the paper-aligned C#).
