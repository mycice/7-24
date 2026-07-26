# Haptic Rendering (§2.3) — SDF Finger-Proxy Force Feedback Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the paper's §2.3 haptic-rendering algorithm — an SDF + Finger-Proxy force-feedback module that keeps a proxy tool on the soft-tissue surface, computes contact + (implicit) friction force/torque from the proxy-vs-physical deviation (Eq 22-27), and applies the reaction force back onto the nearby triangle particles — exactly as the paper specifies.

**Architecture:** New CUDA module `haptics.cu/.h` computes, per physics frame, the closest distance between the capsule tool SDF and every reconstructed soft-tissue triangle SDF via the paper's alternating-projection iteration (Eq 26 / Algorithm 1), accumulates the minimum-separation vector, advances the proxy tool position `P_t += a·v_ms` until converged, and scatters the Newton reaction force onto the triangle's three owner particles through a new per-particle haptic-force buffer that `k_ComputeSlope` already-shaped to add. C# drives the physical tool (keyboard, and an optional OpenHaptics device path), renders the proxy, and reads back the computed force/torque for the device.

**Tech Stack:** CUDA 12.x native plugin (C++/CUDA), Unity 2021.3 + C# (Built-in RP), existing `dc_recon` (surface triangles) + `physics` (mass-spring RK45, `d_extForce`) modules, existing `CuttingTool` capsule geometry.

## Global Constraints

- **PAPER FIDELITY IS THE IRON LAW for this module.** Every equation and the algorithm must match §2.3 verbatim. Exact values copied from the paper: convergence threshold `ε = 1e-3` (Algorithm 1 line 3, Eq 21/26); proxy step `a = 0.1` (Algorithm 1 line 4, Eq 24: `P_t += a·v_ms`); inner-iteration cap `i < 100` (Algorithm 1 line 10); Eq 27 force `F = K_p·ΔP − D_p·v`, torque `τ = K_o·θ·k − D_o·ω`.
- **FRICTION IS IMPLICIT, NOT A SEPARATE TERM (critical fidelity rule).** The paper has NO Coulomb / tangential μ·|F_n| friction equation. §2.3 states the tool "exerts appropriate pressure and friction forces" (intro) but its ONLY force law is Eq 27. Friction emerges from the Finger-Proxy constraint: the proxy is held on the surface (Algorithm 1), so when the physical tool slides tangentially, `ΔP = P_B − P_A` develops a **tangential component** that `K_p·ΔP` turns into the friction force, while the normal component is the pressure. **DO NOT add any explicit Coulomb friction model** — that would violate paper fidelity. The plan's "friction" is exactly the tangential part of Eq 27.
- **Damping only during contact** (Eq 27 text: "The damping term only takes effect during contact to counteract oscillations"): `D_p·v` and `D_o·ω` are applied only when the tool is in contact, zero otherwise.
- **Proxy follows the physical tool** (§2.3 lines 787-789: "the proxy surgical instrument gradually approaches the Physical surgical instrument in discrete steps"). Faithful reading + review H1/H2 consensus: **each frame, when NOT in contact, the proxy is set to the physical pose** (`proxyPos = physPos`, `proxyQuat = physQuat` → `ΔP = 0`, `F = 0`); **when in contact, the outward push `P_t += a·v_ms` holds it back on the surface** (Fig 2.11(a): the proxy stays on the surface while the physical tool penetrates, creating the deviation). Without this the proxy drifts and `Kp·ΔP` produces a bogus free-space force. This is NOT an invented mechanism — it is the finger-proxy "proxy approaches physical" prose realized minimally.
- **Contact test is SIGNED** (review H2 major): since `SDF(Tool,A)` returns the capsule SURFACE point, `‖D‖ = ‖P_surface − A_tri‖` is the unsigned surface-to-surface gap — it is `< toolRadius` both when the triangle is INSIDE the capsule (real contact) and when the tool merely HOVERS within one radius outside. Contact ⟺ the closest triangle point is INSIDE the capsule volume: `axisDist = ‖A_tri − closestAxisPt‖ < toolRadius`. The kernel returns `axisDist` alongside `D`; `haptics_step` gates contact on `axisDist < toolRadius`, not on `‖D‖`.
- **Proxy-capsule radius is a DEDICATED haptic parameter** `toolRadius` (paper's tool radius `r`), decoupled from the cut-blade gap `CuttingTool.D` (which is a thickness, not a capsule diameter). Default to a visible fraction of the voxel size, exposed for Task 8 tuning.
- **Haptic collision is SKIN ONLY** (§2.3: "triangular faces of the soft tissue MODEL"): cut-wall triangles (bit30 wall flag in the aux channel) are skipped in `k_ClosestPerTriangle` — the tool collides with the external tissue surface, not the interior of an open wound. Documented as a deliberate choice in the deviation ledger.
- **Reaction force to particles is thresholded** (Fig 3.1 text: "a reaction force threshold is set and applied to update the particle positions, ensuring smooth collision forces without penetration"): clamp the per-particle reaction force magnitude to a configurable ceiling before scattering.
- **Threading / data-race rule** (Fig 3.1): the force-feedback path reads the reconstructed triangle info and writes only its own haptic buffers; particle updates that accumulate linearly across triangles MUST use GPU atomic add (the paper: "If linear operations need to be performed on the data, atomic addition operations must be used in the GPU"). No two writers to one buffer.
- Native side C++/CUDA only for compute; C# only drives I/O + render (paper §3.1: "All computational processes ... based on C++ and CUDA ... rendering interface implemented using Unity3D and C#").
- Build: `cmake --build cuda_plugin/build --config Release`; deploy DLL to `Assets/Plugins/x86_64/` (Unity must be closed — it locks the file). ASCII-only source (cp950 console).
- The plugin already exposes: `d_extForce[cornerCount]` added in `physics.cu k_ComputeSlope` (`k.dv = (extForce[p]+Fint)/mass[p] − alpha·yTrialVel`); `recon_get_surface`/`recon_get_counts`/`recon_tri()`; `CuttingTool` with `S`,`E`,`D` capsule endpoints. Reuse these — do NOT duplicate.

## Notation map (paper symbol → plan name)

| Paper | Meaning | Plan |
|---|---|---|
| `P_A`, `q_A` | Physical tool position / orientation | `physPos`, `physQuat` |
| `P_B`, `q_B` (= `P_t`) | Proxy tool position / orientation | `proxyPos`, `proxyQuat` |
| `ΔP` (Eq 22) | `P_B − P_A` | `dP` |
| `Δq` (Eq 23), `θ` (Eq 24), `k` (Eq 25) | Orientation deviation quat / angle / axis | `dQ`, `theta`, `axisK` |
| `A_n`, `P_n` (Eq 26) | Nearest point on triangle / on tool | `A`, `Pi` |
| `D` (line 15) | `‖D‖` = penetration depth `r − axisDist`; with axis projection `P_i − A` is already outward (line 916) | `vms` = `depth·normalize(P_i − A)` |
| `SDF(Triangle, P)` | Closest point on triangle to P | `ClosestOnTriangle` |
| `SDF(Tool, A)` | Closest point on the capsule AXIS to A (capsule SDF = dist_to_axis − r) | `ClosestOnSegment` |
| `v_ms` (Alg 1 line 16) | The `D` of the PENETRATING triangle with the smallest `‖D‖` (verbatim `if ‖D‖ < ‖v_ms‖`) | `vMs` |
| `F`, `τ` (Eq 27) | Linear force / torque to device | `hapticForce`, `hapticTorque` |
| `K_p`,`D_p`,`K_o`,`D_o` | Stiffness/damping coeffs | `kP`,`dP_damp`,`kO`,`dO` |
| tool radius `r` | Capsule radius — DEDICATED haptic param, decoupled from the cut-blade gap `D` | `toolRadius` (default `~0.6·Lrt`) |
| `P_i`, `A_i` | Tool/triangle point at iteration `i` (Algorithm 1 line 15 uses the PRE-update `P_i`) | `Pi`, `A` |
| `axisDist` | `‖A_tri − closestAxisPt‖`; SIGNED contact test `axisDist < r` | `axisDist` |

---

## File Structure

- **Create `cuda_plugin/cuda/haptics.cu`** — the whole §2.3 compute: SDF helpers, alternating-projection kernel (Eq 26), min-separation reduction, proxy advance (Algorithm 1), Eq 27 force/torque, thresholded reaction-force scatter. Owns `d_hapticForce`, `d_triVms`, proxy host state.
- **Create `cuda_plugin/cuda/haptics.h`** — host API (init/set-tool/step/get-force/shutdown) called from `plugin_api.cpp`.
- **Modify `cuda_plugin/cuda/physics.cu`** — add `d_hapticForce[cornerCount]` buffer + getter `physics_haptic_force()`; add it into `k_ComputeSlope`'s force sum (one line). `d_hapticForce` is zeroed by `haptics_step` NOT `physics_step` (Task-1 Step-3 BLOCKER note — cross-frame ordering: `physics_step` zeroing would wipe the reaction before RK45 consumes it).
- **Modify `cuda_plugin/cuda/physics.h`** — declare `physics_haptic_force()`.
- **Modify `cuda_plugin/cuda/dc_recon.cu` / `.h`** — expose device triangle data for the haptic thread: getters `recon_expand_pos()` (device `float3*` soup, 3 world verts/tri) + `recon_expand_aux()` (device `float4*`, `.w` = wall flag to skip cut walls). Live index count comes from the EXISTING `recon_get_counts`. (Read-only; no compute change.)
- **Modify `cuda_plugin/src/plugin_api.cpp`** — export `LCS_InitHaptics`, `LCS_SetPhysicalTool`, `LCS_StepHaptics`, `LCS_GetHapticForce`, `LCS_GetProxyTool`, `LCS_ShutdownHaptics`.
- **Modify `cuda_plugin/CMakeLists.txt`** — add `cuda/haptics.cu` to the sources.
- **Create `Assets/ReconGridDC/Cuda/HapticTool.cs`** — C# driver: physical tool from keyboard (and optional OpenHaptics device), P/Invoke the haptic API each frame, render the proxy capsule, feed force/torque to the device, HUD.
- **Create host CPU oracle `cuda_plugin/tests/haptics_oracle.py`** — a NumPy reference of the alternating-projection + Algorithm 1 + Eq 22-27 math to diff the CUDA output against (deterministic fixtures).

---

### Task 1: Per-particle haptic-force buffer in physics

**Files:**
- Modify: `cuda_plugin/cuda/physics.cu` (buffer decl near `d_extForce` ~line 119; alloc in `physics_init` ~line 478; add into `k_ComputeSlope` ~line 275; getter near `physics_active()` ~line 612; free in `physics_shutdown` ~line 630 — do NOT zero it in `physics_step`; `haptics_step` owns the zeroing, Task 5)
- Modify: `cuda_plugin/cuda/physics.h` (declare getter near `physics_active()` line 60)
- Test: `cuda_plugin/tests/haptics_oracle.py` (Step 1 stub asserting the design; full oracle in Task 6)

**Interfaces:**
- Produces: device buffer `d_hapticForce` (`float3[cornerCount]`, zeroed by `haptics_step` NOT `physics_step` — see the BLOCKER note in Step 3, ADDED into the RK45 slope exactly like `d_extForce`); host accessor `void* physics_haptic_force()` for `haptics.cu` to scatter into (cast to `float3*`).

- [ ] **Step 1: Write the failing test (design assertion)**

Create `cuda_plugin/tests/haptics_oracle.py` with a placeholder that fails until the buffer contract is documented:

```python
# haptics_oracle.py — CPU reference for the CUDA haptic module (paper 2.3).
# Task 1 slice: assert the physics haptic-force contract is wired.
import pathlib, re
SRC = pathlib.Path(__file__).resolve().parents[1] / "cuda" / "physics.cu"
def test_haptic_force_wired():
    s = SRC.read_text(encoding="utf-8", errors="ignore")
    assert "d_hapticForce" in s, "physics.cu must declare d_hapticForce"
    assert "hapticForce[p]" in s, "k_ComputeSlope must add hapticForce[p] into the slope"
if __name__ == "__main__":
    test_haptic_force_wired(); print("PASS")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python cuda_plugin/tests/haptics_oracle.py`
Expected: `AssertionError: physics.cu must declare d_hapticForce`

- [ ] **Step 3: Add the buffer + wire it in**

In `cuda_plugin/cuda/physics.cu`, next to `static float3* d_extForce = nullptr;`:

```cpp
static float3*       d_hapticForce = nullptr;  // [cornerCount] §2.3 reaction force (haptics.cu scatters + zeroes; NOT zeroed here)
```

In `k_ComputeSlope` (signature already takes `const float3* extForce`), add a parameter `const float3* hapticForce` and change the slope line:

```cpp
        k.dv = (extForce[p] + hapticForce[p] + Fint) / mass[p] - alpha * yTrialVel[p];
```

Pass `d_hapticForce` at the `k_ComputeSlope<<<...>>>` launch (add the argument in the same order).

In `physics_init` after `d_extForce` alloc:

```cpp
    PK(cudaMalloc(&d_hapticForce, (size_t)cc * sizeof(float3)));
    PK(cudaMemset(d_hapticForce, 0, (size_t)cc * sizeof(float3)));
```

**DO NOT zero `d_hapticForce` in `physics_step`** (review H2/H3 BLOCKER — cross-frame ordering).
The frame order is `LCS_Step` (physics, reads the buffer) → cut → `LCS_Finalize` → `LCS_StepHaptics`
(scatters the new reaction). If `physics_step` zeroed it at its top, frame N+1's `LCS_Step` would
wipe frame N's reaction BEFORE the RK45 substeps read it. Instead **`haptics_step` owns the
lifecycle**: it `cudaMemset`s `d_hapticForce` to 0 at its OWN start each frame (Task 5), immediately
before `k_ScatterReaction`. A no-contact frame still clears it (memset runs, no scatter → zero
force). The value written at frame N's `haptics_step` survives into frame N+1's `LCS_Step` (consumed
once), then is cleared and rewritten at frame N+1's `haptics_step`.

Getter next to `physics_active()`:

```cpp
float3* physics_haptic_force() { return d_hapticForce; }
```

Free in `physics_shutdown`: `cudaFree(d_hapticForce); d_hapticForce = nullptr;`

In `cuda_plugin/cuda/physics.h` after `int* physics_active();`:

```cpp
void* physics_haptic_force();   // §2.3: float3[cornerCount] reaction-force buffer (haptics.cu scatters)
```

(Return `void*` in the header to avoid a `float3` include dependency, cast in `haptics.cu`; OR return `float3*` if physics.h already includes vector_types — match the existing style of `physics_particle_rot()` which returns `void*`. Use `void*`.)

- [ ] **Step 4: Run test to verify it passes**

Run: `python cuda_plugin/tests/haptics_oracle.py`
Expected: `PASS`

- [ ] **Step 5: Build to verify physics still compiles**

Run: `cmake --build cuda_plugin/build --config Release 2>&1 | grep -E "error|LiverCudaSim.dll"`
Expected: `... -> ...\LiverCudaSim.dll` (no `error`)

- [ ] **Step 6: Commit**

```bash
git add cuda_plugin/cuda/physics.cu cuda_plugin/cuda/physics.h cuda_plugin/tests/haptics_oracle.py
git commit -m "feat(haptics): per-particle reaction-force buffer in physics (2.3 prep)"
```

---

### Task 2: Device triangle-vertex + wall-flag access from dc_recon

**Files:**
- Modify: `cuda_plugin/cuda/dc_recon.cu` (getters near `recon_tri()` ~line 543)
- Modify: `cuda_plugin/cuda/dc_recon.h` (declare near `recon_tri()` line 82)
- Test: `cuda_plugin/tests/haptics_oracle.py::test_recon_tri_access` (grep assertion)

**Interfaces:**
- Consumes: existing `d_expandPos` (`float3[3*triCapacity]`, non-indexed soup = 3 world vertices per triangle, already the reconstructed soft-tissue surface), `d_expandAux` (`float4[3*triCapacity]`, `.w` = wall flag per vertex; a triangle is a CUT WALL iff its verts carry the flag), and `d_triCounter`.
- Produces: `float3* recon_expand_pos()` (device ptr to the expanded soup) and `float4* recon_expand_aux()` (device ptr to the aux/wall-flag soup). **Do NOT add `recon_expand_count()`** — it would duplicate the existing `recon_get_counts(int*, int*)` (review H3 major). `haptics_step` calls `recon_get_counts` for the live index count. The haptic kernel treats `[3t, 3t+1, 3t+2]` as one triangle's world vertices and skips the triangle if any of its aux `.w > 0.5` (cut wall).

- [ ] **Step 1: Write the failing test**

Append to `haptics_oracle.py`:

```python
def test_recon_tri_access():
    s = (pathlib.Path(__file__).resolve().parents[1] / "cuda" / "dc_recon.cu").read_text(errors="ignore")
    assert "recon_expand_pos" in s and "recon_expand_aux" in s
```

- [ ] **Step 2: Run to verify it fails**

Run: `python cuda_plugin/tests/haptics_oracle.py`
Expected: `AssertionError` on `test_recon_tri_access`

- [ ] **Step 3: Add the getters**

In `cuda_plugin/cuda/dc_recon.cu` near `unsigned int* recon_tri_counter()`:

```cpp
float3* recon_expand_pos()   { return d_expandPos; }   // §2.3 haptics: 3 world verts / triangle (soup)
float4* recon_expand_aux()   { return d_expandAux; }   // §2.3 haptics: per-vertex wall flag (.w) to skip cut walls
```

In `dc_recon.h` near `unsigned int* recon_tri_counter();`:

```cpp
float3* recon_expand_pos();    // §2.3 haptics: device non-indexed soup (3 world verts per triangle)
float4* recon_expand_aux();    // §2.3 haptics: per-vertex aux (.w = wall flag; skip cut-wall triangles)
```

(The live index count comes from the EXISTING `recon_get_counts(int*, int*)` — do not duplicate it.)

- [ ] **Step 4: Run to verify it passes**

Run: `python cuda_plugin/tests/haptics_oracle.py`
Expected: `PASS`

- [ ] **Step 5: Commit**

```bash
git add cuda_plugin/cuda/dc_recon.cu cuda_plugin/cuda/dc_recon.h
git commit -m "feat(haptics): expose reconstructed triangle vertices to the haptic thread"
```

---

### Task 3: SDF closest-point primitives (device) + CPU oracle

**Files:**
- Create: `cuda_plugin/cuda/haptics.cu` (SDF helpers + a tiny test-only host wrapper)
- Create: `cuda_plugin/cuda/haptics.h` (empty API stub for now — filled in Task 5)
- Modify: `cuda_plugin/CMakeLists.txt` (add `cuda/haptics.cu`)
- Test: `cuda_plugin/tests/haptics_oracle.py::test_sdf_math` (NumPy reference vs hardcoded expected)

**Interfaces:**
- Produces two `__device__ __forceinline__` functions used by Task 4:
  - `float3 ClosestOnTriangle(float3 a, float3 b, float3 c, float3 p)` — `SDF(Triangle, P)` (Eq 26): nearest point on triangle `abc` to `p` (Ericson region test; identical algorithm to `ClosestPointOnTriangle` already in `SofaUnityVisualLiverRenderer.cs`, ported to CUDA).
  - `float3 ClosestOnSegment(float3 s, float3 e, float3 p)` — `SDF(Tool, A)`: nearest point on the capsule AXIS segment `[s,e]` to `p`. **The capsule's directed distance field (paper §2.3: "volume represented by a directed distance field") is `dist_to_axis − radius`, so the SDF-`Tool` closest query projects onto the AXIS and the radius is applied analytically** in Task 4's depth/contact test. (Projecting onto the capsule SURFACE instead degenerates for penetration — the alternating projection would converge to `axisDist ≈ radius` and lose the true penetration depth; the axis projection is the correct, non-degenerate realization.)

- [ ] **Step 1: Write the failing oracle test**

Append to `haptics_oracle.py` a NumPy reference and fixed expected values:

```python
import numpy as np
def closest_on_triangle(a,b,c,p):
    a,b,c,p = map(np.asarray,(a,b,c,p))
    ab,ac,ap = b-a,c-a,p-a
    d1,d2 = ab@ap, ac@ap
    if d1<=0 and d2<=0: return a
    bp=p-b; d3,d4=ab@bp,ac@bp
    if d3>=0 and d4<=d3: return b
    vc=d1*d4-d3*d2
    if vc<=0 and d1>=0 and d3<=0: return a+ab*(d1/(d1-d3))
    cp=p-c; d5,d6=ab@cp,ac@cp
    if d6>=0 and d5<=d6: return c
    vb=d5*d2-d1*d6
    if vb<=0 and d2>=0 and d6<=0: return a+ac*(d2/(d2-d6))
    va=d3*d6-d5*d4
    if va<=0 and (d4-d3)>=0 and (d5-d6)>=0: return b+(c-b)*((d4-d3)/((d4-d3)+(d5-d6)))
    den=1/(va+vb+vc); v=vb*den; w=vc*den
    return a+ab*v+ac*w
def closest_on_segment(s,e,p):
    s,e,p=map(np.asarray,(s,e,p)); se=e-s
    t=np.clip((p-s)@se/max(se@se,1e-12),0,1); return s+se*t
def test_sdf_math():
    # point above a unit triangle in the z=0 plane
    cp=closest_on_triangle([0,0,0],[1,0,0],[0,1,0],[0.25,0.25,1.0])
    assert np.allclose(cp,[0.25,0.25,0.0],atol=1e-6), cp
    # axis along x, query point off to +y -> nearest axis point is directly below in x
    cs=closest_on_segment([0,0,0],[2,0,0],[1.0,2.0,0.0])
    assert np.allclose(cs,[1.0,0.0,0.0],atol=1e-6), cs
```

- [ ] **Step 2: Run to verify it fails**

Run: `python cuda_plugin/tests/haptics_oracle.py`
Expected: `AssertionError` on `test_sdf_math` (functions not yet asserted against CUDA) — actually the NumPy math passes standalone; to make it a real failing gate, ALSO assert the CUDA source contains the primitives:

```python
def test_sdf_in_cuda():
    s=(pathlib.Path(__file__).resolve().parents[1]/"cuda"/"haptics.cu").read_text(errors="ignore")
    assert "ClosestOnTriangle" in s and "ClosestOnSegment" in s
```

Expected: `FileNotFoundError`/`AssertionError` (haptics.cu absent).

- [ ] **Step 3: Create haptics.cu with the primitives**

Create `cuda_plugin/cuda/haptics.cu`:

```cpp
// haptics.cu — paper 2.3 SDF Finger-Proxy force feedback.
// SDF closest-point primitives: SDF(Triangle,P) and SDF(Tool,A) for the Eq 26 alternating projection.
#include "haptics.h"
#include "physics.h"
#include "dc_recon.h"
#include "common.cuh"   // float3 ops dot3/cross3/length3
#include <math.h>

// SDF(Triangle, P) (Eq 26): nearest point on triangle abc to p (Ericson region test).
__device__ __forceinline__ float3 ClosestOnTriangle(float3 a, float3 b, float3 c, float3 p)
{
    float3 ab = b - a, ac = c - a, ap = p - a;
    float d1 = dot3(ab, ap), d2 = dot3(ac, ap);
    if (d1 <= 0.f && d2 <= 0.f) return a;
    float3 bp = p - b; float d3 = dot3(ab, bp), d4 = dot3(ac, bp);
    if (d3 >= 0.f && d4 <= d3) return b;
    float vc = d1 * d4 - d3 * d2;
    if (vc <= 0.f && d1 >= 0.f && d3 <= 0.f) return a + ab * (d1 / (d1 - d3));
    float3 cp = p - c; float d5 = dot3(ab, cp), d6 = dot3(ac, cp);
    if (d6 >= 0.f && d5 <= d6) return c;
    float vb = d5 * d2 - d1 * d6;
    if (vb <= 0.f && d2 >= 0.f && d6 <= 0.f) return a + ac * (d2 / (d2 - d6));
    float va = d3 * d6 - d5 * d4;
    if (va <= 0.f && (d4 - d3) >= 0.f && (d5 - d6) >= 0.f)
        return b + (c - b) * ((d4 - d3) / ((d4 - d3) + (d5 - d6)));
    float den = 1.f / (va + vb + vc); float v = vb * den, w = vc * den;
    return a + ab * v + ac * w;
}

// SDF(Tool, A): nearest point on the capsule AXIS segment [s,e] to p. The capsule's directed
// distance field is dist_to_axis - radius, so the alternating projection runs on the AXIS and the
// radius is applied analytically (Task 4). Projecting on the surface degenerates under penetration.
__device__ __forceinline__ float3 ClosestOnSegment(float3 s, float3 e, float3 p)
{
    float3 se = e - s;
    float t = fminf(fmaxf(dot3(p - s, se) / fmaxf(dot3(se, se), 1e-12f), 0.f), 1.f);
    return s + se * t;
}
```

Create `cuda_plugin/cuda/haptics.h` (stub; filled in Task 5):

```cpp
// haptics.h — host API for the paper 2.3 SDF Finger-Proxy force feedback (defined in haptics.cu).
#pragma once
// (API declared in Task 5.)
```

Add to `cuda_plugin/CMakeLists.txt` sources list (next to `cuda/cut.cu`):

```cmake
    cuda/haptics.cu
```

- [ ] **Step 4: Run to verify the oracle passes + build compiles**

Run: `python cuda_plugin/tests/haptics_oracle.py`
Expected: `PASS`
Run: `cmake --build cuda_plugin/build --config Release 2>&1 | grep -E "error|LiverCudaSim.dll"`
Expected: DLL built, no `error`

- [ ] **Step 5: Commit**

```bash
git add cuda_plugin/cuda/haptics.cu cuda_plugin/cuda/haptics.h cuda_plugin/CMakeLists.txt cuda_plugin/tests/haptics_oracle.py
git commit -m "feat(haptics): SDF closest-point primitives for Eq 26 (triangle + capsule)"
```

---

### Task 4: Alternating-projection kernel + min-separation reduction (Eq 26 / Algorithm 1 inner loop)

**Files:**
- Modify: `cuda_plugin/cuda/haptics.cu` (add `k_ClosestPerTriangle` kernel + device buffers)
- Test: `cuda_plugin/tests/haptics_oracle.py::test_alternating_projection`

**Interfaces:**
- Consumes: `recon_expand_pos()` / `recon_expand_aux()` (Task 2), `recon_get_counts` for the live index count, `ClosestOnTriangle`/`ClosestOnSegment` (Task 3), the current proxy capsule `[proxyS, proxyE]` + `toolRadius`.
- Produces: `k_ClosestPerTriangle` — one thread per triangle `t`; **skips cut-wall triangles** (any vertex aux `.w > 0.5`); runs the Eq 26 alternating projection on the AXIS (`i < 100`, converge when `|Δ| < ε`) starting `P_0` = the axis point nearest the triangle centroid; from the converged axis point `P_i` and triangle point `A` computes `axisDist = ‖P_i − A‖`, `depth = radius − axisDist`, and the OUTWARD push `vms = depth·normalize(P_i − A)` (triangle-normal fallback). Writes `d_triVms[t] = float4(vms.xyz, key)` with `key = depth` if penetrating (`axisDist < radius`) else `+1e30`. Two reduction kernels `k_ReduceVmsKey`/`k_PickVmsTri` (`atomicMin` on the non-negative `__float_as_uint(key)`) find the min-`depth` penetrating triangle; `haptics_step` copies back the winning `float4` + tri index once per outer iteration.

**Fidelity notes (Eq 26 / Algorithm 1):** the iteration is `A_n = SDF(Triangle, P_n); P_{n+1} = SDF(Tool axis, A_n)`; convergence when `Δ = ‖P_i − A_i‖ − ‖P_{i+1} − A_i‖`, `|Δ| < ε` (Algorithm 1 line 13-14); cap `i < 100`; non-expansive ⇒ monotone. **`P_i` captured BEFORE `P = P_{i+1}`** (line 15). Selection = **min `‖D‖` (= min penetration depth)** among PENETRATING triangles (Algorithm 1 line 16 verbatim; text lines 855-857 penetration gate). Push = the OUTWARD separation `depth·normalize(P_i − A)` (with axis projection `P_i − A` is already outward; magnitude = `‖D‖` = depth), per line 916's "moved outward" — ledger D15. Contact ⟺ `axisDist < radius`.

- [ ] **Step 1: Write the failing oracle test (full alternating projection + Algorithm-1 inner loop)**

Append to `haptics_oracle.py`:

```python
def closest_on_segment(s,e,p):
    s,e,p=map(np.asarray,(s,e,p)); se=e-s
    t=np.clip((p-s)@se/max(se@se,1e-12),0,1); return s+se*t
# Alternating projection on the capsule AXIS (Eq 26). Returns (vms_outward, axisDist, depth).
def alt_project(tri, s, e, r, eps=1e-3, itmax=100):
    a,b,c = map(np.asarray, tri)
    P = closest_on_segment(s,e,(a+b+c)/3.0)     # P0 on the AXIS
    A = a; Pi = P
    for _ in range(itmax):
        A = closest_on_triangle(a,b,c,P)         # A_n = SDF(Triangle, P_n)
        Pn = closest_on_segment(s,e,A)           # P_{n+1} = SDF(Tool axis, A_n)
        dprev = np.linalg.norm(P-A); dnew = np.linalg.norm(Pn-A)
        Pi = P                                   # P_i (axis pt) BEFORE the update (line 15)
        P = Pn
        if abs(dprev-dnew) < eps: break
    axisDist = np.linalg.norm(Pi - A)            # axis-to-triangle distance
    depth = r - axisDist                         # >0 iff penetrating (= ||D|| = penetration depth)
    nout = Pi - A; nl = np.linalg.norm(nout)
    if nl > 1e-6: nout = nout/nl                 # outward: triangle -> axis
    else:
        fn = np.cross(np.asarray(tri[1])-tri[0], np.asarray(tri[2])-tri[0]); nout = fn/np.linalg.norm(fn)
    return nout*depth, axisDist, depth           # vms is the OUTWARD separation

def test_alternating_projection():
    # axis HOVERING above the triangle plane (no contact): axis z=1.0, r=0.5, surface gap = 0.5.
    vms,axisDist,depth = alt_project([[0,0,0],[2,0,0],[0,2,0]],[0,0.5,1.0],[2,0.5,1.0],0.5)
    assert abs(axisDist - 1.0) < 5e-3, axisDist        # NOT in contact (axisDist > r)
    assert abs((axisDist - 0.5) - 0.5) < 5e-3, axisDist  # surface gap = axisDist - r = 0.5

def test_penetration_sign():
    # tool pressing DOWN from ABOVE: capsule axis at z=+0.2 over a triangle at z=0 -> penetrates.
    vms,axisDist,depth = alt_project([[-1,-1,0],[3,-1,0],[-1,3,0]],[0,0,0.2],[2,0,0.2],0.5)
    assert axisDist < 0.5, axisDist                    # IN contact (axisDist < r)
    assert vms[2] > 0, vms                             # OUTWARD = +z (away from tissue below); proxy pushed UP
    assert abs(np.linalg.norm(vms) - 0.3) < 5e-3, vms  # ||v_ms|| = penetration depth r - axisDist = 0.3
```

- [ ] **Step 2: Run to verify it fails**

Run: `python cuda_plugin/tests/haptics_oracle.py`
Expected: `AssertionError` on `test_alternating_projection` after you add a `raise` stub — OR (simpler) also gate on the CUDA source: append `def test_kernel_present(): assert "k_ClosestPerTriangle" in open(... "haptics.cu").read()` → fails (kernel absent).

- [ ] **Step 3: Add the kernel + buffers to haptics.cu**

```cpp
// (ClosestOnSegment is defined in Task 3 — the capsule SDF core.)

// Per-triangle alternating projection (Eq 26, Algorithm 1 inner loop). One thread per triangle.
// Skips cut-wall triangles (aux .w set). d_triVms[t] = float4(D.xyz, key) where — VERBATIM
// Algorithm 1 line 16 ('if ||D|| < ||v_ms||') the selection is the SMALLEST ||D|| among the
// PENETRATING triangles (paper text lines 855-857: outward motion only 'if penetration occurs and
// the closest distance is smaller than the surgical instrument radius'):
//   key = ||D||   if the triangle point is INSIDE the capsule volume (axisDist < radius),
//         +1e30f  otherwise (non-penetrating -> excluded from the min so free-space triangles
//                 never drive the proxy off the surface).
__global__ void k_ClosestPerTriangle(int triCount, const float3* soup, const float4* aux,
                                     float3 s, float3 e, float radius,
                                     float eps, float4* triVms)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= triCount) return;
    // SKIN ONLY: skip cut-wall triangles (any vertex carries the bit30 wall flag).
    if (aux[3*t+0].w > 0.5f || aux[3*t+1].w > 0.5f || aux[3*t+2].w > 0.5f)
    { triVms[t] = make_float4(0.f,0.f,0.f, 1e30f); return; }

    float3 a = soup[3 * t + 0], b = soup[3 * t + 1], c = soup[3 * t + 2];
    float3 P = ClosestOnSegment(s, e, (a + b + c) * (1.f / 3.f));   // P0 on the AXIS
    float3 A = a, Pi = P;
    #pragma unroll 1
    for (int i = 0; i < 100; i++)                    // Algorithm 1 line 10
    {
        A = ClosestOnTriangle(a, b, c, P);           // A_n = SDF(Triangle, P_n)
        float3 Pn = ClosestOnSegment(s, e, A);       // P_{n+1} = SDF(Tool axis, A_n)
        float dPrev = length3(P - A);
        float dNew  = length3(Pn - A);
        Pi = P;                                       // capture P_i (axis pt) BEFORE the update
        P = Pn;
        if (fabsf(dPrev - dNew) < eps) break;        // |Δ| < ε
    }
    // Pi is the closest AXIS point, A the closest triangle point; axisDist = ||Pi - A|| is the
    // axis-to-triangle distance. Capsule penetration depth = radius - axisDist (Algorithm 1 line 15
    // magnitude ||D||). SIGN (paper line 916 "moved OUTWARD"): the literal D = P_i(surface) - A_i
    // points INTO the tissue; the outward push is A_i - P_i orientation = (r - axisDist)*n_out with
    // normalize(Pi - A) is already outward (Pi is the axis point, on the tool side). Ledger D15.
    float axisDist = length3(Pi - A);
    float depth = radius - axisDist;                 // >0 iff penetrating
    float3 nOut = Pi - A; float nlen = length3(nOut);
    if (nlen > 1e-6f) nOut = nOut / nlen;
    else {   // degenerate: axis passes through A -> use the triangle face normal (oriented outward)
        float3 fn = cross3(b - a, c - a); float fl = length3(fn);
        nOut = (fl > 1e-9f) ? fn / fl : make_float3(0.f, 1.f, 0.f);
    }
    float3 vms = nOut * depth;                        // outward separation
    float key = (axisDist < radius) ? depth : 1e30f;  // line 16: min ||D|| = min depth over penetrating
    triVms[t] = make_float4(vms.x, vms.y, vms.z, key);
}
```

Add device buffers + host state near the top of haptics.cu:

```cpp
static float4* d_triVms  = nullptr;  // [triCapacity]
static unsigned int* d_vmsKey = nullptr;  // [1] packed float key for atomicMin reduction
static int*    d_vmsTri  = nullptr;  // [1] argmin triangle index
static int     g_triCap  = 0;
static float3  g_proxyS, g_proxyE;   // proxy capsule endpoints (paper P_B)
static float   g_toolRadius = 0.f;
```

Add the GPU reduction kernel (avoids the per-outer-iteration host argmin sync — review H3 major):

```cpp
// Global min over triVms[].w (Algorithm 1 line 16 smallest ||D||). All keys are NON-NEGATIVE now
// (||D|| for penetrating tris, +1e30 otherwise), so __float_as_uint is directly order-preserving —
// no sign-bit trickery needed. Two-pass: pass 1 finds the min key, pass 2 the winning tri index.
__global__ void k_ReduceVmsKey(int triCount, const float4* triVms, unsigned int* outKey)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= triCount) return;
    atomicMin(outKey, __float_as_uint(triVms[t].w));   // valid: keys >= 0
}
__global__ void k_PickVmsTri(int triCount, const float4* triVms, unsigned int winKey, int* outTri)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= triCount) return;
    if (__float_as_uint(triVms[t].w) == winKey) atomicMin(outTri, t);   // lowest tri index wins ties
}
```

- [ ] **Step 4: Run to verify oracle passes + build**

Run: `python cuda_plugin/tests/haptics_oracle.py`
Expected: `PASS`
Run: `cmake --build cuda_plugin/build --config Release 2>&1 | grep -E "error|LiverCudaSim.dll"`
Expected: DLL built.

- [ ] **Step 5: Commit**

```bash
git add cuda_plugin/cuda/haptics.cu cuda_plugin/tests/haptics_oracle.py
git commit -m "feat(haptics): per-triangle alternating projection Eq 26 / Algorithm 1 inner loop"
```

---

### Task 5: Proxy advance (Algorithm 1 outer loop) + Eq 22-27 force/torque + reaction scatter — host API

**Files:**
- Modify: `cuda_plugin/cuda/haptics.cu` (min-reduction, proxy advance, Eq 22-27, reaction scatter kernel, full host API)
- Modify: `cuda_plugin/cuda/haptics.h` (declare the API)
- Test: `cuda_plugin/tests/haptics_oracle.py::test_algorithm1_outer` + `::test_eq27_forcetorque`

**Interfaces:**
- Produces host API (all `extern "C"`-friendly, called by plugin_api):
  - `int haptics_init(int cornerCount, int triCapacity, const HapticParams* p)`
  - `void haptics_set_physical_tool(const float* physS3, const float* physE3, const float* physQuat4, const float* linVel3, const float* angVel3, float toolRadius)` — the physical tool is a CAPSULE `[S,E]` (matches `CuttingTool.S/E`); native stores `g_physS=S, g_physE=E, g_physPos=(S+E)/2, g_physQuat, g_linVel, g_angVel, g_toolRadius`.
  - `int haptics_step()` — runs Algorithm 1 (per-frame: reset `vMs` large; loop `k_ClosestPerTriangle` → reduce min → `proxyPos += a·vMs`; until `‖vMs‖ < ε` or an outer cap), computes Eq 22-27, scatters the thresholded reaction force onto the min-triangle's 3 particles via atomicAdd into `physics_haptic_force()`. Returns 0 on success.
  - `void haptics_get_force(float* outForce3, float* outTorque3)` — Eq 27 result for the device.
  - `void haptics_get_proxy(float* outPos3, float* outQuat4)` — proxy pose for rendering.
  - `void haptics_shutdown()`
  - **`struct HapticParams { float kP, dP, kO, dO, a, eps, reactionMax; int outerMax; };` MUST be defined in `haptics.h`** (BEFORE the `haptics_init` declaration), NOT in haptics.cu — `plugin_api.cpp` (Task 6) and the API declarations both need it visible via the header. `haptics.cu` includes `haptics.h` and does NOT redefine it (review F3 major).

**Fidelity notes:**
- **Proxy follows the physical tool** (§2.3 lines 787-789; review H1/H2 major). At the START of `haptics_step`, BEFORE the outer loop: seed the proxy capsule + `proxyPos`/`proxyQuat` from the physical tool. Then run Algorithm 1; if NOT in contact after the loop, leave `proxyPos = physPos` (so `ΔP = 0`); if in contact, the outward push has displaced it. Concretely: each frame set `g_proxyS/E` and `g_proxyPos_dev`/`g_proxyQuat` = the physical pose first, then let the outer loop push it out. This makes `ΔP` always the physical-vs-surface deviation (Fig 2.11(a)), never a stale drift.
- **Algorithm 1 outer loop** (lines 6-28): repeat {`k_ClosestPerTriangle` → GPU reduce → `P_t += a·v_ms` (a=0.1)} until `‖v_ms‖ < ε` OR no contact. `outerMax` safety cap on the paper's `while True` (sole deviation, ledgered).
- **Selection = Algorithm 1 line 16 verbatim**: `v_ms` = the `D` of the PENETRATING triangle with the SMALLEST `‖D‖` (`if ‖D‖ < ‖v_ms‖`), penetration-gated per the text (lines 855-857: outward motion only when the closest distance < tool radius). Non-penetrating triangles are excluded (`key = 1e30`) so free-space geometry never pushes the proxy off the surface.
- **Penetration / outward push** (§2.3 lines 915-916): contact ⟺ SIGNED `axisDist < toolRadius`. The push vector `v_ms` = the OUTWARD separation `(r − axisDist)·normalize(axisPt − A)` (magnitude = `‖D‖` = the paper's line-15 quantity; orientation = the paper text's "moved outward"). `P_t += a·v_ms` pushes the proxy out. Contact is re-evaluated EVERY outer iteration (no latch); the loop stops the moment no triangle penetrates. `test_penetration_sign` guards the sign; `test_algorithm1_outer` guards convergence to sub-ε residual penetration.
- **Orientation / torque scope** (Algorithm 1 Output is "Proxy Tool Position `P_t`" — position ONLY, paper line 866): the algorithm updates no proxy ORIENTATION, so `proxyQuat = physQuat` each frame ⇒ `Δq = identity`, `θ = 0`, and the Eq 27 restoring torque `K_o·θ·k` is ZERO — only the damping torque `−D_o·ω` is produced. This is faithful to Algorithm 1 (verbatim) but an INCOMPLETENESS relative to the paper's stated model (Fig 2.11's "angular deviation" + Eq 23-25/27 use `θ,k`); Algorithm 1 provides no proxy-orientation rule. Implemented verbatim; ledgered D14. `kO` ships as a prospective coefficient (active only if a future proxy-orientation constraint is added).
- **Eq 22-25**: `dP = proxyPos − physPos`; `dQ = proxyQuat ⊗ physQuat⁻¹`; `theta = 2·acos(clamp(dQ.w,−1,1))`; `axisK = dQ.xyz / sin(theta/2)` guarded `theta≠0` (else zero).
- **Eq 27**: `hapticForce = kP·dP − dP_damp·linVel`; `hapticTorque = kO·theta·axisK − dO·angVel`. **Damping only during contact** (`g_inContact` from the signed test); NOT in contact ⇒ `dP=0` (proxy==phys) ⇒ `F=0`, and damping zeroed too.
- **Reaction to particles** (Newton's third law, thresholded, atomic): the force ON THE TISSUE is `−hapticForce`, clamped to `reactionMax`, scattered as `−hapticForce/3` onto the nearest ACTIVE corner of each of the min triangle's 3 world vertices via `k_ScatterReaction` with `atomicAdd` (paper's atomic rule). Skips inactive/severed corners (`physics_active()`), since `k_ComputeSlope` discards force on frozen particles anyway.
- **Buffer lifecycle** (review H2/H3 BLOCKER): `haptics_step` `cudaMemset(c_hapticForce, 0, …)` at its START each frame (before the scatter), so the reaction written after `LCS_Finalize` survives into the NEXT frame's `LCS_Step` RK45 and `physics_step` never wipes it.

- [ ] **Step 1: Write the failing oracle tests**

Append to `haptics_oracle.py`:

```python
def algorithm1_outer(tris, s, e, r, a=0.1, eps=1e-3, outer=64):
    # Mirrors the CUDA haptics_step outer loop EXACTLY (Algorithm 1 line 16 = smallest ||D|| among
    # PENETRATING triangles; contact re-evaluated each iteration; stop on no-penetration or
    # ||v_ms|| < eps).
    proxyS, proxyE = np.asarray(s,float), np.asarray(e,float)
    for _ in range(outer):
        best=None; bestlen=1e30
        for tri in tris:
            vms,ax,depth = alt_project(tri, proxyS, proxyE, r)
            if ax < r and depth < bestlen:              # penetrating only; min depth (= min ||D||)
                bestlen=depth; best=vms
        if best is None or np.linalg.norm(best) < eps: break   # no penetration => on surface, stop
        proxyS = proxyS + a*best; proxyE = proxyE + a*best     # P_t += a*v_ms
    return proxyS, proxyE

def residual_penetration(tris, s, e, r):
    # deepest remaining penetration depth over all triangles; <=0 if fully separated
    return max(alt_project(tri, s, e, r)[2] for tri in tris)   # [2] = depth = r - axisDist

def test_algorithm1_outer():
    # tool pressing DOWN from above: big triangle at z=0, capsule axis starts at z=+0.2 (penetrating,
    # axisDist 0.2 < 0.5). Outward = +z, so the proxy is pushed UP until the capsule bottom (axis - r)
    # rests on the surface z=0, i.e. axis converges to z ~ +0.5 = r.
    tri=[[-2,-2,0],[3,-2,0],[-2,3,0]]
    ps,pe = algorithm1_outer([tri],[0,0,0.2],[1,0,0.2],0.5)
    assert ps[2] > 0.2, ps                                    # moved outward (+z)
    assert residual_penetration([tri], ps, pe, 0.5) < 2e-3, ps  # residual penetration sub-threshold
    assert abs(ps[2] - 0.5) < 0.05, ps                        # axis ~ r above the surface

def eq27(proxyPos, physPos, linVel, kP, dP, contact):
    dP_vec = np.asarray(proxyPos,float) - np.asarray(physPos,float)
    F = kP*dP_vec - (dP*np.asarray(linVel,float) if contact else 0.0)
    return F
def test_eq27_forcetorque():
    F = eq27([0,0,0.1],[0,0,0.0],[0,0,-1.0], kP=800.0, dP=5.0, contact=True)
    assert np.allclose(F, [0,0, 800*0.1 - 5*(-1.0)]), F   # 80 + 5 = 85 up
    F0 = eq27([0,0,0.0],[0,0,0.0],[0,0,-1.0], kP=800.0, dP=5.0, contact=False)
    assert np.allclose(F0,[0,0,0]), F0                     # no contact -> no force/damping
```

- [ ] **Step 2: Run to verify it fails**

Run: `python cuda_plugin/tests/haptics_oracle.py`
Expected: the two new tests pass as pure NumPy (they define the reference); to gate CUDA, also add `def test_api_present(): assert "haptics_step" in open(...'haptics.cu').read()` → fails.

- [ ] **Step 3: Implement the outer loop + Eq 22-27 + scatter + host API**

In `haptics.cu` add the reaction-scatter kernel and the host functions. Key pieces (full code):

```cpp
// NOTE: struct HapticParams is defined in haptics.h (see the interface block above) so plugin_api
// and the API decls see it; haptics.cu just includes haptics.h. Do NOT redefine it here.

static HapticParams g_hp = {};
static int    g_cornerCount = 0;
static float3 g_physPos, g_proxyPos_dev;   // proxy P_t tracked on host
static float4 g_physQuat, g_proxyQuat;
static float3 g_linVel, g_angVel;
static float3 g_force, g_torque;
static bool   g_inContact = false;
static float3* c_cornerPos = nullptr;      // recon_corner_pos()
static float3* c_hapticForce = nullptr;    // physics_haptic_force()

// Scatter −F/3 (clamped) onto the nearest ACTIVE corner of each of the 3 triangle vertices.
__global__ void k_ScatterReaction(const float3* soup, int triIndex, float3 reaction,
                                  const float3* cornerPos, const int* active,
                                  int cornerCount, float3* hapticForce)
{
    int lane = blockIdx.x * blockDim.x + threadIdx.x;   // 0..2 (three vertices)
    if (lane >= 3) return;
    float3 v = soup[3 * triIndex + lane];
    int best = -1; float bd = 1e30f;
    // PERF (review H3): O(cornerCount) per vertex. At ~54k particles (paper §3.4) this is 3·cc per
    // FRAME (the scatter runs once, after the outer loop — NOT per outer iteration). If it shows on
    // a profile, cache the 3 nearest corners once per frame or bound the search with the voxel grid.
    for (int cptr = 0; cptr < cornerCount; cptr++) {
        if (active[cptr] == 0) continue;                 // skip frozen/severed (slope discards them)
        float3 d = cornerPos[cptr] - v; float dd = dot3(d, d);
        if (dd < bd) { bd = dd; best = cptr; }
    }
    if (best < 0) return;
    float3 share = reaction * (1.f / 3.f);
    atomicAdd(&hapticForce[best].x, share.x);            // paper: atomic add for linear ops
    atomicAdd(&hapticForce[best].y, share.y);
    atomicAdd(&hapticForce[best].z, share.z);
}
```

Host `haptics_step()` outline (fill with the exact ops — reduction via a copy of `d_triVms` to host and a CPU argmin over `triCount`, which is trivial and race-free, OR an atomicMin on packed length; the plan chooses the host argmin for clarity and because triCount is modest):

```cpp
int haptics_step()
{
    // Buffer lifecycle (BLOCKER fix): haptics owns the zeroing — clear last frame's reaction now,
    // AFTER the previous frame's LCS_Step already consumed it, BEFORE this frame's scatter.
    cudaMemset(c_hapticForce, 0, (size_t)g_cornerCount * sizeof(float3));

    int idxCount = 0, vc = 0;
    recon_get_counts(&vc, &idxCount);                 // existing API — NOT a duplicate getter
    int triCount = idxCount / 3;

    // Proxy FOLLOWS the physical tool (fidelity: "proxy gradually approaches physical"). Seed the
    // proxy from the physical pose each frame; Algorithm 1 then pushes it back out during contact.
    g_proxyS = g_physS; g_proxyE = g_physE;           // physical capsule endpoints from set_physical_tool
    g_proxyPos_dev = g_physPos; g_proxyQuat = g_physQuat;
    g_inContact = false; g_minTri = -1;               // reset each frame (no stale contact)

    if (triCount > 0) {
        const float3* soup = recon_expand_pos();
        const float4* aux  = recon_expand_aux();
        for (int outer = 0; outer < g_hp.outerMax; outer++)
        {
            k_ClosestPerTriangle<<<(triCount+63)/64,64>>>(triCount, soup, aux,
                g_proxyS, g_proxyE, g_toolRadius, g_hp.eps, d_triVms);
            // GPU argmin (no per-iteration full-array D2H): reduce the non-negative key, then pick the tri.
            unsigned int initKey = 0xFFFFFFFFu; int initTri = 0x7FFFFFFF;
            cudaMemcpy(d_vmsKey, &initKey, sizeof(unsigned int), cudaMemcpyHostToDevice);
            cudaMemcpy(d_vmsTri, &initTri, sizeof(int), cudaMemcpyHostToDevice);
            k_ReduceVmsKey<<<(triCount+63)/64,64>>>(triCount, d_triVms, d_vmsKey);
            unsigned int winKey = 0; cudaMemcpy(&winKey, d_vmsKey, sizeof(unsigned int), cudaMemcpyDeviceToHost);
            k_PickVmsTri<<<(triCount+63)/64,64>>>(triCount, d_triVms, winKey, d_vmsTri);
            int minTri = 0; cudaMemcpy(&minTri, d_vmsTri, sizeof(int), cudaMemcpyDeviceToHost);
            if (minTri < 0 || minTri >= triCount) break;
            float4 win; cudaMemcpy(&win, &d_triVms[minTri], sizeof(float4), cudaMemcpyDeviceToHost);
            // Contact is RE-EVALUATED every iteration (review V2 major — no latch): key >= 1e29
            // means NO triangle penetrates this iteration => proxy is on the surface, STOP without
            // stepping. Only a penetrating min (key = ||D||) advances the proxy.
            if (win.w >= 1e29f) break;                            // no penetration -> done
            g_minTri = minTri; g_inContact = true;                // we pushed out => in contact
            float3 vMs = make_float3(win.x, win.y, win.z);
            if (length3(vMs) < g_hp.eps) break;                   // converged (||v_ms|| < eps)
            float3 step = vMs * g_hp.a;                            // P_t += a * v_ms
            g_proxyS = g_proxyS + step; g_proxyE = g_proxyE + step;
            g_proxyPos_dev = g_proxyPos_dev + step;
        }
    }

    // Eq 22-25
    float3 dP = g_proxyPos_dev - g_physPos;
    float4 dQ = quatMul(g_proxyQuat, quatConj(g_physQuat));
    float theta = 2.f * acosf(fminf(fmaxf(dQ.w, -1.f), 1.f));
    float3 axisK = make_float3(0,0,0);
    float sh = sinf(theta * 0.5f);
    if (fabsf(sh) > 1e-6f) axisK = make_float3(dQ.x, dQ.y, dQ.z) * (1.f / sh);
    // Eq 27 (damping only in contact; no contact => dP==0 => F==0)
    float dpd = g_inContact ? g_hp.dP : 0.f;
    float dod = g_inContact ? g_hp.dO : 0.f;
    g_force  = dP * g_hp.kP - g_linVel * dpd;
    g_torque = axisK * (g_hp.kO * theta) - g_angVel * dod;
    // Reaction to particles (only in contact): −F, clamped, onto the min triangle's 3 active corners
    if (g_inContact && g_minTri >= 0) {
        float3 reaction = make_float3(-g_force.x, -g_force.y, -g_force.z);
        float rlen = length3(reaction);
        if (rlen > g_hp.reactionMax && rlen > 1e-9f) reaction = reaction * (g_hp.reactionMax / rlen);
        k_ScatterReaction<<<1,3>>>(recon_expand_pos(), g_minTri, reaction,
                                   c_cornerPos, c_active, g_cornerCount, c_hapticForce);
    }
    return (cudaGetLastError() == cudaSuccess) ? 0 : -1;
}
```

Add the quaternion helpers (full bodies; `float4` stored as `(x,y,z,w)`, review H3):

```cpp
__host__ __device__ __forceinline__ float4 quatConj(float4 q)
{ return make_float4(-q.x, -q.y, -q.z, q.w); }
__host__ __device__ __forceinline__ float4 quatMul(float4 a, float4 b)  // Hamilton product a⊗b
{
    return make_float4(
        a.w*b.x + a.x*b.w + a.y*b.z - a.z*b.y,
        a.w*b.y - a.x*b.z + a.y*b.w + a.z*b.x,
        a.w*b.z + a.x*b.y - a.y*b.x + a.z*b.w,
        a.w*b.w - a.x*b.x - a.y*b.y - a.z*b.z);
}
```

(Note: `test_eq27_forcetorque` exercises only the position/force path since `θ=0` per D14; the quat
order is not covered by the TDD gate — add a small `test_quat` unit if the orientation path is ever
activated.) Also add: host state
`g_physS,g_physE` (physical capsule endpoints), `g_minTri`, `c_active` (= `(int*)physics_active()`),
`c_hapticForce` (= `(float3*)physics_haptic_force()`), `c_cornerPos` (= `recon_corner_pos()`);
`haptics_init` (malloc `d_triVms[triCapacity]`, `d_vmsKey[1]`, `d_vmsTri[1]`; cache
`c_cornerPos = recon_corner_pos()`, `c_hapticForce = (float3*)physics_haptic_force()`,
`c_active = (int*)physics_active()`, store `HapticParams`);
`haptics_set_physical_tool` (store phys pose/vel + derive `g_physS/E` from the tool `[S,E]`+radius
EVERY call — the proxy re-seeds from these each frame); `haptics_get_force`/`haptics_get_proxy`/
`haptics_shutdown`. Declare them all in `haptics.h`. **Perf note:** the outer loop still does a few
small (4-16 B) D2H copies per iteration for the reduced key/tri; that is `O(outer)` tiny syncs, not
the `O(outer × triCount)` full-array copy the first draft had (review H3 major). Cap `outerMax` at
~64 and document that Algorithm 1 converges in a handful of iterations for a convex capsule.

- [ ] **Step 4: Run oracle + build**

Run: `python cuda_plugin/tests/haptics_oracle.py`
Expected: `PASS`
Run: `cmake --build cuda_plugin/build --config Release 2>&1 | grep -E "error|LiverCudaSim.dll"`
Expected: DLL built, no error.

- [ ] **Step 5: Commit**

```bash
git add cuda_plugin/cuda/haptics.cu cuda_plugin/cuda/haptics.h cuda_plugin/tests/haptics_oracle.py
git commit -m "feat(haptics): Algorithm 1 outer loop + Eq 22-27 force/torque + reaction scatter"
```

---

### Task 6: Plugin C ABI exports

**Files:**
- Modify: `cuda_plugin/src/plugin_api.cpp` (add exports near `LCS_GetCutDebug`)
- Test: build + a `python` grep test `haptics_oracle.py::test_exports`

**Interfaces:**
- Produces C exports:
  - `int LCS_InitHaptics(int cornerCount, int triCapacity, const HapticParams* p)`
  - `void LCS_SetPhysicalTool(const float* s3, const float* e3, const float* quat4, const float* linVel3, const float* angVel3, float toolRadius)`
  - `int LCS_StepHaptics()`
  - `void LCS_GetHapticForce(float* outForce3, float* outTorque3)`
  - `void LCS_GetProxyTool(float* outPos3, float* outQuat4)`
  - `void LCS_ShutdownHaptics()`

- [ ] **Step 1: Failing test**

Append to `haptics_oracle.py`:

```python
def test_exports():
    s=(pathlib.Path(__file__).resolve().parents[1]/"src"/"plugin_api.cpp").read_text(errors="ignore")
    for e in ("LCS_InitHaptics","LCS_SetPhysicalTool","LCS_StepHaptics","LCS_GetHapticForce","LCS_GetProxyTool","LCS_ShutdownHaptics"):
        assert e in s, e
```

- [ ] **Step 2: Run to verify it fails**

Run: `python cuda_plugin/tests/haptics_oracle.py` → `AssertionError: LCS_InitHaptics`

- [ ] **Step 3: Add the exports**

In `plugin_api.cpp` add `#include "haptics.h"` and:

```cpp
LCS_API int  LCS_InitHaptics(int cornerCount, int triCapacity, const HapticParams* p) { return haptics_init(cornerCount, triCapacity, p); }
LCS_API void LCS_SetPhysicalTool(const float* s3, const float* e3, const float* quat4, const float* linVel3, const float* angVel3, float toolRadius) { haptics_set_physical_tool(s3, e3, quat4, linVel3, angVel3, toolRadius); }
LCS_API int  LCS_StepHaptics() { return haptics_step(); }
LCS_API void LCS_GetHapticForce(float* outForce3, float* outTorque3) { haptics_get_force(outForce3, outTorque3); }
LCS_API void LCS_GetProxyTool(float* outPos3, float* outQuat4) { haptics_get_proxy(outPos3, outQuat4); }
LCS_API void LCS_ShutdownHaptics() { haptics_shutdown(); }
```

- [ ] **Step 4: Run test + build**

Run: `python cuda_plugin/tests/haptics_oracle.py` → `PASS`
Run: `cmake --build cuda_plugin/build --config Release 2>&1 | grep -E "error|LiverCudaSim.dll"` → DLL built.

- [ ] **Step 5: Commit**

```bash
git add cuda_plugin/src/plugin_api.cpp cuda_plugin/tests/haptics_oracle.py
git commit -m "feat(haptics): C ABI exports for the 2.3 haptic module"
```

---

### Task 7: C# driver — physical tool, proxy render, per-frame step, HUD

**Files:**
- Create: `Assets/ReconGridDC/Cuda/HapticTool.cs`
- Modify: `Assets/ReconGridDC/Cuda/LiverCudaManager.cs` (init haptics after cut init; call `LCS_StepHaptics` each frame after `LCS_Finalize`; shutdown)

**Interfaces:**
- Consumes: the six `LCS_*Haptics*` exports.
- Produces: a keyboard-driven physical tool (reusing the rod-style capsule), a rendered proxy capsule (`LineRenderer` or a stretched capsule mesh), and a HUD line showing `|F|`, `|τ|`, contact state. Optional `#if OPENHAPTICS` block feeding force/torque to the device (compiled out by default; documented).

- [ ] **Step 1: Write the C# P/Invoke + struct + driver**

Create `Assets/ReconGridDC/Cuda/HapticTool.cs`:

```csharp
using System.Runtime.InteropServices;
using UnityEngine;

namespace ReconGridDC.Cuda
{
    // Paper 2.3: SDF Finger-Proxy haptic driver. Physical tool = keyboard capsule (or OpenHaptics
    // device); proxy stays on the tissue surface (native Algorithm 1); force/torque from Eq 22-27.
    public sealed class HapticTool
    {
        const string DLL = "LiverCudaSim";
        [StructLayout(LayoutKind.Sequential)]
        public struct HapticParams { public float kP, dP, kO, dO, a, eps, reactionMax; public int outerMax; }

        [DllImport(DLL)] public static extern int  LCS_InitHaptics(int cornerCount, int triCapacity, ref HapticParams p);
        [DllImport(DLL)] public static extern void LCS_SetPhysicalTool(float[] s3, float[] e3, float[] quat4, float[] linVel3, float[] angVel3, float toolRadius);
        [DllImport(DLL)] public static extern int  LCS_StepHaptics();
        [DllImport(DLL)] public static extern void LCS_GetHapticForce([Out] float[] outForce3, [Out] float[] outTorque3);
        [DllImport(DLL)] public static extern void LCS_GetProxyTool([Out] float[] outPos3, [Out] float[] outQuat4);
        [DllImport(DLL)] public static extern void LCS_ShutdownHaptics();

        readonly float[] _s = new float[3], _e = new float[3], _quat = new float[4], _lin = new float[3], _ang = new float[3];
        readonly float[] _force = new float[3], _torque = new float[3], _ppos = new float[3], _pquat = new float[4];
        Vector3 _prevPos; Quaternion _prevRot; bool _have;
        public Vector3 Force, Torque, ProxyPos; public Quaternion ProxyRot; public bool Ready;

        public void Init(int cornerCount, int triCapacity)
        {
            var p = new HapticParams { kP = 800f, dP = 5f, kO = 20f, dO = 0.5f, a = 0.1f, eps = 1e-3f, reactionMax = 50f, outerMax = 64 };
            Ready = LCS_InitHaptics(cornerCount, triCapacity, ref p) == 0;
        }

        // toolS/toolE = the physical tool CAPSULE endpoints (like CuttingTool.S/E); physRot = its
        // orientation; toolRadius = the DEDICATED haptic proxy radius (NOT the cut-blade gap D).
        public void Step(Vector3 toolS, Vector3 toolE, Quaternion physRot, float toolRadius, float dt)
        {
            if (!Ready) return;
            Vector3 physPos = 0.5f * (toolS + toolE);
            // §3.1 note (review H3-m1): the keyboard path finite-differences v/ω here — device INPUT
            // kinematics, not haptic compute. With the real Omni/OpenHaptics device, its API supplies
            // v and ω directly, so no C# compute occurs; all §2.3 force math stays native.
            Vector3 lin = _have && dt > 1e-6f ? (physPos - _prevPos) / dt : Vector3.zero;
            Vector3 ang = Vector3.zero;
            if (_have && dt > 1e-6f) { (physRot * Quaternion.Inverse(_prevRot)).ToAngleAxis(out float a, out Vector3 ax); if (a > 180f) a -= 360f; ang = ax.normalized * (a * Mathf.Deg2Rad / dt); }
            _prevPos = physPos; _prevRot = physRot; _have = true;

            _s[0]=toolS.x; _s[1]=toolS.y; _s[2]=toolS.z; _e[0]=toolE.x; _e[1]=toolE.y; _e[2]=toolE.z;
            _quat[0]=physRot.x; _quat[1]=physRot.y; _quat[2]=physRot.z; _quat[3]=physRot.w;
            _lin[0]=lin.x; _lin[1]=lin.y; _lin[2]=lin.z; _ang[0]=ang.x; _ang[1]=ang.y; _ang[2]=ang.z;
            LCS_SetPhysicalTool(_s, _e, _quat, _lin, _ang, toolRadius);
            LCS_StepHaptics();
            LCS_GetHapticForce(_force, _torque);
            LCS_GetProxyTool(_ppos, _pquat);
            Force = new Vector3(_force[0],_force[1],_force[2]);
            Torque = new Vector3(_torque[0],_torque[1],_torque[2]);
            ProxyPos = new Vector3(_ppos[0],_ppos[1],_ppos[2]);
            ProxyRot = new Quaternion(_pquat[0],_pquat[1],_pquat[2],_pquat[3]);
        }
        public void Shutdown() { if (Ready) LCS_ShutdownHaptics(); Ready = false; }
    }
}
```

- [ ] **Step 2: Wire into LiverCudaManager**

In `LiverCudaManager` add a `HapticTool _haptics = new HapticTool();` field + serialized `bool enableHaptics = true;` + a second keyboard capsule (reuse the rod input path or add IJKL/UO keys). After the cut init in `Start`:

```csharp
if (enableHaptics) _haptics.Init(cc, triCapacity);
```

In `Update` after `LCS_Finalize()` (so the reconstructed triangles are current). Drive a SEPARATE
haptic capsule with its own endpoints `hapticS/hapticE` (keyboard IJKL/UO, independent of the cut
rod) and a dedicated `hapticToolRadius` field (default `0.6f * Lrt`, NOT the cut gap `D`):

```csharp
if (enableHaptics && _haptics.Ready)
    _haptics.Step(hapticS, hapticE, hapticRot, hapticToolRadius, Time.deltaTime);
```

Add serialized fields: `public float hapticToolRadius = 0.6f * 0.37f; // ~0.6*Lrt, tuned in Task 8`
(or set it from `Lrt` in `Start`); `hapticS/hapticE/hapticRot` are the haptic capsule state updated
by the IJKL/UO input each frame.

In `OnDestroy`: `_haptics.Shutdown();` (before `LCS_Shutdown`).

Add a HUD line in `OnGUI`:

```csharp
if (enableHaptics && _haptics.Ready)
    GUI.Label(new Rect(10, 34, 900, 22), $"HAPTIC |F|={_haptics.Force.magnitude:F2} |t|={_haptics.Torque.magnitude:F3}  proxy={_haptics.ProxyPos}");
```

- [ ] **Step 3: Build DLL, deploy, run in Unity**

Run: `cmake --build cuda_plugin/build --config Release 2>&1 | grep -E "error|LiverCudaSim.dll"` → DLL built.
Deploy (Unity closed): copy `cuda_plugin/build/Release/LiverCudaSim.dll` → `Assets/Plugins/x86_64/`.
Play: drive the haptic tool into the liver; verify the HUD shows `|F|` rising on contact and the proxy staying on the surface (no penetration).

- [ ] **Step 4: Commit**

```bash
git add Assets/ReconGridDC/Cuda/HapticTool.cs Assets/ReconGridDC/Cuda/LiverCudaManager.cs
git commit -m "feat(haptics): C# driver — physical tool, proxy render, per-frame Eq 22-27 step"
```

---

### Task 8: Integration verification + parameter tuning + docs

**Files:**
- Modify: `progress.md` (append the haptic-rendering entry)
- Modify: `Assets/ReconGridDC/Core/DeviationLedger.cs` (add the six contiguous entries D10-D15 — see Step 2)

**Interfaces:** none (verification task).

- [ ] **Step 1: Verify the paper-fidelity checklist against the running build**

Confirm, in a live run:
- Proxy tool stays OUTSIDE the tissue on contact (no penetration) — Algorithm 1 working.
- `|F|` along the push axis grows during press (Fig 3.7 behavior); when sliding tangentially, `F` gains a tangential component (the implicit friction) — verify by pushing then sliding and watching the HUD `F` vector.
- Releasing contact drops `|F|` and `|τ|` to ~0 and the damping terms vanish (Eq 27 contact-only damping).
- Reaction force visibly perturbs nearby particles (surface dents slightly under the proxy) but never explodes (reactionMax clamp).

- [ ] **Step 2: Record deviations**

In `DeviationLedger.cs` add the contiguous entries D10-D15 (existing ledger ends at D9):
- `D10`: Algorithm 1's `while True` gets an `outerMax` safety cap (non-convergence guard only; converges well before it in practice).
- `D11`: reaction force scattered onto each triangle vertex's NEAREST active corner (the soup vertices are FP slots, not particle ids) — faithful to "applied to the particles of the nearby triangle faces," atomic-add per the paper's parallelism rule.
- `D12`: haptic collision uses SKIN triangles only (cut-wall bit30 triangles skipped) — the tool collides with the external tissue surface, not the interior of an open wound (§2.3 "triangular faces of the soft tissue MODEL").
- `D13`: the proxy is re-seeded to the physical pose each frame before the outer loop, then pushed out by contact — the minimal realization of §2.3's "the proxy gradually approaches the Physical instrument" (within-frame convergence replaces the paper's cross-frame persistent `P_t`; identical steady-state `ΔP`); NOT a separate spring equation.
- `D14`: Algorithm 1's Output is "Proxy Tool Position `P_t`" (position only, paper line 866) — it specifies no proxy-ORIENTATION update. Consequently `Δq = identity`, `θ = 0`, and Eq 27's restoring torque `K_o·θ·k` is inactive (only the damping torque `−D_o·ω` is produced). **This is faithful to Algorithm 1 (verbatim) but an INCOMPLETENESS relative to the paper's STATED MODEL: Fig 2.11(a) depicts an "angular deviation" and Eq 23-25/27 define/use `θ,k`. A future proxy-orientation constraint would realize it; `kO` ships prospective. No future reader should mistake `θ=0` for a paper requirement — the paper requires an angular deviation the algorithm does not produce.** Eq 22-27 are still implemented verbatim; the angular term is simply zero because Algorithm 1 does not move the proxy's orientation.
- `D15`: the paper realizes `SDF(Tool, A)` (Eq 26 / Algorithm 1 line 12) as the nearest point on the capsule SURFACE; the plan projects onto the capsule AXIS instead, because the capsule's directed distance field is `dist_to_axis − r` (line 799-802) and the surface projection DEGENERATES under penetration (the alternating projection collapses to `axisDist ≈ r`, losing the depth the paper's line 855-857 penetration test needs). With the axis point `P_i`, the separation `v_ms = (r − axisDist)·normalize(P_i − A)` is ALREADY the outward push (under penetration `P_i` sits on the tool side of the surface, so `P_i − A` points out of the tissue) with magnitude `r − axisDist` = the paper's `‖D‖` penetration depth. Triangle-face-normal fallback when the axis passes exactly through `A`. This is a faithful realization of the capsule SDF (magnitude = paper `‖D‖`, direction = paper line-916 "moved outward"), not a hand-flipped sign.

- [ ] **Step 3: Append progress.md**

Add a dated entry summarizing: SDF Finger-Proxy §2.3 implemented (Eq 22-27 + Algorithm 1 verbatim; friction is the implicit tangential Kp·ΔP, NOT a Coulomb term), new `haptics.cu/.h` + 6 exports + `HapticTool.cs`, per-particle `d_hapticForce` reaction path, DLL md5.

- [ ] **Step 4: Commit**

```bash
git add progress.md Assets/ReconGridDC/Core/DeviationLedger.cs
git commit -m "docs(haptics): 2.3 fidelity checklist, deviation ledger D10-D15, progress"
```

---

## Self-Review (author's pass, post review-round-1 wf_567e2ec2)

- **Spec coverage:** Eq 22 (dP) → Task 5; Eq 23-25 (dQ/θ/k) → Task 5; Eq 26 (alternating projection, `D=P_i−A_i`) → Task 4; Algorithm 1 inner loop + `i<100` → Task 4, outer loop + `a=0.1`/`ε=1e-3` → Tasks 4-5; Eq 27 (F/τ, contact-only damping) → Task 5; proxy-follows-physical + penetration outward push (signed `axisDist<r`) → Task 5; reaction to particles (thresholded, atomic, active-only) → Tasks 1+5; skin-only collision → Tasks 2/4; proxy render + physical tool hidden → Task 7; threading/data-race + buffer lifecycle → Tasks 1/5. Friction (implicit tangential `Kp·ΔP`) → Global Constraints + Task 8 verify. **All §2.3 elements mapped.**
- **Review-round-1 fixes folded in:** off-by-one `D=P_i−A_i` (was `P_{i+1}−A_i`) in kernel + oracle; proxy re-seeds from physical each frame (was first-call only); signed contact `axisDist<r` (was unsigned surface gap); `d_hapticForce` zeroed by `haptics_step` not `physics_step` (BLOCKER — was wiped before consumed); `recon_get_counts` reused (dropped duplicate `recon_expand_count`); GPU argmin reduction (was per-outer full-array D2H); dedicated `toolRadius` (was `D/2`); skin-only via aux wall flag; scatter skips inactive corners; strengthened oracle (convergence + penetration-sign fixtures).
- **Placeholder scan:** every code step shows real code; the full `haptics_step` body + `k_ScatterReaction` + `k_ClosestPerTriangle` + reduction kernels are complete; helpers named to implement (`quatMul`/`quatConj`/`ClosestOnSegment`/init/set/get) — no TODOs.
- **Type consistency:** `HapticParams` (7 float + 1 int) identical in haptics.h/plugin_api/C# (Tasks 5/6/7). `LCS_SetPhysicalTool(s3,e3,quat4,linVel3,angVel3,radius)` consistent Tasks 5/6/7. `physics_haptic_force()` → `void*` (Task 1) cast in haptics.cu (Task 5). `recon_expand_pos()/recon_expand_aux()` + `recon_get_counts` consistent Tasks 2/4/5.
- **Grep-gate note (review H3 minor):** the `haptics_oracle.py` source-presence asserts (Tasks 1/2/3/6) are BUILD/presence gates; the NumPy `test_sdf_math`/`test_alternating_projection`/`test_penetration_sign`/`test_algorithm1_outer`/`test_eq27_forcetorque` are the real behavioral gates (they pin the paper math). Task steps label which is which.

## Open fidelity question flagged for review

The paper does not give numeric values for `K_p, D_p, K_o, D_o` (only the Fig 3.7 qualitative curve). Task 5/7 ship defaults (`kP=800, dP=5, kO=20, dO=0.5`) as a STARTING point to be tuned at the visual check — flagged so reviewers confirm this is an acceptable "paper-silent numeric" (like the physics `globalDamp`), not a fidelity break.
