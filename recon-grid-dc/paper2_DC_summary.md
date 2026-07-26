# Paper 2 — DC Cutting (the NEW basis, replacing MC2024)

**"A real-time reconstructed grid method for soft tissue cutting and haptic"** — Li, Zhou, Zhou (China Simulation Sciences Co.), CMPB 273 (2026) 109123. Open access. Video: https://youtu.be/eK_QKEXwyag
Full text: `paper2_dualcontouring_full.txt`. Stack: Unity3D 2021.3 (C# render) + C++/CUDA (compute) + Omni/OpenHaptics.

## Essence
GPU **Dual Contouring (DC)** on a **deformable hexahedral voxel grid** for real-time cutting. The voxel grid's 12 edges double as (a) DC reconstruction edges AND (b) the mass-spring physical links. Cutting = mark edges "cut" → recompute each voxel's **undirected-graph connectivity** (8 verts/12 edges) → a cut voxel yields **up to 8 feature points** (one per connected component) → surface re-stitched. **No new particles/voxels are ever added** → frame rate doesn't fall as cuts accumulate (the headline advantage over tetrahedral). Cut width = tool thickness D (thin blade → thin gap → can approach a slit).

## Pipeline (Fig 3.1)
**Preprocess (once):** AABB → uniform background grid → mark voxel/corner intersection state vs isosurface (MC lookup table) → GPU computes 1 feature point/voxel → connect feature pts of the 4 voxels around each intersecting edge → 2 triangles → outer surface.
**Real-time (3 CPU threads + Unity render):**
1. **Physics thread** — mass-spring forces/positions/orientations of particles → update the 8 corner points of each voxel.
2. **Geometry thread** — apply cuts; recompute feature points + surface; sever particle spring links accordingly.
3. **Haptics thread** — SDF collision (tool↔soft-tissue triangles), finger-proxy position update, force feedback → reaction force on nearby triangle particles.
Rule: no two CPU threads write same data; GPU force accumulation via atomicAdd (≤6 per particle); avoid excess atomics.

## Core algorithm A — Dual Contouring feature point (Eq 1–5)
- QEF (Eq1): E = Σ (n_i·(x − p_i))²  (p_i, n_i = intersection point + normal on the i-th cut edge).
- Avoid matrix solve → iterative particle-movement + Hermite [ref 31]:
  - x0 = mean of intersection points (Eq2).
  - F = Σ −(x0 − p_i)·n_i²  (Eq3, descent direction).
  - x_{k+1} = x_k + a·F, a = 0.1·(1 − i/m), m = max iters (Eq4).
  - Stop when box-SDF of voxel (Eq5) < 0, i.e. feature point inside its voxel → smooth, non-self-intersecting.

## Core algorithm B — Cutting
- **Static (2.1.1):** voxel = undirected graph (8 verts, 12 edges). Each edge: cut/not = 1 bit → 12-bit index. **Precomputed 2^12 = 4096-entry lookup table** of connectivity (bitwise-OR to update when edge i is cut). Per connected component: gather the cut edges + cut points of its verts; **feature point = centroid of those cut points** (a cut voxel → up to 8 feature points). Then per particle: walk its 6 edges + 8 octants; for each cut edge find the 4 neighbor-octant feature points → 2 triangles (Fig 2.5).
- **Dynamic (2.1.2):** before a cut the grid is deformed (not a uniform cube); each particle has a rotated local coordinate frame (computed à la Berndt [22] — the MC2024 ref). The 6 ± axes map to the 6 edges; a cut point is stored as **local coords on the undeformed edge**, then re-evaluated through the particle's current frame each frame so the cut point follows deformation.
- **Cutting path (2.1.3):** tool = a line; **cutting plane = 2 triangles spanning the tool line of the previous + current frame** (persists → no missed collisions at low FPS). Collision = **Möller–Trumbore** ray-triangle (Eq6–8, Cramer).
- **Ablation cutting (2.1.4):** tool path region: particles OUTSIDE isosurface deleted; particles INSIDE → "ablated", their edges → "ablated edges", particles deleted, cut points on ablated edges, surface redrawn (electrocautery, intended volume loss).

## Core algorithm C — Physics (2.2)
Structural + bending springs on hexa grid (Kelvin–Voigt; NO diagonal shear springs). Per particle ≤6 neighbors, 6 structural springs, 15 bending pairs.
- Eq9: M·ẍ_i + F_int = F_ext;  Eq10: F_int = F_s + F_b.
- Structural (Eq11–12): F_s_ij = −k_s·(stretch)·dir − c_s·(ẋ_i−ẋ_j); F_s_i = Σ_{j=0..5}.
- Bending (Eq13–16): moment M_in = k_b·(θ_jk − θ0_jk); forces via cross products + c_b·θ̇; F_b_i = Σ_{n=0..14}.
- **Integration: adaptive Dormand–Prince RK45** (Eq17–21): 7 slopes, 4th+5th order, Δ=||y5−y4||, h_n = h·0.9·(Δ/ε)^{1/5}, ε=1e-3, 0.2h<h_n<5h.

## Core algorithm D — Haptics (2.3, SEPARABLE — defer in v1)
Tool ≈ slender **capsule** as a (signed) distance field; force from **tool↔tissue-triangle** collision (not particle-tool). Hidden physical tool drives a rendered **proxy** kept on the surface. Quaternion deviation → ΔP, axis k, angle θ (Eq22–25). Closest point via **alternating SDF projection** (Eq26 / Algorithm 1): A=SDF(triangle,P), P=SDF(tool,A), min-separation v_ms; if penetrating (dist<radius) push proxy out. Force (Eq27): F = K_p·ΔP − D_p·v, τ = K_o·θ·k − D_o·ω (damping only during contact).

## Params (3.4): voxel 0.25 cm; 54,301 particles; k_s=7.5e4 N/m, k_b=2e4 N/m, c_s=0.92, c_b=0.9; dt=0.02 s.
## Perf (Table 1): mesh reconstruction ~0.87–5.66 ms (9.5k–54k particles) vs Hexahedral+DC 5.13–23.92 ms.

## How it differs from MC2024 (and why thinner cuts + stable FPS)
- MC2024 = density-field MC on a fixed grid + shape-matching + AFCC components (our long pain: 1-component sever + welded wide seam). THIS paper = **deformable grid IS the physical mesh**, cut = real edge-cut + per-voxel graph disconnection, **feature points placed at cut-point centroids** → the cut surface is generated directly at the blade, width = tool thickness. No density splat, no welding, no component-count gating.
- No mesh duplication/subdivision → cuts don't add elements → **frame rate independent of cut count**.

## Known limitations (be honest)
- **Volume loss** persists (DC vertex averaging at internal cuts; lower than MC but nonzero).
- **Feature-point drift**: under large deformation the two cut sides can *suddenly* separate by a big distance instead of smoothly; they mitigate by interpolating post-cut feature point from prev→current frame (visual hack, not fully physical).
- Unlimited cutting / adaptive meshing not supported; heavy atomics (force accum + normal calc need barriers).

## Suggested from-scratch build order
1. Mass-spring + adaptive RK45 physics on a hexa voxel grid (verify: cube deforms/settles, energy stable).
2. DC outer-surface reconstruction (Eq1–5) on the grid (verify: watertight surface follows deformation).
3. Cutting: 4096 connectivity LUT + per-voxel components + cut-point centroid feature points + spring severing (verify: a planar cut severs the graph and renders two faces; FPS flat as cuts grow).
4. Cutting path (line→2-triangle plane, Möller–Trumbore) + curved/cross cuts.
5. Haptics (SDF capsule + finger-proxy) last.
