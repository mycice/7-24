# Liver Tuning — Performance + Surface Smoothness + Stiffness — Design Spec

**Date:** 2026-06-29
**Status:** Design draft → multi-subagent audit next
**Basis:** the paper's own validated parameters + framework (§2.2, §3.2, §3.4, Table 1) where it has a solution; mature established techniques (researched) where it does not.
**Context:** the liver loads, anchors, and cuts (LiverCutManager). Three issues to fix: (1) low frame rate, (2) bumpy / low-resolution surface, (3) liver far too soft (stretches at startup).

## Verified facts (this session)

- **Paper §3.4 (line 1143-1148) — the exact liver parameters:** `ks = 7.5×10⁴ N/m`, `kb = 2×10⁴ N/m`, `cs = 0.92`, `cb = 0.9`, **voxel `L = 0.25 cm`** (⇒ the model is in **centimetres**), `dt = 0.02 s`, **54,301 particles**, NVIDIA RTX A2000.
- **Paper Table 1:** their DC reconstruction at 54,301 particles = **5.66 ms** (vs 23.92 ms for plain Hexahedral+DC). Real-time is reached via the **§3.2 multi-threaded framework** (3 compute sub-threads + the Unity render thread) that **decouples reconstruction from physics**, plus the precomputed connectivity LUT and per-thread feature points (all already implemented here).
- **Perf root cause (code):** `MassSpringSolver.Step` runs an inner adaptive-RK45 substep loop capped by `hMax = 5e-4` ⇒ with `dt = 0.02` that is **≥40 substeps/frame**, each ~38 GPU dispatches (7 DoPri5 stages × ~5 force/stage kernels) ⇒ **~1500 `ComputeShader.Dispatch` calls/frame** at ~58k corners. The `hMax` field doc itself says the DoPri5 stability limit at `ks=7.5e4` is `h ≲ 3e-3`; `5e-4` is ~6× under it (a deliberate ~4× safety margin for the once-per-frame D6 controller).
- **My current liver grid (corrected):** `L` is **DERIVED** = `maxExtent / targetLongAxisVoxels`, NOT settable. At `tv = 48` on the 17.82-unit liver, `L ≈ 0.37` (matching the paper's `0.25 cm` would need `tv ≈ 71` ⇒ ~260k corners, **5× the paper** — not feasible single-threaded). At `tv = 48` the grid is **≈89k TOTAL corners (~23k active/tissue)** — physics dispatches run over ALL `CornerCount` (≈89k), DC over the surface voxels only. Demo softened `ks = 800` (LiverCutManager default) — this is why the liver stretches.
- **Literature (context):** healthy liver Young's modulus ≈ **2–10 kPa** (up to ~90 kPa with the Glisson capsule); shear modulus ~2–5 kPa. The paper's `ks = 7.5e4` is its calibrated soft-unit spring constant (with the code's `mass = 1`/corner giving ω≈hundreds rad/s), NOT a direct E; per the user's instruction we use the **paper's value directly**, with the literature noted for context.

## The three fixes are synergistic
Smoothing the level-set field (Fix 2) removes the need for high voxel resolution to look smooth ⇒ a **moderate** resolution stays smooth ⇒ **fewer particles** ⇒ helps Fix 1 (perf). The paper's `ks` (Fix 3) holds shape. `hMax` tuning (Fix 1) cuts substeps. None conflict.

---

## Fix 3 — Stiffness (do first; simplest; direct from the paper)

**Problem:** `ks = 800` (demo softening) ⇒ gravity overwhelms the springs ⇒ the liver stretches into a long strand at startup.

**Solution (paper §3.4, used VERBATIM):** set the liver defaults to the paper's validated values:
`ks = 7.5e4`, `kb = 2e4`, `cs = 0.92`, `cb = 0.9`, `dt = 0.02`, voxel `L = 0.25 cm`. Keep `mass = 1`/corner (the regime the paper's `ks` is calibrated for; the code's `hMax` doc assumes ω≈950 at this `ks`). The model is interpreted as **centimetres** (paper's `L = 0.25 cm`).

**Change:** `LiverCutManager`/`LiverSceneSetup` field defaults `ks 800 → 7.5e4`, `kb 200 → 2e4`, `cs_damp 20 → 0.92`, `cb 5 → 0.9`. Keep them Inspector-tunable, and **forward ALL four** through `LiverSceneSetup` (it currently forwards only `ks`/`gravity` — add `kb`/`cs`/`cb`). (Optionally expose a "stiffness scale" so the user can soften toward the literature 2–10 kPa feel without re-deriving.)

**Damping caution (audit):** adopting `cs = 0.92` verbatim with `mass = 1`/`ks = 7.5e4` drops the structural damping ratio from `ζ ≈ 0.35` (demo `cs=20/ks=800`) to **`ζ ≈ 0.0017` — i.e. the lattice is essentially UNDAMPED** and will *physically ring/oscillate* at equilibrium (independent of integrator stability, and worsened by a larger `hMax`). The paper's value is still the right starting point (per instruction), but **`cs` (or `mass`) is the equilibrium-settling knob — not `hMax`.** Expose `cs` AND `mass` (a uniform per-corner mass multiplier) in the Inspector; if the liver rings after the `ks` change, raise `cs` or `mass`. Note `ω = √(ks/m)`, so raising mass also lowers ω ⇒ allows a larger stable `hMax` ⇒ helps Fix 1 too.

**Result:** equilibrium sag becomes a few-percent strain (the paper's liver holds shape under a single fixed point + gravity, Fig 3.9) ⇒ no more stretching.

**Risk:** stiffer `ks` ⇒ smaller stable `h` ⇒ more substeps ⇒ feeds Fix 1. That is exactly why Fix 1 tunes `hMax` to the true stability limit (not 6× under it). The relevant ω for the DoPri5 stability limit is the lattice **max** eigenfrequency `ω ≈ 2√(3·ks/m) ≈ 948 rad/s` at `ks=7.5e4, m=1` (NOT the single-spring `√(ks/m) ≈ 274`) — `hMax = 1.5e-3` gives `ω·h ≈ 1.4`, ~2.3× inside the `≈3.3` imaginary-axis limit.

---

## Fix 1 — Performance

**Problem:** low frame rate. **Root cause — TWO co-dominant costs** (audit correction), both serial on the Unity render thread because the paper's §3.2 multi-threaded decoupling cannot be replicated for main-thread `ComputeShader.Dispatch`:
- **(A) Physics substeps:** ~40 substeps/frame × ~38 dispatches ≈ **1500 dispatches/frame**. Cut by the `hMax` lever below.
- **(B) `dc.Build` reconstruction:** the FULL DC pipeline (incl. `DC_FeaturePoints`' `qefIters=20` QEF iterations per **surface** voxel — interior voxels early-out at `cnt==0`) re-runs EVERY frame ≈ the paper's Table-1 **5.66 ms at 54k** (~⅓ of the 16.6 ms 60-fps budget). It is ~13 dispatches with HEAVY per-thread work, so the dispatch-count levers do NOT reduce it — it needs its own lever (below).

**Primary lever — match `hMax` to the real stability limit (cuts substeps ~3×).**
The adaptive RK45 (the paper's integrator) already takes the largest step the error controller allows; the over-conservative `hMax = 5e-4` cap defeats it. Raise it toward the documented DoPri5 limit (`h ≲ 3e-3` at `ks=7.5e4`) with a safety margin: **`hMax = 1.5e-3`** (2× margin) ⇒ ≈13 substeps/frame instead of 40 ⇒ **~3× fewer dispatches**. Expose `hMax` (already a public field) and document the stability/jitter trade-off so the user can tune up (faster) until jitter appears, then back off. (If raising `hMax` proves unstable because the D6 controller adapts only once per frame, add an optional per-substep error guard — halve+retry `h` when the embedded error exceeds tolerance — accepting one extra readback only on rejection.)

**Co-primary lever — cut `dc.Build` cost (cost B).** This is the fixed reconstruction cost the `hMax` lever cannot touch:
- **Reduce `qefIters`** 20 → ~8–12 (Inspector-tunable). `DC_FeaturePoints` runs the gradient-descent QEF `qefIters` times **per surface voxel**; halving the iterations ~halves that kernel. The feature point converges well before 20 iterations for a smoothed field; validate the surface is unchanged.
- **Resolution** (below) reduces the surface-voxel count that `dc.Build` iterates.
- **Optional cadence decouple (paper §3.2-inspired):** when the rod is NOT actively cutting/moving, `dc.Build` need only re-run because the tissue deforms — its rate can be throttled (e.g., every 2nd frame) while physics runs every frame. Only if (qefIters + resolution) are insufficient; risks visible lag during fast deformation. Keep `dc.Build` every frame while cutting.

**Secondary lever — reduce voxel resolution, enabled by Fix 2's smoothing.**
Because Fix 2 smooths the field, the surface no longer needs a fine grid to look smooth. Lower `targetLongAxisVoxels` (e.g., 48 → 36–40) ⇒ fewer total corners (≈89k→≈55k at 40) ⇒ fewer physics threads/dispatch AND fewer surface voxels for `dc.Build`. Tunable; the user's final FPS knob.

**Tertiary lever — fuse the per-substep force kernels (optional, math-preserving).**
Each substep dispatches `kClear + kStruct + kBend` separately per stage. Fusing the force-accumulation into one kernel (where the atomic/dependency structure allows — the paper warns against *excessive* atomics, §3.2) cuts ~2 dispatches/stage × 7 stages × N substeps. Implement only if (1)+(2) are insufficient, and only after confirming the fused kernel preserves the exact force math and avoids new atomic contention. **Not** a first-line change (it edits the core `Physics.compute`).

**Paper-alignment note (multi-threading gap, documented):** the paper's real-time at 54k particles relies on its §3.2 multi-threaded decoupling of reconstruction (5.66 ms) from physics. A single-threaded Unity `Update` cannot move GPU dispatches off the render thread, so we close the gap by reducing per-frame dispatch *count* (levers above) rather than by threading. This is an honest deviation, stated.

---

## Fix 2 — Surface smoothness

**Problem:** the surface is bumpy/faceted (screenshot). **Root cause:** the refined liver tet-mesh **boundary itself is rough**, and the exact pseudonormal SDF + DC faithfully reproduce that roughness; moderate voxel resolution adds faceting.

**Solution — a baked, smoothed SDF grid wrapper (mature technique; perf-neutral; NO core edit).**
Add `SmoothedSdfGrid : ILevelSetProvider` that wraps the expensive `MeshLevelSet`:
1. **Bake:** sample the base `MeshLevelSet.Sample` once onto its own dense scalar grid covering the liver bbox (resolution ≥ the reconstruction grid; tunable `bakeVoxels`). This is the brute-force SDF cost paid **once**.
2. **Smooth:** apply `K` iterations of Laplacian (or Taubin λ|μ, to avoid shrinkage) smoothing to the baked φ field — removes the high-frequency boundary roughness. `K` tunable (default ~2–4).
3. **Serve:** `Sample(p)` = trilinear interpolation of the smoothed φ grid; `Gradient(p)` = central-difference of the smoothed grid, **then `normalize`d** (the raw central difference of a smoothed field is NOT unit-length, but `ILevelSetProvider.Gradient` must be ~unit — `BackgroundGrid` uses it as the isect `normal` in the QEF), with the on-surface epsilon fallback (`length < 1e-8` → a safe unit). Feed THIS to `BackgroundGrid.Build`.
   - **Domain coverage (audit):** the bake grid must span the **FULL reconstruction domain** `[origin, origin + dims·L]` INCLUDING the margin shell (clamp queries to the grid border), so every corner sample AND every isosurface-crossing gradient `BackgroundGrid` requests has a valid trilinear + central-diff stencil. Size `bakeVoxels` so its spacing ≤ the reconstruction `L`, and **bound it** (e.g. cap ≈96³ ≈ 3.5 MB of floats) so a fine bake doesn't blow memory.

**Startup note (audit — modest, NOT a fix):** `BackgroundGrid`'s per-corner samples + per-crossing gradients become cheap trilinear lookups, which DOES eliminate the brute-force closest-tri *gradient* calls at every crossing. BUT the bake still pays the brute-force `MeshLevelSet.Sample` **once per bake node** (≈ the corner count), so startup is only **modestly** improved, not eliminated. The real R3 startup fix — a BVH/spatial-hash on `MeshLevelSet` — remains a documented follow-up.

**Levers:** `bakeVoxels` (bake resolution), `K` (smoothing iterations). Taubin over plain Laplacian if shrinkage is visible. A modest `targetLongAxisVoxels` is fine because the field is pre-smoothed.

**Risk:** over-smoothing rounds off real features (Taubin mitigates shrinkage); a too-coarse bake grid loses detail. Both tunable; validate against the un-smoothed surface.

---

## Architecture / files

- **`SmoothedSdfGrid.cs`** (NEW, `Preprocess/`) — `ILevelSetProvider` wrapper: bake + smooth + trilinear sample/gradient. Pure CPU, testable.
- **`MeshLevelSet.cs`** — unchanged (becomes the bake source).
- **`LiverCutManager.cs` / `LiverSceneSetup.cs`** — wrap `MeshLevelSet` in `SmoothedSdfGrid` before `BackgroundGrid.Build`; update physics defaults to the paper's values; expose `hMax`, `bakeVoxels`, `smoothIters`, `targetLongAxisVoxels` in the Inspector.
- **`MassSpringSolver.cs`** — `hMax` default `5e-4 → 1.5e-3` (tunable; documented). (Optional later: per-substep error guard; force-kernel fusion in `Physics.compute`.)
- Tests: `SmoothedSdfGrid_Tests.cs` (bake/smooth/interp correctness vs the base SDF; smoothness reduces field variance; gradient unit + finite).

## Reused verbatim
DC pipeline, `CutDetector`/`CuttingTool`, `BackgroundGrid.Build`, `MshLoader`, `MeshLevelSet` (as bake source), the RK45 integrator math (only `hMax` tuned). No cut/DC algorithm change.

## Testing
1. **Stiffness:** with the paper's `ks`, a CPU mass-spring equilibrium check (or the user's Unity check) shows sag < ~5% strain — no stretching. (Primary verification is the user in Unity.)
2. **SmoothedSdfGrid bake/interp:** with `K=0` (no smoothing) the wrapped sampler matches `BoxLevelSet` within the trilinear-bake tolerance in flat regions; `Gradient` is unit + outward; on-surface finite. With `K>0`, smoothing only deviates NEAR sharp edges/corners (rounds them) while staying close in flat regions — do NOT assert exact `BoxLevelSet` equality at a cube edge (smoothing legitimately rounds it). (Use a sphere bake for a stricter sign/distance check, since a smooth surface is preserved.)
3. **Smoothing reduces roughness:** the variance of the discrete Laplacian of the φ field decreases monotonically with `K` (a quantitative smoothness proxy), while the zero-level-set volume stays within tolerance (Taubin = low shrinkage).
4. **Perf (user, Unity):** report FPS before/after `hMax` change + resolution; confirm no NaN/jitter; tune `hMax` up until jitter, back off.
5. **Smoothness (user, Unity):** the liver surface is visibly smoother; tune `K`/`bakeVoxels`.

Each stage ends with a **2–3 subagent review** (paper-alignment + spec-conformance + correctness), per the standing requirement.

## Decisions
- Stiffness = the paper's exact `ks/kb/cs/cb` (§3.4), model in cm — direct paper solution.
- Perf = tune `hMax` to the real stability limit (primary) + moderate resolution (enabled by smoothing) + optional kernel fusion; the multi-thread gap is documented, not closed.
- Smoothness = a baked, Taubin-smoothed SDF grid wrapper (mature technique) — also fixes the slow startup; no core edit.

## Risks
- **R1:** raising `hMax` past the stability limit ⇒ jitter/blowup (no mid-frame rejection). Mitigation: 2× margin default + tunable + optional per-substep guard.
- **R2:** over-smoothing rounds real liver features. Mitigation: Taubin (anti-shrink) + tunable `K` + volume check.
- **R3:** the paper's `ks` regime assumes `mass=1`; if mass is later set from density, `ks`/`hMax` must be re-derived (ω = √(ks/m)).
- **R4:** single-threaded Unity cannot match the paper's threaded perf at 54k particles. TWO co-dominant costs (physics substeps + `dc.Build` reconstruction) both run serial on the render thread; `hMax` cuts only the first, `qefIters`+resolution cut the second. On a weaker GPU the levers may not reach 60 fps — resolution is the user's final knob.
- **R5 (audit):** `cs = 0.92` at `mass=1`/`ks=7.5e4` is nearly undamped (ζ≈0.002) ⇒ the equilibrium may physically RING after the `ks` change (separate from integrator stability, and amplified by a larger `hMax`). Mitigation: `cs` AND `mass` Inspector-tunable as the settling knobs; if it rings, raise `cs` (or `mass`) before touching `hMax`.
- **R6:** the rod/anchor geometry (`rodLength`, `moveSpeed`, `anchorBand`) is in model units and is UNAFFECTED by the `ks` change (the cut is geometric M-T, not force-based); no rescale needed. The existing anchor `≥1`-pinned guard still applies; just re-confirm after any `targetLongAxisVoxels` change (a coarser grid shifts which corners fall in the band).

## Out of scope
- The §3.2 multi-threaded CUDA framework (not feasible for Unity main-thread GPU dispatch).
- Haptics / force feedback.
- Re-deriving `ks` from SI liver E (the paper's soft-unit value is used directly).
