# Collision / Contact / Grasp Redesign — soft tissue vs rigid surgical tool

> **Status: DESIGN (design-only, no code yet — per user request 2026-07-07).** Supersedes the step-8
> contact design in `2026-07-05-xpbd-combination-round3-design.md` §4[Contact]. The grasp core-pin/slab
> intent (§5) is kept but re-gated and softened. Produced from a paper re-read + 4-angle multi-agent
> analysis + adversarial design review (workflow wf_bf60c85c).

## The bug being fixed

In the real Unity app the step-8 native contact+grasp is catastrophically unstable: the grasper starts
**inside** the liver at init, so the moment the tool moves the tissue **sticks and follows it** and
cannot be shaken off; the framerate **collapses**; the mesh **shatters** into free fragments; HUD shows
`|F|=2460 N` during contact. None of this reproduced in the headless step-8 self-test because that test
ran at `L≈0.2` (real `L≈0.4`), wrote `XpbdToolState` directly, and **never ran the paper's Finger-Proxy**
or the real `GrasperRig.JawSlabWorld`.

## Root-cause summary (ranked)

The paper (Li/Zhou/Zhou, CMPB 273 (2026) 109123, §2.3) already solves tool↔tissue collision with a
**god-object / Finger-Proxy**: a hidden PHYSICAL tool (may penetrate) drives a displayed PROXY tool that
Algorithm 1 iteratively marches back onto the surface (`P_t += a·v_ms`, a=0.1), and the tissue is deformed
by a **bounded spring** reaction `F = Kp·ΔP − Dp·v` (Eq 27) scattered onto nearby triangle particles.
This is indifferent to a t=0 embedded tool and cannot explode. Step 8 added a **worse, redundant**
mechanism on top of it:

1. **[TRIGGER] Hard non-penetration on the PHYSICAL tool, armed from frame 1.**
   `k_SolveJawShaftContact` (`xpbd_solver.cu:511`) ejects every enclosed corner out of the physical shaft
   capsule + jaw box every substep, gated on `g_toolArmed` which latches true on the first
   `xpbd_set_tool` (`:1200`) — jaws open, no contact test. It projects the **total** `C=dist−shaftR`
   (`:532-539`) at near-rigid `contactAlpha=1e-7`, so an embedded corner is thrown the full penetration
   depth in one substep. No proxy, no god-object — it acts on the penetrating physical geometry directly.
2. **[AMPLIFIER] Tear law → topology catastrophe.** Ejections over-stretch edges past `τ·L=1.6u`
   (`tearStretchRatio=4.0`, `LiverCudaManager.cs:223`); `k_TearOverstretch` severs them → shatter → the DC
   surface reconstruction rebuilds huge non-manifold shells every frame → framerate collapse. A
   positive-feedback loop the paper's bounded soft reaction can never enter.
3. **[AMPLIFIER] ω=1.0, no under-relaxation.** Single global `xp.omega=1.0` (`physics.cu:549`; comment
   `:542-543` admits §3.5 was never applied); both step-8 kernels use it (`:1407,1411`). Full projection,
   zero relaxation inside the K=2 loop → overshoot → rebound → jitter.
4. **[AMPLIFIER] Retired-but-live position dent.** `k_ApplyContactDisp` (`haptics.cu:335`, dispatched
   `:693-694`) still writes `cornerPos += clamp(disp); vel*=0.15` every haptic step. Design §7 ordered it
   deleted at step 8, but haptics.cu was kept byte-unchanged the whole migration — so a **second** un-
   relaxed one-shot position write races the constraint solve, and helps cement the "cannot be shaken off".
5. **[CONDITION] Init pose seeds the tool inside.** Init is `gridCenter+(−3L,+4L,0)` axis +Z
   (`LiverCudaManager.cs:544`, comment "start above/outside") — but that offset is scaled to the **voxel
   size L**, ~1.6u above center in a **~19u** liver → deep inside. Nothing verifies the **jaw slab** (what
   captures tissue) clears the surface. The degenerate jaw box (`_scale≈0.047`) is a quarter-voxel knife.

**Corrected facts** (verified against source, vs the initial brief): shipping shaft radius is
`hapticSkinRadius=0.2u` not 0.9u (`:702`, `useMesh=true` default); `hapticReactionMax=600` not 50 (`:105`);
the core-pin is a **kinematic weld** `pos=prevPos=target; vel=0` (`:502`) with no ω knob; the grasp
**state machine itself is correct** (`grasp01=0` at frame 1 → nothing captures at init) — the bug was
*contact* running on `g_toolArmed`, independent of the grasp gesture.

**The core disease:** three uncoordinated tissue-push channels with no arbiter — the legitimate Eq-27
spring, the live position dent, and the hard constraint — two of which are one-shot position writes racing
each other with no under-relaxation and no incremental accumulation.

## Chosen architecture — Option A: one force path, four clean concerns

Route **all** tool→tissue contact through the paper's surface-resting proxy; the tissue is deformed by the
Eq-27 reaction force **only**. Remove the step-8 hard tissue-side constraint. Rebuild grasp as a
gesture-armed *compliant* attachment. (Options B/C — keep a hard tissue-side non-penetration constraint —
are rejected: a hard constraint against a tool the paper explicitly lets sink to any depth is
over-constrained with no feasible t=0 rest state = exactly the frame-1 catastrophe. Its one unique
benefit, anti-tunnelling, is recovered by a narrow CCD gate below.)

| Concern | Owner after this design | Disposition |
|---|---|---|
| Haptic force feedback (device reaction) | proxy solve — Alg 1 + Eq 27 `F=Kp·ΔP−Dp·v` (haptics.cu) | **KEEP** unchanged |
| Tissue deformation (visible dent) | force-only: `hapticForce` atomic-add → `k_predict` body force | **KEEP + TRIM** |
| Non-penetration (keep tool/tissue apart) | the proxy keeps the *displayed* tool on the surface; tissue never hard-projected; fast approach → narrow CCD gate | **RETIRE** the step-8 constraint |
| Grasp (pinch/hold/drag/release) | gesture-armed compliant breakable core-pin + under-relaxed slab, on the jaw slab only | **REPLACE** |

**Explicit edits (KEEP / RETIRE / REPLACE):**
- **KEEP** the proxy + Eq-27 force (`haptics.cu` proxy logic) — indifferent to t=0 embedding by construction.
- **RETIRE (unambiguous)** the tissue-side position dent in full: delete `k_ApplyContactDisp` position
  write + `vel*=0.15` (`haptics.cu:335-351`) **and** the `k_ScatterContact` contactDisp accumulation
  (`:326-329`); keep only the `hapticForce` atomic-add (`:311-315`). The dent now emerges from the force
  settling over frames — never a direct write. This makes "sole tissue-push path" literally true.
- **RETIRE** `k_SolveJawShaftContact` as a tissue non-penetration authority (`xpbd_solver.cu:511`); its
  anti-tunnelling benefit is re-supplied by the CCD gate under strict limits.
- **REPLACE** the grasp: re-gate arming off `g_toolArmed` onto the actual close gesture; capture on the
  jaw slab (never the shaft); compliant breakable core-pin.

## Initial-interference handling

With the hard constraint gone, the paper's method is indifferent to t=0 embedding (Alg 1 marches the proxy
out; the spring is bounded; force-only push from zero deviation is a non-event). Two residual actions:

- **A · Clear the whole tool ASSEMBLY at the closed-jaw pose.** The init offset (replacing the nominal
  `−3L,+4L` at `LiverCudaManager.cs:544`) must satisfy `SDF > skin` for the **union** of the shaft capsule
  AND the closed-pose jaw slab box (from `GrasperRig.JawSlabWorld`) before `Start` completes — a shaft-only
  clearance can still leave the jaw straddling the surface. Offset along the approach axis; log both
  clearances.
- **B · No tissue-side rest-offset machinery** is needed — nothing is hard-projected against the tissue, so
  there is no infeasible t=0 rest state to resolve. The proxy's own outward march is the rest-
  interpenetration resolver (for the displayed tool).

## Stability measures (sized, not asserted)

- **A · Damp the position↔force loop explicitly.** With tissue push now force-only, the loop
  (force dents tissue → proxy tracks the dented surface → ΔP shrinks → force drops → tissue rebounds) is a
  one-frame-delayed closed loop; `reactionMax` bounds amplitude but an underdamped bounded loop *buzzes*.
  Size the Eq-27 damping `Dp ≥ 2·√(Kp_eff·m)`, `m≈ρ·L³`, so it is over-damped. Verify by logging `|F|`
  over a static deep-press hold and asserting convergence (no limit cycle).
- **B · Re-tune the force cap at real scale.** `reactionMax=600` (not 50). With the position dent deleted,
  a capped body force over a triangle footprint dents *softer* than the old direct write — real risk of the
  mirror-image failure (slow press → "faint push / passes through"). Re-tune `reactionMax`/`contactBeta` at
  `L≈0.4`; assert a slow steady press gives a visible, stable dent.
- **C · ω, corrected in scope.** The core-pin is a kinematic weld — under-relaxing it is a category error;
  it must first *become* a compliant XPBD constraint (below). Expose a **separate** ω for the soft
  contact/grasp set (≈0.25), distinct from `xp.omega=1.0` which stays on the structural+tet set. (Mixed ω
  is the *stabilizing* direction here — we under-relax the soft block, keep the stiff block at 1; residual
  risk is under-convergence/mushy grip, caught by the drift assertion, not divergence.)
- **D · Incremental correction, not total.** The grasp/CCD constraints must cap the total displacement
  applied to a corner within a frame at `0.1–0.25·L`, so no corner is relocated by more than a fraction of
  a voxel per frame regardless of nominal penetration depth — structurally decoupling correction magnitude
  from the tear threshold.
- **E · Scale-transfer rule.** Fix `ρ` and `mass=ρ·L³` at real L *first*; XPBD compliance and `τ·L` then
  follow. (The naïve "rescale α by 1/L²" is dimensionally wrong.)

### CCD anti-tunnelling gate (its own sub-design — build to all four rules or don't ship)

Removing the hard constraint means only the soft Eq-27 force keeps tissue out of the tool, which cannot
stop a fast/stiff-backed corner passing through. A narrow speculative gate recovers anti-tunnelling:

| # | Rule | Why |
|---|---|---|
| a | Fire **only** when the corner's approach velocity toward the tool surface exceeds `v_thresh`. | A static-embedded tool at t=0 → zero correction; a slow press stays on the pure force path. |
| b | Cap the per-substep correction at `0.1·L` regardless of depth. | Incremental not total — never over-stretches to the tear threshold; no full-depth one-shot. |
| c | Run under its own `ω_ccd ≤ 0.25`. | Never a rigid snap; degrades to a soft "ghost bump". |
| d | Early-out on `wc≤0`. | Never fights the compliant core-pin or a pinned neighbour. |

Velocity-gated + amplitude-capped + under-relaxed is exactly what makes this **not** a re-import of the
hard constraint: on every slow or static interaction it is silent and the force path owns the interaction.

## Grasp layer

The grasp state machine (`xpbd_set_tool:1214`) is already correct (arms on `grasp01>graspThreshold`,
`grasp01=0` at frame 1 → nothing captures at init). Three edits make the grasp itself stable:

- **A · Compliant, breakable core-pin (not a weld).** The "cannot be shaken off" is the kinematic weld
  `pos=prevPos=target; vel=0` (`:502`) teleporting each captured corner with infinite authority. Replace
  with a compliant XPBD distance-to-target constraint with finite compliance + a λ-based break threshold —
  a hard pull past threshold cleanly detaches the corner instead of dragging the whole liver.
- **B · Slab + Coulomb genuinely compliant.** The slab STICK branch `pos−=tangential` (`:608`) is a rigid
  in-plane snap; give it real compliance so under-relaxation relaxes a soft residual rather than crawling a
  rigid snap. Capture on the jaw slab only; widen the degenerate quarter-voxel jaw box; `/L`-scale the skins.
- **C · Re-gate coupling + capture (the framerate fix is TWO lanes).** Move the K=2 coupling gate off
  `g_toolArmed` (`:1382`, doubles solve cost from frame 1 forever) onto "actually in contact or grasping".
  Separately, the CAPTURE path does 4 `cudaMalloc` + 4 blocking D2H memcpy per rising edge (`:1221-1234`);
  under the keyboard ramp a `grasp01` chattering at the threshold triggers a per-frame malloc/sync storm —
  add hysteresis / a one-shot latch so capture fires exactly once per genuine close.

The ring tear-exemption (`cut.cu`) stays, so a pulled tent stretches without shattering.

## Verification plan (all stages at real L≈0.4 through the actual proxy path)

1. **Retire hard constraint + position dent** → no frame-1 explosion; tears stay 0 on touch; `|F|` smooth
   and saturating (never 2460 N); static deep-press hold → `|F|` converges, no limit cycle.
2. **Scale & force-cap audit at L≈0.4** → slow steady press gives a visible stable dent (not faint/pass-through).
3. **Clear the assembly at init** → at t=0 before input: `ΔP≈0`, `|F|≈0`, and for first N substeps
   `grasp_count==0 ∧ max|hapticForce|==0 ∧ tear_count==0`.
4. **Shake-off + armed-but-outside cost** → liver does not follow the tool; frame time with tool armed but
   outside proves K=2 is no longer a standing cost; frame-1 G-press from the seeded pose → `grasp_count==0`.
5. **Gesture-armed grasp lifecycle** → `grasp_count==0` until close; bounded slab capture on rising edge;
   release → 0; static tent-hold → grip drift below threshold.
6. **Pinch-hold-pull + fast plunge** → stable tent, ring stretches without shatter, hard pull breaks the pin
   cleanly, framerate flat; fast plunge caught by the CCD gate; zero CCD correction when static-embedded.

## Decisions for the user (4)

1. **Ship the CCD gate now, or defer?** Rec: **ship** (build to rules a–d) — real robustness for fast
   motion. Defer = less code, accepts occasional tunnelling on fast plunges.
2. **Init pose: auto-clearance SDF offset vs authored fixed pose?** Rec: **auto-clear** over shaft∪jaw —
   robust to mesh/scale changes.
3. **Core-pin: compliant-breakable vs stiff weld + explicit release?** Rec: **compliant-breakable** —
   physical, self-releases under hard pull.
4. **K=2 re-gate: grasp-only vs contact-or-grasp?** Rec: **grasp-only** — cheapest; force-only contact
   needs no coupling.
