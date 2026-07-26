// physics.h — host-callable API for the tissue solver (Stage 2).
// Defined in physics.cu (CUDA), called from src/plugin_api.cpp (host C++).
//
// SOLVER = XPBD small-steps since migration step 4 (design rev-B §9-(4), commit a170233): physics_init
// binds the cuda/xpbd/ module to the SHARED buffers (xpbd_bind) and physics_step runs
// xpbd_begin_frame + N x { xpbd_substep; cut_ribbon_tick }. The mass-spring MODEL (paper SS2.1
// topology, rest lengths, bending pairs, severing flags) is UNCHANGED — only the integrator moved
// from the adaptive Dormand-Prince RK45 (Eq17-21, now RETIRED; its kernels remain in physics.cu but
// are no longer dispatched) to the unconditionally-stable XPBD projection. ComputeParticleRot
// (paper SS2.1.2 / Berndt polar frame) is integrator-agnostic and unchanged.
//
// The solver deforms the SHARED corner-position buffer (dc_recon's _CornerPos, via recon_corner_pos())
// in place + its own velocity buffer; the caller then calls recon_rebuild() to re-mesh.

#pragma once

// Scalar physics description + paper SS3.4 parameters. Field order MUST match the C#
// [StructLayout(Sequential)] PhysicsInitDesc in LiverCudaManager.cs (ABI frozen across the XPBD swap).
struct PhysicsInitDesc
{
    int   cornerCount;    // = recon_corner_count()
    int   bendPairCount;  // 15 * cornerCount
    float ks, cs, kb, cb; // paper SS3.4: 7.5e4, 0.92, 2e4, 0.9 (cs/cb = XPBD betaS/betaB, DIRECTLY)
    float hInit;          // DEAD under XPBD (was the RK45 adaptive seed; kept as an ABI placeholder)
    float hMax;           // DEAD under XPBD (was the RK45 substep cap;  kept as an ABI placeholder)
    int   maxSubsteps;    // REPURPOSED (rev-B §9-(4)): the FIXED XPBD substep count N (h = dt/N) —
                          // the exact semantics-bearing count, not a budget
    float alpha;          // PAPER-SILENT global mass-proportional (Rayleigh) damping 1/s (0 = paper-exact)
};

// Allocate + upload physics buffers (vel zero-init, forceInt/kslope/yTrial zero, particleRot=identity).
// Shares the corner-position buffer with dc_recon (recon_corner_pos()); cornerCount must match
// recon_corner_count(). Returns 0 on success; negative on error.
//   mass      : float[cornerCount]
//   pinned    : int[cornerCount]   (1 = pinned)
//   active    : int[cornerCount]   (1 = active tissue corner; 0 = frozen, paper SS2.1.4)
//   extForce  : float[3*cornerCount]  (gravity etc.)
//   nbrIdx    : int[6*cornerCount]    (-1 = no neighbor)
//   restNbr   : float[3*6*cornerCount]
//   bendPairs : BendPairGpu[bendPairCount]  (32-byte entries; passed as void*)
int  physics_init(const PhysicsInitDesc* desc,
                  const float* mass,
                  const int*   pinned,
                  const int*   active,
                  const float* extForce,
                  const int*   nbrIdx,
                  const float* restNbr,
                  const void*  bendPairs);

// Advance one outer fixed dt by N fixed XPBD substeps (rev-B §9-(4): xpbd_begin_frame + N x
// { xpbd_substep; cut_ribbon_tick }), deforming the shared cornerPos + vel in place. tickErr from the
// CCD cut tick is surfaced after the loop (same contract as the retired RK45 body).
// (ComputeParticleRot is SEPARATE — physics_compute_rotations — so Stage 3 can run it after cut+sever.)
// Returns 0 on success; XPBD_ERR_* or negative CUDA error otherwise.
int  physics_step(float dt);

// Run ComputeParticleRot once (paper SS2.1.2 / Berndt polar frame). Stage 3: call AFTER cut detect+sever
// and BEFORE the rebuild. Returns 0 on success.
int  physics_compute_rotations();

// Stage 3 shared accessors (read by cut.cu).
void* physics_particle_rot();   // RotGpu*
int*  physics_nbr_idx();        // int[6*cornerCount]
void* physics_bend_pairs();     // BendPairGpu*
int*  physics_active();         // int[cornerCount] — v4.1 P1b: cut.cu freezes fully-severed debris
int*  physics_pinned();         // int[cornerCount]
void* physics_vel();            // float3[cornerCount] velocity buffer
int   physics_readback_nbr_idx(int* hostOut);  // diagnostic: copy live (post-sever) nbrIdx[6*cc] to host

// Free all physics device buffers.
void physics_shutdown();
