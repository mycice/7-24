// plugin_api.cpp — exported C ABI for the Unity native plugin (Stage 1).
// Thin forwarding layer over the CUDA recon API (cuda/dc_recon.cu). Host C++ only; all kernel
// launches live in the .cu. Mirrors the design spec's P/Invoke surface (Init / GetSurface / Shutdown).
//
// Stage 0's hard-coded-triangle smoke test (plugin.cu) is superseded — the toolchain it validated
// (DLL build/load/P-Invoke/CUDA/render) is exercised by the real recon path below. The Stage 0
// source remains in git history (commit dcaf010).

#include "dc_recon.h"
#include "physics.h"
#include "cut.h"

#define LCS_API extern "C" __declspec(dllexport)

// Allocate device buffers, upload the C#-built grid, run the static DC pipeline, sync.
// Returns 0 on success; negative CUDA error code otherwise.
LCS_API int LCS_Init(const ReconInitDesc* desc,
                     const float* cornerPos,
                     const int*   voxelCorner,
                     const int*   voxelIsectOffset,
                     const int*   voxelIsectCount,
                     const void*  isect,
                     const void*  surfaceEdges)
{
    return recon_init(desc, cornerPos, voxelCorner, voxelIsectOffset, voxelIsectCount, isect, surfaceEdges);
}

// vertexCount = voxelCount; indexCount = 3 * triangle count (clamped to capacity).
LCS_API int LCS_GetSurfaceCounts(int* vertexCount, int* indexCount)
{
    return recon_get_counts(vertexCount, indexCount);
}

// Copy the resolved surface (positions, normals, indices) back to host for a Unity Mesh upload.
LCS_API int LCS_GetSurface(float* outPos3, float* outNrm3, int* outIdx)
{
    return recon_get_surface(outPos3, outNrm3, outIdx);
}

// Liver-render design §4.4: per-index render aux channel (rest anchor xyz + wall flag) — UV1 for
// the rest-position triplanar liver shader. Same clamped-count contract as LCS_GetSurface.
LCS_API int LCS_GetSurfaceAux(float* outAux4)
{
    return recon_get_surface_aux(outAux4);
}

// ── Stage 2: mass-spring RK45 physics ────────────────────────────────────────────────────────
// Allocate + upload the physics buffers. Call AFTER LCS_Init (shares dc_recon's corner grid).
LCS_API int LCS_InitPhysics(const PhysicsInitDesc* desc,
                            const float* mass,
                            const int*   pinned,
                            const int*   active,
                            const float* extForce,
                            const int*   nbrIdx,
                            const float* restNbr,
                            const void*  bendPairs)
{
    return physics_init(desc, mass, pinned, active, extForce, nbrIdx, restNbr, bendPairs);
}

// Advance one outer dt: RK45 substeps deform the shared corner grid (each accepted substep also runs
// the Stage-6 CCD cut tick inside physics_step), then update the current particle frames used by the
// SS2.1.2 cut-point record/reconstruct. NO re-mesh here — rod cut sub-steps run between LCS_Step and
// LCS_Finalize (paper Fig 3.1: physics -> cut -> reconstruct).
LCS_API int LCS_Step(float dt)
{
    int e = physics_step(dt);
    if (e != 0) return e;
    return physics_compute_rotations(); // pre-cut frames for DetectCut / EmitCutPoint local coordinates
}

// Stage 3 coupling uploads externally driven grid positions before the cutter runs. Keeping this
// as a narrow ABI preserves the existing CUDA cut and Dual Contouring implementations.
LCS_API int LCS_SetCornerPos(const float* positions)
{
    return recon_upload_corner_pos(positions);
}

// The cutter records local coordinates in the current particle frames, so externally uploaded
// positions need their frames refreshed before LCS_DetectCut.
LCS_API int LCS_ComputeRotations()
{
    return physics_compute_rotations();
}

LCS_API int LCS_SetCutRestMetric(const float* alignedRestPositions,
                                float voxelSizeX, float voxelSizeY, float voxelSizeZ)
{
    return cut_set_rest_metric(alignedRestPositions, voxelSizeX, voxelSizeY, voxelSizeZ);
}

LCS_API void LCS_ClearCutRestMetric()
{
    cut_clear_rest_metric();
}

// ── Stage 3: cutting ───────────────────────────────────────────────────────────────────────────
// Allocate + upload the cut buffers (Conn4096 LUT, gridEdges, occupancy). Call AFTER LCS_Init + LCS_InitPhysics.
LCS_API int LCS_InitCut(const CutInitDesc* desc, const void* conn4096, const void* gridEdges, const int* voxelOccupied,
                        const int* cornerInside)
{
    return cut_init(desc, conn4096, gridEdges, voxelOccupied, cornerInside);
}

// Set the current swept-plane tool geometry (call once per rod sub-step before LCS_DetectCut).
LCS_API void LCS_SetTool(const CutToolDesc* tool)
{
    cut_set_tool(tool);
}

// DetectCut (Stage 6, paper SS2.1.3 Eq6-8 verbatim: swept world cutting-plane triangles vs the
// CURRENT DEFORMED grid edges) + SeverLinks. Call between LCS_Step and LCS_Finalize.
LCS_API int LCS_DetectCut()
{
    return cut_detect();
}

// Recompute particle frames after severing links, then run the cut-aware DC rebuild
// (cut-FP chain + cut walls + normals + expand).
LCS_API int LCS_Finalize()
{
    int e = physics_compute_rotations();
    if (e != 0) return e;
    return recon_rebuild();
}

// ── Diagnostics (Phase-1 evidence: split? gravity moving corners?) ───────────────────────────
// Copy the LIVE post-sever nbrIdx[6*cornerCount] to host (for a connected-components check).
LCS_API int LCS_GetNbrIdx(int* out) { return physics_readback_nbr_idx(out); }
// Copy the LIVE deformed cornerPos[3*cornerCount] floats to host (for a gravity-motion check).
LCS_API int LCS_GetCornerPos(float* out) { return recon_readback_corner_pos(out); }
// [DEBUG-CUT] (v4.1 P3): 16 per-frame debug counters + 2 raw atomic counters + runtime cutFPInterp.
// Call after LCS_Finalize each frame (frame already synced); layout documented in cuda/cut.h.
LCS_API int LCS_GetCutDebug(unsigned int* out18, float* alphaOut) { return cut_get_debug(out18, alphaOut); }

// Free all device buffers.
LCS_API void LCS_Shutdown()
{
    cut_shutdown();
    physics_shutdown();
    recon_shutdown();
}
