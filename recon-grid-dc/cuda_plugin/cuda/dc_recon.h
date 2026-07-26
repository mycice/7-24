// dc_recon.h  - host-callable API for the static DC reconstruction (Stage 1).
// Defined in dc_recon.cu (CUDA), called from src/plugin_api.cpp (host C++).
//
// Stage 1 maps the paper Fig 3.1 INITIALIZATION boxes:
//   CPU (C#, reused): Level set -> Background Grid -> Intersection Points -> label corner state.
//   GPU (here):       Calculate Feature Points (QEF) -> Construct Triangles + normals.
//   CPU barrier:      cudaDeviceSynchronize ("Wait for GPU Computation to Complete") inside recon_init.
// Copy-based render: recon_get_surface returns the resolved mesh to C# for a Unity Mesh upload.

#pragma once
#include <vector_types.h>   // float3 (host-includable CUDA type header)

// Scalar grid description marshalled from the C#-built BackgroundGrid. Field order MUST match the
// C# [StructLayout(Sequential)] ReconInitDesc in LiverCudaManager.cs.
struct ReconInitDesc
{
    int   voxelCount;        // dims.x*dims.y*dims.z
    int   cornerCount;       // (dims.x+1)*(dims.y+1)*(dims.z+1)
    int   isectCount;        // number of Hermite intersections (g.isect.Length)
    int   surfaceEdgeCount;  // number of surface edges (g.surfaceEdges.Length)
    int   triCapacity;       // _Tri holds 3*triCapacity ints
    int   dimsX, dimsY, dimsZ;
    float L;                 // voxel side length
    int   qefIters;          // QEF gradient-ascent iterations
};

// Allocate device buffers, upload the marshalled grid, run the static DC pipeline once, then
// cudaDeviceSynchronize. Returns 0 on success; negative CUDA error code otherwise.
//   cornerPos        : float[3*cornerCount]      (flattened float3 per corner; rest positions)
//   voxelCorner      : int[8*voxelCount]         (global corner id per voxel-corner)
//   voxelIsectOffset : int[voxelCount]           (per-voxel isect-list start)
//   voxelIsectCount  : int[voxelCount]           (per-voxel isect count)
//   isect            : IsectGpu[isectCount]      (40-byte entries; passed as void*)
//   surfaceEdges     : GridEdgeGpu[surfaceEdgeCount] (32-byte entries; passed as void*)
int  recon_init(const ReconInitDesc* desc,
                const float* cornerPos,
                const int*   voxelCorner,
                const int*   voxelIsectOffset,
                const int*   voxelIsectCount,
                const void*  isect,
                const void*  surfaceEdges);

// Report the surface counts for sizing the C# readback buffers.
//   vertexCount = voxelCount (the mesh is indexed by voxelId; _Tri tags are voxelIds in the static path)
//   indexCount  = _TriCounter[0] clamped to 3*triCapacity (= 3 * triangle count)
// Returns 0 on success; negative on error.
int  recon_get_counts(int* vertexCount, int* indexCount);

// Copy the resolved surface back to host (copy-based render path):
//   outPos3 : float[3*vertexCount]  per-voxel feature point xyz (from _VoxelExternalFP)
//   outNrm3 : float[3*vertexCount]  per-voxel normalized normal (from _ExtNormalF)
//   outIdx  : int[indexCount]       triangle indices (from _Tri; static-path tags == voxelId)
// Caller sizes the arrays via recon_get_counts. Returns 0 on success; negative on error.
int  recon_get_surface(float* outPos3, float* outNrm3, int* outIdx);

// Liver-render design §4.4 (2026-07-02-liver-surface-render-design.md): copy the per-index render
// aux channel  - outAux4 : float[4*indexCount], each entry = (rest anchor xyz, wall flag). Same
// clamped-count contract as recon_get_surface. UV1 for the rest-position triplanar shader.
int  recon_get_surface_aux(float* outAux4);

// Free all device buffers.
void recon_shutdown();

// ── Stage 2 shared accessors ────────────────────────────────────────────────────────────────
// The physics module (physics.cu) deforms the grid IN PLACE: it writes the SAME corner-position
// buffer the DC pipeline reads, so the deformed surface re-meshes from the moved corners.
float3* recon_corner_pos();    // device pointer to _CornerPos (the XPBD solve projects it in place)
int     recon_corner_count();  // cornerCount (for physics_init validation)
void    recon_dims(int* x, int* y, int* z);  // grid dims (xpbd_bind carries them for the step-7 tet builder)
int     recon_readback_corner_pos(float* hostOut);  // diagnostic: copy live cornerPos[3*cc] to host
// Stage 3 coupling: replace the current deformed lattice positions from the host. This updates
// only the existing shared corner-position buffer; reconstruction and cutting remain unchanged.
int     recon_upload_corner_pos(const float* hostPos);

// Re-run the DC pipeline (RebuildIsectWorld -> FeaturePoints -> [cut-FP chain] -> Stitch -> [cut tris]
// -> normals -> expand) on the CURRENT (deformed) cornerPos + cudaDeviceSynchronize. Called each frame
// after physics + cut + rotations. Returns 0 on success; negative CUDA error code otherwise.
int     recon_rebuild();

// ── Stage 3 shared accessors (read by cut.cu) + the cut-buffer handover (set by cut_init) ─────
void*         recon_isect();              // IsectGpu*
float3*       recon_isect_world();
int*          recon_voxel_isect_offset();
int*          recon_voxel_isect_count();
int*          recon_voxel_corner();   // 8 corner ids per voxel (for the cut-FP Eq5 box-SDF clamp)
int*          recon_tri();
unsigned int* recon_tri_counter();
float3*       recon_expand_pos();    // Surface rendering: device non-indexed soup (3 world verts per triangle)
float4*       recon_expand_aux();    // Surface rendering: per-vertex aux (.w = wall flag; skip cut-wall triangles)
float3*       recon_expand_nrm();    // Surface rendering: per-vertex outward normal (sign the face normal  - winding guard)
// Hand cut.cu's cut buffers to dc_recon so the cut-aware Stitch / Normals / Expand resolve cut FPs.
// Pass nullptrs to disable (external-only). float4/float3 from vector_types.h.
// Stage 7 (paper-exact): ONE FP per (voxel,component)  - Table 3's External/Internal are positional
// labels of this single entity; skin and wall quads SHARE it (margin watertight by construction).
void recon_set_cut_buffers(unsigned int* voxelCutMask, unsigned int* voxelOccupied, void* conn4096,
                           float4* cutFP, float3* cutFPNormalF, int* cutFPNormal);

// QEF iteration count from the init desc (cut.cu's per-component skin QEF mirrors k_FeaturePoints).
int recon_qef_iters();

// [DEBUG-CUT] (v4.1 P3, cross-TU hook  - d_tri/FetchPos/voxelExternalFP are private to dc_recon):
// scan the emitted triangles into dbgDev: dbg[6]+= bit31 refs whose cutFP.w<0.5 (dangling ->
// spike-to-origin filaments); tris with max edge > 4L classified dbg[7]=all-cut(wall) /
// dbg[8]=mixed(stitch) / dbg[9]=skin; atomicMax dbg[10]=max edge length (float-as-uint).
// Called by cut_get_debug after the rebuild. Returns 0 on success.
int recon_debug_scan_tris(unsigned int* dbgDev, float voxelL);
