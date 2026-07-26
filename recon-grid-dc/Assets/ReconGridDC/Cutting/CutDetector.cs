// CutDetector.cs — Stage 2B orchestrator (design §5.2 / §5.3; paper §2.1.3)
// Binds CuttingTool's swept-plane geometry to the GPU (Cutting.compute DetectCut),
// dispatches it once per frame over all interior grid edges, then rolls Sprev/Eprev.
//
// Per-frame flow:
//   1. CutDetector.Dispatch(tool) — call after physics, before DC.
//      a. Set uniforms: Sprev, Eprev, S, E, n_cut, D, valid, sweptAABB.
//      b. Bind buffers: CornerPos, GridEdges, VoxelCutMask, CutPoint, CutPointCounter.
//      c. Dispatch DetectCut over ceil(GridEdgeCount / 64) groups.
//      Note: Sprev/Eprev rolling is handled by CuttingTool.Advance() — CutDetector is pure binding.
//
// Persistence model (review R2 #1): BOTH VoxelCutMask and CutPoint accumulate across
// frames. Cut bits persist (cumulative cuts); cut points are emitted ONCE per edge on its
// uncut→cut transition (DetectCut gates emission on the InterlockedOr old value), so the
// CutPoint buffer and CutPointCounter are NOT reset per frame — re-detecting an already-cut
// edge is idempotent. CutPointCounter is zeroed only at init (ReconBuffers.Upload). A future
// re-init (new mesh / clear-cuts) is the only time these are reset.
//
// Traceability: design §5.2 / §5.3 / §10; paper §2.1.3.

using UnityEngine;
using ReconGridDC.Recon;

namespace ReconGridDC.Cutting
{
    public sealed class CutDetector
    {
        readonly ComputeShader _cs;
        readonly int           _kDetectCut;
        readonly int           _kSeverLinks;   // Stage 2D-1 physical severing (paper §2.1.2)

        // ── Construction ──────────────────────────────────────────────────────────────────
        /// <param name="cs">Cutting.compute loaded by the caller (e.g. via AssetDatabase).</param>
        public CutDetector(ComputeShader cs)
        {
            _cs          = cs;
            _kDetectCut  = cs.FindKernel("DetectCut");
            _kSeverLinks = cs.FindKernel("SeverLinks");
        }

        // ── Per-frame dispatch ─────────────────────────────────────────────────────────────
        /// <summary>
        /// Dispatch DetectCut for all interior grid edges using the current CuttingTool state.
        /// Cut points are emitted once per edge (first-cut transition); the caller must NOT
        /// reset CutPointCounter between frames — the cut set is cumulative (review R2 #1).
        /// </summary>
        /// <param name="tool">Current blade state (Advance() already called this frame).</param>
        /// <param name="rb">ReconBuffers (CornerPos, GridEdges, VoxelCutMask, CutPoint, Counter).</param>
        /// <param name="dims">Voxel grid dimensions (x,y,z).</param>
        /// <param name="voxelL">Voxel side length L (rest lattice spacing) — DetectCut tests the REST edge.</param>
        /// <param name="gridOrigin">Rest lattice origin (rest corner pos = gridOrigin + coord*voxelL).</param>
        public void Dispatch(CuttingTool tool, ReconBuffers rb, Vector3Int dims, float voxelL, Vector3 gridOrigin)
        {
            // ── Uniforms ─────────────────────────────────────────────────────────────────
            _cs.SetInts("_Dims", dims.x, dims.y, dims.z);
            _cs.SetInt ("_GridEdgeCount", rb.GridEdgeCount);
            _cs.SetInt ("_CutPointCapacity", rb.CutPointCapacity); // append OOB guard — review R2 #2

            // Swept-plane triangles (paper Fig 2.8; gap C22)
            // T1 = (Sprev, Eprev, E_cur), T2 = (Sprev, E_cur, S_cur)
            _cs.SetVector("_T1V0", tool.T1V0);
            _cs.SetVector("_T1V1", tool.T1V1);
            _cs.SetVector("_T1V2", tool.T1V2);
            _cs.SetVector("_T2V0", tool.T2V0);
            _cs.SetVector("_T2V1", tool.T2V1);
            _cs.SetVector("_T2V2", tool.T2V2);

            // Cut-plane normal + thickness (PAPER-SILENT; design §5.3)
            _cs.SetVector("_NCut",  tool.NCut);
            _cs.SetFloat ("_D",     tool.D);
            _cs.SetInt   ("_ToolValid", tool.Valid ? 1 : 0);

            // Rest lattice (workflow w87ghv7f6): DetectCut M-T-tests the swept blade against the UNDEFORMED
            // edge (rest pos = _GridOrigin + coord*_VoxelL), so the cut targets a fixed material plane and
            // is immune to the gravity-vs-sweep deformation race. The cut POINT is still emitted on the
            // deformed edge (surface tracks the tissue). Declared deviation — see Cutting.compute header.
            _cs.SetFloat ("_VoxelL",     voxelL);
            _cs.SetVector("_GridOrigin", gridOrigin);

            // Swept AABB broadphase (PAPER-SILENT; design §10). DetectCut culls on the REST edge, which is
            // static and always straddles the blade plane for the cut layer, so NO inflation is needed.
            Vector3 aabbMin = Vector3.Min(Vector3.Min(tool.T1V0, tool.T1V1),
                                          Vector3.Min(tool.T1V2, tool.T2V2));
            Vector3 aabbMax = Vector3.Max(Vector3.Max(tool.T1V0, tool.T1V1),
                                          Vector3.Max(tool.T1V2, tool.T2V2));
            _cs.SetVector("_SweptAABBMin", aabbMin);
            _cs.SetVector("_SweptAABBMax", aabbMax);

            // ── Bind buffers ──────────────────────────────────────────────────────────────
            _cs.SetBuffer(_kDetectCut, "_CornerPos",       rb.CornerPos);
            _cs.SetBuffer(_kDetectCut, "_GridEdges",       rb.GridEdges);
            _cs.SetBuffer(_kDetectCut, "_VoxelCutMask",   rb.VoxelCutMask);
            // 2C-fix (plan §7 RC2): DetectCut now gates cut-bit propagation + edge-granularity emit
            // on per-voxel occupancy — bind the gate so the kernel's _VoxelOccupied resolves.
            _cs.SetBuffer(_kDetectCut, "_VoxelOccupied",   rb.VoxelOccupied);
            _cs.SetBuffer(_kDetectCut, "_CutPoint",       rb.CutPoint);
            _cs.SetBuffer(_kDetectCut, "_CutPointCounter",rb.CutPointCounter);
            // Stage 2D-2 (paper §2.1.2): EmitCutPoint reads the per-particle rotation R to record the cut
            // point in the owner's MATERIAL frame (R_owner^T·offset). _ParticleRot is the persisted frame
            // from ComputeParticleRot — frame N-1's R (M1 ordering); identity at frame 0 ⇒ no regression.
            _cs.SetBuffer(_kDetectCut, "_ParticleRot",     rb.ParticleRot);

            // ── Dispatch ─────────────────────────────────────────────────────────────────
            // design §5.2: one thread per GridEdge; group size = 64.
            int groups = Mathf.CeilToInt(rb.GridEdgeCount / 64f);
            if (groups < 1) groups = 1;
            _cs.Dispatch(_kDetectCut, groups, 1, 1);

            // ── Stage 2D-1: physical severing (paper §2.1.2 line 367) ─────────────────────
            // AFTER DetectCut (so EdgeIsCut sees this frame's cut bits). One thread per grid edge;
            // SeverLinks is the SOLE writer of _NbrIdx / _BendPairs (Physics.compute reads them SRV).
            // Does NOT need _VoxelOccupied (EdgeIsCut reads only _VoxelCutMask). Idempotent each frame.
            _cs.SetBuffer(_kSeverLinks, "_GridEdges",    rb.GridEdges);
            _cs.SetBuffer(_kSeverLinks, "_VoxelCutMask", rb.VoxelCutMask);
            _cs.SetBuffer(_kSeverLinks, "_NbrIdx",       rb.NbrIdx);
            _cs.SetBuffer(_kSeverLinks, "_BendPairs",    rb.BendPairs);
            _cs.Dispatch(_kSeverLinks, groups, 1, 1);
        }
    }
}
