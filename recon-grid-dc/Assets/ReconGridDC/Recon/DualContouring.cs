// DualContouring.cs — Task 9 Step 1 (+ Stage 2C cut-surface orchestration, plan §4)
// C# orchestrator: finds all Recon.compute kernels and (optionally) the Cutting.compute cut kernels
// and dispatches them per frame in the unified order:
//
//   set scalars → RebuildIsectWorld → DC_FeaturePoints
//     → [2C] ClearCutFP → LookupConnectivity → AccumulateCutFP → ComputeComponentFP
//     → reset TriCounter (single reset, head of geometry)
//     → DC_Stitch → [2C] BuildCutTriangles
//     → DC_ClearNormals → DC_NormalsScatter (tag-aware) → DC_NormalsNormalize
//     → [2C] DC_NormalizeCutFP → DC_WriteIndirectArgs
//
// ZERO-CUT byte-identical Stage-1 path (plan R-D / deliverable 8): when the Cutting.compute shader is
// NULL (SetCuttingShader never called), ALL 2C dispatches are skipped AND the tag-aware
// DC_NormalsScatter sees only external tags → output is identical to Stage 1A. The 2C cut buffers
// are still BOUND to DC_NormalsScatter so the kernel's _CutFP/_CutFPNormal declarations resolve, but
// they are never written when no cut kernels run.
//
// Binding name mapping (C# ReconBuffers field → HLSL buffer name):
//   CornerPos → _CornerPos, VoxelCorner → _VoxelCorner, VoxelIsectOffset → _VoxelIsectOffset,
//   VoxelIsectCount → _VoxelIsectCount, Isect → _Isect, IsectWorld → _IsectWorld,
//   VoxelExternalFP → _VoxelExternalFP, SurfaceEdges → _SurfaceEdges, Tri → _Tri,
//   TriCounter → _TriCounter, VoxelExternalFPNormal → _ExtNormalFixed,
//   VoxelExternalFPNormalF → _ExtNormalF, IndirectArgs → _IndirectArgs,
//   CutFP → _CutFP, CutFPAccumPos → _CutFPAccumPos, CutFPAccumCnt → _CutFPAccumCnt,
//   VoxelFPCount → _VoxelFPCount, CutFPNormal → _CutFPNormal, CutFPNormalF → _CutFPNormalF,
//   Conn4096 → _Conn4096, VoxelCutMask → _VoxelCutMask, GridEdges → _GridEdges,
//   CutPoint → _CutPoint, CutPointCounter → _CutPointCounter.

using UnityEngine;
using Unity.Mathematics;

namespace ReconGridDC.Recon
{
    public sealed class DualContouring
    {
        readonly ComputeShader cs;

        // Cached Recon.compute kernel ids
        readonly int kRebuild;  // RebuildIsectWorld  (Stage 1B-ii Task 1 — must run FIRST in Build)
        readonly int kFP;       // DC_FeaturePoints
        readonly int kStitch;   // DC_Stitch
        readonly int kClearN;   // DC_ClearNormals
        readonly int kScatterN; // DC_NormalsScatter (Stage 2C: tag-aware)
        readonly int kNormN;    // DC_NormalsNormalize
        readonly int kArgs;     // DC_WriteIndirectArgs

        // ── Stage 2C: Cutting.compute shader + cut kernel ids (null until SetCuttingShader) ──
        ComputeShader cut;       // Cutting.compute — null ⇒ Stage-1-only path (no cut dispatches)
        int kClearCutFP;
        int kLookupConn;
        int kAccumCutFP;
        int kComputeComponentFP; // 2C-fix (plan §7): unified per-component QEF (was kFinalizeCutFP)
        int kInterpCutFP;        // Stage 2D-2b: temporal feature-point interpolation (paper clean:1271-1276)
        int kBuildCutTris;
        int kNormalizeCutFP;
        bool cutReady;           // true once a non-null Cutting.compute is wired

        /// <summary>
        /// Finds all 7 required Recon.compute kernels. Throws if any is missing.
        /// Stage 2C cut kernels are wired separately via <see cref="SetCuttingShader"/>.
        /// </summary>
        public DualContouring(ComputeShader recon)
        {
            cs        = recon;
            kRebuild  = cs.FindKernel("RebuildIsectWorld");
            kFP       = cs.FindKernel("DC_FeaturePoints");
            kStitch   = cs.FindKernel("DC_Stitch");
            kClearN   = cs.FindKernel("DC_ClearNormals");
            kScatterN = cs.FindKernel("DC_NormalsScatter");
            kNormN    = cs.FindKernel("DC_NormalsNormalize");
            kArgs     = cs.FindKernel("DC_WriteIndirectArgs");
        }

        /// <summary>
        /// Convenience 2-arg ctor: Recon + Cutting wired in one call (cutting may be null).
        /// </summary>
        public DualContouring(ComputeShader recon, ComputeShader cutting) : this(recon)
        {
            SetCuttingShader(cutting);
        }

        /// <summary>
        /// Stage 2C: wire the Cutting.compute shader so the unified pass dispatches the cut kernels.
        /// Pass null to disable cutting (Stage-1-only path stays byte-identical). Caching the kernel
        /// handles here keeps Build() allocation-free.
        /// </summary>
        public void SetCuttingShader(ComputeShader cutting)
        {
            cut = cutting;
            if (cut == null) { cutReady = false; return; }
            kClearCutFP          = cut.FindKernel("ClearCutFP");
            kLookupConn          = cut.FindKernel("LookupConnectivity");
            kAccumCutFP          = cut.FindKernel("AccumulateCutFP");
            kComputeComponentFP  = cut.FindKernel("ComputeComponentFP"); // 2C-fix (plan §7)
            kInterpCutFP         = cut.FindKernel("InterpCutFP");        // Stage 2D-2b (paper clean:1271-1276)
            kBuildCutTris        = cut.FindKernel("BuildCutTriangles");
            kNormalizeCutFP      = cut.FindKernel("DC_NormalizeCutFP");
            cutReady = true;
        }

        /// <summary>G(n) = max(1, ceil(n / 64)) — number of 64-thread groups to cover n elements.</summary>
        static int G(int n) => Mathf.Max(1, Mathf.CeilToInt(n / 64f));

        /// <summary>
        /// Runs the full DC pipeline. Dispatch order (plan §4 — MUST NOT be reordered):
        ///   1. Set scalars (Recon + 2C: _CornerCount, _VoxelCount, _TriCapacity, _CutSlotCount)
        ///   2. RebuildIsectWorld    — G(IsectCount)
        ///   3. DC_FeaturePoints     — G(VoxelCount)
        ///   --- 2C cut-FP block (BETWEEN DC_FeaturePoints and the TriCounter reset) ---
        ///   4. ClearCutFP           — G(8*VoxelCount)
        ///   5. LookupConnectivity   — G(VoxelCount)
        ///   6. AccumulateCutFP      — G(CutPointCapacity)   (kernel self-guards on _CutPointCounter[0])
        ///   7. ComputeComponentFP   — G(VoxelCount)         (2C-fix: unified per-component QEF, plan §7)
        ///   7b. InterpCutFP         — G(8*VoxelCount)       (2D-2b: EMA-smooth _CutFP drift, paper clean:1271-1276)
        ///   ---------------------------------------------------------------------------
        ///   8. reset TriCounter=0   — single reset, head of geometry
        ///   9. DC_Stitch            — G(SurfaceEdgeCount)    external faces → triBuf
        ///  10. BuildCutTriangles    — G(CornerCount)         cut faces → SAME triBuf
        ///  11. DC_ClearNormals      — G(VoxelCount)          (cut-FP normals already zeroed in ClearCutFP)
        ///  12. DC_NormalsScatter    — G(TriCapacity)         tag-aware (external + internal)
        ///  13. DC_NormalsNormalize  — G(VoxelCount)          external normals
        ///  14. DC_NormalizeCutFP    — G(8*VoxelCount)        cut-FP normals
        ///  15. DC_WriteIndirectArgs — (1,1,1)
        /// When the Cutting shader is null, steps 4-7b, 10 and 14 are skipped → Stage-1-identical.
        /// cutFPInterp (2D-2b α): default 1.0 = OFF (byte-identical); lower = more smoothing of the cut-FP drift.
        /// </summary>
        public void Build(ReconBuffers rb, int3 dims, float L, int qefIters, float cutFPInterp = 1f)
        {
            int slotCount = 8 * rb.VoxelCount;

            // ── 1. Recon scalar uniforms ─────────────────────────────────────────────
            cs.SetInt("_VoxelCount",       rb.VoxelCount);
            cs.SetFloat("_L",              L);
            cs.SetInt("_QefIters",         qefIters);
            cs.SetInts("_Dims",            dims.x, dims.y, dims.z);
            cs.SetInt("_SurfaceEdgeCount", rb.SurfaceEdgeCount);
            cs.SetInt("_TriCapacity",      rb.TriCapacity);
            cs.SetInt("_IsectCount",       rb.IsectCount);

            // ── 2. RebuildIsectWorld (FIRST: re-project isect positions along deformed edges) ─
            Bind(kRebuild, rb);
            cs.Dispatch(kRebuild, G(rb.IsectCount), 1, 1);

            // ── 3. DC_FeaturePoints (external FP, Stage 1) ────────────────────────────
            Bind(kFP, rb);
            cs.Dispatch(kFP, G(rb.VoxelCount), 1, 1);

            // ── 2C cut-FP block (runs BETWEEN DC_FeaturePoints and the TriCounter reset) ──
            if (cutReady)
            {
                // Cutting.compute scalar uniforms (plan §4 / §7).
                cut.SetInts("_Dims",         dims.x, dims.y, dims.z);
                cut.SetInt ("_VoxelCount",   rb.VoxelCount);
                cut.SetInt ("_CornerCount",  rb.CornerCount);
                cut.SetInt ("_TriCapacity",  rb.TriCapacity);
                cut.SetInt ("_CutSlotCount", slotCount);
                // 2C-fix (plan §7): ComputeComponentFP's QEF needs the iter count + voxel side length.
                // The External-FP QEF box-clamp reads _VoxelL (Cutting.compute), so the dispatcher that
                // consumes it owns it here — do NOT rely on CutDetector.Dispatch having set _VoxelL this
                // frame (cross-object coupling). _L is kept for back-compat (now a no-op on the cut shader).
                cut.SetInt  ("_QefIters", qefIters);
                cut.SetFloat("_L",        L);
                cut.SetFloat("_VoxelL",   L);

                // 4. ClearCutFP — zero accum/cnt/_CutFP.w/cutFPNormal + voxelFPCount.
                BindCut(kClearCutFP, rb);
                cut.Dispatch(kClearCutFP, G(slotCount), 1, 1);

                // 5. LookupConnectivity — voxelFPCount = compCount.
                BindCut(kLookupConn, rb);
                cut.Dispatch(kLookupConn, G(rb.VoxelCount), 1, 1);

                // 6. AccumulateCutFP — scatter cut points → component sums. Dispatched over the
                //    cut-point CAPACITY; the kernel self-guards on _CutPointCounter[0] (counter on GPU).
                BindCut(kAccumCutFP, rb);
                cut.Dispatch(kAccumCutFP, G(rb.CutPointCapacity), 1, 1);

                // 7. ComputeComponentFP — unified per-component QEF → _CutFP (2C-fix, plan §7).
                //    One thread per VOXEL (loops its components). Needs Isect/IsectWorld/VoxelIsect*/
                //    VoxelCorner/VoxelOccupied bound (added in BindCut). Dispatched over VoxelCount.
                BindCut(kComputeComponentFP, rb);
                cut.Dispatch(kComputeComponentFP, G(rb.VoxelCount), 1, 1);

                // 7b. InterpCutFP — temporal feature-point interpolation (Stage 2D-2b, paper clean:1271-1276).
                //    EMA-smooth _CutFP (curr) toward its previous displayed value to fix the FP-drift snap
                //    under large deformations; writes the displayed FP back to _PrevCutFP for next frame.
                //    α = cutFPInterp (default 1.0 ⇒ lerp(prev,curr,1)=curr ⇒ byte-identical OFF switch).
                //    Runs AFTER ComputeComponentFP (curr ready) and BEFORE the consumers (DC_Stitch /
                //    BuildCutTriangles / DC_NormalsScatter / DC_NormalizeCutFP) so they see the smoothed FP.
                //    Slot domain G(8*VoxelCount) — same as ComputeComponentFP/ClearCutFP (guard mirrors them).
                cut.SetFloat("_CutFPInterp", cutFPInterp);
                cut.SetBuffer(kInterpCutFP, "_CutFP",     rb.CutFP);
                cut.SetBuffer(kInterpCutFP, "_PrevCutFP", rb.PrevCutFP);
                cut.Dispatch(kInterpCutFP, G(slotCount), 1, 1);
            }

            // ── 8. Reset TriCounter BEFORE stitch (single reset, head of geometry) ─────
            rb.TriCounter.SetData(new uint[] { 0 });

            // ── 9. DC_Stitch (external faces → triBuf) ────────────────────────────────
            Bind(kStitch, rb);
            cs.Dispatch(kStitch, G(rb.SurfaceEdgeCount), 1, 1);

            // ── 10. BuildCutTriangles (cut faces → SAME triBuf; AFTER DC_Stitch) ──────
            if (cutReady)
            {
                BindCut(kBuildCutTris, rb);
                cut.Dispatch(kBuildCutTris, G(rb.CornerCount), 1, 1);
            }

            // ── 11. DC_ClearNormals (cut-FP normals already zeroed in ClearCutFP) ─────
            Bind(kClearN, rb);
            cs.Dispatch(kClearN, G(rb.VoxelCount), 1, 1);

            // ── 12. DC_NormalsScatter (tag-aware; upper-bound dispatch, kernel guards) ─
            Bind(kScatterN, rb);
            cs.Dispatch(kScatterN, G(rb.TriCapacity), 1, 1);

            // ── 13. DC_NormalsNormalize (external normals) ────────────────────────────
            Bind(kNormN, rb);
            cs.Dispatch(kNormN, G(rb.VoxelCount), 1, 1);

            // ── 14. DC_NormalizeCutFP (cut-FP normals) ────────────────────────────────
            if (cutReady)
            {
                BindCut(kNormalizeCutFP, rb);
                cut.Dispatch(kNormalizeCutFP, G(slotCount), 1, 1);
            }

            // ── 15. DC_WriteIndirectArgs ──────────────────────────────────────────────
            Bind(kArgs, rb);
            cs.Dispatch(kArgs, 1, 1, 1);
        }

        /// <summary>
        /// Binds all Recon-path ReconBuffers to the given Recon.compute kernel.
        /// Stage 2C: also binds _CutFP + _CutFPNormal so the tag-aware DC_NormalsScatter resolves
        /// its declarations (harmless for other kernels that don't declare them).
        /// Kernels that don't declare a given buffer silently ignore the binding.
        /// </summary>
        void Bind(int k, ReconBuffers rb)
        {
            cs.SetBuffer(k, "_CornerPos",        rb.CornerPos);
            cs.SetBuffer(k, "_VoxelCorner",      rb.VoxelCorner);
            cs.SetBuffer(k, "_VoxelIsectOffset", rb.VoxelIsectOffset);
            cs.SetBuffer(k, "_VoxelIsectCount",  rb.VoxelIsectCount);
            cs.SetBuffer(k, "_Isect",            rb.Isect);
            cs.SetBuffer(k, "_IsectWorld",       rb.IsectWorld);
            cs.SetBuffer(k, "_VoxelExternalFP",  rb.VoxelExternalFP);
            cs.SetBuffer(k, "_SurfaceEdges",     rb.SurfaceEdges);
            cs.SetBuffer(k, "_Tri",              rb.Tri);
            cs.SetBuffer(k, "_TriCounter",       rb.TriCounter);
            cs.SetBuffer(k, "_ExtNormalFixed",   rb.VoxelExternalFPNormal);
            cs.SetBuffer(k, "_ExtNormalF",       rb.VoxelExternalFPNormalF);
            cs.SetBuffer(k, "_IndirectArgs",     rb.IndirectArgs);
            // Stage 2C: tag-aware scatter reads/writes these (zero-cut → never touched).
            cs.SetBuffer(k, "_CutFP",            rb.CutFP);
            cs.SetBuffer(k, "_CutFPNormal",      rb.CutFPNormal);
            // 2C-fix (plan §7 RC1): DC_Stitch's per-component keying. Bound to ALL Recon kernels
            // (harmless to those that don't declare them); needed so DC_Stitch resolves _Conn4096/
            // _VoxelCutMask/_VoxelOccupied. Zero-cut → mask all-zero → DC_Stitch else branch.
            cs.SetBuffer(k, "_Conn4096",         rb.Conn4096);
            cs.SetBuffer(k, "_VoxelCutMask",     rb.VoxelCutMask);
            cs.SetBuffer(k, "_VoxelOccupied",    rb.VoxelOccupied);
        }

        /// <summary>
        /// Stage 2C: binds the shared buffers to a Cutting.compute cut kernel. Each kernel declares
        /// only the subset it uses; unreferenced bindings are silently ignored. (One uniform Bind so
        /// the dispatch order in Build() stays terse.)
        /// </summary>
        void BindCut(int k, ReconBuffers rb)
        {
            cut.SetBuffer(k, "_CornerPos",      rb.CornerPos);
            cut.SetBuffer(k, "_GridEdges",      rb.GridEdges);
            cut.SetBuffer(k, "_VoxelCutMask",   rb.VoxelCutMask);
            cut.SetBuffer(k, "_CutPoint",       rb.CutPoint);
            cut.SetBuffer(k, "_CutPointCounter",rb.CutPointCounter);
            cut.SetBuffer(k, "_Conn4096",       rb.Conn4096);
            cut.SetBuffer(k, "_Tri",            rb.Tri);
            cut.SetBuffer(k, "_TriCounter",     rb.TriCounter);
            cut.SetBuffer(k, "_CutFP",          rb.CutFP);
            // Stage 2D-2b: bind the EMA history so any cut kernel declaring _PrevCutFP (InterpCutFP)
            // resolves it. InterpCutFP also binds _CutFP/_PrevCutFP inline at its dispatch (minimal);
            // binding here is harmless to kernels that don't declare it.
            cut.SetBuffer(k, "_PrevCutFP",      rb.PrevCutFP);
            cut.SetBuffer(k, "_CutFPAccumPos",  rb.CutFPAccumPos);
            cut.SetBuffer(k, "_CutFPAccumCnt",  rb.CutFPAccumCnt);
            cut.SetBuffer(k, "_VoxelFPCount",   rb.VoxelFPCount);
            cut.SetBuffer(k, "_CutFPNormal",    rb.CutFPNormal);
            cut.SetBuffer(k, "_CutFPNormalF",   rb.CutFPNormalF);
            // 2C-fix (plan §7): ComputeComponentFP's isect→component QEF inputs + RC2 gate.
            cut.SetBuffer(k, "_Isect",            rb.Isect);
            cut.SetBuffer(k, "_IsectWorld",       rb.IsectWorld);
            cut.SetBuffer(k, "_VoxelIsectOffset", rb.VoxelIsectOffset);
            cut.SetBuffer(k, "_VoxelIsectCount",  rb.VoxelIsectCount);
            cut.SetBuffer(k, "_VoxelOccupied",    rb.VoxelOccupied);
            // Stage 2D-2 (paper §2.1.2): AccumulateCutFP reconstructs world = cornerPos[owner] +
            // mul(R_owner, localOffset) — bind the per-particle rotation R (frame N's, written by
            // ComputeParticleRot before dc.Build). Other cut kernels that don't declare it ignore it.
            cut.SetBuffer(k, "_ParticleRot",      rb.ParticleRot);
        }
    }
}
