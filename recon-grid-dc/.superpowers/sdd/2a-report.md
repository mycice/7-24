# Stage 2A — CONN4096 Report

**Status:** COMPLETE

**Commit:** see short-hash below after commit

**One-line note:** Pure-CPU CONN4096 LUT (union-find on uncut subgraph, dense vertToComp) + GPU buffer allocation in ReconBuffers + 5 CPU EditMode tests.

**Confirmations:**
- (a) Union on bit==0: `if ((config >> e & 1) == 0)` — unites endpoints of UNCUT edges only; cut bits (==1) are skipped. `// paper §2.1.1: cut bit DISCONNECTS the edge`.
- (b) vertToComp dense: first-appearance renumbering assigns ids 0..compCount-1 in vertex scan order. Dense-id invariant tested for all 4096 configs.
- (c) Uses GridConventions.EdgeCorners: `GridConventions.EdgeCorners[e, 0]` / `[e, 1]` for all 12 edge endpoints. No hardcoded alternative table.

**Files written/extended:**
- `Assets/ReconGridDC/Recon/ConnectivityLUT.cs` (NEW)
- `Assets/ReconGridDC/Recon/ReconBuffers.cs` (EXTENDED: Conn4096Gpu struct, Conn4096 buffer field, UploadConn4096 helper, Dispose entry)
- `Assets/ReconGridDC/Tests/ConnectivityLUT_Tests.cs` (NEW: 5 CPU EditMode tests)
