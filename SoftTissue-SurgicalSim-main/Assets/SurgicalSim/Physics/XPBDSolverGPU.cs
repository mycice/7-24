// XPBDSolverGPU.cs 鈥?NeoHookean 寮规€?XPBD 姹傝В鍣?// 鍙傝€? FantasyVR/neohookean_XPBD + Peng Yu EG2025 璁烘枃
// Graph Coloring Gauss-Seidel + NeoHookean (deviatoric + hydrostatic)
using System;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Rendering;

namespace SurgicalSim.Physics
{
    public class XPBDSolverGPU : IDisposable
    {
        // 鈹€鈹€ 鍏紑鍙傛暟 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
        public int     NumSubSteps      = 10;
        public float   EdgeCompliance   = 2.0f;
        public float   YoungsModulus    = 1e5f;   // Pa
        public float   PoissonsRatio    = 0.45f;
        public float   Damping          = 0.05f;  // 璁烘枃榛樿
        public float   Density          = 1000f;  // kg/m^3
        public Vector3 Gravity          = new Vector3(0f, -9.81f, 0f);
        public float   GroundY          = 0.0f;

        // Lam茅 parameters derived from Young's modulus and Poisson's ratio.
        float _lameLambda, _lameMu;

        // 鈹€鈹€ GPU 璧勬簮 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
        readonly ComputeShader _cs;
        int _kIntegrate, _kSolveEdges, _kSolveNHDev, _kSolveNHHyd, _kSolveNHDevHyd,
            _kPostSolve, _kGroundCollision, _kToolCollision,
            _kSolveSurfaceToolContacts, _kClearToolContactMetrics, _kToolContactMetrics;

        ComputeBuffer _bufPos, _bufPrevPos, _bufVel, _bufInvMass;
        ComputeBuffer _bufTetIds, _bufRestVol, _bufTetActive;
        ComputeBuffer _bufEdgeIds, _bufRestLen, _bufEdgeActive;
        ComputeBuffer _bufColorFlat, _bufEdgeColorFlat;
        ComputeBuffer _bufSurfaceTriIds, _bufSurfaceTriColorFlat, _bufSurfaceTriCandidateFlat;
        ComputeBuffer _bufToolContactMetrics;
        // NeoHookean 涓撶敤
        ComputeBuffer _bufInvRestMatrix;    // float4[numT*3]
        ComputeBuffer _bufAlphaDeviatoric;  // float[numT]
        ComputeBuffer _bufAlphaHydrostatic; // float[numT]

        int[] _groupOff, _groupCnt; int _numColors;
        int[] _edgeGroupOff, _edgeGroupCnt; int _numEdgeColors;
        int[] _surfaceTriColorByTri;
        int[] _toolCandidateSelected, _toolCandidateFlat;
        int[] _toolCandidateGroupOff, _toolCandidateGroupCnt, _toolCandidateGroupCursor;

        public const int MaxToolCapsules = 12;

        int _numP, _numT, _numE, _numSurfaceTris;
        const int TH = 64;
        Vector4[] _readBuf;
        Vector4[] _velReadBuf;
        Vector4[] _prevReadBuf;
        Vector4[] _sparsePosUpload;
        Vector4[] _sparseVelUpload;
        Vector4[] _sparsePrevUpload;
        bool _positionReadbackPending;
        bool _toolMetricsReadbackPending;
        int _readbackGeneration;
        int _toolMetricsStepCounter;
        int _toolMetricsLastStep;
        int[] _toolMetricsReadBuf;
        List<int>[] _edgeToTets;
        int[] _surfaceGroupOff, _surfaceGroupCnt; int _numSurfaceColors;
        int _activeToolCapsules;
        int _activeToolSurfaceCandidateTris;
        int _activeToolSurfaceCandidateColors;
        bool _toolCandidatesValid;
        readonly Vector3[] _toolCapsuleBboxMin = new Vector3[MaxToolCapsules];
        readonly Vector3[] _toolCapsuleBboxMax = new Vector3[MaxToolCapsules];
        readonly Vector4[] _capsuleAUpload = new Vector4[MaxToolCapsules];
        readonly Vector4[] _capsuleBUpload = new Vector4[MaxToolCapsules];
        readonly Vector4[] _prevCapsuleAUpload = new Vector4[MaxToolCapsules];
        readonly Vector4[] _prevCapsuleBUpload = new Vector4[MaxToolCapsules];
        readonly float[] _capsuleFrictionUpload = new float[MaxToolCapsules];
        readonly Vector3[] _legacyCapsuleA = new Vector3[3];
        readonly Vector3[] _legacyCapsuleB = new Vector3[3];
        readonly Vector3[] _legacyPrevCapsuleA = new Vector3[3];
        readonly Vector3[] _legacyPrevCapsuleB = new Vector3[3];
        readonly float[] _legacyCapsuleR = new float[3];

        public float ToolContactDistance = 0.01f;
        public float ToolContactCompliance = 1e-7f;
        public int ToolContactIterations = 2;
        public int ToolContactCouplingPasses = 2;
        public float ToolContactTangentialFriction = 0.65f;
        public float ToolContactTangentialDamping = 0.5f;
        public bool UseToolContactCandidateCulling = true;
        public float ToolContactCandidatePadding = 0.06f;
        public ToolContactMetricsSnapshot LastToolContactMetrics { get; private set; }
        public int ToolContactMetricsAge => Mathf.Max(0, _toolMetricsStepCounter - _toolMetricsLastStep);
        public int ActiveToolCapsules => _activeToolCapsules;
        public int NumSurfaceTris => _numSurfaceTris;
        public int ActiveToolSurfaceCandidateTris =>
            _activeToolCapsules > 0
                ? (UseToolContactCandidateCulling ? _activeToolSurfaceCandidateTris : _numSurfaceTris)
                : 0;
        public int InternalDispatchesPerPass => _numEdgeColors + _numColors;
        public int EstimatedInternalDispatchesPerFrame =>
            NumSubSteps * (1 + Mathf.Max(1, ToolContactCouplingPasses)) * InternalDispatchesPerPass;
        public int EstimatedToolContactDispatchesPerFrame =>
            _activeToolCapsules > 0
                ? NumSubSteps * (1 + Mathf.Max(1, ToolContactCouplingPasses)) *
                  Mathf.Max(1, ToolContactIterations) *
                  (UseToolContactCandidateCulling ? _activeToolSurfaceCandidateColors : _numSurfaceColors)
                : 0;

        public readonly struct ToolContactMetricsSnapshot
        {
            public readonly int Version;
            public readonly int ContactCount;
            public readonly float MaxDepth;
            public readonly float DepthSum;
            public readonly Vector3 NormalLocal;
            public readonly float NormalReliability;

            public ToolContactMetricsSnapshot(
                int version,
                int contactCount,
                float maxDepth,
                float depthSum,
                Vector3 normalLocal,
                float normalReliability)
            {
                Version = version;
                ContactCount = contactCount;
                MaxDepth = maxDepth;
                DepthSum = depthSum;
                NormalLocal = normalLocal;
                NormalReliability = normalReliability;
            }
        }

        [System.Runtime.InteropServices.StructLayout(
            System.Runtime.InteropServices.LayoutKind.Sequential)]
        struct UInt4 { public uint x, y, z, w; }
        [System.Runtime.InteropServices.StructLayout(
            System.Runtime.InteropServices.LayoutKind.Sequential)]
        struct UInt2 { public uint x, y; }
        [System.Runtime.InteropServices.StructLayout(
            System.Runtime.InteropServices.LayoutKind.Sequential)]
        struct UInt3 { public uint x, y, z; }

        public XPBDSolverGPU(ComputeShader cs)
        {
            _cs = cs ?? throw new ArgumentNullException(nameof(cs));
            _kIntegrate      = _cs.FindKernel("CSIntegrate");
            _kSolveEdges     = _cs.FindKernel("CSSolveEdges");
            _kSolveNHDev     = _cs.FindKernel("CSSolveNeoHookeanDev");
            _kSolveNHHyd     = _cs.FindKernel("CSSolveNeoHookeanHyd");
            _kSolveNHDevHyd  = _cs.FindKernel("CSSolveNeoHookeanDevHyd");
            _kPostSolve      = _cs.FindKernel("CSPostSolve");
            _kGroundCollision = _cs.FindKernel("CSGroundCollision");
            _kToolCollision   = _cs.FindKernel("CSToolCollision");
            _kSolveSurfaceToolContacts = _cs.FindKernel("CSSolveSurfaceToolContacts");
            _kClearToolContactMetrics = _cs.FindKernel("CSClearToolContactMetrics");
            _kToolContactMetrics = _cs.FindKernel("CSToolContactMetrics");
        }

        // 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲
        // Init 鈥?璁＄畻 B^{-1}, Lam茅 鍙傛暟, restVolume, invMass
        // 鍙傝€? FantasyVR init_phy + init_alpha
        // 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲
        public void Init(Core.TetMeshData data)
        {
            InvalidatePendingReadbacks();
            _numP = data.NumParticles;
            _numT = data.NumTets;
            _numSurfaceTris = data.NumSurfaceTris;
            _activeToolCapsules = 0;

            // Lam茅 鍙傛暟
            float E = YoungsModulus;
            float nu = PoissonsRatio;
            _lameLambda = E * nu / ((1f + nu) * (1f - 2f * nu));
            _lameMu     = E / (2f * (1f + nu));

            Vector3[] restPos = data.RestPositions;
            Vector3[] curPos  = data.Positions;

            // 鈹€鈹€ Per-tet: B^{-1}, restVol, alpha 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
            float[] mass = new float[_numP];
            float[] invMass = new float[_numP];
            float[] restVolumes = new float[_numT];
            Vector4[] invRestMat = new Vector4[_numT * 3]; // 3 rows per tet
            float[] alphaDev    = new float[_numT];
            float[] alphaHyd    = new float[_numT];
            const float minVolForConstraint = 1e-10f;

            for (int i = 0; i < _numT; i++)
            {
                if (!data.TetActive[i]) continue;
                int id0 = data.TetIds[4*i], id1 = data.TetIds[4*i+1];
                int id2 = data.TetIds[4*i+2], id3 = data.TetIds[4*i+3];

                Vector3 e1 = restPos[id1] - restPos[id0];
                Vector3 e2 = restPos[id2] - restPos[id0];
                Vector3 e3 = restPos[id3] - restPos[id0];

                float det = Vector3.Dot(Vector3.Cross(e1, e2), e3);
                float vol = Mathf.Abs(det) / 6f;
                restVolumes[i] = vol;

                float avgMass = Density * vol / 4f;
                mass[id0] += avgMass;
                mass[id1] += avgMass;
                mass[id2] += avgMass;
                mass[id3] += avgMass;

                // Degenerate tetrahedra are not stable Neo-Hookean elements.
                // Edge constraints keep the local mesh connected without huge gradients.
                if (Mathf.Abs(det) < 1e-12f || vol < minVolForConstraint)
                {
                    invRestMat[i*3+0] = Vector4.zero;
                    invRestMat[i*3+1] = Vector4.zero;
                    invRestMat[i*3+2] = Vector4.zero;
                    alphaDev[i] = 0f;
                    alphaHyd[i] = 0f;
                    continue;
                }

                float invDet = 1f / det;
                Vector3 r0 = Vector3.Cross(e2, e3) * invDet;
                Vector3 r1 = Vector3.Cross(e3, e1) * invDet;
                Vector3 r2 = Vector3.Cross(e1, e2) * invDet;

                invRestMat[i*3+0] = new Vector4(r0.x, r0.y, r0.z, 0);
                invRestMat[i*3+1] = new Vector4(r1.x, r1.y, r1.z, 0);
                invRestMat[i*3+2] = new Vector4(r2.x, r2.y, r2.z, 0);

                // 绱姞璐ㄩ噺 (鍙傝€?FantasyVR: mass[a] += density * V / 4)
                // alpha = 1/(h^2 * mu * V)
                float h = Time.fixedDeltaTime;  // ~0.02
                float inv_h2 = 1f / (h * h);

                // Skip constraints for very small tetrahedra to avoid unstable compliance values.
                if (vol < minVolForConstraint)
                {
                    alphaDev[i] = 0f;
                    alphaHyd[i] = 0f;
                }
                else
                {
                    alphaDev[i] = (_lameMu > 0f) ? inv_h2 / (_lameMu * vol) : 0f;
                    alphaHyd[i] = (_lameLambda > 0f) ? inv_h2 / (_lameLambda * vol) : 0f;

                    const float maxAlpha = 1e8f;
                    alphaDev[i] = Mathf.Min(alphaDev[i], maxAlpha);
                    alphaHyd[i] = Mathf.Min(alphaHyd[i], maxAlpha);
                }
            }

            // mass 鈫?invMass
            // Compute the average MASS (not invMass) using the
            // median-ish "robust" mean, then
            // enforce a hard mass floor at avgMass / 100 to absolutely
            // bound the worst-case inverse mass. Averaging invMass
            // directly is non-robust: a single 1e-11 m^3 sliver tet
            // produces an invMass of ~4e+8 which then drives
            // fallbackInvMass into the 1e+7 range and lets the
            // fallbackInvMass*10 clamp accept invMasses up to 1e+8.
            // The observed [SepDiag InvMass max=3.478E+08] is exactly
            // that failure mode.
            float sumParticleMass = 0f;
            int countParticleMass = 0;
            for (int i = 0; i < _numP; i++)
            {
                if (data.InvMass[i] == 0f) continue;
                if (mass[i] > 1e-12f)
                {
                    sumParticleMass += mass[i];
                    countParticleMass++;
                }
            }
            float avgParticleMass = countParticleMass > 0 ? sumParticleMass / countParticleMass : 1f;
            // 鈽?Step 4 (XPBD conditioning): tighten the mass floor to
            // avgMass * 0.1 keeps inverse-mass ratios in a numerically stable range.
            float minReasonableMass = avgParticleMass * 0.1f;

            int massFloorHits = 0;
            for (int i = 0; i < _numP; i++)
            {
                if (data.InvMass[i] == 0f)
                {
                    invMass[i] = 0f;
                    continue;
                }
                float m = mass[i];
                if (m < minReasonableMass)
                {
                    m = minReasonableMass;
                    massFloorHits++;
                }
                invMass[i] = 1f / m;
            }
            if (massFloorHits > 0)
            {
                Debug.Log($"[XPBDSolverGPU] mass floor applied to {massFloorHits} particles " +
                          $"(avgParticleMass={avgParticleMass:G4}, floor={minReasonableMass:G4})");
            }

            // 鈹€鈹€ 寤鸿竟 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
            BuildEdges(data, restPos, out int[] edgeIds, out float[] restLens);
            _numE = edgeIds.Length / 2;

            // 鈹€鈹€ Graph Coloring 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
            BuildTetColorGroups(data, out int[] colorFlat,
                out _groupOff, out _groupCnt, out _numColors);
            BuildEdgeColorGroups(edgeIds, _numE, _numP,
                out int[] edgeColorFlat,
                out _edgeGroupOff, out _edgeGroupCnt, out _numEdgeColors);
            BuildSurfaceTriColorGroups(data.SurfaceTriIds, _numSurfaceTris, _numP,
                out int[] surfaceTriColorFlat,
                out _surfaceGroupOff, out _surfaceGroupCnt, out _surfaceTriColorByTri,
                out _numSurfaceColors);

            // 鈹€鈹€ 涓婁紶 GPU Buffer 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
            _bufPos      = MkV4(ToV4(curPos, _numP));
            _bufPrevPos  = MkV4(ToV4(data.PrevPositions, _numP));
            _bufVel      = MkV4(ToV4(data.Velocities, _numP)); // 浣跨敤瀹為檯閫熷害,涓嶆竻闆?
            _bufInvMass  = MkF(invMass);

            _bufTetIds   = MkU4(ToU4(data.TetIds, _numT));
            _bufRestVol  = MkF(restVolumes);
            int[] actI = new int[_numT];
            for (int i = 0; i < _numT; i++) actI[i] = data.TetActive[i] ? 1 : 0;
            _bufTetActive = MkI(actI);

            // NeoHookean 涓撶敤 buffer
            _bufInvRestMatrix    = MkV4(invRestMat);
            _bufAlphaDeviatoric  = MkF(alphaDev);
            _bufAlphaHydrostatic = MkF(alphaHyd);

            _bufEdgeIds  = MkU2(ToU2(edgeIds, _numE));
            _bufRestLen  = MkF(restLens);
            int[] edgeActI = new int[_numE];
            for (int i = 0; i < _numE; i++) edgeActI[i] = 1;
            _bufEdgeActive = MkI(edgeActI);

            _bufColorFlat     = MkI(colorFlat);
            _bufEdgeColorFlat = MkI(edgeColorFlat);
            if (_numSurfaceTris > 0)
            {
                _bufSurfaceTriIds = MkU3(ToU3(data.SurfaceTriIds, _numSurfaceTris));
                _bufSurfaceTriColorFlat = MkI(surfaceTriColorFlat);
                _bufSurfaceTriCandidateFlat = new ComputeBuffer(Mathf.Max(1, _numSurfaceTris), 4);
                _toolCandidateSelected = new int[_numSurfaceTris];
                _toolCandidateFlat = new int[_numSurfaceTris];
                _toolCandidateGroupOff = new int[_numSurfaceColors];
                _toolCandidateGroupCnt = new int[_numSurfaceColors];
                _toolCandidateGroupCursor = new int[_numSurfaceColors];
            }
            _bufToolContactMetrics = new ComputeBuffer(8, sizeof(int));

            _readBuf = new Vector4[_numP];
            _velReadBuf = new Vector4[_numP];
            _prevReadBuf = new Vector4[_numP];
            _toolMetricsReadBuf = new int[8];
            LastToolContactMetrics = default;
            _toolMetricsStepCounter = 0;
            _toolMetricsLastStep = 0;
            BindAll();

            // 鈹€鈹€ 璇婃柇淇℃伅 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
            int pinnedCount = 0, zeroMassCount = 0, degenTets = 0;
            float minInvM = float.MaxValue, maxInvM = 0f;
            float minVol = float.MaxValue, maxVol = 0f;
            float minAlphaD = float.MaxValue, maxAlphaD = 0f;
            float maxBnorm = 0f;

            for (int i = 0; i < _numP; i++)
            {
                if (invMass[i] == 0f) pinnedCount++;
                if (mass[i] < 1e-12f && invMass[i] == 0f && data.InvMass[i] != 0f) zeroMassCount++;
                if (invMass[i] > 0f) { minInvM = Mathf.Min(minInvM, invMass[i]); maxInvM = Mathf.Max(maxInvM, invMass[i]); }
            }
            for (int i = 0; i < _numT; i++)
            {
                if (restVolumes[i] < 1e-15f) { degenTets++; continue; }
                minVol = Mathf.Min(minVol, restVolumes[i]);
                maxVol = Mathf.Max(maxVol, restVolumes[i]);
                if (alphaDev[i] > 0f) { minAlphaD = Mathf.Min(minAlphaD, alphaDev[i]); maxAlphaD = Mathf.Max(maxAlphaD, alphaDev[i]); }
                // B matrix norm
                Vector3 br0 = new Vector3(invRestMat[i*3].x, invRestMat[i*3].y, invRestMat[i*3].z);
                Vector3 br1 = new Vector3(invRestMat[i*3+1].x, invRestMat[i*3+1].y, invRestMat[i*3+1].z);
                Vector3 br2 = new Vector3(invRestMat[i*3+2].x, invRestMat[i*3+2].y, invRestMat[i*3+2].z);
                float bn = br0.magnitude + br1.magnitude + br2.magnitude;
                maxBnorm = Mathf.Max(maxBnorm, bn);
            }

            Debug.Log($"[XPBDSolverGPU] NeoHookean Init | P:{_numP} T:{_numT} E:{_numE} SurfaceTris:{_numSurfaceTris} | " +
                      $"Colors:{_numColors} EdgeColors:{_numEdgeColors} SurfaceColors:{_numSurfaceColors}");
            Debug.Log($"[XPBDSolverGPU] Lam茅: 位={_lameLambda:G4} 渭={_lameMu:G4} | " +
                      $"E={YoungsModulus:G4} 谓={PoissonsRatio}");
            Debug.Log($"[XPBDSolverGPU] InvMass: [{minInvM:G4}, {maxInvM:G4}] | " +
                      $"Pinned: {pinnedCount} | ZeroMass(bug): {zeroMassCount}");
            Debug.Log($"[XPBDSolverGPU] RestVol: [{minVol:G4}, {maxVol:G4}] | " +
                      $"DegenTets: {degenTets}/{_numT}");
            Debug.Log($"[XPBDSolverGPU] AlphaDev: [{minAlphaD:G4}, {maxAlphaD:G4}] | " +
                      $"MaxBnorm: {maxBnorm:G4}");
        }

        // 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲
        // Step 鈥?姣忓抚璋冪敤
        // 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲
        public void Step(float dt)
        {
            if (_bufPos == null || _bufInvMass == null) return;
            BindAll();

            float sdt = dt / NumSubSteps;
            float sdt2 = sdt * sdt;

            _cs.SetFloat("_Dt",       sdt);
            _cs.SetFloat("_GravityX", Gravity.x);
            _cs.SetFloat("_GravityY", Gravity.y);
            _cs.SetFloat("_GravityZ", Gravity.z);
            _cs.SetFloat("_EdgeAlpha", EdgeCompliance / sdt2);
            _cs.SetFloat("_Damping",  Damping);
            _cs.SetFloat("_Omega",    0.2f);   // SOR 鏉惧紱: GS 姣忎釜绾︽潫鍙簲鐢?20% 鏍℃
            _cs.SetFloat("_GroundY",  GroundY);
            _cs.SetInt("_NumParticles", _numP);
            _cs.SetInt("_NumTets",      _numT);
            _cs.SetInt("_NumEdges",     _numE);
            _cs.SetInt("_NumSurfaceTris", _numSurfaceTris);
            _cs.SetFloat("_ToolContactDistance", ToolContactDistance);
            _cs.SetFloat("_ToolContactAlpha", ToolContactCompliance / sdt2);
            _cs.SetFloat("_ToolContactTangentialFriction", Mathf.Clamp01(ToolContactTangentialFriction));
            _cs.SetFloat("_ToolContactTangentialDamping", Mathf.Clamp01(ToolContactTangentialDamping));
            _toolMetricsStepCounter++;
            ClearToolContactMetrics();
            bool sampledToolMetrics = false;

            // A single graph-colored Gauss-Seidel pass is sufficient per substep.
            const int constraintIter = 1;

            for (int sub = 0; sub < NumSubSteps; sub++)
            {
                // 1. 绉垎 (semi-euler)
                Go(_kIntegrate, _numP);

                // 2. 绾︽潫鎶曞奖 鈥?杩唬澶氭 (鍏抽敭锛?
                for (int iter = 0; iter < constraintIter; iter++)
                {
                    // Edge constraints.
                    for (int c = 0; c < _numEdgeColors; c++)
                    {
                        _cs.SetInt("_GroupOffset", _edgeGroupOff[c]);
                        _cs.SetInt("_GroupCount",  _edgeGroupCnt[c]);
                        Go(_kSolveEdges, _edgeGroupCnt[c]);
                    }

                    // 2b. NeoHookean: Deviatoric + Hydrostatic
                    for (int c = 0; c < _numColors; c++)
                    {
                        _cs.SetInt("_GroupOffset", _groupOff[c]);
                        _cs.SetInt("_GroupCount",  _groupCnt[c]);
                        Go(_kSolveNHDevHyd, _groupCnt[c]);
                    }
                }

                // Sample the contact state once after the elastic constraints.
                if (!sampledToolMetrics)
                {
                    CollectToolContactMetrics();
                    sampledToolMetrics = true;
                }
                SolveToolContacts();

                // 4. 鍦伴潰纰版挒
                Go(_kGroundCollision, _numP);

                // 5. 鍐呴儴绾︽潫鍜屾帴瑙︿氦鏇挎眰瑙ｏ紝璁╁眬閮ㄥ帇闄疯兘鎵╂暎鍒颁綋鍐咃紝鍚屾椂淇濇寔鏈€缁堜笉绌挎ā
                int couplingPasses = Mathf.Max(1, ToolContactCouplingPasses);
                for (int pass = 0; pass < couplingPasses; pass++)
                {
                    SolveInternalOnce();
                    SolveToolContacts();
                }

                // 6. 鏇存柊閫熷害 + 闃诲凹
                Go(_kPostSolve, _numP);
            }
            RequestToolContactMetricsReadback();
        }

        // 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲
        // 璇诲洖浣嶇疆
        // 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲
        void SolveInternalOnce()
        {
            for (int c = 0; c < _numEdgeColors; c++)
            {
                _cs.SetInt("_GroupOffset", _edgeGroupOff[c]);
                _cs.SetInt("_GroupCount",  _edgeGroupCnt[c]);
                Go(_kSolveEdges, _edgeGroupCnt[c]);
            }

            for (int c = 0; c < _numColors; c++)
            {
                _cs.SetInt("_GroupOffset", _groupOff[c]);
                _cs.SetInt("_GroupCount",  _groupCnt[c]);
                Go(_kSolveNHDevHyd, _groupCnt[c]);
            }
        }

        void SolveToolContacts()
        {
            if (_activeToolCapsules <= 0 ||
                _numSurfaceTris <= 0 || _numSurfaceColors <= 0 ||
                _bufSurfaceTriIds == null || _bufSurfaceTriColorFlat == null)
                return;

            int[] groupOff = _surfaceGroupOff;
            int[] groupCnt = _surfaceGroupCnt;
            int numGroups = _numSurfaceColors;

            if (UseToolContactCandidateCulling)
            {
                if (!_toolCandidatesValid || _activeToolSurfaceCandidateTris <= 0 ||
                    _bufSurfaceTriCandidateFlat == null)
                    return;

                _cs.SetBuffer(_kSolveSurfaceToolContacts, "_SurfaceTriColorGroupFlat", _bufSurfaceTriCandidateFlat);
                groupOff = _toolCandidateGroupOff;
                groupCnt = _toolCandidateGroupCnt;
            }
            else
            {
                _cs.SetBuffer(_kSolveSurfaceToolContacts, "_SurfaceTriColorGroupFlat", _bufSurfaceTriColorFlat);
            }

            int iterations = Mathf.Max(1, ToolContactIterations);
            for (int iter = 0; iter < iterations; iter++)
            {
                for (int c = 0; c < numGroups; c++)
                {
                    int count = groupCnt[c];
                    if (count <= 0) continue;

                    _cs.SetInt("_SurfaceGroupOffset", groupOff[c]);
                    _cs.SetInt("_SurfaceGroupCount",  count);
                    Go(_kSolveSurfaceToolContacts, count);
                }
            }
        }

        void ClearToolContactMetrics()
        {
            if (_bufToolContactMetrics == null) return;
            _cs.SetBuffer(_kClearToolContactMetrics, "_ToolContactMetrics", _bufToolContactMetrics);
            Go(_kClearToolContactMetrics, 8);
        }

        void CollectToolContactMetrics()
        {
            if (_activeToolCapsules <= 0 ||
                _numSurfaceTris <= 0 ||
                _bufSurfaceTriIds == null ||
                _bufSurfaceTriColorFlat == null ||
                _bufToolContactMetrics == null)
                return;

            ComputeBuffer triGroupBuffer = _bufSurfaceTriColorFlat;
            int triCount = _numSurfaceTris;
            if (UseToolContactCandidateCulling)
            {
                if (!_toolCandidatesValid ||
                    _activeToolSurfaceCandidateTris <= 0 ||
                    _bufSurfaceTriCandidateFlat == null)
                    return;

                triGroupBuffer = _bufSurfaceTriCandidateFlat;
                triCount = _activeToolSurfaceCandidateTris;
            }

            _cs.SetInt("_SurfaceGroupOffset", 0);
            _cs.SetInt("_SurfaceGroupCount", triCount);
            _cs.SetBuffer(_kToolContactMetrics, "_Positions", _bufPos);
            _cs.SetBuffer(_kToolContactMetrics, "_InvMasses", _bufInvMass);
            _cs.SetBuffer(_kToolContactMetrics, "_SurfaceTriIds", _bufSurfaceTriIds);
            _cs.SetBuffer(_kToolContactMetrics, "_SurfaceTriColorGroupFlat", triGroupBuffer);
            _cs.SetBuffer(_kToolContactMetrics, "_ToolContactMetrics", _bufToolContactMetrics);
            Go(_kToolContactMetrics, triCount);
        }

        void RequestToolContactMetricsReadback()
        {
            if (_bufToolContactMetrics == null || _toolMetricsReadBuf == null) return;
            if (!SystemInfo.supportsAsyncGPUReadback)
            {
                _bufToolContactMetrics.GetData(_toolMetricsReadBuf);
                ProcessToolContactMetrics(_toolMetricsReadBuf, _toolMetricsStepCounter);
                return;
            }

            if (_toolMetricsReadbackPending) return;
            _toolMetricsReadbackPending = true;
            int generation = _readbackGeneration;
            int requestStep = _toolMetricsStepCounter;
            AsyncGPUReadback.Request(_bufToolContactMetrics, request =>
            {
                _toolMetricsReadbackPending = false;
                if (generation != _readbackGeneration) return;
                if (request.hasError) return;

                var src = request.GetData<int>();
                if (_toolMetricsReadBuf == null || _toolMetricsReadBuf.Length < 8)
                    _toolMetricsReadBuf = new int[8];
                int n = Mathf.Min(8, src.Length);
                for (int i = 0; i < n; i++)
                    _toolMetricsReadBuf[i] = src[i];
                for (int i = n; i < 8; i++)
                    _toolMetricsReadBuf[i] = 0;
                ProcessToolContactMetrics(_toolMetricsReadBuf, requestStep);
            });
        }

        void ProcessToolContactMetrics(int[] m, int sourceStep)
        {
            const float invDepthMetricScale = 1f / 1000000f;
            const float invNormalMetricScale = 1f / 10000f;
            int contactCount = m != null && m.Length > 0 ? Mathf.Max(0, m[0]) : 0;
            float maxDepth = m != null && m.Length > 1 ? Mathf.Max(0, m[1]) * invDepthMetricScale : 0f;
            float depthSum = m != null && m.Length > 5 ? Mathf.Max(0, m[5]) * invDepthMetricScale : 0f;
            Vector3 normalSum = Vector3.zero;
            if (m != null && m.Length > 4)
                normalSum = new Vector3(m[2], m[3], m[4]) * invNormalMetricScale;

            float reliability = 0f;
            Vector3 normal = Vector3.up;
            if (contactCount > 0 && normalSum.sqrMagnitude > 1e-10f)
            {
                reliability = Mathf.Clamp01(normalSum.magnitude / Mathf.Max(1, contactCount));
                normal = normalSum.normalized;
            }

            int version = LastToolContactMetrics.Version + 1;
            LastToolContactMetrics = new ToolContactMetricsSnapshot(
                version,
                contactCount,
                maxDepth,
                depthSum,
                normal,
                reliability);
            _toolMetricsLastStep = Mathf.Max(0, sourceStep);
        }

        public void ReadbackPositions(Core.TetMeshData data)
        {
            EnsureReadbackBuffers();
            _bufPos.GetData(_readBuf);
            for (int i = 0; i < _numP; i++)
                data.Positions[i] = new Vector3(_readBuf[i].x, _readBuf[i].y, _readBuf[i].z);
        }

        public bool RequestPositionReadback(Core.TetMeshData data)
        {
            if (data == null || _bufPos == null) return false;
            if (!SystemInfo.supportsAsyncGPUReadback)
            {
                ReadbackPositions(data);
                return true;
            }

            if (_positionReadbackPending) return false;
            _positionReadbackPending = true;
            int count = _numP;
            int generation = _readbackGeneration;
            AsyncGPUReadback.Request(_bufPos, request =>
            {
                if (generation != _readbackGeneration) return;
                _positionReadbackPending = false;
                if (request.hasError || data.Positions == null) return;

                var src = request.GetData<Vector4>();
                int n = Mathf.Min(count, Mathf.Min(src.Length, data.NumParticles));
                for (int i = 0; i < n; i++)
                    data.Positions[i] = new Vector3(src[i].x, src[i].y, src[i].z);
            });
            return true;
        }

        /// <summary>
        /// 鍥炶鍏ㄩ儴鐗╃悊鐘舵€?(浣嶇疆+閫熷害+prevPositions)
        /// 鍦?Dispose+Init 涔嬪墠璋冪敤, 纭繚 CPU 绔暟鎹槸鏈€鏂扮殑
        /// </summary>
        public void ReadbackAll(Core.TetMeshData data)
        {
            InvalidatePendingReadbacks();
            EnsureReadbackBuffers();

            // 浣嶇疆
            _bufPos.GetData(_readBuf);
            for (int i = 0; i < _numP; i++)
                data.Positions[i] = new Vector3(_readBuf[i].x, _readBuf[i].y, _readBuf[i].z);

            // 閫熷害
            _bufVel.GetData(_velReadBuf);
            for (int i = 0; i < _numP; i++)
                data.Velocities[i] = new Vector3(_velReadBuf[i].x, _velReadBuf[i].y, _velReadBuf[i].z);

            // 涓婁竴甯т綅缃?            _bufPrevPos.GetData(_prevReadBuf);
            for (int i = 0; i < _numP; i++)
                data.PrevPositions[i] = new Vector3(_prevReadBuf[i].x, _prevReadBuf[i].y, _prevReadBuf[i].z);
        }

        void EnsureReadbackBuffers()
        {
            if (_readBuf == null || _readBuf.Length != _numP)
                _readBuf = new Vector4[_numP];
            if (_velReadBuf == null || _velReadBuf.Length != _numP)
                _velReadBuf = new Vector4[_numP];
            if (_prevReadBuf == null || _prevReadBuf.Length != _numP)
                _prevReadBuf = new Vector4[_numP];
        }

        void InvalidatePendingReadbacks()
        {
            unchecked { _readbackGeneration++; }
            _positionReadbackPending = false;
            _toolMetricsReadbackPending = false;
        }

        public void UploadPositionsAndVelocities(Core.TetMeshData data)
        {
            var posV4  = ToV4(data.Positions, _numP);
            var velV4  = ToV4(data.Velocities, _numP);
            var prevV4 = ToV4(data.PrevPositions, _numP);
            _bufPos.SetData(posV4);
            _bufVel.SetData(velV4);
            _bufPrevPos.SetData(prevV4);
        }

        /// <summary>
        /// 浠呬笂浼犳寚瀹氱矑瀛愮殑鐘舵€侊紝閬垮厤澶瑰彇绾︽潫鐢ㄦ棫鐨?CPU 鏁版嵁瑕嗙洊澶栧洿鍔ㄦ€佺矑瀛愩€?        /// particleIndices 蹇呴』鎸夊崌搴忔帓鍒椾笖涓嶈兘閲嶅銆?        /// </summary>
        public void UploadParticleStates(Core.TetMeshData data, IList<int> particleIndices)
        {
            if (data == null || particleIndices == null || particleIndices.Count == 0 ||
                _bufPos == null || _bufVel == null || _bufPrevPos == null)
                return;

            int count = particleIndices.Count;
            EnsureSparseUploadBuffers(count);

            int previousIndex = -1;
            for (int i = 0; i < count; i++)
            {
                int particleIndex = particleIndices[i];
                if (particleIndex < 0 || particleIndex >= _numP || particleIndex <= previousIndex)
                    throw new ArgumentException(
                        "particleIndices must contain unique, ascending particle indices.",
                        nameof(particleIndices));

                Vector3 p = data.Positions[particleIndex];
                Vector3 v = data.Velocities[particleIndex];
                Vector3 prev = data.PrevPositions[particleIndex];
                _sparsePosUpload[i] = new Vector4(p.x, p.y, p.z, 0f);
                _sparseVelUpload[i] = new Vector4(v.x, v.y, v.z, 0f);
                _sparsePrevUpload[i] = new Vector4(prev.x, prev.y, prev.z, 0f);
                previousIndex = particleIndex;
            }

            int runStart = 0;
            while (runStart < count)
            {
                int gpuStart = particleIndices[runStart];
                int runEnd = runStart + 1;
                while (runEnd < count &&
                       particleIndices[runEnd] == gpuStart + (runEnd - runStart))
                {
                    runEnd++;
                }

                int runLength = runEnd - runStart;
                _bufPos.SetData(_sparsePosUpload, runStart, gpuStart, runLength);
                _bufVel.SetData(_sparseVelUpload, runStart, gpuStart, runLength);
                _bufPrevPos.SetData(_sparsePrevUpload, runStart, gpuStart, runLength);
                runStart = runEnd;
            }
        }

        void EnsureSparseUploadBuffers(int count)
        {
            if (_sparsePosUpload != null && _sparsePosUpload.Length >= count)
                return;

            _sparsePosUpload = new Vector4[count];
            _sparseVelUpload = new Vector4[count];
            _sparsePrevUpload = new Vector4[count];
        }

        public ComputeBuffer GetPositionsBuffer() => _bufPos;

        /// <summary>
        /// 涓婁紶 InvMass 鍒?GPU锛堢敤浜庡す鍙栨椂閿佸畾/瑙ｉ攣绮掑瓙锛?        /// 灏?data.InvMass 鐩存帴鍐欏叆 GPU buffer锛屾棤闇€瀹屾暣 Dispose+Init
        /// </summary>
        public void UploadInvMass(Core.TetMeshData data)
        {
            if (_bufInvMass == null) return;
            var invM = new float[_numP];
            int count = Mathf.Min(_numP, data.NumParticles);
            for (int i = 0; i < count; i++)
                invM[i] = data.InvMass[i];
            _bufInvMass.SetData(invM);
        }

        /// <summary>
        /// 璁剧疆鑳跺泭浣撶鎾炲弬鏁帮紙姣忓抚璋冪敤锛屽湪 Step 涔嬪墠锛?        /// 鏃х増 3 鑳跺泭鎺ュ彛锛涘唴閮ㄤ細杞埌鏁扮粍寮忓鑳跺泭鎺ュ彛
        /// 鍙傝€?SOFA collision pipeline
        /// </summary>
        public void SetCapsuleCollisionParams(
            Vector3 cap0A, Vector3 cap0B, float cap0R,
            Vector3 cap1A, Vector3 cap1B, float cap1R,
            Vector3 cap2A, Vector3 cap2B, float cap2R,
            Vector3 prevCap0A, Vector3 prevCap0B,
            Vector3 prevCap1A, Vector3 prevCap1B,
            Vector3 prevCap2A, Vector3 prevCap2B,
            int numCapsules)
        {
            _legacyCapsuleA[0] = cap0A;
            _legacyCapsuleB[0] = cap0B;
            _legacyCapsuleR[0] = cap0R;
            _legacyPrevCapsuleA[0] = prevCap0A;
            _legacyPrevCapsuleB[0] = prevCap0B;

            _legacyCapsuleA[1] = cap1A;
            _legacyCapsuleB[1] = cap1B;
            _legacyCapsuleR[1] = cap1R;
            _legacyPrevCapsuleA[1] = prevCap1A;
            _legacyPrevCapsuleB[1] = prevCap1B;

            _legacyCapsuleA[2] = cap2A;
            _legacyCapsuleB[2] = cap2B;
            _legacyCapsuleR[2] = cap2R;
            _legacyPrevCapsuleA[2] = prevCap2A;
            _legacyPrevCapsuleB[2] = prevCap2B;

            SetCapsuleCollisionParams(
                _legacyCapsuleA,
                _legacyCapsuleB,
                _legacyCapsuleR,
                _legacyPrevCapsuleA,
                _legacyPrevCapsuleB,
                Mathf.Min(numCapsules, 3));
        }

        public void SetCapsuleCollisionParams(
            Vector3[] capsuleA,
            Vector3[] capsuleB,
            float[] capsuleR,
            Vector3[] prevCapsuleA,
            Vector3[] prevCapsuleB,
            int numCapsules,
            float[] capsuleFriction = null)
        {
            _activeToolCapsules = Mathf.Clamp(numCapsules, 0, MaxToolCapsules);
            if (_activeToolCapsules <= 0)
            {
                ClearToolCollisionParams();
                return;
            }

            Vector3 bboxMin = Vector3.positiveInfinity;
            Vector3 bboxMax = Vector3.negativeInfinity;
            for (int i = 0; i < MaxToolCapsules; i++)
            {
                bool active = i < _activeToolCapsules;
                Vector3 a = active && capsuleA != null && i < capsuleA.Length
                    ? capsuleA[i]
                    : Vector3.zero;
                Vector3 b = active && capsuleB != null && i < capsuleB.Length
                    ? capsuleB[i]
                    : Vector3.zero;
                Vector3 prevA = active && prevCapsuleA != null && i < prevCapsuleA.Length
                    ? prevCapsuleA[i]
                    : a;
                Vector3 prevB = active && prevCapsuleB != null && i < prevCapsuleB.Length
                    ? prevCapsuleB[i]
                    : b;
                float r = active && capsuleR != null && i < capsuleR.Length
                    ? Mathf.Max(0f, capsuleR[i])
                    : 0f;
                float friction = active
                    ? capsuleFriction != null && i < capsuleFriction.Length
                        ? Mathf.Max(0f, capsuleFriction[i])
                        : 1f
                    : 0f;

                _capsuleAUpload[i] = new Vector4(a.x, a.y, a.z, r);
                _capsuleBUpload[i] = new Vector4(b.x, b.y, b.z, 0f);
                _prevCapsuleAUpload[i] = new Vector4(prevA.x, prevA.y, prevA.z, r);
                _prevCapsuleBUpload[i] = new Vector4(prevB.x, prevB.y, prevB.z, 0f);
                _capsuleFrictionUpload[i] = friction;

                if (!active) continue;

                SetCapsuleBounds(i, a, b, prevA, prevB, r);
                IncludeCapsuleBounds(ref bboxMin, ref bboxMax, a, b, prevA, prevB, r);
            }

            _cs.SetVectorArray("_CapsuleA", _capsuleAUpload);
            _cs.SetVectorArray("_CapsuleB", _capsuleBUpload);
            _cs.SetVectorArray("_PrevCapsuleA", _prevCapsuleAUpload);
            _cs.SetVectorArray("_PrevCapsuleB", _prevCapsuleBUpload);
            _cs.SetFloats("_CapsuleFriction", _capsuleFrictionUpload);
            _cs.SetVector("_Capsule0A", _capsuleAUpload[0]);
            _cs.SetVector("_Capsule0B", _capsuleBUpload[0]);
            _cs.SetVector("_Capsule1A", _capsuleAUpload[1]);
            _cs.SetVector("_Capsule1B", _capsuleBUpload[1]);
            _cs.SetVector("_Capsule2A", _capsuleAUpload[2]);
            _cs.SetVector("_Capsule2B", _capsuleBUpload[2]);
            _cs.SetVector("_PrevCapsule0A", _prevCapsuleAUpload[0]);
            _cs.SetVector("_PrevCapsule0B", _prevCapsuleBUpload[0]);
            _cs.SetVector("_PrevCapsule1A", _prevCapsuleAUpload[1]);
            _cs.SetVector("_PrevCapsule1B", _prevCapsuleBUpload[1]);
            _cs.SetVector("_PrevCapsule2A", _prevCapsuleAUpload[2]);
            _cs.SetVector("_PrevCapsule2B", _prevCapsuleBUpload[2]);
            _cs.SetVector("_ToolBBoxMin", new Vector4(bboxMin.x, bboxMin.y, bboxMin.z, 0f));
            _cs.SetVector("_ToolBBoxMax", new Vector4(bboxMax.x, bboxMax.y, bboxMax.z, _activeToolCapsules));
        }

        void SetCapsuleBounds(int index, Vector3 a, Vector3 b, Vector3 prevA, Vector3 prevB, float radius)
        {
            float pad = Mathf.Max(0f, radius) + Mathf.Max(0f, ToolContactDistance);
            Vector3 padVec = Vector3.one * pad;
            _toolCapsuleBboxMin[index] = Vector3.Min(Vector3.Min(a, b), Vector3.Min(prevA, prevB)) - padVec;
            _toolCapsuleBboxMax[index] = Vector3.Max(Vector3.Max(a, b), Vector3.Max(prevA, prevB)) + padVec;
        }

        void IncludeCapsuleBounds(
            ref Vector3 bboxMin, ref Vector3 bboxMax,
            Vector3 a, Vector3 b, Vector3 prevA, Vector3 prevB, float radius)
        {
            float pad = Mathf.Max(0f, radius) + Mathf.Max(0f, ToolContactDistance);
            Vector3 padVec = Vector3.one * pad;
            Vector3 localMin = Vector3.Min(Vector3.Min(a, b), Vector3.Min(prevA, prevB)) - padVec;
            Vector3 localMax = Vector3.Max(Vector3.Max(a, b), Vector3.Max(prevA, prevB)) + padVec;
            bboxMin = Vector3.Min(bboxMin, localMin);
            bboxMax = Vector3.Max(bboxMax, localMax);
        }

        public void ClearToolCollisionParams()
        {
            _activeToolCapsules = 0;
            ClearToolContactCandidates();
            for (int i = 0; i < MaxToolCapsules; i++)
                _capsuleFrictionUpload[i] = 0f;
            _cs.SetFloats("_CapsuleFriction", _capsuleFrictionUpload);
            _cs.SetVector("_ToolBBoxMin", Vector4.zero);
            _cs.SetVector("_ToolBBoxMax", Vector4.zero);
        }

        public void ClearToolContactCandidates()
        {
            _activeToolSurfaceCandidateTris = 0;
            _activeToolSurfaceCandidateColors = 0;
            _toolCandidatesValid = false;
            if (_toolCandidateGroupCnt != null)
                Array.Clear(_toolCandidateGroupCnt, 0, _toolCandidateGroupCnt.Length);
        }

        public void UpdateToolContactCandidates(Core.TetMeshData data)
        {
            ClearToolContactCandidates();

            if (!UseToolContactCandidateCulling ||
                _activeToolCapsules <= 0 ||
                data == null ||
                data.SurfaceTriIds == null ||
                data.Positions == null ||
                _surfaceTriColorByTri == null ||
                _bufSurfaceTriCandidateFlat == null ||
                _toolCandidateSelected == null ||
                _toolCandidateFlat == null ||
                _toolCandidateGroupCnt == null ||
                _toolCandidateGroupOff == null ||
                _toolCandidateGroupCursor == null)
                return;

            Vector3 extraPad = Vector3.one * Mathf.Max(0f, ToolContactCandidatePadding);
            int total = 0;
            int maxTris = Mathf.Min(_numSurfaceTris, data.NumSurfaceTris);
            int[] triIds = data.SurfaceTriIds;
            Vector3[] positions = data.Positions;

            for (int tri = 0; tri < maxTris; tri++)
            {
                int baseIdx = tri * 3;
                int i0 = triIds[baseIdx + 0];
                int i1 = triIds[baseIdx + 1];
                int i2 = triIds[baseIdx + 2];
                if (i0 < 0 || i0 >= positions.Length ||
                    i1 < 0 || i1 >= positions.Length ||
                    i2 < 0 || i2 >= positions.Length)
                    continue;

                Vector3 p0 = positions[i0];
                Vector3 p1 = positions[i1];
                Vector3 p2 = positions[i2];
                Vector3 triMin = Vector3.Min(p0, Vector3.Min(p1, p2));
                Vector3 triMax = Vector3.Max(p0, Vector3.Max(p1, p2));

                if (!OverlapsAnyActiveToolCapsuleBounds(triMin, triMax, extraPad))
                    continue;

                int color = _surfaceTriColorByTri[tri];
                if (color < 0 || color >= _toolCandidateGroupCnt.Length)
                    continue;

                _toolCandidateSelected[total++] = tri;
                _toolCandidateGroupCnt[color]++;
            }

            if (total <= 0)
            {
                _toolCandidatesValid = true;
                return;
            }

            int offset = 0;
            int activeColors = 0;
            for (int c = 0; c < _numSurfaceColors; c++)
            {
                _toolCandidateGroupOff[c] = offset;
                _toolCandidateGroupCursor[c] = offset;
                int count = _toolCandidateGroupCnt[c];
                if (count > 0)
                    activeColors++;
                offset += count;
            }

            for (int i = 0; i < total; i++)
            {
                int tri = _toolCandidateSelected[i];
                int color = _surfaceTriColorByTri[tri];
                _toolCandidateFlat[_toolCandidateGroupCursor[color]++] = tri;
            }

            _bufSurfaceTriCandidateFlat.SetData(_toolCandidateFlat, 0, 0, total);
            _activeToolSurfaceCandidateTris = total;
            _activeToolSurfaceCandidateColors = activeColors;
            _toolCandidatesValid = true;
        }

        bool OverlapsAnyActiveToolCapsuleBounds(Vector3 triMin, Vector3 triMax, Vector3 extraPad)
        {
            for (int i = 0; i < _activeToolCapsules && i < MaxToolCapsules; i++)
            {
                Vector3 min = _toolCapsuleBboxMin[i] - extraPad;
                Vector3 max = _toolCapsuleBboxMax[i] + extraPad;
                if (!(triMax.x < min.x || triMin.x > max.x ||
                      triMax.y < min.y || triMin.y > max.y ||
                      triMax.z < min.z || triMin.z > max.z))
                    return true;
            }

            return false;
        }

        public void Dispose()
        {
            InvalidatePendingReadbacks();
            _bufPos?.Release(); _bufPrevPos?.Release(); _bufVel?.Release();
            _bufInvMass?.Release(); _bufTetIds?.Release(); _bufRestVol?.Release();
            _bufTetActive?.Release(); _bufEdgeIds?.Release(); _bufRestLen?.Release();
            _bufEdgeActive?.Release();
            _bufColorFlat?.Release(); _bufEdgeColorFlat?.Release();
            _bufSurfaceTriIds?.Release(); _bufSurfaceTriColorFlat?.Release();
            _bufSurfaceTriCandidateFlat?.Release();
            _bufToolContactMetrics?.Release();
            _bufInvRestMatrix?.Release();
            _bufAlphaDeviatoric?.Release(); _bufAlphaHydrostatic?.Release();
        }

        // 鈹€鈹€ 鍒囧壊鏀寔 API 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
        // 鈹€鈹€ 鍐呴儴鏂规硶 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€

        void BindAll()
        {
            int[] ks = { _kIntegrate, _kSolveEdges, _kSolveNHDev,
                         _kSolveNHHyd, _kSolveNHDevHyd, _kPostSolve, _kGroundCollision };
            foreach (int k in ks)
            {
                _cs.SetBuffer(k, "_Positions",     _bufPos);
                _cs.SetBuffer(k, "_PrevPositions", _bufPrevPos);
                _cs.SetBuffer(k, "_Velocities",    _bufVel);
                _cs.SetBuffer(k, "_InvMasses",     _bufInvMass);
                _cs.SetBuffer(k, "_TetIds",        _bufTetIds);
                _cs.SetBuffer(k, "_RestVolumes",   _bufRestVol);
                _cs.SetBuffer(k, "_TetActive",     _bufTetActive);
                _cs.SetBuffer(k, "_EdgeIds",       _bufEdgeIds);
                _cs.SetBuffer(k, "_RestLengths",   _bufRestLen);
                _cs.SetBuffer(k, "_ColorGroupFlat",    _bufColorFlat);
                _cs.SetBuffer(k, "_EdgeColorGroupFlat",_bufEdgeColorFlat);
            }
            _cs.SetBuffer(_kSolveEdges, "_EdgeActive", _bufEdgeActive);

            // NeoHookean buffers 鈥?only needed by NH kernels
            int[] nhKernels = { _kSolveNHDev, _kSolveNHHyd, _kSolveNHDevHyd };
            foreach (int k in nhKernels)
            {
                _cs.SetBuffer(k, "_InvRestMatrix",    _bufInvRestMatrix);
                _cs.SetBuffer(k, "_AlphaDeviatoric",  _bufAlphaDeviatoric);
                _cs.SetBuffer(k, "_AlphaHydrostatic", _bufAlphaHydrostatic);
            }

            // Tool collision 鈥?闇€瑕?_Positions, _PrevPositions(閫熷害淇), _InvMasses
            _cs.SetBuffer(_kToolCollision, "_Positions", _bufPos);
            _cs.SetBuffer(_kToolCollision, "_PrevPositions", _bufPrevPos);
            _cs.SetBuffer(_kToolCollision, "_InvMasses", _bufInvMass);

            if (_bufSurfaceTriIds != null && _bufSurfaceTriColorFlat != null)
            {
                _cs.SetBuffer(_kSolveSurfaceToolContacts, "_Positions", _bufPos);
                _cs.SetBuffer(_kSolveSurfaceToolContacts, "_PrevPositions", _bufPrevPos);
                _cs.SetBuffer(_kSolveSurfaceToolContacts, "_InvMasses", _bufInvMass);
                _cs.SetBuffer(_kSolveSurfaceToolContacts, "_SurfaceTriIds", _bufSurfaceTriIds);
                _cs.SetBuffer(_kSolveSurfaceToolContacts, "_SurfaceTriColorGroupFlat", _bufSurfaceTriColorFlat);
            }

            if (_bufToolContactMetrics != null)
            {
                _cs.SetBuffer(_kClearToolContactMetrics, "_ToolContactMetrics", _bufToolContactMetrics);
                _cs.SetBuffer(_kToolContactMetrics, "_Positions", _bufPos);
                _cs.SetBuffer(_kToolContactMetrics, "_InvMasses", _bufInvMass);
                _cs.SetBuffer(_kToolContactMetrics, "_ToolContactMetrics", _bufToolContactMetrics);
                if (_bufSurfaceTriIds != null && _bufSurfaceTriColorFlat != null)
                {
                    _cs.SetBuffer(_kToolContactMetrics, "_SurfaceTriIds", _bufSurfaceTriIds);
                    _cs.SetBuffer(_kToolContactMetrics, "_SurfaceTriColorGroupFlat", _bufSurfaceTriColorFlat);
                }
            }
        }

        void Go(int k, int n)
        {
            int g = (n + TH - 1) / TH;
            if (g > 0) _cs.Dispatch(k, g, 1, 1);
        }

        void BuildEdges(Core.TetMeshData data, Vector3[] pos,
            out int[] edgeIds, out float[] restLengths)
        {
            var edgeSet = new Dictionary<long, int>();
            var eidList = new List<int>();
            var rlList  = new List<float>();
            var e2tList = new List<List<int>>();

            int[,] pairs = {{0,1},{0,2},{0,3},{1,2},{1,3},{2,3}};
            for (int t = 0; t < _numT; t++)
            {
                if (!data.TetActive[t]) continue;
                for (int p2 = 0; p2 < 6; p2++)
                {
                    int a = data.TetIds[t*4+pairs[p2,0]], b = data.TetIds[t*4+pairs[p2,1]];
                    if (a > b) { int tmp=a; a=b; b=tmp; }
                    long key = (long)a * 200000 + b;
                    if (!edgeSet.ContainsKey(key))
                    {
                        int ei = eidList.Count / 2;
                        edgeSet[key] = ei;
                        eidList.Add(a); eidList.Add(b);
                        rlList.Add((pos[a] - pos[b]).magnitude);
                        e2tList.Add(new List<int> { t });
                    }
                    else
                    {
                        e2tList[edgeSet[key]].Add(t);
                    }
                }
            }
            edgeIds = eidList.ToArray();
            restLengths = rlList.ToArray();
            _edgeToTets = new List<int>[e2tList.Count];
            for (int i = 0; i < e2tList.Count; i++)
                _edgeToTets[i] = e2tList[i];
        }

        void BuildTetColorGroups(Core.TetMeshData data,
            out int[] flat, out int[] offsets, out int[] counts, out int numColors)
        {
            var groups = Core.GraphColoring.Compute(data.TetIds, data.NumTets, data.NumParticles, data.TetActive);
            numColors = groups.Count;
            offsets = new int[numColors]; counts = new int[numColors];
            int total = 0;
            for (int i = 0; i < numColors; i++) total += groups[i].Length;
            flat = new int[total];
            int off = 0;
            for (int i = 0; i < numColors; i++)
            {
                offsets[i] = off; counts[i] = groups[i].Length;
                Array.Copy(groups[i], 0, flat, off, groups[i].Length);
                off += groups[i].Length;
            }
        }

        void BuildEdgeColorGroups(int[] edgeIds, int numEdges, int numParticles,
            out int[] flat, out int[] offsets, out int[] counts, out int numColors)
        {
            int[] color = new int[numEdges];
            Array.Fill(color, -1);
            var vertToEdges = new List<int>[numParticles];
            for (int i = 0; i < numParticles; i++) vertToEdges[i] = new List<int>(8);
            for (int e = 0; e < numEdges; e++)
            {
                vertToEdges[edgeIds[e*2]].Add(e);
                vertToEdges[edgeIds[e*2+1]].Add(e);
            }
            int maxColor = 0;
            for (int e = 0; e < numEdges; e++)
            {
                int a = edgeIds[e*2], b = edgeIds[e*2+1];
                var used = new HashSet<int>();
                foreach (int n in vertToEdges[a]) if (color[n] >= 0) used.Add(color[n]);
                foreach (int n in vertToEdges[b]) if (color[n] >= 0) used.Add(color[n]);
                int c2 = 0; while (used.Contains(c2)) c2++;
                color[e] = c2;
                if (c2 > maxColor) maxColor = c2;
            }
            numColors = maxColor + 1;
            var groups = new List<int>[numColors];
            for (int i = 0; i < numColors; i++) groups[i] = new List<int>();
            for (int e = 0; e < numEdges; e++) groups[color[e]].Add(e);
            offsets = new int[numColors]; counts = new int[numColors];
            int total = 0;
            for (int i = 0; i < numColors; i++) total += groups[i].Count;
            flat = new int[total]; int off = 0;
            for (int i = 0; i < numColors; i++)
            {
                offsets[i] = off; counts[i] = groups[i].Count;
                groups[i].CopyTo(flat, off);
                off += groups[i].Count;
            }
        }

        // 鈹€鈹€ Buffer helpers 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
        void BuildSurfaceTriColorGroups(int[] triIds, int numTris, int numParticles,
            out int[] flat, out int[] offsets, out int[] counts, out int[] colorByTri, out int numColors)
        {
            if (triIds == null || numTris <= 0)
            {
                flat = Array.Empty<int>();
                offsets = Array.Empty<int>();
                counts = Array.Empty<int>();
                colorByTri = Array.Empty<int>();
                numColors = 0;
                return;
            }

            int[] color = new int[numTris];
            Array.Fill(color, -1);
            var vertToTris = new List<int>[numParticles];
            for (int i = 0; i < numParticles; i++) vertToTris[i] = new List<int>(8);

            for (int t = 0; t < numTris; t++)
            {
                vertToTris[triIds[t * 3 + 0]].Add(t);
                vertToTris[triIds[t * 3 + 1]].Add(t);
                vertToTris[triIds[t * 3 + 2]].Add(t);
            }

            int maxColor = 0;
            for (int t = 0; t < numTris; t++)
            {
                int a = triIds[t * 3 + 0];
                int b = triIds[t * 3 + 1];
                int c = triIds[t * 3 + 2];
                var used = new HashSet<int>();
                foreach (int n in vertToTris[a]) if (color[n] >= 0) used.Add(color[n]);
                foreach (int n in vertToTris[b]) if (color[n] >= 0) used.Add(color[n]);
                foreach (int n in vertToTris[c]) if (color[n] >= 0) used.Add(color[n]);
                int chosen = 0; while (used.Contains(chosen)) chosen++;
                color[t] = chosen;
                if (chosen > maxColor) maxColor = chosen;
            }

            numColors = maxColor + 1;
            var groups = new List<int>[numColors];
            for (int i = 0; i < numColors; i++) groups[i] = new List<int>();
            for (int t = 0; t < numTris; t++) groups[color[t]].Add(t);

            offsets = new int[numColors];
            counts = new int[numColors];
            int total = 0;
            for (int i = 0; i < numColors; i++) total += groups[i].Count;
            flat = new int[total];

            int off = 0;
            for (int i = 0; i < numColors; i++)
            {
                offsets[i] = off;
                counts[i] = groups[i].Count;
                groups[i].CopyTo(flat, off);
                off += groups[i].Count;
            }

            colorByTri = color;
        }

        ComputeBuffer MkV4(Vector4[] d) { var b=new ComputeBuffer(d.Length,16); b.SetData(d); return b; }
        ComputeBuffer MkF(float[] d)    { var b=new ComputeBuffer(d.Length,4);  b.SetData(d); return b; }
        ComputeBuffer MkI(int[] d)      { var b=new ComputeBuffer(d.Length,4);  b.SetData(d); return b; }
        ComputeBuffer MkU4(UInt4[] d)   { var b=new ComputeBuffer(d.Length,16); b.SetData(d); return b; }
        ComputeBuffer MkU2(UInt2[] d)   { var b=new ComputeBuffer(d.Length,8);  b.SetData(d); return b; }
        ComputeBuffer MkU3(UInt3[] d)   { var b=new ComputeBuffer(d.Length,12); b.SetData(d); return b; }

        static Vector4[] ToV4(Vector3[] v, int count) {
            var r=new Vector4[count];
            for(int i=0;i<count;i++) r[i]=new Vector4(v[i].x,v[i].y,v[i].z,0);
            return r;
        }
        static UInt4[] ToU4(int[] f,int n) {
            var a=new UInt4[n];
            for(int i=0;i<n;i++) a[i]=new UInt4{x=(uint)f[i*4],y=(uint)f[i*4+1],z=(uint)f[i*4+2],w=(uint)f[i*4+3]};
            return a;
        }
        static UInt2[] ToU2(int[] f,int n) {
            var a=new UInt2[n];
            for(int i=0;i<n;i++) a[i]=new UInt2{x=(uint)f[i*2],y=(uint)f[i*2+1]};
            return a;
        }
        static UInt3[] ToU3(int[] f,int n) {
            var a=new UInt3[n];
            for(int i=0;i<n;i++) a[i]=new UInt3{x=(uint)f[i*3],y=(uint)f[i*3+1],z=(uint)f[i*3+2]};
            return a;
        }
    }
}
