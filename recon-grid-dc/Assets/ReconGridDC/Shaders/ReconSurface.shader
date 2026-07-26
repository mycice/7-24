// ReconSurface.shader — Task 10 Step 1 (Stage 2C tag-aware — plan §3.7)
// Procedural surface shader for the DC outer surface + cut walls.
// Used with Graphics.DrawProceduralIndirect (MeshTopology.Triangles).
//
// Vertex: indexes _Tri[SV_VertexID], decodes the tag (TWO-WAY) — no high bit ⇒ external
//         (idx=voxelId → _VoxelExternalFP / _ExtNormalF); bit31 SET ⇒ cut FP (idx=8*voxelId+comp
//         → _CutFP / _CutFPNormalF — the per-component centroid shared by the cut SKIN and WALL).
// Render: Cull Off + two-sided fragment normal (SV_IsFrontFace) so the cut walls and back faces
//         are lit on both sides — face winding is DISPLAY-SIGN-AGNOSTIC (A1-C2). ZERO-CUT path:
//         all tags external → identical pixels to Stage-1 (the cut buffers hold zeros, never sampled).
Shader "ReconGridDC/ReconSurface"
{
    SubShader
    {
        Tags { "RenderType"="Opaque" }
        Cull Off            // 2C: two-sided — cut walls + outer surface both visible (plan §3.7 / A1-C2)
        ZTest LEqual
        ZWrite On

        Pass
        {
            HLSLPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            #pragma target 5.0

            #include "UnityCG.cginc"

            // ── GPU buffers written by DualContouring.Build ──────────────────────
            StructuredBuffer<int>    _Tri;             // 3*TriCapacity int tagged indices
            StructuredBuffer<float4> _VoxelExternalFP; // xyz=external feature point, w=valid
            StructuredBuffer<float3> _ExtNormalF;       // per-voxel normalized external normal
            // ── Stage 2C: internal cut-FP buffers (bound unconditionally by ReconGridManager) ──
            StructuredBuffer<float4> _CutFP;            // float4[8*VoxelCount] cut feature points (skin+wall shared)
            StructuredBuffer<float3> _CutFPNormalF;     // float3[8*VoxelCount] normalized cut-FP normals

            // DEBUG (delamination diagnosis 2026-06-29): when _DebugTag>0.5 the fragment is colored by which
            // surface buffer the vertex came from — RED = cut feature point (_CutFP, the per-component centroid
            // shared by the cut SKIN and WALL), GRAY = normal body surface (_VoxelExternalFP). Lets us see
            // DIRECTLY whether the bright sliver is the cut surface or the (fallen) lower-half body. Set
            // ReconGridManager.debugTagColor in the Inspector.
            float _DebugTag;

            struct v2f
            {
                float4 pos : SV_POSITION;
                float3 n   : TEXCOORD0;
                float3 dbg : TEXCOORD1;   // debug tag color (see _DebugTag)
            };

            v2f vert(uint vid : SV_VertexID)
            {
                // Decode tag (TWO-WAY, mask 0x3FFFFFFF strips bit31; the bit30 External-cut route was
                // removed with _CutFPExternal):
                //   bit31 (0x80000000) ⇒ cut FP (skin+wall shared) → _CutFP / _CutFPNormalF (idx=8*v+comp).
                //   no high bit         ⇒ per-voxel external surface → _VoxelExternalFP / _ExtNormalF (idx=voxelId).
                uint u   = (uint)_Tri[vid];
                int  idx = (int)(u & 0x3FFFFFFFu);

                float3 wp, n;
                if (u & 0x80000000u) { wp = _CutFP[idx].xyz;          n = _CutFPNormalF[idx]; }   // cut FP (skin+wall)
                else                 { wp = _VoxelExternalFP[idx].xyz; n = _ExtNormalF[idx]; }      // normal surface

                v2f o;
                o.pos = UnityObjectToClipPos(float4(wp, 1.0));
                o.n   = n;
                // DEBUG tag color: RED = cut surface (skin+wall, bit31), GRAY = body (no high bit).
                if (u & 0x80000000u) o.dbg = float3(1.0, 0.15, 0.15);  // cut feature point (skin + wall)
                else                 o.dbg = float3(0.6, 0.6, 0.6);     // normal body surface
                return o;
            }

            // Two-sided: flip the interpolated normal toward the viewer for back faces so the cut
            // walls (and any back-facing outer geometry) are lit, never black (plan §3.7 / R-B).
            fixed4 frag(v2f i, bool front : SV_IsFrontFace) : SV_Target
            {
                float3 n = front ? i.n : -i.n;       // two-sided lit (cut walls + back faces)
                // Simple Lambert with a single directional light and ambient floor
                float3 lightDir = normalize(float3(0.3, 1.0, 0.2));
                float ndl = saturate(dot(normalize(n), lightDir)) * 0.8 + 0.2;
                // DEBUG: tag-colored (× shading so shape stays readable) when enabled, else grayscale lit.
                if (_DebugTag > 0.5) return fixed4(i.dbg * ndl, 1.0);
                return fixed4(ndl.xxx, 1.0);
            }
            ENDHLSL
        }
    }
}
