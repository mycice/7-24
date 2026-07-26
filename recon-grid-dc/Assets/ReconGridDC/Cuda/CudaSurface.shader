// CudaSurface.shader — minimal double-sided lit material for the Stage 1 copy-based liver mesh.
// Built-in pipeline surface shader. Cull Off so the closed DC surface is visible regardless of
// per-triangle winding (a validation-gate convenience; the ported DC_Stitch winding is outward).
Shader "ReconGridDC/CudaSurface"
{
    Properties
    {
        _Color ("Color", Color) = (0.80, 0.32, 0.30, 1)
    }
    SubShader
    {
        Tags { "RenderType" = "Opaque" }
        Cull Off

        CGPROGRAM
        #pragma surface surf Lambert
        fixed4 _Color;
        struct Input { float3 worldPos; };
        void surf (Input IN, inout SurfaceOutput o)
        {
            o.Albedo = _Color.rgb;
        }
        ENDCG
    }
    Fallback "Diffuse"
}
