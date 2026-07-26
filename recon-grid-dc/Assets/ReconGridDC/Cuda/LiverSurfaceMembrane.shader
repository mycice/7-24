Shader "ReconGridDC/LiverSurfaceMembrane"
{
    Properties
    {
        _Color ("Membrane Color", Color) = (1.0, 0.97, 0.72, 0.05)
        _Opacity ("Overall Opacity", Range(0, 1)) = 0.05
        _NormalOffset ("Normal Offset", Float) = 0.015
        _RimStrength ("Rim Strength", Range(0, 2)) = 0.65
        _RimPower ("Rim Power", Range(0.5, 8)) = 2.5
    }

    SubShader
    {
        Tags { "Queue"="Transparent+10" "RenderType"="Transparent" "IgnoreProjector"="True" }
        Cull Off
        ZWrite Off
        Blend SrcAlpha OneMinusSrcAlpha

        CGPROGRAM
        #pragma surface surf Standard alpha:fade vertex:vert noshadow
        #pragma target 3.0

        fixed4 _Color;
        half _Opacity;
        float _NormalOffset;
        half _RimStrength;
        half _RimPower;

        struct Input
        {
            float3 viewDir;
        };

        void vert(inout appdata_full v)
        {
            v.vertex.xyz += v.normal * _NormalOffset;
        }

        void surf(Input IN, inout SurfaceOutputStandard o)
        {
            half3 viewDir = normalize(IN.viewDir);
            half rim = pow(1.0h - saturate(dot(viewDir, o.Normal)), _RimPower);
            o.Albedo = _Color.rgb;
            o.Emission = _Color.rgb * rim * _RimStrength;
            o.Metallic = 0.0h;
            o.Smoothness = 0.82h;
            o.Alpha = saturate(_Opacity * (0.72h + 0.55h * rim));
        }
        ENDCG
    }

    Fallback "Transparent/Diffuse"
}
