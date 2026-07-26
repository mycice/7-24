Shader "ReconGridDC/LiverCutInterior"
{
    Properties
    {
        [MainColor] _Color ("Cut Color", Color) = (0.30, 0.025, 0.018, 1)
        _BaseColor ("Cut Color Alias", Color) = (0.30, 0.025, 0.018, 1)
        _WetSpecColor ("Wet Specular Tint", Color) = (0.95, 0.78, 0.68, 1)
        _SSSColor ("Subsurface Tint", Color) = (0.76, 0.13, 0.055, 1)
        _Smoothness ("Smoothness", Range(0, 1)) = 0.58
        _Glossiness ("Glossiness Alias", Range(0, 1)) = 0.58
        _SpecularStrength ("Specular Strength", Range(0, 2)) = 0.65
        _FresnelStrength ("Fresnel Strength", Range(0, 1)) = 0.22
        _FresnelPow ("Fresnel Power", Range(1, 8)) = 4
        _Wetness ("Wetness", Range(0, 1)) = 0.75
    }

    SubShader
    {
        Tags { "RenderType"="Opaque" "Queue"="Geometry" }
        LOD 250
        Cull Off

        CGPROGRAM
        #pragma surface surf Standard fullforwardshadows addshadow
        #pragma target 3.0

        fixed4 _Color;
        fixed4 _BaseColor;
        fixed4 _WetSpecColor;
        fixed4 _SSSColor;
        half _Smoothness;
        half _Glossiness;
        half _SpecularStrength;
        half _FresnelStrength;
        half _FresnelPow;
        half _Wetness;

        struct Input
        {
            float3 worldPos;
            float3 worldNormal;
        };

        void surf(Input IN, inout SurfaceOutputStandard o)
        {
            half3 baseColor = lerp(_Color.rgb, _BaseColor.rgb, 0.5h);
            half3 viewDir = normalize(_WorldSpaceCameraPos.xyz - IN.worldPos);
            half ndv = saturate(abs(dot(normalize(IN.worldNormal), viewDir)));
            half fresnel = pow(1.0h - ndv, _FresnelPow) * _FresnelStrength;

            o.Albedo = baseColor;
            o.Metallic = 0.0h;
            o.Smoothness = saturate(max(_Smoothness, _Glossiness) + _Wetness * 0.08h);
            o.Occlusion = 1.0h;
            o.Emission = _WetSpecColor.rgb * fresnel * _SpecularStrength * 0.22h +
                         _SSSColor.rgb * _Wetness * 0.035h;
            o.Alpha = 1.0h;
        }
        ENDCG
    }

    FallBack "Standard"
}
