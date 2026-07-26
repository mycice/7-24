// Built-in approximation of Simulation's Runtime_Project2_Liver2 URP/Lit material.
// Project2 binds liver2_spec.png as a regular RGB detail/normal texture, so this
// shader decodes RGB manually instead of using Unity's DXT5nm UnpackNormal path.

Shader "ReconGridDC/Project2LiverSurface"
{
    Properties
    {
        [MainTexture] _MainTex ("Base Map (liver2)", 2D) = "white" {}
        _BaseMap ("Base Map Alias", 2D) = "white" {}
        [MainColor] _Color ("Base Color", Color) = (0.7205882, 0.7205882, 0.7205882, 1)
        _BaseColor ("Base Color Alias", Color) = (0.7205882, 0.7205882, 0.7205882, 1)
        _InteriorColor ("Fallback Cut Interior", Color) = (0.30, 0.025, 0.018, 1)

        _BumpMap ("Project2 RGB Normal (liver2_spec)", 2D) = "gray" {}
        _BumpScale ("RGB Normal Scale", Range(0, 2)) = 1.03
        _ParallaxMap ("Height Map (liver2_height)", 2D) = "gray" {}
        _Parallax ("Parallax", Range(0, 0.12)) = 0.08

        _Metallic ("Metallic", Range(0, 1)) = 0.184
        _Smoothness ("Smoothness", Range(0, 1)) = 0.939
        _Glossiness ("Glossiness Alias", Range(0, 1)) = 0.939
        _CutSmoothness ("Fallback Cut Smoothness", Range(0, 1)) = 0.22

        _AlbedoBrightness ("Albedo Brightness", Range(0.2, 1.5)) = 0.82
        _TextureContrast ("Texture Contrast", Range(0.5, 2.5)) = 1.08
        _TextureTilingOffset ("Texture Tiling Offset", Vector) = (2, 2, 0.15, 0)
        _WetSpecColor ("Wet Specular Tint", Color) = (0.95, 0.90, 0.85, 1)
        _FresnelStrength ("Fresnel Strength", Range(0, 1)) = 0.16
        _FresnelPow ("Fresnel Power", Range(1, 8)) = 4.2
        _SSSColor ("Subsurface Tint", Color) = (0.76, 0.13, 0.055, 1)
        _SSSStrength ("Subsurface Strength", Range(0, 1)) = 0.06
        _SSSWrap ("Wrap Diffuse Approx", Range(0, 1)) = 0.32
        _EnvReflection ("Env Reflection Boost", Range(0, 1)) = 0.18
        _DebugTextureOnly ("Debug Texture Only", Range(0, 1)) = 0
    }

    SubShader
    {
        Tags { "RenderType"="Opaque" "Queue"="Geometry" }
        LOD 300
        Cull Off

        CGPROGRAM
        #pragma surface surf Standard fullforwardshadows addshadow vertex:vert
        #pragma target 3.0
        #include "UnityCG.cginc"

        sampler2D _MainTex;
        sampler2D _BumpMap;
        sampler2D _ParallaxMap;

        fixed4 _Color;
        fixed4 _BaseColor;
        fixed4 _InteriorColor;
        fixed4 _WetSpecColor;
        fixed4 _SSSColor;
        half _BumpScale;
        half _Parallax;
        half _Metallic;
        half _Smoothness;
        half _Glossiness;
        half _CutSmoothness;
        half _AlbedoBrightness;
        half _TextureContrast;
        float4 _TextureTilingOffset;
        half _FresnelStrength;
        half _FresnelPow;
        half _SSSStrength;
        half _SSSWrap;
        half _EnvReflection;
        half _DebugTextureOnly;

        struct appdata_project2
        {
            float4 vertex   : POSITION;
            float3 normal   : NORMAL;
            float4 tangent  : TANGENT;
            float2 texcoord : TEXCOORD0;
            float4 texcoord1: TEXCOORD1;
            float2 texcoord2: TEXCOORD2;
        };

        struct Input
        {
            float2 uv_MainTex;
            float3 viewDir;
            float4 restAux;
        };

        void vert(inout appdata_project2 v, out Input o)
        {
            UNITY_INITIALIZE_OUTPUT(Input, o);
            o.uv_MainTex = v.texcoord * _TextureTilingOffset.xy + _TextureTilingOffset.zw;
            o.restAux = v.texcoord1;
        }

        half3 DecodeProject2RgbNormal(float2 uv)
        {
            half3 n = tex2D(_BumpMap, uv).rgb * 2.0h - 1.0h;
            n.xy *= _BumpScale;
            n.z = max(n.z, 0.05h);
            return normalize(n);
        }

        half3 ApplyTextureTone(half3 c)
        {
            c = saturate((c - 0.5h) * _TextureContrast + 0.5h);
            return saturate(c * _AlbedoBrightness);
        }

        void surf(Input IN, inout SurfaceOutputStandard o)
        {
            half cut = saturate(IN.restAux.w);

            float2 uv = IN.uv_MainTex;
            half height = tex2D(_ParallaxMap, uv).r;
            uv += ParallaxOffset(height, _Parallax * (1.0h - cut), IN.viewDir);

            half3 tex = ApplyTextureTone(tex2D(_MainTex, uv).rgb);
            half3 project2Base = lerp(_Color.rgb, _BaseColor.rgb, 0.5h);
            half3 albedo = tex * project2Base;
            albedo = lerp(albedo, _InteriorColor.rgb, cut);

            half3 normalTS = DecodeProject2RgbNormal(uv);
            normalTS = normalize(lerp(normalTS, half3(0, 0, 1), cut));
            half3 viewTS = normalize(IN.viewDir);
            half ndv = saturate(dot(normalTS, viewTS));
            half fresnel = pow(1.0h - ndv, _FresnelPow) * _FresnelStrength * (1.0h - cut);
            half wrap = saturate((normalTS.z + _SSSWrap) / (1.0h + _SSSWrap));

            half smoothness = saturate(max(_Smoothness, _Glossiness));
            smoothness = lerp(smoothness, _CutSmoothness, cut);

            o.Albedo = lerp(albedo, tex, saturate(_DebugTextureOnly));
            o.Normal = normalTS;
            o.Metallic = _Metallic * (1.0h - cut);
            o.Smoothness = smoothness;
            o.Occlusion = 1.0h;
            o.Emission = _WetSpecColor.rgb * fresnel * _EnvReflection +
                         _SSSColor.rgb * _SSSStrength * wrap * (1.0h - cut) * 0.08h;
            o.Alpha = 1.0h;
        }
        ENDCG
    }

    FallBack "Standard"
}
