// LiverSurface.shader - Built-in pipeline port of the REAL reference shader
// SurgicalSim/LiverTissueGPU (URP), per the wf extraction (P1 exact params / P2 port spec).
//
// KEY reference facts honored here:
//   * LiverTissueGPU NEVER samples _NormalMap. ALL normal detail is object-space and
//     tangent-free: central differences of the triplanar _HeightMap (SampleTriplanarNormal,
//     eps = 0.004/s) blended with a procedural Noise3 normal (ProceduralFineNormal).
//   * Albedo law is TextureDrivenAlbedo: tissueTint (1.04,0.68,0.52), + base*0.08,
//     luminance grading (bypassed at runtime _TextureColorBlend=1), Noise3 dark veins,
//     contrast pivot (tex-0.5)*_TextureContrast+0.5, then *_AlbedoBrightness*mottleGain.
//   * Spec model: wet roughness lerp(_Roughness, max(0.16,_Roughness*0.72), _Wetness*spec),
//     f0 = lerp(0.028, _SpecularColor*0.08, _Wetness), specMask = lerp(0.45,1.15,specDetail),
//     GGX floors a=max(r*r,0.045), denominators 1e-6/1e-5 (reference-verbatim).
//   * SSS: wrapped diffuse divided by (1+wrap) ONCE; scatterTint = lerp(albedo,_SSSColor,0.45).
//   * Fresnel rim carries an extra *_Wetness; ambient = ShadeSH9 * albedo * 0.65.
//   * Double-sided = Cull Off + VFACE only (no ddx/ddy flip in the reference).
//   * _DebugTextureOnly = 1 shows the raw triplanar texture (pipeline diagnostic).
//
// OUR plumbing kept: UV1 = float4(rest anchor, wall flag f) drives the triplanar coords
// (reference restOS equivalent); fwdbase shadows + ShadowCaster pass; target 3.0.
// OUR one addition (design 2026-07-02 section 3.3): the cut-interior fold-in via f:
// albedo -> _InteriorColor (x mottleGain keeps procedural variation inside the wound),
// specular x(1-0.7f), fresnel x(1-f), SSS kept FULL on walls. f is BINARY PER TRIANGLE so
// the exterior texture stops HARD at the wound lip.
//
// URP -> Built-in swaps only: SAMPLE_TEXTURE2D->tex2D, SampleSH->ShadeSH9,
// GetMainLight->_WorldSpaceLightPos0/_LightColor0 + UNITY_LIGHT_ATTENUATION, PI->UNITY_PI.
// The _ADDITIONAL_LIGHTS loop and the URP DepthOnly pass are omitted (single directional
// light; Built-in forward needs no depth prepass).

Shader "ReconGridDC/LiverSurface"
{
    Properties
    {
        [Header(Base Color)]
        [MainTexture] _MainTex ("Liver Albedo", 2D) = "white" {}
        [MainColor] _Color ("Tissue Base Color", Color) = (0.62, 0.18, 0.10, 1)
        _InteriorColor ("Cut Interior Color (our extension)", Color) = (0.30, 0.025, 0.018, 1)
        _TextureStrength ("Texture Strength", Range(0, 1)) = 1.0
        _TextureContrast ("Texture Contrast", Range(0.25, 2.5)) = 1.35
        _TextureColorBlend ("Texture Native Color Blend", Range(0, 1)) = 1.0
        _UvTextureWeight ("Planar UV Texture Weight", Range(0, 1)) = 0.0
        _AlbedoBrightness ("Albedo Brightness", Range(0.2, 1.5)) = 0.82
        _TriplanarScale ("Triplanar Scale", Float) = 5.5
        _TriplanarBlend ("Triplanar Blend", Range(0.1, 16)) = 5.0

        [Header(Detail Maps)]
        _HeightMap ("Liver Height Detail (liver2_height)", 2D) = "gray" {}
        _SpecMap ("Liver Specular Detail (liver2_spec)", 2D) = "black" {}
        _NormalStrength ("Normal Strength", Range(0, 2)) = 0.12
        _ProceduralNormalStrength ("Procedural Detail Normal Strength", Range(0, 2)) = 0.05

        [Header(Wet Specular)]
        _Roughness ("Roughness", Range(0, 1)) = 0.70
        _SpecularStrength ("Wet Specular Strength", Range(0, 2)) = 0.18
        _SpecularColor ("Specular Tint", Color) = (0.9, 0.82, 0.72, 1)
        _Wetness ("Wetness", Range(0, 1)) = 0.18

        [Header(Subsurface Scattering)]
        _SSSColor ("SSS Color", Color) = (0.76, 0.13, 0.055, 1)
        _SSSStrength ("SSS Strength", Range(0, 2)) = 0.06
        _SSSDirect ("Backlight Strength", Range(0, 2)) = 0.04
        _SSSPower ("Backlight Focus", Range(1, 24)) = 7.0
        _SSSWrap ("Diffuse Wrap", Range(0, 1)) = 0.32

        [Header(Fresnel Rim)]
        _FresnelStrength ("Wet Rim Strength", Range(0, 2)) = 0.08
        _FresnelPow ("Wet Rim Falloff", Range(1, 8)) = 4.2

        [Header(GPU Tissue Detail)]
        _MicroMottleStrength ("Micro Mottle Strength", Range(0, 1)) = 0.16
        _MicroMottleScale ("Micro Mottle Scale", Float) = 20.0
        _VeinStrength ("Dark Vein Strength", Range(0, 1)) = 0.10
        _VeinScale ("Dark Vein Scale", Float) = 22.0
        _DebugTextureOnly ("Debug Texture Only", Range(0, 1)) = 0.0
    }

    SubShader
    {
        Tags { "RenderType"="Opaque" "Queue"="Geometry" }
        LOD 300

        // ============================================================
        // Pass 1: ForwardBase (main directional light + SH ambient)
        // ============================================================
        Pass
        {
            Name "FORWARD"
            Tags { "LightMode"="ForwardBase" }
            Cull Off   // double-sided: reference-verbatim (VFACE is the whole story)
            ZWrite On

            CGPROGRAM
            #pragma vertex   vert
            #pragma fragment frag
            #pragma multi_compile_fwdbase
            #pragma target 3.0     // VFACE

            #include "UnityCG.cginc"
            #include "Lighting.cginc"
            #include "AutoLight.cginc"

            sampler2D _MainTex;   float4 _MainTex_ST;
            sampler2D _HeightMap; float4 _HeightMap_ST;
            sampler2D _SpecMap;   float4 _SpecMap_ST;

            half4  _Color;
            half4  _InteriorColor;
            half4  _SpecularColor;
            half4  _SSSColor;
            half   _TextureStrength;
            half   _TextureContrast;
            half   _TextureColorBlend;
            half   _UvTextureWeight;
            half   _AlbedoBrightness;
            float  _TriplanarScale;
            float  _TriplanarBlend;
            half   _NormalStrength;
            half   _ProceduralNormalStrength;
            half   _Roughness;
            half   _SpecularStrength;
            half   _Wetness;
            half   _SSSStrength;
            half   _SSSDirect;
            half   _SSSPower;
            half   _SSSWrap;
            half   _FresnelStrength;
            half   _FresnelPow;
            half   _MicroMottleStrength;
            float  _MicroMottleScale;
            half   _VeinStrength;
            float  _VeinScale;
            half   _DebugTextureOnly;

            // Built-in shadow macros textually reference v.vertex / v.normal and o.pos -
            // struct/field/param names below are load-bearing.
            struct appdata
            {
                float4 vertex  : POSITION;
                float3 normal  : NORMAL;
                float2 uv      : TEXCOORD0;   // soup has no UV0 -> zeros; inert at _UvTextureWeight 0
                float4 restAux : TEXCOORD1;   // xyz = REST anchor (their restOS), w = wall flag f
            };

            struct v2f
            {
                float4 pos        : SV_POSITION;
                float3 positionWS : TEXCOORD0;
                float3 normalWS   : TEXCOORD1;
                float3 normalOS   : TEXCOORD2;
                float4 restAux    : TEXCOORD3;
                float2 uv         : TEXCOORD4;
                UNITY_SHADOW_COORDS(5)
            };

            // -- reference-verbatim helpers (LiverTissueGPU.shader:128-174) -----------------
            float3 SafeNorm(float3 v)
            {
                float lenSq = dot(v, v);
                return lenSq > 1e-12 ? v * rsqrt(lenSq) : float3(0.0, 1.0, 0.0);
            }

            float2 ApplyST(float2 uv, float4 st)
            {
                return uv * st.xy + st.zw;
            }

            float3 TriplanarWeights(float3 nOS)
            {
                float3 w = pow(max(abs(nOS), 1e-4), max(_TriplanarBlend, 1e-3));
                return w / max(w.x + w.y + w.z, 1e-5);
            }

            float Hash31(float3 p)
            {
                p = frac(p * 0.1031);
                p += dot(p, p.yzx + 33.33);
                return frac((p.x + p.y) * p.z);
            }

            float Noise3(float3 p)
            {
                float3 i = floor(p);
                float3 f = frac(p);
                float3 u = f * f * (3.0 - 2.0 * f);

                float n000 = Hash31(i + float3(0, 0, 0));
                float n100 = Hash31(i + float3(1, 0, 0));
                float n010 = Hash31(i + float3(0, 1, 0));
                float n110 = Hash31(i + float3(1, 1, 0));
                float n001 = Hash31(i + float3(0, 0, 1));
                float n101 = Hash31(i + float3(1, 0, 1));
                float n011 = Hash31(i + float3(0, 1, 1));
                float n111 = Hash31(i + float3(1, 1, 1));

                float nx00 = lerp(n000, n100, u.x);
                float nx10 = lerp(n010, n110, u.x);
                float nx01 = lerp(n001, n101, u.x);
                float nx11 = lerp(n011, n111, u.x);
                float nxy0 = lerp(nx00, nx10, u.y);
                float nxy1 = lerp(nx01, nx11, u.y);
                return lerp(nxy0, nxy1, u.z);
            }

            // -- triplanar samplers (reference-verbatim; restOS = our UV1 rest anchor) ------
            half3 SampleTriplanarAlbedo(float3 restOS, float3 nOS)
            {
                float3 w = TriplanarWeights(nOS);
                float s = max(_TriplanarScale, 1e-5);
                half3 cx = tex2D(_MainTex, ApplyST(restOS.zy * s, _MainTex_ST)).rgb;
                half3 cy = tex2D(_MainTex, ApplyST(restOS.xz * s, _MainTex_ST)).rgb;
                half3 cz = tex2D(_MainTex, ApplyST(restOS.xy * s, _MainTex_ST)).rgb;
                half3 tex = cx * w.x + cy * w.y + cz * w.z;
                return saturate((tex - 0.5h) * _TextureContrast + 0.5h);
            }

            half3 SampleUvAlbedo(float2 uv)
            {
                half3 tex = tex2D(_MainTex, ApplyST(uv, _MainTex_ST)).rgb;
                return saturate((tex - 0.5h) * _TextureContrast + 0.5h);
            }

            half SampleHeightDetail(float2 uv)
            {
                return tex2D(_HeightMap, ApplyST(uv, _HeightMap_ST)).r;
            }

            half SampleTriplanarHeight(float3 restOS, float3 nOS)
            {
                float3 w = TriplanarWeights(nOS);
                float s = max(_TriplanarScale, 1e-5);
                half hx = tex2D(_HeightMap, ApplyST(restOS.zy * s, _HeightMap_ST)).r;
                half hy = tex2D(_HeightMap, ApplyST(restOS.xz * s, _HeightMap_ST)).r;
                half hz = tex2D(_HeightMap, ApplyST(restOS.xy * s, _HeightMap_ST)).r;
                return hx * w.x + hy * w.y + hz * w.z;
            }

            half SampleSpecDetail(float2 uv)
            {
                return tex2D(_SpecMap, ApplyST(uv, _SpecMap_ST)).r;
            }

            half SampleTriplanarSpec(float3 restOS, float3 nOS)
            {
                float3 w = TriplanarWeights(nOS);
                float s = max(_TriplanarScale, 1e-5);
                half sx = tex2D(_SpecMap, ApplyST(restOS.zy * s, _SpecMap_ST)).r;
                half sy = tex2D(_SpecMap, ApplyST(restOS.xz * s, _SpecMap_ST)).r;
                half sz = tex2D(_SpecMap, ApplyST(restOS.xy * s, _SpecMap_ST)).r;
                return sx * w.x + sy * w.y + sz * w.z;
            }

            // -- HEIGHT-GRADIENT normal (the reference's ONLY map-driven normal; _NormalMap
            // is never sampled). eps = 0.004/s auto-adapts to any triplanar scale. ----------
            float3 SampleTriplanarNormal(float3 restOS, float3 nOS)
            {
                float s = max(_TriplanarScale, 1e-5);
                float eps = 0.004 / s;
                half hx = SampleTriplanarHeight(restOS + float3(eps, 0.0, 0.0), nOS)
                        - SampleTriplanarHeight(restOS - float3(eps, 0.0, 0.0), nOS);
                half hy = SampleTriplanarHeight(restOS + float3(0.0, eps, 0.0), nOS)
                        - SampleTriplanarHeight(restOS - float3(0.0, eps, 0.0), nOS);
                half hz = SampleTriplanarHeight(restOS + float3(0.0, 0.0, eps), nOS)
                        - SampleTriplanarHeight(restOS - float3(0.0, 0.0, eps), nOS);
                return SafeNorm(nOS - float3(hx, hy, hz) * _NormalStrength * 2.5);
            }

            float3 ProceduralFineNormal(float3 restOS, float3 nOS)
            {
                float scale = max(_MicroMottleScale, 0.001);
                float3 p = restOS * scale;
                float eps = 0.075;
                float h = Noise3(p);
                float hx = Noise3(p + float3(eps, 0.0, 0.0)) - h;
                float hy = Noise3(p + float3(0.0, eps, 0.0)) - h;
                float hz = Noise3(p + float3(0.0, 0.0, eps)) - h;
                return SafeNorm(nOS + float3(hx, hy, hz) * _ProceduralNormalStrength * 2.0);
            }

            // -- TextureDrivenAlbedo (reference-verbatim: tissueTint + grading + veins) -----
            half3 TextureDrivenAlbedo(half3 baseColor, half3 texColor, float3 restOS)
            {
                half3 tissueTint = half3(1.04h, 0.68h, 0.52h);
                half3 nativeTexture = saturate(texColor * tissueTint + baseColor * 0.08h);
                half texLum = dot(nativeTexture, half3(0.299h, 0.587h, 0.114h));
                half3 shadowedTexture = nativeTexture * 0.52h + half3(0.065h, 0.006h, 0.003h);
                half3 brightTexture = nativeTexture * 1.12h + half3(0.045h, 0.010h, 0.004h);
                half3 stainedTexture = lerp(shadowedTexture, brightTexture, smoothstep(0.10h, 0.70h, texLum));
                stainedTexture = lerp(stainedTexture, nativeTexture, saturate(_TextureColorBlend));

                half veinA = Noise3(restOS * max(_VeinScale, 0.001) + 13.7);
                half veinB = Noise3(restOS * max(_VeinScale * 2.17, 0.001) + 41.3);
                half veinMask = smoothstep(0.58h, 0.88h, veinA * 0.68h + veinB * 0.32h);
                half3 darkVein = stainedTexture * 0.42h + half3(0.055h, 0.001h, 0.001h);
                stainedTexture = lerp(stainedTexture, darkVein, veinMask * _VeinStrength);

                return lerp(baseColor, stainedTexture, saturate(_TextureStrength));
            }

            // -- GGX chain (reference-verbatim floors/epsilons) ------------------------------
            half3 Liver_FresnelSchlick(float cosTheta, half3 f0)
            {
                return f0 + (1.0 - f0) * pow(saturate(1.0 - cosTheta), 5.0);
            }

            float Liver_D_GGX(float ndh, float a2)
            {
                float d = ndh * ndh * (a2 - 1.0) + 1.0;
                return a2 / max(UNITY_PI * d * d, 1e-6);
            }

            float Liver_G_Schlick(float ndx, float k)
            {
                return ndx / max(ndx * (1.0 - k) + k, 1e-6);
            }

            half3 Liver_GGXSpecular(float3 n, float3 l, float3 v, half3 f0, float roughness)
            {
                float3 h = SafeNorm(l + v);
                float ndl = saturate(dot(n, l));
                float ndv = max(dot(n, v), 1e-4);
                float ndh = saturate(dot(n, h));
                float vdh = saturate(dot(v, h));
                float a = max(roughness * roughness, 0.045);
                float a2 = a * a;
                float k = (roughness + 1.0);
                k = k * k * 0.125;
                half3 f = Liver_FresnelSchlick(vdh, f0);
                return (Liver_D_GGX(ndh, a2) * Liver_G_Schlick(ndl, k) * Liver_G_Schlick(ndv, k) * f) / max(4.0 * ndl * ndv, 1e-5);
            }

            // -- SSS (reference-verbatim: SINGLE (1+wrap) divisor, albedo-coupled tint) -----
            half3 TissueSSS(float3 n, float3 l, float3 v, half3 albedo)
            {
                float wrap = saturate(_SSSWrap);
                float wrapped = saturate((dot(n, l) + wrap) / max(1.0 + wrap, 1e-4));
                float back = pow(saturate(dot(v, -l)), _SSSPower);
                half3 scatterTint = lerp(albedo, _SSSColor.rgb, 0.45);
                return scatterTint * (wrapped * _SSSStrength + back * _SSSDirect);
            }

            v2f vert(appdata v)
            {
                v2f o;
                o.pos        = UnityObjectToClipPos(v.vertex);
                o.positionWS = mul(unity_ObjectToWorld, v.vertex).xyz;
                o.normalWS   = UnityObjectToWorldNormal(v.normal);
                o.normalOS   = SafeNorm(v.normal);
                o.restAux    = v.restAux;
                o.uv         = v.uv;
                UNITY_TRANSFER_SHADOW(o, float2(0, 0));
                return o;
            }

            half4 frag(v2f i, fixed facing : VFACE) : SV_Target
            {
                // 1. VFACE double-siding (reference-verbatim; no ddx/ddy flip in the reference).
                float side = facing > 0.0 ? 1.0 : -1.0;
                float3 baseOS = SafeNorm(i.normalOS * side);
                float3 baseWS = SafeNorm(i.normalWS * side);
                float3 restOS = i.restAux.xyz;

                // 2. Height-gradient + procedural normals (object-space, tangent-free).
                float3 mappedOS = SampleTriplanarNormal(restOS, baseOS);
                float3 proceduralOS = ProceduralFineNormal(restOS, baseOS);
                mappedOS = SafeNorm(lerp(mappedOS, proceduralOS, saturate(_ProceduralNormalStrength)));
                float3 mappedWS = SafeNorm(UnityObjectToWorldNormal(mappedOS));
                float3 n = SafeNorm(lerp(baseWS, mappedWS, saturate(_NormalStrength)));
                float3 v = SafeNorm(_WorldSpaceCameraPos - i.positionWS);

                // 3. Texture channels: UV0 is the SofaUnity/Project2-style path, UV1.xyz
                // keeps the rest-frame triplanar path stable during deformation/cutting.
                half3 triTex = SampleTriplanarAlbedo(restOS, baseOS);
                half3 uvTex = SampleUvAlbedo(i.uv);
                half uvWeight = saturate(_UvTextureWeight);
                half3 texColor = lerp(triTex, uvTex, uvWeight);
                half heightDetail = lerp(SampleTriplanarHeight(restOS, baseOS), SampleHeightDetail(i.uv), uvWeight);
                half specDetail = lerp(SampleTriplanarSpec(restOS, baseOS), SampleSpecDetail(i.uv), uvWeight);

                // 4. Mottle / height gain.
                half3 baseColor = _Color.rgb;
                half mottle = Noise3(restOS * max(_MicroMottleScale, 0.001));
                half fine = Noise3(restOS * max(_MicroMottleScale * 2.37, 0.001) + 17.0);
                half mottleMix = saturate(mottle * 0.65 + fine * 0.35);
                half heightGain = lerp(0.88h, 1.10h, heightDetail);
                half mottleGain = lerp(1.0 - _MicroMottleStrength, 1.0 + _MicroMottleStrength, mottleMix) * heightGain;

                // 5. Albedo.
                half3 albedo = saturate(TextureDrivenAlbedo(baseColor, texColor, restOS) * _AlbedoBrightness * mottleGain);

                // 6. OUR cut-interior fold-in (design 3.3): hard skin/interior lip at the shared
                // FP; mottleGain kept inside the wound as the fiber-pattern analog.
                half f = saturate(i.restAux.w);
                albedo = lerp(albedo, saturate(_InteriorColor.rgb * mottleGain), f);

                // 7. Main light (Built-in plumbing for GetMainLight + shadowAttenuation).
                float3 l = SafeNorm(_WorldSpaceLightPos0.xyz);
                half3 lightColor = _LightColor0.rgb;
                UNITY_LIGHT_ATTENUATION(atten, i, i.positionWS);
                half shadow = atten;
                float ndl = saturate(dot(n, l));
                float wrap = saturate(_SSSWrap);
                float wrapped = saturate((dot(n, l) + wrap) / max(1.0 + wrap, 1e-4));

                // 8. Wet spec model (reference-verbatim).
                half roughness = saturate(lerp(_Roughness, max(0.16h, _Roughness * 0.72h), _Wetness * specDetail));
                half3 f0 = lerp(half3(0.028h, 0.028h, 0.028h), _SpecularColor.rgb * 0.08h, _Wetness);
                half3 diffuse = albedo * lightColor * wrapped * shadow;
                half specMask = lerp(0.45h, 1.15h, specDetail);
                half3 specular = Liver_GGXSpecular(n, l, v, f0, roughness) * lightColor * ndl * shadow * _SpecularStrength * specMask;
                specular *= (1.0 - 0.7 * f);          // wall = matte moist interior (design 3.3)
                half3 sss = TissueSSS(n, l, v, albedo) * lightColor * shadow;   // FULL on walls
                half rim = pow(1.0 - saturate(dot(n, v)), _FresnelPow);
                half3 fresnel = rim * _SpecularColor.rgb * _FresnelStrength * _Wetness * lightColor;
                fresnel *= (1.0 - f);                 // no wet rim inside the cut (design 3.3)

                // 9. Ambient (reference factor 0.65; ShadeSH9 = Built-in SampleSH).
                half3 ambient = ShadeSH9(half4(n, 1)) * albedo * 0.65h;

                half3 litColor = ambient + diffuse + specular + sss + fresnel;
                return half4(lerp(litColor, texColor, saturate(_DebugTextureOnly)), 1.0);
            }
            ENDCG
        }

        // ============================================================
        // Pass 2: ShadowCaster (Built-in; kept verbatim from our port)
        // ============================================================
        Pass
        {
            Name "ShadowCaster"
            Tags { "LightMode"="ShadowCaster" }
            Cull Off
            ZWrite On

            CGPROGRAM
            #pragma vertex   vertShadow
            #pragma fragment fragShadow
            #pragma multi_compile_shadowcaster
            #pragma target 3.0

            #include "UnityCG.cginc"

            struct appdata
            {
                float4 vertex : POSITION;
                float3 normal : NORMAL;
            };

            struct v2f_shadow
            {
                V2F_SHADOW_CASTER;
            };

            v2f_shadow vertShadow(appdata v)
            {
                v2f_shadow o;
                TRANSFER_SHADOW_CASTER_NORMALOFFSET(o)
                return o;
            }

            float4 fragShadow(v2f_shadow i) : SV_Target
            {
                SHADOW_CASTER_FRAGMENT(i)
            }
            ENDCG
        }
    }

    FallBack Off
}
