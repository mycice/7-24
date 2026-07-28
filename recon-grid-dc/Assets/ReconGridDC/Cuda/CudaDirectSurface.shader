Shader "ReconGridDC/CudaDirectSurface"
{
    Properties
    {
        _Color ("Color", Color) = (0.62, 0.18, 0.10, 1)
    }
    SubShader
    {
        Tags { "RenderType"="Opaque" }
        Cull Off
        ZWrite On
        Pass
        {
            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            #pragma target 4.5
            #include "UnityCG.cginc"

            ByteAddressBuffer _CudaSurfacePositions;
            ByteAddressBuffer _CudaSurfaceNormals;
            ByteAddressBuffer _CudaSurfaceAux;
            ByteAddressBuffer _CudaSurfaceVertexCount;
            fixed4 _Color;

            struct v2f { float4 pos : SV_POSITION; float3 normal : TEXCOORD0; float wall : TEXCOORD1; float valid : TEXCOORD2; };

            v2f vert(uint vertexId : SV_VertexID)
            {
                v2f o;
                uint vertexCount = _CudaSurfaceVertexCount.Load(0);
                uint triangleStart = (vertexId / 3u) * 3u;

                // DrawProcedural uses the fixed capacity. Reject a whole trailing triangle before
                // reading its uninitialized GPU buffer entries, so it cannot form red fragments.
                if (triangleStart + 2u >= vertexCount)
                {
                    o.pos = float4(2.0, 2.0, 2.0, 1.0);
                    o.normal = float3(0.0, 1.0, 0.0);
                    o.wall = 0.0;
                    o.valid = -1.0;
                    return o;
                }

                uint3 packedPosition = _CudaSurfacePositions.Load3(vertexId * 12);
                uint3 packedNormal = _CudaSurfaceNormals.Load3(vertexId * 12);
                uint4 packedAux = _CudaSurfaceAux.Load4(vertexId * 16);
                float3 position = float3(asfloat(packedPosition.x), asfloat(packedPosition.y), asfloat(packedPosition.z));
                o.pos = UnityObjectToClipPos(float4(position, 1));
                o.normal = UnityObjectToWorldNormal(float3(asfloat(packedNormal.x), asfloat(packedNormal.y), asfloat(packedNormal.z)));
                o.wall = asfloat(packedAux.w);
                o.valid = 1.0;
                return o;
            }

            fixed4 frag(v2f i, bool frontFace : SV_IsFrontFace) : SV_Target
            {
                clip(i.valid);
                float3 normal = normalize(frontFace ? i.normal : -i.normal);
                float3 lightDirection = normalize(_WorldSpaceLightPos0.xyz);
                float diffuse = saturate(dot(normal, lightDirection)) * 0.78 + 0.22;
                float3 tissue = lerp(_Color.rgb, float3(0.30, 0.025, 0.018), saturate(i.wall));
                return fixed4(tissue * diffuse, 1);
            }
            ENDCG
        }
    }
}
