using Unity.Mathematics;
namespace ReconGridDC.Preprocess
{
    public interface ILevelSetProvider
    {
        float Sample(float3 p);     // signed: <0 inside, >0 outside
        float3 Gradient(float3 p);  // outward, ~unit length
    }
}
