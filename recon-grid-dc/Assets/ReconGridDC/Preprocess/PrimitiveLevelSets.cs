using Unity.Mathematics;
namespace ReconGridDC.Preprocess
{
    public sealed class SphereLevelSet : ILevelSetProvider
    {
        readonly float3 c; readonly float r;
        public SphereLevelSet(float3 center, float radius){ c=center; r=radius; }
        public float Sample(float3 p) => math.length(p-c) - r;
        public float3 Gradient(float3 p)
        {
            float3 d = p-c; float len = math.length(d);
            return len > 1e-8f ? d/len : new float3(1,0,0);
        }
    }

    public sealed class BoxLevelSet : ILevelSetProvider
    {
        readonly float3 c, h;
        public BoxLevelSet(float3 center, float3 halfExtents){ c=center; h=halfExtents; }
        public float Sample(float3 p)
        {
            float3 q = math.abs(p-c) - h;
            float outside = math.length(math.max(q,0f));
            float inside  = math.min(math.max(q.x, math.max(q.y,q.z)), 0f);
            return outside + inside; // exact signed box distance
        }
        public float3 Gradient(float3 p)
        {
            // central difference (robust at edges/corners)
            const float e = 1e-3f;
            float dx = Sample(p+new float3(e,0,0)) - Sample(p-new float3(e,0,0));
            float dy = Sample(p+new float3(0,e,0)) - Sample(p-new float3(0,e,0));
            float dz = Sample(p+new float3(0,0,e)) - Sample(p-new float3(0,0,e));
            float3 g = new float3(dx,dy,dz);
            float len = math.length(g);
            return len > 1e-8f ? g/len : new float3(1,0,0);
        }
    }
}
