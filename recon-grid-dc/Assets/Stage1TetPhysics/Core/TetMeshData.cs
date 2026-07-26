// TetMeshData.cs
// 鍥涢潰楂旂恫鏍兼牳蹇冩暩鎿氱祼妲?// 鍏煎 Matthias M眉ller Ten-Minute-Physics Blender TetPlugin 灏庡嚭鐨?JSON 鏍煎紡
// 鍙冭€? https://github.com/Habrador/Ten-Minute-Physics-Unity (MIT License)
// 鏁告摎浣堝眬鐐哄钩鍧︽暩绲?(flat array)锛屼究鏂肩洿鎺ヤ笂鍌?GPU ComputeBuffer

using UnityEngine;

namespace ReconGridDC.Stage1TetPhysics.Core
{
    /// <summary>
    /// JSON 鍙嶅簭鍒楀寲瀹瑰櫒锛堝皪鎳?Blender TetPlugin 灏庡嚭鏍煎紡锛?    /// verts: [x0,y0,z0, x1,y1,z1, ...]  骞冲潶闋傞粸鍧愭
    /// tetIds: [i0,i1,i2,i3, i0,i1,i2,i3, ...] 鍥涢潰楂旈爞榛炵储寮曪紙姣?鍊嬩竴绲勶級
    /// tetSurfaceTriIds: [i0,i1,i2, ...] 閭婄晫琛ㄩ潰涓夎褰㈢储寮曪紙姣?鍊嬩竴绲勶級
    /// </summary>
    [System.Serializable]
    public class TetMeshJson
    {
        public float[] verts;             // 闋傞粸鍧愭锛堝钩鍧︼級
        public int[]   tetIds;            // 鍥涢潰楂旂储寮曪紙骞冲潶锛屾瘡4鍊嬩竴绲勶級
        public int[]   tetSurfaceTriIds;  // 閭婄晫涓夎褰㈢储寮曪紙骞冲潶锛屾瘡3鍊嬩竴绲勶級
        public int[]   tetEdgeIds;        // 鐢ㄦ柤 debug 椤ず鐨勯倞锛堝彲閬革級
    }

    /// <summary>
    /// 閬嬭鏅傚洓闈㈤珨缍叉牸鏁告摎
    /// 鎵€鏈夋暩绲勫潎鐐?GPU-friendly 鐨勫钩鍧︿綀灞€
    /// </summary>
    public class TetMeshData
    {
        // 鈹€鈹€ 鎷撴挷锛堝垵濮嬪寲寰屼笉璁婏級鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
        public int   NumParticles  { get; private set; }
        public int   NumTets       { get; private set; }
        public int   NumSurfaceTris { get; private set; }

        /// <summary>鍥涢潰楂旈爞榛炵储寮曪紝骞冲潶鏁哥祫锛岄暦搴?= NumTets * 4</summary>
        public int[] TetIds        { get; private set; }

        /// <summary>閭婄晫涓夎褰㈤爞榛炵储寮曪紝骞冲潶鏁哥祫锛岄暦搴?= NumSurfaceTris * 3</summary>
        public int[] SurfaceTriIds { get; private set; }

        // 鈹€鈹€ 绮掑瓙鐙€鎱嬶紙姣忓箑鐢?XPBD Solver 鏇存柊锛夆攢鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
        /// <summary>鐣跺墠绮掑瓙浣嶇疆 [numParticles]</summary>
        public Vector3[] Positions    { get; set; }

        /// <summary>闈滄浣嶇疆锛堝垵濮嬪寲寰屼笉璁婏級[numParticles]</summary>
        public Vector3[] RestPositions { get; set; }

        /// <summary>涓婁竴骞€浣嶇疆锛岀敤鏂?XPBD 閫熷害瑷堢畻 [numParticles]</summary>
        public Vector3[] PrevPositions { get; set; }

        /// <summary>閫熷害 [numParticles]</summary>
        public Vector3[] Velocities    { get; set; }

        /// <summary>璩噺鍊掓暩 [numParticles]銆? = 鍥哄畾榛烇紙鐒￠檺璩噺锛?/summary>
        public float[] InvMass         { get; set; }

        // 鈹€鈹€ Tetrahedron activity state 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
        /// <summary>Whether each tetrahedron participates in the solver [numTets].</summary>
        public bool[] TetActive        { get; set; }

        // 鈹€鈹€ 绱勬潫闋愯▓绠楋紙鍒濆鍖栧緦涓嶈畩锛夆攢鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
        /// <summary>鍚勫洓闈㈤珨鐨勯潨姝㈤珨绌?[numTets]</summary>
        public float[] RestVolumes     { get; private set; }

        // 鈹€鈹€ 鍦栬憲鑹插垎绲勶紙GPU 涓﹁姹傝В鐢級鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
        /// <summary>椤忚壊绲勶紝姣忓€嬪厓绱犳槸瑭茬祫鍖呭惈鐨?tet 绱㈠紩鍒楄〃</summary>
        public int[][] ColorGroups     { get; set; }

        // 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
        /// <summary>
        /// 寰?JSON 鏁告摎鍒濆鍖栫恫鏍硷紙CPU 鍋达級
        /// </summary>
        public void InitFromJson(TetMeshJson json, float massDensity = 1000f)
        {
            int vertCount = json.verts.Length / 3;
            int tetCount  = json.tetIds.Length / 4;

            NumParticles   = vertCount;
            NumTets        = tetCount;
            NumSurfaceTris = json.tetSurfaceTriIds != null
                           ? json.tetSurfaceTriIds.Length / 3
                           : 0;

            // 鎷疯矟绱㈠紩鏁哥祫
            TetIds        = json.tetIds;
            SurfaceTriIds = json.tetSurfaceTriIds ?? new int[0];

            // Initialize particle state arrays.
            Positions     = new Vector3[vertCount];
            RestPositions = new Vector3[vertCount];
            PrevPositions = new Vector3[vertCount];
            Velocities    = new Vector3[vertCount];
            InvMass       = new float[vertCount];

            // 瑙ｆ瀽闋傞粸鍧愭锛圲nity 鍧愭绯伙細Y 杌告湞涓婏級
            for (int i = 0; i < vertCount; i++)
            {
                float x = json.verts[i * 3 + 0];
                float y = json.verts[i * 3 + 1];
                float z = json.verts[i * 3 + 2];
                Positions[i]     = new Vector3(x, y, z);
                RestPositions[i] = new Vector3(x, y, z);
                PrevPositions[i] = new Vector3(x, y, z);
            }

            // Initialize tetrahedron state arrays.
            TetActive    = new bool[tetCount];
            RestVolumes  = new float[tetCount];

            float totalVol = 0f;
            for (int t = 0; t < tetCount; t++)
            {
                TetActive[t]   = true;
                RestVolumes[t] = Mathf.Abs(ComputeTetVolume(t)); // Abs锛氬吋瀹笵elaunay鍙兘鐨勮矤楂旂
                totalVol      += RestVolumes[t];
            }

            // Distribute each tetrahedron's mass equally to its four particles.
            float[] mass = new float[vertCount];
            float totalMass = massDensity * totalVol;
            float massPerTet = totalMass / tetCount;

            for (int t = 0; t < tetCount; t++)
            {
                float m = massPerTet / 4f;
                mass[TetIds[t*4+0]] += m;
                mass[TetIds[t*4+1]] += m;
                mass[TetIds[t*4+2]] += m;
                mass[TetIds[t*4+3]] += m;
            }

            for (int i = 0; i < vertCount; i++)
                InvMass[i] = (mass[i] > 0f) ? 1f / mass[i] : 0f;

            Debug.Log($"[TetMeshData] 鍒濆鍖栧畬鎴? {NumParticles} 绮掑瓙, " +
                      $"{NumTets} 鍥涢潰楂? {NumSurfaceTris} 琛ㄩ潰涓夎褰? " +
                      $"绺介珨绌?{totalVol * 1e6f:F1} cm鲁"); // 1m鲁=1e6cm鲁
        }

        /// <summary>瑷堢畻绗?t 鍊嬪洓闈㈤珨鐨勬湁绗﹁櫉楂旂</summary>
        public float ComputeTetVolume(int t)
        {
            Vector3 p0 = Positions[TetIds[t*4+0]];
            Vector3 p1 = Positions[TetIds[t*4+1]];
            Vector3 p2 = Positions[TetIds[t*4+2]];
            Vector3 p3 = Positions[TetIds[t*4+3]];
            return Vector3.Dot(Vector3.Cross(p1 - p0, p2 - p0), p3 - p0) / 6f;
        }

        /// <summary>瑷堢畻绗?t 鍊嬪洓闈㈤珨鐨勪腑蹇冧綅缃?/summary>
        public Vector3 TetCenter(int t)
        {
            return (Positions[TetIds[t*4+0]] + Positions[TetIds[t*4+1]] +
                    Positions[TetIds[t*4+2]] + Positions[TetIds[t*4+3]]) * 0.25f;
        }

        /// <summary>鍥哄畾鎸囧畾绮掑瓙锛堣ō invMass = 0锛?/summary>
        public void PinParticle(int i)   => InvMass[i] = 0f;

        /// <summary>鍒ゆ柗绮掑瓙鏄惁琚浐瀹?/summary>
        public bool IsPinned(int i)      => InvMass[i] == 0f;

        // 鈹€鈹€ 鍕曟厠鎷撴挷淇敼锛圴NA vertex splitting 鐢級鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€

        /// <summary>鏇存柊鍥涢潰楂旂储寮曪紙vertex splitting 寰岋級</summary>
        public void SetTetIds(int[] newIds)
        {
            TetIds = newIds;
        }

        public void SetSurfaceTriIds(int[] newIds)
        {
            SurfaceTriIds = newIds ?? new int[0];
            NumSurfaceTris = SurfaceTriIds.Length / 3;
        }

        /// <summary>鏇存柊绮掑瓙鏁搁噺锛坴ertex splitting 寰岋級</summary>
        public void SetNumParticles(int n)
        {
            NumParticles = n;
        }

        /// <summary>瑷疆闈滄浣嶇疆锛坴ertex splitting 寰岄渶瑕佹洿鏂帮級</summary>
        public void SetRestPositions(Vector3[] rp)
        {
            RestPositions = rp;
        }

        // 鈹€鈹€ 椤剁偣鍒嗚鏀寔 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€

        int _capacity = 0; // 绮掑瓙鏁扮粍瀹為檯瀹归噺

        public void EnsureCapacity(int minCapacity)
        {
            if (_capacity >= minCapacity) return;
            _capacity = minCapacity;
            Positions     = ResizeVec3(Positions, _capacity);
            RestPositions = ResizeVec3(RestPositions, _capacity);
            PrevPositions = ResizeVec3(PrevPositions, _capacity);
            Velocities    = ResizeVec3(Velocities, _capacity);
            InvMass       = ResizeFloat(InvMass, _capacity);
        }

        public int AddParticle(Vector3 pos, Vector3 vel, float invMass)
        {
            int newIdx = NumParticles;
            if (newIdx >= Positions.Length)
            {
                int newCap = Mathf.Max(Positions.Length * 3 / 2, newIdx + 1);
                EnsureCapacity(newCap);
            }
            Positions[newIdx]     = pos;
            RestPositions[newIdx] = pos;
            PrevPositions[newIdx] = pos;
            Velocities[newIdx]    = vel;
            InvMass[newIdx]       = invMass;
            NumParticles = newIdx + 1;
            return newIdx;
        }

        // 鈹€鈹€ 鍥涢潰浣撶粏鍒嗘敮鎸?鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€

        int _tetCapacity = 0;

        /// <summary>纭繚 tet 鏁扮粍鏈夎冻澶熷閲?/summary>
        public void EnsureTetCapacity(int minCapacity)
        {
            if (_tetCapacity >= minCapacity) return;
            _tetCapacity = minCapacity;
            TetIds     = ResizeInt(TetIds, _tetCapacity * 4);
            TetActive  = ResizeBool(TetActive, _tetCapacity);
            RestVolumes = ResizeFloat(RestVolumes, _tetCapacity);
        }

        /// <summary>娣诲姞涓€涓柊鍥涢潰浣擄紝杩斿洖鍏剁储寮?/summary>
        public int AddTet(int v0, int v1, int v2, int v3)
        {
            int newIdx = NumTets;
            if (newIdx >= _tetCapacity)
            {
                int newCap = Mathf.Max(_tetCapacity * 3 / 2, newIdx + 100);
                EnsureTetCapacity(newCap);
            }
            int b = newIdx * 4;
            TetIds[b] = v0; TetIds[b+1] = v1; TetIds[b+2] = v2; TetIds[b+3] = v3;
            TetActive[newIdx] = true;

            // 鐢?RestPositions 璁＄畻闈欐浣撶Н
            Vector3 p0 = RestPositions[v0], p1 = RestPositions[v1];
            Vector3 p2 = RestPositions[v2], p3 = RestPositions[v3];
            float vol = Mathf.Abs(Vector3.Dot(Vector3.Cross(p1 - p0, p2 - p0), p3 - p0)) / 6f;
            RestVolumes[newIdx] = vol;

            NumTets = newIdx + 1;
            return newIdx;
        }

        /// <summary>鏇存柊 RestVolumes锛堝閮ㄥ彲鍐欙級</summary>
        public void DeactivateTet(int t)
        {
            if (t >= 0 && t < NumTets) TetActive[t] = false;
        }

        public void SetRestVolumes(float[] rv) { RestVolumes = rv; }

        // 鈹€鈹€ 宸ュ叿鏂规硶 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€

        static Vector3[] ResizeVec3(Vector3[] arr, int newSize)
        {
            if (arr.Length >= newSize) return arr;
            var newArr = new Vector3[newSize];
            System.Array.Copy(arr, newArr, arr.Length);
            return newArr;
        }

        static float[] ResizeFloat(float[] arr, int newSize)
        {
            if (arr.Length >= newSize) return arr;
            var newArr = new float[newSize];
            System.Array.Copy(arr, newArr, arr.Length);
            return newArr;
        }

        static int[] ResizeInt(int[] arr, int newSize)
        {
            if (arr.Length >= newSize) return arr;
            var newArr = new int[newSize];
            System.Array.Copy(arr, newArr, arr.Length);
            return newArr;
        }

        static bool[] ResizeBool(bool[] arr, int newSize)
        {
            if (arr.Length >= newSize) return arr;
            var newArr = new bool[newSize];
            System.Array.Copy(arr, newArr, arr.Length);
            return newArr;
        }
    }
}

