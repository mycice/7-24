// MshLoader.cs — liver-import pipeline (design docs/superpowers/specs/2026-06-29-liver-import-rod-cutting).
// Parses gmsh 4.1 ASCII tetrahedral volume meshes and extracts their boundary surface.
//
// gmsh 4.1 ASCII layout (verified against liver3_refined_1.msh) — the block structure is EXACT;
// a hand-rolled reader breaks if any of it is approximated:
//   $MeshFormat  -> "4.1 0 8"  (version 4.1, file-type 0 = ASCII, data-size 8). Reject non-4.1 / binary.
//   $Entities    -> OPTIONAL; skipped (the bbox is computed from parsed node coords, not from $Entities).
//   $Nodes:
//     SECTION header: "numEntityBlocks numNodes minNodeTag maxNodeTag"
//     per block:      "entityDim entityTag parametric numNodesInBlock"  (3rd field is PARAMETRIC, not a type)
//                     then numNodesInBlock node TAGS (one int per line),
//                     THEN numNodesInBlock coord triples "x y z".
//                     >>> tags and coords are TWO SEPARATE consecutive lists, NOT interleaved "tag x y z". <<<
//   $Elements:
//     SECTION header: "numEntityBlocks numElements minElemTag maxElemTag"
//     per block:      "entityDim entityTag elementType numElementsInBlock"  (elementType 4 = tet4)
//                     then rows "elemTag n1 n2 n3 n4"  (node TAGS, not indices).
// Node tags are remapped tag->dense-index via a Dictionary (no tag==index+1 assumption); element node
// tags are remapped through the same dictionary. design §1.

using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using Unity.Mathematics;

namespace ReconGridDC.Preprocess
{
    /// <summary>Raw parsed gmsh tetrahedral volume mesh (dense 0-based node indices).</summary>
    public struct MshMesh
    {
        public float3[] nodes;   // node positions, dense 0-based
        public int4[]   tets;    // tetrahedra: 4 dense 0-based node indices each
    }

    /// <summary>
    /// Boundary triangulation extracted from a tet mesh: faces belonging to exactly ONE tet,
    /// wound OUTWARD (normal away from the tet's opposite/interior vertex). verts shares MshMesh.nodes.
    /// </summary>
    public struct MshSurface
    {
        public float3[] verts;   // == MshMesh.nodes (dense node positions)
        public int3[]   tris;    // boundary triangles, outward-wound, dense vertex indices
    }

    public static class MshLoader
    {
        // Index-packing multiplier for face/edge canonical keys. > any node index (13412 < 2^22).
        const long KEY_MULT = 4194304L; // 2^22

        static readonly char[] WS = { ' ', '\t', '\r' };
        static string[] Tok(string line) => line.Split(WS, StringSplitOptions.RemoveEmptyEntries);

        // ── gmsh 4.1 ASCII parse ─────────────────────────────────────────────────────────────
        public static MshMesh LoadMsh(string path)
        {
            if (!File.Exists(path))
                throw new FileNotFoundException($"[MshLoader] .msh not found: {path}");

            string[] lines = File.ReadAllLines(path);

            // $MeshFormat — "4.1 0 8"
            int mf = SeekSection(lines, "$MeshFormat", 0);
            string[] fmt = Tok(lines[mf + 1]);
            if (fmt.Length < 3 || !fmt[0].StartsWith("4.1"))
                throw new NotSupportedException(
                    $"[MshLoader] unsupported .msh version '{(fmt.Length > 0 ? fmt[0] : "?")}'; only gmsh 4.1 ASCII is supported.");
            if (fmt[1] != "0")
                throw new NotSupportedException("[MshLoader] only ASCII (.msh file-type 0) is supported; this file is binary.");

            float3[] nodes = ParseNodes(lines, out Dictionary<int, int> tagToIndex);
            int4[] tets    = ParseTets(lines, tagToIndex);
            return new MshMesh { nodes = nodes, tets = tets };
        }

        static float3[] ParseNodes(string[] lines, out Dictionary<int, int> tagToIndex)
        {
            int s = SeekSection(lines, "$Nodes", 0);
            int p = s + 1;
            string[] hdr = Tok(lines[p++]);                  // numEntityBlocks numNodes minTag maxTag
            int numBlocks = int.Parse(hdr[0]);
            int numNodes  = int.Parse(hdr[1]);

            var nodes = new float3[numNodes];
            tagToIndex = new Dictionary<int, int>(numNodes);

            int dense = 0;
            for (int b = 0; b < numBlocks; b++)
            {
                string[] bh = Tok(lines[p++]);               // entityDim entityTag parametric numNodesInBlock
                int parametric = int.Parse(bh[2]);
                int cnt        = int.Parse(bh[3]);
                if (parametric != 0)
                    throw new NotSupportedException("[MshLoader] parametric nodes (parametric==1) are not supported.");

                int blockStart = dense;
                // FIRST list: numNodesInBlock TAGS (one int per line) → tag maps to its dense index.
                for (int k = 0; k < cnt; k++)
                {
                    int tag = int.Parse(lines[p++].Trim());
                    tagToIndex[tag] = blockStart + k;
                }
                // SECOND list: numNodesInBlock coordinate triples (positional pairing with the tags above).
                for (int k = 0; k < cnt; k++)
                {
                    string[] c = Tok(lines[p++]);
                    nodes[blockStart + k] = new float3(
                        float.Parse(c[0], CultureInfo.InvariantCulture),
                        float.Parse(c[1], CultureInfo.InvariantCulture),
                        float.Parse(c[2], CultureInfo.InvariantCulture));
                }
                dense += cnt;
            }
            if (dense != numNodes)
                throw new InvalidDataException($"[MshLoader] $Nodes block counts ({dense}) != header numNodes ({numNodes}).");
            return nodes;
        }

        static int4[] ParseTets(string[] lines, Dictionary<int, int> tagToIndex)
        {
            int s = SeekSection(lines, "$Elements", 0);
            int p = s + 1;
            string[] hdr = Tok(lines[p++]);                  // numEntityBlocks numElements minTag maxTag
            int numBlocks = int.Parse(hdr[0]);
            int numElems  = int.Parse(hdr[1]);

            var tets = new List<int4>(numElems);
            for (int b = 0; b < numBlocks; b++)
            {
                string[] bh = Tok(lines[p++]);               // entityDim entityTag elementType numElementsInBlock
                int elemType = int.Parse(bh[2]);
                int cnt      = int.Parse(bh[3]);

                if (elemType == 4)                           // tet4: "elemTag n1 n2 n3 n4"
                {
                    for (int k = 0; k < cnt; k++)
                    {
                        string[] t = Tok(lines[p++]);
                        tets.Add(new int4(
                            ResolveTag(tagToIndex, t[1]),
                            ResolveTag(tagToIndex, t[2]),
                            ResolveTag(tagToIndex, t[3]),
                            ResolveTag(tagToIndex, t[4])));
                    }
                }
                else
                {
                    p += cnt;                                // skip non-tet block rows
                }
            }
            return tets.ToArray();
        }

        static int ResolveTag(Dictionary<int, int> map, string tagStr)
        {
            int tag = int.Parse(tagStr);
            if (!map.TryGetValue(tag, out int idx))
                throw new InvalidDataException($"[MshLoader] element references undefined node tag {tag}.");
            return idx;
        }

        static int SeekSection(string[] lines, string name, int from)
        {
            for (int i = from; i < lines.Length; i++)
                if (lines[i].Trim() == name) return i;
            throw new InvalidDataException($"[MshLoader] section '{name}' not found.");
        }

        // ── Boundary extraction ──────────────────────────────────────────────────────────────
        struct FaceRec { public int count; public int a, b, c, opp; }

        /// <summary>
        /// Extract the boundary surface: a tet face belongs to the boundary iff it is shared by
        /// exactly ONE tet. Each boundary face is emitted OUTWARD-wound (normal away from the tet's
        /// 4th/opposite vertex, which lies on the interior side). design §1.
        /// </summary>
        public static MshSurface ExtractBoundary(MshMesh mesh)
        {
            var faceMap = new Dictionary<long, FaceRec>(mesh.tets.Length * 2);
            int4[] tets = mesh.tets;
            for (int t = 0; t < tets.Length; t++)
            {
                int4 v = tets[t];
                // 4 faces (3 verts) + the opposite (interior-side) vertex of each.
                AddFace(faceMap, v.x, v.y, v.z, v.w);
                AddFace(faceMap, v.x, v.y, v.w, v.z);
                AddFace(faceMap, v.x, v.z, v.w, v.y);
                AddFace(faceMap, v.y, v.z, v.w, v.x);
            }

            var tris = new List<int3>();
            foreach (var kv in faceMap)
            {
                FaceRec f = kv.Value;
                if (f.count != 1) continue;                  // interior face (shared by 2 tets) → not boundary

                float3 a = mesh.nodes[f.a], b = mesh.nodes[f.b], c = mesh.nodes[f.c], o = mesh.nodes[f.opp];
                float3 n = math.cross(b - a, c - a);
                // Outward = AWAY from the opposite (interior) vertex o. If n points toward o (dot>0), flip.
                if (math.dot(n, o - a) > 0f) tris.Add(new int3(f.a, f.c, f.b));
                else                          tris.Add(new int3(f.a, f.b, f.c));
            }
            return new MshSurface { verts = mesh.nodes, tris = tris.ToArray() };
        }

        static void AddFace(Dictionary<long, FaceRec> map, int x, int y, int z, int opp)
        {
            long key = FaceKey(x, y, z);
            if (map.TryGetValue(key, out FaceRec rec)) { rec.count++; map[key] = rec; }
            else map[key] = new FaceRec { count = 1, a = x, b = y, c = z, opp = opp };
        }

        // Canonical key from the SORTED vertex triple (orientation-independent).
        static long FaceKey(int x, int y, int z)
        {
            int s0 = x, s1 = y, s2 = z, tmp;
            if (s0 > s1) { tmp = s0; s0 = s1; s1 = tmp; }
            if (s1 > s2) { tmp = s1; s1 = s2; s2 = tmp; }
            if (s0 > s1) { tmp = s0; s0 = s1; s1 = tmp; }
            return (s0 * KEY_MULT + s1) * KEY_MULT + s2;
        }

        // ── Helpers ──────────────────────────────────────────────────────────────────────────
        /// <summary>Axis-aligned bounding box over the mesh nodes.</summary>
        public static void Bounds(MshMesh mesh, out float3 min, out float3 max)
        {
            min = new float3(float.PositiveInfinity);
            max = new float3(float.NegativeInfinity);
            for (int i = 0; i < mesh.nodes.Length; i++)
            {
                min = math.min(min, mesh.nodes[i]);
                max = math.max(max, mesh.nodes[i]);
            }
        }

        /// <summary>
        /// Watertight (closed-manifold) check: every boundary edge must be shared by exactly 2
        /// boundary faces. The pseudonormal sign method (MeshLevelSet) requires this; a false return
        /// signals the winding-number fallback. design §1.
        /// </summary>
        public static bool IsWatertight(MshSurface s)
        {
            var edgeCount = new Dictionary<long, int>(s.tris.Length * 2);
            for (int i = 0; i < s.tris.Length; i++)
            {
                int3 t = s.tris[i];
                AddEdge(edgeCount, t.x, t.y);
                AddEdge(edgeCount, t.y, t.z);
                AddEdge(edgeCount, t.z, t.x);
            }
            foreach (var kv in edgeCount) if (kv.Value != 2) return false;
            return true;
        }

        static void AddEdge(Dictionary<long, int> map, int u, int v)
        {
            int lo = math.min(u, v), hi = math.max(u, v);
            long key = lo * KEY_MULT + hi;
            map[key] = map.TryGetValue(key, out int c) ? c + 1 : 1;
        }
    }
}
