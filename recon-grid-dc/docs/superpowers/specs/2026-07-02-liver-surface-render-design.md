# Liver Surface Rendering — Design (port of `PhotorealisticLiver` from the Simulation project)

Date: 2026-07-02
Status: **APPROVED (3 review rounds)** — round 1 wf_ce80c1fe (R1/R3 majors + 6 minors, folded in);
round 2 wf_9078a66a (V1/V2 approve; 5 doc-precision minors folded in); round 3 wf_43e82691
(A1 cut-texture-behavior + A2 CPU/GPU-placement approve, A3's major = the impossible "blend band"
claim corrected to the hard-edge truth + 7 minors folded in: Vector4[] upload, live mesh.bounds,
serialized Shader field, z-fight note, editor/player perf gates, tumble-crossfade & slit-leak &
mirror-band residuals documented in §7c, C#-vs-CUDA verdict in §7b)
Source of truth (reference, read-only): `D:\Desktop\Tissue_Simulation\Simulation\Assets\SurgicalSim\`
  - `Shaders/PhotorealisticLiver.shader`  — THE surface-texture algorithm to reproduce
  - `Rendering/SofaUnityVisualLiverRenderer.cs` — how the reference feeds rest positions / materials
  - `Assets/Texture/liver2.png`, `liver-texture-square_bump.png`, `liver2_spec.png` — texture assets
Target: `SurgicalSim_DC` (this repo) — the CUDA DC liver (`LiverCudaManager` + `CudaSurface.shader`).

---

## 0. Problem statement

The DC liver currently renders with a flat single-color Lambert surface shader
([CudaSurface.shader](../../../Assets/ReconGridDC/Cuda/CudaSurface.shader): `o.Albedo = _Color.rgb`).
The user wants the photorealistic liver look of the reference project, whose entire surface-texture
algorithm lives in `PhotorealisticLiver.shader`:

1. **Triplanar albedo + normal + spec mapping driven by REST positions** (Ben Golus Whiteout blend)
2. **Cook-Torrance GGX** microfacet specular (`Liver_D_GGX` / `Liver_G_Schlick` / `Liver_F_Schlick`)
3. **SSS approximation**: wrapped diffuse + backlight transmission (GPU Gems ch.16)
4. **Schlick Fresnel rim** ("wet" edge highlight)
5. Double-sided rendering with VFACE + `ddx/ddy` geometric-normal sanity flip

## 1. Why the reference algorithm is a PERFECT fit for the DC mesh

The DC surface is a **non-indexed triangle soup rebuilt every frame** (positions + normals from
`k_ExpandVertices`), with **no UV atlas, no tangents, and topology that changes on every cut**.
Classic UV-mapped texturing is impossible here. The reference solves exactly this class of problem:

- **Triplanar mapping needs no UVs and no tangents** (Whiteout blend constructs the normal
  perturbation from the world normal alone).
- **Sampling by REST position** (reference: `restOS` written by the body sim into UV1/TEXCOORD1)
  makes the texture *stick to the material* — it does not swim when the liver sags, swings, or a
  piece falls. Deformed-position triplanar would slide visibly under gravity.

So the port = the reference fragment algorithm, verbatim, plus a small plugin change to supply a
per-vertex **rest anchor** (our analog of `restOS`) and a **wall flag** (cut wall vs skin).

## 2. Pipeline reality: URP 12 → Built-in port

| Fact | Value |
|---|---|
| Reference project pipeline | URP 12.1.15 (`com.unity.render-pipelines.universal` in manifest) |
| Target project pipeline | **Built-in** (no SRP package in `Packages/manifest.json`) |
| Color space | **Linear in BOTH projects** (`m_ActiveColorSpace: 1`) → all tuned color/strength constants carry over unchanged |

The algorithm body (sections 3.1–3.5 below) is pipeline-agnostic HLSL math and is copied
**verbatim** with the `Liver_` prefix kept. Only the lighting *plumbing* is swapped:

| URP construct (reference) | Built-in replacement (port) |
|---|---|
| `TEXTURE2D/SAMPLE_TEXTURE2D` | `sampler2D` / `tex2D` |
| `UnpackNormalScale(n, s)` | `UnpackNormal(n)` then `n.xy *= s` — **no z recompute** (review R3-m2: the Whiteout blend consumes only `tn.xy`, reference shader lines 194-196, so z is dead either way; skipping it also avoids the URP scale-order mismatch) |
| `GetMainLight()` + shadowCoord | `_WorldSpaceLightPos0.xyz` (directional), `_LightColor0.rgb`; **shadow receive (R3-major, complete recipe):** `#pragma multi_compile_fwdbase`, `#include "UnityCG.cginc"`, `"Lighting.cginc"`, `"AutoLight.cginc"`; v2f MUST name the clip-space output `float4 pos : SV_POSITION` and appdata `float4 vertex : POSITION` (the built-in macros reference these names); add `UNITY_SHADOW_COORDS` at the next free TEXCOORD (**4** in the final v2f — the port drops the reference's dead `positionOS` varying), `UNITY_TRANSFER_SHADOW(o, float2(0,0))` in vert, `UNITY_LIGHT_ATTENUATION(atten, i, i.positionWS)` in frag → `atten` replaces the reference's `mainLight.shadowAttenuation` |
| `SampleSH(N)` | `ShadeSH9(half4(N,1))` (same ×0.8 damping as reference) |
| `GetWorldSpaceViewDir` | `_WorldSpaceCameraPos - worldPos` |
| `_ADDITIONAL_LIGHTS` loop | **omitted in v1** — the DC scene has exactly one directional light (verified). A `ForwardAdd` pass is the documented follow-up if point lights are ever added. |
| URP ShadowCaster pass (`ApplyShadowBias`) | Built-in `ShadowCaster` pass with `V2F_SHADOW_CASTER` / `TRANSFER_SHADOW_CASTER_NORMALOFFSET` / `SHADOW_CASTER_FRAGMENT(i)` + `#pragma multi_compile_shadowcaster`. **Both passes' vertex functions must take their input as `(appdata v)`** — the built-in macros reference `v.vertex` / `v.normal` textually (V2-m1) |
| `VFACE` | same (`fixed facing : VFACE`) — **requires `#pragma target 3.0`** (R3-m1; also covers the `ddx/ddy` use); Unity's CGPROGRAM default is 2.5 |

## 3. The shader: `ReconGridDC/LiverSurface` (new file `Assets/ReconGridDC/Cuda/LiverSurface.shader`)

Single SubShader, two passes: `ForwardBase` (Cull Off) + `ShadowCaster` (Cull Off). ASCII-only
source (the repo builds on a cp950 console; no Chinese comments in the shader file).

### 3.1 Verbatim-ported functions (byte-level math parity with the reference)

`Liver_SafeNorm`, `Liver_TriplanarAlbedo`, `Liver_TriplanarSpec`, `Liver_TriplanarNormal`
(Whiteout blend incl. the `nX.zyx` / `nY.xzy` swizzles), `Liver_D_GGX`, `Liver_G_Schlick`,
`Liver_F_Schlick`, `Liver_GGXSpecular`, `Liver_SSS` (wrap + `pow(saturate(dot(V,-L)),_SSSPower)`
backlight), `Liver_FresnelRim`.

Albedo composition — copied exactly, including the tuned constants:

```hlsl
half3 tinted = texColor * baseColor * 3.5;          // reference line: brighten dark texture
tinted      = saturate(tinted);
half3 albedo = lerp(baseColor, tinted, saturate(_TextureStrength));
albedo       = max(albedo, baseColor * 0.30);       // never below 30% of base (no black patches)
```

Fragment normal chain — copied exactly: `ddx/ddy` geometric normal, `dot(vertN, geoN) < -0.25`
flip, VFACE sign, triplanar normal, `lerp(baseNWS, N, saturate(_NormalStrength))`.

Spec-map coupling — copied exactly (review R1-m3: these three formulas are part of the verbatim
contract, reference shader lines 334-344):

```hlsl
float roughness = max(lerp(_Roughness, _Roughness * (1.0 - specFromMap * _SpecMapStrength), 1.0), 0.04);
half3 F0        = lerp(half3(0.04, 0.04, 0.04), _SpecularColor.rgb, 0.35);
half  specBoost = 1.0 + specFromMap * _SpecMapStrength;   // multiplies the GGX term
```

Final composition: `ambient + diffuse + specular + sss + fresnel` with
`diffuse = albedo * lightC * wrappedDiffuse * shadow` (same wrapped-diffuse formula).

### 3.2 Inputs replacing the reference's vertex data

| Reference input | Reference source | Our source |
|---|---|---|
| `positionOS` / world pos | mesh (identity-transform world) | mesh vertices (already world-space; transform is identity) |
| `normalOS` (triplanar weights) | mesh normals | mesh normals (CUDA-computed; object==world here) |
| `restOS` (TEXCOORD1) | body sim writes rest positions into UV1 | **UV1 = `float4(restAnchor.xyz, wallFlag)`** from the plugin (§4) |
| `tangentOS` | mesh tangents (only used by `GetVertexNormalInputs`; triplanar ignores tangents) | not needed — omitted |

### 3.3 The one intentional addition: cut-wall tint (uses a reference property that URP shader reserved)

The reference declares `_InteriorColor ("Cut Interior Color")` and its header comment says the cut
interior is rendered by a separate cut-surface path. In our DC mesh, skin and wall triangles live
in ONE soup. **Wall identification (review R1-major correction):** the bit31 tag alone CANNOT
distinguish wall from skin — post-Stage-7, `StitchVertId` resolves the SKIN stitch of every
cut+occupied voxel to the same bit31 cut-FP tag (dc_recon.cu `StitchVertId`, "skin and wall quads
SHARE the vertex"), so a bit31-derived flag would paint a ~1-voxel matte interior band along the
cut path on the OUTSIDE skin. Instead the flag is **per-emitter**: `EmitCutQuad` (the ONLY wall
emitter, in cut.cu) ORs **bit30** (`0x40000000`) into the tags it writes to `d_tri`. This flows
through every existing consumer untouched — index decode everywhere masks `& 0x3FFFFFFF` (strips
bit30), cut-FP detection everywhere tests bit31 only (`k_NormalsScatter`, `k_ExpandVertices`,
`k_DebugScanTris`, and `EmitCutQuad`'s own `tag & 0x3FFFFFFF` fetch) — zero behavior change
outside the new `aux.w = (u >> 30) & 1` read in `k_ExpandVertices`. Skin stitches that merely
share a cut FP keep `aux.w = 0` (full surface texture + fresnel — they are exterior surface).
The port then folds the interior look into the same pass using the interpolated wall flag `f`
(UV1.w):

```hlsl
albedo    = lerp(albedo, _InteriorColor.rgb, f);   // wall -> darker parenchyma color
specular *= (1.0 - 0.7 * f);                        // wall = matte moist interior
fresnel  *= (1.0 - f);                              // no wet rim inside the cut
// SSS kept at full strength on walls (interior tissue scatters at least as much)
```

**Seam behavior (corrected per round-3 A1-m1/A3-major):** `f` is binary PER TRIANGLE — k_Stitch
skin quads never carry bit30 and EmitCutQuad walls are all-bit30, and in the non-indexed soup each
emitted index gets its OWN aux copy (a shared cut FP is w=0 in its skin copies, w=1 in its wall
copies; no bleed). So the lip is a HARD f=0/f=1 transition meeting exactly at the shared component
FP vertex. This is the CORRECT "texture is cut" behavior: exterior skin texture stops precisely at
the wound edge and never smears inside; the wall starts as interior color at the same vertex,
watertight. (If a soft lip is ever wanted, it needs shader-side distance-based softening — not
vertex interpolation; out of scope v1.)

### 3.4 Properties

All reference properties with the reference defaults, verbatim:
`_MainTex`, `_Color (0.42,0.045,0.030)`, `_InteriorColor (0.24,0.018,0.014)`,
`_TextureStrength 0.55`, `_TriplanarScale 6.0`, `_TriplanarBlend 5.0`, `_NormalMap`,
`_NormalStrength 0.8`, `_SpecMap`, `_SpecMapStrength 0.8`, `_Roughness 0.40`,
`_SpecularStrength 0.8`, `_SpecularColor (0.95,0.90,0.85)`, `_SSSColor (0.80,0.15,0.05)`,
`_SSSStrength 0.35`, `_SSSDirect 0.30`, `_SSSPower 6.0`, `_SSSWrap 0.35`,
`_FresnelStrength 0.20`, `_FresnelPow 4.0`.

One scale note (arithmetic corrected per review R1-m2): the reference `_TriplanarScale = 6.0` was
tuned for a SOFA-scale liver — ASSUMED ~0.1–0.3 units (not verifiable from the permitted reference
files; confirm visually at first run). Our grid is ~17.8 world units across (48 voxels × L≈0.37).
Equivalent texel density: tiles-across-liver = size × scale = 0.6–1.8 → ours needs
`s = (0.6–1.8)/17.8 ≈ 0.033–0.1`. Ship default **0.07** in the material setup code (shader
Property default stays 6.0 = reference-verbatim; the C# material setter writes the DC-scale value;
exposed in the inspector for taste — raise toward 0.2 for denser close-up detail if desired).

## 4. Plugin change: per-vertex rest anchor + wall flag (`d_expandAux`)

The ONLY native change. Render-only; zero physics/cutting code touched.

### 4.1 Buffers (dc_recon.cu)

- `float4* d_voxelExternalFPRest` `[voxelCount]` — snapshot of `d_voxelExternalFP` taken ONCE at
  the end of `recon_init` (the grid is at rest there by construction: `recon_init` runs the first
  `recon_build_static` on the untouched rest `cornerPos` upload). Exact rest-space skin FPs.
- `float4* d_expandAux` `[3 * triCapacity]` — per emitted index: `xyz` = rest anchor, `w` = wall
  flag. Written by `k_ExpandVertices` alongside `expandPos/expandNrm`.

### 4.2 `k_ExpandVertices` extension

```
wallFlag = (u >> 30) & 1                      // bit30 = emitted by EmitCutQuad (see §3.3)
tag bit31 set (cut FP, slot s), voxel v = s>>3:
    anchor = voxelExternalFPRest[v].w > 0 ? voxelExternalFPRest[v].xyz          // surface voxel
                                          : cornerPosRest[voxelCorner[8*v]] + 0.5*L  // rest voxel center
    aux = float4(anchor, wallFlag)
tag plain voxelId v (skin FP):    aux = float4(voxelExternalFPRest[v].xyz, 0.0)
```

Requires one more tiny snapshot: `float3* d_cornerPosRest` `[cornerCount]` — copy of the uploaded
rest `cornerPos` taken in `recon_init` (cornerCount × 12 B — a few hundred KB at the 48-long-axis
grid). The rest lattice is uniform, so `rest corner v000 + (L/2,L/2,L/2)` IS the voxel's rest
center — one fetch, no origin plumbing (CornerOffset[0] = (0,0,0) verified in cut.cuh and
GridConventions.cs; BackgroundGrid fills voxelCorner in that order).

### 4.3 Rest-anchor policy for cut FPs (accepted approximation)

A cut FP has no exact rest twin without recomputing the whole component-FP chain on rest corner
positions (cost + complexity not justified for texturing). Policy:

- **Deep interior walls** (`voxelExternalFPRest[v].w == 0` — the voxel never crossed the
  isosurface): anchor = the voxel's REST CENTER (per review R1-m4 — a `(0,0,0)` sentinel would put
  a multi-unit anchor gradient across wall triangles that span a surface-voxel vert and a deep
  vert, streaking the triplanar NORMAL/spec in the highlight; rest centers keep anchors spatially
  coherent everywhere, so the wall's normal detail stays clean).
- **Surface cut voxels** (skin lip / mixed triangles): anchor = the voxel's rest skin FP —
  piecewise-constant per voxel. Worst case ≈ one voxel (L) of texture stretch confined to the
  1-voxel lip ring around the cut. At `_TriplanarScale ≈ 0.07` that is <3% of one texture tile —
  visually negligible next to an open wound. Documented improvement path if it ever shows:
  per-slot rest FP = component's rest-corner mean over `d_cornerPosRest` (10-line kernel).

### 4.4 Export + C# upload

- New export `LCS_GetSurfaceAux(float* outAux4)` (plugin_api → `recon_get_surface_aux`), same
  clamped-count contract as `recon_get_surface`. Kept separate from `LCS_GetSurface` so the
  existing 3-arg call sites stay ABI-untouched.
- `LiverCudaManager.BuildOrUpdateMesh` (upload path corrected per round-3 A2-m2/A3-m2): read into
  a pre-allocated `float[4*maxIdx]`; fill a pre-allocated **`Vector4[maxIdx]` array** in the
  existing per-vertex conversion loop; upload via the array overload
  `mesh.SetUVs(1, auxArr, 0, N)` (available since 2019.3) — consistent with the surrounding
  verts/norms pattern and measurably faster than a `List<Vector4>` Clear+Add (~0.3-0.4 ms vs
  0.5-2 ms at 300k). While in that loop, ALSO track vertex min/max and set `mesh.bounds` from the
  ACTUAL geometry (round-3 A3-m1: the current fixed grid-sized bounds make a piece falling past
  ~2× the grid extent pop out of the frustum — and would shadow-cull the new ShadowCaster pass
  even earlier).
- Readback growth: +16 B/vertex ≈ +5 MB/frame at ~300 k emitted indices — same order as the
  existing pos+nrm readback (24 B). The frame already syncs; measure, and if it matters, pack to
  half4 later (documented option, not v1).

## 5. Texture assets

Copy from the reference project (assets, not code) into `Assets/ReconGridDC/Textures/`:

| File | Role | Import settings |
|---|---|---|
| `liver2.png` | albedo (`_MainTex`) | sRGB on, wrap Repeat, bilinear |
| `liver-texture-square_bump.png` | normal map (`_NormalMap`) | **Texture Type = Normal map**, wrap Repeat. DELIBERATE deviation from the reference import (Default + sRGB ON — verified in its .meta): URP's `UnpackNormalmapRGorAG` tolerates the Default import, but built-in `UnpackNormal` requires the NormalMap (DXT5nm/linear) import; accepted minor look difference (stage-2 review S2-m3) |
| `liver2_spec.png` | spec/roughness modulation (`_SpecMap`) | **mirror the reference's `.meta` sRGB flag** (R3-m3: look parity requires the same import; check `liver2_spec.png.meta` `sRGBTexture` at implementation time — if the reference shipped sRGB ON, keep it ON; "sRGB off" is only the fallback if the meta is unreadable), wrap Repeat |

`liver2_height.png` is NOT used — the reference `PhotorealisticLiver.shader` never samples a
height map (parallax appears only in its URP-Lit factory variant, out of scope).

`LiverCudaManager` gets three serialized `Texture2D` fields (assigned in the inspector to the
copied assets), a **serialized `Shader liverSurfaceShader` field** (round-3 A3-m4: `Shader.Find`
returns null in PLAYER builds unless the shader is referenced by an asset or Always-Included —
the serialized reference guarantees build inclusion; `Shader.Find` + a logged warning stays as
the fallback), and a `useLiverSurfaceShader = true` toggle. Material creation: use the serialized
shader, assign textures + DC-scale `_TriplanarScale`; fall back to the current `CudaSurface` path
if the shader or textures are missing (never a black/pink screen).

## 6. What does NOT change

- CUDA physics / cutting logic: untouched. The only cut.cu change is ORing `0x40000000` into the
  wall tags — either at tag composition in `k_BuildCutTriangles` (`tag[s] = 0x80000000|slot`,
  cut.cu:847) or at `EmitCutQuad`'s two d_tri writes (cut.cu:795/800); both sites are transparent
  to the internal `& 0x3FFFFFFF` fetch (cut.cu:775) and to every existing d_tri consumer. DC
  reconstruction gains only the aux write in the existing expand pass (one extra `float4` store
  per index).
- `[DEBUG-CUT]`, diagnostics, tear law: untouched.
- Existing `CudaSurface.shader` stays as the fallback material path.
- Mesh channels: positions, normals, UV1(float4) — no tangents, no UV0, no index-format change.

## 7. Verification checklist (after implementation)

1. **Static look**: liver shows the reference's mottled dark-red surface with visible bump detail
   and a moist highlight; no black patches (30% albedo floor active).
2. **No texture swim**: under gravity sag and swing, the texture pattern stays glued to the
   surface (rest-anchor triplanar). A deformed-position implementation would visibly slide —
   this is the acceptance test that UV1 is actually rest data.
3. **Cut look**: walls render `_InteriorColor` matte parenchyma; the lip is a HARD, clean
   skin/interior transition exactly at the wound edge (corrected per round-3 — a blend band is
   impossible and NOT desired: exterior texture must never appear inside the wound); no wet rim
   inside the wound. NOTE (A3-m3): transient z-fighting on freshly cut, NOT-yet-separated walls
   (two coincident opposite-winding quads) is pre-existing double-emission geometry from cut.cu,
   not a shader defect — it resolves as the lips separate by D.
4. **Both sides lit** (Cull Off + VFACE): inspect the inside of a cut-open piece.
5. **No regressions**: cutting/physics identical (render-only); `[DEBUG-CUT]` counters unchanged;
   FPS delta **< 2 ms in-editor / < 1 ms in player** at the 300 k-index worst case (round-3 A2
   measured: aux fill ~0.3-2 ms + ~1-1.6 ms PCIe on top of the pre-existing copy path; the
   documented half4-packing fallback stands if profiling demands it).
6. **Shadows**: liver still casts/receives the directional shadow (fwdbase + ShadowCaster pass).

## 7b. C# vs C++/CUDA placement verdict (round-3 A2, measured)

**C# is enough — no new C++/CUDA modules are needed.** Texture SAMPLING is GPU shader work in any
pipeline regardless of host language; the only data generation this feature needs (rest anchor +
wall flag) is one `float4` store per emitted index inside the EXISTING CUDA expand kernel (already
in this design, ≈0 cost) plus two init-time snapshots (microseconds). A "pure C#" alternative
would be strictly WORSE: it would need a per-frame `d_tri` tag readback plus a ~300 k-iteration
managed slot→anchor loop.

Measured budget at the 300 k-index post-cut worst case: aux fill ~0.3–0.4 ms (RyuJIT; 1–2 ms Mono
editor) + ~1–1.6 ms PCIe — on top of a PRE-EXISTING copy path already costing ~3–5 ms there. The
bottleneck is the existing readback/rebuild architecture, NOT this feature. Staged plan:

1. **Ship v1 as-is** — adequate for a 60 fps desktop demo at the 48-voxel grid.
2. If profiling shows the copy path mattering: ~1-day C# pass — packed vertex struct +
   `NativeArray` + `Mesh.SetVertexBufferData` + persistent vertex-buffer params (kills all
   managed per-element loops and `mesh.Clear()`; saves ~3–8 ms worst-case).
3. CUDA↔D3D11 interop (`cudaGraphicsD3D11RegisterResource` into a `GraphicsBuffer` +
   `DrawProcedural`, zero readback) ONLY when VR/haptics 90–120 fps or a ~96-voxel grid upscale
   (~8× indices) is on the table.

## 7c. Accepted v1 rendering residuals (round-3, all minor, documented improvement paths)

- **Tumbling piece projection crossfade** (A1-m2): triplanar blend weights use the CURRENT normal
  (reference-verbatim), so a piece that ROTATES while falling slowly crossfades between the three
  planar projections (pattern morph, not slide; pure translation unaffected). Improvement path:
  snapshot rest voxel normals at `recon_init` (alongside `voxelExternalFPRest`) and use them for
  the weights/Whiteout basis while keeping current normals for lighting.
- **Slit light leak** (A1-m3): inside the D=0.16 L slit the directional shadow bias exceeds the
  slit width → both walls render lit with full SH ambient. The matte dark `_InteriorColor` hides
  it; optional one-liner if too bright: attenuate ambient by `(1 - 0.5*f)` on walls.
- **Mirrored texel band across the slit** (A1): both lips inside ONE voxel share that voxel's rest
  anchor → a one-voxel-wide mirrored-texture band (~2.6 % of a tile at scale 0.07) — invisible in
  practice next to an open wound; improvement path = per-component rest-corner-mean anchors.

## 8. Risks / notes

- **Gamma vs Linear**: both projects Linear — constants carry over. (Verified in ProjectSettings.)
- **cp950 console**: shader file ASCII-only.
- **`RecalculateNormals` absent** — our normals come from CUDA (area-weighted, cut-aware); the
  triplanar weight vector uses them directly; no tangent dependency anywhere.
- **Non-indexed soup + `SetUVs` order**: `mesh.Clear()` then vertices→normals→UV1→indices, same
  as today's order with UV1 inserted — Unity requires vertex count consistency, which the single
  `N` guarantees.
- **VFACE on flipped normals**: reference logic already handles our occasional inverted soup
  triangles (`dot(vertN, geoN) < -0.25` flip) — keep verbatim.
- **Scene serialization**: new serialized fields default sensibly (`useLiverSurfaceShader=true`,
  textures null → fallback path); the existing scene keeps working without edits until the user
  assigns textures.

## 9b. RENDER CORRECTION (2026-07-02, post-first-run — diagnosis wf_413cc887, 3 agents, high conf.)

The first in-editor run rendered a pale flesh-pink body with one half-liver-sized mottled patch —
NOT the user's expected deep red-brown fine-grained wet look. Root causes (all three confirmed):

1. **Wrong look target**: the user's expected screenshot is the reference's **CuttingV3 pipeline**
   whose tissue shader is `LiverTissueGPU.shader` (texture-NATIVE albedo: `saturate(tex ×
   tissueTint(1.04,0.68,0.52) + base×0.08) × 0.82` — liver2's OWN hue), not PhotorealisticLiver's
   `tex × darkred × 3.5` law (which caps G≤0.157/B≤0.105 and crushes the texture hue; the white
   F0≈0.36 + skybox SH then wash the clipped result to pale beige).
2. **Tiling 10-20× too coarse**: 0.07 × 17.8u ≈ 1.25 tiles across the liver — liver2.png's macro
   blotches blew up to half-liver patches (the §3.4 reference-size assumption was wrong, as its
   own "confirm visually at first run" flag anticipated).
3. **Bump meta defect**: the bump PNG is GRAYSCALE (a height map); `convertToNormalMap: 0` made
   `UnpackNormal` decode garbage — no micro normal grain.

FIX (property/meta-only; shader algorithm UNCHANGED — the LiverTissueGPU MULTIPLICATIVE term is
reproduced exactly through the existing `tinted = tex × _Color × 3.5` pipeline by algebra; its
small `+base×0.08` additive term is approximated by our 30% albedo floor — review C1 quantified
the residual as a modest dark dusty shift in B only, with the one-line floor tweak deferred):
`triplanarScale = 0.55` (≈10 tiles; band 0.45-0.67; serialized-0.07 scenes migrated in code);
`_Color = (0.2437, 0.1593, 0.1218)` (= tissueTint × 0.82 / 3.5); `_TextureStrength = 1.0`;
wetness pack `_Roughness 0.12` (effective ≈0.072 with the measured spec-map mean 0.497 ≈ reference
smoothness 0.939), `_SpecularStrength 1.0`, `_SpecularColor (0.30,0.27,0.25)` (F0≈0.13),
`_FresnelStrength 0.5`, `_SSSStrength 0.12`, `_SSSDirect 0.15`; bump meta `convertToNormalMap: 1`.
Deferred options if contrast still short after visual check: drop the 30% albedo floor;
port LiverTissueGPU's procedural micro-mottle/vein noise (rest-space, scale-independent grain —
also breaks the ~10× macro-blotch repetition C1 noted). Scale-change ripple (C2-m2): the cut-lip
mirrored-anchor band quoted elsewhere as "<3% of a tile" at scale 0.07 is ~14-20% at 0.55 —
accepted pending the visual check (the texture is near self-similar at grain scale). Wall shading
also inherits the wetness pack (tighter glints, calmer SSS wash inside cuts — intended per G3).

## 9c. RENDER CORRECTION v3/v3.1 (2026-07-02, second visual pass — wf_7e66fcc2 + wf_57622e7c + wf_35f93c51)

§9b targeted the WRONG material: with default flags the reference HIDES the LiverTissueGPU tet
surface (SoftBody.cs:917) and shows `CreateProject2Liver2Material` — URP/Lit, **BaseColor
0.7205882 gray × liver2 tiling(2,2)**, Metallic 0.184, **Smoothness 0.939** (the visible wet
sheen), bump=liver2_spec-as-normal 1.03, parallax liver2_height 0.08, Cull Off; scene directional
light **intensity 2.0**. v3: NEW `_EnvReflStrength` env-reflection term in LiverSurface.shader
(skybox probe × Schlick × roughness-mip, ×(1−f) on walls); pack `_Roughness 0.061`,
`_TextureStrength 1`, `_FresnelStrength 0.1`, SSS off, `_EnvReflStrength 1`; `triplanarScale
0.112` (= tiling 2 / 17.8u; migrates BOTH retracted 0.07 and 0.55); `matchReferenceLighting=true`
→ directional 2.0 at Start. **v3.1 (blocker)**: Color-typed properties are sRGB→linear converted
at BIND, so §9b's ÷3.5/×3.5 split did NOT commute (albedo was 3.9× dark). Conversion-proof law:
**albedo = _Color × tex** (×3.5 DELETED; 30% floor DELETED — it desaturated G/B under a gray
base) with `_Color = 0.7205882` = the reference's own BaseColor (GPU-exact by construction);
`_SpecularColor (0.338,0.207,0.178)` = exact-hit F0 (0.059,0.038,0.035). Kept deviations: real
converted bump (not spec-as-normal), parallax + offset(0.15,0) omitted. Supersedes §9b's values.

## 9. Deliverables

1. `Assets/ReconGridDC/Cuda/LiverSurface.shader` (new, Built-in, 2 passes, `#pragma target 3.0`)
2. `Assets/ReconGridDC/Textures/` + 3 copied PNGs (+ import metas, spec-map sRGB mirrored from
   the reference meta)
3. `dc_recon.cu/.h`: `d_voxelExternalFPRest` + `d_cornerPosRest` snapshots, `d_expandAux`,
   `k_ExpandVertices` aux write (bit30 wall flag + rest anchor), `recon_get_surface_aux`
4. `cut.cu`: `EmitCutQuad` ORs bit30 into its emitted tags (wall marker; transparent to all
   existing consumers — every idx decode masks `& 0x3FFFFFFF`, every cut-FP test uses bit31)
5. `plugin_api.cpp`: `LCS_GetSurfaceAux`
6. `LiverCudaManager.cs`: aux readback + `SetUVs(1)`, texture fields, material setup
   (`_TriplanarScale = 0.55` DC default — was 0.07, retracted; see §9b), fallback
7. Rebuilt `LiverCudaSim.dll` (+ md5) — same-commit deploy with the C# change (ABI: one added
   export; no struct changes)
