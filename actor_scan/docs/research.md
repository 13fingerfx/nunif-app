# Actor scan → photoreal texture/detail pipeline — research pass

Status: stage 1 (survey only, no implementation). Goal of this document is to
answer three questions: what already exists, what we'd have to build
ourselves, and whether this should be a Mac or PC (or cross-platform) tool.
§11 adds a second planning pass covering the capture rig, the overlay-fitting
design, a fidelity/provenance model, and what it'd take for this to become
the standard tool in this space rather than a personal one.

## 1. The pipeline, broken into stages

The request decomposes into five distinct problems that different tools are
good at. Nothing on the market solves all five end-to-end — every
"digital double" pipeline used in film/games (Ten24, 3dscanstore, Lightstage
facilities) is itself a chain of several specialized tools, usually with a
human in the loop.

| Stage | What it does | Hard part |
|---|---|---|
| A. Registration | Figure out where each HD photo was standing relative to the scan mesh | Photos usually aren't shot from a calibrated rig, so camera pose per photo has to be solved |
| B. Texture projection | Project photo pixels onto the scan's UVs at much higher resolution than the scan's native capture, blending seams across photos | Multi-photo blending without visible seams/lighting mismatch |
| C. Fine detail extraction | Pull pore/wrinkle-level micro-geometry out of the *photos* (the scan mesh itself is too low-res to have ever captured it) | This is a genuine research problem, not a solved one, for single uncalibrated photos |
| D. Bake as topology | Turn that extracted detail into actual displacement/normal geometry on the mesh, not just a flat color trick | Needs enough mesh resolution (subdivision) to hold the detail, and correct tangent-space alignment |
| E. Overlay library | Store detail by body region, reusable later without source photos | Storage/format problem, not a research problem — and there's a strong existing precedent (see §5) |

## 2. What's already in *this* repo that's directly reusable

`nunif` already contains two of the needed model families, which matters a
lot for stage 1 → stage 2 planning:

- **`waifu2x/`** — general image super-resolution (including GAN-based photo
  models, not just anime). This is a direct fit for "extrapolate a much
  higher resolution texture" from the HD photos before projection — upscale
  each source photo (or the assembled UV texture) before baking it down onto
  the mesh.
- **`iw3/`** — monocular depth estimation (Depth Anything v2/v3, ZoeDepth,
  DepthPro) plus stereo/warping utilities. This is genuinely useful for a
  *coarse* relief pass (general face structure, cheekbone/brow relief) but
  these models predict low-frequency relative depth — they will not recover
  pore- or wrinkle-scale detail; the frequency is below what monocular depth
  networks resolve. Treat `iw3`'s output as a sanity-check / coarse base
  layer, not the source of the "incredibly detailed" part of the request.
- **`dino/`** — DINO ViT features. These are the standard backbone for
  learned image-to-image correspondence, which is relevant to stage A
  (matching photo content to a rendering of the scan to help solve camera
  pose without manual pin placement).
- MPS support: the repo already carries `Enable autocast (FP16) on MPS`
  (iw3), i.e. this codebase already runs its PyTorch models on Apple Silicon,
  not just CUDA. That's a good sign for the Mac-viability question in §6.

## 3. Stage A/B — registration + texture projection

Two realistic situations, depending on how the photos were shot:

- **Calibrated/rig photos** (fixed camera positions, like a scanning booth):
  camera poses are known or easily solved once — trivial projection.
- **Ordinary reference photos** (the likely real case — a few HD stills, not
  a booth): camera pose per photo must be solved against the mesh.

Tools that solve this today:

- **COLMAP** — open source, cross-platform (Windows/Linux/**macOS**),
  structure-from-motion + multi-view stereo. GPU acceleration helps but
  isn't required for pose solving against an existing mesh.
- **[mvs-texturing / texrecon](https://github.com/nmoehrle/mvs-texturing)** —
  open source (BSD), exactly the missing piece for stage B: takes a
  triangulated mesh + photos with known poses and produces a seam-blended,
  multi-band texture. CPU-only, builds with cmake, no CUDA requirement — a
  strong cross-platform candidate to vendor directly into this project.
- **KeenTools FaceBuilder** (Blender/Nuke add-on, ~$18–180/yr, Win/macOS/Linux)
  — commercial but cheap, and it specifically targets "3D head + texture from
  ordinary photos of an actor," including non-neutral expressions. Good
  reference implementation / possible plugin-level dependency rather than
  something to reinvent.
- **R3DS Wrap / Wrap4D** — the industry-standard commercial tool
  (Win/macOS/Linux) for exactly this category of work: non-rigid fitting of a
  clean topology to a scan plus texture projection, node-graph "recipes"
  reusable across many scans of the same actor. This is effectively what a
  finished version of your tool would look like, if it were a commercial VFX
  product.
- Open-source non-rigid alignment building blocks for a from-scratch
  implementation: **probreg** (Coherent Point Drift and friends, plugs into
  Open3D, pure Python) and published "optimal-step non-rigid ICP" reference
  code — usable to refine a template/scan alignment without buying Wrap.

## 4. Stage C — the actual hard/novel part: pore-level detail from photos

This is where "some estimation is acceptable" matters, because nothing
off-the-shelf does this well from ordinary single photos:

- **DECA** (*Learning an Animatable Detailed 3D Face Model from In-the-Wild
  Images*) — predicts a person-specific UV displacement map (wrinkles,
  moles, pore-scale bumps) from a single image. Open source, but the
  released model is under a **non-commercial research license** — flag this
  if the end tool is ever meant to be sold or used commercially.
- **HRN** (*Hierarchical Representation Network*, 2023) — same idea, closer
  to true high-frequency detail (wrinkles/dimples) by explicitly
  disentangling geometry into low/mid/high-frequency layers before
  reconstruction. Same "research-model" caveat likely applies.
- **NextFace** — open source monocular face reconstruction; its own docs are
  explicit that it does **not** recover wrinkle/pore-level geometry from a
  single image, i.e. even the honest open-source projects concede this is
  unsolved for casual photos.
- What real digital-double studios actually do instead of solving this from
  a single photo: **photometric stereo / multi-flash Lightstage-style
  capture** (multiple lights, one camera position, per-pixel normals from
  shading differences). Originally scoped out here as "needs a multi-light
  rig you don't have" — but a small dedicated capture rig turns this from
  "not applicable" into the primary method; see §11.2.
- **The practical, buildable substitute if no multi-light capture is used**:
  classic *frequency separation* —
  high-pass-filter the luminance channel of the HD photo to isolate
  pore/wrinkle-scale micro-contrast, convert that high-frequency layer into a
  tangent-space normal/displacement map. This is the same trick
  texture artists have used for photo-bump-mapping for two decades, is
  trivial to implement in OpenCV/NumPy, needs no ML model, and is a very good
  match for "some estimation is acceptable."

Recommended approach: **combine the two.** Use frequency separation for the
micro-detail (pores, fine skin texture — this is where photos genuinely have
signal), and use a DECA/HRN-style detail-UV prediction only for the
mid-frequency, expression-correlated wrinkles (where structural, not just
photometric, plausibility matters). Feed both onto the mesh through the
camera calibration solved in stage A/B, using the scan's own UVs as the
frame of reference the request asks for ("dimensions from the model as the
authority").

## 5. Stage E — overlay library: this idea already exists commercially, which validates it

**[texturing.xyz](https://texturing.xyz/)** sells exactly the asset type
you're describing: cross-polarized photo-derived skin displacement/normal
maps, organized by body region (face sub-regions, cheeks, brow, lips, neck,
hands, etc.), split into multiple frequency bands (secondary/tertiary/micro),
explicitly meant to be applied to sculpts or scans that lack their own
captured detail in that region. That's strong validation that region-tagged,
frequency-separated detail tiles are the right storage unit — it's a proven
production pattern, not a speculative one. (Their maps are paid, licensed
assets, not something to redistribute — but they're a good reference for tile
format/resolution choices, and could be a bootstrap/fallback library the tool
ships with pointers to, or a paid asset the user already owns.)

For your own library: store each tagged region (`L_cheek`, `R_cheek`,
`throat`, `chin_under`, `eye_L_under`, `eye_R_under`, `brow`, `nose`, `lips`,
`forehead`, ...) as a small set of tangent-space normal + displacement tiles
per frequency band, independent of any single mesh's UV layout, so they can
be projected onto a different scan's UVs (or a from-scratch sculpt) later.

## 6. Stage D — baking onto the mesh

Don't build a mesh-editing/baking engine from scratch — **Blender** already
has everything needed (UV baking, multires/subdivision, displacement, and a
full Python API), is free, and is genuinely cross-platform. The realistic
"standalone application" here is a **Blender add-on with a Python/PyTorch
backend** (the same pattern KeenTools FaceBuilder uses), not a from-scratch
3D engine. That drastically cuts scope versus building a custom viewport/mesh
editor, and lets this project focus its own engineering effort on stages A–C
and E, which are the genuinely unsolved/valuable parts.

## 7. Mac vs PC vs both

Split this by what the tool actually needs to run, because the honest answer
is "it depends which part":

- **Photo super-resolution (waifu2x-style), monocular depth/coarse-relief
  (iw3-style), frequency-separation detail synthesis, and any DECA/HRN-style
  inference** — all standard PyTorch, all run fine on CUDA *or* Apple Silicon
  MPS. This repo already proves the MPS path works (`iw3` added MPS
  autocast). **Fully cross-platform, Mac included.**
- **Blender baking/UV/display workflow** — Blender itself is cross-platform
  (Win/macOS/Linux), no issue.
- **Camera-pose solving (COLMAP) and texture projection (mvs-texturing)** —
  COLMAP officially supports macOS; mvs-texturing is CPU-based C++, builds
  anywhere. **Cross-platform, no CUDA required for this use case** (you're
  registering photos against an *existing* scan, not doing dense multi-view
  reconstruction from scratch — that's the CUDA-hungry case, see below).
- **The one place PC pulls ahead**: if the tool ever needs to *generate* a 3D
  scan from photos itself (full photogrammetry reconstruction, not just
  aligning photos to a scan you already have), the strong open-source tool
  there is **Meshroom (AliceVision)**, which is Windows/Linux only with no
  official macOS build and effectively requires an NVIDIA CUDA GPU for
  anything beyond "draft" quality. Commercial equivalents (RealityCapture)
  are Windows-only too. That entire category of software is CUDA/Windows-
  centric — but it's out of scope for what you described, since you said the
  3D scan already exists and is the dimensional authority.

**Recommendation**: build cross-platform (Mac and PC both fully viable) for
everything actually described in the request — texture upscaling, photo
registration, detail extraction, baking, overlay library. Only reach for a
Windows+NVIDIA box if a later phase adds "generate the scan itself from
photos," which isn't what was asked for here.

## 8. Suggested stack (for stage 2 planning, not yet built)

- Python + PyTorch, consistent with the rest of this repo; reuse/extend
  `waifu2x` for texture upscaling and `iw3`'s depth models for the coarse
  relief sanity layer.
- COLMAP for camera pose solving; `mvs-texturing`/texrecon (vendored, BSD
  licensed) for seam-blended photo→UV projection.
- `probreg` (Open3D-based) for any non-rigid refinement between a reference
  template and the actual scan, if needed.
- Custom (OpenCV/NumPy) frequency-separation module for pore/wrinkle micro
  detail — no ML dependency, no license concerns, fully explainable.
- Optional, license-gated: DECA/HRN for structural expression-correlated
  wrinkle prediction, clearly separated from the above so the non-commercial
  research license doesn't contaminate the rest of the tool.
- Delivery as a **Blender add-on** (Python + `bpy`) rather than a bespoke 3D
  app — bake/multires/UV tooling reused, not rebuilt.
- Overlay library as region-tagged normal/displacement tile sets on disk,
  keyed by body-region tag, independent of any one mesh's UVs.

## 9. Licensing flags to keep in mind going forward

- DECA and HRN releases are typically **research/non-commercial** licenses —
  fine for prototyping, needs a substitute or a commercial license before any
  commercial use.
- texturing.xyz maps are **paid, licensed assets** — reference only, not
  redistributable, unless purchased.
- KeenTools and R3DS Wrap are **commercial subscriptions** — useful as
  reference implementations / possible complementary tools, not something to
  vendor into an open-source project.
- `mvs-texturing` (BSD) and COLMAP (BSD-ish/custom permissive) are safe to
  vendor directly.

## 10. Suggested next step (stage 2)

Before investing in DECA/HRN integration, de-risk the cheap path first:
take one actor scan and a handful of its HD photos, and prototype just
stage A (manual/COLMAP pose solve) → stage B (`mvs-texturing` projection) →
stage C via frequency separation only (no ML) → stage D (bake in Blender).
That validates the full pipeline shape end-to-end with the least engineering
investment, and tells us how much the ML-based detail stages (DECA/HRN) would
actually add over the classical approach before committing to their license
constraints.

## 11. Second planning pass — product scope, capture rig, overlay design

This section captures a follow-up round of discussion after the initial
survey above, driven by a concrete real pipeline (scan with a Revopoint
MIRACO-class device → punch up the texture/detail → 3D print or mould-and-
cast) and a goal of this becoming the standard tool for this niche, not a
one-off personal tool.

### 11.1 Product framing: scanner-agnostic, and a real price gap to bridge

The tool should assume the user already owns a scanner and never integrate
against scanner hardware directly — it should ingest whatever mesh/UV/texture
a scanner already exports (OBJ/PLY/glTF/FBX cover essentially everything from
Revo Scan, Artec Studio, Polycam, Metashape, RealityCapture, Meshroom). This
also happens to be the easiest scope to build: it's a format-parsing problem,
not a hardware-integration one.

The market gap being targeted is real and worth naming explicitly: consumer/
prosumer structured-light scanners (Revopoint MIRACO-class, roughly
£1.5–2k) are good enough for games/previz but not for 3D printing at
close-up scrutiny, while the next tier up (Artec Spider-class) is an order
of magnitude more expensive (£20–30k+). A tool that lets a £1.5–2k scan
approach £20–30k-scanner fidelity via a software + cheap-rig post-process is
a credible wedge product, including for buyers who'd otherwise be customers
of the expensive tier.

### 11.2 A purpose-built capture rig: this is an RTI dome

A rig with lights at fixed positions and a camera/phone mount in the middle
is, functionally, a **Reflectance Transformation Imaging (RTI) dome** — the
established technique museums and heritage-documentation labs use to capture
surface micro-detail (coins, carvings, manuscripts) via multiple fixed
lighting angles. That's directly useful prior art to search when this gets
built, and it validates the specific detail the user raised: because the
rig's light/camera geometry is fixed and known, the light vectors are
calibrated **once**, at build time — not per capture — which is exactly what
removes the need for a chrome-ball reference shot per session. A practical
minimum is 3 fixed lights (e.g. left/right/top) plus one no-flash/ambient
shot for clean albedo, feeding a per-pixel least-squares photometric-stereo
solve for normals (Lambertian assumption).

Because not every user will own the rig, the software should also document a
"no rig" capture recipe (any 3 lights + any camera + a tripod, subject and
camera held still) so the tool has standalone value, and the rig becomes a
paid convenience upsell rather than a hard requirement — the same shape as
OBS being free while capture-card hardware is sold around it.

### 11.3 Overlay fitting, revised: landmark-driven warp + patch-based synthesis

The boundary-editing tool described (a spline glued to a curved surface,
graduating into a warp that "morphs" a texture patch to fit) is the same
underlying technique as landmark-driven face-filter apps (Peachy/Snapchat/
TikTok-style makeup or AR filter fitting): a handful of correspondence points
(e.g. eye corner, pupil center) drive a smooth interpolated warp — a
thin-plate-spline (TPS) or RBF warp, computable with `scipy`. That confirms
the design in §5/§4: TPS warp for the *placement/shape* of a patch.

The distortion risk raised (stretching individual pore shapes when the warp
resizes a patch non-uniformly) is real, and the fix is a named, established
technique: **patch-based texture synthesis** (the family behind Photoshop's
Content-Aware Fill / PatchMatch, and the older Efros-Leung algorithm).
Instead of geometrically stretching pixels to fit a new boundary, stitch
together small real patches (from the source photo, or the exemplar library
in §11.4) to fill whatever irregular area the boundary encloses, preserving
true pore shape/size regardless of target area. This is the same family of
idea as the user's own "contextual fill" description and needs no ML model —
classical, well-understood, and its provenance (every pixel traces back to a
real captured patch) is auditable, which matters for §11.5's "trust" point.

### 11.4 A three-tier fidelity model (resolves the "idealized pore library" question)

The idea of keeping a bank of very-high-resolution, generic pore/wrinkle/
fold exemplars to extrapolate beyond what a given photo actually resolved is
sound and not novel — it's the same thing texturing.xyz already sells
commercially (see §5), used in production specifically because photographs
alone often don't resolve true pore-level detail. The one thing worth being
careful about is the analogy the user raised themselves: phone cameras that
detect "this is the moon" and substitute a stored high-detail image rather
than what the sensor actually resolved were controversial precisely because
they fabricated specific detail and presented it as observed. For a tool
whose purpose is capturing a specific real person's/object's likeness, that
move is fine in some places and not others. A frequency-based split gives a
principled line, and lines up with the pipeline already described elsewhere
in this doc:

- **Macro (the scan geometry itself)** — ground truth, the dimensional
  authority, never touched by synthesis.
- **Mid-frequency (specific wrinkle folds, moles, scars, structural
  asymmetry)** — this is what makes the result recognizably *that*
  person/object. Must always be derived from their own real photos (ideally
  via the photometric-stereo capture in §11.2), even if the estimate is
  rough. Never backfilled from a generic exemplar bank.
- **Micro-frequency (individual pore shape/density/skin grain)** —
  statistically generic across similar skin types/regions; nobody can tell a
  specific pore apart from a plausible one. This is the layer where an
  exemplar library and patch-based synthesis (§11.3) legitimately "top up"
  resolution the source photo didn't capture.

### 11.5 What it would take to become the standard tool, not a niche one

- **Interoperability as the moat**: universal import (see §11.1) and export
  to every common destination (STL for print/mould, glTF/FBX/USD for games,
  a ZBrush GoZ bridge, native Blender) — being the hub between "any scanner"
  and "any downstream tool" is what makes a tool default rather than niche.
- **Lower the skill floor**: auto-place boundary/anchor points with an
  existing open-source face-landmark model (MediaPipe FaceMesh, dlib,
  InsightFace) instead of requiring manual placement, with manual drag-to-
  correct as the fallback; pair with a guided step-by-step capture mode for
  the rig in §11.2 so users don't need to understand photometric stereo to
  use it.
- **Two-sided overlay marketplace**: let users capture and publish their own
  region-tagged detail packs (the same shape as Blender Market or
  texturing.xyz itself). Publish the pack format as an open spec (like
  glTF/USD) so nobody has to gatekeep who can produce compatible content —
  this is a much stronger growth engine than any single feature.
- **Generalize past faces**: statue/bust reproduction, creature/prop
  sculpting, museum replicas, and prosthetics are larger adjacent markets
  than digital-double VFX, and the region-tag design (user-defined tags, not
  a hardcoded face map) already supports this — worth protecting that
  generality rather than hardcoding face-specific assumptions later.
- **Make provenance visible**: expose the macro/mid/micro split from §11.4
  as a confidence/provenance heatmap in the UI (captured vs. estimated vs.
  synthesized). Differentiates for professional/semi-professional users
  (museum, forensic, prop-authentication contexts) who need to audit what's
  real, and it's nearly free once the tiering already exists internally.
- **Avoid the licensing trap**: keep any DECA/HRN-style learned component
  optional and swappable, not load-bearing, given their typical
  non-commercial research licenses (see §9) — lean on the classical
  photometric-stereo + patch-synthesis pipeline as the core specifically
  because it carries no such restriction and is compatible with "tool
  everyone can use commercially."
