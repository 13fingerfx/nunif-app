# actor_scan — build plan (v1) and flaw interrogation

Companion to [`research.md`](research.md). This is the step-by-step
implementation plan, followed by the logical-flaw pass that reshaped it.
The plan below is the *corrected* version; §"Flaws found" records what
changed and why, so the reasoning isn't lost.

## Guiding constraints (from research + discussion)

- The scan mesh is the dimensional authority. Photo-derived data may only
  add mid/high-frequency detail, never move the macro shape.
- Print/mould path is first-class: output must be real dense geometry
  (STL/OBJ/PLY), not just texture maps.
- Scanner-agnostic: ingest standard mesh formats, never talk to hardware.
- Core pipeline must be commercially unencumbered: classical CV
  (photometric stereo, TPS warping, patch synthesis), no research-licensed
  ML models in the load-bearing path.
- Works without the rig (chrome-ball calibration per session) and better
  with it (calibration baked in).

## Build order

Each phase is independently testable with synthetic data before any real
capture exists. Phases 1 and 2 have no dependency on each other and could
be built in either order; Phase 3 needs both.

### Phase 0 — scaffolding  *(this commit)*
- `actor_scan/` Python package: `core/` library modules, thin `cli.py`,
  `tests/` (stdlib `unittest`, synthetic data only, no fixtures needed).
- Dependencies: numpy, scipy, opencv-python-headless, trimesh, pillow.
  All permissive licenses. No torch in the core (waifu2x/iw3 integration
  comes later and stays optional).

### Phase 1 — detail engine (image space)  *(this commit)*
The heart of the tool; zero dependency on registration or meshes.
1. **Light calibration** (`core/lightcal.py`): chrome-ball highlight →
   incident light direction. Needed per-session without the rig; needed
   once ever with it.
2. **Photometric stereo solver** (`core/photometric.py`): ≥3 images from a
   fixed camera under known light directions → per-pixel surface normal +
   albedo (Lambertian least squares; optional intensity weighting to
   soften shadow/specular violations).
3. **Single-photo fallback** (`core/freqsep.py`): frequency-separation
   pseudo-height from one photo's luminance (dark pore → recessed). Lower
   fidelity, no extra capture required.
4. **Normal integration** (`core/integrate.py`): Frankot–Chellappa FFT
   integration of the normal field into a height map, then **band-pass**:
   subtract the low-frequency component so the scan keeps macro authority
   and photometric drift can't fight it.

Validation: render synthetic bumpy surfaces under known lights, recover
normals/height, compare against ground truth (angular error, correlation).

### Phase 2 — registration  *(this commit)*
`core/registration.py`: manual 2D↔3D correspondence points → `solvePnP`
pose, with a focal-length sweep when EXIF/intrinsics are unknown, and
reprojection-error reporting so the user can see fit quality. Automation
(COLMAP, DINO-feature matching) is a later optimization, not v1 — manual
correspondences are deterministic and testable.

### Phase 2.5 — scan repair: zones, hair detection, bald fill  *(implemented)*
Runs on the raw scan, before any detail/texture work — raw head scans
carry wig caps, tied-up hair, and beards that read as noise or junk.
- `core/zones.py`: canonical face/head zone registry (l_cheek,
  r_under_eye, under_chin, ...) with forgiving name resolution ("under
  left eye" -> l_under_eye), mirror pairs, groups, and hair-prone
  markers (scalp/beard). One shared vocabulary for overlay tags, repair,
  and future marketplace packs; custom tags stay allowed.
- `core/defects.py`: automated replace-region estimator, two classical
  signals — spikiness (vertex displacement from the neighbors' centroid
  relative to edge length; normal-based measures self-cancel on noise
  and were rejected after testing) and robust color distance from a
  skin model fitted to the scan's own smooth regions (per-scan fit, so
  any complexion works and wig caps with clean geometry still get
  caught). Speckle-filtered, ring-grown.
- `core/fill.py`: the "bald pass" — masked vertices are re-solved as a
  biharmonic continuation of the surrounding surface (slope-continuous,
  so a filled scalp reads as skull, not soap film). Moves existing
  vertices only; true topological hole closure and library-shaped
  replacement (generic ear/scalp exemplars, non-rigidly fitted) are the
  planned upgrades and will slot in as additional method= options.

### Phase 2.6 — replacement & sizing toolkit  *(implemented)*
Policy change from user direction: **manual selection is the primary
path** — the user outlines what gets replaced; automated detection
(defects.py) is demoted to an optional assist that only ever suggests.

- `core/selection.py`: the outline tool in CLI form — a closed polygon
  drawn on a registered photo back-projects to the enclosed, visible
  vertices (same interaction family as the overlay boundary spline;
  the GUI later draws the spline directly on the mesh).
- `core/fill.py` profiles: fill is no longer only "bald". PROFILES
  presets (scalp = pure skull continuation, brow = gentle ridge,
  beard = fuller dome) plus `fullness`/`taper` parameters — the same
  numbers a character-creator slider would drive interactively.
- `core/measure.py`: the 13FingerFX measurement chart as code. Caliper
  measurements are straight-line distances between named landmarks;
  tape measurements are plane-slice loops or arcs (arc side chosen by
  an "over" hint landmark). compute_chart() runs on the original scan,
  a repaired version, or a candidate replacement; compare_charts()
  reports deltas — "the replacement matches our measurements" as an
  operational check, and the numbers to display live while sizing.
- `core/eyes.py`: replacement eye forms. Eye diameter is nearly
  constant across adults (~24 mm; ~19.5 mm at birth), so presets work.
  style="sphere" is a plain ball; style="sculpted" is the
  figure-sculptor form — corneal plateau, crisp limbus ring, dished
  iris — which reads as an eye in monochrome print output instead of
  the scanned sclera-bulge horror. Placement = center + gaze vector.
- `core/project.py`: the never-erase rule made mechanical. Projects
  copy sources in read-only + hashed, all outputs are auto-versioned
  (_v001, _v002, ...), every step is journaled with its parameters,
  and guard_overwrite() refuses any output path equal to an input.

**"Is there a system we can adopt?" (character-creator sliders +
generic head)** — yes, two usable tiers:
- **MakeHuman**: open-source parametric human (slider "targets" =
  morph deltas on a base mesh). Code is AGPL but the mesh/target
  *assets* are CC0 — the base head and the morph-target paradigm are
  directly adoptable as our generic head and slider system. Blender's
  official Human Base Meshes bundle (CC0) is an alternative base.
- **FLAME** (MPI): the statistical head model behind DECA — best
  auto-*fitting* of a template to a scan/landmarks, but research
  license, commercial license purchasable. Same treatment as DECA:
  optional, swappable, never load-bearing.
- MetaHuman (Epic) and Character Creator (Reallusion) are UX
  references only — licenses bind them to their ecosystems.
Planned flow: generic CC0 head ships with the tool, auto-aligned to
the scan via the measurement landmarks (similarity transform first,
non-rigid refinement later via probreg); the user outlines the region,
the aligned generic supplies the fill target instead of a smooth
continuation (a `method="template"` in fill.py), sliders tweak, and
commit remeshes into the master form — with the original untouched in
sources/ per the project rule.

**Beard from a reference photo (design, not yet built)**: register the
pre-beard photo with the existing PnP flow; the user traces the beard
outline on the photo (selection.py already turns that into a mesh
region); the photo's silhouette edge constrains the fill boundary and
a shading-derived depth hint modulates `fullness` across the region.
Falls out of machinery that already exists — needs a real bearded/
pre-beard pair to calibrate against.

### First real-scan calibration (scan 049, wig cap + bun)
Run against a professional-rig scan (400K faces, OBJ, ~36 cm bust,
i.e. **cm units** — Miraco exports are mm, so units must never be
assumed; the CLI now prints extent + a unit guess on every mesh load).
Findings, in order of importance:

1. **OBJ scans arrive un-welded** — per-corner UV/normal indices split
   199,877 real vertices into 445,561 disconnected corners, silently
   breaking every adjacency-based operation. The CLI loader now welds
   on load (fill/bake/selection all need the connected graph). This
   was the only real bug the calibration found.
2. **Fill scales**: biharmonic solve of a 76K-vertex cranium selection
   in 6.5 s on the 200K-vertex scan. Interactive-enough for the
   commit-step workflow; sliders can operate on a decimated preview.
3. **Artifact removal works**: the wig-cap ring artifacts and the hair
   bun are completely eliminated, boundary blends cleanly into the
   face/neck rim.
4. **Shape verdict — the motivating result**: with the entire cranium
   masked, slope-continuous extrapolation from the forehead/nape rim
   produces a flat-topped "conehead" — geometrically smooth,
   anatomically wrong. Biharmonic fill is the right tool at *patch*
   scale (beard, brow, small dropouts) and the wrong prior for
   full-cranium replacement. This is exactly the case for the
   template-head fill (`method="template"`, generic CC0 head aligned
   via measurement landmarks) — now the top build priority.
5. **Auto-detection verdict**: point-wise roughness does light up the
   cap's ring artifacts but does not globally separate cap from skin
   on real data (the cap fabric is smooth; real skin carries genuine
   micro-noise at the scan's 1.15 mm edge length). Confirms the
   product decision that manual outlining is the primary path;
   `mark` stays an assist at best.

### Phase 2.7 — template-head fill  *(implemented, first version)*
`core/template.py`, validated on scan 049 with the user's average
male/female generic heads (identical 12,466-vertex topology — the
MakeHuman-style paradigm, so zones/overlays defined on one transfer to
the other for free).

Flow, matching the user's spec: the generic is invisibly **tethered**
to the scan by a similarity transform from matched landmarks (Umeyama:
rotation + translation + uniform scale only — alignment can never
distort the scan or the generic's proportions); the user's outline
selects the replacement region; the equivalent area of the generic
supplies per-vertex target positions (dense surface sampling +
KD-tree); a **screened biharmonic** solve drops that shape in —
`adherence` weighs template-following against smoothness (0 = plain
biharmonic), and the feather band zeroes template influence at the rim
so the drop-in stays glued to the scan at the boundary. Locks are
honored identically to plain fill. Sliders (adherence, fullness,
taper) + the measurement chart are the "character creator" controls.

Real-scan result: the same cranium outline that biharmonic turned into
a conehead becomes an actual skull (rounded occiput, correct crown
height) with the female generic at adherence 2.0 — 4.7 s solve, auto-
tether at x105 scale with 1.16 cm landmark rms from crude auto-picked
features (user-placed landmarks will beat that).

**Round 2 (user critique of the first result — steps at junctions,
carried-over texture, narrowed flanks, no mid-process measurements):**
- `anisotropic_align`: the generic is only a shape/ratio indicator, so
  the tether may stretch it independently in width/height/depth to fit
  the scan's landmark frame (per-axis least squares on top of the
  similarity fit). Scan dimensions outside the replaced region remain
  sacrosanct by construction.
- `edge_match` (default on) in template_fill: point-by-point boundary
  conformance. Scan-to-template offsets are measured at every rim
  vertex and harmonically interpolated across the region; targets
  become template + offset field, so the drop-in meets the scan
  EXACTLY at the rim no matter how imperfect the global tether is —
  no global transform can guarantee that, per-vertex warping can.
  Verified: a deliberately mis-tethered template produces max rim
  step 1.5mm on the real scan (visible ledges before the fix).
- `template_targets` averages k nearest surface samples — a single
  nearest sample quantizes to the sampling grid and printed through
  as micro-noise rougher than skin (caught by acceptance metrics).
- Acceptance checks now run before any result is presented: max rim
  step, zero unmoved vertices inside the region, interior roughness
  vs skin baseline, and before/after measurement-chart deltas
  (tragus-to-tragus must not change; ear-to-ear over crown reports
  the removed hair volume). Measurement readout printed per fill =
  the CLI stand-in for live slider feedback.
- Selection lesson (three failed rounds' worth): hand-transcribing a
  drawn outline loses the silhouette edge; the user's stroke is now
  extracted directly from their annotated image and registered onto
  the render projection. In the GUI this is moot (the spline lives on
  the mesh), but any import-an-annotation path must do it this way.

**Landmark doctrine (user ruling):** landmarks must NEVER sit on
removable material — hair, wig cap, debris. Tether from skin only
(ears, nose, chin, brow); dimensions no skin landmark spans (skull
height under hair) are supplied by the generic's own ratios — that is
precisely the generic's job as a "shape and dimension-ratio
indicator." The earlier estimated cap allowance is obsolete: cap
fabric is 2–3 mm and gets removed with the hair anyway. Verified on
scan 049: skin-only tether yields chin-to-crown 25.4 cm (raw
cap-inflated surface 25.2 cm) with no fudge constants. Where chart
values DO exist for a subject, they override ratios as constraints.

Still open in this phase: commit-step remeshing into master topology
(watertight commit is done; retopology to clean quads is not),
color-coded hair regions on the generics (user may supply), per-axis
width/depth sliders as interactive controls, and landmark
auto-suggestion.

### Phase 3 — bake to geometry  *(this commit)*
`core/bake.py`: subdivide the scan mesh to target edge length → project
each vertex through the solved camera → sample the detail height map →
displace along vertex normals, weighted by view angle, with occlusion
ray-testing; multi-photo contributions blended by weight. Export via
trimesh (STL/OBJ/PLY). Displacement is applied **per-vertex in 3D via the
photo projection — never through the scan's UV atlas** (see flaw 4).

### Phase 4 — shaped overlay library  *(this commit, image-space half)*
`core/overlay.py`: extract a feathered, polygon-bounded, landmark-tagged
region from a detail height map; store as a region-tagged `.npz` archive;
apply onto another detail map via thin-plate-spline warp driven by
landmark correspondences (the Peachy-style draggable-dots model).
- **4b (deferred)**: applying an overlay directly onto a mesh with no
  photo at all (scratch characters) needs a local surface
  parameterization around the user's boundary curve — planned approach is
  a local conformal unwrap of the enclosed patch, but it is not in v1.
- **4c — guided synthesis** *(implemented, `core/synthesis.py`)*:
  the "if you see x data it translates to y texture" model — image
  analogies (Hertzmann et al. 2001) / Efros-Freeman texture transfer.
  The observed detail map is a guide channel: each target tile is
  matched (plain SSD on the coarse band — deliberately not mean-
  invariant, since local level is the regime signal) against an
  exemplar, and only the exemplar's high band is transplanted, then
  re-band-passed so the guide keeps sole authority over everything it
  resolved. This is the principled midpoint between measurement and
  moon-pasting: subject-specific structure decides placement, exemplars
  contribute only sub-resolution frequencies. Also covers the "resized
  overlay pore stretching" concern: resynthesize instead of stretch.

### Phase 5 — texture path (games/preview)
UV texture projection + seam blending (vendor `mvs-texturing` or
reimplement its multi-band blend), waifu2x super-resolution integration.
Deliberately after the print path: prints don't show color.

### Phase 6 — GUI
Guided capture wizard, correspondence clicking, boundary-spline editing on
the mesh (Blender add-on and/or wx GUI like the rest of nunif). CLI-first
until the pipeline is proven on real captures.

## Flaws found during interrogation (and what changed)

1. **"Blender add-on" contradicted "standalone" + print path.** The
   research doc proposed delivering as a Blender add-on. For printing, the
   essential operation is subdivide→displace→watertight export, which
   trimesh does in-process. *Change: core bake is native; Blender becomes
   an optional Phase-6 front-end, not the delivery vehicle.*
2. **Naive integration would violate "scan is the authority."**
   Photometric stereo recovers normals, not absolute depth; FFT
   integration accumulates low-frequency drift that would bend the macro
   shape. *Change: mandatory band-pass on the integrated height — photos
   contribute only the frequency band the scanner can't capture.*
3. **Displacement amplitude is unknowable without radiometric
   calibration.** Normals give shape, not physical pore depth. *Change:
   expose `height_scale` (mesh units, typically mm) as an explicit user
   parameter with a sane default; the rig can ship a calibration target
   later. "Some estimation is acceptable" per the spec.*
4. **UV-space detail cracks at atlas seams.** Miraco-class scans have
   fragmented UV atlases; a displacement map in that UV space produces
   visible cracks at every island boundary when baked. *Change: bake
   displaces vertices by projecting each one into the photo directly; the
   scan's UVs are never in the geometry path.*
5. **Automated registration first = building on the hardest unsolved
   step.** *Change: v1 registration is user-clicked correspondences +
   PnP (+ focal sweep). Deterministic, debuggable, testable. COLMAP/DINO
   automation is an upgrade, not a foundation.*
6. **Rig assumptions must not leak into the software.** *Change:
   chrome-ball calibration is implemented from day 1; a rig profile is
   just a saved calibration file the software treats identically.*
7. **Lambertian assumption breaks on shiny skin/shadowed creases.**
   *Mitigation: optional per-pixel intensity weighting in the solver;
   rig notes recommend cross-polarization; residual error lands in the
   band-passed detail layer where it reads as texture, not shape.*
8. **Pure-Python occlusion rays are slow at production density.**
   Acceptable at test scale; *flagged: embree/open3d backend is a drop-in
   optimization later, and occlusion testing is optional per bake.*
9. **Overlay-onto-bare-mesh (no photo) was underspecified.** v1 overlays
   operate on detail height maps in image space, pre-bake — which covers
   the "punch up this scan" pipeline completely. Direct-on-mesh apply for
   scratch characters is Phase 4b with a named approach (local unwrap),
   not hand-waving.
10. **Dependency licensing.** All core deps are BSD/MIT/Apache-class;
    verified before writing code. No DECA/HRN in the core path.

## v1 CLI surface

```
actor_scan light-calibrate  ball.jpg --circle CX CY R -o lights.json
actor_scan normals   img1..imgN --lights lights.json -o normals.npz
actor_scan detail    normals.npz [--sigma 40] -o height.npy
actor_scan detail    --photo face.jpg [--sigma 8] -o height.npy
actor_scan register  scan.obj corres.json --size W H -o pose.json
actor_scan bake      scan.obj pose.json height.npy --scale 0.15 -o out.stl
actor_scan overlay extract height.npy region.json -o l_cheek.aso.npz
actor_scan overlay apply   height.npy l_cheek.aso.npz landmarks.json -o out.npy
```
