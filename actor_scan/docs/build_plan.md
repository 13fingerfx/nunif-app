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
- **4c (deferred)**: patch-based resynthesis (PatchMatch/Efros-Leung
  family) so micro-texture fills a resized boundary instead of
  stretching. v1 uses plain TPS warp; the API keeps the micro layer
  separate so 4c slots in without breaking overlays already saved.

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
