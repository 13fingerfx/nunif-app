# actor_scan

Detail-enhancement pipeline for 3D scans: take a mid-range scan (the
dimensional authority) plus HD photos of the same subject, recover
pore/wrinkle-scale surface detail from the photos, and bake it into the
mesh as real geometry for 3D printing or mould-making. Detail regions can
be saved as shaped, landmark-tagged overlays and refitted onto other
scans later.

Docs:
- [`docs/research.md`](docs/research.md) — stage-1 survey: existing
  tools, techniques, licensing, platform choice, product framing.
- [`docs/build_plan.md`](docs/build_plan.md) — step-by-step build plan
  and the logical-flaw interrogation that shaped it.

## Status

Phases 0–4 (image-space half) implemented and tested:

| Module | Purpose |
|---|---|
| `core/lightcal.py` | chrome-ball light calibration (per session without the rig, once ever with it) |
| `core/photometric.py` | photometric stereo: >=3 fixed-camera shots under known lights -> normals + albedo |
| `core/freqsep.py` | single-photo fallback: frequency-separation pseudo-height |
| `core/integrate.py` | normals -> height (mirror-padded Frankot-Chellappa) + detail band-pass |
| `core/registration.py` | photo <-> scan pose from clicked correspondences (PnP, focal sweep) |
| `core/bake.py` | subdivide -> project -> displace -> STL/OBJ/PLY (never through the UV atlas) |
| `core/overlay.py` | shaped, feathered, landmark-tagged overlays; TPS landmark fitting |
| `core/synthesis.py` | guided detail synthesis: exemplar micro-texture conditioned on observed data (image-analogies family) |
| `core/zones.py` | canonical face-zone vocabulary with loose-name resolution, mirrors, hair-prone groups |
| `core/selection.py` | manual region selection: polygon on a registered photo -> vertex mask (primary path) |
| `core/defects.py` | optional assist: auto-suggested hair/wig-cap regions (user always decides) |
| `core/fill.py` | region re-shaping with profiles: scalp (bald), brow, beard + fullness/taper sliders |
| `core/measure.py` | 13FingerFX head-measurement chart: calipers + tape loops/arcs, chart compare |
| `core/eyes.py` | replacement eye forms: preset diameters, plain sphere or sculpted iris/limbus |
| `core/project.py` | never-erase projects: read-only hashed sources, versioned outputs, step journal |

Not yet built: template-head fill (generic CC0 head as fill target),
overlay-onto-bare-mesh (build_plan 4b), beard-from-reference-photo,
UV color-texture path (5), GUI (6).

Pipeline order on a real scan: `select` (or `mark` assist) -> `fill`
(repair first) -> `register`/`detail` -> `bake` -> `enhance`/overlays,
with `measure` charts before/after to confirm dimensions held.

First real capture: see [`docs/capture_guide.md`](docs/capture_guide.md).

## Install / test

```
pip install -r actor_scan/requirements.txt
python -m unittest discover -s actor_scan/tests
```

## Pipeline (CLI)

```bash
# 1. calibrate lights from chrome-ball shots (skip if using a rig profile)
python -m actor_scan light-calibrate ball0.jpg ball1.jpg ball2.jpg \
    --circle 640 400 220 -o lights.json

# 2. solve normals from the same-lights captures of the subject
python -m actor_scan normals shot0.jpg shot1.jpg shot2.jpg \
    --lights lights.json -o ps.npz

# 3. integrate + band-pass into a detail height map
python -m actor_scan detail ps.npz --sigma 40 -o height.npy
#    (or, single ordinary photo, lower fidelity:)
python -m actor_scan detail --photo face.jpg --sigma 8 -o height.npy

# 4. register the photo camera against the scan (clicked correspondences)
python -m actor_scan register scan.obj corres.json \
    --image-size 6000 4000 -o pose.json

# 5. bake detail into geometry and export for printing
python -m actor_scan bake scan.obj --view pose.json height.npy \
    --scale 0.15 --max-edge 0.4 -o out.stl

# scan repair (before everything else): outline on a registered photo,
# then fill -- bald scalp, brow ridge, or beard dome
python -m actor_scan select scan.ply pose.json lasso.json -o mask.npy
python -m actor_scan fill scan.ply mask.npy --profile beard \
    --fullness 5.0 -o repaired.ply
#   (or `mark` to auto-suggest a hair mask as a starting point)

# measurements: compute the chart, compare against the target sheet
python -m actor_scan measure scan.ply landmarks.json \
    --against actor_chart.json -o measured.json

# replacement eye forms (scanned eyes read as melted; ~24mm is adult)
python -m actor_scan eye-form --preset adult --style sculpted -o eye.stl

# never-erase project: sources copied in read-only, outputs versioned
python -m actor_scan project init job_smith/
python -m actor_scan project add job_smith/ scan.ply

# zone vocabulary
python -m actor_scan zones --resolve "under left eye"   # -> l_under_eye
python -m actor_scan zones --hair beard

# guided synthesis: top up a soft detail map from a real exemplar
python -m actor_scan enhance height.npy --exemplar l_cheek_bank.npy \
    -o height_enhanced.npy

# overlays: cut a shaped region, refit it elsewhere via landmarks
python -m actor_scan overlay-extract height.npy region.json -o l_cheek.aso.npz
python -m actor_scan overlay-apply other_height.npy l_cheek.aso.npz \
    marks.json -o merged.npy
```

`--scale` is the explicit estimation knob: photometric detail has
physically unknowable amplitude, so depth is set in mesh units (mm for
most scanners). The band-pass (`--sigma`) guarantees photo-derived data
only ever adds detail the scanner couldn't capture — the scan's macro
shape is never altered.
