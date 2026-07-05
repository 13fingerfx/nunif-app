""" actor_scan command-line interface.

Thin wrappers over actor_scan.core; every step reads/writes plain files
(images, .npy/.npz, .json, meshes) so the pipeline can be driven
manually, scripted, or later fronted by a GUI without change.

    actor_scan light-calibrate ball.jpg --circle 640 400 220 -o lights.json
    actor_scan normals a.jpg b.jpg c.jpg --lights lights.json -o ps.npz
    actor_scan detail ps.npz --sigma 40 -o height.npy
    actor_scan detail --photo face.jpg --sigma 8 -o height.npy
    actor_scan register scan.obj corres.json --image-size 6000 4000 -o pose.json
    actor_scan bake scan.obj --view pose.json height.npy --scale 0.15 -o out.stl
    actor_scan overlay-extract height.npy region.json -o l_cheek.aso.npz
    actor_scan overlay-apply height.npy l_cheek.aso.npz marks.json -o out.npy
"""
import argparse
import json
import sys

import numpy as np


def _load_gray(path):
    from PIL import Image
    return np.asarray(Image.open(path).convert("L"), dtype=np.float64) / 255.0


def _load_rgb(path):
    from PIL import Image
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float64) / 255.0


def cmd_light_calibrate(args):
    from .core import lightcal
    lights = []
    for path in args.images:
        vec = lightcal.calibrate_light(_load_gray(path), *args.circle)
        lights.append(vec.tolist())
        print(f"{path}: light = [{vec[0]:+.4f} {vec[1]:+.4f} {vec[2]:+.4f}]")
    with open(args.output, "w") as f:
        json.dump({"lights": lights, "images": args.images}, f, indent=2)
    print(f"saved {args.output}")


def cmd_normals(args):
    from .core import photometric
    with open(args.lights) as f:
        lights = np.asarray(json.load(f)["lights"], dtype=np.float64)
    images = [_load_gray(p) for p in args.images]
    weights = "intensity" if args.robust else None
    normals, albedo = photometric.solve_normals(images, lights,
                                                weights=weights)
    np.savez_compressed(args.output, normals=normals, albedo=albedo)
    print(f"saved {args.output}  normals {normals.shape}")


def cmd_detail(args):
    from .core import integrate, freqsep
    if args.photo:
        height = freqsep.detail_height_from_photo(
            _load_rgb(args.photo), sigma_low=args.sigma,
            sigma_high=args.denoise)
    else:
        if not args.normals:
            sys.exit("give either a normals .npz or --photo")
        normals = np.load(args.normals)["normals"]
        height = integrate.detail_height_from_normals(
            normals, sigma_low=args.sigma, sigma_high=args.denoise)
    np.save(args.output, height.astype(np.float32))
    print(f"saved {args.output}  range [{height.min():.4f}, "
          f"{height.max():.4f}]")


def cmd_register(args):
    import trimesh
    from .core import registration
    with open(args.correspondences) as f:
        corres = json.load(f)
    pts3 = np.asarray(corres["points3d"], dtype=np.float64)
    pts2 = np.asarray(corres["points2d"], dtype=np.float64)
    # The mesh is loaded purely to sanity-check that the 3D points
    # actually lie on/near the scan.
    mesh = _load_mesh(args.mesh)
    _, dist, _ = mesh.nearest.on_surface(pts3)
    if dist.max() > 0.05 * float(mesh.scale):
        print(f"warning: a 3D point is {dist.max():.2f} units off the "
              "scan surface -- check the correspondence file")
    pose, err = registration.solve_pose(
        pts3, pts2, tuple(args.image_size), focal_px=args.focal)
    pose.save(args.output)
    print(f"saved {args.output}  reprojection error {err:.2f}px "
          f"(focal {pose.camera_matrix[0, 0]:.0f}px)")


def cmd_bake(args):
    import trimesh
    from .core import bake
    from .core.registration import CameraPose
    mesh = _load_mesh(args.mesh)
    views = []
    for pose_path, height_path in args.view:
        views.append((CameraPose.load(pose_path), np.load(height_path)))
    baked, weight = bake.bake(mesh, views, height_scale=args.scale,
                              max_edge=args.max_edge,
                              occlusion=not args.no_occlusion,
                              min_cos=args.min_cos)
    baked.export(args.output)
    covered = float((weight > 0).mean()) * 100.0
    print(f"saved {args.output}  {len(baked.vertices)} vertices, "
          f"{covered:.1f}% received detail")


def cmd_zones(args):
    from .core import zones
    if args.resolve:
        zid = zones.resolve(args.resolve)
        z = zones.ZONES[zid]
        extra = f", mirror {z.mirror}" if z.mirror else ""
        hair = f", hair:{z.hair}" if z.hair else ""
        print(f"{args.resolve!r} -> {zid}  ({z.label}; group {z.group}"
              f"{extra}{hair})")
        return
    listing = zones.ZONES.values()
    if args.group:
        listing = [z for z in listing if z.group == args.group]
    if args.hair:
        listing = [z for z in listing if z.hair == args.hair]
    for z in listing:
        hair = f"  [hair:{z.hair}]" if z.hair else ""
        print(f"{z.id:18s} {z.label}{hair}")


def _load_mesh(path):
    """Load a scan welded: OBJ exports split vertices at every UV/normal
    seam (a 200K-vertex scan arrives as 445K disconnected corners),
    which breaks adjacency-based work -- fill, roughness, selection
    growth, and displacement all need the connected graph. Also prints
    the extent so unit mix-ups (cm vs mm scans) surface immediately."""
    import trimesh
    mesh = trimesh.load(path, force="mesh", process=False)
    before = len(mesh.vertices)
    mesh.merge_vertices(merge_tex=True, merge_norm=True)
    if len(mesh.vertices) != before:
        print(f"welded {before} -> {len(mesh.vertices)} vertices "
              "(UV/normal seams)")
    extent = mesh.bounds[1] - mesh.bounds[0]
    span = float(extent.max())
    if 100.0 <= span <= 1000.0:
        unit = "mm"
    elif 10.0 <= span < 100.0:
        unit = "cm"
    else:
        unit = "unknown"
    print(f"extent {extent.round(1)} (largest span {span:.0f} -> "
          f"units look like {unit}; fullness/measure values are in "
          "mesh units)")
    return mesh


def _vertex_colors(mesh):
    visual = getattr(mesh, "visual", None)
    if visual is None:
        return None
    if getattr(visual, "kind", None) == "vertex":
        colors = np.asarray(visual.vertex_colors)
    else:
        try:
            colors = np.asarray(visual.to_color().vertex_colors)
        except Exception:
            return None
    if colors.ndim != 2 or len(colors) != len(mesh.vertices):
        return None
    return colors[:, :3]


def cmd_mark(args):
    import trimesh
    from .core import defects
    mesh = _load_mesh(args.mesh)
    colors = None if args.no_color else _vertex_colors(mesh)
    if colors is None and not args.no_color:
        print("warning: no vertex colors found -- geometry-only pass; "
              "smooth wig caps may be missed")
    mask, diag = defects.estimate_replace_mask(
        mesh, colors, rough_threshold=args.rough_threshold,
        color_z=args.color_z, min_component=args.min_component,
        grow_rings=args.grow)
    np.save(args.output, mask)
    print(f"saved {args.output}  {mask.sum()} of {len(mask)} vertices "
          f"marked ({diag['final_fraction'] * 100:.1f}%; roughness "
          f"{diag['rough_fraction'] * 100:.1f}%, color "
          f"{diag['color_fraction'] * 100:.1f}%)")


def cmd_fill(args):
    import trimesh
    from .core import fill
    from .core.project import guard_overwrite
    guard_overwrite(args.output, args.mesh, args.mask)
    mesh = _load_mesh(args.mesh)
    mask = np.load(args.mask)
    filled, effective = fill.fill_regions(
        mesh, mask, method=args.method, profile=args.profile,
        fullness=args.fullness, taper=args.taper)
    filled.export(args.output)
    shape = args.profile or "flat"
    print(f"saved {args.output}  re-shaped {int(effective.sum())} vertices "
          f"({args.method}, profile {shape})")


def cmd_select(args):
    import trimesh
    from .core import selection
    from .core.registration import CameraPose
    with open(args.polygon) as f:
        data = json.load(f)
    polygon = data["polygon"] if isinstance(data, dict) else data
    mesh = _load_mesh(args.mesh)
    pose = CameraPose.load(args.pose)
    mask = selection.polygon_to_mask(mesh, pose, polygon,
                                     grow_rings=args.grow)
    np.save(args.output, mask)
    print(f"saved {args.output}  {int(mask.sum())} of {len(mask)} "
          "vertices selected")


def cmd_measure(args):
    import trimesh
    from .core import measure
    mesh = _load_mesh(args.mesh)
    with open(args.landmarks) as f:
        landmarks = json.load(f)
    chart = measure.compute_chart(mesh, landmarks,
                                  only=set(args.only) if args.only else None)
    missing = chart.pop("_missing", [])
    for mid, value in chart.items():
        print(f"{measure.BY_ID[mid].label:36s} {value:8.1f}")
    if missing:
        print(f"(missing landmarks for: {', '.join(missing)})")
    if args.against:
        target = measure.load_chart(args.against)
        diff = measure.compare_charts(chart, target)
        print("\nvs target chart:")
        for mid, d in diff.items():
            print(f"{measure.BY_ID[mid].label:36s} "
                  f"target {d['target']:7.1f}  measured "
                  f"{d['measured']:7.1f}  delta {d['delta']:+6.1f}")
    if args.output:
        with open(args.output, "w") as f:
            json.dump(chart, f, indent=2)
        print(f"\nsaved {args.output}")


def cmd_eye_form(args):
    from .core import eyes
    diameter = (eyes.PRESET_DIAMETERS[args.preset] if args.preset
                else args.diameter)
    mesh = eyes.eye_form(diameter=diameter, style=args.style,
                         iris_diameter=args.iris_diameter,
                         cornea_bulge=args.cornea_bulge,
                         iris_recess=args.iris_recess)
    if args.center:
        mesh = eyes.place_eye(mesh, args.center, args.aim)
    mesh.export(args.output)
    print(f"saved {args.output}  {args.style} eye, diameter "
          f"{diameter:.1f}mm")


def cmd_project(args):
    from .core.project import Project
    if args.action == "init":
        Project.create(args.dir)
        print(f"project created at {args.dir} (sources/, outputs/, "
              "journal.json)")
    elif args.action == "add":
        proj = Project(args.dir)
        for path in args.files:
            dst = proj.add_source(path)
            print(f"added {path} -> {dst} (read-only)")
    else:
        proj = Project(args.dir)
        for step in proj.journal["steps"]:
            print(f"{step['step']}: {step['params']}")


def cmd_enhance(args):
    from .core import synthesis
    guide = np.load(args.height)
    if args.exemplar.endswith(".npz"):
        exemplar = np.load(args.exemplar)["height"]
    else:
        exemplar = np.load(args.exemplar)
    enhanced, high = synthesis.guided_synthesis(
        guide, exemplar, sigma_split=args.sigma, tile=args.tile,
        overlap=args.overlap, tolerance=args.tolerance,
        strength=args.strength, rng=np.random.default_rng(args.seed))
    np.save(args.output, enhanced.astype(np.float32))
    print(f"saved {args.output}  synthesized band std "
          f"{high.std():.4f} (guide std {guide.std():.4f})")


def cmd_overlay_extract(args):
    from .core import overlay
    height = np.load(args.height)
    with open(args.region) as f:
        region = json.load(f)
    from .core import zones
    tag = region["region_tag"]
    try:
        canonical = zones.resolve(tag)
        if canonical != tag:
            print(f"zone {tag!r} -> canonical {canonical!r}")
        tag = canonical
    except KeyError as e:
        print(f"note: {e.args[0]} -- keeping custom tag {tag!r}")
    ov = overlay.extract_overlay(
        height, region["polygon"], region["landmarks"],
        tag, region.get("name", tag),
        feather_px=args.feather, px_per_mm=region.get("px_per_mm"))
    ov.save(args.output)
    print(f"saved {args.output}  tag={ov.region_tag} "
          f"landmarks={sorted(ov.landmarks)}")


def cmd_overlay_apply(args):
    from .core import overlay
    height = np.load(args.height)
    ov = overlay.Overlay.load(args.overlay)
    with open(args.landmarks) as f:
        marks = json.load(f)
    out = overlay.apply_overlay(height, ov, marks, strength=args.strength,
                                mode=args.mode)
    np.save(args.output, out.astype(np.float32))
    print(f"saved {args.output}")


def build_parser():
    parser = argparse.ArgumentParser(
        prog="actor_scan",
        description="Detail-enhancement pipeline for 3D scans "
                    "(photometric stereo, registration, geometry bake, "
                    "shaped overlays).")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("light-calibrate",
                       help="chrome-ball light calibration")
    p.add_argument("images", nargs="+")
    p.add_argument("--circle", nargs=3, type=float, required=True,
                   metavar=("CX", "CY", "R"),
                   help="ball silhouette circle in pixels")
    p.add_argument("-o", "--output", required=True)
    p.set_defaults(func=cmd_light_calibrate)

    p = sub.add_parser("normals", help="photometric stereo solve")
    p.add_argument("images", nargs="+")
    p.add_argument("--lights", required=True, help="lights .json")
    p.add_argument("--robust", action="store_true",
                   help="intensity-weighted solve (softens shadows)")
    p.add_argument("-o", "--output", required=True, help="output .npz")
    p.set_defaults(func=cmd_normals)

    p = sub.add_parser("detail", help="band-passed detail height map")
    p.add_argument("normals", nargs="?", help="normals .npz")
    p.add_argument("--photo", help="single-photo fallback instead")
    p.add_argument("--sigma", type=float, default=40.0,
                   help="band limit in px; low frequencies belong to "
                        "the scan (default 40; use ~8 for --photo)")
    p.add_argument("--denoise", type=float, default=None,
                   help="optional pre-smooth sigma in px")
    p.add_argument("-o", "--output", required=True, help="output .npy")
    p.set_defaults(func=cmd_detail)

    p = sub.add_parser("register", help="solve camera pose (PnP)")
    p.add_argument("mesh")
    p.add_argument("correspondences",
                   help='json: {"points3d": [[x,y,z],...], '
                        '"points2d": [[u,v],...]}')
    p.add_argument("--image-size", nargs=2, type=int, required=True,
                   metavar=("W", "H"))
    p.add_argument("--focal", type=float, default=None,
                   help="focal length in px (omit to sweep)")
    p.add_argument("-o", "--output", required=True, help="pose .json")
    p.set_defaults(func=cmd_register)

    p = sub.add_parser("bake", help="bake detail into geometry")
    p.add_argument("mesh")
    p.add_argument("--view", nargs=2, action="append", required=True,
                   metavar=("POSE", "HEIGHT"),
                   help="pose .json + height .npy (repeatable)")
    p.add_argument("--scale", type=float, required=True,
                   help="height scale in mesh units (mm for most scans)")
    p.add_argument("--max-edge", type=float, default=None,
                   help="subdivide until edges are below this (mesh units)")
    p.add_argument("--min-cos", type=float, default=0.2)
    p.add_argument("--no-occlusion", action="store_true")
    p.add_argument("-o", "--output", required=True,
                   help="output mesh (.stl/.obj/.ply)")
    p.set_defaults(func=cmd_bake)

    p = sub.add_parser("zones", help="list/resolve face zone names")
    p.add_argument("--resolve", metavar="NAME",
                   help='e.g. "under left eye" -> l_under_eye')
    p.add_argument("--group", help="filter by group (face, eye, ...)")
    p.add_argument("--hair", choices=("scalp", "beard"),
                   help="filter to hair-prone zones")
    p.set_defaults(func=cmd_zones)

    p = sub.add_parser("mark",
                       help="OPTIONAL assist: auto-suggest hair/wig-cap "
                            "regions (manual `select` is the primary "
                            "path; the user always decides)")
    p.add_argument("mesh")
    p.add_argument("--rough-threshold", type=float, default=0.15)
    p.add_argument("--color-z", type=float, default=6.0)
    p.add_argument("--min-component", type=int, default=50)
    p.add_argument("--grow", type=int, default=2,
                   help="grow the mask this many rings")
    p.add_argument("--no-color", action="store_true",
                   help="geometry-only pass")
    p.add_argument("-o", "--output", required=True,
                   help="per-vertex boolean mask .npy")
    p.set_defaults(func=cmd_mark)

    p = sub.add_parser("fill",
                       help="re-shape selected regions (bald/beard/brow)")
    p.add_argument("mesh")
    p.add_argument("mask", help="per-vertex mask .npy from `select`/`mark`")
    p.add_argument("--method", choices=("biharmonic", "laplacian"),
                   default="biharmonic")
    p.add_argument("--profile", choices=("scalp", "brow", "beard"),
                   default=None,
                   help="shaping preset (scalp: skull continuation; "
                        "brow: gentle ridge; beard: fuller dome)")
    p.add_argument("--fullness", type=float, default=None,
                   help="outward offset at region center, mesh units "
                        "(overrides the profile's value; slider-friendly)")
    p.add_argument("--taper", type=float, default=None,
                   help="dome shape: higher = flatter rim, rounder center")
    p.add_argument("-o", "--output", required=True,
                   help="output mesh (.obj/.ply/.stl) -- inputs are "
                        "never overwritten")
    p.set_defaults(func=cmd_fill)

    p = sub.add_parser("select",
                       help="manual region selection: polygon drawn on a "
                            "registered photo -> vertex mask")
    p.add_argument("mesh")
    p.add_argument("pose", help="pose .json from `register`")
    p.add_argument("polygon",
                   help='json: [[x,y],...] or {"polygon": [[x,y],...]} '
                        "in photo pixels")
    p.add_argument("--grow", type=int, default=0,
                   help="widen selection this many rings")
    p.add_argument("-o", "--output", required=True,
                   help="per-vertex mask .npy")
    p.set_defaults(func=cmd_select)

    p = sub.add_parser("measure",
                       help="compute the head-measurement chart "
                            "(13FingerFX set) from named landmarks")
    p.add_argument("mesh")
    p.add_argument("landmarks",
                   help='json: {"pronasale": [x,y,z], "crown": ...}')
    p.add_argument("--only", nargs="+", default=None,
                   help="measurement ids to compute")
    p.add_argument("--against", default=None,
                   help="target chart .json to compare with")
    p.add_argument("-o", "--output", default=None,
                   help="save the measured chart .json")
    p.set_defaults(func=cmd_measure)

    p = sub.add_parser("eye-form",
                       help="generate a replacement eye form")
    p.add_argument("--diameter", type=float, default=24.0,
                   help="eyeball diameter in mm (adult ~24)")
    from .core.eyes import PRESET_DIAMETERS
    p.add_argument("--preset", choices=sorted(PRESET_DIAMETERS),
                   default=None,
                   help="named diameter preset (overrides --diameter)")
    p.add_argument("--style", choices=("sphere", "sculpted"),
                   default="sculpted",
                   help="sphere: plain ball; sculpted: corneal plateau "
                        "+ dished iris (reads better in print)")
    p.add_argument("--iris-diameter", type=float, default=11.8)
    p.add_argument("--cornea-bulge", type=float, default=1.1)
    p.add_argument("--iris-recess", type=float, default=0.45)
    p.add_argument("--center", nargs=3, type=float, default=None,
                   metavar=("X", "Y", "Z"))
    p.add_argument("--aim", nargs=3, type=float, default=(0.0, 0.0, 1.0),
                   metavar=("X", "Y", "Z"), help="gaze direction")
    p.add_argument("-o", "--output", required=True,
                   help="output mesh (.stl/.obj/.ply)")
    p.set_defaults(func=cmd_eye_form)

    p = sub.add_parser("project",
                       help="non-destructive project: originals are "
                            "copied in read-only and never touched")
    p.add_argument("action", choices=("init", "add", "log"))
    p.add_argument("dir")
    p.add_argument("files", nargs="*", help="files for `add`")
    p.set_defaults(func=cmd_project)

    p = sub.add_parser("enhance",
                       help="guided synthesis: top up a soft detail map "
                            "with exemplar-grade micro texture")
    p.add_argument("height", help="observed detail height .npy (guide)")
    p.add_argument("--exemplar", required=True,
                   help="high-res exemplar height .npy or overlay .npz")
    p.add_argument("--sigma", type=float, default=8.0,
                   help="split between observed and synthesized bands (px)")
    p.add_argument("--tile", type=int, default=32)
    p.add_argument("--overlap", type=int, default=12)
    p.add_argument("--tolerance", type=float, default=0.03)
    p.add_argument("--strength", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("-o", "--output", required=True, help="output .npy")
    p.set_defaults(func=cmd_enhance)

    p = sub.add_parser("overlay-extract", help="cut a shaped overlay")
    p.add_argument("height", help="detail height .npy")
    p.add_argument("region",
                   help='json: {"polygon": [[x,y],...], "landmarks": '
                        '{"name": [x,y],...}, "region_tag": "l_cheek"}')
    p.add_argument("--feather", type=float, default=12.0)
    p.add_argument("-o", "--output", required=True, help="output .npz")
    p.set_defaults(func=cmd_overlay_extract)

    p = sub.add_parser("overlay-apply", help="fit an overlay via landmarks")
    p.add_argument("height", help="target detail height .npy")
    p.add_argument("overlay", help="overlay .npz")
    p.add_argument("landmarks",
                   help='json: {"name": [x,y], ...} in target pixels')
    p.add_argument("--strength", type=float, default=1.0)
    p.add_argument("--mode", choices=("blend", "add"), default="blend")
    p.add_argument("-o", "--output", required=True, help="output .npy")
    p.set_defaults(func=cmd_overlay_apply)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
