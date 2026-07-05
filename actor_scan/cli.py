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
    mesh = trimesh.load(args.mesh, force="mesh")
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
    mesh = trimesh.load(args.mesh, force="mesh")
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


def cmd_overlay_extract(args):
    from .core import overlay
    height = np.load(args.height)
    with open(args.region) as f:
        region = json.load(f)
    ov = overlay.extract_overlay(
        height, region["polygon"], region["landmarks"],
        region["region_tag"], region.get("name", region["region_tag"]),
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
