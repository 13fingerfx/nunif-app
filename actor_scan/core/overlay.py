""" Shaped, region-tagged detail overlays.

An overlay is a polygon-bounded, feather-edged patch of a detail height
map plus named landmark points ("eye_corner_outer", ...). Applying one to
a new target is landmark-driven: the user (or a face-landmark model)
places the same named points on the target, and a thin-plate-spline warp
morphs the patch to fit -- the Peachy/AR-filter draggable-dots model.

Overlays live in image/detail-map space, upstream of bake, so a fitted
overlay is printed via exactly the same bake path as directly captured
detail. Micro-texture resynthesis for resized patches (build_plan 4c)
will layer on top of this format without changing it.
"""
import json
import numpy as np
import cv2
from scipy.interpolate import RBFInterpolator

FORMAT_VERSION = 1


class Overlay:
    def __init__(self, height, alpha, landmarks, region_tag, name,
                 px_per_mm=None):
        self.height = np.asarray(height, dtype=np.float32)
        self.alpha = np.asarray(alpha, dtype=np.float32)
        if self.alpha.shape != self.height.shape:
            raise ValueError("alpha and height shapes differ")
        self.landmarks = {k: (float(x), float(y))
                          for k, (x, y) in landmarks.items()}
        self.region_tag = region_tag
        self.name = name
        self.px_per_mm = px_per_mm

    def save(self, path):
        meta = {
            "format_version": FORMAT_VERSION,
            "name": self.name,
            "region_tag": self.region_tag,
            "px_per_mm": self.px_per_mm,
            "landmarks": self.landmarks,
        }
        np.savez_compressed(path, height=self.height, alpha=self.alpha,
                            meta=json.dumps(meta))

    @classmethod
    def load(cls, path):
        data = np.load(path, allow_pickle=False)
        meta = json.loads(str(data["meta"]))
        if meta["format_version"] > FORMAT_VERSION:
            raise ValueError("overlay was saved by a newer format version")
        return cls(data["height"], data["alpha"], meta["landmarks"],
                   meta["region_tag"], meta["name"], meta["px_per_mm"])


def feathered_mask(shape, polygon, feather_px):
    """1.0 inside the polygon fading to 0.0 across feather_px."""
    mask = np.zeros(shape, dtype=np.uint8)
    cv2.fillPoly(mask, [np.asarray(polygon, dtype=np.int32)], 255)
    if feather_px > 0:
        inside = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
        return np.clip(inside / float(feather_px), 0.0, 1.0)
    return mask.astype(np.float32) / 255.0


def extract_overlay(height_map, polygon, landmarks, region_tag, name,
                    feather_px=12, px_per_mm=None, pad=8):
    """Cut a shaped overlay out of a detail height map.

    polygon: (N, 2) boundary in height-map pixels. landmarks: dict of
    named (x, y) points inside/near the region, stored relative to the
    cropped patch so the overlay is self-contained.
    """
    hm = np.asarray(height_map, dtype=np.float32)
    poly = np.asarray(polygon, dtype=np.float64)
    x0 = max(int(np.floor(poly[:, 0].min())) - pad, 0)
    y0 = max(int(np.floor(poly[:, 1].min())) - pad, 0)
    x1 = min(int(np.ceil(poly[:, 0].max())) + pad, hm.shape[1])
    y1 = min(int(np.ceil(poly[:, 1].max())) + pad, hm.shape[0])
    crop = hm[y0:y1, x0:x1]
    local_poly = poly - [x0, y0]
    alpha = feathered_mask(crop.shape, local_poly, feather_px)
    local_marks = {k: (x - x0, y - y0) for k, (x, y) in landmarks.items()}
    return Overlay(crop * (alpha > 0), alpha, local_marks, region_tag, name,
                   px_per_mm=px_per_mm)


def _tps_maps(src_landmarks, dst_landmarks, out_shape):
    """Backward maps (cv2.remap convention) sending dst grid -> src coords
    via thin-plate-spline interpolation of landmark correspondences."""
    names = sorted(set(src_landmarks) & set(dst_landmarks))
    if len(names) < 3:
        raise ValueError("need at least 3 shared landmarks for a TPS warp")
    dst = np.array([dst_landmarks[n] for n in names], dtype=np.float64)
    src = np.array([src_landmarks[n] for n in names], dtype=np.float64)
    interp = RBFInterpolator(dst, src, kernel="thin_plate_spline")
    h, w = out_shape
    ys, xs = np.mgrid[0:h, 0:w]
    grid = np.column_stack([xs.ravel(), ys.ravel()]).astype(np.float64)
    mapped = interp(grid)
    mapx = mapped[:, 0].reshape(h, w).astype(np.float32)
    mapy = mapped[:, 1].reshape(h, w).astype(np.float32)
    return mapx, mapy


def apply_overlay(target_height, overlay, target_landmarks, strength=1.0,
                  mode="blend"):
    """Warp an overlay onto a target detail height map.

    target_landmarks: dict giving, in target pixels, the positions of the
    overlay's named landmarks (>= 3 shared names). strength scales the
    overlay's amplitude. mode: "blend" cross-fades target detail toward
    the overlay inside the patch; "add" sums on top of existing detail.
    Returns a new height map.
    """
    target = np.asarray(target_height, dtype=np.float32).copy()
    mapx, mapy = _tps_maps(overlay.landmarks, target_landmarks, target.shape)
    warped_h = cv2.remap(overlay.height, mapx, mapy, cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    warped_a = cv2.remap(overlay.alpha, mapx, mapy, cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    warped_a = np.clip(warped_a, 0.0, 1.0)
    if mode == "blend":
        return target * (1.0 - warped_a) + warped_h * strength * warped_a
    if mode == "add":
        return target + warped_h * strength * warped_a
    raise ValueError("mode must be 'blend' or 'add'")
