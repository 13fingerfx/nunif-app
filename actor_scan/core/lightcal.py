""" Chrome-ball light calibration.

Coordinate convention (shared with photometric.py): image space, x right,
y down, z toward the camera. Light vectors point from the surface toward
the light and have z > 0 for any light in front of the subject.

Without the capture rig this runs once per session; with the rig the
resulting vectors are saved once and reused (a rig profile is just a saved
calibration file).
"""
import numpy as np


def sphere_normal_at(px, py, cx, cy, radius):
    """Surface normal of a sphere at image point (px, py).

    (cx, cy, radius) is the sphere's silhouette circle in pixels.
    """
    nx = (px - cx) / radius
    ny = (py - cy) / radius
    d2 = nx * nx + ny * ny
    if d2 >= 1.0:
        raise ValueError("point lies outside the sphere silhouette")
    return np.array([nx, ny, np.sqrt(1.0 - d2)], dtype=np.float64)


def light_from_highlight(px, py, cx, cy, radius):
    """Incident light direction from a specular highlight position.

    The viewer direction is (0, 0, 1) (orthographic approximation, valid
    when the ball is small relative to the camera distance). The highlight
    sits where the mirror reflection of the view ray hits the light:
    L = 2 (n . v) n - v.
    """
    n = sphere_normal_at(px, py, cx, cy, radius)
    v = np.array([0.0, 0.0, 1.0])
    light = 2.0 * float(n @ v) * n - v
    return light / np.linalg.norm(light)


def find_highlight(image, cx, cy, radius, percentile=99.8, margin=0.98):
    """Locate the specular highlight on the ball as an intensity centroid.

    image: 2D float or uint8 array (grayscale). Returns (px, py).
    margin shrinks the mask slightly to avoid rim pixels.
    """
    img = np.asarray(image, dtype=np.float64)
    h, w = img.shape[:2]
    ys, xs = np.mgrid[0:h, 0:w]
    mask = (xs - cx) ** 2 + (ys - cy) ** 2 <= (radius * margin) ** 2
    if not mask.any():
        raise ValueError("sphere circle lies outside the image")
    values = img[mask]
    threshold = np.percentile(values, percentile)
    hot = mask & (img >= threshold)
    weights = img[hot]
    px = float((xs[hot] * weights).sum() / weights.sum())
    py = float((ys[hot] * weights).sum() / weights.sum())
    return px, py


def calibrate_light(image, cx, cy, radius, percentile=99.8):
    """Full calibration for one image: find highlight, return light vector."""
    px, py = find_highlight(image, cx, cy, radius, percentile=percentile)
    return light_from_highlight(px, py, cx, cy, radius)
