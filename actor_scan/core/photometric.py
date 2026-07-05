""" Photometric stereo: fixed camera, K >= 3 known light directions.

Solves I_k = albedo * max(n . L_k, 0) per pixel in the least-squares
sense. Coordinate convention matches lightcal.py: x right, y down,
z toward camera; recovered normals have nz > 0.

The default solve is unweighted (one pseudo-inverse shared by all
pixels, fast at any resolution). weights="intensity" performs a
per-pixel weighted solve that down-weights dark samples, softening
shadow and specular violations of the Lambertian assumption at the
cost of a chunked per-pixel solve.
"""
import numpy as np


def solve_normals(images, lights, weights=None, chunk=262144):
    """
    images: sequence of K grayscale float arrays, identical (H, W).
    lights: (K, 3) unit light directions (from surface toward light).
    Returns (normals (H, W, 3) float32, albedo (H, W) float32).
    Pixels with no usable signal get normal (0, 0, 1) and albedo 0.
    """
    stack = np.stack([np.asarray(im, dtype=np.float64) for im in images])
    k, h, w = stack.shape
    lights = np.asarray(lights, dtype=np.float64)
    if lights.shape != (k, 3):
        raise ValueError("lights must be (K, 3) matching the image count")
    if k < 3:
        raise ValueError("photometric stereo needs at least 3 lights")
    intensity = stack.reshape(k, -1)

    if weights is None:
        g = np.linalg.pinv(lights) @ intensity  # (3, N)
    else:
        if weights != "intensity":
            raise ValueError("weights must be None or 'intensity'")
        g = np.empty((3, intensity.shape[1]), dtype=np.float64)
        for start in range(0, intensity.shape[1], chunk):
            block = intensity[:, start:start + chunk]  # (K, B)
            wgt = block / (block.max(axis=0, keepdims=True) + 1e-12)
            # Solve (L^T W L) g = L^T W i per pixel.
            lw = lights[None, :, :] * wgt.T[:, :, None]      # (B, K, 3)
            ata = np.einsum("bki,kj->bij", lw, lights) + np.eye(3) * 1e-9
            atb = np.einsum("bki,kb->bi", lw, block)
            g[:, start:start + chunk] = np.linalg.solve(
                ata, atb[..., None])[..., 0].T

    albedo = np.linalg.norm(g, axis=0)
    valid = albedo > 1e-9
    normals = np.zeros_like(g)
    normals[:, valid] = g[:, valid] / albedo[valid]
    normals[2, ~valid] = 1.0
    # Flip any backward-facing solutions; the surface faces the camera.
    flip = normals[2] < 0
    normals[:, flip] *= -1.0
    normals = normals.T.reshape(h, w, 3).astype(np.float32)
    return normals, albedo.reshape(h, w).astype(np.float32)


def render_lambertian(height, lights, albedo=1.0):
    """Synthetic Lambertian renders of a height field; used by tests and
    for sanity-previewing a solve (re-render and compare to the photos)."""
    normals = normals_from_height(height)
    lights = np.asarray(lights, dtype=np.float64)
    shading = np.einsum("hwc,kc->khw", normals, lights)
    return np.clip(shading, 0.0, None) * albedo


def normals_from_height(height):
    """Unit normals of a height field z = h(x, y), z toward camera."""
    h = np.asarray(height, dtype=np.float64)
    gy, gx = np.gradient(h)
    n = np.dstack([-gx, -gy, np.ones_like(h)])
    n /= np.linalg.norm(n, axis=2, keepdims=True)
    return n
