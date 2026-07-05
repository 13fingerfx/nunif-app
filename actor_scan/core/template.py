""" Template-head fill: the generic form supplies the shape prior.

The workflow this implements: a generic average head is invisibly
tethered to the scan (similarity transform from matched landmarks --
rotation, translation, uniform scale; the scan's proportions are never
distorted by the alignment). The user outlines the region to replace;
the equivalent area of the generic supplies target positions; a
screened biharmonic solve drops that shape in while keeping the
boundary glued to the scan, with a feather band controlling how far
template influence reaches toward the rim. `adherence` is the
follow-the-generic slider (0 = plain biharmonic, higher = hug the
template); shaping sliders (fullness/taper) and measurement charts
layer on top. Locked regions are honored exactly as in fill_regions.

The real-scan calibration motivated this: biharmonic alone turns a
full-cranium fill into a conehead; with a skull-shaped prior the same
outline produces a skull.
"""
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.spatial import cKDTree
import trimesh

from .defects import vertex_adjacency
from .fill import (graph_laplacian, _drop_unconstrained, _boundary_rings,
                   shape_fill)


def similarity_transform(src, dst):
    """Umeyama: 4x4 similarity (rotation + translation + uniform scale)
    mapping src points onto dst points, least squares."""
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    if src.shape != dst.shape or len(src) < 3:
        raise ValueError("need >= 3 matched landmark pairs")
    mu_s, mu_d = src.mean(axis=0), dst.mean(axis=0)
    xs, xd = src - mu_s, dst - mu_d
    cov = xd.T @ xs / len(src)
    u, s, vt = np.linalg.svd(cov)
    sign = np.sign(np.linalg.det(u @ vt))
    d = np.diag([1.0, 1.0, sign])
    rotation = u @ d @ vt
    var_s = (xs ** 2).sum() / len(src)
    scale = float(np.trace(np.diag(s) @ d) / var_s)
    translation = mu_d - scale * rotation @ mu_s
    matrix = np.eye(4)
    matrix[:3, :3] = scale * rotation
    matrix[:3, 3] = translation
    return matrix


def align_template(template, template_landmarks, scan_landmarks):
    """Tether the generic to the scan via shared landmark names.
    Returns (aligned copy of template, 4x4 transform, rms error)."""
    names = sorted(set(template_landmarks) & set(scan_landmarks))
    if len(names) < 3:
        raise ValueError("need >= 3 landmark names present on both")
    src = np.array([template_landmarks[n] for n in names], dtype=np.float64)
    dst = np.array([scan_landmarks[n] for n in names], dtype=np.float64)
    matrix = similarity_transform(src, dst)
    aligned = template.copy()
    aligned.apply_transform(matrix)
    homo = np.hstack([src, np.ones((len(src), 1))])
    residual = (homo @ matrix.T)[:, :3] - dst
    rms = float(np.sqrt((residual ** 2).sum(axis=1).mean()))
    return aligned, matrix, rms


def template_targets(points, template, samples=400000, seed=0):
    """Nearest template-surface point for each query point (dense
    surface sampling + KD-tree; fast and accurate to sampling density)."""
    surface, _ = trimesh.sample.sample_surface(
        template, samples, seed=seed)
    tree = cKDTree(surface)
    _, idx = tree.query(np.asarray(points, dtype=np.float64))
    return surface[idx]


def template_fill(mesh, vertex_mask, template, adherence=1.0,
                  feather_rings=4, locked=None, fullness=0.0, taper=1.0,
                  samples=400000):
    """Replace a region with the template's shape, tethered at the rim.

    template must already be aligned into the scan's space (use
    align_template). adherence >= 0: 0 gives plain biharmonic; ~1
    balances smoothness against the template; larger hugs the template.
    feather_rings: rim band where template influence fades to zero so
    the drop-in stays glued to the scan at the boundary.
    Returns (new Trimesh, effective mask actually filled).
    """
    if adherence < 0:
        raise ValueError("adherence must be >= 0")
    mask = np.asarray(vertex_mask, dtype=bool)
    if mask.shape != (len(mesh.vertices),):
        raise ValueError("mask length must equal the vertex count")
    if locked is not None:
        locked = np.asarray(locked, dtype=bool)
        overlap = int((mask & locked).sum())
        if overlap:
            print(f"lock: excluded {overlap} selected vertices inside "
                  "the locked region")
        mask = mask & ~locked
    adj = vertex_adjacency(mesh)
    mask, dropped = _drop_unconstrained(mask, adj)
    if dropped:
        print(f"warning: skipped {dropped} fully-masked component(s)")
    if not mask.any():
        return mesh.copy(), mask

    lap = graph_laplacian(mesh).tocsr()
    op = (lap @ lap).tocsr()
    free = np.flatnonzero(mask)
    fixed = np.flatnonzero(~mask)

    targets = template_targets(mesh.vertices[free], template,
                               samples=samples)
    rings = _boundary_rings(mask, adj)[free]
    if feather_rings > 0:
        t = np.clip(rings / float(feather_rings), 0.0, 1.0)
        influence = t * t * (3.0 - 2.0 * t)
    else:
        influence = np.ones(len(free))

    a_ff = op[free][:, free]
    # Scale adherence relative to the operator so ~1.0 is balanced.
    alpha = adherence * float(a_ff.diagonal().mean()) * influence
    a = (a_ff + sparse.diags(alpha)).tocsc()
    b = (-op[free][:, fixed] @ mesh.vertices[fixed]
         + alpha[:, None] * targets)
    solution = spsolve(a, b)
    vertices = np.array(mesh.vertices, dtype=np.float64)
    vertices[free] = solution
    filled = trimesh.Trimesh(vertices=vertices, faces=mesh.faces,
                             process=False)
    if fullness != 0.0:
        filled = shape_fill(filled, mask, fullness, taper, adj=adj)
    return filled, mask
