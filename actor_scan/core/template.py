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
                   shape_fill, delete_vertex_faces)


def similarity_transform(src, dst, with_scale=True):
    """Umeyama: 4x4 transform (rotation + translation, and uniform
    scale unless with_scale=False) mapping src onto dst, least squares."""
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
    if with_scale:
        var_s = (xs ** 2).sum() / len(src)
        scale = float(np.trace(np.diag(s) @ d) / var_s)
    else:
        scale = 1.0
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


def anisotropic_align(template, template_landmarks, scan_landmarks):
    """Tether allowing per-axis proportions: the generic is only a
    shape/ratio indicator, so beyond rotation+translation+scale it may
    be stretched independently in width/height/depth to fit the scan's
    landmark frame. (Point-by-point edge matching happens later, in
    template_fill's edge_match stage -- this is just the best global
    fit to start from.)
    Returns (aligned copy, 4x4 transform, rms, per-axis scales)."""
    names = sorted(set(template_landmarks) & set(scan_landmarks))
    if len(names) < 4:
        raise ValueError("need >= 4 shared landmarks for per-axis scales")
    src = np.array([template_landmarks[n] for n in names], dtype=np.float64)
    dst = np.array([scan_landmarks[n] for n in names], dtype=np.float64)
    base = similarity_transform(src, dst)
    src_h = (np.hstack([src, np.ones((len(src), 1))]) @ base.T)[:, :3]
    correction = np.eye(4)
    scales = np.ones(3)
    for axis in range(3):
        a = np.column_stack([src_h[:, axis], np.ones(len(src))])
        (s, t), *_ = np.linalg.lstsq(a, dst[:, axis], rcond=None)
        # Landmarks that barely span this axis can't estimate a scale.
        span = src_h[:, axis].max() - src_h[:, axis].min()
        if span < 1e-6 or not 0.5 < s < 2.0:
            s, t = 1.0, 0.0
        correction[axis, axis] = s
        correction[axis, 3] = t
        scales[axis] = s
    matrix = correction @ base
    aligned = template.copy()
    aligned.apply_transform(matrix)
    homo = np.hstack([src, np.ones((len(src), 1))])
    residual = (homo @ matrix.T)[:, :3] - dst
    rms = float(np.sqrt((residual ** 2).sum(axis=1).mean()))
    return aligned, matrix, rms, scales


def refine_alignment(template, trusted_points, iterations=12,
                     samples=20000, trim_percentile=60.0,
                     allow_scale=False, seed=0):
    """Tighten the tether: trimmed ICP of the template against TRUSTED
    scan geometry (the regions the user is keeping -- face, ears,
    neck), so alignment is never influenced by the hair/junk being
    replaced. Landmarks give the coarse tether; this removes residual
    pose error that shows up as steps where the template meets locked
    geometry.

    RIGID by default (rotation + translation only): scale is the
    landmarks'/measurement chart's job, and letting partial-overlap
    ICP re-estimate scale invites collapse onto the trusted patch
    (observed on real data: a x0.6 shrink). allow_scale=True only for
    well-overlapping, clean geometry.
    Returns (aligned copy, 4x4 transform, final rms)."""
    trusted = np.asarray(trusted_points, dtype=np.float64)
    if len(trusted) < 100:
        raise ValueError("need at least 100 trusted scan points")
    tree = cKDTree(trusted)
    src0, _ = trimesh.sample.sample_surface(template, samples, seed=seed)
    src0 = np.asarray(src0)
    matrix = np.eye(4)
    src = src0.copy()
    rms = np.inf
    for _ in range(iterations):
        dist, idx = tree.query(src)
        # Trim distant pairs so template regions with no trusted
        # counterpart (e.g. its scalp vs the scan's replaced scalp)
        # don't drag the fit toward the trusted patch's rim.
        keep = dist <= np.percentile(dist, trim_percentile)
        step = similarity_transform(src[keep], trusted[idx[keep]],
                                    with_scale=allow_scale)
        matrix = step @ matrix
        src = (np.hstack([src0, np.ones((len(src0), 1))])
               @ matrix.T)[:, :3]
        rms = float(np.sqrt((dist[keep] ** 2).mean()))
    aligned = template.copy()
    aligned.apply_transform(matrix)
    return aligned, matrix, rms


def _edge_matched_targets(mesh, mask, free, targets, adj):
    """Correct template targets with a per-vertex rim offset field.

    offset = scan - template at every rim vertex (ring 0), harmonically
    interpolated over the region interior via the mask-subgraph
    Laplacian. Added to the raw targets, this makes the corrected
    target coincide with the scan at the boundary regardless of how
    imperfect the global tether is."""
    rings = _boundary_rings(mask, adj)
    rim = mask & (rings == 0)
    if not rim.any():
        return targets
    # Offsets known at rim vertices (their position in the free array).
    free_pos = {v: i for i, v in enumerate(free)}
    rim_idx = np.flatnonzero(rim)
    rim_free = np.array([free_pos[i] for i in rim_idx])
    offsets_rim = np.asarray(mesh.vertices)[rim_idx] - targets[rim_free]

    interior_idx = np.flatnonzero(mask & (rings > 0))
    if len(interior_idx) == 0:
        targets = targets.copy()
        targets[rim_free] += offsets_rim
        return targets
    # Harmonic interpolation on the induced subgraph of the region.
    nodes = np.concatenate([rim_idx, interior_idx])
    local = {v: i for i, v in enumerate(nodes)}
    sub = adj[nodes][:, nodes]
    deg = np.asarray(sub.sum(axis=1)).reshape(-1)
    lap = sparse.diags(deg) - sub
    n_rim = len(rim_idx)
    a = lap[n_rim:, n_rim:].tocsc()
    b = -lap[n_rim:, :n_rim] @ offsets_rim
    offsets_interior = spsolve(a, b)
    targets = targets.copy()
    targets[rim_free] += offsets_rim
    interior_free = np.array([free_pos[i] for i in interior_idx])
    targets[interior_free] += np.atleast_2d(offsets_interior)
    return targets


def template_targets(points, template, samples=400000, k=6, seed=0):
    """Template-surface target for each query point: dense surface
    sampling + KD-tree, averaging the k nearest samples -- a single
    nearest sample quantizes to the sampling grid and prints through
    the fill as micro-noise rougher than skin."""
    surface, _ = trimesh.sample.sample_surface(
        template, samples, seed=seed)
    tree = cKDTree(surface)
    _, idx = tree.query(np.asarray(points, dtype=np.float64), k=k)
    if k == 1:
        return surface[idx]
    return surface[idx].mean(axis=1)


def template_fill(mesh, vertex_mask, template, adherence=1.0,
                  feather_rings=4, locked=None, fullness=0.0, taper=1.0,
                  samples=400000, delete_stranded=False, edge_match=True):
    """Replace a region with the template's shape, tethered at the rim.

    template must already be aligned into the scan's space (use
    align_template / anisotropic_align). adherence >= 0: 0 gives plain
    biharmonic; ~1 balances smoothness against the template; larger
    hugs the template. feather_rings: rim band where template
    influence ramps in from the boundary.

    edge_match (default True): point-by-point boundary conformance.
    The scan-to-template offset is measured at every rim vertex (where
    the scan is sacrosanct) and interpolated harmonically across the
    region; targets become template + offset field. The drop-in
    therefore meets the scan EXACTLY at the rim -- no global transform
    of the generic can guarantee that, per-vertex warping can -- while
    the interior still carries the generic's shape, corrected toward
    the subject's real proportions near every edge.
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
    mask, stranded = _drop_unconstrained(mask, adj)
    if stranded.any():
        verb = ("deleting" if delete_stranded else
                "leaving untouched (delete_stranded=True removes)")
        print(f"{int(stranded.sum())} stranded vertices: {verb}")
    if not mask.any():
        out = mesh.copy()
        if delete_stranded and stranded.any():
            out = delete_vertex_faces(out, stranded)
        return out, mask

    lap = graph_laplacian(mesh).tocsr()
    op = (lap @ lap).tocsr()
    free = np.flatnonzero(mask)
    fixed = np.flatnonzero(~mask)

    targets = template_targets(mesh.vertices[free], template,
                               samples=samples)
    if edge_match:
        targets = _edge_matched_targets(mesh, mask, free, targets, adj)
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
    if delete_stranded and stranded.any():
        filled = delete_vertex_faces(filled, stranded)
    return filled, mask
