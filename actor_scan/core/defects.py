""" Automated detection of scan regions that need replacement.

Raw head scans routinely contain non-skin data: hair (spiky, noisy
geometry), wig caps and tied-up hair (smooth but wrong-colored), beards
(often reading as noise or holes). This module scores every vertex with
two classical signals and no ML:

- roughness: disagreement between a vertex normal and the locally
  smoothed normal field. Hair and reconstruction noise score high;
  skin, however wrinkled at scan resolution, scores low.
- color: robust distance from a skin-color model fitted to the scan
  itself (median chromaticity of its smooth regions), so it adapts to
  any complexion and any lighting -- catching wig caps and hair that
  are geometrically clean but chromatically alien.

The result is a per-vertex boolean mask, cleaned of speckle and grown a
few rings, ready for fill.fill_regions(). Runs on the raw scan before
any detail/texture work.
"""
import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components


def vertex_adjacency(mesh):
    e = mesh.edges_unique
    n = len(mesh.vertices)
    data = np.ones(len(e) * 2)
    rows = np.concatenate([e[:, 0], e[:, 1]])
    cols = np.concatenate([e[:, 1], e[:, 0]])
    return sparse.coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()


def _row_normalized(adj):
    deg = np.asarray(adj.sum(axis=1)).reshape(-1)
    inv = sparse.diags(1.0 / np.maximum(deg, 1.0))
    return inv @ adj


def roughness(mesh, adj=None):
    """Per-vertex spikiness: distance from the neighbors' centroid,
    relative to the local mean edge length.

    Scale-free. Smooth surfaces (including gentle curvature) score a few
    hundredths; scan noise and hair score tenths to ~1. Deliberately not
    normal-based: vertex normals of random spikes partially self-cancel
    toward the smooth direction and hide the noise.
    """
    if adj is None:
        adj = vertex_adjacency(mesh)
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    centroid = _row_normalized(adj) @ verts
    displacement = np.linalg.norm(verts - centroid, axis=1)
    e = mesh.edges_unique
    lengths = np.linalg.norm(verts[e[:, 0]] - verts[e[:, 1]], axis=1)
    length_sum = np.zeros(len(verts))
    np.add.at(length_sum, e[:, 0], lengths)
    np.add.at(length_sum, e[:, 1], lengths)
    degree = np.asarray(adj.sum(axis=1)).reshape(-1)
    mean_edge = length_sum / np.maximum(degree, 1.0)
    return displacement / np.maximum(mean_edge, 1e-12)


def color_features(colors):
    """(r-chromaticity, g-chromaticity, value) -- separates hue from
    lighting so shadowed skin isn't mistaken for hair."""
    c = np.asarray(colors, dtype=np.float64)
    if c.max() > 1.5:
        c = c / 255.0
    c = c[:, :3]
    total = c.sum(axis=1, keepdims=True) + 1e-9
    chroma = c[:, :2] / total
    value = c.max(axis=1, keepdims=True)
    return np.hstack([chroma, value])


def color_outlier_score(colors, reference_mask):
    """Robust z-distance of every vertex's color from the skin model
    fitted (median/MAD) over reference_mask vertices."""
    feats = color_features(colors)
    ref = feats[reference_mask]
    center = np.median(ref, axis=0)
    mad = np.median(np.abs(ref - center), axis=0) * 1.4826 + 1e-6
    z = (feats - center) / mad
    return np.sqrt((z * z).sum(axis=1))


def _keep_large_components(mask, adj, min_vertices):
    idx = np.flatnonzero(mask)
    if len(idx) == 0:
        return mask
    sub = adj[idx][:, idx]
    n_comp, labels = connected_components(sub, directed=False)
    keep = np.zeros_like(mask)
    for comp in range(n_comp):
        members = idx[labels == comp]
        if len(members) >= min_vertices:
            keep[members] = True
    return keep


def _grow(mask, adj, rings):
    out = mask.copy()
    for _ in range(rings):
        out = out | (np.asarray(adj @ out.astype(np.float64)).reshape(-1) > 0)
    return out


def estimate_replace_mask(mesh, colors=None, rough_threshold=0.15,
                          color_z=6.0, min_component=50, grow_rings=2):
    """Boolean per-vertex mask of regions to replace (hair/cap/junk).

    colors: optional (N, 3or4) vertex colors; without them only the
    roughness signal runs (wig caps may then be missed -- warn upstream).
    Returns (mask, diagnostics dict).
    """
    adj = vertex_adjacency(mesh)
    rough = roughness(mesh, adj=adj)
    rough_mask = rough > rough_threshold

    color_mask = np.zeros(len(mesh.vertices), dtype=bool)
    if colors is not None:
        # Fit the skin model on smooth vertices only; if nearly the whole
        # scan is rough, fall back to everything.
        reference = ~rough_mask
        if reference.sum() < 0.1 * len(reference):
            reference = np.ones_like(reference)
        score = color_outlier_score(colors, reference)
        color_mask = score > color_z

    mask = rough_mask | color_mask
    mask = _keep_large_components(mask, adj, min_component)
    mask = _grow(mask, adj, grow_rings)
    return mask, {
        "roughness": rough,
        "rough_fraction": float(rough_mask.mean()),
        "color_fraction": float(color_mask.mean()),
        "final_fraction": float(mask.mean()),
    }
