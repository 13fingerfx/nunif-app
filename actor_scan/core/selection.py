""" Manual region selection: the user outlines, the software obeys.

Primary selection path (policy: automated detection like defects.py is
an optional assist only, never the decider): the user draws a closed
polygon on a photo that has been registered against the scan, and the
enclosed, camera-visible vertices become the replacement mask. The
same outline-on-surface interaction the overlay/texture-transfer
tools use, expressed in CLI terms until the GUI exists.
"""
import numpy as np
import cv2
from scipy.spatial import cKDTree
from scipy.sparse.csgraph import connected_components

from .bake import vertex_weights
from .defects import vertex_adjacency


def polygon_to_mask(mesh, pose, polygon, min_cos=0.05, occlusion=True,
                    grow_rings=0):
    """Per-vertex mask of visible vertices projecting inside a polygon.

    polygon: (N, 2) pixel coordinates in the registered photo.
    grow_rings: optionally widen the selection along the mesh graph.
    """
    polygon = np.asarray(polygon, dtype=np.float64)
    if polygon.ndim != 2 or len(polygon) < 3:
        raise ValueError("polygon needs at least 3 (x, y) points")
    weight, proj = vertex_weights(mesh, pose, occlusion=occlusion,
                                  min_cos=min_cos)
    w, h = pose.image_size
    raster = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(raster, [np.round(polygon).astype(np.int32)], 1)
    px = np.clip(np.round(proj[:, 0]).astype(np.int64), 0, w - 1)
    py = np.clip(np.round(proj[:, 1]).astype(np.int64), 0, h - 1)
    inside = raster[py, px].astype(bool)
    mask = inside & (weight > 0)
    if grow_rings > 0:
        adj = vertex_adjacency(mesh)
        for _ in range(grow_rings):
            mask = mask | (np.asarray(
                adj @ mask.astype(np.float64)).reshape(-1) > 0)
    return mask


def mirror_mask(mesh, mask, axis=0, radius=None):
    """Extend a selection to its mirror image across the given axis --
    hair, beards and brows are roughly symmetric, and an outline drawn
    from one side cannot see the far side. radius defaults to twice
    the median edge length."""
    v = np.asarray(mesh.vertices)
    if radius is None:
        radius = 2.0 * float(np.median(mesh.edges_unique_length))
    mirrored = v[mask].copy()
    mirrored[:, axis] *= -1.0
    out = np.asarray(mask, dtype=bool).copy()
    tree = cKDTree(v)
    for hits in tree.query_ball_point(mirrored, r=radius):
        out[hits] = True
    return out


def fence_flood(mesh, fence_mask, seed_mask, adj=None):
    """Surface flood-fill bounded by a fence of vertices.

    The user's outline, transferred onto the mesh, is the fence; seeds
    are any vertices known to be inside. The flood reaches every vertex
    connected to a seed without crossing the fence -- including
    interior shells of hair that no viewpoint can see -- and can never
    leak past the outline onto protected surface. Returns
    flood | fence (the fence itself belongs to the replaced region;
    it is where the feather band lives)."""
    fence = np.asarray(fence_mask, dtype=bool)
    seeds = np.asarray(seed_mask, dtype=bool) & ~fence
    if not seeds.any():
        raise ValueError("no seeds inside the fence")
    if adj is None:
        adj = vertex_adjacency(mesh)
    open_idx = np.flatnonzero(~fence)
    sub = adj[open_idx][:, open_idx]
    _, labels = connected_components(sub, directed=False)
    seed_labels = np.unique(labels[seeds[open_idx]])
    flood = np.zeros(len(fence), dtype=bool)
    flood[open_idx[np.isin(labels, seed_labels)]] = True
    return flood | fence


def stranded_islands(mesh, region_mask, inside_fraction=0.7, adj=None):
    """Disconnected mesh components (floating scan junk) that lie
    mostly inside a region -- reachable by no flood, fillable by no
    solve (nothing anchors them). These are deletion candidates.
    region_mask: any vertex mask describing the replacement volume."""
    region = np.asarray(region_mask, dtype=bool)
    if adj is None:
        adj = vertex_adjacency(mesh)
    n_comp, labels = connected_components(adj, directed=False)
    main = np.argmax(np.bincount(labels))
    out = np.zeros(len(region), dtype=bool)
    for comp in range(n_comp):
        if comp == main:
            continue
        members = labels == comp
        if region[members].mean() >= inside_fraction:
            out[members] = True
    return out
