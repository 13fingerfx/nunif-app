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

from .bake import vertex_weights


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
        from .defects import vertex_adjacency
        adj = vertex_adjacency(mesh)
        for _ in range(grow_rings):
            mask = mask | (np.asarray(
                adj @ mask.astype(np.float64)).reshape(-1) > 0)
    return mask
