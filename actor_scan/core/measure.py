""" Head measurements: the sizing contract for replacements.

Implements the 13FingerFX measurement chart. Two physical kinds:
- caliper: straight-line distance between two landmark points.
- tape: length along the surface -- computed as a plane slice through
  the mesh (full loop, or an arc between two landmarks passing the
  side nearer an "over" hint point).

Landmarks are named 3D points the user places on the scan (later:
auto-suggested by a face-landmark model, always correctable). Every
measurement declares which landmarks it needs, so a chart can be
computed on the original scan, on a repaired/filled version, or on a
candidate replacement -- and compared, which is what "the replacement
matches our measurements" means operationally.
"""
from dataclasses import dataclass
import json
import numpy as np
import trimesh


@dataclass(frozen=True)
class Measurement:
    id: str
    label: str
    kind: str          # "caliper" | "loop" | "arc"
    landmarks: tuple   # names; arc = (a, b, over); loop = 3 on-plane pts


MEASUREMENTS = [
    Measurement("nose_to_back_of_head", "Nose to back of head", "caliper",
                ("pronasale", "back_of_head")),
    Measurement("eye_center_to_eye_center", "Eye ctr to eye ctr", "caliper",
                ("l_eye_center", "r_eye_center")),
    Measurement("brow_to_back_of_head", "Brow to back of head", "caliper",
                ("glabella", "back_of_head")),
    Measurement("neck_width", "Neck width", "caliper",
                ("l_neck_side", "r_neck_side")),
    Measurement("tragus_to_tragus", "Tragus to tragus", "caliper",
                ("l_tragus", "r_tragus")),
    Measurement("peak_to_bridge", "Peak to bridge", "caliper",
                ("hairline_peak", "nose_bridge")),
    Measurement("chin_to_crown_caliper", "Chin to crown (caliper)",
                "caliper", ("chin_bottom", "crown")),
    Measurement("brow_to_nape_caliper", "Brow to nape (caliper)",
                "caliper", ("glabella", "nape")),
    Measurement("ear_to_ear_over_crown", "Ear to ear circ (over crown)",
                "arc", ("l_ear_top", "r_ear_top", "crown")),
    Measurement("ear_to_ear_back", "Ear to ear circ (around back)",
                "arc", ("l_ear_top", "r_ear_top", "back_of_head")),
    Measurement("brow_to_nape_circ", "Brow to nape circ (over crown)",
                "arc", ("glabella", "nape", "crown")),
    Measurement("chin_to_crown_circ", "Chin to crown circ (loop)",
                "loop", ("chin_bottom", "crown", "l_ear_top")),
    Measurement("head_at_brow_circ", "Head at brow circumference",
                "loop", ("glabella", "l_above_ear", "r_above_ear")),
    Measurement("neck_circ", "Neck circumference", "loop",
                ("neck_front", "l_neck_side", "nape")),
]
BY_ID = {m.id: m for m in MEASUREMENTS}


def caliper(a, b):
    return float(np.linalg.norm(np.asarray(a, dtype=np.float64) -
                                np.asarray(b, dtype=np.float64)))


def _section_polylines(mesh, plane_origin, plane_normal, tol=None):
    """Ordered polylines of the mesh/plane section: list of
    (points (N,3), closed bool)."""
    segments = trimesh.intersections.mesh_plane(
        mesh, plane_normal, plane_origin)
    if len(segments) == 0:
        return []
    if tol is None:
        tol = float(mesh.scale) * 1e-8 + 1e-12
    keys = np.round(segments.reshape(-1, 3) / tol).astype(np.int64)
    point_id = {}
    coords = []
    ids = np.empty(len(keys), dtype=np.int64)
    for i, k in enumerate(map(tuple, keys)):
        if k not in point_id:
            point_id[k] = len(coords)
            coords.append(segments.reshape(-1, 3)[i])
        ids[i] = point_id[k]
    coords = np.asarray(coords)
    edges = ids.reshape(-1, 2)
    edges = edges[edges[:, 0] != edges[:, 1]]

    neighbors = {}
    for u, v in edges:
        neighbors.setdefault(u, set()).add(v)
        neighbors.setdefault(v, set()).add(u)

    unvisited_edges = {tuple(sorted(e)) for e in edges.tolist()}
    polylines = []
    while unvisited_edges:
        # Prefer starting from an endpoint (open chain), else any (loop).
        degree_odd = [n for n in neighbors
                      if len([m for m in neighbors[n]
                              if tuple(sorted((n, m))) in unvisited_edges])
                      == 1]
        start = degree_odd[0] if degree_odd else next(
            iter(unvisited_edges))[0]
        path = [start]
        current = start
        while True:
            nxt = None
            for m in neighbors.get(current, ()):
                if tuple(sorted((current, m))) in unvisited_edges:
                    nxt = m
                    break
            if nxt is None:
                break
            unvisited_edges.discard(tuple(sorted((current, nxt))))
            path.append(nxt)
            current = nxt
            if current == start:
                break
        closed = len(path) > 2 and path[0] == path[-1]
        polylines.append((coords[path], closed))
    return polylines


def _polyline_length(points):
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())


def _nearest_component(polylines, anchor_points):
    best = None
    for points, closed in polylines:
        d = 0.0
        for p in anchor_points:
            d += float(np.min(np.linalg.norm(points - p, axis=1)))
        if best is None or d < best[0]:
            best = (d, points, closed)
    return best[1], best[2]


def _plane_from_points(p1, p2, p3):
    p1, p2, p3 = (np.asarray(p, dtype=np.float64) for p in (p1, p2, p3))
    normal = np.cross(p2 - p1, p3 - p1)
    norm = np.linalg.norm(normal)
    if norm < 1e-12:
        raise ValueError("landmarks are collinear; cannot define a plane")
    return p1, normal / norm


def loop_length(mesh, p1, p2, p3):
    """Full tape loop: section by the plane through three landmarks,
    length of the component nearest them."""
    origin, normal = _plane_from_points(p1, p2, p3)
    polylines = _section_polylines(mesh, origin, normal)
    if not polylines:
        raise ValueError("plane does not intersect the mesh")
    points, closed = _nearest_component(polylines, [p1, p2, p3])
    if not closed:
        print("warning: tape path is not a closed loop on this mesh")
    return _polyline_length(points)


def arc_length(mesh, a, b, over):
    """Tape arc from a to b along the section through (a, b, over),
    taking the way around that passes nearer the `over` hint."""
    origin, normal = _plane_from_points(a, b, over)
    polylines = _section_polylines(mesh, origin, normal)
    if not polylines:
        raise ValueError("plane does not intersect the mesh")
    points, closed = _nearest_component(polylines, [a, b, over])
    ia = int(np.argmin(np.linalg.norm(points - np.asarray(a), axis=1)))
    ib = int(np.argmin(np.linalg.norm(points - np.asarray(b), axis=1)))
    if ia == ib:
        raise ValueError("arc endpoints coincide on the section")
    lo, hi = min(ia, ib), max(ia, ib)
    inner = points[lo:hi + 1]
    if closed:
        outer = np.vstack([points[hi:], points[1:lo + 1]])
        d_inner = float(np.min(np.linalg.norm(
            inner - np.asarray(over), axis=1)))
        d_outer = float(np.min(np.linalg.norm(
            outer - np.asarray(over), axis=1)))
        chosen = inner if d_inner <= d_outer else outer
    else:
        chosen = inner
    return _polyline_length(chosen)


def compute_chart(mesh, landmarks, only=None):
    """Compute every measurement whose landmarks are present.

    landmarks: dict name -> (x, y, z). Returns dict id -> value (mesh
    units) plus a "_missing" list of measurements that lacked points.
    """
    chart = {}
    missing = []
    for m in MEASUREMENTS:
        if only and m.id not in only:
            continue
        try:
            pts = [np.asarray(landmarks[name], dtype=np.float64)
                   for name in m.landmarks]
        except KeyError:
            missing.append(m.id)
            continue
        if m.kind == "caliper":
            chart[m.id] = caliper(pts[0], pts[1])
        elif m.kind == "loop":
            chart[m.id] = loop_length(mesh, *pts)
        else:
            chart[m.id] = arc_length(mesh, *pts)
    if missing:
        chart["_missing"] = missing
    return chart


def compare_charts(measured, target):
    """Per-measurement delta of measured vs target chart (dicts)."""
    out = {}
    for key, want in target.items():
        if key.startswith("_") or key not in measured:
            continue
        got = measured[key]
        out[key] = {"target": float(want), "measured": float(got),
                    "delta": float(got - want)}
    return out


def load_chart(path):
    with open(path) as f:
        return json.load(f)
