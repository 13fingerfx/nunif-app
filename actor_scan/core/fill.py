""" Replace marked scan regions with smooth generic geometry.

fill_regions() frees the masked vertices and solves a (bi)harmonic
system over the mesh graph with the surrounding vertices as boundary
constraints: the marked blob (hair, wig cap, beard noise) is re-shaped
into a smooth continuation of the surrounding surface -- the automated
"bald" pass. Biharmonic keeps slope continuity at the boundary, which
is what makes a filled scalp read as a skull instead of a soap film.

Limits, by design: this moves existing vertices; it cannot invent
faces, so true topological holes need closing first (small ones:
trimesh.repair.fill_holes). Library-shaped replacement (fitting a
generic ear/scalp exemplar instead of a smooth continuation) is the
planned upgrade and will slot in as another method= here; the smooth
fill is also what the guided-synthesis/overlay stages expect to
decorate afterwards.
"""
import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import spsolve
import trimesh

from .defects import vertex_adjacency


def graph_laplacian(mesh):
    adj = vertex_adjacency(mesh)
    deg = np.asarray(adj.sum(axis=1)).reshape(-1)
    return sparse.diags(deg) - adj


def _drop_unconstrained(mask, adj):
    """Unmask free components with no fixed neighbor (nothing to anchor
    the solve); returns (mask, number of dropped components)."""
    idx = np.flatnonzero(mask)
    if len(idx) == 0:
        return mask, 0
    sub = adj[idx][:, idx]
    n_comp, labels = connected_components(sub, directed=False)
    fixed = ~mask
    dropped = 0
    out = mask.copy()
    for comp in range(n_comp):
        members = idx[labels == comp]
        touches = np.asarray(
            adj[members][:, fixed].sum()) if fixed.any() else 0
        if touches == 0:
            out[members] = False
            dropped += 1
    return out, dropped


# Per-zone shaping presets: fullness is the outward offset (mesh units,
# mm for most scanners) at the deepest point of the region; taper shapes
# the dome (higher = flatter rim, rounder center). These are starting
# points for character-creator-style sliders, not fixed anatomy.
PROFILES = {
    "scalp": {"fullness": 0.0, "taper": 1.0},   # pure skull continuation
    "brow":  {"fullness": 1.5, "taper": 1.2},   # gentle ridge
    "beard": {"fullness": 4.0, "taper": 0.8},   # fuller, softer dome
}


def _boundary_rings(mask, adj):
    """Integer BFS depth (0 at the region rim) for masked vertices;
    -1 outside the mask."""
    depth = np.full(len(mask), -1, dtype=np.int64)
    outside = ~mask
    touches_out = np.asarray(
        adj @ outside.astype(np.float64)).reshape(-1) > 0
    frontier = np.flatnonzero(mask & touches_out)
    depth[frontier] = 0
    ring = 0
    while len(frontier):
        reached = np.asarray(
            adj[frontier].sum(axis=0)).reshape(-1) > 0
        nxt = np.flatnonzero(mask & reached & (depth < 0))
        ring += 1
        depth[nxt] = ring
        frontier = nxt
    return depth


def _boundary_distance(mask, adj):
    """Normalized BFS depth (0 at the region rim, 1 at the deepest
    interior vertex) for every masked vertex."""
    depth = _boundary_rings(mask, adj)
    deepest = depth[mask].max()
    norm = np.zeros(len(mask))
    if deepest > 0:
        norm[mask] = depth[mask] / float(deepest)
    return norm


def shape_fill(mesh, mask, fullness, taper=1.0, adj=None):
    """Offset a filled region outward along its (smooth) normals with a
    dome falloff -- the 'fullness' slider. Returns a new Trimesh."""
    if fullness == 0.0:
        return mesh.copy()
    if adj is None:
        adj = vertex_adjacency(mesh)
    d = _boundary_distance(mask, adj)
    s = d * d * (3.0 - 2.0 * d)          # smoothstep dome
    falloff = np.power(s, taper, where=s > 0, out=np.zeros_like(s))
    vertices = np.array(mesh.vertices, dtype=np.float64)
    vertices[mask] += (mesh.vertex_normals[mask] *
                       (fullness * falloff[mask])[:, None])
    return trimesh.Trimesh(vertices=vertices, faces=mesh.faces,
                           process=False)


def fill_regions(mesh, vertex_mask, method="biharmonic", profile=None,
                 fullness=None, taper=None, locked=None, feather_rings=0):
    """Re-shape masked vertices as a smooth continuation of the rest.

    method: "biharmonic" (slope-continuous, default) or "laplacian"
    (position-continuous only; cheaper, tent-like near the boundary).
    profile: optional preset name from PROFILES ("scalp"/"brow"/"beard");
    fullness/taper override the preset's values when given, so the same
    parameters double as interactive sliders.
    locked: optional boolean vertex mask of protected geometry. Locked
    vertices can NEVER move, even if the selection includes them --
    they are subtracted from the fill region and act as boundary
    constraints instead. Defense in depth against a sloppy outline.
    feather_rings: width (in edge rings, from the region rim inward) of
    a transition band where displacement fades to zero at the border,
    anchoring the result to the original surface around the edge.
    Returns (new Trimesh, effective mask actually filled).
    """
    if method not in ("biharmonic", "laplacian"):
        raise ValueError("method must be 'biharmonic' or 'laplacian'")
    if profile is not None and profile not in PROFILES:
        raise ValueError(f"profile must be one of {sorted(PROFILES)}")
    preset = dict(PROFILES.get(profile, {"fullness": 0.0, "taper": 1.0}))
    if fullness is not None:
        preset["fullness"] = fullness
    if taper is not None:
        preset["taper"] = taper
    mask = np.asarray(vertex_mask, dtype=bool)
    if mask.shape != (len(mesh.vertices),):
        raise ValueError("mask length must equal the vertex count")
    if locked is not None:
        locked = np.asarray(locked, dtype=bool)
        if locked.shape != mask.shape:
            raise ValueError("locked mask length must equal vertex count")
        overlap = int((mask & locked).sum())
        if overlap:
            print(f"lock: excluded {overlap} selected vertices inside "
                  "the locked region")
        mask = mask & ~locked
    adj = vertex_adjacency(mesh)
    mask, dropped = _drop_unconstrained(mask, adj)
    if dropped:
        print(f"warning: skipped {dropped} fully-masked component(s) "
              "with no anchor vertices")
    if not mask.any():
        return mesh.copy(), mask

    lap = graph_laplacian(mesh).tocsr()
    op = (lap @ lap).tocsr() if method == "biharmonic" else lap
    free = np.flatnonzero(mask)
    fixed = np.flatnonzero(~mask)
    a = op[free][:, free].tocsc()
    b = -op[free][:, fixed] @ mesh.vertices[fixed]
    solution = spsolve(a, b)
    vertices = np.array(mesh.vertices, dtype=np.float64)
    vertices[free] = solution
    filled = trimesh.Trimesh(vertices=vertices, faces=mesh.faces,
                             process=False)
    if preset["fullness"] != 0.0:
        filled = shape_fill(filled, mask, preset["fullness"],
                            preset["taper"], adj=adj)
    if feather_rings > 0:
        rings = _boundary_rings(mask, adj)
        t = np.clip(rings / float(feather_rings), 0.0, 1.0)
        w = t * t * (3.0 - 2.0 * t)
        blend = np.array(filled.vertices, dtype=np.float64)
        blend[mask] = (np.asarray(mesh.vertices)[mask] * (1 - w[mask, None])
                       + blend[mask] * w[mask, None])
        filled = trimesh.Trimesh(vertices=blend, faces=mesh.faces,
                                 process=False)
    return filled, mask
