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


def fill_regions(mesh, vertex_mask, method="biharmonic"):
    """Re-shape masked vertices as a smooth continuation of the rest.

    method: "biharmonic" (slope-continuous, default) or "laplacian"
    (position-continuous only; cheaper, tent-like near the boundary).
    Returns (new Trimesh, effective mask actually filled).
    """
    if method not in ("biharmonic", "laplacian"):
        raise ValueError("method must be 'biharmonic' or 'laplacian'")
    mask = np.asarray(vertex_mask, dtype=bool)
    if mask.shape != (len(mesh.vertices),):
        raise ValueError("mask length must equal the vertex count")
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
    return trimesh.Trimesh(vertices=vertices, faces=mesh.faces,
                           process=False), mask
