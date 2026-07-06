""" Commit: turn a repaired scan into a print-ready solid.

Everything upstream edits geometry non-destructively on raw scan
topology, which is open (bust base, sensor dropouts, junk-deletion
holes) and unprintable. Commit is the terminal step: keep the real
surface, cap every boundary loop, make winding consistent, and verify
watertightness -- reporting volume and dimensions, since a print quote
and a mould both start from those numbers. It produces a NEW artifact;
per the project rule, nothing upstream is touched.
"""
import numpy as np
import trimesh


def boundary_loops(mesh):
    """Ordered vertex loops of every open boundary. Loops touching a
    non-manifold junction (a vertex with more than two boundary edges)
    are returned as-is if walkable, otherwise skipped and reported."""
    edges = mesh.edges_sorted
    unique, counts = np.unique(edges, axis=0, return_counts=True)
    boundary = unique[counts == 1]
    neighbors = {}
    for a, b in boundary:
        neighbors.setdefault(int(a), []).append(int(b))
        neighbors.setdefault(int(b), []).append(int(a))
    unused = {tuple(e) for e in boundary.tolist()}
    loops, skipped = [], 0
    while unused:
        start, current = next(iter(unused))
        loop = [start, current]
        unused.discard((start, current))
        ok = True
        while loop[-1] != start:
            here = loop[-1]
            options = [n for n in neighbors.get(here, [])
                       if tuple(sorted((here, n))) in unused]
            if not options:
                ok = False
                break
            nxt = options[0]
            unused.discard(tuple(sorted((here, nxt))))
            loop.append(nxt)
        if ok and len(loop) > 3:
            loops.append(loop[:-1])
        else:
            skipped += 1
    return loops, skipped


def cap_loops(mesh, loops):
    """Close boundary loops with centroid fans. Winding is repaired
    globally afterwards, so fan orientation here is arbitrary."""
    if not loops:
        return mesh
    vertices = [np.asarray(mesh.vertices)]
    faces = [np.asarray(mesh.faces)]
    next_index = len(mesh.vertices)
    for loop in loops:
        ring = np.asarray(loop)
        centroid = np.asarray(mesh.vertices)[ring].mean(axis=0)
        vertices.append(centroid[None, :])
        fan = np.column_stack([ring, np.roll(ring, -1),
                               np.full(len(ring), next_index)])
        faces.append(fan)
        next_index += 1
    return trimesh.Trimesh(vertices=np.vstack(vertices),
                           faces=np.vstack(faces), process=False)


def drop_nonmanifold_faces(mesh):
    """Remove faces incident to any edge shared by more than two faces
    (bowties and internal fins -- scanners emit a few). The small holes
    this opens are then closed by the normal capping pass.
    Returns (mesh, number of faces removed)."""
    edges = mesh.edges_sorted
    unique, inverse, counts = np.unique(edges, axis=0,
                                        return_inverse=True,
                                        return_counts=True)
    bad_edge = counts > 2
    if not bad_edge.any():
        return mesh, 0
    bad_face = bad_edge[inverse].reshape(-1, 3).any(axis=1)
    out = trimesh.Trimesh(vertices=mesh.vertices.copy(),
                          faces=mesh.faces[~bad_face], process=False)
    return out, int(bad_face.sum())


def commit(mesh, keep="largest", cap=True, min_component_faces=None):
    """Finalize a repaired mesh into a printable solid.

    keep: "largest" keeps only the biggest connected component (scan
    debris outside any selection never gets printed); "all" keeps
    every component above min_component_faces (default 0.1% of faces).
    Returns (solid Trimesh, report dict).
    """
    report = {}
    work = mesh.copy()
    work.merge_vertices(merge_tex=True, merge_norm=True)
    work.update_faces(work.nondegenerate_faces())
    work.remove_unreferenced_vertices()

    pieces = work.split(only_watertight=False)
    if len(pieces) > 1:
        pieces = sorted(pieces, key=lambda p: len(p.faces), reverse=True)
        if keep == "largest":
            kept = [pieces[0]]
        else:
            threshold = (min_component_faces if min_component_faces
                         is not None else max(len(work.faces) // 1000, 1))
            kept = [p for p in pieces if len(p.faces) >= threshold]
        report["components_dropped"] = len(pieces) - len(kept)
        work = (kept[0] if len(kept) == 1
                else trimesh.util.concatenate(kept))
    else:
        report["components_dropped"] = 0

    work, removed = drop_nonmanifold_faces(work)
    report["nonmanifold_faces_removed"] = removed

    if cap:
        loops, skipped = boundary_loops(work)
        work = cap_loops(work, loops)
        report["loops_capped"] = len(loops)
        report["loops_skipped"] = skipped
        work.merge_vertices(merge_tex=True, merge_norm=True)
        work.update_faces(work.nondegenerate_faces())

    trimesh.repair.fix_normals(work)
    report["watertight"] = bool(work.is_watertight)
    report["winding_consistent"] = bool(work.is_winding_consistent)
    report["vertices"] = len(work.vertices)
    report["faces"] = len(work.faces)
    extent = work.bounds[1] - work.bounds[0]
    report["extent"] = [float(e) for e in extent]
    if report["watertight"]:
        report["volume"] = float(abs(work.volume))
    return work, report
