""" Bake detail height maps into real geometry.

The scan mesh is subdivided to the target detail density, every vertex is
projected through the solved camera pose into the detail height map, and
visible vertices are displaced along their normals. Displacement goes
through the photo projection directly -- never through the scan's UV
atlas -- so fragmented UV islands can't produce cracks (build_plan flaw 4).

height_scale is in mesh units (Miraco-class scans export mm); it is the
explicit "estimation knob" for the physically unknowable amplitude of
photometric detail (build_plan flaw 3). Sign flips are allowed (negative
scale) if a source's height convention is inverted.
"""
import numpy as np
import cv2
import trimesh


def subdivide_to_detail(mesh, max_edge):
    """Midpoint-subdivide until no edge exceeds max_edge (mesh units)."""
    v, f, _ = trimesh.remesh.subdivide_to_size(
        mesh.vertices, mesh.faces, max_edge, return_index=True)
    return trimesh.Trimesh(vertices=v, faces=f, process=False)


def sample_height(height_map, points2d):
    """Bilinear height sample; returns (values, in-bounds mask)."""
    hm = np.asarray(height_map, dtype=np.float32)
    pts = np.asarray(points2d, dtype=np.float32)
    h, w = hm.shape
    inside = ((pts[:, 0] >= 0) & (pts[:, 0] <= w - 1) &
              (pts[:, 1] >= 0) & (pts[:, 1] <= h - 1))
    mapx = pts[:, 0].reshape(-1, 1)
    mapy = pts[:, 1].reshape(-1, 1)
    values = cv2.remap(hm, mapx, mapy, cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return values.reshape(-1), inside


def vertex_weights(mesh, pose, occlusion=True, min_cos=0.2, eps_rel=1e-3):
    """Per-vertex projection weight for one camera.

    Weight = cos(view angle) for vertices that face the camera, are inside
    the image, and (optionally) pass an occlusion ray test; 0 otherwise.
    """
    verts = mesh.vertices
    center = pose.camera_center
    to_cam = center - verts
    dist = np.linalg.norm(to_cam, axis=1)
    to_cam_dir = to_cam / dist[:, None]
    cos = (mesh.vertex_normals * to_cam_dir).sum(axis=1)
    weight = np.where(cos >= min_cos, cos, 0.0)

    proj = pose.project(verts)
    w, h = pose.image_size
    inside = ((proj[:, 0] >= 0) & (proj[:, 0] <= w - 1) &
              (proj[:, 1] >= 0) & (proj[:, 1] <= h - 1))
    weight[~inside] = 0.0

    if occlusion:
        active = weight > 0
        if active.any():
            origins = np.repeat(center[None, :], int(active.sum()), axis=0)
            dirs = -to_cam_dir[active]
            locations, ray_idx, _ = mesh.ray.intersects_location(
                origins, dirs, multiple_hits=False)
            hit_dist = np.full(int(active.sum()), np.inf)
            if len(ray_idx):
                hit_dist[ray_idx] = np.linalg.norm(
                    locations - origins[ray_idx], axis=1)
            vert_dist = dist[active]
            visible = hit_dist >= vert_dist * (1.0 - eps_rel)
            idx = np.flatnonzero(active)
            weight[idx[~visible]] = 0.0
    return weight, proj


def bake(mesh, views, height_scale, max_edge=None, occlusion=True,
         min_cos=0.2):
    """Displace a mesh by one or more (pose, height_map) views.

    views: list of (CameraPose, 2D height map) pairs; overlapping views
    are blended by view-angle weight. Returns (new Trimesh, per-vertex
    total weight -- zero means the vertex was untouched).
    """
    if max_edge is not None:
        mesh = subdivide_to_detail(mesh, max_edge)
    else:
        mesh = mesh.copy()

    accum = np.zeros(len(mesh.vertices))
    total_w = np.zeros(len(mesh.vertices))
    for pose, height_map in views:
        weight, proj = vertex_weights(mesh, pose, occlusion=occlusion,
                                      min_cos=min_cos)
        values, inside = sample_height(height_map, proj)
        weight = np.where(inside, weight, 0.0)
        accum += values * weight
        total_w += weight

    covered = total_w > 0
    displacement = np.zeros(len(mesh.vertices))
    displacement[covered] = accum[covered] / total_w[covered] * height_scale
    new_vertices = mesh.vertices + mesh.vertex_normals * displacement[:, None]
    baked = trimesh.Trimesh(vertices=new_vertices, faces=mesh.faces,
                            process=False)
    return baked, total_w
