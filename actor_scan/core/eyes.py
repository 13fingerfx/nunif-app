""" Parametric replacement eye forms.

Scanned eyes are classically horrifying: the transparent cornea and wet
sclera confuse structured light and photogrammetry alike, leaving a
crumpled bulge. Standard practice in figure/prop work is to replace
them with clean sculpted forms, so this module generates them:

- style="sphere": a plain ball -- correct silhouette, no surface
  detail. The right choice when the print will be painted or when the
  eye is mostly closed.
- style="sculpted": the figure-sculptor eye -- corneal plateau raised
  above the sclera sphere, iris dished back inside a crisp limbus ring.
  Reads as an eye in monochrome print/mould output without attempting
  (and failing) photoreal wetness.

Human eyeball diameter is famously consistent (~24 mm transverse in
adults, ~19.5 mm at birth), which is why preset diameters work at all.
Placement is center + gaze direction; blending the form into the scan
socket topology is the same commit/remesh step as other replacements.
"""
import numpy as np
import trimesh

PRESET_DIAMETERS = {
    "infant": 19.5,
    "child": 21.0,
    "teen": 23.0,
    "adult": 24.0,
    "large": 25.5,
}


def _smoothstep(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def eye_form(diameter=24.0, style="sculpted", iris_diameter=11.8,
             cornea_bulge=1.1, iris_recess=0.45, subdivisions=4):
    """Generate an eye form mesh, gaze along +z, centered at origin.

    diameter / iris_diameter / cornea_bulge / iris_recess in mm.
    Returns a watertight Trimesh.
    """
    if style not in ("sphere", "sculpted"):
        raise ValueError("style must be 'sphere' or 'sculpted'")
    radius = diameter / 2.0
    mesh = trimesh.creation.icosphere(subdivisions=subdivisions,
                                      radius=radius)
    if style == "sphere":
        return mesh
    if not 0 < iris_diameter < diameter:
        raise ValueError("iris_diameter must be between 0 and diameter")

    verts = np.array(mesh.vertices, dtype=np.float64)
    directions = verts / np.linalg.norm(verts, axis=1, keepdims=True)
    theta = np.arccos(np.clip(directions[:, 2], -1.0, 1.0))
    theta_iris = np.arcsin(min(iris_diameter / diameter, 0.999))
    theta_cornea = theta_iris * 1.45

    # Corneal plateau: full height across the iris, easing to the
    # sclera sphere beyond the limbus.
    plateau = _smoothstep((theta_cornea - theta) / (0.35 * theta_cornea))
    # Iris dish: sunk back inside the limbus, flat-bottomed.
    dish = _smoothstep((theta_iris - theta) / (0.5 * theta_iris))
    radial = radius + cornea_bulge * plateau - iris_recess * dish
    return trimesh.Trimesh(vertices=directions * radial[:, None],
                           faces=mesh.faces, process=False)


def place_eye(eye_mesh, center, aim=(0.0, 0.0, 1.0)):
    """Position an eye form: gaze axis (+z) rotated onto `aim`, origin
    moved to `center`. Returns a transformed copy."""
    aim = np.asarray(aim, dtype=np.float64)
    norm = np.linalg.norm(aim)
    if norm < 1e-12:
        raise ValueError("aim direction must be nonzero")
    rotation = trimesh.geometry.align_vectors([0.0, 0.0, 1.0], aim / norm)
    placed = eye_mesh.copy()
    placed.apply_transform(rotation)
    placed.apply_translation(np.asarray(center, dtype=np.float64))
    return placed
