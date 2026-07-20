import unittest
import numpy as np
import trimesh

from actor_scan.core import defects

SKIN = np.array([0.80, 0.60, 0.50])
HAIR = np.array([0.15, 0.12, 0.10])
CAP_BLUE = np.array([0.20, 0.30, 0.80])


def head_with_cap(spiky=True, cap_color=HAIR, seed=0):
    """Icosphere 'head': cap region (z > 25) gets hair-like treatment."""
    rng = np.random.default_rng(seed)
    mesh = trimesh.creation.icosphere(subdivisions=4, radius=50.0)
    cap = mesh.vertices[:, 2] > 25.0
    if spiky:
        vertices = np.array(mesh.vertices)
        bump = 1.0 + rng.uniform(0.0, 0.12, size=int(cap.sum()))
        vertices[cap] *= bump[:, None]
        mesh = trimesh.Trimesh(vertices=vertices, faces=mesh.faces,
                               process=False)
    colors = np.tile(SKIN, (len(mesh.vertices), 1))
    colors += rng.normal(0, 0.02, colors.shape)
    colors[cap] = cap_color + rng.normal(0, 0.02, (int(cap.sum()), 3))
    return mesh, np.clip(colors, 0, 1), cap


def precision_recall(mask, truth):
    tp = float((mask & truth).sum())
    return tp / max(mask.sum(), 1), tp / max(truth.sum(), 1)


class TestDefectEstimator(unittest.TestCase):
    def test_spiky_dark_hair_is_detected(self):
        mesh, colors, cap = head_with_cap(spiky=True, cap_color=HAIR)
        mask, diag = defects.estimate_replace_mask(mesh, colors)
        precision, recall = precision_recall(mask, cap)
        self.assertGreater(recall, 0.9, f"recall {recall:.2f}")
        self.assertGreater(precision, 0.7, f"precision {precision:.2f}")

    def test_smooth_wig_cap_needs_color_signal(self):
        mesh, colors, cap = head_with_cap(spiky=False, cap_color=CAP_BLUE)
        # Geometry-only pass misses the smooth cap...
        mask_geo, _ = defects.estimate_replace_mask(mesh, colors=None)
        _, recall_geo = precision_recall(mask_geo, cap)
        self.assertLess(recall_geo, 0.2)
        # ...the color signal catches it.
        mask, _ = defects.estimate_replace_mask(mesh, colors)
        precision, recall = precision_recall(mask, cap)
        self.assertGreater(recall, 0.9, f"recall {recall:.2f}")
        self.assertGreater(precision, 0.7, f"precision {precision:.2f}")

    def test_clean_skin_scan_is_left_alone(self):
        rng = np.random.default_rng(1)
        mesh = trimesh.creation.icosphere(subdivisions=4, radius=50.0)
        colors = np.tile(SKIN, (len(mesh.vertices), 1))
        colors += rng.normal(0, 0.02, colors.shape)
        mask, _ = defects.estimate_replace_mask(mesh, colors)
        self.assertLess(float(mask.mean()), 0.02)

    def test_speckle_is_removed(self):
        """A handful of isolated rough vertices must not survive the
        component filter."""
        rng = np.random.default_rng(2)
        mesh = trimesh.creation.icosphere(subdivisions=4, radius=50.0)
        vertices = np.array(mesh.vertices)
        lone = rng.choice(len(vertices), 5, replace=False)
        vertices[lone] *= 1.05
        mesh = trimesh.Trimesh(vertices=vertices, faces=mesh.faces,
                               process=False)
        mask, _ = defects.estimate_replace_mask(mesh, colors=None)
        self.assertEqual(int(mask.sum()), 0)

    def test_dark_skin_is_not_flagged_as_hair(self):
        """The skin model is fitted per-scan: a uniformly dark
        complexion must not be treated as an outlier."""
        rng = np.random.default_rng(3)
        mesh = trimesh.creation.icosphere(subdivisions=4, radius=50.0)
        dark_skin = np.array([0.35, 0.24, 0.18])
        colors = np.tile(dark_skin, (len(mesh.vertices), 1))
        colors += rng.normal(0, 0.02, colors.shape)
        mask, _ = defects.estimate_replace_mask(mesh, colors)
        self.assertLess(float(mask.mean()), 0.02)


if __name__ == "__main__":
    unittest.main()
