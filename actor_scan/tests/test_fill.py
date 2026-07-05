import unittest
import numpy as np
import trimesh

from actor_scan.core import fill, defects
from actor_scan.tests.test_defects import head_with_cap


class TestFill(unittest.TestCase):
    def test_bald_pass_smooths_spiky_cap(self):
        mesh, _, cap = head_with_cap(spiky=True)
        filled, effective = fill.fill_regions(mesh, cap)
        self.assertTrue(effective.any())
        rough_before = defects.roughness(mesh)[cap].mean()
        rough_after = defects.roughness(filled)[cap].mean()
        self.assertLess(rough_after, 0.15 * rough_before,
                        f"{rough_before:.4f} -> {rough_after:.4f}")
        # Filled dome stays head-sized: no collapse, no balloon.
        radii = np.linalg.norm(filled.vertices[cap], axis=1)
        self.assertGreater(radii.min(), 35.0)
        self.assertLess(radii.max(), 58.0)
        # Untouched vertices are bit-identical.
        np.testing.assert_array_equal(filled.vertices[~cap],
                                      mesh.vertices[~cap])

    def test_boundary_continuity(self):
        mesh, _, cap = head_with_cap(spiky=True)
        filled, _ = fill.fill_regions(mesh, cap)
        adj = defects.vertex_adjacency(mesh)
        rim = cap & (np.asarray(
            adj @ (~cap).astype(float)).reshape(-1) > 0)
        rim_radii = np.linalg.norm(filled.vertices[rim], axis=1)
        np.testing.assert_allclose(rim_radii, 50.0, atol=3.0)

    def test_laplacian_method_also_works(self):
        mesh, _, cap = head_with_cap(spiky=True)
        filled, _ = fill.fill_regions(mesh, cap, method="laplacian")
        rough_after = defects.roughness(filled)[cap].mean()
        self.assertLess(rough_after, 0.02)

    def test_fully_masked_mesh_is_skipped(self):
        mesh = trimesh.creation.icosphere(subdivisions=2, radius=10.0)
        mask = np.ones(len(mesh.vertices), dtype=bool)
        filled, effective = fill.fill_regions(mesh, mask)
        self.assertFalse(effective.any())
        np.testing.assert_array_equal(filled.vertices, mesh.vertices)

    def test_fullness_domes_the_fill(self):
        mesh, _, cap = head_with_cap(spiky=True)
        flat, _ = fill.fill_regions(mesh, cap)
        domed, _ = fill.fill_regions(mesh, cap, profile="beard")
        adj = defects.vertex_adjacency(mesh)
        rim = cap & (np.asarray(
            adj @ (~cap).astype(float)).reshape(-1) > 0)
        interior = cap & ~rim
        lift = (np.linalg.norm(domed.vertices[interior], axis=1) -
                np.linalg.norm(flat.vertices[interior], axis=1))
        # Deepest interior approaches the beard preset's 4mm fullness...
        self.assertGreater(lift.max(), 2.5)
        # ...while the rim stays put (blend into surrounding skin).
        rim_shift = np.linalg.norm(domed.vertices[rim] -
                                   flat.vertices[rim], axis=1)
        self.assertLess(rim_shift.max(), 0.3)

    def test_explicit_fullness_overrides_profile(self):
        mesh, _, cap = head_with_cap(spiky=True)
        a, _ = fill.fill_regions(mesh, cap, profile="beard", fullness=0.0)
        b, _ = fill.fill_regions(mesh, cap)
        np.testing.assert_allclose(a.vertices, b.vertices)

    def test_unknown_profile_rejected(self):
        mesh, _, cap = head_with_cap(spiky=True)
        with self.assertRaises(ValueError):
            fill.fill_regions(mesh, cap, profile="mohawk")

    def test_bad_inputs(self):
        mesh = trimesh.creation.icosphere(subdivisions=2, radius=10.0)
        mask = np.zeros(len(mesh.vertices), dtype=bool)
        with self.assertRaises(ValueError):
            fill.fill_regions(mesh, mask[:-1])
        with self.assertRaises(ValueError):
            fill.fill_regions(mesh, mask, method="magic")

    def test_estimator_to_fill_pipeline(self):
        """mark -> fill end-to-end: hairy scan in, bald scan out."""
        mesh, colors, cap = head_with_cap(spiky=True)
        mask, _ = defects.estimate_replace_mask(mesh, colors)
        filled, effective = fill.fill_regions(mesh, mask)
        self.assertTrue(effective.any())
        # Filled cap should be at least as smooth as pristine geometry.
        baseline = trimesh.creation.icosphere(subdivisions=4, radius=50.0)
        clean = defects.roughness(baseline)[cap].mean()
        rough = defects.roughness(filled)
        self.assertLess(rough[cap].mean(), clean * 1.5)


if __name__ == "__main__":
    unittest.main()
