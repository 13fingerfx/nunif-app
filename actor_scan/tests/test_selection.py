import unittest
import numpy as np
import trimesh

from actor_scan.core import selection
from actor_scan.tests.test_bake import camera_looking_at_origin


class TestSelection(unittest.TestCase):
    def setUp(self):
        self.mesh = trimesh.creation.icosphere(subdivisions=3, radius=50.0)
        self.pose = camera_looking_at_origin()

    def test_polygon_selects_visible_enclosed_vertices(self):
        # Big square around the image center: the camera-facing side.
        polygon = [(120, 40), (520, 40), (520, 440), (120, 440)]
        mask = selection.polygon_to_mask(self.mesh, self.pose, polygon)
        self.assertTrue(mask.any())
        # Camera sits at world +z: nothing on the far side gets picked.
        self.assertEqual(mask[self.mesh.vertices[:, 2] < -10].sum(), 0)
        # Vertices near the facing pole must be inside.
        front = self.mesh.vertices[:, 2] > 45
        self.assertTrue(mask[front].all())

    def test_small_polygon_selects_less(self):
        big = selection.polygon_to_mask(
            self.mesh, self.pose, [(120, 40), (520, 40), (520, 440),
                                   (120, 440)])
        small = selection.polygon_to_mask(
            self.mesh, self.pose, [(300, 220), (340, 220), (340, 260),
                                   (300, 260)])
        self.assertLess(small.sum(), big.sum())
        self.assertTrue((small & ~big).sum() == 0)

    def test_grow_rings(self):
        polygon = [(300, 220), (340, 220), (340, 260), (300, 260)]
        base = selection.polygon_to_mask(self.mesh, self.pose, polygon)
        grown = selection.polygon_to_mask(self.mesh, self.pose, polygon,
                                          grow_rings=2)
        self.assertGreater(grown.sum(), base.sum())
        self.assertTrue((base & ~grown).sum() == 0)

    def test_bad_polygon(self):
        with self.assertRaises(ValueError):
            selection.polygon_to_mask(self.mesh, self.pose,
                                      [(0, 0), (1, 1)])


class TestMirrorMask(unittest.TestCase):
    def test_mirrors_across_axis(self):
        mesh = trimesh.creation.icosphere(subdivisions=3, radius=50.0)
        mask = mesh.vertices[:, 0] > 40.0
        out = selection.mirror_mask(mesh, mask, axis=0)
        far = mesh.vertices[:, 0] < -40.0
        self.assertGreater(out[far].mean(), 0.95)
        self.assertTrue((out & ~mask)[far].any())
        # Original selection preserved.
        self.assertTrue(out[mask].all())


class TestFenceFlood(unittest.TestCase):
    def setUp(self):
        self.mesh = trimesh.creation.icosphere(subdivisions=4, radius=50.0)
        z = self.mesh.vertices[:, 2]
        self.fence = np.abs(z - 25.0) < 2.5   # ring around the cap
        self.seeds = z > 45.0

    def test_flood_fills_cap_and_stops_at_fence(self):
        sel = selection.fence_flood(self.mesh, self.fence, self.seeds)
        z = self.mesh.vertices[:, 2]
        self.assertTrue(sel[z > 30].all())     # whole cap
        self.assertTrue(sel[self.fence].all()) # fence included
        self.assertEqual(sel[z < 15].sum(), 0) # nothing leaks below

    def test_no_seeds_raises(self):
        with self.assertRaises(ValueError):
            selection.fence_flood(self.mesh, self.fence,
                                  np.zeros(len(self.mesh.vertices), bool))


class TestStrandedIslands(unittest.TestCase):
    def test_floating_component_inside_region(self):
        main = trimesh.creation.icosphere(subdivisions=3, radius=50.0)
        junk = trimesh.creation.icosphere(subdivisions=1, radius=3.0)
        junk.apply_translation([0.0, 0.0, 70.0])
        both = trimesh.util.concatenate([main, junk])
        region = both.vertices[:, 2] > 60.0
        out = selection.stranded_islands(both, region)
        is_junk = np.linalg.norm(both.vertices - [0, 0, 70], axis=1) < 3.5
        self.assertTrue(out[is_junk].all())
        self.assertEqual(out[~is_junk].sum(), 0)

    def test_island_outside_region_is_kept(self):
        main = trimesh.creation.icosphere(subdivisions=3, radius=50.0)
        junk = trimesh.creation.icosphere(subdivisions=1, radius=3.0)
        junk.apply_translation([0.0, 0.0, 70.0])
        both = trimesh.util.concatenate([main, junk])
        region = both.vertices[:, 2] < -40.0   # elsewhere entirely
        out = selection.stranded_islands(both, region)
        self.assertEqual(out.sum(), 0)


if __name__ == "__main__":
    unittest.main()
