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


if __name__ == "__main__":
    unittest.main()
