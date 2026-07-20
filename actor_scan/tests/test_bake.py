import unittest
import numpy as np
import trimesh

from actor_scan.core import bake, registration


def camera_looking_at_origin(distance=300.0, focal=800.0, size=(640, 480)):
    """Camera on +z world axis looking back at the origin (OpenCV frame)."""
    # World point (0,0,0) must land at depth `distance` on the optical
    # axis: R = rotate 180deg about x maps world +z to camera -z.
    rvec = np.array([np.pi, 0.0, 0.0])
    tvec = np.array([0.0, 0.0, distance])
    k = registration.camera_matrix(focal, size)
    return registration.CameraPose(rvec, tvec, k, size)


class TestBake(unittest.TestCase):
    def setUp(self):
        self.mesh = trimesh.creation.icosphere(subdivisions=3, radius=50.0)
        self.pose = camera_looking_at_origin()

    def test_constant_height_inflates_visible_side(self):
        height_map = np.ones((480, 640), dtype=np.float32)
        baked, weight = bake.bake(self.mesh, [(self.pose, height_map)],
                                  height_scale=2.0)
        radii_before = np.linalg.norm(self.mesh.vertices, axis=1)
        radii_after = np.linalg.norm(baked.vertices, axis=1)
        moved = weight > 0
        self.assertTrue(moved.any(), "no vertex was touched")
        np.testing.assert_allclose(radii_after[moved],
                                   radii_before[moved] + 2.0, atol=0.05)
        np.testing.assert_allclose(radii_after[~moved], radii_before[~moved])

    def test_far_side_is_untouched(self):
        height_map = np.ones((480, 640), dtype=np.float32)
        _, weight = bake.bake(self.mesh, [(self.pose, height_map)],
                              height_scale=1.0)
        # Camera sits at world +z: vertices with z < 0 face away.
        far_side = self.mesh.vertices[:, 2] < -10.0
        self.assertEqual(weight[far_side].max(), 0.0)
        near_side = self.mesh.vertices[:, 2] > 10.0
        self.assertGreater(weight[near_side].max(), 0.0)

    def test_occlusion_blocks_hidden_geometry(self):
        # A small sphere hiding behind the big one relative to the camera.
        back = trimesh.creation.icosphere(subdivisions=2, radius=10.0)
        back.apply_translation([0.0, 0.0, -80.0])
        scene = trimesh.util.concatenate([self.mesh, back])
        height_map = np.ones((480, 640), dtype=np.float32)
        _, weight = bake.bake(scene, [(self.pose, height_map)],
                              height_scale=1.0, occlusion=True)
        back_verts = np.linalg.norm(
            scene.vertices - [0.0, 0.0, -80.0], axis=1) < 10.5
        self.assertEqual(weight[back_verts].max(), 0.0,
                         "occluded sphere should receive no detail")

    def test_subdivision_densifies(self):
        dense = bake.subdivide_to_detail(self.mesh, max_edge=5.0)
        self.assertGreater(len(dense.vertices), len(self.mesh.vertices))
        self.assertLessEqual(dense.edges_unique_length.max(), 5.0 + 1e-6)

    def test_multi_view_blend(self):
        pose2 = camera_looking_at_origin()  # identical second view
        height_map = np.ones((480, 640), dtype=np.float32)
        baked_one, _ = bake.bake(self.mesh, [(self.pose, height_map)],
                                 height_scale=2.0)
        baked_two, _ = bake.bake(self.mesh,
                                 [(self.pose, height_map),
                                  (pose2, height_map)], height_scale=2.0)
        # Two identical views must agree with one (weighted average).
        np.testing.assert_allclose(baked_two.vertices, baked_one.vertices,
                                   atol=1e-6)


if __name__ == "__main__":
    unittest.main()
