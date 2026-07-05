import unittest
import numpy as np
import cv2

from actor_scan.core import registration


def synthetic_scene(seed=5, n=12):
    """Random 3D points in front of a known camera, projected to 2D."""
    rng = np.random.default_rng(seed)
    pts3 = rng.uniform([-50, -50, 150], [50, 50, 250], size=(n, 3))
    rvec = np.array([0.1, -0.2, 0.05])
    tvec = np.array([2.0, -3.0, 10.0])
    size = (1920, 1080)
    focal = 1600.0
    k = registration.camera_matrix(focal, size)
    proj, _ = cv2.projectPoints(pts3.reshape(-1, 1, 3), rvec, tvec, k, None)
    return pts3, proj.reshape(-1, 2), rvec, tvec, focal, size


class TestRegistration(unittest.TestCase):
    def test_known_focal(self):
        pts3, pts2, rvec, tvec, focal, size = synthetic_scene()
        pose, err = registration.solve_pose(pts3, pts2, size, focal_px=focal)
        self.assertLess(err, 0.5)
        np.testing.assert_allclose(pose.tvec, tvec, atol=0.1)
        np.testing.assert_allclose(pose.rvec, rvec, atol=0.01)

    def test_focal_sweep(self):
        pts3, pts2, _, tvec, _, size = synthetic_scene()
        pose, err = registration.solve_pose(pts3, pts2, size)
        self.assertLess(err, 5.0, f"sweep reprojection error {err:.2f}px")

    def test_project_roundtrip_and_center(self):
        pts3, pts2, _, _, focal, size = synthetic_scene()
        pose, _ = registration.solve_pose(pts3, pts2, size, focal_px=focal)
        np.testing.assert_allclose(pose.project(pts3), pts2, atol=0.5)
        # Camera center must reproject all points consistently:
        # |R c + t| == 0 by definition of the center.
        residual = pose.rotation @ pose.camera_center + pose.tvec
        np.testing.assert_allclose(residual, 0.0, atol=1e-9)

    def test_save_load(self):
        import tempfile
        import os
        pts3, pts2, _, _, focal, size = synthetic_scene()
        pose, _ = registration.solve_pose(pts3, pts2, size, focal_px=focal)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "pose.json")
            pose.save(path)
            loaded = registration.CameraPose.load(path)
        np.testing.assert_allclose(loaded.rvec, pose.rvec)
        np.testing.assert_allclose(loaded.camera_matrix, pose.camera_matrix)

    def test_too_few_points(self):
        pts3, pts2, _, _, focal, size = synthetic_scene(n=12)
        with self.assertRaises(ValueError):
            registration.solve_pose(pts3[:3], pts2[:3], size, focal_px=focal)
        with self.assertRaises(ValueError):
            registration.solve_pose(pts3[:5], pts2[:5], size)


if __name__ == "__main__":
    unittest.main()
