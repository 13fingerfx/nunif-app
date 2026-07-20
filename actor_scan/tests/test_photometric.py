import unittest
import numpy as np

from actor_scan.core import photometric


def bumpy_height(size=128, seed=7):
    rng = np.random.default_rng(seed)
    freq = rng.normal(size=(size // 8, size // 8))
    height = np.kron(freq, np.ones((8, 8)))
    from scipy.ndimage import gaussian_filter
    return gaussian_filter(height, 4.0) * 3.0


LIGHTS = np.array([
    [0.4, 0.0, 0.9165],
    [-0.4, 0.0, 0.9165],
    [0.0, 0.4, 0.9165],
    [0.0, -0.4, 0.9165],
])


class TestPhotometricStereo(unittest.TestCase):
    def _angular_error(self, weights):
        height = bumpy_height()
        truth = photometric.normals_from_height(height)
        images = photometric.render_lambertian(height, LIGHTS, albedo=0.8)
        normals, albedo = photometric.solve_normals(images, LIGHTS,
                                                    weights=weights)
        dots = np.clip((normals.astype(np.float64) * truth).sum(axis=2),
                       -1, 1)
        # Ignore a border where np.gradient's one-sided stencil differs.
        interior = np.degrees(np.arccos(dots))[4:-4, 4:-4]
        return interior.mean(), float(np.median(albedo[albedo > 0]))

    def test_unweighted_solve(self):
        mean_err, albedo_med = self._angular_error(None)
        self.assertLess(mean_err, 2.0, f"mean angular error {mean_err:.2f}")
        self.assertAlmostEqual(albedo_med, 0.8, delta=0.05)

    def test_weighted_solve(self):
        mean_err, _ = self._angular_error("intensity")
        self.assertLess(mean_err, 2.0, f"mean angular error {mean_err:.2f}")

    def test_rejects_too_few_lights(self):
        img = np.ones((8, 8))
        with self.assertRaises(ValueError):
            photometric.solve_normals([img, img], LIGHTS[:2])


if __name__ == "__main__":
    unittest.main()
