import unittest
import numpy as np
from scipy.ndimage import gaussian_filter

from actor_scan.core import integrate, photometric, freqsep


def smooth_random_height(size=256, sigma=6.0, seed=3):
    rng = np.random.default_rng(seed)
    return gaussian_filter(rng.normal(size=(size, size)), sigma) * 20.0


def correlation(a, b):
    a = a - a.mean()
    b = b - b.mean()
    return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum()))


class TestIntegration(unittest.TestCase):
    def test_gradient_roundtrip(self):
        height = smooth_random_height()
        gy, gx = np.gradient(height)
        recovered = integrate.integrate_gradients(gx, gy)
        c = correlation(recovered, height)
        self.assertGreater(c, 0.99, f"roundtrip correlation {c:.4f}")

    def test_normals_to_detail_height(self):
        height = smooth_random_height()
        normals = photometric.normals_from_height(height)
        detail = integrate.detail_height_from_normals(normals, sigma_low=30)
        truth = integrate.bandpass_height(height, sigma_low=30)
        c = correlation(detail[16:-16, 16:-16], truth[16:-16, 16:-16])
        self.assertGreater(c, 0.95, f"detail correlation {c:.4f}")

    def test_bandpass_removes_low_frequency(self):
        ramp = np.linspace(0, 100, 256)[None, :].repeat(256, axis=0)
        out = integrate.bandpass_height(ramp, sigma_low=20)
        self.assertLess(np.abs(out[:, 64:-64]).max(), 1.0)

    def test_end_to_end_photometric_pipeline(self):
        """Photos -> normals -> integration -> band-pass vs ground truth."""
        height = smooth_random_height(size=192, sigma=4.0, seed=11)
        lights = np.array([[0.4, 0.0, 0.9165], [-0.4, 0.0, 0.9165],
                           [0.0, 0.4, 0.9165], [0.0, -0.4, 0.9165]])
        images = photometric.render_lambertian(height, lights)
        normals, _ = photometric.solve_normals(images, lights)
        detail = integrate.detail_height_from_normals(normals, sigma_low=25)
        truth = integrate.bandpass_height(height, sigma_low=25)
        c = correlation(detail[16:-16, 16:-16], truth[16:-16, 16:-16])
        self.assertGreater(c, 0.9, f"end-to-end correlation {c:.4f}")


class TestFreqSep(unittest.TestCase):
    def test_dark_spot_becomes_recess(self):
        img = np.full((64, 64), 0.8)
        img[30:34, 30:34] = 0.2  # dark pore
        height = freqsep.detail_height_from_photo(img, sigma_low=6.0)
        self.assertLess(height[31, 31], -0.01)
        self.assertLess(abs(height[8, 8]), 1e-3)

    def test_rgb_input(self):
        rng = np.random.default_rng(0)
        img = rng.random((32, 32, 3))
        height = freqsep.detail_height_from_photo(img)
        self.assertEqual(height.shape, (32, 32))


if __name__ == "__main__":
    unittest.main()
