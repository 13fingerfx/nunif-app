import unittest
import numpy as np

from actor_scan.core import lightcal


def render_ball(light, cx=64.0, cy=64.0, radius=40.0, size=128,
                shininess=200.0):
    """Synthetic glossy ball under a distant light (Phong-style lobe)."""
    ys, xs = np.mgrid[0:size, 0:size].astype(np.float64)
    nx = (xs - cx) / radius
    ny = (ys - cy) / radius
    d2 = nx * nx + ny * ny
    inside = d2 < 1.0
    nz = np.sqrt(np.clip(1.0 - d2, 0.0, None))
    n = np.dstack([nx, ny, nz])
    v = np.array([0.0, 0.0, 1.0])
    r = 2.0 * (n @ v)[..., None] * n - v
    spec = np.clip((r @ light), 0.0, None) ** shininess
    img = np.where(inside, 0.05 + spec, 0.0)
    return img


class TestLightCalibration(unittest.TestCase):
    def test_recovers_known_lights(self):
        for light in [(0.3, -0.2, 0.9), (-0.5, 0.1, 0.8), (0.0, 0.6, 0.8),
                      (0.0, 0.0, 1.0)]:
            light = np.asarray(light, dtype=np.float64)
            light /= np.linalg.norm(light)
            img = render_ball(light)
            recovered = lightcal.calibrate_light(img, 64.0, 64.0, 40.0)
            angle = np.degrees(np.arccos(np.clip(recovered @ light, -1, 1)))
            self.assertLess(angle, 3.0, f"light {light}: {angle:.2f} deg off")

    def test_rejects_point_outside_sphere(self):
        with self.assertRaises(ValueError):
            lightcal.sphere_normal_at(200, 200, 64, 64, 40)


if __name__ == "__main__":
    unittest.main()
