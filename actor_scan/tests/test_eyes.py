import unittest
import numpy as np

from actor_scan.core import eyes


class TestEyeForms(unittest.TestCase):
    def test_sphere_style_is_a_sphere(self):
        m = eyes.eye_form(diameter=24.0, style="sphere")
        radii = np.linalg.norm(m.vertices, axis=1)
        np.testing.assert_allclose(radii, 12.0, atol=1e-9)
        self.assertTrue(m.is_watertight)

    def test_sculpted_has_cornea_limbus_and_dished_iris(self):
        d, bulge, recess = 24.0, 1.1, 0.45
        m = eyes.eye_form(diameter=d, style="sculpted",
                          cornea_bulge=bulge, iris_recess=recess)
        self.assertTrue(m.is_watertight)
        radii = np.linalg.norm(m.vertices, axis=1)
        r = d / 2.0
        # Corneal rim reaches full bulge height...
        self.assertGreater(radii.max(), r + 0.9 * bulge)
        # ...the iris center is dished back below the rim...
        directions = m.vertices / radii[:, None]
        pole = np.argmax(directions[:, 2])
        self.assertLess(radii[pole], radii.max() - 0.5 * recess)
        # ...and the sclera is an untouched sphere.
        theta = np.arccos(np.clip(directions[:, 2], -1, 1))
        sclera = theta > np.arcsin(11.8 / d) * 1.45 + 0.05
        np.testing.assert_allclose(radii[sclera], r, atol=1e-6)

    def test_preset_diameters_are_sane(self):
        self.assertEqual(eyes.PRESET_DIAMETERS["adult"], 24.0)
        values = list(eyes.PRESET_DIAMETERS.values())
        self.assertEqual(values, sorted(values))
        for v in values:
            self.assertTrue(15.0 < v < 30.0)

    def test_placement(self):
        m = eyes.eye_form(diameter=24.0, style="sphere")
        center = np.array([10.0, -5.0, 40.0])
        aim = np.array([1.0, 0.0, 0.0])
        placed = eyes.place_eye(m, center, aim)
        np.testing.assert_allclose(placed.vertices.mean(axis=0), center,
                                   atol=0.05)
        # The gaze pole should now sit along +x from the center.
        radii_dir = placed.vertices - center
        pole = np.argmax(radii_dir @ aim)
        np.testing.assert_allclose(radii_dir[pole] / 12.0, aim, atol=0.02)

    def test_bad_inputs(self):
        with self.assertRaises(ValueError):
            eyes.eye_form(style="cyborg")
        with self.assertRaises(ValueError):
            eyes.eye_form(diameter=24.0, iris_diameter=30.0,
                          style="sculpted")
        with self.assertRaises(ValueError):
            eyes.place_eye(eyes.eye_form(style="sphere"), (0, 0, 0),
                           aim=(0, 0, 0))


if __name__ == "__main__":
    unittest.main()
