import unittest
import os
import tempfile
import numpy as np

from actor_scan.core import overlay


def make_detail_map(size=200, seed=2):
    rng = np.random.default_rng(seed)
    from scipy.ndimage import gaussian_filter
    return gaussian_filter(rng.normal(size=(size, size)), 2.0).astype(
        np.float32)


SQUARE = [(60, 60), (140, 60), (140, 140), (60, 140)]
MARKS = {"a": (70.0, 70.0), "b": (130.0, 70.0), "c": (100.0, 130.0)}


class TestOverlay(unittest.TestCase):
    def test_extract_masks_outside_region(self):
        detail = make_detail_map()
        ov = overlay.extract_overlay(detail, SQUARE, MARKS, "l_cheek",
                                     "test", feather_px=0)
        self.assertEqual(ov.region_tag, "l_cheek")
        self.assertEqual(ov.alpha.max(), 1.0)
        # Padding ring outside the polygon must be masked out.
        self.assertEqual(ov.height[0, 0], 0.0)
        self.assertEqual(ov.alpha[0, 0], 0.0)

    def test_save_load_roundtrip(self):
        detail = make_detail_map()
        ov = overlay.extract_overlay(detail, SQUARE, MARKS, "brow", "t",
                                     feather_px=4, px_per_mm=10.0)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "brow.aso.npz")
            ov.save(path)
            loaded = overlay.Overlay.load(path)
        np.testing.assert_allclose(loaded.height, ov.height)
        np.testing.assert_allclose(loaded.alpha, ov.alpha)
        self.assertEqual(loaded.landmarks.keys(), ov.landmarks.keys())
        self.assertEqual(loaded.px_per_mm, 10.0)

    def test_identity_apply_reproduces_source(self):
        detail = make_detail_map()
        ov = overlay.extract_overlay(detail, SQUARE, MARKS, "r_cheek", "t",
                                     feather_px=0, pad=8)
        # Place landmarks exactly where they were in the source map.
        target = np.zeros_like(detail)
        out = overlay.apply_overlay(target, ov, MARKS, mode="add")
        inner = (slice(80, 120), slice(80, 120))
        np.testing.assert_allclose(out[inner], detail[inner], atol=1e-3)

    def test_translated_apply(self):
        detail = make_detail_map()
        ov = overlay.extract_overlay(detail, SQUARE, MARKS, "r_cheek", "t",
                                     feather_px=0)
        shift = 30.0
        moved = {k: (x + shift, y) for k, (x, y) in MARKS.items()}
        target = np.zeros_like(detail)
        out = overlay.apply_overlay(target, ov, moved, mode="add")
        inner_src = (slice(80, 120), slice(80, 120))
        inner_dst = (slice(80, 120), slice(80 + int(shift), 120 + int(shift)))
        np.testing.assert_allclose(out[inner_dst], detail[inner_src],
                                   atol=1e-3)

    def test_blend_mode_replaces_inside(self):
        detail = make_detail_map()
        ov = overlay.extract_overlay(detail, SQUARE, MARKS, "chin", "t",
                                     feather_px=0)
        target = np.full_like(detail, 5.0)
        out = overlay.apply_overlay(target, ov, MARKS, mode="blend")
        self.assertAlmostEqual(out[100, 100], detail[100, 100], places=2)
        self.assertEqual(out[10, 10], 5.0)

    def test_requires_three_landmarks(self):
        detail = make_detail_map()
        ov = overlay.extract_overlay(detail, SQUARE, MARKS, "x", "t")
        with self.assertRaises(ValueError):
            overlay.apply_overlay(detail, ov, {"a": (0.0, 0.0)})


if __name__ == "__main__":
    unittest.main()
