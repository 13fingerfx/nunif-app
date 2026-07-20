import unittest
import numpy as np
from scipy.ndimage import gaussian_filter, generic_filter

from actor_scan.core import synthesis


def two_regime_material(h, w, seed):
    """Synthetic 'skin': coarse field decides the micro-texture regime.

    Where the coarse field is positive the surface carries strong fine
    grain (pored skin); where negative it is nearly smooth. This encodes
    exactly the conditional structure guided synthesis must reproduce:
    x coarse data -> y micro texture.
    """
    rng = np.random.default_rng(seed)
    coarse = gaussian_filter(rng.normal(size=(h, w)), 24.0)
    coarse *= 10.0 / coarse.std()
    grain = gaussian_filter(rng.normal(size=(h, w)), 1.0)
    grain *= 1.0 / grain.std()
    amplitude = np.where(coarse > 0, 1.0, 0.05)
    return coarse + grain * amplitude, coarse


def local_std(field, size=9):
    mean = gaussian_filter(field, size / 3.0)
    var = gaussian_filter(field ** 2, size / 3.0) - mean ** 2
    return np.sqrt(np.clip(var, 0, None))


def corr(a, b):
    a = a - a.mean()
    b = b - b.mean()
    return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum()))


class TestGuidedSynthesis(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # One continuous material; left half becomes the exemplar
        # "library capture", right half is the subject we degrade.
        full, coarse = two_regime_material(256, 512, seed=9)
        cls.exemplar = full[:, :256]
        cls.truth = full[:, 256:]
        cls.truth_coarse = coarse[:, 256:]
        # The "photo" only resolved the coarse band.
        cls.guide = gaussian_filter(cls.truth, 8.0)
        cls.enhanced, cls.high = synthesis.guided_synthesis(
            cls.guide, cls.exemplar, sigma_split=8.0,
            rng=np.random.default_rng(1))

    def test_guide_content_is_preserved(self):
        """The observed data is never overwritten (no moon-pasting)."""
        low_out = gaussian_filter(self.enhanced, 8.0)
        self.assertGreater(corr(low_out, self.guide), 0.98)

    def test_synthesized_band_is_high_frequency(self):
        residual = gaussian_filter(self.high, 8.0)
        self.assertLess(np.abs(residual).mean(),
                        0.2 * np.abs(self.high).mean())

    def test_conditioning_places_texture_by_regime(self):
        """Grain must land where the guide says grainy skin exists."""
        inner = (slice(16, -16), slice(16, -16))
        detail_energy = local_std(self.high)[inner]
        regime = (self.truth_coarse > 0)[inner]
        grainy = detail_energy[regime].mean()
        smooth = detail_energy[~regime].mean()
        self.assertGreater(
            grainy, 2.0 * smooth,
            f"grainy {grainy:.3f} vs smooth {smooth:.3f}: synthesis is "
            "not conditioning on the guide")

    def test_amplitude_is_plausible(self):
        """Borrowed detail should be exemplar-scale, not exploded."""
        truth_high = self.truth - gaussian_filter(self.truth, 8.0)
        inner = (slice(16, -16), slice(16, -16))
        ratio = self.high[inner].std() / truth_high[inner].std()
        self.assertGreater(ratio, 0.5)
        self.assertLess(ratio, 1.5)

    def test_not_a_single_repeated_tile(self):
        """Diversity: distinct grainy areas must not be identical."""
        a = self.high[64:96, 32:64]
        b = self.high[160:192, 128:160]
        self.assertLess(abs(corr(a, b)), 0.9)

    def test_rejects_small_exemplar(self):
        with self.assertRaises(ValueError):
            synthesis.guided_synthesis(self.guide, self.exemplar[:32, :32],
                                       tile=48)

    def test_rejects_bad_overlap(self):
        with self.assertRaises(ValueError):
            synthesis.guided_synthesis(self.guide, self.exemplar,
                                       tile=32, overlap=32)

    def test_deterministic_with_seed(self):
        out1, _ = synthesis.guided_synthesis(
            self.guide, self.exemplar, tile=48, overlap=16,
            rng=np.random.default_rng(7))
        out2, _ = synthesis.guided_synthesis(
            self.guide, self.exemplar, tile=48, overlap=16,
            rng=np.random.default_rng(7))
        np.testing.assert_allclose(out1, out2)


if __name__ == "__main__":
    unittest.main()
