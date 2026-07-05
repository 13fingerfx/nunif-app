""" Single-photo fallback: frequency-separation pseudo-height.

When only ordinary photos exist (no multi-light capture), band-passed
luminance stands in for surface relief: a pore or crease is darker than
its surroundings, so darker-than-local-mean maps to recessed. This is
the two-decade-old photo-bump trick -- estimation, not measurement --
and it shares the output format of the photometric path so everything
downstream (bake, overlays) is identical.
"""
import numpy as np
from scipy.ndimage import gaussian_filter


def luminance(image):
    """Rec.709 luminance from an (H, W, 3) RGB or (H, W) gray array."""
    img = np.asarray(image, dtype=np.float64)
    if img.ndim == 2:
        return img
    r, g, b = img[..., 0], img[..., 1], img[..., 2]
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def detail_height_from_photo(image, sigma_low=8.0, sigma_high=None,
                             scale=1.0):
    """Pseudo detail-height map from a single photo.

    sigma_low (px): scale above which variation is treated as lighting/
    albedo rather than relief (much smaller than the photometric path's
    band limit, because low frequencies here are almost all lighting).
    Output sign: darker than local mean -> negative -> recessed.
    """
    lum = luminance(image)
    peak = lum.max()
    if peak > 0:
        lum = lum / peak
    if sigma_high:
        lum = gaussian_filter(lum, sigma_high)
    return (lum - gaussian_filter(lum, sigma_low)) * scale
