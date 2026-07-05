""" Guided detail synthesis: "if you see x data, it translates to y texture".

The halfway point between pure measurement and pure replacement (the
Samsung-moon failure mode). The observed-but-soft detail field is a
*guide*: for every patch of the target, we search an exemplar (a real
high-resolution capture, e.g. from the overlay library) for the patch
whose coarse band best matches what was actually observed there, and
transplant only the exemplar's high band. The guide's own content is
never overwritten -- output = guide + synthesized high band -- so the
subject's real mid-frequency structure (their wrinkle layout, their
pore field) decides everything the data can decide, and the exemplar
contributes only frequencies below the capture's resolving power.

Lineage: image analogies (Hertzmann et al. 2001) / texture transfer
(Efros-Freeman). Classical, no ML, no restrictive licenses.
"""
import numpy as np
import cv2
from scipy.ndimage import gaussian_filter


def split_bands(height, sigma):
    """(low, high) frequency split of a height map."""
    h = np.asarray(height, dtype=np.float64)
    low = gaussian_filter(h, sigma)
    return low, h - low


def _ramp_window(tile, overlap):
    """Blending window: flat top, linear ramps over `overlap` px, and
    nonzero at the borders so uncovered-edge normalization never
    divides by ~0."""
    ramp = np.minimum(np.arange(1, tile + 1), overlap) / float(overlap)
    profile = np.minimum(ramp, ramp[::-1])
    return np.outer(profile, profile)


def _match_costs(ex_low, patch):
    """Plain SSD between `patch` and every same-size window of ex_low.

    Deliberately NOT zero-mean: when the guide's structures are smoother
    than the tile, the local mean level IS the regime signal (inside a
    crease vs. open skin), and mean-invariant matching goes blind to it.
    Guide and exemplar are both band-limited detail heights, so their
    levels are directly comparable.
    """
    patch32 = np.ascontiguousarray(patch, dtype=np.float32)
    ex32 = np.ascontiguousarray(ex_low, dtype=np.float32)
    return cv2.matchTemplate(ex32, patch32, cv2.TM_SQDIFF)


def _grid(length, tile, step):
    positions = list(range(0, max(length - tile, 0) + 1, step))
    if positions[-1] != length - tile:
        positions.append(length - tile)
    return positions


def guided_synthesis(guide, exemplar, sigma_split=8.0, tile=32, overlap=12,
                     tolerance=0.03, strength=1.0, rng=None):
    """Synthesize missing high-frequency detail onto a guide field.

    guide: (H, W) observed detail height map (soft/limited resolution).
    exemplar: (He, We) real high-resolution height map, statistically
    similar material (same skin region tag, ideally).
    tolerance: candidates within this fraction of the cost range of the
    best match are chosen among at random (diversity vs. fidelity).
    Returns (enhanced, high_band): enhanced = guide + strength*high_band.
    """
    guide = np.asarray(guide, dtype=np.float64)
    exemplar = np.asarray(exemplar, dtype=np.float64)
    if min(exemplar.shape) < tile:
        raise ValueError("exemplar is smaller than the synthesis tile")
    if overlap >= tile:
        raise ValueError("overlap must be smaller than tile")
    if rng is None:
        rng = np.random.default_rng(0)

    ex_low, ex_high = split_bands(exemplar, sigma_split)
    guide_low = gaussian_filter(guide, sigma_split)

    h, w = guide.shape
    step = tile - overlap
    window = _ramp_window(tile, overlap)
    accum = np.zeros((h, w))
    weight = np.zeros((h, w))
    for y in _grid(h, tile, step):
        for x in _grid(w, tile, step):
            patch = guide_low[y:y + tile, x:x + tile]
            costs = _match_costs(ex_low, patch)
            lo, hi_ = float(costs.min()), float(costs.max())
            band = lo + tolerance * (hi_ - lo + 1e-12)
            candidates = np.argwhere(costs <= band)
            ey, ex = candidates[rng.integers(len(candidates))]
            accum[y:y + tile, x:x + tile] += (
                ex_high[ey:ey + tile, ex:ex + tile] * window)
            weight[y:y + tile, x:x + tile] += window
    high_band = accum / weight
    # Quilting leaks a little low-frequency seam energy; re-band-pass so
    # the "high band only" contract holds by construction and the guide
    # keeps sole authority over everything it resolved.
    high_band -= gaussian_filter(high_band, sigma_split)
    return guide + strength * high_band, high_band
