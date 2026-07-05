""" Normal-field integration and detail band-passing.

A DCT-domain Poisson solve recovers the least-squares height field from
a normal map. The absolute (low-frequency) component of that
height is untrustworthy -- photometric stereo drift must never bend the
scan's macro shape -- so bandpass_height() removes it, leaving only the
mid/high-frequency detail band the scanner itself could not capture.
Heights are in pixel-relative units; physical amplitude is applied at
bake time via an explicit scale parameter.
"""
import numpy as np
from scipy.ndimage import gaussian_filter


def gradients_from_normals(normals):
    """Height-field gradients (dh/dx, dh/dy) from unit normals."""
    n = np.asarray(normals, dtype=np.float64)
    nz = np.clip(n[..., 2], 1e-6, None)
    p = -n[..., 0] / nz
    q = -n[..., 1] / nz
    return p, q


def _fc_periodic(p, q):
    """Frankot-Chellappa on already-periodic gradient fields."""
    h, w = p.shape
    wx = np.fft.fftfreq(w) * 2.0 * np.pi
    wy = np.fft.fftfreq(h) * 2.0 * np.pi
    wxg, wyg = np.meshgrid(wx, wy)
    fp = np.fft.fft2(p)
    fq = np.fft.fft2(q)
    denom = wxg ** 2 + wyg ** 2
    denom[0, 0] = 1.0
    fh = (-1j * wxg * fp - 1j * wyg * fq) / denom
    fh[0, 0] = 0.0
    return np.real(np.fft.ifft2(fh))


def integrate_gradients(p, q):
    """Least-squares height from gradients (dh/dx, dh/dy).

    Frankot-Chellappa with mirror padding: the height field is even-
    extended (so p is odd in x / even in y, q is odd in y / even in x),
    which makes the padded field genuinely periodic. Photos are not
    periodic, and running the FFT solver without this smears boundary
    error across the whole surface.
    """
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    h, w = p.shape
    pad_p = np.block([[p, -p[:, ::-1]], [p[::-1, :], -p[::-1, ::-1]]])
    pad_q = np.block([[q, q[:, ::-1]], [-q[::-1, :], -q[::-1, ::-1]]])
    return _fc_periodic(pad_p, pad_q)[:h, :w]


def height_from_normals(normals):
    p, q = gradients_from_normals(normals)
    return integrate_gradients(p, q)


def bandpass_height(height, sigma_low, sigma_high=None):
    """Keep only the detail band of a height map.

    sigma_low (px): everything smoother than this is the scan's job and
    is removed. sigma_high (px, optional): pre-smooth to suppress
    pixel-level noise below the printable scale.
    """
    h = np.asarray(height, dtype=np.float64)
    if sigma_high:
        h = gaussian_filter(h, sigma_high)
    return h - gaussian_filter(h, sigma_low)


def detail_height_from_normals(normals, sigma_low=40.0, sigma_high=None):
    """Full Phase-1 output: normals -> band-passed detail height map."""
    return bandpass_height(height_from_normals(normals), sigma_low, sigma_high)
