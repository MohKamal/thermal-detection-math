"""
wavelet.py -- wavelet transform modulus maxima (WTMM) singularity detection.

Field: harmonic analysis / wavelet theory (Mallat & Hwang, 1992;
Mallat & Zhong, 1992).

The continuous wavelet transform with a wavelet that is (proportional to)
the derivative of a smoothing kernel, e.g. the Mexican-hat wavelet
(second derivative of a Gaussian), satisfies

    W_s f(x) = s * d/dx (f * G_s)(x)

so |W_s f(x)| measures the local gradient magnitude of f smoothed at
scale s. Mallat & Hwang's theorem: the local Lipschitz regularity of f at
a point (how "singular" -- i.e. how sharp an edge or spike -- f is there)
is recoverable from how |W_s f| decays as s -> 0 along the maxima lines
(chains of local maxima of |W_s f(x)| tracked across scale). Points whose
modulus maxima do NOT decay as scale shrinks are genuine sharp features
(edges/spikes) rather than noise, whose maxima decay quickly.

Implementation note: rather than looping pywt.cwt() over every row/column
in Python (slow), we build the Mexican-hat kernel once per scale and
convolve the *entire* 2D array along an axis in one vectorized call
(scipy.ndimage.convolve1d) -- mathematically identical to the per-line CWT,
just computed efficiently.
"""
from __future__ import annotations
import numpy as np
import pywt
from scipy import ndimage as ndi


def _mexh_kernel(scale: float) -> np.ndarray:
    wavelet = pywt.ContinuousWavelet("mexh")
    length = max(16, int(10 * scale))
    psi, x = wavelet.wavefun(length=length)
    # resample the reference wavelet to the requested scale (standard CWT
    # discretization: psi_s(t) = (1/sqrt(s)) psi(t/s))
    span = x[-1] - x[0]
    target_span = span * scale / 1.0  # wavefun's own support already ~[-8,8]; scale controls dilation directly
    n_samples = max(8, int(len(psi) * scale / 4))
    xs = np.linspace(x[0], x[-1], n_samples)
    kernel = np.interp(xs, x, psi) / np.sqrt(scale)
    return kernel.astype(np.float64)


def wtmm_response(field: np.ndarray, scales=(2, 4, 8), min_persistent_scales: int | None = None) -> np.ndarray:
    h, w = field.shape
    f = field.astype(np.float64)
    min_persistent_scales = min_persistent_scales or max(1, len(scales) - 1)

    acc = np.zeros((h, w), dtype=np.float64)
    n_persist = np.zeros((h, w), dtype=np.int32)

    for axis in (0, 1):
        for s in scales:
            kernel = _mexh_kernel(s)
            coeffs = ndi.convolve1d(f, kernel, axis=axis, mode="reflect")
            m = np.abs(coeffs)

            footprint = np.ones((5, 1)) if axis == 0 else np.ones((1, 5))
            local_max = ndi.maximum_filter(m, footprint=footprint)
            is_max = (m == local_max) & (m > 0.05 * (m.max() + 1e-9))
            acc += np.where(is_max, m, 0.0)
            n_persist += is_max.astype(np.int32)

    persistent = n_persist >= min_persistent_scales
    resp = acc * persistent
    peak = resp.max()
    return (resp / peak if peak > 0 else resp).astype(np.float32)
