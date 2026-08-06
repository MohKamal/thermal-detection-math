"""
statistical.py -- robust statistics thresholding.

Field: robust statistics / order statistics.

1. Median Absolute Deviation (MAD) z-score:
      z_i = 0.6745 * (x_i - median(x)) / MAD(x)
   MAD is a breakdown-point-0.5 estimator of scale (unlike the standard
   deviation, a handful of extreme hot pixels cannot drag it around), so
   thresholding on |z| > k reliably flags pixels that are statistically
   anomalous relative to the *bulk* of the roof, even when the hotspots
   themselves are outliers big enough to bias mean/std.

2. Otsu's method: finds the threshold that maximizes between-class
   variance of a 2-class partition of the intensity histogram (Otsu, 1979) --
   equivalent to minimizing within-class variance / maximizing the
   Fisher discriminant ratio between "background" and "hot" pixel
   populations. Purely a histogram optimization, no learning.
"""
from __future__ import annotations
import numpy as np


def mad_zscore_mask(field: np.ndarray, k: float = 3.5, exclude_zeros: bool = False) -> np.ndarray:
    """`exclude_zeros=True` is for zero-inflated fields (e.g. a top-hat
    response, which is *exactly* 0 on every flat/background pixel by
    construction). Computing the median/MAD reference over the whole image
    in that case measures a point-mass at zero, not the spread of the
    signal, and the resulting MAD collapses towards 0 -- which blows up
    every z-score and makes the threshold meaningless regardless of k. The
    fix is to estimate the reference distribution only from the non-zero
    (textured/candidate) support, which is what the "outlier relative to
    its own comparison population" idea actually requires here."""
    ref = field[field > 1e-9] if exclude_zeros and np.any(field > 1e-9) else field
    med = np.median(ref)
    mad = np.median(np.abs(ref - med))
    mad = max(mad, 1e-9)
    z = 0.6745 * (field - med) / mad
    return z > k


def otsu_threshold(field: np.ndarray, bins: int = 256) -> float:
    f = field.astype(np.float64)
    hist, edges = np.histogram(f, bins=bins, range=(f.min(), f.max()))
    hist = hist.astype(np.float64)
    centers = 0.5 * (edges[:-1] + edges[1:])
    total = hist.sum()
    if total == 0:
        return float(f.max())

    w0 = np.cumsum(hist)
    w1 = total - w0
    sum_all = np.sum(hist * centers)
    sum0 = np.cumsum(hist * centers)
    with np.errstate(divide="ignore", invalid="ignore"):
        mu0 = np.where(w0 > 0, sum0 / w0, 0)
        mu1 = np.where(w1 > 0, (sum_all - sum0) / w1, 0)
    between = w0 * w1 * (mu0 - mu1) ** 2
    idx = int(np.nanargmax(between))
    return float(centers[idx])


def otsu_mask(field: np.ndarray) -> np.ndarray:
    return field > otsu_threshold(field)
