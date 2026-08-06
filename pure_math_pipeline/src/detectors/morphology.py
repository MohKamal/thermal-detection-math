"""
morphology.py -- grayscale mathematical morphology.

Field: mathematical morphology (Matheron/Serra), a lattice-theoretic
treatment of images as functions ordered by <=.

White top-hat transform:
    WTH(f) = f - (f o b)
where `o` is grayscale opening (erosion then dilation) with structuring
element b. Opening with a disk of radius r removes any bright feature
that cannot contain a disk of radius r -- i.e. it estimates the slowly
varying background while erasing small bright blobs. Subtracting that
estimate from the original image isolates exactly those small bright
blobs (hotspots smaller than the structuring element) regardless of the
absolute background temperature, which is what makes this robust across
a roof with uneven baseline heating.
"""
from __future__ import annotations
import numpy as np
from scipy import ndimage as ndi


def _disk(radius: int) -> np.ndarray:
    y, x = np.ogrid[-radius:radius + 1, -radius:radius + 1]
    return (x ** 2 + y ** 2) <= radius ** 2


def white_tophat(field: np.ndarray, radius: int = 9) -> np.ndarray:
    se = _disk(radius)
    opened = ndi.grey_opening(field, footprint=se)
    return field - opened


def white_tophat_mask(field: np.ndarray, radius: int = 9, k: float = 3.0) -> np.ndarray:
    """Threshold the top-hat response using the same robust MAD z-score
    idea as statistical.py (top-hat response is ~0 over background, so a
    high quantile / MAD cut isolates genuine blobs)."""
    resp = white_tophat(field, radius=radius)
    med = np.median(resp)
    mad = max(np.median(np.abs(resp - med)), 1e-9)
    z = 0.6745 * (resp - med) / mad
    return z > k
