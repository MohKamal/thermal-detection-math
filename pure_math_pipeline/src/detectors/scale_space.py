"""
scale_space.py -- scale-normalized Laplacian-of-Gaussian blob detection.

Field: scale-space theory / differential geometry (Lindeberg, 1994-1998).

The Gaussian-blurred image L(x,y;sigma) = f * G_sigma satisfies the heat
equation dL/dt = (1/2) Laplacian(L), t=sigma^2, so sigma indexes a genuine
1-parameter family of smoothed surfaces. A blob of radius r produces an
extremum of the *scale-normalized* Laplacian

    sigma^2 * Laplacian(L(x,y;sigma))

at sigma ≈ r/sqrt(2) (this is Lindeberg's blob-detection theorem: the
normalization sigma^2 is exactly what makes the Laplacian response
scale-covariant, i.e. comparable across different sigma). So: build a
Laplacian-of-Gaussian stack across a range of sigma, and any point that is
a local extremum simultaneously in (x, y, sigma) is a blob whose physical
radius is read off directly from the sigma at which it peaked. This gives
hotspot detections *and* an estimate of each hotspot's spatial extent for
free, purely from differential geometry of the smoothed surface.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy import ndimage as ndi


@dataclass
class Blob:
    y: int
    x: int
    sigma: float
    response: float


def log_blobs(field: np.ndarray, sigmas=(2, 3, 4, 6, 8, 11, 15), rel_threshold: float = 0.15,
               nms_radius: int = 5) -> list[Blob]:
    stack = np.stack(
        [-(s ** 2) * ndi.gaussian_laplace(field, sigma=s) for s in sigmas],  # sign flip: bright blob -> positive peak
        axis=0,
    )
    max_resp = stack.max()
    if max_resp <= 0:
        return []
    # Robust threshold: a single very strong isolated peak (sun glint, vent,
    # ...) would otherwise set an unreasonably high bar via `rel_threshold *
    # max`. Combine the relative cut with a MAD-based robust z-score cut and
    # take whichever is looser, so genuine but moderate blobs still surface.
    positive = stack[stack > 0]
    med = np.median(positive) if positive.size else 0.0
    mad = np.median(np.abs(positive - med)) if positive.size else 0.0
    robust_threshold = med + 3.5 * 1.4826 * mad
    threshold = min(rel_threshold * max_resp, robust_threshold) if robust_threshold > 0 else rel_threshold * max_resp

    # 3D non-max suppression across (scale, y, x)
    local_max = ndi.maximum_filter(stack, size=(3, 2 * nms_radius + 1, 2 * nms_radius + 1))
    peaks = (stack == local_max) & (stack > threshold)

    blobs = []
    for s_idx, y, x in zip(*np.nonzero(peaks)):
        blobs.append(Blob(y=int(y), x=int(x), sigma=float(sigmas[s_idx]), response=float(stack[s_idx, y, x])))
    return blobs
