"""
pde_diffusion.py -- Perona-Malik anisotropic diffusion.

Field: partial differential equations / nonlinear diffusion (Perona & Malik,
1990). Solves (a discretized, explicit-Euler step of):

    dI/dt = div( g(|grad I|) * grad I )

where g is a monotonically decreasing "edge-stopping" function. Unlike
Gaussian blur (isotropic diffusion, which is the heat equation dI/dt = Laplacian(I)
and smooths everywhere equally), Perona-Malik diffuses *along* edges and
stops *across* them, because g(|grad I|) -> 0 wherever the gradient is large.

Used here as a denoiser: it removes sensor/palette noise while preserving
the sharp boundary of a genuine hotspot, so downstream detectors see a
cleaner signal without their thresholds being blurred out.
"""
from __future__ import annotations
import numpy as np


def perona_malik(field: np.ndarray, n_iter: int = 12, kappa: float = 0.05, gamma: float = 0.2,
                  option: int = 1) -> np.ndarray:
    img = field.astype(np.float64).copy()
    for _ in range(n_iter):
        # forward differences in the 4 directions (north/south/east/west)
        dN = np.roll(img, 1, axis=0) - img
        dS = np.roll(img, -1, axis=0) - img
        dE = np.roll(img, -1, axis=1) - img
        dW = np.roll(img, 1, axis=1) - img

        if option == 1:
            cN = np.exp(-(dN / kappa) ** 2)
            cS = np.exp(-(dS / kappa) ** 2)
            cE = np.exp(-(dE / kappa) ** 2)
            cW = np.exp(-(dW / kappa) ** 2)
        else:
            cN = 1.0 / (1.0 + (dN / kappa) ** 2)
            cS = 1.0 / (1.0 + (dS / kappa) ** 2)
            cE = 1.0 / (1.0 + (dE / kappa) ** 2)
            cW = 1.0 / (1.0 + (dW / kappa) ** 2)

        img += gamma * (cN * dN + cS * dS + cE * dE + cW * dW)
    return img.astype(np.float32)
