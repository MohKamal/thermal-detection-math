"""
quantum_edge.py -- Quantum Hadamard Edge Detection (QHED), classically
simulated.

Field: quantum computing / quantum information theory (Yao et al., "Quantum
Image Processing and Its Application to Edge Detection: Theory and
Experiment," Phys. Rev. X 7, 031041, 2017).

Encode a row (or column) of N=2^n pixel intensities as the amplitudes of an
n-qubit state |psi> = sum_i x_i |i> / ||x||. Apply a Hadamard gate,

    H = (1/sqrt(2)) [[1, 1], [1, -1]]

to the least-significant qubit only. Because that qubit's two basis states
correspond to even/odd pixel index within each adjacent pair, this single
unitary turns every pair (x_2k, x_2k+1) into

    ((x_2k + x_2k+1)/sqrt(2),  (x_2k - x_2k+1)/sqrt(2))

i.e. every "odd" output amplitude is exactly (up to the sqrt(2) that
unitarity requires) the discrete first difference between adjacent pixels.
That is the whole trick: a boundary/edge map falls out of a *single*
one-qubit gate rather than an explicit finite-difference stencil, because
the Hadamard transform IS a difference-and-sum operator on paired
amplitudes.

We simulate the statevector directly with numpy (no quantum hardware
involved -- this is a faithful classical simulation of the unitary, which
is exactly what you'd verify before ever touching real qubits). Running the
pairing twice, once as-is and once cyclically shifted by one pixel, and
interleaving, recovers a difference estimate at *every* adjacent-pixel
boundary rather than just every other one. Applying this along rows and
columns and combining like a gradient magnitude gives a 2D quantum edge
map, used downstream to confirm that a candidate hotspot's boundary is a
genuine sharp thermal transition and not fusion noise.
"""
from __future__ import annotations
import numpy as np

_H = np.array([[1.0, 1.0], [1.0, -1.0]]) / np.sqrt(2.0)


def _hadamard_pair_diff(x: np.ndarray) -> np.ndarray:
    """Apply the Hadamard gate to the LSB-qubit pairing of a 1D signal;
    return only the 'odd' (difference) amplitudes, one per adjacent pair."""
    n = len(x)
    if n % 2:
        x = np.append(x, x[-1])
    pairs = x.reshape(-1, 2)                 # (n/2, 2): [x_2k, x_2k+1]
    transformed = pairs @ _H.T                # unitary Hadamard on each pair
    return transformed[:, 1]                  # the "difference" amplitude


def _qhed_1d(x: np.ndarray) -> np.ndarray:
    n = len(x)
    diff_a = _hadamard_pair_diff(x)                       # boundaries (0,1),(2,3),...
    diff_b = _hadamard_pair_diff(np.roll(x, -1))           # boundaries (1,2),(3,4),...

    edge = np.zeros(n, dtype=np.float64)
    idx_a = np.arange(len(diff_a)) * 2 + 1
    idx_a = idx_a[idx_a < n]
    edge[idx_a] = np.abs(diff_a[: len(idx_a)])

    idx_b = (np.arange(len(diff_b)) * 2 + 2) % n
    edge[idx_b] = np.maximum(edge[idx_b], np.abs(diff_b))
    return edge


def quantum_edge_map(field: np.ndarray) -> np.ndarray:
    """Returns a 2D edge-strength map in [0, 1], combining the simulated
    QHED response along rows and along columns as a gradient magnitude."""
    h, w = field.shape
    horiz = np.apply_along_axis(_qhed_1d, 1, field)
    vert = np.apply_along_axis(_qhed_1d, 0, field)
    mag = np.sqrt(horiz ** 2 + vert ** 2)
    m = mag.max()
    return (mag / m if m > 0 else mag).astype(np.float32)
