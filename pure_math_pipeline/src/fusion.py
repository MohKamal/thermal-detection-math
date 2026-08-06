"""
fusion.py -- consensus fusion of the independent classical detectors.

No learning: this is a weighted-majority vote (a linear combination of
binary indicator functions, thresholded) followed by connected-component
extraction, plus a topological/quantum boundary-confirmation gate. The
idea mirrors an ensemble of independent estimators in classical
statistics -- each detector exploits a different piece of mathematics
(robust statistics, morphology, differential geometry, harmonic analysis,
algebraic topology, quantum-gate simulation), so requiring agreement
across several sharply reduces the false-positive rate any single method
would have on its own, without introducing any trained/learned parameters.
"""
from __future__ import annotations
from dataclasses import dataclass, field as dc_field
import numpy as np
from scipy import ndimage as ndi

from .detectors import statistical, morphology, scale_space, wavelet, topology, quantum_edge


@dataclass
class Detection:
    y0: int
    y1: int
    x0: int
    x1: int
    mask: np.ndarray            # boolean, shape (y1-y0, x1-x0)
    votes: int
    methods: list               # names of methods that voted for this region
    quantum_edge_score: float
    centroid_y: float
    centroid_x: float
    area_px: int
    peak_value: float
    mean_value: float


def _rasterize_blobs(shape, blobs) -> np.ndarray:
    m = np.zeros(shape, dtype=bool)
    yy, xx = np.ogrid[: shape[0], : shape[1]]
    for b in blobs:
        r = max(2, b.sigma * np.sqrt(2))
        m |= (yy - b.y) ** 2 + (xx - b.x) ** 2 <= r ** 2
    return m


def _rasterize_peaks(shape, field, peaks, window=12) -> np.ndarray:
    m = np.zeros(shape, dtype=bool)
    for p in peaks:
        ys, xs, region = topology.peak_region(field, p, window=window)  # window passed in
        m[ys, xs] |= region
    return m


def run_all_detectors(field: np.ndarray, denoised: np.ndarray | None = None, tophat_radius: int = 6):
    """Runs every detector and returns the individual boolean masks plus raw
    outputs, keyed by name.

    Design note: a rooftop (or any real scene) is rarely a single uniform
    material, so a *global* statistic computed on raw temperature is
    contaminated by material-to-material baseline differences (asphalt vs.
    membrane vs. metal flashing, ...) -- those boundaries look like "outliers"
    to a naive global threshold even though they are not anomalies. The
    white top-hat transform (morphology.py) is exactly the tool for this: it
    subtracts a locally-estimated background (grayscale opening) *before*
    anything else runs, so every downstream detector -- statistical,
    scale-space, wavelet, topological -- operates on a background-normalized
    "local excess heat" field instead of the raw scene. The quantum edge
    detector is the one exception: it is used to confirm that a candidate
    region sits on a genuine sharp thermal boundary in the *original* scene,
    so it deliberately runs on the (denoised) raw field.
    """
    f = denoised if denoised is not None else field
    contrast = morphology.white_tophat(f, radius=tophat_radius)

    masks = {}
    masks["mad_zscore"] = statistical.mad_zscore_mask(contrast, k=3.5, exclude_zeros=True)
    masks["otsu"] = statistical.otsu_mask(contrast)

    blobs = scale_space.log_blobs(contrast)
    masks["scale_space_log"] = _rasterize_blobs(f.shape, blobs)

    wtmm_resp = wavelet.wtmm_response(contrast)
    # Percentile-based cut rather than "fraction of the response's own max":
    # a single dominant singularity (a sharp material edge, say) would
    # otherwise set the max and hide every genuine-but-milder hotspot edge.
    nz = wtmm_resp[wtmm_resp > 0]
    wtmm_cut = np.percentile(nz, 97) if nz.size else 1.0
    masks["wavelet_wtmm"] = wtmm_resp > wtmm_cut

    peaks = topology.significant_peaks(contrast, min_persistence=np.ptp(contrast) * 0.12, min_separation=6)
    masks["persistent_homology"] = _rasterize_peaks(f.shape, contrast, peaks)

    qmap = quantum_edge.quantum_edge_map(f)

    extras = {"log_blobs": blobs, "wtmm_response": wtmm_resp, "peaks": peaks,
              "quantum_edge_map": qmap, "contrast": contrast}
    return masks, extras


def fuse(field: np.ndarray, masks: dict, quantum_edge_map: np.ndarray, min_votes: int = 3,
         min_area: int = 6, edge_gate_percentile: float = 12.0) -> list[Detection]:
    names = list(masks.keys())
    vote_map = np.zeros(field.shape, dtype=np.int32)
    for m in masks.values():
        vote_map += m.astype(np.int32)

    consensus = vote_map >= min_votes
    labeled, n = ndi.label(consensus)

    edge_gate = np.percentile(quantum_edge_map, edge_gate_percentile)

    detections: list[Detection] = []
    for lbl in range(1, n + 1):
        comp = labeled == lbl
        area = int(comp.sum())
        if area < min_area:
            continue
        ys, xs = np.nonzero(comp)
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1

        boundary = comp & ~ndi.binary_erosion(comp)
        edge_score = float(quantum_edge_map[boundary].mean()) if boundary.any() else 0.0
        if edge_score < edge_gate:
            continue  # no confirmed sharp thermal boundary -> likely a fusion artifact

        voting_methods = [name for name in names if masks[name][comp].any()]
        vals = field[comp]
        detections.append(
            Detection(
                y0=int(y0), y1=int(y1), x0=int(x0), x1=int(x1),
                mask=comp[y0:y1, x0:x1],
                votes=int(vote_map[comp].max()),
                methods=voting_methods,
                quantum_edge_score=edge_score,
                centroid_y=float(ys.mean()), centroid_x=float(xs.mean()),
                area_px=area,
                peak_value=float(vals.max()),
                mean_value=float(vals.mean()),
            )
        )
    detections.sort(key=lambda d: -d.votes)
    return detections
