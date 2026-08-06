"""
topology.py -- 0-dimensional persistent homology (superlevel-set peak
persistence), a.k.a. topographic prominence.

Field: algebraic topology / Morse theory (persistent homology, Edelsbrunner
& Harer). We treat the temperature field as a Morse function on a grid and
filter its *superlevel* sets {f >= t} as t decreases from max to min. Each
new local maximum that appears is the "birth" of a connected component
(H0 generator); whenever two components merge, the younger one (the one
with the lower peak value) "dies" -- its persistence = birth - death is
exactly the topographic prominence of that peak (the same quantity
mountaineers use to decide whether a bump counts as its own peak).

This is a strictly more principled version of "local maximum + arbitrary
threshold": persistence directly measures how much a bump stands out from
its surroundings *before merging into something more prominent*, which is
precisely what distinguishes a real thermal anomaly from a noisy ripple
sitting on the shoulder of the ambient roof-temperature gradient. We
compute it in O(N alpha(N)) with a union-find over pixels sorted by value.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy import ndimage as ndi


@dataclass
class Peak:
    y: int
    x: int
    birth: float
    death: float

    @property
    def persistence(self) -> float:
        return self.birth - self.death


class _UnionFind:
    def __init__(self, n: int):
        self.parent = np.arange(n, dtype=np.int64)

    def find(self, i: int) -> int:
        root = i
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[i] != root:
            self.parent[i], i = root, self.parent[i]
        return root

    def union_to(self, child_root: int, new_root: int):
        self.parent[child_root] = new_root


def persistence_peaks(field: np.ndarray) -> list[Peak]:
    h, w = field.shape
    flat = field.ravel().astype(np.float64)
    n = flat.size
    order = np.argsort(-flat, kind="stable")

    active = np.zeros(n, dtype=bool)
    birth = np.full(n, np.nan)
    death = np.full(n, np.nan)
    uf = _UnionFind(n)


    peaks: list[Peak] = []
    global_root = None

    for idx in order:
        y, x = divmod(idx, w)
        v = flat[idx]
        active[idx] = True
        birth[idx] = v
        cur_root = idx

        for dy, dx, off in ((-1, 0, -w), (1, 0, w), (0, -1, -1), (0, 1, 1)):
            ny, nx = y + dy, x + dx
            if not (0 <= ny < h and 0 <= nx < w):
                continue
            nidx = idx + off
            if not active[nidx]:
                continue
            nroot = uf.find(nidx)
            if nroot == cur_root:
                continue
            if birth[cur_root] >= birth[nroot]:
                # nroot's peak is less prominent -> it dies now
                death[nroot] = v
                py, px = divmod(nroot, w)
                peaks.append(Peak(y=py, x=px, birth=birth[nroot], death=v))
                uf.union_to(nroot, cur_root)
            else:
                death[cur_root] = v
                py, px = divmod(cur_root, w)
                peaks.append(Peak(y=py, x=px, birth=birth[cur_root], death=v))
                uf.union_to(cur_root, nroot)
                cur_root = nroot
        global_root = cur_root

    # the single component that never dies = global maximum's basin
    if global_root is not None and np.isnan(death[global_root]):
        gy, gx = divmod(global_root, w)
        peaks.append(Peak(y=gy, x=gx, birth=birth[global_root], death=float(flat.min())))

    return peaks


def significant_peaks(field: np.ndarray, min_persistence: float, min_separation: int = 6) -> list[Peak]:
    peaks = sorted(persistence_peaks(field), key=lambda p: -p.persistence)
    kept: list[Peak] = []
    for p in peaks:
        if p.persistence < min_persistence:
            continue
        if any((p.y - k.y) ** 2 + (p.x - k.x) ** 2 < min_separation ** 2 for k in kept):
            continue
        kept.append(p)
    return kept


def peak_region(field: np.ndarray, peak: Peak, window: int = 30) -> tuple[slice, slice, np.ndarray]:
    """Local flood-fill of the peak's basin (field >= death) within a window,
    used to turn a point-peak into an (bbox, mask) blob for reporting/IoU."""
    h, w = field.shape
    y0, y1 = max(0, peak.y - window), min(h, peak.y + window + 1)
    x0, x1 = max(0, peak.x - window), min(w, peak.x + window + 1)
    sub = field[y0:y1, x0:x1]
    mask = sub >= peak.death
    labeled, _ = ndi.label(mask)
    local_label = labeled[peak.y - y0, peak.x - x0]
    region_mask = labeled == local_label
    return slice(y0, y1), slice(x0, x1), region_mask
