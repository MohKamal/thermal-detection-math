"""
validate.py -- score fused detections against labelme-style ground-truth
polygons (the *_T.json files DJI/labelme produced alongside each image).

Metrics: pixel-level precision/recall/F1 (via mask IoU) between the union
of ground-truth polygons and the union of fused detections, plus a
per-polygon hit-rate (fraction of labeled anomalies overlapped by at
least one detection) which is the more forgiving, "did we find it at all"
number.
"""
from __future__ import annotations
import json
from dataclasses import dataclass

import numpy as np
from matplotlib.path import Path as MplPath


@dataclass
class Metrics:
    n_gt: int
    n_det: int
    gt_hit_rate: float          # fraction of GT polygons overlapped by >=1 detection
    pixel_precision: float
    pixel_recall: float
    pixel_f1: float
    pixel_iou: float


def load_gt_mask(json_path: str, shape: tuple[int, int]) -> tuple[np.ndarray, list[np.ndarray]]:
    with open(json_path) as f:
        data = json.load(f)
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w]
    pts = np.column_stack([xx.ravel(), yy.ravel()])

    mask = np.zeros((h, w), dtype=bool)
    polys = []
    for shape_def in data.get("shapes", []):
        poly_pts = np.array(shape_def["points"])
        polys.append(poly_pts)
        path = MplPath(poly_pts)
        inside = path.contains_points(pts).reshape(h, w)
        mask |= inside
    return mask, polys


def detections_to_mask(detections, shape) -> np.ndarray:
    m = np.zeros(shape, dtype=bool)
    for d in detections:
        m[d.y0:d.y1, d.x0:d.x1] |= d.mask
    return m


def evaluate(detections, gt_mask: np.ndarray, gt_polys: list[np.ndarray]) -> Metrics:
    det_mask = detections_to_mask(detections, gt_mask.shape)

    tp = np.logical_and(det_mask, gt_mask).sum()
    fp = np.logical_and(det_mask, ~gt_mask).sum()
    fn = np.logical_and(~det_mask, gt_mask).sum()
    union = np.logical_or(det_mask, gt_mask).sum()

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    iou = tp / union if union else 0.0

    hits = 0
    for poly in gt_polys:
        path = MplPath(poly)
        cx, cy = poly[:, 0].mean(), poly[:, 1].mean()
        # a GT polygon counts as "hit" if any detection overlaps its interior
        y0, y1 = max(0, int(poly[:, 1].min())), int(poly[:, 1].max()) + 1
        x0, x1 = max(0, int(poly[:, 0].min())), int(poly[:, 0].max()) + 1
        sub_det = det_mask[y0:y1, x0:x1]
        if sub_det.any():
            hits += 1

    hit_rate = hits / len(gt_polys) if gt_polys else float("nan")

    return Metrics(
        n_gt=len(gt_polys), n_det=len(detections), gt_hit_rate=hit_rate,
        pixel_precision=float(precision), pixel_recall=float(recall),
        pixel_f1=float(f1), pixel_iou=float(iou),
    )
