"""
pipeline.py -- end-to-end CLI.

    python -m src.pipeline --input KHBPRooftops --out results

For every *_T.JPG in --input:
  1. load_thermal_field()      (Celsius if SDK/CSV available, else PCA proxy)
  2. Perona-Malik denoise
  3. run every detector, fuse via consensus voting + quantum-edge gate
  4. if a matching *_T.json (labelme) exists, score against it
  5. write an annotated overlay PNG and a per-image JSON report
  6. write results/summary.md aggregating metrics across the whole folder
"""
from __future__ import annotations
import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SDK_DIR = PROJECT_ROOT / "vendor" / "dji_thermal_sdk_v1.7" / "linux" / "release_x64"
DEFAULT_SDK_LIB = DEFAULT_SDK_DIR / "libdirp.so"


def _ensure_sdk_ld_path():
    """The vendored libdirp.so dlopen()s several sibling .so files by bare
    name at runtime (libv_dirp.so, libv_girp.so, ...). The dynamic linker
    only consults LD_LIBRARY_PATH for that at *process start*, so setting
    os.environ after the interpreter is already running has no effect --
    we have to set it and then re-exec this same interpreter once, before
    anything else imports ctypes/numpy/etc."""
    if not DEFAULT_SDK_DIR.exists():
        return
    sdk_dir_str = str(DEFAULT_SDK_DIR)
    current = os.environ.get("LD_LIBRARY_PATH", "")
    if sdk_dir_str in current.split(os.pathsep):
        return
    os.environ["LD_LIBRARY_PATH"] = sdk_dir_str + (os.pathsep + current if current else "")
    os.execve(sys.executable, [sys.executable] + sys.argv, os.environ)


if __name__ == "__main__":
    _ensure_sdk_ld_path()

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(PROJECT_ROOT))

from src import io_thermal, fusion, validate
from src.detectors import pde_diffusion


def annotate(rgb: np.ndarray, detections, gt_polys, out_path: Path):
    img = Image.fromarray(rgb).convert("RGB")
    draw = ImageDraw.Draw(img)
    for poly in gt_polys:
        pts = [tuple(p) for p in poly]
        draw.polygon(pts, outline=(0, 200, 255), width=2)  # ground truth = cyan
    for d in detections:
        draw.rectangle([d.x0, d.y0, d.x1, d.y1], outline=(255, 60, 60), width=2)  # detection = red
        draw.text((d.x0, max(0, d.y0 - 10)), f"v{d.votes}", fill=(255, 255, 0))
    img.save(out_path)


def process_image(t_jpg: Path, out_dir: Path, lib_path: str | None):
    stem = t_jpg.stem
    json_path = t_jpg.with_suffix(".json")

    tf = io_thermal.load_thermal_field(str(t_jpg), lib_path=lib_path)
    field = tf.field

    denoised = pde_diffusion.perona_malik(field, n_iter=10, kappa=np.ptp(field) * 0.02 + 1e-6)

    masks, extras = fusion.run_all_detectors(field, denoised)
    detections = fusion.fuse(field, masks, extras["quantum_edge_map"], min_votes=2, min_area=8)

    report = {
        "image": t_jpg.name,
        "calibrated": tf.calibrated,
        "field_source": tf.source,
        "units": tf.units,
        "n_detections": len(detections),
        "detections": [
            {
                "bbox": [d.x0, d.y0, d.x1, d.y1],
                "centroid": [d.centroid_x, d.centroid_y],
                "area_px": d.area_px,
                "votes": d.votes,
                "methods": d.methods,
                "quantum_edge_score": d.quantum_edge_score,
                "peak_value": d.peak_value,
                "mean_value": d.mean_value,
            }
            for d in detections
        ],
    }

    gt_mask, gt_polys = None, []
    if json_path.exists():
        gt_mask, gt_polys = validate.load_gt_mask(str(json_path), field.shape)
        metrics = validate.evaluate(detections, gt_mask, gt_polys)
        report["metrics"] = metrics.__dict__

    rgb = np.array(Image.open(t_jpg).convert("RGB"))
    annotate(rgb, detections, gt_polys, out_dir / f"{stem}_annotated.png")

    with open(out_dir / f"{stem}_report.json", "w") as f:
        json.dump(report, f, indent=2)

    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="folder containing *_T.JPG (+ optional *_T.json labels)")
    ap.add_argument("--out", default="results")
    default_lib = str(DEFAULT_SDK_LIB) if DEFAULT_SDK_LIB.exists() else None
    ap.add_argument("--dji-sdk-lib", default=default_lib,
                     help="path to libdirp.so/.dll (defaults to the vendored SDK if present)")
    args = ap.parse_args()

    in_dir = Path(args.input)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    t_files = sorted(in_dir.glob("*_T.JPG")) + sorted(in_dir.glob("*_T.jpg"))
    reports = []
    for t_jpg in t_files:
        print(f"processing {t_jpg.name} ...")
        report = process_image(t_jpg, out_dir, args.dji_sdk_lib)
        reports.append(report)
        n_det = report["n_detections"]
        m = report.get("metrics")
        if m:
            print(f"  -> {n_det} detections | GT hit-rate {m['gt_hit_rate']:.2f} | "
                  f"pixel P {m['pixel_precision']:.2f} R {m['pixel_recall']:.2f} "
                  f"F1 {m['pixel_f1']:.2f} IoU {m['pixel_iou']:.2f} | calibrated={report['calibrated']}")
        else:
            print(f"  -> {n_det} detections | no ground truth | calibrated={report['calibrated']}")

    # aggregate summary
    with_metrics = [r for r in reports if "metrics" in r]
    lines = ["# Detection run summary\n"]
    lines.append(f"Images processed: {len(reports)}\n")
    if with_metrics:
        avg = lambda key: np.mean([r["metrics"][key] for r in with_metrics])
        lines.append(f"Average GT hit-rate: {avg('gt_hit_rate'):.3f}")
        lines.append(f"Average pixel precision: {avg('pixel_precision'):.3f}")
        lines.append(f"Average pixel recall: {avg('pixel_recall'):.3f}")
        lines.append(f"Average pixel F1: {avg('pixel_f1'):.3f}")
        lines.append(f"Average pixel IoU: {avg('pixel_iou'):.3f}\n")
    lines.append("| image | calibrated | detections | GT polys | hit-rate | P | R | F1 | IoU |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for r in reports:
        m = r.get("metrics")
        if m:
            lines.append(f"| {r['image']} | {r['calibrated']} | {r['n_detections']} | {m['n_gt']} | "
                          f"{m['gt_hit_rate']:.2f} | {m['pixel_precision']:.2f} | {m['pixel_recall']:.2f} | "
                          f"{m['pixel_f1']:.2f} | {m['pixel_iou']:.2f} |")
        else:
            lines.append(f"| {r['image']} | {r['calibrated']} | {r['n_detections']} | - | - | - | - | - | - |")

    (out_dir / "summary.md").write_text("\n".join(lines))
    print(f"\nWrote summary to {out_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
