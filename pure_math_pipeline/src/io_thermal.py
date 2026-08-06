"""
io_thermal.py
=============
Turns a DJI R-JPEG thermal capture into a single-channel 2D numpy field that
the pure-math detectors in this package can operate on.

Three tiers, tried in order, so the pipeline always produces *something*
useful even before the official DJI SDK is wired up:

  1. TRUE CELSIUS via the official DJI Thermal SDK (libdirp).
     -> `dirp_measure_ex` returns one float32 per pixel = real degrees C.
  2. TRUE CELSIUS via a DJI Thermal Analysis Tool (DTAT) CSV export sitting
     next to the image (DTAT is DJI's free desktop app; exporting a CSV
     needs no compiling / no ctypes at all).
  3. PROXY field recovered from the pseudo-color display JPEG itself, via
     PCA colormap inversion (see `palette_invert`). Monotonic with
     temperature but NOT calibrated Celsius -- every function that returns
     this tier sets `calibrated=False` so callers/reports can be honest
     about it.

Nothing here is machine learning. Tier 1/2 are just reading numbers DJI's
sensor already computed; tier 3 is linear algebra (SVD/PCA).
"""
from __future__ import annotations

import csv
import ctypes
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import zoom as _ndi_zoom


@dataclass
class ThermalField:
    field: np.ndarray          # float32, shape (H, W)
    calibrated: bool           # True => field is real Celsius
    source: str                # human-readable provenance string
    units: str                 # "C" or "proxy[0,1]"


# ---------------------------------------------------------------------------
# Tier 1: official DJI Thermal SDK (libdirp) via ctypes
# ---------------------------------------------------------------------------
# Setup (one-time, done by the user -- proprietary binary, we cannot ship it):
#   1. Download "DJI Thermal SDK" from
#      https://www.dji.com/downloads/softwares/dji-thermal-sdk  (free, no
#      login required beyond DJI's standard download click-through).
#   2. Unzip it; find `linux/release_x64/libdirp.so` (or `windows/libdirp.dll`).
#   3. `export DJI_THERMAL_SDK_LIB=/path/to/libdirp.so`
# Once that env var points at a real library, get_celsius_dji_sdk() below
# will produce exact per-pixel Celsius using DJI's own radiometric model
# (Planck-curve inversion with the camera's embedded emissivity, distance,
# humidity and reflected-temperature parameters -- this is DJI's
# calibration, not ours; we are just calling their function).

class _DirpMeasurementParams(ctypes.Structure):
    _fields_ = [
        ("distance", ctypes.c_float),
        ("humidity", ctypes.c_float),
        ("emissivity", ctypes.c_float),
        ("reflection", ctypes.c_float),
    ]


class _DirpResolution(ctypes.Structure):
    _fields_ = [("width", ctypes.c_int32), ("height", ctypes.c_int32)]


def get_celsius_dji_sdk(rjpeg_path: str, lib_path: str | None = None) -> ThermalField | None:
    """Return a calibrated Celsius ThermalField using the official DJI SDK,
    or None if the SDK library isn't available (caller should fall back)."""
    lib_path = lib_path or os.environ.get("DJI_THERMAL_SDK_LIB")
    if not lib_path or not os.path.exists(lib_path):
        return None

    lib = ctypes.CDLL(lib_path)
    data = Path(rjpeg_path).read_bytes()
    buf = ctypes.create_string_buffer(data, len(data))
    handle = ctypes.c_void_p()

    ret = lib.dirp_create_from_rjpeg(buf, len(data), ctypes.byref(handle))
    if ret != 0:
        return None
    try:
        res = _DirpResolution()
        lib.dirp_get_rjpeg_resolution(handle, ctypes.byref(res))
        w, h = res.width, res.height
        n = w * h
        temp_buf = (ctypes.c_float * n)()
        ret = lib.dirp_measure_ex(handle, temp_buf, ctypes.sizeof(temp_buf))
        if ret != 0:
            return None
        arr = np.ctypeslib.as_array(temp_buf).reshape(h, w).astype(np.float32).copy()

        # DJI's raw radiometric frame is the native sensor resolution (e.g.
        # 640x512), while the pseudo-color display JPEG (and any labelme
        # ground-truth polygons drawn on it) is upsampled 2x by the camera's
        # ISP. Resample the Celsius grid onto the display-image pixel grid
        # (bilinear, order=1 -- physically reasonable for a smoothly varying
        # temperature field) so every detector/overlay/GT-comparison in this
        # package can assume "field.shape == display image shape" universally.
        with Image.open(rjpeg_path) as disp:
            disp_w, disp_h = disp.size
        if (h, w) != (disp_h, disp_w):
            arr = _ndi_zoom(arr, (disp_h / h, disp_w / w), order=1).astype(np.float32)

        return ThermalField(field=arr, calibrated=True, source="dji_thermal_sdk", units="C")
    finally:
        lib.dirp_destroy(handle)


# ---------------------------------------------------------------------------
# Tier 2: DJI Thermal Analysis Tool CSV export sitting next to the image
# ---------------------------------------------------------------------------
def get_celsius_dtat_csv(rjpeg_path: str) -> ThermalField | None:
    """Looks for <same_basename>.csv (DTAT's raw temperature-matrix export)
    next to the R-JPEG and loads it if present."""
    csv_path = Path(rjpeg_path).with_suffix(".csv")
    if not csv_path.exists():
        return None
    with open(csv_path, newline="") as f:
        rows = [row for row in csv.reader(f) if row]
    arr = np.array(rows, dtype=np.float32)
    return ThermalField(field=arr, calibrated=True, source="dtat_csv", units="C")


# ---------------------------------------------------------------------------
# Tier 3: PCA colormap inversion of the pseudo-color display JPEG
# ---------------------------------------------------------------------------
# DJI's on-camera ISP renders the raw radiometric frame through a 1D palette
# (white-hot, iron-red, ...): temperature -> RGB is a smooth curve embedded
# in the 3D RGB cube. Because it's a curve (1 degree of freedom) and not a
# volume, the pixel cloud in RGB space is very nearly rank-1 after removing
# its mean -- i.e. it lives close to a line. The first principal component
# of the pixel colors (via SVD) is therefore, up to sign and affine
# rescaling, a recovery of the original scalar the palette encodes.
# This is classical linear algebra (PCA = truncated SVD), no learning
# involved, and it adapts automatically to whichever palette the camera
# used -- no hardcoded color lookup table required.
def palette_invert(rgb: np.ndarray) -> np.ndarray:
    """rgb: (H, W, 3) uint8 image. Returns float32 (H, W) in [0, 1],
    monotonically increasing with apparent temperature (hot = 1)."""
    h, w, _ = rgb.shape
    x = rgb.reshape(-1, 3).astype(np.float64)
    mu = x.mean(axis=0)
    xc = x - mu

    # Truncated SVD: first right-singular vector = dominant color axis.
    # (economy SVD on a subsample for speed on large images)
    n = xc.shape[0]
    if n > 200_000:
        idx = np.random.default_rng(0).choice(n, size=200_000, replace=False)
        sample = xc[idx]
    else:
        sample = xc
    _, _, vt = np.linalg.svd(sample, full_matrices=False)
    axis = vt[0]

    proj = xc @ axis  # scalar coordinate along the palette curve

    # Fix sign so that "hot" (yellow/white in DJI's default ironbow-like
    # palettes) scores high: hot colors have high R and high G; cold colors
    # (deep purple/blue) have high B relative to R+G. Use correlation with
    # (R+G-B) as an orientation reference -- still no learned parameters,
    # just a sign check.
    heuristic = x[:, 0] + x[:, 1] - x[:, 2]
    if np.corrcoef(proj, heuristic)[0, 1] < 0:
        proj = -proj

    proj = proj.reshape(h, w)
    lo, hi = np.percentile(proj, [0.5, 99.5])
    out = np.clip((proj - lo) / max(hi - lo, 1e-6), 0, 1)
    return out.astype(np.float32)


def get_proxy_field(rjpeg_path: str) -> ThermalField:
    rgb = np.array(Image.open(rjpeg_path).convert("RGB"))
    field = palette_invert(rgb)
    return ThermalField(field=field, calibrated=False, source="palette_pca_proxy", units="proxy[0,1]")


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def load_thermal_field(rjpeg_path: str, lib_path: str | None = None) -> ThermalField:
    """Try Celsius sources first, fall back to the uncalibrated proxy."""
    for getter in (
        lambda: get_celsius_dji_sdk(rjpeg_path, lib_path),
        lambda: get_celsius_dtat_csv(rjpeg_path),
    ):
        result = getter()
        if result is not None:
            return result
    return get_proxy_field(rjpeg_path)
