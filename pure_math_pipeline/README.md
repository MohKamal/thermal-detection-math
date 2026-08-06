# Pure-math thermal hotspot detection (no ML)

Detects thermal anomalies in DJI R-JPEG rooftop captures using classical
math only — robust statistics, mathematical morphology, scale-space blob
detection, wavelet singularity detection, persistent homology, and a
classically-simulated quantum edge detector — fused by weighted consensus.
See `THEORY.md` for the math behind each method.

## Install

```bash
pip install -r requirements.txt
```

## Real Celsius data — already wired up

`vendor/dji_thermal_sdk_v1.7/linux/release_x64/` ships the official DJI
Thermal SDK (`libdirp.so` + its dependency libraries), DJI's own v1.7
release (2024-12-05) obtained via the community-maintained
[thermal_parser](https://github.com/SanNianYiSi/thermal_parser) project,
which redistributes DJI's freely downloadable SDK binaries. `pipeline.py`
finds it automatically (`DEFAULT_SDK_LIB`), and re-execs the Python
interpreter once at startup with `LD_LIBRARY_PATH` pointing at it --
that's required because `libdirp.so` `dlopen()`s several sibling `.so`
files by bare name at runtime, and the dynamic linker only reads
`LD_LIBRARY_PATH` at process start, not from a variable set mid-process.
No setup needed; every report in `results/` already says
`"calibrated": true` and gives real degrees Celsius.

The raw radiometric frame DJI's sensor produces is at native sensor
resolution (640x512 here) while the pseudo-color display JPEG (and the
ground-truth polygons drawn on it) is 2x upsampled by the camera's ISP.
`io_thermal.py` resamples the Celsius grid onto the display-image pixel
grid (bilinear) so every downstream detector, overlay, and GT comparison
can assume one consistent pixel grid.

**If you point `--input` at a different SDK version/platform** (e.g. a
Windows `.dll`, or a newer SDK you download yourself from
https://www.dji.com/downloads/softwares/dji-thermal-sdk), pass
`--dji-sdk-lib /path/to/libdirp.so` or set `DJI_THERMAL_SDK_LIB` to
override the vendored default.

**Alternative, if you ever need it**: DJI's free desktop app "DJI Thermal
Analysis Tool" (DTAT) can export a temperature-matrix CSV per image; drop
it next to the JPEG with the same base filename (`DJI_..._0002_T.csv`
alongside `DJI_..._0002_T.JPG`) and the pipeline uses it automatically,
no SDK required.

If for some reason neither the SDK nor a CSV is available, the pipeline
still runs, falling back to an uncalibrated proxy field recovered from the
pseudo-color JPEG via PCA colormap inversion (`THEORY.md` §1) -- every
report says `"calibrated": false` in that case.

## Run

```bash
python -m src.pipeline --input KHBPRooftops --out results
```

For each `*_T.JPG` in the input folder this writes, into `--out`:
- `<name>_annotated.png` — red boxes = fused detections, cyan outlines =
  ground truth (if a matching `*_T.json` labelme file exists next to the
  image)
- `<name>_report.json` — per-detection bbox/centroid/area/votes/scores,
  plus peak & mean temperature (Celsius if calibrated, else proxy [0,1])
- `summary.md` — aggregate precision/recall/F1/IoU/hit-rate across the
  whole folder

## Tuning

The consensus vote threshold, minimum blob area, and the quantum-edge
confirmation gate are the main knobs (`src/fusion.py: fuse()`, called from
`src/pipeline.py`). More agreeing detectors required = fewer false
positives but risk of missing faint anomalies; loosen for a
find-everything sweep, tighten for a low-false-alarm report.

## Layout

```
src/
  io_thermal.py        DJI SDK / DTAT CSV / PCA-proxy field loading
  fusion.py             consensus voting + quantum-edge gate
  validate.py           labelme ground-truth scoring
  pipeline.py            CLI entry point
  detectors/
    statistical.py       MAD z-score, Otsu
    morphology.py         white top-hat
    scale_space.py         Laplacian-of-Gaussian blob detection
    wavelet.py              wavelet transform modulus maxima
    pde_diffusion.py         Perona-Malik anisotropic diffusion
    topology.py                persistent homology (peak prominence)
    quantum_edge.py             simulated Quantum Hadamard Edge Detection
THEORY.md               the math behind every method, one section each
results/                 demo run on KHBPRooftops (proxy field — see
                          THEORY.md's honesty note on precision there)
```
