# Math behind the detectors (no machine learning anywhere)

Every detector in `src/detectors/` implements a closed-form or classical
algorithm from a distinct branch of math/physics. None of them have
trained parameters, loss functions, or learned weights. This file is the
"why", one section per method; the code docstrings carry the same
explanations next to the implementation.

## 1. Getting a temperature field: radiometry, not vision

DJI's R-JPEG format embeds two things in one file: a pseudo-color *display*
JPEG (what you'd see on the controller) and a raw radiometric frame that
only the DJI Thermal SDK (`libdirp`) can turn into calibrated Celsius,
using the sensor's own Planck-curve inversion with the emissivity,
distance, humidity and reflected-temperature parameters baked into the
capture. `src/io_thermal.py` calls that SDK directly (tier 1), falls back
to a DTAT CSV export if you have one (tier 2), and — only if neither is
available — recovers an *uncalibrated proxy* field via PCA (tier 3, see
below). Tiers 1 and 2 are exact Celsius; tier 3 is a best-effort stand-in
for development/validation before the SDK is wired up.

### PCA colormap inversion (tier 3)
A palette (white-hot, iron-red, ...) is a 1-D curve embedded in the 3-D RGB
cube: temperature -> RGB. A cloud of pixel colors sampled from such a
palette is therefore nearly rank-1 after centering. The first principal
component (leading right-singular vector of the centered pixel matrix,
via SVD) recovers the position along that curve — i.e. a scalar that is
monotonic with temperature, without hardcoding which palette DJI used.
Linear algebra, not learning.

## 2. Perona-Malik anisotropic diffusion (PDE)
`dI/dt = div(g(|∇I|) ∇I)` — a nonlinear diffusion equation (Perona &
Malik, 1990) that smooths *along* edges and stops *across* them, because
the edge-stopping function `g` decays wherever the local gradient is
large. Used as a denoiser that doesn't blur out genuine hotspot
boundaries the way Gaussian blur (the ordinary heat equation) would.

## 3. Robust statistics
- **MAD z-score**: `z = 0.6745 (x - median) / MAD`. The median and the
  Median Absolute Deviation have a 50% breakdown point, so a handful of
  very hot pixels can't drag the "normal" reference the way they would
  drag a mean/std-based z-score.
- **Otsu's method** (1979): picks the threshold that maximizes
  between-class variance of a two-class split of the intensity histogram
  — a closed-form histogram optimization, not a learned classifier.

Both run on the **top-hat contrast field** (see next section), not the raw
scene, so they measure "hotter than the local background" rather than
"hotter than the whole frame" — important on a rooftop with several
materials at different baseline temperatures.

## 4. Grayscale mathematical morphology
White top-hat transform: `f - (f ∘ b)`, where `∘` is grayscale opening
(erosion then dilation) with a disk structuring element `b`. Opening
estimates the slowly varying background (anything that can't contain the
disk gets erased); subtracting it from the original isolates small bright
features regardless of the absolute background level. This is the
background-normalization step every other detector runs on.

## 5. Scale-space blob detection (differential geometry)
Lindeberg's scale-space theory: the scale-normalized Laplacian of the
Gaussian-smoothed image, `σ² ∇²L(x,y;σ)`, has an extremum at `σ ≈ r/√2`
for a blob of physical radius `r`. Building a Laplacian-of-Gaussian stack
across a range of σ and finding joint extrema in `(x, y, σ)` gives blob
detections *and* a size estimate, derived purely from the geometry of the
smoothed surface (no learning).

## 6. Wavelet transform modulus maxima (harmonic analysis)
Mallat & Hwang (1992): with a wavelet proportional to the derivative of a
smoothing kernel (Mexican hat = second derivative of a Gaussian),
`|W_s f(x)|` is a scale-`s` gradient magnitude, and whether its local
maxima persist as `s → 0` distinguishes genuine sharp features from noise
(whose maxima decay quickly with scale). Implemented as vectorized 1D
convolutions with the Mexican-hat kernel at several scales, along rows
and columns.

## 7. Persistent homology / topographic prominence (algebraic topology)
0-dimensional persistent homology of the superlevel-set filtration
`{f ≥ t}` as `t` decreases: every local maximum is a "birth"; whenever two
components merge, the less prominent one "dies", and `persistence =
birth − death` is exactly topographic prominence (the same quantity used
to decide whether a bump on a ridge counts as its own mountain peak).
Computed in `O(N α(N))` via union-find over pixels sorted by value
(`src/detectors/topology.py`). This is a more principled version of
"local max + arbitrary threshold": it measures how much a bump stands out
*before* it gets absorbed into something bigger.

## 8. Quantum Hadamard Edge Detection (quantum information theory, simulated classically)
Yao et al., *Quantum Image Processing and Its Application to Edge
Detection*, Phys. Rev. X 7, 031041 (2017). Encode a row/column of pixel
intensities as amplitudes of a quantum state and apply a single Hadamard
gate `H = (1/√2)[[1,1],[1,-1]]` to the qubit that distinguishes
even/odd pixel index. That one unitary turns every adjacent pixel pair
`(x_2k, x_2k+1)` into a sum-and-difference pair — the "difference" output
amplitude *is* a discrete edge map, falling directly out of the linear
algebra of the gate rather than an explicit finite-difference stencil.
`src/detectors/quantum_edge.py` simulates the statevector directly with
numpy (there is no quantum hardware involved — this is exactly the kind
of classical verification you'd run before ever touching a real qubit).
Used to confirm that a fused candidate region sits on a genuine sharp
thermal boundary, gating out regions where several classical detectors
agreed but no real edge exists.

## 9. Fusion: weighted consensus
Each detector above is an independent estimator built on different math.
`src/fusion.py` takes a majority vote across them (a linear combination of
indicator functions, thresholded), extracts connected components, and
keeps only the ones whose boundary shows a confirmed thermal edge in the
quantum-edge map. Still no learning — it's an ensemble of *fixed*
classical estimators, the same logic as combining several independent
statistical tests.

## 10. Validation
`src/validate.py` parses the labelme-style ground-truth polygons that ship
with this dataset, rasterizes them, and reports pixel precision/recall/F1
/IoU plus a per-polygon "hit rate" (did we find each labeled anomaly at
all), so detector quality is measured against real human-labeled thermal
anomalies, not just eyeballed.

## Honest read on the demo results in `results/`
The current run uses **tier 1: real calibrated Celsius** from the vendored
DJI Thermal SDK (`"calibrated": true` in every report). Switching from the
PCA proxy to true radiometric data confirmed something worth knowing:
inside the labeled "thermal anomaly" polygons the mean temperature is
~12.2 C against a surrounding-roof mean of ~9.3 C with a ~2.0 C standard
deviation -- i.e. the real anomalies in this dataset are only about
1.4-1.5 sigma above the local background, a genuinely subtle signal, not
a case of "the proxy field was too noisy to see them." That is the actual
ceiling on precision for any pure-threshold/contrast approach here, ours
included.

Two concrete bugs surfaced and got fixed once real data exposed them:
- **MAD z-score on the top-hat response is zero-inflated.** A white
  top-hat is *exactly* 0 on every flat background pixel by construction,
  so on a real (low-noise) roof more than half the image sits at a single
  point mass at zero. The median-absolute-deviation reference then
  collapses toward 0, which blows up every z-score regardless of
  threshold `k`. Fix: compute the median/MAD reference only over the
  non-zero (candidate) support (`statistical.mad_zscore_mask(...,
  exclude_zeros=True)`), which is what "outlier relative to its
  comparison population" is actually supposed to mean here.
- **Relative-to-max thresholds (LoG blobs, wavelet WTMM) are fragile to a
  single dominant peak.** One very strong isolated pixel (sun glint, a
  vent, a piece of hot equipment) sets the scale for a `rel_threshold *
  max` cut and can hide every genuine-but-milder hotspot. Fix: a
  robust/percentile-based cut on each response instead of a fraction of
  its own maximum.

With those fixes, on real Celsius data: hit-rate (did we find each labeled
anomaly) is 0.99-1.00 across all four images, pixel precision averages
~0.18 and is scene-dependent (0.03-0.33) -- similar in aggregate to the
old proxy-field run, but now for an honest reason (subtle real signal)
rather than an artifact (palette/JPEG noise). Improving precision further
from here means either scene-specific context (roof material segmentation,
a clean reference bake, multi-frame averaging) or accepting this as a
recall-oriented "flag everything worth a human look" tool rather than an
autonomous one -- which is a legitimate, common way to deploy a
threshold-based detector.
