# Results — plain-language summary

## Verdict

**Partial success.** It reliably finds real thermal hotspots (99.7% of labeled anomalies caught on average). It is not accurate enough to run unattended — for every real hotspot it flags, it also flags several things that aren't hot. Rating: **viable as a "flag it for a human to check" tool, not viable as a fully automatic detector.**

## What it is

A hotspot detector for DJI drone thermal photos that uses only classical math — no machine learning, no training data. It reads real temperature values (°C) from DJI's own SDK, then looks for hot spots using five independent math techniques, and only reports a spot if several of them agree.

## Results (4 test images, real Celsius data)

| Image | Real hotspots (labeled) | Found by our tool | Precision | Recall |
|---|---|---|---|---|
| 0002 | 99 | 99% found | 33% | 18% |
| 0003 | 168 | 100% found | 29% | 24% |
| 0004 | 44 | 100% found | 9% | 44% |
| 0005 | 6 | 100% found | 3% | 23% |
| **Average** | — | **99.7% found** | **18%** | **27%** |

Plain meaning: "found" = did it catch each real hotspot at all (yes, almost always). "Precision" = of everything it flagged, what fraction was actually a real hotspot (low — lots of false alarms). "Recall" = of all the actual hot pixels, what fraction it correctly painted (moderate).

## What worked

- Getting real temperature data out of the DJI photos — confirmed working, real °C values, not estimates.
- Finding every real hotspot — almost never missed one across all 4 test images.
- Background removal (ignoring the roof's normal temperature and only looking at "hotter than its surroundings") — this made the biggest difference in cutting down noise.
- Running fast enough to process a folder of images in under a minute each.

## What didn't work

- Telling real hotspots apart from harmless warm spots (roof seams, equipment, edges, sun glare) — too many false alarms.
- The real anomalies in this dataset are only mildly warmer than their surroundings (about 3°C), which is a genuinely hard signal to separate from normal roof texture using thresholds alone.
- Two of the five detection methods (wavelet, persistent homology) rarely contributed much once real data was used — they were tuned for noisier proxy data and stayed too conservative.

## Pipeline (input → output)

```
 INPUT
   DJI thermal photo (*_T.JPG)
        |
        v
   DJI Thermal SDK reads real temperature (°C)         [WORKED]
        |
        v
   Smooth out sensor noise (keep sharp edges)           [WORKED]
        |
        v
   Remove background heat, keep only "hotter than       [WORKED - biggest improvement]
   its surroundings"
        |
        v
   ---------------------------------------------------
   |        |          |          |               |
   Stats   Stats2   Blob shape  Wavelet      Peak
   test A  (Otsu)    test       edge test    prominence
   |        |          |          |               |     [MIXED - 2 of 5 methods
   ---------------------------------------------------    barely contributed]
        |
        v
   Keep a spot only if 2+ methods agree                 [WORKED - cut false alarms]
        |
        v
   Double-check the spot has a real sharp heat edge      [WORKED]
   (quantum-inspired edge check)
        |
        v
   OUTPUT
   Marked-up photo + list of hotspots with °C values     [too many false alarms mixed in]
```

## Bottom line

Use it to shortlist candidate hotspots for a person to review — not to auto-flag problems without a human looking. To do better than this would need either cleaner input (closer, more zoomed-in photos), scene-specific tuning per roof, or accepting more false positives as the cost of not missing real ones.
