# Design: PII removal for published election result forms

## Problem

Polling-station result forms are scanned and published so anyone can check the
tallies. Their **results** (station, candidates, votes) must stay readable.
Their **declaration blocks** (names, national ID numbers, phone numbers and
signatures of presiding officers and party agents) must not be published.

Constraints:

- **Volume and deadline.** Tens of thousands of forms arrive within hours of
  polls closing, and publishing them quickly matters.
- **Inputs.** Scans and phone photos, as PDFs or images: skewed, rotated,
  perspective-distorted, unevenly lit, sometimes with an OCR text layer.
- **Fixed layout.** Each form type always prints its PII in the same boxes.
- **Asymmetric cost.** One leaked ID number is a real harm. One delayed form is
  an inconvenience.

## Approach: register to a template, mask fixed zones

Because the layout is fixed, the tool never has to *find* PII. It only has to
work out *where the form is* in the image. That gives:

- **Speed.** No OCR or NER. Per page the work is feature extraction, matching, a
  homography and one mask warp: about 75–160 ms on one CPU core.
- **Handwriting coverage.** OCR+NER misses handwritten names, scribbled IDs and
  signatures. Zone masking covers whatever is written in the box.
- **Predictable failure.** Alignment either succeeds, with measurable
  confidence, or it doesn't. It never "mostly works" in ways nobody can see.

```
scan/PDF ─► rasterize ─► grayscale ─► ORB features (downscaled)
                                         │
              templates (precomputed) ─► match ─► USAC/MAGSAC homography
                                                   │
                                     ┌── checks ───┤ inliers, inlier ratio,
                                     │             │ geometry, NCC score
                               fail  ▼             ▼ pass
                          review queue      warp padded zone mask ─► fill black
                       (never published)                    │
                                             rebuild image-only PDF, no metadata
```

### 1. Templates (`templates/<form>/template.yaml`)
Each template is a reference image of the blank form plus zones (rectangles or
polygons) in reference-pixel coordinates. Every zone gets a **padding**
(default 12 px) to absorb small alignment error and handwriting that strays
outside its box. `rfpii preview` draws the padded zones over the reference so
they can be signed off by eye. Templates are reviewed configuration, so treat
changes to them like code changes.

When several form types exist (e.g. per-station, per-constituency and
per-county tallies), each page is matched against every template and the best
passing one wins. Template keypoints are computed once per worker.

### 2. Alignment (`align.py`)
- ORB keypoints are extracted on a copy scaled to 1200 px on the long side, then
  scaled back to full resolution.
- Matching uses brute-force Hamming kNN with Lowe's ratio test.
- The homography is fitted with `USAC_MAGSAC`, which copes with rotation (180°
  included), scale and perspective from phone photos.
- Phone-photo EXIF orientation is applied when the image is loaded.

### 3. Fail-closed checks
A page is released only if **all** of these pass:

| Check | Default | Catches |
|---|---|---|
| Inlier count | ≥ 40 | wrong form, blank page, unreadable scan |
| Inlier ratio | ≥ 0.15 | ambiguous or accidental matches |
| Geometry | convex, ≥ 15% of image, opposite sides within 2× | degenerate or mirrored fits |
| Alignment score | NCC ≥ 0.35 | "valid" homography that's actually off by a row |

The **alignment score** is an independent check of the homography. The scan is
warped back into template space, and its normalized cross-correlation with the
reference is measured outside the PII zones. On synthetic scans a correct fit
scores about 0.75.

If any page in a document fails, the **whole document** is withheld. The
document gets a JSON report in `review/` and an entry in
`review/report.jsonl`. Nothing partial is ever published.

### 4. Redaction
The template-space mask, with padding already applied, is warped into scan space
with nearest-neighbour interpolation, and those pixels are set to solid black.
Solid fill, unlike blur or pixelation, cannot be reversed.

### 5. Output sanitization (`io.py`)
The output is always a *new* PDF built from the redacted pixels. It has no text
or OCR layer, no annotations, no embedded original, no XMP and no Info
metadata, and images are re-encoded without EXIF. This matters because a scan
PDF with an invisible OCR layer would otherwise leak the very names that were
blacked out. The tests cover this case.

### 6. Throughput
- One process per core, using the `spawn` start method. Forking after OpenCV
  has started its thread pool deadlocks. OpenCV is limited to one thread per
  worker.
- Measured on synthetic A4 at 1.3× scale, one core:

  | `max_dim` / `n_features` | features | match + fit + score | inliers |
  |---|---|---|---|
  | 1200 / 5000 (default) | 23 ms | 140 ms | ~200 |
  | 1000 / 3000 | 17 ms | 56 ms | ~95 |
  | 800 / 2000 | 12 ms | 26 ms | ~65 |

  The end-to-end demo ran at about 14 files/s on 4 cores with defaults,
  including JPEG decode and PDF encode. 30k forms take about 35 min on 4 cores,
  or about 4 min on 32 cores.
- Matching dominates. If more speed is needed: lower `n_features`, or swap the
  brute-force matcher for FLANN-LSH. The run is idempotent and skips existing
  outputs unless `--overwrite` is given, so it can be re-run after a crash or
  as new scans arrive.

## Residual risks and mitigations

| Risk | Mitigation |
|---|---|
| Writing outside a box (e.g. a signature across the margin) | Generous padding; choose zones covering whole declaration blocks rather than individual fields |
| Form folded or torn, so a single homography doesn't fit | NCC score drops and the form goes to review. Optional extension: local refinement (ECC per zone) |
| New or revised form version | Low inliers or score, so it goes to review. Add a template and re-run |
| PII written in a non-PII area (e.g. a note in the results table) | Not covered by zones. Optional second pass: OCR the unredacted area and flag ID/phone-number patterns for review |
| Template zones drawn wrong | `rfpii preview`, plus a sample audit of the first N outputs per template before bulk publishing |
| Thresholds tuned on synthetic data | Calibrate `min_inliers` and `min_score` on a few hundred real scans; log their distributions from `report.jsonl` |

## Operational flow
1. Build a template per form type: scan a blank form, define zones, run
   `rfpii preview`, get sign-off.
2. Calibrate on a pilot batch of real scans and review the score distribution.
3. Bulk run:
   `rfpii redact incoming/ --templates templates/ --out public/ --review review/`
4. Publish only `public/`. A person clears the review queue, either by fixing
   the template or by redacting by hand.
5. Spot-audit a random sample of published files.
