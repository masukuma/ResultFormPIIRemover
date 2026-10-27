"""Per-page and per-document redaction.

Fail-closed: a page is only released if it aligned confidently to a known
template, its tables were placed against their printed rules (or the zones
widened), no mask touches a protected results region, and the leak check
finds nothing left readable. Anything else goes to human review.

Masks are drawn on a separate canvas, snapped to the 8x8 block grid that the
JPEG proof works on (proof.py), and only then applied. The page result
carries that block grid; nothing else about the image is ever changed.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from . import masking, proof, structure
from .align import AlignSettings, best_alignment
from .masking import MaskSettings
from .templates import Template, Zone


@dataclass
class PageResult:
    ok: bool
    image: np.ndarray | None  # redacted page (pixels), only set when ok
    template: str
    inliers: int
    score: float
    reason: str = ""
    ms: float = 0.0
    masks: list[dict] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    blocks: np.ndarray | None = None  # 8x8 blocks blacked out
    protected: list[dict] = field(default_factory=list)  # results regions, as image polygons
    H: np.ndarray | None = None  # template -> image (page alignment)


@dataclass
class DocResult:
    pages: list[PageResult] = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return bool(self.pages) and all(p.ok for p in self.pages)


def _apply(img: np.ndarray, canvas: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Canvas (0 = hide) -> block grid, and the image with those blocks black."""
    blocks = proof.to_blocks(canvas == 0)
    out = img.copy()
    out[proof.to_pixels(blocks, img.shape[:2])] = 0
    return blocks, out


def redact_page(img: np.ndarray, templates: list[Template], s: AlignSettings,
                ms: MaskSettings | None = None, rng: random.Random | None = None,
                form: str | None = None, strict: bool = False,
                fallback_margin: tuple[float, float] = (50.0, 200.0), verify: bool = True) -> PageResult:
    t0 = time.perf_counter()
    ms = ms or MaskSettings()
    rng = rng or random.SystemRandom()  # unpredictable: hidden digits differ per cell and per run
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    a = best_alignment(gray, templates, s, form)

    def fail(reason, report=(), notes=()):
        return PageResult(False, None, a.template.name if a.template else "", a.inliers, a.score, reason,
                          (time.perf_counter() - t0) * 1000, list(report), list(notes))

    if not a.ok:
        return fail(a.reason)
    # Zones tied to a printed table are positioned by its rules. If they can't
    # be matched, the zones fall back to the feature-based fit but widened by
    # ``fallback_margin`` (px above, px below; more than the worst drift seen
    # on bent pages, which is downward: the fit is extrapolated from features
    # in the header), or with ``strict`` the page fails.
    snaps = {n: structure.snap(gray, a.H, r) for n, r in a.template.regions.items()}
    failed = {n: sn.reason for n, sn in snaps.items() if not sn.ok}
    if failed and strict:
        return fail("; ".join(failed.values()))
    notes = [f"{why}; zones widened by {fallback_margin[0]:.0f}px up, {fallback_margin[1]:.0f}px down"
             for why in failed.values()]

    def margin(zones) -> tuple[float, float]:
        return fallback_margin if zones[0].region in failed else (0.0, 0.0)

    def mapper(zones) -> masking.Mapper:
        if zones[0].region and zones[0].region not in failed:
            return snaps[zones[0].region].map
        x0 = min(z.rect[0] for z in zones)
        x1 = max(z.rect[0] + z.rect[2] for z in zones)
        cy = float(np.mean([z.rect[1] + z.rect[3] / 2 for z in zones]))
        return masking.homography(a.local(((x0 + x1) / 2, cy)))

    columns: dict[str, list] = {}  # ID / phone columns
    signatures: dict[str, list] = {}  # signature boxes and columns
    for z in a.template.zones:
        if z.action == "partial":
            columns.setdefault(z.group or z.name, []).append(z)
        elif z.action == "signature":
            signatures.setdefault(z.group or z.name, []).append(z)
    # The masked columns must sit between their printed rules (catches a table
    # matched one rule off, which the mask-vs-mapping leak check can't see).
    misplaced = [r for cells in columns.values()
                 if verify and (r := masking.check_column_edges(gray, mapper(cells), cells,
                                                                require=cells[0].region in failed))]
    if misplaced:
        return fail("; ".join(misplaced), notes=notes)

    canvas = np.full(gray.shape, 255, np.uint8)  # 0 = hide
    report = []
    for z in a.template.zones:
        if z.action == "blackout":
            up, down = margin([z])
            zz = Zone(z.name, (z.rect[0], z.rect[1] - up, z.rect[2], z.rect[3] + up + down)) if up or down else z
            masking.blackout(canvas, mapper([z]), zz, a.template.padding)
            report.append({"zone": z.name, "mode": "blackout"})
    for cells in signatures.values():
        report += masking.signature_column(canvas, gray, mapper(cells), cells, ms, margin(cells), a.template.image)
    for cells in columns.values():
        report += masking.partial_column(canvas, gray, mapper(cells), cells, ms, rng, margin(cells), a.template.image)
    blocks, out = _apply(img, canvas)

    # Results regions (votes, totals, station codes, QR) must not lose a single
    # block: a mask there would hide a result, which no PII rule needs.
    protected, prot_mask = [], np.zeros(gray.shape, np.uint8)
    for name, (x, y, w, h) in a.template.protect.items():
        poly = masking.homography(a.H)(np.float64([[x, y], [x + w, y], [x + w, y + h], [x, y + h]]))
        cv2.fillPoly(prot_mask, [np.round(poly).astype(np.int32)], 255)
        protected.append({"name": name, "polygon": np.round(poly, 1).tolist()})
    if (blocks & proof.to_blocks(prot_mask)).any():
        return fail("a mask would touch a protected results region", report, notes)

    # Final gate: re-read every masked column from the output.
    leaks = [r for cells in columns.values()
             if verify and (r := masking.verify_column(out, mapper(cells), cells, a.template.image, ms))]
    leaks += [r for cells in signatures.values()
              if verify and (r := masking.verify_signatures(gray, out, mapper(cells), cells, a.template.image, ms,
                                                            margin(cells)))]
    if leaks:
        return fail("leak check: " + "; ".join(leaks), report, notes)
    return PageResult(True, out, a.template.name, a.inliers, a.score, "",
                      (time.perf_counter() - t0) * 1000, report, notes, blocks, protected, a.H)


def redact_pages(pages, templates: list[Template], s: AlignSettings, ms: MaskSettings | None = None,
                 form: str | None = None, strict: bool = False, rng: random.Random | None = None) -> DocResult:
    return DocResult([redact_page(p, templates, s, ms, rng=rng, form=form, strict=strict) for p in pages])


def redact_boxes(img: np.ndarray, boxes, ms: MaskSettings | None = None,
                 rng: random.Random | None = None, pad: int = 10) -> PageResult:
    """Apply the same masking to VLM-located boxes (no template). Signatures
    are blacked out; ID/phone boxes get partial masking."""
    t0 = time.perf_counter()
    ms = ms or MaskSettings()
    rng = rng or random.SystemRandom()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    canvas = np.full(gray.shape, 255, np.uint8)
    I = masking.homography(np.eye(3))
    report = []
    for i, b in enumerate(boxes):
        phone = b.label == "phone_number"
        z = Zone(f"vlm_{b.label}_{i}", (b.x0, b.y0, b.x1 - b.x0, b.y1 - b.y0),
                 "blackout" if b.label == "signature" else "partial",
                 10 if phone else 8, rule="" if b.label == "signature" else ("phone" if phone else "id"))
        if z.action == "blackout":
            masking.blackout(canvas, I, z, pad)
            report.append({"zone": z.name, "mode": "blackout"})
        else:
            report += masking.partial_column(canvas, gray, I, [z], ms, rng)
    blocks, out = _apply(img, canvas)
    return PageResult(True, out, "vlm", 0, 0.0, "", (time.perf_counter() - t0) * 1000, report,
                      blocks=blocks)
