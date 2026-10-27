"""Per-page and per-document redaction.

Fail-closed: a page is only released if it aligned confidently to a known
template. Anything else goes to human review and the document is withheld.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from . import masking, structure
from .align import AlignSettings, best_alignment
from .masking import MaskSettings
from .templates import Template, Zone


@dataclass
class PageResult:
    ok: bool
    image: np.ndarray | None  # redacted page, only set when ok
    template: str
    inliers: int
    score: float
    reason: str = ""
    ms: float = 0.0
    masks: list[dict] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


@dataclass
class DocResult:
    pages: list[PageResult] = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return bool(self.pages) and all(p.ok for p in self.pages)


def redact_page(img: np.ndarray, templates: list[Template], s: AlignSettings,
                ms: MaskSettings | None = None, rng: random.Random | None = None,
                form: str | None = None, strict: bool = False,
                fallback_margin: tuple[float, float] = (50.0, 200.0), verify: bool = True) -> PageResult:
    t0 = time.perf_counter()
    ms = ms or MaskSettings()
    rng = rng or random.SystemRandom()  # unpredictable: hidden digits differ per cell and per run
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    a = best_alignment(gray, templates, s, form)
    if not a.ok:
        return PageResult(False, None, a.template.name if a.template else "", a.inliers, a.score, a.reason,
                          (time.perf_counter() - t0) * 1000)
    # Zones tied to a printed table are positioned by its rules. If they can't
    # be matched, the zones fall back to the feature-based fit but widened by
    # ``fallback_margin`` (px above, px below; more than the worst drift seen
    # on bent pages, which is downward: the fit is extrapolated from features
    # in the header), or with ``strict`` the page fails.
    snaps = {n: structure.snap(gray, a.H, r) for n, r in a.template.regions.items()}
    failed = {n: sn.reason for n, sn in snaps.items() if not sn.ok}
    if failed and strict:
        return PageResult(False, None, a.template.name, a.inliers, a.score, "; ".join(failed.values()),
                          (time.perf_counter() - t0) * 1000)
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

    columns: dict[str, list] = {}
    for z in a.template.zones:
        if z.action == "partial":
            columns.setdefault(z.group or z.name, []).append(z)
    # The masked columns must sit between their printed rules (catches a table
    # matched one rule off, which the mask-vs-mapping leak check can't see).
    misplaced = [r for cells in columns.values()
                 if verify and (r := masking.check_column_edges(gray, mapper(cells), cells,
                                                                require=cells[0].region in failed))]
    if misplaced:
        return PageResult(False, None, a.template.name, a.inliers, a.score, "; ".join(misplaced),
                          (time.perf_counter() - t0) * 1000, [], notes)

    out = img.copy()
    report = []
    for z in a.template.zones:
        if z.action == "blackout":
            up, down = margin([z])
            zz = Zone(z.name, (z.rect[0], z.rect[1] - up, z.rect[2], z.rect[3] + up + down)) if up or down else z
            masking.blackout(out, mapper([z]), zz, a.template.padding)
    for cells in columns.values():
        report += masking.partial_column(out, gray, mapper(cells), cells, ms, rng, margin(cells), a.template.image)
    # Final gate: re-read every masked column from the output.
    leaks = [r for cells in columns.values()
             if verify and (r := masking.verify_column(out, mapper(cells), cells, a.template.image, ms))]
    if leaks:
        return PageResult(False, None, a.template.name, a.inliers, a.score, "leak check: " + "; ".join(leaks),
                          (time.perf_counter() - t0) * 1000, report, notes)
    return PageResult(True, out, a.template.name, a.inliers, a.score, "",
                      (time.perf_counter() - t0) * 1000, report, notes)


def redact_pages(pages, templates: list[Template], s: AlignSettings, ms: MaskSettings | None = None,
                 form: str | None = None, strict: bool = False) -> DocResult:
    return DocResult([redact_page(p, templates, s, ms, form=form, strict=strict) for p in pages])


def redact_boxes(img: np.ndarray, boxes, ms: MaskSettings | None = None,
                 rng: random.Random | None = None, pad: int = 10) -> PageResult:
    """Apply the same masking to VLM-located boxes (no template). Signatures
    are blacked out; ID/phone boxes get partial masking."""
    t0 = time.perf_counter()
    ms = ms or MaskSettings()
    rng = rng or random.SystemRandom()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    out = img.copy()
    I = masking.homography(np.eye(3))
    report = []
    for i, b in enumerate(boxes):
        z = Zone(f"vlm_{b.label}_{i}", (b.x0, b.y0, b.x1 - b.x0, b.y1 - b.y0),
                 "blackout" if b.label == "signature" else "partial",
                 8 if b.label == "id_number" else 10)
        if z.action == "blackout":
            masking.blackout(out, I, z, pad)
        else:
            report += masking.partial_column(out, gray, I, [z], ms, rng)
    return PageResult(True, out, "vlm", 0, 0.0, "", (time.perf_counter() - t0) * 1000, report)
