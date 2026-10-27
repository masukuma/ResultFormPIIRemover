"""Per-page and per-document redaction.

Fail-closed: a page is only released if it aligned confidently to a known
template. Anything else goes to human review and the document is withheld.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from .align import AlignSettings, best_alignment
from .templates import Template


@dataclass
class PageResult:
    ok: bool
    image: np.ndarray | None  # redacted page, only set when ok
    template: str
    inliers: int
    score: float
    reason: str = ""
    ms: float = 0.0


@dataclass
class DocResult:
    pages: list[PageResult] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return bool(self.pages) and all(p.ok for p in self.pages)


def redact_page(img: np.ndarray, templates: list[Template], s: AlignSettings) -> PageResult:
    t0 = time.perf_counter()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    a = best_alignment(gray, templates, s)
    if not a.ok:
        return PageResult(False, None, a.template.name, a.inliers, a.score, a.reason,
                          (time.perf_counter() - t0) * 1000)
    h, w = gray.shape[:2]
    # Project the padded template-space mask into the scan; nearest-neighbour keeps it binary.
    mask = cv2.warpPerspective(a.template.mask, a.H, (w, h), flags=cv2.INTER_NEAREST)
    out = img.copy()
    out[mask > 0] = 0  # solid fill: irreversible, unlike blur or pixelation
    return PageResult(True, out, a.template.name, a.inliers, a.score, "",
                      (time.perf_counter() - t0) * 1000)


def redact_pages(pages, templates: list[Template], s: AlignSettings) -> DocResult:
    return DocResult([redact_page(p, templates, s) for p in pages])
