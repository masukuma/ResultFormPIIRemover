"""Find official stamps, so that no mask ever touches one.

The round "I.E.B.C. PRESIDING OFFICER" stamp is part of what shows a form is
genuine, and it often sits on top of a signature, a date or an agent's row.
Every stamp found here becomes a protected region of the page: masks are
clipped around it (a signature crossing a stamp stays visible inside the
stamp) and the block grid is checked to leave every block that touches a stamp
unchanged. The stamps are recorded in the manifest, so `rfpii verify` can show
the same.

Detection favours recall over precision: a false stamp only means a little
less is masked, and the leak checks then send the page to review if that
leaves PII readable. A missed stamp could be masked, so there are several
cues:

- rings: circles (Hough) whose perimeter is inked nearly all the way round
  and more densely than the paper just outside;
- stamp text: OCR lines reading like "I.E.B.C." / "PRESIDING OFFICER" in
  capitals trigger a more permissive ring search around them, and protect a
  generous disc there even if no ring is found;
- near masks: before a page is released, a permissive ring search runs
  around every mask (see ``near``).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import cv2
import numpy as np

WORK = 1600  # long side of the working copy (px)
STAMP_TEXT = re.compile(r"[IL1lA|]\s*[.,]\s*E\s*[.,]?\s*B\s*[.,]?\s*C|PRESID[I1l]|OFF[I1l]C[EF]", re.I)


def is_stamp_text(text: str) -> bool:
    """A short line in capitals reading like stamp text ("I.E.B.C.",
    "PRESIDING", "OFFICER"); not the printed label "Presiding Officer"."""
    t = text.strip()
    letters = [c for c in t if c.isalpha()]
    return (bool(STAMP_TEXT.search(t)) and len(t) <= 14 and len(letters) >= 3
            and sum(c.isupper() for c in letters) >= 0.8 * len(letters))


@dataclass
class Stamp:
    cx: float  # image px
    cy: float
    r: float
    coverage: float  # share of the ring's perimeter that is inked
    how: str  # "ring" | "text" | "near-mask"

    def polygon(self, margin: float = 0.0, n: int = 64) -> np.ndarray:
        t = np.linspace(0, 2 * np.pi, n, endpoint=False)
        rr = self.r + margin
        return np.stack([self.cx + rr * np.cos(t), self.cy + rr * np.sin(t)], 1)

    def record(self) -> dict:
        return {"name": "stamp", "kind": "stamp", "circle": [round(self.cx, 1), round(self.cy, 1), round(self.r, 1)],
                "found_by": self.how, "ring_coverage": round(self.coverage, 2),
                "polygon": np.round(self.polygon(), 1).tolist()}


def _ink(gray: np.ndarray) -> np.ndarray:
    """Dark marks: print, pen and (dark) stamp ink."""
    c = cv2.GaussianBlur(gray, (3, 3), 0)
    ink = cv2.adaptiveThreshold(c, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 25, 12)
    return cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))


def _color_ink(color: np.ndarray) -> np.ndarray:
    """Coloured marks: stamp ink (violet, blue, red) differs in colour from the
    paper around it, even when it is too faint to stand out in grey. The
    green security print is left out, and so is most black print, which keeps
    rings easy to see."""
    lab8 = cv2.cvtColor(color, cv2.COLOR_BGR2LAB)
    d = lab8.astype(np.float32) - cv2.medianBlur(lab8, 31).astype(np.float32)
    dl, da, db = d[..., 0], d[..., 1], d[..., 2]
    m = (np.hypot(da, db) > 9) & (da > -2) & (dl < 5)
    return cv2.morphologyEx((m * 255).astype(np.uint8), cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))


def refine(ink: np.ndarray, cx: float, cy: float, r: float, iters: int = 4) -> tuple[float, float, float]:
    """Least-squares circle through the ink near a candidate ring, trimming
    outliers each round (Hough centres and radii are rough)."""
    h, w = ink.shape
    x0, x1 = int(max(0, cx - 1.35 * r)), int(min(w, cx + 1.35 * r + 1))
    y0, y1 = int(max(0, cy - 1.35 * r)), int(min(h, cy + 1.35 * r + 1))
    ys, xs = np.nonzero(ink[y0:y1, x0:x1])
    xs, ys = xs + x0, ys + y0
    tol = 0.2 * r
    for _ in range(iters):
        sel = np.abs(np.hypot(xs - cx, ys - cy) - r) <= tol
        if sel.sum() < 20:
            break
        X, Y = xs[sel].astype(float), ys[sel].astype(float)
        (a, b, c), *_ = np.linalg.lstsq(np.column_stack([X, Y, np.ones_like(X)]), X ** 2 + Y ** 2, rcond=None)
        ncx, ncy = a / 2, b / 2
        nr = float(np.sqrt(max(c + ncx ** 2 + ncy ** 2, 1.0)))
        if not 0.7 * r < nr < 1.4 * r or np.hypot(ncx - cx, ncy - cy) > 0.4 * r:
            break
        cx, cy, r = ncx, ncy, nr
        tol = max(2.0, tol * 0.6)
    return float(cx), float(cy), float(r)


def ring(ink: np.ndarray, cx: float, cy: float, r: float, bins: int = 180) -> tuple[float, float, float] | None:
    """(continuity, density, density just outside) of a thin ring at radius r.
    continuity: share of directions with ink on the ring; density: share of
    ring pixels inked. Directions off the image don't count (stamps cut by the
    page edge); None if too little of the ring is on the page."""
    h, w = ink.shape
    t = np.linspace(0, 2 * np.pi, bins, endpoint=False)[:, None]

    def sample(radii):
        x = np.round(cx + radii * np.cos(t)).astype(int)
        y = np.round(cy + radii * np.sin(t)).astype(int)
        ok = (x >= 0) & (x < w) & (y >= 0) & (y < h)
        return ok, ok & (ink[y.clip(0, h - 1), x.clip(0, w - 1)] > 0)

    tol = max(2.0, 0.05 * r)
    ok, hit = sample(r + np.arange(-tol, tol + 0.5, 1.0)[None, :])
    valid = ok.all(1)
    if valid.mean() < 0.4:
        return None
    ook, ohit = sample(r + np.arange(0.1 * r, 0.2 * r, 1.0)[None, :])
    ov = ook.all(1)
    out = float(ohit[ov].mean()) if ov.any() else 1.0
    return float(hit[valid].any(1).mean()), float(hit[valid].mean()), out


# (min continuity, min density, max outside/density) per ink map. Measured on
# the sample forms: real rings in dark ink have density >= 0.64 and continuity
# >= 0.84, while text, table rules and handwriting that happen to fall on a
# circle reach density 0.53 at most. Coloured ink has little else on it, so a
# faint stamp's broken ring is enough there (seen: continuity 0.48, density
# 0.28, nothing outside).
STRICT = {"dark": (0.8, 0.6, 0.4), "color": (0.45, 0.25, 0.25)}
LOOSE = {"dark": (0.7, 0.5, 0.45), "color": (0.4, 0.22, 0.3)}


def _candidates(ink: np.ndarray, r_range: tuple[float, float], param2: float) -> np.ndarray:
    blur = cv2.GaussianBlur(ink, (5, 5), 0)
    lo, hi = r_range
    found = []
    # Several radius bands: one wide Hough search is too coarse for small stamps.
    edges = np.geomspace(lo, hi, 4)
    for a, b in zip(edges, edges[1:]):
        c = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, dp=1.5, minDist=max(10, int(a)), param1=100, param2=param2,
                             minRadius=int(a), maxRadius=int(b) + 1)
        if c is not None:
            found += c[0].tolist()
    return np.float64(found).reshape(-1, 3)


def _measure(maps: dict, r_range, param2: float, offset=(0, 0)) -> list[tuple[str, float, float, float, tuple]]:
    """Every ring candidate, refined and measured: (map, cx, cy, r, (continuity,
    density, outside)). Thresholds are applied by the callers."""
    out = []
    for kind, ink in maps.items():
        fits: list[tuple[float, float, float]] = []
        for cx, cy, r in _candidates(ink, r_range, param2):
            # Hough returns near-copies of each circle; each is fitted, since
            # they converge differently and only one may pass. A stamp's
            # lettering makes circles of its own, a little inside the ring
            # and off centre, and Hough may return one of those instead of
            # the ring, so the fit also starts from a smaller and a larger
            # radius. Every distinct fit is measured; the thresholds decide.
            for k in (1.0, 0.85, 1.2):
                quick = ring(ink, cx, cy, k * r)
                if quick is None or quick[0] < 0.3:
                    continue  # nothing ring-like here even roughly
                fx, fy, fr = refine(ink, cx, cy, k * r)
                if not r_range[0] * 0.8 <= fr <= r_range[1] * 1.2:
                    continue
                if any(np.hypot(fx - a, fy - b) < 2 and abs(fr - c) < 2 for a, b, c in fits):
                    continue
                fits.append((fx, fy, fr))
                m = ring(ink, fx, fy, fr)
                if m:
                    out.append((kind, fx + offset[0], fy + offset[1], fr, m))
    return out


def _passing(measured, limits, how) -> list[Stamp]:
    out = []
    for kind, cx, cy, r, (cont, dens, outside) in measured:
        min_cont, min_dens, max_out = limits[kind]
        if cont >= min_cont and dens >= min_dens and outside <= max_out * dens:
            out.append(Stamp(cx, cy, r, cont, how))
    return out


def _rings(maps: dict, r_range, param2, limits, how, offset=(0, 0)) -> list[Stamp]:
    return _passing(_measure(maps, r_range, param2, offset), limits, how)


class Maps:
    """Working copies of a page (dark and coloured ink), shared by ``find``
    and ``near``."""

    def __init__(self, img: np.ndarray):
        self.scale, self.maps = _maps(img)


def _dedupe(stamps: list[Stamp]) -> list[Stamp]:
    """Concentric or overlapping detections of one stamp: keep the outermost
    ring (the whole stamp is protected), grown to cover the others."""
    stamps = sorted(stamps, key=lambda s: -s.r)
    out: list[Stamp] = []
    for s in stamps:
        for o in out:
            d = np.hypot(s.cx - o.cx, s.cy - o.cy)
            if d < 0.6 * max(s.r, o.r):
                o.r = max(o.r, d + s.r)
                o.coverage = max(o.coverage, s.coverage)
                break
        else:
            out.append(Stamp(s.cx, s.cy, s.r, s.coverage, s.how))
    return out


def ink_maps(img: np.ndarray) -> Maps:
    return Maps(img)


def _maps(img: np.ndarray) -> tuple[float, dict]:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    s = min(1.0, WORK / max(gray.shape))
    rs = (lambda a: cv2.resize(a, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)) if s < 1 else (lambda a: a)
    maps = {"dark": _ink(rs(gray))}
    if img.ndim == 3:
        maps["color"] = _color_ink(rs(img))
    return s, maps


def find(img: np.ndarray, r_range: tuple[float, float], text_lines=(), maps=None) -> list[Stamp]:
    """Stamps on a page. ``r_range``: plausible stamp radius in image px.
    ``text_lines``: OCR lines (ocr.Line, image coords) to look for stamp text.
    ``maps``: from ``ink_maps(img)``, to share the work with ``near``."""
    M = maps or Maps(img)
    s, maps = M.scale, M.maps
    rr = (r_range[0] * s, r_range[1] * s)
    stamps = _rings(maps, rr, 30, STRICT, "ring")
    for ln in text_lines:
        if not is_stamp_text(ln.text):
            continue
        x0, y0, x1, y1 = (v * s for v in ln.box)
        if any(np.hypot((x0 + x1) / 2 - st.cx, (y0 + y1) / 2 - st.cy) < st.r for st in stamps):
            continue
        # The text sits inside the stamp: look for its ring nearby, more permissively.
        pad = int(2.2 * rr[1])
        X0, Y0 = max(0, int(x0) - pad), max(0, int(y0) - pad)
        sub = {k: m[Y0:int(y1) + pad, X0:int(x1) + pad] for k, m in maps.items()}
        local = [t for t in _rings(sub, rr, 20, LOOSE, "text", (X0, Y0))
                 if np.hypot(t.cx - (x0 + x1) / 2, t.cy - (y0 + y1) / 2) < t.r]
        if local:
            stamps += local
        else:  # no ring found: protect a generous disc around the text
            stamps.append(Stamp((x0 + x1) / 2, (y0 + y1) / 2, max(rr[1], 0.6 * (x1 - x0)), 0.0, "text"))
    return [Stamp(t.cx / s, t.cy / s, t.r / s, t.coverage, t.how) for t in _dedupe(stamps)]


def near(img: np.ndarray, hide: np.ndarray, r_range: tuple[float, float], known: list[Stamp],
         maps=None) -> list[Stamp]:
    """Permissive ring search around everything about to be masked (``hide``:
    bool mask, image px). A stamp under a mask is the costly miss, so here a
    ring needs less support."""
    M = maps or Maps(img)
    s, maps = M.scale, M.maps
    shape = maps["dark"].shape
    hs = cv2.resize(hide.astype(np.uint8), (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    rr = (r_range[0] * s, r_range[1] * s)
    pad = int(1.2 * rr[1])
    # one window per cluster of masks (masks closer than a stamp's size share one)
    n, _, st, _ = cv2.connectedComponentsWithStats(cv2.dilate(hs, np.ones((pad, pad), np.uint8)))
    found: list[Stamp] = []
    yy, xx = np.ogrid[:shape[0], :shape[1]]
    for i in range(1, n):
        x, y, w, h, _ = st[i]
        X0, Y0 = max(0, x - pad // 2), max(0, y - pad // 2)
        sub = {k: m[Y0:y + h + pad // 2, X0:x + w + pad // 2] for k, m in maps.items()}
        for t in _rings(sub, rr, 22, LOOSE, "near-mask", (X0, Y0)):
            if not hs[(xx - t.cx) ** 2 + (yy - t.cy) ** 2 <= (t.r * 1.05) ** 2].any():
                continue  # only rings that reach the mask matter here
            if any(np.hypot(t.cx - k.cx * s, t.cy - k.cy * s) < 0.6 * k.r * s for k in known):
                continue
            found.append(t)
    return [Stamp(t.cx / s, t.cy / s, t.r / s, t.coverage, t.how) for t in _dedupe(found)]


def protect_mask(shape, stamps: list[Stamp], margin: float) -> np.ndarray:
    """uint8 mask (255 = stamp, grown by ``margin`` px)."""
    m = np.zeros(shape[:2], np.uint8)
    for st in stamps:
        cv2.fillPoly(m, [np.round(st.polygon(margin)).astype(np.int32)], 255)
    return m
