"""Snap zones to the printed table structure.

Feature matching places the page as a whole, but KIEMS photos are of bent,
curled booklet pages: one homography can be off by several table rows at the
bottom of the page, far from where the features are (the header, QR code and
logo). PII lives in tables at the bottom of the page.

A template can therefore name ``structure`` regions (e.g. the agents table).
For each, the printed rule lines are found in the reference once, and in each
scan (after the global warp, within a generous search window). Horizontal and
vertical rules are matched separately by a 1-D search for scale + offset. A
table's set of line spacings is distinctive even though rows repeat. Then a
homography is fitted to the rule intersections. A region that can't be matched
convincingly fails the page; it is never left at the unverified global
position.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class Line:
    pos: float  # y for horizontal lines (at x = 0), x for vertical (at y = 0)
    slope: float  # dy/dx (horizontal) or dx/dy (vertical)
    lo: float  # extent along the line
    hi: float

    def at(self, t: float) -> float:
        return self.pos + self.slope * t

    @property
    def length(self) -> float:
        return self.hi - self.lo


@dataclass
class Region:
    name: str
    rect: tuple[float, float, float, float]
    hlines: list[float] = field(default_factory=list)  # template y of horizontal rules (at region centre)
    vlines: list[float] = field(default_factory=list)  # template x of vertical rules (at region centre)
    hslope: float = 0.0  # median slope of the template's rules (reference photo isn't square-on)
    vslope: float = 0.0
    top: float = 0.0  # template y where the column rules start / end (table top / bottom border)
    bottom: float = 0.0
    # The table hangs under another with the same columns, so its column rules
    # run on up into that one: only their lower end marks this table.
    open_top: bool = False


def _segments(mask: np.ndarray, horizontal: bool) -> list[Line]:
    n, lab, st, _ = cv2.connectedComponentsWithStats(mask)
    out = []
    for i in range(1, n):
        x, y, w, h, area = st[i]
        if area < 20:
            continue
        ys, xs = np.nonzero(lab[y:y + h, x:x + w] == i)
        xs, ys = xs + x, ys + y
        if horizontal:
            a, b = np.polyfit(xs, ys, 1) if w > 1 else (0.0, float(ys.mean()))
            out.append(Line(float(b), float(a), float(x), float(x + w)))
        else:
            a, b = np.polyfit(ys, xs, 1) if h > 1 else (0.0, float(xs.mean()))
            out.append(Line(float(b), float(a), float(y), float(y + h)))
    return out


def _merge(lines: list[Line], tol: float = 4.0) -> list[Line]:
    """Join collinear pieces of a rule broken by handwriting or stamps."""
    merged: list[Line] = []
    for l in sorted(lines, key=lambda l: l.at((l.lo + l.hi) / 2)):
        mid = (l.lo + l.hi) / 2
        for m in merged:
            if abs(m.at(mid) - l.at(mid)) < tol and abs(m.slope - l.slope) < 0.02:
                total = m.length + l.length
                m.pos = (m.pos * m.length + l.pos * l.length) / total
                m.slope = (m.slope * m.length + l.slope * l.length) / total
                m.lo, m.hi = min(m.lo, l.lo), max(m.hi, l.hi)
                break
        else:
            merged.append(Line(l.pos, l.slope, l.lo, l.hi))
    return merged


def detect_lines(gray: np.ndarray, min_h: float, min_v: float) -> tuple[list[Line], list[Line]]:
    """Printed rules in a roughly axis-aligned image (template frame)."""
    c = cv2.GaussianBlur(gray, (3, 3), 0)
    bw = cv2.adaptiveThreshold(c, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 31, 8)
    hor = cv2.morphologyEx(bw, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (60, 1)))
    ver = cv2.morphologyEx(bw, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 30)))
    hs = [l for l in _merge(_segments(hor, True)) if l.length >= min_h and abs(l.slope) < 0.1]
    vs = [l for l in _merge(_segments(ver, False)) if l.length >= min_v and abs(l.slope) < 0.1]
    return hs, vs


def prepare_region(ref: np.ndarray, name: str, rect, open_top: bool = False) -> Region:
    x, y, w, h = map(int, rect)
    hs, vs = detect_lines(ref[y:y + h, x:x + w], 0.5 * w, 0.5 * h)
    return Region(name, (x, y, w, h), sorted(l.at(w / 2) + y for l in hs), sorted(l.at(h / 2) + x for l in vs),
                  float(np.median([l.slope for l in hs])) if hs else 0.0,
                  float(np.median([l.slope for l in vs])) if vs else 0.0,
                  float(np.median([l.lo for l in vs])) + y if vs else 0.0,
                  float(np.median([l.hi for l in vs])) + y if vs else 0.0, open_top)


def match_1d(expected: list[float], found: list[float], tol: float, scale=(0.75, 1.35),
             weights: list[float] | None = None, max_shift: float | None = None,
             ends: tuple[float, float, float, float] | None = None,
             ) -> tuple[float, float, list[tuple[int, int]]]:
    """Best y' = s*y + t mapping expected positions onto found ones.

    Hypotheses come from pairs of expected/found positions; each is scored by
    the (length-)weighted number of expected positions it lands near, so
    full-width rules outvote handwriting strokes. ``max_shift`` bounds how far
    the first expected position may move: rows repeat, and without an anchor
    a grid shifted by a few rows can score as well as the right one. The
    winner is then refined by tracking outward from the anchor, so gradual
    distortion (curled pages) doesn't push later rows out of tolerance.
    ``ends`` = (first, tol_first, last, tol_last) pins where the first and last
    expected positions must land, when they are known independently.
    Returns (s, t, [(expected_index, found_index), ...])."""
    E, F = np.asarray(expected, float), np.asarray(found, float)
    W = np.ones(len(F)) if weights is None else np.asarray(weights, float)
    if len(E) < 2 or len(F) < 2:
        return 1.0, 0.0, []
    best = (-1.0, np.inf, 1.0, 0.0)
    for i in range(len(E)):
        for j in range(i + 1, len(E)):
            de = E[j] - E[i]
            for k in range(len(F)):
                d = F[k + 1:] - F[k]
                sc_all = d / de
                for l in np.nonzero((sc_all >= scale[0]) & (sc_all <= scale[1]))[0]:
                    sc = sc_all[l]
                    t = F[k] - sc * E[i]
                    if max_shift is not None and abs(sc * E[0] + t - E[0]) > max_shift:
                        continue
                    if ends is not None and (abs(sc * E[0] + t - ends[0]) > ends[1]
                                             or abs(sc * E[-1] + t - ends[2]) > ends[3]):
                        continue
                    dist = np.abs((sc * E + t)[:, None] - F[None, :])
                    jn = dist.argmin(axis=1)
                    dn = dist[np.arange(len(E)), jn]
                    hit = dn < tol
                    score = float((W[jn] * hit).sum())
                    err = float(dn[hit].mean()) if hit.any() else np.inf
                    if score > best[0] or (score == best[0] and err < best[1]):
                        best = (score, err, float(sc), float(t))
    _, _, sc, t = best
    pred = sc * E + t
    # Track: re-predict each position from the nearest matched neighbour.
    match: dict[int, int] = {}

    def take(i: int, p: float) -> None:
        cand = np.abs(F - p)
        cand[list(match.values())] = np.inf
        j = int(cand.argmin())
        if cand[j] < tol:
            match[i] = j

    anchor = int(np.abs(pred[:, None] - F[None, :]).min(axis=1).argmin())
    take(anchor, pred[anchor])
    for step in (1, -1):
        last = anchor if anchor in match else None
        i = anchor + step
        while 0 <= i < len(E):
            take(i, F[match[last]] + sc * (E[i] - E[last]) if last is not None else pred[i])
            if i in match:
                last = i
            i += step
    return sc, t, sorted(match.items())


@dataclass
class Snap:
    ok: bool
    reason: str = ""
    matched_h: int = 0
    matched_v: int = 0
    rms: float = 0.0  # of a single homography over the grid (how bent the page is)
    xs: np.ndarray | None = None  # template x of matched vertical rules (sorted)
    ys: np.ndarray | None = None  # template y of matched horizontal rules (sorted)
    P: np.ndarray | None = None  # scan coords of every intersection, shape (len(ys), len(xs), 2)

    def map(self, pts: np.ndarray) -> np.ndarray:
        """Template -> scan by bilinear interpolation over the rule grid.

        Outside the grid (e.g. the officers' signatures just above the agents
        table) points are extrapolated along the whole grid's average spacing,
        not the nearest cell's: one narrow row or column extrapolated a few
        hundred pixels magnifies its measurement error."""
        pts = np.asarray(pts, np.float64).reshape(-1, 2)
        xs, ys, P = self.xs, self.ys, self.P
        x = np.clip(pts[:, 0], xs[0], xs[-1])
        y = np.clip(pts[:, 1], ys[0], ys[-1])
        j = np.clip(np.searchsorted(xs, x) - 1, 0, len(xs) - 2)
        i = np.clip(np.searchsorted(ys, y) - 1, 0, len(ys) - 2)
        u = ((x - xs[j]) / (xs[j + 1] - xs[j]))[:, None]
        v = ((y - ys[i]) / (ys[i + 1] - ys[i]))[:, None]
        out = ((1 - u) * (1 - v) * P[i, j] + u * (1 - v) * P[i, j + 1]
               + (1 - u) * v * P[i + 1, j] + u * v * P[i + 1, j + 1])
        # average per-pixel steps down and across the grid, at this x / y
        uu = ((x - xs[0]) / (xs[-1] - xs[0]))[:, None]
        vv = ((y - ys[0]) / (ys[-1] - ys[0]))[:, None]
        down = ((1 - uu) * (P[-1, 0] - P[0, 0]) + uu * (P[-1, -1] - P[0, -1])) / (ys[-1] - ys[0])
        across = ((1 - vv) * (P[0, -1] - P[0, 0]) + vv * (P[-1, -1] - P[-1, 0])) / (xs[-1] - xs[0])
        return out + (pts[:, 1] - y)[:, None] * down + (pts[:, 0] - x)[:, None] * across


def snap(gray: np.ndarray, H: np.ndarray, region: Region, margin: float = 0.5,
         min_frac_h: float = 0.6, max_rms: float = 15.0) -> Snap:
    """Refine H inside ``region`` by matching printed rules."""
    x, y, w, h = region.rect
    mx, my = 0.15 * w, max(150.0, margin * h)
    sx, sy, sw, sh = x - mx, y - my, w + 2 * mx, h + 2 * my
    O = np.float64([[1, 0, sx], [0, 1, sy], [0, 0, 1]])
    patch = cv2.warpPerspective(gray, np.linalg.inv(H @ O), (int(sw), int(sh)), flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_REPLICATE)
    hs, vs = detect_lines(patch, 0.35 * w, 0.35 * h)
    # Positions in template coordinates at the region's centre. Each segment is
    # read at its own midpoint and carried to the centre along the template's
    # slope: rules on curled pages are curves, so extending a short piece's own
    # fitted slope across the table scatters it.
    cx, cy = x + w / 2 - sx, y + h / 2 - sy
    fh = [l.at((l.lo + l.hi) / 2) - region.hslope * ((l.lo + l.hi) / 2 - cx) + sy for l in hs]
    fv = [l.at((l.lo + l.hi) / 2) - region.vslope * ((l.lo + l.hi) / 2 - cy) + sx for l in vs]
    pitch = np.median(np.diff(region.hlines)) if len(region.hlines) > 2 else 30.0

    def y_at(l: Line, xx: float) -> float:  # inside the segment its own fit, outside the template slope
        e = min(max(xx, l.lo), l.hi)
        return l.at(e) + region.hslope * (xx - e)

    def x_at(l: Line, yy: float) -> float:
        e = min(max(yy, l.lo), l.hi)
        return l.at(e) + region.vslope * (yy - e)

    # 1. Column rules: x positions, and (from where they start and end) the
    #    table's top and bottom border. Handwriting never makes long vertical
    #    lines, so these are the dependable part of the grid.
    _, _, pv = match_1d(region.vlines, fv, 12.0, (0.85, 1.18), weights=[l.length / h for l in vs], max_shift=80.0)
    if len(pv) < 3:
        return Snap(False, f"{region.name}: only {len(pv)}/{len(region.vlines)} column rules found", 0, len(pv))
    pv.sort(key=lambda p: region.vlines[p[0]])
    tops = np.float64([vs[b].lo for _, b in pv])
    bots = np.float64([vs[b].hi for _, b in pv])
    top, bot = float(np.median(tops)), float(np.median(bots))
    t_top, t_bot = region.top - sy, region.bottom - sy  # template, patch frame
    # Rules running on up past the table, with their lower ends where the
    # table ends: a table hung under another (``open_top``), or a scan's dark
    # band above the table that the rules seem to run into. The lower ends
    # place the table.
    if region.open_top or (abs(bot - t_bot) <= 20 and top < t_top - 80):
        if abs(bot - t_bot) > 80 or top > t_top + 0.25 * (t_bot - t_top):
            return Snap(False, f"{region.name}: column rules don't reach the table", 0, len(pv))
        top = bot - (t_bot - t_top)
        tops = np.full_like(tops, top)
    elif not 0.75 <= (bot - top) / (t_bot - t_top) <= 1.35 or abs(top - t_top) > 80:
        return Snap(False, f"{region.name}: column rules don't span the table", 0, len(pv))
    # a rule cut short (stamp, fold) takes the common end instead
    tops = np.where(np.abs(tops - top) > 20, top, tops)
    bots = np.where(np.abs(bots - bot) > 20, bot, bots)
    xs_t = np.float64([region.vlines[k] for k, _ in pv])

    # 2. Row rules, if they can be matched cleanly, pin every row (curled pages
    #    bend rows unevenly); otherwise rows are interpolated top to bottom.
    # Rows repeat, so a match shifted by a row or two can score as well as the
    # right one: pin the outer rules to where the column rules start and end.
    _, _, ph = match_1d(region.hlines, fh, 0.3 * pitch, weights=[l.length / w for l in hs],
                        ends=(top + sy + (region.hlines[0] - region.top), 15.0,
                              bot + sy + (region.hlines[-1] - region.bottom), 25.0))
    rows: list[tuple[float, Line | None]] = []
    if len(ph) >= max(3, min_frac_h * len(region.hlines)):
        rows = sorted(((region.hlines[i], hs[j]) for i, j in ph), key=lambda r: r[0])
    P_rows = None
    if rows:
        P_rows = np.zeros((len(rows), len(pv), 2))
        for r, (_, hl) in enumerate(rows):
            for c, (_, b) in enumerate(pv):
                xx = x_at(vs[b], y_at(hl, 0.0))
                for _ in range(3):  # fixed-point intersection
                    yy = y_at(hl, xx)
                    xx = x_at(vs[b], yy)
                P_rows[r, c] = (xx, yy)
        ys_r = np.float64([r[0] for r in rows]) - sy
        dy = np.diff(P_rows[:, :, 1], axis=0) / np.diff(ys_r)[:, None]
        if dy.min() < 0.5 or dy.max() > 2.0:
            P_rows = None  # folded: fall back to the column-rule frame
    if P_rows is not None:
        ys_t = np.float64([r[0] for r in rows])
        P = P_rows
    else:
        ys_t = np.float64([region.top, region.bottom])
        P = np.zeros((2, len(pv), 2))
        for c, (_, b) in enumerate(pv):
            P[0, c] = (x_at(vs[b], tops[c]), tops[c])
            P[1, c] = (x_at(vs[b], bots[c]), bots[c])
    P = P + np.float64([sx, sy])  # template frame of the warped scan
    dx = np.diff(P[:, :, 0], axis=1) / np.diff(xs_t)[None, :]
    if dx.min() < 0.5 or dx.max() > 2.0:
        return Snap(False, f"{region.name}: column rules out of order", len(ph), len(pv))
    src = np.stack(np.meshgrid(xs_t, ys_t), -1).reshape(-1, 2).astype(np.float32)
    G, _ = cv2.findHomography(src, P.reshape(-1, 2).astype(np.float32), 0)
    proj = cv2.perspectiveTransform(src.reshape(-1, 1, 2), G).reshape(-1, 2)
    rms = float(np.sqrt(((proj - P.reshape(-1, 2)) ** 2).sum(axis=1).mean()))
    if rms > max_rms:  # a page bent this much is more likely a wrong match
        return Snap(False, f"{region.name}: rule grid doesn't fit (rms {rms:.1f}px)", len(ph), len(pv), rms)
    P_scan = cv2.perspectiveTransform(P.reshape(-1, 1, 2), H.astype(np.float64)).reshape(P.shape)
    return Snap(True, "", len(ph) if P_rows is not None else 0, len(pv), rms, xs_t, ys_t, P_scan)
