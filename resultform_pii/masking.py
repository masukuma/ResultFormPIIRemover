"""Turning aligned zones into masks.

blackout  Solid fill of the padded zone (signatures).
partial   A random subset of the handwritten characters in each cell is
          filled, the rest stays visible: enough to show that a real ID or phone
          number was written, not enough to use it.

Partial masking works on a whole table column at once, rectified to template
pixel scale so slant and perspective in the photo don't matter. Characters are
found once for the column and each is assigned to the row holding its centre,
so digits that dip into a neighbouring row are still masked as part of their
own row. Character slots come from connected components when they look like
separate glyphs, otherwise from dividing the row's ink extent into ``digits``
equal slots. A cell with ink that can't be segmented is hidden whole.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable

import cv2
import numpy as np

from .templates import Zone


@dataclass
class MaskSettings:
    hide: int = 3  # characters hidden per ID / phone number
    phone_tail: int = 6  # phone numbers: hide only within the last N digits (subscriber part);
    # 0734471897 keeps "0734" and hides 3 of "471897"
    vpad: int = 60  # px above/below a column searched for ink: handwriting strays, and
    # on photographed pages the table can be a row off; writing found anywhere in
    # this window is masked
    mask_vpad: int = 3  # px above/below the glyph/row covered by a hidden slot
    min_glyph_h: float = 0.4  # glyph must be at least this share of row height
    empty_ink: float = 0.01  # below this ink share a cell counts as empty


# Template -> scan point mapping: a homography, or a mesh that follows a
# curled page (structure.Snap.map).
Mapper = Callable[[np.ndarray], np.ndarray]


def homography(H: np.ndarray) -> Mapper:
    return lambda pts: cv2.perspectiveTransform(
        np.asarray(pts, np.float64).reshape(-1, 1, 2), H.astype(np.float64)).reshape(-1, 2)


def _fill(img: np.ndarray, f: Mapper, quad: np.ndarray, step: float = 8.0) -> None:
    """Fill a template-space quad, mapped edge by edge (densified, so a curved
    mapping stays covered)."""
    pts = []
    for a, b in zip(quad, np.roll(quad, -1, axis=0)):
        n = max(1, int(np.linalg.norm(b - a) / step))
        pts += [a + (b - a) * t for t in np.arange(n) / n]
    cv2.fillPoly(img, [np.round(f(np.float64(pts))).astype(np.int32)], 0)


def _grow(quad: np.ndarray, dy: float, dx: float = 0.0) -> np.ndarray:
    """Expand an axis-aligned quad (TL, TR, BR, BL) by dx / dy on each side."""
    return quad + np.float64([[-dx, -dy], [dx, -dy], [dx, dy], [-dx, dy]])


def blackout(img: np.ndarray, f: Mapper, zone: Zone, pad: float) -> None:
    _fill(img, f, zone.padded(pad))


def ink_mask(strip: np.ndarray, row_h: float, signature: bool = False) -> np.ndarray:
    """Binary ink of a rectified strip, with rules, strike lines and the
    security-print background removed.

    ``signature``: keep long flat pen strokes (a signature's flourish): only
    horizontal lines spanning nearly the whole strip count as printed rules,
    and the thin-stroke filter meant for digit columns is skipped."""
    # KIEMS photos range from crisp to blurry and grey-on-grey; stretch the
    # contrast (with a floor, so an empty cell's faint security pattern isn't
    # blown up into "ink") before thresholding.
    c = cv2.GaussianBlur(strip, (3, 3), 0).astype(np.float32)
    p1, p99 = np.percentile(c, [1, 99])
    c = np.clip((c - p1) / max(p99 - p1, 60.0) * 255, 0, 255).astype(np.uint8)
    bw = cv2.adaptiveThreshold(c, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 31, 45)
    h, w = bw.shape
    rh = max(10, int(row_h))
    rule_len = int(0.8 * w) if signature else max(15, w // 3)
    lines = cv2.morphologyEx(bw, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (rule_len, 1)))
    lines |= cv2.morphologyEx(bw, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, int(rh * 1.6))))
    ink = cv2.subtract(bw, cv2.dilate(lines, np.ones((3, 3), np.uint8)))
    ink = cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))  # speckle
    # Long straight diagonal strokes are the "Z" lines officials draw through
    # unused rows. They cross a column over several rows, so only segments
    # ~4 rows long are erased: shorter ones can be chance alignments of digit
    # strokes in stacked rows, and erasing those would hide digits from the
    # masking. Horizontal rules are already gone; vertical strokes are digits.
    segs = cv2.HoughLinesP(ink, 1, np.pi / 180, 40, minLineLength=max(120, int(4 * rh)), maxLineGap=8)
    for x1, y1, x2, y2 in (segs.reshape(-1, 4) if segs is not None else ()):
        ang = abs(np.degrees(np.arctan2(int(y2) - int(y1), int(x2) - int(x1))))
        if 8 <= min(ang, 180 - ang) <= 60:
            cv2.line(ink, (int(x1), int(y1)), (int(x2), int(y2)), 0, 7)
    ink[:, :4] = 0  # column rules at the edges
    ink[:, -4:] = 0
    # Leftover pieces of those lines and of rules: thin strokes lying closer to
    # horizontal than vertical. (A handwritten "1" is thin too, but near-vertical.)
    n, lab, st, _ = cv2.connectedComponentsWithStats(ink)
    for i in range(1, n):
        x, _, cw, _, area = st[i]
        if area < 8:
            ink[lab == i] = 0
            continue
        if cw <= 6 and (x < 14 or x + cw > w - 14):  # broken pieces of the column rules
            ink[lab == i] = 0
            continue
        pts = np.column_stack(np.nonzero(lab == i)[::-1]).astype(np.float32)
        _, (rw, rhh), ang = cv2.minAreaRect(pts)
        theta = abs(ang) if rw >= rhh else 90 - abs(ang)  # long side vs horizontal
        # Signatures: only near-flat thin strokes (pieces of printed rules,
        # broken by the photo) go; a signature's curvy strokes stay.
        max_theta = 10 if signature else 50
        if min(rw, rhh) <= 7 and max(rw, rhh) >= 25 and min(theta, 180 - theta) < max_theta:
            ink[lab == i] = 0
    return ink


@dataclass
class Glyph:
    x0: int
    x1: int
    y0: int
    y1: int


def _merge(glyphs: list[Glyph]) -> list[Glyph]:
    """Merge components that overlap horizontally (broken strokes of one glyph)."""
    glyphs = sorted(glyphs, key=lambda g: g.x0)
    out = [glyphs[0]]
    for g in glyphs[1:]:
        m = out[-1]
        if g.x0 <= m.x1 - 2:
            out[-1] = Glyph(m.x0, max(m.x1, g.x1), min(m.y0, g.y0), max(m.y1, g.y1))
        else:
            out.append(g)
    return out


def _looks_printed(strip: np.ndarray, ref: np.ndarray, gx: int, gy: int, gw: int, gh: int,
                   pad: int = 6, min_ncc: float = 0.5, search: tuple[int, int] = (8, 56)) -> bool:
    """Does the scan around this glyph match the reference somewhere nearby
    (i.e. it's print)? Searching a few rows up/down keeps column headers
    recognised when the table is mapped a row off; handwriting matches no
    printed text at any offset."""
    y0, y1 = max(0, gy - pad), min(strip.shape[0], gy + gh + pad)
    x0, x1 = max(0, gx - pad), min(strip.shape[1], gx + gw + pad)
    patch = strip[y0:y1, x0:x1]
    sx, sy = search
    ry0, ry1 = max(0, y0 - sy), min(ref.shape[0], y1 + sy)
    rx0, rx1 = max(0, x0 - sx), min(ref.shape[1], x1 + sx)
    area = ref[ry0:ry1, rx0:rx1]
    if area.shape[0] < patch.shape[0] or area.shape[1] < patch.shape[1] or patch.std() < 1:
        return False
    return float(cv2.matchTemplate(area, patch, cv2.TM_CCOEFF_NORMED).max()) >= min_ncc


def _text_lines(glyphs: list[Glyph], row_h: float) -> list[list[Glyph]]:
    """Group glyphs whose vertical centres lie within half a row of the line's
    running median."""
    lines: list[list[Glyph]] = []
    for g in sorted(glyphs, key=lambda g: (g.y0 + g.y1) / 2):
        cy = (g.y0 + g.y1) / 2
        if lines and abs(cy - np.median([(o.y0 + o.y1) / 2 for o in lines[-1]])) < 0.5 * row_h:
            lines[-1].append(g)
        else:
            lines.append([g])
    return lines


def char_slots(glyphs: list[Glyph], digits: int, row_w: float) -> list[Glyph] | None:
    """One box per written character, or None if the line can't be segmented.

    The masking rules name positions ("keep the first four digits"), so slots
    must be characters, not blobs: joined-up handwriting is split at the
    expected character pitch, and when that still doesn't give a plausible
    count the line's extent is divided into ``digits`` equal slots."""
    merged = _merge(glyphs)
    left, right = merged[0].x0, max(g.x1 for g in merged)
    if right - left < 0.15 * row_w:
        return None
    pitch = (right - left) / digits
    split: list[Glyph] = []
    for g in merged:
        n = max(1, round((g.x1 - g.x0) / pitch))
        step = (g.x1 - g.x0) / n
        split += [Glyph(round(g.x0 + i * step), round(g.x0 + (i + 1) * step), g.y0, g.y1) for i in range(n)]
    if abs(len(split) - digits) <= 1 or (digits >= 10 and len(split) >= digits):
        return split  # IDs/passports run 7-9 characters; phones may carry +254
    top, bottom = min(g.y0 for g in merged), max(g.y1 for g in merged)
    return [Glyph(round(left + i * pitch), round(left + (i + 1) * pitch), top, bottom) for i in range(digits)]


def pick_hidden(n: int, k: int, rng: random.Random, max_run: int = 2) -> list[int]:
    """Random k of n slots to hide, leaving no more than ``max_run`` visible
    slots side by side where possible (a visible run is what makes a number
    usable). Falls back to a plain random choice if no such pattern exists."""
    k = min(k, n)
    best_hidden = None
    for _ in range(200):
        hidden = set(rng.sample(range(n), k))
        run = best = 0
        for i in range(n):
            run = 0 if i in hidden else run + 1
            best = max(best, run)
        if best <= max_run:
            return sorted(hidden)
        best_hidden = best_hidden or hidden
    return sorted(best_hidden)


def choose_hidden(n: int, rule: str, rng: random.Random, s: "MaskSettings") -> list[int]:
    """Which character slots to hide.

    phone  ``s.hide`` of the last ``s.phone_tail`` digits; the prefix (network
           code, e.g. 0734) stays visible.
    id     ``s.hide`` of all characters (ID or passport number)."""
    if rule == "phone":
        start = max(0, n - s.phone_tail)
        return [start + i for i in pick_hidden(n - start, s.hide, rng)]
    return pick_hidden(n, s.hide, rng)


def partial_column(img: np.ndarray, gray: np.ndarray, f: Mapper, cells: list[Zone],
                   s: MaskSettings, rng: random.Random, margin: tuple[float, float] = (0.0, 0.0),
                   ref: np.ndarray | None = None) -> list[dict]:
    """Hide a random subset of characters in every cell of one table column.

    ``margin`` (px above, px below) widens the search when the column's
    position is uncertain (table rules couldn't be matched): every line of
    writing found there is masked as belonging to the nearest row.

    ``ref`` is the template's reference image (handwriting removed). A glyph
    whose surroundings look the same in the scan and in the reference is
    printed text (a column header) and is left alone. Appearance, not just
    position: if the table is mapped a row off, handwriting lands where the
    reference has printed text, and only a look-alike test keeps it from
    being mistaken for print. Header text that no longer lines up just gets
    masked too: over-masking, never a leak."""
    cells = sorted(cells, key=lambda z: z.rect[1])
    x, y, w = cells[0].rect[:3]
    bottom = cells[-1].rect[1] + cells[-1].rect[3]
    vpad = s.vpad + margin[0]
    sw, sh = max(1, round(w)), max(1, round(bottom - y + vpad + s.vpad + margin[1]))
    oy = y - vpad
    # strip pixel -> template -> scan, sampled per pixel so curled pages work
    gx, gy = np.meshgrid(np.arange(sw, dtype=np.float64) + x, np.arange(sh, dtype=np.float64) + oy)
    m = f(np.stack([gx.ravel(), gy.ravel()], 1)).astype(np.float32).reshape(sh, sw, 2)
    strip = cv2.remap(gray, m[..., 0], m[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    to_template = np.float64([x, oy])
    row_h = float(np.median([z.rect[3] for z in cells]))
    ink = ink_mask(strip, row_h)

    # cell bands in strip coordinates
    bands = [(z.rect[1] - y + vpad, z.rect[1] - y + vpad + z.rect[3]) for z in cells]
    per_row: list[list[Glyph]] = [[] for _ in cells]
    _, _, st, _ = cv2.connectedComponentsWithStats(ink)
    R = None
    if ref is not None:
        # reference crop in the same (template) coordinates as the strip
        R = np.full((sh, sw), 255, np.uint8)
        x0, y0 = int(round(x)), int(round(oy))
        sy0, sx0 = max(0, -y0), max(0, -x0)
        crop = ref[max(0, y0):y0 + sh, max(0, x0):x0 + sw]
        R[sy0:sy0 + crop.shape[0], sx0:sx0 + crop.shape[1]] = crop
    found = []
    for gx, gy, gw, gh, area in st[1:]:
        if gh < s.min_glyph_h * row_h or area <= 30:
            continue
        if R is not None and _looks_printed(strip, R, gx, gy, gw, gh):
            continue
        found.append(Glyph(int(gx), int(gx + gw), int(gy), int(gy + gh)))
    # Each written line is masked on its own, using its own glyph boxes, so
    # masks follow the writing even when the row bands are a little off.
    # Lines are assigned to rows only to tell empty rows from unreadable ones,
    # and for the report. Writing sits on a row's bottom rule and descenders
    # cross it, so the upper part of a line says more about its row than its
    # centre does. A line in the search margin goes to the nearest row:
    # writing is never skipped.
    per_row: list[list[dict]] = [[] for _ in cells]
    for line in _text_lines(found, row_h):
        cy = float(np.median([g.y0 + 0.35 * (g.y1 - g.y0) for g in line]))
        i = int(np.argmin([0 if b0 <= cy < b1 else min(abs(cy - b0), abs(cy - b1)) for b0, b1 in bands]))
        top, bot = min(g.y0 for g in line), max(g.y1 for g in line)
        slots = char_slots(line, cells[i].digits, w)
        if slots is None:  # can't segment this line: hide it whole
            quad = np.float64([[0, top], [w, top], [w, bot], [0, bot]])
            _fill(img, f, _grow(quad, s.mask_vpad + 4) + to_template)
            per_row[i].append({"mode": "full"})
            continue
        hidden = choose_hidden(len(slots), cells[i].rule, rng, s)
        # A hidden character's mask runs from the midpoint with its left
        # neighbour to the midpoint with its right one, over the line's full
        # height plus a margin. Glyph boxes come from cleaned ink (rule
        # removal can shave a 5's or 7's top bar), so masking just the box
        # could leave part of a "hidden" digit showing.
        pitch = (slots[-1].x1 - slots[0].x0) / len(slots)
        lo = [slots[0].x0 - pitch / 2] + [(a.x1 + b.x0) / 2 for a, b in zip(slots, slots[1:])]
        hi = [(a.x1 + b.x0) / 2 for a, b in zip(slots, slots[1:])] + [slots[-1].x1 + pitch / 2]
        h = bot - top  # margins cover a shaved top bar, not the rows above/below
        y0, y1 = top - 0.15 * h - 2, bot + 0.1 * h + 2
        for j in hidden:
            quad = np.float64([[lo[j], y0], [hi[j], y0], [hi[j], y1], [lo[j], y1]])
            _fill(img, f, quad + to_template)
        per_row[i].append({"mode": "partial", "slots": len(slots), "hidden": hidden})

    # Rows with no line of writing: empty, or ink we couldn't make glyphs of.
    stray = ink.copy()
    for g in found:
        stray[g.y0:g.y1, g.x0:g.x1] = 0
    report = []
    for z, (b0, b1), lines in zip(cells, bands, per_row):
        if lines:
            modes = {l["mode"] for l in lines}
            report.append({"zone": z.name, "rule": z.rule, "mode": "full" if modes == {"full"} else "partial",
                           "lines": [{k: v for k, v in l.items() if k != "mode"} for l in lines
                                     if l["mode"] == "partial"]})
            continue
        share = stray[int(b0):int(b1)].mean() / 255 if b1 > b0 else 0.0
        if share >= 3 * s.empty_ink:
            blackout(img, f, z, s.mask_vpad + 4)
            report.append({"zone": z.name, "mode": "full"})
        else:
            report.append({"zone": z.name, "mode": "empty"})
    return report


def verify_column(out: np.ndarray, f: Mapper, cells: list[Zone], ref: np.ndarray | None,
                  s: MaskSettings, window: tuple[float, float] = (60.0, 110.0), max_run: int | None = None) -> str:
    """Independent leak check on the redacted image: re-read the column over
    a window (px above, below) at least as wide as the one used for masking --
    drift on bent pages is downward, and the declaration with the officers'
    names sits just above the table -- and look for ``max_run`` or
    more written characters in a row with no mask between them. A masked line
    is broken up every few characters; a missed one isn't. By default the
    allowed run follows the rule: a phone keeps its prefix plus at most two
    more digits visible together (flag at 8), an ID at most two (flag at 5).
    Returns a reason, or "" if clean."""
    if max_run is None:
        max_run = 8 if cells and cells[0].rule == "phone" else 5
    cells = sorted(cells, key=lambda z: z.rect[1])
    x, y, w = cells[0].rect[:3]
    bottom = cells[-1].rect[1] + cells[-1].rect[3]
    sw, sh = max(1, round(w)), max(1, round(bottom - y + window[0] + window[1]))
    oy = y - window[0]
    gx, gy = np.meshgrid(np.arange(sw, dtype=np.float64) + x, np.arange(sh, dtype=np.float64) + oy)
    m = f(np.stack([gx.ravel(), gy.ravel()], 1)).astype(np.float32).reshape(sh, sw, 2)
    gray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY) if out.ndim == 3 else out
    strip = cv2.remap(gray, m[..., 0], m[..., 1], cv2.INTER_NEAREST, borderMode=cv2.BORDER_REPLICATE)
    masked = cv2.dilate((strip == 0).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    row_h = float(np.median([z.rect[3] for z in cells]))
    ink = ink_mask(np.where(masked, 255, strip).astype(np.uint8), row_h)
    R = None
    if ref is not None:
        R = np.full((sh, sw), 255, np.uint8)
        x0, y0 = int(round(x)), int(round(oy))
        crop = ref[max(0, y0):y0 + sh, max(0, x0):x0 + sw]
        R[max(0, -y0):max(0, -y0) + crop.shape[0], max(0, -x0):max(0, -x0) + crop.shape[1]] = crop
    _, _, st, _ = cv2.connectedComponentsWithStats(ink)
    glyphs = [Glyph(int(a), int(a + b), int(c), int(c + d)) for a, c, b, d, area in st[1:]
              if d >= s.min_glyph_h * row_h and area > 30
              and not (R is not None and _looks_printed(strip, R, a, c, b, d))]
    for line in _text_lines(glyphs, row_h):
        line = _merge(line)
        top, bot = min(g.y0 for g in line), max(g.y1 for g in line)
        run = best = 1
        for a, b in zip(line, line[1:]):
            gap = masked[top:bot, a.x1:max(a.x1 + 1, b.x0)]
            run = 1 if gap.any() else run + 1
            best = max(best, run)
        if best >= max_run:
            name = cells[0].group or cells[0].name
            return f"{name}: {best} written characters in a row left unmasked"
    return ""


def check_column_edges(gray: np.ndarray, f: Mapper, cells: list[Zone], tol: float = 14.0,
                       margin: float = 40.0, require: bool = False) -> str:
    """Is the column where we think it is? Straighten it with a horizontal
    margin and compare the printed vertical rules at its edges with any found
    elsewhere. A table matched one rule off puts a column's real bounding rule
    well inside or outside its edges; checking the masks against the same
    mapping can't see that, so this checks the mapping against the paper.

    Fails when a rule away from an edge is clearly stronger than the one at
    it. With ``require`` (table position not verified by its rules), also
    fails when no rule is visible at an edge at all.
    Returns a reason, or "" if the column looks right."""
    cells = sorted(cells, key=lambda z: z.rect[1])
    x, y, w = cells[0].rect[:3]
    bottom = cells[-1].rect[1] + cells[-1].rect[3]
    sw, sh = max(1, round(w + 2 * margin)), max(1, round(bottom - y))
    gx, gy = np.meshgrid(np.arange(sw, dtype=np.float64) + x - margin, np.arange(sh, dtype=np.float64) + y)
    m = f(np.stack([gx.ravel(), gy.ravel()], 1)).astype(np.float32).reshape(sh, sw, 2)
    strip = cv2.remap(gray, m[..., 0], m[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    bw = cv2.adaptiveThreshold(cv2.GaussianBlur(strip, (3, 3), 0), 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                               cv2.THRESH_BINARY_INV, 31, 6)
    row_h = float(np.median([z.rect[3] for z in cells]))
    rules = cv2.morphologyEx(bw, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, int(row_h))))
    profile = cv2.GaussianBlur((rules > 0).sum(axis=0).astype(np.float32).reshape(1, -1), (5, 1), 0).ravel()
    edges = (margin, margin + w)
    near = np.zeros(sw, bool)
    for e in edges:
        near[int(max(0, e - tol)):int(min(sw, e + tol + 1))] = True
    elsewhere = float(profile[~near].max(initial=0))
    name = cells[0].group or cells[0].name
    for e, label in zip(edges, ("left", "right")):
        at = float(profile[int(max(0, e - tol)):int(min(sw, e + tol + 1))].max(initial=0))
        if elsewhere >= 0.15 * sh and at < 0.6 * elsewhere:
            return f"{name}: printed rule found off the column's {label} edge (table placed wrong)"
        if require and at < 0.1 * sh:
            return f"{name}: can't see the printed rule at the column's {label} edge"
    return ""


# --- signatures ---------------------------------------------------------------------------

SIG_OVERRUN = 24  # px beyond each side of the column searched: signatures overrun their box
SIG_MIN_INK = 40  # ink pixels for a blob to count as (part of) a signature
SIG_JOIN = 9  # px: strokes closer than this belong to the same signature
SIG_MIN_H = 8  # px: anything flatter is a printed rule, not a signature
SIG_BELOW = 30.0  # extra px searched below: drift on bent pages is downward


def _column_strip(gray: np.ndarray, f: Mapper, cells: list[Zone], above: float, below: float, extra: float):
    """Rectified strip covering a table column (template scale), widened by
    ``extra`` px each side and ``above``/``below`` px. Returns the strip, its
    template origin and the row bands in strip coordinates."""
    cells = sorted(cells, key=lambda z: z.rect[1])
    x, y, w = cells[0].rect[:3]
    bottom = cells[-1].rect[1] + cells[-1].rect[3]
    sw, sh = max(1, round(w + 2 * extra)), max(1, round(bottom - y + above + below))
    ox, oy = x - extra, y - above
    gx, gy = np.meshgrid(np.arange(sw, dtype=np.float64) + ox, np.arange(sh, dtype=np.float64) + oy)
    m = f(np.stack([gx.ravel(), gy.ravel()], 1)).astype(np.float32).reshape(sh, sw, 2)
    strip = cv2.remap(gray, m[..., 0], m[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    bands = [(z.rect[1] - oy, z.rect[1] - oy + z.rect[3]) for z in cells]
    return strip, np.float64([ox, oy]), bands, cells


def _ref_crop(ref: np.ndarray | None, origin: np.ndarray, shape) -> np.ndarray | None:
    if ref is None:
        return None
    sh, sw = shape
    R = np.full((sh, sw), 255, np.uint8)
    x0, y0 = int(round(origin[0])), int(round(origin[1]))
    crop = ref[max(0, y0):y0 + sh, max(0, x0):x0 + sw]
    R[max(0, -y0):max(0, -y0) + crop.shape[0], max(0, -x0):max(0, -x0) + crop.shape[1]] = crop
    return R


def _remove_stamp_rings(ink: np.ndarray) -> None:
    """Erase the ring of a round official stamp (it would join everything near
    it into one big "signature"). Only the ring: a signature written across
    the stamp stays ink and is still masked."""
    blur = cv2.GaussianBlur(ink, (5, 5), 0)
    circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, dp=1.5, minDist=150, param1=100, param2=40,
                               minRadius=55, maxRadius=130)
    for cx, cy, r in (np.round(circles[0]).astype(int) if circles is not None else ()):
        ring = np.zeros_like(ink)
        cv2.circle(ring, (cx, cy), r, 255, 9)
        if (ink[ring > 0] > 0).mean() > 0.35:  # really a drawn circle, not chance
            ink[ring > 0] = 0


def _signature_blobs(strip: np.ndarray, ink: np.ndarray, extra: float,
                     w: float, min_ink: int, bands: list[tuple[float, float]]) -> list[tuple[int, int, int, int, int]]:
    """Separate signatures: strokes joined mostly sideways (signatures run
    horizontally; rows are stacked). A blob spanning several rows (strokes of
    neighbouring rows touching) is cut at the row boundaries, one box per row.
    Returns (x0, y0, x1, y1, ink_pixels) per part that is mostly inside the
    column."""
    joined = cv2.dilate(ink, cv2.getStructuringElement(cv2.MORPH_RECT, (SIG_JOIN + 6, 5)))
    n, lab = cv2.connectedComponents(joined)
    edges = [bands[0][0]] + [(a[1] + b[0]) / 2 for a, b in zip(bands, bands[1:])] + [bands[-1][1]]
    parts = []
    for i in range(1, n):
        ys, xs = np.nonzero((lab == i) & (ink > 0))
        if len(xs) < min_ink:
            continue
        row = np.clip(np.searchsorted(edges, ys, side="right") - 1, 0, len(bands) - 1)
        for r in np.unique(row):
            sel = row == r
            if sel.sum() >= min(min_ink, len(xs)) or len(np.unique(row)) == 1:
                parts.append((xs[sel], ys[sel]))
    blobs = []
    for xs, ys in parts:
        if ((xs >= extra) & (xs < extra + w)).mean() < 0.5:
            continue  # writing from the neighbouring column
        x0, x1, y0, y1 = int(xs.min()), int(xs.max()) + 1, int(ys.min()), int(ys.max()) + 1
        if y1 - y0 <= SIG_MIN_H:
            continue  # flat: a printed rule or its remains, not a signature
        # No "looks printed" test here, unlike for digits: in testing, a faint
        # trace in the reference matched real signatures well enough to skip
        # them. Masking a printed "Signature" label is the safe failure.
        blobs.append((x0, y0, x1, y1, len(xs)))
    return blobs


def _sig_window(cells: list[Zone], s: MaskSettings, margin=(0.0, 0.0), more_below: float = 0.0):
    above = (cells[0].search[0] if cells[0].search else s.vpad) + margin[0]
    below = (cells[0].search[1] if cells[0].search else s.vpad) + margin[1] + more_below
    return above, below


def _find_signatures(gray, f, cells, s, margin, ref, more_below=0.0):
    above, below = _sig_window(cells, s, margin, more_below)
    strip, origin, bands, cells = _column_strip(gray, f, cells, above, below, SIG_OVERRUN)
    row_h = float(np.median([z.rect[3] for z in cells]))
    ink = ink_mask(strip, row_h, signature=True)
    _remove_stamp_rings(ink)
    return _signature_blobs(strip, ink, SIG_OVERRUN, cells[0].rect[2], SIG_MIN_INK, bands), origin, bands, cells


def signature_column(img: np.ndarray, gray: np.ndarray, f: Mapper, cells: list[Zone], s: MaskSettings,
                     margin: tuple[float, float] = (0.0, 0.0), ref: np.ndarray | None = None,
                     pad: float = 6.0) -> list[dict]:
    """Black out each signature on its own box, instead of the whole column.

    Ink in the column (plus SIG_OVERRUN px each side, plus the zone's search
    margin above/below) is split into signatures; each gets its own padded
    box. Empty rows stay untouched."""
    blobs, origin, bands, cells = _find_signatures(gray, f, cells, s, margin, ref, SIG_BELOW)
    counts = [0] * len(cells)
    for x0, y0, x1, y1, _ in blobs:
        quad = np.float64([[x0 - pad, y0 - pad], [x1 + pad, y0 - pad], [x1 + pad, y1 + pad], [x0 - pad, y1 + pad]])
        _fill(img, f, quad + origin)
        cy = (y0 + y1) / 2
        i = int(np.argmin([0 if b0 <= cy < b1 else min(abs(cy - b0), abs(cy - b1)) for b0, b1 in bands]))
        counts[i] += 1
    return [{"zone": z.name, "mode": "signature" if c else "empty", **({"signatures": c} if c else {})}
            for z, c in zip(cells, counts)]


def verify_signatures(gray: np.ndarray, out: np.ndarray, f: Mapper, cells: list[Zone], ref: np.ndarray | None,
                      s: MaskSettings, margin=(0.0, 0.0), more_below: float = SIG_BELOW) -> str:
    """Coverage check: find signatures again on the *original* image, over a
    window reaching further down (drift on bent pages is downward), and
    require each to be at least 90% under black in the redacted image."""
    blobs, origin, _, cells = _find_signatures(gray, f, cells, s, margin, ref, more_below)
    above, below = _sig_window(cells, s, margin, more_below)
    o = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY) if out.ndim == 3 else out
    ostrip = _column_strip(o, f, cells, above, below, SIG_OVERRUN)[0]
    gstrip = _column_strip(gray, f, cells, above, below, SIG_OVERRUN)[0]
    ink = ink_mask(gstrip, float(np.median([z.rect[3] for z in cells])), signature=True)
    bad = 0
    for x0, y0, x1, y1, n in blobs:
        if n < 2 * SIG_MIN_INK:
            continue
        sel = ink[y0:y1, x0:x1] > 0
        if (ostrip[y0:y1, x0:x1][sel] <= 8).mean() < 0.9:
            bad += 1
    if bad:
        name = cells[0].group or cells[0].name
        return f"{name}: {bad} signature(s) not fully masked"
    return ""
