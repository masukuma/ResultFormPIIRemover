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
    # ...and outside the rows themselves (the search margins, where headers and
    # neighbouring lines are) at least this share, whatever min_glyph_h is
    min_glyph_h_out: float = 0.4
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


def _ink_span(ink: np.ndarray, y0: float, y1: float, pad: float) -> tuple[float, float] | None:
    """Left and right ends of the ink between rows y0 and y1 (pieces over 30
    px), widened by ``pad``: what a mask hiding all the writing there must
    cover. None if there is none."""
    _, _, st, _ = cv2.connectedComponentsWithStats(
        (ink[max(int(y0), 0):max(int(np.ceil(y1)), 0)] > 0).astype(np.uint8))
    big = st[1:][st[1:, 4] > 30]
    if not len(big):
        return None
    return float(big[:, 0].min()) - pad, float((big[:, 0] + big[:, 2]).max()) + pad


Strike = tuple[float, float, float, float, float]  # x1, y1, x2, y2 (template px), stroke width


def find_strikes(gray: np.ndarray, f: Mapper, rect, row_h: float) -> list[Strike]:
    """Lines officials draw through unused rows of a table, found over the
    whole table (``rect``, template coordinates), where they are long: within
    one column they are too short to tell from a pen stroke.

    A strike is one straight, unbroken stroke at least 3 rows long, at an
    angle a printed rule can't have (3-80 degrees in the template's frame).
    Writing a straight line happens to pass through is broken up, so it never
    qualifies: what counts as a strike is erased from the ink the masking
    reads, and erasing digits that way could leave them unmasked."""
    x, y, w, h = map(float, rect)
    sw, sh = max(1, int(round(w))), max(1, int(round(h)))
    gx, gy = np.meshgrid(np.arange(sw, dtype=np.float64) + x, np.arange(sh, dtype=np.float64) + y)
    m = f(np.stack([gx.ravel(), gy.ravel()], 1)).astype(np.float32).reshape(sh, sw, 2)
    patch = cv2.remap(gray, m[..., 0], m[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    c = cv2.GaussianBlur(patch, (3, 3), 0).astype(np.float32)
    p1, p99 = np.percentile(c, [1, 99])
    c = np.clip((c - p1) / max(p99 - p1, 60.0) * 255, 0, 255).astype(np.uint8)
    bw = cv2.adaptiveThreshold(c, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 31, 45)
    min_len = max(120.0, 3 * row_h)
    segs = cv2.HoughLinesP(bw, 1, np.pi / 720, 60, minLineLength=int(min_len), maxLineGap=6)
    out: list[Strike] = []
    for x1, y1, x2, y2 in (segs.reshape(-1, 4) if segs is not None else ()):
        dx, dy = float(x2 - x1), float(y2 - y1)
        length = float(np.hypot(dx, dy))
        ang = abs(np.degrees(np.arctan2(dy, dx))) % 180
        ang = min(ang, 180 - ang)
        if not 3 <= ang <= 80:
            continue  # a printed rule (or a column rule), not a strike
        # unbroken along its whole length, and thin: a stroke, not text
        n = int(length)
        t = np.linspace(0, 1, n)[:, None]
        pts = np.float64([x1, y1]) + t * np.float64([dx, dy])
        nx, ny = -dy / length, dx / length
        widths, hits = [], 0
        for px, py in pts[:: max(1, n // 60)]:
            prof = [bw[int(round(py + k * ny)), int(round(px + k * nx))] > 0
                    if 0 <= int(round(py + k * ny)) < sh and 0 <= int(round(px + k * nx)) < sw else False
                    for k in range(-8, 9)]
            if prof[8]:
                hits += 1
                lo = hi = 8
                while lo > 0 and prof[lo - 1]:
                    lo -= 1
                while hi < 16 and prof[hi + 1]:
                    hi += 1
                widths.append(hi - lo + 1)
        samples = len(pts[:: max(1, n // 60)])
        if hits < 0.9 * samples or not widths or np.median(widths) > 0.25 * row_h:
            continue
        out.append((x1 + x, y1 + y, x2 + x, y2 + y, float(np.median(widths))))
    return _merge_strikes(out)


def _merge_strikes(segs: list[Strike]) -> list[Strike]:
    """Hough returns one drawn line as many overlapping pieces: merge pieces
    lying on the same line into one, spanning them all."""
    lines: list[list] = []
    for x1, y1, x2, y2, sw in sorted(segs, key=lambda s: -np.hypot(s[2] - s[0], s[3] - s[1])):
        for ln in lines:
            ax, ay, bx, by, w = ln
            d = np.float64([bx - ax, by - ay])
            u = d / np.linalg.norm(d)
            n = np.float64([-u[1], u[0]])
            if all(abs(np.dot(np.float64([px - ax, py - ay]), n)) <= max(8.0, 2 * w)
                   for px, py in ((x1, y1), (x2, y2))):
                ts = [0.0, float(np.dot(d, u))] + [float(np.dot(np.float64([px - ax, py - ay]), u))
                                                   for px, py in ((x1, y1), (x2, y2))]
                a, b = min(ts), max(ts)
                ln[:] = [ax + a * u[0], ay + a * u[1], ax + b * u[0], ay + b * u[1], max(w, sw)]
                break
        else:
            lines.append([x1, y1, x2, y2, sw])
    return [tuple(map(float, ln)) for ln in lines]


def _from_above(cells: list[Zone], bands, gy: float, gh: float) -> bool:
    """On a single underlined field: writing cut off by the window's top edge
    that ends in the field's upper half is the line above's (a name written
    large and low), not part of the number written on this line."""
    return not cells[0].region and gy <= 1 and gy + gh < (bands[0][0] + bands[0][1]) / 2


def _in_rows(y: float, bands) -> bool:
    return any(b0 <= y < b1 for b0, b1 in bands)


def _exclude(ink: np.ndarray, exclude: np.ndarray | None, m: np.ndarray) -> None:
    """Drop ink where ``exclude`` (image-sized, nonzero: e.g. stamps) is set;
    ``m`` maps strip pixels to image coordinates."""
    if exclude is not None:
        e = cv2.remap(exclude, m[..., 0], m[..., 1], cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        ink[e > 0] = 0


def _erase_strikes(ink: np.ndarray, origin, strikes: list[Strike] | None) -> None:
    """Erase strike lines (template coordinates) from a strip's ink, in place."""
    for x1, y1, x2, y2, sw in strikes or ():
        p = (int(round(x1 - origin[0])), int(round(y1 - origin[1])))
        q = (int(round(x2 - origin[0])), int(round(y2 - origin[1])))
        cv2.line(ink, p, q, 0, int(round(sw)) + 4)


def _remove_dotted_leaders(ink: np.ndarray, row_h: float, min_dots: int = 6) -> None:
    """Erase printed dotted leaders ("ID Number: ..........") in place: runs
    of at least ``min_dots`` tiny blobs on one line at an even pitch. Writing
    on the line keeps its strokes (they are not tiny); a handwritten full
    stop has no run of evenly spaced neighbours."""
    n, lab, st, cen = cv2.connectedComponentsWithStats(ink)
    small = [i for i in range(1, n) if st[i][3] <= max(6, 0.15 * row_h) and st[i][2] <= max(10, 0.3 * row_h)]
    if len(small) < min_dots:
        return
    rows: list[list[int]] = []
    for i in sorted(small, key=lambda i: cen[i][1]):
        for r in rows:
            if abs(np.mean([cen[j][1] for j in r]) - cen[i][1]) <= 3:
                r.append(i)
                break
        else:
            rows.append([i])
    for r in rows:
        r.sort(key=lambda i: cen[i][0])
        run = [r[0]]
        for a, b in zip(r, r[1:] + [None]):
            gap = cen[b][0] - cen[a][0] if b is not None else None
            if gap is not None and run and (len(run) < 2 or abs(gap - (cen[run[1]][0] - cen[run[0]][0])) <= 3) \
                    and 4 <= gap <= max(25, 0.5 * row_h):
                run.append(b)
                continue
            if len(run) >= min_dots:
                for j in run:
                    ink[lab == j] = 0
            run = [b] if b is not None else []


def _binarize(strip: np.ndarray) -> np.ndarray:
    """Dark marks of a rectified strip (255 = ink), printed or written."""
    # KIEMS photos range from crisp to blurry and grey-on-grey; stretch the
    # contrast (with a floor, so an empty cell's faint security pattern isn't
    # blown up into "ink") before thresholding.
    c = cv2.GaussianBlur(strip, (3, 3), 0).astype(np.float32)
    p1, p99 = np.percentile(c, [1, 99])
    c = np.clip((c - p1) / max(p99 - p1, 60.0) * 255, 0, 255).astype(np.uint8)
    bw = cv2.adaptiveThreshold(c, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 31, 45)
    return _destripe(bw)


STRIPED = 0.08  # share of a strip's rows mostly covered by thin lines that makes it a striped scan


def _destripe(bw: np.ndarray) -> np.ndarray:
    """Some forms are embossed, or scanned with a striped background: at
    template scale the background comes out as thin, dark horizontal lines
    every few pixels, its lettering as kinks in them. In empty fields they
    were taken for writing and masked. Where many rows of a strip are mostly
    thin line (no plain scan comes near: 0-1% of rows, against 16-31% on the
    one embossed form in the samples), thin horizontal runs are dropped. Pen
    strokes are thicker; a stripe crossing one leaves it whole."""
    thin = cv2.morphologyEx(bw, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (12, 1)))
    thin &= ~cv2.morphologyEx(bw, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3)))
    if ((thin > 0).mean(axis=1) > 0.2).mean() < STRIPED:
        return bw
    return cv2.subtract(bw, thin)


def ink_mask(strip: np.ndarray, row_h: float, signature: bool = False) -> np.ndarray:
    """Binary ink of a rectified strip, with rules, strike lines and the
    security-print background removed.

    ``signature``: keep long flat pen strokes (a signature's flourish): only
    horizontal lines spanning nearly the whole strip count as printed rules,
    and the thin-stroke filter meant for digit columns is skipped."""
    bw = _binarize(strip)
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
    # Signature columns: a line drawn across many empty rows can be nearly
    # flat, and blobs of it would be masked as signatures; there the lines
    # are found before the rules are taken out (that cuts them into pieces at
    # every rule they cross). Digit columns keep to clearly sloped lines: on
    # a curled page a printed rule can be tilted a few degrees in the strip.
    if signature:
        segs, lo = cv2.HoughLinesP(bw, 1, np.pi / 180, 40, minLineLength=max(120, int(4 * rh)), maxLineGap=8), 2.5
    else:
        segs, lo = cv2.HoughLinesP(ink, 1, np.pi / 180, 40, minLineLength=max(120, int(4 * rh)), maxLineGap=8), 8
    for x1, y1, x2, y2 in (segs.reshape(-1, 4) if segs is not None else ()):
        ang = abs(np.degrees(np.arctan2(int(y2) - int(y1), int(x2) - int(x1))))
        if lo <= min(ang, 180 - ang) <= 80:  # steeper is a column rule, or digits that line up
            cv2.line(ink, (int(x1), int(y1)), (int(x2), int(y2)), 0, 7)
    ink[:, :4] = 0  # column rules at the edges
    ink[:, -4:] = 0
    _remove_dotted_leaders(ink, rh)
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


def _looks_printed(ink: np.ndarray, ref_ink: np.ndarray, gx: int, gy: int, gw: int, gh: int,
                   pad: int = 6, min_ncc: float = 0.6, search: tuple[int, int] = (8, 56)) -> bool:
    """Is this glyph printed text (e.g. a column header) rather than writing?

    Compares *ink with printed rules removed*, the glyph's against the clean
    reference's around the same place (a few rows up/down, so headers are
    still recognised when a table is mapped a row off). Comparing raw pixels
    is unsafe: a digit written on an underline matches the reference's
    underline and gets mistaken for print -- and then is neither masked nor
    flagged. Where the reference has (almost) no ink nearby, nothing printed
    can be there, so the glyph is writing."""
    y0, y1 = max(0, gy - pad), min(ink.shape[0], gy + gh + pad)
    x0, x1 = max(0, gx - pad), min(ink.shape[1], gx + gw + pad)
    patch = cv2.GaussianBlur(ink[y0:y1, x0:x1].astype(np.float32), (5, 5), 0)
    sx, sy = search
    ry0, ry1 = max(0, y0 - sy), min(ref_ink.shape[0], y1 + sy)
    rx0, rx1 = max(0, x0 - sx), min(ref_ink.shape[1], x1 + sx)
    area = ref_ink[ry0:ry1, rx0:rx1]
    if area.shape[0] < patch.shape[0] or area.shape[1] < patch.shape[1] or patch.std() < 1:
        return False
    if (area > 0).sum() < 0.3 * (ink[y0:y1, x0:x1] > 0).sum():
        return False  # no printed text nearby in the reference
    area = cv2.GaussianBlur(area.astype(np.float32), (5, 5), 0)
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
                   ref: np.ndarray | None = None, strikes: list[Strike] | None = None,
                   exclude: np.ndarray | None = None) -> list[dict]:
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
    # Zones on a single underlined line set their own (small) search window,
    # so writing on the neighbouring lines (e.g. the officer's name) is never
    # swept into the ID masking.
    above, below = cells[0].search if cells[0].search else (s.vpad, s.vpad)
    vpad = above + margin[0]
    sw, sh = max(1, round(w)), max(1, round(bottom - y + vpad + below + margin[1]))
    oy = y - vpad
    # strip pixel -> template -> scan, sampled per pixel so curled pages work
    gx, gy = np.meshgrid(np.arange(sw, dtype=np.float64) + x, np.arange(sh, dtype=np.float64) + oy)
    m = f(np.stack([gx.ravel(), gy.ravel()], 1)).astype(np.float32).reshape(sh, sw, 2)
    strip = cv2.remap(gray, m[..., 0], m[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    to_template = np.float64([x, oy])
    row_h = float(np.median([z.rect[3] for z in cells]))
    ink = ink_mask(strip, row_h)
    _erase_strikes(ink, to_template, strikes)
    _exclude(ink, exclude, m)

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
    R_ink = ink_mask(R, row_h) if R is not None else None
    found, printed = [], []
    for gx, gy, gw, gh, area in st[1:]:
        if gh < s.min_glyph_h * row_h or area <= 30:
            continue
        if gh < s.min_glyph_h_out * row_h and not _in_rows(gy + gh / 2, bands):
            continue  # small, and not in a row: header text or the line above, not an entry
        if _from_above(cells, bands, gy, gh):
            continue
        if R_ink is not None and _looks_printed(ink, R_ink, gx, gy, gw, gh):
            printed.append(Glyph(int(gx), int(gx + gw), int(gy), int(gy + gh)))
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
            # All of its ink, glyph or not (characters too small to count
            # are part of it), but not the field's whole width: on an
            # officer's line that ran on behind the stamp.
            x0, x1 = _ink_span(ink, top, bot, 0.25 * row_h) or (0, w)
            x0, x1 = max(x0, 0), min(x1, w)
            quad = np.float64([[x0, top], [x1, top], [x1, bot], [x0, bot]])
            _fill(img, f, _grow(quad, s.mask_vpad + 4) + to_template)
            per_row[i].append({"mode": "full"})
            continue
        hidden = choose_hidden(len(slots), cells[i].rule, rng, s)
        # The line's typical extent, not its tallest outlier: a stroke from the
        # line above (or a stamp's edge) would otherwise stretch every hidden
        # digit's mask up into a name.
        mt = float(np.median([g.y0 for g in line]))
        mb = float(np.median([g.y1 for g in line]))
        # A hidden character's mask runs from the midpoint with its left
        # neighbour to the midpoint with its right one, over the line's full
        # height plus a margin. Glyph boxes come from cleaned ink (rule
        # removal can shave a 5's or 7's top bar), so masking just the box
        # could leave part of a "hidden" digit showing.
        pitch = (slots[-1].x1 - slots[0].x0) / len(slots)
        lo = [slots[0].x0 - pitch / 2] + [(a.x1 + b.x0) / 2 for a, b in zip(slots, slots[1:])]
        hi = [(a.x1 + b.x0) / 2 for a, b in zip(slots, slots[1:])] + [slots[-1].x1 + pitch / 2]
        h = mb - mt  # margins cover a shaved top bar, not the rows above/below
        for j in hidden:
            y0 = min(slots[j].y0, mt) - 0.15 * h - 2  # the whole of this character, at least the line's height
            y1 = max(slots[j].y1, mb) + 0.1 * h + 2
            quad = np.float64([[lo[j], y0], [hi[j], y0], [hi[j], y1], [lo[j], y1]])
            _fill(img, f, quad + to_template)
        per_row[i].append({"mode": "partial", "slots": len(slots), "hidden": hidden})

    # Rows with no line of writing: empty, or ink we couldn't make glyphs of.
    stray = ink.copy()
    for g in found + printed:  # printed text isn't unreadable writing either
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
            # Ink, but no characters made of it: hide all of it (the cell's
            # height, the ink's width).
            span = _ink_span(stray, b0, b1, 0.25 * row_h)
            if span is None:
                blackout(img, f, z, s.mask_vpad + 4)
            else:
                x0, x1 = max(span[0], 0), min(span[1], w)
                quad = np.float64([[x0, b0], [x1, b0], [x1, b1], [x0, b1]])
                _fill(img, f, _grow(quad, s.mask_vpad + 4, s.mask_vpad + 4) + to_template)
            report.append({"zone": z.name, "mode": "full"})
        else:
            report.append({"zone": z.name, "mode": "empty"})
    return report


def verify_column(out: np.ndarray, f: Mapper, cells: list[Zone], ref: np.ndarray | None,
                  s: MaskSettings, window: tuple[float, float] | None = None, max_run: int | None = None,
                  strikes: list[Strike] | None = None, exclude: np.ndarray | None = None) -> str:
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
    if window is None:
        # tables: wide (drift on bent pages); single underlined fields: just
        # beyond their own search window, so the next line (a name) isn't read
        window = (cells[0].search[0] + 8, cells[0].search[1] + 8) if cells and cells[0].search else (60.0, 110.0)
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
    _erase_strikes(ink, (x, oy), strikes)
    _exclude(ink, exclude, m)
    R = None
    if ref is not None:
        R = np.full((sh, sw), 255, np.uint8)
        x0, y0 = int(round(x)), int(round(oy))
        crop = ref[max(0, y0):y0 + sh, max(0, x0):x0 + sw]
        R[max(0, -y0):max(0, -y0) + crop.shape[0], max(0, -x0):max(0, -x0) + crop.shape[1]] = crop
    _, _, st, _ = cv2.connectedComponentsWithStats(ink)
    bands = [(z.rect[1] - oy, z.rect[1] - oy + z.rect[3]) for z in cells]
    glyphs = [Glyph(int(a), int(a + b), int(c), int(c + d)) for a, c, b, d, area in st[1:]
              if d >= s.min_glyph_h * row_h and area > 30
              and (d >= s.min_glyph_h_out * row_h or _in_rows(c + d / 2, bands))
              and not _from_above(cells, bands, c, d)
              and not (R is not None and _looks_printed(ink, ink_mask(R, row_h), a, c, b, d))]
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


def _remove_stamp_rings(ink: np.ndarray) -> list[tuple[int, int, int]]:
    """Find round official stamps; erase their rings (a ring would join
    everything near it into one big "signature") and return them as
    (cx, cy, r). A signature written across a stamp stays ink and is masked."""
    blur = cv2.GaussianBlur(ink, (5, 5), 0)
    circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, dp=1.5, minDist=150, param1=100, param2=40,
                               minRadius=55, maxRadius=130)
    found = []
    for cx, cy, r in (np.round(circles[0]).astype(int) if circles is not None else ()):
        ring = np.zeros_like(ink)
        cv2.circle(ring, (cx, cy), r, 255, 9)
        if (ink[ring > 0] > 0).mean() > 0.35:  # really a drawn circle, not chance
            ink[ring > 0] = 0
            found.append((int(cx), int(cy), int(r)))
    return found


def _inside_stamp(xs: np.ndarray, ys: np.ndarray, stamps, share: float = 0.9) -> bool:
    """Ink lying (almost) wholly inside a stamp is the stamp's own printed
    text, which stays visible: it is part of what shows a form is genuine."""
    for cx, cy, r in stamps:
        if (((xs - cx) ** 2 + (ys - cy) ** 2) < (0.95 * r) ** 2).mean() >= share:
            return True
    return False


def _signature_blobs(strip: np.ndarray, ink: np.ndarray, extra: float,
                     w: float, min_ink: int, bands: list[tuple[float, float]],
                     stamps=(), bridge: np.ndarray | None = None, line: tuple[int, int] | None = None) -> list:
    """Separate signatures: strokes joined mostly sideways (signatures run
    horizontally; rows are stacked). A blob spanning several rows (strokes of
    neighbouring rows touching) is cut at the row boundaries, one box per row.
    ``bridge``: print (a field's underline) that joins the strokes touching
    it without being part of any signature.
    Returns (x0, y0, x1, y1, ink_pixels, boxes) per part that is mostly inside
    the column. ``boxes`` cover its ink: one box, or with ``line`` (the
    field's line, top and bottom rows), one for the ink on the line and one
    for each stroke above or below it."""
    joined = cv2.dilate(ink if bridge is None else ink | bridge,
                        cv2.getStructuringElement(cv2.MORPH_RECT, (SIG_JOIN + 6, 5)))
    n, lab = cv2.connectedComponents(joined)
    _, own = cv2.connectedComponents((ink > 0).astype(np.uint8))
    stroke = np.bincount(own.ravel())  # ink pixels of each unbroken stroke
    stroke[0] = 0
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
        if stroke[own[ys, xs]].max() < min_ink:
            # Only crumbs (what is left of a security print's lettering, or of
            # a striped scan's stripes), joined into a "signature" by being
            # close together: writing has unbroken strokes.
            continue
        if _inside_stamp(xs, ys, stamps):
            continue
        x0, x1, y0, y1 = int(xs.min()), int(xs.max()) + 1, int(ys.min()), int(ys.max()) + 1
        if y1 - y0 <= SIG_MIN_H:
            continue  # flat: a printed rule or its remains, not a signature
        # No "looks printed" test here, unlike for digits: in testing, a faint
        # trace in the reference matched real signatures well enough to skip
        # them. Masking a printed "Signature" label is the safe failure.
        boxes = [(x0, y0, x1, y1)]
        if line is not None:
            # On the line one box; above and below it one per stroke, so a
            # loop reaching down covers its own width of the next line, and
            # two strokes reaching down don't cover the date between them.
            sec, strokes = np.digitize(ys, line), own[ys, xs]
            sel = [sec == 1] + [(sec == i) & (strokes == k) for i in (0, 2) for k in np.unique(strokes[sec == i])]
            boxes = [(int(xs[k].min()), int(ys[k].min()), int(xs[k].max()) + 1, int(ys[k].max()) + 1)
                     for k in sel if k.any()]
        blobs.append((x0, y0, x1, y1, len(xs), boxes))
    return blobs


def _own_rule(strip: np.ndarray, lo: int, hi: int, ref: np.ndarray | None = None) -> np.ndarray:
    """The printed line a field is written on (255): long and flat, within
    the field's rows. Taken out of a signature's ink it sets no box's size;
    the strokes it joins stay one signature.

    ``ref``: the reference cropped like the strip; the line must be where
    the reference has it. A signature's long flat flourish is a line too,
    and was taken out (left visible) as one. Without a reference, the
    lowest line in the field's lower part."""
    bw = _binarize(strip)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (max(150, bw.shape[1] // 4), 1))
    r0, r1 = max(lo - 4, 0), max(hi + 4, 0)
    rows = np.flatnonzero(cv2.morphologyEx(bw, cv2.MORPH_OPEN, k)[r0:r1].any(axis=1)) + r0
    rule = np.zeros_like(bw)
    lines = np.split(rows, np.flatnonzero(np.diff(rows) > 3) + 1) if len(rows) else []
    if ref is None and (len(lines) > 2 or any(line[-1] - line[0] > 12 for line in lines)):
        # Not a printed line (a scan striped all over, say): taking it out
        # would take out whatever is written across it. With a reference the
        # line is the one where it has it, however many long strokes an
        # officer drew along it.
        return rule
    lines = [line for line in lines if line[-1] - line[0] <= 12]
    if ref is not None:
        printed = cv2.morphologyEx(np.where(ref < 128, 255, 0).astype(np.uint8), cv2.MORPH_OPEN, k)
        prows = np.flatnonzero(printed[r0:r1].any(axis=1)) + r0
        if not len(prows):
            return rule
        at, reach = prows[np.argmin(np.abs(prows - hi))], 20  # a layout variant moves it ~10 px
    else:
        at, reach = hi, 0.4 * (hi - lo)
    lines = [line for line in lines if abs(line.mean() - at) <= reach]
    if not lines:
        return rule
    line = min(lines, key=lambda l: abs(l.mean() - at))
    # The whole line, ends included (a long run finds it, but a gap or a
    # stroke crossing it breaks that run): horizontal runs on its rows.
    runs = cv2.morphologyEx(bw, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1)))
    top, bot = max(line[0] - 2, 0), line[-1] + 3
    rule[top:bot] = runs[top:bot]
    return cv2.dilate(rule, np.ones((3, 3), np.uint8))  # ink_mask's speckle opening shifts ink by a pixel


def _sig_window(cells: list[Zone], s: MaskSettings, margin=(0.0, 0.0), more_below: float = 0.0):
    above = (cells[0].search[0] if cells[0].search else s.vpad) + margin[0]
    below = (cells[0].search[1] if cells[0].search else s.vpad) + margin[1] + more_below
    return above, below


def _find_signatures(gray, f, cells, s, margin, ref, more_below=0.0, printed=None, centered=False, strikes=None,
                     exclude=None):
    """Signature blobs in a column. ``printed`` (template space, nonzero =
    printed text known to be there, e.g. typed "XXXX" in unused rows, or a
    label an officer signed next to): its ink is no part of any signature.
    ``centered``: only blobs centred
    within the cells, not ones reaching in from the search margins (a field
    on a text line, where the margins hold the neighbouring lines)."""
    above, below = _sig_window(cells, s, margin, more_below)
    strip, origin, bands, cells = _column_strip(gray, f, cells, above, below, SIG_OVERRUN)
    row_h = float(np.median([z.rect[3] for z in cells]))
    ink = ink_mask(strip, row_h, signature=True)
    _erase_strikes(ink, origin, strikes)
    if exclude is not None:  # stamps: their rings are not signatures, and are never masked
        ink[_column_strip(exclude, f, cells, above, below, SIG_OVERRUN)[0] > 127] = 0
    if printed is not None:
        ink[_ref_crop(255 - printed, origin, strip.shape) < 128] = 0
    stamps = _remove_stamp_rings(ink)
    lo, hi = int(bands[0][0]), int(np.ceil(bands[-1][1]))
    bridge = line = None
    if centered:
        # Strokes lying mostly on the line below or above (the date, the ID)
        # are no part of the signature, even where they run close enough to
        # it to be joined into one blob. Tails and loops of the signature
        # itself are drawn in one stroke with it, or cross its underline,
        # and stay.
        rule = _own_rule(strip, lo, hi, _ref_crop(ref, origin, strip.shape))
        ink[rule > 0] = 0
        # What is left of the line where the scan broke it into dashes: ink
        # lying wholly within a few pixels of it. Joined through the line,
        # those pieces stretched a signature's mask along all of it.
        near = cv2.dilate(rule.any(axis=1).astype(np.uint8)[:, None], np.ones((7, 1), np.uint8))[:, 0] == 0
        n, lab, st, _ = cv2.connectedComponentsWithStats((ink > 0).astype(np.uint8))
        off_line = np.concatenate([[0], np.cumsum(near)])
        flat = off_line[st[:, 1] + st[:, 3]] == off_line[st[:, 1]]
        flat[0] = False
        ink[flat[lab]] = 0
        # The line joins only strokes crossing it (ink just above and just
        # below it): not everything that touches it, such as the lettering
        # of a security print sitting on it.
        bridge = np.zeros_like(rule)
        rows = np.flatnonzero(rule.any(axis=1))
        if len(rows):
            top, bot = rows[0], rows[-1] + 1
            above = ink[max(top - 4, 0):top].any(axis=0)
            below = ink[bot:bot + 4].any(axis=0)
            wide = np.ones(7, np.uint8)  # a slanted stroke comes out a few px further along
            cross = (np.convolve(above, wide, "same") > 0) & (np.convolve(below, wide, "same") > 0)
            bridge[max(top - 2, 0):bot + 2, cross] = 255
        n, lab = cv2.connectedComponents(((ink > 0) | (bridge > 0)).astype(np.uint8))
        on = ink > 0
        rows = slice(max(lo, 0), max(hi, 0))
        inside = np.bincount(lab[rows][on[rows]], minlength=n)
        off = inside < 0.3 * np.maximum(np.bincount(lab[on], minlength=n), 1)
        off[0] = False
        ink[off[lab]] = 0
        line = (lo, hi)
    blobs = _signature_blobs(strip, ink, SIG_OVERRUN, cells[0].rect[2], SIG_MIN_INK, bands, stamps, bridge, line)
    # Writing cut off by the window's bottom edge that starts well below the
    # last row is the next line's (an officer's name written large under a
    # table), not a signature running over its row: that starts in the row.
    end, sh = bands[-1][1], strip.shape[0]
    blobs = [b for b in blobs if not (b[3] >= sh - 1 and b[1] > end + 0.5 * (sh - end))]
    if centered:
        # A blob belongs to the field when a good share of its ink is on the
        # field's line: tails and loops (a "g", a flourish) reach far above
        # or below, so the box's centre can't tell. When in doubt it's masked.
        def share(b):
            sub = ink[b[1]:b[3], b[0]:b[2]] > 0
            rows = np.arange(b[1], b[3])
            inside = sub[(rows >= lo) & (rows < hi)].sum()
            return inside / max(sub.sum(), 1)
        # Ink cut off by the window's top edge that ends in the field's upper
        # half hangs down from the line above (its underline, the foot of
        # an ID); a signature written on this line reaches further down.
        blobs = [b for b in blobs if share(b) >= 0.3 and not (b[1] <= 1 and b[3] < (lo + hi) / 2)]
    return blobs, origin, bands, cells


def signature_column(img: np.ndarray, gray: np.ndarray, f: Mapper, cells: list[Zone], s: MaskSettings,
                     margin: tuple[float, float] = (0.0, 0.0), ref: np.ndarray | None = None,
                     pad: float = 6.0, printed: np.ndarray | None = None, centered: bool = False,
                     strikes: list[Strike] | None = None, exclude: np.ndarray | None = None) -> list[dict]:
    """Black out each signature on its own box, instead of the whole column.

    Ink in the column (plus SIG_OVERRUN px each side, plus the zone's search
    margin above/below) is split into signatures; each gets its own padded
    box. Empty rows stay untouched."""
    blobs, origin, bands, cells = _find_signatures(gray, f, cells, s, margin, ref, SIG_BELOW, printed, centered,
                                                   strikes, exclude)
    counts = [0] * len(cells)
    for x0, y0, x1, y1, _, boxes in blobs:
        for bx0, by0, bx1, by1 in boxes:
            quad = np.float64([[bx0 - pad, by0 - pad], [bx1 + pad, by0 - pad], [bx1 + pad, by1 + pad],
                               [bx0 - pad, by1 + pad]])
            _fill(img, f, quad + origin)
        cy = (y0 + y1) / 2
        i = int(np.argmin([0 if b0 <= cy < b1 else min(abs(cy - b0), abs(cy - b1)) for b0, b1 in bands]))
        counts[i] += 1
    return [{"zone": z.name, "mode": "signature" if c else "empty", **({"signatures": c} if c else {})}
            for z, c in zip(cells, counts)]


def verify_signatures(gray: np.ndarray, out: np.ndarray, f: Mapper, cells: list[Zone], ref: np.ndarray | None,
                      s: MaskSettings, margin=(0.0, 0.0), more_below: float = SIG_BELOW,
                      keep: np.ndarray | None = None, printed: np.ndarray | None = None,
                      centered: bool = False, strikes: list[Strike] | None = None,
                      exclude: np.ndarray | None = None) -> str:
    """Coverage check: find signatures again on the *original* image, over a
    window reaching further down (drift on bent pages is downward), and
    require each to be at least 90% under black in the redacted image.

    ``keep`` (image-sized, nonzero = must stay visible: stamps) is left out
    of the count: a signature written across a stamp is masked only outside
    it, and ink lying mostly inside a stamp (the stamp's own lettering, or
    writing on top of it) is not required to be masked at all."""
    blobs, origin, _, cells = _find_signatures(gray, f, cells, s, margin, ref, more_below, printed, centered,
                                               strikes, exclude)
    above, below = _sig_window(cells, s, margin, more_below)
    o = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY) if out.ndim == 3 else out
    ostrip = _column_strip(o, f, cells, above, below, SIG_OVERRUN)[0]
    gstrip = _column_strip(gray, f, cells, above, below, SIG_OVERRUN)[0]
    kstrip = _column_strip(keep, f, cells, above, below, SIG_OVERRUN)[0] if keep is not None else None
    ink = ink_mask(gstrip, float(np.median([z.rect[3] for z in cells])), signature=True)
    _erase_strikes(ink, origin, strikes)
    bad = 0
    for x0, y0, x1, y1, n, boxes in blobs:
        if n < 2 * SIG_MIN_INK:
            continue
        sel = np.zeros((y1 - y0, x1 - x0), bool)
        for bx0, by0, bx1, by1 in boxes:
            sel[by0 - y0:by1 - y0, bx0 - x0:bx1 - x0] = True
        sel &= ink[y0:y1, x0:x1] > 0
        if kstrip is not None:
            kept = kstrip[y0:y1, x0:x1] > 0
            if (sel & kept).sum() > 0.5 * sel.sum():
                continue
            sel &= ~kept
        if (ostrip[y0:y1, x0:x1][sel] <= 8).mean() < 0.9:
            bad += 1
    if bad:
        name = cells[0].group or cells[0].name
        return f"{name}: {bad} signature(s) not fully masked"
    return ""


def find_anchor(gray: np.ndarray, f: Mapper, ref: np.ndarray, anchor, reach: tuple[float, float] = (60, 120),
                min_ncc: float = 0.5) -> tuple[np.ndarray | None, float]:
    """Locate a printed label near where the template expects it.

    Layout variants of the same form (e.g. Form 34B generators) place a
    section a line or two higher or lower. Fields tied to a label are moved
    by the label's offset, so the mask lands on the line the label names.
    Returns (offset (dx, dy) in template px, or None if not found; NCC)."""
    x, y, w, h = anchor
    rx, ry = reach
    sw, sh = int(w + 2 * rx), int(h + 2 * ry)
    ox, oy = x - rx, y - ry
    gx, gy = np.meshgrid(np.arange(sw, dtype=np.float64) + ox, np.arange(sh, dtype=np.float64) + oy)
    m = f(np.stack([gx.ravel(), gy.ravel()], 1)).astype(np.float32).reshape(sh, sw, 2)
    area = cv2.remap(gray, m[..., 0], m[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    label = ref[int(y):int(y + h), int(x):int(x + w)]
    best, where = -1.0, None
    for sc in (0.9, 1.0, 1.1):  # print and render sizes differ slightly
        lab = cv2.resize(label, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA)
        if lab.shape[0] >= area.shape[0] or lab.shape[1] >= area.shape[1]:
            continue
        res = cv2.matchTemplate(cv2.GaussianBlur(area, (3, 3), 0), cv2.GaussianBlur(lab, (3, 3), 0),
                                cv2.TM_CCOEFF_NORMED)
        _, v, _, loc = cv2.minMaxLoc(res)
        if v > best:
            # label centre found vs expected
            best, where = v, (loc[0] + lab.shape[1] / 2 - (rx + w / 2), loc[1] + lab.shape[0] / 2 - (ry + h / 2))
    if best < min_ncc:
        return None, best
    return np.float64(where), best


def shifted(f: Mapper, d: np.ndarray) -> Mapper:
    return lambda pts: f(np.asarray(pts, np.float64).reshape(-1, 2) + d)
