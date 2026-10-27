"""Forms without a template: find the PII fields from the page itself.

Older and by-election forms (34A 2017, 35A, 36A, 37A, 38A, ...) share a family
layout but differ in detail: the candidates table grows with the number of
candidates, so everything below it moves; agents tables have 5 to 24 rows, in
one or two blocks; pages arrive rotated, photographed on a table or scanned
on A3. Instead of a template per variant, the page is read:

1. Frame. Text direction (0/90/180/270 degrees) from OCR line detection and
   the line-orientation classifier, skew from the text lines, and scale from
   the height of printed text, so all later sizes are in units of text height.
   Everything below happens in that upright, deskewed frame; masks are mapped
   back to the original pixels.
2. Labels. OCR (local, see ocr.py) reads the printed labels.
3. Tables. A header cell reading "ID/Passport No.", "Tel. Contact" or
   "Signature" names a column; the column is bounded by the printed vertical
   rules around that label and cut into rows by the horizontal rules below
   it, down to where the rules stop.
4. Declaration lines. A "Signature" label outside a table names the field to
   its right, up to the next printed label on the line ("Date"). Likewise an
   "ID Number" or "Tel" label outside a table.
5. Free text. An ID or phone number written in the comments ("ID: 2503...")
   is found from the OCR text and partially masked like any other.
6. Results. Everything above "Declaration" (codes, candidates' votes,
   polling-station counts, disputed votes) is a protected region; so is
   every stamp.

Anything the reading can't resolve fails the page (human review): a PII
label without a field, a table column without rules, no "Declaration" and no
results table to protect.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

import cv2
import numpy as np

from . import masking
from . import ocr as ocr_mod
from . import stamps as stamps_mod
from . import structure
from .templates import Zone

TEXT_H = 34.0  # text-line height (OCR box, px) in the working frame: row pitch then ~ 1 text line
STAMP_R = (1.6 * TEXT_H, 4.2 * TEXT_H)  # round stamps measure 2.2-2.9 text lines in radius

ID_RE = re.compile(r"passport|(?<![a-z])[I1l]\s?\.?\s?D\s*\.?\s*(?:/|no\b|no\.|num|number)|national\s*id", re.I)
TEL_RE = re.compile(r"(?<![a-z])tel(?![a-z])|telephone|contact|phone|mobile", re.I)
SIG_RE = re.compile(r"signature|(?<![a-z])sign(?![a-z])", re.I)
DATE_RE = re.compile(r"(?<!di)date", re.I)  # not "Candidate"
REFUSAL_RE = re.compile(r"refus", re.I)
LABEL_RE = re.compile(r"(?<!di)date|signature|name|officer|polling|station|constituency|county|ward|"
                      r"passport|(?<![a-z])tel(?![a-z])|contact|(?<![a-z])I\.?D(?![a-z])", re.I)
DECL_RE = re.compile(r"^\s*declaration", re.I)
RESULTS_RE = re.compile(r"valid\s*votes|votes\s*cast|votes\s*obtained|registered\s*voters|rejected", re.I)
# handwriting in free text: "ID 25039403", "1D: 2503940 3", "07xx xxx xxx", "+254 7.."
FREE_ID_RE = re.compile(r"(?<![\d/-])((?:\d\s?){6,8}\d)(?![\d/-])")  # 7-9 digits
# "ID: 2503..." -- whatever follows the label, however OCR garbled the digits
ID_PREFIX_RE = re.compile(r"(?<![a-z])(?:[I1l|]\s?\.?\s?D|ID\s*no|ID\s*number)\s*\.?\s*[:.,;#-]?\s*(\S.{4,15})", re.I)
FREE_PHONE_RE = re.compile(r"(?<!\d)((?:\+\s?254|0)\s?[17](?:\s?\d){8})(?!\d)")


@dataclass
class Frame:
    k: int  # quarter turns clockwise that make the text upright
    skew: float  # degrees, removed after the quarter turns
    scale: float  # original px -> working px
    M: np.ndarray  # 3x3: original -> working
    size: tuple[int, int]  # working image (w, h)
    text_h: float  # median text-line height in the original (px)

    @property
    def to_image(self) -> np.ndarray:
        """3x3: working frame -> original image."""
        return np.linalg.inv(self.M)


@dataclass
class Field:
    kind: str  # "id" | "phone" | "signature"
    zones: list[Zone]  # working-frame cells (one per table row, or one)
    source: str  # "table" | "label" | "free-text"
    label: str  # the printed text it was found from
    label_box: tuple[float, float, float, float]

    def record(self, to_image) -> dict:
        x0, y0, x1, y1 = self.label_box
        quad = to_image(np.float64([[x0, y0], [x1, y0], [x1, y1], [x0, y1]]))
        return {"kind": self.kind, "source": self.source, "label": self.label, "cells": len(self.zones),
                "label_polygon": np.round(quad, 1).tolist()}


@dataclass
class Layout:
    frame: Frame
    image: np.ndarray  # working-frame image (BGR or gray)
    gray: np.ndarray
    lines: list  # OCR lines (working frame)
    fields: list[Field] = field(default_factory=list)
    protected: list[tuple[str, np.ndarray]] = field(default_factory=list)  # name, polygon (working frame)
    stamps: list = field(default_factory=list)  # stamps.Stamp, working frame
    problems: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    officer_names: set = field(default_factory=set)  # words of the officers' names, as written in the declaration


def _quad_dims(q: np.ndarray) -> tuple[float, float]:
    w = (np.linalg.norm(q[1] - q[0]) + np.linalg.norm(q[2] - q[3])) / 2
    h = (np.linalg.norm(q[3] - q[0]) + np.linalg.norm(q[2] - q[1])) / 2
    return float(w), float(h)


def find_frame(img: np.ndarray, o: ocr_mod.Ocr, work: int = 1200) -> Frame | None:
    h0, w0 = img.shape[:2]
    s0 = min(1.0, work / max(h0, w0))
    small = cv2.resize(img, None, fx=s0, fy=s0, interpolation=cv2.INTER_AREA) if s0 < 1 else img
    if small.ndim == 2:
        small = cv2.cvtColor(small, cv2.COLOR_GRAY2BGR)

    def dims(qs):
        return [(q, *_quad_dims(q)) for q in qs]

    boxes = dims(o.detect(small))
    wide = sum(w * h for _, w, h in boxes if w > 1.5 * h)
    tall = sum(w * h for _, w, h in boxes if h > 1.5 * w)
    if wide + tall == 0:
        return None
    k = 1 if tall > wide else 0
    up = np.ascontiguousarray(np.rot90(small, -k))
    boxes = dims(o.detect(up)) if k else boxes
    lines = sorted([b for b in boxes if b[1] > 2 * b[2]], key=lambda b: -b[1])[:40]
    if len(lines) < 5:
        return None
    votes = o.classify(up, [b[0] for b in lines])
    flipped = sum(1 for lab, conf in votes if lab == "180" and conf > 0.8)
    upright = sum(1 for lab, conf in votes if lab == "0" and conf > 0.8)
    if flipped > upright:
        k += 2
        up = np.ascontiguousarray(np.rot90(up, 2))
        boxes = dims(o.detect(up))
        lines = sorted([b for b in boxes if b[1] > 2 * b[2]], key=lambda b: -b[1])[:40]
    angles = [math.degrees(math.atan2(q[1][1] - q[0][1], q[1][0] - q[0][0])) for q, _, _ in lines]
    skew = float(np.clip(np.median(angles), -8, 8)) if angles else 0.0
    th = float(np.median([h for _, w, h in boxes if w > 1.5 * h])) / s0
    scale = TEXT_H / th
    # original -> working: scale, then rotate by k quarter turns clockwise and the skew
    ang = -90.0 * k + skew  # OpenCV: positive = counter-clockwise
    W, H = w0 * scale, h0 * scale
    R = cv2.getRotationMatrix2D((W / 2, H / 2), ang, 1.0)
    corners = np.float64([[0, 0, 1], [W, 0, 1], [W, H, 1], [0, H, 1]]) @ R.T
    lo, hi = corners.min(0), corners.max(0)
    R[:, 2] -= lo
    M = np.vstack([R, [0, 0, 1]]) @ np.diag([scale, scale, 1.0])
    size = (int(math.ceil(hi[0] - lo[0])), int(math.ceil(hi[1] - lo[1])))
    return Frame(k % 4, skew, scale, M, size, th)


def warp(img: np.ndarray, f: Frame) -> np.ndarray:
    if f.scale < 0.8:  # area-average first: plain warping aliases fine print when shrinking a lot
        img = cv2.resize(img, None, fx=f.scale, fy=f.scale, interpolation=cv2.INTER_AREA)
        M = f.M @ np.diag([1 / f.scale, 1 / f.scale, 1.0])
    else:
        M = f.M
    return cv2.warpAffine(img, M[:2], f.size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


# --- tables -----------------------------------------------------------------------------------

def _merge(lines, max_gap: float, tol: float = 4.0):
    """Join collinear pieces of a rule broken by handwriting, but not across
    a wide gap: two tables one above the other keep separate column rules."""
    out = []
    for l in sorted(lines, key=lambda l: (round(l.at((l.lo + l.hi) / 2) / tol), l.lo)):
        for m in out:
            mid = (l.lo + l.hi) / 2
            if abs(m.at(mid) - l.at(mid)) < tol and abs(m.slope - l.slope) < 0.03 \
                    and l.lo - m.hi <= max_gap and m.lo - l.hi <= max_gap:
                total = m.length + l.length
                m.pos = (m.pos * m.length + l.pos * l.length) / total
                m.slope = (m.slope * m.length + l.slope * l.length) / total
                m.lo, m.hi = min(m.lo, l.lo), max(m.hi, l.hi)
                break
        else:
            out.append(structure.Line(l.pos, l.slope, l.lo, l.hi))
    return out


def _rules(gray: np.ndarray, stamp_list=()):
    """Printed rules. Vertical rules must be longer than any handwritten
    stroke: table rules run the height of the table. Stamps are erased
    first (the sides of a ring look like short vertical rules)."""
    bw = _binary(gray)
    for st in stamp_list:
        cv2.circle(bw, (int(st.cx), int(st.cy)), int(st.r * 1.1 + 4), 0, -1)
    hor = cv2.morphologyEx(bw, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (60, 1)))
    ver = cv2.morphologyEx(bw, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 30)))
    hs = [l for l in _merge(structure._segments(hor, True), 1.5 * TEXT_H) if l.length >= 3 * TEXT_H
          and abs(l.slope) < 0.1]
    vs = [l for l in _merge(structure._segments(ver, False), 1.5 * TEXT_H) if l.length >= 2.5 * TEXT_H
          and abs(l.slope) < 0.1]
    return hs, vs


def _binary(gray: np.ndarray) -> np.ndarray:
    return cv2.adaptiveThreshold(cv2.GaussianBlur(gray, (3, 3), 0), 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                 cv2.THRESH_BINARY_INV, 31, 8)


def _hidden_rule(gray: np.ndarray, x0: float, x1: float, y0: float, y1: float) -> float | None:
    """x of a column rule between x0 and x1 that writing breaks in every row
    (digits run into it), found as the x where the page is dark in most rows
    from y0 to y1. Handwriting falls at random x and doesn't line up."""
    a, b = int(max(0, x0)), int(min(gray.shape[1], x1))
    ya, yb = int(max(0, y0)), int(min(gray.shape[0], y1))
    if b - a < 3 or yb - ya < 2 * TEXT_H:
        return None
    bw = _binary(gray[ya:yb, a:b])
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 7)))
    prof = (bw > 0).mean(axis=0)
    prof = np.convolve(prof, np.ones(3) / 3, mode="same")
    k = int(np.argmax(prof))
    return a + k if prof[k] >= 0.45 else None


def _column(hs, vs, x0: float, x1: float, cy: float):
    """Vertical rules left and right of a header label at height cy, or None
    if the label isn't in a table header cell: the column's rules must run
    beside the label (a label on a declaration line can sit just above a
    table, but the table's rules start below it), there must be a printed
    rule under the label (the header's bottom rule) and the rules must carry
    on down into the rows."""
    h = TEXT_H
    xm = (x0 + x1) / 2
    below = [l.at(xm) for l in hs if l.lo - 0.2 * h <= xm <= l.hi + 0.2 * h and cy + 0.3 * h < l.at(xm) < cy + 3 * h]
    if not below:
        return None
    yb = min(below)

    def x_at(v):
        return v.at(cy)
    span = [v for v in vs if cy - 3 * h <= v.lo <= cy + 0.2 * h and v.hi >= yb + 1.5 * h]
    left = [v for v in span if x_at(v) <= x0 + 0.3 * h]
    right = [v for v in span if x_at(v) >= x1 - 0.3 * h]
    if not left or not right:
        return None
    lv = _extend(vs, max(left, key=x_at), cy)
    rv = _extend(vs, min(right, key=x_at), cy)
    if x_at(rv) - x_at(lv) < (x1 - x0) * 0.9:
        return None
    if abs(lv.lo - rv.lo) > 3.5 * h or abs(lv.hi - rv.hi) > 6 * h:
        return None  # a table's column rules start and end together
    return lv, rv


def _extend(vs, v, cy: float):
    """The rule extended down over the pieces in line with it below (a rule
    broken where handwriting crosses it, even across a wide gap). Not
    upwards: a table above may have a rule at the same x."""
    pieces = [u for u in vs if abs(u.at(cy) - v.at(cy)) < 5 and abs(u.slope - v.slope) < 0.03
              and u.lo >= v.lo - 0.5 * TEXT_H]
    return structure.Line(v.pos, v.slope, v.lo, max(u.hi for u in pieces + [v]))


def _rows(hs, lv, rv, cy: float, h: float) -> list[tuple[float, float]]:
    """Row bands (top, bottom) under a header: the table's horizontal rules,
    from the header's bottom rule down to where the column's rules end. Row
    rules run the width of the table, so one broken in this column (writing
    across it) is still found in the neighbouring columns; a row rule missing
    altogether is filled in from the row pitch."""
    xm = (lv.at(cy) + rv.at(cy)) / 2
    bottom = max(lv.hi, rv.hi) + 0.5 * h  # the column's rules end with the table
    near = [l for l in hs if l.lo - 6 * h <= xm <= l.hi + 6 * h]
    ys = sorted(l.at(xm) for l in near if cy + 0.3 * h < l.at(xm) <= bottom)
    merged: list[float] = []
    for y in ys:  # double-drawn rules, or one rule found twice
        if merged and y - merged[-1] < 0.35 * h:
            continue
        merged.append(y)
    if len(merged) < 2:
        return []
    gaps = np.diff(merged)
    pitch = float(np.median(gaps))
    rows = []
    for a, b in zip(merged, merged[1:]):
        n = int(round((b - a) / pitch))
        if n > 4:
            break
        n = max(n, 1)
        rows += [(a + (b - a) * k / n, a + (b - a) * (k + 1) / n) for k in range(n)]
    # slivers: a stray line just under the table
    return [(a, b) for a, b in rows if b - a >= 0.5 * pitch and b <= max(lv.hi, rv.hi) + 0.3 * h]


def _tables(L: Layout, hs, vs) -> set[int]:
    """Find PII columns from their header labels. Returns the indexes of the
    OCR lines used as table headers."""
    used: set[int] = set()
    seen: list[tuple[str, float, float, float]] = []
    for i, ln in enumerate(L.lines):
        for kind, rx in (("id", ID_RE), ("phone", TEL_RE), ("signature", SIG_RE)):
            if kind == "signature" and REFUSAL_RE.search(ln.text):
                continue
            sp = ln.find(rx)
            if not sp:
                continue
            col = _column(hs, vs, sp[0], sp[1], ln.cy)
            if col is None:
                continue
            lv, rv = col
            xl, xr = lv.at(ln.cy), rv.at(ln.cy)
            if (xr - xl) > 12 * TEXT_H:  # not a table cell: a wide box around a declaration
                continue
            used.add(i)
            if any(k == kind and abs(a - xl) < TEXT_H and abs(c - ln.cy) < 4 * TEXT_H
                   for k, a, b, c in seen):
                continue  # "Tel." / "Contact" on two lines of one header cell
            # Another header inside the column: the rule between them is
            # broken in every row (writing runs into it). Look for it by
            # projection; without it the page goes to review.
            others = [o for o in L.lines if o is not ln and abs(o.cy - ln.cy) < TEXT_H
                      and (o.find(ID_RE) or o.find(TEL_RE) or o.find(SIG_RE) or o.find(DATE_RE))
                      and xl + 3 < (o.box[0] + o.box[2]) / 2 < xr - 3
                      and not (sp[0] - TEXT_H <= (o.box[0] + o.box[2]) / 2 <= sp[1] + TEXT_H)]
            bottom_y = max(lv.hi, rv.hi)
            for o in others:
                if (o.box[0] + o.box[2]) / 2 > sp[1]:  # to the right: find the right rule
                    x = _hidden_rule(L.gray, sp[1], o.box[0], ln.cy + TEXT_H, bottom_y)
                    if x is not None:
                        rv = structure.Line(x, 0.0, rv.lo, rv.hi)
                else:
                    x = _hidden_rule(L.gray, o.box[2], sp[0], ln.cy + TEXT_H, bottom_y)
                    if x is not None:
                        lv = structure.Line(x, 0.0, lv.lo, lv.hi)
                if x is None:
                    break
            xl, xr = lv.at(ln.cy), rv.at(ln.cy)
            if others and x is None:
                L.problems.append(f"'{ln.text}' column also holds '{others[0].text}': a column rule is missing")
                used.add(i)
                continue
            rows = _rows(hs, lv, rv, ln.cy, TEXT_H)
            if not rows:
                L.problems.append(f"'{ln.text}' header at y={ln.cy:.0f}: no table rows found under it")
                continue
            seen.append((kind, xl, xr, ln.cy))
            zones = []
            for r, (y0, y1) in enumerate(rows):
                ym = (y0 + y1) / 2
                x0, x1 = lv.at(ym) + 3, rv.at(ym) - 3
                name = f"{kind}@{xl:.0f},{ln.cy:.0f}"
                zones.append(Zone(f"{name}[{r}]", (x0, y0, x1 - x0, y1 - y0),
                                  "signature" if kind == "signature" else "partial",
                                  10 if kind == "phone" else (8 if kind == "id" else 0), name, r, "",
                                  {"phone": "phone", "id": "id"}.get(kind, ""),
                                  (8.0, 30.0) if kind == "signature" else (6.0, 12.0)))
            L.fields.append(Field(kind, zones, "table", ln.text, ln.box))
    return used


# --- declaration: the officers' lines -----------------------------------------------------

def _date_ish(text: str) -> int | None:
    """Index where a handwritten date starts in OCR text ("18-03-2021",
    "7/11/2017", "15-I2-2o2O"), or None."""
    m = re.search(r"\d[\dOoIlQ]?\s*[-/.~,]+\s*\S{1,3}?\s*[-/.~,]+\s*[\dOoIlQ]{1,4}|\d{1,2}\s*[-/.~]\s*20\d\d", text)
    return m.start() if m and sum(c.isdigit() for c in m.group(0)) >= 4 else None


def _sig_labels(ln) -> list[tuple[int, int]]:
    """"Signature" in OCR text, allowing for misreads ("sinature", "Shature",
    "Signatu" cut off by a stamp). Not "Refusal to Sign"."""
    if REFUSAL_RE.search(ln.text):
        return []
    out = [(m.start(), m.end()) for m in SIG_RE.finditer(ln.text)]
    for i, j, d in ln.fuzzy("signature", 3):
        if d <= 2 or ln.text[i:j].lower().endswith(("ure", "tur")):
            if all(j <= a or i >= b for a, b in out):
                out.append((i, j))
    return out


def _date_labels(ln, misread: bool = False) -> list[tuple[int, int]]:
    """"Date" labels. A misread one ("Pate", "bate", "Dat") counts if a date
    follows it, or with ``misread`` (asked for on a line already known to be
    an officer's line, where "GATE" in a station name can't be meant)."""
    out = [(m.start(), m.end()) for m in DATE_RE.finditer(ln.text)]
    for i, j, d in ln.fuzzy("date", 1):
        if ln.text[max(0, i - 2):i].lower() == "di":
            continue
        if (misread or _date_ish(ln.text[j:]) is not None) and all(j <= a or i >= b for a, b in out):
            out.append((i, j))
    return out


def _declaration(L: Layout, top: float, bottom: float) -> set[int]:
    """Signature fields on the officers' lines of the declaration
    ("Presiding Officer ____ Signature ____ Date ____").

    OCR often can't read these lines whole: a printed label runs into the
    handwritten name next to it. The "Date" label and the date written after
    it are read reliably, so each marks an officer's line, and the signature
    field ends just before "Date". It starts where the "Signature" label ends
    -- read on this line, or on another officer line (the labels are printed
    one above the other) -- or else where the printed underline in front of
    "Date" begins, after the gap where the label is printed. Returns the OCR
    lines used."""
    h = TEXT_H
    region = [(i, ln) for i, ln in enumerate(L.lines) if top < ln.cy < bottom]
    rows: list[dict] = []

    def row(cy):
        for r in rows:
            if abs(r["cy"] - cy) < 0.6 * h:
                return r
        rows.append({"cy": cy, "labels": [], "values": [], "sigs": [], "misread": []})
        return rows[-1]

    used = set()
    for i, ln in region:
        for a, b in _date_labels(ln):
            row(ln.cy)["labels"].append(ln.span(a, b)[0])
            used.add(i)
        k = _date_ish(ln.text)
        if k is not None:
            row(ln.cy)["values"].append(ln.span(k, k + 1)[0])
        for a, b in _sig_labels(ln):
            row(ln.cy)["sigs"].append(ln.span(a, b)[1])
            used.add(i)
        for a, b in _date_labels(ln, misread=True):
            row(ln.cy)["misread"].append(ln.span(a, b)[0])
        if re.search(r"presid|officer|deputy", ln.text, re.I) and not stamps_mod.is_stamp_text(ln.text):
            row(ln.cy)["officer"] = True
    # An officer's line has a "Signature" or "Date" label on it; a date
    # alone counts only on a line that names an officer.
    officer_rows = [r for r in rows if r["sigs"] or r["labels"] or (r["values"] and r.get("officer"))]
    # The officers' names, as written on their lines (left of "Signature"):
    # their sign-off under the comments is found by them.
    for r in officer_rows:
        end = min(r["sigs"]) if r["sigs"] else min(r["labels"] + r["values"], default=None)
        for _, ln in region:
            if abs(ln.cy - r["cy"]) >= 0.6 * h:
                continue
            # names are written in capitals; OCR runs label and name together
            # ("PresidingOfficerMAYGLADYS"), so take the runs of capitals
            for mm in re.finditer(r"[A-Z]{3,}", ln.text):
                word = mm.group(0)
                if re.fullmatch(r"PRESIDING|DEPUTY|OFFICERS?|SIGNATURE|SIGN|DATE", word):
                    continue
                if end is not None and ln.span(*mm.span())[0] >= end:
                    continue
                L.officer_names.add(word)
    label_x = [x for r in officer_rows for x in r["labels"]]
    sig_x = [x for r in officer_rows for x in r["sigs"]]
    for r in sorted(officer_rows, key=lambda r: r["cy"]):
        cy = r["cy"]
        if r["sigs"]:
            start, how = max(r["sigs"]), "'Signature' label"
        elif sig_x:
            start, how = float(np.median(sig_x)), "'Signature' label on the other officer line"
        else:
            L.problems.append(f"declaration line at y={cy:.0f}: no 'Signature' label read")
            continue
        right = [x for x in r["labels"] if x > start]
        maybe = [x for x in r["misread"] if x > start + 1.5 * h] if r["sigs"] else []
        if right:
            end = min(right)
        elif maybe:
            end, how = min(maybe), how + "; 'Date' read approximately"
        elif [x for x in label_x if x > start]:
            end = float(np.median([x for x in label_x if x > start]))
        elif [x for x in r["values"] if x > start]:
            end = min(x for x in r["values"] if x > start) - 2.6 * h  # "Date" printed just before the date
        else:
            end = start + 9 * h
            how += "; no 'Date' label: field taken as 9 text heights"
        x0, x1 = start + 0.15 * h, end - 0.15 * h
        if x1 - x0 < 1.5 * h:
            L.problems.append(f"declaration line at y={cy:.0f}: no room for a signature before 'Date'")
            continue
        # The field band is the officer's line; writing centred in it is the
        # signature (masked whole, however far it reaches up or down), writing
        # centred above or below belongs to the neighbouring lines.
        z = Zone(f"signature@{x0:.0f},{cy:.0f}", (x0, cy - 0.75 * h, x1 - x0, 1.3 * h), "signature",
                 search=(0.9 * h, 0.5 * h))
        L.fields.append(Field("signature", [z], "declaration", how, (x0, cy - 0.5 * h, x1, cy + 0.5 * h)))
    named = [r for r in rows if r.get("officer")]
    if len(officer_rows) < min(2, len(named)):
        L.problems.append(f"declaration: {len(named)} officer(s) named but {len(officer_rows)} signature line(s) found")
    return used


# --- labelled fields outside tables -----------------------------------------------------------

def _next_label_x(L: Layout, ln, after: float) -> float | None:
    """x where the next printed label on the same text line starts."""
    best = None
    for o in L.lines:
        if abs(o.cy - ln.cy) > 0.6 * ln.height:
            continue
        for m in LABEL_RE.finditer(o.text):
            x0, _ = o.span(m.start(), m.end())
            if x0 > after + 0.3 * ln.height and (best is None or x0 < best):
                best = x0
    return best


def _labels(L: Layout, used: set[int], top: float) -> None:
    right = max((ln.box[2] for ln in L.lines), default=L.frame.size[0])
    for i, ln in enumerate(L.lines):
        if i in used or ln.cy < top:
            continue
        for kind, rx in (("signature", SIG_RE), ("id", ID_RE), ("phone", TEL_RE)):
            if kind == "signature" and REFUSAL_RE.search(ln.text):
                continue
            m = rx.search(ln.text)
            if not m:
                continue
            if kind != "signature" and re.search(r"\d{5}", ln.text[m.end():]):
                continue  # digits right after the label in the same OCR line: free text, handled below
            x0, x1 = ln.span(m.start(), m.end())
            h = ln.height
            end = _next_label_x(L, ln, x1)
            fx0 = x1 + 0.15 * h
            fx1 = (end - 0.15 * h) if end else min(right, x1 + 14 * h)
            if fx1 - fx0 < 1.5 * h:
                L.problems.append(f"'{ln.text}': no room for the field after the label")
                continue
            if kind == "signature":
                rect = (fx0, ln.cy - 0.9 * h, fx1 - fx0, 1.5 * h)
                z = Zone(f"signature@{fx0:.0f},{ln.cy:.0f}", rect, "signature", search=(0.8 * h, 0.4 * h))
            else:
                rect = (fx0, ln.cy - 0.7 * h, fx1 - fx0, 1.3 * h)
                z = Zone(f"{kind}@{fx0:.0f},{ln.cy:.0f}", rect, "partial", 10 if kind == "phone" else 8,
                         rule=kind, search=(0.3 * h, 0.3 * h))
            L.fields.append(Field(kind, [z], "label", ln.text, ln.box))


def _covered(L: Layout, x: float, y: float, h: float) -> bool:
    """Is (x, y) inside a field found already, or inside a table's rows
    (numbers written in a table belong to its columns)?"""
    rows = [z.rect for fl in L.fields if fl.source == "table" for z in fl.zones]
    for fl in L.fields:
        for zx, zy, zw, zh in (z.rect for z in fl.zones):
            if zx - 0.5 * h <= x <= zx + zw + 0.5 * h and zy - 0.5 * h <= y <= zy + zh + 0.5 * h:
                return True
    same_row = [(zx, zx + zw) for zx, zy, zw, zh in rows if zy - 0.3 * h <= y <= zy + zh + 0.3 * h]
    return bool(same_row) and min(a for a, _ in same_row) - 4 * h <= x <= max(b for _, b in same_row) + 4 * h


def _date_digits(d: str) -> bool:
    """Digits that read as a date written without separators (15122020,
    1512020 for 15/1/2020)."""
    for dl in (1, 2):
        for ml in (1, 2):
            for yl in (2, 4):
                if dl + ml + yl != len(d):
                    continue
                day, mon, yr = int(d[:dl]), int(d[dl:dl + ml]), int(d[dl + ml:])
                year_ok = 1990 <= yr <= 2040 if yl == 4 else yr <= 40
                if 1 <= day <= 31 and 1 <= mon <= 12 and year_ok:
                    return True
    return False


def _free_text(L: Layout, used: set[int], top: float) -> None:
    for i, ln in enumerate(L.lines):
        if i in used or ln.cy < top:
            continue
        hits = []
        for m in ID_PREFIX_RE.finditer(ln.text):
            if sum(c.isdigit() for c in m.group(1)) >= 4:
                hits.append(("id", m.start(1), m.end(1), m.group(0)))
        for kind, rx in (("id", FREE_ID_RE), ("phone", FREE_PHONE_RE)):
            for m in rx.finditer(ln.text):
                digits = re.sub(r"\D", "", m.group(1))
                if kind == "id" and _date_digits(digits):
                    continue
                hits.append((kind, m.start(1), m.end(1), m.group(0)))
        for kind, a, b, text in hits:
            x0, x1 = ln.span(a, b)
            h = ln.height
            if _covered(L, (x0 + x1) / 2, ln.cy, h):
                continue
            rect = (x0 - 0.3 * h, ln.cy - 0.7 * h, x1 - x0 + 0.6 * h, 1.4 * h)
            z = Zone(f"{kind}-text@{x0:.0f},{ln.cy:.0f}", rect, "partial", 10 if kind == "phone" else 8,
                     rule=kind, search=(0.2 * h, 0.2 * h))
            L.fields.append(Field(kind, [z], "free-text", text, ln.box))
            L.notes.append(f"{kind} number written in free text ('{text}') partially masked")


CAPS_WORD = re.compile(r"[A-Z]{4,}")


def _free_signatures(L: Layout, top: float) -> None:
    """Signatures written outside any labelled field: an officer signing off
    under the comments (name, ID, signature, date).

    Next to the officer's own name (as written in the declaration) every
    piece of writing that OCR can't read confidently is taken for a
    signature: names, dates and numbers read well, signatures don't. Away
    from it, only joined-up writing next to a name, a date or an ID is."""
    h = TEXT_H
    W, H = L.frame.size
    y0 = int(max(0, top))
    if H - y0 < h:
        return
    # pen ink, without rules, strike lines or the security print
    ink = masking.ink_mask(L.gray[y0:], h, signature=True)
    rules = cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (int(3 * h), 1)))
    ink = cv2.subtract(ink, cv2.dilate(rules, np.ones((3, 3), np.uint8)))
    for st in L.stamps:
        cv2.circle(ink, (int(st.cx), int(st.cy - y0)), int(st.r * 1.1 + 4), 0, -1)
    lines = [ln for ln in L.lines if ln.cy >= top and not stamps_mod.is_stamp_text(ln.text)]
    vetoed, cues, zones = [], [], []
    for ln in lines:
        if re.search(r"refus|comment|presiding\s*officer", ln.text, re.I) or PRINTED_LINE.match(ln.text):
            q = ln.quad - np.float32([0, y0])
            cv2.fillPoly(ink, [np.round(q).astype(np.int32)], 0)  # printed labels
            continue
        # Stretches OCR reads as a name in capitals, a number or "N/A" are
        # writing, not signatures -- only those stretches, not the whole line,
        # which may run on into a signature.
        spans = [m.span() for m in re.finditer(r"[A-Z][A-Z.'-]{3,}", ln.text)]
        spans += [m.span() for m in re.finditer(r"\d[\d\s/.-]{3,}\d", ln.text)]
        if re.fullmatch(r"\W*(?:n\s*/\s*a|nil|none|na)\W*", ln.text, re.I):
            spans.append((0, len(ln.text)))
        for a, b in spans:
            probs = ln.probs[a:b] if len(ln.probs) == len(ln.text) else [ln.score]
            if float(np.mean(probs)) < 0.7:
                continue  # OCR guessing (a signature read as "1015"): no proof of writing
            x0, x1 = ln.span(a, b)
            vetoed.append((x0, ln.box[1], x1, ln.box[3]))
        if (re.search(r"[A-Z]{4,}|\d{4,}", ln.text) or re.search(r"sign|(?<![a-z])I\.?D(?![a-z])", ln.text, re.I)) \
                and not re.fullmatch(r"\W*(?:n\s*/\s*a|nil|none|na)\W*", ln.text, re.I):
            cues.append(ln.box)  # a name, a number or date, "sign", "ID"
        words = re.findall(r"[A-Z]{3,}", ln.text.upper())
        if any(_name_match(w, n) for w in words for n in L.officer_names):
            bx0, by0, bx1, by1 = ln.box
            zones.append((bx0 - 2 * h, by0 - 0.5 * h, bx1 + 10 * h, by1 + 3.5 * h))
    ink = cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(ink)
    groups = cv2.dilate(ink, cv2.getStructuringElement(cv2.MORPH_RECT, (int(0.3 * h), 1)))
    m, wlab, wst, _ = cv2.connectedComponentsWithStats(groups)

    def inside(cx, cy, boxes):
        return any(bx0 - 4 <= cx <= bx1 + 4 and by0 - 4 <= cy <= by1 + 4 for bx0, by0, bx1, by1 in boxes)

    def near(box, boxes, dx, dy):
        x0, y0_, x1, y1 = box
        return any(bx0 - dx <= x1 and x0 <= bx1 + dx and by0 - dy <= y1 and y0_ <= by1 + dy
                   for bx0, by0, bx1, by1 in boxes)

    def read_well(box) -> bool:
        x0, y0_, x1, y1 = box
        cy = (y0_ + y1) / 2
        chars = [p for ln in lines if ln.box[1] - 0.3 * h <= cy <= ln.box[3] + 0.3 * h
                 for _, p in ln.chars_in(x0, x1)]
        return len(chars) >= max(2, 0.4 * (x1 - x0) / h) and float(np.mean(chars)) >= 0.8

    for k in range(1, m):
        x, y, w, hh, _ = wst[k]
        if w < 1.0 * h or hh < 0.5 * h:
            continue
        parts = [j for j in np.unique(lab[y:y + hh, x:x + w][wlab[y:y + hh, x:x + w] == k]) if j and st[j][4] >= 20]
        if not parts:
            continue
        box = (x, y + y0, x + w, y + y0 + hh)
        cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        if inside(cx, cy, vetoed) or _covered(L, cx, cy, h):
            continue  # a name in capitals, a date, a number; or masked already
        if inside(cx, cy, zones):
            if read_well(box):
                continue
            why = "next to the officer's name, not legible as text"
        else:
            widest = max(st[j][2] for j in parts)
            joined = len(parts) <= 0.9 * (w / h) + 1 and widest >= 0.6 * w and w >= 1.5 * h
            if not (joined and near(box, cues, 8 * h, 2.5 * h)):
                continue
            why = "joined strokes next to a name, date or ID"
        z = Zone(f"signature-text@{x:.0f},{cy:.0f}", (box[0] - 4, box[1] - 4, w + 8, hh + 8), "signature",
                 search=(0.3 * h, 0.3 * h))
        L.fields.append(Field("signature", [z], "free-text", f"unlabelled signature ({why})", box))
        L.notes.append(f"unlabelled handwriting at y={cy:.0f} taken for a signature ({why})")


def _name_match(word: str, name: str) -> bool:
    """Same name, allowing for OCR misreading it in both places."""
    a, b = word.upper(), name.upper()
    if min(len(a), len(b)) < 4:
        return False
    if len(a) >= 6 and len(b) >= 6 and (a in b or b in a):
        return True
    return ocr_mod._edits(a, b) <= (1 if min(len(a), len(b)) < 5 else 2 if min(len(a), len(b)) < 8 else 3)


# --- printed text inside fields -----------------------------------------------------------

PRINTED_LINE = re.compile(r"^[\s.:,/()_-]*(?:no|name\s*of\s*candidate(?:\s*or\s*agent)?|or\s*agent|agent|"
                          r"id\s*/?\s*passport(?:\s*no)?|passport|party\s*name|independent(?:\s*candidate)?|"
                          r"candidate|tel\.?(?:\s*contact)?|contact|signature|date)[\s.:,/()_-]*$", re.I)


def _x_fill(text: str) -> bool:
    """Typed "XXXXXXXX" that form generators put in unused rows."""
    t = re.sub(r"\s", "", text)
    return len(t) >= 5 and sum(c in "xX×*" for c in t) >= 0.8 * len(t)


def printed(L: Layout) -> tuple[np.ndarray, np.ndarray]:
    """What OCR shows to be print, not writing, where a field could reach it:
    typed "XXXX" fill and lines that are nothing but a printed label. Returns
    (mask, reference image: the page's own pixels there, white elsewhere),
    both in the working frame. Masking leaves this text alone."""
    mask = np.zeros(L.gray.shape, np.uint8)
    for ln in L.lines:
        if _x_fill(ln.text) or PRINTED_LINE.match(ln.text):
            q = ln.quad + np.float32([[-2, -2], [2, -2], [2, 2], [-2, 2]])
            cv2.fillPoly(mask, [np.round(q).astype(np.int32)], 255)
    return mask, np.where(mask > 0, L.gray, 255).astype(np.uint8)


# --- whole page --------------------------------------------------------------------------------

def _read_band(o: ocr_mod.Ocr, work: np.ndarray, y0: float, y1: float, zoom: float = 2.5) -> list:
    """OCR a horizontal band of the page at ``zoom`` x. The band is padded to
    the detector's minimum side, so it is never scaled up further."""
    a, b = int(max(0, y0)), int(min(work.shape[0], y1))
    if b - a < TEXT_H:
        return []
    band = cv2.resize(work[a:b], None, fx=zoom, fy=zoom, interpolation=cv2.INTER_CUBIC)
    if band.ndim == 2:
        band = cv2.cvtColor(band, cv2.COLOR_GRAY2BGR)
    pad = max(0, 740 - band.shape[0])
    band = cv2.copyMakeBorder(band, 0, pad, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    lines = o.read(band)
    for ln in lines:
        ln.quad = ln.quad / zoom + np.float32([0, a])
    return [ln for ln in lines if ln.cy < b]


def analyze(img: np.ndarray, o: ocr_mod.Ocr | None = None) -> Layout | str:
    """Read the page. Returns the layout, or why it couldn't be read."""
    o = o or ocr_mod.engine()
    f = find_frame(img, o)
    if f is None:
        return "no printed text found to orient the page"
    work = warp(img, f)
    gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY) if work.ndim == 3 else work
    lines = o.read(work if work.ndim == 3 else cv2.cvtColor(work, cv2.COLOR_GRAY2BGR))
    L = Layout(f, work, gray, lines)
    L.stamps = stamps_mod.find(work, STAMP_R, lines)
    hs, vs = _rules(gray, L.stamps)

    # Results end where the declaration starts; failing that, below the last
    # line that names a count.
    W, H = f.size
    decl = [ln for ln in lines if DECL_RE.search(ln.text)]
    counts = [ln for ln in lines if RESULTS_RE.search(ln.text)]
    if decl:
        boundary = min(ln.box[1] for ln in decl) - 0.3 * decl[0].height
    elif counts:
        boundary = max(ln.box[3] for ln in counts) + 0.5 * counts[0].height
        L.notes.append("no 'Declaration' label: results protected down to the last count line")
    else:
        L.problems.append("can't tell where the results end (no 'Declaration', no count labels)")
        return L
    L.protected.append(("results", np.float64([[0, 0], [W, 0], [W, boundary], [0, boundary]])))

    used = _tables(L, hs, vs)
    heads = [fl.label_box[1] for fl in L.fields if fl.source == "table"]  # the agents table's header
    decl_bottom = min([y for y in heads if y > boundary], default=boundary + 9 * TEXT_H) - 0.3 * TEXT_H
    # Re-read the declaration on its own. At page scale OCR's text detector
    # misses printed labels that run into handwriting ("Signature" next to a
    # signature); a band of the page is upscaled before detection and they
    # are found. Both readings are kept: each finds things the other misses.
    L.lines = L.lines + _read_band(o, work, boundary, decl_bottom)
    used |= _declaration(L, boundary, decl_bottom)
    _labels(L, used, boundary)
    # Free text below the tables: re-read it (as the declaration), then look
    # for numbers and signatures written there.
    bottoms = [z.rect[1] + z.rect[3] for fl in L.fields if fl.source == "table" for z in fl.zones]
    free_top = max(bottoms) if bottoms else decl_bottom
    L.lines = L.lines + _read_band(o, work, free_top, H)
    _free_text(L, used, boundary)
    _free_signatures(L, free_top)
    kinds = {fl.kind for fl in L.fields if fl.source == "table"}
    if kinds and kinds != {"id", "phone", "signature"}:
        L.problems.append(f"agents table: found only {', '.join(sorted(kinds))} columns")
    if not L.fields:
        L.problems.append("no PII fields recognised")
    return L
