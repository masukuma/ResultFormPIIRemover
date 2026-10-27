"""Per-page and per-document redaction.

Fail-closed: a page is only released if it aligned confidently to a known
template, its tables were placed against their printed rules (or the zones
widened), no mask touches a protected results region, and the leak check
finds nothing left readable. Anything else goes to human review.

Masks are drawn on a separate canvas, snapped to the 8x8 block grid that the
JPEG proof works on (proof.py), and only then applied. The page result
carries that block grid; nothing else about the image is ever changed.

Stamps are never masked. Every page is searched for them (stamps.py); each
becomes a protected region, and no block touching one is blacked out, even
where a signature runs across it.

Pages no template matches are read instead (generic.py), if enabled: the
same masking, checks and proof, with the fields found from the page's own
labels and rules.
"""

from __future__ import annotations

import dataclasses
import random
import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from . import masking, proof, stamps, structure
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
    protected: list[dict] = field(default_factory=list)  # results regions and stamps, as image polygons
    H: np.ndarray | None = None  # template -> image (page alignment)
    layout: dict | None = None  # template-free pages: what the fields were found from


@dataclass
class DocResult:
    pages: list[PageResult] = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return bool(self.pages) and all(p.ok for p in self.pages)


def _apply(img: np.ndarray, canvas: np.ndarray, keep: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Canvas (0 = hide) -> block grid, and the image with those blocks black.
    No block touching ``keep`` (nonzero: stamps) is blacked out."""
    blocks = proof.to_blocks(canvas == 0)
    if keep is not None:
        blocks &= ~proof.to_blocks(keep > 0)
    out = img.copy()
    out[proof.to_pixels(blocks, img.shape[:2])] = 0
    return blocks, out


STAMP_MARGIN = 6.0  # px kept clear around a stamp, plus 4% of its radius


def protect_stamps(img: np.ndarray, canvas: np.ndarray, found: list, r_range,
                   maps=None) -> tuple[list, np.ndarray, list[str]]:
    """Add stamps found around the masks (a permissive search: a stamp under a
    mask is the miss that matters), and lift the masks off every stamp.
    Returns (stamps, keep mask, notes)."""
    hide = canvas == 0
    extra = stamps.near(img, hide, r_range, found, maps) if hide.any() else []
    found = list(found) + extra
    keep = np.zeros(canvas.shape, np.uint8)
    for st in found:
        cv2.fillPoly(keep, [np.round(st.polygon(STAMP_MARGIN + 0.04 * st.r)).astype(np.int32)], 255)
    # Masks are applied in whole 8x8 blocks, so what stays clear is every
    # block touching a stamp; the checks must allow for exactly that.
    keep = proof.to_pixels(proof.to_blocks(keep > 0), canvas.shape).astype(np.uint8) * 255
    notes = []
    covered = int((hide & (keep > 0)).sum())
    if covered:
        notes.append(f"masks lifted off {covered} px inside stamps: writing on a stamp stays visible")
    canvas[keep > 0] = 255
    return found, keep, notes


def _stamp_records(found: list) -> list[dict]:
    return [st.record() for st in found]


def _scale(H: np.ndarray, pt) -> float:
    """Local scale (image px per template px) of a homography at ``pt``."""
    p = np.float64([[pt, (pt[0] + 1, pt[1]), (pt[0], pt[1] + 1)]])
    q = cv2.perspectiveTransform(p, H)[0]
    (ax, ay), (bx, by) = q[1] - q[0], q[2] - q[0]
    return float(np.sqrt(abs(ax * by - ay * bx)))


def redact_page(img: np.ndarray, templates: list[Template], s: AlignSettings,
                ms: MaskSettings | None = None, rng: random.Random | None = None,
                form: str | None = None, strict: bool = False,
                fallback_margin: tuple[float, float] = (50.0, 200.0), verify: bool = True,
                generic: bool = False) -> PageResult:
    t0 = time.perf_counter()
    ms = ms or MaskSettings()
    rng = rng or random.SystemRandom()  # unpredictable: hidden digits differ per cell and per run
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    a = best_alignment(gray, templates, s, form) if templates else None

    def fail(reason, report=(), notes=()):
        return PageResult(False, None, a.template.name if a and a.template else "", a.inliers if a else 0,
                          a.score if a else 0.0, reason, (time.perf_counter() - t0) * 1000, list(report), list(notes))

    if a is None or not a.ok:
        if generic:
            return redact_generic(img, ms, rng, verify, why=a.reason if a else "no templates")
        return fail(a.reason if a else "no templates")
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

    def base_mapper(zones) -> masking.Mapper:
        if zones[0].region and zones[0].region not in failed:
            return snaps[zones[0].region].map
        x0 = min(z.rect[0] for z in zones)
        x1 = max(z.rect[0] + z.rect[2] for z in zones)
        cy = float(np.mean([z.rect[1] + z.rect[3] / 2 for z in zones]))
        return masking.homography(a.local(((x0 + x1) / 2, cy)))

    # Zones tied to a printed label move with it (layout variants); a label
    # that can't be found fails the page.
    # Each distinct label is searched once (through the zone's own mapping)
    # and the offset is shared by every zone tied to it.
    offsets: dict[tuple, np.ndarray] = {}
    lost = []
    for z in a.template.zones:
        if z.anchor and z.anchor not in offsets:
            d, ncc = masking.find_anchor(gray, base_mapper([z]), a.template.image, z.anchor)
            if d is None:
                lost.append(f"{z.group or z.name}: printed label not found (best match {ncc:.2f})")
            else:
                offsets[z.anchor] = d
    if lost:
        return fail("; ".join(lost))
    notes += [f"label at {list(k[:2])} found {np.hypot(*d):.0f}px from its template position"
              for k, d in offsets.items() if np.hypot(*d) > 25]

    def mapper(zones) -> masking.Mapper:
        f = base_mapper(zones)
        return masking.shifted(f, offsets[zones[0].anchor]) if zones[0].anchor else f

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
                 if verify and cells[0].region  # table columns; single underlined fields have no column rules
                 and (r := masking.check_column_edges(gray, mapper(cells), cells,
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

    # Stamps: never masked.
    t = a.template
    tw, th = t.size
    k = _scale(a.H, (tw / 2, th / 2)) * (t.page_long_side or max(tw, th)) / 2340.0
    r_range = (40 * k, 135 * k)  # stamp radius: 40-135 px on a 2340 px page
    maps = stamps.ink_maps(img)
    found, keep, stamp_notes = protect_stamps(img, canvas, stamps.find(img, r_range, maps=maps), r_range, maps)
    notes += stamp_notes
    blocks, out = _apply(img, canvas, keep)

    # Results regions (votes, totals, station codes, QR) must not lose a single
    # block: a mask there would hide a result, which no PII rule needs.
    protected, prot_mask = _stamp_records(found), np.zeros(gray.shape, np.uint8)
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
                                                            margin(cells), keep=keep))]
    if leaks:
        return fail("leak check: " + "; ".join(leaks), report, notes)
    return PageResult(True, out, a.template.name, a.inliers, a.score, "",
                      (time.perf_counter() - t0) * 1000, report, notes, blocks, protected, a.H)


def redact_generic(img: np.ndarray, ms: MaskSettings | None = None, rng: random.Random | None = None,
                   verify: bool = True, why: str = "") -> PageResult:
    """A page no template matched: read its layout (generic.py), then mask
    and check exactly as for a template."""
    from . import generic, ocr

    t0 = time.perf_counter()
    ms = ms or MaskSettings()
    rng = rng or random.SystemRandom()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    prefix = f"no template matched ({why}); " if why else ""

    def fail(reason, report=(), notes=(), layout=None):
        return PageResult(False, None, "generic", 0, 0.0, prefix + reason, (time.perf_counter() - t0) * 1000,
                          list(report), list(notes), layout=layout)

    if not ocr.available():
        return fail("reading unknown layouts needs the OCR models: pip install 'resultform-pii[generic]'")
    L = generic.analyze(img)
    if isinstance(L, str):
        return fail(L)
    to_image = L.frame.to_image
    f = masking.homography(to_image)
    layout = {
        "frame": {"quarter_turns": L.frame.k, "skew_deg": round(L.frame.skew, 2), "scale": round(L.frame.scale, 5),
                  "to_image": np.round(to_image, 8).tolist()},
        "fields": [fl.record(f) for fl in L.fields],
    }
    if L.problems:
        return fail("; ".join(L.problems), notes=L.notes, layout=layout)

    def field_settings(fl) -> MaskSettings:
        # Glyphs are told from specks by size: a share of the row height for
        # template tables, but rows on these forms range from 1 to 1.4 text
        # lines, so here it is a share of the printed text height.
        row_h = float(np.median([z.rect[3] for z in fl.zones]))
        return dataclasses.replace(ms, min_glyph_h=min(ms.min_glyph_h, 0.3 * generic.TEXT_H / row_h))

    print_mask, print_ref = generic.printed(L)
    canvas = np.full(gray.shape, 255, np.uint8)  # 0 = hide
    report = []
    for fl in L.fields:
        if fl.kind == "signature":
            # a field on a text line takes only writing centred on it, not the
            # lines above and below that its search window reaches
            report += masking.signature_column(canvas, gray, f, fl.zones, ms, printed=print_mask,
                                               centered=fl.source != "table")
        else:
            report += masking.partial_column(canvas, gray, f, fl.zones, field_settings(fl), rng, ref=print_ref)

    r_range = tuple(v / L.frame.scale for v in generic.STAMP_R)
    on_page = [stamps.Stamp(*f(np.float64([[st.cx, st.cy]]))[0], st.r / L.frame.scale, st.coverage, st.how)
               for st in L.stamps]
    found, keep, notes = protect_stamps(img, canvas, on_page, r_range)
    notes = L.notes + notes
    blocks, out = _apply(img, canvas, keep)

    protected, prot_mask = _stamp_records(found), np.zeros(gray.shape, np.uint8)
    for name, poly in L.protected:
        quad = f(np.asarray(poly, np.float64))
        cv2.fillPoly(prot_mask, [np.round(quad).astype(np.int32)], 255)
        protected.append({"name": name, "polygon": np.round(quad, 1).tolist()})
    if (blocks & proof.to_blocks(prot_mask)).any():
        return fail("a mask would touch a protected results region", report, notes, layout)

    leaks = []
    if verify:
        for fl in L.fields:
            if fl.kind == "signature":
                r = masking.verify_signatures(gray, out, f, fl.zones, None, ms, keep=keep, printed=print_mask,
                                              centered=fl.source != "table")
            else:
                r = masking.verify_column(out, f, fl.zones, print_ref, field_settings(fl))
            if r:
                leaks.append(r)
    if leaks:
        return fail("leak check: " + "; ".join(leaks), report, notes, layout)
    return PageResult(True, out, "generic", 0, 0.0, "", (time.perf_counter() - t0) * 1000, report, notes,
                      blocks, protected, to_image, layout)


def redact_pages(pages, templates: list[Template], s: AlignSettings, ms: MaskSettings | None = None,
                 form: str | None = None, strict: bool = False, rng: random.Random | None = None,
                 generic: bool = False) -> DocResult:
    return DocResult([redact_page(p, templates, s, ms, rng=rng, form=form, strict=strict, generic=generic)
                      for p in pages])


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
