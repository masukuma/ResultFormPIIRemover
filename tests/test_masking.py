import random

import cv2
import numpy as np
import pytest

from resultform_pii import masking
from resultform_pii.masking import MaskSettings
from resultform_pii.templates import Zone

ROW_H, ROWS, X, Y, W = 36, 6, 100, 200, 220


def _table(entries: dict[int, str], strike: bool = False) -> np.ndarray:
    """A ruled column with handwriting-like digits in some rows."""
    img = np.full((600, 500), 235, np.uint8)
    for r in range(ROWS + 1):
        cv2.line(img, (X - 60, Y + r * ROW_H), (X + W + 60, Y + r * ROW_H), 60, 2)
    for x in (X, X + W):
        cv2.line(img, (x, Y - 40), (x, Y + ROWS * ROW_H), 60, 2)
    for r, text in entries.items():
        cv2.putText(img, text, (X + 8, Y + r * ROW_H + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.85, 20, 2, cv2.LINE_AA)
    if strike:  # the diagonal officials draw through unused rows
        cv2.line(img, (X, Y + ROWS * ROW_H), (X + W, Y + 2 * ROW_H), 40, 2)
    return img


def _cells(digits: int = 8) -> list[Zone]:
    return [Zone(f"id[{r}]", (X, Y + r * ROW_H, W, ROW_H), "partial", digits, "id", r,
                 rule="phone" if digits >= 10 else "id") for r in range(ROWS)]


IDENTITY = masking.homography(np.eye(3))


def _middle_hidden(canvas: np.ndarray, sig: np.ndarray, chunk: int = 24, at_most: float = 0.85) -> bool:
    """One signature's ink (``sig``) is masked as prescribed: in every piece
    along it, the middle of its rows is under black, while a fair share of
    its ink (its tops and bottoms) stays visible."""
    ys, xs = np.nonzero(sig)
    covered = canvas[ys, xs] == 0
    for a in range(xs.min(), xs.max() + 1, chunk):
        sel = (xs >= a) & (xs < a + chunk)
        if sel.sum() < 10 or np.ptp(xs[sel]) < 0.75 * chunk:
            continue  # little ink, or the last few px at an end
        lo, hi = np.quantile(ys[sel], (0.4, 0.6))
        if covered[sel & (ys >= lo) & (ys <= hi)].mean() < 0.8:
            return False
    return 0.2 < covered.mean() < at_most


def _visible_runs(row_out: np.ndarray, row_in: np.ndarray) -> list[int]:
    """Runs of character columns left untouched in one row."""
    ink = (row_in < 120).any(axis=0)
    hidden = (row_out == 0).any(axis=0)
    runs, run = [], 0
    for i, h in zip(ink, hidden):
        if h:
            runs.append(run)
            run = 0
        elif i:
            run += 1
    return runs + [run]


def _cells_rule(rule: str, digits: int) -> list[Zone]:
    return [Zone(f"c[{r}]", (X, Y + r * ROW_H, W, ROW_H), "partial", digits, "c", r, rule=rule) for r in range(ROWS)]


def _hidden_chars(out: np.ndarray, gray: np.ndarray, row: int, text: str) -> list[int]:
    """Which characters (by position) ended up fully under black."""
    band = slice(Y + row * ROW_H + 4, Y + (row + 1) * ROW_H - 4)
    cols_ = slice(X + 4, X + W - 4)  # inside the column rules
    ink = gray[band, cols_] < 120
    covered = out[band, cols_] == 0
    cols = np.flatnonzero(ink.any(axis=0))
    runs = np.split(cols, np.flatnonzero(np.diff(cols) > 1) + 1)
    assert len(runs) == len(text), (len(runs), text)
    return [i for i, r in enumerate(runs) if covered[:, r][ink[:, r]].mean() >= 0.9]  # antialiased edges


def test_id_rule_hides_exactly_three_characters():
    img = _table({0: "36676945", 2: "24571109"})
    for seed in range(20):
        canvas = np.full(img.shape, 255, np.uint8)
        rep = masking.partial_column(canvas, img, IDENTITY, _cells_rule("id", 8), MaskSettings(), random.Random(seed))
        by = {r["zone"]: r for r in rep}
        assert by["c[0]"]["mode"] == "partial" and by["c[1]"]["mode"] == "empty"
        assert len(by["c[0]"]["lines"][0]["hidden"]) == 3
        for row, text in ((0, "36676945"), (2, "24571109")):
            hid = _hidden_chars(canvas, img, row, text)
            assert len(hid) == 3, hid


def test_phone_rule_keeps_prefix_and_hides_three_of_last_six():
    img = _table({1: "0734471897"})
    seen = set()
    for seed in range(30):
        canvas = np.full(img.shape, 255, np.uint8)
        rep = masking.partial_column(canvas, img, IDENTITY, _cells_rule("phone", 10), MaskSettings(),
                                     random.Random(seed))
        line = {r["zone"]: r for r in rep}["c[1]"]["lines"][0]
        assert line["slots"] == 10 and len(line["hidden"]) == 3 and min(line["hidden"]) >= 4
        hid = _hidden_chars(canvas, img, 1, "0734471897")
        assert not set(hid) & {0, 1, 2}, hid  # 073 always readable
        seen.add(tuple(line["hidden"]))
    assert len(seen) > 3  # random, not a fixed pattern


def test_partial_masks_pass_the_leak_check():
    img = _table({0: "36676945", 2: "0719108076"})
    out = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    masking.partial_column(out, img, IDENTITY, _cells(), MaskSettings(), random.Random(1))
    assert masking.verify_column(out, IDENTITY, _cells(), None, MaskSettings()) == ""


def test_strike_line_through_unused_rows_is_not_masked():
    img = _table({0: "12345678"}, strike=True)
    out = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    rep = masking.partial_column(out, img, IDENTITY, _cells(), MaskSettings(), random.Random(0))
    assert [r["mode"] for r in rep[2:]] == ["empty"] * 4
    assert (out[Y + 3 * ROW_H:Y + 6 * ROW_H, X:X + W] == 0).sum() == 0


def test_writing_off_the_expected_rows_is_still_masked():
    """Table mapped a row low: row 0 writing sits above the first band."""
    img = _table({0: "36676945"})
    shifted = [Zone(z.name, (z.rect[0], z.rect[1] + ROW_H, z.rect[2], z.rect[3]), "partial", 8, "id", z.row,
                    rule="id") for z in _cells()]
    out = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    masking.partial_column(out, img, IDENTITY, shifted, MaskSettings(), random.Random(2))
    band = slice(Y + 4, Y + ROW_H - 4)
    assert (out[band, X:X + W, 0] == 0).any(axis=0).sum() > 30  # three characters' worth


def test_verify_flags_unmasked_number():
    img = _table({1: "0719108076"})
    out = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)  # nothing masked
    assert "unmasked" in masking.verify_column(out, IDENTITY, _cells(10), None, MaskSettings())


@pytest.mark.parametrize("n,k", [(8, 5), (10, 6), (6, 4), (12, 8)])
def test_pick_hidden_never_leaves_long_visible_runs(n, k):
    rng = random.Random(0)
    for _ in range(50):
        hidden = set(masking.pick_hidden(n, k, rng))
        run = best = 0
        for i in range(n):
            run = 0 if i in hidden else run + 1
            best = max(best, run)
        assert best <= 2 and len(hidden) >= k


def test_column_edge_check_catches_shifted_column():
    img = _table({0: "36676945"})
    assert masking.check_column_edges(img, IDENTITY, _cells()) == ""
    shift = masking.homography(np.float64([[1, 0, 60], [0, 1, 0], [0, 0, 1]]))
    assert "placed wrong" in masking.check_column_edges(img, shift, _cells())


def _signature_table(rows: dict[int, int]) -> np.ndarray:
    """Ruled column; rows -> x offset of a scribbled signature."""
    img = np.full((600, 500), 235, np.uint8)
    for r in range(ROWS + 1):
        cv2.line(img, (X - 60, Y + r * ROW_H), (X + W + 60, Y + r * ROW_H), 60, 2)
    for x in (X, X + W):
        cv2.line(img, (x, Y - 40), (x, Y + ROWS * ROW_H), 60, 2)
    rng = np.random.default_rng(0)
    for r, dx in rows.items():  # cursive: up and down strokes, 20 px tall
        x = np.linspace(X + dx, X + dx + 70, 40)
        pts = np.column_stack([x, Y + r * ROW_H + 18 + 9 * np.sin((x - X - dx) / 4) + rng.integers(-1, 2, len(x))])
        cv2.polylines(img, [pts.astype(np.int32)], False, 20, 2)
    return img


def test_each_signature_is_masked_through_its_middle():
    """One band per signature, through its middle: that it is signed stays
    visible, the signature can't be made out. Empty rows are untouched."""
    img = _signature_table({0: 20, 3: 110})
    cells = [Zone(f"s[{r}]", (X, Y + r * ROW_H, W, ROW_H), "signature", group="s", row=r) for r in range(ROWS)]
    canvas = np.full(img.shape, 255, np.uint8)
    rep = masking.signature_column(canvas, img, IDENTITY, cells, MaskSettings())
    assert [r["mode"] for r in rep] == ["signature", "empty", "empty", "signature", "empty", "empty"]
    sig_ink = (img < 120) & (np.abs(np.arange(img.shape[1]) - (X + W / 2)) < W / 2 - 4)[None, :]
    sig_ink[:Y + 2] = False
    for r in range(ROWS + 1):  # printed rules are not signatures
        sig_ink[Y + r * ROW_H - 3:Y + r * ROW_H + 3] = False
    for r in (0, 3):
        assert _middle_hidden(canvas, sig_ink & (np.abs(np.arange(img.shape[0]) - (Y + r * ROW_H + 18)) < 18)[:, None])
    hidden = canvas[Y:Y + ROWS * ROW_H, X:X + W] == 0
    assert hidden.mean() < 0.35  # far from the whole column
    assert not hidden[1 * ROW_H + 8:3 * ROW_H - 8].any()  # empty rows untouched
    out = cv2.cvtColor(np.where(canvas == 0, 0, img).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    assert masking.verify_signatures(img, out, IDENTITY, cells, None, MaskSettings()) == ""
    assert "not masked through the middle" in masking.verify_signatures(img, cv2.cvtColor(img, cv2.COLOR_GRAY2BGR),
                                                                        IDENTITY, cells, None, MaskSettings())
    writing = cv2.dilate((sig_ink * 255).astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
    full = cv2.cvtColor(np.where(writing, 0, img).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    assert masking.verify_signatures(img, full, IDENTITY, cells, None, MaskSettings()) == ""  # more is fine


def test_dotted_leaders_are_print_not_writing():
    from resultform_pii.masking import ink_mask
    strip = np.full((60, 700), 235, np.uint8)
    for x in range(20, 680, 14):  # "ID Number: ..........."
        cv2.rectangle(strip, (x, 44), (x + 5, 47), 30, -1)
    assert ink_mask(strip, 56).sum() == 0  # an empty field
    cv2.putText(strip, "2345678", (120, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, 20, 3)
    ink = ink_mask(strip, 56)
    assert ink[:, 120:330].sum() > 0  # the digits stay
    assert ink[44:48, 400:680].sum() == 0  # the leader beyond them goes


def test_a_line_drawn_through_empty_rows_is_not_writing():
    """Officials strike out unused rows with a long diagonal line. Within one
    column it is only a short stroke, and was masked as digits or as a
    signature. Found over the whole table, it is left alone; digits it
    passes near are still masked."""
    import random
    from resultform_pii.templates import Zone
    row_h, rows = 60, 4
    img = np.full((60 + rows * row_h + 40, 1400), 235, np.uint8)
    for i in range(rows + 1):
        cv2.line(img, (20, 60 + i * row_h), (1380, 60 + i * row_h), 60, 2)
    for x in (20, 400, 700, 1000, 1380):
        cv2.line(img, (x, 60), (x, 60 + rows * row_h), 60, 2)
    cv2.putText(img, "23456781", (420, 60 + row_h - 14), cv2.FONT_HERSHEY_SIMPLEX, 1.2, 20, 3)  # row 1 filled
    cv2.line(img, (30, 60 + rows * row_h - 4), (1370, 60 + row_h + 4), 25, 9)  # struck through rows 2-4 (a scan's pen width)
    I = masking.homography(np.eye(3))
    strikes = masking.find_strikes(img, I, (10, 50, 1380, rows * row_h + 20), row_h)
    assert len(strikes) == 1
    cells = [Zone(f"id[{i}]", (403, 60 + i * row_h + 3, 294, row_h - 6), "partial", 8, rule="id")
             for i in range(rows)]
    canvas = np.full(img.shape, 255, np.uint8)
    rep = masking.partial_column(canvas, img, I, cells, MaskSettings(), random.Random(1), strikes=strikes)
    assert rep[0]["mode"] == "partial" and len(rep[0]["lines"]) == 1  # the written ID
    assert all(r["mode"] == "empty" for r in rep[1:]), rep  # struck rows: nothing masked
    canvas2 = np.full(img.shape, 255, np.uint8)
    rep2 = masking.partial_column(canvas2, img, I, cells, MaskSettings(), random.Random(1))
    assert any(r["mode"] != "empty" for r in rep2[1:])  # without it, the line was masked


@pytest.mark.parametrize("underline", [720, 420])  # the whole field's width, or shorter than the field
def test_the_date_below_an_underlined_signature_stays_visible(underline):
    """An officer's signature on its line, a loop of it running down beside
    the date written on the line below: joined into one blob, the date was
    masked with it. Strokes mostly on the next line are not the signature's,
    and neither the underline nor the loop stretches the mask over the
    date: the band through the signature's middle leaves the loop's foot,
    beside the date, showing."""
    img = np.full((320, 900), 235, np.uint8)
    cv2.line(img, (60, 200), (60 + underline, 200), 60, 2)  # "Signature: ____"
    cv2.line(img, (60, 250), (780, 250), 60, 2)  # "Date: ____"
    x = np.linspace(150, 400, 120)  # cursive, 30 px tall
    pts = np.column_stack([x, 177 + 14 * np.sin((x - 150) / 5)])
    cv2.polylines(img, [pts.astype(np.int32)], False, 20, 2)
    cv2.polylines(img, [np.int32([[300, 185], [310, 228], [330, 230], [338, 186]])], False, 20, 2)  # the loop
    cv2.putText(img, "11-08-2022", (420, 242), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 20, 2)
    field = [Zone("ro_signature", (100, 150, 600, 55), "signature", search=(0, 10))]
    date_ink = np.zeros(img.shape, bool)
    date_ink[210:244, 415:700] = img[210:244, 415:700] < 120
    sig_ink = img < 120
    sig_ink[:, :140] = sig_ink[:, 410:] = sig_ink[196:205] = sig_ink[246:] = False
    canvas = np.full(img.shape, 255, np.uint8)
    masking.signature_column(canvas, img, IDENTITY, field, MaskSettings(), centered=True)
    main = sig_ink & (np.arange(img.shape[0]) < 196)[:, None]
    assert _middle_hidden(canvas, main)
    assert (canvas[date_ink] == 0).mean() == 0
    assert (canvas[150:205, 415:] == 255).all()  # the underline beyond the signature
    assert (canvas[205:, :290] == 255).all() and (canvas[205:, 350:] == 255).all()  # below the line: the loop only
    out = cv2.cvtColor(np.where(canvas == 0, 0, img).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    assert masking.verify_signatures(img, out, IDENTITY, field, None, MaskSettings(), centered=True) == ""
    assert "not masked through the middle" in masking.verify_signatures(
        img, cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), IDENTITY, field, None, MaskSettings(), centered=True)


def _underlined_signature(line) -> tuple[np.ndarray, np.ndarray, list[Zone]]:
    """A signature on its line, 150-400 px; ``line(img)`` draws the line."""
    img = np.full((320, 900), 235, np.uint8)
    line(img)
    rng = np.random.default_rng(1)
    x = np.linspace(150, 400, 120)  # cursive: up and down strokes, 30 px tall
    pts = np.column_stack([x, 178 + 14 * np.sin((x - 150) / 5) + rng.integers(-2, 3, len(x))])
    pts = np.clip(pts, (0, 160), (900, 196))
    cv2.polylines(img, [pts.astype(np.int32)], False, 20, 2)
    sig = np.zeros(img.shape, np.uint8)
    cv2.polylines(sig, [pts.astype(np.int32)], False, 1, 2)
    return img, sig > 0, [Zone("ro_signature", (100, 150, 600, 55), "signature", search=(0, 10))]


def test_a_broken_underline_does_not_stretch_the_signature_mask():
    """A scan breaks the line into dashes; joined through the line, they
    stretched the mask along all of it, up to a stamp beyond."""
    def dashed(img):
        cv2.line(img, (60, 200), (780, 200), 60, 2)
        for x in range(450, 780, 40):  # gaps the line runs through less cleanly
            img[197:204, x:x + 20] = 235
            img[201:203, x + 4:x + 12] = 60
    img, sig, field = _underlined_signature(dashed)
    canvas = np.full(img.shape, 255, np.uint8)
    masking.signature_column(canvas, img, IDENTITY, field, MaskSettings(), centered=True)
    assert _middle_hidden(canvas, sig)
    assert (canvas[:, 420:] == 255).all()


def test_a_striped_scan_is_no_underline():
    """Some scans are striped with long horizontal lines all over. They are
    not the field's line: taking them out would take the signature with them."""
    def stripes(img):
        rng = np.random.default_rng(2)
        for y in range(140, 260, 5):  # broken and uneven, as scanned
            x = int(rng.integers(0, 40))
            while x < 900:
                n = int(rng.integers(60, 250))
                img[y:y + 2, x:x + n] = 110
                x += n + int(rng.integers(8, 30))
    img, sig, field = _underlined_signature(stripes)
    canvas = np.full(img.shape, 255, np.uint8)
    masking.signature_column(canvas, img, IDENTITY, field, MaskSettings(), centered=True)
    assert _middle_hidden(canvas, sig, at_most=0.95)  # stripes add to its height: a little more is hidden
    out = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    assert "not masked through the middle" in masking.verify_signatures(img, out, IDENTITY, field, None, MaskSettings(),
                                                           centered=True)


@pytest.mark.parametrize("with_ref", [False, True])
def test_a_flat_flourish_is_not_the_printed_line(with_ref):
    """A signature's long straight tail above its line is a long flat line
    too; taken for the printed line, it was left visible."""
    def line(img):
        cv2.line(img, (60, 200), (780, 200), 60, 2)
    img, sig, field = _underlined_signature(line)
    end = tuple(int(v) for v in np.argwhere(sig)[np.argmax(np.argwhere(sig)[:, 1])][::-1])
    cv2.line(img, end, (end[0] + 10, 186), 20, 2)  # the tail, drawn on from the signature
    cv2.line(img, (end[0] + 10, 186), (720, 186), 20, 2)
    tail = np.zeros(img.shape, np.uint8)
    cv2.line(tail, (end[0] + 10, 186), (720, 186), 1, 2)
    ref = None
    if with_ref:
        ref = np.full(img.shape, 235, np.uint8)
        cv2.line(ref, (60, 208), (780, 208), 60, 2)  # a layout variant: 8 px off
    canvas = np.full(img.shape, 255, np.uint8)
    masking.signature_column(canvas, img, IDENTITY, field, MaskSettings(), ref=ref, centered=True)
    assert _middle_hidden(canvas, sig)
    assert (canvas[tail > 0] == 0).mean() > 0.97  # here the tail is the middle
    assert (canvas[150:205, 740:] == 255).all()  # the line beyond the tail


def _id_field(img: np.ndarray) -> list[Zone]:
    """An officer's ID field: from just below the name's line (y 115) to just
    below its own (y 200), on a line running 60-780."""
    cv2.line(img, (60, 115), (780, 115), 60, 2)
    cv2.line(img, (60, 197), (780, 197), 60, 2)
    return [Zone("ro_id", (100, 118, 600, 85), "partial", 8, rule="id", search=(0, 6))]


def test_writing_too_small_to_read_is_hidden_at_its_own_width():
    """An ID written too small to read as characters is hidden whole, but
    only as wide as the writing: across the field's full width the mask ran
    on behind the stamp beside it."""
    img = np.full((320, 900), 235, np.uint8)
    field = _id_field(img)
    cv2.putText(img, "2442587", (130, 190), cv2.FONT_HERSHEY_SIMPLEX, 1.2, 20, 4)  # ~28 px: under 40% of 85
    canvas = np.full(img.shape, 255, np.uint8)
    rep = masking.partial_column(canvas, img, IDENTITY, field, MaskSettings(), random.Random(1))
    assert rep[0]["mode"] == "full"
    digits = img[150:196] < 120
    assert (canvas[150:196][digits] == 0).all()
    assert (canvas[:, 400:] == 255).all()


def test_the_name_above_reaching_into_an_id_field_is_not_part_of_the_id():
    """A name written large and low reaches into the ID field below it: cut
    off at the field's top, its foot was taken for an eighth character, and
    one of the three hidden slots went to it."""
    img = np.full((320, 900), 235, np.uint8)
    field = _id_field(img)
    cv2.putText(img, "KIP", (420, 145), cv2.FONT_HERSHEY_SIMPLEX, 2.0, 20, 4)  # the name, low
    cv2.putText(img, "6665112", (120, 190), cv2.FONT_HERSHEY_SIMPLEX, 1.3, 20, 3)
    s = MaskSettings(min_glyph_h=0.25)
    canvas = np.full(img.shape, 255, np.uint8)
    rep = masking.partial_column(canvas, img, IDENTITY, field, s, random.Random(3))
    assert rep[0]["mode"] == "partial" and rep[0]["lines"][0]["slots"] == 7, rep
    assert (canvas[100:150, 410:560] == 255).all()  # the name stays visible
    out = cv2.cvtColor(np.where(canvas == 0, 0, img).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    assert masking.verify_column(out, IDENTITY, field, None, s) == ""


def test_print_sitting_on_the_line_is_not_joined_to_the_signature():
    """The line joins strokes crossing it, not everything touching it: a
    security print's lettering sitting on the line stretched a signature's
    mask along it."""
    def line(img):
        cv2.line(img, (60, 200), (780, 200), 60, 2)
        cv2.putText(img, "ELECTORAL", (560, 198), cv2.FONT_HERSHEY_SIMPLEX, 0.9, 90, 2)
    img, sig, field = _underlined_signature(line)
    canvas = np.full(img.shape, 255, np.uint8)
    masking.signature_column(canvas, img, IDENTITY, field, MaskSettings(), centered=True)
    assert _middle_hidden(canvas, sig)
    assert (canvas[150:205, 420:550] == 255).all()


def test_the_next_line_cut_off_below_a_signature_column_is_not_a_signature():
    """The window below a table reaches the top of an officer's name written
    large on the next line; cut off by the window's edge, its letters were
    masked as a signature."""
    img = _signature_table({0: 20})
    bottom = Y + ROWS * ROW_H
    cv2.putText(img, "NA", (X + 40, bottom + 120), cv2.FONT_HERSHEY_SIMPLEX, 2.0, 20, 4)  # top ~ bottom + 75
    cells = [Zone(f"s[{r}]", (X, Y + r * ROW_H, W, ROW_H), "signature", group="s", row=r) for r in range(ROWS)]
    canvas = np.full(img.shape, 255, np.uint8)
    rep = masking.signature_column(canvas, img, IDENTITY, cells, MaskSettings())
    assert [r["mode"] for r in rep] == ["signature"] + ["empty"] * (ROWS - 1)
    assert (canvas[bottom + 10:] == 255).all()


def test_crumbs_of_print_in_an_empty_field_are_not_a_signature():
    """What is left of a security print's lettering is crumbs; close
    together they were joined into a "signature" and masked in empty fields."""
    def line(img):
        cv2.line(img, (60, 200), (780, 200), 60, 2)
    img, _, field = _underlined_signature(line)
    img[150:197] = 235  # no signature: an empty field...
    rng = np.random.default_rng(4)
    for x in range(420, 600, 12):  # ...with crumbs of print in it, close together
        for y in range(160, 192, 7):
            x0, y0 = x + int(rng.integers(0, 3)), y + int(rng.integers(0, 3))
            img[y0:y0 + 2, x0:x0 + int(rng.integers(4, 9))] = 40
    canvas = np.full(img.shape, 255, np.uint8)
    rep = masking.signature_column(canvas, img, IDENTITY, field, MaskSettings(), centered=True)
    assert rep[0]["mode"] == "empty" and (canvas == 255).all()


def test_an_embossed_background_in_empty_fields_is_not_writing():
    """Embossed forms scan with a background of thin dark horizontal lines
    every few pixels, its lettering kinks in them. In empty fields it was
    taken for writing and masked; written fields are still masked."""
    def embossed(img):
        rng = np.random.default_rng(5)
        for y in range(0, img.shape[0], 5):
            x = 0
            while x < img.shape[1]:  # runs of line, each kinked a pixel or two up or down
                n = int(rng.integers(20, 120))
                dy = int(rng.integers(-1, 2))
                img[y + dy:y + dy + 2, x:x + n] = 25
                x += n
        for _ in range(img.size // 900):  # the lettering: kinks bridging neighbouring lines
            x, y = int(rng.integers(0, img.shape[1] - 4)), int(rng.integers(0, img.shape[0] - 6))
            img[y:y + 6, x:x + 2] = 25
    img, sig, field = _underlined_signature(lambda im: (embossed(im), cv2.line(im, (60, 200), (780, 200), 30, 2)))
    canvas = np.full(img.shape, 255, np.uint8)
    rep = masking.signature_column(canvas, img, IDENTITY, field, MaskSettings(), centered=True)
    assert rep[0]["mode"] == "signature" and _middle_hidden(canvas, sig)
    assert (canvas[:, 430:] == 255).all()  # the empty rest of the field
    empty = np.full((320, 900), 235, np.uint8)
    embossed(empty)
    ids = _id_field(empty)
    canvas = np.full(empty.shape, 255, np.uint8)
    assert masking.partial_column(canvas, empty, IDENTITY, ids, MaskSettings(), random.Random(1))[0]["mode"] == "empty"
    sigs = [Zone("s", (100, 210, 600, 55), "signature", search=(0, 10))]
    assert masking.signature_column(canvas, empty, IDENTITY, sigs, MaskSettings(), centered=True)[0]["mode"] == "empty"
    assert (canvas == 255).all()


def test_strokes_reaching_down_cover_only_their_own_width_of_the_date():
    """An officer signs with long flat strokes along the field (lines of
    their own beside the printed one) and two downstrokes into the date line.
    The printed line is still found, where the reference has it, and only
    the downstrokes' own width of the date line is masked, not the date
    written between them."""
    def line(img):
        cv2.line(img, (60, 200), (780, 200), 60, 2)
    img, sig, field = _underlined_signature(line)
    for y in (160, 188):  # long flat strokes along the field
        cv2.line(img, (150, y), (620, y), 20, 2)
    for x in (152, 398):  # downstrokes crossing the line, reaching into the date's
        cv2.line(img, (x, 188), (x - 6, 240), 20, 2)
    cv2.putText(img, "11-08-22", (200, 242), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 20, 2)
    date_ink = np.zeros(img.shape, bool)
    date_ink[214:244, 200:360] = img[214:244, 200:360] < 120
    ref = np.full(img.shape, 235, np.uint8)
    cv2.line(ref, (60, 200), (780, 200), 60, 2)
    canvas = np.full(img.shape, 255, np.uint8)
    masking.signature_column(canvas, img, IDENTITY, field, MaskSettings(), ref=ref, centered=True)
    assert (canvas[sig] == 0).mean() > 0.6  # between the long strokes, most of it is the middle
    assert (canvas[date_ink] == 0).mean() < 0.05
