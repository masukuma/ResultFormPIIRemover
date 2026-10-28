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
    for r, dx in rows.items():
        pts = np.cumsum(rng.integers(-6, 7, (30, 2)), axis=0) + (X + dx, Y + r * ROW_H + 18)
        pts[:, 0] = np.linspace(X + dx, X + dx + 70, 30)
        cv2.polylines(img, [pts.astype(np.int32)], False, 20, 2)
    return img


def test_each_signature_gets_its_own_box():
    img = _signature_table({0: 20, 3: 110})
    cells = [Zone(f"s[{r}]", (X, Y + r * ROW_H, W, ROW_H), "signature", group="s", row=r) for r in range(ROWS)]
    canvas = np.full(img.shape, 255, np.uint8)
    rep = masking.signature_column(canvas, img, IDENTITY, cells, MaskSettings())
    assert [r["mode"] for r in rep] == ["signature", "empty", "empty", "signature", "empty", "empty"]
    sig_ink = (img < 120) & (np.abs(np.arange(img.shape[1]) - (X + W / 2)) < W / 2 - 4)[None, :]
    sig_ink[:Y + 2] = False
    for r in range(ROWS + 1):  # printed rules are not signatures
        sig_ink[Y + r * ROW_H - 3:Y + r * ROW_H + 3] = False
    assert (canvas[sig_ink] == 0).mean() > 0.97  # signatures covered
    hidden = canvas[Y:Y + ROWS * ROW_H, X:X + W] == 0
    assert hidden.mean() < 0.35  # far from the whole column
    assert not hidden[1 * ROW_H + 8:3 * ROW_H - 8].any()  # empty rows untouched
    out = cv2.cvtColor(np.where(canvas == 0, 0, img).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    assert masking.verify_signatures(img, out, IDENTITY, cells, None, MaskSettings()) == ""
    assert "not fully masked" in masking.verify_signatures(img, cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), IDENTITY,
                                                           cells, None, MaskSettings())


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
    the loop (cut off where the underline is taken out) still is, and neither
    the underline nor the loop stretches the mask over the date."""
    img = np.full((320, 900), 235, np.uint8)
    cv2.line(img, (60, 200), (60 + underline, 200), 60, 2)  # "Signature: ____"
    cv2.line(img, (60, 250), (780, 250), 60, 2)  # "Date: ____"
    rng = np.random.default_rng(1)
    pts = np.cumsum(rng.integers(-5, 6, (60, 2)), axis=0) + (150, 178)
    pts[:, 0] = np.linspace(150, 400, 60)
    pts = np.clip(pts, (0, 160), (900, 192))
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
    assert (canvas[sig_ink] == 0).mean() > 0.97  # the loop below the line included
    assert (canvas[date_ink] == 0).mean() == 0
    assert (canvas[150:205, 415:] == 255).all()  # the underline beyond the signature
    assert (canvas[212:, :290] == 255).all() and (canvas[212:, 350:] == 255).all()  # below: the loop only
    out = cv2.cvtColor(np.where(canvas == 0, 0, img).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    assert masking.verify_signatures(img, out, IDENTITY, field, None, MaskSettings(), centered=True) == ""
    unmasked = np.where(sig_ink[:, :, None], 235, out).astype(np.uint8)
    unmasked[sig_ink & (np.arange(img.shape[0]) > 205)[:, None]] = 20  # only the loop left showing
    assert "not fully masked" in masking.verify_signatures(img, unmasked, IDENTITY, field, None, MaskSettings(),
                                                           centered=True)
