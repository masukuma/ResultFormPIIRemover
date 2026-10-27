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
