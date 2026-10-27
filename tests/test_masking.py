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
    return [Zone(f"id[{r}]", (X, Y + r * ROW_H, W, ROW_H), "partial", digits, "id", r) for r in range(ROWS)]


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


def test_partial_hides_most_digits_but_not_all():
    img = _table({0: "36676945", 2: "0719108076"})
    gray = img.copy()
    out = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    rep = masking.partial_column(out, gray, IDENTITY, _cells(), MaskSettings(), random.Random(1))
    by = {r["zone"]: r for r in rep}
    assert by["id[0]"]["mode"] == "partial" and by["id[2]"]["mode"] == "partial"
    assert by["id[1]"]["mode"] == "empty" and by["id[5]"]["mode"] == "empty"
    for r in (0, 2):
        band = slice(Y + r * ROW_H + 4, Y + (r + 1) * ROW_H - 4)
        written = (gray[band, X:X + W] < 120).any(axis=0)
        hidden = (out[band, X:X + W, 0] == 0).any(axis=0)
        share = (written & hidden).sum() / written.sum()
        assert 0.4 < share < 0.95, share  # some, but not all, of the number is hidden
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
    shifted = [Zone(z.name, (z.rect[0], z.rect[1] + ROW_H, z.rect[2], z.rect[3]), "partial", 8, "id", z.row)
               for z in _cells()]
    out = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    masking.partial_column(out, img, IDENTITY, shifted, MaskSettings(), random.Random(2))
    band = slice(Y + 4, Y + ROW_H - 4)
    assert (out[band, X:X + W, 0] == 0).any(axis=0).sum() > 0.4 * W * 0.5


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
