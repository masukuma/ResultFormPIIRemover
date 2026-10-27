import cv2
import numpy as np

from resultform_pii import structure

ROWS = [500 + 50 + 36 * i for i in range(11)]  # header bottom + row rules
H_LINES = [500] + ROWS
V_LINES = [100, 150, 500, 680, 1050, 1220, 1450, 1560]


def _grid(dy: float = 0.0, scale: float = 1.0) -> np.ndarray:
    img = np.full((1200, 1700), 235, np.uint8)
    top, bot = 500 + dy, 500 + dy + (H_LINES[-1] - 500) * scale
    for y in H_LINES:
        yy = int(round(500 + dy + (y - 500) * scale))
        cv2.line(img, (100, yy), (1560, yy), 50, 2)
    for x in V_LINES:
        cv2.line(img, (x, int(top)), (x, int(bot)), 50, 2)
    return img


def test_match_1d_ignores_spurious_lines_and_repetition():
    found = [h + 7 for h in H_LINES] + [430, 455, 760, 1040]  # extra underlines etc.
    s, t, pairs = structure.match_1d(H_LINES, found, 10, ends=(507, 15, H_LINES[-1] + 7, 25))
    assert len(pairs) == len(H_LINES)
    assert all(found[j] - H_LINES[i] == 7 for i, j in pairs)


def test_snap_finds_shifted_table():
    ref = _grid()
    region = structure.prepare_region(ref, "agents", (80, 480, 1500, 480))
    assert len(region.vlines) == len(V_LINES) and len(region.hlines) >= 10
    scan = _grid(dy=30, scale=1.05)  # table 30 px lower and stretched
    sn = structure.snap(scan, np.eye(3), region)
    assert sn.ok, sn.reason
    p = sn.map(np.float64([[500, ROWS[5]], [1220, ROWS[-1]]]))
    expect = np.float64([[500, 530 + (ROWS[5] - 500) * 1.05], [1220, 530 + (ROWS[-1] - 500) * 1.05]])
    assert np.abs(p - expect).max() < 4
