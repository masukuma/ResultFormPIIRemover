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


def test_snap_a_table_hung_under_another():
    """Layout C: the agents table hangs under the results table, with the same
    columns, so the column rules run on up into it. Only their lower end marks
    the table; without ``open_top`` the span check rejects it."""
    def page(dy=0):
        img = _grid(dy)
        for x in V_LINES:  # the results table above, same columns
            cv2.line(img, (x, 120 + dy), (x, 500 + dy), 50, 2)
        for y in range(120, 500, 40):
            cv2.line(img, (100, y + dy), (1560, y + dy), 50, 2)
        return img
    ref, scan = _grid(), page(dy=40)
    closed = structure.prepare_region(ref, "agents", (80, 480, 1500, 480))
    assert not structure.snap(scan, np.eye(3), closed).ok
    region = structure.prepare_region(ref, "agents", (80, 480, 1500, 480), open_top=True)
    sn = structure.snap(scan, np.eye(3), region)
    assert sn.ok, sn.reason
    p = sn.map(np.float64([[500, ROWS[5]]]))
    assert abs(p[0, 1] - (ROWS[5] + 40)) < 4


def test_rules_running_up_past_a_table_that_ends_where_expected():
    """A scan's dark band above the table can make its column rules seem to
    run on up the page. Where their lower ends are where the table ends, the
    table is placed by them, as for a table hung under another; otherwise its
    zones fell back to widened windows that took in the next lines."""
    scan = _grid(dy=6)
    for x in V_LINES:
        cv2.line(scan, (x, 250), (x, 506), 50, 2)
    region = structure.prepare_region(_grid(), "agents", (80, 480, 1500, 480))
    sn = structure.snap(scan, np.eye(3), region)
    assert sn.ok, sn.reason
    p = sn.map(np.float64([[500, ROWS[5]]]))
    assert abs(p[0, 1] - (ROWS[5] + 6)) < 4
