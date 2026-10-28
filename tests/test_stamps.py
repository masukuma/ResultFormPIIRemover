"""Stamps are found and never masked."""

import cv2
import numpy as np
import pytest

from resultform_pii import proof, stamps
from resultform_pii.align import AlignSettings, prepare
from resultform_pii.pipeline import redact_page
from resultform_pii.synthetic import filled_form, simulate_scan, write_template
from resultform_pii.templates import load_templates

S = AlignSettings()
STAMP = (700, 1262, 70)  # template px: across the presiding officer's box, over the signature


def _stamped(gray: np.ndarray, color=(150, 40, 110)) -> np.ndarray:
    """A round stamp: thick ring, inner ring, lettering."""
    img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR) if gray.ndim == 2 else gray.copy()
    cx, cy, r = STAMP
    cv2.circle(img, (cx, cy), r, color, 5, cv2.LINE_AA)
    cv2.circle(img, (cx, cy), int(r * 0.78), color, 2, cv2.LINE_AA)
    cv2.putText(img, "I.E.B.C.", (cx - 42, cy - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
    cv2.putText(img, "OFFICER", (cx - 40, cy + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)
    return img


def test_finds_a_stamp_and_not_a_table():
    page = np.full((1400, 1000, 3), 255, np.uint8)
    for y in range(100, 700, 40):  # a table with writing in it
        cv2.line(page, (50, y), (950, y), (0, 0, 0), 2)
        cv2.putText(page, "0712 345 678  KAMAU W.", (70, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (90, 30, 10), 2)
    for x in (50, 400, 950):
        cv2.line(page, (x, 100), (x, 660), (0, 0, 0), 2)
    cx, cy, r = 600, 1000, 90
    cv2.circle(page, (cx, cy), r, (150, 40, 110), 5, cv2.LINE_AA)
    cv2.putText(page, "I.E.B.C.", (cx - 55, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (150, 40, 110), 2)
    found = stamps.find(page, (40, 140))
    assert len(found) == 1
    st = found[0]
    assert abs(st.cx - cx) < 6 and abs(st.cy - cy) < 6 and abs(st.r - r) < 8


def test_faint_coloured_stamp_is_found():
    page = np.full((1200, 900, 3), 235, np.uint8)
    cv2.circle(page, (450, 600), 100, (200, 170, 205), 3, cv2.LINE_AA)  # pale violet on grey paper
    found = stamps.find(page, (50, 150))
    assert found and abs(found[0].r - 100) < 10


@pytest.fixture(scope="module")
def templates(tmp_path_factory):
    d = tmp_path_factory.mktemp("tpl") / "sample_rs1"
    write_template(d)
    return [prepare(t, S) for t in load_templates(d)]


def test_stamp_over_a_signature_is_never_masked(templates):
    form = cv2.cvtColor(_stamped(filled_form(3)), cv2.COLOR_BGR2GRAY)
    scan, H = simulate_scan(form, seed=3, angle=2.0)
    r = redact_page(scan, templates, S)
    assert r.ok, r.reason
    recorded = [p for p in r.protected if p.get("kind") == "stamp"]
    assert recorded, "stamp not recorded as protected"
    # The stamp's disc, as it lies on the scan: not one blacked block may touch it.
    cx, cy, rad = STAMP
    t = np.linspace(0, 2 * np.pi, 72, endpoint=False)
    ring = np.stack([cx + rad * np.cos(t), cy + rad * np.sin(t)], 1).reshape(-1, 1, 2)
    disc = np.zeros(scan.shape[:2], np.uint8)
    cv2.fillPoly(disc, [np.round(cv2.perspectiveTransform(ring.astype(np.float64), H)).astype(np.int32)], 255)
    assert not (r.blocks & proof.to_blocks(disc > 0)).any()
    assert np.array_equal(r.image[disc > 0], scan[disc > 0])
    # ... while the rest of the officer's box is still masked
    box = np.zeros(scan.shape[:2], np.uint8)
    q = np.float64([[340, 1220], [600, 1220], [600, 1310], [340, 1310]]).reshape(-1, 1, 2)
    cv2.fillPoly(box, [np.round(cv2.perspectiveTransform(q, H)).astype(np.int32)], 255)
    assert r.image[box > 0].max() == 0


def _half_ring(page, cx, cy, r, color):
    """A broken coloured ring: arcs covering about half the way round."""
    for a0 in range(0, 360, 60):
        cv2.ellipse(page, (cx, cy), (r, r), 0, a0, a0 + 32, color, 3, cv2.LINE_AA)


def test_a_small_broken_ring_near_a_mask_is_not_a_stamp():
    """A handwritten "0" and the strokes around it, under a signature's mask,
    fit a small, broken ring. Taken for a stamp, the signature inside it was
    left visible. Near a mask, a ring smaller than any stamp must run all the
    way round; one in the stamp size range is still taken when broken."""
    pen = (160, 60, 20)
    small = np.full((800, 800, 3), 245, np.uint8)
    _half_ring(small, 400, 400, 47, pen)  # under the smallest stamp, as seen (92%)
    hide = np.zeros(small.shape[:2], bool)
    hide[380:420, 300:500] = True
    assert stamps.near(small, hide, (50, 150), []) == []
    big = np.full((800, 800, 3), 245, np.uint8)
    _half_ring(big, 400, 400, 90, pen)
    found = stamps.near(big, hide, (50, 150), [])
    assert len(found) == 1 and abs(found[0].r - 90) < 10


def test_writing_crossing_a_circle_near_a_mask_is_not_a_stamp():
    """A column of IDs in blue pen ("coloured" ink, like stamp ink) can put
    short strokes all round a circle. A stamp's ring, even broken, runs in
    long arcs; crossings are short. Taken for a stamp, the IDs inside lost
    their masks."""
    pen = (160, 60, 20)
    page = np.full((800, 800, 3), 245, np.uint8)
    for a0 in range(0, 360, 10):  # short strokes all round a circle
        cv2.ellipse(page, (400, 400), (90, 90), 0, a0, a0 + 5, pen, 3, cv2.LINE_AA)
    hide = np.zeros(page.shape[:2], bool)
    hide[380:420, 300:500] = True
    assert stamps.near(page, hide, (50, 150), []) == []
