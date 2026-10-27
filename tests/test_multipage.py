"""Multi-page forms (Form 34B): the declaration block must be found on
exactly one page, it wins over a results template on the same page, and
fields anchored to printed labels follow layout variants."""

import random

import cv2
import numpy as np
import pytest

from resultform_pii import masking
from resultform_pii.pipeline import DocResult, PageResult
from resultform_pii.process import document_rules
from resultform_pii.templates import Template, Zone


def _t(name, form="34B", once=False, priority=0):
    return Template(name, np.zeros((10, 10), np.uint8), [], form=form, once=once, priority=priority)


def _doc(*names):
    return DocResult([PageResult(True, None, n, 100, 0.9) for n in names])


@pytest.mark.parametrize("pages,ok", [
    (("results", "results", "block"), True),
    (("results", "results"), False),  # block not recognised: never publish
    (("results", "block", "block"), False),  # split or merged document
])
def test_block_must_be_on_exactly_one_page(pages, ok):
    ts = [_t("results"), _t("block", once=True, priority=1)]
    assert (document_rules(_doc(*pages), ts) == []) == ok


def test_mixed_forms_are_rejected():
    ts = [_t("a34", form="34A"), _t("results"), _t("block", once=True)]
    assert "different forms" in document_rules(_doc("a34", "block"), ts)[0]


def _label_page(dy: int) -> np.ndarray:
    img = np.full((400, 900), 240, np.uint8)
    cv2.putText(img, "ID Number:", (40, 150 + dy), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 30, 2, cv2.LINE_AA)
    cv2.line(img, (230, 158 + dy), (800, 158 + dy), 60, 2)
    return img


def test_anchor_follows_a_shifted_label():
    ref = _label_page(0)
    anchor = (36, 120, 190, 42)
    I = masking.homography(np.eye(3))
    d, ncc = masking.find_anchor(_label_page(70), I, ref, anchor)
    assert d is not None and abs(d[1] - 70) <= 3 and abs(d[0]) <= 3 and ncc > 0.8
    d, _ = masking.find_anchor(np.full((400, 900), 240, np.uint8), I, ref, anchor)
    assert d is None  # label absent: the page must fail, not guess


def test_digits_on_an_underline_are_not_mistaken_for_print():
    """A digit written on a printed underline must count as writing (an
    earlier test compared raw pixels, matched the underline, and skipped it)."""
    ref = np.full((90, 600), 240, np.uint8)
    cv2.line(ref, (0, 70), (600, 70), 60, 2)
    img = ref.copy()
    cv2.putText(img, "23029912", (20, 66), cv2.FONT_HERSHEY_SIMPLEX, 1.6, 20, 3, cv2.LINE_AA)
    cell = [Zone("id", (0, 10, 600, 70), "partial", 8, rule="id", search=(0, 6))]
    canvas = np.full(img.shape, 255, np.uint8)
    rep = masking.partial_column(canvas, img, masking.homography(np.eye(3)), cell, masking.MaskSettings(),
                                 random.Random(0), ref=ref)
    assert rep[0]["mode"] == "partial" and len(rep[0]["lines"][0]["hidden"]) == 3
