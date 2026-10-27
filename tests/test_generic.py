"""Reading forms without a template: label matching and free-text rules."""

import numpy as np
import pytest

from resultform_pii import generic, ocr
from resultform_pii.ocr import Line


def line(text, x0=0.0, x1=100.0, probs=None):
    q = np.float32([[x0, 0], [x1, 0], [x1, 10], [x0, 10]])
    n = len(text)
    return Line(q, text, 1.0, [(i + 0.5) / n for i in range(n)], probs or [1.0] * n)


@pytest.mark.parametrize("text,found", [
    ("Pate 18-05-2021", True),  # misread, but a date follows
    ("bate 17-10-2019", True),
    ("Date", True),
    ("Name of Candidate", False),  # "...date"
    ("UNIQUE ESTATE GATE 02 Polling Station", False),  # GATE without a date
])
def test_date_labels(text, found):
    assert bool(generic._date_labels(line(text))) == found


def test_misread_date_on_an_officer_line():
    assert generic._date_labels(line("PRESiD Dat"), misread=True)
    assert not generic._date_labels(line("PRESiD Dat"))


@pytest.mark.parametrize("text,found", [
    ("Signature:", True),
    ("AGNES KADURAsinature AK.", True),
    ("ESTHERAZOI MA Shature", True),
    ("Signatu", True),  # cut off by a stamp
    ("Reasons for Refusal to Sign (if any)", False),
    ("THE COUNTING AND SIGNING WAS SMOOTH", False),
])
def test_signature_labels(text, found):
    assert bool(generic._sig_labels(line(text))) == found


@pytest.mark.parametrize("digits,is_date", [
    ("15122020", True), ("1512020", True), ("26012017", True),
    ("25039403", False), ("33086703", False), ("2748235", False),
])
def test_dates_written_without_separators(digits, is_date):
    assert generic._date_digits(digits) == is_date


def test_free_text_numbers():
    ids = [m.group(1) for m in generic.FREE_ID_RE.finditer("PO NAME 33086703 15-12-2020 total 231")]
    assert ids == ["33086703"]
    pre = generic.ID_PREFIX_RE.search("1D. 2≤03qy0 7")
    assert pre and sum(c.isdigit() for c in pre.group(1)) >= 4
    phones = [m.group(1) for m in generic.FREE_PHONE_RE.finditer("call 0724 441656 or +254 712345678")]
    assert phones == ["0724 441656", "+254 712345678"]


def test_char_positions_and_confidence():
    ln = line("ABC Kaad", 0, 80, [1, 1, 1, 1, 0.7, 1, 0.4, 0.8])
    x0, x1 = ln.find("Kaad")
    assert 35 < x0 < 45 and x1 > 75
    assert [c for c, _ in ln.chars_in(40, 80)] == list("Kaad")
    assert ln.fuzzy("signature", 3) == []


def test_officer_name_matching():
    assert generic._name_match("MWICHEE", "MWISHEE")
    assert generic._name_match("KIPLIMO", "KIPLI")
    assert not generic._name_match("VOTES", "MWISHEE")
    assert not generic._name_match("MAO", "MAO")  # too short to tell


def test_printed_fill_is_recognised():
    assert generic._x_fill("XXXXXXXXXXXXXX")
    assert generic._x_fill("xxxxxxxx xxxx")
    assert not generic._x_fill("XAVIER")
    assert generic.PRINTED_LINE.match("Tel. Contact")
    assert not generic.PRINTED_LINE.match("Signature: Jutiq")


@pytest.mark.skipif(not ocr.available(), reason="OCR models not installed")
def test_ocr_reads_printed_labels():
    import cv2
    img = np.full((120, 900, 3), 255, np.uint8)
    cv2.putText(img, "ID/Passport No.    Tel. Contact    Signature", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                (0, 0, 0), 2, cv2.LINE_AA)
    text = " ".join(l.text for l in ocr.engine().read(img)).lower()
    assert "passport" in text and "signature" in text
