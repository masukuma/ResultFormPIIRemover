import cv2
import numpy as np

from resultform_pii import proof


def _jpeg(seed=0, w=320, h=240) -> bytes:
    rng = np.random.default_rng(seed)
    img = cv2.GaussianBlur(rng.integers(0, 255, (h, w, 3), np.uint8), (0, 0), 3)
    cv2.putText(img, "123", (40, 120), cv2.FONT_HERSHEY_SIMPLEX, 2, (20, 20, 20), 4)
    return cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes()


def test_block_packing_roundtrip():
    b = np.random.default_rng(0).random((37, 29)) > 0.7
    assert np.array_equal(proof.unpack(proof.pack(b)), b)


def test_jpeg_redaction_changes_only_listed_blocks_to_black():
    data = _jpeg()
    mask = np.zeros((240, 320), np.uint8)
    mask[90:130, 40:100] = 255
    blocks = proof.to_blocks(mask)
    red = proof.redact_jpeg(data, blocks)
    cmp = proof.compare_jpeg(data, red.data, red.luma_blocks)
    assert cmp.ok, cmp.problems
    assert cmp.changed_blocks.sum() <= blocks.sum() and cmp.identical_blocks == cmp.total_blocks - cmp.changed_blocks.sum()
    dec = cv2.imdecode(np.frombuffer(red.data, np.uint8), cv2.IMREAD_GRAYSCALE)
    assert dec[proof.to_pixels(blocks, dec.shape)].max() <= 8  # decodes to black


def test_jpeg_comparison_catches_a_single_changed_coefficient():
    data = _jpeg(1)
    red = proof.redact_jpeg(data, np.zeros((30, 40), bool))  # no masks: re-encoded only
    assert proof.compare_jpeg(data, red.data).ok
    im = proof.read_dct(red.data)
    Y = im.Y.copy()
    Y[5, 5, 0, 1] += 3  # a nudge to one coefficient, e.g. to alter a digit
    im.Y = Y
    import os, tempfile
    p = tempfile.mktemp(suffix=".jpg")
    im.write_dct(p)
    bad = open(p, "rb").read()
    os.unlink(p)
    cmp = proof.compare_jpeg(data, bad)
    assert not cmp.ok and "other than black" in cmp.problems[0]


def test_undeclared_black_block_is_caught():
    data = _jpeg(2)
    b = np.zeros((30, 40), bool)
    b[3, 3] = True
    red = proof.redact_jpeg(data, b)
    assert not proof.compare_jpeg(data, red.data, np.zeros_like(b)).ok


def test_pixel_path():
    img = cv2.imdecode(np.frombuffer(_jpeg(3), np.uint8), cv2.IMREAD_COLOR)
    b = np.zeros(proof.to_blocks(np.zeros(img.shape[:2])).shape, bool)
    b[10:12, 10:15] = True
    red = proof.redact_pixels(img, b)
    out = cv2.imdecode(np.frombuffer(red.data, np.uint8), cv2.IMREAD_COLOR)
    assert proof.compare_pixels(img, out, b).ok
    out[0, 0] = (1, 2, 3)
    assert not proof.compare_pixels(img, out, b).ok
