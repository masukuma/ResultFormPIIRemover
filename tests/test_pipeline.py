import cv2
import numpy as np
import pymupdf
import pytest

from resultform_pii import io
from resultform_pii.align import AlignSettings, prepare
from resultform_pii.cli import main
from resultform_pii.pipeline import redact_page
from resultform_pii.synthetic import ZONES, blank_form, filled_form, simulate_scan, write_template
from resultform_pii.templates import load_templates

S = AlignSettings()


@pytest.fixture(scope="module")
def template_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("tpl") / "sample_rs1"
    write_template(d)
    return d


@pytest.fixture(scope="module")
def templates(template_dir):
    return [prepare(t, S) for t in load_templates(template_dir)]


def _project(rect, H, shrink=0):
    x, y, w, h = rect
    x, y, w, h = x + shrink, y + shrink, w - 2 * shrink, h - 2 * shrink
    pts = np.float32([[x, y], [x + w, y], [x + w, y + h], [x, y + h]]).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(pts, H).reshape(-1, 2).astype(np.int32)


def _region(img, poly):
    m = np.zeros(img.shape[:2], np.uint8)
    cv2.fillPoly(m, [poly], 255)
    return img[m > 0]


@pytest.mark.parametrize("seed,angle", [(0, 0.0), (1, 4.0), (2, -7.0), (3, 12.0), (4, 180.0)])
def test_pii_zones_fully_blacked_out(templates, seed, angle):
    scan, H = simulate_scan(filled_form(seed), seed=seed, angle=angle)
    r = redact_page(scan, templates, S)
    assert r.ok, r.reason
    for name, rect in ZONES.items():
        # The true zone (from the known warp) must be entirely black.
        assert _region(r.image, _project(rect, H)).max() == 0, name


def test_results_table_survives(templates):
    scan, H = simulate_scan(filled_form(5), seed=5, angle=3.0)
    r = redact_page(scan, templates, S)
    assert r.ok
    table = _project((60, 520, 1120, 540), H, shrink=4)
    assert np.array_equal(_region(r.image, table), _region(scan, table))


@pytest.mark.parametrize("bad", [
    np.random.default_rng(0).integers(0, 255, (1800, 1300, 3), np.uint8),  # noise
    np.full((1800, 1300, 3), 255, np.uint8),  # blank page
    cv2.cvtColor(cv2.flip(blank_form(), 1), cv2.COLOR_GRAY2BGR),  # mirrored: not a homography we accept
])
def test_unrecognised_pages_go_to_review(templates, bad):
    r = redact_page(bad, templates, S)
    assert not r.ok and r.image is None


def test_pdf_output_has_no_text_or_metadata(tmp_path, template_dir):
    # Source PDF with a hidden text layer containing PII, plus metadata.
    scan, _ = simulate_scan(filled_form(7), seed=7)
    src = tmp_path / "in.pdf"
    doc = pymupdf.open()
    page = doc.new_page(width=scan.shape[1] * 72 / 200, height=scan.shape[0] * 72 / 200)
    page.insert_image(page.rect, stream=io.encode(scan))
    page.insert_text((50, 50), "JANE A. DOE 0712345678", render_mode=3)
    doc.set_metadata({"author": "Jane Doe"})
    doc.save(src)

    out, review = tmp_path / "out", tmp_path / "review"
    assert main(["redact", str(src), "--templates", str(template_dir), "--out", str(out),
                 "--review", str(review), "--workers", "1"]) == 0
    with pymupdf.open(out / "in.pdf") as res:
        assert res[0].get_text().strip() == ""
        assert not any(v for k, v in res.metadata.items() if k not in ("format", "encryption"))


def test_batch_withholds_failed_document(tmp_path, template_dir):
    scans = tmp_path / "scans"
    scans.mkdir()
    good, _ = simulate_scan(filled_form(1), seed=1)
    cv2.imwrite(str(scans / "good.jpg"), good)
    cv2.imwrite(str(scans / "bad.jpg"), np.full((800, 600, 3), 255, np.uint8))
    out, review = tmp_path / "out", tmp_path / "review"
    rc = main(["redact", str(scans), "--templates", str(template_dir), "--out", str(out),
               "--review", str(review), "--workers", "2"])
    assert rc == 2
    assert (out / "good.pdf").exists() and not (out / "bad.pdf").exists()
    assert (review / "bad.json").exists()
