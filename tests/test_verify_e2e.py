"""Redact a KIEMS-style PDF, then verify it the way a scrutineer would."""

import json
import os
import tempfile

import cv2
import numpy as np
import pymupdf
import pytest

from resultform_pii import io, proof
from resultform_pii.cli import main
from resultform_pii.synthetic import filled_form, simulate_scan, write_template
from resultform_pii.verify import verify

NAME = "1_34_A_007029014000102_F230450M00203609_20220810_132155.PDF"


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    root = tmp_path_factory.mktemp("e2e")
    write_template(root / "templates" / "sample_rs1")
    scan, _ = simulate_scan(filled_form(4), seed=4, angle=2.0)
    jpg = cv2.imencode(".jpg", scan, [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tobytes()
    doc = pymupdf.open()
    page = doc.new_page(width=scan.shape[1], height=scan.shape[0])
    page.insert_image(page.rect, stream=jpg)
    (root / "in").mkdir()
    doc.save(root / "in" / NAME)
    assert main(["keygen", str(root / "keys")]) == 0
    rc = main(["redact", str(root / "in"), "--templates", str(root / "templates"), "--out", str(root / "out"),
               "--review", str(root / "review"), "--workers", "1", "--sign-key", str(root / "keys/signing_key.pem")])
    assert rc == 0, (root / "review" / "report.jsonl").read_text()
    return root


def _args(root, redacted=None, **kw):
    out = root / "out" / NAME.replace(".PDF", ".pdf")
    return dict(redacted=redacted or out, manifest_path=out.with_suffix(".manifest.json"),
                original=root / "in" / NAME, pubkey=root / "keys/signing_key.pub.pem",
                ledger=root / "out/ledger.jsonl", **kw)


def test_published_form_verifies(run):
    r = verify(**_args(run, overlay_dir=run / "overlay", rerun_templates=run / "templates"))
    assert r.ok, r.text()
    m = json.loads((run / "out" / NAME.replace(".PDF", ".manifest.json")).read_text())
    assert m["pages"][0]["proof"] == "jpeg-dct" and m["pages"][0]["blocks_blacked"] > 0
    assert m["source"]["kiems"]["station"] == "007029014000102"
    assert list((run / "overlay").glob("*.png"))
    batch = next((run / "out" / "batches").glob("*.json"))
    assert json.loads(batch.read_text())["count"] == 1


def test_altered_result_is_detected(run, tmp_path):
    out = run / "out" / NAME.replace(".PDF", ".pdf")
    jpg = io.pdf_images(out.read_bytes())[0]
    im = proof.read_dct(jpg)
    Y = im.Y.copy()
    Y[90, 60, 0, 0] += 4  # results table area: brighten one block slightly
    im.Y = Y
    p = tempfile.mktemp(suffix=".jpg")
    im.write_dct(p)
    bad = open(p, "rb").read()
    os.unlink(p)
    sp = io.source_pages(str(run / "in" / NAME))[0]
    tampered = tmp_path / "tampered.pdf"
    io.write_pdf_images([(bad, sp)], tampered)
    r = verify(**_args(run, redacted=tampered))
    failed = [m for ok, m in r.checks if not ok]
    assert not r.ok and any("other than black" in m for m in failed), r.text()


def test_wrong_original_and_forged_manifest(run, tmp_path):
    other = tmp_path / NAME
    other.write_bytes((run / "in" / NAME).read_bytes() + b"\n")
    assert not verify(**{**_args(run), "original": other}).ok
    mpath = run / "out" / NAME.replace(".PDF", ".manifest.json")
    m = json.loads(mpath.read_text())
    m["pages"][0]["protected"] = []  # try to hide where results were
    forged = tmp_path / "forged.json"
    forged.write_text(json.dumps(m))
    r = verify(**{**_args(run), "manifest_path": forged})
    assert not r.ok and any("signature" in msg for ok, msg in r.checks if not ok)


def test_png_source_pages_verify(tmp_path):
    """Non-JPEG sources (PNG scans, rendered pages) take the lossless pixel
    path; the image hash must survive the PDF's PNG re-encoding."""
    write_template(tmp_path / "templates" / "sample_rs1")
    scan, _ = simulate_scan(filled_form(6), seed=6, angle=1.0)
    doc = pymupdf.open()
    page = doc.new_page(width=scan.shape[1], height=scan.shape[0])
    page.insert_image(page.rect, stream=cv2.imencode(".png", scan)[1].tobytes())
    (tmp_path / "in").mkdir()
    doc.save(tmp_path / "in" / "png_form.pdf")
    rc = main(["redact", str(tmp_path / "in"), "--templates", str(tmp_path / "templates"), "--out",
               str(tmp_path / "out"), "--review", str(tmp_path / "review"), "--workers", "1"])
    assert rc == 0
    m = json.loads((tmp_path / "out" / "png_form.manifest.json").read_text())
    assert m["pages"][0]["proof"] == "pixels" and m["pages"][0]["blocks_blacked"] > 0
    r = verify(tmp_path / "out" / "png_form.pdf", original=tmp_path / "in" / "png_form.pdf")
    assert r.ok, r.text()
