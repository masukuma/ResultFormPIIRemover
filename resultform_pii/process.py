"""One document, start to finish, with the evidence to back it.

Used by `rfpii redact` (in worker processes) and by `rfpii verify --rerun`,
which repeats a redaction from its manifest and must get the same result.
"""

from __future__ import annotations

import datetime as dt
import random
import secrets
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from . import __version__, io, kiems, proof
from .align import AlignSettings
from .masking import MaskSettings
from .pipeline import DocResult, redact_pages
from .templates import Template
from .vlm import Ollama, VlmSettings

SCHEMA = "rfpii/manifest/1"


@dataclass
class Config:
    templates: str
    out: str
    review: str
    align: AlignSettings = field(default_factory=AlignSettings)
    mask: MaskSettings = field(default_factory=MaskSettings)
    vlm: VlmSettings | None = None  # None = no VLM use
    vlm_audit: bool = False
    vlm_fallback: bool = False
    strict_structure: bool = False
    generic: bool = True  # read pages no template matches (needs the OCR extra)


def tool_info() -> dict:
    """Version and exact code revision, so a scrutineer can run the same code."""
    here = Path(__file__).resolve().parent
    info = {"name": "resultform-pii", "version": __version__, "commit": None}
    try:
        commit = subprocess.run(["git", "-C", str(here), "rev-parse", "HEAD"], capture_output=True, text=True,
                                timeout=5).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(here), "status", "--porcelain", "--", "."], capture_output=True,
                               text=True, timeout=5).stdout.strip()
        info["commit"] = (commit + ("+uncommitted-changes" if dirty else "")) or None
    except Exception:
        pass
    return info


def _ocr_info() -> dict:
    from .ocr import Ocr
    return Ocr.info()


def document_rules(doc: DocResult, templates: list[Template]) -> list[str]:
    """Checks across the pages of one document.

    - All pages must be of one form type.
    - A template marked ``once`` (e.g. the Form 34B declaration block) must
      match exactly one page. Zero means the block wasn't recognised (a bad
      scan, or a layout we don't know) and the page holding it might
      otherwise be published as a plain results page; two or more means a
      split block or a merged document. Either way: review, not publication."""
    by_name = {t.name: t for t in templates}
    used = [by_name.get(p.template) for p in doc.pages if p.ok]
    forms = {t.form for t in used if t and t.form}
    problems = []
    if len(forms) > 1:
        problems.append(f"pages match different forms: {', '.join(sorted(forms))}")
    for t in templates:
        if t.once and t.form in forms:
            n = sum(1 for u in used if u is t)
            if n != 1:
                problems.append(f"Form {t.form}: {t.name} found on {n} pages (expected exactly 1)")
    return problems


@dataclass
class Outcome:
    ok: bool
    report: dict  # for review/report.jsonl
    manifest: dict | None = None  # unsigned; signed by the caller
    pdf: bytes | None = None
    redacted_pages: list[np.ndarray] = field(default_factory=list)  # for VLM fallback proposals
    source_pages: list = field(default_factory=list)


def _gray(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img


def process_document(src: str, rel: str, templates: list[Template], cfg: Config,
                     vlm: Ollama | None = None, seed: int | None = None, data: bytes | None = None) -> Outcome:
    data = io.read_bytes(src) if data is None else data
    source_sha = proof.sha256(data)
    report: dict = {"file": src, "ok": False, "sha256": source_sha}
    pages = io.source_pages(src, data)
    if not pages:
        report["reason"] = "no pages"
        return Outcome(False, report)

    # Recorded seed: the hidden characters are random, and the same seed
    # (with the same code, templates and input) reproduces the same output.
    seed = secrets.randbits(63) if seed is None else int(seed)
    rng = random.Random(seed)
    cv2.setRNGSeed(seed % 2**31)

    name = kiems.parse_name(Path(rel).stem)
    payload = kiems.decode_qr(_gray(pages[0].pixels))
    qr = kiems.parse_qr(payload) if payload else {}
    form = qr.get("form") or (name.form if name else None)
    meta = {"form": form, "kiems": name.to_dict() if name else None, "qr": payload, "qr_fields": qr or None}
    report.update(meta)

    doc: DocResult = redact_pages([p.pixels for p in pages], templates, cfg.align, cfg.mask, form,
                                  cfg.strict_structure, rng, cfg.generic)
    problems = [f"page {i + 1}: {p.reason}" for i, p in enumerate(doc.pages) if not p.ok]
    if mismatch := kiems.check(name, qr):
        problems.append(mismatch)
    problems += document_rules(doc, templates)

    audits = []
    if vlm and cfg.vlm_audit and not problems:
        for i, p in enumerate(doc.pages):
            a = vlm.audit(p.image)  # only ever sends the *redacted* page
            audits.append({"page": i + 1, **a.record})
            if not a.ok:
                problems.append(f"page {i + 1}: VLM audit flagged: {a.findings.get('details', '')}")

    report["pages"] = [
        {"template": p.template, "ok": p.ok, "inliers": p.inliers, "score": round(p.score, 3),
         "reason": p.reason, "ms": round(p.ms, 1),
         "cells": {m: sum(1 for r in p.masks if r.get("mode") == m) for m in ("partial", "full", "empty")},
         **({"notes": p.notes} if p.notes else {})}
        for p in doc.pages
    ]
    if problems:
        report["reason"] = "; ".join(problems)
        return Outcome(False, report, redacted_pages=[p.image for p in doc.pages if p.image is not None],
                       source_pages=pages)

    # Apply the block grid to the *stored* image and check the result bit for
    # bit before anything is written: nothing but the listed blocks, nothing
    # but black.
    outputs, mpages = [], []
    for i, (sp, p) in enumerate(zip(pages, doc.pages)):
        if sp.kind == "jpeg":
            red = proof.redact_jpeg(sp.data, p.blocks)
            cmp = proof.compare_jpeg(sp.data, red.data, red.luma_blocks)
        else:
            red = proof.redact_pixels(sp.pixels, p.blocks)
            cmp = proof.compare_pixels(sp.pixels, io._decode_bytes(red.data), red.luma_blocks)
        if not cmp.ok:
            report["reason"] = f"page {i + 1}: self-check failed: {'; '.join(cmp.problems)}"
            return Outcome(False, report)
        outputs.append((red.data, sp))
        h, w = sp.pixels.shape[:2]
        mpages.append({
            "index": i + 1,
            "source_ref": sp.ref,
            "proof": "jpeg-dct" if sp.kind == "jpeg" else "pixels",
            "source_image_sha256": proof.sha256(sp.data) if sp.data else proof.sha256(sp.pixels.tobytes()),
            "output_image_sha256": proof.image_digest(red.kind, red.data),
            "width": w, "height": h, "block": proof.BLOCK,
            "blocks": proof.pack(red.luma_blocks),
            "blocks_blacked": int(red.luma_blocks.sum()),
            "blocks_unchanged": cmp.identical_blocks,
            "protected": p.protected,
            "template": p.template,
            "alignment": {"inliers": p.inliers, "score": round(p.score, 4),
                          "H": np.round(p.H, 8).tolist() if p.H is not None else None},
            "notes": p.notes,
            "masks": p.masks,  # rules and hidden positions only, never the characters
            **({"layout": p.layout} if p.layout else {}),
        })
    out_path = Path(cfg.out) / Path(rel).with_suffix(".pdf")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pdf = io.write_pdf_images(outputs, out_path)
    used = {p.template for p in doc.pages}
    manifest = {
        "schema": SCHEMA,
        "tool": tool_info(),
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "source": {"name": Path(rel).name, "path": rel, "sha256": source_sha, **meta},
        "output": {"file": str(Path(rel).with_suffix(".pdf")), "pdf_sha256": proof.sha256(pdf)},
        "seed": format(seed, "x"),
        "settings": {"hide": cfg.mask.hide, "phone_tail": cfg.mask.phone_tail,
                     "strict_structure": cfg.strict_structure, "generic": cfg.generic},
        "templates": {t.name: t.fingerprint for t in templates if t.name in used},
        **({"ocr": _ocr_info()} if "generic" in used else {}),
        "pages": mpages,
        "vlm_audit": audits or None,
    }
    report.update(ok=True, output=str(out_path))
    return Outcome(True, report, manifest, pdf)
