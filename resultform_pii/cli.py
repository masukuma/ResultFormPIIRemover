"""Command line: batch redaction, zone preview/proposal, demo data.

    rfpii redact Forms/ --templates templates/ --out public/ --review review/
    rfpii redact county.zip ... --vlm-audit --vlm-fallback
    rfpii preview templates/form34a_2022 -o preview.png
    rfpii propose reference.png -o templates/new_form/template.yaml
    rfpii demo demo/
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from . import io, kiems
from .align import AlignSettings, prepare
from .masking import MaskSettings
from .pipeline import DocResult, redact_boxes, redact_pages
from .templates import load_templates
from .vlm import Ollama, VlmSettings


@dataclass
class Config:
    templates: str
    out: str
    review: str
    dpi: int | None = None
    quality: int = 85
    align: AlignSettings = field(default_factory=AlignSettings)
    mask: MaskSettings = field(default_factory=MaskSettings)
    vlm: VlmSettings | None = None  # None = no VLM use
    vlm_audit: bool = False
    vlm_fallback: bool = False
    strict_structure: bool = False


_W: dict = {}  # per-worker state


def _init_worker(cfg: Config) -> None:
    cv2.setNumThreads(1)  # parallelism comes from processes
    _W["cfg"] = cfg
    _W["templates"] = [prepare(t, cfg.align) for t in load_templates(cfg.templates)]
    _W["vlm"] = Ollama(cfg.vlm) if cfg.vlm else None


def _gray(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img


def _out_path(root: str, rel: str, ext: str) -> Path:
    p = Path(root) / Path(rel).with_suffix(ext)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _process(job: tuple[str, str]) -> dict:
    src, rel = job
    cfg: Config = _W["cfg"]
    vlm: Ollama | None = _W["vlm"]
    t0 = time.perf_counter()
    report: dict = {"file": src, "ok": False}
    try:
        data = io.read_bytes(src)
        report["sha256"] = hashlib.sha256(data).hexdigest()
        pages = list(io.iter_pages(src, cfg.dpi, data))
        if not pages:
            raise ValueError("no pages")
        name = kiems.parse_name(Path(rel).stem)
        payload = kiems.decode_qr(_gray(pages[0]))
        qr = kiems.parse_qr(payload) if payload else {}
        form = qr.get("form") or (name.form if name else None)
        meta = {"form": form, "kiems": name.to_dict() if name else None, "qr": payload, "qr_fields": qr or None}
        doc: DocResult = redact_pages(pages, _W["templates"], cfg.align, cfg.mask, form, cfg.strict_structure)
        doc.meta = meta
        problems = [f"page {i + 1}: {p.reason}" for i, p in enumerate(doc.pages) if not p.ok]
        if mismatch := kiems.check(name, qr):
            problems.append(mismatch)
        if vlm and cfg.vlm_audit and not problems:
            for i, p in enumerate(doc.pages):
                a = vlm.audit(p.image)
                p.audit = a.findings
                if not a.ok:
                    problems.append(f"page {i + 1}: VLM audit flagged: {a.findings.get('details', '')}")
        report.update(meta)
        report["pages"] = [
            {"template": p.template, "ok": p.ok, "inliers": p.inliers, "score": round(p.score, 3),
             "reason": p.reason, "ms": round(p.ms, 1),
             "cells": {m: sum(1 for r in p.masks if r["mode"] == m) for m in ("partial", "full", "empty")},
             **({"notes": p.notes} if p.notes else {}),
             **({"audit": p.audit} if getattr(p, "audit", None) else {})}
            for p in doc.pages
        ]
        report["ok"] = not problems
        if problems:
            report["reason"] = "; ".join(problems)
    except Exception as e:  # unreadable or broken file -> review, never publish
        report["reason"] = f"{type(e).__name__}: {e}"
        pages, doc = [], None

    if report["ok"]:
        out = _out_path(cfg.out, rel, ".pdf")
        io.write_pdf([p.image for p in doc.pages], out, cfg.dpi or 150, cfg.quality)
        # Sidecar: provenance only (station, kit, capture time, source hash). No PII.
        side = {k: report.get(k) for k in ("form", "kiems", "qr", "qr_fields", "sha256")}
        side["source_name"] = Path(rel).name
        side["templates"] = sorted({p.template for p in doc.pages})
        out.with_suffix(".json").write_text(json.dumps(side, indent=2))
        report["output"] = str(out)
    else:
        _out_path(cfg.review, rel, ".json").write_text(json.dumps(report, indent=2))
        if vlm and cfg.vlm_fallback and pages:
            # Unrecognised layout: let the VLM find the PII, apply the same
            # masking, and park the result for a person to approve.
            try:
                fixed = [redact_boxes(pg, vlm.locate(pg), cfg.mask) for pg in pages]
                vout = _out_path(Path(cfg.review) / "vlm", rel, ".pdf")
                io.write_pdf([p.image for p in fixed], vout, cfg.dpi or 150, cfg.quality)
                report["vlm_proposal"] = str(vout)
            except Exception as e:
                report["vlm_error"] = f"{type(e).__name__}: {e}"
    report["ms"] = round((time.perf_counter() - t0) * 1000, 1)
    return report


def _jobs(inputs: list[str]) -> list[tuple[str, str]]:
    """(source, output-relative path) for every input; keeps folder structure so
    same-named files from different folders (p-1.jpg, ...) don't collide."""
    jobs = []
    for root in map(Path, inputs):
        for src in io.list_inputs([root]):
            if io.ZIP_SEP in src:
                archive, member = src.split(io.ZIP_SEP, 1)
                base = Path(archive).relative_to(root.parent if root.is_file() else root)
                rel = str(base.with_suffix("") / member)
            else:
                rel = str(Path(src).relative_to(root)) if root.is_dir() else Path(src).name
            jobs.append((src, rel))
    return jobs


def cmd_redact(a) -> int:
    vs = None
    if a.vlm_audit or a.vlm_fallback:
        vs = VlmSettings(model=a.vlm_model, coords=a.vlm_coords, **({"host": a.vlm_host} if a.vlm_host else {}))
    cfg = Config(a.templates, a.out, a.review, a.dpi, a.quality,
                 AlignSettings(min_score=a.min_score, min_inliers=a.min_inliers),
                 MaskSettings(fraction=a.hide_fraction), vs, a.vlm_audit, a.vlm_fallback, a.strict_structure)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    Path(a.review).mkdir(parents=True, exist_ok=True)
    jobs = [j for j in _jobs(a.inputs) if a.overwrite or not _out_path(a.out, j[1], ".pdf").exists()]
    t0 = time.perf_counter()
    n_ok = 0
    with open(Path(a.review) / "report.jsonl", "a") as log, ProcessPoolExecutor(
        a.workers, initializer=_init_worker, initargs=(cfg,),
        # fork after OpenCV has started its thread pool can deadlock the child
        mp_context=mp.get_context("spawn"),
    ) as pool:
        for r in pool.map(_process, jobs, chunksize=4):
            n_ok += r["ok"]
            log.write(json.dumps(r) + "\n")
            if not r["ok"]:
                print(f"REVIEW  {r['file']}: {r.get('reason', '')}", file=sys.stderr)
    dt = time.perf_counter() - t0
    print(f"{n_ok}/{len(jobs)} published, {len(jobs) - n_ok} sent to review, "
          f"{dt:.1f}s ({len(jobs) / dt if dt else 0:.1f} files/s)")
    return 0 if n_ok == len(jobs) else 2


def cmd_preview(a) -> int:
    """Draw the padded zones over the reference so they can be checked by eye."""
    t = load_templates(a.template)[0]
    img = cv2.cvtColor(t.image, cv2.COLOR_GRAY2BGR)
    overlay = img.copy()
    colours = {"blackout": (0, 0, 255), "partial": (255, 120, 0)}
    for z in t.zones:
        cv2.fillPoly(overlay, [np.round(z.padded(t.padding)).astype(np.int32)], colours[z.action])
    img = cv2.addWeighted(overlay, 0.35, img, 0.65, 0)
    for z in t.zones:
        p = np.round(z.polygon).astype(np.int32)
        cv2.polylines(img, [p], True, colours[z.action], 1)
        if z.row == 0:
            label = z.group or z.name
            cv2.putText(img, label, tuple(int(v) for v in p[0] + [3, -5]), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        colours[z.action], 1)
    cv2.imwrite(a.output, img)
    print(a.output)
    return 0


def cmd_propose(a) -> int:
    """Ask the VLM for PII boxes on a reference image and write a draft template."""
    import yaml

    img = next(io.iter_pages(a.reference))
    client = Ollama(VlmSettings(model=a.vlm_model, coords=a.vlm_coords,
                                **({"host": a.vlm_host} if a.vlm_host else {})))
    boxes = client.locate(img)
    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out.parent / "reference.png"), _gray(img))
    zones = []
    for i, b in enumerate(boxes):
        z = {"name": f"{b.label}_{i}", "action": "blackout" if b.label == "signature" else "partial",
             "rect": [round(b.x0), round(b.y0), round(b.x1 - b.x0), round(b.y1 - b.y0)]}
        if b.label != "signature":
            z["digits"] = 8 if b.label == "id_number" else 10
        zones.append(z)
    spec = {"name": out.parent.name, "form": a.form or "", "reference": "reference.png", "padding": 8, "zones": zones}
    out.write_text("# DRAFT from a VLM: check every zone with `rfpii preview` before use.\n"
                   "# Paint handwriting out of reference.png before committing it.\n"
                   + yaml.safe_dump(spec, sort_keys=False))
    print(f"{len(zones)} zones -> {out}")
    return 0


def cmd_demo(a) -> int:
    from .synthetic import filled_form, simulate_scan, write_template

    root = Path(a.directory)
    write_template(root / "templates" / "sample_rs1")
    (root / "scans").mkdir(parents=True, exist_ok=True)
    for i in range(a.count):
        scan, _ = simulate_scan(filled_form(i), seed=i, angle=float(np.random.default_rng(i).uniform(-5, 5)))
        cv2.imwrite(str(root / "scans" / f"station_{i:04d}.jpg"), scan, [cv2.IMWRITE_JPEG_QUALITY, 85])
    noise = np.random.default_rng(99).integers(0, 255, (1800, 1300, 3), np.uint8)
    cv2.imwrite(str(root / "scans" / "not_a_form.jpg"), noise)
    print(f"wrote template and {a.count + 1} scans under {root}")
    return 0


def _vlm_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--vlm-model", default=VlmSettings.model, help="Ollama vision model")
    p.add_argument("--vlm-host", default=None, help="Ollama URL (default $OLLAMA_HOST or localhost:11434)")
    p.add_argument("--vlm-coords", choices=("pixel", "norm1000"), default="pixel",
                   help="box convention of the model (qwen2.5vl: pixel)")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="rfpii", description="Remove PII from scanned result forms.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("redact", help="redact files, directories and zip archives in bulk")
    r.add_argument("inputs", nargs="+")
    r.add_argument("--templates", required=True, help="template dir, or dir of template dirs")
    r.add_argument("--out", required=True, help="publishable output (redacted PDFs + provenance JSON)")
    r.add_argument("--review", required=True, help="reports for documents that failed checks")
    r.add_argument("--workers", type=int, default=os.cpu_count())
    r.add_argument("--dpi", type=int, default=None, help="render DPI for PDFs (default: native image resolution)")
    r.add_argument("--quality", type=int, default=85, help="output JPEG quality")
    r.add_argument("--hide-fraction", type=float, default=MaskSettings.fraction,
                   help="share of characters hidden in ID/phone cells")
    r.add_argument("--min-score", type=float, default=AlignSettings.min_score)
    r.add_argument("--min-inliers", type=int, default=AlignSettings.min_inliers)
    r.add_argument("--strict-structure", action="store_true",
                   help="send pages whose tables can't be matched to review instead of masking with wide margins")
    r.add_argument("--vlm-audit", action="store_true", help="have a VLM check every redacted page before publishing")
    r.add_argument("--vlm-fallback", action="store_true",
                   help="VLM-redact pages no template matches, into review/vlm/ for approval")
    _vlm_args(r)
    r.add_argument("--overwrite", action="store_true")
    r.set_defaults(fn=cmd_redact)

    p = sub.add_parser("preview", help="render template zones for visual check")
    p.add_argument("template")
    p.add_argument("-o", "--output", default="preview.png")
    p.set_defaults(fn=cmd_preview)

    q = sub.add_parser("propose", help="draft a template's zones from a reference image with a VLM")
    q.add_argument("reference")
    q.add_argument("-o", "--output", required=True, help="path of the template.yaml to write")
    q.add_argument("--form", default="", help="form code, e.g. 34A")
    _vlm_args(q)
    q.set_defaults(fn=cmd_propose)

    d = sub.add_parser("demo", help="generate a sample template and synthetic scans")
    d.add_argument("directory")
    d.add_argument("--count", type=int, default=20)
    d.set_defaults(fn=cmd_demo)

    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
