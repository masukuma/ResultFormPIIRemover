"""Command line: batch redaction, zone preview, demo data.

    rfpii redact scans/ --templates templates/ --out public/ --review review/
    rfpii preview templates/sample_rs1 -o preview.png
    rfpii demo demo/
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from . import io
from .align import AlignSettings, prepare
from .pipeline import redact_pages
from .templates import load_templates

_TEMPLATES = None
_SETTINGS = None


def _init_worker(template_root: str, settings: AlignSettings) -> None:
    global _TEMPLATES, _SETTINGS
    cv2.setNumThreads(1)  # parallelism comes from processes
    _SETTINGS = settings
    _TEMPLATES = [prepare(t, settings) for t in load_templates(template_root)]


def _process(job: tuple[str, str, str, int, int]) -> dict:
    src, out_dir, review_dir, dpi, quality = job
    src_p = Path(src)
    t0 = time.perf_counter()
    try:
        doc = redact_pages(io.iter_pages(src_p, dpi), _TEMPLATES, _SETTINGS)
    except Exception as e:  # unreadable file -> review, never publish
        doc, err = None, f"{type(e).__name__}: {e}"
    report = {
        "file": src,
        "ok": bool(doc and doc.ok),
        "ms": round((time.perf_counter() - t0) * 1000, 1),
        "pages": [
            {"template": p.template, "ok": p.ok, "inliers": p.inliers, "score": round(p.score, 3),
             "reason": p.reason, "ms": round(p.ms, 1)}
            for p in (doc.pages if doc else [])
        ],
    }
    if doc is None:
        report["reason"] = err
    elif not doc.pages:
        report["reason"] = "no pages"
    if report["ok"]:
        out = Path(out_dir) / (src_p.stem + ".pdf")
        io.write_pdf([p.image for p in doc.pages], out, dpi, quality)
        report["output"] = str(out)
    else:
        Path(review_dir, src_p.stem + ".json").write_text(json.dumps(report, indent=2))
    return report


def _collect(inputs: list[str]) -> list[Path]:
    files: list[Path] = []
    for i in map(Path, inputs):
        if i.is_dir():
            files += sorted(p for p in i.rglob("*") if p.suffix.lower() in io.INPUT_EXTS)
        else:
            files.append(i)
    return files


def cmd_redact(a) -> int:
    files = _collect(a.inputs)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    Path(a.review).mkdir(parents=True, exist_ok=True)
    settings = AlignSettings(min_score=a.min_score, min_inliers=a.min_inliers)
    jobs = [(str(f), a.out, a.review, a.dpi, a.quality) for f in files
            if a.overwrite or not (Path(a.out) / (f.stem + ".pdf")).exists()]
    t0 = time.perf_counter()
    n_ok = 0
    with open(Path(a.review) / "report.jsonl", "a") as log, ProcessPoolExecutor(
        a.workers, initializer=_init_worker, initargs=(a.templates, settings),
        # fork after OpenCV has started its thread pool can deadlock the child
        mp_context=mp.get_context("spawn"),
    ) as pool:
        for r in pool.map(_process, jobs, chunksize=4):
            n_ok += r["ok"]
            log.write(json.dumps(r) + "\n")
            if not r["ok"]:
                why = r.get("reason") or next((p["reason"] for p in r["pages"] if not p["ok"]), "")
                print(f"REVIEW  {r['file']}: {why}", file=sys.stderr)
    dt = time.perf_counter() - t0
    print(f"{n_ok}/{len(jobs)} published, {len(jobs) - n_ok} sent to review, "
          f"{dt:.1f}s ({len(jobs) / dt if dt else 0:.1f} files/s)")
    return 0 if n_ok == len(jobs) else 2


def cmd_preview(a) -> int:
    """Draw the padded zones over the reference so they can be checked by eye."""
    t = load_templates(a.template)[0]
    img = cv2.cvtColor(t.image, cv2.COLOR_GRAY2BGR)
    overlay = img.copy()
    overlay[t.build_mask() > 0] = (0, 0, 255)
    img = cv2.addWeighted(overlay, 0.4, img, 0.6, 0)
    for z in t.zones:
        p = np.round(z.polygon).astype(np.int32)
        cv2.polylines(img, [p], True, (0, 0, 200), 2)
        cv2.putText(img, z.name, tuple(p[0] + [5, -6]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 200), 2)
    cv2.imwrite(a.output, img)
    print(a.output)
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


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="rfpii", description="Remove PII from scanned result forms.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("redact", help="redact files/directories in bulk")
    r.add_argument("inputs", nargs="+")
    r.add_argument("--templates", required=True, help="template dir, or dir of template dirs")
    r.add_argument("--out", required=True, help="publishable output (redacted PDFs)")
    r.add_argument("--review", required=True, help="reports for pages that failed checks")
    r.add_argument("--workers", type=int, default=os.cpu_count())
    r.add_argument("--dpi", type=int, default=200, help="PDF render / output DPI")
    r.add_argument("--quality", type=int, default=85, help="output JPEG quality")
    r.add_argument("--min-score", type=float, default=AlignSettings.min_score)
    r.add_argument("--min-inliers", type=int, default=AlignSettings.min_inliers)
    r.add_argument("--overwrite", action="store_true")
    r.set_defaults(fn=cmd_redact)

    p = sub.add_parser("preview", help="render template zones for visual check")
    p.add_argument("template")
    p.add_argument("-o", "--output", default="preview.png")
    p.set_defaults(fn=cmd_preview)

    d = sub.add_parser("demo", help="generate a sample template and synthetic scans")
    d.add_argument("directory")
    d.add_argument("--count", type=int, default=20)
    d.set_defaults(fn=cmd_demo)

    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
