"""Command line.

    rfpii keygen keys/
    rfpii redact Forms/ --templates templates --out public --review review --sign-key keys/signing_key.pem
    rfpii verify public/x.pdf --original x.PDF --pubkey keys/signing_key.pub.pem --ledger public/ledger.jsonl
    rfpii preview templates/form34a_2022 -o preview.png
    rfpii propose reference.png -o templates/new_form/template.yaml
    rfpii demo demo/
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import multiprocessing as mp
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from . import io, manifest as mf
from .align import AlignSettings, prepare
from .masking import MaskSettings
from .pipeline import redact_boxes
from .process import Config, process_document
from .templates import load_templates
from .vlm import Ollama, VlmSettings

_W: dict = {}  # per-worker state


def _init_worker(cfg: Config) -> None:
    cv2.setNumThreads(1)  # parallelism comes from processes (and determinism needs one thread)
    _W["cfg"] = cfg
    _W["templates"] = [prepare(t, cfg.align) for t in load_templates(cfg.templates)]
    _W["vlm"] = Ollama(cfg.vlm) if cfg.vlm else None


def _out_path(root: str, rel: str, ext: str) -> Path:
    p = Path(root) / Path(rel).with_suffix(ext)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _process(job: tuple[str, str]) -> dict:
    src, rel = job
    cfg: Config = _W["cfg"]
    vlm: Ollama | None = _W["vlm"]
    t0 = time.perf_counter()
    try:
        o = process_document(src, rel, _W["templates"], cfg, vlm)
    except Exception as e:  # unreadable or broken file -> review, never publish
        return {"report": {"file": src, "ok": False, "reason": f"{type(e).__name__}: {e}",
                           "ms": round((time.perf_counter() - t0) * 1000, 1)}, "rel": rel}
    report = o.report
    if not o.ok:
        _out_path(cfg.review, rel, ".json").write_text(json.dumps(report, indent=2))
        if vlm and cfg.vlm_fallback and o.source_pages:
            # Unrecognised layout: let the VLM find the PII, apply the same
            # masking, and park the result for a person to approve.
            try:
                fixed = [redact_boxes(sp.pixels, vlm.locate(sp.pixels), cfg.mask) for sp in o.source_pages]
                vout = _out_path(Path(cfg.review) / "vlm", rel, ".pdf")
                io.write_pdf([p.image for p in fixed], vout)  # a proposal for review, not published
                report["vlm_proposal"] = str(vout)
            except Exception as e:
                report["vlm_error"] = f"{type(e).__name__}: {e}"
    report["ms"] = round((time.perf_counter() - t0) * 1000, 1)
    return {"report": report, "manifest": o.manifest, "rel": rel}


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


def _vlm_settings(a) -> VlmSettings:
    kw = {"coords": a.vlm_coords, "allow_remote_unredacted": getattr(a, "allow_remote_unredacted", False)}
    if a.vlm_model:
        kw["model"] = a.vlm_model
    if a.vlm_host:
        kw["host"] = a.vlm_host
    return VlmSettings(**kw)


def cmd_redact(a) -> int:
    vs = _vlm_settings(a) if (a.vlm_audit or a.vlm_fallback) else None
    if vs:
        Ollama(vs)  # fail fast on bad cloud configuration
    cfg = Config(a.templates, a.out, a.review,
                 AlignSettings(min_score=a.min_score, min_inliers=a.min_inliers),
                 MaskSettings(hide=a.hide, phone_tail=a.phone_tail), vs, a.vlm_audit, a.vlm_fallback,
                 a.strict_structure)
    key = mf.load_private(a.sign_key) if a.sign_key else None
    if not key:
        print("warning: no --sign-key; manifests will be unsigned", file=sys.stderr)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    Path(a.review).mkdir(parents=True, exist_ok=True)
    ledger = Path(a.ledger or Path(a.out) / "ledger.jsonl")
    jobs = [j for j in _jobs(a.inputs) if a.overwrite or not _out_path(a.out, j[1], ".pdf").exists()]
    t0 = time.perf_counter()
    n_ok, batch = 0, []
    with open(Path(a.review) / "report.jsonl", "a") as log, ProcessPoolExecutor(
        a.workers, initializer=_init_worker, initargs=(cfg,),
        # fork after OpenCV has started its thread pool can deadlock the child
        mp_context=mp.get_context("spawn"),
    ) as pool:
        for res in pool.map(_process, jobs, chunksize=4):
            r = res["report"]
            if r["ok"]:
                m = mf.sign(res["manifest"], key) if key else res["manifest"]
                mpath = _out_path(a.out, res["rel"], ".manifest.json")
                mpath.write_text(json.dumps(m, indent=1, sort_keys=True))
                h = mf.digest(m)
                mf.ledger_append(ledger, h, m["source"]["sha256"], m["output"]["pdf_sha256"], res["rel"])
                batch.append({"name": res["rel"], "manifest_sha256": h})
                n_ok += 1
            log.write(json.dumps(r) + "\n")
            if not r["ok"]:
                print(f"REVIEW  {r['file']}: {r.get('reason', '')}", file=sys.stderr)
    if batch:
        # One root for the whole batch: publish it widely when the batch is released.
        bdir = Path(a.out) / "batches"
        bdir.mkdir(exist_ok=True)
        ok, _, entries = mf.ledger_check(ledger)
        b = {"schema": "rfpii/batch/1", "created_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
             "count": len(batch), "merkle_root": mf.merkle_root([x["manifest_sha256"] for x in batch]),
             "ledger_head": entries[-1]["entry_hash"] if entries else None, "documents": batch}
        b = mf.sign(b, key) if key else b
        bpath = bdir / f"batch-{b['created_at'].replace(':', '')}.json"
        bpath.write_text(json.dumps(b, indent=1, sort_keys=True))
        print(f"batch root {b['merkle_root']}  ({bpath})")
    dt_ = time.perf_counter() - t0
    print(f"{n_ok}/{len(jobs)} published, {len(jobs) - n_ok} sent to review, "
          f"{dt_:.1f}s ({len(jobs) / dt_ if dt_ else 0:.1f} files/s)")
    return 0 if n_ok == len(jobs) else 2


def cmd_verify(a) -> int:
    from .verify import verify
    rep = verify(a.redacted, a.manifest, a.original, a.pubkey, a.ledger, a.overlay, a.rerun)
    print(rep.text())
    return 0 if rep.ok else 1


def cmd_keygen(a) -> int:
    priv, pub = mf.keygen(a.directory)
    print(f"private key {priv} (keep offline/secret)\npublic key  {pub} (publish)  "
          f"id {mf.key_id(mf.load_public(pub))}")
    return 0


def cmd_preview(a) -> int:
    """Draw zones (red blackout, purple signature, blue partial) and protected results regions
    (green) over the reference so they can be checked by eye."""
    t = load_templates(a.template)[0]
    img = cv2.cvtColor(t.image, cv2.COLOR_GRAY2BGR)
    overlay = img.copy()
    colours = {"blackout": (0, 0, 255), "signature": (180, 0, 200), "partial": (255, 120, 0)}
    for z in t.zones:
        cv2.fillPoly(overlay, [np.round(z.padded(t.padding)).astype(np.int32)], colours[z.action])
    img = cv2.addWeighted(overlay, 0.35, img, 0.65, 0)
    for name, (x, y, w, h) in t.protect.items():
        cv2.rectangle(img, (int(x), int(y)), (int(x + w), int(y + h)), (0, 160, 0), 3)
        cv2.putText(img, f"protected: {name}", (int(x) + 4, int(y) + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (0, 140, 0), 2)
    for z in t.zones:
        p = np.round(z.polygon).astype(np.int32)
        cv2.polylines(img, [p], True, colours[z.action], 1)
        if z.row == 0:
            label = (z.group or z.name) + (f" ({z.rule})" if z.rule else "")
            cv2.putText(img, label, tuple(int(v) for v in p[0] + [3, -5]), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        colours[z.action], 1)
    cv2.imwrite(a.output, img)
    print(a.output)
    return 0


def cmd_propose(a) -> int:
    """Ask the VLM for PII boxes on a reference image and write a draft template."""
    import yaml

    img = io.source_pages(a.reference)[0].pixels
    client = Ollama(_vlm_settings(a))
    boxes = client.locate(img)
    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out.parent / "reference.png"), cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    zones = []
    for i, b in enumerate(boxes):
        z = {"name": f"{b.label}_{i}", "action": "blackout" if b.label == "signature" else "partial",
             "rect": [round(b.x0), round(b.y0), round(b.x1 - b.x0), round(b.y1 - b.y0)]}
        if b.label != "signature":
            z["rule"] = "phone" if b.label == "phone_number" else "id"
            z["digits"] = 10 if b.label == "phone_number" else 8
        zones.append(z)
    spec = {"name": out.parent.name, "form": a.form or "", "reference": "reference.png", "padding": 8,
            "protect": [], "zones": zones}
    out.write_text("# DRAFT from a VLM: check every zone with `rfpii preview` before use, and add\n"
                   "# `protect` rects for every results region. Erase handwriting from reference.png\n"
                   "# (keep printed rules) before committing it.\n" + yaml.safe_dump(spec, sort_keys=False))
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
    p.add_argument("--vlm-model", default="", help="vision model (default: qwen3-vl:235b on Ollama Cloud, "
                                                   "qwen3.8:27b locally); see ollama.com/library")
    p.add_argument("--vlm-host", default=None,
                   help="Ollama URL (default $OLLAMA_HOST, else https://ollama.com if $OLLAMA_API_KEY, else local)")
    p.add_argument("--vlm-coords", choices=("auto", "pixel", "norm1000"), default="auto",
                   help="box convention of the model (auto: norm1000 for Qwen3.x/Qwen3-VL, pixel otherwise)")
    p.add_argument("--allow-remote-unredacted", action="store_true",
                   help="allow sending UNREDACTED forms to a non-local VLM (fallback/propose); "
                        "only if that data transfer is approved")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="rfpii", description="Verifiable PII removal for result forms.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("redact", help="redact files, directories and zip archives in bulk")
    r.add_argument("inputs", nargs="+")
    r.add_argument("--templates", required=True, help="template dir, or dir of template dirs")
    r.add_argument("--out", required=True, help="publishable output (redacted PDFs, manifests, ledger)")
    r.add_argument("--review", required=True, help="reports for documents that failed checks")
    r.add_argument("--sign-key", help="Ed25519 private key (rfpii keygen) to sign manifests and batches")
    r.add_argument("--ledger", help="append-only ledger (default <out>/ledger.jsonl)")
    r.add_argument("--workers", type=int, default=os.cpu_count())
    r.add_argument("--hide", type=int, default=MaskSettings.hide, help="characters hidden per ID / phone number")
    r.add_argument("--phone-tail", type=int, default=MaskSettings.phone_tail,
                   help="phone numbers: hide only within the last N digits")
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

    v = sub.add_parser("verify", help="prove a redacted form changed nothing but the listed blocks")
    v.add_argument("redacted")
    v.add_argument("--manifest", help="default: <redacted>.manifest.json")
    v.add_argument("--original", help="the original transmission (e.g. from the IEBC portal)")
    v.add_argument("--pubkey", help="publisher's public key")
    v.add_argument("--ledger", help="the published ledger.jsonl")
    v.add_argument("--overlay", help="directory for images showing every changed block")
    v.add_argument("--rerun", metavar="TEMPLATES", help="also re-run the redaction from the manifest's seed")
    v.set_defaults(fn=cmd_verify)

    k = sub.add_parser("keygen", help="create an Ed25519 signing key pair")
    k.add_argument("directory")
    k.set_defaults(fn=cmd_keygen)

    p = sub.add_parser("preview", help="render template zones and protected regions for visual check")
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
