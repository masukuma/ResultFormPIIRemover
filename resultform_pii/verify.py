"""`rfpii verify`: check a published form against its original.

Meant to be run by anyone: observers, parties, courts, journalists. Given the
redacted PDF, its manifest and (ideally) the original transmission, it checks:

 1. the manifest's signature (publisher's public key) and the ledger chain;
 2. that the redacted PDF is the one the manifest describes;
 3. that the original is the one the manifest describes (SHA-256, comparable
    with the copy on the IEBC portal or the KIEMS transmission);
 4. block by block, that every 8x8 block of the image is either *identical*
    to the original (JPEG: identical quantised DCT coefficients, independent
    of any decoder) or *solid black and listed* in the manifest;
 5. that no blacked block touches a protected results region, and that those
    regions decode to identical pixels in both files;
 6. optionally (--rerun), that re-running the same code, templates and seed
    on the original reproduces the same output.

It also writes an overlay per page: the original with every changed block
shaded red and the results regions outlined green, for people to look at.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from . import io, manifest as mf, proof


@dataclass
class Report:
    checks: list[tuple[bool, str]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def add(self, ok: bool, msg: str) -> bool:
        self.checks.append((bool(ok), msg))
        return bool(ok)

    @property
    def ok(self) -> bool:
        return all(ok for ok, _ in self.checks)

    def text(self) -> str:
        lines = [("PASS  " if ok else "FAIL  ") + m for ok, m in self.checks]
        lines += ["note  " + n for n in self.notes]
        lines.append("RESULT: " + ("VERIFIED - only the listed blocks were blacked out; nothing else changed"
                                   if self.ok else "NOT VERIFIED"))
        return "\n".join(lines)


def _overlay(img: np.ndarray, blocks: np.ndarray, protected: list[dict]) -> np.ndarray:
    out = img.copy()
    m = proof.to_pixels(blocks, img.shape[:2])
    red = out.copy()
    red[m] = (0, 0, 255)
    out = cv2.addWeighted(red, 0.55, out, 0.45, 0)
    for p in protected:
        cv2.polylines(out, [np.round(np.float64(p["polygon"])).astype(np.int32)], True, (0, 170, 0), 3)
    return out


def verify(redacted: str | Path, manifest_path: str | Path | None = None, original: str | Path | None = None,
           pubkey: str | Path | None = None, ledger: str | Path | None = None,
           overlay_dir: str | Path | None = None, rerun_templates: str | Path | None = None) -> Report:
    r = Report()
    redacted = Path(redacted)
    manifest_path = Path(manifest_path) if manifest_path else redacted.with_suffix(".manifest.json")
    m = json.loads(manifest_path.read_text())
    pdf = redacted.read_bytes()

    # 1. signature and ledger
    if pubkey:
        why = mf.check_signature(m, mf.load_public(pubkey))
        r.add(not why, "manifest signature valid" if not why else f"manifest signature: {why}")
    else:
        r.notes.append("no public key given: signature not checked")
    if ledger:
        ok, why, entries = mf.ledger_check(ledger)
        r.add(ok, "ledger hash chain intact" if ok else why)
        h = mf.digest(m)
        hit = [e for e in entries if e["manifest_sha256"] == h]
        r.add(bool(hit), f"manifest recorded in ledger (entry {hit[0]['seq']})" if hit else "manifest not in ledger")

    # 2. the published file is the one described
    r.add(proof.sha256(pdf) == m["output"]["pdf_sha256"], "redacted PDF matches the manifest's SHA-256")
    try:
        images = io.pdf_images(pdf)
    except Exception as e:  # extra/missing images on a page are themselves a red flag
        r.add(False, f"redacted PDF structure: {e}")
        return r
    if not r.add(len(images) == len(m["pages"]), f"{len(images)} page image(s), as recorded"):
        return r

    src_pages = None
    if original:
        data = Path(original).read_bytes()
        r.add(proof.sha256(data) == m["source"]["sha256"],
              f"original matches the manifest's source SHA-256 {m['source']['sha256'][:16]}...")
        src_pages = io.source_pages(str(original), data)
        r.add(len(src_pages) == len(m["pages"]), "original has the same number of pages")
    else:
        r.notes.append("no original given: changed blocks are checked to be black, but equality of all "
                       "other blocks needs the original (e.g. the copy on the IEBC portal)")

    for pm, img_bytes in zip(m["pages"], images):
        n = pm["index"]
        declared = proof.unpack(pm["blocks"])
        kind = "jpeg" if pm["proof"] == "jpeg-dct" else "pixels"
        r.add(proof.image_digest(kind, img_bytes) == pm["output_image_sha256"], f"page {n}: image matches the manifest")
        red_px = io._decode_bytes(img_bytes)
        # blacked blocks must not touch results regions
        prot = np.zeros(red_px.shape[:2], np.uint8)
        for p in pm["protected"]:
            cv2.fillPoly(prot, [np.round(np.float64(p["polygon"])).astype(np.int32)], 255)
        hits = int((declared & proof.to_blocks(prot)).sum())
        r.add(hits == 0, f"page {n}: no blacked block touches the {len(pm['protected'])} protected results "
                         f"region(s)" if not hits else f"page {n}: {hits} blacked blocks inside results regions")
        if src_pages is None:
            dark = red_px[proof.to_pixels(declared, red_px.shape[:2])]
            r.add(dark.size == 0 or dark.max() <= 8, f"page {n}: all {int(declared.sum())} listed blocks are black")
            if overlay_dir:
                _save(overlay_dir, redacted, n, _overlay(red_px, declared, pm["protected"]))
            continue
        sp = src_pages[n - 1]
        if pm["proof"] == "jpeg-dct":
            if not r.add(sp.kind == "jpeg", f"page {n}: original page is a JPEG, as recorded"):
                continue
            cmp = proof.compare_jpeg(sp.data, img_bytes, declared)
            what = "DCT coefficients"
        else:
            cmp = proof.compare_pixels(sp.pixels, red_px, declared)
            what = "pixels"
            if "rendered" in pm["source_ref"]:
                r.notes.append(f"page {n}: original was rendered ({pm['source_ref']}); "
                               "comparison depends on the same renderer version")
        r.add(cmp.ok, f"page {n}: {cmp.identical_blocks} of {cmp.total_blocks} blocks identical to the original "
                      f"({what}); {int(cmp.changed_blocks.sum())} changed, all listed and solid black"
              if cmp.ok else f"page {n}: " + "; ".join(cmp.problems))
        # Results regions: same decoded pixels in both files, inside each
        # region's polygon. For JPEG this compares luminance, which decodes
        # block by block; colour is upsampled across block edges by decoders,
        # so a black block nearby may shift a few chroma values even though
        # the stored data is identical (proven above at coefficient level).
        orig_px = sp.pixels
        if pm["proof"] == "jpeg-dct":
            flag = cv2.IMREAD_GRAYSCALE | cv2.IMREAD_IGNORE_ORIENTATION
            a = cv2.imdecode(np.frombuffer(sp.data, np.uint8), flag)
            b = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), flag)
        else:
            a, b = orig_px, red_px
        same = a.shape == b.shape
        for p in pm["protected"]:
            if not same:
                break
            region = np.zeros(a.shape[:2], np.uint8)
            cv2.fillPoly(region, [np.round(np.float64(p["polygon"])).astype(np.int32)], 255)
            same = np.array_equal(a[region > 0], b[region > 0])
        r.add(same, f"page {n}: results regions decode to identical pixels in original and redacted"
                    + (" (luminance)" if pm["proof"] == "jpeg-dct" else ""))
        if overlay_dir:
            _save(overlay_dir, redacted, n, _overlay(orig_px, cmp.changed_blocks, pm["protected"]))

    if rerun_templates and original:
        from .process import Config, process_document
        from .templates import load_templates
        from .align import prepare, AlignSettings
        from .masking import MaskSettings
        st = m["settings"]
        cfg = Config(str(rerun_templates), str(Path(overlay_dir or ".") / "_rerun"), str(Path(overlay_dir or ".")),
                     AlignSettings(), MaskSettings(hide=st["hide"], phone_tail=st["phone_tail"]),
                     strict_structure=st.get("strict_structure", False))
        templates = [prepare(t, cfg.align) for t in load_templates(rerun_templates)]
        fp = {t.name: t.fingerprint for t in templates}
        r.add(all(fp.get(k) == v for k, v in m["templates"].items()), "templates are the versions recorded")
        o = process_document(str(original), m["source"]["path"], templates, cfg, seed=int(m["seed"], 16))
        same = o.ok and [p["output_image_sha256"] for p in o.manifest["pages"]] == \
            [p["output_image_sha256"] for p in m["pages"]]
        r.add(same, "re-running the recorded code, templates and seed reproduces the same output")
    return r


def _save(directory, redacted: Path, n: int, img: np.ndarray) -> None:
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(d / f"{redacted.stem}.page{n}.changes.png"), img)
