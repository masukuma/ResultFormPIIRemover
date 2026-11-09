"""Small datasets for the tests: one made by the rfpii binary, and hand-made
ones."""

import hashlib
import io
import json
import os
import shutil
import subprocess
import tempfile

from PIL import Image

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RFPII = os.path.join(ROOT, "rust", "target", "release", "rfpii")
TEMPLATES = os.path.join(ROOT, "templates")


def have_rfpii():
    return os.path.exists(RFPII)


def synth(out, count=3, seed=10, extra=()):
    env = dict(os.environ, SOURCE_DATE_EPOCH="1786000000")
    subprocess.run([RFPII, "dataset", "synth", "--templates", TEMPLATES, "--out", out, "--count", str(count),
                    "--seed", str(seed), "--max-side", "700", "--workers", "2"] + list(extra),
                   check=True, env=env, stdout=subprocess.DEVNULL)


def sha(b):
    return hashlib.sha256(b).hexdigest()


def fake_dataset(path, kind, entries):
    """A dataset folder from (id, fields, boxes) triples: a 40x30 PNG page
    each, its labels and index line (no SHA256SUMS: not for `check`)."""
    from rfpii_nn.dataset import split_of
    for sub in ("pages", "labels", "masks"):
        os.makedirs(os.path.join(path, sub), exist_ok=True)
    lines = []
    for pid, fields, boxes in entries:
        buf = io.BytesIO()
        Image.new("RGB", (40, 30), (250, 250, 250 - len(lines) % 200)).save(buf, "PNG")
        img = buf.getvalue()
        with open(os.path.join(path, "pages", pid + ".png"), "wb") as f:
            f.write(img)
        mbuf = io.BytesIO()
        Image.new("L", (40, 30), 2).save(mbuf, "PNG")
        with open(os.path.join(path, "masks", pid + ".png"), "wb") as f:
            f.write(mbuf.getvalue())
        key = fields.get("split_key", "folder:" + pid)
        labels = {"schema": "rfpii/nn-labels/1", "id": pid, "page": {"width": 40, "height": 30, "kind": fields.get("kind", "form"), "held": fields.get("held", False)},
                  "split_key": key, "teacher" if kind == "teacher" else "synthetic": {}, "boxes": boxes}
        lb = json.dumps(labels).encode()
        with open(os.path.join(path, "labels", pid + ".json"), "wb") as f:
            f.write(lb)
        line = {"id": pid, "image": "pages/%s.png" % pid, "image_sha256": fields.get("image_sha256", sha(img)), "labels": "labels/%s.json" % pid,
                "labels_sha256": sha(lb), "mask": "masks/%s.png" % pid, "split_key": key, "split": split_of(key)}
        line.update({k: v for k, v in fields.items() if k not in ("split_key",)})
        lines.append(line)
    with open(os.path.join(path, "index.jsonl"), "w") as f:
        for l in lines:
            f.write(json.dumps(l) + "\n")
    with open(os.path.join(path, "dataset.json"), "w") as f:
        json.dump({"schema": "rfpii/nn-dataset/1", "kind": kind, "personal_data": kind == "teacher", "pages": len(lines)}, f)
    return path


class TempDir:
    def __enter__(self):
        self.path = tempfile.mkdtemp(prefix="rfpii-nn-test-")
        return self.path

    def __exit__(self, *a):
        shutil.rmtree(self.path, ignore_errors=True)


def fake_inkset(path, n=3, size=(128, 96), groups=("synthetic",)):
    """An ink set of small drawn pages: paper, a long dark rule, a short dark
    stroke and a violet patch, with masks marking them (bits: 1 dark, 2
    rule, 4 stamp). Not the pipeline's masks: for the tools' tests."""
    import numpy as np
    w, h = size
    for sub in ("pages", "masks"):
        os.makedirs(os.path.join(path, sub), exist_ok=True)
    lines = []
    for i in range(n):
        img = np.full((h, w, 3), (236, 234, 228), np.uint8)
        bits = np.zeros((h, w), np.uint8)
        y = 20 + 7 * i
        img[y:y + 3, 8:w - 8] = 30
        bits[y:y + 3, 8:w - 8] |= 1 | 2
        img[60:66, 20 + i:24 + i] = 40
        bits[60:66, 20 + i:24 + i] |= 1
        img[70:80, 60:72] = (120, 70, 170)
        bits[70:80, 60:72] |= 4
        pid = "p%02d" % i
        for sub, arr in (("pages", img), ("masks", bits)):
            buf = io.BytesIO()
            Image.fromarray(arr).save(buf, "PNG")
            with open(os.path.join(path, sub, pid + ".png"), "wb") as f:
                f.write(buf.getvalue())
        key = "seed:%d" % i
        lines.append({"id": pid, "source": "drawn", "page": 1, "group": groups[i % len(groups)], "split_key": key,
                      "split": "train" if i < n - 1 else "test", "width": w, "height": h,
                      "image": "pages/%s.png" % pid, "mask": "masks/%s.png" % pid})
    lines[0]["split"] = "train"
    with open(os.path.join(path, "index.jsonl"), "w") as f:
        for l in lines:
            f.write(json.dumps(l) + "\n")
    with open(os.path.join(path, "inkset.json"), "w") as f:
        json.dump({"schema": "rfpii/nn-inkset/1", "max_side": max(size), "pages": n, "personal_data": False}, f)
    return path
