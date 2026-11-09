"""Ink sets (``rfpii dataset ink``): pages and the pipeline's own ink masks,
and how a stem's masks are scored against them.

An ink set is a folder: ``inkset.json`` (``rfpii/nn-inkset/1``),
``pages/<id>.png``, ``masks/<id>.png`` (bits: 1 dark ink, 2 rules, 4 stamp
ink), ``index.jsonl`` and ``SHA256SUMS``. Pages are split by their split
key (by constituency for real forms, by seed for synthetic ones) with the
datasets' rule: ``train`` is the training pages, ``val`` and ``test``
together the held-out ones.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from .stem import OUTPUT_BITS, OUTPUTS

SCHEMA = "rfpii/nn-inkset/1"


class InkSet:
    def __init__(self, root):
        self.root = Path(root)
        meta = json.loads((self.root / "inkset.json").read_text())
        if meta.get("schema") != SCHEMA:
            raise ValueError(f"{self.root}: not an ink set ({meta.get('schema')!r})")
        self.meta = meta
        self.index = [json.loads(l) for l in (self.root / "index.jsonl").read_text().splitlines() if l.strip()]

    def __len__(self):
        return len(self.index)

    def pages(self, side=None, exclude=()):
        """Index lines: side "train", "held" (val and test) or None (all)."""
        ex = set(exclude)
        out = []
        for l in self.index:
            if l["id"] in ex:
                continue
            if side == "train" and l["split"] != "train":
                continue
            if side == "held" and l["split"] == "train":
                continue
            out.append(l)
        return out

    def image(self, line):
        """(H, W, 3) uint8, R, G, B."""
        return np.asarray(Image.open(self.root / line["image"]).convert("RGB"))

    def mask(self, line):
        """(H, W) uint8 bits."""
        return np.asarray(Image.open(self.root / line["mask"]))


def open_sets(paths):
    return [InkSet(p) for p in paths]


def pages_of(sets, side=None, exclude=()):
    """[(set, line)] over several ink sets."""
    return [(s, l) for s in sets for l in s.pages(side, exclude)]


# --- scoring ------------------------------------------------------------------------------


def counts(pred, bits):
    """{output: [tp, fp, fn, tn]} of predicted probabilities against a mask.

    pred: {output: (H, W) probability}; a pixel is marked above 0.5.
    """
    out = {}
    for n in OUTPUTS:
        p = np.asarray(pred[n]) > 0.5
        t = (bits & OUTPUT_BITS[n]) != 0
        tp = int(np.count_nonzero(p & t))
        fp = int(np.count_nonzero(p & ~t))
        fn = int(np.count_nonzero(~p & t))
        out[n] = [tp, fp, fn, int(p.size) - tp - fp - fn]
    return out


def add(total, c):
    for n, v in c.items():
        total[n] = [a + b for a, b in zip(total.get(n, [0, 0, 0, 0]), v)]
    return total


def summary(total):
    """Agreement (share of pixels where the stem and the pipeline agree) and
    IoU of the ink class, per output, from summed counts; plus recall and
    precision of the ink class, and the pipeline's ink share."""
    s = {}
    for n in OUTPUTS:
        tp, fp, fn, tn = total.get(n, [0, 0, 0, 0])
        px = tp + fp + fn + tn
        s[n] = {
            "agreement": (tp + tn) / px if px else 1.0,
            "iou": tp / (tp + fp + fn) if tp + fp + fn else 1.0,
            "recall": tp / (tp + fn) if tp + fn else 1.0,
            "precision": tp / (tp + fp) if tp + fp else 1.0,
            "ink_share": (tp + fn) / px if px else 0.0,
            "pixels": px,
        }
    s["min_agreement"] = min(s[n]["agreement"] for n in OUTPUTS)
    s["mean_iou"] = sum(s[n]["iou"] for n in OUTPUTS) / len(OUTPUTS)
    return s


def passes(s, at=0.99):
    """The milestone's test: every mask agrees with the pipeline's on at
    least 99% of the pixels."""
    return s["min_agreement"] >= at


def score(forward, pages, per_page=False):
    """Score ``forward(rgb) -> {output: prob}`` on [(set, line)] pages."""
    total, rows = {}, []
    for st, line in pages:
        c = counts(forward(st.image(line)), st.mask(line))
        add(total, c)
        if per_page:
            rows.append({"id": line["id"], "group": line.get("group"), "split": line["split"], **summary(c)})
    out = summary(total)
    out["pages"] = len(pages)
    if per_page:
        out["per_page"] = rows
    return out


def fmt(s):
    """One line per output: agreement and IoU."""
    lines = []
    for n in OUTPUTS:
        v = s[n]
        lines.append(f"{n:6s} agreement {v['agreement']*100:7.3f}%  IoU {v['iou']:.3f}  recall {v['recall']:.3f}  precision {v['precision']:.3f}  ink {v['ink_share']*100:5.2f}%")
    return "\n".join(lines)
