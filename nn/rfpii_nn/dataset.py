"""Read an rfpii dataset (`rfpii/nn-dataset/1`, see nn/README.md).

A dataset is a folder written by `rfpii dataset synth` or `rfpii dataset
build`: `dataset.json`, `index.jsonl`, `splits.json`, `pages/`, `labels/`,
`masks/` and `SHA256SUMS`. This module reads it; training reads pages through
`Dataset.pages`, which needs to be told what the gold set excludes
(`rfpii_nn.gold.Exclusion`), so that no gold page is ever trained on.
"""

import hashlib
import json
import os

import numpy as np
from PIL import Image

DATASET_SCHEMA = "rfpii/nn-dataset/1"
LABELS_SCHEMA = "rfpii/nn-labels/1"
CLASSES = ["signature", "id_number", "phone_number", "name", "stamp",
           "results_region", "agents_table", "declaration", "comments"]
SPLITS = ("train", "val", "test")

# mask bits (masks/<id>.png)
BIT_PRINT = 1   # print and rules
BIT_PEN = 2     # pen
BIT_STAMP = 4   # stamp ink
BIT_BLACK = 8   # blacked out by redaction (redacted pages)


def split_of(key):
    """The split of a page from its split key, as `rfpii` computes it: the
    first 8 bytes of SHA-256(key), big-endian, as v; train if v*10 < 8*2^64,
    val if v*10 < 9*2^64, else test."""
    v = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big")
    if v * 10 < 8 << 64:
        return "train"
    if v * 10 < 9 << 64:
        return "val"
    return "test"


def seed_of(entry):
    """The seed of a synthetic page (its split key is `seed:N`), else None."""
    key = entry.get("split_key") or ""
    if key.startswith("seed:"):
        try:
            return int(key[5:])
        except ValueError:
            return None
    return None


class _Required:
    def __repr__(self):
        return "REQUIRED"


REQUIRED = _Required()


class DatasetError(ValueError):
    pass


class Dataset:
    """One dataset folder."""

    def __init__(self, path):
        self.path = os.path.abspath(path)
        with open(os.path.join(self.path, "dataset.json"), encoding="utf-8") as f:
            self.meta = json.load(f)
        if self.meta.get("schema") != DATASET_SCHEMA:
            raise DatasetError("%s: schema %r, expected %s" % (path, self.meta.get("schema"), DATASET_SCHEMA))
        self.index = []
        with open(os.path.join(self.path, "index.jsonl"), encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.index.append(json.loads(line))
        self.by_id = {e["id"]: e for e in self.index}

    @property
    def kind(self):
        return self.meta.get("kind")

    @property
    def personal_data(self):
        return bool(self.meta.get("personal_data"))

    def __len__(self):
        return len(self.index)

    def entry(self, page):
        return page if isinstance(page, dict) else self.by_id[page]

    def _file(self, rel):
        """A file of the dataset by its relative path; never outside it."""
        full = os.path.realpath(os.path.join(self.path, rel))
        if os.path.commonpath([full, os.path.realpath(self.path)]) != os.path.realpath(self.path):
            raise DatasetError("%s leaves the dataset" % rel)
        return full

    def image_path(self, page):
        return self._file(self.entry(page)["image"])

    def mask_path(self, page):
        rel = self.entry(page).get("mask")
        return self._file(rel) if rel else None

    def labels(self, page):
        with open(self._file(self.entry(page)["labels"]), encoding="utf-8") as f:
            labels = json.load(f)
        if labels.get("schema") != LABELS_SCHEMA:
            raise DatasetError("%s: labels schema %r" % (self.entry(page)["id"], labels.get("schema")))
        return labels

    def image(self, page):
        """The page as an RGB PIL image (pixels as stored: no EXIF turn)."""
        return Image.open(self.image_path(page)).convert("RGB")

    def mask(self, page):
        """The mask as a uint8 array (bits BIT_*), or None."""
        p = self.mask_path(page)
        if p is None:
            return None
        return np.array(Image.open(p))

    def pages(self, split=None, exclude=REQUIRED):
        """Index entries of `split` (or all), leaving out what `exclude`
        excludes. `exclude` must be given: the gold set's exclusion
        (`rfpii_nn.gold.Exclusion.load(gold_dir)`), or
        `rfpii_nn.gold.NOTHING` where no gold set applies (looking at data,
        tests). Training must pass the gold set's."""
        if exclude is REQUIRED:
            raise DatasetError("say what the gold set excludes: pass exclude=Exclusion.load(gold_dir) (or gold.NOTHING when not training)")
        for e in self.index:
            if split is not None and e.get("split") != split:
                continue
            if exclude is not None and exclude.excludes(e):
                continue
            yield e

    def verify_splits(self):
        """Pages whose recorded split isn't the one their key gives."""
        return [e["id"] for e in self.index if e.get("split") != split_of(e.get("split_key", ""))]


def training_pages(dataset, exclusion):
    """The pages a training run may use: the train split, less every gold
    page. Training code uses this (or `Dataset.pages` with the gold set's
    exclusion), never the index directly."""
    if exclusion is None:
        raise DatasetError("training needs the gold set's exclusion")
    return list(dataset.pages("train", exclude=exclusion))
