"""Form templates: a reference image plus the zones that hold PII.

A template lives in its own directory:

    templates/<name>/
        template.yaml
        reference.png      # clean scan or photo of the form type

template.yaml:

    name: form34a_2022
    form: 34A              # matched against the form code in the QR / KIEMS file name
    reference: reference.png
    padding: 8             # px (template space) added around every zone
    protect:               # results regions: no mask may touch them
      - name: votes
        rect: [x, y, w, h]
    structure:             # printed tables whose rules re-position their zones
      - name: agents
        rect: [x, y, w, h]
    zones:
      - name: officer_signatures
        action: signature  # each signature blacked out on its own box
        rect: [x, y, w, h]
        search: [12, 24]   # px searched above/below for strokes that stray
      - name: agent_id
        action: partial    # mask random digits, keep the rest visible
        digits: 8          # expected digit count (ID 8, phone 10)
        rect: [x, y, w, h] # one cell...
        rows: 11           # ...repeated down a table
        row_step: 36.4
        region: agents     # placed by the table's rules; page fails if they can't be matched

Names are deliberately not zones: they stay visible. All coordinates are pixels
in the reference image.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import yaml

ACTIONS = ("blackout", "partial", "signature")


@dataclass
class Zone:
    name: str
    rect: tuple[float, float, float, float]  # x, y, w, h in template pixels
    action: str = "blackout"
    digits: int = 0  # partial only
    group: str = ""  # table column this cell belongs to ("" = standalone)
    row: int = 0
    region: str = ""  # structure region whose printed rules position this zone
    rule: str = ""  # partial zones: "phone" (keep prefix, hide 3 of last 6) or "id" (hide 3)
    search: tuple[float, float] | None = None  # px searched above/below the cells (default: MaskSettings.vpad)
    anchor: tuple[float, float, float, float] | None = None  # printed label (x, y, w, h) the zone moves with

    @property
    def polygon(self) -> np.ndarray:
        x, y, w, h = self.rect
        return np.float32([[x, y], [x + w, y], [x + w, y + h], [x, y + h]])

    def padded(self, pad: float) -> np.ndarray:
        x, y, w, h = self.rect
        return Zone(self.name, (x - pad, y - pad, w + 2 * pad, h + 2 * pad)).polygon


@dataclass
class Template:
    name: str
    image: np.ndarray  # grayscale reference
    zones: list[Zone]
    padding: int = 8
    form: str = ""
    structure: dict = field(default_factory=dict)  # region name -> rect (x, y, w, h)
    protect: dict = field(default_factory=dict)  # results regions no mask may touch: name -> rect
    fingerprint: str = ""  # sha256 of template.yaml + reference image, recorded in manifests
    min_area_frac: float | None = None  # override AlignSettings.min_area_frac (reference is part of a page)
    priority: int = 0  # among templates a page passes, the highest priority wins
    once: bool = False  # a document of this form must contain exactly one page matching this template
    page_long_side: float | None = None  # reference cut from a page: that page's long side (px)
    min_inliers: int | None = None  # override AlignSettings thresholds for this template
    min_score: float | None = None
    # Filled in by prepare(): computed once per worker, reused for every page.
    regions: dict = field(default_factory=dict, repr=False)  # name -> structure.Region
    mask: np.ndarray | None = field(default=None, repr=False)
    keypoints: np.ndarray | None = field(default=None, repr=False)
    descriptors: np.ndarray | None = field(default=None, repr=False)

    @property
    def size(self) -> tuple[int, int]:
        h, w = self.image.shape[:2]
        return w, h

    def build_mask(self) -> np.ndarray:
        """Template-space mask (255 = any zone) with padding applied."""
        h, w = self.image.shape[:2]
        mask = np.zeros((h, w), np.uint8)
        for z in self.zones:
            cv2.fillPoly(mask, [np.round(z.padded(self.padding)).astype(np.int32)], 255)
        return mask


def _zones_from_spec(spec: dict) -> list[Zone]:
    name = spec.get("name", "zone")
    action = spec.get("action", "blackout")
    if action not in ACTIONS:
        raise ValueError(f"zone {name!r}: action must be one of {ACTIONS}")
    if "rect" not in spec:
        raise ValueError(f"zone {name!r} needs 'rect'")
    digits = int(spec.get("digits", 0))
    if action == "partial" and digits < 2:
        raise ValueError(f"zone {name!r}: partial zones need 'digits' >= 2")
    rule = str(spec.get("rule", "phone" if digits >= 10 else "id")) if action == "partial" else ""
    if rule not in ("", "phone", "id"):
        raise ValueError(f"zone {name!r}: rule must be 'phone' or 'id'")
    x, y, w, h = map(float, spec["rect"])
    rows = int(spec.get("rows", 1))
    step = float(spec.get("row_step", h))
    return [
        Zone(f"{name}[{r}]" if rows > 1 else name, (x, y + r * step, w, h), action, digits,
             name if rows > 1 else "", r, str(spec.get("region", "")), rule,
             tuple(map(float, spec["search"])) if "search" in spec else None,
             tuple(map(float, spec["anchor"])) if "anchor" in spec else None)
        for r in range(rows)
    ]


def load_template(directory: str | Path) -> Template:
    directory = Path(directory)
    spec = yaml.safe_load((directory / "template.yaml").read_text())
    image = cv2.imread(str(directory / spec.get("reference", "reference.png")), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"reference image missing for template in {directory}")
    # A template may have no zones: pages of a known form that carry no PII
    # (e.g. Form 34B results pages) still have to be recognised to be published.
    zones = [z for s in spec.get("zones") or [] for z in _zones_from_spec(s)]
    structure = {r["name"]: tuple(map(float, r["rect"])) for r in spec.get("structure") or []}
    for z in zones:
        if z.region and z.region not in structure:
            raise ValueError(f"zone {z.name!r} refers to unknown structure region {z.region!r}")
    protect = {r["name"]: tuple(map(float, r["rect"])) for r in spec.get("protect") or []}
    h = hashlib.sha256((directory / "template.yaml").read_bytes())
    h.update((directory / spec.get("reference", "reference.png")).read_bytes())
    return Template(spec.get("name", directory.name), image, zones,
                    int(spec.get("padding", 8)), str(spec.get("form", "")), structure, protect, h.hexdigest(),
                    float(spec["min_area_frac"]) if "min_area_frac" in spec else None,
                    int(spec.get("priority", 0)), bool(spec.get("once", False)),
                    float(spec["page_long_side"]) if "page_long_side" in spec else None,
                    int(spec["min_inliers"]) if "min_inliers" in spec else None,
                    float(spec["min_score"]) if "min_score" in spec else None)


def load_templates(root: str | Path) -> list[Template]:
    root = Path(root)
    if (root / "template.yaml").exists():
        return [load_template(root)]
    templates = [load_template(p.parent) for p in sorted(root.glob("*/template.yaml"))]
    if not templates:
        raise FileNotFoundError(f"no templates found under {root}")
    return templates
