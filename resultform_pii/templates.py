"""Form templates: a reference image plus the zones that hold PII.

A template lives in its own directory:

    templates/<name>/
        template.yaml
        reference.png      # clean scan or render of the blank form

template.yaml:

    name: form_a
    reference: reference.png
    padding: 12            # px (template space) added around every zone
    zones:
      - name: presiding_officer_name
        rect: [x, y, w, h]
      - name: agent_signatures
        polygon: [[x1, y1], [x2, y2], ...]

All coordinates are pixels in the reference image.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import yaml


@dataclass
class Zone:
    name: str
    polygon: np.ndarray  # (N, 2) float32, template pixel coords


@dataclass
class Template:
    name: str
    image: np.ndarray  # grayscale reference
    zones: list[Zone]
    padding: int = 12
    # Filled in by prepare(): computed once per worker, reused for every page.
    mask: np.ndarray | None = field(default=None, repr=False)
    keypoints: np.ndarray | None = field(default=None, repr=False)
    descriptors: np.ndarray | None = field(default=None, repr=False)

    @property
    def size(self) -> tuple[int, int]:
        h, w = self.image.shape[:2]
        return w, h

    def build_mask(self) -> np.ndarray:
        """Template-space mask (255 = redact) with padding applied."""
        h, w = self.image.shape[:2]
        mask = np.zeros((h, w), np.uint8)
        for z in self.zones:
            cv2.fillPoly(mask, [np.round(z.polygon).astype(np.int32)], 255)
        if self.padding > 0:
            k = 2 * self.padding + 1
            mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
        return mask


def _zone_from_spec(spec: dict) -> Zone:
    if "rect" in spec:
        x, y, w, h = spec["rect"]
        poly = [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]
    elif "polygon" in spec:
        poly = spec["polygon"]
    else:
        raise ValueError(f"zone {spec.get('name')!r} needs 'rect' or 'polygon'")
    return Zone(spec.get("name", "zone"), np.asarray(poly, np.float32))


def load_template(directory: str | Path) -> Template:
    directory = Path(directory)
    spec = yaml.safe_load((directory / "template.yaml").read_text())
    image = cv2.imread(str(directory / spec.get("reference", "reference.png")), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"reference image missing for template in {directory}")
    zones = [_zone_from_spec(z) for z in spec.get("zones", [])]
    if not zones:
        raise ValueError(f"template {directory} defines no zones")
    return Template(spec.get("name", directory.name), image, zones, int(spec.get("padding", 12)))


def load_templates(root: str | Path) -> list[Template]:
    root = Path(root)
    if (root / "template.yaml").exists():
        return [load_template(root)]
    templates = [load_template(p.parent) for p in sorted(root.glob("*/template.yaml"))]
    if not templates:
        raise FileNotFoundError(f"no templates found under {root}")
    return templates
