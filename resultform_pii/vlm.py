"""Vision-language model assistance through a local Ollama server.

The template pipeline is the fast path; a VLM is slow (seconds per page on a
GPU), so it is used where judgement helps and speed doesn't matter:

audit      After redaction, ask whether any complete ID number, phone number or
           signature is still readable anywhere on the page, including
           free-text areas such as the presiding officer's comments. A "yes"
           sends the form to review. A gate, not a guarantee: a VLM can miss
           things, so it adds to the template's guarantees rather than
           replacing them.
locate     Find signatures, ID numbers and phone numbers on a page. Used to
           (a) draft zones for a new template (``rfpii propose``) and
           (b) redact pages no template matches. Fallback output always goes
           to review/vlm/ for a person to approve; it is never auto-published.

Any vision model served by Ollama works; grounding-capable models such as
``qwen2.5vl`` give usable boxes. Run ``ollama pull qwen2.5vl:7b`` first.
"""

from __future__ import annotations

import base64
import json
import os
import urllib.request
from dataclasses import dataclass, field

import cv2
import numpy as np

DEFAULT_MODEL = "qwen2.5vl:7b"
LABELS = ("signature", "id_number", "phone_number")

AUDIT_PROMPT = """You are checking a Kenyan election results form before it is published.
Black rectangles are intentional redactions. ID numbers and phone numbers are
deliberately only partly hidden, so seeing a few isolated digits is expected and fine.
Names of officials and agents are allowed to be visible.

Look at the whole page, including tables, margins, stamps and handwritten comments.
Answer:
- id_number_visible: can you read a COMPLETE national ID or passport number (7 or more consecutive digits, unbroken by black boxes)?
- phone_number_visible: can you read a COMPLETE phone number (10 digits such as 07XXXXXXXX or 01XXXXXXXX, or +254...)?
- signature_visible: is any handwritten signature visible that is not covered by black?
- details: one short sentence naming where, if anything is visible."""

LOCATE_PROMPT = """This image is {w}x{h} pixels and shows a Kenyan election results form.
Find every handwritten signature, every national ID or passport number, and every
phone number (in tables or anywhere else, e.g. comments or margins).
Return one item per occurrence with a tight bounding box [x1, y1, x2, y2] in pixel
coordinates of this image. Do not return names, vote counts, dates or printed text."""

AUDIT_SCHEMA = {
    "type": "object",
    "properties": {
        "id_number_visible": {"type": "boolean"},
        "phone_number_visible": {"type": "boolean"},
        "signature_visible": {"type": "boolean"},
        "details": {"type": "string"},
    },
    "required": ["id_number_visible", "phone_number_visible", "signature_visible", "details"],
}

LOCATE_SCHEMA = {
    "type": "object",
    "properties": {
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "label": {"type": "string", "enum": list(LABELS)},
                    "bbox": {"type": "array", "items": {"type": "number"}, "minItems": 4, "maxItems": 4},
                },
                "required": ["label", "bbox"],
            },
        }
    },
    "required": ["items"],
}


@dataclass
class VlmSettings:
    host: str = field(default_factory=lambda: os.environ.get("OLLAMA_HOST", "http://localhost:11434"))
    model: str = DEFAULT_MODEL
    max_side: int = 1288  # long side sent to the model (multiple of 28 for Qwen-VL patching)
    coords: str = "pixel"  # "pixel" (Qwen2.5-VL) or "norm1000" (Qwen2-VL style 0..1000)
    timeout: float = 300.0


@dataclass
class AuditResult:
    ok: bool
    findings: dict


@dataclass
class Box:
    label: str
    x0: float
    y0: float
    x1: float
    y1: float


class Ollama:
    def __init__(self, s: VlmSettings | None = None):
        self.s = s or VlmSettings()
        if "://" not in self.s.host:
            self.s.host = "http://" + self.s.host

    def _resize(self, img: np.ndarray) -> tuple[np.ndarray, float]:
        h, w = img.shape[:2]
        f = min(1.0, self.s.max_side / max(h, w))
        # round to multiples of 28 so the model sees exactly these pixels
        nw, nh = max(28, int(w * f) // 28 * 28), max(28, int(h * f) // 28 * 28)
        return cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA), nw / w

    def ask(self, prompt: str, img: np.ndarray, schema: dict) -> dict:
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if not ok:
            raise ValueError("encode failed")
        body = {
            "model": self.s.model,
            "messages": [{"role": "user", "content": prompt,
                          "images": [base64.b64encode(buf.tobytes()).decode()]}],
            "format": schema,
            "stream": False,
            "options": {"temperature": 0},
        }
        req = urllib.request.Request(self.s.host.rstrip("/") + "/api/chat", json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.s.timeout) as r:
            reply = json.load(r)
        return json.loads(reply["message"]["content"])

    def audit(self, img: np.ndarray) -> AuditResult:
        small, _ = self._resize(img)
        f = self.ask(AUDIT_PROMPT, small, AUDIT_SCHEMA)
        leak = any(bool(f.get(k)) for k in ("id_number_visible", "phone_number_visible", "signature_visible"))
        return AuditResult(not leak, f)

    def locate(self, img: np.ndarray) -> list[Box]:
        small, scale = self._resize(img)
        h, w = small.shape[:2]
        out = self.ask(LOCATE_PROMPT.format(w=w, h=h), small, LOCATE_SCHEMA)
        boxes = []
        for it in out.get("items", []):
            if it.get("label") not in LABELS or len(it.get("bbox", [])) != 4:
                continue
            x0, y0, x1, y1 = map(float, it["bbox"])
            if self.s.coords == "norm1000":
                x0, x1, y0, y1 = x0 * w / 1000, x1 * w / 1000, y0 * h / 1000, y1 * h / 1000
            x0, x1 = sorted((x0, x1))
            y0, y1 = sorted((y0, y1))
            if x1 - x0 < 2 or y1 - y0 < 2:
                continue
            boxes.append(Box(it["label"], x0 / scale, y0 / scale, x1 / scale, y1 / scale))
        return boxes
