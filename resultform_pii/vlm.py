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

Any vision model served by Ollama works, local or Ollama Cloud:

- Ollama Cloud directly: set OLLAMA_API_KEY (from ollama.com/settings/keys);
  requests go to https://ollama.com/api/chat with a bearer token.
- Through a local Ollama signed in to the cloud (`ollama signin`): use a
  ``*-cloud`` model name with the local host.
- Fully local: e.g. ``qwen2.5vl:7b`` after ``ollama pull``.

Check the current cloud catalogue (ollama.com/search?c=cloud) for model names.

Privacy and scrutiny: the audit only ever sends the *redacted* page. Sending
an unredacted form (fallback redaction, template proposals) to a non-local
host is refused unless explicitly allowed. Every verdict is recorded in the
document's manifest with the model, host and a hash of the exact image sent,
and a VLM never edits pixels itself: its boxes go through the same masking
and the same block-level proof as everything else.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import urllib.parse
import urllib.request
from dataclasses import dataclass, field

import cv2
import numpy as np

CLOUD_HOST = "https://ollama.com"
# Qwen's newest models with image input on Ollama (Oct 2026). Qwen3.8 is the
# latest family (vision built in) but has no Ollama Cloud tag that we could
# confirm; Qwen3-VL 235B is the newest Qwen vision model offered on Ollama
# Cloud. Both report boxes on a 0-1000 grid. Override with --vlm-model.
DEFAULT_CLOUD_MODEL = "qwen3-vl:235b"  # direct Ollama Cloud API name; "qwen3-vl:235b-cloud" via a local proxy
DEFAULT_LOCAL_MODEL = "qwen3.8:27b"


def _default_host() -> str:
    if os.environ.get("OLLAMA_HOST"):
        return os.environ["OLLAMA_HOST"]
    return CLOUD_HOST if os.environ.get("OLLAMA_API_KEY") else "http://localhost:11434"
LABELS = ("signature", "id_number", "phone_number")
LOCAL_HOSTS = ("localhost", "127.0.0.1", "::1")

AUDIT_PROMPT = """You are checking a Kenyan election results form before it is published.
Black rectangles are intentional redactions. Each ID/passport number has 3 of its
characters covered by black boxes, and each phone number keeps its first 4 digits
(e.g. 0734) and has 3 of its remaining 6 digits covered. So most digits of these
numbers are meant to be visible; what must never be visible is a COMPLETE number.
Names of officials and agents are allowed to be visible.

Look at the whole page, including tables, margins, stamps and handwritten comments.
Answer:
- id_number_visible: is any national ID or passport number fully readable, with no black box hiding any of its characters?
- phone_number_visible: is any phone number (07XXXXXXXX, 01XXXXXXXX, +254...) fully readable, with no black box hiding any digit?
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
    host: str = field(default_factory=_default_host)
    model: str = ""  # "" = default for the host (cloud vs local)
    max_side: int = 1288  # long side sent to the model (multiple of 28 for Qwen-VL patching)
    coords: str = "auto"  # "pixel" (Qwen2.5-VL), "norm1000" (Qwen3-VL, Qwen2-VL), "auto" = by model name
    timeout: float = 300.0
    api_key: str | None = field(default_factory=lambda: os.environ.get("OLLAMA_API_KEY"))
    allow_remote_unredacted: bool = False  # may unredacted forms be sent to a non-local host?


@dataclass
class AuditResult:
    ok: bool
    findings: dict
    record: dict = field(default_factory=dict)  # for the manifest


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
        if not self.s.model:
            self.s.model = DEFAULT_LOCAL_MODEL if self.local else DEFAULT_CLOUD_MODEL
        if self.s.coords == "auto":
            m = self.s.model.lower()
            self.s.coords = "norm1000" if ("qwen3" in m or "qwen2-vl" in m) else "pixel"
        if not self.local and self.s.host.rstrip("/") == CLOUD_HOST and not self.s.api_key:
            raise ValueError("Ollama Cloud needs OLLAMA_API_KEY (create one at ollama.com/settings/keys)")

    @property
    def local(self) -> bool:
        return urllib.parse.urlparse(self.s.host).hostname in LOCAL_HOSTS

    def _guard_unredacted(self) -> None:
        if not self.local and not self.s.allow_remote_unredacted:
            raise PermissionError(
                f"refusing to send an unredacted form to {self.s.host}; use a local model or pass "
                "--allow-remote-unredacted if that is an approved data transfer")

    def _resize(self, img: np.ndarray) -> tuple[np.ndarray, tuple[float, float]]:
        h, w = img.shape[:2]
        f = min(1.0, self.s.max_side / max(h, w))
        # round to multiples of 28 so the model sees exactly these pixels
        nw, nh = max(28, int(w * f) // 28 * 28), max(28, int(h * f) // 28 * 28)
        # per-axis: rounding each side to 28 px makes the two factors differ
        return cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA), (nw / w, nh / h)

    def ask(self, prompt: str, img: np.ndarray, schema: dict) -> tuple[dict, str]:
        """Returns the parsed reply and the sha256 of the exact image sent."""
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if not ok:
            raise ValueError("encode failed")
        sent = buf.tobytes()
        body = {
            "model": self.s.model,
            "messages": [{"role": "user", "content": prompt, "images": [base64.b64encode(sent).decode()]}],
            "format": schema,
            "stream": False,
            "options": {"temperature": 0},
        }
        headers = {"Content-Type": "application/json"}
        if self.s.api_key and not self.local:
            headers["Authorization"] = f"Bearer {self.s.api_key}"
        req = urllib.request.Request(self.s.host.rstrip("/") + "/api/chat", json.dumps(body).encode(), headers)
        with urllib.request.urlopen(req, timeout=self.s.timeout) as r:
            reply = json.load(r)
        return json.loads(reply["message"]["content"]), hashlib.sha256(sent).hexdigest()

    def audit(self, img: np.ndarray) -> AuditResult:
        """Audit a *redacted* page (safe to send anywhere)."""
        small, _ = self._resize(img)
        f, sent = self.ask(AUDIT_PROMPT, small, AUDIT_SCHEMA)
        leak = any(bool(f.get(k)) for k in ("id_number_visible", "phone_number_visible", "signature_visible"))
        record = {"model": self.s.model, "host": urllib.parse.urlparse(self.s.host).hostname,
                  "prompt_sha256": hashlib.sha256(AUDIT_PROMPT.encode()).hexdigest(),
                  "image_sent_sha256": sent, "verdict": f}
        return AuditResult(not leak, f, record)

    def locate(self, img: np.ndarray) -> list[Box]:
        """Find PII on an *unredacted* page (guarded for remote hosts)."""
        self._guard_unredacted()
        small, scale = self._resize(img)
        h, w = small.shape[:2]
        out, _ = self.ask(LOCATE_PROMPT.format(w=w, h=h), small, LOCATE_SCHEMA)
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
            sx, sy = scale
            boxes.append(Box(it["label"], x0 / sx, y0 / sy, x1 / sx, y1 / sy))
        return boxes
