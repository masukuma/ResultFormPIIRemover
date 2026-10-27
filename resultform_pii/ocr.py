"""Local OCR of printed labels (RapidOCR / PP-OCR models on ONNX Runtime).

Used to find the printed labels ("ID/Passport No.", "Tel. Contact",
"Signature", "Date", ...) of forms there is no template for. The models ship
inside the ``rapidocr_onnxruntime`` wheel and run on the CPU; nothing leaves
the machine. Their hashes go into every manifest made with them, so a re-run
can be checked against the same models.

Recognition is decoded here rather than by the library so that each
character comes with its position along the text line: a label is often read
together with handwriting next to it ("Presiding Officer JOHN DOE Signature"),
and the field starts where the word "Signature" ends, not where the line does.
"""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np


@dataclass
class Line:
    quad: np.ndarray  # 4x2: top-left, top-right, bottom-right, bottom-left (image px)
    text: str
    score: float
    pos: list[float] = field(default_factory=list)  # per character: centre as a share of the line's width
    probs: list[float] = field(default_factory=list)  # per character: recognition confidence

    @property
    def box(self) -> tuple[float, float, float, float]:
        x0, y0 = self.quad.min(axis=0)
        x1, y1 = self.quad.max(axis=0)
        return float(x0), float(y0), float(x1), float(y1)

    @property
    def height(self) -> float:
        return float(np.linalg.norm(self.quad[3] - self.quad[0]) + np.linalg.norm(self.quad[2] - self.quad[1])) / 2

    @property
    def cy(self) -> float:
        return float(self.quad[:, 1].mean())

    def chars_in(self, x0: float, x1: float) -> list[tuple[str, float]]:
        """(character, confidence) of the characters centred between image x0 and x1."""
        if not self.pos or len(self.pos) != len(self.text):
            return []
        return [(c, p) for c, s, p in zip(self.text, self.pos, self.probs or [self.score] * len(self.text))
                if x0 <= self.x_at(s) <= x1]

    def x_at(self, share: float) -> float:
        """Image x of a point ``share`` of the way along the line."""
        top = self.quad[0] + (self.quad[1] - self.quad[0]) * share
        bot = self.quad[3] + (self.quad[2] - self.quad[3]) * share
        return float((top[0] + bot[0]) / 2)

    def find(self, pattern: str | re.Pattern) -> tuple[float, float] | None:
        """Image x-range (x0, x1) of the first match of ``pattern`` in the text."""
        m = re.search(pattern, self.text, re.I) if isinstance(pattern, str) else pattern.search(self.text)
        if not m:
            return None
        return self.span(m.start(), m.end())

    def fuzzy(self, word: str, max_err: int) -> list[tuple[int, int, int]]:
        """Non-overlapping approximate occurrences of ``word`` (case-insensitive):
        (start, end, edits), best first. OCR misreads labels ("Pate", "Shature")
        and a label cut by a stamp is a prefix ("Signatu") at the end of the line."""
        t, w = self.text.lower(), word.lower()
        found = []
        for i in range(len(t)):
            for L in range(max(1, len(w) - max_err), len(w) + max_err + 1):
                if i + L > len(t):
                    break
                d = _edits(t[i:i + L], w)
                if i + L == len(t) and L < len(w):  # truncated at the end of the line
                    d = min(d, _edits(t[i:], w[:L]) + (1 if L < len(w) - 2 else 0))
                if d <= max_err:
                    found.append((d, -L, i, i + L))
        out: list[tuple[int, int, int]] = []
        for d, _, i, j in sorted(found):
            if all(j <= a or i >= b for a, b, _ in out):
                out.append((i, j, d))
        return out

    def span(self, i: int, j: int) -> tuple[float, float]:
        """Image x-range of characters i..j-1."""
        n = len(self.text)
        if not self.pos or len(self.pos) != n:  # fall back to even spacing
            return self.x_at(i / max(n, 1)), self.x_at(j / max(n, 1))
        half = 0.5 / max(n, 1)
        if j - i > 1:  # half a character beyond the outer centres
            half = (self.pos[j - 1] - self.pos[i]) / (2 * (j - i - 1))
        return self.x_at(max(0.0, self.pos[i] - half)), self.x_at(min(1.0, self.pos[j - 1] + half))


def _edits(a: str, b: str) -> int:
    """Levenshtein distance."""
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


class Ocr:
    def __init__(self):
        from rapidocr_onnxruntime import RapidOCR  # optional dependency: pip install .[generic]

        self.engine = RapidOCR()
        self.rec = self.engine.text_recognizer
        self.chars = self.rec.postprocess_op.character  # index 0 = CTC blank

    @staticmethod
    def info() -> dict:
        """Engine version and model hashes, for the manifest."""
        import rapidocr_onnxruntime as r
        root = Path(r.__file__).parent
        models = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((root / "models").glob("*.onnx"))}
        try:
            from importlib.metadata import version
            v = version("rapidocr_onnxruntime")
        except Exception:
            v = None
        return {"engine": "rapidocr_onnxruntime", "version": v, "models": models}

    def detect(self, img: np.ndarray) -> list[np.ndarray]:
        boxes, _ = self.engine.text_detector(img)
        return [np.float32(b) for b in (boxes if boxes is not None else [])]

    def classify(self, img: np.ndarray, quads: list[np.ndarray]) -> list[tuple[str, float]]:
        """Text-line orientation ('0' / '180', confidence) per quad."""
        crops = self.engine.get_crop_img_list(img, quads)
        _, res, _ = self.engine.text_cls(crops)
        return [(str(a), float(b)) for a, b in res]

    def read(self, img: np.ndarray, min_score: float = 0.3) -> list[Line]:
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        quads = [q for q in self.detect(img)
                 if np.linalg.norm(q[1] - q[0]) >= 0.8 * np.linalg.norm(q[3] - q[0])]  # horizontal lines only
        quads.sort(key=lambda q: (float(q[:, 1].mean()), float(q[:, 0].min())))
        crops = self.engine.get_crop_img_list(img, quads)
        lines = []
        for q, crop, (text, score, pos, probs) in zip(quads, crops, self._recognize(crops)):
            if text.strip() and score >= min_score:
                lines.append(Line(q, text, score, pos, probs))
        return lines

    def _recognize(self, crops: list[np.ndarray], batch: int = 8):
        c, H, _ = self.rec.rec_image_shape
        order = np.argsort([cr.shape[1] / max(cr.shape[0], 1) for cr in crops])
        out: list = [("", 0.0, [], [])] * len(crops)
        for b in range(0, len(order), batch):
            idx = order[b:b + batch]
            ratio = max(crops[i].shape[1] / max(crops[i].shape[0], 1) for i in idx)
            W = int(H * ratio)
            norm, widths = [], []
            for i in idx:
                h, w = crops[i].shape[:2]
                rw = min(W, int(math.ceil(H * w / max(h, 1))))
                im = cv2.resize(crops[i], (max(rw, 1), H)).astype(np.float32).transpose(2, 0, 1) / 255
                pad = np.zeros((c, H, W), np.float32)
                pad[:, :, :rw] = (im - 0.5) / 0.5
                norm.append(pad)
                widths.append(max(rw, 1))
            preds = self.rec.session(np.stack(norm))[0]  # (n, T, classes)
            T = preds.shape[1]
            for k, i in enumerate(idx):
                out[i] = self._decode(preds[k], W / T, widths[k])
        return out

    def _decode(self, p: np.ndarray, stride: float, width: int):
        best, prob = p.argmax(1), p.max(1)
        text, probs, pos = [], [], []
        t = 0
        while t < len(best):
            k = best[t]
            u = t
            while u + 1 < len(best) and best[u + 1] == k:
                u += 1
            if k != 0:
                text.append(self.chars[int(k)])
                probs.append(float(prob[t:u + 1].max()))
                pos.append(min(1.0, ((t + u) / 2 + 0.5) * stride / width))
            t = u + 1
        return "".join(text), float(np.mean(probs)) if probs else 0.0, pos, probs


@lru_cache(maxsize=1)
def engine() -> Ocr:
    return Ocr()


def available() -> bool:
    try:
        import rapidocr_onnxruntime  # noqa: F401
        return True
    except ImportError:
        return False
