"""Synthetic polling-station result form, used for the demo template and tests.

Layout is generic (not any real authority's form): a header, a candidate
results table (public data) and a declaration block with officials' and
agents' names, ID numbers, phone numbers and signatures (PII).
"""

from __future__ import annotations

import cv2
import numpy as np

W, H = 1240, 1754  # A4 at 150 dpi
FONT = cv2.FONT_HERSHEY_SIMPLEX

# PII zones in template coordinates: name -> (x, y, w, h)
ZONES = {
    "presiding_officer": (330, 1210, 860, 110),
    "deputy_officer": (330, 1330, 860, 110),
    "agents_table": (60, 1500, 1120, 200),
}


def _text(img, s, xy, scale=0.6, thick=1):
    cv2.putText(img, s, xy, FONT, scale, 0, thick, cv2.LINE_AA)


def blank_form() -> np.ndarray:
    img = np.full((H, W), 255, np.uint8)
    cv2.rectangle(img, (40, 40), (W - 40, H - 40), 0, 3)
    _text(img, "ELECTION RESULTS AT THE POLLING STATION", (180, 110), 1.0, 2)
    _text(img, "FORM RS-1  (SAMPLE TEMPLATE)", (380, 160), 0.8, 2)
    for i, label in enumerate(["County", "Constituency", "Ward", "Polling Station", "Station Code"]):
        y = 230 + i * 50
        _text(img, f"{label}:", (70, y))
        cv2.line(img, (300, y + 5), (1180, y + 5), 0, 1)
    # Results table: public data, must survive redaction.
    top, rows = 520, 8
    cv2.rectangle(img, (60, top), (1180, top + 60 * (rows + 1)), 0, 2)
    for c, (x, hdr) in enumerate([(60, "No."), (160, "Candidate"), (700, "Party"), (950, "Votes")]):
        cv2.line(img, (x, top), (x, top + 60 * (rows + 1)), 0, 2)
        _text(img, hdr, (x + 15, top + 40), 0.7, 2)
    for r in range(1, rows + 1):
        y = top + 60 * r
        cv2.line(img, (60, y), (1180, y), 0, 1)
        _text(img, str(r), (90, y + 40))
    _text(img, "Total valid votes:", (70, 1110), 0.7, 2)
    _text(img, "Rejected:", (650, 1110), 0.7, 2)
    _text(img, "DECLARATION", (70, 1180), 0.8, 2)
    for i, role in enumerate(["Presiding Officer", "Deputy P. Officer"]):
        y = 1210 + 120 * i
        _text(img, role, (70, y + 40), 0.6, 2)
        cv2.rectangle(img, (330, y), (1190, y + 110), 0, 1)
        _text(img, "Name", (340, y + 25), 0.5)
        _text(img, "ID No.", (760, y + 25), 0.5)
        _text(img, "Signature", (340, y + 95), 0.5)
        _text(img, "Phone", (760, y + 95), 0.5)
    _text(img, "AGENTS", (70, 1480), 0.8, 2)
    cv2.rectangle(img, (60, 1500), (1180, 1700), 0, 1)
    for r in range(1, 4):
        cv2.line(img, (60, 1500 + 50 * r), (1180, 1500 + 50 * r), 0, 1)
    return img


def filled_form(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    img = blank_form()
    for i, v in enumerate(["NORTH", "RIVERSIDE", "CENTRAL", "St. Mary Primary", "045-112-03"]):
        _text(img, v, (320, 225 + i * 50), 0.7, 2)
    for r in range(1, 9):
        y = 520 + 60 * r
        _text(img, f"Candidate {chr(64 + r)}", (180, y + 40), 0.7, 2)
        _text(img, f"PTY{r}", (720, y + 40), 0.7, 2)
        _text(img, str(int(rng.integers(0, 500))), (980, y + 40), 0.9, 2)
    for i in range(2):
        y = 1210 + 120 * i
        _text(img, "JANE A. DOE" if i == 0 else "JOHN B. ROE", (420, y + 30), 0.8, 2)
        _text(img, f"{int(rng.integers(10**7, 10**8))}", (860, y + 30), 0.8, 2)
        _text(img, f"07{int(rng.integers(10**7, 10**8))}", (840, y + 100), 0.8, 2)
        pts = np.cumsum(rng.integers(-12, 13, (25, 2)), axis=0) + (520, y + 85)
        cv2.polylines(img, [pts.astype(np.int32)], False, 0, 2)  # "signature"
    for r in range(4):
        _text(img, f"Agent {r + 1}  ID {int(rng.integers(10**7, 10**8))}  Tel 07{int(rng.integers(10**7, 10**8))}",
              (80, 1535 + 50 * r), 0.7, 2)
    return img


def simulate_scan(img: np.ndarray, seed: int = 0, angle: float = 3.0, out_scale: float = 1.3):
    """Rotate, scale, add perspective, lighting and noise. Returns (scan_bgr, H)
    where H maps template coords to scan coords."""
    rng = np.random.default_rng(seed)
    h, w = img.shape[:2]
    ow, oh = int(w * out_scale), int(h * out_scale)
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    jitter = rng.uniform(-0.02, 0.02, (4, 2)) * [w, h]
    a = np.deg2rad(angle)
    R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    dst = (src - [w / 2, h / 2] + jitter) @ R.T * 1.1 + [ow / 2, oh / 2]
    Hm = cv2.getPerspectiveTransform(src, dst.astype(np.float32))
    scan = cv2.warpPerspective(img, Hm, (ow, oh), borderValue=200)
    grad = np.linspace(0.85, 1.05, ow)[None, :]
    scan = np.clip(scan * grad + rng.normal(0, 6, scan.shape), 0, 255).astype(np.uint8)
    return cv2.cvtColor(scan, cv2.COLOR_GRAY2BGR), Hm


def write_template(directory) -> None:
    from pathlib import Path

    import yaml

    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(d / "reference.png"), blank_form())
    spec = {
        "name": "sample_rs1",
        "reference": "reference.png",
        "padding": 12,
        "zones": [{"name": k, "rect": list(v)} for k, v in ZONES.items()],
    }
    (d / "template.yaml").write_text(yaml.safe_dump(spec, sort_keys=False))
