"""KIEMS results-transmission metadata: file names and form QR codes.

Results forms captured and transmitted by KIEMS kits arrive as files named

    1_34_A_007029014000102_F230450M00203609_20220810_132155.PDF
    | |  | |               |                |        |
    | |  | |               |                |        time of capture (HHMMSS)
    | |  | |               |                date of capture (YYYYMMDD)
    | |  | |               KIEMS kit serial
    | |  | polling station code: county(3) constituency(3) ward(4) centre(3) stream(2)
    | |  form suffix
    | form number (34 = presidential)
    election type (1 = presidential)

The printed forms carry a QR code. On Form 34A it encodes the 15-digit
polling station code followed by the form number ("00702901400010234"); on
Form 34B it encodes election, county, constituency, form and page
("GE2022040228FORM34B03").

None of this is personal data. It is kept and published in a sidecar so
readers can tie each redacted form to its polling station and to the original
transmission (via the source file's SHA-256), and a QR that disagrees with
the file name sends the form to review instead of publishing it under the
wrong station.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass

import cv2
import numpy as np

NAME_RE = re.compile(
    r"^(?P<election>\d)_(?P<form>\d{2})_(?P<suffix>[A-Z])_(?P<station>\d{15})_(?P<kit>[A-Z0-9]+)"
    r"_(?P<date>\d{8})_(?P<time>\d{6})$", re.I)
QR_34A_RE = re.compile(r"^(?P<station>\d{15})(?P<form>\d{2})$")
QR_PAGE_RE = re.compile(
    r"^(?P<election>[A-Z]{2})(?P<year>\d{4})(?P<county>\d{3})(?P<constituency>\d{3})"
    r"FORM(?P<form>\d{2}[A-Z])(?P<page>\d{2})$", re.I)

ELECTION_TYPES = {"1": "presidential"}


def split_station(code: str) -> dict:
    return {"county": code[0:3], "constituency": code[3:6], "ward": code[6:10],
            "centre": code[10:13], "stream": code[13:15]}


@dataclass
class KiemsName:
    election: str
    form: str  # e.g. "34A"
    station: str
    kit: str
    captured_at: str  # ISO 8601, kit local time

    def to_dict(self) -> dict:
        return {**asdict(self), **split_station(self.station),
                "election_type": ELECTION_TYPES.get(self.election, self.election)}


def parse_name(stem: str) -> KiemsName | None:
    m = NAME_RE.match(stem)
    if not m:
        return None
    d, t = m["date"], m["time"]
    return KiemsName(m["election"], f"{m['form']}{m['suffix'].upper()}", m["station"], m["kit"].upper(),
                     f"{d[:4]}-{d[4:6]}-{d[6:]}T{t[:2]}:{t[2:4]}:{t[4:]}")


_qr = None


def decode_qr(gray: np.ndarray) -> str | None:
    """Decode the form's QR code. It sits in the header, so try the top of the
    page first (faster, fewer false finds), then the whole page."""
    global _qr
    _qr = _qr or cv2.QRCodeDetector()
    h = gray.shape[0]
    for region in (gray[: h // 4], gray):
        for img in (region, cv2.resize(region, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)):
            try:
                ok, payloads, _, _ = _qr.detectAndDecodeMulti(img)
            except cv2.error:
                continue
            for p in payloads if ok else ():
                if p:
                    return p.strip()
    return None


def parse_qr(payload: str) -> dict:
    """Structured view of a form QR payload ({} if the format is unknown)."""
    if m := QR_34A_RE.match(payload):
        return {"station": m["station"], "form": f"{m['form']}A", **split_station(m["station"])}
    if m := QR_PAGE_RE.match(payload):
        return {"election": m["election"].upper(), "year": m["year"], "county": m["county"],
                "constituency": m["constituency"], "form": m["form"].upper(), "page": int(m["page"])}
    return {}


def check(name: KiemsName | None, qr: dict) -> str:
    """Reason the file name and QR disagree, or "" if they're consistent."""
    if not name or not qr:
        return ""
    if qr.get("form") and qr["form"] != name.form:
        return f"QR form {qr['form']} != file name form {name.form}"
    if qr.get("station") and qr["station"] != name.station:
        return f"QR station {qr['station']} != file name station {name.station}"
    if qr.get("county") and qr["county"] != name.station[:3]:
        return f"QR county {qr['county']} != file name county {name.station[:3]}"
    return ""
