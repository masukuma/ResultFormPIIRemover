"""Edits that can be proven to change nothing but PII.

The redacted file must withstand the question "were any results changed?".
So the only edit ever made to a form image is replacing whole 8x8 pixel blocks
with solid black, and that is checked bit for bit:

jpeg   (KIEMS photos, most scans) Work on the JPEG's quantised DCT
       coefficients, not on decoded pixels. A masked luma block gets the DC
       value that decodes to black and all-zero AC; a chroma block whose whole
       area is masked is set to neutral grey. Every other block keeps its
       coefficients and the quantisation tables are copied, so a verifier can
       show that apart from the listed black blocks the redacted image is
       *the same data* as the original, independently of any JPEG decoder.
pixels (PNG, TIFF, ...) The same on decoded pixels, written losslessly (PNG):
       listed blocks are 0, every other pixel is identical.

A black block can hide a digit but cannot turn one digit into another. With
every changed block listed and checked to be pure black, no result can have
been altered, only obscured, and the manifest records where.
"""

from __future__ import annotations

import base64
import hashlib
import os
import tempfile
from dataclasses import dataclass

import cv2
import numpy as np

BLOCK = 8


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def image_digest(kind: str, data: bytes) -> str:
    """Hash identifying a page image independently of how a PDF stores it.

    jpeg    the JPEG bytes (embedded in the PDF unchanged, extracted unchanged)
    pixels  the decoded pixels (h, w, channels, then BGR bytes): a PDF stores
            a PNG as compressed raw pixels and re-encodes it on extraction,
            so its bytes aren't stable, but its pixels are."""
    if kind == "jpeg":
        return sha256(data)
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if img is None:
        raise ValueError("cannot decode image")
    return sha256(np.int64(img.shape).tobytes() + np.ascontiguousarray(img).tobytes())


# --- block masks ---------------------------------------------------------------

def to_blocks(mask: np.ndarray, block: int = BLOCK) -> np.ndarray:
    """Pixel mask (nonzero = hide) -> boolean grid of blocks touching it."""
    h, w = mask.shape[:2]
    bh, bw = -(-h // block), -(-w // block)
    pad = np.zeros((bh * block, bw * block), bool)
    pad[:h, :w] = mask > 0
    return pad.reshape(bh, block, bw, block).any(axis=(1, 3))


def to_pixels(blocks: np.ndarray, shape: tuple[int, int], block: int = BLOCK) -> np.ndarray:
    """Block grid -> boolean pixel mask of the given (h, w)."""
    m = np.repeat(np.repeat(blocks, block, axis=0), block, axis=1)
    return m[:shape[0], :shape[1]]


def pack(blocks: np.ndarray) -> dict:
    return {"rows": int(blocks.shape[0]), "cols": int(blocks.shape[1]),
            "bits": base64.b64encode(np.packbits(blocks.ravel())).decode()}


def unpack(d: dict) -> np.ndarray:
    n = d["rows"] * d["cols"]
    bits = np.unpackbits(np.frombuffer(base64.b64decode(d["bits"]), np.uint8))[:n]
    return bits.reshape(d["rows"], d["cols"]).astype(bool)


# --- JPEG at coefficient level -----------------------------------------------------

def _with_file(data: bytes, fn):
    fd, path = tempfile.mkstemp(suffix=".jpg")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        return fn(path)
    finally:
        os.unlink(path)


def read_dct(data: bytes):
    import jpeglib

    def load(path):
        im = jpeglib.read_dct(path)
        im.load()
        return im
    return _with_file(data, load)


def black_dc(q00: int) -> int:
    """Quantised DC that decodes a block to 0 (level shift 128 -> DC = 8*(0-128))."""
    return int(np.floor(-1024 / q00))


def _chroma_blocks(luma: np.ndarray, im) -> np.ndarray:
    """Chroma blocks whose entire luma footprint is masked."""
    ch, cw = im.Cb.shape[:2]
    # subsampling from the block grids themselves (4:2:0 -> 2x2, 4:4:4 -> 1x1)
    sy, sx = max(1, round(im.Y.shape[0] / ch)), max(1, round(im.Y.shape[1] / cw))
    pad = np.zeros((ch * sy, cw * sx), bool)
    pad[:min(luma.shape[0], ch * sy), :min(luma.shape[1], cw * sx)] = \
        luma[:ch * sy, :cw * sx]
    return pad.reshape(ch, sy, cw, sx).all(axis=(1, 3))


@dataclass
class Redacted:
    data: bytes  # new JPEG / PNG
    kind: str  # "jpeg" | "pixels"
    luma_blocks: np.ndarray  # blocks set to black
    chroma_blocks: np.ndarray | None  # chroma blocks set to neutral (jpeg only)


def redact_jpeg(data: bytes, blocks: np.ndarray) -> Redacted:
    """Black out ``blocks`` (luma grid) in a JPEG without touching any other
    coefficient."""
    import jpeglib  # noqa: F401  (import error surfaces here, clearly)

    im = read_dct(data)
    if im.progressive_mode:
        raise ValueError("progressive JPEG: not supported for coefficient-level redaction")
    if blocks.shape != im.Y.shape[:2]:  # image dims rounded up to MCUs can add a row/col
        b = np.zeros(im.Y.shape[:2], bool)
        h, w = min(b.shape[0], blocks.shape[0]), min(b.shape[1], blocks.shape[1])
        b[:h, :w] = blocks[:h, :w]
        blocks = b
    # Keep only markers that affect how colours decode (JFIF, ICC profile,
    # Adobe); drop EXIF/XMP/comments, which can carry camera serials or GPS.
    # Markers aren't image data, so the coefficient proof is unaffected.
    im.markers = [m for m in (im.markers or []) if m.type.name in ("JPEG_APP0", "JPEG_APP2", "JPEG_APP14")]
    Y = im.Y.copy()
    Y[blocks] = 0
    Y[blocks, 0, 0] = black_dc(int(im.qt[im.quant_tbl_no[0]][0, 0]))
    im.Y = Y
    cb = None
    if im.has_chrominance:
        cb = _chroma_blocks(blocks, im)
        Cb, Cr = im.Cb.copy(), im.Cr.copy()
        Cb[cb] = 0
        Cr[cb] = 0  # DC 0 = neutral
        im.Cb, im.Cr = Cb, Cr
    fd, path = tempfile.mkstemp(suffix=".jpg")
    os.close(fd)
    try:
        im.write_dct(path)
        with open(path, "rb") as f:
            out = f.read()
    finally:
        os.unlink(path)
    return Redacted(out, "jpeg", blocks, cb)


def redact_pixels(img: np.ndarray, blocks: np.ndarray) -> Redacted:
    out = img.copy()
    out[to_pixels(blocks, img.shape[:2])] = 0
    ok, buf = cv2.imencode(".png", out, [cv2.IMWRITE_PNG_COMPRESSION, 6])
    if not ok:
        raise ValueError("png encode failed")
    return Redacted(buf.tobytes(), "pixels", blocks, None)


# --- verification --------------------------------------------------------------------

@dataclass
class Comparison:
    ok: bool
    problems: list[str]
    total_blocks: int
    changed_blocks: np.ndarray  # luma grid: blocks that differ
    identical_blocks: int


def compare_jpeg(original: bytes, redacted: bytes, declared: np.ndarray | None = None) -> Comparison:
    """Coefficient-level comparison. Every block must be identical, or black
    (and declared, if a declaration is given)."""
    a, b = read_dct(original), read_dct(redacted)
    problems = []
    if (a.width, a.height) != (b.width, b.height):
        return Comparison(False, ["image size differs"], 0, np.zeros((0, 0), bool), 0)
    if not np.array_equal(a.qt, b.qt) or list(a.quant_tbl_no) != list(b.quant_tbl_no):
        problems.append("quantisation tables differ")
    if [list(x) for x in a.samp_factor] != [list(x) for x in b.samp_factor]:
        problems.append("chroma sampling differs")
    diff = (a.Y != b.Y).any(axis=(2, 3))
    dc = black_dc(int(b.qt[b.quant_tbl_no[0]][0, 0]))
    is_black = (b.Y[..., 0, 0] == dc) & (b.Y.reshape(*b.Y.shape[:2], 64)[..., 1:] == 0).all(axis=2)
    if (diff & ~is_black).any():
        problems.append(f"{int((diff & ~is_black).sum())} luma blocks changed to something other than black")
    d = np.zeros_like(diff)
    if declared is not None:
        h, w = min(d.shape[0], declared.shape[0]), min(d.shape[1], declared.shape[1])
        d[:h, :w] = declared[:h, :w]
        if (diff & ~d).any():
            problems.append(f"{int((diff & ~d).sum())} changed luma blocks not in the declared mask")
    if a.has_chrominance:
        cdiff = (a.Cb != b.Cb).any(axis=(2, 3)) | (a.Cr != b.Cr).any(axis=(2, 3))
        neutral = (b.Cb == 0).all(axis=(2, 3)) & (b.Cr == 0).all(axis=(2, 3))
        allowed = _chroma_blocks((diff | d) & is_black, b)
        if (cdiff & ~(neutral & allowed)).any():
            problems.append(f"{int((cdiff & ~(neutral & allowed)).sum())} chroma blocks changed outside black areas")
    return Comparison(not problems, problems, int(diff.size), diff, int((~diff).sum()))


def compare_pixels(original: np.ndarray, redacted: np.ndarray, declared: np.ndarray | None = None) -> Comparison:
    if original.shape != redacted.shape:
        return Comparison(False, ["image size differs"], 0, np.zeros((0, 0), bool), 0)
    px = (original != redacted).reshape(*original.shape[:2], -1).any(axis=2)
    diff = to_blocks(px)
    problems = []
    black = (redacted.reshape(*redacted.shape[:2], -1) == 0).all(axis=2)
    if (px & ~black).any():
        problems.append(f"{int((px & ~black).sum())} pixels changed to something other than black")
    if declared is not None:
        allowed = to_pixels(declared, original.shape[:2])
        if (px & ~allowed).any():
            problems.append(f"{int((px & ~allowed).sum())} changed pixels outside the declared mask")
    return Comparison(not problems, problems, int(diff.size), diff, int((~diff).sum()))
