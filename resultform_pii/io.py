"""Reading scans and writing sanitized output.

Output is always rebuilt from pixels: every page is rasterized, redacted and
re-encoded into a brand-new PDF. Nothing from the source file (text layers, OCR
layers, annotations, metadata, EXIF, embedded originals) can survive.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import cv2
import numpy as np
import pymupdf

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
INPUT_EXTS = IMAGE_EXTS | {".pdf"}


def iter_pages(path: str | Path, dpi: int = 200) -> Iterator[np.ndarray]:
    """Yield BGR pages from a PDF or image file."""
    path = Path(path)
    if path.suffix.lower() == ".pdf":
        with pymupdf.open(path) as doc:
            for page in doc:
                pix = page.get_pixmap(dpi=dpi, colorspace=pymupdf.csRGB, alpha=False, annots=True)
                rgb = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, 3)
                yield cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        return
    if path.suffix.lower() in {".tif", ".tiff"}:
        ok, pages = cv2.imreadmulti(str(path), flags=cv2.IMREAD_COLOR)
        if not ok:
            raise ValueError(f"cannot read {path}")
        yield from pages
        return
    # IMREAD_COLOR honours EXIF orientation, so phone photos come in upright.
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"cannot read {path}")
    yield img


def encode(img: np.ndarray, fmt: str = "jpg", quality: int = 85) -> bytes:
    params = [cv2.IMWRITE_JPEG_QUALITY, quality] if fmt == "jpg" else [cv2.IMWRITE_PNG_COMPRESSION, 6]
    ok, buf = cv2.imencode("." + fmt, img, params)
    if not ok:
        raise ValueError("encode failed")
    return buf.tobytes()


def write_pdf(pages: list[np.ndarray], out: str | Path, dpi: int = 200, quality: int = 85) -> None:
    """Write an image-only PDF with empty metadata."""
    doc = pymupdf.open()
    for img in pages:
        h, w = img.shape[:2]
        page = doc.new_page(width=w * 72 / dpi, height=h * 72 / dpi)
        page.insert_image(page.rect, stream=encode(img, "jpg", quality))
    doc.set_metadata({})
    doc.del_xml_metadata()
    doc.save(str(out), garbage=4, deflate=True, no_new_id=True)
    doc.close()


def write_image(img: np.ndarray, out: str | Path, quality: int = 85) -> None:
    out = Path(out)
    fmt = "png" if out.suffix.lower() == ".png" else "jpg"
    out.write_bytes(encode(img, fmt, quality))  # cv2 encoding writes no EXIF
