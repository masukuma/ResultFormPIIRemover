"""Reading scans and writing sanitized output.

Output is always rebuilt from pixels: every page is rasterized, redacted and
re-encoded into a brand-new PDF. Nothing from the source file (text layers, OCR
layers, annotations, metadata, EXIF, embedded originals) can survive.
"""

from __future__ import annotations

import zipfile
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np
import pymupdf

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
INPUT_EXTS = IMAGE_EXTS | {".pdf"}


ZIP_SEP = "::"  # "archive.zip::member.pdf" addresses a file inside a zip


def _page_zoom(page, dpi: int | None) -> float:
    """Render scanned pages at the resolution of their embedded image (KIEMS
    PDFs declare a 1654 pt page around a 1654 px photo; 200 dpi would blow it up
    2.8x for nothing). Vector-only pages fall back to ``dpi`` or 200."""
    if dpi:
        return dpi / 72
    widths = [im[2] for im in page.get_images(full=True)]
    if not widths:
        return 200 / 72
    return float(np.clip(max(widths) / page.rect.width, 72 / 72, 400 / 72))


def _decode(data: bytes, suffix: str, dpi: int | None) -> Iterator[np.ndarray]:
    if suffix == ".pdf":
        with pymupdf.open(stream=data, filetype="pdf") as doc:
            for page in doc:
                z = _page_zoom(page, dpi)
                pix = page.get_pixmap(matrix=pymupdf.Matrix(z, z), colorspace=pymupdf.csRGB, alpha=False, annots=True)
                rgb = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, 3)
                yield cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        return
    buf = np.frombuffer(data, np.uint8)
    if suffix in {".tif", ".tiff"}:
        ok, pages = cv2.imdecodemulti(buf, cv2.IMREAD_COLOR)
        if not ok:
            raise ValueError("cannot decode tiff")
        yield from pages
        return
    # IMREAD_COLOR honours EXIF orientation, so phone photos come in upright.
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("cannot decode image")
    yield img


def read_bytes(src: str) -> bytes:
    if ZIP_SEP in src:
        archive, member = src.split(ZIP_SEP, 1)
        with zipfile.ZipFile(archive) as z:
            return z.read(member)
    return Path(src).read_bytes()


def iter_pages(src: str | Path, dpi: int | None = None, data: bytes | None = None) -> Iterator[np.ndarray]:
    """Yield BGR pages from a PDF/image path, or an ``archive.zip::member`` path."""
    src = str(src)
    data = read_bytes(src) if data is None else data
    yield from _decode(data, Path(src.split(ZIP_SEP)[-1]).suffix.lower(), dpi)


def list_inputs(paths) -> list[str]:
    """Expand files, directories and zip archives into readable sources."""
    out: list[str] = []
    for p in map(Path, paths):
        files = sorted(f for f in p.rglob("*") if f.is_file()) if p.is_dir() else [p]
        for f in files:
            ext = f.suffix.lower()
            if ext == ".zip":
                with zipfile.ZipFile(f) as z:
                    out += [f"{f}{ZIP_SEP}{n}" for n in sorted(z.namelist())
                            if Path(n).suffix.lower() in INPUT_EXTS and not n.startswith("__MACOSX")]
            elif ext in INPUT_EXTS:
                out.append(str(f))
    return out


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
