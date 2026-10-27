"""Reading scans and writing sanitized output.

Output is always rebuilt from pixels: every page is rasterized, redacted and
re-encoded into a brand-new PDF. Nothing from the source file (text layers, OCR
layers, annotations, metadata, EXIF, embedded originals) can survive.
"""

from __future__ import annotations

import zipfile
from dataclasses import dataclass
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


# --- source images for provable redaction ---------------------------------------

@dataclass
class SourcePage:
    """One page image exactly as stored in the source, so that the proof
    compares like with like."""
    pixels: np.ndarray  # BGR, stored orientation (EXIF rotation NOT applied)
    kind: str  # "jpeg": data is the original JPEG bytes; "pixels": lossless compare
    data: bytes | None
    ref: str  # where it came from, e.g. "page 1, image xref 6" / "file" / "page 1, rendered"
    page_size: tuple[float, float] | None = None  # PDF points, to rebuild the same layout
    rect: tuple[float, float, float, float] | None = None  # where the image sits on the page


_DECODE = cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION


def _decode_bytes(data: bytes) -> np.ndarray:
    img = cv2.imdecode(np.frombuffer(data, np.uint8), _DECODE)
    if img is None:
        raise ValueError("cannot decode image")
    return img


def source_pages(src: str, data: bytes | None = None) -> list[SourcePage]:
    """Pages of a source file as stored. A PDF page that is a single full-page
    image (KIEMS transmissions, scanner output) yields that embedded image
    itself; JPEGs stay JPEG so the redaction can be proven at coefficient
    level. Pages that aren't a single image are rendered (weaker proof: the
    verifier re-renders the original with the same renderer)."""
    data = read_bytes(src) if data is None else data
    suffix = Path(src.split(ZIP_SEP)[-1]).suffix.lower()
    if suffix == ".pdf":
        out = []
        with pymupdf.open(stream=data, filetype="pdf") as doc:
            for i, page in enumerate(doc):
                size = (page.rect.width, page.rect.height)
                imgs = page.get_images(full=True)
                if len(imgs) == 1 and not imgs[0][1]:  # one image, no soft mask
                    xref = imgs[0][0]
                    rects = page.get_image_rects(xref)
                    info = doc.extract_image(xref)
                    if rects and info and abs(rects[0].get_area() - page.rect.get_area()) < 0.02 * page.rect.get_area():
                        r = rects[0]
                        kind = "jpeg" if info["ext"] in ("jpeg", "jpg") else "pixels"
                        out.append(SourcePage(_decode_bytes(info["image"]), kind,
                                              info["image"] if kind == "jpeg" else None,
                                              f"page {i + 1}, image xref {xref}", size, (r.x0, r.y0, r.x1, r.y1)))
                        continue
                z = _page_zoom(page, None)
                pix = page.get_pixmap(matrix=pymupdf.Matrix(z, z), colorspace=pymupdf.csRGB, alpha=False, annots=True)
                rgb = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, 3)
                out.append(SourcePage(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), "pixels", None,
                                      f"page {i + 1}, rendered at zoom {z:.4f} by PyMuPDF {pymupdf.VersionBind}",
                                      size, (0, 0, size[0], size[1])))
        return out
    if data[:2] == b"\xff\xd8":
        return [SourcePage(_decode_bytes(data), "jpeg", data, "file")]
    if suffix in {".tif", ".tiff"}:
        ok, pages = cv2.imdecodemulti(np.frombuffer(data, np.uint8), _DECODE)
        if not ok:
            raise ValueError("cannot decode tiff")
        return [SourcePage(p, "pixels", None, f"tiff page {i + 1}") for i, p in enumerate(pages)]
    return [SourcePage(_decode_bytes(data), "pixels", None, "file")]


def write_pdf_images(images: list[tuple[bytes, SourcePage]], out: str | Path) -> bytes:
    """PDF with each page holding one image *as given* (JPEG passed through
    byte-for-byte, PNG stored losslessly), laid out like the source page.
    Empty metadata. Returns the PDF bytes."""
    doc = pymupdf.open()
    for data, sp in images:
        h, w = sp.pixels.shape[:2]
        pw, ph = sp.page_size or (w * 72 / 150, h * 72 / 150)
        page = doc.new_page(width=pw, height=ph)
        page.insert_image(pymupdf.Rect(*(sp.rect or (0, 0, pw, ph))), stream=data)
    doc.set_metadata({})
    doc.del_xml_metadata()
    pdf = doc.tobytes(garbage=4, deflate=True, no_new_id=True)
    doc.close()
    Path(out).write_bytes(pdf)
    return pdf


def pdf_images(data: bytes) -> list[bytes]:
    """The (single) image stored on each page of a redacted PDF."""
    out = []
    with pymupdf.open(stream=data, filetype="pdf") as doc:
        for page in doc:
            imgs = page.get_images(full=True)
            if len(imgs) != 1:
                raise ValueError(f"page {page.number + 1}: expected one image, found {len(imgs)}")
            out.append(doc.extract_image(imgs[0][0])["image"])
    return out
