//! Reading scans and writing sanitized output.
//!
//! Output is always rebuilt: every page image is redacted and placed in a
//! brand-new PDF. Nothing from the source file (text layers, annotations,
//! metadata, EXIF, embedded originals) can survive.

use anyhow::{anyhow, bail, Context, Result};
use lopdf::{content::Content, Dictionary, Document, Object, ObjectId};
use opencv::core::{Mat, Vector};
use opencv::imgcodecs;
use opencv::prelude::*;
use std::io::{Read, Write};
use std::path::Path;

pub const ZIP_SEP: &str = "::"; // "archive.zip::member.pdf" addresses a file inside a zip
const IMAGE_EXTS: &[&str] = &[".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"];

fn suffix(src: &str) -> String {
    let name = src.rsplit(ZIP_SEP).next().unwrap_or(src);
    Path::new(name).extension().map(|e| format!(".{}", e.to_string_lossy().to_lowercase())).unwrap_or_default()
}

pub fn read_bytes(src: &str) -> Result<Vec<u8>> {
    if let Some((archive, member)) = src.split_once(ZIP_SEP) {
        let mut z = zip::ZipArchive::new(std::fs::File::open(archive)?)?;
        let mut f = z.by_name(member)?;
        let mut out = Vec::new();
        f.read_to_end(&mut out)?;
        return Ok(out);
    }
    std::fs::read(src).with_context(|| format!("reading {}", src))
}

/// Expand files, directories and zip archives into readable sources.
pub fn list_inputs(paths: &[String]) -> Result<Vec<String>> {
    let mut out = Vec::new();
    let input = |e: &str| e == ".pdf" || IMAGE_EXTS.contains(&e);
    for p in paths {
        let p = Path::new(p);
        let mut files: Vec<std::path::PathBuf> = if p.is_dir() {
            walk(p)?
        } else {
            vec![p.to_path_buf()]
        };
        files.sort();
        for f in files {
            let ext = f.extension().map(|e| format!(".{}", e.to_string_lossy().to_lowercase())).unwrap_or_default();
            if ext == ".zip" {
                let z = zip::ZipArchive::new(std::fs::File::open(&f)?)?;
                let mut names: Vec<String> = z.file_names().filter_map(|s| s.ok().map(|s| s.to_string())).collect();
                names.sort();
                for n in names {
                    if input(&suffix(&n)) && !n.starts_with("__MACOSX") {
                        out.push(format!("{}{}{}", f.display(), ZIP_SEP, n));
                    }
                }
            } else if input(&ext) {
                out.push(f.display().to_string());
            }
        }
    }
    Ok(out)
}

fn walk(dir: &Path) -> Result<Vec<std::path::PathBuf>> {
    let mut out = Vec::new();
    for e in std::fs::read_dir(dir)? {
        let p = e?.path();
        if p.is_dir() {
            out.extend(walk(&p)?);
        } else if p.is_file() {
            out.push(p);
        }
    }
    Ok(out)
}

/// Decode an image as stored (EXIF orientation not applied), 3 channels.
pub fn decode_bytes(data: &[u8]) -> Result<Mat> {
    let buf = Vector::<u8>::from_slice(data);
    let img = imgcodecs::imdecode(&buf, imgcodecs::IMREAD_COLOR | imgcodecs::IMREAD_IGNORE_ORIENTATION)?;
    if img.empty() {
        bail!("cannot decode image");
    }
    Ok(img)
}

/// Decode as a viewer would (EXIF orientation applied): phone photos upright.
pub fn decode_upright(data: &[u8]) -> Result<Mat> {
    let buf = Vector::<u8>::from_slice(data);
    let img = imgcodecs::imdecode(&buf, imgcodecs::IMREAD_COLOR)?;
    if img.empty() {
        bail!("cannot decode image");
    }
    Ok(img)
}

pub fn encode(img: &Mat, fmt: &str, quality: i32) -> Result<Vec<u8>> {
    let params: Vector<i32> = if fmt == "jpg" {
        vec![imgcodecs::IMWRITE_JPEG_QUALITY, quality]
    } else {
        vec![imgcodecs::IMWRITE_PNG_COMPRESSION, 6]
    }
    .into_iter()
    .collect();
    let mut buf = Vector::<u8>::new();
    if !imgcodecs::imencode(&format!(".{}", fmt), img, &mut buf, &params)? {
        bail!("encode failed");
    }
    Ok(buf.to_vec())
}

// --- source pages ----------------------------------------------------------------------

/// One page image exactly as stored in the source, so that the proof
/// compares like with like.
pub struct SourcePage {
    pub pixels: Mat,          // BGR, stored orientation (EXIF rotation not applied)
    pub kind: &'static str,   // "jpeg": data is the original JPEG; "pixels": lossless compare
    pub data: Option<Vec<u8>>,
    pub reference: String,    // where it came from, e.g. "page 1, image xref 6"
    pub page_size: Option<(f64, f64)>,          // PDF points, to rebuild the same layout
    pub rect: Option<(f64, f64, f64, f64)>,     // where the image sits on the page (PDF coordinates)
    pub rotate: i64,
}

pub fn source_pages(src: &str, data: Option<&[u8]>) -> Result<Vec<SourcePage>> {
    let owned;
    let data = match data {
        Some(d) => d,
        None => {
            owned = read_bytes(src)?;
            &owned
        }
    };
    let sfx = suffix(src);
    if sfx == ".pdf" {
        return Ok(pdf_source_pages(data, None)?.0);
    }
    if data.starts_with(&[0xff, 0xd8]) {
        return Ok(vec![SourcePage {
            pixels: decode_bytes(data)?,
            kind: "jpeg",
            data: Some(data.to_vec()),
            reference: "file".into(),
            page_size: None,
            rect: None,
            rotate: 0,
        }]);
    }
    if sfx == ".tif" || sfx == ".tiff" {
        // OpenCV 4.6 reads multi-page images only from a file
        let mut pages = Vector::<Mat>::new();
        let tmp = std::env::temp_dir().join(format!("rfpii-{}-{}.tif", std::process::id(), crate::proof::sha256(data)));
        std::fs::write(&tmp, data)?;
        let ok = imgcodecs::imreadmulti(&tmp, &mut pages, imgcodecs::IMREAD_COLOR | imgcodecs::IMREAD_IGNORE_ORIENTATION);
        let _ = std::fs::remove_file(&tmp);
        if !ok? {
            bail!("cannot decode tiff");
        }
        return Ok(pages
            .iter()
            .enumerate()
            .map(|(i, p)| SourcePage {
                pixels: p,
                kind: "pixels",
                data: None,
                reference: format!("tiff page {}", i + 1),
                page_size: None,
                rect: None,
                rotate: 0,
            })
            .collect());
    }
    Ok(vec![SourcePage {
        pixels: decode_bytes(data)?,
        kind: "pixels",
        data: None,
        reference: "file".into(),
        page_size: None,
        rect: None,
        rotate: 0,
    }])
}

fn as_f64(o: &Object) -> Option<f64> {
    match o {
        Object::Integer(i) => Some(*i as f64),
        Object::Real(r) => Some(*r as f64),
        _ => None,
    }
}

fn page_box(doc: &Document, page: ObjectId) -> Result<(f64, f64, f64, f64)> {
    // CropBox or MediaBox, inherited from the page tree
    let mut id = Some(page);
    while let Some(pid) = id {
        let d = doc.get_dictionary(pid)?;
        for key in [b"CropBox".as_slice(), b"MediaBox".as_slice()] {
            if let Ok(o) = d.get(key) {
                let (_, o) = doc.dereference(o)?;
                if let Object::Array(a) = o {
                    let v: Vec<f64> = a.iter().filter_map(|x| doc.dereference(x).ok().and_then(|(_, x)| as_f64(x))).collect();
                    if v.len() == 4 {
                        return Ok((v[0].min(v[2]), v[1].min(v[3]), v[0].max(v[2]), v[1].max(v[3])));
                    }
                }
            }
        }
        id = d.get(b"Parent").ok().and_then(|p| p.as_reference().ok());
    }
    Ok((0.0, 0.0, 612.0, 792.0))
}

fn page_rotate(doc: &Document, page: ObjectId) -> i64 {
    let mut id = Some(page);
    while let Some(pid) = id {
        let Ok(d) = doc.get_dictionary(pid) else { break };
        if let Ok(Object::Integer(r)) = d.get(b"Rotate") {
            return *r;
        }
        id = d.get(b"Parent").ok().and_then(|p| p.as_reference().ok());
    }
    0
}

/// Image XObjects reachable from a resources dictionary: (name path, stream id).
fn image_xobjects(doc: &Document, res: &Dictionary, out: &mut Vec<(Vec<u8>, ObjectId)>, depth: usize) {
    let Ok(xo) = res.get(b"XObject") else { return };
    let Ok((_, xo)) = doc.dereference(xo) else { return };
    let Object::Dictionary(xo) = xo else { return };
    for (name, v) in xo.iter() {
        let Ok(id) = v.as_reference() else { continue };
        let Ok(Object::Stream(s)) = doc.get_object(id) else { continue };
        match s.dict.get(b"Subtype").and_then(|o| o.as_name()) {
            Ok(b"Image") => out.push((name.clone(), id)),
            Ok(b"Form") if depth < 4 => {
                if let Ok(r) = s.dict.get(b"Resources") {
                    if let Ok((_, Object::Dictionary(r))) = doc.dereference(r) {
                        image_xobjects(doc, r, out, depth + 1);
                    }
                }
            }
            _ => {}
        }
    }
}

type M6 = [f64; 6];

fn mat_mul(a: &M6, b: &M6) -> M6 {
    // a then b (PDF row-vector convention: [x y 1] * a * b)
    [
        a[0] * b[0] + a[1] * b[2],
        a[0] * b[1] + a[1] * b[3],
        a[2] * b[0] + a[3] * b[2],
        a[2] * b[1] + a[3] * b[3],
        a[4] * b[0] + a[5] * b[2] + b[4],
        a[4] * b[1] + a[5] * b[3] + b[5],
    ]
}

/// Where the page's content stream draws XObject `name`: bounding box of the
/// unit square under the CTM at the `Do`, in PDF coordinates.
fn image_rect(doc: &Document, page: ObjectId, name: &[u8]) -> Option<(f64, f64, f64, f64)> {
    let content = Content::decode(&doc.get_page_content(page)).ok()?;
    let mut stack: Vec<M6> = Vec::new();
    let mut ctm: M6 = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0];
    for op in content.operations {
        match op.operator.as_str() {
            "q" => stack.push(ctm),
            "Q" => ctm = stack.pop().unwrap_or([1.0, 0.0, 0.0, 1.0, 0.0, 0.0]),
            "cm" => {
                let v: Vec<f64> = op.operands.iter().filter_map(as_f64).collect();
                if v.len() == 6 {
                    ctm = mat_mul(&[v[0], v[1], v[2], v[3], v[4], v[5]], &ctm);
                }
            }
            "Do" => {
                if op.operands.first().and_then(|o| o.as_name().ok()) == Some(name) {
                    let pts = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)]
                        .map(|(x, y)| (x * ctm[0] + y * ctm[2] + ctm[4], x * ctm[1] + y * ctm[3] + ctm[5]));
                    let xs = pts.map(|p| p.0);
                    let ys = pts.map(|p| p.1);
                    return Some((
                        xs.iter().cloned().fold(f64::MAX, f64::min),
                        ys.iter().cloned().fold(f64::MAX, f64::min),
                        xs.iter().cloned().fold(f64::MIN, f64::max),
                        ys.iter().cloned().fold(f64::MIN, f64::max),
                    ));
                }
            }
            _ => {}
        }
    }
    None
}

fn filters(dict: &Dictionary) -> Vec<Vec<u8>> {
    match dict.get(b"Filter") {
        Ok(Object::Name(n)) => vec![n.clone()],
        Ok(Object::Array(a)) => a.iter().filter_map(|o| o.as_name().ok().map(|n| n.to_vec())).collect(),
        _ => vec![],
    }
}

fn inflate(data: &[u8]) -> Result<Vec<u8>> {
    let mut out = Vec::new();
    flate2::read::ZlibDecoder::new(data).read_to_end(&mut out)?;
    Ok(out)
}

/// The JPEG bytes of an image stream, if it is one (DCTDecode, possibly
/// under FlateDecode).
fn stream_jpeg(s: &lopdf::Stream) -> Result<Option<Vec<u8>>> {
    let f = filters(&s.dict);
    match f.iter().map(|x| x.as_slice()).collect::<Vec<_>>().as_slice() {
        [b"DCTDecode"] => Ok(Some(s.content.clone())),
        [b"FlateDecode", b"DCTDecode"] => Ok(Some(inflate(&s.content)?)),
        _ => Ok(None),
    }
}

/// Raw 8-bit RGB / gray pixels of a FlateDecode (or unfiltered) image stream,
/// with PNG predictors undone, as BGR.
fn stream_pixels(doc: &Document, s: &lopdf::Stream) -> Result<Option<Mat>> {
    let f = filters(&s.dict);
    let raw = match f.iter().map(|x| x.as_slice()).collect::<Vec<_>>().as_slice() {
        [] => s.content.clone(),
        [b"FlateDecode"] => inflate(&s.content)?,
        _ => return Ok(None),
    };
    let geti = |k: &[u8]| s.dict.get(k).ok().and_then(|o| doc.dereference(o).ok()).and_then(|(_, o)| o.as_i64().ok());
    let (w, h) = (geti(b"Width").unwrap_or(0) as usize, geti(b"Height").unwrap_or(0) as usize);
    if geti(b"BitsPerComponent").unwrap_or(8) != 8 || w == 0 || h == 0 {
        return Ok(None);
    }
    let cs = s.dict.get(b"ColorSpace").ok().and_then(|o| doc.dereference(o).ok()).map(|(_, o)| o.clone());
    let n = match cs {
        Some(Object::Name(ref c)) if c == b"DeviceRGB" => 3,
        Some(Object::Name(ref c)) if c == b"DeviceGray" => 1,
        _ => return Ok(None),
    };
    let mut predictor = 1;
    let mut columns = w as i64;
    if let Ok(dp) = s.dict.get(b"DecodeParms") {
        if let Ok((_, Object::Dictionary(dp))) = doc.dereference(dp) {
            predictor = dp.get(b"Predictor").ok().and_then(|o| o.as_i64().ok()).unwrap_or(1);
            columns = dp.get(b"Columns").ok().and_then(|o| o.as_i64().ok()).unwrap_or(w as i64);
        }
    }
    let row = columns as usize * n;
    let pix: Vec<u8> = if predictor >= 10 {
        let mut out = vec![0u8; row * h];
        let mut prev = vec![0u8; row];
        for y in 0..h {
            let src = raw.get(y * (row + 1)..(y + 1) * (row + 1)).ok_or_else(|| anyhow!("short image stream"))?;
            let (t, line) = (src[0], &src[1..]);
            let cur = &mut out[y * row..(y + 1) * row];
            for i in 0..row {
                let a = if i >= n { cur[i - n] as i32 } else { 0 };
                let b = prev[i] as i32;
                let c = if i >= n { prev[i - n] as i32 } else { 0 };
                let pred = match t {
                    0 => 0,
                    1 => a,
                    2 => b,
                    3 => (a + b) / 2,
                    4 => {
                        let p = a + b - c;
                        let (pa, pb, pc) = ((p - a).abs(), (p - b).abs(), (p - c).abs());
                        if pa <= pb && pa <= pc {
                            a
                        } else if pb <= pc {
                            b
                        } else {
                            c
                        }
                    }
                    _ => bail!("bad PNG predictor"),
                };
                cur[i] = (line[i] as i32 + pred) as u8;
            }
            prev.copy_from_slice(cur);
        }
        out
    } else {
        raw
    };
    if pix.len() < w * h * n {
        bail!("short image stream");
    }
    let mut bgr = crate::util::zeros_u8(h as i32, w as i32)?;
    let mut out = Mat::default();
    if n == 1 {
        bgr.data_bytes_mut()?.copy_from_slice(&pix[..w * h]);
        opencv::imgproc::cvt_color_def(&bgr, &mut out, opencv::imgproc::COLOR_GRAY2BGR)?;
    } else {
        let mut rgb = Mat::new_rows_cols_with_default(h as i32, w as i32, opencv::core::CV_8UC3, opencv::core::Scalar::all(0.0))?;
        rgb.data_bytes_mut()?.copy_from_slice(&pix[..w * h * 3]);
        opencv::imgproc::cvt_color_def(&rgb, &mut out, opencv::imgproc::COLOR_RGB2BGR)?;
    }
    Ok(Some(out))
}

unsafe extern "C" {
    fn mrender_page(data: *const u8, len: usize, pageno: i32, zoom: f32, out: *mut *mut u8, w: *mut i32, h: *mut i32, err: *mut std::os::raw::c_char, errlen: i32) -> i32;
    fn mrender_free(p: *mut u8);
    fn mrender_version() -> *const std::os::raw::c_char;
    fn mimage_extract(data: *const u8, len: usize, pageno: i32, xref: i32, out: *mut *mut u8, outlen: *mut usize, ext: *mut std::os::raw::c_char, extlen: i32, err: *mut std::os::raw::c_char, errlen: i32) -> i32;
    fn mrepair(data: *const u8, len: usize, out: *mut *mut u8, outlen: *mut usize, pages: *mut i32, err: *mut std::os::raw::c_char, errlen: i32) -> i32;
}

fn c_text(b: &[std::os::raw::c_char]) -> String {
    unsafe { std::ffi::CStr::from_ptr(b.as_ptr()) }.to_string_lossy().into_owned()
}

/// Image object `xref` of page `page` (0-based) as PyMuPDF's
/// `Document.extract_image` gives it after reading the page: (bytes,
/// extension). JPEGs come as stored; fax, Flate and other raw encodings
/// decoded and written as PNG.
fn extract_image(data: &[u8], page: usize, xref: u32) -> Result<(Vec<u8>, String)> {
    let (mut out, mut n) = (std::ptr::null_mut(), 0usize);
    let mut ext = vec![0 as std::os::raw::c_char; 32];
    let mut err = vec![0 as std::os::raw::c_char; 512];
    let ok = unsafe { mimage_extract(data.as_ptr(), data.len(), page as i32, xref as i32, &mut out, &mut n, ext.as_mut_ptr(), ext.len() as i32, err.as_mut_ptr(), err.len() as i32) };
    if ok == 0 {
        bail!("image xref {}: {}", xref, c_text(&err));
    }
    let bytes = unsafe { std::slice::from_raw_parts(out, n) }.to_vec();
    unsafe { mrender_free(out) };
    Ok((bytes, c_text(&ext)))
}

/// A PDF lopdf can't read, as MuPDF repairs it: (pages, repaired PDF).
fn repair(data: &[u8]) -> Result<(usize, Vec<u8>)> {
    let (mut out, mut n, mut pages) = (std::ptr::null_mut(), 0usize, 0i32);
    let mut err = vec![0 as std::os::raw::c_char; 512];
    let ok = unsafe { mrepair(data.as_ptr(), data.len(), &mut out, &mut n, &mut pages, err.as_mut_ptr(), err.len() as i32) };
    if ok == 0 {
        bail!("{}", c_text(&err));
    }
    let bytes = if out.is_null() { vec![] } else { unsafe { std::slice::from_raw_parts(out, n) }.to_vec() };
    unsafe { mrender_free(out) };
    Ok((pages.max(0) as usize, bytes))
}

/// MuPDF's version (the renderer of pages that aren't a single image).
pub fn mupdf_version() -> String {
    unsafe { std::ffi::CStr::from_ptr(mrender_version()) }.to_string_lossy().into_owned()
}

/// Render a page as PyMuPDF's `page.get_pixmap(matrix=Matrix(zoom, zoom))`, BGR.
fn render_page(data: &[u8], index: i32, zoom: f64) -> Result<Mat> {
    let (mut out, mut w, mut h) = (std::ptr::null_mut(), 0i32, 0i32);
    let mut err = vec![0 as std::os::raw::c_char; 512];
    let ok = unsafe { mrender_page(data.as_ptr(), data.len(), index, zoom as f32, &mut out, &mut w, &mut h, err.as_mut_ptr(), err.len() as i32) };
    if ok == 0 {
        let msg = unsafe { std::ffi::CStr::from_ptr(err.as_ptr()) }.to_string_lossy().into_owned();
        bail!("rendering page {}: {}", index + 1, msg);
    }
    let px = unsafe { std::slice::from_raw_parts(out, w as usize * h as usize * 3) }.to_vec();
    unsafe { mrender_free(out) };
    let mut rgb = Mat::new_rows_cols_with_default(h, w, opencv::core::CV_8UC3, opencv::core::Scalar::all(0.0))?;
    rgb.data_bytes_mut()?.copy_from_slice(&px);
    let mut bgr = Mat::default();
    opencv::imgproc::cvt_color_def(&rgb, &mut bgr, opencv::imgproc::COLOR_RGB2BGR)?;
    Ok(bgr)
}

/// Page `k` (from 1) of a source by itself, and how many pages the source
/// has: only that page is decoded (a scanned batch may hold hundreds).
pub fn source_page(src: &str, data: &[u8], k: usize) -> Result<(SourcePage, usize)> {
    let (mut pages, n) = if suffix(src) == ".pdf" {
        pdf_source_pages(data, Some(k))?
    } else {
        let all = source_pages(src, Some(data))?;
        let n = all.len();
        (all.into_iter().skip(k.max(1) - 1).take(1).collect(), n)
    };
    match (k >= 1, pages.pop()) {
        (true, Some(p)) => Ok((p, n)),
        _ => bail!("page {} of {}: no such page", k, n),
    }
}

/// How many pages a source has, none of them decoded for a PDF.
pub fn page_count(src: &str, data: &[u8]) -> Result<usize> {
    if suffix(src) == ".pdf" {
        return Ok(load_pdf(data)?.map(|(d, _)| d.get_pages().len()).unwrap_or(0));
    }
    Ok(source_pages(src, Some(data))?.len())
}

/// The PDF, and its bytes as MuPDF repaired them where lopdf can't read it
/// as it is (as with PyMuPDF: a truncated download may have no pages left:
/// None).
fn load_pdf(data: &[u8]) -> Result<Option<(Document, Option<Vec<u8>>)>> {
    match Document::load_mem(data) {
        Ok(d) => Ok(Some((d, None))),
        Err(e) => {
            let (pages, f) = repair(data).map_err(|m| anyhow!("reading PDF: {e}; MuPDF: {m}"))?;
            if pages == 0 {
                return Ok(None);
            }
            Ok(Some((Document::load_mem(&f).context("reading repaired PDF")?, Some(f))))
        }
    }
}

/// A PDF's pages (only page `only`, from 1, if given), and how many it has.
fn pdf_source_pages(data: &[u8], only: Option<usize>) -> Result<(Vec<SourcePage>, usize)> {
    let Some((doc, fixed)) = load_pdf(data)? else { return Ok((vec![], 0)) };
    let data = fixed.as_deref().unwrap_or(data);
    let all = doc.get_pages();
    let n = all.len();
    let mut out = Vec::new();
    for (i, (_, page)) in all.into_iter().enumerate() {
        if only.is_some_and(|k| k != i + 1) {
            continue;
        }
        let (x0, y0, x1, y1) = page_box(&doc, page)?;
        let size = (x1 - x0, y1 - y0);
        let area = size.0 * size.1;
        let rotate = page_rotate(&doc, page);
        let mut imgs = Vec::new();
        if let Ok((res, ids)) = doc.get_page_resources(page) {
            if let Some(r) = res {
                image_xobjects(&doc, r, &mut imgs, 0);
            }
            for id in ids {
                if let Ok(r) = doc.get_dictionary(id) {
                    image_xobjects(&doc, r, &mut imgs, 0);
                }
            }
        }
        imgs.sort_by_key(|(_, id)| *id);
        imgs.dedup_by_key(|(_, id)| *id);
        let mut widths = Vec::new();
        for (_, id) in &imgs {
            if let Ok(Object::Stream(s)) = doc.get_object(*id) {
                widths.push(s.dict.get(b"Width").ok().and_then(|o| o.as_i64().ok()).unwrap_or(0) as f64);
            }
        }
        if imgs.len() == 1 {
            let (name, id) = &imgs[0];
            if let Ok(Object::Stream(s)) = doc.get_object(*id) {
                let masked = s.dict.get(b"SMask").is_ok();
                if let (false, Some(r)) = (masked, image_rect(&doc, page, name)) {
                    let rarea = (r.2 - r.0) * (r.3 - r.1);
                    if (rarea - area).abs() < 0.02 * area {
                        // the image as stored: JPEGs stay JPEG, so the redaction
                        // can be proven coefficient by coefficient
                        let (bytes, ext) = extract_image(data, i, id.0)?;
                        let jpeg = ext == "jpeg" || ext == "jpg";
                        out.push(SourcePage {
                            pixels: decode_bytes(&bytes)?,
                            kind: if jpeg { "jpeg" } else { "pixels" },
                            data: if jpeg { Some(bytes) } else { None },
                            reference: format!("page {}, image xref {}", i + 1, id.0),
                            page_size: Some(size),
                            rect: Some((r.0 - x0, r.1 - y0, r.2 - x0, r.3 - y0)),
                            rotate,
                        });
                        continue;
                    }
                }
            }
        }
        // Not a single full-page image: render it, at the resolution of its
        // largest image across the page as displayed (rotation applied).
        let shown = if rotate.rem_euclid(180) == 90 { (size.1, size.0) } else { size };
        let zoom = if widths.is_empty() {
            200.0 / 72.0
        } else {
            (widths.iter().cloned().fold(0.0, f64::max) / shown.0).clamp(1.0, 400.0 / 72.0)
        };
        out.push(SourcePage {
            pixels: render_page(data, i as i32, zoom)?,
            kind: "pixels",
            data: None,
            reference: format!("page {}, rendered at zoom {:.4} by MuPDF {}", i + 1, zoom, mupdf_version()),
            page_size: Some(shown),
            rect: Some((0.0, 0.0, shown.0, shown.1)),
            rotate: 0, // the rendering is upright already
        });
    }
    Ok((out, n))
}

// --- writing --------------------------------------------------------------------------------

struct PdfWriter {
    buf: Vec<u8>,
    offsets: Vec<usize>,
}

impl PdfWriter {
    fn new() -> Self {
        let mut buf = Vec::new();
        buf.extend_from_slice(b"%PDF-1.7\n%\xe2\xe3\xcf\xd3\n");
        PdfWriter { buf, offsets: vec![0] }
    }
    fn obj(&mut self, id: usize, body: &[u8]) {
        if self.offsets.len() <= id {
            self.offsets.resize(id + 1, 0);
        }
        self.offsets[id] = self.buf.len();
        self.buf.extend_from_slice(format!("{} 0 obj\n", id).as_bytes());
        self.buf.extend_from_slice(body);
        self.buf.extend_from_slice(b"\nendobj\n");
    }
    fn stream(&mut self, id: usize, dict: &str, data: &[u8]) {
        let mut body = format!("<<{} /Length {}>>\nstream\n", dict, data.len()).into_bytes();
        body.extend_from_slice(data);
        body.extend_from_slice(b"\nendstream");
        self.obj(id, &body);
    }
    fn finish(mut self, root: usize) -> Vec<u8> {
        let xref = self.buf.len();
        let n = self.offsets.len();
        self.buf.extend_from_slice(format!("xref\n0 {}\n0000000000 65535 f \n", n).as_bytes());
        for off in &self.offsets[1..] {
            self.buf.extend_from_slice(format!("{:010} 00000 n \n", off).as_bytes());
        }
        self.buf
            .extend_from_slice(format!("trailer\n<</Size {} /Root {} 0 R>>\nstartxref\n{}\n%%EOF\n", n, root, xref).as_bytes());
        self.buf
    }
}

fn fmt_num(x: f64) -> String {
    let s = format!("{:.4}", x);
    let s = s.trim_end_matches('0').trim_end_matches('.');
    if s.is_empty() || s == "-0" {
        "0".into()
    } else {
        s.to_string()
    }
}

/// An image to place on a page: JPEG bytes (passed through byte for byte), or
/// pixels (stored losslessly).
pub enum PageImage<'a> {
    Jpeg(&'a [u8]),
    Pixels(&'a Mat),
}

/// PDF with one image per page, laid out like the source page; no metadata.
pub fn build_pdf(pages: &[(PageImage, Option<(f64, f64)>, Option<(f64, f64, f64, f64)>, i64)]) -> Result<Vec<u8>> {
    let mut w = PdfWriter::new();
    let n = pages.len();
    // ids: 1 catalog, 2 pages, then 3 per page (page, content, image)
    let kids: Vec<String> = (0..n).map(|i| format!("{} 0 R", 3 + 3 * i)).collect();
    w.obj(1, b"<</Type /Catalog /Pages 2 0 R>>");
    w.obj(2, format!("<</Type /Pages /Kids [{}] /Count {}>>", kids.join(" "), n).as_bytes());
    for (i, (img, size, rect, rotate)) in pages.iter().enumerate() {
        let (pid, cid, iid) = (3 + 3 * i, 4 + 3 * i, 5 + 3 * i);
        let (iw, ih, dict, data): (i32, i32, String, Vec<u8>) = match img {
            PageImage::Jpeg(d) => {
                let m = decode_bytes(d)?;
                let gray = crate::proof::read_dct(d).map(|x| x.comps.len() == 1).unwrap_or(false);
                let cs = if gray { "/DeviceGray" } else { "/DeviceRGB" };
                (
                    m.cols(),
                    m.rows(),
                    format!("/Type /XObject /Subtype /Image /Width {} /Height {} /ColorSpace {} /BitsPerComponent 8 /Filter /DCTDecode", m.cols(), m.rows(), cs),
                    d.to_vec(),
                )
            }
            PageImage::Pixels(m) => {
                let mut rgb = Mat::default();
                opencv::imgproc::cvt_color_def(*m, &mut rgb, opencv::imgproc::COLOR_BGR2RGB)?;
                let raw = crate::util::bytes(&rgb)?;
                let mut z = flate2::write::ZlibEncoder::new(Vec::new(), flate2::Compression::new(6));
                z.write_all(&raw)?;
                (
                    m.cols(),
                    m.rows(),
                    format!("/Type /XObject /Subtype /Image /Width {} /Height {} /ColorSpace /DeviceRGB /BitsPerComponent 8 /Filter /FlateDecode", m.cols(), m.rows()),
                    z.finish()?,
                )
            }
        };
        let (pw, ph) = size.unwrap_or((iw as f64 * 72.0 / 150.0, ih as f64 * 72.0 / 150.0));
        let (x0, y0, x1, y1) = rect.unwrap_or((0.0, 0.0, pw, ph));
        let rot = if *rotate != 0 { format!(" /Rotate {}", rotate) } else { String::new() };
        w.obj(
            pid,
            format!(
                "<</Type /Page /Parent 2 0 R /MediaBox [0 0 {} {}]{} /Resources <</XObject <</Im0 {} 0 R>>>> /Contents {} 0 R>>",
                fmt_num(pw),
                fmt_num(ph),
                rot,
                iid,
                cid
            )
            .as_bytes(),
        );
        let content = format!("q {} 0 0 {} {} {} cm /Im0 Do Q", fmt_num(x1 - x0), fmt_num(y1 - y0), fmt_num(x0), fmt_num(y0));
        w.stream(cid, "", content.as_bytes());
        w.stream(iid, &format!(" {}", dict), &data);
    }
    Ok(w.finish(1))
}

/// Image-only PDF from pixels (JPEG at `quality`), e.g. for proposals in review.
pub fn write_pdf(pages: &[Mat], out: &Path, dpi: f64, quality: i32) -> Result<()> {
    let jpegs: Vec<Vec<u8>> = pages.iter().map(|p| encode(p, "jpg", quality)).collect::<Result<_>>()?;
    let items: Vec<_> = pages
        .iter()
        .zip(&jpegs)
        .map(|(p, j)| {
            let size = (p.cols() as f64 * 72.0 / dpi, p.rows() as f64 * 72.0 / dpi);
            (PageImage::Jpeg(j.as_slice()), Some(size), None, 0)
        })
        .collect();
    std::fs::write(out, build_pdf(&items)?)?;
    Ok(())
}

/// The (single) image stored on each page of a redacted PDF: JPEG bytes as
/// stored, or for a lossless page its pixels re-encoded as PNG (its digest
/// is taken over the pixels).
pub fn pdf_images(data: &[u8]) -> Result<Vec<(Vec<u8>, &'static str)>> {
    let doc = Document::load_mem(data).context("reading PDF")?;
    let mut out = Vec::new();
    for (n, (_, page)) in doc.get_pages().into_iter().enumerate() {
        let mut imgs = Vec::new();
        if let Ok((res, ids)) = doc.get_page_resources(page) {
            if let Some(r) = res {
                image_xobjects(&doc, r, &mut imgs, 0);
            }
            for id in ids {
                if let Ok(r) = doc.get_dictionary(id) {
                    image_xobjects(&doc, r, &mut imgs, 0);
                }
            }
        }
        imgs.sort_by_key(|(_, id)| *id);
        imgs.dedup_by_key(|(_, id)| *id);
        if imgs.len() != 1 {
            bail!("page {}: expected one image, found {}", n + 1, imgs.len());
        }
        let Ok(Object::Stream(s)) = doc.get_object(imgs[0].1) else { bail!("page {}: bad image", n + 1) };
        if let Some(j) = stream_jpeg(s)? {
            out.push((j, "jpeg"));
        } else if let Some(px) = stream_pixels(&doc, s)? {
            out.push((encode(&px, "png", 0)?, "pixels"));
        } else {
            bail!("page {}: unsupported image encoding", n + 1);
        }
    }
    Ok(out)
}
