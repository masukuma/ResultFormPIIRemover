//! Edits that can be proven to change nothing but PII.
//!
//! The only edit ever made to a form image is replacing whole 8x8 pixel
//! blocks with solid black, checked bit for bit:
//!
//! jpeg   work on the JPEG's quantised DCT coefficients. A masked luma block
//!        gets the DC value that decodes to black and all-zero AC; a chroma
//!        block whose whole area is masked is set to neutral. Every other
//!        block keeps its coefficients and the quantisation tables are
//!        copied, so a verifier can show that apart from the listed black
//!        blocks the redacted image is the same data as the original,
//!        independently of any JPEG decoder.
//! pixels the same on decoded pixels, written losslessly (PNG).

use anyhow::{anyhow, bail, Result};
use base64::Engine;
use opencv::core::{Mat, Vector};
use opencv::imgcodecs;
use opencv::prelude::*;
use sha2::{Digest, Sha256};
use std::os::raw::{c_char, c_int, c_uchar, c_ulong};

use crate::util::Grid;

pub const BLOCK: usize = 8;

pub fn sha256(data: &[u8]) -> String {
    hex::encode(Sha256::digest(data))
}

/// Hash identifying a page image independently of how a PDF stores it:
/// JPEG by its bytes; anything else by its decoded pixels (h, w, channels as
/// int64, then the BGR bytes), as the Python version computes it.
pub fn image_digest(kind: &str, data: &[u8]) -> Result<String> {
    if kind == "jpeg" {
        return Ok(sha256(data));
    }
    let img = crate::io::decode_bytes(data)?;
    Ok(pixel_digest(&img)?)
}

pub fn pixel_digest(img: &Mat) -> Result<String> {
    let mut h = Sha256::new();
    let shape = [img.rows() as i64, img.cols() as i64, img.channels() as i64];
    for v in shape {
        h.update(v.to_le_bytes());
    }
    let c = if img.is_continuous() { img.try_clone()? } else { img.try_clone()? };
    h.update(c.data_bytes()?);
    Ok(hex::encode(h.finalize()))
}

// --- block masks -------------------------------------------------------------------

/// Pixel mask (nonzero = hide) -> grid of the blocks touching it.
pub fn to_blocks(mask: &Mat) -> Result<Grid> {
    let (h, w) = (mask.rows() as usize, mask.cols() as usize);
    let (bh, bw) = (h.div_ceil(BLOCK), w.div_ceil(BLOCK));
    let mut g = Grid::new(bh, bw);
    let data = crate::util::bytes(mask)?;
    for y in 0..h {
        let row = &data[y * w..(y + 1) * w];
        for (x, &v) in row.iter().enumerate() {
            if v != 0 {
                g.set(y / BLOCK, x / BLOCK, true);
            }
        }
    }
    Ok(g)
}

/// Block grid -> pixel mask (255 = in a listed block) of the given size.
pub fn to_pixels(blocks: &Grid, rows: i32, cols: i32) -> Result<Mat> {
    let mut m = crate::util::zeros_u8(rows, cols)?;
    {
        let d = m.data_bytes_mut()?;
        for y in 0..rows as usize {
            for x in 0..cols as usize {
                let (r, c) = (y / BLOCK, x / BLOCK);
                if r < blocks.rows && c < blocks.cols && blocks.get(r, c) {
                    d[y * cols as usize + x] = 255;
                }
            }
        }
    }
    Ok(m)
}

/// `np.packbits` of the grid, base64: {"rows", "cols", "bits"}.
pub fn pack(blocks: &Grid) -> serde_json::Value {
    let mut bytes = vec![0u8; blocks.data.len().div_ceil(8)];
    for (i, &b) in blocks.data.iter().enumerate() {
        if b {
            bytes[i / 8] |= 0x80 >> (i % 8);
        }
    }
    serde_json::json!({
        "rows": blocks.rows, "cols": blocks.cols,
        "bits": base64::engine::general_purpose::STANDARD.encode(bytes),
    })
}

pub fn unpack(v: &serde_json::Value) -> Result<Grid> {
    let rows = v["rows"].as_u64().ok_or_else(|| anyhow!("blocks: rows"))? as usize;
    let cols = v["cols"].as_u64().ok_or_else(|| anyhow!("blocks: cols"))? as usize;
    let bytes = base64::engine::general_purpose::STANDARD.decode(v["bits"].as_str().unwrap_or(""))?;
    let mut g = Grid::new(rows, cols);
    for i in 0..rows * cols {
        if i / 8 < bytes.len() && bytes[i / 8] & (0x80 >> (i % 8)) != 0 {
            g.data[i] = true;
        }
    }
    Ok(g)
}

// --- JPEG at coefficient level --------------------------------------------------------

#[repr(C)]
struct JDct {
    width: c_int,
    height: c_int,
    num_components: c_int,
    progressive: c_int,
    h_samp: [c_int; 4],
    v_samp: [c_int; 4],
    bw: [c_int; 4],
    bh: [c_int; 4],
    qtno: [c_int; 4],
    qt: [[u16; 64]; 4],
    coef: [*mut i16; 4],
}

unsafe extern "C" {
    fn jdct_read(data: *const c_uchar, len: c_ulong, out: *mut JDct, err: *mut c_char, errlen: c_int) -> c_int;
    fn jdct_free(d: *mut JDct);
    fn jdct_redact(
        data: *const c_uchar,
        len: c_ulong,
        luma: *const c_uchar,
        luma_rows: c_int,
        luma_cols: c_int,
        chroma: *const c_uchar,
        chroma_rows: c_int,
        chroma_cols: c_int,
        out: *mut *mut c_uchar,
        outlen: *mut c_ulong,
        err: *mut c_char,
        errlen: c_int,
    ) -> c_int;
    fn jdct_free_buffer(p: *mut c_uchar);
}

fn err_text(buf: &[c_char]) -> String {
    let bytes: Vec<u8> = buf.iter().take_while(|&&c| c != 0).map(|&c| c as u8).collect();
    String::from_utf8_lossy(&bytes).into_owned()
}

/// One component's coefficients: rows x cols blocks of 64 (natural order).
pub struct Component {
    pub rows: usize,
    pub cols: usize,
    pub h_samp: i32,
    pub v_samp: i32,
    pub qtno: usize,
    pub coef: Vec<i16>,
}

impl Component {
    #[inline]
    pub fn block(&self, r: usize, c: usize) -> &[i16] {
        let i = (r * self.cols + c) * 64;
        &self.coef[i..i + 64]
    }
}

pub struct Dct {
    pub width: usize,
    pub height: usize,
    pub progressive: bool,
    pub qt: [[u16; 64]; 4],
    pub comps: Vec<Component>,
}

impl Dct {
    pub fn y(&self) -> &Component {
        &self.comps[0]
    }
    pub fn has_chroma(&self) -> bool {
        self.comps.len() >= 3
    }
    pub fn dc_quant(&self) -> i32 {
        self.qt[self.comps[0].qtno][0] as i32
    }
}

pub fn read_dct(data: &[u8]) -> Result<Dct> {
    let mut d: JDct = unsafe { std::mem::zeroed() };
    let mut err = [0 as c_char; 256];
    let ok = unsafe { jdct_read(data.as_ptr(), data.len() as c_ulong, &mut d, err.as_mut_ptr(), err.len() as c_int) };
    if ok == 0 {
        bail!("cannot read JPEG coefficients: {}", err_text(&err));
    }
    let mut comps = Vec::new();
    for c in 0..d.num_components as usize {
        let n = d.bw[c] as usize * d.bh[c] as usize * 64;
        let coef = unsafe { std::slice::from_raw_parts(d.coef[c], n) }.to_vec();
        comps.push(Component {
            rows: d.bh[c] as usize,
            cols: d.bw[c] as usize,
            h_samp: d.h_samp[c],
            v_samp: d.v_samp[c],
            qtno: d.qtno[c] as usize,
            coef,
        });
    }
    let out = Dct { width: d.width as usize, height: d.height as usize, progressive: d.progressive != 0, qt: d.qt, comps };
    unsafe { jdct_free(&mut d) };
    Ok(out)
}

/// floor(-1024 / q00): the quantised DC that decodes a block to 0.
pub fn black_dc(q00: i32) -> i32 {
    (-1024.0 / q00 as f64).floor() as i32
}

/// Chroma blocks whose entire luma footprint is masked.
pub fn chroma_blocks(luma: &Grid, y: &Component, cb: &Component) -> Grid {
    let (ch, cw) = (cb.rows, cb.cols);
    // subsampling from the block grids themselves (4:2:0 -> 2x2, 4:4:4 -> 1x1)
    let sy = ((y.rows as f64 / ch as f64).round_ties_even() as usize).max(1);
    let sx = ((y.cols as f64 / cw as f64).round_ties_even() as usize).max(1);
    let mut g = Grid::new(ch, cw);
    for r in 0..ch {
        for c in 0..cw {
            let mut all = true;
            'outer: for dy in 0..sy {
                for dx in 0..sx {
                    let (lr, lc) = (r * sy + dy, c * sx + dx);
                    if lr >= luma.rows || lc >= luma.cols || !luma.get(lr, lc) {
                        all = false;
                        break 'outer;
                    }
                }
            }
            g.set(r, c, all);
        }
    }
    g
}

pub struct Redacted {
    pub data: Vec<u8>,
    pub kind: &'static str,
    pub luma_blocks: Grid,
    pub chroma_blocks: Option<Grid>,
}

/// Black out `blocks` (luma grid) in a JPEG without touching any other coefficient.
pub fn redact_jpeg(data: &[u8], blocks: &Grid) -> Result<Redacted> {
    let im = read_dct(data)?;
    if im.progressive {
        bail!("progressive JPEG: not supported for coefficient-level redaction");
    }
    let y = im.y();
    // image dimensions rounded up to MCUs can add a row or column
    let blocks = blocks.resized(y.rows, y.cols);
    let luma: Vec<u8> = blocks.data.iter().map(|&b| b as u8).collect();
    let cb = if im.has_chroma() { Some(chroma_blocks(&blocks, y, &im.comps[1])) } else { None };
    let chroma: Vec<u8> = cb.as_ref().map(|g| g.data.iter().map(|&b| b as u8).collect()).unwrap_or_default();
    let (cr, cc) = cb.as_ref().map(|g| (g.rows, g.cols)).unwrap_or((0, 0));
    let mut out: *mut c_uchar = std::ptr::null_mut();
    let mut outlen: c_ulong = 0;
    let mut err = [0 as c_char; 256];
    let ok = unsafe {
        jdct_redact(
            data.as_ptr(),
            data.len() as c_ulong,
            luma.as_ptr(),
            blocks.rows as c_int,
            blocks.cols as c_int,
            if cb.is_some() { chroma.as_ptr() } else { std::ptr::null() },
            cr as c_int,
            cc as c_int,
            &mut out,
            &mut outlen,
            err.as_mut_ptr(),
            err.len() as c_int,
        )
    };
    if ok == 0 {
        bail!("{}", err_text(&err));
    }
    let bytes = unsafe { std::slice::from_raw_parts(out, outlen as usize) }.to_vec();
    unsafe { jdct_free_buffer(out) };
    Ok(Redacted { data: bytes, kind: "jpeg", luma_blocks: blocks, chroma_blocks: cb })
}

pub fn redact_pixels(img: &Mat, blocks: &Grid) -> Result<Redacted> {
    let mut out = img.try_clone()?;
    let m = to_pixels(blocks, img.rows(), img.cols())?;
    out.set_to(&opencv::core::Scalar::all(0.0), &m)?;
    let mut buf = Vector::<u8>::new();
    let params: Vector<i32> = vec![imgcodecs::IMWRITE_PNG_COMPRESSION, 6].into_iter().collect();
    if !imgcodecs::imencode(".png", &out, &mut buf, &params)? {
        bail!("png encode failed");
    }
    Ok(Redacted { data: buf.to_vec(), kind: "pixels", luma_blocks: blocks.clone(), chroma_blocks: None })
}

// --- verification ------------------------------------------------------------------------

pub struct Comparison {
    pub ok: bool,
    pub problems: Vec<String>,
    pub total_blocks: usize,
    pub changed_blocks: Grid,
    pub identical_blocks: usize,
}

fn failed(why: &str) -> Comparison {
    Comparison { ok: false, problems: vec![why.into()], total_blocks: 0, changed_blocks: Grid::new(0, 0), identical_blocks: 0 }
}

/// Coefficient-level comparison: every block identical, or black (and
/// declared, if a declaration is given).
pub fn compare_jpeg(original: &[u8], redacted: &[u8], declared: Option<&Grid>) -> Result<Comparison> {
    let a = read_dct(original)?;
    let b = read_dct(redacted)?;
    if (a.width, a.height) != (b.width, b.height) || a.comps.len() != b.comps.len() {
        return Ok(failed("image size differs"));
    }
    let mut problems = Vec::new();
    let same_qt = a.comps.iter().zip(&b.comps).all(|(x, y)| x.qtno == y.qtno && a.qt[x.qtno] == b.qt[y.qtno]);
    if !same_qt {
        problems.push("quantisation tables differ".to_string());
    }
    if a.comps.iter().zip(&b.comps).any(|(x, y)| (x.v_samp, x.h_samp) != (y.v_samp, y.h_samp)) {
        problems.push("chroma sampling differs".to_string());
    }
    let (ya, yb) = (a.y(), b.y());
    let mut diff = Grid::new(ya.rows, ya.cols);
    let mut is_black = Grid::new(ya.rows, ya.cols);
    let dc = black_dc(b.dc_quant());
    for r in 0..ya.rows {
        for c in 0..ya.cols {
            let (p, q) = (ya.block(r, c), yb.block(r, c));
            diff.set(r, c, p != q);
            is_black.set(r, c, q[0] as i32 == dc && q[1..].iter().all(|&v| v == 0));
        }
    }
    let bad = diff.and_not(&is_black).count();
    if bad > 0 {
        problems.push(format!("{} luma blocks changed to something other than black", bad));
    }
    let mut d = Grid::new(diff.rows, diff.cols);
    if let Some(decl) = declared {
        d = decl.resized(diff.rows, diff.cols);
        let n = diff.and_not(&d).count();
        if n > 0 {
            problems.push(format!("{} changed luma blocks not in the declared mask", n));
        }
    }
    if a.has_chroma() {
        let (cba, cbb, cra, crb) = (&a.comps[1], &b.comps[1], &a.comps[2], &b.comps[2]);
        let mut allowed_src = Grid::new(diff.rows, diff.cols);
        for i in 0..diff.data.len() {
            allowed_src.data[i] = (diff.data[i] || d.data[i]) && is_black.data[i];
        }
        let allowed = chroma_blocks(&allowed_src, yb, cbb);
        let mut n = 0;
        for r in 0..cba.rows {
            for c in 0..cba.cols {
                let changed = cba.block(r, c) != cbb.block(r, c) || cra.block(r, c) != crb.block(r, c);
                let neutral = cbb.block(r, c).iter().all(|&v| v == 0) && crb.block(r, c).iter().all(|&v| v == 0);
                if changed && !(neutral && allowed.get(r, c)) {
                    n += 1;
                }
            }
        }
        if n > 0 {
            problems.push(format!("{} chroma blocks changed outside black areas", n));
        }
    }
    let identical = diff.data.iter().filter(|&&x| !x).count();
    Ok(Comparison { ok: problems.is_empty(), problems, total_blocks: diff.data.len(), changed_blocks: diff, identical_blocks: identical })
}

pub fn compare_pixels(original: &Mat, redacted: &Mat, declared: Option<&Grid>) -> Result<Comparison> {
    if original.rows() != redacted.rows() || original.cols() != redacted.cols() || original.channels() != redacted.channels() {
        return Ok(failed("image size differs"));
    }
    let (h, w, ch) = (original.rows() as usize, original.cols() as usize, original.channels() as usize);
    let a = crate::util::bytes(original)?;
    let b = crate::util::bytes(redacted)?;
    let mut px = crate::util::zeros_u8(h as i32, w as i32)?;
    let (mut not_black, mut outside) = (0usize, 0usize);
    let allowed = match declared {
        Some(g) => Some(crate::util::bytes(&to_pixels(g, h as i32, w as i32)?)?),
        None => None,
    };
    {
        let pd = px.data_bytes_mut()?;
        for i in 0..h * w {
            let changed = (0..ch).any(|k| a[i * ch + k] != b[i * ch + k]);
            if changed {
                pd[i] = 255;
                if (0..ch).any(|k| b[i * ch + k] != 0) {
                    not_black += 1;
                }
                if let Some(al) = &allowed {
                    if al[i] == 0 {
                        outside += 1;
                    }
                }
            }
        }
    }
    let diff = to_blocks(&px)?;
    let mut problems = Vec::new();
    if not_black > 0 {
        problems.push(format!("{} pixels changed to something other than black", not_black));
    }
    if outside > 0 {
        problems.push(format!("{} changed pixels outside the declared mask", outside));
    }
    let identical = diff.data.iter().filter(|&&x| !x).count();
    Ok(Comparison { ok: problems.is_empty(), problems, total_blocks: diff.data.len(), changed_blocks: diff, identical_blocks: identical })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pack_roundtrip() {
        let mut g = Grid::new(3, 5);
        g.set(0, 0, true);
        g.set(2, 4, true);
        let p = pack(&g);
        // python: base64(np.packbits([1,0,0,0,0, 0,0,0,0,0, 0,0,0,0,1])) == "gAI="
        assert_eq!(p["bits"], "gAI=");
        assert_eq!(unpack(&p).unwrap(), g);
        assert_eq!(black_dc(16), -64);
        assert_eq!(black_dc(3), -342);
    }
}
