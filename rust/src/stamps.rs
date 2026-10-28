//! Find official stamps, so that no mask ever touches one.
//!
//! The round "I.E.B.C. PRESIDING OFFICER" stamp is part of what shows a form is
//! genuine, and it often sits on top of a signature, a date or an agent's row.
//! Every stamp found here becomes a protected region of the page: masks are
//! clipped around it and the block grid is checked to leave every block that
//! touches a stamp unchanged.
//!
//! Detection favours recall over precision: a false stamp only means a little
//! less is masked, and the leak checks then send the page to review if that
//! leaves PII readable. A missed stamp could be masked, so there are several
//! cues:
//!
//! - rings: circles (Hough) whose perimeter is inked nearly all the way round
//!   and more densely than the paper just outside;
//! - stamp text: OCR lines reading like "I.E.B.C." / "PRESIDING OFFICER" in
//!   capitals trigger a more permissive ring search around them, and protect a
//!   generous disc there even if no ring is found;
//! - near masks: before a page is released, a permissive ring search runs
//!   around every mask (`near`). There a ring smaller than any stamp must run
//!   all the way round, and a broken one must run in long arcs: a handwritten
//!   "0", or pen strokes crossing a circle, are not stamps.
//!
//! The stamps are recorded in the manifest, so `rfpii verify` can show them.

use anyhow::Result;
use once_cell::sync::Lazy;
use opencv::core::{self, Mat, Vector};
use opencv::imgproc;
use opencv::prelude::*;
use regex::Regex;
use serde_json::{json, Value};

use crate::masking::Img;
use crate::pyjson;
use crate::util::{self, round_even};

pub const WORK: f64 = 1600.0;
static STAMP_TEXT: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"(?i)[IL1lA|]\s*[.,]\s*E\s*[.,]?\s*B\s*[.,]?\s*C|PRESID[I1l]|OFF[I1l]C[EF]").unwrap());

/// A short line in capitals reading like stamp text ("I.E.B.C.", "PRESIDING",
/// "OFFICER"); not the printed label "Presiding Officer".
pub fn is_stamp_text(text: &str) -> bool {
    let t = text.trim();
    let letters: Vec<char> = t.chars().filter(|c| c.is_alphabetic()).collect();
    STAMP_TEXT.is_match(t)
        && t.chars().count() <= 14
        && letters.len() >= 3
        && letters.iter().filter(|c| c.is_uppercase()).count() as f64 >= 0.8 * letters.len() as f64
}

#[derive(Clone, Debug)]
pub struct Stamp {
    pub cx: f64, // image px
    pub cy: f64,
    pub r: f64,
    pub coverage: f64, // share of the ring's perimeter that is inked
    pub how: String,   // "ring" | "text" | "near-mask"
}

/// `np.linspace(0, stop, n, endpoint=False)`.
fn linspace_open(stop: f64, n: usize) -> Vec<f64> {
    let step = stop / n as f64;
    (0..n).map(|k| k as f64 * step).collect()
}

/// `np.arange(start, stop, step)` for floats, filled the way numpy does.
fn arange(start: f64, stop: f64, step: f64) -> Vec<f64> {
    let len = ((stop - start) / step).ceil();
    if !(len > 0.0) {
        return Vec::new();
    }
    let len = len as usize;
    let mut v = vec![start];
    if len == 1 {
        return v;
    }
    let next = start + step;
    v.push(next);
    let delta = next - start;
    for i in 2..len {
        v.push(start + i as f64 * delta);
    }
    v
}

impl Stamp {
    pub fn new(cx: f64, cy: f64, r: f64, coverage: f64, how: &str) -> Self {
        Stamp { cx, cy, r, coverage, how: how.into() }
    }

    pub fn polygon(&self, margin: f64) -> Vec<[f64; 2]> {
        let rr = self.r + margin;
        linspace_open(2.0 * std::f64::consts::PI, 64).iter().map(|&t| [self.cx + rr * t.cos(), self.cy + rr * t.sin()]).collect()
    }

    pub fn record(&self) -> Value {
        let poly: Vec<Value> = self.polygon(0.0).iter().map(|p| json!([pyjson::np_round(p[0], 1), pyjson::np_round(p[1], 1)])).collect();
        json!({"name": "stamp", "kind": "stamp",
               "circle": [pyjson::round_n(self.cx, 1), pyjson::round_n(self.cy, 1), pyjson::round_n(self.r, 1)],
               "found_by": self.how, "ring_coverage": pyjson::round_n(self.coverage, 2), "polygon": poly})
    }
}

/// Dark marks: print, pen and (dark) stamp ink.
fn dark_ink(gray: &Mat) -> Result<Mat> {
    let ink = util::adaptive_inv(&util::gaussian(gray, 3)?, 25, 12.0)?;
    util::morph(&ink, imgproc::MORPH_OPEN, &util::ones_kernel(2, 2)?)
}

/// Coloured marks: stamp ink (violet, blue, red) differs in colour from the
/// paper around it. The green security print and most black print are left out.
fn color_ink(color: &Mat) -> Result<Mat> {
    let mut lab8 = Mat::default();
    imgproc::cvt_color_def(color, &mut lab8, imgproc::COLOR_BGR2Lab)?;
    let mut med = Mat::default();
    imgproc::median_blur(&lab8, &mut med, 31)?;
    let (a, b) = (util::bytes(&lab8)?, util::bytes(&med)?);
    let mut m = vec![0u8; a.len() / 3];
    for (i, o) in m.iter_mut().enumerate() {
        let dl = a[3 * i] as i32 - b[3 * i] as i32;
        let da = a[3 * i + 1] as i32 - b[3 * i + 1] as i32;
        let db = a[3 * i + 2] as i32 - b[3 * i + 2] as i32;
        if da * da + db * db > 81 && da > -2 && dl < 5 {
            *o = 255;
        }
    }
    let mm = util::from_bytes(lab8.rows(), lab8.cols(), &m)?;
    util::morph(&mm, imgproc::MORPH_OPEN, &util::ones_kernel(2, 2)?)
}

/// Least-squares circle through the ink near a candidate ring, trimming
/// outliers each round (Hough centres and radii are rough).
pub fn refine(ink: &Img, cx: f64, cy: f64, r: f64, iters: usize) -> (f64, f64, f64) {
    let (h, w) = (ink.h as f64, ink.w as f64);
    let x0 = (cx - 1.35 * r).max(0.0) as usize;
    let x1 = (cx + 1.35 * r + 1.0).min(w) as usize;
    let y0 = (cy - 1.35 * r).max(0.0) as usize;
    let y1 = (cy + 1.35 * r + 1.0).min(h) as usize;
    let mut pts: Vec<(f64, f64)> = Vec::new();
    for y in y0..y1.max(y0) {
        for x in x0..x1.max(x0) {
            if ink.at(x, y) > 0 {
                pts.push((x as f64, y as f64));
            }
        }
    }
    let (mut cx, mut cy, mut r) = (cx, cy, r);
    let mut tol = 0.2 * r;
    for _ in 0..iters {
        let sel: Vec<(f64, f64)> = pts.iter().cloned().filter(|&(x, y)| ((x - cx).hypot(y - cy) - r).abs() <= tol).collect();
        if sel.len() < 20 {
            break;
        }
        let a: Vec<[f64; 3]> = sel.iter().map(|&(x, y)| [x, y, 1.0]).collect();
        let yv: Vec<f64> = sel.iter().map(|&(x, y)| x * x + y * y).collect();
        let flat: Vec<f64> = a.iter().flat_map(|r| r.iter().cloned()).collect();
        let Ok(c) = crate::npla::lstsq(&flat, a.len(), 3, &yv, f64::EPSILON * a.len().max(3) as f64) else { break };
        let (pa, pb, pc) = (c[0], c[1], c[2]);
        let (ncx, ncy) = (pa / 2.0, pb / 2.0);
        let nr = (pc + ncx * ncx + ncy * ncy).max(1.0).sqrt();
        if !(0.7 * r < nr && nr < 1.4 * r) || (ncx - cx).hypot(ncy - cy) > 0.4 * r {
            break;
        }
        (cx, cy, r) = (ncx, ncy, nr);
        tol = (tol * 0.6).max(2.0);
    }
    (cx, cy, r)
}

/// Sample a ring: for each of `bins` directions, which radii are on the page
/// and which of those are inked.
fn sample(ink: &Img, cx: f64, cy: f64, radii: &[f64], cos_sin: &[(f64, f64)]) -> (Vec<Vec<bool>>, Vec<Vec<bool>>) {
    let (h, w) = (ink.h as i64, ink.w as i64);
    let mut ok = Vec::with_capacity(cos_sin.len());
    let mut hit = Vec::with_capacity(cos_sin.len());
    for &(c, s) in cos_sin {
        let (mut o, mut t) = (Vec::with_capacity(radii.len()), Vec::with_capacity(radii.len()));
        for &rad in radii {
            let x = round_even(cx + rad * c) as i64;
            let y = round_even(cy + rad * s) as i64;
            let on = x >= 0 && x < w && y >= 0 && y < h;
            o.push(on);
            t.push(on && ink.at(x.clamp(0, w - 1) as usize, y.clamp(0, h - 1) as usize) > 0);
        }
        ok.push(o);
        hit.push(t);
    }
    (ok, hit)
}

fn directions(bins: usize) -> Vec<(f64, f64)> {
    linspace_open(2.0 * std::f64::consts::PI, bins).iter().map(|t| (t.cos(), t.sin())).collect()
}

/// (continuity, density, density just outside) of a thin ring at radius r,
/// or None if too little of the ring is on the page.
pub fn ring(ink: &Img, cx: f64, cy: f64, r: f64, bins: usize) -> Option<(f64, f64, f64)> {
    let cs = directions(bins);
    let tol = (0.05 * r).max(2.0);
    let radii: Vec<f64> = arange(-tol, tol + 0.5, 1.0).iter().map(|d| r + d).collect();
    let (ok, hit) = sample(ink, cx, cy, &radii, &cs);
    let valid: Vec<bool> = ok.iter().map(|o| o.iter().all(|&v| v)).collect();
    let nvalid = valid.iter().filter(|&&v| v).count();
    if (nvalid as f64 / bins as f64) < 0.4 {
        return None;
    }
    let oradii: Vec<f64> = arange(0.1 * r, 0.2 * r, 1.0).iter().map(|d| r + d).collect();
    let (ook, ohit) = sample(ink, cx, cy, &oradii, &cs);
    let ov: Vec<bool> = ook.iter().map(|o| o.iter().all(|&v| v)).collect();
    let out = if ov.iter().any(|&v| v) {
        let (mut n, mut k) = (0usize, 0usize);
        for (i, row) in ohit.iter().enumerate() {
            if ov[i] {
                n += row.len();
                k += row.iter().filter(|&&v| v).count();
            }
        }
        k as f64 / n as f64
    } else {
        1.0
    };
    let (mut any, mut n, mut k) = (0usize, 0usize, 0usize);
    for (i, row) in hit.iter().enumerate() {
        if valid[i] {
            n += row.len();
            let c = row.iter().filter(|&&v| v).count();
            k += c;
            any += (c > 0) as usize;
        }
    }
    Some((any as f64 / nvalid as f64, k as f64 / n as f64, out))
}

pub const MIN_ARC: f64 = 10.0;

/// Mean length, in steps of 360/bins degrees, of the inked arcs of a ring.
pub fn mean_arc(ink: &Img, cx: f64, cy: f64, r: f64, bins: usize) -> f64 {
    let cs = directions(bins);
    let tol = (0.05 * r).max(2.0);
    let radii: Vec<f64> = arange(-tol, tol + 0.5, 1.0).iter().map(|d| r + d).collect();
    let (_, hit) = sample(ink, cx, cy, &radii, &cs);
    let hit: Vec<bool> = hit.iter().map(|row| row.iter().any(|&v| v)).collect();
    if hit.iter().all(|&v| v) {
        return bins as f64;
    }
    if !hit.iter().any(|&v| v) {
        return 0.0;
    }
    let start = hit.iter().position(|&v| !v).unwrap();
    let rolled: Vec<bool> = (0..bins).map(|i| hit[(i + start) % bins]).collect();
    let (mut runs, mut cur) = (Vec::new(), 0usize);
    for &v in rolled.iter().chain(std::iter::once(&false)) {
        if v {
            cur += 1;
        } else if cur > 0 {
            runs.push(cur);
            cur = 0;
        }
    }
    runs.iter().sum::<usize>() as f64 / runs.len() as f64
}

type Limits = [(f64, f64, f64); 2]; // per map: dark, color

pub const STRICT: Limits = [(0.8, 0.6, 0.4), (0.45, 0.25, 0.25)];
pub const LOOSE: Limits = [(0.7, 0.5, 0.45), (0.4, 0.22, 0.3)];
pub const FAINT: Limits = [(0.9, 0.4, 0.25), (0.9, 0.2, 0.25)];

/// `np.geomspace(lo, hi, 4)`.
fn geomspace4(lo: f64, hi: f64) -> [f64; 4] {
    let (a, b) = (lo.log10(), hi.log10());
    let step = (b - a) / 3.0;
    [lo, 10f64.powf(a + step), 10f64.powf(a + 2.0 * step), hi]
}

pub fn candidates(ink: &Img, r_range: (f64, f64), param2: f64) -> Result<Vec<[f64; 3]>> {
    let blur = util::gaussian(&ink.to_mat()?, 5)?;
    let e = geomspace4(r_range.0, r_range.1);
    let mut found = Vec::new();
    for i in 0..3 {
        let (a, b) = (e[i], e[i + 1]);
        let mut c = Vector::<core::Vec3f>::new();
        imgproc::hough_circles(&blur, &mut c, imgproc::HOUGH_GRADIENT, 1.5, (a as i32).max(10) as f64, 100.0, param2, a as i32, b as i32 + 1)?;
        for v in c.iter() {
            found.push([v[0] as f64, v[1] as f64, v[2] as f64]);
        }
    }
    Ok(found)
}

#[derive(Clone)]
pub struct Measured {
    pub kind: usize, // 0 dark, 1 color
    pub cx: f64,
    pub cy: f64,
    pub r: f64,
    pub m: (f64, f64, f64),
}

/// Every ring candidate, refined and measured.
fn measure(maps: &[Img], r_range: (f64, f64), param2: f64, offset: (f64, f64)) -> Result<Vec<Measured>> {
    let mut out = Vec::new();
    for (kind, ink) in maps.iter().enumerate() {
        let mut fits: Vec<(f64, f64, f64)> = Vec::new();
        for [cx, cy, r] in candidates(ink, r_range, param2)? {
            for k in [1.0, 0.85, 1.2] {
                match ring(ink, cx, cy, k * r, 180) {
                    Some(q) if q.0 >= 0.3 => {}
                    _ => continue,
                }
                let (fx, fy, fr) = refine(ink, cx, cy, k * r, 4);
                if !(r_range.0 * 0.8 <= fr && fr <= r_range.1 * 1.2) {
                    continue;
                }
                if fits.iter().any(|&(a, b, c)| (fx - a).hypot(fy - b) < 2.0 && (fr - c).abs() < 2.0) {
                    continue;
                }
                fits.push((fx, fy, fr));
                if let Some(m) = ring(ink, fx, fy, fr, 180) {
                    out.push(Measured { kind, cx: fx + offset.0, cy: fy + offset.1, r: fr, m });
                }
            }
        }
    }
    Ok(out)
}

fn passes(m: &Measured, limits: &[&Limits]) -> bool {
    let (cont, dens, outside) = m.m;
    limits.iter().any(|lim| {
        let l = lim[m.kind];
        cont >= l.0 && dens >= l.1 && outside <= l.2 * dens
    })
}

/// Concentric or overlapping detections of one stamp: keep the outermost
/// ring, grown to cover the others.
fn dedupe(stamps: &[Stamp]) -> Vec<Stamp> {
    let mut s = stamps.to_vec();
    s.sort_by(|a, b| b.r.partial_cmp(&a.r).unwrap());
    let mut out: Vec<Stamp> = Vec::new();
    for st in s {
        let mut merged = false;
        for o in out.iter_mut() {
            let d = (st.cx - o.cx).hypot(st.cy - o.cy);
            if d < 0.6 * st.r.max(o.r) {
                o.r = o.r.max(d + st.r);
                o.coverage = o.coverage.max(st.coverage);
                merged = true;
                break;
            }
        }
        if !merged {
            out.push(st.clone());
        }
    }
    out
}

/// Working copies of a page (dark and coloured ink), shared by `find` and `near`.
pub struct Maps {
    pub scale: f64,
    pub maps: Vec<Img>,
}

pub fn ink_maps(img: &Mat) -> Result<Maps> {
    let gray = util::gray(img)?;
    let s = (WORK / gray.rows().max(gray.cols()) as f64).min(1.0);
    let rs = |a: &Mat| -> Result<Mat> { if s < 1.0 { util::resize_by(a, s, imgproc::INTER_AREA) } else { Ok(a.try_clone()?) } };
    let mut maps = vec![Img::from_mat(&dark_ink(&rs(&gray)?)?)?];
    if img.channels() == 3 {
        maps.push(Img::from_mat(&color_ink(&rs(img)?)?)?);
    }
    Ok(Maps { scale: s, maps })
}

/// An OCR line: text and box (x0, y0, x1, y1) in image px.
pub struct TextLine {
    pub text: String,
    pub bbox: [f64; 4],
}

fn crop_maps(maps: &[Img], x0: usize, y0: usize, x1: usize, y1: usize) -> Vec<Img> {
    maps.iter().map(|m| m.crop(x0, y0, x1, y1)).collect()
}

/// Stamps on a page. `r_range`: plausible stamp radius in image px.
pub fn find(img: &Mat, r_range: (f64, f64), text_lines: &[TextLine], maps: Option<&Maps>) -> Result<Vec<Stamp>> {
    let own;
    let m = match maps {
        Some(m) => m,
        None => {
            own = ink_maps(img)?;
            &own
        }
    };
    let s = m.scale;
    let rr = (r_range.0 * s, r_range.1 * s);
    let mut stamps: Vec<Stamp> = measure(&m.maps, rr, 30.0, (0.0, 0.0))?
        .iter()
        .filter(|x| passes(x, &[&STRICT]))
        .map(|x| Stamp::new(x.cx, x.cy, x.r, x.m.0, "ring"))
        .collect();
    for ln in text_lines {
        if !is_stamp_text(&ln.text) {
            continue;
        }
        let [x0, y0, x1, y1] = ln.bbox.map(|v| v * s);
        let (mx, my) = ((x0 + x1) / 2.0, (y0 + y1) / 2.0);
        if stamps.iter().any(|st| (mx - st.cx).hypot(my - st.cy) < st.r) {
            continue;
        }
        let pad = (2.2 * rr.1) as i64;
        let (bx0, by0) = ((x0 as i64 - pad).max(0) as usize, (y0 as i64 - pad).max(0) as usize);
        let sub = crop_maps(&m.maps, bx0, by0, (x1 as i64 + pad).max(0) as usize, (y1 as i64 + pad).max(0) as usize);
        let local: Vec<Stamp> = measure(&sub, rr, 20.0, (bx0 as f64, by0 as f64))?
            .iter()
            .filter(|x| passes(x, &[&LOOSE]))
            .map(|x| Stamp::new(x.cx, x.cy, x.r, x.m.0, "text"))
            .filter(|t| (t.cx - mx).hypot(t.cy - my) < t.r)
            .collect();
        if !local.is_empty() {
            stamps.extend(local);
        } else {
            stamps.push(Stamp::new(mx, my, rr.1.max(0.6 * (x1 - x0)), 0.0, "text"));
        }
    }
    Ok(dedupe(&stamps).into_iter().map(|t| Stamp::new(t.cx / s, t.cy / s, t.r / s, t.coverage, &t.how)).collect())
}

/// Permissive ring search around everything about to be masked (`hide`:
/// 255 = masked, image px). A stamp under a mask is the costly miss.
pub fn near(img: &Mat, hide: &Mat, r_range: (f64, f64), known: &[Stamp], maps: Option<&Maps>) -> Result<Vec<Stamp>> {
    let own;
    let m = match maps {
        Some(m) => m,
        None => {
            own = ink_maps(img)?;
            &own
        }
    };
    let s = m.scale;
    let (mh, mw) = (m.maps[0].h, m.maps[0].w);
    let hs_m = util::resize(hide, mw as i32, mh as i32, imgproc::INTER_NEAREST)?;
    let hs = Img::from_mat(&hs_m)?;
    let rr = (r_range.0 * s, r_range.1 * s);
    let pad = (1.2 * rr.1) as i32;
    let (_, st, _) = util::components(&util::dilate(&hs_m, &util::ones_kernel(pad, pad)?)?)?;
    let reach = (2.2 * rr.1) as i64 - (pad / 2) as i64;
    let mut found: Vec<Stamp> = Vec::new();
    for stv in &st[1..] {
        let [x, y, w, h, _] = stv.map(|v| v as i64);
        let (bx0, by0) = ((x - reach).max(0), (y - reach).max(0));
        let sub = crop_maps(&m.maps, bx0 as usize, by0 as usize, (x + w + reach).max(0) as usize, (y + h + reach).max(0) as usize);
        for ms in measure(&sub, rr, 22.0, (bx0 as f64, by0 as f64))? {
            if !passes(&ms, &[&LOOSE, &FAINT]) {
                continue;
            }
            let t = Stamp::new(ms.cx, ms.cy, ms.r, ms.m.0, "near-mask");
            let lim = (t.r * 1.05).powi(2);
            let (ya, yb) = (((t.cy - t.r * 1.05).floor().max(0.0)) as usize, ((t.cy + t.r * 1.05).ceil() + 1.0).min(mh as f64).max(0.0) as usize);
            let (xa, xb) = (((t.cx - t.r * 1.05).floor().max(0.0)) as usize, ((t.cx + t.r * 1.05).ceil() + 1.0).min(mw as f64).max(0.0) as usize);
            let mut reaches = false;
            'scan: for yy in ya..yb {
                for xx in xa..xb {
                    if hs.at(xx, yy) > 0 && (xx as f64 - t.cx).powi(2) + (yy as f64 - t.cy).powi(2) <= lim {
                        reaches = true;
                        break 'scan;
                    }
                }
            }
            if !reaches {
                continue; // only rings that reach the mask matter here
            }
            if mean_arc(&sub[ms.kind], ms.cx - bx0 as f64, ms.cy - by0 as f64, ms.r, 180) < MIN_ARC {
                continue; // crossings, not a ring
            }
            if t.r < rr.0 && t.coverage < FAINT[1].0 {
                continue; // smaller than any stamp, and broken
            }
            if known.iter().any(|k| (t.cx - k.cx * s).hypot(t.cy - k.cy * s) < 0.6 * k.r * s) {
                continue;
            }
            found.push(t);
        }
    }
    Ok(dedupe(&found).into_iter().map(|t| Stamp::new(t.cx / s, t.cy / s, t.r / s, t.coverage, &t.how)).collect())
}

/// Mask (255 = stamp, grown by `margin` px).
pub fn protect_mask(rows: i32, cols: i32, stamps: &[Stamp], margin: f64) -> Result<Mat> {
    let mut m = util::zeros_u8(rows, cols)?;
    for st in stamps {
        util::fill_poly(&mut m, &st.polygon(margin), 255.0)?;
    }
    Ok(m)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn numpy_arange() {
        assert_eq!(arange(-2.0, 2.5, 1.0), vec![-2.0, -1.0, 0.0, 1.0, 2.0]);
        assert_eq!(arange(7.3, 14.6, 1.0).len(), 8);
        assert!(is_stamp_text("I.E.B.C"));
        assert!(!is_stamp_text("Presiding Officer"));
    }

}

#[cfg(test)]
#[path = "tests/stamps.rs"]
mod py_tests;
