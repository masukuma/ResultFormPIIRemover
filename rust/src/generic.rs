//! Forms without a template: find the PII fields from the page itself.
//!
//! Older and by-election forms (34A 2017, 35A, 36A, 37A, 38A, ...) share a
//! family layout but differ in detail: the candidates table grows with the
//! number of candidates, so everything below it moves; agents tables have 5
//! to 24 rows, in one or two blocks; pages arrive rotated, photographed on a
//! table or scanned on A3. Instead of a template per variant, the page is read:
//!
//! 1. Frame. Text direction (0/90/180/270 degrees) from OCR line detection
//!    and the line-orientation classifier, skew from the text lines, and scale
//!    from the height of printed text, so all later sizes are in units of
//!    text height. Everything below happens in that upright, deskewed frame;
//!    masks are mapped back to the original pixels.
//! 2. Labels. OCR (local, see `ocr.rs`) reads the printed labels.
//! 3. Tables. A header cell reading "ID/Passport No.", "Tel. Contact" or
//!    "Signature" names a column; the column is bounded by the printed
//!    vertical rules around that label and cut into rows by the horizontal
//!    rules below it, down to where the rules stop.
//! 4. Declaration lines. A "Signature" label outside a table names the field
//!    to its right, up to the next printed label on the line ("Date").
//!    Likewise an "ID Number" or "Tel" label outside a table.
//! 5. Free text. An ID or phone number written in the comments ("ID:
//!    2503...") is found from the OCR text and partially masked like any
//!    other; unlabelled handwriting on a sign-off line that OCR can't read as
//!    text is taken for a signature.
//! 6. Results. Everything above "Declaration" (codes, candidates' votes,
//!    polling-station counts, disputed votes) is a protected region; so is
//!    every stamp.
//!
//! Anything the reading can't resolve fails the page (human review): a PII
//! label without a field, a table column without rules, no "Declaration" and
//! no results table to protect. DESIGN.md has the measurements behind the
//! thresholds.

use anyhow::Result;
use fancy_regex::Regex;
use once_cell::sync::Lazy;
use opencv::core::{self, Mat, Point, Point2f, Rect, Scalar, Size, Vector};
use opencv::imgproc;
use opencv::prelude::*;
use serde_json::{json, Value};
use std::collections::BTreeSet;
use std::time::Instant;

use crate::masking::{self, Img, MaskSettings, SigOptions};
use crate::ocr::{self, char_index, Line as OLine, Ocr};
use crate::pipeline::{apply, protect_stamps, PageResult, STAMP_MARGIN};
use crate::proof;
use crate::pyjson;
use crate::pyrand::PyRandom;
use crate::stamps::{self, Stamp};
use crate::structure::{self, Line};
use crate::templates::Zone;
use crate::util::{self, median, round_even};

pub const TEXT_H: f64 = 34.0; // text-line height (OCR box, px) in the working frame
pub const STAMP_R: (f64, f64) = (1.6 * TEXT_H, 4.2 * TEXT_H);

fn re(p: &str) -> Regex {
    Regex::new(p).unwrap()
}

pub(crate) static ID_RE: Lazy<Regex> = Lazy::new(|| re(r"(?i)passp[a-z]{0,4}|(?<![a-z])[I1l]\s?\.?\s?D\s*\.?\s*(?:/|no\b|no\.|num|number)|national\s*id"));
// "Mobile" heads a label ("Mobile No.:") but also ends polling stations'
// names ("LOMUKE MOBILE (039)", read "LOMUKEMOBILE(039)"): there the code
// after it is the station's
pub(crate) static TEL_RE: Lazy<Regex> = Lazy::new(|| re(r"(?i)(?<![a-z])tel(?![a-z])|telephone|contact|phone|^\W*mobile|mobile\s*(?:no\b|no\.|num|:)"));
pub(crate) static SIG_RE: Lazy<Regex> = Lazy::new(|| re(r"(?i)signature|(?<![a-z])sign(?![a-z])"));
static DATE_RE: Lazy<Regex> = Lazy::new(|| re(r"(?i)(?<!di)date"));
pub(crate) static REFUSAL_RE: Lazy<Regex> = Lazy::new(|| re(r"(?i)refus"));
static LABEL_RE: Lazy<Regex> = Lazy::new(|| {
    re(r"(?i)(?<!di)date|signature|name|officer|polling|station|constituency|county|ward|passport|(?<![a-z])tel(?![a-z])|contact|(?<![a-z])I\.?D(?![a-z])")
});
// the section, not a title "Declaration of ... results" (2013's Form 34)
static DECL_RE: Lazy<Regex> = Lazy::new(|| re(r"(?i)^\s*(?:\d{1,2}\s*[.)]\s*)?declaration(?!\s*of)|^\W*we,?\s*the\s*undersigned"));
static COUNT_LABEL_RE: Lazy<Regex> = Lazy::new(|| re(r"(?i)total\s*(?:number|valid)|registered\s*voters|valid\s*votes\s*(?:cast|obtained)"));
static FREE_ID_RE: Lazy<Regex> = Lazy::new(|| re(r"(?<![\d/-])((?:\d\s?){6,8}\d)(?![\d/-])"));
static ID_PREFIX_RE: Lazy<Regex> = Lazy::new(|| re(r"(?i)(?<![a-z])(?:[I1l|]\s?\.?\s?D|ID\s*no|ID\s*number)\s*\.?\s*[:.,;#-]?\s*(\S.{4,15})"));
static FREE_PHONE_RE: Lazy<Regex> = Lazy::new(|| re(r"(?<!\d)((?:\+\s?254|0)\s?[17](?:\s?\d){8})(?!\d)"));
static PRINTED_LINE: Lazy<Regex> = Lazy::new(|| {
    re(r"(?i)^[\s.:,/()_-]*(?:no|name\s*of\s*candidate(?:\s*or\s*agent)?|or\s*agent|agent|id\s*/?\s*passport(?:\s*no)?|i\.?\s*d\.?\s*no\.?\s*/?\s*passp[a-z]{0,4}|passport|party\s*name|independent(?:\s*candidate)?|candidate|tel\.?(?:\s*contact)?|contact|signature|date)[\s.:,/()_-]*$")
});
static FREE_LABEL: Lazy<Regex> = Lazy::new(|| {
    re(r"(?i)presiding\s*officer'?s?\s*comments?\s*:?|reasons?\s*for\s*refus[ae]l\s*to\s*sign(?:\s*\(?\s*if\s*any\s*\)?)?")
});
static NA_RE: Lazy<Regex> = Lazy::new(|| re(r"(?i)^\W*(?:n\s*/\s*a|nil|none|na)\W*$"));

pub(crate) fn has(r: &Regex, t: &str) -> bool {
    r.is_match(t).unwrap_or(false)
}

/// (start, end) char indexes of the matches of `r` in `t`.
fn matches(r: &Regex, t: &str) -> Vec<(usize, usize)> {
    r.find_iter(t).filter_map(|m| m.ok()).map(|m| (char_index(t, m.start()), char_index(t, m.end()))).collect()
}

/// Python 3.12+'s `sum()` of floats (Neumaier-compensated).
pub fn py_sum(v: &[f64]) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    let (mut f, mut c) = (0.0 + v[0], 0.0);
    for &x in &v[1..] {
        let t = f + x;
        if f.abs() >= x.abs() {
            c += (f - t) + x;
        } else {
            c += (x - t) + f;
        }
        f = t;
    }
    if c != 0.0 && c.is_finite() {
        f += c;
    }
    f
}

fn py_isdigit(c: char) -> bool {
    c.is_numeric()
}

// --- layout --------------------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct Frame {
    pub k: i32,       // quarter turns clockwise that make the text upright
    pub skew: f64,    // degrees, removed after the quarter turns
    pub scale: f64,   // original px -> working px
    pub m: [[f64; 3]; 3], // original -> working
    pub size: (i32, i32), // working image (w, h)
    pub text_h: f64,
}

impl Frame {
    pub fn to_image(&self) -> Result<[[f64; 3]; 3]> {
        util::inv3(&self.m)
    }
}

#[derive(Clone, Debug)]
pub struct Field {
    pub kind: String,   // "id" | "phone" | "signature"
    pub zones: Vec<Zone>,
    pub source: String, // "table" | "label" | "declaration" | "free-text"
    pub label: String,
    pub label_box: (f64, f64, f64, f64),
}

impl Field {
    pub fn record(&self, f: &masking::Mapper) -> Value {
        let (x0, y0, x1, y1) = self.label_box;
        let quad = f(&[[x0, y0], [x1, y0], [x1, y1], [x0, y1]]);
        let q: Vec<Value> = quad.iter().map(|p| json!([pyjson::np_round(p[0], 1), pyjson::np_round(p[1], 1)])).collect();
        json!({"kind": self.kind, "source": self.source, "label": self.label, "cells": self.zones.len(), "label_polygon": q})
    }
}

pub struct Layout {
    pub frame: Frame,
    pub image: Mat,
    pub gray: Mat,
    pub lines: Vec<OLine>,
    pub fields: Vec<Field>,
    pub protected: Vec<(String, Vec<[f64; 2]>)>,
    pub stamps: Vec<Stamp>,
    pub problems: Vec<String>,
    pub notes: Vec<String>,
    pub officer_names: BTreeSet<String>,
}

fn quad_dims(q: &ocr::Quad) -> (f64, f64) {
    let n = |a: [f32; 2], b: [f32; 2]| {
        let (dx, dy) = (a[0] - b[0], a[1] - b[1]);
        (dx * dx + dy * dy).sqrt()
    };
    let w = (n(q[1], q[0]) + n(q[2], q[3])) / 2.0;
    let h = (n(q[3], q[0]) + n(q[2], q[1])) / 2.0;
    (w as f64, h as f64)
}

fn rot90_cw(img: &Mat, quarter_turns: i32) -> Result<Mat> {
    // np.rot90(img, -k): k quarter turns clockwise
    let k = quarter_turns.rem_euclid(4);
    if k == 0 {
        return Ok(img.try_clone()?);
    }
    let code = match k {
        1 => core::ROTATE_90_CLOCKWISE,
        2 => core::ROTATE_180,
        _ => core::ROTATE_90_COUNTERCLOCKWISE,
    };
    let mut out = Mat::default();
    core::rotate(img, &mut out, code)?;
    Ok(out)
}

fn to_bgr(img: &Mat) -> Result<Mat> {
    if img.channels() == 3 {
        return Ok(img.try_clone()?);
    }
    let mut out = Mat::default();
    imgproc::cvt_color_def(img, &mut out, imgproc::COLOR_GRAY2BGR)?;
    Ok(out)
}

pub fn find_frame(img: &Mat, o: &mut Ocr, work: f64) -> Result<Option<Frame>> {
    let (h0, w0) = (img.rows() as f64, img.cols() as f64);
    let s0 = (work / h0.max(w0)).min(1.0);
    let small = if s0 < 1.0 { util::resize_by(img, s0, imgproc::INTER_AREA)? } else { img.try_clone()? };
    let small = to_bgr(&small)?;
    let dims = |qs: Vec<ocr::Quad>| -> Vec<(ocr::Quad, f64, f64)> {
        qs.into_iter().map(|q| {
            let (w, h) = quad_dims(&q);
            (q, w, h)
        }).collect()
    };
    let mut boxes = dims(o.detect(&small)?);
    let wide = py_sum(&boxes.iter().filter(|b| b.1 > 1.5 * b.2).map(|b| b.1 * b.2).collect::<Vec<_>>());
    let tall = py_sum(&boxes.iter().filter(|b| b.2 > 1.5 * b.1).map(|b| b.1 * b.2).collect::<Vec<_>>());
    if wide + tall == 0.0 {
        return Ok(None);
    }
    let mut k = if tall > wide { 1 } else { 0 };
    let mut up = rot90_cw(&small, k)?;
    if k != 0 {
        boxes = dims(o.detect(&up)?);
    }
    let pick = |boxes: &Vec<(ocr::Quad, f64, f64)>| -> Vec<(ocr::Quad, f64, f64)> {
        let mut l: Vec<(ocr::Quad, f64, f64)> = boxes.iter().filter(|b| b.1 > 2.0 * b.2).cloned().collect();
        l.sort_by(|a, b| (-a.1).partial_cmp(&(-b.1)).unwrap());
        l.truncate(40);
        l
    };
    let mut lines = pick(&boxes);
    if lines.len() < 5 {
        return Ok(None);
    }
    let votes = o.classify(&up, &lines.iter().map(|b| b.0).collect::<Vec<_>>())?;
    let flipped = votes.iter().filter(|(l, c)| l == "180" && *c > 0.8).count();
    let upright = votes.iter().filter(|(l, c)| l == "0" && *c > 0.8).count();
    if flipped > upright {
        k += 2;
        up = rot90_cw(&up, 2)?;
        boxes = dims(o.detect(&up)?);
        lines = pick(&boxes);
    }
    let angles: Vec<f64> = lines.iter().map(|(q, _, _)| ((q[1][1] - q[0][1]) as f64).atan2((q[1][0] - q[0][0]) as f64).to_degrees()).collect();
    let skew = if angles.is_empty() { 0.0 } else { median(&angles).clamp(-8.0, 8.0) };
    let th = median(&boxes.iter().filter(|b| b.1 > 1.5 * b.2).map(|b| b.2).collect::<Vec<_>>()) / s0;
    let scale = TEXT_H / th;
    let ang = -90.0 * k as f64 + skew;
    let (ww, hh) = (w0 * scale, h0 * scale);
    let r = opencv::geometry::get_rotation_matrix_2d(Point2f::new((ww / 2.0) as f32, (hh / 2.0) as f32), ang, 1.0)?;
    let mut rm = [[0.0f64; 3]; 2];
    for i in 0..2 {
        for j in 0..3 {
            rm[i][j] = *r.at_2d::<f64>(i as i32, j as i32)?;
        }
    }
    let corners = [[0.0, 0.0], [ww, 0.0], [ww, hh], [0.0, hh]];
    let mut lo = [f64::INFINITY; 2];
    let mut hi = [f64::NEG_INFINITY; 2];
    for c in corners {
        for i in 0..2 {
            let v = c[0] * rm[i][0] + c[1] * rm[i][1] + rm[i][2];
            lo[i] = lo[i].min(v);
            hi[i] = hi[i].max(v);
        }
    }
    rm[0][2] -= lo[0];
    rm[1][2] -= lo[1];
    let m = [
        [rm[0][0] * scale, rm[0][1] * scale, rm[0][2]],
        [rm[1][0] * scale, rm[1][1] * scale, rm[1][2]],
        [0.0, 0.0, 1.0],
    ];
    let size = ((hi[0] - lo[0]).ceil() as i32, (hi[1] - lo[1]).ceil() as i32);
    Ok(Some(Frame { k: k.rem_euclid(4), skew, scale, m, size, text_h: th }))
}

/// The working-frame image.
pub fn warp(img: &Mat, f: &Frame) -> Result<Mat> {
    let (src, m) = if f.scale < 0.8 {
        let s = util::resize_by(img, f.scale, imgproc::INTER_AREA)?;
        let r = 1.0 / f.scale;
        (s, [[f.m[0][0] * r, f.m[0][1] * r, f.m[0][2]], [f.m[1][0] * r, f.m[1][1] * r, f.m[1][2]]])
    } else {
        (img.try_clone()?, [[f.m[0][0], f.m[0][1], f.m[0][2]], [f.m[1][0], f.m[1][1], f.m[1][2]]])
    };
    let mut mm = Mat::new_rows_cols_with_default(2, 3, core::CV_64F, Scalar::all(0.0))?;
    for i in 0..2 {
        for j in 0..3 {
            *mm.at_2d_mut::<f64>(i as i32, j as i32)? = m[i][j];
        }
    }
    let mut out = Mat::default();
    imgproc::warp_affine(&src, &mut out, &mm, Size::new(f.size.0, f.size.1), imgproc::INTER_LINEAR, core::BORDER_REPLICATE, Scalar::default(), core::AlgorithmHint::ALGO_HINT_DEFAULT)?;
    Ok(out)
}

// --- tables --------------------------------------------------------------------------------------

/// Join collinear pieces of a rule broken by handwriting, but not across a wide gap.
fn merge_lines(lines: Vec<Line>, max_gap: f64, tol: f64) -> Vec<Line> {
    let mut sorted = lines;
    sorted.sort_by(|a, b| {
        let ka = (round_even(a.at((a.lo + a.hi) / 2.0) / tol), a.lo);
        let kb = (round_even(b.at((b.lo + b.hi) / 2.0) / tol), b.lo);
        ka.partial_cmp(&kb).unwrap()
    });
    let mut out: Vec<Line> = Vec::new();
    for l in sorted {
        let mut done = false;
        for m in out.iter_mut() {
            let mid = (l.lo + l.hi) / 2.0;
            if (m.at(mid) - l.at(mid)).abs() < tol && (m.slope - l.slope).abs() < 0.03 && l.lo - m.hi <= max_gap && m.lo - l.hi <= max_gap {
                let total = m.length() + l.length();
                m.pos = (m.pos * m.length() + l.pos * l.length()) / total;
                m.slope = (m.slope * m.length() + l.slope * l.length()) / total;
                m.lo = m.lo.min(l.lo);
                m.hi = m.hi.max(l.hi);
                done = true;
                break;
            }
        }
        if !done {
            out.push(Line { pos: l.pos, slope: l.slope, lo: l.lo, hi: l.hi });
        }
    }
    out
}

fn binary(gray: &Mat) -> Result<Mat> {
    util::adaptive_inv(&util::gaussian(gray, 3)?, 31, 8.0)
}

/// Printed rules (stamps erased first).
pub fn rules(gray: &Mat, stamp_list: &[Stamp]) -> Result<(Vec<Line>, Vec<Line>)> {
    let mut bw = binary(gray)?;
    for st in stamp_list {
        imgproc::circle(&mut bw, Point::new(st.cx as i32, st.cy as i32), (st.r * 1.1 + 4.0) as i32, Scalar::all(0.0), -1, imgproc::LINE_8, 0)?;
    }
    let hor = util::morph(&bw, imgproc::MORPH_OPEN, &util::rect_kernel(60, 1)?)?;
    let ver = util::morph(&bw, imgproc::MORPH_OPEN, &util::rect_kernel(1, 30)?)?;
    let hs = merge_lines(structure::segments(&hor, true)?, 1.5 * TEXT_H, 4.0).into_iter().filter(|l| l.length() >= 3.0 * TEXT_H && l.slope.abs() < 0.1).collect();
    let vs = merge_lines(structure::segments(&ver, false)?, 1.5 * TEXT_H, 4.0).into_iter().filter(|l| l.length() >= 2.5 * TEXT_H && l.slope.abs() < 0.1).collect();
    Ok((hs, vs))
}

fn crop(m: &Mat, x0: i32, y0: i32, x1: i32, y1: i32) -> Result<Mat> {
    Ok(Mat::roi(m, Rect::new(x0, y0, x1 - x0, y1 - y0))?.try_clone()?)
}

/// x of a column rule between x0 and x1 that writing breaks in every row.
fn hidden_rule(gray: &Mat, x0: f64, x1: f64, y0: f64, y1: f64) -> Result<Option<f64>> {
    let (a, b) = (x0.max(0.0) as i32, x1.min(gray.cols() as f64) as i32);
    let (ya, yb) = (y0.max(0.0) as i32, y1.min(gray.rows() as f64) as i32);
    if b - a < 3 || ((yb - ya) as f64) < 2.0 * TEXT_H {
        return Ok(None);
    }
    let bw = binary(&crop(gray, a, ya, b, yb)?)?;
    let bw = Img::from_mat(&util::morph(&bw, imgproc::MORPH_OPEN, &util::rect_kernel(1, 7)?)?)?;
    let prof: Vec<f64> = (0..bw.w).map(|x| (0..bw.h).filter(|&y| bw.at(x, y) > 0).count() as f64 / bw.h as f64).collect();
    // np.convolve(prof, ones(3) / 3, "same")
    let third = 1.0 / 3.0;
    let n = prof.len();
    let sm: Vec<f64> = (0..n)
        .map(|i| {
            let mut s = 0.0;
            for d in [-1i64, 0, 1] {
                let j = i as i64 - d;
                if j >= 0 && (j as usize) < n {
                    s += prof[j as usize] * third;
                }
            }
            s
        })
        .collect();
    let mut k = 0;
    for i in 1..n {
        if sm[i] > sm[k] {
            k = i;
        }
    }
    Ok(if sm[k] >= 0.45 { Some((a as usize + k) as f64) } else { None })
}

/// The rule extended down over the pieces in line with it below.
fn extend(vs: &[Line], v: &Line, cy: f64) -> Line {
    let hi = vs
        .iter()
        .filter(|u| (u.at(cy) - v.at(cy)).abs() < 5.0 && (u.slope - v.slope).abs() < 0.03 && u.lo >= v.lo - 0.5 * TEXT_H)
        .map(|u| u.hi)
        .fold(v.hi, f64::max);
    Line { pos: v.pos, slope: v.slope, lo: v.lo, hi }
}

/// Vertical rules left and right of a header label at height cy.
fn column(hs: &[Line], vs: &[Line], x0: f64, x1: f64, cy: f64) -> Option<(Line, Line)> {
    let h = TEXT_H;
    let xm = (x0 + x1) / 2.0;
    let below: Vec<f64> = hs.iter().filter(|l| l.lo - 0.2 * h <= xm && xm <= l.hi + 0.2 * h && cy + 0.3 * h < l.at(xm) && l.at(xm) < cy + 3.0 * h).map(|l| l.at(xm)).collect();
    if below.is_empty() {
        return None;
    }
    let yb = below.iter().cloned().fold(f64::INFINITY, f64::min);
    let x_at = |v: &Line| v.at(cy);
    let span: Vec<&Line> = vs.iter().filter(|v| cy - 3.0 * h <= v.lo && v.lo <= cy + 0.2 * h && v.hi >= yb + 1.5 * h).collect();
    let left: Vec<&Line> = span.iter().cloned().filter(|v| x_at(v) <= x0 + 0.3 * h).collect();
    let right: Vec<&Line> = span.iter().cloned().filter(|v| x_at(v) >= x1 - 0.3 * h).collect();
    if left.is_empty() || right.is_empty() {
        return None;
    }
    // max(left, key=x_at): the first maximal; min(right, key=x_at): the first minimal
    let mut lb = left[0];
    for v in &left[1..] {
        if x_at(v) > x_at(lb) {
            lb = v;
        }
    }
    let mut rb = right[0];
    for v in &right[1..] {
        if x_at(v) < x_at(rb) {
            rb = v;
        }
    }
    let lv = extend(vs, lb, cy);
    let rv = extend(vs, rb, cy);
    if x_at(&rv) - x_at(&lv) < (x1 - x0) * 0.9 {
        return None;
    }
    if (lv.lo - rv.lo).abs() > 3.5 * h || (lv.hi - rv.hi).abs() > 6.0 * h {
        return None;
    }
    Some((lv, rv))
}

/// Row bands (top, bottom) under a header.
fn rows(hs: &[Line], lv: &Line, rv: &Line, cy: f64, h: f64) -> Vec<(f64, f64)> {
    let xm = (lv.at(cy) + rv.at(cy)) / 2.0;
    let bottom = lv.hi.max(rv.hi) + 0.5 * h;
    let mut ys: Vec<f64> = hs.iter().filter(|l| l.lo - 6.0 * h <= xm && xm <= l.hi + 6.0 * h).map(|l| l.at(xm)).filter(|&y| cy + 0.3 * h < y && y <= bottom).collect();
    ys.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mut merged: Vec<f64> = Vec::new();
    for y in ys {
        if let Some(&last) = merged.last() {
            if y - last < 0.35 * h {
                continue;
            }
        }
        merged.push(y);
    }
    if merged.len() < 2 {
        return vec![];
    }
    let pitch = median(&merged.windows(2).map(|p| p[1] - p[0]).collect::<Vec<_>>());
    let mut out = Vec::new();
    for p in merged.windows(2) {
        let (a, b) = (p[0], p[1]);
        let n = round_even((b - a) / pitch) as i64;
        if n > 4 {
            break;
        }
        let n = n.max(1);
        for k in 0..n {
            out.push((a + (b - a) * k as f64 / n as f64, a + (b - a) * (k + 1) as f64 / n as f64));
        }
    }
    let lim = lv.hi.max(rv.hi) + 0.3 * h;
    out.into_iter().filter(|&(a, b)| b - a >= 0.6 * pitch && b <= lim).collect()
}

fn center_x(o: &OLine) -> f64 {
    let b = o.bbox();
    (b.0 + b.2) / 2.0
}

/// PII columns from their header labels. Returns the indexes of the OCR lines used.
fn tables(l: &mut Layout, hs: &[Line], vs: &[Line]) -> Result<BTreeSet<usize>> {
    let mut used = BTreeSet::new();
    let mut seen: Vec<(String, f64, f64, f64)> = Vec::new();
    let n_lines = l.lines.len();
    for i in 0..n_lines {
        for (kind, rx) in [("id", &*ID_RE), ("phone", &*TEL_RE), ("signature", &*SIG_RE)] {
            let ln = l.lines[i].clone();
            if kind == "signature" && has(&REFUSAL_RE, &ln.text) {
                continue;
            }
            let Some(sp) = ln.find(rx) else { continue };
            let Some((mut lv, mut rv)) = column(hs, vs, sp.0, sp.1, ln.cy()) else { continue };
            let (mut xl, mut xr) = (lv.at(ln.cy()), rv.at(ln.cy()));
            if xr - xl > 12.0 * TEXT_H {
                continue;
            }
            used.insert(i);
            if seen.iter().any(|(k, a, _, c)| k == kind && (a - xl).abs() < TEXT_H && (c - ln.cy()).abs() < 4.0 * TEXT_H) {
                continue;
            }
            let others: Vec<OLine> = l
                .lines
                .iter()
                .enumerate()
                .filter(|(j, o)| {
                    *j != i
                        && (o.cy() - ln.cy()).abs() < TEXT_H
                        && (o.find(&ID_RE).is_some() || o.find(&TEL_RE).is_some() || o.find(&SIG_RE).is_some() || o.find(&DATE_RE).is_some())
                        && xl + 3.0 < center_x(o)
                        && center_x(o) < xr - 3.0
                        && !(sp.0 - TEXT_H <= center_x(o) && center_x(o) <= sp.1 + TEXT_H)
                })
                .map(|(_, o)| o.clone())
                .collect();
            let bottom_y = lv.hi.max(rv.hi);
            let mut x: Option<f64> = None;
            for o in &others {
                let ob = o.bbox();
                if center_x(o) > sp.1 {
                    x = hidden_rule(&l.gray, sp.1, ob.0, ln.cy() + TEXT_H, bottom_y)?;
                    if let Some(xx) = x {
                        rv = Line { pos: xx, slope: 0.0, lo: rv.lo, hi: rv.hi };
                    }
                } else {
                    x = hidden_rule(&l.gray, ob.2, sp.0, ln.cy() + TEXT_H, bottom_y)?;
                    if let Some(xx) = x {
                        lv = Line { pos: xx, slope: 0.0, lo: lv.lo, hi: lv.hi };
                    }
                }
                if x.is_none() {
                    break;
                }
            }
            xl = lv.at(ln.cy());
            xr = rv.at(ln.cy());
            if !others.is_empty() && x.is_none() {
                l.problems.push(format!("'{}' column also holds '{}': a column rule is missing", ln.text, others[0].text));
                used.insert(i);
                continue;
            }
            // the header may run on below the label
            let mut top = ln.cy();
            let mut by_y: Vec<&OLine> = l.lines.iter().collect();
            by_y.sort_by(|a, b| a.cy().partial_cmp(&b.cy()).unwrap());
            for o in by_y {
                if top < o.cy() && o.cy() <= top + 1.5 * TEXT_H && has(&PRINTED_LINE, &o.text) && xl < center_x(o) && center_x(o) < xr {
                    top = o.cy();
                }
            }
            let rws = rows(hs, &lv, &rv, top, TEXT_H);
            if rws.is_empty() {
                l.problems.push(format!("'{}' header at y={:.0}: no table rows found under it", ln.text, ln.cy()));
                continue;
            }
            seen.push((kind.to_string(), xl, xr, ln.cy()));
            let mut zones = Vec::new();
            for (r, &(y0, y1)) in rws.iter().enumerate() {
                let ym = (y0 + y1) / 2.0;
                let (x0, x1) = (lv.at(ym) + 3.0, rv.at(ym) - 3.0);
                let name = format!("{}@{:.0},{:.0}", kind, xl, ln.cy());
                let mut z = Zone::new(&format!("{}[{}]", name, r), [x0, y0, x1 - x0, y1 - y0], if kind == "signature" { "signature" } else { "partial" });
                z.digits = match kind {
                    "phone" => 10,
                    "id" => 8,
                    _ => 0,
                };
                z.group = name.clone();
                z.row = r;
                z.rule = match kind {
                    "phone" => "phone".into(),
                    "id" => "id".into(),
                    _ => String::new(),
                };
                z.search = Some(if kind == "signature" { [8.0, 30.0] } else { [6.0, 12.0] });
                zones.push(z);
            }
            l.fields.push(Field { kind: kind.into(), zones, source: "table".into(), label: ln.text.clone(), label_box: ln.bbox() });
        }
    }
    Ok(used)
}

// --- declaration -----------------------------------------------------------------------------

static DATE_ISH: Lazy<Regex> = Lazy::new(|| re(r"\d[\dOoIlQ]?\s*[-/.~,]+\s*\S{1,3}?\s*[-/.~,]+\s*[\dOoIlQ]{1,4}|\d{1,2}\s*[-/.~]\s*20\d\d"));

/// Char index where a handwritten date starts in OCR text, or None.
fn date_ish(text: &str) -> Option<usize> {
    let m = DATE_ISH.find(text).ok().flatten()?;
    if m.as_str().chars().filter(|c| py_isdigit(*c)).count() >= 4 {
        Some(char_index(text, m.start()))
    } else {
        None
    }
}

fn char_slice(text: &str, from: usize) -> String {
    text.chars().skip(from).collect()
}

/// "Signature" in OCR text, allowing for misreads. Not "Refusal to Sign".
fn sig_labels(ln: &OLine) -> Vec<(usize, usize)> {
    if has(&REFUSAL_RE, &ln.text) {
        return vec![];
    }
    let mut out = matches(&SIG_RE, &ln.text);
    for (i, j, d) in ln.fuzzy("signature", 3) {
        let piece: String = ln.chars[i..j].iter().collect::<String>().to_lowercase();
        if d <= 2 || piece.ends_with("ure") || piece.ends_with("tur") {
            if out.iter().all(|&(a, b)| j <= a || i >= b) {
                out.push((i, j));
            }
        }
    }
    out
}

/// "Date" labels; a misread one counts if a date follows it, or with `misread`.
fn date_labels(ln: &OLine, misread: bool) -> Vec<(usize, usize)> {
    let mut out = matches(&DATE_RE, &ln.text);
    for (i, j, _) in ln.fuzzy("date", 1) {
        let before: String = ln.chars[i.saturating_sub(2)..i].iter().collect::<String>().to_lowercase();
        if before == "di" {
            continue;
        }
        if (misread || date_ish(&char_slice(&ln.text, j)).is_some()) && out.iter().all(|&(a, b)| j <= a || i >= b) {
            out.push((i, j));
        }
    }
    out
}

#[derive(Default)]
struct DeclRow {
    cy: f64,
    labels: Vec<f64>,
    values: Vec<f64>,
    sigs: Vec<f64>,
    misread: Vec<f64>,
    officer: bool,
}

static OFFICER_RE: Lazy<Regex> = Lazy::new(|| re(r"(?i)presid|officer|deputy"));
static CAPS3: Lazy<Regex> = Lazy::new(|| re(r"[A-Z]{3,}"));
static NOT_NAME: Lazy<Regex> = Lazy::new(|| re(r"^(?:PRESIDING|DEPUTY|OFFICERS?|SIGNATURE|SIGN|DATE)$"));

/// Signature fields on the officers' lines of the declaration.
fn declaration(l: &mut Layout, top: f64, bottom: f64) -> BTreeSet<usize> {
    let h = TEXT_H;
    let region: Vec<(usize, OLine)> = l.lines.iter().enumerate().filter(|(_, ln)| top < ln.cy() && ln.cy() < bottom).map(|(i, ln)| (i, ln.clone())).collect();
    let mut rws: Vec<DeclRow> = Vec::new();
    fn row(rws: &mut Vec<DeclRow>, cy: f64, h: f64) -> usize {
        for (k, r) in rws.iter().enumerate() {
            if (r.cy - cy).abs() < 0.6 * h {
                return k;
            }
        }
        rws.push(DeclRow { cy, ..Default::default() });
        rws.len() - 1
    }
    let mut used = BTreeSet::new();
    for (i, ln) in &region {
        for (a, b) in date_labels(ln, false) {
            let k = row(&mut rws, ln.cy(), h);
            rws[k].labels.push(ln.span(a, b).0);
            used.insert(*i);
        }
        if let Some(k0) = date_ish(&ln.text) {
            let k = row(&mut rws, ln.cy(), h);
            rws[k].values.push(ln.span(k0, k0 + 1).0);
        }
        for (a, b) in sig_labels(ln) {
            let k = row(&mut rws, ln.cy(), h);
            rws[k].sigs.push(ln.span(a, b).1);
            used.insert(*i);
        }
        for (a, b) in date_labels(ln, true) {
            let k = row(&mut rws, ln.cy(), h);
            rws[k].misread.push(ln.span(a, b).0);
        }
        // a stamp's "PRESIDING OFFICER" read into the name written under
        // it ("PRESiDINGNSOZOGZ") names no officer; the printed label, in
        // small letters, does even under a stamp
        let printed = matches(&OFFICER_RE, &ln.text).into_iter().any(|(a, b)| {
            let (x0, x1) = ln.span(a, b);
            let caps = ln.chars[a..b].iter().filter(|c| c.is_uppercase()).count() as f64 >= 0.8 * (b - a) as f64;
            !caps || !l.stamps.iter().any(|s| (0.5 * (x0 + x1) - s.cx).hypot(ln.cy() - s.cy) < s.r)
        });
        if printed && !stamps::is_stamp_text(&ln.text) {
            let k = row(&mut rws, ln.cy(), h);
            rws[k].officer = true;
        }
    }
    let officer_rows: Vec<usize> = (0..rws.len()).filter(|&k| !rws[k].sigs.is_empty() || !rws[k].labels.is_empty() || (!rws[k].values.is_empty() && rws[k].officer)).collect();
    for &k in &officer_rows {
        let r = &rws[k];
        let end: Option<f64> = if !r.sigs.is_empty() {
            Some(r.sigs.iter().cloned().fold(f64::INFINITY, f64::min))
        } else {
            r.labels.iter().chain(r.values.iter()).cloned().reduce(f64::min)
        };
        for (_, ln) in &region {
            if (ln.cy() - r.cy).abs() >= 0.6 * h {
                continue;
            }
            for (a, b) in matches(&CAPS3, &ln.text) {
                let word: String = ln.chars[a..b].iter().collect();
                if has(&NOT_NAME, &word) {
                    continue;
                }
                if let Some(e) = end {
                    if ln.span(a, b).0 >= e {
                        continue;
                    }
                }
                l.officer_names.insert(word);
            }
        }
    }
    let label_x: Vec<f64> = officer_rows.iter().flat_map(|&k| rws[k].labels.clone()).collect();
    let sig_x: Vec<f64> = officer_rows.iter().flat_map(|&k| rws[k].sigs.clone()).collect();
    let mut order = officer_rows.clone();
    order.sort_by(|&a, &b| rws[a].cy.partial_cmp(&rws[b].cy).unwrap());
    for k in order {
        let r = &rws[k];
        let cy = r.cy;
        let (start, mut how) = if !r.sigs.is_empty() {
            (r.sigs.iter().cloned().fold(f64::NEG_INFINITY, f64::max), "'Signature' label".to_string())
        } else if !sig_x.is_empty() {
            (median(&sig_x), "'Signature' label on the other officer line".to_string())
        } else {
            l.problems.push(format!("declaration line at y={:.0}: no 'Signature' label read", cy));
            continue;
        };
        let right: Vec<f64> = r.labels.iter().cloned().filter(|&x| x > start).collect();
        let maybe: Vec<f64> = if !r.sigs.is_empty() { r.misread.iter().cloned().filter(|&x| x > start + 1.5 * h).collect() } else { vec![] };
        let lab_right: Vec<f64> = label_x.iter().cloned().filter(|&x| x > start).collect();
        let val_right: Vec<f64> = r.values.iter().cloned().filter(|&x| x > start).collect();
        let end = if !right.is_empty() {
            right.iter().cloned().fold(f64::INFINITY, f64::min)
        } else if !maybe.is_empty() {
            how += "; 'Date' read approximately";
            maybe.iter().cloned().fold(f64::INFINITY, f64::min)
        } else if !lab_right.is_empty() {
            median(&lab_right)
        } else if !val_right.is_empty() {
            val_right.iter().cloned().fold(f64::INFINITY, f64::min) - 2.6 * h
        } else {
            how += "; no 'Date' label: field taken as 9 text heights";
            start + 9.0 * h
        };
        let (x0, x1) = (start + 0.15 * h, end - 0.15 * h);
        if x1 - x0 < 1.5 * h {
            l.problems.push(format!("declaration line at y={:.0}: no room for a signature before 'Date'", cy));
            continue;
        }
        let mut z = Zone::new(&format!("signature@{:.0},{:.0}", x0, cy), [x0, cy - 0.75 * h, x1 - x0, 1.3 * h], "signature");
        z.search = Some([0.9 * h, 0.5 * h]);
        l.fields.push(Field { kind: "signature".into(), zones: vec![z], source: "declaration".into(), label: how, label_box: (x0, cy - 0.5 * h, x1, cy + 0.5 * h) });
    }
    let named = rws.iter().filter(|r| r.officer).count();
    if officer_rows.len() < 2.min(named) {
        l.problems.push(format!("declaration: {} officer(s) named but {} signature line(s) found", named, officer_rows.len()));
    }
    used
}

static PARTY_RE: Lazy<Regex> = Lazy::new(|| re(r"(?i)party|independent"));
static HEAD_WORDS: Lazy<Regex> = Lazy::new(|| re(r"(?i)names?|of|candidates?|or|agents?|party|independent|date|reasons?|for|refusal|to|sign|serial|remarks?|the|if|any|present|no"));

/// Has every column of the table whose heading row is at `cy` a heading
/// read, each a PII column found (`kinds`) or of none (number, name, party,
/// date, reasons for refusal)? Then a table without ID or phone columns (a
/// 2013 Form 34's: name, signature, reasons for refusal) has none. An
/// unread or unknown heading may be the one missed; a column too narrow for
/// a number needs none. A stamp over the headings, or a party column (the
/// agents' table since 2017, which has them), says it may be missing.
fn headings_read(l: &Layout, hs: &[Line], vs: &[Line], cy: f64, kinds: &BTreeSet<String>) -> bool {
    let h = TEXT_H;
    // the table's width: its top rule, the longest just above the headings
    let Some(top) = hs.iter().filter(|r| r.at((r.lo + r.hi) / 2.0) < cy && r.at((r.lo + r.hi) / 2.0) > cy - 2.0 * h).max_by(|a, b| a.length().partial_cmp(&b.length()).unwrap()) else {
        return false;
    };
    let row: Vec<&OLine> = l.lines.iter().filter(|o| (o.cy() - cy).abs() < 0.6 * h).collect();
    // a stamp on the heading row may hide a heading, and the rule beside it
    if l.stamps.iter().any(|s| (s.cy - cy).abs() < s.r + h && top.lo - h < s.cx + s.r && s.cx - s.r < top.hi + h) {
        return false;
    }
    // a table with a party column is the agents' table of the forms since
    // 2017, which has both ID and phone columns
    if row.iter().any(|o| has(&PARTY_RE, &o.text)) && !(kinds.contains("id") && kinds.contains("phone")) {
        return false;
    }
    // a column rule doesn't run through a heading (a scanner's streak may)
    let mut xs: Vec<f64> = vs
        .iter()
        .filter(|v| v.lo < cy && cy < v.hi)
        .map(|v| v.at(cy))
        .filter(|&x| top.lo - h < x && x < top.hi + h && !row.iter().any(|o| o.bbox().0 + 0.5 * h < x && x < o.bbox().2 - 0.5 * h))
        .collect();
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    if xs.len() < 3 {
        return false;
    }
    for p in xs.windows(2) {
        let (a, b) = (p[0], p[1]);
        // too narrow for an ID or phone number (the rows' numbers)
        if b - a < 3.0 * h {
            continue;
        }
        let heads: Vec<&&OLine> = row.iter().filter(|o| a < center_x(o) && center_x(o) < b).collect();
        if heads.is_empty() {
            return false;
        }
        for o in heads {
            let kind = if has(&ID_RE, &o.text) {
                Some("id")
            } else if has(&TEL_RE, &o.text) {
                Some("phone")
            } else if has(&SIG_RE, &o.text) && !has(&REFUSAL_RE, &o.text) {
                Some("signature")
            } else {
                None
            };
            let known = match kind {
                Some(k) => kinds.contains(k),
                // a letter or two misread in a long heading ("to ign")
                None => {
                    let letters = o.text.chars().filter(|c| c.is_alphabetic()).count() as f64;
                    let left = HEAD_WORDS.replace_all(&o.text, "").chars().filter(|c| c.is_alphabetic()).count() as f64;
                    left <= 2.0f64.max(0.2 * letters)
                }
            };
            if !known {
                return false;
            }
        }
    }
    true
}

/// A page that opens with an officer's lines, no declaration above them:
/// it goes on with the declaration of the page before (a 2013 Form 34's
/// back: the deputy's lines, the agents, comments), and has no results on
/// it if no count is printed on it either. The deputy's label may be
/// garbled ("DeputyPresigingOfficer") or unread, his 'ID No. / Passport'
/// read, or the page open with the agents (the deputy signed on the
/// front). A word of results anywhere above the comments (a blurred front
/// page whose labels OCR garbled) says otherwise.
fn goes_on_with_declaration(lines: &[OLine], page_h: f64) -> bool {
    lines.iter().any(|ln| ln.cy() < 0.2 * page_h && has(&OPENER_RE, &ln.text) && !stamps::is_stamp_text(&ln.text))
        && !lines.iter().any(|ln| ln.cy() < 0.7 * page_h && has(&RESULTS_WORD_RE, &ln.text))
}

// an officer's role (not "presidential"), his ID line, the agents' heading
static OPENER_RE: Lazy<Regex> = Lazy::new(|| re(r"(?i)deputy|presiding\s*officer|id\s*no\W*passport|candidates?\W*or\W*candidates?\W*agents"));
// "vote", not a name written "VOTU"
static RESULTS_WORD_RE: Lazy<Regex> = Lazy::new(|| re(r"(?i)vote|ballot|regist|reject|disput|result"));

static AGENTS_RE: Lazy<Regex> = Lazy::new(|| re(r"(?i)agent"));

/// Is there an agents' section below the results? With no table read under
/// it, what they wrote there would be left to the free-text reading, which
/// misses table cells.
fn agents_section(lines: &[OLine], boundary: f64) -> bool {
    lines.iter().any(|ln| ln.cy() > boundary && has(&AGENTS_RE, &ln.text))
}

static SECTION_RE: Lazy<Regex> = Lazy::new(|| re(r"(?i)agents|candidates|comments|refus"));

/// Where officers' lines that run on from `from` end: each next labelled line
/// (name, ID, signature, date) starting within a line and a half of the last.
fn officer_lines_end(lines: &[OLine], from: f64) -> f64 {
    let mut bottom = from;
    loop {
        let next = lines
            .iter()
            .filter(|ln| ln.bbox().1 < bottom + 1.5 * TEXT_H && ln.bbox().3 > bottom && has(&LABEL_RE, &ln.text))
            .map(|ln| ln.bbox().3 + 0.3 * TEXT_H)
            .reduce(f64::max);
        match next {
            Some(b) if b > bottom => bottom = b,
            _ => return bottom,
        }
    }
}

/// How far below an officer's signature line writing is still the line's:
/// a printed line where nothing is written under it but a stamp's lettering
/// (a 2013 Form 34's last line, over the security stripe), there being no
/// next line's writing to tell it from.
fn hang(l: &Layout, fl: &Field) -> f64 {
    if fl.source != "declaration" {
        return 0.0;
    }
    let (x0, y0, x1, y1) = fl.label_box;
    let (cy, h) = ((y0 + y1) / 2.0, y1 - y0);
    let under = l.lines.iter().any(|o| {
        let (ox0, oy0, ox1, oy1) = o.bbox();
        let in_stamp = l.stamps.iter().any(|s| ((ox0 + ox1) / 2.0 - s.cx).hypot((oy0 + oy1) / 2.0 - s.cy) < s.r);
        oy0 > cy + 0.5 * h && oy0 < cy + 2.0 * h && ox1 > x0 && ox0 < x1 && !in_stamp
    });
    if under { 0.0 } else { h }
}

// --- labelled fields outside tables -------------------------------------------------------------

/// The same words read twice: the officers' lines are read again, zoomed
/// in, and both reads are kept.
fn twins(a: &OLine, b: &OLine) -> bool {
    let (a, b) = (a.bbox(), b.bbox());
    let w = a.2.min(b.2) - a.0.max(b.0);
    let h = a.3.min(b.3) - a.1.max(b.1);
    let area = |r: (f64, f64, f64, f64)| (r.2 - r.0) * (r.3 - r.1);
    w > 0.0 && h > 0.0 && w * h >= 0.6 * area(a).min(area(b))
}

fn next_label_x(l: &Layout, i: usize, after: f64) -> Option<f64> {
    let ln = &l.lines[i];
    let mut best: Option<f64> = None;
    for (j, o) in l.lines.iter().enumerate() {
        // the other read of this line has this line's own label further on
        if (o.cy() - ln.cy()).abs() > 0.6 * ln.height() || (j != i && twins(ln, o)) {
            continue;
        }
        for (a, b) in matches(&LABEL_RE, &o.text) {
            let x0 = o.span(a, b).0;
            if x0 > after + 0.3 * ln.height() && best.map(|bb| x0 < bb).unwrap_or(true) {
                best = Some(x0);
            }
        }
    }
    best
}

static FIVE_DIGITS: Lazy<Regex> = Lazy::new(|| re(r"\d{5}"));
static JOIN_RE: Lazy<Regex> = Lazy::new(|| re(r"^[\s/.:,-]*$"));
// "Passport:" after "ID No." as OCR reads it, with the slash before it
// read as a letter or digit ("ID No. ypassport:", "IDNo.1Passport:") or the
// word itself misread ("Rassport:", "Passoort:"): two letters off at most,
// so an ID read as letters is never taken for it
static PASSPORT_TAIL: Lazy<Regex> = Lazy::new(|| re(r"^[\s/.:,|\\-]*(?:[A-Za-z0-9]\s*)?([PpRr][A-Za-z0-9&]{4,8})\s*:"));

/// Letters to change, add or remove to make `a` into `b`.
fn edits(a: &str, b: &str) -> usize {
    let b: Vec<char> = b.chars().collect();
    let mut row: Vec<usize> = (0..=b.len()).collect();
    for (i, ca) in a.chars().enumerate() {
        let mut prev = row[0];
        row[0] = i + 1;
        for j in 0..b.len() {
            let cur = row[j + 1];
            row[j + 1] = (prev + (ca != b[j]) as usize).min(row[j] + 1).min(cur + 1);
            prev = cur;
        }
    }
    row[b.len()]
}

/// Where a label of this kind on the line ends, unless a number follows it
/// on the line (a station's code, read as part of a heading).
fn label_end(ln: &OLine, kind: &str, rx: &Regex) -> Option<f64> {
    if kind == "signature" && has(&REFUSAL_RE, &ln.text) {
        return None;
    }
    let m = rx.find(&ln.text).ok().flatten()?;
    // a label of several words ("ID No. / Passport:") ends after the last
    let mut end = m.end();
    for n in rx.find_iter(&ln.text).filter_map(|n| n.ok()).filter(|n| n.start() >= m.end()) {
        if !has(&JOIN_RE, &ln.text[end..n.start()]) {
            break;
        }
        end = n.end();
    }
    if kind == "id" && !ln.text[m.start()..end].to_lowercase().contains("passp") {
        if let Some(c) = PASSPORT_TAIL.captures(&ln.text[end..]).ok().flatten() {
            if edits(&c[1].to_lowercase(), "passport") <= 2 {
                end += c.get(0).unwrap().end();
            }
        }
    }
    if kind != "signature" && has(&FIVE_DIGITS, &ln.text[end..]) {
        return None;
    }
    Some(ln.span(char_index(&ln.text, m.start()), char_index(&ln.text, end)).1)
}

fn labels(l: &mut Layout, used: &BTreeSet<usize>, top: f64) {
    let right = l.lines.iter().map(|ln| ln.bbox().2).reduce(f64::max).unwrap_or(l.frame.size.0 as f64);
    for i in 0..l.lines.len() {
        let ln = l.lines[i].clone();
        if used.contains(&i) || ln.cy() < top {
            continue;
        }
        for (kind, rx) in [("signature", &*SIG_RE), ("id", &*ID_RE), ("phone", &*TEL_RE)] {
            let Some(x1) = label_end(&ln, kind, rx) else { continue };
            // a label read twice makes one field, from the read that reads
            // more of it ("ID No. / Rassport:" ends at "No.")
            let better = l.lines.iter().enumerate().any(|(j, o)| {
                j != i && !used.contains(&j) && o.cy() >= top && twins(&ln, o) && label_end(o, kind, rx).is_some_and(|x| x > x1 || (x == x1 && j < i))
            });
            if better {
                continue;
            }
            // a line taller than print has the writing after the label in its
            // box ("ID No. / Passport: 28' @rs0&n?"): a field that tall would
            // reach the next line's "Date:"
            let h = ln.height().min(TEXT_H);
            let end = next_label_x(l, i, x1);
            let fx0 = x1 + 0.15 * h;
            let fx1 = match end {
                Some(e) => e - 0.15 * h,
                None => right.min(x1 + 14.0 * h),
            };
            if fx1 - fx0 < 1.5 * h {
                l.problems.push(format!("'{}': no room for the field after the label", ln.text));
                continue;
            }
            let z = if kind == "signature" {
                let mut z = Zone::new(&format!("signature@{:.0},{:.0}", fx0, ln.cy()), [fx0, ln.cy() - 0.9 * h, fx1 - fx0, 1.5 * h], "signature");
                z.search = Some([0.8 * h, 0.4 * h]);
                z
            } else {
                let mut z = Zone::new(&format!("{}@{:.0},{:.0}", kind, fx0, ln.cy()), [fx0, ln.cy() - 0.7 * h, fx1 - fx0, 1.3 * h], "partial");
                z.digits = if kind == "phone" { 10 } else { 8 };
                z.rule = kind.into();
                z.search = Some([0.3 * h, 0.3 * h]);
                z
            };
            l.fields.push(Field { kind: kind.into(), zones: vec![z], source: "label".into(), label: ln.text.clone(), label_box: ln.bbox() });
        }
    }
}

/// Is (x, y) inside a field found already, or inside a table's rows?
fn covered(l: &Layout, x: f64, y: f64, h: f64, margin: f64) -> bool {
    let m = margin * h;
    for fl in &l.fields {
        for z in &fl.zones {
            let [zx, zy, zw, zh] = z.rect;
            if zx - m <= x && x <= zx + zw + m && zy - m <= y && y <= zy + zh + m {
                return true;
            }
        }
    }
    let same_row: Vec<(f64, f64)> = l
        .fields
        .iter()
        .filter(|fl| fl.source == "table")
        .flat_map(|fl| fl.zones.iter())
        .filter(|z| z.rect[1] - 0.3 * h <= y && y <= z.rect[1] + z.rect[3] + 0.3 * h)
        .map(|z| (z.rect[0], z.rect[0] + z.rect[2]))
        .collect();
    !same_row.is_empty()
        && same_row.iter().map(|r| r.0).fold(f64::INFINITY, f64::min) - 4.0 * h <= x
        && x <= same_row.iter().map(|r| r.1).fold(f64::NEG_INFINITY, f64::max) + 4.0 * h
}

/// Digits that read as a date written without separators.
fn date_digits(d: &str) -> bool {
    let n = d.chars().count();
    for dl in [1usize, 2] {
        for ml in [1usize, 2] {
            for yl in [2usize, 4] {
                if dl + ml + yl != n {
                    continue;
                }
                let p = |a: usize, b: usize| -> i64 { d.chars().skip(a).take(b - a).collect::<String>().parse().unwrap_or(-1) };
                let (day, mon, yr) = (p(0, dl), p(dl, dl + ml), p(dl + ml, n));
                let year_ok = if yl == 4 { (1990..=2040).contains(&yr) } else { yr <= 40 };
                if (1..=31).contains(&day) && (1..=12).contains(&mon) && year_ok {
                    return true;
                }
            }
        }
    }
    false
}

fn free_text(l: &mut Layout, used: &BTreeSet<usize>, top: f64) {
    for i in 0..l.lines.len() {
        let ln = l.lines[i].clone();
        if used.contains(&i) || ln.cy() < top {
            continue;
        }
        let mut hits: Vec<(String, usize, usize, String)> = Vec::new();
        for c in ID_PREFIX_RE.captures_iter(&ln.text).filter_map(|c| c.ok()) {
            let g1 = c.get(1).unwrap();
            if g1.as_str().chars().filter(|c| py_isdigit(*c)).count() >= 4 {
                hits.push(("id".into(), char_index(&ln.text, g1.start()), char_index(&ln.text, g1.end()), c.get(0).unwrap().as_str().to_string()));
            }
        }
        for (kind, rx) in [("id", &*FREE_ID_RE), ("phone", &*FREE_PHONE_RE)] {
            for c in rx.captures_iter(&ln.text).filter_map(|c| c.ok()) {
                let g1 = c.get(1).unwrap();
                let digits: String = g1.as_str().chars().filter(|c| py_isdigit(*c)).collect();
                if kind == "id" && date_digits(&digits) {
                    continue;
                }
                hits.push((kind.into(), char_index(&ln.text, g1.start()), char_index(&ln.text, g1.end()), c.get(0).unwrap().as_str().to_string()));
            }
        }
        for (kind, a, b, text) in hits {
            let (x0, x1) = ln.span(a, b);
            let h = ln.height();
            if covered(l, (x0 + x1) / 2.0, ln.cy(), h, 0.5) {
                continue;
            }
            let mut z = Zone::new(&format!("{}-text@{:.0},{:.0}", kind, x0, ln.cy()), [x0 - 0.3 * h, ln.cy() - 0.7 * h, x1 - x0 + 0.6 * h, 1.4 * h], "partial");
            z.digits = if kind == "phone" { 10 } else { 8 };
            z.rule = kind.clone();
            z.search = Some([0.2 * h, 0.2 * h]);
            l.fields.push(Field { kind: kind.clone(), zones: vec![z], source: "free-text".into(), label: text.clone(), label_box: ln.bbox() });
            l.notes.push(format!("{} number written in free text ('{}') partially masked", kind, text));
        }
    }
}

// --- unlabelled signatures -----------------------------------------------------------------------

static REFUSAL_COMMENT: Lazy<Regex> = Lazy::new(|| re(r"(?i)refus|comment|presiding\s*officer"));
static CAPS_NAME: Lazy<Regex> = Lazy::new(|| re(r"[A-Z][A-Z.'-]{3,}"));
static NUMBER: Lazy<Regex> = Lazy::new(|| re(r"\d[\d\s/.-]{3,}\d"));
static STRONG_NUM: Lazy<Regex> = Lazy::new(|| re(r"\d{4,}|\d{1,2}\s*[-/.]\s*\d{1,2}\s*[-/.]\s*\d{2}"));
static STRONG_WORD: Lazy<Regex> = Lazy::new(|| re(r"(?i)sign|(?<![a-z])I\.?D(?![a-z])"));
static CAPS4: Lazy<Regex> = Lazy::new(|| re(r"[A-Z]{4,}"));

type BoxF = (f64, f64, f64, f64);

fn inside(cx: f64, cy: f64, boxes: &[BoxF]) -> bool {
    boxes.iter().any(|&(bx0, by0, bx1, by1)| bx0 - 4.0 <= cx && cx <= bx1 + 4.0 && by0 - 4.0 <= cy && cy <= by1 + 4.0)
}

fn near(b: BoxF, boxes: &[BoxF], dx: f64, dy: f64) -> bool {
    let (x0, y0, x1, y1) = b;
    boxes.iter().any(|&(bx0, by0, bx1, by1)| bx0 - dx <= x1 && x0 <= bx1 + dx && by0 - dy <= y1 && y0 <= by1 + dy)
}

/// On the same written line as one of `boxes`, within dx of it.
fn on_line(b: BoxF, boxes: &[BoxF], dx: f64) -> bool {
    let (x0, y0, x1, y1) = b;
    boxes.iter().any(|&(bx0, by0, bx1, by1)| bx0 - dx <= x1 && x0 <= bx1 + dx && y1.min(by1) - y0.max(by0) >= 0.5 * (y1 - y0).min(by1 - by0))
}

/// What each OCR pass reads in `box`: (text, per-character confidence).
fn readings(passes: &[Vec<&OLine>], b: BoxF) -> Vec<(String, Vec<f64>)> {
    let (x0, y0, x1, y1) = b;
    passes
        .iter()
        .map(|lines| {
            let mut cs: Vec<(char, f64)> = Vec::new();
            for ln in lines {
                let lb = ln.bbox();
                if y1.min(lb.3) - y0.max(lb.1) >= 0.5 * (y1 - y0).min(lb.3 - lb.1) {
                    cs.extend(ln.chars_in(x0, x1));
                }
            }
            (cs.iter().map(|c| c.0).collect(), cs.iter().map(|c| c.1).collect())
        })
        .collect()
}

static NONWORD: Lazy<regex::Regex> = Lazy::new(|| regex::Regex::new(r"\W").unwrap());

/// Is the writing read as text (confidently, or the same way twice)?
fn legible(reads: &[(String, Vec<f64>)], width: f64) -> bool {
    let need = (0.3 * width / TEXT_H).max(1.0);
    let mut texts = Vec::new();
    for (text, probs) in reads {
        if text.chars().any(|c| c as u32 > 0x2FF) {
            return false;
        }
        let t = NONWORD.replace_all(&text.to_uppercase(), "").to_string();
        if t.chars().count() as f64 >= need && util::np_mean(probs) >= 0.8 {
            return true;
        }
        texts.push(t);
    }
    (0..texts.len()).any(|i| {
        (i + 1..texts.len()).any(|j| {
            let (a, b) = (&texts[i], &texts[j]);
            let (la, lb) = (a.chars().count(), b.chars().count());
            la.min(lb) as f64 >= need && 1.0 - ocr::edits_str(a, b) as f64 / la.max(lb) as f64 >= 0.5
        })
    })
}

/// Read as joined-up lower case (letters whose capital looks different).
fn cursive(reads: &[(String, Vec<f64>)]) -> bool {
    let found: Vec<Vec<char>> = reads
        .iter()
        .map(|(t, _)| t.chars().filter(|c| c.is_alphabetic() && !c.to_lowercase().all(|l| "cosuvwxz".contains(l))).collect::<Vec<char>>())
        .filter(|f| !f.is_empty())
        .collect();
    !found.is_empty()
        && found.iter().map(|f| f.len()).max().unwrap() >= 3
        && found.iter().all(|f| f.iter().filter(|c| c.is_lowercase()).count() as f64 >= 0.6 * f.len() as f64)
}

/// A single straight stroke (a line struck through)?
fn straight(lab: &[i32], w: usize, parts: &[i32], h: f64) -> Result<bool> {
    let mut pts: Vector<Point2f> = Vector::new();
    for (k, &v) in lab.iter().enumerate() {
        if parts.contains(&v) {
            pts.push(Point2f::new((k % w) as f32, (k / w) as f32));
        }
    }
    if pts.len() < 10 {
        return Ok(false);
    }
    let r = opencv::geometry::min_area_rect(&pts)?;
    let (a, b) = (r.size.width as f64, r.size.height as f64);
    Ok(a.min(b) <= 0.25 * h && a.max(b) >= 2.0 * h)
}

/// Same name, allowing for OCR misreading it in both places.
fn name_match(word: &str, name: &str) -> bool {
    let (a, b) = (word.to_uppercase(), name.to_uppercase());
    let (la, lb) = (a.chars().count(), b.chars().count());
    if la.min(lb) < 4 {
        return false;
    }
    if la >= 6 && lb >= 6 && (b.contains(&a) || a.contains(&b)) {
        return true;
    }
    let m = la.min(lb);
    ocr::edits_str(&a, &b) <= if m < 5 { 1 } else if m < 8 { 2 } else { 3 }
}

/// Read as block capitals or digits with fair confidence: how comments, names
/// and dates are written on these forms; a signature is not.
fn in_capitals(reads: &[(String, Vec<f64>)], width: f64) -> bool {
    let need = (0.3 * width / TEXT_H).max(1.0);
    reads.iter().any(|(text, probs)| {
        let al: Vec<char> = text.chars().filter(|c| c.is_alphanumeric()).collect();
        let caps = al.iter().filter(|c| c.is_ascii_uppercase() || c.is_ascii_digit()).count();
        !al.is_empty() && al.len() as f64 >= need && caps as f64 >= 0.7 * al.len() as f64 && util::np_mean(probs) >= 0.5
    })
}

/// `strict` (the comments of a templated form, where sentences run close to
/// the officers' sign-off lines): only a date or a number makes a sign-off
/// line ("all agents signed" doesn't), writing read as capitals or digits
/// with fair confidence is text, and a signature needs a joined stroke wider
/// than a letter (OCR misses much of the handwriting, so "not read" says
/// little there); unread writing within a stroke's gap of it is part of it.
fn free_signatures(l: &mut Layout, top: f64, passes: &[usize], strict: bool) -> Result<()> {
    // Names are left alone where OCR read them; where it read a name's last
    // letters doubtfully, those may be a signature beside it (a first loop
    // read as a "U") or the name's own. Whichever reading finds writing to be
    // a signature, it is masked.
    let mut found = find_free_signatures(l, top, passes, strict, false)?;
    for f in find_free_signatures(l, top, passes, strict, true)? {
        let inter = |a: BoxF, b: BoxF| (a.2.min(b.2) - a.0.max(b.0)).max(0.0) * (a.3.min(b.3) - a.1.max(b.1)).max(0.0);
        let area = (f.0 .2 - f.0 .0) * (f.0 .3 - f.0 .1);
        if !found.iter().any(|g| inter(g.0, f.0) >= 0.9 * area) {
            found.push(f);
        }
    }
    let h = TEXT_H;
    for (bx, why) in found {
        let cy = (bx.1 + bx.3) / 2.0;
        let mut z = Zone::new(&format!("signature-text@{:.0},{:.0}", bx.0, cy), [bx.0 - 4.0, bx.1 - 4.0, bx.2 - bx.0 + 8.0, bx.3 - bx.1 + 8.0], "signature");
        z.search = Some([0.3 * h, 0.3 * h]);
        l.fields.push(Field { kind: "signature".into(), zones: vec![z], source: "free-text".into(), label: format!("unlabelled signature ({})", why), label_box: bx });
        l.notes.push(format!("unlabelled handwriting at y={:.0} taken for a signature ({})", cy, why));
    }
    Ok(())
}

/// The signatures `free_signatures` finds, with names' doubtful end letters
/// trimmed off them (`trim`) or not.
fn find_free_signatures(l: &Layout, top: f64, passes: &[usize], strict: bool, trim: bool) -> Result<Vec<(BoxF, &'static str)>> {
    let h = TEXT_H;
    let (_, hh_total) = l.frame.size;
    let y0 = top.max(0.0) as i32;
    if ((hh_total - y0) as f64) < h {
        return Ok(vec![]);
    }
    let sub = crop(&l.gray, 0, y0, l.gray.cols(), l.gray.rows())?;
    let mut ink = masking::ink_mask(&sub, h, true)?;
    let rls = util::morph(&ink, imgproc::MORPH_OPEN, &util::rect_kernel((3.0 * h) as i32, 1)?)?;
    ink = util::subtract(&ink, &util::dilate(&rls, &util::ones_kernel(3, 3)?)?)?;
    for st in &l.stamps {
        imgproc::circle(&mut ink, Point::new(st.cx as i32, (st.cy - y0 as f64) as i32), (st.r * 1.1 + 4.0) as i32, Scalar::all(0.0), -1, imgproc::LINE_8, 0)?;
    }
    let idx: Vec<usize> = (0..l.lines.len()).filter(|&i| l.lines[i].cy() >= top && !stamps::is_stamp_text(&l.lines[i].text)).collect();
    let lines: Vec<&OLine> = idx.iter().map(|&i| &l.lines[i]).collect();
    let passes: Vec<Vec<&OLine>> = (0..passes.len())
        .map(|k| {
            let (a, b) = (passes[k], passes.get(k + 1).copied().unwrap_or(usize::MAX));
            idx.iter().filter(|&&i| a <= i && i < b).map(|&i| &l.lines[i]).collect()
        })
        .collect();
    let (mut vetoed, mut strong, mut weak, mut worded, mut zones): (Vec<BoxF>, Vec<BoxF>, Vec<BoxF>, Vec<BoxF>, Vec<BoxF>) = Default::default();
    for ln in &lines {
        if has(&REFUSAL_COMMENT, &ln.text) || has(&PRINTED_LINE, &ln.text) {
            let pts: Vector<Point> = ln.quad.iter().map(|p| Point::new(round_even(p[0] as f64) as i32, round_even((p[1] - y0 as f32) as f64) as i32)).collect();
            let polys: Vector<Vector<Point>> = vec![pts].into_iter().collect();
            imgproc::fill_poly_def(&mut ink, &polys, Scalar::all(0.0))?;
            continue;
        }
        let mut spans = matches(&CAPS_NAME, &ln.text);
        spans.extend(matches(&NUMBER, &ln.text));
        let na = has(&NA_RE, &ln.text);
        if na {
            spans.push((0, ln.chars.len()));
        }
        let lb = ln.bbox();
        for (a, b) in spans {
            let probs: Vec<f64> = if ln.probs.len() == ln.chars.len() { ln.probs[a..b].to_vec() } else { vec![ln.score] };
            if util::np_mean(&probs) < 0.7 {
                continue;
            }
            // letters read doubtfully at its ends may be the writing beside
            // the name (a signature's first loop read as a "U")
            let (mut a, mut b) = (a, b);
            if trim && ln.probs.len() == ln.chars.len() {
                while b > a + 1 && ln.probs[b - 1] < 0.7 {
                    b -= 1;
                }
                while a + 1 < b && ln.probs[a] < 0.7 {
                    a += 1;
                }
            }
            let (x0, x1) = ln.span(a, b);
            vetoed.push((x0, lb.1, x1, lb.3));
        }
        if na {
            continue;
        }
        if has(&STRONG_NUM, &ln.text) || (!strict && has(&STRONG_WORD, &ln.text)) {
            strong.push(lb);
        } else if has(&CAPS4, &ln.text) {
            weak.push(lb);
        }
        if has(&CAPS4, &ln.text) {
            worded.push(lb);
        }
        let up = ln.text.to_uppercase();
        let words: Vec<String> = matches(&CAPS3, &up).iter().map(|&(a, b)| up.chars().skip(a).take(b - a).collect()).collect();
        if words.iter().any(|w| l.officer_names.iter().any(|n| name_match(w, n))) {
            zones.push((lb.0 - 2.0 * h, lb.1 - 0.5 * h, lb.2 + 10.0 * h, lb.3 + 3.5 * h));
            strong.push(lb);
        }
    }
    let stamped: Vec<BoxF> = l.stamps.iter().map(|c| (c.cx - c.r, c.cy - c.r, c.cx + c.r, c.cy + c.r)).collect();
    for &(bx0, by0, bx1, by1) in &vetoed {
        imgproc::rectangle_points(
            &mut ink,
            Point::new(bx0 as i32 - 2, (by0 - y0 as f64) as i32 - 2),
            Point::new(bx1.ceil() as i32 + 2, (by1 - y0 as f64).ceil() as i32 + 2),
            Scalar::all(0.0),
            -1,
            imgproc::LINE_8,
            0,
        )?;
    }
    let ink = util::morph(&ink, imgproc::MORPH_OPEN, &util::ones_kernel(2, 2)?)?;
    let (lab, st, _) = util::components(&ink)?;
    let labv = util::labels_vec(&lab)?;
    let groups = util::dilate(&ink, &util::rect_kernel((0.3 * h) as i32, 1)?)?;
    let (wlab, wst, _) = util::components(&groups)?;
    let wlabv = util::labels_vec(&wlab)?;
    let iw = ink.cols() as usize;
    let mut found: Vec<(BoxF, &str, bool)> = Vec::new();
    for k in 1..wst.len() {
        let [x, y, w, hh, _] = wst[k];
        if (w as f64) < 1.0 * h || (hh as f64) < 0.4 * h {
            continue;
        }
        let mut sub_lab = Vec::with_capacity((w * hh) as usize);
        let mut parts: BTreeSet<i32> = BTreeSet::new();
        for yy in y..y + hh {
            for xx in x..x + w {
                let p = yy as usize * iw + xx as usize;
                let lv = labv[p];
                sub_lab.push(lv);
                if wlabv[p] == k as i32 && lv != 0 && st[lv as usize][4] >= 20 {
                    parts.insert(lv);
                }
            }
        }
        let parts: Vec<i32> = parts.into_iter().collect();
        if parts.is_empty() {
            continue;
        }
        let bx: BoxF = (x as f64, y as f64 + y0 as f64, (x + w) as f64, (y + hh) as f64 + y0 as f64);
        let (cx, cy) = ((bx.0 + bx.2) / 2.0, (bx.1 + bx.3) / 2.0);
        if covered(l, cx, cy, h, 0.0) || (!strict && inside(cx, cy, &found.iter().map(|f| f.0).collect::<Vec<_>>())) {
            continue;
        }
        let reads = readings(&passes, bx);
        let signing = inside(cx, cy, &zones) || near(bx, &strong, 8.0 * h, 2.5 * h);
        let signed = signing && cursive(&reads) && !on_line(bx, &worded, 2.0 * h);
        if (legible(&reads, w as f64) || (strict && in_capitals(&reads, w as f64))) && !signed {
            continue;
        }
        let unread = !reads.iter().any(|(t, _)| !t.trim().is_empty());
        if straight(&sub_lab, w as usize, &parts, h)? {
            continue;
        }
        let few = parts.len() as f64 <= 0.9 * (w as f64 / h) + 1.0 && w as f64 >= 2.0 * h;
        let joined = few && parts.iter().map(|&j| st[j as usize][2]).max().unwrap() as f64 >= 0.6 * w as f64;
        let why = if inside(cx, cy, &zones) {
            "next to the officer's name, not legible as text"
        } else if on_line(bx, &strong, 8.0 * h) {
            "on a sign-off line, not legible as text"
        } else if signed || (signing && joined && unread) {
            "joined strokes next to a name, date or ID"
        } else if !strict && joined && unread && on_line(bx, &weak, 8.0 * h) {
            "joined strokes next to a word in capitals"
        } else if few && (joined || unread) && on_line(bx, &stamped, 3.0 * h) {
            "unreadable strokes beside the stamp"
        } else if strict {
            "" // may yet be part of a signature beside it
        } else {
            continue;
        };
        // in comments, a joined stroke wider than a letter: block capitals
        // written close together and digits OCR missed are not that, and a
        // run of letters with a flourish is a name
        let stroke = parts.iter().filter(|&&j| !straight(&sub_lab, w as usize, &[j], h).unwrap_or(true)).map(|&j| st[j as usize][2]).max().unwrap_or(0) as f64;
        let letters = parts.iter().filter(|&&j| (st[j as usize][2] as f64) <= 1.0 * h && (st[j as usize][3] as f64) <= 1.4 * h).count();
        found.push((bx, why, !strict || (!why.is_empty() && stroke >= 1.4 * h && letters < 8)));
    }
    if strict {
        // unread writing within a stroke's gap of a signature is part of it
        loop {
            let sure: Vec<BoxF> = found.iter().filter(|f| f.2).map(|f| f.0).collect();
            let mut more = false;
            for f in found.iter_mut().filter(|f| !f.2) {
                if near(f.0, &sure, 0.2 * h, 0.2 * h) {
                    *f = (f.0, "part of the signature beside it", true);
                    more = true;
                }
            }
            if !more {
                break;
            }
        }
    }
    Ok(found.into_iter().filter(|f| f.2).map(|f| (f.0, f.1)).collect())
}

// --- printed text inside fields --------------------------------------------------------------

fn x_fill(text: &str) -> bool {
    let t: Vec<char> = text.chars().filter(|c| !c.is_whitespace()).collect();
    t.len() >= 5 && t.iter().filter(|c| "xX×*".contains(**c)).count() as f64 >= 0.8 * t.len() as f64
}

/// What OCR shows to be print where a field could reach it: (mask, reference).
pub fn printed(l: &Layout) -> Result<(Mat, Mat)> {
    let mut mask = util::zeros_u8(l.gray.rows(), l.gray.cols())?;
    // nothing is printed on an officer's line between 'Signature' and
    // 'Date': a word read there was written ("No", for a signature)
    let written: Vec<[f64; 4]> = l.fields.iter().filter(|fl| fl.source == "declaration").map(|fl| fl.zones[0].rect).collect();
    for ln in &l.lines {
        let lb = ln.bbox();
        let (cx, cy) = (0.5 * (lb.0 + lb.2), ln.cy());
        if written.iter().any(|&[x, y, w, h]| cx > x && cx < x + w && cy > y && cy < y + h) {
            continue;
        }
        if x_fill(&ln.text) || has(&PRINTED_LINE, &ln.text) {
            // the characters read, not the whole box: OCR may box a label
            // with the signature written after it and read the label alone
            // ("Signature: " over 290 px, the label 130 of them), or with
            // the signature's first stroke for a stop or a bracket
            // ("signature:)")
            let ink: Vec<usize> = (0..ln.chars.len()).filter(|&i| ln.chars[i].is_alphanumeric() || ln.chars[i] == ':').collect();
            if let (Some(&a), Some(&b)) = (ink.first(), ink.last()) {
                let (x0, x1) = ln.span(a, b + 1);
                imgproc::rectangle_points(&mut mask, Point::new(x0 as i32 - 2, lb.1 as i32 - 2), Point::new(x1 as i32 + 2, lb.3 as i32 + 2), Scalar::all(255.0), -1, imgproc::LINE_8, 0)?;
            }
        }
        for (a, b) in matches(&FREE_LABEL, &ln.text) {
            let (x0, x1) = ln.span(a, b);
            imgproc::rectangle_points(&mut mask, Point::new(x0 as i32 - 2, lb.1 as i32 - 2), Point::new(x1 as i32 + 2, lb.3 as i32 + 2), Scalar::all(255.0), -1, imgproc::LINE_8, 0)?;
        }
    }
    let (m, g) = (util::bytes(&mask)?, util::bytes(&l.gray)?);
    let r: Vec<u8> = m.iter().zip(&g).map(|(&mv, &gv)| if mv > 0 { gv } else { 255 }).collect();
    Ok((mask, util::from_bytes(l.gray.rows(), l.gray.cols(), &r)?))
}

// --- whole page ------------------------------------------------------------------------------

/// OCR a horizontal band of the page at `zoom` x.
fn read_band(o: &mut Ocr, work: &Mat, y0: f64, y1: f64, zoom: f64) -> Result<Vec<OLine>> {
    let (a, b) = (y0.max(0.0) as i32, y1.min(work.rows() as f64) as i32);
    if ((b - a) as f64) < TEXT_H {
        return Ok(vec![]);
    }
    let band = util::resize_by(&crop(work, 0, a, work.cols(), b)?, zoom, imgproc::INTER_CUBIC)?;
    let band = to_bgr(&band)?;
    let pad = (740 - band.rows()).max(0);
    let mut padded = Mat::default();
    core::copy_make_border(&band, &mut padded, 0, pad, 0, 0, core::BORDER_CONSTANT, Scalar::new(255.0, 255.0, 255.0, 0.0))?;
    let mut lines = o.read(&padded, 0.3)?;
    let (z, af) = (zoom as f32, a as f32);
    for ln in lines.iter_mut() {
        for p in ln.quad.iter_mut() {
            p[0] = p[0] / z + 0.0;
            p[1] = p[1] / z + af;
        }
    }
    Ok(lines.into_iter().filter(|ln| ln.cy() < b as f64).collect())
}

pub enum Analysis {
    Layout(Box<Layout>),
    Fail(String),
}

pub fn analyze(img: &Mat, o: &mut Ocr) -> Result<Analysis> {
    let Some(f) = find_frame(img, o, 1200.0)? else {
        return Ok(Analysis::Fail("no printed text found to orient the page".into()));
    };
    let work = warp(img, &f)?;
    let gray = util::gray(&work)?;
    let lines = o.read(&to_bgr(&work)?, 0.3)?;
    let tl: Vec<stamps::TextLine> = lines.iter().map(|ln| {
        let b = ln.bbox();
        stamps::TextLine { text: ln.text.clone(), bbox: [b.0, b.1, b.2, b.3] }
    }).collect();
    let st = stamps::find(&work, STAMP_R, &tl, None)?;
    let mut l = Layout {
        frame: f.clone(),
        image: work.try_clone()?,
        gray,
        lines,
        fields: vec![],
        protected: vec![],
        stamps: st,
        problems: vec![],
        notes: vec![],
        officer_names: BTreeSet::new(),
    };
    let (hs, vs) = rules(&l.gray, &l.stamps)?;
    let (ww, hh) = (f.size.0 as f64, f.size.1 as f64);
    let decl: Vec<OLine> = l.lines.iter().filter(|ln| has(&DECL_RE, &ln.text)).cloned().collect();
    let counts: Vec<OLine> = l.lines.iter().filter(|ln| has(&COUNT_LABEL_RE, &ln.text)).cloned().collect();
    let continued = decl.is_empty() && counts.is_empty() && goes_on_with_declaration(&l.lines, hh);
    if decl.is_empty() && !continued {
        l.problems.push("can't tell where the results end (no 'Declaration' label or declaration text)".into());
        return Ok(Analysis::Layout(Box::new(l)));
    }
    let boundary = if continued {
        l.notes.push("no results on this page: it goes on with the declaration".into());
        0.0
    } else {
        decl.iter().map(|ln| ln.bbox().1).fold(f64::INFINITY, f64::min) - 0.3 * decl[0].height()
    };
    if !counts.is_empty() && counts.iter().map(|ln| ln.bbox().3).fold(f64::NEG_INFINITY, f64::max) > boundary {
        l.problems.push("count labels below the declaration: can't tell where the results end".into());
        return Ok(Analysis::Layout(Box::new(l)));
    }
    l.protected.push(("results".into(), vec![[0.0, 0.0], [ww, 0.0], [ww, boundary], [0.0, boundary]]));
    let mut used = tables(&mut l, &hs, &vs)?;
    let heads: Vec<f64> = l.fields.iter().filter(|fl| fl.source == "table").map(|fl| fl.label_box.1).collect();
    // the declaration runs down to the agents' table, or else to the next
    // section's heading; with neither, over the officers' own lines (a 2013
    // Form 34 stacks name, ID, signature and date one under another)
    let next = l.lines.iter().filter(|ln| ln.bbox().1 > boundary && has(&SECTION_RE, &ln.text)).map(|ln| ln.bbox().1).reduce(f64::min);
    let lines_on = boundary + 9.0 * TEXT_H;
    let decl_bottom = match heads.iter().cloned().filter(|&y| y > boundary).reduce(f64::min).or(next) {
        Some(y) => y,
        None => officer_lines_end(&l.lines, lines_on),
    } - 0.3 * TEXT_H;
    let extra = read_band(o, &work, boundary, decl_bottom, 2.5)?;
    l.lines.extend(extra);
    used.extend(declaration(&mut l, boundary, decl_bottom));
    labels(&mut l, &used, boundary);
    let bottoms: Vec<f64> = l.fields.iter().filter(|fl| fl.source == "table").flat_map(|fl| fl.zones.iter()).map(|z| z.rect[1] + z.rect[3]).collect();
    // free writing below the tables; with none found, below the officers'
    // lines, but no higher than nine lines down (a table there, unread, has
    // headings in print no one wrote)
    let free_top = bottoms.iter().cloned().reduce(f64::max).unwrap_or(decl_bottom.max(lines_on - 0.3 * TEXT_H));
    let band = read_band(o, &work, free_top, hh, 2.5)?;
    let band_from = l.lines.len();
    l.lines.extend(band);
    free_text(&mut l, &used, boundary);
    free_signatures(&mut l, free_top, &[0, band_from], false)?;
    let kinds: BTreeSet<String> = l.fields.iter().filter(|fl| fl.source == "table").map(|fl| fl.kind.clone()).collect();
    let all: BTreeSet<String> = ["id", "phone", "signature"].iter().map(|s| s.to_string()).collect();
    if !kinds.is_empty() && kinds != all {
        let found = kinds.iter().cloned().collect::<Vec<_>>().join(", ");
        let head = l.fields.iter().filter(|fl| fl.source == "table").map(|fl| (fl.label_box.1 + fl.label_box.3) / 2.0).fold(f64::INFINITY, f64::min);
        if headings_read(&l, &hs, &vs, head, &kinds) {
            l.notes.push(format!("agents table: {} columns only, every other heading read and none for an ID or phone number", found));
        } else {
            l.problems.push(format!("agents table: found only {} columns", found));
        }
    }
    if kinds.is_empty() && agents_section(&l.lines, boundary) {
        l.problems.push("agents' section found, but no table read under it".into());
    }
    if l.fields.is_empty() {
        l.problems.push("no PII fields recognised".into());
    }
    Ok(Analysis::Layout(Box::new(l)))
}

// --- redaction of a page without a template ---------------------------------------------------

fn fail(t0: Instant, prefix: &str, reason: String, report: Vec<Value>, notes: Vec<String>, layout: Option<Value>) -> PageResult {
    PageResult {
        ok: false,
        image: None,
        template: "generic".into(),
        inliers: 0,
        score: 0.0,
        reason: format!("{}{}", prefix, reason),
        ms: t0.elapsed().as_secs_f64() * 1000.0,
        masks: report,
        notes,
        blocks: None,
        protected: vec![],
        h: None,
        layout,
        unsure: vec![],
    }
}

/// Mask settings for a field read here: glyph sizes scaled to the text height.
pub fn field_settings(ms: &MaskSettings, fl: &Field) -> MaskSettings {
    let row_h = median(&fl.zones.iter().map(|z| z.rect[3]).collect::<Vec<_>>());
    let g = ms.min_glyph_h.min(0.3 * TEXT_H / row_h);
    let mut m = ms.clone();
    m.min_glyph_h = g;
    m.min_glyph_h_out = g;
    m
}

/// What was found in an area of free writing on a templated form.
pub struct FreeText {
    /// Working frame (the area, upright, printed text TEXT_H px high) -> image.
    pub to_image: [[f64; 3]; 3],
    /// The area, read: OCR lines, and the fields found in it.
    pub layout: Layout,
    /// Printed text OCR found there (mask, reference), as `printed`.
    pub print_mask: Mat,
    pub print_ref: Mat,
}

impl FreeText {
    pub fn mapper(&self) -> masking::Mapper {
        masking::homography(self.to_image)
    }

    pub fn record(&self, name: &str) -> Value {
        let f = self.mapper();
        let ti: Vec<Value> = self.to_image.iter().map(|r| json!(r.iter().map(|&x| pyjson::np_round(x, 8)).collect::<Vec<_>>())).collect();
        json!({"name": name, "to_image": ti, "fields": self.layout.fields.iter().map(|fl| fl.record(&f)).collect::<Vec<_>>()})
    }
}

/// Read an area of free writing on a templated form (the comments under the
/// agents table) as a form without a template is read below its tables
/// (steps 6 and 7): IDs and phone numbers written there, and signatures
/// that aren't legible as text. `h` maps template -> image; `rect` is the
/// area and `text_h` the reference's printed text height, both template px.
/// `known`: stamps found on the whole page (image px); one the area cuts
/// through can't be found in the area alone, and its half ring would pass
/// for joined strokes. `reference`: the template's reference image, whose
/// printed labels and rules are no one's signature.
pub fn read_free_text(img: &Mat, h: &[[f64; 3]; 3], rect: [f64; 4], text_h: f64, known: &[Stamp], reference: Option<&Mat>, o: &mut Ocr) -> Result<FreeText> {
    let s = TEXT_H / text_h;
    let [x, y, w, hh] = rect;
    // working <- template: the area, scaled so printed text is TEXT_H high
    let m = [[s, 0.0, -s * x], [0.0, s, -s * y], [0.0, 0.0, 1.0]];
    let to_image = util::mul3(h, &util::inv3(&m)?);
    // An area reaching past its reference (a learned layout's comments run
    // on below the page it was learned on) ends where the photo does: past
    // it the photo's last rows, repeated, would be read as writing.
    let mut hh = hh;
    if reference.is_some_and(|r| y + hh > r.rows() as f64) {
        let below = |yw: f64| [0.0, w * s].iter().map(|&xw| util::project(&to_image, [xw, yw])[1]).fold(f64::NEG_INFINITY, f64::max) - img.rows() as f64;
        if below(hh * s) > 0.0 && below(0.0) < 0.0 {
            let (mut lo, mut hi) = (0.0, hh * s);
            for _ in 0..30 {
                let mid = (lo + hi) / 2.0;
                if below(mid) > 0.0 {
                    hi = mid
                } else {
                    lo = mid
                }
            }
            hh = lo / s;
        }
    }
    let size = ((w * s).ceil() as i32, (hh * s).ceil() as i32);
    let mut work = Mat::default();
    imgproc::warp_perspective(
        img,
        &mut work,
        &util::mat3(&util::inv3(&to_image)?)?,
        Size::new(size.0, size.1),
        imgproc::INTER_LINEAR,
        core::BORDER_REPLICATE,
        Scalar::default(),
        core::AlgorithmHint::ALGO_HINT_DEFAULT,
    )?;
    let gray = util::gray(&work)?;
    let mut lines = o.read(&to_bgr(&work)?, 0.3)?;
    let band_from = lines.len();
    // read again at 2.5x: each reading finds things the other misses
    lines.extend(read_band(o, &work, 0.0, size.1 as f64, 2.5)?);
    let ink_from = lines.len();
    // and on the ink alone: the security print behind handwriting hides
    // words from the line finder
    let mut clean = Mat::default();
    core::bitwise_not(&masking::ink_mask(&gray, TEXT_H, false)?, &mut clean, &core::no_array())?;
    lines.extend(o.read(&to_bgr(&clean)?, 0.3)?);
    let tl: Vec<stamps::TextLine> = lines.iter().map(|ln| {
        let b = ln.bbox();
        stamps::TextLine { text: ln.text.clone(), bbox: [b.0, b.1, b.2, b.3] }
    }).collect();
    let mut st = stamps::find(&work, STAMP_R, &tl, None)?;
    let from_image = util::inv3(&to_image)?;
    for k in known {
        let c = util::project(&from_image, [k.cx, k.cy]);
        let e = util::project(&from_image, [k.cx + k.r, k.cy]);
        let r = (e[0] - c[0]).hypot(e[1] - c[1]);
        if c[0] > -r && c[1] > -r && c[0] < size.0 as f64 + r && c[1] < size.1 as f64 + r {
            st.push(Stamp::new(c[0], c[1], r, k.coverage, &k.how));
        }
    }
    let frame = Frame { k: 0, skew: 0.0, scale: s, m: from_image, size, text_h };
    let mut l = Layout {
        frame,
        image: work,
        gray,
        lines,
        fields: vec![],
        protected: vec![],
        stamps: st,
        problems: vec![],
        notes: vec![],
        officer_names: BTreeSet::new(),
    };
    free_text(&mut l, &BTreeSet::new(), 0.0);
    // signatures are looked for with the reference's print taken out: a
    // label OCR missed (blurred, half under a stamp) is not writing
    let gray = l.gray.try_clone()?;
    if let Some(r) = reference {
        let mut rw = Mat::default();
        let mm = util::mat3(&m)?;
        imgproc::warp_perspective(r, &mut rw, &mm, Size::new(size.0, size.1), imgproc::INTER_AREA, core::BORDER_CONSTANT, Scalar::all(255.0), core::AlgorithmHint::ALGO_HINT_DEFAULT)?;
        // printed letters only: rules and strike lines are taken out of the
        // ink anyway, and whitening them would cut signatures crossing them
        let ink = util::adaptive_inv(&util::gaussian(&rw, 3)?, 31, 25.0)?;
        let (lab, st, _) = util::components(&ink)?;
        let small: Vec<bool> = st.iter().map(|s| (s[2] as f64) <= 1.5 * TEXT_H && (s[3] as f64) <= 1.5 * TEXT_H).collect();
        let letters: Vec<u8> = util::labels_vec(&lab)?.iter().map(|&k| if k > 0 && small[k as usize] { 255 } else { 0 }).collect();
        let print = util::dilate(&util::from_bytes(size.1, size.0, &letters)?, &util::ones_kernel(7, 7)?)?;
        l.gray.set_to(&Scalar::all(255.0), &print)?;
    }
    free_signatures(&mut l, 0.0, &[0, band_from, ink_from], true)?;
    l.gray = gray;
    let (print_mask, print_ref) = printed(&l)?;
    Ok(FreeText { to_image, layout: l, print_mask, print_ref })
}

/// A page no template matched: read its layout, then mask and check exactly
/// as for a template.
/// `extra`: what a vision review found still showing on an earlier pass.
pub fn redact_generic(img: &Mat, ms: &MaskSettings, rng: &mut PyRandom, verify: bool, why: &str, extra: &[crate::pipeline::Found]) -> Result<PageResult> {
    let t0 = Instant::now();
    let gray = util::gray(img)?;
    let prefix = if why.is_empty() { String::new() } else { format!("no template matched ({}); ", why) };
    if !ocr::available() {
        let e = ocr::check().err().map(|e| e.to_string()).unwrap_or_default();
        return Ok(fail(t0, &prefix, e, vec![], vec![], None));
    }
    let analysis = ocr::with_engine(|o| analyze(img, o))?;
    let l = match analysis {
        Analysis::Fail(s) => return Ok(fail(t0, &prefix, s, vec![], vec![], None)),
        Analysis::Layout(l) => l,
    };
    let to_image = l.frame.to_image()?;
    let f = masking::homography(to_image);
    let ti: Vec<Value> = to_image.iter().map(|r| json!(r.iter().map(|&x| pyjson::np_round(x, 8)).collect::<Vec<_>>())).collect();
    let layout = json!({
        "frame": {"quarter_turns": l.frame.k, "skew_deg": pyjson::round_n(l.frame.skew, 2), "scale": pyjson::round_n(l.frame.scale, 5), "to_image": ti},
        "fields": l.fields.iter().map(|fl| fl.record(&f)).collect::<Vec<_>>(),
    });
    if !l.problems.is_empty() {
        return Ok(fail(t0, &prefix, l.problems.join("; "), vec![], l.notes.clone(), Some(layout)));
    }
    let field_settings = |fl: &Field| field_settings(ms, fl);
    let (print_mask, print_ref) = printed(&l)?;
    let mut notes = l.notes.clone();
    let rows: Vec<&Zone> = l.fields.iter().filter(|fl| fl.source == "table").flat_map(|fl| fl.zones.iter()).collect();
    let mut strikes = Vec::new();
    if !rows.is_empty() {
        let x0 = rows.iter().map(|z| z.rect[0]).fold(f64::INFINITY, f64::min);
        let y0 = rows.iter().map(|z| z.rect[1]).fold(f64::INFINITY, f64::min);
        let x1 = rows.iter().map(|z| z.rect[0] + z.rect[2]).fold(f64::NEG_INFINITY, f64::max);
        let y1 = rows.iter().map(|z| z.rect[1] + z.rect[3]).fold(f64::NEG_INFINITY, f64::max);
        strikes = masking::find_strikes(&gray, &f, [x0 - 10.0, y0 - 10.0, x1 - x0 + 20.0, y1 - y0 + 20.0], median(&rows.iter().map(|z| z.rect[3]).collect::<Vec<_>>()))?;
    }
    if !strikes.is_empty() {
        notes.push(format!("agents table: {} line(s) drawn through it left unmasked", strikes.len()));
    }
    let struck = |fl: &Field| -> Option<&[masking::Strike]> { if fl.source == "table" { Some(strikes.as_slice()) } else { None } };
    let mut canvas = util::full_u8(gray.rows(), gray.cols(), 255)?;
    let r_range = (STAMP_R.0 / l.frame.scale, STAMP_R.1 / l.frame.scale);
    let on_page: Vec<Stamp> = l
        .stamps
        .iter()
        .map(|st| {
            let p = f(&[[st.cx, st.cy]])[0];
            Stamp::new(p[0], p[1], st.r / l.frame.scale, st.coverage, &st.how)
        })
        .collect();
    let stamp_ink = if on_page.is_empty() { None } else { Some(stamps::protect_mask(gray.rows(), gray.cols(), &on_page, STAMP_MARGIN)?) };
    let mut report: Vec<Value> = Vec::new();
    for fl in &l.fields {
        if fl.kind == "signature" && fl.source == "free-text" {
            let [x, y, w, h] = fl.zones[0].rect;
            masking::fill(&mut canvas, &f, &[[x, y], [x + w, y], [x + w, y + h], [x, y + h]])?;
            report.push(json!({"zone": fl.zones[0].name, "mode": "signature", "signatures": 1}));
        } else if fl.kind == "signature" {
            let opts = SigOptions { reach: masking::Reach::default(), reference: None, printed: Some(&print_mask), centered: fl.source != "table", strikes: struck(fl), exclude: stamp_ink.as_ref(), hang: hang(&l, fl) };
            report.extend(masking::signature_column(&mut canvas, &gray, &f, &fl.zones, ms, &opts)?);
        } else {
            report.extend(masking::partial_column(&mut canvas, &gray, &f, &fl.zones, &field_settings(fl), rng, masking::Reach::default(), Some(&print_ref), struck(fl), stamp_ink.as_ref())?);
        }
    }
    let found_zones = crate::pipeline::mask_found(&mut canvas, &gray, extra, ms, rng, stamp_ink.as_ref(), &mut report)?;
    if ms.mask_stamps && !on_page.is_empty() {
        let mut bands = util::full_u8(gray.rows(), gray.cols(), 255)?;
        for z in l.fields.iter().flat_map(|fl| fl.zones.iter()) {
            masking::fill(&mut bands, &f, &masking::band(z))?;
        }
        let ident = masking::homography(util::IDENTITY);
        for z in &found_zones {
            masking::fill(&mut bands, &ident, &masking::band(z))?;
        }
        let n = crate::pipeline::hide_under_stamps(&mut canvas, &bands, &on_page)?;
        if n > 0 {
            notes.push(format!("fields' bands across stamps hidden whole: {} px", n));
        }
    }
    let (found, keep, stamp_notes) = protect_stamps(img, &mut canvas, &on_page, r_range, None, ms)?;
    notes.extend(stamp_notes);
    let (blocks, out) = apply(img, &canvas, Some(&keep))?;
    let mut protected: Vec<Value> = found.iter().map(|s| s.record()).collect();
    let mut prot_mask = util::zeros_u8(gray.rows(), gray.cols())?;
    for (name, poly) in &l.protected {
        let quad = f(poly);
        util::fill_poly(&mut prot_mask, &quad, 255.0)?;
        let pv: Vec<Value> = quad.iter().map(|p| json!([pyjson::np_round(p[0], 1), pyjson::np_round(p[1], 1)])).collect();
        protected.push(json!({"name": name, "polygon": pv}));
    }
    if blocks.and(&proof::to_blocks(&prot_mask)?).any() {
        return Ok(fail(t0, &prefix, "a mask would touch a protected results region".into(), report, notes, Some(layout)));
    }
    let mut leaks = Vec::new();
    if verify {
        for fl in &l.fields {
            if fl.kind == "signature" && fl.source == "free-text" {
                continue; // masked whole, box and all
            }
            let r = if fl.kind == "signature" {
                let opts = SigOptions { reach: masking::Reach::default(), reference: None, printed: Some(&print_mask), centered: fl.source != "table", strikes: struck(fl), exclude: stamp_ink.as_ref(), hang: hang(&l, fl) };
                masking::verify_signatures(&gray, &out, &f, &fl.zones, ms, Some(&keep), &opts)?
            } else {
                masking::verify_column(&out, &f, &fl.zones, Some(&print_ref), &field_settings(fl), masking::Reach::default(), struck(fl), stamp_ink.as_ref())?
            };
            if !r.is_empty() {
                leaks.push(r);
            }
        }
        leaks.extend(crate::pipeline::verify_found(&out, &found_zones, ms, stamp_ink.as_ref())?);
    }
    if !leaks.is_empty() {
        return Ok(fail(t0, &prefix, format!("leak check: {}", leaks.join("; ")), report, notes, Some(layout)));
    }
    Ok(PageResult {
        ok: true,
        image: Some(out),
        template: "generic".into(),
        inliers: 0,
        score: 0.0,
        reason: String::new(),
        ms: t0.elapsed().as_secs_f64() * 1000.0,
        masks: report,
        notes,
        blocks: Some(blocks),
        protected,
        h: Some(to_image),
        layout: Some(layout),
        unsure: vec![],
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn regexes_as_python() {
        assert!(has(&ID_RE, "ID/Passport No."));
        assert!(has(&ID_RE, "I.D No"));
        assert!(!has(&TEL_RE, "Hotel"));
        assert!(has(&TEL_RE, "Tel. Contact"));
        assert!(has(&DATE_RE, "Date"));
        assert!(!has(&DATE_RE, "Candidate"));
        assert!(has(&PRINTED_LINE, " Name of Candidate or Agent "));
        assert_eq!(date_ish("signed 18-03-2021"), Some(7));
        assert!(date_digits("15122020"));
        assert!(!date_digits("25039403"));
        assert_eq!(py_sum(&[0.1, 0.2, 0.3]), 0.6);
        assert!(name_match("KAHENYA", "KAHENYAH"));
        assert!(x_fill("XXXX XXXX"));
    }
}

#[cfg(test)]
#[path = "tests/generic.rs"]
mod py_tests;
