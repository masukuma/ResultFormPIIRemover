//! Local OCR of printed labels: the PP-OCR models (text detection, line
//! orientation, recognition) of RapidOCR, run on ONNX Runtime.
//!
//! Used to find the printed labels of forms there is no template for. The
//! models and ONNX Runtime are local files; nothing leaves the machine. Their
//! hashes go into every manifest made with them, so a re-run can be checked
//! against the same models.
//!
//! The pre- and post-processing are those of `rapidocr_onnxruntime` 1.2.3,
//! step for step, so that the same lines are read. Recognition is decoded
//! here rather than by the library so that each character comes with its
//! position along the text line: a label is often read together with
//! handwriting next to it ("Presiding Officer JOHN DOE Signature"), and the
//! field starts where the word "Signature" ends, not where the line does.
//!
//! Configuration: `RFPII_ORT_LIB` (the ONNX Runtime shared library, default
//! /opt/onnxruntime/lib/libonnxruntime.so.1.29.0) and `RFPII_OCR_MODELS` (the
//! directory holding the three models, default /opt/rfpii/models).

use anyhow::{anyhow, bail, Context, Result};
use once_cell::sync::OnceCell;
use opencv::core::{self, Mat, Point, Point2f, Scalar, Size, Vector};
use opencv::imgproc;
use opencv::prelude::*;
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::Tensor;
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};
use std::cell::RefCell;
use std::path::PathBuf;

use crate::util::{self, round_even};

pub const DET_MODEL: &str = "ch_PP-OCRv3_det_infer.onnx";
pub const CLS_MODEL: &str = "ch_ppocr_mobile_v2.0_cls_infer.onnx";
pub const REC_MODEL: &str = "ch_PP-OCRv3_rec_infer.onnx";

fn ort_lib() -> PathBuf {
    std::env::var_os("RFPII_ORT_LIB").map(PathBuf::from).unwrap_or_else(|| PathBuf::from("/opt/onnxruntime/lib/libonnxruntime.so.1.29.0"))
}

pub fn models_dir() -> PathBuf {
    std::env::var_os("RFPII_OCR_MODELS").map(PathBuf::from).unwrap_or_else(|| PathBuf::from("/opt/rfpii/models"))
}

/// Can pages be read without a template (runtime and models present)?
pub fn available() -> bool {
    ort_lib().exists() && [DET_MODEL, CLS_MODEL, REC_MODEL].iter().all(|m| models_dir().join(m).exists())
}

/// Engine and model hashes, for the manifest.
pub fn info() -> Result<Value> {
    let mut models = Map::new();
    let mut names = vec![DET_MODEL, CLS_MODEL, REC_MODEL];
    names.sort();
    for n in names {
        let data = std::fs::read(models_dir().join(n)).with_context(|| format!("reading OCR model {}", n))?;
        models.insert(n.to_string(), json!(hex::encode(Sha256::digest(&data))));
    }
    let lib = ort_lib();
    let runtime = lib.file_name().map(|f| f.to_string_lossy().into_owned()).unwrap_or_default();
    Ok(json!({"engine": "PP-OCR (RapidOCR 1.2.3 pipeline) on ONNX Runtime", "runtime": runtime, "models": models}))
}

// --- text lines -------------------------------------------------------------------------------

pub type Quad = [[f32; 2]; 4]; // top-left, top-right, bottom-right, bottom-left (image px)

fn norm32(a: [f32; 2], b: [f32; 2]) -> f32 {
    let (dx, dy) = (a[0] - b[0], a[1] - b[1]);
    (dx * dx + dy * dy).sqrt()
}

#[derive(Clone, Debug)]
pub struct Line {
    pub quad: Quad,
    pub text: String,
    pub chars: Vec<char>,
    pub score: f64,
    pub pos: Vec<f64>,   // per character: centre as a share of the line's width
    pub probs: Vec<f64>, // per character: recognition confidence
}

/// Character index of a byte offset into `s`.
pub fn char_index(s: &str, byte: usize) -> usize {
    s[..byte].chars().count()
}

impl Line {
    pub fn new(quad: Quad, text: String, score: f64, pos: Vec<f64>, probs: Vec<f64>) -> Self {
        let chars = text.chars().collect();
        Line { quad, text, chars, score, pos, probs }
    }

    pub fn bbox(&self) -> (f64, f64, f64, f64) {
        let q = &self.quad;
        let f = |k: usize, m: fn(f32, f32) -> f32| q.iter().map(|p| p[k]).reduce(m).unwrap() as f64;
        (f(0, f32::min), f(1, f32::min), f(0, f32::max), f(1, f32::max))
    }

    pub fn height(&self) -> f64 {
        (norm32(self.quad[3], self.quad[0]) + norm32(self.quad[2], self.quad[1])) as f64 / 2.0
    }

    /// `float(quad[:, 1].mean())`, in float32 as numpy does.
    pub fn cy(&self) -> f64 {
        let mut s = 0f32;
        for p in &self.quad {
            s += p[1];
        }
        (s / 4.0) as f64
    }

    /// Image x of a point `share` of the way along the line (float32, as numpy).
    pub fn x_at(&self, share: f64) -> f64 {
        let s = share as f32;
        let q = &self.quad;
        let top = q[0][0] + (q[1][0] - q[0][0]) * s;
        let bot = q[3][0] + (q[2][0] - q[3][0]) * s;
        ((top + bot) / 2.0) as f64
    }

    /// (character, confidence) of the characters centred between image x0 and x1.
    pub fn chars_in(&self, x0: f64, x1: f64) -> Vec<(char, f64)> {
        let n = self.chars.len();
        if self.pos.is_empty() || self.pos.len() != n {
            return vec![];
        }
        (0..n)
            .filter(|&i| {
                let x = self.x_at(self.pos[i]);
                x0 <= x && x <= x1
            })
            .map(|i| (self.chars[i], if self.probs.is_empty() { self.score } else { self.probs[i] }))
            .collect()
    }

    /// Image x-range of the first match of `re` in the text.
    pub fn find(&self, re: &fancy_regex::Regex) -> Option<(f64, f64)> {
        let m = re.find(&self.text).ok().flatten()?;
        Some(self.span(char_index(&self.text, m.start()), char_index(&self.text, m.end())))
    }

    /// Non-overlapping approximate occurrences of `word` (case-insensitive):
    /// (start, end, edits), best first.
    pub fn fuzzy(&self, word: &str, max_err: usize) -> Vec<(usize, usize, usize)> {
        let t: Vec<char> = self.text.chars().flat_map(|c| c.to_lowercase()).collect();
        let w: Vec<char> = word.chars().flat_map(|c| c.to_lowercase()).collect();
        let mut found: Vec<(usize, isize, usize, usize)> = Vec::new();
        for i in 0..t.len() {
            for l in (w.len().saturating_sub(max_err)).max(1)..=w.len() + max_err {
                if i + l > t.len() {
                    break;
                }
                let mut d = edits(&t[i..i + l], &w);
                if i + l == t.len() && l < w.len() {
                    d = d.min(edits(&t[i..], &w[..l]) + if l + 2 < w.len() { 1 } else { 0 });
                }
                if d <= max_err {
                    found.push((d, -(l as isize), i, i + l));
                }
            }
        }
        found.sort();
        let mut out: Vec<(usize, usize, usize)> = Vec::new();
        for (d, _, i, j) in found {
            if out.iter().all(|&(a, b, _)| j <= a || i >= b) {
                out.push((i, j, d));
            }
        }
        out
    }

    /// Image x-range of characters i..j-1.
    pub fn span(&self, i: usize, j: usize) -> (f64, f64) {
        let n = self.chars.len();
        if self.pos.is_empty() || self.pos.len() != n {
            let d = n.max(1) as f64;
            return (self.x_at(i as f64 / d), self.x_at(j as f64 / d));
        }
        let mut half = 0.5 / n.max(1) as f64;
        if j > i + 1 {
            half = (self.pos[j - 1] - self.pos[i]) / (2 * (j - i - 1)) as f64;
        }
        (self.x_at((self.pos[i] - half).max(0.0)), self.x_at((self.pos[j - 1] + half).min(1.0)))
    }
}

/// Levenshtein distance.
pub fn edits(a: &[char], b: &[char]) -> usize {
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    for (i, ca) in a.iter().enumerate() {
        let mut cur = vec![i + 1];
        for (j, cb) in b.iter().enumerate() {
            cur.push((prev[j + 1] + 1).min(cur[j] + 1).min(prev[j] + (ca != cb) as usize));
        }
        prev = cur;
    }
    prev[b.len()]
}

pub fn edits_str(a: &str, b: &str) -> usize {
    edits(&a.chars().collect::<Vec<_>>(), &b.chars().collect::<Vec<_>>())
}

// --- polygon offsetting (Clipper 6.4.2, as pyclipper) ------------------------------------------

fn clipper_round(v: f64) -> i64 {
    if v < 0.0 {
        (v - 0.5) as i64
    } else {
        (v + 0.5) as i64
    }
}

/// `pyclipper.PyclipperOffset` of one closed polygon with round joins
/// (arc tolerance 0.25), before Clipper's final union: for the convex boxes
/// this is used on, the union changes nothing a minimum-area rectangle sees.
pub fn offset_round(path: &[[f64; 2]], delta: f64) -> Vec<[i64; 2]> {
    // points are truncated to integers, duplicates dropped
    let mut pts: Vec<[i64; 2]> = Vec::new();
    let mut high = path.len();
    let ip: Vec<[i64; 2]> = path.iter().map(|p| [p[0] as i64, p[1] as i64]).collect();
    while high > 1 && ip[0] == ip[high - 1] {
        high -= 1;
    }
    pts.push(ip[0]);
    for p in &ip[1..high] {
        if *pts.last().unwrap() != *p {
            pts.push(*p);
        }
    }
    if pts.len() < 3 {
        return vec![];
    }
    // orientation: the offset grows outwards for a positive-area polygon
    let area = {
        let n = pts.len();
        let mut a = 0.0;
        let mut j = n - 1;
        for i in 0..n {
            a += (pts[j][0] as f64 + pts[i][0] as f64) * (pts[j][1] as f64 - pts[i][1] as f64);
            j = i;
        }
        -a * 0.5
    };
    if area < 0.0 {
        pts.reverse();
    }
    let len = pts.len();
    let arc_tol = 0.25;
    let y = if arc_tol > delta.abs() * 0.25 { delta.abs() * 0.25 } else { arc_tol };
    let mut steps = std::f64::consts::PI / (1.0 - y / delta.abs()).acos();
    if steps > delta.abs() * std::f64::consts::PI {
        steps = delta.abs() * std::f64::consts::PI;
    }
    let two_pi = std::f64::consts::PI * 2.0;
    let (mut sin, cos) = ((two_pi / steps).sin(), (two_pi / steps).cos());
    let steps_per_rad = steps / two_pi;
    if delta < 0.0 {
        sin = -sin;
    }
    let normal = |a: [i64; 2], b: [i64; 2]| -> [f64; 2] {
        if a == b {
            return [0.0, 0.0];
        }
        let (mut dx, mut dy) = ((b[0] - a[0]) as f64, (b[1] - a[1]) as f64);
        let f = 1.0 / (dx * dx + dy * dy).sqrt();
        dx *= f;
        dy *= f;
        [dy, -dx]
    };
    let mut normals: Vec<[f64; 2]> = (0..len - 1).map(|j| normal(pts[j], pts[j + 1])).collect();
    normals.push(normal(pts[len - 1], pts[0]));
    let mut out: Vec<[i64; 2]> = Vec::new();
    let mut k = len - 1;
    for j in 0..len {
        let (nk, nj, p) = (normals[k], normals[j], pts[j]);
        let mut sin_a = nk[0] * nj[1] - nj[0] * nk[1];
        let at = |n: [f64; 2]| [clipper_round(p[0] as f64 + n[0] * delta), clipper_round(p[1] as f64 + n[1] * delta)];
        let mut done = false;
        if (sin_a * delta).abs() < 1.0 {
            let cos_a = nk[0] * nj[0] + nj[1] * nk[1];
            if cos_a > 0.0 {
                out.push(at(nk));
                done = true;
            }
        } else if sin_a > 1.0 {
            sin_a = 1.0;
        } else if sin_a < -1.0 {
            sin_a = -1.0;
        }
        if !done {
            if sin_a * delta < 0.0 {
                out.push(at(nk));
                out.push(p);
                out.push(at(nj));
            } else {
                let a = sin_a.atan2(nk[0] * nj[0] + nk[1] * nj[1]);
                let n = (clipper_round(steps_per_rad * a.abs()) as i64).max(1);
                let (mut x, mut yy) = (nk[0], nk[1]);
                for _ in 0..n {
                    out.push([clipper_round(p[0] as f64 + x * delta), clipper_round(p[1] as f64 + yy * delta)]);
                    let x2 = x;
                    x = x * cos - sin * yy;
                    yy = x2 * sin + yy * cos;
                }
                out.push(at(nj));
            }
        }
        k = j;
    }
    out
}

/// Shapely's area and perimeter of a closed polygon (GEOS formulas).
fn area_length(p: &[[f64; 2]]) -> (f64, f64) {
    let mut ring = p.to_vec();
    ring.push(p[0]);
    let n = ring.len();
    let x0 = ring[0][0];
    let mut sum = 0.0;
    for i in 1..n - 1 {
        let x = ring[i][0] - x0;
        sum += x * (ring[i - 1][1] - ring[i + 1][1]);
    }
    let area = (sum / 2.0).abs();
    let mut len = 0.0;
    for i in 1..n {
        let (dx, dy) = (ring[i][0] - ring[i - 1][0], ring[i][1] - ring[i - 1][1]);
        len += (dx * dx + dy * dy).sqrt();
    }
    (area, len)
}

// --- the engine --------------------------------------------------------------------------------

static ENV: OnceCell<ort::environment::Environment> = OnceCell::new();

fn environment() -> Result<&'static ort::environment::Environment> {
    ENV.get_or_try_init(|| {
        let b = ort::init_from(ort_lib()).map_err(|e| anyhow!("loading ONNX Runtime from {}: {e}", ort_lib().display()))?;
        b.build().map_err(|e| anyhow!("ONNX Runtime: {e}"))
    })
}

pub struct Ocr {
    det: Session,
    cls: Session,
    rec: Session,
    chars: Vec<String>, // index 0 = CTC blank
}

thread_local! {
    static ENGINE: RefCell<Option<Ocr>> = const { RefCell::new(None) };
}

/// Intra-op threads per session: 0 = ONNX Runtime's default.
static THREADS: OnceCell<usize> = OnceCell::new();

pub fn set_threads(n: usize) {
    let _ = THREADS.set(n);
}

/// Run `f` with this thread's engine (created on first use).
pub fn with_engine<T>(f: impl FnOnce(&mut Ocr) -> Result<T>) -> Result<T> {
    ENGINE.with(|e| {
        let mut e = e.borrow_mut();
        if e.is_none() {
            *e = Some(Ocr::new()?);
        }
        f(e.as_mut().unwrap())
    })
}

fn session(name: &str) -> Result<Session> {
    let env = environment()?;
    let mut b = Session::builder(env).map_err(|e| anyhow!("{e}"))?;
    b = b.with_optimization_level(GraphOptimizationLevel::All).map_err(|e| anyhow!("{e}"))?;
    let t = *THREADS.get().unwrap_or(&0);
    if t > 0 {
        b = b.with_intra_threads(t).map_err(|e| anyhow!("{e}"))?;
    }
    b.commit_from_file(models_dir().join(name)).map_err(|e| anyhow!("loading OCR model {}: {e}", name))
}

fn tensor(shape: [usize; 4], data: Vec<f32>) -> Result<Tensor<f32>> {
    Tensor::from_array((shape, data)).map_err(|e| anyhow!("{e}"))
}

/// BGR u8 image -> CHW float32 with per-pixel `f`.
fn chw(img: &Mat, f: impl Fn(usize, u8) -> f32) -> Result<(usize, usize, Vec<f32>)> {
    let (h, w) = (img.rows() as usize, img.cols() as usize);
    let d = util::bytes(img)?;
    let mut out = vec![0f32; 3 * h * w];
    for y in 0..h {
        for x in 0..w {
            for c in 0..3 {
                out[c * h * w + y * w + x] = f(c, d[(y * w + x) * 3 + c]);
            }
        }
    }
    Ok((h, w, out))
}

fn bgr(img: &Mat) -> Result<Mat> {
    if img.channels() == 3 {
        return Ok(img.try_clone()?);
    }
    let mut out = Mat::default();
    imgproc::cvt_color_def(img, &mut out, imgproc::COLOR_GRAY2BGR)?;
    Ok(out)
}

/// `get_mini_boxes`: the minimum-area rectangle's corners (TL, TR, BR, BL as
/// RapidOCR orders them) and its shorter side.
fn mini_box(points: &Vector<Point>) -> Result<([[f32; 2]; 4], f32)> {
    let r = opencv::geometry::min_area_rect(points)?;
    let mut bp = Mat::default();
    opencv::geometry::box_points(r, &mut bp)?;
    let mut pts: Vec<[f32; 2]> = (0..4).map(|i| Ok([*bp.at_2d::<f32>(i, 0)?, *bp.at_2d::<f32>(i, 1)?])).collect::<Result<_>>()?;
    pts.sort_by(|a, b| a[0].partial_cmp(&b[0]).unwrap());
    let (i1, i4) = if pts[1][1] > pts[0][1] { (0, 1) } else { (1, 0) };
    let (i2, i3) = if pts[3][1] > pts[2][1] { (2, 3) } else { (3, 2) };
    Ok(([pts[i1], pts[i2], pts[i3], pts[i4]], r.size.width.min(r.size.height)))
}

impl Ocr {
    pub fn new() -> Result<Self> {
        let det = session(DET_MODEL)?;
        let cls = session(CLS_MODEL)?;
        let rec = session(REC_MODEL)?;
        let meta = rec.metadata().map_err(|e| anyhow!("{e}"))?;
        let list = meta.custom("character").ok_or_else(|| anyhow!("recognition model has no character list"))?;
        let mut chars = vec!["blank".to_string()];
        chars.extend(list.lines().map(String::from));
        chars.push(" ".into());
        drop(meta);
        Ok(Ocr { det, cls, rec, chars })
    }

    /// Text-line quads (RapidOCR's TextDetector), float32 image px.
    pub fn detect(&mut self, img: &Mat) -> Result<Vec<Quad>> {
        let img = bgr(img)?;
        let (h, w) = (img.rows() as f64, img.cols() as f64);
        // DetResizeForTest: limit_side_len 736, limit_type min
        let ratio = if h.min(w) < 736.0 { if h < w { 736.0 / h } else { 736.0 / w } } else { 1.0 };
        let (rh, rw) = ((h * ratio) as i64, (w * ratio) as i64);
        let (rh, rw) = (round_even(rh as f64 / 32.0) as i64 * 32, round_even(rw as f64 / 32.0) as i64 * 32);
        if rh <= 0 || rw <= 0 {
            return Ok(vec![]);
        }
        let resized = util::resize(&img, rw as i32, rh as i32, imgproc::INTER_LINEAR)?;
        let scale = (1.0f64 / 255.0) as f32;
        let mean = [0.485f64 as f32, 0.456f64 as f32, 0.406f64 as f32];
        let std = [0.229f64 as f32, 0.224f64 as f32, 0.225f64 as f32];
        let (ih, iw, data) = chw(&resized, |c, v| (v as f32 * scale - mean[c]) / std[c])?;
        let out = self.det.run(ort::inputs![tensor([1, 3, ih, iw], data)?]).map_err(|e| anyhow!("{e}"))?;
        let (shape, pred) = out[0].try_extract_tensor::<f32>().map_err(|e| anyhow!("{e}"))?;
        let (ph, pw) = (shape[2] as usize, shape[3] as usize);
        let pred = pred.to_vec();
        drop(out);
        // DBPostProcess: thresh 0.3, box_thresh 0.5, unclip 1.6, dilation, fast score
        let seg: Vec<u8> = pred.iter().map(|&v| (v > 0.3f32) as u8).collect();
        let mask = util::dilate(&util::from_bytes(ph as i32, pw as i32, &seg)?, &util::ones_kernel(2, 2)?)?;
        let m255: Vec<u8> = util::bytes(&mask)?.iter().map(|&v| v.wrapping_mul(255)).collect();
        let mut contours = Vector::<Vector<Point>>::new();
        imgproc::find_contours_def(&util::from_bytes(ph as i32, pw as i32, &m255)?, &mut contours, imgproc::RETR_LIST, imgproc::CHAIN_APPROX_SIMPLE)?;
        let mut pmat = Mat::new_rows_cols_with_default(ph as i32, pw as i32, core::CV_32F, Scalar::all(0.0))?;
        pmat.data_typed_mut::<f32>()?.copy_from_slice(&pred);
        let mut boxes: Vec<[[i16; 2]; 4]> = Vec::new();
        for contour in contours.iter().take(1000) {
            let (pts, sside) = mini_box(&contour)?;
            if sside < 3.0 {
                continue;
            }
            // box_score_fast
            let xs = pts.map(|p| p[0]);
            let ys = pts.map(|p| p[1]);
            let fmin = |v: [f32; 4]| v.iter().cloned().fold(f32::INFINITY, f32::min);
            let fmax = |v: [f32; 4]| v.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let xmin = (fmin(xs).floor() as i32).clamp(0, pw as i32 - 1);
            let xmax = (fmax(xs).ceil() as i32).clamp(0, pw as i32 - 1);
            let ymin = (fmin(ys).floor() as i32).clamp(0, ph as i32 - 1);
            let ymax = (fmax(ys).ceil() as i32).clamp(0, ph as i32 - 1);
            let mut bm = util::zeros_u8(ymax - ymin + 1, xmax - xmin + 1)?;
            let poly: Vector<Point> = pts.iter().map(|p| Point::new((p[0] - xmin as f32) as i32, (p[1] - ymin as f32) as i32)).collect();
            let polys: Vector<Vector<Point>> = vec![poly].into_iter().collect();
            imgproc::fill_poly_def(&mut bm, &polys, Scalar::all(1.0))?;
            let roi = Mat::roi(&pmat, core::Rect::new(xmin, ymin, xmax - xmin + 1, ymax - ymin + 1))?;
            let score = core::mean(&roi, &bm)?[0];
            if 0.5 > score {
                continue;
            }
            // unclip
            let p64: Vec<[f64; 2]> = pts.iter().map(|p| [p[0] as f64, p[1] as f64]).collect();
            let (area, length) = area_length(&p64);
            let expanded = offset_round(&p64, area * 1.6 / length);
            if expanded.is_empty() {
                continue;
            }
            let ev: Vector<Point> = expanded.iter().map(|p| Point::new(p[0] as i32, p[1] as i32)).collect();
            let (bx, sside) = mini_box(&ev)?;
            if sside < 5.0 {
                continue;
            }
            let mut b = [[0i16; 2]; 4];
            for (k, p) in bx.iter().enumerate() {
                let x = (round_even((p[0] / pw as f32) as f64 * w)).clamp(0.0, w) as f32;
                let y = (round_even((p[1] / ph as f32) as f64 * h)).clamp(0.0, h) as f32;
                b[k] = [x as i16, y as i16];
            }
            boxes.push(b);
        }
        // filter_tag_det_res
        let mut out = Vec::new();
        for b in boxes {
            let mut idx = [0usize, 1, 2, 3];
            idx.sort_by_key(|&i| b[i][0]);
            let mut left = [b[idx[0]], b[idx[1]]];
            let mut right = [b[idx[2]], b[idx[3]]];
            if left[1][1] < left[0][1] {
                left.swap(0, 1);
            }
            if right[1][1] < right[0][1] {
                right.swap(0, 1);
            }
            let rect = [left[0], right[0], right[1], left[1]];
            let q: Quad = rect.map(|p| [(p[0] as f32).max(0.0).min(w as f32 - 1.0).trunc(), (p[1] as f32).max(0.0).min(h as f32 - 1.0).trunc()]);
            if norm32(q[0], q[1]) as i64 <= 3 || norm32(q[0], q[3]) as i64 <= 3 {
                continue;
            }
            out.push(q);
        }
        Ok(out)
    }

    /// RapidOCR's `get_crop_img_list`: each quad straightened (cubic), turned
    /// upright when much taller than wide.
    pub fn crops(img: &Mat, quads: &[Quad]) -> Result<Vec<Mat>> {
        let mut out = Vec::new();
        for q in quads {
            let cw = norm32(q[0], q[1]).max(norm32(q[2], q[3])) as i32;
            let ch = norm32(q[0], q[3]).max(norm32(q[1], q[2])) as i32;
            let src: Vector<Point2f> = q.iter().map(|p| Point2f::new(p[0], p[1])).collect();
            let dst: Vector<Point2f> = [[0.0, 0.0], [cw as f32, 0.0], [cw as f32, ch as f32], [0.0, ch as f32]].iter().map(|p| Point2f::new(p[0], p[1])).collect();
            let m = opencv::geometry::get_perspective_transform(&src, &dst, core::DECOMP_LU)?;
            let mut crop = Mat::default();
            imgproc::warp_perspective(img, &mut crop, &m, Size::new(cw, ch), imgproc::INTER_CUBIC, core::BORDER_REPLICATE, Scalar::default(), core::AlgorithmHint::ALGO_HINT_DEFAULT)?;
            if crop.rows() as f64 / crop.cols() as f64 >= 1.5 {
                let mut r = Mat::default();
                core::rotate(&crop, &mut r, core::ROTATE_90_COUNTERCLOCKWISE)?;
                crop = r;
            }
            out.push(crop);
        }
        Ok(out)
    }

    /// Text-line orientation ("0" / "180", confidence) per quad.
    pub fn classify(&mut self, img: &Mat, quads: &[Quad]) -> Result<Vec<(String, f64)>> {
        let img = bgr(img)?;
        let crops = Self::crops(&img, quads)?;
        let mut res = vec![(String::new(), 0.0); crops.len()];
        let ratios: Vec<f64> = crops.iter().map(|c| c.cols() as f64 / c.rows() as f64).collect();
        let mut order: Vec<usize> = (0..crops.len()).collect();
        order.sort_by(|&a, &b| ratios[a].partial_cmp(&ratios[b]).unwrap());
        for batch in order.chunks(6) {
            let (ch, cwid) = (48usize, 192usize);
            let mut data = vec![0f32; batch.len() * 3 * ch * cwid];
            for (k, &i) in batch.iter().enumerate() {
                let (h, w) = (crops[i].rows() as f64, crops[i].cols() as f64);
                let ratio = w / h;
                let rw = if (ch as f64 * ratio).ceil() > cwid as f64 { cwid } else { (ch as f64 * ratio).ceil() as usize };
                let r = util::resize(&crops[i], rw as i32, ch as i32, imgproc::INTER_LINEAR)?;
                let (_, _, v) = chw(&r, |_, x| ((x as f32 / 255.0) - 0.5) / 0.5)?;
                for c in 0..3 {
                    for y in 0..ch {
                        let dst = k * 3 * ch * cwid + c * ch * cwid + y * cwid;
                        data[dst..dst + rw].copy_from_slice(&v[c * ch * rw + y * rw..c * ch * rw + (y + 1) * rw]);
                    }
                }
            }
            let out = self.cls.run(ort::inputs![tensor([batch.len(), 3, ch, cwid], data)?]).map_err(|e| anyhow!("{e}"))?;
            let (_, p) = out[0].try_extract_tensor::<f32>().map_err(|e| anyhow!("{e}"))?;
            for (k, &i) in batch.iter().enumerate() {
                let (a, b) = (p[2 * k], p[2 * k + 1]);
                res[i] = if b > a { ("180".to_string(), b as f64) } else { ("0".to_string(), a as f64) };
            }
        }
        Ok(res)
    }

    /// Read the text lines of an image (horizontal lines only).
    pub fn read(&mut self, img: &Mat, min_score: f64) -> Result<Vec<Line>> {
        let img = bgr(img)?;
        let mut quads: Vec<Quad> = self.detect(&img)?.into_iter().filter(|q| norm32(q[1], q[0]) >= 0.8f32 * norm32(q[3], q[0])).collect();
        let key = |q: &Quad| {
            let mut s = 0f32;
            for p in q {
                s += p[1];
            }
            ((s / 4.0) as f64, q.iter().map(|p| p[0]).fold(f32::INFINITY, f32::min) as f64)
        };
        quads.sort_by(|a, b| key(a).partial_cmp(&key(b)).unwrap());
        let crops = Self::crops(&img, &quads)?;
        let recs = self.recognize(&crops)?;
        let mut lines = Vec::new();
        for (q, (text, score, pos, probs)) in quads.into_iter().zip(recs) {
            if !text.trim().is_empty() && score >= min_score {
                lines.push(Line::new(q, text, score, pos, probs));
            }
        }
        Ok(lines)
    }

    #[allow(clippy::type_complexity)]
    fn recognize(&mut self, crops: &[Mat]) -> Result<Vec<(String, f64, Vec<f64>, Vec<f64>)>> {
        let (c, hh) = (3usize, 48usize);
        let ratios: Vec<f64> = crops.iter().map(|cr| cr.cols() as f64 / cr.rows().max(1) as f64).collect();
        let mut order: Vec<usize> = (0..crops.len()).collect();
        order.sort_by(|&a, &b| ratios[a].partial_cmp(&ratios[b]).unwrap());
        let mut out = vec![(String::new(), 0.0, vec![], vec![]); crops.len()];
        for idx in order.chunks(8) {
            let ratio = idx.iter().map(|&i| ratios[i]).fold(f64::NEG_INFINITY, f64::max);
            let ww = (hh as f64 * ratio) as usize;
            let mut data = vec![0f32; idx.len() * c * hh * ww];
            let mut widths = Vec::new();
            for (k, &i) in idx.iter().enumerate() {
                let (h, w) = (crops[i].rows().max(1) as f64, crops[i].cols() as f64);
                let rw = ww.min((hh as f64 * w / h).ceil() as usize);
                let r = util::resize(&crops[i], rw.max(1) as i32, hh as i32, imgproc::INTER_LINEAR)?;
                let rw1 = rw.max(1);
                let (_, _, v) = chw(&r, |_, x| ((x as f32 / 255.0) - 0.5) / 0.5)?;
                let n = rw.min(rw1);
                for cc in 0..c {
                    for y in 0..hh {
                        let dst = k * c * hh * ww + cc * hh * ww + y * ww;
                        data[dst..dst + n].copy_from_slice(&v[cc * hh * rw1 + y * rw1..cc * hh * rw1 + y * rw1 + n]);
                    }
                }
                widths.push(rw1);
            }
            let res = self.rec.run(ort::inputs![tensor([idx.len(), c, hh, ww], data)?]).map_err(|e| anyhow!("{e}"))?;
            let (shape, p) = res[0].try_extract_tensor::<f32>().map_err(|e| anyhow!("{e}"))?;
            let (t, ncls) = (shape[1] as usize, shape[2] as usize);
            for (k, &i) in idx.iter().enumerate() {
                let rows = &p[k * t * ncls..(k + 1) * t * ncls];
                out[i] = decode(&self.chars, rows, t, ncls, ww as f64 / t as f64, widths[k]);
            }
        }
        Ok(out)
    }

}

fn decode(chars: &[String], p: &[f32], t_len: usize, ncls: usize, stride: f64, width: usize) -> (String, f64, Vec<f64>, Vec<f64>) {
        let mut best = Vec::with_capacity(t_len);
        let mut prob = Vec::with_capacity(t_len);
        for t in 0..t_len {
            let row = &p[t * ncls..(t + 1) * ncls];
            let mut bi = 0;
            for j in 1..ncls {
                if row[j] > row[bi] {
                    bi = j;
                }
            }
            best.push(bi);
            prob.push(row[bi]);
        }
        let (mut text, mut probs, mut pos) = (String::new(), Vec::new(), Vec::new());
        let mut t = 0;
        while t < best.len() {
            let k = best[t];
            let mut u = t;
            while u + 1 < best.len() && best[u + 1] == k {
                u += 1;
            }
            if k != 0 {
                text.push_str(chars.get(k).map(|s| s.as_str()).unwrap_or(""));
                probs.push(prob[t..=u].iter().cloned().fold(f32::NEG_INFINITY, f32::max) as f64);
                pos.push((((t + u) as f64 / 2.0 + 0.5) * stride / width as f64).min(1.0));
            }
            t = u + 1;
        }
        let score = if probs.is_empty() { 0.0 } else { util::np_mean(&probs) };
        (text, score, pos, probs)
    }

/// Read a page: `with_engine(|o| o.read(...))`.
pub fn read(img: &Mat, min_score: f64) -> Result<Vec<Line>> {
    with_engine(|o| o.read(img, min_score))
}

pub fn check() -> Result<()> {
    if !available() {
        bail!(
            "reading unknown layouts needs ONNX Runtime ({}) and the OCR models in {} (RFPII_ORT_LIB, RFPII_OCR_MODELS)",
            ort_lib().display(),
            models_dir().display()
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn offset_like_pyclipper() {
        // pyclipper: PyclipperOffset().AddPath(box, JT_ROUND, ET_CLOSEDPOLYGON).Execute(5.3)
        // its union only drops the collinear points [10,15], [115,20], [110,45], [5,40]
        let r = offset_round(&[[10.7, 20.9], [110.2, 20.1], [110.9, 40.6], [10.1, 40.2]], 5.3);
        let py = [[113, 16], [115, 18], [115, 40], [114, 43], [112, 45], [10, 45], [7, 44], [5, 42], [5, 20], [6, 17], [8, 15], [110, 15]];
        assert_eq!(r.len(), 16);
        assert!(py.iter().all(|p| r.contains(p)));
        assert_eq!(edits_str("signature", "sinature"), 1);
    }
}

