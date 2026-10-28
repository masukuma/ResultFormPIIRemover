//! Helpers for the tests: images, drawing, temporary directories.

use opencv::core::{self, Mat, Point, Scalar, Vector};
use opencv::imgproc;
use opencv::prelude::*;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};

use crate::masking::{self, Img, Mapper};
use crate::util;

/// A new, empty directory for one test.
pub fn tmpdir(name: &str) -> PathBuf {
    static N: AtomicUsize = AtomicUsize::new(0);
    let d = std::env::temp_dir().join(format!("rfpii-test-{}-{}-{}", std::process::id(), N.fetch_add(1, Ordering::SeqCst), name));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

pub fn identity() -> Mapper {
    masking::homography([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
}

pub fn gray(h: i32, w: i32, v: u8) -> Mat {
    util::full_u8(h, w, v).unwrap()
}

pub fn bgr(h: i32, w: i32, v: [u8; 3]) -> Mat {
    Mat::new_rows_cols_with_default(h, w, core::CV_8UC3, Scalar::new(v[0] as f64, v[1] as f64, v[2] as f64, 0.0)).unwrap()
}

pub fn to_bgr(img: &Mat) -> Mat {
    let mut out = Mat::default();
    imgproc::cvt_color_def(img, &mut out, imgproc::COLOR_GRAY2BGR).unwrap();
    out
}

pub fn to_gray(img: &Mat) -> Mat {
    util::gray(img).unwrap()
}

pub fn colour(v: f64) -> Scalar {
    Scalar::all(v)
}

pub fn text(img: &mut Mat, s: &str, at: (i32, i32), scale: f64, c: Scalar, thick: i32, aa: bool) {
    let kind = if aa { imgproc::LINE_AA } else { imgproc::LINE_8 };
    imgproc::put_text(img, s, Point::new(at.0, at.1), imgproc::FONT_HERSHEY_SIMPLEX, scale, c, thick, kind, false).unwrap();
}

pub fn line(img: &mut Mat, a: (i32, i32), b: (i32, i32), c: Scalar, thick: i32) {
    imgproc::line(img, Point::new(a.0, a.1), Point::new(b.0, b.1), c, thick, imgproc::LINE_8, 0).unwrap();
}

pub fn polyline(img: &mut Mat, pts: &[(i32, i32)], c: Scalar, thick: i32) {
    let p: Vector<Point> = pts.iter().map(|&(x, y)| Point::new(x, y)).collect();
    let polys: Vector<Vector<Point>> = vec![p].into_iter().collect();
    imgproc::polylines(img, &polys, false, c, thick, imgproc::LINE_8, 0).unwrap();
}

/// Fill rows y0..y1, columns x0..x1 (clipped, numpy-slice style) of a gray image.
pub fn fill_rect(img: &mut Mat, x0: i32, y0: i32, x1: i32, y1: i32, v: u8) {
    let (w, h) = (img.cols(), img.rows());
    let (x0, x1, y0, y1) = (x0.clamp(0, w), x1.clamp(0, w), y0.clamp(0, h), y1.clamp(0, h));
    for y in y0..y1 {
        for x in x0..x1 {
            *img.at_2d_mut::<u8>(y, x).unwrap() = v;
        }
    }
}

pub fn img(m: &Mat) -> Img {
    Img::from_mat(m).unwrap()
}

/// Per-pixel boolean mask of a gray image.
pub fn mask(m: &Mat, f: impl Fn(u8) -> bool) -> Vec<bool> {
    util::bytes(m).unwrap().into_iter().map(f).collect()
}

/// Every pixel in rows y0..y1, columns x0..x1 of a gray image equals v.
pub fn all_eq(m: &Mat, x0: i32, y0: i32, x1: i32, y1: i32, v: u8) -> bool {
    let i = img(m);
    let (x1, y1) = ((x1.min(m.cols())) as usize, (y1.min(m.rows())) as usize);
    (y0 as usize..y1).all(|y| (x0 as usize..x1).all(|x| i.at(x, y) == v))
}

/// Share of the selected pixels that are 0 in `canvas`.
pub fn covered(canvas: &Mat, sel: &[bool]) -> f64 {
    let c = util::bytes(canvas).unwrap();
    let n = sel.iter().filter(|&&s| s).count();
    if n == 0 {
        return f64::NAN;
    }
    sel.iter().zip(&c).filter(|&(&s, &v)| s && v == 0).count() as f64 / n as f64
}

/// One signature's ink (`sig`, w x h booleans) is masked as prescribed: in
/// every piece along it, the middle of its rows is under black, while a fair
/// share of its ink (its tops and bottoms) stays visible.
pub fn middle_hidden(canvas: &Mat, sig: &[bool], chunk: i64, at_most: f64) -> bool {
    let w = canvas.cols() as usize;
    let c = util::bytes(canvas).unwrap();
    let pts: Vec<(i64, i64, bool)> = sig.iter().enumerate().filter(|&(_, &s)| s).map(|(i, _)| ((i % w) as i64, (i / w) as i64, c[i] == 0)).collect();
    let (xmin, xmax) = (pts.iter().map(|p| p.0).min().unwrap(), pts.iter().map(|p| p.0).max().unwrap());
    let mut a = xmin;
    while a <= xmax {
        let sel: Vec<&(i64, i64, bool)> = pts.iter().filter(|p| p.0 >= a && p.0 < a + chunk).collect();
        let span = sel.iter().map(|p| p.0).max().unwrap_or(0) - sel.iter().map(|p| p.0).min().unwrap_or(0);
        if sel.len() >= 10 && span as f64 >= 0.75 * chunk as f64 {
            let ys: Vec<f64> = sel.iter().map(|p| p.1 as f64).collect();
            let (lo, hi) = (util::quantile(&ys, 0.4), util::quantile(&ys, 0.6));
            let mid: Vec<bool> = sel.iter().filter(|p| p.1 as f64 >= lo && p.1 as f64 <= hi).map(|p| p.2).collect();
            if (mid.iter().filter(|&&v| v).count() as f64) < 0.8 * mid.len() as f64 {
                return false;
            }
        }
        a += chunk;
    }
    let share = pts.iter().filter(|p| p.2).count() as f64 / pts.len() as f64;
    0.2 < share && share < at_most
}

/// `np.where(canvas == 0, 0, gray)` as BGR: the redacted image.
pub fn apply(canvas: &Mat, gray_img: &Mat) -> Mat {
    let c = util::bytes(canvas).unwrap();
    let g = util::bytes(gray_img).unwrap();
    let out: Vec<u8> = c.iter().zip(&g).map(|(&a, &b)| if a == 0 { 0 } else { b }).collect();
    to_bgr(&util::from_bytes(gray_img.rows(), gray_img.cols(), &out).unwrap())
}

/// Booleans (w x h) to a 0/255 image.
pub fn from_mask(m: &[bool], h: i32, w: i32) -> Mat {
    let d: Vec<u8> = m.iter().map(|&b| if b { 255 } else { 0 }).collect();
    util::from_bytes(h, w, &d).unwrap()
}

/// A template with nothing but its name and multi-page rules.
pub fn bare_template(name: &str, form: &str, once: bool, priority: i64) -> crate::templates::Template {
    crate::templates::Template {
        name: name.into(),
        image: gray(10, 10, 0),
        zones: vec![],
        padding: 0,
        form: form.into(),
        structure: vec![],
        protect: vec![],
        fingerprint: String::new(),
        min_area_frac: None,
        priority,
        once,
        page_long_side: None,
        min_inliers: None,
        min_score: None,
        feature_dim: None,
        n_features: None,
        open_top: vec![],
        min_glyph_h: None,
        free_text: vec![],
        text_h: None,
        regions: vec![],
        mask: Mat::default(),
        keypoints: vec![],
        descriptors: Mat::default(),
    }
}

/// A page that matched `template`.
pub fn matched_page(template: &str) -> crate::pipeline::PageResult {
    crate::pipeline::PageResult {
        ok: true,
        image: None,
        template: template.into(),
        inliers: 100,
        score: 0.9,
        reason: String::new(),
        ms: 0.0,
        masks: vec![],
        notes: vec![],
        blocks: None,
        protected: vec![],
        h: None,
        layout: None,
    }
}

/// `np.linspace(a, b, n)`.
pub fn linspace(a: f64, b: f64, n: usize) -> Vec<f64> {
    let step = (b - a) / (n - 1) as f64;
    let mut v: Vec<f64> = (0..n).map(|k| k as f64 * step + a).collect();
    v[n - 1] = b;
    v
}
