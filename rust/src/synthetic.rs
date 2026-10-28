//! Synthetic polling-station result form, for the demo template and tests.
//!
//! The layout is generic (not any real authority's form): a header, a
//! candidate results table (public data) and a declaration block with
//! officials' and agents' names, ID numbers, phone numbers and signatures.

use anyhow::Result;
use opencv::core::{self, Mat, Point, Point2f, Scalar, Size, Vector};
use opencv::imgproc;
use opencv::prelude::*;
use rand::{RngExt, SeedableRng};
use rand_chacha::ChaCha8Rng;
use std::path::Path;

use crate::util;

pub const W: i32 = 1240;
pub const H: i32 = 1754; // A4 at 150 dpi

/// PII zones in template coordinates: name -> (x, y, w, h).
pub const ZONES: &[(&str, [i32; 4])] = &[
    ("presiding_officer", [330, 1210, 860, 110]),
    ("deputy_officer", [330, 1330, 860, 110]),
    ("agents_table", [60, 1500, 1120, 200]),
];

fn text(img: &mut Mat, s: &str, x: i32, y: i32, scale: f64, thick: i32) -> Result<()> {
    imgproc::put_text(img, s, Point::new(x, y), imgproc::FONT_HERSHEY_SIMPLEX, scale, Scalar::all(0.0), thick, imgproc::LINE_AA, false)?;
    Ok(())
}

fn line(img: &mut Mat, a: (i32, i32), b: (i32, i32), t: i32) -> Result<()> {
    imgproc::line(img, Point::new(a.0, a.1), Point::new(b.0, b.1), Scalar::all(0.0), t, imgproc::LINE_8, 0)?;
    Ok(())
}

fn rect(img: &mut Mat, a: (i32, i32), b: (i32, i32), t: i32) -> Result<()> {
    imgproc::rectangle_points(img, Point::new(a.0, a.1), Point::new(b.0, b.1), Scalar::all(0.0), t, imgproc::LINE_8, 0)?;
    Ok(())
}

pub fn blank_form() -> Result<Mat> {
    let mut img = util::full_u8(H, W, 255)?;
    rect(&mut img, (40, 40), (W - 40, H - 40), 3)?;
    text(&mut img, "ELECTION RESULTS AT THE POLLING STATION", 180, 110, 1.0, 2)?;
    text(&mut img, "FORM RS-1  (SAMPLE TEMPLATE)", 380, 160, 0.8, 2)?;
    for (i, label) in ["County", "Constituency", "Ward", "Polling Station", "Station Code"].iter().enumerate() {
        let y = 230 + i as i32 * 50;
        text(&mut img, &format!("{}:", label), 70, y, 0.6, 1)?;
        line(&mut img, (300, y + 5), (1180, y + 5), 1)?;
    }
    // results table: public data, must survive redaction
    let (top, rows) = (520, 8);
    rect(&mut img, (60, top), (1180, top + 60 * (rows + 1)), 2)?;
    for (x, hdr) in [(60, "No."), (160, "Candidate"), (700, "Party"), (950, "Votes")] {
        line(&mut img, (x, top), (x, top + 60 * (rows + 1)), 2)?;
        text(&mut img, hdr, x + 15, top + 40, 0.7, 2)?;
    }
    for r in 1..=rows {
        let y = top + 60 * r;
        line(&mut img, (60, y), (1180, y), 1)?;
        text(&mut img, &r.to_string(), 90, y + 40, 0.6, 1)?;
    }
    text(&mut img, "Total valid votes:", 70, 1110, 0.7, 2)?;
    text(&mut img, "Rejected:", 650, 1110, 0.7, 2)?;
    text(&mut img, "DECLARATION", 70, 1180, 0.8, 2)?;
    for (i, role) in ["Presiding Officer", "Deputy P. Officer"].iter().enumerate() {
        let y = 1210 + 120 * i as i32;
        text(&mut img, role, 70, y + 40, 0.6, 2)?;
        rect(&mut img, (330, y), (1190, y + 110), 1)?;
        text(&mut img, "Name", 340, y + 25, 0.5, 1)?;
        text(&mut img, "ID No.", 760, y + 25, 0.5, 1)?;
        text(&mut img, "Signature", 340, y + 95, 0.5, 1)?;
        text(&mut img, "Phone", 760, y + 95, 0.5, 1)?;
    }
    text(&mut img, "AGENTS", 70, 1480, 0.8, 2)?;
    rect(&mut img, (60, 1500), (1180, 1700), 1)?;
    for r in 1..4 {
        line(&mut img, (60, 1500 + 50 * r), (1180, 1500 + 50 * r), 1)?;
    }
    Ok(img)
}

pub fn filled_form(seed: u64) -> Result<Mat> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut img = blank_form()?;
    for (i, v) in ["NORTH", "RIVERSIDE", "CENTRAL", "St. Mary Primary", "045-112-03"].iter().enumerate() {
        text(&mut img, v, 320, 225 + i as i32 * 50, 0.7, 2)?;
    }
    for r in 1..=8 {
        let y = 520 + 60 * r;
        text(&mut img, &format!("Candidate {}", (64 + r as u8) as char), 180, y + 40, 0.7, 2)?;
        text(&mut img, &format!("PTY{}", r), 720, y + 40, 0.7, 2)?;
        text(&mut img, &rng.random_range(0..500).to_string(), 980, y + 40, 0.9, 2)?;
    }
    for i in 0..2 {
        let y = 1210 + 120 * i;
        text(&mut img, if i == 0 { "JANE A. DOE" } else { "JOHN B. ROE" }, 420, y + 30, 0.8, 2)?;
        text(&mut img, &rng.random_range(10_000_000..100_000_000).to_string(), 860, y + 30, 0.8, 2)?;
        text(&mut img, &format!("07{}", rng.random_range(10_000_000..100_000_000)), 840, y + 100, 0.8, 2)?;
        let (mut x, mut yy) = (520, y + 85); // a "signature"
        let mut pts = Vector::<Point>::new();
        for _ in 0..25 {
            x += rng.random_range(-12..13);
            yy += rng.random_range(-12..13);
            pts.push(Point::new(x, yy));
        }
        let polys: Vector<Vector<Point>> = vec![pts].into_iter().collect();
        imgproc::polylines(&mut img, &polys, false, Scalar::all(0.0), 2, imgproc::LINE_8, 0)?;
    }
    for r in 0..4 {
        let s = format!("Agent {}  ID {}  Tel 07{}", r + 1, rng.random_range(10_000_000..100_000_000), rng.random_range(10_000_000..100_000_000));
        text(&mut img, &s, 80, 1535 + 50 * r, 0.7, 2)?;
    }
    Ok(img)
}

/// Rotate, scale, add perspective, lighting and noise: a "scan" (BGR), and
/// the homography mapping template coordinates to scan coordinates.
pub fn simulate_scan(img: &Mat, seed: u64, angle: f64, out_scale: f64) -> Result<(Mat, [[f64; 3]; 3])> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed ^ 0x5ca1ab1e);
    let (h, w) = (img.rows() as f64, img.cols() as f64);
    let (ow, oh) = ((w * out_scale) as i32, (h * out_scale) as i32);
    let src = [[0.0, 0.0], [w, 0.0], [w, h], [0.0, h]];
    let a = angle.to_radians();
    let (c, s) = (a.cos(), a.sin());
    let mut sv = Vector::<Point2f>::new();
    let mut dv = Vector::<Point2f>::new();
    for p in src {
        let (jx, jy) = (rng.random_range(-0.02..0.02) * w, rng.random_range(-0.02..0.02) * h);
        let (x, y) = (p[0] - w / 2.0 + jx, p[1] - h / 2.0 + jy);
        sv.push(Point2f::new(p[0] as f32, p[1] as f32));
        dv.push(Point2f::new(((c * x - s * y) * 1.1 + ow as f64 / 2.0) as f32, ((s * x + c * y) * 1.1 + oh as f64 / 2.0) as f32));
    }
    let hm = opencv::geometry::get_perspective_transform(&sv, &dv, core::DECOMP_LU)?;
    let mut scan = Mat::default();
    imgproc::warp_perspective(img, &mut scan, &hm, Size::new(ow, oh), imgproc::INTER_LINEAR, core::BORDER_CONSTANT, Scalar::all(200.0), core::AlgorithmHint::ALGO_HINT_DEFAULT)?;
    let mut d = util::bytes(&scan)?;
    let owu = ow as usize;
    for (k, v) in d.iter_mut().enumerate() {
        let grad = 0.85 + 0.2 * (k % owu) as f64 / (owu - 1).max(1) as f64;
        // normal noise (sigma 6) by Box-Muller
        let (u1, u2): (f64, f64) = (rng.random_range(f64::EPSILON..1.0), rng.random());
        let n = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos() * 6.0;
        *v = (*v as f64 * grad + n).clamp(0.0, 255.0) as u8;
    }
    let g = util::from_bytes(oh, ow, &d)?;
    let mut out = Mat::default();
    imgproc::cvt_color_def(&g, &mut out, imgproc::COLOR_GRAY2BGR)?;
    Ok((out, util::arr3(&hm)?))
}

pub fn write_template(dir: &Path) -> Result<()> {
    std::fs::create_dir_all(dir)?;
    opencv::imgcodecs::imwrite(&dir.join("reference.png"), &blank_form()?, &Vector::new())?;
    let mut y = String::from("name: sample_rs1\nreference: reference.png\npadding: 12\nprotect:\n- name: results\n  rect:\n  - 60\n  - 520\n  - 1120\n  - 620\nzones:\n");
    for (name, r) in ZONES {
        y += &format!("- name: {}\n  rect:\n  - {}\n  - {}\n  - {}\n  - {}\n", name, r[0], r[1], r[2], r[3]);
    }
    std::fs::write(dir.join("template.yaml"), y)?;
    Ok(())
}
