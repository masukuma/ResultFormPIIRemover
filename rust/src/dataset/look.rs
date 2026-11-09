//! How a synthetic page is photographed or scanned: one homography (angle,
//! perspective, scale, margins), then light, a fold, blur, noise, a turn and
//! at last a smaller size. The page and its mask go through the same
//! geometry (the mask by nearest neighbour), and the homography is kept, so
//! every label is moved exactly as the ink was.

use anyhow::Result;
use opencv::core::{self, Mat, Point2f, Scalar, Size, Vector};
use opencv::imgproc;
use opencv::prelude::*;
use serde_json::{json, Value};

use super::draw::{BIT_BLACK, BIT_PEN, BIT_PRINT, BIT_STAMP};
use crate::util;

pub type H3 = [[f64; 3]; 3];

/// How a page is degraded, chosen per sample.
#[derive(Clone, Debug)]
pub struct Look {
    /// a photo (perspective, a background round the page, uneven light) or
    /// a scan
    pub photo: bool,
    pub angle: f64,
    /// each corner moved by up to this share of the page
    pub persp: f64,
    pub scale: f64,
    /// margin round the page, as a share of its long side
    pub margin: f64,
    /// quarter turns clockwise of the whole image
    pub quarter_turns: i32,
    pub blur: f64,
    pub noise: f64,
    pub quality: i32,
    /// a crease: a point and direction (unit page coordinates) and how much
    /// darker the far side is
    pub fold: Option<[f64; 5]>,
    /// light across the page: direction (radians), darkest and brightest
    pub light: [f64; 3],
    pub background: [u8; 3],
    pub tint: [f64; 3],
    pub pen_opacity: f64,
}

impl Look {
    pub fn to_json(&self) -> Value {
        let r = |v: f64| (v * 1000.0).round() / 1000.0;
        json!({"photo": self.photo, "angle": r(self.angle), "perspective": r(self.persp), "scale": r(self.scale),
               "margin": r(self.margin), "turn": 90 * self.quarter_turns, "blur": r(self.blur), "noise": r(self.noise),
               "jpeg_quality": self.quality, "fold": self.fold.map(|f| f.iter().map(|&v| r(v)).collect::<Vec<_>>()),
               "light": self.light.iter().map(|&v| r(v)).collect::<Vec<_>>()})
    }
}

pub fn mul(a: &H3, b: &H3) -> H3 {
    util::mul3(a, b)
}

/// The homography from the w x h canvas to the photo or scan (before any
/// turn), and the output size.
pub fn geometry(w: f64, h: f64, look: &Look, jitter: &[f64; 8]) -> Result<(H3, i32, i32)> {
    let (cx, cy) = (w / 2.0, h / 2.0);
    let a = look.angle.to_radians();
    let (c, s) = (a.cos(), a.sin());
    let long = w.max(h) * look.scale;
    let src = [[0.0, 0.0], [w, 0.0], [w, h], [0.0, h]];
    let mut dst = [[0.0f64; 2]; 4];
    for (i, p) in src.iter().enumerate() {
        let (x, y) = ((p[0] - cx) * look.scale, (p[1] - cy) * look.scale);
        dst[i] = [c * x - s * y + jitter[2 * i] * look.persp * long, s * x + c * y + jitter[2 * i + 1] * look.persp * long];
    }
    let b = super::bounds(&dst);
    let m = look.margin * long;
    for d in dst.iter_mut() {
        d[0] += m - b[0];
        d[1] += m - b[1];
    }
    let (ow, oh) = ((b[2] - b[0] + 2.0 * m).ceil() as i32, (b[3] - b[1] + 2.0 * m).ceil() as i32);
    let sv: Vector<Point2f> = src.iter().map(|p| Point2f::new(p[0] as f32, p[1] as f32)).collect();
    let dv: Vector<Point2f> = dst.iter().map(|p| Point2f::new(p[0] as f32, p[1] as f32)).collect();
    let hm = opencv::geometry::get_perspective_transform(&sv, &dv, core::DECOMP_LU)?;
    Ok((util::arr3(&hm)?, ow.max(1), oh.max(1)))
}

/// The matrix of `k` quarter turns clockwise of a w x h image (pixel
/// centres at whole numbers, as `cv::rotate` moves them), and the new size.
pub fn turn_matrix(k: i32, w: i32, h: i32) -> (H3, i32, i32) {
    let (w1, h1) = ((w - 1) as f64, (h - 1) as f64);
    match k.rem_euclid(4) {
        1 => ([[0.0, -1.0, h1], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], h, w),
        2 => ([[-1.0, 0.0, w1], [0.0, -1.0, h1], [0.0, 0.0, 1.0]], w, h),
        3 => ([[0.0, 1.0, 0.0], [-1.0, 0.0, w1], [0.0, 0.0, 1.0]], h, w),
        _ => (util::IDENTITY, w, h),
    }
}

pub fn turn(img: &Mat, k: i32) -> Result<Mat> {
    let code = match k.rem_euclid(4) {
        1 => core::ROTATE_90_CLOCKWISE,
        2 => core::ROTATE_180,
        3 => core::ROTATE_90_COUNTERCLOCKWISE,
        _ => return Ok(img.try_clone()?),
    };
    let mut out = Mat::default();
    core::rotate(img, &mut out, code)?;
    Ok(out)
}

pub fn warp(img: &Mat, h: &H3, w: i32, hh: i32, nearest: bool, border: Scalar) -> Result<Mat> {
    let mut out = Mat::default();
    let interp = if nearest { imgproc::INTER_NEAREST } else { imgproc::INTER_LINEAR };
    imgproc::warp_perspective(img, &mut out, &util::mat3(h)?, Size::new(w, hh), interp, core::BORDER_CONSTANT, border, core::AlgorithmHint::ALGO_HINT_DEFAULT)?;
    Ok(out)
}

/// Light, a fold, blur and noise on a BGR photo or scan; `seed` seeds the
/// noise.
pub fn photometric(img: &mut Mat, look: &Look, seed: u64) -> Result<()> {
    let (w, h) = (img.cols() as usize, img.rows() as usize);
    let [dir, lo, hi] = look.light;
    let (dx, dy) = (dir.cos(), dir.sin());
    let span = (w as f64 * dx.abs() + h as f64 * dy.abs()).max(1.0);
    let off = (if dx < 0.0 { -dx * w as f64 } else { 0.0 }) + (if dy < 0.0 { -dy * h as f64 } else { 0.0 });
    let fold = look.fold.map(|[px, py, ang, depth, soft]| {
        let (nx, ny) = (-(ang.sin()), ang.cos());
        (px * w as f64, py * h as f64, nx, ny, depth, soft * w.max(h) as f64)
    });
    {
        let d = img.data_bytes_mut()?;
        for y in 0..h {
            for x in 0..w {
                let t = ((x as f64 * dx + y as f64 * dy) + off) / span;
                let mut f = lo + (hi - lo) * t.clamp(0.0, 1.0);
                if let Some((px, py, nx, ny, depth, soft)) = fold {
                    let s = (x as f64 - px) * nx + (y as f64 - py) * ny;
                    if s > 0.0 {
                        f *= 1.0 - depth * (1.0 - (-s / soft).exp());
                    }
                    if s.abs() < 1.5 {
                        f *= 0.82;
                    } else if (-4.5..-1.5).contains(&s) {
                        f *= 1.05;
                    }
                }
                let i = 3 * (y * w + x);
                for c in 0..3 {
                    d[i + c] = (d[i + c] as f64 * f).round().clamp(0.0, 255.0) as u8;
                }
            }
        }
    }
    if look.blur > 0.05 {
        let mut b = Mat::default();
        imgproc::gaussian_blur(img, &mut b, Size::new(0, 0), look.blur, 0.0, core::BORDER_REPLICATE, core::AlgorithmHint::ALGO_HINT_DEFAULT)?;
        *img = b;
    }
    if look.noise > 0.05 {
        core::set_rng_seed((seed % (1u64 << 31)) as i32)?;
        let mut n = Mat::new_rows_cols_with_default(h as i32, w as i32, core::CV_16SC3, Scalar::all(0.0))?;
        core::randn(&mut n, &Scalar::all(0.0), &Scalar::all(look.noise))?;
        let mut img16 = Mat::default();
        img.convert_to(&mut img16, core::CV_16SC3, 1.0, 0.0)?;
        let mut sum = Mat::default();
        core::add_def(&img16, &n, &mut sum)?;
        let mut out = Mat::default();
        sum.convert_to(&mut out, core::CV_8UC3, 1.0, 0.0)?;
        *img = out;
    }
    Ok(())
}

/// A mask of bits made smaller: a bit is set where it covers at least a
/// quarter of the new pixel (thin strokes stay in the mask).
pub fn shrink_bits(bits: &Mat, w: i32, h: i32) -> Result<Mat> {
    let src = util::bytes(bits)?;
    let mut out = vec![0u8; (w * h) as usize];
    for b in [BIT_PRINT, BIT_PEN, BIT_STAMP, BIT_BLACK] {
        if !src.iter().any(|&v| v & b != 0) {
            continue;
        }
        let plane: Vec<u8> = src.iter().map(|&v| if v & b != 0 { 255 } else { 0 }).collect();
        let small = util::resize(&util::from_bytes(bits.rows(), bits.cols(), &plane)?, w, h, imgproc::INTER_AREA)?;
        for (o, &v) in out.iter_mut().zip(util::bytes(&small)?.iter()) {
            if v >= 64 {
                *o |= b;
            }
        }
    }
    util::from_bytes(h, w, &out)
}

/// The matrix of a resize from (w, h) to (nw, nh), pixel centres mapped as
/// `cv::resize` maps them.
pub fn scale_matrix(w: i32, h: i32, nw: i32, nh: i32) -> H3 {
    let (fx, fy) = (nw as f64 / w as f64, nh as f64 / h as f64);
    [[fx, 0.0, 0.5 * fx - 0.5], [0.0, fy, 0.5 * fy - 0.5], [0.0, 0.0, 1.0]]
}

/// The size a w x h image takes with its long side at most `max_side`.
pub fn fit(w: i32, h: i32, max_side: Option<u32>) -> (i32, i32) {
    match max_side {
        Some(m) if w.max(h) > m as i32 => {
            let f = m as f64 / w.max(h) as f64;
            (((w as f64 * f).round() as i32).max(1), ((h as f64 * f).round() as i32).max(1))
        }
        _ => (w, h),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn turns_move_pixels_as_their_matrix_says() {
        let mut m = util::zeros_u8(3, 5).unwrap();
        *m.at_2d_mut::<u8>(1, 4).unwrap() = 9; // x 4, y 1
        for k in 0..4 {
            let (t, w, h) = turn_matrix(k, 5, 3);
            let r = turn(&m, k).unwrap();
            assert_eq!((r.cols(), r.rows()), (w, h));
            let p = util::project(&t, [4.0, 1.0]);
            assert_eq!(*r.at_2d::<u8>(p[1].round() as i32, p[0].round() as i32).unwrap(), 9, "turn {k}");
        }
    }

    #[test]
    fn the_page_stays_in_view_and_shrinking_keeps_thin_strokes() {
        let look = Look { photo: true, angle: 7.0, persp: 0.05, scale: 0.9, margin: 0.05, quarter_turns: 0, blur: 0.0, noise: 0.0, quality: 90, fold: None, light: [0.0, 1.0, 1.0], background: [0, 0, 0], tint: [1.0; 3], pen_opacity: 1.0 };
        let (h, ow, oh) = geometry(1000.0, 1400.0, &look, &[1.0, -1.0, 0.5, 0.5, -1.0, 1.0, 0.0, -0.5]).unwrap();
        for p in [[0.0, 0.0], [1000.0, 0.0], [1000.0, 1400.0], [0.0, 1400.0]] {
            let q = util::project(&h, p);
            assert!(q[0] >= 0.0 && q[1] >= 0.0 && q[0] <= ow as f64 && q[1] <= oh as f64, "{q:?}");
        }
        let mut bits = util::zeros_u8(100, 100).unwrap();
        for y in 0..100 {
            *bits.at_2d_mut::<u8>(y, 50).unwrap() = BIT_PEN;
        }
        let small = shrink_bits(&bits, 40, 40).unwrap();
        assert!(util::bytes(&small).unwrap().iter().filter(|&&v| v == BIT_PEN).count() >= 40);
        assert_eq!(fit(2000, 1000, Some(1280)), (1280, 640));
        assert_eq!(fit(800, 600, Some(1280)), (800, 600));
        let s = scale_matrix(100, 100, 50, 50);
        assert_eq!(util::project(&s, [1.5, 1.5]), [0.5, 0.5]);
    }
}
