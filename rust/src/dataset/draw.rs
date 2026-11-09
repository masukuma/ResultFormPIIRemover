//! The layers a synthetic page is drawn on, kept apart so that every pixel's
//! class is known exactly: the printed form on its paper, the pen, and the
//! stamp ink. They are blended into the page only at the end (`compose`),
//! and the same layers give the mask (`bits`).

use anyhow::Result;
use opencv::core::{self, Mat, Point, Rect, Scalar, Size, Vector};
use opencv::imgproc;
use opencv::prelude::*;

use super::glyphs::{self, Hand, Poly};
use crate::util;

/// Mask bits (masks/<id>.png): print and rules, pen, stamp ink, and on a
/// redacted page the blocks blacked out.
pub const BIT_PRINT: u8 = 1;
pub const BIT_PEN: u8 = 2;
pub const BIT_STAMP: u8 = 4;
pub const BIT_BLACK: u8 = 8;

/// Stamp ink this opaque or more (of 255) is stamp ink in the mask: the
/// faint gaps of a worn stamp are not.
pub const STAMP_FROM: u8 = 64;

pub struct Layers {
    pub w: i32,
    pub h: i32,
    /// the printed form on its paper (grey levels)
    pub paper: Mat,
    /// 255 where print and rules
    pub print: Mat,
    /// 255 where pen, and the pen's colour there (BGR)
    pub pen: Mat,
    pub pen_color: Mat,
    /// stamp ink opacity (0..255) and colour
    pub stamp: Mat,
    pub stamp_color: Mat,
    /// which piece of writing drew each pen pixel (0: none); kept only when
    /// a redacted page needs to know what each character shows
    pub owner: Option<Mat>,
}

fn pts8(p: &Poly) -> Vector<Point> {
    p.iter().map(|q| Point::new((q[0] * 8.0).round() as i32, (q[1] * 8.0).round() as i32)).collect()
}

impl Layers {
    pub fn new(w: i32, h: i32, paper: u8, owner: bool) -> Result<Layers> {
        Ok(Layers {
            w,
            h,
            paper: util::full_u8(h, w, paper)?,
            print: util::zeros_u8(h, w)?,
            pen: util::zeros_u8(h, w)?,
            pen_color: Mat::new_rows_cols_with_default(h, w, core::CV_8UC3, Scalar::all(0.0))?,
            stamp: util::zeros_u8(h, w)?,
            stamp_color: Mat::new_rows_cols_with_default(h, w, core::CV_8UC3, Scalar::all(0.0))?,
            owner: if owner { Some(Mat::new_rows_cols_with_default(h, w, core::CV_16UC1, Scalar::all(0.0))?) } else { None },
        })
    }

    /// Put a printed reference (grey) on the paper at (x, y). Its print is
    /// what the pipeline's dark-ink rule finds on it (`stamps::dark_ink`:
    /// darker than the paper around it), so the shadows of a photographed
    /// reference are not print.
    pub fn place(&mut self, reference: &Mat, x: i32, y: i32) -> Result<()> {
        let r = Rect::new(x, y, reference.cols(), reference.rows());
        reference.copy_to(&mut Mat::roi_mut(&mut self.paper, r)?)?;
        let dark = crate::stamps::dark_ink(reference)?;
        dark.copy_to(&mut Mat::roi_mut(&mut self.print, r)?)?;
        Ok(())
    }

    /// Typed text (a computer-printed table): print.
    pub fn typed(&mut self, s: &str, x: i32, y: i32, scale: f64, thick: i32) -> Result<()> {
        for (m, v) in [(&mut self.paper, 35.0), (&mut self.print, 255.0)] {
            imgproc::put_text(m, s, Point::new(x, y), imgproc::FONT_HERSHEY_SIMPLEX, scale, Scalar::all(v), thick, imgproc::LINE_8, false)?;
        }
        Ok(())
    }

    /// A printed rule from a to b.
    pub fn rule(&mut self, a: [f64; 2], b: [f64; 2], thick: i32) -> Result<()> {
        let (p, q) = (Point::new(a[0] as i32, a[1] as i32), Point::new(b[0] as i32, b[1] as i32));
        imgproc::line(&mut self.paper, p, q, Scalar::all(35.0), thick, imgproc::LINE_8, 0)?;
        imgproc::line(&mut self.print, p, q, Scalar::all(255.0), thick, imgproc::LINE_8, 0)?;
        Ok(())
    }

    /// Pen strokes, `thick` px wide, in `color` (BGR), drawn by `owner`
    /// (0: writing whose pixels aren't tracked).
    pub fn pen(&mut self, polys: &[Poly], thick: i32, color: [u8; 3], owner: u16) -> Result<()> {
        let v: Vector<Vector<Point>> = polys.iter().filter(|p| !p.is_empty()).map(pts8).collect();
        let t = thick.max(1);
        imgproc::polylines(&mut self.pen, &v, false, Scalar::all(255.0), t, imgproc::LINE_8, 3)?;
        let c = Scalar::new(color[0] as f64, color[1] as f64, color[2] as f64, 0.0);
        imgproc::polylines(&mut self.pen_color, &v, false, c, t, imgproc::LINE_8, 3)?;
        if let (Some(o), true) = (self.owner.as_mut(), owner > 0) {
            imgproc::polylines(o, &v, false, Scalar::all(owner as f64), t, imgproc::LINE_8, 3)?;
        }
        Ok(())
    }

    /// Cursive writing (a Hershey script font) with its baseline at (x, y);
    /// returns its box (x0, y0, x1, y1).
    #[allow(clippy::too_many_arguments)]
    pub fn pen_script(&mut self, s: &str, x: f64, y: f64, scale: f64, thick: i32, color: [u8; 3], owner: u16) -> Result<[f64; 4]> {
        let font = imgproc::FONT_HERSHEY_SCRIPT_SIMPLEX;
        let at = Point::new(x.round() as i32, y.round() as i32);
        let mut base = 0;
        let sz = imgproc::get_text_size(s, font, scale, thick, &mut base)?;
        let r = thick as f64 + 2.0;
        let b = [at.x as f64 - r, at.y as f64 - sz.height as f64 - r, at.x as f64 + sz.width as f64 + r, at.y as f64 + base as f64 + r];
        // drawn on a stencil the size of its box, then onto each layer
        // (text is drawn on 8-bit images only)
        let (x0, y0) = (b[0].floor() as i32, b[1].floor() as i32);
        let (sw, sh) = ((b[2].ceil() as i32 - x0).max(1), (b[3].ceil() as i32 - y0).max(1));
        let mut st = util::zeros_u8(sh, sw)?;
        imgproc::put_text(&mut st, s, Point::new(at.x - x0, at.y - y0), font, scale, Scalar::all(255.0), thick, imgproc::LINE_8, false)?;
        let sv = util::bytes(&st)?;
        let w = self.w;
        let (pen, pc) = (self.pen.data_bytes_mut()?, self.pen_color.data_bytes_mut()?);
        let mut own = match self.owner.as_mut() {
            Some(o) if owner > 0 => Some(o.data_typed_mut::<u16>()?),
            _ => None,
        };
        for yy in 0..sh {
            for xx in 0..sw {
                let (px, py) = (x0 + xx, y0 + yy);
                if sv[(yy * sw + xx) as usize] == 0 || px < 0 || py < 0 || px >= w || py >= self.h {
                    continue;
                }
                let i = (py * w + px) as usize;
                pen[i] = 255;
                pc[3 * i..3 * i + 3].copy_from_slice(&color);
                if let Some(o) = own.as_mut() {
                    o[i] = owner;
                }
            }
        }
        Ok(b)
    }

    /// A round stamp; `noise` gives values in [-1, 1).
    pub fn stamp(&mut self, s: &StampSpec, noise: &mut dyn FnMut() -> f64) -> Result<()> {
        let half = (s.r * 1.1).ceil() as i32 + 4;
        let (x0, y0) = (s.cx.round() as i32 - half, s.cy.round() as i32 - half);
        let side = 2 * half;
        let mut shape = util::zeros_u8(side, side)?;
        let (cx, cy) = (s.cx - x0 as f64, s.cy - y0 as f64);
        let ring = |m: &mut Mat, r: f64, t: f64| -> Result<()> {
            let polys: Vec<Poly> = vec![(0..=96).map(|k| {
                let a = k as f64 / 96.0 * std::f64::consts::TAU;
                [cx + r * a.cos(), cy + r * a.sin()]
            })
            .collect()];
            let v: Vector<Vector<Point>> = polys.iter().map(pts8).collect();
            imgproc::polylines(m, &v, false, Scalar::all(255.0), t.max(1.0).round() as i32, imgproc::LINE_8, 3)?;
            Ok(())
        };
        ring(&mut shape, s.r, 0.035 * s.r)?;
        ring(&mut shape, 0.68 * s.r, 0.02 * s.r)?;
        let h = 0.17 * s.r;
        let hand = Hand { height: h, width: 0.62 * h, slant: 0.0, gap: 0.25, jitter: 0.0, thick: (0.022 * s.r).max(1.0).round() as i32 };
        let mut polys: Vec<Poly> = Vec::new();
        polys.extend(ring_text(&s.top, cx, cy, 0.76 * s.r, &hand, true));
        polys.extend(ring_text(&s.bottom, cx, cy, 0.95 * s.r - 2.0, &hand, false));
        let lh = 0.15 * s.r;
        let small = Hand { height: lh, width: 0.62 * lh, ..hand.clone() };
        let n = s.middle.len() as f64;
        for (i, line) in s.middle.iter().enumerate() {
            let w = glyphs::width(line, &small);
            let base = cy + (i as f64 - (n - 1.0) / 2.0) * 1.6 * lh + lh / 2.0;
            for c in glyphs::write(line, cx - w / 2.0, base, &small, &mut || 0.0) {
                polys.extend(c.polys);
            }
        }
        let v: Vector<Vector<Point>> = polys.iter().map(pts8).collect();
        imgproc::polylines(&mut shape, &v, false, Scalar::all(255.0), hand.thick, imgproc::LINE_8, 3)?;
        // patchy ink: a smooth field of strength, and worn specks
        let mut field = Mat::new_rows_cols_with_default(9, 9, core::CV_32FC1, Scalar::all(0.0))?;
        for v in field.data_typed_mut::<f32>()? {
            *v = (0.72 + 0.28 * noise()) as f32;
        }
        let mut smooth = Mat::default();
        imgproc::resize(&field, &mut smooth, Size::new(side, side), 0.0, 0.0, imgproc::INTER_CUBIC)?;
        let sh = util::bytes(&shape)?;
        let sm = smooth.data_typed::<f32>()?.to_vec();
        let (w, hh) = (self.w, self.h);
        let st = self.stamp.data_bytes_mut()?;
        let sc = self.stamp_color.data_bytes_mut()?;
        for yy in 0..side {
            let y = y0 + yy;
            if y < 0 || y >= hh {
                continue;
            }
            for xx in 0..side {
                let x = x0 + xx;
                let k = (yy * side + xx) as usize;
                if x < 0 || x >= w || sh[k] == 0 {
                    continue;
                }
                let speck = if noise() > 0.82 { 0.25 } else { 1.0 };
                let a = (s.opacity * (sm[k] as f64).clamp(0.0, 1.0) * speck * 255.0).round().clamp(0.0, 255.0) as u8;
                let i = (y * w + x) as usize;
                if a > st[i] {
                    st[i] = a;
                    sc[3 * i..3 * i + 3].copy_from_slice(&s.color);
                }
            }
        }
        Ok(())
    }

    /// The page: paper tinted by `tint` (BGR factors), stamp ink and pen
    /// multiplied over it; pen at `pen_opacity`.
    pub fn compose(&self, tint: [f64; 3], pen_opacity: f64) -> Result<Mat> {
        let mut out = Mat::new_rows_cols_with_default(self.h, self.w, core::CV_8UC3, Scalar::all(0.0))?;
        let paper = util::bytes(&self.paper)?;
        let (pen, pc) = (util::bytes(&self.pen)?, util::bytes(&self.pen_color)?);
        let (st, sc) = (util::bytes(&self.stamp)?, util::bytes(&self.stamp_color)?);
        let d = out.data_bytes_mut()?;
        for i in 0..paper.len() {
            let a = st[i] as f64 / 255.0;
            for c in 0..3 {
                let mut v = paper[i] as f64 * tint[c];
                if a > 0.0 {
                    v *= 1.0 - a + a * sc[3 * i + c] as f64 / 255.0;
                }
                if pen[i] != 0 {
                    v *= 1.0 - pen_opacity + pen_opacity * pc[3 * i + c] as f64 / 255.0;
                }
                d[3 * i + c] = v.round().clamp(0.0, 255.0) as u8;
            }
        }
        Ok(out)
    }

    /// The mask: print, pen and stamp-ink bits per pixel.
    pub fn bits(&self) -> Result<Mat> {
        let (pr, pen, st) = (util::bytes(&self.print)?, util::bytes(&self.pen)?, util::bytes(&self.stamp)?);
        let v: Vec<u8> = (0..pr.len())
            .map(|i| (if pr[i] != 0 { BIT_PRINT } else { 0 }) | (if pen[i] != 0 { BIT_PEN } else { 0 }) | (if st[i] >= STAMP_FROM { BIT_STAMP } else { 0 }))
            .collect();
        util::from_bytes(self.h, self.w, &v)
    }
}

/// A round stamp: centre and radius (px), ink colour (BGR) and opacity,
/// the text along its top and bottom and the lines across its middle.
#[derive(Clone, Debug)]
pub struct StampSpec {
    pub cx: f64,
    pub cy: f64,
    pub r: f64,
    pub color: [u8; 3],
    pub opacity: f64,
    pub top: String,
    pub bottom: String,
    pub middle: Vec<String>,
}

impl StampSpec {
    /// Its outline as a polygon of 64 points (as manifests record stamps).
    pub fn polygon(&self) -> Vec<[f64; 2]> {
        (0..64).map(|k| {
            let a = k as f64 / 64.0 * std::f64::consts::TAU;
            [self.cx + self.r * a.cos(), self.cy + self.r * a.sin()]
        })
        .collect()
    }
}

/// Text along a ring of radius `rb` round (cx, cy): along the top reading
/// clockwise with the letters' tops outward, or along the bottom reading
/// left to right with their tops inward. Centred on the top or bottom.
fn ring_text(s: &str, cx: f64, cy: f64, rb: f64, hand: &Hand, top: bool) -> Vec<Poly> {
    let total = glyphs::width(s, hand);
    let mid = if top { -std::f64::consts::FRAC_PI_2 } else { std::f64::consts::FRAC_PI_2 };
    let mut along = -total / 2.0;
    let mut out = Vec::new();
    for ch in s.chars() {
        let (g, adv) = glyphs::glyph(ch);
        let w = adv * hand.width;
        let centre = along + w / 2.0;
        let th = if top { mid + centre / rb } else { mid - centre / rb };
        let (o, t) = if top { ([th.cos(), th.sin()], [-th.sin(), th.cos()]) } else { ([th.cos(), th.sin()], [th.sin(), -th.cos()]) };
        for p in &g {
            out.push(
                p.iter()
                    .map(|&[u, v]| {
                        let radial = if top { rb + (1.0 - v) * hand.height } else { rb - (1.0 - v) * hand.height };
                        let tang = u * hand.width - w / 2.0;
                        [cx + o[0] * radial + t[0] * tang, cy + o[1] * radial + t[1] * tang]
                    })
                    .collect(),
            );
        }
        along += w + hand.gap * hand.width;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pen_and_stamp_bits_are_what_was_drawn() {
        let mut l = Layers::new(300, 200, 240, true).unwrap();
        let mut refimg = util::full_u8(50, 300, 250).unwrap();
        imgproc::line(&mut refimg, Point::new(0, 25), Point::new(299, 25), Scalar::all(0.0), 3, imgproc::LINE_8, 0).unwrap();
        l.place(&refimg, 0, 0).unwrap();
        l.pen(&[vec![[20.0, 100.0], [120.0, 100.0]]], 3, [150, 40, 20], 7).unwrap();
        let mut k = 0.0f64;
        let mut noise = || {
            k += 0.1;
            k.sin()
        };
        l.stamp(&StampSpec { cx: 220.0, cy: 120.0, r: 60.0, color: [170, 40, 120], opacity: 0.9, top: "PRESIDING OFFICER".into(), bottom: "KAMBU".into(), middle: vec!["09 AUG 2022".into()] }, &mut noise).unwrap();
        let bits = util::bytes(&l.bits().unwrap()).unwrap();
        let at = |x: usize, y: usize| bits[y * 300 + x];
        assert_eq!(at(50, 100), BIT_PEN);
        assert_eq!(at(50, 25) & BIT_PRINT, BIT_PRINT);
        assert_eq!(at(50, 150), 0);
        assert!(bits.iter().filter(|&&b| b & BIT_STAMP != 0).count() > 300);
        // the stamp stays inside its circle
        for (i, &b) in bits.iter().enumerate() {
            if b & BIT_STAMP != 0 {
                let (x, y) = ((i % 300) as f64, (i / 300) as f64);
                assert!((x - 220.0).hypot(y - 120.0) <= 63.0);
            }
        }
        let o = l.owner.as_ref().unwrap();
        assert_eq!(*o.at_2d::<u16>(100, 50).unwrap(), 7);
        let page = l.compose([1.0, 1.0, 1.0], 1.0).unwrap();
        let px = page.at_2d::<core::Vec3b>(100, 50).unwrap();
        assert!(px[0] > px[2], "blue pen: {:?}", px);
        assert!(page.at_2d::<core::Vec3b>(150, 50).unwrap()[0] == 240);
    }
}
