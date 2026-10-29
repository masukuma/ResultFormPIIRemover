//! Turning aligned zones into masks.
//!
//! partial    A random subset of the handwritten characters in each cell is
//!            hidden, the rest stays visible: enough to show that a real ID
//!            or phone number was written, not enough to use it.
//! signature  Each signature gets a black band through its middle: a reader
//!            sees that it is signed, not the signature.
//! blackout   Solid fill of the padded zone.
//!
//! Partial masking works on a whole table column at once, rectified to
//! template pixel scale so slant and perspective in the photo don't matter.
//! Characters are found once for the column and each is assigned to the row
//! holding its centre, so digits that dip into a neighbouring row are still
//! masked as part of their own row. Character slots come from connected
//! components when they look like separate glyphs, otherwise from dividing
//! the row's ink extent into `digits` equal slots. A cell with ink that can't
//! be segmented is hidden whole, at the writing's own width.
//!
//! What is not writing is taken out first: the printed rules and lines
//! (where the reference has them, or long and flat), dotted leaders, lines
//! officials strike through unused rows (found over the whole table, where
//! they are long), the security print's lettering and the striped or
//! embossed backgrounds some scans have. The tests in `tests/masking.rs`
//! hold one case for each.

use anyhow::Result;
use opencv::core::{self, Mat, Point, Point2f, Scalar, Vector, CV_32F};
use opencv::imgproc;
use opencv::prelude::*;
use std::sync::Arc;

use crate::pyrand::PyRandom;
use crate::templates::Zone;
use crate::util::{self, iround, median, quantile, round_even};

#[derive(Clone, Debug)]
pub struct MaskSettings {
    pub hide: usize,
    pub phone_tail: usize,
    pub vpad: f64,
    pub mask_vpad: f64,
    pub min_glyph_h: f64,
    pub min_glyph_h_out: f64,
    pub empty_ink: f64,
}

impl Default for MaskSettings {
    fn default() -> Self {
        MaskSettings { hide: 3, phone_tail: 6, vpad: 60.0, mask_vpad: 3.0, min_glyph_h: 0.4, min_glyph_h_out: 0.4, empty_ink: 0.01 }
    }
}

/// How far a column's search window may reach beyond its cells: its usual
/// search plus `margin` px (up, down), widened when its table couldn't be
/// snapped; but never past `top` / `bottom` (template y: printed labels
/// found just outside the table, such as its header or the next section's).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Reach {
    pub margin: (f64, f64),
    pub top: Option<f64>,
    pub bottom: Option<f64>,
}

impl Reach {
    pub fn margin(up: f64, down: f64) -> Self {
        Reach { margin: (up, down), ..Reach::default() }
    }

    /// px the window extends above the cells' top `y` and below their
    /// `bottom`, for a usual search of `above` / `below`. Negative where a
    /// fence lies inside the cells as placed (a table placed a little off):
    /// the window then starts or ends at the fence.
    pub fn clip(&self, y: f64, bottom: f64, above: f64, below: f64) -> (f64, f64) {
        let mut a = above + self.margin.0;
        let mut b = below + self.margin.1;
        if let Some(t) = self.top {
            a = a.min(y - t);
        }
        if let Some(t) = self.bottom {
            b = b.min(t - bottom);
        }
        (a, b)
    }

    /// The same fences, without the widening (for the leak checks, whose
    /// windows are wide already).
    pub fn fences(&self) -> Reach {
        Reach { margin: (0.0, 0.0), ..*self }
    }
}

/// Template -> scan point mapping: a homography, or a mesh that follows a
/// curled page (structure::Snap::map).
pub type Mapper = Arc<dyn Fn(&[[f64; 2]]) -> Vec<[f64; 2]> + Send + Sync>;

pub fn homography(h: [[f64; 3]; 3]) -> Mapper {
    Arc::new(move |pts: &[[f64; 2]]| pts.iter().map(|p| util::project(&h, *p)).collect())
}

pub fn shifted(f: Mapper, d: [f64; 2]) -> Mapper {
    Arc::new(move |pts: &[[f64; 2]]| {
        let moved: Vec<[f64; 2]> = pts.iter().map(|p| [p[0] + d[0], p[1] + d[1]]).collect();
        f(&moved)
    })
}

/// Fill a template-space quad, mapped edge by edge (densified, so a curved
/// mapping stays covered).
pub fn fill(img: &mut Mat, f: &Mapper, quad: &[[f64; 2]; 4]) -> Result<()> {
    let step = 8.0;
    let mut pts = Vec::new();
    for i in 0..4 {
        let (a, b) = (quad[i], quad[(i + 1) % 4]);
        let len = util::norm2(b[0] - a[0], b[1] - a[1]);
        let n = ((len / step) as usize).max(1);
        for k in 0..n {
            let t = k as f64 / n as f64;
            pts.push([a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t]);
        }
    }
    util::fill_poly(img, &f(&pts), 0.0)
}

fn quad(x0: f64, y0: f64, x1: f64, y1: f64) -> [[f64; 2]; 4] {
    [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
}

fn grow(q: [[f64; 2]; 4], dy: f64, dx: f64) -> [[f64; 2]; 4] {
    [[q[0][0] - dx, q[0][1] - dy], [q[1][0] + dx, q[1][1] - dy], [q[2][0] + dx, q[2][1] + dy], [q[3][0] - dx, q[3][1] + dy]]
}

fn offset(q: [[f64; 2]; 4], o: [f64; 2]) -> [[f64; 2]; 4] {
    q.map(|p| [p[0] + o[0], p[1] + o[1]])
}

pub fn blackout(img: &mut Mat, f: &Mapper, zone: &Zone, pad: f64) -> Result<()> {
    fill(img, f, &zone.padded(pad))
}

// --- image helpers -----------------------------------------------------------------------

/// A single-channel 8-bit image as a plain buffer.
#[derive(Clone)]
pub struct Img {
    pub w: usize,
    pub h: usize,
    pub d: Vec<u8>,
}

impl Img {
    pub fn new(w: usize, h: usize) -> Self {
        Img { w, h, d: vec![0; w * h] }
    }
    pub fn from_mat(m: &Mat) -> Result<Self> {
        Ok(Img { w: m.cols() as usize, h: m.rows() as usize, d: util::bytes(m)? })
    }
    pub fn to_mat(&self) -> Result<Mat> {
        util::from_bytes(self.h as i32, self.w as i32, &self.d)
    }
    #[inline]
    pub fn at(&self, x: usize, y: usize) -> u8 {
        self.d[y * self.w + x]
    }
    #[inline]
    pub fn set(&mut self, x: usize, y: usize, v: u8) {
        self.d[y * self.w + x] = v;
    }
    /// Rows y0..y1 (clipped) as a new image.
    pub fn rows(&self, y0: i64, y1: i64) -> Img {
        let (a, b) = (y0.clamp(0, self.h as i64) as usize, y1.clamp(0, self.h as i64) as usize);
        let b = b.max(a);
        Img { w: self.w, h: b - a, d: self.d[a * self.w..b * self.w].to_vec() }
    }
    pub fn crop(&self, x0: usize, y0: usize, x1: usize, y1: usize) -> Img {
        let (x1, y1) = (x1.min(self.w), y1.min(self.h));
        let (x0, y0) = (x0.min(x1), y0.min(y1));
        let mut out = Img::new(x1 - x0, y1 - y0);
        for y in y0..y1 {
            out.d[(y - y0) * out.w..(y - y0 + 1) * out.w].copy_from_slice(&self.d[y * self.w + x0..y * self.w + x1]);
        }
        out
    }
}

/// Strip pixel -> template -> scan maps for a rectified strip whose template
/// origin is (x0, y0), sampled per pixel so curled pages work.
fn strip_maps(f: &Mapper, x0: f64, y0: f64, sw: usize, sh: usize) -> Result<(Mat, Mat)> {
    let mut pts = Vec::with_capacity(sw * sh);
    for j in 0..sh {
        for i in 0..sw {
            pts.push([i as f64 + x0, j as f64 + y0]);
        }
    }
    let m = f(&pts);
    let mut mx = Mat::new_rows_cols_with_default(sh as i32, sw as i32, CV_32F, Scalar::all(0.0))?;
    let mut my = Mat::new_rows_cols_with_default(sh as i32, sw as i32, CV_32F, Scalar::all(0.0))?;
    {
        let dx = mx.data_typed_mut::<f32>()?;
        for (k, p) in m.iter().enumerate() {
            dx[k] = p[0] as f32;
        }
    }
    {
        let dy = my.data_typed_mut::<f32>()?;
        for (k, p) in m.iter().enumerate() {
            dy[k] = p[1] as f32;
        }
    }
    Ok((mx, my))
}

fn remap(src: &Mat, maps: &(Mat, Mat), interp: i32, border: i32) -> Result<Mat> {
    let mut out = Mat::default();
    imgproc::remap(src, &mut out, &maps.0, &maps.1, interp, border, Scalar::all(0.0), core::AlgorithmHint::ALGO_HINT_DEFAULT)?;
    Ok(out)
}

/// Linear `np.percentile` of the pixels of a float image.
fn percentiles(v: &[f64], qs: &[f64]) -> Vec<f64> {
    let mut s: Vec<f64> = v.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    qs.iter().map(|q| util::quantile_sorted(&s, q / 100.0)).collect()
}

/// Contrast stretch with a floor, then adaptive threshold (255 = ink).
fn stretch_threshold(strip: &Mat) -> Result<Mat> {
    let blur = util::gaussian(strip, 3)?;
    let c: Vec<f64> = util::bytes(&blur)?.iter().map(|&v| v as f64).collect();
    let p = percentiles(&c, &[1.0, 99.0]);
    let (p1, p99) = (p[0], p[1]);
    let d = (p99 - p1).max(60.0);
    let out: Vec<u8> = c.iter().map(|&v| ((v - p1) / d * 255.0).clamp(0.0, 255.0) as u8).collect();
    let m = util::from_bytes(strip.rows(), strip.cols(), &out)?;
    util::adaptive_inv(&m, 31, 45.0)
}

// --- strike lines --------------------------------------------------------------------------

pub type Strike = (f64, f64, f64, f64, f64); // x1, y1, x2, y2 (template px), stroke width

/// Lines officials draw through unused rows of a table, found over the whole
/// table (`rect`, template coordinates), where they are long.
pub fn find_strikes(gray: &Mat, f: &Mapper, rect: [f64; 4], row_h: f64) -> Result<Vec<Strike>> {
    let [x, y, w, h] = rect;
    let (sw, sh) = ((round_even(w) as usize).max(1), (round_even(h) as usize).max(1));
    let maps = strip_maps(f, x, y, sw, sh)?;
    let patch = remap(gray, &maps, imgproc::INTER_LINEAR, core::BORDER_REPLICATE)?;
    let bw = Img::from_mat(&stretch_threshold(&patch)?)?;
    let min_len = (3.0 * row_h).max(120.0);
    let mut segs = Vector::<core::Vec4i>::new();
    imgproc::hough_lines_p(&bw.to_mat()?, &mut segs, 1.0, std::f64::consts::PI / 720.0, 60, min_len as i32 as f64, 6.0)?;
    let mut out = Vec::new();
    for s in segs.iter() {
        let (x1, y1, x2, y2) = (s[0] as f64, s[1] as f64, s[2] as f64, s[3] as f64);
        let (dx, dy) = (x2 - x1, y2 - y1);
        let length = dx.hypot(dy);
        let mut ang = dy.atan2(dx).to_degrees().abs() % 180.0;
        ang = ang.min(180.0 - ang);
        if !(3.0..=80.0).contains(&ang) {
            continue;
        }
        let n = length as usize;
        let (nx, ny) = (-dy / length, dx / length);
        let step = (n / 60).max(1);
        let (mut widths, mut hits, mut samples) = (Vec::new(), 0usize, 0usize);
        let mut k = 0;
        while k < n {
            let t = if n == 1 { 0.0 } else if k == n - 1 { 1.0 } else { k as f64 * (1.0 / (n - 1) as f64) };
            let (px, py) = (x1 + t * dx, y1 + t * dy);
            samples += 1;
            let prof: Vec<bool> = (-8..=8)
                .map(|kk: i32| {
                    let yy = round_even(py + kk as f64 * ny) as i64;
                    let xx = round_even(px + kk as f64 * nx) as i64;
                    yy >= 0 && (yy as usize) < sh && xx >= 0 && (xx as usize) < sw && bw.at(xx as usize, yy as usize) > 0
                })
                .collect();
            if prof[8] {
                hits += 1;
                let (mut lo, mut hi) = (8usize, 8usize);
                while lo > 0 && prof[lo - 1] {
                    lo -= 1;
                }
                while hi < 16 && prof[hi + 1] {
                    hi += 1;
                }
                widths.push((hi - lo + 1) as f64);
            }
            k += step;
        }
        if (hits as f64) < 0.9 * samples as f64 || widths.is_empty() || median(&widths) > 0.25 * row_h {
            continue;
        }
        out.push((x1 + x, y1 + y, x2 + x, y2 + y, median(&widths)));
    }
    Ok(merge_strikes(out))
}

/// Hough returns one drawn line as many overlapping pieces: merge pieces
/// lying on the same line into one, spanning them all.
fn merge_strikes(segs: Vec<Strike>) -> Vec<Strike> {
    let mut segs = segs;
    segs.sort_by(|a, b| {
        let la = (a.2 - a.0).hypot(a.3 - a.1);
        let lb = (b.2 - b.0).hypot(b.3 - b.1);
        lb.partial_cmp(&la).unwrap()
    });
    let mut lines: Vec<[f64; 5]> = Vec::new();
    for (x1, y1, x2, y2, sw) in segs {
        let mut merged = false;
        for ln in lines.iter_mut() {
            let [ax, ay, bx, by, w] = *ln;
            let d = [bx - ax, by - ay];
            let dl = util::norm2(d[0], d[1]);
            let u = [d[0] / dl, d[1] / dl];
            let nrm = [-u[1], u[0]];
            let near = [(x1, y1), (x2, y2)].iter().all(|&(px, py)| util::dot2([px - ax, py - ay], nrm).abs() <= (2.0 * w).max(8.0));
            if near {
                let mut ts = vec![0.0, util::dot2(d, u)];
                for (px, py) in [(x1, y1), (x2, y2)] {
                    ts.push(util::dot2([px - ax, py - ay], u));
                }
                let a = ts.iter().cloned().fold(f64::INFINITY, f64::min);
                let b = ts.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                *ln = [ax + a * u[0], ay + a * u[1], ax + b * u[0], ay + b * u[1], w.max(sw)];
                merged = true;
                break;
            }
        }
        if !merged {
            lines.push([x1, y1, x2, y2, sw]);
        }
    }
    lines.into_iter().map(|l| (l[0], l[1], l[2], l[3], l[4])).collect()
}

fn erase_strikes(ink: &mut Mat, origin: [f64; 2], strikes: Option<&[Strike]>) -> Result<()> {
    for &(x1, y1, x2, y2, sw) in strikes.unwrap_or(&[]) {
        let p = Point::new(iround(x1 - origin[0]), iround(y1 - origin[1]));
        let q = Point::new(iround(x2 - origin[0]), iround(y2 - origin[1]));
        imgproc::line(ink, p, q, Scalar::all(0.0), iround(sw) + 4, imgproc::LINE_8, 0)?;
    }
    Ok(())
}

fn from_above(cells: &[Zone], bands: &[(f64, f64)], gy: f64, gh: f64) -> bool {
    cells[0].region.is_empty() && gy <= 1.0 && gy + gh < (bands[0].0 + bands[0].1) / 2.0
}

fn in_rows(y: f64, bands: &[(f64, f64)]) -> bool {
    bands.iter().any(|&(b0, b1)| b0 <= y && y < b1)
}

fn exclude_ink(ink: &mut Img, exclude: Option<&Mat>, maps: &(Mat, Mat)) -> Result<()> {
    if let Some(e) = exclude {
        let mut out = Mat::default();
        imgproc::remap(e, &mut out, &maps.0, &maps.1, imgproc::INTER_NEAREST, core::BORDER_CONSTANT, Scalar::all(0.0), core::AlgorithmHint::ALGO_HINT_DEFAULT)?;
        let ev = util::bytes(&out)?;
        for (i, v) in ev.iter().enumerate() {
            if *v > 0 {
                ink.d[i] = 0;
            }
        }
    }
    Ok(())
}

/// Erase printed dotted leaders ("ID Number: ..........") in place.
fn remove_dotted_leaders(ink: &mut Mat, row_h: f64, min_dots: usize) -> Result<()> {
    let (lab, st, cen) = util::components(ink)?;
    let small: Vec<usize> = (1..st.len())
        .filter(|&i| st[i][3] as f64 <= (0.15 * row_h).max(6.0) && st[i][2] as f64 <= (0.3 * row_h).max(10.0))
        .collect();
    if small.len() < min_dots {
        return Ok(());
    }
    let mut order = small.clone();
    order.sort_by(|&a, &b| cen[a][1].partial_cmp(&cen[b][1]).unwrap());
    let mut rows: Vec<Vec<usize>> = Vec::new();
    for i in order {
        let mut placed = false;
        for r in rows.iter_mut() {
            let m = util::np_mean(&r.iter().map(|&j| cen[j][1]).collect::<Vec<_>>());
            if (m - cen[i][1]).abs() <= 3.0 {
                r.push(i);
                placed = true;
                break;
            }
        }
        if !placed {
            rows.push(vec![i]);
        }
    }
    let mut erase: Vec<usize> = Vec::new();
    for mut r in rows {
        r.sort_by(|&a, &b| cen[a][0].partial_cmp(&cen[b][0]).unwrap());
        let mut run: Vec<usize> = vec![r[0]];
        let mut seq: Vec<Option<usize>> = r.iter().skip(1).map(|&x| Some(x)).collect();
        seq.push(None);
        for (idx, b) in seq.into_iter().enumerate() {
            let a = r[idx];
            let gap = b.map(|b| cen[b][0] - cen[a][0]);
            if let (Some(g), Some(bb)) = (gap, b) {
                if !run.is_empty()
                    && (run.len() < 2 || (g - (cen[run[1]][0] - cen[run[0]][0])).abs() <= 3.0)
                    && (4.0..=(0.5 * row_h).max(25.0)).contains(&g)
                {
                    run.push(bb);
                    continue;
                }
            }
            if run.len() >= min_dots {
                erase.extend(run.iter().cloned());
            }
            run = b.map(|b| vec![b]).unwrap_or_default();
        }
    }
    if !erase.is_empty() {
        let labels = util::labels_vec(&lab)?;
        let mut kill = vec![false; st.len()];
        for j in erase {
            kill[j] = true;
        }
        let d = ink.data_bytes_mut()?;
        for (k, &l) in labels.iter().enumerate() {
            if kill[l as usize] {
                d[k] = 0;
            }
        }
    }
    Ok(())
}

/// Dark marks of a rectified strip (255 = ink), printed or written.
pub fn binarize(strip: &Mat) -> Result<Mat> {
    destripe(&stretch_threshold(strip)?)
}

pub const STRIPED: f64 = 0.08;

/// Embossed / striped backgrounds: thin horizontal runs dropped where many
/// rows of a strip are mostly thin line.
fn destripe(bw: &Mat) -> Result<Mat> {
    let a = util::morph(bw, imgproc::MORPH_OPEN, &util::rect_kernel(12, 1)?)?;
    let b = util::morph(bw, imgproc::MORPH_OPEN, &util::rect_kernel(1, 3)?)?;
    let thin = util::bitand(&a, &util::bitnot(&b)?)?;
    let t = Img::from_mat(&thin)?;
    let mut striped_rows = 0usize;
    for y in 0..t.h {
        let n = t.d[y * t.w..(y + 1) * t.w].iter().filter(|&&v| v > 0).count();
        if n as f64 / t.w as f64 > 0.2 {
            striped_rows += 1;
        }
    }
    if (striped_rows as f64 / t.h.max(1) as f64) < STRIPED {
        return Ok(bw.try_clone()?);
    }
    util::subtract(bw, &thin)
}

/// Binary ink of a rectified strip, with rules, strike lines and the
/// security-print background removed.
pub fn ink_mask(strip: &Mat, row_h: f64, signature: bool) -> Result<Mat> {
    let bw = binarize(strip)?;
    let (h, w) = (bw.rows(), bw.cols());
    let rh = (row_h as i32).max(10);
    let rule_len = if signature { (0.8 * w as f64) as i32 } else { (w / 3).max(15) };
    let l1 = util::morph(&bw, imgproc::MORPH_OPEN, &util::rect_kernel(rule_len, 1)?)?;
    let l2 = util::morph(&bw, imgproc::MORPH_OPEN, &util::rect_kernel(1, (rh as f64 * 1.6) as i32)?)?;
    let lines = util::bitor(&l1, &l2)?;
    let mut ink = util::subtract(&bw, &util::dilate(&lines, &util::ones_kernel(3, 3)?)?)?;
    ink = util::morph(&ink, imgproc::MORPH_OPEN, &util::ones_kernel(2, 2)?)?;
    let mut segs = Vector::<core::Vec4i>::new();
    let min_len = (4 * rh).max(120) as f64;
    let lo = if signature {
        imgproc::hough_lines_p(&bw, &mut segs, 1.0, std::f64::consts::PI / 180.0, 40, min_len, 8.0)?;
        2.5
    } else {
        imgproc::hough_lines_p(&ink, &mut segs, 1.0, std::f64::consts::PI / 180.0, 40, min_len, 8.0)?;
        8.0
    };
    for s in segs.iter() {
        let ang = ((s[3] - s[1]) as f64).atan2((s[2] - s[0]) as f64).to_degrees().abs();
        let a = ang.min(180.0 - ang);
        if lo <= a && a <= 80.0 {
            imgproc::line(&mut ink, Point::new(s[0], s[1]), Point::new(s[2], s[3]), Scalar::all(0.0), 7, imgproc::LINE_8, 0)?;
        }
    }
    {
        let d = ink.data_bytes_mut()?;
        for y in 0..h as usize {
            for x in 0..(w as usize) {
                if x < 4 || x + 4 >= w as usize {
                    d[y * w as usize + x] = 0;
                }
            }
        }
    }
    remove_dotted_leaders(&mut ink, rh as f64, 6)?;
    let (lab, st, _) = util::components(&ink)?;
    let labels = util::labels_vec(&lab)?;
    let mut pts: Vec<Vec<Point2f>> = vec![Vec::new(); st.len()];
    for (k, &l) in labels.iter().enumerate() {
        if l > 0 {
            pts[l as usize].push(Point2f::new((k % w as usize) as f32, (k / w as usize) as f32));
        }
    }
    let mut kill = vec![false; st.len()];
    let max_theta = if signature { 10.0 } else { 50.0 };
    for i in 1..st.len() {
        let [x, _, cw, _, area] = st[i];
        if area < 8 {
            kill[i] = true;
            continue;
        }
        if cw <= 6 && (x < 14 || x + cw > w - 14) {
            kill[i] = true;
            continue;
        }
        let v: Vector<Point2f> = pts[i].iter().cloned().collect();
        let r = opencv::geometry::min_area_rect(&v)?;
        let (rw, rhh, ang) = (r.size.width as f64, r.size.height as f64, r.angle as f64);
        let theta = if rw >= rhh { ang.abs() } else { 90.0 - ang.abs() };
        if rw.min(rhh) <= 7.0 && rw.max(rhh) >= 25.0 && theta.min(180.0 - theta) < max_theta {
            kill[i] = true;
        }
    }
    {
        let d = ink.data_bytes_mut()?;
        for (k, &l) in labels.iter().enumerate() {
            if kill[l as usize] {
                d[k] = 0;
            }
        }
    }
    Ok(ink)
}

// --- characters -----------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Glyph {
    pub x0: i64,
    pub x1: i64,
    pub y0: i64,
    pub y1: i64,
}

fn merge_glyphs(glyphs: &[Glyph]) -> Vec<Glyph> {
    let mut g = glyphs.to_vec();
    g.sort_by_key(|g| g.x0);
    let mut out = vec![g[0]];
    for x in &g[1..] {
        let m = *out.last().unwrap();
        if x.x0 <= m.x1 - 2 {
            *out.last_mut().unwrap() = Glyph { x0: m.x0, x1: m.x1.max(x.x1), y0: m.y0.min(x.y0), y1: m.y1.max(x.y1) };
        } else {
            out.push(*x);
        }
    }
    out
}

fn blur_f32(img: &Img, x0: usize, y0: usize, x1: usize, y1: usize) -> Result<Mat> {
    let c = img.crop(x0, y0, x1, y1);
    let m = util::to_f32(&c.to_mat()?)?;
    let mut out = Mat::default();
    imgproc::gaussian_blur_def(&m, &mut out, core::Size::new(5, 5), 0.0)?;
    Ok(out)
}

/// Is this glyph printed text (a column header) rather than writing?
#[allow(clippy::too_many_arguments)]
fn looks_printed(ink: &Img, ref_ink: &Img, gx: i64, gy: i64, gw: i64, gh: i64) -> Result<bool> {
    let (pad, min_ncc, sx, sy) = (6i64, 0.6, 8i64, 56i64);
    let y0 = (gy - pad).max(0) as usize;
    let y1 = ((gy + gh + pad) as usize).min(ink.h);
    let x0 = (gx - pad).max(0) as usize;
    let x1 = ((gx + gw + pad) as usize).min(ink.w);
    let patch = blur_f32(ink, x0, y0, x1, y1)?;
    let ry0 = (y0 as i64 - sy).max(0) as usize;
    let ry1 = ((y1 as i64 + sy) as usize).min(ref_ink.h);
    let rx0 = (x0 as i64 - sx).max(0) as usize;
    let rx1 = ((x1 as i64 + sx) as usize).min(ref_ink.w);
    let area_img = ref_ink.crop(rx0, ry0, rx1, ry1);
    if (area_img.h as i32) < patch.rows() || (area_img.w as i32) < patch.cols() {
        return Ok(false);
    }
    let mut mean = Scalar::default();
    let mut sd = Scalar::default();
    core::mean_std_dev(&patch, &mut mean, &mut sd, &core::no_array())?;
    if sd[0] < 1.0 {
        return Ok(false);
    }
    let ink_n = ink.crop(x0, y0, x1, y1).d.iter().filter(|&&v| v > 0).count();
    let ref_n = area_img.d.iter().filter(|&&v| v > 0).count();
    if (ref_n as f64) < 0.3 * ink_n as f64 {
        return Ok(false);
    }
    let area = blur_f32(&area_img, 0, 0, area_img.w, area_img.h)?;
    let mut res = Mat::default();
    imgproc::match_template(&area, &patch, &mut res, imgproc::TM_CCOEFF_NORMED, &core::no_array())?;
    let mut mx = 0.0;
    core::min_max_loc(&res, None, Some(&mut mx), None, None, &core::no_array())?;
    Ok(mx >= min_ncc)
}

fn glyph_center(g: &Glyph) -> f64 {
    (g.y0 + g.y1) as f64 / 2.0
}

/// Group glyphs whose vertical centres lie within half a row of the line's
/// running median.
fn text_lines(glyphs: &[Glyph], row_h: f64) -> Vec<Vec<Glyph>> {
    let mut g = glyphs.to_vec();
    g.sort_by(|a, b| glyph_center(a).partial_cmp(&glyph_center(b)).unwrap());
    let mut lines: Vec<Vec<Glyph>> = Vec::new();
    for x in g {
        let cy = glyph_center(&x);
        if let Some(last) = lines.last_mut() {
            let m = median(&last.iter().map(glyph_center).collect::<Vec<_>>());
            if (cy - m).abs() < 0.5 * row_h {
                last.push(x);
                continue;
            }
        }
        lines.push(vec![x]);
    }
    lines
}

/// One box per written character, or None if the line can't be segmented.
pub fn char_slots(glyphs: &[Glyph], digits: usize, row_w: f64) -> Option<Vec<Glyph>> {
    let merged = merge_glyphs(glyphs);
    let left = merged[0].x0;
    let right = merged.iter().map(|g| g.x1).max().unwrap();
    if ((right - left) as f64) < 0.15 * row_w {
        return None;
    }
    let pitch = (right - left) as f64 / digits as f64;
    let mut split = Vec::new();
    for g in &merged {
        let n = (round_even((g.x1 - g.x0) as f64 / pitch) as i64).max(1);
        let step = (g.x1 - g.x0) as f64 / n as f64;
        for i in 0..n {
            split.push(Glyph {
                x0: round_even(g.x0 as f64 + i as f64 * step) as i64,
                x1: round_even(g.x0 as f64 + (i + 1) as f64 * step) as i64,
                y0: g.y0,
                y1: g.y1,
            });
        }
    }
    if (split.len() as i64 - digits as i64).abs() <= 1 || (digits >= 10 && split.len() >= digits) {
        return Some(split);
    }
    let top = merged.iter().map(|g| g.y0).min().unwrap();
    let bottom = merged.iter().map(|g| g.y1).max().unwrap();
    Some(
        (0..digits)
            .map(|i| Glyph {
                x0: round_even(left as f64 + i as f64 * pitch) as i64,
                x1: round_even(left as f64 + (i + 1) as f64 * pitch) as i64,
                y0: top,
                y1: bottom,
            })
            .collect(),
    )
}

/// Random k of n slots to hide, leaving no more than `max_run` visible slots
/// side by side where possible.
pub fn pick_hidden(n: usize, k: usize, rng: &mut PyRandom, max_run: usize) -> Vec<usize> {
    let k = k.min(n);
    let mut best_hidden: Option<Vec<usize>> = None;
    for _ in 0..200 {
        let hidden = rng.sample_range(n, k);
        let set: std::collections::HashSet<usize> = hidden.iter().cloned().collect();
        let (mut run, mut best) = (0usize, 0usize);
        for i in 0..n {
            run = if set.contains(&i) { 0 } else { run + 1 };
            best = best.max(run);
        }
        let mut sorted = hidden.clone();
        sorted.sort();
        if best <= max_run {
            return sorted;
        }
        if best_hidden.as_ref().map(|h| h.is_empty()).unwrap_or(true) {
            best_hidden = Some(sorted);
        }
    }
    best_hidden.unwrap_or_default()
}

/// Which character slots to hide: phone, `hide` of the last `phone_tail`
/// digits (the prefix stays visible); id, `hide` of all characters.
pub fn choose_hidden(n: usize, rule: &str, rng: &mut PyRandom, s: &MaskSettings) -> Vec<usize> {
    if rule == "phone" {
        let start = n.saturating_sub(s.phone_tail);
        return pick_hidden(n - start, s.hide, rng, 2).into_iter().map(|i| start + i).collect();
    }
    pick_hidden(n, s.hide, rng, 2)
}

fn ink_span(ink: &Img, y0: f64, y1: f64, pad: f64) -> Result<Option<(f64, f64)>> {
    let band = ink.rows((y0 as i64).max(0), (y1.ceil() as i64).max(0));
    if band.h == 0 {
        return Ok(None);
    }
    let b: Vec<u8> = band.d.iter().map(|&v| (v > 0) as u8).collect();
    let (_, st, _) = util::components(&util::from_bytes(band.h as i32, band.w as i32, &b)?)?;
    let big: Vec<&[i32; 5]> = st[1..].iter().filter(|s| s[4] > 30).collect();
    if big.is_empty() {
        return Ok(None);
    }
    let x0 = big.iter().map(|s| s[0]).min().unwrap() as f64 - pad;
    let x1 = big.iter().map(|s| s[0] + s[2]).max().unwrap() as f64 + pad;
    Ok(Some((x0, x1)))
}

fn ref_crop(reference: &Mat, origin: [f64; 2], h: usize, w: usize) -> Result<Img> {
    let r = Img::from_mat(reference)?;
    let mut out = Img { w, h, d: vec![255; w * h] };
    let (x0, y0) = (round_even(origin[0]) as i64, round_even(origin[1]) as i64);
    for y in 0..h as i64 {
        let ry = y0 + y;
        if ry < 0 || ry >= r.h as i64 {
            continue;
        }
        for x in 0..w as i64 {
            let rx = x0 + x;
            if rx < 0 || rx >= r.w as i64 {
                continue;
            }
            out.d[(y as usize) * w + x as usize] = r.at(rx as usize, ry as usize);
        }
    }
    Ok(out)
}

fn median_row_h(cells: &[Zone]) -> f64 {
    median(&cells.iter().map(|z| z.rect[3]).collect::<Vec<_>>())
}

fn sorted_cells(cells: &[Zone]) -> Vec<Zone> {
    let mut c = cells.to_vec();
    c.sort_by(|a, b| a.rect[1].partial_cmp(&b.rect[1]).unwrap());
    c
}

fn row_of(cy: f64, bands: &[(f64, f64)]) -> usize {
    let d: Vec<f64> = bands
        .iter()
        .map(|&(b0, b1)| if b0 <= cy && cy < b1 { 0.0 } else { (cy - b0).abs().min((cy - b1).abs()) })
        .collect();
    util::argmin(&d)
}

/// Hide a random subset of characters in every cell of one table column.
#[allow(clippy::too_many_arguments)]
pub fn partial_column(
    img: &mut Mat,
    gray: &Mat,
    f: &Mapper,
    cells: &[Zone],
    s: &MaskSettings,
    rng: &mut PyRandom,
    reach: Reach,
    reference: Option<&Mat>,
    strikes: Option<&[Strike]>,
    exclude: Option<&Mat>,
) -> Result<Vec<serde_json::Value>> {
    use serde_json::json;
    let cells = sorted_cells(cells);
    let [x, y, w, _] = cells[0].rect;
    let bottom = cells.last().unwrap().rect[1] + cells.last().unwrap().rect[3];
    let (above, below) = cells[0].search.map(|s| (s[0], s[1])).unwrap_or((s.vpad, s.vpad));
    let (vpad, below) = reach.clip(y, bottom, above, below);
    let (sw, sh) = ((round_even(w) as usize).max(1), (round_even(bottom - y + vpad + below) as usize).max(1));
    let oy = y - vpad;
    let maps = strip_maps(f, x, oy, sw, sh)?;
    let strip = remap(gray, &maps, imgproc::INTER_LINEAR, core::BORDER_REPLICATE)?;
    let to_template = [x, oy];
    let row_h = median_row_h(&cells);
    let mut ink_m = ink_mask(&strip, row_h, false)?;
    erase_strikes(&mut ink_m, to_template, strikes)?;
    let mut ink = Img::from_mat(&ink_m)?;
    exclude_ink(&mut ink, exclude, &maps)?;

    let bands: Vec<(f64, f64)> = cells.iter().map(|z| (z.rect[1] - y + vpad, z.rect[1] - y + vpad + z.rect[3])).collect();
    let (_, st, _) = util::components(&ink.to_mat()?)?;
    let r_ink = match reference {
        Some(r) => {
            let rc = ref_crop(r, [round_even(x), round_even(oy)], sh, sw)?;
            Some(Img::from_mat(&ink_mask(&rc.to_mat()?, row_h, false)?)?)
        }
        None => None,
    };
    let (mut found, mut printed) = (Vec::new(), Vec::new());
    for stv in &st[1..] {
        let [gx, gy, gw, gh, area] = stv.map(|v| v as i64);
        if (gh as f64) < s.min_glyph_h * row_h || area <= 30 {
            continue;
        }
        if (gh as f64) < s.min_glyph_h_out * row_h && !in_rows(gy as f64 + gh as f64 / 2.0, &bands) {
            continue;
        }
        if from_above(&cells, &bands, gy as f64, gh as f64) {
            continue;
        }
        let g = Glyph { x0: gx, x1: gx + gw, y0: gy, y1: gy + gh };
        if let Some(ri) = &r_ink {
            if looks_printed(&ink, ri, gx, gy, gw, gh)? {
                printed.push(g);
                continue;
            }
        }
        found.push(g);
    }
    let mut per_row: Vec<Vec<serde_json::Value>> = vec![Vec::new(); cells.len()];
    for line in text_lines(&found, row_h) {
        let cy = median(&line.iter().map(|g| g.y0 as f64 + 0.35 * (g.y1 - g.y0) as f64).collect::<Vec<_>>());
        let i = row_of(cy, &bands);
        let top = line.iter().map(|g| g.y0).min().unwrap() as f64;
        let bot = line.iter().map(|g| g.y1).max().unwrap() as f64;
        let Some(slots) = char_slots(&line, cells[i].digits, w) else {
            let (x0, x1) = ink_span(&ink, top, bot, 0.25 * row_h)?.unwrap_or((0.0, w));
            let (x0, x1) = (x0.max(0.0), x1.min(w));
            fill(img, f, &offset(grow(quad(x0, top, x1, bot), s.mask_vpad + 4.0, 0.0), to_template))?;
            per_row[i].push(json!({"mode": "full"}));
            continue;
        };
        let hidden = choose_hidden(slots.len(), &cells[i].rule, rng, s);
        let mt = median(&line.iter().map(|g| g.y0 as f64).collect::<Vec<_>>());
        let mb = median(&line.iter().map(|g| g.y1 as f64).collect::<Vec<_>>());
        let pitch = (slots.last().unwrap().x1 - slots[0].x0) as f64 / slots.len() as f64;
        let mut lo = vec![slots[0].x0 as f64 - pitch / 2.0];
        let mut hi = Vec::new();
        for p in slots.windows(2) {
            let m = (p[0].x1 + p[1].x0) as f64 / 2.0;
            lo.push(m);
            hi.push(m);
        }
        hi.push(slots.last().unwrap().x1 as f64 + pitch / 2.0);
        let h = mb - mt;
        for &j in &hidden {
            let y0 = (slots[j].y0 as f64).min(mt) - 0.15 * h - 2.0;
            let y1 = (slots[j].y1 as f64).max(mb) + 0.1 * h + 2.0;
            fill(img, f, &offset(quad(lo[j], y0, hi[j], y1), to_template))?;
        }
        per_row[i].push(json!({"mode": "partial", "slots": slots.len(), "hidden": hidden}));
    }

    // rows with no line of writing: empty, or ink we couldn't make glyphs of
    let mut stray = ink.clone();
    for g in found.iter().chain(printed.iter()) {
        for yy in g.y0.max(0)..g.y1.min(stray.h as i64) {
            for xx in g.x0.max(0)..g.x1.min(stray.w as i64) {
                stray.set(xx as usize, yy as usize, 0);
            }
        }
    }
    let mut report = Vec::new();
    for ((z, &(b0, b1)), lines) in cells.iter().zip(&bands).zip(per_row) {
        if !lines.is_empty() {
            let all_full = lines.iter().all(|l| l["mode"] == "full");
            let partial: Vec<serde_json::Value> = lines
                .iter()
                .filter(|l| l["mode"] == "partial")
                .map(|l| json!({"slots": l["slots"], "hidden": l["hidden"]}))
                .collect();
            report.push(json!({"zone": z.name, "rule": z.rule, "mode": if all_full {"full"} else {"partial"}, "lines": partial}));
            continue;
        }
        let band = stray.rows(b0 as i64, b1 as i64);
        let share = if b1 > b0 && band.h > 0 {
            band.d.iter().map(|&v| v as f64).sum::<f64>() / band.d.len() as f64 / 255.0
        } else {
            0.0
        };
        if share >= 3.0 * s.empty_ink {
            match ink_span(&stray, b0, b1, 0.25 * row_h)? {
                None => blackout(img, f, z, s.mask_vpad + 4.0)?,
                Some((x0, x1)) => {
                    let (x0, x1) = (x0.max(0.0), x1.min(w));
                    fill(img, f, &offset(grow(quad(x0, b0, x1, b1), s.mask_vpad + 4.0, s.mask_vpad + 4.0), to_template))?;
                }
            }
            report.push(json!({"zone": z.name, "mode": "full"}));
        } else {
            report.push(json!({"zone": z.name, "mode": "empty"}));
        }
    }
    Ok(report)
}

/// Independent leak check on the redacted image: re-read the column and look
/// for `max_run` or more written characters in a row with no mask between.
#[allow(clippy::too_many_arguments)]
pub fn verify_column(
    out: &Mat,
    f: &Mapper,
    cells: &[Zone],
    reference: Option<&Mat>,
    s: &MaskSettings,
    reach: Reach,
    strikes: Option<&[Strike]>,
    exclude: Option<&Mat>,
) -> Result<String> {
    let max_run = if cells[0].rule == "phone" { 8 } else { 5 };
    let window = cells[0].search.map(|s| (s[0] + 8.0, s[1] + 8.0)).unwrap_or((60.0, 110.0));
    let cells = sorted_cells(cells);
    let [x, y, w, _] = cells[0].rect;
    let bottom = cells.last().unwrap().rect[1] + cells.last().unwrap().rect[3];
    let window = reach.fences().clip(y, bottom, window.0, window.1);
    let (sw, sh) = ((round_even(w) as usize).max(1), (round_even(bottom - y + window.0 + window.1) as usize).max(1));
    let oy = y - window.0;
    let maps = strip_maps(f, x, oy, sw, sh)?;
    let g = util::gray(out)?;
    let strip = Img::from_mat(&remap(&g, &maps, imgproc::INTER_NEAREST, core::BORDER_REPLICATE)?)?;
    let zero: Vec<u8> = strip.d.iter().map(|&v| (v == 0) as u8).collect();
    let masked = Img::from_mat(&util::dilate(&util::from_bytes(sh as i32, sw as i32, &zero)?, &util::ones_kernel(5, 5)?)?)?;
    let row_h = median_row_h(&cells);
    let cleaned: Vec<u8> = strip.d.iter().zip(&masked.d).map(|(&v, &m)| if m > 0 { 255 } else { v }).collect();
    let mut ink_m = ink_mask(&util::from_bytes(sh as i32, sw as i32, &cleaned)?, row_h, false)?;
    erase_strikes(&mut ink_m, [x, oy], strikes)?;
    let mut ink = Img::from_mat(&ink_m)?;
    exclude_ink(&mut ink, exclude, &maps)?;
    let r_ink = match reference {
        Some(r) => {
            let rc = ref_crop(r, [round_even(x), round_even(oy)], sh, sw)?;
            Some(Img::from_mat(&ink_mask(&rc.to_mat()?, row_h, false)?)?)
        }
        None => None,
    };
    let (_, st, _) = util::components(&ink.to_mat()?)?;
    let bands: Vec<(f64, f64)> = cells.iter().map(|z| (z.rect[1] - oy, z.rect[1] - oy + z.rect[3])).collect();
    let mut glyphs = Vec::new();
    for stv in &st[1..] {
        let [a, c, b, d, area] = stv.map(|v| v as i64);
        if (d as f64) < s.min_glyph_h * row_h || area <= 30 {
            continue;
        }
        if !((d as f64) >= s.min_glyph_h_out * row_h || in_rows(c as f64 + d as f64 / 2.0, &bands)) {
            continue;
        }
        if from_above(&cells, &bands, c as f64, d as f64) {
            continue;
        }
        if let Some(ri) = &r_ink {
            if looks_printed(&ink, ri, a, c, b, d)? {
                continue;
            }
        }
        glyphs.push(Glyph { x0: a, x1: a + b, y0: c, y1: c + d });
    }
    for line in text_lines(&glyphs, row_h) {
        let line = merge_glyphs(&line);
        let top = line.iter().map(|g| g.y0).min().unwrap();
        let bot = line.iter().map(|g| g.y1).max().unwrap();
        let (mut run, mut best) = (1, 1);
        for p in line.windows(2) {
            let (a, b) = (p[0], p[1]);
            let xe = (a.x1 + 1).max(b.x0);
            let mut any = false;
            'gap: for yy in top.max(0)..bot.min(masked.h as i64) {
                for xx in a.x1.max(0)..xe.min(masked.w as i64) {
                    if masked.at(xx as usize, yy as usize) > 0 {
                        any = true;
                        break 'gap;
                    }
                }
            }
            run = if any { 1 } else { run + 1 };
            best = best.max(run);
        }
        if best >= max_run {
            return Ok(format!("{}: {} written characters in a row left unmasked", cells[0].key(), best));
        }
    }
    Ok(String::new())
}

/// How much printed vertical rule there is at each x across a column and
/// `margin` px either side (template px from the column's left edge minus
/// `margin`), counted over the column's rows; and the rows' height in px.
fn rule_profile(gray: &Mat, f: &Mapper, cells: &[Zone], margin: f64) -> Result<(Vec<f64>, usize)> {
    let [x, y, w, _] = cells[0].rect;
    let bottom = cells.last().unwrap().rect[1] + cells.last().unwrap().rect[3];
    let (sw, sh) = ((round_even(w + 2.0 * margin) as usize).max(1), (round_even(bottom - y) as usize).max(1));
    let maps = strip_maps(f, x - margin, y, sw, sh)?;
    let strip = remap(gray, &maps, imgproc::INTER_LINEAR, core::BORDER_REPLICATE)?;
    let bw = util::adaptive_inv(&util::gaussian(&strip, 3)?, 31, 6.0)?;
    let row_h = median_row_h(cells);
    let rules = Img::from_mat(&util::morph(&bw, imgproc::MORPH_OPEN, &util::rect_kernel(1, row_h as i32)?)?)?;
    let mut prof = Mat::new_rows_cols_with_default(1, sw as i32, CV_32F, Scalar::all(0.0))?;
    {
        let p = prof.data_typed_mut::<f32>()?;
        for xx in 0..sw {
            p[xx] = (0..sh).filter(|&yy| rules.at(xx, yy) > 0).count() as f32;
        }
    }
    let mut smooth = Mat::default();
    imgproc::gaussian_blur_def(&prof, &mut smooth, core::Size::new(5, 1), 0.0)?;
    Ok((smooth.data_typed::<f32>()?.iter().map(|&v| v as f64).collect(), sh))
}

/// A column the edge check found off its printed rules, placed again: moved
/// across to where its printed `header` is (searched for a column's width
/// either side), then each edge onto the printed rule nearest it, when one
/// shows within a few px (a table photographed narrower or wider than the
/// page around it). Returns the new mapper, the column's x as `f` sees it
/// for each template x (`a * x + b`), and how well the header matched; None
/// if the header isn't found.
pub fn place_by_header(gray: &Mat, f: &Mapper, reference: &Mat, header: [f64; 4], cells: &[Zone]) -> Result<Option<(Mapper, [f64; 2], f64)>> {
    let cells = sorted_cells(cells);
    let [x0, _, w, _] = cells[0].rect;
    let (d, ncc) = find_label(gray, f, reference, header, w.max(120.0), 120.0)?;
    let Some(d) = d else { return Ok(None) };
    let (tol, margin) = (25i64, 40.0);
    let (profile, sh) = rule_profile(gray, &shifted(f.clone(), [d[0], 0.0]), &cells, margin)?;
    let near = |e: f64| -> f64 {
        let mut best = (0.1 * sh as f64, 0i64); // a rule must show over a tenth of the rows
        for k in -tol..=tol {
            let v = profile.get((e + k as f64).round() as usize).copied().unwrap_or(0.0);
            if v > best.0 || (v == best.0 && k.abs() < best.1.abs()) {
                best = (v, k);
            }
        }
        best.1 as f64
    };
    let (l, r) = (x0 + d[0] + near(margin), x0 + w + d[0] + near(margin + w));
    let a = (r - l) / w;
    if !(0.85..=1.15).contains(&a) {
        return Ok(None);
    }
    let b = l - a * x0;
    let f = f.clone();
    let g: Mapper = Arc::new(move |pts: &[[f64; 2]]| f(&pts.iter().map(|p| [a * p[0] + b, p[1]]).collect::<Vec<_>>()));
    Ok(Some((g, [a, b], ncc)))
}

/// Is the column where we think it is? The printed vertical rules at its
/// edges are compared with any found elsewhere near it.
pub fn check_column_edges(gray: &Mat, f: &Mapper, cells: &[Zone], require: bool) -> Result<String> {
    let (tol, margin) = (14.0f64, 40.0f64);
    let cells = sorted_cells(cells);
    let w = cells[0].rect[2];
    let (profile, sh) = rule_profile(gray, f, &cells, margin)?;
    let sw = profile.len();
    let edges = [margin, margin + w];
    let mut near = vec![false; sw];
    for e in edges {
        for k in ((e - tol).max(0.0) as usize)..(((e + tol + 1.0).min(sw as f64)) as usize) {
            near[k] = true;
        }
    }
    let elsewhere = (0..sw).filter(|&k| !near[k]).map(|k| profile[k]).fold(0.0, f64::max);
    let name = cells[0].key();
    for (e, label) in edges.iter().zip(["left", "right"]) {
        let a = ((e - tol).max(0.0)) as usize;
        let b = ((e + tol + 1.0).min(sw as f64)) as usize;
        let at = profile[a..b].iter().cloned().fold(0.0, f64::max);
        if elsewhere >= 0.15 * sh as f64 && at < 0.6 * elsewhere {
            return Ok(format!("{}: printed rule found off the column's {} edge (table placed wrong)", name, label));
        }
        if require && at < 0.1 * sh as f64 {
            return Ok(format!("{}: can't see the printed rule at the column's {} edge", name, label));
        }
    }
    Ok(String::new())
}

// --- signatures ------------------------------------------------------------------------------

pub const SIG_OVERRUN: f64 = 24.0;
pub const SIG_MIN_INK: usize = 40;
pub const SIG_JOIN: i32 = 9;
pub const SIG_MIN_H: i64 = 8;
pub const SIG_BELOW: f64 = 30.0;
pub const SIG_BAND_H: f64 = 0.5;
pub const SIG_BAND_MIN: f64 = 10.0;
pub const SIG_CHUNK: i64 = 24;
pub const SIG_BAND_PAD: f64 = 2.0;
/// A signature hidden whole is covered slice by slice this wide, each slice
/// from the top to the bottom of the ink in it; ink more than `SIG_ENV_GAP`
/// apart within a slice is covered piece by piece.
pub const SIG_ENV_CHUNK: i64 = 8;
pub const SIG_ENV_GAP: i64 = 10;

pub struct ColumnStrip {
    pub strip: Mat,
    pub origin: [f64; 2],
    pub bands: Vec<(f64, f64)>,
    pub cells: Vec<Zone>,
}

/// Rectified strip covering a table column (template scale), widened by
/// `extra` px each side and `above` / `below` px.
pub fn column_strip(gray: &Mat, f: &Mapper, cells: &[Zone], above: f64, below: f64, extra: f64) -> Result<ColumnStrip> {
    let cells = sorted_cells(cells);
    let [x, y, w, _] = cells[0].rect;
    let bottom = cells.last().unwrap().rect[1] + cells.last().unwrap().rect[3];
    let (sw, sh) = ((round_even(w + 2.0 * extra) as usize).max(1), (round_even(bottom - y + above + below) as usize).max(1));
    let (ox, oy) = (x - extra, y - above);
    let maps = strip_maps(f, ox, oy, sw, sh)?;
    let strip = remap(gray, &maps, imgproc::INTER_LINEAR, core::BORDER_REPLICATE)?;
    let bands = cells.iter().map(|z| (z.rect[1] - oy, z.rect[1] - oy + z.rect[3])).collect();
    Ok(ColumnStrip { strip, origin: [ox, oy], bands, cells })
}

/// Round stamps in a signature column: their rings are erased (a ring would
/// join everything near it into one "signature") and returned as (cx, cy, r).
fn remove_stamp_rings(ink: &mut Img) -> Result<Vec<(i64, i64, i64)>> {
    let blur = util::gaussian(&ink.to_mat()?, 5)?;
    let mut circles = Vector::<core::Vec3f>::new();
    imgproc::hough_circles(&blur, &mut circles, imgproc::HOUGH_GRADIENT, 1.5, 150.0, 100.0, 40.0, 55, 130)?;
    let mut found = Vec::new();
    for c in circles.iter() {
        let (cx, cy, r) = (round_even(c[0] as f64) as i32, round_even(c[1] as f64) as i32, round_even(c[2] as f64) as i32);
        let mut ring = util::zeros_u8(ink.h as i32, ink.w as i32)?;
        imgproc::circle(&mut ring, Point::new(cx, cy), r, Scalar::all(255.0), 9, imgproc::LINE_8, 0)?;
        let rv = util::bytes(&ring)?;
        let on: Vec<usize> = (0..rv.len()).filter(|&k| rv[k] > 0).collect();
        if on.is_empty() {
            continue;
        }
        let inked = on.iter().filter(|&&k| ink.d[k] > 0).count() as f64 / on.len() as f64;
        if inked > 0.35 {
            for k in on {
                ink.d[k] = 0;
            }
            found.push((cx as i64, cy as i64, r as i64));
        }
    }
    Ok(found)
}

fn inside_stamp(xs: &[i64], ys: &[i64], stamps: &[(i64, i64, i64)]) -> bool {
    for &(cx, cy, r) in stamps {
        let lim = (0.95 * r as f64).powi(2);
        let inside = xs.iter().zip(ys).filter(|&(&x, &y)| (((x - cx).pow(2) + (y - cy).pow(2)) as f64) < lim).count();
        if inside as f64 / xs.len() as f64 >= 0.9 {
            return true;
        }
    }
    false
}

fn nanmedian(v: &[f64]) -> f64 {
    let w: Vec<f64> = v.iter().cloned().filter(|x| !x.is_nan()).collect();
    median(&w)
}

/// Boxes (x0, y0, x1, y1) hiding the middle of one signature's ink: a band
/// along it, its tops and bottoms left showing.
pub fn middle_band(xs: &[i64], ys: &[i64]) -> Vec<(i64, i64, i64, i64)> {
    let x0 = *xs.iter().min().unwrap();
    let x1 = *xs.iter().max().unwrap() + 1;
    let starts: Vec<i64> = (x0..x1).step_by(SIG_CHUNK as usize).collect();
    let own: Vec<Vec<f64>> = starts
        .iter()
        .map(|&a| xs.iter().zip(ys).filter(|&(&x, _)| x >= a && x < a + SIG_CHUNK).map(|(_, &y)| y as f64).collect())
        .collect();
    let heights: Vec<f64> = own.iter().filter(|k| k.len() >= 4).map(|k| quantile(k, 0.95) - quantile(k, 0.05) + 1.0).collect();
    if heights.is_empty() {
        return vec![(x0, *ys.iter().min().unwrap(), x1, *ys.iter().max().unwrap() + 1)];
    }
    let half = SIG_BAND_MIN.max(SIG_BAND_H * quantile(&heights, 0.75)) / 2.0;
    let mids: Vec<f64> = own.iter().map(|k| if k.len() >= 4 { median(k) } else { f64::NAN }).collect();
    let mut boxes = Vec::new();
    for (i, (&a, k)) in starts.iter().zip(&own).enumerate() {
        if k.len() < 4 {
            continue;
        }
        let c_near = nanmedian(&mids[i.saturating_sub(2)..(i + 3).min(mids.len())]);
        let own_mid = if (mids[i] - c_near).abs() > half / 2.0 { mids[i] } else { c_near };
        let (lo, hi) = (own_mid.min(c_near), own_mid.max(c_near));
        boxes.push((a, (lo - half).floor() as i64, (a + SIG_CHUNK).min(x1), (hi + half).ceil() as i64));
    }
    boxes
}

/// Boxes (x0, y0, x1, y1) covering all of one signature's ink, narrow slice
/// by slice: every stroke is hidden, and the paper beside and between its
/// strokes (the space after a phone number, the gap to the next signature)
/// stays visible.
pub fn envelope(xs: &[i64], ys: &[i64]) -> Vec<(i64, i64, i64, i64)> {
    let x0 = *xs.iter().min().unwrap();
    let x1 = *xs.iter().max().unwrap() + 1;
    let mut boxes = Vec::new();
    for a in (x0..x1).step_by(SIG_ENV_CHUNK as usize) {
        let mut col: Vec<i64> = xs.iter().zip(ys).filter(|&(&x, _)| x >= a && x < a + SIG_ENV_CHUNK).map(|(_, &y)| y).collect();
        if col.is_empty() {
            continue;
        }
        col.sort_unstable();
        let mut lo = col[0];
        for w in col.windows(2) {
            if w[1] - w[0] > SIG_ENV_GAP {
                boxes.push((a, lo, (a + SIG_ENV_CHUNK).min(x1), w[0] + 1));
                lo = w[1];
            }
        }
        boxes.push((a, lo, (a + SIG_ENV_CHUNK).min(x1), col[col.len() - 1] + 1));
    }
    boxes
}

pub struct SigBlob {
    pub x0: i64,
    pub y0: i64,
    pub x1: i64,
    pub y1: i64,
    pub n: usize,
    pub band: Vec<(i64, i64, i64, i64)>,
}

/// Ink running down a column's edges at `xs`: vertical runs within
/// `EDGE_TOL` px of one. A stroke crossing an edge loses only what runs along it.
fn remove_edges(ink: &mut Img, xs: &[f64]) -> Result<()> {
    const EDGE_TOL: f64 = 4.0;
    let v = Img::from_mat(&util::morph(&ink.to_mat()?, imgproc::MORPH_OPEN, &util::rect_kernel(1, 5)?)?)?;
    for xx in 0..ink.w {
        if !xs.iter().any(|&x| (xx as f64 - x).abs() <= EDGE_TOL) {
            continue;
        }
        for yy in 0..ink.h {
            if v.at(xx, yy) > 0 {
                ink.set(xx, yy, 0);
            }
        }
    }
    Ok(())
}

/// Separate signatures: strokes joined mostly sideways; a blob spanning
/// several rows is cut at the row boundaries. `whole`: the rows are only
/// the template's (their table's rules weren't matched), so a blob is
/// neither cut at them nor banded: all its ink is hidden (`envelope`).
fn signature_blobs(ink: &Img, extra: f64, w: f64, min_ink: usize, bands: &[(f64, f64)], stamps: &[(i64, i64, i64)], bridge: Option<&Img>, whole: bool) -> Result<Vec<SigBlob>> {
    let mut src = ink.clone();
    if let Some(b) = bridge {
        for (v, &bv) in src.d.iter_mut().zip(&b.d) {
            *v |= bv;
        }
    }
    // hidden whole, a stroke's pieces either side of a printed rule (taken
    // out of the ink) are joined too: a tip cut off alone is too small to be
    // a signature, and would be left showing
    let joined = util::dilate(&src.to_mat()?, &util::rect_kernel(SIG_JOIN + 6, if whole { 11 } else { 5 })?)?;
    let mut lab = Mat::default();
    let n = imgproc::connected_components_def(&joined, &mut lab)? as usize;
    let lab = util::labels_vec(&lab)?;
    let bin: Vec<u8> = ink.d.iter().map(|&v| (v > 0) as u8).collect();
    let mut own_m = Mat::default();
    let n_own = imgproc::connected_components_def(&util::from_bytes(ink.h as i32, ink.w as i32, &bin)?, &mut own_m)? as usize;
    let own = util::labels_vec(&own_m)?;
    let mut stroke = vec![0usize; n_own];
    for &l in &own {
        stroke[l as usize] += 1;
    }
    stroke[0] = 0;
    let mut edges = vec![bands[0].0];
    for p in bands.windows(2) {
        edges.push((p[0].1 + p[1].0) / 2.0);
    }
    edges.push(bands.last().unwrap().1);
    // pixels per joined component, row-major
    let mut comp: Vec<Vec<(i64, i64)>> = vec![Vec::new(); n];
    for (k, &l) in lab.iter().enumerate() {
        if l > 0 && ink.d[k] > 0 {
            comp[l as usize].push(((k % ink.w) as i64, (k / ink.w) as i64));
        }
    }
    let mut parts: Vec<(Vec<i64>, Vec<i64>)> = Vec::new();
    for pix in comp.iter().skip(1) {
        if pix.len() < min_ink {
            continue;
        }
        let rows: Vec<usize> = pix
            .iter()
            .map(|&(_, y)| {
                if whole {
                    return 0;
                }
                let r = edges.partition_point(|&e| e <= y as f64) as i64 - 1;
                r.clamp(0, bands.len() as i64 - 1) as usize
            })
            .collect();
        let mut uniq: Vec<usize> = rows.clone();
        uniq.sort();
        uniq.dedup();
        for &r in &uniq {
            let sel: Vec<usize> = (0..pix.len()).filter(|&k| rows[k] == r).collect();
            if sel.len() >= min_ink.min(pix.len()) || uniq.len() == 1 {
                parts.push((sel.iter().map(|&k| pix[k].0).collect(), sel.iter().map(|&k| pix[k].1).collect()));
            }
        }
    }
    let mut blobs = Vec::new();
    for (xs, ys) in parts {
        let inside = xs.iter().filter(|&&x| x as f64 >= extra && (x as f64) < extra + w).count();
        if (inside as f64) < 0.5 * xs.len() as f64 {
            continue; // writing from the neighbouring column
        }
        let longest = xs.iter().zip(&ys).map(|(&x, &y)| stroke[own[(y as usize) * ink.w + x as usize] as usize]).max().unwrap_or(0);
        if longest < min_ink {
            continue; // only crumbs
        }
        if inside_stamp(&xs, &ys, stamps) {
            continue;
        }
        let (x0, x1) = (*xs.iter().min().unwrap(), *xs.iter().max().unwrap() + 1);
        let (y0, y1) = (*ys.iter().min().unwrap(), *ys.iter().max().unwrap() + 1);
        if y1 - y0 <= SIG_MIN_H {
            continue; // flat: a printed rule or its remains
        }
        let band = if whole { envelope(&xs, &ys) } else { middle_band(&xs, &ys) };
        blobs.push(SigBlob { x0, y0, x1, y1, n: xs.len(), band });
    }
    Ok(blobs)
}

/// The printed line a field is written on (255).
fn own_rule(strip: &Mat, lo: i64, hi: i64, reference: Option<&Img>) -> Result<Img> {
    let bw = Img::from_mat(&binarize(strip)?)?;
    let klen = 150.max(bw.w as i32 / 4);
    let long = Img::from_mat(&util::morph(&bw.to_mat()?, imgproc::MORPH_OPEN, &util::rect_kernel(klen, 1)?)?)?;
    let (r0, r1) = ((lo - 4).max(0) as usize, ((hi + 4).max(0) as usize).min(bw.h));
    let rows: Vec<i64> = (r0..r1).filter(|&yy| long.d[yy * long.w..(yy + 1) * long.w].iter().any(|&v| v > 0)).map(|v| v as i64).collect();
    let mut rule = Img::new(bw.w, bw.h);
    let mut lines: Vec<Vec<i64>> = Vec::new();
    for &r in &rows {
        match lines.last_mut() {
            Some(l) if r - *l.last().unwrap() <= 3 => l.push(r),
            _ => lines.push(vec![r]),
        }
    }
    if reference.is_none() && (lines.len() > 2 || lines.iter().any(|l| l.last().unwrap() - l[0] > 12)) {
        return Ok(rule);
    }
    lines.retain(|l| l.last().unwrap() - l[0] <= 12);
    let (at, reach) = if let Some(r) = reference {
        let printed: Vec<u8> = r.d.iter().map(|&v| if v < 128 { 255 } else { 0 }).collect();
        let p = Img::from_mat(&util::morph(&util::from_bytes(r.h as i32, r.w as i32, &printed)?, imgproc::MORPH_OPEN, &util::rect_kernel(klen, 1)?)?)?;
        let prows: Vec<i64> = (r0..r1.min(p.h)).filter(|&yy| p.d[yy * p.w..(yy + 1) * p.w].iter().any(|&v| v > 0)).map(|v| v as i64).collect();
        if prows.is_empty() {
            return Ok(rule);
        }
        let d: Vec<f64> = prows.iter().map(|&q| (q - hi).abs() as f64).collect();
        (prows[util::argmin(&d)] as f64, 20.0)
    } else {
        (hi as f64, 0.4 * (hi - lo) as f64)
    };
    let mean = |l: &Vec<i64>| l.iter().sum::<i64>() as f64 / l.len() as f64;
    lines.retain(|l| (mean(l) - at).abs() <= reach);
    if lines.is_empty() {
        return Ok(rule);
    }
    let line = lines.iter().min_by(|a, b| (mean(a) - at).abs().partial_cmp(&(mean(b) - at).abs()).unwrap()).unwrap().clone();
    let runs = Img::from_mat(&util::morph(&bw.to_mat()?, imgproc::MORPH_OPEN, &util::rect_kernel(15, 1)?)?)?;
    let (top, bot) = ((line[0] - 2).max(0) as usize, ((line.last().unwrap() + 3) as usize).min(bw.h));
    for yy in top..bot {
        rule.d[yy * bw.w..(yy + 1) * bw.w].copy_from_slice(&runs.d[yy * bw.w..(yy + 1) * bw.w]);
    }
    Img::from_mat(&util::dilate(&rule.to_mat()?, &util::ones_kernel(3, 3)?)?)
}

fn sig_window(cells: &[Zone], s: &MaskSettings, reach: Reach, more_below: f64) -> (f64, f64) {
    let y = cells.iter().map(|z| z.rect[1]).fold(f64::INFINITY, f64::min);
    let bottom = cells.iter().map(|z| z.rect[1] + z.rect[3]).fold(f64::NEG_INFINITY, f64::max);
    let above = cells[0].search.map(|v| v[0]).unwrap_or(s.vpad);
    let below = cells[0].search.map(|v| v[1]).unwrap_or(s.vpad) + more_below;
    reach.clip(y, bottom, above, below)
}

pub struct SigOptions<'a> {
    pub reach: Reach,
    pub reference: Option<&'a Mat>,
    pub printed: Option<&'a Mat>,
    pub centered: bool,
    pub strikes: Option<&'a [Strike]>,
    pub exclude: Option<&'a Mat>,
}

/// Signature blobs in a column. `o.printed` (template space, nonzero =
/// printed text known to be there, e.g. typed "XXXX" in unused rows, or a
/// label an officer signed next to): its ink is no part of any signature.
/// `o.centered`: only blobs centred within the cells, not ones reaching in
/// from the search margins (a field on a text line, where the margins hold
/// the neighbouring lines). Strokes mostly on the next line belong to it; an
/// underline joins only the strokes crossing it.
fn find_signatures(gray: &Mat, f: &Mapper, cells: &[Zone], s: &MaskSettings, more_below: f64, o: &SigOptions) -> Result<(Vec<SigBlob>, [f64; 2], Vec<(f64, f64)>, Vec<Zone>, Img)> {
    let (above, below) = sig_window(cells, s, o.reach, more_below);
    let cs = column_strip(gray, f, cells, above, below, SIG_OVERRUN)?;
    let row_h = median_row_h(&cs.cells);
    let mut ink_m = ink_mask(&cs.strip, row_h, true)?;
    erase_strikes(&mut ink_m, cs.origin, o.strikes)?;
    let mut ink = Img::from_mat(&ink_m)?;
    if let Some(e) = o.exclude {
        let ex = Img::from_mat(&column_strip(e, f, cells, above, below, SIG_OVERRUN)?.strip)?;
        for (v, &x) in ink.d.iter_mut().zip(&ex.d) {
            if x > 127 {
                *v = 0;
            }
        }
    }
    if let Some(p) = o.printed {
        let inv = util::bitnot(p)?;
        let r = ref_crop(&inv, cs.origin, ink.h, ink.w)?;
        for (v, &x) in ink.d.iter_mut().zip(&r.d) {
            if x < 128 {
                *v = 0;
            }
        }
    }
    let stamps = remove_stamp_rings(&mut ink)?;
    if !cs.cells[0].region.is_empty() && cs.cells.len() > 2 {
        // a table column: a faint rule breaks into dashes the rule removal
        // leaves, and where a row's line meets the column's edge they cross
        // in small "+"s, tall enough for writing. Their upright strokes go;
        // what is left is flat, and dropped as rule when signatures are
        // grouped.
        let w = cs.cells[0].rect[2];
        remove_edges(&mut ink, &[SIG_OVERRUN, SIG_OVERRUN + w])?;
    }
    let lo = cs.bands[0].0 as i64;
    let hi = cs.bands.last().unwrap().1.ceil() as i64;
    let mut bridge: Option<Img> = None;
    if o.centered {
        let rref = match o.reference {
            Some(r) => Some(ref_crop(r, cs.origin, ink.h, ink.w)?),
            None => None,
        };
        let rule = own_rule(&cs.strip, lo, hi, rref.as_ref())?;
        for (v, &r) in ink.d.iter_mut().zip(&rule.d) {
            if r > 0 {
                *v = 0;
            }
        }
        // what is left of a dashed line: ink lying wholly within a few px of it
        let rule_rows: Vec<bool> = (0..rule.h).map(|yy| rule.d[yy * rule.w..(yy + 1) * rule.w].iter().any(|&v| v > 0)).collect();
        let mut near_rows = vec![false; rule.h];
        for yy in 0..rule.h {
            for k in yy.saturating_sub(3)..(yy + 4).min(rule.h) {
                if rule_rows[k] {
                    near_rows[yy] = true;
                }
            }
        }
        let far: Vec<bool> = near_rows.iter().map(|&b| !b).collect();
        let mut off_line = vec![0usize; rule.h + 1];
        for yy in 0..rule.h {
            off_line[yy + 1] = off_line[yy] + far[yy] as usize;
        }
        let bin: Vec<u8> = ink.d.iter().map(|&v| (v > 0) as u8).collect();
        let (lab, st, _) = util::components(&util::from_bytes(ink.h as i32, ink.w as i32, &bin)?)?;
        let labels = util::labels_vec(&lab)?;
        let flat: Vec<bool> = st
            .iter()
            .enumerate()
            .map(|(i, s)| i > 0 && off_line[(s[1] + s[3]) as usize] == off_line[s[1] as usize])
            .collect();
        for (k, &l) in labels.iter().enumerate() {
            if flat[l as usize] {
                ink.d[k] = 0;
            }
        }
        // the line joins only strokes crossing it
        let mut br = Img::new(rule.w, rule.h);
        let rows: Vec<usize> = (0..rule.h).filter(|&yy| rule_rows[yy]).collect();
        if !rows.is_empty() {
            let (top, bot) = (rows[0], rows[rows.len() - 1] + 1);
            let above: Vec<bool> = (0..ink.w).map(|xx| (top.saturating_sub(4)..top).any(|yy| ink.at(xx, yy) > 0)).collect();
            let below: Vec<bool> = (0..ink.w).map(|xx| (bot..(bot + 4).min(ink.h)).any(|yy| ink.at(xx, yy) > 0)).collect();
            let wide = |v: &Vec<bool>, xx: usize| (xx.saturating_sub(3)..(xx + 4).min(v.len())).any(|k| v[k]);
            for xx in 0..ink.w {
                if wide(&above, xx) && wide(&below, xx) {
                    for yy in top.saturating_sub(2)..(bot + 2).min(br.h) {
                        br.set(xx, yy, 255);
                    }
                }
            }
        }
        let both: Vec<u8> = ink.d.iter().zip(&br.d).map(|(&a, &b)| ((a > 0) || (b > 0)) as u8).collect();
        let mut lab2 = Mat::default();
        let n = imgproc::connected_components_def(&util::from_bytes(ink.h as i32, ink.w as i32, &both)?, &mut lab2)? as usize;
        let lab2 = util::labels_vec(&lab2)?;
        let (mut inside, mut total) = (vec![0usize; n], vec![0usize; n]);
        for (k, &l) in lab2.iter().enumerate() {
            if ink.d[k] > 0 {
                total[l as usize] += 1;
                let yy = (k / ink.w) as i64;
                if yy >= lo.max(0) && yy < hi.max(0) {
                    inside[l as usize] += 1;
                }
            }
        }
        for (k, &l) in lab2.iter().enumerate() {
            let l = l as usize;
            if l > 0 && (inside[l] as f64) < 0.3 * (total[l].max(1)) as f64 {
                ink.d[k] = 0;
            }
        }
        bridge = Some(br);
    }
    // a widened search down a table's rows: its rules weren't matched, so
    // the rows are the template's, and down a column of them the writing
    // may drift a row or more off (two signature lines a line apart don't)
    let whole = o.reach.margin != (0.0, 0.0) && cs.cells.len() > 2;
    let mut blobs = signature_blobs(&ink, SIG_OVERRUN, cs.cells[0].rect[2], SIG_MIN_INK, &cs.bands, &stamps, bridge.as_ref(), whole)?;
    // writing cut off at the bottom edge, starting well below the last row: the next line's
    let (end, sh) = (cs.bands.last().unwrap().1, ink.h as i64);
    blobs.retain(|b| !(b.y1 >= sh - 1 && b.y0 as f64 > end + 0.5 * (sh as f64 - end)));
    if o.centered {
        let share = |b: &SigBlob| {
            let (mut inside, mut total) = (0usize, 0usize);
            for yy in b.y0..b.y1 {
                for xx in b.x0..b.x1 {
                    if ink.at(xx as usize, yy as usize) > 0 {
                        total += 1;
                        if yy >= lo && yy < hi {
                            inside += 1;
                        }
                    }
                }
            }
            inside as f64 / total.max(1) as f64
        };
        blobs.retain(|b| share(b) >= 0.3 && !(b.y0 <= 1 && (b.y1 as f64) < (lo + hi) as f64 / 2.0));
    } else if cs.cells.len() <= 2 {
        // signature lines, not a table's rows: writing reaching in from the
        // top of the search and stopping above the first line's middle is
        // the line above's (a constituency written just over the line)
        let mid = (cs.bands[0].0 + cs.bands[0].1) / 2.0;
        blobs.retain(|b| !(b.y0 <= 1 && (b.y1 as f64) < mid));
    }
    Ok((blobs, cs.origin, cs.bands, cs.cells, ink))
}

/// Mask each signature by a band through its middle (whole, down a table
/// whose rows couldn't be matched).
pub fn signature_column(img: &mut Mat, gray: &Mat, f: &Mapper, cells: &[Zone], s: &MaskSettings, o: &SigOptions) -> Result<Vec<serde_json::Value>> {
    use serde_json::json;
    let pad = 6.0;
    let (blobs, origin, bands, cells, _) = find_signatures(gray, f, cells, s, SIG_BELOW, o)?;
    let mut counts = vec![0usize; cells.len()];
    for b in &blobs {
        for &(bx0, by0, bx1, by1) in &b.band {
            let x0 = bx0 as f64 - if bx0 == b.x0 { pad } else { 0.0 };
            let x1 = bx1 as f64 + if bx1 == b.x1 { pad } else { 0.0 };
            let (y0, y1) = (by0 as f64 - SIG_BAND_PAD, by1 as f64 + SIG_BAND_PAD);
            fill(img, f, &offset(quad(x0, y0, x1, y1), origin))?;
        }
        counts[row_of((b.y0 + b.y1) as f64 / 2.0, &bands)] += 1;
    }
    Ok(cells
        .iter()
        .zip(counts)
        .map(|(z, c)| if c > 0 { json!({"zone": z.name, "mode": "signature", "signatures": c}) } else { json!({"zone": z.name, "mode": "empty"}) })
        .collect())
}

/// Coverage check: the ink in each signature's middle band, found again on
/// the original, must be at least 90% under black in the redacted image.
pub fn verify_signatures(gray: &Mat, out: &Mat, f: &Mapper, cells: &[Zone], s: &MaskSettings, keep: Option<&Mat>, o: &SigOptions) -> Result<String> {
    let (blobs, origin, _, cells2, _) = find_signatures(gray, f, cells, s, SIG_BELOW, o)?;
    let _ = origin;
    let (above, below) = sig_window(&cells2, s, o.reach, SIG_BELOW);
    let og = util::gray(out)?;
    let ostrip = Img::from_mat(&column_strip(&og, f, &cells2, above, below, SIG_OVERRUN)?.strip)?;
    let gstrip = column_strip(gray, f, &cells2, above, below, SIG_OVERRUN)?;
    let kstrip = match keep {
        Some(k) => Some(Img::from_mat(&column_strip(k, f, &cells2, above, below, SIG_OVERRUN)?.strip)?),
        None => None,
    };
    let mut ink_m = ink_mask(&gstrip.strip, median_row_h(&cells2), true)?;
    erase_strikes(&mut ink_m, gstrip.origin, o.strikes)?;
    let ink = Img::from_mat(&ink_m)?;
    let mut bad = 0;
    for b in &blobs {
        if b.n < 2 * SIG_MIN_INK {
            continue;
        }
        let in_band = |xx: i64, yy: i64| b.band.iter().any(|&(bx0, by0, bx1, by1)| xx >= bx0 && xx < bx1 && yy >= by0.max(b.y0) && yy < by1);
        let (mut sel, mut kept_n) = (Vec::new(), 0usize);
        for yy in b.y0..b.y1 {
            for xx in b.x0..b.x1 {
                if in_band(xx, yy) && ink.at(xx as usize, yy as usize) > 0 {
                    let kept = kstrip.as_ref().map(|k| k.at(xx as usize, yy as usize) > 0).unwrap_or(false);
                    if kept {
                        kept_n += 1;
                    }
                    sel.push((xx, yy, kept));
                }
            }
        }
        if kstrip.is_some() && kept_n as f64 > 0.5 * sel.len() as f64 {
            continue;
        }
        let rest: Vec<&(i64, i64, bool)> = sel.iter().filter(|p| !p.2).collect();
        if rest.is_empty() {
            continue;
        }
        let dark = rest.iter().filter(|p| ostrip.at(p.0 as usize, p.1 as usize) <= 8).count();
        if (dark as f64) / (rest.len() as f64) < 0.9 {
            bad += 1;
        }
    }
    if bad > 0 {
        return Ok(format!("{}: {} signature(s) not masked through the middle", cells2[0].key(), bad));
    }
    Ok(String::new())
}

/// Template-space mask (255) of printed `labels` that lie in a zone's search:
/// each found on the page near where `f` puts it, and its print in the
/// reference copied there, a few px wider, for `SigOptions::printed`.
pub fn printed_labels(gray: &Mat, f: &Mapper, reference: &Mat, labels: &[[f64; 4]]) -> Result<Mat> {
    let refg = Img::from_mat(&util::gray(reference)?)?;
    let mut m = Img::new(refg.w, refg.h);
    for &l in labels {
        let (d, _) = find_label(gray, f, reference, l, 30.0, 30.0)?;
        let [dx, dy] = d.unwrap_or([0.0, 0.0]).map(|v| v.round() as i64);
        let [x, y, w, h] = l.map(|v| v.round() as i64);
        for yy in y.max(0)..(y + h).min(refg.h as i64) {
            for xx in x.max(0)..(x + w).min(refg.w as i64) {
                let (tx, ty) = (xx + dx, yy + dy);
                if refg.at(xx as usize, yy as usize) < 128 && tx >= 0 && ty >= 0 && (tx as usize) < m.w && (ty as usize) < m.h {
                    m.set(tx as usize, ty as usize, 255);
                }
            }
        }
    }
    util::dilate(&m.to_mat()?, &util::ones_kernel(5, 5)?)
}

/// Locate a printed label near where the template expects it. Returns the
/// offset (dx, dy) in template px, or None if not found; and the NCC.
pub fn find_anchor(gray: &Mat, f: &Mapper, reference: &Mat, anchor: [f64; 4]) -> Result<(Option<[f64; 2]>, f64)> {
    find_label(gray, f, reference, anchor, 60.0, 120.0)
}

/// Where a printed label of the reference lies on the page, as an offset in
/// template px from where `f` puts it, searching `rx` px either side and `ry`
/// px above and below; and how well it matched.
pub fn find_label(gray: &Mat, f: &Mapper, reference: &Mat, anchor: [f64; 4], rx: f64, ry: f64) -> Result<(Option<[f64; 2]>, f64)> {
    let min_ncc = 0.5;
    let [x, y, w, h] = anchor;
    let (sw, sh) = ((w + 2.0 * rx) as usize, (h + 2.0 * ry) as usize);
    let maps = strip_maps(f, x - rx, y - ry, sw, sh)?;
    let area = remap(gray, &maps, imgproc::INTER_LINEAR, core::BORDER_REPLICATE)?;
    let label = Mat::roi(reference, core::Rect::new(x as i32, y as i32, (x + w) as i32 - x as i32, (y + h) as i32 - y as i32))?.try_clone()?;
    let area_b = util::gaussian(&area, 3)?;
    let (mut best, mut where_) = (-1.0f64, None);
    for sc in [0.9, 1.0, 1.1] {
        let lab = util::resize_by(&label, sc, imgproc::INTER_AREA)?;
        if lab.rows() >= area.rows() || lab.cols() >= area.cols() {
            continue;
        }
        let mut res = Mat::default();
        imgproc::match_template(&area_b, &util::gaussian(&lab, 3)?, &mut res, imgproc::TM_CCOEFF_NORMED, &core::no_array())?;
        let mut v = 0.0;
        let mut loc = Point::default();
        core::min_max_loc(&res, None, Some(&mut v), None, Some(&mut loc), &core::no_array())?;
        if v > best {
            best = v;
            where_ = Some([
                loc.x as f64 + lab.cols() as f64 / 2.0 - (rx + w / 2.0),
                loc.y as f64 + lab.rows() as f64 / 2.0 - (ry + h / 2.0),
            ]);
        }
    }
    if best < min_ncc {
        return Ok((None, best));
    }
    Ok((where_, best))
}

#[cfg(test)]
#[path = "tests/masking.rs"]
mod py_tests;
