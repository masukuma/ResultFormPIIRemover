//! Small numeric and image helpers, written to give the same results as the
//! numpy / OpenCV calls of the Python version (rounding half to even, linear
//! percentiles, medians of even counts...): masks are placed from these
//! numbers, and the two implementations must agree on them.

use anyhow::Result;
use opencv::core::{self, Mat, Point, Scalar, Size, Vector, CV_32F, CV_8U};
use opencv::prelude::*;

/// numpy's `np.round` / Python's `round`: half to even.
pub fn round_even(x: f64) -> f64 {
    x.round_ties_even()
}

pub fn iround(x: f64) -> i32 {
    round_even(x) as i32
}

/// `np.percentile(a, q)` with the default (linear) method.
pub fn percentile(values: &[f64], q: f64) -> f64 {
    quantile(values, q / 100.0)
}

/// `np.quantile(a, q)`, linear method, including numpy's way of
/// interpolating (from the upper neighbour when past the midpoint).
pub fn quantile(values: &[f64], q: f64) -> f64 {
    let mut v: Vec<f64> = values.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    quantile_sorted(&v, q)
}

pub fn quantile_sorted(v: &[f64], q: f64) -> f64 {
    let n = v.len();
    if n == 0 {
        return f64::NAN;
    }
    let idx = q * (n - 1) as f64;
    let lo = idx.floor() as usize;
    let hi = (lo + 1).min(n - 1);
    let t = idx - lo as f64;
    let (a, b) = (v[lo], v[hi]);
    let d = b - a;
    if t >= 0.5 {
        b - d * (1.0 - t)
    } else {
        a + d * t
    }
}

/// `np.median`.
pub fn median(values: &[f64]) -> f64 {
    let mut v: Vec<f64> = values.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = v.len();
    if n == 0 {
        return f64::NAN;
    }
    if n % 2 == 1 {
        v[n / 2]
    } else {
        (v[n / 2 - 1] + v[n / 2]) / 2.0
    }
}

pub fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        return f64::NAN;
    }
    values.iter().sum::<f64>() / values.len() as f64
}

/// Python's `%.Nf`-free float formatting for messages: `{:.2f}`.
pub fn f2(x: f64) -> String {
    format!("{:.2}", x)
}

// --- boolean grids (8x8 block masks) ----------------------------------------------

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Grid {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<bool>,
}

impl Grid {
    pub fn new(rows: usize, cols: usize) -> Self {
        Grid { rows, cols, data: vec![false; rows * cols] }
    }
    #[inline]
    pub fn get(&self, r: usize, c: usize) -> bool {
        self.data[r * self.cols + c]
    }
    #[inline]
    pub fn set(&mut self, r: usize, c: usize, v: bool) {
        self.data[r * self.cols + c] = v;
    }
    pub fn count(&self) -> usize {
        self.data.iter().filter(|&&b| b).count()
    }
    pub fn any(&self) -> bool {
        self.data.iter().any(|&b| b)
    }
    pub fn and(&self, o: &Grid) -> Grid {
        Grid { rows: self.rows, cols: self.cols, data: self.data.iter().zip(&o.data).map(|(a, b)| *a && *b).collect() }
    }
    pub fn and_not(&self, o: &Grid) -> Grid {
        Grid { rows: self.rows, cols: self.cols, data: self.data.iter().zip(&o.data).map(|(a, b)| *a && !*b).collect() }
    }
    /// Copy into a grid of another size: the overlap is kept, the rest false.
    pub fn resized(&self, rows: usize, cols: usize) -> Grid {
        let mut g = Grid::new(rows, cols);
        for r in 0..rows.min(self.rows) {
            for c in 0..cols.min(self.cols) {
                g.set(r, c, self.get(r, c));
            }
        }
        g
    }
}

// --- OpenCV helpers -----------------------------------------------------------------

pub fn zeros_u8(rows: i32, cols: i32) -> Result<Mat> {
    Ok(Mat::new_rows_cols_with_default(rows, cols, CV_8U, Scalar::all(0.0))?)
}

pub fn full_u8(rows: i32, cols: i32, v: u8) -> Result<Mat> {
    Ok(Mat::new_rows_cols_with_default(rows, cols, CV_8U, Scalar::all(v as f64))?)
}

/// Bytes of a continuous single-channel 8-bit Mat (cloned to be continuous if not).
pub fn bytes(m: &Mat) -> Result<Vec<u8>> {
    if m.is_continuous() {
        Ok(m.data_bytes()?.to_vec())
    } else {
        Ok(m.try_clone()?.data_bytes()?.to_vec())
    }
}

pub fn from_bytes(rows: i32, cols: i32, data: &[u8]) -> Result<Mat> {
    let mut m = zeros_u8(rows, cols)?;
    m.data_bytes_mut()?.copy_from_slice(data);
    Ok(m)
}

pub fn gray(img: &Mat) -> Result<Mat> {
    if img.channels() == 1 {
        return Ok(img.try_clone()?);
    }
    let mut g = Mat::default();
    opencv::imgproc::cvt_color_def(img, &mut g, opencv::imgproc::COLOR_BGR2GRAY)?;
    Ok(g)
}

pub fn rect_kernel(w: i32, h: i32) -> Result<Mat> {
    Ok(opencv::imgproc::get_structuring_element_def(opencv::imgproc::MORPH_RECT, Size::new(w, h))?)
}

pub fn ones_kernel(w: i32, h: i32) -> Result<Mat> {
    Ok(Mat::new_rows_cols_with_default(h, w, CV_8U, Scalar::all(1.0))?)
}

pub fn morph(src: &Mat, op: i32, kernel: &Mat) -> Result<Mat> {
    let mut out = Mat::default();
    opencv::imgproc::morphology_ex(
        src,
        &mut out,
        op,
        kernel,
        Point::new(-1, -1),
        1,
        core::BORDER_CONSTANT,
        opencv::imgproc::morphology_default_border_value()?,
    )?;
    Ok(out)
}

pub fn dilate(src: &Mat, kernel: &Mat) -> Result<Mat> {
    let mut out = Mat::default();
    opencv::imgproc::dilate(
        src,
        &mut out,
        kernel,
        Point::new(-1, -1),
        1,
        core::BORDER_CONSTANT,
        opencv::imgproc::morphology_default_border_value()?,
    )?;
    Ok(out)
}

pub fn gaussian(src: &Mat, k: i32) -> Result<Mat> {
    let mut out = Mat::default();
    opencv::imgproc::gaussian_blur_def(src, &mut out, Size::new(k, k), 0.0)?;
    Ok(out)
}

pub fn adaptive_inv(src: &Mat, block: i32, c: f64) -> Result<Mat> {
    let mut out = Mat::default();
    opencv::imgproc::adaptive_threshold(
        src,
        &mut out,
        255.0,
        opencv::imgproc::ADAPTIVE_THRESH_MEAN_C,
        opencv::imgproc::THRESH_BINARY_INV,
        block,
        c,
    )?;
    Ok(out)
}

pub fn subtract(a: &Mat, b: &Mat) -> Result<Mat> {
    let mut out = Mat::default();
    core::subtract_def(a, b, &mut out)?;
    Ok(out)
}

pub fn bitand(a: &Mat, b: &Mat) -> Result<Mat> {
    let mut out = Mat::default();
    core::bitwise_and_def(a, b, &mut out)?;
    Ok(out)
}

pub fn bitor(a: &Mat, b: &Mat) -> Result<Mat> {
    let mut out = Mat::default();
    core::bitwise_or_def(a, b, &mut out)?;
    Ok(out)
}

pub fn bitnot(a: &Mat) -> Result<Mat> {
    let mut out = Mat::default();
    core::bitwise_not_def(a, &mut out)?;
    Ok(out)
}

pub fn resize(src: &Mat, w: i32, h: i32, interp: i32) -> Result<Mat> {
    let mut out = Mat::default();
    opencv::imgproc::resize(src, &mut out, Size::new(w, h), 0.0, 0.0, interp)?;
    Ok(out)
}

/// `cv2.resize(img, None, fx=f, fy=f)`: OpenCV rounds the size.
pub fn resize_by(src: &Mat, f: f64, interp: i32) -> Result<Mat> {
    let mut out = Mat::default();
    opencv::imgproc::resize(src, &mut out, Size::new(0, 0), f, f, interp)?;
    Ok(out)
}

/// `cv2.fillPoly(img, [pts], value)` with points rounded half to even.
pub fn fill_poly(img: &mut Mat, pts: &[[f64; 2]], value: f64) -> Result<()> {
    let p: Vector<Point> = pts.iter().map(|q| Point::new(iround(q[0]), iround(q[1]))).collect();
    let polys: Vector<Vector<Point>> = vec![p].into_iter().collect();
    opencv::imgproc::fill_poly_def(img, &polys, Scalar::all(value))?;
    Ok(())
}

/// Connected components with stats: (labels, stats rows [x, y, w, h, area]).
pub fn components(bin: &Mat) -> Result<(Mat, Vec<[i32; 5]>, Vec<[f64; 2]>)> {
    let mut lab = Mat::default();
    let mut st = Mat::default();
    let mut cen = Mat::default();
    let n = opencv::imgproc::connected_components_with_stats_def(bin, &mut lab, &mut st, &mut cen)?;
    let mut stats = Vec::with_capacity(n as usize);
    let mut cents = Vec::with_capacity(n as usize);
    for i in 0..n {
        stats.push([
            *st.at_2d::<i32>(i, 0)?,
            *st.at_2d::<i32>(i, 1)?,
            *st.at_2d::<i32>(i, 2)?,
            *st.at_2d::<i32>(i, 3)?,
            *st.at_2d::<i32>(i, 4)?,
        ]);
        cents.push([*cen.at_2d::<f64>(i, 0)?, *cen.at_2d::<f64>(i, 1)?]);
    }
    Ok((lab, stats, cents))
}

/// Labels as a flat i32 vector.
pub fn labels_vec(lab: &Mat) -> Result<Vec<i32>> {
    let m = if lab.is_continuous() { lab.try_clone()? } else { lab.try_clone()? };
    Ok(m.data_typed::<i32>()?.to_vec())
}

pub fn to_f32(src: &Mat) -> Result<Mat> {
    let mut out = Mat::default();
    src.convert_to(&mut out, CV_32F, 1.0, 0.0)?;
    Ok(out)
}

/// A 3x3 homography as a CV_64F Mat.
pub fn mat3(h: &[[f64; 3]; 3]) -> Result<Mat> {
    let mut m = Mat::new_rows_cols_with_default(3, 3, core::CV_64F, Scalar::all(0.0))?;
    for r in 0..3 {
        for c in 0..3 {
            *m.at_2d_mut::<f64>(r, c)? = h[r as usize][c as usize];
        }
    }
    Ok(m)
}

pub fn arr3(m: &Mat) -> Result<[[f64; 3]; 3]> {
    let mut h = [[0.0; 3]; 3];
    for r in 0..3 {
        for c in 0..3 {
            h[r][c] = *m.at_2d::<f64>(r as i32, c as i32)?;
        }
    }
    Ok(h)
}

pub fn mul3(a: &[[f64; 3]; 3], b: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let mut o = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            o[i][j] = (0..3).map(|k| a[i][k] * b[k][j]).sum();
        }
    }
    o
}

/// `np.linalg.inv` (numpy's own LAPACK, see npla.rs).
pub fn inv3(m: &[[f64; 3]; 3]) -> Result<[[f64; 3]; 3]> {
    crate::npla::inv3(m)
}

/// `cv2.perspectiveTransform` of one point.
pub fn project(h: &[[f64; 3]; 3], p: [f64; 2]) -> [f64; 2] {
    let x = h[0][0] * p[0] + h[0][1] * p[1] + h[0][2];
    let y = h[1][0] * p[0] + h[1][1] * p[1] + h[1][2];
    let w = h[2][0] * p[0] + h[2][1] * p[1] + h[2][2];
    if w.abs() > f32::EPSILON as f64 {
        let r = 1.0 / w;
        [x * r, y * r]
    } else {
        [0.0, 0.0]
    }
}

pub const IDENTITY: [[f64; 3]; 3] = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn numpy_compatible() {
        // np.percentile([1, 2, 3, 4], [1, 99]) -> [1.03, 3.97]
        let v = [1.0, 2.0, 3.0, 4.0];
        assert!((percentile(&v, 1.0) - 1.03).abs() < 1e-12);
        assert!((percentile(&v, 99.0) - 3.97).abs() < 1e-12);
        assert_eq!(median(&v), 2.5);
        assert_eq!(round_even(2.5), 2.0);
        assert_eq!(round_even(3.5), 4.0);
        assert_eq!(round_even(-0.5), -0.0);
    }
}

/// `np.sum` of a float64 array: numpy's pairwise summation (8 partial sums
/// per block of up to 128, halves recursively above that). Sums decide ties
/// between candidate table fits, so they must add up the same way.
pub fn np_sum(a: &[f64]) -> f64 {
    let n = a.len();
    if n < 8 {
        let mut s = -0.0f64;
        for &x in a {
            s += x;
        }
        return if n == 0 { 0.0 } else { s };
    }
    if n <= 128 {
        let mut r = [a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7]];
        let mut i = 8;
        while i < n - (n % 8) {
            for k in 0..8 {
                r[k] += a[i + k];
            }
            i += 8;
        }
        let mut res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        while i < n {
            res += a[i];
            i += 1;
        }
        return res;
    }
    let mut n2 = n / 2;
    n2 -= n2 % 8;
    np_sum(&a[..n2]) + np_sum(&a[n2..])
}

/// `np.sum` of a float32 array (the same pairwise scheme, float32 partial sums).
pub fn np_sum_f32(a: &[f32]) -> f32 {
    let n = a.len();
    if n < 8 {
        let mut s = -0.0f32;
        for &x in a {
            s += x;
        }
        return if n == 0 { 0.0 } else { s };
    }
    if n <= 128 {
        let mut r = [a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7]];
        let mut i = 8;
        while i < n - (n % 8) {
            for k in 0..8 {
                r[k] += a[i + k];
            }
            i += 8;
        }
        let mut res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        while i < n {
            res += a[i];
            i += 1;
        }
        return res;
    }
    let mut n2 = n / 2;
    n2 -= n2 % 8;
    np_sum_f32(&a[..n2]) + np_sum_f32(&a[n2..])
}

/// `np.linalg.norm` of a float64 2-vector: numpy's BLAS dot fuses the
/// second multiply-add.
pub fn norm2(dx: f64, dy: f64) -> f64 {
    dy.mul_add(dy, dx * dx).sqrt()
}

/// `np.dot` of two float64 2-vectors (BLAS, fused as above).
pub fn dot2(a: [f64; 2], b: [f64; 2]) -> f64 {
    a[1].mul_add(b[1], a[0] * b[0])
}

pub fn np_mean(a: &[f64]) -> f64 {
    np_sum(a) / a.len() as f64
}


/// `np.argmin`: first index of the minimum.
pub fn argmin(a: &[f64]) -> usize {
    let mut best = 0;
    for i in 1..a.len() {
        if a[i] < a[best] {
            best = i;
        }
    }
    best
}

/// `np.searchsorted(a, v)` (side="left").
pub fn searchsorted(a: &[f64], v: f64) -> usize {
    a.partition_point(|&x| x < v)
}
