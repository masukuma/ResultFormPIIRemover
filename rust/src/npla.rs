//! numpy's linear algebra, computed by the very library numpy uses.
//!
//! `np.polyfit`, `np.linalg.lstsq` and `np.linalg.inv` run LAPACK (`dgelsd`,
//! `dgesv`) from the OpenBLAS build bundled with numpy. A closed-form fit agrees
//! with them only to about 1e-12, and that is not enough: on a striped scan,
//! table rules are merged in an order those last digits decide. So the same
//! routines are called here, from the same library, with the same arguments
//! numpy passes.
//!
//! The library is found at `RFPII_NUMPY_BLAS` (default
//! /opt/rfpii/lib/libscipy_openblas64_-f48b354e.so), with its libgfortran and
//! libquadmath beside it.

use anyhow::{anyhow, bail, Result};
use once_cell::sync::OnceCell;
use std::path::PathBuf;

type I = i64;
type Dgelsd = unsafe extern "C" fn(*const I, *const I, *const I, *mut f64, *const I, *mut f64, *const I, *mut f64, *const f64, *mut I, *mut f64, *const I, *mut I, *mut I);
type Dgesv = unsafe extern "C" fn(*const I, *const I, *mut f64, *const I, *mut I, *mut f64, *const I, *mut I);

struct Lib {
    _deps: Vec<libloading::Library>,
    _lib: libloading::Library,
    dgelsd: Dgelsd,
    dgesv: Dgesv,
}

static LIB: OnceCell<Lib> = OnceCell::new();

pub fn library_path() -> PathBuf {
    std::env::var_os("RFPII_NUMPY_BLAS").map(PathBuf::from).unwrap_or_else(|| PathBuf::from("/opt/rfpii/lib/libscipy_openblas64_-f48b354e.so"))
}

fn lib() -> Result<&'static Lib> {
    LIB.get_or_try_init(|| {
        let path = library_path();
        let dir = path.parent().map(|p| p.to_path_buf()).unwrap_or_default();
        // its dependencies first, by path: they are then found by their soname
        let mut deps = Vec::new();
        let mut names: Vec<PathBuf> = std::fs::read_dir(&dir).map(|d| d.filter_map(|e| e.ok().map(|e| e.path())).collect()).unwrap_or_default();
        names.sort();
        for prefix in ["libquadmath", "libgfortran"] {
            for p in names.iter().filter(|p| p.file_name().map(|f| f.to_string_lossy().starts_with(prefix)).unwrap_or(false)) {
                deps.push(unsafe { libloading::Library::new(p) }.map_err(|e| anyhow!("loading {}: {e}", p.display()))?);
            }
        }
        let l = unsafe { libloading::Library::new(&path) }.map_err(|e| anyhow!("numpy's OpenBLAS ({}): {e}", path.display()))?;
        let dgelsd: Dgelsd = *unsafe { l.get::<Dgelsd>(b"scipy_dgelsd_64_\0") }.map_err(|e| anyhow!("{e}"))?;
        let dgesv: Dgesv = *unsafe { l.get::<Dgesv>(b"scipy_dgesv_64_\0") }.map_err(|e| anyhow!("{e}"))?;
        Ok(Lib { _deps: deps, _lib: l, dgelsd, dgesv })
    })
}

/// `np.linalg.lstsq(a, b, rcond)` for an m x n `a` (row-major) and one right
/// hand side: the solution x (n values).
pub fn lstsq(a: &[f64], m: usize, n: usize, b: &[f64], rcond: f64) -> Result<Vec<f64>> {
    let l = lib()?;
    let (mm, nn, nrhs) = (m as I, n as I, 1 as I);
    let lda = mm.max(1);
    let ldb = mm.max(nn).max(1);
    let mut af = vec![0.0; m * n]; // column-major
    for i in 0..m {
        for j in 0..n {
            af[j * m + i] = a[i * n + j];
        }
    }
    let mut bf = vec![0.0; ldb as usize];
    bf[..m].copy_from_slice(&b[..m]);
    let mut s = vec![0.0; m.min(n).max(1)];
    let (mut rank, mut info) = (0 as I, 0 as I);
    // workspace query, then the call (as numpy does)
    let (mut wq, mut iwq) = (0.0f64, 0 as I);
    unsafe {
        (l.dgelsd)(&mm, &nn, &nrhs, af.as_mut_ptr(), &lda, bf.as_mut_ptr(), &ldb, s.as_mut_ptr(), &rcond, &mut rank, &mut wq, &(-1), &mut iwq, &mut info);
    }
    if info != 0 {
        bail!("dgelsd workspace query failed ({})", info);
    }
    let lwork = wq as I;
    let mut work = vec![0.0; lwork.max(1) as usize];
    let mut iwork = vec![0 as I; iwq.max(1) as usize];
    unsafe {
        (l.dgelsd)(&mm, &nn, &nrhs, af.as_mut_ptr(), &lda, bf.as_mut_ptr(), &ldb, s.as_mut_ptr(), &rcond, &mut rank, work.as_mut_ptr(), &lwork, iwork.as_mut_ptr(), &mut info);
    }
    if info != 0 {
        bail!("least squares did not converge ({})", info);
    }
    Ok(bf[..n].to_vec())
}

/// `np.polyfit(x, y, 1)`: (slope, intercept).
pub fn polyfit1(x: &[f64], y: &[f64]) -> Result<(f64, f64)> {
    let n = x.len();
    // the columns of vander(x, 2) scaled to unit norm (norms summed in order)
    let mut s0 = 0.0f64;
    for &v in x {
        s0 += v * v;
    }
    let mut s1 = 0.0f64;
    for _ in 0..n {
        s1 += 1.0;
    }
    let (s0, s1) = (s0.sqrt(), s1.sqrt());
    let mut a = vec![0.0; 2 * n];
    for i in 0..n {
        a[2 * i] = x[i] / s0;
        a[2 * i + 1] = 1.0 / s1;
    }
    let c = lstsq(&a, n, 2, y, n as f64 * f64::EPSILON)?;
    Ok((c[0] / s0, c[1] / s1))
}

/// `np.linalg.inv` of a 3x3 matrix.
pub fn inv3(m: &[[f64; 3]; 3]) -> Result<[[f64; 3]; 3]> {
    let l = lib()?;
    let n = 3 as I;
    let mut a = [0.0; 9];
    let mut b = [0.0; 9];
    for i in 0..3 {
        for j in 0..3 {
            a[j * 3 + i] = m[i][j];
        }
        b[i * 3 + i] = 1.0;
    }
    let mut ipiv = [0 as I; 3];
    let mut info = 0 as I;
    unsafe { (l.dgesv)(&n, &n, a.as_mut_ptr(), &n, ipiv.as_mut_ptr(), b.as_mut_ptr(), &n, &mut info) };
    if info != 0 {
        bail!("singular matrix");
    }
    let mut out = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            out[i][j] = b[j * 3 + i];
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn polyfit_as_numpy() {
        // np.polyfit([3, 4, 5, 9, 10], [7.0, 7.5, 7.9, 9.6, 10.2], 1), bit for bit
        assert_eq!(polyfit1(&[3.0, 4.0, 5.0, 9.0, 10.0], &[7.0, 7.5, 7.9, 9.6, 10.2]).unwrap(), (0.4448453608247417, 5.6819587628866));
        // x = np.arange(200) * 1.7 + 3; y = 0.013 * x + 42.5 + np.sin(x)
        let x: Vec<f64> = (0..200).map(|i| i as f64 * 1.7 + 3.0).collect();
        let y: Vec<f64> = x.iter().map(|&v| 0.013 * v + 42.5 + v.sin()).collect();
        assert_eq!(polyfit1(&x, &y).unwrap(), (0.013088378701483338, 42.48615891827679));
        let m = inv3(&[[1.0002, 0.013, 512.3], [-0.011, 0.9987, 1890.25], [1.1e-6, -2.3e-6, 1.0]]).unwrap();
        assert_eq!(m[0], [1.000178683598741, -0.014137737612177763, -485.667681086216]);
        assert_eq!(m[2], [-1.0702002601880784e-06, 2.3081393817706225e-06, 0.9961853031269025]);
    }
}
