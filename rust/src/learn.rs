//! Learned layouts: the app writes its own template for a printing of a form
//! from the forms of that printing (GENERALIZING.md).
//!
//! 1. Group. Pages printed alike: a KIEMS name gives the form and the
//!    station, and so the printing (Form 34A is printed once for the
//!    country, 35A per constituency, 36A per ward, 37A to 39A per county).
//! 2. Upright. Each page is turned upright, deskewed and scaled so printed
//!    text is `TEXT_H` high (`generic::find_frame`); lighting is flattened.
//! 3. Align. Every page onto the sharpest one (the seed) by their printed
//!    features (ORB): a homography, then dense optical flow, smoothed, for
//!    curled and folded pages.
//! 4. The blank form. Print is the same on every page; handwriting differs
//!    from page to page and drops out of their pixel-wise median, which is
//!    the reference (its fields emptied once they are known).
//! 5. Where people write. Ink a page has that the blank form doesn't,
//!    counted over the group.
//! 6. Read. The structural reader reads the blank form: on a flat, sharp
//!    form with no writing on it, its labels and rules are found reliably.
//!    Each field takes its role from its printed label (`generic.rs`).
//! 7. Write. A template like a hand-made one (`templates.rs`): the results
//!    protected, the officers' signature lines, the agents' columns with
//!    their rows and the labels around them, the comments read as free text.
//!
//! Each field is then checked against where people actually wrote: a field
//! no one writes in, or writing no field explains, is reported.

use anyhow::{anyhow, bail, Result};
use opencv::core::{self, DMatch, KeyPoint, Mat, Point2f, Scalar, Size, Vector};
use opencv::features::{BFMatcher, ORB_ScoreType, ORB};
use opencv::geometry;
use opencv::imgcodecs;
use opencv::imgproc;
use opencv::prelude::*;
use opencv::video;
use rayon::prelude::*;
use serde_json::{json, Value};
use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use crate::generic::{self, Analysis, Field, Layout, TEXT_H};
use crate::ocr::Line as OLine;
use crate::templates::Zone;
use crate::{io, kiems, ocr, util};

type M3 = [[f64; 3]; 3];

/// Printed headings of the free-text areas.
static HEADING: once_cell::sync::Lazy<regex::Regex> =
    once_cell::sync::Lazy::new(|| regex::Regex::new(r"(?i)refus|comment|reason|remark|observ").unwrap());

/// The sharpest pages of a printing are read too, besides its blank form.
const READ_PAGES: usize = 4;

#[derive(Clone, Debug)]
pub struct LearnOptions {
    /// At most this many pages of a printing are used (spread over the group).
    pub max_pages: usize,
    /// A printing with fewer pages that align is not learned.
    pub min_pages: usize,
    /// Share of the aligned pages that must have ink at a pixel for it to
    /// be print (where people don't write often: see `blank`).
    pub print_share: f64,
}

impl Default for LearnOptions {
    fn default() -> Self {
        LearnOptions { max_pages: 40, min_pages: 5, print_share: 0.45 }
    }
}

/// The printing a file belongs to: its form ("34A") and a key shared by
/// every page printed alike. `form`: the form of files whose name doesn't
/// say (all are taken to be one printing).
pub fn printing(src: &str, form: Option<&str>) -> (String, String) {
    let name = src.rsplit(io::ZIP_SEP).next().unwrap_or(src);
    let stem = Path::new(name).file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
    if let Some(k) = kiems::parse_name(&stem) {
        let c = &k.station;
        let area = match &k.form[..2] {
            "34" => "national".to_string(),
            "35" => format!("constituency {}", &c[..6]),
            "36" => format!("ward {}", &c[..10]),
            _ => format!("county {}", &c[..3]),
        };
        return (k.form.clone(), format!("{} {}", k.form, area));
    }
    let f = form.unwrap_or("unknown").to_string();
    (f.clone(), f)
}

/// A page, upright and scaled to TEXT_H, lighting flattened. Pixels are
/// kept as bytes: shared between the threads aligning pages.
struct Upright {
    src: String,
    px: Vec<u8>,
    w: i32,
    h: i32,
    sharp: f64,
    skew: f64,
}

impl Upright {
    fn gray(&self) -> Result<Mat> {
        util::from_bytes(self.h, self.w, &self.px)
    }
}

fn upright(src: &str) -> Result<Option<Upright>> {
    let pages = io::source_pages(src, None)?;
    let Some(p) = pages.first() else { return Ok(None) };
    let Some(f) = ocr::with_engine(|o| generic::find_frame(&p.pixels, o, 1200.0))? else { return Ok(None) };
    let work = generic::warp(&p.pixels, &f)?;
    let gray = util::flatten_light(&util::gray(&work)?)?;
    Ok(Some(Upright { src: src.to_string(), sharp: sharpness(&gray)?, skew: f.skew, px: util::bytes(&gray)?, w: gray.cols(), h: gray.rows() }))
}


/// Variance of the Laplacian: how sharp the print is.
fn sharpness(g: &Mat) -> Result<f64> {
    let mut lap = Mat::default();
    imgproc::laplacian(g, &mut lap, core::CV_64F, 3, 1.0, 0.0, core::BORDER_DEFAULT)?;
    let (mut mean, mut sd) = (Mat::default(), Mat::default());
    core::mean_std_dev(&lap, &mut mean, &mut sd, &core::no_array())?;
    let s = *sd.at_2d::<f64>(0, 0)?;
    Ok(s * s)
}

/// ORB keypoints and descriptors (rows of 32 bytes), as plain data.
struct Orb {
    pts: Vec<Point2f>,
    desc: Vec<u8>,
    n: i32,
}

impl Orb {
    fn desc(&self) -> Result<Mat> {
        let mut m = Mat::new_rows_cols_with_default(self.n, 32, core::CV_8U, Scalar::all(0.0))?;
        m.data_bytes_mut()?.copy_from_slice(&self.desc);
        Ok(m)
    }
}

/// ORB features spread over the page, tile by tile: the strongest corners
/// of a photo are often all in what lies around the form (a table's grain, a
/// dark floor), and a form's own corners, softer, would get none.
fn orb(g: &Mat) -> Result<Orb> {
    const TILES: (i32, i32) = (6, 8);
    const PER_TILE: i32 = 8000 / (TILES.0 * TILES.1);
    let (w, h) = (g.cols(), g.rows());
    let mut o = ORB::create(PER_TILE, 1.2, 8, 31, 0, 2, ORB_ScoreType::HARRIS_SCORE, 31, 10)?;
    let mut k = Vector::<KeyPoint>::new();
    for j in 0..TILES.1 {
        for i in 0..TILES.0 {
            let mut mask = Mat::new_rows_cols_with_default(h, w, core::CV_8U, Scalar::all(0.0))?;
            let r = core::Rect::new(i * w / TILES.0, j * h / TILES.1, w / TILES.0 + 1, h / TILES.1 + 1) & core::Rect::new(0, 0, w, h);
            Mat::roi_mut(&mut mask, r)?.set_to(&Scalar::all(255.0), &core::no_array())?;
            let mut t = Vector::<KeyPoint>::new();
            o.detect(g, &mut t, &mask)?;
            k.extend(t);
        }
    }
    let mut d = Mat::default();
    ORB::create(k.len().max(1) as i32, 1.2, 8, 31, 0, 2, ORB_ScoreType::HARRIS_SCORE, 31, 10)?.compute(g, &mut k, &mut d)?;
    let desc = if d.rows() > 0 { util::bytes(&d)? } else { vec![] };
    Ok(Orb { pts: k.iter().map(|k| k.pt()).collect(), desc, n: d.rows() })
}

/// `page` warped onto the seed (`seed_px`, its pixels) by their printed
/// features; None if they don't match well (another printing, a bad
/// photo). One homography leaves a curled or folded page off by up to a few
/// text lines in places, and a table's repeating rows give few distinct
/// features to fit locally. So the homography is followed by dense optical
/// flow (DIS) between the seed and the warped page, smoothed over
/// `FLOW_SMOOTH` pixels: paper bends smoothly, and one page's handwriting,
/// which the other doesn't have, averages out.
fn onto(seed: &Orb, seed_px: &[u8], size: Size, page: &Mat) -> Result<Option<(Mat, usize)>> {
    const FLOW_SMOOTH: f64 = 25.0;
    let p = orb(page)?;
    if p.n < 2 || seed.n < 2 {
        return Ok(None);
    }
    let m = BFMatcher::create(core::NORM_HAMMING, false)?;
    let mut knn = Vector::<Vector<DMatch>>::new();
    m.knn_train_match(&seed.desc()?, &p.desc()?, &mut knn, 2, &core::no_array(), false)?;
    let good: Vec<DMatch> = knn
        .iter()
        .filter_map(|v| if v.len() >= 2 && v.get(0).ok()?.distance < 0.75 * v.get(1).ok()?.distance { v.get(0).ok() } else { None })
        .collect();
    if good.len() < 60 {
        return Ok(None);
    }
    let src: Vector<Point2f> = good.iter().map(|g| p.pts[g.train_idx as usize]).collect();
    let dst: Vector<Point2f> = good.iter().map(|g| seed.pts[g.query_idx as usize]).collect();
    let mut inl = Mat::default();
    // seed -> page
    let h = geometry::find_homography(&dst, &src, geometry::USAC_MAGSAC, 4.0, &mut inl, 5000, 0.999)?;
    if h.empty() {
        return Ok(None);
    }
    let keep = util::bytes(&inl)?.iter().filter(|&&v| v != 0).count();
    if keep < 60 || (keep as f64) < 0.2 * good.len() as f64 {
        return Ok(None);
    }
    let hg = util::arr3(&h)?;
    let (w, ht) = (size.width as f64, size.height as f64);
    let q: Vec<[f64; 2]> = [[0.0, 0.0], [w, 0.0], [w, ht], [0.0, ht]].iter().map(|&c| util::project(&hg, c)).collect();
    let poly: Vector<Point2f> = q.iter().map(|c| Point2f::new(c[0] as f32, c[1] as f32)).collect();
    let area = geometry::contour_area(&poly, false)?;
    if !geometry::is_contour_convex(&poly)? || !(0.1..3.0).contains(&(area / (page.cols() as f64 * page.rows() as f64))) {
        return Ok(None);
    }
    let mut warped = Mat::default();
    imgproc::warp_perspective(page, &mut warped, &h, size, imgproc::INTER_LINEAR | imgproc::WARP_INVERSE_MAP, core::BORDER_CONSTANT, Scalar::all(255.0), core::AlgorithmHint::ALGO_HINT_DEFAULT)?;
    let blur = |g: &Mat| -> Result<Mat> {
        let mut b = Mat::default();
        imgproc::gaussian_blur(g, &mut b, Size::new(0, 0), 1.5, 0.0, core::BORDER_REPLICATE, core::AlgorithmHint::ALGO_HINT_DEFAULT)?;
        Ok(b)
    };
    let mut dis = video::DISOpticalFlow::create(video::DISOpticalFlow_PRESET_MEDIUM)?;
    dis.set_finest_scale(1)?;
    let mut flow = Mat::default();
    dis.calc(&blur(&util::from_bytes(size.height, size.width, seed_px)?)?, &blur(&warped)?, &mut flow)?;
    let mut ch = Vector::<Mat>::new();
    core::split(&flow, &mut ch)?;
    let smooth = |m: Mat| -> Result<Mat> {
        let mut b = Mat::default();
        imgproc::gaussian_blur(&m, &mut b, Size::new(0, 0), FLOW_SMOOTH, 0.0, core::BORDER_REPLICATE, core::AlgorithmHint::ALGO_HINT_DEFAULT)?;
        Ok(b)
    };
    let (fx, fy) = (smooth(ch.get(0)?)?, smooth(ch.get(1)?)?);
    // one remap from the page itself: seed (x, y) -> warped (x, y) + flow -> page
    let mut mx = Mat::new_rows_cols_with_default(size.height, size.width, core::CV_32F, Scalar::all(0.0))?;
    let mut my = mx.clone();
    for y in 0..size.height {
        for x in 0..size.width {
            let g = util::project(&hg, [x as f64 + *fx.at_2d::<f32>(y, x)? as f64, y as f64 + *fy.at_2d::<f32>(y, x)? as f64]);
            *mx.at_2d_mut::<f32>(y, x)? = g[0] as f32;
            *my.at_2d_mut::<f32>(y, x)? = g[1] as f32;
        }
    }
    let mut out = Mat::default();
    imgproc::remap(page, &mut out, &mx, &my, imgproc::INTER_LINEAR, core::BORDER_CONSTANT, Scalar::all(255.0), core::AlgorithmHint::ALGO_HINT_DEFAULT)?;
    Ok(Some((out, keep)))
}

/// Ink: darker than the paper around it.
fn ink(g: &Mat) -> Result<Mat> {
    util::adaptive_inv(&util::gaussian(g, 3)?, 31, 18.0)
}

/// A signature column's heading as OCR may misread it ("Signaure").
static SIG_HEAD: once_cell::sync::Lazy<fancy_regex::Regex> =
    once_cell::sync::Lazy::new(|| fancy_regex::Regex::new(r"(?i)s[il1]gn|sig\w{0,4}ur").unwrap());

/// Where a profile peaks against its surroundings: the profile less its
/// running median over `half` on each side. A printed rule a few pixels
/// wide stands out of it however light its ink; writing and shading, wider
/// than a rule, don't.
fn ridge(p: &[f64], half: usize) -> Vec<f64> {
    (0..p.len())
        .map(|i| {
            let (a, b) = (i.saturating_sub(half), (i + half + 1).min(p.len()));
            p[i] - util::median(&p[a..b])
        })
        .collect()
}

/// The agents' table of the blank form, read from its printed column
/// headings and the darkness of its rules (`gray`: the blank form in the
/// reading's frame).
///
/// Its rules are often printed light grey, and where most people write (the
/// first rows) the ghosts of their writing are as dark: the reader's
/// thresholds lose them. But a rule is straight and runs the table's whole
/// height or width, so averaged along its length it is a clear ridge: down
/// the table between each two headings for the columns, across it for the
/// rows. The table ends where its outer rules do; its rows are equal.
/// Returns its ID, phone and signature columns and its rules, or None if no
/// heading row with at least two of them is found.
fn profile_table(l: &Layout, gray: &Mat) -> Result<Option<ProfileTable>> {
    let h = TEXT_H;
    let results_bottom = l
        .protected
        .iter()
        .filter(|(n, _)| n == "results")
        .flat_map(|(_, p)| p.iter().map(|q| q[1]))
        .fold(0.0, f64::max);
    let kind = |t: &str| -> Option<&'static str> {
        if generic::has(&generic::REFUSAL_RE, t) {
            None
        } else if generic::has(&generic::ID_RE, t) {
            Some("id")
        } else if generic::has(&generic::TEL_RE, t) {
            Some("phone")
        } else if generic::has(&generic::SIG_RE, t) || generic::has(&SIG_HEAD, t) {
            Some("signature")
        } else {
            None
        }
    };
    // the heading row: the topmost line below the results with two kinds
    let mut cands: Vec<&OLine> = l.lines.iter().filter(|o| o.cy() > results_bottom && kind(&o.text).is_some()).collect();
    cands.sort_by(|a, b| a.cy().partial_cmp(&b.cy()).unwrap());
    let mut head: Option<Vec<&OLine>> = None;
    for c in cands {
        let mut row: Vec<&OLine> = l.lines.iter().filter(|o| (o.cy() - c.cy()).abs() < 0.6 * h && o.bbox().2 - o.bbox().0 < 15.0 * h).collect();
        let kinds: BTreeSet<&str> = row.iter().filter_map(|o| kind(&o.text)).collect();
        if kinds.len() >= 2 {
            row.sort_by(|a, b| a.bbox().0.partial_cmp(&b.bbox().0).unwrap());
            row.dedup_by(|b, a| b.bbox().0 < a.bbox().2);
            head = Some(row);
            break;
        }
    }
    let Some(head) = head else { return Ok(None) };
    let (gw, gh) = (gray.cols(), gray.rows());
    let g = util::bytes(gray)?;
    let mut bg = Mat::default();
    imgproc::gaussian_blur(&util::dilate(gray, &util::ones_kernel(15, 15)?)?, &mut bg, Size::new(0, 0), 10.0, 0.0, core::BORDER_REPLICATE, core::AlgorithmHint::ALGO_HINT_DEFAULT)?;
    let bg = util::bytes(&bg)?;
    let dark = |x: i32, y: i32| -> f64 {
        let i = (y.clamp(0, gh - 1) * gw + x.clamp(0, gw - 1)) as usize;
        (bg[i] as f64 - g[i] as f64).max(0.0)
    };
    let top = head.iter().map(|o| o.bbox().1).fold(f64::INFINITY, f64::min);
    let head_bot = head.iter().map(|o| o.bbox().3).fold(f64::NEG_INFINITY, f64::max);
    // column rules: down the first rows, outside each heading and between each two
    let (ya, yb) = ((top - 0.3 * h).max(0.0) as i32, ((top + 8.0 * h) as i32).min(gh));
    let prof: Vec<f64> = (0..gw).map(|x| (ya..yb).map(|y| dark(x, y)).sum::<f64>() / (yb - ya).max(1) as f64).collect();
    let rx = ridge(&prof, 6);
    let peak = |a: f64, b: f64| -> Option<(f64, f64)> {
        let (a, b) = (a.max(0.0) as usize, (b.min(gw as f64 - 1.0)) as usize);
        (a..=b.max(a)).map(|x| (x as f64, rx[x.min(rx.len() - 1)])).max_by(|p, q| p.1.partial_cmp(&q.1).unwrap())
    };
    let strong = |p: Option<(f64, f64)>| p.filter(|p| p.1 >= 3.0).map(|p| p.0);
    let first = head[0].bbox();
    let last = head[head.len() - 1].bbox();
    let mut xs: Vec<Option<f64>> = vec![strong(peak(first.0 - 2.5 * h, first.0 + 0.2 * h))];
    for p in head.windows(2) {
        let (a, b) = (p[0].bbox(), p[1].bbox());
        xs.push(strong(peak(a.2 - 0.2 * h, b.0 + 0.2 * h)).or(Some((a.2 + b.0) / 2.0)));
    }
    xs.push(strong(peak(last.2 - 0.2 * h, last.2 + 2.5 * h)));
    let (Some(xl), Some(xr)) = (xs[0], xs[xs.len() - 1]) else { return Ok(None) };
    // the table's bottom: where its outer rules end, followed down the page
    // (a photo is seldom quite square: they drift a few pixels)
    let side = |x: i32, y: i32| -> f64 {
        let d = |c: i32| (y - 4..=y + 4).map(|r| dark(c, r)).sum::<f64>() / 9.0;
        (x - 1..=x + 1).map(d).sum::<f64>() / 3.0 - ((x - 9..=x - 6).chain(x + 6..=x + 9)).map(d).sum::<f64>() / 8.0
    };
    let follow = |x0: f64| -> Vec<f64> {
        let mut x = x0.round() as i32;
        (0..gh)
            .map(|y| {
                if y < head_bot as i32 {
                    return side(x, y);
                }
                let (k, v) = (-2..=2).map(|d| (d, side(x + d, y))).max_by(|a, b| a.1.partial_cmp(&b.1).unwrap()).unwrap();
                if v > 0.0 {
                    x += k;
                }
                v
            })
            .collect()
    };
    let (el, er) = (follow(xl), follow(xr));
    let edge: Vec<f64> = el.iter().zip(&er).map(|(a, b)| a.max(*b)).collect();
    let y0 = head_bot as i32;
    let level = util::median(&edge[y0.clamp(0, gh - 1) as usize..((head_bot + 3.0 * h) as i32).clamp(1, gh) as usize]);
    if level < 3.0 {
        return Ok(None);
    }
    let mut bottom = gh - 1;
    let mut gap = 0;
    for y in y0..gh {
        let v = edge[(y - 3).max(0) as usize..(y + 4).min(gh) as usize].iter().sum::<f64>() / 7.0;
        gap = if v < 0.35 * level { gap + 1 } else { 0 };
        if gap as f64 > 0.8 * h {
            bottom = y - gap;
            break;
        }
    }
    // rows: rules across the table, from the heading's to the bottom one
    let rowp: Vec<f64> = (0..gh).map(|y| ((xl + 4.0) as i32..(xr - 4.0) as i32).map(|x| dark(x, y)).sum::<f64>() / (xr - xl - 8.0).max(1.0)).collect();
    let ry = ridge(&rowp, 6);
    let peak_y = |a: f64, b: f64| -> f64 {
        let (a, b) = (a.max(0.0) as usize, (b.min(gh as f64 - 1.0)) as usize);
        (a..=b.max(a)).max_by(|&p, &q| ry[p].partial_cmp(&ry[q]).unwrap()).unwrap_or(a) as f64
    };
    let hb = peak_y(head_bot - 0.2 * h, head_bot + 1.2 * h);
    // the bottom border: drawn heavier than the rows, near where the sides end
    let tb = peak_y(bottom as f64 - 2.5 * h, bottom as f64 + 0.3 * h);
    let peaks: Vec<f64> = ((hb + 0.5 * h) as usize..(tb - 0.5 * h).max(hb + 0.5 * h) as usize)
        .filter(|&y| ry[y] > 1.0 && ry[y] >= ry[y.saturating_sub(6)..(y + 7).min(ry.len())].iter().cloned().fold(f64::NEG_INFINITY, f64::max))
        .map(|y| y as f64)
        .collect();
    let mut steps: Vec<f64> = std::iter::once(hb).chain(peaks.iter().cloned()).chain(std::iter::once(tb)).collect::<Vec<_>>().windows(2).map(|p| p[1] - p[0]).filter(|&d| d > 0.6 * h && d < 1.6 * h).collect();
    if steps.is_empty() {
        steps.push(tb - hb);
    }
    let n = ((tb - hb) / util::median(&steps)).round().max(1.0) as usize;
    let pitch = (tb - hb) / n as f64;
    if !(0.6 * h..=1.8 * h).contains(&pitch) {
        return Ok(None);
    }
    let mut fields = Vec::new();
    for (i, o) in head.iter().enumerate() {
        let Some(k) = kind(&o.text) else { continue };
        let (Some(a), Some(b)) = (xs[i], xs[i + 1]) else { continue };
        let zones = (0..n)
            .map(|r| {
                let mut z = Zone::new(k, [a, hb + r as f64 * pitch, b - a, pitch], if k == "signature" { "signature" } else { "partial" });
                z.row = r;
                z
            })
            .collect();
        fields.push(Field { kind: k.to_string(), zones, source: "table".into(), label: o.text.clone(), label_box: o.bbox() });
    }
    // every rule, followed along its length (a photo is seldom quite
    // square): down from the table's top border, across from side to side
    let ridge_at = |x: i32, y: i32, down: bool| -> f64 {
        let d = |a: i32, b: i32| if down { dark(a, b) } else { dark(b, a) };
        let (u, v) = if down { (x, y) } else { (y, x) };
        (v - 4..=v + 4).map(|t| (u - 1..=u + 1).map(|c| d(c, t)).sum::<f64>() / 3.0 - ((u - 7..=u - 5).chain(u + 5..=u + 7)).map(|c| d(c, t)).sum::<f64>() / 6.0).sum::<f64>() / 9.0
    };
    // its offset from straight, window by window along it: paper bends
    // slowly, and print it passes (labels, writing) shouldn't pull it
    let trace = |at: f64, from: f64, to: f64, down: bool| -> Vec<[f64; 2]> {
        const WIN: i32 = 40;
        let (a, b) = (from as i32, to as i32);
        let centres: Vec<i32> = (a..=b).step_by(WIN as usize).map(|c| (c + WIN / 2).min(b)).collect();
        let u0 = at.round() as i32;
        let raw: Vec<i32> = centres
            .iter()
            .map(|&c| (-6..=6).max_by(|&d, &e| {
                let sum = |d: i32| (c - WIN / 2..c + WIN / 2).map(|v| ridge_at(u0 + d, v, down)).sum::<f64>();
                sum(d).partial_cmp(&sum(e)).unwrap()
            }).unwrap())
            .collect();
        let mut off: Vec<i32> = (0..raw.len())
            .map(|i| {
                let mut w: Vec<i32> = raw[i.saturating_sub(1)..(i + 2).min(raw.len())].to_vec();
                w.sort();
                w[w.len() / 2]
            })
            .collect();
        for i in 1..off.len() {
            off[i] = off[i].clamp(off[i - 1] - 2, off[i - 1] + 2);
        }
        let mut pts: Vec<[f64; 2]> = Vec::new();
        let at_v = |v: f64, o: f64| if down { [u0 as f64 + o, v] } else { [v, u0 as f64 + o] };
        pts.push(at_v(a as f64, *off.first().unwrap_or(&0) as f64));
        for (c, o) in centres.iter().zip(&off) {
            pts.push(at_v(*c as f64, *o as f64));
        }
        pts.push(at_v(b as f64, *off.last().unwrap_or(&0) as f64));
        pts
    };
    let ht = peak_y(top - 1.2 * h, top + 0.1 * h);
    let mut rules: Vec<Vec<[f64; 2]>> = xs.iter().flatten().map(|&x| trace(x, ht, tb, true)).collect();
    rules.push(trace(ht, xl, xr, false));
    for r in 0..=n {
        rules.push(trace(hb + r as f64 * pitch, xl, xr, false));
    }
    Ok(Some(ProfileTable { fields, rules, rect: [xl, ht, xr - xl, tb - ht], rows: (0..=n).map(|r| hb + r as f64 * pitch).collect() }))
}

/// An agents' table read from the darkness of its rules (`profile_table`).
struct ProfileTable {
    /// Its ID, phone and signature columns.
    fields: Vec<Field>,
    /// Its rules as polylines (x, y), each followed along its length.
    rules: Vec<Vec<[f64; 2]>>,
    /// Its outer border, [x, y, w, h].
    rect: [f64; 4],
    /// Its row rules, top (the heading's) to bottom.
    rows: Vec<f64>,
}

/// How much of an A-series form's structure a reading found: the kinds of
/// its agents' table columns, then the officers' signature lines. (A page's
/// own reading also finds its writing, as free text: not structure.)
fn structure(l: &Layout) -> (usize, usize) {
    let kinds: BTreeSet<&str> = l.fields.iter().filter(|f| f.source == "table").map(|f| f.kind.as_str()).collect();
    (kinds.len(), l.fields.iter().filter(|f| f.source == "declaration").count())
}

/// `l`'s agents' table taken from the rules of the blank form `median`
/// (`profile_table`) where the reader found less of it: its columns
/// replace the reader's, and what the reader made of their headings
/// (fields after them taken for labels, a table it couldn't complete) is
/// dropped.
fn with_profile_table(l: &mut Layout, median: &Mat) -> Result<()> {
    let found: BTreeSet<String> = l.fields.iter().filter(|f| f.source == "table").map(|f| f.kind.clone()).collect();
    if found.len() == 3 {
        return Ok(());
    }
    let mut bgr = Mat::default();
    imgproc::cvt_color_def(median, &mut bgr, imgproc::COLOR_GRAY2BGR)?;
    let gray = util::gray(&generic::warp(&bgr, &l.frame)?)?;
    let Some(ProfileTable { fields: cols, .. }) = profile_table(l, &gray)? else { return Ok(()) };
    let kinds: BTreeSet<String> = cols.iter().map(|f| f.kind.clone()).collect();
    if kinds.len() <= found.len() {
        return Ok(());
    }
    let (y0, y1) = cols.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |a, f| (a.0.min(f.label_box.1), a.1.max(f.label_box.3)));
    let heads: Vec<String> = l.lines.iter().filter(|o| o.cy() > y0 && o.cy() < y1).map(|o| format!("'{}'", o.text)).collect();
    l.fields.retain(|f| f.source != "table" && !(f.source == "label" && (f.label_box.1 + f.label_box.3) / 2.0 > y0 && (f.label_box.1 + f.label_box.3) / 2.0 < y1));
    l.fields.extend(cols);
    l.problems.retain(|p| !p.starts_with("agents table") && !heads.iter().any(|t| p.contains(t.as_str())));
    l.notes.push(format!("agents' table read from its rules on the blank form ({})", kinds.into_iter().collect::<Vec<_>>().join(", ")));
    Ok(())
}

/// The blank form and where people write, from the aligned pages.
pub(crate) struct Blank {
    /// The pixel-wise median of the pages: the reference pages are matched
    /// against (its print sharp where they align, its noise averaged out,
    /// little writing left but where most people write).
    median: Mat,
    /// Print found on most pages (where people rarely write).
    print: Mat,
    /// Share of the pages with writing, per pixel.
    freq: Vec<f32>,
    /// Each page's writing (255), per pixel.
    writing: Vec<Vec<u8>>,
}

/// The blank form and where people write.
///
/// Writing is told from print pixel by pixel, without deciding first what is
/// print: print is as dark on nearly every page, so 85% of the pages are at
/// least about as dark there (`level`, taken within a pixel for pages a
/// pixel off). A page has writing where it is much darker than that. Where
/// most people write (officers' names, the first agents' rows) their
/// strokes still differ from page to page: no pixel is inked on more than
/// about 70% of them, and `level` there is paper.
///
/// The print mask (for reading the blank form) is where at least
/// `print_share` of the pages have ink within two pixels and few people
/// write, or the rules of tables where they do.
fn blank(stack: &[Mat], print_share: f64) -> Result<Blank> {
    let n = stack.len();
    let (h, w) = (stack[0].rows(), stack[0].cols());
    let len = (h * w) as usize;
    let px: Vec<Vec<u8>> = stack.iter().map(util::bytes).collect::<Result<_>>()?;
    let mut med = vec![255u8; len];
    let mut p85 = vec![255u8; len];
    let mut buf = vec![0u8; n];
    let (mid, hi) = (n / 2, ((0.85 * (n - 1) as f64).round() as usize).min(n - 1));
    for i in 0..len {
        for (k, p) in px.iter().enumerate() {
            buf[k] = p[i];
        }
        buf.select_nth_unstable(mid);
        med[i] = buf[mid];
        buf.select_nth_unstable(hi);
        p85[i] = buf[hi];
    }
    let level = util::bytes(&util::morph(&util::gaussian(&util::from_bytes(h, w, &p85)?, 3)?, imgproc::MORPH_ERODE, &util::ones_kernel(3, 3)?)?)?;
    let writing: Vec<Vec<u8>> = stack
        .par_iter()
        .map(|g| -> Result<Vec<u8>> {
            let s = util::bytes(&util::gaussian(g, 3)?)?;
            let wr: Vec<u8> = (0..len).map(|i| if (s[i] as i32) + 40 < level[i] as i32 { 255 } else { 0 }).collect();
            // specks (noise, a pixel's misalignment on a stroke) aren't writing
            util::bytes(&util::morph(&util::from_bytes(h, w, &wr)?, imgproc::MORPH_OPEN, &util::ones_kernel(2, 2)?)?)
        })
        .collect::<Result<_>>()?;
    let mut freq = vec![0f32; len];
    for wr in &writing {
        for i in 0..len {
            if wr[i] != 0 {
                freq[i] += 1.0;
            }
        }
    }
    for f in freq.iter_mut() {
        *f /= n as f32;
    }
    // print: inked (within two pixels) on most pages, away from where people
    // often write but for the rules there
    let mut count = vec![0u8; len];
    for g in stack {
        let near = util::bytes(&util::dilate(&ink(g)?, &util::ones_kernel(5, 5)?)?)?;
        for i in 0..len {
            if near[i] != 0 {
                count[i] = count[i].saturating_add(1);
            }
        }
    }
    let need = (print_share * n as f64).ceil() as u8;
    let loose = util::from_bytes(h, w, &count.iter().map(|&c| if c >= need { 255 } else { 0 }).collect::<Vec<u8>>())?;
    let hot = util::bytes(&util::dilate(
        &util::morph(&util::from_bytes(h, w, &freq.iter().map(|&f| if f >= 0.12 { 255 } else { 0 }).collect::<Vec<u8>>())?, imgproc::MORPH_CLOSE, &util::ones_kernel(9, 9)?)?,
        &util::ones_kernel((0.5 * TEXT_H) as i32, (0.5 * TEXT_H) as i32)?,
    )?)?;
    let rules = util::bytes(&util::dilate(&rule_mask(&loose)?, &util::ones_kernel(3, 3)?)?)?;
    let lb = util::bytes(&loose)?;
    let print: Vec<u8> = (0..len).map(|i| if lb[i] != 0 && (hot[i] == 0 || rules[i] != 0) { 255 } else { 0 }).collect();
    Ok(Blank {
        median: util::from_bytes(h, w, &med)?,
        print: util::from_bytes(h, w, &print)?,
        freq,
        writing,
    })
}

/// Printed rules in a binary image: long straight runs, across or down.
fn rule_mask(bin: &Mat) -> Result<Mat> {
    util::bitor(
        &util::morph(bin, imgproc::MORPH_OPEN, &util::rect_kernel((2.0 * TEXT_H) as i32, 1)?)?,
        &util::morph(bin, imgproc::MORPH_OPEN, &util::rect_kernel(1, (1.2 * TEXT_H) as i32)?)?,
    )
}

/// The fields of the blank form hold nothing printed but their rules (and
/// the labels listed in `keep`). Any ink left in `areas` is erased from the
/// reference, to its paper's tone: masking takes ink a page shares with the
/// reference for print and leaves it, so a ghost of many people's writing
/// left there would leave real writing unmasked. The rules stay (the tables
/// are matched by them): long thin runs, which no ghost is (ghosts are
/// blots); and so does anything too light to count as ink, light grey
/// rules among it.
fn clean(reference: &Mat, areas: &[[f64; 4]], keep: &[[f64; 4]]) -> Result<Mat> {
    let (w, h) = (reference.cols(), reference.rows());
    let ink_m = ink(reference)?;
    let (lh, lv) = ((2.0 * TEXT_H) as i32, (1.2 * TEXT_H) as i32);
    let thin = |long: Mat, thick: Mat| -> Result<Mat> {
        let mut out = Mat::default();
        core::bitwise_and(&long, &util::bitnot(&util::dilate(&thick, &util::ones_kernel(5, 5)?)?)?, &mut out, &core::no_array())?;
        Ok(out)
    };
    let hor = thin(util::morph(&ink_m, imgproc::MORPH_OPEN, &util::rect_kernel(lh, 1)?)?, util::morph(&ink_m, imgproc::MORPH_OPEN, &util::rect_kernel(lh, 7)?)?)?;
    let ver = thin(util::morph(&ink_m, imgproc::MORPH_OPEN, &util::rect_kernel(1, lv)?)?, util::morph(&ink_m, imgproc::MORPH_OPEN, &util::rect_kernel(7, lv)?)?)?;
    let rules = util::bytes(&util::dilate(&util::bitor(&hor, &ver)?, &util::ones_kernel(3, 3)?)?)?;
    let inked = util::bytes(&util::dilate(&ink_m, &util::ones_kernel(3, 3)?)?)?;
    let mut paper = Mat::default();
    imgproc::gaussian_blur(&util::dilate(reference, &util::ones_kernel(15, 15)?)?, &mut paper, Size::new(0, 0), 10.0, 0.0, core::BORDER_REPLICATE, core::AlgorithmHint::ALGO_HINT_DEFAULT)?;
    let paper = util::bytes(&paper)?;
    let mut px = util::bytes(reference)?;
    let inside = |r: &[f64; 4], x: i32, y: i32| (x as f64) >= r[0] && (x as f64) < r[0] + r[2] && (y as f64) >= r[1] && (y as f64) < r[1] + r[3];
    for y in 0..h {
        for x in 0..w {
            let i = (y * w + x) as usize;
            if inked[i] == 0 || rules[i] != 0 || !areas.iter().any(|r| inside(r, x, y)) || keep.iter().any(|r| inside(r, x, y)) {
                continue;
            }
            px[i] = paper[i];
        }
    }
    util::from_bytes(h, w, &px)
}

/// Bounding box [x, y, w, h] of `r` (working frame of the reading) in reference pixels.
fn to_ref(m: &M3, r: [f64; 4]) -> [f64; 4] {
    let [x, y, w, h] = r;
    let p: Vec<[f64; 2]> = [[x, y], [x + w, y], [x + w, y + h], [x, y + h]].iter().map(|&q| util::project(m, q)).collect();
    let x0 = p.iter().map(|q| q[0]).fold(f64::INFINITY, f64::min);
    let y0 = p.iter().map(|q| q[1]).fold(f64::INFINITY, f64::min);
    let x1 = p.iter().map(|q| q[0]).fold(f64::NEG_INFINITY, f64::max);
    let y1 = p.iter().map(|q| q[1]).fold(f64::NEG_INFINITY, f64::max);
    [x0, y0, x1 - x0, y1 - y0]
}

fn box_ref(m: &M3, b: (f64, f64, f64, f64)) -> [f64; 4] {
    to_ref(m, [b.0, b.1, b.2 - b.0, b.3 - b.1])
}

fn r1(v: f64) -> f64 {
    (v * 10.0).round() / 10.0
}

fn rect_yaml(r: [f64; 4]) -> String {
    format!("[{}, {}, {}, {}]", r1(r[0]), r1(r[1]), r1(r[2]), r1(r[3]))
}

/// Share of pages with writing in `r` (reference px): at least `min_px`
/// writing pixels there.
fn written(stack_writing: &[Vec<u8>], w: i32, h: i32, r: [f64; 4], min_px: usize) -> f64 {
    let (x0, y0) = (r[0].max(0.0) as i32, r[1].max(0.0) as i32);
    let (x1, y1) = (((r[0] + r[2]) as i32).min(w), ((r[1] + r[3]) as i32).min(h));
    if x1 <= x0 || y1 <= y0 || stack_writing.is_empty() {
        return 0.0;
    }
    let mut pages = 0;
    for wr in stack_writing {
        let mut c = 0;
        for y in y0..y1 {
            let row = (y * w) as usize;
            for x in x0..x1 {
                if wr[row + x as usize] != 0 {
                    c += 1;
                }
            }
        }
        if c >= min_px {
            pages += 1;
        }
    }
    pages as f64 / stack_writing.len() as f64
}

/// What was learned for one printing.
pub struct Learned {
    pub name: String,
    pub dir: PathBuf,
    pub record: Value,
}

/// Learn the layout of one printing from `sources` (its files) and write it
/// as a template under `out`/`name`.
pub fn learn(sources: &[String], form: &str, name: &str, out: &Path, opts: &LearnOptions) -> Result<Learned> {
    // spread the pages used over the group
    let step = (sources.len() as f64 / opts.max_pages as f64).max(1.0);
    let picked: Vec<&String> = (0..sources.len().min(opts.max_pages)).map(|k| &sources[(k as f64 * step) as usize]).collect();
    let pages: Vec<(String, Result<Option<Upright>>)> = picked.par_iter().map(|s| (s.to_string(), upright(s))).collect();
    let mut ups = Vec::new();
    let mut rejected: Vec<Value> = Vec::new();
    for (s, r) in pages {
        match r {
            Ok(Some(u)) => ups.push(u),
            Ok(None) => rejected.push(json!({"file": s, "why": "no printed text found to turn it upright"})),
            Err(e) => rejected.push(json!({"file": s, "why": e.to_string()})),
        }
    }
    if ups.len() < opts.min_pages {
        bail!("{}: only {} page(s) could be turned upright (need {})", name, ups.len(), opts.min_pages);
    }
    // the seed: sharp and square to the frame
    let si = (0..ups.len())
        .max_by(|&a, &b| {
            let s = |u: &Upright| u.sharp / (1.0 + u.skew.abs());
            s(&ups[a]).partial_cmp(&s(&ups[b])).unwrap()
        })
        .unwrap();
    let seed = &ups[si];
    let size = Size::new(seed.w, seed.h);
    let seed_orb = orb(&seed.gray()?)?;
    let aligned: Vec<(String, Result<Option<Vec<u8>>>)> = ups
        .par_iter()
        .enumerate()
        .map(|(k, u)| {
            if k == si {
                return (u.src.clone(), Ok(Some(u.px.clone())));
            }
            let r = (|| -> Result<Option<Vec<u8>>> {
                Ok(match onto(&seed_orb, &seed.px, size, &u.gray()?)? {
                    Some((m, _)) => Some(util::bytes(&m)?),
                    None => None,
                })
            })();
            (u.src.clone(), r)
        })
        .collect();
    let mut stack = Vec::new();
    let mut used = Vec::new();
    let mut seed_at = 0;
    for (s, r) in aligned {
        match r {
            Ok(Some(px)) => {
                if s == seed.src {
                    seed_at = stack.len();
                }
                stack.push(util::from_bytes(seed.h, seed.w, &px)?);
                used.push(s);
            }
            Ok(None) => rejected.push(json!({"file": s, "why": "doesn't align with the others (another printing, or a poor photo)"})),
            Err(e) => rejected.push(json!({"file": s, "why": e.to_string()})),
        }
    }
    if stack.len() < opts.min_pages {
        bail!("{}: only {} page(s) align (need {})", name, stack.len(), opts.min_pages);
    }
    if let Ok(d) = std::env::var("RFPII_LEARN_DEBUG") {
        let d = Path::new(&d);
        std::fs::create_dir_all(d)?;
        for (k, m) in stack.iter().enumerate() {
            imgcodecs::imwrite(&d.join(format!("aligned_{:02}{}.png", k, if k == seed_at { "_seed" } else { "" })), m, &Vector::new())?;
        }
    }
    let Blank { median, print, freq, writing } = blank(&stack, opts.print_share)?;
    // The template is the form: the frame is cut to its print, with a text
    // line's margin. What lies around it on the first page (a table, the
    // floor) is not on the others, and pages are scored against it.
    let (fw, fh) = (median.cols(), median.rows());
    let cut = {
        let pb = util::bytes(&util::morph(&print, imgproc::MORPH_OPEN, &util::ones_kernel(3, 3)?)?)?;
        let cols: Vec<usize> = (0..fw as usize).map(|x| (0..fh as usize).filter(|&y| pb[y * fw as usize + x] != 0).count()).collect();
        let rows: Vec<usize> = (0..fh as usize).map(|y| (0..fw as usize).filter(|&x| pb[y * fw as usize + x] != 0).count()).collect();
        // the extent of the print but its outer half percent each side
        let span = |v: &[usize]| -> (i32, i32) {
            let total: usize = v.iter().sum();
            let (mut acc, mut a, mut b) = (0usize, 0usize, v.len());
            for (i, &c) in v.iter().enumerate() {
                acc += c;
                if acc * 200 < total {
                    a = i + 1;
                }
                if acc * 200 <= total * 199 {
                    b = i + 1;
                }
            }
            (a as i32, b as i32)
        };
        let ((x0, x1), (y0, y1)) = (span(&cols), span(&rows));
        let m = TEXT_H as i32;
        core::Rect::new((x0 - m).max(0), (y0 - m).max(0), (x1 + m).min(fw) - (x0 - m).max(0), (y1 + m).min(fh) - (y0 - m).max(0))
    };
    let crop_m = |m: &Mat| -> Result<Mat> { Ok(Mat::roi(m, cut)?.try_clone()?) };
    let crop_v = |v: &[u8]| -> Vec<u8> { (cut.y..cut.y + cut.height).flat_map(|y| (cut.x..cut.x + cut.width).map(move |x| (y * fw + x) as usize)).map(|i| v[i]).collect() };
    let (median, print) = (crop_m(&median)?, crop_m(&print)?);
    let freq: Vec<f32> = (cut.y..cut.y + cut.height).flat_map(|y| (cut.x..cut.x + cut.width).map(move |x| (y * fw + x) as usize)).map(|i| freq[i]).collect();
    let writing: Vec<Vec<u8>> = writing.iter().map(|v| crop_v(v)).collect();
    let stack: Vec<Mat> = stack.iter().map(|m| crop_m(m)).collect::<Result<_>>()?;
    let reference = median.try_clone()?;
    let (w, h) = (reference.cols(), reference.rows());
    // written first, so a printing that can't be learned can be looked at
    let dir = out.join(slug(name));
    std::fs::create_dir_all(&dir)?;
    let _ = std::fs::remove_file(dir.join("template.yaml"));
    imgcodecs::imwrite(&dir.join("reference.png"), &reference, &Vector::new())?;
    imgcodecs::imwrite(&dir.join("median.png"), &median, &Vector::new())?;
    let heat: Vec<u8> = freq.iter().map(|f| (255.0 - (f * 3.0).min(1.0) * 255.0) as u8).collect();
    imgcodecs::imwrite(&dir.join("writing.png"), &util::from_bytes(h, w, &heat)?, &Vector::new())?;

    // Read the form. The blank form's print is as dark as on the photos, the
    // ghosts of writing and the security print light grey, but light print
    // (some tables' rules) is light grey too, and where most people write
    // (the first agents' rows) the ghosts are as dark as print. So it is read
    // in a few versions, and so are the sharpest pages themselves (aligned:
    // in the same frame), whose rules are as clear as photographed. The
    // reading without problems that finds the most of the form's structure
    // is taken, the blank form's first; the fields are then checked against
    // where people wrote on all the pages.
    let md = util::bytes(&median)?;
    let pm = util::bytes(&print)?;
    let mut versions: Vec<(String, Vec<u8>)> = vec![
        ("dark ink".into(), md.iter().map(|&v| if v < 170 { v } else { 255 }).collect()),
        ("ink".into(), md.iter().map(|&v| if v < 205 { v } else { 255 }).collect()),
        ("median".into(), md.clone()),
        ("print found on most pages".into(), md.iter().zip(&pm).map(|(&v, &p)| if p != 0 { v } else { 255 }).collect()),
    ];
    let sharp: Vec<f64> = stack.iter().map(sharpness).collect::<Result<_>>()?;
    let mut by_sharp: Vec<usize> = (0..stack.len()).collect();
    by_sharp.sort_by(|&a, &b| sharp[b].partial_cmp(&sharp[a]).unwrap());
    for &k in by_sharp.iter().take(READ_PAGES) {
        versions.push((format!("page {}", used[k]), util::bytes(&stack[k])?));
    }
    let debug = std::env::var("RFPII_LEARN_DEBUG").ok();
    let readings: Vec<(String, Result<Analysis>)> = versions
        .par_iter()
        .enumerate()
        .map(|(i, (vname, px))| {
            let r = (|| -> Result<Analysis> {
                let mut bgr = Mat::default();
                imgproc::cvt_color_def(&util::from_bytes(h, w, px)?, &mut bgr, imgproc::COLOR_GRAY2BGR)?;
                if let Some(d) = &debug {
                    imgcodecs::imwrite(&Path::new(d).join(format!("read_{:02}.png", i)), &bgr, &Vector::new())?;
                }
                let mut a = ocr::with_engine(|o| generic::analyze(&bgr, o))?;
                if let Analysis::Layout(l) = &mut a {
                    with_profile_table(l, &median)?;
                }
                Ok(a)
            })();
            (vname.clone(), r)
        })
        .collect();
    let mut tried: Vec<String> = Vec::new();
    let mut read: Option<(Box<Layout>, String)> = None;
    for (vname, r) in readings {
        match r? {
            Analysis::Fail(why) => tried.push(format!("{}: {}", vname, why)),
            Analysis::Layout(l) if !l.problems.is_empty() => tried.push(format!("{}: {}", vname, l.problems.join("; "))),
            Analysis::Layout(l) => {
                tried.push(format!("{}: {} field(s)", vname, l.fields.len()));
                // a page's own reading only where the blank form's finds less
                let rank = |l: &Layout, v: &str| (structure(l).0, !v.starts_with("page "), structure(l).1);
                if read.as_ref().map_or(true, |r| rank(&l, &vname) > rank(&r.0, &r.1)) {
                    read = Some((l, vname));
                }
            }
        }
    }
    let Some((l, read_from)) = read else {
        bail!("{}: the blank form can't be read ({})", name, tried.join(" | "));
    };
    let m = l.frame.to_image()?;
    // printed text height in the reference: every page was scaled to TEXT_H
    // when turned upright (OCR boxes on the softer median measure smaller)
    let text_h = TEXT_H;
    let line_box = |ln: &OLine| box_ref(&m, ln.bbox());

    // each page's writing, for checking the fields against it

    // results: everything above the declaration
    let results_bottom = l
        .protected
        .iter()
        .filter(|(n, _)| n == "results")
        .flat_map(|(_, p)| p.iter().map(|q| util::project(&m, *q)[1]))
        .fold(f64::NEG_INFINITY, f64::max);
    if !results_bottom.is_finite() {
        bail!("{}: no results region found on the blank form", name);
    }
    let results_bottom = results_bottom.min(h as f64);

    let mut yaml = String::new();
    let mut fields_rec: Vec<Value> = Vec::new();
    let mut notes: Vec<String> = Vec::new();
    yaml += &format!(
        "# Learned by rfpii {} from {} pages of {} (no one drew this template).\n# reference.png is the pixel-wise median of those pages, its fields emptied;\n# writing.png shows where people wrote on them. learned.json lists the pages and checks.\n",
        env!("CARGO_PKG_VERSION"),
        stack.len(),
        name
    );
    // twice the features a photographed reference needs: a median's are
    // fewer and softer
    yaml += &format!("name: {}\nform: {}\nreference: reference.png\npadding: 8\nlearned: true\nn_features: 10000\n", slug(name), form);
    yaml += &format!("protect:\n  - name: results\n    rect: {}\n", rect_yaml([0.0, 0.0, w as f64, results_bottom]));
    yaml += &format!("review_area: {}\n", rect_yaml([0.0, results_bottom, w as f64, h as f64 - results_bottom]));
    yaml += &format!("text_h: {}\n", r1(text_h));

    // tables: the PII columns, grouped by the table they head
    let mut cols: Vec<&Field> = l.fields.iter().filter(|f| f.source == "table").collect();
    cols.sort_by(|a, b| a.zones[0].rect[1].partial_cmp(&b.zones[0].rect[1]).unwrap());
    let mut tables: Vec<Vec<&Field>> = Vec::new();
    for f in cols {
        let (y0, y1) = (f.zones[0].rect[1], f.zones.last().unwrap().rect[1] + f.zones.last().unwrap().rect[3]);
        match tables.iter_mut().find(|t| {
            let (a, b) = (t[0].zones[0].rect[1], t[0].zones.last().unwrap().rect[1] + t[0].zones.last().unwrap().rect[3]);
            y0 < b && a < y1
        }) {
            Some(t) => t.push(f),
            None => tables.push(vec![f]),
        }
    }
    let (hs, vs) = generic::rules(&l.gray, &[])?;
    // the agents' table from its rules on the blank form: its border and
    // rows, where the reader's thresholds lose its light grey rules
    let mut bgr = Mat::default();
    imgproc::cvt_color_def(&median, &mut bgr, imgproc::COLOR_GRAY2BGR)?;
    let profiled = profile_table(&l, &util::gray(&generic::warp(&bgr, &l.frame)?)?)?;
    let mut structure = String::new();
    let mut zones = String::new();
    let mut lowest = results_bottom;
    let mut table_spans: Vec<(f64, f64)> = Vec::new(); // reference y, header top to last row
    // the fields' areas, erased from the reference but for their rules and these labels
    let mut erase: Vec<[f64; 4]> = Vec::new();
    let mut keep: Vec<[f64; 4]> = Vec::new();
    for (ti, t) in tables.iter().enumerate() {
        let tname = if ti == 0 { "agents".to_string() } else { format!("agents_{}", ti + 1) };
        // working frame: header top to last row, between the outer rules
        let top = t.iter().map(|f| f.label_box.1).fold(f64::INFINITY, f64::min) - 0.6 * TEXT_H;
        let head = t.iter().map(|f| f.label_box.3).fold(f64::NEG_INFINITY, f64::max);
        let bot = t.iter().map(|f| f.zones.last().unwrap().rect[1] + f.zones.last().unwrap().rect[3]).fold(f64::NEG_INFINITY, f64::max);
        let span: Vec<f64> = vs.iter().filter(|v| v.lo <= top + 2.0 * TEXT_H && v.hi >= bot - 2.0 * TEXT_H).map(|v| v.at((top + bot) / 2.0)).collect();
        let fx0 = t.iter().map(|f| f.zones[0].rect[0]).fold(f64::INFINITY, f64::min);
        let fx1 = t.iter().map(|f| f.zones[0].rect[0] + f.zones[0].rect[2]).fold(f64::NEG_INFINITY, f64::max);
        let pt = profiled.as_ref().filter(|p| p.rect[1] < bot && p.rect[1] + p.rect[3] > top);
        let (x0, x1) = if let Some(p) = pt {
            (p.rect[0] - 0.3 * TEXT_H, p.rect[0] + p.rect[2] + 0.3 * TEXT_H)
        } else if span.len() >= 2 {
            (span.iter().cloned().fold(f64::INFINITY, f64::min) - 0.3 * TEXT_H, span.iter().cloned().fold(f64::NEG_INFINITY, f64::max) + 0.3 * TEXT_H)
        } else {
            (fx0 - 8.0 * TEXT_H, fx1 + 3.0 * TEXT_H)
        };
        // the table's rows, shared by its columns: the rules across most of
        // its width, below the header
        let xm = (x0 + x1) / 2.0;
        let mut ys: Vec<f64> = hs
            .iter()
            .filter(|r| r.lo <= x0 + 0.3 * (x1 - x0) && r.hi >= x1 - 0.3 * (x1 - x0))
            .map(|r| r.at(xm))
            .filter(|&y| y > head && y < bot + 1.5 * TEXT_H)
            .collect();
        ys.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mut lines_y: Vec<f64> = Vec::new();
        for y in ys {
            if lines_y.last().map(|&l| y - l < 0.35 * TEXT_H).unwrap_or(false) {
                continue;
            }
            lines_y.push(y);
        }
        let mut bands: Vec<(f64, f64)> = match pt {
            Some(p) => p.rows.windows(2).map(|r| (r[0], r[1])).collect(),
            None => lines_y.windows(2).map(|p| (p[0], p[1])).collect(),
        };
        if !bands.is_empty() {
            let pitch = util::median(&bands.iter().map(|b| b.1 - b.0).collect::<Vec<_>>());
            bands.retain(|b| b.1 - b.0 >= 0.6 * pitch && b.1 - b.0 <= 1.6 * pitch);
        }
        let rows_from_rules = bands.len() >= 2;
        let last_y = if rows_from_rules { bands.last().unwrap().1 } else { bot };
        let region = to_ref(&m, [x0, top, x1 - x0, last_y - top + 0.3 * TEXT_H]);
        structure += &format!("  - name: {}\n    rect: {}\n", tname, rect_yaml(region));
        table_spans.push((region[1], region[1] + region[3]));
        // the printed line just under the table ("Reasons for Refusal to Sign")
        let below: Option<&OLine> = l
            .lines
            .iter()
            .filter(|o| o.cy() > last_y && o.cy() < last_y + 4.0 * TEXT_H && o.bbox().0 < x1 && o.bbox().2 > x0)
            .min_by(|a, b| a.cy().partial_cmp(&b.cy()).unwrap());
        for f in t {
            let kind = f.kind.as_str();
            let zname = format!("{}_{}", if ti == 0 { "agent".to_string() } else { format!("agent{}", ti + 1) }, match kind {
                "id" => "id",
                "phone" => "phone",
                _ => "signatures",
            });
            // the column's sides from the reader, its rows from the table
            let (cx0, cx1) = (f.zones[0].rect[0], f.zones[0].rect[0] + f.zones[0].rect[2]);
            let cells: Vec<[f64; 4]> = if rows_from_rules {
                bands.iter().map(|b| [cx0, b.0, cx1 - cx0, b.1 - b.0]).collect()
            } else {
                f.zones.iter().map(|z| z.rect).collect()
            };
            let refs: Vec<[f64; 4]> = cells.iter().map(|c| to_ref(&m, *c)).collect();
            let (first, last) = (refs[0], *refs.last().unwrap());
            let n = refs.len();
            let step = if n > 1 { (last[1] - first[1]) / (n - 1) as f64 } else { first[3] };
            zones += &format!("  - name: {}\n    action: {}\n", zname, if kind == "signature" { "signature" } else { "partial" });
            if kind != "signature" {
                zones += &format!("    rule: {}\n    digits: {}\n", kind, if kind == "phone" { 10 } else { 8 });
            }
            zones += &format!("    rect: {}\n    rows: {}\n    row_step: {}\n    region: {}\n", rect_yaml(first), n, r1(step), tname);
            zones += &format!("    fence:\n      above: {}\n", rect_yaml(box_ref(&m, f.label_box)));
            if let Some(b) = below {
                zones += &format!("      below: {}\n", rect_yaml(line_box(b)));
                keep.push(line_box(b));
            }
            keep.push(box_ref(&m, f.label_box));
            // what masking searches: signatures overrun their column
            let (mx, below_px) = if kind == "signature" { (24.0, 30.0) } else { (8.0, 12.0) };
            erase.push([first[0] - mx, first[1] - 6.0, first[2] + 2.0 * mx, last[1] + last[3] - first[1] + 6.0 + below_px]);
            let rows_written: Vec<f64> = refs.iter().map(|r| written(&writing, w, h, *r, 25)).collect();
            fields_rec.push(json!({"zone": zname, "kind": kind, "label": f.label, "rows": n, "rows_from": if rows_from_rules { "table rules" } else { "column" }, "written": rows_written.iter().map(|v| (v * 100.0).round() / 100.0).collect::<Vec<_>>()}));
            lowest = lowest.max(last[1] + last[3]);
        }
    }
    // officers' signature lines in the declaration; IDs and phones by a label
    let mut decl: Vec<&Field> = l.fields.iter().filter(|f| f.source == "declaration" || f.source == "label").collect();
    decl.sort_by(|a, b| a.zones[0].rect[1].partial_cmp(&b.zones[0].rect[1]).unwrap());
    for (k, f) in decl.iter().enumerate() {
        let z = &f.zones[0];
        let r = to_ref(&m, z.rect);
        let zname = match (f.source.as_str(), f.kind.as_str()) {
            ("declaration", _) => format!("officer_signature_{}", k + 1),
            (_, kind) => format!("{}_{}", kind, k + 1),
        };
        zones += &format!("  - name: {}\n    action: {}\n", zname, if f.kind == "signature" { "signature" } else { "partial" });
        if f.kind != "signature" {
            zones += &format!("    rule: {}\n    digits: {}\n", f.kind, z.digits);
        }
        zones += &format!("    rect: {}\n", rect_yaml(r));
        if f.kind == "signature" {
            zones += "    search: [12, 24]\n";
            // "Date" at the end of the line: never taken for writing
            let cy = z.rect[1] + z.rect[3] / 2.0;
            let dates: Vec<[f64; 4]> = l
                .lines
                .iter()
                .filter(|o| (o.cy() - cy).abs() < 0.6 * TEXT_H)
                .flat_map(|o| {
                    let mut v = Vec::new();
                    let t = o.text.to_lowercase();
                    if let Some(p) = t.find("date") {
                        let i = t[..p].chars().count();
                        let (a, b) = o.span(i, i + 4);
                        let bb = o.bbox();
                        if a > z.rect[0] - TEXT_H {
                            v.push(box_ref(&m, (a, bb.1, b, bb.3)));
                        }
                    }
                    v
                })
                .collect();
            let mut dates_u: Vec<[f64; 4]> = Vec::new();
            for d in dates {
                if !dates_u.iter().any(|e| (e[0] - d[0]).abs() < 0.5 * text_h && (e[1] - d[1]).abs() < 0.5 * text_h) {
                    dates_u.push(d);
                }
            }
            let dates = dates_u;
            keep.extend(dates.iter().cloned());
            if !dates.is_empty() {
                zones += "    print:\n";
                for d in dates {
                    zones += &format!("      - {}\n", rect_yaml(d));
                }
            }
        }
        erase.push([r[0] - 8.0, r[1] - 12.0, r[2] + 16.0, r[3] + 36.0]);
        let share = written(&writing, w, h, r, 40);
        fields_rec.push(json!({"zone": zname, "kind": f.kind, "label": f.label, "written": (share * 100.0).round() / 100.0}));
        if f.kind == "signature" && share < 0.3 {
            notes.push(format!("{}: written in on only {:.0}% of the pages", zname, 100.0 * share));
        }
        lowest = lowest.max(r[1] + r[3]);
    }
    if fields_rec.is_empty() {
        bail!("{}: no fields found on the blank form", name);
    }
    if !structure.is_empty() {
        yaml += &format!("structure:\n{}", structure);
    }
    // below the last table or officer line: comments and reasons, read as free text
    let ft_top = lowest + 0.3 * text_h;
    if (h as f64) - ft_top > 2.0 * text_h {
        // down past the reference's bottom: photos show more or less of the
        // page below the comments, and officers sign at the very bottom
        let ft = [0.02 * w as f64, ft_top, 0.96 * w as f64, h as f64 - ft_top + 0.1 * h as f64];
        yaml += &format!("free_text:\n  - name: comments\n    rect: {}\n", rect_yaml(ft));
        // its printed headings stay ("Reasons for Refusal to Sign", "Comments")
        erase.push(ft);
        for o in &l.lines {
            let b = line_box(o);
            if b[1] + b[3] / 2.0 > ft_top && HEADING.is_match(&o.text) {
                keep.push(b);
            }
        }
    }
    yaml += &format!("zones:\n{}", zones);

    // writing no field explains: frequent writing outside the results, the
    // tables, the declaration lines and the free text
    let mut unexplained = Vec::new();
    {
        let mut hot = vec![0u8; freq.len()];
        for (i, f) in freq.iter().enumerate() {
            if *f >= 0.25 {
                hot[i] = 255;
            }
        }
        let hm = util::morph(&util::from_bytes(h, w, &hot)?, imgproc::MORPH_CLOSE, &util::ones_kernel(9, 9)?)?;
        let (_, st, _) = util::components(&hm)?;
        for s in st.iter().skip(1) {
            if (s[4] as f64) < 4.0 * text_h * text_h {
                continue;
            }
            let cy = s[1] as f64 + s[3] as f64 / 2.0;
            if cy < results_bottom || cy > lowest {
                continue; // results, or free text
            }
            let in_table = table_spans.iter().any(|&(a, b)| cy > a && cy < b);
            let on_decl = decl.iter().any(|f| {
                let r = to_ref(&m, f.zones[0].rect);
                (cy - (r[1] + r[3] / 2.0)).abs() < 1.2 * text_h
            });
            if !in_table && !on_decl {
                unexplained.push(json!([s[0], s[1], s[2], s[3]]));
            }
        }
    }
    if !unexplained.is_empty() {
        notes.push(format!("{} area(s) people write in that no field explains (see writing.png)", unexplained.len()));
    }

    // The table's rules, light grey on the blank form, drawn where they are
    // printed: pages are matched to the table by its rules
    let mut reference = reference;
    if let Some(t) = &profiled {
        let mut lines = Mat::new_rows_cols_with_default(h, w, core::CV_8U, Scalar::all(0.0))?;
        for r in &t.rules {
            let pts: Vector<core::Point> = r.iter().map(|&p| { let q = util::project(&m, p); core::Point::new(q[0].round() as i32, q[1].round() as i32) }).collect();
            imgproc::polylines(&mut lines, &Vector::<Vector<core::Point>>::from_iter([pts]), false, Scalar::all(255.0), 2, imgproc::LINE_8, 0)?;
        }
        let lb = util::bytes(&lines)?;
        let mut px = util::bytes(&reference)?;
        for i in 0..px.len() {
            if lb[i] != 0 {
                px[i] = px[i].min(70);
            }
        }
        reference = util::from_bytes(h, w, &px)?;
    }
    let reference = clean(&reference, &erase, &keep)?;
    // its edges sharpened back: averaging pages a pixel apart softens them,
    // and pages are matched to it by its corners
    let reference = {
        let (mut soft, mut sharp) = (Mat::default(), Mat::default());
        imgproc::gaussian_blur(&reference, &mut soft, Size::new(0, 0), 2.0, 0.0, core::BORDER_REPLICATE, core::AlgorithmHint::ALGO_HINT_DEFAULT)?;
        core::add_weighted(&reference, 3.0, &soft, -2.0, 0.0, &mut sharp, -1)?;
        sharp
    };
    imgcodecs::imwrite(&dir.join("reference.png"), &reference, &Vector::new())?;
    std::fs::write(dir.join("template.yaml"), &yaml)?;
    let record = json!({
        "printing": name,
        "form": form,
        "template": slug(name),
        "pages_used": used,
        "seed": seed.src,
        "pages_rejected": rejected,
        "read_from": read_from,
        "readings": tried,
        "fields": fields_rec,
        "unexplained_writing": unexplained,
        "notes": notes,
    });
    std::fs::write(dir.join("learned.json"), serde_json::to_string_pretty(&record)?)?;
    Ok(Learned { name: slug(name), dir, record })
}

/// A directory name for a printing.
pub fn slug(name: &str) -> String {
    let s: String = name.to_lowercase().chars().map(|c| if c.is_ascii_alphanumeric() { c } else { '_' }).collect();
    format!("learned_{}", s.split('_').filter(|p| !p.is_empty()).collect::<Vec<_>>().join("_"))
}

/// Group `sources` by printing, learn each group with enough pages
/// (polling-station forms only). With `skip_learned`, a printing that
/// already has a template under `out` is left as it is (forms arriving in
/// batches).
pub fn learn_all(sources: &[String], form: Option<&str>, out: &Path, opts: &LearnOptions, skip_learned: bool) -> Result<Vec<Result<Learned>>> {
    let mut groups: Vec<(String, String, Vec<String>)> = Vec::new();
    for s in sources {
        let (f, key) = printing(s, form);
        if f.ends_with('B') || f.ends_with('C') {
            continue; // tally forms: GENERALIZING.md, milestone 2
        }
        match groups.iter_mut().find(|g| g.1 == key) {
            Some(g) => g.2.push(s.clone()),
            None => groups.push((f, key, vec![s.clone()])),
        }
    }
    if groups.is_empty() {
        return Err(anyhow!("no polling-station forms to learn from"));
    }
    let mut out_v = Vec::new();
    for (f, key, files) in groups {
        if skip_learned && out.join(slug(&key)).join("template.yaml").exists() {
            continue;
        }
        if files.len() < opts.min_pages {
            out_v.push(Err(anyhow!("{}: {} file(s), too few to learn from (need {})", key, files.len(), opts.min_pages)));
            continue;
        }
        out_v.push(learn(&files, &f, &key, out, opts));
    }
    Ok(out_v)
}

#[cfg(test)]
#[path = "tests/learn.rs"]
mod tests;
