//! Register a scanned page to a template with ORB features + MAGSAC homography.
//!
//! Features are taken on a downscaled copy (the template's `feature_dim`,
//! else `max_dim`); keypoints are scaled back so the homography maps
//! full-resolution template pixels to full-resolution scan pixels.

use anyhow::Result;
use opencv::geometry;
use opencv::core::{self, DMatch, KeyPoint, Mat, Point2f, Size, Vector};
use opencv::features::{BFMatcher, ORB_ScoreType, ORB};
use opencv::imgproc;
use opencv::prelude::*;
use std::collections::HashMap;

use crate::structure;
use crate::templates::Template;
use crate::util;

#[derive(Clone, Debug)]
pub struct AlignSettings {
    pub max_dim: i64,
    pub n_features: i64,
    pub ratio: f64,
    pub ransac_thresh: f64,
    pub min_inliers: i64,
    pub min_inlier_ratio: f64,
    pub min_area_frac: f64,
    pub max_side_ratio: f64,
    pub min_score: f64,
}

impl Default for AlignSettings {
    fn default() -> Self {
        AlignSettings {
            max_dim: 1200,
            n_features: 5000,
            ratio: 0.75,
            ransac_thresh: 6.0,
            min_inliers: 40,
            min_inlier_ratio: 0.15,
            min_area_frac: 0.15,
            max_side_ratio: 2.0,
            min_score: 0.35,
        }
    }
}

impl AlignSettings {
    /// As a manifest records them: all of them, since any can change which
    /// pages pass and where their zones land, and `verify --rerun` must
    /// repeat the redaction exactly.
    pub fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "max_dim": self.max_dim,
            "n_features": self.n_features,
            "ratio": self.ratio,
            "ransac_thresh": self.ransac_thresh,
            "min_inliers": self.min_inliers,
            "min_inlier_ratio": self.min_inlier_ratio,
            "min_area_frac": self.min_area_frac,
            "max_side_ratio": self.max_side_ratio,
            "min_score": self.min_score,
        })
    }

    /// From a manifest's record. What it doesn't record is the default:
    /// manifests written before alignment settings were recorded were
    /// redacted with the defaults.
    pub fn from_json(v: &serde_json::Value) -> Self {
        let d = AlignSettings::default();
        let int = |k: &str, x: i64| v.get(k).and_then(|y| y.as_i64()).unwrap_or(x);
        let num = |k: &str, x: f64| v.get(k).and_then(|y| y.as_f64()).unwrap_or(x);
        AlignSettings {
            max_dim: int("max_dim", d.max_dim),
            n_features: int("n_features", d.n_features),
            ratio: num("ratio", d.ratio),
            ransac_thresh: num("ransac_thresh", d.ransac_thresh),
            min_inliers: int("min_inliers", d.min_inliers),
            min_inlier_ratio: num("min_inlier_ratio", d.min_inlier_ratio),
            min_area_frac: num("min_area_frac", d.min_area_frac),
            max_side_ratio: num("max_side_ratio", d.max_side_ratio),
            min_score: num("min_score", d.min_score),
        }
    }
}

#[derive(Clone)]
pub struct Alignment {
    pub template: Option<usize>, // index into the template list
    pub h: Option<[[f64; 3]; 3]>, // template -> scan
    pub inliers: i64,
    pub matches: i64,
    pub score: f64,
    pub ok: bool,
    pub reason: String,
    pub src: Vec<[f32; 2]>, // inlier points, template space
    pub dst: Vec<[f32; 2]>, // inlier points, scan space
}

impl Alignment {
    fn fail(t: Option<usize>, reason: String, inl: i64, m: i64) -> Self {
        Alignment { template: t, h: None, inliers: inl, matches: m, score: 0.0, ok: false, reason, src: vec![], dst: vec![] }
    }

    /// Homography fitted only to inliers near `center` (template coords):
    /// curled pages make one global fit wrong by 10-20 px in places.
    pub fn local(&self, center: [f64; 2]) -> Result<[[f64; 3]; 3]> {
        let (radius, min_points, max_shift) = (350.0, 15, 40.0);
        let h = self.h.unwrap();
        if self.src.is_empty() {
            return Ok(h);
        }
        let near: Vec<usize> = (0..self.src.len())
            .filter(|&i| {
                let dx = self.src[i][0] as f64 - center[0] as f32 as f64;
                let dy = self.src[i][1] as f64 - center[1] as f32 as f64;
                ((dx * dx + dy * dy) as f32).sqrt() < radius as f32
            })
            .collect();
        if near.len() < min_points {
            return Ok(h);
        }
        let s: Vector<Point2f> = near.iter().map(|&i| Point2f::new(self.src[i][0], self.src[i][1])).collect();
        let d: Vector<Point2f> = near.iter().map(|&i| Point2f::new(self.dst[i][0], self.dst[i][1])).collect();
        let hl = geometry::find_homography(&s, &d, geometry::RANSAC, 4.0, &mut Mat::default(), 2000, 0.995)?;
        if hl.empty() {
            return Ok(h);
        }
        let hl = util::arr3(&hl)?;
        // numpy: float32 projections, a float32 norm
        let c = [center[0] as f32 as f64, center[1] as f32 as f64];
        let (a, b) = (util::project(&hl, c), util::project(&h, c));
        let (dx, dy) = (a[0] as f32 - b[0] as f32, a[1] as f32 - b[1] as f32);
        if ((dx * dx + dy * dy).sqrt() as f64) > max_shift {
            return Ok(h);
        }
        Ok(hl)
    }
}

pub type Features = (Vec<[f32; 2]>, Mat);

/// ORB features on a copy scaled so the *page* is `max_dim` long. For a
/// template cut from a page (`page_long_side`), that is the page it came from.
pub fn features(gray: &Mat, s: &AlignSettings, page_long_side: Option<f64>, n_features: Option<i64>, max_dim: Option<i64>) -> Result<Features> {
    let long = page_long_side.unwrap_or(gray.rows().max(gray.cols()) as f64);
    let scale = (max_dim.unwrap_or(s.max_dim) as f64 / long).min(1.0);
    let small = if scale < 1.0 { util::resize_by(gray, scale, imgproc::INTER_AREA)? } else { gray.try_clone()? };
    let mut orb = ORB::create(n_features.unwrap_or(s.n_features) as i32, 1.2, 8, 31, 0, 2, ORB_ScoreType::HARRIS_SCORE, 31, 10)?;
    let mut kps = Vector::<KeyPoint>::new();
    let mut desc = Mat::default();
    orb.detect_and_compute(&small, &core::no_array(), &mut kps, &mut desc, false)?;
    // numpy: float32 points divided by the (float32) scale
    let s32 = scale as f32;
    let pts = kps.iter().map(|k| [k.pt().x / s32, k.pt().y / s32]).collect();
    Ok((pts, desc))
}

pub fn prepare(t: &mut Template, s: &AlignSettings) -> Result<()> {
    let (k, d) = features(&t.image, s, t.page_long_side, t.n_features, t.feature_dim)?;
    t.keypoints = k;
    t.descriptors = d;
    t.mask = t.build_mask()?;
    t.regions.clear();
    for (n, r) in t.structure.clone() {
        let open = t.open_top.contains(&n);
        let region = structure::prepare_region(&t.image, &n, r, open)?;
        t.regions.push((n, region));
    }
    Ok(())
}

fn geometry_ok(h: &[[f64; 3]; 3], t: &Template, scan: (i32, i32), s: &AlignSettings) -> Result<String> {
    let (w, hh) = t.size();
    let corners = [[0.0, 0.0], [w as f64, 0.0], [w as f64, hh as f64], [0.0, hh as f64]];
    // cv2.perspectiveTransform on float32 corners
    let q: Vec<[f32; 2]> = corners.iter().map(|c| {
        let p = util::project(h, *c);
        [p[0] as f32, p[1] as f32]
    }).collect();
    let poly: Vector<Point2f> = q.iter().map(|p| Point2f::new(p[0], p[1])).collect();
    if !geometry::is_contour_convex(&poly)? {
        return Ok("projected form is not convex".into());
    }
    let area = geometry::contour_area(&poly, false)?;
    let min_frac = t.min_area_frac.unwrap_or(s.min_area_frac);
    let total = scan.0 as f64 * scan.1 as f64;
    if area < min_frac * total {
        return Ok(format!("projected form too small ({:.2} of scan)", area / total));
    }
    let side = |a: usize, b: usize| {
        let (dx, dy) = (q[a][0] - q[b][0], q[a][1] - q[b][1]);
        ((dx * dx + dy * dy) as f32).sqrt() as f64
    };
    let sides = [side(0, 1), side(1, 2), side(2, 3), side(3, 0)];
    for (a, b) in [(sides[0], sides[2]), (sides[1], sides[3])] {
        if a.max(b) / a.min(b).max(1e-6) > s.max_side_ratio {
            return Ok("implausible perspective".into());
        }
    }
    Ok(String::new())
}

/// NCC of the template vs the scan warped into template space, ignoring PII
/// zones (they hold handwriting the blank template lacks).
fn score(h: &[[f64; 3]; 3], t: &Template, scan: &Mat) -> Result<f64> {
    let size = 500.0;
    let (w, hh) = t.size();
    let f = size / (w.max(hh) as f64);
    let (sw, sh) = ((w as f64 * f).round_ties_even().max(1.0) as i32, (hh as f64 * f).round_ties_even().max(1.0) as i32);
    let sm = [[f, 0.0, 0.0], [0.0, f, 0.0], [0.0, 0.0, 1.0]];
    let m = util::mul3(&sm, &util::inv3(h)?);
    let mut warped = Mat::default();
    // A learned reference's frame is the first page's: other photos may not
    // reach its edges. Beyond the photo is no evidence either way, so it is
    // left out (rather than black, which no reference matches).
    let border = if t.learned { core::BORDER_REPLICATE } else { core::BORDER_CONSTANT };
    imgproc::warp_perspective(scan, &mut warped, &util::mat3(&m)?, Size::new(sw, sh), imgproc::INTER_AREA, border, core::Scalar::default(), core::AlgorithmHint::ALGO_HINT_DEFAULT)?;
    // a learned reference has its lighting flattened: so has the page, or
    // its shadows outweigh its print
    let warped = if t.learned { util::flatten_light(&warped)? } else { warped };
    let reference = util::resize(&t.image, sw, sh, imgproc::INTER_AREA)?;
    let mut keep = util::resize(&t.mask, sw, sh, imgproc::INTER_NEAREST)?;
    if t.learned {
        let mut inside = Mat::default();
        let ones = util::full_u8(scan.rows(), scan.cols(), 255)?;
        imgproc::warp_perspective(&ones, &mut inside, &util::mat3(&m)?, Size::new(sw, sh), imgproc::INTER_NEAREST, core::BORDER_CONSTANT, core::Scalar::default(), core::AlgorithmHint::ALGO_HINT_DEFAULT)?;
        let outside = util::bitnot(&util::morph(&inside, imgproc::MORPH_ERODE, &util::ones_kernel(5, 5)?)?)?;
        keep = util::bitor(&keep, &outside)?;
    }
    let (a, b, k) = (util::bytes(&util::gaussian(&reference, 5)?)?, util::bytes(&util::gaussian(&warped, 5)?)?, util::bytes(&keep)?);
    let mut av: Vec<f32> = a.iter().zip(&k).filter(|p| *p.1 == 0).map(|p| *p.0 as f32).collect();
    let mut bv: Vec<f32> = b.iter().zip(&k).filter(|p| *p.1 == 0).map(|p| *p.0 as f32).collect();
    if av.is_empty() {
        return Ok(0.0);
    }
    // as numpy does it: float32 throughout, pairwise sums
    let ma = util::np_sum_f32(&av) / av.len() as f32;
    let mb = util::np_sum_f32(&bv) / bv.len() as f32;
    av.iter_mut().for_each(|x| *x -= ma);
    bv.iter_mut().for_each(|x| *x -= mb);
    let saa = util::np_sum_f32(&av.iter().map(|x| x * x).collect::<Vec<_>>());
    let sbb = util::np_sum_f32(&bv.iter().map(|x| x * x).collect::<Vec<_>>());
    let sab = util::np_sum_f32(&av.iter().zip(&bv).map(|(x, y)| x * y).collect::<Vec<_>>());
    let denom = (saa * sbb).sqrt() as f64;
    Ok(if denom > 0.0 { (sab / denom as f32) as f64 } else { 0.0 })
}

pub fn align(scan: &Mat, ti: usize, t: &Template, s: &AlignSettings, feats: &Features) -> Result<Alignment> {
    let (pts, desc) = feats;
    if desc.rows() < 2 || t.descriptors.empty() {
        return Ok(Alignment::fail(Some(ti), "no features".into(), 0, 0));
    }
    let matcher = BFMatcher::create(core::NORM_HAMMING, false)?;
    let mut knn = Vector::<Vector<DMatch>>::new();
    matcher.knn_train_match(&t.descriptors, desc, &mut knn, 2, &core::no_array(), false)?;
    let good: Vec<DMatch> = knn
        .iter()
        .filter_map(|m| if m.len() >= 2 && (m.get(0).ok()?.distance as f64) < s.ratio * m.get(1).ok()?.distance as f64 { m.get(0).ok() } else { None })
        .collect();
    let min_inliers = t.min_inliers.unwrap_or(s.min_inliers);
    let min_score = t.min_score.unwrap_or(s.min_score);
    if (good.len() as i64) < min_inliers {
        return Ok(Alignment::fail(Some(ti), format!("too few matches ({})", good.len()), 0, good.len() as i64));
    }
    let src: Vec<[f32; 2]> = good.iter().map(|m| t.keypoints[m.query_idx as usize]).collect();
    let dst: Vec<[f32; 2]> = good.iter().map(|m| pts[m.train_idx as usize]).collect();
    let sv: Vector<Point2f> = src.iter().map(|p| Point2f::new(p[0], p[1])).collect();
    let dv: Vector<Point2f> = dst.iter().map(|p| Point2f::new(p[0], p[1])).collect();
    let mut inl = Mat::default();
    let hm = geometry::find_homography(&sv, &dv, geometry::USAC_MAGSAC, s.ransac_thresh, &mut inl, 5000, 0.999)?;
    if hm.empty() {
        return Ok(Alignment::fail(Some(ti), "homography failed".into(), 0, good.len() as i64));
    }
    let mask = util::bytes(&inl)?;
    let n_in = mask.iter().filter(|&&v| v != 0).count() as i64;
    if n_in < min_inliers || (n_in as f64) / (good.len() as f64) < s.min_inlier_ratio {
        return Ok(Alignment::fail(Some(ti), format!("weak match ({}/{} inliers)", n_in, good.len()), n_in, good.len() as i64));
    }
    let h = util::arr3(&hm)?;
    let why = geometry_ok(&h, t, (scan.rows(), scan.cols()), s)?;
    if !why.is_empty() {
        return Ok(Alignment::fail(Some(ti), why, n_in, good.len() as i64));
    }
    let sc = score(&h, t, scan)?;
    let ok = sc >= min_score;
    let keep: Vec<usize> = (0..mask.len()).filter(|&i| mask[i] != 0).collect();
    Ok(Alignment {
        template: Some(ti),
        h: Some(h),
        inliers: n_in,
        matches: good.len() as i64,
        score: sc,
        ok,
        reason: if ok { String::new() } else { format!("low alignment score ({:.2})", sc) },
        src: keep.iter().map(|&i| src[i]).collect(),
        dst: keep.iter().map(|&i| dst[i]).collect(),
    })
}

/// Try every template (only those for `form`, if known); the passing
/// alignment with the highest (priority, inliers, score), or the best failure.
pub fn best_alignment(scan: &Mat, templates: &[Template], s: &AlignSettings, form: Option<&str>) -> Result<Alignment> {
    let idx: Vec<usize> = (0..templates.len())
        .filter(|&i| form.is_none() || templates[i].form.is_empty() || Some(templates[i].form.as_str()) == form)
        .collect();
    if idx.is_empty() {
        return Ok(Alignment::fail(None, format!("no template for form {}", form.unwrap_or("")), 0, 0));
    }
    let mut feats: HashMap<(Option<i64>, Option<i64>), Features> = HashMap::new();
    let mut results = Vec::new();
    for &i in &idx {
        let t = &templates[i];
        let key = (t.feature_dim, t.n_features);
        if !feats.contains_key(&key) {
            feats.insert(key, features(scan, s, None, t.n_features, t.feature_dim)?);
        }
        results.push(align(scan, i, t, s, &feats[&key])?);
    }
    let any_ok = results.iter().any(|r| r.ok);
    let pool: Vec<&Alignment> = results.iter().filter(|r| r.ok || !any_ok).collect();
    let rank = |r: &Alignment| (if r.ok { templates[r.template.unwrap()].priority } else { 0 }, r.inliers, r.score);
    let mut best = pool[0];
    for r in &pool[1..] {
        let (a, b) = (rank(r), rank(best));
        if a.0 > b.0 || (a.0 == b.0 && (a.1 > b.1 || (a.1 == b.1 && a.2 > b.2))) {
            best = r;
        }
    }
    Ok(best.clone())
}
