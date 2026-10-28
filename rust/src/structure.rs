//! Snap zones to the printed table structure.
//!
//! Feature matching places the page as a whole, but KIEMS photos are of bent,
//! curled booklet pages: one homography can be off by several table rows at
//! the bottom of the page, where the PII tables are. A template names
//! `structure` regions; their printed rules are found in the reference once
//! and in each scan, matched in x and y separately by a 1-D search for scale
//! and offset, and the zones are mapped through the grid of rule
//! intersections. A region that can't be matched convincingly fails.

use anyhow::Result;
use opencv::calib3d;
use opencv::core::{self, Mat, Point2f, Size, Vector};
use opencv::imgproc;
use opencv::prelude::*;

use crate::util::{self, median, np_mean, np_sum};

#[derive(Clone, Debug)]
pub struct Line {
    pub pos: f64,   // y for horizontal lines (at x = 0), x for vertical (at y = 0)
    pub slope: f64, // dy/dx (horizontal) or dx/dy (vertical)
    pub lo: f64,    // extent along the line
    pub hi: f64,
}

impl Line {
    pub fn at(&self, t: f64) -> f64 {
        self.pos + self.slope * t
    }
    pub fn length(&self) -> f64 {
        self.hi - self.lo
    }
}

#[derive(Clone, Debug, Default)]
pub struct Region {
    pub name: String,
    pub rect: [f64; 4],
    pub hlines: Vec<f64>,
    pub vlines: Vec<f64>,
    pub hslope: f64,
    pub vslope: f64,
    pub top: f64,
    pub bottom: f64,
    /// The table hangs under another with the same columns, so its column
    /// rules run on up into that one: only their lower end marks this table.
    pub open_top: bool,
}

pub fn segments(mask: &Mat, horizontal: bool) -> Result<Vec<Line>> {
    let (lab, stats, _) = util::components(mask)?;
    let labels = util::labels_vec(&lab)?;
    let cols = mask.cols() as usize;
    let mut out = Vec::new();
    for (i, st) in stats.iter().enumerate().skip(1) {
        let [x, y, w, h, area] = *st;
        if area < 20 {
            continue;
        }
        let (mut xs, mut ys) = (Vec::new(), Vec::new());
        for yy in y..y + h {
            for xx in x..x + w {
                if labels[yy as usize * cols + xx as usize] == i as i32 {
                    xs.push(xx as f64);
                    ys.push(yy as f64);
                }
            }
        }
        if horizontal {
            let (a, b) = if w > 1 { util::polyfit1(&xs, &ys) } else { (0.0, util::mean(&ys)) };
            out.push(Line { pos: b, slope: a, lo: x as f64, hi: (x + w) as f64 });
        } else {
            let (a, b) = if h > 1 { util::polyfit1(&ys, &xs) } else { (0.0, util::mean(&xs)) };
            out.push(Line { pos: b, slope: a, lo: y as f64, hi: (y + h) as f64 });
        }
    }
    Ok(out)
}

/// Join collinear pieces of a rule broken by handwriting or stamps.
fn merge(lines: Vec<Line>, tol: f64) -> Vec<Line> {
    let mut sorted = lines;
    sorted.sort_by(|a, b| a.at((a.lo + a.hi) / 2.0).partial_cmp(&b.at((b.lo + b.hi) / 2.0)).unwrap());
    let mut merged: Vec<Line> = Vec::new();
    for l in sorted {
        let mid = (l.lo + l.hi) / 2.0;
        let mut done = false;
        for m in merged.iter_mut() {
            if (m.at(mid) - l.at(mid)).abs() < tol && (m.slope - l.slope).abs() < 0.02 {
                let total = m.length() + l.length();
                m.pos = (m.pos * m.length() + l.pos * l.length()) / total;
                m.slope = (m.slope * m.length() + l.slope * l.length()) / total;
                m.lo = m.lo.min(l.lo);
                m.hi = m.hi.max(l.hi);
                done = true;
                break;
            }
        }
        if !done {
            merged.push(l);
        }
    }
    merged
}

/// Printed rules in a roughly axis-aligned image (template frame).
pub fn detect_lines(gray: &Mat, min_h: f64, min_v: f64) -> Result<(Vec<Line>, Vec<Line>)> {
    let c = util::gaussian(gray, 3)?;
    let bw = util::adaptive_inv(&c, 31, 8.0)?;
    let hor = util::morph(&bw, imgproc::MORPH_OPEN, &util::rect_kernel(60, 1)?)?;
    let ver = util::morph(&bw, imgproc::MORPH_OPEN, &util::rect_kernel(1, 30)?)?;
    let hs = merge(segments(&hor, true)?, 4.0).into_iter().filter(|l| l.length() >= min_h && l.slope.abs() < 0.1).collect();
    let vs = merge(segments(&ver, false)?, 4.0).into_iter().filter(|l| l.length() >= min_v && l.slope.abs() < 0.1).collect();
    Ok((hs, vs))
}

pub fn prepare_region(reference: &Mat, name: &str, rect: [f64; 4], open_top: bool) -> Result<Region> {
    let (x, y, w, h) = (rect[0] as i32, rect[1] as i32, rect[2] as i32, rect[3] as i32);
    let roi = Mat::roi(reference, core::Rect::new(x, y, w, h))?.try_clone()?;
    let (hs, vs) = detect_lines(&roi, 0.5 * w as f64, 0.5 * h as f64)?;
    let mut hl: Vec<f64> = hs.iter().map(|l| l.at(w as f64 / 2.0) + y as f64).collect();
    let mut vl: Vec<f64> = vs.iter().map(|l| l.at(h as f64 / 2.0) + x as f64).collect();
    hl.sort_by(|a, b| a.partial_cmp(b).unwrap());
    vl.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let med = |v: Vec<f64>| if v.is_empty() { 0.0 } else { median(&v) };
    Ok(Region {
        name: name.into(),
        rect: [x as f64, y as f64, w as f64, h as f64],
        hlines: hl,
        vlines: vl,
        hslope: med(hs.iter().map(|l| l.slope).collect()),
        vslope: med(vs.iter().map(|l| l.slope).collect()),
        top: if vs.is_empty() { 0.0 } else { median(&vs.iter().map(|l| l.lo).collect::<Vec<_>>()) + y as f64 },
        bottom: if vs.is_empty() { 0.0 } else { median(&vs.iter().map(|l| l.hi).collect::<Vec<_>>()) + y as f64 },
        open_top,
    })
}

/// Best y' = s*y + t mapping expected positions onto found ones (see the
/// Python version for the full reasoning). Returns (s, t, matched pairs).
pub fn match_1d(
    expected: &[f64],
    found: &[f64],
    tol: f64,
    scale: (f64, f64),
    weights: Option<&[f64]>,
    max_shift: Option<f64>,
    ends: Option<(f64, f64, f64, f64)>,
) -> (f64, f64, Vec<(usize, usize)>) {
    let (e, f) = (expected, found);
    if e.len() < 2 || f.len() < 2 {
        return (1.0, 0.0, vec![]);
    }
    let ones = vec![1.0; f.len()];
    let w = weights.unwrap_or(&ones);
    let mut best = (-1.0f64, f64::INFINITY, 1.0f64, 0.0f64);
    for i in 0..e.len() {
        for j in i + 1..e.len() {
            let de = e[j] - e[i];
            for k in 0..f.len() {
                for l in k + 1..f.len() {
                    let sc = (f[l] - f[k]) / de;
                    if !(sc >= scale.0 && sc <= scale.1) {
                        continue;
                    }
                    let t = f[k] - sc * e[i];
                    if let Some(ms) = max_shift {
                        if (sc * e[0] + t - e[0]).abs() > ms {
                            continue;
                        }
                    }
                    if let Some((a, at, b, bt)) = ends {
                        if (sc * e[0] + t - a).abs() > at || (sc * e[e.len() - 1] + t - b).abs() > bt {
                            continue;
                        }
                    }
                    let mut scores = Vec::with_capacity(e.len());
                    let mut hits = Vec::new();
                    for &x in e {
                        let p = sc * x + t;
                        let mut jn = 0;
                        let mut dn = f64::INFINITY;
                        for (q, &y) in f.iter().enumerate() {
                            let d = (p - y).abs();
                            if d < dn {
                                dn = d;
                                jn = q;
                            }
                        }
                        let hit = dn < tol;
                        scores.push(if hit { w[jn] } else { 0.0 });
                        if hit {
                            hits.push(dn);
                        }
                    }
                    let score = np_sum(&scores);
                    let err = if hits.is_empty() { f64::INFINITY } else { np_mean(&hits) };
                    if score > best.0 || (score == best.0 && err < best.1) {
                        best = (score, err, sc, t);
                    }
                }
            }
        }
    }
    let (_, _, sc, t) = best;
    let pred: Vec<f64> = e.iter().map(|x| sc * x + t).collect();
    let mut matched: std::collections::BTreeMap<usize, usize> = std::collections::BTreeMap::new();
    let take = |m: &mut std::collections::BTreeMap<usize, usize>, i: usize, p: f64| {
        let mut cand: Vec<f64> = f.iter().map(|y| (y - p).abs()).collect();
        for &j in m.values() {
            cand[j] = f64::INFINITY;
        }
        let j = util::argmin(&cand);
        if cand[j] < tol {
            m.insert(i, j);
        }
    };
    let mins: Vec<f64> = pred.iter().map(|p| f.iter().map(|y| (p - y).abs()).fold(f64::INFINITY, f64::min)).collect();
    let anchor = util::argmin(&mins);
    take(&mut matched, anchor, pred[anchor]);
    for step in [1i64, -1] {
        let mut last = if matched.contains_key(&anchor) { Some(anchor) } else { None };
        let mut i = anchor as i64 + step;
        while i >= 0 && (i as usize) < e.len() {
            let iu = i as usize;
            let p = match last {
                Some(lst) => f[matched[&lst]] + sc * (e[iu] - e[lst]),
                None => pred[iu],
            };
            take(&mut matched, iu, p);
            if matched.contains_key(&iu) {
                last = Some(iu);
            }
            i += step;
        }
    }
    (sc, t, matched.into_iter().collect())
}

#[derive(Clone, Debug, Default)]
pub struct Snap {
    pub ok: bool,
    pub reason: String,
    pub matched_h: usize,
    pub matched_v: usize,
    pub rms: f64,
    pub xs: Vec<f64>,
    pub ys: Vec<f64>,
    pub p: Vec<Vec<[f64; 2]>>, // scan coords of every intersection [row][col]
}

impl Snap {
    fn fail(reason: String, h: usize, v: usize) -> Snap {
        Snap { ok: false, reason, matched_h: h, matched_v: v, ..Default::default() }
    }

    /// Template -> scan by bilinear interpolation over the rule grid;
    /// extrapolated along the whole grid's average spacing outside it.
    pub fn map(&self, pts: &[[f64; 2]]) -> Vec<[f64; 2]> {
        let (xs, ys, p) = (&self.xs, &self.ys, &self.p);
        let (nx, ny) = (xs.len(), ys.len());
        let p00 = p[0][0];
        let p0n = p[0][nx - 1];
        let pn0 = p[ny - 1][0];
        let pnn = p[ny - 1][nx - 1];
        pts.iter()
            .map(|&[px, py]| {
                let x = px.clamp(xs[0], xs[nx - 1]);
                let y = py.clamp(ys[0], ys[ny - 1]);
                let j = (util::searchsorted(xs, x) as i64 - 1).clamp(0, nx as i64 - 2) as usize;
                let i = (util::searchsorted(ys, y) as i64 - 1).clamp(0, ny as i64 - 2) as usize;
                let u = (x - xs[j]) / (xs[j + 1] - xs[j]);
                let v = (y - ys[i]) / (ys[i + 1] - ys[i]);
                let mut out = [0.0; 2];
                for k in 0..2 {
                    out[k] = (1.0 - u) * (1.0 - v) * p[i][j][k]
                        + u * (1.0 - v) * p[i][j + 1][k]
                        + (1.0 - u) * v * p[i + 1][j][k]
                        + u * v * p[i + 1][j + 1][k];
                }
                let uu = (x - xs[0]) / (xs[nx - 1] - xs[0]);
                let vv = (y - ys[0]) / (ys[ny - 1] - ys[0]);
                for k in 0..2 {
                    let down = ((1.0 - uu) * (pn0[k] - p00[k]) + uu * (pnn[k] - p0n[k])) / (ys[ny - 1] - ys[0]);
                    let across = ((1.0 - vv) * (p0n[k] - p00[k]) + vv * (pnn[k] - pn0[k])) / (xs[nx - 1] - xs[0]);
                    out[k] += (py - y) * down + (px - x) * across;
                }
                out
            })
            .collect()
    }
}

/// Refine H inside `region` by matching printed rules.
pub fn snap(gray: &Mat, h: &[[f64; 3]; 3], region: &Region) -> Result<Snap> {
    let (margin, min_frac_h, max_rms) = (0.5, 0.6, 15.0);
    let [x, y, w, hh] = region.rect;
    let (mx, my) = (0.15 * w, (margin * hh).max(150.0));
    let (sx, sy, sw, sh) = (x - mx, y - my, w + 2.0 * mx, hh + 2.0 * my);
    let o = [[1.0, 0.0, sx], [0.0, 1.0, sy], [0.0, 0.0, 1.0]];
    let m = util::inv3(&util::mul3(h, &o))?;
    let mut patch = Mat::default();
    imgproc::warp_perspective(
        gray,
        &mut patch,
        &util::mat3(&m)?,
        Size::new(sw as i32, sh as i32),
        imgproc::INTER_LINEAR,
        core::BORDER_REPLICATE,
        core::Scalar::default(),
    )?;
    let (hs, vs) = detect_lines(&patch, 0.35 * w, 0.35 * hh)?;
    let (cx, cy) = (x + w / 2.0 - sx, y + hh / 2.0 - sy);
    let fh: Vec<f64> = hs.iter().map(|l| l.at((l.lo + l.hi) / 2.0) - region.hslope * ((l.lo + l.hi) / 2.0 - cx) + sy).collect();
    let fv: Vec<f64> = vs.iter().map(|l| l.at((l.lo + l.hi) / 2.0) - region.vslope * ((l.lo + l.hi) / 2.0 - cy) + sx).collect();
    let pitch = if region.hlines.len() > 2 {
        median(&region.hlines.windows(2).map(|p| p[1] - p[0]).collect::<Vec<_>>())
    } else {
        30.0
    };
    let y_at = |l: &Line, xx: f64| {
        let e = xx.max(l.lo).min(l.hi);
        l.at(e) + region.hslope * (xx - e)
    };
    let x_at = |l: &Line, yy: f64| {
        let e = yy.max(l.lo).min(l.hi);
        l.at(e) + region.vslope * (yy - e)
    };

    // 1. column rules
    let wv: Vec<f64> = vs.iter().map(|l| l.length() / hh).collect();
    let (_, _, mut pv) = match_1d(&region.vlines, &fv, 12.0, (0.85, 1.18), Some(&wv), Some(80.0), None);
    if pv.len() < 3 {
        return Ok(Snap::fail(format!("{}: only {}/{} column rules found", region.name, pv.len(), region.vlines.len()), 0, pv.len()));
    }
    pv.sort_by(|a, b| region.vlines[a.0].partial_cmp(&region.vlines[b.0]).unwrap());
    let mut tops: Vec<f64> = pv.iter().map(|&(_, b)| vs[b].lo).collect();
    let mut bots: Vec<f64> = pv.iter().map(|&(_, b)| vs[b].hi).collect();
    let mut top = median(&tops);
    let bot = median(&bots);
    let (t_top, t_bot) = (region.top - sy, region.bottom - sy);
    if region.open_top || ((bot - t_bot).abs() <= 20.0 && top < t_top - 80.0) {
        if (bot - t_bot).abs() > 80.0 || top > t_top + 0.25 * (t_bot - t_top) {
            return Ok(Snap::fail(format!("{}: column rules don't reach the table", region.name), 0, pv.len()));
        }
        top = bot - (t_bot - t_top);
        tops = vec![top; tops.len()];
    } else if !(0.75..=1.35).contains(&((bot - top) / (t_bot - t_top))) || (top - t_top).abs() > 80.0 {
        return Ok(Snap::fail(format!("{}: column rules don't span the table", region.name), 0, pv.len()));
    }
    for t in tops.iter_mut() {
        if (*t - top).abs() > 20.0 {
            *t = top;
        }
    }
    for b in bots.iter_mut() {
        if (*b - bot).abs() > 20.0 {
            *b = bot;
        }
    }
    let xs_t: Vec<f64> = pv.iter().map(|&(k, _)| region.vlines[k]).collect();

    // 2. row rules
    let wh: Vec<f64> = hs.iter().map(|l| l.length() / w).collect();
    let ends = (
        top + sy + (region.hlines.first().copied().unwrap_or(0.0) - region.top),
        15.0,
        bot + sy + (region.hlines.last().copied().unwrap_or(0.0) - region.bottom),
        25.0,
    );
    let (_, _, ph) = match_1d(&region.hlines, &fh, 0.3 * pitch, (0.75, 1.35), Some(&wh), None, Some(ends));
    let mut rows: Vec<(f64, &Line)> = Vec::new();
    if ph.len() as f64 >= (3f64).max(min_frac_h * region.hlines.len() as f64) {
        rows = ph.iter().map(|&(i, j)| (region.hlines[i], &hs[j])).collect();
        rows.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    }
    let mut p_rows: Option<Vec<Vec<[f64; 2]>>> = None;
    if !rows.is_empty() {
        let mut pr = vec![vec![[0.0; 2]; pv.len()]; rows.len()];
        for (r, (_, hl)) in rows.iter().enumerate() {
            for (c, &(_, b)) in pv.iter().enumerate() {
                let mut xx = x_at(&vs[b], y_at(hl, 0.0));
                let mut yy = 0.0;
                for _ in 0..3 {
                    yy = y_at(hl, xx);
                    xx = x_at(&vs[b], yy);
                }
                pr[r][c] = [xx, yy];
            }
        }
        let ys_r: Vec<f64> = rows.iter().map(|r| r.0 - sy).collect();
        let mut dmin = f64::INFINITY;
        let mut dmax = f64::NEG_INFINITY;
        for r in 0..rows.len() - 1 {
            for c in 0..pv.len() {
                let d = (pr[r + 1][c][1] - pr[r][c][1]) / (ys_r[r + 1] - ys_r[r]);
                dmin = dmin.min(d);
                dmax = dmax.max(d);
            }
        }
        if rows.len() < 2 || !(dmin < 0.5 || dmax > 2.0) {
            p_rows = Some(pr);
        }
    }
    let (ys_t, mut p): (Vec<f64>, Vec<Vec<[f64; 2]>>) = if let Some(pr) = p_rows.clone() {
        (rows.iter().map(|r| r.0).collect(), pr)
    } else {
        let mut pp = vec![vec![[0.0; 2]; pv.len()]; 2];
        for (c, &(_, b)) in pv.iter().enumerate() {
            pp[0][c] = [x_at(&vs[b], tops[c]), tops[c]];
            pp[1][c] = [x_at(&vs[b], bots[c]), bots[c]];
        }
        (vec![region.top, region.bottom], pp)
    };
    for row in p.iter_mut() {
        for q in row.iter_mut() {
            q[0] += sx;
            q[1] += sy;
        }
    }
    for row in &p {
        for c in 0..row.len() - 1 {
            let d = (row[c + 1][0] - row[c][0]) / (xs_t[c + 1] - xs_t[c]);
            if !(0.5..=2.0).contains(&d) {
                return Ok(Snap::fail(format!("{}: column rules out of order", region.name), ph.len(), pv.len()));
            }
        }
    }
    let src: Vector<Point2f> =
        ys_t.iter().flat_map(|&yy| xs_t.iter().map(move |&xx| Point2f::new(xx as f32, yy as f32))).collect();
    let dst: Vector<Point2f> = p.iter().flat_map(|r| r.iter().map(|q| Point2f::new(q[0] as f32, q[1] as f32))).collect();
    let g = calib3d::find_homography(&src, &dst, &mut Mat::default(), 0, 3.0)?;
    let gh = util::arr3(&g)?;
    let mut se = 0.0;
    let mut n = 0.0;
    for (a, b) in src.iter().zip(dst.iter()) {
        let q = util::project(&gh, [a.x as f64, a.y as f64]);
        se += (q[0] - b.x as f64).powi(2) + (q[1] - b.y as f64).powi(2);
        n += 1.0;
    }
    let rms = (se / n).sqrt();
    if rms > max_rms {
        return Ok(Snap { rms, ..Snap::fail(format!("{}: rule grid doesn't fit (rms {:.1}px)", region.name, rms), ph.len(), pv.len()) });
    }
    let p_scan: Vec<Vec<[f64; 2]>> = p.iter().map(|r| r.iter().map(|q| util::project(h, *q)).collect()).collect();
    Ok(Snap {
        ok: true,
        reason: String::new(),
        matched_h: if p_rows.is_some() { ph.len() } else { 0 },
        matched_v: pv.len(),
        rms,
        xs: xs_t,
        ys: ys_t,
        p: p_scan,
    })
}
