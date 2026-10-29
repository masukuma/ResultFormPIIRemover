//! Per-page redaction.
//!
//! Fail-closed: a page is only released if it aligned confidently to a known
//! template, its tables were placed against their printed rules (or the zones
//! widened), no mask touches a protected results region, and the leak check
//! finds nothing left readable. Anything else goes to human review.
//!
//! Masks are drawn on a separate canvas, snapped to the 8x8 block grid that the
//! JPEG proof works on (proof.rs), and only then applied. The page result
//! carries that block grid; nothing else about the image is ever changed.
//!
//! Stamps are never masked. Every page is searched for them (stamps.rs); each
//! becomes a protected region, and no block touching one is blacked out, even
//! where a signature runs across it.

use anyhow::Result;
use opencv::core::Mat;
use opencv::prelude::*;
use serde_json::{json, Value};
use std::sync::Arc;
use std::time::Instant;

use crate::align::{best_alignment, AlignSettings, Alignment};
use crate::masking::{self, MaskSettings, Mapper, Reach, SigOptions, Strike};
use crate::proof;
use crate::pyjson;
use crate::pyrand::PyRandom;
use crate::stamps::{self, Stamp};
use crate::structure::{self, Snap};
use crate::templates::{Template, Zone};
use crate::util::{self, Grid};

pub struct PageResult {
    pub ok: bool,
    pub image: Option<Mat>, // redacted page (pixels), only set when ok
    pub template: String,
    pub inliers: i64,
    pub score: f64,
    pub reason: String,
    pub ms: f64,
    pub masks: Vec<Value>,
    pub notes: Vec<String>,
    pub blocks: Option<Grid>,    // 8x8 blocks blacked out
    pub protected: Vec<Value>,   // results regions and stamps, as image polygons
    pub h: Option<[[f64; 3]; 3]>, // template -> image (page alignment)
    pub layout: Option<Value>,   // template-free pages: what the fields were found from
    /// Columns whose place or masking couldn't be confirmed (why the page failed)
    pub unsure: Vec<String>,
}

/// Something a vision model still saw on a redacted page: a label
/// ("id_number", "phone_number", "signature") and a box, image px.
#[derive(Clone, Debug, PartialEq)]
pub struct Found {
    pub label: String,
    pub rect: [f64; 4], // x0, y0, x1, y1
}

impl Found {
    pub fn to_value(&self) -> Value {
        json!({"label": self.label, "box": self.rect.iter().map(|&v| pyjson::round_n(v, 1)).collect::<Vec<_>>()})
    }
    pub fn from_value(v: &Value) -> Option<Found> {
        let b: Vec<f64> = v["box"].as_array()?.iter().filter_map(|x| x.as_f64()).collect();
        (b.len() == 4).then(|| Found { label: v["label"].as_str().unwrap_or("").to_string(), rect: [b[0], b[1], b[2], b[3]] })
    }
}

pub struct PageOptions {
    pub form: Option<String>,
    pub strict: bool,
    pub fallback_margin: (f64, f64),
    pub verify: bool,
    pub generic: bool,
    /// per page: what a vision model found still showing, masked too
    pub found: Vec<Vec<Found>>,
    /// per page: columns to hide whole (their place couldn't be confirmed)
    pub whole: Vec<Vec<String>>,
}

impl Default for PageOptions {
    fn default() -> Self {
        PageOptions { form: None, strict: false, fallback_margin: (50.0, 200.0), verify: true, generic: false, found: vec![], whole: vec![] }
    }
}

/// Canvas (0 = hide) -> block grid, and the image with those blocks black.
/// No block touching `keep` (nonzero: stamps) is blacked out.
pub fn apply(img: &Mat, canvas: &Mat, keep: Option<&Mat>) -> Result<(Grid, Mat)> {
    let hide: Vec<u8> = util::bytes(canvas)?.iter().map(|&v| (v == 0) as u8).collect();
    let mut blocks = proof::to_blocks(&util::from_bytes(canvas.rows(), canvas.cols(), &hide)?)?;
    if let Some(k) = keep {
        blocks = blocks.and_not(&proof::to_blocks(k)?);
    }
    let px = util::bytes(&proof::to_pixels(&blocks, img.rows(), img.cols())?)?;
    let mut out = img.try_clone()?;
    let ch = img.channels() as usize;
    {
        let d = out.data_bytes_mut()?;
        for (i, &v) in px.iter().enumerate() {
            if v != 0 {
                for c in 0..ch {
                    d[i * ch + c] = 0;
                }
            }
        }
    }
    Ok((blocks, out))
}

pub const STAMP_MARGIN: f64 = 6.0; // px kept clear around a stamp, plus 4% of its radius

/// Add stamps found around the masks, and lift the masks off every stamp.
/// Returns (stamps, keep mask, notes).
pub fn protect_stamps(img: &Mat, canvas: &mut Mat, found: &[Stamp], r_range: (f64, f64), maps: Option<&stamps::Maps>) -> Result<(Vec<Stamp>, Mat, Vec<String>)> {
    let cv = util::bytes(canvas)?;
    let hide: Vec<u8> = cv.iter().map(|&v| if v == 0 { 255 } else { 0 }).collect();
    let mut all = found.to_vec();
    if hide.iter().any(|&v| v > 0) {
        let hm = util::from_bytes(canvas.rows(), canvas.cols(), &hide)?;
        all.extend(stamps::near(img, &hm, r_range, found, maps)?);
    }
    let mut keep = util::zeros_u8(canvas.rows(), canvas.cols())?;
    for st in &all {
        util::fill_poly(&mut keep, &st.polygon(STAMP_MARGIN + 0.04 * st.r), 255.0)?;
    }
    // Masks are applied in whole 8x8 blocks, so what stays clear is every
    // block touching a stamp; the checks must allow for exactly that.
    let keep = proof::to_pixels(&proof::to_blocks(&keep)?, canvas.rows(), canvas.cols())?;
    let kv = util::bytes(&keep)?;
    let covered = hide.iter().zip(&kv).filter(|&(&h, &k)| h > 0 && k > 0).count();
    let mut notes = Vec::new();
    if covered > 0 {
        notes.push(format!("masks lifted off {} px inside stamps: writing on a stamp stays visible", covered));
    }
    {
        let d = canvas.data_bytes_mut()?;
        for (i, &k) in kv.iter().enumerate() {
            if k > 0 {
                d[i] = 255;
            }
        }
    }
    Ok((all, keep, notes))
}

/// Local scale (image px per template px) of a homography at `pt`.
fn local_scale(h: &[[f64; 3]; 3], pt: [f64; 2]) -> f64 {
    let q0 = util::project(h, pt);
    let q1 = util::project(h, [pt[0] + 1.0, pt[1]]);
    let q2 = util::project(h, [pt[0], pt[1] + 1.0]);
    let (ax, ay, bx, by) = (q1[0] - q0[0], q1[1] - q0[1], q2[0] - q0[0], q2[1] - q0[1]);
    (ax * by - ay * bx).abs().sqrt()
}

/// A number as Python prints it in a list: integral values without ".0"
/// (template coordinates are integers in the YAML).
fn py_num(x: f64) -> String {
    if x.fract() == 0.0 && x.abs() < 1e15 {
        format!("{}", x as i64)
    } else {
        pyjson::float_repr(x)
    }
}

fn group_by_key(zones: &[Zone], action: &str) -> Vec<(String, Vec<Zone>)> {
    let mut out: Vec<(String, Vec<Zone>)> = Vec::new();
    for z in zones.iter().filter(|z| z.action == action) {
        let k = z.key().to_string();
        match out.iter_mut().find(|(n, _)| *n == k) {
            Some(e) => e.1.push(z.clone()),
            None => out.push((k, vec![z.clone()])),
        }
    }
    out
}

fn mean_center(zones: &[Zone]) -> f64 {
    util::np_mean(&zones.iter().map(|z| z.rect[1] + z.rect[3] / 2.0).collect::<Vec<_>>())
}

/// A table found again by its printed column `headers` (two or more),
/// searched for far around where the page alignment `h` puts them: the
/// template -> image homography for the table, and the fit [x scale, x
/// shift, y shift] in template px. None if fewer than two are found or they
/// disagree on how far the table moved.
fn refit_by_headers(gray: &Mat, h: &[[f64; 3]; 3], reference: &Mat, headers: &[[f64; 4]]) -> Result<Option<([[f64; 3]; 3], [f64; 3])>> {
    if headers.len() < 2 {
        return Ok(None);
    }
    let f = masking::homography(*h);
    let mut found: Vec<([f64; 4], [f64; 2])> = Vec::new();
    for &hd in headers {
        if let (Some(d), ncc) = masking::find_label(gray, &f, reference, hd, 250.0, 250.0)? {
            if ncc >= 0.55 {
                found.push((hd, d));
            }
        }
    }
    if found.len() < 2 {
        return Ok(None);
    }
    let dys: Vec<f64> = found.iter().map(|(_, d)| d[1]).collect();
    let dy = util::median(&dys);
    let xs: Vec<f64> = found.iter().map(|(hd, _)| hd[0] + hd[2] / 2.0).collect();
    let xf: Vec<f64> = found.iter().zip(&xs).map(|((_, d), x)| x + d[0]).collect();
    let (a, b) = crate::npla::polyfit1(&xs, &xf)?;
    let off = xs.iter().zip(&xf).map(|(x, y)| (a * x + b - y).abs()).fold(0.0, f64::max);
    if dys.iter().any(|v| (v - dy).abs() > 20.0) || !(0.85..=1.15).contains(&a) || off > 10.0 {
        return Ok(None);
    }
    Ok(Some((util::mul3(h, &[[a, 0.0, b], [0.0, 1.0, dy], [0.0, 0.0, 1.0]]), [a, b, dy])))
}

/// A column's printed header matched at least this well (NCC) is trusted to
/// say where the column is; found more than `HEADER_ACROSS` px across from
/// it, or half a row up or down, the column was placed wrong.
const HEADER_NCC: f64 = 0.6;
const HEADER_ACROSS: f64 = 25.0;

pub fn redact_page(img: &Mat, templates: &[Template], s: &AlignSettings, ms: &MaskSettings, rng: &mut PyRandom, o: &PageOptions) -> Result<PageResult> {
    redact_page_at(img, 0, templates, s, ms, rng, o)
}

/// Page `index` of a document: `o.found` and `o.whole` are per page.
pub fn redact_page_at(img: &Mat, index: usize, templates: &[Template], s: &AlignSettings, ms: &MaskSettings, rng: &mut PyRandom, o: &PageOptions) -> Result<PageResult> {
    let extra: &[Found] = o.found.get(index).map(|v| v.as_slice()).unwrap_or(&[]);
    let whole: &[String] = o.whole.get(index).map(|v| v.as_slice()).unwrap_or(&[]);
    let t0 = Instant::now();
    let gray = util::gray(img)?;
    let a: Option<Alignment> = if templates.is_empty() { None } else { Some(best_alignment(&gray, templates, s, o.form.as_deref())?) };
    let tname = |a: &Option<Alignment>| a.as_ref().and_then(|a| a.template).map(|i| templates[i].name.clone()).unwrap_or_default();
    let fail = |reason: String, report: Vec<Value>, notes: Vec<String>| PageResult {
        ok: false,
        image: None,
        template: tname(&a),
        inliers: a.as_ref().map(|a| a.inliers).unwrap_or(0),
        score: a.as_ref().map(|a| a.score).unwrap_or(0.0),
        reason,
        ms: t0.elapsed().as_secs_f64() * 1000.0,
        masks: report,
        notes,
        blocks: None,
        protected: vec![],
        h: None,
        layout: None,
        unsure: vec![],
    };
    let al = match &a {
        Some(x) if x.ok => x,
        _ => {
            // A form type that has templates is never read without one.
            let templated = o.form.as_ref().map(|f| templates.iter().any(|t| &t.form == f)).unwrap_or(false);
            let why = a.as_ref().map(|x| x.reason.clone()).unwrap_or_else(|| "no templates".into());
            if o.generic && !templated {
                return crate::generic::redact_generic(img, ms, rng, o.verify, &why, extra);
            }
            return Ok(fail(why, vec![], vec![]));
        }
    };
    let t = &templates[al.template.unwrap()];
    let hh = al.h.unwrap();

    // Zones tied to a printed table are positioned by its rules; if they can't
    // be matched, the zones are widened by `fallback_margin` (or with
    // `strict` the page fails).
    // A table the page alignment puts far off (the bottom of a curled page)
    // is first found again by its printed column headers.
    let mut snaps: Vec<(String, Arc<Snap>)> = Vec::new();
    let mut refits: Vec<(String, [[f64; 3]; 3])> = Vec::new();
    let mut found_notes: Vec<String> = Vec::new();
    for (n, r) in &t.regions {
        let mut sn = structure::snap(&gray, &hh, r)?;
        if !sn.ok {
            let mut headers: Vec<[f64; 4]> = Vec::new();
            for z in t.zones.iter().filter(|z| z.region == *n) {
                if let Some(hd) = z.fence[0] {
                    if !headers.contains(&hd) {
                        headers.push(hd);
                    }
                }
            }
            if let Some((h2, [a, _, dy])) = refit_by_headers(&gray, &hh, &t.image, &headers)? {
                refits.push((n.clone(), h2));
                let s2 = structure::snap(&gray, &h2, r)?;
                if s2.ok {
                    found_notes.push(format!("{}: table found by its printed headers ({:+.0}px down, {:.2}x across), then snapped ({})", n, dy, a, sn.reason));
                    sn = s2;
                }
            }
        }
        snaps.push((n.clone(), Arc::new(sn)));
    }
    let failed: Vec<(String, String)> = snaps.iter().filter(|(_, sn)| !sn.ok).map(|(n, sn)| (n.clone(), sn.reason.clone())).collect();
    let is_failed = |region: &str| failed.iter().any(|(n, _)| n == region);
    if !failed.is_empty() && o.strict {
        return Ok(fail(failed.iter().map(|(_, r)| r.clone()).collect::<Vec<_>>().join("; "), vec![], vec![]));
    }
    let fm = o.fallback_margin;
    let mut notes: Vec<String> = found_notes;
    notes.extend(failed.iter().map(|(_, why)| format!("{}; zones widened by {:.0}px up, {:.0}px down", why, fm.0, fm.1)));
    let margin = |zones: &[Zone]| if is_failed(&zones[0].region) { fm } else { (0.0, 0.0) };

    let base_mapper = |zones: &[Zone]| -> Result<Mapper> {
        if !zones[0].region.is_empty() && !is_failed(&zones[0].region) {
            let sn = snaps.iter().find(|(n, _)| *n == zones[0].region).unwrap().1.clone();
            return Ok(Arc::new(move |pts: &[[f64; 2]]| sn.map(pts)));
        }
        if let Some((_, h2)) = refits.iter().find(|(n, _)| *n == zones[0].region) {
            return Ok(masking::homography(*h2));
        }
        let x0 = zones.iter().map(|z| z.rect[0]).fold(f64::INFINITY, f64::min);
        let x1 = zones.iter().map(|z| z.rect[0] + z.rect[2]).fold(f64::NEG_INFINITY, f64::max);
        let cy = mean_center(zones);
        Ok(masking::homography(al.local([(x0 + x1) / 2.0, cy])?))
    };

    // Zones tied to a printed label move with it (layout variants); a label
    // that can't be found fails the page. Each distinct label is searched once.
    let mut offsets: Vec<([f64; 4], Mapper)> = Vec::new();
    let mut lost = Vec::new();
    for z in &t.zones {
        let Some(anchor) = z.anchor else { continue };
        if offsets.iter().any(|(k, _)| *k == anchor) {
            continue;
        }
        let f = base_mapper(std::slice::from_ref(z))?;
        let (d, ncc) = masking::find_anchor(&gray, &f, &t.image, anchor)?;
        match d {
            None => lost.push(format!("{}: printed label not found (best match {:.2})", z.key(), ncc)),
            Some(d) => {
                offsets.push((anchor, masking::shifted(f, d)));
                let dist = d[0].hypot(d[1]);
                if dist > 25.0 {
                    notes.push(format!("label at [{}, {}] found {:.0}px from its template position", py_num(anchor[0]), py_num(anchor[1]), dist));
                }
            }
        }
    }
    if !lost.is_empty() {
        return Ok(fail(lost.join("; "), vec![], vec![]));
    }
    let placed = |zones: &[Zone]| -> Result<Mapper> {
        match zones[0].anchor {
            Some(a) => Ok(offsets.iter().find(|(k, _)| *k == a).unwrap().1.clone()),
            None => base_mapper(zones),
        }
    };

    let columns = group_by_key(&t.zones, "partial");
    let signatures = group_by_key(&t.zones, "signature");

    // The masked columns must sit between their printed rules. A column that
    // doesn't (the table matched a rule off, or the snap gave up on faint
    // rules) is placed again by its printed header, the fence above it, and
    // must then pass the same check; or the page fails. The signature
    // columns of its table are moved by their own headers when that passes
    // the check too.
    //
    // Each column's printed header is also a witness of its own: found
    // clearly a good way across from the column, or half a row or more above
    // or below it (a table matched a rule or a row off: its rules look the
    // same at every column's edges), the column is moved down to it and
    // placed again by it, for signatures too.
    let mut moved: Vec<(String, Mapper, [f64; 3])> = Vec::new();
    let mut misplaced = Vec::new();
    let mut unsure: Vec<String> = Vec::new();
    if o.verify {
        // A header found clearly is itself evidence of where the column is,
        // when faint rules can't be seen.
        let again = |f: &Mapper, cells: &[Zone], require: bool, dy: f64| -> Result<Option<(Mapper, [f64; 3])>> {
            let Some(header) = cells[0].fence[0] else { return Ok(None) };
            let f = masking::shifted(f.clone(), [0.0, dy]);
            let Some((g, ab, ncc)) = masking::place_by_header(&gray, &f, &t.image, header, cells)? else { return Ok(None) };
            Ok(if masking::check_column_edges(&gray, &g, cells, require && ncc < 0.55)?.is_empty() { Some((g, [ab[0], ab[1], dy])) } else { None })
        };
        let header_off = |f: &Mapper, cells: &[Zone]| -> Result<Option<[f64; 2]>> {
            let Some(header) = cells[0].fence[0] else { return Ok(None) };
            let row_h = util::median(&cells.iter().map(|z| z.rect[3]).collect::<Vec<_>>());
            let (d, ncc) = masking::find_label(&gray, f, &t.image, header, cells[0].rect[2].max(120.0), 120.0)?;
            Ok(d.filter(|d| ncc >= HEADER_NCC && (d[0].abs() > HEADER_ACROSS || d[1].abs() >= 0.5 * row_h)))
        };
        let across = |cells: &[Zone], ab: [f64; 3]| {
            let mid = cells[0].rect[0] + cells[0].rect[2] / 2.0;
            ab[0] * mid + ab[1] - mid
        };
        let mut tables: Vec<String> = Vec::new();
        for (key, cells) in &columns {
            if cells[0].region.is_empty() || whole.contains(key) {
                continue;
            }
            let f = placed(cells)?;
            let require = is_failed(&cells[0].region);
            let off = header_off(&f, cells)?;
            let r = match off {
                Some(d) => format!("{}: printed header found {:+.0}px across, {:+.0}px down from the column", key, d[0], d[1]),
                None => masking::check_column_edges(&gray, &f, cells, require)?,
            };
            if r.is_empty() {
                continue;
            }
            match again(&f, cells, require, off.map(|d| d[1]).unwrap_or(0.0))? {
                Some((g, ab)) => {
                    notes.push(format!("{}: placed by its printed header, {:+.0}px across, {:+.0}px down ({})", key, across(cells, ab), ab[2], r.split(": ").nth(1).unwrap_or(&r)));
                    moved.push((key.clone(), g, ab));
                    tables.push(cells[0].region.clone());
                }
                None => {
                    misplaced.push(r);
                    unsure.push(key.clone());
                }
            }
        }
        for (key, cells) in &signatures {
            if !misplaced.is_empty() || cells[0].region.is_empty() {
                continue;
            }
            let f = placed(cells)?;
            let off = header_off(&f, cells)?;
            if off.is_none() && !tables.contains(&cells[0].region) {
                continue;
            }
            match again(&f, cells, false, off.map(|d| d[1]).unwrap_or(0.0))? {
                Some((g, ab)) => {
                    notes.push(format!("{}: placed by its printed header, {:+.0}px across, {:+.0}px down", key, across(cells, ab), ab[2]));
                    moved.push((key.clone(), g, ab));
                }
                // its own header says it is off and it can't be placed
                None if off.is_some() => {
                    let d = off.unwrap();
                    misplaced.push(format!("{}: printed header found {:+.0}px across, {:+.0}px down from the column", key, d[0], d[1]));
                    unsure.push(key.clone());
                }
                None => {}
            }
        }
    }
    if !misplaced.is_empty() {
        return Ok(PageResult { unsure, ..fail(misplaced.join("; "), vec![], notes) });
    }
    let mapper = |zones: &[Zone]| -> Result<Mapper> {
        match moved.iter().find(|(k, _, _)| k == zones[0].key()) {
            Some((_, g, _)) => Ok(g.clone()),
            None => placed(zones),
        }
    };

    // A widened search stops at the printed labels just outside its table
    // (the zones' `fence`), found on the page as anchors are, so it doesn't
    // run into the table's header or the next section. A label that can't be
    // found leaves that side wide: the search then errs on hiding more.
    // A placed table's searches stop at them too: the overrun a signature is
    // looked for in mustn't take the header's print for its ink.
    // A label to one side of a column bounds it at the label's level carried
    // across to the column: along the reference's rows, which may slope (the
    // "Reasons for Refusal" label lies 1,100px left of the Form 34A agents'
    // signatures, which hang below the table's last rule); and for a widened
    // search `FENCE_SLOPE` of the distance either way, for a page tilted or
    // curled against the fit.
    const FENCE_SLOPE: f64 = 0.05;
    let mut reaches: Vec<(String, Reach)> = Vec::new();
    for (key, cells) in columns.iter().chain(&signatures) {
        let (up, down) = margin(cells);
        let mut r = Reach::margin(up, down);
        let wide = (up, down) != (0.0, 0.0);
        if wide || cells[0].fence.iter().any(|f| f.is_some()) {
            let f = mapper(cells)?;
            let top = cells.iter().map(|z| z.rect[1]).fold(f64::INFINITY, f64::min);
            let bottom = cells.iter().map(|z| z.rect[1] + z.rect[3]).fold(f64::NEG_INFINITY, f64::max);
            let half_row = 0.5 * util::median(&cells.iter().map(|z| z.rect[3]).collect::<Vec<_>>());
            let mid = cells.iter().map(|z| z.rect[0] + z.rect[2] / 2.0).sum::<f64>() / cells.len() as f64;
            let slope = t.regions.iter().find(|(n, _)| *n == cells[0].region).map(|(_, r)| r.hslope).unwrap_or(0.0);
            for (side, label) in cells[0].fence.iter().enumerate() {
                let Some(label) = label else { continue };
                let (d, ncc) = masking::find_anchor(&gray, &f, &t.image, *label)?;
                let place = if side == 0 { "above" } else { "below" };
                let across = mid - (label[0] + label[2] / 2.0);
                let (level, tilt) = (slope * across, if wide { FENCE_SLOPE * across.abs() } else { 0.0 });
                // up to half a row into the cells as placed; deeper, it can't
                // be the label outside the table
                match d.map(|d| level + if side == 0 { label[1] + label[3] + d[1] } else { label[1] + d[1] }) {
                    // a snapped table's cells are where they are: a label
                    // only bounds what lies outside them
                    Some(y) if side == 0 && y <= top + half_row => r.top = Some(if wide { y - tilt } else { y.min(top) }),
                    Some(y) if side == 1 && y >= bottom - half_row => r.bottom = Some(if wide { y + tilt } else { y.max(bottom) }),
                    Some(_) if wide => notes.push(format!("{}: printed label {} the table found inside it; search left wide", key, place)),
                    None if wide => notes.push(format!("{}: printed label {} the table not found (best match {:.2}); search left wide", key, place, ncc)),
                    _ => {}
                }
            }
        }
        reaches.push((key.clone(), r));
    }
    let reach = |cells: &[Zone]| -> Reach { reaches.iter().find(|(k, _)| k == cells[0].key()).map(|(_, r)| *r).unwrap_or_default() };

    // Lines drawn through unused rows, found per printed table.
    let mut strikes: Vec<(String, Vec<Strike>)> = Vec::new();
    for (n, rect) in &t.structure {
        let cells: Vec<Zone> = t.zones.iter().filter(|z| z.region == *n).cloned().collect();
        if !cells.is_empty() {
            let row_h = util::median(&cells.iter().map(|z| z.rect[3]).collect::<Vec<_>>());
            strikes.push((n.clone(), masking::find_strikes(&gray, &placed(&cells)?, *rect, row_h)?));
        }
    }
    for (n, v) in &strikes {
        if !v.is_empty() {
            notes.push(format!("{}: {} line(s) drawn through the table left unmasked", n, v.len()));
        }
    }
    // a column placed again finds them over its table again, as placed:
    // where the table first stood can be rows off, and a line's part there
    // is not where the column's rows now are
    let mut own: Vec<(String, Vec<Strike>)> = Vec::new();
    for (key, _, _) in &moved {
        let Some((_, cells)) = columns.iter().chain(&signatures).find(|(k, _)| k == key) else { continue };
        let Some((_, rect)) = t.structure.iter().find(|(n, _)| *n == cells[0].region) else { continue };
        let row_h = util::median(&cells.iter().map(|z| z.rect[3]).collect::<Vec<_>>());
        own.push((key.clone(), masking::find_strikes(&gray, &mapper(cells)?, *rect, row_h)?));
    }
    let struck = |cells: &[Zone]| -> Vec<Strike> {
        match own.iter().find(|(k, _)| k == cells[0].key()) {
            Some((_, v)) => v.clone(),
            None => strikes.iter().find(|(n, _)| *n == cells[0].region).map(|(_, v)| v.clone()).unwrap_or_default(),
        }
    };

    // Stamps: found before masking; never masked (below).
    let (tw, th) = t.size();
    let k = local_scale(&hh, [tw as f64 / 2.0, th as f64 / 2.0]) * t.page_long_side.unwrap_or(tw.max(th) as f64) / 2340.0;
    let r_range = (60.0 * k, 135.0 * k);
    let maps = stamps::ink_maps(img)?;
    let on_page = stamps::find(img, r_range, &[], Some(&maps))?;
    let stamp_ink = if on_page.is_empty() { None } else { Some(stamps::protect_mask(gray.rows(), gray.cols(), &on_page, STAMP_MARGIN)?) };
    let zms = |_: &[Zone]| -> MaskSettings {
        let mut m = ms.clone();
        if let Some(g) = t.min_glyph_h.filter(|&g| g != 0.0) {
            m.min_glyph_h = g;
        }
        m
    };

    let mut canvas = util::full_u8(gray.rows(), gray.cols(), 255)?; // 0 = hide
    let mut report: Vec<Value> = Vec::new();
    for z in t.zones.iter().filter(|z| z.action == "blackout") {
        let (up, down) = margin(std::slice::from_ref(z));
        let zz = if up != 0.0 || down != 0.0 {
            Zone::new(&z.name, [z.rect[0], z.rect[1] - up, z.rect[2], z.rect[3] + up + down], "blackout")
        } else {
            z.clone()
        };
        masking::blackout(&mut canvas, &mapper(std::slice::from_ref(z))?, &zz, t.padding as f64)?;
        report.push(json!({"zone": z.name, "mode": "blackout"}));
    }
    // Columns whose place couldn't be confirmed (`whole`), with the
    // signatures of their table: hidden whole, a quarter of a column wider
    // either side and half a row (or the widened search) above and below.
    let whole_tables: Vec<&str> = columns.iter().filter(|(k, _)| whole.contains(k)).map(|(_, c)| c[0].region.as_str()).collect();
    let hidden = |key: &str, cells: &[Zone]| whole.iter().any(|k| k == key) || (cells[0].action == "signature" && whole_tables.contains(&cells[0].region.as_str()));
    for (key, cells) in columns.iter().chain(&signatures) {
        if !hidden(key, cells) {
            continue;
        }
        let x0 = cells.iter().map(|z| z.rect[0]).fold(f64::INFINITY, f64::min);
        let x1 = cells.iter().map(|z| z.rect[0] + z.rect[2]).fold(f64::NEG_INFINITY, f64::max);
        let y0 = cells.iter().map(|z| z.rect[1]).fold(f64::INFINITY, f64::min);
        let y1 = cells.iter().map(|z| z.rect[1] + z.rect[3]).fold(f64::NEG_INFINITY, f64::max);
        let half_row = 0.5 * util::median(&cells.iter().map(|z| z.rect[3]).collect::<Vec<_>>());
        let (up, down) = margin(cells);
        let wx = 0.25 * (x1 - x0);
        let z = Zone::new(&format!("{}:whole", key), [x0 - wx, y0 - up.max(half_row), x1 - x0 + 2.0 * wx, y1 - y0 + up.max(half_row) + down.max(half_row)], "blackout");
        masking::blackout(&mut canvas, &mapper(cells)?, &z, t.padding as f64)?;
        report.push(json!({"zone": key, "mode": "whole"}));
        notes.push(format!("{}: its place couldn't be confirmed; the whole column hidden", key));
    }
    // An underlined field (not a table column, and placed where expected):
    // only writing on its own line is a signature.
    let on_line = |cells: &[Zone]| cells[0].region.is_empty() && reach(cells).margin == (0.0, 0.0);
    for (key, cells) in &signatures {
        if hidden(key, cells) {
            continue;
        }
        let st = struck(cells);
        let opts = SigOptions { reach: reach(cells), reference: Some(&t.image), printed: None, centered: on_line(cells), strikes: Some(st.as_slice()), exclude: stamp_ink.as_ref() };
        report.extend(masking::signature_column(&mut canvas, &gray, &mapper(cells)?, cells, ms, &opts)?);
    }
    for (key, cells) in &columns {
        if hidden(key, cells) {
            continue;
        }
        let st = struck(cells);
        report.extend(masking::partial_column(&mut canvas, &gray, &mapper(cells)?, cells, &zms(cells), rng, reach(cells), Some(&t.image), Some(st.as_slice()), stamp_ink.as_ref())?);
    }

    // Areas of free writing (the comments): read as forms without a template
    // are, for IDs, phone numbers and signatures written there. A signature
    // found there is hidden whole, box and all.
    let mut free: Vec<(String, crate::generic::FreeText)> = Vec::new();
    if !t.free_text.is_empty() {
        if let Err(e) = crate::ocr::check() {
            return Ok(fail(format!("free text can't be read: {}", e), report, notes));
        }
        for (name, rect) in &t.free_text {
            let hl = al.local([rect[0] + rect[2] / 2.0, rect[1] + rect[3] / 2.0])?;
            let text_h = t.text_h.unwrap_or(crate::generic::TEXT_H);
            let ft = crate::ocr::with_engine(|o| crate::generic::read_free_text(img, &hl, *rect, text_h, &on_page, Some(&t.image), o))?;
            let f = ft.mapper();
            for fl in &ft.layout.fields {
                if fl.kind == "signature" {
                    let [x, y, w, h] = fl.zones[0].rect;
                    masking::fill(&mut canvas, &f, &[[x, y], [x + w, y], [x + w, y + h], [x, y + h]])?;
                    report.push(json!({"zone": format!("{}:{}", name, fl.zones[0].name), "mode": "signature", "signatures": 1}));
                } else {
                    report.extend(masking::partial_column(&mut canvas, &gray, &f, &fl.zones, &crate::generic::field_settings(ms, fl), rng, Reach::default(), Some(&ft.print_ref), None, stamp_ink.as_ref())?);
                }
            }
            notes.extend(ft.layout.notes.iter().map(|n| format!("{}: {}", name, n)));
            free.push((name.clone(), ft));
        }
    }

    let found_zones = mask_found(&mut canvas, &gray, extra, &zms(&[]), rng, stamp_ink.as_ref(), &mut report)?;

    // Stamps: never masked.
    // stamps found while reading the free text are protected as well
    let mut known = on_page.clone();
    for (_, ft) in &free {
        let f = ft.mapper();
        for st in &ft.layout.stamps {
            let p = f(&[[st.cx, st.cy], [st.cx + st.r, st.cy]]);
            let r = (p[1][0] - p[0][0]).hypot(p[1][1] - p[0][1]);
            if !known.iter().any(|k| (k.cx - p[0][0]).hypot(k.cy - p[0][1]) < k.r.max(r)) {
                known.push(Stamp::new(p[0][0], p[0][1], r, st.coverage, &st.how));
            }
        }
    }
    let (found, keep, stamp_notes) = protect_stamps(img, &mut canvas, &known, r_range, Some(&maps))?;
    notes.extend(stamp_notes);
    let (blocks, out) = apply(img, &canvas, Some(&keep))?;

    // Results regions must not lose a single block.
    let mut protected: Vec<Value> = found.iter().map(|s| s.record()).collect();
    let mut prot_mask = util::zeros_u8(gray.rows(), gray.cols())?;
    let hf = masking::homography(hh);
    for (name, [x, y, w, h]) in &t.protect {
        let poly = hf(&[[*x, *y], [x + w, *y], [x + w, y + h], [*x, y + h]]);
        util::fill_poly(&mut prot_mask, &poly, 255.0)?;
        let pv: Vec<Value> = poly.iter().map(|p| json!([pyjson::np_round(p[0], 1), pyjson::np_round(p[1], 1)])).collect();
        protected.push(json!({"name": name, "polygon": pv}));
    }
    let pb = proof::to_blocks(&prot_mask)?;
    if blocks.and(&pb).any() {
        return Ok(fail("a mask would touch a protected results region".into(), report, notes));
    }

    // Final gate: re-read every masked column from the output.
    let mut leaks = Vec::new();
    if o.verify {
        for (key, cells) in &columns {
            if hidden(key, cells) {
                continue;
            }
            let st = struck(cells);
            let r = masking::verify_column(&out, &mapper(cells)?, cells, Some(&t.image), &zms(cells), reach(cells), Some(st.as_slice()), stamp_ink.as_ref())?;
            if !r.is_empty() {
                leaks.push(r);
                unsure.push(key.clone());
            }
        }
        for (key, cells) in &signatures {
            if hidden(key, cells) {
                continue;
            }
            let st = struck(cells);
            let opts = SigOptions { reach: reach(cells), reference: Some(&t.image), printed: None, centered: on_line(cells), strikes: Some(st.as_slice()), exclude: stamp_ink.as_ref() };
            let r = masking::verify_signatures(&gray, &out, &mapper(cells)?, cells, ms, Some(&keep), &opts)?;
            if !r.is_empty() {
                leaks.push(r);
                unsure.push(key.clone());
            }
        }
        for (_, ft) in &free {
            for fl in ft.layout.fields.iter().filter(|fl| fl.kind != "signature") {
                let r = masking::verify_column(&out, &ft.mapper(), &fl.zones, Some(&ft.print_ref), &crate::generic::field_settings(ms, fl), Reach::default(), None, stamp_ink.as_ref())?;
                if !r.is_empty() {
                    leaks.push(r);
                }
            }
        }
        leaks.extend(verify_found(&out, &found_zones, &zms(&[]), stamp_ink.as_ref())?);
    }
    if !leaks.is_empty() {
        return Ok(PageResult { unsure, ..fail(format!("leak check: {}", leaks.join("; ")), report, notes) });
    }
    Ok(PageResult {
        ok: true,
        image: Some(out),
        template: t.name.clone(),
        inliers: al.inliers,
        score: al.score,
        reason: String::new(),
        ms: t0.elapsed().as_secs_f64() * 1000.0,
        masks: report,
        notes,
        blocks: Some(blocks),
        protected,
        h: Some(hh),
        layout: if free.is_empty() { None } else { Some(json!({"free_text": free.iter().map(|(n, ft)| ft.record(n)).collect::<Vec<_>>()})) },
        unsure: vec![],
    })
}

/// What a vision review still saw on a redacted page, masked (the manifest
/// records it, so a re-run masks the same): signatures hidden whole, numbers
/// as any number is. The number zones, for the leak check.
pub fn mask_found(canvas: &mut Mat, gray: &Mat, extra: &[Found], ms: &MaskSettings, rng: &mut PyRandom, stamp_ink: Option<&Mat>, report: &mut Vec<Value>) -> Result<Vec<Zone>> {
    let ident = masking::homography(util::IDENTITY);
    let mut zones = Vec::new();
    for b in extra {
        let [x0, y0, x1, y1] = b.rect;
        let name = format!("vlm:{}@{:.0},{:.0}", b.label, x0, y0);
        if b.label == "signature" {
            masking::fill(canvas, &ident, &[[x0 - 4.0, y0 - 4.0], [x1 + 4.0, y0 - 4.0], [x1 + 4.0, y1 + 4.0], [x0 - 4.0, y1 + 4.0]])?;
            report.push(json!({"zone": name, "mode": "signature", "signatures": 1}));
        } else {
            let phone = b.label == "phone_number";
            let mut z = Zone::new(&name, [x0, y0, x1 - x0, y1 - y0], "partial");
            z.digits = if phone { 10 } else { 8 };
            z.rule = if phone { "phone".into() } else { "id".into() };
            z.search = Some([8.0, 8.0]);
            report.extend(masking::partial_column(canvas, gray, &ident, std::slice::from_ref(&z), ms, rng, Reach::default(), None, None, stamp_ink)?);
            zones.push(z);
        }
    }
    Ok(zones)
}

/// The leak check, over the numbers `mask_found` masked.
pub fn verify_found(out: &Mat, zones: &[Zone], ms: &MaskSettings, stamp_ink: Option<&Mat>) -> Result<Vec<String>> {
    let ident = masking::homography(util::IDENTITY);
    let mut leaks = Vec::new();
    for z in zones {
        let r = masking::verify_column(out, &ident, std::slice::from_ref(z), None, ms, Reach::default(), None, stamp_ink)?;
        if !r.is_empty() {
            leaks.push(r);
        }
    }
    Ok(leaks)
}

pub fn redact_pages(pages: &[Mat], templates: &[Template], s: &AlignSettings, ms: &MaskSettings, rng: &mut PyRandom, o: &PageOptions) -> Result<Vec<PageResult>> {
    pages.iter().enumerate().map(|(i, p)| redact_page_at(p, i, templates, s, ms, rng, o)).collect()
}

/// Apply the same masking to boxes located by a VLM (no template): signatures
/// blacked out, ID/phone boxes partially masked. (x0, y0, x1, y1, label)
pub fn redact_boxes(img: &Mat, boxes: &[(f64, f64, f64, f64, String)], ms: &MaskSettings, rng: &mut PyRandom, pad: f64) -> Result<PageResult> {
    let t0 = Instant::now();
    let gray = util::gray(img)?;
    let mut canvas = util::full_u8(gray.rows(), gray.cols(), 255)?;
    let ident = masking::homography(util::IDENTITY);
    let mut report = Vec::new();
    for (i, (x0, y0, x1, y1, label)) in boxes.iter().enumerate() {
        let phone = label == "phone_number";
        let sig = label == "signature";
        let mut z = Zone::new(&format!("vlm_{}_{}", label, i), [*x0, *y0, x1 - x0, y1 - y0], if sig { "blackout" } else { "partial" });
        z.digits = if phone { 10 } else { 8 };
        z.rule = if sig { String::new() } else if phone { "phone".into() } else { "id".into() };
        if sig {
            masking::blackout(&mut canvas, &ident, &z, pad)?;
            report.push(json!({"zone": z.name, "mode": "blackout"}));
        } else {
            report.extend(masking::partial_column(&mut canvas, &gray, &ident, &[z], ms, rng, Reach::default(), None, None, None)?);
        }
    }
    let (blocks, out) = apply(img, &canvas, None)?;
    Ok(PageResult {
        ok: true,
        image: Some(out),
        template: "vlm".into(),
        inliers: 0,
        score: 0.0,
        reason: String::new(),
        ms: t0.elapsed().as_secs_f64() * 1000.0,
        masks: report,
        notes: vec![],
        blocks: Some(blocks),
        protected: vec![],
        h: None,
        layout: None,
        unsure: vec![],
    })
}

#[cfg(test)]
#[path = "tests/pipeline.rs"]
mod py_tests;
