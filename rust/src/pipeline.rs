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
}

pub struct PageOptions {
    pub form: Option<String>,
    pub strict: bool,
    pub fallback_margin: (f64, f64),
    pub verify: bool,
    pub generic: bool,
}

impl Default for PageOptions {
    fn default() -> Self {
        PageOptions { form: None, strict: false, fallback_margin: (50.0, 200.0), verify: true, generic: false }
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

pub fn redact_page(img: &Mat, templates: &[Template], s: &AlignSettings, ms: &MaskSettings, rng: &mut PyRandom, o: &PageOptions) -> Result<PageResult> {
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
    };
    let al = match &a {
        Some(x) if x.ok => x,
        _ => {
            // A form type that has templates is never read without one.
            let templated = o.form.as_ref().map(|f| templates.iter().any(|t| &t.form == f)).unwrap_or(false);
            let why = a.as_ref().map(|x| x.reason.clone()).unwrap_or_else(|| "no templates".into());
            if o.generic && !templated {
                return crate::generic::redact_generic(img, ms, rng, o.verify, &why);
            }
            return Ok(fail(why, vec![], vec![]));
        }
    };
    let t = &templates[al.template.unwrap()];
    let hh = al.h.unwrap();

    // Zones tied to a printed table are positioned by its rules; if they can't
    // be matched, the zones are widened by `fallback_margin` (or with
    // `strict` the page fails).
    let mut snaps: Vec<(String, Arc<Snap>)> = Vec::new();
    for (n, r) in &t.regions {
        snaps.push((n.clone(), Arc::new(structure::snap(&gray, &hh, r)?)));
    }
    let failed: Vec<(String, String)> = snaps.iter().filter(|(_, sn)| !sn.ok).map(|(n, sn)| (n.clone(), sn.reason.clone())).collect();
    let is_failed = |region: &str| failed.iter().any(|(n, _)| n == region);
    if !failed.is_empty() && o.strict {
        return Ok(fail(failed.iter().map(|(_, r)| r.clone()).collect::<Vec<_>>().join("; "), vec![], vec![]));
    }
    let fm = o.fallback_margin;
    let mut notes: Vec<String> = failed
        .iter()
        .map(|(_, why)| format!("{}; zones widened by {:.0}px up, {:.0}px down", why, fm.0, fm.1))
        .collect();
    let margin = |zones: &[Zone]| if is_failed(&zones[0].region) { fm } else { (0.0, 0.0) };

    let base_mapper = |zones: &[Zone]| -> Result<Mapper> {
        if !zones[0].region.is_empty() && !is_failed(&zones[0].region) {
            let sn = snaps.iter().find(|(n, _)| *n == zones[0].region).unwrap().1.clone();
            return Ok(Arc::new(move |pts: &[[f64; 2]]| sn.map(pts)));
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
    let mapper = |zones: &[Zone]| -> Result<Mapper> {
        match zones[0].anchor {
            Some(a) => Ok(offsets.iter().find(|(k, _)| *k == a).unwrap().1.clone()),
            None => base_mapper(zones),
        }
    };

    let columns = group_by_key(&t.zones, "partial");
    let signatures = group_by_key(&t.zones, "signature");

    // A widened search stops at the printed labels just outside its table
    // (the zones' `fence`), found on the page as anchors are, so it doesn't
    // run into the table's header or the next section. A label that can't be
    // found leaves that side wide: the search then errs on hiding more.
    let mut reaches: Vec<(String, Reach)> = Vec::new();
    for (key, cells) in columns.iter().chain(&signatures) {
        let (up, down) = margin(cells);
        let mut r = Reach::margin(up, down);
        if (up, down) != (0.0, 0.0) {
            let f = mapper(cells)?;
            let top = cells.iter().map(|z| z.rect[1]).fold(f64::INFINITY, f64::min);
            let bottom = cells.iter().map(|z| z.rect[1] + z.rect[3]).fold(f64::NEG_INFINITY, f64::max);
            let half_row = 0.5 * util::median(&cells.iter().map(|z| z.rect[3]).collect::<Vec<_>>());
            for (side, label) in cells[0].fence.iter().enumerate() {
                let Some(label) = label else { continue };
                let (d, ncc) = masking::find_anchor(&gray, &f, &t.image, *label)?;
                let place = if side == 0 { "above" } else { "below" };
                // up to half a row into the cells as placed; deeper, it can't
                // be the label outside the table
                match d.map(|d| if side == 0 { label[1] + label[3] + d[1] } else { label[1] + d[1] }) {
                    Some(y) if side == 0 && y <= top + half_row => r.top = Some(y),
                    Some(y) if side == 1 && y >= bottom - half_row => r.bottom = Some(y),
                    Some(_) => notes.push(format!("{}: printed label {} the table found inside it; search left wide", key, place)),
                    None => notes.push(format!("{}: printed label {} the table not found (best match {:.2}); search left wide", key, place, ncc)),
                }
            }
        }
        reaches.push((key.clone(), r));
    }
    let reach = |cells: &[Zone]| -> Reach { reaches.iter().find(|(k, _)| k == cells[0].key()).map(|(_, r)| *r).unwrap_or_default() };
    // The masked columns must sit between their printed rules.
    let mut misplaced = Vec::new();
    if o.verify {
        for (_, cells) in &columns {
            if cells[0].region.is_empty() {
                continue;
            }
            let r = masking::check_column_edges(&gray, &mapper(cells)?, cells, is_failed(&cells[0].region))?;
            if !r.is_empty() {
                misplaced.push(r);
            }
        }
    }
    if !misplaced.is_empty() {
        return Ok(fail(misplaced.join("; "), vec![], notes));
    }

    // Lines drawn through unused rows, found per printed table.
    let mut strikes: Vec<(String, Vec<Strike>)> = Vec::new();
    for (n, rect) in &t.structure {
        let cells: Vec<Zone> = t.zones.iter().filter(|z| z.region == *n).cloned().collect();
        if !cells.is_empty() {
            let row_h = util::median(&cells.iter().map(|z| z.rect[3]).collect::<Vec<_>>());
            strikes.push((n.clone(), masking::find_strikes(&gray, &mapper(&cells)?, *rect, row_h)?));
        }
    }
    for (n, v) in &strikes {
        if !v.is_empty() {
            notes.push(format!("{}: {} line(s) drawn through the table left unmasked", n, v.len()));
        }
    }
    let struck = |cells: &[Zone]| -> Vec<Strike> { strikes.iter().find(|(n, _)| *n == cells[0].region).map(|(_, v)| v.clone()).unwrap_or_default() };

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
    // An underlined field (not a table column, and placed where expected):
    // only writing on its own line is a signature.
    let on_line = |cells: &[Zone]| cells[0].region.is_empty() && reach(cells).margin == (0.0, 0.0);
    for (_, cells) in &signatures {
        let st = struck(cells);
        let opts = SigOptions { reach: reach(cells), reference: Some(&t.image), printed: None, centered: on_line(cells), strikes: Some(st.as_slice()), exclude: stamp_ink.as_ref() };
        report.extend(masking::signature_column(&mut canvas, &gray, &mapper(cells)?, cells, ms, &opts)?);
    }
    for (_, cells) in &columns {
        let st = struck(cells);
        report.extend(masking::partial_column(&mut canvas, &gray, &mapper(cells)?, cells, &zms(cells), rng, reach(cells), Some(&t.image), Some(st.as_slice()), stamp_ink.as_ref())?);
    }

    // Stamps: never masked.
    let (found, keep, stamp_notes) = protect_stamps(img, &mut canvas, &on_page, r_range, Some(&maps))?;
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
        for (_, cells) in &columns {
            let st = struck(cells);
            let r = masking::verify_column(&out, &mapper(cells)?, cells, Some(&t.image), &zms(cells), reach(cells), Some(st.as_slice()), stamp_ink.as_ref())?;
            if !r.is_empty() {
                leaks.push(r);
            }
        }
        for (_, cells) in &signatures {
            let st = struck(cells);
            let opts = SigOptions { reach: reach(cells), reference: Some(&t.image), printed: None, centered: on_line(cells), strikes: Some(st.as_slice()), exclude: stamp_ink.as_ref() };
            let r = masking::verify_signatures(&gray, &out, &mapper(cells)?, cells, ms, Some(&keep), &opts)?;
            if !r.is_empty() {
                leaks.push(r);
            }
        }
    }
    if !leaks.is_empty() {
        return Ok(fail(format!("leak check: {}", leaks.join("; ")), report, notes));
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
        layout: None,
    })
}

pub fn redact_pages(pages: &[Mat], templates: &[Template], s: &AlignSettings, ms: &MaskSettings, rng: &mut PyRandom, o: &PageOptions) -> Result<Vec<PageResult>> {
    pages.iter().map(|p| redact_page(p, templates, s, ms, rng, o)).collect()
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
    })
}

#[cfg(test)]
#[path = "tests/pipeline.rs"]
mod py_tests;
