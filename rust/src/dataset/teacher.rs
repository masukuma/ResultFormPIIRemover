//! The teacher's labels (`rfpii dataset build`): what the current pipeline
//! found on the pages it published, from their manifests, and the pages it
//! held, as the pool to learn from next.
//!
//! For each published page the original is found by its SHA-256 among
//! `--originals` (zip members included) and decoded exactly as the pipeline
//! stored it (`io::source_page`): a JPEG page is copied byte for byte. Its
//! labels:
//!
//! - **fields**: each template zone's cell (rows expanded), placed by the
//!   manifest's alignment (template to image); its class from the zone
//!   (signature, ID, phone), its role from the mask record. Where blocks
//!   were blacked in the cell's search area, the box is drawn from them
//!   instead: a number hidden whole is the box of its blocks; a number with
//!   some characters hidden is fitted from where those characters' blocks
//!   are (`lines[].hidden`: which of how many), which also gives each
//!   character's slot; a signature's band is its middle half, so the box is
//!   the band grown to twice its height. A cell the masking found empty is
//!   `empty: true`. The alignment is the page's; a table column snapped to
//!   its printed rules can sit a few pixels off it, which the blocks correct
//!   where there are any.
//! - **stamps** and **results regions** from `protected`; the **agents
//!   table**, **comments** and **declaration** areas from the template and
//!   `layout.rs`; the vision review's finds (`vlm_found`) and spot masks
//!   as boxes with their source.
//! - in the comments (free text) and on pages read without a template,
//!   the blacked areas, joined, are the boxes, their class told from the
//!   mask records or the nearest field label; these boxes are approximate
//!   (`approximate: true`), and a page read without a template lists no
//!   class as complete.
//! - names are never labelled: the pipeline doesn't look for them.
//!
//! The ink mask is the pipeline's own rules (`stamps::dark_ink`,
//! `stamps::color_ink`): dark ink in bit 1 (print and pen together: rules
//! can't tell them apart), coloured ink in bit 4.

use anyhow::{bail, Context, Result};
use opencv::core::Mat;
use opencv::prelude::*;
use rayon::prelude::*;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

use super::draw::{BIT_PRINT, BIT_STAMP};
use super::synth::{template_year, zone_class};
use super::{bounds, clip, r1, rbox, rpoly, PageOut, Writer, LABELS_SCHEMA};
use crate::proof::BLOCK;
use crate::templates::{Template, Zone};
use crate::util::{self, Grid};

type H3 = [[f64; 3]; 3];

pub struct BuildOptions {
    /// manifests, or folders holding them
    pub manifests: Vec<PathBuf>,
    pub originals: PathBuf,
    pub templates: Option<PathBuf>,
    /// a run's review folder: its report.jsonl lists the documents held
    pub review: Option<PathBuf>,
    pub out: PathBuf,
    pub append: bool,
    pub workers: usize,
    /// write the rules' ink masks
    pub masks: bool,
}

/// Classes a templated page's labels list completely, as far as the
/// teacher knows (names are never labelled).
pub const TEACHER_EXHAUSTIVE: &[&str] = &["signature", "id_number", "phone_number", "stamp", "results_region", "agents_table", "declaration", "comments"];

/// `folder` of a source: the folder holding the file (a zip member's
/// folder inside the zip, or the zip's name for a member at its root).
pub fn folder_key(path: &str) -> String {
    let (file, member) = match path.split_once(crate::io::ZIP_SEP) {
        Some((a, m)) => (a, Some(m)),
        None => (path, None),
    };
    let last = |p: &Path| p.parent().and_then(|q| q.file_name()).map(|s| s.to_string_lossy().into_owned());
    if let Some(m) = member {
        if let Some(f) = last(Path::new(m)) {
            return f;
        }
        return Path::new(file).file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_else(|| ".".into());
    }
    last(Path::new(file)).unwrap_or_else(|| ".".into())
}

/// The split key of a source: its county and constituency from the KIEMS
/// file name or the form's QR code, else the folder it is in. `meta` is a
/// manifest's `source` or a held document's report line.
pub fn split_key(meta: &Value, path: &str) -> String {
    for k in ["kiems", "qr_fields"] {
        let (c, s) = (meta[k]["county"].as_str(), meta[k]["constituency"].as_str());
        if let (Some(c), Some(s)) = (c, s) {
            return format!("county:{}/constituency:{}", c, s);
        }
    }
    format!("folder:{}", folder_key(path))
}

/// How a source was captured: a KIEMS kit's photo (KIEMS file name), a
/// photo (an image file), or a scan (a PDF).
pub fn capture(meta: &Value, path: &str) -> &'static str {
    if meta["kiems"].is_object() {
        return "kiems";
    }
    let name = path.rsplit(crate::io::ZIP_SEP).next().unwrap_or(path).to_ascii_lowercase();
    if name.ends_with(".pdf") { "scan" } else { "photo" }
}

fn year(meta: &Value, template: Option<&str>) -> Option<i64> {
    meta["kiems"]["captured_at"].as_str().and_then(|s| s.get(..4)?.parse().ok()).or_else(|| meta["qr_fields"]["year"].as_str().and_then(|s| s.parse().ok())).or_else(|| template.and_then(template_year))
}

fn place(meta: &Value) -> Value {
    for k in ["kiems", "qr_fields"] {
        if meta[k]["county"].is_string() {
            return json!({"county": meta[k]["county"], "constituency": meta[k]["constituency"], "station": meta[k]["station"]});
        }
    }
    Value::Null
}

fn h_of(v: &Value) -> Option<H3> {
    let rows = v.as_array()?;
    let mut h = [[0.0; 3]; 3];
    for (i, r) in rows.iter().enumerate().take(3) {
        for (j, x) in r.as_array()?.iter().enumerate().take(3) {
            h[i][j] = x.as_f64()?;
        }
    }
    Some(h)
}

fn quad(r: [f64; 4]) -> [[f64; 2]; 4] {
    let [x, y, w, h] = r;
    [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]
}

fn on_page(h: &H3, r: [f64; 4]) -> Vec<[f64; 2]> {
    quad(r).iter().map(|&p| util::project(h, p)).collect()
}

/// The blocks of a page in template space: each blacked block's centre
/// and box (template px).
struct TBlocks {
    centres: Vec<[f64; 2]>,
    boxes: Vec<[f64; 4]>,
    rc: Vec<(usize, usize)>,
}

fn template_blocks(g: &Grid, inv: &H3) -> TBlocks {
    let mut t = TBlocks { centres: vec![], boxes: vec![], rc: vec![] };
    let b = BLOCK as f64;
    for r in 0..g.rows {
        for c in 0..g.cols {
            if !g.get(r, c) {
                continue;
            }
            let (x0, y0) = (c as f64 * b, r as f64 * b);
            let corners: Vec<[f64; 2]> = [[x0, y0], [x0 + b, y0], [x0 + b, y0 + b], [x0, y0 + b]].iter().map(|&p| util::project(inv, p)).collect();
            t.centres.push(util::project(inv, [x0 + b / 2.0, y0 + b / 2.0]));
            t.boxes.push(bounds(&corners));
            t.rc.push((r, c));
        }
    }
    t
}

fn inside(p: [f64; 2], r: [f64; 4]) -> bool {
    p[0] >= r[0] && p[0] <= r[2] && p[1] >= r[1] && p[1] <= r[3]
}

/// A zone's search area (x0, y0, x1, y1, template px): its cell, the
/// padding either side, its search above and below.
fn search_area(z: &Zone, pad: f64) -> [f64; 4] {
    let [x, y, w, h] = z.rect;
    let [up, down] = z.search.unwrap_or([pad, pad]);
    [x - pad, y - up.max(pad), x + w + pad, y + h + down.max(pad)]
}

/// Fit where a partly hidden number's characters are from its hidden
/// characters' blocks: `runs` the x-spans (template px) of the runs of
/// blocks, left to right; `hidden` which of `n` characters were hidden.
/// Returns the number's x0 and the pitch of its characters.
pub fn fit_slots(runs: &[[f64; 2]], hidden: &[usize], n: usize) -> Option<(f64, f64)> {
    let mut groups: Vec<Vec<usize>> = Vec::new();
    let mut hs = hidden.to_vec();
    hs.sort();
    for i in hs {
        match groups.last_mut() {
            Some(g) if *g.last().unwrap() + 1 == i => g.push(i),
            _ => groups.push(vec![i]),
        }
    }
    if groups.is_empty() || groups.len() != runs.len() || n == 0 {
        return None;
    }
    // each run's centre is at x0 + pitch * (mean index + 0.5)
    let pts: Vec<(f64, f64)> = groups.iter().zip(runs).map(|(g, r)| (g.iter().sum::<usize>() as f64 / g.len() as f64 + 0.5, (r[0] + r[1]) / 2.0)).collect();
    let (x0, pitch) = if pts.len() >= 2 && pts.iter().any(|p| (p.0 - pts[0].0).abs() > 1e-9) {
        let m = pts.len() as f64;
        let (sx, sy) = (pts.iter().map(|p| p.0).sum::<f64>(), pts.iter().map(|p| p.1).sum::<f64>());
        let sxx = pts.iter().map(|p| p.0 * p.0).sum::<f64>();
        let sxy = pts.iter().map(|p| p.0 * p.1).sum::<f64>();
        let pitch = (m * sxy - sx * sy) / (m * sxx - sx * sx);
        ((sy - pitch * sx) / m, pitch)
    } else {
        let pitch = (runs[0][1] - runs[0][0]) / groups[0].len() as f64;
        (pts[0].1 - pitch * pts[0].0, pitch)
    };
    (pitch > 0.0).then_some((x0, pitch))
}

/// Runs of blocks along x (template px), left to right: blocks less than
/// `gap` apart join.
fn runs_of(boxes: &[[f64; 4]], gap: f64) -> Vec<[f64; 2]> {
    let mut spans: Vec<[f64; 2]> = boxes.iter().map(|b| [b[0], b[2]]).collect();
    spans.sort_by(|a, b| a[0].total_cmp(&b[0]));
    let mut out: Vec<[f64; 2]> = Vec::new();
    for s in spans {
        match out.last_mut() {
            Some(l) if s[0] <= l[1] + gap => l[1] = l[1].max(s[1]),
            _ => out.push(s),
        }
    }
    out
}

/// The labels of a page read with a template.
fn template_boxes(t: &Template, mp: &Value, w: f64, hh: f64, boxes: &mut Vec<Value>, notes: &mut Vec<String>) -> Result<()> {
    let Some(h) = h_of(&mp["alignment"]["H"]) else {
        notes.push("no alignment recorded".into());
        return Ok(());
    };
    let inv = util::inv3(&h)?;
    let grid = crate::proof::unpack(&mp["blocks"])?;
    let tb = template_blocks(&grid, &inv);
    let records: HashMap<String, &Value> = mp["masks"].as_array().into_iter().flatten().filter_map(|r| Some((r["zone"].as_str()?.to_string(), r))).collect();
    let pad = t.padding as f64;
    let img_box = |r: [f64; 4]| clip(bounds(&on_page(&h, r)), w, hh);
    let tbox_img = |b: [f64; 4]| clip(bounds(&on_page(&h, [b[0], b[1], b[2] - b[0], b[3] - b[1]])), w, hh);
    // turned or upright: slots are x-spans only on a page that is upright
    let ax = util::project(&h, [1.0, 0.0]);
    let a0 = util::project(&h, [0.0, 0.0]);
    let upright = (ax[0] - a0[0]) > 2.0 * (ax[1] - a0[1]).abs();
    let mut used = vec![false; tb.centres.len()];
    // the zones, column by column
    let mut keys: Vec<&str> = Vec::new();
    for z in &t.zones {
        if !keys.contains(&z.key()) {
            keys.push(z.key());
        }
    }
    for key in keys {
        let cells: Vec<&Zone> = t.zones.iter().filter(|z| z.key() == key).collect();
        let Some((class, rule)) = zone_class(cells[0]) else { continue };
        let whole = records.get(key).is_some_and(|r| r["mode"] == "whole");
        // each blacked block in the column's search goes to the nearest cell
        let areas: Vec<[f64; 4]> = cells.iter().map(|z| search_area(z, pad)).collect();
        let mut mine: Vec<Vec<usize>> = vec![Vec::new(); cells.len()];
        for (i, c) in tb.centres.iter().enumerate() {
            if used[i] {
                continue;
            }
            let near = cells.iter().enumerate().filter(|(k, _)| inside(*c, areas[*k])).min_by(|(_, a), (_, b)| {
                let d = |z: &Zone| (c[1] - (z.rect[1] + z.rect[3] / 2.0)).abs();
                d(a).total_cmp(&d(b))
            });
            if let Some((k, _)) = near {
                mine[k].push(i);
            }
        }
        for (k, z) in cells.iter().enumerate() {
            let rec = records.get(&z.name).copied();
            let mode = rec.and_then(|r| r["mode"].as_str()).unwrap_or(if whole { "whole" } else { "" });
            let role = rec.and_then(|r| r["role"].as_str()).unwrap_or(crate::masking::role_of(&z.name));
            let Some(cell) = img_box(z.rect) else { continue };
            let mut v = json!({"class": class, "box": rbox(cell), "source": "template_zone", "zone": z.name, "role": role,
                               "polygon": rpoly(&on_page(&h, z.rect)), "cell": rbox(cell), "mode": mode});
            if let Some(r) = rule {
                v["rule"] = json!(r);
            }
            if mode == "empty" {
                v["empty"] = json!(true);
                boxes.push(v);
                continue;
            }
            let blocks: Vec<[f64; 4]> = mine[k].iter().map(|&i| tb.boxes[i]).collect();
            if blocks.is_empty() || mode == "whole" || mode == "blackout" {
                boxes.push(v);
                continue;
            }
            for &i in &mine[k] {
                used[i] = true;
            }
            let u = bounds(&blocks.iter().flat_map(|b| [[b[0], b[1]], [b[2], b[3]]]).collect::<Vec<_>>());
            let tbox = if class == "signature" {
                // the band is the middle half of the signature
                let (cy, bh) = ((u[1] + u[3]) / 2.0, u[3] - u[1]);
                let a = search_area(z, pad);
                [u[0], (cy - bh).max(a[1]), u[2], (cy + bh).min(a[3])]
            } else if mode == "partial" {
                let line = rec.and_then(|r| r["lines"].as_array()).and_then(|l| l.iter().find(|x| x["hidden"].as_array().is_some_and(|h| !h.is_empty())));
                let n = line.and_then(|l| l["slots"].as_u64()).unwrap_or(0) as usize;
                let hidden: Vec<usize> = line.and_then(|l| l["hidden"].as_array()).map(|a| a.iter().filter_map(|x| x.as_u64().map(|v| v as usize)).collect()).unwrap_or_default();
                let bw = (blocks[0][2] - blocks[0][0]).max(1.0);
                let runs = runs_of(&blocks, 0.5 * bw);
                match fit_slots(&runs, &hidden, n) {
                    Some((x0, pitch)) if x0 > z.rect[0] - 0.3 * z.rect[2] && x0 + pitch * (n as f64) < z.rect[0] + 1.3 * z.rect[2] => {
                        let tb2 = [x0, u[1], x0 + pitch * n as f64, u[3]];
                        if upright {
                            let slots: Vec<Value> = (0..n)
                                .filter_map(|i| tbox_img([x0 + pitch * i as f64, u[1], x0 + pitch * (i + 1) as f64, u[3]]))
                                .map(|b| json!([r1(b[0]), r1(b[2])]))
                                .collect();
                            v["slots"] = Value::Array(slots);
                            v["slots_from"] = json!("hidden_blocks");
                        }
                        tb2
                    }
                    _ => [z.rect[0].min(u[0]), u[1], (z.rect[0] + z.rect[2]).max(u[2]), u[3]],
                }
            } else {
                u
            };
            if let Some(b) = tbox_img(tbox) {
                v["box"] = rbox(b);
                v["polygon"] = rpoly(&on_page(&h, [tbox[0], tbox[1], tbox[2] - tbox[0], tbox[3] - tbox[1]]));
                v["source"] = json!("blocks");
            }
            boxes.push(v);
        }
    }
    // the template's regions
    for (name, r) in &t.structure {
        if name == "agents" {
            if let Some(b) = img_box(*r) {
                boxes.push(json!({"class": "agents_table", "box": rbox(b), "source": "template_zone", "zone": name, "polygon": rpoly(&on_page(&h, *r))}));
            }
        }
    }
    for (name, r) in &t.free_text {
        if let Some(b) = img_box(*r) {
            boxes.push(json!({"class": "comments", "box": rbox(b), "source": "template_zone", "zone": name, "polygon": rpoly(&on_page(&h, *r))}));
        }
    }
    for r in super::layout::layout(&t.name).map(|l| l.declaration).unwrap_or(&[]) {
        if let Some(b) = img_box(*r) {
            boxes.push(json!({"class": "declaration", "box": rbox(b), "source": "template_zone", "zone": "declaration", "polygon": rpoly(&on_page(&h, *r))}));
        }
    }
    // what is blacked in the comments and not a zone's: joined, as boxes
    let free_numbers: Vec<&str> = records.iter().filter(|(k, r)| !k.starts_with("vlm:") && !t.zones.iter().any(|z| &z.name == *k || z.key() == k.as_str()) && r["rule"].is_string()).filter_map(|(_, r)| r["rule"].as_str()).collect();
    for (name, r) in &t.free_text {
        let area = [r[0], r[1], r[0] + r[2], r[1] + r[3]];
        let left: Vec<usize> = (0..tb.centres.len()).filter(|&i| !used[i] && inside(tb.centres[i], area)).collect();
        for comp in components(&left.iter().map(|&i| tb.rc[i]).collect::<Vec<_>>()) {
            let bxs: Vec<[f64; 4]> = comp.iter().map(|&k| tb.boxes[left[k]]).collect();
            let u = bounds(&bxs.iter().flat_map(|b| [[b[0], b[1]], [b[2], b[3]]]).collect::<Vec<_>>());
            let fill = comp.len() as f64 * BLOCK as f64 * BLOCK as f64 / ((u[2] - u[0]) * (u[3] - u[1])).max(1.0) * area_scale(&h);
            let class = if fill >= 0.6 || free_numbers.is_empty() {
                "signature"
            } else if free_numbers.iter().all(|r| *r == "phone") {
                "phone_number"
            } else {
                "id_number"
            };
            if let Some(b) = tbox_img(u) {
                boxes.push(json!({"class": class, "box": rbox(b), "source": "blocks", "zone": format!("{}:writing", name), "role": "officer", "approximate": true}));
            }
        }
    }
    Ok(())
}

/// Image px per template px, squared, at the template's middle.
fn area_scale(h: &H3) -> f64 {
    let a = util::project(h, [0.0, 0.0]);
    let b = util::project(h, [100.0, 0.0]);
    let c = util::project(h, [0.0, 100.0]);
    let det = ((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])).abs() / 10_000.0;
    if det > 0.0 { 1.0 / det } else { 1.0 }
}

/// 8-connected groups of (row, col) blocks; each group lists indices.
fn components(rc: &[(usize, usize)]) -> Vec<Vec<usize>> {
    let at: HashMap<(usize, usize), usize> = rc.iter().enumerate().map(|(i, &p)| (p, i)).collect();
    let mut seen = vec![false; rc.len()];
    let mut out = Vec::new();
    for s in 0..rc.len() {
        if seen[s] {
            continue;
        }
        let mut stack = vec![s];
        seen[s] = true;
        let mut comp = Vec::new();
        while let Some(i) = stack.pop() {
            comp.push(i);
            let (r, c) = rc[i];
            for dr in -2i64..=2 {
                for dc in -2i64..=2 {
                    let (nr, nc) = (r as i64 + dr, c as i64 + dc);
                    if nr < 0 || nc < 0 {
                        continue;
                    }
                    if let Some(&j) = at.get(&(nr as usize, nc as usize)) {
                        if !seen[j] {
                            seen[j] = true;
                            stack.push(j);
                        }
                    }
                }
            }
        }
        out.push(comp);
    }
    out
}

/// The labels of a page read without a template: the blacked areas,
/// joined, classed by the nearest field label the layout records.
fn generic_boxes(mp: &Value, w: f64, hh: f64, boxes: &mut Vec<Value>) -> Result<()> {
    let grid = crate::proof::unpack(&mp["blocks"])?;
    let rc: Vec<(usize, usize)> = (0..grid.rows).flat_map(|r| (0..grid.cols).map(move |c| (r, c))).filter(|&(r, c)| grid.get(r, c)).collect();
    let fields: Vec<(String, [f64; 2])> = mp["layout"]["fields"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|f| {
            let q: Vec<[f64; 2]> = f["label_polygon"].as_array()?.iter().filter_map(|p| Some([p[0].as_f64()?, p[1].as_f64()?])).collect();
            let b = bounds(&q);
            Some((f["kind"].as_str()?.to_string(), [(b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0]))
        })
        .collect();
    let b = BLOCK as f64;
    for comp in components(&rc) {
        let pts: Vec<[f64; 2]> = comp.iter().flat_map(|&i| {
            let (r, c) = rc[i];
            [[c as f64 * b, r as f64 * b], [(c + 1) as f64 * b, (r + 1) as f64 * b]]
        })
        .collect();
        let u = bounds(&pts);
        let centre = [(u[0] + u[2]) / 2.0, (u[1] + u[3]) / 2.0];
        let kind = fields.iter().min_by(|a, c| (a.1[0] - centre[0]).hypot(a.1[1] - centre[1]).total_cmp(&(c.1[0] - centre[0]).hypot(c.1[1] - centre[1]))).map(|f| f.0.as_str()).unwrap_or("signature");
        let class = match kind {
            "phone" => "phone_number",
            "id" => "id_number",
            _ => "signature",
        };
        if let Some(bb) = clip(u, w, hh) {
            boxes.push(json!({"class": class, "box": rbox(bb), "source": "blocks", "zone": "generic", "role": "unknown", "approximate": true}));
        }
    }
    Ok(())
}

fn found_boxes(list: &Value, source: &str, w: f64, hh: f64, boxes: &mut Vec<Value>) {
    for f in list.as_array().into_iter().flatten() {
        let b: Vec<f64> = f["box"].as_array().into_iter().flatten().filter_map(|x| x.as_f64()).collect();
        let class = match f["label"].as_str().unwrap_or("") {
            "phone_number" => "phone_number",
            "id_number" => "id_number",
            _ => "signature",
        };
        if b.len() == 4 {
            if let Some(c) = clip([b[0], b[1], b[2], b[3]], w, hh) {
                boxes.push(json!({"class": class, "box": rbox(c), "source": source, "role": "unknown"}));
            }
        }
    }
}

fn protected_boxes(list: &Value, w: f64, hh: f64, boxes: &mut Vec<Value>) {
    for p in list.as_array().into_iter().flatten() {
        let poly: Vec<[f64; 2]> = p["polygon"].as_array().into_iter().flatten().filter_map(|q| Some([q[0].as_f64()?, q[1].as_f64()?])).collect();
        if poly.is_empty() {
            continue;
        }
        let Some(b) = clip(bounds(&poly), w, hh) else { continue };
        if p["kind"] == "stamp" {
            boxes.push(json!({"class": "stamp", "box": rbox(b), "source": "stamp", "zone": "stamp", "circle": p["circle"], "polygon": rpoly(&poly), "found_by": p["found_by"]}));
        } else {
            boxes.push(json!({"class": "results_region", "box": rbox(b), "source": "protect", "zone": p["name"], "polygon": rpoly(&poly)}));
        }
    }
}

/// The rules' ink mask of a page: dark ink (print and pen) and coloured
/// ink (stamps), as PNG.
pub fn rules_mask(img: &Mat) -> Result<Vec<u8>> {
    let dark = util::bytes(&crate::stamps::dark_ink(&util::gray(img)?)?)?;
    let color = if img.channels() == 3 { util::bytes(&crate::stamps::color_ink(img)?)? } else { vec![0; dark.len()] };
    let v: Vec<u8> = dark.iter().zip(&color).map(|(&d, &c)| (if d != 0 { BIT_PRINT } else { 0 }) | (if c != 0 { BIT_STAMP } else { 0 })).collect();
    crate::io::encode(&util::from_bytes(img.rows(), img.cols(), &v)?, "png", 0)
}

/// Where the originals are: SHA-256 -> source (a file, or zip::member).
pub fn originals_by_hash(dir: &Path) -> Result<HashMap<String, String>> {
    let srcs = crate::io::list_inputs(&[dir.to_string_lossy().into_owned()])?;
    let hashed: Vec<(String, String)> = srcs.par_iter().filter_map(|s| crate::io::read_bytes(s).ok().map(|d| (crate::proof::sha256(&d), s.clone()))).collect();
    Ok(hashed.into_iter().collect())
}

/// One page to write: its source, page number, and how to label it.
struct Job {
    src: String,
    sha: String,
    k: usize,
    page_index: usize,
    manifest: Option<(PathBuf, String, Value)>,
    held: Option<Value>,
}

pub fn page_id(sha: &str, k: usize) -> String {
    format!("t-{}-p{:03}", &sha[..16.min(sha.len())], k)
}

fn label_page(j: &Job, templates: &[Template], masks: bool) -> Result<Option<PageOut>> {
    let data = crate::io::read_bytes(&j.src)?;
    let sha = crate::proof::sha256(&data);
    let (sp, _) = crate::io::source_page(&j.src, &data, j.k)?;
    let (w, hh) = (sp.pixels.cols() as f64, sp.pixels.rows() as f64);
    let (image, ext) = match (&sp.data, sp.kind) {
        (Some(d), "jpeg") => (d.clone(), "jpg"),
        _ => (crate::io::encode(&sp.pixels, "png", 0)?, "png"),
    };
    let stored_sha = match &sp.data {
        Some(d) => crate::proof::sha256(d),
        None => crate::proof::sha256(&util::bytes(&sp.pixels)?),
    };
    let id = page_id(&sha, j.k);
    let mut notes: Vec<String> = Vec::new();
    let mut boxes = Vec::new();
    let (meta, template, kind, held, reason, teacher, exhaustive): (Value, Option<String>, &str, bool, Value, Value, Vec<&str>) = match (&j.manifest, &j.held) {
        (Some((path, msha, m)), _) => {
            let mp = &m["pages"][j.page_index];
            if mp["source_image_sha256"].as_str() != Some(stored_sha.as_str()) {
                bail!("{}: page {} is not the image its manifest names", j.src, j.k);
            }
            let tname = mp["template"].as_str().unwrap_or("").to_string();
            let mut ex: Vec<&str> = Vec::new();
            match templates.iter().find(|t| t.name == tname) {
                Some(t) => {
                    if m["templates"][&tname].as_str().is_some_and(|f| f != t.fingerprint) {
                        notes.push(format!("template {} differs from the one the manifest names (fingerprint)", tname));
                    }
                    template_boxes(t, mp, w, hh, &mut boxes, &mut notes)?;
                    ex = TEACHER_EXHAUSTIVE.to_vec();
                }
                None if tname == "generic" => generic_boxes(mp, w, hh, &mut boxes)?,
                None => notes.push(format!("template {} not found under --templates: its fields are not labelled", tname)),
            }
            protected_boxes(&mp["protected"], w, hh, &mut boxes);
            found_boxes(&mp["vlm_found"], "vlm_found", w, hh, &mut boxes);
            found_boxes(&mp["spot_masks"], "spot_mask", w, hh, &mut boxes);
            let teacher = json!({"rfpii": m["tool"], "manifest": path.file_name().map(|f| f.to_string_lossy().into_owned()), "manifest_sha256": msha,
                                 "settings": m["settings"], "seed": m["seed"], "template_fingerprint": m["templates"][&tname],
                                 "turn": mp["orientation"]["turn"], "notes": mp["notes"]});
            (m["source"].clone(), Some(tname), "form", false, Value::Null, teacher, ex)
        }
        (None, Some(r)) => {
            let pr = &r["pages"][j.page_index];
            let t = pr["template"].as_str().filter(|s| !s.is_empty()).map(String::from);
            (r.clone(), t, "unknown", true, r["reason"].clone(), json!({"rfpii": crate::process::tool_info(), "report": "report.jsonl", "page_reason": pr["reason"]}), vec![])
        }
        _ => bail!("nothing to label"),
    };
    let path = meta["path"].as_str().or(meta["file"].as_str()).unwrap_or(&j.src).to_string();
    let form = meta["form"].as_str().map(String::from).or_else(|| template.as_ref().and_then(|t| templates.iter().find(|x| &x.name == t)).map(|t| t.form.clone()));
    let labels = json!({
        "schema": LABELS_SCHEMA,
        "id": id,
        "page": {
            "width": w as i64, "height": hh as i64, "kind": kind,
            "template": template, "form": form, "printing": template,
            "year": year(&meta, template.as_deref()), "capture": capture(&meta, &path),
            "turn": teacher["turn"].as_i64().unwrap_or(0),
            "source": {"sha256": sha, "page": j.k, "path": path, "image_sha256": stored_sha, "kind": sp.kind},
            "kiems": place(&meta), "held": held, "reason": reason,
        },
        "split_key": split_key(&meta, &path),
        "teacher": teacher,
        "exhaustive": exhaustive,
        "notes": notes,
        "boxes": boxes,
    });
    let mask = if masks { Some(rules_mask(&sp.pixels)?) } else { None };
    Ok(Some(PageOut { id, image, ext, mask, labels }))
}

/// Manifests under the paths given (files, or folders searched).
pub fn manifest_files(paths: &[PathBuf]) -> Vec<PathBuf> {
    let mut out = Vec::new();
    for p in paths {
        if p.is_dir() {
            out.extend(crate::annex::files_under(p, ".manifest.json"));
        } else {
            out.push(p.clone());
        }
    }
    out
}

/// `rfpii dataset build`.
pub fn run(o: &BuildOptions) -> Result<Value> {
    let templates = match &o.templates {
        Some(d) => crate::templates::load_templates(d)?,
        None => vec![],
    };
    let pool = rayon::ThreadPoolBuilder::new().num_threads(o.workers.max(1)).build()?;
    let by_hash = pool.install(|| originals_by_hash(&o.originals))?;
    let w = Writer::open(&o.out, "teacher", o.append)?;
    let mut jobs = Vec::new();
    let mut missing = 0;
    let mut published = std::collections::HashSet::new();
    for mpath in manifest_files(&o.manifests) {
        let bytes = std::fs::read(&mpath).with_context(|| format!("reading {}", mpath.display()))?;
        let m: Value = serde_json::from_slice(&bytes)?;
        if m["schema"] != crate::process::SCHEMA {
            continue;
        }
        let sha = m["source"]["sha256"].as_str().unwrap_or("").to_string();
        let Some(src) = by_hash.get(&sha) else {
            missing += 1;
            continue;
        };
        for (i, _) in m["pages"].as_array().into_iter().flatten().enumerate() {
            let k = m["source"]["page"].as_u64().map(|p| p as usize).unwrap_or(i + 1);
            published.insert((sha.clone(), k));
            jobs.push(Job { src: src.clone(), sha: sha.clone(), k, page_index: i, manifest: Some((mpath.clone(), super::sha256_hex(&bytes), m.clone())), held: None });
        }
    }
    let mut held = 0;
    if let Some(rv) = &o.review {
        let text = std::fs::read_to_string(rv.join("report.jsonl")).with_context(|| format!("reading {}/report.jsonl", rv.display()))?;
        let mut seen = std::collections::HashSet::new();
        for line in text.lines().filter(|l| !l.trim().is_empty()) {
            let r: Value = serde_json::from_str(line)?;
            if r["ok"] == true {
                continue;
            }
            let src = r["sha256"].as_str().and_then(|s| by_hash.get(s).cloned()).or_else(|| r["file"].as_str().filter(|f| Path::new(f.split(crate::io::ZIP_SEP).next().unwrap_or(f)).exists()).map(String::from));
            let Some(src) = src else {
                missing += 1;
                continue;
            };
            let data = crate::io::read_bytes(&src)?;
            let sha = crate::proof::sha256(&data);
            let pages: Vec<usize> = match r["page"].as_u64() {
                Some(k) => vec![k as usize],
                None => (1..=crate::io::page_count(&src, &data)?).collect(),
            };
            for (i, k) in pages.into_iter().enumerate() {
                if published.contains(&(sha.clone(), k)) || !seen.insert((sha.clone(), k)) {
                    continue;
                }
                held += 1;
                let index = if r["page"].is_u64() { 0 } else { i };
                jobs.push(Job { src: src.clone(), sha: sha.clone(), k, page_index: index, manifest: None, held: Some(r.clone()) });
            }
        }
    }
    jobs.retain(|j| !w.has(&page_id(&j.sha, j.k)));
    opencv::core::set_num_threads(1)?;
    let t0 = std::time::Instant::now();
    let out = o.out.clone();
    let results: Vec<Result<Option<Value>>> = pool.install(|| {
        jobs.par_iter()
            .map(|j| match label_page(j, &templates, o.masks)? {
                Some(p) => Ok(Some(super::write_page(&out, &p)?)),
                None => Ok(None),
            })
            .collect()
    });
    let mut lines = Vec::new();
    let mut failed = Vec::new();
    for (r, j) in results.into_iter().zip(&jobs) {
        match r {
            Ok(Some(l)) => lines.push(l),
            Ok(None) => {}
            Err(e) => failed.push(format!("{} page {}: {:#}", j.src, j.k, e)),
        }
    }
    let secs = t0.elapsed().as_secs_f64();
    let n = lines.len();
    let params = json!({"manifests": o.manifests, "originals": o.originals, "templates": o.templates, "review": o.review, "masks": o.masks,
                        "sources_missing": missing, "held_pages": held, "failed": failed});
    let notes = vec![
        "Teacher labels: what the pipeline masked, from its manifests. Personal data: keep on an encrypted local disk, for named people only, under the retention schedule (`rfpii retain`); never commit or share.".to_string(),
        "Field boxes are the pipeline's: placed by the page's alignment and drawn from the blacked blocks where there are any (a number's characters fitted from its hidden ones, a signature's box twice its band). Names are never labelled; boxes marked `approximate` come from blacked areas outside template zones.".to_string(),
        "Ink masks are the pipeline's ink rules: bit 1 is all dark ink (print and pen together), bit 4 coloured ink.".to_string(),
    ];
    let mut meta = w.finish(lines, params, true, if o.masks { "rules" } else { "none" }, notes)?;
    meta["this_run"] = json!({"pages": n, "failed": failed.len(), "seconds": (secs * 10.0).round() / 10.0, "pages_per_second": if secs > 0.0 { ((n as f64 / secs) * 100.0).round() / 100.0 } else { 0.0 }});
    Ok(meta)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slots_are_fitted_from_the_hidden_characters() {
        // 8 characters 20 px apart from x 100; characters 1, 4 and 5 hidden
        let runs = [[120.0, 140.0], [180.0, 220.0]];
        let (x0, pitch) = fit_slots(&runs, &[4, 1, 5], 8).unwrap();
        assert!((x0 - 100.0).abs() < 1e-9 && (pitch - 20.0).abs() < 1e-9, "{x0} {pitch}");
        // one run: its width over its characters
        let (x0, pitch) = fit_slots(&[[160.0, 200.0]], &[3, 4], 8).unwrap();
        assert!((x0 - 100.0).abs() < 1e-9 && (pitch - 20.0).abs() < 1e-9);
        assert!(fit_slots(&[[1.0, 2.0]], &[1, 5], 8).is_none());
        assert_eq!(runs_of(&[[0.0, 0.0, 8.0, 1.0], [8.0, 0.0, 16.0, 1.0], [40.0, 0.0, 48.0, 1.0]], 4.0), vec![[0.0, 16.0], [40.0, 48.0]]);
    }

    #[test]
    fn split_keys_come_from_kiems_else_the_folder() {
        let m = json!({"kiems": {"county": "007", "constituency": "032"}});
        assert_eq!(split_key(&m, "a/b.PDF"), "county:007/constituency:032");
        assert_eq!(split_key(&json!({}), "Garissa/Ijara/x.pdf"), "folder:Ijara");
        assert_eq!(split_key(&json!({}), "/data/Garissa.zip::x.pdf"), "folder:Garissa");
        assert_eq!(split_key(&json!({}), "/data/Garissa.zip::Ijara/x.pdf"), "folder:Ijara");
        assert_eq!(capture(&m, "x.PDF"), "kiems");
        assert_eq!(capture(&json!({}), "x.pdf"), "scan");
        assert_eq!(capture(&json!({}), "z.zip::x.jpg"), "photo");
    }

    #[test]
    fn blocks_join_into_areas() {
        let c = components(&[(0, 0), (0, 1), (5, 5), (6, 7), (20, 20)]);
        let mut sizes: Vec<usize> = c.iter().map(|g| g.len()).collect();
        sizes.sort();
        assert_eq!(sizes, vec![1, 2, 2]);
    }
}
