//! Synthetic forms with exact labels (`rfpii dataset synth`).
//!
//! Each sample starts from a template's reference image (a real blank
//! printing, handwriting erased) and writes on it, in template coordinates:
//! ID and phone numbers in the zones' cells, character by character in a
//! stroke font with its own height, slant, spacing and jitter; signatures as
//! splines with loops, some with a flourish leaving the cell; names, parties
//! and dates in the columns the templates don't list (`layout.rs`); round
//! stamps in violet, blue or red ink, sometimes over a signature or an
//! agent's number; comments, a signature or a number in the comments area;
//! lines struck through unused rows. Some samples are negatives: a results
//! page, a form nobody filled, a blank page, a page that is not a form.
//!
//! Then the page is photographed or scanned (`look.rs`). Pen, stamp ink and
//! print are drawn on separate layers, so the mask is exact; every label is
//! moved by the same homography as the ink. A sample depends only on its
//! seed: the same seed gives the same files, byte for byte.

use anyhow::{bail, Result};
use opencv::core::{Mat, Scalar};
use opencv::prelude::*;
use rand::{RngExt, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;
use serde_json::{json, Value};
use std::path::Path;

use super::draw::{Layers, StampSpec};
use super::glyphs::{self, Hand, Poly};
use super::layout::{self, Kind, Layout};
use super::look::{self, Look, H3};
use super::{r1, rbox, rpoly, PageOut, Writer, CLASSES, LABELS_SCHEMA};
use crate::templates::{Template, Zone};
use crate::util;

pub const GENERATOR: &str = "rfpii-synth/1";

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Profile {
    /// photos and scans, some turned, folded, blurred
    Mixed,
    /// light scans only: no turns, little perspective (for measuring the
    /// teacher where the pipeline is expected to read the page)
    Clean,
    /// photos only
    Photo,
}

impl Profile {
    pub fn parse(s: &str) -> Result<Profile> {
        Ok(match s {
            "mixed" => Profile::Mixed,
            "clean" => Profile::Clean,
            "photo" => Profile::Photo,
            _ => bail!("profile must be mixed, clean or photo"),
        })
    }
    pub fn name(&self) -> &'static str {
        match self {
            Profile::Mixed => "mixed",
            Profile::Clean => "clean",
            Profile::Photo => "photo",
        }
    }
}

#[derive(Clone, Debug)]
pub struct Options {
    pub seed: u64,
    pub count: usize,
    /// templates to draw on (none: all)
    pub only: Vec<String>,
    pub max_side: Option<u32>,
    pub profile: Profile,
    /// share of samples that are negatives (results pages, unfilled forms,
    /// blank pages, pages that are not forms)
    pub negatives: f64,
    /// also redact each page with the pipeline and label what still shows
    pub redacted: bool,
    pub workers: usize,
}

impl Options {
    pub fn to_json(&self, templates: &[String]) -> Value {
        json!({"generator": GENERATOR, "seed": self.seed, "count": self.count, "seeds": [self.seed, self.seed.wrapping_add(self.count as u64)],
               "templates": templates, "max_side": self.max_side, "profile": self.profile.name(), "negatives": self.negatives,
               "redacted": self.redacted})
    }
}

/// A label in canvas coordinates, before the page is photographed.
#[derive(Clone, Debug)]
pub struct Item {
    pub class: &'static str,
    pub zone: String,
    pub role: Option<&'static str>,
    /// x0, y0, x1, y1: the writing's box (its strokes, their width
    /// included), or the cell when nothing is written, or the region
    pub rect: [f64; 4],
    pub cell: Option<[f64; 4]>,
    pub empty: bool,
    /// a line struck through the (empty) cell's row
    pub struck: bool,
    /// each character's box, for numbers
    pub chars: Vec<[f64; 4]>,
    /// the ink drawn for it: one owner per character of a number; a
    /// signature's body, then its flourish if it has one; a name's one
    pub owners: Vec<u16>,
    pub rule: Option<&'static str>,
    pub circle: Option<[f64; 3]>,
    pub polygon: Option<Vec<[f64; 2]>>,
}

impl Item {
    fn region(class: &'static str, zone: &str, r: [f64; 4]) -> Item {
        Item { class, zone: zone.into(), role: None, rect: r, cell: None, empty: false, struck: false, chars: vec![], owners: vec![], rule: None, circle: None, polygon: None }
    }
}

/// The sample's random choices.
pub struct Gen {
    rng: ChaCha8Rng,
    next_owner: u16,
}

impl Gen {
    pub fn new(seed: u64) -> Gen {
        Gen { rng: ChaCha8Rng::seed_from_u64(seed), next_owner: 1 }
    }
    pub fn u(&mut self, a: f64, b: f64) -> f64 {
        a + (b - a) * self.rng.random::<f64>()
    }
    pub fn p(&mut self, p: f64) -> bool {
        self.rng.random::<f64>() < p
    }
    /// a whole number in a..=b
    pub fn int(&mut self, a: i64, b: i64) -> i64 {
        self.rng.random_range(a..=b)
    }
    pub fn pick<'a, T>(&mut self, s: &'a [T]) -> &'a T {
        &s[self.rng.random_range(0..s.len())]
    }
    /// in [-1, 1)
    pub fn noise(&mut self) -> f64 {
        2.0 * self.rng.random::<f64>() - 1.0
    }
    pub fn owner(&mut self) -> u16 {
        let o = self.next_owner;
        self.next_owner = self.next_owner.saturating_add(1);
        o
    }
}

const FIRST: &[&str] = &["JOHN", "MARY", "PETER", "GRACE", "JAMES", "FAITH", "DAVID", "MERCY", "JOSEPH", "ANN", "SAMUEL", "ESTHER", "DANIEL", "JANE", "PAUL", "RUTH", "MOSES", "LUCY", "HASSAN", "AMINA", "ALI", "FATUMA", "OMAR", "HALIMA", "KIPROP", "CHEBET", "OTIENO", "AKINYI", "MWANGI", "WANJIRU"];
const LAST: &[&str] = &["KAMAU", "OCHIENG", "WAFULA", "MUTUA", "KIPKEMBOI", "NJOROGE", "ODHIAMBO", "WANJALA", "MOHAMED", "ABDI", "CHERUIYOT", "KARIUKI", "NYAMBURA", "OMONDI", "KIBET", "MUSYOKA", "BARASA", "HUSSEIN", "LANGAT", "MAINA", "ADAN", "OWINO", "KILONZO", "ROTICH"];
const PARTIES: &[&str] = &["UDA", "ODM", "JUBILEE", "WIPER", "ANC", "FORD K", "KANU", "DAP K", "ROOTS", "INDEPENDENT", "AGANO"];
const COMMENTS: &[&str] = &["ALL AGENTS SIGNED", "NO DISPUTE", "COUNTING WAS FAIR", "AGENT REFUSED TO SIGN", "NO AGENT PRESENT", "RESULTS ANNOUNCED", "PROCESS PEACEFUL", "AGENTS SATISFIED", "NONE", "NIL"];
const SYLLABLES: &[&str] = &["KA", "MU", "NJI", "WA", "KI", "RA", "TO", "BO", "SE", "LU", "NDA", "GU", "YO", "MA", "TU", "RI", "KO", "LE", "NYA", "GI"];

fn person(g: &mut Gen) -> String {
    format!("{} {}", g.pick(FIRST), g.pick(LAST))
}

/// A made-up place name (syllables, so no real station is named).
fn place(g: &mut Gen) -> String {
    let n = g.int(2, 4);
    (0..n).map(|_| *g.pick(SYLLABLES)).collect::<String>()
}

fn digits(g: &mut Gen, n: usize) -> String {
    let mut s = String::new();
    for i in 0..n {
        let d = if i == 0 { g.int(1, 9) } else { g.int(0, 9) };
        s.push(char::from(b'0' + d as u8));
    }
    s
}

fn id_number(g: &mut Gen) -> String {
    let n = if g.p(0.15) { 7 } else { 8 };
    digits(g, n)
}

fn phone_number(g: &mut Gen) -> String {
    let lead = if g.p(0.85) { "07" } else { "01" };
    let rest: String = (0..8).map(|_| char::from(b'0' + g.int(0, 9) as u8)).collect();
    let s = format!("{}{}", lead, rest);
    if g.p(0.2) { format!("{} {} {}", &s[..4], &s[4..7], &s[7..]) } else { s }
}

fn date(g: &mut Gen) -> String {
    let d = g.int(8, 10);
    match g.int(0, 2) {
        0 => format!("{:02}/08/2022", d),
        1 => format!("{}-8-22", d),
        _ => format!("{:02}.08.2022", d),
    }
}

fn pen_color(g: &mut Gen) -> [u8; 3] {
    if g.p(0.65) {
        [g.int(110, 190) as u8, g.int(25, 80) as u8, g.int(0, 40) as u8]
    } else {
        let v = g.int(15, 55) as u8;
        [v, v, v + g.int(0, 10) as u8]
    }
}

/// A hand for writing `text` in a cell (x, y, w, h), fitted to its width.
fn hand_for(g: &mut Gen, text: &str, w: f64, h: f64) -> Hand {
    let k = if h > 60.0 { g.u(0.3, 0.7) } else { g.u(0.42, 0.8) };
    let height = h * k;
    let mut hand = Hand { height, width: height * g.u(0.5, 0.78), slant: g.u(-0.1, 0.35), gap: g.u(0.08, 0.45), jitter: g.u(0.01, 0.045), thick: 1 };
    hand.thick = ((height * g.u(0.06, 0.12)).round() as i32).max(1);
    let tw = glyphs::width(text, &hand);
    if tw > 0.94 * w {
        hand.width *= 0.94 * w / tw;
    }
    hand
}

/// Write `text` in `cell` (x, y, w, h); each character drawn by its own
/// owner when `per_char`, else one owner for all. Returns the characters'
/// boxes and the owners.
fn write_text(l: &mut Layers, g: &mut Gen, cell: [f64; 4], text: &str, color: [u8; 3], per_char: bool, track: bool) -> Result<(Vec<[f64; 4]>, Vec<u16>)> {
    let [x, y, w, h] = cell;
    let hand = hand_for(g, text, w, h);
    let tw = glyphs::width(text, &hand);
    let x0 = x + (w - tw).max(0.0) * g.u(0.03, 0.6) + hand.slant.min(0.0).abs() * hand.height;
    let mut base = y + h * g.u(0.72, 0.95);
    if g.p(0.1) {
        base -= hand.height * g.u(0.1, 0.4); // written high, off its line
    }
    let written = glyphs::write(text, x0, base, &hand, &mut || g.noise());
    let shared = if track && !per_char { g.owner() } else { 0 };
    let mut boxes = Vec::new();
    let mut owners = Vec::new();
    for c in &written {
        let o = if !track { 0 } else if per_char { g.owner() } else { shared };
        l.pen(&c.polys, hand.thick, color, o)?;
        boxes.push(c.bbox);
        if track && per_char {
            owners.push(o);
        }
    }
    if track && !per_char {
        owners.push(shared);
    }
    Ok((boxes, owners))
}

fn union(boxes: &[[f64; 4]]) -> [f64; 4] {
    let mut b = [f64::INFINITY, f64::INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY];
    for r in boxes {
        b[0] = b[0].min(r[0]);
        b[1] = b[1].min(r[1]);
        b[2] = b[2].max(r[2]);
        b[3] = b[3].max(r[3]);
    }
    b
}

/// Catmull-Rom spline through `p`, 8 points per span.
fn spline(p: &[[f64; 2]]) -> Poly {
    let n = p.len();
    if n < 3 {
        return p.to_vec();
    }
    let at = |i: isize| p[i.clamp(0, n as isize - 1) as usize];
    let mut out = Vec::new();
    for i in 0..n - 1 {
        let (p0, p1, p2, p3) = (at(i as isize - 1), at(i as isize), at(i as isize + 1), at(i as isize + 2));
        for k in 0..8 {
            let t = k as f64 / 8.0;
            let (t2, t3) = (t * t, t * t * t);
            let f = |a: f64, b: f64, c: f64, d: f64| 0.5 * ((2.0 * b) + (-a + c) * t + (2.0 * a - 5.0 * b + 4.0 * c - d) * t2 + (-a + 3.0 * b - 3.0 * c + d) * t3);
            out.push([f(p0[0], p1[0], p2[0], p3[0]), f(p0[1], p1[1], p2[1], p3[1])]);
        }
    }
    out.push(p[n - 1]);
    out
}

/// A signature in `cell` (x, y, w, h), in one of three hands: a running
/// line of loops and peaks, initials followed by a trailing stroke, or a
/// tight scrawl. Some start with a big loop, some end in a flourish that
/// leaves the cell, some have a dot or a dash. Returns its body's strokes,
/// its flourish's, and their width.
fn signature_strokes(g: &mut Gen, cell: [f64; 4], may_leave: bool) -> (Vec<Poly>, Vec<Poly>, i32) {
    let [x, y, w, h] = cell;
    let sw = w * g.u(0.3, 0.85);
    let sh = (h * g.u(0.6, 1.4)).min(1.2 * sw);
    let x0 = x + (w - sw) * g.u(0.0, 1.0);
    let cy = y + h * g.u(0.4, 0.65);
    let mut strokes: Vec<Poly> = Vec::new();
    let style = g.int(0, 9);
    let mut start = x0;
    if style >= 7 {
        // initials in a large slanted hand, then the line runs on
        let n = g.int(1, 3) as usize;
        let letters: String = (0..n).map(|_| char::from(b'A' + g.int(0, 25) as u8)).collect();
        let hand = Hand { height: sh * g.u(0.7, 1.0), width: sh * g.u(0.4, 0.6), slant: g.u(0.1, 0.5), gap: g.u(-0.1, 0.2), jitter: 0.06, thick: 1 };
        let base = cy + hand.height / 2.0;
        for c in glyphs::write(&letters, x0, base, &hand, &mut || g.noise()) {
            start = start.max(c.bbox[2]);
            strokes.extend(c.polys);
        }
    }
    let run = (x0 + sw - start).max(0.2 * sw);
    let n = if style >= 5 && style < 7 { g.int(10, 22) } else { g.int(5, 12) } as usize;
    let amp = if style >= 5 && style < 7 { sh * g.u(0.15, 0.35) } else { sh / 2.0 };
    let mut pts = Vec::new();
    let mut up = g.p(0.5);
    for i in 0..n {
        let t = i as f64 / (n - 1) as f64;
        let back = if i > 0 && i < n - 1 && g.p(0.25) { -g.u(0.03, 0.12) * run } else { 0.0 };
        if g.p(0.75) {
            up = !up;
        }
        let px = start + t * run + back;
        let py = cy + if up { -1.0 } else { 1.0 } * amp * g.u(0.2, 1.0);
        pts.push([px, py]);
        if g.p(0.15) {
            // a small loop
            let r = amp * g.u(0.2, 0.5);
            for k in 1..6 {
                let a = k as f64 / 6.0 * std::f64::consts::TAU;
                pts.push([px + r * a.sin(), py - r + r * a.cos()]);
            }
        }
    }
    if g.p(0.35) {
        // a big loop at the start
        let (rx, ry) = (sw * g.u(0.08, 0.18), sh * g.u(0.3, 0.5));
        let (lx, ly) = (start + rx, cy);
        let lp: Vec<[f64; 2]> = (0..=10).map(|k| {
            let a = k as f64 / 10.0 * std::f64::consts::TAU + 1.2;
            [lx + rx * a.cos(), ly + ry * a.sin()]
        })
        .collect();
        pts.splice(0..0, lp);
    }
    strokes.push(spline(&pts));
    let mut flourish = Vec::new();
    if may_leave && g.p(0.25) {
        // back under the signature, out past the cell
        let end = *pts.last().unwrap();
        let fl = vec![end, [end[0] + 0.15 * sw, end[1] + 0.3 * sh], [x0 + 0.5 * sw, cy + 0.7 * sh], [x0 - g.u(0.0, 0.25) * w, cy + sh * g.u(0.4, 0.9)]];
        flourish.push(spline(&fl));
    }
    if g.p(0.3) {
        let (dx, dy) = (x0 + sw * g.u(0.2, 0.9), cy - sh * g.u(0.5, 0.8));
        strokes.push(vec![[dx, dy], [dx + g.u(2.0, 8.0), dy + g.u(-1.0, 2.0)]]);
    }
    let thick = (((h / 18.0).sqrt() * g.u(1.4, 2.4)).round() as i32).clamp(1, 6);
    (strokes, flourish, thick)
}

/// Draw a signature: its body and its flourish each by their own owner
/// (the body first). Returns its box and owners.
fn draw_signature(l: &mut Layers, g: &mut Gen, cell: [f64; 4], may_leave: bool, pen: [u8; 3]) -> Result<([f64; 4], Vec<u16>)> {
    let (body, flourish, thick) = signature_strokes(g, cell, may_leave);
    let mut owners = vec![g.owner()];
    l.pen(&body, thick, pen, owners[0])?;
    if !flourish.is_empty() {
        owners.push(g.owner());
        l.pen(&flourish, thick, pen, owners[1])?;
    }
    let all: Vec<Poly> = body.into_iter().chain(flourish).collect();
    Ok((glyphs::bbox(&all, thick), owners))
}

/// A cell as a box (x0, y0, x1, y1).
fn cell_box(c: [f64; 4]) -> [f64; 4] {
    [c[0], c[1], c[0] + c[2], c[1] + c[3]]
}

fn number_item(zone: &str, role: &'static str, rule: &'static str, cell: [f64; 4], chars: Vec<[f64; 4]>, owners: Vec<u16>) -> Item {
    let class = if rule == "phone" { "phone_number" } else { "id_number" };
    Item { class, zone: zone.into(), role: Some(role), rect: union(&chars), cell: Some(cell_box(cell)), empty: false, struck: false, chars, owners, rule: Some(rule), circle: None, polygon: None }
}

fn empty_item(class: &'static str, zone: &str, role: &'static str, rule: Option<&'static str>, cell: [f64; 4]) -> Item {
    Item { class, zone: zone.into(), role: Some(role), rect: cell_box(cell), cell: Some(cell_box(cell)), empty: true, struck: false, chars: vec![], owners: vec![], rule, circle: None, polygon: None }
}

/// The class of a template zone, and the rule of a number zone.
pub fn zone_class(z: &Zone) -> Option<(&'static str, Option<&'static str>)> {
    match z.action.as_str() {
        "signature" => Some(("signature", None)),
        "partial" if z.rule == "phone" => Some(("phone_number", Some("phone"))),
        "partial" => Some(("id_number", Some("id"))),
        _ => {
            let n = z.name.to_ascii_lowercase();
            if n.contains("sign") {
                Some(("signature", None))
            } else if n.contains("phone") || n.contains("tel") {
                Some(("phone_number", Some("phone")))
            } else if n.contains("id") {
                Some(("id_number", Some("id")))
            } else {
                None
            }
        }
    }
}

/// The cell (x, y, w, h) a zone's writing goes in: its rect, moved onto
/// the printed column where the layout says the template's rect isn't.
fn zone_cell(z: &Zone, lay: Option<&Layout>, off: [f64; 2]) -> [f64; 4] {
    let mut r = z.rect;
    if let Some(&(_, x0, x1)) = lay.and_then(|l| l.columns.iter().find(|c| c.0 == z.key())) {
        r[0] = x0;
        r[2] = x1 - x0;
    }
    [r[0] + off[0], r[1] + off[1], r[2], r[3]]
}

/// Fill a form's fields. `unfilled`: nobody wrote on it.
#[allow(clippy::too_many_arguments)]
fn fill(t: &Template, lay: Option<&Layout>, off: [f64; 2], unfilled: bool, l: &mut Layers, g: &mut Gen, items: &mut Vec<Item>) -> Result<()> {
    let role_of = |name: &str| crate::masking::role_of(name);
    // the agents' rows: how many are filled, who writes each
    let n_rows = t.zones.iter().filter(|z| role_of(&z.name) == "agent").map(|z| z.row + 1).max().unwrap_or(0);
    let filled_rows = if unfilled || n_rows == 0 || g.p(0.08) {
        0
    } else if g.p(0.8) {
        g.int(1, n_rows.min(6) as i64) as usize
    } else {
        g.int(1, n_rows as i64) as usize
    };
    let row_filled: Vec<bool> = (0..n_rows).map(|r| r < filled_rows && !g.p(0.05)).collect();
    let row_pens: Vec<[u8; 3]> = (0..n_rows).map(|_| pen_color(g)).collect();
    let officer_pen = pen_color(g);
    let strike = n_rows > 0 && filled_rows < n_rows && !unfilled && g.p(0.35);
    let fill_p = |class: &str| match class {
        "id_number" => 0.9,
        "phone_number" => 0.85,
        _ => 0.85,
    };
    for z in &t.zones {
        let Some((class, rule)) = zone_class(z) else { continue };
        let role = role_of(&z.name);
        let cell = zone_cell(z, lay, off);
        let (written, pen) = if role == "agent" {
            (row_filled.get(z.row).copied().unwrap_or(false) && g.p(fill_p(class)), row_pens.get(z.row).copied().unwrap_or(officer_pen))
        } else {
            (!unfilled && g.p(0.88), officer_pen)
        };
        let role_s: &'static str = if role == "agent" { "agent" } else if role == "officer" { "officer" } else { "unknown" };
        if !written {
            let mut it = empty_item(class, &z.name, role_s, rule, cell);
            it.struck = strike && role == "agent" && z.row >= filled_rows;
            items.push(it);
            continue;
        }
        match class {
            "signature" => {
                let (b, owners) = draw_signature(l, g, cell, true, pen)?;
                items.push(Item { class, zone: z.name.clone(), role: Some(role_s), rect: b, cell: Some(cell_box(cell)), empty: false, struck: false, chars: vec![], owners, rule: None, circle: None, polygon: None });
            }
            _ => {
                let text = if rule == Some("phone") { phone_number(g) } else { id_number(g) };
                let (chars, owners) = write_text(l, g, cell, &text, pen, true, true)?;
                items.push(number_item(&z.name, role_s, rule.unwrap_or("id"), cell, chars, owners));
            }
        }
    }
    // names, parties, dates and counts, where the layout says
    if let Some(lay) = lay {
        for c in lay.cells {
            for (r, rect) in c.rows().into_iter().enumerate() {
                let cell = [rect[0] + off[0], rect[1] + off[1], rect[2], rect[3]];
                let (go, pen) = if c.role == "agent" {
                    let p = match c.kind {
                        Kind::Name => 0.95,
                        Kind::Party => 0.7,
                        _ => 0.5,
                    };
                    (row_filled.get(r).copied().unwrap_or(false) && g.p(p), row_pens.get(r).copied().unwrap_or(officer_pen))
                } else {
                    let p = if c.kind == Kind::Count { 0.6 } else { 0.85 };
                    (!unfilled && g.p(p), officer_pen)
                };
                let zone = c.name(r);
                if !go {
                    if c.kind == Kind::Name {
                        let mut it = empty_item("name", &zone, c.role, None, cell);
                        it.struck = strike && c.role == "agent" && r >= filled_rows;
                        items.push(it);
                    }
                    continue;
                }
                match c.kind {
                    Kind::Name => {
                        let name = person(g);
                        let (rect, owners) = if g.p(0.3) {
                            // cursive, in mixed case
                            let mixed: String = name.split(' ').map(|w| format!("{}{}", &w[..1], w[1..].to_lowercase())).collect::<Vec<_>>().join(" ");
                            let h = cell[3] * g.u(0.45, 0.75);
                            let mut scale = h / 22.0;
                            let mut base = 0;
                            let thick = ((h / 14.0).round() as i32).max(1);
                            let sz = opencv::imgproc::get_text_size(&mixed, opencv::imgproc::FONT_HERSHEY_SCRIPT_SIMPLEX, scale, thick, &mut base)?;
                            if sz.width as f64 > 0.94 * cell[2] {
                                scale *= 0.94 * cell[2] / sz.width as f64;
                            }
                            let o = g.owner();
                            let x = cell[0] + g.u(0.02, 0.15) * cell[2];
                            let y = cell[1] + cell[3] * g.u(0.68, 0.85);
                            (l.pen_script(&mixed, x, y, scale, thick, pen, o)?, vec![o])
                        } else {
                            let (boxes, owners) = write_text(l, g, cell, &name, pen, false, true)?;
                            (union(&boxes), owners)
                        };
                        items.push(Item { class: "name", zone, role: Some(c.role), rect, cell: Some(cell_box(cell)), empty: false, struck: false, chars: vec![], owners, rule: None, circle: None, polygon: None });
                    }
                    Kind::Party => {
                        let p = *g.pick(PARTIES);
                        write_text(l, g, cell, p, pen, false, false)?;
                    }
                    Kind::Date => {
                        let d = date(g);
                        write_text(l, g, cell, &d, pen, false, false)?;
                    }
                    Kind::Count => {
                        let len = g.int(2, 3) as usize;
                        let n = digits(g, len);
                        write_text(l, g, cell, &n, pen, false, false)?;
                    }
                }
            }
        }
    }
    if strike {
        // a line, or a Z, through the unused rows of the agents table
        let rows: Vec<&Zone> = t.zones.iter().filter(|z| role_of(&z.name) == "agent").collect();
        let top = rows.iter().filter(|z| z.row == filled_rows).map(|z| z.rect[1]).fold(f64::INFINITY, f64::min) + off[1];
        let bot = rows.iter().map(|z| z.rect[1] + z.rect[3]).fold(f64::NEG_INFINITY, f64::max) + off[1];
        if let Some((_, s)) = t.structure.iter().find(|(n, _)| n == "agents") {
            let (x0, x1) = (s[0] + off[0] + g.u(0.0, 0.1) * s[2], s[0] + off[0] + s[2] * g.u(0.85, 1.0));
            let line: Poly = if g.p(0.3) { vec![[x0, top], [x1, top], [x0, bot], [x1, bot]] } else { vec![[x0, bot], [x0 + (x1 - x0) * g.u(0.3, 0.6), top], [x1, bot]] };
            if top.is_finite() && bot > top {
                l.pen(&[line], g.int(2, 4) as i32, officer_pen, 0)?;
            }
        }
    }
    // the comments: words, and sometimes a signature, a number or a name
    for (name, r) in &t.free_text {
        if unfilled {
            continue;
        }
        let (x, y, w, h) = (r[0] + off[0], r[1] + off[1], r[2], r[3]);
        let line_h = 52.0;
        let mut line = 0.0;
        let mut next = |g: &mut Gen| -> Option<[f64; 4]> {
            line += 1.0;
            let yy = y + 20.0 + (line - 1.0) * line_h * g.u(1.0, 1.25);
            (yy + line_h < y + h).then(|| [x + w * g.u(0.02, 0.2), yy, w * g.u(0.4, 0.75), line_h])
        };
        if g.p(0.6) {
            for _ in 0..g.int(1, 3) {
                if let Some(c) = next(g) {
                    let text = *g.pick(COMMENTS);
                    write_text(l, g, c, text, officer_pen, false, false)?;
                }
            }
        }
        if g.p(0.2) {
            if let Some(c) = next(g) {
                let who = person(g);
                let (boxes, owners) = write_text(l, g, [c[0], c[1], c[2] * 0.6, c[3]], &who, officer_pen, false, true)?;
                items.push(Item { class: "name", zone: format!("{}:name", name), role: Some("officer"), rect: union(&boxes), cell: None, empty: false, struck: false, chars: vec![], owners, rule: None, circle: None, polygon: None });
            }
        }
        if g.p(0.15) {
            if let Some(c) = next(g) {
                let phone = g.p(0.5);
                let text = if phone { phone_number(g) } else { id_number(g) };
                let (chars, owners) = write_text(l, g, [c[0], c[1], c[2] * 0.45, c[3]], &text, officer_pen, true, true)?;
                let mut it = number_item(&format!("{}:number", name), "officer", if phone { "phone" } else { "id" }, c, chars, owners);
                it.cell = None;
                items.push(it);
            }
        }
        if g.p(0.35) {
            if let Some(c) = next(g) {
                let cell = [c[0] + c[2] * g.u(0.3, 0.6), c[1], w * g.u(0.15, 0.25), line_h * 1.3];
                let (b, owners) = draw_signature(l, g, cell, false, officer_pen)?;
                items.push(Item { class: "signature", zone: format!("{}:signature", name), role: Some("officer"), rect: b, cell: None, empty: false, struck: false, chars: vec![], owners, rule: None, circle: None, polygon: None });
            }
        }
    }
    Ok(())
}

/// Stamps on a form: usually where the layout says, sometimes over an
/// agent's ID or phone number (pen under the stamp).
fn stamps(t: &Template, lay: Option<&Layout>, off: [f64; 2], l: &mut Layers, g: &mut Gen, items: &mut Vec<Item>) -> Result<()> {
    let n = if g.p(0.2) { 0 } else if g.p(0.88) { 1 } else { 2 };
    let k = t.page_long_side.unwrap_or(t.image.rows().max(t.image.cols()) as f64) / 2340.0;
    for _ in 0..n {
        let numbers: Vec<[f64; 4]> = items.iter().filter(|i| (i.class == "id_number" || i.class == "phone_number") && !i.empty && i.role == Some("agent")).map(|i| i.rect).collect();
        let (cx, cy) = if !numbers.is_empty() && g.p(0.25) {
            let b = *g.pick(&numbers);
            ((b[0] + b[2]) / 2.0 + g.u(-30.0, 30.0) * k, (b[1] + b[3]) / 2.0 + g.u(-20.0, 20.0) * k)
        } else if let (Some(lay), true) = (lay, g.p(0.85)) {
            let s = *g.pick(lay.stamp_spots);
            (s[0] + off[0] + g.u(-50.0, 50.0) * k, s[1] + off[1] + g.u(-40.0, 40.0) * k)
        } else {
            let (w, h) = (t.image.cols() as f64, t.image.rows() as f64);
            (off[0] + w * g.u(0.15, 0.85), off[1] + h * g.u(0.5, 0.9))
        };
        let r = g.u(80.0, 115.0) * k;
        let color = match g.int(0, 9) {
            0..=4 => [g.int(140, 200) as u8, g.int(30, 70) as u8, g.int(100, 150) as u8],
            5..=7 => [g.int(150, 200) as u8, g.int(60, 110) as u8, g.int(10, 50) as u8],
            _ => [g.int(30, 70) as u8, g.int(30, 70) as u8, g.int(170, 210) as u8],
        };
        let title = lay.map(|l| l.stamp_title).unwrap_or("PRESIDING OFFICER");
        let spec = StampSpec {
            cx,
            cy,
            r,
            color,
            opacity: g.u(0.5, 0.9),
            top: title.to_string(),
            bottom: format!("{} PRIMARY", place(g)),
            middle: vec!["I.E.B.C.".into(), format!("{:02} AUG 2022", g.int(8, 10))],
        };
        let mut noise_src = ChaCha8Rng::seed_from_u64(g.rng.random::<u64>());
        l.stamp(&spec, &mut || 2.0 * noise_src.random::<f64>() - 1.0)?;
        let poly = spec.polygon();
        let mut it = Item::region("stamp", "stamp", super::bounds(&poly));
        it.circle = Some([cx, cy, r]);
        it.polygon = Some(poly);
        items.push(it);
    }
    Ok(())
}

/// A typed results table (a Form 34B's rows above its declaration block,
/// or below a results header): print, and a results region.
fn typed_table(l: &mut Layers, g: &mut Gen, r: [f64; 4], k: f64) -> Result<Item> {
    let [x, y, w, h] = r;
    let row_h = 46.0 * k;
    let rows = ((h / row_h).floor() as usize).clamp(1, 60);
    let thick = ((2.0 * k).round() as i32).max(1);
    let cols = [0.0, 0.3, 0.45, 0.57, 0.69, 0.81, 1.0];
    for i in 0..=rows {
        let yy = y + i as f64 * row_h;
        l.rule([x, yy], [x + w, yy], thick)?;
    }
    for c in cols {
        l.rule([x + c * w, y], [x + c * w, y + rows as f64 * row_h], thick)?;
    }
    let scale = 0.9 * k;
    for i in 0..rows {
        let base = (y + (i as f64 + 0.75) * row_h) as i32;
        let code = format!("{:03}{:03}{:04}{:03}{:02}", g.int(1, 47), g.int(1, 290), g.int(1, 1500), g.int(1, 999), g.int(1, 9));
        l.typed(&code, (x + 0.01 * w) as i32, base, scale, thick)?;
        let st = format!("{} PRIMARY", place(g));
        l.typed(&st[..st.len().min(14)], (x + 0.31 * w) as i32, base, scale * 0.8, thick)?;
        for c in 2..6 {
            let v = g.int(0, 700).to_string();
            l.typed(&v, (x + (cols[c] + 0.02) * w) as i32, base, scale, thick)?;
        }
    }
    Ok(Item::region("results_region", "typed_results", [x, y, x + w, y + rows as f64 * row_h]))
}

/// The paper's grey level in a reference (its 90th percentile).
fn paper_level(img: &Mat) -> Result<u8> {
    let mut hist = [0usize; 256];
    for &v in util::bytes(img)?.iter() {
        hist[v as usize] += 1;
    }
    let total: usize = hist.iter().sum();
    let mut acc = 0;
    for (v, &n) in hist.iter().enumerate() {
        acc += n;
        if acc * 10 >= total * 9 {
            return Ok(v as u8);
        }
    }
    Ok(240)
}

/// What one sample is.
pub enum PagePlan {
    Form { template: usize, unfilled: bool },
    Results { template: usize },
    Blank,
    NotForm,
}

/// The templates a sample draws on: those with fields, and results pages.
pub struct Ctx {
    pub templates: Vec<Template>,
    pub fielded: Vec<usize>,
    pub results: Vec<usize>,
    pub opts: Options,
    /// set for redacted samples: the pipeline's settings
    pub redact: Option<super::redacted::Redactor>,
}

impl Ctx {
    pub fn new(dir: &Path, opts: Options) -> Result<Ctx> {
        let mut templates = crate::templates::load_templates(dir)?;
        if !opts.only.is_empty() {
            for n in &opts.only {
                if !templates.iter().any(|t| &t.name == n) {
                    bail!("no template {} under {}", n, dir.display());
                }
            }
        }
        let chosen = |t: &Template| opts.only.is_empty() || opts.only.contains(&t.name);
        let fielded: Vec<usize> = (0..templates.len()).filter(|&i| chosen(&templates[i]) && templates[i].zones.iter().any(|z| zone_class(z).is_some())).collect();
        let results: Vec<usize> = (0..templates.len()).filter(|&i| chosen(&templates[i]) && templates[i].zones.is_empty()).collect();
        if fielded.is_empty() && results.is_empty() {
            bail!("no templates to draw on");
        }
        let redact = if opts.redacted {
            for t in templates.iter_mut() {
                crate::align::prepare(t, &crate::align::AlignSettings::default())?;
            }
            Some(super::redacted::Redactor::default())
        } else {
            None
        };
        Ok(Ctx { templates, fielded, results, opts, redact })
    }

    pub fn names(&self) -> Vec<String> {
        self.fielded.iter().chain(&self.results).map(|&i| self.templates[i].name.clone()).collect()
    }

    fn plan(&self, g: &mut Gen) -> PagePlan {
        if self.fielded.is_empty() || g.p(self.opts.negatives) {
            let r = g.u(0.0, 1.0);
            if (r < 0.45 || self.fielded.is_empty()) && !self.results.is_empty() {
                return PagePlan::Results { template: *g.pick(&self.results) };
            }
            if r < 0.65 && !self.fielded.is_empty() {
                return PagePlan::Form { template: *g.pick(&self.fielded), unfilled: true };
            }
            return if r < 0.85 { PagePlan::Blank } else { PagePlan::NotForm };
        }
        PagePlan::Form { template: *g.pick(&self.fielded), unfilled: false }
    }
}

/// A sample's page before it is photographed: the canvas layers, its
/// labels and what it is.
pub struct Drawn {
    pub layers: Layers,
    pub items: Vec<Item>,
    pub kind: &'static str,
    pub template: Option<usize>,
    pub offset: [f64; 2],
}

fn choose_look(g: &mut Gen, profile: Profile) -> Look {
    let photo = match profile {
        Profile::Clean => false,
        Profile::Photo => true,
        Profile::Mixed => g.p(0.4),
    };
    let clean = profile == Profile::Clean;
    let quarter_turns = if clean || !g.p(0.12) { 0 } else { g.int(1, 3) as i32 };
    let fold = (!clean && g.p(0.25)).then(|| {
        let horizontal = g.p(0.7);
        let ang = if horizontal { g.u(-0.08, 0.08) } else { std::f64::consts::FRAC_PI_2 + g.u(-0.08, 0.08) };
        [g.u(0.3, 0.7), g.u(0.3, 0.7), ang, g.u(0.03, 0.12), g.u(0.01, 0.05)]
    });
    let bg = if photo { [g.int(30, 120) as u8, g.int(40, 110) as u8, g.int(50, 130) as u8] } else { [235, 235, 235] };
    Look {
        photo,
        angle: if photo { g.u(-4.0, 4.0) } else if clean { g.u(-1.0, 1.0) } else { g.u(-2.0, 2.0) },
        persp: if photo { g.u(0.0, 0.06) } else if clean { g.u(0.0, 0.004) } else { g.u(0.0, 0.012) },
        scale: if clean { g.u(0.9, 1.0) } else { g.u(0.75, 1.05) },
        margin: if photo { g.u(0.02, 0.08) } else { g.u(0.0, 0.02) },
        quarter_turns,
        blur: if g.p(if clean { 0.3 } else { 0.5 }) { g.u(0.3, if clean { 0.8 } else { 1.3 }) } else { 0.0 },
        noise: g.u(1.5, if clean { 3.5 } else { 6.0 }),
        quality: g.int(if clean { 80 } else { 60 }, 95) as i32,
        fold,
        light: if photo { [g.u(0.0, std::f64::consts::TAU), g.u(0.75, 0.92), g.u(0.95, 1.05)] } else { [g.u(0.0, std::f64::consts::TAU), g.u(0.94, 0.99), g.u(0.99, 1.02)] },
        background: bg,
        tint: [g.u(0.93, 1.0), g.u(0.96, 1.0), g.u(0.97, 1.0)],
        pen_opacity: g.u(0.8, 1.0),
    }
}

/// Draw a sample's page (before it is photographed).
pub fn draw(ctx: &Ctx, g: &mut Gen, owner: bool) -> Result<Drawn> {
    let plan = ctx.plan(g);
    match plan {
        PagePlan::Blank | PagePlan::NotForm => {
            let (w, h) = (g.int(1400, 1800) as i32, g.int(2000, 2500) as i32);
            let mut l = Layers::new(w, h, g.int(225, 250) as u8, owner)?;
            let not_form = matches!(plan, PagePlan::NotForm);
            if not_form {
                // a typed letter: lines of text, not a form
                let lines = g.int(10, 40);
                for i in 0..lines {
                    let words: Vec<String> = (0..g.int(3, 9)).map(|_| place(g)).collect();
                    l.typed(&words.join(" "), 120, 200 + (i as i32) * 48, g.u(0.9, 1.2), 2)?;
                }
            }
            Ok(Drawn { layers: l, items: vec![], kind: if not_form { "not_form" } else { "blank" }, template: None, offset: [0.0, 0.0] })
        }
        PagePlan::Results { template } | PagePlan::Form { template, .. } => {
            let t = &ctx.templates[template];
            let (rw, rh) = (t.image.cols(), t.image.rows());
            let long = t.page_long_side.unwrap_or(0.0);
            let k = if long > 0.0 { long / 3308.0 } else { 1.0 };
            let fielded = matches!(plan, PagePlan::Form { .. });
            let paper = paper_level(&t.image)?;
            // a block or a results header is part of a page: put it on one
            let (w, h, off) = if long > rw.max(rh) as f64 * 1.05 {
                let h = long.round() as i32;
                let w = ((long / std::f64::consts::SQRT_2).round() as i32).max(rw + 80);
                let x = ((w - rw) as f64 * g.u(0.3, 0.7)).round();
                let y = if fielded { (30.0 + (h - rh - 60) as f64 * g.u(0.25, 1.0)).round() } else { g.u(40.0, 160.0).round() };
                (w, h, [x, y])
            } else {
                (rw, rh, [0.0, 0.0])
            };
            let mut l = Layers::new(w, h, paper, owner)?;
            l.place(&t.image, off[0] as i32, off[1] as i32)?;
            let mut items = Vec::new();
            // the typed results table on the rest of the page
            if long > 0.0 {
                let table = if fielded {
                    (off[1] > 400.0 * k).then(|| [off[0], 100.0 * k, rw as f64, off[1] - 160.0 * k])
                } else {
                    let top = off[1] + rh as f64 + 10.0;
                    (h as f64 - top > 400.0 * k).then(|| [off[0], top, rw as f64, h as f64 - top - 150.0 * k])
                };
                if let Some(r) = table {
                    items.push(typed_table(&mut l, g, r, k)?);
                }
            }
            for (name, r) in &t.protect {
                items.push(Item::region("results_region", name, [r[0] + off[0], r[1] + off[1], r[0] + r[2] + off[0], r[1] + r[3] + off[1]]));
            }
            if !fielded {
                return Ok(Drawn { layers: l, items, kind: "form", template: Some(template), offset: off });
            }
            let unfilled = matches!(plan, PagePlan::Form { unfilled: true, .. });
            let lay = layout::layout(&t.name);
            for (name, r) in &t.structure {
                if name == "agents" {
                    items.push(Item::region("agents_table", name, [r[0] + off[0], r[1] + off[1], r[0] + r[2] + off[0], r[1] + r[3] + off[1]]));
                }
            }
            for (name, r) in &t.free_text {
                items.push(Item::region("comments", name, [r[0] + off[0], r[1] + off[1], r[0] + r[2] + off[0], r[1] + r[3] + off[1]]));
            }
            for r in lay.map(|l| l.declaration).unwrap_or(&[]) {
                items.push(Item::region("declaration", "declaration", [r[0] + off[0], r[1] + off[1], r[0] + r[2] + off[0], r[1] + r[3] + off[1]]));
            }
            // the stamp printed on the reference: its ink is print, but it is a stamp
            for s in lay.map(|l| l.reference_stamps).unwrap_or(&[]) {
                let spec = StampSpec { cx: s[0] + off[0], cy: s[1] + off[1], r: s[2], color: [0, 0, 0], opacity: 0.0, top: String::new(), bottom: String::new(), middle: vec![] };
                let poly = spec.polygon();
                let mut it = Item::region("stamp", "reference_stamp", super::bounds(&poly));
                it.circle = Some([spec.cx, spec.cy, spec.r]);
                it.polygon = Some(poly);
                items.push(it);
            }
            fill(t, lay, off, unfilled, &mut l, g, &mut items)?;
            if !unfilled || g.p(0.3) {
                stamps(t, lay, off, &mut l, g, &mut items)?;
            }
            Ok(Drawn { layers: l, items, kind: "form", template: Some(template), offset: off })
        }
    }
}

/// The page photographed: image, mask, owners (if kept), and the
/// homography from the canvas to the image.
pub struct Shot {
    pub image: Mat,
    pub bits: Mat,
    pub owner: Option<Mat>,
    pub h: H3,
}

pub fn photograph(d: &Drawn, look: &Look, g: &mut Gen) -> Result<Shot> {
    let page = d.layers.compose(look.tint, look.pen_opacity)?;
    let bits = d.layers.bits()?;
    let jitter: [f64; 8] = std::array::from_fn(|_| g.noise());
    let (hm, ow, oh) = look::geometry(d.layers.w as f64, d.layers.h as f64, look, &jitter)?;
    let bgc = Scalar::new(look.background[0] as f64, look.background[1] as f64, look.background[2] as f64, 0.0);
    let mut img = look::warp(&page, &hm, ow, oh, false, bgc)?;
    let mut bw = look::warp(&bits, &hm, ow, oh, true, Scalar::all(0.0))?;
    let mut owner = match &d.layers.owner {
        Some(o) => Some(look::warp(o, &hm, ow, oh, true, Scalar::all(0.0))?),
        None => None,
    };
    look::photometric(&mut img, look, g.rng.random::<u64>())?;
    let (t, _, _) = look::turn_matrix(look.quarter_turns, ow, oh);
    if look.quarter_turns != 0 {
        img = look::turn(&img, look.quarter_turns)?;
        bw = look::turn(&bw, look.quarter_turns)?;
        if let Some(o) = owner.as_ref() {
            owner = Some(look::turn(o, look.quarter_turns)?);
        }
    }
    Ok(Shot { image: img, bits: bw, owner, h: look::mul(&t, &hm) })
}

/// A canvas box (x0, y0, x1, y1) on the image: the box round its corners
/// moved by `h`, a pixel wider each way (the mask moves by nearest
/// neighbour), clipped.
pub fn map_box(h: &H3, b: [f64; 4], w: f64, hh: f64) -> Option<[f64; 4]> {
    let c: Vec<[f64; 2]> = [[b[0], b[1]], [b[2], b[1]], [b[2], b[3]], [b[0], b[3]]].iter().map(|&p| util::project(h, p)).collect();
    let r = super::bounds(&c);
    super::clip([r[0] - 1.0, r[1] - 1.0, r[2] + 1.0, r[3] + 1.0], w, hh)
}

pub fn map_poly(h: &H3, p: &[[f64; 2]]) -> Vec<[f64; 2]> {
    p.iter().map(|&q| util::project(h, q)).collect()
}

fn quad(b: [f64; 4]) -> Vec<[f64; 2]> {
    vec![[b[0], b[1]], [b[2], b[1]], [b[2], b[3]], [b[0], b[3]]]
}

/// An item's label on the image (`h`: canvas to image, w x hh).
pub fn item_label(it: &Item, h: &H3, w: f64, hh: f64, stamps: &[[f64; 3]], turned: bool) -> Option<Value> {
    let b = map_box(h, it.rect, w, hh)?;
    let mut v = json!({"class": it.class, "box": rbox(b), "source": "synthetic", "zone": it.zone});
    if let Some(r) = it.role {
        v["role"] = json!(r);
    }
    if let Some(r) = it.rule {
        v["rule"] = json!(r);
    }
    match (&it.polygon, it.circle) {
        (Some(p), Some(c)) => {
            let mp = map_poly(h, p);
            let cc = util::project(h, [c[0], c[1]]);
            let r = mp.iter().map(|q| (q[0] - cc[0]).hypot(q[1] - cc[1])).sum::<f64>() / mp.len() as f64;
            v["circle"] = json!([r1(cc[0]), r1(cc[1]), r1(r)]);
            v["polygon"] = rpoly(&mp);
        }
        _ => {
            v["polygon"] = rpoly(&map_poly(h, &quad(it.rect)));
        }
    }
    if let Some(c) = it.cell {
        if let Some(cb) = map_box(h, c, w, hh) {
            v["cell"] = rbox(cb);
        }
    }
    if it.empty {
        v["empty"] = json!(true);
    }
    if it.struck {
        v["struck"] = json!(true);
    }
    if !it.chars.is_empty() {
        let cb: Vec<[f64; 4]> = it.chars.iter().filter_map(|&c| map_box(h, c, w, hh)).collect();
        v["char_boxes"] = Value::Array(cb.iter().map(|&c| rbox(c)).collect());
        if !turned {
            // a character's x-span, inside the field's box
            v["slots"] = Value::Array(cb.iter().map(|c| json!([r1(c[0].max(b[0])), r1(c[2].min(b[2]))])).collect());
        }
    }
    if it.circle.is_none() && it.class != "results_region" && it.class != "agents_table" && it.class != "declaration" && it.class != "comments" {
        let [x0, y0, x1, y1] = it.rect;
        let under = stamps.iter().any(|s| {
            let (nx, ny) = (s[0].clamp(x0, x1), s[1].clamp(y0, y1));
            (nx - s[0]).hypot(ny - s[1]) < s[2] && !it.empty
        });
        if under {
            v["under_stamp"] = json!(true);
        }
    }
    Some(v)
}

/// The year a template is for (the four digits in its name), if any.
pub fn template_year(name: &str) -> Option<i64> {
    regex::Regex::new(r"(19|20)\d\d").ok()?.find(name).and_then(|m| m.as_str().parse().ok())
}

fn round_h(h: &H3) -> Value {
    json!(h.iter().map(|r| r.iter().map(|&x| (x * 1e8).round() / 1e8).collect::<Vec<_>>()).collect::<Vec<_>>())
}

pub fn sample_id(seed: u64) -> String {
    format!("syn-{:010}", seed)
}

/// One sample: its page, mask and labels; None when the redacting
/// pipeline held the page (redacted samples only).
pub fn sample(ctx: &Ctx, seed: u64) -> Result<Option<PageOut>> {
    let mut g = Gen::new(seed);
    let redacted = ctx.redact.is_some();
    let d = draw(ctx, &mut g, redacted)?;
    let look = choose_look(&mut g, ctx.opts.profile);
    let shot = photograph(&d, &look, &mut g)?;
    let (w, h) = (shot.image.cols(), shot.image.rows());
    let jpeg = crate::io::encode(&shot.image, "jpg", look.quality)?;
    let t = d.template.map(|i| &ctx.templates[i]);
    let stamps: Vec<[f64; 3]> = d.items.iter().filter_map(|i| i.circle).collect();
    let turned = look.quarter_turns != 0;
    let label_all = |hh: &H3, w: f64, h: f64| -> Vec<(usize, Value)> {
        d.items.iter().enumerate().filter_map(|(i, it)| item_label(it, hh, w, h, &stamps, turned).map(|v| (i, v))).collect()
    };
    let mut boxes = label_all(&shot.h, w as f64, h as f64);
    let mut labels = json!({
        "schema": LABELS_SCHEMA,
        "id": sample_id(seed),
        "page": {
            "width": w, "height": h, "kind": d.kind,
            "template": t.map(|t| t.name.clone()), "form": t.map(|t| t.form.clone()), "printing": t.map(|t| t.name.clone()),
            "year": t.and_then(|t| template_year(&t.name)),
            "capture": if look.photo { "synthetic_photo" } else { "synthetic_scan" },
            "turn": 90 * look.quarter_turns, "source": Value::Null, "kiems": Value::Null, "held": false, "reason": Value::Null,
        },
        "split_key": format!("seed:{}", seed),
        "synthetic": {"seed": seed, "generator": GENERATOR, "profile": ctx.opts.profile.name(), "look": look.to_json(),
                      "canvas": [d.layers.w, d.layers.h], "template_offset": d.offset, "H": round_h(&shot.h)},
        "exhaustive": CLASSES,
    });
    let (mut image_bytes, mut bits) = (jpeg, shot.bits.try_clone()?);
    if let Some(rd) = &ctx.redact {
        match super::redacted::redact(rd, &ctx.templates, &image_bytes, &shot, look.quarter_turns, &d.items, &mut boxes, &mut g, seed)? {
            None => return Ok(None),
            Some(r) => {
                image_bytes = r.jpeg;
                bits = r.bits;
                labels["redaction"] = r.record;
            }
        }
    }
    let (nw, nh) = look::fit(w, h, ctx.opts.max_side);
    if (nw, nh) != (w, h) {
        let s = look::scale_matrix(w, h, nw, nh);
        let img = crate::io::decode_bytes(&image_bytes)?;
        let small = util::resize(&img, nw, nh, opencv::imgproc::INTER_AREA)?;
        image_bytes = crate::io::encode(&small, "jpg", look.quality)?;
        bits = look::shrink_bits(&bits, nw, nh)?;
        let hs = look::mul(&s, &shot.h);
        let status: std::collections::HashMap<usize, Value> = boxes.iter().filter(|(_, b)| b["redaction"].is_object()).map(|(i, b)| (*i, b["redaction"].clone())).collect();
        boxes = label_all(&hs, nw as f64, nh as f64);
        for (i, b) in boxes.iter_mut() {
            if let Some(r) = status.get(i) {
                b["redaction"] = r.clone();
            }
        }
        labels["page"]["width"] = json!(nw);
        labels["page"]["height"] = json!(nh);
        labels["synthetic"]["H"] = round_h(&hs);
        labels["synthetic"]["scaled_from"] = json!([w, h]);
    }
    let boxes: Vec<Value> = boxes.into_iter().map(|(_, b)| b).collect();
    labels["boxes"] = Value::Array(boxes);
    let mask = crate::io::encode(&bits, "png", 0)?;
    Ok(Some(PageOut { id: sample_id(seed), image: image_bytes, ext: "jpg", mask: Some(mask), labels }))
}

/// `rfpii dataset synth`: `count` samples from seed `seed` on (sample i has
/// seed `seed + i`), written to `out`.
pub fn run(templates: &Path, out: &Path, opts: Options, append: bool) -> Result<Value> {
    let ctx = Ctx::new(templates, opts.clone())?;
    let w = Writer::open(out, "synthetic", append)?;
    let seeds: Vec<u64> = (0..opts.count as u64).map(|i| opts.seed.wrapping_add(i)).filter(|&s| !w.has(&sample_id(s))).collect();
    opencv::core::set_num_threads(1)?;
    crate::ocr::set_threads(1);
    let pool = rayon::ThreadPoolBuilder::new().num_threads(opts.workers.max(1)).build()?;
    let t0 = std::time::Instant::now();
    let results: Vec<Result<Option<Value>>> = pool.install(|| {
        seeds
            .par_iter()
            .map(|&s| match sample(&ctx, s)? {
                Some(p) => Ok(Some(super::write_page(out, &p)?)),
                None => Ok(None),
            })
            .collect()
    });
    let mut lines = Vec::new();
    let mut held = 0;
    for r in results {
        match r? {
            Some(l) => lines.push(l),
            None => held += 1,
        }
    }
    let secs = t0.elapsed().as_secs_f64();
    let mut params = opts.to_json(&ctx.names());
    if opts.redacted {
        params["held_by_pipeline"] = json!(held);
    }
    let mut notes = vec![
        "Synthetic forms: exact labels and masks, drawn on the templates' reference images (real blank printings, handwriting erased; a few marks of the original filling remain on some references and are labelled print).".to_string(),
        "A field's box is its writing (strokes and their width), or its cell when it is empty (`empty: true`). `slots` are characters' x-spans, given when the page is not turned; `char_boxes` always.".to_string(),
    ];
    if opts.redacted {
        notes.push("Redacted pages: masked by the pipeline (templates, rules, the leak check), some masks then undone or thinned on purpose; each field says what still shows (`redaction.status`). Mask bit 8 is the blacked blocks; bits 1, 2, 4 are the ink before redaction.".into());
    }
    if opts.max_side.is_some() {
        notes.push("Pages made smaller: a mask bit is set where it covers at least a quarter of the pixel.".into());
    }
    let n = lines.len();
    let mut meta = w.finish(lines, params, false, "exact", notes)?;
    meta["this_run"] = json!({"pages": n, "held": held, "seconds": (secs * 10.0).round() / 10.0, "pages_per_second": if secs > 0.0 { ((n as f64 / secs) * 100.0).round() / 100.0 } else { 0.0 }});
    Ok(meta)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn templates() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../templates")
    }

    fn opts(seed: u64, only: &[&str]) -> Options {
        Options { seed, count: 1, only: only.iter().map(|s| s.to_string()).collect(), max_side: None, profile: Profile::Mixed, negatives: 0.0, redacted: false, workers: 1 }
    }

    /// Every pen pixel of each field drawn alone, moved as the page was,
    /// lies inside the field's box; each character's x-span is inside it.
    #[test]
    fn boxes_hold_their_strokes_and_slots_lie_in_their_boxes() {
        let ctx = Ctx::new(&templates(), opts(3, &["form34a_2022"])).unwrap();
        for seed in [3u64, 11, 12] {
            let mut g = Gen::new(seed);
            let d = draw(&ctx, &mut g, true).unwrap();
            let look = choose_look(&mut g, Profile::Mixed);
            let shot = photograph(&d, &look, &mut g).unwrap();
            let (w, h) = (shot.image.cols(), shot.image.rows());
            let owner = shot.owner.as_ref().unwrap();
            let ov = owner.data_typed::<u16>().unwrap();
            let stamps: Vec<[f64; 3]> = d.items.iter().filter_map(|i| i.circle).collect();
            let mut checked = 0;
            for it in d.items.iter().filter(|i| !i.owners.is_empty()) {
                let lab = item_label(it, &shot.h, w as f64, h as f64, &stamps, look.quarter_turns != 0).unwrap();
                let b: Vec<f64> = lab["box"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect();
                for (i, &o) in ov.iter().enumerate() {
                    if it.owners.contains(&o) {
                        let (x, y) = ((i as i32 % w) as f64, (i as i32 / w) as f64);
                        assert!(x >= b[0] - 0.5 && x <= b[2] + 0.5 && y >= b[1] - 0.5 && y <= b[3] + 0.5, "seed {seed} {}: ({x},{y}) outside {b:?}", it.zone);
                        checked += 1;
                    }
                }
                for s in lab["slots"].as_array().into_iter().flatten() {
                    let (a, z) = (s[0].as_f64().unwrap(), s[1].as_f64().unwrap());
                    assert!(a >= b[0] && z <= b[2] && a <= z);
                }
            }
            assert!(checked > 100, "seed {seed}: {checked}");
        }
    }

    #[test]
    fn the_mask_pen_pixels_are_the_strokes_drawn() {
        let ctx = Ctx::new(&templates(), opts(5, &["form34b_2022_block"])).unwrap();
        let mut g = Gen::new(5);
        let d = draw(&ctx, &mut g, true).unwrap();
        let pen = util::bytes(&d.layers.pen).unwrap();
        let bits = util::bytes(&d.layers.bits().unwrap()).unwrap();
        assert!(pen.iter().filter(|&&v| v != 0).count() > 500);
        for (p, b) in pen.iter().zip(&bits) {
            assert_eq!(*p != 0, b & super::super::draw::BIT_PEN != 0);
        }
    }

    #[test]
    fn a_seed_gives_the_same_page_and_labels() {
        let ctx = Ctx::new(&templates(), opts(9, &["form34b_2022_block_b"])).unwrap();
        let a = sample(&ctx, 9).unwrap().unwrap();
        let b = sample(&ctx, 9).unwrap().unwrap();
        assert_eq!(a.image, b.image);
        assert_eq!(a.mask, b.mask);
        assert_eq!(a.labels, b.labels);
        let c = sample(&ctx, 10).unwrap().unwrap();
        assert_ne!(a.image, c.image);
    }

    #[test]
    fn a_smaller_page_keeps_its_labels_in_proportion() {
        let mut o = opts(21, &["form34a_2022"]);
        let ctx = Ctx::new(&templates(), o.clone()).unwrap();
        let full = sample(&ctx, 21).unwrap().unwrap();
        o.max_side = Some(800);
        let ctx = Ctx::new(&templates(), o).unwrap();
        let small = sample(&ctx, 21).unwrap().unwrap();
        let (fw, sw) = (full.labels["page"]["width"].as_f64().unwrap(), small.labels["page"]["width"].as_f64().unwrap());
        assert!(small.labels["page"]["height"].as_f64().unwrap() <= 800.0);
        let f = sw / fw;
        for (a, b) in full.labels["boxes"].as_array().unwrap().iter().zip(small.labels["boxes"].as_array().unwrap()) {
            assert_eq!(a["zone"], b["zone"]);
            let (ax, bx) = (a["box"][0].as_f64().unwrap(), b["box"][0].as_f64().unwrap());
            assert!((ax * f - bx).abs() < 3.0, "{} {} {}", a["zone"], ax * f, bx);
        }
        let m = crate::io::decode_bytes(small.mask.as_ref().unwrap()).unwrap();
        assert_eq!(m.cols() as f64, sw);
    }

    #[test]
    fn negatives_have_no_fields() {
        let mut o = opts(1, &["form34a_2022", "form34b_2022_results"]);
        o.negatives = 1.0;
        let ctx = Ctx::new(&templates(), o).unwrap();
        let mut kinds = std::collections::BTreeSet::new();
        for s in 0..12 {
            let p = sample(&ctx, s).unwrap().unwrap();
            let k = p.labels["page"]["kind"].as_str().unwrap().to_string();
            for b in p.labels["boxes"].as_array().unwrap() {
                assert!(b["empty"] == true || ["results_region", "agents_table", "declaration", "comments", "stamp"].contains(&b["class"].as_str().unwrap()), "{}", b);
            }
            kinds.insert(k);
        }
        assert!(kinds.contains("form") && kinds.len() >= 2, "{kinds:?}");
    }
}
