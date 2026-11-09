//! Redacted synthetic pages, for the local reviewer (NEURAL.md §7, stage
//! C): what a published page still shows, known exactly.
//!
//! The synthetic page is redacted as `rfpii redact` would redact it (the
//! templates, the masking rules and the leak check, on the JPEG as stored).
//! Then, on some pages, a few masks are undone or cut back on purpose: a
//! signature left whole, a band made thin, a number shown in full or with
//! one hidden character too few. The blocks are blacked in the JPEG's
//! coefficients, as a published page's are. Each field is then labelled
//! by what its own ink shows (every pen pixel is known, `draw.rs`):
//!
//! - a number is `masked_ok` when the rule is met: at least `hide` of its
//!   characters hidden (an ID), or `hide` of a phone number's last
//!   `phone_tail` digits; a character is hidden when at least half its ink
//!   is under black. Otherwise `leak_number`.
//! - a signature is `masked_ok` when its middle is hidden: along it, in
//!   pieces 32 px long, the ink between the 40% and 60% of its height is at
//!   least 80% under black, in pieces holding at least 90% of its ink.
//!   Otherwise `leak_signature`.
//!
//! The pipeline's own misses are labelled the same way: no mask was undone
//! there (`perturbed` is null), but the ink still shows.

use anyhow::Result;
use opencv::core::Mat;
use opencv::prelude::*;
use serde_json::{json, Value};

use super::draw::BIT_BLACK;
use super::synth::{Gen, Item, Shot};
use crate::align::AlignSettings;
use crate::masking::MaskSettings;
use crate::pipeline::{redact_pages, PageOptions};
use crate::proof::BLOCK;
use crate::templates::Template;
use crate::util::{self, Grid};

pub struct Redactor {
    pub align: AlignSettings,
    pub mask: MaskSettings,
    /// share of pages with some masks undone or cut back on purpose
    pub perturb: f64,
}

impl Default for Redactor {
    fn default() -> Self {
        Redactor { align: AlignSettings::default(), mask: MaskSettings::default(), perturb: 0.6 }
    }
}

pub struct Redacted {
    pub jpeg: Vec<u8>,
    pub bits: Mat,
    pub record: Value,
}

/// What each owner's ink shows: its pixels (x, y) and which are under black.
struct Ink {
    pts: Vec<Vec<(i32, i32, bool)>>,
}

fn ink_of(owner: &Mat, blocks: &Grid, n: usize) -> Result<Ink> {
    let (w, h) = (owner.cols(), owner.rows());
    let o = owner.data_typed::<u16>()?;
    let mut pts = vec![Vec::new(); n + 1];
    for y in 0..h {
        for x in 0..w {
            let v = o[(y * w + x) as usize] as usize;
            if v == 0 || v > n {
                continue;
            }
            let (r, c) = (y as usize / BLOCK, x as usize / BLOCK);
            let black = r < blocks.rows && c < blocks.cols && blocks.get(r, c);
            pts[v].push((x, y, black));
        }
    }
    Ok(Ink { pts })
}

fn share_black(p: &[(i32, i32, bool)]) -> f64 {
    if p.is_empty() {
        return 0.0;
    }
    p.iter().filter(|q| q.2).count() as f64 / p.len() as f64
}

/// The share of a signature's ink in pieces whose middle is hidden, and
/// the share of its ink under black. `across`: the signature runs down the
/// image (a page on its side).
fn signature_hidden(p: &[(i32, i32, bool)], across: bool) -> (f64, f64) {
    let pos = |q: &(i32, i32, bool)| if across { (q.1, q.0) } else { (q.0, q.1) };
    let mut chunks: std::collections::BTreeMap<i32, Vec<(i32, bool)>> = std::collections::BTreeMap::new();
    for q in p {
        let (a, b) = pos(q);
        chunks.entry(a.div_euclid(32)).or_default().push((b, q.2));
    }
    let (mut good, mut seen) = (0usize, 0usize);
    for v in chunks.values() {
        if v.len() < 10 {
            continue;
        }
        let mut ys: Vec<i32> = v.iter().map(|q| q.0).collect();
        ys.sort();
        let (lo, hi) = (ys[(ys.len() as f64 * 0.4) as usize], ys[((ys.len() as f64 * 0.6) as usize).min(ys.len() - 1)]);
        let mid: Vec<bool> = v.iter().filter(|q| q.0 >= lo && q.0 <= hi).map(|q| q.1).collect();
        seen += v.len();
        if mid.iter().filter(|&&b| b).count() as f64 >= 0.8 * mid.len() as f64 {
            good += v.len();
        }
    }
    (if seen > 0 { good as f64 / seen as f64 } else { 0.0 }, share_black(p))
}

/// Whether a number's hidden characters meet the rule.
pub fn number_ok(hidden: &[bool], rule: &str, ms: &MaskSettings) -> bool {
    let n = hidden.len();
    if n == 0 {
        return true;
    }
    if ms.whole(rule) {
        return hidden.iter().all(|&h| h);
    }
    let need = ms.hide.min(n);
    let tail = if rule == "phone" { &hidden[n.saturating_sub(ms.phone_tail)..] } else { hidden };
    tail.iter().filter(|&&h| h).count() >= need.min(tail.len())
}

fn clear(blocks: &mut Grid, b: [f64; 4]) {
    let r0 = (b[1].max(0.0) as usize) / BLOCK;
    let r1 = ((b[3].max(0.0) as usize) / BLOCK).min(blocks.rows.saturating_sub(1));
    let c0 = (b[0].max(0.0) as usize) / BLOCK;
    let c1 = ((b[2].max(0.0) as usize) / BLOCK).min(blocks.cols.saturating_sub(1));
    for r in r0..=r1 {
        for c in c0..=c1 {
            blocks.set(r, c, false);
        }
    }
}

fn vbox(v: &Value) -> Option<[f64; 4]> {
    let a: Vec<f64> = v.as_array()?.iter().filter_map(|x| x.as_f64()).collect();
    (a.len() == 4).then(|| [a[0], a[1], a[2], a[3]])
}

/// Undo or cut back one field's mask: how, or None if there was nothing
/// to change.
fn perturb(blocks: &mut Grid, it: &Item, label: &Value, ink: &Ink, g: &mut Gen, across: bool) -> Option<&'static str> {
    let b = vbox(&label["box"])?;
    if it.class == "signature" {
        let how = match g.int(0, 9) {
            0..=4 => "undone",
            5..=7 => "thinned",
            _ => "half",
        };
        match how {
            "undone" => clear(blocks, b),
            "half" => {
                let mut h = b;
                if across { h[1] = (b[1] + b[3]) / 2.0 } else { h[0] = (b[0] + b[2]) / 2.0 }
                clear(blocks, h)
            }
            _ => {
                // keep one line of blocks through the middle of the band
                let (r0, r1) = ((b[1].max(0.0) as usize) / BLOCK, ((b[3].max(0.0) as usize) / BLOCK).min(blocks.rows - 1));
                let (c0, c1) = ((b[0].max(0.0) as usize) / BLOCK, ((b[2].max(0.0) as usize) / BLOCK).min(blocks.cols - 1));
                if across {
                    let cols: Vec<usize> = (c0..=c1).filter(|&c| (r0..=r1).any(|r| blocks.get(r, c))).collect();
                    if cols.len() < 2 {
                        return None;
                    }
                    let keep = cols[cols.len() / 2];
                    for &c in cols.iter().filter(|&&c| c != keep) {
                        (r0..=r1).for_each(|r| blocks.set(r, c, false));
                    }
                } else {
                    let rows: Vec<usize> = (r0..=r1).filter(|&r| (c0..=c1).any(|c| blocks.get(r, c))).collect();
                    if rows.len() < 2 {
                        return None;
                    }
                    let keep = rows[rows.len() / 2];
                    for &r in rows.iter().filter(|&&r| r != keep) {
                        (c0..=c1).for_each(|c| blocks.set(r, c, false));
                    }
                }
            }
        }
        return Some(how);
    }
    if g.p(0.5) {
        clear(blocks, b);
        return Some("undone");
    }
    // one hidden character fewer
    let chars: Vec<[f64; 4]> = label["char_boxes"].as_array()?.iter().filter_map(vbox).collect();
    let hidden: Vec<usize> = it.owners.iter().enumerate().filter(|(_, o)| share_black(&ink.pts[**o as usize]) >= 0.5).map(|(i, _)| i).collect();
    if hidden.is_empty() || chars.len() != it.owners.len() {
        return None;
    }
    let k = *g.pick(&hidden);
    clear(blocks, chars[k]);
    Some("one_less")
}

/// Redact a synthetic page (`jpeg`, as stored) with the pipeline, cut some
/// masks back, and label what each field shows (in `boxes`, matched to
/// `items` by index). None when the pipeline holds the page.
#[allow(clippy::too_many_arguments)]
pub fn redact(rd: &Redactor, templates: &[Template], jpeg: &[u8], shot: &Shot, quarter_turns: i32, items: &[Item], boxes: &mut [(usize, Value)], g: &mut Gen, seed: u64) -> Result<Option<Redacted>> {
    let page = crate::io::decode_bytes(jpeg)?;
    opencv::core::set_rng_seed((seed % (1u64 << 31)) as i32)?;
    let mut rng = crate::pyrand::PyRandom::new(seed ^ 0x5eed_5eed);
    let po = PageOptions { generic: false, ..Default::default() };
    let res = redact_pages(std::slice::from_ref(&page), templates, &rd.align, &rd.mask, &mut rng, &po)?.remove(0);
    if !res.ok {
        return Ok(None);
    }
    let mut blocks = res.blocks.clone().unwrap();
    let owner = shot.owner.as_ref().expect("redacted samples keep the pen's owners");
    let n_owner = items.iter().flat_map(|i| i.owners.iter()).copied().max().unwrap_or(0) as usize;
    let across = quarter_turns % 2 != 0;
    let fields: Vec<usize> = (0..boxes.len()).filter(|&k| {
        let it = &items[boxes[k].0];
        !it.owners.is_empty() && ["signature", "id_number", "phone_number"].contains(&it.class)
    })
    .collect();
    let mut changed: Vec<(usize, &'static str)> = Vec::new();
    if !fields.is_empty() && g.p(rd.perturb) {
        let before = ink_of(owner, &blocks, n_owner)?;
        let m = g.int(1, fields.len().min(3) as i64) as usize;
        let mut pool = fields.clone();
        for _ in 0..m {
            let j = g.int(0, pool.len() as i64 - 1) as usize;
            let k = pool.remove(j);
            let (i, label) = (&boxes[k].0, &boxes[k].1);
            if let Some(how) = perturb(&mut blocks, &items[*i], label, &before, g, across) {
                changed.push((k, how));
            }
        }
    }
    let red = crate::proof::redact_jpeg(jpeg, &blocks)?;
    let ink = ink_of(owner, &blocks, n_owner)?;
    let mut leaks = 0;
    for &k in &fields {
        let it = &items[boxes[k].0];
        let how = changed.iter().find(|c| c.0 == k).map(|c| c.1);
        let rec = if it.class == "signature" {
            let p: Vec<(i32, i32, bool)> = it.owners.iter().flat_map(|&o| ink.pts[o as usize].iter().copied()).collect();
            let (middle, all) = signature_hidden(&p, across);
            let ok = middle >= 0.9 && all >= 0.25;
            json!({"status": if ok { "masked_ok" } else { "leak_signature" }, "ink_hidden": (all * 1000.0).round() / 1000.0, "middle_hidden": (middle * 1000.0).round() / 1000.0, "perturbed": how})
        } else {
            let hidden: Vec<bool> = it.owners.iter().filter(|&&o| !ink.pts[o as usize].is_empty()).map(|&o| share_black(&ink.pts[o as usize]) >= 0.5).collect();
            let rule = it.rule.unwrap_or("id");
            let ok = number_ok(&hidden, rule, &rd.mask);
            let all: Vec<(i32, i32, bool)> = it.owners.iter().flat_map(|&o| ink.pts[o as usize].iter().copied()).collect();
            json!({"status": if ok { "masked_ok" } else { "leak_number" }, "ink_hidden": (share_black(&all) * 1000.0).round() / 1000.0,
                   "chars": hidden.len(), "chars_hidden": hidden.iter().filter(|&&h| h).count(), "perturbed": how})
        };
        if rec["status"] != "masked_ok" {
            leaks += 1;
        }
        boxes[k].1["redaction"] = rec;
    }
    // the mask gains the blacked blocks
    let black = util::bytes(&crate::proof::to_pixels(&blocks, shot.bits.rows(), shot.bits.cols())?)?;
    let mut bits = util::bytes(&shot.bits)?;
    for (b, &k) in bits.iter_mut().zip(&black) {
        if k != 0 {
            *b |= BIT_BLACK;
        }
    }
    let record = json!({
        "template": res.template, "inliers": res.inliers, "score": (res.score * 1000.0).round() / 1000.0,
        "blocks_blacked": blocks.count(), "blocks_by_pipeline": res.blocks.as_ref().map(|b| b.count()),
        "perturbed": changed.iter().map(|(k, how)| json!({"zone": items[boxes[*k].0].zone, "how": how})).collect::<Vec<_>>(),
        "fields": fields.len(), "leaks": leaks,
        "settings": {"hide": rd.mask.hide, "phone_tail": rd.mask.phone_tail, "stamps": rd.mask.stamps.name(), "full_ids": rd.mask.full_ids, "full_phones": rd.mask.full_phones},
        "notes": res.notes,
    });
    Ok(Some(Redacted { jpeg: red.data, bits: util::from_bytes(shot.bits.rows(), shot.bits.cols(), &bits)?, record }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_number_rule_counts_hidden_characters_where_it_should() {
        let ms = MaskSettings::default();
        let h = |s: &str| s.chars().map(|c| c == 'x').collect::<Vec<bool>>();
        assert!(number_ok(&h("x.x..x.."), "id", &ms));
        assert!(!number_ok(&h("x.x....."), "id", &ms));
        // a phone number's hidden digits count only in its last six
        assert!(!number_ok(&h("xxx......."), "phone", &ms));
        assert!(number_ok(&h("....x.x.x."), "phone", &ms));
        let full = MaskSettings { full_ids: true, ..MaskSettings::default() };
        assert!(!number_ok(&h("xxxxxxx."), "id", &full));
    }

    #[test]
    fn a_signature_is_hidden_when_its_middle_is() {
        // a flat signature 200 px long, 30 px high
        let pts: Vec<(i32, i32)> = (0..200).flat_map(|x| (0..30).map(move |y| (x, 100 + y))).collect();
        let band = |y: i32| (110..=120).contains(&y);
        let p: Vec<(i32, i32, bool)> = pts.iter().map(|&(x, y)| (x, y, band(y))).collect();
        let (mid, all) = signature_hidden(&p, false);
        assert!(mid > 0.99 && all > 0.3 && all < 0.4, "{mid} {all}");
        let thin: Vec<(i32, i32, bool)> = pts.iter().map(|&(x, y)| (x, y, y == 115)).collect();
        assert!(signature_hidden(&thin, false).0 < 0.5);
        // the same, on its side
        let side: Vec<(i32, i32, bool)> = p.iter().map(|&(x, y, b)| (y, x, b)).collect();
        assert!(signature_hidden(&side, true).0 > 0.99);
    }
}
