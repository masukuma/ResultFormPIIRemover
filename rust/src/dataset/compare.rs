//! `rfpii dataset compare TRUTH OTHER`: how close one dataset's boxes come
//! to another's on the same pages (matched by image SHA-256). Made for the
//! teacher against synthetic forms, where the truth is known: per class,
//! how many true boxes the other dataset finds (IoU at or above a bar), how
//! many of its boxes are true, and how well each true box is covered (its
//! best IoU); per field cell, whether both say it is empty; per number,
//! whether the character slots agree.

use anyhow::Result;
use serde_json::{json, Map, Value};
use std::collections::{BTreeMap, HashMap};
use std::path::Path;

use super::{iou, read_index, CLASSES};

fn bx(v: &Value) -> Option<[f64; 4]> {
    let a: Vec<f64> = v.as_array()?.iter().filter_map(|x| x.as_f64()).collect();
    (a.len() == 4).then(|| [a[0], a[1], a[2], a[3]])
}

fn labels(dir: &Path, line: &Value) -> Option<Value> {
    serde_json::from_slice(&std::fs::read(dir.join(line["labels"].as_str()?)).ok()?).ok()
}

#[derive(Default)]
struct Tally {
    truth: usize,
    other: usize,
    matched: usize,
    best: Vec<f64>,
}

/// Greedy one-to-one matching by IoU, best pairs first: how many pairs
/// reach `bar`.
fn matched(t: &[[f64; 4]], o: &[[f64; 4]], bar: f64) -> usize {
    let mut pairs: Vec<(f64, usize, usize)> = Vec::new();
    for (i, a) in t.iter().enumerate() {
        for (j, b) in o.iter().enumerate() {
            let v = iou(*a, *b);
            if v >= bar {
                pairs.push((v, i, j));
            }
        }
    }
    pairs.sort_by(|a, b| b.0.total_cmp(&a.0));
    let (mut ut, mut uo) = (vec![false; t.len()], vec![false; o.len()]);
    let mut n = 0;
    for (_, i, j) in pairs {
        if !ut[i] && !uo[j] {
            ut[i] = true;
            uo[j] = true;
            n += 1;
        }
    }
    n
}

fn median(v: &mut [f64]) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    v.sort_by(|a, b| a.total_cmp(b));
    v[v.len() / 2]
}

pub fn compare(truth: &Path, other: &Path, bar: f64) -> Result<Value> {
    let ti = read_index(truth)?;
    let oi = read_index(other)?;
    let by_sha: HashMap<&str, &Value> = oi.iter().filter_map(|l| Some((l["image_sha256"].as_str()?, l))).collect();
    let mut tally: BTreeMap<String, Tally> = BTreeMap::new();
    let mut by_template: BTreeMap<String, BTreeMap<String, Tally>> = BTreeMap::new();
    let (mut pages, mut found, mut held) = (0, 0, 0);
    let mut cells = json!({"both_empty": 0, "both_written": 0, "written_seen_empty": 0, "empty_seen_written": 0});
    let (mut slot_pairs, mut slot_same, mut slot_err, mut slot_n) = (0, 0, 0.0, 0);
    for tl in &ti {
        pages += 1;
        let Some(ol) = tl["image_sha256"].as_str().and_then(|s| by_sha.get(s)) else { continue };
        found += 1;
        if ol["held"] == true {
            held += 1;
            continue;
        }
        let (Some(tlab), Some(olab)) = (labels(truth, tl), labels(other, ol)) else { continue };
        let template = tlab["page"]["template"].as_str().unwrap_or("-").to_string();
        let tb: Vec<&Value> = tlab["boxes"].as_array().map(|a| a.iter().collect()).unwrap_or_default();
        let ob: Vec<&Value> = olab["boxes"].as_array().map(|a| a.iter().collect()).unwrap_or_default();
        let complete: Vec<&str> = olab["exhaustive"].as_array().map(|a| a.iter().filter_map(|x| x.as_str()).collect()).unwrap_or_default();
        for c in CLASSES {
            let t: Vec<[f64; 4]> = tb.iter().filter(|b| b["class"] == *c && b["empty"] != true).filter_map(|b| bx(&b["box"])).collect();
            let o: Vec<[f64; 4]> = ob.iter().filter(|b| b["class"] == *c && b["empty"] != true).filter_map(|b| bx(&b["box"])).collect();
            if t.is_empty() && o.is_empty() {
                continue;
            }
            // a class the other dataset never labels isn't counted against it
            if o.is_empty() && !complete.contains(c) {
                continue;
            }
            let m = matched(&t, &o, bar);
            let best: Vec<f64> = t.iter().map(|a| o.iter().map(|b| iou(*a, *b)).fold(0.0, f64::max)).collect();
            for x in [tally.entry(c.to_string()).or_default(), by_template.entry(template.clone()).or_default().entry(c.to_string()).or_default()] {
                x.truth += t.len();
                x.other += o.len();
                x.matched += m;
                x.best.extend(best.iter().copied());
            }
        }
        // field cells, by zone: written or empty on both sides?
        for b in tb.iter().filter(|b| b["cell"].is_array() && ["signature", "id_number", "phone_number"].contains(&b["class"].as_str().unwrap_or(""))) {
            let Some(o) = ob.iter().find(|o| o["zone"] == b["zone"] && o["class"] == b["class"]) else { continue };
            let k = match (b["empty"] == true, o["empty"] == true) {
                (true, true) => "both_empty",
                (false, false) => "both_written",
                (false, true) => "written_seen_empty",
                (true, false) => "empty_seen_written",
            };
            cells[k] = json!(cells[k].as_u64().unwrap_or(0) + 1);
            let (ts, os) = (b["slots"].as_array(), o["slots"].as_array());
            if let (Some(ts), Some(os)) = (ts, os) {
                slot_pairs += 1;
                if ts.len() == os.len() {
                    slot_same += 1;
                    for (a, c) in ts.iter().zip(os) {
                        for i in 0..2 {
                            slot_err += (a[i].as_f64().unwrap_or(0.0) - c[i].as_f64().unwrap_or(0.0)).abs();
                            slot_n += 1;
                        }
                    }
                }
            }
        }
    }
    let table = |t: &mut BTreeMap<String, Tally>| -> Value {
        let mut m = Map::new();
        for (c, x) in t.iter_mut() {
            let mean = if x.best.is_empty() { 0.0 } else { x.best.iter().sum::<f64>() / x.best.len() as f64 };
            let r3 = |v: f64| (v * 1000.0).round() / 1000.0;
            m.insert(c.clone(), json!({"truth": x.truth, "other": x.other, "matched": x.matched,
                "recall": if x.truth > 0 { r3(x.matched as f64 / x.truth as f64) } else { 0.0 },
                "precision": if x.other > 0 { r3(x.matched as f64 / x.other as f64) } else { 0.0 },
                "mean_best_iou": r3(mean), "median_best_iou": r3(median(&mut x.best))}));
        }
        Value::Object(m)
    };
    let per_template: Map<String, Value> = by_template.iter_mut().map(|(k, v)| (k.clone(), table(v))).collect();
    Ok(json!({
        "truth": truth.to_string_lossy(), "other": other.to_string_lossy(), "iou_bar": bar,
        "pages": pages, "pages_in_other": found, "held_in_other": held,
        "classes": table(&mut tally), "by_template": per_template,
        "cells": cells,
        "slots": {"numbers_with_slots_on_both": slot_pairs, "same_count": slot_same,
                  "mean_boundary_error_px": if slot_n > 0 { (slot_err / slot_n as f64 * 10.0).round() / 10.0 } else { 0.0 }},
    }))
}

pub fn summary(r: &Value) -> String {
    let mut out = format!("{} pages; {} in the other dataset, {} of them held there; IoU bar {}\n", r["pages"], r["pages_in_other"], r["held_in_other"], r["iou_bar"]);
    out += "class            truth  other  matched  recall  precision  mean IoU  median IoU\n";
    for (c, v) in r["classes"].as_object().into_iter().flatten() {
        out += &format!("{:<16} {:>5}  {:>5}  {:>7}  {:>6}  {:>9}  {:>8}  {:>10}\n", c, v["truth"], v["other"], v["matched"], v["recall"], v["precision"], v["mean_best_iou"], v["median_best_iou"]);
    }
    out += &format!("cells: {}\nslots: {}\n", r["cells"], r["slots"]);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matching_is_one_to_one() {
        let t = [[0.0, 0.0, 10.0, 10.0], [20.0, 0.0, 30.0, 10.0]];
        let o = [[0.0, 0.0, 10.0, 10.0], [1.0, 0.0, 11.0, 10.0]];
        assert_eq!(matched(&t, &o, 0.5), 1);
        assert_eq!(matched(&t, &[[0.0, 0.0, 10.0, 10.0], [20.0, 0.0, 30.0, 10.0]], 0.5), 2);
        assert_eq!(matched(&t, &[], 0.5), 0);
    }
}
