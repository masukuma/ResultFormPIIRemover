//! Analytics over a run's records (HARNESS.md §4.5, §8): pages published
//! and held per file, held reasons, mask modes, and each published page's
//! risk signals, from its manifest alone (no image is opened).

use anyhow::Result;
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::path::Path;

use crate::annex::{self, files_under, normalise_reason};

/// Pages scoring this or more are all reviewed (HARNESS.md §8).
pub const RISK_THRESHOLD: f64 = 3.0;

/// Pages of a group needed before its layout is compared with theirs.
const LAYOUT_PEERS: usize = 5;

/// The last result for each document (file and page) in a review folder's
/// `report.jsonl`, in the order first seen.
pub fn last_results(review: &Path) -> Vec<Value> {
    let text = std::fs::read_to_string(review.join("report.jsonl")).unwrap_or_default();
    let mut order: Vec<(String, String)> = Vec::new();
    let mut last: BTreeMap<(String, String), Value> = BTreeMap::new();
    for r in text.lines().filter_map(|l| serde_json::from_str::<Value>(l).ok()) {
        let k = (r["file"].as_str().unwrap_or("").to_string(), r["page"].to_string());
        if !last.contains_key(&k) {
            order.push(k.clone());
        }
        last.insert(k, r);
    }
    order.into_iter().filter_map(|k| last.remove(&k)).collect()
}

/// The field a mask record is of: its zone's name without the row or place
/// ("agent_id[3]" and "signature@120,840" are of "agent_id", "signature").
fn field(zone: &str) -> &str {
    zone.split(['[', '@']).next().unwrap_or(zone)
}

/// One published page, as the risk signals see it.
struct Page {
    manifest: String,
    output: String,
    index: usize,
    template: String,
    signals: Vec<(f64, String)>,
    layout: Vec<(String, usize)>,
    id_chars: Option<f64>,
}

/// The signals a page's manifest record shows by itself.
fn own_signals(p: &Value, turn: i64) -> Vec<(f64, String)> {
    let masks = p["masks"].as_array().cloned().unwrap_or_default();
    let stamps: Vec<&Value> = p["protected"].as_array().map(|a| a.iter().filter(|x| x["kind"] == "stamp").collect()).unwrap_or_default();
    let notes: Vec<&str> = p["notes"].as_array().map(|a| a.iter().filter_map(|n| n.as_str()).collect()).unwrap_or_default();
    let mut s = Vec::new();
    let full = masks.iter().filter(|m| m["mode"] == "full").count();
    if full > 0 {
        s.push(((2 * full).min(4) as f64, format!("{} field line(s) hidden whole", full)));
    }
    let empty_officer = masks.iter().filter(|m| m["mode"] == "empty" && m["role"] == "officer" && field(m["zone"].as_str().unwrap_or("")).contains("signature")).count();
    if empty_officer > 0 && !stamps.is_empty() {
        s.push((3.0, format!("{} officer signature line(s) judged empty on a stamped page", empty_officer)));
    }
    if notes.iter().any(|n| n.starts_with("masks lifted off")) {
        s.push((2.0, "writing left visible on a stamp".into()));
    }
    if stamps.iter().any(|st| st.get("except_blocks").is_some()) {
        s.push((1.0, "a field's line hidden across a stamp".into()));
    }
    let collateral = notes.iter().filter(|n| n.contains("collateral text")).count();
    if collateral > 0 {
        s.push((1.0, format!("{} collateral-text note(s)", collateral)));
    }
    if turn.rem_euclid(180) == 90 {
        s.push((1.0, "read at a quarter turn".into()));
    }
    s
}

fn page_of(path: &Path, m: &Value, index: usize, p: &Value) -> Page {
    let masks = p["masks"].as_array().cloned().unwrap_or_default();
    // the fields read, by kind: those a layout records (pages read without
    // a template), else each masked field once
    let mut layout: BTreeMap<String, usize> = BTreeMap::new();
    match p["layout"]["fields"].as_array() {
        Some(fs) => fs.iter().for_each(|f| *layout.entry(f["kind"].as_str().unwrap_or("?").to_string()).or_insert(0) += 1),
        None => {
            let mut seen: Vec<&str> = masks.iter().map(|m| field(m["zone"].as_str().unwrap_or(""))).collect();
            seen.sort_unstable();
            seen.dedup();
            seen.iter().for_each(|f| *layout.entry(f.to_string()).or_insert(0) += 1);
        }
    }
    let slots: Vec<f64> = masks.iter().filter(|m| m["rule"] == "id").flat_map(|m| m["lines"].as_array().cloned().unwrap_or_default()).filter_map(|l| l["slots"].as_f64()).collect();
    let turn = p["orientation"]["turn"].as_i64().unwrap_or(0) + 90 * p["layout"]["frame"]["quarter_turns"].as_i64().unwrap_or(0);
    Page {
        manifest: path.to_string_lossy().into_owned(),
        output: m["output"]["file"].as_str().unwrap_or("").to_string(),
        index,
        template: p["template"].as_str().unwrap_or("").to_string(),
        signals: own_signals(p, turn),
        layout: layout.into_iter().collect(),
        id_chars: (!slots.is_empty()).then(|| crate::util::median(&slots)),
    }
}

/// The most common value among `v`, if at least half share it.
fn mode<T: Ord + Clone>(v: &[T]) -> Option<T> {
    let mut count: BTreeMap<&T, usize> = BTreeMap::new();
    v.iter().for_each(|x| *count.entry(x).or_insert(0) += 1);
    let (best, n) = count.into_iter().max_by_key(|(_, n)| *n)?;
    (2 * n >= v.len()).then(|| best.clone())
}

/// Every published page under `out` with its risk signals and score,
/// riskiest first. A page's layout is compared with its group's (the pages
/// read with the same template): the fields read, and the characters per
/// ID number.
pub fn risk(out: &Path) -> Vec<Value> {
    let mut pages: Vec<Page> = Vec::new();
    for path in files_under(out, ".manifest.json") {
        let Ok(m) = crate::manifest::read_json(&path) else { continue };
        for (i, p) in m["pages"].as_array().cloned().unwrap_or_default().iter().enumerate() {
            pages.push(page_of(&path, &m, i + 1, p));
        }
    }
    let mut groups: BTreeMap<String, Vec<usize>> = BTreeMap::new();
    for (i, p) in pages.iter().enumerate() {
        groups.entry(p.template.clone()).or_default().push(i);
    }
    for idx in groups.values().filter(|v| v.len() >= LAYOUT_PEERS) {
        let usual = mode(&idx.iter().map(|&i| pages[i].layout.clone()).collect::<Vec<_>>());
        let chars: Vec<f64> = idx.iter().filter_map(|&i| pages[i].id_chars).collect();
        let typical = (chars.len() >= LAYOUT_PEERS).then(|| crate::util::median(&chars));
        for &i in idx {
            if usual.as_ref().is_some_and(|u| *u != pages[i].layout) {
                pages[i].signals.push((3.0, "fields read unlike its group's".into()));
            }
            if let (Some(t), Some(c)) = (typical, pages[i].id_chars) {
                if c < t - 1.0 {
                    pages[i].signals.push((3.0, format!("ID numbers of {} characters where its group's have {}", c, t)));
                }
            }
        }
    }
    let mut out: Vec<Value> = pages
        .into_iter()
        .map(|p| {
            let score: f64 = p.signals.iter().map(|(w, _)| w).sum();
            json!({"manifest": p.manifest, "output": p.output, "page": p.index, "template": p.template, "score": score,
                   "signals": p.signals.into_iter().map(|(_, s)| s).collect::<Vec<_>>()})
        })
        .collect();
    out.sort_by(|a, b| b["score"].as_f64().partial_cmp(&a["score"].as_f64()).unwrap().then_with(|| a["manifest"].as_str().cmp(&b["manifest"].as_str())));
    out
}

/// Counts over a run's output folder (and its review folder): documents and
/// pages published and held, per source file; held reasons, normalised;
/// masks by mode; the DPIA annex's figures; and, with `with_risk`, every
/// published page's risk.
pub fn stats(out: &Path, review: Option<&Path>, with_risk: bool) -> Result<Value> {
    let figures = annex::figures(out, review)?;
    let mut per_file: BTreeMap<String, (usize, usize)> = BTreeMap::new();
    let mut held_reasons: BTreeMap<String, usize> = BTreeMap::new();
    match review {
        Some(rv) => {
            for r in last_results(rv) {
                let e = per_file.entry(r["file"].as_str().unwrap_or("").to_string()).or_default();
                if r["ok"] == true {
                    e.0 += 1;
                } else {
                    e.1 += 1;
                    *held_reasons.entry(normalise_reason(r["reason"].as_str().unwrap_or(""))).or_insert(0) += 1;
                }
            }
        }
        None => {
            for path in files_under(out, ".manifest.json") {
                if let Ok(m) = crate::manifest::read_json(&path) {
                    per_file.entry(m["source"]["path"].as_str().unwrap_or("").to_string()).or_default().0 += 1;
                }
            }
        }
    }
    let mut modes: BTreeMap<String, usize> = BTreeMap::new();
    for path in files_under(out, ".manifest.json") {
        let Ok(m) = crate::manifest::read_json(&path) else { continue };
        for p in m["pages"].as_array().cloned().unwrap_or_default() {
            for mk in p["masks"].as_array().cloned().unwrap_or_default() {
                *modes.entry(mk["mode"].as_str().unwrap_or("?").to_string()).or_insert(0) += 1;
            }
        }
    }
    let mut s = json!({
        "documents_published": figures["documents"],
        "pages_published": figures["pages_published"],
        "documents_held": per_file.values().map(|v| v.1).sum::<usize>(),
        "files": per_file.into_iter().map(|(f, (ok, held))| json!({"file": f, "published": ok, "held": held})).collect::<Vec<_>>(),
        "held_reasons": held_reasons,
        "mask_modes": modes,
        "figures": figures,
    });
    if with_risk {
        let r = risk(out);
        s["risk_threshold"] = json!(RISK_THRESHOLD);
        s["pages_at_risk"] = json!(r.iter().filter(|p| p["score"].as_f64().unwrap_or(0.0) >= RISK_THRESHOLD).count());
        s["risk"] = json!(r);
    }
    Ok(s)
}

/// A few lines for people: counts, held reasons, the riskiest pages.
pub fn summary(s: &Value) -> String {
    let mut t = format!("{} documents ({} pages) published, {} held\n", s["documents_published"], s["pages_published"], s["documents_held"]);
    for (k, v) in s["held_reasons"].as_object().into_iter().flatten() {
        t += &format!("held  {:>5}  {}\n", v, k);
    }
    for (k, v) in s["mask_modes"].as_object().into_iter().flatten() {
        t += &format!("masks {:>5}  {}\n", v, k);
    }
    if let Some(r) = s["risk"].as_array() {
        t += &format!("{} page(s) at or above the risk threshold ({})\n", s["pages_at_risk"], RISK_THRESHOLD);
        for p in r.iter().take(20).filter(|p| p["score"].as_f64().unwrap_or(0.0) > 0.0) {
            t += &format!("risk  {:>4}  {} page {}: {}\n", p["score"], p["output"].as_str().unwrap_or(""), p["page"], p["signals"].as_array().into_iter().flatten().filter_map(|x| x.as_str()).collect::<Vec<_>>().join("; "));
        }
    }
    t
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write(dir: &Path, name: &str, pages: Value) {
        let m = json!({"schema": "rfpii/manifest/1", "source": {"path": name, "sha256": name}, "output": {"file": format!("{}.pdf", name)},
                       "settings": {"hide": 3}, "tool": {}, "pages": pages});
        std::fs::write(dir.join(format!("{}.manifest.json", name)), m.to_string()).unwrap();
    }

    /// A page of `fields` ID columns and an officer's signature line,
    /// masked as `officer` says.
    fn page(fields: usize, officer: &str, extra: Value) -> Value {
        let mut masks: Vec<Value> = (0..fields).map(|k| json!({"zone": format!("f{}[0]", k), "mode": "partial", "rule": "id", "role": "agent", "lines": [{"slots": 8, "hidden": [1, 2, 3]}]})).collect();
        masks.push(json!({"zone": "officer_signatures[1]", "mode": officer, "role": "officer"}));
        json!({"template": "t", "masks": masks, "protected": extra["protected"], "notes": extra["notes"]})
    }

    #[test]
    fn pages_are_ranked_by_what_their_manifests_show() {
        let d = crate::testutil::tmpdir("risk");
        let stamp = json!([{"kind": "stamp", "circle": [10, 10, 5]}]);
        for k in 0..5 {
            write(&d, &format!("plain{}", k), json!([page(2, "signature", json!({"protected": stamp}))]));
        }
        write(&d, "empty_officer", json!([page(2, "empty", json!({"protected": stamp}))]));
        write(&d, "odd", json!([page(3, "signature", json!({"notes": ["masks lifted off 40 px inside stamps: writing on a stamp stays visible"]}))]));
        let r = risk(&d);
        assert_eq!(r.len(), 7);
        let by = |name: &str| r.iter().find(|p| p["manifest"].as_str().unwrap().contains(name)).unwrap().clone();
        // an officer's line judged empty beside a stamp
        assert_eq!(by("empty_officer")["score"], 3.0, "{}", by("empty_officer"));
        // a field more than its group, and writing left on a stamp
        assert_eq!(by("odd")["score"], 5.0);
        assert_eq!(by("plain0")["score"], 0.0);
        assert!(r[0]["score"].as_f64() >= r[1]["score"].as_f64() && r[1]["score"].as_f64() >= r[6]["score"].as_f64());
    }

    #[test]
    fn the_last_result_for_each_document_counts() {
        let d = crate::testutil::tmpdir("last");
        let lines = [json!({"file": "a.pdf", "page": 1, "ok": false, "reason": "x"}), json!({"file": "a.pdf", "page": 2, "ok": true}), json!({"file": "a.pdf", "page": 1, "ok": true})];
        std::fs::write(d.join("report.jsonl"), lines.iter().map(|l| l.to_string() + "\n").collect::<String>()).unwrap();
        let r = last_results(&d);
        assert_eq!(r.len(), 2);
        assert!(r.iter().all(|x| x["ok"] == true));
    }
}
