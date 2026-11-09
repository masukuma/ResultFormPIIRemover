//! `rfpii dataset check` and `rfpii dataset stats`.

use anyhow::Result;
use opencv::imgcodecs;
use opencv::prelude::*;
use serde_json::{json, Map, Value};
use std::collections::{BTreeMap, HashMap};
use std::path::Path;

use super::{read_index, sha256_hex, split_of, CLASSES, DATASET_SCHEMA, KINDS, LABELS_SCHEMA, SOURCES, SPLITS_SCHEMA, STATUSES};

fn nums(v: &Value) -> Option<Vec<f64>> {
    v.as_array()?.iter().map(|x| x.as_f64()).collect()
}

/// The size of an encoded image, from its header where possible.
fn image_size(data: &[u8]) -> Option<(i64, i64)> {
    if data.starts_with(b"\x89PNG") && data.len() >= 24 {
        let w = u32::from_be_bytes(data[16..20].try_into().ok()?) as i64;
        let h = u32::from_be_bytes(data[20..24].try_into().ok()?) as i64;
        return Some((w, h));
    }
    let m = imgcodecs::imdecode(&opencv::core::Vector::<u8>::from_slice(data), imgcodecs::IMREAD_GRAYSCALE | imgcodecs::IMREAD_IGNORE_ORIENTATION).ok()?;
    (!m.empty()).then(|| (m.cols() as i64, m.rows() as i64))
}

/// Problems with one page's labels (`w` x `h` its image).
pub fn check_labels(l: &Value, w: f64, h: f64) -> Vec<String> {
    let mut p = Vec::new();
    if l["schema"] != LABELS_SCHEMA {
        p.push(format!("schema {} (expected {})", l["schema"], LABELS_SCHEMA));
    }
    let pg = &l["page"];
    if pg["width"].as_f64() != Some(w) || pg["height"].as_f64() != Some(h) {
        p.push(format!("page size {}x{} but the image is {}x{}", pg["width"], pg["height"], w, h));
    }
    if !KINDS.contains(&pg["kind"].as_str().unwrap_or("")) {
        p.push(format!("kind {}", pg["kind"]));
    }
    if !pg["held"].is_boolean() {
        p.push("page.held missing".into());
    }
    if !l["split_key"].is_string() {
        p.push("split_key missing".into());
    }
    if l["teacher"].is_null() == l["synthetic"].is_null() {
        p.push("labels must say teacher or synthetic".into());
    }
    for (i, b) in l["boxes"].as_array().into_iter().flatten().enumerate() {
        let at = format!("box {} ({})", i, b["zone"].as_str().unwrap_or(""));
        if !CLASSES.contains(&b["class"].as_str().unwrap_or("")) {
            p.push(format!("{}: class {}", at, b["class"]));
        }
        if !SOURCES.contains(&b["source"].as_str().unwrap_or("")) {
            p.push(format!("{}: source {}", at, b["source"]));
        }
        let Some(r) = nums(&b["box"]).filter(|r| r.len() == 4) else {
            p.push(format!("{}: box must be 4 numbers", at));
            continue;
        };
        if !(r[0] < r[2] && r[1] < r[3]) {
            p.push(format!("{}: box {:?} is empty or reversed", at, r));
        }
        if r[0] < 0.0 || r[1] < 0.0 || r[2] > w || r[3] > h {
            p.push(format!("{}: box {:?} outside the {}x{} image", at, r, w, h));
        }
        for s in b["slots"].as_array().into_iter().flatten() {
            match nums(s).filter(|s| s.len() == 2) {
                Some(s) if s[0] <= s[1] && s[0] >= r[0] - 0.5 && s[1] <= r[2] + 0.5 => {}
                _ => p.push(format!("{}: slot {} not inside its box {:?}", at, s, r)),
            }
        }
        for c in b["char_boxes"].as_array().into_iter().flatten() {
            match nums(c).filter(|c| c.len() == 4) {
                Some(c) if c[0] >= 0.0 && c[1] >= 0.0 && c[2] <= w && c[3] <= h => {}
                _ => p.push(format!("{}: character box {} outside the image", at, c)),
            }
        }
        if let Some(s) = b["redaction"]["status"].as_str()
            && !STATUSES.contains(&s)
        {
            p.push(format!("{}: redaction status {}", at, s));
        }
    }
    p
}

/// Every problem with a dataset; empty if it is sound.
pub fn check(dir: &Path) -> Result<(Vec<String>, usize)> {
    let mut p = Vec::new();
    // every file listed with its hash, and nothing unlisted
    let sums = std::fs::read_to_string(dir.join("SHA256SUMS")).unwrap_or_default();
    if sums.is_empty() {
        p.push("SHA256SUMS missing or empty".into());
    }
    let mut listed = std::collections::HashSet::new();
    for line in sums.lines() {
        let Some((h, f)) = line.split_once("  ") else {
            p.push(format!("SHA256SUMS: bad line {:?}", line));
            continue;
        };
        listed.insert(f.to_string());
        if f.contains("..") || f.starts_with('/') {
            p.push(format!("SHA256SUMS: path {} leaves the dataset", f));
            continue;
        }
        match std::fs::read(dir.join(f)) {
            Ok(d) if sha256_hex(&d) == h => {}
            Ok(_) => p.push(format!("{}: SHA-256 differs from SHA256SUMS", f)),
            Err(_) => p.push(format!("{}: listed in SHA256SUMS but missing", f)),
        }
    }
    for f in crate::annex::files_under(dir, "") {
        let rel = f.strip_prefix(dir).unwrap_or(&f).to_string_lossy().into_owned();
        if rel != "SHA256SUMS" && !listed.contains(&rel) {
            p.push(format!("{}: not listed in SHA256SUMS", rel));
        }
    }
    let meta = crate::manifest::read_json(&dir.join("dataset.json")).unwrap_or(Value::Null);
    if meta["schema"] != DATASET_SCHEMA {
        p.push(format!("dataset.json: schema {} (expected {})", meta["schema"], DATASET_SCHEMA));
    }
    match meta["kind"].as_str() {
        Some("teacher") if meta["personal_data"] != true => p.push("dataset.json: a teacher dataset holds personal data (personal_data: true)".into()),
        Some("synthetic") if meta["personal_data"] != false => p.push("dataset.json: a synthetic dataset holds no personal data (personal_data: false)".into()),
        Some("teacher") | Some("synthetic") => {}
        _ => p.push(format!("dataset.json: kind {}", meta["kind"])),
    }
    if meta["classes"] != json!(CLASSES) {
        p.push("dataset.json: classes differ from this version's".into());
    }
    let index = match read_index(dir) {
        Ok(i) => i,
        Err(e) => {
            p.push(format!("index.jsonl: {:#}", e));
            return Ok((p, 0));
        }
    };
    if meta["pages"].as_u64() != Some(index.len() as u64) {
        p.push(format!("dataset.json says {} pages, the index lists {}", meta["pages"], index.len()));
    }
    let splits = crate::manifest::read_json(&dir.join("splits.json")).unwrap_or(Value::Null);
    if splits["schema"] != SPLITS_SCHEMA {
        p.push("splits.json: schema".into());
    }
    let mut in_split: HashMap<String, String> = HashMap::new();
    for s in ["train", "val", "test"] {
        for id in splits["pages"][s].as_array().into_iter().flatten() {
            if in_split.insert(id.as_str().unwrap_or("").to_string(), s.to_string()).is_some() {
                p.push(format!("splits.json: {} in two splits", id));
            }
        }
    }
    let mut ids = std::collections::HashSet::new();
    for l in &index {
        let id = l["id"].as_str().unwrap_or("").to_string();
        if !ids.insert(id.clone()) {
            p.push(format!("{}: twice in the index", id));
        }
        let key = l["split_key"].as_str().unwrap_or("");
        if l["split"].as_str() != Some(split_of(key)) {
            p.push(format!("{}: split {} but its key gives {}", id, l["split"], split_of(key)));
        }
        if in_split.get(&id).map(String::as_str) != l["split"].as_str() {
            p.push(format!("{}: splits.json disagrees with the index", id));
        }
        let read = |k: &str| l[k].as_str().and_then(|f| std::fs::read(dir.join(f)).ok());
        let Some(image) = read("image") else {
            p.push(format!("{}: image missing", id));
            continue;
        };
        if l["image_sha256"].as_str() != Some(sha256_hex(&image).as_str()) {
            p.push(format!("{}: image SHA-256 differs from the index", id));
        }
        let Some(lb) = read("labels") else {
            p.push(format!("{}: labels missing", id));
            continue;
        };
        if l["labels_sha256"].as_str() != Some(sha256_hex(&lb).as_str()) {
            p.push(format!("{}: labels SHA-256 differs from the index", id));
        }
        let labels: Value = match serde_json::from_slice(&lb) {
            Ok(v) => v,
            Err(e) => {
                p.push(format!("{}: labels: {}", id, e));
                continue;
            }
        };
        if labels["id"].as_str() != Some(id.as_str()) || labels["split_key"].as_str() != Some(key) {
            p.push(format!("{}: labels name another page or key", id));
        }
        let Some((w, h)) = image_size(&image) else {
            p.push(format!("{}: image can't be decoded", id));
            continue;
        };
        p.extend(check_labels(&labels, w as f64, h as f64).into_iter().map(|e| format!("{}: {}", id, e)));
        if let Some(m) = read("mask") {
            if l["mask_sha256"].as_str() != Some(sha256_hex(&m).as_str()) {
                p.push(format!("{}: mask SHA-256 differs from the index", id));
            }
            let mm = imgcodecs::imdecode(&opencv::core::Vector::<u8>::from_slice(&m), imgcodecs::IMREAD_UNCHANGED)?;
            if mm.cols() as i64 != w || mm.rows() as i64 != h || mm.channels() != 1 {
                p.push(format!("{}: mask is {}x{}x{}, the image {}x{}", id, mm.cols(), mm.rows(), mm.channels(), w, h));
            } else if crate::util::bytes(&mm)?.iter().any(|&v| v > 15) {
                p.push(format!("{}: mask has values beyond its 4 bits", id));
            }
        } else if l["mask"].is_string() {
            p.push(format!("{}: mask missing", id));
        }
    }
    Ok((p, index.len()))
}

fn bump(m: &mut BTreeMap<String, u64>, k: &str) {
    *m.entry(k.to_string()).or_insert(0) += 1;
}

/// Pages by kind, template, split and capture; boxes by class; empty
/// cells, held pages, redaction status; bytes on disk.
pub fn stats(dir: &Path) -> Result<Value> {
    let meta = crate::manifest::read_json(&dir.join("dataset.json"))?;
    let index = read_index(dir)?;
    let mut by: BTreeMap<&str, BTreeMap<String, u64>> = BTreeMap::new();
    let (mut empty, mut held, mut struck, mut under) = (0u64, 0u64, 0u64, 0u64);
    let mut classes: BTreeMap<String, u64> = BTreeMap::new();
    let mut status: BTreeMap<String, u64> = BTreeMap::new();
    let mut sources: BTreeMap<String, u64> = BTreeMap::new();
    for l in &index {
        for k in ["kind", "template", "split", "capture"] {
            let v = l[k].as_str().unwrap_or("-");
            bump(by.entry(k).or_default(), v);
        }
        if l["held"] == true {
            held += 1;
        }
        let Ok(labels) = std::fs::read(dir.join(l["labels"].as_str().unwrap_or(""))).map_err(anyhow::Error::from).and_then(|b| serde_json::from_slice::<Value>(&b).map_err(Into::into)) else { continue };
        for b in labels["boxes"].as_array().into_iter().flatten() {
            if b["struck"] == true {
                struck += 1;
            }
            if b["empty"] == true {
                empty += 1;
                continue;
            }
            bump(&mut classes, b["class"].as_str().unwrap_or("?"));
            bump(&mut sources, b["source"].as_str().unwrap_or("?"));
            if b["under_stamp"] == true {
                under += 1;
            }
            if let Some(s) = b["redaction"]["status"].as_str() {
                bump(&mut status, s);
            }
        }
    }
    let mut bytes = Map::new();
    for sub in ["pages", "masks", "labels"] {
        let n: u64 = crate::annex::files_under(&dir.join(sub), "").iter().filter_map(|f| f.metadata().ok()).map(|m| m.len()).sum();
        bytes.insert(sub.into(), json!(n));
    }
    let total: u64 = bytes.values().filter_map(|v| v.as_u64()).sum();
    let per_page = if index.is_empty() { 0 } else { total / index.len() as u64 };
    Ok(json!({
        "dataset": dir.to_string_lossy(), "kind": meta["kind"], "personal_data": meta["personal_data"], "pages": index.len(),
        "pages_by": by, "held": held, "boxes_by_class": classes, "boxes_by_source": sources, "empty_cells": empty,
        "struck_cells": struck, "fields_under_stamps": under, "redaction_status": status,
        "bytes": bytes, "bytes_per_page": per_page,
    }))
}

pub fn summary(s: &Value) -> String {
    let mut out = format!("{} ({} dataset{}): {} pages, {} held\n", s["dataset"].as_str().unwrap_or(""), s["kind"].as_str().unwrap_or("?"), if s["personal_data"] == true { ", PERSONAL DATA" } else { "" }, s["pages"], s["held"]);
    for k in ["kind", "template", "split", "capture"] {
        let m = s["pages_by"][k].as_object().map(|m| m.iter().map(|(a, b)| format!("{} {}", a, b)).collect::<Vec<_>>().join(", ")).unwrap_or_default();
        out += &format!("  by {:<9} {}\n", k, m);
    }
    let join = |v: &Value| v.as_object().map(|m| m.iter().map(|(a, b)| format!("{} {}", a, b)).collect::<Vec<_>>().join(", ")).unwrap_or_default();
    out += &format!("  boxes      {}\n", join(&s["boxes_by_class"]));
    out += &format!("  sources    {}\n", join(&s["boxes_by_source"]));
    out += &format!("  empty cells {}, struck {}, fields under stamps {}\n", s["empty_cells"], s["struck_cells"], s["fields_under_stamps"]);
    if s["redaction_status"].as_object().is_some_and(|m| !m.is_empty()) {
        out += &format!("  redacted   {}\n", join(&s["redaction_status"]));
    }
    out += &format!("  on disk    {} (pages {}, masks {}, labels {}; {} per page)\n", mb(s["bytes"].as_object().map(|m| m.values().filter_map(|v| v.as_u64()).sum()).unwrap_or(0)), mb(s["bytes"]["pages"].as_u64().unwrap_or(0)), mb(s["bytes"]["masks"].as_u64().unwrap_or(0)), mb(s["bytes"]["labels"].as_u64().unwrap_or(0)), mb(s["bytes_per_page"].as_u64().unwrap_or(0)));
    out
}

fn mb(n: u64) -> String {
    if n >= 1 << 20 { format!("{:.1} MB", n as f64 / (1u64 << 20) as f64) } else { format!("{:.0} kB", n as f64 / 1024.0) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn labels_outside_the_image_or_slots_outside_boxes_are_found() {
        let good = json!({"schema": LABELS_SCHEMA, "page": {"width": 100, "height": 50, "kind": "form", "held": false}, "split_key": "seed:1", "synthetic": {},
                          "boxes": [{"class": "id_number", "source": "synthetic", "box": [10, 10, 60, 30], "slots": [[10, 20], [21, 60]]}]});
        assert!(check_labels(&good, 100.0, 50.0).is_empty(), "{:?}", check_labels(&good, 100.0, 50.0));
        let mut bad = good.clone();
        bad["boxes"][0]["box"] = json!([10, 10, 120, 30]);
        bad["boxes"][0]["slots"] = json!([[5, 20]]);
        bad["boxes"][0]["class"] = json!("face");
        let p = check_labels(&bad, 100.0, 50.0);
        assert!(p.iter().any(|e| e.contains("outside")) && p.iter().any(|e| e.contains("slot")) && p.iter().any(|e| e.contains("class")), "{p:?}");
        assert!(!check_labels(&good, 90.0, 50.0).is_empty());
    }
}
