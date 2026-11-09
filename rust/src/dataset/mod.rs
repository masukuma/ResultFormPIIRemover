//! Training data for the local networks (NEURAL.md §3; the format is in
//! nn/README.md). A dataset is a folder:
//!
//! ```text
//! dataset.json        what it is: schema, kind, how it was made, classes
//! pages/<id>.<ext>    each page image as the pipeline saw it
//! labels/<id>.json    its labels (rfpii/nn-labels/1)
//! masks/<id>.png      per-pixel ink bits (optional)
//! index.jsonl         one line per page
//! splits.json         train / val / test, from each page's split key
//! SHA256SUMS          every file
//! ```
//!
//! `rfpii dataset synth` writes synthetic forms with exact labels (`synth`),
//! `rfpii dataset build` the teacher's labels from published manifests
//! (`teacher`), `check` verifies a dataset and `stats` counts it.

pub mod check;
pub mod compare;
pub mod draw;
pub mod glyphs;
pub mod layout;
pub mod look;
pub mod redacted;
pub mod synth;
pub mod teacher;

use anyhow::{bail, Context, Result};
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};

pub const DATASET_SCHEMA: &str = "rfpii/nn-dataset/1";
pub const LABELS_SCHEMA: &str = "rfpii/nn-labels/1";
pub const SPLITS_SCHEMA: &str = "rfpii/nn-splits/1";

/// The classes a box can have.
pub const CLASSES: &[&str] = &["signature", "id_number", "phone_number", "name", "stamp", "results_region", "agents_table", "declaration", "comments"];
/// Where a box came from.
pub const SOURCES: &[&str] = &["template_zone", "blocks", "vlm_found", "spot_mask", "stamp", "protect", "synthetic"];
/// What a page is.
pub const KINDS: &[&str] = &["form", "blank", "not_form", "unknown"];
/// What a redacted page shows of a field (`redaction.status`).
pub const STATUSES: &[&str] = &["masked_ok", "leak_signature", "leak_number"];

/// What each bit of a mask means.
pub fn mask_bits() -> Value {
    json!({"1": "print and rules", "2": "pen", "4": "stamp ink", "8": "blacked out by redaction (redacted pages)"})
}

pub fn sha256_hex(data: &[u8]) -> String {
    hex::encode(Sha256::digest(data))
}

/// The split of a page: the first 8 bytes of SHA-256(split key), as a
/// big-endian number v, give train if v * 10 < 8 * 2^64, val if
/// v * 10 < 9 * 2^64, else test (80 / 10 / 10). A page's split depends on
/// its key alone, so adding pages never moves one.
pub fn split_of(key: &str) -> &'static str {
    let h = Sha256::digest(key.as_bytes());
    let v = u64::from_be_bytes(h[..8].try_into().unwrap()) as u128;
    if v * 10 < 8u128 << 64 {
        "train"
    } else if v * 10 < 9u128 << 64 {
        "val"
    } else {
        "test"
    }
}

/// The time a dataset records: now, or `SOURCE_DATE_EPOCH` when set (so a
/// dataset made twice from the same seed is identical, file for file).
pub fn created_now() -> String {
    match std::env::var("SOURCE_DATE_EPOCH").ok().and_then(|s| s.trim().parse::<i64>().ok()) {
        Some(t) => crate::process::utc_at(t),
        None => crate::process::utc_now(),
    }
}

/// A box or point rounded to 0.1 px, as labels record them.
pub fn r1(v: f64) -> f64 {
    (v * 10.0).round() / 10.0
}

pub fn rbox(b: [f64; 4]) -> Value {
    json!([r1(b[0]), r1(b[1]), r1(b[2]), r1(b[3])])
}

pub fn rpoly(p: &[[f64; 2]]) -> Value {
    Value::Array(p.iter().map(|q| json!([r1(q[0]), r1(q[1])])).collect())
}

/// The box (x0, y0, x1, y1) around `pts`.
pub fn bounds(pts: &[[f64; 2]]) -> [f64; 4] {
    let mut b = [f64::INFINITY, f64::INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY];
    for p in pts {
        b[0] = b[0].min(p[0]);
        b[1] = b[1].min(p[1]);
        b[2] = b[2].max(p[0]);
        b[3] = b[3].max(p[1]);
    }
    b
}

/// A box clipped to a w x h image; None if nothing of it is left.
pub fn clip(b: [f64; 4], w: f64, h: f64) -> Option<[f64; 4]> {
    let c = [b[0].max(0.0), b[1].max(0.0), b[2].min(w), b[3].min(h)];
    (c[2] > c[0] && c[3] > c[1]).then_some(c)
}

pub fn iou(a: [f64; 4], b: [f64; 4]) -> f64 {
    let ix = (a[2].min(b[2]) - a[0].max(b[0])).max(0.0);
    let iy = (a[3].min(b[3]) - a[1].max(b[1])).max(0.0);
    let inter = ix * iy;
    let area = |r: [f64; 4]| (r[2] - r[0]).max(0.0) * (r[3] - r[1]).max(0.0);
    let u = area(a) + area(b) - inter;
    if u > 0.0 { inter / u } else { 0.0 }
}

/// One page, ready to be written.
pub struct PageOut {
    pub id: String,
    pub image: Vec<u8>,
    /// "jpg" or "png"
    pub ext: &'static str,
    pub mask: Option<Vec<u8>>,
    pub labels: Value,
}

/// Counts of each class among a page's boxes.
pub fn class_counts(labels: &Value) -> Map<String, Value> {
    let mut m = Map::new();
    for b in labels["boxes"].as_array().into_iter().flatten() {
        let c = b["class"].as_str().unwrap_or("?").to_string();
        let n = m.get(&c).and_then(|v| v.as_u64()).unwrap_or(0);
        m.insert(c, json!(n + 1));
    }
    let mut keys: Vec<String> = m.keys().cloned().collect();
    keys.sort();
    keys.into_iter().map(|k| (k.clone(), m[&k].clone())).collect()
}

/// Write a page's files under `dir`; returns its index line.
pub fn write_page(dir: &Path, p: &PageOut) -> Result<Value> {
    let image = format!("pages/{}.{}", p.id, p.ext);
    std::fs::write(dir.join(&image), &p.image)?;
    let mut labels = p.labels.clone();
    labels["image"] = json!(image);
    let mask = p.mask.as_ref().map(|_| format!("masks/{}.png", p.id));
    labels["mask"] = json!(mask);
    let lbytes = format!("{}\n", serde_json::to_string(&labels)?);
    let lpath = format!("labels/{}.json", p.id);
    std::fs::write(dir.join(&lpath), &lbytes)?;
    let mask_sha = match (&p.mask, &mask) {
        (Some(m), Some(path)) => {
            std::fs::write(dir.join(path), m)?;
            Some(sha256_hex(m))
        }
        _ => None,
    };
    let pg = &labels["page"];
    let empty = labels["boxes"].as_array().into_iter().flatten().filter(|b| b["empty"] == true).count();
    let leaks = labels["boxes"].as_array().into_iter().flatten().filter(|b| b["redaction"]["status"].as_str().is_some_and(|s| s.starts_with("leak"))).count();
    let key = labels["split_key"].as_str().unwrap_or("").to_string();
    let mut line = json!({
        "id": p.id, "image": image, "image_sha256": sha256_hex(&p.image),
        "labels": lpath, "labels_sha256": sha256_hex(lbytes.as_bytes()),
        "mask": mask, "mask_sha256": mask_sha,
        "split_key": key, "split": split_of(&key),
        "kind": pg["kind"], "held": pg["held"], "template": pg["template"], "form": pg["form"],
        "year": pg["year"], "capture": pg["capture"],
        "classes": class_counts(&labels), "empty": empty,
    });
    if labels["redaction"].is_object() {
        line["leaks"] = json!(leaks);
    }
    Ok(line)
}

/// A dataset folder being written: new, or (with `append`) one to add to.
pub struct Writer {
    pub dir: PathBuf,
    pub kind: String,
    /// the index lines already there
    pub existing: Vec<Value>,
    pub meta: Option<Value>,
}

impl Writer {
    pub fn open(dir: &Path, kind: &str, append: bool) -> Result<Writer> {
        let mut existing = Vec::new();
        let mut meta = None;
        if dir.exists() && std::fs::read_dir(dir)?.next().is_some() {
            if !append {
                bail!("{} is not empty (use --append to add to a dataset)", dir.display());
            }
            let m = crate::manifest::read_json(&dir.join("dataset.json")).with_context(|| format!("{} is not a dataset", dir.display()))?;
            if m["kind"].as_str() != Some(kind) {
                bail!("{} is a {} dataset, not {}", dir.display(), m["kind"], kind);
            }
            existing = read_index(dir)?;
            meta = Some(m);
        }
        for sub in ["pages", "labels", "masks"] {
            std::fs::create_dir_all(dir.join(sub))?;
        }
        Ok(Writer { dir: dir.to_path_buf(), kind: kind.into(), existing, meta })
    }

    pub fn has(&self, id: &str) -> bool {
        self.existing.iter().any(|l| l["id"].as_str() == Some(id))
    }

    /// Write the index, splits, dataset.json and SHA256SUMS, with `added`
    /// (index lines of the pages written by this run).
    pub fn finish(self, added: Vec<Value>, parameters: Value, personal_data: bool, ink: &str, notes: Vec<String>) -> Result<Value> {
        let n_added = added.len();
        let mut lines = self.existing;
        lines.extend(added);
        lines.sort_by(|a, b| a["id"].as_str().cmp(&b["id"].as_str()));
        lines.dedup_by(|a, b| a["id"] == b["id"]);
        let mut index = String::new();
        for l in &lines {
            index += &serde_json::to_string(l)?;
            index.push('\n');
        }
        std::fs::write(self.dir.join("index.jsonl"), index)?;
        let mut splits = json!({"train": [], "val": [], "test": []});
        for l in &lines {
            splits[l["split"].as_str().unwrap_or("train")].as_array_mut().unwrap().push(l["id"].clone());
        }
        let counts = json!({"train": splits["train"].as_array().unwrap().len(), "val": splits["val"].as_array().unwrap().len(), "test": splits["test"].as_array().unwrap().len()});
        let sp = json!({"schema": SPLITS_SCHEMA,
                        "rule": "the first 8 bytes of SHA-256(split_key), big-endian, as v: train if v*10 < 8*2^64, val if v*10 < 9*2^64, else test",
                        "counts": counts, "pages": splits});
        std::fs::write(self.dir.join("splits.json"), format!("{}\n", serde_json::to_string_pretty(&sp)?))?;
        let created = created_now();
        let run = json!({"created": created, "rfpii": crate::process::tool_info(), "parameters": parameters, "pages_added": n_added});
        let mut runs: Vec<Value> = self.meta.as_ref().and_then(|m| m["runs"].as_array().cloned()).unwrap_or_default();
        runs.push(run.clone());
        let mut all_notes: Vec<Value> = self.meta.as_ref().and_then(|m| m["notes"].as_array().cloned()).unwrap_or_default();
        for n in notes {
            if !all_notes.iter().any(|x| x.as_str() == Some(n.as_str())) {
                all_notes.push(json!(n));
            }
        }
        let meta = json!({
            "schema": DATASET_SCHEMA,
            "kind": self.kind,
            "created": self.meta.as_ref().map(|m| m["created"].clone()).unwrap_or(json!(created)),
            "updated": created,
            "rfpii": crate::process::tool_info(),
            "parameters": parameters,
            "personal_data": personal_data,
            "ink": ink,
            "mask_bits": mask_bits(),
            "classes": CLASSES,
            "pages": lines.len(),
            "splits": counts,
            "runs": runs,
            "notes": all_notes,
        });
        std::fs::write(self.dir.join("dataset.json"), format!("{}\n", serde_json::to_string_pretty(&meta)?))?;
        write_sums(&self.dir)?;
        Ok(meta)
    }
}

pub fn read_index(dir: &Path) -> Result<Vec<Value>> {
    let text = std::fs::read_to_string(dir.join("index.jsonl")).with_context(|| format!("reading {}/index.jsonl", dir.display()))?;
    text.lines().filter(|l| !l.trim().is_empty()).map(|l| serde_json::from_str(l).map_err(Into::into)).collect()
}

/// SHA256SUMS over every file of the dataset but itself, sorted by path.
pub fn write_sums(dir: &Path) -> Result<()> {
    let mut sums = String::new();
    for f in crate::annex::files_under(dir, "") {
        let rel = f.strip_prefix(dir).unwrap_or(&f).to_string_lossy().into_owned();
        if rel == "SHA256SUMS" {
            continue;
        }
        sums += &format!("{}  {}\n", sha256_hex(&std::fs::read(&f)?), rel);
    }
    std::fs::write(dir.join("SHA256SUMS"), sums)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splits_are_fixed_by_the_key_and_near_80_10_10() {
        assert_eq!(split_of("county:047/constituency:281"), split_of("county:047/constituency:281"));
        let mut n = std::collections::HashMap::new();
        for k in 0..10_000 {
            *n.entry(split_of(&format!("seed:{}", k))).or_insert(0) += 1;
        }
        assert!((7700..8300).contains(&n["train"]), "{:?}", n);
        assert!((800..1200).contains(&n["val"]) && (800..1200).contains(&n["test"]), "{:?}", n);
    }

    #[test]
    fn boxes_meet_and_clip() {
        assert!((iou([0.0, 0.0, 10.0, 10.0], [5.0, 0.0, 15.0, 10.0]) - 1.0 / 3.0).abs() < 1e-12);
        assert_eq!(iou([0.0, 0.0, 1.0, 1.0], [2.0, 2.0, 3.0, 3.0]), 0.0);
        assert_eq!(clip([-5.0, 2.0, 50.0, 8.0], 40.0, 10.0), Some([0.0, 2.0, 40.0, 8.0]));
        assert_eq!(clip([45.0, 2.0, 50.0, 8.0], 40.0, 10.0), None);
        assert_eq!(bounds(&[[3.0, 4.0], [1.0, 9.0]]), [1.0, 4.0, 3.0, 9.0]);
    }

    #[test]
    fn a_writer_refuses_a_full_folder_unless_appending() {
        let d = crate::testutil::tmpdir("writer");
        std::fs::write(d.join("x"), "x").unwrap();
        assert!(Writer::open(&d, "synthetic", false).is_err());
        let d2 = crate::testutil::tmpdir("writer2");
        let w = Writer::open(&d2, "synthetic", false).unwrap();
        let labels = json!({"schema": LABELS_SCHEMA, "split_key": "seed:1", "page": {"kind": "blank", "held": false}, "boxes": []});
        let line = write_page(&d2, &PageOut { id: "a".into(), image: vec![1, 2, 3], ext: "png", mask: None, labels }).unwrap();
        w.finish(vec![line], json!({}), false, "exact", vec![]).unwrap();
        let w = Writer::open(&d2, "synthetic", true).unwrap();
        assert!(w.has("a") && !w.has("b"));
        assert!(Writer::open(&d2, "teacher", true).is_err());
        let sums = std::fs::read_to_string(d2.join("SHA256SUMS")).unwrap();
        assert!(sums.contains("  pages/a.png\n") && sums.contains("  dataset.json\n") && !sums.contains("SHA256SUMS"));
        let _ = std::fs::remove_dir_all(&d);
        let _ = std::fs::remove_dir_all(&d2);
    }
}
