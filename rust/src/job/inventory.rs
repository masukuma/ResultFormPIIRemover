//! What a job is given, before anything is redacted (HARNESS.md §4.1): every
//! file, zips expanded, with its hash, pages and kind; the duplicates and
//! the files that can't be read; and the groups the pages fall into.

use anyhow::Result;
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::path::Path;

use crate::io;

pub const SCHEMA: &str = "rfpii/inventory/1";

/// What kind of file a source is: a KIEMS transmission (by its name), a
/// photo (an image file), a scan (a PDF of several pages) or a PDF of one
/// page; with the form its name gives, if any.
pub fn kind(src: &str, rel: &str, pages: usize) -> (&'static str, Option<String>) {
    let name = src.rsplit(io::ZIP_SEP).next().unwrap_or(src);
    let ext = Path::new(name).extension().map(|e| e.to_string_lossy().to_lowercase()).unwrap_or_default();
    let stem = Path::new(rel).file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
    if ext != "pdf" {
        return ("photo", None);
    }
    match crate::kiems::parse_name(&stem) {
        Some(n) => ("kiems", Some(n.form)),
        None if pages > 1 => ("scan", None),
        None => ("pdf", None),
    }
}

/// The group a file's pages go to: its kind, and its form where the name
/// gives one ("kiems-34A").
pub fn group_id(kind: &str, form: Option<&str>) -> String {
    match form {
        Some(f) => format!("{}-{}", kind, f),
        None => kind.to_string(),
    }
}

/// The folder a file is counted under: its path's folder, "." at the top.
pub fn folder(rel: &str) -> String {
    match Path::new(rel).parent().map(|p| p.to_string_lossy().into_owned()) {
        Some(p) if !p.is_empty() => p,
        _ => ".".into(),
    }
}

/// One file: its record, with the reason it can't be read if it can't.
fn file_record(src: &str, rel: &str) -> Value {
    let mut rec = json!({"source": src, "path": rel, "folder": folder(rel)});
    let data = match io::read_bytes(src) {
        Ok(d) => d,
        Err(e) => {
            rec["broken"] = json!(format!("{:#}", e));
            return rec;
        }
    };
    rec["sha256"] = json!(crate::proof::sha256(&data));
    rec["bytes"] = json!(data.len());
    let pages = match io::page_count(src, &data) {
        Ok(0) => {
            rec["broken"] = json!("no pages can be read");
            0
        }
        Ok(n) => n,
        Err(e) => {
            rec["broken"] = json!(format!("{:#}", e));
            0
        }
    };
    let (k, form) = kind(src, rel, pages);
    // a PDF that opens only repaired, as MuPDF repairs it: often a truncated
    // download ("cannot find startxref"), its last pages lost or blank
    if pages > 0 && k != "photo" && lopdf::Document::load_mem(&data).is_err() {
        rec["repaired"] = json!(true);
    }
    rec["pages"] = json!(pages);
    rec["kind"] = json!(k);
    rec["form"] = json!(form);
    rec["group"] = json!(group_id(k, form.as_deref()));
    rec
}

/// The inventory of `inputs` (files, folders, zips): files in the order
/// `redact` lists them; the second and later copies of a file marked
/// `duplicate_of` the first; the files no page of which can be read
/// (`broken`), and the PDFs that open only repaired; groups of the files
/// that can be read, each file counted once, with pages per folder.
pub fn inventory(inputs: &[String]) -> Result<Value> {
    let mut files: Vec<Value> = io::sources(inputs, None)?.iter().map(|(src, rel)| file_record(src, rel)).collect();
    let mut first: BTreeMap<String, String> = BTreeMap::new();
    for f in files.iter_mut().filter(|f| f["sha256"].is_string()) {
        let sha = f["sha256"].as_str().unwrap_or("").to_string();
        match first.get(&sha) {
            Some(rel) => f["duplicate_of"] = json!(rel),
            None => {
                first.insert(sha, f["path"].as_str().unwrap_or("").to_string());
            }
        }
    }
    let mut dups: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for f in &files {
        if let Some(orig) = f["duplicate_of"].as_str() {
            dups.entry(orig.to_string()).or_default().push(f["path"].as_str().unwrap_or("").to_string());
        }
    }
    let broken: Vec<Value> = files.iter().filter(|f| f.get("broken").is_some()).map(|f| json!({"path": f["path"], "reason": f["broken"]})).collect();
    let repaired: Vec<Value> = files.iter().filter(|f| f.get("repaired").is_some()).map(|f| json!({"path": f["path"], "pages": f["pages"]})).collect();
    let mut groups: BTreeMap<String, Value> = BTreeMap::new();
    for f in files.iter().filter(|f| f.get("broken").is_none() && f.get("duplicate_of").is_none()) {
        let g = groups.entry(f["group"].as_str().unwrap_or("").to_string()).or_insert_with(|| json!({"id": f["group"], "kind": f["kind"], "form": f["form"], "files": 0, "pages": 0, "folders": {}}));
        let n = f["pages"].as_u64().unwrap_or(0);
        g["files"] = json!(g["files"].as_u64().unwrap_or(0) + 1);
        g["pages"] = json!(g["pages"].as_u64().unwrap_or(0) + n);
        let fo = f["folder"].as_str().unwrap_or(".").to_string();
        g["folders"][&fo] = json!(g["folders"][&fo].as_u64().unwrap_or(0) + n);
    }
    let pages: u64 = groups.values().map(|g| g["pages"].as_u64().unwrap_or(0)).sum();
    Ok(json!({
        "schema": SCHEMA,
        "inputs": inputs,
        "totals": {"files": files.len(), "pages": pages, "duplicates": files.iter().filter(|f| f.get("duplicate_of").is_some()).count(), "broken": broken.len(), "repaired": repaired.len()},
        "groups": groups.into_values().collect::<Vec<_>>(),
        "duplicates": dups.into_iter().map(|(k, v)| json!({"kept": k, "copies": v})).collect::<Vec<_>>(),
        "broken": broken,
        "repaired": repaired,
        "files": files,
    }))
}

/// A few lines for people: totals, groups, duplicates, broken files.
pub fn summary(inv: &Value) -> String {
    let t = &inv["totals"];
    let mut s = format!("{} files, {} pages to redact; {} duplicate(s), {} broken, {} opened only repaired\n", t["files"], t["pages"], t["duplicates"], t["broken"], t["repaired"]);
    for g in inv["groups"].as_array().into_iter().flatten() {
        s += &format!("group {}: {} files, {} pages in {} folder(s)\n", g["id"].as_str().unwrap_or(""), g["files"], g["pages"], g["folders"].as_object().map(|o| o.len()).unwrap_or(0));
    }
    for d in inv["duplicates"].as_array().into_iter().flatten() {
        s += &format!("duplicate: {} (kept) = {}\n", d["kept"].as_str().unwrap_or(""), d["copies"].as_array().into_iter().flatten().filter_map(|c| c.as_str()).collect::<Vec<_>>().join(", "));
    }
    for b in inv["broken"].as_array().into_iter().flatten() {
        s += &format!("broken: {}: {}\n", b["path"].as_str().unwrap_or(""), b["reason"].as_str().unwrap_or(""));
    }
    for r in inv["repaired"].as_array().into_iter().flatten() {
        s += &format!("repaired: {} ({} pages read; truncated?)\n", r["path"].as_str().unwrap_or(""), r["pages"]);
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn files_are_told_apart_by_name_pages_and_type() {
        let k = "1_34_A_007029014000102_F230450M00203609_20220810_132155";
        assert_eq!(kind(&format!("x/{}.PDF", k), &format!("{}.PDF", k), 1), ("kiems", Some("34A".into())));
        assert_eq!(kind("a.zip::scan.pdf", "a/scan.pdf", 40), ("scan", None));
        assert_eq!(kind("one.pdf", "one.pdf", 1), ("pdf", None));
        assert_eq!(kind("p.JPG", "p.JPG", 1), ("photo", None));
        assert_eq!(group_id("kiems", Some("34B")), "kiems-34B");
        assert_eq!((folder("Nyeri/Othaya/a.pdf"), folder("a.pdf")), ("Nyeri/Othaya".to_string(), ".".to_string()));
    }

    #[test]
    fn copies_and_unreadable_files_are_found_and_kept_out_of_the_groups() {
        let d = crate::testutil::tmpdir("inventory");
        let img = crate::io::encode(&crate::testutil::gray(40, 30, 200), "jpg", 90).unwrap();
        std::fs::create_dir_all(d.join("A")).unwrap();
        std::fs::create_dir_all(d.join("B")).unwrap();
        std::fs::write(d.join("A/p.jpg"), &img).unwrap();
        std::fs::write(d.join("B/p.jpg"), &img).unwrap();
        std::fs::write(d.join("B/cut.pdf"), b"%PDF-1.4\n1 0 obj\n<<").unwrap();
        let inv = inventory(&[d.to_string_lossy().into_owned()]).unwrap();
        assert_eq!(inv["totals"], json!({"files": 3, "pages": 1, "duplicates": 1, "broken": 1, "repaired": 0}));
        assert_eq!(inv["duplicates"], json!([{"kept": "A/p.jpg", "copies": ["B/p.jpg"]}]));
        assert_eq!(inv["broken"][0]["path"], "B/cut.pdf");
        assert_eq!(inv["groups"], json!([{"id": "photo", "kind": "photo", "form": null, "files": 1, "pages": 1, "folders": {"A": 1}}]));
    }
}
