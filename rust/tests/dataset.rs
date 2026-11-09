//! `rfpii dataset` end to end: synthetic forms made twice from one seed are
//! the same files; a teacher dataset built from redacting them is checked
//! against the synthetic truth; redacted synthetic pages say what shows.

use serde_json::Value;
use std::path::{Path, PathBuf};
use std::process::Command;

fn tmpdir(name: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("rfpii-dataset-{}-{}", std::process::id(), name));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn templates() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../templates")
}

fn rfpii(args: &[&str]) -> (i32, String) {
    let out = Command::new(env!("CARGO_BIN_EXE_rfpii")).args(args).env("SOURCE_DATE_EPOCH", "1786000000").output().unwrap();
    let text = format!("{}{}", String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr));
    (out.status.code().unwrap_or(-1), text)
}

fn s(p: &Path) -> &str {
    p.to_str().unwrap()
}

fn files(dir: &Path) -> Vec<(String, Vec<u8>)> {
    let mut out: Vec<(String, Vec<u8>)> = rfpii::annex::files_under(dir, "").into_iter().map(|f| (f.strip_prefix(dir).unwrap().to_string_lossy().into_owned(), std::fs::read(&f).unwrap())).collect();
    out.sort();
    out
}

#[test]
fn a_seed_makes_the_same_dataset_and_check_finds_tampering() {
    let d = tmpdir("same");
    let t = templates();
    let args = |out: &Path| vec!["dataset".to_string(), "synth".into(), "--templates".into(), s(&t).into(), "--out".into(), s(out).into(), "--count".into(), "5".into(), "--seed".into(), "40".into(), "--max-side".into(), "900".into(), "--workers".into(), "2".into()];
    for name in ["a", "b"] {
        let a = args(&d.join(name));
        let (rc, text) = rfpii(&a.iter().map(String::as_str).collect::<Vec<_>>());
        assert_eq!(rc, 0, "{text}");
    }
    let (fa, fb) = (files(&d.join("a")), files(&d.join("b")));
    assert_eq!(fa.len(), fb.len());
    assert!(fa.len() >= 5 * 3 + 4);
    for (x, y) in fa.iter().zip(&fb) {
        assert!(x == y, "{} differs", x.0);
    }
    let (rc, text) = rfpii(&["dataset", "check", s(&d.join("a"))]);
    assert_eq!(rc, 0, "{text}");
    // appending the same seeds adds nothing; new seeds add pages
    let (rc, text) = rfpii(&["dataset", "synth", "--templates", s(&t), "--out", s(&d.join("a")), "--count", "7", "--seed", "40", "--max-side", "900", "--append"]);
    assert_eq!(rc, 0, "{text}");
    let index = std::fs::read_to_string(d.join("a/index.jsonl")).unwrap();
    assert_eq!(index.lines().count(), 7);
    assert!(rfpii(&["dataset", "synth", "--templates", s(&t), "--out", s(&d.join("a")), "--count", "1"]).0 != 0, "a full folder needs --append");
    let (rc, text) = rfpii(&["dataset", "check", s(&d.join("a"))]);
    assert_eq!(rc, 0, "{text}");
    // a label moved, or a mask swapped, is found
    let lab = d.join("b/labels/syn-0000000040.json");
    let mut v: Value = serde_json::from_str(&std::fs::read_to_string(&lab).unwrap()).unwrap();
    v["boxes"][0]["box"][2] = serde_json::json!(1e6);
    std::fs::write(&lab, serde_json::to_string(&v).unwrap()).unwrap();
    let (rc, text) = rfpii(&["dataset", "check", s(&d.join("b"))]);
    assert_ne!(rc, 0);
    assert!(text.contains("SHA-256 differs") && text.contains("outside"), "{text}");
    let (rc, text) = rfpii(&["dataset", "stats", s(&d.join("a")), "--json"]);
    assert_eq!(rc, 0, "{text}");
    let st: Value = serde_json::from_str(&text).unwrap();
    assert_eq!(st["pages"], 7);
    let _ = std::fs::remove_dir_all(&d);
}

/// The teacher's labels, from redacting synthetic Forms 34B, against the
/// truth those forms were drawn with.
#[test]
fn the_teacher_labels_redacted_synthetic_forms_close_to_the_truth() {
    let d = tmpdir("teacher");
    let t = templates();
    let truth = d.join("truth");
    let (rc, text) = rfpii(&["dataset", "synth", "--templates", s(&t), "--out", s(&truth), "--count", "4", "--seed", "1000", "--profile", "clean", "--negatives", "0", "--templates-only", "form34b_2022_block,form34b_2022_block_b"]);
    assert_eq!(rc, 0, "{text}");
    let (rc, text) = rfpii(&["redact", s(&truth.join("pages")), "--templates", s(&t), "--out", s(&d.join("out")), "--review", s(&d.join("review")), "--workers", "2", "--no-pdf-signature"]);
    assert!(rc == 0 || rc == 2, "{text}");
    let teacher = d.join("teacher");
    let (rc, text) = rfpii(&["dataset", "build", s(&d.join("out")), "--originals", s(&truth.join("pages")), "--templates", s(&t), "--review", s(&d.join("review")), "--out", s(&teacher)]);
    assert_eq!(rc, 0, "{text}");
    let (rc, text) = rfpii(&["dataset", "check", s(&teacher)]);
    assert_eq!(rc, 0, "{text}");
    let meta: Value = serde_json::from_str(&std::fs::read_to_string(teacher.join("dataset.json")).unwrap()).unwrap();
    assert_eq!(meta["kind"], "teacher");
    assert_eq!(meta["personal_data"], true);
    assert_eq!(meta["pages"], 4, "every page, published or held");
    // the same image bytes as the source
    let ti = std::fs::read_to_string(truth.join("index.jsonl")).unwrap();
    let oi = std::fs::read_to_string(teacher.join("index.jsonl")).unwrap();
    for l in ti.lines() {
        let v: Value = serde_json::from_str(l).unwrap();
        assert!(oi.contains(v["image_sha256"].as_str().unwrap()));
    }
    let (rc, text) = rfpii(&["dataset", "compare", s(&truth), s(&teacher), "--json", "--iou", "0.3"]);
    assert_eq!(rc, 0, "{text}");
    let r: Value = serde_json::from_str(&text).unwrap();
    let c = &r["classes"];
    let published = r["pages_in_other"].as_u64().unwrap() - r["held_in_other"].as_u64().unwrap();
    assert!(published >= 3, "{}", text);
    assert!(c["agents_table"]["mean_best_iou"].as_f64().unwrap() > 0.95, "{}", text);
    assert!(c["id_number"]["recall"].as_f64().unwrap() >= 0.7, "{}", text);
    assert!(c["phone_number"]["recall"].as_f64().unwrap() >= 0.7, "{}", text);
    assert!(c["signature"]["recall"].as_f64().unwrap() >= 0.6, "{}", text);
    assert!(r["cells"]["both_written"].as_u64().unwrap() > 10 && r["cells"]["both_empty"].as_u64().unwrap() > 5, "{}", text);
    let _ = std::fs::remove_dir_all(&d);
}

#[test]
fn redacted_synthetic_pages_say_what_each_field_shows() {
    let d = tmpdir("redacted");
    let (rc, text) = rfpii(&["dataset", "synth", "--templates", s(&templates()), "--out", s(&d.join("r")), "--count", "3", "--seed", "2002", "--negatives", "0", "--redacted", "--templates-only", "form34b_2022_block", "--max-side", "1400"]);
    assert_eq!(rc, 0, "{text}");
    let (rc, text) = rfpii(&["dataset", "check", s(&d.join("r"))]);
    assert_eq!(rc, 0, "{text}");
    let index = std::fs::read_to_string(d.join("r/index.jsonl")).unwrap();
    let (mut ok, mut leaks, mut black) = (0, 0, 0);
    for l in index.lines() {
        let v: Value = serde_json::from_str(l).unwrap();
        let lab: Value = serde_json::from_str(&std::fs::read_to_string(d.join("r").join(v["labels"].as_str().unwrap())).unwrap()).unwrap();
        assert!(lab["redaction"]["blocks_blacked"].as_u64().unwrap() > 0);
        for b in lab["boxes"].as_array().unwrap() {
            match b["redaction"]["status"].as_str() {
                Some("masked_ok") => ok += 1,
                Some(_) => leaks += 1,
                None => assert!(b["empty"] == true || !["signature", "id_number", "phone_number"].contains(&b["class"].as_str().unwrap()), "{b}"),
            }
        }
        let mask = std::fs::read(d.join("r").join(v["mask"].as_str().unwrap())).unwrap();
        let m = rfpii::io::decode_bytes(&mask).unwrap();
        black += rfpii::util::bytes(&rfpii::util::gray(&m).unwrap()).unwrap().iter().filter(|&&x| x & 8 != 0).count();
    }
    assert!(ok > 5 && black > 1000, "{ok} {leaks} {black}");
    let _ = std::fs::remove_dir_all(&d);
}
