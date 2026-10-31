//! The harness's tools and `rfpii job`, with the `rfpii` binary, on
//! synthetic forms (and a stub model server, which needs no network).

use opencv::prelude::*;
use rfpii::io::{self, PageImage};
use rfpii::synthetic::{filled_form, simulate_scan, write_template};
use serde_json::{json, Value};
use std::path::{Path, PathBuf};
use std::process::Command;

fn tmpdir(name: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("rfpii-job-{}-{}", std::process::id(), name));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

/// Run `rfpii`: its exit code, and what it printed on stdout and stderr.
fn run(args: &[&str]) -> (i32, String, String) {
    let out = Command::new(env!("CARGO_BIN_EXE_rfpii")).args(args).env("RFPII_SIGN_PASSWORD", "test-sign-password").env_remove("OLLAMA_API_KEY").output().unwrap();
    let (stdout, stderr) = (String::from_utf8_lossy(&out.stdout).into_owned(), String::from_utf8_lossy(&out.stderr).into_owned());
    if !out.status.success() {
        eprintln!("rfpii {}\n{}\n{}", args.join(" "), stdout, stderr);
    }
    (out.status.code().unwrap_or(-1), stdout, stderr)
}

/// Run `rfpii`: its exit code and all it printed.
fn rfpii(args: &[&str]) -> (i32, String) {
    let (rc, out, err) = run(args);
    (rc, out + &err)
}

/// Run `rfpii`, which must succeed and print JSON.
fn json_of(args: &[&str]) -> Value {
    let (rc, out, _) = run(args);
    assert_eq!(rc, 0);
    serde_json::from_str(out.trim()).unwrap()
}

fn s(p: &Path) -> &str {
    p.to_str().unwrap()
}

fn read(p: &Path) -> Value {
    serde_json::from_str(&std::fs::read_to_string(p).unwrap_or_else(|e| panic!("{}: {}", p.display(), e))).unwrap()
}

/// A scan of synthetic form `k`, JPEG-encoded.
fn form_jpeg(k: u64, quality: i32) -> Vec<u8> {
    let (scan, _) = simulate_scan(&filled_form(k).unwrap(), k, 2.0, 1.3).unwrap();
    io::encode(&scan, "jpg", quality).unwrap()
}

fn blank_jpeg() -> Vec<u8> {
    let white = opencv::core::Mat::new_rows_cols_with_default(2280, 1612, opencv::core::CV_8UC3, opencv::core::Scalar::all(255.0)).unwrap();
    io::encode(&white, "jpg", 80).unwrap()
}

/// A PDF of the JPEG pages given.
fn pdf_of(pages: &[&[u8]]) -> Vec<u8> {
    let items: Vec<_> = pages.iter().map(|p| (PageImage::Jpeg(p), Some((1612.0, 2280.0)), None, 0)).collect();
    io::build_pdf(&items).unwrap()
}

#[test]
fn redact_resumes_takes_pages_and_masks_spots_it_replays() {
    let root = tmpdir("flags");
    write_template(&root.join("templates/sample_rs1")).unwrap();
    std::fs::create_dir_all(root.join("in/Nyeri")).unwrap();
    let (a, b, c) = (form_jpeg(4, 80), blank_jpeg(), form_jpeg(5, 85));
    let orig = root.join("in/Nyeri/batch.pdf");
    std::fs::write(&orig, pdf_of(&[&a, &b, &c])).unwrap();
    let (out, review, input, templates) = (root.join("out"), root.join("review"), root.join("in"), root.join("templates"));
    let redact = |extra: &[&str]| {
        let mut args = vec!["redact", s(&orig), "--base", s(&input), "--templates", s(&templates), "--out", s(&out), "--review", s(&review), "--workers", "2", "--no-pdf-signature", "--per-page"];
        args.extend_from_slice(extra);
        rfpii(&args).0
    };
    // two of the three pages, in the folder a run over in/ would use
    assert_eq!(redact(&["--pages", "1,3"]), 0);
    let p = |k: usize| out.join(format!("Nyeri/batch.p00{}.pdf", k));
    assert!(p(1).exists() && p(3).exists() && !p(2).exists());
    let before = std::fs::read(p(1).with_extension("manifest.json")).unwrap();
    // resumed: only the page not done yet is tried (and held)
    assert_eq!(redact(&["--resume"]), 2);
    assert_eq!(std::fs::read(p(1).with_extension("manifest.json")).unwrap(), before);
    assert!(review.join("Nyeri/batch.p002.json").exists());
    // a spot mask on page 3: that page alone is redone, and records it
    let m3 = read(&p(3).with_extension("manifest.json"));
    let sha = m3["source"]["sha256"].as_str().unwrap();
    // in the margin below the form, where no rule masks anything
    let spot = json!([{"sha256": sha, "page": 3, "label": "signature", "box": [300.0, 2140.0, 520.0, 2200.0], "source": {"finding": "f-1"}}]);
    std::fs::write(root.join("spots.json"), spot.to_string()).unwrap();
    assert_eq!(redact(&["--resume", "--pages", "1,3", "--spot-masks", s(&root.join("spots.json"))]), 0);
    assert_eq!(std::fs::read(p(1).with_extension("manifest.json")).unwrap(), before);
    let m3 = read(&p(3).with_extension("manifest.json"));
    assert_eq!(m3["pages"][0]["spot_masks"][0]["source"]["finding"], "f-1");
    let img = io::decode_bytes(&io::pdf_images(&std::fs::read(p(3)).unwrap()).unwrap().remove(0).0).unwrap();
    let px = img.at_2d::<opencv::core::Vec3b>(2170, 410).unwrap();
    assert!(px[0] < 20 && px[1] < 20 && px[2] < 20, "{:?}", px);
    // the re-run applies it from the manifest
    let (rc, text) = rfpii(&["verify", s(&p(3)), "--original", s(&orig), "--rerun", s(&root.join("templates"))]);
    assert!(rc == 0 && text.contains("reproduces the same output"), "{}", text);
    // a malformed spot is refused before anything runs
    std::fs::write(root.join("bad.json"), json!([{"sha256": sha, "page": 3, "label": "name", "box": [0, 0, 1, 1], "source": "x"}]).to_string()).unwrap();
    assert_eq!(redact(&["--spot-masks", s(&root.join("bad.json"))]), 1);
}

#[test]
fn the_tools_count_rank_and_show_only_what_was_published() {
    let root = tmpdir("tools");
    write_template(&root.join("templates/sample_rs1")).unwrap();
    for (dir, k) in [("A", 1u64), ("A", 2), ("B", 3)] {
        std::fs::create_dir_all(root.join("in").join(dir)).unwrap();
        std::fs::write(root.join("in").join(dir).join(format!("f{}.jpg", k)), form_jpeg(k, 90)).unwrap();
    }
    std::fs::copy(root.join("in/A/f1.jpg"), root.join("in/B/copy.jpg")).unwrap();
    std::fs::write(root.join("in/B/blank.jpg"), blank_jpeg()).unwrap();
    std::fs::write(root.join("in/B/cut.pdf"), &pdf_of(&[&blank_jpeg()])[..300]).unwrap();
    let inv = json_of(&["inventory", s(&root.join("in")), "--json"]);
    assert_eq!(inv["totals"]["duplicates"], 1);
    // a truncated download opens only repaired
    assert_eq!(inv["repaired"], json!([{"path": "B/cut.pdf", "pages": 1}]));
    assert_eq!(inv["groups"][1]["folders"], json!({"A": 2, "B": 2}));

    let (out, review) = (root.join("out"), root.join("review"));
    let (rc, _) = rfpii(&["redact", s(&root.join("in")), "--templates", s(&root.join("templates")), "--out", s(&out), "--review", s(&review), "--workers", "2", "--no-pdf-signature"]);
    assert_eq!(rc, 2);
    let st = json_of(&["stats", s(&out), "--review", s(&review), "--risk", "--json"]);
    assert_eq!((st["documents_published"].as_u64(), st["documents_held"].as_u64()), (Some(4), Some(2)));
    assert_eq!(st["risk"].as_array().unwrap().len(), 4);

    // a crop of what was published, verified first
    let manifest = out.join("A/f2.manifest.json");
    let c = json_of(&["crop", s(&manifest), "--region", "declaration", "--templates", s(&root.join("templates")), "--out", s(&root.join("c.png"))]);
    assert_eq!(c["sha256"], rfpii::proof::sha256(&std::fs::read(root.join("c.png")).unwrap()));
    // never an original, nor a held page's report
    for refused in [root.join("in/A/f2.jpg"), review.join("B/blank.json")] {
        let (rc, text) = rfpii(&["crop", s(&refused)]);
        assert!(rc != 0 && text.contains("refusing"), "{}", text);
    }
    // a published page changed after the fact fails verify, and is refused
    let pdf = out.join("A/f2.pdf");
    let mut bytes = std::fs::read(&pdf).unwrap();
    bytes.extend_from_slice(b"\n% appended\n");
    std::fs::write(&pdf, bytes).unwrap();
    let (rc, text) = rfpii(&["crop", s(&manifest)]);
    assert!(rc != 0 && text.contains("verify fails"), "{}", text);

    // a held page is shown only in a job that allows it
    let blank = s(&root.join("in/B/blank.jpg")).to_string();
    let (rc, text) = rfpii(&["held-view", &blank, "--mode", "triage", "--job", s(&root)]);
    assert!(rc != 0 && text.contains("not a job folder"), "{}", text);
    // its diagnostics are numbers and printed words
    let d = json_of(&["diagnose", &blank, "--templates", s(&root.join("templates")), "--json"]);
    assert_eq!((d["ink"].as_f64(), d["read"]["ok"].as_bool()), (Some(0.0), Some(false)));
}
