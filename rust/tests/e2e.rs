//! Redact KIEMS-style PDFs with the `rfpii` binary, then verify them the way
//! a scrutineer would.

use lopdf::{dictionary, Document, Object, Stream};
use opencv::prelude::*;
use rfpii::io::{self, PageImage};
use rfpii::synthetic::{filled_form, simulate_scan, write_template};
use rfpii::verify::{verify, Report, VerifyArgs};
use serde_json::Value;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::OnceLock;

const NAME: &str = "1_34_A_007029014000102_F230450M00203609_20220810_132155.PDF";

fn tmpdir(name: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("rfpii-e2e-{}-{}", std::process::id(), name));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn rfpii(args: &[&str]) -> i32 {
    let out = Command::new(env!("CARGO_BIN_EXE_rfpii")).args(args).env("RFPII_CERT_PASSWORD", "test-password").output().unwrap();
    if !out.status.success() {
        eprintln!("{}\n{}", String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr));
    }
    out.status.code().unwrap_or(-1)
}

fn s(p: &Path) -> &str {
    p.to_str().unwrap()
}

/// A one-page PDF holding `img` (an encoded image) as its page.
fn image_pdf(img: PageImage, w: f64, h: f64) -> Vec<u8> {
    io::build_pdf(&[(img, Some((w, h)), None, 0)]).unwrap()
}

/// A KIEMS-named, JPEG-in-PDF form redacted with a signing key and a
/// certificate (the fixture all of the first tests share).
fn run() -> &'static Path {
    static ROOT: OnceLock<PathBuf> = OnceLock::new();
    ROOT.get_or_init(|| {
        let root = tmpdir("kiems");
        write_template(&root.join("templates").join("sample_rs1")).unwrap();
        let (scan, _) = simulate_scan(&filled_form(4).unwrap(), 4, 2.0, 1.3).unwrap();
        let jpg = io::encode(&scan, "jpg", 80).unwrap();
        std::fs::create_dir_all(root.join("in")).unwrap();
        std::fs::write(root.join("in").join(NAME), image_pdf(PageImage::Jpeg(&jpg), 1612.0, 2280.0)).unwrap();
        assert_eq!(rfpii(&["keygen", s(&root.join("keys"))]), 0);
        assert_eq!(rfpii(&["certgen", "--name", "Test Officer", "--country", "KE", "--out", s(&root.join("keys/signer.p12"))]), 0);
        let rc = rfpii(&[
            "redact", s(&root.join("in")), "--templates", s(&root.join("templates")), "--out", s(&root.join("out")),
            "--review", s(&root.join("review")), "--workers", "1", "--sign-key", s(&root.join("keys/signing_key.pem")),
            "--certificate", s(&root.join("keys/signer.p12")),
        ]);
        assert_eq!(rc, 0, "{}", std::fs::read_to_string(root.join("review/report.jsonl")).unwrap_or_default());
        root
    })
}

fn published(root: &Path) -> PathBuf {
    root.join("out").join(NAME.replace(".PDF", ".pdf"))
}

fn check(root: &Path, redacted: &Path, manifest: Option<&Path>, original: &Path, trust: &[PathBuf], overlay: Option<&Path>, rerun: Option<&Path>) -> Report {
    verify(&VerifyArgs {
        redacted,
        manifest,
        original: Some(original),
        pubkey: Some(&root.join("keys/signing_key.pub.pem")),
        ledger: Some(&root.join("out/ledger.jsonl")),
        overlay_dir: overlay,
        rerun_templates: rerun,
        trust,
    })
    .unwrap()
}

fn manifest(root: &Path) -> Value {
    serde_json::from_str(&std::fs::read_to_string(root.join("out").join(NAME.replace(".PDF", ".manifest.json"))).unwrap()).unwrap()
}

#[test]
fn published_form_verifies() {
    let root = run();
    let r = check(root, &published(root), None, &root.join("in").join(NAME), &[root.join("keys/signer.crt")], Some(&root.join("overlay")), Some(&root.join("templates")));
    assert!(r.ok(), "{}", r.text());
    let m = manifest(root);
    assert!(m["pages"][0]["proof"] == "jpeg-dct" && m["pages"][0]["blocks_blacked"].as_u64().unwrap() > 0);
    assert_eq!(m["source"]["kiems"]["station"], "007029014000102");
    assert_eq!(m["output"]["pdf_signature"]["subject"], "C=KE, CN=Test Officer");
    assert!(std::fs::read_dir(root.join("overlay")).unwrap().any(|e| e.unwrap().path().extension().map(|x| x == "png").unwrap_or(false)));
    // the re-run's copy of the form is not left behind
    assert!(std::fs::read_dir(root.join("overlay")).unwrap().all(|e| e.unwrap().path().is_file()));
    assert!(!std::env::temp_dir().join(format!("rfpii-rerun-{}", std::process::id())).exists());
    let batch = std::fs::read_dir(root.join("out/batches")).unwrap().next().unwrap().unwrap().path();
    let b: Value = serde_json::from_str(&std::fs::read_to_string(batch).unwrap()).unwrap();
    assert_eq!(b["count"], 1);
    for (ok, msg) in &r.checks {
        if msg.contains("digital signature") {
            assert!(ok);
        }
    }
}

#[test]
fn altered_result_is_detected() {
    let root = run();
    let pdf = std::fs::read(published(root)).unwrap();
    let jpg = io::pdf_images(&pdf).unwrap().remove(0).0;
    let bad = rfpii::proof::nudge_jpeg(&jpg, 0, 90, 60, 0, 4).unwrap(); // results table area: brighten one block slightly
    let sp = io::source_pages(s(&root.join("in").join(NAME)), None).unwrap().remove(0);
    let d = tmpdir("tampered");
    let tampered = d.join("tampered.pdf");
    std::fs::write(&tampered, io::build_pdf(&[(PageImage::Jpeg(&bad), sp.page_size, sp.place, sp.rotate)]).unwrap()).unwrap();
    let mpath = root.join("out").join(NAME.replace(".PDF", ".manifest.json"));
    let r = check(root, &tampered, Some(&mpath), &root.join("in").join(NAME), &[], None, None);
    let failed: Vec<&String> = r.checks.iter().filter(|(ok, _)| !ok).map(|(_, m)| m).collect();
    assert!(!r.ok() && failed.iter().any(|m| m.contains("other than black")), "{}", r.text());
}

#[test]
fn bytes_added_after_signing_are_detected() {
    let root = run();
    let mut pdf = std::fs::read(published(root)).unwrap();
    pdf.extend_from_slice(b"\n% appended\n");
    let d = tmpdir("appended");
    let p = d.join("appended.pdf");
    std::fs::write(&p, pdf).unwrap();
    let mpath = root.join("out").join(NAME.replace(".PDF", ".manifest.json"));
    let r = check(root, &p, Some(&mpath), &root.join("in").join(NAME), &[], None, None);
    assert!(!r.ok() && r.checks.iter().any(|(ok, m)| !ok && m.contains("digital signature")), "{}", r.text());
}

#[test]
fn wrong_original_and_forged_manifest() {
    let root = run();
    let d = tmpdir("forged");
    let other = d.join(NAME);
    let mut data = std::fs::read(root.join("in").join(NAME)).unwrap();
    data.push(b'\n');
    std::fs::write(&other, data).unwrap();
    assert!(!check(root, &published(root), None, &other, &[], None, None).ok());
    let mut m = manifest(root);
    m["pages"][0]["protected"] = serde_json::json!([]); // try to hide where results were
    let forged = d.join("forged.json");
    std::fs::write(&forged, serde_json::to_string(&m).unwrap()).unwrap();
    let r = check(root, &published(root), Some(&forged), &root.join("in").join(NAME), &[], None, None);
    assert!(!r.ok() && r.checks.iter().any(|(ok, m)| !ok && m.contains("signature")), "{}", r.text());
}

#[test]
fn png_source_pages_verify() {
    // Non-JPEG sources (PNG scans, rendered pages) take the lossless pixel
    // path; the image hash must survive the PDF's re-encoding.
    let root = tmpdir("png");
    write_template(&root.join("templates").join("sample_rs1")).unwrap();
    let (scan, _) = simulate_scan(&filled_form(6).unwrap(), 6, 1.0, 1.3).unwrap();
    std::fs::create_dir_all(root.join("in")).unwrap();
    std::fs::write(root.join("in/png_form.pdf"), image_pdf(PageImage::Pixels(&scan), 1612.0, 2280.0)).unwrap();
    let rc = rfpii(&["redact", s(&root.join("in")), "--templates", s(&root.join("templates")), "--out", s(&root.join("out")), "--review", s(&root.join("review")), "--workers", "1", "--no-pdf-signature"]);
    assert_eq!(rc, 0);
    let m: Value = serde_json::from_str(&std::fs::read_to_string(root.join("out/png_form.manifest.json")).unwrap()).unwrap();
    assert!(m["pages"][0]["proof"] == "pixels" && m["pages"][0]["blocks_blacked"].as_u64().unwrap() > 0);
    let r = verify(&VerifyArgs {
        redacted: &root.join("out/png_form.pdf"),
        manifest: None,
        original: Some(&root.join("in/png_form.pdf")),
        pubkey: None,
        ledger: None,
        overlay_dir: None,
        rerun_templates: None,
        trust: &[],
    })
    .unwrap();
    assert!(r.ok(), "{}", r.text());
}

#[test]
fn pdf_output_has_no_text_or_metadata() {
    // a source PDF with a hidden text layer holding PII, and metadata
    let root = tmpdir("textlayer");
    write_template(&root.join("templates").join("sample_rs1")).unwrap();
    let (scan, _) = simulate_scan(&filled_form(7).unwrap(), 7, 3.0, 1.3).unwrap();
    let jpg = io::encode(&scan, "jpg", 90).unwrap();
    let (w, h) = (scan.cols() as f64 * 72.0 / 200.0, scan.rows() as f64 * 72.0 / 200.0);
    let mut doc = Document::with_version("1.7");
    let pages = doc.new_object_id();
    let font = doc.add_object(dictionary! {"Type" => "Font", "Subtype" => "Type1", "BaseFont" => "Helvetica"});
    let image = doc.add_object(Stream::new(
        dictionary! {"Type" => "XObject", "Subtype" => "Image", "Width" => scan.cols(), "Height" => scan.rows(),
                     "ColorSpace" => "DeviceRGB", "BitsPerComponent" => 8, "Filter" => "DCTDecode"},
        jpg,
    ));
    let content = format!("q {w} 0 0 {h} 0 0 cm /Im0 Do Q BT /F1 12 Tf 3 Tr 50 50 Td (JANE A. DOE 0712345678) Tj ET");
    let contents = doc.add_object(Stream::new(dictionary! {}, content.into_bytes()));
    let page = doc.add_object(dictionary! {
        "Type" => "Page", "Parent" => pages, "MediaBox" => vec![0.into(), 0.into(), Object::Real(w as f32), Object::Real(h as f32)],
        "Resources" => dictionary! {"XObject" => dictionary! {"Im0" => image}, "Font" => dictionary! {"F1" => font}},
        "Contents" => contents,
    });
    doc.objects.insert(pages, Object::Dictionary(dictionary! {"Type" => "Pages", "Kids" => vec![page.into()], "Count" => 1}));
    let catalog = doc.add_object(dictionary! {"Type" => "Catalog", "Pages" => pages});
    let info = doc.add_object(dictionary! {"Author" => Object::string_literal("Jane Doe")});
    doc.trailer.set("Root", catalog);
    doc.trailer.set("Info", info);
    std::fs::create_dir_all(root.join("in")).unwrap();
    doc.save(root.join("in/in.pdf")).unwrap();

    let rc = rfpii(&["redact", s(&root.join("in/in.pdf")), "--templates", s(&root.join("templates")), "--out", s(&root.join("out")), "--review", s(&root.join("review")), "--workers", "1", "--no-pdf-signature"]);
    assert_eq!(rc, 0);
    let out = std::fs::read(root.join("out/in.pdf")).unwrap();
    let res = Document::load_mem(&out).unwrap();
    assert!(res.trailer.get(b"Info").is_err());
    for (_, id) in res.get_pages() {
        let text = String::from_utf8_lossy(&res.get_page_content(id)).into_owned();
        assert!(!text.contains("Tj") && !text.contains("Tf"), "{}", text);
    }
    assert!(!String::from_utf8_lossy(&out).contains("JANE"));
}

#[test]
fn batch_withholds_failed_document() {
    let root = tmpdir("batch");
    write_template(&root.join("templates").join("sample_rs1")).unwrap();
    let scans = root.join("scans");
    std::fs::create_dir_all(&scans).unwrap();
    let (good, _) = simulate_scan(&filled_form(1).unwrap(), 1, 3.0, 1.3).unwrap();
    std::fs::write(scans.join("good.jpg"), io::encode(&good, "jpg", 90).unwrap()).unwrap();
    let white = opencv::core::Mat::new_rows_cols_with_default(800, 600, opencv::core::CV_8UC3, opencv::core::Scalar::all(255.0)).unwrap();
    std::fs::write(scans.join("bad.jpg"), io::encode(&white, "jpg", 90).unwrap()).unwrap();
    let rc = rfpii(&["redact", s(&scans), "--templates", s(&root.join("templates")), "--out", s(&root.join("out")), "--review", s(&root.join("review")), "--workers", "2", "--no-pdf-signature"]);
    assert_eq!(rc, 2);
    assert!(root.join("out/good.pdf").exists() && !root.join("out/bad.pdf").exists());
    assert!(root.join("review/bad.json").exists());
}

/// A local stand-in for Ollama answering /api/chat with `replies` in turn
/// (and /api/show for a vision model); returns its URL and how many chat
/// requests it has had.
fn ollama_stub(replies: Vec<String>) -> (String, std::sync::Arc<std::sync::atomic::AtomicUsize>) {
    use std::io::{BufRead, BufReader, Read, Write};
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let url = format!("http://127.0.0.1:{}", listener.local_addr().unwrap().port());
    let count = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let seen = count.clone();
    std::thread::spawn(move || {
        let mut replies = replies.into_iter();
        for conn in listener.incoming() {
            let Ok(mut conn) = conn else { break };
            let mut r = BufReader::new(conn.try_clone().unwrap());
            let (mut line, mut len) = (String::new(), 0usize);
            r.read_line(&mut line).unwrap();
            loop {
                let mut h = String::new();
                r.read_line(&mut h).unwrap();
                if h.trim().is_empty() {
                    break;
                }
                if let Some((k, v)) = h.split_once(':') {
                    if k.eq_ignore_ascii_case("content-length") {
                        len = v.trim().parse().unwrap();
                    }
                }
            }
            let mut body = vec![0u8; len];
            r.read_exact(&mut body).unwrap();
            let data = if line.contains("/api/show") {
                serde_json::to_vec(&serde_json::json!({"capabilities": ["completion", "vision"]})).unwrap()
            } else {
                seen.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                serde_json::to_vec(&serde_json::json!({"message": {"role": "assistant", "content": replies.next().unwrap_or_else(|| "[]".into())}})).unwrap()
            };
            let head = format!("HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n", data.len());
            conn.write_all(head.as_bytes()).unwrap();
            conn.write_all(&data).unwrap();
        }
    });
    (url, count)
}

#[test]
fn what_the_review_finds_is_masked_and_replayed() {
    let root = tmpdir("review");
    write_template(&root.join("templates").join("sample_rs1")).unwrap();
    let (scan, _) = simulate_scan(&filled_form(2).unwrap(), 2, 2.0, 1.3).unwrap();
    std::fs::create_dir_all(root.join("in")).unwrap();
    std::fs::write(root.join("in/form.jpg"), io::encode(&scan, "jpg", 90).unwrap()).unwrap();
    // round 1: a "signature" in the bottom margin, confirmed; round 2: nothing
    let spot = "```json\n[{\"box_2d\": [940, 100, 960, 300], \"label\": \"signature\"}]\n```";
    let (url, asked) = ollama_stub(vec![spot.into(), r#"{"unmasked_signature": true, "complete_number": false, "what": "scribble"}"#.into(), "[]".into()]);
    let rc = rfpii(&[
        "redact", s(&root.join("in")), "--templates", s(&root.join("templates")), "--out", s(&root.join("out")),
        "--review", s(&root.join("review")), "--workers", "1", "--no-pdf-signature", "--vlm-audit", "--vlm-host", &url, "--vlm-model", "gemma4:31b",
    ]);
    assert_eq!(rc, 0, "{}", std::fs::read_to_string(root.join("review/report.jsonl")).unwrap_or_default());
    assert_eq!(asked.load(std::sync::atomic::Ordering::SeqCst), 3);
    let m: Value = serde_json::from_str(&std::fs::read_to_string(root.join("out/form.manifest.json")).unwrap()).unwrap();
    let found = m["pages"][0]["vlm_found"].as_array().unwrap();
    assert!(found.len() == 1 && found[0]["label"] == "signature");
    assert_eq!(m["vlm_review"].as_array().unwrap().len(), 2);
    // the spot is black in the published image
    let pdf = std::fs::read(root.join("out/form.pdf")).unwrap();
    let img = io::decode_bytes(&io::pdf_images(&pdf).unwrap().remove(0).0).unwrap();
    let b: Vec<f64> = found[0]["box"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect();
    let (cx, cy) = (((b[0] + b[2]) / 2.0) as i32, ((b[1] + b[3]) / 2.0) as i32);
    let px = img.at_2d::<opencv::core::Vec3b>(cy, cx).unwrap();
    assert!(px[0] < 20 && px[1] < 20 && px[2] < 20, "{:?}", px);
    // a re-run applies what the review found without asking anything
    let r = verify(&VerifyArgs {
        redacted: &root.join("out/form.pdf"),
        manifest: None,
        original: Some(&root.join("in/form.jpg")),
        pubkey: None,
        ledger: None,
        overlay_dir: None,
        rerun_templates: Some(&root.join("templates")),
        trust: &[],
    })
    .unwrap();
    assert!(r.ok(), "{}", r.text());
    assert_eq!(asked.load(std::sync::atomic::Ordering::SeqCst), 3);
}

/// `jpg` with an EXIF orientation tag `o`, right after its SOI.
fn with_exif(jpg: &[u8], o: u8) -> Vec<u8> {
    let tiff = [b"MM\0*\0\0\0\x08\0\x01".as_slice(), &[1, 0x12, 0, 3, 0, 0, 0, 1, 0, o, 0, 0], b"\0\0\0\0"].concat();
    let seg = [b"Exif\0\0".as_slice(), &tiff].concat();
    let mut tagged = vec![0xff, 0xd8, 0xff, 0xe1];
    tagged.extend(((seg.len() + 2) as u16).to_be_bytes());
    tagged.extend(&seg);
    tagged.extend(&jpg[2..]);
    tagged
}

#[test]
fn a_photo_turned_by_its_exif_tag_is_published_as_it_was_shown() {
    // A phone photo stored sideways, with an EXIF tag telling viewers to turn
    // it upright: it is redacted where it is stored, and published turned the
    // same way, which the check against the original confirms.
    let root = tmpdir("exif");
    write_template(&root.join("templates").join("sample_rs1")).unwrap();
    let (scan, _) = simulate_scan(&filled_form(5).unwrap(), 5, 1.0, 1.3).unwrap();
    let mut stored = opencv::core::Mat::default();
    opencv::core::rotate(&scan, &mut stored, opencv::core::ROTATE_90_COUNTERCLOCKWISE).unwrap();
    // EXIF orientation 6, "turn 90 degrees clockwise to show"
    let tagged = with_exif(&io::encode(&stored, "jpg", 90).unwrap(), 6);
    assert_eq!(io::exif_orientation(&tagged), 6);
    std::fs::create_dir_all(root.join("in")).unwrap();
    std::fs::write(root.join("in/photo.jpg"), &tagged).unwrap();
    let rc = rfpii(&["redact", s(&root.join("in")), "--templates", s(&root.join("templates")), "--out", s(&root.join("out")), "--review", s(&root.join("review")), "--workers", "1", "--no-pdf-signature"]);
    assert_eq!(rc, 0, "{}", std::fs::read_to_string(root.join("review/report.jsonl")).unwrap_or_default());
    let out = std::fs::read(root.join("out/photo.pdf")).unwrap();
    let page = &io::source_pages("photo.pdf", Some(&out)).unwrap()[0];
    let ((w, h), place, rotate) = io::shown(page);
    assert!(h > w && rotate == 0, "published upright, as viewers show the photo");
    let want = io::exif_page(6, (stored.cols(), stored.rows())).1;
    assert!(place.iter().zip(want).all(|(a, b)| (a - b).abs() < 0.01), "{:?}", place);
    let m: Value = serde_json::from_str(&std::fs::read_to_string(root.join("out/photo.manifest.json")).unwrap()).unwrap();
    assert!(m["pages"][0]["blocks_blacked"].as_u64().unwrap() > 0);
    let r = verify(&VerifyArgs {
        redacted: &root.join("out/photo.pdf"),
        manifest: None,
        original: Some(&root.join("in/photo.jpg")),
        pubkey: None,
        ledger: None,
        overlay_dir: None,
        rerun_templates: None,
        trust: &[],
    })
    .unwrap();
    assert!(r.ok() && r.checks.iter().any(|(ok, m)| *ok && m.contains("shown as the original")), "{}", r.text());
    // the same photo stored upright and untagged is a different page: caught
    let plain = io::encode(&scan, "jpg", 90).unwrap();
    std::fs::write(root.join("plain.jpg"), &plain).unwrap();
    let sp = io::source_pages("plain.jpg", Some(&plain)).unwrap();
    assert_ne!(io::shown(&sp[0]).1, place);
}

/// The re-run check of `verify` on `m` (the manifest at `manifest_path`).
fn rerun_check(root: &Path, manifest_path: &Path, original: &Path, redacted: &Path) -> bool {
    let r = check(root, redacted, Some(manifest_path), original, &[], None, Some(&root.join("templates")));
    r.checks.iter().find(|(_, msg)| msg.contains("re-running")).map(|(ok, _)| *ok).expect("a re-run check")
}

#[test]
fn alignment_settings_are_recorded_and_the_rerun_uses_them() {
    // redacted with alignment settings of its own
    let root = tmpdir("align-settings");
    write_template(&root.join("templates").join("sample_rs1")).unwrap();
    let (scan, _) = simulate_scan(&filled_form(4).unwrap(), 5, 2.0, 1.3).unwrap();
    std::fs::create_dir_all(root.join("in")).unwrap();
    std::fs::write(root.join("in").join(NAME), image_pdf(PageImage::Jpeg(&io::encode(&scan, "jpg", 80).unwrap()), 1612.0, 2280.0)).unwrap();
    assert_eq!(rfpii(&["keygen", s(&root.join("keys"))]), 0);
    let rc = rfpii(&[
        "redact", s(&root.join("in")), "--templates", s(&root.join("templates")), "--out", s(&root.join("out")),
        "--review", s(&root.join("review")), "--workers", "1", "--sign-key", s(&root.join("keys/signing_key.pem")),
        "--no-pdf-signature", "--min-inliers", "20", "--min-score", "0.2",
    ]);
    assert_eq!(rc, 0, "{}", std::fs::read_to_string(root.join("review/report.jsonl")).unwrap_or_default());
    let m = manifest(&root);
    assert_eq!(m["settings"]["align"]["min_inliers"], 20);
    assert_eq!(m["settings"]["align"]["min_score"], 0.2);
    assert_eq!(m["settings"]["align"]["max_dim"], 1200);
    let mpath = root.join("out").join(NAME.replace(".PDF", ".manifest.json"));
    let (orig, red) = (root.join("in").join(NAME), published(&root));
    assert!(rerun_check(&root, &mpath, &orig, &red), "the recorded settings reproduce the output");
    // a recorded threshold the page doesn't meet: the re-run, which uses it, fails
    let mut bad = m.clone();
    bad["settings"]["align"]["min_score"] = serde_json::json!(0.999);
    let bpath = root.join("bad.manifest.json");
    std::fs::write(&bpath, serde_json::to_string(&bad).unwrap()).unwrap();
    assert!(!rerun_check(&root, &bpath, &orig, &red), "the re-run must use the recorded alignment settings");
}

#[test]
fn manifests_without_alignment_settings_rerun_with_the_defaults() {
    let d = rfpii::align::AlignSettings::default();
    let a = rfpii::align::AlignSettings::from_json(&Value::Null);
    assert_eq!((a.min_inliers, a.min_score, a.max_dim, a.ratio), (d.min_inliers, d.min_score, d.max_dim, d.ratio));
    let b = rfpii::align::AlignSettings::from_json(&serde_json::json!({"min_inliers": 20}));
    assert_eq!((b.min_inliers, b.min_score), (20, d.min_score));
    let c = rfpii::align::AlignSettings::from_json(&d.to_json());
    assert_eq!(c.to_json(), d.to_json());
}

#[test]
fn numbers_hidden_whole_are_recorded_and_the_rerun_reads_them() {
    let d = &manifest(run())["settings"];
    assert!(d["full_ids"] == false && d["full_phones"] == false);
    let root = tmpdir("full-numbers");
    write_template(&root.join("templates").join("sample_rs1")).unwrap();
    let (scan, _) = simulate_scan(&filled_form(4).unwrap(), 6, 2.0, 1.3).unwrap();
    std::fs::create_dir_all(root.join("in")).unwrap();
    std::fs::write(root.join("in").join(NAME), image_pdf(PageImage::Jpeg(&io::encode(&scan, "jpg", 80).unwrap()), 1612.0, 2280.0)).unwrap();
    assert_eq!(rfpii(&["keygen", s(&root.join("keys"))]), 0);
    let rc = rfpii(&[
        "redact", s(&root.join("in")), "--templates", s(&root.join("templates")), "--out", s(&root.join("out")),
        "--review", s(&root.join("review")), "--workers", "1", "--sign-key", s(&root.join("keys/signing_key.pem")),
        "--no-pdf-signature", "--full-ids", "--full-phones",
    ]);
    assert_eq!(rc, 0, "{}", std::fs::read_to_string(root.join("review/report.jsonl")).unwrap_or_default());
    let m = manifest(&root);
    assert!(m["settings"]["full_ids"] == true && m["settings"]["full_phones"] == true, "{}", m["settings"]);
    let mpath = root.join("out").join(NAME.replace(".PDF", ".manifest.json"));
    assert!(rerun_check(&root, &mpath, &root.join("in").join(NAME), &published(&root)));
    let ms = rfpii::masking::MaskSettings::from_record(&m["settings"]);
    use rfpii::masking::StampRule;
    assert!(ms.full_ids && ms.full_phones && ms.stamps == StampRule::Bands && (ms.hide, ms.phone_tail) == (3, 6));
    assert_eq!(m["settings"]["stamps"], "bands");
    // manifests from before the flags: numbers hidden as by default, and
    // stamps as they were recorded then
    let old = rfpii::masking::MaskSettings::from_record(&serde_json::json!({"hide": 4, "phone_tail": 6, "mask_stamps": true}));
    assert!(!old.full_ids && !old.full_phones && old.stamps == StampRule::Masked && old.hide == 4);
    let older = rfpii::masking::MaskSettings::from_record(&serde_json::json!({"hide": 3, "phone_tail": 6, "mask_stamps": false}));
    assert_eq!(older.stamps, StampRule::Protected);
}

#[test]
fn the_controller_record_and_whose_data_each_mask_holds_are_in_the_manifest() {
    let root = tmpdir("controller");
    write_template(&root.join("templates").join("sample_rs1")).unwrap();
    let (scan, _) = simulate_scan(&filled_form(4).unwrap(), 7, 2.0, 1.3).unwrap();
    std::fs::create_dir_all(root.join("in")).unwrap();
    std::fs::write(root.join("in").join(NAME), image_pdf(PageImage::Jpeg(&io::encode(&scan, "jpg", 80).unwrap()), 1612.0, 2280.0)).unwrap();
    let controller = serde_json::json!({"controller": {"name": "Example Observer Group"}, "operator_role": "controller",
        "legal_basis": "s.30(1)(b)(vii)", "purpose": "republishing results forms", "commercial": false, "processors": []});
    std::fs::write(root.join("controller.json"), controller.to_string()).unwrap();
    assert_eq!(rfpii(&["keygen", s(&root.join("keys"))]), 0);
    // a record without its legal basis is refused before anything runs
    std::fs::write(root.join("bad.json"), r#"{"controller": {"name": "X"}, "operator_role": "controller", "purpose": "p", "commercial": false}"#).unwrap();
    let args = |c: &Path| vec![
        "redact".to_string(), s(&root.join("in")).into(), "--templates".into(), s(&root.join("templates")).into(), "--out".into(), s(&root.join("out")).into(),
        "--review".into(), s(&root.join("review")).into(), "--workers".into(), "1".into(), "--sign-key".into(), s(&root.join("keys/signing_key.pem")).into(),
        "--no-pdf-signature".into(), "--controller".into(), s(c).into(),
    ];
    let a: Vec<String> = args(&root.join("bad.json"));
    assert_ne!(rfpii(&a.iter().map(|x| x.as_str()).collect::<Vec<_>>()), 0);
    let a: Vec<String> = args(&root.join("controller.json"));
    assert_eq!(rfpii(&a.iter().map(|x| x.as_str()).collect::<Vec<_>>()), 0);
    let m = manifest(&root);
    assert!(m["controller"]["controller"] == "Example Observer Group" && m["controller"]["legal_basis"] == "s.30(1)(b)(vii)", "{}", m["controller"]);
    assert_eq!(m["controller"]["sha256"], rfpii::proof::sha256(controller.to_string().as_bytes()));
    let roles: Vec<&str> = m["pages"][0]["masks"].as_array().unwrap().iter().map(|x| x["role"].as_str().unwrap_or("")).collect();
    assert!(roles.contains(&"officer") && roles.contains(&"agent") && !roles.contains(&""), "{:?}", roles);
    // verify names the publisher
    let r = check(&root, &published(&root), None, &root.join("in").join(NAME), &[], None, None);
    assert!(r.notes.iter().any(|n| n.contains("published by Example Observer Group")), "{:?}", r.notes);
}

#[test]
fn a_scanned_batch_is_published_page_by_page() {
    // one file of three pages: a form, a page nothing can be read on, the
    // form again (encoded anew: other coefficients)
    let root = tmpdir("per-page");
    write_template(&root.join("templates").join("sample_rs1")).unwrap();
    let (scan, _) = simulate_scan(&filled_form(4).unwrap(), 4, 2.0, 1.3).unwrap();
    let blank = opencv::core::Mat::new_rows_cols_with_default(scan.rows(), scan.cols(), opencv::core::CV_8UC3, opencv::core::Scalar::all(255.0)).unwrap();
    let (a, b, c) = (io::encode(&scan, "jpg", 80).unwrap(), io::encode(&blank, "jpg", 80).unwrap(), io::encode(&scan, "jpg", 90).unwrap());
    let size = Some((1612.0, 2280.0));
    let pdf = io::build_pdf(&[(PageImage::Jpeg(&a), size, None, 0), (PageImage::Jpeg(&b), size, None, 0), (PageImage::Jpeg(&c), size, None, 0)]).unwrap();
    std::fs::create_dir_all(root.join("in")).unwrap();
    let orig = root.join("in").join("batch.pdf");
    std::fs::write(&orig, pdf).unwrap();
    assert_eq!(rfpii(&["keygen", s(&root.join("keys"))]), 0);
    let rc = rfpii(&[
        "redact", s(&root.join("in")), "--templates", s(&root.join("templates")), "--out", s(&root.join("out")),
        "--review", s(&root.join("review")), "--workers", "2", "--sign-key", s(&root.join("keys/signing_key.pem")),
        "--no-pdf-signature", "--per-page",
    ]);
    // the page that can't be read is held back alone
    assert_eq!(rc, 2);
    let out = root.join("out");
    assert!(out.join("batch.p001.pdf").exists() && out.join("batch.p003.pdf").exists());
    assert!(!out.join("batch.p002.pdf").exists() && root.join("review").join("batch.p002.json").exists());
    let mpath = out.join("batch.p003.manifest.json");
    let m: Value = serde_json::from_str(&std::fs::read_to_string(&mpath).unwrap()).unwrap();
    assert_eq!((m["source"]["page"].as_u64(), m["source"]["pages"].as_u64(), m["source"]["name"].as_str()), (Some(3), Some(3), Some("batch.pdf")));
    // checked against the whole original, by that page, and re-run
    let r = check(&root, &out.join("batch.p003.pdf"), None, &orig, &[], None, Some(&root.join("templates")));
    assert!(r.ok(), "{}", r.text());
    assert!(r.checks.iter().any(|(ok, msg)| *ok && msg.contains("this is page 3")), "{}", r.text());
    // said to be another page of the same file, it doesn't verify
    let mut other = m.clone();
    other["source"]["page"] = serde_json::json!(1);
    let opath = root.join("other.manifest.json");
    std::fs::write(&opath, serde_json::to_string(&other).unwrap()).unwrap();
    let r = check(&root, &out.join("batch.p003.pdf"), Some(&opath), &orig, &[], None, None);
    assert!(r.checks.iter().any(|(ok, msg)| !*ok && msg.contains("page 1: ") && msg.contains("block")), "{}", r.text());
}

/// A local stand-in for Clef answering which way a page is turned by
/// comparing it with `upright` (an encoded image of the page upright) in its
/// four quarter turns; returns its URL and how many pages it was asked about.
fn clef_stub(upright: Vec<u8>) -> (String, std::sync::Arc<std::sync::atomic::AtomicUsize>) {
    use base64::Engine;
    use std::io::{BufRead, BufReader, Read, Write};
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let url = format!("http://127.0.0.1:{}", listener.local_addr().unwrap().port());
    let count = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let seen = count.clone();
    std::thread::spawn(move || {
        let reference = io::decode_bytes(&upright).unwrap();
        for conn in listener.incoming() {
            let Ok(mut conn) = conn else { break };
            let mut r = BufReader::new(conn.try_clone().unwrap());
            let (mut line, mut len) = (String::new(), 0usize);
            r.read_line(&mut line).unwrap();
            loop {
                let mut h = String::new();
                r.read_line(&mut h).unwrap();
                if h.trim().is_empty() {
                    break;
                }
                if let Some((k, v)) = h.split_once(':') {
                    if k.eq_ignore_ascii_case("content-length") {
                        len = v.trim().parse().unwrap();
                    }
                }
            }
            let mut body = vec![0u8; len];
            r.read_exact(&mut body).unwrap();
            let data = if line.contains("/api/show") {
                serde_json::json!({"capabilities": ["decision", "vision"]})
            } else {
                seen.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                let body: Value = serde_json::from_slice(&body).unwrap();
                let sent = io::decode_bytes(&base64::engine::general_purpose::STANDARD.decode(body["images"][0].as_str().unwrap()).unwrap()).unwrap();
                // the reference turned k quarters clockwise that looks most like it
                let diff = |k: usize| -> f64 {
                    let mut t = reference.clone();
                    if k > 0 {
                        let code = [opencv::core::ROTATE_90_CLOCKWISE, opencv::core::ROTATE_180, opencv::core::ROTATE_90_COUNTERCLOCKWISE][k - 1];
                        opencv::core::rotate(&reference, &mut t, code).unwrap();
                    }
                    if (t.cols() > t.rows()) != (sent.cols() > sent.rows()) {
                        return f64::INFINITY;
                    }
                    let mut small = opencv::core::Mat::default();
                    opencv::imgproc::resize(&t, &mut small, sent.size().unwrap(), 0.0, 0.0, opencv::imgproc::INTER_AREA).unwrap();
                    let mut d = opencv::core::Mat::default();
                    opencv::core::absdiff(&small, &sent, &mut d).unwrap();
                    opencv::core::mean_def(&d).unwrap()[0]
                };
                let k = (0..4).min_by(|&a, &b| diff(a).total_cmp(&diff(b))).unwrap();
                let names = ["upright", "lying_on_right_side", "upside_down", "lying_on_left_side"];
                let p: serde_json::Map<String, Value> = names.iter().enumerate().map(|(i, n)| (n.to_string(), serde_json::json!(if i == k { 0.91 } else { 0.03 }))).collect();
                serde_json::json!({"answers": {"turn": {"type": "choice", "choice": names[k], "probabilities": p}}})
            };
            let data = serde_json::to_vec(&data).unwrap();
            let head = format!("HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n", data.len());
            conn.write_all(head.as_bytes()).unwrap();
            conn.write_all(&data).unwrap();
        }
    });
    (url, count)
}

#[test]
fn a_photo_with_no_orientation_tag_is_turned_upright_as_clef_judges() {
    // A phone photo stored on its side with no EXIF tag: viewers would show
    // it sideways. Clef, shown the redacted page, judges it lying on its left
    // side, and it is published turned a quarter clockwise. A copy carrying
    // a tag that says "as stored" is published as stored, without asking.
    let root = tmpdir("orient");
    write_template(&root.join("templates").join("sample_rs1")).unwrap();
    let (scan, _) = simulate_scan(&filled_form(6).unwrap(), 6, 1.0, 1.3).unwrap();
    let mut stored = opencv::core::Mat::default();
    opencv::core::rotate(&scan, &mut stored, opencv::core::ROTATE_90_COUNTERCLOCKWISE).unwrap();
    let jpg = io::encode(&stored, "jpg", 90).unwrap();
    std::fs::create_dir_all(root.join("in")).unwrap();
    std::fs::write(root.join("in/photo.jpg"), &jpg).unwrap();
    std::fs::write(root.join("in/tagged.jpg"), with_exif(&jpg, 1)).unwrap();
    let (url, asked) = clef_stub(io::encode(&scan, "jpg", 90).unwrap());
    let rc = rfpii(&[
        "redact", s(&root.join("in")), "--templates", s(&root.join("templates")), "--out", s(&root.join("out")),
        "--review", s(&root.join("review")), "--workers", "1", "--no-pdf-signature", "--orient", "--vlm-host", &url,
    ]);
    assert_eq!(rc, 0, "{}", std::fs::read_to_string(root.join("review/report.jsonl")).unwrap_or_default());
    // not upright as it is: asked about in all four turns; the tagged copy not at all
    assert_eq!(asked.load(std::sync::atomic::Ordering::SeqCst), 4);
    let shown = |name: &str| io::shown(&io::source_pages(name, Some(&std::fs::read(root.join("out").join(name)).unwrap())).unwrap()[0]);
    let ((w, h), _, rotate) = shown("photo.pdf");
    assert!(w > h && rotate == 90, "stored landscape, shown turned a quarter clockwise: {w} x {h}, {rotate}");
    assert_eq!(shown("tagged.pdf").2, 0);
    let m: Value = serde_json::from_str(&std::fs::read_to_string(root.join("out/photo.manifest.json")).unwrap()).unwrap();
    let o = &m["pages"][0]["orientation"];
    assert!(o["turn"] == 90 && o["judged"] == "lying_on_left_side" && o["model"] == "clef" && o["asked"].as_array().unwrap().len() == 4, "{o}");
    let tagged: Value = serde_json::from_str(&std::fs::read_to_string(root.join("out/tagged.manifest.json")).unwrap()).unwrap();
    assert!(tagged["pages"][0].get("orientation").is_none());
    // verified against the original, turn included; the re-run applies the
    // recorded turn without asking Clef again
    let check = |manifest: Option<&Path>| {
        verify(&VerifyArgs {
            redacted: &root.join("out/photo.pdf"),
            manifest,
            original: Some(&root.join("in/photo.jpg")),
            pubkey: None,
            ledger: None,
            overlay_dir: None,
            rerun_templates: Some(&root.join("templates")),
            trust: &[],
        })
        .unwrap()
    };
    let r = check(None);
    assert!(r.ok() && r.checks.iter().any(|(ok, m)| *ok && m.contains("turned 90° clockwise as recorded")), "{}", r.text());
    assert_eq!(asked.load(std::sync::atomic::Ordering::SeqCst), 4);
    // a manifest recording another turn than the page shows fails
    let mut forged = m.clone();
    forged["pages"][0]["orientation"]["turn"] = serde_json::json!(270);
    std::fs::write(root.join("forged.json"), serde_json::to_vec(&forged).unwrap()).unwrap();
    let r = check(Some(&root.join("forged.json")));
    assert!(r.checks.iter().any(|(ok, m)| !*ok && m.contains("turned 270°")), "{}", r.text());
}
