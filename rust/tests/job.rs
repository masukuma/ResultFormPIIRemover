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
/// It reaches no network: the only model host is the stub, on this machine.
fn run(args: &[&str]) -> (i32, String, String) {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_rfpii"));
    cmd.args(args).env("RFPII_SIGN_PASSWORD", "test-sign-password").env_remove("OLLAMA_API_KEY").env_remove("OLLAMA_HOST");
    for v in ["HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy", "ALL_PROXY", "all_proxy"] {
        cmd.env_remove(v);
    }
    let out = cmd.output().unwrap();
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

/// A job folder: three synthetic forms in two folders, a blank page, a copy
/// of a form, a key, a controller record and a job file with `extra` lines;
/// the job file's path.
fn job_setup(root: &Path, extra: &str) -> PathBuf {
    write_template(&root.join("templates/sample_rs1")).unwrap();
    for (dir, k) in [("North", 1u64), ("North", 2), ("South", 3)] {
        std::fs::create_dir_all(root.join("in").join(dir)).unwrap();
        std::fs::write(root.join("in").join(dir).join(format!("f{}.jpg", k)), form_jpeg(k, 90)).unwrap();
    }
    std::fs::write(root.join("in/South/blank.jpg"), blank_jpeg()).unwrap();
    std::fs::copy(root.join("in/North/f1.jpg"), root.join("in/South/copy.jpg")).unwrap();
    assert_eq!(rfpii(&["keygen", s(&root.join("keys"))]).0, 0);
    let controller = json!({"controller": {"name": "Example Observer Group"}, "operator_role": "processor", "legal_basis": "s.30(1)(b)(vii)",
                            "purpose": "republishing results forms", "commercial": false, "processors": [{"name": "Stub", "country": "US", "service": "vision models"}]});
    std::fs::write(root.join("controller.json"), controller.to_string()).unwrap();
    let job = format!("inputs: [in/]\noutputs: out\nsign_key: keys/signing_key.pem\ncontroller: controller.json\ntemplates: templates\n{}", extra);
    std::fs::write(root.join("job.yaml"), job).unwrap();
    root.join("job.yaml")
}

/// The journal's events of `kind`.
fn events(root: &Path, kind: &str) -> Vec<Value> {
    std::fs::read_to_string(root.join("out/journal.jsonl")).unwrap().lines().map(|l| serde_json::from_str::<Value>(l).unwrap()).filter(|e| e["kind"] == kind).collect()
}

#[test]
fn a_job_without_models_runs_to_its_report_and_releases_nothing() {
    let root = tmpdir("job-no-model");
    let job = job_setup(&root, "instructions: \"\"\ndpia: {reference: ODPC/DPIA/2026/1, submitted_on: 2026-10-20, status: submitted}\n");
    let (rc, out) = rfpii(&["job", s(&job)]);
    assert_eq!(rc, 2, "{}", out);
    let o = root.join("out");
    for f in ["plan.json", "journal.jsonl", "report.md", "report.json", "staging", "public", "review", "quarantine", "defects"] {
        assert!(o.join(f).exists(), "{} missing", f);
    }
    // everything redacted into staging, nothing released
    assert!(o.join("staging/North/f1.pdf").exists() && o.join("staging/South/f3.pdf").exists());
    assert!(!o.join("public/ledger.jsonl").exists() && std::fs::read_dir(o.join("public")).unwrap().count() == 0);
    let r = read(&o.join("report.json"));
    let g = &r["groups"][0];
    assert_eq!((g["published"].as_u64(), g["held"].as_u64(), g["released"].as_u64(), g["verified"].as_u64()), (Some(3), Some(1), Some(0), Some(3)));
    let failed: Vec<String> = g["gate"].as_array().unwrap().iter().filter(|c| c["ok"] == false).map(|c| c["condition"].as_str().unwrap().to_string()).collect();
    // no vision model: the leak rate can't be bounded; the DPIA is too recent
    assert!(failed.iter().any(|c| c.starts_with("4.")) && failed.iter().any(|c| c.starts_with("DPIA")), "{:?}", failed);
    assert!(std::fs::read_to_string(o.join("report.md")).unwrap().contains("processing may start 60 days after submission"));
    // the copy is left out, the blank page triaged from its diagnostics
    assert_eq!(r["plan"]["left_out"][0]["path"], "South/copy.jpg");
    assert_eq!(r["held"]["categories"], json!({"blank": 1}));
    assert!(o.join("review/triage/South/blank.json").exists());
    // the journal holds every tool call and decision, and no model call
    let tools: Vec<String> = events(&root, "tool").iter().map(|e| e["tool"].as_str().unwrap().to_string()).collect();
    for t in ["inventory", "redact", "verify", "stats", "diagnose", "report"] {
        assert!(tools.iter().any(|x| x == t), "no {} in the journal", t);
    }
    assert!(events(&root, "model").is_empty());
    let plan = read(&o.join("plan.json"));
    assert_eq!(plan["held_pages"], "none");
    // run again: nothing redone, the same decisions
    let before = std::fs::read(o.join("staging/North/f1.manifest.json")).unwrap();
    assert_eq!(rfpii(&["job", s(&job)]).0, 2);
    assert_eq!(std::fs::read(o.join("staging/North/f1.manifest.json")).unwrap(), before);
    let choices = |es: Vec<Value>| -> Vec<String> { es.iter().map(|e| format!("{} {}", e["phase"], e["choice"])).collect() };
    let all = choices(events(&root, "decision"));
    assert_eq!(all[..all.len() / 2], all[all.len() / 2..]);
    // a held page of a job that doesn't allow it is never shown
    let blank = s(&root.join("in/South/blank.jpg")).to_string();
    let (rc, text) = rfpii(&["held-view", &blank, "--mode", "triage", "--job", s(&o)]);
    assert!(rc != 0 && text.contains("does not let held pages go to the cloud model"), "{}", text);
}

#[test]
fn a_job_without_a_dpia_stops_at_staging_and_says_why() {
    let root = tmpdir("job-no-dpia");
    let job = job_setup(&root, "policy: {ids: whole}\n");
    assert_eq!(rfpii(&["job", s(&job)]).0, 2);
    let r = read(&root.join("out/report.json"));
    let gate = &r["groups"][0]["gate"];
    let dpia = gate.as_array().unwrap().iter().find(|c| c["condition"].as_str().unwrap().starts_with("DPIA")).unwrap();
    assert!(dpia["ok"] == false && dpia["why"].as_str().unwrap().contains("names no DPIA"));
    // the policy given ran: its flag on every redact call, its setting in every manifest
    let m = read(&root.join("out/staging/North/f2.manifest.json"));
    assert_eq!(m["settings"]["full_ids"], true);
    assert!(events(&root, "tool").iter().filter(|e| e["tool"] == "redact").all(|e| e["args"]["argv"].as_array().unwrap().iter().any(|a| a == "--full-ids")));
    assert_eq!(r["policy"]["flags"], json!(["--full-ids"]));
}

/// A stand-in for a model host, at 127.0.0.2: not a local host, so what is
/// sent to it is a transfer abroad, and an unredacted page is refused
/// unless the job allows it. It answers /api/show as a vision model would,
/// and each /api/chat with `answer(prompt, body)`; it records every request.
struct Host {
    url: String,
    requests: std::sync::Arc<std::sync::Mutex<Vec<(String, Value)>>>,
}

fn host(answer: impl Fn(&str, &Value) -> Value + Send + 'static) -> Host {
    use std::io::{BufRead, BufReader, Read, Write};
    let listener = std::net::TcpListener::bind("127.0.0.2:0").unwrap();
    let url = format!("http://127.0.0.2:{}", listener.local_addr().unwrap().port());
    let requests = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
    let seen = requests.clone();
    std::thread::spawn(move || {
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
            let body: Value = serde_json::from_slice(&body).unwrap();
            let path = line.split_whitespace().nth(1).unwrap_or("").to_string();
            seen.lock().unwrap().push((path.clone(), body.clone()));
            let reply = if path == "/api/show" {
                json!({"capabilities": ["completion", "vision"]})
            } else {
                let content = match answer(body["messages"][0]["content"].as_str().unwrap_or(""), &body) {
                    Value::String(t) => t,
                    v => v.to_string(),
                };
                json!({"message": {"role": "assistant", "content": content}})
            };
            let data = serde_json::to_vec(&reply).unwrap();
            let head = format!("HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n", data.len());
            conn.write_all(head.as_bytes()).unwrap();
            conn.write_all(&data).unwrap();
        }
    });
    Host { url, requests }
}

/// What a request asks, by the fields its answer must have.
fn asks(body: &Value) -> &'static str {
    let props = &body["format"]["properties"];
    let prompt = body["messages"][0]["content"].as_str().unwrap_or("");
    if props.get("ids").is_some() {
        "policy"
    } else if props.get("note").is_some() {
        "pilot note"
    } else if props.get("kind").is_some() {
        "triage"
    } else if props.get("choice").is_some() {
        "remedy"
    } else if props.get("unmasked_signature").is_some() {
        "check"
    } else if prompt.contains("after redaction") {
        "find"
    } else {
        "locate"
    }
}

fn default_policy() -> Value {
    json!({"ids": "three", "phones": "tail", "hide": 3, "phone_tail": 6, "stamps": "bands", "signatures": "band", "uncertain_tables": "widen",
           "vision_review": "sample", "per_page": "auto", "raised": [], "refused": [], "unsupported": []})
}

/// Every image the host was sent, by SHA-256.
fn images_received(h: &Host) -> Vec<String> {
    use base64::Engine;
    h.requests.lock().unwrap().iter().flat_map(|(_, b)| b["messages"].as_array().cloned().unwrap_or_default()).flat_map(|m| m["images"].as_array().cloned().unwrap_or_default())
        .map(|i| rfpii::proof::sha256(&base64::engine::general_purpose::STANDARD.decode(i.as_str().unwrap()).unwrap())).collect()
}

/// The privacy canaries over a job's journal and what its host received:
/// every image sent is a crop or a held view the journal logged, or a view
/// cut from one, and logged as sent; no prompt carries a run of 5 digits.
fn canaries(root: &Path, h: &Host) {
    let tools = events(root, "tool");
    let made: Vec<&str> = tools.iter().filter(|e| e["tool"] == "crop" || e["tool"] == "held_view").filter_map(|e| e["pixels_sha256"].as_str()).collect();
    let sent: Vec<Value> = events(root, "model").iter().flat_map(|e| e["images_sent"].as_array().cloned().unwrap_or_default()).collect();
    assert!(!sent.is_empty());
    for s in &sent {
        assert!(made.contains(&s["from"].as_str().unwrap()), "an image sent that no crop or held view made: {}", s);
    }
    let logged: Vec<&str> = sent.iter().filter_map(|s| s["sha256"].as_str()).collect();
    for img in images_received(h) {
        assert!(logged.contains(&img.as_str()), "the host received an image the journal doesn't list: {}", img);
    }
    let five = regex::Regex::new(r"\d{5,}").unwrap();
    for (_, b) in h.requests.lock().unwrap().iter() {
        let p = b["messages"][0]["content"].as_str().unwrap_or("");
        assert!(!five.is_match(p), "a prompt with a run of 5 digits: {}", p);
    }
}

#[test]
fn a_job_with_models_reviews_fixes_releases_and_reports() {
    let root = tmpdir("job-models");
    // the first page reviewed shows a "signature" in its bottom margin,
    // which the checker confirms; nothing else is found. The held page is
    // taken for the job's form, and located.
    let finds = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let f = finds.clone();
    let h = host(move |_, body| match asks(body) {
        "policy" => default_policy(),
        "pilot note" => json!({"note": "all well", "groups": []}),
        "find" if f.fetch_add(1, std::sync::atomic::Ordering::SeqCst) == 0 => json!("[{\"box_2d\": [940, 100, 960, 300], \"label\": \"signature\"}]"),
        "find" => json!("[]"),
        "check" => json!({"unmasked_signature": true, "complete_number": false, "what": "a scribble"}),
        "remedy" => json!({"choice": "spot_masks", "why": "one leak"}),
        "triage" => json!({"kind": "this_form", "turned": false, "cut_off": false, "faint": false, "declaration_visible": true, "agents_visible": true, "action": "locate"}),
        _ => json!({"items": [{"label": "signature", "bbox": [100, 100, 300, 160]}]}),
    });
    let job = job_setup(&root, &format!("instructions: |\n  Standard redaction. Station 007029014000102.\ndpia: {{reference: ODPC/DPIA/2026/2, status: approved}}\nheld_pages: cloud\nmodels: {{orchestrator: [orch], vision: [eyes], host: \"{}\"}}\nbudget: {{model_requests: 500}}\n", h.url));
    let (rc, out) = rfpii(&["job", s(&job)]);
    assert_eq!(rc, 2, "{}", out); // the blank page stays held
    let o = root.join("out");
    let r = read(&o.join("report.json"));
    let g = &r["groups"][0];
    assert_eq!((g["published"].as_u64(), g["released"].as_u64()), (Some(3), Some(3)), "{}", serde_json::to_string_pretty(&g["gate"]).unwrap());
    assert!(g["leak_rate"]["met"] == true && g["findings"] == 1);
    // released: PDFs, manifests, ledger, batch root, key and transfers record
    for f in ["North/f1.pdf", "North/f2.manifest.json", "South/f3.pdf", "ledger.jsonl", "signing_key.pub.pem", "transfers.jsonl"] {
        assert!(o.join("public").join(f).exists(), "{} not released", f);
    }
    assert!(r["batch_root"].as_str().unwrap().len() == 64 && std::fs::read_dir(o.join("public/batches")).unwrap().count() == 1);
    // the leak was masked by a spot mask, recorded and replayed
    let spotted: Vec<PathBuf> = ["North/f1", "North/f2", "South/f3"].iter().map(|n| o.join(format!("public/{}.manifest.json", n))).filter(|p| read(p)["pages"][0]["spot_masks"].is_array()).collect();
    assert_eq!(spotted.len(), 1);
    let pdf = spotted[0].with_extension("").with_extension("pdf");
    let orig = root.join("in").join(pdf.strip_prefix(o.join("public")).unwrap()).with_extension("jpg");
    let (rc, text) = rfpii(&["verify", s(&pdf), "--original", s(&orig), "--pubkey", s(&o.join("public/signing_key.pub.pem")), "--ledger", s(&o.join("public/ledger.jsonl")), "--rerun", s(&root.join("templates"))]);
    assert!(rc == 0 && text.contains("reproduces the same output"), "{}", text);
    // the held page: a triage view, then full resolution to locate fields
    let held = &r["held_sent_to_cloud"][0];
    assert_eq!(held["document"], "South/blank.jpg");
    let modes: Vec<&str> = held["images"].as_array().unwrap().iter().map(|i| i["mode"].as_str().unwrap()).collect();
    assert_eq!(modes, vec!["triage", "locate"]);
    assert_eq!(r["held"]["pages"][0]["action"], "located: still held");
    let transfers: Vec<Value> = std::fs::read_to_string(o.join("staging/transfers.jsonl")).unwrap().lines().map(|l| serde_json::from_str(l).unwrap()).collect();
    let purposes: Vec<&str> = transfers.iter().map(|t| t["purpose"].as_str().unwrap()).collect();
    for p in ["review of a published page", "triage of a held page", "locating the fields of a held page"] {
        assert!(purposes.contains(&p), "no transfer for {}: {:?}", p, purposes);
    }
    assert!(transfers.iter().all(|t| t["recipient"]["host"] == "127.0.0.2" && !t["images_sha256"].as_array().unwrap().is_empty()));
    canaries(&root, &h);
    // the orchestrator saw the pilot's figures, scrubbed but readable
    let note = h.requests.lock().unwrap().iter().find(|(_, b)| asks(b) == "pilot note").map(|(_, b)| b["messages"][0]["content"].as_str().unwrap().to_string()).unwrap();
    assert!(note.contains(r#""group":"photo""#) && note.contains(r#""published":3"#), "{}", note);

    // leaks found after release: each form withdrawn, the incident recorded;
    // the third of one kind re-opens the group
    for (k, name) in ["North/f1", "North/f2", "South/f3"].iter().enumerate() {
        let pdf = o.join(format!("public/{}.pdf", name));
        let (rc, text) = rfpii(&["job-incident", s(&o), s(&pdf), "--reason", "an ID number readable", "--kind", "id_number", "--aware-at", "2026-10-31T09:00:00+00:00"]);
        assert_eq!(rc, 0, "{}", text);
        assert!(!pdf.exists());
        assert_eq!(text.contains("re-opened"), k == 2, "{}", text);
    }
    let incidents: Vec<Value> = std::fs::read_dir(o.join("incidents")).unwrap().map(|e| e.unwrap().path()).filter(|p| p.extension().is_some_and(|x| x == "json") && !p.ends_with("reopened.json")).map(|p| read(&p)).collect();
    assert_eq!(incidents.len(), 3);
    let i = &incidents[0];
    assert_eq!(i["deadlines"]["commissioner"], "2026-11-03T09:00:00+00:00");
    assert_eq!(i["deadlines"]["controller"], "2026-11-02T09:00:00+00:00"); // the operator is a processor
    assert!(i["affected"]["officer"].as_u64().unwrap() > 0 && i["withdrawal"]["seq"].is_number());
    assert!(read(&o.join("incidents/reopened.json"))["photo"].as_str().unwrap().contains("3 id_number leaks"));
    let notice = std::fs::read_dir(o.join("incidents")).unwrap().map(|e| e.unwrap().path()).find(|p| p.extension().is_some_and(|x| x == "md")).unwrap();
    assert!(std::fs::read_to_string(notice).unwrap().contains("DRAFT: notification of a personal data breach"));
}

#[test]
fn a_job_asking_to_hide_names_is_redacted_but_not_released() {
    let root = tmpdir("job-names");
    // the orchestrator misses the instruction it can't follow; plain
    // matching catches it. Held pages stay on the machine: triaged from
    // their diagnostics alone.
    let h = host(|_, body| match asks(body) {
        "policy" => json!({"ids": "whole", "phones": "tail", "hide": 3, "phone_tail": 6, "stamps": "bands", "signatures": "band", "uncertain_tables": "widen",
                           "vision_review": "sample", "per_page": "auto", "raised": [], "refused": [], "unsupported": []}),
        "pilot note" => json!({"note": "fine", "groups": []}),
        "triage" => json!({"kind": "blank", "turned": false, "cut_off": false, "faint": false, "declaration_visible": false, "agents_visible": false, "action": "not_for_this_job"}),
        _ => json!("[]"),
    });
    let job = job_setup(&root, &format!("instructions: |\n  Hide ID numbers completely. Also hide the agents' names.\ndpia: {{reference: ODPC/DPIA/2026/3, status: approved}}\nmodels: {{orchestrator: [orch], vision: [eyes], host: \"{}\"}}\n", h.url));
    assert_eq!(rfpii(&["job", s(&job)]).0, 2);
    let o = root.join("out");
    let r = read(&o.join("report.json"));
    // redacted and checked as compiled, every page reviewed, nothing released
    assert_eq!(read(&o.join("staging/North/f1.manifest.json"))["settings"]["full_ids"], true);
    let g = &r["groups"][0];
    assert!(g["released"] == 0 && g["leak_rate"]["met"] == true);
    let failed: Vec<&Value> = g["gate"].as_array().unwrap().iter().filter(|c| c["ok"] == false).collect();
    assert!(failed.len() == 1 && failed[0]["condition"].as_str().unwrap().starts_with("7.") && failed[0]["why"].as_str().unwrap().contains("agents' names"), "{:?}", failed);
    assert!(std::fs::read_dir(o.join("public")).unwrap().count() == 0);
    let md = std::fs::read_to_string(o.join("report.md")).unwrap();
    assert!(md.contains("Instructions asking more than the redactor can do") && md.contains("names are never masked"), "{}", md);
    // the held page was sorted without its image leaving the machine
    assert_eq!(r["held"]["categories"], json!({"blank": 1}));
    assert!(r["held_sent_to_cloud"].as_array().unwrap().is_empty());
    let triage = events(&root, "model").into_iter().find(|e| e["purpose"] == "triage").unwrap();
    assert!(triage["images_sent"].as_array().unwrap().is_empty());
    let transfers = std::fs::read_to_string(o.join("staging/transfers.jsonl")).unwrap();
    assert!(transfers.lines().all(|l| l.contains("review of a published page")));
    canaries(&root, &h);
}

#[test]
fn a_held_view_sent_abroad_is_recorded_even_when_the_reply_is_unreadable() {
    let root = tmpdir("job-garbled");
    // the vision model answers the triage view with something that isn't
    // JSON: no verdict, so the page is sorted by fixed rules and never sent
    // at full resolution; but its triage view left, so it is a transfer.
    let h = host(|_, body| match asks(body) {
        "policy" => default_policy(),
        "pilot note" => json!({"note": "fine", "groups": []}),
        "triage" => json!("sorry, I can't tell"),
        _ => json!("[]"),
    });
    let job = job_setup(&root, &format!("dpia: {{reference: ODPC/DPIA/2026/4, status: approved}}\nheld_pages: cloud\nmodels: {{orchestrator: [orch], vision: [eyes], host: \"{}\"}}\n", h.url));
    assert_eq!(rfpii(&["job", s(&job)]).0, 2);
    let o = root.join("out");
    let triage = events(&root, "model").into_iter().find(|e| e["purpose"] == "triage").unwrap();
    assert!(triage.get("error").is_some() && triage["images_sent"].as_array().unwrap().len() == 1, "{}", triage);
    let transfers: Vec<Value> = std::fs::read_to_string(o.join("staging/transfers.jsonl")).unwrap().lines().map(|l| serde_json::from_str(l).unwrap()).collect();
    let held: Vec<&Value> = transfers.iter().filter(|t| t["purpose"] == "triage of a held page").collect();
    assert_eq!(held.len(), 1, "{:?}", transfers);
    assert_eq!(held[0]["images_sha256"][0], triage["images_sent"][0]["sha256"]);
    assert!(!transfers.iter().any(|t| t["purpose"] == "locating the fields of a held page"));
    canaries(&root, &h);
}
