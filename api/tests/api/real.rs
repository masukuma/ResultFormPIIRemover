//! One job through the API with the real `rfpii` binary, on synthetic
//! forms it makes itself (`rfpii demo`); and `POST /v1/verify` on a form
//! `rfpii redact` published. Skipped, saying so, if the binary is absent.
//! It uses no model: the release gate holds every form (it needs the
//! visual review), so nothing is released and nothing is charged.

use crate::common::{TestServer, test_config, tmpdir};
use reqwest::multipart;
use serde_json::Value;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Duration, Instant};

const DEFAULT_BIN: &str = "/home/user/apiwork/bin/rfpii";

fn real_bin() -> Option<PathBuf> {
    let p = PathBuf::from(std::env::var("RFPII_REAL_BIN").unwrap_or_else(|_| DEFAULT_BIN.into()));
    p.exists().then_some(p)
}

fn run(bin: &Path, args: &[&str]) {
    let out = Command::new(bin).args(args).env("RFPII_SIGN_PASSWORD", "unused").output().unwrap();
    assert!(out.status.code().is_some_and(|c| c == 0 || c == 2), "rfpii {}: {}{}", args.join(" "), String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr));
}

fn s(p: &Path) -> &str {
    p.to_str().unwrap()
}

#[tokio::test(flavor = "multi_thread")]
async fn a_whole_job_runs_with_the_real_binary_and_verify_checks_its_output() {
    let Some(bin) = real_bin() else {
        eprintln!("SKIPPED: the real rfpii binary is not at {DEFAULT_BIN} (set RFPII_REAL_BIN to run this test)");
        return;
    };
    let dir = tmpdir("real");
    run(&bin, &["demo", s(&dir.join("demo")), "--count", "2"]);
    run(&bin, &["keygen", s(&dir.join("keys")), "--unencrypted"]);
    let mut cfg = test_config(&dir);
    cfg.rfpii_bin = bin.clone();
    cfg.rfpii_templates = dir.join("demo/templates");
    cfg.sign_key = dir.join("keys/signing_key.pem");
    cfg.job_timeout_secs = 600;
    cfg.admin_token = Some("test-admin-token".into());
    let t = TestServer::start_with(dir.clone(), cfg, 1).await;
    let (account, key) = t.account("free", true).await;
    let r: Value = t.client.get(t.url("/readyz")).send().await.unwrap().json().await.unwrap();
    assert!(r["rfpii_version"].as_str().unwrap().starts_with("rfpii "));

    let scans = dir.join("demo/scans");
    let mut ids = Vec::new();
    for name in ["station_0000.jpg", "station_0001.jpg"] {
        let r = t.upload_raw(&key, name, std::fs::read(scans.join(name)).unwrap()).await;
        assert_eq!(r.status(), 201);
        ids.push(r.json::<Value>().await.unwrap()["id"].as_str().unwrap().to_string());
    }
    let r = t.create_job(&key, serde_json::json!({"file_ids": ids})).await;
    assert_eq!(r.status(), 202, "{}", r.text().await.unwrap());
    let j: Value = r.json().await.unwrap();
    assert_eq!(j["forms_requested"], 2);
    let id = j["id"].as_str().unwrap().to_string();
    let start = Instant::now();
    let j = loop {
        let j: Value = t.get(&key, &format!("/v1/jobs/{id}")).await.json().await.unwrap();
        if ["succeeded", "failed", "canceled"].contains(&j["status"].as_str().unwrap()) {
            break j;
        }
        assert!(start.elapsed() < Duration::from_secs(300), "the job did not end: {j}");
        tokio::time::sleep(Duration::from_millis(200)).await;
    };
    let log = std::fs::read_to_string(dir.join("data/jobs").join(&account).join(&id).join("rfpii.log")).unwrap_or_default();
    assert_eq!(j["status"], "succeeded", "{j}\n{log}");
    // held by the gate (no visual review without a model): nothing charged
    assert_eq!((j["exit_code"].as_i64(), j["forms_released"].as_i64(), j["forms_held"].as_i64()), (Some(2), Some(0), Some(2)));
    assert_eq!((j["price"]["forms_charged"].as_i64(), j["price"]["amount_microusd"].as_i64()), (Some(0), Some(0)));
    let cost = &t.operator_job(&id).await["cost"];
    assert!(cost["cpu_seconds"].as_f64().unwrap() > 0.0, "{cost}");
    let a: Value = t.get(&key, "/v1/account").await.json().await.unwrap();
    assert_eq!((a["free_forms_left"].as_i64(), a["forms_reserved"].as_i64()), (Some(5), Some(0)));
    let report: Value = t.get(&key, &format!("/v1/jobs/{id}/report")).await.json().await.unwrap();
    assert_eq!(report["plan"]["documents"], 2);
    let gate = &report["groups"][0]["gate"];
    assert!(gate.as_array().unwrap().iter().any(|c| c["ok"] == false && c["condition"].as_str().unwrap().starts_with("4.")), "{gate}");
    let outs: Value = t.get(&key, &format!("/v1/jobs/{id}/outputs")).await.json().await.unwrap();
    assert!(outs["data"].as_array().unwrap().is_empty(), "{outs}");

    // a form rfpii redact published, checked through the API
    let (pubd, rev) = (dir.join("pub"), dir.join("rev"));
    let original = scans.join("station_0000.jpg");
    run(
        &bin,
        &[
            "redact",
            s(&original),
            "--templates",
            s(&dir.join("demo/templates")),
            "--out",
            s(&pubd),
            "--review",
            s(&rev),
            "--sign-key",
            s(&dir.join("keys/signing_key.pem")),
            "--no-pdf-signature",
        ],
    );
    let pdf = std::fs::read(pubd.join("station_0000.pdf")).unwrap();
    let part = |p: &Path| multipart::Part::bytes(std::fs::read(p).unwrap()).file_name(p.file_name().unwrap().to_string_lossy().into_owned());
    let form = |pdf: Vec<u8>| {
        multipart::Form::new()
            .part("pdf", multipart::Part::bytes(pdf).file_name("station_0000.pdf"))
            .part("manifest", part(&pubd.join("station_0000.manifest.json")))
            .part("original", part(&original))
            .part("pubkey", part(&dir.join("keys/signing_key.pub.pem")))
            .part("ledger", part(&pubd.join("ledger.jsonl")))
    };
    let v: Value = t.client.post(t.url("/v1/verify")).multipart(form(pdf.clone())).send().await.unwrap().json().await.unwrap();
    assert_eq!(v["verified"], true, "{v}");
    assert!(v["result"].as_str().unwrap().starts_with("RESULT: VERIFIED"));
    assert!(v["checks"].as_array().unwrap().len() >= 10 && v["checks"].as_array().unwrap().iter().all(|c| c["ok"] == true));
    // one byte changed: not verified
    let mut tampered = pdf;
    let n = tampered.len();
    tampered[n / 2] ^= 0x01;
    let v: Value = t.client.post(t.url("/v1/verify")).multipart(form(tampered)).send().await.unwrap().json().await.unwrap();
    assert_eq!(v["verified"], false, "{v}");
    t.stop().await;
}
