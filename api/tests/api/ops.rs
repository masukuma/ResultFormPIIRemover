//! Cross-cutting behaviour: problem details, request ids, rate limits,
//! the OpenAPI description, readiness, metrics, verification and the
//! retention sweeper.

use crate::common::{TestServer, problem};
use reqwest::multipart;
use serde_json::{Value, json};

#[tokio::test(flavor = "multi_thread")]
async fn every_error_is_a_problem_naming_its_request() {
    let t = TestServer::start(0, |_| {}).await;
    let (_, key) = t.account("free", false).await;
    let r = t.client.get(t.url("/v1/jobs/job_nope")).bearer_auth(&key).header("x-request-id", "req-abc-123").send().await.unwrap();
    assert_eq!(r.headers()["x-request-id"], "req-abc-123");
    let p = problem(r, 404).await;
    assert_eq!((p["instance"].as_str(), p["request_id"].as_str()), (Some("urn:rfpii:request:req-abc-123"), Some("req-abc-123")));
    assert_eq!(p["title"], "Not found");
    // generated when the client sends none
    let r = t.get(&key, "/v1/jobs/job_nope").await;
    assert_eq!(r.headers()["x-request-id"].len(), 36);
    // errors made outside the handlers too: no route, wrong method
    let p = problem(t.client.get(t.url("/v2/nothing")).send().await.unwrap(), 404).await;
    assert_eq!(p["type"], "about:blank");
    let r = t.client.delete(t.url("/v1/pricing")).send().await.unwrap();
    assert!(r.headers().contains_key("allow"));
    problem(r, 405).await;
    t.stop().await;
}

#[tokio::test(flavor = "multi_thread")]
async fn a_key_over_its_rate_gets_429_with_ratelimit_fields() {
    let t = TestServer::start(0, |c| {
        c.rate_limit_burst = 3;
        c.rate_limit_per_sec = 0.01;
        c.public_rate_limit_burst = 2;
        c.public_rate_limit_per_sec = 0.01;
    })
    .await;
    let (_, key) = t.account("free", false).await;
    for left in [2, 1, 0] {
        let r = t.get(&key, "/v1/account").await;
        assert_eq!(r.status(), 200);
        assert_eq!(r.headers()["ratelimit-policy"], "\"per-key\";q=3;w=300");
        assert!(r.headers()["ratelimit"].to_str().unwrap().starts_with(&format!("\"per-key\";r={left};t=")));
    }
    let r = t.get(&key, "/v1/account").await;
    let retry: u64 = r.headers()["retry-after"].to_str().unwrap().parse().unwrap();
    assert!((1..=100).contains(&retry));
    assert!(r.headers().contains_key("ratelimit") && r.headers().contains_key("ratelimit-policy"));
    let p = problem(r, 429).await;
    assert!(p["type"].as_str().unwrap().ends_with("#problem-rate-limited"));
    // another key has its own bucket
    let (_, other) = t.account("free", false).await;
    assert_eq!(t.get(&other, "/v1/account").await.status(), 200);
    // the public routes are limited per IP address
    for _ in 0..2 {
        assert_eq!(t.client.get(t.url("/v1/pricing")).send().await.unwrap().status(), 200);
    }
    let r = t.client.get(t.url("/v1/pricing")).send().await.unwrap();
    assert!(r.headers()["ratelimit"].to_str().unwrap().starts_with("\"per-ip\";r=0"));
    problem(r, 429).await;
    t.stop().await;
}

#[tokio::test(flavor = "multi_thread")]
async fn the_openapi_description_covers_every_route() {
    let t = TestServer::start(0, |_| {}).await;
    let spec: Value = t.client.get(t.url("/v1/openapi.json")).send().await.unwrap().json().await.unwrap();
    assert!(spec["openapi"].as_str().unwrap().starts_with("3.1"));
    let routes = &t.server.as_ref().unwrap().routes;
    let mut described = 0;
    for (method, path) in routes {
        let p = path.replace("{*path}", "{path}");
        let op = &spec["paths"][&p][method.as_str().to_lowercase()];
        assert!(op.is_object(), "{method} {p} is not in openapi.json");
        assert!(op["summary"].is_string() && op["responses"].is_object(), "{method} {p} has no summary or responses");
        described += 1;
    }
    // and nothing is described that isn't served
    let mut ops = 0;
    for (p, item) in spec["paths"].as_object().unwrap() {
        for m in item.as_object().unwrap().keys().filter(|k| ["get", "post", "put", "patch", "delete"].contains(&k.as_str())) {
            ops += 1;
            assert!(routes.iter().any(|(rm, rp)| rp.replace("{*path}", "{path}") == *p && rm.as_str().eq_ignore_ascii_case(m)), "{m} {p} is described but not served");
        }
    }
    assert_eq!(ops, described);
    t.stop().await;
}

#[tokio::test(flavor = "multi_thread")]
async fn readyz_checks_the_database_the_disk_and_the_binary() {
    let t = TestServer::start(0, |_| {}).await;
    let r = t.client.get(t.url("/readyz")).send().await.unwrap();
    assert_eq!(r.status(), 200);
    let v: Value = r.json().await.unwrap();
    assert_eq!(v["checks"], json!({"database": "ok", "data_dir": "ok", "rfpii": "ok"}));
    assert_eq!(v["rfpii_version"], "rfpii 0.3.0 (fake)");
    assert_eq!(t.client.get(t.url("/healthz")).send().await.unwrap().status(), 200);
    t.stop().await;

    let t = TestServer::start(0, |c| c.rfpii_bin = "/nonexistent/rfpii".into()).await;
    let p = problem(t.client.get(t.url("/readyz")).send().await.unwrap(), 503).await;
    assert_eq!(p["checks"]["database"], "ok");
    assert!(p["checks"]["rfpii"].as_str().unwrap().contains("/nonexistent/rfpii"));
    t.stop().await;
}

#[tokio::test(flavor = "multi_thread")]
async fn metrics_count_requests_jobs_forms_and_tokens() {
    let t = TestServer::start(1, |c| c.metrics_token = Some("m-token".into())).await;
    let (_, key) = t.account("paid", true).await;
    let j = t.start_job(&key).await;
    t.wait_end(&key, j["id"].as_str().unwrap()).await;
    problem(t.client.get(t.url("/metrics")).send().await.unwrap(), 401).await;
    let m = t.client.get(t.url("/metrics")).bearer_auth("m-token").send().await.unwrap().text().await.unwrap();
    assert!(m.contains("rfpii_api_http_requests_total{method=\"POST\",route=\"/v1/jobs\",status=\"202\"} 1"), "{m}");
    assert!(m.contains("rfpii_api_http_request_duration_seconds_bucket{route=\"/v1/jobs/{id}\""));
    assert!(m.contains("rfpii_api_jobs{status=\"succeeded\"} 1") && m.contains("rfpii_api_queue_depth 0"));
    assert!(m.contains("rfpii_api_forms_released_total 3"));
    assert!(m.contains("rfpii_api_model_tokens_total{model=\"gemma4:31b\",direction=\"input\"} 1000"));
    t.stop().await;
}

fn part(bytes: &[u8], name: &str) -> multipart::Part {
    multipart::Part::bytes(bytes.to_vec()).file_name(name.to_string())
}

#[tokio::test(flavor = "multi_thread")]
async fn anyone_can_verify_a_published_form() {
    let t = TestServer::start(0, |_| {}).await;
    let verify = |form: multipart::Form| t.client.post(t.url("/v1/verify")).multipart(form).send();
    let pdf = b"%PDF-1.4 a redacted form".to_vec();
    let form = multipart::Form::new()
        .part("pdf", part(&pdf, "x.pdf"))
        .part("manifest", part(br#"{"signature": "ok"}"#, "x.manifest.json"))
        .part("original", part(&crate::common::jpeg(), "x.JPG"))
        .part("pubkey", part(b"PUBLIC KEY", "k.pem"));
    let r = verify(form).await.unwrap();
    assert_eq!(r.status(), 200);
    assert!(r.headers().contains_key("ratelimit"));
    let v: Value = r.json().await.unwrap();
    assert_eq!(v["verified"], true);
    assert_eq!(v["checks"][0], json!({"ok": true, "text": "manifest signature valid"}));
    assert_eq!(v["notes"], json!(["the PDF is not digitally signed"]));
    let args = t.fake_file("last-verify.txt").unwrap();
    assert!(args.contains("--original") && args.contains("original.jpg") && args.contains("--pubkey") && !args.contains("--ledger"), "{args}");

    let form = multipart::Form::new().part("pdf", part(&pdf, "x.pdf")).part("manifest", part(br#"{"signature": "bad"}"#, "m.json"));
    let v: Value = verify(form).await.unwrap().json().await.unwrap();
    assert_eq!(v["verified"], false);
    assert_eq!(v["checks"][1]["ok"], false);
    problem(verify(multipart::Form::new().part("pdf", part(&pdf, "x.pdf"))).await.unwrap(), 422).await;
    let form = multipart::Form::new().part("pdf", part(b"<html>", "x.pdf")).part("manifest", part(b"{}", "m.json"));
    problem(verify(form).await.unwrap(), 415).await;
    // nothing is kept
    let left: Vec<_> = std::fs::read_dir(t.dir.join("data/tmp")).unwrap().collect();
    assert!(left.is_empty());
    t.stop().await;
}

#[tokio::test(flavor = "multi_thread")]
async fn the_sweeper_deletes_what_is_past_its_time() {
    let t = TestServer::start(1, |_| {}).await;
    let (account, key) = t.account("free", true).await;
    let unused = t.upload(&key, "unused.jpg").await;
    let j = t.start_job(&key).await;
    let id = j["id"].as_str().unwrap().to_string();
    t.wait_end(&key, &id).await;
    // a day later: the unused upload goes
    t.state.clock.advance(25 * 3600);
    let s = rfpii_api::sweeper::sweep_once(&t.state).await.unwrap();
    assert_eq!(s.unused_files, 1);
    let f: Value = t.get(&key, &format!("/v1/files/{unused}")).await.json().await.unwrap();
    assert!(f["status"] == "deleted" && f["deleted_why"].as_str().unwrap().contains("not used"));
    assert!(!t.dir.join("data/files").join(&account).join(&unused).exists());
    assert_eq!(t.get(&key, &format!("/v1/jobs/{id}/outputs")).await.status(), 200);
    // 30 days after the job: its outputs go, its billing record stays
    t.state.clock.advance(30 * 86_400);
    let s = rfpii_api::sweeper::sweep_once(&t.state).await.unwrap();
    assert_eq!(s.job_outputs, 1);
    problem(t.get(&key, &format!("/v1/jobs/{id}/outputs")).await, 410).await;
    let j: Value = t.get(&key, &format!("/v1/jobs/{id}")).await.json().await.unwrap();
    assert!(j["outputs_deleted_at"].is_string() && j["price"]["forms_charged"] == 3);
    assert!(!t.dir.join("data/jobs").join(&account).join(&id).exists());
    let n: i64 = t.state.db.call(|c| Ok(c.query_row("SELECT COUNT(*) FROM deletions", [], |r| r.get(0))?)).await.unwrap();
    assert!(n >= 4, "{n}");
    t.stop().await;
}
