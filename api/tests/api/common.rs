//! The test harness: a server on a random port with a temporary data
//! folder and a fake `rfpii`; accounts and keys made through the library;
//! a webhook receiver and a Stripe stub, all on this machine.

use clap::Parser;
use reqwest::{Client, Response, multipart};
use rfpii_api::config::Config;
use rfpii_api::server::{self, AppState, Options, Server};
use serde_json::{Value, json};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

#[derive(Parser)]
struct ConfigOnly {
    #[command(flatten)]
    config: Config,
}

static COUNTER: AtomicUsize = AtomicUsize::new(0);

/// A fresh, empty folder for one test.
pub fn tmpdir(name: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("rfpii-api-test-{}-{}-{}", std::process::id(), COUNTER.fetch_add(1, Ordering::Relaxed), name));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

/// The configuration of a test server: no models, quick polling and
/// retries, webhooks to this machine allowed, generous rate limits.
pub fn test_config(dir: &Path) -> Config {
    let mut c = ConfigOnly::try_parse_from(["rfpii-api"]).unwrap().config;
    c.data_dir = dir.join("data");
    c.bind = "127.0.0.1:0".into();
    c.public_base_url = "http://localhost".into();
    c.rfpii_bin = dir.join("fake").join("rfpii");
    c.rfpii_templates = dir.join("templates");
    c.sign_key = dir.join("signing_key.pem");
    c.orchestrator_models = vec![];
    c.vision_models = vec![];
    c.poll_interval_ms = 50;
    c.webhook_retry_schedule = vec![1, 1, 1];
    c.webhook_allow_private = true;
    c.rate_limit_per_sec = 1000.0;
    c.rate_limit_burst = 1000;
    c.public_rate_limit_per_sec = 1000.0;
    c.public_rate_limit_burst = 1000;
    c.shutdown_grace_secs = 1;
    c
}

pub struct TestServer {
    pub dir: PathBuf,
    pub base: String,
    pub client: Client,
    pub server: Option<Server>,
    pub state: AppState,
}

impl TestServer {
    /// A server with `workers` in-process workers and the fake `rfpii`,
    /// its configuration changed by `edit`.
    pub async fn start(workers: usize, edit: impl FnOnce(&mut Config)) -> TestServer {
        let dir = tmpdir("server");
        let fake = dir.join("fake");
        std::fs::create_dir_all(&fake).unwrap();
        std::fs::create_dir_all(dir.join("templates")).unwrap();
        std::fs::write(dir.join("signing_key.pem"), "not a real key").unwrap();
        let script = fake.join("rfpii");
        std::fs::copy(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/api/fake-rfpii.sh"), &script).unwrap();
        make_executable(&script);
        let mut cfg = test_config(&dir);
        cfg.admin_token = Some("test-admin-token".into());
        edit(&mut cfg);
        Self::start_with(dir, cfg, workers).await
    }

    /// A server on `cfg` as given.
    pub async fn start_with(dir: PathBuf, cfg: Config, workers: usize) -> TestServer {
        let state = AppState::new(cfg).unwrap();
        let server = server::start_with(state.clone(), Options { http: true, workers }).await.unwrap();
        let base = format!("http://{}", server.addr.unwrap());
        let client = Client::builder().no_proxy().timeout(Duration::from_secs(60)).build().unwrap();
        TestServer { dir, base, client, server: Some(server), state }
    }

    /// Tell the fake `rfpii` how to behave (fake.conf).
    pub fn fake(&self, conf: &str) {
        std::fs::write(self.dir.join("fake").join("fake.conf"), conf).unwrap();
    }

    /// A file the fake wrote (last-job.yaml, job.pid, ...).
    pub fn fake_file(&self, name: &str) -> Option<String> {
        std::fs::read_to_string(self.dir.join("fake").join(name)).ok()
    }

    /// A job as the operator sees it, with its cost (`GET /v1/admin/jobs/{id}`,
    /// with the server's admin token).
    pub async fn operator_job(&self, id: &str) -> Value {
        let token = self.state.cfg.admin_token.clone().expect("the test server needs an admin token");
        let r = self.client.get(self.url(&format!("/v1/admin/jobs/{id}"))).bearer_auth(token).send().await.unwrap();
        assert_eq!(r.status(), 200);
        r.json().await.unwrap()
    }

    pub fn url(&self, path: &str) -> String {
        format!("{}{}", self.base, path)
    }

    /// An account on `plan` and a key with every scope; with a controller
    /// record and an approved DPIA when `ready`.
    pub async fn account(&self, plan: &str, ready: bool) -> (String, String) {
        let now = self.state.clock.now();
        let plan = plan.to_string();
        let (id, key) = self
            .state
            .db
            .call(move |c| {
                let a = rfpii_api::accounts::create(c, now, "Example Observer Group", Some("ops@example.org"), &plan, "test")?;
                let (_, key) = rfpii_api::auth::create(c, now, &a.id, Some("test"), &rfpii_api::auth::ALL_SCOPES, "test")?;
                Ok((a.id, key))
            })
            .await
            .unwrap();
        if ready {
            let r = self.put(&key, "/v1/account/controller", &controller()).await;
            assert_eq!(r.status(), 200, "{}", r.text().await.unwrap());
            let r = self.put(&key, "/v1/account/dpia", &json!({"reference": "ODPC/DPIA/2026/14", "status": "approved"})).await;
            assert_eq!(r.status(), 200);
        }
        (id, key)
    }

    /// A key of `account` with only `scopes`.
    pub async fn key_with(&self, account: &str, scopes: &[rfpii_api::auth::Scope]) -> String {
        let (now, account, scopes) = (self.state.clock.now(), account.to_string(), scopes.to_vec());
        self.state.db.call(move |c| Ok(rfpii_api::auth::create(c, now, &account, None, &scopes, "test")?.1)).await.unwrap()
    }

    pub async fn get(&self, key: &str, path: &str) -> Response {
        self.client.get(self.url(path)).bearer_auth(key).send().await.unwrap()
    }

    pub async fn post(&self, key: &str, path: &str, body: &Value) -> Response {
        self.client.post(self.url(path)).bearer_auth(key).json(body).send().await.unwrap()
    }

    pub async fn put(&self, key: &str, path: &str, body: &Value) -> Response {
        self.client.put(self.url(path)).bearer_auth(key).json(body).send().await.unwrap()
    }

    pub async fn patch(&self, key: &str, path: &str, body: &Value) -> Response {
        self.client.patch(self.url(path)).bearer_auth(key).json(body).send().await.unwrap()
    }

    pub async fn delete(&self, key: &str, path: &str) -> Response {
        self.client.delete(self.url(path)).bearer_auth(key).send().await.unwrap()
    }

    /// Upload `bytes` as `name` (multipart); the response.
    pub async fn upload_raw(&self, key: &str, name: &str, bytes: Vec<u8>) -> Response {
        let form = multipart::Form::new().part("file", multipart::Part::bytes(bytes).file_name(name.to_string()));
        self.client.post(self.url("/v1/files")).bearer_auth(key).multipart(form).send().await.unwrap()
    }

    /// Upload a small JPEG; its id.
    pub async fn upload(&self, key: &str, name: &str) -> String {
        let r = self.upload_raw(key, name, jpeg()).await;
        assert_eq!(r.status(), 201);
        r.json::<Value>().await.unwrap()["id"].as_str().unwrap().to_string()
    }

    /// Create a job of one uploaded file; the response.
    pub async fn create_job(&self, key: &str, body: Value) -> Response {
        self.post(key, "/v1/jobs", &body).await
    }

    /// Upload a file and start a job on it; the job.
    pub async fn start_job(&self, key: &str) -> Value {
        let f = self.upload(key, "station_0001.jpg").await;
        let r = self.create_job(key, json!({"file_ids": [f]})).await;
        assert_eq!(r.status(), 202, "{}", r.text().await.unwrap());
        r.json().await.unwrap()
    }

    /// Wait until the job satisfies `done` (20 s at most); the job.
    pub async fn wait_job(&self, key: &str, id: &str, done: impl Fn(&Value) -> bool) -> Value {
        let start = Instant::now();
        loop {
            let j: Value = self.get(key, &format!("/v1/jobs/{id}")).await.json().await.unwrap();
            if done(&j) {
                return j;
            }
            assert!(start.elapsed() < Duration::from_secs(20), "job never got there: {j}");
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
    }

    /// Wait until the job has ended; the job.
    pub async fn wait_end(&self, key: &str, id: &str) -> Value {
        self.wait_job(key, id, |j| ["succeeded", "failed", "canceled"].contains(&j["status"].as_str().unwrap_or(""))).await
    }

    pub async fn stop(mut self) {
        if let Some(s) = self.server.take() {
            s.stop().await;
        }
        let _ = std::fs::remove_dir_all(&self.dir);
    }
}

pub fn make_executable(p: &Path) {
    use std::os::unix::fs::PermissionsExt;
    std::fs::set_permissions(p, std::fs::Permissions::from_mode(0o755)).unwrap();
}

/// The controller record a customer gives.
pub fn controller() -> Value {
    json!({"controller": {"name": "Example Observer Group", "registration_no": "ODPC/123"},
           "legal_basis": "s.30(1)(b)(vii) legitimate interests",
           "purpose": "republishing polling-station results forms", "commercial": false})
}

/// The first bytes of a JPEG (enough to be told one; the fake doesn't
/// decode it).
pub fn jpeg() -> Vec<u8> {
    let mut b = vec![0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x10, b'J', b'F', b'I', b'F', 0];
    b.extend(std::iter::repeat_n(7u8, 500));
    b
}

/// A problem document's checks: the content type and RFC 9457 members.
pub async fn problem(r: Response, status: u16) -> Value {
    assert_eq!(r.status().as_u16(), status);
    assert_eq!(r.headers()["content-type"], "application/problem+json");
    let rid = r.headers().get("x-request-id").map(|v| v.to_str().unwrap().to_string());
    let p: Value = r.json().await.unwrap();
    assert_eq!(p["status"], status, "{p}");
    for k in ["type", "title", "detail", "instance", "request_id"] {
        assert!(p[k].is_string(), "{k} missing in {p}");
    }
    assert_eq!(rid.as_deref(), p["request_id"].as_str());
    p
}

/// What a local receiver was sent: headers and body of each request.
#[derive(Clone, Debug)]
pub struct Received {
    pub headers: axum::http::HeaderMap,
    pub body: String,
}

/// A local HTTP receiver that records requests and answers 500 to the
/// first `fail` of them.
pub struct Receiver {
    pub url: String,
    pub got: Arc<Mutex<Vec<Received>>>,
}

impl Receiver {
    pub async fn start(fail: usize, reply: Value) -> Receiver {
        let got: Arc<Mutex<Vec<Received>>> = Arc::default();
        let failures = Arc::new(AtomicUsize::new(fail));
        let (g, f) = (got.clone(), failures.clone());
        let app = axum::Router::new().fallback(move |headers: axum::http::HeaderMap, body: String| {
            let (g, f, reply) = (g.clone(), f.clone(), reply.clone());
            async move {
                g.lock().unwrap().push(Received { headers, body });
                let left = f.load(Ordering::SeqCst);
                if left > 0 {
                    f.store(left - 1, Ordering::SeqCst);
                    return (axum::http::StatusCode::INTERNAL_SERVER_ERROR, axum::Json(json!({"error": "try later"})));
                }
                (axum::http::StatusCode::OK, axum::Json(reply))
            }
        });
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        Receiver { url, got }
    }

    pub fn count(&self) -> usize {
        self.got.lock().unwrap().len()
    }

    pub fn all(&self) -> Vec<Received> {
        self.got.lock().unwrap().clone()
    }

    /// Wait until `n` requests came (10 s at most).
    pub async fn wait_for(&self, n: usize) {
        let start = Instant::now();
        while self.count() < n {
            assert!(start.elapsed() < Duration::from_secs(10), "only {} of {n} requests came", self.count());
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
    }
}

/// Decode a form-encoded body.
pub fn form(body: &str) -> Vec<(String, String)> {
    body.split('&').filter_map(|kv| kv.split_once('=')).map(|(k, v)| (decode(k), decode(v))).collect()
}

fn decode(s: &str) -> String {
    let s = s.replace('+', " ");
    let mut out = Vec::new();
    let b = s.as_bytes();
    let mut i = 0;
    while i < b.len() {
        if b[i] == b'%'
            && i + 2 < b.len()
            && let Ok(x) = u8::from_str_radix(&s[i + 1..i + 3], 16)
        {
            out.push(x);
            i += 3;
            continue;
        }
        out.push(b[i]);
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

pub fn field<'a>(f: &'a [(String, String)], k: &str) -> Option<&'a str> {
    f.iter().find(|(a, _)| a == k).map(|(_, v)| v.as_str())
}
