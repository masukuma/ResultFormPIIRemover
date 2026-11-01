//! API keys: missing, wrong and revoked keys get 401; a key without the
//! scope gets 403.

use crate::common::{TestServer, problem};
use rfpii_api::auth::Scope;
use serde_json::{Value, json};

#[tokio::test(flavor = "multi_thread")]
async fn requests_without_a_valid_key_get_401() {
    let t = TestServer::start(0, |_| {}).await;
    let (_, key) = t.account("free", false).await;
    let r = t.client.get(t.url("/v1/account")).send().await.unwrap();
    assert_eq!(r.headers()["www-authenticate"], "Bearer realm=\"rfpii\"");
    let p = problem(r, 401).await;
    assert!(p["type"].as_str().unwrap().ends_with("#problem-unauthorized"));
    let wrong = format!("rfpii_live_{}", "0".repeat(43));
    let r = t.get(&wrong, "/v1/account").await;
    assert!(r.headers()["www-authenticate"].to_str().unwrap().contains("invalid_token"));
    problem(r, 401).await;
    problem(t.get("sk_live_something_else", "/v1/account").await, 401).await;

    assert_eq!(t.get(&key, "/v1/account").await.status(), 200);
    let keys: Value = t.get(&key, "/v1/keys").await.json().await.unwrap();
    let k = &keys["data"][0];
    assert!(k["last_used_at"].is_string() && k.get("secret").is_none() && key.starts_with(k["prefix"].as_str().unwrap()));
    // revoked: refused from then on
    let r = t.delete(&key, &format!("/v1/keys/{}", k["id"].as_str().unwrap())).await;
    assert_eq!(r.status(), 200);
    let p = problem(t.get(&key, "/v1/account").await, 401).await;
    assert!(p["detail"].as_str().unwrap().contains("revoked"));
    t.stop().await;
}

#[tokio::test(flavor = "multi_thread")]
async fn a_key_without_the_scope_gets_403() {
    let t = TestServer::start(0, |_| {}).await;
    let (account, admin) = t.account("free", true).await;
    let reader = t.key_with(&account, &[Scope::JobsRead]).await;
    let p = problem(t.post(&reader, "/v1/jobs", &json!({"file_ids": ["file_x"]})).await, 403).await;
    assert_eq!(p["required_scope"], "jobs:write");
    problem(t.get(&reader, "/v1/keys").await, 403).await;
    problem(t.put(&reader, "/v1/account/dpia", &json!({"reference": "x"})).await, 403).await;
    assert_eq!(t.get(&reader, "/v1/jobs").await.status(), 200);
    assert_eq!(t.get(&reader, "/v1/account").await.status(), 200);

    // a key made through the API: its secret shown once, its scopes as asked
    let r = t.post(&admin, "/v1/keys", &json!({"name": "ci", "scopes": ["jobs:read"]})).await;
    assert_eq!(r.status(), 201);
    let k: Value = r.json().await.unwrap();
    let secret = k["secret"].as_str().unwrap();
    assert!(secret.starts_with("rfpii_live_") && k["scopes"] == json!(["jobs:read"]));
    assert_eq!(t.get(secret, "/v1/jobs").await.status(), 200);
    problem(t.post(secret, "/v1/files", &json!({})).await, 403).await;
    let p = problem(t.post(&admin, "/v1/keys", &json!({"scopes": ["jobs:delete"]})).await, 422).await;
    assert!(p["detail"].as_str().unwrap().contains("unknown scope"));
    // another account's key can't be revoked from here
    let (_, other) = t.account("free", false).await;
    problem(t.delete(&other, &format!("/v1/keys/{}", k["id"].as_str().unwrap())).await, 404).await;
    t.stop().await;
}
