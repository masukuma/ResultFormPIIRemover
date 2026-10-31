//! Plans and usage: the free allowance (5 released forms a calendar
//! month, reserved at creation, settled at the end), paid usage at the
//! unit price, the spend cap, pricing, and the admin CLI and API.

use crate::common::{TestServer, problem};
use serde_json::{Value, json};

#[tokio::test(flavor = "multi_thread")]
async fn the_free_allowance_is_five_released_forms_a_month() {
    let t = TestServer::start(1, |_| {}).await;
    t.fake("FORMS=3\nRELEASED=2\nDELAY=1\n");
    let (_, key) = t.account("free", true).await;
    let period = rfpii_api::clock::period_of(t.state.clock.now());
    let a = t.start_job(&key).await;
    // 3 reserved, 2 left: a second job of 3 is refused
    let acct: Value = t.get(&key, "/v1/account").await.json().await.unwrap();
    assert_eq!((acct["forms_reserved"].as_i64(), acct["free_forms_left"].as_i64()), (Some(3), Some(2)));
    let f = t.upload(&key, "b.jpg").await;
    let p = problem(t.create_job(&key, json!({"file_ids": [f]})).await, 402).await;
    assert!(p["type"].as_str().unwrap().ends_with("#problem-free-allowance-exceeded"));
    assert_eq!((p["forms_requested"].as_i64(), p["forms_left"].as_i64(), p["free_forms_per_month"].as_i64()), (Some(3), Some(2), Some(5)));
    assert!(p["upgrade"]["pricing"].as_str().unwrap().ends_with("PRICING.md") && p["detail"].as_str().unwrap().contains("/v1/billing/checkout"));

    // settled: 2 released and counted, the held one isn't, the rest let go
    let a = t.wait_end(&key, a["id"].as_str().unwrap()).await;
    assert_eq!((a["forms_released"].as_i64(), a["price"]["forms_charged"].as_i64(), a["price"]["amount_microusd"].as_i64()), (Some(2), Some(2), Some(0)));
    let acct: Value = t.get(&key, "/v1/account").await.json().await.unwrap();
    assert_eq!((acct["forms_released_this_month"].as_i64(), acct["forms_reserved"].as_i64(), acct["free_forms_left"].as_i64()), (Some(2), Some(0), Some(3)));

    // now 3 fit; then 1 is left, and 3 don't
    let r = t.create_job(&key, json!({"file_ids": [f]})).await;
    assert_eq!(r.status(), 202);
    let b: Value = r.json().await.unwrap();
    t.wait_end(&key, b["id"].as_str().unwrap()).await;
    let c = t.upload(&key, "c.jpg").await;
    let p = problem(t.create_job(&key, json!({"file_ids": [c]})).await, 402).await;
    assert_eq!(p["forms_left"], 1);

    // a new month: 5 again
    t.state.clock.advance(32 * 86_400);
    let r = t.create_job(&key, json!({"file_ids": [c]})).await;
    assert_eq!(r.status(), 202, "{}", r.text().await.unwrap());
    let u: Value = t.get(&key, &format!("/v1/usage?period={period}")).await.json().await.unwrap();
    assert_eq!((u["free_forms"].as_i64(), u["paid_forms"].as_i64(), u["amount_due_microusd"].as_i64()), (Some(4), Some(0), Some(0)));
    assert_eq!(u["records"].as_array().unwrap().len(), 2);
    problem(t.get(&key, "/v1/usage?period=2026-13").await, 400).await;
    t.stop().await;
}

#[tokio::test(flavor = "multi_thread")]
async fn paid_usage_is_recorded_at_the_unit_price_with_the_actual_cost() {
    let t = TestServer::start(1, |_| {}).await;
    t.fake("FORMS=3\nRELEASED=2\nTOKENS_IN=1000\nTOKENS_OUT=200\n");
    let (_, key) = t.account("paid", true).await;
    let j = t.start_job(&key).await;
    assert_eq!((j["price"]["unit_microusd"].as_i64(), j["price"]["estimate_microusd"].as_i64()), (Some(4900), Some(14700)));
    let j = t.wait_end(&key, j["id"].as_str().unwrap()).await;
    assert_eq!((j["price"]["forms_charged"].as_i64(), j["price"]["amount_microusd"].as_i64(), j["price"]["amount_usd"].as_str()), (Some(2), Some(9800), Some("0.0098")));
    // the cost next to the price: gemma4:31b, 1000 in and 200 out at $0.14 and $0.40 a million
    assert_eq!(j["cost"]["models"][0]["cost_microusd"], 220);
    assert!(j["cost"]["total_microusd"].as_i64().unwrap() >= 220);
    let u: Value = t.get(&key, "/v1/usage").await.json().await.unwrap();
    assert_eq!((u["paid_forms"].as_i64(), u["amount_due_microusd"].as_i64(), u["amount_due_usd"].as_str()), (Some(2), Some(9800), Some("0.0098")));
    assert_eq!((u["records"][0]["forms"].as_i64(), u["records"][0]["unit_price_microusd"].as_i64()), (Some(2), Some(4900)));
    let a: Value = t.get(&key, "/v1/account").await.json().await.unwrap();
    assert_eq!((a["spend_this_month_microusd"].as_i64(), a["free_forms_left"].as_i64()), (Some(9800), Some(0)));
    t.stop().await;
}

#[tokio::test(flavor = "multi_thread")]
async fn a_spend_cap_refuses_jobs_that_could_pass_it() {
    let t = TestServer::start(0, |_| {}).await;
    let (account, key) = t.account("paid", true).await;
    let (now, acct) = (t.state.clock.now(), account.clone());
    t.state.db.call(move |c| rfpii_api::accounts::set_plan(c, now, &acct, "paid", Some(Some(10_000)), "test")).await.unwrap();
    t.fake("FORMS=3\n");
    let f = t.upload(&key, "a.jpg").await;
    let p = problem(t.create_job(&key, json!({"file_ids": [f]})).await, 402).await;
    assert!(p["type"].as_str().unwrap().ends_with("#problem-spend-cap-reached"));
    assert_eq!((p["job_estimate_microusd"].as_i64(), p["spend_cap_microusd"].as_i64()), (Some(14700), Some(10_000)));
    t.fake("FORMS=2\n");
    assert_eq!(t.create_job(&key, json!({"file_ids": [f]})).await.status(), 202);
    // 9800 held for the job: another of 2 would pass the cap
    let g = t.upload(&key, "b.jpg").await;
    problem(t.create_job(&key, json!({"file_ids": [g]})).await, 402).await;
    t.stop().await;
}

#[tokio::test(flavor = "multi_thread")]
async fn pricing_is_public() {
    let t = TestServer::start(0, |_| {}).await;
    let p: Value = t.client.get(t.url("/v1/pricing")).send().await.unwrap().json().await.unwrap();
    assert_eq!(p["plans"][0]["forms_per_month"], 5);
    assert_eq!((p["plans"][1]["price_per_form_microusd"].as_i64(), p["plans"][1]["price_per_form_usd"].as_str()), (Some(4900), Some("0.0049")));
    assert!(p["how_the_price_was_set"].as_str().unwrap().ends_with("PRICING.md"));
    t.stop().await;
}

#[tokio::test(flavor = "multi_thread")]
async fn the_admin_cli_creates_accounts_keys_and_plans() {
    use rfpii_api::cli::{AdminCmd, admin};
    let t = TestServer::start(0, |_| {}).await;
    let cfg = t.state.cfg.as_ref().clone();
    let id = admin(&cfg, AdminCmd::CreateAccount { name: "County Observers".into(), email: None, plan: "free".into() }).unwrap();
    assert!(id.starts_with("acct_"));
    let out = admin(&cfg, AdminCmd::CreateKey { account: id.clone(), name: Some("ops".into()), scopes: "jobs:read,account:admin".into() }).unwrap();
    let key = out.lines().next().unwrap().to_string();
    let a: Value = t.get(&key, "/v1/account").await.json().await.unwrap();
    assert_eq!((a["id"].as_str(), a["plan"].as_str()), (Some(id.as_str()), Some("free")));
    admin(&cfg, AdminCmd::SetPlan { account: id.clone(), plan: "paid".into(), spend_cap_microusd: Some("5000000".into()) }).unwrap();
    let a: Value = t.get(&key, "/v1/account").await.json().await.unwrap();
    assert_eq!((a["plan"].as_str(), a["spend_cap_microusd"].as_i64()), (Some("paid"), Some(5_000_000)));
    assert!(admin(&cfg, AdminCmd::SetPlan { account: "acct_nope".into(), plan: "paid".into(), spend_cap_microusd: None }).is_err());
    assert!(admin(&cfg, AdminCmd::SetPlan { account: id.clone(), plan: "gold".into(), spend_cap_microusd: None }).is_err());
    let list = admin(&cfg, AdminCmd::ListAccounts { json: true }).unwrap();
    assert!(list.contains(&id));
    // every change is in the audit log
    let n: i64 = t.state.db.call(move |c| Ok(c.query_row("SELECT COUNT(*) FROM audit_log WHERE account_id = ?1", [id], |r| r.get(0))?)).await.unwrap();
    assert_eq!(n, 3);
    t.stop().await;
}

#[tokio::test(flavor = "multi_thread")]
async fn the_admin_api_needs_its_token() {
    let t = TestServer::start(0, |c| c.admin_token = Some("admin-secret".into())).await;
    let admin = |path: &str, token: &str, body: Value| t.client.post(t.url(path)).bearer_auth(token).json(&body).send();
    problem(admin("/v1/admin/accounts", "wrong", json!({"name": "X"})).await.unwrap(), 401).await;
    let r = admin("/v1/admin/accounts", "admin-secret", json!({"name": "Observers", "plan": "free"})).await.unwrap();
    assert_eq!(r.status(), 201);
    let id = r.json::<Value>().await.unwrap()["id"].as_str().unwrap().to_string();
    let r = admin(&format!("/v1/admin/accounts/{id}/keys"), "admin-secret", json!({})).await.unwrap();
    let key = r.json::<Value>().await.unwrap()["secret"].as_str().unwrap().to_string();
    let r =
        t.client.put(t.url(&format!("/v1/admin/accounts/{id}/plan"))).bearer_auth("admin-secret").json(&json!({"plan": "paid", "spend_cap_microusd": null})).send().await.unwrap();
    assert_eq!(r.json::<Value>().await.unwrap()["plan"], "paid");
    let a: Value = t.get(&key, "/v1/account").await.json().await.unwrap();
    assert_eq!(a["plan"], "paid");
    let list: Value = t.client.get(t.url("/v1/admin/accounts")).bearer_auth("admin-secret").send().await.unwrap().json().await.unwrap();
    assert_eq!(list["data"].as_array().unwrap().len(), 1);
    t.stop().await;

    // without ADMIN_TOKEN the admin routes don't exist
    let t = TestServer::start(0, |_| {}).await;
    problem(t.client.get(t.url("/v1/admin/accounts")).bearer_auth("anything").send().await.unwrap(), 404).await;
    t.stop().await;
}
