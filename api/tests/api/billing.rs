//! Free forms and credits: the monthly free allowance (5 released forms a
//! calendar month), form credits bought in packages, reservation at
//! creation and settlement at the end (free forms first, then credits),
//! the 402 that says how to buy, expiry when configured, the public price
//! and quote, purchases, and the admin CLI and API.

use crate::common::{TestServer, problem};
use rfpii_api::worker::{Final, finalize};
use serde_json::{Value, json};

/// End a queued job as if `rfpii job` had released `released` of its forms.
async fn settle(t: &TestServer, id: &str, released: i64) -> Value {
    let report = json!({"plan": {"documents": released}, "groups": [{"id": "photo", "released": released}], "model_usage": []});
    let f = Final { status: "succeeded", exit_code: Some(0), error: None, report: Some(report), cpu_seconds: 0.0 };
    finalize(&t.state, id, None, f).await.unwrap().expect("the job ended now").resource()
}

/// Start a job of `forms` forms (the fake's inventory); the response.
async fn job_of(t: &TestServer, key: &str, forms: i64) -> reqwest::Response {
    t.fake(&format!("FORMS={forms}\n"));
    let f = t.upload(key, "station.jpg").await;
    t.create_job(key, json!({"file_ids": [f]})).await
}

/// Start a job of `forms` forms, which must be accepted; the job.
async fn accepted(t: &TestServer, key: &str, forms: i64) -> Value {
    let r = job_of(t, key, forms).await;
    assert_eq!(r.status(), 202, "{}", r.text().await.unwrap());
    r.json().await.unwrap()
}

#[tokio::test(flavor = "multi_thread")]
async fn free_forms_are_reserved_first_then_credits_and_402_says_how_to_buy() {
    let t = TestServer::start(0, |_| {}).await;
    let (account, key) = t.account(0, true).await;
    let a = accepted(&t, &key, 3).await;
    assert_eq!(a["billing"]["reserved"], json!({"free_forms": 3, "credits": 0}));
    let acct = t.account_now(&key).await;
    assert_eq!((acct["free_forms"]["reserved"].as_i64(), acct["free_forms"]["left"].as_i64()), (Some(3), Some(2)));

    // 3 more: 2 free forms left and no credits
    let p = problem(job_of(&t, &key, 3).await, 402).await;
    assert!(p["type"].as_str().unwrap().ends_with("#problem-insufficient-credits"), "{p}");
    assert_eq!(
        (p["forms_requested"].as_i64(), p["free_forms_left"].as_i64(), p["credits_available"].as_i64(), p["free_forms_per_month"].as_i64()),
        (Some(3), Some(2), Some(0), Some(5))
    );
    assert_eq!(p["buy"]["packages_needed"], 1);
    assert_eq!((p["buy"]["quote"].as_str(), p["buy"]["checkout"].as_str()), (Some("GET /v1/billing/quote?packages=1"), Some("POST /v1/billing/checkout")));
    assert_eq!((p["buy"]["package_forms"].as_i64(), p["buy"]["package_price_cents"].as_i64(), p["buy"]["checkout_available"].as_bool()), (Some(100), Some(3500), Some(false)));
    let detail = p["detail"].as_str().unwrap();
    assert!(detail.contains("/v1/billing/checkout") && detail.contains("/v1/billing/quote?packages=1") && detail.contains("$35.00"), "{detail}");

    // a package: the 2 free forms, then a credit
    t.add_credits(&account, 1).await;
    let b = accepted(&t, &key, 3).await;
    assert_eq!(b["billing"]["reserved"], json!({"free_forms": 2, "credits": 1}));
    let acct = t.account_now(&key).await;
    assert_eq!(acct["free_forms"]["left"], 0);
    assert_eq!(acct["credits"], json!({"bought": 100, "used": 0, "expired": 0, "adjusted": 0, "reserved": 1, "available": 99, "expire_after_days": null}));
    // a job bigger than both: how many packages it needs
    let p = problem(job_of(&t, &key, 250).await, 402).await;
    assert_eq!((p["credits_available"].as_i64(), p["buy"]["packages_needed"].as_i64()), (Some(99), Some(2)));
    t.stop().await;
}

#[tokio::test(flavor = "multi_thread")]
async fn settlement_counts_free_forms_first_then_credits_and_lets_the_rest_go() {
    let t = TestServer::start(0, |_| {}).await;
    let (account, key) = t.account(0, true).await;
    let a = accepted(&t, &key, 4).await;
    let a = settle(&t, a["id"].as_str().unwrap(), 4).await;
    assert_eq!(a["billing"]["counted"], json!({"forms": 4, "free_forms": 4, "credits": 0}));
    t.add_credits(&account, 1).await;

    // 5 forms: 1 free and 4 credits reserved; 3 released, 2 held
    let b = accepted(&t, &key, 5).await;
    assert_eq!(b["billing"]["reserved"], json!({"free_forms": 1, "credits": 4}));
    let b = settle(&t, b["id"].as_str().unwrap(), 3).await;
    assert_eq!(b["billing"]["counted"], json!({"forms": 3, "free_forms": 1, "credits": 2}));
    // the held forms took nothing; the 2 credits not used are back
    let acct = t.account_now(&key).await;
    assert_eq!((acct["credits"]["used"].as_i64(), acct["credits"]["reserved"].as_i64(), acct["credits"]["available"].as_i64()), (Some(2), Some(0), Some(98)));
    assert_eq!((acct["free_forms"]["used"].as_i64(), acct["free_forms"]["left"].as_i64(), acct["forms_released_this_month"].as_i64()), (Some(5), Some(0), Some(7)));

    let u: Value = t.get(&key, "/v1/usage").await.json().await.unwrap();
    assert_eq!((u["forms"].as_i64(), u["free_forms"].as_i64(), u["credit_forms"].as_i64()), (Some(7), Some(5), Some(2)));
    let records = u["records"].as_array().unwrap();
    assert_eq!(records.len(), 2);
    assert_eq!((records[1]["forms"].as_i64(), records[1]["free_forms"].as_i64(), records[1]["credit_forms"].as_i64()), (Some(3), Some(1), Some(2)));
    assert_eq!((u["credits"]["available"].as_i64(), u["purchases"]["paid"].as_i64(), u["purchases"]["forms_bought"].as_i64()), (Some(98), Some(1), Some(100)));

    // every change to the credits is in the ledger, and the balance is its sum
    let acct_id = account.clone();
    let (sum, entries): (i64, i64) = t
        .state
        .db
        .call(move |c| Ok(c.query_row("SELECT SUM(forms), COUNT(*) FROM credit_ledger WHERE account_id = ?1", [acct_id], |r| Ok((r.get(0)?, r.get(1)?)))?))
        .await
        .unwrap();
    assert_eq!((sum, entries), (98, 2));
    t.stop().await;
}

#[tokio::test(flavor = "multi_thread")]
async fn free_forms_a_job_let_go_count_first_for_a_job_that_ends_later() {
    let t = TestServer::start(0, |_| {}).await;
    let (_, key) = t.account(1, true).await;
    // A holds the month's 5 free forms; B's 2 forms are reserved from credits
    let a = accepted(&t, &key, 5).await;
    let b = accepted(&t, &key, 2).await;
    assert_eq!(b["billing"]["reserved"], json!({"free_forms": 0, "credits": 2}));
    // A releases nothing: its free forms are let go, and B's 2 forms are free
    settle(&t, a["id"].as_str().unwrap(), 0).await;
    let b = settle(&t, b["id"].as_str().unwrap(), 2).await;
    assert_eq!(b["billing"]["counted"], json!({"forms": 2, "free_forms": 2, "credits": 0}));
    let acct = t.account_now(&key).await;
    assert_eq!((acct["free_forms"]["left"].as_i64(), acct["credits"]["available"].as_i64()), (Some(3), Some(100)));
    t.stop().await;
}

#[tokio::test(flavor = "multi_thread")]
async fn a_worker_settles_a_job_on_free_forms_and_credits() {
    let t = TestServer::start(1, |_| {}).await;
    let (_, key) = t.account(1, true).await;
    t.fake("FORMS=7\nRELEASED=6\n");
    let j = t.start_job(&key).await;
    assert_eq!(j["billing"]["reserved"], json!({"free_forms": 5, "credits": 2}));
    let j = t.wait_end(&key, j["id"].as_str().unwrap()).await;
    assert_eq!(j["billing"]["counted"], json!({"forms": 6, "free_forms": 5, "credits": 1}));
    assert_eq!(t.account_now(&key).await["credits"]["available"], 99);
    t.stop().await;
}

#[tokio::test(flavor = "multi_thread")]
async fn a_new_month_resets_the_free_forms_but_not_the_credits() {
    let t = TestServer::start(0, |_| {}).await;
    let (_, key) = t.account(1, true).await;
    let period = rfpii_api::clock::period_of(t.state.clock.now());
    let j = accepted(&t, &key, 8).await;
    settle(&t, j["id"].as_str().unwrap(), 8).await;
    let acct = t.account_now(&key).await;
    assert_eq!((acct["free_forms"]["left"].as_i64(), acct["credits"]["available"].as_i64()), (Some(0), Some(97)));

    t.state.clock.advance(32 * 86_400);
    let acct = t.account_now(&key).await;
    assert_ne!(acct["period"], period.as_str());
    assert_eq!((acct["free_forms"]["left"].as_i64(), acct["free_forms"]["used"].as_i64(), acct["credits"]["available"].as_i64()), (Some(5), Some(0), Some(97)));
    let j = accepted(&t, &key, 6).await;
    assert_eq!(j["billing"]["reserved"], json!({"free_forms": 5, "credits": 1}));
    // the old month's usage is still there
    let u: Value = t.get(&key, &format!("/v1/usage?period={period}")).await.json().await.unwrap();
    assert_eq!((u["free_forms"].as_i64(), u["credit_forms"].as_i64(), u["free"]["left"].as_i64()), (Some(5), Some(3), Some(0)));
    problem(t.get(&key, "/v1/usage?period=2026-13").await, 400).await;
    t.stop().await;
}

#[tokio::test(flavor = "multi_thread")]
async fn concurrent_jobs_never_reserve_more_than_there_is() {
    let t = TestServer::start(0, |_| {}).await;
    let (_, key) = t.account(0, true).await;
    t.fake("FORMS=1\n");
    let mut files = Vec::new();
    for i in 0..12 {
        files.push(t.upload(&key, &format!("s{i}.jpg")).await);
    }
    // all at once: admission is one IMMEDIATE transaction, so 5 fit and 7 don't
    let mut sends = tokio::task::JoinSet::new();
    for f in files {
        let (client, url, key) = (t.client.clone(), t.url("/v1/jobs"), key.clone());
        sends.spawn(async move { client.post(url).bearer_auth(key).json(&json!({"file_ids": [f]})).send().await.unwrap().status().as_u16() });
    }
    let codes = sends.join_all().await;
    assert_eq!(codes.iter().filter(|c| **c == 202).count(), 5, "{codes:?}");
    assert_eq!(codes.iter().filter(|c| **c == 402).count(), 7, "{codes:?}");
    assert_eq!(t.account_now(&key).await["free_forms"]["reserved"], 5);
    t.stop().await;
}

#[tokio::test(flavor = "multi_thread")]
async fn credits_expire_when_configured_but_not_while_a_job_holds_them() {
    let t = TestServer::start(0, |c| {
        c.credit_expiry_days = 30;
        c.free_forms_per_month = 0;
    })
    .await;
    let (account, key) = t.account(1, true).await;
    assert_eq!(t.account_now(&key).await["credits"]["expire_after_days"], 30);
    let j = accepted(&t, &key, 10).await;
    assert_eq!(j["billing"]["reserved"], json!({"free_forms": 0, "credits": 10}));
    // 31 days on: the lot has expired, but the job keeps its 10
    t.state.clock.advance(31 * 86_400);
    t.add_credits(&account, 1).await;
    let c = &t.account_now(&key).await["credits"];
    assert_eq!((c["expired"].as_i64(), c["reserved"].as_i64(), c["available"].as_i64()), (Some(90), Some(10), Some(100)));
    // the job uses 4 of them: the other 6 expire, the new package is whole
    settle(&t, j["id"].as_str().unwrap(), 4).await;
    let c = &t.account_now(&key).await["credits"];
    assert_eq!(
        (c["bought"].as_i64(), c["used"].as_i64(), c["expired"].as_i64(), c["reserved"].as_i64(), c["available"].as_i64()),
        (Some(200), Some(4), Some(96), Some(0), Some(100))
    );
    let acct_id = account.clone();
    let expiry: i64 =
        t.state.db.call(move |c| Ok(c.query_row("SELECT -SUM(forms) FROM credit_ledger WHERE account_id = ?1 AND kind = 'expiry'", [acct_id], |r| r.get(0))?)).await.unwrap();
    assert_eq!(expiry, 96);
    t.stop().await;
}

#[tokio::test(flavor = "multi_thread")]
async fn pricing_and_quotes_are_public() {
    let t = TestServer::start(0, |_| {}).await;
    let p: Value = t.client.get(t.url("/v1/pricing")).send().await.unwrap().json().await.unwrap();
    assert_eq!(p["free"]["forms_per_month"], 5);
    assert_eq!(
        (p["package"]["forms"].as_i64(), p["package"]["price_cents"].as_i64(), p["package"]["price_per_form_usd"].as_str(), p["package"]["price_usd"].as_str()),
        (Some(100), Some(3500), Some("0.35"), Some("35.00"))
    );
    assert_eq!((p["card_fee"]["basis_points"].as_i64(), p["card_fee"]["fixed_cents"].as_i64(), p["vat"]["rate_basis_points"].as_i64()), (Some(440), Some(30), Some(1600)));
    let q = &p["quote"];
    assert_eq!((q["net_cents"].as_i64(), q["card_fee_cents"].as_i64(), q["vat_cents"].as_i64(), q["total_cents"].as_i64()), (Some(3500), Some(220), Some(595), Some(4315)));
    assert_eq!(q["lines"].as_array().unwrap().len(), 2);
    assert_eq!((p["checkout"].as_bool(), p["package"]["credits_expire_after_days"].is_null()), (Some(false), true));
    assert!(p["how_the_price_was_set"].as_str().unwrap().ends_with("PRICING.md"));

    let q: Value = t.client.get(t.url("/v1/billing/quote?packages=10")).send().await.unwrap().json().await.unwrap();
    assert_eq!((q["packages"].as_i64(), q["forms"].as_i64(), q["net_cents"].as_i64(), q["card_fee_cents"].as_i64()), (Some(10), Some(1000), Some(35000), Some(1914)));
    assert_eq!((q["vat_net_cents"].as_i64(), q["vat_card_fee_cents"].as_i64(), q["vat_cents"].as_i64()), (Some(5600), Some(306), Some(5906)));
    assert_eq!((q["total_cents"].as_i64(), q["total_usd"].as_str(), q["effective_price_per_form_usd"].as_str()), (Some(42820), Some("428.20"), Some("0.4282")));
    assert_eq!((q["lines"][0]["quantity"].as_i64(), q["lines"][0]["unit_amount_cents"].as_i64(), q["lines"][1]["amount_cents"].as_i64()), (Some(10), Some(3500), Some(1914)));
    let one: Value = t.client.get(t.url("/v1/billing/quote")).send().await.unwrap().json().await.unwrap();
    assert_eq!(one["total_cents"], 4315);
    for bad in ["packages=0", "packages=101", "packages=x", "colour=red"] {
        problem(t.client.get(t.url(&format!("/v1/billing/quote?{bad}"))).send().await.unwrap(), 400).await;
    }
    t.stop().await;
}

#[tokio::test(flavor = "multi_thread")]
async fn purchases_are_listed_with_their_lines() {
    let t = TestServer::start(0, |_| {}).await;
    let (account, key) = t.account(2, false).await;
    t.add_credits(&account, 1).await;
    let r = t.get(&key, "/v1/billing/purchases?limit=1").await;
    assert!(r.headers()["link"].to_str().unwrap().contains("rel=\"next\""));
    let page: Value = r.json().await.unwrap();
    let p = &page["data"][0];
    assert_eq!((p["status"].as_str(), p["method"].as_str(), p["packages"].as_i64(), p["forms"].as_i64()), (Some("paid"), Some("manual"), Some(1), Some(100)));
    assert_eq!((p["net_cents"].as_i64(), p["card_fee_cents"].as_i64(), p["vat_cents"].as_i64(), p["total_cents"].as_i64()), (Some(3500), Some(0), Some(560), Some(4060)));
    assert_eq!(
        p["lines"],
        json!([{"item": "packages", "description": "1 package(s) of 100 forms", "quantity": 1, "unit_amount_cents": 3500, "amount_cents": 3500, "vat_rate_basis_points": 1600, "vat_cents": 560}])
    );
    assert_eq!((p["reference"].as_str(), p["paid_at"].is_string()), (Some("test transfer"), true));
    let rest: Value = t.get(&key, &format!("/v1/billing/purchases?after={}", p["id"].as_str().unwrap())).await.json().await.unwrap();
    assert_eq!((rest["data"][0]["packages"].as_i64(), rest["has_more"].as_bool()), (Some(2), Some(false)));
    // a key that reads jobs, or administers the account, may list them; one that only writes jobs may not
    use rfpii_api::auth::Scope;
    for (scopes, status) in [(vec![Scope::JobsRead], 200), (vec![Scope::AccountAdmin], 200), (vec![Scope::JobsWrite], 403)] {
        let k = t.key_with(&account, &scopes).await;
        assert_eq!(t.get(&k, "/v1/billing/purchases").await.status(), status, "{scopes:?}");
    }
    let acct = t.account_now(&key).await;
    assert_eq!(
        (acct["purchases"]["paid"].as_i64(), acct["purchases"]["forms_bought"].as_i64(), acct["purchases"]["total_paid_cents"].as_i64()),
        (Some(2), Some(300), Some(8120 + 4060))
    );
    t.stop().await;
}

#[tokio::test(flavor = "multi_thread")]
async fn the_admin_cli_creates_accounts_and_keys_and_adds_and_adjusts_credits() {
    use rfpii_api::cli::{AdminCmd, admin};
    let t = TestServer::start(0, |_| {}).await;
    let cfg = t.state.cfg.as_ref().clone();
    let id = admin(&cfg, AdminCmd::CreateAccount { name: "County Observers".into(), email: None }).unwrap();
    assert!(id.starts_with("acct_"));
    let out = admin(&cfg, AdminCmd::CreateKey { account: id.clone(), name: Some("ops".into()), scopes: "jobs:read,account:admin".into() }).unwrap();
    let key = out.lines().next().unwrap().to_string();
    let a = t.account_now(&key).await;
    assert_eq!((a["id"].as_str(), a["credits"]["available"].as_i64(), a["free_forms"]["left"].as_i64()), (Some(id.as_str()), Some(0), Some(5)));

    let out = admin(&cfg, AdminCmd::AddCredits { account: id.clone(), packages: 2, reference: "bank transfer 123".into() }).unwrap();
    assert!(out.contains("200 forms added") && out.contains("$81.20 with VAT") && out.contains("200 credit(s) available"), "{out}");
    let p: Value = t.get(&key, "/v1/billing/purchases").await.json().await.unwrap();
    let p = &p["data"][0];
    assert_eq!((p["method"].as_str(), p["status"].as_str(), p["reference"].as_str()), (Some("manual"), Some("paid"), Some("bank transfer 123")));
    assert_eq!((p["net_cents"].as_i64(), p["card_fee_cents"].as_i64(), p["vat_cents"].as_i64(), p["total_cents"].as_i64()), (Some(7000), Some(0), Some(1120), Some(8120)));
    for bad in [
        AdminCmd::AddCredits { account: "acct_nope".into(), packages: 1, reference: "x".into() },
        AdminCmd::AddCredits { account: id.clone(), packages: 0, reference: "x".into() },
        AdminCmd::AddCredits { account: id.clone(), packages: 1, reference: " ".into() },
        AdminCmd::AdjustCredits { account: id.clone(), forms: -201, reason: "refund".into() },
        AdminCmd::AdjustCredits { account: id.clone(), forms: 5, reason: "".into() },
    ] {
        assert!(admin(&cfg, bad).is_err());
    }
    let out = admin(&cfg, AdminCmd::AdjustCredits { account: id.clone(), forms: -50, reason: "refund of 50 forms".into() }).unwrap();
    assert!(out.contains("150 available"), "{out}");
    admin(&cfg, AdminCmd::AdjustCredits { account: id.clone(), forms: 10, reason: "goodwill".into() }).unwrap();
    let c = &t.account_now(&key).await["credits"];
    assert_eq!((c["bought"].as_i64(), c["adjusted"].as_i64(), c["available"].as_i64()), (Some(200), Some(-40), Some(160)));
    let list = admin(&cfg, AdminCmd::ListAccounts { json: true }).unwrap();
    assert!(list.contains(&id) && list.contains("\"available\": 160"), "{list}");
    assert!(admin(&cfg, AdminCmd::ListAccounts { json: false }).unwrap().contains("160"));
    // every change is in the audit log
    let acct = id.clone();
    let actions: Vec<String> = t
        .state
        .db
        .call(move |c| {
            let mut st = c.prepare("SELECT action FROM audit_log WHERE account_id = ?1 ORDER BY id")?;
            let rows = st.query_map([acct], |r| r.get(0))?.collect::<rusqlite::Result<Vec<String>>>()?;
            Ok(rows)
        })
        .await
        .unwrap();
    assert_eq!(actions, ["account.created", "key.created", "credits.added", "credits.adjusted", "credits.adjusted"]);
    t.stop().await;
}

#[tokio::test(flavor = "multi_thread")]
async fn the_admin_api_needs_its_token() {
    let t = TestServer::start(0, |c| c.admin_token = Some("admin-secret".into())).await;
    let admin = |path: &str, token: &str, body: Value| t.client.post(t.url(path)).bearer_auth(token).json(&body).send();
    problem(admin("/v1/admin/accounts", "wrong", json!({"name": "X"})).await.unwrap(), 401).await;
    // a job's cost is for the operator only
    problem(t.client.get(t.url("/v1/admin/jobs/job_x")).bearer_auth("wrong").send().await.unwrap(), 401).await;
    problem(t.client.get(t.url("/v1/admin/jobs/job_x")).bearer_auth("admin-secret").send().await.unwrap(), 404).await;
    // accounts have no plan
    problem(admin("/v1/admin/accounts", "admin-secret", json!({"name": "Observers", "plan": "paid"})).await.unwrap(), 422).await;
    let r = admin("/v1/admin/accounts", "admin-secret", json!({"name": "Observers"})).await.unwrap();
    assert_eq!(r.status(), 201);
    let id = r.json::<Value>().await.unwrap()["id"].as_str().unwrap().to_string();
    let r = admin(&format!("/v1/admin/accounts/{id}/keys"), "admin-secret", json!({})).await.unwrap();
    let key = r.json::<Value>().await.unwrap()["secret"].as_str().unwrap().to_string();

    let r = admin(&format!("/v1/admin/accounts/{id}/credits"), "admin-secret", json!({"packages": 3, "reference": "bank transfer 456"})).await.unwrap();
    assert_eq!(r.status(), 201);
    let v: Value = r.json().await.unwrap();
    assert_eq!((v["purchase"]["total_cents"].as_i64(), v["purchase"]["method"].as_str(), v["credits"]["available"].as_i64()), (Some(12180), Some("manual"), Some(300)));
    problem(admin(&format!("/v1/admin/accounts/{id}/credits"), "wrong", json!({"packages": 3, "reference": "x"})).await.unwrap(), 401).await;
    problem(admin("/v1/admin/accounts/acct_nope/credits", "admin-secret", json!({"packages": 1, "reference": "x"})).await.unwrap(), 404).await;
    problem(admin(&format!("/v1/admin/accounts/{id}/credits"), "admin-secret", json!({"packages": 101, "reference": "x"})).await.unwrap(), 422).await;
    let r = admin(&format!("/v1/admin/accounts/{id}/credit-adjustments"), "admin-secret", json!({"forms": -100, "reason": "refund"})).await.unwrap();
    assert_eq!(r.json::<Value>().await.unwrap()["credits"]["available"], 200);
    problem(admin(&format!("/v1/admin/accounts/{id}/credit-adjustments"), "admin-secret", json!({"forms": -201, "reason": "too many"})).await.unwrap(), 422).await;
    assert_eq!(t.account_now(&key).await["credits"]["available"], 200);
    let list: Value = t.client.get(t.url("/v1/admin/accounts")).bearer_auth("admin-secret").send().await.unwrap().json().await.unwrap();
    assert_eq!((list["data"].as_array().unwrap().len(), list["data"][0]["credits"]["available"].as_i64()), (1, Some(200)));
    t.stop().await;

    // without ADMIN_TOKEN the admin routes don't exist
    let t = TestServer::start(0, |c| c.admin_token = None).await;
    problem(t.client.get(t.url("/v1/admin/accounts")).bearer_auth("anything").send().await.unwrap(), 404).await;
    t.stop().await;
}
