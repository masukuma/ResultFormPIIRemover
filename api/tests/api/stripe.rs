//! Stripe, against a local stub: a Checkout Session in payment mode for a
//! purchase of packages (two lines with the VAT tax rate), and the signed
//! webhook that pays the purchase and adds its credits once, refuses an
//! amount that isn't the quote's, and marks an expired session.

use crate::common::{Receiver, TestServer, field, form, problem};
use serde_json::{Value, json};

fn event(id: &str, kind: &str, object: Value) -> String {
    json!({"id": id, "object": "event", "type": kind, "data": {"object": object}}).to_string()
}

async fn send_event(t: &TestServer, body: &str, secret: &str) -> reqwest::Response {
    let sig = rfpii_api::stripe::signature_header(secret, t.state.clock.now(), body.as_bytes());
    t.client.post(t.url("/v1/billing/stripe-webhook")).header("stripe-signature", sig).header("content-type", "application/json").body(body.to_string()).send().await.unwrap()
}

/// A Stripe stub that answers each Checkout with a session of its own
/// (cs_test_1, cs_test_2, ...).
async fn stripe_stub() -> Receiver {
    Receiver::start_with(0, |n| json!({"id": format!("cs_test_{n}"), "url": format!("https://checkout.stripe.com/c/pay/cs_test_{n}")})).await
}

/// A server whose Stripe is the stub at `stub`.
async fn with_stripe(stub: &str) -> TestServer {
    let stub = stub.to_string();
    TestServer::start(0, move |c| {
        c.stripe_secret_key = Some("sk_test_123".into());
        c.stripe_tax_rate_id = Some("txr_vat16".into());
        c.stripe_webhook_secret = Some("whsec_stripe_test".into());
        c.stripe_api_base = stub;
    })
    .await
}

#[tokio::test(flavor = "multi_thread")]
async fn checkout_is_501_without_stripe_or_its_tax_rate() {
    let t = TestServer::start(0, |_| {}).await;
    let (_, key) = t.account(0, false).await;
    let p = problem(t.post(&key, "/v1/billing/checkout", &json!({"packages": 1})).await, 501).await;
    assert!(p["type"].as_str().unwrap().ends_with("#problem-billing-not-configured"));
    t.stop().await;
    // a key but no VAT tax rate: Stripe would charge no VAT, so no Checkout
    let t = TestServer::start(0, |c| c.stripe_secret_key = Some("sk_test_123".into())).await;
    let (_, key) = t.account(0, false).await;
    problem(t.post(&key, "/v1/billing/checkout", &json!({"packages": 1})).await, 501).await;
    let pricing: Value = t.client.get(t.url("/v1/pricing")).send().await.unwrap().json().await.unwrap();
    assert_eq!(pricing["checkout"], false);
    t.stop().await;
}

#[tokio::test(flavor = "multi_thread")]
async fn checkout_makes_a_payment_session_with_two_taxed_lines_and_a_pending_purchase() {
    let stub = stripe_stub().await;
    let t = with_stripe(&stub.url).await;
    let (account, key) = t.account(0, false).await;
    let r = t.post(&key, "/v1/billing/checkout", &json!({"packages": 10, "success_url": "https://observers.example.org/billing/done"})).await;
    assert_eq!(r.status(), 200);
    let s: Value = r.json().await.unwrap();
    assert_eq!((s["id"].as_str(), s["url"].as_str()), (Some("cs_test_1"), Some("https://checkout.stripe.com/c/pay/cs_test_1")));
    let purchase = s["purchase"]["id"].as_str().unwrap().to_string();
    assert_eq!((s["purchase"]["status"].as_str(), s["purchase"]["total_cents"].as_i64()), (Some("pending"), Some(42820)));

    let sent = &stub.all()[0];
    assert_eq!(sent.headers["authorization"], "Bearer sk_test_123");
    let f = form(&sent.body);
    assert_eq!((field(&f, "mode"), field(&f, "client_reference_id")), (Some("payment"), Some(account.as_str())));
    assert_eq!((field(&f, "metadata[purchase_id]"), field(&f, "metadata[packages]")), (Some(purchase.as_str()), Some("10")));
    // the packages: $35.00 each, 10 of them; the card fee: $19.14 once; each with the VAT rate
    assert_eq!(
        (
            field(&f, "line_items[0][price_data][currency]"),
            field(&f, "line_items[0][price_data][unit_amount]"),
            field(&f, "line_items[0][quantity]"),
            field(&f, "line_items[0][tax_rates][0]")
        ),
        (Some("usd"), Some("3500"), Some("10"), Some("txr_vat16"))
    );
    assert_eq!(
        (
            field(&f, "line_items[1][price_data][currency]"),
            field(&f, "line_items[1][price_data][unit_amount]"),
            field(&f, "line_items[1][quantity]"),
            field(&f, "line_items[1][tax_rates][0]")
        ),
        (Some("usd"), Some("1914"), Some("1"), Some("txr_vat16"))
    );
    assert!(field(&f, "line_items[0][price_data][product_data][name]").unwrap().contains("package of 100 forms"));
    assert!(field(&f, "line_items[1][price_data][product_data][name]").unwrap().contains("4.4% + $0.30"));
    assert!(field(&f, "line_items[2][quantity]").is_none());
    assert_eq!((field(&f, "success_url"), field(&f, "customer_email")), (Some("https://observers.example.org/billing/done"), Some("ops@example.org")));
    assert!(field(&f, "cancel_url").unwrap().ends_with("/v1/pricing?checkout=canceled"));

    // the purchase is pending, with its quote line by line
    let list: Value = t.get(&key, "/v1/billing/purchases").await.json().await.unwrap();
    let p = &list["data"][0];
    assert_eq!(
        (p["id"].as_str(), p["status"].as_str(), p["method"].as_str(), p["stripe_session_id"].as_str()),
        (Some(purchase.as_str()), Some("pending"), Some("stripe"), Some("cs_test_1"))
    );
    assert_eq!((p["net_cents"].as_i64(), p["card_fee_cents"].as_i64(), p["vat_cents"].as_i64(), p["total_cents"].as_i64()), (Some(35000), Some(1914), Some(5906), Some(42820)));
    assert_eq!((p["lines"][0]["vat_cents"].as_i64(), p["lines"][1]["vat_cents"].as_i64()), (Some(5600), Some(306)));
    // nothing is credited before Stripe says it is paid
    assert_eq!(t.account_now(&key).await["credits"]["available"], 0);

    // the number of packages and the return URLs are checked
    for bad in [json!({"packages": 0}), json!({"packages": 101}), json!({}), json!({"packages": 1, "success_url": "javascript:alert(1)"}), json!({"packages": 1, "colour": "red"})]
    {
        problem(t.post(&key, "/v1/billing/checkout", &bad).await, 422).await;
    }
    problem(t.client.post(t.url("/v1/billing/checkout")).bearer_auth(&key).send().await.unwrap(), 422).await;
    let read_only = t.key_with(&account, &[rfpii_api::auth::Scope::JobsRead]).await;
    problem(t.post(&read_only, "/v1/billing/checkout", &json!({"packages": 1})).await, 403).await;
    let pricing: Value = t.client.get(t.url("/v1/pricing")).send().await.unwrap().json().await.unwrap();
    assert_eq!(pricing["checkout"], true);
    t.stop().await;
}

#[tokio::test(flavor = "multi_thread")]
async fn a_checkout_stripe_refuses_leaves_no_purchase() {
    let stub = Receiver::start(1, json!({"id": "cs_test_2", "url": "https://checkout.stripe.com/c/pay/cs_test_2"})).await;
    let t = with_stripe(&stub.url).await;
    let (_, key) = t.account(0, false).await;
    let p = problem(t.post(&key, "/v1/billing/checkout", &json!({"packages": 1})).await, 502).await;
    assert!(p["type"].as_str().unwrap().ends_with("#problem-billing-unavailable"));
    let list: Value = t.get(&key, "/v1/billing/purchases").await.json().await.unwrap();
    assert!(list["data"].as_array().unwrap().is_empty());
    t.stop().await;
}

/// Start a Checkout for `packages` packages; the purchase id, the session
/// id and its total.
async fn checkout(t: &TestServer, key: &str, packages: i64) -> (String, String, i64) {
    let s: Value = t.post(key, "/v1/billing/checkout", &json!({"packages": packages})).await.json().await.unwrap();
    (s["purchase"]["id"].as_str().unwrap().into(), s["id"].as_str().unwrap().into(), s["purchase"]["total_cents"].as_i64().unwrap())
}

/// A Checkout Session object as Stripe sends it in an event.
fn session(id: &str, account: &str, purchase: &str, packages: i64, amount: i64, currency: &str, paid: &str) -> Value {
    json!({"id": id, "object": "checkout.session", "client_reference_id": account, "metadata": {"purchase_id": purchase, "packages": packages.to_string()},
           "amount_total": amount, "currency": currency, "payment_status": paid, "customer": "cus_123", "mode": "payment"})
}

#[tokio::test(flavor = "multi_thread")]
async fn the_webhook_credits_a_paid_session_once() {
    let stub = stripe_stub().await;
    let t = with_stripe(&stub.url).await;
    let (account, key) = t.account(0, true).await;
    let (purchase, cs, total) = checkout(&t, &key, 2).await;
    assert_eq!(total, 8593);
    let body = event("evt_1", "checkout.session.completed", session(&cs, &account, &purchase, 2, total, "usd", "paid"));

    // a bad signature, or none, or an old one, changes nothing
    let p = problem(send_event(&t, &body, "whsec_wrong").await, 400).await;
    assert!(p["type"].as_str().unwrap().ends_with("#problem-invalid-signature"));
    problem(t.client.post(t.url("/v1/billing/stripe-webhook")).body(body.clone()).send().await.unwrap(), 400).await;
    let old = rfpii_api::stripe::signature_header("whsec_stripe_test", t.state.clock.now() - 3600, body.as_bytes());
    problem(t.client.post(t.url("/v1/billing/stripe-webhook")).header("stripe-signature", old).body(body.clone()).send().await.unwrap(), 400).await;
    assert_eq!(t.account_now(&key).await["credits"]["available"], 0);

    let r: Value = send_event(&t, &body, "whsec_stripe_test").await.json().await.unwrap();
    assert_eq!(r, json!({"received": true, "changed": true}));
    let a = t.account_now(&key).await;
    assert_eq!((a["credits"]["bought"].as_i64(), a["credits"]["available"].as_i64(), a["billing"]["stripe_customer"].as_bool()), (Some(200), Some(200), Some(true)));
    assert_eq!((a["purchases"]["paid"].as_i64(), a["purchases"]["total_paid_cents"].as_i64()), (Some(1), Some(8593)));
    let list: Value = t.get(&key, "/v1/billing/purchases").await.json().await.unwrap();
    assert_eq!((list["data"][0]["status"].as_str(), list["data"][0]["paid_at"].is_string()), (Some("paid"), true));

    // the same event again, or another event for the same session: credited once
    let r: Value = send_event(&t, &body, "whsec_stripe_test").await.json().await.unwrap();
    assert_eq!(r["changed"], false);
    let again = event("evt_2", "checkout.session.async_payment_succeeded", session(&cs, &account, &purchase, 2, total, "usd", "paid"));
    let r: Value = send_event(&t, &again, "whsec_stripe_test").await.json().await.unwrap();
    assert_eq!(r["changed"], false);
    assert_eq!(t.account_now(&key).await["credits"]["available"], 200);
    // the credits pay for jobs beyond the free forms
    t.fake("FORMS=7\n");
    let j = t.start_job(&key).await;
    assert_eq!(j["billing"]["reserved"], json!({"free_forms": 5, "credits": 2}));
    let acct = account.clone();
    let paid: i64 = t
        .state
        .db
        .call(move |c| Ok(c.query_row("SELECT COUNT(*) FROM audit_log WHERE account_id = ?1 AND action = 'purchase.paid' AND actor = 'stripe'", [acct], |r| r.get(0))?))
        .await
        .unwrap();
    assert_eq!(paid, 1);
    t.stop().await;
}

#[tokio::test(flavor = "multi_thread")]
async fn the_webhook_credits_nothing_for_an_amount_or_currency_not_quoted() {
    let stub = stripe_stub().await;
    let t = with_stripe(&stub.url).await;
    let (account, key) = t.account(0, false).await;

    // a tax rate that isn't 16% exclusive makes Stripe charge another total
    let (purchase, cs, total) = checkout(&t, &key, 1).await;
    let body = event("evt_1", "checkout.session.completed", session(&cs, &account, &purchase, 1, total - 595, "usd", "paid"));
    let r: Value = send_event(&t, &body, "whsec_stripe_test").await.json().await.unwrap();
    assert_eq!(r["changed"], true);
    let list: Value = t.get(&key, "/v1/billing/purchases").await.json().await.unwrap();
    let p = &list["data"][0];
    assert_eq!((p["status"].as_str(), p["amount_received_cents"].as_i64(), p["closed_at"].is_string()), (Some("mismatch"), Some(3720), true));
    assert_eq!(t.account_now(&key).await["credits"]["available"], 0);
    // and a later event for it changes nothing
    let body = event("evt_2", "checkout.session.completed", session(&cs, &account, &purchase, 1, total, "usd", "paid"));
    assert_eq!(send_event(&t, &body, "whsec_stripe_test").await.json::<Value>().await.unwrap()["changed"], false);
    assert_eq!(t.account_now(&key).await["credits"]["available"], 0);

    // the right amount in another currency
    let (purchase, cs, total) = checkout(&t, &key, 1).await;
    let body = event("evt_3", "checkout.session.completed", session(&cs, &account, &purchase, 1, total, "kes", "paid"));
    send_event(&t, &body, "whsec_stripe_test").await;
    let list: Value = t.get(&key, "/v1/billing/purchases").await.json().await.unwrap();
    assert_eq!((list["data"][0]["status"].as_str(), list["data"][0]["currency_received"].as_str()), (Some("mismatch"), Some("kes")));

    // a session not yet paid (a delayed payment method) waits
    let (purchase, cs, total) = checkout(&t, &key, 1).await;
    let body = event("evt_4", "checkout.session.completed", session(&cs, &account, &purchase, 1, total, "usd", "unpaid"));
    assert_eq!(send_event(&t, &body, "whsec_stripe_test").await.json::<Value>().await.unwrap()["changed"], false);
    // a session naming another account's purchase is ignored
    let (other, _) = t.account(0, false).await;
    let body = event("evt_5", "checkout.session.completed", session(&cs, &other, &purchase, 1, total, "usd", "paid"));
    assert_eq!(send_event(&t, &body, "whsec_stripe_test").await.json::<Value>().await.unwrap()["changed"], false);
    let list: Value = t.get(&key, "/v1/billing/purchases").await.json().await.unwrap();
    assert_eq!(list["data"][0]["status"], "pending");
    assert_eq!(t.account_now(&key).await["credits"]["available"], 0);
    let acct = account.clone();
    let mismatches: i64 =
        t.state.db.call(move |c| Ok(c.query_row("SELECT COUNT(*) FROM audit_log WHERE account_id = ?1 AND action = 'purchase.mismatch'", [acct], |r| r.get(0))?)).await.unwrap();
    assert_eq!(mismatches, 2);
    t.stop().await;
}

#[tokio::test(flavor = "multi_thread")]
async fn an_expired_session_marks_its_purchase_expired() {
    let stub = stripe_stub().await;
    let t = with_stripe(&stub.url).await;
    let (account, key) = t.account(0, false).await;
    let (purchase, cs, total) = checkout(&t, &key, 3).await;
    // Stripe's expiry event carries no amount paid
    let body = event(
        "evt_1",
        "checkout.session.expired",
        json!({"id": cs, "client_reference_id": account, "metadata": {"purchase_id": purchase}, "payment_status": "unpaid", "status": "expired"}),
    );
    assert_eq!(send_event(&t, &body, "whsec_stripe_test").await.json::<Value>().await.unwrap()["changed"], true);
    let list: Value = t.get(&key, "/v1/billing/purchases").await.json().await.unwrap();
    assert_eq!((list["data"][0]["status"].as_str(), list["data"][0]["closed_at"].is_string()), (Some("expired"), true));
    let a = t.account_now(&key).await;
    assert_eq!((a["purchases"]["expired"].as_i64(), a["credits"]["available"].as_i64()), (Some(1), Some(0)));
    // a completion arriving after it credits nothing
    let body = event("evt_2", "checkout.session.completed", session(&cs, &account, &purchase, 3, total, "usd", "paid"));
    assert_eq!(send_event(&t, &body, "whsec_stripe_test").await.json::<Value>().await.unwrap()["changed"], false);
    // events of other kinds, or for no purchase, are received and change nothing
    let r: Value = send_event(&t, &event("evt_3", "customer.subscription.deleted", json!({"id": "sub_1"})), "whsec_stripe_test").await.json().await.unwrap();
    assert_eq!(r, json!({"received": true, "changed": false}));
    let body = event("evt_4", "checkout.session.completed", json!({"id": "cs_unknown", "payment_status": "paid", "amount_total": 1, "currency": "usd"}));
    assert_eq!(send_event(&t, &body, "whsec_stripe_test").await.json::<Value>().await.unwrap()["changed"], false);
    // without a webhook secret the endpoint is off
    t.stop().await;
    let t = TestServer::start(0, |_| {}).await;
    problem(t.client.post(t.url("/v1/billing/stripe-webhook")).body("{}").send().await.unwrap(), 501).await;
    t.stop().await;
}
