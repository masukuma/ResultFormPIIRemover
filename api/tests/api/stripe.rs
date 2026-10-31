//! Stripe, against a local stub: Checkout, the signed webhook that sets
//! the plan, and meter events sent once per usage record.

use crate::common::{Receiver, TestServer, field, form, problem};
use serde_json::{Value, json};
use std::time::Duration;

fn event(id: &str, kind: &str, object: Value) -> String {
    json!({"id": id, "object": "event", "type": kind, "data": {"object": object}}).to_string()
}

async fn send_event(t: &TestServer, body: &str, secret: &str) -> reqwest::Response {
    let sig = rfpii_api::stripe::signature_header(secret, t.state.clock.now(), body.as_bytes());
    t.client.post(t.url("/v1/billing/stripe-webhook")).header("stripe-signature", sig).header("content-type", "application/json").body(body.to_string()).send().await.unwrap()
}

#[tokio::test(flavor = "multi_thread")]
async fn checkout_is_501_without_stripe_and_a_session_with_it() {
    let t = TestServer::start(0, |_| {}).await;
    let (_, key) = t.account("free", false).await;
    let p = problem(t.post(&key, "/v1/billing/checkout", &json!({})).await, 501).await;
    assert!(p["type"].as_str().unwrap().ends_with("#problem-billing-not-configured"));
    t.stop().await;

    let stub = Receiver::start(0, json!({"id": "cs_test_1", "url": "https://checkout.stripe.com/c/pay/cs_test_1"})).await;
    let t = TestServer::start(0, |c| {
        c.stripe_secret_key = Some("sk_test_123".into());
        c.stripe_price_id = Some("price_metered".into());
        c.stripe_api_base = stub.url.clone();
    })
    .await;
    let (account, key) = t.account("free", false).await;
    let s: Value = t.post(&key, "/v1/billing/checkout", &json!({"success_url": "https://observers.example.org/billing/done"})).await.json().await.unwrap();
    assert_eq!(s["url"], "https://checkout.stripe.com/c/pay/cs_test_1");
    let sent = &stub.all()[0];
    assert_eq!(sent.headers["authorization"], "Bearer sk_test_123");
    let f = form(&sent.body);
    assert_eq!((field(&f, "mode"), field(&f, "line_items[0][price]"), field(&f, "client_reference_id")), (Some("subscription"), Some("price_metered"), Some(account.as_str())));
    assert_eq!((field(&f, "success_url"), field(&f, "customer_email")), (Some("https://observers.example.org/billing/done"), Some("ops@example.org")));
    assert!(field(&f, "cancel_url").unwrap().ends_with("/v1/pricing?checkout=canceled"));
    // a body is optional; a return URL must be a URL
    assert_eq!(t.client.post(t.url("/v1/billing/checkout")).bearer_auth(&key).send().await.unwrap().status(), 200);
    problem(t.post(&key, "/v1/billing/checkout", &json!({"success_url": "javascript:alert(1)"})).await, 422).await;
    let pricing: Value = t.client.get(t.url("/v1/pricing")).send().await.unwrap().json().await.unwrap();
    assert_eq!(pricing["plans"][1]["checkout"], true);
    t.stop().await;
}

#[tokio::test(flavor = "multi_thread")]
async fn the_stripe_webhook_is_verified_and_sets_the_plan() {
    let t = TestServer::start(0, |c| c.stripe_webhook_secret = Some("whsec_stripe_test".into())).await;
    let (account, key) = t.account("free", false).await;
    let body = event("evt_1", "checkout.session.completed", json!({"client_reference_id": account, "customer": "cus_123", "subscription": "sub_1"}));
    // a bad signature, or none, changes nothing
    let p = problem(send_event(&t, &body, "whsec_wrong").await, 400).await;
    assert!(p["type"].as_str().unwrap().ends_with("#problem-invalid-signature"));
    problem(t.client.post(t.url("/v1/billing/stripe-webhook")).body(body.clone()).send().await.unwrap(), 400).await;
    let old = rfpii_api::stripe::signature_header("whsec_stripe_test", t.state.clock.now() - 3600, body.as_bytes());
    problem(t.client.post(t.url("/v1/billing/stripe-webhook")).header("stripe-signature", old).body(body.clone()).send().await.unwrap(), 400).await;
    let a: Value = t.get(&key, "/v1/account").await.json().await.unwrap();
    assert_eq!(a["plan"], "free");

    let r: Value = send_event(&t, &body, "whsec_stripe_test").await.json().await.unwrap();
    assert_eq!(r["changed"], true);
    let a: Value = t.get(&key, "/v1/account").await.json().await.unwrap();
    assert_eq!((a["plan"].as_str(), a["billing"]["stripe_customer"].as_bool()), (Some("paid"), Some(true)));
    // the same event again: handled once
    let r: Value = send_event(&t, &body, "whsec_stripe_test").await.json().await.unwrap();
    assert_eq!(r["changed"], false);
    // the subscription lapses, then is deleted
    send_event(&t, &event("evt_2", "customer.subscription.updated", json!({"id": "sub_1", "customer": "cus_123", "status": "unpaid"})), "whsec_stripe_test").await;
    let a: Value = t.get(&key, "/v1/account").await.json().await.unwrap();
    assert_eq!(a["plan"], "free");
    send_event(&t, &event("evt_3", "customer.subscription.updated", json!({"id": "sub_1", "customer": "cus_123", "status": "active"})), "whsec_stripe_test").await;
    send_event(&t, &event("evt_4", "customer.subscription.deleted", json!({"id": "sub_1", "customer": "cus_123", "status": "canceled"})), "whsec_stripe_test").await;
    let a: Value = t.get(&key, "/v1/account").await.json().await.unwrap();
    assert_eq!(a["plan"], "free");
    t.stop().await;
}

#[tokio::test(flavor = "multi_thread")]
async fn meter_events_are_sent_once_per_usage_record() {
    let stub = Receiver::start(1, json!({"object": "billing.meter_event"})).await;
    let t = TestServer::start(1, |c| {
        c.stripe_secret_key = Some("sk_test_123".into());
        c.stripe_webhook_secret = Some("whsec_stripe_test".into());
        c.stripe_api_base = stub.url.clone();
    })
    .await;
    t.fake("FORMS=3\nRELEASED=2\n");
    let (account, key) = t.account("free", true).await;
    send_event(&t, &event("evt_1", "checkout.session.completed", json!({"client_reference_id": account, "customer": "cus_123"})), "whsec_stripe_test").await;
    let j = t.start_job(&key).await;
    t.wait_end(&key, j["id"].as_str().unwrap()).await;
    let u: Value = t.get(&key, "/v1/usage").await.json().await.unwrap();
    let record = u["records"][0]["id"].as_str().unwrap().to_string();

    // the first attempt fails (500): it is tried again later, the same event
    stub.wait_for(1).await;
    tokio::time::sleep(Duration::from_millis(300)).await;
    assert_eq!(stub.count(), 1);
    t.state.clock.advance(61);
    stub.wait_for(2).await;
    tokio::time::sleep(Duration::from_millis(500)).await;
    assert_eq!(stub.count(), 2, "reported more than once");
    for r in stub.all() {
        let f = form(&r.body);
        assert_eq!(r.headers["authorization"], "Bearer sk_test_123");
        assert_eq!(field(&f, "identifier"), Some(record.as_str()));
        assert_eq!((field(&f, "event_name"), field(&f, "payload[stripe_customer_id]"), field(&f, "payload[value]")), (Some("rfpii_forms"), Some("cus_123"), Some("2")));
    }
    let u: Value = t.get(&key, "/v1/usage").await.json().await.unwrap();
    assert_eq!(u["records"][0]["reported_to_stripe"], true);
    let id = record.clone();
    let attempts: i64 = t.state.db.call(move |c| Ok(c.query_row("SELECT stripe_attempts FROM usage_records WHERE id = ?1", [id], |r| r.get(0))?)).await.unwrap();
    assert_eq!(attempts, 2);

    // free usage is never reported
    let (_, free) = t.account("free", true).await;
    let j = t.start_job(&free).await;
    t.wait_end(&free, j["id"].as_str().unwrap()).await;
    tokio::time::sleep(Duration::from_millis(300)).await;
    assert_eq!(stub.count(), 2);
    t.stop().await;
}
