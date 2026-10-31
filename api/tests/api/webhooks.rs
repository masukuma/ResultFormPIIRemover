//! Webhooks: Standard Webhooks signatures, the same id across retries,
//! retries on failure, every attempt recorded.

use crate::common::{Receiver, TestServer};
use serde_json::{Value, json};

fn check_signed(secret: &str, r: &crate::common::Received) -> (String, Value) {
    let h = |k: &str| r.headers.get(k).unwrap().to_str().unwrap().to_string();
    let (id, ts, sig) = (h("webhook-id"), h("webhook-timestamp"), h("webhook-signature"));
    assert!(sig.starts_with("v1,"));
    assert!(rfpii_api::webhooks::verify(secret, &id, ts.parse().unwrap(), r.body.as_bytes(), &sig), "bad signature");
    assert!(!rfpii_api::webhooks::verify("whsec_AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA", &id, ts.parse().unwrap(), r.body.as_bytes(), &sig));
    (id, serde_json::from_str(&r.body).unwrap())
}

#[tokio::test(flavor = "multi_thread")]
async fn a_job_event_is_delivered_signed_and_retried() {
    let t = TestServer::start(1, |_| {}).await;
    let rx = Receiver::start(2, json!({"ok": true})).await;
    let (_, key) = t.account("free", true).await;
    let a: Value = t.patch(&key, "/v1/account/settings", &json!({"webhook_url": format!("{}/hook", rx.url)})).await.json().await.unwrap();
    let secret = a["webhook_secret"].as_str().unwrap().to_string();
    assert!(secret.starts_with("whsec_") && a["settings"]["webhook_secret_set"] == true);
    // the secret is shown once
    let again: Value = t.get(&key, "/v1/account").await.json().await.unwrap();
    assert!(again.get("webhook_secret").is_none());

    let j = t.start_job(&key).await;
    let id = j["id"].as_str().unwrap();
    rx.wait_for(3).await;
    let got = rx.all();
    let (first, body) = check_signed(&secret, &got[0]);
    for r in &got[1..] {
        assert_eq!(check_signed(&secret, r).0, first, "a retry keeps its webhook-id");
    }
    assert_eq!((body["type"].as_str(), body["data"]["id"].as_str(), body["data"]["status"].as_str()), (Some("job.succeeded"), Some(id), Some("succeeded")));
    let (status, attempts): (String, i64) =
        t.state.db.call(move |c| Ok(c.query_row("SELECT status, attempts FROM webhook_deliveries WHERE id = ?1", [first], |r| Ok((r.get(0)?, r.get(1)?)))?)).await.unwrap();
    assert_eq!((status.as_str(), attempts), ("delivered", 3));
    t.stop().await;
}

#[tokio::test(flavor = "multi_thread")]
async fn a_jobs_own_webhook_gets_its_events_with_the_rotated_secret() {
    let t = TestServer::start(1, |_| {}).await;
    t.fake("FORMS=1\nRELEASED=1\nDELAY=30\n");
    let rx = Receiver::start(0, json!({})).await;
    let (_, key) = t.account("free", true).await;
    let r: Value = t.post(&key, "/v1/account/webhook-secret", &json!({})).await.json().await.unwrap();
    let old = r["secret"].as_str().unwrap().to_string();
    let r: Value = t.post(&key, "/v1/account/webhook-secret", &json!({})).await.json().await.unwrap();
    let secret = r["secret"].as_str().unwrap().to_string();
    assert_ne!(old, secret);
    let f = t.upload(&key, "a.jpg").await;
    let j: Value = t.create_job(&key, json!({"file_ids": [f], "webhook_url": format!("{}/job-hook", rx.url)})).await.json().await.unwrap();
    let id = j["id"].as_str().unwrap();
    t.wait_job(&key, id, |j| j["status"] == "running").await;
    t.post(&key, &format!("/v1/jobs/{id}/cancel"), &json!({})).await;
    rx.wait_for(1).await;
    let got = &rx.all()[0];
    let (id, body) = check_signed(&secret, got);
    assert_eq!(body["type"], "job.canceled");
    // the secret it replaced no longer signs
    let h = |k: &str| got.headers.get(k).unwrap().to_str().unwrap().to_string();
    assert!(!rfpii_api::webhooks::verify(&old, &id, h("webhook-timestamp").parse().unwrap(), got.body.as_bytes(), &h("webhook-signature")));
    t.stop().await;
}

#[tokio::test(flavor = "multi_thread")]
async fn a_delivery_that_keeps_failing_is_given_up_after_its_retries() {
    let t = TestServer::start(1, |c| c.webhook_retry_schedule = vec![1, 1]).await;
    t.fake("FORMS=1\nRELEASED=0\nEXIT=1\nNO_REPORT=1\n");
    let rx = Receiver::start(100, json!({})).await;
    let (_, key) = t.account("free", true).await;
    t.patch(&key, "/v1/account/settings", &json!({"webhook_url": format!("{}/hook", rx.url)})).await;
    let j = t.start_job(&key).await;
    t.wait_end(&key, j["id"].as_str().unwrap()).await;
    rx.wait_for(3).await;
    let start = std::time::Instant::now();
    loop {
        let (status, attempts, last): (String, i64, Option<i64>) =
            t.state.db.call(|c| Ok(c.query_row("SELECT status, attempts, last_status FROM webhook_deliveries", [], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)))?)).await.unwrap();
        if status == "failed" {
            assert_eq!((attempts, last), (3, Some(500)));
            break;
        }
        assert!(start.elapsed().as_secs() < 10, "{status} {attempts}");
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    }
    let body: Value = serde_json::from_str(&rx.all()[0].body).unwrap();
    assert_eq!(body["type"], "job.failed");
    assert_eq!(rx.count(), 3);
    t.stop().await;
}
