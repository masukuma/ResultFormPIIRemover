//! Stripe, when configured: a Checkout Session for the metered price, the
//! webhook that sets an account's plan, and paid usage reported as billing
//! meter events. Each meter event's `identifier` is the usage record's id,
//! so a retry is never counted twice. Without Stripe the operator sets
//! plans and invoices from `GET /v1/usage`.

use anyhow::{Context, Result};
use axum::http::StatusCode;
use hmac::{KeyInit, Mac};
use rusqlite::params;
use serde_json::{Value, json};

use crate::problem::ApiError;
use crate::server::AppState;

type HmacSha256 = hmac::Hmac<sha2::Sha256>;

/// Seconds before a failed meter event is tried again: 1, 5, 30 minutes,
/// then hourly.
const METER_RETRY: [i64; 4] = [60, 300, 1800, 3600];

/// Whether Stripe Checkout is configured.
pub fn checkout_configured(state: &AppState) -> bool {
    state.cfg.stripe_secret_key.is_some() && state.cfg.stripe_price_id.is_some()
}

/// The `Stripe-Signature` value for a payload (tests and tools).
pub fn signature_header(secret: &str, timestamp: i64, body: &[u8]) -> String {
    format!("t={timestamp},v1={}", v1(secret, timestamp, body))
}

fn v1(secret: &str, timestamp: i64, body: &[u8]) -> String {
    let mut mac = <HmacSha256 as KeyInit>::new_from_slice(secret.as_bytes()).expect("HMAC takes any key length");
    mac.update(format!("{timestamp}.").as_bytes());
    mac.update(body);
    crate::ids::hex(&mac.finalize().into_bytes())
}

/// Check a `Stripe-Signature` header: a `t=` timestamp within `tolerance`
/// seconds of `now`, and a `v1=` HMAC-SHA256 of "t.body" with the
/// endpoint's secret, compared in constant time.
pub fn verify_signature(header: &str, body: &[u8], secret: &str, now: i64, tolerance: i64) -> Result<(), &'static str> {
    let mut t: Option<i64> = None;
    let mut sigs = Vec::new();
    for part in header.split(',') {
        match part.trim().split_once('=') {
            Some(("t", v)) => t = v.parse().ok(),
            Some(("v1", v)) => sigs.push(v.to_string()),
            _ => {}
        }
    }
    let t = t.ok_or("no timestamp in Stripe-Signature")?;
    if (now - t).abs() > tolerance {
        return Err("the Stripe-Signature timestamp is outside the tolerance");
    }
    let expected = v1(secret, t, body);
    if sigs.iter().any(|s| crate::ids::ct_eq(s.as_bytes(), expected.as_bytes())) { Ok(()) } else { Err("no valid v1 signature in Stripe-Signature") }
}

/// Where Stripe Checkout sends the customer back: the client's pages, or
/// the API's public pricing.
#[derive(Debug, Default)]
pub struct ReturnUrls {
    pub success: Option<String>,
    pub cancel: Option<String>,
}

/// Create a Checkout Session subscribing `account` to the metered price.
pub async fn checkout(state: &AppState, account: &crate::accounts::Account, back: ReturnUrls) -> Result<Value, ApiError> {
    let (Some(key), Some(price)) = (&state.cfg.stripe_secret_key, &state.cfg.stripe_price_id) else {
        return Err(ApiError::new(
            StatusCode::NOT_IMPLEMENTED,
            Some("billing-not-configured"),
            "Billing not configured",
            "this server has no Stripe account; the operator sets plans and invoices from /v1/usage",
        ));
    };
    let base = state.cfg.public_base_url.trim_end_matches('/');
    let mut form: Vec<(String, String)> = vec![
        ("mode".into(), "subscription".into()),
        ("line_items[0][price]".into(), price.clone()),
        ("client_reference_id".into(), account.id.clone()),
        ("metadata[account_id]".into(), account.id.clone()),
        ("subscription_data[metadata][account_id]".into(), account.id.clone()),
        ("success_url".into(), back.success.unwrap_or_else(|| format!("{base}/v1/pricing?checkout=success"))),
        ("cancel_url".into(), back.cancel.unwrap_or_else(|| format!("{base}/v1/pricing?checkout=canceled"))),
    ];
    match (&account.stripe_customer_id, &account.email) {
        (Some(c), _) => form.push(("customer".into(), c.clone())),
        (None, Some(e)) => form.push(("customer_email".into(), e.clone())),
        _ => {}
    }
    let res = state
        .http
        .post(format!("{}/v1/checkout/sessions", state.cfg.stripe_api_base.trim_end_matches('/')))
        .bearer_auth(key)
        .form(&form)
        .send()
        .await
        .map_err(|e| ApiError::new(StatusCode::BAD_GATEWAY, Some("billing-unavailable"), "Billing unavailable", format!("Stripe could not be reached: {e}")))?;
    let status = res.status();
    let body: Value = res.json().await.unwrap_or(Value::Null);
    if !status.is_success() {
        tracing::warn!(status = status.as_u16(), "stripe checkout session refused");
        return Err(ApiError::new(StatusCode::BAD_GATEWAY, Some("billing-unavailable"), "Billing unavailable", format!("Stripe answered {status}")));
    }
    Ok(json!({"object": "checkout_session", "id": body["id"], "url": body["url"]}))
}

/// Act on a verified Stripe event; whether it changed anything. Each event
/// is handled once (its id is recorded).
pub async fn handle_event(state: &AppState, event: Value) -> Result<bool> {
    let now = state.clock.now();
    state
        .db
        .call(move |c| {
            let tx = c.transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;
            let id = event["id"].as_str().unwrap_or("").to_string();
            let kind = event["type"].as_str().unwrap_or("").to_string();
            let fresh = tx.execute("INSERT OR IGNORE INTO stripe_events (id, type, received_at) VALUES (?1, ?2, ?3)", params![id, kind, now])? > 0;
            if !fresh {
                return Ok(false);
            }
            let changed = apply_event(&tx, now, &kind, &event["data"]["object"])?;
            tx.commit()?;
            Ok(changed)
        })
        .await
}

/// The plan a subscription in `status` gives.
fn plan_of(status: &str) -> &'static str {
    match status {
        "active" | "trialing" | "past_due" => "paid",
        _ => "free",
    }
}

fn apply_event(tx: &rusqlite::Transaction, now: i64, kind: &str, obj: &Value) -> Result<bool> {
    let set = |account: &str, plan: &str, customer: Option<&str>, subscription: Option<&str>| -> Result<bool> {
        let n = tx.execute(
            "UPDATE accounts SET plan = ?2, stripe_customer_id = COALESCE(?3, stripe_customer_id), stripe_subscription_id = COALESCE(?4, stripe_subscription_id) WHERE id = ?1",
            params![account, plan, customer, subscription],
        )?;
        if n > 0 {
            crate::audit::record(tx, now, Some(account), "stripe", "account.plan_set", &json!({"plan": plan, "event": kind}))?;
        }
        Ok(n > 0)
    };
    match kind {
        "checkout.session.completed" => {
            let account = obj["client_reference_id"].as_str().or(obj["metadata"]["account_id"].as_str()).context("a checkout session without the account")?;
            set(account, "paid", obj["customer"].as_str(), obj["subscription"].as_str())
        }
        "customer.subscription.updated" | "customer.subscription.deleted" => {
            let Some(customer) = obj["customer"].as_str() else { return Ok(false) };
            let Some(a) = crate::accounts::by_stripe_customer(tx, customer)? else { return Ok(false) };
            let plan = if kind.ends_with("deleted") { "free" } else { plan_of(obj["status"].as_str().unwrap_or("")) };
            set(&a.id, plan, None, obj["id"].as_str())
        }
        _ => Ok(false),
    }
}

struct Unreported {
    id: String,
    customer: String,
    forms: i64,
    created_at: i64,
    attempts: i64,
}

/// Report paid usage not yet reported as meter events; how many records
/// were tried. Records of accounts without a Stripe customer wait (the
/// operator invoices them from /v1/usage).
pub async fn report_due(state: &AppState) -> Result<usize> {
    let Some(key) = state.cfg.stripe_secret_key.clone() else { return Ok(0) };
    let now = state.clock.now();
    let due = state
        .db
        .call(move |c| {
            let mut st = c.prepare(
                "UPDATE usage_records SET stripe_claimed_until = ?1 + 120
                 WHERE id IN (SELECT u.id FROM usage_records u JOIN accounts a ON a.id = u.account_id
                              WHERE u.plan = 'paid' AND u.forms > 0 AND u.stripe_reported_at IS NULL AND u.stripe_error IS NULL
                                AND u.stripe_next_at <= ?1 AND (u.stripe_claimed_until IS NULL OR u.stripe_claimed_until < ?1)
                                AND a.stripe_customer_id IS NOT NULL LIMIT 50)
                 RETURNING id, (SELECT stripe_customer_id FROM accounts WHERE accounts.id = usage_records.account_id), forms, created_at, stripe_attempts",
            )?;
            let rows = st
                .query_map([now], |r| Ok(Unreported { id: r.get(0)?, customer: r.get(1)?, forms: r.get(2)?, created_at: r.get(3)?, attempts: r.get(4)? }))?
                .collect::<rusqlite::Result<Vec<_>>>()?;
            Ok(rows)
        })
        .await?;
    let n = due.len();
    for u in due {
        let outcome = send_meter_event(state, &key, &u).await;
        settle_report(state, u, outcome).await?;
    }
    Ok(n)
}

/// What Stripe said: reported, try again later, or refused for good.
enum Outcome {
    Reported,
    Retry(String),
    Refused(String),
}

async fn send_meter_event(state: &AppState, key: &str, u: &Unreported) -> Outcome {
    let form = [
        ("event_name", state.cfg.stripe_meter_event_name.clone()),
        ("payload[stripe_customer_id]", u.customer.clone()),
        ("payload[value]", u.forms.to_string()),
        ("identifier", u.id.clone()),
        ("timestamp", u.created_at.to_string()),
    ];
    let url = format!("{}/v1/billing/meter_events", state.cfg.stripe_api_base.trim_end_matches('/'));
    let res = match state.http.post(url).bearer_auth(key).form(&form).send().await {
        Ok(r) => r,
        Err(e) => return Outcome::Retry(format!("{e}")),
    };
    let status = res.status();
    let body = res.text().await.unwrap_or_default();
    if status.is_success() {
        Outcome::Reported
    } else if status.as_u16() == 429 || status.is_server_error() {
        Outcome::Retry(format!("Stripe answered {status}"))
    } else if body.contains("identifier") && (body.contains("already") || body.contains("duplicate")) {
        // a retry of an event Stripe already has: it is counted once
        Outcome::Reported
    } else {
        Outcome::Refused(format!("Stripe answered {status}: {}", body.chars().take(300).collect::<String>()))
    }
}

async fn settle_report(state: &AppState, u: Unreported, outcome: Outcome) -> Result<()> {
    let now = state.clock.now();
    state
        .db
        .call(move |c| {
            let attempts = u.attempts + 1;
            match outcome {
                Outcome::Reported => {
                    c.execute("UPDATE usage_records SET stripe_reported_at = ?2, stripe_attempts = ?3, stripe_claimed_until = NULL WHERE id = ?1", params![u.id, now, attempts])?
                }
                Outcome::Retry(why) => {
                    let wait = METER_RETRY[(attempts as usize - 1).min(METER_RETRY.len() - 1)];
                    tracing::warn!(usage_record = %u.id, error = %why, "meter event not reported; will retry");
                    c.execute("UPDATE usage_records SET stripe_attempts = ?2, stripe_next_at = ?3, stripe_claimed_until = NULL WHERE id = ?1", params![u.id, attempts, now + wait])?
                }
                Outcome::Refused(why) => {
                    tracing::error!(usage_record = %u.id, error = %why, "meter event refused by Stripe");
                    c.execute("UPDATE usage_records SET stripe_attempts = ?2, stripe_error = ?3, stripe_claimed_until = NULL WHERE id = ?1", params![u.id, attempts, why])?
                }
            };
            Ok(())
        })
        .await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_stripe_signature_is_checked_with_its_tolerance() {
        let body = br#"{"id":"evt_1","type":"checkout.session.completed"}"#;
        let h = signature_header("whsec_test", 1_000, body);
        assert!(verify_signature(&h, body, "whsec_test", 1_100, 300).is_ok());
        assert!(verify_signature(&h, body, "whsec_test", 1_400, 300).is_err());
        assert!(verify_signature(&h, body, "whsec_other", 1_100, 300).is_err());
        assert!(verify_signature(&h, b"{}", "whsec_test", 1_100, 300).is_err());
        assert!(verify_signature("v1=abc", body, "whsec_test", 1_100, 300).is_err());
        // several signatures (a secret being rolled): one valid is enough
        let two = format!("t=1000,v1=00,{}", h.split(',').nth(1).unwrap());
        assert!(verify_signature(&two, body, "whsec_test", 1_000, 300).is_ok());
    }

    #[test]
    fn subscriptions_give_the_paid_plan_while_they_stand() {
        assert_eq!(plan_of("active"), "paid");
        assert_eq!(plan_of("past_due"), "paid");
        assert_eq!(plan_of("canceled"), "free");
        assert_eq!(plan_of("incomplete_expired"), "free");
    }
}
