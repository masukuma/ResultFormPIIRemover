//! Stripe, when configured: a Checkout Session in payment mode for a
//! purchase of packages (the packages and the card fee as two lines, each
//! with the VAT tax rate), and the webhook that marks the purchase paid,
//! and adds its credits, once Stripe says the session was paid for the
//! quoted amount. Without Stripe, the operator records packages paid
//! otherwise (`rfpii-api admin add-credits`).

use anyhow::Result;
use axum::http::StatusCode;
use hmac::{KeyInit, Mac};
use rusqlite::params;
use serde_json::{Value, json};

use crate::config::Config;
use crate::problem::ApiError;
use crate::purchases::Purchase;
use crate::server::AppState;

type HmacSha256 = hmac::Hmac<sha2::Sha256>;

/// Whether Stripe Checkout is configured: a secret key, and the tax rate
/// that puts VAT on each line.
pub fn checkout_configured(cfg: &Config) -> bool {
    cfg.stripe_secret_key.is_some() && cfg.stripe_tax_rate_id.is_some()
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

/// 501 when Checkout is not configured.
pub fn not_configured() -> ApiError {
    ApiError::new(
        StatusCode::NOT_IMPLEMENTED,
        Some("billing-not-configured"),
        "Billing not configured",
        "this server takes no card payments (it needs STRIPE_SECRET_KEY and STRIPE_TAX_RATE_ID); ask the operator, who adds packages paid otherwise",
    )
}

/// The form fields of a Checkout Session for a pending purchase: payment
/// mode; the packages (the package price, `packages` of them) and the card
/// fee as lines priced here, each with the VAT tax rate; the account as
/// `client_reference_id`; the purchase and its packages in the metadata.
pub fn session_form(cfg: &Config, account: &crate::accounts::Account, p: &Purchase, back: ReturnUrls) -> Vec<(String, String)> {
    let tax = cfg.stripe_tax_rate_id.clone().unwrap_or_default();
    let base = cfg.public_base_url.trim_end_matches('/');
    let q = &p.quote;
    let mut form: Vec<(String, String)> = vec![
        ("mode".into(), "payment".into()),
        ("client_reference_id".into(), account.id.clone()),
        ("metadata[purchase_id]".into(), p.id.clone()),
        ("metadata[packages]".into(), q.packages.to_string()),
        ("metadata[account_id]".into(), account.id.clone()),
        ("payment_intent_data[metadata][purchase_id]".into(), p.id.clone()),
        ("line_items[0][price_data][currency]".into(), p.currency.clone()),
        ("line_items[0][price_data][unit_amount]".into(), q.package_price_cents.to_string()),
        ("line_items[0][price_data][product_data][name]".into(), format!("{}: package of {} forms", cfg.platform_name, q.package_forms)),
        ("line_items[0][quantity]".into(), q.packages.to_string()),
        ("line_items[0][tax_rates][0]".into(), tax.clone()),
    ];
    if q.card_fee_cents > 0 {
        form.extend([
            ("line_items[1][price_data][currency]".into(), p.currency.clone()),
            ("line_items[1][price_data][unit_amount]".into(), q.card_fee_cents.to_string()),
            ("line_items[1][price_data][product_data][name]".into(), format!("Card processing fee ({} of the charge)", cfg.rates().card_fee_terms())),
            ("line_items[1][quantity]".into(), "1".into()),
            ("line_items[1][tax_rates][0]".into(), tax),
        ]);
    }
    form.push(("success_url".into(), back.success.unwrap_or_else(|| format!("{base}/v1/pricing?checkout=success"))));
    form.push(("cancel_url".into(), back.cancel.unwrap_or_else(|| format!("{base}/v1/pricing?checkout=canceled"))));
    match (&account.stripe_customer_id, &account.email) {
        (Some(c), _) => form.push(("customer".into(), c.clone())),
        (None, Some(e)) => form.push(("customer_email".into(), e.clone())),
        _ => {}
    }
    form
}

/// Create the Checkout Session for a pending purchase; its id and URL.
pub async fn create_session(state: &AppState, account: &crate::accounts::Account, p: &Purchase, back: ReturnUrls) -> Result<(String, String), ApiError> {
    let Some(key) = &state.cfg.stripe_secret_key else { return Err(not_configured()) };
    let form = session_form(&state.cfg, account, p, back);
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
    match (status.is_success(), body["id"].as_str(), body["url"].as_str()) {
        (true, Some(id), Some(url)) => Ok((id.to_string(), url.to_string())),
        _ => {
            tracing::warn!(status = status.as_u16(), "stripe checkout session refused");
            Err(ApiError::new(StatusCode::BAD_GATEWAY, Some("billing-unavailable"), "Billing unavailable", format!("Stripe answered {status} without a session")))
        }
    }
}

/// Act on a verified Stripe event; whether it changed anything. Each event
/// is handled once (its id is recorded), and each purchase is paid once.
pub async fn handle_event(state: &AppState, event: Value) -> Result<bool> {
    let now = state.clock.now();
    let prices = state.prices.clone();
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
            let session = &event["data"]["object"];
            let changed = match kind.as_str() {
                "checkout.session.completed" | "checkout.session.async_payment_succeeded" => crate::purchases::complete(&tx, now, &prices, session)?,
                "checkout.session.expired" => crate::purchases::expire(&tx, now, session)?,
                _ => false,
            };
            tx.commit()?;
            Ok(changed)
        })
        .await
}

/// The purchase as a Checkout answer gives it.
pub fn session_value(id: &str, url: &str, p: &Purchase) -> Value {
    json!({"object": "checkout_session", "id": id, "url": url, "purchase": p.resource()})
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
}
