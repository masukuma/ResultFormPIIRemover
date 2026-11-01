//! Pricing, usage and Stripe: `/v1/pricing` (public), `/v1/usage`,
//! `/v1/billing/checkout`, and the Stripe webhook.

use axum::Extension;
use axum::body::Bytes;
use axum::extract::State;
use axum::http::{HeaderMap, StatusCode};
use serde::Deserialize;
use serde_json::{Value, json};

use super::{ApiQuery, db};
use crate::auth::{Caller, Scope};
use crate::billing::format_usd;
use crate::problem::{ApiError, ApiResult};
use crate::server::AppState;

/// `GET /v1/pricing`: the plans and the price per form, public.
pub async fn pricing(State(s): State<AppState>) -> axum::Json<Value> {
    let p = &s.prices;
    axum::Json(json!({
        "object": "pricing",
        "currency": "usd",
        "unit": "form",
        "form": "a document as rfpii inventory counts it: a file, or one page of a scanned batch read page by page; only forms a job releases are counted",
        "plans": [
            {"id": "free", "price_microusd": 0, "forms_per_month": p.free_forms_per_month, "period": "calendar month, UTC"},
            {"id": "paid", "billing": "metered subscription, billed monthly", "price_per_form_microusd": p.unit_microusd, "price_per_form_usd": format_usd(p.unit_microusd),
             "checkout": crate::stripe::checkout_configured(&s)},
        ],
        "how_the_price_was_set": s.cfg.pricing_url,
    }))
}

#[derive(Deserialize, Debug)]
#[serde(deny_unknown_fields)]
pub struct UsageQuery {
    /// YYYY-MM; the current month if left out
    pub period: Option<String>,
}

/// `GET /v1/usage?period=YYYY-MM`.
pub async fn usage(State(s): State<AppState>, Extension(caller): Extension<Caller>, ApiQuery(q): ApiQuery<UsageQuery>) -> ApiResult<axum::Json<Value>> {
    caller.require(Scope::JobsRead)?;
    let period = q.period.unwrap_or_else(|| crate::clock::period_of(s.clock.now()));
    if !crate::clock::is_period(&period) {
        return Err(ApiError::bad_request("period must be a month as YYYY-MM"));
    }
    let (account, prices) = (caller.account_id.clone(), s.prices.clone());
    Ok(axum::Json(db(&s, move |c| crate::billing::usage(c, &account, &period, &prices)).await?))
}

/// `POST /v1/billing/checkout`: a Stripe Checkout Session for the paid
/// plan; 501 without Stripe.
pub async fn checkout(State(s): State<AppState>, Extension(caller): Extension<Caller>, body: Bytes) -> ApiResult<axum::Json<Value>> {
    caller.require(Scope::AccountAdmin)?;
    let req: CheckoutRequest = if body.is_empty() { CheckoutRequest::default() } else { serde_json::from_slice(&body).map_err(|e| ApiError::validation(format!("{e}")))? };
    for u in [&req.success_url, &req.cancel_url].into_iter().flatten() {
        crate::webhooks::check_url(u, s.cfg.webhook_allow_private).map_err(|e| ApiError::validation(e.replace("webhook_url", "a return URL")))?;
    }
    let id = caller.account_id.clone();
    let account = db(&s, move |c| crate::accounts::load(c, &id)).await?.ok_or_else(|| ApiError::not_found("account"))?;
    let back = crate::stripe::ReturnUrls { success: req.success_url, cancel: req.cancel_url };
    Ok(axum::Json(crate::stripe::checkout(&s, &account, back).await?))
}

/// `POST /v1/billing/checkout`'s optional body: the client's pages Stripe
/// Checkout returns the customer to.
#[derive(Deserialize, Debug, Default)]
#[serde(deny_unknown_fields)]
pub struct CheckoutRequest {
    pub success_url: Option<String>,
    pub cancel_url: Option<String>,
}

/// `POST /v1/billing/stripe-webhook`: verified by its `Stripe-Signature`
/// (400 if it doesn't verify), then acted on once.
pub async fn stripe_webhook(State(s): State<AppState>, headers: HeaderMap, body: Bytes) -> ApiResult<axum::Json<Value>> {
    let Some(secret) = s.cfg.stripe_webhook_secret.clone() else {
        return Err(ApiError::new(StatusCode::NOT_IMPLEMENTED, Some("billing-not-configured"), "Billing not configured", "this server has no Stripe webhook secret"));
    };
    let sig = headers.get("stripe-signature").and_then(|v| v.to_str().ok()).unwrap_or("");
    crate::stripe::verify_signature(sig, &body, &secret, s.clock.now(), s.cfg.stripe_signature_tolerance_secs)
        .map_err(|why| ApiError::new(StatusCode::BAD_REQUEST, Some("invalid-signature"), "Invalid signature", why))?;
    let event: Value = serde_json::from_slice(&body).map_err(|e| ApiError::bad_request(format!("the event is not JSON: {e}")))?;
    let changed = crate::stripe::handle_event(&s, event).await.map_err(|e| ApiError::internal(&e))?;
    Ok(axum::Json(json!({"received": true, "changed": changed})))
}
