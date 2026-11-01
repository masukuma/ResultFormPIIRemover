//! Pricing, quotes, usage, purchases and Stripe: `/v1/pricing` and
//! `/v1/billing/quote` (public), `/v1/usage`, `/v1/billing/checkout`,
//! `/v1/billing/purchases`, and the Stripe webhook.

use axum::Extension;
use axum::body::Bytes;
use axum::extract::State;
use axum::http::{HeaderMap, StatusCode};
use axum::response::Response;
use serde::Deserialize;
use serde_json::{Value, json};

use super::{ApiQuery, PageQuery, db, page};
use crate::auth::{Caller, Scope};
use crate::billing::format_usd;
use crate::problem::{ApiError, ApiResult};
use crate::quote::{format_cents, percent};
use crate::server::AppState;

/// `GET /v1/pricing`: the free forms, the package, the card fee and VAT,
/// and the quote for one package; public.
pub async fn pricing(State(s): State<AppState>) -> axum::Json<Value> {
    let p = &s.prices;
    let r = &p.rates;
    axum::Json(json!({
        "object": "pricing",
        "currency": "usd",
        "unit": "form",
        "form": "a document as rfpii inventory counts it: a file, or one page of a scanned batch read page by page; only forms a job releases are counted",
        "free": {"forms_per_month": p.free_forms_per_month, "period": "calendar month, UTC"},
        "package": {
            "forms": r.package_forms,
            "price_cents": r.package_price_cents,
            "price_usd": format_cents(r.package_price_cents),
            "price_per_form_usd": format_usd(r.price_per_form_microusd()),
            "max_per_purchase": p.max_packages_per_purchase,
            "credits_expire_after_days": (p.credit_expiry_days > 0).then_some(p.credit_expiry_days),
        },
        "card_fee": {
            "basis_points": r.card_fee_basis_points,
            "fixed_cents": r.card_fee_fixed_cents,
            "terms": r.card_fee_terms(),
            "paid_by": "the customer: a line of its own, the smallest amount with which the operator keeps the packages' price after the card processor's fee",
        },
        "vat": {"rate_basis_points": r.vat_basis_points, "rate": percent(r.vat_basis_points), "on": ["packages", "card_fee"], "included_in_prices": false},
        "quote": r.quote(1).to_value(r),
        "checkout": crate::stripe::checkout_configured(&s.cfg),
        "how_the_price_was_set": s.cfg.pricing_url,
    }))
}

#[derive(Deserialize, Debug)]
#[serde(deny_unknown_fields)]
pub struct QuoteQuery {
    /// packages to buy; 1 if left out
    pub packages: Option<i64>,
}

/// 400 unless `n` packages may be bought at once.
fn check_packages(s: &AppState, n: i64) -> Result<(), String> {
    let max = s.prices.max_packages_per_purchase;
    if (1..=max).contains(&n) { Ok(()) } else { Err(format!("packages must be from 1 to {max}")) }
}

/// `GET /v1/billing/quote?packages=n`: what `n` packages cost, line by
/// line; public.
pub async fn quote(State(s): State<AppState>, ApiQuery(q): ApiQuery<QuoteQuery>) -> ApiResult<axum::Json<Value>> {
    let n = q.packages.unwrap_or(1);
    check_packages(&s, n).map_err(ApiError::bad_request)?;
    Ok(axum::Json(s.prices.rates.quote(n).to_value(&s.prices.rates)))
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
    let now = s.clock.now();
    let period = q.period.unwrap_or_else(|| crate::clock::period_of(now));
    if !crate::clock::is_period(&period) {
        return Err(ApiError::bad_request("period must be a month as YYYY-MM"));
    }
    let (account, prices) = (caller.account_id.clone(), s.prices.clone());
    Ok(axum::Json(db(&s, move |c| crate::billing::usage(c, &account, &period, now, &prices)).await?))
}

/// `POST /v1/billing/checkout`'s body.
#[derive(Deserialize, Debug)]
#[serde(deny_unknown_fields)]
pub struct CheckoutRequest {
    pub packages: i64,
    /// the client's pages Stripe Checkout returns the customer to
    pub success_url: Option<String>,
    pub cancel_url: Option<String>,
}

/// `POST /v1/billing/checkout`: a pending purchase of `packages` packages
/// at today's quote, and the Stripe Checkout Session (payment mode) that
/// pays for it; 501 without Stripe.
pub async fn checkout(State(s): State<AppState>, Extension(caller): Extension<Caller>, body: Bytes) -> ApiResult<axum::Json<Value>> {
    caller.require(Scope::AccountAdmin)?;
    if !crate::stripe::checkout_configured(&s.cfg) {
        return Err(crate::stripe::not_configured());
    }
    let req: CheckoutRequest = serde_json::from_slice(&body).map_err(|e| ApiError::validation(format!("{e}")))?;
    check_packages(&s, req.packages).map_err(ApiError::validation)?;
    for u in [&req.success_url, &req.cancel_url].into_iter().flatten() {
        crate::webhooks::check_url(u, s.cfg.webhook_allow_private).map_err(|e| ApiError::validation(e.replace("webhook_url", "a return URL")))?;
    }
    let quote = s.prices.rates.quote(req.packages);
    let (id, now, actor) = (caller.account_id.clone(), s.clock.now(), caller.actor());
    let (account, purchase) = db(&s, move |c| {
        let a = crate::accounts::load(c, &id)?.ok_or_else(|| anyhow::anyhow!("account {id} vanished"))?;
        let p = crate::purchases::create_pending(c, now, &id, &quote, &actor)?;
        Ok((a, p))
    })
    .await?;
    let back = crate::stripe::ReturnUrls { success: req.success_url, cancel: req.cancel_url };
    match crate::stripe::create_session(&s, &account, &purchase, back).await {
        Ok((session, url)) => {
            let (pid, sid) = (purchase.id.clone(), session.clone());
            let purchase = db(&s, move |c| {
                crate::purchases::set_session(c, &pid, &sid)?;
                crate::purchases::load_any(c, &pid)?.ok_or_else(|| anyhow::anyhow!("purchase {pid} vanished"))
            })
            .await?;
            Ok(axum::Json(crate::stripe::session_value(&session, &url, &purchase)))
        }
        Err(e) => {
            // no session, nothing to pay: the purchase goes
            let pid = purchase.id.clone();
            db(&s, move |c| crate::purchases::discard(c, &pid)).await?;
            Err(e)
        }
    }
}

/// `GET /v1/billing/purchases`: the account's purchases, newest first,
/// with their lines and status. A key that reads jobs or administers the
/// account may list them.
pub async fn purchases(State(s): State<AppState>, Extension(caller): Extension<Caller>, ApiQuery(q): ApiQuery<PageQuery>) -> ApiResult<Response> {
    if caller.require(Scope::JobsRead).is_err() {
        caller.require(Scope::AccountAdmin)?;
    }
    let limit = q.limit()?;
    let (account, after) = (caller.account_id.clone(), q.after.clone());
    let (rows, more) = db(&s, move |c| crate::purchases::list(c, &account, limit, after.as_deref())).await?;
    Ok(page("/v1/billing/purchases", rows.iter().map(crate::purchases::Purchase::resource).collect(), more, limit))
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
