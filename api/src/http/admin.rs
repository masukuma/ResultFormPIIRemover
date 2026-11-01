//! `/v1/admin/*`: what the `admin` command does, over HTTP, for a bearer of
//! ADMIN_TOKEN. Without ADMIN_TOKEN these routes answer 404.

use axum::extract::{Path, Request, State};
use axum::http::{StatusCode, header};
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};
use serde::Deserialize;
use serde_json::{Value, json};

use super::{ApiJson, db};
use crate::accounts::{self, Account};
use crate::auth;
use crate::problem::{ApiError, ApiResult};
use crate::server::AppState;

const ACTOR: &str = "admin-api";

/// Let the request through only with `Authorization: Bearer ADMIN_TOKEN`,
/// compared in constant time.
pub async fn require_token(State(s): State<AppState>, req: Request, next: Next) -> Response {
    let Some(token) = &s.cfg.admin_token else {
        return ApiError::new(StatusCode::NOT_FOUND, None, "Not Found", "no such route; see /v1/openapi.json").into_response();
    };
    let given = req.headers().get(header::AUTHORIZATION).and_then(|v| v.to_str().ok()).and_then(|v| v.strip_prefix("Bearer ")).unwrap_or("");
    if !crate::ids::ct_eq(given.as_bytes(), token.as_bytes()) {
        return ApiError::unauthorized("the admin token is required", Some("invalid_token")).into_response();
    }
    next.run(req).await
}

/// An account as the operator lists it, with its credits.
pub fn account_value(a: &Account, credits: &crate::billing::Credits, prices: &crate::billing::Prices) -> Value {
    json!({"id": a.id, "object": "account", "name": a.name, "email": a.email, "credits": credits.to_value(prices),
           "ready_for_jobs": accounts::missing_for_jobs(a).is_empty(), "stripe_customer": a.stripe_customer_id.is_some(), "created_at": crate::clock::rfc3339(a.created_at)})
}

/// Every account, with its credits at `now`.
pub fn accounts_with_credits(c: &rusqlite::Connection, now: i64) -> anyhow::Result<Vec<(Account, crate::billing::Credits)>> {
    let mut out = Vec::new();
    for a in accounts::list(c)? {
        let credits = crate::billing::credits(c, &a.id, now)?;
        out.push((a, credits));
    }
    Ok(out)
}

/// `GET /v1/admin/jobs/{id}`: a job, whoever's it is, with what it cost.
pub async fn get_job(State(s): State<AppState>, Path(id): Path<String>) -> ApiResult<axum::Json<Value>> {
    match db(&s, move |c| crate::jobs::load_any(c, &id)).await? {
        Some(job) => Ok(axum::Json(job.operator_resource())),
        None => Err(ApiError::new(StatusCode::NOT_FOUND, Some("not-found"), "Not found", "no such job")),
    }
}

/// `GET /v1/admin/accounts`.
pub async fn list_accounts(State(s): State<AppState>) -> ApiResult<axum::Json<Value>> {
    let now = s.clock.now();
    let all = db(&s, move |c| accounts_with_credits(c, now)).await?;
    let data: Vec<Value> = all.iter().map(|(a, cr)| account_value(a, cr, &s.prices)).collect();
    Ok(axum::Json(json!({"object": "list", "data": data, "has_more": false})))
}

#[derive(Deserialize, Debug)]
#[serde(deny_unknown_fields)]
pub struct NewAccount {
    pub name: String,
    pub email: Option<String>,
}

/// `POST /v1/admin/accounts`.
pub async fn create_account(State(s): State<AppState>, ApiJson(a): ApiJson<NewAccount>) -> ApiResult<Response> {
    if a.name.trim().is_empty() {
        return Err(ApiError::validation("an account needs a name"));
    }
    let now = s.clock.now();
    let acct = db(&s, move |c| accounts::create(c, now, &a.name, a.email.as_deref(), ACTOR)).await?;
    Ok((StatusCode::CREATED, axum::Json(account_value(&acct, &Default::default(), &s.prices))).into_response())
}

/// `POST /v1/admin/accounts/{id}/credits`'s body.
#[derive(Deserialize, Debug)]
#[serde(deny_unknown_fields)]
pub struct NewCredits {
    pub packages: i64,
    /// the payment's reference ("bank transfer 123")
    pub reference: String,
}

/// `POST /v1/admin/accounts/{id}/credits`: packages paid outside Stripe,
/// recorded as a paid purchase with VAT and no card fee, their forms added
/// to the account's credits.
pub async fn add_credits(State(s): State<AppState>, Path(id): Path<String>, ApiJson(n): ApiJson<NewCredits>) -> ApiResult<Response> {
    let (now, prices) = (s.clock.now(), s.prices.clone());
    let (p, credits) = db(&s, move |c| add_credits_now(c, now, &prices, &id, n.packages, &n.reference, ACTOR)).await?.map_err(Refused::problem)?;
    Ok((StatusCode::CREATED, axum::Json(json!({"purchase": p.resource(), "credits": credits.to_value(&s.prices)}))).into_response())
}

/// Why the operator's change to credits was refused.
#[derive(Debug)]
pub enum Refused {
    NoAccount,
    Invalid(String),
}

impl Refused {
    fn problem(self) -> ApiError {
        match self {
            Refused::NoAccount => ApiError::new(StatusCode::NOT_FOUND, Some("not-found"), "Not found", "no such account"),
            Refused::Invalid(why) => ApiError::validation(why),
        }
    }
}

impl std::fmt::Display for Refused {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Refused::NoAccount => f.write_str("no such account"),
            Refused::Invalid(why) => f.write_str(why),
        }
    }
}

/// Record packages paid outside Stripe for `account`, in one IMMEDIATE
/// transaction; the purchase and the account's credits after it.
pub fn add_credits_now(
    c: &mut rusqlite::Connection,
    now: i64,
    prices: &crate::billing::Prices,
    account: &str,
    packages: i64,
    reference: &str,
    actor: &str,
) -> anyhow::Result<Result<(crate::purchases::Purchase, crate::billing::Credits), Refused>> {
    let tx = c.transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;
    if accounts::load(&tx, account)?.is_none() {
        return Ok(Err(Refused::NoAccount));
    }
    let p = match crate::purchases::add_manual(&tx, now, prices, account, packages, reference, actor)? {
        Ok(p) => p,
        Err(why) => return Ok(Err(Refused::Invalid(why))),
    };
    let credits = crate::billing::credits(&tx, account, now)?;
    tx.commit()?;
    Ok(Ok((p, credits)))
}

/// `POST /v1/admin/accounts/{id}/credit-adjustments`'s body.
#[derive(Deserialize, Debug)]
#[serde(deny_unknown_fields)]
pub struct Adjustment {
    /// forms to add (positive) or take back (negative)
    pub forms: i64,
    pub reason: String,
}

/// `POST /v1/admin/accounts/{id}/credit-adjustments`: add or take back
/// credits with a reason (a goodwill grant; credits of a refunded
/// payment), recorded in the ledger and the audit log.
pub async fn adjust_credits(State(s): State<AppState>, Path(id): Path<String>, ApiJson(a): ApiJson<Adjustment>) -> ApiResult<Response> {
    let (now, prices) = (s.clock.now(), s.prices.clone());
    let credits = db(&s, move |c| adjust_credits_now(c, now, &prices, &id, a.forms, &a.reason, ACTOR)).await?.map_err(Refused::problem)?;
    Ok((StatusCode::CREATED, axum::Json(json!({"credits": credits.to_value(&s.prices)}))).into_response())
}

/// Adjust `account`'s credits by `forms` in one IMMEDIATE transaction; its
/// credits after.
pub fn adjust_credits_now(
    c: &mut rusqlite::Connection,
    now: i64,
    prices: &crate::billing::Prices,
    account: &str,
    forms: i64,
    reason: &str,
    actor: &str,
) -> anyhow::Result<Result<crate::billing::Credits, Refused>> {
    let tx = c.transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;
    if accounts::load(&tx, account)?.is_none() {
        return Ok(Err(Refused::NoAccount));
    }
    if let Err(why) = crate::billing::adjust(&tx, account, forms, reason, actor, now, prices)? {
        return Ok(Err(Refused::Invalid(why)));
    }
    let credits = crate::billing::credits(&tx, account, now)?;
    tx.commit()?;
    Ok(Ok(credits))
}

#[derive(Deserialize, Debug, Default)]
#[serde(deny_unknown_fields)]
pub struct NewKey {
    pub name: Option<String>,
    pub scopes: Option<Vec<String>>,
}

/// `POST /v1/admin/accounts/{id}/keys`: a key for the account (by default
/// with every scope); its secret is returned once.
pub async fn create_key(State(s): State<AppState>, Path(id): Path<String>, ApiJson(k): ApiJson<NewKey>) -> ApiResult<Response> {
    let scopes = match k.scopes {
        Some(list) => auth::parse_scopes(&list.join(" ")).map_err(ApiError::validation)?,
        None => auth::ALL_SCOPES.to_vec(),
    };
    let now = s.clock.now();
    let made = db(&s, move |c| {
        if accounts::load(c, &id)?.is_none() {
            return Ok(None);
        }
        Ok(Some(auth::create(c, now, &id, k.name.as_deref(), &scopes, ACTOR)?))
    })
    .await?
    .ok_or_else(|| ApiError::not_found("account"))?;
    let mut v = made.0.to_value();
    v["secret"] = json!(made.1);
    Ok((StatusCode::CREATED, axum::Json(v)).into_response())
}
