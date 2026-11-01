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

pub fn account_value(a: &Account) -> Value {
    json!({"id": a.id, "object": "account", "name": a.name, "email": a.email, "plan": a.plan, "spend_cap_microusd": a.spend_cap_microusd,
           "ready_for_jobs": accounts::missing_for_jobs(a).is_empty(), "stripe_customer": a.stripe_customer_id.is_some(), "created_at": crate::clock::rfc3339(a.created_at)})
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
    let all = db(&s, |c| accounts::list(c)).await?;
    Ok(axum::Json(json!({"object": "list", "data": all.iter().map(account_value).collect::<Vec<_>>(), "has_more": false})))
}

#[derive(Deserialize, Debug)]
#[serde(deny_unknown_fields)]
pub struct NewAccount {
    pub name: String,
    pub email: Option<String>,
    #[serde(default = "free")]
    pub plan: String,
}

fn free() -> String {
    "free".into()
}

/// `POST /v1/admin/accounts`.
pub async fn create_account(State(s): State<AppState>, ApiJson(a): ApiJson<NewAccount>) -> ApiResult<Response> {
    if !accounts::is_plan(&a.plan) || a.name.trim().is_empty() {
        return Err(ApiError::validation("an account needs a name, and plan free or paid"));
    }
    let now = s.clock.now();
    let acct = db(&s, move |c| accounts::create(c, now, &a.name, a.email.as_deref(), &a.plan, ACTOR)).await?;
    Ok((StatusCode::CREATED, axum::Json(account_value(&acct))).into_response())
}

#[derive(Deserialize, Debug)]
#[serde(deny_unknown_fields)]
pub struct Plan {
    pub plan: String,
    /// micro-dollars a month; null for no cap; left out to keep it
    #[serde(default, with = "cap")]
    pub spend_cap_microusd: Option<Option<i64>>,
}

mod cap {
    use serde::{Deserialize, Deserializer};
    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Option<Option<i64>>, D::Error> {
        Option::<i64>::deserialize(d).map(Some)
    }
}

/// `PUT /v1/admin/accounts/{id}/plan`.
pub async fn set_plan(State(s): State<AppState>, Path(id): Path<String>, ApiJson(p): ApiJson<Plan>) -> ApiResult<axum::Json<Value>> {
    if !accounts::is_plan(&p.plan) || p.spend_cap_microusd.flatten().is_some_and(|c| c < 0) {
        return Err(ApiError::validation("plan must be free or paid, and a spend cap not negative"));
    }
    let now = s.clock.now();
    let a = db(&s, move |c| {
        if !accounts::set_plan(c, now, &id, &p.plan, p.spend_cap_microusd, ACTOR)? {
            return Ok(None);
        }
        accounts::load(c, &id)
    })
    .await?
    .ok_or_else(|| ApiError::not_found("account"))?;
    Ok(axum::Json(account_value(&a)))
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
