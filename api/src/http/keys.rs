//! `/v1/keys`: the account's API keys. A new key's secret is returned
//! once; only its SHA-256 and prefix are kept.

use axum::Extension;
use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use serde::Deserialize;
use serde_json::{Value, json};

use super::{ApiJson, db};
use crate::auth::{self, Caller, Scope};
use crate::problem::{ApiError, ApiResult};
use crate::server::AppState;

/// `GET /v1/keys`.
pub async fn list(State(s): State<AppState>, Extension(caller): Extension<Caller>) -> ApiResult<axum::Json<Value>> {
    caller.require(Scope::AccountAdmin)?;
    let account = caller.account_id.clone();
    let keys = db(&s, move |c| auth::list(c, &account)).await?;
    Ok(axum::Json(json!({"object": "list", "data": keys.iter().map(auth::KeyRow::to_value).collect::<Vec<_>>(), "has_more": false})))
}

#[derive(Deserialize, Debug, Default)]
#[serde(deny_unknown_fields)]
pub struct NewKey {
    pub name: Option<String>,
    /// default: jobs:read and jobs:write
    pub scopes: Option<Vec<String>>,
}

/// `POST /v1/keys`.
pub async fn create(State(s): State<AppState>, Extension(caller): Extension<Caller>, ApiJson(k): ApiJson<NewKey>) -> ApiResult<Response> {
    caller.require(Scope::AccountAdmin)?;
    let scopes = auth::parse_scopes(&k.scopes.unwrap_or_else(|| vec!["jobs:read".into(), "jobs:write".into()]).join(" ")).map_err(ApiError::validation)?;
    if k.name.as_ref().is_some_and(|n| n.len() > 100) {
        return Err(ApiError::validation("name is limited to 100 characters"));
    }
    let (account, now, actor, name) = (caller.account_id.clone(), s.clock.now(), caller.actor(), k.name);
    let (row, secret) = db(&s, move |c| auth::create(c, now, &account, name.as_deref(), &scopes, &actor)).await?;
    let mut v = row.to_value();
    v["secret"] = json!(secret);
    Ok((StatusCode::CREATED, axum::Json(v)).into_response())
}

/// `DELETE /v1/keys/{id}`: revoke a key; it is refused from then on.
pub async fn revoke(State(s): State<AppState>, Extension(caller): Extension<Caller>, Path(id): Path<String>) -> ApiResult<axum::Json<Value>> {
    caller.require(Scope::AccountAdmin)?;
    let (account, now, actor) = (caller.account_id.clone(), s.clock.now(), caller.actor());
    let row = db(&s, move |c| auth::revoke(c, now, &account, &id, &actor)).await?.ok_or_else(|| ApiError::not_found("API key"))?;
    Ok(axum::Json(row.to_value()))
}
