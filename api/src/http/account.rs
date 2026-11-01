//! `/v1/account`: the free forms left (given once), the form credits and
//! the purchases; the controller record and DPIA a job needs; settings (held
//! pages to the cloud only on the customer's opt-in, the webhook URL); the
//! webhook signing secret.

use axum::Extension;
use axum::extract::State;
use rusqlite::params;
use serde::Deserialize;
use serde_json::{Value, json};

use super::{ApiJson, db};
use crate::accounts::{self, Account, Dpia};
use crate::auth::{Caller, Scope};
use crate::billing::{self, Credits, Free, format_usd};
use crate::problem::{ApiError, ApiResult};
use crate::server::AppState;

/// What an account's resource shows of its billing.
pub struct Balance {
    pub period: String,
    pub free: Free,
    pub credits: Credits,
    pub purchases: Value,
    /// forms counted this month (free and credits)
    pub forms_this_month: i64,
}

/// The account's billing now.
pub fn balance(c: &rusqlite::Connection, account: &str, now: i64, prices: &billing::Prices) -> anyhow::Result<Balance> {
    let period = crate::clock::period_of(now);
    let forms_this_month: i64 = c.query_row("SELECT COALESCE(SUM(forms), 0) FROM usage_records WHERE account_id = ?1 AND period = ?2", params![account, period], |r| r.get(0))?;
    Ok(Balance {
        free: billing::free(c, account, prices)?,
        credits: billing::credits(c, account, now)?,
        purchases: crate::purchases::summary(c, account)?,
        forms_this_month,
        period,
    })
}

/// The account as the API returns it.
pub fn resource(s: &AppState, a: &Account, b: &Balance) -> Value {
    let now = s.clock.now();
    let controller = a.controller.as_ref().map(|c| accounts::effective_controller(c, &a.held_pages, a.dpia.as_ref(), &s.cfg));
    let dpia = a.dpia.as_ref().map(|d| {
        let mut d = d.clone();
        d["processing_may_start"] = json!(accounts::dpia_bars(&d, now).is_none());
        d["bars"] = json!(accounts::dpia_bars(&d, now));
        d
    });
    json!({
        "id": a.id, "object": "account", "name": a.name, "email": a.email,
        "period": b.period,
        "forms_released_this_month": b.forms_this_month,
        "free_forms": b.free.to_value(),
        "credits": b.credits.to_value(&s.prices),
        "purchases": b.purchases,
        "price": {"currency": "usd", "package_forms": s.prices.rates.package_forms, "package_price_cents": s.prices.rates.package_price_cents,
                  "price_per_form_usd": format_usd(s.prices.rates.price_per_form_microusd()), "quote": "/v1/billing/quote", "pricing": s.cfg.pricing_url},
        "settings": {"held_pages": a.held_pages, "webhook_url": a.webhook_url, "webhook_secret_set": a.webhook_secret.is_some()},
        "controller": controller,
        "dpia": dpia,
        "ready_for_jobs": accounts::missing_for_jobs(a).is_empty(),
        "missing": accounts::missing_for_jobs(a),
        "billing": {"stripe_customer": a.stripe_customer_id.is_some()},
        "created_at": crate::clock::rfc3339(a.created_at),
    })
}

async fn current(s: &AppState, caller: &Caller) -> ApiResult<Value> {
    let (id, now, prices) = (caller.account_id.clone(), s.clock.now(), s.prices.clone());
    let (a, b) = db(s, move |c| {
        let a = accounts::load(c, &id)?.ok_or_else(|| anyhow::anyhow!("account {id} vanished"))?;
        let b = balance(c, &id, now, &prices)?;
        Ok((a, b))
    })
    .await?;
    Ok(resource(s, &a, &b))
}

/// `GET /v1/account`: any valid key may read it.
pub async fn get(State(s): State<AppState>, Extension(caller): Extension<Caller>) -> ApiResult<axum::Json<Value>> {
    Ok(axum::Json(current(&s, &caller).await?))
}

/// `PUT /v1/account/controller`: the controller record, checked as `rfpii`
/// checks it once the platform is added as the controller's processor.
pub async fn put_controller(State(s): State<AppState>, Extension(caller): Extension<Caller>, ApiJson(body): ApiJson<Value>) -> ApiResult<axum::Json<Value>> {
    caller.require(Scope::AccountAdmin)?;
    let mut given = body;
    if !given.is_object() {
        return Err(ApiError::validation("the controller record must be a JSON object"));
    }
    if given.get("operator_role").is_some_and(|r| r != "processor") {
        return Err(ApiError::validation("operator_role is \"processor\": the platform processes the forms for the customer, who is the controller"));
    }
    given["operator_role"] = json!("processor");
    let effective = accounts::effective_controller(&given, "none", None, &s.cfg);
    accounts::check_controller(&effective).map_err(|e| ApiError::new(axum::http::StatusCode::UNPROCESSABLE_ENTITY, Some("invalid-controller"), "Invalid controller record", e))?;
    let (id, now, actor) = (caller.account_id.clone(), s.clock.now(), caller.actor());
    db(&s, move |c| {
        c.execute("UPDATE accounts SET controller_json = ?2 WHERE id = ?1", params![id, given.to_string()])?;
        crate::audit::record(c, now, Some(&id), &actor, "account.controller_set", &json!({"controller": given["controller"]["name"], "legal_basis": given["legal_basis"]}))?;
        Ok(())
    })
    .await?;
    Ok(axum::Json(current(&s, &caller).await?))
}

/// `PUT /v1/account/dpia`.
pub async fn put_dpia(State(s): State<AppState>, Extension(caller): Extension<Caller>, ApiJson(d): ApiJson<Dpia>) -> ApiResult<axum::Json<Value>> {
    caller.require(Scope::AccountAdmin)?;
    d.check().map_err(ApiError::validation)?;
    let (id, now, actor, v) = (caller.account_id.clone(), s.clock.now(), caller.actor(), d.to_value());
    db(&s, move |c| {
        c.execute("UPDATE accounts SET dpia_json = ?2 WHERE id = ?1", params![id, v.to_string()])?;
        crate::audit::record(c, now, Some(&id), &actor, "account.dpia_set", &v)?;
        Ok(())
    })
    .await?;
    Ok(axum::Json(current(&s, &caller).await?))
}

/// `PATCH /v1/account/settings`. A field left out is unchanged; a
/// `webhook_url` of null removes it.
#[derive(Deserialize, Debug)]
#[serde(deny_unknown_fields)]
pub struct Settings {
    pub held_pages: Option<String>,
    #[serde(default, with = "double_option")]
    pub webhook_url: Option<Option<String>>,
}

/// Tell a field left out (None) from one set to null (Some(None)).
mod double_option {
    use serde::{Deserialize, Deserializer};
    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Option<Option<String>>, D::Error> {
        Option::<String>::deserialize(d).map(Some)
    }
}

pub async fn patch_settings(State(s): State<AppState>, Extension(caller): Extension<Caller>, ApiJson(p): ApiJson<Settings>) -> ApiResult<axum::Json<Value>> {
    caller.require(Scope::AccountAdmin)?;
    if let Some(h) = &p.held_pages
        && h != "none"
        && h != "cloud"
    {
        return Err(ApiError::validation("held_pages must be \"none\" or \"cloud\""));
    }
    if let Some(Some(u)) = &p.webhook_url {
        crate::webhooks::check_url(u, s.cfg.webhook_allow_private).map_err(ApiError::validation)?;
    }
    let (id, now, actor) = (caller.account_id.clone(), s.clock.now(), caller.actor());
    let new_secret = db(&s, move |c| {
        let tx = c.transaction()?;
        if let Some(h) = &p.held_pages {
            tx.execute("UPDATE accounts SET held_pages = ?2 WHERE id = ?1", params![id, h])?;
        }
        let mut secret = None;
        if let Some(u) = &p.webhook_url {
            tx.execute("UPDATE accounts SET webhook_url = ?2 WHERE id = ?1", params![id, u])?;
            let has: Option<String> = tx.query_row("SELECT webhook_secret FROM accounts WHERE id = ?1", [&id], |r| r.get(0))?;
            if u.is_some() && has.is_none() {
                let s = crate::webhooks::new_secret();
                tx.execute("UPDATE accounts SET webhook_secret = ?2 WHERE id = ?1", params![id, s])?;
                secret = Some(s);
            }
        }
        crate::audit::record(
            &tx,
            now,
            Some(&id),
            &actor,
            "account.settings_set",
            &json!({"held_pages": p.held_pages, "webhook_url": p.webhook_url, "webhook_secret_created": secret.is_some()}),
        )?;
        tx.commit()?;
        Ok(secret)
    })
    .await?;
    let mut v = current(&s, &caller).await?;
    if let Some(secret) = new_secret {
        // shown once: only its holder can check deliveries
        v["webhook_secret"] = json!(secret);
    }
    Ok(axum::Json(v))
}

/// `POST /v1/account/webhook-secret`: a new signing secret, returned once.
pub async fn rotate_webhook_secret(State(s): State<AppState>, Extension(caller): Extension<Caller>) -> ApiResult<axum::Json<Value>> {
    caller.require(Scope::AccountAdmin)?;
    let (id, now, actor) = (caller.account_id.clone(), s.clock.now(), caller.actor());
    let secret = crate::webhooks::new_secret();
    let stored = secret.clone();
    db(&s, move |c| {
        c.execute("UPDATE accounts SET webhook_secret = ?2 WHERE id = ?1", params![id, stored])?;
        crate::audit::record(c, now, Some(&id), &actor, "account.webhook_secret_rotated", &json!({}))?;
        Ok(())
    })
    .await?;
    Ok(axum::Json(json!({"object": "webhook_secret", "secret": secret, "scheme": "Standard Webhooks (v1, HMAC-SHA256)"})))
}
