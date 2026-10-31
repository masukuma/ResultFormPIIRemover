//! Webhooks, by the Standard Webhooks scheme (standardwebhooks.com): each
//! delivery carries `webhook-id`, `webhook-timestamp` and
//! `webhook-signature: v1,<base64 HMAC-SHA256 of "id.timestamp.body">`,
//! keyed with the base64-decoded secret after its `whsec_` prefix. A
//! delivery keeps its id across retries, which back off (5 s, 30 s, 2 min,
//! 10 min, 1 h, 6 h by default); every attempt is recorded.

use anyhow::{Context, Result};
use base64::Engine;
use base64::engine::general_purpose::STANDARD as B64;
use hmac::{KeyInit, Mac};
use rusqlite::{Connection, params};
use serde_json::{Value, json};
use std::net::IpAddr;
use std::time::Duration;

use crate::server::AppState;

type HmacSha256 = hmac::Hmac<sha2::Sha256>;

/// The events a job sends.
pub const EVENTS: [&str; 3] = ["job.succeeded", "job.failed", "job.canceled"];

/// A new signing secret: `whsec_` and 24 random bytes in base64.
pub fn new_secret() -> String {
    format!("whsec_{}", B64.encode(crate::ids::random_bytes::<24>()))
}

fn key_of(secret: &str) -> Result<Vec<u8>> {
    B64.decode(secret.strip_prefix("whsec_").unwrap_or(secret)).context("the webhook secret is not base64")
}

/// The `webhook-signature` value for a message.
pub fn sign(secret: &str, id: &str, timestamp: i64, body: &[u8]) -> Result<String> {
    let mut mac = <HmacSha256 as KeyInit>::new_from_slice(&key_of(secret)?).context("HMAC key")?;
    mac.update(format!("{id}.{timestamp}.").as_bytes());
    mac.update(body);
    Ok(format!("v1,{}", B64.encode(mac.finalize().into_bytes())))
}

/// Whether `header` (space-separated signatures) holds a valid v1
/// signature of the message; compared in constant time. Receivers do
/// this, and check the timestamp is recent.
pub fn verify(secret: &str, id: &str, timestamp: i64, body: &[u8], header: &str) -> bool {
    let Ok(expected) = sign(secret, id, timestamp, body) else { return false };
    header.split(' ').any(|s| crate::ids::ct_eq(s.as_bytes(), expected.as_bytes()))
}

/// Whether `url` may receive webhooks: https (or http, and private
/// addresses, only when the operator allows them), and not written with a
/// private, loopback or link-local address.
pub fn check_url(url: &str, allow_private: bool) -> Result<reqwest::Url, String> {
    let u = reqwest::Url::parse(url).map_err(|e| format!("webhook_url is not a URL: {e}"))?;
    match u.scheme() {
        "https" => {}
        "http" if allow_private => {}
        s => return Err(format!("webhook_url must use https, not {s}")),
    }
    let Some(host) = u.host_str() else { return Err("webhook_url has no host".into()) };
    if !u.username().is_empty() || u.password().is_some() {
        return Err("webhook_url must not hold credentials".into());
    }
    if !allow_private {
        let bare = host.trim_start_matches('[').trim_end_matches(']');
        if bare.eq_ignore_ascii_case("localhost") || bare.parse::<IpAddr>().is_ok_and(|ip| !is_public(ip)) {
            return Err("webhook_url must be a public address".into());
        }
    }
    if url.len() > 2048 {
        return Err("webhook_url is longer than 2048 characters".into());
    }
    Ok(u)
}

/// Whether an address is on the public internet.
pub fn is_public(ip: IpAddr) -> bool {
    match ip {
        IpAddr::V4(v) => {
            !(v.is_private()
                || v.is_loopback()
                || v.is_link_local()
                || v.is_unspecified()
                || v.is_broadcast()
                || v.is_documentation()
                || v.octets()[0] == 100 && (v.octets()[1] & 0xC0) == 64)
        }
        IpAddr::V6(v) => {
            let s = v.segments();
            !(v.is_loopback() || v.is_unspecified() || (s[0] & 0xFE00) == 0xFC00 || (s[0] & 0xFFC0) == 0xFE80 || v.to_ipv4_mapped().is_some_and(|m| !is_public(IpAddr::V4(m))))
        }
    }
}

/// Queue an event for delivery to `url`.
pub fn enqueue(conn: &Connection, now: i64, account: &str, job: Option<&str>, event: &str, url: &str, data: &Value) -> Result<String> {
    let id = crate::ids::new_id("msg");
    let payload = json!({"type": event, "timestamp": crate::clock::rfc3339(now), "data": data});
    conn.execute(
        "INSERT INTO webhook_deliveries (id, account_id, job_id, event, url, payload, next_attempt_at, created_at) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?7)",
        params![id, account, job, event, url, payload.to_string(), now],
    )?;
    Ok(id)
}

struct Due {
    id: String,
    account: String,
    url: String,
    payload: String,
    attempts: i64,
}

/// Claim deliveries that are due, so no other process sends them too.
fn claim(conn: &Connection, now: i64) -> Result<Vec<Due>> {
    let mut st = conn.prepare(
        "UPDATE webhook_deliveries SET claimed_until = ?1 + 120
         WHERE id IN (SELECT id FROM webhook_deliveries WHERE status = 'pending' AND next_attempt_at <= ?1 AND (claimed_until IS NULL OR claimed_until < ?1) ORDER BY next_attempt_at LIMIT 20)
         RETURNING id, account_id, url, payload, attempts",
    )?;
    let rows =
        st.query_map([now], |r| Ok(Due { id: r.get(0)?, account: r.get(1)?, url: r.get(2)?, payload: r.get(3)?, attempts: r.get(4)? }))?.collect::<rusqlite::Result<Vec<_>>>()?;
    Ok(rows)
}

/// Send the deliveries that are due; how many were attempted.
pub async fn deliver_due(state: &AppState) -> Result<usize> {
    let now = state.clock.now();
    let due = state.db.call(move |c| claim(c, now)).await?;
    let n = due.len();
    for d in due {
        let outcome = attempt(state, &d).await;
        record(state, d, outcome).await?;
    }
    Ok(n)
}

/// One attempt: Ok(status) for a 2xx answer, Err((status, why)) otherwise.
async fn attempt(state: &AppState, d: &Due) -> Result<u16, (Option<u16>, String)> {
    let account = d.account.clone();
    let secret = state
        .db
        .call(move |c| Ok(crate::accounts::load(c, &account)?.and_then(|a| a.webhook_secret)))
        .await
        .map_err(|e| (None, format!("{e:#}")))?
        .ok_or((None, "the account has no webhook secret".to_string()))?;
    let url = check_url(&d.url, state.cfg.webhook_allow_private).map_err(|e| (None, e))?;
    resolve_public(&url, state.cfg.webhook_allow_private).await.map_err(|e| (None, e))?;
    let ts = state.clock.now();
    let sig = sign(&secret, &d.id, ts, d.payload.as_bytes()).map_err(|e| (None, format!("{e:#}")))?;
    let res = state
        .http
        .post(url)
        .timeout(Duration::from_secs(state.cfg.webhook_timeout_secs))
        .header("content-type", "application/json")
        .header("webhook-id", &d.id)
        .header("webhook-timestamp", ts.to_string())
        .header("webhook-signature", sig)
        .body(d.payload.clone())
        .send()
        .await
        .map_err(|e| (None, format!("{e}")))?;
    let status = res.status().as_u16();
    if res.status().is_success() { Ok(status) } else { Err((Some(status), format!("the receiver answered {status}"))) }
}

/// Check every address the host resolves to is public (unless allowed).
async fn resolve_public(url: &reqwest::Url, allow_private: bool) -> Result<(), String> {
    if allow_private {
        return Ok(());
    }
    let host = url.host_str().unwrap_or("");
    let port = url.port_or_known_default().unwrap_or(443);
    let addrs = tokio::net::lookup_host((host, port)).await.map_err(|e| format!("can't resolve {host}: {e}"))?;
    for a in addrs {
        if !is_public(a.ip()) {
            return Err(format!("{host} resolves to a private address"));
        }
    }
    Ok(())
}

/// Record an attempt: delivered, or retried later, or failed for good.
async fn record(state: &AppState, d: Due, outcome: Result<u16, (Option<u16>, String)>) -> Result<()> {
    let now = state.clock.now();
    let schedule = state.cfg.webhook_retry_schedule.clone();
    state
        .db
        .call(move |c| {
            let attempts = d.attempts + 1;
            match outcome {
                Ok(status) => {
                    c.execute(
                        "UPDATE webhook_deliveries SET status = 'delivered', attempts = ?2, last_status = ?3, last_error = NULL, delivered_at = ?4, claimed_until = NULL WHERE id = ?1",
                        params![d.id, attempts, status, now],
                    )?;
                }
                Err((status, why)) => {
                    let next = schedule.get((attempts - 1) as usize).map(|s| now + *s as i64);
                    c.execute(
                        "UPDATE webhook_deliveries SET status = ?2, attempts = ?3, last_status = ?4, last_error = ?5, next_attempt_at = COALESCE(?6, next_attempt_at), claimed_until = NULL WHERE id = ?1",
                        params![d.id, if next.is_some() { "pending" } else { "failed" }, attempts, status, why, next],
                    )?;
                }
            }
            Ok(())
        })
        .await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn signatures_follow_the_standard_webhooks_example() {
        // the example of the Standard Webhooks specification
        let secret = "whsec_MfKQ9r8GKYqrTwjUPD8ILPZIo2LaLaSw";
        let body = br#"{"test": 2432232314}"#;
        let sig = sign(secret, "msg_p5jXN8AQM9LWM0D4loKWxJek", 1614265330, body).unwrap();
        assert_eq!(sig, "v1,g0hM9SsE+OTPJTGt/tmIKtSyZlE3uFJELVlNIOLJ1OE=");
        assert!(verify(secret, "msg_p5jXN8AQM9LWM0D4loKWxJek", 1614265330, body, &format!("v1,bm90IGl0 {sig}")));
        assert!(!verify(secret, "msg_p5jXN8AQM9LWM0D4loKWxJek", 1614265331, body, &sig));
        assert!(new_secret().starts_with("whsec_") && key_of(&new_secret()).unwrap().len() == 24);
    }

    #[test]
    fn webhook_urls_must_be_public_https() {
        assert!(check_url("https://hooks.example.org/rfpii", false).is_ok());
        assert!(check_url("http://hooks.example.org/rfpii", false).is_err());
        assert!(check_url("https://127.0.0.1/x", false).is_err());
        assert!(check_url("https://[::1]/x", false).is_err());
        assert!(check_url("https://10.1.2.3/x", false).is_err());
        assert!(check_url("https://localhost/x", false).is_err());
        assert!(check_url("https://user:pw@hooks.example.org/x", false).is_err());
        assert!(check_url("http://127.0.0.1:9/x", true).is_ok());
        assert!(!is_public("100.64.1.1".parse().unwrap()) && is_public("8.8.8.8".parse().unwrap()));
        assert!(!is_public("::ffff:192.168.1.1".parse().unwrap()) && !is_public("fd00::1".parse().unwrap()));
    }
}
