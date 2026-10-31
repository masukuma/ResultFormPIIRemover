//! Jobs: what a client asks for, how it is stored, and the resource the
//! API returns. A job is a long-running operation (`202 Accepted`, then
//! polled or announced by webhook): queued, running, then succeeded,
//! failed or canceled.

use anyhow::Result;
use rusqlite::{Connection, OptionalExtension, Row, params};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::billing::format_usd;
use crate::clock::rfc3339;
use crate::rfpii::PerPage;

/// Idempotency keys are kept this long.
pub const IDEMPOTENCY_SECS: i64 = 24 * 3600;

/// `POST /v1/jobs`.
#[derive(Deserialize, Debug)]
#[serde(deny_unknown_fields)]
pub struct JobRequest {
    pub file_ids: Vec<String>,
    #[serde(default)]
    pub instructions: Option<String>,
    #[serde(default)]
    pub policy: Option<PolicyRequest>,
    #[serde(default)]
    pub per_page: Option<PerPageArg>,
    #[serde(default)]
    pub webhook_url: Option<String>,
    #[serde(default)]
    pub metadata: Option<Value>,
}

/// The policy a client may set without instructions (HARNESS.md §3).
#[derive(Deserialize, Serialize, Debug, Clone, Default, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct PolicyRequest {
    /// hide every character of each ID or passport number (`ids: whole`)
    #[serde(default)]
    pub full_ids: bool,
    /// hide every digit of each phone number (`phones: whole`)
    #[serde(default)]
    pub full_phones: bool,
    /// never mask a stamp (`stamps: protected`)
    #[serde(default)]
    pub keep_stamps: bool,
}

/// `per_page`: "auto", "yes", "no", or true and false.
#[derive(Deserialize, Debug, Clone, Copy)]
#[serde(untagged)]
pub enum PerPageArg {
    Bool(bool),
    Name(PerPage),
}

impl PerPageArg {
    pub fn value(self) -> PerPage {
        match self {
            PerPageArg::Bool(true) => PerPage::Yes,
            PerPageArg::Bool(false) => PerPage::No,
            PerPageArg::Name(p) => p,
        }
    }
}

/// The job file's `policy`, when the client set one: the structured policy
/// `rfpii job` uses when no orchestrator compiles the instructions. Left
/// out otherwise, so instructions with nothing to compile them can't be
/// shown to be met, as `rfpii job` decides.
pub fn policy_value(p: Option<&PolicyRequest>, per_page: Option<PerPage>) -> Option<Value> {
    if p.is_none() && per_page.is_none() {
        return None;
    }
    let p = p.cloned().unwrap_or_default();
    Some(json!({
        "ids": if p.full_ids { "whole" } else { "three" },
        "phones": if p.full_phones { "whole" } else { "tail" },
        "stamps": if p.keep_stamps { "protected" } else { "bands" },
        "per_page": per_page.unwrap_or_default().name(),
    }))
}

/// A stored job.
#[derive(Clone, Debug)]
pub struct JobRow {
    pub id: String,
    pub account_id: String,
    pub status: String,
    pub request: Value,
    pub plan: String,
    pub unit_price_microusd: i64,
    pub period: String,
    pub forms_requested: i64,
    pub reserved_forms: i64,
    pub settled: bool,
    pub forms_released: Option<i64>,
    pub forms_held: Option<i64>,
    pub forms_charged: Option<i64>,
    pub amount_microusd: Option<i64>,
    pub cost: Option<Value>,
    pub inventory: Option<Value>,
    pub exit_code: Option<i64>,
    pub error: Option<String>,
    pub attempts: i64,
    pub max_attempts: i64,
    pub lease_owner: Option<String>,
    pub lease_expires_at: Option<i64>,
    pub cancel_requested: bool,
    pub webhook_url: Option<String>,
    pub metadata: Option<Value>,
    pub created_at: i64,
    pub started_at: Option<i64>,
    pub finished_at: Option<i64>,
    pub inputs_deleted_at: Option<i64>,
    pub outputs_expire_at: Option<i64>,
    pub outputs_deleted_at: Option<i64>,
    pub deleted_at: Option<i64>,
}

pub const COLUMNS: &str = "id, account_id, status, request_json, plan, unit_price_microusd, period, forms_requested, reserved_forms, settled, \
    forms_released, forms_held, forms_charged, amount_microusd, cost_json, inventory_json, exit_code, error, attempts, max_attempts, \
    lease_owner, cancel_requested, webhook_url, metadata_json, created_at, started_at, finished_at, inputs_deleted_at, outputs_expire_at, \
    outputs_deleted_at, deleted_at, lease_expires_at";

pub fn from_row(r: &Row) -> rusqlite::Result<JobRow> {
    let json = |s: Option<String>| s.and_then(|s| serde_json::from_str(&s).ok());
    Ok(JobRow {
        id: r.get(0)?,
        account_id: r.get(1)?,
        status: r.get(2)?,
        request: json(r.get(3)?).unwrap_or(Value::Null),
        plan: r.get(4)?,
        unit_price_microusd: r.get(5)?,
        period: r.get(6)?,
        forms_requested: r.get(7)?,
        reserved_forms: r.get(8)?,
        settled: r.get::<_, i64>(9)? != 0,
        forms_released: r.get(10)?,
        forms_held: r.get(11)?,
        forms_charged: r.get(12)?,
        amount_microusd: r.get(13)?,
        cost: json(r.get(14)?),
        inventory: json(r.get(15)?),
        exit_code: r.get(16)?,
        error: r.get(17)?,
        attempts: r.get(18)?,
        max_attempts: r.get(19)?,
        lease_owner: r.get(20)?,
        cancel_requested: r.get::<_, i64>(21)? != 0,
        webhook_url: r.get(22)?,
        metadata: json(r.get(23)?),
        created_at: r.get(24)?,
        started_at: r.get(25)?,
        finished_at: r.get(26)?,
        inputs_deleted_at: r.get(27)?,
        outputs_expire_at: r.get(28)?,
        outputs_deleted_at: r.get(29)?,
        deleted_at: r.get(30)?,
        lease_expires_at: r.get(31)?,
    })
}

impl JobRow {
    pub fn is_terminal(&self) -> bool {
        matches!(self.status.as_str(), "succeeded" | "failed" | "canceled")
    }

    /// Whether its outputs can still be read.
    pub fn outputs_available(&self) -> bool {
        self.deleted_at.is_none() && self.outputs_deleted_at.is_none()
    }

    /// The job as the API returns it.
    pub fn resource(&self) -> Value {
        let t = |x: Option<i64>| x.map(rfc3339);
        let path = format!("/v1/jobs/{}", self.id);
        let paid = self.plan == "paid";
        let price = json!({
            "currency": "usd",
            "plan": self.plan,
            "unit_microusd": if paid { self.unit_price_microusd } else { 0 },
            "estimate_microusd": if paid { self.forms_requested * self.unit_price_microusd } else { 0 },
            "forms_charged": self.forms_charged,
            "amount_microusd": self.amount_microusd,
            "amount_usd": self.amount_microusd.map(format_usd),
        });
        json!({
            "id": self.id,
            "object": "job",
            "status": self.status,
            "cancel_requested": self.cancel_requested && !self.is_terminal(),
            "file_ids": self.request["file_ids"],
            "instructions": self.request["instructions"],
            "policy": self.request["policy"],
            "per_page": self.request["per_page"],
            "held_pages": self.request["held_pages"],
            "webhook_url": self.webhook_url,
            "metadata": self.metadata,
            "forms_requested": self.forms_requested,
            "forms_released": self.forms_released,
            "forms_held": self.forms_held,
            "inventory": self.inventory,
            "price": price,
            "cost": self.cost,
            "exit_code": self.exit_code,
            "error": self.error,
            "attempts": self.attempts,
            "created_at": rfc3339(self.created_at),
            "started_at": t(self.started_at),
            "finished_at": t(self.finished_at),
            "inputs_deleted_at": t(self.inputs_deleted_at),
            "outputs_expire_at": t(self.outputs_expire_at),
            "outputs_deleted_at": t(self.outputs_deleted_at),
            "deleted_at": t(self.deleted_at),
            "links": {"self": path, "report": format!("{path}/report"), "outputs": format!("{path}/outputs"), "archive": format!("{path}/archive"), "cancel": format!("{path}/cancel")},
        })
    }
}

/// A job of `account`, or None.
pub fn load(conn: &Connection, account: &str, id: &str) -> Result<Option<JobRow>> {
    Ok(conn.query_row(&format!("SELECT {COLUMNS} FROM jobs WHERE id = ?1 AND account_id = ?2"), params![id, account], from_row).optional()?)
}

/// A job, whoever's it is (workers).
pub fn load_any(conn: &Connection, id: &str) -> Result<Option<JobRow>> {
    Ok(conn.query_row(&format!("SELECT {COLUMNS} FROM jobs WHERE id = ?1"), [id], from_row).optional()?)
}

/// A page of an account's jobs, newest first, after the job `after`; and
/// whether there are more.
pub fn list(conn: &Connection, account: &str, limit: i64, after: Option<&str>) -> Result<(Vec<JobRow>, bool)> {
    let mut st = conn.prepare(&format!("SELECT {COLUMNS} FROM jobs WHERE account_id = ?1 AND (?2 IS NULL OR id < ?2) ORDER BY id DESC LIMIT ?3"))?;
    let mut rows = st.query_map(params![account, after, limit + 1], from_row)?.collect::<rusqlite::Result<Vec<_>>>()?;
    let more = rows.len() as i64 > limit;
    rows.truncate(limit as usize);
    Ok((rows, more))
}

/// The job an idempotency key was used for, with the hash of its request,
/// if the key was used in the last 24 hours.
pub fn idempotent(conn: &Connection, account: &str, key: &str, now: i64) -> Result<Option<(String, String)>> {
    Ok(conn
        .query_row("SELECT request_hash, job_id FROM idempotency_keys WHERE account_id = ?1 AND key = ?2 AND created_at > ?3", params![account, key, now - IDEMPOTENCY_SECS], |r| {
            Ok((r.get(0)?, r.get(1)?))
        })
        .optional()?)
}

/// The SHA-256 of a request body as canonical JSON (keys sorted), so the
/// same request written differently is the same request.
pub fn request_hash(body: &Value) -> String {
    let mut s = String::new();
    canonical(body, &mut s);
    crate::ids::sha256_hex(s.as_bytes())
}

/// `v` as JSON with every object's keys sorted, whatever order the map
/// keeps them in.
fn canonical(v: &Value, out: &mut String) {
    match v {
        Value::Object(m) => {
            let mut keys: Vec<&String> = m.keys().collect();
            keys.sort();
            out.push('{');
            for (i, k) in keys.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                out.push_str(&Value::String((*k).clone()).to_string());
                out.push(':');
                canonical(&m[k.as_str()], out);
            }
            out.push('}');
        }
        Value::Array(a) => {
            out.push('[');
            for (i, x) in a.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                canonical(x, out);
            }
            out.push(']');
        }
        other => out.push_str(&other.to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_policy_goes_to_the_job_file_only_when_the_client_set_one() {
        assert_eq!(policy_value(None, None), None);
        let p = policy_value(Some(&PolicyRequest { full_ids: true, ..Default::default() }), None).unwrap();
        assert_eq!(p, json!({"ids": "whole", "phones": "tail", "stamps": "bands", "per_page": "auto"}));
        let p = policy_value(None, Some(PerPage::Yes)).unwrap();
        assert_eq!((p["ids"].as_str(), p["per_page"].as_str()), (Some("three"), Some("yes")));
        let r: JobRequest = serde_json::from_value(json!({"file_ids": ["f"], "per_page": true})).unwrap();
        assert_eq!(r.per_page.unwrap().value(), PerPage::Yes);
        assert!(serde_json::from_value::<JobRequest>(json!({"file_ids": [], "colour": "red"})).is_err());
        assert!(serde_json::from_value::<JobRequest>(json!({"file_ids": [], "policy": {"hide_names": true}})).is_err());
    }

    #[test]
    fn the_same_request_written_differently_hashes_the_same() {
        let a: Value = serde_json::from_str(r#"{"file_ids": ["a"], "instructions": "x"}"#).unwrap();
        let b: Value = serde_json::from_str(r#"{"instructions":"x","file_ids":["a"]}"#).unwrap();
        assert_eq!(request_hash(&a), request_hash(&b));
        assert_ne!(request_hash(&a), request_hash(&json!({"file_ids": ["b"]})));
    }
}
