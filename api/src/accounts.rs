//! Accounts: the customer, its plan, and the legal records `rfpii job`
//! needs before it runs anything (HARNESS.md §9 conditions 8 and 9): the
//! controller record and the DPIA. The customer is the controller; the
//! platform's operator is its processor (`legal/C-recommendations.md` §2.9).

use anyhow::Result;
use rusqlite::{Connection, OptionalExtension, Row, params};
use serde::Deserialize;
use serde_json::{Value, json};

use crate::config::Config;

#[derive(Clone, Debug)]
pub struct Account {
    pub id: String,
    pub name: String,
    pub email: Option<String>,
    pub plan: String,
    pub spend_cap_microusd: Option<i64>,
    pub controller: Option<Value>,
    pub dpia: Option<Value>,
    pub held_pages: String,
    pub webhook_url: Option<String>,
    pub webhook_secret: Option<String>,
    pub stripe_customer_id: Option<String>,
    pub stripe_subscription_id: Option<String>,
    pub created_at: i64,
}

const COLUMNS: &str =
    "id, name, email, plan, spend_cap_microusd, controller_json, dpia_json, held_pages, webhook_url, webhook_secret, stripe_customer_id, stripe_subscription_id, created_at";

fn from_row(r: &Row) -> rusqlite::Result<Account> {
    let json = |s: Option<String>| s.and_then(|s| serde_json::from_str(&s).ok());
    Ok(Account {
        id: r.get(0)?,
        name: r.get(1)?,
        email: r.get(2)?,
        plan: r.get(3)?,
        spend_cap_microusd: r.get(4)?,
        controller: json(r.get(5)?),
        dpia: json(r.get(6)?),
        held_pages: r.get(7)?,
        webhook_url: r.get(8)?,
        webhook_secret: r.get(9)?,
        stripe_customer_id: r.get(10)?,
        stripe_subscription_id: r.get(11)?,
        created_at: r.get(12)?,
    })
}

pub fn load(conn: &Connection, id: &str) -> Result<Option<Account>> {
    Ok(conn.query_row(&format!("SELECT {COLUMNS} FROM accounts WHERE id = ?1"), [id], from_row).optional()?)
}

pub fn by_stripe_customer(conn: &Connection, customer: &str) -> Result<Option<Account>> {
    Ok(conn.query_row(&format!("SELECT {COLUMNS} FROM accounts WHERE stripe_customer_id = ?1"), [customer], from_row).optional()?)
}

pub fn list(conn: &Connection) -> Result<Vec<Account>> {
    let mut st = conn.prepare(&format!("SELECT {COLUMNS} FROM accounts ORDER BY id"))?;
    let rows = st.query_map([], from_row)?.collect::<rusqlite::Result<Vec<_>>>()?;
    Ok(rows)
}

/// Whether `plan` is a plan.
pub fn is_plan(plan: &str) -> bool {
    plan == "free" || plan == "paid"
}

/// A new account, recorded in the audit log.
pub fn create(conn: &Connection, now: i64, name: &str, email: Option<&str>, plan: &str, actor: &str) -> Result<Account> {
    anyhow::ensure!(is_plan(plan), "plan must be free or paid");
    anyhow::ensure!(!name.trim().is_empty(), "an account needs a name");
    let id = crate::ids::new_id("acct");
    conn.execute("INSERT INTO accounts (id, name, email, plan, created_at) VALUES (?1, ?2, ?3, ?4, ?5)", params![id, name.trim(), email, plan, now])?;
    crate::audit::record(conn, now, Some(&id), actor, "account.created", &json!({"plan": plan}))?;
    load(conn, &id)?.ok_or_else(|| anyhow::anyhow!("account {id} vanished"))
}

/// Set an account's plan and, if given, its monthly spend cap (None in
/// the option: no cap). Whether the account exists.
pub fn set_plan(conn: &Connection, now: i64, id: &str, plan: &str, cap: Option<Option<i64>>, actor: &str) -> Result<bool> {
    anyhow::ensure!(is_plan(plan), "plan must be free or paid");
    let n = match cap {
        Some(c) => conn.execute("UPDATE accounts SET plan = ?2, spend_cap_microusd = ?3 WHERE id = ?1", params![id, plan, c])?,
        None => conn.execute("UPDATE accounts SET plan = ?2 WHERE id = ?1", params![id, plan])?,
    };
    if n > 0 {
        crate::audit::record(conn, now, Some(id), actor, "account.plan_set", &json!({"plan": plan, "spend_cap_microusd": cap}))?;
    }
    Ok(n > 0)
}

/// The fields a controller record can't go without, checked as
/// `Controller::check` in `rust/src/legal.rs` checks them.
pub fn check_controller(r: &Value) -> Result<(), String> {
    let text = |v: &Value| v.as_str().is_some_and(|s| !s.trim().is_empty());
    if !r.is_object() {
        return Err("the controller record must be a JSON object".into());
    }
    if !text(&r["controller"]["name"]) {
        return Err("controller.name is required".into());
    }
    for k in ["legal_basis", "purpose"] {
        if !text(&r[k]) {
            return Err(format!("{k} is required"));
        }
    }
    if !r["commercial"].is_boolean() {
        return Err("commercial must be true or false (Data Protection Act s.37)".into());
    }
    if !matches!(r["operator_role"].as_str(), Some("controller") | Some("processor")) {
        return Err("operator_role must be \"controller\" or \"processor\"".into());
    }
    if r["commercial"] == true && !text(&r["statutory_authority"]) {
        return Err("a commercial use needs statutory_authority (Data Protection Act s.37)".into());
    }
    if !r["processors"].is_null() && !r["processors"].is_array() {
        return Err("processors must be a list".into());
    }
    for (i, p) in r["processors"].as_array().into_iter().flatten().enumerate() {
        for k in ["name", "country", "service"] {
            if !text(&p[k]) {
                return Err(format!("processors[{i}].{k} is required"));
            }
        }
    }
    Ok(())
}

/// Where the model host is: on the machine, in Kenya, or abroad (as
/// `host_region` in `rust/src/legal.rs`).
pub fn model_host_region(cfg: &Config) -> &'static str {
    let host = cfg.ollama_host.clone().unwrap_or_else(|| "https://ollama.com".into());
    let name = host_name(&host);
    if ["localhost", "127.0.0.1", "::1", "0.0.0.0"].contains(&name.as_str()) {
        "local"
    } else if cfg.kenya_hosts.iter().any(|k| host_name(k) == name) {
        "kenya"
    } else {
        "foreign"
    }
}

/// The host name of a URL or `host:port`.
fn host_name(s: &str) -> String {
    let rest = s.split_once("://").map(|(_, r)| r).unwrap_or(s);
    let authority = rest.split('/').next().unwrap_or("");
    let host = if let Some(v6) = authority.strip_prefix('[') { v6.split(']').next().unwrap_or("") } else { authority.split(':').next().unwrap_or("") };
    host.to_ascii_lowercase()
}

/// The controller record a job runs under: the customer's, with the
/// operator as its processor (`operator_role: processor`), the platform
/// and, when jobs send pages to a model host outside the machine, that
/// host listed among the processors, and the DPIA's reference.
pub fn effective_controller(given: &Value, held_pages: &str, dpia: Option<&Value>, cfg: &Config) -> Value {
    let mut r = given.clone();
    r["operator_role"] = json!("processor");
    let mut processors: Vec<Value> = r["processors"].as_array().cloned().unwrap_or_default();
    let mut add = |p: Value| {
        if !processors.iter().any(|q| q["name"] == p["name"]) {
            processors.push(p);
        }
    };
    add(json!({"name": cfg.platform_name, "country": cfg.platform_country, "service": "redaction of election result forms with rfpii, as the controller's processor"}));
    if cfg.has_models() && model_host_region(cfg) != "local" {
        let what = if held_pages == "cloud" {
            "model hosting: crops of redacted pages, and held pages as the job allows (the controller opted in to sending them)"
        } else {
            "model hosting: crops of redacted pages only"
        };
        let country = if model_host_region(cfg) == "kenya" { "KE".to_string() } else { cfg.model_host_country.clone() };
        add(json!({"name": cfg.model_host_name, "country": country, "service": what}));
    }
    r["processors"] = json!(processors);
    if let Some(reference) = dpia.and_then(|d| d["reference"].as_str()) {
        r["dpia"] = json!({"reference": reference});
    }
    r
}

/// The DPIA a job's processing rests on (Data Protection Act s.31).
#[derive(Deserialize, Debug)]
#[serde(deny_unknown_fields)]
pub struct Dpia {
    pub reference: String,
    pub submitted_on: Option<String>,
    pub status: Option<String>,
}

/// Days a DPIA must have been with the Commissioner before processing,
/// unless approved sooner (`rust/src/job/spec.rs`).
pub const DPIA_DAYS: i64 = 60;

impl Dpia {
    /// Check the fields: a reference, and a date as YYYY-MM-DD if given.
    pub fn check(&self) -> Result<(), String> {
        if self.reference.trim().is_empty() {
            return Err("reference is required".into());
        }
        if let Some(d) = &self.submitted_on {
            crate::clock::parse_date(d).ok_or_else(|| format!("submitted_on {d:?} is not a date as YYYY-MM-DD"))?;
        }
        Ok(())
    }

    pub fn to_value(&self) -> Value {
        json!({"reference": self.reference.trim(), "submitted_on": self.submitted_on, "status": self.status})
    }
}

/// Why the DPIA doesn't let processing start at `now`, or None if it does:
/// approved (or deemed approved), or submitted at least 60 days before, as
/// the release gate checks it (HARNESS.md §9, condition 8).
pub fn dpia_bars(d: &Value, now: i64) -> Option<String> {
    let reference = d["reference"].as_str().unwrap_or("");
    let status = d["status"].as_str().unwrap_or("").trim().to_lowercase();
    if reference.trim().is_empty() {
        return Some("the DPIA has no reference".into());
    }
    if status == "approved" || status == "deemed approved" {
        return None;
    }
    let Some(on) = d["submitted_on"].as_str() else {
        return Some(format!("DPIA {reference} is {} and has no submission date", if status.is_empty() { "not approved" } else { &status }));
    };
    let Some(at) = crate::clock::parse_date(on) else {
        return Some(format!("DPIA {reference}: submission date {on:?} is not YYYY-MM-DD"));
    };
    let days = (now - at).div_euclid(86_400);
    (days < DPIA_DAYS).then(|| format!("DPIA {reference} was submitted on {on}, {days} day(s) ago; processing may start {DPIA_DAYS} days after submission unless it is approved"))
}

/// What the account still needs before a job may run.
pub fn missing_for_jobs(a: &Account) -> Vec<&'static str> {
    let mut m = Vec::new();
    if a.controller.is_none() {
        m.push("controller");
    }
    if a.dpia.is_none() {
        m.push("dpia");
    }
    m
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    #[derive(Parser)]
    struct C {
        #[command(flatten)]
        c: Config,
    }

    fn cfg() -> Config {
        C::try_parse_from(["x"]).unwrap().c
    }

    fn valid() -> Value {
        json!({"controller": {"name": "Example Observer Group", "registration_no": "ODPC/123"},
               "operator_role": "controller", "legal_basis": "s.30(1)(b)(vii) legitimate interests",
               "purpose": "republishing polling-station results forms", "commercial": false})
    }

    #[test]
    fn a_controller_record_is_checked_as_rfpii_checks_it() {
        assert!(check_controller(&valid()).is_ok());
        for k in ["legal_basis", "purpose", "operator_role", "commercial"] {
            let mut v = valid();
            v.as_object_mut().unwrap().remove(k);
            assert!(check_controller(&v).is_err(), "{k} missing");
        }
        let mut v = valid();
        v["commercial"] = json!(true);
        assert!(check_controller(&v).unwrap_err().contains("s.37"));
        v["statutory_authority"] = json!("Elections Act s.39(1C)(c)");
        assert!(check_controller(&v).is_ok());
        v["processors"] = json!([{"name": "X", "country": "KE"}]);
        assert!(check_controller(&v).unwrap_err().contains("processors[0].service"));
        assert!(check_controller(&json!([1])).is_err());
    }

    #[test]
    fn the_operator_is_the_processor_and_the_model_host_is_listed() {
        let mut c = cfg();
        let r = effective_controller(&valid(), "none", Some(&json!({"reference": "ODPC/DPIA/1"})), &c);
        assert_eq!(r["operator_role"], "processor");
        assert_eq!(r["dpia"]["reference"], "ODPC/DPIA/1");
        let names: Vec<&str> = r["processors"].as_array().unwrap().iter().map(|p| p["name"].as_str().unwrap()).collect();
        assert_eq!(names, ["rfpii API", "Ollama"]);
        assert!(r["processors"][1]["service"].as_str().unwrap().contains("redacted pages only"));
        assert!(check_controller(&r).is_ok());
        let cloud = effective_controller(&valid(), "cloud", None, &c);
        assert!(cloud["processors"][1]["service"].as_str().unwrap().contains("held pages"));
        // a model host on this machine is no processor; no models, none either
        c.ollama_host = Some("http://localhost:11434".into());
        assert_eq!(effective_controller(&valid(), "cloud", None, &c)["processors"].as_array().unwrap().len(), 1);
        c.ollama_host = None;
        c.vision_models.clear();
        c.orchestrator_models.clear();
        assert_eq!(effective_controller(&valid(), "cloud", None, &c)["processors"].as_array().unwrap().len(), 1);
    }

    #[test]
    fn a_dpia_lets_processing_start_once_approved_or_60_days_after_submission() {
        let now = crate::clock::parse_date("2026-10-31").unwrap() + 9 * 3600;
        let d = |on: Option<&str>, status: Option<&str>| json!({"reference": "ODPC/1", "submitted_on": on, "status": status});
        assert!(dpia_bars(&d(Some("2026-09-01"), Some("submitted")), now).is_none());
        assert!(dpia_bars(&d(Some("2026-09-02"), None), now).unwrap().contains("59 day(s) ago"));
        assert!(dpia_bars(&d(None, Some("Deemed approved")), now).is_none());
        assert!(dpia_bars(&d(None, Some("draft")), now).unwrap().contains("no submission date"));
        assert!(Dpia { reference: "x".into(), submitted_on: Some("2026-13-01".into()), status: None }.check().is_err());
        assert!(Dpia { reference: " ".into(), submitted_on: None, status: None }.check().is_err());
    }
}
