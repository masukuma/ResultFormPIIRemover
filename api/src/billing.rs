//! The free allowance, form credits, admission, settlement, usage and the
//! actual cost (PRICING.md). A form is a document as `rfpii inventory`
//! counts it; only forms a job releases are counted. Every account has
//! `FREE_FORMS_PER_MONTH` free forms a calendar month (UTC); beyond them,
//! forms are drawn from its form credits, bought in packages. A job
//! reserves its forms when it is created (free forms first, then credits)
//! and settles when it ends: the forms it released are counted (free forms
//! first, then credits) and the rest of the reservation is let go. Credits
//! are kept in a ledger of signed entries, whose sum is the balance; each
//! purchase or positive adjustment is a lot that settled jobs, expiries
//! and negative adjustments take from, soonest-expiring first. Admission,
//! settlement and every change to credits happen in IMMEDIATE
//! transactions. Purchases are in integer cents; the operator's cost is in
//! micro-dollars.

use anyhow::Result;
use axum::http::StatusCode;
use rusqlite::{Connection, OptionalExtension, Transaction, params};
use serde_json::{Value, json};
use std::collections::BTreeMap;

use crate::config::{Config, TokenPrice};
use crate::problem::ApiError;
use crate::quote::Rates;

/// The prices and rates in force.
#[derive(Clone, Debug)]
pub struct Prices {
    pub free_forms_per_month: i64,
    pub rates: Rates,
    pub max_packages_per_purchase: i64,
    /// Days credits last once added; 0: forever.
    pub credit_expiry_days: i64,
    pub tokens: BTreeMap<String, TokenPrice>,
    pub vcpu_hour_microusd: i64,
}

impl Prices {
    pub fn from_config(cfg: &Config) -> Result<Prices> {
        Ok(Prices {
            free_forms_per_month: cfg.free_forms_per_month,
            rates: cfg.rates(),
            max_packages_per_purchase: cfg.max_packages_per_purchase,
            credit_expiry_days: cfg.credit_expiry_days,
            tokens: cfg.token_prices()?,
            vcpu_hour_microusd: cfg.server_price_per_vcpu_hour_microusd,
        })
    }

    /// When credits added at `now` expire, if they do.
    pub fn credit_expiry(&self, now: i64) -> Option<i64> {
        (self.credit_expiry_days > 0).then(|| now + self.credit_expiry_days * 86_400)
    }
}

/// An account's free forms in a month.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Free {
    pub per_month: i64,
    /// counted by settled jobs of the month
    pub used: i64,
    /// held by jobs of the month not yet settled
    pub reserved: i64,
    pub left: i64,
}

impl Free {
    pub fn to_value(&self, period: &str) -> Value {
        json!({"period": period, "per_month": self.per_month, "used": self.used, "reserved": self.reserved, "left": self.left})
    }
}

/// Free forms of `account` in `period`.
pub fn free(conn: &Connection, account: &str, period: &str, prices: &Prices) -> Result<Free> {
    let used: i64 = conn.query_row("SELECT COALESCE(SUM(free_forms), 0) FROM usage_records WHERE account_id = ?1 AND period = ?2", params![account, period], |r| r.get(0))?;
    let reserved: i64 =
        conn.query_row("SELECT COALESCE(SUM(reserved_free), 0) FROM jobs WHERE account_id = ?1 AND period = ?2 AND settled = 0", params![account, period], |r| r.get(0))?;
    Ok(Free { per_month: prices.free_forms_per_month, used, reserved, left: (prices.free_forms_per_month - used - reserved).max(0) })
}

/// An account's form credits.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Credits {
    /// added by purchases (Stripe or recorded by the operator)
    pub bought: i64,
    /// counted by settled jobs
    pub used: i64,
    /// expired, recorded or due
    pub expired: i64,
    /// the operator's adjustments, signed
    pub adjusted: i64,
    /// held by jobs not yet settled
    pub reserved: i64,
    /// what a new job may reserve
    pub available: i64,
}

impl Credits {
    pub fn to_value(&self, prices: &Prices) -> Value {
        json!({"bought": self.bought, "used": self.used, "expired": self.expired, "adjusted": self.adjusted, "reserved": self.reserved,
               "available": self.available, "expire_after_days": (prices.credit_expiry_days > 0).then_some(prices.credit_expiry_days)})
    }
}

/// A lot of credits: a purchase or positive adjustment, with what is left
/// of it.
#[derive(Clone, Debug)]
struct Lot {
    id: i64,
    remaining: i64,
    expires_at: Option<i64>,
}

/// The account's lots, soonest-expiring first, then oldest first.
fn lots(conn: &Connection, account: &str) -> Result<Vec<Lot>> {
    let mut st = conn.prepare_cached(
        "SELECT l.id, l.forms + COALESCE((SELECT SUM(d.forms) FROM credit_ledger d WHERE d.lot_id = l.id), 0), l.expires_at
         FROM credit_ledger l WHERE l.account_id = ?1 AND l.lot_id IS NULL ORDER BY l.expires_at IS NULL, l.expires_at, l.id",
    )?;
    let rows = st.query_map([account], |r| Ok(Lot { id: r.get(0)?, remaining: r.get(1)?, expires_at: r.get(2)? }))?.collect::<rusqlite::Result<Vec<_>>>()?;
    Ok(rows)
}

/// Credits held by the account's jobs not yet settled.
fn reserved_credits(conn: &Connection, account: &str) -> Result<i64> {
    Ok(conn.query_row("SELECT COALESCE(SUM(reserved_credits), 0) FROM jobs WHERE account_id = ?1 AND settled = 0", [account], |r| r.get(0))?)
}

/// What of lots expired by `now` may go: their remaining forms less what
/// jobs hold (jobs hold the soonest-expiring credits first, so a job keeps
/// what it reserved even if its lot expires while it runs).
fn expirable(lots: &[Lot], reserved: i64, now: i64) -> i64 {
    let expired: i64 = lots.iter().filter(|l| l.remaining > 0 && l.expires_at.is_some_and(|t| t <= now)).map(|l| l.remaining).sum();
    (expired - reserved).max(0)
}

/// The account's credits at `now`.
pub fn credits(conn: &Connection, account: &str, now: i64) -> Result<Credits> {
    let mut c = Credits::default();
    let mut balance = 0;
    let mut st = conn.prepare_cached("SELECT kind, COALESCE(SUM(forms), 0) FROM credit_ledger WHERE account_id = ?1 GROUP BY kind")?;
    for row in st.query_map([account], |r| Ok((r.get::<_, String>(0)?, r.get::<_, i64>(1)?)))? {
        let (kind, sum) = row?;
        balance += sum;
        match kind.as_str() {
            "purchase" => c.bought = sum,
            "job" => c.used = -sum,
            "expiry" => c.expired = -sum,
            _ => c.adjusted = sum,
        }
    }
    c.reserved = reserved_credits(conn, account)?;
    let due = expirable(&lots(conn, account)?, c.reserved, now);
    c.expired += due;
    c.available = (balance - due - c.reserved).max(0);
    Ok(c)
}

/// Why a ledger entry was made: its kind ("purchase", "job", "expiry" or
/// "adjustment"), the purchase or job it is for, who made it, a note.
#[derive(Clone, Copy, Debug)]
pub struct Cause<'a> {
    pub kind: &'a str,
    pub purchase: Option<&'a str>,
    pub job: Option<&'a str>,
    pub actor: &'a str,
    pub note: Option<&'a str>,
}

impl<'a> Cause<'a> {
    /// An entry of `kind` by `actor`, for nothing in particular.
    pub fn of(kind: &'a str, actor: &'a str) -> Cause<'a> {
        Cause { kind, purchase: None, job: None, actor, note: None }
    }
}

/// Insert a ledger entry: a lot (`forms > 0`, `lot` None) or what is taken
/// from one.
fn insert(tx: &Transaction, account: &str, now: i64, forms: i64, cause: Cause, lot: Option<i64>, expires_at: Option<i64>) -> Result<i64> {
    tx.execute(
        "INSERT INTO credit_ledger (account_id, at, forms, kind, purchase_id, job_id, lot_id, expires_at, actor, note) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)",
        params![account, now, forms, cause.kind, cause.purchase, cause.job, lot, expires_at, cause.actor, cause.note],
    )?;
    Ok(tx.last_insert_rowid())
}

/// Add `forms` credits to the account as a lot (a purchase, or a positive
/// adjustment), expiring as CREDIT_EXPIRY_DAYS says. The entry's id.
pub fn add_lot(tx: &Transaction, account: &str, forms: i64, cause: Cause, now: i64, prices: &Prices) -> Result<i64> {
    anyhow::ensure!(forms > 0, "a lot adds forms");
    insert(tx, account, now, forms, cause, None, prices.credit_expiry(now))
}

/// Take `forms` credits from the account's lots, soonest-expiring first,
/// one entry per lot taken from. What was taken (all of it, unless the
/// lots hold less).
fn draw(tx: &Transaction, account: &str, forms: i64, cause: Cause, now: i64) -> Result<i64> {
    let mut left = forms;
    for lot in lots(tx, account)? {
        if left == 0 {
            break;
        }
        let take = left.min(lot.remaining);
        if take > 0 {
            insert(tx, account, now, -take, cause, Some(lot.id), None)?;
            left -= take;
        }
    }
    if left > 0 {
        tracing::error!(account, wanted = forms, missing = left, "the account's credits fell short");
    }
    Ok(forms - left)
}

/// Record the expiry of credits whose lots expired by `now`, but not of
/// what jobs hold. Forms expired now.
pub fn expire_due(tx: &Transaction, account: &str, now: i64) -> Result<i64> {
    let lots = lots(tx, account)?;
    let mut left = expirable(&lots, reserved_credits(tx, account)?, now);
    let total = left;
    // the latest-expiring first, so what jobs hold stays in the soonest
    for lot in lots.iter().rev().filter(|l| l.remaining > 0 && l.expires_at.is_some_and(|t| t <= now)) {
        let take = left.min(lot.remaining);
        if take > 0 {
            // recorded when first seen; the note says when the lot expired
            let note = format!("the lot expired at {}", crate::clock::rfc3339(lot.expires_at.unwrap_or(now)));
            insert(tx, account, now, -take, Cause { note: Some(&note), ..Cause::of("expiry", "system") }, Some(lot.id), None)?;
            left -= take;
        }
    }
    Ok(total)
}

/// Adjust the account's credits by `forms` (signed), with a reason, for
/// the operator: a goodwill grant, or credits taken back after a refund.
/// A negative adjustment can take only credits that are available.
pub fn adjust(tx: &Transaction, account: &str, forms: i64, reason: &str, actor: &str, now: i64, prices: &Prices) -> Result<Result<(), String>> {
    if forms == 0 || reason.trim().is_empty() {
        return Ok(Err("an adjustment needs a number of forms other than 0, and a reason".into()));
    }
    expire_due(tx, account, now)?;
    let cause = Cause { note: Some(reason.trim()), ..Cause::of("adjustment", actor) };
    if forms > 0 {
        add_lot(tx, account, forms, cause, now, prices)?;
    } else {
        let available = credits(tx, account, now)?.available;
        if -forms > available {
            return Ok(Err(format!("the account has {available} credit(s) available; an adjustment can't take {}", -forms)));
        }
        draw(tx, account, -forms, cause, now)?;
    }
    crate::audit::record(tx, now, Some(account), actor, "credits.adjusted", &json!({"forms": forms, "reason": reason.trim()}))?;
    Ok(Ok(()))
}

/// What a job reserves: free forms, then credits.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Reservation {
    pub free: i64,
    pub credits: i64,
}

/// Whether `account` may start a job of `forms` forms in `period`: within
/// the free forms its month has left and the credits it has available,
/// free forms first. 402 Payment Required otherwise, saying how to buy.
pub fn admit(tx: &Transaction, account: &str, forms: i64, period: &str, now: i64, prices: &Prices, cfg: &Config) -> Result<Reservation, ApiError> {
    expire_due(tx, account, now)?;
    let free = free(tx, account, period, prices)?;
    let credits = credits(tx, account, now)?;
    if forms <= free.left + credits.available {
        let f = forms.min(free.left);
        return Ok(Reservation { free: f, credits: forms - f });
    }
    let short = forms - free.left - credits.available;
    let r = &prices.rates;
    let packages = (short + r.package_forms - 1) / r.package_forms;
    let buy = json!({
        "packages_needed": packages,
        "package_forms": r.package_forms,
        "package_price_cents": r.package_price_cents,
        "quote": format!("GET /v1/billing/quote?packages={packages}"),
        "checkout": "POST /v1/billing/checkout",
        "checkout_available": crate::stripe::checkout_configured(cfg),
        "pricing": cfg.pricing_url,
    });
    Err(ApiError::new(
        StatusCode::PAYMENT_REQUIRED,
        Some("insufficient-credits"),
        "Not enough free forms or credits",
        format!(
            "this job has {forms} form(s); the account has {} of its {} free forms left in {period} and {} credit(s) available. \
             Buy {packages} package(s) of {} forms (${} each before the card fee and VAT): GET /v1/billing/quote?packages={packages} gives the price, \
             POST /v1/billing/checkout takes the payment; or ask the operator",
            free.left,
            free.per_month,
            credits.available,
            r.package_forms,
            crate::quote::format_cents(r.package_price_cents),
        ),
    )
    .with("forms_requested", forms)
    .with("free_forms_left", free.left)
    .with("free_forms_per_month", free.per_month)
    .with("credits_available", credits.available)
    .with("period", period)
    .with("buy", buy))
}

/// How a job settled.
#[derive(Clone, Debug, PartialEq)]
pub struct Settlement {
    pub forms_counted: i64,
    pub free_forms: i64,
    pub credit_forms: i64,
    pub usage_record: Option<String>,
}

/// Settle a job: of the forms it released (never more than it reserved),
/// as many as its month has free forms left (its own reservation
/// included) are counted as free, the rest are taken from its credits (at
/// most what it reserved); the rest of its reservation is let go.
/// Settling twice changes nothing.
pub fn settle(tx: &Transaction, job_id: &str, released: i64, now: i64, prices: &Prices) -> Result<Settlement> {
    type Row = (String, String, i64, i64, i64, Option<i64>, Option<i64>, Option<i64>);
    let (account, period, requested, reserved_credits, settled, counted, free_forms, credit_forms): Row =
        tx.query_row("SELECT account_id, period, forms_requested, reserved_credits, settled, forms_counted, free_forms, credit_forms FROM jobs WHERE id = ?1", [job_id], |r| {
            Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?, r.get(4)?, r.get(5)?, r.get(6)?, r.get(7)?))
        })?;
    if settled != 0 {
        let record = tx.query_row("SELECT id FROM usage_records WHERE job_id = ?1", [job_id], |r| r.get(0)).optional()?;
        return Ok(Settlement { forms_counted: counted.unwrap_or(0), free_forms: free_forms.unwrap_or(0), credit_forms: credit_forms.unwrap_or(0), usage_record: record });
    }
    let forms = released.clamp(0, requested);
    let used: i64 = tx.query_row("SELECT COALESCE(SUM(free_forms), 0) FROM usage_records WHERE account_id = ?1 AND period = ?2", params![account, period], |r| r.get(0))?;
    let others: i64 = tx.query_row(
        "SELECT COALESCE(SUM(reserved_free), 0) FROM jobs WHERE account_id = ?1 AND period = ?2 AND settled = 0 AND id <> ?3",
        params![account, period, job_id],
        |r| r.get(0),
    )?;
    let free = forms.min((prices.free_forms_per_month - used - others).max(0));
    let wanted = (forms - free).min(reserved_credits);
    let credit = if wanted > 0 { draw(tx, &account, wanted, Cause { job: Some(job_id), ..Cause::of("job", "settlement") }, now)? } else { 0 };
    let mut record = None;
    if free + credit > 0 {
        let id = crate::ids::new_id("usage");
        tx.execute(
            "INSERT INTO usage_records (id, account_id, job_id, period, forms, free_forms, credit_forms, created_at) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
            params![id, account, job_id, period, free + credit, free, credit, now],
        )?;
        record = Some(id);
    }
    tx.execute("UPDATE jobs SET settled = 1, forms_counted = ?2, free_forms = ?3, credit_forms = ?4 WHERE id = ?1", params![job_id, free + credit, free, credit])?;
    // what the job held of an expired lot, and didn't use, expires now
    expire_due(tx, &account, now)?;
    Ok(Settlement { forms_counted: free + credit, free_forms: free, credit_forms: credit, usage_record: record })
}

/// A model's use in one job, as its report counts it (`model_usage`).
#[derive(Clone, Debug, PartialEq)]
pub struct ModelUse {
    pub host: String,
    pub model: String,
    pub requests: i64,
    pub input_tokens: i64,
    pub output_tokens: i64,
}

/// The `model_usage` of a job report.
pub fn model_uses(report: &Value) -> Vec<ModelUse> {
    report["model_usage"]
        .as_array()
        .into_iter()
        .flatten()
        .map(|u| ModelUse {
            host: u["host"].as_str().unwrap_or("").into(),
            model: u["model"].as_str().unwrap_or("").into(),
            requests: u["requests"].as_i64().unwrap_or(0),
            input_tokens: u["input_tokens"].as_i64().unwrap_or(0),
            output_tokens: u["output_tokens"].as_i64().unwrap_or(0),
        })
        .collect()
}

/// `tokens` at `per_million` micro-dollars per million tokens, rounded.
fn token_cost(tokens: i64, per_million: i64) -> i64 {
    ((i128::from(tokens) * i128::from(per_million) + 500_000) / 1_000_000) as i64
}

/// What a job actually cost the operator: its CPU time at the server's
/// price per vCPU-hour, and its tokens at each model's prices (PRICING.md,
/// "Cost"). The cost as JSON, its total, and each model's cost.
pub fn cost(cpu_seconds: f64, uses: &[ModelUse], prices: &Prices) -> (Value, i64, Vec<i64>) {
    let server = (cpu_seconds * prices.vcpu_hour_microusd as f64 / 3600.0).round() as i64;
    let mut per_model = Vec::new();
    let models: Vec<Value> = uses
        .iter()
        .map(|u| {
            let p = prices.tokens.get(&u.model);
            let c = p.map(|p| token_cost(u.input_tokens, p.input) + token_cost(u.output_tokens, p.output)).unwrap_or(0);
            per_model.push(c);
            json!({"host": u.host, "model": u.model, "requests": u.requests, "input_tokens": u.input_tokens, "output_tokens": u.output_tokens,
                   "cost_microusd": c, "price_known": p.is_some()})
        })
        .collect();
    let total = server + per_model.iter().sum::<i64>();
    let v = json!({"currency": "usd", "cpu_seconds": (cpu_seconds * 1000.0).round() / 1000.0, "server_microusd": server,
                   "models": models, "total_microusd": total, "total_usd": format_usd(total)});
    (v, total, per_model)
}

/// An account's usage in a month: its records (one per settled job that
/// counted forms), its free forms then, and its credits and purchases now.
pub fn usage(conn: &Connection, account: &str, period: &str, now: i64, prices: &Prices) -> Result<Value> {
    let mut st = conn.prepare(
        "SELECT id, job_id, forms, free_forms, credit_forms, created_at, legacy_plan, legacy_unit_price_microusd, legacy_amount_microusd
         FROM usage_records WHERE account_id = ?1 AND period = ?2 ORDER BY id",
    )?;
    let (mut forms, mut free_forms, mut credit_forms) = (0, 0, 0);
    let records: Vec<Value> = st
        .query_map(params![account, period], |r| {
            let mut v = json!({"id": r.get::<_, String>(0)?, "job_id": r.get::<_, String>(1)?, "forms": r.get::<_, i64>(2)?, "free_forms": r.get::<_, i64>(3)?,
                               "credit_forms": r.get::<_, i64>(4)?, "created_at": crate::clock::rfc3339(r.get(5)?)});
            if let Some(plan) = r.get::<_, Option<String>>(6)? {
                // made under the metered plan, before packages
                v["legacy"] = json!({"plan": plan, "unit_price_microusd": r.get::<_, Option<i64>>(7)?, "amount_microusd": r.get::<_, Option<i64>>(8)?});
            }
            Ok(v)
        })?
        .collect::<rusqlite::Result<_>>()?;
    for r in &records {
        forms += r["forms"].as_i64().unwrap_or(0);
        free_forms += r["free_forms"].as_i64().unwrap_or(0);
        credit_forms += r["credit_forms"].as_i64().unwrap_or(0);
    }
    let reserved_credits: i64 =
        conn.query_row("SELECT COALESCE(SUM(reserved_credits), 0) FROM jobs WHERE account_id = ?1 AND period = ?2 AND settled = 0", params![account, period], |r| r.get(0))?;
    let free = free(conn, account, period, prices)?;
    Ok(json!({
        "object": "usage", "period": period,
        "records": records,
        "forms": forms,
        "free_forms": free_forms,
        "credit_forms": credit_forms,
        "reserved": {"free_forms": free.reserved, "credits": reserved_credits},
        "free": free.to_value(period),
        "credits": credits(conn, account, now)?.to_value(prices),
        "purchases": crate::purchases::summary(conn, account)?,
    }))
}

/// Micro-dollars as dollars: at least two decimals, at most six
/// (4900 → "0.0049", 50_000_000 → "50.00").
pub fn format_usd(micro: i64) -> String {
    let sign = if micro < 0 { "-" } else { "" };
    let m = micro.unsigned_abs();
    let frac = format!("{:06}", m % 1_000_000);
    let trimmed = frac.trim_end_matches('0');
    let frac = if trimmed.len() < 2 { &frac[..2] } else { trimmed };
    format!("{sign}{}.{frac}", m / 1_000_000)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn prices() -> Prices {
        Prices {
            free_forms_per_month: 5,
            rates: Rates { package_forms: 100, package_price_cents: 3500, vat_basis_points: 1600, card_fee_basis_points: 440, card_fee_fixed_cents: 30 },
            max_packages_per_purchase: 100,
            credit_expiry_days: 0,
            tokens: crate::config::parse_model_prices("gemma4:31b=140000/400000,gpt-oss:120b=150000/600000").unwrap(),
            vcpu_hour_microusd: 59850,
        }
    }

    #[test]
    fn dollars_are_written_from_micro_dollars() {
        assert_eq!(format_usd(4900), "0.0049");
        assert_eq!(format_usd(0), "0.00");
        assert_eq!(format_usd(9800), "0.0098");
        assert_eq!(format_usd(1_234_500), "1.2345");
        assert_eq!(format_usd(50_000_000), "50.00");
        assert_eq!(format_usd(-4900), "-0.0049");
    }

    #[test]
    fn the_worst_path_of_pricing_md_costs_what_it_says() {
        // PRICING.md: 74.1 CPU-seconds, gemma4:31b 5,122 in / 807 out,
        // gpt-oss:120b 1,098 in / 1,353 out: $0.003247 in all
        let uses = vec![
            ModelUse { host: "ollama.com".into(), model: "gemma4:31b".into(), requests: 9, input_tokens: 5122, output_tokens: 807 },
            ModelUse { host: "ollama.com".into(), model: "gpt-oss:120b".into(), requests: 3, input_tokens: 1098, output_tokens: 1353 },
        ];
        let (v, total, per_model) = cost(74.1, &uses, &prices());
        assert_eq!(v["server_microusd"], 1232);
        assert_eq!(per_model, vec![1040, 977]);
        assert_eq!(total, 3249); // each item rounded to the micro-dollar
        // $0.35 a form is about 108 times the worst path's $0.003247
        assert_eq!((prices().rates.price_per_form_microusd() as f64 / 3247.0).round(), 108.0);
        let (v, _, _) = cost(0.0, &[ModelUse { host: "h".into(), model: "unknown".into(), requests: 1, input_tokens: 10, output_tokens: 10 }], &prices());
        assert_eq!(v["models"][0]["price_known"], false);
    }

    #[test]
    fn only_what_jobs_do_not_hold_of_an_expired_lot_may_go() {
        let lot = |remaining, expires_at| Lot { id: 0, remaining, expires_at };
        let lots = [lot(30, Some(100)), lot(50, Some(200)), lot(100, None)];
        assert_eq!(expirable(&lots, 0, 99), 0);
        assert_eq!(expirable(&lots, 0, 100), 30);
        assert_eq!(expirable(&lots, 10, 150), 20);
        assert_eq!(expirable(&lots, 10, 200), 70);
        assert_eq!(expirable(&lots, 95, 200), 0);
        assert_eq!(prices().credit_expiry(1000), None);
        assert_eq!(Prices { credit_expiry_days: 2, ..prices() }.credit_expiry(1000), Some(1000 + 2 * 86_400));
    }
}
