//! Plans, the free allowance, usage and the price (PRICING.md). A form is
//! a document as `rfpii inventory` counts it; only forms a job releases are
//! charged or count against the free allowance. A job reserves its forms
//! when it is created, and settles when it ends: the forms released are
//! recorded, the rest of the reservation is let go. Money is integer
//! micro-dollars.

use anyhow::Result;
use axum::http::StatusCode;
use rusqlite::{Connection, Transaction, params};
use serde_json::{Value, json};
use std::collections::BTreeMap;

use crate::accounts::Account;
use crate::config::{Config, TokenPrice};
use crate::problem::ApiError;

/// The rates in force.
#[derive(Clone, Debug)]
pub struct Prices {
    pub free_forms_per_month: i64,
    pub unit_microusd: i64,
    pub tokens: BTreeMap<String, TokenPrice>,
    pub vcpu_hour_microusd: i64,
}

impl Prices {
    pub fn from_config(cfg: &Config) -> Result<Prices> {
        Ok(Prices {
            free_forms_per_month: cfg.free_forms_per_month,
            unit_microusd: cfg.price_per_form_microusd,
            tokens: cfg.token_prices()?,
            vcpu_hour_microusd: cfg.server_price_per_vcpu_hour_microusd,
        })
    }
}

/// A month of an account: forms released on each plan, what they cost,
/// and what jobs not yet settled hold.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Month {
    pub released_free: i64,
    pub released_paid: i64,
    pub spend_microusd: i64,
    pub reserved_free: i64,
    pub reserved_paid: i64,
    pub reserved_microusd: i64,
}

pub fn month(conn: &Connection, account: &str, period: &str) -> Result<Month> {
    let mut m = Month::default();
    let mut st =
        conn.prepare_cached("SELECT plan, COALESCE(SUM(forms), 0), COALESCE(SUM(amount_microusd), 0) FROM usage_records WHERE account_id = ?1 AND period = ?2 GROUP BY plan")?;
    for row in st.query_map(params![account, period], |r| Ok((r.get::<_, String>(0)?, r.get::<_, i64>(1)?, r.get::<_, i64>(2)?)))? {
        let (plan, forms, amount) = row?;
        if plan == "paid" {
            m.released_paid = forms;
        } else {
            m.released_free = forms;
        }
        m.spend_microusd += amount;
    }
    let mut st = conn.prepare_cached(
        "SELECT plan, COALESCE(SUM(reserved_forms), 0), COALESCE(SUM(reserved_forms * unit_price_microusd), 0) FROM jobs WHERE account_id = ?1 AND period = ?2 AND settled = 0 GROUP BY plan",
    )?;
    for row in st.query_map(params![account, period], |r| Ok((r.get::<_, String>(0)?, r.get::<_, i64>(1)?, r.get::<_, i64>(2)?)))? {
        let (plan, forms, amount) = row?;
        if plan == "paid" {
            m.reserved_paid = forms;
            m.reserved_microusd = amount;
        } else {
            m.reserved_free = forms;
        }
    }
    Ok(m)
}

/// Free forms left in a month.
pub fn free_left(m: &Month, prices: &Prices) -> i64 {
    (prices.free_forms_per_month - m.released_free - m.reserved_free).max(0)
}

/// Whether `account` may start a job of `forms` forms in `period`: a free
/// account within what is left of its allowance, a paid one within its
/// spend cap (if it has one). 402 Payment Required otherwise.
pub fn admit(conn: &Connection, account: &Account, forms: i64, period: &str, prices: &Prices, cfg: &Config) -> Result<(), ApiError> {
    let m = month(conn, &account.id, period)?;
    if account.plan == "free" {
        let left = free_left(&m, prices);
        if forms > left {
            let mut upgrade = json!({"pricing": cfg.pricing_url, "price_per_form_microusd": prices.unit_microusd});
            if cfg.stripe_secret_key.is_some() && cfg.stripe_price_id.is_some() {
                upgrade["checkout"] = json!("POST /v1/billing/checkout");
            }
            return Err(ApiError::new(
                StatusCode::PAYMENT_REQUIRED,
                Some("free-allowance-exceeded"),
                "Free allowance exceeded",
                format!(
                    "this job has {forms} form(s) and the free plan has {left} of its {} left in {period}; upgrade to the paid plan (${} per form) with POST /v1/billing/checkout, or ask the operator",
                    prices.free_forms_per_month,
                    format_usd(prices.unit_microusd)
                ),
            )
            .with("forms_requested", forms)
            .with("forms_left", left)
            .with("free_forms_per_month", prices.free_forms_per_month)
            .with("period", period)
            .with("upgrade", upgrade));
        }
        return Ok(());
    }
    if let Some(cap) = account.spend_cap_microusd {
        let estimate = forms * prices.unit_microusd;
        if m.spend_microusd + m.reserved_microusd + estimate > cap {
            return Err(ApiError::new(
                StatusCode::PAYMENT_REQUIRED,
                Some("spend-cap-reached"),
                "Monthly spend cap reached",
                format!(
                    "this job could cost ${} and the account's cap for {period} is ${}, of which ${} is spent and ${} is held by jobs not yet ended",
                    format_usd(estimate),
                    format_usd(cap),
                    format_usd(m.spend_microusd),
                    format_usd(m.reserved_microusd)
                ),
            )
            .with("spend_cap_microusd", cap)
            .with("spent_microusd", m.spend_microusd)
            .with("reserved_microusd", m.reserved_microusd)
            .with("job_estimate_microusd", estimate));
        }
    }
    Ok(())
}

/// What a job was charged when it settled.
#[derive(Clone, Debug, PartialEq)]
pub struct Settlement {
    pub forms_charged: i64,
    pub amount_microusd: i64,
    pub usage_record: Option<String>,
}

/// Settle a job: the forms it released (never more than it reserved) are
/// recorded as usage of its plan at its unit price, and its reservation is
/// let go. Settling twice changes nothing.
pub fn settle(tx: &Transaction, job_id: &str, released: i64, now: i64) -> Result<Settlement> {
    let (account, plan, period, requested, unit, settled, charged, amount): (String, String, String, i64, i64, i64, Option<i64>, Option<i64>) =
        tx.query_row("SELECT account_id, plan, period, forms_requested, unit_price_microusd, settled, forms_charged, amount_microusd FROM jobs WHERE id = ?1", [job_id], |r| {
            Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?, r.get(4)?, r.get(5)?, r.get(6)?, r.get(7)?))
        })?;
    if settled != 0 {
        let record = tx.query_row("SELECT id FROM usage_records WHERE job_id = ?1", [job_id], |r| r.get(0)).ok();
        return Ok(Settlement { forms_charged: charged.unwrap_or(0), amount_microusd: amount.unwrap_or(0), usage_record: record });
    }
    let forms = released.clamp(0, requested);
    let amount = if plan == "paid" { forms * unit } else { 0 };
    let mut record = None;
    if forms > 0 {
        let id = crate::ids::new_id("usage");
        tx.execute(
            "INSERT INTO usage_records (id, account_id, job_id, period, plan, forms, unit_price_microusd, amount_microusd, created_at, stripe_next_at) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?9)",
            params![id, account, job_id, period, plan, forms, unit, amount, now],
        )?;
        record = Some(id);
    }
    tx.execute("UPDATE jobs SET settled = 1, reserved_forms = 0, forms_charged = ?2, amount_microusd = ?3 WHERE id = ?1", params![job_id, forms, amount])?;
    Ok(Settlement { forms_charged: forms, amount_microusd: amount, usage_record: record })
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

/// An account's usage in a month: its records, and the amount due.
pub fn usage(conn: &Connection, account: &str, period: &str, prices: &Prices) -> Result<Value> {
    let mut st = conn.prepare(
        "SELECT id, job_id, plan, forms, unit_price_microusd, amount_microusd, created_at, stripe_reported_at FROM usage_records WHERE account_id = ?1 AND period = ?2 ORDER BY id",
    )?;
    let records: Vec<Value> = st
        .query_map(params![account, period], |r| {
            Ok(json!({"id": r.get::<_, String>(0)?, "job_id": r.get::<_, String>(1)?, "plan": r.get::<_, String>(2)?, "forms": r.get::<_, i64>(3)?,
                      "unit_price_microusd": r.get::<_, i64>(4)?, "amount_microusd": r.get::<_, i64>(5)?,
                      "created_at": crate::clock::rfc3339(r.get(6)?), "reported_to_stripe": r.get::<_, Option<i64>>(7)?.is_some()}))
        })?
        .collect::<rusqlite::Result<_>>()?;
    let m = month(conn, account, period)?;
    // a paid account pays for every form: it has no free allowance left
    let paid = crate::accounts::load(conn, account)?.is_some_and(|a| a.plan == "paid");
    Ok(json!({
        "object": "usage", "period": period, "currency": "usd",
        "records": records,
        "forms": m.released_free + m.released_paid,
        "free_forms": m.released_free,
        "paid_forms": m.released_paid,
        "free_forms_left": if paid { 0 } else { free_left(&m, prices) },
        "reserved_forms": m.reserved_free + m.reserved_paid,
        "amount_due_microusd": m.spend_microusd,
        "amount_due_usd": format_usd(m.spend_microusd),
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
            unit_microusd: 4900,
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
        // the price is that cost times 1.5, rounded up to $0.0001
        assert_eq!(prices().unit_microusd, ((total as f64 * 1.5 / 100.0).ceil() as i64) * 100);
        let (v, _, _) = cost(0.0, &[ModelUse { host: "h".into(), model: "unknown".into(), requests: 1, input_tokens: 10, output_tokens: 10 }], &prices());
        assert_eq!(v["models"][0]["price_known"], false);
    }
}
