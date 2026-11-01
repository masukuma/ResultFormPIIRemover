//! Purchases of form packages: by Stripe Checkout, pending until Stripe's
//! webhook says the session was paid for the quoted amount; or recorded by
//! the operator for a payment made outside Stripe (a bank transfer), with
//! VAT and no card fee. Each keeps its quote line by line, with the VAT of
//! each line, as a tax invoice needs it. A paid purchase adds its forms to
//! the account's credits, once.

use anyhow::Result;
use rusqlite::{Connection, OptionalExtension, Row, Transaction, params};
use serde_json::{Value, json};

use crate::billing::Prices;
use crate::quote::{Quote, format_cents};

/// A stored purchase.
#[derive(Clone, Debug)]
pub struct Purchase {
    pub id: String,
    pub account_id: String,
    /// "stripe" or "manual"
    pub method: String,
    /// "pending", "paid", "expired" or "mismatch"
    pub status: String,
    pub quote: Quote,
    pub currency: String,
    pub stripe_session_id: Option<String>,
    pub reference: Option<String>,
    pub amount_received_cents: Option<i64>,
    pub currency_received: Option<String>,
    pub created_at: i64,
    pub paid_at: Option<i64>,
    pub closed_at: Option<i64>,
}

const COLUMNS: &str = "id, account_id, method, status, packages, forms, package_price_cents, net_cents, card_fee_cents, vat_rate_basis_points, \
    vat_net_cents, vat_card_fee_cents, vat_cents, total_cents, currency, stripe_session_id, reference, amount_received_cents, currency_received, \
    created_at, paid_at, closed_at";

fn from_row(r: &Row) -> rusqlite::Result<Purchase> {
    let (packages, forms): (i64, i64) = (r.get(4)?, r.get(5)?);
    Ok(Purchase {
        id: r.get(0)?,
        account_id: r.get(1)?,
        method: r.get(2)?,
        status: r.get(3)?,
        quote: Quote {
            packages,
            forms,
            package_forms: forms / packages.max(1),
            package_price_cents: r.get(6)?,
            net_cents: r.get(7)?,
            card_fee_cents: r.get(8)?,
            vat_basis_points: r.get(9)?,
            vat_net_cents: r.get(10)?,
            vat_card_fee_cents: r.get(11)?,
            vat_cents: r.get(12)?,
            total_cents: r.get(13)?,
            // the operator's figures are not kept
            processor_fee_cents: 0,
            kept_cents: 0,
        },
        currency: r.get(14)?,
        stripe_session_id: r.get(15)?,
        reference: r.get(16)?,
        amount_received_cents: r.get(17)?,
        currency_received: r.get(18)?,
        created_at: r.get(19)?,
        paid_at: r.get(20)?,
        closed_at: r.get(21)?,
    })
}

impl Purchase {
    /// The purchase as the API returns it.
    pub fn resource(&self) -> Value {
        let q = &self.quote;
        let t = |x: Option<i64>| x.map(crate::clock::rfc3339);
        json!({
            "id": self.id,
            "object": "purchase",
            "status": self.status,
            "method": self.method,
            "packages": q.packages,
            "forms": q.forms,
            "currency": self.currency,
            "lines": q.lines(None),
            "net_cents": q.net_cents,
            "card_fee_cents": q.card_fee_cents,
            "vat_rate_basis_points": q.vat_basis_points,
            "vat_cents": q.vat_cents,
            "total_cents": q.total_cents,
            "total_usd": format_cents(q.total_cents),
            "reference": self.reference,
            "stripe_session_id": self.stripe_session_id,
            "amount_received_cents": self.amount_received_cents,
            "currency_received": self.currency_received,
            "created_at": crate::clock::rfc3339(self.created_at),
            "paid_at": t(self.paid_at),
            "closed_at": t(self.closed_at),
        })
    }
}

fn insert(conn: &Connection, now: i64, account: &str, method: &str, status: &str, q: &Quote, reference: Option<&str>) -> Result<String> {
    let id = crate::ids::new_id("pur");
    conn.execute(
        "INSERT INTO purchases (id, account_id, method, status, packages, forms, package_price_cents, net_cents, card_fee_cents, vat_rate_basis_points,
                                vat_net_cents, vat_card_fee_cents, vat_cents, total_cents, reference, created_at, paid_at)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16, ?17)",
        params![
            id,
            account,
            method,
            status,
            q.packages,
            q.forms,
            q.package_price_cents,
            q.net_cents,
            q.card_fee_cents,
            q.vat_basis_points,
            q.vat_net_cents,
            q.vat_card_fee_cents,
            q.vat_cents,
            q.total_cents,
            reference,
            now,
            (status == "paid").then_some(now)
        ],
    )?;
    Ok(id)
}

/// A purchase, whoever's it is.
pub fn load_any(conn: &Connection, id: &str) -> Result<Option<Purchase>> {
    Ok(conn.query_row(&format!("SELECT {COLUMNS} FROM purchases WHERE id = ?1"), [id], from_row).optional()?)
}

fn by_session(conn: &Connection, session: &str) -> Result<Option<Purchase>> {
    Ok(conn.query_row(&format!("SELECT {COLUMNS} FROM purchases WHERE stripe_session_id = ?1"), [session], from_row).optional()?)
}

/// A purchase by Stripe Checkout of `quote`, pending until it is paid.
pub fn create_pending(conn: &Connection, now: i64, account: &str, quote: &Quote, actor: &str) -> Result<Purchase> {
    let id = insert(conn, now, account, "stripe", "pending", quote, None)?;
    crate::audit::record(conn, now, Some(account), actor, "purchase.checkout_started", &json!({"purchase": id, "packages": quote.packages, "total_cents": quote.total_cents}))?;
    load_any(conn, &id)?.ok_or_else(|| anyhow::anyhow!("purchase {id} vanished"))
}

/// Record the Checkout Session made for a pending purchase.
pub fn set_session(conn: &Connection, id: &str, session: &str) -> Result<()> {
    conn.execute("UPDATE purchases SET stripe_session_id = ?2 WHERE id = ?1 AND status = 'pending'", params![id, session])?;
    Ok(())
}

/// Forget a pending purchase whose Checkout Session could not be made.
pub fn discard(conn: &Connection, id: &str) -> Result<()> {
    conn.execute("DELETE FROM purchases WHERE id = ?1 AND status = 'pending' AND stripe_session_id IS NULL", [id])?;
    Ok(())
}

/// A page of an account's purchases, newest first, after `after`; and
/// whether there are more.
pub fn list(conn: &Connection, account: &str, limit: i64, after: Option<&str>) -> Result<(Vec<Purchase>, bool)> {
    let mut st = conn.prepare(&format!("SELECT {COLUMNS} FROM purchases WHERE account_id = ?1 AND (?2 IS NULL OR id < ?2) ORDER BY id DESC LIMIT ?3"))?;
    let mut rows = st.query_map(params![account, after, limit + 1], from_row)?.collect::<rusqlite::Result<Vec<_>>>()?;
    let more = rows.len() as i64 > limit;
    rows.truncate(limit as usize);
    Ok((rows, more))
}

/// An account's purchases in brief: how many in each status, the forms
/// bought and what was paid.
pub fn summary(conn: &Connection, account: &str) -> Result<Value> {
    let mut v = json!({"paid": 0, "pending": 0, "expired": 0, "mismatch": 0});
    let mut st = conn.prepare_cached("SELECT status, COUNT(*) FROM purchases WHERE account_id = ?1 GROUP BY status")?;
    for row in st.query_map([account], |r| Ok((r.get::<_, String>(0)?, r.get::<_, i64>(1)?)))? {
        let (status, n) = row?;
        v[status] = json!(n);
    }
    let (forms, paid, last): (i64, i64, Option<i64>) =
        conn.query_row("SELECT COALESCE(SUM(forms), 0), COALESCE(SUM(total_cents), 0), MAX(paid_at) FROM purchases WHERE account_id = ?1 AND status = 'paid'", [account], |r| {
            Ok((r.get(0)?, r.get(1)?, r.get(2)?))
        })?;
    v["forms_bought"] = json!(forms);
    v["total_paid_cents"] = json!(paid);
    v["total_paid_usd"] = json!(format_cents(paid));
    v["last_paid_at"] = json!(last.map(crate::clock::rfc3339));
    Ok(v)
}

/// Record `packages` packages paid outside Stripe (a bank transfer, named
/// by `reference`): a paid purchase with VAT and no card fee, its forms
/// added to the account's credits, audited. A message if the request is
/// not one the API accepts.
pub fn add_manual(tx: &Transaction, now: i64, prices: &Prices, account: &str, packages: i64, reference: &str, actor: &str) -> Result<Result<Purchase, String>> {
    let reference = reference.trim();
    if !(1..=prices.max_packages_per_purchase).contains(&packages) {
        return Ok(Err(format!("packages must be from 1 to {} (MAX_PACKAGES_PER_PURCHASE)", prices.max_packages_per_purchase)));
    }
    if reference.is_empty() || reference.chars().count() > 200 {
        return Ok(Err("a purchase paid outside Stripe needs a reference of 1 to 200 characters (the payment's)".into()));
    }
    let q = prices.rates.quote_without_card_fee(packages);
    let id = insert(tx, now, account, "manual", "paid", &q, Some(reference))?;
    let cause = crate::billing::Cause { purchase: Some(&id), note: Some(reference), ..crate::billing::Cause::of("purchase", actor) };
    crate::billing::add_lot(tx, account, q.forms, cause, now, prices)?;
    crate::audit::record(
        tx,
        now,
        Some(account),
        actor,
        "credits.added",
        &json!({"purchase": id, "packages": packages, "forms": q.forms, "total_cents": q.total_cents, "reference": reference}),
    )?;
    Ok(Ok(load_any(tx, &id)?.ok_or_else(|| anyhow::anyhow!("purchase {id} vanished"))?))
}

/// The purchase a Checkout Session is for: by its `metadata.purchase_id`
/// (of the account it names, and of this session if one is recorded), else
/// by the session's id.
fn of_session(tx: &Transaction, s: &Value) -> Result<Option<Purchase>> {
    let session = s["id"].as_str().unwrap_or("");
    if let Some(id) = s["metadata"]["purchase_id"].as_str()
        && let Some(p) = load_any(tx, id)?
    {
        let account_ok = s["client_reference_id"].as_str().is_none_or(|a| a == p.account_id);
        let session_ok = p.stripe_session_id.as_deref().is_none_or(|x| x == session);
        if account_ok && session_ok {
            return Ok(Some(p));
        }
        tracing::warn!(purchase = %p.id, session, "a Checkout Session names a purchase of another account or session; ignored");
        return Ok(None);
    }
    by_session(tx, session)
}

/// A Checkout Session was completed (`checkout.session.completed`, or
/// `checkout.session.async_payment_succeeded`). If it is paid, for the
/// quoted total in the quoted currency, the purchase is paid and its forms
/// are added to the account's credits; if the amount or currency differ,
/// it is recorded as a mismatch and nothing is added. Whether anything
/// changed; a purchase already settled is left as it is.
pub fn complete(tx: &Transaction, now: i64, prices: &Prices, s: &Value) -> Result<bool> {
    let Some(p) = of_session(tx, s)? else {
        tracing::warn!(session = s["id"].as_str().unwrap_or(""), "a completed Checkout Session matches no purchase");
        return Ok(false);
    };
    if p.status != "pending" || s["payment_status"].as_str() != Some("paid") {
        return Ok(false);
    }
    let session = s["id"].as_str();
    let amount = s["amount_total"].as_i64();
    let currency = s["currency"].as_str().map(str::to_ascii_lowercase);
    if amount != Some(p.quote.total_cents) || currency.as_deref() != Some(p.currency.as_str()) {
        tx.execute(
            "UPDATE purchases SET status = 'mismatch', amount_received_cents = ?2, currency_received = ?3, closed_at = ?4, stripe_session_id = COALESCE(stripe_session_id, ?5) WHERE id = ?1",
            params![p.id, amount, currency, now, session],
        )?;
        tracing::error!(purchase = %p.id, expected_cents = p.quote.total_cents, received_cents = ?amount, currency = ?currency, "a Stripe payment does not match its quote; no credits added");
        crate::audit::record(
            tx,
            now,
            Some(&p.account_id),
            "stripe",
            "purchase.mismatch",
            &json!({"purchase": p.id, "expected_cents": p.quote.total_cents, "received_cents": amount, "currency": currency}),
        )?;
        return Ok(true);
    }
    tx.execute(
        "UPDATE purchases SET status = 'paid', paid_at = ?2, amount_received_cents = ?3, currency_received = ?4, stripe_session_id = COALESCE(stripe_session_id, ?5) WHERE id = ?1",
        params![p.id, now, amount, currency, session],
    )?;
    if let Some(customer) = s["customer"].as_str() {
        // a customer id another account has already is left alone
        tx.execute("UPDATE OR IGNORE accounts SET stripe_customer_id = ?2 WHERE id = ?1 AND stripe_customer_id IS NULL", params![p.account_id, customer])?;
    }
    crate::billing::add_lot(tx, &p.account_id, p.quote.forms, crate::billing::Cause { purchase: Some(&p.id), ..crate::billing::Cause::of("purchase", "stripe") }, now, prices)?;
    crate::audit::record(
        tx,
        now,
        Some(&p.account_id),
        "stripe",
        "purchase.paid",
        &json!({"purchase": p.id, "packages": p.quote.packages, "forms": p.quote.forms, "total_cents": p.quote.total_cents}),
    )?;
    Ok(true)
}

/// A Checkout Session expired unpaid (`checkout.session.expired`): its
/// pending purchase is marked expired. Whether anything changed.
pub fn expire(tx: &Transaction, now: i64, s: &Value) -> Result<bool> {
    let Some(p) = of_session(tx, s)? else { return Ok(false) };
    if p.status != "pending" {
        return Ok(false);
    }
    tx.execute("UPDATE purchases SET status = 'expired', closed_at = ?2, stripe_session_id = COALESCE(stripe_session_id, ?3) WHERE id = ?1", params![p.id, now, s["id"].as_str()])?;
    crate::audit::record(tx, now, Some(&p.account_id), "stripe", "purchase.expired", &json!({"purchase": p.id}))?;
    Ok(true)
}
