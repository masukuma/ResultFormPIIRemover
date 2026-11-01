//! The audit log: every change to keys, plans and settings, with who made
//! it. Details hold ids and settings, never a secret.

use anyhow::Result;
use rusqlite::{Connection, params};
use serde_json::Value;

/// Record that `actor` did `action` (to `account`), with `detail`.
pub fn record(conn: &Connection, now: i64, account: Option<&str>, actor: &str, action: &str, detail: &Value) -> Result<()> {
    conn.execute("INSERT INTO audit_log (at, account_id, actor, action, detail) VALUES (?1, ?2, ?3, ?4, ?5)", params![now, account, actor, action, detail.to_string()])?;
    Ok(())
}

/// Record a deletion of `what` (a path relative to the data folder, or a
/// description), and why.
pub fn deletion(conn: &Connection, now: i64, account: &str, job: Option<&str>, file: Option<&str>, what: &str, reason: &str) -> Result<()> {
    conn.execute("INSERT INTO deletions (at, account_id, job_id, file_id, what, reason) VALUES (?1, ?2, ?3, ?4, ?5, ?6)", params![now, account, job, file, what, reason])?;
    Ok(())
}
