//! API keys: `rfpii_live_` and 32 random bytes in base62. Only the key's
//! SHA-256 and a short prefix are stored; a key has scopes, a last-used
//! time, and can be revoked.

use anyhow::Result;
use rusqlite::{Connection, OptionalExtension, Row, params};
use serde_json::{Value, json};

use crate::problem::ApiError;

pub const KEY_PREFIX: &str = "rfpii_live_";

/// What a key may do.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Scope {
    JobsWrite,
    JobsRead,
    AccountAdmin,
}

pub const ALL_SCOPES: [Scope; 3] = [Scope::JobsWrite, Scope::JobsRead, Scope::AccountAdmin];

impl Scope {
    pub fn as_str(self) -> &'static str {
        match self {
            Scope::JobsWrite => "jobs:write",
            Scope::JobsRead => "jobs:read",
            Scope::AccountAdmin => "account:admin",
        }
    }

    pub fn parse(s: &str) -> Option<Scope> {
        ALL_SCOPES.into_iter().find(|x| x.as_str() == s)
    }
}

/// Scopes from a space- or comma-separated list; an unknown one is an error.
pub fn parse_scopes(s: &str) -> Result<Vec<Scope>, String> {
    let mut out = Vec::new();
    for w in s.split([' ', ',']).filter(|w| !w.is_empty()) {
        let sc = Scope::parse(w).ok_or_else(|| format!("unknown scope {w:?} (scopes: jobs:write, jobs:read, account:admin)"))?;
        if !out.contains(&sc) {
            out.push(sc);
        }
    }
    if out.is_empty() {
        return Err("a key needs at least one scope".into());
    }
    Ok(out)
}

fn join(scopes: &[Scope]) -> String {
    scopes.iter().map(|s| s.as_str()).collect::<Vec<_>>().join(" ")
}

/// The caller of an authenticated request.
#[derive(Clone, Debug)]
pub struct Caller {
    pub account_id: String,
    pub key_id: String,
    pub scopes: Vec<Scope>,
}

impl Caller {
    /// 403 unless the key has `scope`.
    pub fn require(&self, scope: Scope) -> Result<(), ApiError> {
        if self.scopes.contains(&scope) { Ok(()) } else { Err(ApiError::forbidden_scope(scope.as_str())) }
    }

    /// Who did it, for the audit log.
    pub fn actor(&self) -> String {
        format!("key:{}", self.key_id)
    }
}

/// A new key's secret, hash and display prefix.
pub fn generate() -> (String, String, String) {
    // 32 bytes are at most 43 base62 digits: padded, every key has 43
    let secret = format!("{KEY_PREFIX}{:0>43}", crate::ids::base62(&crate::ids::random_bytes::<32>()));
    let prefix = secret[..KEY_PREFIX.len() + 6].to_string();
    (crate::ids::sha256_hex(secret.as_bytes()), prefix, secret)
}

/// A stored key, as listed (never the secret).
#[derive(Clone, Debug)]
pub struct KeyRow {
    pub id: String,
    pub account_id: String,
    pub name: Option<String>,
    pub prefix: String,
    pub scopes: String,
    pub created_at: i64,
    pub last_used_at: Option<i64>,
    pub revoked_at: Option<i64>,
}

impl KeyRow {
    pub fn to_value(&self) -> Value {
        let t = |x: Option<i64>| x.map(crate::clock::rfc3339);
        json!({"id": self.id, "object": "api_key", "name": self.name, "prefix": self.prefix,
               "scopes": self.scopes.split(' ').collect::<Vec<_>>(), "created_at": crate::clock::rfc3339(self.created_at),
               "last_used_at": t(self.last_used_at), "revoked_at": t(self.revoked_at)})
    }
}

const COLUMNS: &str = "id, account_id, name, prefix, scopes, created_at, last_used_at, revoked_at";

fn from_row(r: &Row) -> rusqlite::Result<KeyRow> {
    Ok(KeyRow {
        id: r.get(0)?,
        account_id: r.get(1)?,
        name: r.get(2)?,
        prefix: r.get(3)?,
        scopes: r.get(4)?,
        created_at: r.get(5)?,
        last_used_at: r.get(6)?,
        revoked_at: r.get(7)?,
    })
}

/// Create a key for `account`; the row and the secret, which is shown once.
pub fn create(conn: &Connection, now: i64, account: &str, name: Option<&str>, scopes: &[Scope], actor: &str) -> Result<(KeyRow, String)> {
    let (hash, prefix, secret) = generate();
    let id = crate::ids::new_id("key");
    conn.execute(
        "INSERT INTO api_keys (id, account_id, name, prefix, sha256, scopes, created_at) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
        params![id, account, name, prefix, hash, join(scopes), now],
    )?;
    crate::audit::record(conn, now, Some(account), actor, "key.created", &json!({"key_id": id, "prefix": prefix, "scopes": join(scopes)}))?;
    let row = conn.query_row(&format!("SELECT {COLUMNS} FROM api_keys WHERE id = ?1"), [&id], from_row)?;
    Ok((row, secret))
}

pub fn list(conn: &Connection, account: &str) -> Result<Vec<KeyRow>> {
    let mut st = conn.prepare(&format!("SELECT {COLUMNS} FROM api_keys WHERE account_id = ?1 ORDER BY id"))?;
    let rows = st.query_map([account], from_row)?.collect::<rusqlite::Result<Vec<_>>>()?;
    Ok(rows)
}

/// Revoke a key of `account`; the key, or None if it isn't one of its keys.
pub fn revoke(conn: &Connection, now: i64, account: &str, id: &str, actor: &str) -> Result<Option<KeyRow>> {
    let n = conn.execute("UPDATE api_keys SET revoked_at = COALESCE(revoked_at, ?3) WHERE id = ?1 AND account_id = ?2", params![id, account, now])?;
    if n == 0 {
        return Ok(None);
    }
    crate::audit::record(conn, now, Some(account), actor, "key.revoked", &json!({"key_id": id}))?;
    Ok(conn.query_row(&format!("SELECT {COLUMNS} FROM api_keys WHERE id = ?1"), [id], from_row).optional()?)
}

/// The caller holding the bearer token in `authorization`, or 401. The key
/// is looked up by its SHA-256, so the secret is never compared byte by
/// byte; its last-used time is updated at most once a minute.
pub fn authenticate(conn: &Connection, now: i64, authorization: Option<&str>) -> Result<Caller, ApiError> {
    let Some(h) = authorization else {
        return Err(ApiError::unauthorized("an API key is required: Authorization: Bearer rfpii_live_...", None));
    };
    let token = h.strip_prefix("Bearer ").or_else(|| h.strip_prefix("bearer ")).map(str::trim).unwrap_or("");
    if !token.starts_with(KEY_PREFIX) {
        return Err(ApiError::unauthorized("the Authorization header must be: Bearer rfpii_live_...", Some("invalid_request")));
    }
    let hash = crate::ids::sha256_hex(token.as_bytes());
    let found: Option<(String, String, String, Option<i64>)> = conn
        .query_row("SELECT id, account_id, scopes, revoked_at FROM api_keys WHERE sha256 = ?1", [&hash], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?)))
        .optional()
        .map_err(|e| ApiError::internal(&e.into()))?;
    let Some((id, account, scopes, revoked)) = found else {
        return Err(ApiError::unauthorized("the API key is not valid", Some("invalid_token")));
    };
    if revoked.is_some() {
        return Err(ApiError::unauthorized("the API key has been revoked", Some("invalid_token")));
    }
    conn.execute("UPDATE api_keys SET last_used_at = ?2 WHERE id = ?1 AND (last_used_at IS NULL OR last_used_at < ?2 - 60)", params![id, now])
        .map_err(|e| ApiError::internal(&e.into()))?;
    let scopes = scopes.split(' ').filter_map(Scope::parse).collect();
    Ok(Caller { account_id: account, key_id: id, scopes })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_key_is_its_prefix_and_43_base62_characters_and_only_its_hash_is_kept() {
        let (hash, prefix, secret) = generate();
        assert!(secret.starts_with(KEY_PREFIX) && secret.len() == KEY_PREFIX.len() + 43);
        assert!(secret.starts_with(&prefix) && prefix.len() == KEY_PREFIX.len() + 6);
        assert_eq!(hash, crate::ids::sha256_hex(secret.as_bytes()));
        assert_ne!(generate().2, secret);
    }

    #[test]
    fn scopes_are_parsed_strictly() {
        assert_eq!(parse_scopes("jobs:read, jobs:write jobs:read").unwrap(), vec![Scope::JobsRead, Scope::JobsWrite]);
        assert!(parse_scopes("jobs:delete").is_err());
        assert!(parse_scopes(" ").is_err());
    }
}
