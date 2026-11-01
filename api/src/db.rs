//! The database: SQLite in WAL mode with a busy timeout, so the API and
//! any number of worker processes can share it; versioned migrations.

use anyhow::{Context, Result};
use rusqlite::{Connection, OptionalExtension};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Duration;

/// The migrations, in order. A version, once released, never changes: a
/// change to the schema is a new version.
pub const MIGRATIONS: &[(i64, &str)] = &[(1, include_str!("../migrations/0001_init.sql")), (2, include_str!("../migrations/0002_packages.sql"))];

/// How long a statement waits for another connection's write lock.
const BUSY_TIMEOUT: Duration = Duration::from_secs(10);

/// Open connections, kept for reuse.
pub struct Db {
    path: PathBuf,
    pool: Mutex<Vec<Connection>>,
}

impl Db {
    /// The database at `path`, created if missing, in WAL mode.
    pub fn open(path: &Path) -> Result<Arc<Db>> {
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
        }
        let db = Db { path: path.to_path_buf(), pool: Mutex::new(Vec::new()) };
        let conn = db.connect()?;
        let mode: String = conn.query_row("PRAGMA journal_mode=WAL", [], |r| r.get(0))?;
        anyhow::ensure!(mode.eq_ignore_ascii_case("wal"), "{} can't use WAL mode (got {mode})", path.display());
        db.pool.lock().map_err(|_| anyhow::anyhow!("pool poisoned"))?.push(conn);
        Ok(Arc::new(db))
    }

    fn connect(&self) -> Result<Connection> {
        let conn = Connection::open(&self.path).with_context(|| format!("opening {}", self.path.display()))?;
        conn.busy_timeout(BUSY_TIMEOUT)?;
        conn.execute_batch("PRAGMA foreign_keys=ON; PRAGMA synchronous=NORMAL;")?;
        Ok(conn)
    }

    /// Run `f` on a connection, here and now (blocking).
    pub fn with<R>(&self, f: impl FnOnce(&mut Connection) -> Result<R>) -> Result<R> {
        let pooled = self.pool.lock().map_err(|_| anyhow::anyhow!("pool poisoned"))?.pop();
        let mut conn = match pooled {
            Some(c) => c,
            None => self.connect()?,
        };
        let out = f(&mut conn);
        if let Ok(mut pool) = self.pool.lock()
            && pool.len() < 16
        {
            pool.push(conn);
        }
        out
    }

    /// Run `f` on a connection on the blocking thread pool.
    pub async fn call<R, F>(self: &Arc<Self>, f: F) -> Result<R>
    where
        R: Send + 'static,
        F: FnOnce(&mut Connection) -> Result<R> + Send + 'static,
    {
        let db = self.clone();
        tokio::task::spawn_blocking(move || db.with(f)).await.context("database task")?
    }

    /// Apply the migrations not yet applied; the versions applied now.
    pub fn migrate(&self) -> Result<Vec<i64>> {
        self.with(|conn| {
            conn.execute_batch("CREATE TABLE IF NOT EXISTS schema_migrations (version INTEGER PRIMARY KEY, applied_at INTEGER NOT NULL)")?;
            let mut applied = Vec::new();
            for (version, sql) in MIGRATIONS {
                let tx = conn.transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;
                let done: Option<i64> = tx.query_row("SELECT version FROM schema_migrations WHERE version = ?1", [version], |r| r.get(0)).optional()?;
                if done.is_none() {
                    tx.execute_batch(sql).with_context(|| format!("migration {version}"))?;
                    tx.execute("INSERT INTO schema_migrations (version, applied_at) VALUES (?1, strftime('%s','now'))", [version])?;
                    applied.push(*version);
                }
                tx.commit()?;
            }
            Ok(applied)
        })
    }

    /// The latest migration applied, or 0.
    pub fn version(&self) -> Result<i64> {
        self.with(|conn| Ok(conn.query_row("SELECT COALESCE(MAX(version), 0) FROM schema_migrations", [], |r| r.get(0))?))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn migrations_apply_once_and_the_database_is_in_wal_mode() {
        let dir = std::env::temp_dir().join(format!("rfpii-api-db-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let db = Db::open(&dir.join("x.sqlite3")).unwrap();
        assert_eq!(db.migrate().unwrap(), vec![1, 2]);
        assert!(db.migrate().unwrap().is_empty());
        assert_eq!(db.version().unwrap(), 2);
        let mode: String = db.with(|c| Ok(c.query_row("PRAGMA journal_mode", [], |r| r.get(0))?)).unwrap();
        assert_eq!(mode, "wal");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn the_metered_plan_migrates_to_packages_and_its_records_stay_readable() {
        let dir = std::env::temp_dir().join(format!("rfpii-api-db-v1-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let db = Db::open(&dir.join("x.sqlite3")).unwrap();
        // a database of the metered plan: a free and a paid account, each with
        // a settled job and its usage record, and a free job still running
        db.with(|c| {
            c.execute_batch("CREATE TABLE schema_migrations (version INTEGER PRIMARY KEY, applied_at INTEGER NOT NULL)")?;
            c.execute_batch(MIGRATIONS[0].1)?;
            c.execute_batch(
                "INSERT INTO schema_migrations VALUES (1, 0);
                 INSERT INTO accounts (id, name, plan, created_at) VALUES ('acct_free', 'Free', 'free', 0);
                 INSERT INTO accounts (id, name, plan, spend_cap_microusd, stripe_customer_id, stripe_subscription_id, created_at) VALUES ('acct_paid', 'Paid', 'paid', 50000000, 'cus_1', 'sub_1', 0);
                 INSERT INTO jobs (id, account_id, status, request_json, plan, unit_price_microusd, period, forms_requested, reserved_forms, settled, forms_released, forms_charged, amount_microusd, max_attempts, created_at)
                   VALUES ('job_f', 'acct_free', 'succeeded', '{}', 'free', 0, '2026-10', 3, 0, 1, 2, 2, 0, 3, 0),
                          ('job_p', 'acct_paid', 'succeeded', '{}', 'paid', 4900, '2026-10', 3, 0, 1, 2, 2, 9800, 3, 0),
                          ('job_r', 'acct_free', 'running', '{}', 'free', 0, '2026-10', 2, 2, 0, NULL, NULL, NULL, 3, 0);
                 INSERT INTO usage_records (id, account_id, job_id, period, plan, forms, unit_price_microusd, amount_microusd, created_at, stripe_reported_at)
                   VALUES ('usage_f', 'acct_free', 'job_f', '2026-10', 'free', 2, 0, 0, 0, NULL),
                          ('usage_p', 'acct_paid', 'job_p', '2026-10', 'paid', 2, 4900, 9800, 0, 5);",
            )?;
            Ok(())
        })
        .unwrap();
        assert_eq!(db.migrate().unwrap(), vec![2]);
        let prices = crate::billing::Prices {
            free_forms: 5,
            rates: crate::quote::Rates { package_forms: 100, package_price_cents: 3500, vat_basis_points: 1600, card_fee_basis_points: 440, card_fee_fixed_cents: 30 },
            max_packages_per_purchase: 100,
            credit_expiry_days: 0,
            tokens: Default::default(),
            vcpu_hour_microusd: 0,
        };
        db.with(|c| {
            // the free account: 2 used, 2 held by the running job
            let free = crate::billing::free(c, "acct_free", &prices)?;
            assert_eq!((free.used, free.reserved, free.left), (2, 2, 1));
            // the paid account keeps nothing special: its metered forms were not free ones
            assert_eq!(crate::billing::free(c, "acct_paid", &prices)?.left, 5);
            assert_eq!(crate::billing::credits(c, "acct_paid", 0)?, crate::billing::Credits::default());
            let a = crate::accounts::load(c, "acct_paid")?.unwrap();
            assert_eq!(a.stripe_customer_id.as_deref(), Some("cus_1"));
            // old usage records are read with what the metered plan charged
            let u = crate::billing::usage(c, "acct_paid", "2026-10", 0, &prices)?;
            assert_eq!(u["records"][0]["legacy"], serde_json::json!({"plan": "paid", "unit_price_microusd": 4900, "amount_microusd": 9800}));
            assert_eq!((u["forms"].as_i64(), u["free_forms"].as_i64(), u["credit_forms"].as_i64()), (Some(2), Some(0), Some(0)));
            let u = crate::billing::usage(c, "acct_free", "2026-10", 0, &prices)?;
            assert_eq!((u["records"][0]["free_forms"].as_i64(), u["records"][0]["legacy"]["plan"].as_str()), (Some(2), Some("free")));
            // and old jobs keep what they counted
            let j = crate::jobs::load_any(c, "job_p")?.unwrap();
            assert_eq!((j.forms_counted, j.free_forms, j.credit_forms), (Some(2), Some(0), Some(0)));
            let j = crate::jobs::load_any(c, "job_f")?.unwrap().resource();
            assert_eq!(j["billing"]["counted"], serde_json::json!({"forms": 2, "free_forms": 2, "credits": 0}));
            let j = crate::jobs::load_any(c, "job_r")?.unwrap();
            assert_eq!((j.reserved_free, j.reserved_credits, j.settled), (2, 0, false));
            Ok(())
        })
        .unwrap();
        let _ = std::fs::remove_dir_all(&dir);
    }
}
