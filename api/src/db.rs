//! The database: SQLite in WAL mode with a busy timeout, so the API and
//! any number of worker processes can share it; versioned migrations.

use anyhow::{Context, Result};
use rusqlite::{Connection, OptionalExtension};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Duration;

/// The migrations, in order. A version, once released, never changes: a
/// change to the schema is a new version.
pub const MIGRATIONS: &[(i64, &str)] = &[(1, include_str!("../migrations/0001_init.sql"))];

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
        assert_eq!(db.migrate().unwrap(), vec![1]);
        assert!(db.migrate().unwrap().is_empty());
        assert_eq!(db.version().unwrap(), 1);
        let mode: String = db.with(|c| Ok(c.query_row("PRAGMA journal_mode", [], |r| r.get(0))?)).unwrap();
        assert_eq!(mode, "wal");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
