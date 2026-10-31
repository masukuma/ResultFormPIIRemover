//! Workers: each leases a queued job from the database (an atomic
//! `UPDATE … RETURNING` with a lease that a heartbeat renews), runs one
//! `rfpii job` process for it in the job's folder, and settles it when the
//! process ends. A lease that runs out puts the job back in the queue,
//! until it has been tried `MAX_ATTEMPTS` times. Cancellation, the job's
//! time limit and shutdown stop the process (its whole process group).

use anyhow::{Context, Result};
use rusqlite::{OptionalExtension, params};
use serde_json::Value;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};
use tokio::sync::{oneshot, watch};

use crate::jobs::{self, JobRow};
use crate::server::AppState;
use crate::storage;

/// Seconds between SIGTERM and SIGKILL when a job is stopped.
const KILL_GRACE: Duration = Duration::from_secs(5);

/// Folders of a job's output that hold originals or pages not released;
/// deleted with its inputs.
const UNRELEASED: [&str; 4] = ["staging", "review", "quarantine", "defects"];

/// Paths inside a job's folder.
pub struct JobPaths {
    pub dir: PathBuf,
}

impl JobPaths {
    pub fn new(data_dir: &Path, account: &str, job: &str) -> JobPaths {
        JobPaths { dir: storage::job_dir(data_dir, account, job) }
    }
    pub fn inputs(&self) -> PathBuf {
        self.dir.join("in")
    }
    pub fn job_file(&self) -> PathBuf {
        self.dir.join("job.yaml")
    }
    pub fn controller(&self) -> PathBuf {
        self.dir.join("controller.json")
    }
    pub fn out(&self) -> PathBuf {
        self.dir.join("out")
    }
    pub fn public(&self) -> PathBuf {
        self.out().join("public")
    }
    pub fn report(&self) -> PathBuf {
        self.out().join("report.json")
    }
    pub fn log(&self) -> PathBuf {
        self.dir.join("rfpii.log")
    }
    /// Records kept with the outputs: the transfers abroad.
    pub fn records(&self) -> PathBuf {
        self.dir.join("records")
    }
}

/// Start `n` workers; they stop when `shutdown` turns true.
pub fn spawn(state: &AppState, n: usize, shutdown: &watch::Receiver<bool>) -> Vec<tokio::task::JoinHandle<()>> {
    (0..n)
        .map(|i| {
            let (state, shutdown) = (state.clone(), shutdown.clone());
            let owner = format!("{}-{}-{}", hostname(), std::process::id(), i);
            tokio::spawn(async move { worker_loop(state, owner, shutdown).await })
        })
        .collect()
}

fn hostname() -> String {
    std::fs::read_to_string("/proc/sys/kernel/hostname").map(|s| s.trim().to_string()).unwrap_or_else(|_| "worker".into())
}

async fn worker_loop(state: AppState, owner: String, mut shutdown: watch::Receiver<bool>) {
    tracing::info!(worker = %owner, "worker started");
    while !*shutdown.borrow() {
        if let Err(e) = reap(&state).await {
            tracing::error!(error = %format!("{e:#}"), "reaping expired leases failed");
        }
        match lease(&state, &owner).await {
            Ok(Some(job)) => run_job(&state, job, &owner, &mut shutdown).await,
            Ok(None) => idle(&state, &mut shutdown).await,
            Err(e) => {
                tracing::error!(error = %format!("{e:#}"), "leasing a job failed");
                idle(&state, &mut shutdown).await;
            }
        }
    }
    tracing::info!(worker = %owner, "worker stopped");
}

async fn idle(state: &AppState, shutdown: &mut watch::Receiver<bool>) {
    tokio::select! {
        _ = tokio::time::sleep(Duration::from_millis(state.cfg.poll_interval_ms)) => {}
        _ = shutdown.changed() => {}
    }
}

/// Lease the oldest queued job for `owner`.
pub async fn lease(state: &AppState, owner: &str) -> Result<Option<JobRow>> {
    let (now, secs, owner) = (state.clock.now(), state.cfg.lease_secs, owner.to_string());
    state
        .db
        .call(move |c| {
            let sql = format!(
                "UPDATE jobs SET status = 'running', lease_owner = ?1, lease_expires_at = ?2 + ?3, attempts = attempts + 1, started_at = COALESCE(started_at, ?2)
                 WHERE id = (SELECT id FROM jobs WHERE status = 'queued' AND cancel_requested = 0 ORDER BY created_at, id LIMIT 1)
                 RETURNING {}",
                jobs::COLUMNS
            );
            Ok(c.query_row(&sql, params![owner, now, secs], jobs::from_row).optional()?)
        })
        .await
}

/// Jobs whose lease ran out: back to the queue, or failed once they have
/// been tried as often as allowed (canceled, if that was asked).
pub async fn reap(state: &AppState) -> Result<()> {
    let now = state.clock.now();
    let (requeued, spent) = state
        .db
        .call(move |c| {
            let mut st = c.prepare("UPDATE jobs SET status = 'queued', lease_owner = NULL, lease_expires_at = NULL WHERE status = 'running' AND lease_expires_at < ?1 AND attempts < max_attempts AND cancel_requested = 0 RETURNING id")?;
            let requeued = st.query_map([now], |r| r.get::<_, String>(0))?.collect::<rusqlite::Result<Vec<_>>>()?;
            let mut st = c.prepare("SELECT id, cancel_requested FROM jobs WHERE status = 'running' AND lease_expires_at < ?1")?;
            let spent = st.query_map([now], |r| Ok((r.get::<_, String>(0)?, r.get::<_, i64>(1)? != 0)))?.collect::<rusqlite::Result<Vec<_>>>()?;
            Ok((requeued, spent))
        })
        .await?;
    for id in requeued {
        tracing::warn!(job = %id, "lease expired; job re-queued");
    }
    for (id, canceled) in spent {
        let f = if canceled {
            Final::stopped("canceled", "canceled at the client's request")
        } else {
            Final::stopped("failed", "the job's lease expired on its last attempt (its worker stopped without finishing it)")
        };
        finalize(state, &id, None, f).await?;
    }
    Ok(())
}

/// How a job ended, for `finalize`.
pub struct Final {
    pub status: &'static str,
    pub exit_code: Option<i32>,
    pub error: Option<String>,
    pub report: Option<Value>,
    pub cpu_seconds: f64,
}

impl Final {
    pub fn stopped(status: &'static str, why: &str) -> Final {
        Final { status, exit_code: None, error: Some(why.into()), report: None, cpu_seconds: 0.0 }
    }
}

/// Why a running job was stopped.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Stop {
    Canceled,
    TimedOut,
    Shutdown,
    LostLease,
}

/// Run a leased job to its end.
async fn run_job(state: &AppState, job: JobRow, owner: &str, shutdown: &mut watch::Receiver<bool>) {
    tracing::info!(job = %job.id, attempt = job.attempts, "job started");
    let paths = JobPaths::new(&state.cfg.data_dir, &job.account_id, &job.id);
    let (stop, exit) = match start(state, &paths) {
        Ok((pid, done)) => supervise(state, &job, owner, pid, done, shutdown).await,
        Err(e) => {
            tracing::error!(job = %job.id, error = %format!("{e:#}"), "rfpii job could not be started");
            (None, Exit { code: -1, cpu_seconds: 0.0, error: Some(format!("rfpii job could not be started: {e:#}")) })
        }
    };
    let outcome = match stop {
        Some(Stop::LostLease) => {
            tracing::warn!(job = %job.id, "lease lost to another worker; this run is abandoned");
            return;
        }
        Some(Stop::Shutdown) => {
            requeue(state, &job.id, owner).await;
            return;
        }
        Some(Stop::Canceled) => Final { status: "canceled", exit_code: None, error: Some("canceled at the client's request".into()), report: None, cpu_seconds: exit.cpu_seconds },
        Some(Stop::TimedOut) => Final {
            status: "failed",
            exit_code: None,
            error: Some(format!("the job ran longer than its limit of {} s", state.cfg.job_timeout_secs)),
            report: None,
            cpu_seconds: exit.cpu_seconds,
        },
        None => ended(&paths, exit),
    };
    if let Err(e) = finalize(state, &job.id, Some(owner), outcome).await {
        tracing::error!(job = %job.id, error = %format!("{e:#}"), "finishing the job failed");
    }
}

/// What the process did.
struct Exit {
    code: i32,
    cpu_seconds: f64,
    error: Option<String>,
}

/// A job whose process ended by itself: succeeded if `rfpii job` exited 0
/// (everything released) or 2 (something held) and wrote its report.
fn ended(paths: &JobPaths, exit: Exit) -> Final {
    let report: Option<Value> = std::fs::read(paths.report()).ok().and_then(|b| serde_json::from_slice(&b).ok());
    match (exit.code, report) {
        (0 | 2, Some(r)) => Final { status: "succeeded", exit_code: Some(exit.code), error: None, report: Some(r), cpu_seconds: exit.cpu_seconds },
        (code, _) => {
            let why = exit.error.unwrap_or_else(|| format!("rfpii job exited with code {code}{}", last_log_line(&paths.log()).map(|l| format!(": {l}")).unwrap_or_default()));
            Final { status: "failed", exit_code: Some(code), error: Some(why), report: None, cpu_seconds: exit.cpu_seconds }
        }
    }
}

/// The last line `rfpii` wrote to the job's log (an error message), cut
/// short; it names files by their path in the job's folder only.
fn last_log_line(log: &Path) -> Option<String> {
    let text = std::fs::read_to_string(log).ok()?;
    let line = text.lines().rev().find(|l| !l.trim().is_empty())?;
    Some(line.chars().take(300).collect())
}

/// Start `rfpii job` in its own process group, its output to the job's
/// log; a thread waits for it and reports its exit code and CPU time.
fn start(state: &AppState, paths: &JobPaths) -> Result<(i32, oneshot::Receiver<(i32, f64)>)> {
    use std::os::unix::process::CommandExt;
    let _ = storage::remove(&paths.out());
    let mut log = std::fs::OpenOptions::new().create(true).append(true).open(paths.log()).context("opening the job's log")?;
    std::io::Write::write_all(&mut log, format!("--- attempt at {}\n", crate::clock::rfc3339(state.clock.now())).as_bytes())?;
    let child = std::process::Command::new(&state.cfg.rfpii_bin)
        .arg("job")
        .arg(paths.job_file())
        .current_dir(&paths.dir)
        .env_clear()
        .envs(crate::rfpii::child_env())
        .stdin(std::process::Stdio::null())
        .stdout(log.try_clone()?)
        .stderr(log)
        .process_group(0)
        .spawn()
        .with_context(|| format!("running {}", state.cfg.rfpii_bin.display()))?;
    let pid = child.id() as i32;
    // the child is reaped by wait4 below, not by `Child::wait`
    drop(child);
    let (tx, rx) = oneshot::channel();
    std::thread::spawn(move || {
        let _ = tx.send(wait_pid(pid));
    });
    Ok((pid, rx))
}

/// Wait for `pid` with wait4: its exit code (minus the signal's number if
/// a signal ended it) and the CPU time, user and system, of it and of the
/// processes it waited for.
fn wait_pid(pid: i32) -> (i32, f64) {
    let mut status: libc::c_int = 0;
    // SAFETY: rusage is plain data; zero is a valid value of every field.
    let mut ru: libc::rusage = unsafe { std::mem::zeroed() };
    loop {
        // SAFETY: wait4 writes only to the two locals passed.
        let r = unsafe { libc::wait4(pid, &mut status, 0, &mut ru) };
        if r == pid {
            break;
        }
        if r == -1 && std::io::Error::last_os_error().raw_os_error() == Some(libc::EINTR) {
            continue;
        }
        return (-1, 0.0);
    }
    let secs = |t: libc::timeval| t.tv_sec as f64 + t.tv_usec as f64 / 1e6;
    let code = if libc::WIFEXITED(status) {
        libc::WEXITSTATUS(status)
    } else if libc::WIFSIGNALED(status) {
        -libc::WTERMSIG(status)
    } else {
        -1
    };
    (code, secs(ru.ru_utime) + secs(ru.ru_stime))
}

/// Send `sig` to the job's process group.
fn signal_group(pid: i32, sig: libc::c_int) {
    // SAFETY: kill only sends a signal; -pid names the job's own group.
    unsafe {
        libc::kill(-pid, sig);
    }
}

/// Watch a running job: renew its lease, and stop it when it is canceled,
/// runs out of time, loses its lease or the server shuts down (after the
/// grace period). Why it was stopped, if it was, and how it exited.
async fn supervise(state: &AppState, job: &JobRow, owner: &str, pid: i32, mut done: oneshot::Receiver<(i32, f64)>, shutdown: &mut watch::Receiver<bool>) -> (Option<Stop>, Exit) {
    let started = Instant::now();
    let limit = Duration::from_secs(state.cfg.job_timeout_secs);
    let mut beat = tokio::time::interval(Duration::from_secs((state.cfg.lease_secs / 3).max(1) as u64));
    let mut poll = tokio::time::interval(Duration::from_millis(state.cfg.poll_interval_ms));
    let mut stop: Option<Stop> = None;
    let mut termed: Option<Instant> = None;
    let mut grace_ends: Option<Instant> = None;
    let exit = loop {
        tokio::select! {
            r = &mut done => {
                let (code, cpu) = r.unwrap_or((-1, 0.0));
                break Exit { code, cpu_seconds: cpu, error: None };
            }
            _ = beat.tick() => {
                if stop.is_none() && !heartbeat(state, &job.id, owner).await {
                    stop = Some(Stop::LostLease);
                }
            }
            _ = poll.tick() => {
                if stop.is_none() && cancel_requested(state, &job.id).await {
                    stop = Some(Stop::Canceled);
                }
                if stop.is_none() && started.elapsed() > limit {
                    stop = Some(Stop::TimedOut);
                }
            }
            _ = shutdown.changed(), if grace_ends.is_none() => {
                grace_ends = Some(Instant::now() + Duration::from_secs(state.cfg.shutdown_grace_secs));
            }
        }
        if stop.is_none() && grace_ends.is_some_and(|g| Instant::now() >= g) {
            stop = Some(Stop::Shutdown);
        }
        match (stop, termed) {
            (Some(s), None) => {
                tracing::info!(job = %job.id, reason = ?s, "stopping rfpii job");
                signal_group(pid, libc::SIGTERM);
                termed = Some(Instant::now());
            }
            (Some(_), Some(t)) if t.elapsed() > KILL_GRACE => signal_group(pid, libc::SIGKILL),
            _ => {}
        }
    };
    if stop.is_some() {
        // anything of the group still running goes too
        signal_group(pid, libc::SIGKILL);
    }
    (stop, exit)
}

/// Renew the lease; false if this worker no longer holds it.
async fn heartbeat(state: &AppState, job: &str, owner: &str) -> bool {
    let (now, secs, job, owner) = (state.clock.now(), state.cfg.lease_secs, job.to_string(), owner.to_string());
    state
        .db
        .call(move |c| Ok(c.execute("UPDATE jobs SET lease_expires_at = ?3 + ?4 WHERE id = ?1 AND lease_owner = ?2 AND status = 'running'", params![job, owner, now, secs])? == 1))
        .await
        .unwrap_or(true)
}

async fn cancel_requested(state: &AppState, job: &str) -> bool {
    let job = job.to_string();
    state.db.call(move |c| Ok(c.query_row("SELECT cancel_requested FROM jobs WHERE id = ?1", [job], |r| r.get::<_, i64>(0))? != 0)).await.unwrap_or(false)
}

/// Put a job stopped by shutdown back in the queue; the attempt doesn't
/// count.
async fn requeue(state: &AppState, job: &str, owner: &str) {
    let (job, owner) = (job.to_string(), owner.to_string());
    let r = state
        .db
        .call(move |c| {
            c.execute(
                "UPDATE jobs SET status = 'queued', lease_owner = NULL, lease_expires_at = NULL, attempts = MAX(attempts - 1, 0) WHERE id = ?1 AND lease_owner = ?2 AND status = 'running'",
                params![job, owner],
            )?;
            Ok(())
        })
        .await;
    if let Err(e) = r {
        tracing::error!(error = %format!("{e:#}"), "re-queueing a job at shutdown failed");
    }
}

/// Forms a report says were released: the sum of its groups' `released`.
pub fn released_in(report: &Value) -> i64 {
    report["groups"].as_array().into_iter().flatten().map(|g| g["released"].as_i64().unwrap_or(0)).sum()
}

/// End a job: record how it ended, settle its usage, record its actual
/// cost, queue its webhook, and delete its inputs unless they are kept.
/// With `owner`, only if that worker still holds the job; without, only a
/// job no worker holds (queued, or its lease ran out). Ending a job
/// that already ended changes nothing. The job as it ended, if it ended now.
pub async fn finalize(state: &AppState, job_id: &str, owner: Option<&str>, f: Final) -> Result<Option<JobRow>> {
    let now = state.clock.now();
    let (id, owner) = (job_id.to_string(), owner.map(String::from));
    let prices = state.prices.clone();
    let retention = state.cfg.retention_output_days;
    let ended = state
        .db
        .call(move |c| {
            let tx = c.transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;
            let Some(job) = jobs::load_any(&tx, &id)? else { return Ok(None) };
            let held_by_other = match &owner {
                Some(o) => job.lease_owner.as_ref() != Some(o),
                // without an owner: a queued job, or one whose lease ran out
                None => job.status == "running" && job.lease_expires_at.is_some_and(|t| t >= now),
            };
            if job.is_terminal() || held_by_other {
                return Ok(None);
            }
            let released = f.report.as_ref().map(released_in).unwrap_or(0);
            let held = f.report.as_ref().map(|r| (r["plan"]["documents"].as_i64().unwrap_or(job.forms_requested) - released).max(0));
            crate::billing::settle(&tx, &id, released, now)?;
            let uses = f.report.as_ref().map(crate::billing::model_uses).unwrap_or_default();
            let (cost, total, per_model) = crate::billing::cost(f.cpu_seconds, &uses, &prices);
            for (u, c) in uses.iter().zip(per_model) {
                tx.execute(
                    "INSERT INTO job_model_usage (job_id, host, model, requests, input_tokens, output_tokens, cost_microusd) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
                    params![id, u.host, u.model, u.requests, u.input_tokens, u.output_tokens, c],
                )?;
            }
            tx.execute(
                "UPDATE jobs SET status = ?2, exit_code = ?3, error = ?4, forms_released = ?5, forms_held = ?6, cost_json = ?7, cost_microusd = ?8,
                        finished_at = ?9, outputs_expire_at = ?10, lease_owner = NULL, lease_expires_at = NULL WHERE id = ?1",
                params![id, f.status, f.exit_code, f.error, released, held, cost.to_string(), total, now, now + retention * 86_400],
            )?;
            let job = jobs::load_any(&tx, &id)?.context("the job vanished")?;
            queue_webhook(&tx, now, &job)?;
            tx.commit()?;
            Ok(Some(job))
        })
        .await?;
    if let Some(job) = &ended {
        tracing::info!(job = %job.id, status = %job.status, forms_released = job.forms_released.unwrap_or(0), "job ended");
        keep_transfers(state, job);
        if state.cfg.retention_input_days == 0 {
            delete_inputs(state, job, "deleted when the job ended (least data: originals are not kept)").await?;
        }
    }
    Ok(ended)
}

/// Queue the job's event for its webhook (the job's, else the account's),
/// if there is one and the account has a signing secret.
fn queue_webhook(tx: &rusqlite::Transaction, now: i64, job: &JobRow) -> Result<()> {
    let Some(account) = crate::accounts::load(tx, &job.account_id)? else { return Ok(()) };
    let Some(url) = job.webhook_url.clone().or(account.webhook_url.clone()) else { return Ok(()) };
    if account.webhook_secret.is_none() {
        tracing::warn!(job = %job.id, "webhook not sent: the account has no signing secret");
        return Ok(());
    }
    crate::webhooks::enqueue(tx, now, &job.account_id, Some(&job.id), &format!("job.{}", job.status), &url, &job.resource())?;
    Ok(())
}

/// The transfers record (`transfers.jsonl`) is part of every job's
/// outputs: `rfpii job` copies it to `public/` when it releases anything;
/// otherwise it is kept from `staging/` before that is deleted.
fn keep_transfers(state: &AppState, job: &JobRow) {
    let p = JobPaths::new(&state.cfg.data_dir, &job.account_id, &job.id);
    let staged = p.out().join("staging").join("transfers.jsonl");
    if !p.public().join("transfers.jsonl").exists() && staged.exists() {
        let r = std::fs::create_dir_all(p.records()).and_then(|_| std::fs::copy(&staged, p.records().join("transfers.jsonl")));
        if let Err(e) = r {
            tracing::error!(job = %job.id, error = %e, "keeping the transfers record failed");
        }
    }
}

/// Delete a job's inputs: its `in/` folder, what of its output was not
/// released, and the uploaded files it used. Each deletion is recorded.
pub async fn delete_inputs(state: &AppState, job: &JobRow, reason: &str) -> Result<()> {
    if job.inputs_deleted_at.is_some() {
        return Ok(());
    }
    let p = JobPaths::new(&state.cfg.data_dir, &job.account_id, &job.id);
    // what is there now is recorded, by whichever process records first
    let mut there: Vec<String> = Vec::new();
    for d in std::iter::once(p.inputs()).chain(UNRELEASED.iter().map(|u| p.out().join(u))) {
        if storage::remove(&d)? {
            there.push(d.strip_prefix(&state.cfg.data_dir).unwrap_or(&d).display().to_string());
        }
    }
    let (now, job_id, account, reason, data_dir) = (state.clock.now(), job.id.clone(), job.account_id.clone(), reason.to_string(), state.cfg.data_dir.clone());
    state
        .db
        .call(move |c| {
            let tx = c.transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;
            if tx.execute("UPDATE jobs SET inputs_deleted_at = ?2 WHERE id = ?1 AND inputs_deleted_at IS NULL", params![job_id, now])? == 0 {
                return Ok(());
            }
            for what in &there {
                crate::audit::deletion(&tx, now, &account, Some(&job_id), None, what, &reason)?;
            }
            let mut st = tx.prepare("SELECT f.id FROM job_files jf JOIN files f ON f.id = jf.file_id WHERE jf.job_id = ?1 AND f.deleted_at IS NULL")?;
            let files = st.query_map([&job_id], |r| r.get::<_, String>(0))?.collect::<rusqlite::Result<Vec<_>>>()?;
            drop(st);
            for f in files {
                storage::remove(&storage::file_path(&data_dir, &account, &f))?;
                tx.execute("UPDATE files SET deleted_at = ?2, deleted_why = ?3 WHERE id = ?1", params![f, now, reason])?;
                crate::audit::deletion(&tx, now, &account, Some(&job_id), Some(&f), &format!("files/{account}/{f}"), &reason)?;
            }
            tx.commit()?;
            Ok(())
        })
        .await
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn released_forms_are_summed_over_the_groups() {
        let r = json!({"groups": [{"released": 2}, {"released": 0}, {"released": 5}]});
        assert_eq!(released_in(&r), 7);
        assert_eq!(released_in(&json!({})), 0);
    }

    #[test]
    #[allow(clippy::zombie_processes)] // the children are reaped by wait_pid
    fn wait4_reports_the_exit_code_and_the_cpu_time() {
        use std::os::unix::process::CommandExt;
        let child = std::process::Command::new("sh").args(["-c", "i=0; while [ $i -lt 20000 ]; do i=$((i+1)); done; exit 3"]).process_group(0).spawn().unwrap();
        let (code, cpu) = wait_pid(child.id() as i32);
        assert_eq!(code, 3);
        assert!(cpu > 0.0, "{cpu}");
        let child = std::process::Command::new("sleep").arg("30").process_group(0).spawn().unwrap();
        signal_group(child.id() as i32, libc::SIGTERM);
        assert_eq!(wait_pid(child.id() as i32).0, -libc::SIGTERM);
    }
}
