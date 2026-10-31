//! Retention: the sweeper deletes what is past its time and records each
//! deletion. Uploaded files no job used go after RETENTION_UNUSED_FILE_HOURS;
//! a job's inputs when it ends (or RETENTION_INPUT_DAYS later); its outputs
//! RETENTION_OUTPUT_DAYS after it ends. Idempotency keys go after 24 hours.
//! Billing records are kept.

use anyhow::Result;
use rusqlite::params;

use crate::jobs::{self, JobRow};
use crate::server::AppState;
use crate::storage;
use crate::worker::JobPaths;

/// What one sweep deleted.
#[derive(Debug, Default, PartialEq)]
pub struct Swept {
    pub unused_files: usize,
    pub job_inputs: usize,
    pub job_outputs: usize,
    pub idempotency_keys: usize,
}

/// One sweep.
pub async fn sweep_once(state: &AppState) -> Result<Swept> {
    let mut s = Swept { unused_files: unused_files(state).await?, ..Default::default() };
    let now = state.clock.now();
    let input_age = state.cfg.retention_input_days * 86_400;
    let due_inputs: Vec<JobRow> = state
        .db
        .call(move |c| {
            let mut st = c.prepare(&format!("SELECT {} FROM jobs WHERE finished_at IS NOT NULL AND inputs_deleted_at IS NULL AND finished_at + ?2 <= ?1", jobs::COLUMNS))?;
            Ok(st.query_map(params![now, input_age], jobs::from_row)?.collect::<rusqlite::Result<Vec<_>>>()?)
        })
        .await?;
    for job in due_inputs {
        crate::worker::delete_inputs(state, &job, &format!("kept {} day(s) after the job ended, as configured", state.cfg.retention_input_days)).await?;
        s.job_inputs += 1;
    }
    let due_outputs: Vec<JobRow> = state
        .db
        .call(move |c| {
            let mut st =
                c.prepare(&format!("SELECT {} FROM jobs WHERE outputs_expire_at IS NOT NULL AND outputs_expire_at <= ?1 AND outputs_deleted_at IS NULL", jobs::COLUMNS))?;
            Ok(st.query_map([now], jobs::from_row)?.collect::<rusqlite::Result<Vec<_>>>()?)
        })
        .await?;
    for job in due_outputs {
        delete_job_folder(state, &job, &format!("outputs kept {} day(s) after the job ended, as configured", state.cfg.retention_output_days)).await?;
        s.job_outputs += 1;
    }
    s.idempotency_keys = state.db.call(move |c| Ok(c.execute("DELETE FROM idempotency_keys WHERE created_at <= ?1", [now - jobs::IDEMPOTENCY_SECS])?)).await?;
    Ok(s)
}

/// Delete uploaded files that no job used and that are older than allowed.
async fn unused_files(state: &AppState) -> Result<usize> {
    let now = state.clock.now();
    let cutoff = now - state.cfg.retention_unused_file_hours * 3600;
    let data_dir = state.cfg.data_dir.clone();
    state
        .db
        .call(move |c| {
            let tx = c.transaction()?;
            let mut st = tx.prepare("SELECT id, account_id FROM files WHERE deleted_at IS NULL AND created_at <= ?1 AND id NOT IN (SELECT file_id FROM job_files)")?;
            let due = st.query_map([cutoff], |r| Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?)))?.collect::<rusqlite::Result<Vec<_>>>()?;
            drop(st);
            let why = "uploaded and not used by any job within the retention period";
            for (id, account) in &due {
                storage::remove(&storage::file_path(&data_dir, account, id))?;
                tx.execute("UPDATE files SET deleted_at = ?2, deleted_why = ?3 WHERE id = ?1", params![id, now, why])?;
                crate::audit::deletion(&tx, now, account, None, Some(id), &format!("files/{account}/{id}"), why)?;
            }
            tx.commit()?;
            Ok(due.len())
        })
        .await
}

/// Delete everything of a job on disk (inputs first, recorded as such),
/// keeping its row: the billing record stays.
pub async fn delete_job_folder(state: &AppState, job: &JobRow, reason: &str) -> Result<()> {
    crate::worker::delete_inputs(state, job, reason).await?;
    let p = JobPaths::new(&state.cfg.data_dir, &job.account_id, &job.id);
    let removed = storage::remove(&p.dir)?;
    let (now, id, account, reason) = (state.clock.now(), job.id.clone(), job.account_id.clone(), reason.to_string());
    let what = format!("jobs/{account}/{id}");
    state
        .db
        .call(move |c| {
            if removed {
                crate::audit::deletion(c, now, &account, Some(&id), None, &what, &reason)?;
            }
            c.execute("UPDATE jobs SET outputs_deleted_at = COALESCE(outputs_deleted_at, ?2) WHERE id = ?1", params![id, now])?;
            Ok(())
        })
        .await
}

/// Sweep every `SWEEP_INTERVAL_SECS` until shutdown.
pub async fn run(state: AppState, mut shutdown: tokio::sync::watch::Receiver<bool>) {
    let every = std::time::Duration::from_secs(state.cfg.sweep_interval_secs.max(1));
    while !*shutdown.borrow() {
        match sweep_once(&state).await {
            Ok(s) if s != Swept::default() => tracing::info!(?s, "retention sweep"),
            Ok(_) => {}
            Err(e) => tracing::error!(error = %format!("{e:#}"), "retention sweep failed"),
        }
        tokio::select! {
            _ = tokio::time::sleep(every) => {}
            _ = shutdown.changed() => {}
        }
    }
}
