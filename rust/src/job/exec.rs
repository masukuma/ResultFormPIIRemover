//! Running `rfpii redact` for a job (HARNESS.md §4.4): one process per
//! source file, or per batch of single-document files, so that one file's
//! memory doesn't add to another's and a crash costs one process. Code, not
//! a model, watches each process: a heartbeat every minute in the journal;
//! no progress for ten minutes, and it is killed and resumed; killed for
//! memory, and it is resumed with one worker fewer.

use anyhow::{bail, Context, Result};
use serde_json::json;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use super::journal::Journal;

/// Seconds between heartbeats.
const HEARTBEAT: u64 = 60;
/// Seconds without a page done before a process is taken as stuck.
const STALL: u64 = 600;
/// Times a process is started for the same work.
const ATTEMPTS: usize = 3;

/// What every `redact` call of a job shares.
pub struct Redact<'a> {
    pub exe: &'a Path,
    /// flags every call carries: folders, key, controller, policy
    pub common: Vec<String>,
    pub workers: usize,
    pub journal: &'a Journal,
    /// the review folder's `report.jsonl`, whose lines count pages done
    pub report: PathBuf,
    /// where each process's output is kept
    pub log: PathBuf,
    /// free disk below which the run stops (bytes), and where to look
    pub disk_floor: Option<(f64, PathBuf)>,
}

fn lines(p: &Path) -> usize {
    std::fs::read_to_string(p).map(|t| t.lines().count()).unwrap_or(0)
}

impl Redact<'_> {
    /// Run `redact` on `inputs` with `extra` flags, supervised; its exit
    /// code (0 all published, 2 some held).
    pub fn run(&self, phase: &str, inputs: &[String], extra: &[String]) -> Result<i32> {
        let mut workers = self.workers;
        for attempt in 1..=ATTEMPTS {
            let mut args: Vec<String> = vec!["redact".into()];
            args.extend(inputs.iter().cloned());
            args.extend(self.common.iter().cloned());
            args.extend(["--workers".into(), workers.to_string()]);
            args.extend(extra.iter().cloned());
            let (code, why) = self.supervise(phase, &args)?;
            self.journal.tool(phase, "redact", json!({"argv": args, "attempt": attempt}), &json!({"exit": code, "stopped": why}))?;
            match (code, why.as_deref()) {
                (Some(c), None) => return Ok(c),
                (_, Some("out of memory")) => workers = (workers - 1).max(1),
                (_, Some(w)) if w.starts_with("disk") => bail!("{}", w),
                _ => {}
            }
        }
        bail!("redact did not finish after {} attempts: {}", ATTEMPTS, inputs.join(", "))
    }

    /// One process: its exit code, or why it was stopped.
    fn supervise(&self, phase: &str, args: &[String]) -> Result<(Option<i32>, Option<String>)> {
        if let Some(d) = self.log.parent() {
            std::fs::create_dir_all(d)?;
        }
        let log = std::fs::OpenOptions::new().create(true).append(true).open(&self.log)?;
        let mut child = std::process::Command::new(self.exe).args(args).stdout(log.try_clone()?).stderr(log).spawn().with_context(|| format!("starting {}", self.exe.display()))?;
        let t0 = Instant::now();
        let (start, mut last_done, mut last_change, mut last_beat) = (lines(&self.report), lines(&self.report), Instant::now(), Instant::now());
        loop {
            if let Some(st) = child.try_wait()? {
                use std::os::unix::process::ExitStatusExt;
                return Ok(match (st.code(), st.signal()) {
                    (Some(c), _) => (Some(c), None),
                    (None, Some(9)) => (None, Some("out of memory".into())),
                    (None, s) => (None, Some(format!("killed by signal {:?}", s))),
                });
            }
            std::thread::sleep(Duration::from_millis(250));
            let done = lines(&self.report);
            if done != last_done {
                (last_done, last_change) = (done, Instant::now());
            }
            if last_beat.elapsed() >= Duration::from_secs(HEARTBEAT) {
                last_beat = Instant::now();
                let free = super::plan::free_disk(self.disk_floor.as_ref().map(|d| d.1.as_path()).unwrap_or(Path::new(".")));
                let secs = t0.elapsed().as_secs_f64();
                self.journal.write(phase, "heartbeat", json!({"done": done - start, "seconds": secs.round(), "rate": ((done - start) as f64 / secs * 100.0).round() / 100.0,
                                                              "memory_available": super::plan::memory_available(), "free_disk": free}))?;
                if let (Some((floor, _)), Some(f)) = (&self.disk_floor, free)
                    && (f as f64) < *floor
                {
                    let _ = child.kill();
                    let _ = child.wait();
                    return Ok((None, Some(format!("disk: {:.1} GB free, below the {:.1} GB the rest of the run needs", f as f64 / 1e9, floor / 1e9))));
                }
            }
            if last_change.elapsed() >= Duration::from_secs(STALL) {
                let _ = child.kill();
                let _ = child.wait();
                return Ok((None, Some(format!("no page done in {} s", STALL))));
            }
        }
    }
}
