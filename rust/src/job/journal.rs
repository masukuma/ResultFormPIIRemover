//! The journal (HARNESS.md §10): one JSON line per event, in order. Tool
//! calls with their arguments and the hashes of their results; model calls
//! with the model, host, prompt hash, the hashes of the images sent and the
//! parsed reply; decisions with their reasons; heartbeats. From it and the
//! manifests, anyone can see what every model saw and decided.

use anyhow::{anyhow, Result};
use serde_json::{json, Value};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

pub struct Journal {
    path: PathBuf,
    seq: Mutex<u64>,
}

impl Journal {
    /// The journal at `path`, continued if it exists.
    pub fn open(path: &Path) -> Result<Journal> {
        let n = std::fs::read_to_string(path).map(|t| t.lines().count() as u64).unwrap_or(0);
        Ok(Journal { path: path.to_path_buf(), seq: Mutex::new(n) })
    }

    /// Append an event of `kind` in `phase`; its id ("e12").
    pub fn write(&self, phase: &str, kind: &str, fields: Value) -> Result<String> {
        let mut seq = self.seq.lock().map_err(|_| anyhow!("journal lock poisoned"))?;
        let id = format!("e{}", *seq);
        let mut e = json!({"id": id, "at": crate::process::utc_now(), "phase": phase, "kind": kind});
        for (k, v) in fields.as_object().cloned().unwrap_or_default() {
            e[k] = v;
        }
        let mut f = std::fs::OpenOptions::new().create(true).append(true).open(&self.path)?;
        writeln!(f, "{}", serde_json::to_string(&e)?)?;
        *seq += 1;
        Ok(id)
    }

    /// A tool call: its arguments, and the SHA-256 of its result.
    pub fn tool(&self, phase: &str, tool: &str, args: Value, result: &Value) -> Result<String> {
        self.write(phase, "tool", json!({"tool": tool, "args": args, "result_sha256": crate::proof::sha256(serde_json::to_string(result)?.as_bytes())}))
    }

    /// A decision: what was chosen, and why.
    pub fn decision(&self, phase: &str, choice: &str, reasons: Value, extra: Value) -> Result<String> {
        let mut f = json!({"choice": choice, "reasons": reasons});
        for (k, v) in extra.as_object().cloned().unwrap_or_default() {
            f[k] = v;
        }
        self.write(phase, "decision", f)
    }

    /// Every event so far.
    pub fn events(&self) -> Vec<Value> {
        std::fs::read_to_string(&self.path).map(|t| t.lines().filter_map(|l| serde_json::from_str(l).ok()).collect()).unwrap_or_default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn events_are_numbered_in_order_and_continued() {
        let d = crate::testutil::tmpdir("journal");
        let j = Journal::open(&d.join("journal.jsonl")).unwrap();
        assert_eq!(j.tool("intake", "inventory", json!({"inputs": ["a"]}), &json!({"files": 1})).unwrap(), "e0");
        assert_eq!(j.decision("plan", "per_page", json!(["a scan"]), json!({"group": "scan"})).unwrap(), "e1");
        let j = Journal::open(&d.join("journal.jsonl")).unwrap();
        assert_eq!(j.write("run", "heartbeat", json!({"done": 3})).unwrap(), "e2");
        let e = j.events();
        assert!(e.len() == 3 && e[1]["choice"] == "per_page" && e[1]["group"] == "scan" && e[0]["result_sha256"].as_str().unwrap().len() == 64);
    }
}
