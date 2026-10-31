//! Keeping files no longer than their schedule says (Data Protection Act
//! s.39; the Elections (General) Regulations reg. 93 keep election
//! materials three years): `retention.yaml` names each class of file, where
//! it is and how long it is kept; `rfpii retain` reports what is due and,
//! with `--apply`, deletes it and records the disposal in the ledger, by
//! hash only (`legal/C-recommendations.md` §2.10).

use anyhow::{bail, Context, Result};
use serde_json::{json, Value};
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime};

/// One class of file and how long it is kept.
#[derive(Debug, Clone)]
pub struct Class {
    pub name: String,
    pub path: PathBuf,
    /// days from when a file was last written; None: kept indefinitely
    pub keep_days: Option<u64>,
}

pub struct Policy {
    pub classes: Vec<Class>,
    pub ledger: Option<PathBuf>,
    pub text: String,
}

/// Read a retention schedule. Relative paths are taken from its folder.
///
/// ```yaml
/// ledger: public/ledger.jsonl
/// classes:
///   - {name: originals, path: /data/forms, keep_days: 1095}
///   - {name: held pages, path: review, keep_days: 90}
///   - {name: published, path: public}          # no keep_days: kept
/// ```
pub fn load(path: &Path) -> Result<Policy> {
    let text = std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    let y: serde_yaml::Value = serde_yaml::from_str(&text).with_context(|| format!("{} is not YAML", path.display()))?;
    let base = path.parent().unwrap_or(Path::new("."));
    let at = |p: &str| if Path::new(p).is_absolute() { PathBuf::from(p) } else { base.join(p) };
    let mut classes = Vec::new();
    for c in y["classes"].as_sequence().cloned().unwrap_or_default() {
        let (Some(name), Some(p)) = (c["name"].as_str(), c["path"].as_str()) else { bail!("{}: each class needs a name and a path", path.display()) };
        classes.push(Class { name: name.into(), path: at(p), keep_days: c["keep_days"].as_u64() });
    }
    if classes.is_empty() {
        bail!("{}: no classes", path.display());
    }
    Ok(Policy { classes, ledger: y["ledger"].as_str().map(at), text })
}

/// The files of each class past their time: (class, path, age in days).
pub fn due(p: &Policy, now: SystemTime) -> Vec<(String, PathBuf, u64)> {
    let mut out = Vec::new();
    for c in &p.classes {
        let Some(days) = c.keep_days else { continue };
        let files = if c.path.is_file() { vec![c.path.clone()] } else { crate::annex::files_under(&c.path, "") };
        for f in files {
            let age = std::fs::metadata(&f).and_then(|m| m.modified()).ok().and_then(|t| now.duration_since(t).ok()).unwrap_or(Duration::ZERO);
            let d = age.as_secs() / 86_400;
            if d >= days {
                out.push((c.name.clone(), f, d));
            }
        }
    }
    out
}

/// Delete what is due, and record each class's disposal in the ledger
/// (signed by `key`): the files' names relative to their class's folder and
/// their SHA-256, never their contents. Returns the files deleted.
pub fn apply(p: &Policy, now: SystemTime, key: Option<&ed25519_dalek::SigningKey>) -> Result<Vec<PathBuf>> {
    let list = due(p, now);
    if list.is_empty() {
        return Ok(vec![]);
    }
    let (Some(ledger), Some(key)) = (&p.ledger, key) else { bail!("deleting needs the schedule's ledger and --sign-key, to record what was deleted") };
    let mut done = Vec::new();
    for c in &p.classes {
        let files: Vec<&(String, PathBuf, u64)> = list.iter().filter(|(n, _, _)| *n == c.name).collect();
        if files.is_empty() {
            continue;
        }
        let mut items = Vec::new();
        for (_, f, age) in &files {
            let sha = crate::proof::sha256(&std::fs::read(f)?);
            items.push(json!({"file": f.strip_prefix(&c.path).unwrap_or(f).to_string_lossy(), "sha256": sha, "age_days": age}));
        }
        crate::manifest::ledger_event(ledger, "disposal", json!({"class": c.name, "keep_days": c.keep_days, "files": items}), key)?;
        for (_, f, _) in files {
            std::fs::remove_file(f)?;
            done.push(f.clone());
        }
    }
    Ok(done)
}

/// What `retain` reports: per class, files kept and due.
pub fn audit(p: &Policy, now: SystemTime) -> Value {
    let due = due(p, now);
    let rows: Vec<Value> = p
        .classes
        .iter()
        .map(|c| {
            let n = if c.path.is_file() { 1 } else { crate::annex::files_under(&c.path, "").len() };
            json!({"class": c.name, "path": c.path.to_string_lossy(), "keep_days": c.keep_days, "files": n,
                   "due": due.iter().filter(|(k, _, _)| *k == c.name).count()})
        })
        .collect();
    json!(rows)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn files_past_their_time_are_deleted_and_recorded_by_hash() {
        let d = crate::testutil::tmpdir("retain");
        std::fs::create_dir_all(d.join("review/x")).unwrap();
        std::fs::create_dir_all(d.join("public")).unwrap();
        std::fs::write(d.join("review/x/held.json"), b"{\"reason\": \"page 1: ...\"}").unwrap();
        std::fs::write(d.join("public/form.pdf"), b"%PDF").unwrap();
        std::fs::write(d.join("retention.yaml"), "ledger: public/ledger.jsonl\nclasses:\n  - {name: held pages, path: review, keep_days: 90}\n  - {name: published, path: public}\n").unwrap();
        let p = load(&d.join("retention.yaml")).unwrap();
        let soon = SystemTime::now() + Duration::from_secs(30 * 86_400);
        let later = SystemTime::now() + Duration::from_secs(91 * 86_400);
        assert!(due(&p, soon).is_empty());
        let list = due(&p, later);
        assert_eq!(list.len(), 1, "published files are kept: {:?}", list);
        assert!(apply(&p, later, None).is_err(), "deleting needs a key to record it");
        let (k, _) = crate::manifest::keygen(&d.join("keys"), None).unwrap();
        let key = crate::manifest::load_private(&k).unwrap();
        let gone = apply(&p, later, Some(&key)).unwrap();
        assert_eq!(gone.len(), 1);
        assert!(!d.join("review/x/held.json").exists() && d.join("public/form.pdf").exists());
        let (ok, _, entries) = crate::manifest::ledger_check(&d.join("public/ledger.jsonl")).unwrap();
        assert!(ok);
        let e = &entries[0];
        assert!(e["type"] == "disposal" && e["class"] == "held pages" && e["files"][0]["file"] == "x/held.json" && e["files"][0]["sha256"].as_str().unwrap().len() == 64);
        assert!(!e.to_string().contains("reason"), "contents never recorded");
    }
}
