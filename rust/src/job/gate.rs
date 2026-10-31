//! The release gate (HARNESS.md §9), checked by code, per group, against
//! what the tools recorded; and the release of the groups that pass: their
//! PDFs and manifests copied to `public/`, each entered in its ledger, and
//! a signed batch root over what was released.

use anyhow::Result;
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::path::Path;

use super::plan::Unit;
use super::run::Job;
use super::{sample, stats};
use crate::manifest as mf;

/// Groups re-opened by incidents after release (`job-incident`): group ->
/// why.
pub fn reopened(dir: &Path) -> BTreeMap<String, String> {
    mf::read_json(&dir.join("incidents").join("reopened.json")).ok().and_then(|v| serde_json::from_value(v).ok()).unwrap_or_default()
}

/// Whether `settings` (a manifest's) records what `want` (the policy's) does.
fn same_settings(settings: &Value, want: &Value) -> bool {
    want.as_object().is_some_and(|w| w.iter().all(|(k, v)| settings.get(k) == Some(v)))
}

/// Names of `units` for which `bad` holds, a few of them, for a reason.
fn some(units: &[Unit], bad: impl Fn(&Unit) -> bool) -> (bool, String) {
    let names: Vec<String> = units.iter().filter(|u| bad(u)).map(Unit::name).collect();
    let more = if names.len() > 3 { format!(" and {} more", names.len() - 3) } else { String::new() };
    (names.is_empty(), format!("{}{}", names.iter().take(3).cloned().collect::<Vec<_>>().join(", "), more))
}

/// The gate's conditions for group `gid` with its staged `units`: each
/// condition, whether it holds, and why not.
pub fn conditions(job: &Job, gid: &str, units: &[Unit]) -> Vec<(String, bool, String)> {
    let check = |u: &Unit| job.checks.get(&u.name()).cloned().unwrap_or_default();
    let state = job.groups.get(gid).cloned().unwrap_or_default();
    let mut out: Vec<(String, bool, String)> = Vec::new();
    let mut add = |name: &str, (ok, why): (bool, String)| out.push((name.to_string(), ok, if ok { String::new() } else { why }));

    let (ok, who) = some(units, |u| check(u).verified != Some(true));
    add("1. every page passes verify", (ok, format!("not verified, or failing: {}", who)));
    let reruns = units.iter().filter(|u| check(u).rerun.is_some()).count();
    let (ok, who) = some(units, |u| check(u).rerun == Some(false));
    let enough = reruns >= sample::rerun_size(units.len());
    add("2. every verify --rerun passes", (ok && enough, if ok { format!("{} of {} re-runs made", reruns, sample::rerun_size(units.len())) } else { format!("re-run differs: {}", who) }));
    let (ok, who) = some(units, |u| check(u).score >= stats::RISK_THRESHOLD && check(u).reviewed.is_none());
    add("3. every page above the risk threshold was reviewed", (ok, format!("not reviewed: {}", who)));
    let (fixed, who) = some(units, |u| check(u).findings.iter().any(|f| f["fixed"] != true));
    let bounded = sample::bounded(state.population, state.sampled, state.sample_leaks);
    add("4. the random sample bounds the leak rate below 0.5%, every leak fixed", (bounded && fixed, if !fixed { format!("leaks not fixed: {}", who) } else { format!("{} of {} pages reviewed at random, {} leak(s)", state.sampled, state.population, state.sample_leaks) }));
    let open: Vec<String> = state.patterns.iter().filter(|p| p["open"] == true).map(|p| p["kind"].as_str().unwrap_or("").to_string()).collect();
    let why = state.review_only.clone().or_else(|| reopened(&job.dir).get(gid).cloned()).or_else(|| (!open.is_empty()).then(|| format!("open pattern(s): {}", open.join(", "))));
    add("5. no pattern touching the group is open", (why.is_none(), why.unwrap_or_default()));
    let want = job.policy.settings();
    let (ok, who) = some(units, |u| mf::read_json(&u.at(&job.staging(), ".manifest.json")).map(|m| !same_settings(&m["settings"], &want)).unwrap_or(true));
    let flags = job.policy.flags();
    let calls_ok = job.journal.events().iter().filter(|e| e["tool"] == "redact").all(|e| {
        let argv: Vec<&str> = e["args"]["argv"].as_array().into_iter().flatten().filter_map(Value::as_str).collect();
        flags.iter().all(|f| argv.contains(&f.as_str()))
    });
    add("6. the policy run is the policy compiled", (ok && calls_ok, if ok { "a redact call lacks the policy's flags".into() } else { format!("settings differ: {}", who) }));
    let unsupported: Vec<String> = job.policy.unsupported.iter().map(|n| n.instruction.clone()).collect();
    add("7. every instruction was met", (unsupported.is_empty(), format!("asked more than the redactor can do: {}", unsupported.join("; "))));
    let dpia = match &job.spec.dpia {
        Some(d) => d.bars(&job.started),
        None => Some("the job file names no DPIA".into()),
    };
    add("DPIA: approved, or submitted 60 days before", (dpia.is_none(), dpia.unwrap_or_default()));
    let controller = crate::legal::Controller::load(&job.spec.controller).err().map(|e| format!("{:#}", e));
    add("the controller record loads", (controller.is_none(), controller.unwrap_or_default()));
    out
}

/// Every group with staged pages was released whole.
pub fn all_released(job: &Job) -> bool {
    job.published().iter().all(|(g, units)| job.groups.get(g).is_some_and(|s| s.released == units.len()))
}

/// Copy one staged unit to `public/`, entering it in the public ledger
/// unless an earlier release did; its manifest's digest.
fn publish(job: &Job, u: &Unit) -> Result<String> {
    let public = job.dir.join("public");
    let m = mf::read_json(&u.at(&job.staging(), ".manifest.json"))?;
    let digest = mf::digest(&m);
    let target = u.at(&public, ".manifest.json");
    let ledger = public.join("ledger.jsonl");
    let entered = ledger.exists() && mf::ledger_check(&ledger)?.2.iter().any(|e| e["manifest_sha256"].as_str() == Some(digest.as_str()) && e.get("type").is_none());
    std::fs::create_dir_all(target.parent().unwrap())?;
    std::fs::copy(u.at(&job.staging(), ".pdf"), u.at(&public, ".pdf"))?;
    std::fs::copy(u.at(&job.staging(), ".manifest.json"), &target)?;
    if !entered {
        mf::ledger_append(&ledger, &digest, m["source"]["sha256"].as_str().unwrap_or(""), m["output"]["pdf_sha256"].as_str().unwrap_or(""), &u.name())?;
    }
    Ok(digest)
}

/// Phase 8: the gate for each group, and the release of those that pass;
/// a signed batch root over what was released in this run.
pub fn release(job: &mut Job) -> Result<()> {
    let published = job.published();
    let mut batch: Vec<Value> = Vec::new();
    for (gid, units) in &published {
        let conds = conditions(job, gid, units);
        let pass = conds.iter().all(|(_, ok, _)| *ok) && job.stopped.is_none();
        let failed: Vec<String> = conds.iter().filter(|(_, ok, _)| !ok).map(|(n, _, why)| format!("{}: {}", n, why)).collect();
        job.journal.decision("release", if pass { "release" } else { "hold" }, json!(failed), json!({"group": gid, "documents": units.len()}))?;
        if pass {
            for u in units {
                let digest = publish(job, u)?;
                batch.push(json!({"name": u.name(), "manifest_sha256": digest}));
            }
        }
        let state = job.groups.entry(gid.clone()).or_default();
        state.gate = conds;
        state.released = if pass { units.len() } else { 0 };
    }
    if batch.is_empty() {
        return Ok(());
    }
    let public = job.dir.join("public");
    std::fs::copy(&job.pubkey, public.join("signing_key.pub.pem"))?;
    if job.staging().join("transfers.jsonl").exists() {
        std::fs::copy(job.staging().join("transfers.jsonl"), public.join("transfers.jsonl"))?;
    }
    let (_, _, entries) = mf::ledger_check(&public.join("ledger.jsonl"))?;
    let leaves: Vec<String> = batch.iter().map(|b| b["manifest_sha256"].as_str().unwrap_or("").to_string()).collect();
    let created = crate::process::utc_now();
    let b = mf::sign(&json!({"schema": "rfpii/batch/1", "created_at": created, "count": batch.len(), "merkle_root": mf::merkle_root(&leaves)?,
                             "ledger_head": entries.last().map(|e| e["entry_hash"].clone()).unwrap_or(Value::Null), "documents": batch}), job.key());
    std::fs::create_dir_all(public.join("batches"))?;
    std::fs::write(public.join("batches").join(format!("batch-{}.json", created.replace(':', ""))), crate::pyjson::pretty(&b, 1, true))?;
    job.batch_root = b["merkle_root"].as_str().map(String::from);
    job.journal.tool("release", "batch_root", json!({"documents": leaves.len()}), &b)?;
    Ok(())
}
