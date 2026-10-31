//! A leak found on a released form (HARNESS.md, `legal/C-recommendations.md`
//! §2.11, B2): once the form is withdrawn (`rfpii withdraw`), the incident
//! is recorded under the job's `incidents/`: when it became known, whose
//! data the form holds by class (from its masks' role tags, never the
//! values), the deadlines of Data Protection Act s.43 (the Commissioner
//! within 72 hours; the controller within 48 hours if the operator is its
//! processor), and a notice drafted for a person to check and send. Three
//! leaks of one kind in a group after release re-open the group.

use anyhow::{Context, Result};
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::path::Path;

use super::journal::Journal;

/// Hours to notify the Data Commissioner (s.43(1)(a)), and for a processor
/// to notify the controller (s.43(3)).
pub const COMMISSIONER_HOURS: i64 = 72;
pub const CONTROLLER_HOURS: i64 = 48;

/// Whose data a manifest's form holds, by class: masks of fields with
/// writing on them (not those found empty), counted by role.
pub fn affected(m: &Value) -> BTreeMap<String, usize> {
    let mut by: BTreeMap<String, usize> = BTreeMap::new();
    for p in m["pages"].as_array().into_iter().flatten() {
        for mk in p["masks"].as_array().into_iter().flatten().filter(|mk| mk["mode"] != "empty") {
            *by.entry(mk["role"].as_str().unwrap_or("unknown").to_string()).or_insert(0) += 1;
        }
    }
    by
}

/// The group of the job's plan the published form `file` (a manifest's
/// `output.file`) belongs to.
fn group_of(plan: &Value, file: &str) -> Option<String> {
    let units: Vec<super::plan::Unit> = plan["plan"]["groups"].as_array()?.iter().flat_map(|g| g["units"].as_array().cloned().unwrap_or_default()).filter_map(|u| serde_json::from_value(u).ok()).collect();
    units.iter().find(|u| crate::process::output_rel(&u.rel, u.page, ".pdf").to_string_lossy() == file).map(|u| u.group.clone())
}

pub struct Incident<'a> {
    pub job: &'a Path,
    /// the withdrawn form's manifest, as it was published
    pub manifest: &'a Value,
    pub reason: &'a str,
    /// what showed: signature, id_number, phone_number, other
    pub kind: &'a str,
    pub aware_at: &'a str,
    /// whether the IEBC portal showed the same form when the leak became
    /// known (reg. 37(3)(a)), if checked
    pub portal_online: Option<bool>,
    /// the ledger entry of the withdrawal
    pub withdrawal: &'a Value,
}

/// The notice, drafted for a person to check, complete and send.
fn notice(r: &Value, controller: &Value) -> String {
    let roles: Vec<String> = r["affected"].as_object().into_iter().flatten().map(|(k, v)| format!("{} field(s) of {}s", v, k)).collect();
    format!(
        "# DRAFT: notification of a personal data breach (Data Protection Act 2019, s.43; reg. 38)\n\n\
         To check, complete and send by a person. Nothing here is sent automatically.\n\n\
         - Controller: {}\n- Data protection officer: {}\n- Became aware: {}\n- Notify the Data Commissioner by: {}\n{}\
         - The IEBC portal showed the same form at the time: {}\n\n\
         ## Nature of the breach\n\nA redacted copy of an election result form ({}) was published showing personal data its redaction should have hidden: {} ({}).\n\n\
         ## Data subjects and data concerned\n\nThe form holds the data of: {}. Classes: names (published by law), ID or passport numbers, phone numbers, signatures.\n\n\
         ## Chronology\n\nPublished as manifest `{}`; withdrawn in ledger entry {} at {}.\n\n\
         ## Measures taken\n\nThe form was taken down from the published folder and its withdrawal recorded in the signed ledger. It is to be republished corrected, naming the withdrawn manifest (`rfpii redact --supersedes`).\n\n\
         ## Assessment under s.43(1)\n\nWhether there is a real risk of harm, and whether the data was publicly available (reg. 37(3)(a)), is the controller's to record here.\n",
        controller["controller"]["name"].as_str().unwrap_or("?"),
        controller["dpo"]["name"].as_str().unwrap_or("not given"),
        r["aware_at"].as_str().unwrap_or(""),
        r["deadlines"]["commissioner"].as_str().unwrap_or(""),
        r["deadlines"]["controller"].as_str().map(|d| format!("- As processor, notify the controller by: {}\n", d)).unwrap_or_default(),
        r["portal_online"].as_bool().map(|b| if b { "yes" } else { "no" }).unwrap_or("not checked"),
        r["document"].as_str().unwrap_or(""),
        r["kind"].as_str().unwrap_or(""),
        r["reason"].as_str().unwrap_or(""),
        if roles.is_empty() { "no one recorded".to_string() } else { roles.join(", ") },
        r["manifest_sha256"].as_str().unwrap_or(""),
        r["withdrawal"]["seq"],
        r["withdrawal"]["at"].as_str().unwrap_or(""),
    )
}

/// Record an incident: `incidents/<time>-<name>.json` and the notice draft
/// beside it, a journal entry, and the group re-opened at its third leak
/// of one kind. The record.
pub fn record(i: &Incident) -> Result<Value> {
    let plan = crate::manifest::read_json(&i.job.join("plan.json")).context("not a job folder (no plan.json)")?;
    let controller_path = plan["controller"].as_str().unwrap_or("");
    let controller = crate::manifest::read_json(Path::new(controller_path)).unwrap_or(Value::Null);
    let aware = super::epoch_of(i.aware_at).with_context(|| format!("--aware-at {:?} is not a time such as 2026-10-31T09:00:00+00:00", i.aware_at))?;
    let processor = controller["operator_role"] == "processor";
    let file = i.manifest["output"]["file"].as_str().unwrap_or("").to_string();
    let group = group_of(&plan, &file);
    let mut r = json!({
        "aware_at": crate::process::utc_at(aware),
        "document": file,
        "group": group,
        "kind": i.kind,
        "reason": i.reason,
        "manifest_sha256": crate::manifest::digest(i.manifest),
        "affected": affected(i.manifest),
        "deadlines": {"commissioner": crate::process::utc_at(aware + COMMISSIONER_HOURS * 3600),
                      "controller": processor.then(|| crate::process::utc_at(aware + CONTROLLER_HOURS * 3600))},
        "portal_online": i.portal_online,
        "withdrawal": {"seq": i.withdrawal["seq"], "at": i.withdrawal["at"]},
    });
    let dir = i.job.join("incidents");
    std::fs::create_dir_all(&dir)?;
    let stem = format!("{}-{}", r["aware_at"].as_str().unwrap_or("").replace(':', ""), Path::new(&file).file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default());
    std::fs::write(dir.join(format!("{}.json", stem)), serde_json::to_string_pretty(&r)?)?;
    std::fs::write(dir.join(format!("{}.md", stem)), notice(&r, &controller))?;
    let journal = Journal::open(&i.job.join("journal.jsonl"))?;
    journal.write("incident", "incident", json!({"record": r}))?;
    // the third leak of one kind in a group after release re-opens it
    let same = crate::annex::files_under(&dir, ".json").iter().filter_map(|p| crate::manifest::read_json(p).ok()).filter(|x| x["group"] == r["group"] && x["kind"] == i.kind && x.get("aware_at").is_some()).count();
    if let (Some(g), true) = (group, same >= super::sample::PATTERN) {
        let mut reopened = super::gate::reopened(i.job);
        reopened.insert(g.clone(), format!("{} {} leaks found after release", same, i.kind));
        std::fs::write(dir.join("reopened.json"), serde_json::to_string_pretty(&reopened)?)?;
        journal.decision("incident", "reopen group", json!([format!("{} leaks of kind {} after release: a pattern", same, i.kind)]), json!({"group": g}))?;
        r["reopened"] = json!(g);
    }
    Ok(r)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn those_affected_are_counted_by_role_from_the_masks_with_writing() {
        let m = json!({"pages": [{"masks": [{"role": "agent", "mode": "partial"}, {"role": "agent", "mode": "empty"}, {"role": "officer", "mode": "signature"}, {"mode": "full"}]}]});
        assert_eq!(affected(&m), [("agent".to_string(), 1), ("officer".to_string(), 1), ("unknown".to_string(), 1)].into_iter().collect());
    }
}
