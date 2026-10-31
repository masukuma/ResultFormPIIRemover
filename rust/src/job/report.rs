//! The job's report (HARNESS.md §10): `report.json` in fixed fields, and
//! `report.md`, which opens with the policy in plain words.

use anyhow::Result;
use serde_json::{json, Value};
use std::collections::BTreeMap;

use super::run::Job;
use super::{sample, stats};

/// What the report says, as JSON.
pub fn figures(job: &Job) -> Value {
    let published = job.published();
    let held = job.held_units();
    let mut per_file: BTreeMap<String, (usize, usize)> = BTreeMap::new();
    for u in published.values().flatten() {
        per_file.entry(u.rel.clone()).or_default().0 += 1;
    }
    for (u, _) in &held {
        per_file.entry(u.rel.clone()).or_default().1 += 1;
    }
    let mut reasons: BTreeMap<String, usize> = BTreeMap::new();
    for (_, r) in &held {
        *reasons.entry(crate::annex::normalise_reason(r)).or_insert(0) += 1;
    }
    let mut categories: BTreeMap<String, usize> = BTreeMap::new();
    for h in &job.held {
        *categories.entry(h.category.clone()).or_insert(0) += 1;
    }
    let groups: Vec<Value> = job.plan.groups.iter().map(|g| {
        let st = job.groups.get(&g.id).cloned().unwrap_or_default();
        let units = published.get(&g.id).cloned().unwrap_or_default();
        let checks: Vec<_> = units.iter().map(|u| job.checks.get(&u.name()).cloned().unwrap_or_default()).collect();
        let reviewed = |why: &str| checks.iter().filter(|c| c.reviewed.as_deref() == Some(why)).count();
        json!({
            "id": g.id, "kind": g.kind, "per_page": g.per_page, "documents": g.units.len(),
            "published": units.len(), "held": held.iter().filter(|(u, _)| u.group == g.id).count(), "released": st.released,
            "verified": checks.iter().filter(|c| c.verified == Some(true)).count(),
            "verify_failed": checks.iter().filter(|c| c.verified == Some(false)).count(),
            "reruns": checks.iter().filter(|c| c.rerun.is_some()).count(),
            "reruns_failed": checks.iter().filter(|c| c.rerun == Some(false)).count(),
            "at_risk": checks.iter().filter(|c| c.score >= stats::RISK_THRESHOLD).count(),
            "reviewed": {"risk": reviewed("risk"), "sample": reviewed("sample"), "pilot": reviewed("pilot"), "located": reviewed("located"), "every page": reviewed("every page")},
            "findings": checks.iter().map(|c| c.findings.len()).sum::<usize>(),
            "quarantined": checks.iter().filter(|c| c.quarantined.is_some()).count(),
            "leak_rate": {"population": st.population, "sampled": st.sampled, "leaks": st.sample_leaks,
                          "bound": sample::bound(st.sampled, st.sample_leaks).filter(|_| st.sampled > 0), "met": sample::bounded(st.population, st.sampled, st.sample_leaks)},
            "patterns": st.patterns,
            "review_only": st.review_only,
            "gate": st.gate.iter().map(|(n, ok, why)| json!({"condition": n, "ok": ok, "why": why})).collect::<Vec<_>>(),
        })
    }).collect();
    let events = job.journal.events();
    let models: Vec<&Value> = events.iter().filter(|e| e["kind"] == "model").collect();
    let defects: Vec<String> = crate::annex::files_under(&job.dir.join("defects"), ".json").iter().map(|p| p.to_string_lossy().into_owned()).collect();
    json!({
        "schema": "rfpii/job-report/1",
        "started": job.started,
        "finished": crate::process::utc_now(),
        "stopped": job.stopped,
        "policy": {"words": job.policy.words(), "flags": job.policy.flags(), "raised": job.policy.raised, "refused": job.policy.refused, "unsupported": job.policy.unsupported},
        "plan": {"groups": job.plan.groups.len(), "documents": job.plan.units().count(), "pages": job.plan.pages, "workers": job.plan.workers, "reader": job.plan.reader,
                 "hours": job.plan.hours, "disk_bytes": job.plan.disk_bytes, "left_out": job.plan.left_out},
        "files": per_file.into_iter().map(|(f, (p, h))| json!({"file": f, "published": p, "held": h})).collect::<Vec<_>>(),
        "groups": groups,
        "held": {"documents": held.len(), "reasons": reasons, "categories": categories,
                 "pages": job.held.iter().map(|h| json!({"document": h.unit.name(), "reason": h.reason, "category": h.category, "action": h.action, "images_sent": h.sent})).collect::<Vec<_>>()},
        "held_sent_to_cloud": job.held.iter().filter(|h| !h.sent.is_empty()).map(|h| json!({"document": h.unit.name(), "images": h.sent})).collect::<Vec<_>>(),
        "defects": defects,
        "batch_root": job.batch_root,
        "model_requests": models.len(),
        "journal_events": events.len(),
        "left_for_people": job.held.iter().filter(|h| h.action == "people").map(|h| h.unit.name()).collect::<Vec<_>>(),
    })
}

fn notes(title: &str, list: &Value) -> String {
    let items: Vec<String> = list.as_array().into_iter().flatten().map(|n| format!("- \"{}\": {}\n", n["instruction"].as_str().unwrap_or(""), n["settled"].as_str().unwrap_or(""))).collect();
    if items.is_empty() { String::new() } else { format!("\n{}\n\n{}", title, items.concat()) }
}

/// The report as Markdown.
pub fn markdown(f: &Value) -> String {
    let n = |v: &Value| v.as_u64().unwrap_or(0);
    let mut s = String::from("# Redaction job: report\n\n");
    if let Some(why) = f["stopped"].as_str() {
        s += &format!("**The job stopped before the run:** {}.\n\n", why);
    }
    s += "## Policy\n\nThe instructions were compiled into this policy, which ran as compiled:\n\n";
    for w in f["policy"]["words"].as_array().into_iter().flatten() {
        s += &format!("- {}\n", w.as_str().unwrap_or(""));
    }
    let flags: Vec<&str> = f["policy"]["flags"].as_array().into_iter().flatten().filter_map(Value::as_str).collect();
    s += &format!("\nFlags: {}\n", if flags.is_empty() { "none (the defaults)".to_string() } else { format!("`{}`", flags.join(" ")) });
    s += &notes("Unclear points, settled by the reading that hides more:", &f["policy"]["raised"]);
    s += &notes("Instructions below the redactor's floor, not followed (the forms are masked as usual):", &f["policy"]["refused"]);
    s += &notes("Instructions asking more than the redactor can do (nothing from the job is released):", &f["policy"]["unsupported"]);
    let p = &f["plan"];
    s += &format!("\n## Plan\n\n{} documents ({} pages) in {} group(s); {} worker(s); about {:.1} hours and {:.2} GB. Reader: {}.\n",
        n(&p["documents"]), n(&p["pages"]), n(&p["groups"]), n(&p["workers"]), p["hours"].as_f64().unwrap_or(0.0), p["disk_bytes"].as_f64().unwrap_or(0.0) / 1e9, p["reader"].as_str().unwrap_or(""));
    for l in p["left_out"].as_array().into_iter().flatten() {
        s += &format!("- left out: {}: {}\n", l["path"].as_str().unwrap_or(""), l["why"].as_str().unwrap_or(""));
    }
    s += "\n## Groups\n\n| Group | Documents | Published | Held | Released | Verified | Re-runs | At risk | Reviewed (risk, sample) | Leak-rate bound |\n|---|---|---|---|---|---|---|---|---|---|\n";
    for g in f["groups"].as_array().into_iter().flatten() {
        let lr = &g["leak_rate"];
        let bound = match lr["bound"].as_f64() {
            Some(b) => format!("{:.2}% ({} of {}, {} leak(s)){}", b * 100.0, n(&lr["sampled"]), n(&lr["population"]), n(&lr["leaks"]), if lr["met"] == true { "" } else { ", not met" }),
            None if lr["met"] == true => "met".into(),
            None => "not met: no review".into(),
        };
        s += &format!("| {} | {} | {} | {} | {} | {} | {} ({} failed) | {} | {}, {} | {} |\n", g["id"].as_str().unwrap_or(""), n(&g["documents"]), n(&g["published"]), n(&g["held"]), n(&g["released"]),
            n(&g["verified"]), n(&g["reruns"]), n(&g["reruns_failed"]), n(&g["at_risk"]), n(&g["reviewed"]["risk"]), n(&g["reviewed"]["sample"]), bound);
    }
    s += "\n## Release gate\n\n";
    for g in f["groups"].as_array().into_iter().flatten() {
        let failed: Vec<String> = g["gate"].as_array().into_iter().flatten().filter(|c| c["ok"] == false).map(|c| format!("  - {}: {}\n", c["condition"].as_str().unwrap_or(""), c["why"].as_str().unwrap_or(""))).collect();
        s += &if failed.is_empty() && n(&g["released"]) > 0 { format!("- {}: released\n", g["id"].as_str().unwrap_or("")) } else if failed.is_empty() { format!("- {}: nothing to release\n", g["id"].as_str().unwrap_or("")) } else { format!("- {}: held in staging\n{}", g["id"].as_str().unwrap_or(""), failed.concat()) };
    }
    if let Some(r) = f["batch_root"].as_str() {
        s += &format!("\nBatch root of what was released: `{}` (in `public/batches/`). Announce it when the batch is published.\n", r);
    }
    s += &format!("\n## Held pages\n\n{} document(s) held.\n\n", n(&f["held"]["documents"]));
    s += "| Why held | Documents |\n|---|---|\n";
    for (k, v) in f["held"]["reasons"].as_object().into_iter().flatten() {
        s += &format!("| {} | {} |\n", k, v);
    }
    s += "\n| Triage category | Documents |\n|---|---|\n";
    for (k, v) in f["held"]["categories"].as_object().into_iter().flatten() {
        s += &format!("| {} | {} |\n", k, v);
    }
    let sent = f["held_sent_to_cloud"].as_array().map(|a| a.len()).unwrap_or(0);
    s += &format!("\nHeld pages sent to the cloud model: {}.\n", sent);
    for h in f["held_sent_to_cloud"].as_array().into_iter().flatten() {
        let views: Vec<String> = h["images"].as_array().into_iter().flatten().map(|i| format!("{} ({})", i["mode"].as_str().unwrap_or(""), i["size"].as_str().unwrap_or(""))).collect();
        s += &format!("- {}: {}\n", h["document"].as_str().unwrap_or(""), views.join(", "));
    }
    s += "\n## Files\n\n| File | Published | Held |\n|---|---|---|\n";
    for x in f["files"].as_array().into_iter().flatten() {
        s += &format!("| {} | {} | {} |\n", x["file"].as_str().unwrap_or(""), n(&x["published"]), n(&x["held"]));
    }
    s += &format!("\n## Defects\n\n{}\n", if f["defects"].as_array().is_none_or(|d| d.is_empty()) { "None.".to_string() } else { f["defects"].as_array().unwrap().iter().filter_map(Value::as_str).map(|d| format!("- `{}`", d)).collect::<Vec<_>>().join("\n") });
    s += &format!("\n## Use\n\nStarted {}, finished {}; {} model request(s); {} journal events.\n", f["started"].as_str().unwrap_or(""), f["finished"].as_str().unwrap_or(""), n(&f["model_requests"]), n(&f["journal_events"]));
    let people: Vec<&str> = f["left_for_people"].as_array().into_iter().flatten().filter_map(Value::as_str).collect();
    s += &format!("\n## Left for people\n\n{}\n", if people.is_empty() { "Nothing.".to_string() } else { people.iter().map(|p| format!("- {}", p)).collect::<Vec<_>>().join("\n") });
    s
}

/// Phase 9: `report.json` and `report.md`.
pub fn write(job: &Job) -> Result<()> {
    let f = figures(job);
    std::fs::write(job.dir.join("report.json"), serde_json::to_string_pretty(&f)?)?;
    std::fs::write(job.dir.join("report.md"), markdown(&f))?;
    job.journal.tool("report", "report", json!({}), &f)?;
    Ok(())
}
