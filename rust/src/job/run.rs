//! `rfpii job JOB.yaml`: a whole job, in fixed phases (HARNESS.md §4):
//! intake, plan, pilot, run, QA, remediation, triage, release, report.
//! Every tool call and decision goes in the journal. Without models the
//! run is deterministic: samples are drawn from a seed the inputs give, and
//! decisions follow fixed rules. Nothing leaves `staging/` but through the
//! release gate.

use anyhow::{bail, Context, Result};
use serde::Serialize;
use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use super::exec::Redact;
use super::journal::Journal;
use super::plan::{Plan, Unit};
use super::policy::{Policy, Review};
use super::spec::JobFile;
use super::{diagnose, inventory, sample, stats};
use crate::legal::Controller;
use crate::templates::Template;

/// Units in one `redact` process, at most, when each is a whole file.
const BATCH: usize = 200;

/// A group whose pilot publishes less than this share is held for review.
pub const PILOT_PUBLISH: f64 = 0.5;

/// What QA found on one document.
#[derive(Serialize, Clone, Debug, Default)]
pub struct Check {
    /// `verify` passed (None: not verified)
    pub verified: Option<bool>,
    pub problems: Vec<String>,
    /// `verify --rerun` passed (None: not re-run)
    pub rerun: Option<bool>,
    pub score: f64,
    pub signals: Vec<String>,
    /// why it was looked at by the vision model: risk, sample, pilot...
    pub reviewed: Option<String>,
    /// confirmed findings, each with its journal id and whether it was fixed
    pub findings: Vec<Value>,
    pub quarantined: Option<String>,
}

/// What a group came to.
#[derive(Serialize, Clone, Debug, Default)]
pub struct GroupState {
    /// set: nothing of the group is released until this is understood
    pub review_only: Option<String>,
    /// documents below the risk threshold, and how many of them were
    /// reviewed at random, with the leaks found
    pub population: usize,
    pub sampled: usize,
    pub sample_leaks: usize,
    pub patterns: Vec<Value>,
    pub released: usize,
    pub gate: Vec<(String, bool, String)>,
}

/// A held page and what triage made of it.
#[derive(Serialize, Clone, Debug)]
pub struct Held {
    pub unit: Unit,
    pub reason: String,
    pub category: String,
    pub action: String,
    pub sent: Vec<Value>,
}

pub struct Job {
    pub spec: JobFile,
    pub dir: PathBuf,
    pub journal: Journal,
    exe: PathBuf,
    pub controller: Controller,
    key: ed25519_dalek::SigningKey,
    pub pubkey: PathBuf,
    pub templates: Vec<Template>,
    pub started: String,
    pub inventory: Value,
    pub policy: Policy,
    pub plan: Plan,
    pub checks: BTreeMap<String, Check>,
    pub groups: BTreeMap<String, GroupState>,
    pub held: Vec<Held>,
    pub stopped: Option<String>,
    pub batch_root: Option<String>,
}

/// The folders a job fills (HARNESS.md §1).
pub const FOLDERS: [&str; 5] = ["staging", "public", "review", "quarantine", "defects"];

impl Job {
    pub fn staging(&self) -> PathBuf {
        self.dir.join("staging")
    }

    pub fn review_dir(&self) -> PathBuf {
        self.dir.join("review")
    }

    /// The last `redact` result of each unit, by name.
    fn results(&self) -> BTreeMap<String, Value> {
        let by_key: BTreeMap<(String, Option<u64>), Value> = stats::last_results(&self.review_dir()).into_iter().map(|r| ((r["file"].as_str().unwrap_or("").to_string(), r["page"].as_u64()), r)).collect();
        self.plan.units().filter_map(|u| by_key.get(&(u.src.clone(), u.page.map(|p| p as u64))).map(|r| (u.name(), r.clone()))).collect()
    }

    /// The units published to staging (and not quarantined), by group.
    pub fn published(&self) -> BTreeMap<String, Vec<Unit>> {
        let results = self.results();
        let mut out: BTreeMap<String, Vec<Unit>> = BTreeMap::new();
        for u in self.plan.units() {
            let ok = results.get(&u.name()).is_some_and(|r| r["ok"] == true) && u.at(&self.staging(), ".manifest.json").exists();
            if ok && self.checks.get(&u.name()).is_none_or(|c| c.quarantined.is_none()) {
                out.entry(u.group.clone()).or_default().push(u.clone());
            }
        }
        out
    }

    /// The units held, with their reasons.
    pub fn held_units(&self) -> Vec<(Unit, String)> {
        let results = self.results();
        self.plan.units().filter_map(|u| results.get(&u.name()).filter(|r| r["ok"] == false).map(|r| (u.clone(), r["reason"].as_str().unwrap_or("").to_string()))).collect()
    }

    fn check(&mut self, u: &Unit) -> &mut Check {
        self.checks.entry(u.name()).or_default()
    }

    /// `redact` for `units`: per source file a process (its pages listed
    /// when page by page), whole files in batches; units already run are
    /// left out unless `again`.
    fn redact(&self, phase: &str, units: &[Unit], again: bool) -> Result<()> {
        let results = self.results();
        let todo: Vec<&Unit> = units.iter().filter(|u| again || !results.contains_key(&u.name())).collect();
        let mut by_page: BTreeMap<(String, String), Vec<usize>> = BTreeMap::new();
        let mut whole: BTreeMap<(String, String), Vec<String>> = BTreeMap::new();
        for u in &todo {
            match u.page {
                Some(k) => by_page.entry((u.src.clone(), u.base.clone())).or_default().push(k),
                None => whole.entry((u.base.clone(), u.group.clone())).or_default().push(u.src.clone()),
            }
        }
        let spots = self.dir.join("spot_masks.json");
        let r = Redact {
            exe: &self.exe,
            common: self.common_flags(spots.exists().then_some(spots.as_path())),
            workers: self.plan.workers,
            journal: &self.journal,
            report: self.review_dir().join("report.jsonl"),
            log: self.dir.join("logs").join("redact.log"),
            disk_floor: Some((todo.len() as f64 * super::plan::BYTES_PER_PAGE, self.dir.clone())),
        };
        for ((src, base), pages) in by_page {
            let extra = vec!["--base".into(), base, "--per-page".into(), "--pages".into(), super::format_pages(&pages)];
            r.run(phase, &[src], &extra)?;
        }
        for ((base, _), srcs) in whole {
            for chunk in srcs.chunks(BATCH) {
                r.run(phase, chunk, &["--base".into(), base.clone()])?;
            }
        }
        Ok(())
    }

    /// The flags every `redact` call of the job carries: its folders, key,
    /// controller, templates, the policy's flags, and spot masks if any.
    fn common_flags(&self, spots: Option<&Path>) -> Vec<String> {
        let p = |x: &Path| x.to_string_lossy().into_owned();
        let mut f: Vec<String> = vec![
            "--templates".into(), p(&self.spec.templates),
            "--out".into(), p(&self.staging()),
            "--review".into(), p(&self.review_dir()),
            "--sign-key".into(), p(&self.spec.sign_key),
            "--controller".into(), p(&self.spec.controller),
            "--resume".into(),
        ];
        match &self.spec.certificate {
            Some(c) => f.extend(["--certificate".into(), p(c)]),
            None => f.push("--no-pdf-signature".into()),
        }
        for h in &self.spec.kenya_hosts {
            f.extend(["--kenya-host".into(), h.clone()]);
        }
        f.extend(self.policy.flags());
        if let Some(s) = spots {
            f.extend(["--spot-masks".into(), p(s)]);
        }
        f
    }

    /// The source of `u` as a file `verify` can read: itself, or a zip's
    /// member written out under the job's `tmp/`.
    fn original(&self, u: &Unit) -> Result<PathBuf> {
        if !u.src.contains(crate::io::ZIP_SEP) {
            return Ok(PathBuf::from(&u.src));
        }
        let ext = Path::new(&u.rel).extension().map(|e| e.to_string_lossy().into_owned()).unwrap_or_default();
        let p = self.dir.join("tmp").join(format!("{}.{}", u.sha256, ext));
        if !p.exists() {
            std::fs::create_dir_all(p.parent().unwrap())?;
            std::fs::write(&p, crate::io::read_bytes(&u.src)?)?;
        }
        Ok(p)
    }

    /// `verify` on a staged unit against its original, and with `rerun`
    /// the redaction repeated from its manifest.
    pub fn verify(&mut self, phase: &str, u: &Unit, rerun: bool) -> Result<bool> {
        let orig = self.original(u)?;
        let ledger = self.staging().join("ledger.jsonl");
        let rep = crate::verify::verify(&crate::verify::VerifyArgs {
            redacted: &u.at(&self.staging(), ".pdf"),
            manifest: None,
            original: Some(&orig),
            pubkey: Some(&self.pubkey),
            ledger: Some(&ledger),
            overlay_dir: None,
            rerun_templates: rerun.then_some(self.spec.templates.as_path()),
            trust: &[],
        })?;
        let problems: Vec<String> = rep.checks.iter().filter(|(ok, _)| !ok).map(|(_, m)| m.clone()).collect();
        self.journal.tool(phase, "verify", json!({"document": u.name(), "rerun": rerun}), &json!({"ok": rep.ok(), "failed": problems}))?;
        let c = self.check(u);
        if rerun {
            c.rerun = Some(rep.ok());
        }
        c.verified = Some(c.verified.unwrap_or(true) && rep.ok());
        c.problems.extend(problems);
        Ok(rep.ok())
    }

    /// The risk signals of every staged page, onto the checks.
    fn rank(&mut self, phase: &str) -> Result<()> {
        let s = stats::stats(&self.staging(), Some(&self.review_dir()), true)?;
        self.journal.tool(phase, "stats", json!({"out": "staging", "review": "review", "risk": true}), &s)?;
        let by_manifest: BTreeMap<String, Unit> = self.plan.units().map(|u| (u.at(&self.staging(), ".manifest.json").to_string_lossy().into_owned(), u.clone())).collect();
        for r in s["risk"].as_array().into_iter().flatten() {
            if let Some(u) = by_manifest.get(r["manifest"].as_str().unwrap_or("")) {
                let c = self.check(u);
                c.score = c.score.max(r["score"].as_f64().unwrap_or(0.0));
                for sig in r["signals"].as_array().into_iter().flatten().filter_map(Value::as_str) {
                    if !c.signals.iter().any(|x| x == sig) {
                        c.signals.push(sig.to_string());
                    }
                }
            }
        }
        Ok(())
    }

    /// Every unit of `units` checked as QA checks (HARNESS.md §4.5):
    /// verified, the riskiest and a sample reviewed, and `rerun` re-run.
    fn qa(&mut self, phase: &str, units: &[Unit], rerun: &BTreeSet<String>) -> Result<()> {
        for u in units {
            self.verify(phase, u, rerun.contains(&u.name()))?;
        }
        Ok(())
    }

    /// The vision review of `units` would go here; without a vision model
    /// nothing is reviewed, and the report says so.
    fn review(&mut self, phase: &str, units: &[Unit], why: &str) -> Result<()> {
        if !units.is_empty() {
            self.journal.decision(phase, "not reviewed", json!(["no vision model in the job"]), json!({"documents": units.len(), "would_be": why}))?;
        }
        Ok(())
    }

    /// Phase 3: a sample of each group redacted first and checked whole.
    fn pilot(&mut self) -> Result<()> {
        let sizes = sample::pilot_sizes(&self.plan.groups.iter().map(|g| g.units.len()).collect::<Vec<_>>());
        let mut picked: Vec<Unit> = Vec::new();
        for (g, &k) in self.plan.groups.iter().zip(&sizes) {
            let items: Vec<(String, Unit)> = g.units.iter().map(|u| (u.folder.clone(), u.clone())).collect();
            picked.extend(sample::spread(&items, k, sample::seed(self.plan.seed, &format!("pilot:{}", g.id))));
        }
        self.journal.decision("pilot", "sample", json!([format!("2% of each group, at least 20, at most 500 in all: {} documents", picked.len())]), json!({"documents": picked.iter().map(Unit::name).collect::<Vec<_>>()}))?;
        let t0 = std::time::Instant::now();
        self.redact("pilot", &picked, false)?;
        let secs = t0.elapsed().as_secs_f64();
        let published = self.published();
        let names: BTreeSet<String> = picked.iter().map(Unit::name).collect();
        let pilot_pub: Vec<Unit> = published.values().flatten().filter(|u| names.contains(&u.name())).cloned().collect();
        let all: BTreeSet<String> = pilot_pub.iter().map(Unit::name).collect();
        self.qa("pilot", &pilot_pub, &all)?;
        self.rank("pilot")?;
        self.review("pilot", &pilot_pub, "pilot")?;
        // going on: a group publishing less than half its pilot is held for review
        for g in self.plan.groups.clone() {
            let tried = picked.iter().filter(|u| u.group == g.id).count();
            let ok = pilot_pub.iter().filter(|u| u.group == g.id).count();
            let rate = if tried == 0 { 1.0 } else { ok as f64 / tried as f64 };
            let state = self.groups.entry(g.id.clone()).or_default();
            if rate < PILOT_PUBLISH {
                state.review_only = Some(format!("its pilot published {} of {} documents", ok, tried));
            }
            self.journal.decision("pilot", if rate < PILOT_PUBLISH { "review only" } else { "go on" }, json!([format!("{} of {} pilot documents published", ok, tried)]), json!({"group": g.id}))?;
        }
        // estimates again, from what the pilot measured
        let bytes: u64 = pilot_pub.iter().map(|u| std::fs::metadata(u.at(&self.staging(), ".pdf")).map(|m| m.len()).unwrap_or(0)).sum();
        if !picked.is_empty() && !pilot_pub.is_empty() {
            self.plan.remeasure(secs / picked.len() as f64, bytes as f64 / pilot_pub.len() as f64, &self.dir);
            self.write_plan()?;
        }
        Ok(())
    }

    fn write_plan(&self) -> Result<()> {
        let plan = json!({
            "schema": "rfpii/job-plan/1",
            "created_at": self.started,
            "held_pages": self.spec.held_pages.name(),
            "templates": self.spec.templates,
            "controller": self.spec.controller,
            "sign_key": self.spec.sign_key,
            "policy": {"compiled": self.policy, "flags": self.policy.flags(), "settings": self.policy.settings(), "words": self.policy.words()},
            "plan": self.plan,
            "inventory": self.inventory,
        });
        std::fs::write(self.dir.join("plan.json"), serde_json::to_string_pretty(&plan)?)?;
        Ok(())
    }

    /// Phases 5 and 6: QA over everything staged, and its remediation.
    fn qa_all(&mut self) -> Result<()> {
        let published = self.published();
        self.rank("qa")?;
        for (gid, units) in &published {
            let n = units.len();
            let items: Vec<(String, Unit)> = units.iter().map(|u| (u.folder.clone(), u.clone())).collect();
            let mut rerun: BTreeSet<String> = sample::spread(&items, sample::rerun_size(n), sample::seed(self.plan.seed, &format!("rerun:{}", gid))).iter().map(Unit::name).collect();
            rerun.extend(units.iter().filter(|u| self.spot_masked(u)).map(Unit::name));
            let todo: Vec<Unit> = units.iter().filter(|u| self.checks.get(&u.name()).is_none_or(|c| c.verified.is_none() || (rerun.contains(&u.name()) && c.rerun.is_none()))).cloned().collect();
            self.qa("qa", &todo, &rerun)?;
            // the riskiest pages, then a random sample of the rest
            let risky: Vec<Unit> = units.iter().filter(|u| self.checks.get(&u.name()).is_some_and(|c| c.score >= stats::RISK_THRESHOLD)).cloned().collect();
            let rest: Vec<(String, Unit)> = items.iter().filter(|(_, u)| !risky.contains(u)).cloned().collect();
            let k = match self.policy.vision_review {
                Review::EveryPage => rest.len(),
                Review::Sample => sample::sample_size(rest.len(), 0).unwrap_or(rest.len()),
            };
            let drawn = sample::spread(&rest, k, sample::seed(self.plan.seed, &format!("review:{}", gid)));
            let state = self.groups.entry(gid.clone()).or_default();
            state.population = rest.len();
            self.review("qa", &risky, "risk")?;
            self.review("qa", &drawn, "sample")?;
        }
        Ok(())
    }

    /// Whether a unit's source has spot masks given for it.
    fn spot_masked(&self, u: &Unit) -> bool {
        let spots = self.dir.join("spot_masks.json");
        crate::process::read_spot_masks(&spots).unwrap_or_default().iter().any(|s| s["sha256"].as_str() == Some(u.sha256.as_str()) && s["page"].as_u64() == Some(u.page_no() as u64))
    }

    /// Phase 7: what each held page is (HARNESS.md §4.7), from its text
    /// diagnostics, with a category and a suggested action; a triage note
    /// per page in `review/triage/`.
    fn triage(&mut self) -> Result<()> {
        let ms = self.policy.mask();
        for (u, reason) in self.held_units() {
            let d = match diagnose::diagnose(&u.src, u.page_no(), &self.templates, &ms) {
                Ok(d) => d,
                Err(e) => json!({"error": format!("{:#}", e)}),
            };
            self.journal.tool("triage", "diagnose", json!({"source": u.src, "page": u.page_no()}), &d)?;
            let (category, action, why) = categorise(&d, &reason);
            self.journal.decision("triage", category, json!([why]), json!({"source": u.src, "page": u.page_no(), "action": action}))?;
            let note = json!({"document": u.name(), "source": u.src, "page": u.page_no(), "reason": crate::annex::normalise_reason(&reason),
                              "diagnostics": d, "category": category, "action": action, "why": why, "images_sent": []});
            let p = u.at(&self.review_dir().join("triage"), ".json");
            std::fs::create_dir_all(p.parent().unwrap())?;
            std::fs::write(&p, serde_json::to_string_pretty(&note)?)?;
            self.held.push(Held { unit: u, reason: crate::annex::normalise_reason(&reason), category: category.into(), action: action.into(), sent: vec![] });
        }
        Ok(())
    }

    /// Phases 1 to 9, from a job file.
    pub fn start(job_file: &Path, exe: &Path) -> Result<Job> {
        let spec = JobFile::load(job_file)?;
        let dir = spec.outputs.clone();
        for f in FOLDERS {
            std::fs::create_dir_all(dir.join(f))?;
        }
        let journal = Journal::open(&dir.join("journal.jsonl"))?;
        let controller = Controller::load(&spec.controller).context("the job's controller record")?;
        let key = crate::manifest::load_private(&spec.sign_key)?;
        let pubkey = dir.join("signing_key.pub.pem");
        std::fs::write(&pubkey, ed25519_dalek::pkcs8::EncodePublicKey::to_public_key_pem(&key.verifying_key(), ed25519_dalek::pkcs8::spki::der::pem::LineEnding::LF).map_err(|e| anyhow::anyhow!("{e}"))?)?;
        let templates = super::templates(&spec.templates)?;
        let started = crate::process::utc_now();
        journal.write("intake", "start", json!({"job": job_file, "held_pages": spec.held_pages.name(), "controller_sha256": controller.sha256}))?;
        // 1 intake
        let inv = inventory::inventory(&spec.inputs)?;
        journal.tool("intake", "inventory", json!({"inputs": spec.inputs}), &inv)?;
        // 2 plan: the policy, then the groups
        let policy = Policy::without_model(&spec.instructions, spec.policy.as_ref())?;
        journal.decision("plan", "policy", json!(policy.words()), json!({"policy": policy, "flags": policy.flags(), "compiled_by": "the job file's policy, or the defaults"}))?;
        let plan = Plan::new(&inv, &policy, &spec.templates, &dir);
        for g in &plan.groups {
            journal.decision("plan", if g.per_page { "page by page" } else { "whole files" }, json!([format!("{} {} documents", g.units.len(), g.kind)]), json!({"group": g.id}))?;
        }
        Ok(Job { spec, dir, journal, exe: exe.to_path_buf(), controller, key, pubkey, templates, started, inventory: inv, policy, plan, checks: BTreeMap::new(), groups: BTreeMap::new(), held: vec![], stopped: None, batch_root: None })
    }

    /// Run the phases after the plan; the exit code: 0 if everything was
    /// released, 2 if anything was held or stays in staging.
    pub fn run(&mut self) -> Result<i32> {
        self.write_plan()?;
        match self.plan.refusal(self.spec.budget.hours) {
            Some(why) => {
                self.journal.decision("plan", "stop", json!([why.clone()]), json!({}))?;
                self.stopped = Some(why);
            }
            None => {
                self.pilot()?;
                match self.plan.refusal(self.spec.budget.hours) {
                    Some(why) => {
                        self.journal.decision("pilot", "stop", json!([why.clone()]), json!({}))?;
                        self.stopped = Some(why);
                    }
                    None => {
                        let all: Vec<Unit> = self.plan.units().cloned().collect();
                        self.redact("run", &all, false)?;
                        self.qa_all()?;
                        self.triage()?;
                    }
                }
            }
        }
        super::gate::release(self)?;
        super::report::write(self)?;
        Ok(if self.stopped.is_none() && self.held.is_empty() && super::gate::all_released(self) { 0 } else { 2 })
    }

    pub fn key(&self) -> &ed25519_dalek::SigningKey {
        &self.key
    }
}

/// A held page's category and suggested action from its diagnostics, by
/// fixed rules, with the reason: blank, too faint, another form (a tally
/// sheet), turned, or not read.
pub fn categorise(d: &Value, reason: &str) -> (&'static str, &'static str, String) {
    let ink = d["ink"].as_f64().unwrap_or(1.0);
    let contrast = d["contrast"].as_f64().unwrap_or(100.0);
    let conf = d["ocr"]["mean_confidence"].as_f64();
    let words: Vec<&str> = d["ocr"]["vocabulary"].as_array().into_iter().flatten().filter_map(Value::as_str).collect();
    let turn = d["ocr"]["turn"].as_i64().unwrap_or(0);
    if d.get("error").is_some() {
        return ("unknown", "people", format!("diagnostics failed: {}", d["error"].as_str().unwrap_or("")));
    }
    if ink < 0.005 {
        return ("blank", "not_for_this_job", format!("ink covers {:.2}% of the page", ink * 100.0));
    }
    if words.contains(&"TALLY") {
        return ("tally_sheet", "not_for_this_job", "the printed word TALLY is on the page".into());
    }
    if contrast < 25.0 || conf.is_some_and(|c| c < 0.6) {
        return ("faint", "people", format!("contrast {:.1}, OCR confidence {}", contrast, conf.map(|c| format!("{:.2}", c)).unwrap_or("none".into())));
    }
    if turn != 0 {
        return ("turned", "retry", format!("its text reads at a {}° turn", turn));
    }
    if words.is_empty() {
        return ("not_a_form", "people", "none of the forms' printed words was found".into());
    }
    ("not_read", "people", format!("held: {}", crate::annex::normalise_reason(reason)))
}

/// `rfpii job JOB.yaml`.
pub fn run(job_file: &Path, exe: &Path) -> Result<i32> {
    if !job_file.exists() {
        bail!("no job file {}", job_file.display());
    }
    let mut job = Job::start(job_file, exe)?;
    job.run()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn held_pages_are_sorted_by_fixed_rules() {
        let d = |ink: f64, contrast: f64, conf: Option<f64>, words: &[&str], turn: i64| json!({"ink": ink, "contrast": contrast, "ocr": {"mean_confidence": conf, "vocabulary": words, "turn": turn}});
        assert_eq!(categorise(&d(0.001, 2.0, None, &[], 0), "").0, "blank");
        assert_eq!(categorise(&d(0.05, 50.0, Some(0.9), &["TALLY", "COUNTY"], 0), "").0, "tally_sheet");
        assert_eq!(categorise(&d(0.05, 18.0, Some(0.9), &["DECLARATION"], 0), "").0, "faint");
        assert_eq!(categorise(&d(0.05, 50.0, Some(0.9), &["DECLARATION"], 90), "").0, "turned");
        assert_eq!(categorise(&d(0.05, 50.0, Some(0.9), &[], 0), "").0, "not_a_form");
        let (c, a, why) = categorise(&d(0.05, 50.0, Some(0.9), &["DECLARATION"], 0), "page 3: leak check: agent_id: 7 in a row");
        assert!(c == "not_read" && a == "people" && why.contains("N in a row"), "{}", why);
    }
}
