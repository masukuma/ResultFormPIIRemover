//! Commands of the harness (HARNESS.md §6): the tools a job runs, each
//! taking and giving JSON, usable without a model.

use anyhow::Result;
use clap::Args;
use serde_json::{json, Value};
use std::path::{Path, PathBuf};

use rfpii::job::{self, diagnose, inventory, stats, view};

#[derive(Args)]
pub struct InventoryArgs {
    /// Files, folders and zip archives
    #[arg(required = true)]
    pub inputs: Vec<String>,
    /// Print the whole inventory as JSON
    #[arg(long)]
    pub json: bool,
}

pub fn inventory(a: InventoryArgs) -> Result<i32> {
    let inv = inventory::inventory(&a.inputs)?;
    if a.json {
        println!("{}", serde_json::to_string_pretty(&inv)?);
    } else {
        print!("{}", inventory::summary(&inv));
    }
    Ok(0)
}

#[derive(Args)]
pub struct StatsArgs {
    /// A run's output folder (manifests)
    pub out: PathBuf,
    /// Its review folder, for the documents held and why
    #[arg(long)]
    pub review: Option<PathBuf>,
    /// Rank every published page by its risk signals (HARNESS.md §8)
    #[arg(long)]
    pub risk: bool,
    /// Print everything as JSON
    #[arg(long)]
    pub json: bool,
}

pub fn stats(a: StatsArgs) -> Result<i32> {
    let s = stats::stats(&a.out, a.review.as_deref(), a.risk)?;
    if a.json {
        println!("{}", serde_json::to_string_pretty(&s)?);
    } else {
        print!("{}", stats::summary(&s));
    }
    Ok(0)
}

/// The templates of `dir`, or none.
fn templates(dir: Option<&Path>) -> Result<Vec<rfpii::templates::Template>> {
    dir.map(job::templates).transpose().map(Option::unwrap_or_default)
}

/// Write a view where asked and say what it is: its file, SHA-256, size
/// and where it lies on the page.
fn show(v: &view::View, out: Option<&Path>) -> Result<i32> {
    if let Some(p) = out {
        std::fs::write(p, &v.encoded)?;
    }
    let rec = json!({"image": out.map(|p| p.to_string_lossy().into_owned()), "sha256": v.sha256,
                     "width": opencv::prelude::MatTraitConst::cols(&v.image), "height": opencv::prelude::MatTraitConst::rows(&v.image), "origin": v.origin});
    println!("{}", rec);
    Ok(0)
}

/// "png" for a .png file, else "jpg".
fn format_of(out: Option<&Path>) -> &'static str {
    match out.and_then(|p| p.extension()).map(|e| e.to_string_lossy().to_lowercase()) {
        Some(e) if e == "png" => "png",
        _ => "jpg",
    }
}

#[derive(Args)]
pub struct CropArgs {
    /// The published page's manifest (NAME.manifest.json, its PDF beside it)
    pub manifest: PathBuf,
    /// What to show: page, declaration (from the declaration down) or agents
    #[arg(long, default_value = "page")]
    pub region: String,
    /// Page of the document, from 1
    #[arg(long, default_value_t = 1)]
    pub page: usize,
    /// Templates, for where the regions are (without: the whole page)
    #[arg(long)]
    pub templates: Option<PathBuf>,
    /// Where to write the image (PNG for .png, else JPEG)
    #[arg(long)]
    pub out: Option<PathBuf>,
    /// Long side of the image, px at most
    #[arg(long, default_value_t = 1800)]
    pub max_side: i32,
    /// The publisher's public key, to check the manifest's signature too
    #[arg(long)]
    pub pubkey: Option<PathBuf>,
}

pub fn crop(a: CropArgs) -> Result<i32> {
    let ts = templates(a.templates.as_deref())?;
    let v = view::crop(&view::CropArgs {
        manifest: &a.manifest,
        region: view::Region::parse(&a.region)?,
        page: a.page,
        templates: &ts,
        max_side: a.max_side,
        pubkey: a.pubkey.as_deref(),
        within: None,
        format: format_of(a.out.as_deref()),
    })?;
    show(&v, a.out.as_deref())
}

#[derive(Args)]
pub struct HeldViewArgs {
    /// The held page's source file (as the job's review report names it)
    pub source: String,
    #[arg(long, default_value_t = 1)]
    pub page: usize,
    /// triage (masked, 512 px long) or locate (the page at full resolution)
    #[arg(long)]
    pub mode: String,
    /// The job's folder: its plan must say `held_pages: cloud`
    #[arg(long)]
    pub job: PathBuf,
    /// Templates (default: the job's)
    #[arg(long)]
    pub templates: Option<PathBuf>,
    /// Where to write the image (PNG for .png, else JPEG)
    #[arg(long)]
    pub out: Option<PathBuf>,
}

pub fn held_view(a: HeldViewArgs) -> Result<i32> {
    let mode = view::Mode::parse(&a.mode)?;
    view::held_access(&a.job, &a.source, a.page, mode)?;
    let plan = rfpii::manifest::read_json(&a.job.join("plan.json"))?;
    let dir = a.templates.clone().or_else(|| plan["templates"].as_str().map(PathBuf::from));
    let ms = rfpii::masking::MaskSettings::from_record(&plan["policy"]["settings"]);
    let v = view::held_view(&a.source, a.page, mode, &templates(dir.as_deref())?, &ms, format_of(a.out.as_deref()))?;
    show(&v, a.out.as_deref())
}

#[derive(Args)]
pub struct DiagnoseArgs {
    /// The held page's source file
    pub source: String,
    #[arg(long, default_value_t = 1)]
    pub page: usize,
    /// Templates, to read the page as `redact` did
    #[arg(long)]
    pub templates: Option<PathBuf>,
    /// Print the diagnostics as JSON
    #[arg(long)]
    pub json: bool,
}

pub fn diagnose(a: DiagnoseArgs) -> Result<i32> {
    let d = diagnose::diagnose(&a.source, a.page, &templates(a.templates.as_deref())?, &Default::default())?;
    if a.json {
        println!("{}", serde_json::to_string_pretty(&d)?);
    } else {
        let r = &d["read"];
        let words: Vec<&str> = d["ocr"]["vocabulary"].as_array().into_iter().flatten().filter_map(Value::as_str).collect();
        println!("page {} of {}: contrast {}, ink {}", d["page"], d["pages"], d["contrast"], d["ink"]);
        println!("read: template {:?}, ok {}, {}", r["template"].as_str().unwrap_or(""), r["ok"], r["reason"].as_str().unwrap_or(""));
        println!("ocr: turn {}, skew {}, confidence {}; printed words: {}", d["ocr"]["turn"], d["ocr"]["skew"], d["ocr"]["mean_confidence"], words.join(", "));
    }
    Ok(0)
}

#[derive(Args)]
pub struct JobArgs {
    /// The job file (JOB.yaml: inputs, instructions, outputs, sign_key,
    /// controller, dpia, templates, held_pages, models, budget)
    pub job: PathBuf,
}

pub fn run_job(a: JobArgs) -> Result<i32> {
    let code = job::run::run(&a.job, &std::env::current_exe()?)?;
    let spec = job::spec::JobFile::load(&a.job)?;
    println!("report: {}", spec.outputs.join("report.md").display());
    Ok(code)
}

#[derive(Args)]
pub struct IncidentArgs {
    /// The job's folder
    pub job: PathBuf,
    /// The released form found leaking (in the job's public/)
    pub pdf: PathBuf,
    /// What was found showing
    #[arg(long)]
    pub reason: String,
    /// What kind of data showed
    #[arg(long, value_parser = ["signature", "id_number", "phone_number", "other"])]
    pub kind: String,
    /// When the leak became known (default: now), e.g. 2026-10-31T09:00:00+00:00
    #[arg(long)]
    pub aware_at: Option<String>,
    /// Whether the IEBC portal showed the same form then (reg. 37(3)(a))
    #[arg(long)]
    pub portal_online: Option<bool>,
}

/// A leak on a released form: withdrawn, and the incident recorded with
/// its deadlines and a notice drafted.
pub fn incident(a: IncidentArgs) -> Result<i32> {
    let plan = rfpii::manifest::read_json(&a.job.join("plan.json"))?;
    let manifest = rfpii::manifest::read_json(&a.pdf.with_extension("manifest.json"))?;
    let ledger = a.job.join("public").join("ledger.jsonl");
    crate::cli_legal::withdraw(crate::cli_legal::WithdrawArgs {
        pdf: a.pdf.clone(),
        reason: a.reason.clone(),
        sign_key: PathBuf::from(plan["sign_key"].as_str().unwrap_or("")),
        ledger: Some(ledger.clone()),
    })?;
    let digest = rfpii::manifest::digest(&manifest);
    let (_, _, entries) = rfpii::manifest::ledger_check(&ledger)?;
    let withdrawal = entries.iter().rev().find(|e| e["type"] == "withdrawal" && e["manifest_sha256"].as_str() == Some(digest.as_str())).cloned().unwrap_or_default();
    let now = rfpii::process::utc_now();
    let r = job::incident::record(&job::incident::Incident {
        job: &a.job,
        manifest: &manifest,
        reason: &a.reason,
        kind: &a.kind,
        aware_at: a.aware_at.as_deref().unwrap_or(&now),
        portal_online: a.portal_online,
        withdrawal: &withdrawal,
    })?;
    println!("incident recorded: notify the Data Commissioner by {}{}", r["deadlines"]["commissioner"].as_str().unwrap_or(""), r["deadlines"]["controller"].as_str().map(|d| format!("; as processor, the controller by {}", d)).unwrap_or_default());
    if let Some(g) = r["reopened"].as_str() {
        println!("group {} re-opened: its third leak of this kind since release", g);
    }
    Ok(0)
}
