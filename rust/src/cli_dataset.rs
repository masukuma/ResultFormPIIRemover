//! `rfpii dataset`: training data for the local networks (NEURAL.md,
//! nn/README.md).

use anyhow::Result;
use clap::{Args, Subcommand};
use std::path::PathBuf;

use rfpii::dataset::{check, compare, synth, teacher};

#[derive(Args)]
pub struct DatasetArgs {
    #[command(subcommand)]
    pub cmd: DatasetCmd,
}

#[derive(Subcommand)]
pub enum DatasetCmd {
    /// Synthetic forms with exact labels, drawn on the templates' references
    Synth(SynthArgs),
    /// The teacher's labels, from published manifests and their originals
    Build(BuildArgs),
    /// Check a dataset: hashes, schemas, boxes inside images, masks' sizes
    Check {
        dataset: PathBuf,
    },
    /// Count a dataset's pages, boxes, empty cells and held pages
    Stats {
        dataset: PathBuf,
        #[arg(long)]
        json: bool,
    },
    /// How close a dataset's boxes come to another's on the same pages (the
    /// teacher against synthetic truth)
    Compare {
        truth: PathBuf,
        other: PathBuf,
        /// IoU at or above which two boxes match
        #[arg(long, default_value_t = 0.5)]
        iou: f64,
        #[arg(long)]
        json: bool,
    },
}

fn workers() -> usize {
    std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1)
}

#[derive(Args)]
pub struct SynthArgs {
    /// Template directory (each template's reference.png is drawn on)
    #[arg(long, default_value = "templates")]
    templates: PathBuf,
    /// Dataset folder to write (new, or with --append an existing one)
    #[arg(long)]
    out: PathBuf,
    #[arg(long, default_value_t = 100)]
    count: usize,
    /// First sample's seed; sample i has seed SEED + i
    #[arg(long, default_value_t = 0)]
    seed: u64,
    /// Only these templates (comma-separated names)
    #[arg(long, value_delimiter = ',')]
    templates_only: Vec<String>,
    /// Pages made at most this many pixels on their long side (labels and
    /// masks scaled with them)
    #[arg(long)]
    max_side: Option<u32>,
    /// mixed (photos and scans, some turned), clean (light scans), photo
    #[arg(long, default_value = "mixed")]
    profile: String,
    /// Share of samples that are negatives: results pages, unfilled forms,
    /// blank pages, pages that are not forms
    #[arg(long, default_value_t = 0.12)]
    negatives: f64,
    /// Also redact each page with the pipeline, cut some masks back on
    /// purpose, and label what each field still shows (for the local
    /// reviewer); pages the pipeline holds are left out
    #[arg(long)]
    redacted: bool,
    /// Add to an existing synthetic dataset
    #[arg(long)]
    append: bool,
    #[arg(long, default_value_t = workers())]
    workers: usize,
}

#[derive(Args)]
pub struct BuildArgs {
    /// Published manifests, or folders holding them (a run's --out)
    #[arg(required = true)]
    manifests: Vec<PathBuf>,
    /// Where the originals are (files, folders, zips): found by SHA-256
    #[arg(long)]
    originals: PathBuf,
    /// The templates the run used (their zones place the fields)
    #[arg(long, default_value = "templates")]
    templates: PathBuf,
    /// The run's review folder: the documents it held are added, unlabelled
    #[arg(long)]
    review: Option<PathBuf>,
    #[arg(long)]
    out: PathBuf,
    /// Don't write the rules' ink masks
    #[arg(long)]
    no_masks: bool,
    #[arg(long)]
    append: bool,
    #[arg(long, default_value_t = workers())]
    workers: usize,
}

pub fn run(a: DatasetArgs) -> Result<i32> {
    match a.cmd {
        DatasetCmd::Synth(s) => {
            let opts = synth::Options {
                seed: s.seed,
                count: s.count,
                only: s.templates_only,
                max_side: s.max_side,
                profile: synth::Profile::parse(&s.profile)?,
                negatives: s.negatives.clamp(0.0, 1.0),
                redacted: s.redacted,
                workers: s.workers,
            };
            if opts.redacted && !rfpii::ocr::available() {
                eprintln!("warning: OCR is not available: pages of templates with free-text areas (Form 34A) will be held by the pipeline and left out");
            }
            let meta = synth::run(&s.templates, &s.out, opts, s.append)?;
            let r = &meta["this_run"];
            println!("{}: {} pages written ({} held by the pipeline), {} in the dataset; {}s, {} pages/s", s.out.display(), r["pages"], r["held"], meta["pages"], r["seconds"], r["pages_per_second"]);
            Ok(0)
        }
        DatasetCmd::Build(b) => {
            let meta = teacher::run(&teacher::BuildOptions {
                manifests: b.manifests,
                originals: b.originals,
                templates: Some(b.templates),
                review: b.review,
                out: b.out.clone(),
                append: b.append,
                workers: b.workers,
                masks: !b.no_masks,
            })?;
            let r = &meta["this_run"];
            println!("{}: {} pages written, {} failed, {} in the dataset; {}s, {} pages/s", b.out.display(), r["pages"], r["failed"], meta["pages"], r["seconds"], r["pages_per_second"]);
            for f in meta["parameters"]["failed"].as_array().into_iter().flatten() {
                eprintln!("FAILED  {}", f.as_str().unwrap_or(""));
            }
            if meta["parameters"]["sources_missing"].as_u64().unwrap_or(0) > 0 {
                eprintln!("warning: {} source(s) not found under --originals", meta["parameters"]["sources_missing"]);
            }
            Ok(if r["failed"].as_u64().unwrap_or(0) > 0 { 2 } else { 0 })
        }
        DatasetCmd::Check { dataset } => {
            let (problems, n) = check::check(&dataset)?;
            for p in &problems {
                println!("FAIL  {}", p);
            }
            if problems.is_empty() {
                println!("OK  {}: {} pages, every file, schema, box, slot and mask checked", dataset.display(), n);
                Ok(0)
            } else {
                println!("{} problem(s) in {}", problems.len(), dataset.display());
                Ok(1)
            }
        }
        DatasetCmd::Stats { dataset, json } => {
            let s = check::stats(&dataset)?;
            if json {
                println!("{}", serde_json::to_string_pretty(&s)?);
            } else {
                print!("{}", check::summary(&s));
            }
            Ok(0)
        }
        DatasetCmd::Compare { truth, other, iou, json } => {
            let r = compare::compare(&truth, &other, iou)?;
            if json {
                println!("{}", serde_json::to_string_pretty(&r)?);
            } else {
                print!("{}", compare::summary(&r));
            }
            Ok(0)
        }
    }
}
