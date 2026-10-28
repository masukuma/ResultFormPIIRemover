//! Command line.
//!
//!     rfpii keygen keys/
//!     rfpii redact Forms/ --templates templates --out public --review review --sign-key keys/signing_key.pem
//!     rfpii verify public/x.pdf --original x.PDF --pubkey keys/signing_key.pub.pem --ledger public/ledger.jsonl
//!     rfpii preview templates/form34a_2022 -o preview.png

use anyhow::{Context, Result};
use clap::{Args, Parser, Subcommand};
use opencv::core::{Point, Scalar, Vector};
use opencv::imgcodecs;
use opencv::imgproc;
use opencv::prelude::*;
use rayon::prelude::*;
use serde_json::{json, Value};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;

use rfpii::align::{prepare, AlignSettings};
use rfpii::io;
use rfpii::manifest as mf;
use rfpii::masking::MaskSettings;
use rfpii::process::{process_document, utc_now, with_suffix, Config};
use rfpii::pyjson;
use rfpii::templates::{load_templates, Template};
use rfpii::util;
use rfpii::verify::{verify, VerifyArgs};

#[derive(Parser)]
#[command(name = "rfpii", version, about = "Verifiable PII removal for result forms.")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Redact files, directories and zip archives in bulk
    Redact(RedactArgs),
    /// Prove a redacted form changed nothing but the listed blocks
    Verify(VerifyCli),
    /// Create an Ed25519 signing key pair
    Keygen { directory: PathBuf },
    /// Render template zones and protected regions for a visual check
    Preview {
        template: PathBuf,
        #[arg(short, long, default_value = "preview.png")]
        output: PathBuf,
    },
}

#[derive(Args)]
struct RedactArgs {
    #[arg(required = true)]
    inputs: Vec<String>,
    /// Template dir, or dir of template dirs
    #[arg(long)]
    templates: String,
    /// Publishable output (redacted PDFs, manifests, ledger)
    #[arg(long)]
    out: String,
    /// Reports for documents that failed checks
    #[arg(long)]
    review: String,
    /// Ed25519 private key (rfpii keygen) to sign manifests and batches
    #[arg(long)]
    sign_key: Option<PathBuf>,
    /// Append-only ledger (default <out>/ledger.jsonl)
    #[arg(long)]
    ledger: Option<PathBuf>,
    #[arg(long, default_value_t = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1))]
    workers: usize,
    /// Characters hidden per ID / phone number
    #[arg(long, default_value_t = 3)]
    hide: usize,
    /// Phone numbers: hide only within the last N digits
    #[arg(long, default_value_t = 6)]
    phone_tail: usize,
    #[arg(long, default_value_t = 0.35)]
    min_score: f64,
    #[arg(long, default_value_t = 40)]
    min_inliers: i64,
    /// Send pages whose tables can't be matched to review instead of masking with wide margins
    #[arg(long)]
    strict_structure: bool,
    /// Send pages no template matches to review instead of reading their layout
    #[arg(long)]
    no_generic: bool,
    #[arg(long)]
    overwrite: bool,
    /// Seed (hex) for every document instead of a fresh random one: for
    /// reproducing a run, never for publishing
    #[arg(long, hide = true)]
    seed: Option<String>,
}

#[derive(Args)]
struct VerifyCli {
    redacted: PathBuf,
    /// Default: <redacted>.manifest.json
    #[arg(long)]
    manifest: Option<PathBuf>,
    /// The original transmission (e.g. from the IEBC portal)
    #[arg(long)]
    original: Option<PathBuf>,
    /// Publisher's public key
    #[arg(long)]
    pubkey: Option<PathBuf>,
    /// The published ledger.jsonl
    #[arg(long)]
    ledger: Option<PathBuf>,
    /// Directory for images showing every changed block
    #[arg(long)]
    overlay: Option<PathBuf>,
    /// Also re-run the redaction from the manifest's seed
    #[arg(long, value_name = "TEMPLATES")]
    rerun: Option<PathBuf>,
}

fn out_path(root: &str, rel: &str, ext: &str) -> Result<PathBuf> {
    let p = Path::new(root).join(with_suffix(rel, ext));
    if let Some(d) = p.parent() {
        std::fs::create_dir_all(d)?;
    }
    Ok(p)
}

/// (source, output-relative path) for every input; keeps folder structure so
/// same-named files from different folders don't collide.
fn jobs(inputs: &[String]) -> Result<Vec<(String, String)>> {
    let mut out = Vec::new();
    for root in inputs {
        let rp = Path::new(root);
        for src in io::list_inputs(std::slice::from_ref(root))? {
            let rel = if let Some((archive, member)) = src.split_once(io::ZIP_SEP) {
                let base_root = if rp.is_file() { rp.parent().unwrap_or(Path::new("")) } else { rp };
                let base = Path::new(archive).strip_prefix(base_root).unwrap_or(Path::new(archive));
                let stem = with_suffix(&base.to_string_lossy(), "");
                stem.join(member).to_string_lossy().into_owned()
            } else if rp.is_dir() {
                Path::new(&src).strip_prefix(rp).map(|p| p.to_string_lossy().into_owned()).unwrap_or(src.clone())
            } else {
                Path::new(&src).file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or(src.clone())
            };
            out.push((src, rel));
        }
    }
    Ok(out)
}

struct Done {
    report: Value,
    manifest: Option<Value>,
    rel: String,
}

fn run_one(src: &str, rel: &str, templates: &[Template], cfg: &Config, seed: Option<u64>) -> Done {
    let t0 = Instant::now();
    let ms = |t0: Instant| pyjson::round_n(t0.elapsed().as_secs_f64() * 1000.0, 1);
    match process_document(src, rel, templates, cfg, seed, None) {
        Err(e) => Done { report: json!({"file": src, "ok": false, "reason": format!("Error: {:#}", e), "ms": ms(t0)}), manifest: None, rel: rel.into() },
        Ok(o) => {
            let mut report = o.report;
            if !o.ok {
                if let Ok(p) = out_path(&cfg.review, rel, ".json") {
                    let _ = std::fs::write(p, pyjson::pretty(&report, 2, false));
                }
            }
            report["ms"] = json!(ms(t0));
            Done { report, manifest: o.manifest, rel: rel.into() }
        }
    }
}

fn cmd_redact(a: RedactArgs) -> Result<i32> {
    let mut align = AlignSettings::default();
    align.min_score = a.min_score;
    align.min_inliers = a.min_inliers;
    let mut mask = MaskSettings::default();
    mask.hide = a.hide;
    mask.phone_tail = a.phone_tail;
    let cfg = Config { templates: a.templates.clone(), out: a.out.clone(), review: a.review.clone(), align, mask, strict_structure: a.strict_structure, generic: !a.no_generic };
    let key = match &a.sign_key {
        Some(p) => Some(mf::load_private(p)?),
        None => {
            eprintln!("warning: no --sign-key; manifests will be unsigned");
            None
        }
    };
    let seed = a.seed.as_deref().map(|s| u64::from_str_radix(s, 16)).transpose().context("--seed must be hex")?;
    std::fs::create_dir_all(&a.out)?;
    std::fs::create_dir_all(&a.review)?;
    let ledger = a.ledger.clone().unwrap_or_else(|| Path::new(&a.out).join("ledger.jsonl"));
    let all = jobs(&a.inputs)?;
    let mut todo = Vec::new();
    for j in all {
        if a.overwrite || !Path::new(&a.out).join(with_suffix(&j.1, ".pdf")).exists() {
            todo.push(j);
        }
    }
    let mut templates = load_templates(Path::new(&cfg.templates))?;
    for t in templates.iter_mut() {
        prepare(t, &cfg.align)?;
    }
    // Parallelism comes from documents; OpenCV itself runs one thread per
    // document, as the results must not depend on how work is split.
    opencv::core::set_num_threads(1)?;
    let pool = rayon::ThreadPoolBuilder::new().num_threads(a.workers.max(1)).build()?;
    let t0 = Instant::now();
    let (mut n_ok, mut batch) = (0usize, Vec::new());
    let mut log = std::fs::OpenOptions::new().create(true).append(true).open(Path::new(&a.review).join("report.jsonl"))?;
    for chunk in todo.chunks(4 * a.workers.max(1)) {
        let done: Vec<Done> = pool.install(|| chunk.par_iter().map(|(src, rel)| run_one(src, rel, &templates, &cfg, seed)).collect());
        for d in done {
            let r = &d.report;
            if r["ok"].as_bool() == Some(true) {
                let m = d.manifest.unwrap();
                let m = match &key {
                    Some(k) => mf::sign(&m, k),
                    None => m,
                };
                std::fs::write(out_path(&a.out, &d.rel, ".manifest.json")?, pyjson::pretty(&m, 1, true))?;
                let h = mf::digest(&m);
                mf::ledger_append(&ledger, &h, m["source"]["sha256"].as_str().unwrap_or(""), m["output"]["pdf_sha256"].as_str().unwrap_or(""), &d.rel)?;
                batch.push(json!({"name": d.rel, "manifest_sha256": h}));
                n_ok += 1;
            }
            writeln!(log, "{}", pyjson::dumps(r))?;
            if r["ok"].as_bool() != Some(true) {
                eprintln!("REVIEW  {}: {}", r["file"].as_str().unwrap_or(""), r.get("reason").and_then(|v| v.as_str()).unwrap_or(""));
            }
        }
    }
    if !batch.is_empty() {
        // One root for the whole batch: publish it widely when the batch is released.
        let bdir = Path::new(&a.out).join("batches");
        std::fs::create_dir_all(&bdir)?;
        let (_, _, entries) = mf::ledger_check(&ledger)?;
        let leaves: Vec<String> = batch.iter().map(|x| x["manifest_sha256"].as_str().unwrap().to_string()).collect();
        let created = utc_now();
        let b = json!({"schema": "rfpii/batch/1", "created_at": created, "count": batch.len(),
                       "merkle_root": mf::merkle_root(&leaves)?,
                       "ledger_head": entries.last().map(|e| e["entry_hash"].clone()).unwrap_or(Value::Null),
                       "documents": batch});
        let b = match &key {
            Some(k) => mf::sign(&b, k),
            None => b,
        };
        let bpath = bdir.join(format!("batch-{}.json", created.replace(':', "")));
        std::fs::write(&bpath, pyjson::pretty(&b, 1, true))?;
        println!("batch root {}  ({})", b["merkle_root"].as_str().unwrap_or(""), bpath.display());
    }
    let dt = t0.elapsed().as_secs_f64();
    let n = todo.len();
    println!("{}/{} published, {} sent to review, {:.1}s ({:.1} files/s)", n_ok, n, n - n_ok, dt, if dt > 0.0 { n as f64 / dt } else { 0.0 });
    Ok(if n_ok == n { 0 } else { 2 })
}

fn cmd_verify(a: VerifyCli) -> Result<i32> {
    let rep = verify(&VerifyArgs {
        redacted: &a.redacted,
        manifest: a.manifest.as_deref(),
        original: a.original.as_deref(),
        pubkey: a.pubkey.as_deref(),
        ledger: a.ledger.as_deref(),
        overlay_dir: a.overlay.as_deref(),
        rerun_templates: a.rerun.as_deref(),
    })?;
    println!("{}", rep.text());
    Ok(if rep.ok() { 0 } else { 1 })
}

fn cmd_keygen(dir: &Path) -> Result<i32> {
    let (p, q) = mf::keygen(dir)?;
    println!("private key {} (keep offline/secret)\npublic key  {} (publish)  id {}", p.display(), q.display(), mf::key_id(&mf::load_public(&q)?));
    Ok(0)
}

/// Draw zones (red blackout, purple signature, blue partial) and protected
/// results regions (green) over the reference so they can be checked by eye.
fn cmd_preview(template: &Path, output: &Path) -> Result<i32> {
    let t = load_templates(template)?.remove(0);
    let mut img = opencv::core::Mat::default();
    imgproc::cvt_color_def(&t.image, &mut img, imgproc::COLOR_GRAY2BGR)?;
    let mut over = img.try_clone()?;
    let colour = |a: &str| match a {
        "blackout" => Scalar::new(0.0, 0.0, 255.0, 0.0),
        "signature" => Scalar::new(180.0, 0.0, 200.0, 0.0),
        _ => Scalar::new(255.0, 120.0, 0.0, 0.0),
    };
    for z in &t.zones {
        let p: Vector<Point> = z.padded(t.padding as f64).iter().map(|q| Point::new(util::iround(q[0]), util::iround(q[1]))).collect();
        let polys: Vector<Vector<Point>> = vec![p].into_iter().collect();
        imgproc::fill_poly_def(&mut over, &polys, colour(&z.action))?;
    }
    let mut out = opencv::core::Mat::default();
    opencv::core::add_weighted_def(&over, 0.35, &img, 0.65, 0.0, &mut out)?;
    for (name, [x, y, w, h]) in &t.protect {
        imgproc::rectangle_points(&mut out, Point::new(*x as i32, *y as i32), Point::new((x + w) as i32, (y + h) as i32), Scalar::new(0.0, 160.0, 0.0, 0.0), 3, imgproc::LINE_8, 0)?;
        imgproc::put_text(&mut out, &format!("protected: {}", name), Point::new(*x as i32 + 4, *y as i32 + 18), imgproc::FONT_HERSHEY_SIMPLEX, 0.55, Scalar::new(0.0, 140.0, 0.0, 0.0), 2, imgproc::LINE_8, false)?;
    }
    for z in &t.zones {
        let q = z.polygon();
        let p: Vector<Point> = q.iter().map(|q| Point::new(util::iround(q[0]), util::iround(q[1]))).collect();
        let polys: Vector<Vector<Point>> = vec![p].into_iter().collect();
        imgproc::polylines(&mut out, &polys, true, colour(&z.action), 1, imgproc::LINE_8, 0)?;
        if z.row == 0 {
            let label = format!("{}{}", z.key(), if z.rule.is_empty() { String::new() } else { format!(" ({})", z.rule) });
            let at = Point::new(util::iround(q[0][0]) + 3, util::iround(q[0][1]) - 5);
            imgproc::put_text(&mut out, &label, at, imgproc::FONT_HERSHEY_SIMPLEX, 0.5, colour(&z.action), 1, imgproc::LINE_8, false)?;
        }
    }
    imgcodecs::imwrite(output, &out, &Vector::new())?;
    println!("{}", output.display());
    Ok(0)
}

fn main() {
    let cli = Cli::parse();
    let r = match cli.cmd {
        Cmd::Redact(a) => cmd_redact(a),
        Cmd::Verify(a) => cmd_verify(a),
        Cmd::Keygen { directory } => cmd_keygen(&directory),
        Cmd::Preview { template, output } => cmd_preview(&template, &output),
    };
    match r {
        Ok(code) => std::process::exit(code),
        Err(e) => {
            eprintln!("error: {:#}", e);
            std::process::exit(1);
        }
    }
}
