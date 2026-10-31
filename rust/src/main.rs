//! Command line.
//!
//!     rfpii keygen keys/
//!     rfpii certgen --name "Returning Officer, Kibwezi West"
//!     rfpii redact Forms/ --templates templates --out public --review review --sign-key keys/signing_key.pem
//!     rfpii verify public/x.pdf --original x.PDF --pubkey keys/signing_key.pub.pem --ledger public/ledger.jsonl
//!     rfpii preview templates/form34a_2022 -o preview.png

mod cli_legal;

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
use rfpii::process::{output_rel, process_document, utc_now, with_suffix, Config};
use rfpii::pyjson;
use rfpii::templates::{load_templates, Template};
use rfpii::util;
use rfpii::pdfsign;
use rfpii::pipeline::redact_boxes;
use rfpii::pyrand::PyRandom;
use rfpii::verify::{verify, VerifyArgs};
use rfpii::vlm::{Ollama, VlmSettings};

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
    /// Create an Ed25519 signing key pair (the private key encrypted)
    Keygen(cli_legal::KeygenArgs),
    /// Hand signing over to a new key, recorded in the ledger
    RotateKey(cli_legal::RotateKeyArgs),
    /// Revoke a signing key (lost or stolen), recorded in the ledger
    RevokeKey(cli_legal::RevokeKeyArgs),
    /// Take a published form down (a leak found), recorded in the ledger
    Withdraw(cli_legal::WithdrawArgs),
    /// Create a self-signed certificate for digitally signing the PDFs
    Certgen(CertgenArgs),
    /// Render template zones and protected regions for a visual check
    Preview {
        template: PathBuf,
        #[arg(short, long, default_value = "preview.png")]
        output: PathBuf,
    },
    /// Draft a template's zones from a reference image with a VLM
    Propose(ProposeArgs),
    /// Learn each printing's layout from its forms and write it as a template
    Learn(LearnArgs),
    /// Generate a sample template and synthetic scans
    Demo {
        directory: PathBuf,
        #[arg(long, default_value_t = 20)]
        count: u64,
    },
}

#[derive(Args)]
struct LearnArgs {
    /// Files, directories or zip archives of forms (KIEMS-named files are
    /// grouped by printing; others are taken as one printing)
    #[arg(required = true)]
    inputs: Vec<String>,
    /// Directory to write the learned templates to
    #[arg(long)]
    out: PathBuf,
    /// The form of files whose name doesn't say (e.g. 34A)
    #[arg(long)]
    form: Option<String>,
    /// Pages used per printing, at most
    #[arg(long, default_value_t = 40)]
    max_pages: usize,
    /// Pages a printing needs to be learned
    #[arg(long, default_value_t = 5)]
    min_pages: usize,
    #[arg(long, default_value_t = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1))]
    workers: usize,
}

#[derive(Args, Clone)]
struct VlmArgs {
    /// Vision model (default: gemma4:31b on Ollama Cloud, qwen3.8:27b locally)
    #[arg(long, default_value = "")]
    vlm_model: String,
    /// Ollama URL (default $OLLAMA_HOST, else https://ollama.com if $OLLAMA_API_KEY, else local)
    #[arg(long)]
    vlm_host: Option<String>,
    /// Box convention of the model (auto: norm1000 for Qwen3.x/Qwen3-VL, pixel otherwise)
    #[arg(long, default_value = "auto", value_parser = ["auto", "pixel", "norm1000"])]
    vlm_coords: String,
    /// Allow sending UNREDACTED forms to a non-local VLM (fallback/propose); only if that data transfer is approved
    #[arg(long)]
    allow_remote_unredacted: bool,
}

impl VlmArgs {
    fn settings(&self) -> VlmSettings {
        let mut s = VlmSettings::default();
        s.coords = self.vlm_coords.clone();
        s.allow_remote_unredacted = self.allow_remote_unredacted;
        if !self.vlm_model.is_empty() {
            s.model = self.vlm_model.clone();
        }
        if let Some(h) = &self.vlm_host {
            s.host = h.clone();
        }
        s
    }
}

#[derive(Args)]
struct ProposeArgs {
    reference: String,
    /// Path of the template.yaml to write
    #[arg(short, long)]
    output: PathBuf,
    /// Form code, e.g. 34A
    #[arg(long, default_value = "")]
    form: String,
    #[command(flatten)]
    vlm: VlmArgs,
}

#[derive(Args)]
struct RedactArgs {
    #[arg(required = true)]
    inputs: Vec<String>,
    /// Template dir, or dir of template dirs
    #[arg(long, required_unless_present = "learn")]
    templates: Option<String>,
    /// Learn the layout of each printing among the inputs that has no
    /// template in this directory yet, write it there, then redact with
    /// those templates (and --templates, if given)
    #[arg(long)]
    learn: Option<PathBuf>,
    /// With --learn: the form of files whose name doesn't say (e.g. 34A)
    #[arg(long, requires = "learn")]
    form: Option<String>,
    /// Publishable output (redacted PDFs, manifests, ledger)
    #[arg(long)]
    out: String,
    /// Reports for documents that failed checks
    #[arg(long)]
    review: String,
    /// Ed25519 private key (rfpii keygen) to sign manifests and batches
    #[arg(long)]
    sign_key: Option<PathBuf>,
    /// Certificate to digitally sign the PDFs with: PKCS#12 (.p12/.pfx), or
    /// PEM with the key (password from $RFPII_CERT_PASSWORD, else asked).
    /// Default: ~/.config/rfpii/signer.p12 if it exists; else you are
    /// offered to create a self-signed one
    #[arg(long)]
    certificate: Option<PathBuf>,
    /// PEM private key, if not in the --certificate file
    #[arg(long, requires = "certificate")]
    certificate_key: Option<PathBuf>,
    /// Don't sign the PDFs (and don't offer to create a certificate)
    #[arg(long, conflicts_with = "certificate")]
    no_pdf_signature: bool,
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
    /// Never mask a stamp: an ID, phone number or signature a stamp lies
    /// over stays visible. By default each field's band across a stamp is
    /// blacked whole, and the rest of the stamp is protected and verified
    /// unchanged.
    #[arg(long, conflicts_with = "mask_stamps")]
    keep_stamps: bool,
    /// Hide writing under stamps along each field's band (the default;
    /// accepted for older scripts)
    #[arg(long)]
    mask_stamps: bool,
    /// Hide every character of each ID or passport number, not `--hide` of
    /// them
    #[arg(long)]
    full_ids: bool,
    /// Hide every digit of each phone number, its first digits too
    #[arg(long)]
    full_phones: bool,
    /// Who publishes the forms and on what legal basis (controller.json: the
    /// controller and its registration, the operator's role, legal basis,
    /// purpose, commercial use, processors, DPIA). Recorded in every
    /// manifest; see legal/C-recommendations.md §2.9
    #[arg(long)]
    controller: Option<PathBuf>,
    /// A model host in Kenya (repeatable): calls to it are not transfers
    /// abroad. Every call to any other remote host is recorded in
    /// <out>/transfers.jsonl (Data Protection (General) Regulations reg. 41(2))
    #[arg(long = "kenya-host")]
    kenya_hosts: Vec<String>,
    /// The manifest of a withdrawn form (rfpii withdraw) this run republishes
    /// corrected (repeatable): matched by source file and page, and named in
    /// the new manifest
    #[arg(long)]
    supersedes: Vec<PathBuf>,
    /// Don't write the redaction notice below each published page (by
    /// default a line outside the form image says it is a redacted copy,
    /// names the original by its hash, and says how to check it)
    #[arg(long)]
    no_notice: bool,
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
    /// Review every redacted page with vision models before publishing: what
    /// they find still showing is masked and the page reviewed again. Only
    /// redacted pages are sent.
    #[arg(long)]
    vlm_audit: bool,
    /// With --vlm-audit: a page whose ID/phone columns can't be placed has
    /// those columns (and its table's signatures) hidden whole, and is
    /// published if the review then finds nothing
    #[arg(long)]
    vlm_rescue: bool,
    /// Review rounds before a page that still shows PII goes to review
    #[arg(long, default_value_t = 2)]
    vlm_rounds: usize,
    /// Model that confirms each spot the review model finds (default:
    /// gemma4:31b on Ollama Cloud, the review model locally). A decision
    /// model such as clef is asked through /v1/systemone
    #[arg(long, default_value = "")]
    vlm_checker: String,
    /// With a decision model as --vlm-checker: the probability at or above
    /// which a spot is masked
    #[arg(long, default_value_t = 0.5)]
    vlm_check_threshold: f64,
    /// Requests to the model host at once, across all workers (Ollama Cloud
    /// answered 429 beyond 12 at once on the account this was tested with)
    #[arg(long, default_value_t = 10)]
    vlm_concurrency: usize,
    /// VLM-redact pages no template matches, into review/vlm/ for approval
    #[arg(long)]
    vlm_fallback: bool,
    /// Photos with no EXIF orientation tag: ask a decision model (Clef, on
    /// Ollama Cloud) which way up each published page stands, and turn the
    /// page to show it upright. Only redacted pages are sent.
    #[arg(long)]
    orient: bool,
    /// The decision model --orient asks
    #[arg(long, default_value = "clef")]
    orient_model: String,
    #[command(flatten)]
    vlm: VlmArgs,
    #[arg(long)]
    overwrite: bool,
    /// Each page of a file of several pages is a document of its own,
    /// published (as NAME.p012.pdf) or sent to review by itself: for scanned
    /// batches of many forms. Its manifest names the whole file and the page.
    #[arg(long)]
    per_page: bool,
    /// Seed (hex) for every document instead of a fresh random one: for
    /// reproducing a run, never for publishing
    #[arg(long, hide = true)]
    seed: Option<String>,
}

#[derive(Args)]
struct CertgenArgs {
    /// Signer's name, as PDF readers will show it (e.g. "Returning Officer, Kibwezi West")
    #[arg(long)]
    name: String,
    /// Organisation
    #[arg(long)]
    org: Option<String>,
    /// Two-letter country code (e.g. KE)
    #[arg(long)]
    country: Option<String>,
    /// Days the certificate is valid
    #[arg(long, default_value_t = 1095)]
    days: u32,
    /// PKCS#12 file to create (default ~/.config/rfpii/signer.p12); the
    /// public certificate is written beside it as .crt
    #[arg(long)]
    out: Option<PathBuf>,
    /// Replace an existing certificate
    #[arg(long)]
    force: bool,
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
    /// Certificate (PEM or DER) the PDF's signer must chain to; for a
    /// self-signed certificate, the certificate itself
    #[arg(long)]
    trust: Vec<PathBuf>,
}

fn out_path(root: &str, rel: &str, page: Option<usize>, ext: &str) -> Result<PathBuf> {
    let p = Path::new(root).join(output_rel(rel, page, ext));
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

/// A document: a file (`rel` its path under the output folder), or one page
/// of it with --per-page.
#[derive(Clone)]
struct Job {
    src: String,
    rel: String,
    page: Option<usize>,
}

impl Job {
    /// What the ledger and batch call it.
    fn name(&self) -> String {
        match self.page {
            None => self.rel.clone(),
            Some(_) => output_rel(&self.rel, self.page, "").to_string_lossy().into_owned(),
        }
    }
}

/// With `per_page`, a job for each page of the files of several pages.
fn page_jobs(jobs: Vec<(String, String)>, per_page: bool) -> Result<Vec<Job>> {
    let mut out = Vec::new();
    for (src, rel) in jobs {
        let n = if per_page { io::page_count(&src, &io::read_bytes(&src)?)? } else { 1 };
        if n > 1 {
            out.extend((1..=n).map(|k| Job { src: src.clone(), rel: rel.clone(), page: Some(k) }));
        } else {
            out.push(Job { src, rel, page: None });
        }
    }
    Ok(out)
}

struct Done {
    report: Value,
    manifest: Option<Value>,
    job: Job,
}

fn run_one(j: &Job, templates: &[Template], cfg: &Config, vlm: Option<&Ollama>, seed: Option<u64>) -> Done {
    let t0 = Instant::now();
    let ms = |t0: Instant| pyjson::round_n(t0.elapsed().as_secs_f64() * 1000.0, 1);
    let (src, rel) = (j.src.as_str(), j.rel.as_str());
    match process_document(src, rel, j.page, templates, cfg, seed, None, None) {
        Err(e) => {
            let mut report = json!({"file": src, "ok": false, "reason": format!("Error: {:#}", e), "ms": ms(t0)});
            if let Some(k) = j.page {
                report["page"] = json!(k);
            }
            Done { report, manifest: None, job: j.clone() }
        }
        Ok(o) => {
            let mut report = o.report;
            if !o.ok {
                if let Ok(p) = out_path(&cfg.review, rel, j.page, ".json") {
                    let _ = std::fs::write(p, pyjson::pretty(&report, 2, false));
                }
                if let (Some(v), true, false) = (vlm, cfg.vlm_fallback, o.source_pages.is_empty()) {
                    // Unrecognised layout: let the VLM find the PII, apply the
                    // same masking, and park the result for a person to approve.
                    match vlm_proposal(v, &o.source_pages, cfg, rel, j.page) {
                        Ok(p) => report["vlm_proposal"] = json!(p.to_string_lossy()),
                        Err(e) => report["vlm_error"] = json!(format!("Error: {:#}", e)),
                    }
                }
            }
            report["ms"] = json!(ms(t0));
            Done { report, manifest: o.manifest, job: j.clone() }
        }
    }
}

/// A VLM-redacted copy of a document, for review (never published).
fn vlm_proposal(v: &Ollama, pages: &[io::SourcePage], cfg: &Config, rel: &str, page: Option<usize>) -> Result<PathBuf> {
    let mut seed = [0u8; 8];
    mf::getrandom_fill(&mut seed)?;
    let mut rng = PyRandom::new(u64::from_le_bytes(seed) >> 1);
    let mut images = Vec::new();
    for (k, sp) in pages.iter().enumerate() {
        let boxes: Vec<(f64, f64, f64, f64, String)> = v.locate(&sp.pixels)?.into_iter().map(|b| (b.x0, b.y0, b.x1, b.y1, b.label)).collect();
        cfg.transfer(rfpii::legal::Transfer {
            host: &v.s.host,
            model: &v.s.model,
            purpose: "locating fields on a page that couldn't be read",
            data: "the unredacted page: names, parties, ID and phone numbers and signatures of officers and agents".into(),
            document: rel,
            page: page.unwrap_or(k + 1),
            images: vec![],
        })?;
        images.push(redact_boxes(&sp.pixels, &boxes, &cfg.mask, &mut rng, 10.0)?.image.unwrap());
    }
    let out = out_path(&Path::new(&cfg.review).join("vlm").to_string_lossy(), rel, page, ".pdf")?;
    io::write_pdf(&images, &out, 200.0, 85)?;
    Ok(out)
}

/// The review model and the model that confirms what it finds, each asked
/// what it is before the run starts.
fn reviewer(a: &RedactArgs) -> Result<rfpii::vlm::Reviewer> {
    let mut finder = Ollama::new(a.vlm.settings())?;
    finder.probe()?;
    if finder.decision {
        anyhow::bail!("{} is a decision model: it can check spots (--vlm-checker) but not find them", finder.s.model);
    }
    let mut cs = a.vlm.settings();
    cs.model = if !a.vlm_checker.is_empty() {
        a.vlm_checker.clone()
    } else if finder.local() {
        finder.s.model.clone()
    } else {
        rfpii::vlm::DEFAULT_CLOUD_CHECKER.to_string()
    };
    cs.check_threshold = a.vlm_check_threshold;
    let mut checker = Ollama::new(cs)?;
    checker.probe()?;
    Ok(rfpii::vlm::Reviewer { finder, checker })
}

/// The decision model `--orient` asks, asked what it is before the run.
fn orienter(a: &RedactArgs) -> Result<Ollama> {
    let mut s = a.vlm.settings();
    s.model = a.orient_model.clone();
    let mut o = Ollama::new(s)?;
    o.probe()?;
    if !o.decision {
        anyhow::bail!("--orient needs a decision model such as clef; {} at {} is not one", o.s.model, o.s.host);
    }
    Ok(o)
}

fn cmd_redact(a: RedactArgs) -> Result<i32> {
    let mut align = AlignSettings::default();
    align.min_score = a.min_score;
    align.min_inliers = a.min_inliers;
    let mut mask = MaskSettings::default();
    mask.hide = a.hide;
    mask.phone_tail = a.phone_tail;
    mask.stamps = if a.keep_stamps { rfpii::masking::StampRule::Protected } else { rfpii::masking::StampRule::Bands };
    mask.full_ids = a.full_ids;
    mask.full_phones = a.full_phones;
    let cfg = Config {
        templates: a.templates.clone().unwrap_or_default(),
        out: a.out.clone(),
        review: a.review.clone(),
        align,
        mask,
        strict_structure: a.strict_structure,
        generic: !a.no_generic,
        vlm_audit: a.vlm_audit,
        vlm_fallback: a.vlm_fallback,
        reviewer: if a.vlm_audit { Some(std::sync::Arc::new(reviewer(&a)?)) } else { None },
        vlm_rescue: a.vlm_rescue,
        vlm_rounds: a.vlm_rounds,
        pdf_signer: pdf_signer(&a)?.map(std::sync::Arc::new),
        orient: if a.orient { Some(std::sync::Arc::new(orienter(&a)?)) } else { None },
        controller: match &a.controller {
            Some(p) => Some(std::sync::Arc::new(rfpii::legal::Controller::load(p)?)),
            None => None,
        },
        transfers: Some(std::sync::Arc::new(rfpii::legal::Transfers::new(Path::new(&a.out).join("transfers.jsonl")))),
        kenya_hosts: a.kenya_hosts.clone(),
        supersedes: a.supersedes.iter().map(|p| mf::read_json(p)).collect::<Result<_>>()?,
        notice: !a.no_notice,
    };
    if a.vlm_rescue && !a.vlm_audit {
        anyhow::bail!("--vlm-rescue needs --vlm-audit: hiding a column whole is only safe when the review checks the result");
    }
    rfpii::vlm::set_concurrency(a.vlm_concurrency);
    // fail fast on a bad VLM configuration
    let vlm = if a.vlm_audit || a.vlm_fallback { Some(Ollama::new(a.vlm.settings())?) } else { None };
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
    let todo: Vec<Job> = page_jobs(all.clone(), a.per_page)?
        .into_iter()
        .filter(|j| a.overwrite || !Path::new(&a.out).join(output_rel(&j.rel, j.page, ".pdf")).exists())
        .collect();
    let mut templates = match &a.templates {
        Some(t) => load_templates(Path::new(t))?,
        None => Vec::new(),
    };
    if let Some(dir) = &a.learn {
        if !rfpii::ocr::available() {
            anyhow::bail!("--learn needs OCR: {}", rfpii::ocr::check().err().map(|e| e.to_string()).unwrap_or_default());
        }
        let sources: Vec<String> = all.iter().map(|j| j.0.clone()).collect();
        let pool = rayon::ThreadPoolBuilder::new().num_threads(a.workers.max(1)).build()?;
        let opts = rfpii::learn::LearnOptions::default();
        let results = match pool.install(|| rfpii::learn::learn_all(&sources, a.form.as_deref(), dir, &opts, true)) {
            Ok(r) => r,
            Err(e) => {
                println!("nothing to learn: {}", e);
                Vec::new()
            }
        };
        report_learned(&results, dir)?;
        if dir.exists() {
            if let Ok(l) = load_templates(dir) {
                templates.extend(l);
            }
        }
    }
    if templates.is_empty() {
        anyhow::bail!("no templates: none given, and none could be learned");
    }
    for t in templates.iter_mut() {
        prepare(t, &cfg.align)?;
    }
    // Parallelism comes from documents; OpenCV itself runs one thread per
    // document, as the results must not depend on how work is split.
    opencv::core::set_num_threads(1)?;
    rfpii::ocr::set_threads(1);
    let pool = rayon::ThreadPoolBuilder::new().num_threads(a.workers.max(1)).build()?;
    let t0 = Instant::now();
    let (mut n_ok, mut batch) = (0usize, Vec::new());
    let mut log = std::fs::OpenOptions::new().create(true).append(true).open(Path::new(&a.review).join("report.jsonl"))?;
    for chunk in todo.chunks(4 * a.workers.max(1)) {
        let done: Vec<Done> = pool.install(|| chunk.par_iter().map(|j| run_one(j, &templates, &cfg, vlm.as_ref(), seed)).collect());
        for d in done {
            let r = &d.report;
            if r["ok"].as_bool() == Some(true) {
                let m = d.manifest.unwrap();
                let m = match &key {
                    Some(k) => mf::sign(&m, k),
                    None => m,
                };
                std::fs::write(out_path(&a.out, &d.job.rel, d.job.page, ".manifest.json")?, pyjson::pretty(&m, 1, true))?;
                let h = mf::digest(&m);
                let name = d.job.name();
                mf::ledger_append(&ledger, &h, m["source"]["sha256"].as_str().unwrap_or(""), m["output"]["pdf_sha256"].as_str().unwrap_or(""), &name)?;
                batch.push(json!({"name": name, "manifest_sha256": h}));
                n_ok += 1;
            }
            writeln!(log, "{}", pyjson::dumps(r))?;
            if r["ok"].as_bool() != Some(true) {
                let at = d.job.page.map(|k| format!(" (page {} alone)", k)).unwrap_or_default();
                eprintln!("REVIEW  {}{}: {}", r["file"].as_str().unwrap_or(""), at, r.get("reason").and_then(|v| v.as_str()).unwrap_or(""));
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

fn interactive() -> bool {
    use std::io::IsTerminal;
    std::io::stdin().is_terminal() && std::io::stderr().is_terminal()
}

fn ask(question: &str) -> Result<String> {
    eprint!("{}", question);
    std::io::stderr().flush()?;
    let mut line = String::new();
    std::io::stdin().read_line(&mut line)?;
    Ok(line.trim().to_string())
}

/// The password of a certificate file: $RFPII_CERT_PASSWORD, else asked.
fn cert_password(path: &Path) -> Result<String> {
    if let Ok(p) = std::env::var("RFPII_CERT_PASSWORD") {
        return Ok(p);
    }
    if !interactive() {
        anyhow::bail!("{} needs a password: set RFPII_CERT_PASSWORD", path.display());
    }
    Ok(rpassword::prompt_password(format!("Password for {}: ", path.display()))?)
}

/// A new password: $RFPII_CERT_PASSWORD, else asked twice.
fn new_password() -> Result<String> {
    if let Ok(p) = std::env::var("RFPII_CERT_PASSWORD") {
        return Ok(p);
    }
    if !interactive() {
        anyhow::bail!("set RFPII_CERT_PASSWORD to the password protecting the private key");
    }
    loop {
        let p = rpassword::prompt_password("Password to protect the private key: ")?;
        if p.is_empty() {
            eprintln!("The password can't be empty.");
            continue;
        }
        if rpassword::prompt_password("Again: ")? == p {
            return Ok(p);
        }
        eprintln!("They don't match.");
    }
}

fn show_certificate(s: &pdfsign::Signer, p12: Option<&Path>) -> Result<()> {
    let d = pdfsign::describe(s.certificate())?;
    eprintln!("signing PDFs as {}", d["subject"].as_str().unwrap_or(""));
    eprintln!("  certificate SHA-256 {}", d["certificate_sha256"].as_str().unwrap_or(""));
    if d["self_signed"].as_bool() == Some(true) {
        let crt = p12.map(|p| format!(" ({})", pdfsign::certificate_path(p).display())).unwrap_or_default();
        eprintln!("  self-signed: publish this fingerprint or the certificate{} so readers can check who signed", crt);
    }
    Ok(())
}

/// The certificate to sign the PDFs with: the one given, the default one,
/// or (asked) a new self-signed one; None to publish unsigned PDFs.
fn pdf_signer(a: &RedactArgs) -> Result<Option<pdfsign::Signer>> {
    if a.no_pdf_signature {
        return Ok(None);
    }
    let default = pdfsign::default_path();
    let path = match &a.certificate {
        Some(p) => p.clone(),
        None if default.exists() => default,
        None => return offer_certificate(&default),
    };
    let s = pdfsign::Signer::load(&path, a.certificate_key.as_deref(), &mut || cert_password(&path))?;
    show_certificate(&s, path.extension().filter(|e| *e == "p12").map(|_| path.as_path()))?;
    Ok(Some(s))
}

/// No certificate given: offer to create a self-signed one.
fn offer_certificate(path: &Path) -> Result<Option<pdfsign::Signer>> {
    let unsigned = || eprintln!("warning: no --certificate; the PDFs will not be digitally signed (`rfpii certgen` creates a certificate)");
    if !interactive() {
        unsigned();
        return Ok(None);
    }
    eprintln!("No certificate to digitally sign the published PDFs with.");
    let yes = ask(&format!("Create a self-signed certificate now (saved as {})? [y/N] ", path.display()))?;
    if !matches!(yes.to_lowercase().as_str(), "y" | "yes") {
        unsigned();
        return Ok(None);
    }
    let name = loop {
        let n = ask("Signer's name, as PDF readers will show it (e.g. Returning Officer, Kibwezi West): ")?;
        if !n.is_empty() {
            break n;
        }
    };
    let org = ask("Organisation (optional): ")?;
    let country = loop {
        let c = ask("Two-letter country code (optional, e.g. KE): ")?.to_uppercase();
        if c.is_empty() || (c.len() == 2 && c.chars().all(|x| x.is_ascii_alphabetic())) {
            break c;
        }
    };
    let s = pdfsign::certgen(path, &name, Some(&org), Some(&country), 1095, &new_password()?)?;
    eprintln!("created {} (private: keep it safe) and {} (public)", path.display(), pdfsign::certificate_path(path).display());
    show_certificate(&s, Some(path))?;
    Ok(Some(s))
}

fn cmd_certgen(a: CertgenArgs) -> Result<i32> {
    let default = a.out.is_none();
    let out = a.out.unwrap_or_else(pdfsign::default_path);
    if out.exists() && !a.force {
        anyhow::bail!("{} exists (--force replaces it)", out.display());
    }
    if let Some(c) = a.country.as_deref().filter(|c| !c.is_empty()) {
        if c.len() != 2 || !c.chars().all(|x| x.is_ascii_alphabetic()) {
            anyhow::bail!("--country takes a two-letter code such as KE");
        }
    }
    let country = a.country.map(|c| c.to_uppercase());
    let s = pdfsign::certgen(&out, &a.name, a.org.as_deref(), country.as_deref(), a.days, &new_password()?)?;
    let d = pdfsign::describe(s.certificate())?;
    println!("private  {} (keep it safe; rfpii redact uses it{})", out.display(), if default { " by default" } else { " with --certificate" });
    println!("public   {} (publish it)", pdfsign::certificate_path(&out).display());
    println!("subject  {}", d["subject"].as_str().unwrap_or(""));
    println!("SHA-256  {}", d["certificate_sha256"].as_str().unwrap_or(""));
    println!("valid to {}", d["not_after"].as_str().unwrap_or(""));
    Ok(0)
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
        trust: &a.trust,
    })?;
    println!("{}", rep.text());
    Ok(if rep.ok() { 0 } else { 1 })
}

/// Draw zones (red blackout, purple signature, blue partial), protected
/// results regions (green) and fence labels (orange) over the reference so
/// they can be checked by eye.
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
    // fence labels (orange): where a widened search stops
    let mut fences: Vec<[f64; 4]> = t.zones.iter().flat_map(|z| z.fence.iter().flatten().copied()).collect();
    fences.sort_by(|a, b| a.partial_cmp(b).unwrap());
    fences.dedup();
    for [x, y, w, h] in fences {
        imgproc::rectangle_points(&mut out, Point::new(x as i32, y as i32), Point::new((x + w) as i32, (y + h) as i32), Scalar::new(0.0, 140.0, 255.0, 0.0), 2, imgproc::LINE_8, 0)?;
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

/// Ask the VLM for PII boxes on a reference image and write a draft template.
fn cmd_propose(a: ProposeArgs) -> Result<i32> {
    let img = io::source_pages(&a.reference, None)?.remove(0).pixels;
    let client = Ollama::new(a.vlm.settings())?;
    let boxes = client.locate(&img)?;
    if let Some(d) = a.output.parent() {
        std::fs::create_dir_all(d)?;
    }
    let dir = a.output.parent().unwrap_or(Path::new("."));
    imgcodecs::imwrite(&dir.join("reference.png"), &util::gray(&img)?, &Vector::new())?;
    let mut y = format!(
        "# DRAFT from a VLM: check every zone with `rfpii preview` before use, and add\n# `protect` rects for every results region. Erase handwriting from reference.png\n# (keep printed rules) before committing it.\nname: {}\nform: '{}'\nreference: reference.png\npadding: 8\nprotect: []\nzones:\n",
        dir.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default(),
        a.form
    );
    for (i, b) in boxes.iter().enumerate() {
        let r = [util::round_even(b.x0), util::round_even(b.y0), util::round_even(b.x1 - b.x0), util::round_even(b.y1 - b.y0)];
        let sig = b.label == "signature";
        y += &format!("- name: {}_{}\n  action: {}\n  rect:\n  - {}\n  - {}\n  - {}\n  - {}\n", b.label, i, if sig { "blackout" } else { "partial" }, r[0] as i64, r[1] as i64, r[2] as i64, r[3] as i64);
        if !sig {
            let phone = b.label == "phone_number";
            y += &format!("  rule: {}\n  digits: {}\n", if phone { "phone" } else { "id" }, if phone { 10 } else { 8 });
        }
    }
    if boxes.is_empty() {
        y = y.replace("zones:\n", "zones: []\n");
    }
    std::fs::write(&a.output, y)?;
    println!("{} zones -> {}", boxes.len(), a.output.display());
    Ok(0)
}

fn cmd_learn(a: LearnArgs) -> Result<i32> {
    if !rfpii::ocr::available() {
        anyhow::bail!("learning needs OCR: {}", rfpii::ocr::check().err().map(|e| e.to_string()).unwrap_or_default());
    }
    let sources = io::list_inputs(&a.inputs)?;
    let opts = rfpii::learn::LearnOptions { max_pages: a.max_pages, min_pages: a.min_pages, ..Default::default() };
    let pool = rayon::ThreadPoolBuilder::new().num_threads(a.workers.max(1)).build()?;
    let results = pool.install(|| rfpii::learn::learn_all(&sources, a.form.as_deref(), &a.out, &opts, false))?;
    let learned = report_learned(&results, &a.out)?;
    Ok(if learned == results.len() { 0 } else { 2 })
}

/// Print what was learned and append it to `<out>/learn_report.jsonl`;
/// returns how many printings were learned.
fn report_learned(results: &[Result<rfpii::learn::Learned>], out: &Path) -> Result<usize> {
    std::fs::create_dir_all(out)?;
    let mut report = std::fs::OpenOptions::new().create(true).append(true).open(out.join("learn_report.jsonl"))?;
    let mut learned = 0;
    for r in results {
        match r {
            Ok(l) => {
                learned += 1;
                let used = l.record["pages_used"].as_array().map(|v| v.len()).unwrap_or(0);
                let fields = l.record["fields"].as_array().map(|v| v.len()).unwrap_or(0);
                println!("LEARNED {}: {} pages, {} fields -> {}", l.name, used, fields, l.dir.display());
                for n in l.record["notes"].as_array().into_iter().flatten() {
                    println!("  note: {}", n.as_str().unwrap_or(""));
                }
                writeln!(report, "{}", serde_json::to_string(&l.record)?)?;
            }
            Err(e) => {
                println!("NOT LEARNED {}", e);
                writeln!(report, "{}", serde_json::json!({"error": e.to_string()}))?;
            }
        }
    }
    println!("{}/{} printing(s) learned", learned, results.len());
    Ok(learned)
}

fn cmd_demo(dir: &Path, count: u64) -> Result<i32> {
    use rfpii::synthetic;
    synthetic::write_template(&dir.join("templates").join("sample_rs1"))?;
    let scans = dir.join("scans");
    std::fs::create_dir_all(&scans)?;
    let q: Vector<i32> = vec![imgcodecs::IMWRITE_JPEG_QUALITY, 85].into_iter().collect();
    for i in 0..count {
        let angle = (i as f64 * 0.6180339887).fract() * 10.0 - 5.0;
        let (scan, _) = synthetic::simulate_scan(&synthetic::filled_form(i)?, i, angle, 1.3)?;
        imgcodecs::imwrite(&scans.join(format!("station_{:04}.jpg", i)), &scan, &q)?;
    }
    // something that is not a form at all
    let mut noise = vec![0u8; 1800 * 1300 * 3];
    mf::getrandom_fill(&mut noise)?;
    let mut nm = opencv::core::Mat::new_rows_cols_with_default(1800, 1300, opencv::core::CV_8UC3, Scalar::all(0.0))?;
    nm.data_bytes_mut()?.copy_from_slice(&noise);
    imgcodecs::imwrite(&scans.join("not_a_form.jpg"), &nm, &q)?;
    println!("wrote template and {} scans under {}", count + 1, dir.display());
    Ok(0)
}

fn main() {
    let cli = Cli::parse();
    let r = match cli.cmd {
        Cmd::Redact(a) => cmd_redact(a),
        Cmd::Verify(a) => cmd_verify(a),
        Cmd::Keygen(a) => cli_legal::keygen(a),
        Cmd::RotateKey(a) => cli_legal::rotate_key(a),
        Cmd::RevokeKey(a) => cli_legal::revoke_key(a),
        Cmd::Withdraw(a) => cli_legal::withdraw(a),
        Cmd::Certgen(a) => cmd_certgen(a),
        Cmd::Preview { template, output } => cmd_preview(&template, &output),
        Cmd::Propose(a) => cmd_propose(a),
        Cmd::Learn(a) => cmd_learn(a),
        Cmd::Demo { directory, count } => cmd_demo(&directory, count),
    };
    match r {
        Ok(code) => std::process::exit(code),
        Err(e) => {
            eprintln!("error: {:#}", e);
            std::process::exit(1);
        }
    }
}
