//! `rfpii verify`: check a published form against its original.
//!
//! Meant to be run by anyone: observers, parties, courts, journalists. Given the
//! redacted PDF, its manifest and (ideally) the original transmission, it checks:
//!
//!  1. the manifest's signature (publisher's public key) and the ledger chain;
//!  2. that the redacted PDF is the one the manifest describes;
//!  3. that the original is the one the manifest describes (SHA-256);
//!  4. block by block, that every 8x8 block of the image is either *identical*
//!     to the original (JPEG: identical quantised DCT coefficients, independent
//!     of any decoder) or *solid black and listed* in the manifest;
//!  5. that no blacked block touches a protected results region, and that those
//!     regions decode to identical pixels in both files;
//!  6. optionally (--rerun), that re-running the same code, templates and seed
//!     on the original reproduces the same output.
//!
//! It also writes an overlay per page: the original with every changed block
//! shaded red and the results regions outlined, for people to look at.

use anyhow::Result;
use opencv::core::{self, Mat, Point, Scalar, Vector};
use opencv::imgcodecs;
use opencv::imgproc;
use opencv::prelude::*;
use serde_json::Value;
use std::path::{Path, PathBuf};

use crate::align::{prepare, AlignSettings};
use crate::io;
use crate::manifest as mf;
use crate::masking::MaskSettings;
use crate::pdfsign;
use crate::process::{process_document, Config};
use crate::proof;
use crate::templates::load_templates;
use crate::util::{self, Grid};

#[derive(Default)]
pub struct Report {
    pub checks: Vec<(bool, String)>,
    pub notes: Vec<String>,
}

impl Report {
    pub fn add(&mut self, ok: bool, msg: impl Into<String>) -> bool {
        self.checks.push((ok, msg.into()));
        ok
    }

    pub fn ok(&self) -> bool {
        self.checks.iter().all(|(ok, _)| *ok)
    }

    pub fn text(&self) -> String {
        let mut lines: Vec<String> = self.checks.iter().map(|(ok, m)| format!("{}{}", if *ok { "PASS  " } else { "FAIL  " }, m)).collect();
        lines.extend(self.notes.iter().map(|n| format!("note  {}", n)));
        lines.push(format!(
            "RESULT: {}",
            if self.ok() { "VERIFIED - only the listed blocks were blacked out; nothing else changed" } else { "NOT VERIFIED" }
        ));
        lines.join("\n")
    }
}

fn polygon(p: &Value) -> Vec<[f64; 2]> {
    p["polygon"].as_array().map(|a| a.iter().map(|q| [q[0].as_f64().unwrap_or(0.0), q[1].as_f64().unwrap_or(0.0)]).collect()).unwrap_or_default()
}

fn overlay(img: &Mat, blocks: &Grid, protected: &[Value]) -> Result<Mat> {
    let m = proof::to_pixels(blocks, img.rows(), img.cols())?;
    let mut red = img.try_clone()?;
    red.set_to(&Scalar::new(0.0, 0.0, 255.0, 0.0), &m)?;
    let mut out = Mat::default();
    core::add_weighted_def(&red, 0.55, img, 0.45, 0.0, &mut out)?;
    for p in protected {
        // stamps magenta, results green
        let color = if p.get("kind").and_then(|v| v.as_str()) == Some("stamp") { Scalar::new(200.0, 0.0, 200.0, 0.0) } else { Scalar::new(0.0, 170.0, 0.0, 0.0) };
        let pts: Vector<Point> = polygon(p).iter().map(|q| Point::new(util::iround(q[0]), util::iround(q[1]))).collect();
        let polys: Vector<Vector<Point>> = vec![pts].into_iter().collect();
        imgproc::polylines(&mut out, &polys, true, color, 3, imgproc::LINE_8, 0)?;
    }
    Ok(out)
}

fn save(dir: &Path, redacted: &Path, n: u64, img: &Mat) -> Result<()> {
    std::fs::create_dir_all(dir)?;
    let stem = redacted.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
    imgcodecs::imwrite(&dir.join(format!("{}.page{}.changes.png", stem, n)), img, &Vector::new())?;
    Ok(())
}

fn decode_gray(data: &[u8]) -> Result<Mat> {
    let buf = Vector::<u8>::from_slice(data);
    Ok(imgcodecs::imdecode(&buf, imgcodecs::IMREAD_GRAYSCALE | imgcodecs::IMREAD_IGNORE_ORIENTATION)?)
}

/// Pixels of `a` and `b` (same size) inside a polygon are identical.
fn same_inside(a: &Mat, b: &Mat, poly: &[[f64; 2]]) -> Result<bool> {
    let mut region = util::zeros_u8(a.rows(), a.cols())?;
    util::fill_poly(&mut region, poly, 255.0)?;
    let (r, av, bv) = (util::bytes(&region)?, util::bytes(a)?, util::bytes(b)?);
    let ch = a.channels() as usize;
    Ok(r.iter().enumerate().all(|(i, &v)| v == 0 || av[i * ch..(i + 1) * ch] == bv[i * ch..(i + 1) * ch]))
}

pub struct VerifyArgs<'a> {
    pub redacted: &'a Path,
    pub manifest: Option<&'a Path>,
    pub original: Option<&'a Path>,
    pub pubkey: Option<&'a Path>,
    pub ledger: Option<&'a Path>,
    pub overlay_dir: Option<&'a Path>,
    pub rerun_templates: Option<&'a Path>,
    /// certificates the PDF's signer must chain to (a self-signed one: itself)
    pub trust: &'a [PathBuf],
}

/// The PDF's digital signature (what PDF readers show), if it has one.
fn pdf_signature(r: &mut Report, pdf: &[u8], m: &Value, trust: &[PathBuf]) -> Result<()> {
    let trust = pdfsign::load_certificates(trust)?;
    let named = m["output"].get("pdf_signature");
    let c = match pdfsign::check(pdf, &trust) {
        Err(e) => {
            r.add(false, format!("digital signature unreadable: {:#}", e));
            return Ok(());
        }
        Ok(None) => {
            if named.is_some() || !trust.is_empty() {
                r.add(false, "the PDF carries no digital signature");
            } else {
                r.notes.push("the PDF is not digitally signed".into());
            }
            return Ok(());
        }
        Ok(Some(c)) => c,
    };
    let who = match &c.certificate {
        Some(x) => pdfsign::describe(x)?,
        None => Value::Null,
    };
    let subject = who["subject"].as_str().unwrap_or("?");
    let fp = who["certificate_sha256"].as_str().unwrap_or("");
    if c.valid && c.whole_file {
        r.add(true, format!("digital signature valid and covering the whole file (signed by {} at {})", subject, c.signed_at));
    } else {
        r.add(false, format!("digital signature: {}", c.error.as_deref().unwrap_or("does not cover the whole file (bytes were added after signing)")));
    }
    if let Some(n) = named {
        r.add(n["certificate_sha256"].as_str() == Some(fp), "the signing certificate is the one the manifest names");
    }
    match c.trusted {
        Some(true) => {
            r.add(true, "the signing certificate is trusted (--trust)");
        }
        Some(false) => {
            r.add(false, "the signing certificate does not chain to any --trust certificate");
        }
        None => r.notes.push(format!(
            "signing certificate{} SHA-256 {}: compare it with the one the publisher announced, or pass it with --trust",
            if who["self_signed"].as_bool() == Some(true) { " (self-signed)" } else { "" },
            fp
        )),
    }
    Ok(())
}

pub fn verify(a: &VerifyArgs) -> Result<Report> {
    let mut r = Report::default();
    let manifest_path: PathBuf = match a.manifest {
        Some(p) => p.to_path_buf(),
        None => a.redacted.with_extension("manifest.json"),
    };
    let m = mf::read_json(&manifest_path)?;
    let pdf = std::fs::read(a.redacted)?;

    // 1. signature and ledger
    if let Some(pk) = a.pubkey {
        let why = mf::check_signature(&m, &mf::load_public(pk)?);
        let ok = why.is_empty();
        r.add(ok, if ok { "manifest signature valid".to_string() } else { format!("manifest signature: {}", why) });
    } else {
        r.notes.push("no public key given: signature not checked".into());
    }
    if let Some(l) = a.ledger {
        let (ok, why, entries) = mf::ledger_check(l)?;
        r.add(ok, if ok { "ledger hash chain intact".to_string() } else { why });
        let h = mf::digest(&m);
        let hit = entries.iter().find(|e| e["manifest_sha256"].as_str() == Some(h.as_str()));
        match hit {
            Some(e) => r.add(true, format!("manifest recorded in ledger (entry {})", e["seq"])),
            None => r.add(false, "manifest not in ledger"),
        };
    }

    // 2. the published file is the one described
    r.add(proof::sha256(&pdf) == m["output"]["pdf_sha256"].as_str().unwrap_or(""), "redacted PDF matches the manifest's SHA-256");
    pdf_signature(&mut r, &pdf, &m, a.trust)?;
    let images = match io::pdf_images(&pdf) {
        Ok(i) => i,
        Err(e) => {
            r.add(false, format!("redacted PDF structure: {}", e));
            return Ok(r);
        }
    };
    let mpages = m["pages"].as_array().cloned().unwrap_or_default();
    if !r.add(images.len() == mpages.len(), format!("{} page image(s), as recorded", images.len())) {
        return Ok(r);
    }

    let mut src_pages = None;
    if let Some(orig) = a.original {
        let data = std::fs::read(orig)?;
        let want = m["source"]["sha256"].as_str().unwrap_or("");
        r.add(proof::sha256(&data) == want, format!("original matches the manifest's source SHA-256 {}...", &want[..want.len().min(16)]));
        let sp = io::source_pages(&orig.to_string_lossy(), Some(&data))?;
        r.add(sp.len() == mpages.len(), "original has the same number of pages");
        src_pages = Some(sp);
    } else {
        r.notes.push(
            "no original given: changed blocks are checked to be black, but equality of all other blocks needs the original (e.g. the copy on the IEBC portal)".into(),
        );
    }

    for (pm, (img_bytes, _)) in mpages.iter().zip(&images) {
        let n = pm["index"].as_u64().unwrap_or(0);
        let declared = proof::unpack(&pm["blocks"])?;
        let is_jpeg = pm["proof"].as_str() == Some("jpeg-dct");
        let kind = if is_jpeg { "jpeg" } else { "pixels" };
        r.add(proof::image_digest(kind, img_bytes)? == pm["output_image_sha256"].as_str().unwrap_or(""), format!("page {}: image matches the manifest", n));
        let red_px = io::decode_bytes(img_bytes)?;
        // blacked blocks must not touch results regions
        let protected = pm["protected"].as_array().cloned().unwrap_or_default();
        let mut prot = util::zeros_u8(red_px.rows(), red_px.cols())?;
        for p in &protected {
            util::fill_poly(&mut prot, &polygon(p), 255.0)?;
        }
        let pb = proof::to_blocks(&prot)?;
        let hits = if pb.rows == declared.rows && pb.cols == declared.cols { declared.and(&pb).count() } else { declared.count() };
        let n_stamps = protected.iter().filter(|p| p.get("kind").and_then(|v| v.as_str()) == Some("stamp")).count();
        let what = format!("{} protected results region(s) and {} stamp(s)", protected.len() - n_stamps, n_stamps);
        r.add(
            hits == 0,
            if hits == 0 { format!("page {}: no blacked block touches the {}", n, what) } else { format!("page {}: {} blacked blocks inside protected results regions or stamps", n, hits) },
        );
        let Some(sps) = &src_pages else {
            let mask = util::bytes(&proof::to_pixels(&declared, red_px.rows(), red_px.cols())?)?;
            let px = util::bytes(&red_px)?;
            let ch = red_px.channels() as usize;
            let dark = mask.iter().enumerate().filter(|&(_, &v)| v != 0).all(|(i, _)| px[i * ch..(i + 1) * ch].iter().all(|&c| c <= 8));
            r.add(dark, format!("page {}: all {} listed blocks are black", n, declared.count()));
            if let Some(d) = a.overlay_dir {
                save(d, a.redacted, n, &overlay(&red_px, &declared, &protected)?)?;
            }
            continue;
        };
        let Some(sp) = sps.get(n as usize - 1) else { continue };
        let (cmp, what) = if is_jpeg {
            if !r.add(sp.kind == "jpeg", format!("page {}: original page is a JPEG, as recorded", n)) {
                continue;
            }
            (proof::compare_jpeg(sp.data.as_ref().unwrap(), img_bytes, Some(&declared))?, "DCT coefficients")
        } else {
            if pm["source_ref"].as_str().unwrap_or("").contains("rendered") {
                r.notes.push(format!("page {}: original was rendered ({}); comparison depends on the same renderer version", n, pm["source_ref"].as_str().unwrap_or("")));
            }
            (proof::compare_pixels(&sp.pixels, &red_px, Some(&declared))?, "pixels")
        };
        r.add(
            cmp.ok,
            if cmp.ok {
                format!(
                    "page {}: {} of {} blocks identical to the original ({}); {} changed, all listed and solid black",
                    n,
                    cmp.identical_blocks,
                    cmp.total_blocks,
                    what,
                    cmp.changed_blocks.count()
                )
            } else {
                format!("page {}: {}", n, cmp.problems.join("; "))
            },
        );
        // Results regions: same decoded pixels in both files, inside each
        // region's polygon (luminance for JPEG: colour is upsampled across
        // block edges by decoders).
        let (ia, ib) = if is_jpeg { (decode_gray(sp.data.as_ref().unwrap())?, decode_gray(img_bytes)?) } else { (sp.pixels.try_clone()?, red_px.try_clone()?) };
        let mut same = ia.rows() == ib.rows() && ia.cols() == ib.cols() && ia.channels() == ib.channels();
        for p in &protected {
            if !same {
                break;
            }
            same = same_inside(&ia, &ib, &polygon(p))?;
        }
        r.add(
            same,
            format!("page {}: results regions and stamps decode to identical pixels in original and redacted{}", n, if is_jpeg { " (luminance)" } else { "" }),
        );
        if let Some(d) = a.overlay_dir {
            save(d, a.redacted, n, &overlay(&sp.pixels, &cmp.changed_blocks, &protected)?)?;
        }
    }

    if let (Some(tdir), Some(orig)) = (a.rerun_templates, a.original) {
        let st = &m["settings"];
        // the re-run's own output is a redacted copy of the form: it goes to
        // a folder of its own, removed afterwards, never next to the user's files
        let work = std::env::temp_dir().join(format!("rfpii-rerun-{}", std::process::id()));
        let mut ms = MaskSettings::default();
        ms.hide = st["hide"].as_u64().unwrap_or(3) as usize;
        ms.phone_tail = st["phone_tail"].as_u64().unwrap_or(6) as usize;
        let cfg = Config {
            templates: tdir.to_string_lossy().into_owned(),
            out: work.join("out").to_string_lossy().into_owned(),
            review: work.join("review").to_string_lossy().into_owned(),
            align: AlignSettings::default(),
            mask: ms,
            strict_structure: st["strict_structure"].as_bool().unwrap_or(false),
            generic: st["generic"].as_bool().unwrap_or(false),
            vlm_audit: false,
            vlm_fallback: false,
            pdf_signer: None,
        };
        if let Some(o) = m.get("ocr") {
            let now = crate::ocr::info().map(|i| i["models"].clone()).unwrap_or(Value::Null);
            r.add(now == o["models"], "OCR models are the versions recorded");
        }
        let mut templates = load_templates(tdir)?;
        for t in templates.iter_mut() {
            prepare(t, &cfg.align)?;
        }
        let fp_ok = m["templates"].as_object().map(|o| o.iter().all(|(k, v)| templates.iter().any(|t| &t.name == k && Some(t.fingerprint.as_str()) == v.as_str()))).unwrap_or(true);
        r.add(fp_ok, "templates are the versions recorded");
        let seed = u64::from_str_radix(m["seed"].as_str().unwrap_or("0"), 16)?;
        let o = process_document(&orig.to_string_lossy(), m["source"]["path"].as_str().unwrap_or(""), &templates, &cfg, None, Some(seed), None);
        let _ = std::fs::remove_dir_all(&work);
        let o = o?;
        let digests = |pages: &Value| -> Vec<String> { pages.as_array().map(|a| a.iter().map(|p| p["output_image_sha256"].as_str().unwrap_or("").to_string()).collect()).unwrap_or_default() };
        let same = o.ok && o.manifest.as_ref().map(|om| digests(&om["pages"]) == digests(&m["pages"])).unwrap_or(false);
        r.add(same, "re-running the recorded code, templates and seed reproduces the same output");
    }
    Ok(r)
}
