//! One document, start to finish, with the evidence to back it.
//!
//! Used by `rfpii redact` (on worker threads) and by `rfpii verify --rerun`,
//! which repeats a redaction from its manifest and must get the same result.

use anyhow::Result;
use opencv::core::Mat;
use opencv::prelude::*;
use serde_json::{json, Map, Value};
use std::path::{Path, PathBuf};

use crate::align::AlignSettings;
use crate::io::{self, PageImage, SourcePage};
use crate::kiems;
use crate::masking::MaskSettings;
use crate::pipeline::{redact_pages, PageOptions, PageResult};
use crate::proof;
use crate::pyjson::{np_round, round_n};
use crate::pyrand::PyRandom;
use crate::templates::Template;
use crate::util;

pub const SCHEMA: &str = "rfpii/manifest/1";
pub const TOOL_NAME: &str = "resultform-pii";

#[derive(Clone)]
pub struct Config {
    pub templates: String,
    pub out: String,
    pub review: String,
    pub align: AlignSettings,
    pub mask: MaskSettings,
    pub strict_structure: bool,
    pub generic: bool, // read pages no template matches
    pub vlm_audit: bool,
    pub vlm_fallback: bool,
    /// digitally sign the published PDFs with this certificate
    pub pdf_signer: Option<std::sync::Arc<crate::pdfsign::Signer>>,
}

/// Version and exact code revision, so a scrutineer can run the same code.
pub fn tool_info() -> Value {
    let commit = option_env!("RFPII_COMMIT").filter(|c| !c.is_empty());
    json!({"name": TOOL_NAME, "version": env!("CARGO_PKG_VERSION"), "commit": commit})
}

/// Checks across the pages of one document: one form type; exactly one page
/// matching a `once` template (the Form 34B declaration block) per form.
pub fn document_rules(pages: &[PageResult], templates: &[Template]) -> Vec<String> {
    let used: Vec<Option<&Template>> = pages.iter().filter(|p| p.ok).map(|p| templates.iter().find(|t| t.name == p.template)).collect();
    let mut forms: Vec<String> = used.iter().flatten().filter(|t| !t.form.is_empty()).map(|t| t.form.clone()).collect();
    forms.sort();
    forms.dedup();
    let mut problems = Vec::new();
    if forms.len() > 1 {
        problems.push(format!("pages match different forms: {}", forms.join(", ")));
    }
    for form in &forms {
        let blocks: Vec<&Template> = templates.iter().filter(|t| t.once && &t.form == form).collect();
        if blocks.is_empty() {
            continue;
        }
        let found: Vec<String> = used.iter().flatten().filter(|u| u.once && &u.form == form).map(|u| u.name.clone()).collect();
        if found.len() != 1 {
            let names = if found.is_empty() { blocks.iter().map(|t| t.name.clone()).collect::<Vec<_>>().join(" / ") } else { found.join(", ") };
            problems.push(format!("Form {}: declaration block ({}) found on {} pages (expected exactly 1)", form, names, found.len()));
        }
    }
    problems
}

pub struct Outcome {
    pub ok: bool,
    pub report: Value,           // for review/report.jsonl
    pub manifest: Option<Value>, // unsigned; signed by the caller
    pub pdf: Option<Vec<u8>>,
    pub redacted_pages: Vec<Mat>,
    pub source_pages: Vec<SourcePage>,
}

/// `datetime.now(timezone.utc).isoformat(timespec="seconds")`.
pub fn utc_now() -> String {
    let secs = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs() as i64).unwrap_or(0);
    let (days, rem) = (secs.div_euclid(86400), secs.rem_euclid(86400));
    // civil from days (H. Hinnant)
    let z = days + 719468;
    let era = z.div_euclid(146097);
    let doe = z - era * 146097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = yoe + era * 400 + (m <= 2) as i64;
    format!("{:04}-{:02}-{:02}T{:02}:{:02}:{:02}+00:00", y, m, d, rem / 3600, rem % 3600 / 60, rem % 60)
}

/// `Path(rel).with_suffix(ext)`.
pub fn with_suffix(rel: &str, ext: &str) -> PathBuf {
    let p = Path::new(rel);
    let stem = p.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
    let name = format!("{}{}", stem, ext);
    match p.parent() {
        Some(par) if !par.as_os_str().is_empty() => par.join(name),
        _ => PathBuf::from(name),
    }
}

fn new_seed() -> Result<u64> {
    let mut b = [0u8; 8];
    crate::manifest::getrandom_fill(&mut b)?;
    Ok(u64::from_le_bytes(b) >> 1) // 63 bits
}

fn page_report(p: &PageResult) -> Value {
    let count = |m: &str| p.masks.iter().filter(|r| r.get("mode").and_then(|v| v.as_str()) == Some(m)).count();
    let mut v = json!({"template": p.template, "ok": p.ok, "inliers": p.inliers, "score": round_n(p.score, 3),
                       "reason": p.reason, "ms": round_n(p.ms, 1),
                       "cells": {"partial": count("partial"), "full": count("full"), "empty": count("empty")}});
    if !p.notes.is_empty() {
        v["notes"] = json!(p.notes);
    }
    v
}

pub fn process_document(src: &str, rel: &str, templates: &[Template], cfg: &Config, vlm: Option<&crate::vlm::Ollama>, seed: Option<u64>, data: Option<Vec<u8>>) -> Result<Outcome> {
    let data = match data {
        Some(d) => d,
        None => io::read_bytes(src)?,
    };
    let source_sha = proof::sha256(&data);
    let mut report = json!({"file": src, "ok": false, "sha256": source_sha});
    let pages = io::source_pages(src, Some(&data))?;
    if pages.is_empty() {
        report["reason"] = json!("no pages");
        return Ok(Outcome { ok: false, report, manifest: None, pdf: None, redacted_pages: vec![], source_pages: vec![] });
    }

    // Recorded seed: the hidden characters are random, and the same seed
    // (with the same code, templates and input) reproduces the same output.
    let seed = match seed {
        Some(s) => s,
        None => new_seed()?,
    };
    let mut rng = PyRandom::new(seed);
    opencv::core::set_rng_seed((seed % (1u64 << 31)) as i32)?;

    let stem = Path::new(rel).file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
    let name = kiems::parse_name(&stem);
    let payload = kiems::decode_qr(&util::gray(&pages[0].pixels)?);
    let qr = payload.as_deref().map(kiems::parse_qr).unwrap_or_else(|| json!({}));
    let qr_empty = qr.as_object().map(|m| m.is_empty()).unwrap_or(true);
    let form: Option<String> = qr.get("form").and_then(|v| v.as_str()).filter(|s| !s.is_empty()).map(String::from).or_else(|| name.as_ref().map(|n| n.form.clone()));
    let meta = json!({"form": form, "kiems": name.as_ref().map(|n| n.to_value()), "qr": payload,
                      "qr_fields": if qr_empty { Value::Null } else { qr.clone() }});
    for (k, v) in meta.as_object().unwrap() {
        report[k] = v.clone();
    }

    let pixels: Vec<Mat> = pages.iter().map(|p| p.pixels.clone()).collect();
    let po = PageOptions { form: form.clone(), strict: cfg.strict_structure, generic: cfg.generic, ..Default::default() };
    let doc = redact_pages(&pixels, templates, &cfg.align, &cfg.mask, &mut rng, &po)?;
    let mut problems: Vec<String> = doc.iter().enumerate().filter(|(_, p)| !p.ok).map(|(i, p)| format!("page {}: {}", i + 1, p.reason)).collect();
    let mismatch = kiems::check(name.as_ref(), &qr);
    if !mismatch.is_empty() {
        problems.push(mismatch);
    }
    problems.extend(document_rules(&doc, templates));

    let mut audits: Vec<Value> = Vec::new();
    if let (Some(v), true, true) = (vlm, cfg.vlm_audit, problems.is_empty()) {
        for (i, p) in doc.iter().enumerate() {
            let a = v.audit(p.image.as_ref().unwrap())?; // only ever the *redacted* page
            let mut rec = json!({"page": i + 1});
            for (k, x) in a.record.as_object().unwrap() {
                rec[k] = x.clone();
            }
            audits.push(rec);
            if !a.ok {
                problems.push(format!("page {}: VLM audit flagged: {}", i + 1, a.findings.get("details").and_then(|d| d.as_str()).unwrap_or("")));
            }
        }
    }

    report["pages"] = Value::Array(doc.iter().map(page_report).collect());
    if !problems.is_empty() {
        report["reason"] = json!(problems.join("; "));
        let redacted = doc.iter().filter_map(|p| p.image.clone()).collect();
        return Ok(Outcome { ok: false, report, manifest: None, pdf: None, redacted_pages: redacted, source_pages: pages });
    }

    // Apply the block grid to the *stored* image and check the result bit for
    // bit before anything is written: nothing but the listed blocks, nothing
    // but black.
    let mut reds = Vec::new();
    let mut mpages = Vec::new();
    for (i, (sp, p)) in pages.iter().zip(&doc).enumerate() {
        let blocks = p.blocks.as_ref().unwrap();
        let (red, cmp) = if sp.kind == "jpeg" {
            let d = sp.data.as_ref().unwrap();
            let red = proof::redact_jpeg(d, blocks)?;
            let cmp = proof::compare_jpeg(d, &red.data, Some(&red.luma_blocks))?;
            (red, cmp)
        } else {
            let red = proof::redact_pixels(&sp.pixels, blocks)?;
            let cmp = proof::compare_pixels(&sp.pixels, &io::decode_bytes(&red.data)?, Some(&red.luma_blocks))?;
            (red, cmp)
        };
        if !cmp.ok {
            report["reason"] = json!(format!("page {}: self-check failed: {}", i + 1, cmp.problems.join("; ")));
            return Ok(Outcome { ok: false, report, manifest: None, pdf: None, redacted_pages: vec![], source_pages: vec![] });
        }
        let (h, w) = (sp.pixels.rows(), sp.pixels.cols());
        let src_sha = match &sp.data {
            Some(d) => proof::sha256(d),
            None => proof::sha256(&util::bytes(&sp.pixels)?),
        };
        let hv = p.h.map(|m| json!(m.iter().map(|r| r.iter().map(|&x| np_round(x, 8)).collect::<Vec<_>>()).collect::<Vec<_>>()));
        let mut mp = json!({
            "index": i + 1,
            "source_ref": sp.reference,
            "proof": if sp.kind == "jpeg" { "jpeg-dct" } else { "pixels" },
            "source_image_sha256": src_sha,
            "output_image_sha256": proof::image_digest(red.kind, &red.data)?,
            "width": w, "height": h, "block": proof::BLOCK,
            "blocks": proof::pack(&red.luma_blocks),
            "blocks_blacked": red.luma_blocks.count(),
            "blocks_unchanged": cmp.identical_blocks,
            "protected": p.protected,
            "template": p.template,
            "alignment": {"inliers": p.inliers, "score": round_n(p.score, 4), "H": hv},
            "notes": p.notes,
            "masks": p.masks,
        });
        if let Some(l) = &p.layout {
            mp["layout"] = l.clone();
        }
        mpages.push(mp);
        reds.push(red);
    }
    let out_rel = with_suffix(rel, ".pdf");
    let out_path = Path::new(&cfg.out).join(&out_rel);
    if let Some(par) = out_path.parent() {
        std::fs::create_dir_all(par)?;
    }
    let decoded: Vec<Option<Mat>> = reds.iter().map(|r| if r.kind == "jpeg" { Ok(None) } else { io::decode_bytes(&r.data).map(Some) }).collect::<Result<_>>()?;
    let items: Vec<_> = reds
        .iter()
        .zip(&decoded)
        .zip(&pages)
        .map(|((r, dm), sp)| {
            let img = match dm {
                Some(m) => PageImage::Pixels(m),
                None => PageImage::Jpeg(r.data.as_slice()),
            };
            (img, sp.page_size, sp.rect, sp.rotate)
        })
        .collect();
    let pdf = io::build_pdf(&items)?;
    let created = utc_now();
    let pdf = match &cfg.pdf_signer {
        Some(s) => crate::pdfsign::sign_pdf(&pdf, s, &created)?,
        None => pdf,
    };
    std::fs::write(&out_path, &pdf)?;

    let mut used: Vec<&str> = doc.iter().map(|p| p.template.as_str()).collect();
    used.dedup();
    let mut tmap = Map::new();
    for t in templates {
        if used.contains(&t.name.as_str()) {
            tmap.insert(t.name.clone(), json!(t.fingerprint));
        }
    }
    let mut source = json!({"name": Path::new(rel).file_name().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default(),
                            "path": rel, "sha256": source_sha});
    for (k, v) in meta.as_object().unwrap() {
        source[k] = v.clone();
    }
    let manifest = json!({
        "schema": SCHEMA,
        "tool": tool_info(),
        "created_at": created,
        "source": source,
        "output": {"file": out_rel.to_string_lossy(), "pdf_sha256": proof::sha256(&pdf)},
        "seed": format!("{:x}", seed),
        "settings": {"hide": cfg.mask.hide, "phone_tail": cfg.mask.phone_tail,
                     "strict_structure": cfg.strict_structure, "generic": cfg.generic},
        "templates": tmap,
        "pages": mpages,
        "vlm_audit": if audits.is_empty() { Value::Null } else { Value::Array(audits) },
    });
    let mut manifest = manifest;
    // OCR read the page (no template), or a template's free-text areas
    if used.contains(&"generic") || mpages.iter().any(|p| p["layout"].get("free_text").is_some()) {
        manifest["ocr"] = crate::ocr::info()?;
    }
    if let Some(s) = &cfg.pdf_signer {
        manifest["output"]["pdf_signature"] = crate::pdfsign::describe(s.certificate())?;
    }
    report["ok"] = json!(true);
    report["output"] = json!(out_path.to_string_lossy());
    Ok(Outcome { ok: true, report, manifest: Some(manifest), pdf: Some(pdf), redacted_pages: vec![], source_pages: vec![] })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn paths_and_time() {
        assert_eq!(with_suffix("a/b/x.PDF", ".pdf"), PathBuf::from("a/b/x.pdf"));
        assert_eq!(with_suffix("x.tar.gz", ".pdf"), PathBuf::from("x.tar.pdf"));
        assert_eq!(with_suffix("noext", ".json"), PathBuf::from("noext.json"));
        let t = utc_now();
        assert_eq!(t.len(), 25);
        assert!(t.ends_with("+00:00"));
    }
}

#[cfg(test)]
#[path = "tests/process.rs"]
mod py_tests;
