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
use crate::pipeline::{redact_pages, Found, PageOptions, PageResult};
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
    /// review every redacted page with `reviewer` before publishing
    pub vlm_audit: bool,
    pub vlm_fallback: bool,
    pub reviewer: Option<std::sync::Arc<crate::vlm::Reviewer>>,
    /// a page whose columns couldn't be placed: those columns hidden whole,
    /// published only if the review then finds nothing
    pub vlm_rescue: bool,
    /// what the review finds is masked and the page reviewed again, this
    /// many times at most
    pub vlm_rounds: usize,
    /// digitally sign the published PDFs with this certificate
    pub pdf_signer: Option<std::sync::Arc<crate::pdfsign::Signer>>,
    /// a decision model (Clef) asked which way up each published page of a
    /// photo with no EXIF orientation tag stands
    pub orient: Option<std::sync::Arc<crate::vlm::Ollama>>,
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

/// Where a document's output goes, under the output folder: its source's
/// path with `ext`; for one page of a source, with ".p012" before it.
pub fn output_rel(rel: &str, page: Option<usize>, ext: &str) -> PathBuf {
    match page {
        None => with_suffix(rel, ext),
        Some(k) => with_suffix(rel, &format!(".p{:03}{}", k, ext)),
    }
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

/// The part of a page the review looks at: the template's `review_area`
/// where it has one, else the whole page. Image px x, y, w, h.
fn review_area(p: &PageResult, templates: &[Template], w: i32, h: i32) -> [i32; 4] {
    let t = templates.iter().find(|t| t.name == p.template);
    if let (Some([x, y, rw, rh]), Some(hh)) = (t.and_then(|t| t.review_area), p.h) {
        let c: Vec<[f64; 2]> = [[x, y], [x + rw, y], [x + rw, y + rh], [x, y + rh]].iter().map(|&q| util::project(&hh, q)).collect();
        let x0 = c.iter().map(|q| q[0]).fold(f64::INFINITY, f64::min).max(0.0) as i32;
        let y0 = c.iter().map(|q| q[1]).fold(f64::INFINITY, f64::min).max(0.0) as i32;
        let x1 = (c.iter().map(|q| q[0]).fold(f64::NEG_INFINITY, f64::max).ceil() as i32).min(w);
        let y1 = (c.iter().map(|q| q[1]).fold(f64::NEG_INFINITY, f64::max).ceil() as i32).min(h);
        if x1 - x0 > 32 && y1 - y0 > 32 {
            return [x0, y0, x1 - x0, y1 - y0];
        }
    }
    [0, 0, w, h]
}

/// Share of `f`'s box inside the page's protected stamps.
fn on_stamps(f: &Found, protected: &[Value]) -> f64 {
    let circles: Vec<[f64; 3]> = protected
        .iter()
        .filter(|p| p.get("kind").and_then(|k| k.as_str()) == Some("stamp"))
        .filter_map(|p| {
            let c: Vec<f64> = p.get("circle")?.as_array()?.iter().filter_map(|v| v.as_f64()).collect();
            (c.len() == 3).then(|| [c[0], c[1], c[2]])
        })
        .collect();
    let [x0, y0, x1, y1] = f.rect;
    let mut inside = 0;
    for i in 0..10 {
        for j in 0..10 {
            let (x, y) = (x0 + (x1 - x0) * (i as f64 + 0.5) / 10.0, y0 + (y1 - y0) * (j as f64 + 0.5) / 10.0);
            if circles.iter().any(|c| (x - c[0]).hypot(y - c[1]) <= c[2]) {
                inside += 1;
            }
        }
    }
    inside as f64 / 100.0
}

/// `page`: only that page of the source (from 1) is the document (a
/// scanned batch of many forms, each page published on its own).
///
/// `replay`: the manifest of an earlier run of this document, whose review
/// findings, columns hidden whole and turns are applied again instead of
/// asking any model (a re-run must give the same output).
#[allow(clippy::too_many_arguments)]
pub fn process_document(src: &str, rel: &str, page: Option<usize>, templates: &[Template], cfg: &Config, seed: Option<u64>, data: Option<Vec<u8>>, replay: Option<&Value>) -> Result<Outcome> {
    let data = match data {
        Some(d) => d,
        None => io::read_bytes(src)?,
    };
    let source_sha = proof::sha256(&data);
    let mut report = json!({"file": src, "ok": false, "sha256": source_sha});
    let (pages, of) = match page {
        None => (io::source_pages(src, Some(&data))?, None),
        Some(k) => {
            report["page"] = json!(k);
            let (p, n) = io::source_page(src, &data, k)?;
            (vec![p], Some(n))
        }
    };
    // pages as the source numbers them
    let number = |i: usize| page.unwrap_or(i + 1);
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
    let n = pixels.len();
    let mut found: Vec<Vec<Found>> = vec![vec![]; n];
    let mut whole: Vec<Vec<String>> = vec![vec![]; n];
    if let Some(m) = replay {
        for (i, p) in m["pages"].as_array().into_iter().flatten().enumerate().take(n) {
            found[i] = p["vlm_found"].as_array().map(|a| a.iter().filter_map(Found::from_value).collect()).unwrap_or_default();
            whole[i] = p["whole_columns"].as_array().map(|a| a.iter().filter_map(|v| v.as_str().map(String::from)).collect()).unwrap_or_default();
        }
    }
    // Every pass starts from the seed: the output depends only on the seed
    // and what is masked, however many passes the review took.
    let run = |found: &[Vec<Found>], whole: &[Vec<String>]| -> Result<Vec<PageResult>> {
        let mut rng = PyRandom::new(seed);
        opencv::core::set_rng_seed((seed % (1u64 << 31)) as i32)?;
        let po = PageOptions { form: form.clone(), strict: cfg.strict_structure, generic: cfg.generic, found: found.to_vec(), whole: whole.to_vec(), ..Default::default() };
        redact_pages(&pixels, templates, &cfg.align, &cfg.mask, &mut rng, &po)
    };
    let mut doc = run(&found, &whole)?;

    // The review sees only redacted pages. What it finds still showing is
    // masked (signatures whole, numbers as any number) and the pages are
    // reviewed again; a page is published once a review finds nothing.
    let mut reviews: Vec<Value> = Vec::new();
    let mut unresolved: Vec<String> = Vec::new();
    if let (Some(rv), true, None) = (cfg.reviewer.as_deref(), cfg.vlm_audit, replay) {
        // Pages held only because a column's place or masking couldn't be
        // confirmed: those columns (and their table's signatures) hidden
        // whole, then reviewed like any other.
        if cfg.vlm_rescue && doc.iter().any(|p| !p.ok) && doc.iter().all(|p| p.ok || !p.unsure.is_empty()) {
            for (i, p) in doc.iter().enumerate() {
                if !p.ok {
                    whole[i] = p.unsure.clone();
                }
            }
            doc = run(&found, &whole)?;
        }
        if doc.iter().all(|p| p.ok) {
            let mut clean = false;
            for round in 0..=cfg.vlm_rounds {
                let mut fresh = false;
                for (i, p) in doc.iter().enumerate() {
                    let img = p.image.as_ref().unwrap();
                    let r = rv.finder.find(img, review_area(p, templates, img.cols(), img.rows()), &rv.checker)?;
                    let mut rec = json!({"page": i + 1, "round": round + 1});
                    for (k, x) in r.record.as_object().unwrap() {
                        rec[k] = x.clone();
                    }
                    // writing on a stamp stays visible (the stamp rule lifts
                    // any mask there): not the review's to mask
                    let (stamped, found_now): (Vec<Found>, Vec<Found>) = r.found.into_iter().partition(|f| on_stamps(f, &p.protected) >= 0.5);
                    if !stamped.is_empty() {
                        rec["on_stamps"] = json!(stamped.iter().map(|f| f.to_value()).collect::<Vec<_>>());
                    }
                    reviews.push(rec);
                    for f in found_now {
                        let (cx, cy) = ((f.rect[0] + f.rect[2]) / 2.0, (f.rect[1] + f.rect[3]) / 2.0);
                        if !found[i].iter().any(|g| g.rect[0] - 4.0 <= cx && cx <= g.rect[2] + 4.0 && g.rect[1] - 4.0 <= cy && cy <= g.rect[3] + 4.0) {
                            found[i].push(f);
                            fresh = true;
                        }
                    }
                }
                if !fresh {
                    clean = true;
                    break;
                }
                if round == cfg.vlm_rounds {
                    break;
                }
                doc = run(&found, &whole)?;
                if !doc.iter().all(|p| p.ok) {
                    break;
                }
            }
            if !clean && doc.iter().all(|p| p.ok) {
                unresolved.push(format!("the vision review still finds PII after {} round(s) of masking it", cfg.vlm_rounds));
            }
        }
    }

    let mut problems: Vec<String> = doc.iter().enumerate().filter(|(_, p)| !p.ok).map(|(i, p)| format!("page {}: {}", number(i), p.reason)).collect();
    let mismatch = kiems::check(name.as_ref(), &qr);
    if !mismatch.is_empty() {
        problems.push(mismatch);
    }
    problems.extend(document_rules(&doc, templates));
    problems.extend(unresolved);

    report["pages"] = Value::Array(doc.iter().map(page_report).collect());
    if !problems.is_empty() {
        report["reason"] = json!(problems.join("; "));
        if !reviews.is_empty() {
            report["vlm_review"] = Value::Array(reviews);
        }
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
            report["reason"] = json!(format!("page {}: self-check failed: {}", number(i), cmp.problems.join("; ")));
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
        if !found[i].is_empty() {
            mp["vlm_found"] = json!(found[i].iter().map(|f| f.to_value()).collect::<Vec<_>>());
        }
        if !whole[i].is_empty() {
            mp["whole_columns"] = json!(whole[i]);
        }
        mpages.push(mp);
        reds.push(red);
    }
    let out_rel = output_rel(rel, page, ".pdf");
    let out_path = Path::new(&cfg.out).join(&out_rel);
    if let Some(par) = out_path.parent() {
        std::fs::create_dir_all(par)?;
    }
    let decoded: Vec<Option<Mat>> = reds.iter().map(|r| if r.kind == "jpeg" { Ok(None) } else { io::decode_bytes(&r.data).map(Some) }).collect::<Result<_>>()?;
    // A photo with no orientation tag is shown as stored, which may be on
    // its side: Clef is asked which way up the page stands, as published
    // (redacted), and the page is turned to show it upright.
    let mut turns = vec![0i64; pages.len()];
    for (i, sp) in pages.iter().enumerate() {
        let rec = match (replay, cfg.orient.as_deref()) {
            (Some(m), _) => m["pages"][i].get("orientation").cloned(),
            (None, Some(clef)) if sp.untagged => {
                let img = match &decoded[i] {
                    Some(m) => m.clone(),
                    None => io::decode_bytes(&reds[i].data)?,
                };
                Some(clef.orient(&img)?.1)
            }
            _ => None,
        };
        if let Some(r) = rec {
            turns[i] = r["turn"].as_i64().unwrap_or(0);
            if turns[i] != 0 {
                report["pages"][i]["turned"] = json!(turns[i]);
            }
            mpages[i]["orientation"] = r;
        }
    }
    let items: Vec<_> = reds
        .iter()
        .zip(&decoded)
        .zip(&pages)
        .zip(&turns)
        .map(|(((r, dm), sp), turn)| {
            let img = match dm {
                Some(m) => PageImage::Pixels(m),
                None => PageImage::Jpeg(r.data.as_slice()),
            };
            (img, sp.page_size, sp.place, (sp.rotate + turn).rem_euclid(360))
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
    // one page of the source published on its own: which, of how many
    if let (Some(k), Some(n)) = (page, of) {
        source["page"] = json!(k);
        source["pages"] = json!(n);
    }
    let manifest = json!({
        "schema": SCHEMA,
        "tool": tool_info(),
        "created_at": created,
        "source": source,
        "output": {"file": out_rel.to_string_lossy(), "pdf_sha256": proof::sha256(&pdf)},
        "seed": format!("{:x}", seed),
        "settings": {"hide": cfg.mask.hide, "phone_tail": cfg.mask.phone_tail,
                     "strict_structure": cfg.strict_structure, "generic": cfg.generic,
                     "align": cfg.align.to_json()},
        "templates": tmap,
        "pages": mpages,
        "vlm_review": if reviews.is_empty() { Value::Null } else { Value::Array(reviews) },
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
    fn spots_on_a_stamp_are_measured_by_their_area_inside_it() {
        let stamp = [json!({"kind": "stamp", "circle": [100.0, 100.0, 50.0]}), json!({"kind": "results", "rect": [0, 0, 400, 400]})];
        let spot = |x0: f64, x1: f64| Found { label: "signature".into(), rect: [x0, 90.0, x1, 110.0] };
        assert_eq!(on_stamps(&spot(80.0, 120.0), &stamp), 1.0);
        assert_eq!(on_stamps(&spot(160.0, 200.0), &stamp), 0.0);
        let half = on_stamps(&spot(100.0, 200.0), &stamp);
        assert!(0.4 < half && half < 0.6, "{half}");
        assert_eq!(on_stamps(&spot(80.0, 120.0), &stamp[1..]), 0.0); // only stamps count
    }

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
