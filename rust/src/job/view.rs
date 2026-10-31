//! The only two ways a page's pixels are given to a model (HARNESS.md §7):
//!
//! crop       a published page, rendered from its PDF once `verify` passes
//!            on it; it takes a manifest, never a path to any other file;
//! held_view  a held page, only in a job that lets held pages go to the
//!            cloud model: a triage view (the page with every mask the
//!            redactor placed, 512 px long, too small to read handwriting),
//!            or, to locate its fields, the page itself at full resolution.
//!
//! Each returns the image as it is to be sent, with its SHA-256. A job
//! registers each with its guard (`vlm::Guard`), which checks every image a
//! request carries against those registered.

use anyhow::{bail, Context, Result};
use opencv::core::Mat;
use opencv::imgproc;
use opencv::prelude::*;
use serde_json::Value;
use std::path::{Path, PathBuf};

use crate::align::AlignSettings;
use crate::io;
use crate::masking::MaskSettings;
use crate::pipeline::{redact_page, PageOptions};
use crate::pyrand::PyRandom;
use crate::templates::Template;
use crate::util;

/// Long side of a triage view, px: too small to read handwriting on a form
/// (to be confirmed by measurement, HARNESS.md §11).
pub const TRIAGE_SIDE: i32 = 512;

/// An image for a model, as it is to be sent.
pub struct View {
    pub image: Mat,
    pub encoded: Vec<u8>,
    pub sha256: String,
    /// where the image lies on the page: x, y (page px) and the scale from
    /// page px to image px
    pub origin: [f64; 3],
}

impl View {
    fn new(image: Mat, origin: [f64; 3], format: &str) -> Result<View> {
        let encoded = io::encode(&image, format, 90)?;
        let sha256 = crate::proof::sha256(&encoded);
        Ok(View { image, encoded, sha256, origin })
    }

    /// A box in image px (x0, y0, x1, y1) as page px.
    pub fn to_page(&self, r: [f64; 4]) -> [f64; 4] {
        let [x, y, s] = self.origin;
        [x + r[0] / s, y + r[1] / s, x + r[2] / s, y + r[3] / s]
    }
}

/// `img` with its long side at most `max_side`, and the scale it was shrunk by.
fn shrink(img: &Mat, max_side: i32) -> Result<(Mat, f64)> {
    let f = (max_side as f64 / img.cols().max(img.rows()) as f64).min(1.0);
    if f >= 1.0 {
        return Ok((img.try_clone()?, 1.0));
    }
    Ok((util::resize_by(img, f, imgproc::INTER_AREA)?, f))
}

/// The part of a published page a crop shows.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Region {
    Page,
    /// from the declaration down: the template's `review_area`
    Declaration,
    /// the agents' table: the template's `agents` structure
    Agents,
}

impl Region {
    pub fn parse(s: &str) -> Result<Region> {
        Ok(match s {
            "page" => Region::Page,
            "declaration" => Region::Declaration,
            "agents" => Region::Agents,
            _ => bail!("region must be page, declaration or agents"),
        })
    }

    pub fn name(self) -> &'static str {
        match self {
            Region::Page => "page",
            Region::Declaration => "declaration",
            Region::Agents => "agents",
        }
    }
}

/// The published PDF beside a manifest. Anything else is refused: an
/// original, a held page's report, a PDF named directly.
pub fn published_pdf(manifest: &Path) -> Result<(PathBuf, Value)> {
    let name = manifest.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
    let Some(stem) = name.strip_suffix(".manifest.json") else {
        bail!("refusing {}: crop takes a published page's manifest (NAME.manifest.json beside NAME.pdf)", manifest.display());
    };
    let m = crate::manifest::read_json(manifest).context("refusing: not a manifest")?;
    if m["schema"] != crate::process::SCHEMA {
        bail!("refusing {}: not a manifest ({})", manifest.display(), crate::process::SCHEMA);
    }
    let pdf = manifest.with_file_name(format!("{}.pdf", stem));
    if !pdf.exists() {
        bail!("refusing {}: no published PDF beside it", manifest.display());
    }
    Ok((pdf, m))
}

pub struct CropArgs<'a> {
    pub manifest: &'a Path,
    pub region: Region,
    /// page of the document, from 1
    pub page: usize,
    /// templates, for where the regions are (the whole page without)
    pub templates: &'a [Template],
    pub max_side: i32,
    /// the publisher's key, if the manifest's signature is to be checked too
    pub pubkey: Option<&'a Path>,
    /// a folder the manifest must be in (a job's `staging/` or `public/`)
    pub within: Option<&'a Path>,
    /// "jpg" or "png"
    pub format: &'a str,
}

/// The rectangle of `region` on page `p` (a manifest's page record), image
/// px x, y, w, h: from its template and alignment, else the whole page.
fn region_rect(region: Region, p: &Value, templates: &[Template], w: i32, h: i32) -> [i32; 4] {
    let t = templates.iter().find(|t| p["template"].as_str() == Some(t.name.as_str()));
    let hh: Option<[[f64; 3]; 3]> = serde_json::from_value(p["alignment"]["H"].clone()).ok();
    let rect = match (region, t) {
        (Region::Page, _) | (_, None) => None,
        (Region::Declaration, Some(t)) => t.review_area,
        (Region::Agents, Some(t)) => t.structure.iter().find(|(n, _)| n == "agents").map(|(_, r)| *r).or(t.review_area),
    };
    match (rect, hh) {
        (Some(r), Some(hh)) => crate::process::area_on_page(r, &hh, w, h).unwrap_or([0, 0, w, h]),
        _ => [0, 0, w, h],
    }
}

/// A region of a published page, for a model: refused unless the manifest
/// is beside its PDF (and within `within`), and `verify` passes on them
/// without the original (the PDF is the one recorded, every listed block is
/// black, none touches a protected region, the notice is there).
pub fn crop(a: &CropArgs) -> Result<View> {
    if let Some(root) = a.within {
        let (m, r) = (a.manifest.canonicalize()?, root.canonicalize()?);
        if !m.starts_with(&r) {
            bail!("refusing {}: not under {}", a.manifest.display(), root.display());
        }
    }
    let (pdf, m) = published_pdf(a.manifest)?;
    let rep = crate::verify::verify(&crate::verify::VerifyArgs {
        redacted: &pdf,
        manifest: Some(a.manifest),
        original: None,
        pubkey: a.pubkey,
        ledger: None,
        overlay_dir: None,
        rerun_templates: None,
        trust: &[],
    })?;
    if !rep.ok() {
        let failed: Vec<&str> = rep.checks.iter().filter(|(ok, _)| !ok).map(|(_, m)| m.as_str()).collect();
        bail!("refusing {}: verify fails: {}", pdf.display(), failed.join("; "));
    }
    let images = io::pdf_images(&std::fs::read(&pdf)?)?;
    let Some((bytes, _)) = images.get(a.page.max(1) - 1) else { bail!("{} has no page {}", pdf.display(), a.page) };
    let img = io::decode_bytes(bytes)?;
    let [x, y, w, h] = region_rect(a.region, &m["pages"][a.page.max(1) - 1], a.templates, img.cols(), img.rows());
    let cut = Mat::roi(&img, opencv::core::Rect::new(x, y, w, h))?.try_clone()?;
    let (small, f) = shrink(&cut, a.max_side)?;
    View::new(small, [x as f64, y as f64, f], a.format)
}

/// What a held page is shown for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    /// what kind of page it is: a triage view
    Triage,
    /// where its fields are, on a page triage found to be the right form:
    /// the page itself
    Locate,
}

impl Mode {
    pub fn parse(s: &str) -> Result<Mode> {
        Ok(match s {
            "triage" => Mode::Triage,
            "locate" => Mode::Locate,
            _ => bail!("mode must be triage or locate"),
        })
    }

    pub fn name(self) -> &'static str {
        match self {
            Mode::Triage => "triage",
            Mode::Locate => "locate",
        }
    }
}

/// A held page with every mask the redactor can place: read as `redact`
/// reads it, then, if it fails, read again with the gates off and the
/// columns it couldn't confirm hidden whole (as `examples/held_dbg.rs`
/// does). The page as it is if nothing could be placed.
fn masked(img: &Mat, templates: &[Template], ms: &MaskSettings) -> Result<Mat> {
    let al = AlignSettings::default();
    let first = redact_page(img, templates, &al, ms, &mut PyRandom::new(0), &PageOptions { generic: true, ..Default::default() })?;
    let image = match first.image {
        Some(i) => Some(i),
        None => {
            let po = PageOptions { generic: true, verify: false, whole: vec![first.unsure.clone()], ..Default::default() };
            redact_page(img, templates, &al, ms, &mut PyRandom::new(0), &po)?.image
        }
    };
    Ok(image.unwrap_or_else(|| img.clone()))
}

/// Whether the job in `job_dir` lets page `page` of `src` be shown to the
/// cloud model as `mode` asks: its plan says `held_pages: cloud`, the page
/// is held in the job (its last result in `review/report.jsonl`), and, to
/// locate fields, triage found it to be the job's form (a decision in the
/// journal).
pub fn held_access(job_dir: &Path, src: &str, page: usize, mode: Mode) -> Result<()> {
    let plan = crate::manifest::read_json(&job_dir.join("plan.json")).context("refusing: not a job folder (no plan.json)")?;
    if plan["held_pages"] != "cloud" {
        bail!("refusing: the job does not let held pages go to the cloud model (held_pages: {})", plan["held_pages"].as_str().unwrap_or("none"));
    }
    let held = crate::job::stats::last_results(&job_dir.join("review"))
        .iter()
        .any(|r| r["file"].as_str() == Some(src) && r["ok"] == false && r["page"].as_u64().is_none_or(|p| p as usize == page));
    if !held {
        bail!("refusing: page {} of {} is not held in this job", page, src);
    }
    if mode == Mode::Locate {
        let journal = std::fs::read_to_string(job_dir.join("journal.jsonl")).unwrap_or_default();
        let this_form = journal.lines().filter_map(|l| serde_json::from_str::<Value>(l).ok()).any(|e| {
            e["kind"] == "decision" && e["phase"] == "triage" && e["source"].as_str() == Some(src) && e["page"].as_u64() == Some(page as u64) && e["choice"] == "this_form"
        });
        if !this_form {
            bail!("refusing: triage has not found page {} of {} to be the job's form; it is not sent at full resolution", page, src);
        }
    }
    Ok(())
}

/// Page `page` (from 1) of a held source, as `mode` needs it. Whether the
/// job allows it is the caller's to check (`held_access`): this only makes
/// the image.
pub fn held_view(src: &str, page: usize, mode: Mode, templates: &[Template], ms: &MaskSettings, format: &str) -> Result<View> {
    let data = io::read_bytes(src)?;
    let (sp, _) = io::source_page(src, &data, page)?;
    match mode {
        Mode::Locate => View::new(sp.pixels, [0.0, 0.0, 1.0], format),
        Mode::Triage => {
            let (small, f) = shrink(&masked(&sp.pixels, templates, ms)?, TRIAGE_SIDE)?;
            View::new(small, [0.0, 0.0, f], format)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crop_takes_only_a_manifest_beside_its_pdf() {
        let d = crate::testutil::tmpdir("crop");
        std::fs::write(d.join("x.pdf"), b"%PDF").unwrap();
        // an original, or a held page's report: not a manifest
        assert!(published_pdf(&d.join("x.pdf")).unwrap_err().to_string().contains("refusing"));
        std::fs::write(d.join("held.json"), r#"{"ok": false}"#).unwrap();
        assert!(published_pdf(&d.join("held.json")).unwrap_err().to_string().contains("refusing"));
        // named like one, but not one; or one with no PDF beside it
        std::fs::write(d.join("x.manifest.json"), r#"{"ok": false}"#).unwrap();
        assert!(published_pdf(&d.join("x.manifest.json")).is_err());
        std::fs::write(d.join("y.manifest.json"), format!(r#"{{"schema": "{}"}}"#, crate::process::SCHEMA)).unwrap();
        assert!(published_pdf(&d.join("y.manifest.json")).unwrap_err().to_string().contains("no published PDF"));
        std::fs::write(d.join("x.manifest.json"), format!(r#"{{"schema": "{}"}}"#, crate::process::SCHEMA)).unwrap();
        assert_eq!(published_pdf(&d.join("x.manifest.json")).unwrap().0, d.join("x.pdf"));
    }

    #[test]
    fn a_triage_view_is_small_and_a_view_maps_boxes_back_to_the_page() {
        let d = crate::testutil::tmpdir("held-view");
        let page = crate::testutil::bgr(2000, 1400, [255, 255, 255]);
        let src = d.join("p.png");
        std::fs::write(&src, io::encode(&page, "png", 0).unwrap()).unwrap();
        let s = src.to_string_lossy();
        let v = held_view(&s, 1, Mode::Triage, &[], &MaskSettings::default(), "png").unwrap();
        assert_eq!((v.image.rows(), v.image.cols()), (512, 358));
        assert_eq!(v.sha256, crate::proof::sha256(&v.encoded));
        let full = held_view(&s, 1, Mode::Locate, &[], &MaskSettings::default(), "png").unwrap();
        assert_eq!((full.image.rows(), full.image.cols()), (2000, 1400));
        let b = View { image: Mat::default(), encoded: vec![], sha256: String::new(), origin: [100.0, 200.0, 0.5] }.to_page([10.0, 20.0, 30.0, 40.0]);
        assert_eq!(b, [120.0, 240.0, 160.0, 280.0]);
    }
}
