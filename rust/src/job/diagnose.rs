//! Text diagnostics of a held page (HARNESS.md §4.7): numbers, yes/no
//! answers and words from a fixed list of the forms' printed vocabulary.
//! Nothing read off the page leaves here except a word of that list: no
//! line of OCR text, no name, no number written on the form.

use anyhow::Result;
use opencv::core::Mat;
use opencv::imgproc;
use opencv::prelude::*;
use serde_json::{json, Value};
use std::collections::BTreeMap;

use crate::align::AlignSettings;
use crate::io;
use crate::masking::MaskSettings;
use crate::pipeline::{redact_page, PageOptions};
use crate::pyjson::round_n;
use crate::pyrand::PyRandom;
use crate::templates::Template;
use crate::util;

/// Printed words of the result forms, the only text a diagnosis reports:
/// which of them OCR found on the page.
pub const VOCABULARY: &[&str] = &[
    "FORM 34A", "FORM 34B", "FORM 34C", "FORM 35A", "FORM 36A", "FORM 37A", "FORM 38A", "DECLARATION", "PRESIDING OFFICER", "DEPUTY",
    "RETURNING OFFICER", "AGENTS", "CANDIDATES", "TALLY", "POLLING STATION", "COUNTY", "CONSTITUENCY", "WARD", "SIGNATURE", "PASSPORT",
    "TEL", "VOTES", "REJECTED", "VALID", "COMMENTS", "REASONS FOR REFUSAL", "RESULTS", "ELECTION", "IEBC", "STAMP",
];

/// Pixels darker than this count as ink.
const INK_BELOW: u8 = 128;

/// Long side the page is read at for its OCR confidence and vocabulary.
const READ_SIDE: f64 = 1600.0;

/// Contrast (the standard deviation of grey) and ink coverage (the share of
/// pixels darker than `INK_BELOW`).
fn contrast_and_ink(gray: &Mat) -> Result<(f64, f64)> {
    let v = util::bytes(gray)?;
    let n = v.len().max(1) as f64;
    let mean = v.iter().map(|&x| x as f64).sum::<f64>() / n;
    let var = v.iter().map(|&x| (x as f64 - mean).powi(2)).sum::<f64>() / n;
    Ok((var.sqrt(), v.iter().filter(|&&x| x < INK_BELOW).count() as f64 / n))
}

/// The words of `VOCABULARY` in `lines` of OCR text, each allowing an error
/// per eight characters.
pub fn vocabulary(lines: &[crate::ocr::Line]) -> Vec<&'static str> {
    VOCABULARY.iter().copied().filter(|w| lines.iter().any(|l| !l.fuzzy(w, w.len() / 8).is_empty())).collect()
}

/// The page turned upright by `k` quarter turns clockwise.
fn turned(img: &Mat, k: i32) -> Result<Mat> {
    let code = match k.rem_euclid(4) {
        0 => return Ok(img.try_clone()?),
        1 => opencv::core::ROTATE_90_CLOCKWISE,
        2 => opencv::core::ROTATE_180,
        _ => opencv::core::ROTATE_90_COUNTERCLOCKWISE,
    };
    let mut out = Mat::default();
    opencv::core::rotate(img, &mut out, code)?;
    Ok(out)
}

/// The turn and skew the text reads at, the OCR's mean confidence and the
/// vocabulary found; None without the OCR models.
fn read(img: &Mat) -> Result<Option<Value>> {
    if !crate::ocr::available() {
        return Ok(None);
    }
    crate::ocr::with_engine(|o| {
        let frame = crate::generic::find_frame(img, o, 1200.0)?;
        let up = turned(img, frame.as_ref().map(|f| f.k).unwrap_or(0))?;
        let f = (READ_SIDE / up.cols().max(up.rows()) as f64).min(1.0);
        let small = if f < 1.0 { util::resize_by(&up, f, imgproc::INTER_AREA)? } else { up };
        let lines = o.read(&small, 0.0)?;
        let conf: Vec<f64> = lines.iter().map(|l| l.score).collect();
        Ok(Some(json!({
            "turn": frame.as_ref().map(|f| 90 * f.k),
            "skew": frame.as_ref().map(|f| round_n(f.skew, 2)),
            "lines": lines.len(),
            "mean_confidence": (!conf.is_empty()).then(|| round_n(conf.iter().sum::<f64>() / conf.len() as f64, 3)),
            "vocabulary": vocabulary(&lines),
        })))
    })
}

/// What a mask record is of: its rule (id, phone), or a signature, or its
/// mode.
fn field_kind(m: &Value) -> String {
    match (m["rule"].as_str().filter(|r| !r.is_empty()), m["mode"].as_str().unwrap_or("?")) {
        (Some(r), _) => r.to_string(),
        (None, mode) if mode == "signature" || m["zone"].as_str().is_some_and(|z| z.contains("signature")) => "signature".into(),
        (None, mode) => mode.to_string(),
    }
}

/// How the redactor reads the page, as `redact` does: the template it
/// matched (by name) and how well, whether it passed and why not
/// (normalised), and its fields by kind and how each was masked.
fn attempt(img: &Mat, templates: &[Template], ms: &MaskSettings) -> Result<Value> {
    let r = redact_page(img, templates, &AlignSettings::default(), ms, &mut PyRandom::new(0), &PageOptions { generic: true, ..Default::default() })?;
    let mut fields: BTreeMap<String, BTreeMap<String, usize>> = BTreeMap::new();
    for m in &r.masks {
        *fields.entry(field_kind(m)).or_default().entry(m["mode"].as_str().unwrap_or("?").to_string()).or_insert(0) += 1;
    }
    Ok(json!({
        "template": r.template,
        "inliers": r.inliers,
        "score": round_n(r.score, 3),
        "ok": r.ok,
        "reason": crate::annex::normalise_reason(&r.reason),
        "fields": fields,
        "columns_unsure": r.unsure.len(),
    }))
}

/// Diagnostics of page `page` (from 1) of `src`, read with `templates`
/// (prepared) and the mask settings of the run that held it.
pub fn diagnose(src: &str, page: usize, templates: &[Template], ms: &MaskSettings) -> Result<Value> {
    let data = io::read_bytes(src)?;
    let (sp, pages) = io::source_page(src, &data, page)?;
    let gray = util::gray(&sp.pixels)?;
    let (contrast, ink) = contrast_and_ink(&gray)?;
    Ok(json!({
        "page": page,
        "pages": pages,
        "size": [sp.pixels.cols(), sp.pixels.rows()],
        "contrast": round_n(contrast, 2),
        "ink": round_n(ink, 4),
        "ocr": read(&sp.pixels)?,
        "read": attempt(&sp.pixels, templates, ms)?,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn line(text: &str) -> crate::ocr::Line {
        crate::ocr::Line::new([[0.0; 2]; 4], text.into(), 0.9, vec![], vec![])
    }

    #[test]
    fn only_words_of_the_vocabulary_are_reported() {
        let lines = [line("DECLARATI0N BY THE PRESIDING OFFICER"), line("JANE A. DOE 0712345678"), line("Tally sheet")];
        assert_eq!(vocabulary(&lines), vec!["DECLARATION", "PRESIDING OFFICER", "TALLY"]);
    }

    #[test]
    fn a_blank_page_has_no_contrast_and_no_ink() {
        let (c, i) = contrast_and_ink(&crate::testutil::gray(100, 80, 240)).unwrap();
        assert_eq!((c, i), (0.0, 0.0));
        let mut g = crate::testutil::gray(100, 80, 255);
        crate::testutil::fill_rect(&mut g, 0, 0, 80, 25, 0);
        let (c, i) = contrast_and_ink(&g).unwrap();
        assert!((i - 0.25).abs() < 1e-9 && c > 100.0);
    }
}
