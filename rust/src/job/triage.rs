//! Triage of held pages (HARNESS.md §4.7). Each held page gets its text
//! diagnostics (numbers and printed words). With `held_pages: cloud`, its
//! triage view (every mask the redactor placed, 512 px long) goes to the
//! vision model, which says what kind of page it is; a page of the job's
//! form whose fields couldn't be placed then goes at full resolution to
//! locate them, and is redacted again with those boxes as spot masks, to go
//! through every gate and QA like any other page, always reviewed. Without
//! the cloud, the orchestrator sorts it from its diagnostics alone, or,
//! without models, fixed rules do. Each image sent is in the journal, in
//! the page's triage note and in `transfers.jsonl`.

use anyhow::Result;
use serde_json::{json, Value};

use super::diagnose;
use super::plan::Unit;
use super::run::{Held, Job};
use super::spec::HeldPages;
use super::view::{self, Mode, View};

/// A held page's category and suggested action from its diagnostics, by
/// fixed rules, with the reason: blank, a tally sheet, too faint, turned,
/// not a form, or not read.
pub fn categorise(d: &Value, reason: &str) -> (String, String, String) {
    let out = |c: &str, a: &str, why: String| (c.to_string(), a.to_string(), why);
    let ink = d["ink"].as_f64().unwrap_or(1.0);
    let contrast = d["contrast"].as_f64().unwrap_or(100.0);
    let conf = d["ocr"]["mean_confidence"].as_f64();
    let words: Vec<&str> = d["ocr"]["vocabulary"].as_array().into_iter().flatten().filter_map(Value::as_str).collect();
    let turn = d["ocr"]["turn"].as_i64().unwrap_or(0);
    if d.get("error").is_some() {
        return out("unknown", "people", format!("diagnostics failed: {}", d["error"].as_str().unwrap_or("")));
    }
    if ink < 0.005 {
        return out("blank", "not_for_this_job", format!("ink covers {:.2}% of the page", ink * 100.0));
    }
    // a form names itself ("FORM 34A"), or a template matched it; Form
    // 34A's own title reads "TALLYING", so TALLY alone is not enough
    let a_form = words.iter().any(|w| w.starts_with("FORM ")) || d["read"]["template"].is_string();
    if words.contains(&"TALLY") && !a_form {
        return out("tally_sheet", "not_for_this_job", "the printed word TALLY is on the page, and no form's name".into());
    }
    if contrast < 25.0 || conf.is_some_and(|c| c < 0.6) {
        return out("faint", "people", format!("contrast {:.1}, OCR confidence {}", contrast, conf.map(|c| format!("{:.2}", c)).unwrap_or("none".into())));
    }
    if turn != 0 {
        return out("turned", "retry", format!("its text reads at a {}° turn", turn));
    }
    if words.is_empty() {
        return out("not_a_form", "people", "none of the forms' printed words was found".into());
    }
    out("not_read", "people", format!("held: {}", crate::annex::normalise_reason(reason)))
}

impl Job {
    /// A held page's view for `mode`, made only as the job allows
    /// (`view::held_access`), registered with the guard and journalled.
    fn held_view(&mut self, u: &Unit, mode: Mode) -> Result<View> {
        view::held_access(&self.dir, &u.src, u.page_no(), mode)?;
        let v = view::held_view(&u.src, u.page_no(), mode, &self.templates, &self.policy.mask(), "jpg")?;
        let (sha, px) = self.models.as_ref().expect("held views go to a model").guard.register(&v.image, &v.encoded)?;
        self.journal.write("triage", "tool", json!({"tool": "held_view", "args": {"source": u.src, "page": u.page_no(), "mode": mode.name()}, "image_sha256": sha, "pixels_sha256": px}))?;
        Ok(v)
    }

    /// Phase 7: each held page diagnosed, sorted, and where it can be,
    /// located and redacted again; a triage note per page in
    /// `review/triage/`.
    pub(super) fn triage(&mut self) -> Result<()> {
        let ms = self.policy.mask();
        let cloud = self.spec.held_pages == HeldPages::Cloud && self.models.as_ref().is_some_and(|m| m.has_vision());
        for (u, reason) in self.held_units() {
            let mut d = diagnose::diagnose(&u.src, u.page_no(), &self.templates, &ms).unwrap_or_else(|e| json!({"error": format!("{:#}", e)}));
            d["reason"] = json!(crate::annex::normalise_reason(&reason));
            self.journal.tool("triage", "diagnose", json!({"source": u.src, "page": u.page_no()}), &d)?;
            let mut sent: Vec<Value> = Vec::new();
            let v = if cloud { Some(self.held_view(&u, Mode::Triage)?) } else { None };
            if let Some(v) = &v {
                sent.push(json!({"mode": "triage", "size": format!("{} px", view::TRIAGE_SIDE), "sha256": v.sha256}));
            }
            let verdict = match self.models.as_ref() {
                Some(m) => m.triage(&d, v.as_ref(), &u.name(), u.page_no(), &self.journal)?,
                None => None,
            };
            let (category, mut action, why) = match &verdict {
                Some(x) => (x["kind"].as_str().unwrap_or("").to_string(), x["action"].as_str().unwrap_or("").to_string(), format!("the {} model's verdict", if v.is_some() { "vision" } else { "orchestrator" })),
                None => categorise(&d, &reason),
            };
            self.journal.decision("triage", &category, json!([why]), json!({"source": u.src, "page": u.page_no(), "action": action, "verdict": verdict}))?;
            if category == "this_form" && cloud {
                action = self.locate(&u, &mut sent)?;
            }
            if !sent.is_empty() {
                self.cloud.push(json!({"document": u.name(), "images": sent}));
            }
            let note = json!({"document": u.name(), "source": u.src, "page": u.page_no(), "reason": d["reason"], "diagnostics": d, "category": category, "action": action, "why": why, "verdict": verdict, "images_sent": sent});
            let p = u.at(&self.review_dir().join("triage"), ".json");
            std::fs::create_dir_all(p.parent().unwrap())?;
            std::fs::write(&p, serde_json::to_string_pretty(&note)?)?;
            if action != "located: published" {
                self.held.push(Held { unit: u, reason: crate::annex::normalise_reason(&reason), category, action, sent });
            }
        }
        Ok(())
    }

    /// A held page of the job's form: sent at full resolution to locate its
    /// fields, then redacted again with them as spot masks. Published, it
    /// is verified, re-run and always reviewed. The action it came to.
    fn locate(&mut self, u: &Unit, sent: &mut Vec<Value>) -> Result<String> {
        let v = self.held_view(u, Mode::Locate)?;
        sent.push(json!({"mode": "locate", "size": "full resolution", "sha256": v.sha256}));
        let boxes = match self.models.as_ref() {
            Some(m) => m.locate(&v, &u.name(), u.page_no(), &self.journal)?,
            None => None,
        }
        .unwrap_or_default();
        if boxes.is_empty() {
            return Ok("people".into());
        }
        let model = self.models.as_ref().and_then(|m| m.reviewer.as_ref()).map(|r| r.finder.s.model.clone());
        let add: Vec<Value> = boxes.iter().map(|b| json!({"sha256": u.sha256, "page": u.page_no(), "label": b.label, "box": [b.x0.round(), b.y0.round(), b.x1.round(), b.y1.round()],
                                                          "source": {"located_by": model, "image_sha256": v.sha256}})).collect();
        self.spot_mask("triage", std::slice::from_ref(u), add)?;
        if !self.published().get(&u.group).is_some_and(|g| g.contains(u)) {
            return Ok("located: still held".into());
        }
        self.check(u).located = true;
        self.verify("triage", u, true)?;
        self.review("triage", std::slice::from_ref(u), "located", true)?;
        Ok("located: published".into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn held_pages_are_sorted_by_fixed_rules() {
        let d = |ink: f64, contrast: f64, conf: Option<f64>, words: &[&str], turn: i64| json!({"ink": ink, "contrast": contrast, "ocr": {"mean_confidence": conf, "vocabulary": words, "turn": turn}});
        assert_eq!(categorise(&d(0.001, 2.0, None, &[], 0), "").0, "blank");
        assert_eq!(categorise(&d(0.05, 50.0, Some(0.9), &["TALLY", "COUNTY"], 0), "").0, "tally_sheet");
        // a Form 34A ("... TALLYING ..." in its title): never set aside as a tally sheet
        assert_eq!(categorise(&d(0.05, 50.0, Some(0.9), &["FORM 34A", "TALLY", "DECLARATION"], 0), "").0, "not_read");
        let mut matched = d(0.05, 50.0, Some(0.9), &["TALLY", "DECLARATION"], 0);
        matched["read"] = json!({"template": "form34a_2022", "ok": false});
        assert_eq!(categorise(&matched, "").0, "not_read");
        assert_eq!(categorise(&d(0.05, 18.0, Some(0.9), &["DECLARATION"], 0), "").0, "faint");
        assert_eq!(categorise(&d(0.05, 50.0, Some(0.9), &["DECLARATION"], 90), "").0, "turned");
        assert_eq!(categorise(&d(0.05, 50.0, Some(0.9), &[], 0), "").0, "not_a_form");
        let (c, a, why) = categorise(&d(0.05, 50.0, Some(0.9), &["DECLARATION"], 0), "page 3: leak check: agent_id: 7 in a row");
        assert!(c == "not_read" && a == "people" && why.contains("N in a row"), "{}", why);
    }
}
