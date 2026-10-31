//! Remediation (HARNESS.md §4.6). Code acts on every confirmed finding:
//!
//! - a single leak: the page redacted again with the finding's box as a spot
//!   mask (`redact --spot-masks`, recorded in its manifest and replayed by
//!   `verify --rerun`), verified, re-run and reviewed again; still showing
//!   after two rounds, or with no box (a reply that couldn't be read), it is
//!   quarantined;
//! - a pattern: three or more confirmed leaks of one kind in a group, or a
//!   kind the orchestrator calls a pattern. Every page of the group with the
//!   signals all its leaking pages share is quarantined; with no shared
//!   signal the pattern stays open and the group is held. Either way a
//!   defect is written, with crops of the leaking pages as published.

use anyhow::Result;
use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;

use super::plan::Unit;
use super::run::Job;
use super::view::{self, Region};

/// Rounds of spot masks before a page still showing is quarantined.
const ROUNDS: usize = 2;

/// A risk signal with its numbers taken out, so like ones compare equal.
fn signal(s: &str) -> String {
    regex::Regex::new(r"\d+").unwrap().replace_all(s, "N").into_owned()
}

impl Job {
    /// Move a staged unit to `quarantine/`, with why.
    pub(super) fn quarantine(&mut self, phase: &str, u: &Unit, why: &str) -> Result<()> {
        let q = self.dir.join("quarantine");
        for ext in [".pdf", ".manifest.json"] {
            let (from, to) = (u.at(&self.staging(), ext), u.at(&q, ext));
            std::fs::create_dir_all(to.parent().unwrap())?;
            if from.exists() {
                std::fs::rename(&from, &to)?;
            }
        }
        std::fs::write(u.at(&q, ".json"), serde_json::to_string_pretty(&json!({"document": u.name(), "why": why, "at": crate::process::utc_now()}))?)?;
        self.check(u).quarantined = Some(why.to_string());
        self.journal.tool(phase, "quarantine", json!({"document": u.name(), "why": why}), &json!({"moved_to": "quarantine"}))?;
        Ok(())
    }

    /// The spot masks given so far (`spot_masks.json`).
    fn spots(&self) -> Vec<Value> {
        crate::process::read_spot_masks(&self.dir.join("spot_masks.json")).unwrap_or_default()
    }

    /// Add spot masks for a unit, and `redact` it again.
    pub(super) fn spot_mask(&mut self, phase: &str, units: &[Unit], add: Vec<Value>) -> Result<()> {
        let mut spots = self.spots();
        for s in add {
            if !spots.contains(&s) {
                spots.push(s);
            }
        }
        std::fs::write(self.dir.join("spot_masks.json"), serde_json::to_string_pretty(&spots)?)?;
        self.journal.tool(phase, "spot_masks", json!({"documents": units.iter().map(Unit::name).collect::<Vec<_>>()}), &json!(spots))?;
        self.redact(phase, units, true)
    }

    /// Phase 6: patterns first, then each page's open findings.
    pub(super) fn remediate(&mut self, phase: &str) -> Result<()> {
        self.patterns(phase)?;
        for _ in 0..ROUNDS {
            let todo: Vec<Unit> = self.published().into_values().flatten().filter(|u| !self.looked(u).open().is_empty()).collect();
            if todo.is_empty() {
                return Ok(());
            }
            let (mut redo, mut add) = (Vec::new(), Vec::new());
            for u in &todo {
                let open: Vec<Value> = self.looked(u).open().into_iter().cloned().collect();
                if open.iter().any(|f| f["box"].is_null()) {
                    self.quarantine(phase, u, "the vision review's reply could not be read")?;
                    continue;
                }
                add.extend(open.iter().map(|f| json!({"sha256": u.sha256, "page": f["page"], "label": f["label"], "box": f["box"],
                                                     "source": {"finding": f["id"], "model": f["model"], "crop_sha256": f["crop_sha256"], "find_prompt_sha256": f["detail"]}})));
                redo.push(u.clone());
            }
            if redo.is_empty() {
                continue;
            }
            self.spot_mask(phase, &redo, add)?;
            let now = self.published();
            for u in &redo {
                let c = self.check(u);
                for f in c.findings.iter_mut() {
                    f["fixed"] = json!(true);
                }
                let why = c.reviewed.clone().unwrap_or_else(|| "spot-masked".into());
                (c.verified, c.rerun) = (None, None);
                c.problems.clear();
                if now.get(&u.group).is_some_and(|g| g.contains(u)) {
                    self.verify(phase, u, true)?;
                    self.review(phase, std::slice::from_ref(u), &why, true)?;
                }
            }
        }
        for u in self.published().into_values().flatten().collect::<Vec<_>>() {
            if !self.looked(&u).open().is_empty() {
                self.quarantine(phase, &u, &format!("still showing after {} rounds of spot masks", ROUNDS))?;
            }
        }
        Ok(())
    }

    /// Patterns among a group's confirmed leaks (HARNESS.md §4.6).
    fn patterns(&mut self, phase: &str) -> Result<()> {
        for (gid, units) in self.published() {
            let mut by_kind: BTreeMap<String, Vec<Unit>> = BTreeMap::new();
            for u in &units {
                let kinds: BTreeSet<String> = self.looked(u).findings.iter().filter_map(|f| f["label"].as_str()).filter(|k| *k != "unreadable").map(String::from).collect();
                for k in kinds {
                    by_kind.entry(k).or_default().push(u.clone());
                }
            }
            for (kind, leaking) in by_kind {
                let known = self.groups.get(&gid).is_some_and(|g| g.patterns.iter().any(|p| p["kind"] == kind.as_str()));
                if known {
                    continue;
                }
                let summary: Vec<Value> = leaking.iter().map(|u| json!({"kind": kind, "signals": self.looked(u).signals.iter().map(|s| signal(s)).collect::<Vec<_>>()})).collect();
                let called = leaking.len() < super::sample::PATTERN && self.models.as_ref().and_then(|m| m.remedy(&json!(summary), phase, &self.journal)).as_deref() == Some("pattern");
                if leaking.len() < super::sample::PATTERN && !called {
                    continue;
                }
                let sets: Vec<BTreeSet<String>> = leaking.iter().map(|u| self.looked(u).signals.iter().map(|s| signal(s)).collect()).collect();
                let shared: Vec<String> = sets.iter().skip(1).fold(sets[0].clone(), |a, b| a.intersection(b).cloned().collect()).into_iter().collect();
                let defect = self.defect(&gid, &kind, &leaking, &shared)?;
                let mut quarantined = 0;
                if !shared.is_empty() {
                    for u in units.iter().filter(|u| shared.iter().all(|s| self.looked(u).signals.iter().any(|x| signal(x) == *s))).cloned().collect::<Vec<_>>() {
                        self.quarantine(phase, &u, &format!("a pattern of {} leaks: shares {}", kind, shared.join("; ")))?;
                        quarantined += 1;
                    }
                }
                let p = json!({"kind": kind, "leaks": leaking.len(), "signal": shared, "open": shared.is_empty(), "quarantined": quarantined, "defect": defect, "called_by_model": called});
                self.journal.decision(phase, "pattern", json!([format!("{} {} leak(s) in group {}", leaking.len(), kind, gid)]), json!({"group": gid, "pattern": p}))?;
                self.groups.entry(gid.clone()).or_default().patterns.push(p);
            }
        }
        Ok(())
    }

    /// A defect for the developers (`defects/<group>-<kind>.json`): what
    /// leaks, the signal that finds it, the pages, and crops of them as
    /// published (masks and all), cut before they are quarantined.
    fn defect(&mut self, gid: &str, kind: &str, leaking: &[Unit], shared: &[String]) -> Result<PathBuf> {
        let dir = self.dir.join("defects").join(format!("{}-{}", gid, kind));
        std::fs::create_dir_all(&dir)?;
        let mut crops = Vec::new();
        for u in leaking {
            let manifest = u.at(&self.staging(), ".manifest.json");
            let Ok(v) = view::crop(&view::CropArgs { manifest: &manifest, region: Region::Declaration, page: 1, templates: &self.templates, max_side: 1800, pubkey: Some(&self.pubkey), within: Some(&self.staging()), format: "jpg" }) else { continue };
            let p = dir.join(format!("{}.jpg", u.name().replace('/', "_")));
            std::fs::write(&p, &v.encoded)?;
            crops.push(p.to_string_lossy().into_owned());
        }
        let findings: Vec<Value> = leaking.iter().flat_map(|u| self.looked(u).findings.iter().filter(|f| f["label"] == kind).map(|f| json!({"document": u.name(), "id": f["id"], "box": f["box"]})).collect::<Vec<_>>()).collect();
        let d = json!({"title": format!("{} leaks in group {}", kind, gid), "what": kind, "signal": shared, "pages": leaking.iter().map(Unit::name).collect::<Vec<_>>(), "findings": findings, "crops": crops,
                       "fix": "in the developers' loop, not by the harness: once a new version is released, the affected pages are run again and their masks compared with this run's"});
        let path = self.dir.join("defects").join(format!("{}-{}.json", gid, kind));
        std::fs::write(&path, serde_json::to_string_pretty(&d)?)?;
        self.journal.tool("remediate", "defect", json!({"group": gid, "kind": kind}), &d)?;
        Ok(path)
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn signals_compare_without_their_numbers() {
        assert_eq!(super::signal("2 field line(s) hidden whole"), super::signal("4 field line(s) hidden whole"));
    }
}
