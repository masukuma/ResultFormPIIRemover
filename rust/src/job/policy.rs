//! The masking policy a job runs (HARNESS.md §3): a closed set of choices,
//! each mapped to `rfpii redact` flags, compiled from the requester's
//! instructions. What the instructions ask that the redactor has no rule
//! for is listed, never improvised:
//!
//! - raised: a point the instructions left unclear, settled by the reading
//!   that hides more;
//! - refused: less than the redactor's floor ("don't mask the signatures"):
//!   masked as usual, and noted;
//! - unsupported: more than the redactor can do ("hide the names", "hide
//!   the totals"): redacted and checked, but nothing released.

use anyhow::{bail, Result};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum Ids {
    /// `hide` characters of each
    #[default]
    Three,
    /// every character (`--full-ids`)
    Whole,
}

#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum Phones {
    /// `hide` digits within the last `phone_tail`
    #[default]
    Tail,
    /// every digit (`--full-phones`)
    Whole,
}

#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum Stamps {
    /// a number's line, or a line through a signature row's middle, hidden
    /// across a stamp; the rest kept and verified
    #[default]
    Bands,
    /// never masked: writing under a stamp shows (`--keep-stamps`)
    Protected,
}

#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum Signatures {
    /// a black band through the middle of each
    #[default]
    Band,
}

#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum Tables {
    /// zones widened where a table's rules can't be matched
    #[default]
    Widen,
    /// such pages held (`--strict-structure`)
    Hold,
    /// columns that can't be placed hidden whole once a review passes the
    /// page (`--vlm-rescue --vlm-audit`); run as `hold` in this version
    HideColumnsIfReviewed,
}

#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum Review {
    /// the riskiest pages and a random sample (HARNESS.md §8)
    #[default]
    Sample,
    EveryPage,
}

#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum PerPage {
    /// scans of many forms per file page by page, other files whole
    #[default]
    Auto,
    Yes,
    No,
}

/// An instruction the policy couldn't follow as written, and why or how it
/// was settled.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Note {
    pub instruction: String,
    pub settled: String,
}

fn three() -> usize {
    3
}

fn six() -> usize {
    6
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Policy {
    #[serde(default)]
    pub ids: Ids,
    #[serde(default)]
    pub phones: Phones,
    #[serde(default = "three")]
    pub hide: usize,
    #[serde(default = "six")]
    pub phone_tail: usize,
    #[serde(default)]
    pub stamps: Stamps,
    #[serde(default)]
    pub signatures: Signatures,
    #[serde(default)]
    pub uncertain_tables: Tables,
    #[serde(default)]
    pub vision_review: Review,
    #[serde(default)]
    pub per_page: PerPage,
    #[serde(default)]
    pub raised: Vec<Note>,
    #[serde(default)]
    pub refused: Vec<Note>,
    #[serde(default)]
    pub unsupported: Vec<Note>,
}

impl Default for Policy {
    fn default() -> Self {
        serde_json::from_value(json!({})).unwrap()
    }
}

/// Characters hidden per number: at least this many, and no more than an
/// ID has.
const HIDE_RANGE: (usize, usize) = (3, 8);

impl Policy {
    /// A policy from its JSON, checked: unknown fields or values are
    /// refused, and `hide` below the floor is raised to it (noted).
    pub fn from_value(v: &Value) -> Result<Policy> {
        let mut p: Policy = serde_json::from_value(v.clone()).map_err(|e| anyhow::anyhow!("policy: {e}"))?;
        if p.hide < HIDE_RANGE.0 {
            p.refused.push(Note { instruction: format!("hide {} characters", p.hide), settled: format!("{} is the redactor's floor: {} hidden", HIDE_RANGE.0, HIDE_RANGE.0) });
            p.hide = HIDE_RANGE.0;
        }
        if p.hide > HIDE_RANGE.1 || p.phone_tail < p.hide || p.phone_tail > 10 {
            bail!("policy: hide {} and phone_tail {} must satisfy {} <= hide <= phone_tail <= 10 and hide <= {}", p.hide, p.phone_tail, HIDE_RANGE.0, HIDE_RANGE.1);
        }
        if p.uncertain_tables == Tables::HideColumnsIfReviewed {
            p.uncertain_tables = Tables::Hold;
            p.raised.push(Note {
                instruction: "hide a table's columns whole where they can't be placed".into(),
                settled: "this version doesn't run the vision review inside redact that makes it safe: such pages are held instead (hides more)".into(),
            });
        }
        Ok(p)
    }

    /// The policy of a job: the one compiled, else the job file's, else the
    /// defaults if there are no instructions. Instructions with nothing to
    /// compile them can't be shown to be met: nothing is released.
    pub fn without_model(instructions: &str, given: Option<&Value>) -> Result<Policy> {
        let mut p = match given {
            Some(v) => Policy::from_value(v)?,
            None => Policy::default(),
        };
        if given.is_none() && !instructions.trim().is_empty() {
            p.unsupported.push(Note {
                instruction: "the instructions as a whole".into(),
                settled: "no orchestrator model compiled them and the job file gives no policy, so it can't be shown they were met".into(),
            });
        }
        p.backstop(instructions);
        Ok(p)
    }

    /// Instructions to hide what `rfpii` never masks (names, results),
    /// found by plain matching, sentence by sentence, whatever compiled the
    /// policy: each is unsupported unless already listed. A sentence that
    /// asks not to hide, or to leave something visible, is passed over.
    pub fn backstop(&mut self, instructions: &str) {
        let hide = regex::Regex::new(r"(?i)\b(hide|hidden|mask|cover|redact|remove|black ?out|blur|conceal|erase|obscure)").unwrap();
        let keep = regex::Regex::new(r"(?i)\b(don'?t|do not|never|not|no need|except|other than|visible|readable|show)\b").unwrap();
        let targets = [
            (r"(?i)\bnames?\b", "names are never masked by rfpii (prescribed fields readers check); it would need a rule to find and mask names"),
            (r"(?i)\b(totals?|results?|votes?|tall(y|ies)|counts?)\b", "results are never masked: they are what is published, and protected regions no mask may touch"),
        ];
        for sentence in instructions.split(['.', ';', '!', '?', '\n']).map(str::trim).filter(|s| !s.is_empty()) {
            if !hide.is_match(sentence) || keep.is_match(sentence) {
                continue;
            }
            for (pat, why) in targets {
                let listed = self.unsupported.iter().any(|n| n.instruction.contains(sentence) || sentence.contains(n.instruction.as_str()));
                if regex::Regex::new(pat).unwrap().is_match(sentence) && !listed {
                    self.unsupported.push(Note { instruction: sentence.to_string(), settled: why.to_string() });
                }
            }
        }
    }

    /// The `redact` flags this policy sets (per page is the plan's, by group).
    pub fn flags(&self) -> Vec<String> {
        let mut f: Vec<String> = Vec::new();
        if self.ids == Ids::Whole {
            f.push("--full-ids".into());
        }
        if self.phones == Phones::Whole {
            f.push("--full-phones".into());
        }
        if self.hide != 3 {
            f.extend(["--hide".into(), self.hide.to_string()]);
        }
        if self.phone_tail != 6 {
            f.extend(["--phone-tail".into(), self.phone_tail.to_string()]);
        }
        if self.stamps == Stamps::Protected {
            f.push("--keep-stamps".into());
        }
        if self.uncertain_tables != Tables::Widen {
            f.push("--strict-structure".into());
        }
        f
    }

    /// What every manifest of the job must record in its `settings`.
    pub fn settings(&self) -> Value {
        json!({"hide": self.hide, "phone_tail": self.phone_tail, "stamps": match self.stamps { Stamps::Bands => "bands", Stamps::Protected => "protected" },
               "full_ids": self.ids == Ids::Whole, "full_phones": self.phones == Phones::Whole, "strict_structure": self.uncertain_tables != Tables::Widen})
    }

    /// The mask settings `redact` runs with under this policy.
    pub fn mask(&self) -> crate::masking::MaskSettings {
        crate::masking::MaskSettings::from_record(&self.settings())
    }

    /// The policy in plain words, a line per choice.
    pub fn words(&self) -> Vec<String> {
        let mut w = vec![
            match self.ids {
                Ids::Three => format!("ID and passport numbers: {} characters of each hidden", self.hide),
                Ids::Whole => "ID and passport numbers: hidden whole".into(),
            },
            match self.phones {
                Phones::Tail => format!("phone numbers: {} of the last {} digits hidden, the prefix kept", self.hide, self.phone_tail),
                Phones::Whole => "phone numbers: hidden whole".into(),
            },
            match self.stamps {
                Stamps::Bands => "stamps: a number's line across a stamp hidden, and a line through the middle of a signature's row; the rest of the stamp kept and verified unchanged".into(),
                Stamps::Protected => "stamps: never masked; writing under a stamp shows".into(),
            },
            "signatures: a black band through the middle of each".into(),
            match self.uncertain_tables {
                Tables::Widen => "tables whose rules can't be matched: masked with widened zones".into(),
                _ => "tables whose rules can't be matched: the page is held".into(),
            },
            match self.vision_review {
                Review::Sample => "vision review: the riskiest pages and a random sample of each group".into(),
                Review::EveryPage => "vision review: every published page".into(),
            },
        ];
        w.push(format!("per page: {}", match self.per_page { PerPage::Auto => "scans of many forms page by page, other files whole", PerPage::Yes => "every file page by page", PerPage::No => "every file whole" }));
        w
    }

    /// The JSON schema a model's compiled policy must follow (`format`).
    pub fn schema() -> Value {
        let e = |v: &[&str]| json!({"type": "string", "enum": v});
        let notes = json!({"type": "array", "items": {"type": "object", "properties": {"instruction": {"type": "string"}, "settled": {"type": "string"}}, "required": ["instruction", "settled"]}});
        json!({"type": "object", "properties": {
            "ids": e(&["three", "whole"]), "phones": e(&["tail", "whole"]),
            "hide": {"type": "integer", "minimum": 3, "maximum": 8}, "phone_tail": {"type": "integer", "minimum": 3, "maximum": 10},
            "stamps": e(&["bands", "protected"]), "signatures": e(&["band"]),
            "uncertain_tables": e(&["widen", "hold", "hide_columns_if_reviewed"]), "vision_review": e(&["sample", "every_page"]),
            "per_page": e(&["auto", "yes", "no"]),
            "raised": notes, "refused": notes, "unsupported": notes},
            "required": ["ids", "phones", "hide", "phone_tail", "stamps", "signatures", "uncertain_tables", "vision_review", "per_page", "raised", "refused", "unsupported"]})
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_policy_maps_to_flags_and_settings() {
        let p = Policy::default();
        assert!(p.flags().is_empty());
        assert_eq!(p.settings(), json!({"hide": 3, "phone_tail": 6, "stamps": "bands", "full_ids": false, "full_phones": false, "strict_structure": false}));
        let p = Policy::from_value(&json!({"ids": "whole", "hide": 4, "stamps": "protected", "uncertain_tables": "hold"})).unwrap();
        assert_eq!(p.flags(), vec!["--full-ids", "--hide", "4", "--keep-stamps", "--strict-structure"]);
        // the settings the manifests of such a run record
        let ms = p.mask();
        assert!(ms.full_ids && ms.hide == 4 && ms.stamps == crate::masking::StampRule::Protected);
    }

    #[test]
    fn a_policy_is_closed_and_has_a_floor() {
        assert!(Policy::from_value(&json!({"ids": "most"})).is_err());
        assert!(Policy::from_value(&json!({"names": "hidden"})).is_err());
        let p = Policy::from_value(&json!({"hide": 1})).unwrap();
        assert!(p.hide == 3 && p.refused.len() == 1);
        let p = Policy::from_value(&json!({"uncertain_tables": "hide_columns_if_reviewed"})).unwrap();
        assert!(p.uncertain_tables == Tables::Hold && p.raised.len() == 1);
        // every field a model must fill in is in the schema
        let req = Policy::schema()["required"].as_array().unwrap().len();
        assert_eq!(req, serde_json::to_value(Policy::default()).unwrap().as_object().unwrap().len());
    }

    #[test]
    fn instructions_to_hide_names_or_results_are_never_released() {
        let p = Policy::without_model("Hide ID numbers completely. Also hide the agents' names.", Some(&json!({"ids": "whole"}))).unwrap();
        assert_eq!(p.unsupported.len(), 1);
        assert!(p.unsupported[0].instruction.contains("agents' names") && p.unsupported[0].settled.contains("names are never masked"));
        let p = Policy::without_model("Hide the totals", Some(&json!({}))).unwrap();
        assert!(p.unsupported[0].settled.contains("results are never masked"));
        assert_eq!(Policy::without_model("Keep the names hidden", Some(&json!({}))).unwrap().unsupported.len(), 1);
        // asked to keep, or not to hide: nothing to flag
        for ok in ["Keep the names visible; hide the ID numbers", "Don't hide the names", "Hide every number except the totals", "Hide what is under the stamps too"] {
            assert!(Policy::without_model(ok, Some(&json!({}))).unwrap().unsupported.is_empty(), "{}", ok);
        }
        // instructions nothing compiled can't be shown to be met
        assert_eq!(Policy::without_model("Standard redaction", None).unwrap().unsupported.len(), 1);
        assert!(Policy::without_model("", None).unwrap().unsupported.is_empty());
    }
}
