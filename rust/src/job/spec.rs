//! A job file (HARNESS.md §1): what to redact, what the requester asked,
//! where it all goes, who publishes it, and the models and budget.
//!
//! ```yaml
//! inputs: [/data/2013/IEBC2/]
//! instructions: |
//!   Hide ID numbers completely. Keep the phone prefixes.
//! policy: {ids: whole}          # optional: used when no model compiles the instructions
//! outputs: /jobs/2013-34
//! sign_key: /keys/signing_key.pem
//! controller: /keys/controller.json
//! dpia: {reference: ODPC/DPIA/2026/14, submitted_on: 2026-08-01, status: submitted}
//! templates: /opt/rfpii/templates
//! held_pages: cloud             # or none (the default)
//! models: {orchestrator: [qwen3.5:397b], vision: [gemma4:31b]}
//! budget: {model_requests: 20000, hours: 12}
//! ```

use anyhow::{Context, Result};
use serde::Deserialize;
use serde_json::Value;
use std::path::{Path, PathBuf};

/// Whether held pages may go to the vision model (HARNESS.md §4.7).
#[derive(Deserialize, Clone, Copy, Debug, PartialEq, Eq, Default)]
#[serde(rename_all = "lowercase")]
pub enum HeldPages {
    Cloud,
    #[default]
    None,
}

impl HeldPages {
    pub fn name(self) -> &'static str {
        match self {
            HeldPages::Cloud => "cloud",
            HeldPages::None => "none",
        }
    }
}

/// The data protection impact assessment the processing rests on (Data
/// Protection Act s.31): its reference, when it was submitted to the
/// Commissioner, and where it stands.
#[derive(Deserialize, Clone, Debug, Default)]
#[serde(deny_unknown_fields)]
pub struct Dpia {
    pub reference: String,
    /// YYYY-MM-DD
    pub submitted_on: Option<String>,
    /// approved, deemed approved, submitted...
    pub status: Option<String>,
}

/// Days a DPIA must have been with the Commissioner before the processing,
/// unless approved sooner.
pub const DPIA_DAYS: i64 = 60;

impl Dpia {
    /// Why processing may not start at `now` (a time as `utc_now` writes
    /// it), or None if it may: the DPIA is approved (or deemed approved),
    /// or was submitted at least 60 days before.
    pub fn bars(&self, now: &str) -> Option<String> {
        let status = self.status.as_deref().unwrap_or("").trim().to_lowercase();
        if self.reference.trim().is_empty() {
            return Some("the DPIA has no reference".into());
        }
        if status == "approved" || status == "deemed approved" {
            return None;
        }
        let Some(on) = self.submitted_on.as_deref() else {
            return Some(format!("DPIA {} is {} and has no submission date", self.reference, if status.is_empty() { "not approved" } else { &status }));
        };
        match (super::epoch_of(on), super::epoch_of(now)) {
            (Some(a), Some(b)) if (b - a) / 86400 >= DPIA_DAYS => None,
            (Some(a), Some(b)) => Some(format!("DPIA {} was submitted on {}, {} day(s) ago; processing may start {} days after submission unless it is approved", self.reference, on, (b - a).div_euclid(86400), DPIA_DAYS)),
            _ => Some(format!("DPIA {}: submission date {:?} is not YYYY-MM-DD", self.reference, on)),
        }
    }
}

/// The models of each role, in order of preference; Ollama Cloud unless a
/// host is given (HARNESS.md §5).
#[derive(Deserialize, Clone, Debug, Default)]
#[serde(deny_unknown_fields)]
pub struct Models {
    #[serde(default)]
    pub orchestrator: Vec<String>,
    #[serde(default)]
    pub vision: Vec<String>,
    pub host: Option<String>,
}

#[derive(Deserialize, Clone, Debug, Default)]
#[serde(deny_unknown_fields)]
pub struct Budget {
    pub model_requests: Option<u64>,
    pub hours: Option<f64>,
}

#[derive(Deserialize, Clone, Debug)]
#[serde(deny_unknown_fields)]
pub struct JobFile {
    /// files, folders and zips
    pub inputs: Vec<String>,
    /// free text from the requester
    #[serde(default)]
    pub instructions: String,
    /// the policy, structured (`policy::Policy`): used when no model
    /// compiles the instructions
    pub policy: Option<Value>,
    pub outputs: PathBuf,
    pub sign_key: PathBuf,
    /// who publishes the forms and on what legal basis (`controller.json`)
    pub controller: PathBuf,
    pub dpia: Option<Dpia>,
    pub templates: PathBuf,
    #[serde(default)]
    pub held_pages: HeldPages,
    #[serde(default)]
    pub models: Models,
    #[serde(default)]
    pub budget: Budget,
    /// a certificate to sign the PDFs with (password in
    /// $RFPII_CERT_PASSWORD); without one they are not digitally signed
    pub certificate: Option<PathBuf>,
    /// model hosts in Kenya: calls to them are not transfers abroad
    #[serde(default)]
    pub kenya_hosts: Vec<String>,
}

impl JobFile {
    /// The job file at `path`, its paths taken relative to its folder.
    pub fn load(path: &Path) -> Result<JobFile> {
        let text = std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
        let mut j: JobFile = serde_yaml::from_str(&text).with_context(|| format!("{} is not a job file", path.display()))?;
        // an empty name would be taken for the host's default model
        for (role, names) in [("orchestrator", &j.models.orchestrator), ("vision", &j.models.vision)] {
            if names.iter().any(|n| n.trim().is_empty()) {
                anyhow::bail!("{}: models.{} names an empty model; leave the list out for no model", path.display(), role);
            }
        }
        let dir = path.parent().unwrap_or(Path::new("."));
        let abs = |p: &Path| if p.is_absolute() { p.to_path_buf() } else { dir.join(p) };
        j.inputs = j.inputs.iter().map(|i| abs(Path::new(i)).to_string_lossy().into_owned()).collect();
        j.outputs = abs(&j.outputs);
        j.sign_key = abs(&j.sign_key);
        j.controller = abs(&j.controller);
        j.templates = abs(&j.templates);
        j.certificate = j.certificate.as_deref().map(abs);
        Ok(j)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_dpia_lets_processing_start_once_approved_or_60_days_after_submission() {
        let d = |on: Option<&str>, status: Option<&str>| Dpia { reference: "ODPC/1".into(), submitted_on: on.map(String::from), status: status.map(String::from) };
        let now = "2026-10-31T09:00:00+00:00";
        assert!(d(Some("2026-09-01"), Some("submitted")).bars(now).is_none());
        assert!(d(Some("2026-09-02"), None).bars(now).unwrap().contains("59 day(s) ago"));
        assert!(d(None, Some("Deemed approved")).bars(now).is_none());
        assert!(d(Some("2026-10-30"), Some("approved")).bars(now).is_none());
        assert!(d(None, Some("draft")).bars(now).unwrap().contains("no submission date"));
        assert!(Dpia { reference: " ".into(), ..d(None, Some("approved")) }.bars(now).is_some());
    }

    #[test]
    fn a_job_file_is_read_with_its_paths_beside_it() {
        let dir = crate::testutil::tmpdir("jobfile");
        std::fs::write(dir.join("job.yaml"), "inputs: [in/]\noutputs: out\nsign_key: /k/key.pem\ncontroller: c.json\ntemplates: t\nheld_pages: cloud\nmodels: {vision: [gemma4:31b]}\n").unwrap();
        let j = JobFile::load(&dir.join("job.yaml")).unwrap();
        assert!(j.inputs[0].starts_with(dir.to_str().unwrap()) && j.outputs == dir.join("out") && j.sign_key == Path::new("/k/key.pem"));
        assert!(j.held_pages == HeldPages::Cloud && j.models.vision == ["gemma4:31b"] && j.models.orchestrator.is_empty());
        std::fs::write(dir.join("bad.yaml"), "inputs: [in/]\noutputs: out\nsign_key: k\ncontroller: c\ntemplates: t\nheld_pages: sometimes\n").unwrap();
        assert!(JobFile::load(&dir.join("bad.yaml")).is_err());
        // an empty model name is refused, not taken for the host's default
        std::fs::write(dir.join("empty.yaml"), "inputs: [in/]\noutputs: out\nsign_key: k\ncontroller: c\ntemplates: t\nmodels: {orchestrator: [''], vision: ['']}\n").unwrap();
        assert!(JobFile::load(&dir.join("empty.yaml")).unwrap_err().to_string().contains("empty model"));
    }
}
