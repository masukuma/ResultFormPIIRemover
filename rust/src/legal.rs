//! The records the Data Protection Act asks of whoever publishes the forms:
//! who is responsible and on what legal basis (`controller.json`, referenced
//! from every manifest), and every transfer of personal data out of Kenya
//! (`transfers.jsonl`, the record reg. 41(2) of the Data Protection (General)
//! Regulations 2021 requires). See `legal/C-recommendations.md` §2.1, §2.9
//! and §2.16.

use anyhow::{anyhow, bail, Context, Result};
use serde_json::{json, Value};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use crate::proof;

/// Who publishes the forms and why: the controller, its registration with
/// the Data Protection Commissioner, the operator's role, the legal basis,
/// the purpose, whether the use is commercial, the processors (model hosts
/// among them, with their country), and the DPIA.
#[derive(Debug)]
pub struct Controller {
    pub raw: Value,
    pub sha256: String,
}

impl Controller {
    pub fn load(path: &Path) -> Result<Controller> {
        let bytes = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
        let raw: Value = serde_json::from_slice(&bytes).with_context(|| format!("{} is not JSON", path.display()))?;
        let c = Controller { raw, sha256: proof::sha256(&bytes) };
        c.check()?;
        Ok(c)
    }

    /// The fields a published batch can't go without.
    fn check(&self) -> Result<()> {
        let r = &self.raw;
        let text = |k: &str| r[k].as_str().filter(|s| !s.trim().is_empty());
        if r["controller"]["name"].as_str().filter(|s| !s.trim().is_empty()).is_none() {
            bail!("controller.json: controller.name is required");
        }
        for k in ["legal_basis", "purpose"] {
            if text(k).is_none() {
                bail!("controller.json: {} is required", k);
            }
        }
        if !r["commercial"].is_boolean() {
            bail!("controller.json: commercial must be true or false (Data Protection Act s.37)");
        }
        match text("operator_role") {
            Some("controller") | Some("processor") => {}
            _ => bail!("controller.json: operator_role must be \"controller\" or \"processor\""),
        }
        if r["commercial"] == true && text("statutory_authority").is_none() {
            bail!("controller.json: a commercial use needs statutory_authority (Data Protection Act s.37)");
        }
        for (i, p) in r["processors"].as_array().cloned().unwrap_or_default().iter().enumerate() {
            for k in ["name", "country", "service"] {
                if p[k].as_str().filter(|s| !s.trim().is_empty()).is_none() {
                    bail!("controller.json: processors[{}].{} is required", i, k);
                }
            }
        }
        Ok(())
    }

    /// What every manifest carries: the file's hash and its main fields.
    pub fn summary(&self) -> Value {
        let r = &self.raw;
        let mut s = json!({
            "sha256": self.sha256,
            "controller": r["controller"]["name"],
            "operator_role": r["operator_role"],
            "legal_basis": r["legal_basis"],
            "purpose": r["purpose"],
            "commercial": r["commercial"],
        });
        for (k, v) in [("registration_no", &r["controller"]["registration_no"]), ("dpia_reference", &r["dpia"]["reference"])] {
            if !v.is_null() {
                s[k] = v.clone();
            }
        }
        s
    }

    /// Why personal data goes to a processor abroad, as recorded with each
    /// transfer.
    pub fn justification(&self) -> String {
        format!("{} ({})", self.raw["purpose"].as_str().unwrap_or(""), self.raw["legal_basis"].as_str().unwrap_or(""))
    }
}

/// Where a model host is: on this machine, in Kenya (a host the operator
/// names as such), or abroad.
pub fn host_region(host: &str, kenya_hosts: &[String]) -> &'static str {
    let h = crate::vlm::host_name(host);
    if crate::vlm::LOCAL_HOSTS.contains(&h.as_str()) {
        "local"
    } else if kenya_hosts.iter().any(|k| crate::vlm::host_name(k) == h) {
        "kenya"
    } else {
        "foreign"
    }
}

/// The record of transfers out of Kenya: one JSON line per transfer, with
/// what reg. 41(2) asks: "(i) the date and time of the transfer; (ii) the
/// name of the recipient; (iii) the justification for the transfer; and
/// (iv) a description of the personal data transferred". No personal data
/// is written there, only hashes of what was sent.
pub struct Transfers {
    path: PathBuf,
    lock: Mutex<()>,
}

/// One transfer, before it is recorded.
pub struct Transfer<'a> {
    /// the model host (URL or host name) and model
    pub host: &'a str,
    pub model: &'a str,
    /// what the model was asked to do: "review", "orientation", …
    pub purpose: &'a str,
    /// what was sent, in words
    pub data: String,
    /// the document (as published) and page
    pub document: &'a str,
    pub page: usize,
    /// SHA-256 of each image sent
    pub images: Vec<String>,
}

impl Transfers {
    pub fn new(path: PathBuf) -> Transfers {
        Transfers { path, lock: Mutex::new(()) }
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Record `t` if its host is abroad; returns the transfer's id, or None
    /// for a host on this machine or in Kenya.
    pub fn record(&self, t: &Transfer, kenya_hosts: &[String], controller: Option<&Controller>) -> Result<Option<String>> {
        if host_region(t.host, kenya_hosts) != "foreign" {
            return Ok(None);
        }
        let at = crate::process::utc_now();
        let id = proof::sha256(format!("{}|{}|{}|{}|{}", at, t.document, t.page, t.purpose, t.images.join(",")).as_bytes())[..16].to_string();
        let justification = match controller {
            Some(c) => c.justification(),
            None => format!("{} of redacted pages before publication (no controller record given)", t.purpose),
        };
        let entry = json!({
            "id": id,
            "at": at,
            "recipient": {"host": crate::vlm::host_name(t.host), "model": t.model, "region": "foreign"},
            "justification": justification,
            "data": t.data,
            "purpose": t.purpose,
            "document": t.document,
            "page": t.page,
            "images_sha256": t.images,
            "controller_sha256": controller.map(|c| c.sha256.clone()),
        });
        let _g = self.lock.lock().map_err(|_| anyhow!("transfer record lock poisoned"))?;
        if let Some(d) = self.path.parent() {
            std::fs::create_dir_all(d)?;
        }
        let mut f = std::fs::OpenOptions::new().create(true).append(true).open(&self.path).with_context(|| format!("opening {}", self.path.display()))?;
        writeln!(f, "{}", serde_json::to_string(&entry)?)?;
        Ok(Some(id))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn controller(v: Value) -> Result<Controller> {
        let c = Controller { sha256: proof::sha256(v.to_string().as_bytes()), raw: v };
        c.check().map(|_| c)
    }

    fn valid() -> Value {
        json!({"controller": {"name": "Example Observer Group", "registration_no": "ODPC/123"},
               "operator_role": "controller", "legal_basis": "s.30(1)(b)(vii) legitimate interests",
               "purpose": "republishing polling-station results forms", "commercial": false,
               "processors": [{"name": "Ollama", "country": "US", "service": "vision model hosting"}]})
    }

    #[test]
    fn a_controller_record_needs_its_basis_purpose_and_role() {
        assert!(controller(valid()).is_ok());
        for k in ["legal_basis", "purpose", "operator_role", "commercial"] {
            let mut v = valid();
            v.as_object_mut().unwrap().remove(k);
            assert!(controller(v).is_err(), "{} missing", k);
        }
        let mut v = valid();
        v["commercial"] = json!(true);
        assert!(controller(v.clone()).unwrap_err().to_string().contains("s.37"));
        v["statutory_authority"] = json!("Elections Act s.39(1C)(c)");
        assert!(controller(v).is_ok());
        let mut v = valid();
        v["processors"][0].as_object_mut().unwrap().remove("country");
        assert!(controller(v).is_err());
        let s = controller(valid()).unwrap().summary();
        assert!(s["controller"] == "Example Observer Group" && s["registration_no"] == "ODPC/123" && s["sha256"].as_str().unwrap().len() == 64);
    }

    #[test]
    fn hosts_are_local_in_kenya_or_abroad() {
        let ke = vec!["gpu.example.co.ke".to_string()];
        assert_eq!(host_region("http://localhost:11434", &ke), "local");
        assert_eq!(host_region("https://gpu.example.co.ke", &ke), "kenya");
        assert_eq!(host_region("https://ollama.com", &ke), "foreign");
    }

    #[test]
    fn only_transfers_abroad_are_recorded_with_what_reg_41_asks() {
        let dir = crate::testutil::tmpdir("transfers");
        let tr = Transfers::new(dir.join("transfers.jsonl"));
        let c = controller(valid()).unwrap();
        let t = |host| Transfer { host, model: "gemma4:31b", purpose: "review", data: "the redacted page".into(), document: "a.pdf", page: 2, images: vec!["ab".repeat(32)] };
        assert_eq!(tr.record(&t("http://localhost:11434"), &[], Some(&c)).unwrap(), None);
        let id = tr.record(&t("https://ollama.com"), &[], Some(&c)).unwrap().expect("a transfer abroad");
        let lines: Vec<Value> = std::fs::read_to_string(tr.path()).unwrap().lines().map(|l| serde_json::from_str(l).unwrap()).collect();
        assert_eq!(lines.len(), 1);
        let e = &lines[0];
        assert_eq!(e["id"], id.as_str());
        // (i) date and time, (ii) recipient, (iii) justification, (iv) the data
        assert!(e["at"].as_str().unwrap().ends_with("+00:00") && e["recipient"]["host"] == "ollama.com" && e["recipient"]["region"] == "foreign");
        assert!(e["justification"].as_str().unwrap().contains("republishing") && e["data"] == "the redacted page");
        assert!(e["page"] == 2 && e["controller_sha256"] == c.sha256.as_str());
    }
}
