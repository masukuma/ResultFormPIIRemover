//! Vision-language model assistance through Ollama (local or Ollama Cloud).
//!
//! audit   After redaction, ask whether any complete ID number, phone number
//!         or unmasked signature is still visible anywhere on the page. A
//!         "yes" sends the form to review. A gate, not a guarantee.
//! locate  Find signatures, ID numbers and phone numbers on a page: to draft
//!         a new template's zones (`rfpii propose`), or to redact a page no
//!         template matches into review/vlm/ for a person to approve. Never
//!         auto-published.
//!
//! Privacy: the audit only ever sends the *redacted* page. Sending an
//! unredacted form to a non-local host is refused unless explicitly allowed.
//! Every verdict is recorded in the manifest with the model, host and a hash
//! of the exact image sent; a VLM never edits pixels itself.

use anyhow::{anyhow, bail, Result};
use base64::Engine;
use opencv::core::{Mat, Vector};
use opencv::imgcodecs;
use opencv::imgproc;
use opencv::prelude::*;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::time::Duration;

use crate::util;

pub const CLOUD_HOST: &str = "https://ollama.com";
pub const DEFAULT_CLOUD_MODEL: &str = "qwen3-vl:235b";
pub const DEFAULT_LOCAL_MODEL: &str = "qwen3.8:27b";
pub const LABELS: &[&str] = &["signature", "id_number", "phone_number"];
const LOCAL_HOSTS: &[&str] = &["localhost", "127.0.0.1", "::1"];

pub const AUDIT_PROMPT: &str = "You are checking a Kenyan election results form before it is published.
Black rectangles are intentional redactions. Each ID/passport number has 3 of its
characters covered by black boxes, and each phone number keeps its first 4 digits
(e.g. 0734) and has 3 of its remaining 6 digits covered. So most digits of these
numbers are meant to be visible; what must never be visible is a COMPLETE number.
Each signature has a black band across its middle: the tops and bottoms of its
strokes showing above and below the band are intended (they show that the form
was signed). What must never be visible is a signature with no band across it.
Names of officials and agents are allowed to be visible. Official stamps (the round
\"I.E.B.C. PRESIDING OFFICER\" stamp) must never be covered, so writing that lies inside
a stamp is left visible on purpose: ignore anything inside a stamp.

Look at the whole page, including tables, margins, stamps and handwritten comments.
Answer:
- id_number_visible: is any national ID or passport number fully readable, with no black box hiding any of its characters?
- phone_number_visible: is any phone number (07XXXXXXXX, 01XXXXXXXX, +254...) fully readable, with no black box hiding any digit?
- signature_visible: is any handwritten signature visible with no black band across its middle (outside stamps)?
- stamp_covered: does any black box cover part of an official stamp?
- details: one short sentence naming where, if anything is wrong.";

const LOCATE_PROMPT: &str = "This image is {w}x{h} pixels and shows a Kenyan election results form.
Find every handwritten signature, every national ID or passport number, and every
phone number (in tables or anywhere else, e.g. comments or margins).
Return one item per occurrence with a tight bounding box [x1, y1, x2, y2] in pixel
coordinates of this image. Do not return names, vote counts, dates or printed text.";

fn audit_schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "id_number_visible": {"type": "boolean"},
            "phone_number_visible": {"type": "boolean"},
            "signature_visible": {"type": "boolean"},
            "stamp_covered": {"type": "boolean"},
            "details": {"type": "string"},
        },
        "required": ["id_number_visible", "phone_number_visible", "signature_visible", "stamp_covered", "details"],
    })
}

fn locate_schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "label": {"type": "string", "enum": LABELS},
                        "bbox": {"type": "array", "items": {"type": "number"}, "minItems": 4, "maxItems": 4},
                    },
                    "required": ["label", "bbox"],
                },
            }
        },
        "required": ["items"],
    })
}

#[derive(Clone, Debug)]
pub struct VlmSettings {
    pub host: String,
    pub model: String, // "" = default for the host
    pub max_side: i32,
    pub coords: String, // "pixel" | "norm1000" | "auto"
    pub timeout: f64,
    pub api_key: Option<String>,
    pub allow_remote_unredacted: bool,
}

impl Default for VlmSettings {
    fn default() -> Self {
        let api_key = std::env::var("OLLAMA_API_KEY").ok().filter(|k| !k.is_empty());
        let host = std::env::var("OLLAMA_HOST").ok().filter(|h| !h.is_empty()).unwrap_or_else(|| {
            if api_key.is_some() { CLOUD_HOST.to_string() } else { "http://localhost:11434".to_string() }
        });
        VlmSettings { host, model: String::new(), max_side: 1288, coords: "auto".into(), timeout: 300.0, api_key, allow_remote_unredacted: false }
    }
}

pub struct AuditResult {
    pub ok: bool,
    pub findings: Value,
    pub record: Value, // for the manifest
}

#[derive(Clone, Debug)]
pub struct Box {
    pub label: String,
    pub x0: f64,
    pub y0: f64,
    pub x1: f64,
    pub y1: f64,
}

pub struct Ollama {
    pub s: VlmSettings,
}

fn hostname(url: &str) -> String {
    let rest = url.split("://").nth(1).unwrap_or(url);
    let hostport = rest.split('/').next().unwrap_or("");
    if let Some(h) = hostport.strip_prefix('[') {
        return h.split(']').next().unwrap_or("").to_string();
    }
    hostport.split(':').next().unwrap_or("").to_string()
}

impl Ollama {
    pub fn new(mut s: VlmSettings) -> Result<Self> {
        if !s.host.contains("://") {
            s.host = format!("http://{}", s.host);
        }
        let local = LOCAL_HOSTS.contains(&hostname(&s.host).as_str());
        if s.model.is_empty() {
            s.model = if local { DEFAULT_LOCAL_MODEL.into() } else { DEFAULT_CLOUD_MODEL.into() };
        }
        if s.coords == "auto" {
            let m = s.model.to_lowercase();
            s.coords = if m.contains("qwen3") || m.contains("qwen2-vl") { "norm1000".into() } else { "pixel".into() };
        }
        if !local && s.host.trim_end_matches('/') == CLOUD_HOST && s.api_key.is_none() {
            bail!("Ollama Cloud needs OLLAMA_API_KEY (create one at ollama.com/settings/keys)");
        }
        Ok(Ollama { s })
    }

    pub fn local(&self) -> bool {
        LOCAL_HOSTS.contains(&hostname(&self.s.host).as_str())
    }

    fn guard_unredacted(&self) -> Result<()> {
        if !self.local() && !self.s.allow_remote_unredacted {
            bail!(
                "refusing to send an unredacted form to {}; use a local model or pass --allow-remote-unredacted if that is an approved data transfer",
                self.s.host
            );
        }
        Ok(())
    }

    pub fn chat_url(&self) -> String {
        format!("{}/api/chat", self.s.host.trim_end_matches('/'))
    }

    /// The API key goes to remote hosts only.
    pub fn authorization(&self) -> Option<String> {
        match (&self.s.api_key, self.local()) {
            (Some(k), false) => Some(format!("Bearer {}", k)),
            _ => None,
        }
    }

    /// Long side at most `max_side`, each side a multiple of 28 px.
    fn resize(&self, img: &Mat) -> Result<(Mat, (f64, f64))> {
        let (h, w) = (img.rows() as f64, img.cols() as f64);
        let f = (self.s.max_side as f64 / h.max(w)).min(1.0);
        let nw = (((w * f) as i64) / 28 * 28).max(28) as i32;
        let nh = (((h * f) as i64) / 28 * 28).max(28) as i32;
        Ok((util::resize(img, nw, nh, imgproc::INTER_AREA)?, (nw as f64 / w, nh as f64 / h)))
    }

    /// The parsed reply and the sha256 of the exact image sent.
    pub fn ask(&self, prompt: &str, img: &Mat, schema: Value) -> Result<(Value, String)> {
        let mut buf = Vector::<u8>::new();
        let params: Vector<i32> = vec![imgcodecs::IMWRITE_JPEG_QUALITY, 90].into_iter().collect();
        if !imgcodecs::imencode(".jpg", img, &mut buf, &params)? {
            bail!("encode failed");
        }
        let sent = buf.to_vec();
        let body = json!({
            "model": self.s.model,
            "messages": [{"role": "user", "content": prompt, "images": [base64::engine::general_purpose::STANDARD.encode(&sent)]}],
            "format": schema,
            "stream": false,
            "options": {"temperature": 0},
        });
        let agent: ureq::Agent = ureq::Agent::config_builder().timeout_global(Some(Duration::from_secs_f64(self.s.timeout))).build().into();
        let mut req = agent.post(self.chat_url()).header("Content-Type", "application/json");
        if let Some(a) = self.authorization() {
            req = req.header("Authorization", a);
        }
        let mut resp = req.send_json(&body).map_err(|e| anyhow!("Ollama at {}: {e}", self.s.host))?;
        let reply: Value = resp.body_mut().read_json().map_err(|e| anyhow!("Ollama reply: {e}"))?;
        let content = reply["message"]["content"].as_str().ok_or_else(|| anyhow!("Ollama reply has no message"))?;
        Ok((serde_json::from_str(content)?, hex::encode(Sha256::digest(&sent))))
    }

    /// Audit a *redacted* page (safe to send anywhere).
    pub fn audit(&self, img: &Mat) -> Result<AuditResult> {
        let (small, _) = self.resize(img)?;
        let (f, sent) = self.ask(AUDIT_PROMPT, &small, audit_schema())?;
        let leak = ["id_number_visible", "phone_number_visible", "signature_visible", "stamp_covered"].iter().any(|k| f.get(*k).and_then(|v| v.as_bool()).unwrap_or(false));
        let record = json!({"model": self.s.model, "host": hostname(&self.s.host),
                            "prompt_sha256": hex::encode(Sha256::digest(AUDIT_PROMPT.as_bytes())),
                            "image_sent_sha256": sent, "verdict": f});
        Ok(AuditResult { ok: !leak, findings: f, record })
    }

    /// Find PII on an *unredacted* page (guarded for remote hosts).
    pub fn locate(&self, img: &Mat) -> Result<Vec<Box>> {
        self.guard_unredacted()?;
        let (small, (sx, sy)) = self.resize(img)?;
        let (h, w) = (small.rows() as f64, small.cols() as f64);
        let prompt = LOCATE_PROMPT.replace("{w}", &format!("{}", w as i64)).replace("{h}", &format!("{}", h as i64));
        let (out, _) = self.ask(&prompt, &small, locate_schema())?;
        let mut boxes = Vec::new();
        for it in out["items"].as_array().cloned().unwrap_or_default() {
            let label = it["label"].as_str().unwrap_or("");
            let bb: Vec<f64> = it["bbox"].as_array().map(|a| a.iter().filter_map(|v| v.as_f64()).collect()).unwrap_or_default();
            if !LABELS.contains(&label) || bb.len() != 4 {
                continue;
            }
            let (mut x0, mut y0, mut x1, mut y1) = (bb[0], bb[1], bb[2], bb[3]);
            if self.s.coords == "norm1000" {
                (x0, x1, y0, y1) = (x0 * w / 1000.0, x1 * w / 1000.0, y0 * h / 1000.0, y1 * h / 1000.0);
            }
            let (x0, x1) = (x0.min(x1), x0.max(x1));
            let (y0, y1) = (y0.min(y1), y0.max(y1));
            if x1 - x0 < 2.0 || y1 - y0 < 2.0 {
                continue;
            }
            boxes.push(Box { label: label.into(), x0: x0 / sx, y0: y0 / sy, x1: x1 / sx, y1: y1 / sy });
        }
        Ok(boxes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hosts() {
        assert_eq!(hostname("http://localhost:11434"), "localhost");
        assert_eq!(hostname("https://ollama.com"), "ollama.com");
        assert_eq!(hostname("http://[::1]:11434/x"), "::1");
        let o = Ollama::new(VlmSettings { host: "localhost:11434".into(), api_key: None, ..VlmSettings::default() }).unwrap();
        assert!(o.local());
        assert_eq!(o.s.coords, "norm1000");
        let r = Ollama::new(VlmSettings { host: "http://10.0.0.5:11434".into(), model: "qwen2.5vl:7b".into(), api_key: None, ..VlmSettings::default() }).unwrap();
        assert!(r.guard_unredacted().is_err());
        assert_eq!(r.s.coords, "pixel");
    }
}

#[cfg(test)]
#[path = "tests/vlm.rs"]
mod py_tests;
