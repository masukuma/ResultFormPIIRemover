//! Vision-language model assistance through Ollama (local or Ollama Cloud).
//!
//! audit   After redaction, ask whether any complete ID number, phone number
//!         or unmasked signature is still visible anywhere on the page. A
//!         "yes" sends the form to review. A gate, not a guarantee.
//! locate  Find signatures, ID numbers and phone numbers on a page: to draft
//!         a new template's zones (`rfpii propose`), or to redact a page no
//!         template matches into review/vlm/ for a person to approve. Never
//!         auto-published.
//! orient  Which way up a redacted page stands (Clef), for photos that carry
//!         no EXIF orientation tag.
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

use crate::masking::MaskSettings;
use crate::util;

pub const CLOUD_HOST: &str = "https://ollama.com";
pub const DEFAULT_CLOUD_MODEL: &str = "gemma4:31b";
/// The model that confirms each spot the first one finds (`find`). On 88
/// spots gemma4 flagged on redacted Forms 34A it kept 26 of 31 real misses
/// and dropped 53 of 56 false alarms; minimax-m3.1-flash did worse at both.
pub const DEFAULT_CLOUD_CHECKER: &str = "gemma4:31b";
pub const DEFAULT_LOCAL_MODEL: &str = "qwen3.8:27b";
pub const LABELS: &[&str] = &["signature", "id_number", "phone_number"];
pub const LOCAL_HOSTS: &[&str] = &["localhost", "127.0.0.1", "::1"];

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


/// Handwriting must stay legible to the model: the area is sent at most
/// this many px on its long side.
const FIND_MAX_SIDE: f64 = 1800.0;

/// Asked of a *redacted* page: what still shows that the rules hide.
pub const FIND_PROMPT: &str = "This is part of a Kenyan election results form, {w}x{h} pixels, after redaction.
Black rectangles are intentional redactions. The rules are:
- every national ID / passport number must have at least 3 of its characters hidden under black;
- every phone number must have 3 of its last 6 digits hidden under black (the first 4 digits may show);
- every handwritten signature must have a black band across its middle (the tops and bottoms of its strokes may show).
Names, party names, dates, vote counts, printed text and writing inside the round official stamp are allowed to show.
Find every place on the image that breaks these rules: an ID or phone number whose digits are all readable,
or a handwritten signature (in tables, next to names, in the comments, anywhere) with no black band across it.
Return one item per problem with its label (id_number, phone_number or signature) and a tight bounding box
[x1, y1, x2, y2] in pixels of this image. Return an empty list if nothing breaks the rules.";

/// `FIND_PROMPT` with the rules in force: writing inside a stamp follows
/// them like any other (`--mask-stamps`); IDs or phone numbers are hidden
/// whole (`--full-ids`, `--full-phones`).
pub fn find_prompt(ms: &MaskSettings) -> String {
    let mut p = FIND_PROMPT.to_string();
    if ms.masks_stamps() {
        p = p.replace(
            "Names, party names, dates, vote counts, printed text and writing inside the round official stamp are allowed to show.",
            "Names, party names, dates, vote counts and printed text are allowed to show. Writing inside the round official stamp follows the same rules, and black boxes over part of a stamp are intended.",
        );
    }
    if ms.full_ids {
        p = p.replace("must have at least 3 of its characters hidden under black;", "must be hidden under black entirely;");
    }
    if ms.full_phones {
        p = p.replace("must have 3 of its last 6 digits hidden under black (the first 4 digits may show);", "must be hidden under black entirely;");
    }
    if ms.full_ids || ms.full_phones {
        p = p.replace("an ID or phone number whose digits are all readable", "an ID or phone number showing more of its digits than the rules allow");
    }
    p
}

/// Asked of a zoomed crop around each spot `FIND_PROMPT` returned.
pub const CHECK_PROMPT: &str = "This crop is from a Kenyan election results form that has been redacted: black rectangles are redactions.
Look only at the writing inside the red box.
- unmasked_signature: true only if the red box contains a handwritten signature (a scribble or stylised initials,
  not a name written in capital letters, not a date, not printed text, not a stamp) that has NO black rectangle over its middle.
  A signature with a black bar or black blocks across it is masked: answer false.
- complete_number: true only if the red box contains a handwritten ID/passport number or phone number with ALL of its digits
  readable and no black box over any of them. A phone number showing only its first 4 digits followed by black is masked: false.
  Dates (e.g. 09-08-2022) and vote counts are not ID or phone numbers: false.
Answer with JSON: {\"unmasked_signature\": bool, \"complete_number\": bool, \"what\": \"a few words\"}";

/// The same check for a decision model (Clef): what the writing in the red
/// box is, scored over every option at once.
const DECIDE_STATE: &str = "A zoomed crop of a photographed Kenyan election result form after redaction: solid black rectangles are masks \
added on purpose. Names of officials and agents, dates, vote counts, printed text and official stamps are allowed to show.";

fn decide_questions() -> Value {
    json!({"what": {"type": "choice", "instructions": "What is the handwriting inside the red box?", "criteria": {
        "unmasked_signature": "A handwritten signature (a scribble, flourish or stylised initials, not readable as words) with no black mask across it",
        "masked_signature": "A signature with a black bar or black blocks across it",
        "name": "A name or other words written in readable letters (capitals or joined-up), even if signed beside",
        "date": "A date or a vote count",
        "complete_number": "An ID, passport or phone number with every digit readable",
        "masked_number": "An ID, passport or phone number with some digits hidden under black",
        "printed": "Printed text, lines, a stamp, or nothing handwritten"}}})
}

/// Which way a page is turned, for a decision model (Clef).
const ORIENT_STATE: &str = "A photo of a Kenyan election result form. Solid black rectangles are redactions added on purpose.";

/// The options, in quarter turns clockwise from upright.
const TURNS: [&str; 4] = ["upright", "lying_on_right_side", "upside_down", "lying_on_left_side"];

fn orient_questions() -> Value {
    json!({"turn": {"type": "choice", "instructions": "Which way is the form turned in this image? Judge by its printed text: the title, headings and table labels.", "criteria": {
        "upright": "Upright: printed text reads normally, left to right, the title at the top",
        "lying_on_right_side": "Turned a quarter clockwise: lines of text run downwards, from the top of the image to the bottom; the title is along the right edge",
        "upside_down": "Upside down: printed text is inverted, the title at the bottom",
        "lying_on_left_side": "Turned a quarter anticlockwise: lines of text run upwards, from the bottom of the image to the top; the title is along the left edge"}}})
}

/// Long side of the page Clef is shown when asked which way it is turned.
const ORIENT_MAX_SIDE: f64 = 1024.0;

/// Clef's probability of "upright" at or above which a page is taken as
/// upright. On 13 redacted pages of six printings (2013 to 2022), each in
/// its four quarter turns, upright ones scored 0.94 or more and turned ones
/// 0.09 or less.
pub const UPRIGHT_AT: f64 = 0.5;

fn find_schema() -> Value {
    json!({"type": "object", "properties": {"items": {"type": "array", "items": {"type": "object", "properties": {
        "label": {"type": "string", "enum": LABELS},
        "bbox": {"type": "array", "items": {"type": "number"}, "minItems": 4, "maxItems": 4}}, "required": ["label", "bbox"]}}},
        "required": ["items"]})
}

fn check_schema() -> Value {
    json!({"type": "object", "properties": {"unmasked_signature": {"type": "boolean"}, "complete_number": {"type": "boolean"}, "what": {"type": "string"}},
        "required": ["unmasked_signature", "complete_number", "what"]})
}

/// The JSON in a reply: many models wrap it in a markdown fence or prose
/// whatever `format` asks for.
pub fn loose_json(content: &str) -> Option<Value> {
    let s = content.trim();
    let s = match (s.find("```"), s.rfind("```")) {
        (Some(a), Some(b)) if b > a => {
            let inner = &s[a + 3..b];
            inner.strip_prefix("json").unwrap_or(inner)
        }
        _ => s,
    };
    if let Ok(v) = serde_json::from_str::<Value>(s.trim()) {
        return Some(v);
    }
    // the outermost value: whichever bracket opens first
    let mut pairs = [('[', ']'), ('{', '}')];
    pairs.sort_by_key(|(o, _)| s.find(*o).unwrap_or(usize::MAX));
    for (o, c) in pairs {
        if let (Some(a), Some(b)) = (s.find(o), s.rfind(c)) {
            if b > a {
                if let Ok(v) = serde_json::from_str::<Value>(&s[a..=b]) {
                    return Some(v);
                }
            }
        }
    }
    None
}

/// What a label names, whatever the model's words for it.
fn label_kind(label: &str) -> Option<&'static str> {
    let l = label.to_lowercase();
    if l.contains("sign") {
        Some("signature")
    } else if l.contains("phone") || l.contains("tel") {
        Some("phone_number")
    } else if l.contains("id") || l.contains("passport") || l.contains("national") {
        Some("id_number")
    } else {
        None
    }
}

/// Boxes in a reply, as image px of a `w` x `h` image. Gemini-style models
/// answer `box_2d` as [y1, x1, y2, x2] on a 0-1000 grid; others `bbox` in
/// the convention `coords` names.
pub fn reply_boxes(v: &Value, w: f64, h: f64, coords: &str) -> Vec<crate::pipeline::Found> {
    let items: Vec<Value> = match v {
        Value::Array(a) => a.clone(),
        Value::Object(o) => o.get("items").and_then(|x| x.as_array()).cloned().unwrap_or_default(),
        _ => vec![],
    };
    let mut out = Vec::new();
    for it in items {
        let Some(kind) = label_kind(it.get("label").and_then(|l| l.as_str()).unwrap_or("")) else { continue };
        let Some((key, b)) = ["box_2d", "bbox", "box", "bounding_box"].iter().find_map(|k| {
            let b: Vec<f64> = it.get(*k)?.as_array()?.iter().filter_map(|x| x.as_f64()).collect();
            (b.len() == 4).then_some((*k, b))
        }) else {
            continue;
        };
        let (x0, y0, x1, y1) = if key == "box_2d" {
            (b[1] * w / 1000.0, b[0] * h / 1000.0, b[3] * w / 1000.0, b[2] * h / 1000.0)
        } else if coords == "norm1000" {
            (b[0] * w / 1000.0, b[1] * h / 1000.0, b[2] * w / 1000.0, b[3] * h / 1000.0)
        } else {
            (b[0], b[1], b[2], b[3])
        };
        let (x0, x1, y0, y1) = (x0.min(x1).max(0.0), x0.max(x1).min(w), y0.min(y1).max(0.0), y0.max(y1).min(h));
        if x1 - x0 >= 2.0 && y1 - y0 >= 2.0 {
            out.push(crate::pipeline::Found { label: kind.into(), rect: [x0, y0, x1, y1] });
        }
    }
    out
}

// Requests in flight to the model host, across all workers: Ollama Cloud
// answers 429 to an account that sends too many at once.
static LIMIT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(4);
static INFLIGHT: std::sync::Mutex<usize> = std::sync::Mutex::new(0);
static FREED: std::sync::Condvar = std::sync::Condvar::new();

pub fn set_concurrency(n: usize) {
    LIMIT.store(n.max(1), std::sync::atomic::Ordering::SeqCst);
}

struct Permit;

impl Permit {
    fn take() -> Permit {
        let mut n = INFLIGHT.lock().unwrap();
        while *n >= LIMIT.load(std::sync::atomic::Ordering::SeqCst) {
            n = FREED.wait(n).unwrap();
        }
        *n += 1;
        Permit
    }
}

impl Drop for Permit {
    fn drop(&mut self) {
        *INFLIGHT.lock().unwrap() -= 1;
        FREED.notify_one();
    }
}

/// Reviews redacted pages: `finder` looks for what still shows, `checker`
/// confirms each spot on a zoomed crop.
pub struct Reviewer {
    pub finder: Ollama,
    pub checker: Ollama,
}

pub struct FindResult {
    pub found: Vec<crate::pipeline::Found>,
    pub record: Value, // for the manifest
}

/// Share of pure black (redacted) pixels in `r` (image px x0, y0, x1, y1).
fn black_share(gray: &Mat, r: [f64; 4]) -> Result<f64> {
    let (w, h) = (gray.cols(), gray.rows());
    let (x0, y0) = ((r[0].max(0.0) as i32).min(w), (r[1].max(0.0) as i32).min(h));
    let (x1, y1) = ((r[2].max(0.0) as i32).min(w), (r[3].max(0.0) as i32).min(h));
    if x1 <= x0 || y1 <= y0 {
        return Ok(0.0);
    }
    let c = Mat::roi(gray, opencv::core::Rect::new(x0, y0, x1 - x0, y1 - y0))?.try_clone()?;
    let v = util::bytes(&c)?;
    Ok(v.iter().filter(|&&p| p <= 12).count() as f64 / v.len() as f64)
}

fn jpeg(img: &Mat) -> Result<Vec<u8>> {
    let mut buf = Vector::<u8>::new();
    let params: Vector<i32> = vec![imgcodecs::IMWRITE_JPEG_QUALITY, 90].into_iter().collect();
    if !imgcodecs::imencode(".jpg", img, &mut buf, &params)? {
        bail!("encode failed");
    }
    Ok(buf.to_vec())
}

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
    /// A decision model's probability at or above which `check` confirms a
    /// spot (a chat model answers 0 or 1).
    pub check_threshold: f64,
}

impl Default for VlmSettings {
    fn default() -> Self {
        let api_key = std::env::var("OLLAMA_API_KEY").ok().filter(|k| !k.is_empty());
        let host = std::env::var("OLLAMA_HOST").ok().filter(|h| !h.is_empty()).unwrap_or_else(|| {
            if api_key.is_some() { CLOUD_HOST.to_string() } else { "http://localhost:11434".to_string() }
        });
        VlmSettings { host, model: String::new(), max_side: 1288, coords: "auto".into(), timeout: 300.0, api_key, allow_remote_unredacted: false, check_threshold: 0.5 }
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
    /// A decision model (Clef): answers typed questions through
    /// /v1/systemone instead of chatting. Set by `probe`.
    pub decision: bool,
}

/// The host name of a model host's URL (or of a bare host name).
pub fn host_name(url: &str) -> String {
    hostname(url)
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
        Ok(Ollama { s, decision: false })
    }

    /// Ask the host what the model is (/api/show): a decision model is asked
    /// through /v1/systemone. Fails if the host doesn't know the model, so a
    /// misspelt model stops a run before its first page.
    pub fn probe(&mut self) -> Result<()> {
        let url = format!("{}/api/show", self.s.host.trim_end_matches('/'));
        let reply = self.post(&url, &json!({"model": self.s.model}))?;
        self.decision = reply["capabilities"].as_array().is_some_and(|c| c.iter().any(|v| v == "decision"));
        Ok(())
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

    /// One request, retried when the host is busy (429, 5xx) or the
    /// connection drops; the reply.
    fn post(&self, url: &str, body: &Value) -> Result<Value> {
        let _permit = Permit::take();
        let agent: ureq::Agent = ureq::Agent::config_builder().timeout_global(Some(Duration::from_secs_f64(self.s.timeout))).http_status_as_error(false).build().into();
        let mut last = String::new();
        for attempt in 0..6u32 {
            let mut req = agent.post(url).header("Content-Type", "application/json");
            if let Some(a) = self.authorization() {
                req = req.header("Authorization", a);
            }
            match req.send_json(body) {
                Ok(mut resp) => {
                    let status = resp.status().as_u16();
                    if status == 200 {
                        return resp.body_mut().read_json().map_err(|e| anyhow!("Ollama reply: {e}"));
                    }
                    last = format!("HTTP {}", status);
                    if !(status == 429 || status >= 500) {
                        // the host's reason, e.g. a decision model asked to chat
                        let why = resp.body_mut().read_to_string().unwrap_or_default();
                        bail!("Ollama at {}: {} {:.200}", self.s.host, last, why);
                    }
                }
                Err(e) => last = e.to_string(),
            }
            std::thread::sleep(Duration::from_secs((15u64 << attempt).min(240)));
        }
        bail!("Ollama at {}: gave up after retries ({})", self.s.host, last)
    }

    fn chat(&self, prompt: &str, sent: &[u8], schema: Value) -> Result<String> {
        let reply = self.post(&self.chat_url(), &json!({
            "model": self.s.model,
            "messages": [{"role": "user", "content": prompt, "images": [base64::engine::general_purpose::STANDARD.encode(sent)]}],
            "format": schema,
            "stream": false,
            "options": {"temperature": 0},
        }))?;
        reply["message"]["content"].as_str().map(String::from).ok_or_else(|| anyhow!("Ollama reply has no message"))
    }

    /// A decision model's answer to `questions` about `sent`, given what it
    /// shows (`state`): per question, the probability of each option.
    fn decide(&self, sent: &[u8], state: &str, questions: Value) -> Result<Value> {
        let url = format!("{}/v1/systemone", self.s.host.trim_end_matches('/'));
        let reply = self.post(&url, &json!({
            "model": self.s.model,
            "state": state,
            "images": [base64::engine::general_purpose::STANDARD.encode(sent)],
            "questions": questions,
        }))?;
        reply.get("answers").cloned().ok_or_else(|| anyhow!("{} answered without answers", self.s.model))
    }

    /// The prompt the checker is asked, for the record.
    fn check_prompt(&self) -> String {
        if self.decision { format!("{}\n{}", DECIDE_STATE, decide_questions()) } else { CHECK_PROMPT.to_string() }
    }

    /// What still shows on a *redacted* page inside `area` (image px x, y,
    /// w, h) that the rules hide: spots this model finds, each confirmed by
    /// `checker` on a zoomed crop (the spot boxed in red). Boxes in image px.
    /// A reply that can't be read confirms the spot: masking more is the
    /// safe side. `ms`: the rules in force (`find_prompt`).
    pub fn find(&self, redacted: &Mat, area: [i32; 4], checker: &Ollama, ms: &MaskSettings) -> Result<FindResult> {
        let [ax, ay, aw, ah] = area;
        let crop = Mat::roi(redacted, opencv::core::Rect::new(ax, ay, aw, ah))?.try_clone()?;
        let f = (FIND_MAX_SIDE / aw.max(ah) as f64).min(1.0);
        let small = if f < 1.0 { util::resize_by(&crop, f, imgproc::INTER_AREA)? } else { crop.clone() };
        let (w, h) = (small.cols() as f64, small.rows() as f64);
        let sent = jpeg(&small)?;
        let rules = find_prompt(ms);
        let prompt = rules.replace("{w}", &format!("{}", w as i64)).replace("{h}", &format!("{}", h as i64));
        let reply = self.chat(&prompt, &sent, find_schema())?;
        let spots = match loose_json(&reply) {
            Some(v) => reply_boxes(&v, w, h, &self.s.coords),
            None => bail!("{} answered something that isn't JSON: {:.120}", self.s.model, reply),
        };
        let mut found = Vec::new();
        let mut cands = Vec::new();
        let gray = util::gray(redacted)?;
        for sp in spots {
            // whole px: what is masked is exactly what the manifest records
            let r = [sp.rect[0] / f + ax as f64, sp.rect[1] / f + ay as f64, sp.rect[2] / f + ax as f64, sp.rect[3] / f + ay as f64].map(f64::round);
            // a spot that is largely black already is something masked (a
            // banded signature, a number with digits hidden): not asked about
            let black = black_share(&gray, r)?;
            let mut cand = json!({"label": sp.label, "box": r.iter().map(|&v| crate::pyjson::round_n(v, 1)).collect::<Vec<_>>()});
            let keep = if black >= 0.08 {
                cand["what"] = json!(format!("{:.0}% already black", black * 100.0));
                false
            } else {
                // the checker must see what the finder named: a name taken
                // for a number, or a header for a phone, is confirmed as
                // neither
                let (sig, num, what) = checker.check(redacted, r)?;
                let p = if sp.label == "signature" { sig } else { num };
                cand["what"] = json!(what);
                if checker.decision {
                    cand["p"] = json!(crate::pyjson::round_n(p, 4));
                }
                p >= checker.s.check_threshold
            };
            cand["kept"] = json!(keep);
            cands.push(cand);
            if keep {
                found.push(crate::pipeline::Found { label: sp.label, rect: r });
            }
        }
        let mut record = json!({"model": self.s.model, "checker": checker.s.model, "host": hostname(&self.s.host),
                                "find_prompt_sha256": hex::encode(Sha256::digest(rules.as_bytes())),
                                "check_prompt_sha256": hex::encode(Sha256::digest(checker.check_prompt().as_bytes())),
                                "area": area, "image_sent_sha256": hex::encode(Sha256::digest(&sent)), "spots": cands});
        if checker.decision {
            record["check_threshold"] = json!(checker.s.check_threshold);
        }
        Ok(FindResult { found, record })
    }

    /// Is something the rules hide still showing in `r` (image px x0, y0,
    /// x1, y1) of a *redacted* page: the probability of an unmasked
    /// signature, of a complete number, and what it is. Asked of a crop 80
    /// px wider all round, zoomed 2x, with `r` boxed in red. A chat model
    /// answers 0 or 1, and a reply that can't be read says yes.
    pub fn check(&self, redacted: &Mat, r: [f64; 4]) -> Result<(f64, f64, String)> {
        let (iw, ih) = (redacted.cols(), redacted.rows());
        let x0 = ((r[0] - 80.0).max(0.0) as i32).min(iw - 1);
        let y0 = ((r[1] - 80.0).max(0.0) as i32).min(ih - 1);
        let x1 = ((r[2] + 80.0) as i32).clamp(x0 + 1, iw);
        let y1 = ((r[3] + 80.0) as i32).clamp(y0 + 1, ih);
        let crop = Mat::roi(redacted, opencv::core::Rect::new(x0, y0, x1 - x0, y1 - y0))?.try_clone()?;
        let mut z = util::resize_by(&crop, 2.0, imgproc::INTER_CUBIC)?;
        if z.channels() == 1 {
            let mut c = Mat::default();
            imgproc::cvt_color_def(&z, &mut c, imgproc::COLOR_GRAY2BGR)?;
            z = c;
        }
        let p = |x: f64, o: i32| ((x - o as f64) * 2.0) as i32;
        imgproc::rectangle_points(&mut z, opencv::core::Point::new(p(r[0], x0), p(r[1], y0)), opencv::core::Point::new(p(r[2], x0), p(r[3], y0)), opencv::core::Scalar::new(0.0, 0.0, 255.0, 0.0), 3, imgproc::LINE_8, 0)?;
        if self.decision {
            let a = &self.decide(&jpeg(&z)?, DECIDE_STATE, decide_questions())?["what"];
            let p = |k: &str| a["probabilities"][k].as_f64().unwrap_or(0.0);
            // spots asked about are under 8% black, so a "masked" one can
            // only be crossed by the writer's own stroke: it still shows
            return Ok((p("unmasked_signature") + p("masked_signature"), p("complete_number") + p("masked_number"), a["choice"].as_str().unwrap_or("").to_string()));
        }
        let reply = self.chat(CHECK_PROMPT, &jpeg(&z)?, check_schema())?;
        let yes = |k: &str, v: &Value| if v.get(k).and_then(|b| b.as_bool()).unwrap_or(true) { 1.0 } else { 0.0 };
        Ok(match loose_json(&reply) {
            Some(v) => (yes("unmasked_signature", &v), yes("complete_number", &v), v.get("what").and_then(|w| w.as_str()).unwrap_or("").to_string()),
            None => (1.0, 1.0, "unreadable reply".to_string()),
        })
    }

    /// Which way up a *redacted* page stands, as a decision model (Clef)
    /// judges it: `img` is asked about as it is shown; if it doesn't look
    /// upright, also turned a quarter, a half and three quarters clockwise,
    /// and the answers taken together. The turn (degrees clockwise) that
    /// shows it upright, confirmed by that turn of it having been judged
    /// upright; 0 if none was. With the record for the manifest.
    pub fn orient(&self, img: &Mat) -> Result<(i64, Value)> {
        let f = (ORIENT_MAX_SIDE / img.cols().max(img.rows()) as f64).min(1.0);
        let small = if f < 1.0 { util::resize_by(img, f, imgproc::INTER_AREA)? } else { img.clone() };
        let mut asked: Vec<(usize, [f64; 4], String)> = Vec::new();
        for q in 0..4 {
            let sent = jpeg(&match q {
                0 => small.clone(),
                _ => {
                    let mut m = Mat::default();
                    let code = [opencv::core::ROTATE_90_CLOCKWISE, opencv::core::ROTATE_180, opencv::core::ROTATE_90_COUNTERCLOCKWISE][q - 1];
                    opencv::core::rotate(&small, &mut m, code)?;
                    m
                }
            })?;
            let a = &self.decide(&sent, ORIENT_STATE, orient_questions())?["turn"];
            let p = TURNS.map(|k| a["probabilities"][k].as_f64().unwrap_or(0.0));
            asked.push((q, p, hex::encode(Sha256::digest(&sent))));
            if q == 0 && p[0] >= UPRIGHT_AT {
                break;
            }
        }
        // turned t quarters from upright, and q more when asked, the page
        // stands at t + q
        let score = |t: usize| asked.iter().map(|(q, p, _)| p[(t + q) % 4].max(1e-6).ln()).sum::<f64>();
        let t = (0..4).max_by(|&a, &b| score(a).total_cmp(&score(b))).unwrap_or(0);
        let fix = (4 - t) % 4;
        let confirmed = asked.iter().any(|(q, p, _)| *q == fix && p[0] >= UPRIGHT_AT);
        let turn = if confirmed { 90 * fix as i64 } else { 0 };
        let question = format!("{}\n{}", ORIENT_STATE, orient_questions());
        let mut record = json!({"model": self.s.model, "host": hostname(&self.s.host),
                                "question_sha256": hex::encode(Sha256::digest(question.as_bytes())), "upright_at": UPRIGHT_AT,
                                "asked": asked.iter().map(|(q, p, sha)| json!({"turned": 90 * q, "image_sent_sha256": sha,
                                    "p": TURNS.iter().zip(p).map(|(k, v)| (k.to_string(), json!(crate::pyjson::round_n(*v, 4)))).collect::<serde_json::Map<_, _>>()})).collect::<Vec<_>>(),
                                "judged": TURNS[t], "turn": turn});
        if !confirmed {
            record["undecided"] = json!(true);
        }
        Ok((turn, record))
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
