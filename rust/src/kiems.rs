//! KIEMS results-transmission metadata: file names and form QR codes.
//!
//! Files are named 1_34_A_007029014000102_F230450M00203609_20220810_132155.PDF
//! (election, form, suffix, polling station, kit serial, capture date and
//! time). Form 34A's QR encodes the station code and the form number; Form
//! 34B's encodes election, county, constituency, form and page. None of this
//! is personal data; a QR that disagrees with the file name sends the form
//! to review instead of publishing it under the wrong station.

use once_cell::sync::Lazy;
use opencv::core::{Mat, Vector};
use opencv::objdetect::QRCodeDetector;
use opencv::prelude::*;
use regex::Regex;
use serde_json::{json, Value};

static NAME_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(?i)^(?P<election>\d)_(?P<form>\d{2})_(?P<suffix>[A-Z])_(?P<station>\d{15})_(?P<kit>[A-Z0-9]+)_(?P<date>\d{8})_(?P<time>\d{6})$").unwrap()
});
static QR_34A_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"^(?P<station>\d{15})(?P<form>\d{2})$").unwrap());
static QR_PAGE_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(?i)^(?P<election>[A-Z]{2})(?P<year>\d{4})(?P<county>\d{3})(?P<constituency>\d{3})FORM(?P<form>\d{2}[A-Z])(?P<page>\d{2})$")
        .unwrap()
});

pub fn split_station(code: &str) -> Value {
    json!({"county": &code[0..3], "constituency": &code[3..6], "ward": &code[6..10],
           "centre": &code[10..13], "stream": &code[13..15]})
}

#[derive(Clone, Debug)]
pub struct KiemsName {
    pub election: String,
    pub form: String,
    pub station: String,
    pub kit: String,
    pub captured_at: String,
}

impl KiemsName {
    pub fn to_value(&self) -> Value {
        let mut v = json!({"election": self.election, "form": self.form, "station": self.station,
                           "kit": self.kit, "captured_at": self.captured_at});
        for (k, x) in split_station(&self.station).as_object().unwrap() {
            v[k] = x.clone();
        }
        v["election_type"] = Value::String(if self.election == "1" { "presidential".into() } else { self.election.clone() });
        v
    }
}

pub fn parse_name(stem: &str) -> Option<KiemsName> {
    let m = NAME_RE.captures(stem)?;
    let (d, t) = (&m["date"], &m["time"]);
    Some(KiemsName {
        election: m["election"].to_string(),
        form: format!("{}{}", &m["form"], m["suffix"].to_uppercase()),
        station: m["station"].to_string(),
        kit: m["kit"].to_uppercase(),
        captured_at: format!("{}-{}-{}T{}:{}:{}", &d[..4], &d[4..6], &d[6..], &t[..2], &t[2..4], &t[4..]),
    })
}

/// Decode the form's QR code: the top of the page first (faster, fewer false
/// finds), then the whole page; each also enlarged 1.5x.
pub fn decode_qr(gray: &Mat) -> Option<String> {
    let qr = QRCodeDetector::default().ok()?;
    let h = gray.rows();
    let top = Mat::roi(gray, opencv::core::Rect::new(0, 0, gray.cols(), (h / 4).max(1))).ok()?.try_clone().ok()?;
    for region in [top, gray.try_clone().ok()?] {
        let big = crate::util::resize_by(&region, 1.5, opencv::imgproc::INTER_CUBIC).ok()?;
        for img in [&region, &big] {
            let mut texts = Vector::<String>::new();
            let mut pts = Mat::default();
            let ok = qr.detect_multi(img, &mut pts).unwrap_or(false)
                && qr.decode_multi_def(img, &pts, &mut texts).unwrap_or(false);
            if ok {
                for t in texts.iter() {
                    if !t.is_empty() {
                        return Some(t.trim().to_string());
                    }
                }
            }
        }
    }
    None
}

/// Structured view of a form QR payload (empty if the format is unknown).
pub fn parse_qr(payload: &str) -> Value {
    if let Some(m) = QR_34A_RE.captures(payload) {
        let mut v = json!({"station": &m["station"], "form": format!("{}A", &m["form"])});
        for (k, x) in split_station(&m["station"]).as_object().unwrap() {
            v[k] = x.clone();
        }
        return v;
    }
    if let Some(m) = QR_PAGE_RE.captures(payload) {
        return json!({"election": m["election"].to_uppercase(), "year": &m["year"], "county": &m["county"],
                      "constituency": &m["constituency"], "form": m["form"].to_uppercase(),
                      "page": m["page"].parse::<i64>().unwrap_or(0)});
    }
    json!({})
}

/// Why the file name and QR disagree, or "" if they're consistent.
pub fn check(name: Option<&KiemsName>, qr: &Value) -> String {
    let Some(name) = name else { return String::new() };
    let Some(q) = qr.as_object().filter(|m| !m.is_empty()) else { return String::new() };
    if let Some(f) = q.get("form").and_then(|v| v.as_str()) {
        if f != name.form {
            return format!("QR form {} != file name form {}", f, name.form);
        }
    }
    if let Some(s) = q.get("station").and_then(|v| v.as_str()) {
        if s != name.station {
            return format!("QR station {} != file name station {}", s, name.station);
        }
    }
    if let Some(c) = q.get("county").and_then(|v| v.as_str()) {
        if c != &name.station[..3] {
            return format!("QR county {} != file name county {}", c, &name.station[..3]);
        }
    }
    String::new()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_and_qr() {
        let n = parse_name("1_34_A_007029014000102_F230450M00203609_20220810_132155").unwrap();
        assert_eq!(n.form, "34A");
        assert_eq!(n.captured_at, "2022-08-10T13:21:55");
        assert_eq!(parse_qr("00702901400010234")["form"], "34A");
        assert_eq!(parse_qr("GE2022040228FORM34B03")["page"], 3);
        assert_eq!(check(Some(&n), &parse_qr("00702901400010234")), "");
        assert!(check(Some(&n), &parse_qr("00702901400010334")).contains("station"));
    }
}
