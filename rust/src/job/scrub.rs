//! The text scrubber (HARNESS.md §7): what goes to a cloud model as text
//! passes through it first.
//!
//! - Every run of five or more digits becomes `#` (an ID has 7 to 8 digits,
//!   a phone number 10; counts and coordinates have 4 or fewer). The guard
//!   refuses any request that still has one.
//! - Quoted text in what the redactor wrote (its reasons and notes quote
//!   what OCR read off a form) becomes `'…'`. The requester's instructions
//!   keep their quotes: they are not read off a form, and an apostrophe
//!   ("the officers' signatures") is not a quote.
//! - Only allow-listed fields of a record are passed on.

use serde_json::{Map, Value};

/// Runs of five or more digits replaced by `#`.
pub fn digits(s: &str) -> String {
    regex::Regex::new(r"\d{5,}").unwrap().replace_all(s, "#").into_owned()
}

/// Text the redactor wrote: quoted text replaced by `'…'`, then runs of
/// digits by `#`.
pub fn text(s: &str) -> String {
    let single = regex::Regex::new(r#"(^|[\s(\[:,=])'[^'\n]*'"#).unwrap();
    let double = regex::Regex::new(r#""[^"\n]*""#).unwrap();
    let s = single.replace_all(s, "$1'…'");
    digits(&double.replace_all(&s, "'…'"))
}

/// Only the fields of `v` that `allow` lists (dotted paths: "ocr.turn"; a
/// path names all below it; the items of a list are under the list's
/// path), every string in them scrubbed by `text`.
pub fn fields(v: &Value, allow: &[&str]) -> Value {
    fn walk(v: &Value, path: &str, allow: &[&str]) -> Option<Value> {
        let listed = allow.iter().any(|a| path == *a || path.starts_with(&format!("{}.", a)));
        let below = allow.iter().any(|a| a.starts_with(&format!("{}.", path)));
        match v {
            Value::Object(o) if !listed => {
                let m: Map<String, Value> = o.iter().filter_map(|(k, x)| {
                    let p = if path.is_empty() { k.clone() } else { format!("{}.{}", path, k) };
                    walk(x, &p, allow).map(|y| (k.clone(), y))
                }).collect();
                (path.is_empty() || below).then_some(Value::Object(m))
            }
            Value::Array(a) if !listed => (path.is_empty() || below).then(|| Value::Array(a.iter().filter_map(|x| walk(x, path, allow)).collect())),
            _ if listed => Some(scrub_all(v)),
            _ => None,
        }
    }
    fn scrub_all(v: &Value) -> Value {
        match v {
            Value::String(s) => Value::String(text(s)),
            Value::Array(a) => Value::Array(a.iter().map(scrub_all).collect()),
            Value::Object(o) => Value::Object(o.iter().map(|(k, x)| (text(k), scrub_all(x))).collect()),
            x => x.clone(),
        }
    }
    walk(v, "", allow).unwrap_or(Value::Null)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn numbers_and_what_was_read_off_a_form_are_taken_out() {
        assert_eq!(digits("ID 23456789, phone 0712345678, 1234 votes"), "ID #, phone #, 1234 votes");
        assert_eq!(text("label 'JANE DOE 07123' not found; \"Tel 0712 345678\" near"), "label '…' not found; '…' near");
        // an apostrophe is not a quote
        assert_eq!(text("the officers' signatures don't show"), "the officers' signatures don't show");
    }

    #[test]
    fn only_allowed_fields_pass() {
        let d = json!({"contrast": 40.5, "ocr": {"turn": 90, "vocabulary": ["TALLY"], "lines_text": ["JANE 0712345678"]}, "read": {"reason": "label 'X Y' at 1234567"}, "path": "/data/x.pdf"});
        let out = fields(&d, &["contrast", "ocr.turn", "ocr.vocabulary", "read"]);
        assert_eq!(out, json!({"contrast": 40.5, "ocr": {"turn": 90, "vocabulary": ["TALLY"]}, "read": {"reason": "label '…' at #"}}));
        let groups = json!([{"group": "scan", "held_reasons": ["label 'JANE' at 1234567"], "files": ["/d/x.pdf"]}]);
        assert_eq!(fields(&groups, &["group", "held_reasons"]), json!([{"group": "scan", "held_reasons": ["label '…' at #"]}]));
    }
}
