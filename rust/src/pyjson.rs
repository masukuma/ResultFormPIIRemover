//! JSON written the way Python's `json.dumps` writes it.
//!
//! Manifests are hashed and signed over their canonical JSON
//! (`sort_keys=True, separators=(",", ":"), ensure_ascii=False`). A manifest
//! made by either implementation must verify with the other, so the bytes
//! must match, floats included: Python writes the shortest repr ("1e-05",
//! "100.0", "1e+16"), which is not how Rust formats them.

use serde_json::Value;

/// Python's `repr(float)`.
pub fn float_repr(x: f64) -> String {
    if x.is_nan() {
        return "NaN".into();
    }
    if x.is_infinite() {
        return if x > 0.0 { "Infinity".into() } else { "-Infinity".into() };
    }
    if x == 0.0 {
        return if x.is_sign_negative() { "-0.0".into() } else { "0.0".into() };
    }
    // shortest round-trip digits and decimal exponent, from Rust's {:e}
    let e = format!("{:e}", x); // e.g. "-1.2345e-5"
    let (mant, exp) = e.split_once('e').unwrap();
    let exp: i32 = exp.parse().unwrap();
    let neg = mant.starts_with('-');
    let digits: String = mant.chars().filter(|c| c.is_ascii_digit()).collect();
    let decpt = exp + 1; // value = 0.DIGITS x 10^decpt
    let sign = if neg { "-" } else { "" };
    if -4 < decpt && decpt <= 16 {
        let n = digits.len() as i32;
        let s = if decpt <= 0 {
            format!("0.{}{}", "0".repeat((-decpt) as usize), digits)
        } else if decpt >= n {
            format!("{}{}.0", digits, "0".repeat((decpt - n) as usize))
        } else {
            format!("{}.{}", &digits[..decpt as usize], &digits[decpt as usize..])
        };
        format!("{}{}", sign, s)
    } else {
        let m = if digits.len() > 1 { format!("{}.{}", &digits[..1], &digits[1..]) } else { digits.clone() };
        let ex = decpt - 1;
        format!("{}{}e{}{:02}", sign, m, if ex < 0 { '-' } else { '+' }, ex.abs())
    }
}

fn number(n: &serde_json::Number) -> String {
    if let Some(i) = n.as_i64() {
        i.to_string()
    } else if let Some(u) = n.as_u64() {
        u.to_string()
    } else {
        float_repr(n.as_f64().unwrap())
    }
}

fn string(s: &str, out: &mut String) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{08}' => out.push_str("\\b"),
            '\u{0c}' => out.push_str("\\f"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
}

fn sorted(m: &serde_json::Map<String, Value>) -> Vec<(&String, &Value)> {
    let mut v: Vec<_> = m.iter().collect();
    v.sort_by(|a, b| a.0.cmp(b.0));
    v
}

/// `json.dumps(v, sort_keys=True, separators=(",", ":"), ensure_ascii=False)`.
pub fn canonical(v: &Value) -> String {
    let mut out = String::new();
    compact(v, &mut out, ",", ":", true);
    out
}

/// `json.dumps(v)`: separators ", " and ": ", keys in their own order.
pub fn dumps(v: &Value) -> String {
    let mut out = String::new();
    compact(v, &mut out, ", ", ": ", false);
    out
}

fn compact(v: &Value, out: &mut String, item: &str, key: &str, sort: bool) {
    match v {
        Value::Null => out.push_str("null"),
        Value::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        Value::Number(n) => out.push_str(&number(n)),
        Value::String(s) => string(s, out),
        Value::Array(a) => {
            out.push('[');
            for (i, x) in a.iter().enumerate() {
                if i > 0 {
                    out.push_str(item);
                }
                compact(x, out, item, key, sort);
            }
            out.push(']');
        }
        Value::Object(m) => {
            out.push('{');
            let entries: Vec<(&String, &Value)> = if sort { sorted(m) } else { m.iter().collect() };
            for (i, (k, x)) in entries.into_iter().enumerate() {
                if i > 0 {
                    out.push_str(item);
                }
                string(k, out);
                out.push_str(key);
                compact(x, out, item, key, sort);
            }
            out.push('}');
        }
    }
}

/// `json.dumps(v, indent=indent, sort_keys=sort)`.
pub fn pretty(v: &Value, indent: usize, sort: bool) -> String {
    let mut out = String::new();
    pretty_into(v, &mut out, indent, 0, sort);
    out
}

fn pretty_into(v: &Value, out: &mut String, indent: usize, level: usize, sort: bool) {
    let pad = |n: usize| " ".repeat(n * indent);
    match v {
        Value::Array(a) if !a.is_empty() => {
            out.push('[');
            for (i, x) in a.iter().enumerate() {
                out.push_str(if i > 0 { ",\n" } else { "\n" });
                out.push_str(&pad(level + 1));
                pretty_into(x, out, indent, level + 1, sort);
            }
            out.push('\n');
            out.push_str(&pad(level));
            out.push(']');
        }
        Value::Object(m) if !m.is_empty() => {
            out.push('{');
            let entries: Vec<(&String, &Value)> = if sort { sorted(m) } else { m.iter().collect() };
            for (i, (k, x)) in entries.into_iter().enumerate() {
                out.push_str(if i > 0 { ",\n" } else { "\n" });
                out.push_str(&pad(level + 1));
                string(k, out);
                out.push_str(": ");
                pretty_into(x, out, indent, level + 1, sort);
            }
            out.push('\n');
            out.push_str(&pad(level));
            out.push('}');
        }
        _ => compact(v, out, ", ", ": ", sort),
    }
}

/// A float as a JSON number (NaN / infinity as null).
pub fn num(x: f64) -> Value {
    serde_json::Number::from_f64(x).map(Value::Number).unwrap_or(Value::Null)
}

/// Python's `round(x, n)` (half to even on the decimal representation).
pub fn round_n(x: f64, n: i32) -> f64 {
    // Python rounds correctly in decimal; formatting with the exact digits
    // and parsing back gives the same double in all but pathological cases.
    let s = format!("{:.*}", n.max(0) as usize, x);
    s.parse().unwrap_or(x)
}

/// `np.round(x, n)`: scaled, rounded half to even, scaled back.
pub fn np_round(x: f64, n: i32) -> f64 {
    let f = 10f64.powi(n);
    (x * f).round_ties_even() / f
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn floats_as_python() {
        for (x, s) in [
            (1e-05, "1e-05"),
            (0.0001, "0.0001"),
            (100.0, "100.0"),
            (1e16, "1e+16"),
            (1e15, "1000000000000000.0"),
            (0.1, "0.1"),
            (-2.5e-7, "-2.5e-07"),
            (123456.789, "123456.789"),
            (1.0, "1.0"),
            (3.14e20, "3.14e+20"),
        ] {
            assert_eq!(float_repr(x), s, "{}", x);
        }
    }

    #[test]
    fn canonical_as_python() {
        // python: json.dumps({"b": [1, 2.5, None], "a": "é\n", "c": {}}, sort_keys=True,
        //                    separators=(",", ":"), ensure_ascii=False)
        let v = json!({"b": [1, 2.5, null], "a": "é\n", "c": {}});
        assert_eq!(canonical(&v), "{\"a\":\"é\\n\",\"b\":[1,2.5,null],\"c\":{}}");
        assert_eq!(pretty(&json!({"x": [1, {}], "a": 1}), 1, true), "{\n \"a\": 1,\n \"x\": [\n  1,\n  {}\n ]\n}");
    }
}
