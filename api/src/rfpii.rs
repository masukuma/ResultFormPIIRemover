//! The `rfpii` binary, which the API runs as a subprocess: `inventory` to
//! count a job's forms, `verify` for anyone checking a published form, and
//! the job file `rfpii job` is given. The API never links `rfpii` itself.

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::time::Duration;

/// How long `inventory` may take.
const INVENTORY_TIMEOUT: Duration = Duration::from_secs(600);
/// How long `verify` may take.
pub const VERIFY_TIMEOUT: Duration = Duration::from_secs(300);

/// Whether a file of several pages is redacted page by page (the policy's
/// `per_page`, HARNESS.md §3).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum PerPage {
    /// scans of many forms page by page, other files whole
    #[default]
    Auto,
    Yes,
    No,
}

impl PerPage {
    pub fn name(self) -> &'static str {
        match self {
            PerPage::Auto => "auto",
            PerPage::Yes => "yes",
            PerPage::No => "no",
        }
    }
}

/// The environment a child process gets: only what `rfpii` needs (its
/// libraries, the proxy, its passphrases, the model host's key), never the
/// API's own secrets (Stripe, admin and metrics tokens).
pub fn child_env() -> Vec<(OsString, OsString)> {
    const KEEP: &[&str] = &[
        "PATH",
        "HOME",
        "TMPDIR",
        "LANG",
        "LC_ALL",
        "LD_LIBRARY_PATH",
        "XDG_CONFIG_HOME",
        "SSL_CERT_FILE",
        "SSL_CERT_DIR",
        "HTTPS_PROXY",
        "https_proxy",
        "HTTP_PROXY",
        "http_proxy",
        "NO_PROXY",
        "no_proxy",
        "ALL_PROXY",
        "all_proxy",
    ];
    std::env::vars_os()
        .filter(|(k, _)| {
            let k = k.to_string_lossy();
            KEEP.contains(&k.as_ref()) || k.starts_with("RFPII_") || k.starts_with("OLLAMA_")
        })
        .collect()
}

/// Run `bin args`, with a time limit: exit code, stdout, stderr.
async fn run(bin: &Path, args: &[OsString], limit: Duration) -> Result<(i32, String, String)> {
    let mut cmd = tokio::process::Command::new(bin);
    cmd.args(args).env_clear().envs(child_env()).stdin(std::process::Stdio::null()).kill_on_drop(true);
    let out = tokio::time::timeout(limit, cmd.output())
        .await
        .with_context(|| format!("{} took longer than {}s", bin.display(), limit.as_secs()))?
        .with_context(|| format!("running {}", bin.display()))?;
    Ok((out.status.code().unwrap_or(-1), String::from_utf8_lossy(&out.stdout).into_owned(), String::from_utf8_lossy(&out.stderr).into_owned()))
}

/// `rfpii --version`, to show the binary runs.
pub async fn version(bin: &Path) -> Result<String> {
    let (code, out, err) = run(bin, &["--version".into()], Duration::from_secs(20)).await?;
    if code != 0 {
        bail!("{} --version exited with {code}: {}", bin.display(), err.trim());
    }
    Ok(out.trim().to_string())
}

/// `rfpii inventory DIR --json`.
pub async fn inventory(bin: &Path, dir: &Path) -> Result<Value> {
    let (code, out, err) = run(bin, &["inventory".into(), dir.into(), "--json".into()], INVENTORY_TIMEOUT).await?;
    if code != 0 {
        bail!("rfpii inventory exited with {code}: {}", last_line(&err));
    }
    serde_json::from_str(&out).context("rfpii inventory printed no JSON")
}

fn last_line(s: &str) -> &str {
    s.trim().lines().last().unwrap_or("")
}

/// What an inventory says about a job, and the forms it would redact.
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct Counted {
    pub files: i64,
    pub pages: i64,
    pub duplicates: i64,
    pub broken: i64,
    pub forms: i64,
}

/// The forms ("documents") a job redacts, as `rfpii job` plans them
/// (`rust/src/job/plan.rs`): each readable file that isn't a copy of
/// another is one form, or one per page when it is redacted page by page:
/// a file of several pages with `per_page: yes`, or a scan (a PDF of
/// several pages not named as a KIEMS transmission) with `auto`.
pub fn count_forms(inv: &Value, per_page: PerPage) -> Counted {
    let mut c = Counted {
        files: inv["totals"]["files"].as_i64().unwrap_or(0),
        pages: inv["totals"]["pages"].as_i64().unwrap_or(0),
        duplicates: inv["totals"]["duplicates"].as_i64().unwrap_or(0),
        broken: inv["totals"]["broken"].as_i64().unwrap_or(0),
        forms: 0,
    };
    for f in inv["files"].as_array().into_iter().flatten() {
        if f.get("broken").is_some_and(|b| !b.is_null()) || f.get("duplicate_of").is_some_and(|d| !d.is_null()) {
            continue;
        }
        let pages = f["pages"].as_i64().unwrap_or(1).max(1);
        let by_page = pages > 1
            && match per_page {
                PerPage::Auto => f["kind"] == "scan",
                PerPage::Yes => true,
                PerPage::No => false,
            };
        c.forms += if by_page { pages } else { 1 };
    }
    c
}

/// What `rfpii verify` found.
#[derive(Clone, Debug, Serialize, PartialEq)]
pub struct Verified {
    pub verified: bool,
    pub checks: Vec<Check>,
    pub notes: Vec<String>,
    pub result: Option<String>,
}

#[derive(Clone, Debug, Serialize, PartialEq)]
pub struct Check {
    pub ok: bool,
    pub text: String,
}

/// Read `rfpii verify`'s output: `PASS  …` and `FAIL  …` lines, `note  …`
/// lines and the `RESULT: …` line. Verified only with exit code 0 and
/// `RESULT: VERIFIED`, and no check failed.
pub fn parse_verify(code: i32, stdout: &str) -> Verified {
    let mut v = Verified { verified: false, checks: vec![], notes: vec![], result: None };
    for line in stdout.lines() {
        if let Some(t) = line.strip_prefix("PASS") {
            v.checks.push(Check { ok: true, text: t.trim().into() });
        } else if let Some(t) = line.strip_prefix("FAIL") {
            v.checks.push(Check { ok: false, text: t.trim().into() });
        } else if let Some(t) = line.strip_prefix("note") {
            v.notes.push(t.trim().into());
        } else if line.starts_with("RESULT:") {
            v.result = Some(line.trim().into());
        }
    }
    v.verified = code == 0 && v.result.as_deref().is_some_and(|r| r.starts_with("RESULT: VERIFIED")) && v.checks.iter().all(|c| c.ok);
    v
}

/// The files given to `rfpii verify`.
pub struct VerifyFiles {
    pub pdf: PathBuf,
    pub manifest: PathBuf,
    pub original: Option<PathBuf>,
    pub pubkey: Option<PathBuf>,
    pub ledger: Option<PathBuf>,
}

/// `rfpii verify PDF --manifest M [--original O] [--pubkey K] [--ledger L]`.
pub async fn verify(bin: &Path, f: &VerifyFiles) -> Result<Verified> {
    let mut args: Vec<OsString> = vec!["verify".into(), f.pdf.clone().into(), "--manifest".into(), f.manifest.clone().into()];
    for (flag, p) in [("--original", &f.original), ("--pubkey", &f.pubkey), ("--ledger", &f.ledger)] {
        if let Some(p) = p {
            args.push(flag.into());
            args.push(p.into());
        }
    }
    let (code, out, _) = run(bin, &args, VERIFY_TIMEOUT).await?;
    Ok(parse_verify(code, &out))
}

/// The job file for `rfpii job` (HARNESS.md §1, `rust/src/job/spec.rs`).
/// Paths are absolute; secrets are never in it (passphrases and the model
/// host's key reach `rfpii` through its environment).
#[derive(Serialize, Debug)]
pub struct JobFile {
    pub inputs: Vec<PathBuf>,
    pub instructions: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub policy: Option<Value>,
    pub outputs: PathBuf,
    pub sign_key: PathBuf,
    pub controller: PathBuf,
    pub dpia: Value,
    pub templates: PathBuf,
    pub held_pages: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub models: Option<Value>,
    pub budget: Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub certificate: Option<PathBuf>,
    pub kenya_hosts: Vec<String>,
}

/// The models part of a job file, or None when no model is configured.
pub fn models_of(orchestrator: &[String], vision: &[String], host: Option<&str>) -> Option<Value> {
    if orchestrator.is_empty() && vision.is_empty() {
        return None;
    }
    let mut m = json!({"orchestrator": orchestrator, "vision": vision});
    if let Some(h) = host.filter(|h| !h.is_empty()) {
        m["host"] = json!(h);
    }
    Some(m)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn inv() -> Value {
        json!({"totals": {"files": 5, "pages": 44, "duplicates": 1, "broken": 1},
               "files": [
                   {"path": "a.jpg", "kind": "photo", "pages": 1},
                   {"path": "scan.pdf", "kind": "scan", "pages": 40},
                   {"path": "k.pdf", "kind": "kiems", "pages": 2},
                   {"path": "copy.jpg", "kind": "photo", "pages": 1, "duplicate_of": "a.jpg"},
                   {"path": "cut.pdf", "broken": "cannot find startxref"}]})
    }

    #[test]
    fn forms_are_counted_as_the_job_plans_them() {
        assert_eq!(count_forms(&inv(), PerPage::Auto), Counted { files: 5, pages: 44, duplicates: 1, broken: 1, forms: 42 });
        assert_eq!(count_forms(&inv(), PerPage::Yes).forms, 43);
        assert_eq!(count_forms(&inv(), PerPage::No).forms, 3);
        assert_eq!(count_forms(&json!({}), PerPage::Auto).forms, 0);
    }

    #[test]
    fn verify_output_is_read_line_by_line() {
        let out = "PASS  manifest signature valid\nPASS  ledger hash chain intact\nnote  the PDF is not digitally signed\nRESULT: VERIFIED - only the listed blocks were blacked out; nothing else changed\n";
        let v = parse_verify(0, out);
        assert!(v.verified && v.checks.len() == 2 && v.checks[0] == Check { ok: true, text: "manifest signature valid".into() });
        assert_eq!(v.notes, ["the PDF is not digitally signed"]);
        let bad = parse_verify(1, "PASS  a\nFAIL  redacted PDF matches the manifest's SHA-256\nRESULT: NOT VERIFIED\n");
        assert!(!bad.verified && !bad.checks[1].ok);
        assert!(!parse_verify(0, "RESULT: NOT VERIFIED").verified);
    }

    #[test]
    fn a_job_file_holds_no_secret_and_omits_models_when_there_are_none() {
        let jf = JobFile {
            inputs: vec!["/j/in".into()],
            instructions: String::new(),
            policy: Some(json!({"ids": "whole"})),
            outputs: "/j/out".into(),
            sign_key: "/k/key.pem".into(),
            controller: "/j/controller.json".into(),
            dpia: json!({"reference": "ODPC/1"}),
            templates: "/t".into(),
            held_pages: "none".into(),
            models: models_of(&[], &[], None),
            budget: json!({"model_requests": 10, "hours": 1.0}),
            certificate: None,
            kenya_hosts: vec![],
        };
        let y = serde_yaml::to_string(&jf).unwrap();
        assert!(y.contains("outputs: /j/out") && y.contains("held_pages: none") && !y.contains("models") && !y.contains("certificate"));
        let m = models_of(&["gpt-oss:120b".into()], &["gemma4:31b".into()], Some("https://ollama.com")).unwrap();
        assert_eq!(m["host"], "https://ollama.com");
    }
}
