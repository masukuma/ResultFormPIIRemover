//! A scrutiny bundle for a court, a petitioner or agents: the published
//! forms asked for, their manifests, each one's `verify` log, the ledger,
//! the batch roots, the public key and the controller and transfer records,
//! with a cover sheet and a list of the SHA-256 of every file in it. The
//! originals go in only when their release is authorised
//! (`legal/C-recommendations.md` §2.13).

use anyhow::{bail, Result};
use std::path::{Path, PathBuf};

use crate::verify::{verify, VerifyArgs};

pub struct BundleArgs<'a> {
    /// the published forms (PDFs, their manifests beside them)
    pub forms: &'a [PathBuf],
    pub out: &'a Path,
    /// the published folder: ledger, batch roots, transfers record
    pub public: &'a Path,
    pub pubkey: Option<&'a Path>,
    /// where the originals are (found by their SHA-256), to verify against
    pub originals: Option<&'a Path>,
    /// copy the originals into the bundle (their release authorised)
    pub include_originals: bool,
    /// templates, to re-run each redaction from its seed
    pub rerun: Option<&'a Path>,
}

fn copy_into(src: &Path, dir: &Path) -> Result<PathBuf> {
    std::fs::create_dir_all(dir)?;
    let dest = dir.join(src.file_name().unwrap_or_default());
    std::fs::copy(src, &dest)?;
    Ok(dest)
}

/// Write the bundle; returns the number of forms that verified.
pub fn bundle(a: &BundleArgs) -> Result<usize> {
    if a.out.exists() && std::fs::read_dir(a.out)?.next().is_some() {
        bail!("{} is not empty", a.out.display());
    }
    std::fs::create_dir_all(a.out)?;
    // the originals, by their SHA-256
    let mut by_hash = std::collections::HashMap::new();
    if let Some(o) = a.originals {
        for f in crate::annex::files_under(o, "") {
            if let Ok(b) = std::fs::read(&f) {
                by_hash.insert(crate::proof::sha256(&b), f);
            }
        }
    }
    let ledger = a.public.join("ledger.jsonl");
    let mut rows = Vec::new();
    let mut verified = 0;
    for pdf in a.forms {
        let mpath = pdf.with_extension("manifest.json");
        let m = crate::manifest::read_json(&mpath)?;
        copy_into(pdf, &a.out.join("forms"))?;
        copy_into(&mpath, &a.out.join("forms"))?;
        let src_sha = m["source"]["sha256"].as_str().unwrap_or("").to_string();
        let original = by_hash.get(&src_sha).cloned();
        if let (Some(o), true) = (&original, a.include_originals) {
            copy_into(o, &a.out.join("originals"))?;
        }
        let r = verify(&VerifyArgs {
            redacted: pdf,
            manifest: Some(&mpath),
            original: original.as_deref(),
            pubkey: a.pubkey,
            ledger: ledger.exists().then_some(ledger.as_path()),
            overlay_dir: None,
            rerun_templates: if original.is_some() { a.rerun } else { None },
            trust: &[],
        })?;
        let name = pdf.file_name().unwrap_or_default().to_string_lossy().into_owned();
        std::fs::create_dir_all(a.out.join("verify"))?;
        std::fs::write(a.out.join("verify").join(format!("{}.txt", name)), r.text())?;
        if r.ok() {
            verified += 1;
        }
        rows.push(format!(
            "| {} | {} | {} | {} |",
            name,
            &src_sha[..16.min(src_sha.len())],
            if original.is_some() { "yes" } else { "not found" },
            if r.ok() { "VERIFIED" } else { "FAILED" }
        ));
    }
    for f in ["ledger.jsonl", "transfers.jsonl"] {
        if a.public.join(f).exists() {
            copy_into(&a.public.join(f), a.out)?;
        }
    }
    for b in crate::annex::files_under(&a.public.join("batches"), ".json") {
        copy_into(&b, &a.out.join("batches"))?;
    }
    if let Some(k) = a.pubkey {
        copy_into(k, a.out)?;
    }
    let cover = format!(
        "# Scrutiny bundle\n\nPrepared by rfpii {} on {}.\n\nEach published form below is in `forms/` with its signed manifest; `verify/` holds the output of `rfpii verify` for it, run against the original where one was found by its SHA-256. The ledger, the batch roots and the public key let anyone repeat every check. {}\n\n| Form | Original SHA-256 (first 16) | Original found | Verify |\n|---|---|---|---|\n{}\n\n`SHA256SUMS` lists every file in this bundle.\n",
        env!("CARGO_PKG_VERSION"),
        crate::process::utc_now(),
        if a.include_originals { "The originals are in `originals/`, their release authorised." } else { "The originals are not included: ask the IEBC for them, or the court for an order." },
        rows.join("\n")
    );
    std::fs::write(a.out.join("README.md"), cover)?;
    let mut sums = String::new();
    for f in crate::annex::files_under(a.out, "") {
        sums += &format!("{}  {}\n", crate::proof::sha256(&std::fs::read(&f)?), f.strip_prefix(a.out).unwrap_or(&f).to_string_lossy());
    }
    std::fs::write(a.out.join("SHA256SUMS"), sums)?;
    Ok(verified)
}
