//! Commands for the publisher's duties around a published batch: keeping
//! the signing key safe (encrypted, rotated, revoked) and taking a
//! published form down when a leak is found (`legal/C-recommendations.md`
//! §2.11, §2.14).

use anyhow::{bail, Context, Result};
use clap::Args;
use rfpii::manifest as mf;
use serde_json::json;
use std::path::{Path, PathBuf};

#[derive(Args)]
pub struct KeygenArgs {
    pub directory: PathBuf,
    /// Write the private key unencrypted. By default it is encrypted with a
    /// passphrase ($RFPII_SIGN_PASSWORD, or asked for on the terminal)
    #[arg(long)]
    pub unencrypted: bool,
}

#[derive(Args)]
pub struct RotateKeyArgs {
    /// The key in use, which signs the hand-over to the new one
    #[arg(long)]
    pub key: PathBuf,
    /// Folder for the new key pair
    #[arg(long)]
    pub new: PathBuf,
    /// The published ledger.jsonl
    #[arg(long)]
    pub ledger: PathBuf,
    #[arg(long)]
    pub unencrypted: bool,
}

#[derive(Args)]
pub struct RevokeKeyArgs {
    /// The key revoked (it signs its own revocation)
    #[arg(long)]
    pub key: PathBuf,
    #[arg(long)]
    pub ledger: PathBuf,
    /// Why: lost, stolen, retired…
    #[arg(long)]
    pub reason: String,
    /// When it may first have been misused (UTC, e.g. 2026-10-30T00:00:00):
    /// manifests made since then no longer verify, wherever they are
    #[arg(long)]
    pub since: Option<String>,
}

#[derive(Args)]
pub struct WithdrawArgs {
    /// The published PDF (its manifest beside it)
    pub pdf: PathBuf,
    /// Why it is taken down: what was found showing
    #[arg(long)]
    pub reason: String,
    /// Key that signs the withdrawal (the publisher's)
    #[arg(long)]
    pub sign_key: PathBuf,
    /// The ledger (default: the nearest ledger.jsonl above the PDF)
    #[arg(long)]
    pub ledger: Option<PathBuf>,
}

/// A new passphrase: $RFPII_SIGN_PASSWORD, or asked for twice.
fn new_passphrase() -> Result<String> {
    if let Ok(p) = std::env::var("RFPII_SIGN_PASSWORD") {
        return Ok(p);
    }
    if !std::io::IsTerminal::is_terminal(&std::io::stdin()) {
        bail!("set RFPII_SIGN_PASSWORD to encrypt the private key, or pass --unencrypted");
    }
    let a = rpassword::prompt_password("passphrase for the new private key: ")?;
    let b = rpassword::prompt_password("again: ")?;
    if a != b || a.is_empty() {
        bail!("the passphrases differ, or are empty");
    }
    Ok(a)
}

pub fn keygen(a: KeygenArgs) -> Result<i32> {
    let pass = if a.unencrypted { None } else { Some(new_passphrase()?) };
    let (p, q) = mf::keygen(&a.directory, pass.as_deref())?;
    println!(
        "private key {} ({}; keep offline)\npublic key  {} (publish)  id {}",
        p.display(),
        if pass.is_some() { "encrypted" } else { "UNENCRYPTED" },
        q.display(),
        mf::key_id(&mf::load_public(&q)?)
    );
    Ok(0)
}

pub fn rotate_key(a: RotateKeyArgs) -> Result<i32> {
    let old = mf::load_private(&a.key)?;
    let pass = if a.unencrypted { None } else { Some(new_passphrase()?) };
    let (p, q) = mf::keygen(&a.new, pass.as_deref())?;
    let new_pub = std::fs::read_to_string(&q)?;
    let newk = mf::load_public(&q)?;
    let e = mf::ledger_event(
        &a.ledger,
        "key_rotation",
        json!({"old_key_id": mf::key_id(&old.verifying_key()), "new_key_id": mf::key_id(&newk), "new_public_key": new_pub}),
        &old,
    )?;
    println!("key {} replaced by {} (ledger entry {})\nnew private key {}\nnew public key  {}", mf::key_id(&old.verifying_key()), mf::key_id(&newk), e["seq"], p.display(), q.display());
    println!("Manifests signed with the new key verify with the old published public key and this ledger.");
    Ok(0)
}

pub fn revoke_key(a: RevokeKeyArgs) -> Result<i32> {
    let key = mf::load_private(&a.key)?;
    let mut f = json!({"key_id": mf::key_id(&key.verifying_key()), "reason": a.reason});
    if let Some(s) = &a.since {
        f["since"] = json!(s);
    }
    let e = mf::ledger_event(&a.ledger, "key_revocation", f, &key)?;
    println!("key {} revoked (ledger entry {}). Manifests it signs from now on{} no longer verify.", mf::key_id(&key.verifying_key()), e["seq"], if a.since.is_some() { ", and those made since the date given," } else { "" });
    Ok(0)
}

/// The nearest ledger.jsonl in `pdf`'s folder or above it.
fn find_ledger(pdf: &Path) -> Option<PathBuf> {
    pdf.ancestors().skip(1).map(|d| d.join("ledger.jsonl")).find(|p| p.exists())
}

pub fn withdraw(a: WithdrawArgs) -> Result<i32> {
    let mpath = a.pdf.with_extension("manifest.json");
    let m = mf::read_json(&mpath).with_context(|| format!("no manifest beside {}", a.pdf.display()))?;
    let ledger = match a.ledger.clone().or_else(|| find_ledger(&a.pdf)) {
        Some(l) => l,
        None => bail!("no ledger.jsonl above {}; pass --ledger", a.pdf.display()),
    };
    let (_, _, entries) = mf::ledger_check(&ledger)?;
    let digest = mf::digest(&m);
    if !entries.iter().any(|e| e["manifest_sha256"].as_str() == Some(digest.as_str()) && e.get("type").is_none()) {
        bail!("{} is not recorded in {}", mpath.display(), ledger.display());
    }
    let key = mf::load_private(&a.sign_key)?;
    let e = mf::ledger_event(
        &ledger,
        "withdrawal",
        json!({"manifest_sha256": digest, "output_sha256": m["output"]["pdf_sha256"], "name": m["output"]["file"], "reason": a.reason}),
        &key,
    )?;
    // the PDF comes down, out of the published folder (it shows what it
    // shouldn't) into one beside it; its manifest stays as the record of
    // what was published
    let root = ledger.parent().unwrap_or(Path::new("."));
    let rel = a.pdf.strip_prefix(root).unwrap_or(&a.pdf).to_path_buf();
    let name = root.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_else(|| "out".into());
    let dest = root.with_file_name(format!("{}-withdrawn", name)).join(&rel);
    std::fs::create_dir_all(dest.parent().unwrap())?;
    std::fs::rename(&a.pdf, &dest)?;
    std::fs::copy(&mpath, dest.with_extension("manifest.json"))?;
    println!("withdrawn (ledger entry {}): {}\nmoved to {}", e["seq"], rel.display(), dest.display());
    println!("Republish it corrected with: rfpii redact <original> ... --supersedes {}", mpath.display());
    Ok(0)
}
