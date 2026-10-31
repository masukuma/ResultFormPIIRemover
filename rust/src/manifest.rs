//! Signed manifests, an append-only ledger and batch roots.
//!
//! Every published form gets a manifest saying exactly what was done to it,
//! signed (Ed25519) by the publisher, appended to a hash-chained ledger, and
//! covered by the Merkle root of its batch. Formats, hashes and signatures
//! are those of the first (Python) version, so its manifests still verify.

use anyhow::{anyhow, bail, Context, Result};
use base64::Engine;
use ed25519_dalek::pkcs8::{DecodePrivateKey, DecodePublicKey, EncodePrivateKey, EncodePublicKey};
use ed25519_dalek::{Signer, SigningKey, Verifier, VerifyingKey};
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};
use std::io::Write;
use std::path::{Path, PathBuf};

use crate::pyjson::canonical;

pub fn digest(v: &Value) -> String {
    hex::encode(Sha256::digest(canonical(v).as_bytes()))
}

// --- keys -------------------------------------------------------------------------------

/// A new key pair in `directory`. With a passphrase the private key is
/// written encrypted (PKCS#8, AES-256-CBC with a key derived from the
/// passphrase): a stolen private key is a reportable breach (Second
/// Schedule of the Data Protection (General) Regulations), and an encrypted
/// one is of no use without the passphrase.
pub fn keygen(directory: &Path, passphrase: Option<&str>) -> Result<(PathBuf, PathBuf)> {
    std::fs::create_dir_all(directory)?;
    let mut seed = [0u8; 32];
    getrandom_fill(&mut seed)?;
    let key = SigningKey::from_bytes(&seed);
    let priv_path = directory.join("signing_key.pem");
    let pub_path = directory.join("signing_key.pub.pem");
    write_private(&key, &priv_path, passphrase)?;
    let pubpem = key
        .verifying_key()
        .to_public_key_pem(ed25519_dalek::pkcs8::spki::der::pem::LineEnding::LF)
        .map_err(|e| anyhow!("{e}"))?;
    std::fs::write(&pub_path, pubpem.as_bytes())?;
    Ok((priv_path, pub_path))
}

/// Write `key` as PKCS#8 PEM, encrypted with `passphrase` if given,
/// readable by its owner only.
pub fn write_private(key: &SigningKey, path: &Path, passphrase: Option<&str>) -> Result<()> {
    let pem: Vec<u8> = match passphrase {
        Some(p) => {
            let k = openssl::pkey::PKey::private_key_from_raw_bytes(&key.to_bytes(), openssl::pkey::Id::ED25519)?;
            k.private_key_to_pem_pkcs8_passphrase(openssl::symm::Cipher::aes_256_cbc(), p.as_bytes())?
        }
        None => key.to_pkcs8_pem(ed25519_dalek::pkcs8::spki::der::pem::LineEnding::LF).map_err(|e| anyhow!("{e}"))?.as_bytes().to_vec(),
    };
    std::fs::write(path, &pem)?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600))?;
    }
    Ok(())
}

/// The passphrase of an encrypted signing key: `$RFPII_SIGN_PASSWORD`, or
/// asked for on the terminal.
pub fn sign_passphrase(what: &str) -> Result<String> {
    if let Ok(p) = std::env::var("RFPII_SIGN_PASSWORD") {
        return Ok(p);
    }
    if std::io::IsTerminal::is_terminal(&std::io::stdin()) {
        return Ok(rpassword::prompt_password(format!("passphrase for {}: ", what))?);
    }
    bail!("{} is encrypted: set RFPII_SIGN_PASSWORD or run on a terminal", what)
}

pub fn getrandom_fill(buf: &mut [u8]) -> Result<()> {
    use rand::TryRng;
    rand::rngs::SysRng.try_fill_bytes(buf).map_err(|e| anyhow!("no system randomness: {e}"))
}

pub fn load_private(path: &Path) -> Result<SigningKey> {
    load_private_with(path, None)
}

/// A private key; an encrypted one with `passphrase`, or `sign_passphrase`.
pub fn load_private_with(path: &Path, passphrase: Option<&str>) -> Result<SigningKey> {
    let pem = std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    if pem.contains("ENCRYPTED PRIVATE KEY") {
        let pass = match passphrase {
            Some(p) => p.to_string(),
            None => sign_passphrase(&path.display().to_string())?,
        };
        let k = openssl::pkey::PKey::private_key_from_pem_passphrase(pem.as_bytes(), pass.as_bytes())
            .map_err(|_| anyhow!("{}: wrong passphrase, or not an encrypted private key", path.display()))?;
        let raw = k.raw_private_key().map_err(|e| anyhow!("{}: not an Ed25519 private key: {e}", path.display()))?;
        let seed = <[u8; 32]>::try_from(raw.as_slice()).map_err(|_| anyhow!("{}: not an Ed25519 private key", path.display()))?;
        return Ok(SigningKey::from_bytes(&seed));
    }
    SigningKey::from_pkcs8_pem(&pem).map_err(|e| anyhow!("{}: not an Ed25519 private key: {e}", path.display()))
}

pub fn load_public(path: &Path) -> Result<VerifyingKey> {
    let pem = std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    VerifyingKey::from_public_key_pem(&pem).map_err(|e| anyhow!("{}: not an Ed25519 public key: {e}", path.display()))
}

pub fn key_id(k: &VerifyingKey) -> String {
    hex::encode(Sha256::digest(k.as_bytes()))[..16].to_string()
}

fn body(obj: &Value) -> Value {
    let mut m: Map<String, Value> = obj.as_object().cloned().unwrap_or_default();
    m.remove("signature");
    Value::Object(m)
}

/// obj with a `signature` over everything else in it.
pub fn sign(obj: &Value, key: &SigningKey) -> Value {
    let b = body(obj);
    let sig = key.sign(canonical(&b).as_bytes());
    let mut m = b.as_object().cloned().unwrap();
    m.insert(
        "signature".into(),
        json!({"alg": "ed25519", "key_id": key_id(&key.verifying_key()),
               "value": base64::engine::general_purpose::STANDARD.encode(sig.to_bytes())}),
    );
    Value::Object(m)
}

/// "" if the signature is valid, else the reason.
pub fn check_signature(obj: &Value, k: &VerifyingKey) -> String {
    let Some(sig) = obj.get("signature").filter(|s| !s.is_null()) else { return "not signed".into() };
    let kid = sig.get("key_id").and_then(|v| v.as_str()).unwrap_or("");
    if kid != key_id(k) {
        return format!("signed with key {}, not the given key {}", kid, key_id(k));
    }
    let Some(val) = sig.get("value").and_then(|v| v.as_str()) else { return "malformed signature".into() };
    let Ok(raw) = base64::engine::general_purpose::STANDARD.decode(val) else { return "malformed signature".into() };
    let Ok(raw) = <[u8; 64]>::try_from(raw.as_slice()) else { return "malformed signature".into() };
    let s = ed25519_dalek::Signature::from_bytes(&raw);
    if k.verify(canonical(&body(obj)).as_bytes(), &s).is_err() {
        return "signature does not match the manifest contents".into();
    }
    String::new()
}

// --- ledger -----------------------------------------------------------------------------------

fn entry_hash(e: &Value) -> String {
    let mut m = e.as_object().cloned().unwrap_or_default();
    m.remove("entry_hash");
    digest(&Value::Object(m))
}

/// The last entry's hash and the next sequence number.
fn ledger_tail(path: &Path) -> Result<(String, i64)> {
    let (mut prev, mut seq) = ("0".repeat(64), 0i64);
    if let Ok(text) = std::fs::read_to_string(path) {
        if let Some(last) = text.trim().lines().last() {
            let l: Value = serde_json::from_str(last)?;
            prev = l["entry_hash"].as_str().unwrap_or("").to_string();
            seq = l["seq"].as_i64().unwrap_or(0) + 1;
        }
    }
    Ok((prev, seq))
}

pub fn ledger_append(path: &Path, manifest_hash: &str, source_sha256: &str, output_sha256: &str, name: &str) -> Result<Value> {
    let (prev, seq) = ledger_tail(path)?;
    let mut e = json!({"seq": seq, "prev": prev, "name": name, "manifest_sha256": manifest_hash,
                       "source_sha256": source_sha256, "output_sha256": output_sha256});
    let h = entry_hash(&e);
    e["entry_hash"] = Value::String(h);
    write_entry(path, &e)?;
    Ok(e)
}

/// Append a signed event to the ledger: a key rotation, a key revocation,
/// or the withdrawal of a published form. `fields` go in with `type` and
/// `at`; the entry is signed by `key` and chained like any other.
pub fn ledger_event(path: &Path, kind: &str, fields: Value, key: &SigningKey) -> Result<Value> {
    let (prev, seq) = ledger_tail(path)?;
    let mut e = json!({"seq": seq, "prev": prev, "type": kind, "at": crate::process::utc_now()});
    for (k, v) in fields.as_object().cloned().unwrap_or_default() {
        e[k] = v;
    }
    let mut e = sign(&e, key);
    let h = entry_hash(&e);
    e["entry_hash"] = Value::String(h);
    write_entry(path, &e)?;
    Ok(e)
}

/// "" if a ledger event's signature is valid under `k`, else the reason.
pub fn check_event(e: &Value, k: &VerifyingKey) -> String {
    let mut m = e.as_object().cloned().unwrap_or_default();
    m.remove("entry_hash");
    check_signature(&Value::Object(m), k)
}

/// The key that signed `m`, reached from the `published` key through the
/// ledger's key rotations (each signed by the key it replaces), or the
/// reason it can't be trusted: not reached, or revoked before `m` entered
/// the ledger or within the window the revocation names (`since`).
pub fn signing_key(m: &Value, published: &VerifyingKey, entries: &[Value]) -> std::result::Result<(VerifyingKey, Vec<String>), String> {
    let kid = m["signature"]["key_id"].as_str().unwrap_or("").to_string();
    let mut key = *published;
    let mut notes = Vec::new();
    for e in entries.iter().filter(|e| e["type"] == "key_rotation") {
        if key_id(&key) == kid {
            break;
        }
        if e["old_key_id"].as_str() != Some(key_id(&key).as_str()) || !check_event(e, &key).is_empty() {
            continue;
        }
        let Ok(next) = VerifyingKey::from_public_key_pem(e["new_public_key"].as_str().unwrap_or("")) else { continue };
        notes.push(format!("key {} replaced by {} (ledger entry {}, {})", key_id(&key), key_id(&next), e["seq"], e["at"].as_str().unwrap_or("?")));
        key = next;
    }
    if key_id(&key) != kid {
        return Err(format!("signed with key {}, which the published key {} does not lead to through the ledger", kid, key_id(published)));
    }
    let own = digest(m);
    let entered = entries.iter().find(|e| e["manifest_sha256"].as_str() == Some(own.as_str())).and_then(|e| e["seq"].as_i64());
    for e in entries.iter().filter(|e| e["type"] == "key_revocation" && e["key_id"].as_str() == Some(kid.as_str())) {
        if !check_event(e, &key).is_empty() {
            continue;
        }
        let after = entered.is_some_and(|s| Some(s) > e["seq"].as_i64());
        let within = e["since"].as_str().is_some_and(|since| m["created_at"].as_str().is_some_and(|c| c >= since));
        if after || within {
            return Err(format!("signed with key {}, revoked in ledger entry {} ({})", kid, e["seq"], e["reason"].as_str().unwrap_or("")));
        }
    }
    Ok((key, notes))
}

/// One ledger line, keys sorted as Python's json.dumps(sort_keys=True).
fn write_entry(path: &Path, e: &Value) -> Result<()> {
    let mut f = std::fs::OpenOptions::new().create(true).append(true).open(path)?;
    // python: json.dumps(entry, sort_keys=True) -> ", " and ": " separators
    let mut sorted = String::new();
    sorted.push('{');
    let mut keys: Vec<&String> = e.as_object().unwrap().keys().collect();
    keys.sort();
    for (i, k) in keys.iter().enumerate() {
        if i > 0 {
            sorted.push_str(", ");
        }
        sorted.push_str(&crate::pyjson::dumps(&Value::String((*k).clone())));
        sorted.push_str(": ");
        sorted.push_str(&crate::pyjson::dumps(&e[k.as_str()]));
    }
    sorted.push('}');
    writeln!(f, "{}", sorted)?;
    Ok(())
}

pub fn ledger_check(path: &Path) -> Result<(bool, String, Vec<Value>)> {
    let text = std::fs::read_to_string(path)?;
    let entries: Vec<Value> =
        text.lines().filter(|l| !l.trim().is_empty()).map(serde_json::from_str).collect::<Result<_, _>>()?;
    let mut prev = "0".repeat(64);
    for (i, e) in entries.iter().enumerate() {
        if e["seq"].as_i64() != Some(i as i64)
            || e["prev"].as_str() != Some(prev.as_str())
            || e["entry_hash"].as_str() != Some(entry_hash(e).as_str())
        {
            return Ok((false, format!("ledger broken at entry {}", i), entries));
        }
        prev = e["entry_hash"].as_str().unwrap().to_string();
    }
    Ok((true, String::new(), entries))
}

// --- Merkle root over a batch ---------------------------------------------------------------------

fn h2(a: &[u8], b: &[u8]) -> Vec<u8> {
    let mut h = Sha256::new();
    h.update(a);
    h.update(b);
    h.finalize().to_vec()
}

pub fn merkle_root(leaves: &[String]) -> Result<String> {
    let mut level: Vec<Vec<u8>> = leaves.iter().map(hex::decode).collect::<Result<_, _>>()?;
    if level.is_empty() {
        level.push(Sha256::digest(b"").to_vec());
    }
    while level.len() > 1 {
        if level.len() % 2 == 1 {
            level.push(level.last().unwrap().clone());
        }
        level = level.chunks(2).map(|p| h2(&p[0], &p[1])).collect();
    }
    Ok(hex::encode(&level[0]))
}

pub fn merkle_proof(leaves: &[String], mut index: usize) -> Result<Vec<(String, String)>> {
    let mut level: Vec<Vec<u8>> = leaves.iter().map(hex::decode).collect::<Result<_, _>>()?;
    let mut proof = Vec::new();
    while level.len() > 1 {
        if level.len() % 2 == 1 {
            level.push(level.last().unwrap().clone());
        }
        let sib = index ^ 1;
        proof.push((if sib < index { "left" } else { "right" }.to_string(), hex::encode(&level[sib])));
        level = level.chunks(2).map(|p| h2(&p[0], &p[1])).collect();
        index /= 2;
    }
    Ok(proof)
}

pub fn merkle_verify(leaf: &str, proof: &[(String, String)], root: &str) -> Result<bool> {
    let mut h = hex::decode(leaf)?;
    for (side, sib) in proof {
        let s = hex::decode(sib)?;
        h = if side == "left" { h2(&s, &h) } else { h2(&h, &s) };
    }
    Ok(hex::encode(h) == root)
}

pub fn read_json(path: &Path) -> Result<Value> {
    let text = std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    serde_json::from_str(&text).map_err(|e| anyhow!("{}: {e}", path.display()))
}

pub fn ensure(cond: bool, msg: &str) -> Result<()> {
    if !cond {
        bail!("{}", msg);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn merkle() {
        let leaves: Vec<String> = (0..5u8).map(|i| hex::encode(Sha256::digest([i]))).collect();
        let root = merkle_root(&leaves).unwrap();
        for i in 0..5 {
            assert!(merkle_verify(&leaves[i], &merkle_proof(&leaves, i).unwrap(), &root).unwrap());
        }
    }
}

#[cfg(test)]
#[path = "tests/manifest.rs"]
mod py_tests;
