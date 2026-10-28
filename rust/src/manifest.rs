//! Signed manifests, an append-only ledger and batch roots.
//!
//! Every published form gets a manifest saying exactly what was done to it,
//! signed (Ed25519) by the publisher, appended to a hash-chained ledger, and
//! covered by the Merkle root of its batch. Formats, hashes and signatures
//! are the same as the Python version's, so either can verify the other's.

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

pub fn keygen(directory: &Path) -> Result<(PathBuf, PathBuf)> {
    std::fs::create_dir_all(directory)?;
    let mut seed = [0u8; 32];
    getrandom_fill(&mut seed)?;
    let key = SigningKey::from_bytes(&seed);
    let priv_path = directory.join("signing_key.pem");
    let pub_path = directory.join("signing_key.pub.pem");
    let pem = key.to_pkcs8_pem(ed25519_dalek::pkcs8::spki::der::pem::LineEnding::LF).map_err(|e| anyhow!("{e}"))?;
    std::fs::write(&priv_path, pem.as_bytes())?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&priv_path, std::fs::Permissions::from_mode(0o600))?;
    }
    let pubpem = key
        .verifying_key()
        .to_public_key_pem(ed25519_dalek::pkcs8::spki::der::pem::LineEnding::LF)
        .map_err(|e| anyhow!("{e}"))?;
    std::fs::write(&pub_path, pubpem.as_bytes())?;
    Ok((priv_path, pub_path))
}

pub fn getrandom_fill(buf: &mut [u8]) -> Result<()> {
    use rand::TryRng;
    rand::rngs::SysRng.try_fill_bytes(buf).map_err(|e| anyhow!("no system randomness: {e}"))
}

pub fn load_private(path: &Path) -> Result<SigningKey> {
    let pem = std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
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

pub fn ledger_append(path: &Path, manifest_hash: &str, source_sha256: &str, output_sha256: &str, name: &str) -> Result<Value> {
    let (mut prev, mut seq) = ("0".repeat(64), 0i64);
    if let Ok(text) = std::fs::read_to_string(path) {
        if let Some(last) = text.trim().lines().last() {
            let l: Value = serde_json::from_str(last)?;
            prev = l["entry_hash"].as_str().unwrap_or("").to_string();
            seq = l["seq"].as_i64().unwrap_or(0) + 1;
        }
    }
    let mut e = json!({"seq": seq, "prev": prev, "name": name, "manifest_sha256": manifest_hash,
                       "source_sha256": source_sha256, "output_sha256": output_sha256});
    let h = entry_hash(&e);
    e["entry_hash"] = Value::String(h);
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
    Ok(e)
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
