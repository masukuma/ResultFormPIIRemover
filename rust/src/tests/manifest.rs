use super::*;
use crate::testutil::tmpdir;

#[test]
fn sign_and_detect_edits() {
    let d = tmpdir("manifest-sign");
    let (private, public) = keygen(&d, None).unwrap();
    let mut m = sign(&json!({"a": 1, "pages": [{"blocks": "xyz"}]}), &load_private(&private).unwrap());
    let pk = load_public(&public).unwrap();
    assert_eq!(check_signature(&m, &pk), "");
    m["pages"][0]["blocks"] = json!("xy");
    assert!(check_signature(&m, &pk).contains("does not match"));
    let other = keygen(&d.join("other"), None).unwrap().1;
    assert!(check_signature(&m, &load_public(&other).unwrap()).contains("not the given key"));
}

#[test]
fn an_encrypted_key_needs_its_passphrase() {
    let d = tmpdir("manifest-encrypted");
    let (private, public) = keygen(&d, Some("correct horse")).unwrap();
    let pem = std::fs::read_to_string(&private).unwrap();
    assert!(pem.contains("ENCRYPTED PRIVATE KEY") && !pem.contains("BEGIN PRIVATE KEY"));
    assert!(load_private_with(&private, Some("wrong")).unwrap_err().to_string().contains("wrong passphrase"));
    let k = load_private_with(&private, Some("correct horse")).unwrap();
    assert_eq!(key_id(&k.verifying_key()), key_id(&load_public(&public).unwrap()));
}

/// A manifest of `name`, signed by `key`, recorded in `led`.
fn published(name: &str, key: &SigningKey, led: &Path) -> Value {
    let m = sign(&json!({"name": name, "created_at": crate::process::utc_now()}), key);
    ledger_append(led, &digest(&m), &"s".repeat(64), &"o".repeat(64), name).unwrap();
    m
}

#[test]
fn a_rotated_key_verifies_through_the_ledger_until_it_is_revoked() {
    let d = tmpdir("manifest-rotate");
    let led = d.join("ledger.jsonl");
    let (p1, q1) = keygen(&d.join("k1"), None).unwrap();
    let (k1, pub1) = (load_private(&p1).unwrap(), load_public(&q1).unwrap());
    let before = published("before", &k1, &led);
    // the old key hands over to a new one, in the ledger
    let (p2, q2) = keygen(&d.join("k2"), None).unwrap();
    let k2 = load_private(&p2).unwrap();
    ledger_event(&led, "key_rotation", json!({"old_key_id": key_id(&pub1), "new_key_id": key_id(&k2.verifying_key()), "new_public_key": std::fs::read_to_string(&q2).unwrap()}), &k1).unwrap();
    let after = published("after", &k2, &led);
    let (ok, _, entries) = ledger_check(&led).unwrap();
    assert!(ok, "signed events keep the chain");
    for m in [&before, &after] {
        let (k, _) = signing_key(m, &pub1, &entries).unwrap();
        assert_eq!(check_signature(m, &k), "");
    }
    // a key nothing hands over to is not trusted
    let (p3, _) = keygen(&d.join("k3"), None).unwrap();
    let stray = published("stray", &load_private(&p3).unwrap(), &led);
    let (_, _, entries) = ledger_check(&led).unwrap();
    assert!(signing_key(&stray, &pub1, &entries).unwrap_err().contains("does not lead to"));
    // the new key is revoked: what it signed before still verifies, what it signs after doesn't
    ledger_event(&led, "key_revocation", json!({"key_id": key_id(&k2.verifying_key()), "reason": "laptop stolen"}), &k2).unwrap();
    let late = published("late", &k2, &led);
    let (ok, _, entries) = ledger_check(&led).unwrap();
    assert!(ok);
    assert!(signing_key(&after, &pub1, &entries).is_ok());
    assert!(signing_key(&late, &pub1, &entries).unwrap_err().contains("revoked"));
    // ... and back to a date: everything made since then
    ledger_event(&led, "key_revocation", json!({"key_id": key_id(&k2.verifying_key()), "reason": "stolen earlier", "since": "2000-01-01T00:00:00"}), &k2).unwrap();
    let (_, _, entries) = ledger_check(&led).unwrap();
    assert!(signing_key(&after, &pub1, &entries).unwrap_err().contains("revoked"));
}

#[test]
fn ledger_chain_detects_rewrites() {
    let d = tmpdir("ledger");
    let led = d.join("ledger.jsonl");
    for i in 0..4 {
        ledger_append(&led, &format!("{:064x}", i), &"s".repeat(64), &"o".repeat(64), &format!("doc{}", i)).unwrap();
    }
    assert!(ledger_check(&led).unwrap().0);
    let text = std::fs::read_to_string(&led).unwrap();
    let mut lines: Vec<String> = text.lines().map(|l| l.to_string()).collect();
    let mut e: Value = serde_json::from_str(&lines[1]).unwrap();
    e["output_sha256"] = json!("x".repeat(64)); // swap a published file
    lines[1] = serde_json::to_string(&e).unwrap();
    std::fs::write(&led, lines.join("\n") + "\n").unwrap();
    let (ok, why, _) = ledger_check(&led).unwrap();
    assert!(!ok && why.contains("entry 1"), "{}", why);
}

#[test]
fn merkle_inclusion() {
    let leaves: Vec<String> = (0..7).map(|i| format!("{:064x}", i)).collect();
    let root = merkle_root(&leaves).unwrap();
    for (i, leaf) in leaves.iter().enumerate() {
        assert!(merkle_verify(leaf, &merkle_proof(&leaves, i).unwrap(), &root).unwrap());
    }
    assert!(!merkle_verify(&"f".repeat(64), &merkle_proof(&leaves, 0).unwrap(), &root).unwrap());
}
