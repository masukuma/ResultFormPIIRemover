use super::*;
use crate::testutil::tmpdir;

#[test]
fn sign_and_detect_edits() {
    let d = tmpdir("manifest-sign");
    let (private, public) = keygen(&d).unwrap();
    let mut m = sign(&json!({"a": 1, "pages": [{"blocks": "xyz"}]}), &load_private(&private).unwrap());
    let pk = load_public(&public).unwrap();
    assert_eq!(check_signature(&m, &pk), "");
    m["pages"][0]["blocks"] = json!("xy");
    assert!(check_signature(&m, &pk).contains("does not match"));
    let other = keygen(&d.join("other")).unwrap().1;
    assert!(check_signature(&m, &load_public(&other).unwrap()).contains("not the given key"));
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
