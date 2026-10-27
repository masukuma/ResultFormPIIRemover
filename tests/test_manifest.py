import json

from resultform_pii import manifest as mf


def test_sign_and_detect_edits(tmp_path):
    priv, pub = mf.keygen(tmp_path)
    m = mf.sign({"a": 1, "pages": [{"blocks": "xyz"}]}, mf.load_private(priv))
    assert mf.check_signature(m, mf.load_public(pub)) == ""
    m["pages"][0]["blocks"] = "xy"
    assert "does not match" in mf.check_signature(m, mf.load_public(pub))
    other = mf.keygen(tmp_path / "other")[1]
    assert "not the given key" in mf.check_signature(m, mf.load_public(other))


def test_ledger_chain_detects_rewrites(tmp_path):
    led = tmp_path / "ledger.jsonl"
    for i in range(4):
        mf.ledger_append(led, f"{i:064x}", "s" * 64, "o" * 64, f"doc{i}")
    assert mf.ledger_check(led)[0]
    lines = led.read_text().splitlines()
    e = json.loads(lines[1])
    e["output_sha256"] = "x" * 64  # swap a published file
    lines[1] = json.dumps(e)
    led.write_text("\n".join(lines) + "\n")
    ok, why, _ = mf.ledger_check(led)
    assert not ok and "entry 1" in why


def test_merkle_inclusion():
    leaves = [f"{i:064x}" for i in range(7)]
    root = mf.merkle_root(leaves)
    for i, leaf in enumerate(leaves):
        assert mf.merkle_verify(leaf, mf.merkle_proof(leaves, i), root)
    assert not mf.merkle_verify("f" * 64, mf.merkle_proof(leaves, 0), root)
