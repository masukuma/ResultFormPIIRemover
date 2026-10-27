"""Signed manifests, an append-only ledger and batch roots.

Every published form gets a manifest saying exactly what was done to it:
the source file's hash (to match the copy on the IEBC portal / KIEMS
transmission), the output's hash, the tool and template versions, the random
seed, and the bitmap of 8x8 blocks that were blacked out. The manifest is
signed (Ed25519) by the publisher.

Manifests are appended to a hash-chained ledger: each entry commits to the
previous one, so an entry can't be changed, dropped or back-dated without
breaking the chain. Each batch run ends with a Merkle root over its manifests,
signed, small enough to publish widely (website, newspaper, social media) the
moment the batch is released. Anyone holding a manifest can later prove it
was in that batch, and nobody can quietly swap a published file afterwards.
"""

from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey


def canonical(obj) -> bytes:
    """Deterministic JSON bytes for hashing and signing."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def digest(obj) -> str:
    return hashlib.sha256(canonical(obj)).hexdigest()


# --- keys ------------------------------------------------------------------------

def keygen(directory: str | Path) -> tuple[Path, Path]:
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    key = Ed25519PrivateKey.generate()
    priv = d / "signing_key.pem"
    pub = d / "signing_key.pub.pem"
    priv.write_bytes(key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                                       serialization.NoEncryption()))
    priv.chmod(0o600)
    pub.write_bytes(key.public_key().public_bytes(serialization.Encoding.PEM,
                                                  serialization.PublicFormat.SubjectPublicKeyInfo))
    return priv, pub


def load_private(path: str | Path) -> Ed25519PrivateKey:
    return serialization.load_pem_private_key(Path(path).read_bytes(), password=None)


def load_public(path: str | Path) -> Ed25519PublicKey:
    return serialization.load_pem_public_key(Path(path).read_bytes())


def key_id(pub: Ed25519PublicKey) -> str:
    raw = pub.public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    return hashlib.sha256(raw).hexdigest()[:16]


def sign(obj: dict, key: Ed25519PrivateKey) -> dict:
    """Return obj with a ``signature`` over everything else in it."""
    body = {k: v for k, v in obj.items() if k != "signature"}
    sig = key.sign(canonical(body))
    return {**body, "signature": {"alg": "ed25519", "key_id": key_id(key.public_key()),
                                  "value": base64.b64encode(sig).decode()}}


def check_signature(obj: dict, pub: Ed25519PublicKey) -> str:
    """"" if the signature is valid, else the reason."""
    sig = obj.get("signature")
    if not sig:
        return "not signed"
    if sig.get("key_id") != key_id(pub):
        return f"signed with key {sig.get('key_id')}, not the given key {key_id(pub)}"
    body = {k: v for k, v in obj.items() if k != "signature"}
    try:
        pub.verify(base64.b64decode(sig["value"]), canonical(body))
    except Exception:
        return "signature does not match the manifest contents"
    return ""


# --- ledger ------------------------------------------------------------------------

def _entry_hash(entry: dict) -> str:
    return digest({k: v for k, v in entry.items() if k != "entry_hash"})


def ledger_append(path: str | Path, manifest_hash: str, source_sha256: str, output_sha256: str,
                  name: str) -> dict:
    p = Path(path)
    prev, seq = "0" * 64, 0
    if p.exists() and p.stat().st_size:
        last = json.loads(p.read_text().strip().splitlines()[-1])
        prev, seq = last["entry_hash"], last["seq"] + 1
    entry = {"seq": seq, "prev": prev, "name": name, "manifest_sha256": manifest_hash,
             "source_sha256": source_sha256, "output_sha256": output_sha256}
    entry["entry_hash"] = _entry_hash(entry)
    with open(p, "a") as f:
        f.write(json.dumps(entry, sort_keys=True) + "\n")
    return entry


def ledger_check(path: str | Path) -> tuple[bool, str, list[dict]]:
    entries = [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
    prev = "0" * 64
    for i, e in enumerate(entries):
        if e["seq"] != i or e["prev"] != prev or e["entry_hash"] != _entry_hash(e):
            return False, f"ledger broken at entry {i}", entries
        prev = e["entry_hash"]
    return True, "", entries


# --- Merkle root over a batch ---------------------------------------------------------

def merkle_root(leaves: list[str]) -> str:
    level = [bytes.fromhex(h) for h in leaves] or [hashlib.sha256(b"").digest()]
    while len(level) > 1:
        if len(level) % 2:
            level.append(level[-1])
        level = [hashlib.sha256(level[i] + level[i + 1]).digest() for i in range(0, len(level), 2)]
    return level[0].hex()


def merkle_proof(leaves: list[str], index: int) -> list[tuple[str, str]]:
    """Sibling hashes (with side) from leaf ``index`` up to the root."""
    level = [bytes.fromhex(h) for h in leaves]
    proof = []
    while len(level) > 1:
        if len(level) % 2:
            level.append(level[-1])
        sib = index ^ 1
        proof.append(("left" if sib < index else "right", level[sib].hex()))
        level = [hashlib.sha256(level[i] + level[i + 1]).digest() for i in range(0, len(level), 2)]
        index //= 2
    return proof


def merkle_verify(leaf: str, proof: list, root: str) -> bool:
    h = bytes.fromhex(leaf)
    for side, sib in proof:
        s = bytes.fromhex(sib)
        h = hashlib.sha256(s + h if side == "left" else h + s).digest()
    return h.hex() == root
