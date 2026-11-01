//! Identifiers, random secrets, hashes and constant-time comparison.

use sha2::{Digest, Sha256};
use subtle::ConstantTimeEq;

/// A new identifier: a prefix and a UUIDv7 (time-ordered), e.g.
/// `job_019a3c...`. Sorting ids sorts by creation time.
pub fn new_id(prefix: &str) -> String {
    format!("{}_{}", prefix, uuid::Uuid::now_v7().simple())
}

/// `N` bytes from the operating system's random source.
pub fn random_bytes<const N: usize>() -> [u8; N] {
    let mut b = [0u8; N];
    getrandom::fill(&mut b).expect("the operating system's random source failed");
    b
}

const BASE62: &[u8; 62] = b"0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";

/// `bytes` as a base62 number (big-endian), most significant digit first.
pub fn base62(bytes: &[u8]) -> String {
    let mut n: Vec<u8> = bytes.to_vec();
    let mut digits = Vec::new();
    while n.iter().any(|&b| b != 0) {
        let mut rem: u32 = 0;
        for b in n.iter_mut() {
            let acc = (rem << 8) | u32::from(*b);
            *b = (acc / 62) as u8;
            rem = acc % 62;
        }
        digits.push(BASE62[rem as usize]);
    }
    if digits.is_empty() {
        digits.push(b'0');
    }
    digits.reverse();
    String::from_utf8(digits).unwrap_or_default()
}

pub fn sha256_hex(data: &[u8]) -> String {
    hex(&Sha256::digest(data))
}

pub fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// Whether `a` and `b` are equal, in time that doesn't depend on where they
/// differ.
pub fn ct_eq(a: &[u8], b: &[u8]) -> bool {
    a.len() == b.len() && bool::from(a.ct_eq(b))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn base62_is_a_big_endian_number() {
        assert_eq!(base62(&[0]), "0");
        assert_eq!(base62(&[61]), "z");
        assert_eq!(base62(&[62]), "10");
        assert_eq!(base62(&[1, 0]), "48"); // 256 = 4 * 62 + 8
        let k = base62(&[0xff; 32]);
        assert!(k.len() == 43 && k.bytes().all(|c| c.is_ascii_alphanumeric()));
    }

    #[test]
    fn ids_sort_by_creation_and_secrets_compare_in_constant_time() {
        let (a, b) = (new_id("job"), new_id("job"));
        assert!(a.starts_with("job_") && a.len() == 36 && a < b);
        assert!(ct_eq(b"abc", b"abc") && !ct_eq(b"abc", b"abd") && !ct_eq(b"abc", b"ab"));
        assert_ne!(random_bytes::<16>(), random_bytes::<16>());
        assert_eq!(sha256_hex(b"abc"), "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
    }
}
