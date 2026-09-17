//! P1 CONTRACT 3 (`docs/P1-STATE-API.md`): `GET /v1/state`, `POST
//! /v1/state/export`, `POST /v1/state/import`. A thin adapter over
//! `checkpoints.rs`'s existing engine-submission plumbing (`Registry`,
//! `Ckpt`, `Gen`/`check_and_submit`) plus a LAT1 wrap/unwrap, not a parallel
//! reimplementation: `create()` already runs the prefill and `state_save`
//! export needs, `fork()` already runs the `state_load` restore import
//! needs (room A, 2026-09-17 design review with codex).
//!
//! Build order: this commit lands only `exact_prefix_hash`, the one piece
//! the routes below all depend on and the one place a byte off-by-one would
//! be silent. Routes follow in later commits.

use crate::checkpoints::sha256;

/// CONTRACT 1: the unsalted `prefix_hash` field, `sha256(tokens[0:pos]` as
/// little-endian u32 bytes `)`, first 8 digest bytes read back as a
/// little-endian u64. Deliberately not `checkpoints::Registry::id_for`:
/// that hashes the WHOLE `tokens` slice (an off-by-one against `pos`, since
/// `pos` there is only a label in the id string) and formats a hex string
/// for the HTTP-facing checkpoint id, not this binary LAT1 header field.
///
/// `#[allow(dead_code)]`: this commit lands only the hash helper (room A,
/// 2026-09-17 build order); the `GET/POST /v1/state*` routes that call it
/// land in a following commit. Remove the allow once they do.
#[allow(dead_code)]
pub fn exact_prefix_hash(tokens: &[u32], pos: usize) -> u64 {
    let mut bytes = Vec::with_capacity(pos * 4);
    for &t in &tokens[..pos] {
        bytes.extend_from_slice(&t.to_le_bytes());
    }
    let digest = sha256(&bytes);
    u64::from_le_bytes(digest[..8].try_into().unwrap())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_a_hand_computed_vector() {
        // tokens[0:2] = [1, 2] as little-endian u32: 01 00 00 00 02 00 00 00.
        // Independently computed with Python (not just testing the algorithm
        // against itself):
        //   python3 -c "import hashlib,struct; \
        //     h=hashlib.sha256(bytes([1,0,0,0,2,0,0,0])).hexdigest(); \
        //     print(h[:16], struct.unpack('<Q', bytes.fromhex(h[:16]))[0])"
        //   -> 34fb5c825de7ca4a 5389374292907326260
        let tokens = [1u32, 2, 999999]; // pos=2 must ignore the trailing token
        assert_eq!(exact_prefix_hash(&tokens, 2), 5389374292907326260u64);
    }

    #[test]
    fn pos_excludes_the_token_at_pos() {
        // tokens[0:pos], not tokens[0:pos+1]: the token AT pos must not
        // change the hash (this is exactly the off-by-one id_for has).
        let a = exact_prefix_hash(&[10, 20, 30], 2);
        let b = exact_prefix_hash(&[10, 20, 999], 2);
        assert_eq!(a, b, "the hash must not depend on tokens[pos]");
        let c = exact_prefix_hash(&[10, 20, 30], 3);
        assert_ne!(a, c, "but it must depend on everything before pos");
    }

    #[test]
    fn empty_prefix_hashes_the_empty_byte_string() {
        assert_eq!(exact_prefix_hash(&[1, 2, 3], 0), exact_prefix_hash(&[], 0));
    }
}
