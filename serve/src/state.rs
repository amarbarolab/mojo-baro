//! P1 CONTRACT 3 (`docs/P1-STATE-API.md`): `GET /v1/state`, `POST
//! /v1/state/export`, `POST /v1/state/import`. A thin adapter over
//! `checkpoints.rs`'s existing engine-submission plumbing (`Registry`,
//! `Ckpt`, `Gen`/`check_and_submit`) plus a LAT1 wrap/unwrap, not a parallel
//! reimplementation: `create()` already runs the prefill and `state_save`
//! export needs, `fork()` already runs the `state_load` restore import
//! needs (room A, 2026-09-17 design review with codex).
//!
//! Build order: `GET /v1/state` (read-only, no GPU) landed first. This
//! commit adds `POST /v1/state/export`, `path` variant only: the caller
//! names a file, the response is JSON metadata, not the LAT1-wrapped
//! streaming body CONTRACT 1 also describes -- that is a following commit,
//! built on this one's engine-submission logic. `POST /v1/state/import`
//! follows after.
//!
//! Two spec-vs-reality gaps found while building this (room A, 2026-09-17):
//! CONTRACT 3's per-request `n = 0` prefill does not exist (`check_and_submit`
//! rejects `n == 0`); this uses `n = 1`, the same working pattern
//! `checkpoints::create` already ships. And the per-request `"format"` field
//! cannot be honored: `serve/engine.mojo:86` reads `BARO_STATE_INT8` once at
//! server startup, there is no per-request override on the wire, so a
//! request whose `format` disagrees with the server's actual fixed format
//! gets a 409 naming both, not a silently wrong file.

use super::*;
use crate::checkpoints::sha256;

/// CONTRACT 1: the unsalted `prefix_hash` field, `sha256(tokens[0:pos]` as
/// little-endian u32 bytes `)`, first 8 digest bytes read back as a
/// little-endian u64. Deliberately not `checkpoints::Registry::id_for`:
/// that hashes the WHOLE `tokens` slice (an off-by-one against `pos`, since
/// `pos` there is only a label in the id string) and formats a hex string
/// for the HTTP-facing checkpoint id, not this binary LAT1 header field.
pub fn exact_prefix_hash(tokens: &[u32], pos: usize) -> u64 {
    let mut bytes = Vec::with_capacity(pos * 4);
    for &t in &tokens[..pos] {
        bytes.extend_from_slice(&t.to_le_bytes());
    }
    let digest = sha256(&bytes);
    u64::from_le_bytes(digest[..8].try_into().unwrap())
}

/// `BARO_KVQ=int8` is the only non-`f32` value the engine (`serve/engine.mojo`)
/// acts on (A2 step 2); anything else, including unset, is the f32 default.
fn kv_kind() -> &'static str {
    if std::env::var("BARO_KVQ").map(|v| v == "int8").unwrap_or(false) {
        "int8"
    } else {
        "f32"
    }
}

/// `GET /v1/state` (CONTRACT 3): no GPU work, reads straight from the
/// existing HTTP-facing `checkpoints::Registry` (room A design review: this
/// is "the chain" CONTRACT 3 means, not the Mojo engine's internal prefix
/// `Chain`, a separate, GPU-resident structure this Rust layer cannot see
/// into). `pinned`/`boundary` are always `false`: the Registry is a flat
/// TTL-and-cap cache with no priority-retention concept, unlike the Mojo
/// engine's own checkpoint chain (`serve/prefix.mojo`) -- reporting them as
/// `false` here is honest about that, not a stand-in for real values.
pub async fn list(State(app): State<Shared>) -> Json<Value> {
    let states: Vec<Value> = app
        .ckpts
        .live(now())
        .into_iter()
        .map(|c| {
            json!({
                "prefix_hash": format!("{:016x}", exact_prefix_hash(&c.tokens, c.pos)),
                "pos": c.pos,
                "bytes": c.bytes,
                "pinned": false,
                "boundary": false,
                "age_s": now().saturating_sub(c.created),
            })
        })
        .collect();
    Json(json!({
        "model": app.model,
        "weights_uuid": app.identity.weights_uuid_hex(),
        "portable": app.identity.portable,
        "kv": kv_kind(),
        "states": states,
    }))
}

/// `BARO_STATE_INT8=1` picks BAROST02 (int8) state files at server startup
/// (`serve/engine.mojo:86`); anything else, including unset, is BAROST01
/// (f32). No per-request override exists on the wire.
fn state_file_format() -> &'static str {
    if std::env::var("BARO_STATE_INT8").map(|v| v == "1").unwrap_or(false) {
        "int8"
    } else {
        "f32"
    }
}

/// CONTRACT 3: an engine running `BARO_KVQ=int8` refuses `state_save`/
/// `state_load` with its own wire error (`serve/engine.mojo:68,157`); map
/// that specific, known refusal to the documented 501 `kvq_state_open`
/// instead of `collect`'s generic 400 for an unrecognized engine error.
fn map_kvq_refusal(e: ApiError) -> ApiError {
    match &e {
        ApiError::Plain(_, msg) if msg.contains("quantized KV state is A2 step 3") => ApiError::Plain(
            StatusCode::NOT_IMPLEMENTED,
            "kvq_state_open: this engine's KV is quantized (BARO_KVQ), export/import need f32 KV (A2 step 3)".into(),
        ),
        _ => e,
    }
}

#[derive(Deserialize)]
struct ExportMessage {
    role: String,
    #[serde(default)]
    content: Value,
}

#[derive(Deserialize)]
pub struct ExportReq {
    #[serde(default)]
    prompt: Option<Value>,
    #[serde(default)]
    messages: Option<Vec<ExportMessage>>,
    #[serde(default)]
    tokens: Option<Vec<u32>>,
    #[serde(default)]
    pos: Option<usize>,
    #[serde(default)]
    format: Option<String>,
    #[serde(default)]
    path: Option<String>,
}

/// `(prompt token ids, role-boundary positions)`; boundaries are only ever
/// non-empty for a `messages` request (`Text::role_boundaries`), matching
/// CONTRACT 3's default-`pos` rule.
fn export_prompt(app: &App, r: &ExportReq) -> Result<(Vec<u32>, Vec<u32>), ApiError> {
    if let Some(msgs) = &r.messages {
        let t = need_text(app)?;
        if msgs.is_empty() {
            return Err(bad("messages is empty"));
        }
        let chat_msgs: Vec<ChatMessage> = msgs
            .iter()
            .map(|m| Ok(ChatMessage { role: m.role.clone(), content: content_text(&m.content)?, tool_calls: None }))
            .collect::<Result<Vec<_>, ApiError>>()?;
        let rendered = t.apply_chat_template(&chat_msgs, None, None).map_err(|e| ApiError::Plain(StatusCode::INTERNAL_SERVER_ERROR, e))?;
        let prompt = t.encode(&rendered, true).map_err(bad)?;
        let boundaries = t.role_boundaries(&chat_msgs);
        return Ok((prompt, boundaries));
    }
    if let Some(tokens) = &r.tokens {
        if tokens.is_empty() {
            return Err(bad("tokens is empty"));
        }
        return Ok((tokens.clone(), vec![]));
    }
    if let Some(p) = &r.prompt {
        return Ok((prompt_ids(app, p)?, vec![]));
    }
    Err(bad("one of prompt, messages, or tokens is required"))
}

/// CONTRACT 3: "pos defaults to the last role boundary at or before the
/// prompt end, else the prompt end minus one."
fn default_pos(prompt_len: usize, boundaries: &[u32]) -> usize {
    boundaries
        .iter()
        .copied()
        .filter(|&b| (b as usize) <= prompt_len)
        .max()
        .map(|b| b as usize)
        .unwrap_or_else(|| prompt_len.saturating_sub(1))
}

/// `POST /v1/state/export`, `path` variant: `{"path","bytes","pos",
/// "prefix_hash"}`. Runs the same `n=1, state_save` request
/// `checkpoints::create` already proves works, writing to the caller's own
/// path rather than the Registry's managed one -- this is a one-off dump,
/// not a tracked, TTL'd checkpoint.
pub async fn export(State(app): State<Shared>, Json(r): Json<ExportReq>) -> Result<Json<Value>, ApiError> {
    let Some(path) = r.path.clone() else {
        return Err(ApiError::Plain(
            StatusCode::NOT_IMPLEMENTED,
            "the streaming LAT1 response (no \"path\") is not built yet; pass \"path\" for the file-metadata form".into(),
        ));
    };
    let server_format = state_file_format();
    if let Some(requested) = &r.format {
        if requested != server_format {
            return Err(ApiError::Plain(
                StatusCode::CONFLICT,
                format!("this server writes state files as {server_format} (BARO_STATE_INT8 at startup), it cannot honor format={requested} per request"),
            ));
        }
    }
    let (prompt, boundaries) = export_prompt(&app, &r)?;
    if prompt.is_empty() {
        return Err(bad("prompt resolves to zero tokens"));
    }
    let pos = r.pos.unwrap_or_else(|| default_pos(prompt.len(), &boundaries));
    if pos == 0 || pos > prompt.len() {
        return Err(bad(format!("pos must be in 1..={} (the prompt length), got {pos}", prompt.len())));
    }
    let g = Gen {
        prompt: prompt.clone(),
        n: 1,
        spec: false,
        stream: false,
        stop: vec![],
        ckpt: vec![],
        state: (Some(path.clone()), None),
        sample: protocol::SampleParams::default(),
        schema: None,
        reasoning: None,
        embed: None,
    };
    let (_req_id, rx) = check_and_submit(&app, &g)?;
    collect(&app, rx).await.map_err(map_kvq_refusal)?;
    let bytes = std::fs::metadata(&path)
        .map(|m| m.len())
        .map_err(|e| ApiError::Plain(StatusCode::BAD_GATEWAY, format!("engine finished but wrote no state file at {path}: {e}")))?;
    Ok(Json(json!({
        "path": path,
        "bytes": bytes,
        "pos": pos,
        "prefix_hash": format!("{:016x}", exact_prefix_hash(&prompt, pos)),
    })))
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

    #[test]
    fn default_pos_picks_the_last_boundary_at_or_before_the_prompt_end() {
        assert_eq!(default_pos(10, &[3, 7, 12]), 7, "12 is past the prompt end, must be excluded");
        assert_eq!(default_pos(10, &[3, 7, 10]), 10, "a boundary AT the prompt end still counts");
    }

    #[test]
    fn default_pos_falls_back_to_prompt_end_minus_one_with_no_boundaries() {
        assert_eq!(default_pos(10, &[]), 9);
        assert_eq!(default_pos(0, &[]), 0, "saturating, never underflows");
    }

    #[test]
    fn map_kvq_refusal_rewrites_only_the_documented_engine_message() {
        let refusal = ApiError::Plain(
            StatusCode::BAD_REQUEST,
            "engine: BARO_STATE_SAVE: state files store f32 KV; this engine has BARO_KVQ=int8 (quantized KV state is A2 step 3)".into(),
        );
        match map_kvq_refusal(refusal) {
            ApiError::Plain(code, msg) => {
                assert_eq!(code, StatusCode::NOT_IMPLEMENTED);
                assert!(msg.contains("kvq_state_open"), "{msg}");
            }
            _ => panic!("expected ApiError::Plain"),
        }
        let unrelated = ApiError::Plain(StatusCode::BAD_REQUEST, "engine: prompt+n exceeds TMAX 128".into());
        match map_kvq_refusal(unrelated) {
            ApiError::Plain(code, msg) => {
                assert_eq!(code, StatusCode::BAD_REQUEST, "an unrelated engine error must pass through unchanged");
                assert!(msg.contains("TMAX"));
            }
            _ => panic!("expected ApiError::Plain"),
        }
    }
}
