//! P1 CONTRACT 3 (`docs/P1-STATE-API.md`): `GET /v1/state`, `POST
//! /v1/state/export`, `POST /v1/state/import`. A thin adapter over
//! `checkpoints.rs`'s existing engine-submission plumbing (`Registry`,
//! `Ckpt`, `Gen`/`check_and_submit`) plus a LAT1 wrap/unwrap, not a parallel
//! reimplementation: `create()` already runs the prefill and `state_save`
//! export needs, `fork()` already runs the `state_load` restore import
//! needs (room A, 2026-09-17 design review with codex).
//!
//! Build order: `GET /v1/state` (read-only, no GPU), then `POST
//! /v1/state/export` (`path` variant), both landed. This commit adds `POST
//! /v1/state/import`, also the `path` variant: the caller names an existing
//! raw `BAROST01`/`BAROST02` file (as `export`'s `path` form writes), the
//! response is JSON metadata. The LAT1-wrapped streaming body/response
//! CONTRACT 1 also describes, for both routes, is a following commit.
//!
//! Gaps found while building this, handled honestly rather than silently
//! (room A, 2026-09-17): CONTRACT 3's per-request `n = 0` prefill does not
//! exist (`check_and_submit` rejects `n == 0`); this uses `n = 1`, the same
//! working pattern `checkpoints::create` already ships. The per-request
//! `"format"` field cannot be honored: `serve/engine.mojo:86` reads
//! `BARO_STATE_INT8` once at server startup, so export 409s a mismatched
//! request instead of writing a silently wrong file. And `runtime_differs`
//! in import's response is always `null`, not `false`: a raw BAROST0x file
//! (`serve/engine.mojo`'s `save_state`) carries no `runtime` field at all
//! (magic, pos, CONV_SLOT, SSM_SLOT, kvn, salt, tokens -- that's the whole
//! 72-byte header plus the token array), only the LAT1 wrapper does, so
//! there is nothing to compare yet; `false` would claim a check that never
//! happened.

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

/// `(pos, tokens[0:pos])` embedded in a `BAROST01`/`BAROST02` file
/// (`serve/engine.mojo`'s `save_state`/`load_state`): magic(8) pos(i64 LE)
/// CONV_SLOT(i64 LE) SSM_SLOT(i64 LE) kvn(i64 LE) salt(32) tokens(pos*4,
/// i32 LE each) -- a 72-byte fixed header, then the token array; the salt
/// and payload past it are not read here, the engine re-checks the salt
/// itself during `load_state` and refuses a mismatch on its own.
fn read_state_header(path: &str) -> Result<(usize, Vec<u32>), ApiError> {
    let data = std::fs::read(path).map_err(|e| ApiError::Plain(StatusCode::BAD_REQUEST, format!("cannot read {path}: {e}")))?;
    if data.len() < 72 {
        return Err(bad(format!("{path} is too short to be a BAROST01/BAROST02 state file")));
    }
    if &data[0..8] != b"BAROST01" && &data[0..8] != b"BAROST02" {
        return Err(bad(format!("{path} is not a BAROST01/BAROST02 state file")));
    }
    let pos = i64::from_le_bytes(data[8..16].try_into().unwrap());
    if pos < 0 {
        return Err(bad(format!("{path}: negative pos in header")));
    }
    let pos = pos as usize;
    let tok_start = 72;
    let tok_end = tok_start + pos * 4;
    if data.len() < tok_end {
        return Err(bad(format!(
            "{path} is truncated: expected {} token bytes after the header, has {}",
            tok_end - tok_start,
            data.len().saturating_sub(tok_start)
        )));
    }
    let tokens = data[tok_start..tok_end].as_chunks::<4>().0.iter().map(|c| u32::from_le_bytes(*c)).collect();
    Ok((pos, tokens))
}

#[derive(Deserialize)]
pub struct ImportReq {
    #[serde(default)]
    path: Option<String>,
}

/// `POST /v1/state/import`, `path` variant: `{"prefix_hash","pos",
/// "restore_ms","runtime_differs"}`. Reads the file's own embedded tokens
/// (`read_state_header`) to build the request `chain.lookup` needs to find
/// the checkpoint `state_load` just brought in (`serve/engine.mojo:852-867`:
/// load happens, THEN the request's own prompt is looked up against the
/// chain by salted hash).
///
/// `Chain::lookup` (`serve/prefix.mojo:337`) only accepts a checkpoint at
/// `pos <= n - 1`, n the request's prompt length: it always reserves the
/// prompt's last token as the live decode seed (confirmed against
/// `checkpoints::create`/`fork` above, which always submit `pos + 1`
/// tokens -- the checkpoint's own defining prompt plus a suffix). The file
/// embeds exactly `pos` tokens, so a request built from them alone always
/// has `n == pos` and can never clear that bar -- a live smoke against this
/// route 502'd every time before this fix (room A, 2026-09-17). The request
/// sent to the engine repeats the file's own last token once, giving
/// `n == pos + 1`; `exact_prefix_hash` below and the checkpoint's own
/// registered hash both still only read `tokens[0:pos]`, so the repeat
/// changes nothing that is checked, it only clears the reservation.
pub async fn import(State(app): State<Shared>, Json(r): Json<ImportReq>) -> Result<Json<Value>, ApiError> {
    let Some(path) = r.path.clone() else {
        return Err(ApiError::Plain(
            StatusCode::NOT_IMPLEMENTED,
            "the LAT1 stream-as-body form (no \"path\") is not built yet; pass \"path\" for the file form".into(),
        ));
    };
    let (pos, tokens) = read_state_header(&path)?;
    if pos == 0 {
        return Err(bad(format!("{path}: pos is 0, nothing to import")));
    }
    let mut request_prompt = tokens.clone();
    request_prompt.push(tokens[pos - 1]);
    let g = Gen {
        prompt: request_prompt,
        n: 1,
        spec: false,
        stream: false,
        stop: vec![],
        ckpt: vec![],
        state: (None, Some(path.clone())),
        sample: protocol::SampleParams::default(),
        schema: None,
        reasoning: None,
        embed: None,
    };
    let (_req_id, rx) = check_and_submit(&app, &g)?;
    let (_acc, _text, stats) = collect(&app, rx).await.map_err(map_kvq_refusal)?;
    let restore_s = stats.get("restore_s").and_then(Value::as_f64).unwrap_or(0.0);
    let cached = stats.get("cached").and_then(Value::as_u64).unwrap_or(0) as usize;
    if cached < pos {
        return Err(ApiError::Plain(
            StatusCode::BAD_GATEWAY,
            format!("import ran but the chain lookup did not find the loaded state (cached={cached}, expected >={pos}): the file's salt or tokens likely do not match this pack"),
        ));
    }
    Ok(Json(json!({
        "prefix_hash": format!("{:016x}", exact_prefix_hash(&tokens, pos)),
        "pos": pos,
        "restore_ms": restore_s * 1000.0,
        "runtime_differs": Value::Null,
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

    fn ok<T>(r: Result<T, ApiError>) -> T {
        match r {
            Ok(v) => v,
            Err(_) => panic!("expected Ok"),
        }
    }

    /// A hand-built BAROST01 file, matching `serve/engine.mojo::save_state`
    /// byte for byte: magic, pos, CONV_SLOT, SSM_SLOT, kvn (i64 LE each),
    /// a 32-byte salt, then `pos` i32-LE tokens, then arbitrary payload
    /// bytes (never read by `read_state_header`).
    fn barost01_fixture(tokens: &[u32], extra_payload: &[u8]) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(b"BAROST01");
        buf.extend_from_slice(&(tokens.len() as i64).to_le_bytes()); // pos
        buf.extend_from_slice(&100i64.to_le_bytes()); // CONV_SLOT, arbitrary
        buf.extend_from_slice(&200i64.to_le_bytes()); // SSM_SLOT, arbitrary
        buf.extend_from_slice(&50i64.to_le_bytes()); // kvn, arbitrary
        buf.extend_from_slice(&[0u8; 32]); // salt, not read by this function
        for &t in tokens {
            buf.extend_from_slice(&t.to_le_bytes());
        }
        buf.extend_from_slice(extra_payload);
        buf
    }

    fn write_temp(name: &str, bytes: &[u8]) -> std::path::PathBuf {
        let path = std::env::temp_dir().join(format!("baro-state-test-{name}-{}", std::process::id()));
        std::fs::write(&path, bytes).unwrap();
        path
    }

    #[test]
    fn read_state_header_extracts_pos_and_tokens() {
        let path = write_temp("ok", &barost01_fixture(&[7, 8, 9], b"payload"));
        let (pos, tokens) = ok(read_state_header(path.to_str().unwrap()));
        assert_eq!(pos, 3);
        assert_eq!(tokens, vec![7, 8, 9]);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn read_state_header_accepts_the_int8_magic_too() {
        let mut bytes = barost01_fixture(&[1], b"");
        bytes[0..8].copy_from_slice(b"BAROST02");
        let path = write_temp("int8", &bytes);
        let (pos, tokens) = ok(read_state_header(path.to_str().unwrap()));
        assert_eq!(pos, 1);
        assert_eq!(tokens, vec![1]);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn read_state_header_rejects_a_bad_magic() {
        let mut bytes = barost01_fixture(&[1], b"");
        bytes[0..8].copy_from_slice(b"NOTASTAT");
        let path = write_temp("badmagic", &bytes);
        assert!(read_state_header(path.to_str().unwrap()).is_err());
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn read_state_header_rejects_a_truncated_token_array() {
        let mut bytes = barost01_fixture(&[1, 2, 3], b"");
        bytes.truncate(bytes.len() - 2); // cut into the last token's bytes
        let path = write_temp("truncated", &bytes);
        assert!(read_state_header(path.to_str().unwrap()).is_err());
        let _ = std::fs::remove_file(&path);
    }
}
