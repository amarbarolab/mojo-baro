//! Checkpoint API (LatentOS `docs/design/latent-os/10-checkpoint-api.md`,
//! sec 8): the engine's BAROST01 state file as a first-class object.
//!
//! `POST /v1/checkpoints` runs a prompt once and asks the engine to write its
//! full state (KV + conv + SSM + tokens) to a file under `BARO_CKPT_DIR`;
//! `POST /v1/checkpoints/{id}/fork` runs branches that load that file before
//! the prefix lookup, so only the suffix is prefilled. The payload never
//! crosses HTTP: the object is metadata, the file is the payload.
//!
//! Identity: `pack` is `<pack>/identity.json`'s `pack_sha256` (P1-STATE-API.md
//! CONTRACT 2, coordinator amendment 0c68cb4) when that file is present and
//! its recorded `pack_bytes`/`pack_mtime_ns` still match `pack.bin` on disk;
//! otherwise the hex sha256 of the pack PATH STRING, as before (`load_state`
//! refuses a mismatch here with "saved from a different pack"). `runtime` is
//! the engine repo's git HEAD; `tokenizer_sha` is sha256 of the tokenizer
//! file, the same algorithm `tools/engine-pack.py --identity` uses for
//! `identity.json`'s `tokenizer_sha256`, computed independently here (one
//! algorithm, not one shared computation) rather than trusted from the file.
//! `weights_uuid` stays null: CONTRACT 2 does not fill it from this struct.

use super::*;
use axum::extract::Path as UrlPath;
use std::collections::HashMap;
use std::path::Path;
use std::sync::Mutex;
use std::time::UNIX_EPOCH;

// ---- sha256, dependency-free -----------------------------------------------

const K: [u32; 64] = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be,
    0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa,
    0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85,
    0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3,
    0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f,
    0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
];

pub fn sha256(data: &[u8]) -> [u8; 32] {
    let mut h: [u32; 8] = [0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19];
    let mut msg = data.to_vec();
    let bit_len = (data.len() as u64).wrapping_mul(8);
    msg.push(0x80);
    while msg.len() % 64 != 56 {
        msg.push(0);
    }
    msg.extend_from_slice(&bit_len.to_be_bytes());
    for chunk in msg.chunks(64) {
        let mut w = [0u32; 64];
        for i in 0..16 {
            w[i] = u32::from_be_bytes([chunk[4 * i], chunk[4 * i + 1], chunk[4 * i + 2], chunk[4 * i + 3]]);
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16].wrapping_add(s0).wrapping_add(w[i - 7]).wrapping_add(s1);
        }
        let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut hh] = h;
        for i in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ (!e & g);
            let t1 = hh.wrapping_add(s1).wrapping_add(ch).wrapping_add(K[i]).wrapping_add(w[i]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let t2 = s0.wrapping_add(maj);
            hh = g;
            g = f;
            f = e;
            e = d.wrapping_add(t1);
            d = c;
            c = b;
            b = a;
            a = t1.wrapping_add(t2);
        }
        for (x, y) in h.iter_mut().zip([a, b, c, d, e, f, g, hh]) {
            *x = x.wrapping_add(y);
        }
    }
    let mut out = [0u8; 32];
    for (i, v) in h.iter().enumerate() {
        out[4 * i..4 * i + 4].copy_from_slice(&v.to_be_bytes());
    }
    out
}

fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

// ---- identity ---------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct Identity {
    pub pack: String,
    pub runtime: String,
    pub tokenizer_sha: Option<String>,
    /// True when `pack` came from a validated `identity.json` (CONTRACT 2's
    /// `pack_sha256`), false when it fell back to a hash of the pack PATH
    /// string. `GET /v1/state` reports the false case as `"portable":false`.
    pub portable: bool,
}

/// `<pack>/identity.json`'s shape (`tools/engine-pack.py`); only the fields
/// this reader needs. `tokenizer_sha256` and `source`/`general_uuid` are not
/// read here: this struct exists to validate and extract `pack_sha256`.
#[derive(Deserialize)]
struct PackIdentityFile {
    pack_sha256: String,
    pack_bytes: u64,
    pack_mtime_ns: u64,
}

/// `pack_sha256` from `<pack>/identity.json`, only when the file parses and
/// its `pack_bytes`/`pack_mtime_ns` still match `<pack>/pack.bin` right now
/// (a stat, not a re-hash of a file that can be many GB). Any miss returns
/// `None`, never an error: the caller's own path-hash fallback is not a
/// defect, it is CONTRACT 2's documented "identity.json absent" path.
fn read_pack_sha256(pack: &Path) -> Option<String> {
    let text = std::fs::read_to_string(pack.join("identity.json")).ok()?;
    let parsed: PackIdentityFile = serde_json::from_str(&text).ok()?;
    // A sha256 hex digest is always exactly 64 ASCII hex chars; reject
    // anything else here rather than let a hand-edited or truncated file
    // panic later at `Identity::weights_uuid_hex`'s `&self.pack[..32]`.
    if parsed.pack_sha256.len() != 64 || !parsed.pack_sha256.bytes().all(|b| b.is_ascii_hexdigit()) {
        return None;
    }
    let meta = std::fs::metadata(pack.join("pack.bin")).ok()?;
    let mtime_ns = meta.modified().ok()?.duration_since(UNIX_EPOCH).ok()?.as_nanos() as u64;
    if meta.len() != parsed.pack_bytes || mtime_ns != parsed.pack_mtime_ns {
        return None;
    }
    Some(parsed.pack_sha256)
}

impl Identity {
    pub fn compute(pack: &Path, tokenizer: &Path) -> Identity {
        // The engine salts its checkpoint hashes with sha256 of the pack
        // path STRING it was given (serve/prefix.mojo `Chain.__init__`), so
        // the same spelling is hashed here -- unchanged fallback for a pack
        // with no valid identity.json (CONTRACT 2).
        let pack_s = pack.to_string_lossy().into_owned();
        let runtime = std::env::var("BARO_RUNTIME").ok().unwrap_or_else(|| {
            std::process::Command::new("git")
                .args(["rev-parse", "--short", "HEAD"])
                .output()
                .ok()
                .filter(|o| o.status.success())
                .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
                .unwrap_or_else(|| "unknown".into())
        });
        let tokenizer_sha = std::fs::read(tokenizer).ok().map(|b| hex(&sha256(&b)));
        let recorded_pack_sha256 = read_pack_sha256(pack);
        let portable = recorded_pack_sha256.is_some();
        let pack_field = recorded_pack_sha256.unwrap_or_else(|| hex(&sha256(pack_s.as_bytes())));
        Identity { pack: pack_field, runtime, tokenizer_sha, portable }
    }

    /// CONTRACT 2 LAT1 mapping: `weights_uuid` (16 B) is `pack_sha256`'s
    /// first 16 bytes, as hex -- only meaningful when `portable` (a
    /// path-hash fallback is not a content identity, so it must not be
    /// dressed up as one).
    pub fn weights_uuid_hex(&self) -> Option<&str> {
        self.portable.then(|| &self.pack[..32])
    }

    fn json(&self) -> Value {
        json!({"pack": self.pack, "runtime": self.runtime, "tokenizer_sha": self.tokenizer_sha, "weights_uuid": Value::Null})
    }

    /// First field a client-supplied identity disagrees on, if any. Absent
    /// or null client fields are not compared.
    fn first_mismatch(&self, claimed: &Value) -> Option<String> {
        for (name, ours) in [("pack", Some(self.pack.as_str())), ("runtime", Some(self.runtime.as_str())), ("tokenizer_sha", self.tokenizer_sha.as_deref())] {
            if let Some(theirs) = claimed.get(name).and_then(Value::as_str) {
                if ours != Some(theirs) {
                    return Some(name.to_string());
                }
            }
        }
        None
    }
}

// ---- registry ---------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct Ckpt {
    pub id: String,
    pub pos: usize,
    pub tokens: Vec<u32>,
    pub path: PathBuf,
    pub bytes: u64,
    pub created: u64,
    pub ttl_s: u64,
    pub expires_at: u64,
}

impl Ckpt {
    fn json(&self, ident: &Identity) -> Value {
        json!({"id": self.id, "object": "checkpoint", "pos": self.pos, "bytes": self.bytes,
               "created": self.created, "ttl_s": self.ttl_s, "expires_at": self.expires_at,
               "identity": ident.json()})
    }
}

pub struct Registry {
    pub dir: PathBuf,
    pub cap: usize,
    items: Mutex<HashMap<String, Ckpt>>,
}

pub const DEFAULT_TTL_S: u64 = 300;

impl Registry {
    pub fn from_env() -> Registry {
        let dir = PathBuf::from(std::env::var("BARO_CKPT_DIR").unwrap_or_else(|_| ".work/checkpoints".into()));
        let cap = std::env::var("BARO_CKPT_API").ok().and_then(|s| s.parse().ok()).unwrap_or(8);
        Registry::new(dir, cap)
    }

    pub fn new(dir: PathBuf, cap: usize) -> Registry {
        Registry { dir, cap, items: Mutex::new(HashMap::new()) }
    }

    /// `lat:state@pos<pos>:<first 16 hex of sha256(tokens as little-endian u32)>`.
    pub fn id_for(tokens: &[u32]) -> String {
        let mut bytes = Vec::with_capacity(tokens.len() * 4);
        for t in tokens {
            bytes.extend_from_slice(&t.to_le_bytes());
        }
        let pos = tokens.len().saturating_sub(1);
        format!("lat:state@pos{pos}:{}", &hex(&sha256(&bytes))[..16])
    }

    pub fn path_for(&self, id: &str) -> PathBuf {
        self.dir.join(format!("{}.baro", id.replace([':', '@'], "_")))
    }

    /// Drop expired entries (and their files); returns the live count.
    fn sweep(&self, items: &mut HashMap<String, Ckpt>, now: u64) -> usize {
        let gone: Vec<String> = items.values().filter(|c| c.expires_at <= now).map(|c| c.id.clone()).collect();
        for id in gone {
            if let Some(c) = items.remove(&id) {
                let _ = std::fs::remove_file(&c.path);
            }
        }
        items.len()
    }

    pub fn live(&self, now: u64) -> Vec<Ckpt> {
        let mut items = self.items.lock().unwrap_or_else(|e| e.into_inner());
        self.sweep(&mut items, now);
        let mut v: Vec<Ckpt> = items.values().cloned().collect();
        v.sort_by_key(|c| c.created);
        v
    }

    pub fn get(&self, id: &str, now: u64) -> Option<Ckpt> {
        let mut items = self.items.lock().unwrap_or_else(|e| e.into_inner());
        self.sweep(&mut items, now);
        items.get(id).cloned()
    }

    /// Err(live count) when the cap is reached and `id` is not already held.
    pub fn insert(&self, c: Ckpt, now: u64) -> Result<(), usize> {
        let mut items = self.items.lock().unwrap_or_else(|e| e.into_inner());
        let n = self.sweep(&mut items, now);
        if n >= self.cap && !items.contains_key(&c.id) {
            return Err(n);
        }
        items.insert(c.id.clone(), c);
        Ok(())
    }

    pub fn remove(&self, id: &str) -> Option<Ckpt> {
        let mut items = self.items.lock().unwrap_or_else(|e| e.into_inner());
        let c = items.remove(id);
        if let Some(c) = &c {
            let _ = std::fs::remove_file(&c.path);
        }
        c
    }
}

// ---- routes -----------------------------------------------------------------

#[derive(Deserialize)]
pub struct CreateReq {
    #[serde(default)]
    model: Option<String>,
    prompt: Value,
    #[serde(default)]
    max_tokens: Option<u32>,
    #[serde(default)]
    ttl_s: Option<u64>,
    #[serde(default)]
    spec: Option<bool>,
    #[serde(default)]
    stop: Option<StopParam>,
    #[serde(flatten)]
    sampler: SamplerFields,
}

fn not_found(id: &str) -> ApiError {
    ApiError::Plain(StatusCode::NOT_FOUND, format!("no live checkpoint {id}"))
}

/// The engine refuses a state file saved from another pack; that is an
/// identity mismatch on `pack`, not an engine fault.
fn map_engine_err(e: ApiError) -> ApiError {
    match &e {
        ApiError::Plain(_, msg) if msg.contains("saved from a different pack") => ApiError::Mismatch("pack".into()),
        _ => e,
    }
}

pub async fn create(State(app): State<Shared>, Json(r): Json<CreateReq>) -> Result<Response, ApiError> {
    let tokens = prompt_ids(&app, &r.prompt)?;
    if tokens.len() < 2 {
        return Err(bad("prompt must hold at least 2 tokens (the checkpoint is taken at len-1)"));
    }
    let ttl_s = r.ttl_s.unwrap_or(DEFAULT_TTL_S).max(1);
    let id = Registry::id_for(&tokens);
    let t_now = now();
    if let Some(c) = app.ckpts.get(&id, t_now) {
        return Ok(Json(json!({"created": false, "checkpoint": c.json(&app.identity)})).into_response());
    }
    let live = app.ckpts.live(t_now).len();
    if live >= app.ckpts.cap {
        return Err(ApiError::Plain(StatusCode::INSUFFICIENT_STORAGE, format!("checkpoint cap {} reached ({live} live)", app.ckpts.cap)));
    }
    std::fs::create_dir_all(&app.ckpts.dir).map_err(|e| ApiError::Plain(StatusCode::INTERNAL_SERVER_ERROR, format!("{}: {e}", app.ckpts.dir.display())))?;
    let path = app.ckpts.path_for(&id);
    let abs = std::fs::canonicalize(&app.ckpts.dir).unwrap_or_else(|_| app.ckpts.dir.clone()).join(path.file_name().unwrap_or_default());
    let model = r.model.unwrap_or_else(|| app.model.clone());
    let g = Gen {
        prompt: tokens.clone(),
        n: r.max_tokens.unwrap_or(1),
        spec: spec_default(&app, r.spec),
        stream: false,
        stop: compute_stop(&app, r.stop),
        ckpt: vec![],
        state: (Some(abs.to_string_lossy().into_owned()), None),
        sample: r.sampler.to_sample_params(None),
        schema: None,
        reasoning: None,
        embed: None,
    };
    let n_prompt = g.prompt.len();
    let (req_id, rx) = check_and_submit(&app, &g)?;
    let (acc, text_out, stats) = collect(&app, rx).await?;
    let bytes = std::fs::metadata(&abs).map(|m| m.len()).map_err(|e| {
        ApiError::Plain(StatusCode::BAD_GATEWAY, format!("engine finished but wrote no state file at {}: {e}", abs.display()))
    })?;
    let c = Ckpt { id: id.clone(), pos: tokens.len() - 1, tokens, path: abs, bytes, created: t_now, ttl_s, expires_at: t_now + ttl_s };
    let cj = c.json(&app.identity);
    if let Err(n) = app.ckpts.insert(c, t_now) {
        let _ = std::fs::remove_file(app.ckpts.path_for(&id));
        return Err(ApiError::Plain(StatusCode::INSUFFICIENT_STORAGE, format!("checkpoint cap {} reached ({n} live)", app.ckpts.cap)));
    }
    let reason = acc.finish_reason(stats.get("finish").and_then(Value::as_str));
    Ok(Json(json!({
        "created": true, "checkpoint": cj,
        "completion": {"id": format!("cmpl-{req_id}"), "model": model, "text": text_out, "tokens": acc.tokens,
                       "finish_reason": reason, "usage": usage_json(n_prompt, acc.tokens.len(), &stats), "timings": stats},
    }))
    .into_response())
}

pub async fn list(State(app): State<Shared>) -> Json<Value> {
    let items: Vec<Value> = app.ckpts.live(now()).iter().map(|c| c.json(&app.identity)).collect();
    Json(json!({"object": "list", "data": items, "cap": app.ckpts.cap, "dir": app.ckpts.dir}))
}

pub async fn get_one(State(app): State<Shared>, UrlPath(id): UrlPath<String>) -> Result<Json<Value>, ApiError> {
    let c = app.ckpts.get(&id, now()).ok_or_else(|| not_found(&id))?;
    Ok(Json(c.json(&app.identity)))
}

pub async fn delete(State(app): State<Shared>, UrlPath(id): UrlPath<String>) -> Json<Value> {
    Json(json!({"id": id, "deleted": app.ckpts.remove(&id).is_some()}))
}

#[derive(Deserialize)]
pub struct CkptForkReq {
    #[serde(default)]
    model: Option<String>,
    branches: Vec<CkptForkBranch>,
    /// Optional identity the client believes the checkpoint has; any field
    /// that disagrees with this server's is refused with 409 before the
    /// engine is touched.
    #[serde(default)]
    identity: Option<Value>,
}

#[derive(Deserialize)]
pub struct CkptForkBranch {
    /// Text or token ids appended after the checkpointed prompt.
    #[serde(default)]
    prompt_suffix: Option<Value>,
    #[serde(default)]
    max_tokens: Option<u32>,
    #[serde(default)]
    spec: Option<bool>,
    #[serde(default)]
    stop: Option<StopParam>,
    #[serde(flatten)]
    sampler: SamplerFields,
}

pub async fn fork(State(app): State<Shared>, UrlPath(id): UrlPath<String>, Json(r): Json<CkptForkReq>) -> Result<Response, ApiError> {
    if r.branches.is_empty() {
        return Err(bad("branches must be non-empty"));
    }
    let c = app.ckpts.get(&id, now()).ok_or_else(|| not_found(&id))?;
    if let Some(field) = r.identity.as_ref().and_then(|v| app.identity.first_mismatch(v)) {
        return Err(ApiError::Mismatch(field));
    }
    let model = r.model.unwrap_or_else(|| app.model.clone());
    let mut branches = Vec::with_capacity(r.branches.len());
    for (i, b) in r.branches.into_iter().enumerate() {
        let mut prompt = c.tokens.clone();
        if let Some(sfx) = &b.prompt_suffix {
            let extra = match sfx {
                // A suffix continues the prompt, so no BOS / special prefix.
                Value::String(s) => need_text(&app)?.encode(s, false).map_err(bad)?,
                other => prompt_ids(&app, other)?,
            };
            prompt.extend(extra);
        }
        let g = Gen {
            prompt,
            n: b.max_tokens.unwrap_or(DEFAULT_MAX_TOKENS),
            spec: spec_default(&app, b.spec),
            stream: false,
            stop: compute_stop(&app, b.stop),
            ckpt: vec![],
            state: (None, Some(c.path.to_string_lossy().into_owned())),
            sample: b.sampler.to_sample_params(None),
            schema: None,
            reasoning: None,
            embed: None,
        };
        let n_prompt = g.prompt.len();
        let (req_id, rx) = check_and_submit(&app, &g)?;
        let (acc, text_out, stats) = collect(&app, rx).await.map_err(map_engine_err)?;
        let cached = stats.get("cached").and_then(Value::as_u64).unwrap_or(0) as usize;
        let reason = acc.finish_reason(stats.get("finish").and_then(Value::as_str));
        branches.push(json!({
            "index": i, "id": format!("fork-{req_id}"), "text": text_out, "tokens": acc.tokens,
            "finish_reason": reason, "restored": cached >= c.pos, "cached_tokens": cached,
            "usage": usage_json(n_prompt, acc.tokens.len(), &stats), "timings": stats,
        }));
    }
    Ok(Json(json!({"object": "checkpoint.fork", "model": model, "checkpoint": id, "pos": c.pos, "branches": branches})).into_response())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sha256_matches_known_vectors() {
        assert_eq!(hex(&sha256(b"")), "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
        assert_eq!(hex(&sha256(b"abc")), "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
    }

    fn write_pack_fixture(dir: &Path, pack_bytes: &[u8]) -> u64 {
        std::fs::write(dir.join("pack.bin"), pack_bytes).unwrap();
        let meta = std::fs::metadata(dir.join("pack.bin")).unwrap();
        meta.modified().unwrap().duration_since(UNIX_EPOCH).unwrap().as_nanos() as u64
    }

    // sha256(b"hello world"), verified with Python's hashlib, not guessed
    // (the same lesson as state.rs's fabricated-then-corrected test vector):
    //   python3 -c "import hashlib; print(hashlib.sha256(b'hello world').hexdigest())"
    const HELLO_WORLD_SHA256: &str = "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9";

    #[test]
    fn read_pack_sha256_returns_the_recorded_hash_when_stat_matches() {
        let dir = std::env::temp_dir().join(format!("baro-identity-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let mtime_ns = write_pack_fixture(&dir, b"hello world");
        std::fs::write(
            dir.join("identity.json"),
            format!(r#"{{"pack_sha256":"{HELLO_WORLD_SHA256}","pack_bytes":11,"pack_mtime_ns":{mtime_ns}}}"#),
        )
        .unwrap();
        assert_eq!(read_pack_sha256(&dir), Some(HELLO_WORLD_SHA256.to_string()));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn read_pack_sha256_is_none_on_a_size_mismatch() {
        let dir = std::env::temp_dir().join(format!("baro-identity-test-size-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let mtime_ns = write_pack_fixture(&dir, b"hello world");
        std::fs::write(
            dir.join("identity.json"),
            format!(r#"{{"pack_sha256":"{HELLO_WORLD_SHA256}","pack_bytes":999,"pack_mtime_ns":{mtime_ns}}}"#),
        )
        .unwrap();
        assert_eq!(read_pack_sha256(&dir), None);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn read_pack_sha256_is_none_when_identity_json_is_absent() {
        let dir = std::env::temp_dir().join(format!("baro-identity-test-absent-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        write_pack_fixture(&dir, b"hello world");
        assert_eq!(read_pack_sha256(&dir), None);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn read_pack_sha256_rejects_a_malformed_hash_instead_of_panicking_later() {
        let dir = std::env::temp_dir().join(format!("baro-identity-test-malformed-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let mtime_ns = write_pack_fixture(&dir, b"hello world");
        // Too short: weights_uuid_hex's &pack[..32] would panic on this if
        // it were ever accepted.
        std::fs::write(
            dir.join("identity.json"),
            format!(r#"{{"pack_sha256":"short","pack_bytes":11,"pack_mtime_ns":{mtime_ns}}}"#),
        )
        .unwrap();
        assert_eq!(read_pack_sha256(&dir), None);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn weights_uuid_hex_is_none_when_not_portable() {
        let me = Identity { pack: "p".into(), runtime: "r".into(), tokenizer_sha: None, portable: false };
        assert_eq!(me.weights_uuid_hex(), None);
    }

    #[test]
    fn weights_uuid_hex_is_the_first_32_hex_chars_when_portable() {
        let first_half = "a".repeat(32);
        let pack = first_half.clone() + &"b".repeat(32);
        let me = Identity { pack, runtime: "r".into(), tokenizer_sha: None, portable: true };
        assert_eq!(me.weights_uuid_hex(), Some(first_half.as_str()));
    }

    #[test]
    fn id_is_stable_and_position_bound() {
        let a = Registry::id_for(&[1, 2, 3]);
        assert_eq!(a, Registry::id_for(&[1, 2, 3]));
        assert!(a.starts_with("lat:state@pos2:"));
        assert_ne!(a, Registry::id_for(&[1, 2, 4]));
    }

    #[test]
    fn registry_expires_and_caps() {
        let dir = std::env::temp_dir().join(format!("baro-ckpt-test-{}", std::process::id()));
        let reg = Registry::new(dir.clone(), 1);
        let mk = |id: &str, exp: u64| Ckpt { id: id.into(), pos: 1, tokens: vec![1, 2], path: reg.path_for(id), bytes: 0, created: 0, ttl_s: 1, expires_at: exp };
        assert!(reg.insert(mk("a", 10), 5).is_ok());
        assert_eq!(reg.insert(mk("b", 10), 5), Err(1));
        assert!(reg.get("a", 5).is_some());
        assert!(reg.get("a", 10).is_none(), "expired at its expires_at");
        assert!(reg.insert(mk("b", 20), 11).is_ok(), "cap frees after expiry");
        assert_eq!(reg.live(11).len(), 1);
    }

    #[test]
    fn identity_mismatch_names_the_first_field() {
        let me = Identity { pack: "p".into(), runtime: "r".into(), tokenizer_sha: Some("t".into()), portable: false };
        assert_eq!(me.first_mismatch(&json!({"pack": "p", "runtime": "x"})), Some("runtime".into()));
        assert_eq!(me.first_mismatch(&json!({"tokenizer_sha": null})), None);
        assert_eq!(me.first_mismatch(&json!({})), None);
    }
}
