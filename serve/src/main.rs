//! baro-serve: OpenAI-compatible HTTP front for serve/engine.mojo.
//!
//!   baro-serve [--engine .work/engine] [--pack .work/engine-pack-q4]
//!              [--tokenizer <pack>/tokenizer.json] [--chat-template-file PATH]
//!              [--host 127.0.0.1] [--port 8080]
//!
//! Endpoints: GET /health, GET /v1/models, POST /v1/completions,
//! POST /v1/chat/completions (stream:true => SSE), POST /v1/fork
//! (branches from one shared prompt, B5), POST /tokenize, POST /detokenize.
//! The device runs one request at a time; the wire worker may admit later
//! request lines before the current request completes.

mod audio;
mod checkpoints;
mod embeddings;
mod engine;
mod fork_target;
mod ollama;
mod protocol;
mod state;
mod text;
mod web;

use std::convert::Infallible;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use axum::extract::{Request as AxumRequest, State};
use axum::http::StatusCode;
use axum::middleware::{self, Next};
use axum::response::sse::{Event as SseEvent, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use futures_core::Stream;
use serde::Deserialize;
use serde_json::{json, Value};
use tokio::sync::mpsc;
use tokio_stream::StreamExt;

use engine::{EnginePool, Event};
use text::{ChatMessage, Detok, Text};

struct App {
    engine: EnginePool,
    text: Option<Text>,
    model: String,
    /// Checkpoint API registry (LatentOS plan 10 sec 8).
    ckpts: checkpoints::Registry,
    identity: checkpoints::Identity,
    /// P3a speech-in sidecar (`serve/src/audio.rs`).
    audio: audio::AudioSidecar,
}

type Shared = Arc<App>;

/// Default generation length; GEN_N in serve/registry.mojo.
const DEFAULT_MAX_TOKENS: u32 = 64;

struct Opts {
    engine: PathBuf,
    pack: PathBuf,
    tokenizer: Option<PathBuf>,
    chat_template_file: Option<PathBuf>,
    host: String,
    port: u16,
    /// P3a (`docs/PLATFORM-PLAN.md`): no LLM engine spawned, no pack loaded,
    /// no VRAM held for it; `EnginePool::empty()` backs every route that
    /// would otherwise need one with its existing 503 error path.
    audio_only: bool,
    /// P4: per-process (pool-wide) environment seam so a device pin
    /// (`ROCR_VISIBLE_DEVICES`, `HSA_OVERRIDE_GFX_VERSION`) is an explicit,
    /// logged baro-serve argument rather than only whatever launched it.
    /// Applies to every engine this process's pool spawns, since one
    /// baro-serve process is pinned to one GPU; cross-device is two
    /// processes, each with its own `--engine-env`. `KEY=` (empty value)
    /// unsets KEY.
    engine_env: Vec<(String, String)>,
}

fn parse_opts() -> Result<Opts, String> {
    let mut o = Opts {
        engine: PathBuf::from(".work/engine"),
        pack: PathBuf::from(std::env::var("BARO_PACK").unwrap_or_else(|_| ".work/engine-pack-q4".into())),
        tokenizer: None,
        chat_template_file: None,
        host: "127.0.0.1".into(),
        port: 8080,
        audio_only: false,
        engine_env: Vec::new(),
    };
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        let mut val = |name: &str| args.next().ok_or(format!("{name} needs a value"));
        match a.as_str() {
            "--engine" => o.engine = PathBuf::from(val("--engine")?),
            "--pack" => o.pack = PathBuf::from(val("--pack")?),
            "--tokenizer" => o.tokenizer = Some(PathBuf::from(val("--tokenizer")?)),
            "--chat-template-file" => o.chat_template_file = Some(PathBuf::from(val("--chat-template-file")?)),
            "--host" => o.host = val("--host")?,
            "--port" => o.port = val("--port")?.parse().map_err(|e| format!("--port: {e}"))?,
            "--audio-only" => o.audio_only = true,
            "--engine-env" => {
                let kv = val("--engine-env")?;
                let (k, v) = kv.split_once('=').ok_or_else(|| format!("--engine-env {kv}: needs KEY=VALUE (KEY= to unset)"))?;
                if k.is_empty() {
                    return Err(format!("--engine-env {kv}: KEY must not be empty"));
                }
                o.engine_env.push((k.to_string(), v.to_string()));
            }
            "-h" | "--help" => {
                println!("usage: baro-serve [--engine PATH] [--pack DIR] [--tokenizer tokenizer.json] [--chat-template-file PATH] [--host H] [--port N] [--audio-only] [--engine-env KEY=VALUE]...");
                std::process::exit(0);
            }
            other => return Err(format!("unknown argument {other}")),
        }
    }
    Ok(o)
}

#[tokio::main]
async fn main() {
    let opts = match parse_opts() {
        Ok(o) => o,
        Err(e) => {
            eprintln!("baro-serve: {e}");
            std::process::exit(2);
        }
    };
    let tok_path = if opts.audio_only { PathBuf::new() } else { opts.tokenizer.clone().unwrap_or_else(|| opts.pack.join("tokenizer.json")) };
    let text = if opts.audio_only {
        eprintln!("audio-only: tokenizer not loaded, text endpoints disabled");
        None
    } else if tok_path.exists() {
        match Text::load(&tok_path, opts.chat_template_file.as_deref()) {
            Ok(t) => {
                eprintln!("tokenizer: {} (stop ids {:?})", tok_path.display(), t.stop_ids);
                Some(t)
            }
            Err(e) => {
                eprintln!("baro-serve: {e}");
                std::process::exit(2);
            }
        }
    } else {
        eprintln!("tokenizer: none ({} missing); text endpoints disabled", tok_path.display());
        None
    };
    let pool_size = std::env::var("BARO_POOL").ok().and_then(|s| s.parse::<usize>().ok()).unwrap_or(1).max(1);
    let engine = if opts.audio_only {
        eprintln!("audio-only: no LLM engine spawned, no pack loaded");
        EnginePool::empty()
    } else {
        match EnginePool::spawn(&opts.engine, &opts.pack, pool_size, &opts.engine_env).await {
            Ok(e) => e,
            Err(e) => {
                eprintln!("baro-serve: {e}");
                std::process::exit(1);
            }
        }
    };
    if !opts.engine_env.is_empty() {
        eprintln!("engine env overrides: {:?}", opts.engine_env);
    }
    eprintln!("engine pool ready: {} engine(s), limits {:?}", engine.pool_size(), engine.limits);
    let model = opts
        .pack
        .file_name()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "mojo-baro".into());
    let identity = if opts.audio_only {
        checkpoints::Identity { pack: "audio-only".into(), runtime: "audio-only".into(), tokenizer_sha: None, portable: false }
    } else {
        checkpoints::Identity::compute(&opts.pack, &tok_path)
    };
    let ckpts = checkpoints::Registry::from_env();
    eprintln!("checkpoints: dir {} cap {} identity {:?}", ckpts.dir.display(), ckpts.cap, identity);
    let audio = audio::AudioSidecar::from_env();
    let app = Arc::new(App { engine, text, model, ckpts, identity, audio });

    let router = Router::new()
        .route("/", get(web::index))
        .route("/web/{*path}", get(web::asset))
        .route("/health", get(health))
        .route("/v1/models", get(models))
        .route("/v1/completions", post(completions))
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/fork", post(fork))
        .route("/v1/audio/transcriptions", post(audio::transcriptions))
        .route("/v1/checkpoints", post(checkpoints::create).get(checkpoints::list))
        .route("/v1/checkpoints/{id}", get(checkpoints::get_one).delete(checkpoints::delete))
        .route("/v1/checkpoints/{id}/fork", post(checkpoints::fork))
        .route("/v1/cancel", post(cancel))
        .route("/tokenize", post(tokenize))
        .route("/detokenize", post(detokenize))
        // P0a: Ollama-compatible API (docs/PLATFORM-PLAN.md).
        .route("/api/tags", get(ollama::tags))
        .route("/api/ps", get(ollama::ps))
        .route("/api/version", get(ollama::version))
        .route("/api/show", post(ollama::show))
        .route("/api/pull", post(ollama::pull))
        .route("/api/chat", post(ollama::chat))
        .route("/api/generate", post(ollama::generate))
        .route("/api/embeddings", post(embeddings::embeddings))
        .route("/v1/embeddings", post(embeddings::embeddings))
        // P1: cross-node state API (docs/P1-STATE-API.md).
        .route("/v1/state", get(state::list))
        .route("/v1/state/export", post(state::export))
        .route("/v1/state/import", post(state::import))
        .with_state(app.clone())
        .layer(middleware::from_fn(access_log));

    let listener = match tokio::net::TcpListener::bind((opts.host.as_str(), opts.port)).await {
        Ok(l) => l,
        Err(e) => {
            eprintln!("baro-serve: bind {}:{}: {e}", opts.host, opts.port);
            std::process::exit(1);
        }
    };
    let addr = listener.local_addr().map(|a| a.to_string()).unwrap_or_default();
    // The one line on stdout: scripts read the bound port from it.
    println!("listening on http://{addr}");
    let serve = axum::serve(listener, router).with_graceful_shutdown(async {
        // A gate script's cleanup trap sends SIGTERM, not Ctrl+C's SIGINT;
        // without a handler for it the kernel's default disposition kills
        // this process immediately, skipping graceful shutdown entirely and
        // orphaning the audio sidecar child (found live, P3a timed gate).
        let mut term = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            .expect("install SIGTERM handler");
        tokio::select! {
            _ = tokio::signal::ctrl_c() => {}
            _ = term.recv() => {}
        }
        eprintln!("shutting down");
    });
    if let Err(e) = serve.await {
        eprintln!("baro-serve: {e}");
    }
    app.engine.shutdown().await;
    app.audio.shutdown().await;
}

// ---- errors -----------------------------------------------------------------

enum ApiError {
    Plain(StatusCode, String),
    Exceed { n_prompt_tokens: u64, n_ctx: u64 },
    /// Checkpoint API: the checkpoint's identity and this server's differ in `field`.
    Mismatch(String),
    /// P1 CONTRACT 2: a LAT1 import stream's `field` disagrees with this
    /// server's own identity (or magic/version/pos). The exact top-level
    /// shape `{"error":"state_identity","field","ours","theirs"}` is the
    /// contract's, not this repo's usual `{"error":{"message",...}}`
    /// envelope -- P0b's rank term and P4/P6 are written against it.
    StateIdentity { field: String, ours: String, theirs: String },
}

impl ApiError {
    /// llama.cpp's `exceed_context_size_error` shape, verbatim: harnesses
    /// (DeerFlow) branch on `type` and read `n_prompt_tokens` / `n_ctx`.
    fn exceed_context(n_prompt_tokens: u64, n_ctx: u64) -> ApiError {
        ApiError::Exceed { n_prompt_tokens, n_ctx }
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        match self {
            ApiError::Plain(code, msg) => {
                let body = json!({"error": {"message": msg, "type": "invalid_request_error", "code": code.as_u16()}});
                (code, Json(body)).into_response()
            }
            ApiError::Mismatch(field) => {
                let body = json!({"error": {"code": 409, "message": "IDENTITY_MISMATCH",
                    "type": "identity_mismatch", "field": field}});
                (StatusCode::CONFLICT, Json(body)).into_response()
            }
            ApiError::StateIdentity { field, ours, theirs } => {
                let body = json!({"error": "state_identity", "field": field, "ours": ours, "theirs": theirs});
                (StatusCode::CONFLICT, Json(body)).into_response()
            }
            ApiError::Exceed { n_prompt_tokens, n_ctx } => {
                let body = json!({"error": {
                    "code": 400,
                    "message": "the request exceeds the available context size, try increasing it",
                    "type": "exceed_context_size_error",
                    "n_prompt_tokens": n_prompt_tokens,
                    "n_ctx": n_ctx}});
                (StatusCode::BAD_REQUEST, Json(body)).into_response()
            }
        }
    }
}

fn bad(msg: impl Into<String>) -> ApiError {
    ApiError::Plain(StatusCode::BAD_REQUEST, msg.into())
}

fn need_text(app: &App) -> Result<&Text, ApiError> {
    app.text.as_ref().ok_or(ApiError::Plain(
        StatusCode::SERVICE_UNAVAILABLE,
        "no tokenizer.json loaded; pass token ids or start with --tokenizer".into(),
    ))
}

fn now() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
}

/// One stderr line per request: method, path, status, elapsed ms. Never the
/// body or headers (coordinator, gate-3 gap b: proves a request reached
/// `baro-serve` at all, e.g. one routed by PAIR's proxy).
async fn access_log(req: AxumRequest, next: Next) -> Response {
    let method = req.method().clone();
    let path = req.uri().path().to_string();
    let start = Instant::now();
    let resp = next.run(req).await;
    eprintln!("access: {method} {path} {} {}ms", resp.status().as_u16(), start.elapsed().as_millis());
    resp
}

// ---- health / models --------------------------------------------------------

async fn health(State(app): State<Shared>) -> Json<Value> {
    Json(json!({
        "status": if app.engine.alive() { "ok" } else { "engine_dead" },
        "queue": app.engine.queue_depth(),
        "pool": app.engine.queue_depths(),
        "tokenizer": app.text.is_some(),
        "limits": {"tmax": app.engine.limits.tmax, "mrows": app.engine.limits.mrows,
                   "kmax": app.engine.limits.kmax, "spec_k": app.engine.limits.spec_k},
        "pack": app.engine.limits.pack,
    }))
}

async fn models(State(app): State<Shared>) -> Json<Value> {
    Json(json!({"object": "list", "data": [{"id": app.model, "object": "model", "created": 0, "owned_by": "mojo-baro"}]}))
}

// ---- tokenize ---------------------------------------------------------------

#[derive(Deserialize)]
struct TokenizeReq {
    content: String,
    #[serde(default)]
    add_special: bool,
}

async fn tokenize(State(app): State<Shared>, Json(r): Json<TokenizeReq>) -> Result<Json<Value>, ApiError> {
    let t = need_text(&app)?;
    let ids = t.encode(&r.content, r.add_special).map_err(bad)?;
    Ok(Json(json!({"tokens": ids})))
}

#[derive(Deserialize)]
struct DetokenizeReq {
    tokens: Vec<u32>,
}

async fn detokenize(State(app): State<Shared>, Json(r): Json<DetokenizeReq>) -> Result<Json<Value>, ApiError> {
    let t = need_text(&app)?;
    let s = t.decode(&r.tokens).map_err(bad)?;
    Ok(Json(json!({"content": s})))
}

// ---- shared generation plumbing ----------------------------------------------

/// Everything both completion endpoints need once the prompt is token ids.
struct Gen {
    prompt: Vec<u32>,
    n: u32,
    spec: bool,
    stream: bool,
    /// Token-id sequences that end generation early (M2 control block): the
    /// tokenizer's own stop ids plus any caller-supplied `stop` strings,
    /// tokenized. Empty when there is no tokenizer.
    stop: Vec<Vec<u32>>,
    /// M1b role-boundary checkpoint hints (`Text::role_boundaries`); empty
    /// for `/v1/completions`, which has no message list.
    ckpt: Vec<u32>,
    /// Checkpoint API: (state_save, state_load) file paths for the engine.
    state: (Option<String>, Option<String>),
    /// C3 control block (parsed here, not yet acted on by the engine).
    sample: protocol::SampleParams,
    /// JSON-enforcement item 1: `response_format.json_schema.schema`,
    /// verbatim. `None` for every request that does not set
    /// `response_format` -- every endpoint but `/v1/chat/completions`
    /// today (spark's own 400 stays unconditional; see chat_completions).
    schema: Option<Value>,
    /// Item 2: `chat_template_kwargs.enable_thinking`, default true.
    reasoning: Option<bool>,
    /// P0a-e: ask the engine for the last-prompt-token embedding. `None` on
    /// every endpoint but `/api/embeddings` and `/v1/embeddings`.
    embed: Option<bool>,
}

/// OpenAI's `stop`: a single string or an array of strings.
#[derive(Deserialize)]
#[serde(untagged)]
enum StopParam {
    One(String),
    Many(Vec<String>),
}

impl StopParam {
    fn into_vec(self) -> Vec<String> {
        match self {
            StopParam::One(s) => vec![s],
            StopParam::Many(v) => v,
        }
    }
}

/// The tokenizer's own EOS-like ids (each a length-1 sequence, moving
/// today's client-side-only cut into the engine) plus the caller's `stop`
/// strings tokenized with no special tokens added. `[]` without a tokenizer.
fn compute_stop(app: &App, user_stop: Option<StopParam>) -> Vec<Vec<u32>> {
    let Some(t) = app.text.as_ref() else { return vec![] };
    let mut stop: Vec<Vec<u32>> = t.stop_ids.iter().map(|&id| vec![id]).collect();
    for s in user_stop.map(StopParam::into_vec).unwrap_or_default() {
        if let Ok(ids) = t.encode(&s, false) {
            if !ids.is_empty() {
                stop.push(ids);
            }
        }
    }
    stop
}

fn check_and_submit(app: &App, g: &Gen) -> Result<(u64, mpsc::UnboundedReceiver<Event>), ApiError> {
    let tmax = app.engine.limits.tmax;
    if g.prompt.is_empty() {
        return Err(bad("prompt is empty"));
    }
    if g.n == 0 {
        return Err(bad("max_tokens must be >= 1"));
    }
    if g.prompt.len() as u64 + g.n as u64 > tmax as u64 {
        return Err(ApiError::exceed_context(g.prompt.len() as u64, tmax as u64));
    }
    app.engine
        .submit(
            g.prompt.clone(), g.n, g.spec, g.stop.clone(), g.ckpt.clone(), g.sample.clone(), g.schema.clone(), g.reasoning,
            g.state.clone(), g.embed,
        )
        .map_err(|e| ApiError::Plain(StatusCode::SERVICE_UNAVAILABLE, e))
}

/// `POST /v1/cancel {"id": "cmpl-7"|"chatcmpl-7"}`: cancels the named
/// request if the engine is decoding it right now (M2 control block).
#[derive(Deserialize)]
struct CancelReq {
    id: String,
}

async fn cancel(State(app): State<Shared>, Json(r): Json<CancelReq>) -> Result<Json<Value>, ApiError> {
    let num = r.id.rsplit('-').next().unwrap_or("");
    let id: u64 = num.parse().map_err(|_| bad(format!("id {:?} has no trailing request number", r.id)))?;
    Ok(Json(json!({"cancelled": app.engine.cancel(id).await})))
}

/// One token's logprob data (item 4, briefs/2026-09-16-sampling-all-models-lane.md),
/// present only for tokens the engine actually sampled with `top_logprobs > 0`.
#[derive(Debug, Clone)]
struct LogprobEntry {
    token: u32,
    logprob: f64,
    top_logprobs: Vec<(u32, f64)>,
}

/// Streaming state shared by both SSE shapes.
struct Acc {
    detok: Detok,
    stopped: bool,
    tokens: Vec<u32>,
    logprobs: Vec<LogprobEntry>,
    hidden: Vec<Vec<f32>>,
    logits_topk: Vec<Vec<(u32, f64)>>,
}

impl Acc {
    fn new() -> Acc {
        Acc {
            detok: Detok::new(),
            stopped: false,
            tokens: Vec::new(),
            logprobs: Vec::new(),
            hidden: Vec::new(),
            logits_topk: Vec::new(),
        }
    }

    /// Text delta for one token, or None once a stop token was seen.
    fn take(&mut self, text: Option<&Text>, tok: u32, logprob: Option<f64>, top_logprobs: Vec<(u32, f64)>) -> Option<String> {
        self.tokens.push(tok);
        if let Some(lp) = logprob {
            self.logprobs.push(LogprobEntry { token: tok, logprob: lp, top_logprobs });
        }
        if self.stopped {
            return None;
        }
        match text {
            Some(t) => {
                if t.is_stop(tok) {
                    self.stopped = true;
                    return None;
                }
                Some(self.detok.push(t, tok))
            }
            None => Some(String::new()),
        }
    }

    /// The engine's own `finish` (M2 control block: "length"/"stop"/
    /// "cancelled") wins when present; an older engine that never sent one
    /// falls back to this client-side EOS check.
    fn finish_reason(&self, engine_finish: Option<&str>) -> String {
        match engine_finish {
            Some(f) => f.to_string(),
            None if self.stopped => "stop".to_string(),
            None => "length".to_string(),
        }
    }
}

fn stats_json(s: &protocol::DoneStats) -> Value {
    json!({"prefill_s": s.prefill_s, "decode_s": s.decode_s, "tok_s_gen": s.tok_s,
           "drafted": s.drafted, "accepted": s.accepted,
           "cached": s.cached, "prefill_rows": s.prefill_rows, "restore_s": s.restore_s,
           "finish": s.finish})
}

/// OpenAI `usage` plus the engine's prefix-checkpoint receipt (M1a):
/// `baro.cached_tokens` = prompt tokens restored from a checkpoint,
/// `baro.prefill_rows` = prompt rows actually replayed.
fn usage_json(n_prompt: usize, n_completion: usize, stats: &Value) -> Value {
    json!({"prompt_tokens": n_prompt, "completion_tokens": n_completion, "total_tokens": n_prompt + n_completion,
           "baro": {"cached_tokens": stats.get("cached").cloned().unwrap_or(Value::Null),
                    "prefill_rows": stats.get("prefill_rows").cloned().unwrap_or(Value::Null)}})
}

/// Collect a whole request (non-streaming).
async fn collect(app: &App, mut rx: mpsc::UnboundedReceiver<Event>) -> Result<(Acc, String, Value), ApiError> {
    let mut acc = Acc::new();
    let mut text_out = String::new();
    let mut stats = Value::Null;
    while let Some(ev) = rx.recv().await {
        match ev {
            Event::Tok { tok, logprob, top_logprobs } => {
                if let Some(d) = acc.take(app.text.as_ref(), tok, logprob, top_logprobs) {
                    text_out.push_str(&d);
                }
            }
            // P0a-e: no OpenAI/Ollama endpoint sets `embed`, so this never
            // fires on those paths; embeddings.rs collects it separately.
            Event::Embed(_) => {}
            Event::Hidden(vector) => acc.hidden.push(vector),
            Event::LogitsTopK(values) => acc.logits_topk.push(values),
            Event::Done(s) => {
                stats = stats_json(&s);
                break;
            }
            // serve/PROTOCOL.md: an {"id","error"} line means the request
            // was "rejected before any GPU work" -- true exactly when no Tok
            // arrived first, which is the client-error (400) case; an error
            // after generation started is a mid-request engine failure (502).
            Event::Error(e) if acc.tokens.is_empty() => return Err(ApiError::Plain(StatusCode::BAD_REQUEST, format!("engine: {e}"))),
            Event::Error(e) => return Err(ApiError::Plain(StatusCode::BAD_GATEWAY, format!("engine: {e}"))),
        }
    }
    Ok((acc, text_out, stats))
}

fn sse_stream(
    app: Shared,
    rx: mpsc::UnboundedReceiver<Event>,
    mut chunk: impl FnMut(&App, ChunkKind) -> Value + Send + 'static,
) -> Sse<impl Stream<Item = Result<SseEvent, Infallible>>> {
    let mut acc = Acc::new();
    let mut ended = false;
    let body = tokio_stream::wrappers::UnboundedReceiverStream::new(rx)
        .filter_map(move |ev| {
            if ended {
                return None;
            }
            let v = match ev {
                Event::Tok { tok, logprob, top_logprobs } => {
                    let lp = logprob.map(|l| (tok, l, top_logprobs.clone()));
                    let delta = acc.take(app.text.as_ref(), tok, logprob, top_logprobs)?;
                    chunk(&app, ChunkKind::Delta { text: delta, token: tok, logprob: lp })
                }
                Event::Embed(_) => return None,
                Event::Hidden(vector) => chunk(&app, ChunkKind::Hidden(vector)),
                Event::LogitsTopK(values) => chunk(&app, ChunkKind::LogitsTopK(values)),
                Event::Done(s) => {
                    ended = true;
                    let reason = acc.finish_reason(s.finish.as_deref());
                    chunk(&app, ChunkKind::Finish { reason, stats: stats_json(&s), tokens: acc.tokens.clone() })
                }
                Event::Error(e) => {
                    ended = true;
                    json!({"error": {"message": format!("engine: {e}"), "type": "engine_error"}})
                }
            };
            Some(Ok(SseEvent::default().data(v.to_string())))
        })
        .chain(tokio_stream::once(Ok(SseEvent::default().data("[DONE]"))));
    Sse::new(body).keep_alive(KeepAlive::default())
}

/// `(token, logprob, top_logprobs)` for one sampled token.
type TokLogprob = (u32, f64, Vec<(u32, f64)>);

enum ChunkKind {
    Delta { text: String, token: u32, logprob: Option<TokLogprob> },
    Hidden(Vec<f32>),
    LogitsTopK(Vec<(u32, f64)>),
    Finish { reason: String, stats: Value, tokens: Vec<u32> },
}

/// `{"id": token_id, "logprob": ...}` list, OpenAI-shaped per token.
fn top_logprobs_json(top: &[(u32, f64)]) -> Value {
    Value::Array(top.iter().map(|(id, lp)| json!({"id": id, "logprob": lp})).collect())
}

fn logits_topk_json(top: &[(u32, f64)]) -> Value {
    Value::Array(top.iter().map(|(id, logit)| json!({"id": id, "logit": logit})).collect())
}

fn add_latent_fields(out: &mut Value, acc: &Acc, hidden: bool, logits_topk: bool) {
    if hidden {
        out["hidden"] = json!(acc.hidden);
    }
    if logits_topk {
        out["logits_topk"] = Value::Array(acc.logits_topk.iter().map(|v| logits_topk_json(v)).collect());
    }
}

/// `/v1/completions`' `logprobs` object (OpenAI shape, ids not text since
/// this engine is token-id native): `null` when nothing was requested, so a
/// request with no `logprobs` field gets exactly the pre-item-4 response.
fn completion_logprobs_json(entries: &[LogprobEntry]) -> Value {
    if entries.is_empty() {
        return Value::Null;
    }
    json!({
        "tokens": entries.iter().map(|e| e.token).collect::<Vec<_>>(),
        "token_logprobs": entries.iter().map(|e| e.logprob).collect::<Vec<_>>(),
        "top_logprobs": entries.iter().map(|e| top_logprobs_json(&e.top_logprobs)).collect::<Vec<_>>(),
    })
}

/// `/v1/chat/completions`' `logprobs.content[]` (OpenAI shape, ids not
/// text). `null` when nothing was requested.
fn chat_logprobs_json(entries: &[LogprobEntry]) -> Value {
    if entries.is_empty() {
        return Value::Null;
    }
    json!({"content": entries.iter().map(|e| json!({
        "id": e.token, "logprob": e.logprob, "top_logprobs": top_logprobs_json(&e.top_logprobs),
    })).collect::<Vec<_>>()})
}

fn spec_default(app: &App, req_spec: Option<bool>) -> bool {
    req_spec.unwrap_or_else(|| std::env::var("BARO_SPEC").map(|v| v == "1").unwrap_or(false) && app.engine.limits.spec_k > 0)
}

// ---- C3 sampler fields (shared by both completion endpoints) ------------------

/// `temperature`/`top_p`/`top_k`/`min_p`/`seed`/`presence_penalty`/
/// `frequency_penalty`: parsed and carried in the request's control block
/// (C3), acted on by the engine at `temperature > 0` (items 3-4,
/// briefs/2026-09-16-sampling-all-models-lane.md). `logprobs`/`top_logprobs`
/// are not here: their shape differs per endpoint, see `to_sample_params`.
#[derive(Deserialize, Default)]
struct SamplerFields {
    #[serde(default)]
    temperature: Option<f32>,
    #[serde(default)]
    top_p: Option<f32>,
    #[serde(default)]
    top_k: Option<i32>,
    #[serde(default)]
    min_p: Option<f32>,
    #[serde(default)]
    seed: Option<u64>,
    #[serde(default)]
    presence_penalty: Option<f32>,
    #[serde(default)]
    frequency_penalty: Option<f32>,
}

impl SamplerFields {
    /// `logprobs` is not one of `SamplerFields`' own fields: `/v1/completions`
    /// spells it as a plain integer count and `/v1/chat/completions` as
    /// `logprobs: bool` + `top_logprobs: int`, incompatible shapes for the
    /// same JSON key, so each request struct parses its own and resolves it
    /// to the wire's single `top_logprobs: u32` (0 = off) before calling this.
    fn to_sample_params(&self, top_logprobs: Option<u32>) -> protocol::SampleParams {
        protocol::SampleParams {
            temperature: self.temperature,
            top_p: self.top_p,
            top_k: self.top_k,
            min_p: self.min_p,
            seed: self.seed,
            presence_penalty: self.presence_penalty,
            frequency_penalty: self.frequency_penalty,
            top_logprobs: top_logprobs.filter(|&n| n > 0),
            ..Default::default()
        }
    }
}

// ---- /v1/completions -----------------------------------------------------------

#[derive(Deserialize)]
struct CompletionReq {
    #[serde(default)]
    model: Option<String>,
    prompt: Value,
    #[serde(default)]
    max_tokens: Option<u32>,
    #[serde(default)]
    stream: bool,
    /// Extension: speculative (MTP) decode for this request; default BARO_SPEC.
    #[serde(default)]
    spec: Option<bool>,
    #[serde(default)]
    stop: Option<StopParam>,
    /// OpenAI: the count of top logprobs to return per token (0/absent = off).
    #[serde(default)]
    logprobs: Option<u32>,
    /// Baro extension: emit one post-final-norm hidden row per generated token.
    #[serde(default)]
    hidden: bool,
    /// Baro extension: emit raw pre-penalty logits for the top K ids.
    #[serde(default)]
    logits_topk: Option<u32>,
    #[serde(flatten)]
    sampler: SamplerFields,
}

fn prompt_ids(app: &App, prompt: &Value) -> Result<Vec<u32>, ApiError> {
    match prompt {
        Value::String(s) => need_text(app)?.encode(s, true).map_err(bad),
        Value::Array(items) if items.iter().all(Value::is_u64) => items
            .iter()
            .map(|v| u32::try_from(v.as_u64().unwrap_or(u64::MAX)).map_err(|_| bad("token id out of range")))
            .collect(),
        Value::Array(items) if items.len() == 1 && items[0].is_string() => {
            need_text(app)?.encode(items[0].as_str().unwrap_or(""), true).map_err(bad)
        }
        _ => Err(bad("prompt must be a string or an array of token ids")),
    }
}

fn latent_sample(mut sample: protocol::SampleParams, hidden: bool, logits_topk: Option<u32>) -> Result<protocol::SampleParams, ApiError> {
    if logits_topk.unwrap_or(0) > 20 {
        return Err(bad("logits_topk must be an integer from 0 to 20"));
    }
    sample.hidden = hidden.then_some(true);
    sample.logits_topk = logits_topk.filter(|&n| n > 0);
    Ok(sample)
}

async fn completions(State(app): State<Shared>, Json(r): Json<CompletionReq>) -> Result<Response, ApiError> {
    let g = Gen {
        prompt: prompt_ids(&app, &r.prompt)?,
        n: r.max_tokens.unwrap_or(DEFAULT_MAX_TOKENS),
        spec: spec_default(&app, r.spec),
        stream: r.stream,
        stop: compute_stop(&app, r.stop),
        ckpt: vec![],
        state: (None, None),
        sample: latent_sample(r.sampler.to_sample_params(r.logprobs), r.hidden, r.logits_topk)?,
        schema: None,
        reasoning: None,
        embed: None,
    };
    let model = r.model.unwrap_or_else(|| app.model.clone());
    let (req_id, rx) = check_and_submit(&app, &g)?;
    let id = format!("cmpl-{req_id}");
    let n_prompt = g.prompt.len();
    if g.stream {
        let created = now();
        let sse = sse_stream(app.clone(), rx, move |_, kind| match kind {
            ChunkKind::Delta { text, token, logprob } => json!({
                "id": id, "object": "text_completion", "created": created, "model": model,
                "choices": [{"index": 0, "text": text, "tokens": [token], "finish_reason": null,
                    "logprobs": logprob.map(|(_, lp, top)| json!({"tokens": [token], "token_logprobs": [lp], "top_logprobs": [top_logprobs_json(&top)]}))}]}),
            ChunkKind::Hidden(vector) => json!({
                "id": id, "object": "text_completion.chunk", "created": created, "model": model,
                "hidden": vector, "choices": []}),
            ChunkKind::LogitsTopK(values) => json!({
                "id": id, "object": "text_completion.chunk", "created": created, "model": model,
                "logits_topk": logits_topk_json(&values), "choices": []}),
            ChunkKind::Finish { reason, stats, tokens } => json!({
                "id": id, "object": "text_completion", "created": created, "model": model,
                "choices": [{"index": 0, "text": "", "finish_reason": reason}],
                "usage": usage_json(n_prompt, tokens.len(), &stats),
                "tokens": tokens, "timings": stats}),
        });
        return Ok(sse.into_response());
    }
    let (acc, text_out, stats) = collect(&app, rx).await?;
    let reason = acc.finish_reason(stats.get("finish").and_then(Value::as_str));
    let mut response = json!({
        "id": id, "object": "text_completion", "created": now(), "model": model,
        "choices": [{"index": 0, "text": text_out, "tokens": acc.tokens, "finish_reason": reason, "logprobs": completion_logprobs_json(&acc.logprobs)}],
        "usage": usage_json(n_prompt, acc.tokens.len(), &stats),
        "timings": stats,
    });
    add_latent_fields(&mut response, &acc, r.hidden, r.logits_topk.unwrap_or(0) > 0);
    Ok(Json(response).into_response())
}

// ---- /v1/fork -------------------------------------------------------------------

/// B5 (`briefs/2026-09-15-p2-b5-fork.md`, `bench/fork-protocol.md`): fork
/// one shared prompt into N branches, each a first-class request with its
/// own sampler state. No new engine plumbing -- `serve/prefix.mojo`'s
/// checkpoint chain already restores any repeated prompt automatically, on
/// every request, matched by a hash of the prompt tokens; sending the same
/// `prompt` to N branches here already gets branches 2..N served from a
/// restored checkpoint instead of a fresh prefill. This endpoint is a
/// convenience wrapper around that existing mechanism: branches run
/// sequentially (the engine already serializes one request at a time), so
/// branch 1 always prefills and saves the checkpoint that every later
/// branch restores.
#[derive(Deserialize)]
struct ForkReq {
    #[serde(default)]
    model: Option<String>,
    prompt: Value,
    branches: Vec<ForkBranch>,
    /// P1 item 2: a node address; the fork is exported here, imported there
    /// and answered from there (`fork_target.rs`).
    #[serde(default)]
    target: Option<String>,
}

#[derive(Deserialize)]
struct ForkBranch {
    #[serde(default)]
    max_tokens: Option<u32>,
    #[serde(default)]
    spec: Option<bool>,
    #[serde(default)]
    stop: Option<StopParam>,
    #[serde(flatten)]
    sampler: SamplerFields,
}

async fn fork(State(app): State<Shared>, Json(body): Json<Value>) -> Result<Response, ApiError> {
    let r: ForkReq = serde_json::from_value(body.clone()).map_err(|e| bad(format!("invalid fork request: {e}")))?;
    if r.branches.is_empty() {
        return Err(bad("branches must be non-empty"));
    }
    let prompt = prompt_ids(&app, &r.prompt)?;
    if let Some(target) = &r.target {
        return fork_target::fork_on_target(&app, target, prompt, body).await;
    }
    let model = r.model.unwrap_or_else(|| app.model.clone());
    let mut branches = Vec::with_capacity(r.branches.len());
    for (i, b) in r.branches.into_iter().enumerate() {
        let g = Gen {
            prompt: prompt.clone(),
            n: b.max_tokens.unwrap_or(DEFAULT_MAX_TOKENS),
            spec: spec_default(&app, b.spec),
            stream: false,
            stop: compute_stop(&app, b.stop),
            ckpt: vec![],
            state: (None, None),
            // logprobs not supported on /v1/fork branches yet (not in
            // serve/PROTOCOL.md's branch shape); item 4 scoped to the two
            // completion endpoints.
            sample: b.sampler.to_sample_params(None),
            schema: None,
            reasoning: None,
            embed: None,
        };
        let n_prompt = g.prompt.len();
        let (req_id, rx) = check_and_submit(&app, &g)?;
        let (acc, text_out, stats) = collect(&app, rx).await?;
        let reason = acc.finish_reason(stats.get("finish").and_then(Value::as_str));
        branches.push(json!({
            "index": i,
            "id": format!("fork-{req_id}"),
            "text": text_out,
            "tokens": acc.tokens,
            "finish_reason": reason,
            "usage": usage_json(n_prompt, acc.tokens.len(), &stats),
            "timings": stats,
        }));
    }
    Ok(Json(json!({
        "object": "fork", "model": model, "prompt_tokens": prompt.len(),
        "branches": branches,
    }))
    .into_response())
}

// ---- /v1/chat/completions ----------------------------------------------------

#[derive(Deserialize)]
struct ChatReq {
    #[serde(default)]
    model: Option<String>,
    messages: Vec<ChatMsgIn>,
    #[serde(default)]
    max_tokens: Option<u32>,
    #[serde(default)]
    max_completion_tokens: Option<u32>,
    #[serde(default)]
    stream: bool,
    #[serde(default)]
    spec: Option<bool>,
    #[serde(default)]
    stop: Option<StopParam>,
    /// A5: the OpenAI `tools` array, handed to the chat template verbatim
    /// (the template decides how to render function definitions, same as
    /// vLLM and llama.cpp).
    #[serde(default)]
    tools: Option<Value>,
    /// A5: extra top-level chat-template variables. Same field name as
    /// vLLM and llama.cpp; Qwen/Qwythos's own template reads
    /// `enable_thinking` from here.
    #[serde(default)]
    chat_template_kwargs: Option<Value>,
    /// A5: `{"type":"json_schema","json_schema":{...}}`. Parsed and
    /// validated, never silently ignored (bench/PROTOCOL-RULES.md P1) --
    /// see `check_response_format`.
    #[serde(default)]
    response_format: Option<Value>,
    /// OpenAI: whether to return logprobs at all.
    #[serde(default)]
    logprobs: Option<bool>,
    /// OpenAI: how many top logprobs per token (0-20); meaningful only with
    /// `logprobs: true`. `logprobs: true` alone still returns the chosen
    /// token's own logprob (resolved to `top_logprobs: 1` below).
    #[serde(default)]
    top_logprobs: Option<u32>,
    /// Baro extension: emit one post-final-norm hidden row per generated token.
    #[serde(default)]
    hidden: bool,
    /// Baro extension: emit raw pre-penalty logits for the top K ids.
    #[serde(default)]
    logits_topk: Option<u32>,
    #[serde(flatten)]
    sampler: SamplerFields,
}

#[derive(Deserialize)]
struct ChatMsgIn {
    role: String,
    #[serde(default)]
    content: Value,
    /// A5: an assistant turn's own prior tool calls, echoed back by the
    /// client on a follow-up request so the template can reconstruct
    /// multi-turn tool-use history.
    #[serde(default)]
    tool_calls: Option<Vec<ToolCallIn>>,
    /// A5: OpenAI's tool-result linkage id. Accepted and not silently
    /// dropped from the request shape, but unused: Qwythos's chat template
    /// keys a tool result off message order and `role == "tool"`, not this
    /// id (checked directly in the template text).
    #[serde(default)]
    #[allow(dead_code)]
    tool_call_id: Option<String>,
}

/// A5: one entry of an incoming `tool_calls` array (OpenAI shape:
/// `arguments` is a JSON-encoded string, not an object).
#[derive(Deserialize)]
struct ToolCallIn {
    function: ToolCallFunctionIn,
}

#[derive(Deserialize)]
struct ToolCallFunctionIn {
    name: String,
    #[serde(default)]
    arguments: Option<String>,
}

/// Converts an incoming OpenAI-shape `tool_calls` array (`arguments` as a
/// JSON string) into the shape Qwythos's chat template expects
/// (`tool_call.arguments|items`, i.e. a real object): parsed once here so
/// `text.rs` never has to know the wire format.
fn tool_calls_for_template(calls: &[ToolCallIn]) -> Value {
    Value::Array(
        calls
            .iter()
            .map(|c| {
                let args: Value = c
                    .function
                    .arguments
                    .as_deref()
                    .and_then(|s| serde_json::from_str(s).ok())
                    .unwrap_or_else(|| json!({}));
                json!({"function": {"name": c.function.name, "arguments": args}})
            })
            .collect(),
    )
}

fn content_text(v: &Value) -> Result<String, ApiError> {
    match v {
        Value::String(s) => Ok(s.clone()),
        Value::Array(parts) => Ok(parts
            .iter()
            .filter_map(|p| p.get("text").and_then(Value::as_str))
            .collect::<Vec<_>>()
            .join("")),
        Value::Null => Ok(String::new()),
        _ => Err(bad("message content must be a string or an array of text parts")),
    }
}

/// A5/JSON-enforcement item 1 (briefs/2026-09-16-json-enforcement-lane.md):
/// validates `response_format` shape and returns the schema object itself
/// (never silently ignored, bench/PROTOCOL-RULES.md P1). Enforcement is the
/// caller's job (`chat_completions`): the dense/MoE engine now takes the
/// schema per request and masks the device draw against it
/// (`serve/grammar_rt.mojo`, `kernels/sample.mojo` amar_sample_row_masked);
/// `serve/spark.mojo`'s engine has no such wiring, so the caller 400s
/// before ever reaching this function's Ok path for that family (item 4 --
/// gated on `app.engine.limits.kmax`, spark's ready line always reports 0).
fn check_response_format(rf: &Value) -> Result<Value, ApiError> {
    let obj = rf.as_object().ok_or_else(|| bad("response_format must be an object"))?;
    let kind = obj.get("type").and_then(Value::as_str).ok_or_else(|| bad("response_format.type is required"))?;
    if kind != "json_schema" {
        return Err(bad(format!("response_format.type {kind:?} is not supported (only \"json_schema\")")));
    }
    let schema_obj = obj
        .get("json_schema")
        .and_then(Value::as_object)
        .ok_or_else(|| bad("response_format.json_schema is required for type \"json_schema\""))?;
    let schema = schema_obj.get("schema").ok_or_else(|| bad("response_format.json_schema.schema is required"))?;
    Ok(schema.clone())
}

/// A5: Qwythos's own tool-call wire format (from its `tokenizer.chat_template`,
/// not an OpenAI convention):
///   <tool_call>
///   <function=NAME>
///   <parameter=P1>
///   V1
///   </parameter>
///   </function>
///   </tool_call>
/// one or more, back to back. Returns (content before the first tool_call,
/// parsed calls); content is the full trimmed text when there are none.
/// A parameter value that parses as JSON (number, bool, object, array)
/// keeps that type; anything else is kept as a plain string.
type ToolCall = (String, serde_json::Map<String, Value>);

fn parse_tool_calls(text: &str) -> (String, Vec<ToolCall>) {
    let mut calls = Vec::new();
    let mut rest = text;
    let mut first_start: Option<usize> = None;
    while let Some(start) = rest.find("<tool_call>") {
        if first_start.is_none() {
            first_start = Some(text.len() - rest.len() + start);
        }
        let after_open = &rest[start + "<tool_call>".len()..];
        let Some(end) = after_open.find("</tool_call>") else { break };
        let body = &after_open[..end];
        if let Some(call) = parse_one_tool_call(body) {
            calls.push(call);
        }
        rest = &after_open[end + "</tool_call>".len()..];
    }
    let content = match first_start {
        Some(i) => text[..i].trim().to_string(),
        None => text.trim().to_string(),
    };
    (content, calls)
}

fn parse_one_tool_call(body: &str) -> Option<ToolCall> {
    let body = body.trim().strip_prefix("<function=")?;
    let (name, after_name) = body.split_once('>')?;
    let inner = after_name.strip_suffix("</function>").unwrap_or(after_name);
    let mut args = serde_json::Map::new();
    let mut r = inner;
    while let Some(pstart) = r.find("<parameter=") {
        let after = &r[pstart + "<parameter=".len()..];
        let Some((pname, after_pname)) = after.split_once('>') else { break };
        let Some(pend) = after_pname.find("</parameter>") else { break };
        let pval = after_pname[..pend].trim();
        let value: Value = serde_json::from_str(pval).unwrap_or_else(|_| Value::String(pval.to_string()));
        args.insert(pname.to_string(), value);
        r = &after_pname[pend + "</parameter>".len()..];
    }
    Some((name.to_string(), args))
}

struct ToolDelta {
    name: Option<String>,
    arguments: String,
}

#[derive(Default)]
struct ToolStream {
    raw: String,
    content_sent: usize,
    marker: Option<usize>,
    function_end: Option<usize>,
    scan: usize,
    params: usize,
    name_sent: bool,
    args_open: bool,
    closed: bool,
}

impl ToolStream {
    fn has_tool(&self) -> bool {
        self.marker.is_some()
    }

    fn finish_content(&mut self) -> String {
        if self.marker.is_some() {
            return String::new();
        }
        let content = self.raw[self.content_sent..].to_string();
        self.content_sent = self.raw.len();
        content
    }

    fn push(&mut self, text: &str) -> (String, Option<ToolDelta>) {
        self.raw.push_str(text);
        const OPEN: &str = "<tool_call>";
        if self.marker.is_none() {
            if let Some(i) = self.raw.find(OPEN) {
                self.marker = Some(i);
                let content = self.raw[self.content_sent..i].to_string();
                self.content_sent = i;
                return (content, self.next_tool_delta());
            }
            let mut safe = self.raw.len().saturating_sub(OPEN.len() - 1);
            while safe > self.content_sent && !self.raw.is_char_boundary(safe) {
                safe -= 1;
            }
            let content = self.raw[self.content_sent..safe].to_string();
            self.content_sent = safe;
            return (content, None);
        }
        (String::new(), self.next_tool_delta())
    }

    fn next_tool_delta(&mut self) -> Option<ToolDelta> {
        if self.closed {
            return None;
        }
        let start = self.marker? + "<tool_call>".len();
        if self.function_end.is_none() {
            let i = self.raw[start..].find("<function=")? + start;
            let end = self.raw[i..].find('>')? + i;
            self.function_end = Some(end + 1);
        }
        let function_end = self.function_end?;
        if !self.name_sent {
            let name_start = self.marker? + "<tool_call><function=".len();
            let name = self.raw[name_start..function_end - 1].to_string();
            self.name_sent = true;
            self.scan = function_end;
            self.args_open = true;
            return Some(ToolDelta { name: Some(name), arguments: String::new() });
        }
        if let Some(i) = self.raw[self.scan..].find("<parameter=") {
            let i = self.scan + i;
            let name_end = self.raw[i..].find('>')? + i;
            let value_start = name_end + 1;
            let value_end = self.raw[value_start..].find("</parameter>")? + value_start;
            let name = &self.raw[i + "<parameter=".len()..name_end];
            let value = self.raw[value_start..value_end].trim();
            let parsed: Value = serde_json::from_str(value).unwrap_or_else(|_| Value::String(value.to_string()));
            self.scan = value_end + "</parameter>".len();
            let prefix = if self.params == 0 { "{" } else { "," };
            self.params += 1;
            let key = serde_json::to_string(name).unwrap_or_else(|_| "\"parameter\"".into());
            return Some(ToolDelta { name: None, arguments: format!("{prefix}{key}:{parsed}") });
        }
        if self.raw[self.scan..].contains("</function>") {
            self.closed = true;
            return Some(ToolDelta { name: None, arguments: if self.args_open { "}".into() } else { String::new() } });
        }
        None
    }
}

fn tool_delta_json(delta: ToolDelta) -> Value {
    let function = json!({"arguments": delta.arguments});
    let mut call = json!({"index": 0, "function": function});
    if let Some(name) = delta.name {
        call["id"] = json!("call_0");
        call["type"] = json!("function");
        call["function"]["name"] = json!(name);
    }
    Value::Array(vec![call])
}

/// `tool_calls` in the OpenAI response shape: `arguments` is a JSON-encoded
/// string (not an object -- that shape is only for what the template reads,
/// see `tool_calls_for_template`).
fn tool_calls_response_json(calls: &[ToolCall]) -> Value {
    Value::Array(
        calls
            .iter()
            .enumerate()
            .map(|(i, (name, args))| {
                json!({
                    "id": format!("call_{i}"),
                    "type": "function",
                    "function": {"name": name, "arguments": Value::Object(args.clone()).to_string()},
                })
            })
            .collect(),
    )
}

async fn chat_completions(State(app): State<Shared>, Json(r): Json<ChatReq>) -> Result<Response, ApiError> {
    let t = need_text(&app)?;
    if r.messages.is_empty() {
        return Err(bad("messages is empty"));
    }
    // JSON-enforcement item 4/5 (briefs/2026-09-16-json-enforcement-lane.md):
    // `serve/spark.mojo`'s engine has no grammar wiring, so it keeps the
    // 400 unconditionally; the dense/MoE engine (serve/engine.mojo) is
    // told apart by its `ready` line's kmax, which spark's always reports
    // as 0 (serve/PROTOCOL.md) and the dense/MoE engine never does.
    let schema = match &r.response_format {
        Some(rf) if app.engine.limits.kmax == 0 => {
            return Err(bad("response_format is not supported on this engine (no draft head / grammar wiring); dense and MoE only"));
        }
        Some(rf) => Some(check_response_format(rf)?),
        None => None,
    };
    // Item 2: reasoning is on unless the caller explicitly turned it off
    // the same way Qwythos's own chat template does
    // (`{%- if enable_thinking is defined and enable_thinking is false %}`).
    // Only meaningful when a schema is set; None otherwise so the wire
    // line omits it, matching every request before this lane.
    let reasoning = schema.as_ref().map(|_| {
        !matches!(
            r.chat_template_kwargs.as_ref().and_then(|k| k.get("enable_thinking")),
            Some(Value::Bool(false))
        )
    });
    let msgs = r
        .messages
        .iter()
        .map(|m| {
            Ok(ChatMessage {
                role: m.role.clone(),
                content: content_text(&m.content)?,
                tool_calls: m.tool_calls.as_deref().map(tool_calls_for_template),
            })
        })
        .collect::<Result<Vec<_>, ApiError>>()?;
    let rendered = t
        .apply_chat_template(&msgs, r.tools.as_ref(), r.chat_template_kwargs.as_ref())
        .map_err(|e| ApiError::Plain(StatusCode::INTERNAL_SERVER_ERROR, e))?;
    // A5: the rendered prompt is otherwise invisible to a caller (no
    // endpoint returns it directly); a debug print is the brief's own
    // suggested check for chat_template_kwargs actually changing the
    // render (briefs/2026-09-15-p2-a5-grammar-api.md item 1).
    eprintln!("chat_template render ({} chars): {rendered:?}", rendered.len());
    let ckpt = t.role_boundaries(&msgs);
    let g = Gen {
        prompt: t.encode(&rendered, true).map_err(bad)?,
        n: r.max_completion_tokens.or(r.max_tokens).unwrap_or(DEFAULT_MAX_TOKENS),
        spec: spec_default(&app, r.spec),
        stream: r.stream,
        stop: compute_stop(&app, r.stop),
        ckpt,
        state: (None, None),
        sample: latent_sample(r.sampler.to_sample_params(if r.logprobs == Some(true) { Some(r.top_logprobs.unwrap_or(1)) } else { r.top_logprobs }), r.hidden, r.logits_topk)?,
        schema,
        reasoning,
        embed: None,
    };
    let model = r.model.unwrap_or_else(|| app.model.clone());
    let (req_id, rx) = check_and_submit(&app, &g)?;
    let id = format!("chatcmpl-{req_id}");
    let n_prompt = g.prompt.len();
    if g.stream {
        let created = now();
        let mut first = true;
        let mut tool_stream = ToolStream::default();
        let sse = sse_stream(app.clone(), rx, move |_, kind| match kind {
            ChunkKind::Delta { text, token, logprob } => {
                let (content, tool) = tool_stream.push(&text);
                let mut delta = json!({"content": content});
                if first {
                    first = false;
                    delta["role"] = json!("assistant");
                }
                if let Some(tool) = tool {
                    delta["tool_calls"] = tool_delta_json(tool);
                    if delta["content"] == "" {
                        delta.as_object_mut().unwrap().remove("content");
                    }
                }
                let lp = logprob.map(|(_, l, top)| json!({"content": [{"id": token, "logprob": l, "top_logprobs": top_logprobs_json(&top)}]}));
                json!({"id": id, "object": "chat.completion.chunk", "created": created, "model": model,
                       "choices": [{"index": 0, "delta": delta, "tokens": [token], "finish_reason": null, "logprobs": lp}]})
            }
            ChunkKind::Hidden(vector) => json!({
                "id": id, "object": "chat.completion.chunk", "created": created, "model": model,
                "hidden": vector, "choices": []}),
            ChunkKind::LogitsTopK(values) => json!({
                "id": id, "object": "chat.completion.chunk", "created": created, "model": model,
                "logits_topk": logits_topk_json(&values), "choices": []}),
            ChunkKind::Finish { reason, stats, tokens } => {
                let mut delta = json!({});
                let tail = tool_stream.finish_content();
                if !tail.is_empty() {
                    delta["content"] = json!(tail);
                }
                json!({
                    "id": id, "object": "chat.completion.chunk", "created": created, "model": model,
                    "choices": [{"index": 0, "delta": delta, "finish_reason": if tool_stream.has_tool() { "tool_calls" } else { &reason }}],
                    "usage": usage_json(n_prompt, tokens.len(), &stats),
                    "timings": stats})
            }
        });
        return Ok(sse.into_response());
    }
    let (acc, text_out, stats) = collect(&app, rx).await?;
    let reason = acc.finish_reason(stats.get("finish").and_then(Value::as_str));
    // A5: tool_calls parsed out of the generated text for the non-streaming
    // response. The streaming path emits the same calls incrementally.
    let (content, calls) = parse_tool_calls(&text_out);
    let mut message = json!({"role": "assistant", "content": content});
    if !calls.is_empty() {
        message["tool_calls"] = tool_calls_response_json(&calls);
    }
    let mut response = json!({
        "id": id, "object": "chat.completion", "created": now(), "model": model,
        "choices": [{"index": 0, "message": message, "tokens": acc.tokens, "finish_reason": reason, "logprobs": chat_logprobs_json(&acc.logprobs)}],
        "usage": usage_json(n_prompt, acc.tokens.len(), &stats),
        "timings": stats,
    });
    add_latent_fields(&mut response, &acc, r.hidden, r.logits_topk.unwrap_or(0) > 0);
    Ok(Json(response).into_response())
}

#[cfg(test)]
mod a5_tests {
    use super::*;

    #[test]
    fn parses_one_tool_call_with_typed_and_string_parameters() {
        let text = "Let me check that.\n<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n<parameter=days>\n3\n</parameter>\n</function>\n</tool_call>";
        let (content, calls) = parse_tool_calls(text);
        assert_eq!(content, "Let me check that.");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].0, "get_weather");
        assert_eq!(calls[0].1.get("city"), Some(&json!("Paris")));
        assert_eq!(calls[0].1.get("days"), Some(&json!(3)));
    }

    #[test]
    fn parses_multiple_back_to_back_tool_calls() {
        let text = "<tool_call>\n<function=a>\n</function>\n</tool_call>\n<tool_call>\n<function=b>\n<parameter=x>\ny\n</parameter>\n</function>\n</tool_call>";
        let (content, calls) = parse_tool_calls(text);
        assert_eq!(content, "");
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].0, "a");
        assert_eq!(calls[1].0, "b");
        assert_eq!(calls[1].1.get("x"), Some(&json!("y")));
    }

    #[test]
    fn plain_text_has_no_tool_calls() {
        let (content, calls) = parse_tool_calls("just an answer, no calls here");
        assert_eq!(content, "just an answer, no calls here");
        assert!(calls.is_empty());
    }

    #[test]
    fn streams_tool_name_and_argument_fragments() {
        let mut stream = ToolStream::default();
        let (content, name) = stream.push("answer\n<tool_call><function=get_weather>");
        assert_eq!(content, "answer\n");
        let name = name.unwrap();
        assert_eq!(name.name.as_deref(), Some("get_weather"));
        assert_eq!(stream.push("<parameter=city>Paris</parameter>").1.unwrap().arguments, "{\"city\":\"Paris\"");
        assert_eq!(stream.push("</function></tool_call>").1.unwrap().arguments, "}");
        assert!(stream.has_tool());
    }

    #[test]
    fn plain_stream_flushes_marker_lookahead() {
        let mut stream = ToolStream::default();
        assert_eq!(stream.push("short").0, "");
        assert_eq!(stream.finish_content(), "short");
    }

    #[test]
    fn tool_calls_response_json_encodes_arguments_as_a_string() {
        let mut args = serde_json::Map::new();
        args.insert("city".into(), json!("Paris"));
        let v = tool_calls_response_json(&[("get_weather".to_string(), args)]);
        let arr = v.as_array().unwrap();
        assert_eq!(arr[0]["id"], json!("call_0"));
        assert_eq!(arr[0]["type"], json!("function"));
        assert_eq!(arr[0]["function"]["name"], json!("get_weather"));
        // arguments must be a JSON-encoded STRING, the OpenAI wire shape,
        // not the object shape the template itself reads.
        assert!(arr[0]["function"]["arguments"].is_string());
        let parsed: Value = serde_json::from_str(arr[0]["function"]["arguments"].as_str().unwrap()).unwrap();
        assert_eq!(parsed["city"], json!("Paris"));
    }

    #[test]
    fn tool_calls_for_template_parses_the_openai_string_back_to_an_object() {
        let calls = vec![ToolCallIn { function: ToolCallFunctionIn { name: "f".into(), arguments: Some(r#"{"x":1}"#.into()) } }];
        let v = tool_calls_for_template(&calls);
        assert_eq!(v[0]["function"]["name"], json!("f"));
        assert_eq!(v[0]["function"]["arguments"]["x"], json!(1));
    }

    #[test]
    fn response_format_rejects_unsupported_type() {
        let err = check_response_format(&json!({"type": "json_object"})).unwrap_err();
        match err {
            ApiError::Plain(code, msg) => {
                assert_eq!(code, StatusCode::BAD_REQUEST);
                assert!(msg.contains("json_object"), "{msg}");
            }
            _ => panic!("expected ApiError::Plain"),
        }
    }

    #[test]
    fn response_format_returns_the_schema_object_on_a_well_formed_shape() {
        match check_response_format(&json!({"type": "json_schema", "json_schema": {"name": "x", "schema": {"type": "object"}}})) {
            Ok(schema) => assert_eq!(schema, json!({"type": "object"})),
            Err(_) => panic!("expected Ok"),
        }
    }

    #[test]
    fn response_format_requires_the_schema_key() {
        let err = check_response_format(&json!({"type": "json_schema", "json_schema": {"name": "x"}})).unwrap_err();
        match err {
            ApiError::Plain(code, msg) => {
                assert_eq!(code, StatusCode::BAD_REQUEST);
                assert!(msg.contains("schema"), "{msg}");
            }
            _ => panic!("expected ApiError::Plain"),
        }
    }
}
