//! baro-serve: OpenAI-compatible HTTP front for serve/engine.mojo.
//!
//!   baro-serve [--engine .work/engine] [--pack .work/engine-pack-q4]
//!              [--tokenizer <pack>/tokenizer.json] [--host 127.0.0.1] [--port 8080]
//!
//! Endpoints: GET /health, GET /v1/models, POST /v1/completions,
//! POST /v1/chat/completions (stream:true => SSE), POST /tokenize,
//! POST /detokenize. One request runs at a time; the rest queue.

mod engine;
mod protocol;
mod text;

use std::convert::Infallible;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::sse::{Event as SseEvent, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use futures_core::Stream;
use serde::Deserialize;
use serde_json::{json, Value};
use tokio::sync::mpsc;
use tokio_stream::StreamExt;

use engine::{Engine, Event};
use text::{ChatMessage, Detok, Text};

struct App {
    engine: Engine,
    text: Option<Text>,
    model: String,
}

type Shared = Arc<App>;

/// Default generation length; GEN_N in serve/registry.mojo.
const DEFAULT_MAX_TOKENS: u32 = 64;

struct Opts {
    engine: PathBuf,
    pack: PathBuf,
    tokenizer: Option<PathBuf>,
    host: String,
    port: u16,
}

fn parse_opts() -> Result<Opts, String> {
    let mut o = Opts {
        engine: PathBuf::from(".work/engine"),
        pack: PathBuf::from(std::env::var("BARO_PACK").unwrap_or_else(|_| ".work/engine-pack-q4".into())),
        tokenizer: None,
        host: "127.0.0.1".into(),
        port: 8080,
    };
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        let mut val = |name: &str| args.next().ok_or(format!("{name} needs a value"));
        match a.as_str() {
            "--engine" => o.engine = PathBuf::from(val("--engine")?),
            "--pack" => o.pack = PathBuf::from(val("--pack")?),
            "--tokenizer" => o.tokenizer = Some(PathBuf::from(val("--tokenizer")?)),
            "--host" => o.host = val("--host")?,
            "--port" => o.port = val("--port")?.parse().map_err(|e| format!("--port: {e}"))?,
            "-h" | "--help" => {
                println!("usage: baro-serve [--engine PATH] [--pack DIR] [--tokenizer tokenizer.json] [--host H] [--port N]");
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
    let tok_path = opts.tokenizer.clone().unwrap_or_else(|| opts.pack.join("tokenizer.json"));
    let text = if tok_path.exists() {
        match Text::load(&tok_path) {
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
    let engine = match Engine::spawn(&opts.engine, &opts.pack).await {
        Ok(e) => e,
        Err(e) => {
            eprintln!("baro-serve: {e}");
            std::process::exit(1);
        }
    };
    eprintln!("engine ready: {:?}", engine.limits);
    let model = opts
        .pack
        .file_name()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "mojo-baro".into());
    let app = Arc::new(App { engine, text, model });

    let router = Router::new()
        .route("/health", get(health))
        .route("/v1/models", get(models))
        .route("/v1/completions", post(completions))
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/cancel", post(cancel))
        .route("/tokenize", post(tokenize))
        .route("/detokenize", post(detokenize))
        .with_state(app.clone());

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
        let _ = tokio::signal::ctrl_c().await;
        eprintln!("shutting down");
    });
    if let Err(e) = serve.await {
        eprintln!("baro-serve: {e}");
    }
    app.engine.shutdown().await;
}

// ---- errors -----------------------------------------------------------------

enum ApiError {
    Plain(StatusCode, String),
    Exceed { n_prompt_tokens: u64, n_ctx: u64 },
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

// ---- health / models --------------------------------------------------------

async fn health(State(app): State<Shared>) -> Json<Value> {
    Json(json!({
        "status": if app.engine.alive() { "ok" } else { "engine_dead" },
        "queue": app.engine.queue_depth(),
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
        .submit(g.prompt.clone(), g.n, g.spec, g.stop.clone())
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

/// Streaming state shared by both SSE shapes.
struct Acc {
    detok: Detok,
    stopped: bool,
    tokens: Vec<u32>,
}

impl Acc {
    fn new() -> Acc {
        Acc {
            detok: Detok::new(),
            stopped: false,
            tokens: Vec::new(),
        }
    }

    /// Text delta for one token, or None once a stop token was seen.
    fn take(&mut self, text: Option<&Text>, tok: u32) -> Option<String> {
        self.tokens.push(tok);
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
            Event::Tok(t) => {
                if let Some(d) = acc.take(app.text.as_ref(), t) {
                    text_out.push_str(&d);
                }
            }
            Event::Done(s) => {
                stats = stats_json(&s);
                break;
            }
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
                Event::Tok(t) => {
                    let delta = acc.take(app.text.as_ref(), t)?;
                    chunk(&app, ChunkKind::Delta { text: delta, token: t })
                }
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

enum ChunkKind {
    Delta { text: String, token: u32 },
    Finish { reason: String, stats: Value, tokens: Vec<u32> },
}

fn spec_default(app: &App, req_spec: Option<bool>) -> bool {
    req_spec.unwrap_or_else(|| std::env::var("BARO_SPEC").map(|v| v == "1").unwrap_or(false) && app.engine.limits.spec_k > 0)
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

async fn completions(State(app): State<Shared>, Json(r): Json<CompletionReq>) -> Result<Response, ApiError> {
    let g = Gen {
        prompt: prompt_ids(&app, &r.prompt)?,
        n: r.max_tokens.unwrap_or(DEFAULT_MAX_TOKENS),
        spec: spec_default(&app, r.spec),
        stream: r.stream,
        stop: compute_stop(&app, r.stop),
    };
    let model = r.model.unwrap_or_else(|| app.model.clone());
    let (req_id, rx) = check_and_submit(&app, &g)?;
    let id = format!("cmpl-{req_id}");
    let n_prompt = g.prompt.len();
    if g.stream {
        let created = now();
        let sse = sse_stream(app.clone(), rx, move |_, kind| match kind {
            ChunkKind::Delta { text, token } => json!({
                "id": id, "object": "text_completion", "created": created, "model": model,
                "choices": [{"index": 0, "text": text, "tokens": [token], "finish_reason": null}]}),
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
    Ok(Json(json!({
        "id": id, "object": "text_completion", "created": now(), "model": model,
        "choices": [{"index": 0, "text": text_out, "tokens": acc.tokens, "finish_reason": reason, "logprobs": null}],
        "usage": usage_json(n_prompt, acc.tokens.len(), &stats),
        "timings": stats,
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
}

#[derive(Deserialize)]
struct ChatMsgIn {
    role: String,
    content: Value,
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

async fn chat_completions(State(app): State<Shared>, Json(r): Json<ChatReq>) -> Result<Response, ApiError> {
    let t = need_text(&app)?;
    if r.messages.is_empty() {
        return Err(bad("messages is empty"));
    }
    let msgs = r
        .messages
        .iter()
        .map(|m| Ok(ChatMessage { role: m.role.clone(), content: content_text(&m.content)? }))
        .collect::<Result<Vec<_>, ApiError>>()?;
    let rendered = t.apply_chat_template(&msgs).map_err(|e| ApiError::Plain(StatusCode::INTERNAL_SERVER_ERROR, e))?;
    let g = Gen {
        prompt: t.encode(&rendered, true).map_err(bad)?,
        n: r.max_completion_tokens.or(r.max_tokens).unwrap_or(DEFAULT_MAX_TOKENS),
        spec: spec_default(&app, r.spec),
        stream: r.stream,
        stop: compute_stop(&app, r.stop),
    };
    let model = r.model.unwrap_or_else(|| app.model.clone());
    let (req_id, rx) = check_and_submit(&app, &g)?;
    let id = format!("chatcmpl-{req_id}");
    let n_prompt = g.prompt.len();
    if g.stream {
        let created = now();
        let mut first = true;
        let sse = sse_stream(app.clone(), rx, move |_, kind| match kind {
            ChunkKind::Delta { text, token } => {
                let mut delta = json!({"content": text});
                if first {
                    first = false;
                    delta["role"] = json!("assistant");
                }
                json!({"id": id, "object": "chat.completion.chunk", "created": created, "model": model,
                       "choices": [{"index": 0, "delta": delta, "tokens": [token], "finish_reason": null}]})
            }
            ChunkKind::Finish { reason, stats, tokens } => json!({
                "id": id, "object": "chat.completion.chunk", "created": created, "model": model,
                "choices": [{"index": 0, "delta": {}, "finish_reason": reason}],
                "usage": usage_json(n_prompt, tokens.len(), &stats),
                "timings": stats}),
        });
        return Ok(sse.into_response());
    }
    let (acc, text_out, stats) = collect(&app, rx).await?;
    let reason = acc.finish_reason(stats.get("finish").and_then(Value::as_str));
    Ok(Json(json!({
        "id": id, "object": "chat.completion", "created": now(), "model": model,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": text_out}, "tokens": acc.tokens, "finish_reason": reason}],
        "usage": usage_json(n_prompt, acc.tokens.len(), &stats),
        "timings": stats,
    }))
    .into_response())
}
