//! `/api/embeddings` and `/v1/embeddings` (P0a gate 4). Stubbed 501 until
//! P0a-e lands the wire-protocol addition that gets a hidden state out of
//! the engine at all (room A, 2026-09-17: `serve/src/protocol.rs`'s
//! `Request`/`EngineMsg` carry no hidden-state field today, and the data
//! only exists inside `serve/engine.mojo`'s decode loop, never on the
//! stdin/stdout wire). Design once P0a-e lands: last-token pooling of the
//! final hidden state, L2-normalized (the maintainer, 2026-09-17).

use super::*;

fn pending() -> ApiError {
    ApiError::Plain(
        StatusCode::NOT_IMPLEMENTED,
        "embeddings_pending: see P0a-e (engine wire addition for the hidden state)".into(),
    )
}

#[derive(Deserialize)]
pub struct EmbeddingsReq {
    #[allow(dead_code)]
    #[serde(default)]
    model: Option<String>,
    #[allow(dead_code)]
    #[serde(default)]
    input: Option<Value>,
    #[allow(dead_code)]
    #[serde(default)]
    prompt: Option<Value>,
}

pub async fn embeddings(State(_app): State<Shared>, Json(_r): Json<EmbeddingsReq>) -> Result<Json<Value>, ApiError> {
    Err(pending())
}
