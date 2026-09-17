//! `/api/embeddings` and `/v1/embeddings` (P0a gate 4). Stubbed 501 until
//! P0a-e lands the wire-protocol addition that gets a hidden state out of
//! the engine at all (room A, 2026-09-17: `serve/src/protocol.rs`'s
//! `Request`/`EngineMsg` carry no hidden-state field today, and the data
//! only exists inside `serve/engine.mojo`'s decode loop, never on the
//! stdin/stdout wire). Design once P0a-e lands: last-token pooling of the
//! final hidden state, L2-normalized (the maintainer, 2026-09-17).

use super::*;

/// Coordinator's answer (room A, 2026-09-17): keep the repo-wide
/// `ApiError::Plain` shape (`{"error": {"message", "type", "code"}}`); the
/// earlier literal `{"error":"embeddings_pending",...}` was shorthand, not
/// a contract. `embeddings_pending` and `P0a-e` stay in the message text so
/// a gate can still assert on them.
fn pending() -> ApiError {
    ApiError::Plain(StatusCode::NOT_IMPLEMENTED, "embeddings_pending: see P0a-e (engine wire addition for the hidden state)".into())
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

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn pending_response_is_501_naming_embeddings_pending_and_p0a_e() {
        let resp = pending().into_response();
        assert_eq!(resp.status(), StatusCode::NOT_IMPLEMENTED);
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX).await.unwrap();
        let v: Value = serde_json::from_slice(&body).unwrap();
        let msg = v["error"]["message"].as_str().unwrap();
        assert!(msg.contains("embeddings_pending") && msg.contains("P0a-e"), "{msg}");
        assert_eq!(v["error"]["code"], 501);
    }
}
