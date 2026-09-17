//! `/api/embeddings` and `/v1/embeddings` (P0a gate 4). Stubbed 501 until
//! P0a-e lands the wire-protocol addition that gets a hidden state out of
//! the engine at all (room A, 2026-09-17: `serve/src/protocol.rs`'s
//! `Request`/`EngineMsg` carry no hidden-state field today, and the data
//! only exists inside `serve/engine.mojo`'s decode loop, never on the
//! stdin/stdout wire). Design once P0a-e lands: last-token pooling of the
//! final hidden state, L2-normalized (the maintainer, 2026-09-17).

use super::*;

/// The coordinator's literal contract (room A, 2026-09-17): `{"error":
/// "embeddings_pending", "see": "P0a-e"}`, not `ApiError::Plain`'s generic
/// `{"error": {"message": ..., "type": ..., "code": ...}}` shape -- a gate
/// or client parsing `error` as a string would otherwise see an object.
fn pending() -> Response {
    (StatusCode::NOT_IMPLEMENTED, Json(json!({"error": "embeddings_pending", "see": "P0a-e"}))).into_response()
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

pub async fn embeddings(State(_app): State<Shared>, Json(_r): Json<EmbeddingsReq>) -> Response {
    pending()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn pending_response_is_501_with_the_coordinators_literal_shape() {
        let resp = pending();
        assert_eq!(resp.status(), StatusCode::NOT_IMPLEMENTED);
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX).await.unwrap();
        let v: Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(v, json!({"error": "embeddings_pending", "see": "P0a-e"}));
    }
}
