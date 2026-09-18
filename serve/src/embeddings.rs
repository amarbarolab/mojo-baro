//! `/api/embeddings` and `/v1/embeddings` (P0a gate 4, landed once P0a-e's
//! engine wire merged to main). One engine request per input string: `n:1`,
//! `spec:false`, `embed:true`. The vector is the engine's last-prompt-token
//! post-final-norm hidden state, already L2-normalized on the engine side
//! (`serve/PROTOCOL.md`); this layer does not normalize it again.

use super::*;

fn input_strings(r: &EmbeddingsReq) -> Result<Vec<String>, ApiError> {
    let v = r.input.clone().or_else(|| r.prompt.clone()).ok_or_else(|| bad("input is required"))?;
    match v {
        Value::String(s) => Ok(vec![s]),
        Value::Array(items) => items
            .into_iter()
            .map(|item| match item {
                Value::String(s) => Ok(s),
                _ => Err(bad("input array must contain only strings (token-id embeddings are not supported by this item)")),
            })
            .collect(),
        _ => Err(bad("input must be a string or an array of strings")),
    }
}

/// Collects one embed request to its vector. `Embed` arrives before `Done`
/// (`serve/PROTOCOL.md`); a `Tok` line is expected too (the head runs
/// unfolded for this request) and carries nothing we need. An `Error` line
/// is the engine's documented refusal (spark, or `BARO_SEQS > 1`), mapped to
/// 501, not a generic failure code.
async fn collect_embed(mut rx: mpsc::UnboundedReceiver<Event>) -> Result<Vec<f32>, ApiError> {
    let mut vector: Option<Vec<f32>> = None;
    while let Some(ev) = rx.recv().await {
        match ev {
            Event::Tok { .. } => {}
            Event::Embed(v) => vector = Some(v),
            Event::Hidden(_) | Event::LogitsTopK(_) => {}
            Event::Done(_) => {
                return vector.ok_or_else(|| {
                    ApiError::Plain(StatusCode::INTERNAL_SERVER_ERROR, "engine finished an embed:true request with no embed line".into())
                });
            }
            Event::Error(e) => {
                return Err(ApiError::Plain(StatusCode::NOT_IMPLEMENTED, format!("embeddings_pending: {e} (spark engine or BARO_SEQS > 1)")));
            }
        }
    }
    Err(ApiError::Plain(StatusCode::BAD_GATEWAY, "engine closed the connection before finishing the embed request".into()))
}

#[derive(Deserialize)]
pub struct EmbeddingsReq {
    #[serde(default)]
    model: Option<String>,
    #[serde(default)]
    input: Option<Value>,
    /// Some clients (and the plan's own wording) call it `prompt`; accepted
    /// as an alias for `input` when `input` is absent.
    #[serde(default)]
    prompt: Option<Value>,
}

pub async fn embeddings(State(app): State<Shared>, Json(r): Json<EmbeddingsReq>) -> Result<Json<Value>, ApiError> {
    let t = need_text(&app)?;
    let inputs = input_strings(&r)?;
    if inputs.is_empty() {
        return Err(bad("input must not be empty"));
    }
    let model = r.model.clone().unwrap_or_else(|| app.model.clone());
    let mut data = Vec::with_capacity(inputs.len());
    let mut total_prompt_tokens = 0usize;
    for (i, text) in inputs.iter().enumerate() {
        let prompt = t.encode(text, true).map_err(bad)?;
        if prompt.is_empty() {
            return Err(bad("input tokenizes to zero tokens"));
        }
        total_prompt_tokens += prompt.len();
        let g = Gen {
            prompt,
            n: 1,
            spec: false,
            stream: false,
            stop: vec![],
            ckpt: vec![],
            state: (None, None),
            sample: protocol::SampleParams::default(),
            schema: None,
            reasoning: None,
            embed: Some(true),
        };
        let (_req_id, rx) = check_and_submit(&app, &g)?;
        let vector = collect_embed(rx).await?;
        data.push(json!({"object": "embedding", "index": i, "embedding": vector}));
    }
    Ok(Json(json!({
        "object": "list",
        "model": model,
        "data": data,
        "usage": {"prompt_tokens": total_prompt_tokens, "total_tokens": total_prompt_tokens},
    })))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `ApiError` has no `Debug` impl (main.rs), so `.unwrap()` does not
    /// compile on a `Result<_, ApiError>`; this is the local substitute.
    fn ok<T>(r: Result<T, ApiError>) -> T {
        match r {
            Ok(v) => v,
            Err(_) => panic!("expected Ok"),
        }
    }

    #[test]
    fn input_strings_accepts_a_single_string() {
        let r = EmbeddingsReq { model: None, input: Some(json!("hello")), prompt: None };
        assert_eq!(ok(input_strings(&r)), vec!["hello".to_string()]);
    }

    #[test]
    fn input_strings_accepts_an_array() {
        let r = EmbeddingsReq { model: None, input: Some(json!(["a", "b"])), prompt: None };
        assert_eq!(ok(input_strings(&r)), vec!["a".to_string(), "b".to_string()]);
    }

    #[test]
    fn input_strings_falls_back_to_prompt() {
        let r = EmbeddingsReq { model: None, input: None, prompt: Some(json!("via prompt")) };
        assert_eq!(ok(input_strings(&r)), vec!["via prompt".to_string()]);
    }

    #[test]
    fn input_strings_rejects_token_id_arrays() {
        let r = EmbeddingsReq { model: None, input: Some(json!([1, 2, 3])), prompt: None };
        assert!(input_strings(&r).is_err());
    }

    #[test]
    fn input_strings_requires_one_of_input_or_prompt() {
        let r = EmbeddingsReq { model: None, input: None, prompt: None };
        assert!(input_strings(&r).is_err());
    }

    #[tokio::test]
    async fn collect_embed_returns_the_vector_sent_before_done() {
        let (tx, rx) = mpsc::unbounded_channel();
        tx.send(Event::Tok { tok: 5, logprob: None, top_logprobs: vec![] }).unwrap();
        tx.send(Event::Embed(vec![0.1, 0.2, 0.3])).unwrap();
        tx.send(Event::Done(protocol::DoneStats {
            n: 1,
            prefill_s: 0.0,
            decode_s: 0.0,
            tok_s: 0.0,
            drafted: None,
            accepted: None,
            cached: None,
            prefill_rows: None,
            restore_s: None,
            finish: None,
        }))
        .unwrap();
        drop(tx);
        assert_eq!(ok(collect_embed(rx).await), vec![0.1, 0.2, 0.3]);
    }

    #[tokio::test]
    async fn collect_embed_maps_an_engine_refusal_to_501() {
        let (tx, rx) = mpsc::unbounded_channel();
        tx.send(Event::Error("embeddings not supported on this engine".into())).unwrap();
        drop(tx);
        match collect_embed(rx).await {
            Err(ApiError::Plain(code, msg)) => {
                assert_eq!(code, StatusCode::NOT_IMPLEMENTED);
                assert!(msg.contains("embeddings_pending"), "{msg}");
            }
            Err(_) => panic!("expected ApiError::Plain"),
            Ok(_) => panic!("expected an error"),
        }
    }

    #[tokio::test]
    async fn collect_embed_errors_when_done_arrives_with_no_vector() {
        let (tx, rx) = mpsc::unbounded_channel();
        tx.send(Event::Done(protocol::DoneStats {
            n: 1,
            prefill_s: 0.0,
            decode_s: 0.0,
            tok_s: 0.0,
            drafted: None,
            accepted: None,
            cached: None,
            prefill_rows: None,
            restore_s: None,
            finish: None,
        }))
        .unwrap();
        drop(tx);
        assert!(collect_embed(rx).await.is_err());
    }
}
