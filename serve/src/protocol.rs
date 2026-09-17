//! The stdin/stdout line protocol between this server and `serve/engine.mojo`
//! running with `BARO_SERVE=1`. Documented in `serve/PROTOCOL.md`.
//!
//! Every function here is total: a malformed or unexpected line is returned
//! as [`EngineMsg::Log`] and never panics or is trusted.

use serde::Serialize;
use serde_json::Value;

/// C3 sampler control block (bench/chat-protocol.md): parsed by the engine
/// (`serve/engine.mojo::SampleParams`) but not yet acted on there -- the
/// live decode loop still always takes the greedy/MTP path. `#[serde(skip_serializing_if)]`
/// on every field means a request that sets none of them serialises
/// byte-for-byte as it did before this struct existed.
#[derive(Debug, Clone, Default, Serialize, PartialEq)]
pub struct SampleParams {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub min_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub presence_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frequency_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_logprobs: Option<u32>,
}

/// One request line, serialised exactly as the engine's parser expects.
#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct Request {
    pub id: u64,
    pub prompt: Vec<u32>,
    pub n: u32,
    pub spec: bool,
    /// Token-id sequences; generation stops as soon as the tail of the
    /// generated tokens equals any of them (EOS ids included, as length-1
    /// sequences -- `serve/src/main.rs` builds this list).
    pub stop: Vec<Vec<u32>>,
    /// M1b role-boundary checkpoint hint positions (`Text::role_boundaries`),
    /// ascending; empty when the request has no message list.
    pub ckpt: Vec<u32>,
    /// Checkpoint API (LatentOS plan 10 sec 8): BAROST01 state file the
    /// engine writes after prefill / loads before the chain lookup. Absent
    /// keys, so an engine without the feature ignores them.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub state_save: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub state_load: Option<String>,
    #[serde(flatten)]
    pub sample: SampleParams,
    /// JSON-enforcement item 1 (briefs/2026-09-16-json-enforcement-lane.md):
    /// `response_format.json_schema.schema`, verbatim -- `serve/grammar_rt.mojo`
    /// compiles it. Absent for every request that does not set
    /// `response_format`, which is every request before this lane.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub schema: Option<Value>,
    /// Item 2: whether the model is reasoning for this request (from
    /// `chat_template_kwargs.enable_thinking`, default true). Only read by
    /// the engine when `schema` is also set.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<bool>,
    /// P0a-e (`serve/PROTOCOL.md`): ask for the last-prompt-token hidden
    /// state as an extra `{"id","embed":[...]}` line between the first
    /// token and `done`. Absent (not `Some(false)`) on every request that
    /// does not ask, which is every request before this lane. The caller
    /// still sends `n:1, spec:false`; the engine does not infer them from
    /// this flag.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub embed: Option<bool>,
}

/// The line that cancels the request currently decoding, if its id matches.
/// Written to the same stdin the request line uses; the engine polls for it
/// once per decode window.
pub fn cancel_line(id: u64) -> String {
    format!("{{\"cancel\":{id}}}\n")
}

impl Request {
    /// The wire form: one JSON object plus a trailing newline.
    pub fn line(&self) -> String {
        // A struct of integers, a vector and a bool cannot fail to serialise.
        let mut s = serde_json::to_string(self).unwrap_or_default();
        s.push('\n');
        s
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct DoneStats {
    pub n: u32,
    pub prefill_s: f64,
    pub decode_s: f64,
    pub tok_s: f64,
    pub drafted: Option<u64>,
    pub accepted: Option<u64>,
    pub cached: Option<u64>,
    pub prefill_rows: Option<u64>,
    pub restore_s: Option<f64>,
    /// "length" | "stop" | "cancelled"; absent on an older engine that has
    /// no concept of an early stop, in which case the caller falls back to
    /// its own client-side EOS check.
    pub finish: Option<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum EngineMsg {
    /// Printed once after the pack is loaded and every buffer is allocated.
    Ready {
        tmax: u32,
        mrows: u32,
        kmax: u32,
        spec_k: u32,
        pack: String,
    },
    /// One generated token id for request `id`, in generation order.
    /// `logprob`/`top_logprobs` are present only when the engine actually
    /// sampled with `top_logprobs > 0` for this token (item 4,
    /// briefs/2026-09-16-sampling-all-models-lane.md); absent on every
    /// other line, including every line before this feature existed.
    Tok { id: u64, tok: u32, logprob: Option<f64>, top_logprobs: Vec<(u32, f64)> },
    /// P0a-e: the requested embedding vector for request `id`, sent once
    /// between the first `Tok` line and `Done`. Does not end the request.
    Embed { id: u64, vector: Vec<f32> },
    /// Request `id` finished; no more `Tok` lines follow for it.
    Done { id: u64, stats: DoneStats },
    /// Request `id` was rejected before any token was produced.
    Error { id: u64, error: String },
    /// Anything else: engine diagnostics, or a line that failed validation.
    Log(String),
}

fn get_u64(v: &Value, key: &str) -> Option<u64> {
    v.get(key)?.as_u64()
}

fn get_u32(v: &Value, key: &str) -> Option<u32> {
    u32::try_from(get_u64(v, key)?).ok()
}

fn get_f64(v: &Value, key: &str) -> Option<f64> {
    let f = v.get(key)?.as_f64()?;
    f.is_finite().then_some(f)
}

/// Classify one engine stdout line. Never trusts a malformed line.
pub fn parse_line(line: &str) -> EngineMsg {
    let raw = line.trim_end_matches(['\r', '\n']);
    if !raw.starts_with('{') {
        return EngineMsg::Log(raw.to_string());
    }
    let v: Value = match serde_json::from_str(raw) {
        Ok(Value::Object(m)) => Value::Object(m),
        _ => return EngineMsg::Log(raw.to_string()),
    };
    if v.get("ready").and_then(Value::as_bool) == Some(true) {
        if let (Some(tmax), Some(mrows), Some(kmax), Some(spec_k), Some(pack)) = (
            get_u32(&v, "tmax"),
            get_u32(&v, "mrows"),
            get_u32(&v, "kmax"),
            get_u32(&v, "spec_k"),
            v.get("pack").and_then(Value::as_str),
        ) {
            return EngineMsg::Ready {
                tmax,
                mrows,
                kmax,
                spec_k,
                pack: pack.to_string(),
            };
        }
        return EngineMsg::Log(raw.to_string());
    }
    let Some(id) = get_u64(&v, "id") else {
        return EngineMsg::Log(raw.to_string());
    };
    if let Some(err) = v.get("error") {
        return match err.as_str() {
            Some(s) => EngineMsg::Error {
                id,
                error: s.to_string(),
            },
            None => EngineMsg::Log(raw.to_string()),
        };
    }
    if v.get("done").and_then(Value::as_bool) == Some(true) {
        if let (Some(n), Some(prefill_s), Some(decode_s), Some(tok_s)) = (
            get_u32(&v, "n"),
            get_f64(&v, "prefill_s"),
            get_f64(&v, "decode_s"),
            get_f64(&v, "tok_s"),
        ) {
            return EngineMsg::Done {
                id,
                stats: DoneStats {
                    n,
                    prefill_s,
                    decode_s,
                    tok_s,
                    drafted: get_u64(&v, "drafted"),
                    accepted: get_u64(&v, "accepted"),
                    cached: get_u64(&v, "cached"),
                    prefill_rows: get_u64(&v, "prefill_rows"),
                    restore_s: get_f64(&v, "restore_s"),
                    finish: v.get("finish").and_then(Value::as_str).map(str::to_string),
                },
            };
        }
        return EngineMsg::Log(raw.to_string());
    }
    if let Some(vector) = v.get("embed") {
        return match vector.as_array() {
            Some(arr) => match arr.iter().map(|e| e.as_f64()).collect::<Option<Vec<f64>>>() {
                Some(floats) if floats.iter().all(|f| f.is_finite()) => {
                    EngineMsg::Embed { id, vector: floats.into_iter().map(|f| f as f32).collect() }
                }
                _ => EngineMsg::Log(raw.to_string()),
            },
            None => EngineMsg::Log(raw.to_string()),
        };
    }
    if let Some(tok) = get_u32(&v, "tok") {
        let logprob = get_f64(&v, "logprob");
        let top_logprobs = v
            .get("top_logprobs")
            .and_then(Value::as_array)
            .map(|arr| {
                arr.iter()
                    .filter_map(|e| Some((get_u32(e, "id")?, get_f64(e, "logprob")?)))
                    .collect()
            })
            .unwrap_or_default();
        return EngineMsg::Tok { id, tok, logprob, top_logprobs };
    }
    EngineMsg::Log(raw.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn request_line_is_the_engine_shape() {
        let r = Request {
            id: 7,
            prompt: vec![760, 6511, 314],
            n: 64,
            spec: false,
            stop: vec![],
            ckpt: vec![],
            state_save: None,
            state_load: None,
            sample: SampleParams::default(),
            schema: None,
            reasoning: None,
            embed: None,
        };
        assert_eq!(r.line(), "{\"id\":7,\"prompt\":[760,6511,314],\"n\":64,\"spec\":false,\"stop\":[],\"ckpt\":[]}\n");
    }

    #[test]
    fn request_line_carries_stop_sequences() {
        let r = Request {
            id: 1,
            prompt: vec![1],
            n: 8,
            spec: false,
            stop: vec![vec![151645], vec![9707, 11]],
            ckpt: vec![],
            state_save: None,
            state_load: None,
            sample: SampleParams::default(),
            schema: None,
            reasoning: None,
            embed: None,
        };
        assert_eq!(
            r.line(),
            "{\"id\":1,\"prompt\":[1],\"n\":8,\"spec\":false,\"stop\":[[151645],[9707,11]],\"ckpt\":[]}\n"
        );
    }

    #[test]
    fn request_line_carries_ckpt_hints() {
        let r = Request {
            id: 2,
            prompt: vec![1, 2, 3],
            n: 8,
            spec: false,
            stop: vec![],
            ckpt: vec![7914, 8020],
            state_save: None,
            state_load: None,
            sample: SampleParams::default(),
            schema: None,
            reasoning: None,
            embed: None,
        };
        assert_eq!(
            r.line(),
            "{\"id\":2,\"prompt\":[1,2,3],\"n\":8,\"spec\":false,\"stop\":[],\"ckpt\":[7914,8020]}\n"
        );
    }

    #[test]
    fn request_line_carries_sample_params_only_when_set() {
        let r = Request {
            id: 3,
            prompt: vec![1],
            n: 8,
            spec: false,
            stop: vec![],
            ckpt: vec![],
            state_save: None,
            state_load: None,
            sample: SampleParams {
                temperature: Some(0.8),
                top_p: Some(0.9),
                top_k: Some(40),
                min_p: Some(0.05),
                seed: Some(42),
                presence_penalty: None,
                frequency_penalty: None,
                top_logprobs: None,
            },
            schema: None,
            reasoning: None,
            embed: None,
        };
        assert_eq!(
            r.line(),
            "{\"id\":3,\"prompt\":[1],\"n\":8,\"spec\":false,\"stop\":[],\"ckpt\":[],\"temperature\":0.8,\"top_p\":0.9,\"top_k\":40,\"min_p\":0.05,\"seed\":42}\n"
        );
    }

    #[test]
    fn request_line_carries_schema_and_reasoning_only_when_set() {
        let r = Request {
            id: 4,
            prompt: vec![1],
            n: 8,
            spec: false,
            stop: vec![],
            ckpt: vec![],
            state_save: None,
            state_load: None,
            sample: SampleParams {
                temperature: Some(0.7),
                ..SampleParams::default()
            },
            schema: Some(json!({"type": "object", "properties": {"a": {"type": "string"}}})),
            reasoning: Some(false),
            embed: None,
        };
        assert_eq!(
            r.line(),
            "{\"id\":4,\"prompt\":[1],\"n\":8,\"spec\":false,\"stop\":[],\"ckpt\":[],\"temperature\":0.7,\"schema\":{\"properties\":{\"a\":{\"type\":\"string\"}},\"type\":\"object\"},\"reasoning\":false}\n"
        );
    }

    #[test]
    fn request_line_carries_embed_only_when_set() {
        let r = Request {
            id: 6,
            prompt: vec![1],
            n: 1,
            spec: false,
            stop: vec![],
            ckpt: vec![],
            state_save: None,
            state_load: None,
            sample: SampleParams::default(),
            schema: None,
            reasoning: None,
            embed: Some(true),
        };
        assert_eq!(r.line(), "{\"id\":6,\"prompt\":[1],\"n\":1,\"spec\":false,\"stop\":[],\"ckpt\":[],\"embed\":true}\n");
    }

    #[test]
    fn cancel_line_is_a_bare_id() {
        assert_eq!(cancel_line(42), "{\"cancel\":42}\n");
    }

    #[test]
    fn ready_line() {
        let m = parse_line(
            "{\"ready\":true,\"tmax\":128,\"mrows\":8,\"kmax\":8,\"spec_k\":2,\"pack\":\".work/engine-pack-q4\"}\n",
        );
        assert_eq!(
            m,
            EngineMsg::Ready {
                tmax: 128,
                mrows: 8,
                kmax: 8,
                spec_k: 2,
                pack: ".work/engine-pack-q4".into()
            }
        );
    }

    #[test]
    fn tok_and_done_lines() {
        assert_eq!(parse_line("{\"id\":3,\"tok\":11751}"), EngineMsg::Tok { id: 3, tok: 11751, logprob: None, top_logprobs: vec![] });
        let m = parse_line(
            "{\"id\":3,\"done\":true,\"n\":64,\"prefill_s\":0.01,\"decode_s\":0.5,\"tok_s\":126.0,\"drafted\":40,\"accepted\":28,\"k\":2,\"cached\":7913,\"prefill_rows\":41,\"restore_s\":0.002}",
        );
        match m {
            EngineMsg::Done { id, stats } => {
                assert_eq!(id, 3);
                assert_eq!(stats.n, 64);
                assert_eq!(stats.drafted, Some(40));
                assert_eq!(stats.accepted, Some(28));
                assert_eq!(stats.tok_s, 126.0);
                assert_eq!(stats.cached, Some(7913));
                assert_eq!(stats.prefill_rows, Some(41));
                assert_eq!(stats.restore_s, Some(0.002));
                assert_eq!(stats.finish, None);
            }
            other => panic!("expected Done, got {other:?}"),
        }
        let m = parse_line("{\"id\":4,\"done\":true,\"n\":1,\"prefill_s\":0.01,\"decode_s\":0.0,\"tok_s\":0.0}");
        assert!(matches!(m, EngineMsg::Done { id: 4, .. }));
    }

    #[test]
    fn done_line_carries_finish_reason() {
        for (finish, n) in [("stop", 9u32), ("cancelled", 3u32)] {
            let m = parse_line(&format!(
                "{{\"id\":5,\"done\":true,\"n\":{n},\"prefill_s\":0.01,\"decode_s\":0.1,\"tok_s\":1.0,\"finish\":\"{finish}\"}}"
            ));
            match m {
                EngineMsg::Done { stats, .. } => {
                    assert_eq!(stats.n, n);
                    assert_eq!(stats.finish, Some(finish.to_string()));
                }
                other => panic!("expected Done, got {other:?}"),
            }
        }
    }

    #[test]
    fn embed_line() {
        assert_eq!(
            parse_line("{\"id\":6,\"embed\":[0.1,0.2,-0.3]}"),
            EngineMsg::Embed { id: 6, vector: vec![0.1, 0.2, -0.3] }
        );
        assert_eq!(parse_line("{\"id\":6,\"embed\":[]}"), EngineMsg::Embed { id: 6, vector: vec![] });
    }

    #[test]
    fn error_line() {
        assert_eq!(
            parse_line("{\"id\":9,\"error\":\"prompt+n exceeds TMAX 128\"}"),
            EngineMsg::Error {
                id: 9,
                error: "prompt+n exceeds TMAX 128".into()
            }
        );
    }

    #[test]
    fn diagnostics_are_logs() {
        assert_eq!(parse_line("tok/s_gen: 126.5\n"), EngineMsg::Log("tok/s_gen: 126.5".into()));
        assert_eq!(parse_line("GENERATED: 1 2 3 "), EngineMsg::Log("GENERATED: 1 2 3 ".into()));
        assert_eq!(parse_line(""), EngineMsg::Log(String::new()));
    }

    #[test]
    fn malformed_lines_are_never_trusted() {
        for bad in [
            "{\"id\":",                              // truncated
            "{\"id\":\"x\",\"tok\":5}",              // id not an integer
            "{\"id\":1,\"tok\":-5}",                 // negative token
            "{\"id\":1,\"tok\":4294967296}",         // token above u32
            "{\"id\":1,\"tok\":\"5\"}",              // token as string
            "{\"tok\":5}",                           // no id
            "{\"id\":1,\"done\":true}",              // done without stats
            "{\"id\":1,\"done\":true,\"n\":1,\"prefill_s\":\"a\",\"decode_s\":0,\"tok_s\":0}",
            "{\"id\":1,\"error\":5}",                // error not a string
            "{\"ready\":true}",                      // ready without limits
            "[1,2,3]",                               // not an object
            "{\"id\":1,\"done\":true,\"n\":1,\"prefill_s\":1e999,\"decode_s\":0,\"tok_s\":0}",
            "{\"id\":1,\"embed\":\"x\"}",             // embed not an array
            "{\"id\":1,\"embed\":[\"x\"]}",           // embed element not a number
        ] {
            assert!(matches!(parse_line(bad), EngineMsg::Log(_)), "{bad} must not parse");
        }
    }

    #[test]
    fn extra_fields_are_ignored() {
        assert_eq!(
            parse_line("{\"id\":1,\"tok\":2,\"extra\":{\"x\":[1]}}"),
            EngineMsg::Tok { id: 1, tok: 2, logprob: None, top_logprobs: vec![] }
        );
    }
}
