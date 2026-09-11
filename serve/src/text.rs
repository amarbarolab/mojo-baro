//! Text in, text out: the pack's `tokenizer.json` (HF `tokenizers` format,
//! written by tools/gguf-tokenizer.py) plus the optional
//! `tokenizer-meta.json` (bos/eos ids, chat template). Absent tokenizer =>
//! the server still runs, token-id endpoints only.

use std::path::Path;

use minijinja::Environment;
use serde::Deserialize;
use serde_json::Value;
use tokenizers::Tokenizer;

#[derive(Debug, Default, Deserialize)]
struct Meta {
    #[serde(default)]
    chat_template: Option<String>,
    #[serde(default, alias = "bos_id")]
    bos_token_id: Option<u32>,
    #[serde(default, alias = "eos_id")]
    eos_token_id: Option<u32>,
    #[serde(default)]
    bos_token: Option<String>,
    #[serde(default)]
    eos_token: Option<String>,
    #[serde(default)]
    add_bos: Option<bool>,
}

#[derive(Debug, Clone)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

pub struct Text {
    tok: Tokenizer,
    meta: Meta,
    /// Token ids that end a generation (eos, plus Qwen's <|im_end|> /
    /// <|endoftext|> when the vocabulary has them).
    pub stop_ids: Vec<u32>,
}

impl Text {
    pub fn load(tokenizer_json: &Path) -> Result<Text, String> {
        let tok = Tokenizer::from_file(tokenizer_json).map_err(|e| format!("{}: {e}", tokenizer_json.display()))?;
        let meta_path = tokenizer_json.with_file_name("tokenizer-meta.json");
        let meta: Meta = match std::fs::read_to_string(&meta_path) {
            Ok(s) => serde_json::from_str(&s).map_err(|e| format!("{}: {e}", meta_path.display()))?,
            Err(_) => Meta::default(),
        };
        let mut stop_ids = Vec::new();
        if let Some(e) = meta.eos_token_id {
            stop_ids.push(e);
        }
        for s in ["<|im_end|>", "<|endoftext|>"] {
            if let Some(id) = tok.token_to_id(s) {
                if !stop_ids.contains(&id) {
                    stop_ids.push(id);
                }
            }
        }
        Ok(Text { tok, meta, stop_ids })
    }

    pub fn encode(&self, text: &str, add_special: bool) -> Result<Vec<u32>, String> {
        let enc = self.tok.encode(text, add_special).map_err(|e| e.to_string())?;
        let mut ids = enc.get_ids().to_vec();
        if add_special && self.meta.add_bos == Some(true) {
            if let Some(bos) = self.meta.bos_token_id {
                if ids.first() != Some(&bos) {
                    ids.insert(0, bos);
                }
            }
        }
        Ok(ids)
    }

    pub fn decode(&self, ids: &[u32]) -> Result<String, String> {
        self.tok.decode(ids, false).map_err(|e| e.to_string())
    }

    /// Render the chat template (the pack's, else ChatML which is what Qwen
    /// ships) with the generation prompt appended.
    pub fn apply_chat_template(&self, messages: &[ChatMessage]) -> Result<String, String> {
        self.render(messages, true)
    }

    fn render(&self, messages: &[ChatMessage], add_generation_prompt: bool) -> Result<String, String> {
        let Some(tpl) = &self.meta.chat_template else {
            let mut s = String::new();
            for m in messages {
                s.push_str("<|im_start|>");
                s.push_str(&m.role);
                s.push('\n');
                s.push_str(&m.content);
                s.push_str("<|im_end|>\n");
            }
            if add_generation_prompt {
                s.push_str("<|im_start|>assistant\n");
            }
            return Ok(s);
        };
        let mut env = Environment::new();
        minijinja_contrib::add_to_environment(&mut env);
        env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
        env.add_template("chat", tpl).map_err(|e| format!("chat_template: {e}"))?;
        let msgs: Vec<Value> = messages
            .iter()
            .map(|m| serde_json::json!({"role": m.role, "content": m.content}))
            .collect();
        let ctx = serde_json::json!({
            "messages": msgs,
            "add_generation_prompt": add_generation_prompt,
            "bos_token": self.meta.bos_token.clone().unwrap_or_default(),
            "eos_token": self.meta.eos_token.clone().unwrap_or_default(),
        });
        env.get_template("chat")
            .and_then(|t| t.render(minijinja::Value::from_serialize(&ctx)))
            .map_err(|e| format!("chat_template: {e}"))
    }

    /// M1b role-boundary checkpoint hints: the token length after each
    /// message when the conversation up to that message is rendered with no
    /// generation prompt. A later turn's full prompt starts with exactly
    /// these same bytes (the history before it does not change), so these
    /// positions double as restore points -- and the engine's prefix hash
    /// (`serve/prefix.mojo`) catches a mismatch rather than trusting this
    /// blindly, so a template this trick does not fit just wastes a hint,
    /// never corrupts a response.
    ///
    /// A prefix that does not render or tokenize on its own is skipped, not
    /// propagated: real templates validate the *whole* conversation (e.g.
    /// Qwen agent templates `raise_exception` a system-only prefix with "no
    /// user query found"), so an early k can legitimately fail here even
    /// though the full render (this function's caller renders separately)
    /// succeeds. Hints are an optimization; a request must never 500 over
    /// one being unavailable.
    pub fn role_boundaries(&self, messages: &[ChatMessage]) -> Vec<u32> {
        let mut out = Vec::with_capacity(messages.len());
        for k in 1..=messages.len() {
            let Ok(rendered) = self.render(&messages[..k], false) else { continue };
            let Ok(ids) = self.encode(&rendered, true) else { continue };
            out.push(ids.len() as u32);
        }
        out.dedup();
        out
    }

    pub fn is_stop(&self, id: u32) -> bool {
        self.stop_ids.contains(&id)
    }
}

/// Incremental detokenizer for streaming: decodes the whole prefix each
/// step (prefixes are <= TMAX tokens) and emits only the new, complete
/// text, holding back a trailing replacement character from a split
/// multi-byte sequence.
pub struct Detok {
    ids: Vec<u32>,
    emitted: usize,
}

impl Detok {
    pub fn new() -> Detok {
        Detok {
            ids: Vec::new(),
            emitted: 0,
        }
    }

    pub fn push(&mut self, text: &Text, id: u32) -> String {
        self.ids.push(id);
        let Ok(full) = text.decode(&self.ids) else {
            return String::new();
        };
        let stable = full.strip_suffix('\u{FFFD}').unwrap_or(&full);
        if stable.len() <= self.emitted || !stable.is_char_boundary(self.emitted) {
            return String::new();
        }
        let delta = stable[self.emitted..].to_string();
        self.emitted = stable.len();
        delta
    }

}

impl Default for Detok {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chatml_fallback_shape() {
        let t = Text {
            tok: Tokenizer::new(tokenizers::models::bpe::BPE::default()),
            meta: Meta::default(),
            stop_ids: vec![],
        };
        let s = t
            .apply_chat_template(&[
                ChatMessage {
                    role: "system".into(),
                    content: "be brief".into(),
                },
                ChatMessage {
                    role: "user".into(),
                    content: "hi".into(),
                },
            ])
            .unwrap();
        assert_eq!(
            s,
            "<|im_start|>system\nbe brief<|im_end|>\n<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n"
        );
    }

    #[test]
    fn role_boundary_renders_are_prefixes_of_the_full_render() {
        // The BPE-with-no-vocab tokenizer here can't produce meaningful
        // token lengths (every encode is empty), so this exercises the
        // string-level invariant role_boundaries relies on -- that each
        // prefix render is a byte-exact prefix of the full render -- not
        // the token-length numbers themselves (covered with a real
        // tokenizer by tools/test_server.sh's M1b case).
        let t = Text {
            tok: Tokenizer::new(tokenizers::models::bpe::BPE::default()),
            meta: Meta::default(),
            stop_ids: vec![],
        };
        let msgs = [
            ChatMessage { role: "system".into(), content: "be brief".into() },
            ChatMessage { role: "user".into(), content: "hi".into() },
        ];
        let full = t.render(&msgs, false).unwrap();
        for k in 1..=msgs.len() {
            let prefix = t.render(&msgs[..k], false).unwrap();
            assert!(full.starts_with(&prefix), "message {k} render is not a prefix: {prefix:?} vs {full:?}");
        }
        assert!(t.role_boundaries(&msgs).len() <= msgs.len());
    }

    #[test]
    fn role_boundaries_skips_a_prefix_that_fails_to_render() {
        // messages[1] is out of bounds for a 1-message prefix -- a stand-in
        // for a real chat template's own validation rejecting a shorter
        // prefix (Qwen agent templates `raise_exception` a system-only
        // prefix with "no user query found"). k=1 must be skipped, not
        // turn the whole call into an error.
        let t = Text {
            tok: Tokenizer::new(tokenizers::models::bpe::BPE::default()),
            meta: Meta {
                chat_template: Some("{% for m in messages %}<{{ m.role }}>{% endfor %}{{ messages[1].role }}".into()),
                ..Meta::default()
            },
            stop_ids: vec![],
        };
        let msgs = [
            ChatMessage { role: "system".into(), content: "be brief".into() },
            ChatMessage { role: "user".into(), content: "hi".into() },
        ];
        // Doesn't panic or return an Err; k=1 (renders fine, but as an
        // empty-BPE encode -> filtered by the same empty-encode behaviour
        // as any other prefix) contributes nothing, k=2 does not error.
        let _ = t.role_boundaries(&msgs);
        assert!(t.render(&msgs, false).is_ok(), "the full render (k=2, what a real request sends) must succeed");
        assert!(t.render(&msgs[..1], false).is_err(), "k=1 must actually fail to render, or this test proves nothing");
    }

    #[test]
    fn jinja_template_renders_with_pycompat() {
        let t = Text {
            tok: Tokenizer::new(tokenizers::models::bpe::BPE::default()),
            meta: Meta {
                chat_template: Some(
                    "{% for m in messages %}<{{ m.role }}>{{ m.content.strip() }}</>{% endfor %}{% if add_generation_prompt %}<assistant>{% endif %}".into(),
                ),
                ..Meta::default()
            },
            stop_ids: vec![],
        };
        let s = t
            .apply_chat_template(&[ChatMessage {
                role: "user".into(),
                content: "  hi  ".into(),
            }])
            .unwrap();
        assert_eq!(s, "<user>hi</><assistant>");
    }
}
