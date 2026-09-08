#!/usr/bin/env python3
"""Build an HF `tokenizers` BPE tokenizer from GGUF metadata alone.

Reads tokenizer.ggml.{model,pre,tokens,merges,token_type}, the bos/eos/pad ids,
the add_bos/add_eos flags and tokenizer.chat_template from MODEL.gguf and
writes, next to a pack:

  OUTDIR/tokenizer.json        HF tokenizers serialization (Python and Rust)
  OUTDIR/tokenizer-meta.json   ids, flags, special-token map, chat template

Byte-level BPE (tokenizer.ggml.model == "gpt2") with the llama.cpp
pre-tokenizer regex for the recorded `pre` type; the goal is token ids
bit-equal to llama.cpp on the same file (tools/test_tokenizer.py checks).

Usage: tools/gguf-tokenizer.py MODEL.gguf OUTDIR [OUTDIR ...]
"""
import datetime
import json
import struct
import sys
from pathlib import Path

# llama.cpp src/llama-vocab.cpp, "original regex from tokenizer.json" per pre type.
PRE_REGEX = {
    "qwen2": r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
    "qwen35": r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
}
PRE_REGEX["deepseek-r1-qwen"] = PRE_REGEX["qwen2"]

# gguf token_type: 1 normal, 2 unknown, 3 control, 4 user_defined, 5 unused, 6 byte
CONTROL, USER_DEFINED = 3, 4

SCALAR_FMT = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i",
              6: "<f", 7: "<B", 10: "<Q", 11: "<q", 12: "<d"}


def _read_str(f):
    (n,) = struct.unpack("<Q", f.read(8))
    return f.read(n).decode("utf-8")


def _read_value(f, vtype):
    if vtype in SCALAR_FMT:
        fmt = SCALAR_FMT[vtype]
        v = struct.unpack(fmt, f.read(struct.calcsize(fmt)))[0]
        return bool(v) if vtype == 7 else v
    if vtype == 8:
        return _read_str(f)
    if vtype == 9:
        (etype,) = struct.unpack("<I", f.read(4))
        (n,) = struct.unpack("<Q", f.read(8))
        return [_read_value(f, etype) for _ in range(n)]
    raise ValueError(f"unknown kv type {vtype}")


def read_metadata(path):
    """All GGUF key/value metadata, arrays fully materialized."""
    with open(path, "rb") as f:
        magic, version = struct.unpack("<4sI", f.read(8))
        assert magic == b"GGUF" and version == 3, (magic, version)
        _n_tensors, n_kv = struct.unpack("<QQ", f.read(16))
        kv = {}
        for _ in range(n_kv):
            key = _read_str(f)
            (vtype,) = struct.unpack("<I", f.read(4))
            kv[key] = _read_value(f, vtype)
    return kv


def build(kv, source):
    model = kv["tokenizer.ggml.model"]
    pre = kv.get("tokenizer.ggml.pre", "default")
    if model != "gpt2":
        raise SystemExit(f"tokenizer.ggml.model={model!r}: only byte-level BPE (gpt2) is supported")
    if pre not in PRE_REGEX:
        raise SystemExit(f"tokenizer.ggml.pre={pre!r}: no pre-tokenizer regex on file (known: {sorted(PRE_REGEX)})")
    tokens = kv["tokenizer.ggml.tokens"]
    types = kv["tokenizer.ggml.token_type"]
    merges = kv["tokenizer.ggml.merges"]
    assert len(tokens) == len(types) == len(set(tokens)), "vocab must be unique"

    added = []
    for i, (tok, ty) in enumerate(zip(tokens, types)):
        if ty in (CONTROL, USER_DEFINED):
            added.append({"id": i, "content": tok, "single_word": False, "lstrip": False,
                          "rstrip": False, "normalized": False, "special": ty == CONTROL})

    tokenizer = {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": added,
        "normalizer": None,
        "pre_tokenizer": {"type": "Sequence", "pretokenizers": [
            {"type": "Split", "pattern": {"Regex": PRE_REGEX[pre]}, "behavior": "Isolated", "invert": False},
            {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": False, "use_regex": False},
        ]},
        "post_processor": None,
        "decoder": {"type": "ByteLevel", "add_prefix_space": True, "trim_offsets": True, "use_regex": True},
        "model": {
            "type": "BPE", "dropout": None, "unk_token": None,
            "continuing_subword_prefix": "", "end_of_word_suffix": "",
            "fuse_unk": False, "byte_fallback": False, "ignore_merges": False,
            "vocab": {tok: i for i, tok in enumerate(tokens)},
            "merges": merges,
        },
    }

    def tok_of(key):
        i = kv.get(key)
        return None if i is None else tokens[i]

    meta = {
        "source_gguf": str(Path(source).resolve()),
        "built": datetime.datetime.now().isoformat(timespec="seconds"),
        "model": model,
        "pre": pre,
        "vocab_size": len(tokens),
        "n_merges": len(merges),
        "bos_token_id": kv.get("tokenizer.ggml.bos_token_id"),
        "eos_token_id": kv.get("tokenizer.ggml.eos_token_id"),
        "pad_token_id": kv.get("tokenizer.ggml.padding_token_id"),
        "bos_token": tok_of("tokenizer.ggml.bos_token_id"),
        "eos_token": tok_of("tokenizer.ggml.eos_token_id"),
        "pad_token": tok_of("tokenizer.ggml.padding_token_id"),
        "add_bos": bool(kv.get("tokenizer.ggml.add_bos_token", False)),
        "add_eos": bool(kv.get("tokenizer.ggml.add_eos_token", False)),
        "special_tokens": {a["content"]: a["id"] for a in added if a["special"]},
        "added_tokens": {a["content"]: a["id"] for a in added if not a["special"]},
        "chat_template": kv.get("tokenizer.chat_template"),
    }
    return tokenizer, meta


def load_meta(pack):
    return json.loads((Path(pack) / "tokenizer-meta.json").read_text())


def load_tokenizer(pack):
    from tokenizers import Tokenizer
    return Tokenizer.from_file(str(Path(pack) / "tokenizer.json"))


def render_chat(meta, messages, add_generation_prompt=True, **kwargs):
    """Apply the GGUF chat template the way transformers does (jinja2, same helpers)."""
    import jinja2
    tpl = meta.get("chat_template")
    if not tpl:
        raise SystemExit("pack has no chat template (tokenizer.chat_template missing in the GGUF)")

    def raise_exception(msg):
        raise jinja2.exceptions.TemplateError(msg)

    env = jinja2.Environment(trim_blocks=True, lstrip_blocks=True, extensions=["jinja2.ext.loopcontrols"])
    env.globals["raise_exception"] = raise_exception
    env.globals["strftime_now"] = lambda fmt: datetime.datetime.now().strftime(fmt)
    env.filters["tojson"] = lambda v, **kw: json.dumps(v, ensure_ascii=False, **kw)
    return env.from_string(tpl).render(
        messages=messages, add_generation_prompt=add_generation_prompt,
        bos_token=meta.get("bos_token") or "", eos_token=meta.get("eos_token") or "",
        pad_token=meta.get("pad_token") or "", **kwargs)


def main():
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    source, outdirs = sys.argv[1], sys.argv[2:]
    kv = read_metadata(source)
    tokenizer, meta = build(kv, source)
    for out in outdirs:
        out = Path(out)
        out.mkdir(parents=True, exist_ok=True)
        (out / "tokenizer.json").write_text(json.dumps(tokenizer, ensure_ascii=False))
        (out / "tokenizer-meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1))
        print(f"{out}/tokenizer.json: {meta['model']}/{meta['pre']} vocab={meta['vocab_size']} "
              f"merges={meta['n_merges']} special={len(meta['special_tokens'])} "
              f"added={len(meta['added_tokens'])} eos={meta['eos_token_id']} add_bos={meta['add_bos']}")


if __name__ == "__main__":
    main()
