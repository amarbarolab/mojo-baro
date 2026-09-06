#!/usr/bin/env python3
"""Tokenizer identity gate: <pack>/tokenizer.json must match llama.cpp bit for bit.

Three checks, exit non-zero on any mismatch:
  1. every bench/mtp-prompts/*.txt encodes to its .tokens file (llama.cpp ref)
  2. a hard set (unicode, CJK, emoji, code, whitespace, numbers, special tokens,
     the rendered chat template, empty string) encodes to what llama-tokenize
     prints on the pack's source GGUF
  3. decode(encode(x)) == x on every string above

Needs the GGUF (read from tokenizer-meta.json "source_gguf" unless --gguf) and
~/llama.cpp/build/bin/llama-tokenize (CPU, vocab only), so it is not part of
tools/ci-checks.sh. Run: .venv/bin/python tools/test_tokenizer.py [--pack DIR]
"""
import argparse
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("gt", ROOT / "tools/gguf-tokenizer.py")
gt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gt)

HARD_SET = {
    "empty": "",
    "ascii": "The quick brown fox jumps over the lazy dog.",
    "contractions": "I'm sure they've seen it, but we'll see; IT'S NOT what you'd think. Don't.",
    "leading-space": " hello world",
    "trailing-space": "hello world ",
    "many-spaces": "   three   spaces   everywhere   ",
    "tabs-code": "def f(x):\n\tif x:\n\t\treturn [1, 2, 3]\n\treturn {}\n",
    "spaces-code": "class A:\n    def __init__(self, n=0):\n        self.n = n  # count\n",
    "newlines": "line one\n\nline three\n\n\n\nline seven\n",
    "crlf": "windows\r\nline endings\r\n",
    "mixed-ws": "  \n  \t \n x \t\t y\n",
    "nbsp-zwj": "a\u00a0b\u200bc\u200dd",
    "numbers": "1234567890 3.14159 -42 1,000,000 2026-09-06 0x1F 1e-9 ½ ²",
    "unicode-latin": "Ça va? Señor, façade, naïve, Æsir, Ørsted, Łódź, Ünïcödé",
    "unicode-nfd": "cafe\u0301 e\u0301 n\u0303 A\u030a",
    "cyrillic-greek": "Привет, мир! Καλημέρα κόσμε. Ёлка.",
    "arabic-hebrew": "مرحبا بالعالم שלום עולם",
    "devanagari-thai": "नमस्ते दुनिया สวัสดีชาวโลก",
    "cjk-zh": "你好，世界。今天天气很好。",
    "cjk-ja": "こんにちは世界。カタカナとひらがなと漢字。",
    "cjk-ko": "안녕하세요 세계. 한국어 문장입니다.",
    "cjk-mixed-ascii": "GPU显存24GB, 温度85°C。",
    "emoji": "hello 👋 world 🌍🚀 🇩🇪 👨‍👩‍👧‍👦 👍🏽 ❤️ 😂😂😂",
    "symbols": "∑ ∫ ≠ ≈ → ← ∞ © ® ™ € £ ¥ § ¶ • …",
    "punct-runs": "!!! ??? ... --- *** ((( ))) [[[ ]]] {{{ }}} <<< >>> ::: ;;; ,,,",
    "url-email": "See https://example.com/a/b?x=1&y=2#frag or mail root@example.org.",
    "markdown": "# Title\n\n- item **bold** _it_ `code`\n\n| a | b |\n|---|---|\n| 1 | 2 |\n",
    "json": '{"key": "value", "n": [1, 2.5, -3], "nested": {"ok": true, "x": null}}',
    "special-chat": "<|im_start|>system\nYou are helpful.<|im_end|>\n<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n",
    "special-eot": "<|endoftext|>text after eot<|endoftext|>",
    "special-added": "<think>\nthinking\n</think>\n<tool_call>{}</tool_call><tool_response>r</tool_response>",
    "special-vision": "<|vision_start|><|image_pad|><|vision_end|> <|fim_prefix|>a<|fim_suffix|>b<|fim_middle|>",
    "special-fake": "<|not_a_token|> <| im_start |> <|im_start",
    "special-adjacent": "a<|im_end|>b <|im_end|> c\n<|im_end|>\n",
    "bytelevel-chars": "Ġ Ċ ĉ ā Ā",
    "long-repeat": "ab" * 200 + " " + "x" * 300,
    "long-ws": " " * 100 + "\n" * 50 + "\t" * 30,
    "apostrophes": "rock'n'roll 'quoted' don't ‘curly’ “double”",
    "hyphen-dash": "well-known — em-dash – en-dash ‐ hyphen",
    "control-chars": "bell\x07 esc\x1b[0m nul-free \x7f",
}


def encode_ref(llama_tokenize, gguf, text):
    if text == "":
        return []  # llama-tokenize refuses an empty prompt; add_bos=false means []
    r = subprocess.run([llama_tokenize, "-m", gguf, "--ids", "-p", text],
                       capture_output=True, text=True, check=True)
    line = [l for l in r.stdout.splitlines() if l.startswith("[")][-1]
    return json.loads(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", default=str(ROOT / ".work/engine-pack-q4"))
    ap.add_argument("--gguf", default=None)
    ap.add_argument("--llama-tokenize", default=os.path.expanduser("~/llama.cpp/build/bin/llama-tokenize"))
    ap.add_argument("--dump", default=None, help="write every case's ids to this JSON file")
    a = ap.parse_args()

    meta = gt.load_meta(a.pack)
    tok = gt.load_tokenizer(a.pack)
    gguf = a.gguf or meta["source_gguf"]
    for p in (gguf, a.llama_tokenize):
        if not Path(p).exists():
            print(f"FAIL: missing {p}"); return 1
    print(f"pack {a.pack}: {meta['model']}/{meta['pre']} vocab={meta['vocab_size']} add_bos={meta['add_bos']}")
    print(f"ref  {gguf} via {a.llama_tokenize}")

    fails, n, dump = 0, 0, {}

    def check(name, text, ref):
        nonlocal fails, n
        n += 1
        got = tok.encode(text, add_special_tokens=False).ids
        dump[name] = {"text": text, "ids": got}
        if got != ref:
            fails += 1
            i = next((i for i, (x, y) in enumerate(zip(got, ref)) if x != y), min(len(got), len(ref)))
            print(f"  FAIL encode {name}: first diff at {i}: ours={got[i:i+6]} ref={ref[i:i+6]} (len {len(got)} vs {len(ref)})")
            return
        back = tok.decode(got, skip_special_tokens=False)
        if back != text:
            fails += 1
            print(f"  FAIL roundtrip {name}: {back!r} != {text!r}")

    print("== bench/mtp-prompts (ref = .tokens files)")
    prompts = json.loads((ROOT / "bench/mtp-prompts/prompts.json").read_text())
    for name, text in prompts.items():
        txt = (ROOT / f"bench/mtp-prompts/{name}.txt").read_text()
        if txt != text:
            fails += 1; print(f"  FAIL {name}.txt differs from prompts.json")
        ref = [int(x) for x in (ROOT / f"bench/mtp-prompts/{name}.tokens").read_text().split()]
        check(name, txt, ref)
    print(f"  {len(prompts)} prompts, {fails} failures")

    print("== hard set (ref = llama-tokenize)")
    hard = dict(HARD_SET)
    if meta.get("chat_template"):
        hard["chat-template"] = gt.render_chat(meta, [
            {"role": "system", "content": "You are a terse assistant."},
            {"role": "user", "content": "Wie spät ist es? 🕒"},
            {"role": "assistant", "content": "Keine Ahnung."},
            {"role": "user", "content": "Ok, 2+2?"}])
    before = fails
    for name, text in hard.items():
        check(name, text, encode_ref(a.llama_tokenize, gguf, text))
    print(f"  {len(hard)} cases, {fails - before} failures")

    if a.dump:
        Path(a.dump).write_text(json.dumps(dump, ensure_ascii=False, indent=1))
    print(f"{'PASS' if fails == 0 else 'FAIL'}: {n} cases, {fails} failures")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
