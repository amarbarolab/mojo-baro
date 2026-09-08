#!/usr/bin/env python3
"""Gate for the Mojo tokenizer (serve/tokenizer.mojo via .work/baro-tokenize).

Same cases as tools/test_tokenizer.py (bench/mtp-prompts .tokens files + the
hard set + rendered chat template + empty string), reference = llama-tokenize on
the source GGUF; plus decode(encode(x)) == x. Exit 1 on any mismatch.
Usage: tools/test_tokenizer_mojo.py [--gguf G] [--llama-tokenize B] [--cli .work/baro-tokenize]
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import test_tokenizer as tt  # noqa: E402
gt = tt.gt


def ours_batch(cli, gguf, texts):
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as f:
        f.write("\0".join(texts))
        path = f.name
    r = subprocess.run([cli, "batch", path, gguf], capture_output=True, text=True, encoding="utf-8")
    if r.returncode != 0:
        raise SystemExit(f"cli failed: {r.stderr[-800:]}")
    return [[int(x) for x in line.split()] for line in r.stdout.split("\n")[: len(texts)]]


def ours_decode(cli, gguf, ids):
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        f.write(" ".join(map(str, ids)))
        path = f.name
    r = subprocess.run([cli, "decode-keep", path, gguf], capture_output=True)
    return r.stdout.decode("utf-8", errors="surrogateescape")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", default=str(ROOT / ".work/engine-pack-q4"))
    ap.add_argument("--gguf", default=None)
    ap.add_argument("--llama-tokenize", default=os.path.expanduser("~/llama.cpp/build/bin/llama-tokenize"))
    ap.add_argument("--cli", default=str(ROOT / ".work/baro-tokenize"))
    ap.add_argument("--extra", action="append", default=[], metavar="GGUF:TEXT_FILE:IDS_FILE",
                    help="fixed-id case for another model, e.g. Spark ids from llama-server /tokenize")
    a = ap.parse_args()
    meta = gt.load_meta(a.pack)
    gguf = a.gguf or meta["source_gguf"]
    print(f"gguf {gguf}\nref  {a.llama_tokenize}\ncli  {a.cli}")

    cases = []
    prompts = json.loads((ROOT / "bench/mtp-prompts/prompts.json").read_text())
    for stem in prompts:
        txt = (ROOT / "bench/mtp-prompts" / f"{stem}.txt").read_text()
        ref = [int(x) for x in (ROOT / "bench/mtp-prompts" / f"{stem}.tokens").read_text().split()]
        cases.append((f"prompt:{stem}", txt, ref))
    for name, text in tt.HARD_SET.items():
        cases.append((f"hard:{name}", text, tt.encode_ref(a.llama_tokenize, gguf, text)))
    chat = gt.render_chat(meta, [{"role": "user", "content": "What is 2+2?"}])
    cases.append(("chat", chat, tt.encode_ref(a.llama_tokenize, gguf, chat)))
    cases.append(("empty", "", []))

    got = ours_batch(a.cli, gguf, [c[1] for c in cases])
    fails = 0
    for (name, text, ref), g in zip(cases, got):
        if g != ref:
            fails += 1
            i = next((k for k in range(min(len(g), len(ref))) if g[k] != ref[k]), min(len(g), len(ref)))
            print(f"FAIL {name}: first diff at {i}: ref {ref[i:i+5]} got {g[i:i+5]} (len {len(ref)} vs {len(g)})")
        else:
            rt = ours_decode(a.cli, gguf, g)
            if rt != text:
                fails += 1
                print(f"FAIL decode {name}: {rt[:60]!r} != {text[:60]!r}")
    for spec in a.extra:
        g2, tf, idf = spec.split(":")
        text = Path(tf).read_text()
        ref = [int(x) for x in Path(idf).read_text().split()]
        got2 = ours_batch(a.cli, g2, [text])[0]
        ok = got2 == ref and ours_decode(a.cli, g2, got2) == text
        fails += 0 if ok else 1
        print(f"{'ok  ' if ok else 'FAIL'} extra {Path(g2).name}: {len(ref)} ids{'' if ok else f' got {got2[:8]}'}")
        cases.append(("extra", text, ref))
    print(f"{'PASS' if fails == 0 else 'FAIL'}: {len(cases)} cases, {fails} failures")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
