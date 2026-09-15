#!/usr/bin/env python3
"""bench/e14-check.py: E14's identity gate and wall-clock table.

Identity: per follower, KV arm's generated_ids must equal Text arm's
generated_ids, token for token (the frozen gate, briefs/2026-09-15-p3-b2-e14.md
and 06-experiments.md E14) -- any mismatch is a FAIL for that N, not averaged
away. Wall clock: reader_prefill_s + sum(KV receiver_s) vs sum(Text
receiver_s) vs sum(llama.cpp wall_s), for N=3 (first 3 followers) and N=10
(all of them), against the frozen prediction table.

Accepts multiple RAW.json (comma-separated): the 5-minutes-per-arm budget
split the real N=10 run into two 5-follower invocations sharing the same
document, each paying its own reader prefill. Followers are concatenated
across files in the order given; reader_prefill_s used in the wall-clock
totals is the FIRST file's only (one real run pays it once -- the second
file's reader_prefill_s is reported separately, as a receipt that the two
are consistent, never added into a total).

Usage: bench/e14-check.py RAW1.json[,RAW2.json,...] [LLAMA_TIMINGS.jsonl]
"""
import json
import sys

PRED = {
    3: {"kv": 48.81, "text": 136.98, "llama": 45.39},
    10: {"kv": 60.71, "text": 456.6, "llama": 151.3},
}


def main():
    raw_paths = sys.argv[1].split(",")
    llama_path = sys.argv[2] if len(sys.argv) > 2 else None

    docs = [json.loads(open(p).read()) for p in raw_paths]
    followers = []
    for doc in docs:
        followers.extend(doc["followers"])
    reader_prefill_s = docs[0]["reader_prefill_s"]
    if len(docs) > 1:
        others = [f"{doc['reader_prefill_s']:.3f}" for doc in docs[1:]]
        print(f"reader_prefill_s from other files (receipt only, not summed): {', '.join(others)}")

    llama_by_id = {}
    if llama_path:
        for line in open(llama_path):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            # bench/e14-llama-run.sh names rows by prompt filename
            # (prompt-q01), E14's own follower id is q01 -- strip the prefix.
            llama_by_id[row["id"].removeprefix("prompt-")] = row["wall_s"]

    print(f"reader_prefill_s: {reader_prefill_s:.3f}")
    print(f"followers: {len(followers)}")
    print()

    overall_pass = True
    for f in followers:
        kv_ids = f["kv"]["generated_ids"]
        text_ids = f["text"]["generated_ids"]
        match = kv_ids == text_ids
        if not match:
            overall_pass = False
            first_diff = next((i for i, (a, b) in enumerate(zip(kv_ids, text_ids)) if a != b), min(len(kv_ids), len(text_ids)))
            print(f"  {f['id']} ({f['key']}, expected {f['expected']}): IDENTITY MISMATCH at token {first_diff}")
            print(f"    KV:   {kv_ids}")
            print(f"    Text: {text_ids}")
        else:
            print(f"  {f['id']} ({f['key']}, expected {f['expected']}): identity OK, "
                  f"KV receiver_s={f['kv']['receiver_s']:.3f} ingest_s={f['kv']['ingest_s']:.3f} "
                  f"Text receiver_s={f['text']['receiver_s']:.3f}")

    print()
    for n in (3, 10):
        if n > len(followers):
            continue
        subset = followers[:n]
        n_match = sum(1 for f in subset if f["kv"]["generated_ids"] == f["text"]["generated_ids"])
        kv_total = reader_prefill_s + sum(f["kv"]["receiver_s"] for f in subset)
        text_total = sum(f["text"]["receiver_s"] for f in subset)
        llama_total = None
        if llama_by_id:
            ids_have = [f["id"] for f in subset if f["id"] in llama_by_id]
            if len(ids_have) == n:
                llama_total = sum(llama_by_id[f["id"]] for f in subset)

        verdict = "PASS" if n_match == n else "FAIL"
        print(f"N={n}: identity {n_match}/{n} -> {verdict}")
        print(f"  KV total   {kv_total:.2f} s  (predicted {PRED[n]['kv']:.2f} s)")
        print(f"  Text total {text_total:.2f} s  (predicted {PRED[n]['text']:.2f} s)")
        if llama_total is not None:
            print(f"  llama.cpp total {llama_total:.2f} s  (predicted {PRED[n]['llama']:.2f} s)")
            print(f"  KV/Text ratio: {text_total / kv_total:.2f}x faster "
                  f"(predicted {PRED[n]['text'] / PRED[n]['kv']:.2f}x)")
            print(f"  KV/llama.cpp ratio: {llama_total / kv_total:.2f}x "
                  f"({'faster' if kv_total < llama_total else 'SLOWER'}) "
                  f"(predicted {PRED[n]['llama'] / PRED[n]['kv']:.2f}x)")
        else:
            print(f"  KV/Text ratio: {text_total / kv_total:.2f}x faster "
                  f"(predicted {PRED[n]['text'] / PRED[n]['kv']:.2f}x)")
            print("  llama.cpp total: not available")
        print()

    print("OVERALL:", "PASS" if overall_pass else "FAIL")
    sys.exit(0 if overall_pass else 1)


if __name__ == "__main__":
    main()
