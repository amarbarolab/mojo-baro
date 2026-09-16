#!/usr/bin/env python3
"""Perplexity, ours arm. bench/quality-protocol.md. Drives serve/engine.mojo
or serve/spark.mojo directly over its BARO_SERVE=1 stdin/stdout line
protocol (serve/PROTOCOL.md), one chunk request at a time, teacher-forced
(per-request "force") with top_logprobs:1 to route decode through the
sampling branch that dumps the full pre-penalty logits row
(BARO_DUMP_LOGITS_DIR, window.mojo:1534). Never pipes all requests at once:
each request's dump rows are read and the dump dir cleared before the next
request is written, because BARO_DUMP_LOGITS_DIR is one fixed directory for
the whole resident process and every request's row numbering restarts at 0.
"""
import argparse
import json
import math
import os
import shutil
import struct
import subprocess
import sys
import time


def read_ids(path):
    return [int(x) for x in open(path).read().split()]


def log_softmax_at(row_path, ref_id):
    with open(row_path, "rb") as f:
        data = f.read()
    n = len(data) // 4
    vals = struct.unpack(f"<{n}f", data)
    m = max(vals)
    lse = m + math.log(sum(math.exp(v - m) for v in vals))
    return vals[ref_id] - lse, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True)
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--pack", required=True)
    ap.add_argument("--text", required=True)
    ap.add_argument("--ctx", type=int, default=512)
    ap.add_argument("--chunks", type=int, default=8)
    ap.add_argument("--tokenize-bin", default=".work/baro-tokenize")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    dumpdir = os.path.join(args.out, "dump")
    os.makedirs(dumpdir, exist_ok=True)

    ids_path = os.path.join(args.out, "text.ids")
    with open(ids_path, "w") as f:
        subprocess.run([args.tokenize_bin, "encode", args.text, args.gguf], stdout=f, check=True)
    ids = read_ids(ids_path)
    need = args.ctx * args.chunks
    if len(ids) < need:
        print(f"VOID: text tokenizes to {len(ids)} ids, need {need} for {args.chunks}x{args.ctx}", file=sys.stderr)
        sys.exit(3)
    ids = ids[:need]
    chunks = [ids[i * args.ctx:(i + 1) * args.ctx] for i in range(args.chunks)]

    env = dict(os.environ)
    env.update(BARO_SERVE="1", BARO_SPEC="0", BARO_PACK=args.pack, BARO_DUMP_LOGITS_DIR=dumpdir)
    proc = subprocess.Popen([args.engine], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                             stderr=open(os.path.join(args.out, "engine.err"), "w"),
                             env=env, text=True, bufsize=1)

    readback = []
    ready = False
    while True:
        line = proc.stdout.readline()
        if not line:
            print("VOID: engine exited before ready line", file=sys.stderr)
            sys.exit(4)
        line = line.strip()
        if line:
            readback.append(line)
        if line.startswith('{"ready":true'):
            ready = True
            break
    assert ready

    total_nll = 0.0
    total_n = 0
    vocab = None
    per_chunk = []
    for i, chunk in enumerate(chunks):
        req = {"id": i, "prompt": [chunk[0]], "n": args.ctx - 1, "spec": False,
               "top_logprobs": 1, "force": chunk[1:]}
        proc.stdin.write(json.dumps(req) + "\n")
        proc.stdin.flush()
        done = None
        req_lines = []
        while True:
            line = proc.stdout.readline()
            if not line:
                print(f"VOID: engine exited mid chunk {i}", file=sys.stderr)
                sys.exit(5)
            line = line.strip()
            if not line:
                continue
            req_lines.append(line)
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if d.get("id") == i and d.get("done"):
                done = d
                break
        rows = sorted(
            (fn for fn in os.listdir(dumpdir) if fn.startswith("row-") and fn.endswith(".bin")),
            key=lambda fn: int(fn[4:-4]),
        )
        expect = args.ctx - 1
        if len(rows) != expect:
            print(f"VOID: chunk {i} dumped {len(rows)} rows, expected {expect}", file=sys.stderr)
            sys.exit(6)
        chunk_nll = 0.0
        for hn in range(expect):
            row_path = os.path.join(dumpdir, f"row-{hn}.bin")
            ref_id = chunk[hn + 1]
            lp, n = log_softmax_at(row_path, ref_id)
            if vocab is None:
                vocab = n
            elif vocab != n:
                print(f"VOID: row {hn} vocab {n} != {vocab}", file=sys.stderr)
                sys.exit(7)
            chunk_nll += -lp
        shutil.rmtree(dumpdir)
        os.makedirs(dumpdir, exist_ok=True)
        total_nll += chunk_nll
        total_n += expect
        per_chunk.append({"chunk": i, "nll": chunk_nll, "n": expect, "done": done})
        print(f"chunk {i}/{args.chunks}: nll={chunk_nll:.3f} n={expect}", file=sys.stderr)

    proc.stdin.close()
    proc.wait(timeout=30)

    ppl = math.exp(total_nll / total_n)
    result = {
        "engine": args.engine,
        "pack": args.pack,
        "gguf": args.gguf,
        "ctx": args.ctx,
        "chunks": args.chunks,
        "token_count": len(ids),
        "vocab": vocab,
        "total_nll": total_nll,
        "total_n": total_n,
        "ppl": ppl,
        "per_chunk": per_chunk,
        "readback": readback,
    }
    with open(os.path.join(args.out, "ppl-result.json"), "w") as f:
        json.dump(result, f, indent=2)
    print(f"PPL(ours) = {ppl:.4f}  ({total_n} predicted tokens, vocab {vocab})")


if __name__ == "__main__":
    main()
