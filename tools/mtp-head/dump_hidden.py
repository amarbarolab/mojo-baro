#!/usr/bin/env python3
"""tools/mtp-head/dump_hidden.py: training data for a draft head on a model
that ships none. For N text prompts, saves the model's post-final-norm hidden
state per token (llama-embedding --pooling none, experts on CPU) and the token
ids (llama-tokenize), sharded as OUT/shard-XXX.npz {h: f16 [T,D], ids: i32 [T],
starts: i32 [P]}. Row counts are checked against token counts per shard.

Run through gpu-wait. The embedding JSON is redirected to a file inside the
job: piping it through the gpu-wait wrapper drops bytes (seen 2026-09-19).
Usage: dump_hidden.py MODEL.gguf PROMPTS.txt OUT N_PROMPTS [SHARD=100]
"""
import json, os, subprocess, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np

BIN = Path.home() / "llama.cpp/build/bin"


def tokenize(model, text):
    r = subprocess.run([str(BIN / "llama-tokenize"), "-m", model, "--stdin", "--ids", "--log-disable"],
                       input=text, text=True, capture_output=True, check=True)
    return json.loads(r.stdout)


def main():
    model, prompts, out, n = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3]), int(sys.argv[4])
    shard = int(sys.argv[5]) if len(sys.argv) > 5 else 100
    if not os.environ.get("GPU_WAITING_ROOM_JOB"):
        raise RuntimeError("run through gpu-wait")
    out.mkdir(parents=True, exist_ok=True)
    lines = [l for l in prompts.read_text().split("\n") if l.strip()][:n]
    total = 0
    for s in range(0, len(lines), shard):
        dst = out / f"shard-{s // shard:03d}.npz"
        if dst.exists():
            continue
        part = lines[s:s + shard]
        with ThreadPoolExecutor(8) as ex:
            ids = list(ex.map(lambda t: tokenize(model, t), part))
        (out / "part.txt").write_text("\n".join(part))
        with (out / "part.json").open("w") as so, (out / "part.err").open("w") as se:
            subprocess.run([str(BIN / "llama-embedding"), "-m", model, "-f", str(out / "part.txt"),
                            "--pooling", "none", "--embd-normalize", "-1", "--embd-output-format", "json",
                            "-ngl", "99", "--cpu-moe", "-c", "2048", "-b", "2048", "-ub", "2048",
                            "-t", "12", "--log-verbosity", "0"], stdout=so, stderr=se, check=True)
        raw = (out / "part.json").read_text()
        rows = json.loads(raw[raw.index("{"):])["data"]
        want = sum(map(len, ids))
        if len(rows) != want:
            raise RuntimeError(f"shard {s}: {len(rows)} hidden rows for {want} tokens")
        h = np.array([r["embedding"] for r in rows], dtype=np.float16)
        starts = np.cumsum([0] + [len(i) for i in ids[:-1]]).astype(np.int32)
        np.savez(dst, h=h, ids=np.concatenate(ids).astype(np.int32), starts=starts)
        (out / "part.json").unlink()
        total += want
        print(f"{dst.name}: {want} tokens", flush=True)
    print(f"done, {total} new tokens")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"FAIL dump_hidden: {error}", file=sys.stderr)
        sys.exit(1)
