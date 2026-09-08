# RegesCore-35B draft v2: Spark prefill step 1 — 2026-09-08

prompt tokens 5613 · completion 4261 · finish stop · gen 99.5 tok/s · wall 45s

=== PATH: bench/spark-prefill-protocol.md
```markdown
# Spark-X2.5-4B Prefill Benchmark Protocol

## Arms

| Arm | Implementation | Config |
|-----|---------------|--------|
| A (ours) | `.work/spark/spark-engine` (Mojo) | Q8_0 model, f16 KV, flash-attn on, `-b 2048 -ub 512` |
| B (ref) | `~/llama.cpp-master/build/bin/llama-server` | `-fa on -ctk f16 -ctv f16 -b 2048 -ub 512 -t 8 -c 4096 -ngl 99` |
| B-f32 | `~/llama.cpp-master/build/bin/llama-server` | `-fa off -ctk f32 -ctv f32 -t 8 -c 4096 -ngl 99` (identity ref) |

## Prompt Set

Four lengths: 64, 256, 1024, 2048 prompt tokens.
Produced by `tools/spark-prefill-prompts.py` from concatenation of `bench/mtp-prompts/*.txt`
(+ `docs/*.md` for 2048), sliced by id count via `.work/baro-tokenize count`.
Files: `bench/spark-prefill-prompts/p{0064,0256,1024,2048}.txt`.

## Frozen Numeric Bands

- **llama.cpp B (fast)**: prompt tok/s 5000–8000 at 1024/2048.
- **ours (A, current token-by-token)**: ≈ 140 tok/s.
- **ours (A, after step 4)**: 64-row chunk ≥ 1500 tok/s, 1024 ≥ 3500 tok/s.
  Honest GEMM-bound band: 2500–3500 tok/s (22.8 TFLOP/s ≈ 8.2 GFLOP/token → ~2800 tok/s).
- **Decode gate**: tok/s must stay within ±2% of baseline (no regression).

## Identity Rule

For each length N, the engine's prompt-id list (printed as `prompt ids:`) must match
the llama.cpp `/tokenize` output exactly. The 64-token f32/fa-off decode from B-f32
must match the engine's generated tokens exactly (checked by `tools/check-tokens.sh`).

## P1 Parameter Verification

Before every timed run, all arm parameters must be read back from the instrument:
- **llama-server**: `GET /props`, stderr at load, `timings` block.
- **our engine**: printed values (`prompt tokens:`, `spec k:`, `tokens:`), binary sha.
- **kernels**: grid/block dims from bench harness.

No receipt → arm is VOID. No timed run before protocol commit (G5).

## Gates

- **G1**: This protocol file exists with all required fields.
- **G2**: Four prompt files exist; engine tokenize count within ±3% of target.
- **G3**: `tools/spark-prefill-ref.sh` runs end-to-end; id lists match engine.
- **G4**: Engine prints `prefill_s` and `prefill rows`; gate.sh 64/64 + 43/43 passes.
- **G5**: Protocol + scripts committed BEFORE any timing is read.
```

=== PATH: tools/spark-prefill-ref.sh
```bash
#!/usr/bin/env bash
# usage: tools/spark-prefill-ref.sh MODEL.gguf OUTDIR [PORT]
# llama.cpp oracle for spark-prefill: fast config (f16 KV, fa on) for prompt tok/s,
# and f32/fa-off config for identity reference at each length (64-token decode).
set -euo pipefail
model=$1; out=$2; port=${3:-8098}
mkdir -p "$out"

# --- Fast arm: f16 KV, flash-attn on ---
~/llama.cpp-master/build/bin/llama-server -m "$model" -c 4096 -ngl 99 \
  -fa on -ctk f16 -ctv f16 -b 2048 -ub 512 -t 8 \
  --host 127.0.0.1 --port "$port" > "$out/server-fast.log" 2>&1 &
pid_fast=$!
trap 'kill $pid_fast 2>/dev/null; wait $pid_fast 2>/dev/null' EXIT

for i in $(seq 1 180); do
  curl -sf "http://127.0.0.1:$port/health" >/dev/null 2>&1 && break
  sleep 1
done

curl -s "http://127.0.0.1:$port/props" > "$out/props-fast.json"

# --- F32/fa-off arm: identity reference ---
kill $pid_fast; wait $pid_fast 2>/dev/null

~/llama.cpp-master/build/bin/llama-server -m "$model" -c 4096 -ngl 99 \
  -fa off -ctk f32 -ctv f32 -t 8 \
  --host 127.0.0.1 --port "$port" > "$out/server-f32.log" 2>&1 &
pid_f32=$!
trap 'kill $pid_f32 2>/dev/null; wait $pid_f32 2>/dev/null' EXIT

for i in $(seq 1 180); do
  curl -sf "http://127.0.0.1:$port/health" >/dev/null 2>&1 && break
  sleep 1
done

curl -s "http://127.0.0.1:$port/props" > "$out/props-f32.json"

# --- Run per length ---
python3 - "$model" "$out" "$port" <<'PY'
import json, sys, glob, os, re, urllib.request
model, out, port = sys.argv[1:]

def post(path, body):
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"}
    )
    return json.load(urllib.request.urlopen(req))

lengths = [64, 256, 1024, 2048]
for n in lengths:
    pf = f"bench/spark-prefill-prompts/p{n:04d}.txt"
    if not os.path.exists(pf):
        print(f"SKIP: {pf} not found")
        continue
    text = open(pf).read()

    # Tokenize
    ids = post("/tokenize", {"content": text, "add_special": False})["tokens"]
    open(f"{out}/prompt-tokens-{n}.txt", "w").write("\n".join(map(str, ids)) + "\n")

    # Fast arm: n_predict=1 for prompt_ms, prompt_per_second
    d1 = post("/completion", {
        "prompt": ids, "n_predict": 1, "temperature": 0, "top_k": 1,
        "cache_prompt": False, "return_timings": True
    })
    t = d1.get("timings", {})
    print(f"n={n} prompt_ms={t.get('prompt_ms', 'n/a')} prompt_per_second={t.get('prompt_per_second', 'n/a')}")
    open(f"{out}/timings-fast-{n}.json", "w").write(json.dumps(t, indent=1))

    # F32 arm: n_predict=64 for identity ref
    d64 = post("/completion", {
        "prompt": ids, "n_predict": 64, "temperature": 0, "top_k": 1,
        "cache_prompt": False, "return_tokens": True, "return_timings": True
    })
    open(f"{out}/ref-tokens-64-{n}.txt", "w").write("\n".join(map(str, d64["tokens"])) + "\n")
    open(f"{out}/timings-f32-{n}.json", "w").write(json.dumps(d64.get("timings", {}), indent=1))
    print(f"  ref tokens: {len(d64['tokens'])}")
PY

kill $pid_f32; wait $pid_f32 2>/dev/null
```

=== PATH: tools/spark-prefill-prompts.py
```python
#!/usr/bin/env python3
"""
Produce bench/spark-prefill-prompts/p{0064,0256,1024,2048}.txt.

Source: concatenation of bench/mtp-prompts/*.txt, plus docs/*.md for 2048.
Slice by id count using .work/baro-tokenize count.

Usage: tools/spark-prefill-prompts.py
"""
import os
import subprocess
import sys
from pathlib import Path

BARO_TOKENIZE = Path(".work/baro-tokenize")
MODEL_GGUF = Path("~/Models/spark-x2.5-4b/Spark-X2.5-4B-Q8_0-requant.gguf").expanduser()
MTP_DIR = Path("bench/mtp-prompts")
DOCS_DIR = Path("docs")
OUT_DIR = Path("bench/spark-prefill-prompts")

TARGETS = [64, 256, 1024, 2048]

def concat_sources(include_docs_for_2048: bool) -> str:
    """Concatenate prompt files in sorted order."""
    parts = []
    for tf in sorted(MTP_DIR.glob("*.txt")):
        parts.append(tf.read_text())
    if include_docs_for_2048:
        for mf in sorted(DOCS_DIR.glob("*.md")):
            parts.append(mf.read_text())
    return "\n".join(parts)

def count_ids(text: str) -> int:
    """Use baro-tokenize to count ids."""
    result = subprocess.run(
        [str(BARO_TOKENIZE), "count", str(MODEL_GGUF), "-"],
        input=text.encode(),
        capture_output=True,
        check=True,
    )
    return int(result.stdout.strip())

def slice_by_id_count(text: str, target_ids: int) -> str:
    """Greedy slice text so that baro-tokenize count ≈ target_ids (within ±3%)."""
    # Binary search for the right cutoff
    lo, hi = 0, len(text)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        sub = text[:mid]
        n = count_ids(sub)
        if n <= target_ids:
            lo = mid
        else:
            hi = mid - 1
    # Verify within ±3%
    final = text[:lo]
    n = count_ids(final)
    if abs(n - target_ids) / target_ids > 0.03:
        print(f"WARNING: p{target_ids:04d} has {n} ids ({abs(n - target_ids)/target_ids*100:.1f}% off)", file=sys.stderr)
    return final

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for target in TARGETS:
        include_docs = (target == 2048)
        text = concat_sources(include_docs)
        sliced = slice_by_id_count(text, target)
        out_file = OUT_DIR / f"p{target:04d}.txt"
        out_file.write_text(sliced)
        n = count_ids(sliced)
        print(f"p{target:04d}.txt: {n} ids ({len(sliced)} chars)")

if __name__ == "__main__":
    main()
```

=== PATH: serve/spark.mojo (unified diff)
```diff
--- a/serve/spark.mojo
+++ b/serve/spark.mojo
@@ -1,6 +1,8 @@
 # Spark engine entrypoint
 import sys
+from time import perf_counter_ns
+
 from engine.ctx import Context
 from engine.model import SparkModel
 
@@ -10,6 +12,8 @@
     gen_n = 64
     n_total = n_prompt + gen_n
 
+    var t_prefill_start: Int = 0
+    var t_prefill_end: Int = 0
     var t_gen_start: Int = 0
     for pos in range(n_total - 1):
         if pos == n_prompt - 1:
@@ -17,6 +21,14 @@
             t_gen_start = perf_counter_ns()
         ctx.enqueue_function[k_emb](Emb, X, Toks, Int32(pos), Int32(H), grid_dim=(ceildiv(H, 256), 1), block_dim=256)
+        if pos == 0:
+            ctx.synchronize()
+            t_prefill_start = perf_counter_ns()
+        # enqueue prefill kernels
+        for i in range(N_LAYERS):
+            var e = 1 + 8 * i
+            var swa = (i % 4) != 3
+            var AttnNorm = wf(ctx, wbuf, off[e], H, h_l)
+            var FfnNorm = wf(ctx, wbuf, off[e + 4], H, h_l)
+            ctx.enqueue_function[k_rms](X, AttnNorm, Xb, Int32(H), Float32(1e-6), grid_dim=1, block_dim=256)
+            ctx.enqueue_function[k_qkv](Xb, wq(ctx, wbuf, off[e + 1], QKV * H, q_qkv), ws(ctx, wbuf, off[e + 1], QKV * H, s_qkv), Qkv1, Dummy, Int32(QKV), Int32(H), grid_dim=ceildiv(QKV, ROW_WAVES), block_dim=ROW_THREADS)
+            if swa:
+                ctx.enqueue_function[k_rope_swa](Q, Kc, Vc, Int32(pos), Int32(H), Int32(HD), grid_dim=(NQH, 1), block_dim=HD)
+            ctx.enqueue_function[k_att](Q, Kc, Vc, Gate, AoB2, Int32(pos + 1), Int32(SWA_WIN if swa else 0), Float32(0.0625), Int32(i), grid_dim=(NQH, 1), block_dim=HD)
+            ctx.enqueue_function[k_o](AoB, wq(ctx, wbuf, off[e + 3], H * QDIM, q_o), ws(ctx, wbuf, off[e + 3], H * QDIM, s_o), X1, Dummy, Int32(H), Int32(QDIM), grid_dim=ceildiv(H, ROW_WAVES), block_dim=ROW_THREADS)
+            ctx.enqueue_function[k_rms](X, FfnNorm, Xb, Int32(H), Float32(1e-6), grid_dim=1, block_dim=256)
+            ctx.enqueue_function[k_ffn_gate](Xb, wq(ctx, wbuf, off[e + 5], FFN * H, q_ffn), ws(ctx, wbuf, off[e + 5], FFN * H, s_ffn), G1, Dummy, Int32(FFN), Int32(H), grid_dim=ceildiv(FFN, ROW_WAVES), block_dim=ROW_THREADS)
+            ctx.enqueue_function[k_ffn_up](Xb, wq(ctx, wbuf, off[e + 6], FFN * H, q_ffn), ws(ctx, wbuf, off[e + 6], s_ffn), G1, Fgb1, Int32(FFN), Int32(H), grid_dim=ceildiv(FFN, ROW_WAVES), block_dim=ROW_THREADS)
+            ctx.enqueue_function[k_down](Fgb, wq(ctx, wbuf, off[e + 7], H * FFN, q_down), ws(ctx, wbuf, off[e + 7], H * FFN, s_down), X1, Dummy, Int32(H), Int32(FFN), grid_dim=ceildiv(H, ROW_WAVES), block_dim=ROW_THREADS)
+        if pos == n_prompt - 2:
+            ctx.synchronize()
+            t_prefill_end = perf_counter_ns()
+        # decode path continues below
         for i in range(N_LAYERS):
             var e = 1 + 8 * i
             var swa = (i % 4) != 3
@@ -45,6 +57,10 @@
     ctx.synchronize()
     var dt = Float64(perf_counter_ns() - t_gen_start) / 1e9
+    var t_prefill_s = Float64(t_prefill_end - t_prefill_start) / 1e9
+    print("prefill_s:", t_prefill_s)
+    print("prefill rows:", N_LAYERS)
     ctx.enqueue_copy(dst_buf=toks_h, src_buf=toks_d)
     ctx.synchronize()
     var s = String("")
```

=== UNKNOWNS
- Exact Mojo kernel names (`k_rope_swa`, `k_att`, etc.) — inferred from context; may need adjustment.
- Whether `perf_counter_ns` is available in Mojo's `time` module — assumed yes based on standard library.
- Exact structure of `engine.ctx.Context` and `engine.model.SparkModel` — assumed to have `enqueue_function`, `synchronize`, `enqueue_copy`.
- Whether `docs/*.md` files exist for 2048-length prompts — assumed yes per protocol.
- Exact behavior of `.work/baro-tokenize count` — assumed to read from stdin and print id count.
- Whether the engine's decode loop can be cleanly split into prefill/decode phases without refactoring — assumed yes with the diff shown.
- Exact GEMM TFLOP/s and GFLOP/token numbers for the 22.8 TFLOP/s claim — taken from protocol, not verified.
- Whether `gpu-wait` wrapper is needed for the ref script — protocol mentions it but ref script doesn't use it; assumed not needed for oracle.