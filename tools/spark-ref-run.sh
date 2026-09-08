#!/usr/bin/env bash
# usage: tools/spark-ref-run.sh MODEL.gguf OUTDIR [PORT] [PROMPT_TEXT_FILE]
# llama.cpp master oracle for the spark2_5 engine: tokenizes the prompt text,
# writes prompt-tokens.txt (ids the engine consumes), then greedy-decodes 64
# tokens from the SAME ids with f32 KV and no flash-attn (closest to the
# engine's math) and writes ref-tokens-64.txt + props.json + timings.json.
set -euo pipefail
model=$1; out=$2; port=${3:-8098}; ptxt=${4:-bench/spark-prompt.txt}
mkdir -p "$out"
~/llama.cpp-master/build/bin/llama-server -m "$model" -c 4096 -ngl 99 -fa off -ctk f32 -ctv f32 -t 8 \
  --host 127.0.0.1 --port "$port" > "$out/server.log" 2>&1 &
pid=$!
trap 'kill $pid 2>/dev/null; wait $pid 2>/dev/null' EXIT
for i in $(seq 1 180); do curl -sf "http://127.0.0.1:$port/health" >/dev/null 2>&1 && break; sleep 1; done
curl -s "http://127.0.0.1:$port/props" > "$out/props.json"
python3 - "$ptxt" "$port" "$out" <<'PY'
import json, sys, urllib.request
ptxt, port, out = sys.argv[1:]
text = open(ptxt).read()
req = urllib.request.Request(f"http://127.0.0.1:{port}/tokenize", data=json.dumps({"content": text, "add_special": False}).encode(), headers={"Content-Type": "application/json"})
ids = json.load(urllib.request.urlopen(req))["tokens"]
open(f"{out}/prompt-tokens.txt", "w").write("\n".join(map(str, ids)) + "\n")
req = urllib.request.Request(f"http://127.0.0.1:{port}/completion", data=json.dumps({"prompt": ids, "n_predict": 64, "temperature": 0, "top_k": 1, "cache_prompt": False, "return_tokens": True}).encode(), headers={"Content-Type": "application/json"})
d = json.load(urllib.request.urlopen(req))
open(f"{out}/ref-tokens-64.txt", "w").write("\n".join(map(str, d["tokens"])) + "\n")
json.dump(d.get("timings", {}), open(f"{out}/timings.json", "w"), indent=1)
print("prompt ids:", len(ids), "gen:", len(d["tokens"]), "tg tok/s:", d.get("timings", {}).get("predicted_per_second"))
print(repr(d["content"][:200]))
PY
