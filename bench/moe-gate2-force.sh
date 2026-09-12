#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
model=${1:-$HOME/Models/RegesCore-1.0-35/RegesCore-1.0-35B-UD-Q4_K_S-BARO.gguf}
engine=${2:-.work/moe-engine-g2}
out=${3:-.work/moe-w3/gate2-force}
port=${4:-18083}
mkdir -p "$out"
llama_bin="$HOME/llama.cpp/build/bin/llama-server"
for arm in "$llama_bin" "$engine"; do
    [ -x "$arm" ] || { echo "missing executable arm: $arm" >&2; exit 2; }
done
llama_hash=$(sha256sum "$llama_bin" | cut -d' ' -f1)
engine_hash=$(sha256sum "$engine" | cut -d' ' -f1)
[ "$llama_hash" != "$engine_hash" ] || { echo "REFUSED: arm hashes are equal" >&2; exit 2; }
echo "llama=$llama_bin shaLlama=$llama_hash engine=$engine shaEngine=$engine_hash" | tee "$out/arm.txt"
"$llama_bin" -m "$model" -c 4096 -ngl 99 -fa on -ctk f16 -ctv f16 -b 2048 -ub 512 -t 8 -np 1 --no-cont-batching --host 127.0.0.1 --port "$port" > "$out/llama.log" 2>&1 &
pid=$!
cleanup() { kill "$pid" 2>/dev/null || true; wait "$pid" 2>/dev/null || true; }
trap cleanup EXIT
for _ in $(seq 1 180); do
    curl -sf "http://127.0.0.1:$port/health" >/dev/null 2>&1 && break
    kill -0 "$pid" 2>/dev/null || { tail -20 "$out/llama.log"; exit 1; }
    sleep 1
done
curl -sf "http://127.0.0.1:$port/props" > "$out/props.json"
for tf in bench/mtp-prompts/p*.tokens; do
    p=$(basename "$tf" .tokens)
    ids=$(tr -s ' \n' ',,' < "$tf" | sed 's/,$//')
    curl -sf "http://127.0.0.1:$port/completion" -H 'Content-Type: application/json' \
        -d "{\"prompt\": [$ids], \"n_predict\": 128, \"temperature\": 0, \"top_k\": 1, \"seed\": 1, \"ignore_eos\": true, \"cache_prompt\": false, \"return_tokens\": true}" \
        | python3 -c 'import json,sys; d=json.load(sys.stdin); n=d["timings"]["prompt_n"]; ids=d["tokens"][n:n+64]; assert len(ids) == 64, (n, len(d["tokens"])); print(" ".join(map(str,ids)))' > "$out/$p.ref.ids"
done
cleanup
trap - EXIT
for tf in bench/mtp-prompts/p*.tokens; do
    p=$(basename "$tf" .tokens)
    BARO_MEGA=0 BARO_PACK=.work/moe-w1/pack BARO_PROMPT="$tf" BARO_FORCE="$out/$p.ref.ids" "$engine" > "$out/$p.cand.log" 2>&1
    grep -q '^forced agreement: 64 / 64' "$out/$p.cand.log"
done
echo "20/20 prompts, 64/64 forced agreement"
