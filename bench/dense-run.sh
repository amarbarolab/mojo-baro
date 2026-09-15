#!/usr/bin/env bash
# Dense-family per-target verify: BARO_FORCE teacher-forced agreement vs llama.cpp
# on the 20-prompt set, plus no-spec tok/s_gen. bench/dense-protocol.md, PROFILE step 3.
# Usage: bench/dense-run.sh GGUF PACKDIR PROFILEDIR OUTDIR [PORT]
# Run inside gpu-wait; engine and llama-server never overlap.
set -uo pipefail
cd "$(dirname "$0")/.."
GGUF=$1; PACK=$2; PROFILE=$3; OUT=${4:-.work/dense/run}; PORT=${5:-8099}
mkdir -p "$OUT"
BIN="$OUT/spark-engine"
ok() { echo "PASS $1: $2" | tee -a "$OUT/SUMMARY.txt"; }
die() { echo "FAIL $1: $2" | tee -a "$OUT/SUMMARY.txt"; exit 1; }

echo "== build =="
./.venv/bin/mojo build serve/spark.mojo -I . -I kernels -I serve -I "$PROFILE" -o "$BIN" > "$OUT/build.log" 2>&1 \
  || die build "$(grep -m1 error: "$OUT/build.log" | cut -c1-200)"
ok build "$BIN"

echo "== tokenize 20 prompts with our own tokenizer =="
for tf in bench/mtp-prompts/p*.txt; do
  p=$(basename "$tf" .txt)
  ./.work/baro-tokenize encode "$tf" "$GGUF" > "$OUT/$p.ids" 2>"$OUT/$p.tok.log" \
    || die tok "$p: $(cat "$OUT/$p.tok.log")"
done
ok tok "20/20 tokenized"

echo "== llama.cpp: start server (fastest config, native K-quant) =="
~/llama.cpp/build/bin/llama-server -m "$GGUF" -c 4096 -ngl 99 -fa on -ctk f16 -ctv f16 \
  -b 2048 -ub 512 -t 8 --host 127.0.0.1 --port "$PORT" > "$OUT/llama-server.log" 2>&1 &
LPID=$!
cleanup_llama() { kill "$LPID" 2>/dev/null; wait "$LPID" 2>/dev/null; }
trap cleanup_llama EXIT
up=0
for _ in $(seq 1 120); do
  curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && { up=1; break; }
  kill -0 "$LPID" 2>/dev/null || die llama "server exited early: $(tail -5 "$OUT/llama-server.log")"
  sleep 1
done
[ "$up" = 1 ] || die llama "server never came up"
curl -s "http://127.0.0.1:$PORT/props" > "$OUT/llama-props.json"
ok llama "up"

echo "== per-prompt: llama greedy (reference), ours no-spec (tok/s), ours forced (agreement) =="
echo "prompt n_ids tok_s_gen forced_agreement" > "$OUT/results.txt"
for tf in bench/mtp-prompts/p*.txt; do
  p=$(basename "$tf" .txt)
  ids=$(tr -s '\n' ',' < "$OUT/$p.ids" | sed 's/,$//')
  resp=$(curl -s "http://127.0.0.1:$PORT/completion" -H 'Content-Type: application/json' \
    -d "{\"prompt\": [$ids], \"n_predict\": 64, \"temperature\": 0, \"top_k\": 1, \"cache_prompt\": false, \"return_tokens\": true}")
  echo "$resp" | python3 -c 'import sys,json; d=json.load(sys.stdin); print(" ".join(str(t) for t in d["tokens"]))' \
    > "$OUT/$p.llama-tokens.txt" || die run "$p: bad /completion response: $resp"

  env BARO_PROMPT="$OUT/$p.ids" BARO_PACK="$PACK" BARO_GEN=64 BARO_SPEC=0 "$BIN" > "$OUT/$p.ours-nospec.log" 2>&1 \
    || die run "$p: engine exit $?"
  ts=$(grep -oE 'tok/s_gen: [0-9.]+' "$OUT/$p.ours-nospec.log" | cut -d' ' -f2)

  env BARO_PROMPT="$OUT/$p.ids" BARO_PACK="$PACK" BARO_GEN=64 BARO_SPEC=0 BARO_FORCE="$OUT/$p.llama-tokens.txt" "$BIN" \
    > "$OUT/$p.ours-forced.log" 2>&1 || die run "$p: forced engine exit $?"
  agree=$(grep '^forced agreement:' "$OUT/$p.ours-forced.log" | sed 's/forced agreement: //; s/ \/ /\//')

  echo "$p $(wc -l < "$OUT/$p.ids") $ts $agree" >> "$OUT/results.txt"
done
cleanup_llama; trap - EXIT
ok run "$(cat "$OUT/results.txt")"
echo "== done, see $OUT/SUMMARY.txt =="
