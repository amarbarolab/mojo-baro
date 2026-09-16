#!/usr/bin/env bash
# One model's quality row: build, perplexity (ours + llama.cpp), task eval
# (ours + llama.cpp), score against bench/quality-bands.json, cleanup.
# bench/quality-protocol.md item 3. usage: bench/quality-run.sh KEY
# KEY is a key in bench/quality-models.json / bench/quality-bands.json.
# Run inside gpu-wait; not self-wrapping (the sweep script wraps the whole
# 10-model loop in one gpu-wait job so the GPU is not released between
# models that each hold it for well under its --timeout).
set -uo pipefail
cd "$(dirname "$0")/.."
KEY=$1
PY=$HOME/Projects/mojo/mojo-baro/.venv/bin/python3
WIKI=$HOME/Models/quant-lab/wikitext-2-raw/wiki.test.raw
LLAMA_BIN=$HOME/llama.cpp/build/bin
OUT=.work/quality/$KEY
mkdir -p "$OUT"
ok() { echo "OK $1: $2" | tee -a "$OUT/SUMMARY.txt"; }
die() { echo "FAIL $1: $2" | tee -a "$OUT/SUMMARY.txt"; exit 1; }

baro_gguf=$("$PY" -c "import json; print(json.load(open('bench/quality-models.json'))['$KEY']['baro_gguf'])")
llama_gguf=$("$PY" -c "import json; print(json.load(open('bench/quality-models.json'))['$KEY']['llama_gguf'])")
[ -f "$baro_gguf" ] || die setup "missing $baro_gguf"
[ -f "$llama_gguf" ] || die setup "missing $llama_gguf"
echo "KEY=$KEY baro_gguf=$baro_gguf llama_gguf=$llama_gguf" | tee "$OUT/arm.txt"

echo "== build ours =="
bench/quality-build.sh "$baro_gguf" "$OUT/build" > "$OUT/build.log" 2>&1 || die build "see $OUT/build.log"
ok build "$(tail -1 "$OUT/build.log")"

echo "== baro-tokenize (shared, build once) =="
[ -x .work/baro-tokenize ] || $HOME/Projects/mojo/mojo-baro/.venv/bin/mojo build tools/baro-tokenize.mojo -I . -I serve -o .work/baro-tokenize \
  || die tok "baro-tokenize build failed"

echo "== baro-serve (shared, build once) =="
[ -x serve/target/release/baro-serve ] || (cd serve && cargo build --release) > "$OUT/cargo-build.log" 2>&1 \
  || die cargo "see $OUT/cargo-build.log"

echo "== perplexity: ours =="
"$PY" bench/quality-ppl-run.py --engine "$OUT/build/engine" --gguf "$baro_gguf" --pack "$OUT/build/pack" \
  --text "$WIKI" --ctx 512 --chunks 8 --tokenize-bin .work/baro-tokenize --out "$OUT/ppl-ours" \
  > "$OUT/ppl-ours.log" 2>&1 || die ppl-ours "see $OUT/ppl-ours.log"
ok ppl-ours "$(tail -1 "$OUT/ppl-ours.log")"

echo "== perplexity: llama.cpp =="
"$LLAMA_BIN/llama-perplexity" -m "$llama_gguf" -f "$WIKI" -c 512 --chunks 8 -ngl 99 -fa on -ctk f16 -ctv f16 -t 8 \
  > "$OUT/ppl-llama.log" 2>&1 || die ppl-llama "see $OUT/ppl-llama.log"
ok ppl-llama "$(grep 'Final estimate' "$OUT/ppl-llama.log")"

echo "== task eval: start both servers =="
: > "$OUT/ours-server.stdout"
BARO_PACK="$OUT/build/pack" serve/target/release/baro-serve --engine "$OUT/build/engine" --pack "$OUT/build/pack" --port 0 \
  > "$OUT/ours-server.stdout" 2> "$OUT/ours-server.stderr" &
OURS_PID=$!
LLAMA_PORT=8199
"$LLAMA_BIN/llama-server" -m "$llama_gguf" -c 4096 -ngl 99 -fa on -ctk f16 -ctv f16 -b 2048 -ub 512 -t 8 \
  --host 127.0.0.1 --port "$LLAMA_PORT" > "$OUT/llama-server.log" 2>&1 &
LLAMA_PID=$!
cleanup() { kill "$OURS_PID" "$LLAMA_PID" 2>/dev/null; wait "$OURS_PID" "$LLAMA_PID" 2>/dev/null; }
trap cleanup EXIT

OURS_URL=""
for _ in $(seq 1 120); do
  line=$(grep -m1 '^listening on' "$OUT/ours-server.stdout" 2>/dev/null || true)
  [ -n "$line" ] && { OURS_URL=$(echo "$line" | grep -oE 'http://[0-9.]+:[0-9]+'); break; }
  kill -0 "$OURS_PID" 2>/dev/null || die start "ours server exited: $(tail -5 "$OUT/ours-server.stderr")"
  sleep 1
done
[ -n "$OURS_URL" ] || die start "ours server never printed listening line"
LLAMA_URL="http://127.0.0.1:$LLAMA_PORT"
up=0
for _ in $(seq 1 120); do
  curl -sf "$LLAMA_URL/health" >/dev/null 2>&1 && { up=1; break; }
  kill -0 "$LLAMA_PID" 2>/dev/null || die start "llama server exited: $(tail -5 "$OUT/llama-server.log")"
  sleep 1
done
[ "$up" = 1 ] || die start "llama server never came up"
ok start "ours=$OURS_URL llama=$LLAMA_URL"

echo "== task eval: run =="
"$PY" bench/quality-task-eval.py --ours-url "$OURS_URL" --llama-url "$LLAMA_URL" \
  --tasks bench/data/e8_tasks.json --out "$OUT/task" > "$OUT/task-eval.log" 2>&1 \
  || die task "see $OUT/task-eval.log"
ok task "$(tail -2 "$OUT/task-eval.log" | head -1)"

cleanup
trap - EXIT

echo "== score =="
"$PY" bench/quality-score.py --key "$KEY" --ppl-ours "$OUT/ppl-ours/ppl-result.json" \
  --ppl-llama-log "$OUT/ppl-llama.log" --task-dir "$OUT/task" --out "$OUT/result.json" \
  || die score "see above"
ok score "$(head -c 200 "$OUT/result.json")"

echo "== cleanup: delete pack (disk floor, keep result/logs) =="
rm -rf "$OUT/build/pack" "$OUT/ppl-ours/dump"
df -h /home | tail -1 | tee -a "$OUT/SUMMARY.txt"
echo "DONE $KEY -> $OUT/result.json"
