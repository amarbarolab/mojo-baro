#!/usr/bin/env bash
# One model's quality row, bench/quality-protocol.md amendment 2.
#   bench/quality-run.sh KEY        (inside gpu-wait; bench/quality-sweep.sh wraps all 10)
# CPU llama-server (-ngl 0) on the reference GGUF tokenizes and detokenizes for
# both arms; the GPU is held by one arm at a time.
set -uo pipefail
cd "$(dirname "$0")/.."
KEY=$1
PY=$HOME/Projects/mojo/mojo-baro/.venv/bin/python3
MOJO=$HOME/Projects/mojo/mojo-baro/.venv/bin/mojo
WIKI=$HOME/Models/quant-lab/wikitext-2-raw/wiki.test.raw
LB=$HOME/llama.cpp/build/bin
OUT=.work/quality/$KEY
rm -rf "$OUT"; mkdir -p "$OUT"
ok() { echo "OK $1: $2" | tee -a "$OUT/SUMMARY.txt"; }
die() { echo "FAIL $1: $2" | tee -a "$OUT/SUMMARY.txt"; exit 1; }
field() { "$PY" -c "import json; print(json.load(open('bench/quality-models.json'))['$KEY']['$1'])"; }
baro_gguf=$(field baro_gguf); llama_gguf=$(field llama_gguf); engine=$(field engine)
echo "KEY=$KEY engine=$engine baro_gguf=$baro_gguf llama_gguf=$llama_gguf llama_sha=$(git -C $HOME/llama.cpp rev-parse --short HEAD) head=$(git rev-parse --short HEAD)" | tee "$OUT/arm.txt"

tools/baro serve "$baro_gguf" --no-serve > "$OUT/cache.log" 2>&1 || die cache "see $OUT/cache.log"
cache_engine=$(grep -oE '^engine: (hit|miss) \S+' "$OUT/cache.log" | awk '{print $3}')
cache_pack=$(grep -oE '^pack: (hit|miss) \S+' "$OUT/cache.log" | awk '{print $3}')
[ -x "$cache_engine" ] && [ -s "$cache_pack/index.txt" ] || die cache "no engine/pack in $OUT/cache.log"
ok cache "engine=$cache_engine pack=$cache_pack"

TOKP=8198
"$LB/llama-server" -m "$llama_gguf" -ngl 0 -c 8192 -t 8 --host 127.0.0.1 --port $TOKP > "$OUT/tok-server.log" 2>&1 &
TOKPID=$!
GPUPID=""
cleanup() { kill $TOKPID $GPUPID 2>/dev/null; wait $TOKPID $GPUPID 2>/dev/null; }
trap cleanup EXIT
for _ in $(seq 600); do curl -sf localhost:$TOKP/health >/dev/null 2>&1 && break; kill -0 $TOKPID 2>/dev/null || die tok "cpu server exited"; sleep 1; done
TOK=http://127.0.0.1:$TOKP

if [ "$engine" != spark ]; then
  dflag=""; [ "$engine" = moe ] && dflag="-D BARO_MODEL=qwen35moe"
  # shellcheck disable=SC2086
  "$MOJO" build serve/engine.mojo -I . -I kernels $dflag -o "$OUT/engine-head" > "$OUT/build.log" 2>&1 || die build "see $OUT/build.log"
  meta=$("$PY" tools/gguf-extract.py "$baro_gguf" --meta)
  rp=$(jq -r '.["baro.run.prompt.tokens"]' <<<"$meta"); rt=$(jq -r '.["baro.run.ref.tokens"]' <<<"$meta")
  "$PY" bench/quality-ppl-run.py --engine "$OUT/engine-head" --pack "$cache_pack" --text "$WIKI" --tok-url $TOK \
    --ref-prompt "$rp" --ref-tokens "$rt" --out "$OUT/ppl-ours" > "$OUT/ppl-ours.log" 2>&1 || die ppl-ours "$(tail -2 "$OUT/ppl-ours.log")"
  ok ppl-ours "$(tail -1 "$OUT/ppl-ours.log")"
  "$LB/llama-perplexity" -m "$llama_gguf" -f "$WIKI" -c 512 --chunks 8 -ngl 99 -fa on -ctk f16 -ctv f16 -t 8 \
    > "$OUT/ppl-llama.log" 2>&1 || die ppl-llama "see $OUT/ppl-llama.log"
  ok ppl-llama "$(grep 'Final estimate' "$OUT/ppl-llama.log")"
fi

"$PY" bench/quality-task-ids.py prep --tok-url $TOK --out "$OUT/task" > "$OUT/task.log" 2>&1 || die prep "$(tail -3 "$OUT/task.log")"
ok prep "$(tail -1 "$OUT/task.log")"
"$PY" bench/quality-task-ids.py ours --engine "$cache_engine" --pack "$cache_pack" --out "$OUT/task" >> "$OUT/task.log" 2>&1 || die ours "$(tail -3 "$OUT/task.log")"
ok ours "$(tail -1 "$OUT/task.log")"

GPUP=8199
"$LB/llama-server" -m "$llama_gguf" -c 16384 -np 8 -ngl 99 -fa on -ctk f16 -ctv f16 -t 8 --host 127.0.0.1 --port $GPUP > "$OUT/llama-server.log" 2>&1 &
GPUPID=$!
for _ in $(seq 600); do curl -sf localhost:$GPUP/health >/dev/null 2>&1 && break; kill -0 $GPUPID 2>/dev/null || die llama "gpu server exited"; sleep 1; done
"$PY" bench/quality-task-ids.py llama --url http://127.0.0.1:$GPUP --out "$OUT/task" >> "$OUT/task.log" 2>&1 || die llama "$(tail -3 "$OUT/task.log")"
ok llama "$(tail -1 "$OUT/task.log")"
kill $GPUPID; wait $GPUPID 2>/dev/null; GPUPID=""

"$PY" bench/quality-task-ids.py score --tok-url $TOK --out "$OUT/task" >> "$OUT/task.log" 2>&1 || die score-task "$(tail -3 "$OUT/task.log")"
ok task "$(tail -1 "$OUT/task.log")"

if [ "$engine" != spark ]; then
  "$PY" bench/quality-score.py --key "$KEY" --ppl-ours "$OUT/ppl-ours/ppl-result.json" --ppl-llama-log "$OUT/ppl-llama.log" \
    --task-dir "$OUT/task" --out "$OUT/result.json" || die score "see above"
else
  "$PY" bench/quality-score.py --key "$KEY" --task-dir "$OUT/task" --out "$OUT/result.json" || die score "see above"
fi
ok result "$(head -c 300 "$OUT/result.json")"
rm -rf "$OUT/ppl-ours/dump" "$OUT/engine-head"
echo "DONE $KEY"
