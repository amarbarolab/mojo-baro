#!/usr/bin/env bash
# bench/ruler/run-baseline.sh: llama.cpp baseline arms for the RULER gate
# (bench/ruler-protocol.md item 4). One server launch per (arm, size);
# -c = size + 512; server killed between sizes; /props + server log saved
# next to the run as the P1 receipt.
#
# Usage: bench/ruler/run-baseline.sh ARM SIZE [SIZE ...]
#   ARM  = bf16 | k8v4
#   SIZE = 4096 | 8192 | 16384 | 32768 | 65536 | 131072
# N = 25 at 4096-32768, N = 10 at 65536/131072 (brief's own scope gate).
set -euo pipefail
cd "$(dirname "$0")/../.."

MODEL=${MODEL:-$HOME/Models/qwythos-9b-claude-mythos-5-1m-mtp-bf16/Qwythos-9B-Claude-Mythos-5-1M-MTP-Q4_0-pure.gguf}
PORT=${PORT:-8199}
OUT=${OUT:-.work/ruler-baseline}

ARM=$1; shift
ctk_flags=()
if [ "$ARM" = "k8v4" ]; then
  ctk_flags=(-ctk q8_0 -ctv q4_0)
elif [ "$ARM" != "bf16" ]; then
  echo "ARM must be bf16 or k8v4, got $ARM" >&2; exit 1
fi

for size in "$@"; do
  limit=25
  case $size in
    65536|131072) limit=10 ;;
  esac
  ctx=$((size + 512))
  dst="$OUT/$ARM/$size"
  mkdir -p "$dst"

  $HOME/.local/bin/gpu-wait run --priority 50 --vram 8 -- \
    ~/llama.cpp/build/bin/llama-server -m "$MODEL" -c "$ctx" -ngl 99 -fa on \
    -b 2048 -ub 512 -t 8 -np 1 "${ctk_flags[@]}" --host 127.0.0.1 --port "$PORT" \
    > "$dst/server.log" 2>&1 &
  srv_wrapper=$!

  up=0
  for _ in $(seq 1 240); do
    if curl -sf -m 2 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then up=1; break; fi
    sleep 2
  done
  if [ "$up" != "1" ]; then
    echo "server did not come up for size=$size arm=$ARM" >&2
    kill "$srv_wrapper" 2>/dev/null || true
    exit 1
  fi
  curl -s "http://127.0.0.1:$PORT/props" > "$dst/props.json"
  pid=$(pgrep -f "llama-server -m $MODEL -c $ctx " | head -1)

  echo "=== arm=$ARM size=$size limit=$limit ctx=$ctx pid=$pid ===" | tee -a "$dst/run.log"
  t0=$(date +%s)
  ./.venv/bin/python3 bench/ruler/run.py --base-url "http://127.0.0.1:$PORT/v1" --model qwythos \
    --prompts bench/ruler/prompts --out "$dst/responses" --sizes "$size" --max-tokens 1024 \
    --limit "$limit" | tee -a "$dst/run.log"
  t1=$(date +%s)
  echo "wall_s=$((t1 - t0))" | tee -a "$dst/run.log"

  [ -n "$pid" ] && kill "$pid" 2>/dev/null || true
  kill "$srv_wrapper" 2>/dev/null || true
  sleep 3
done
