#!/usr/bin/env bash
# All 10 models, bench/quality-protocol.md item 3. One gpu-wait job for the
# whole sweep (each model holds the GPU well under a few minutes; releasing
# and re-queuing between models buys nothing and risks losing the slot to
# another lane mid-sweep).
set -uo pipefail
cd "$(dirname "$0")/.."
if [ -z "${GPU_WAITING_ROOM_JOB:-}" ] && command -v gpu-wait >/dev/null; then
  exec gpu-wait run --vram 22 --priority 20 --timeout 14400 -- "$0" "$@"
fi
OUT=.work/quality
mkdir -p "$OUT"
: > "$OUT/sweep.log"
KEYS="llama-3.2-1b lily-7b qwen25-7b qwen25-coder-7b granite-4.2-3b ornith-1.5-9b qwythos-v2 qwythos-champion regescore-35b spark-x2.5-4b"
for k in $KEYS; do
  echo "=== $k $(date -Is) ===" | tee -a "$OUT/sweep.log"
  bench/quality-run.sh "$k" 2>&1 | tee -a "$OUT/sweep.log"
  status=${PIPESTATUS[0]}
  if [ "$status" != 0 ]; then
    echo "=== $k FAILED (exit $status), see $OUT/$k/SUMMARY.txt, continuing to next model ===" | tee -a "$OUT/sweep.log"
  fi
done
echo "=== sweep done $(date -Is) ===" | tee -a "$OUT/sweep.log"
