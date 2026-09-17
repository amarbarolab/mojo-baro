#!/usr/bin/env bash
# usage: bench/a2-ruler.sh ENGINE OUTDIR SIZE [LIMIT] [TASK]
#   RULER subset (bench/ruler-protocol.md tasks and scoring) through baro-serve on our engine, for the
#   A2 step 2 KV-format gate (bench/a2-protocol.md P-S2g). Prompts: .work/a2s2/ruler/prompts, made by
#   bench/ruler/gen.py -n 5 --tasks niah_single --sizes 65536,131072 (BARO_GGUF = the Q4_0-pure GGUF).
#   BARO_TMAX = SIZE rounded up to a page past prompt + chat template + 1024 generated; spec off.
#   Receipts (P1): engine sha256, the engine's BARO_KVQ / BARO_KVTAB / ready lines from the server log.
set -euo pipefail
cd "$(dirname "$0")/.."
eng=$1; out=$2; size=$3; limit=${4:-5}; task=${5:-niah_single}
prompts=${RULER_PROMPTS:-.work/a2s2/ruler/prompts}
port=${PORT:-8231}
tmax=$(( (size + 2048 + 127) / 128 * 128 ))
if [ -z "${GPU_WAITING_ROOM_JOB:-}" ]; then
  exec gpu-wait run --vram 22 --timeout 7200 -- env RULER_PROMPTS="$prompts" PORT="$port" "$0" "$@"
fi
[ -f "$prompts/${task}_${size}.jsonl" ] || { echo "FAIL a2-ruler: $prompts/${task}_${size}.jsonl missing"; exit 1; }
mkdir -p "$out"
{
  echo "engine=$eng sha=$(sha256sum "$eng" | cut -c1-16) size=$size tmax=$tmax limit=$limit task=$task prompts_sha=$(sha256sum "$prompts/${task}_${size}.jsonl" | cut -c1-16)"
  echo "power_cap_uW=$(cat /sys/class/drm/card1/device/hwmon/hwmon*/power1_cap) commit=$(git rev-parse --short HEAD)"
} | tee "$out/arm.txt"
BARO_TMAX=$tmax BARO_SPEC=0 serve/target/release/baro-serve --engine "$eng" --port "$port" > "$out/server.log" 2>&1 &
srv=$!
trap 'rc=$?; kill "$srv" 2>/dev/null; wait "$srv" 2>/dev/null; exit $rc' EXIT
up=0
for _ in $(seq 1 300); do
  curl -sf -m 2 "http://127.0.0.1:$port/health" > /dev/null 2>&1 && { up=1; break; }
  kill -0 "$srv" 2>/dev/null || break
  sleep 2
done
[ "$up" = 1 ] || { echo "FAIL a2-ruler: baro-serve not healthy, see $out/server.log"; tail -5 "$out/server.log"; exit 1; }
grep -E "BARO_KVQ|BARO_KVTAB|\"ready\"" "$out/server.log" | tee -a "$out/arm.txt"
./.venv/bin/python3 bench/ruler/run.py --base-url "http://127.0.0.1:$port/v1" --model qwythos \
  --prompts "$prompts" --out "$out/responses" --tasks "$task" --sizes "$size" --max-tokens 1024 --limit "$limit" \
  | tee "$out/run.log"
[ "${PIPESTATUS[0]}" = 0 ] || { echo "FAIL a2-ruler: run.py, see $out/run.log"; exit 1; }
./.venv/bin/python3 bench/ruler/score.py "$out/responses" --prompts "$prompts" --tasks "$task" --sizes "$size" --json "$out/table.json" | tee "$out/score.txt"
