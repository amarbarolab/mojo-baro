#!/usr/bin/env bash
# usage: bench/latent-handoff.sh [--items N] [--ids id1,id2,...] [--arms a,b,c] [--out PREFIX]
# E8 HARNESS (exchange/e8-lane-plan-2026-09-09.md item HARNESS): builds the
# dual-engine evaluator, runs it under the GPU waiting room, then scores the
# raw dump with bench/e8_score.py. Default: all 120 items, all 5 arms,
# results/e8/topology1-q4-<date>. Smoke: --items 4 --arms 0,T,L8-raw.
# --ids picks specific item ids (round 2: --items N can't express a mix like
# json_01..04 + math_01..04) and is passed straight through to the binary,
# which selects those ids in the given order instead of the first N in file
# order; --ids overrides --items when both are given.
set -euo pipefail
cd "$(dirname "$0")/.."

items=120
ids=""
arms="0,T,L8-raw,L8-soft,L32-soft"
date_tag=$(date +%Y-%m-%d)
out_prefix="results/e8/topology1-q4-${date_tag}"

while [ $# -gt 0 ]; do
  case "$1" in
    --items) items="$2"; shift 2 ;;
    --ids) ids="$2"; shift 2 ;;
    --arms) arms="$2"; shift 2 ;;
    --out) out_prefix="$2"; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 1 ;;
  esac
done

bin_args=(--items "$items" --arms "$arms" --out "$out_prefix")
if [ -n "$ids" ]; then
  bin_args=(--ids "$ids" --arms "$arms" --out "$out_prefix")
fi

mkdir -p .work results/e8
rm -f .work/e8-vram-ready.marker

echo "building bench_latent_handoff..."
./.venv/bin/mojo build bench/bench_latent_handoff.mojo -o .work/bench_latent_handoff \
  -I . -I kernels -I serve -I bench -I ~/Projects/mojo-uregex/src

read_vram() { gpu-wait gpu --json | python3 -c 'import json,sys; print(json.load(sys.stdin)["vram_used"])'; }

vram_before=$(read_vram)
echo "VRAM before both loads: ${vram_before}"

# gpu-wait run blocks until the command exits, and VRAM is freed the moment it
# does -- a reading taken after that call returns is just baseline again. Run
# it in the background and poll the marker bench_latent_handoff.mojo writes
# once both packs are resident, so the "after" reading is taken mid-run.
gpu-wait run --priority 30 --vram 14 -- \
  .work/bench_latent_handoff "${bin_args[@]}" &
job_pid=$!

waited=0
while [ ! -f .work/e8-vram-ready.marker ] && [ "$waited" -lt 120 ] && kill -0 "$job_pid" 2>/dev/null; do
  sleep 1
  waited=$((waited + 1))
done
if [ -f .work/e8-vram-ready.marker ]; then
  vram_after=$(read_vram)
  echo "VRAM after both loads (mid-run): ${vram_after}"
else
  vram_after=""
  echo "WARNING: e8-vram-ready.marker never appeared; no mid-run VRAM reading" >&2
fi

wait "$job_pid"

topo2_status="see .work/e8-harness-topology2-blocker.md (structure does not transfer, Spark harness not attempted)"

vram_args=(--vram-before "$vram_before")
if [ -n "$vram_after" ]; then
  vram_args+=(--vram-after "$vram_after")
fi
python3 bench/e8_score.py "${out_prefix}.raw.json" bench/data/e8_tasks.json \
  "${out_prefix}.json" "${out_prefix}.md" \
  "${vram_args[@]}" --topology2-status "$topo2_status"

echo "done: ${out_prefix}.json ${out_prefix}.md"
