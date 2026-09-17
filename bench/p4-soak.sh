#!/usr/bin/env bash
# P4 round 2 soak: N identical requests in ONE engine-qwen process on the iGPU, T=0, while a
# 1 Hz poller records KFD's per-process queue-eviction time for the iGPU
# (/sys/class/kfd/kfd/proc/<pid>/stats_<gpu_id>/evicted_ms). An evicted queue is preempted
# mid-kernel and restored through CWSR. The report lines up every request's token stream
# against the majority stream and against the eviction time that accrued while it ran, so a
# glitch can be tied to (or cleared of) a save/restore event without adding any sync to the engine.
# P4_SOAK_ANALYZE=1 recomputes the report from an existing OUT dir without touching the GPU.
# Run under gpu-wait. P4_IGPU_ENGINE / P4_IGPU_PACK / P4_OUT / P4_SOAK_PROMPT / P4_SOAK_N.
# P4_SOAK_ENV="K=V K=V" adds engine environment for an arm; the script reads the effect back
# (bytes mapped through the render node) instead of trusting that the variable took.
set -euo pipefail
export PATH="/opt/rocm/bin:$HOME/iTools/bin:/usr/bin:/bin:${PATH:-}"
unset HSA_OVERRIDE_GFX_VERSION HIP_VISIBLE_DEVICES ROCR_VISIBLE_DEVICES
ROOT=$(cd "$(dirname "$0")/.." && pwd)
ENGINE=${P4_IGPU_ENGINE:-$ROOT/.work/p4/bin/engine-qwen}
PACK=${P4_IGPU_PACK:-$ROOT/.work/team-B/codex/p4/qwen-pack}
OUT=${P4_OUT:-$ROOT/.work/p4/soak}
TARGET=${P4_SOAK_PROMPT:-p08-sql}
N=${P4_SOAK_N:-40}
GEN=${P4_SOAK_GEN:-64}
read -r -a EXTRA_ENV <<< "${P4_SOAK_ENV:-}"
fail() { echo "FAIL $1: $2 (log ${3:-none})"; exit 1; }

PROMPT="$ROOT/bench/mtp-prompts/$TARGET.tokens"
for p in "$ENGINE" "$PACK/index.txt" "$PACK/pack.bin" "$PROMPT"; do
  [ -e "$p" ] || fail preflight "missing $p"
done
[ "$(sha256sum "$PACK/index.txt" | awk '{print $1}')" = \
  e2ef587fa49be96a8d95d2715038398b8a45757428fb0f80c7469e9be2471084 ] || fail preflight "Qwen7 pack index hash mismatch"
IGPU_ID=$(for n in /sys/class/kfd/kfd/topology/nodes/*; do
  grep -q "gfx_target_version 100306" "$n/properties" 2>/dev/null && cat "$n/gpu_id"; done | head -1)
[ -n "$IGPU_ID" ] || fail preflight "no KFD node with gfx_target_version 100306"
mkdir -p "$OUT"
ids=$(tr -s ' \n' ',' < "$PROMPT" | sed 's/^,//; s/,$//')
for k in $(seq 1 "$N"); do
  printf '{"id":%d,"prompt":[%s],"n":%d,"spec":false}\n' "$k" "$ids" "$GEN"
done > "$OUT/req.jsonl"
if [ "${P4_CPU_PREFLIGHT:-0}" = 1 ]; then
  [ "$(wc -l < "$OUT/req.jsonl")" -eq "$N" ] || fail preflight "request file"
  echo "PASS p4-soak CPU preflight: paths, pack hash, $N request lines, iGPU gpu_id $IGPU_ID"
  exit 0
fi
if [ "${P4_SOAK_ANALYZE:-0}" != 1 ]; then
[ -n "${GPU_WAITING_ROOM_JOB:-}" ] || fail admission "run under gpu-wait"

{
  echo "engine=$ENGINE sha256=$(sha256sum "$ENGINE" | awk '{print $1}')"
  echo "pack=$PACK"
  echo "target=$TARGET n=$N gen=$GEN igpu_gpu_id=$IGPU_ID"
  echo "extra_env=${P4_SOAK_ENV:-none}"
  echo "cwsr_enable=$(cat /sys/module/amdgpu/parameters/cwsr_enable) sched_policy=$(cat /sys/module/amdgpu/parameters/sched_policy)"
  echo "igpu_env:"; igpu-env
} > "$OUT/arm.txt" 2>&1

igpu-env --run env BARO_SERVE=1 BARO_PACK="$PACK" "${EXTRA_ENV[@]}" stdbuf -oL "$ENGINE" < "$OUT/req.jsonl" > "$OUT/engine.log" 2>&1 &
pid=$!
stat="/sys/class/kfd/kfd/proc/$pid/stats_$IGPU_ID/evicted_ms"
echo "t_s,done,evicted_ms" > "$OUT/poll.csv"
t0=$(date +%s)
seen_stat=0
mapped=NA
while kill -0 "$pid" 2>/dev/null; do
  m=$(awk '$6 ~ /renderD/ { split($1, a, "-"); t += strtonum("0x" a[2]) - strtonum("0x" a[1]) } END { printf "%d", t / 1048576 }' "/proc/$pid/maps" 2>/dev/null || true)
  [ -n "$m" ] && [ "$m" != 0 ] && mapped=$m
  ev=$(cat "$stat" 2>/dev/null || echo NA)
  [ "$ev" != NA ] && seen_stat=1
  echo "$(( $(date +%s) - t0 )),$(grep -c '"done":true' "$OUT/engine.log" 2>/dev/null || true),$ev" >> "$OUT/poll.csv"
  sleep 1
done
wait "$pid" || fail soak "engine exited non-zero" "$OUT/engine.log"
[ "$seen_stat" = 1 ] || fail soak "never read $stat, the eviction receipt is missing" "$OUT/poll.csv"
grep -q '"error"' "$OUT/engine.log" && fail soak "engine rejected a request" "$OUT/engine.log"
echo "readback.render_node_mapped_mb=$mapped" | tee -a "$OUT/arm.txt"
fi
[ "$(grep -c '^generated:' "$OUT/engine.log")" -eq "$N" ] || fail soak "expected $N generated lines" "$OUT/engine.log"

grep '^generated:' "$OUT/engine.log" | sed 's/^generated: *//; s/ *$//' > "$OUT/streams.txt"
major=$(sort "$OUT/streams.txt" | uniq -c | sort -rn | head -1 | sed 's/^ *[0-9]* //')
{
  echo "request,first_mismatch_vs_majority,evicted_ms_during"
  k=0
  while IFS= read -r line; do
    k=$((k + 1))
    mm=$(paste -d' ' <(tr ' ' '\n' <<< "$major") <(tr ' ' '\n' <<< "$line") | awk '$1 != $2 {print NR - 1; exit}')
    lo=$(awk -F, -v d=$((k - 1)) 'NR > 1 && $2 == d && $3 != "NA" {print $3; exit}' "$OUT/poll.csv")
    hi=$(awk -F, -v d="$k" 'NR > 1 && $2 >= d && $3 != "NA" {print $3; exit}' "$OUT/poll.csv")
    [ -n "$hi" ] || hi=$(awk -F, 'NR > 1 && $3 != "NA" {v = $3} END {print v}' "$OUT/poll.csv")
    echo "$k,${mm:-none},$(( ${hi:-0} - ${lo:-0} ))"
  done < "$OUT/streams.txt"
} > "$OUT/soak.csv"
awk -F, 'NR > 1 { n++; g += ($2 != "none"); e += ($3 > 0); ge += ($2 != "none" && $3 > 0) }
  END { printf "SUMMARY requests=%d glitched=%d with_eviction=%d glitched_and_evicted=%d\n", n, g, e, ge }' "$OUT/soak.csv" | tee "$OUT/SUMMARY.txt"
grep -v ",none,0$" "$OUT/soak.csv" || true
echo "total_evicted_ms=$(awk -F, 'NR > 1 && $3 != "NA" {v = $3} END {print v + 0}' "$OUT/poll.csv")"
