#!/usr/bin/env bash
# Runs the UNCHANGED P4 gate body (.work/p4/run-two-engines.sh) as a child and watches it from
# outside: a 10 Hz poller records, for both engine children, KFD queue-eviction time
# (stats_<gpu_id>/evicted_ms), the count of finished requests in each server log, and the iGPU's
# sclk, with millisecond timestamps. The gate's verdict and exit code pass through untouched;
# the watch only adds receipts, so a glitch can be placed in time against evictions and idle gaps.
# Run under gpu-wait with the same environment the gate body takes (P4_OUT, P4_IGPU_PACK, ...).
set -euo pipefail
export PATH="/opt/rocm/bin:$HOME/iTools/bin:/usr/bin:/bin:${PATH:-}"
ROOT=$(cd "$(dirname "$0")/.." && pwd)
GATE=${P4_GATE_BODY:-$ROOT/.work/p4/run-two-engines.sh}
OUT=${P4_OUT:?set P4_OUT, the gate body writes its receipts there}
fail() { echo "FAIL $1: $2 (log ${3:-none})"; exit 1; }
[ -x "$GATE" ] || fail preflight "missing gate body $GATE"
mkdir -p "$OUT"
sha256sum "$GATE" > "$OUT/gate-body.sha256"

node_of() { for n in /sys/class/kfd/kfd/topology/nodes/*; do
  grep -q "gfx_target_version $1" "$n/properties" 2>/dev/null && { cat "$n/gpu_id"; return; }; done; }
IGPU_ID=$(node_of 100306); XTX_ID=$(node_of 110000)
[ -n "$IGPU_ID" ] && [ -n "$XTX_ID" ] || fail preflight "KFD gpu_id lookup (igpu=$IGPU_ID xtx=$XTX_ID)"
if [ "${P4_CPU_PREFLIGHT:-0}" = 1 ]; then
  echo "PASS p4-gate-watch CPU preflight: gate body present, igpu gpu_id $IGPU_ID, xtx gpu_id $XTX_ID"
  exit 0
fi

"$GATE" > "$OUT/gate.out" 2>&1 &
gate_pid=$!
trap 'kill -TERM "$gate_pid" 2>/dev/null || true' EXIT
pid_from() { [ -f "$1" ] || return 0; sed -n 's/.*engine child pid: \([0-9][0-9]*\).*/\1/p' "$1" | tail -1; }
echo "t_ms,igpu_done,xtx_done,igpu_evicted_ms,xtx_evicted_ms" > "$OUT/watch.csv"
while kill -0 "$gate_pid" 2>/dev/null; do
  ip=$(pid_from "$OUT/igpu-server.log"); xp=$(pid_from "$OUT/xtx-server.log")
  ie=NA; xe=NA
  [ -n "$ip" ] && ie=$(cat "/sys/class/kfd/kfd/proc/$ip/stats_$IGPU_ID/evicted_ms" 2>/dev/null || echo NA)
  [ -n "$xp" ] && xe=$(cat "/sys/class/kfd/kfd/proc/$xp/stats_$XTX_ID/evicted_ms" 2>/dev/null || echo NA)
  echo "$(date +%s%3N),$(grep -c 'engine: generated:' "$OUT/igpu-server.log" 2>/dev/null || true),$(grep -c 'engine: generated:' "$OUT/xtx-server.log" 2>/dev/null || true),$ie,$xe" >> "$OUT/watch.csv"
  sleep 0.1
done
rc=0; wait "$gate_pid" || rc=$?
tail -3 "$OUT/gate.out"
awk -F, 'NR > 1 && $4 != "NA" { if ($2 != d) { if (d != "") printf "igpu request %d: evicted_ms +%d\n", d + 1, last - start; d = $2; start = $4 } last = $4 }' "$OUT/watch.csv" > "$OUT/watch-summary.txt"
grep -v "+0$" "$OUT/watch-summary.txt" || true
echo "p4-gate-watch: gate exit $rc, watch rows $(wc -l < "$OUT/watch.csv")"
exit "$rc"
