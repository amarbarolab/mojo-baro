#!/usr/bin/env bash
# usage: gpu-wait run --vram 23 --timeout 900 -- bench/fork-cap-probe.sh [OUT] [PERCENT] [TMAX]
#
# Rig question for the cross-node fork lane (briefs/2026-09-17-latentos-cross-node-fork.md):
# can two engine processes share the XTX when MAX's device memory manager is capped with
# MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_SIZE_PERCENT? Starts node A, reads VRAM, starts node B
# beside it, reads VRAM, then runs one completion on each. The answer picks the gate 2 rig
# (two live nodes, or time-sliced engines), so it is a receipt, not a smoke.
set -euo pipefail
cd "$(dirname "$0")/.."
out=${1:-.work/fork/cap-probe}
pct=${2:-45}
tmax=${3:-33024}
engine=${BARO_ENGINE:-.work/fork/engine}
pack=${BARO_PACK:-$HOME/Projects/mojo/mojo-baro/.work/engine-pack-q4}
serve=${BARO_SERVE_BIN:-serve/target/release/baro-serve}
mkdir -p "$out"
exec > >(tee "$out/probe.log") 2>&1

fail() { echo "FAIL $1: $2 (log $out/probe.log)"; exit 1; }
[ -x "$engine" ] || fail setup "missing $engine"
[ -x "$serve" ] || fail setup "missing $serve"
[ -d "$pack" ] || fail setup "missing pack $pack"
vram() { cat /sys/class/drm/card*/device/mem_info_vram_used | sort -n | tail -1; }

pids=()
cleanup() { for p in "${pids[@]}"; do kill -INT "$p" 2>/dev/null || true; done; wait 2>/dev/null || true; }
trap cleanup EXIT

start() {
  name=$1
  env MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_SIZE_PERCENT="$pct" BARO_TMAX="$tmax" BARO_SPEC=0 \
    BARO_CKPT_DIR="$out/ckpts-$name" "$serve" --engine "$engine" --pack "$pack" --port 0 \
    > "$out/$name.stdout" 2> "$out/$name.stderr" &
  pids+=($!)
  for _ in $(seq 1 600); do
    grep -q '^listening on' "$out/$name.stdout" && break
    kill -0 "${pids[-1]}" 2>/dev/null || fail "start-$name" "server exited: $(tail -3 "$out/$name.stderr")"
    sleep 0.5
  done
  grep -q '^listening on' "$out/$name.stdout" || fail "start-$name" "no listening line in 300 s"
  grep -m1 -oE 'http://[0-9.:]+' "$out/$name.stdout"
}

v0=$(vram)
echo "arm: percent=$pct tmax=$tmax engine_sha=$(sha256sum "$engine" | cut -c1-16) pack=$pack vram_before=$v0"
ua=$(start a); va=$(vram)
echo "node a up at $ua vram_used=$va delta=$(( (va - v0) / 1048576 )) MiB"
ub=$(start b); vb=$(vram)
echo "node b up at $ub vram_used=$vb delta=$(( (vb - va) / 1048576 )) MiB"
grep -h -m1 'TMAX' "$out/a.stderr" "$out/b.stderr" || true

for u in "$ua" "$ub"; do
  curl -fsS -X POST "$u/v1/completions" -H 'content-type: application/json' \
    -d '{"prompt":"The capital of France is","max_tokens":8,"temperature":0,"spec":false}' \
    > "$out/completion-$(echo "$u" | tr -dc 0-9).json" || fail completion "POST $u/v1/completions"
done
python3 - "$out" <<'PY'
import glob, json, sys
ids = [json.load(open(f))["choices"][0].get("tokens") or json.load(open(f))["choices"][0]["text"] for f in sorted(glob.glob(sys.argv[1] + "/completion-*.json"))]
print("completions:", ids)
if len(ids) != 2 or ids[0] != ids[1]:
    print("FAIL identity: the two nodes disagree"); sys.exit(1)
PY
echo "vram_after_requests=$(vram)"
echo "PASS fork-cap-probe percent=$pct tmax=$tmax"
