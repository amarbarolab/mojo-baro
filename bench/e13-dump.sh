#!/usr/bin/env bash
# usage: bench/e13-dump.sh [--mode check|dump] [any e13_engine_dump.mojo CLI arg]
# Builds and runs bench/e13_engine_dump.mojo behind the GPU waiting room,
# forwarding the dump's env knobs (E13_DUMP_LIMIT, etc.) INSIDE the `--`
# command. gpu-wait run does NOT forward the caller's environment to the job
# it launches -- bench/latent-handoff.sh hit this first (PROTOCOL-RULES P1,
# its own e8_env forwarding below). A bare `env VAR=val gpu-wait run -- cmd`
# sets VAR on the gpu-wait wrapper process only; the job reads its own unset
# default and never notices. E13-build-2026-09-11: launching the 50-item
# smoke dump this way (`env E13_DUMP_LIMIT=50 gpu-wait run -- ...`) set
# E13_DUMP_LIMIT on the wrapper, not the job -- the job saw E13_DUMP_LIMIT
# unset, defaulted to 0 (unlimited), and ran the full 4,799-item sweep
# instead of the smoke's 50. This script is the fix: it always exists so
# nobody re-learns the gotcha by hand.
set -euo pipefail
cd "$(dirname "$0")/.."

mode="check"
extra_args=()
while [ $# -gt 0 ]; do
  case "$1" in
    --mode) mode="$2"; shift 2 ;;
    *) extra_args+=("$1"); shift ;;
  esac
done

mkdir -p .work/e13
echo "building e13_engine_dump..."
./.venv/bin/mojo build bench/e13_engine_dump.mojo -o .work/e13/e13_engine_dump \
  -I . -I kernels -I serve -I bench

fwd_env=()
for v in BARO_PACK BARO_E8_GGUF BARO_E8_TMAX E13_DUMP_LIMIT E13_MIN_STEPS E8_GSM8K_TRAIN E13_DUMP_DIR E13_WORK_DIR; do
  [ -n "${!v-}" ] && fwd_env+=("$v=${!v}")
done

gpu-wait run --priority 20 --vram 13 -- \
  env "${fwd_env[@]}" .work/e13/e13_engine_dump --mode "$mode" "${extra_args[@]}"
