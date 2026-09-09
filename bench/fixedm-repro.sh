#!/usr/bin/env bash
# bench/fixedm-repro.sh — E10 FIXED(M) Position Invariance Probe
set -euo pipefail

cd "$(dirname "$0")/.."
echo "Running E10 FIXED(M) position-invariance benchmark on GPU..."
./.venv/bin/mojo run -I kernels -I serve bench/bench_fixedm_kernel_invariance.mojo
