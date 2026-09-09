#!/usr/bin/env bash
# bench/hidden-dtype.sh — E9 HIDDEN Precision Probe (f32 vs bf16)
set -euo pipefail

cd "$(dirname "$0")/.."
echo "Building E9 HIDDEN dtype benchmark binary..."
./.venv/bin/mojo build bench/bench_hidden_dtype.mojo -I kernels -I serve -o .work/bench_hidden_dtype

echo "Running E9 HIDDEN dtype benchmark on GPU..."
./.work/bench_hidden_dtype
