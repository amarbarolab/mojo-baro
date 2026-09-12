#!/usr/bin/env bash
# Build the C++ shim, then build and run the Mojo verification tests.
set -euo pipefail
cd "$(dirname "$0")"
S="$PWD/.work/shim-build"
cmake -S shim -B "$S" -DCMAKE_BUILD_TYPE=Release >/dev/null
cmake --build "$S" -j"$(nproc)" >/dev/null
./.venv/bin/mojo build kernels/test_gemm.mojo -o .work/test_gemm -I kernels \
  -Xlinker -L"$S" -Xlinker -lamarbaro_shim -Xlinker -rpath -Xlinker "$S"
./.work/test_gemm

# M1a prefix checkpoints: byte-exact restore against the real engine path
# (needs the q4 pack at BARO_PACK, default .work/engine-pack-q4).
./.venv/bin/mojo build kernels/test_prefix.mojo -o .work/test_prefix -I kernels -I serve
./.work/test_prefix

# C3 host reference sampler: pure CPU, no accelerator needed, matched
# against kernels/sample.mojo's semantics (lane-KSAMP).
./.venv/bin/mojo build kernels/test_sample_ref.mojo -o .work/test_sample_ref -I kernels -I serve
./.work/test_sample_ref

# KATT head-dimension parity: the Spark attention path at HD 64/128/256,
# judged by the numpy float64 oracle (bench/dense-protocol.md, KATT).
./.venv/bin/mojo build kernels/test_spark_attn.mojo -o .work/test_spark_attn -I kernels -I serve
mkdir -p .work/katt
./.work/test_spark_attn
./.venv/bin/python tools/spark-attn-ref.py .work/katt

# serve/latent.mojo sat broken from 506e91d to 2e7b5d3 because nothing here
# built it. This test names both mint and ingest, so the whole sidecar is
# type-checked: Mojo checks lazily per reached symbol, and building a tool that
# imports only one side leaves the other side's breakage invisible.
./.venv/bin/mojo build kernels/test_latent.mojo -I kernels -I serve -o .work/test_latent
./.work/test_latent

./.venv/bin/mojo build tools/kernel-census.mojo -o .work/kernel-census
./.work/kernel-census --check
