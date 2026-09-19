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

# MOEPF: row-batched MoE kernels bit-exact against their m=1 siblings
./.venv/bin/mojo build kernels/test_moe_rows.mojo -o .work/test_moe_rows -I kernels
./.work/test_moe_rows
./.venv/bin/mojo build kernels/test_ssm_rows.mojo -o .work/test_ssm_rows -I kernels
./.work/test_ssm_rows

# M1a prefix checkpoints: byte-exact restore against the real engine path
# (needs the q4 pack at BARO_PACK, default .work/engine-pack-q4).
./.venv/bin/mojo build kernels/test_prefix.mojo -o .work/test_prefix -I . -I kernels -I serve
./.work/test_prefix

# C3 host reference sampler: pure CPU, no accelerator needed, matched
# against kernels/sample.mojo's semantics (lane-KSAMP).
./.venv/bin/mojo build kernels/test_sample_ref.mojo -o .work/test_sample_ref -I kernels -I serve
./.work/test_sample_ref

# Penalties and top-N logprobs: device kernels vs the host reference at real
# VOCAB width (briefs/2026-09-16-fable-sample-kernels.md).
./.venv/bin/mojo build kernels/test_sample_pen.mojo -o .work/test_sample_pen -I . -I kernels -I serve
./.work/test_sample_pen

# Grammar-masked sampling: mask before truncation, masked argmax at T = 0,
# masked probs rows (briefs/2026-09-16-fable-masked-sampler.md).
./.venv/bin/mojo build kernels/test_sample_mask.mojo -o .work/test_sample_mask -I kernels -I serve
./.work/test_sample_mask

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
./.venv/bin/mojo build kernels/test_latent.mojo -I . -I kernels -I serve -o .work/test_latent
./.work/test_latent

# JSON-enforcement lane item 1 prep: the request-line schema slice
# (briefs/2026-09-16-json-enforcement-lane.md), pure string handling.
./.venv/bin/mojo build serve/test_serve_proto.mojo -I . -I serve -o .work/test_serve_proto
./.work/test_serve_proto

./.venv/bin/mojo build tools/kernel-census.mojo -o .work/kernel-census
./.work/kernel-census --check
