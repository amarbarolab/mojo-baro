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

python3 tools/kernel-census.py --check
