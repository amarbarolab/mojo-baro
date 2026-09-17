#!/usr/bin/env bash
# Mojo kernel -> Metal LLVM IR -> SPIR-V (OpenCL kernel) -> Mesa rusticl on aihq-lab's Radeon R5 M330.
set -euo pipefail
R="$(cd "$(dirname "$0")/../.." && pwd)"
W="$R/.work/spirv-probe"
mkdir -p "$W"
cd "$W"
"$R/.venv/bin/mojo" build -I "$R/kernels" "$R/kernels/test_elementwise.mojo" --target-accelerator apple-m1 --emit asm -o ew.s || { echo "FAIL mojo-emit: see $W"; exit 1; }
python3 "$R/tools/spirv-probe/air2spv.py" ew_elementwise_amar_swiglu_*.ll swiglu swiglu.ll
clang --target=spirv64 -c swiglu.ll -o swiglu.spv 2> clang.log || { echo "FAIL spirv: $W/clang.log"; exit 1; }
scp -q "$R/tools/spirv-probe/host.c" swiglu.spv root@lab-host.example:/root/
ssh -o BatchMode=yes root@lab-host.example 'cd /root && gcc -O2 host.c -o host -lOpenCL -lm && RUSTICL_ENABLE=radeonsi ./host swiglu.spv'
