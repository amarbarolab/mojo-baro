#!/usr/bin/env bash
# Every amar_* elementwise kernel: Mojo -> Metal LLVM IR -> SPIR-V (OpenCL kernel) -> Mesa rusticl on
# aihq-lab's Radeon R5 M330, checked against CPU references. Last line: PASS N/N or FAIL k/N <names>.
set -euo pipefail
R="$(cd "$(dirname "$0")/../.." && pwd)"
T="$R/tools/spirv-probe"
W="$R/.work/spirv-probe"
LAB="${LAB:-root@lab-host.example}"
die() { echo "FAIL $1: $2"; exit 1; }
rm -rf "$W/ir" "$W/out"
mkdir -p "$W/ir" "$W/out"
cd "$W/ir"
"$R/.venv/bin/mojo" build -I "$R/kernels" "$T/probe_all.mojo" --target-accelerator apple-m1 --emit asm -o pa.s > "$W/mojo.log" 2>&1 || die mojo-emit "$W/mojo.log"
clang -O2 "$T/helpers_check.c" -o "$W/helpers_check" -lm 2> "$W/helpers_check.log" || die helpers-build "$W/helpers_check.log"
"$W/helpers_check" || die helpers "integer bf16/f16 conversions differ from native casts"
clang --target=spirv64 -O2 -emit-llvm -S "$T/helpers.c" -o "$W/helpers.ll" 2> "$W/helpers.log" || die helpers-ir "$W/helpers.log"
python3 "$T/inventory.py" "$W"/ir/*.ll > "$W/inventory.md" || die inventory "$W/inventory.md"
rc=0
python3 "$T/air2spv.py" "$W/helpers.ll" "$W/out" "$W"/ir/*.ll || rc=1
for ll in "$W"/out/*.ll; do
    n="$(basename "$ll" .ll)"
    if ! clang --target=spirv64 -O2 -c "$ll" -o "$W/out/$n.spv" 2> "$W/out/$n.clang.log"; then echo "FAIL spirv $n: $W/out/$n.clang.log"; rm -f "$W/out/$n.spv"; rc=1; continue; fi
    if ! spirv-val "$W/out/$n.spv" > "$W/out/$n.val.log" 2>&1; then echo "FAIL spirv-val $n: $W/out/$n.val.log"; rm -f "$W/out/$n.spv"; rc=1; fi
done
ssh -o BatchMode=yes "$LAB" 'rm -rf /root/g0/spv && mkdir -p /root/g0/spv' || die lab-ssh "$LAB unreachable"
scp -q "$T/host.c" "$LAB:/root/g0/" || die lab-scp host.c
scp -q "$W"/out/*.spv "$LAB:/root/g0/spv/" || die lab-scp spv
scp -q "$T/divprobe.c" "$LAB:/root/g0/" || die lab-scp divprobe.c
ssh -o BatchMode=yes "$LAB" 'cd /root/g0 && gcc -O2 divprobe.c -o divprobe -lOpenCL && RUSTICL_ENABLE=radeonsi ./divprobe 2>/dev/null' | tee "$W/divprobe.log" || die divprobe "$W/divprobe.log"
ssh -o BatchMode=yes "$LAB" 'cd /root/g0 && gcc -O2 -Wall host.c -o host -lOpenCL -lm 2> gcc.log || { echo "FAIL host-build: $(head -5 gcc.log)"; exit 1; }; RUSTICL_ENABLE=radeonsi ./host spv' | tee "$W/card.log"
[ "$rc" -eq 0 ] || die convert "a kernel failed conversion upstream yet the card run passed; see lines above"
