#!/usr/bin/env bash
# One-shot gate for the baked-in megakernel (BARO_MEGA=1 is the engine default).
# usage: tools/mega-gate.sh [OUTDIR]     (run under gpu-wait; needs the GPU)
# Stages, fail-closed, one receipt file per stage in OUTDIR:
#   build      engine + kernel gate binary
#   kernel     kernels/test_mega_block (4-layer synthetic pack, bit-identical, head argmax)
#   tests      ./run-tests.sh (GEMM parity + kernel census)
#   identity   every runnable pack (.work/engine-pack-q8, -q8d): mega == launch path
#              GENERATED, no-spec and BARO_SPEC=1; q8 also vs ref-tokens-64
#   perf       q8 no-spec: 3 runs each arm, median + spread
set -uo pipefail
cd "$(dirname "$0")/.."
out=${1:-.work/mega-gate}; mkdir -p "$out"; : > "$out/SUMMARY.txt"
ok() { echo "PASS $1: $2" | tee -a "$out/SUMMARY.txt"; }
die() { echo "FAIL $1: $2" | tee -a "$out/SUMMARY.txt"; exit 1; }
./.venv/bin/mojo build serve/engine.mojo -I kernels -o .work/engine > "$out/build-engine.log" 2>&1 || die build "$(grep -m1 error: "$out/build-engine.log" | cut -c1-160)"
./.venv/bin/mojo build kernels/test_mega_block.mojo -o .work/test_mega_block -I kernels > "$out/build-test.log" 2>&1 || die build "$(grep -m1 error: "$out/build-test.log" | cut -c1-160)"
ok build "engine + test_mega_block"
./.work/test_mega_block > "$out/kernel.log" 2>&1; grep -q '^PASS' "$out/kernel.log" || die kernel "$(grep -E 'FAIL|mismatches' "$out/kernel.log" | head -1)"
ok kernel "$(grep -oE 'ratio= [0-9.]+' "$out/kernel.log")"
./run-tests.sh > "$out/tests.log" 2>&1 || die tests "$(tail -1 "$out/tests.log")"
ok tests "$(grep -E 'GEMM OK|orphans' "$out/tests.log" | tr '\n' ';')"
for pack in .work/engine-pack-q8 .work/engine-pack-q8d; do
  p=$(basename "$pack")
  [ -f "$pack/index.txt" ] || die identity "$p: no index.txt"
  for spec in 0 1; do
    for arm in 0 1; do
      BARO_PACK="$pack" BARO_SPEC=$spec BARO_MEGA=$arm ./.work/engine > "$out/$p.spec$spec.mega$arm.log" 2>&1 || die identity "$p spec=$spec mega=$arm: engine exited $?"
    done
    a=$(grep '^GENERATED' "$out/$p.spec$spec.mega0.log"); m=$(grep '^GENERATED' "$out/$p.spec$spec.mega1.log")
    [ -n "$m" ] && [ "$a" = "$m" ] || die identity "$p spec=$spec: mega GENERATED differs from launch path"
    ok identity "$p spec=$spec mega==launch ($(grep -oE 'tok/s_gen: [0-9.]+' "$out/$p.spec$spec.mega1.log"))"
  done
  if [ "$p" = engine-pack-q8 ]; then
    tools/check-tokens.sh .work/engine-pack/ref-tokens-64.txt "$out/$p.spec0.mega1.log" > "$out/$p.ref.log" 2>&1 || die identity "$p vs ref-tokens-64: $(head -1 "$out/$p.ref.log")"
    ok identity "$p vs ref-tokens-64"
  fi
done
t0=(); t1=()
for k in 1 2 3; do
  BARO_MEGA=0 ./.work/engine > "$out/perf.mega0.$k.log" 2>&1; t0+=("$(grep -oE 'tok/s_gen: [0-9.]+' "$out/perf.mega0.$k.log" | awk '{print $2}')")
  BARO_MEGA=1 ./.work/engine > "$out/perf.mega1.$k.log" 2>&1; t1+=("$(grep -oE 'tok/s_gen: [0-9.]+' "$out/perf.mega1.$k.log" | awk '{print $2}')")
done
python3 - "${t0[@]}" -- "${t1[@]}" <<'PY' | tee -a "$out/SUMMARY.txt"
import sys, statistics as st
a=sys.argv[1:sys.argv.index('--')]; m=sys.argv[sys.argv.index('--')+1:]
a=[float(x) for x in a]; m=[float(x) for x in m]
sp=lambda x:(max(x)-min(x))/st.median(x)*100
print(f"PASS perf: launch median {st.median(a):.2f} (spread {sp(a):.1f}%)  mega median {st.median(m):.2f} (spread {sp(m):.1f}%)  ratio {st.median(m)/st.median(a):.3f}")
PY
echo "ALL PASS" | tee -a "$out/SUMMARY.txt"
