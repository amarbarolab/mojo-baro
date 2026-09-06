#!/usr/bin/env bash
# usage: bench/ab-prompts.sh ENGINE OUTDIR "ENV_A" "ENV_B" [label_A label_B]
#   e.g. bench/ab-prompts.sh ./.work/engine .work/ab "BARO_MEGA=0" "BARO_MEGA=1"
#        bench/ab-prompts.sh ./.work/engine .work/ab "BARO_SPEC=1 BARO_MEGA_WIN=0" "BARO_SPEC=1 BARO_MEGA_WIN=1"
# PROTOCOL-RULES P4 receipt for any two arms: every bench/mtp-prompts/*.tokens, A then B
# back to back (same stint), identity = B GENERATED == A GENERATED, per-arm median and
# spread, power cap / voltage offset read back first (P1). Wrap in bench/clock-probe.sh
# for the clock receipt.
set -uo pipefail
cd "$(dirname "$0")/.."
eng=$1; out=$2; envA=$3; envB=$4; la=${5:-A}; lb=${6:-B}
mkdir -p "$out"
echo "power_cap_uW=$(cat /sys/class/drm/card1/device/hwmon/hwmon*/power1_cap) vddgfx=$(grep -A1 OD_VDDGFX_OFFSET /sys/class/drm/card1/device/pp_od_clk_voltage | tail -1) armA='$envA' armB='$envB'" | tee "$out/arm.txt"
echo "prompt n_prompt ${la}_tok_s ${lb}_tok_s identity" > "$out/results.txt"
for tf in bench/mtp-prompts/p*.tokens; do
  p=$(basename "$tf" .tokens); n=$(wc -w < "$tf")
  env BARO_PROMPT="$tf" $envA "$eng" > "$out/$p.$la.log" 2>&1
  env BARO_PROMPT="$tf" $envB "$eng" > "$out/$p.$lb.log" 2>&1
  ga=$(grep '^GENERATED' "$out/$p.$la.log"); gb=$(grep '^GENERATED' "$out/$p.$lb.log")
  ta=$(grep -oE 'tok/s_gen: [0-9.]+' "$out/$p.$la.log" | cut -d' ' -f2); tb=$(grep -oE 'tok/s_gen: [0-9.]+' "$out/$p.$lb.log" | cut -d' ' -f2)
  [ -n "$gb" ] && [ "$ga" = "$gb" ] && id=PASS || id=FAIL
  echo "$p $n ${ta:-nan} ${tb:-nan} $id" >> "$out/results.txt"
done
column -t "$out/results.txt"
python3 - "$out/results.txt" "$la" "$lb" <<'PY'
import sys, statistics as st
rows=[l.split() for l in open(sys.argv[1]).read().splitlines()[1:]]
a=[float(r[2]) for r in rows]; b=[float(r[3]) for r in rows]
fails=[r[0] for r in rows if r[4]!="PASS"]
sp=lambda x:(max(x)-min(x))/st.median(x)*100
print(f"{sys.argv[2]} median {st.median(a):.2f} tok/s_gen spread {sp(a):.1f}%  |  {sys.argv[3]} median {st.median(b):.2f} spread {sp(b):.1f}%  |  ratio {st.median(b)/st.median(a):.3f}  |  identity fails: {fails or 'none'}")
PY
