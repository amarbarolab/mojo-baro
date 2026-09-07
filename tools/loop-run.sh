#!/usr/bin/env bash
# One loop iteration as ONE GPU job: proposer server up -> propose -> server down
# -> champion from the gguf's own sources (3 runs) -> gate. bench/loop-protocol.md.
# usage: gpu-wait run --vram 22 -- env [LOOP_PROMPT2=ids] tools/loop-run.sh ITER MODEL-BARO.gguf [propose args...]
# The server is started and stopped INSIDE the job so the GPU never changes hands
# mid-iteration; PID-exact stop; /props read back before proposing (P1).
set -uo pipefail
cd "$(dirname "$0")/.."
iter=$1; model=$2; shift 2
dir=.work/loop/$iter; mkdir -p "$dir"; log="$dir/run.log"; exec > >(tee -a "$log") 2>&1
echo "== loop $iter  $(date -Is)  model $model  gate $(git rev-parse --short HEAD)"
ss -ltn | grep -q ':8083 ' && { echo "port 8083 already in use; refusing"; exit 1; }
"$HOME/llama.cpp/build/bin/llama-server" -m "$model" -c 20480 -np 1 -ngl 99 -fa on -b 2048 -ub 512 -t 8 \
  -ctk q8_0 -ctv q8_0 --jinja --reasoning-format deepseek \
  --temp 0.6 --top-p 0.95 --top-k 20 --min-p 0 --repeat-penalty 1.05 --presence-penalty 1.0 \
  --host 127.0.0.1 --port 8083 --metrics > "$dir/server.log" 2>&1 &
spid=$!
for i in $(seq 1 180); do curl -sf http://127.0.0.1:8083/health >/dev/null 2>&1 && break; kill -0 $spid 2>/dev/null || { echo "server died: $(tail -3 "$dir/server.log")"; exit 1; }; sleep 2; done
curl -sf http://127.0.0.1:8083/health >/dev/null || { echo "server never healthy"; kill $spid; exit 1; }
curl -s http://127.0.0.1:8083/props > "$dir/props.json"
echo "server pid $spid; /props: $(python3 -c "import json;d=json.load(open('$dir/props.json'));p=d.get('default_generation_settings',{}).get('params',{});print({k:p.get(k) for k in ('temperature','top_p','top_k','min_p','repeat_penalty','presence_penalty')}, 'n_ctx', d.get('default_generation_settings',{}).get('n_ctx'))")"
python3 tools/loop-propose.py "$model" "$iter" "$@" > "$dir/propose.log" 2>&1; prc=$?
cat "$dir/propose.log"
kill $spid; wait $spid 2>/dev/null; sleep 3
ss -ltn | grep -q ':8083 ' && { echo "server still on 8083 after kill"; exit 1; }
[ $prc = 0 ] || { echo "propose failed ($prc)"; exit 1; }
echo "== champion from the gguf's own sources"
t=()
for k in 1 2 3; do
  tools/gguf-closure.sh "$model" > "$dir/champion-closure$k.log" 2>&1 || { echo "gguf-closure run $k FAILED: $(tail -2 "$dir/champion-closure$k.log")"; exit 1; }
  t+=("$(grep -oE 'tok/s_gen: [0-9.]+' "$dir/champion-closure$k.log" | awk '{print $2}')")
done
champ=$(printf '%s\n' "${t[@]}" | sort -n | sed -n 2p)
echo "champion tok/s_gen ${t[*]} -> median $champ ($(grep -h PASS "$dir/champion-closure1.log"))"
echo "$champ" > "$dir/CHAMPION_TOKPS"
echo "== gate"
tools/loop-gate.sh "$iter" "$champ"
echo "== done $(date -Is)"
