#!/usr/bin/env bash
# usage: bench/bridge-roundtrip.sh [OUT] [P_TOKENS] [Q_TOKENS]
#
# CPU gate for tools/state-to-llama-slot.mojo (P1 item 4, the forward bridge). No GPU:
# llama-server runs with --device none and the GPU hidden.
#   1. llama-server prefills P on the CPU and saves its slot           -> orig.slot
#   2. tools/llama-slot-to-state (the reverse tool, landed)            -> BAROST01
#   3. tools/state-to-llama-slot (the tool under test)                 -> round.slot
#   4. round.slot must equal orig.slot byte for byte
#   5. a fresh llama-server restores round.slot and continues P+Q: cache_n must be the
#      restored length and the ids must equal the cold continuation of P+Q from step 1
#      (a prompt equal to the restored prefix is re-evaluated whole by llama-server, a
#      recurrent state cannot step back one token, so the suffix Q is what makes reuse visible)
# Step 4 proves the writer emits llama.cpp's own layout (cell ext, framing, row order,
# recurrent section); step 5 proves llama.cpp accepts and uses the file.
set -euo pipefail
cd "$(dirname "$0")/.."
out=${1:-.work/fork/bridge-roundtrip}
p=${2:-bench/mtp-prompts/p05-math.tokens}
q=${3:-bench/mtp-prompts/p02-python-fib.tokens}
model=${LH_GGUF:-$HOME/Models/qwythos-9b-claude-mythos-5-1m-mtp-bf16/Qwythos-9B-Claude-Mythos-5-1M-MTP-Q4_0-pure.gguf}
server=${LLAMA_SERVER:-$HOME/llama.cpp/build/bin/llama-server}
kv=${BRIDGE_KV:-f16}
fwd=.work/fork/state-to-llama-slot
rev=.work/fork/llama-slot-to-state
mkdir -p "$out/slots"
exec > >(tee "$out/roundtrip.log") 2>&1

fail() { echo "FAIL $1: $2 (log $out/roundtrip.log)"; exit 1; }
for f in "$model" "$server" "$p" "$q"; do [ -e "$f" ] || fail setup "missing $f"; done
./.venv/bin/mojo build tools/state-to-llama-slot.mojo -I . -I serve -I kernels -o "$fwd" > "$out/build-fwd.log" 2>&1 || fail build "forward tool, $out/build-fwd.log"
./.venv/bin/mojo build tools/llama-slot-to-state.mojo -I . -I serve -I kernels -o "$rev" > "$out/build-rev.log" 2>&1 || fail build "reverse tool, $out/build-rev.log"
echo "arm: model=$(basename "$model") kv=$kv prompt=$p ($(wc -w < "$p") tokens) server=$("$server" --version 2>&1 | grep -m1 -o 'version: .*' || true)"

srv=""
stop() { [ -n "$srv" ] && { kill "$srv" 2>/dev/null || true; wait "$srv" 2>/dev/null || true; }; srv=""; }
trap stop EXIT
start() {
  port=$((20000 + RANDOM % 20000))
  env HIP_VISIBLE_DEVICES=-1 ROCR_VISIBLE_DEVICES=-1 "$server" -m "$model" --device none -ngl 0 -fa on -np 1 -c 2048 \
    -ctk "$kv" -ctv "$kv" --port "$port" --no-webui --slot-save-path "$out/slots" > "$out/server-$1.log" 2>&1 &
  srv=$!
  for _ in $(seq 1 600); do
    curl -sf "localhost:$port/health" >/dev/null 2>&1 && return 0
    kill -0 "$srv" 2>/dev/null || fail "server-$1" "exited: $(tail -2 "$out/server-$1.log")"
    sleep 0.5
  done
  fail "server-$1" "no /health in 300 s"
}
csv() { cat "$@" | tr -s ' \n' ',' | sed 's/^,//; s/,$//'; }
req() { printf '{"prompt":[%s],"n_predict":%s,"temperature":0,"cache_prompt":true,"return_tokens":true}' "$1" "$2"; }

# 1. cold continuation (32 tokens, the reference), then a clean prefill of P and the slot save
start a
req "$(csv "$p" "$q")" 32 > "$out/req32.json"; req "$(csv "$p")" 0 > "$out/req0.json"
curl -sf "localhost:$port/completion" -H 'content-type: application/json' --data @"$out/req32.json" > "$out/cold.json" || fail cold "completion"
curl -sf -X POST "localhost:$port/slots/0?action=erase" > /dev/null || fail erase "slot erase"
curl -sf "localhost:$port/completion" -H 'content-type: application/json' --data @"$out/req0.json" > "$out/prefill.json" || fail prefill "completion"
curl -sf -X POST "localhost:$port/slots/0?action=save" -H 'content-type: application/json' -d '{"filename":"orig.slot"}' > "$out/save.json" || fail save "slot save"
stop

# 2 to 4. reverse, forward, compare
"$rev" "$out/slots/orig.slot" roundtrip-pack "$out/from-llama.state" || fail reverse "llama-slot-to-state"
"$fwd" "$out/from-llama.state" "$out/slots/round.slot" --kv "$kv" || fail forward "state-to-llama-slot"
if cmp "$out/slots/orig.slot" "$out/slots/round.slot" > "$out/cmp.txt" 2>&1; then
  echo "byte identity: PASS ($(stat -c%s "$out/slots/orig.slot") bytes)"
else
  fail identity "$(cat "$out/cmp.txt"); sizes $(stat -c%s "$out/slots/orig.slot") vs $(stat -c%s "$out/slots/round.slot")"
fi

# 5. a fresh server restores the written file and continues
start b
curl -sf -X POST "localhost:$port/slots/0?action=restore" -H 'content-type: application/json' -d '{"filename":"round.slot"}' > "$out/restore.json" || fail restore "slot restore refused: $(tail -3 "$out/server-b.log")"
curl -sf "localhost:$port/completion" -H 'content-type: application/json' --data @"$out/req32.json" > "$out/restored.json" || fail restored "completion"
stop
python3 - "$out" <<'PY'
import json, sys
o = sys.argv[1]
cold, rest, rs = (json.load(open(f"{o}/{n}.json")) for n in ("cold", "restored", "restore"))
n_restored = rs.get("n_restored")
cache_n = rest["timings"].get("cache_n")
same = cold["tokens"] == rest["tokens"]
print(f"restore n_restored={n_restored} cache_n={cache_n} prompt_n={rest['timings'].get('prompt_n')} ids_equal={same} ({len(cold['tokens'])} tokens)")
if not n_restored or cache_n != n_restored:
    print("FAIL reuse: the restored prefix was not reused"); sys.exit(1)
if not same or len(cold["tokens"]) != 32:
    print("FAIL ids: restored continuation differs from the cold one"); sys.exit(1)
PY
echo "PASS bridge-roundtrip kv=$kv"
