#!/usr/bin/env bash
# P0b gates 1/2/4 mechanism smoke, one real engine.
#
# The plan's own gates 1/2/4 need "two engines on this box (P4's rig)" --
# P4 (multi-GPU wiring) is a separate, not-yet-built platform item, so the
# full placement-spread and live-failover claims cannot be made yet (see
# exchange/lane-P0B-report.md). This proves everything that IS provable with
# one real GPU engine: the proxy forwards a real request correctly (gate 1's
# "response identical to single-engine run", the part that does not need a
# second engine), SSE streaming pass-through works end to end, the catalog
# logs placement, and CONTRACT 4's full wire path (P1 export -> router's
# GET /v1/state poll -> hash match -> placement="locality") actually fires,
# not just its unit-tested pieces. Gate 2's live "kill one engine mid-run"
# and gate 1's placement-spread across engines stay BLOCKED on P4, named as
# such, never faked with a single-engine substitute.
# GPU: minutes. Run through gpu-wait, never bare.
set -euo pipefail
cd "$(dirname "$0")/.."

out=${1:-.work/p0b-proxy-smoke}
engine=${BARO_ENGINE:-.work/engine}
pack=${BARO_PACK:-.work/engine-pack-q4}
serve=${BARO_SERVE_BIN:-.work/team-A/sonnet/target/release/baro-serve}
router_bin=${ROUTER_BIN:-.work/team-A/sonnet/target/release/router}
mkdir -p "$out"
exec > >(tee "$out/smoke.log") 2>&1

fail() { echo "FAIL $1: $2"; exit 1; }
[ -x "$serve" ] || fail setup "missing $serve"
[ -x "$router_bin" ] || fail setup "missing $router_bin"
[ -x "$engine" ] || fail setup "missing $engine"
[ -d "$pack" ] || fail setup "missing pack $pack"
command -v curl >/dev/null || fail setup "curl is required"
command -v python3 >/dev/null || fail setup "python3 is required"

echo "engine sha256=$(sha256sum "$engine" | cut -d' ' -f1) mtime=$(date -r "$engine")"

"$serve" --engine "$engine" --pack "$pack" --port 0 > "$out/serve.stdout" 2> "$out/serve.stderr" &
srv=$!
router_pid=""
cleanup() {
  [ -n "$router_pid" ] && kill -TERM "$router_pid" 2>/dev/null
  kill -INT "$srv" 2>/dev/null
  wait "$srv" 2>/dev/null
  [ -n "$router_pid" ] && wait "$router_pid" 2>/dev/null
  true
}
trap cleanup EXIT

for _ in $(seq 1 600); do
  grep -q '^listening on' "$out/serve.stdout" && break
  kill -0 "$srv" 2>/dev/null || fail start "serve exited: $(tail -3 "$out/serve.stderr")"
  sleep 0.5
done
direct_url=$(python3 -c "import re,sys; print(re.search(r'https?://\S+', open(sys.argv[1]).read()).group(0))" "$out/serve.stdout") \
  || fail start "no serve listening URL"
direct_port=$(python3 -c "import sys,urllib.parse; print(urllib.parse.urlsplit(sys.argv[1]).port)" "$direct_url")
echo "direct engine at $direct_url"

router_port=18300
BARO_ROUTER_ENGINES="real=http://127.0.0.1:${direct_port},dead=127.0.0.1:18399" \
  "$router_bin" --host 127.0.0.1 --port "$router_port" --node-id 00000000-0000-4000-8000-0000000000c0 \
  > "$out/router.stdout" 2> "$out/router.stderr" &
router_pid=$!
for _ in $(seq 1 60); do
  grep -q '^listening on' "$out/router.stdout" && break
  kill -0 "$router_pid" 2>/dev/null || fail start "router exited: $(tail -5 "$out/router.stderr")"
  sleep 0.5
done
router_url="http://127.0.0.1:${router_port}"

# Let probe_loop pass "real" healthy (needs 2 consecutive 5s-interval
# passes) and poll its GET /v1/state at least once.
sleep 12
curl -fsS "$router_url/v1/node-info" > "$out/node-info-before.json" || fail health "router node-info failed"
python3 - "$out/node-info-before.json" <<'PY'
import json, sys
info = json.load(open(sys.argv[1]))
real = next((e for e in info["engines"] if e["id"] == "real"), None)
assert real is not None, info
assert real["healthy"] is True, real
dead = next((e for e in info["engines"] if e["id"] == "dead"), None)
assert dead is not None and dead["healthy"] is False, dead
print(f"node-info OK: real healthy, dead not, resident_state present={('resident_state' in real)}")
PY

# ---- gate 1 mechanism: proxied response matches the direct single-engine one

body='{"prompt":"The capital of France is","max_tokens":8,"temperature":0.0}'
curl -fsS -X POST "$direct_url/v1/completions" -H 'content-type: application/json' -d "$body" > "$out/direct.json" \
  || fail gate1 "direct /v1/completions failed"
curl -fsS -X POST "$router_url/v1/completions" -H 'content-type: application/json' -d "$body" > "$out/proxied.json" \
  || fail gate1 "proxied /v1/completions failed"
python3 - "$out/direct.json" "$out/proxied.json" <<'PY'
import json, sys
direct = json.load(open(sys.argv[1]))
proxied = json.load(open(sys.argv[2]))
dt = direct["choices"][0]["text"]
pt = proxied["choices"][0]["text"]
assert dt == pt, (dt, pt)
# Not the whole usage object: proxied is the SAME prompt sent a second
# time, so the engine's own prefix chain now has it cached (usage.baro's
# cached_tokens/prefill_rows legitimately differ, cold vs warm) -- that is
# correct engine behavior, not a proxy defect. Compare only the
# model-behavior counts gate 1 actually cares about.
for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
    assert direct["usage"][key] == proxied["usage"][key], (key, direct["usage"], proxied["usage"])
print(f"gate1 mechanism OK: proxied text matches direct byte-for-byte: {pt!r}")
PY

# ---- SSE streaming pass-through: chat completions, stream:true

chat_body='{"messages":[{"role":"user","content":"Say hi in one word."}],"max_tokens":8,"temperature":0.0,"stream":true}'
curl -fsS -N -X POST "$router_url/v1/chat/completions" -H 'content-type: application/json' -d "$chat_body" \
  > "$out/proxied-stream.sse" || fail stream "proxied SSE stream failed"
sse_frames=$(grep -c '^data: ' "$out/proxied-stream.sse" || true)
[ "$sse_frames" -gt 1 ] || fail stream "expected multiple SSE data frames through the proxy, got $sse_frames"
grep -q '^data: \[DONE\]' "$out/proxied-stream.sse" || fail stream "no terminal [DONE] frame reached through the proxy"
echo "SSE streaming pass-through OK: $sse_frames data frames, terminal [DONE] reached"

# ---- catalog: the completions request above must show placement=rank

curl -fsS "$router_url/v1/workloads" > "$out/workloads-1.json" || fail catalog "GET /v1/workloads failed"
python3 - "$out/workloads-1.json" <<'PY'
import json, sys
rows = json.load(open(sys.argv[1]))
done = [r for r in rows if r["path"] == "/v1/completions" and r["state"] == "done"]
assert done, rows
assert done[-1]["engine"] == "real", done[-1]
assert done[-1]["placement"] == "rank", done[-1]
print(f"catalog OK: {len(done)} completions row(s), last one engine=real placement=rank")
PY

# ---- gate 4: CONTRACT 4's full wire path, resident checkpoint -> router poll -> locality
#
# GET /v1/state (state.rs::list) reads checkpoints::Registry -- the same
# store POST /v1/checkpoints populates -- not the state_load-file path
# alone, so a tracked checkpoint via /v1/checkpoints is the direct way to
# make a prefix genuinely resident and visible to the router's poll.

locality_prompt="A distinctive prefix for the P0b locality gate smoke, unlikely to appear anywhere else"
curl -fsS -X POST "$direct_url/v1/checkpoints" -H 'content-type: application/json' \
  -d "$(python3 -c 'import json,sys; print(json.dumps({"prompt": sys.argv[1], "max_tokens": 1}))' "$locality_prompt")" \
  > "$out/checkpoint.json" || fail gate4 "checkpoint create failed"

sleep 6  # one more probe_loop cycle to pick up the new resident state

curl -fsS "$router_url/v1/node-info" > "$out/node-info-after.json" || fail gate4 "router node-info failed"
resident_hash=$(python3 - "$out/node-info-after.json" <<'PY'
import json, sys
info = json.load(open(sys.argv[1]))
real = next(e for e in info["engines"] if e["id"] == "real")
hashes = real.get("resident_prefix_hashes", [])
assert hashes, real
print(hashes[0])
PY
)
echo "resident prefix_hash after checkpoint: $resident_hash"

locality_body=$(python3 -c 'import json,sys; print(json.dumps({"prompt": sys.argv[1], "max_tokens": 4, "temperature": 0.0}))' "$locality_prompt")
curl -fsS -X POST "$router_url/v1/completions" -H 'content-type: application/json' -d "$locality_body" > "$out/locality-response.json" \
  || fail gate4 "proxied locality completions request failed"

curl -fsS "$router_url/v1/workloads" > "$out/workloads-2.json" || fail gate4 "GET /v1/workloads failed"
python3 - "$out/workloads-2.json" "$locality_prompt" <<'PY'
import json, sys
rows = json.load(open(sys.argv[1]))
done = [r for r in rows if r["path"] == "/v1/completions" and r["state"] == "done"]
assert done, rows
last = done[-1]
assert last["engine"] == "real", last
assert last["placement"] == "locality", (last, "expected the locality term to name itself as the reason")
print("gate4 OK: locality-prefixed request placed with placement=locality")
PY

echo "PASS p0b-proxy-smoke"
