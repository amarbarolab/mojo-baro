#!/usr/bin/env bash
# P1 POST /v1/state/export smoke (path variant): resident engine, one real
# export request, checks the file landed and the response fields agree with
# it. Not the full plan gate (that needs the cross-node rig and both
# formats); this proves the route is reachable and correct on one node.
# GPU: minutes. Run through gpu-wait, never bare.
set -euo pipefail
cd "$(dirname "$0")/.."

out=${1:-.work/p1-export-smoke}
engine=${BARO_ENGINE:-.work/engine}
pack=${BARO_PACK:-.work/engine-pack-q4}
serve=${BARO_SERVE_BIN:-.work/team-A/sonnet/target/release/baro-serve}
mkdir -p "$out"
exec > >(tee "$out/smoke.log") 2>&1

fail() { echo "FAIL $1: $2"; exit 1; }
[ -x "$serve" ] || fail setup "missing $serve"
[ -x "$engine" ] || fail setup "missing $engine"
[ -d "$pack" ] || fail setup "missing pack $pack"
command -v curl >/dev/null || fail setup "curl is required"
command -v python3 >/dev/null || fail setup "python3 is required"

echo "engine sha256=$(sha256sum "$engine" | cut -d' ' -f1) mtime=$(date -r "$engine")"

"$serve" --engine "$engine" --pack "$pack" --port 0 > "$out/server.stdout" 2> "$out/server.stderr" &
srv=$!
cleanup() { kill -INT "$srv" 2>/dev/null || true; wait "$srv" 2>/dev/null || true; }
trap cleanup EXIT

for _ in $(seq 1 600); do
  grep -q '^listening on' "$out/server.stdout" && break
  kill -0 "$srv" 2>/dev/null || fail start "server exited: $(tail -3 "$out/server.stderr")"
  sleep 0.5
done
url=$(python3 -c "import re,sys; print(re.search(r'https?://\S+', open(sys.argv[1]).read()).group(0))" "$out/server.stdout") \
  || fail start "no listening URL"

curl -fsS "$url/health" > "$out/health.json" || fail health "GET /health failed"

state_path="$PWD/$out/exported.baro"
rm -f "$state_path"
curl -fsS -X POST "$url/v1/state/export" -H 'content-type: application/json' \
  -d "$(python3 -c 'import json; print(json.dumps({"prompt": "The capital of France is", "path": "'"$state_path"'"}))')" \
  > "$out/export.json" || fail export "POST /v1/state/export failed"

python3 - "$out/export.json" "$state_path" <<'PY'
import json, os, sys
resp = json.load(open(sys.argv[1]))
path = sys.argv[2]
for k in ("path", "bytes", "pos", "prefix_hash"):
    assert k in resp, f"missing field {k}: {resp}"
assert resp["path"] == path, (resp["path"], path)
assert os.path.exists(path), f"export claimed a file that does not exist: {path}"
real_bytes = os.path.getsize(path)
assert real_bytes == resp["bytes"], (real_bytes, resp["bytes"])
assert resp["pos"] > 0, resp["pos"]
assert len(resp["prefix_hash"]) == 16 and int(resp["prefix_hash"], 16) >= 0, resp["prefix_hash"]
print(f"export OK: pos={resp['pos']} bytes={resp['bytes']} prefix_hash={resp['prefix_hash']}")
PY

# path present but no prompt/messages/tokens: exercises the "one of ..."
# refusal specifically, not the separate "path required" 501.
curl -sS -X POST "$url/v1/state/export" -H 'content-type: application/json' \
  -d '{"path":"'"$state_path"'.unused"}' > "$out/export-missing-input.json" 2>&1
# No -f: a 400 here is the expected result, not a curl failure to detect.
grep -q "prompt, messages, or tokens" "$out/export-missing-input.json" || fail falsify "export with no input did not refuse as expected"

# no path at all: exercises the streaming-not-built-yet 501.
curl -sS -X POST "$url/v1/state/export" -H 'content-type: application/json' \
  -d '{"prompt":"hi"}' > "$out/export-no-path.json" 2>&1
grep -q "not built yet" "$out/export-no-path.json" || fail falsify "export with no path did not 501 as expected"

echo "PASS p1-state-export-smoke"
