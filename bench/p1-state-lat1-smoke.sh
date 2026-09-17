#!/usr/bin/env bash
# P1 CONTRACT 1: the raw LAT1-wrapped stream, both routes' "no path" form.
# Resident engine, one real POST /v1/state/export with no path (the response
# IS the stream: 256-byte LAT1 header + BAROST0x body), saved to a file and
# checked byte-for-byte against the header fields; that same file re-posted
# to /v1/state/import as the request body; a corrupted copy of it posted to
# import too, checking the payload_sha 409. Not the full plan gate (needs
# both KV formats and the cross-node rig); this proves the container is
# reachable, self-consistent, and the identity checks actually refuse a
# corrupt stream on one node.
# GPU: minutes. Run through gpu-wait, never bare.
set -euo pipefail
cd "$(dirname "$0")/.."

out=${1:-.work/p1-lat1-smoke}
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

# ---- export, no path: the raw LAT1 stream --------------------------------

stream_path="$out/exported.lat1"
curl -fsS -X POST "$url/v1/state/export" -H 'content-type: application/json' \
  -d '{"prompt":"The capital of France is"}' \
  -D "$out/export-headers.txt" -o "$stream_path" || fail export "POST /v1/state/export (stream) failed"
ct=$(grep -i '^content-type:' "$out/export-headers.txt" | tr -d '\r')
echo "content-type: $ct"
echo "$ct" | grep -qi 'application/vnd.baro.state' || fail export "unexpected content-type: $ct"

python3 - "$stream_path" <<'PY'
import struct, sys, hashlib
path = sys.argv[1]
data = open(path, "rb").read()
assert len(data) >= 256, f"stream too short: {len(data)} bytes"
header, payload = data[:256], data[256:]
magic, version, kind, dtype = struct.unpack_from("<IHBB", header, 0)
assert magic == 0x3154414C, f"bad magic {magic:#x}"
assert version == 1, f"bad version {version}"
assert kind == 1, f"bad kind {kind} (want KV_PAGES=1)"
assert dtype in (1, 3), f"bad dtype {dtype}"
pos_hi, = struct.unpack_from("<I", header, 160)
ttl_s, = struct.unpack_from("<I", header, 172)
prefix_hash, = struct.unpack_from("<Q", header, 176)
payload_len, = struct.unpack_from("<Q", header, 184)
payload_sha = header[192:224]
assert pos_hi == 4, f"pos_hi {pos_hi}, want 4 (same prompt as the roundtrip smoke)"
assert ttl_s == 600, f"ttl_s {ttl_s}"
assert payload_len == len(payload), f"payload_len {payload_len} != actual {len(payload)}"
assert payload[:8] == b"BAROST01" or payload[:8] == b"BAROST02", f"payload magic {payload[:8]!r}"
real_sha = hashlib.sha256(payload).digest()
assert real_sha == payload_sha, f"payload_sha mismatch: header {payload_sha.hex()} real {real_sha.hex()}"
print(f"export stream OK: pos_hi={pos_hi} payload_len={payload_len} prefix_hash={prefix_hash:016x}")
with open(sys.argv[1] + ".prefix_hash", "w") as f:
    f.write(f"{prefix_hash:016x}")
PY
export_prefix_hash=$(cat "$stream_path.prefix_hash")

# ---- import, no path: the same stream posted back as the request body ---

curl -fsS -X POST "$url/v1/state/import" -H 'content-type: application/vnd.baro.state' \
  --data-binary "@$stream_path" > "$out/import.json" || fail import "POST /v1/state/import (stream) failed"

python3 - "$out/import.json" "$export_prefix_hash" <<'PY'
import json, sys
imp = json.load(open(sys.argv[1]))
want_prefix_hash = sys.argv[2]
for k in ("prefix_hash", "pos", "restore_ms", "runtime_differs"):
    assert k in imp, f"missing field {k}: {imp}"
assert imp["prefix_hash"] == want_prefix_hash, (imp["prefix_hash"], want_prefix_hash)
assert imp["pos"] == 4, imp["pos"]
assert imp["runtime_differs"] is False, "same server, same runtime: must be false, not null or true"
print(f"import stream OK: pos={imp['pos']} prefix_hash={imp['prefix_hash']} runtime_differs={imp['runtime_differs']}")
PY

# ---- falsifier: a corrupted payload byte must 409 payload_sha, not restore

corrupt_path="$out/corrupted.lat1"
python3 - "$stream_path" "$corrupt_path" <<'PY'
import sys
data = bytearray(open(sys.argv[1], "rb").read())
data[300] ^= 0xFF  # one byte inside the payload, well past the 256-byte header
open(sys.argv[2], "wb").write(data)
PY
curl -sS -X POST "$url/v1/state/import" -H 'content-type: application/vnd.baro.state' \
  --data-binary "@$corrupt_path" > "$out/import-corrupt.json" 2>&1
python3 - "$out/import-corrupt.json" <<'PY'
import json, sys
resp = json.load(open(sys.argv[1]))
assert resp.get("error") == "state_identity", resp
assert resp.get("field") == "payload_sha", resp
print("corrupt-payload falsifier OK:", resp["field"])
PY

# ---- falsifier: a bad magic must 409 magic, not silently pass through ---

badmagic_path="$out/badmagic.lat1"
python3 - "$stream_path" "$badmagic_path" <<'PY'
import sys
data = bytearray(open(sys.argv[1], "rb").read())
data[0:4] = b"NOPE"
open(sys.argv[2], "wb").write(data)
PY
curl -sS -X POST "$url/v1/state/import" -H 'content-type: application/vnd.baro.state' \
  --data-binary "@$badmagic_path" > "$out/import-badmagic.json" 2>&1
python3 - "$out/import-badmagic.json" <<'PY'
import json, sys
resp = json.load(open(sys.argv[1]))
assert resp.get("error") == "state_identity", resp
assert resp.get("field") == "magic", resp
print("bad-magic falsifier OK:", resp["field"])
PY

echo "PASS p1-state-lat1-smoke"
