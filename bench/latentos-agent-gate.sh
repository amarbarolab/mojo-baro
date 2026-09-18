#!/usr/bin/env bash
# LatentOS daemon liveness gate.
# amarbaro.org - Copyright (c) 2026 amarbaro.org. All rights reserved.
set -euo pipefail
cd "$(dirname "$0")/.."

out=${1:-.work/p10-latentos-agent-gate}
bin=${LATENTOS_AGENT_BIN:-.work/p10-latentos-agent/latentos-agent}
mkdir -p "$out"
log="$out/daemon.log"
manifest="$out/manifest.json"
rm -f "$log" "$manifest"
[ -x "$bin" ] || { echo "FAIL setup: missing executable $bin"; exit 1; }

"$bin" --daemon --out "$manifest" >"$log" 2>&1 &
pid=$!
cleanup() {
    kill -TERM "$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
}
trap cleanup EXIT

for _ in $(seq 1 40); do
    grep -q 'latentos-agent daemon serving' "$log" && break
    kill -0 "$pid" 2>/dev/null || { echo "FAIL daemon: $(tail -n 8 "$log")"; exit 1; }
    sleep 0.1
done
grep -q 'latentos-agent daemon serving' "$log" || { echo "FAIL daemon: readiness missing"; exit 1; }
test -s "$manifest" || { echo "FAIL manifest: missing or empty"; exit 1; }
sleep 1
kill -0 "$pid" || { echo "FAIL daemon: exited before liveness check"; exit 1; }
printf 'daemon_pid=%s\nmanifest_bytes=%s\n' "$pid" "$(stat -c %s "$manifest")" > "$out/SUMMARY.txt"
echo "PASS latentos-agent daemon liveness: $out/SUMMARY.txt"
