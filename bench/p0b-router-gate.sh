#!/usr/bin/env bash
# P0b gate 3: prove that PAIR can discover and query the CPU router.
# Usage: bench/p0b-router-gate.sh OUTDIR [PORT] [NODE_UUID]
# This gate is CPU-only.  The PAIR scanner is started before the router so the
# browse subscription is live before the service advertisement appears.
set -euo pipefail
cd "$(dirname "$0")/.."

out=${1:-.work/team-A/codex/p0b/pair-gate}
port=${2:-18103}
node_uuid=${3:-00000000-0000-4000-8000-0000000000b3}
router_bin=${ROUTER_BIN:-.work/team-A/codex/router-target/release/router}
scanner_bin=${PAIR_SCANNER_BIN:-$HOME/Projects/imports/Personal-AI-Router/services/build/bin/nvpair-node-scanner}

mkdir -p "$out"
exec > >(tee "$out/contract.log") 2>&1

fail() { echo "FAIL $1: $2"; exit 1; }
[[ "$port" =~ ^[0-9]+$ ]] || fail args "port must be numeric"
[[ "$node_uuid" =~ ^[0-9a-fA-F-]{36}$ ]] || fail args "node UUID must be UUID-shaped"

cat > "$out/arm.txt" <<EOF
router_bin=$router_bin
pair_scanner=$scanner_bin
port=$port
node_uuid=$node_uuid
cpu_only=true
EOF

# gate-dryrun must prove the arm reached this script and stop before either
# process is launched.
if [ "${GATE_DRYRUN:-0}" = 1 ]; then
  echo "FAIL GPU: GATE_DRYRUN stops before router and PAIR scanner launch"
  exit 97
fi

[ -x "$router_bin" ] || fail setup "missing executable $router_bin"
[ -x "$scanner_bin" ] || fail setup "missing executable $scanner_bin"
command -v python3 >/dev/null || fail setup "python3 is required"

python3 - "$out" "$router_bin" "$scanner_bin" "$port" "$node_uuid" <<'PY'
import json
import os
import pathlib
import selectors
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

out = pathlib.Path(sys.argv[1]).resolve()
router_bin, scanner_bin = sys.argv[2], sys.argv[3]
port, node_uuid = int(sys.argv[4]), sys.argv[5]
out.mkdir(parents=True, exist_ok=True)

def advertise_address():
    forced = os.environ.get("ROUTER_ADVERTISE_IP")
    if forced:
        return forced
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # UDP connect selects the local route without sending a payload.
        sock.connect(("192.0.2.1", 9))
        address = sock.getsockname()[0]
        if address and not address.startswith("127."):
            return address
    except OSError:
        pass
    finally:
        sock.close()
    return "127.0.0.1"

advertise_ip = advertise_address()
mdns_host = socket.gethostname().split(".", 1)[0] + ".local."

scanner = None
router = None
scanner_frames = []
router_stdout = out / "router.stdout"
router_stderr = out / "router.stderr"

def terminate(proc):
    if proc is None or proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=3)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=3)

def read_json_lines(proc, selector, timeout=0.25):
    events = selector.select(timeout)
    for key, _ in events:
        line = key.fileobj.readline()
        if not line:
            selector.unregister(key.fileobj)
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        scanner_frames.append(value)

try:
    # Start PAIR first.  Its stdout is newline-delimited JSON-RPC.
    scanner = subprocess.Popen(
        [scanner_bin], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=(out / "scanner.stderr").open("w"), text=True, bufsize=1,
    )
    scanner_sel = selectors.DefaultSelector()
    scanner_sel.register(scanner.stdout, selectors.EVENT_READ)
    ready_deadline = time.monotonic() + 8
    while time.monotonic() < ready_deadline:
        read_json_lines(scanner, scanner_sel)
        if any(frame.get("jsonrpc") == "2.0" and frame.get("method") == "ready"
               for frame in scanner_frames):
            break
        if scanner.poll() is not None:
            raise RuntimeError(f"PAIR scanner exited {scanner.returncode}")
    else:
        raise RuntimeError("PAIR scanner did not emit ready")

    with router_stdout.open("w") as stdout, router_stderr.open("w") as stderr:
        router_env = os.environ.copy()
        router_env["BARO_ROUTER_ADVERTISE_IP"] = advertise_ip
        router_env["BARO_ROUTER_MDNS_HOST"] = mdns_host
        router = subprocess.Popen(
            [router_bin, "--host", "0.0.0.0", "--port", str(port),
             "--node-id", node_uuid],
            stdout=stdout, stderr=stderr, text=True, env=router_env,
        )

    listen_deadline = time.monotonic() + 8
    while time.monotonic() < listen_deadline:
        if router.poll() is not None:
            raise RuntimeError(f"router exited {router.returncode}")
        if router_stdout.exists() and "listening on" in router_stdout.read_text():
            break
        time.sleep(0.1)
    else:
        raise RuntimeError("router did not announce listening")

    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/node-info", method="GET"
    )
    with urllib.request.urlopen(request, timeout=3) as response:
        node_info = json.load(response)
    (out / "node-info.json").write_text(json.dumps(node_info, indent=2) + "\n")
    assert node_info.get("hostUuid") == node_uuid, node_info
    assert "state_locality" not in node_info, node_info

    # Let the browse event and node-info enrichment settle, then ask the
    # scanner for the consolidated directory record.
    query_sent = False
    # The scanner's shared browser deliberately scans on a multi-second
    # cadence.  Give one full browse interval plus mDNS propagation time; this
    # remains a bounded CPU gate rather than assuming the first packet wins.
    query_deadline = time.monotonic() + 18
    while time.monotonic() < query_deadline:
        read_json_lines(scanner, scanner_sel)
        if not query_sent and time.monotonic() >= query_deadline - 2:
            scanner.stdin.write(json.dumps({
                "jsonrpc": "2.0", "id": 1, "method": "discovery:get-nodes",
            }) + "\n")
            scanner.stdin.flush()
            query_sent = True
        if any(frame.get("id") == 1 for frame in scanner_frames):
            break

    (out / "pair-frames.json").write_text(json.dumps(scanner_frames, indent=2) + "\n")
    responses = [frame for frame in scanner_frames if frame.get("id") == 1]
    assert responses, scanner_frames
    nodes = responses[-1].get("result", {}).get("nodes", [])
    node = next((item for item in nodes if item.get("hostUuid") == node_uuid), None)
    assert node is not None, nodes
    assert node.get("ip") == advertise_ip, (advertise_ip, node)
    assert node.get("services", {}).get("ni", {}).get("port") == port, node
    discovered = [
        frame.get("params", {}).get("node", {})
        for frame in scanner_frames
        if frame.get("method") == "discovery:node-discovered"
    ]
    assert any(item.get("hostUuid") == node_uuid for item in discovered), discovered
    print(f"PAIR node discovery PASS uuid={node_uuid} ni.port={port}")
    print("node-info hostUuid PASS and state_locality absent")
finally:
    terminate(router)
    terminate(scanner)
    (out / "router.exit").write_text(
        f"router_returncode={router.returncode if router is not None else 'not-started'}\n"
    )
    (out / "scanner.exit").write_text(
        f"scanner_returncode={scanner.returncode if scanner is not None else 'not-started'}\n"
    )
PY

echo "P0b gate 3 PASS: PAIR discovered router and verified /v1/node-info"
