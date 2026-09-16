#!/usr/bin/env bash
# usage: bench/b4-cross-host.sh ENGINE PACK OUTDIR [PORT]
#
# B4-mini: payload vs recipe on a shaped link (docs/design/latent-os/06-experiments.md
# section "B4-mini", AMDHQ repo). Veth pair + netns stands in for A3's two-VM rig.
# Engine stays on the host, one GPU; every HTTP call in both arms crosses the veth
# link from a peer network namespace back to baro-serve in the root namespace.
#
#   for each profile in 100mbit 1gbit 10gbit:
#     tc qdisc replace dev veth-a root tbf <profile>
#     for each of 3 items (same P, distinct Q suffix):
#       payload: flush -> create checkpoint (P) -> move .baro file across the
#                shaped link via nc, sha256 both ends -> fork (Q) -> ids
#       recipe:  flush -> cold /v1/completions (P+Q) -> ids
#     compare payload ids == recipe ids per item; report breakeven
set -uo pipefail
cd "$(dirname "$0")/.."
eng=$1; pack=$2; out=$3; port=${4:-8099}
mkdir -p "$out" "$out/ckpts" "$out/xfer"
serve=serve/target/release/baro-serve
[ -x "$eng" ] || { echo "no engine at $eng" >&2; exit 2; }
[ -x "$serve" ] || { echo "no $serve; cargo build --release --manifest-path serve/Cargo.toml" >&2; exit 2; }

NS=b4peer
VA=veth-b4a
VB=veth-b4b
HOST_IP=10.99.7.1
PEER_IP=10.99.7.2
XFER_PORT=9911

cleanup() {
  [ -n "${spid:-}" ] && pkill -P "$spid" 2>/dev/null
  [ -n "${spid:-}" ] && kill "$spid" 2>/dev/null
  pkill -f "baro-serve --engine .* --port $port" 2>/dev/null
  sudo firewall-cmd --zone=trusted --remove-interface="$VA" 2>/dev/null
  sudo ip netns exec "$NS" true 2>/dev/null && sudo ip netns del "$NS" 2>/dev/null
  sudo ip link del "$VA" 2>/dev/null
  true
}
trap cleanup EXIT

sudo ip netns del "$NS" 2>/dev/null; sudo ip link del "$VA" 2>/dev/null
sudo ip netns add "$NS"
sudo ip link add "$VA" type veth peer name "$VB"
sudo ip link set "$VB" netns "$NS"
sudo ip addr add "$HOST_IP/24" dev "$VA"
sudo ip link set "$VA" up
# firewalld's default zone drops unsolicited inbound on veth-b4a (verified: TCP
# "no route to host" while ICMP passes); runtime-only, reverted in cleanup.
sudo firewall-cmd --zone=trusted --add-interface="$VA"
sudo ip netns exec "$NS" ip addr add "$PEER_IP/24" dev "$VB"
sudo ip netns exec "$NS" ip link set "$VB" up
sudo ip netns exec "$NS" ip link set lo up
peer() { sudo ip netns exec "$NS" "$@"; }

# ~1000-token P: every mtp-prompts file, sorted, concatenated 3x, truncated to 1000
python3 - <<'PY' > "$out/P.tokens"
import glob
toks = []
for f in sorted(glob.glob("bench/mtp-prompts/p*.tokens")):
    toks += [int(x) for x in open(f).read().split()]
rep = (toks * 3)[:1000]
print(" ".join(str(t) for t in rep))
PY
P="$out/P.tokens"
Q1=bench/mtp-prompts/p02-python-fib.tokens
Q2=bench/mtp-prompts/p05-math.tokens
Q3=bench/mtp-prompts/p11-email.tokens
FLUSH=bench/mtp-prompts/p03-story.tokens
for f in "$P" "$Q1" "$Q2" "$Q3" "$FLUSH"; do [ -f "$f" ] || { echo "missing $f" >&2; exit 2; }; done
{
  echo "engine=$eng sha=$(sha256sum "$eng" | cut -c1-16) serve=$serve sha=$(sha256sum "$serve" | cut -c1-16) pack=$pack port=$port"
  echo "commit=$(git rev-parse --short HEAD) dirty='$(git status --porcelain | tr '\n' ';')'"
  echo "P=$P ($(wc -w < "$P") tokens) Q1=$Q1 Q2=$Q2 Q3=$Q3 flush=$FLUSH"
  echo "rig: netns=$NS host=$HOST_IP peer=$PEER_IP veth=$VA/$VB"
} | tee "$out/arm.txt"

gpu-wait run --priority 20 --vram 14 -- \
  env BARO_PACK="$pack" BARO_CKPT_DIR="$out/ckpts" BARO_SPEC=0 "$serve" --engine "$eng" --pack "$pack" --port "$port" --host 0.0.0.0 \
  > "$out/serve.out" 2>"$out/serve.err" &
spid=$!
ready=0
for _ in $(seq 1 360); do
  peer curl -sf "http://$HOST_IP:$port/health" -o "$out/health.json" 2>/dev/null && { ready=1; break; }
  sleep 5
done
[ "$ready" = 1 ] || { echo "VOID: no /health on $HOST_IP:$port from peer ns; see $out/serve.err" >&2; exit 3; }
echo "health: $(cat "$out/health.json")" | tee -a "$out/arm.txt"

python3 - "$out" "$HOST_IP" "$port" "$P" "$Q1" "$Q2" "$Q3" "$FLUSH" <<'PY'
import json, pathlib, sys, urllib.request, urllib.error
out, host, port, P, Q1, Q2, Q3, F = pathlib.Path(sys.argv[1]), sys.argv[2], sys.argv[3], *sys.argv[4:9]
ids = lambda f: [int(x) for x in open(f).read().split()]
p = ids(P); items = [("q1", Q1, ids(Q1)), ("q2", Q2, ids(Q2)), ("q3", Q3, ids(Q3))]
fl = ids(F)
base = f"http://{host}:{port}"
def call(method, path, body=None):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(base + path, data=data, method=method, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")
greedy = {"max_tokens": 32, "spec": False, "temperature": 0}
def flush():
    call("POST", "/v1/completions", {"prompt": fl, "max_tokens": 4, "spec": False})
(out / "results.json").write_text("[]")
results = []
for name, qfile, q in items:
    flush()
    import time
    t0 = time.monotonic()
    s, cr = call("POST", "/v1/checkpoints", {"prompt": p, "ttl_s": 600, "max_tokens": 1, "spec": False})
    create_s = time.monotonic() - t0
    ck = cr.get("checkpoint", {}); cid = ck.get("id", "")
    files = list((out / "ckpts").glob("*.baro"))
    src = max(files, key=lambda f: f.stat().st_mtime) if files else None
    results.append({
        "item": name, "create_ok": s == 200, "cid": cid, "bytes": ck.get("bytes"),
        "src_file": str(src) if src else None,
    })
    print(f"CREATE {name}: {s} id={cid} bytes={ck.get('bytes')}", flush=True)
(out / "create.json").write_text(json.dumps(results, indent=2))
PY

commonfile=$(ls -t "$out"/ckpts/*.baro 2>/dev/null | head -1)
[ -n "$commonfile" ] || { echo "VOID: no checkpoint file created" >&2; exit 3; }
cid=$(python3 -c "import json;print(json.load(open('$out/create.json'))[0]['cid'])")
srcsha=$(sha256sum "$commonfile" | cut -d' ' -f1)
echo "checkpoint file: $commonfile ($(stat -c%s "$commonfile") bytes) sha256=$srcsha" | tee -a "$out/arm.txt"

run_profile() {
  prof=$1; tcspec=$2
  echo "=== profile $prof ===" | tee -a "$out/arm.txt"
  if [ -n "$tcspec" ]; then
    sudo tc qdisc replace dev "$VA" root $tcspec
  else
    sudo tc qdisc del dev "$VA" root 2>/dev/null || true
  fi
  tc qdisc show dev "$VA" | tee -a "$out/$prof.tc.txt"

  # transfer: nc listener in peer ns, sender in root ns, both cross the shaped veth link
  xferfile="$out/xfer/$prof.baro"
  rm -f "$xferfile"
  peer bash -c "timeout 60 nc -l -p $XFER_PORT > '$xferfile'" &
  ncpid=$!
  sleep 0.3
  t0=$(date +%s.%N)
  nc -N "$PEER_IP" "$XFER_PORT" < "$commonfile"
  wait "$ncpid" 2>/dev/null
  t1=$(date +%s.%N)
  xfer_s=$(python3 -c "print($t1-$t0)")
  xfersha=$(sha256sum "$xferfile" | cut -d' ' -f1)
  match="FAIL"; [ "$xfersha" = "$srcsha" ] && match="PASS"
  echo "$prof transfer_s=$xfer_s bytes=$(stat -c%s "$xferfile" 2>/dev/null) sha_match=$match" | tee -a "$out/arm.txt"

  python3 - "$out" "$HOST_IP" "$port" "$P" "$Q1" "$Q2" "$Q3" "$FLUSH" "$prof" "$cid" "$xfer_s" "$match" <<'PY'
import json, pathlib, sys, time, urllib.request, urllib.error
out, host, port, P, Q1, Q2, Q3, F, prof, cid, xfer_s, xmatch = sys.argv[1:13]
xfer_s = float(xfer_s)
ids = lambda f: [int(x) for x in open(f).read().split()]
p = ids(P); items = [("q1", Q1, ids(Q1)), ("q2", Q2, ids(Q2)), ("q3", Q3, ids(Q3))]
fl = ids(F)
base = f"http://{host}:{port}"
def call(method, path, body=None):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(base + path, data=data, method=method, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")
greedy = {"max_tokens": 32, "spec": False, "temperature": 0}
def flush():
    call("POST", "/v1/completions", {"prompt": fl, "max_tokens": 4, "spec": False})

rows = []
for name, qfile, q in items:
    flush()
    t0 = time.monotonic()
    s, fk = call("POST", f"/v1/checkpoints/{cid}/fork", {"branches": [{"prompt_suffix": q, **greedy}]})
    fork_s = time.monotonic() - t0
    br = (fk.get("branches") or [{}])[0]
    payload_ids = br.get("tokens")
    payload_s = xfer_s + fork_s

    flush()
    t0 = time.monotonic()
    s2, rc = call("POST", "/v1/completions", {"prompt": p + q, **greedy})
    recipe_s = time.monotonic() - t0
    recipe_ids = rc["choices"][0]["tokens"] if s2 == 200 else None

    ids_match = payload_ids == recipe_ids
    rows.append({
        "profile": prof, "item": name, "fork_ok": s == 200, "recipe_ok": s2 == 200,
        "payload_s": payload_s, "recipe_s": recipe_s, "xfer_s": xfer_s,
        "ids_match": ids_match, "restored": br.get("restored"),
        "cached_tokens": br.get("cached_tokens"), "sha_match": xmatch == "PASS",
    })
    print(f"{prof} {name}: payload_s={payload_s:.3f} recipe_s={recipe_s:.3f} ids_match={ids_match}", flush=True)

resfile = pathlib.Path(out) / "results.json"
existing = json.loads(resfile.read_text()) if resfile.exists() else []
resfile.write_text(json.dumps(existing + rows, indent=2))
PY
}

run_profile unshaped ""
run_profile 100mbit "tbf rate 100mbit burst 32kbit latency 400ms"
run_profile 1gbit "tbf rate 1gbit burst 320kbit latency 100ms"
run_profile 10gbit "tbf rate 10gbit burst 3200kbit latency 50ms"

sudo tc qdisc del dev "$VA" root 2>/dev/null || true

python3 - "$out" "$commonfile" <<'PY'
import json, pathlib, sys
out, ckfile = sys.argv[1], sys.argv[2]
rows = json.loads((pathlib.Path(out) / "results.json").read_text())
fails = [r for r in rows if not r["ids_match"] or not r["sha_match"]]
size = pathlib.Path(ckfile).stat().st_size
recipe_fast = min(r["recipe_s"] for r in rows if r["profile"] == "10gbit")
breakeven = size / recipe_fast
pred = 152.75e6  # 03 sec 5, 1088 tok: 118 MiB / 0.81 s
within_2x = (pred / 2) <= breakeven <= (pred * 2)
verdict = "PASS" if not fails and within_2x else "FAIL"
lines = ["# B4-mini receipt", "", f"checkpoint file bytes: {size}",
         f"fastest recipe_s (10gbit profile): {recipe_fast:.4f} s",
         f"measured breakeven: {breakeven/1e6:.2f} MB/s ({breakeven*8/1e9:.3f} Gbit/s)",
         f"03 sec 5 prediction (1088 tok): 152.75 MB/s (1.22 Gbit/s)",
         f"within 2x: {within_2x}", "", "## per-profile, per-item", ""]
for r in rows:
    lines.append(f"- {r['profile']}/{r['item']}: payload_s={r['payload_s']:.3f} recipe_s={r['recipe_s']:.3f} "
                  f"ids_match={r['ids_match']} sha_match={r['sha_match']}")
lines += ["", f"## VERDICT: {verdict}", ""]
if fails:
    lines.append(f"FAILING rows: {json.dumps(fails, indent=2)}")
(pathlib.Path(out) / "receipt.md").write_text("\n".join(lines))
print("RESULT", verdict)
sys.exit(0 if verdict == "PASS" else 1)
PY
rc=$?
exit $rc
