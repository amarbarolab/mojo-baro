#!/usr/bin/env bash
# usage: bench/a2-gate.sh ENGINE_REF ENGINE_CAND OUTDIR [ENV]
#   QUICK=N    first N prompts, 32k set only (iteration gate, P18); default: 20 prompts x 3 sets (merge gate)
#   MIN_PCT=P  forced-agreement bar per prompt (default 100; an int8 KV arm sets its own P14 bar)
# A2 identity gate (docs/NEXT-PLAN.md A2) on the shared-document sets of bench/a2-prompts.sh:
# teacher-forced agreement of ENGINE_CAND against ENGINE_REF's greedy ids at 8k, 16k and 32k, both
# engines resident (BARO_SERVE=1, serve/PROTOCOL.md), per request "force":[ids] and "ckpt":[doclen].
# Sets run 8k -> 16k -> 32k so each set's pinned doc-end checkpoint restores the next set's prefix:
# one 32k prefill in total per engine, the other 59 prompts restore (done line "cached" > 0).
# Reference ids are cached the P17 way (key = ref engine sha256, pack.bin sha256, sampler T=0 spec
# off n=64, sha256 of the request ids); "refcache hit|miss key=" is printed; the candidate is never
# cached. Receipts (P1): ready line (tmax), BARO_MEGA/BARO_SPEC/BARO_TMAX echoes, per request
# cached / restore_s / prefill_s / tok_s and "forced agreement a / n". The 32k set's decode median is
# printed as "decode after 32k" (the 101 tok/s bar). Exit 1 on any void or any prompt under MIN_PCT.
set -euo pipefail
cd "$(dirname "$0")/.."
ref=$1; cand=$2; out=$3; envx=${4:-}
quick=${QUICK:-0}; minpct=${MIN_PCT:-100}
pack=${BARO_PACK:-.work/engine-pack-q4}
if [ -z "${GPU_WAITING_ROOM_JOB:-}" ] && command -v gpu-wait >/dev/null; then
  # the queue does not carry this shell's environment: QUICK=3 ran the full gate once (2026-09-16)
  exec gpu-wait run --timeout 1800 -- env QUICK="$quick" MIN_PCT="$minpct" BARO_PACK="$pack" "$0" "$@"
fi
mkdir -p "$out"
bench/preflight.sh --check
ha=$(sha256sum "$ref" | cut -c1-16); hb=$(sha256sum "$cand" | cut -c1-16)
[ "$ha" != "$hb" ] || { echo "REFUSED: ref and cand are the same binary ($ha)"; exit 2; }
[ -d .work/a2-prompts/L32768 ] || bench/a2-prompts.sh
# pack identity: sha256 of pack.bin, cached beside it against size+mtime (6.8 GB, ~20 s once)
pk="$pack/pack.bin"; st=$(stat -c '%s-%Y' "$pk")
if [ ! -f "$pack/pack.sha256" ] || [ "$(cut -d' ' -f2 "$pack/pack.sha256")" != "$st" ]; then
  echo "$(sha256sum "$pk" | cut -c1-16) $st" > "$pack/pack.sha256"
fi
packsha=$(cut -d' ' -f1 "$pack/pack.sha256")
{
  echo "ref=$ref cand=$cand shaRef=$ha shaCand=$hb pack=$pack packsha=$packsha env='$envx' quick=$quick min_pct=$minpct"
  echo "power_cap_uW=$(cat /sys/class/drm/card1/device/hwmon/hwmon*/power1_cap) vddgfx=$(grep -A1 OD_VDDGFX_OFFSET /sys/class/drm/card1/device/pp_od_clk_voltage | tail -1)"
  echo "commit=$(git rev-parse --short HEAD) dirty='$(git status --porcelain | tr '\n' ';')'"
} | tee "$out/arm.txt"

# request set: id = set_index * 100 + prompt_index; map.txt keeps set and name per id
if [ "$quick" -gt 0 ]; then sets="32768"; n=$quick; else sets="8192 16384 32768"; n=20; fi
: > "$out/ref-requests.jsonl"; : > "$out/map.txt"
si=0
for L in $sets; do
  si=$((si + 1)); d=$(cat ".work/a2-prompts/L$L/doclen"); i=0
  for tf in .work/a2-prompts/L$L/p*.tokens; do
    i=$((i + 1)); [ "$i" -le "$n" ] || break
    id=$((si * 100 + i)); p=$(tr -s ' \n' ',' < "$tf" | sed 's/,$//')
    echo "{\"id\":$id,\"prompt\":[$p],\"n\":64,\"spec\":false,\"temperature\":0,\"ckpt\":[$d]}" >> "$out/ref-requests.jsonl"
    echo "$id L$L $(basename "$tf" .tokens)" >> "$out/map.txt"
  done
done
runenv="BARO_SERVE=1 BARO_SPEC=0 BARO_TMAX=32768"

# reference arm, cached the P17 way
key=$(printf '%s\n' "$ha" "$packsha" "n64 T0 spec0" "$(sha256sum "$out/ref-requests.jsonl" | cut -c1-64)" | sha256sum | cut -c1-24)
refdir=.work/refcache/a2/$key
if [ -s "$refdir/ref-ids.jsonl" ]; then
  echo "refcache hit key=$key" | tee -a "$out/arm.txt"
else
  echo "refcache miss key=$key" | tee -a "$out/arm.txt"; mkdir -p "$refdir"
  env $runenv $envx "$ref" < "$out/ref-requests.jsonl" > "$out/ref.out" 2> "$out/ref.err" || true
  grep -q '"ready":true' "$out/ref.out" || { echo "FAIL a2-gate: reference never printed its ready line, see $out/ref.err"; exit 1; }
  python3 - "$out/ref.out" "$refdir/ref-ids.jsonl" <<'PY'
import json, sys
toks = {}
for line in open(sys.argv[1]):
    if not line.startswith("{"): continue
    try: d = json.loads(line)
    except Exception: continue
    if "tok" in d and "id" in d: toks.setdefault(d["id"], []).append(d["tok"])
with open(sys.argv[2], "w") as f:
    for i, ids in sorted(toks.items()): f.write(json.dumps({"id": i, "ids": ids}) + "\n")
print(f"reference: {len(toks)} prompts, {sum(len(v) for v in toks.values())} ids")
PY
  grep -E '^(BARO_MEGA|BARO_SPEC|BARO_TMAX|BARO_FORCE):|"ready"' "$out/ref.out" | head -5 > "$refdir/receipt.txt"
  cp "$out/arm.txt" "$refdir/arm.txt"
fi

# candidate arm: same requests plus "force":[ref ids]
python3 - "$out/ref-requests.jsonl" "$refdir/ref-ids.jsonl" "$out/cand-requests.jsonl" <<'PY'
import json, sys
ids = {json.loads(l)["id"]: json.loads(l)["ids"] for l in open(sys.argv[2])}
n = 0
with open(sys.argv[3], "w") as f:
    for line in open(sys.argv[1]):
        d = json.loads(line)
        if d["id"] not in ids or not ids[d["id"]]: sys.exit(f"FAIL a2-gate: no reference ids for request {d['id']}")
        d["force"] = ids[d["id"]]; f.write(json.dumps(d) + "\n"); n += 1
print(f"candidate requests: {n}")
PY
env $runenv $envx "$cand" < "$out/cand-requests.jsonl" > "$out/cand.out" 2> "$out/cand.err" || true
grep -q '"ready":true' "$out/cand.out" || { echo "FAIL a2-gate: candidate never printed its ready line, see $out/cand.err"; exit 1; }
echo "== receipts"; grep -E "^(BARO_MEGA|BARO_SPEC|BARO_TMAX):" "$out/cand.out" | head -3 || true; grep -m1 "\"ready\"" "$out/cand.out" | grep -oE "\"tmax\":[0-9]+" || true

python3 - "$out" "$minpct" <<'PY'
import json, re, statistics as st, sys, pathlib
out = pathlib.Path(sys.argv[1]); minpct = float(sys.argv[2])
names = {int(l.split()[0]): (l.split()[1], l.split()[2]) for l in (out / "map.txt").read_text().splitlines()}
agree, done, order = [], {}, []
for line in (out / "cand.out").read_text().splitlines():
    m = re.match(r"forced agreement: (\d+) / (\d+)", line)
    if m: agree.append((int(m[1]), int(m[2]))); continue
    if line.startswith("{"):
        try: d = json.loads(line)
        except Exception: continue
        if d.get("done") and "id" in d: done[d["id"]] = d; order.append(d["id"])
bad = False
rows = ["set prompt agree checked pct cached restore_s prefill_s tok_s"]
if len(agree) != len(order):
    print(f"VOID: {len(agree)} forced-agreement lines for {len(order)} done lines"); bad = True
for k, i in enumerate(order):
    s, name = names[i]; d = done[i]
    a, c = agree[k] if k < len(agree) else (0, 0)
    pct = 100.0 * a / c if c else 0.0
    if c == 0: print(f"VOID: {s} {name}: nothing checked"); bad = True
    elif pct < minpct: print(f"FAIL {s} {name}: forced agreement {a}/{c} = {pct:.1f}% < {minpct}%"); bad = True
    rows.append(f"{s} {name} {a} {c} {pct:.1f} {d.get('cached', 0)} {d.get('restore_s', 0):.4f} {d.get('prefill_s', 0):.3f} {d.get('tok_s', 0):.2f}")
(out / "results.txt").write_text("\n".join(rows) + "\n")
print("\n".join(rows))
for s in sorted({v[0] for v in names.values()}, key=lambda x: int(x[1:])):
    ids = [i for i in order if names[i][0] == s]
    if not ids: continue
    pcts = [100.0 * agree[order.index(i)][0] / max(agree[order.index(i)][1], 1) for i in ids]
    restored = sum(1 for i in ids if done[i].get("cached", 0) > 0)
    ts = [done[i]["tok_s"] for i in ids]
    print(f"{s}: {len(ids)} prompts, min agreement {min(pcts):.1f}%, mean {st.mean(pcts):.1f}%, restored {restored}/{len(ids)}, decode median {st.median(ts):.2f} tok/s (min {min(ts):.2f} max {max(ts):.2f})")
    if restored < len(ids) - 1: print(f"VOID: {s}: only {restored} of {len(ids)} requests restored a prefix checkpoint (expected all but the first)"); bad = True
    if s == "L32768": print(f"decode after 32k: {st.median(ts):.2f} tok/s")
sys.exit(1 if bad else 0)
PY
