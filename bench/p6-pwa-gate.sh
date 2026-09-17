#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
root=$PWD
out=${1:-.work/team-C/codex/p6-pwa}
pack="$out/fake-pack"
bin=${BARO_SERVE_BIN:-serve/target/release/baro-serve}
mkdir -p "$out" "$pack"
: > "$out/SUMMARY.txt"

pass() { echo "PASS $1: $2" | tee -a "$out/SUMMARY.txt"; }
fail() { echo "FAIL $1: $2" | tee -a "$out/SUMMARY.txt" >&2; exit 1; }

cleanup() {
    if [ -n "${srv:-}" ] && kill -0 "$srv" 2>/dev/null; then
        kill "$srv" 2>/dev/null || true
        wait "$srv" 2>/dev/null || true
    fi
    if [ -n "${browser_port:-}" ]; then
        ~/iTools/bin/cdp-headless --kill "$browser_port" >> "$out/cdp-kill.log" 2>&1 || true
    fi
}
trap cleanup EXIT

[ -x bench/fixtures/fake-engine.py ] || fail setup "fake engine is not executable"
[ -x "$bin" ] || fail setup "build $bin first"

python3 - "$pack/tokenizer.json" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
tokenizer = {
    "version": "1.0",
    "truncation": None,
    "padding": None,
    "added_tokens": [],
    "normalizer": None,
    "pre_tokenizer": {"type": "Whitespace"},
    "post_processor": None,
    "decoder": {"type": "WordPiece", "prefix": "##", "cleanup": True},
    "model": {
        "type": "WordLevel",
        "vocab": {"<unk>": 0, "hello": 1, "world": 2, "!": 3},
        "unk_token": "<unk>",
    },
}
path.write_text(json.dumps(tokenizer))
PY
printf '{"add_bos":false}\n' > "$pack/tokenizer-meta.json"

"$root/$bin" \
    --engine "$root/bench/fixtures/fake-engine.py" \
    --pack "$pack" --port 0 \
    > "$out/server.stdout" 2> "$out/server.stderr" &
srv=$!
for _ in $(seq 1 200); do
    if grep -q '^listening on ' "$out/server.stdout"; then
        break
    fi
    kill -0 "$srv" 2>/dev/null || fail start "server exited: $(tail -3 "$out/server.stderr")"
    sleep 0.05
done
url=$(grep -m1 -oE 'http://[^ ]+' "$out/server.stdout") || fail start "no listening line"
pass start "$url"

curl -fsS "$url/" > "$out/index.html" || fail assets "GET / failed"
python3 - "$url" "$out/index.html" "$out/assets.tsv" <<'PY'
import pathlib
import re
import sys
import urllib.request
from urllib.parse import urljoin

base, html_path, receipt_path = sys.argv[1:]
html = pathlib.Path(html_path).read_text()
refs = sorted(set(re.findall(r'(?:src|href)=["\']([^"\']+)["\']', html)))
if not refs:
    raise SystemExit("index has no referenced assets")
rows = []
for ref in refs:
    if ref.startswith(("#", "http:", "https:", "data:")):
        continue
    target = urljoin(base + "/", ref)
    with urllib.request.urlopen(target, timeout=10) as response:
        body = response.read()
        content_type = response.headers.get("content-type", "")
        if response.status != 200:
            raise SystemExit(f"{ref}: HTTP {response.status}")
        if ref.endswith((".js", ".mjs")) and "javascript" not in content_type:
            raise SystemExit(f"{ref}: wrong content type {content_type}")
        if ref.endswith(".css") and "text/css" not in content_type:
            raise SystemExit(f"{ref}: wrong content type {content_type}")
        if ref.endswith(".webmanifest") and "json" not in content_type:
            raise SystemExit(f"{ref}: wrong content type {content_type}")
        rows.append(f"{ref}\t{response.status}\t{content_type}\t{len(body)}")
pathlib.Path(receipt_path).write_text("\n".join(rows) + "\n")
print(f"{len(rows)} assets: 200 with expected content types")
PY
pass assets "GET / and $(wc -l < "$out/assets.tsv") referenced assets returned 200"

python3 - "$url" "$out/manifest.json" <<'PY'
import json
import pathlib
import sys
import urllib.request

url, output = sys.argv[1:]
with urllib.request.urlopen(url + "/web/manifest.webmanifest", timeout=10) as response:
    manifest = json.load(response)
if not manifest.get("name") or not manifest.get("start_url"):
    raise SystemExit("manifest needs name and start_url")
pathlib.Path(output).write_text(json.dumps(manifest, indent=2))
print(f"manifest name={manifest['name']!r} start_url={manifest['start_url']!r}")
PY
pass manifest "manifest parses"

browser_info=$(~/iTools/bin/cdp-headless --width 400 --height 860 2> "$out/cdp-headless.stderr") || fail browser "cdp-headless failed"
printf '%s\n' "$browser_info" > "$out/cdp-headless.log"
browser_port=$(printf '%s\n' "$browser_info" | tail -n1)
[[ "$browser_port" =~ ^[0-9]+$ ]] || fail browser "invalid CDP port $browser_port"

python3 ~/iTools/browser-cdp/page-shot/page-shot.py "$browser_port" "$url/" "$out/p6-pwa-400x860.png" --size=400x860 --wait=1
~/iTools/bin/cdp-eval --port "$browser_port" --tab "$url" \
    'navigator.serviceWorker.ready.then(r => r.active ? r.active.scriptURL : "")' \
    --timeout 20 > "$out/sw-url.txt"
grep -q 'sw.js' "$out/sw-url.txt" || fail service-worker "service worker did not become active: $(cat "$out/sw-url.txt")"
pass service-worker "registered $(tr -d '\n' < "$out/sw-url.txt")"

expected=$(curl -fsS "$url/detokenize" -H 'content-type: application/json' \
    -d '{"tokens":[1,2,3]}' | jq -r .content)
printf '%s\n' "$expected" > "$out/fake-sequence.txt"
[ -n "$expected" ] || fail chat "fake sequence detokenized to empty text"

~/iTools/bin/cdp-eval --port "$browser_port" --tab "$url" --timeout 20 \
    '(() => { const a = document.querySelector("#server-address"); if (a) { a.value = location.origin; a.dispatchEvent(new Event("input", {bubbles:true})); a.dispatchEvent(new Event("change", {bubbles:true})); } const input = document.querySelector("#message-input"); const send = document.querySelector("#send-button"); if (!input || !send) throw new Error("P6 selectors missing"); input.value = "hello"; input.dispatchEvent(new Event("input", {bubbles:true})); send.click(); return "submitted"; })()' \
    > "$out/chat-submit.txt"
~/iTools/bin/cdp-eval --port "$browser_port" --tab "$url" --timeout 30 \
    'new Promise((resolve, reject) => { const deadline = Date.now() + 20000; const tick = () => { const nodes = [...document.querySelectorAll("[data-role=assistant-message]")]; const text = nodes.length ? nodes.at(-1).textContent.trim() : ""; if (text) resolve(text); else if (Date.now() > deadline) reject(new Error("assistant reply missing")); else setTimeout(tick, 100); }; tick(); })' \
    > "$out/chat-dom.txt" || fail chat "DOM reply did not arrive"
grep -Fqx "$expected" "$out/chat-dom.txt" || fail chat "DOM reply mismatch: expected $(printf '%q' "$expected"), got $(cat "$out/chat-dom.txt")"
pass chat "DOM assistant reply equals fake sequence $(printf '%q' "$expected")"

~/iTools/bin/cdp-eval --port "$browser_port" --tab "$url" \
    'location.reload(); "reloading"' > "$out/reload.txt"
sleep 1
~/iTools/bin/cdp-eval --port "$browser_port" --tab "$url" --timeout 20 \
    '(() => { const nodes = [...document.querySelectorAll("[data-role=assistant-message]")]; return nodes.length ? nodes.at(-1).textContent.trim() : ""; })()' \
    > "$out/reload-dom.txt"
grep -Fqx "$expected" "$out/reload-dom.txt" || fail reload "conversation was not restored"
pass reload "conversation restored from localStorage"

overflow=$(~/iTools/bin/cdp-eval --port "$browser_port" --tab "$url" \
    '({scrollWidth: document.documentElement.scrollWidth, clientWidth: document.documentElement.clientWidth})')
printf '%s\n' "$overflow" > "$out/overflow.json"
python3 - "$out/overflow.json" <<'PY'
import json
import sys
d = json.load(open(sys.argv[1]))
if d["scrollWidth"] > d["clientWidth"]:
    raise SystemExit(d)
print(f"scrollWidth={d['scrollWidth']} clientWidth={d['clientWidth']}")
PY
pass responsive "400px document has no horizontal overflow: $(tr -d '\n' < "$out/overflow.json")"

python3 ~/iTools/browser-cdp/page-shot/page-shot.py "$browser_port" "$url/" "$out/p6-pwa-1024x768.png" --size=1024x768 --wait=1

kill "$srv"
wait "$srv" 2>/dev/null || true
srv=
~/iTools/bin/cdp-eval --port "$browser_port" --tab "$url" --timeout 20 \
    'location.reload(); "offline reload"' > "$out/offline-reload.txt"
sleep 1
~/iTools/bin/cdp-eval --port "$browser_port" --tab "$url" --timeout 20 \
    '({controlled: Boolean(navigator.serviceWorker.controller), shell: Boolean(document.querySelector("#message-input")), title: document.title})' \
    > "$out/offline.json"
python3 - "$out/offline.json" <<'PY'
import json
import sys
d = json.load(open(sys.argv[1]))
if not d["controlled"] or not d["shell"]:
    raise SystemExit(d)
print(f"controlled={d['controlled']} shell={d['shell']} title={d['title']!r}")
PY
pass offline "service-worker shell loads with server stopped: $(tr -d '\n' < "$out/offline.json")"

echo "PASS P6-pwa gate: $out/SUMMARY.txt"
