#!/usr/bin/env bash
# P0a contract gate. The server stays resident for all HTTP checks in one GPU job.
# Usage: bench/p0a-contract-gate.sh OUTDIR [COUNT] [PORT]
# Live usage must be wrapped by gpu-wait, for example:
#   gpu-wait run --vram 24 --timeout 3600 -- bench/p0a-contract-gate.sh OUTDIR
# GATE_DRYRUN is the CPU-only preflight and intentionally stops before launch.
set -euo pipefail
cd "$(dirname "$0")/.."

out=${1:-.work/team-A/codex/p0a}
count=${2:-5}
port=${3:-0}
engine=${BARO_ENGINE:-.work/engine}
pack=${BARO_PACK:-.work/engine-pack-q4}
serve=${BARO_SERVE_BIN:-serve/target/release/baro-serve}
mkdir -p "$out"
exec > >(tee "$out/contract.log") 2>&1

fail() { echo "FAIL $1: $2"; exit 1; }
[ "$count" -ge 1 ] || fail args "count must be positive"
arm="$out/arm.txt"
cat > "$arm" <<EOF
backend=ollama
count=$count
seed=0
temperature=0
embeddings=deferred-P0a-e
EOF

# gate-dryrun must stop before a GPU process is launched while proving that
# arm parameters reached the gate.
if [ "${GATE_DRYRUN:-0}" = 1 ]; then
  echo "FAIL GPU: GATE_DRYRUN stops before baro-serve launch"
  exit 97
fi

[ -x "$serve" ] || fail setup "missing $serve"
[ -x "$engine" ] || fail setup "missing $engine"
[ -d "$pack" ] || fail setup "missing pack $pack"
command -v curl >/dev/null || fail setup "curl is required"
command -v python3 >/dev/null || fail setup "python3 is required"

"$serve" --engine "$engine" --pack "$pack" --port "$port" \
  > "$out/server.stdout" 2> "$out/server.stderr" &
srv=$!
cleanup() {
  kill -INT "$srv" 2>/dev/null || true
  wait "$srv" 2>/dev/null || true
  pkill -TERM -P "$srv" 2>/dev/null || true
}
trap cleanup EXIT

for _ in $(seq 1 1200); do
  if grep -q '^listening on' "$out/server.stdout"; then break; fi
  kill -0 "$srv" 2>/dev/null || fail start "server exited: $(tail -3 "$out/server.stderr")"
  sleep 0.5
done
url=$(python3 - "$out/server.stdout" <<'PY'
import re, sys
text = open(sys.argv[1]).read()
match = re.search(r"https?://[^\s]+", text)
if not match:
    raise SystemExit("no listening URL")
print(match.group(0))
PY
) || fail start "no listening URL"
actual_port=${url##*:}

curl -fsS "$url/health" > "$out/health.json" || fail health "GET /health failed"
curl -fsS "$url/v1/models" > "$out/models.json" || fail models "GET /v1/models failed"
python3 - "$out/health.json" "$out/models.json" "$arm" "$actual_port" <<'PY'
import json, sys
health = json.load(open(sys.argv[1]))
models = json.load(open(sys.argv[2]))
assert health.get("status") == "ok", health
items = models.get("data", [])
assert len(items) == 1 and items[0].get("id"), models
with open(sys.argv[3], "a") as f:
    f.write(f"port={sys.argv[4]}\n")
    f.write(f"model={items[0]['id']}\n")
print("read-back", health["limits"], items[0]["id"])
PY
model=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["data"][0]["id"])' "$out/models.json")

python3 - "$url" "$out" <<'PY'
import json, pathlib, sys
import urllib.error
import urllib.request

url, out = sys.argv[1], pathlib.Path(sys.argv[2])
rows = []
for path in ("/api/embeddings", "/v1/embeddings"):
    request = urllib.request.Request(
        url + path,
        json.dumps({"model": "pending", "input": "pending"}).encode(),
        {"content-type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request) as response:
            status, body = response.status, response.read()
    except urllib.error.HTTPError as error:
        status, body = error.code, error.read()
    payload = json.loads(body)
    error = payload.get("error", {})
    message = error.get("message", "") if isinstance(error, dict) else str(error)
    assert status == 501, (path, status, payload)
    assert "embeddings_pending" in message and "P0a-e" in message, (path, payload)
    rows.append({"path": path, "status": status, "message": message})
(out / "embeddings-pending.json").write_text(json.dumps(rows, indent=2) + "\n")
print("embeddings pending contract: 2/2 routes return 501")
PY

prompts=(
  "Reply with exactly the word amber."
  "What is two plus two? Reply with one digit."
  "Name the first month of the year."
  "Reply with exactly the word router."
  "Give one short word for a cold color."
)
pair_args=(--backend ollama --port "$actual_port" --model "$model" --count "$count" --mode parallel
  --seed 0 --temperature 0 --max-tokens 8 --result-log "$out/pair-results.jsonl")
for ((i=0; i<count; i++)); do
  pair_args+=(--prompt "${prompts[$((i % ${#prompts[@]}))]}")
done
rm -f "$out/pair-results.jsonl"
~/iTools/bin/pair-dispatch "${pair_args[@]}" > "$out/pair-dispatch.log" 2>&1 || fail pair-dispatch "see $out/pair-dispatch.log"
python3 - "$out/pair-results.jsonl" "$count" <<'PY'
import json, sys
rows = [json.loads(line) for line in open(sys.argv[1]) if line.strip()]
want = int(sys.argv[2])
assert len(rows) >= want, (len(rows), want)
assert all(row.get("ok") is True for row in rows[-want:]), rows[-want:]
print(f"PAIR client requests: {want}/{want} ok")
PY

python3 - "$url" "$model" "$count" "$out" <<'PY'
import json, pathlib, sys, urllib.request
url, model, count, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), pathlib.Path(sys.argv[4])
prompts = [
    "Reply with exactly the word amber.",
    "What is two plus two? Reply with one digit.",
    "Name the first month of the year.",
    "Reply with exactly the word router.",
    "Give one short word for a cold color.",
]
def post(path, body):
    request = urllib.request.Request(url + path, json.dumps(body).encode(),
                                      {"content-type": "application/json"})
    with urllib.request.urlopen(request) as response:
        return json.load(response)
rows = []
for i in range(count):
    prompt = prompts[i % len(prompts)]
    # Gate 1 compares the raw Ollama prompt with OpenAI completions. Without
    # raw=true, /api/generate deliberately applies the chat template and the
    # prompt token counts are not comparable.
    ollama = post("/api/generate", {"model": model, "prompt": prompt, "raw": True, "stream": False,
                                    "options": {"temperature": 0, "seed": 0, "num_predict": 8}})
    openai = post("/v1/completions", {"model": model, "prompt": prompt, "stream": False,
                                      "temperature": 0, "seed": 0, "max_tokens": 8})
    ollama_count = ollama.get("eval_count")
    usage = openai.get("usage", {})
    openai_count = usage.get("completion_tokens",
                              len(openai.get("choices", [{}])[0].get("tokens", [])))
    assert ollama_count is not None and ollama_count == openai_count, (i, ollama_count, openai_count)
    rows.append({"index": i, "ollama_eval_count": ollama_count,
                 "openai_completion_tokens": openai_count})
(out / "gate1-counts.json").write_text(json.dumps(rows, indent=2) + "\n")
print(f"gate 1 token counts: {count}/{count} equal")
PY

venv="$out/ollama-venv"
if [ ! -x "$venv/bin/python" ]; then
  python3 -m venv "$venv"
  "$venv/bin/pip" install --disable-pip-version-check --quiet ollama
fi
OLLAMA_URL="$url" OLLAMA_MODEL="$model" OLLAMA_OUT="$out" \
  "$venv/bin/python" - "$count" <<'PY'
import json, os, pathlib, sys
import ollama
count = int(sys.argv[1])
url = os.environ["OLLAMA_URL"]
model = os.environ["OLLAMA_MODEL"]
out = pathlib.Path(os.environ["OLLAMA_OUT"])
prompts = [
    "Reply with exactly the word amber.", "What is two plus two? Reply with one digit.",
    "Name the first month of the year.", "Reply with exactly the word router.",
    "Give one short word for a cold color.", "Reply with exactly the word cobalt.",
    "What is three plus one? Reply with one digit.", "Name one primary color.",
    "Reply with exactly the word engine.", "Give one short word for a warm color.",
    "Reply with exactly the word state.", "What is five minus two? Reply with one digit.",
    "Name one season.", "Reply with exactly the word client.",
    "Give one short word for a bright color.", "Reply with exactly the word model.",
    "What is six divided by two? Reply with one digit.", "Name one weekday.",
    "Reply with exactly the word latent.", "Give one short word for a dark color.",
]
client = ollama.Client(host=url)
def openai_chat(prompt, stream):
    import urllib.request
    body = {"model": model, "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 8, "temperature": 0, "seed": 0, "stream": stream}
    request = urllib.request.Request(url + "/v1/chat/completions", json.dumps(body).encode(),
                                     {"content-type": "application/json"})
    if stream:
        with urllib.request.urlopen(request) as response:
            chunks = []
            for line in response.read().decode().splitlines():
                if line.startswith("data: ") and line[6:] != "[DONE]":
                    chunks.append(json.loads(line[6:])["choices"][0].get("delta", {}).get("content", ""))
            return "".join(chunks)
    with urllib.request.urlopen(request) as response:
        return json.load(response)["choices"][0]["message"]["content"]
rows = []
for i in range(count * 4):
    prompt = prompts[i % len(prompts)]
    chat = client.chat(model=model, messages=[{"role": "user", "content": prompt}],
                       stream=False, options={"temperature": 0, "seed": 0, "num_predict": 8})
    chat_text = chat["message"]["content"] if isinstance(chat, dict) else chat.message.content
    assert chat_text == openai_chat(prompt, False), (i, chat_text)
    chunks = client.chat(model=model, messages=[{"role": "user", "content": prompt}],
                         stream=True, options={"temperature": 0, "seed": 0, "num_predict": 8})
    stream_text = "".join((chunk.get("message", {}).get("content", "") if isinstance(chunk, dict)
                           else chunk.message.content) for chunk in chunks)
    assert stream_text == chat_text == openai_chat(prompt, True), (i, stream_text, chat_text)
    generated = client.generate(model=model, prompt=prompt, stream=False,
                                options={"temperature": 0, "seed": 0, "num_predict": 8})
    generated_text = generated["response"] if isinstance(generated, dict) else generated.response
    assert generated_text == openai_chat(prompt, False), (i, generated_text)
    rows.append({"index": i, "chat": "pass", "chat_stream": "pass", "generate": "pass"})
(out / "gate2-client.json").write_text(json.dumps(rows, indent=2) + "\n")
print(f"gate 2 ollama client: {len(rows)}/{len(rows)} chat/generate stream and non-stream equal")
PY

echo "PASS p0a contract gates 1 and 2"
echo "DEFERRED p0a embeddings gate 4: coordinator-owned P0a-e wire"
