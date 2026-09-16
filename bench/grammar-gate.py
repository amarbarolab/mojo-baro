"""JSON enforcement gates 1, 2 and 4 (bench/grammar-protocol.md) against a
running baro-serve. Python is the oracle here: `json` parses, `jsonschema`
validates; neither shares code with grammar/.

  bench/grammar-gate.py URL SERVE_LOG OUT_DIR

Requests go one at a time, so the Nth "grammar masked draws:" receipt in
SERVE_LOG (the engine's stdout as forwarded by baro-serve) belongs to the
Nth request sent here. Gate 4 per request: masked draws == accepted, no
MISMATCH word, and masked draws == completion_tokens (no reasoning) or
<= completion_tokens (reasoning on, the think tokens are unmasked).
"""
import json, pathlib, re, sys, time, urllib.request

import jsonschema

url, serve_log, out = sys.argv[1], pathlib.Path(sys.argv[2]), pathlib.Path(sys.argv[3])
out.mkdir(parents=True, exist_ok=True)
corpus = pathlib.Path("grammar/corpus")
names = [n for n in (corpus / "manifest.txt").read_text().split() if n]


def post(body):
    req = urllib.request.Request(f"{url}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.load(r)


def receipts():
    return re.findall(r"grammar masked draws: (\d+)\s+accepted: (\d+)\s+terminated: (\w+)(.*)", serve_log.read_text(errors="replace"))


cases = [(n, t, False) for t in (0.0, 0.7) for n in names]
cases.append((names[0], 0.7, True))
cases.append((names[20], 0.0, True))
rows, fails = [], 0
for name, temp, think in cases:
    schema = json.loads((corpus / name).read_text())
    body = {
        "messages": [{"role": "user", "content": "Reply with one JSON value that matches this JSON schema, filled with realistic data: " + json.dumps(schema)}],
        "max_tokens": 1024 if think else 400, "temperature": temp, "seed": 7,
        "response_format": {"type": "json_schema", "json_schema": {"name": name[:-5], "schema": schema}},
        "chat_template_kwargs": {"enable_thinking": think},
    }
    before = len(receipts())
    t0 = time.time()
    try:
        resp = post(body)
    except Exception as e:
        resp = {"error": str(e)}
    time.sleep(0.2)
    rc = receipts()
    rec = rc[before] if len(rc) > before else None
    text = ((resp.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
    if think and "</think>" in text:
        text = text.split("</think>", 1)[1]
    ntok = (resp.get("usage") or {}).get("completion_tokens")
    why = []
    try:
        jsonschema.validate(json.loads(text), schema)
    except Exception as e:
        why.append(f"invalid: {type(e).__name__}: {str(e)[:80]}")
    if rec is None:
        why.append("no masked-draw receipt")
    else:
        draws, acc, _, tail = int(rec[0]), int(rec[1]), rec[2], rec[3]
        if draws != acc or "MISMATCH" in tail:
            why.append(f"receipt desync {draws}/{acc}")
        if ntok is None or (draws != ntok if not think else not (0 < draws <= ntok)):
            why.append(f"masked draws {draws} vs completion_tokens {ntok}")
    ok = not why
    fails += not ok
    row = {"schema": name, "temperature": temp, "thinking": think, "ok": ok, "why": why, "receipt": rec,
           "completion_tokens": ntok, "finish": ((resp.get("choices") or [{}])[0]).get("finish_reason"),
           "seconds": round(time.time() - t0, 2), "text": text, "raw": resp}
    rows.append(row)
    print(("PASS" if ok else "FAIL"), name, f"T={temp}", "think" if think else "", f"tokens={ntok}", f"receipt={rec[:2] if rec else None}", "; ".join(why), flush=True)

(out / "results.json").write_text(json.dumps(rows, indent=1))
print(f"RESULT {'PASS' if fails == 0 else 'FAIL'}: {len(rows) - fails}/{len(rows)} requests valid with matching receipts")
sys.exit(1 if fails else 0)
