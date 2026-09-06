#!/usr/bin/env python3
"""Logging proxy for an OpenAI-compatible backend: every request and response
body (SSE reassembled) lands as one JSON line in LOG. Used to see exactly what
an agent harness (DeerFlow) sends and what the model answers, independent of
the harness's own event stream.

usage: tools/openai-tap.py LISTEN_PORT UPSTREAM_BASE LOG
   eg: tools/openai-tap.py 8085 http://127.0.0.1:8084 .work/chat/tap.jsonl
"""
import json, sys, time, urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

port, upstream, logp = int(sys.argv[1]), sys.argv[2].rstrip("/"), sys.argv[3]


class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def _fwd(self, body):
        req = urllib.request.Request(upstream + self.path, data=body, method=self.command)
        for k in ("content-type", "authorization", "accept"):
            if self.headers.get(k):
                req.add_header(k, self.headers[k])
        t0 = time.time()
        try:
            r = urllib.request.urlopen(req, timeout=900)
            status, ctype, data = r.status, r.headers.get("content-type", ""), r.read()
        except urllib.error.HTTPError as e:
            status, ctype, data = e.code, e.headers.get("content-type", ""), e.read()
        self.send_response(status)
        self.send_header("content-type", ctype)
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)
        rec = {"ts": t0, "dt": round(time.time() - t0, 3), "path": self.path, "status": status}
        try:
            rec["request"] = json.loads(body) if body else None
        except Exception:
            rec["request"] = body.decode(errors="replace")[:4000]
        if "text/event-stream" in ctype:
            chunks = []
            for line in data.decode(errors="replace").split("\n"):
                if line.startswith("data: ") and line[6:].strip() != "[DONE]":
                    try:
                        chunks.append(json.loads(line[6:]))
                    except Exception:
                        pass
            msg = {"content": "", "reasoning_content": "", "tool_calls": {}}
            fr = None
            for c in chunks:
                for ch in c.get("choices", []):
                    d = ch.get("delta", {})
                    msg["content"] += d.get("content") or ""
                    msg["reasoning_content"] += d.get("reasoning_content") or ""
                    for tc in d.get("tool_calls") or []:
                        i = tc.get("index", 0)
                        m = msg["tool_calls"].setdefault(i, {"name": "", "arguments": ""})
                        f = tc.get("function", {})
                        m["name"] += f.get("name") or ""
                        m["arguments"] += f.get("arguments") or ""
                    fr = ch.get("finish_reason") or fr
            rec["response"] = {"stream": True, "n_chunks": len(chunks), "finish_reason": fr, "message": msg,
                               "usage": chunks[-1].get("usage") if chunks else None}
        else:
            try:
                rec["response"] = json.loads(data)
            except Exception:
                rec["response"] = data.decode(errors="replace")[:4000]
        with open(logp, "a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def do_POST(self):
        self._fwd(self.rfile.read(int(self.headers.get("content-length", 0))))

    def do_GET(self):
        self._fwd(None)


ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()
