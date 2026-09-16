#!/usr/bin/env bash
# Live check that `tools/baro serve` answers /v1/chat/completions (not 503) per model.
#   tools/baro-serve-chat-check.sh OUT_DIR MODEL.gguf [...]   (run inside gpu-wait)
set -uo pipefail
here=$(cd "$(dirname "$0")/.." && pwd)
out=$1; shift; mkdir -p "$out"; port=8093; fails=0
for m in "$@"; do
  n=$(basename "${m%.gguf}")
  "$here/tools/baro" serve "$m" --port $port > "$out/$n.serve.log" 2>&1 &
  pid=$!
  for _ in $(seq 300); do curl -sf localhost:$port/health >/dev/null 2>&1 && break; kill -0 $pid 2>/dev/null || break; sleep 1; done
  code=$(curl -s -o "$out/$n.json" -w '%{http_code}' localhost:$port/v1/chat/completions -H 'content-type: application/json' \
    -d '{"messages":[{"role":"user","content":"What is the capital of France? Answer in one word."}],"max_tokens":24,"temperature":0,"chat_template_kwargs":{"enable_thinking":false}}')
  text=$(jq -r '.choices[0].message.content // .error.message' "$out/$n.json" 2>/dev/null | tr '\n' ' ' | cut -c1-80)
  if [ "$code" = 200 ] && grep -qi paris <<<"$text"; then echo "PASS $n: $text"; else echo "FAIL $n: HTTP $code $text"; fails=$((fails+1)); fi
  pkill -P $pid 2>/dev/null; kill $pid 2>/dev/null; wait $pid 2>/dev/null; sleep 2
done
echo "RESULT $([ $fails = 0 ] && echo PASS || echo "FAIL $fails")"
