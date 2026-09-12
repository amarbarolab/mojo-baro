#!/usr/bin/env bash
# usage: bench/fr-gen-corpus.sh OUTDIR [MAX_TOKENS_PER_PROMPT] [PARALLEL]
#
# FR-Spec option B: a corpus of the target model's OWN output, so the draft
# vocabulary is ranked by what Qwythos actually emits. llama-server on the
# Qwythos Q4_0 GGUF (same weights family as .work/engine-pack-q4), PARALLEL
# slots, one completion per line of bench/fr-gen-prompts.txt (never the 20
# bench/mtp-prompts, which are the A/B test set). Writes OUTDIR/gen-NNNN.txt,
# one file per prompt, for tools/fr-vocab.mojo.
#
# Runs as one gpu-wait job (the server holds VRAM for the whole run). Re-read
# the project whiteboard for a GPU hold before launching (CLAUDE.md s17).
set -euo pipefail
cd "$(dirname "$0")/.."
out=${1:?OUTDIR}; ntok=${2:-768}; par=${3:-8}
model=${FR_GEN_GGUF:-$HOME/Models/qwythos-9b-claude-mythos-5-1m-mtp-bf16/Qwythos-9B-Claude-Mythos-5-1M-MTP-Q4_0-pure.gguf}
server=${LLAMA_SERVER:-$HOME/llama.cpp/build/bin/llama-server}
prompts=bench/fr-gen-prompts.txt
if [ -z "${GPU_WAITING_ROOM_JOB:-}" ]; then
  exec gpu-wait run --vram 16 --timeout 14400 -- "$0" "$@"
fi
mkdir -p "$out"
port=$((20000 + RANDOM % 20000))
"$server" -m "$model" -ngl 99 -fa on -np "$par" -c $((par * (ntok + 512))) --port "$port" --no-webui \
  > "$out/server.log" 2>&1 &
srv=$!
trap 'kill "$srv" 2>/dev/null || true' EXIT
for _ in $(seq 1 240); do curl -sf "localhost:$port/health" >/dev/null && break; sleep 0.5; done
t0=$(date +%s)
n=0
while IFS= read -r p; do
  [ -z "$p" ] && continue
  n=$((n + 1)); f=$(printf '%s/gen-%04d.txt' "$out" "$n")
  jq -n --arg p "$p" --argjson m "$ntok" \
    '{messages:[{role:"user",content:$p}],max_tokens:$m,temperature:0.8,top_p:0.95,seed:1}' |
    curl -sf "localhost:$port/v1/chat/completions" -H 'content-type: application/json' -d @- |
    jq -r '.choices[0].message.content // empty' > "$f" &
  while [ "$(jobs -rp | wc -l)" -gt "$par" ]; do wait -n || true; done
done < "$prompts"
wait "$(jobs -rp | grep -v "^$srv\$")" 2>/dev/null || true
t1=$(date +%s)
words=$(cat "$out"/gen-*.txt | wc -w)
echo "fr-gen: $n prompts, $words words in $((t1 - t0)) s -> $out"
grep -E 'eval time' "$out/server.log" | tail -3
