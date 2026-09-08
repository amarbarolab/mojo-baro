#!/usr/bin/env bash
# Strict identity closure: build the engine FROM the sources embedded in a BARO
# gguf (baro.kernel.src.* KVs), run it on .work/engine-pack/, gate on ref tokens.
# usage: tools/gguf-closure.sh MODEL-BARO.gguf [ref-tokens-file] [outdir]
# Spark ggufs (spark_kernels.mojo in the file list): harness = serve/spark.mojo at the
# gguf commit, run on .work/spark/pack-q8 with the gguf itself as tokenizer source,
# ref default .work/spark/ref/ref-tokens-64.txt. Run under gpu-wait.
set -euo pipefail
cd "$(dirname "$0")/.."
model=$1; ref=${2:-}; out=${3:-.work/gguf-src}
rm -rf "$out"; mkdir -p "$out"
./.venv/bin/python3 tools/gguf-extract.py "$model" --meta > "$out/meta.json"
jq -r '.["baro.kernel.files"]' "$out/meta.json" | tr ',' '\n' > "$out/FILES"
while read -r f; do mkdir -p "$out/$(dirname "$f")"; jq -r --arg k "baro.kernel.src.$f" '.[$k]' "$out/meta.json" > "$out/$f"; done < "$out/FILES"
jq -r '"commit: " + .["baro.kernel.commit"] + "  arch: " + .["baro.kernel.arch"] + "  files: " + .["baro.kernel.files"]' "$out/meta.json"
export MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_SIZE_PERCENT=10
# vendor arm: if the gguf carries the hipBLASLt shim sources, build them too
if [ -f "$out/shim/CMakeLists.txt" ]; then
  cmake -S "$out/shim" -B "$out/shim-build" -DCMAKE_BUILD_TYPE=Release >/dev/null && cmake --build "$out/shim-build" -j"$(nproc)" >/dev/null \
    && echo "shim built from gguf: $(ls "$out"/shim-build/*.so)" || { echo "shim build FAILED"; exit 1; }
fi
# Split layout (2026-09-08): no main() in the embedded sources; the harness is
# serve/engine.mojo at the gguf's commit, from git, never from the gguf.
entry="$out/engine.mojo"
if grep -qx spark_kernels.mojo "$out/FILES"; then
  kcommit=$(jq -r '.["baro.kernel.commit"]' "$out/meta.json")
  git show "$kcommit:serve/spark.mojo" > "$out/harness.mojo" || { echo "no serve/spark.mojo at gguf commit $kcommit"; exit 1; }
  ./.venv/bin/mojo build "$out/harness.mojo" -I "$out" -o .work/engine-closure 2>&1 | grep -E "error" -A3 && exit 1 || true
  BARO_PROMPT_TEXT=.work/spark/prompt.txt BARO_GGUF="$model" BARO_PACK=.work/spark/pack-q8 BARO_GEN=64 ./.work/engine-closure > "$out/run.log"
  grep -E "tok/s" "$out/run.log"
  sed 's/^generated:/GENERATED:/' "$out/run.log" | tools/check-tokens.sh "${ref:-.work/spark/ref/ref-tokens-64.txt}" /dev/stdin
  exit
fi
ref=${ref:-.work/engine-pack/ref-tokens-64.txt}
if [ -f "$out/window.mojo" ] && ! grep -q '^def main' "$out/engine.mojo" 2>/dev/null; then
  kcommit=$(jq -r '.["baro.kernel.commit"]' "$out/meta.json")
  git show "$kcommit:serve/engine.mojo" > "$out/harness.mojo" || { echo "no serve/engine.mojo at gguf commit $kcommit"; exit 1; }
  entry="$out/harness.mojo"; echo "split layout: harness serve/engine.mojo@$kcommit"
fi
./.venv/bin/mojo build "$entry" -I "$out" -o .work/engine-closure 2>&1 | grep -E "error" -A3 && exit 1 || true
./.work/engine-closure > "$out/run.log"
grep -E "tok/s|host_enqueue" "$out/run.log"
tools/check-tokens.sh "$ref" "$out/run.log"
