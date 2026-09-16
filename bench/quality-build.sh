#!/usr/bin/env bash
# Build an engine binary + pack from a self-describing BARO gguf, for the
# quality-protocol harness (bench/quality-protocol.md). Adapted from
# tools/gguf-closure.sh's build steps (engine.mojo/spark.mojo split-layout
# + build_pack_from_file), dropping its ref-token verify tail: this script's
# job ends at "binary + pack exist", never runs a timed/scored request.
# usage: bench/quality-build.sh MODEL.gguf OUTDIR
set -euo pipefail
cd "$(dirname "$0")/.."
model=$1; out=$2
rm -rf "$out"; mkdir -p "$out"
$HOME/Projects/mojo/mojo-baro/.venv/bin/python3 tools/gguf-extract.py "$model" --meta > "$out/meta.json"
jq -r '.["baro.kernel.files"]' "$out/meta.json" | tr ',' '\n' > "$out/FILES"
while read -r f; do
  mkdir -p "$out/$(dirname "$f")"
  jq -r --arg k "baro.kernel.src.$f" '.[$k]' "$out/meta.json" > "$out/$f"
done < "$out/FILES"
kcommit=$(jq -r '.["baro.kernel.commit"]' "$out/meta.json")
kmodel=$(jq -r '.["baro.kernel.model"] // empty' "$out/meta.json")
echo "commit: $kcommit  arch: $(jq -r '.["baro.kernel.arch"]' "$out/meta.json")  model: $kmodel"

build_pack_from_file() {
  local packtool packflags
  packtool=$(jq -r '.["baro.run.pack.tool"] // empty' "$out/meta.json")
  [ -n "$packtool" ] || { echo "no baro.run.pack.tool: file carries no pack builder"; exit 1; }
  jq -er --arg k "baro.run.src.$packtool" '.[$k]' "$out/meta.json" > "$out/$packtool" 2>/dev/null \
    || { echo "no baro.run.src.$packtool embedded"; exit 1; }
  jq -er '.["baro.run.src.gguf-extract.py"]' "$out/meta.json" > "$out/gguf-extract.py" 2>/dev/null \
    || cp tools/gguf-extract.py "$out/gguf-extract.py"
  packflags=$(jq -r '.["baro.run.pack.flags"] // empty' "$out/meta.json")
  # shellcheck disable=SC2086
  $HOME/Projects/mojo/mojo-baro/.venv/bin/python3 "$out/$packtool" "$model" "$out/pack" $packflags > "$out/pack.log" 2>&1 \
    || { tail -20 "$out/pack.log"; echo "pack build FAILED, see $out/pack.log"; exit 1; }
}

if [ "$kmodel" = "qwen35moe" ]; then
  git show "$kcommit:serve/engine.mojo" > "$out/closure_main.mojo" || { echo "no serve/engine.mojo at $kcommit"; exit 1; }
  for m in $(sed -n 's/^from \([a-z_]*\) import.*/\1/p' "$out/closure_main.mojo"); do
    [ -f "$out/$m.mojo" ] || ! git cat-file -e "$kcommit:serve/$m.mojo" 2>/dev/null || git show "$kcommit:serve/$m.mojo" > "$out/$m.mojo"
  done
  $HOME/Projects/mojo/mojo-baro/.venv/bin/mojo build "$out/closure_main.mojo" -I "$out" -D BARO_MODEL=qwen35moe -o "$out/engine" 2>&1 | tee "$out/build.log" | grep -E "error" -A3 && exit 1 || true
  [ -x "$out/engine" ] || { echo "engine build FAILED"; cat "$out/build.log"; exit 1; }
  build_pack_from_file
  echo "engine: $out/engine  pack: $out/pack"
  exit 0
fi

if grep -qx spark_kernels.mojo "$out/FILES"; then
  git show "$kcommit:serve/spark.mojo" > "$out/harness.mojo" || { echo "no serve/spark.mojo at $kcommit"; exit 1; }
  [ -f "$out/profile.mojo" ] || { echo "no profile.mojo embedded"; exit 1; }
  $HOME/Projects/mojo/mojo-baro/.venv/bin/mojo build "$out/harness.mojo" -I "$out" -o "$out/engine" 2>&1 | tee "$out/build.log" | grep -E "error" -A3 && exit 1 || true
  [ -x "$out/engine" ] || { echo "engine build FAILED"; cat "$out/build.log"; exit 1; }
  build_pack_from_file
  echo "engine: $out/engine  pack: $out/pack"
  exit 0
fi

# split layout (qwen35, dense): serve/engine.mojo from git at the gguf commit
entry="$out/engine.mojo"
if [ -f "$out/window.mojo" ] && ! grep -q '^def main' "$out/engine.mojo" 2>/dev/null; then
  git show "$kcommit:serve/engine.mojo" > "$out/closure_main.mojo" || { echo "no serve/engine.mojo at $kcommit"; exit 1; }
  for m in $(sed -n 's/^from \([a-z_]*\) import.*/\1/p' "$out/closure_main.mojo"); do
    [ -f "$out/$m.mojo" ] || ! git cat-file -e "$kcommit:serve/$m.mojo" 2>/dev/null || git show "$kcommit:serve/$m.mojo" > "$out/$m.mojo"
  done
  entry="$out/closure_main.mojo"
fi
$HOME/Projects/mojo/mojo-baro/.venv/bin/mojo build "$entry" -I "$out" -o "$out/engine" 2>&1 | tee "$out/build.log" | grep -E "error" -A3 && exit 1 || true
[ -x "$out/engine" ] || { echo "engine build FAILED"; cat "$out/build.log"; exit 1; }
build_pack_from_file
echo "engine: $out/engine  pack: $out/pack"
