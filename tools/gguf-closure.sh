#!/usr/bin/env bash
# Strict identity closure: build the engine FROM the sources embedded in a BARO
# gguf (baro.kernel.src.* KVs), run it on .work/engine-pack/, gate on ref tokens.
# usage: tools/gguf-closure.sh MODEL-BARO.gguf [ref-tokens-file] [outdir]
# Spark-harness ggufs (spark_kernels.mojo in the file list -- spark2_5 itself and
# the profile-driven dense families llama/qwen2/granite all use serve/spark.mojo):
# harness = serve/spark.mojo at the gguf commit, profile/prompt/pack/reference all
# from the file itself (baro.kernel.src.profile.mojo, baro.run.prompt.tokens,
# baro.run.pack.tool + baro.run.src.<tool> + baro.run.pack.flags, baro.run.ref.tokens).
# Run under gpu-wait.
set -euo pipefail
cd "$(dirname "$0")/.."
model=$1; ref=${2:-}; out=${3:-.work/gguf-src}
rm -rf "$out"; mkdir -p "$out"
./.venv/bin/python3 tools/gguf-extract.py "$model" --meta > "$out/meta.json"
jq -r '.["baro.kernel.files"]' "$out/meta.json" | tr ',' '\n' > "$out/FILES"
while read -r f; do mkdir -p "$out/$(dirname "$f")"; jq -r --arg k "baro.kernel.src.$f" '.[$k]' "$out/meta.json" > "$out/$f"; done < "$out/FILES"
jq -r '"commit: " + .["baro.kernel.commit"] + "  arch: " + .["baro.kernel.arch"] + "  files: " + .["baro.kernel.files"]' "$out/meta.json"
# The file's own prompt and reference ids (baro.run.*, B1). A bake that carries
# them verifies with nothing outside itself; before they existed the qwen35moe
# branch defaulted to .work/moe-w3/one.tokens, a local path the gguf never
# carried and which no longer exists, so tools/gguf-verify.sh exited 1 on every
# MoE bake.
jq -er '.["baro.run.prompt.tokens"]' "$out/meta.json" > "$out/prompt.tokens" 2>/dev/null \
  && echo "prompt: from the file ($(wc -w < "$out/prompt.tokens") ids)" || rm -f "$out/prompt.tokens"
# tools/check-tokens.sh reads its ref file with `mapfile` (one id per line);
# baro.run.ref.tokens is a single space-separated KV string (bake.sh joins it
# that way so it round-trips through GGUF's string type), so it is reflowed
# to one-per-line here rather than in every branch that reads $ref (the MoE
# branch below used to do this itself; a spark bake hit the same mismatch --
# "expected <all 64 ids>, got <first id>" -- the day this comment was added).
# `// empty` (not `-e`) so a missing/null key yields a truly empty file, not
# the literal text "null" surviving through tr/grep as a false "from the file".
jq -r '.["baro.run.ref.tokens"] // empty' "$out/meta.json" 2>/dev/null | tr -s ' ' '\n' | grep -v '^$' > "$out/ref-embedded.txt"
if [ -s "$out/ref-embedded.txt" ]; then
  echo "reference: from the file ($(wc -l < "$out/ref-embedded.txt") ids)"
else
  rm -f "$out/ref-embedded.txt"
fi
[ -n "$ref" ] || { [ -s "$out/ref-embedded.txt" ] && ref="$out/ref-embedded.txt"; } || true
[ -n "${BARO_PROMPT:-}" ] || { [ -s "$out/prompt.tokens" ] && export BARO_PROMPT="$out/prompt.tokens"; } || true
export MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_SIZE_PERCENT=10
# vendor arm: if the gguf carries the hipBLASLt shim sources, build them too
if [ -f "$out/shim/CMakeLists.txt" ]; then
  cmake -S "$out/shim" -B "$out/shim-build" -DCMAKE_BUILD_TYPE=Release >/dev/null && cmake --build "$out/shim-build" -j"$(nproc)" >/dev/null \
    && echo "shim built from gguf: $(ls "$out"/shim-build/*.so)" || { echo "shim build FAILED"; exit 1; }
fi
# Split layout (2026-09-08): no main() in the embedded sources; the harness is
# serve/engine.mojo at the gguf's commit, from git, never from the gguf.
entry="$out/engine.mojo"
# serve/engine.mojo imports latentos; vendored at the repo root since
# briefs/2026-09-15-vendor-latentos.md, so tools/embed-files.py's EXT walk
# pulls the package into the file's own baro.kernel.src.latentos/* KVs and
# this closure needs no include outside $out for it (no more than it does
# for kernels/ or serve/).
kmodel=$(jq -r '.["baro.kernel.model"] // empty' "$out/meta.json")
if [ "$kmodel" = "qwen35moe" ]; then
  # qwen35moe FULL ENGINE closure (2026-09-12): the file carries the whole
  # engine closure, not just the expert kernels, so rebuild the engine from it
  # exactly as the qwythos path does and gate on reference tokens. The harness
  # (serve/engine.mojo) comes from git at the gguf commit, never from the gguf,
  # so a baked file cannot carry its own clock.
  kcommit=$(jq -r '.["baro.kernel.commit"]' "$out/meta.json")
  git show "$kcommit:serve/engine.mojo" > "$out/closure_main.mojo" || { echo "no serve/engine.mojo at gguf commit $kcommit"; exit 1; }
  for m in $(sed -n 's/^from \([a-z_]*\) import.*/\1/p' "$out/closure_main.mojo"); do
    [ -f "$out/$m.mojo" ] || ! git cat-file -e "$kcommit:serve/$m.mojo" 2>/dev/null || git show "$kcommit:serve/$m.mojo" > "$out/$m.mojo"
  done
  ./.venv/bin/mojo build "$out/closure_main.mojo" -I "$out" -D BARO_MODEL=qwen35moe -o .work/moe-engine-closure 2>&1 | grep -E "error" -A3 && exit 1 || true
  [ -x .work/moe-engine-closure ] || { echo "closure build FAILED (no binary)"; exit 1; }
  moeref=${ref:-.work/moe-closure-ref.txt}
  [ -f "$moeref" ] || { echo "no reference token file $moeref: the gguf carries no baro.run.ref.tokens and no file was given"; exit 1; }
  # The reference holds bare ids. A file still carrying the engine's own
  # "GENERATED:" prefix compares that word against a token id and reports a
  # mismatch at position 1 on a run that is actually identical.
  sed 's/^GENERATED: *//' "$moeref" | tr -s ' ' '\n' | grep -v '^$' > "$out/ref-ids.txt"
  moeref="$out/ref-ids.txt"
  moeprompt=${BARO_PROMPT:-.work/moe-w3/one.tokens}
  [ -f "$moeprompt" ] || { echo "no prompt token file $moeprompt: the gguf carries no baro.run.prompt.tokens and BARO_PROMPT is unset"; exit 1; }
  env BARO_MEGA=0 BARO_PREFILL=1 BARO_PACK=${BARO_PACK:-.work/moe-w1/pack} \
      BARO_PROMPT="$moeprompt" ./.work/moe-engine-closure > "$out/run.log" 2>&1 \
      || { echo "closure engine FAILED"; tail -20 "$out/run.log"; exit 1; }
  grep -E "tok/s_gen|mega fail word" "$out/run.log"
  tools/check-tokens.sh "$moeref" "$out/run.log"
  exit
fi
if grep -qx moe.mojo "$out/FILES" && ! grep -qx window.mojo "$out/FILES"; then
  # qwen35moe kernels-only bake (no engine closure in the file); the closure rebuilds the parity test from the embedded
  # kernels and checks blk.0 of THIS gguf against the numpy oracle (tools/moe-ref.py).
  kcommit=$(jq -r '.["baro.kernel.commit"]' "$out/meta.json")
  git show "$kcommit:kernels/test_moe_block.mojo" > "$out/harness.mojo" || { echo "no kernels/test_moe_block.mojo at gguf commit $kcommit"; exit 1; }
  ./.venv/bin/mojo build "$out/harness.mojo" -I "$out" -o .work/moe-closure 2>&1 | grep -E "error" -A3 && exit 1 || true
  ./.venv/bin/python3 tools/moe-ref.py --gguf "$model" --layer 0 > "$out/oracle.log"
  ./.work/moe-closure > "$out/run.log" 2>&1 || { echo "closure test FAILED"; tail -20 "$out/run.log"; exit 1; }
  grep -E "PASS|FAIL|rel|exact" "$out/run.log" | tail -12
  exit
fi
if grep -qx spark_kernels.mojo "$out/FILES"; then
  # serve/spark.mojo is its own harness (baked from git at the gguf's commit,
  # same rule as every other arch: the file cannot carry its own clock), but
  # unlike qwythos/qwen35moe it needs three more things the file must carry,
  # because before this there was no local .work/spark/ fixture on this box
  # and the closure could not run at all (2026-09-16, coordinator diagnosis):
  # a profile.mojo (dims/rope/activation read from THIS gguf by
  # tools/gen-profile.mojo, extracted automatically above since it rides in
  # baro.kernel.files like any other source), a prompt (baro.run.prompt.tokens,
  # extracted above into $out/prompt.tokens), and a pack, built fresh from the
  # embedded pack tool (baro.run.pack.tool names which of engine-pack.py /
  # spark-pack.py; baro.run.src.<that name> is its source, baro.run.pack.flags
  # its flags) rather than reused from a local directory.
  kcommit=$(jq -r '.["baro.kernel.commit"]' "$out/meta.json")
  git show "$kcommit:serve/spark.mojo" > "$out/harness.mojo" || { echo "no serve/spark.mojo at gguf commit $kcommit"; exit 1; }
  [ -f "$out/profile.mojo" ] || { echo "no profile.mojo embedded (baro.kernel.src.profile.mojo missing; re-bake with BARO_PROFILE set)"; exit 1; }
  ./.venv/bin/mojo build "$out/harness.mojo" -I "$out" -o .work/engine-closure 2>&1 | grep -E "error" -A3 && exit 1 || true
  [ -x .work/engine-closure ] || { echo "closure build FAILED (no binary)"; exit 1; }
  packtool=$(jq -r '.["baro.run.pack.tool"] // empty' "$out/meta.json")
  [ -n "$packtool" ] || { echo "no baro.run.pack.tool: file carries no pack builder, cannot build a pack from itself"; exit 1; }
  jq -er --arg k "baro.run.src.$packtool" '.[$k]' "$out/meta.json" > "$out/$packtool" 2>/dev/null \
    || { echo "no baro.run.src.$packtool embedded"; exit 1; }
  jq -er '.["baro.run.src.gguf-extract.py"]' "$out/meta.json" > "$out/gguf-extract.py" 2>/dev/null \
    || cp tools/gguf-extract.py "$out/gguf-extract.py"
  packflags=$(jq -r '.["baro.run.pack.flags"] // empty' "$out/meta.json")
  # shellcheck disable=SC2086
  ./.venv/bin/python3 "$out/$packtool" "$model" "$out/pack" $packflags > "$out/pack.log" 2>&1 \
    || { tail -20 "$out/pack.log"; echo "pack build FAILED, see $out/pack.log"; exit 1; }
  [ -s "$out/prompt.tokens" ] || { echo "no prompt (baro.run.prompt.tokens missing)"; exit 1; }
  [ -n "$ref" ] || { echo "no reference tokens (baro.run.ref.tokens missing and none given)"; exit 1; }
  BARO_PACK="$out/pack" BARO_PROMPT="$out/prompt.tokens" BARO_GEN=64 ./.work/engine-closure > "$out/run.log" 2>&1 \
    || { tail -20 "$out/run.log"; echo "closure engine FAILED"; exit 1; }
  grep -E "tok/s" "$out/run.log"
  sed 's/^generated:/GENERATED:/' "$out/run.log" | tools/check-tokens.sh "$ref" /dev/stdin
  exit
fi
ref=${ref:-.work/engine-pack-q4/ref-tokens-64.txt}
if [ -f "$out/window.mojo" ] && ! grep -q '^def main' "$out/engine.mojo" 2>/dev/null; then
  kcommit=$(jq -r '.["baro.kernel.commit"]' "$out/meta.json")
  git show "$kcommit:serve/engine.mojo" > "$out/closure_main.mojo" || { echo "no serve/engine.mojo at gguf commit $kcommit"; exit 1; }
  for m in $(sed -n 's/^from \([a-z_]*\) import.*/\1/p' "$out/closure_main.mojo"); do
    [ -f "$out/$m.mojo" ] || ! git cat-file -e "$kcommit:serve/$m.mojo" 2>/dev/null || git show "$kcommit:serve/$m.mojo" > "$out/$m.mojo"
  done
  entry="$out/closure_main.mojo"; echo "split layout: harness serve/engine.mojo@$kcommit"
fi
./.venv/bin/mojo build "$entry" -I "$out" -o .work/engine-closure 2>&1 | grep -E "error" -A3 && exit 1 || true
[ -x .work/engine-closure ] || { echo "closure build FAILED (no binary)"; exit 1; }
./.work/engine-closure > "$out/run.log"
grep -E "tok/s|host_enqueue" "$out/run.log"
tools/check-tokens.sh "$ref" "$out/run.log"
