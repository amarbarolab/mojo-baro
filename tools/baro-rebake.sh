#!/usr/bin/env bash
# Re-bake an existing BARO GGUF at the current commit's sources.
# Extracts arch, prompt, ref ids, pack flags and env from the old file,
# then calls tools/bake.sh to produce a new file.
#
# usage: tools/baro-rebake.sh EXISTING-BARO.gguf [DST.gguf]
#   DST defaults to the same name with the sha replaced by the current HEAD.
#   Extra --kv=... arguments are passed through to bake.sh (hw receipt keys).
#
# The old file's reference tokens are the gate: if the new sources produce
# different output, tools/baro verify on the new file will catch it.
# No GPU needed here; the bake itself is CPU-only (embedding sources).
set -euo pipefail
cd "$(dirname "$0")/.."
[ $# -ge 1 ] || { echo "usage: tools/baro-rebake.sh EXISTING-BARO.gguf [DST.gguf] [--kv=...]" >&2; exit 1; }
src=$1; shift
[ -f "$src" ] || { echo "no file: $src" >&2; exit 1; }

sha=$(git rev-parse --short HEAD)
meta=$(./.venv/bin/python3 tools/gguf-extract.py "$src" --meta)
jget() { echo "$meta" | ./.venv/bin/python3 -c "import json,sys;v=json.load(sys.stdin).get('$1','');print(v if v else '')"; }

gpu_arch=$(jget baro.kernel.arch)
[ -n "$gpu_arch" ] || { echo "no baro.kernel.arch in $src: not a BARO gguf" >&2; exit 1; }

harness=$(jget baro.run.harness.path)
kmodel=$(jget baro.kernel.model)
if [ "$harness" = "serve/spark.mojo" ]; then
  arch=spark
elif [ "$kmodel" = "qwen35moe" ]; then
  arch=qwen35moe
else
  arch=qwythos
fi

old_commit=$(jget baro.kernel.commit)
if [ "$old_commit" = "$sha" ]; then
  echo "already at $sha, nothing to rebake" >&2; exit 0
fi

# Destination: replace the old sha with the new one, or use the argument
if [ $# -ge 1 ] && [[ "$1" != --* ]]; then
  dst=$1; shift
else
  dst=$(echo "$src" | sed "s/-BARO-[0-9a-f]*\.gguf/-BARO-$sha.gguf/")
fi
[ ! -e "$dst" ] || { echo "destination exists: $dst" >&2; exit 1; }

# Extract prompt and ref from the old file into a temp dir
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

prompt_str=$(jget baro.run.prompt.tokens)
[ -n "$prompt_str" ] || { echo "no baro.run.prompt.tokens in $src" >&2; exit 1; }
echo "$prompt_str" | tr ' ' '\n' | grep -v '^$' > "$tmp/prompt.tokens"

ref_str=$(jget baro.run.ref.tokens)
[ -n "$ref_str" ] || { echo "no baro.run.ref.tokens in $src" >&2; exit 1; }
echo "$ref_str" | tr ' ' '\n' | grep -v '^$' > "$tmp/ref-ids.txt"

packflags=$(jget baro.run.pack.flags)
run_env=$(jget baro.run.env)
pack_tool=$(jget baro.run.pack.tool)

# The source GGUF for bake.sh is the ORIGINAL model (without BARO keys),
# but using the old BARO file works too: gguf-embed.py replaces the baro.*
# keys and copies all tensors. With btrfs reflinks the tensor data is shared.
echo "rebaking $src ($old_commit) -> $dst ($sha), arch $arch"

env_args=()
[ -z "$run_env" ] || env_args=(BARO_RUN_ENV="$run_env")
[ -z "$pack_tool" ] || env_args+=("BARO_PACK_TOOL=$pack_tool")

# Spark needs BARO_PROFILE; extract it from the old file if present
profile_str=$(jget baro.kernel.src.profile.mojo)
if [ -n "$profile_str" ]; then
  echo "$profile_str" > "$tmp/profile.mojo"
  env_args+=("BARO_PROFILE=$tmp/profile.mojo")
fi

env "${env_args[@]}" tools/bake.sh "$src" "$dst" "$arch" "$tmp/ref-ids.txt" "$tmp/prompt.tokens" "$packflags" "$@"
echo "rebaked: $dst"
echo "verify:  tools/baro verify $dst"
