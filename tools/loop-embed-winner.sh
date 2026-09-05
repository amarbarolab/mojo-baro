#!/usr/bin/env bash
# Re-embed the COMMITTED repo sources into a new gguf with lineage; never overwrites.
# usage: tools/loop-embed-winner.sh SRC-BARO.gguf ITER
set -euo pipefail
cd "$(dirname "$0")/.."
src=$1; iter=$2
dst="${src%.gguf}-loop-$iter.gguf"
[ ! -e "$dst" ] || { echo "refusing to overwrite $dst"; exit 1; }
parent=$(./.venv/bin/python3 tools/gguf-extract.py "$src" --meta | jq -r '.["baro.kernel.commit"]')
# gguf-embed flattens kernels/ and serve/ to basename; map each name back to its
# real repo path instead of guessing a directory. Fail loud on a miss -- a wrong
# path here silently ships a gguf with a source file missing.
files=""
for f in $(./.venv/bin/python3 tools/gguf-extract.py "$src" --meta | jq -r '.["baro.kernel.files"]' | tr ',' '\n'); do
  case "$f" in
    */*) p=$f ;;
    *)   if   [ -f "kernels/$f" ]; then p="kernels/$f"
         elif [ -f "serve/$f" ];   then p="serve/$f"
         else echo "cannot locate '$f' in kernels/ or serve/" >&2; exit 1; fi ;;
  esac
  [ -f "$p" ] || { echo "missing source: $p" >&2; exit 1; }
  files="$files $p"
done
BARO_KERNEL_PARENT="$parent" ./.venv/bin/python3 tools/gguf-embed.py "$src" "$dst" $files
echo "$dst" > .work/loop/CHAMPION
echo "wrote $dst (parent $parent); CHAMPION updated"
