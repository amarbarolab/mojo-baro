#!/usr/bin/env bash
# CPU preflight before any GPU job (PROTOCOL-RULES P15).
#   bench/preflight.sh          run everything below, write .work/preflight.ok keyed to the content of every tracked and untracked (non-ignored) file
#   bench/preflight.sh --check  exit 1 unless .work/preflight.ok matches the current tree (gate scripts call this)
# Checks: tools/ci-checks.sh (docs, py/sh syntax, census, every bench/*.mojo builds), every run-tests.sh test
# binary builds, dense and MoE engines build, baro-serve builds and its unit tests pass.
set -euo pipefail
cd "$(dirname "$0")/.."
accel=${BARO_TARGET_ACCELERATOR:-gfx1100}
# exchange/ holds reports and conference notes that other lanes write while a job is queued; nothing there feeds a build or a gate.
stamp() { git ls-files -co --exclude-standard | grep -v '^exchange/' | sort | while IFS= read -r f; do [ -f "$f" ] && printf '%s\n' "$f"; done | tr '\n' '\0' | xargs -0 sha256sum | sha256sum | cut -c1-32; }
if [ "${1:-}" = --check ]; then
  [ "$(cat .work/preflight.ok 2>/dev/null)" = "$(stamp)" ] && exit 0
  echo "FAIL preflight: no passing bench/preflight.sh for the current file contents; run it first" >&2
  exit 1
fi
rm -f .work/preflight.ok; mkdir -p .work/preflight
step() { echo "== $1"; }
fail() { echo "FAIL preflight $1: see $2" >&2; exit 1; }
step ci-checks;  tools/ci-checks.sh > .work/preflight/ci.log 2>&1 || fail ci-checks .work/preflight/ci.log
step run-tests-builds
S="$PWD/.work/shim-build"
awk '/\\$/{printf "%s ", substr($0, 1, length($0) - 1); next} {print}' run-tests.sh | grep -E 'mojo build ' | sed 's#-o [^ ]*#-o .work/preflight/bin#' | while read -r cmd; do
  cmd="$cmd --target-accelerator $accel"
  eval "$cmd" > .work/preflight/build.log 2>&1 || fail "build: $cmd" .work/preflight/build.log
done
step engines
./.venv/bin/mojo build --target-accelerator "$accel" serve/engine.mojo -I . -I kernels -o .work/preflight/bin > .work/preflight/build.log 2>&1 || fail "engine dense" .work/preflight/build.log
./.venv/bin/mojo build --target-accelerator "$accel" serve/engine.mojo -I . -I kernels -D BARO_MODEL=qwen35moe -o .work/preflight/bin > .work/preflight/build.log 2>&1 || fail "engine moe" .work/preflight/build.log
step baro-serve
cargo test --release --manifest-path serve/Cargo.toml > .work/preflight/cargo.log 2>&1 || fail cargo .work/preflight/cargo.log
cargo build --release --manifest-path serve/Cargo.toml >> .work/preflight/cargo.log 2>&1 || fail cargo .work/preflight/cargo.log
rm -f .work/preflight/bin
st=$(stamp) || fail stamp "file hashing failed"
echo "$st" > .work/preflight.ok
echo "PASS preflight ${st:0:12}"
