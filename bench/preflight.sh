#!/usr/bin/env bash
# CPU preflight before any GPU job (PROTOCOL-RULES P15).
#   bench/preflight.sh          run everything below, write .work/preflight.ok for this exact tree
#   bench/preflight.sh --check  exit 1 unless .work/preflight.ok matches the current tree (gate scripts call this)
# Checks: tools/ci-checks.sh (docs, py/sh syntax, census, every bench/*.mojo builds), every run-tests.sh test
# binary builds, dense and MoE engines build, baro-serve builds and its unit tests pass.
set -euo pipefail
cd "$(dirname "$0")/.."
stamp() { printf '%s %s\n' "$(git rev-parse HEAD)" "$( (git diff HEAD; git ls-files -o --exclude-standard | xargs -r cat) | sha256sum | cut -c1-16)"; }
if [ "${1:-}" = --check ]; then
  [ "$(cat .work/preflight.ok 2>/dev/null)" = "$(stamp)" ] && exit 0
  echo "FAIL preflight: no passing bench/preflight.sh for this tree (HEAD $(git rev-parse --short HEAD) + working changes); run it first" >&2
  exit 1
fi
rm -f .work/preflight.ok; mkdir -p .work/preflight
step() { echo "== $1"; }
fail() { echo "FAIL preflight $1: see $2" >&2; exit 1; }
step ci-checks;  tools/ci-checks.sh > .work/preflight/ci.log 2>&1 || fail ci-checks .work/preflight/ci.log
step run-tests-builds
S="$PWD/.work/shim-build"
awk '/\\$/{printf "%s ", substr($0, 1, length($0) - 1); next} {print}' run-tests.sh | grep -E 'mojo build ' | sed 's#-o [^ ]*#-o .work/preflight/bin#' | while read -r cmd; do
  eval "$cmd" > .work/preflight/build.log 2>&1 || fail "build: $cmd" .work/preflight/build.log
done
step engines
./.venv/bin/mojo build serve/engine.mojo -I . -I kernels -o .work/preflight/bin > .work/preflight/build.log 2>&1 || fail "engine dense" .work/preflight/build.log
./.venv/bin/mojo build serve/engine.mojo -I . -I kernels -D BARO_MODEL=qwen35moe -o .work/preflight/bin > .work/preflight/build.log 2>&1 || fail "engine moe" .work/preflight/build.log
step baro-serve
cargo test --release --manifest-path serve/Cargo.toml > .work/preflight/cargo.log 2>&1 || fail cargo .work/preflight/cargo.log
cargo build --release --manifest-path serve/Cargo.toml >> .work/preflight/cargo.log 2>&1 || fail cargo .work/preflight/cargo.log
rm -f .work/preflight/bin
stamp > .work/preflight.ok
echo "PASS preflight $(cut -c1-12 .work/preflight.ok)"
