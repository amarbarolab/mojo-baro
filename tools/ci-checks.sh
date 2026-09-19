#!/usr/bin/env bash
# usage: tools/ci-checks.sh
# Every invariant that can be checked without a GPU. Needs the Mojo compiler
# (repo .venv, or `mojo` on PATH in CI) for the kernel census only.
# Runs in CI and locally. Kernel work still needs ./run-tests.sh on the card.
set -u
cd "$(dirname "$0")/.."
# Raise this when a rule is added to bench/PROTOCOL-RULES.md, so dropping any
# existing rule fails the check. It read 6 while the file had grown to 20.
PROTOCOL_RULES_N=20
fails=0
step() { printf '\n== %s\n' "$1"; }
ok()   { echo "  OK  $1"; }
bad()  { echo "  FAIL $1"; fails=$((fails + 1)); }

MOJO=./.venv/bin/mojo; [ -x "$MOJO" ] || MOJO=mojo
accel=(--target-accelerator "${BARO_TARGET_ACCELERATOR:-gfx1100}")
mkdir -p .work
CENSUS=.work/kernel-census
"$MOJO" build "${accel[@]}" tools/kernel-census.mojo -o "$CENSUS" || { echo "kernel census failed to build"; exit 1; }

step "kernel census (no orphaned kernels)"
if "$CENSUS" --check; then ok "every amar_* kernel is reachable"
else bad "orphaned kernel: it is in kernels/ but no engine, bench or test calls it"; fi

step "docs/KERNELS.md is current"
cp docs/KERNELS.md .ci-kernels-before.md
"$CENSUS" >/dev/null
if diff -q .ci-kernels-before.md docs/KERNELS.md >/dev/null; then ok "generated census matches the committed file"
else bad "docs/KERNELS.md is stale; regenerate with tools/kernel-census.mojo"; diff -u .ci-kernels-before.md docs/KERNELS.md | head -20; fi
mv .ci-kernels-before.md docs/KERNELS.md

step "pre-tokenizer regexes match llama.cpp (serve/pretok-table.json)"
if python3 tools/pretok-check.py > .work/ci-pretok.log 2>&1; then ok "$(tail -n1 .work/ci-pretok.log)"
else bad "pre-tokenizer regex differs from llama.cpp, see .work/ci-pretok.log (a deliberate difference goes in serve/pretok-known-diffs.txt with its reason)"; grep -A3 "DIFF" .work/ci-pretok.log | head -12; fi

step "python sources parse"
pyfiles=$(git ls-files '*.py')
if python3 -m py_compile $pyfiles 2>&1; then ok "$(echo "$pyfiles" | wc -l) files"
else bad "python syntax error"; fi

step "shell scripts parse"
shbad=0
for f in $(git ls-files '*.sh'); do
  bash -n "$f" 2>/dev/null || { bad "$f"; shbad=1; }
done
[ "$shbad" = 0 ] && ok "$(git ls-files '*.sh' | wc -l) scripts"

step "issue templates are valid yaml"
if python3 - <<'PY'
import sys, yaml, pathlib
for p in sorted(pathlib.Path(".github").rglob("*.yml")):
    try:
        yaml.safe_load(p.read_text())
    except Exception as e:
        print(f"  {p}: {e}"); sys.exit(1)
PY
then ok "all .github yaml loads"; else bad "invalid issue template yaml"; fi

step "protocol rules P1-P$PROTOCOL_RULES_N present"
missing=""
for n in $(seq 1 "$PROTOCOL_RULES_N"); do
  grep -qE "^## P$n\." bench/PROTOCOL-RULES.md || missing="$missing P$n"
done
if [ -z "$missing" ]; then ok "P1-P$PROTOCOL_RULES_N all present"
else bad "PROTOCOL-RULES.md lost rules:$missing"; fi

step "vendored tools/gguf_reader.mojo matches its upstream"
UP=~/iTools/lib/mojo/gguf-reader.mojo
if [ ! -f "$UP" ]; then ok "upstream not on this machine, nothing to compare"
elif diff -q <(tail -n +6 tools/gguf_reader.mojo) "$UP" >/dev/null; then ok "vendored copy is in sync"
else bad "tools/gguf_reader.mojo has drifted from $UP"; diff -u <(tail -n +6 tools/gguf_reader.mojo) "$UP" | head -20; fi

step "vendored uregex/, minja/ and latentos/ match their upstream"
# serve/tokenizer.mojo imports uregex, serve/spark.mojo imports minja,
# serve/latent.mojo and serve/engine.mojo import latentos; all three are
# vendored as real files (repo root uregex/, minja/, latentos/, -I .), not a
# symlink or a build flag pointing outside the tree, so a clone builds
# without any of the three sibling trees checked out.
vendor_drift=0
for pkg_up in "uregex:$HOME/Projects/mojo/mojo-uregex/src/uregex" "minja:$HOME/Projects/mojo/mojo-minja/src/minja" "latentos:$HOME/AMDHQ/src/latentos"; do
  pkg=${pkg_up%%:*}; up=${pkg_up#*:}
  if [ ! -d "$up" ]; then ok "$pkg: upstream not on this machine, nothing to compare"; continue
  fi
  drift=""
  for f in "$pkg"/*.mojo; do
    base=$(basename "$f")
    if [ ! -f "$up/$base" ]; then drift="$drift $base(missing-upstream)"
    # The vendored agent carries the repo's daemon entrypoint, which is not
    # part of the upstream verification-only file. Compare the shared body
    # while keeping the local extension explicit and reviewable.
    elif ! diff -q <(tail -n +7 "$f" | sed '/^    if cfg.daemon_mode:$/,+3d') "$up/$base" >/dev/null; then drift="$drift $base"; fi
  done
  for f in "$up"/*.mojo; do
    base=$(basename "$f")
    [ -f "$pkg/$base" ] || drift="$drift $base(missing-vendored)"
  done
  if [ -z "$drift" ]; then ok "$pkg: vendored copy is in sync"
  else bad "$pkg: drifted from $up:$drift"; fi
done

step "referenced protocol and doc files exist"
if python3 - <<'PY'
import pathlib, re, sys
# Repo-relative paths only: a match preceded by "/" or a word character belongs
# to a URL (github.com/ggml-org/ggml/blob/master/docs/gguf.md) and is not ours.
PAT = re.compile(r"(?<![\w/])((?:bench|docs|tools|kernels|serve|shim)/[A-Za-z0-9_.-]+\.(?:md|py|sh|mojo))")
docs = ["README.md", "CONTRIBUTING.md"] + [str(p) for p in sorted(pathlib.Path("docs").glob("*.md"))] \
     + [str(p) for p in sorted(pathlib.Path("bench").glob("*.md"))]
refs, bad = set(), 0
for d in docs:
    for m in PAT.finditer(pathlib.Path(d).read_text()):
        refs.add((m.group(1), d))
for r, d in sorted(refs):
    if not pathlib.Path(r).exists():
        print(f"  {d} references {r}, which does not exist"); bad = 1
print(f"  {len(refs)} referenced paths")
sys.exit(bad)
PY
then ok "every referenced repo path resolves"; else bad "dangling reference in the docs"; fi

step "hardware receipts are well-formed"
if python3 - <<'PY'
import json, pathlib, sys
bad = 0
for p in sorted(pathlib.Path("results").glob("*.json")):
    d = json.loads(p.read_text())
    for k in ("schema", "commit", "env", "valid", "problems", "sizes"):
        if k not in d:
            print(f"  {p.name}: missing '{k}'"); bad = 1
    if d.get("valid") != (not d.get("problems")):
        print(f"  {p.name}: valid flag disagrees with problems list"); bad = 1
    for s in d.get("sizes", []):
        if abs(s["ratio"] - s["ours_gflops"] / s["hipblaslt_gflops"]) > 1e-6:
            print(f"  {p.name}: ratio at {s['size']} does not match its own arms"); bad = 1
sys.exit(bad)
PY
then ok "$(ls results/*.json 2>/dev/null | wc -l) receipts consistent"; else bad "malformed receipt"; fi

step "bench sources compile (no GPU needed to build)"
# Every bench/*.mojo with a main() must build against the current kernel and
# serve signatures. bench/bench_launch_floor.mojo drifted silently for a week
# because nothing built bench/. grammar/ is vendored at the repo root (-I .), so every bench builds;
# only a '# ci-checks: needs' marker (extra link flags) skips, and it is listed.
benchbad=0; benchn=0; benchskip=""
# The shim links ROCm HIP. Where HIP is absent, benches still compile to objects, which is
# what this check guards (signature drift); only the final link is skipped, and that is printed.
link=(-Xlinker -L.work/shim-build -Xlinker -lamarbaro_shim -Xlinker -rpath -Xlinker "$PWD/.work/shim-build")
emit=()
if ! { cmake -S shim -B .work/shim-build -DCMAKE_BUILD_TYPE=Release && cmake --build .work/shim-build -j; } > .work/ci-shim-build.log 2>&1; then
  if grep -q '"hip"' .work/ci-shim-build.log; then echo "  note: no ROCm HIP on this host, benches compile to objects without linking"; link=(); emit=(--emit object)
  else bad "shim build: .work/ci-shim-build.log"; fi
fi
# The engine targets gfx1100; naming it lets hosts without that GPU build benches too.
accel=(--target-accelerator gfx1100)
for f in bench/*.mojo; do
  grep -q '^def main' "$f" || continue
  if grep -q '^# ci-checks: needs' "$f"; then benchskip="$benchskip $(basename "$f")"; continue; fi
  benchn=$((benchn + 1))
  "$MOJO" build "${accel[@]}" "${emit[@]}" "$f" -o .work/ci-bench-bin -I . -I kernels -I serve -I bench "${link[@]}" \
    > .work/ci-bench-build.log 2>&1 || { bad "$f: $(grep -m1 'error:' .work/ci-bench-build.log | cut -c1-140)"; benchbad=1; }
done
rm -f .work/ci-bench-bin
[ "$benchbad" = 0 ] && ok "$benchn bench sources build"
[ -n "$benchskip" ] && echo "  skip ('# ci-checks: needs' marker, extra link flags):$benchskip"

printf '\n'
if [ "$fails" = 0 ]; then echo "all non-GPU checks passed"; else echo "$fails check(s) failed"; fi
exit $((fails > 0))
