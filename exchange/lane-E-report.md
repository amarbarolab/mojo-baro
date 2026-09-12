# Lane E report

Worktree `$HOME/Projects/mojo/mojo-baro-lanes/E`, branch `lane-E`. Plan item **E1**
(`bench/ruler/score-arm.sh` and `tok.py`'s default GGUF) only -- built, nothing else touched.

## E1a -- `bench/ruler/tok.py` default GGUF fallback

`tok.load()`'s default source (`tokenizer-meta.json`'s `source_gguf`) names
`Qwythos-9B-Claude-Mythos-5-1M-MTP-Q4_0.gguf`, which was never built -- only the
`-pure` variant exists under `~/Models/qwythos-9b-claude-mythos-5-1m-mtp-bf16/`.
Confirmed before the fix: `test_gen.py` failed outright (`SystemExit`) with
`BARO_GGUF` unset. Fixed to fall back to the `-pure` sibling when the recorded
path is missing, still failing and naming `BARO_GGUF` if neither exists.

Commit: `a867b81` bench(ruler): fall back to the -pure GGUF when the default is missing

## E1b -- `bench/ruler/score-arm.sh` (new)

`score-arm.sh ARM_DIR [SIZES] [--out OUT_DIR] [--prompts DIR]`: symlinks every
`ARM_DIR/<size>/responses/<task>_<size>` into `OUT_DIR/all/` (default
`OUT_DIR=ARM_DIR`) and runs `score.py --json OUT_DIR/table.json` over it.
`--out` exists so a read-only reference arm dir is never written into;
`--prompts` passes through to `score.py` (its own default `bench/ruler/prompts`
is gitignored/generated, so scoring a foreign arm dir needs its source repo's
prompts).

Verified against the read-only reference
(`~/Projects/mojo/mojo-baro-clean/.work/ruler-baseline/bf16`, its own
`bench/ruler/prompts`, output redirected to this worktree's `.work/E1-score/bf16`,
removed after verification -- reference dir left untouched, confirmed no diff):

```
bench/ruler/score-arm.sh ~/Projects/mojo/mojo-baro-clean/.work/ruler-baseline/bf16 \
  --out .work/E1-score/bf16 --prompts ~/Projects/mojo/mojo-baro-clean/bench/ruler/prompts
```

Output table byte-for-byte equal (dict comparison) to that folder's
`table-granite.json`: niah_single/niah_multikey/vt/cwe rows and effective_length
all match (niah_single 32768, niah_multikey 32768, vt 32768, cwe 16384).

Commit: `75f39b0` bench(ruler): add score-arm.sh to link and score one arm's responses

## Gate

Full suite, captured to `.work/E-gate.txt`:
- `./run-tests.sh` (shim build, `test_gemm`, `test_prefix`, `kernel-census --check`)
- `bench/ruler/test_gen.py` with `BARO_GGUF` unset (the item's own regression check)
- `bench/ruler/test_score.py`

```
RUN_TESTS_EXIT=0
TEST_GEN_EXIT=0   (6/6, was: SystemExit before any test ran -- BARO_GGUF fallback missing)
TEST_SCORE_EXIT=0 (6/6, unaffected by this item)
```

Test count: before = 78 passing checks (71 kernel/prefix assertions + 1
`run-tests.sh` top-level `PASS: prefix checkpoints byte-exact` + 0/6 `test_gen`,
crashed before any assertion + 6/6 `test_score`); after = 84 (same 72 mojo
checks + 6/6 `test_gen` + 6/6 `test_score`), floor (all green) met.

Evidence: `$HOME/Projects/mojo/mojo-baro-lanes/E/.work/E-gate.txt`

## Commits (lane-E, no attribution trailers)

- `a867b81` bench(ruler): fall back to the -pure GGUF when the default is missing
- `75f39b0` bench(ruler): add score-arm.sh to link and score one arm's responses

## Out of scope, not touched

Only `bench/ruler/tok.py` and `bench/ruler/score-arm.sh` (new) were touched, per item E1.
