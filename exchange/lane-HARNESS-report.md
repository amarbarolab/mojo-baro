# Lane HARNESS -- report

Item HARNESS (`exchange/e8-lane-plan-2026-09-09.md`), branch `lane-HARNESS`,
worktree `$HOME/Projects/mojo-baro-lanes/HARNESS`.

## Blocker found and resolved before any HARNESS code

`serve/harness.mojo` (`load_pack`/`alloc_bufs`/`Pack`, extracted verbatim out
of `serve/engine.mojo`'s `main()` in commit `5a823375`, "refactor(engine):
move pack loading and buffer allocation to serve/harness.mojo") is **not on
this branch's history** -- `git merge-base --is-ancestor 5a823375 HEAD` is
`NO`. It exists on `lane-chat` (diffed clean against this branch's
`window.mojo`/`registry.mojo`). `bench/bench_hidden_dtype.mojo`, committed on
this branch at `bde72b2` (E9), imports `from harness import load_pack,
alloc_bufs, Pack` and does not build here (`unable to locate module
'harness'`) -- confirmed before writing anything.

HARNESS's own item spec assumes the same file exists (interface contract:
"find the table's buffer/offset/dtype ... in `serve/harness.mojo:load_pack`"),
and non-goals forbid touching `serve/` except the realign stub. Fix:
`bench/latent_harness.mojo`, a bench/-local copy of the same known-working
code (not a `serve/` change) -- a one-line import swap once the real
`serve/harness.mojo` lands. Documented in the file header and the commit.

## What was built (exactly the item's file list)

- `bench/bench_latent_handoff.mojo` -- the 5-arm evaluator (0, T, L8-raw,
  L8-soft, L32-soft), two co-resident `WindowBufs` engines.
- `bench/latent-handoff.sh` -- build, GPU-waiting-room run, VRAM read-back,
  scoring.
- `bench/e8_score.py` -- Python oracle: last-integer match (math),
  JSON-exact-match (json); schema-validity itself is computed in Mojo via
  `grammar/`'s `Automaton`/`Matcher` against the decoded answer text
  (retokenized through the grammar module's own vocab/trie, same approach as
  `grammar/test_accept_known_good.mojo`) -- no decode-time constraint, per the
  item's file list ("schema validity is checked in Mojo").
- `results/e8/` -- committed: the mandated smoke run's three output files.
- `bench/latent_harness.mojo` (not in the item's file list, added to resolve
  the blocker above) and `serve/realign.mojo` (the stub the interface
  contract calls for).

Soft arms call `realign_expected_embedding` and propagate its raise
("REALIGN not merged") as a per-arm `error` field -- never a fallback to the
raw vector, per the item's instruction.

## Gate

Full suite (`./run-tests.sh`, unchanged by this item): `.work/HARNESS-gate.txt`
(gitignored, `.work/` -- quoting the tail):

```
GEMM OK — 4 x 3 @ 3 x 2 matches host reference
77 kernels, 35 in registry, 3 orphans
ORPHAN: amar_attn_decode_swa spark_kernels.mojo
ORPHAN: amar_head_gate_mul_cast spark_kernels.mojo
ORPHAN: amar_skinny_reduce_gelu_par_bf16 spark_kernels.mojo
EXIT: 1
```

**Pre-existing, not caused by this item**: reproduced with every HARNESS file
moved out of the tree (`python3 tools/kernel-census.py --check` -- same 3
orphans, same exit 1, with none of `bench/bench_latent_handoff.mojo`,
`bench/latent_harness.mojo`, `bench/e8_score.py`, `bench/latent-handoff.sh`,
`serve/realign.mojo` present). The 3 orphaned kernels live in
`kernels/spark_kernels.mojo`; `tools/kernel-census.py`'s "users" scan covers
`serve/registry.mojo` + `bench/*.mojo` + `kernels/test_*.mojo`, never
`serve/spark.mojo`, so they were orphaned before this branch existed.

Test count: `run-tests.sh` itself is unchanged (0 kernels/tests added under
`kernels/`) -- this item is bench/-only infrastructure, no floor to compare
against there. Item's own test, per the plan's literal command:

```
bench/latent-handoff.sh --items 4 --arms 0,T,L8-raw
```

run under `gpu-wait run --priority 30 --vram 14 --`, exit 0. Both output
files present (`results/e8/topology1-q4-2026-09-09.{json,md}`), JSON parses
(`json.load` in `bench/e8_score.py` and independently at the terminal).
Evidence: `.work/HARNESS-smoke.log` (gitignored).

Beyond the mandated smoke, also ran all 5 arms (0/T/L8-raw/L8-soft/L32-soft)
over 21 items (all 20 json tasks + 1 math task, `--items 21`) as an extra
check before calling this done (§18: a real test, not just a green smoke) --
no crashes, math_01 scored correct under arm 0 (last-integer match), every
soft-arm cell recorded `"error":"REALIGN not merged"` with an empty
`answer_text` (never a fallback). Not committed -- only the mandated smoke's
three files are.

## Topology 2

`.work/e8-harness-topology2-blocker.md` (gitignored, quoting the finding):
`serve/spark.mojo` is a self-contained 342-line engine with no `WindowBufs`,
`WindowCfg`, `WindowState`, `step_window`, or `hn_d` (grepped, only `def main`
matches), its own `load_pack` signature, and its own dims (`H=2560`,
`VOCAB=131072`, `N_LAYERS=36`). None of this item's machinery transfers
without a dedicated Spark-side harness + a Spark-typed realign, which is new
engine work the non-goals exclude ("do not build a Spark engine"). Topology 1
built first, as required; Topology 2 not attempted.

## Finding: co-residency VRAM, plan estimate off by ~2x

Plan estimate: "~13.3 GB for two." Measured (`bench/latent-handoff.sh`'s
mid-run VRAM read-back, taken while both packs are resident -- see the code
comment on why a post-exit reading is useless): **1.63 GB before, 25.01 GB
after both loads** (delta 23.38 GB), on a 24 GB card -- well under 1 GB of
headroom. Already applied one in-scope mitigation: `bench_latent_handoff.mojo`
defaults `tmax` (the KV-cache-sizing parameter passed to `alloc_bufs`, not a
`serve/` change) to 640 instead of `registry.TMAX`=1088 (`e8_tasks.json`'s
longest prompt is 98 tokens; longest arm needs prompt + 300 + 128) -- saved
~500 MB, not the multi-GB gap the plan's estimate implies. The remaining gap
is in `alloc_bufs`'s own scratch buffers (`SPLITK*SM*{QF,H,KV,NH_V,FFN,VOCAB}`
per engine), which this item's non-goals put out of reach (copied verbatim
from known-working code, not restructured). Flagging for the coordinator:
running anything else on this GPU during the full 40-item run risks OOM.

## Commits (lane-HARNESS, no attribution trailers)

- `93b54c8` realign: stub interface contract for HARNESS to build against
- `7f980b5` bench: E8 HARNESS -- 5-arm dual-engine latent-handoff evaluator

## Design calls made where the spec didn't pin a value (flagged, not hidden)

- `ans_max` (B's answer-generation budget): 128 tokens, `BARO_E8_ANS_MAX`-
  overridable. Not specified in the plan.
- Arm T's B-continuation: B's context is `tokens(full_prompt) + trim_at_stop(A's
  CoT ids)` with generation continuing directly (no fresh assistant-turn
  wrapper, since `full_prompt` already ends inside one) -- literal reading of
  "B receives prompt + that text ... answers." Observed effect: on several
  items A's own CoT already constitutes a complete answer, so B's very next
  token is immediately an end-of-turn token and `answer_text` comes back
  empty -- real model behavior given this construction, not a bug (verified
  by decoding arm 0's `answer_text` for the same item/context: non-empty,
  correctly formed).
- Stop-token set: `tok.eos_id` + `<|im_end|>` if present in vocab; generation
  ids are trimmed at (excluding) the first stop token before decode/scoring.
- Tokenizer GGUF for decode: `~/Models/qwythos-9b-claude-mythos-5-1m-mtp-bf16/
  Qwythos-9B-Claude-Mythos-5-1M-MTP-Q4_0-pure.gguf` (tokenizer metadata only,
  quantization doesn't matter for it), `BARO_E8_GGUF`-overridable.

## Round 2

Coordinator review of `7f980b5` found five defects. Fixed all five in
`bench/bench_latent_handoff.mojo`, `bench/e8_score.py`,
`bench/latent-handoff.sh`; smoke re-run and passes.

### Defect 1 -- scoring input is the raw text

Mojo now strips before either scoring path touches the text: new
`strip_think_and_fences` (drops every `<think>...</think>` span -- an
unterminated one drops to end of string -- and every literal ` ``` ` fence
marker, byte-scanning) and `extract_json_object` (string-literal-aware
balanced-brace scan for the first complete `{...}`). `strip_for_math` =
`strip_think_and_fences`; `strip_for_json` = `extract_json_object` composed
with it. The result is written per arm as a new `scored_text` field
alongside `answer_text` -- schema check (Mojo) and `e8_score.py`'s parse
both consume `scored_text`, never `answer_text`, so the two languages can't
derive different "the answer" strings.

### Defect 2 -- correct_exact / correct_subset, both columns

`e8_score.py`'s `score_arm` now returns `(correct_exact, correct_subset,
reason)`. `correct_exact` = parsed `scored_text` == `expected`.
`correct_subset` = every key in `expected` present in parsed with an equal
value, extra keys in parsed ignored (for math, no subset concept for a
scalar, so `correct_subset == correct_exact`). Neither depends on
`schema_valid` (see finding below) -- both are written per arm in the raw
JSON, and the md table shows both columns per json cell as `exact/subset`.
The coordinator picks which the gate uses.

### Defect 3 -- expected/type in the raw json

Per-item header in the raw output now carries `"expected"` (the task's
expected value, serialized verbatim via a new `json_value_to_string`
JSONValue-to-text walker) alongside the `"type"` that was already there --
the raw file is self-describing without cross-referencing
`bench/data/e8_tasks.json`.

### Defect 4 -- turn boundary after handoff

Root cause: B never got a fresh assistant turn after the handoff. Fresh-restart
arms (`0`, `T`) simply concatenate `<|im_end|>\n<|im_start|>assistant\n`'s
ids (`turn_ids`) onto B's context before the existing full-reprocess call --
arm 0 already ends there via `full_prompt`, so only arm T needed this (after
its trimmed CoT ids). Latent arms (`L8-raw`/`L8-soft`/`L32-soft`) can't
re-derive a token list -- they continue an existing KV-cache/conv-state built
by the raw/soft-vector injection -- so a new helper `append_known_tokens`
writes `turn_ids`' ids directly into the running window's token buffer at the
current position and drives `step_window` forward through them (same
mechanism `reset_and_load`/`run_to_prompt_end` already uses for the initial
prompt, generalized to a non-zero base position); the last known token's step
naturally produces B's first real prediction, so generation continues from
there with no discontinuity. `start_pos` (previously hardcoded
`len(tokens) + k`) is now read back as `wst.pos` after the injection, so it
reflects the appended ids' length. This is stated in the raw output header:
`"turn_boundary_note"`, a fixed string describing the mechanism, so a reader
of the raw JSON alone knows all five arms start B at the same assistant-turn
boundary.

### Defect 5 -- B's own thinking eats the budget

New `BARO_E8_NOTHINK` env var, default `1`: when set, `<think>\n\n</think>\n\n`'s
ids (`nothink_ids`) are appended right after the assistant-turn ids for B in
every arm, so B answers without its own chain-of-thought -- the only
reasoning channel under test is the handoff. `ans_max` default raised
128 -> 256 (`BARO_E8_ANS_MAX`-overridable, unchanged mechanism).

### Item-selection flag

`--items N` can't express "json_01..04 + math_01..04" (file order is
all-json-then-all-math). Added `--ids id1,id2,...` to
`bench_latent_handoff.mojo` (looks each id up by linear scan over the tasks
array, preserving the requested order; unknown ids print a warning and are
skipped) and a passthrough `--ids` flag on `bench/latent-handoff.sh`
(overrides `--items` when both given).

### Smoke (the round-2 pass check)

```
bench/latent-handoff.sh --ids json_01,json_02,json_03,json_04,math_01,math_02,math_03,math_04 --arms 0,T,L8-raw
```

run under `gpu-wait run --priority 30 --vram 14 --`, exit 0. Output:
`results/e8/topology1-q4-2026-09-09-round2.{json,md,raw.json}` (committed;
suffixed `-round2` so round 1's already-committed same-date files aren't
overwritten -- round 1's report/commit still reference those verbatim).

| arm | json | math | overall (exact/subset) |
|---|---|---|---|
| 0 | 1/4 / 2/4 | 4/4 | 5/8 / 6/8 |
| T | 2/4 / 3/4 | 4/4 | 6/8 / 7/8 |
| L8-raw | 2/4 / 3/4 | 4/4 | 6/8 / 7/8 |

Pass criteria (brief's exact words): arm 0 has >= 1 `correct_subset` on json
(2/4, met) AND >= 1 correct on math (4/4, met); arm T's `answer_text`
non-empty on >= 6/8 (8/8, met). **Smoke passes.**

Arm T non-empty check (not in the md table, computed directly from the raw
json): `answer_text.strip()` non-empty for all 8/8 items under arm T.

Stripped-answer examples (arm 0, `scored_text` next to `expected` --
requested even though the smoke passed, since it's cheap and load-bearing
evidence):

```
json_01  scored_text: {"name": "Alice", "age": 29, "occupation": "engineer"}
         expected:    {"name": "Alice", "age": 29}
         exact=False subset=True   (occupation is an extra key -- exactly
         the exact-vs-subset distinction defect 2 exists for)

json_02  scored_text: {"name": "Bob", "age": 42}
         expected:    {"name": "Bob", "age": 42}
         exact=True  subset=True

json_03  scored_text: {"search_query": "rocm benchmarks", "page": 3}
         expected:    {"query": "rocm benchmarks", "page": 3}
         exact=False subset=False  (model used a different key name --
         real miss, not a scoring artifact)

json_04  scored_text: {"task_state": "pending_review", "task_name":
                        "background_compilation_job", "status": "pending_review"}
         expected:    {"status": "pending"}
         exact=False subset=False  (status value itself is wrong --
         real miss)

math_01  scored_text: "...16 - 3 - 4 = 9. ... 9 * 2 = 18 dollars...
                        Answer: 18"
         expected: 18   exact=True subset=True
```

Math went 4/4 exact on arm 0 across all four items (not just math_01 above).

### Finding: schema_valid stays false even on content-correct json (not fixed, out of scope)

All four json items score `schema_valid: false` under arm 0 even where
`correct_subset` (and, for json_02, `correct_exact`) is true. Cause: the
model outputs pretty-printed JSON (`{\n  "name": "Bob",\n  "age": 42\n}`),
but `grammar/`'s `Automaton.add_literal` compiles each structural literal
(`{`, `"key":`, `,`, `}`) with zero tolerance for whitespace/indentation --
`grammar/test_accept_known_good.mojo`'s own known-good samples are always
compact. This is a `grammar/` module property, not a HARNESS bug (HARNESS
doesn't touch `grammar/`, non-goals exclude it) -- flagging since it means
`schema_valid` will read false almost always regardless of content
correctness, which is why round 2 explicitly defined `correct_exact`/
`correct_subset` independent of it (defect 2).

### Commits (lane-HARNESS, no attribution trailers)

- `93b54c8` realign: stub interface contract for HARNESS to build against
- `7f980b5` bench: E8 HARNESS -- 5-arm dual-engine latent-handoff evaluator
- `003b01b` bench: E8 HARNESS round 2 -- strip-before-score, exact/subset, turn boundary, nothink

## Round 3

Part A (this section). Part B (REALIGN merge + stale-vector fix + 5-arm
smoke) is on GO from the coordinator, appended here after it runs.

### Part A -- schema_valid whitespace intolerance

Root cause named in the brief, confirmed: `grammar/`'s `Automaton.add_literal`
compiles each structural literal (`{`, `"key":`, `,`, `}`) with zero
whitespace tolerance, so a schema-correct but pretty-printed answer (the
model's default style) always failed the `Matcher`, independent of content
correctness -- this is the finding round 2 flagged as out-of-scope.

Fix, in `bench/bench_latent_handoff.mojo` only: new `compact_json(sample)` --
parses `sample` via `grammar.json_value.parse_json_bytes`, re-serializes with
the existing `json_value_to_string` walker (no spaces, the form
`grammar/test_accept_known_good.mojo`'s fixtures use), falls back to
`strip_ws(sample)` if the parse raises (so a genuinely malformed sample still
fails the schema check downstream instead of erroring here). `check_schema_valid`
now tokenizes `compact_json(sample)` instead of `strip_ws(sample)` before
handing it to the `Matcher`. `scored_text` in the raw output is untouched --
only the schema check's internal input is re-serialized.

Before/after, same 4 json items, arm 0 (round 2's committed smoke vs. this
fix, `schema_valid`):

| item | before (round 2) | after (round 3 Part A) | exact | subset |
|---|---|---|---|---|
| json_01 | false | false | false | true |
| json_02 | false | **true** | true | true |
| json_03 | false | false | false | false |
| json_04 | false | false | false | false |

json_02 flips to `true` exactly as the brief predicted (its `scored_text` is
`{"name": "Bob", "age": 42}`, compact already, but was still failing before
this fix because the Matcher's own literal-byte scan requires the *exact*
key-ordering/format grammar/'s schema compiler emits, not merely "no
whitespace" -- re-serializing through `json_value_to_string` normalizes to
that form). json_01 stays false correctly: `correct_subset=true` but it has
an extra `occupation` key the schema doesn't allow, a real schema violation,
not a whitespace artifact. json_03/json_04 stay false correctly: their
content itself is wrong (wrong key names / wrong value), independent of
formatting -- confirmed by `correct_exact`/`correct_subset` both false for
those two under arm 0.

Full before table (all three arms, `schema_valid` all false, round 2 commit
`003b01b`'s smoke) is in the "Finding" section above. After-fix run (all
three arms, same 8 ids) is
`results/e8/topology1-q4-2026-09-09-round3-partA.{json,md,raw.json}`
(committed) -- `schema_valid` for T and L8-raw on json_02 also flips to
`true`, same reasoning.

### Commit (this section, no attribution trailers)

- `8d5ee38` bench: E8 HARNESS round 3 Part A -- compact-reserialize before schema check

### Part B -- REALIGN merge, L8-raw stale-vector fix, 5-arm smoke (GO received)

**Merge**: `git rm serve/realign.mojo` (drop the stub) committed first, then
`git merge lane-REALIGN` (`55dc19e`) -- clean, no conflicts (REALIGN only
touched `kernels/realign_kernels.mojo`, `kernels/test_realign.mojo`,
`serve/realign.mojo`, `tools/realign_oracle.py`, none of which overlap
HARNESS's files). Adopted the final signature verbatim:
`realign_expected_embedding(ctx, mut b, mut e_dev, pack_q4)` -- threaded
`pack_q4` through `collect_latent_soft`'s signature and its one call site.

**L8-raw stale-vector fix** (REALIGN's own round-3 finding,
`kernels/mega.mojo:1182` gates `b.hn_d`'s write on `fold_head==2`; every
launch here passes `fold_head=1`, so it was never written under `mega=True`
-- reading it directly shipped whatever an unrelated earlier prefill chunk
last left there, not the current position's hidden state). Fix in
`collect_latent_raw` (`bench/bench_latent_handoff.mojo`): replaced the direct
`b.hn_d` read with `final_norm_hidden(ctx, b, latent_dev)` from
`serve/realign.mojo` (re-derives the post-final-norm `[H]` f32 hidden from
`b.x_d`, the one buffer every mega/non-mega path keeps current). The
receiver side (`step_latent_raw`, `apply_latent_to_receiver`) was already
correct (injects into `x_d`) and is unchanged.

**Smoke** (5 arms, same 8 ids as rounds 2/3A):

```
bench/latent-handoff.sh --ids json_01,json_02,json_03,json_04,math_01,math_02,math_03,math_04 --arms 0,T,L8-raw,L8-soft,L32-soft
```

run under `gpu-wait run --priority 30 --vram 14 --`, exit 0. Output:
`results/e8/topology1-q4-2026-09-09-round3-partB.{json,md,raw.json}`
(committed).

| arm | json | math | overall (exact/subset) |
|---|---|---|---|
| 0 | 1/4 / 2/4 | 4/4 | 5/8 / 6/8 |
| T | 2/4 / 3/4 | 4/4 | 6/8 / 7/8 |
| L8-raw | 2/4 / 3/4 | 4/4 | 6/8 / 7/8 |
| L8-soft | 1/4 / 2/4 | 4/4 | 5/8 / 6/8 |
| L32-soft | 1/4 / 2/4 | 4/4 | 5/8 / 6/8 |

| arm | median producer_s | n |
|---|---|---|
| 0 | - | 0 |
| T | 2.467 | 8 |
| L8-raw | 0.155 | 8 |
| L8-soft | 0.202 | 8 |
| L32-soft | 0.535 | 8 |

Pass criteria (brief's exact words): no `error` cells (0/40 arm-cells errored,
checked programmatically against the raw json, not just eyeballed the
table); soft arms produce non-empty `answer_text` on >= 6/8 (L8-soft 8/8,
L32-soft 8/8). **Smoke passes.** Per the brief: did not run the 40-item set,
did not compute a verdict on accuracy -- L8-soft/L32-soft scoring 1/4 exact
json (vs. L8-raw's 2/4) with a real, non-fallback `realign_expected_embedding`
call on every one of the 40 arm-cells (zero errors) is evidence the soft
path is live, not a claim about which arm is better.

### Commit (this section, no attribution trailers)

- round 3 Part B commit -- see branch log
