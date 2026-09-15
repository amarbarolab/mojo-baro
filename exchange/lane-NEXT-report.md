# Lane report: `docs/NEXT-PLAN.md` build lane (opus, w82:p4)

Brief: `~/Brain/mojo/mojo-baro/briefs/2026-09-15-opus-build-lane.md`.
Coordinator: fable in `w82:p1`. Assistants: `w82:p2` (sonnet) and `w82:p3`
(sonnet), one bounded item each, briefs in `~/Brain/mojo/mojo-baro/briefs/`.
One section per item, in build order. GPU minutes are reported per gate.

---

## 1. A0.1 served-path timing of the MoE model: LANDED

Protocol, preregistered before the first run:
`bench/moe-served-protocol.md`. Harness: `bench/served-prompts.sh` plus
`bench/served-summary.py`. Receipts: `.work/moe-served/run/`.
**GPU held 2.73 minutes** (163.6 s, one job, inside the 10-minute rule).

| | one-shot | served (`timings.tok_s_gen`) | served wall clock |
|---|---|---|---|
| 20-prompt median tok/s_gen | **93.52** (86.97 to 94.11) | **92.89** (90.56 to 93.07) | **74.74** (49.81 to 83.92) |
| ratio to one-shot | 1.000 | 0.993 | 0.799 |

- **Identity 20/20 PASS.** The served `choices[0].tokens` equal the one-shot
  `GENERATED:` ids on every prompt, so the two arms are the same
  computation and the comparison is a path comparison, not a model one.
- **The stint is comparable to the R3 receipt**: arm A's 93.52 reproduces
  `bench/moe-perf-protocol.md` R3's 93.46 to 0.06%, on a binary built from a
  clean `git archive` of `aa9f263` (sha `68bc4eb42e6793d2`), which is not the
  R3 binary (`8184f7d` vendored latentos and touched `serve/`).
- **Serving costs 5.5 ms per request**, not 20 tok/s. Served medians:
  prefill 159.3 ms, decode 0.678 s, wall 0.843 s. The gap between 92.89 and
  74.74 is prefill on a 13 to 59 token prompt generating 64 tokens, which is
  19% of the request, plus 5.5 ms of HTTP, JSON, line protocol and queueing.
  p17 (59 prompt tokens) is the slowest wall-clock row at 49.81, exactly as
  that arithmetic predicts.
- All four frozen predictions held (identity, arm A within 2% of 93.46,
  served decode within 2% of arm A, wall clock in 65 to 85).

P1 read-back is in the protocol's Result section: engine and `baro-serve`
sha256, `GET /health` (`pack`, `tmax`, `spec_k`, `mrows`, `tokenizer:false`),
the per-run parameter echo, `timings.finish`, power cap 290 W, vddgfx
-100 mV, sclk 141/3270/3318 MHz, junction max 75 C.

Two notes a later lane needs:

- The MoE pack carries no `tokenizer.json`, so `/v1/chat/completions` is not
  available for it and the prompts went in as token ids on
  `/v1/completions`. That is also what makes the arms bit-comparable: a chat
  template would change the prompt. The brief asked for chat completions;
  this is the deviation, and it is the reason for it.
- The first summary pass voided all 20 rows on two parser bugs of mine
  (`tok/s_gen:` is mid-line, and the HTTP layer renames the engine's `tok_s`
  to `tok_s_gen` at `serve/src/main.rs:373`). Fixed against the run's own
  artifacts with no second GPU stint, which is why the summary is a separate
  program from the harness. P10 behaved: 20 VOID rows printed FAIL and
  exited non-zero rather than averaging over survivors.

`docs/BASELINE.md` gains the served row with its falsifier.
Gates at the commit: `tools/ci-checks.sh` green, `./run-tests.sh` green
(both re-run on the committed tree).

---

## 2. A0.3 B3 C-language IPC probe: LANDED (w82:p2), verified by me

Commit `d060a72`. Run record
`~/AMDHQ/runs/latent-os/E12-ipc-c-2026-09-15.md`, source
`bench/latentos-ipc-probe.c`, report from the pane in its own words inside
that run record. GPU: 8.7 s of execution for 3 runs (0.15 GPU-minutes),
153 s of that job's wall time was queueing behind item 1.

**Verdict: C succeeds where Mojo failed. B3 is reopened.**
`hipIpcGetMemHandle` and `hipIpcOpenMemHandle` both return `hipSuccess` on
all 3 runs in separate processes, and the 2 GiB buffer's sha256 matches every
run. The Mojo probe's `hipErrorInvalidValue` (`47e2896`) was not a driver
refusal: the same sequence, same card, same driver, same day, succeeds from
hipcc. The difference is the calling convention for the 64-byte by-value
`hipIpcMemHandle_t` (SysV MEMORY class), which Mojo's `external_call` does
not lower correctly.

Suggested fix, from the pane and consistent with this repo's own practice: a
tiny C shim taking the handle by pointer, the pattern
`kernels/amarbaro.mojo` already uses for `libamarbaro_shim.so`. Trying a
different Mojo-side aggregate spelling first is cheaper but not guaranteed:
the open question is whether Mojo lowers MEMORY-class aggregates at all.

**Separate finding, not gating that verdict**: the D2D copy measured about
22 ms for 2 GiB on all 3 runs (about 93 GB/s), against the 10 ms E12-ipc
predicted. IPC handoff is correct but not yet shown to be fast, and that gate
FAILs on its own terms. Before any B3 speed claim, it needs its own probe
(first-touch faults on the mapped region, or a non-peer-optimised copy path,
are the candidates).

---

## 3. B6 measurement as a product: LANDED (w82:p3), verified by me

Commit `cd25815` (`docs/METHOD.md`, 223 lines), report `c56a81f`
(`exchange/2026-09-15-p3-b6-method-report.md`). No GPU beyond check 2 below
(under 1 minute).

The document states what a `docs/BASELINE.md` number claims, the rules in
short with `bench/PROTOCOL-RULES.md` cited rather than copied, the two
negative results as results, the two instruments with their exact command
lines, and what a third party does and does not get. The `baro run` section
is left as one named stub for item 4 below, as briefed.

**The check the brief demanded produced a real finding**, which is why it was
worth demanding: `tools/gguf-verify.sh` on
`RegesCore-1.0-35B-UD-Q4_K_S-BARO-8184f7d.gguf` **exits 1**. The closure
rebuilds the engine from the file's own sources and loads the 21 GB pack in
4.3 s, then dies on
`Failed to open file '.work/moe-w3/one.tokens': No such file or directory`.
Cause, which I confirmed by reading the script rather than taking the pane's
word: `tools/gguf-closure.sh`'s qwen35moe branch defaults `BARO_PROMPT` to
`.work/moe-w3/one.tokens` and its reference to `.work/moe-closure-ref.txt`,
both local `.work/` paths that the gguf does not carry and the closure does
not create. The prompt file does not exist on this box today.

So `docs/BASELINE.md`'s claim that the bake "rebuilds the harness with no
path outside the file" does not hold for the qwen35moe class as of
`8184f7d`, and the contributor's one-command check in `docs/amd-family.md`
fails on that model. This is item 4's (B1's) business and is being fixed
there: the file must carry its own reference prompt and reference ids.

---

## 4. B1 a model file that runs itself and proves its numbers: LANDED

Commits `e977116` (the tooling), `abfc452` (run closure), `e340ee1` (env and
the pack tool's dependency). Bake:
`~/Models/RegesCore-1.0-35/RegesCore-1.0-35B-UD-Q4_K_S-BARO-e340ee1.gguf`.
Protocol for the design: the coordinator's two-mode ruling, recorded in
`docs/METHOD.md` section 6.

### What was actually broken

`tools/gguf-verify.sh`, the command `docs/amd-family.md` gives a contributor,
**exited 1 on every MoE bake**. The closure's qwen35moe branch defaulted
`BARO_PROMPT` to `.work/moe-w3/one.tokens` and its reference to
`.work/moe-closure-ref.txt`, local scratch paths the file never carried, and
the first is gone since the 2026-09-12 `.work` clear. `docs/BASELINE.md`
claimed the bake "rebuilds the harness with no path outside the file"; for
that model class it did not. Found by w82:p3's B6 check, not by reading.

### The two modes

The plan wants a file that runs with no checkout, and the repo's integrity
rule (`exchange/scorer-integrity-report.md` P-A) wants the stopwatch to come
from outside the artifact under test. Both hold, in separate modes:

- `tools/baro run` builds the engine from the harness the FILE carries
  (`baro.run.src.engine.mojo` plus its serve-module closure), builds the pack
  from the file's own tensors with the `engine-pack.py` the file carries,
  applies `baro.run.env`, and serves. Every number it prints is labelled
  self-reported.
- `tools/baro verify` takes `serve/engine.mojo` from git at the file's own
  `baro.kernel.commit`, refuses on any byte of difference from the embedded
  copy, then rebuilds and gates tokens. Only it may write a receipt.
- `tools/gguf-receipt.py` is the ledger and refuses any record without
  `mode: verified` and the commit its harness was checked against. Sidecar
  JSONL by default, `baro.hw.receipts` inside the file with `--in-file`
  (a 21 GB rewrite, hence opt-in).
- `tools/bake.sh` makes the bake re-runnable and refuses a dirty source tree.

The closure walk is unchanged: `baro.kernel.src.*` still carries no harness.

### Checks, all run, none skipped

1. **verify on the new bake: PASS 64 tokens match, exit 0**, with
   `prompt: from the file (13 ids)` and `reference: from the file (64 ids)` in
   the closure log, so nothing outside the file was consulted for either. The
   harness check printed `embedded copy is byte-identical to serve/engine.mojo
   at <commit> (23b37b6c...)` before any GPU work.
2. **run mode served the model and answered a real HTTP request**: a live
   `baro-serve` started from the file alone (`engine ... built from this
   file's harness`, `pack ... (733 tensors)`, `env from the file:
   BARO_MEGA=0 BARO_SPEC=0`), `GET /health` returned `"status":"ok"`, and
   `POST /v1/completions` with p01-water's token ids returned 32 tokens
   beginning 279 42900 7035 369 220 16, matching the reference, at
   `tok_s_gen` 91.38 self-reported.
3. The receipt ledger: `tools/baro verify --append` then
   `tools/baro receipts`.

Two defects were found by running it, not by reading it, and each is a commit:
the run-mode build died on `unable to locate module 'prefix'` (the harness's
serve-module closure was not in the file, because `gguf-closure.sh` pulls
those from git and run mode cannot), and then the engine exited before the
server's ready line with `BARO_MEGA=1 is not supported by the qwen35moe model
profile` (the engine's own default; the file now carries `baro.run.env`).

### Limits, stated

- A pack has no `tokenizer.json` unless one was built beside it, so a model
  served this way takes token ids on `/v1/completions` and refuses the text
  and chat endpoints. Converting the gguf's own tokenizer is not built.
- `tools/baro verify` holds the GPU through the closure build, which is
  minutes of CPU inside a GPU reservation. `baro run --no-serve` already
  splits the CPU half out; `gguf-verify.sh` should do the same. Not done here.
- The ledger's `--in-file` mode copies the whole container. For the 21 GB MoE
  bake that is a real cost and a disk-space consideration, which is why the
  sidecar is the default.

---

## 5. B4 stage 1 PCIe bandwidth and expert bytes: LANDED (w82:p3), section 1 verified, section 2 CORRECTED BY ME

Commit `9f45e86` (`bench/pcie-bandwidth.mojo` and
`exchange/2026-09-15-p3-b4-stage1-report.md`). GPU: under 1 minute for a
19.5 s sweep, plus 20 s for my own re-run.

### Section 1, the measurement: verified, keep it

Sustained H2D at 2 GiB, pinned: **28.78 GB/s median** on their run, **28.51
GB/s** on my independent re-run of the same binary, with the link read back
from sysfs as 16.0 GT/s x16, current equal to max. Pinned and pageable are
within 0.6% of each other on this box, which is worth knowing and is the
opposite of the usual assumption. Their "both at once" arm shows no
concurrent transfer, and their report says plainly that the API has no stream
selector so the arm cannot settle the question, which is the right way to
report it.

### Section 2, the arithmetic: wrong, and the correction matters

The report treats column 4 of `.work/moe-w1/pack/index.txt` as bytes. It is
`n_elem`, as `tools/engine-pack.py`'s own docstring says. The index proves it
without any outside knowledge: `blk.0.ffn_down_exps.weight` is at offset
26746880 and the next tensor at 177741824, so it occupies 150,994,944 bytes
for 268,435,456 elements, which is 0.5625 bytes per element, exactly q4_k's
144 bytes per 256 elements. The same method gives q8_0 1.0625, q6_k 0.8203,
f32 4.0, and summing every tensor's gap gives 21,005,191,680 bytes, equal to
`pack.bin` to the byte.

So every byte figure in that section is inflated by 1/0.5625 = 1.78x, and its
headline conclusion is backwards:

| | report | correct |
|---|---|---|
| per expert per matrix | 1,048,576 B | 589,824 B |
| top-8 per layer | 25,165,824 B | 14,155,776 B |
| routed per token, 40 layers | 1.007 GB | **0.573 GB** |
| routed + shared + router per token | 1.154 GB | **0.791 GB** |
| 100B-class scaled (114.3 layers) | 2.876 / 3.296 GB | **1.64 / 2.26 GB** |
| ms per token at 28.7 GB/s, 100B-class | 100 to 115 ms | **57 to 79 ms** |
| residency needed, 100B-class at 30 tok/s | 67 to 71% | **41 to 58%** |

**`docs/NEXT-PLAN.md`'s 0.78 GB per token reproduces** (0.7906 GB is routed
plus shared expert plus router, uncached, 1.4% off the plan's figure), and its
"1 to 2 GB per token, 40 to 80 ms" for a 100B-class model is confirmed by the
corrected arithmetic rather than refuted. The plan needs no change.

Fixed by w82:p3 at `b7f7db5`, and their fix is better than the correction I
sent them: 3 of the 40 layers store `ffn_down_exps` as q6_k rather than q4_k,
which my own first pass flattened to q4_k everywhere. 0.573 and 0.791 GB are
the per-layer-dtype figures, confirmed against the index by both of us
independently.

Their stage 2 question stands and is the right one: does an expert-weight H2D
transfer overlap with compute on already-resident layers through this API, and
does the overlap save wall clock. Every number above is transfer-only and
charges nothing to compute.
