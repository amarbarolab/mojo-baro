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

## 4 onwards

`baro run` and the receipt ledger (B1) are in progress; the coordinator
approved a two-mode design (run mode with an embedded harness under
`baro.run.src.engine.mojo` whose numbers are labelled self-reported, verify
mode which takes the harness from git at `baro.kernel.commit`, refuses on any
difference, and is the only mode that may append to `baro.hw.receipts`).
Untouched so far: A5, A1, B2, B5, A2, A3, B4.
