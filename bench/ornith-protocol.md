# Ornith-1.5-9B (K-quant source) in the engine

Preregistered 2026-09-11, before any Ornith timed run, tree at the lane-ORNITH commit that
lands `tools/engine-pack.py`'s K-quant path. Binds to `bench/PROTOCOL-RULES.md` P1-P6.
Question: does a K-quant-sourced (Q4_K_M) engine pack decode Ornith-1.5-9B correctly, and at
what speed against llama.cpp running the same GGUF natively?

## Why this model needs its own protocol

`~/Models/ornith-1.5-9b-q4_K_M/Ornith-1.5-9B-Q4_K_M.gguf` is `qwen35`, the same shape as
Qwythos-9B (H=4096, FFN=12288, NQH=16, NKVH=4, HD=256, SSM inner 4096, state 128, 16 groups,
conv 4, full attention every 4th layer, VOCAB 248320, tokenizer sha-identical to Qwythos), but
it exists ONLY as K-quants (223 Q4_K, 35 Q6_K, 184 F32 — no bf16/f32 source). `tools/engine-pack.py`
dequantises each K-quant tensor to f32 with gguf-py (`gguf.quants.dequantize`), rounds to bf16,
then quantises through the existing `--q8` path unchanged. That dequantised f32 IS the exact
ground truth for the source weights (gguf-py implements the same block math as ggml/llama.cpp),
so it doubles as the reference forward pass: `tools/model-ref.py`'s numpy stack is shape-driven,
not source-format-driven, and needs no change to run over the Ornith pack.

**Amendment (mid-lane, coordinator-approved 2026-09-11):** step 3a's teacher-forced agreement
needs `BARO_FORCE` in `serve/engine.mojo`, which did not exist (only `serve/spark.mojo` had it).
`serve/engine.mojo`'s decode path selects every token on-device inside `mega_token_k` /
`mega_win_k` / `argmax_k` — there is no per-step host round trip to intercept like spark's
per-layer host loop. Added at the host level instead, with no kernel change: the outer decode
loop in `serve/engine.mojo` (`while wst.pos < n_total - 1: step_window(...)`) already runs one
`step_window` call per generated token when `BARO_SPEC=0`, so after each call — only when
`BARO_FORCE` is set and the position is past the prompt — the token `step_window` just wrote
into `toks_d` is copied back to host (recorded as "predicted"), then a host->device copy of the
reference id overwrites it before the next step reads it as context. `BARO_FORCE` + `BARO_SPEC=1`
raises (teacher forcing is defined for the no-spec path only, matching `spark.mojo`). With
`BARO_FORCE` unset this adds zero device work: the whole block is behind `if len(force) > 0`.
Own commit, named in the report.

## Arms

- **ours**: `.work/engine` built from `serve/engine.mojo` (this lane's `BARO_FORCE` addition),
  `BARO_PACK=.work/engine-pack-ornith-q8` (`tools/engine-pack.py MODEL.gguf OUTDIR --q8`, K-quant
  path), greedy (`BARO_SPEC=0`) unless noted.
- **ours/f32-ref**: `tools/model-ref.py` (`BARO_PACK` pointed at a plain, unquantised copy of the
  Ornith pack — `engine-pack.py` with no `--q8`/`--q4` flag — so the numpy stack reads gguf-py's
  own dequantised bf16 values, one rounding step closer to the K-quant source than the q8 pack).
  Not a timed arm; it produces the reference id stream for teacher forcing.
- **llama.cpp**: `~/llama.cpp/build/bin/llama-server -m Ornith-1.5-9B-Q4_K_M.gguf -c 8192 -ngl 99
  -fa on -b 2048 -ub 512 -t 8 -ctk q8_0 -ctv q8_0`, greedy (`temperature 0, top_k 1`), token-id
  prompt (`tools/llama-ref-run.sh` pattern), its own native K-quant kernels — not requantised
  through our pack.
- Prompt set: `bench/mtp-prompts/*.tokens` (20 prompts) — reusable as-is, tokenizer sha-identical
  to Qwythos per `exchange/lane-moe-report.md`. `GEN_N = 64` (`serve/registry.mojo`).
- Read-back before any number is read (P1): engine sha, `BARO_PACK` path and pack index tensor
  count (442) printed by the engine at load, `BARO_FORCE` id count printed when set, llama.cpp
  `/props` (`ctk`/`ctv`, `n_ctx`), `prompt_n == N` every run. llama-server down whenever `ours`
  holds the GPU and vice versa (P1, `bench/mtp-protocol.md`).

## Gates and frozen predictions

**G1 (step 2, already run and PASSED):** `tools/test_engine_pack_kquant.py` — packed-q8
dequantised vs gguf-py's dequantise of the source K-quant bytes, max error within q8 rounding
(d/2 plus one bf16-rounding ulp) on 4 sample tensors (a Q4_K and a Q6_K trunk tensor, `output.weight`,
and a `blk.32` NextN tensor). Result: PASS, 0 elements over tolerance, observed error 27-34% of
the tolerance band on all 4.

**G2 (`BARO_FORCE` no-op when unset):** 20-prompt A/B, this lane's engine vs `main`'s engine, same
Qwythos q4 pack, greedy, temperature 0. Land rule: 20/20 identical `GENERATED` lines — a single
new `if len(force) > 0:` branch around unchanged code should not be able to move output, but the
branch itself must be proven never taken, not assumed. Falsifier: any of the 20 differs.

**Step 3a — identity.** `BARO_FORCE=<ours/f32-ref stream>` on `ours`, at the same 20 prompts x 64
positions; `llama.cpp` runs its own greedy (not forced) and is compared position-by-position to
the same `ours/f32-ref` stream. Per CLAUDE.md, greedy 64-token equality is not itself the gate
past ~256 ids, but at exactly 64 ids — the identity length already used repo-wide for this class
of check (`bench/q8-protocol.md` G1/G2, `bench/spark-prefill-protocol.md`) — teacher-forced
agreement is still the metric of record because `ours` and `llama.cpp` take genuinely different
paths from the same K-quant bytes (ours: dequant -> bf16 -> int8; llama.cpp: native mixed-K-quant
GEMV) and are not expected to be bit-identical. **Land rule: per-prompt, ours' agreement against
the f32-ref stream >= llama.cpp's own agreement against the same stream, median over 20 prompts.**
Falsifier: ours' median agreement below llama.cpp's median agreement (ours would be *less* faithful
to the exact dequantised weights than llama.cpp's own quantisation, which the error-budget math in
G1 says should not happen: llama.cpp's Q4_K/Q6_K quantisation of the SAME source is a different
lossy step, not a strictly better one, but its per-block error is not obviously worse than our
K-quant-to-q8 chain either — no directional claim beyond "roughly comparable" is frozen here).

**Step 3b — decode speed, 20-prompt median (P4).** Derivation: the Ornith q8 pack measured **9.99
GiB** (`.work/engine-pack-ornith-q8/`), matching the Qwythos-shape q8 prediction (10.35 GB,
`bench/q8-protocol.md`) almost exactly — same architecture, same byte budget. The q8 decode path's
last recorded number on this shape was 68.8 tok/s_gen (2026-09-04, `docs/BASELINE.md`), before the
shared-infrastructure wins landed since (megakernel launch fusion +22%, rmsnorm fold): those apply
to the q8 dot loop too, but the q4-specific chunked-delta schedule that took q4 to 133.9/136.37 does
not. **Prediction: ours 70-95 tok/s_gen.** Falsifier: outside 55-120 (either a regression the shared
wins should have prevented, or a suspiciously large unexplained gain).

llama.cpp Q4_K_M averages ~4.5 bits/weight against Q8_0's 8 bits/weight (~1.7x fewer stream bytes),
scaling the existing Q8_0 no-spec bar (74.1 tok/s_gen, `docs/BASELINE.md`). **Prediction: llama.cpp
110-150 tok/s_gen.** Falsifier: outside 80-200.

No ratio claim is frozen between the two arms — the point of this round is the numbers, not a
predetermined verdict.

**Step 3c — MTP on identical to no-spec (P4).** `bench/mtp-prompts.sh .work/engine .work/ornith-mtp`
at `k=2` (repo default, `BARO_SPEC_K`): arm B (`BARO_SPEC=1`) `GENERATED` must equal arm A
(`BARO_SPEC=0`) `GENERATED`, every one of the 20 prompts — the existing gate, unmodified by this
lane, already track-recorded at 20/20 on Qwythos. **Prediction: 20/20.** Falsifier: any prompt's
arm B differs from arm A — MTP correctness is a kernel property (draft head + acceptance), not a
weight-source property, so a new model diverging here would be a real finding, not noise.

**Step 3d — chat smoke.** One `/v1/chat/completions` request through the Rust front
(`serve/src/main.rs`, `serve/PROTOCOL.md`) against the Ornith pack. No quantitative prediction:
pass/fail is a well-formed response whose `content` a human reads as a coherent answer to the
prompt asked.

## Not in this round

Requantising Ornith through anything other than `--q8`; `--q4-draft` (MTP draft-head Q4, untouched
by the K-quant path — would `KeyError` if invoked on a K-quant `output.weight`, since it still reads
`ge.GGML_BYTES` directly; out of scope, not fixed); serving concurrent requests; any kernel change
(the item's Rules forbid it outright; `BARO_FORCE` needed none, per the amendment above).
