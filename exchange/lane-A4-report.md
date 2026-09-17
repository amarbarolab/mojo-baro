# Lane A4 report: trained draft head, smoke ran, kill line FAILS (large regression)

Brief: `briefs/2026-09-17-A4-draft-head.md`. Plan: `docs/NEXT-PLAN.md` A4.
Preregistration: `bench/draft-head-protocol.md`. This session's first pass
stopped at the parity gate because the numbers looked inconclusive; the
first follow-up reran against the real 20-prompt gate set through the real,
unmodified engine and harness, reproduced the documented 65-66% baseline
directly, and resolved the earlier ambiguity in favor of domain shift over
a wiring defect. the maintainer then authorized the smoke (Order step 3); it ran to
completion this session and its kill line FAILS: the trained head's real
accepted/drafted rate dropped from 67.78% to 3.06% on the 5-prompt subset
(-64.7 pp, not the required +4 pp lift), full 20-prompt greedy identity
stayed 20/20 PASS. Detail in "Order step 3" below. Order step 4 (the full
preemptible training run) has not started and should not start against
this recipe. Per CLAUDE.md s18, every number below names the check that
produced it; nothing is claimed as more than what its own receipt shows.

## The E13 question, answered

`bench/draft-head-protocol.md`'s "Why the E13 null does not predict an A4
null" section is the required paragraph. Short version: E13 trained a
projector to solve a task the trunk cannot do at all unaided (read its own
out-of-manifold hidden state fed back as a substitute embedding, answer a
multi-step problem from one end-of-sequence bit of supervision). A4 predicts
the literal next real token from the trunk's own real hidden state, dense
supervision at every position, on an architecture (`blk.32`) that already
gets 42% second-token acceptance out of the box. None of E13's three sources
of difficulty apply here. E13's null does not predict an A4 null.

## Order step 1: shapes and architecture, FITS, no kernel change (verified)

`blk.32`'s 15 packed tensors (`tools/engine-pack.py`'s `mtp_names()`) are
retrained AS THEMSELVES: same shapes, same pack slot, same kernels,
`docs/mtp-notes.md` SS1-2 confirmed structurally identical to a full-attention
trunk decoder block plus one input stage (`eh_proj`) and one output stage
(`shared_head_norm`). Verified this session, not asserted:

- `~/AMDHQ/tools/latent-os/gguf_to_hf_qwen35.py`'s HF checkpoint has ZERO
  `mtp.*` keys (`model.safetensors.index.json`, checked directly): the MTP
  head's own real weights are not in the training-side checkpoint, only in
  the GGUF. `tools/mtp_head.py --mode extract` pulls them straight from the
  GGUF (`Qwythos-9B-Claude-Mythos-5-1M-MTP-BF16-BARO-04867e2.gguf`), reusing
  the converter's own tensor-mapping convention for a full-attention block
  (verbatim transpose, verbatim "+1" RMSNorm bias), applied to block 32.
- `tools/mtp_head.py`'s `MTPHead` module (`eh_proj` + 3 norms + one
  `transformers` `Qwen3_5DecoderLayer` forced `full_attention`) loads those
  extracted weights with an EXACT key match (`missing`/`extra` both empty
  sets, checked this session) and runs a forward pass end to end on CPU with
  real weights, finite output, correct shape.
- One real bug the CPU check caught before any GPU spend: `rotary_emb`
  wants the 3-channel mrope split (`position_ids[1:]`), not the 4-channel
  form `Qwen3_5TextModel.forward` builds before splitting it; fixed in
  `tools/mtp_head.py`.

## Parity gate: INCONCLUSIVE, not passed, not failed clean

`bench/draft_dump.mojo --mode parity` (new tool, builds clean, `-I . -I
kernels -I serve -I bench -I ~/Projects/mojo/mojo-uregex/src`) calls the real
`blk32_forward` exactly as `serve/window.mojo`'s own two call sites do and
records its argmax alongside the trunk's real hidden state, for later
torch-side comparison. **Building this tool is what found a real alignment
bug**: my first version assumed `tok_pos = pos + 1`; reading both real call
sites in `window.mojo` (`blk32_forward(..., st.pos_prev+1, st.pos_prev+1,
...)` and the self-chained `blk32_forward(..., st.pos+j, st.pos+j, ...)`
followed by `tokcp_k(..., st.pos+j+1, ...)`) shows `pos == tok_pos` always,
output landing at `tok_pos+1`. Fixed and rebuilt; `bench/draft-head-protocol.md`
now states the correct alignment with the receipt.

After the fix, on 6 documents / 69 positions from `bench/data/e8_tasks.json`
(structural check only, not training data: this is the gate set's own
sibling text, never to be trained on):

| metric | value |
|---|---|
| torch argmax == engine argmax | 43/69 (62%) |
| torch argmax == true next token | 7/69 (10%) |
| engine argmax == true next token | 9/69 (13%) |

13% is well below `bench/mtp-protocol.md`'s documented 42% baseline, and
would fail the smoke's own void check (baseline within 5 pp of 42%). Two
explanations are open and NOT distinguished this session: a genuine domain
shift (42% was measured on natural-language chat prompts, these 6 documents
are math/JSON items), or a remaining wiring defect on top of the one already
found. The 62% torch-vs-engine figure is weak evidence either way: an
earlier, WRONG alignment produced a similar 66%, because consecutive real
hidden states are correlated enough that even a shifted pairing looks
plausible. **Next concrete step, not done here: rerun this same parity
check against the actual 20-prompt gate set through the real spec-decode
harness (`cfg.spec=True`, `bench/mtp-prompts.sh`-class), which gives an
apples-to-apples reproduction of 42% instead of a hand-rolled
`blk32_forward` call on different text.**

## What exists, on `lane-a4`, not yet merged

- `bench/draft-head-protocol.md`: frozen preregistration, E13-null analysis,
  shape verdict, data/training/write-back design, smoke gate, this session's
  real parity numbers.
- `bench/draft_dump.mojo`: real-text dense (h, next-token) dump (`--mode
  dump`, not yet run against real held-out data, only `--mode parity` run so
  far) and the parity-check dump (`--mode parity`, run). Builds clean.
- `tools/mtp_head.py`: GGUF extraction (`--mode extract`, run, verified),
  `MTPHead` torch module (CPU-verified forward), parity comparison (`--mode
  parity`, run), and an untested `--mode writeback` (patches a copy of the
  GGUF in place with trained blk.32 weights, then the unmodified
  `tools/engine-pack.py` repacks it; the byte-offset math is written but has
  no receipt yet since nothing has been trained).

## Not done (CLAUDE.md s18)

No training ran (Order step 3's smoke). No write-back happened. No
acceptance number exists for a trained head. `kernels/` untouched, as the
brief requires. `run-tests.sh`/`tools/ci-checks.sh` run against this branch
before this report (results in the commit this report lands with).

## Update: real gate-set rerun (this session's follow-up)

Did the recommended step. Built `.work/a4/engine` (`mojo build serve/engine.mojo
-I . -I kernels`), ran the REAL, unmodified `bench/mtp-prompts.sh` (not a
hand-rolled call) through `gpu-wait`, quick subset first (3 prompts), then
all 20:

**Quick subset (p01-p03, k=2):** 3/3 greedy identity PASS, accepted/drafted
34/59, 39/50, 32/62 (aggregate 61.4%). Already a strong signal against the
earlier 13% math-text number.

**Full 20 prompts, `bench/mtp-prompts.sh .work/a4/engine .work/a4/mtp-full 2`
(`.work/a4/mtp-full/results.txt`):** **20/20 greedy identity PASS.** Aggregate
`accepted/drafted` = 721/1093 = **65.97%**, mean per-prompt = 67.0%. **This
reproduces the documented "0.66 on the 20 prompts" baseline directly, via
the real unmodified engine and harness, cleanly.** The void check the smoke
section requires (baseline within 5 pp of the documented figure) PASSES on
this reproduction.

**Torch-vs-engine on the same real prompt files** (`bench/draft_dump.mojo
--mode parity` with a new `A4_TEXT_MODE=tokens` path added this session to
read `bench/mtp-prompts/*.tokens` directly, same files, never trained on):
quick subset (3 prompts, 42 positions) 35/42 (83%) torch-vs-engine argmax
agreement, 14/42 (33%) engine-vs-true-next; full 20 prompts (339 positions)
241/339 (71%) torch-vs-engine, 64/339 (19%) engine-vs-true-next. Both well
above the 62%/13% figures measured on `e8_tasks.json`'s math/JSON text,
confirming domain shift was the dominant effect there, not a wiring defect.
**A second real bug found while doing this: `tools/mtp_head.py`'s
`read_parity` only read the FIRST document's block from a multi-document
dump file** (`bench/draft_dump.mojo` writes one `[n_pairs][pairs...]` block
per document, back to back; the reader had no outer loop, so the "full
20-prompt" run's first pass silently returned the same 11-position numbers
as the 3-prompt quick run, byte-for-byte, until noticed and fixed).

**Why 19% (engine-vs-true-next, prompt replay) does not match 42%/66%
(the documented decode-phase figure), and this is not a further bug:**
`bench/mtp-prompts/*.tokens` are PROMPTS ONLY (`n_prompt` in
`bench/mtp-prompts.sh`'s own output, 7 to 59 tokens) with no generated
continuation appended. `bench/draft_dump.mojo --mode parity` walks that
prompt and compares blk.32's prediction against the PROMPT's own literal
next word, i.e. against human-written prose. The documented 42%/66% figures
are measured during DECODE, where the accept rule compares the draft's
prediction against the TARGET MODEL's OWN greedy choice at that step, not
against arbitrary human text: a materially easier, more self-consistent
target (both numbers come from the same weights on the same context). These
are two different statistics; a numeric gap between 19% and 42% says
nothing about correctness by itself, unlike the 13%-vs-62%-torch-vs-engine
comparison, which was informative because it held the SAME statistic
constant and only changed the text domain. **This wiring's real-engine
reproduction of the 66% headline figure, direct and unmodified, is the
answer to the question this recommendation asked.**

## Revised verdict

The baseline reproduces cleanly (20/20 identity, 65.97% aggregate acceptance
against a documented 65-66%). Domain shift, not a wiring defect, explains
the earlier low numbers on math/JSON text; that reading is now supported by
a second, independent measurement (torch-vs-engine rising from 62% to 71%
on real gate-set text) rather than assumed. The alignment fix from the first
pass of this report (`pos == tok_pos`) and this pass's `read_parity` fix are
both load-bearing and now behind real receipts. **Still not done: no
training has run.** The parity gate is now a reasonable basis to proceed to
Order step 3 (the 10-minute smoke), which stays the maintainer's call, not
self-authorized here.

## Order step 3: the smoke ran. Kill line FAIL, large regression, not just NO SIGNAL

the maintainer authorized the smoke and restated the kill line against the reproduced
statistic (`bench/draft-head-protocol.md`, frozen commit `7044828` before
any training minute): same harness, `bench/mtp-prompts.sh`, 5-prompt quick
subset, trained head must lift `accepted/drafted` by >= 4 pp over the
untrained head in the same stint, full-20-prompt greedy identity must stay
20/20 PASS regardless.

**Data (Order step 2).** `bench/draft_dump.mojo`'s new `A4_TEXT_MODE=gsm8k`
tokenizes GSM8K `train.jsonl` question+answer text through the engine's own
tokenizer (genuinely held out from both eval sets used in this lane). 35
documents dumped, 6,923 total (h, token) pairs available.

**Training (`tools/mtp_train.py`, one optimizer step per document, capped at
2000 total supervised positions -- blk.32's own attention keeps a KV cache
across a document, `serve/window.mojo`'s `kc32_d`/`vc32_d`, so shuffling
individual positions across documents would feed the wrong causal context).**
Ran via `gpu-wait run --priority 10 --preemptible --vram 22 --timeout 900`.
9 documents, exactly 2000 pairs, **22.3 s wall** (well under the 10-minute
budget). AdamW lr 2e-4, `blk.32`'s real GGUF weights as the starting point
(fine-tune, not from scratch). **Loss went up, not down**: first 6.99, last
7.97, one step spiked to 24.55 (worse than the ln(248320) ~= 12.4 random-guess
floor for part of the run) -- recorded, not gated, but a real warning sign
this session did not act on before the gate. One PyTorch OOM warning at
peak VRAM (22.15 GB against a 22 GB request, other GPU load present) that
did not crash the run.

**Write-back receipt (frozen requirement, checked independently of the
write-back code's own arithmetic).** `tools/mtp_head.py --mode writeback`
patched a copy of the source GGUF; `--mode verify-writeback` (new this
session, using `numpy.memmap` after a first attempt that tried to
byte-compare two 18 GB files with a pure-Python loop and silently produced
nothing for several minutes) confirms **zero differing bytes outside the 15
`blk.32` tensor ranges, and all 15 ranges changed** (`.work/a4/writeback-verify.json`,
`"pass": true`). `tools/engine-pack.py`, UNMODIFIED, repacked the patched
GGUF: 442 tensors, 6.72 GiB.

**The gate, run.** Untrained pack, `p01`-`p05`, same stint: drafted 270,
accepted 183, **67.78%**. Trained pack, same 5 prompts, same engine binary,
same session: drafted 589, accepted 18, **3.06%**. **Lift: -64.7 pp, the
opposite of the +4 pp the kill line requires.** Full 20-prompt greedy
identity: **20/20 PASS on the trained pack** (the accept rule's own
guarantee held: a badly miscalibrated draft still falls back to the
target's own token on every rejection, so final output is unaffected).
Full-set aggregate acceptance on the trained pack: 51/2398 = **2.13%**
against the untrained 65.97%, so the 5-prompt subset's collapse is not a
small-sample artifact.

**Diagnostic (not part of the frozen gate, run to separate a training-recipe
problem from a wiring regression): torch replica loaded with the TRAINED
weights vs the trained-pack engine, same 5 real gate-set prompts.** Argmax
agreement 31/69 = 45% (down from 71-91% on the untrained head); torch's own
argmax matches the true next token 4/69 = 5.8%, and the real engine's
matches 4/69 = 5.8% -- the SAME low number, in the exact-precision torch
replica as well as the quantized engine. That the degradation shows up
identically in float32 torch (no q4 quantization involved) is why this
reads as a genuinely damaged set of weights, not a write-back or repack
defect layered on top of a fine head: the write-back receipt already showed
the bytes moved exactly where intended, and now the MEANING of those bytes
independently checks out as bad in two different numerical paths.

**Read on the training recipe, not verified further this session but named
because it is the obvious next lever.** `lr = 2e-4` was carried over from
`e13_train.py` unchanged, which used it under 16-item gradient accumulation
before every optimizer step; this run's own design does ONE step per
document (batch effectively 1, no accumulation), so the same learning rate
is applied to a much noisier, un-averaged gradient every step -- a likely
mismatch, not a like-for-like reuse of E13's tuned setting. 9 steps on one
narrow domain (GSM8K math/answer text only) with no warmup and no gradient
clipping is a plausible recipe for exactly this kind of coarse, uniform,
damaging update (every one of the 15 tensors' weights moved by the same
~0.0015-0.0018 max-abs amount, the Adam-at-this-lr-and-step-count
signature, not a gradient-direction-driven one).

**Verdict, per the frozen rule.** This is NOT a kill of the A4 hypothesis
(the protocol's own language: a minutes-scale, unconverged, un-tuned probe
cannot kill it, only say what this specific attempt showed) but it is a
clear NO SIGNAL, and a starker one than "no lift": this recipe actively
damaged the head. Before any further training GPU time: lower the learning
rate and/or restore E13's own multi-item gradient accumulation before an
optimizer step, add gradient clipping, and widen the training text beyond
one narrow domain. None of that is done here -- reported, not fixed,
per the maintainer's own call on whether and how to spend more GPU time on this.

**Still not done: Order step 4 (the full preemptible run) has not started,
and should not start against this recipe.**
