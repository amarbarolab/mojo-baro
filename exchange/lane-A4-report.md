# Lane A4 report: trained draft head, PARTIAL, stopped at the parity gate

Brief: `briefs/2026-09-17-A4-draft-head.md`. Plan: `docs/NEXT-PLAN.md` A4.
Preregistration: `bench/draft-head-protocol.md`. **Not done: no training ran,
no smoke result exists. This session stopped at the parity gate the plan's
own Order step 1 puts before any GPU training minute, because the gate did
not clearly pass (CLAUDE.md s18: no check landed, so the word is
UNVERIFIED, not done).**

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

## Recommendation

Before spending any training GPU minutes: resolve the parity ambiguity by
rerunning against the real 20-prompt gate set through the real spec-decode
harness. If that reproduces 42% and torch-vs-engine agreement rises well
above the current 62%, the smoke (Order step 3) is next. If 42% still does
not reproduce on real gate-set text, the bug is real and unfixed, and no
training should run against this wiring.
