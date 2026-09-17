# P1 gate 4, the llama.cpp bridge: frozen before the first GPU run

Gate 4's bar has never been cleared by anything in this repo, so a miss could not tell a broken
bridge from ordinary numerics. The plan asks that "E15's three models continue from our state with
the first 32 tokens identical". Our dense path, the one that ships, reaches 51.90 of 64 against
llama.cpp on a quant-matched arm (`bench/PROTOCOL-RULES.md` P14), because our activations pass through
bf16 and llama keeps f32. This note keeps the plan's bar word for word as the primary, measures what
a known-good configuration reaches on the same comparison (P14), and proves the gate can fail (P11),
all before the bridge is judged. Frozen by commit before any gate 4 GPU minute.

Lane FORK, `briefs/2026-09-17-latentos-cross-node-fork.md`. Tool under test:
`tools/state-to-llama-slot.mojo` (`2d4ea95`). Spark's export-only `state_save` is new in this lane.

## The bar is the plan's, unchanged, and it is not mine to move

**Primary.** Per model, 20 prompts (`bench/mtp-prompts/p*.txt`, tokenized per model with our
tokenizer, as `bench/dense-run.sh` does). Our engine prefills P and saves its state at `|P| - 1`. The
tool writes a slot file. `llama-server` restores it and completes P for 32 tokens, greedy. Those 32
ids are compared with `llama-server`'s own COLD completion of P. PASS as written: 20 of 20 identical,
on each of the three models.

If the primary is not met, the report says NOT MET with the counts, in the same register as a pass.
Whether the plan's bar should be amended under P14 is the coordinator's decision, because the
coordinator owns the plan. This lane measures and classifies. It does not rewrite the bar, and it
does not call a numerics flip a pass.

## A silent recompute would pass 20 of 20, so reuse is checked on every item

If `llama-server` ignores the restored slot and re-prefills P, the bridged ids equal the cold ids
trivially and the gate passes having tested nothing. So every bridged item must report
`cache_n == |P| - 1` and the restore must report `n_restored == |P| - 1`. Any other value voids the
item. **One void makes the gate VOID, and a void is a failure (P10), never a pass.** Every cold item
must report `cache_n == 0` after a slot erase, or the reference was not cold (the reference arm is an
arm).

## Two controls say what a known-good path reaches (P14)

- **Control L, the restore path with the bridge taken out.** `llama-server` saves its OWN slot at
  `|P| - 1`, a fresh server restores it and completes P. Same ids comparison. This is E15's KV arm.
  It bounds what llama.cpp's restore delivers when the bytes are its own.
- **Control N, our numerics with the bridge taken out.** Our engine's own cold 32 ids against
  `llama-server`'s cold 32 ids. No state moves. This is how often the two engines agree greedily on
  these prompts at all.

The bridged arm perturbs only the prompt's KV (ours, about 0.44 percent per layer off llama's) and
then decodes all 32 tokens on llama.cpp's kernels. Control N also decodes on ours. So the primary
should land at or above control N, and at or below control L.

## The gate is fed a known-bad state first (P11)

Before any pass is read, the harness runs the primary on one model with a deliberately wrong slot
file: the tool's K and V sections swapped (`BRIDGE_FALSIFY=swapkv`, a harness-side byte swap of the
written file, not a flag in the tool). The reuse checks still hold (same cell count, same layout), so
only the ids can catch it. The gate must FAIL loudly on it. If the swapped file passes, the gate
cannot see the bridge and nothing below means anything.

Already on record for the layout, no GPU: `bench/bridge-roundtrip.sh` PASS on Qwythos-9B (qwen35,
f16 KV): a slot file written by llama.cpp, through the reverse tool and back through this tool, is
byte-identical (53,183,716 bytes), restores with `cache_n 15`, and continues with 32 equal ids. The
same round trip runs on the qwen2 and llama GGUFs with `--ext 0` before their GPU runs (P15).

## Arms are read back, not assumed (P1)

`llama-server` `ca3d5a3e1`, `-ngl 99 -fa on -np 1 -ctk f16 -ctv f16`, the same `-b`/`-ub` on every
arm, greedy, `cache_prompt: true`. Read back from `/props` and the server log into the arm file:
model path and sha256, KV types, `n_ctx`, flash attention on, build commit. Ours: engine binary
sha256, pack path, `pack.sha256`, `BARO_SPEC=0`, KV dtype line from the engine log, the `state saved`
line with `pos` for every item. The tool's `--kv` matches the server's KV type; a mismatch is refused
by llama.cpp's own type check, which is a loud failure, not a silent one.

Models (E15's three): Qwen2.5-7B-Instruct Q4_K_M (`qwen2`, `serve/spark.mojo`),
lily-cybersecurity-7b-v0.2 Q6_K (`llama`, `serve/spark.mojo`), Ornith-1.5-9B Q4_K_M (`qwen35`,
`serve/engine.mojo`). Our engine and `llama-server` never overlap on the GPU; every process runs
under `gpu-wait run --timeout`. The cold llama.cpp ids are the deterministic reference arm and are
cached by model sha, binary commit, params and input ids (P17); ours are never cached.

## Frozen predictions

| | Qwen2.5-7B | lily-7B | Ornith-9B |
|---|---|---|---|
| Control L, identical of 20 | 20 | 20 | 19 to 20 |
| Control N, identical of 20 | 8 to 14 | 8 to 14 | 12 to 18 |
| **Primary, identical of 20** | **17 to 20** | **17 to 20** | **17 to 20** |
| Falsifier (K and V swapped), identical of 20 | 0 to 2, run on one model | | |
| Voids | 0 | 0 | 0 |

My point estimate for the primary is 18 of 20 per model. That is a prediction that the gate AS
WRITTEN is more likely NOT MET than met on at least one model, from numerics and not from the
bridge. I would rather write that down now than explain it afterwards.

## A miss is classified by rule, decided here and not after the fact

- **Layout defect (the bridge is wrong):** divergence at index 0 to 2 on most prompts of a model, or
  a primary more than 2 prompts below that model's control N. I expect a correct bridge to do no
  worse than moving no state at all and decoding on our own kernels, but that is an expectation and
  not a theorem (their kernels over our KV is a mix control N never runs), so the margin of 2 in 20
  keeps a chance inversion from firing the kill line on a correct bridge. This fires the kill line.
- **Numerics (the bridge is right, the bar is P14's problem):** primary within 2 prompts of control N
  or above it, misses on a minority of prompts, first divergence at index 3 or later. This does NOT
  fire the kill line, and it is still reported as NOT MET against the plan's wording.
- Anything that fits neither is reported as UNCLASSIFIED with the per-prompt divergence indices, and
  no cause is proposed for it in the report.

## What this cannot show

That a phone or a Mac continues our conversation: the receiver here is `llama-server` on the same
XTX. That 8k or 32k states bridge correctly: the prompts are short, and the tool's cell loop is
scalar, so a 32k conversion is minutes of CPU and unmeasured. That the int8 state format bridges
with the same ids: the primary runs f32 states; int8 is a secondary row, reported, not gated.

## Amendment 1, 2026-09-17, before any gate 4 GPU run: control L already missed once, on CPU

The CPU preflight this note requires ran the round trip on all three architectures
(`bench/bridge-roundtrip.sh`, llama-server `--device none`, f16 KV). The tool's output was
byte-identical to llama.cpp's own slot file on all three, and llama.cpp reused it on all three
(`cache_n` equal to the restored length). On qwen35 and llama the restored continuation equalled the
cold one. **On qwen2 it did not: llama.cpp restoring its OWN bytes diverged from its own cold run at
index 1** (`.work/fork/bridge-roundtrip-qwen25`). Both continuations are fluent and correct (an
iterative and a recursive fibonacci), a near-tie opening that flips with batch shape: the cold run
evaluates 27 tokens in one batch, the restored run evaluates 13 over 14 cached f16 cells.

What this changes, and what it does not:

- **My control L prediction for Qwen2.5 (20 of 20) is contradicted by this item before the GPU run.**
  It is left in the table as frozen. It is one CPU item on a different backend, so it does not
  predict the GPU count, and I am not replacing the number with a better guess now that I have seen
  data.
- It shows the classifier's early-index signal is weaker than I assumed when I froze it: a
  byte-perfect restore produced "divergence at index 1". The rule already says "on MOST prompts of a
  model", so one prompt does not trip it, and **no threshold is changed**.
- **Added reporting, not a changed bar:** every primary miss is also labelled by whether control L
  missed the same prompt. A prompt llama.cpp cannot reproduce from its own bytes is attributed to
  llama.cpp's restore path, not to the bridge and not to our numerics. The raw primary count against
  the plan's wording is reported unchanged beside it.
- `bench/bridge-roundtrip.sh` exits on what can implicate the tool (byte identity with an
  independent producer's file, `n_restored`, `cache_n`, 32 tokens generated) and prints
  `restored_equals_cold` as its own line. That line was a hard failure until it failed on bytes the
  tool did not author. The change was made after that failure, which is why it is written down here.
