# Team B P5b report

Status: writer, corpus, trainer, and writeback all landed; forced agreement and the quality
table are measured on the corrected patched file; identity is 20/20. One prompt (p09) sits
0.94 percentage points under the frozen 90% forced-agreement floor; that call is left open
below rather than decided here. Codex ran out of usage partway through the CPU writer work;
Sonnet took over the remainder of the item solo, with the coordinator verifying each gate
receipt directly (no partner review).

## Target and corpus

Base source, immutable throughout: `$HOME/Models/qwythos-9b-claude-mythos-5-1m-mtp-bf16/`
`Qwythos-9B-Claude-Mythos-5-1M-MTP-BF16-BARO-04867e2.gguf`, sha256
`742a01c0747b389fc0bf38c8817e6844d8dde9ee216ae4962d80394c42b94130`. Named tensor set, verified
present with the expected shape and dtype: `blk.24.ffn_down.weight` through
`blk.31.ffn_down.weight`, each `[12288, 4096]` BF16 in GGUF's own (reversed) metadata order.

Training corpus: 1,000 GSM8K documents, seed 13, sampled from the 7,323-line pool remaining
after excluding ids 0-149 (P5a's exact self-distillation dump set; verified no blank line
shifts that range, so the exclusion is exact). Disjoint from `bench/mtp-prompts` by content:
those files are 14-25 token synthetic snippets, never GSM8K question-and-answer text (7,473
unique GSM8K questions checked, all far longer). Manifest with id and sha256 per document:
`bench/p5b-corpus/manifest.json`, committed `a6193cf` before any training.

## Generic write-back

`tools/gguf-writeback.py` (codex, commit `064e347`): copies the source GGUF byte for byte and
patches only the reported `data_offset:n_bytes` range for each named tensor, verified against
an independent `verify` pass and a negative control (a byte corrupted outside the named ranges
is caught, `outside_range_diff_bytes=1`, exit 1).

A real bug was found and fixed during the first patched run (commit `031d1bc`, Sonnet):
`tensor_table()` reported GGUF's raw metadata shape (dims fastest-first) directly as the
expected NPZ patch shape. The on-disk bytes are in the reversed, numpy/torch row-major order,
verified byte for byte against an HF-loaded `down_proj.weight` (exact match, diff 0.0, only
when read as `(4096, 12288)`) and independently with a synthetic asymmetric tensor whose
values encode row and column, so a transpose is immediately visible. An identity patch is now
byte-exact only in that reversed shape, and a transposed array is rejected by the shape check
instead of silently accepted; neither property was previously testable, since every earlier
test used a random-valued fixture with no way to detect a transpose. `tools/p5b_lora_train.py`
had a matching stray `.t()` in its merge step, fixed in the same commit.

## Trainer and training run

`tools/p5b_lora_train.py` (Sonnet, commit `0bd225d`, 129 LOC, no peft, no new pip installs):
a hand-written LoRA wrapper on the 8 `mlp.down_proj` Linear layers. The whole forward pass
runs under `torch.no_grad()`; a forward pre-hook on layer 24 flips
`torch.set_grad_enabled(True)` partway through, so blocks 0-23 build no backward graph at all
and blocks 24-31 (a mix of 6 linear-attention and 2 full-attention layer types in this
checkpoint) get normal autograd, verified with an isolated unit test showing zero gradient
signal reaches anything before the hook fires. Merge is `W + (alpha/r) B @ A` in BF16.

Training job `mu5bgdqs4db9`, `gpu-wait run --priority 10 --preemptible --vram 22 --timeout
3600`, terminal `succeeded`, exit 0, wall 240.6 s (well under the 1 GPU-hour cap). Rank 16,
alpha 32, lr 2e-4, accum 10 x n-steps 100, the full 1,000-document corpus in one run. Peak
VRAM per the gpu-wait daemon: 22.46 GB, slightly over the requested 22 GB reservation though
the job did not fail; the script's own `torch.cuda.max_memory_allocated()` reading was 20.02
GB. Loss: 1.0927 to 0.8530, gradual and non-degenerate. Report:
`.work/team-B/sonnet/p5b/train/report.json`.

A first merge attempt (the pre-fix transpose bug above) produced a scrambled patched file:
llama-server returned invalid-UTF8, non-decodable output on 4 of the 20 mtp-prompts (`p03`,
`p04`, `p06`, `p16`), confirmed via a controlled comparison against a freshly Q4_0-quantized
copy of the *unpatched* base model on the identical prompt, which was perfectly coherent. Per
tensor delta diagnostics on that first merge showed `||delta||_F / ||W||_F ~ sqrt(2)` for all
8 tensors, i.e. delta and the base weight looked statistically uncorrelated, exactly what a
transposed matrix produces. After the fix, recomputed from the same training run with no
retrain: `||delta||_F / ||W||_F` 1.5-2.7%, `max|delta| / max|W|` 0.3-0.8% across all 8 tensors,
consistent with a gentle, correctly-applied LoRA patch. The original training run was never
unstable; the bug was entirely in the write-back step.

## Write-back receipt (corrected)

`.work/team-B/sonnet/p5b/train/write-receipt-fixed.json`: `outside_range_diff_bytes=0`, all 8
named ranges changed (44.8-48.8 million of 100,663,296 bytes each), `pass=true`. Repacked with
the unmodified `tools/engine-pack.py --q4`: 442 tensors, 6.18 GiB, matching the baseline pack's
size class.

## Criterion (b): forced agreement vs llama.cpp on the patched file

Reference: `patched-fixed.gguf` quantized to Q4_0 (`llama-quantize --pure`, 4.94 GiB, VRAM-safe
alongside our own engine's 6.18 GiB pack). Gate script:
`.work/team-B/sonnet/p5b/forced-agreement-run.sh` (reuses `bench/qwythos-v2-run.sh`'s
`BARO_FORCE`-against-llama.cpp pattern, pointed at the P5b pack/engine).

First run (job `mu5bxdhelyha`, then `mu5c1161281v` after the script was hardened to survive a
bad response and continue) used the pre-fix scrambled file: 4 of 20 prompts returned llama.cpp
server errors (invalid-UTF8 model output), the rest 55.9%-100% agreement. Corrected run, job
`mu5cexycbl7n`, terminal `succeeded`, exit 0, wall 40.3 s, against the byte-verified fixed
patch: all 20 prompts produce coherent output, zero errors.

| prompt | agreement | pct |
|---|---|---|
| p01-water | 61/64 | 95.31% |
| p02-python-fib | 64/64 | 100.00% |
| p03-story | 62/64 | 96.88% |
| p04-list-planets | 64/64 | 100.00% |
| p05-math | 61/64 | 95.31% |
| p06-translate | 63/64 | 98.44% |
| p07-json | 63/64 | 98.44% |
| p08-sql | 64/64 | 100.00% |
| p09-explain-gpu | 57/64 | 89.06% |
| p10-recipe | 62/64 | 96.88% |
| p11-email | 56/57 | 98.25% |
| p12-rust | 61/64 | 95.31% |
| p13-haiku | 44/46 | 95.65% |
| p14-history | 64/64 | 100.00% |
| p15-bash | 63/64 | 98.44% |
| p16-chat | 62/64 | 96.88% |
| p17-summarize | 63/64 | 98.44% |
| p18-regex | 60/64 | 93.75% |
| p19-numbers | 62/64 | 96.88% |
| p20-dialog | 60/64 | 93.75% |

Aggregate: 1,216/1,255 = 96.89%. Minimum: `p09-explain-gpu` at 89.06%, 0.94 percentage points
under the frozen 90% floor. Every other prompt clears the floor comfortably, next-lowest
93.75%. This is the first correctly-measured result for the original training run; no
hyperparameter repair (the coordinator's proposed lr 5e-5 / 50-step round) has been applied,
since the failure that prompted it turned out not to be a training problem. Whether p09's
narrow miss counts as a kill under the strict "min over 20 prompts" wording, or whether the
allowed repair round should now be spent on it, is left to the coordinator; receipts:
`.work/team-B/sonnet/p5b/forced/results.txt` and per-prompt logs in the same directory.

## Criterion (c): quality table

`bench/quality-run.sh` was broken for every model, not just this one: both its `refcache`
call sites pointed at `~/iTools/bin/refcache`, which now resolves to a rewritten,
CLI-incompatible tool (`harness/refcache/refcache.sh`, positional `key`/`get`/`put`/`ls`
subcommands) instead of the `--key`/`--key-file`/`--out -- CMD` interface the script was
written against. That interface still exists at `~/iTools/dev/refcache/refcache.sh`, unwired
from the `bin/` shim. Fixed by pointing both call sites at the interface that still exists
(commit `ef2c7dd`), rather than rewriting the calls against the newer one.

A second gap surfaced once the refcache fix let the patched row run far enough to reach it:
`bench/quality-ppl-run.py`'s VOID self-check verifies the engine and pack reproduce
`baro.run.ref.tokens`, embedded in the GGUF at its original bake time, before trusting any PPL
number. `gguf-writeback.py` deliberately never touches metadata, so the patched file still
carried the *unpatched* model's reference; the patched model's greedy continuation legitimately
differs after training (`prefix_match` 4/64), which is not a bug, just a reference that no
longer describes this file. Fixed with a full self-describing rebake (`tools/gguf-embed.py`,
the closure list from `tools/embed-files.py`, `--run-harness=serve/engine.mojo`, and a fresh
`--run-ref` generated by running the patched engine against the original
`baro.run.prompt.tokens` prompt) rather than a partial `--run-prompt`/`--run-ref`-only call,
since `gguf-embed.py`'s `rewrite()` drops all `baro.kernel.*`/`baro.run.*`/`baro.hw.*` keys
and a partial call would have silently erased the kernel-source and harness provenance the
original bake carried. Verified the rebake changed only metadata: tensor bytes for
`blk.24.ffn_down.weight` are `numpy.array_equal` before and after
(`patched-fixed.gguf` vs `patched-selfdesc.gguf`). Registered as `qwythos-p5b-patched` in
`bench/quality-models.json` (commits `09271d4`, `a6a73f6`); `bench/quality-bands.json` was
deliberately left untouched, since it is a frozen-predictions file per its own header comment
and a new entry there is a protocol decision, not a bug fix.

| | baseline (qwythos-champion) | patched (qwythos-p5b-patched) |
|---|---|---|
| PPL ours | 9.3007 | 9.3695 |
| PPL llama.cpp | 8.4101 | 8.5046 |
| ppl_ratio | 1.1059 | 1.1017 |
| task ours (raw) | 70/120 | 64/120 |
| task llama (raw) | 80/120 | 73/120 |
| delta_pp | -8.33 | -7.50 |

The two runs' task sets are drawn independently, so the raw counts are not directly
comparable; `delta_pp` (ours minus llama within the same run) is the designed, self-normalized
metric. Both `ppl_ratio` and `delta_pp` land inside `qwythos-champion`'s own frozen band
(`ppl_ratio` 0.85-1.15, `delta_pp` <= 10), and both are within about 0.5-1 point of the
baseline's own numbers: no quality regression. Receipts:
`.work/quality/qwythos-champion/SUMMARY.txt` and `.work/quality/qwythos-p5b-patched/SUMMARY.txt`.

## Criterion (d): 20-prompt identity, unpatched prompts

Job `mu5cxztp53bk`, `bench/mtp-prompts.sh .work/team-B/sonnet/p5b/train/engine
.work/team-B/sonnet/p5b/identity 2`, terminal `succeeded`, exit 0, wall 55.1 s. 20/20 greedy
identity, k=2 spec vs no-spec, zero mismatches. Receipt:
`.work/team-B/sonnet/p5b/identity/results.txt`.

## Open item

Criterion (b)'s minimum, 89.06% on `p09-explain-gpu`, is the only unresolved number against
the frozen bars. Everything else (write-back byte isolation, aggregate forced agreement,
quality table, identity) clears cleanly. Base bake and pack are untouched throughout; nothing
here has modified `qwythos-champion`'s own files.
