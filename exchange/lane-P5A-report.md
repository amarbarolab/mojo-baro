# Team B P5a report

Status: dump path repaired; 150-document rerun and training gate not started after the repair.

## Void run

The original 150-document dump, training smoke, and all downstream gate artifacts are void.
The dump `.work/team-B/sonnet/p5a/dump-smoke-v2.bin` contained 150 documents and 28,360
records, but every record had `top8_ids=[0..7]`, `top8_probs=[0.125]*8`, and
`target_argmax=0`. The original CPU validator passed only finite values, normalization,
alignment, and internal top-8 consistency, so it accepted the all-equal logits.

The void training job was `mu559784w5t0`, using 128 documents and 24,468 pairs. Its loss
fell from 5.320196 to 0.097546 by learning the constant target 0. The checkpoint,
writeback, repack, and any gate based on them must not be used. The training report also
records peak VRAM 22.6616 GiB against `--vram 22` and the two HIP OOM warnings from
`.work/team-B/sonnet/p5a/train-run.log`.

The old 1-document sample also failed the new validator: 98/98 records had one distinct
argmax and `target_argmax == tokens[P+1]` was 0/98.

## Validator repair

Commit `40653a3` adds and enforces:

- distinct `target_argmax` count greater than one for every document;
- reported `target_argmax == tokens[P+1]` count and fraction, requiring a fraction above
  0.30;
- the existing finite, normalization, top-8, and input-alignment checks.

The old sample now fails with exit code 1 and reports `distinct_argmax_by_doc=[1]` and
`greedy_next_fraction=0.0`.

## Dump repair and 1-document sanity

Commit `69469ae` repairs `bench/draft_dump.mojo` without changing shared `make_cfg` or
`serve/realign.mojo`. Dump mode locally disables the megakernel and prefill paths, forces
one-row teacher-forced host-layer steps, captures hidden `h_(P-1)`, then advances one
teacher-forced row and reads head logits from `h_P` for the target predicting `tokens[P+1]`.
An enforced token readback rejects any teacher-forcing overwrite.

Clean Mojo build exit code: 0. Final 1-document GPU verification exit code: 0. The debug
sample `.work/team-B/sonnet/p5a/dump-debug6.bin` passed the CPU validator with:

- 1 document, 98 records;
- 43 distinct target argmax values;
- `target_argmax == tokens[P+1]`: 86/98, 87.7551%;
- target argmax equals `tokens[P]`: 0/98;
- finite, normalized, top-8-consistent, and input-aligned: all pass;
- teacher-forcing token identity: 0/100 mismatches.

Validator receipt: `.work/team-B/codex/p5a/dump-debug6-check.json` and
`.work/team-B/codex/p5a/dump-debug6-check.log`.

The repaired 150-document dump was generated and validated before the training smoke below.

## Repaired 150-document dump

The apparent post-GO truncation was a read-while-write observation, not a failed dump. The
only gpu-wait job recorded as writing `dump-150.bin` was `mu56eut4wbxt`, started at 08:56:22,
which completed in 269.1 seconds with exit 0 and cause `ok`. The validator read the file while
that job was still running, when it had reached document 54, record 128. `mu568dunh0qm` was a
separate 2.6-second one-document `dump-final-check.bin` verification job. No timeout or early
exit caused the 150-document file, and the in-progress read was not used as a gate result.

The operational rule is now: validate a dump only after `gpu-wait status <job-id>` is terminal
with `succeeded` and the producer log contains its completion line.

The dump process itself does not yet fail loudly for that interruption path: its completion
line is printed only after the document loop, and the binary has no committed completion
marker while the `write_bytes` return values are not checked. The CPU validator fails loudly
on an in-progress partial read with a truncation error and exit 1, but that exploratory read
was not a gate result. The successful rerun below proves the expected size and completion
line, while a future hardening change should make the producer write and verify an explicit
completion marker and propagate short-write errors.

The rerun uses the committed fixed binary `69469ae` (binary SHA recorded by Sonnet as
`ad76d02...143ca0`) and completed under job `mu56eut4wbxt`. Its log ends with
`draft_dump: wrote 150 documents`, and the file size is 467,034,884 bytes. The full CPU
validator passed with exit code 0:

- 150 documents, 28,360 records;
- every document has more than one distinct argmax, minimum distinct count 37;
- aggregate `target_argmax == tokens[P+1]`: 21,556/28,360, 76.0085%;
- minimum per-document next-token fraction: 0.6000;
- finite, normalized, top-8-consistent, and input-aligned: all pass;
- teacher-forcing token identity: all documents pass the runtime invariant.

The validator now reports and enforces the per-document minimum in commit `5519b3a`.
Receipts: `.work/team-B/sonnet/p5a/dump-150.log`,
`.work/team-B/codex/p5a/dump-150-check.json`, and
`.work/team-B/codex/p5a/dump-150-check.log`.

## Repaired training smoke and acceptance gate

The corrected 128-document smoke completed under gpu-wait job `mu56rggzgtqm`, terminal
state `succeeded`, exit 0, with the original recipe: learning rate 2e-5, accumulation 16,
gradient clip 1.0, 10 percent warmup, seed 13, and 8 steps. It consumed all 24,468 expected
pairs from 128 documents. Per-step loss was:

`0.8223648, 0.6731094, 0.6557963, 0.6977744, 0.6533771, 0.6624775, 0.7071903, 0.6903085`.

Per-step gradient norms were:

`7.9445219, 6.8329229, 5.4829082, 5.2668986, 4.4691958, 4.4910812, 4.4835353, 4.5246639`.

Loss decreased from 0.8223648 to 0.6903085. The checkpoint is
`.work/team-B/sonnet/p5a/trained-blk32-v2.pt`, and the report is
`.work/team-B/sonnet/p5a/train-report-v2.json`. The report records peak VRAM 22.6616 GiB
against `--vram 22`; `.work/team-B/sonnet/p5a/train-run-v2.log` retains two recovered
HIPCachingAllocator OOM warnings. This smoke is not void.

Writeback receipt `.work/team-B/sonnet/p5a/writeback-verify-v2.json` passed: all 15
`blk.32` tensors changed and `outside_blk32_diff_bytes` is zero. The q4 repack produced
`.work/team-B/sonnet/p5a/trained-pack-v2`, and engine `engine-v2` has SHA-256
`f54e06678ab182f4911c7faf27a7bb0bc90875ac7885b41675f5fe7fcd6439f6`; Sonnet reports it
matches the baseline engine exactly.

The acceptance gate ran under gpu-wait job `mu56xxfgxv0c`, terminal state `succeeded`, exit 0,
cause `ok`, 54.3 seconds wall, peak VRAM 21.81 GB, with command
`bench/mtp-prompts.sh .work/team-B/sonnet/p5a/engine-v2 .work/team-B/sonnet/p5a/trained-full-v2 2`.
It ran against the trained pack at k=2. All 20/20 greedy identity checks
passed. The full drafted set accepted 726/1080, 67.22%, versus baseline 721/1093, 65.97%,
for +1.26 percentage points. The p01-p05 subset accepted 185/264, 70.08%, versus baseline
183/270, 67.78%, for +2.30 percentage points. The required subset lift is +4 points, so
the acceptance gate is `NO SIGNAL`: P5a parks under the plan kill line. No gate-breaking
identity or data-path failure was observed. Gate receipt: `.work/team-B/sonnet/p5a/trained-run-v2.log`
and `.work/team-B/sonnet/p5a/trained-full-v2/results.txt`.
