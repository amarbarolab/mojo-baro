# lane-chat report — M1a prefix checkpoints (2026-09-08, paused)

Branch `lane-chat` at 7b4b9d0 (worktree `~/Projects/mojo-baro-lanes/chat`).
Status/logs: `~/Projects/mojo-baro/.work/briefs/status-chat.md`, worktree `.work/m1a/`.

## Delivered
- `serve/harness.mojo` (new): `load_pack`, `alloc_bufs` moved out of engine main.
- `serve/prefix.mojo` (new): `Checkpoint`, `Chain` (FNV-1a 64 hash, cap `BARO_CKPT`=8,
  52.69 MB pinned each, lookup / save / commit / restore / invalidate_above).
- `serve/engine.mojo`: serve loop restores the longest hashed prefix, keeps the KV
  pool, replays `tokens[r:]`, checkpoints every 1024 rows and at `len-1`;
  receipts `cached`, `prefill_rows`, `restore_s`, `checkpoints` (done line),
  `checkpoints: cap N, bytes X MB each` at start.
- `serve/src`: `usage.baro.cached_tokens` / `prefill_rows`; cargo test 9/9.
- `kernels/test_prefix.mojo` in `run-tests.sh`; `docs/KERNELS.md` gains a generated
  Tests table (`tools/kernel-census.py`). `tools/tap-replay.py` for P-F3.

## Gates
- P-F1 byte-exact restore: PASS, megakernel and window path, every comparison
  zero differing words (conv, delta, KV [0,1153), hmax/hidx / logits, next token);
  all lookup mutations as predicted (with the len-1 checkpoint, "last token of A"
  = index 1086). Log `.work/m1a/test_prefix.log`.
- Serve smoke: exact repeat of 1088 tokens cached 1087 / 0 rows / prefill_s 7.6 ms
  (cold 810 ms), A||B cached 1087 / 64 rows / 83 ms, tokens identical to cold.
- NOT run: merge-gate.sh (§4.1), 20-prompt A/B (P-F2/P-F4), tap replay (P-F3),
  Result section, M1b preregistration. Commands in the status HANDOFF.

## Deviations from the frozen text (stated)
- Prompt-end checkpoint at `pos = len-1` (after prefill_forward, before the last
  prompt token's decode step), because prefill and decode kernels are only
  rel-1e-4 equal; one token less reuse.
- Restore invalidates checkpoints above the restore point; a miss drops all.
- New unowned file `serve/harness.mojo`; `tools/kernel-census.py` extended.

PAUSED: budget stop (the maintainer 13:3x) before the gated timed runs
