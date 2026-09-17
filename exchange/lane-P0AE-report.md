# P0a-e report: embeddings engine wire (coordinator, fable, 2026-09-17)

Branch `lane-p0ae`, commit `4603891`. Item split off P0a when team A found that no wire field
carries a hidden state.

## What landed

- Request field `"embed":true` (`serve/serve_proto.mojo`, carried on `SampleParams.embed` so
  `parse_request` keeps its signature).
- `serve/engine.mojo`: after the step that processes the last prompt token, copies `hn_d` row 0
  (post-final-norm hidden state), L2-normalizes on the host and prints `{"id":ID,"embed":[H floats]}`
  after the first token line and before `done`.
- `serve/window.mojo`: `plain_head` is false for an embed request, so the head is not folded and
  the launch path writes `hn_d`. The folded head (`fold_head` 1) never writes it.
- `serve/spark.mojo` and the multi-sequence parser answer an error line for the flag.
- `serve/PROTOCOL.md`: the line is in the engine output table.

## Gate

`bench/p0ae-embed-gate.sh` (dry-run stop, re-execs under `gpu-wait run --vram 24 --timeout 1500`).
Reference arm: `llama-embedding --pooling last --embd-normalize 2 -ngl 99` on the Q4_0-pure GGUF,
flags echoed to `.work/p0ae/llama-flags.txt`. Bars frozen in the script header before the first run.

| run | log | result |
|---|---|---|
| 1 | `.work/p0ae/gate-1.log` | FAIL: cosine 0.26 to 0.72, retrieval 7/8; identity, norm, determinism passed. Cause: stale `hn_d` under the folded head. |
| 2 | `.work/p0ae/gate-2.log` | PASS exit 0: cosine min 0.9882 (code prompt), 7 of 8 at 0.998 or higher, retrieval 8/8, norm 1, two runs identical, `ref-tokens-64.txt` reproduced with and without the flag, `"embed":7` refused. |

## Suite and checks

- `gpu-wait run --vram 24 --timeout 3600 -- ./run-tests.sh` on `4603891`: exit 0, 0 FAIL lines,
  `.work/p0ae/run-tests.log` ends `rc=0`.
- `tools/ci-checks.sh`: all non-GPU checks passed.
- spark builds with a profile on its include path (`.work/p0ae/build-spark.log`, exit 0).

## UNVERIFIED

- The spark refusal and the `BARO_SEQS > 1` refusal compile but were not exercised by a request.
- The HTTP routes: team A builds the Rust side against this wire; P0a gate 4 judges it.
- Cost of the unfolded head on an embed request was not timed; it is one decode step per request.
