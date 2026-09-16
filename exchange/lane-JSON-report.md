# lane JSON report (2026-09-16)

Brief: `~/Brain/mojo/mojo-baro/briefs/2026-09-16-json-enforcement-lane.md`. Pane `w82:p9` was killed by
the 14:33 OOM with items 1, 2, 4, 5 written and uncommitted; the main session committed that state
(`043e5a6`), merged `main` (`3aefbe9`), added masked greedy to the kernel and ran the gates. Protocol
and every number: `bench/grammar-protocol.md`.

## Items

1. **Host glue: DONE.** `serve/grammar_rt.mojo` (vocab/trie loaded on first schema request, schema to
   Matcher), `serve/serve_proto.mojo` request fields `schema`/`reasoning`, `window.mojo` fills the mask,
   uploads it, draws with `amar_sample_row_masked`, advances the matcher, stops on termination.
   `f6c3767`: the kernel's temperature <= 0 branch now applies the mask (masked argmax), so T=0 works.
   `2a96e4b`: spec, megakernel and megakernel window off for a grammar request.
2. **Reasoning boundary: DONE.** Mask starts after `</think>`; verified live (below).
3. **Speculation composition: NOT BUILT.** Grammar requests run with spec off. Per-row window masks and
   matcher rollback remain open, as does a masked megakernel head.
4. **spark.mojo 400: DONE.** Live Qwen2.5-7B spark server: `response_format` -> HTTP 400
   `response_format is not supported on this engine (no draft head / grammar wiring); dense and MoE
   only`, plain request -> 200.
5. **main.rs 400 removed for dense/MoE: DONE** (gated on the ready line's `kmax`).

## Gates

| gate | result |
|---|---|
| 1 corpus, 32 schemas x T=0 and T=0.7, reasoning off | **64/64 PASS** (json + jsonschema oracle) |
| 1 reasoning on (schema 01 T=0.7, schema 21 T=0) | **2/2 PASS**; 292 and 460 tokens generated, 11 masked draws each |
| 2 real HTTP body | PASS, below |
| 3 no response_format, 20 prompts, main vs lane build | **identity 20/20**, 151.16 vs 151.24 tok/s_gen (1.001), fail word 0 |
| 4 masked draws == accepted == completion_tokens | PASS on all 66 requests |
| 5 cost, as frozen (< 5%) | **FAIL: 1.253x** (8.371 vs 6.682 ms/token; plain arm used the megakernel) |
| 5 diagnostic, both arms `BARO_MEGA=0` | 1.013x (8.392 vs 8.287 ms/token): mask cost 1.3% |

`run-tests.sh` exit 0 (58 in registry, 0 orphans), `tools/ci-checks.sh` 0, `cargo test` 27/27,
`kernels/test_sample_device` PASS including the new T=0 mask checks on three real decode rows.

Gate 2 body (`POST /v1/chat/completions`, schema `21_weather_tool_call.json`, T=0, reasoning off):

```json
{"choices": [{"finish_reason": "stop", "index": 0, "logprobs": null, "message": {"content": "{\"location\":\"The North Pole\",\"unit\":\"celsius\"}", "role": "assistant"}}], "timings": {"accepted": null, "cached": 0, "decode_s": 0.19040655, "drafted": null, "finish": "stop", "prefill_rows": 102, "prefill_s": 0.134201856, "restore_s": 0.001079674, "tok_s_gen": 63.02304201194759}, "usage": {"baro": {"cached_tokens": 0, "prefill_rows": 102}, "completion_tokens": 13, "prompt_tokens": 103, "total_tokens": 116}}
```

Engine receipt for it: `grammar masked draws: 13 accepted: 13 terminated: True`.

## Open

- Gate 5 fails as frozen. A grammar request loses the megakernel (1.24x) and spec; the mask itself is
  1.3%. Next: item 3 (spec composition) or a masked megakernel head.
- Gate harness defect fixed mid-run: reasoning-on max_tokens 1024 exceeded TMAX 1088 (HTTP 400 before
  the engine); lowered to 800 and rerun (`bench/grammar-protocol.md`).
- My `window.mojo` T=0 routing edits landed inside merge commit `3aefbe9`, not in `2a96e4b`.
