# JSON `response_format` enforcement

Preregistered 2026-09-16, before any engine.mojo/window.mojo edit, at commit
`b7969eb` (`~/Brain/mojo-baro/briefs/2026-09-16-json-enforcement-lane.md`).
Binds `bench/PROTOCOL-RULES.md`.

## What changes

The dense/MoE engine (`serve/engine.mojo`, qwen35 + qwen35moe) gains real
`response_format: {type: json_schema}` enforcement. `serve/spark.mojo` (the
other dense families) does not change: `main.rs` keeps its 400 for it, gated
on the running engine's family (existing `kmax` field in the `ready` line --
`serve/spark.mojo`'s is hardcoded `0`, `serve/engine.mojo`'s is a real
compile constant > 0 for both qwen35 profiles, so no new wire field is
needed to tell them apart).

Per generated token: the host fills a `Bitset` from the grammar's current
state (`grammar.matcher.Matcher.fill_mask`, ~28-30us measured,
`docs/grammar.md` item 5e), uploads it to the device (no host sync -- same
shape as any other per-step device buffer write), draws with
`amar_sample_row_masked` (landed `b7cecb1`, one mask per window row, draft
head unmasked), advances the matcher with the chosen id
(`Matcher.accept`), and stops the request (`finish` reason `"grammar"`, a
new value alongside `"length"`/`"stop"`/`"cancelled"`) once
`Matcher.is_terminated()`.

Reasoning models: the matcher is not attached (no `fill_mask`/`accept`
calls) until the reasoning boundary (`</think>`) is observed in the
generated token stream, matching `grammar/test_reasoning_boundary.mojo`'s
own convention -- the matcher starts life already positioned at the
boundary, reasoning tokens are never fed to it.

Speculation: the verify window snapshots the matcher
(`Matcher.snapshot`) before filling the k+1 row masks, fills one mask per
row from the matcher's state as of that row's position (row `i`'s mask
assumes the previous `i` draft tokens were accepted -- the matcher is
walked forward by `accept()`-ing each draft id in turn to build row `i`'s
mask, not truly advanced), and after the real accept count is known from
the verify step, `rollback`s to the snapshot and `accept()`s only the
tokens the window actually kept. A forbidden draft token is already
handled for free by the masked kernel: `amar_sample_row_masked`'s row mask
zeros its target probability, so `amar_spec_accept`'s `min(1, p/q)` is 0
and the draft is rejected by the existing accept rule -- no separate
grammar-side veto is needed on the accept path itself, only on which
tokens get compared.

## P1 read-back, before any timed run

- engine sha256, built in the same command as the run.
- the run's own echo: `spec`, `temperature`, `top_p`, `top_k`, `min_p`,
  `seed`, `prompt tokens`, `tokens`, plus the new masked-draw counter (see
  gate 4).
- for spec rows, `drafted` / `accepted` / `k` from the done line.
- power cap, vddgfx offset, sclk from `bench/clock-probe.sh`.

## Gates, frozen

1. **Corpus validity.** Every output over the 32-schema corpus
   (`grammar/corpus/`) parses as JSON and validates against its own schema,
   default spec on, at T=0 and T=0.7. A real prompt per schema (the
   schema's own `description`/property names paraphrased into a one-line
   ask), generation capped at the schema's compiled `maxLength`/`maxItems`
   defaults times a safety factor so a non-terminating grammar can't hang
   the gate.
2. **Real HTTP round trip.** One live `POST /v1/chat/completions` with a
   `response_format` body against a running `baro-serve`, response body
   pasted into the report, parses and validates.
3. **T=0 unaffected without the field.** 20 prompts (`bench/ab-prompts.sh`
   shape), no `response_format` in the request, forced identity against the
   champion engine build -- this round must not touch the no-grammar path's
   output.
4. **Mask-applied receipt.** The running engine prints a count of masked
   draws for a constrained request; that count must equal the number of
   constrained tokens generated (every generated token once
   `response_format` is set and, for reasoning models, the boundary has
   passed). Read on every run -- a silent NOT-RESIDENT-style skip would
   otherwise look identical to correct enforcement (`bench/PROTOCOL-RULES.md`
   P1, and CLAUDE.md's "gate the candidate can write is not a gate": the
   mask-fill and the accept/reject decision are on opposite sides of the
   host/device boundary, so this receipt is the only thing that proves the
   two stayed in sync).
5. **tok/s cost.** 20-prompt medians, with and without `response_format`, at
   T=0, no-spec and spec-on, same stint, cold-cache rotation per
   `bench/coldcache-protocol.md`.

## Prediction, frozen before wiring

`fill_mask` measured ~28-30us/call at steady state (`docs/grammar.md`). A
no-spec decode step is host-bound by the existing per-token round trip
(prompt/token JSON line + device launch); at champion 136.37 tok/s_gen
(7.33 ms/token) an added 30us is 0.4% -- **predicted no-spec cost < 1%,
gate 5 fails this round if the measured cost exceeds 5%** (the same +5%
line used elsewhere in this repo's protocols, e.g.
`bench/moe-persist-protocol.md` R6.2). Spec mode pays the fill k+1 times
per window instead of once; at k=2 that is 3x30us = 90us against a window
that already amortizes over up to 3 tokens, so the per-token cost is the
same order -- same <5% prediction, not separately re-derived.

## Amendment 2026-09-16, before any gate run (lane taken over after the 14:33 OOM kill)

Scope as built, frozen here before the gates run:

- **Temperature.** `amar_sample_row_masked` now threads the mask through its
  temperature <= 0 branch (masked argmax, -1 on an empty mask, `f6c3767`), so
  gate 1 runs at T=0 and T=0.7 as written.
- **Speculation (item 3) is NOT built this round.** A grammar request runs
  with spec, the megakernel and the megakernel window forced off, exactly like
  a penalties request (`2a96e4b`). Server default spec stays on for every
  other request. Gate 1's "default spec on" therefore means: the server runs
  with its default spec setting and the engine turns spec off per grammar
  request; no per-row window masks or matcher rollback are claimed.
- **Truncation.** Grammar requests force top_p=1, top_k=0, min_p=0 (the mask
  is applied after truncation in the kernel).
- **Gate 1 prompts.** One chat request per schema: "Reply with one JSON value
  that matches this JSON schema, filled with realistic data: <schema>",
  `enable_thinking: false`, max_tokens 400, seed 7. Plus two reasoning-on
  requests (schemas 01 at T=0.7 and 21 at T=0, max_tokens 1024) for item 2.
  Oracle: Python `json` + `jsonschema` (`bench/grammar-gate.py`). A request
  that hits max_tokens before the document closes counts as a FAIL.
- **Gate 4** is read per request from the engine log: masked draws ==
  accepted, and masked draws == completion_tokens (reasoning off) or
  0 < masked draws <= completion_tokens (reasoning on).
- **Gate 3** runs `tools/test_server.sh` (64/64 T=0 against
  `ref-tokens-64`) plus the 20-prompt forced identity of the new engine build
  against the champion build, no `response_format` in any request.
- **Gate 5 prediction unchanged:** no-spec cost < 1% per token, fail above
  5%. The spec-on comparison is replaced by the cost a grammar request pays
  for losing spec: predicted equal to the spec speedup itself (~1.1x at k=2),
  reported, not gated.

## Status

Gates not yet run at the amendment commit. `serve/serve_proto.mojo`'s `parse_schema_field` (request-line
schema slice) and `serve/test_serve_proto.mojo` land ahead of the
engine.mojo/window.mojo change, which needs the coordinator's go
(`briefs/2026-09-16-json-enforcement-lane.md` file-ownership rule:
`serve/window.mojo`/`serve/engine.mojo` owned by the sampling lane until it
hands off).

## Result

**Gate 1 run 1** (`8ed860a` + `run-tests` fix): 64/64 reasoning-off requests PASS (32 schemas at
T=0 and T=0.7, valid JSON, receipt masked draws == accepted == completion_tokens). The two
reasoning-on requests returned HTTP 400 before reaching the engine: prompt + max_tokens 1024
exceeds TMAX 1088. That is a defect in the gate script, not in the enforcement; reasoning-on
max_tokens lowered to 800 and those two cases rerun alone (`GATE_THINK_ONLY=1`).

**Gate 1 reasoning-on rerun:** 2/2 PASS (schema 01 T=0.7: 292 tokens, 11 masked draws; schema 21
T=0: 460 tokens, 11 masked draws; the think tokens were unmasked, the JSON after `</think>`
validated). Gate 1 total 66/66.

**Gate 3:** `bench/ab-prompts.sh` main build (`engine-main`, sha 230fb7bf) vs lane build (sha
e768a477), server defaults (spec on, megakernel on), power cap 290 W, vddgfx -100 mV: identity 20/20
PASS, medians 151.16 vs 151.24 tok/s_gen (ratio 1.001), fail word 0 on every run.

**Gate 5, as frozen: FAIL.** `bench/grammar-cost.py`, 20 schemas, equal decode length (all
same_len, no drafts in either arm): grammar 8.371 ms/token (119.46 tok/s) vs plain 6.682 ms/token
(149.65 tok/s), **cost 1.253x** against the < 5% line. The prediction missed: the plain arm runs the
megakernel (T=0, no penalties), the grammar arm is forced onto the launch path, so the ratio mixes
the megakernel's own advantage with the mask cost. Diagnostic (not a gate): same script against a
server started with `BARO_MEGA=0`, isolating the host mask fill and upload.

**Gate 5 diagnostic** (server `BARO_MEGA=0`, read back in the engine log as `BARO_MEGA: False`),
same script: grammar 8.392 ms/token vs plain 8.287 ms/token, **1.013x**. The mask fill, upload and
matcher advance cost 1.3% per token, inside the < 1% to 5% band the prediction was about; the
remaining 1.24x of the frozen comparison is the megakernel, which a grammar request cannot use (its
token launch bakes an unmasked argmax). Closing the gap needs a masked megakernel head or spec
composition (item 3), neither built this round.

**Gate 5 closed 2026-09-16 (A6.3, `39f6a4c`, `bench/chat-protocol.md` A6.3).** The grammar request now
runs the megakernel layers with `fold_head = 0` and the masked draw after the launch-path head, no
masked megakernel head needed. Same script, live `baro-serve` on engine `e55a56aa954f49e4`
(`BARO_MEGA: True`): grammar 6.751 ms/token vs plain 6.685, **cost 1.010x** (was 1.253x), under the
< 5% line; gate 1 re-run on the same server 66/66 valid with masked draws == accepted
(`.work/a6/gates4/`). The 1.013x diagnostic (mask fill, upload, matcher advance) is the whole cost now.
