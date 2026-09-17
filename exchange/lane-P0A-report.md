# Team A P0a report

Date: 2026-09-17
Branch: `lane-team-a`
Status: gates 1-3 PASS; embeddings gate 4 deferred to coordinator P0a-e wire

## Landed

- `30d4410` adds the Ollama request types, metadata routes, chat/generate
  translation, and NDJSON streaming on the existing request path.
- `6d07a86` briefly changed the embeddings pending response to a literal JSON
  shape; coordinator clarification says the repo-wide `ApiError::Plain` shape
  is the contract.
- `d61c2fe` restores the generic 501 error shape while retaining
  `embeddings_pending` and `P0a-e` in its message. Embeddings remain deferred
  until the coordinator-owned engine wire is gated.
- `978de8d` adds the contract gate and the Ollama protocol note. It is the only
  Codex-owned commit in this report.
- `46f4a01` corrects gate 1 to compare `raw: true` Ollama generation with the
  raw OpenAI completions prompt; default Ollama generation is chat-templated and
  remains covered by gate 2.
- `9572069` adds the PAIR engine-manager adoption and routed-request check.
- `b7f521f` changes gate 3 to explicit `engine:start` adoption with absolute
  isolated PAIR user paths, then routes through `engine:action`.

## Checks and receipts

The following checks have verifiable receipts:

```text
bash -n bench/p0a-contract-gate.sh
exit 0

GATE_DRYRUN_OUT=.work/team-A/codex/p0a \
  ~/iTools/bin/gate-dryrun \
  --arm .work/team-A/codex/p0a/arm.txt \
  --expect backend=ollama --expect count=5 --expect seed=0 \
  --stop 'FAIL.*GPU' -- \
  bench/p0a-contract-gate.sh .work/team-A/codex/p0a 5 0
PASS gate-dryrun bench/p0a-contract-gate.sh: reached its first GPU step in 0s
(rc=97); FAIL GPU: GATE_DRYRUN stops before baro-serve launch
exit 0
```

Receipt files are `.work/team-A/codex/p0a/arm.txt`,
`.work/team-A/codex/p0a/contract.log`, and
`.work/team-A/codex/p0a/p0a-contract-gate.sh.log`. The arm file records
`backend=ollama`, `count=5`, `seed=0`, `temperature=0`, and deferred
embeddings.

Sonnet reported release cargo build, clippy with `-D warnings`, and nextest
`36/36` passing for the Rust source commit. The live HTTP and PAIR receipts
below are from the separate gpu-wait receipt, not that code-level check.

## Gates

| gate | result | reason or receipt |
|---|---|---|
| 1, PAIR parallel count and token parity | PASS | `p0a-live-r4/contract.log`: PAIR 5/5 and token counts 5/5 equal. |
| 2, Ollama Python client, 20 prompts, stream and non-stream | PASS | `p0a-live-r4/contract.log`: 20/20 chat/generate stream and non-stream equal. |
| 3, PAIR engine-manager adoption | PASS | `p0a-live-r4/pair-manager.json`: running and healthy on 11434, routed response done; stderr names external adoption. |
| 4, embeddings vector parity | DEFERRED | Coordinator owns P0a-e: request `embed:true`, one extra `{"id": ID, "embed": [H floats]}` line, last-prompt-token post-final-norm hidden state, engine-side L2 normalization. |

The required live invocation is:

```text
gpu-wait run --vram 24 --timeout 3600 -- \
  env BARO_ENGINE=$HOME/Projects/mojo/mojo-baro/.work/engine \
      BARO_PACK=$HOME/Projects/mojo/mojo-baro/.work/engine-pack-q4 \
      BARO_SERVE_BIN=.work/team-A/sonnet/target/release/baro-serve \
      PAIR_MANAGER_BIN=$HOME/Projects/imports/Personal-AI-Router/services/build/bin/nvpair-engine-manager \
  bench/p0a-contract-gate.sh .work/team-A/codex/p0a-live-r4 5 11434
```

Receipt: gpu-wait job `mu51onyvoq8l`, exit code 0. The socket fix required
`GPUWR_SOCKET=/run/user/1000/gpu-waiting-room.sock`. No bare GPU job was
submitted by Team A.

The first live attempt `p0a-live` correctly killed on a gate-script comparison
bug: default `/api/generate` applied the chat template while `/v1/completions`
used a raw prompt (8 versus 3 completion tokens). The corrected r2 passed gates
1 and 2. The r3 gate3 probe used `engine:get-installed`, which timed out during
PAIR's unrelated LM Studio sweep after Ollama adoption; its manager stderr and
state notification still prove adoption. The r4 gate3 uses explicit
`engine:start` and passed the adoption plus routed-request check.

## Kill lines and next action

Any gate 1 or 2 mismatch kills the P0a claim. Gate 3 failure must be reported
immediately while work continues. P0a gates 1-3 are complete. Sonnet may build
the Rust P0a-e side after the coordinator merges the passed engine-wire lane.
Do not claim embeddings parity until that side is built and gated. P0a-e
remains coordinator-owned.

UNVERIFIED: embeddings gate 4 on this Team A branch.
