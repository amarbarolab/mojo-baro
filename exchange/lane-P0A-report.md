# Team A P0a report

Date: 2026-09-17
Branch: `lane-team-a`
Status: preflight complete; live gates 1-3 UNVERIFIED

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
below are intentionally not claimed from that code-level check.

## Gates

| gate | result | reason or receipt |
|---|---|---|
| 1, PAIR parallel count and token parity | UNVERIFIED | Requires one resident GPU engine. |
| 2, Ollama Python client, 20 prompts, stream and non-stream | UNVERIFIED | Requires one resident GPU engine. |
| 3, PAIR engine-manager adoption | UNVERIFIED | Requires PAIR workload log naming the baro node. |
| 4, embeddings vector parity | DEFERRED | Coordinator owns P0a-e: request `embed:true`, one extra `{"id": ID, "embed": [H floats]}` line, last-prompt-token post-final-norm hidden state, engine-side L2 normalization. |

The required live invocation is:

```text
gpu-wait run --vram 24 --timeout 3600 -- \
  env BARO_ENGINE=$HOME/Projects/mojo/mojo-baro/.work/engine \
      BARO_PACK=$HOME/Projects/mojo/mojo-baro/.work/engine-pack-q4 \
      BARO_SERVE_BIN=.work/team-A/sonnet/target/release/baro-serve \
  bench/p0a-contract-gate.sh .work/team-A/codex/p0a 5 0
```

This was not launched bare. At report time `gpu-wait list` exits 1 because its
daemon socket is unavailable:
`$HOME/.local/run/gpu-waiting-room.sock`; it suggests starting the user
unit `gpu-waitd`. No GPU job was submitted by Team A.

## Kill lines and next action

Any gate 1 or 2 mismatch kills the P0a claim. Gate 3 failure must be reported
immediately while work continues. Do not start P1 from this report until the
coordinator restores the gpu-wait daemon and grants the live gate window, or
explicitly accepts gates 1-3 as UNVERIFIED. P0a-e remains coordinator-owned.

UNVERIFIED: live gates 1-3, PAIR engine-manager adoption, and embeddings gate 4.
