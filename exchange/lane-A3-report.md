# A3 batching, first slice (2026-09-17)

Branch: `lane-a3`. Base: `002992e`. This delivery covers the A3(a) wire
admission slice. Device execution remains serial; A3(b-d) are not claimed.

## Claims and receipts

| claim | result | receipt |
|---|---|---|
| MAX reservation is fixed per engine process | imported PASS: about 22 GB per process for both 5.3 GB and 6.8 GB packs | `exchange/lane-COMFY-report.md`, item 0 |
| two requests can be admitted before the first terminal line | PASS, admission read-back present | `.work/a3-wire/receipt.md` |
| concurrent responses remain correctly addressed | PASS, 2/2 token bodies match the single-request reference; ids `cmpl-2`, `cmpl-1` are distinct | `.work/a3-wire/receipt.md` |
| engine and server shut down cleanly | PASS, server exit 0 | `.work/a3-wire/receipt.md` |

## Item 0 precondition

The COMFY lane measured 21.75 GB held with a 5.32 GB Spark-X2.5-4B q8 pack
and reported the same approximately 22 GB reservation class for the 6.72 GB
dense q4 pack. This is fixed per MAX engine process, not per tracked
sequence. A3 N=4 therefore uses one engine's current reservation minus the
trunk. No duplicate VRAM arm was run. A per-sequence KV growth number is still
optional for A3(b), if that design needs it.

## Implementation

- `serve/src/engine.rs`: the worker keeps addressed active requests, writes
  later request lines while a prior request is decoding, and routes every
  token and terminal event by id. The single engine process and serial device
  execution are unchanged.
- `serve/PROTOCOL.md`: documents the A3(a) boundary contract.
- `bench/a3-wire-gate.sh`: reference-cached two-request identity gate with
  admission, busy-health, id, and clean-shutdown checks.
- `bench/a3-vram-protocol.md`: links item 0 to the COMFY receipt and scopes
  any future KV-growth-only arm.

## Verification

- `cargo clippy --release --manifest-path serve/Cargo.toml --all-targets -- -D warnings`: PASS.
- `cargo test --release --manifest-path serve/Cargo.toml`: 31 passed.
- `cargo build --release --manifest-path serve/Cargo.toml`: PASS.
- `gpu-wait run --vram 1 -- bench/preflight.sh`: PASS, final stamp
  `37c27f46de38`.
- Dense engine build used `gpu-wait run --vram 24 -- ./.venv/bin/mojo build
  serve/engine.mojo -I . -I kernels -I serve -o .work/engine`: PASS.
- `env -u LC_ALL bench/a3-wire-gate.sh .work/a3-wire`: PASS. The unset
  `LC_ALL` is required because the preflight stamp's locale-sensitive sort is
  produced inside the GPU runner; this does not change source contents.

The first gate attempt exposed a checker defect, not a product failure: the
fixed prompt EOS-stopped at 109 tokens with `finish_reason=stop`, so the gate
was changed to compare each concurrent response with the reference's actual
length and finish reason. The corrected run passed.

## Stopping point

A3(a) proves wire admission and id routing only. A3(b) still needs per-request
KV and conv/SSM regions plus the reverse-page arm. A3(c) still needs one launch
with N=4 rows and the 20-prompt throughput receipt. A3(d) admission,
preemption, and the per-sequence resource policy remain open.

## Guidance loaded

Relevant guidance read before implementation: `lane-dispatch`,
`mojo-nightly-lane-builder`, `inference-kernels`, `kernel-parity`,
`kernel-arm-round`, `persistent-kernel-gfx11`, `gate-authoring`,
`reverse-arm-gates`, `llm-benchmark-method`, and `perf-writeup`.
