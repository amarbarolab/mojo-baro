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

## A3(b) resident-state result

Protocol was frozen before the live run in
`bench/a3-b-protocol.md`. The item 0 VRAM precondition remains inherited from
`exchange/lane-COMFY-report.md`, item 0: MAX holds about 22 GB per engine
process for both the 5.3 GB and 6.8 GB packs. It is process-fixed, not
sequence-scaled, so the A3 N=4 budget is one current engine reservation minus
the trunk. No duplicate VRAM measurement was run.

Implementation gives two resident sequence slots in one process. Each slot
has private token history, KV page-table region, physical KV pages, convolution
state, and SSM state. `serve/kvpage.mojo` now allocates from the full physical
page pool, and `serve/harness.mojo` exposes per-sequence buffer views. The
normal single-request path remains on plane 0 of the expanded token buffer.

## A3(b) verification

- `bench/a3-b-gate.sh`: frozen identity and reverse-page arms, each using 20
  cyclic pairs from `.work/a2-prompts/L8192` with `n=64`, `temperature=0`,
  `BARO_SPEC=0`, `BARO_CKPT=0`, and `BARO_MEGA=0`.
- Dry-run: PASS, stopped at `FAIL GPU step: dry-run reached engine startup`
  before reference or resident arms.
- Engine rebuilt after merge and source changes through
  `gpu-wait run --vram 24`; `.work/engine` sha256 is
  `ccc041dd06a778b0b42f30202785424a2a39524fd7503774ad8777d710a2a6ba`.
- `gpu-wait run --vram 1 -- bench/preflight.sh`: PASS, final stamp
  `97d97ebfb680`.
- Live receipt: `.work/a3-b/receipt.md`, refcache key
  `2482e1c1766c1aea19e486c6`.
- Identity mapping: 20/20 A and 20/20 B, 40/40 terminal responses.
- Reverse mapping: 20/20 A and 20/20 B, 40/40 terminal responses; 20
  non-identity mapping receipts.
- Every pair printed distinct slots `0,1`, resident boundary receipts were
  present, and both arms exited cleanly with status 0.

This proves resident KV, convolution, SSM, token, and page-table isolation for
the serial two-sequence greedy path. It makes no A3(c) launch-batching or
throughput claim. A3(c) and A3(d) remain open.
