# MoE engine wiring protocol

This protocol records W0-W4 predictions before each step is built. Each step
gets its own preregistration commit and a result section after its gate.

## W0: model profile

### Question

Can the existing qwen35 engine move its scattered model constants into a
build-selected profile while preserving the default qwen35 behavior and
rejecting the megakernel for qwen35moe? Inputs, compiler, pack, prompt set,
and default runtime behavior remain fixed.

### Treatment

Add `serve/model_qwen35.mojo` and `serve/model_qwen35moe.mojo`, select with
`BARO_MODEL` at build time, and route registry, attention, and SSM dimensions
through the selected profile. No kernel arithmetic or default qwen35 model
weights change.

### Method

Build the default engine and the qwen35moe profile in the same lane. Run the
existing `./run-tests.sh`, the 20-prompt `bench/force-ab.sh` teacher-forced
gate between a main-built reference and the W0 build, and
`.work/kernel-census --check`. Record binary hashes, prompt agreement, census
status, and compile results. No speed claim is made in W0.

### Registered prediction

- Default qwen35: 20/20 prompts pass the teacher-forced A/B gate, with every
  prompt at 64/64, and `./run-tests.sh` exits 0.
- qwen35moe: profile compilation exits 0, with H=2048, QF=8192, KV=512,
  N_LAYERS=40, N_SSM=30, and N_ATT=10 visible in the build receipt.
- `BARO_MEGA=1` under qwen35moe is rejected with the registered clear error.
- `kernel-census --check` exits 0.

Confidence ordering: default qwen35 identity, qwen35moe compile, census,
then megakernel refusal.

### Scoring

W0 passes only if every listed gate passes. Any compile failure, A/B void,
teacher-forced mismatch, census failure, or silent megakernel acceptance fails
W0. No partial pass is reported as done.

### Failure meanings

- Default identity failure means profile routing changed qwen35 behavior.
- qwen35moe compile failure means the profile interface is incomplete or
  incompatible with current Mojo.
- Census failure means reachability or generated kernel documentation is stale.
- Megakernel acceptance means the safety refusal is not enforced.

### Outputs

Durable outputs are this protocol, the W0 report in
`exchange/lane-MOE-report.md`, and gate receipts under `.work/moe-w0/`.
No output is written to `/tmp`.

### Result

W0 passed on 2026-09-12. Preregistration commit: `f90e351`. The final code
commit is recorded in the lane report. Default engine build and qwen35moe
build both exited 0. The profile probe printed H 2048, QF 8192, KV 512,
N_LAYERS 40, N_SSM 30, N_ATT 10, and MEGA_ALLOWED False. The final
`./run-tests.sh` gate exited 0. Final `bench/force-ab.sh` used distinct
binary hashes and passed 20/20 prompts, 64/64 each, min 100.0%, mean 100.0%,
void none. The qwen35moe runtime refusal gate exited 1 as required and
reported `BARO_MEGA=1 is not supported by the qwen35moe model profile`.
`kernel-census --check` passed inside `run-tests.sh`.

Operational deviation: the first test attempt lacked the lane-local q4 pack
and failed before tests; a symlink to the existing main q4 pack was added in
`.work`, then the unchanged test gate was rerun successfully. The first
direct build also failed because it bypassed gpu-wait; all GPU builds and
runtime gates after that were run through gpu-wait.
