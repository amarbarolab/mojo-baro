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

Pending W0 build and gates.
