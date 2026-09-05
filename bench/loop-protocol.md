# Self-optimising loop protocol

Bound by `PROTOCOL-RULES.md`. The served model proposes rewrites of the engine
sources embedded in its own gguf; a mechanical gate decides; winners are
re-embedded into a NEW gguf with lineage. Claude reads survivors only.

## Tools

- `tools/gguf-closure.sh MODEL` -- build the engine from the gguf's own sources, gate 64/64. The floor.
- `BARO_PROFILE=1 .work/engine` -- per-sub-block GPU shares; picks the target region.
- `tools/loop-propose.py MODEL ITER --n N --start I` -- N identity framings, one diff each, into `.work/loop/ITER/`.
- `tools/loop-gate.sh ITER CHAMPION_TOKPS` -- scope -> compile -> identity@64 -> perf -> ISA. Receipt per candidate.
- `tools/loop-embed-winner.sh SRC.gguf ITER` -- embeds the COMMITTED repo sources into `<src>-loop-ITER.gguf`, adds `baro.kernel.parent`.

## Frozen acceptance rule (2026-09-01)

A candidate lands only if: touches only `baro.kernel.files`; compiles; 64/64
greedy tokens identical; median of 3 `tok/s_gen` read back from the engine's
own output >= champion + 2% with spread < 5%; no scratch and no spills in any
embedded code object. The candidate's own `PREDICT` line is its
preregistration and is recorded in the receipt. Server (port 8083) must be
stopped for stages 2-4; it is the proposer, not the instrument.

## GPU choreography

propose (server up) -> stop server by exact PID -> gate (engine owns GPU) ->
Claude commits winner -> embed -> restart server from
`~/Brain/mojo-baro/llama-server-cmdline.txt`, `curl /health`.

## Worth-it rule

After 5 iterations: < 1 accepted winner or < 2% aggregate tok/s gain =>
widen the region to whole-layer rewrites instead of more iterations.

## Receipts

| iter | region | identities | candidates | survivors | champion before -> after | gguf |
|---|---|---|---|---|---|---|
| 001 (2026-09-01, commit 7833260) | ffn (52%) | 18-skeptic, 19-builder, 20-stranger | 3 | 0 | 41.3 -> 41.3 | none |
| 002 (2026-09-05, commit 9b8a399 in gguf) | ffn (51.2%) | 01-father, 02-grandfather, 03-uncle, 04-mother | 4 | 0 | 67.48 -> 67.48 | none |

Iteration 001 notes: prompt 25k chars (bindings + ffn region + elementwise +
matmul_skinny). All three failed before any timed run: no diff fence; patch
does not apply (invented `g_ffn`, `Wfg`, `g_ffn_fused`); compiles against an
invented `w_h_ffn2` and silently dropped the up and down projections. The
ladder held. Proposer quality, not the gate, is the bottleneck: next iteration
gives the model a symbol table of real binding names and a smaller region.


## Iteration 002 notes (2026-09-05)

Two proposer defects fixed before the run, both silent:

- `slice_region` fed the model engine.mojo's `# --- kernel bindings` block,
  which has been **empty since the aliases moved to registry.mojo** (9b8a399).
  Every run since then showed the proposer zero binding names while telling it
  every symbol must already exist -- the mechanical cause of iteration 001's
  3/3 invented-symbol failures. Now a real 30-alias table, and the tool
  refuses to prompt if it finds none (`cf94ab6`).
- `baro.kernel.files` carries nested paths (`shim/CMakeLists.txt`); source
  materialisation crashed on the missing parent dir before any candidate was
  generated.

Result: 4/4 candidates produced a parseable diff with a PREDICT line (iter 001:
0/3 got that far). **All four then failed at `apply`.** Every one emitted the
same shape: a hunk header `@@ -721,7 +721,7 @@` declaring 7 context lines above
and below while supplying 4, on line numbers taken from the region slice's
absolute numbering. `patch` rejects it before any compile.

Two of the four also swapped `r_swiglu` for an undefined `r_swiglu_bf16` /
`r_swiglu_fused` without writing the kernel, so the symbol table did not stop
invention -- it only moved the failure from parse to apply. Those would have
died at compile regardless.

Next iteration's fix is in the gate, not the prompt: apply with fuzz and
context matching (`patch -l --fuzz=3`, or reconstruct the hunk from the
context lines) so that a correct edit with a miscounted header is judged on
its merits. Counting hunk lines is not the skill under test.

Worth-it rule status: 2 iterations, 0 survivors, 0% aggregate gain. Three more
iterations before the rule forces widening the region.
