# Lane MOE-LOADER report

Branch `lane-MOE-LOADER`, worktree `mojo-baro-lanes/MOE-LOADER`, rebased onto
`9dfcb45` (lane-MOE W1) at `a8fdc21` mid-session (see commit history: the
worktree originally branched from main `6496229`, before lane-MOE's W0/W1
landed, and had no model profile to read; the coordinator rebased it).

## Status: DONE and gated, for the loader/profile-sizing half of W3 only

Nothing here wires MoE decode or touches a MoE kernel. Scope stayed inside
`serve/harness.mojo` and a new `serve/moe_pack.mojo`, per the brief's file
ownership split.

## Commits

- `bb1c149` wip(moe-loader): pack index resolver and expert-offset semantics
  (written before the rebase, against invented placeholder constants; the
  coordinator confirmed it survives unchanged as the arithmetic core).
- `98cbb49` bench: preregister MoE loader probe (post-rebase: wired to the
  real `serve/model_qwen35moe.mojo`, protocol frozen before running).
- `9632b70` feat(harness): qwen35moe pack dtypes plus a pack/profile mismatch
  guard (the loader dtype support the brief asked for, plus a robustness
  guard the maintainer asked for mid-session after reproducing a SIGILL).

## What changed

**`serve/moe_pack.mojo` (new, host-only, no GPU imports).** Name-keyed
tensor resolution over the real pack's `index.txt`: `TensorInfo` (dtype,
offset, n_elem) in a `Dict[String, TensorInfo]`, never a positional list.
`tensor_byte_size(dtype, n_elem)` for `f32`/`q8_0`/`q4_k`/`q6_k`/`q8`.
`resolve_expert(tensors, layer, proj, is_shared, expert, row_start,
row_count, out_dim, in_dim) -> ExpertLoc` looks the tensor up **by name**
(`blk.<layer>.ffn_<proj>_exps.weight` or `..._shexp.weight`), then computes
the per-expert byte offset from `out_dim`/`in_dim` and the tensor's own
dtype-driven superblock geometry -- expert `e`'s block starts at
`tensor.offset + e * out_dim * row_bytes` (row-major, expert is the
slowest-varying axis, confirmed against `tools/moe-ref.py`'s own convention
and against the real pack's recorded offsets). `resolve_plain` for whole
non-expert tensors (router, norms). `size_host_state()` sizes KV pool and
SSM state slots from `serve/model_qwen35moe.mojo`'s own H/QF/KV/N_LAYERS/
N_SSM/N_ATT/HD/NKVH -- no second copy of those numbers. `N_EXP`/`TOPK` are
declared locally (routing facts, not in the model profile module).

**`tools/moe-loader-probe.mojo` (new, host-only).** Opens the real
`.work/moe-w1/pack`, resolves all 733 tensors, sums resolved bytes against
the real `pack.bin` size, prints `blk.0`'s router/shared/expert-0 offsets,
checks expert 255 of each routed projection lands exactly on its tensor's
end, checks the three Q6_K `ffn_down_exps.weight` layers, and prints the
profile-sized host state. `MOE LOADER PROBE: PASS` or `FAIL` with the
specific mismatch.

**`serve/harness.mojo` (existing, yours to build on).** Two changes to
`load_pack`, the shared entry point for every pack format:

1. New dtype branches: `q8_0` (`(n // 32) * 34` bytes), `q4_k` (`(n // 256) *
   144`), `q6_k` (`(n // 256) * 210`) alongside the existing `bf16`/`f32`/
   `q8`/`q4`/`i32`. The qwen35moe pack now loads through the exact same
   `load_pack` the q4/q8 dense packs use. Verified byte-identical: existing
   q4 pack regression below.
2. A pack/profile mismatch guard, added mid-session at the maintainer's request after
   he reproduced a real crash (see "Robustness finding" below). Raises
   before any offset is used if `token_embd.weight`'s element count implies
   a different H than the selected profile's `H`, or if the pack's real
   transformer layer count (see algorithm below) differs from `N_LAYERS`.

## What the wiring lane can rely on today

- `load_pack(ctx, packdir)` in `serve/harness.mojo` loads the real moe pack
  (`.work/moe-w1/pack`, or any future qwen35moe pack in the same format)
  when the engine is built `-D BARO_MODEL=qwen35moe`, exactly like it loads
  the q4/q8 dense packs today. `Pack.off` is still the old positional list
  (kept for the dense path's existing consumers); **the moe pack's own
  offsets are NOT in `Pack.off` in a name-addressable way** -- use
  `serve/moe_pack.mojo`'s `parse_moe_index("<packdir>/index.txt")` for a
  name-keyed `Dict[String, TensorInfo]`, and `resolve_expert`/`resolve_plain`
  to get a byte offset for a specific layer/projection/expert. Byte offsets
  from `moe_pack.mojo` are into the SAME `pack.bin` blob `load_pack` loads
  into `Pack.wbuf` -- add the resolved offset to `wbuf`'s base pointer, the
  same pattern `window.mojo` already uses for the dense `off` list.
- `resolve_expert`'s signature: `(tensors, layer: Int, proj: String
  ("gate"|"up"|"down"), is_shared: Bool, expert: Int, row_start: Int,
  row_count: Int, out_dim: Int, in_dim: Int) raises -> ExpertLoc` with fields
  `byte_offset, block_elems, block_bytes, row_blocks, row_bytes, n_rows`.
  `out_dim`/`in_dim` per projection: gate/up are `(P.FFN, P.H)`, down is
  `(P.H, P.FFN)`, same for routed and shared (both use `P.FFN = 512`).
- `size_host_state(tmax=1088)` returns `kvpool, kvpool1, conv_slot, ssm_slot,
  slots` sized from the qwen35moe profile -- for reference only. **The real
  engine's own `serve/registry.mojo` already computes the same `KVPOOL`,
  `KVPOOL1`, `CONV_SLOT`, `SSM_SLOT` correctly under `-D
  BARO_MODEL=qwen35moe`** (W0 routed `N_ATT`/`NKVH`/`N_SSM` through
  `serve/model_qwen35moe.mojo` already), so `serve/harness.mojo`'s
  `alloc_bufs` -- unchanged by this lane -- already sizes KV/SSM state
  correctly for the moe profile. `moe_pack.mojo`'s own copy is a
  self-contained, GPU-free demonstration/gate, not a second source of truth
  the real engine reads from.
- The guard: if you build a moe-profile engine and point it at the wrong
  pack, you get a raised `Error` naming the expected/found H or layer count,
  not a crash. Do not remove it without a replacement; see below for why it
  exists.

## Gate: `bench/moe-loader-protocol.md`

Preregistered before running (independent Python oracle read of the real
`index.txt` first, frozen into the protocol, then the Mojo probe run and
compared). Full numbers are in that file's Result section; summary:

```
tensors resolved: 733
resolved bytes: 21005191680 pack bytes: 21005191680
blk.0 router offset: 329850880
blk.0 shared gate/up/down offsets: 331956224 / 484065280 / 177741824
blk.0 expert0 gate/up/down offsets: 178855936 / 333070336 / 26746880
blk.34/38/39 down_exps dtype: q6_k (expert0 offsets match tensor base)
profile H 2048 QF 8192 KV 512 N_LAYERS 40 N_SSM 30 N_ATT 10 HD 256 NKVH 2
host state kvpool 5898240 kvpool1 589824 conv_slot 737280 ssm_slot 15728640 slots 9
MOE LOADER PROBE: PASS
```

Regression (both through `gpu-wait`, whiteboard head re-read before each
launch, no hold recorded): `./run-tests.sh` exit 0. `bench/force-ab.sh`
between a main-built engine (`38478b34aa4aaf4d`) and this branch's
(`bd850620005e0e8c`), distinct hashes: 20/20 prompts, 64/64 each, min 100.0%,
mean 100.0%, void none -- the existing qwen35 path is provably unaffected.

## Robustness finding (the maintainer's addition, in scope, gated the same way)

the maintainer reproduced, in this worktree: a moe-profile-built engine
(`-D BARO_MODEL=qwen35moe`) pointed at the qwen35 q4 pack (`BARO_PACK=
.work/engine-pack-q4`) crashed with an out-of-bounds abort, exit -4
(`window.mojo:830: Assert Error: index 442 is out of bounds, valid range is
0 to 441`) -- the moe profile's dims read the wrong pack's offsets and the
engine executed on garbage. Separately, before this lane's dtype work, any
engine pointed at the real moe pack raised `unknown pack dtype q8_0` from
`load_pack` (a clean error, not a crash, but still a hard stop this lane's
step 1 was already meant to close).

**Before/after, reproduced on this branch:**

| scenario | before (harness.mojo pre-guard) | after (this commit) |
|---|---|---|
| moe profile + q4 pack | exit -4, `window.mojo:830` bounds abort, no message naming the cause | exit 1, `pack/profile mismatch: profile H 2048, pack .work/engine-pack-q4 implies H 4096 from token_embd.weight` |
| default profile + moe pack | `unknown pack dtype q8_0` (already a clean error, pre-existing) | exit 1, `pack/profile mismatch: profile H 4096, pack .work/moe-w1/pack implies H 2048 from token_embd.weight` |
| default profile + q4 pack (the real, shipped path) | works | still works: real generated tokens, `force-ab` 20/20 above |
| moe profile + moe pack | `unknown pack dtype q8_0` | loads cleanly (21,005,191,680 bytes, matches exactly), runs to completion. **`GENERATED` is all zeros -- this is a pre-wiring observation, not a working engine.** No MoE kernel is wired into `serve/window.mojo`/`serve/engine.mojo` yet (out of this lane's scope by design), so the loaded weights are never read by a matching kernel; the loader doing its job correctly is all this row demonstrates. |

**The guard, in `load_pack`:** reads `token_embd.weight`'s element count /
`VOCAB` as the pack's implied `H`, and computes the pack's real transformer
layer count by grouping index entries by `blk.N` and excluding a WHOLE group
if ANY of its tensors is a `nextn.*` one. This exclusion rule matters: the
qwen35 q4 pack's optional MTP draft head appends a full extra `blk.32.*`
group containing 11 ordinary-looking tensor names (`attn_norm`, `attn_q`,
`attn_k`, `attn_v`, `attn_q_norm`, `attn_k_norm`, `attn_output`,
`post_attention_norm`, `ffn_gate`, `ffn_up`, `ffn_down`) plus only 4
`nextn.*` ones -- filtering out individual `nextn.*` tensors and counting
everything else still finds 33 distinct `blk.N` groups against a profile
`N_LAYERS` of 32, a false positive that broke the real q4 pack during this
same session (caught before commit by `run-tests.sh` failing red; never
shipped). The corrected rule (exclude the whole group) gives 32 for the q4
pack and 40 for the moe pack, matching both profiles exactly -- confirmed by
scenario 1 and 2 in the table above both loading and running to completion.

Both directions of the guard, both pack shapes loading correctly, and the
full regression are captured together in `bench/moe-loader-protocol.md`'s
Result section and the commit body of `9632b70`.

## Skills read

`mojo-syntax` (this session; Mojo's current syntax, Dict/Tuple/String
patterns, `.copy()` requirement for `Dict.__getitem__` on a Copyable-only
value).

## Deviations, disclosed

- The worktree originally branched from main before lane-MOE's W0/W1 landed
  (a dispatch error, not this lane's), so the first ~30 minutes of work
  (commit `bb1c149`) used invented placeholder profile constants. The
  coordinator caught it, had the WIP committed to lose nothing, rebased the
  branch onto lane-MOE's W1, and this lane re-read the real
  `serve/model_qwen35moe.mojo` and re-checked `serve/harness.mojo` before
  continuing, per instruction.
- The pack/profile guard's layer-count check was wrong on the first attempt
  in this session (per-tensor-name `nextn` filtering, not whole-group
  exclusion) and briefly broke `run-tests.sh` locally. Caught by
  `run-tests.sh` itself before any commit; the exact correct rule was
  supplied by the maintainer mid-session and is the one shipped.
