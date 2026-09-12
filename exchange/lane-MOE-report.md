# Lane MOE report

## Skills read

- `$HOME/.agents/skills/mojo-syntax/SKILL.md`
- `$HOME/.agents/skills/mojo-gpu-fundamentals/SKILL.md`
- `$HOME/Brain/Skills/preregister-experiment/SKILL.md`
- `$HOME/Brain/Skills/llm-benchmark-method/SKILL.md`

The listed Django, Laravel, LatentOS, and perf skills were checked. They are
not relevant to this Mojo engine lane. The listed gguf-kv, gguf-sweep,
isa-loops, publish-purge, and rocprof-kernels iTools were checked and were not
present. No substitute was created.

## W0: model profile

Preregistration: `f90e351`.

Changes: added build-selected qwen35 and qwen35moe profile modules; routed
registry, attention, and megakernel dimensions through the selected profile;
added the qwen35moe megakernel refusal; added the standalone attention test
include path.

Gate results:

- Default engine build: exit 0, final binary `.work/moe-w0/engine-default`.
- qwen35moe engine build: exit 0, final binary `.work/moe-w0/engine-moe`.
- qwen35moe profile probe: H 2048, QF 8192, KV 512, N_LAYERS 40, N_SSM 30,
  N_ATT 10, MEGA_ALLOWED False.
- `./run-tests.sh`: exit 0, including prefix, sampler, attention parity, and
  `kernel-census --check`.
- `bench/force-ab.sh`: 20/20 prompts, every prompt 64/64, min 100.0%, mean
  100.0%, no voids. Arm hashes: ref `38478b34aa4aaf4d`, candidate
  `ad49d4c2e9176c89`.
- qwen35moe `BARO_MEGA=1` refusal: exit 1 with the registered error.

W0 gate: PASS.

Commits: preregistration `f90e351`; implementation `a879cdf`.

Operational deviations: the first test attempt failed because the lane had no
q4 pack. The existing main q4 pack was referenced through a lane-local `.work`
symlink and the gate was rerun. One direct build bypassed gpu-wait and failed
GPU target detection; subsequent GPU builds and runtime gates used gpu-wait.

## W1: qwen35moe pack

Preregistration: `22e9b82`.

Changes: added `tools/engine-pack.py --arch qwen35moe` with lexical tensor
order, raw Q4_K and Q8_0 block retention, F32 retention, and Q6_K output-head
requantisation. Added `tools/test_moe_pack.py`.

Gate results:

- 733/733 tensors represented.
- Raw-copy verification: 732/732 non-head tensors exact.
- Q6_K output head: 0 values over bound.
- Pack size: 21,005,191,680 bytes, 19.56 GiB.
- `./run-tests.sh`: exit 0.
- `bench/force-ab.sh`: 20/20 prompts at 64/64, min and mean 100.0%, no
  voids. Arm hashes: `38478b34aa4aaf4d` and `a81f2c583ae876e5`.

W1 gate: PASS.

Commit: `9dfcb45`.

## W2

Preregistration commit: `3bd08d6`.

Changes: added raw GGUF Q4_K block decode and routed gate/up/down expert
kernels, plus real-weight decoder, parity, and rotating-cache benchmark
coverage. `tools/moe-ref.py --gguf` now emits raw expert tensors, a decoder
vector, and a tensor-to-pack map for W3. `serve/harness.mojo` was not edited.

Gate results:

- Real layer 0: decoder max relative error `0.0`, exact 8/8 expert ids, q4
  routed `8.789e-8`, q4 y `1.333e-4`.
- Real layer 3: decoder max relative error `0.0`, exact 8/8 expert ids, q4
  routed `1.483e-6`, q4 y `1.598e-6`.
- Extended `test_moe_block`: exit 0. `./run-tests.sh`: exit 0.
- Rotating eight-arm timing: bf16 `90.27 us/token/layer`, q4k
  `213.43 us/token/layer`; q4k traffic `14.16 MB/token/layer`.

W2 correctness gate: PASS. The registered q4k performance prediction of 55
to 100 us/token/layer was falsified by the first implementation; timing was
registered as a measurement, not a hard acceptance threshold.

Operational deviation: inherited W0 engine A/B binaries were not a valid
source-only W2 regression target. The qwen35moe candidate rejected its
compiled mega setting, then faulted with illegal instruction under
`BARO_MEGA=0` before producing agreement lines. Dedicated W2 kernel gates
and the full GPU regression remained passing.

Receipts: `.work/moe-w2/kernel-gate-layer0.txt`,
`.work/moe-w2/kernel-gate-layer3.txt`, and `.work/moe-w2/timing.txt`.

Implementation commit: `b5293b9`.

## W3

W3 acceptance criteria preregistered in lane commit `2a6353e`: named-loader
resolution of all 733 tensors, one routed-plus-shared real block at max error
`5e-3`, then 20-prompt 64/64 teacher-forced decode and one non-empty Rust
front response.

Gate 1 is PASS at implementation commit `142d2c9`. The standalone gate loads
the real `21,005,191,680` byte pack through `load_pack`, resolves the named
router and routed/shared expert tensors through `parse_moe_index` and
`resolve_expert`, and reads every weight from the same `Pack.wbuf` blob. Real
layers 0 through 3 each selected 8/8 expert ids exactly and passed the `5e-3`
bound for routed, shared, and summed output. `./run-tests.sh` also exited 0
with 91 kernels, 41 registry entries, and 0 orphans.

Gate 1 receipt: `.work/moe-w3/gate1.txt`.

Gate 2 remains UNVERIFIED at implementation commit `9e5cc60`. The fix builds
qwen35moe's attention and SSM records as exact 16/19-entry named maps, uses
named terminal norm/head offsets, preserves dense positional offsets, and
exits the non-mega request path before the no-head NextN receipt. The qwen
build exits 0. `./run-tests.sh` reaches `92 kernels, 42 in registry, 0
orphans` but has one unrelated pre-existing host sampler failure (`real decode
row`). After removing the stale MoE-written draft fixtures, `test_sample_ref`
passes all five checks (`PASS: host reference sampler`) and `./run-tests.sh`
reaches `92 kernels, 42 in registry, 0 orphans`. The corrected llama reference
extractor strips `timings.prompt_n` and requests 128 combined tokens for a
64-token continuation. Gate 2 then fails at p01 `1/64`; the candidate is
non-zero and the reference is not being mistaken for a clean run. Receipt:
`.work/moe-w3/gate2-force-final2/`; candidate build:
`.work/moe-engine-positional-fix3`.

Layer-0 diagnostic remains open. An opt-in four-slot capture was built from
the working tree (post-SSM residual, post-FFN norm, post-MoE residual,
post-final norm) as `.work/moe-w3/diag-layer0.f32`; the fresh diagnostic binary
hash is `7cc046c841c9b7c462580ef59cf08de515b334171f1229062fdb84fa222859f5`.
A follow-up diagnostic correction is committed as `4d255da`. The prior
four-slot dump was invalid because ordinary per-layer writes overwrote custom
slots 1-3; the corrected capture suppresses those writes. It also fixed two
real MoE-path defects: router/shared gating consumed the unnormalized
residual, and normalized, routed, shared, and residual values aliased
`p_h_d`. A fresh qwen35moe build hash
`56d7fba8f27c0a683a45dc72b824bdccee0dbeda67a68d4fa7b58a4e0752e696`
produces p01 `32/64` forced agreement (`mega fail word: 0`, `42.48
tok/s_gen`). Gate 2 remains UNVERIFIED pending the full 20-prompt gate and
Rust-front request.

The full teacher-forced runner was launched against this committed binary
and stopped at p01 because its required `64/64` assertion failed at `32/64`;
no 20/20 claim is made. Receipt: `.work/moe-w3/gate2-force-4d255da/`.

## W4

Pending W3 and preregistration.
