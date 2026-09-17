# Team B P4 report

Status: wiring is implemented and the corrected two-process gate ran on both devices, but the
round-robin identity kill line fired on `p06-translate` at the iGPU. P4 is not ready to claim.

## Changes and fixtures

The per-process environment seam is in commits `02f899c` and `48d01d5`. Each `baro-serve` process
passes repeatable `--engine-env KEY=VALUE` settings to every engine in its own pool; `KEY=` removes
the inherited variable. Commit `6b688fc` logs each spawned engine child's PID so the device receipt
checks the process that actually uses the GPU, not the Rust parent.

The fixture and protocol commits are `e3cb77d`, `24403c6`, and `d2ffe58`:

- XTX: the existing dense q4 pack, index SHA
  `cd420c015201d47b1a198e1eb96cfe0407d48668d273d9a0cc2a5d5d283941bf`.
- iGPU: the self-describing Qwen2.5-7B-Instruct BARO GGUF, source SHA
  `a0fb4588c1baf9067aa280753b1a7c2e202c4a7cb763f3e8f5206d5d91edd191`, packed with the embedded
  `engine-pack.py --dense`; index SHA
  `e2ef587fa49be96a8d95d2715038398b8a45757428fb0f80c7469e9be2471084`, matching its import receipt.
- The invalid Qwen2.5-0.5B partial pack was not used and remains quarantined at
  `.work/team-B/codex/p4/qwen05b-invalid-partial`.

The manifest is `bench/p4-fixtures/manifest.tsv`; the protocol is
`bench/p4-multigpu-protocol.md`; the gate body is `.work/p4/run-two-engines.sh`.

## CPU preflight and queue hardening

Three failures were found before trusting a GPU result:

1. `mu593fyff5of` exited 1 after 1.029 seconds with an empty job environment. The script's bare
   commands therefore had no usable `PATH`; no GPU was used.
2. `mu59gh5zshgz` exited 1 after about 1.1 seconds after the PATH fix. `rocm-smi --showuniqueid`
   returned a bare `0x859baafa301986cb`, not a `GPU-*` string. Selector discovery now uses the
   gfx1100 `Uuid: GPU-859baafa301986cb` from `rocminfo` and retains the rocm-smi suffix as a
   cross-check.
3. A clean direct preflight initially passed while omitting the variables injected by gpu-wait.
   The exact injected environment then reproduced the iGPU failure: `HSA_OVERRIDE_GFX_VERSION=11.0.0`,
   `HIP_VISIBLE_DEVICES=0`, and `ROCR_VISIBLE_DEVICES=0` hide or misidentify the iGPU. The gate now
   unsets those defaults for topology discovery and reapplies explicit values per arm.

The final preflight used the same injected variables and the same PATH as the queue:

```text
env PATH=/opt/rocm/bin:$HOME/iTools/bin:/usr/bin:/bin HOME=$HOME \
  HSA_OVERRIDE_GFX_VERSION=11.0.0 HIP_VISIBLE_DEVICES=0 ROCR_VISIBLE_DEVICES=0 \
  P4_CPU_PREFLIGHT=1 $HOME/Projects/mojo/mojo-baro-lanes/team-b/.work/p4/run-two-engines.sh
```

It exited 0 before any engine launch. Receipt was posted in room B at line 368. Read-back was:

```text
selector.xtx_rocr=GPU-859baafa301986cb
selector.xtx_rocm_smi_unique_id=0x859baafa301986cb
selector.igpu_rocr=1
igpu-env probe: vadd mismatches 0 of 1024
igpu rocminfo: gfx1030 agent present
XTX rocminfo: gfx1100, AMD Radeon RX 7900 XTX
PASS p4 CPU preflight: selectors, device readbacks, and fixture hashes verified
```

CPU checks also passed: `bash -n` exit 0, injected-variable dry run exit 77 at the first GPU
step, and `gate-dryrun` exit 0.

## Timed gate

The final submission used an absolute script path and explicit PATH inside `gpu-wait`:

```text
gpu-wait run --priority 20 --timeout 3600 --vram 22 -- env \
  PATH=/opt/rocm/bin:$HOME/iTools/bin:/usr/bin:/bin \
  $HOME/Projects/mojo/mojo-baro-lanes/team-b/.work/p4/run-two-engines.sh
```

Job `mu5a1vmll4y3` was terminal `failed`, exit 1, cause `exit`, wall 327.6 seconds, peak VRAM
21.97 GiB. This was a real gate run, not an infrastructure or environment failure. Both services
became healthy, both child PIDs appeared in `rocm-smi`, and the logs read back the intended engine
environment:

- XTX child PID `909623`, `ROCR_VISIBLE_DEVICES=GPU-859baafa301986cb`.
- iGPU child PID `909530`, `ROCR_VISIBLE_DEVICES=1`, `HSA_OVERRIDE_GFX_VERSION=10.3.0`, and
  `HIP_VISIBLE_DEVICES` unset.

The gate receipt is `.work/team-B/codex/p4/gate/round-robin.csv`; device and server receipts are
in the same directory. Results were:

```text
p01-water,xtx,PASS
p02-python-fib,igpu,PASS
p03-story,xtx,PASS
p04-list-planets,igpu,PASS
p05-math,xtx,PASS
p06-translate,igpu,FAIL
```

For `p06-translate`, the iGPU solo and split streams matched for the first six token IDs
(`220 16 15 15 15 15`) and then diverged. The solo stream degenerated into repeated token 15,
while the later split request produced varied tokens. Both requests used the same long-lived iGPU
process, prompt, temperature 0, and `spec=false`. This is evidence of request-state bleeding or
another engine/kernel-level determinism defect, not a device-pinning or queue failure. The full
token receipts are `.work/team-B/codex/p4/gate/p06-translate.igpu-solo.tokens` and
`.work/team-B/codex/p4/gate/p06-translate.split.tokens`.

The router-backed placement gate was blocked until P0b. The state movement gate was not run while
P1 was pending; after the timed gate ended, the requested merge brought P1 into this lane. No
claim is made for either blocked/unrun gate.

## Merge and next action

Per coordinator instruction, `main` was merged only after `mu5a1vmll4y3` reached terminal state.
Merge commit: `57c5e8a`, with main parent `c0e9297`. The lane is clean after the merge.

P4 remains failed on the iGPU determinism kill line. The next investigation belongs in the engine
request-reset/state path, outside this gate harness's owned files; do not replace the Qwen7 arm or
weaken the identity check.
