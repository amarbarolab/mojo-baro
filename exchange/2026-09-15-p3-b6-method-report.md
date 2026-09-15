# B6 report: docs/METHOD.md, the two required checks, real output

Lane: w82:p3, reporting to w82:p4. Deliverable: `docs/METHOD.md`, committed
at `cd25815` on `main`. This file is the s18 evidence the brief asked for,
not necessarily copied into `docs/METHOD.md` itself.

## Check 1: `bench/carryover-stamp.py`, no GPU

Command:

```
bench/carryover-stamp.py
```

Real output:

```
wrote $HOME/Projects/mojo/mojo-baro/.work/carryover/src-C
wrote $HOME/Projects/mojo/mojo-baro/.work/carryover/src-D2
```

Verified it produced what section 4.1 of `docs/METHOD.md` claims, not just
that it exited zero:

```
$ grep -c "amar_matmul_skinny_q4rowb_st" .work/carryover/src-C/kernels/matmul_skinny.mojo
1
$ grep -c "gemm_w_st" .work/carryover/src-C/serve/window.mojo
14
$ diff -rq .work/carryover/src-C .work/carryover/src-D2
Files .work/carryover/src-C/kernels/ssm.mojo and .work/carryover/src-D2/kernels/ssm.mojo differ
Files .work/carryover/src-C/serve/registry.mojo and .work/carryover/src-D2/serve/registry.mojo differ
```

The stamped kernel variant exists, the 13 documented call sites (plus the
one `def gemm_w_st[` itself, 14 total) route through it, and arm D2 differs
from arm C in exactly the two files the tool's own docstring says it
swaps, nothing else. No GPU time.

## Check 2: `tools/gguf-verify.sh`, under gpu-wait

Command:

```
$HOME/.local/bin/gpu-wait run --vram 22 -- \
  tools/gguf-verify.sh $HOME/Models/RegesCore-1.0-35/RegesCore-1.0-35B-UD-Q4_K_S-BARO-8184f7d.gguf .work/method-verify
```

Real output (stdout, verbatim):

```
file:      $HOME/Models/RegesCore-1.0-35/RegesCore-1.0-35B-UD-Q4_K_S-BARO-8184f7d.gguf
commit:    8184f7d   files: 6
expects:   card=AMD Radeon RX 7900 XTX (gfx1100, 96 CU, 24 GB) driver=amdgpu 7.2.4-arch1-2 rocm=7.2.4 power_cap_w=290 tok_s_gen_20p=93.46 config=moe-w1 pack, BARO_SPEC=0, BARO_MEGA=0, 20-prompt median protocol=bench/moe-perf-protocol.md 2026-09-15
this card: gfx=gfx1100 name=? amdgpu=7.2.4-arch1-2 rocm=7.2.4 power_cap_w=290
commit: 8184f7d  arch: gfx1100  files: attn.mojo,dattn.mojo,elementwise.mojo,matmul.mojo,matmul_prefill.mojo,matmul_prefill_lds.mojo,matmul_skinny.mojo,matmul_ternary.mojo,mega.mojo,moe.mojo,sample.mojo,ssm.mojo,model.mojo,model_qwen35.mojo,model_qwen35moe.mojo,registry.mojo,serve_proto.mojo,window.mojo,latentos/__init__.mojo,latentos/agent.mojo,latentos/ipc.mojo,latentos/proto.mojo,latentos/sys.mojo
closure engine FAILED
rebuilt engine tok/s_gen on this card (one prompt, a receipt not a bar): n/a   embedded 20-prompt median: 93.46
closure exit: 1
```

Exit code: **1**. `gpu-wait` itself reported `job mu2a0kqs56wd failed: exit
code 1`.

Full closure log (`.work/method-verify.closure.log`), the part stdout
alone did not show:

```
loading pack: 21005191680 bytes
pack loaded in 4.298720012 s
BARO_DRAFT_Q4: False
BARO_FR: False k 0
BARO_DOT: False
pack q4 trunk: False
BARO_MEGA: False
BARO_MEGA_WIN: False
BARO_SERVE: False
spec k: 2
att split: 1088
BARO_SPEC: False
checkpoints: cap 0 , bytes 65.86368 MB each, period 1024
prompt file: .work/moe-w3/one.tokens
Unhandled exception caught during execution: Failed to open file '.work/moe-w3/one.tokens': No such file or directory
```

**This is a finding, not a blocker, per the brief's own instruction.**
Cause, confirmed by reading `tools/gguf-closure.sh`: its qwen35moe branch
defaults `BARO_PROMPT` to `.work/moe-w3/one.tokens`
(`gpu-wait run` line 52 of that script), a path under gitignored `.work/`
that the closure never creates and the gguf never carries in its
`baro.kernel.src.*` keys. It is leftover local state from an earlier
session, not part of the self-describing file. `docs/BASELINE.md` claims
the bake "rebuilds the harness with no path outside the file"; for the
qwen35moe model class that claim does not hold as of this commit. The
closure otherwise worked correctly up to that point: it extracted 6 source
files plus the 5 vendored `latentos/*.mojo` files from the gguf's own
metadata, built the engine, and loaded a 21 GB pack in 4.3 s before failing
on the missing prompt file, so the closure/build/embed mechanism itself is
sound; only the qwen35moe harness's default prompt path is the gap.

**What I changed in `docs/METHOD.md` because of this:** added a "Finding,
this document (2026-09-15)" paragraph to section 4.2, immediately after
the exit-code description, stating the exact failure, its cause, and that
the dense (Qwythos) closure branch does not share this default (untested
here, noted as such rather than assumed). I did not mark the command
UNVERIFIED, because it was run, not skipped; I reported what it actually
does today, which is fail on this model class. I did not touch
`tools/gguf-closure.sh` (no new tooling, out of scope for this lane).

## GPU minutes used

Under 1 minute. The job (`mu2a0kqs56wd`) built the closure, loaded a 21 GB
pack (4.3 s), and failed on the missing prompt file before any decode
started; `gpu-wait list` was empty before the launch and the queue was not
contended. `bench/carryover-stamp.py`'s own check used no GPU time.

## Files touched

- `docs/METHOD.md`: new file, committed at `cd25815`.
- `.work/carryover/src-C`, `.work/carryover/src-D2`: check 1's artifacts,
  gitignored.
- `.work/method-verify/`, `.work/method-verify.closure.log`: check 2's
  artifacts, gitignored.
- Nothing under `kernels/*.mojo` touched. `kernels/sample.mojo`,
  `serve/engine.mojo`, `serve/sample_ref.mojo` were not staged, not
  stashed, not committed; they showed no uncommitted changes by the time
  I checked (the coordinator's working state, not mine to report on).
- `tools/ci-checks.sh`: green at the commit (`all non-GPU checks passed`,
  exit 0), run before committing.
