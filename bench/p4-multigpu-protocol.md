# P4 multi-GPU protocol

Status: wiring preflight, timed gate pending. Frozen before any P4 timed GPU run.

## Scope

P4 tests process-level data parallel wiring, not cross-model quality or throughput. The discrete
arm uses the dense q4 fixture and the iGPU arm uses self-describing Qwen2.5-7B-Instruct BARO
`b4271cd`. Each arm is compared only with
its own solo reference. The iGPU is a functional harness device, not a performance arm.

The fixture manifest is `bench/p4-fixtures/manifest.tsv`. It records model and pack paths and
hashes, but does not copy model bytes. The 20 prompts are the existing
`bench/mtp-prompts/p*.tokens` set, whose sorted-file aggregate hash is recorded there.
The Qwen7 pack is extracted from the GGUF's embedded `engine-pack.py` and `--dense` metadata,
matching the `tools/baro serve` cache builder. The import receipt reports min 96.9%, mean 99.4%
on its 20-prompt gate and pack index hash `e2ef587f...`; the generated local pack must reproduce
that index hash before a GPU run.

## Effective-device preflight

Before any prompt is sent, the receipt must contain all of:

1. `igpu-env --probe` exit 0 and its printed `ROCR_VISIBLE_DEVICES`, unset
   `HIP_VISIBLE_DEVICES`, and `HSA_OVERRIDE_GFX_VERSION=10.3.0`.
2. `rocm-smi --showpids` for every running engine process, with PID-to-device mapping.
3. The iGPU process's engine log showing the expected HIP API path, plus external
   `rocminfo`/`rocm-smi` output naming the gfx1030 object and mapping the running PID to it.
   MAX's `DeviceContext.name()` is not accepted as the device identity: under this override it
   reports the host CPU name even while `ctx.api()` reports HIP. `ROCR_VISIBLE_DEVICES=1` alone
   is insufficient because MAX lists the Raphael device as gfx1030 objects under this override
   and does not list gfx1036. The misleading engine self-name is retained in the receipt.
4. The per-process environment read-back from the Rust spawn seam, including the server port,
   selector, and pack path. P4 uses two OS `baro-serve` processes, one per GPU; a pool inside
   one process remains same-device concurrency and is not the cross-device arm.

Any missing or contradictory read-back stops the gate before prompts.

## Gate 1: round-robin substitute

P0b is not merged, so the router-backed placement gate is `BLOCKED: P0b not shipped`. The
independent wiring gate uses a 20-line round-robin dispatcher. It sends each prompt to the engine
assigned by the dispatcher and compares the resulting token ids with that engine's solo arm using
the same engine, pack, prompt, seed, and T=0. It must report 20/20 per-process identity and the
engine assignment for every prompt. No comparison is made between the XTX and iGPU models.

The timed invocation is from the lane root, using an absolute script path and an explicit
runtime `PATH`. `gpu-wait` admits jobs with a clean environment and may not preserve the caller's
cwd, so the short relative form is not a valid submission receipt:

```text
cd $HOME/Projects/mojo/mojo-baro-lanes/team-b
$HOME/.local/bin/gpu-wait run --priority 20 --timeout 3600 --vram 22 -- env PATH=/opt/rocm/bin:$HOME/iTools/bin:/usr/bin:/bin $HOME/Projects/mojo/mojo-baro-lanes/team-b/.work/p4/run-two-engines.sh
```

The full harness is one serialized GPU job. The iGPU is included in the queue. No nested
`gpu-wait` is allowed inside the admitted job.

## Gate 2: state movement

P1 exists and is verified on `lane-team-a@e07b883`, including `/v1/state/export`,
`/v1/state/import`, and `GET /v1/state`, but is not merged into this lane or main. Gate 2 is
therefore `BLOCKED: P1 integration pending coordinator merge`, not `BLOCKED: feature missing`.
Once merged, the gate moves state XTX to iGPU and back and compares ids with the single-node arm.

## Kill line and receipts

Kill the item on any identity miss, wrong device pin, missing effective-parameter read-back,
wrong engine assignment, or stale/missing fixture hash. A router or state gate that lacks its
dependency remains explicitly blocked and cannot be replaced by a weaker claim.

The command, exit code, preflight, device mapping, per-prompt assignments, and verdicts land in
`.work/team-B/codex/p4/`. The item report is `exchange/lane-P4-report.md`. The CPU dry-run must
stop before the first GPU step and is recorded before the queued invocation.
