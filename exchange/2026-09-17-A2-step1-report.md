# A2 step 1: paged KV through a block table (2026-09-17, lane-A2, fable)

Protocol `bench/a2-protocol.md` (frozen `bc78ad2` before the build). Change `09a8cd9`, preflight
`d80dd53efbf1`. Reference engine: the A6 lane's final binary `7237e557c9a357a9` (main's kernels before
this change). Candidate `.work/a2/engine-a2`. Receipts `.work/a2/gates/`, `.work/a2/gates2/`, `.work/a2/isa/`.

## What changed

`kv_tab_off` / `dkv_tab_off` read the physical page of logical page `t >> 7` from a device table;
every kernel that reads or writes KV takes the table (attention append, append2, decode, prefill,
prefill_wmma, head_span; split decode exact, split, load_span; both dense megakernels; the MoE
megakernel; the spark profile's append and gated decode) and every launch passes it. Kernel files stay
comment-free. Host: `serve/kvpage.mojo` (`PageTable`: identity, reverse, alloc, release, upload),
`kvtab_d` and `kvtab_h` in `WindowBufs` (identity at allocation), `BARO_KVTAB=identity|reverse`
echoed with the page count, state save gathers logical pages through the table, state load resets
the table to identity. Page size stays 128 (the split kernel loads each 8-token span from one base
address, so spans must stay inside a page; 16-token pages would multiply the table by 8 and force a
per-token lookup for no fragmentation benefit at N <= 4). Two bench sources that launch the
megakernel directly (`bench_hidden_dtype`, `bench_latent_handoff`) pass the table too.

## ISA gate (CPU, before any GPU run)

`tools/isa-receipt.py` + `isa-loops` on the candidate: q4 m=1 token megakernel 256 VGPR, 0 spills,
dot-loop fingerprint unchanged (dual 124 / 79 / 79 / 59, the champion's); split decode 165 VGPR, 0
spills (unchanged); decode 38 / 0 (unchanged); prefill WMMA 192 VGPR, spills 28 -> 25; q8 window
47 -> 48, q4 window 44 -> 42. No lottery re-roll.

Addendum (isa-diff, built the same day): the q4 token megakernel's fingerprint is SAME; the split
decode kernel's loop fingerprint CHANGED (four loops of dual 183/183/134/135 before, two of 134/135
after, same 165 VGPR and 0 spills): the per-span table read restructured its load loops. The measured
cost is the 0.9% at 32k above; there is no lottery rule for that kernel, so this is recorded, not
gated.

## Gates (`bench/a2-gate.sh`, both engines resident, `BARO_TMAX=32768`, refcache on the reference)

| arm | agreement | restored | decode after 32k | receipt |
|---|---|---|---|---|
| P-A2a identity table | 60/60 at 100.0% (8k, 16k, 32k) | 59/60 | **100.67** tok/s | `BARO_KVTAB: identity kv pages: 256` |
| P-A2b **reverse table** | 60/60 at 100.0% | 59/60 | **100.71** tok/s | `BARO_KVTAB: reverse kv pages: 256` |

Main's same-harness number is 101.56 (yesterday's smoke on the reference engine): 99.1% and 99.2%,
inside the frozen 1% band (bar 100.5). The reverse arm is the one that proves the kernels read the
table: with logical page p on physical page 255-p, an unread table would read the wrong page for
every logical page but one.

P-A2c short context, `bench/ab-prompts.sh` main vs candidate, T=0, megakernel, spec off, 20
prompts, one stint under `bench/clock-probe.sh`: main 136.95, paged 136.07 tok/s_gen, ratio
**0.994**, identity 20/20 (band +-2%). sclk median 2897 MHz, 290 W cap.

P-A2d prefill: the identity arm's first request per set prefilled 8064 ids in 3.06 s, then 8192
more in 0.11 s of restore plus prefill, then 16384 more; equal to the reference's own run within the
noise of one request (both engines' per-request `prefill_s` in `.work/a2/gates/id/`).

P-A2e tests: `run-tests.sh` exit 0 (103 kernels, 58 in registry, 0 orphans); `tools/ci-checks.sh`
exit 0 inside the passing preflight; the MoE and spark engines build with identity tables.
`kernels/test_mega_block.mojo`: the m=1 arms PASS bit-identical (q4 and q8); **the m=3 window-kernel
parity arm FAILS on this build AND on a build from main's kernels with identical numbers** (residual
958/12288 mismatches, max 0.0078; norm rows 847, max 4.8e-7; the G=192 residency probe aborts on
both, fail word 1). Pre-existing on main, not this change: `amar_mega_window[MR=3]` has drifted from
the bit-identity W3 recorded, behind `BARO_MEGA_WIN` which defaults off (A6.4 measured it at 0.761x
with identical tokens). Left open on the board; not chased in this lane. `kernels/test_attn_block.mojo`
compiles with the table but needs a numpy-oracle fixture (`tools/attn-ref.py` dumps) that no tree
here has; not part of any standing gate, not run.

P-A2f state round trip: `BARO_STATE_SAVE` under `BARO_KVTAB=reverse` (pos 12, 1 page, format
BAROST01), `BARO_STATE_LOAD` under `identity`, continuation teacher-forced against the reverse run's
64 ids: **forced agreement 64/64**.

## Kill line

Not reached: identity held on every arm; decode after 32k 100.67 and 100.71 against 101.56 (bar
100.5). The extra load per KV access costs under 1% at 32k and nothing measurable at short context.

## Left for step 2 and A3

- Int8 KV (KIVI axes) on the same table, `MIN_PCT` set from a known-good lossy configuration (P14).
- `PageTable.alloc/release` exist and are untested by any gate: A3 is the first caller; the prefix
  checkpoint chain restores KV in place and assumes one request's identity table.
- The window megakernel's m=3 parity drift (above) and the G=192 residency probe on a desktop GPU.

## Post-merge receipt

main (with the MoE stage 3 merge) merged into the lane at `d3a0bf0`; engine rebuilt
(`367ab62323748a3a`), preflight PASS `c277c167e5ca`, `QUICK=3` identity gate 3/3 at 100.0%,
decode after 32k 101.05 tok/s.

## GPU budget

Lane jobs: 4 (one died on the preflight check after a commit, the stamp-ordering bug now fixed in
`bench/preflight.sh`; one stopped at the fixture-less attention test, the rest of that gate re-run).
