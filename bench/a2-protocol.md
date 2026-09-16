# A2 step 1: paged KV through a block table (preregistered 2026-09-17, before its build)

Plan `docs/NEXT-PLAN.md` A2. Lane `lane-A2`, fable (kernel scope). Binds `bench/PROTOCOL-RULES.md`.
Gate harness `bench/a2-gate.sh` (landed `652a86c`, verified: 60/60 at 100%, 91 s wall cached).

## What exists

The KV pool is already page-major with 128-token pages: `kv_off` (`kernels/attn.mojo`) and
`dkv_off` (`kernels/dattn.mojo`) compute `(((t >> 7) * NAT + att_i) * NKVH + kvh) * KVHSTR +
(t & 127) * HD`, so the physical page is the logical page. Every KV read and write in the dense engine
goes through one of those two functions (attn: append, append2, decode, prefill, prefill_wmma,
head_span; dattn: exact, split, load_span; mega token and window), plus the MoE megakernel and the
spark profile's attention. The pool is `ceil(TMAX / 128)` pages, one request at a time, so a block
table is an identity map today; its value is A3 (N requests share the pool) and B3 (pages leave VRAM).

## Change

- `kv_off` and `dkv_off` take a device table `tab: UnsafePointer[Int32]`; physical page =
  `tab[t >> 7]`. Page size stays **128, not the plan's 16**: `dattn_load_span` loads each 8-token span
  from one base address and walks it linearly, so spans must stay inside a page and the lookup is one
  load per span, not per token; 16-token pages would multiply the table by 8 and force a per-token
  lookup in every loop for a fragmentation benefit that does not exist at N <= 4 on one card.
- Every kernel that touches KV takes the table as one more pointer argument; every launch site passes
  `b.kvtab_d`. The MoE megakernel and the spark profile get the same argument and an identity table
  (one rule, no special cases). Kernel files stay comment-free.
- Host: a new `kvpage.mojo` under `serve/` (named this way until the lane lands, for the
  dangling-reference check), a `PageTable` (free list over the pool's physical pages, `alloc(n)`,
  `free`, `identity()`, `reverse()`, `upload(ctx)`), owned by the engine's window state; `kvtab_d`
  (`int32[tpages]`) in `WindowBufs`. `BARO_KVTAB=identity|reverse` selects the mapping at start-up
  and is echoed (P1). State save gathers logical pages through the table; state load resets the table
  to identity and copies as before. Prefix checkpoints are unchanged (KV stays in place; the table is
  per request and identity while there is one request).

## Predictions (frozen)

- P-A2a identity table: `bench/a2-gate.sh MAIN CAND` 60/60 at 100% (MIN_PCT 100), 59/60 restored,
  decode after 32k within 1% of the main engine's number from the same stint (main measured 101.56
  today: candidate >= 100.5).
- P-A2b **reverse table** (`BARO_KVTAB=reverse`, logical page p on physical page tpages-1-p): the same
  gate 60/60 at 100% against the main engine. This is the only prediction that proves the table is
  read: a kernel that ignores it would read the wrong physical page for every logical page but one and
  agreement would collapse past the first 128 tokens.
- P-A2c short context: `bench/ab-prompts.sh` main vs candidate, T=0, megakernel, spec off, 20
  prompts: candidate within the standing +-2% band of main (main 135.07 today).
- P-A2d prefill: the first request of each set in the a2 gate (the one that prefills 8k, then 8k more,
  then 16k more) within 2% of the main engine's `prefill_s` for the same request.
- P-A2e `run-tests.sh` exit 0 with `kernels/test_mega_block.mojo` bit-identical (tools/mega-gate.sh
  kernel step), `tools/ci-checks.sh` exit 0, the MoE engine and the spark engine build and pass their
  standing checks in run-tests with identity tables.
- P-A2f state round trip: `BARO_STATE_SAVE` under the reverse table, `BARO_STATE_LOAD` into an
  identity table, the continuation's forced agreement 64/64 on one prompt (instrument receipt, P4).

## Kill line

Any identity miss under the identity table is a bug and is fixed before anything else. Decode after
32k below 99% of main means the page lookup did not hoist out of the token loop: restructure the
dattn span base per page and re-run; below 97% after that, step 1 is not merged as written and the
report says why. Receipts on every timed run: engine sha256, `BARO_KVTAB:` echo, `TMAX:`, `kv pages`,
`bench/clock-probe.sh` line, arm parameters from the running engine.

## Result (2026-09-17, change `09a8cd9`, merged tree `d3a0bf0`, engine `367ab62323748a3a`)

Receipts `.work/a2/gates/`, `.work/a2/gates2/`, `.work/a2/gates/postmerge-quick/`, ISA `.work/a2/isa/`;
report `exchange/2026-09-17-A2-step1-report.md`. Preflight PASS before every job.
- P-A2a PASS: identity table 60/60 at 100.0%, 59/60 restored, decode after 32k **100.67** (main
  101.56, 99.1%, bar 100.5). `BARO_KVTAB: identity kv pages: 256`.
- P-A2b PASS: reverse table 60/60 at 100.0%, 59/60 restored, decode after 32k **100.71**. The
  kernels read the table.
- P-A2c PASS: main 136.95 vs paged 136.07 tok/s_gen, ratio 0.994, identity 20/20.
- P-A2d PASS: first-request prefills equal within one request's noise (8064 ids 3.06 s on both).
- P-A2e PASS with one pre-existing red: run-tests exit 0 (103 kernels, 0 orphans), ci-checks exit 0,
  MoE and spark engines build; `test_mega_block` m=1 arms bit-identical, its m=3 window-kernel parity
  arm FAILS identically on main's kernels (958/12288 residual mismatches, max 0.0078; G=192 residency
  probe aborts): pre-existing, behind `BARO_MEGA_WIN=0`, recorded on the board, not chased.
  `test_attn_block` compiles but has no fixture on any tree (needs `tools/attn-ref.py` dumps): not run.
- P-A2f PASS: save under reverse (pos 12, 1 page), load under identity, forced agreement 64/64.
- ISA: q4 token megakernel dual 124/79/79/59, 0 spills (unchanged); dattn split 165/0; prefill WMMA
  spills 28 -> 25.
- Post-merge (main's MoE stage 3 merged into the lane): quick gate 3/3 at 100.0%, decode after 32k
  101.05. Kill line not reached.
