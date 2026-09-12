NOT "make mega.mojo faster". That is how a round gets spent tuning a phase
that was never the cost.

Round 1 is a MEASUREMENT: produce a per-phase cost breakdown of
`amar_mega_token` on the REAL weight pack, at m=1, and name the single
largest addressable cost with a number. Then stop and report. Do not
optimise in the same round.

Use the device timestamps the kernel already supports: block 0 thread 0
stamps `llvm_intrinsic["llvm.amdgcn.s.sendmsg.rtn", Int64](Int32(131))`
(REALTIME, 100 MHz) after each grid barrier into the prof buffer. The
phases are already barrier-separated, so the breakdown is a matter of
reading stamps, not of adding instrumentation to the hot path.

Deliverable: a table of us/token per phase, summing to the measured
per-token cost, with the residual named. An unexplained residual larger
than the phase you want to attack means the breakdown is wrong.

## Constraints that decide whether the round counts

- Every GPU run through `gpu-wait run`. The desktop shares this card.
- Read the `mega fail word` on EVERY run. A grid barrier timeout is silent:
  you still get tokens, they are just invalid.
- Ship-config is `G = 96`. Do not launch above the occupancy-derived
  ceiling; probe upward from below with the bounded barrier, never
  downward from a guess. A 240-block probe hung the gfx ring, forced a
  MODE1 reset, lost VRAM, killed every GL client and needed a reboot.
- Judge on the REAL pack, in one stint. The register allocator responds to
  the whole kernel, not the phase you touched: a delta-scan rewrite
  measured 17 -> 10 us on a synthetic 4-layer pack and LOST 1.5% on the
  real pack, because VGPRs shifted 256 -> 234 and an untouched FFN phase
  picked up 180 us of worse scheduling. Spill count is not a fitness
  function.
- Any pack under ~96 MB lives in Infinity Cache and inflates the number.
- A decode number is a 20-prompt median (`bench/PROTOCOL-RULES.md` P4), not
  one prompt. Measured today: one prompt read 1.012x on a change whose
  20-prompt median was 1.104x.
- Identity gate on any kernel change: teacher-forced agreement
  (`bench/force-ab.sh`, `BARO_FORCE`), 20 prompts at 64/64. Never greedy
  equality past ~256 ids. Pin `BARO_SPEC=0`: spec decode is ON by default
  since 2026-09-12 and the engine raises if it is combined with BARO_FORCE.
- Preregister the prediction BY COMMIT before the timed run. A frozen
  prediction with no numbers against it is the most perishable state in the
  repo.

## Dead ends: do not spend the round re-testing these

- `rocdl.waves_per_eu` is not reachable from Mojo. Six spellings tried, all
  rejected. Check upstream changed before trying again.
- `@no_inline` pins to force a stable schedule made things WORSE twice
  (7739 vs 7358 us/token; delta-scan variant 8035 us). A device call fixes
  the callee's code, not the caller's allocation.
- m>1 is not a persistent-kernel win here. The inner loop is
  instruction-identical to the native q8row, occupancy is not the lever
  (G=192 and the 192 VGPR cap both measured), the cost is per-row
  serialisation once the loop is FMA-bound, and cross-row prefetch is
  register-infeasible. Multi-row windows stay on launches.
- G=192 is worse than G=96 and has zero slack: kwin or a browser steals a
  slot, NOT-RESIDENT once in 110 launches.
- Split-K has OPPOSITE signs depending on what you have. It won 3.3-3.7x on
  attention, which was occupancy-starved at 16 of 96 blocks. It LOST 9% on
  an already block-strided 96-block GEMM phase. Establish which case a
  phase is in before reaching for it.
- The 48 KB LDS variant (+1.7%, 127.58 vs 125.43) is REJECTED as a shipping
  config, not un-measured: a grid-barrier kernel at 55.6 KB LDS can be
  evicted by a compositor shader on a desktop.

## What a good report looks like

The per-phase table, the arm file (engine shas both sides), the fail word
for every run, and ONE named target with its measured cost and why it is
addressable. If the breakdown says the largest cost is a bandwidth floor
(the head is 1.02 GB/token = 1.2 ms, already at floor), say so and
recommend NOT optimising it. A round that ends "nothing here is worth
taking" is a successful round and has happened before in this repo
(Round C failed its own floor and was not merged).
