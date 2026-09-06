# Structural round: occupancy probe and LDS-staged activations (frozen 2026-09-06, before any change)

Follows `bench/q4-splitk-protocol.md` (closed). State: q4 pack default, m=1 megakernel 119-120 tok/s_gen no-spec
(`930e012`), 8.32 ms/token, GEMM phases 85% at 0.85x the q8 phases' GB/s, VGPR 256 (spills 77) at 512 threads,
G=96 = one block per CU = 16 wave64 waves... (8 waves/block; see persistent-kernel-gfx11).

Prior receipts that bear on this round (recalled first, CLAUDE.md 5.0): stage 0 measured G=192 worst of
{96,144,192} on q8row phases; the 192-VGPR cap spilled in the GEMM loop (1.09x slower than launches); W3 ruled
out G=192 / 192-cap for the m>1 window; G=192 has zero residency slack on the desktop (1 NOT-RESIDENT in 110).
All of that was the q8 path. The q4 loop carries half the bytes in flight per wave, so the occupancy question is
re-measured here, cheaply, before the structural change.

## Memory logic applied (from `inference-kernels` s3 and `persistent-kernel-gfx11`)

Every wave re-reads the whole activation vector A per row from L0/L1 (K=4096: 8 KB bf16; K=12288: 24 KB, which
does not fit L0) and unpacks it bf16->f32 per row: ffn down moves ~100 MB of A per phase against 28 MB of weights;
the wide GEMMs ~100 MB against 57 MB. Those reads sit in the same vector-memory queue as the weight stream and
the unpack is ~25% of the loop's VALU. Staging A once per block into LDS as f32 (K x 4 B: 16-48 KB; LDS today 6.4 KB
of 64) turns the per-row A reads into ds_read_b128 with no unpack and no VMEM slot. Values identical, so the
megakernel stays bit-identical to the launch path without touching it.

## Stages

| stage | what | check |
|---|---|---|
| P0 | probe: 192-VGPR cap (drop `rocdl.flat_work_group_size`) at G=96 and at G=192, q4 pack, no code change otherwise | ISA spills; single runs + one 20-prompt A/B vs HEAD for the better of the two; NOT-RESIDENT count over the run |
| P1 | LDS-staged A (f32) in the m=1 megakernel's seven GEMM phases (ssm in / out, attn qkv / o, ffn gate+up / down, head); the launch path unchanged | `test_mega_block` q4 0 mismatches; engine mega == launch 20/20; 64/64 vs model-ref; ISA: no `v_mov_b16`/shift unpack of A in the loops, `ds_read_b128` present, VGPR/spills |
| P2 | 20-prompt A/B HEAD vs P1, one stint, clock-probe; per-phase profile | median, spread, identity |

## Frozen predictions

P0: the 192 cap at G=96 loses (spills in the dot loop, as before); at G=192 the extra block per CU buys back
some of it: **-5% to +3%**, and any NOT-RESIDENT in 20 runs disqualifies it as a shipping config regardless.
P1+P2: **119.5 -> 126-132 (+5-10%)**: wide phases lose the A unpack (VALU -25%) and the A VMEM reads; ffn down and
the head gain most. Land >= 124 (+4%); close < 122. Stop rules: P1 any mismatch = a staging index bug, fix it,
never the check; P1 LDS > 56 KB = drop the f32 form for bf16 in LDS (unpack stays) and re-freeze the number.

## P0 result (2026-09-06, `results/mega-structural/p0-*.log`)

| binary | VGPR / spills | tok/s_gen (3 runs) | tokens | 20-prompt A/B vs HEAD |
|---|---|---|---|---|
| HEAD (256 cap, G=96) | 256 / 77 | 119-120 | 64/64 | 119.14 (spread 3.2%) |
| 192 cap, G=96 | 192 / 310 | 105.6-105.8 | 64/64 | **105.00 (8.5%), 0.881x** |
| 192 cap, G=192 | 192 / 310 | 61.6-68.1 | **wrong from token 14-25 (argmax 0)** | not run |

The cap alone costs 12% (spills 77 -> 310, in the dot loops); G=192 on top runs at half speed and stops
producing tokens mid-sequence with no host-visible fail line: consistent with the zero-slack residency
receipt (a stolen CU slot trips the bounded barrier; the kernel returns early; the host reads zeros). Occupancy
via the register cap is closed for the q4 path too, with numbers. The 1024-thread-block variant (32 waves per
CU in one block, no residency risk) would need the same 192 cap and therefore the same spills; not pursued.
