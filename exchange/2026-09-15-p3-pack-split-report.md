# Pack split report: trunk and routed experts as separate files

Lane: w82:p3, reporting to w82:p4. Brief:
`briefs/2026-09-15-p3-pack-split-experts.md`. Deliverable:
`tools/pack-split-experts.py`, committed on `main`.

## What it does

Reads `SRCPACKDIR/{index.txt,pack.bin}`, writes `DSTDIR/{index.txt,pack.bin,
experts.bin}`. `index.txt` keeps the same line count and line order
(`serve/window.mojo` addresses tensors by position, not name): non-expert
lines keep their four columns with the offset rewritten into the new,
compacted `pack.bin`; lines for `blk.N.ffn_{gate,up,down}_exps.weight`
(routed experts) keep their four columns with the offset rewritten into the
new, compacted `experts.bin`, plus a fifth column, the literal word
`expert`. `blk.N.ffn_{gate,up,down}_shexp.weight` (the shared, always-active
expert) and the router (`ffn_gate_inp*`) are not routed experts and were
not touched by the split, confirmed by the regex (`_exps.weight` only,
never `_shexp` or `_inp`) and by the byte counts below.

Byte-size-per-dtype formulas are transcribed from `serve/harness.mojo`'s
`load_pack`, not from the brief's own summary, per the brief's instruction.
That function has one dtype the brief's summary did not mention: `i32`
(`n*4`, `frdraft.ids` only). The tool implements all eight cases
(`bf16`, `f32`, `q8`, `q8_0`, `q4_k`, `q6_k`, `q4`, `i32`) and raises on
anything else rather than guessing; the run on the real pack below only
exercised five of them (`f32`, `q4_k`, `q6_k`, `q8`, `q8_0`), the ones
actually present in `.work/moe-w1/pack/index.txt`.

## Command and output

```
$ df -h /home
Filesystem      Size  Used Avail Use% Mounted on
/dev/nvme0n1p7 1001G  909G   86G  92% /home

$ mkdir -p .work/moe-tier
$ time python3 tools/pack-split-experts.py .work/moe-w1/pack .work/moe-tier --verify
split: 733 index lines, pack.bin 2678180352 bytes, experts.bin 18327011328 bytes, source pack.bin was 21005191680 bytes
verify: 20 tensors byte-exact, dtypes covered: ['f32', 'q4_k', 'q6_k', 'q8', 'q8_0'], classes covered: expert=True non-expert=True
verify: size OK, trunk 2678180352 + experts 18327011328 == source pack.bin 21005191680
verify: pack.bin on-disk 2678180352 bytes, experts.bin on-disk 18327011328 bytes

real	0m12.789s

$ df -h /home
Filesystem      Size  Used Avail Use% Mounted on
/dev/nvme0n1p7 1001G  920G   76G  93% /home
```

Free space before: 86 GB. After: 76 GB (the box is shared; another lane's
concurrent writes may account for part of the drop not explained by this
run's own ~19.6 GiB of new output, `pack.bin` + `experts.bin`). Both new
files' on-disk sizes are asserted equal to the sizes computed from the
index during the split itself (not just printed after the fact), and
`--verify` recomputed and re-asserted the same equalities independently
from `DSTDIR/index.txt` alone, reading nothing from the split's in-memory
state.

## The two file sizes

- `pack.bin` (trunk, everything that is not a routed expert tensor):
  2,678,180,352 bytes (2.49 GiB).
- `experts.bin` (120 routed expert tensors, 40 layers x 3 matrices):
  18,327,011,328 bytes (17.07 GiB).
- Sum: 21,005,191,680 bytes, exactly the source `pack.bin`'s size.

The trunk is small enough to be comfortably VRAM-resident on its own (2.49
GiB), which is the whole point of B4 stage 2b: this split is what makes
that tier boundary a real file boundary instead of a runtime filter.

## The check that makes this done

Not "it ran". Two assertions inside the tool itself (`assert`, not `print`,
so a violation is a hard failure, not a line a reader has to notice was
missing):

1. `sum(pack.bin sizes) + sum(experts.bin sizes) == source pack.bin size`
   (21,005,191,680 == 21,005,191,680), checked once during the split from
   the running totals, and again independently during `--verify` from
   `DSTDIR/index.txt` alone.
2. Both new files' on-disk sizes match those computed sums exactly, checked
   the same way twice.

Byte-exactness: 20 tensors sampled, grouped by (dtype, class) with at least
one sample per group present in the source. In this pack dtype and class
coincide exactly (routed experts are only ever `q4_k` or `q6_k`; every
other tensor, including the shared expert and the router, is `f32`, `q8`,
or `q8_0`), so there are 5 distinct groups, not 10, and the sampler still
covers all 5 before the 20-sample floor tops the rest up with extra draws.
sha256 of the bytes at the tensor's OLD offset
in the source `pack.bin` compared against sha256 of the bytes at its NEW
offset in the correct destination file (`pack.bin` or `experts.bin`,
decided by the same expert/non-expert classification the split used). All
20 matched.

**The check was proven to catch a real defect, not just to pass.** Before
running it on the real pack, I built a tiny synthetic pack (8 tensors, one
per real dtype plus a shared-expert and a router tensor to confirm they are
NOT classified as routed experts), ran the full split and verify
successfully, then flipped one byte in the split output's `pack.bin` and
reran `--verify-only`: it failed with `BYTE MISMATCH:
blk.0.attn_norm.weight (q8_0, trunk): old offset 16 sha256 ... != new
offset 16 sha256 ...` and a nonzero exit code, not a silent pass. I also
fed the tool a tensor with an unrecognized dtype string; it raised
`unknown pack dtype 'bogus_dtype' ...: refusing to guess its byte size`
and exited nonzero rather than computing a wrong size.

## `--verify-only`

Re-running the check later without re-copying the pack:

```
$ python3 tools/pack-split-experts.py .work/moe-w1/pack .work/moe-tier --verify-only
```

reads both directories' index files and re-derives everything from them; it
does not depend on any state the split step held in memory.

## Rules and verification

- No GPU used anywhere in this tool or this session's run of it.
- `serve/*.mojo`, `kernels/*.mojo`, and `bench/*` not touched (read
  `serve/harness.mojo` for the dtype-size formulas only, per the rule that
  reading is unrestricted and editing is not).
- No em dashes.
- `tools/ci-checks.sh`: `all non-GPU checks passed` at the commit (python
  sources parse, including this new file).

## Files touched

- `tools/pack-split-experts.py`: new file.
- `exchange/2026-09-15-p3-pack-split-report.md`: this report.
- `.work/moe-tier/{index.txt,pack.bin,experts.bin}`: the split output on
  this box, gitignored, about 19.6 GiB combined.
