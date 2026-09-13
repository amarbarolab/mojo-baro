# 7900 XT test kit

For running `mojo-baro` on a Radeon RX 7900 XT (gfx1100, 20 GB) on a machine
that normally boots Windows. Everything here assumes a native Linux boot. Read
the "Why not Windows" section before you spend time on anything else.

## Why not Windows

Mojo and MAX ship for Linux only. There is no Windows build of the `mojo`
compiler, so none of this repo compiles natively on Windows: not the kernels,
not the C++ shim, not the engine.

WSL2 with ROCm is theoretically possible and gfx1100 is on AMD's WSL support
list, but that support has not been re-verified for this kit and WSL puts a
virtualized memory path between the process and the GPU. Correctness results
from WSL would be fine. Timings would not be comparable to the reference box,
and should not be recorded as benchmark arms.

So: a spare partition, a second drive, or a live USB with persistence. Any
current Arch, Ubuntu 24.04+ or Fedora will do.

## Prerequisites

| | |
|---|---|
| GPU | RX 7900 XT, confirmed as `gfx1100` by `rocminfo` |
| ROCm | 7.2 preferred. 7.x should work; anything older is untested here |
| Python | 3.12 |
| Tools | `uv`, `cmake`, a C++ toolchain, `git`. Rust only if you want the HTTP server |

Confirm the card first, before installing anything else:

```sh
rocminfo | grep -i gfx
```

You want to see `gfx1100`. If you see anything else, stop: the rest of this kit
does not apply to your card.

## Setup

The kit archive contains three repos. Unpack them side by side in one parent
directory, because two of them are referenced by relative path from the third.

```sh
mkdir -p ~/baro && cd ~/baro
tar xf baro-7900xt-kit.tar.gz     # gives mojo-baro/ mojo-uregex/ mojo-minja/
cd mojo-baro
uv sync
```

`uv sync` creates a repo-local `.venv` pinning `max[all]==26.5.0`, which is
Mojo 1.0.0, from PyPI. Not the nightly index: 26.5.0 is a stable release and is
not carried on nightly, so a nightly pin will fail to resolve. Nothing is
installed machine-wide, and every script in the repo calls `./.venv/bin/mojo`
rather than a system Mojo.

Anything that touches `serve/tokenizer.mojo` or `serve/spark.mojo` needs the two
sibling repos on the include path:

```
-I ~/baro/mojo-uregex/src -I ~/baro/mojo-minja/src
```

## Tier 1: correctness

```sh
./run-tests.sh
```

Builds the hipBLASLt shim, then builds and runs the Mojo parity tests and the
kernel census. This proves the toolchain is sane on your box. It makes no
performance claim and it will not tell you anything about your card that it
would not also tell you about any other gfx1100.

Expected: every test prints a pass line and the script exits 0. If the shim
build fails, that is a ROCm or CMake problem, not a repo problem, and the error
will say which library it could not find.

One tier 1 note specific to the 7900 XT: nothing in `run-tests.sh` is sized
against 24 GB, so the smaller framebuffer should not matter here.

## Tier 2: GEMM benchmark

```sh
./bench/run.py
```

This is the interesting tier. It runs the correctness gate first, then the
throughput table, and it logs the commit and the GPU name with every row, so a
number is always traceable to the code that produced it.

The 7900 XT differs from the reference 7900 XTX in ways that should be visible
here: 84 compute units against 96, and roughly 800 GB/s of memory bandwidth
against 960. Compute-bound and bandwidth-bound rows should separate by
different ratios, and that separation is the actual result worth sending back.

Two rules carry over from the reference box unchanged, and results that break
them are not usable:

- **Working sets at or above 96 MB are invalid for single-buffer GEMM timing.**
  Your card has the same 96 MB Infinity Cache, so the same contamination
  applies. A number measured out of cache is not a memory-bandwidth number.
- **Read every arm-defining parameter back from the running system and record
  it.** Passing a flag is not evidence that the flag took effect. A silently
  inert parameter produces clean numbers with a tight spread, and the spread
  will not catch it. No receipt, no arm. See `bench/PROTOCOL-RULES.md` P1.

Also run the clock probe (`bench/clock-probe.sh`) before and after a timed set.
Clock ramp has inflated results on the reference box before, and a cold card
ramping into a run is a real effect, not noise to average away.

## Tier 3: decode

This one needs a model, and it needs the most care.

You need a Q4_K GGUF. the maintainer will send you the file or the exact model
identifier separately; do not substitute a different quant or a different
checkpoint, because the receipts are only comparable on the same weights. Pack
it locally:

```sh
./.venv/bin/python tools/engine-pack.py MODEL.gguf .work/engine-pack-q4 --q4
./.venv/bin/mojo build serve/engine.mojo -I kernels -o .work/engine
```

**The 20 GB framebuffer is the live problem here.** On the reference 24 GB card,
one engine's MAX runtime reserves roughly 23.5 to 24.8 GB. That does not fit on
your card, and the failure mode is an allocation error or a segfault at startup,
not a graceful message. Before you conclude anything is broken, lower two
things:

1. `MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_SIZE_PERCENT`, which caps the MAX
   device pool. `bench/run.py` already sets this to 10 for its own runs.
2. The context length. The 32k configurations are sized for 24 GB and will not
   fit. Start well below that and work up until it stops fitting.

Report the largest context length that actually runs. That number is itself a
result, and it is one the reference box cannot produce.

Decode numbers are a 20-prompt median, never a single prompt. See
`bench/PROTOCOL-RULES.md` P4. A one-prompt number is a receipt, not a claim.

## What to send back

Per tier, whatever you got plus the receipts that make it mean something:

- The `rocminfo` gfx line and your ROCm version.
- For tier 2: the printed table, plus the parameter read-back for each arm and
  the clock probe output.
- For tier 3: the pack flags used, the context length that fit, the memory
  percent setting, and the 20-prompt median rather than the individual runs.

A number without its receipt cannot be compared to anything on the reference
box, so it is better to send four rows with receipts than the whole table
without them.

## If something fails

`docs/BASELINE.md` is the repo's current truth: every verified number and every
known trap, with the command that reproduces it. Read it before concluding that
something is broken. The protocols in `bench/*.md` are frozen, which means they
describe how a measurement must be taken for its result to count, not just how
to run the script.
