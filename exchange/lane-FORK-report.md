# Lane FORK report: no gate passed, and the byte check found an engine bug

Gate 4 is NOT MET on all three models (16, 13 and 14 of 20), against a bar llama.cpp cannot clear
when it hands state to ITSELF (15, 16 and 16 of 20). Gate 2 did not run. What the lane did
establish is narrower and solid: the forward bridge has no layout defect.

**Correction, 2026-09-17 afternoon, after this report was first committed (`856f6da`).** That version
said gate 2's rig "cannot be built on one XTX: two of our engines do not fit, and MAX's memory cap
does not change that". Wrong as a general claim. I had tested the 9B at `BARO_TMAX=33024` only and
generalised. the maintainer asked whether 1B models could be tried, and the probe then showed two 1B engines
live together (11.3 + 6.2 GB) and, at TMAX 4096 with the cap at 10, **two 9B engines live together,
10.7 GB each, 23.5 GB total, identical answers**. Gate 2's identity half therefore HAS a literal rig;
only its 32k timing half does not. Details in `docs/P1-FORK-TARGET.md`. I had already routed the
wrong claim to the board, a Brain note and memory; all three are corrected.

Date 2026-09-17. Branch `lane-fork`, 11 commits ahead of `main`, builder fable. Brief
`briefs/2026-09-17-latentos-cross-node-fork.md`. Receipts under `.work/fork/`.

| gate | result | receipt |
|---|---|---|
| Gate 2, identity half (two live 9B nodes, veth at 100 Mbit, 1 Gbit, 10 Gbit) | **RAN: FAIL by the frozen falsifier rule.** f32 20/20 at every rate, int8 16/20 at every rate, but 2 of 5 K/V-swapped states still gave the right ids, so the ids cannot certify a state | `bench/p1-fork-protocol.md`, `.work/fork/g2-gate-fixed` |
| Gate 2, 32k timing half | **NOT RUN.** Two engines do not fit at 32k; rig choice is the coordinator's or the maintainer's | `docs/P1-FORK-TARGET.md` |
| Engine bug found and fixed | `save_state` could export another prompt's conv and SSM state; fixed `511cdb4` | `.work/fork/bytes-check`, `.work/fork/bytes-check-fixed` |
| Gate 4, E15's three models continue from our state | **NOT MET** on all three; UNCLASSIFIED by the frozen rule | `.work/fork/g4-gate/{lily,qwen25,ornith}` |
| Kill line | **Not decided by this lane.** See "The kill line" below | |

## Gate 4: the bar is one llama.cpp misses against itself

Protocol `bench/p1-bridge-protocol.md`, frozen at `599d8fd` before any GPU run, three dated
amendments after. 20 prompts, first 32 ids, greedy, f16 KV, flash attention on, all read back from
files the running systems wrote. Identical to llama.cpp's COLD run, of 20:

| | control L: llama restores its own bytes | control N: our engine, no state moved | **primary: llama continues from OUR state** | primary equals control L |
|---|---|---|---|---|
| lily-7B Q6_K (`llama`, spark) | 15 | 15 | **16** | 17 |
| Qwen2.5-7B Q4_K_M (`qwen2`, spark) | 16 | 15 | **13** | 13 |
| Ornith-1.5-9B Q4_K_M (`qwen35`, engine) | 16 | 11 | **14** | 18 |

Voids 0 on all three: every restored arm reported `n_restored` and `cache_n` equal to `|P| - 1`, and
every cold arm `cache_n 0`. Without that check a silent recompute would have scored 20 of 20.

The plan asks for 20 of 20. llama.cpp restoring its own bytes reaches 15 or 16, because a restored
run evaluates one token over cached f16 cells and a cold run evaluates the whole prompt in one
batch, and near-tie tokens flip between the two. On lily p01, control L, control N and the primary
agree with each other on all 32 tokens and only the cold arm differs. The bar measures batch shape
as much as it measures the bridge.

## Every prediction I froze was wrong except the two that mattered least

| | predicted | measured |
|---|---|---|
| control L | 20, 20, 19 to 20 | 15, 16, 16 |
| control N | 8 to 14, 8 to 14, 12 to 18 | 15, 15, 11 (each one outside its band by one) |
| primary | 17 to 20 on each | 16, 13, 14 |
| falsifier, voids | 0 to 2, 0 | 0, 0 |

I did predict, in writing and before data, that the gate as written was more likely NOT MET than
met. That held. Everything quantitative beside it was too optimistic, control L worst of all: I
assumed a byte-exact restore implies identical ids, and it does not.

## The bridge has no layout defect, shown four ways

1. **Byte identity, no GPU.** `bench/bridge-roundtrip.sh`: a slot file written by llama.cpp, through
   a reverse leg and back through `tools/state-to-llama-slot.mojo`, is byte-identical on all three
   architectures (53,183,716, 2,098,220 and 803,756 bytes), and llama.cpp reuses it.
2. **The gate can fail.** K and V swapped in the written file: 0 of 20 on lily, every one diverging
   at index 0, while llama.cpp ACCEPTED and REUSED every swapped file. Only the ids catch a wrong
   state, which is why the ids are the arbiter. On the hybrid the swap touches 8 of 32 layers and
   one preflight state held until index 3, so the falsifier is gentler there.
3. **Full identity on most prompts.** 13 to 16 of 20 continuations match for all 32 tokens. A wrong
   layout cannot do that once.
4. **Our K/V against llama.cpp's own, elementwise** (`tools/slot-kv-diff.py`, calibrated 0.0000
   against itself and 1.0 to 48 against the swapped file): lily K 1.2 percent, V 2.3 percent
   median; Qwen2.5 K 2.0 percent, V 3.6 to 4.8 percent. Smooth over layers and positions.

## Open finding, not diagnosed: Qwen2.5's K/V sits twice as far out as lily's

Qwen2.5 is the weak model: primary 13, and 3 prompts (p01, p10, p16) miss on the primary while BOTH
controls pass, so neither llama.cpp's restore path nor our own decoding flips them; only their
kernels over our KV does. I suspected the qwen2 QKV bias. The measurement killed that: Qwen2.5's
layer-0 K, which already carries the bias and RoPE, matches llama.cpp's within 0.17 percent. Why
the mid-stack error is about 2x lily's (Q4_K_M against Q6_K, or bf16 activations meeting qwen2's
outliers) is unmeasured. It is a question about the spark path's numerics, not about the bridge.

## The kill line: my rule did not return an answer, so I am not supplying one

The brief's kill line is "an identity miss outside the documented E14 one". Read literally, gate 4
has identity misses. My frozen classifier fires the kill line only on a layout defect (most
prompts diverging by index 2, or the primary more than 2 below control N), and neither condition
is met on any model. But it did not return NUMERICS either: each model has one primary miss before
index 3 (lily p11 at 1, Qwen2.5 p10 at 1, Ornith p06 at 0), and the numerics clause requires all at
3 or later. Two of those three are prompts where control L and control N miss at the same index, so
the early index is the cold arm's. Amendment 1 had flagged that signal as weak and changed no
threshold; I did not change one after the rule produced an awkward answer. Qwen2.5's primary sits
exactly at the margin (13 against control N 15). The verdict is UNCLASSIFIED, and whether that
fires the kill line, and whether the plan's bar is amended under `PROTOCOL-RULES.md` P14, belongs
to the coordinator.

## Gate 2: a literal rig exists at 4k, not at 32k (corrected, see the top)

`bench/fork-cap-probe.sh`, dense q4 pack, `BARO_TMAX=33024`. With
`MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_SIZE_PERCENT` at 45, 25 and 10, node A held 19.5, 21.5 and
18.9 GB and node B died with `hipErrorOutOfMemory` each time. I concluded "the knob does not govern
it" and called the lever closed. It is closed at 32k only: at that context one engine needs about
19 GB and grows past any pool the cap sets. At TMAX 4096 two 9B engines fit (table in the doc).

Four runnable rigs, what each cannot judge, and my recommendation (sequential XTX plus a
self-target forwarder) are in `docs/P1-FORK-TARGET.md`, sent to `w82:pC`. Each rig changes what the
verdict means, so I did not pick one and build it.

`POST /v1/fork` with `"target":"HOST:PORT"` landed (`8ba8294`): `cargo nextest` 75 of 75, including a
mock node receiving header and file with the exact `Content-Length` and the 409 relay.

**Added the same afternoon: it has now moved state between two live nodes.**
`bench/fork-live-smoke.sh`, two 9B nodes alive together (TMAX 4096 read back from each engine's
limits line, cap 10), node B cold (`resident_states 0`) and serving the same pack through a
different path: 3 of 3 prompts, B reported `cached` equal to the exported position every time (it
restored the import, it did not re-prefill), 32 ids equal the single-node ids, A saved 3 states and
B loaded 3, 0.7 to 1.1 s wall for a 61 MB f32 state (`.work/fork/live-smoke`). That is a smoke, not
gate 2: three short prompts, loopback, no link shaping, f32 not int8, no 32k, no frozen
predictions, and the smoke has not been fed a known-bad state. **Gate 2 remains NOT RUN.**

## Gate 2's identity half ran, and what it found was an engine bug

Added 2026-09-17 evening, after the maintainer asked for the run. Protocol `bench/p1-fork-protocol.md`,
frozen at `b499cfb` before any gate data, two amendments after. Two live 9B nodes, node B restarted
cold before every rate, link read back at 96, 957 and 9610 Mbit/s, 20 prompts, 0 voids. Numbers from
the fixed engine `aee16f63`; the first run on the old engine gave the same ones.

| | control S | 100 Mbit | 1 Gbit | 10 Gbit | rate-dependent prompts | flip refused | swapkv restored / wrong ids |
|---|---|---|---|---|---|---|---|
| f32 | 20 | 20 | 20 | 20 | 0 | 5 of 5 | 5 of 5 / **3 of 5** |
| int8 (the plan's format) | 20 | 16 | 16 | 16 | 0 | 5 of 5 | 5 of 5 / **3 of 5** |

**Verdict, as the frozen scorer prints it: FAIL, the harness did not prove itself.** I froze "at
least 4 of 5 K/V-swapped states give wrong ids", and two strongly determined short prompts still
gave the right 32 ids. I left the verdict and moved no threshold, and I do not report the f32 row
as a pass. My int8 prediction (17 to 20) missed: 16, the same four prompts at every rate, all
attributed to int8 quantization because the f32 arm matched each. The plan's own export format does
not meet the plan's identity bar on prompts of 16 to 32 tokens.

Since the ids could not certify a state, I compared bytes (`bench/fork-bytes-check.sh`): node B
re-exports what it imported, and a second arm sends a K/V-swapped state on purpose so that only a
node which kept the imported bytes can produce the expected result. **Its first run found a
pre-existing bug in `serve/engine.mojo`:** `save_state` chose the checkpoint by position alone, so
`p05-math`'s export carried `p02-python-fib`'s conv and SSM state (both at `pos 14`), 52 of the 61
MB, under a valid sha and passing identity checks. One conversation's state leaking into another's
export, accepted by any receiver. Fixed in `511cdb4`; reproduced first (6 of 7), then 7 of 7
bit-identical including deliberately swapped states; `ci-checks` exit 0, nextest 75 of 75.

Established by checks that do not rest on ids: the transport is byte-exact (node B's sha256 on
every import), a corrupted state comes back as the 409 path relayed by node A, and node B keeps and
places every imported byte of conv, SSM and K/V. Established by ids only, and so weakly: that 32
tokens continue the same. Control S is 20 of 20, so our engine's restored run reproduces its cold
run on these prompts, which llama.cpp's does not.

## What I got wrong along the way

- **I declared the two-node rig impossible from one operating point.** I varied the memory cap and
  never the context length, wrote "cannot be built" into a doc, this report, the board, a Brain
  note and memory, and recommended substitute rigs on that basis. It took the maintainer asking about 1B
  models to find that two 9B engines fit at TMAX 4096. This is the most expensive mistake in the
  lane: it is why gate 2's identity half, which could have run today, did not.

- **I blocked the GPU queue for 9 minutes in front of a priority-90 job.** The cap probe started
  its servers in a command substitution, lost their pids, and hung after node B's OOM. Cancelled
  by hand, fixed (`ebb406d`) with a CPU preflight of the failure path, ledger
  `gpuwaitingroom.md` 2026-09-17. Reading the gpu-wait README later caught the same class of bug in
  the gate harness before it ran: a server launched as `gpu-wait run ... &` belongs to the daemon
  and cannot be reaped by the caller.
- **My scorer invented a void.** It voided prompts whose cold run reached EOS before 32 tokens; the
  frozen note's void list does not contain that. As first run both spark models read VOID; as the
  note is written they read NOT MET. Neither is a pass, which is the only reason I corrected it
  after seeing data (amendment 3).
- **Two GPU jobs died on harness errors a careful read would have caught:** baro-serve's
  `/tokenize` is 503 on a spark pack (I had deviated from my own note's tokenizer path), and
  `engine.mojo` prints two `state saved:` lines per save where I assumed one.
- A nested heredoc ran the tail of a Python patch as shell text. I audited before continuing:
  nothing outside the worktree, two empty stray files, removed.

GPU: 13 jobs of mine, 8 clean. Of the 5 non-zero exits, 2 were the cap probe's OOM (the failure is
its finding), 1 the hung probe, 2 the harness errors above. The queue's stats do not isolate a
lane's minutes, so I have counts and no minutes.

## Verify

At `6bf0bc2`: `tools/ci-checks.sh` (GPU-free) exit 0, 14 OK steps, no FAIL line
(`.work/fork/ci-checks.log`); `cargo nextest run --bin baro-serve` 75 of 75. `serve/spark.mojo` was
built from this worktree for two profiles and its `state_save` ran on the GPU in the lily and
Qwen2.5 gates. `./run-tests.sh` NOT RUN: it is the on-card kernel suite and this lane touched no
kernel file. None of this verifies `/v1/fork` `target` between live nodes.

## What would prove this wrong, and what it does not show

If the bridge were subtly wrong in a way ids cannot see at 32 tokens, `tools/slot-kv-diff.py` on
any prompt would show a broken layer or position instead of a smooth few percent; anyone can run
it on the slot files under `.work/fork/g4-gate/`. Not shown here: a phone or a Mac as receiver (the
receiver was `llama-server` on the same XTX), 8k or 32k states (the tool's cell loop is scalar and
unmeasured at depth), the int8 state format through the bridge, and two distinct live engines of
ours exchanging anything at all.
