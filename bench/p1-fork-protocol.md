# P1 gate 2, identity half: frozen before the first gate run

A fork exported on node A and answered by node B must give the ids one node gives alone, at 100
Mbit, 1 Gbit and 10 Gbit. I expect that to hold on every prompt with f32 states and on 19 of 20
with int8 states, and I expect the link rate to change nothing, because a transport cannot change a
token unless it is broken. Frozen by commit before any gate run. the maintainer asked for this run on
2026-09-17 after two 9B nodes were shown to fit the XTX at `BARO_TMAX=4096`.

Lane FORK, `briefs/2026-09-17-latentos-cross-node-fork.md`. Harness `bench/p1-fork-gate.sh`, driver
`tools/p1-fork-drive.py`, scorer `tools/p1-fork-score.py`, link `tools/fork-link-forwarder.py`.

## What this is, and the half it is not

The plan's gate 2 has two claims. This note covers the first: "fork-on-target ids equal single-node
ids" on the 20-prompt set, across the B4-mini veth link at three rates. The prompts are under 256
ids, so greedy equality over 32 tokens is the right bar by the repo's own rule. The second claim,
payload beating re-prefill for a 32k prefix, is NOT here and cannot run on this rig: one 9B engine
needs about 19 GB at 32k (`docs/P1-FORK-TARGET.md`). Timings are recorded and claim nothing.

## The rig

Two 9B dense q4 nodes alive together on one XTX, `BARO_TMAX=4096`,
`MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_SIZE_PERCENT=10`, `BARO_SPEC=0`, both ordinary user
processes. Node B serves the same pack through a different path. Node A reaches node B through a
forwarder in a peer network namespace, so the state crosses the shaped veth hop (tbf on egress)
exactly once and returns over the unshaped hop. That is link emulation through a relay, not two
machines, and it is one GPU: nothing here shows two GPUs or two hosts.

Read back from the running systems on every run, as hard checks (P1): each engine's own
`limits ... tmax: 4096` line; each node portable and holding zero resident states at start; the
state format in every fork answer equal to the arm's; the link rate measured at the RECEIVER by
pushing 64 MiB into the forwarder's sink, required within 0.7 to 1.1 of nominal; `tc qdisc show`.

## A node that remembers would pass without importing anything

Node B is restarted cold before every rate and before the falsifiers. If it kept a prefix from the
100 Mbit round, then at 1 Gbit it would report `cached == pos` from its own checkpoint and the
reuse check would prove nothing. Every fork must report `pos == |P| - 1` and `cached >= pos` on a
node that has never seen the prompt, or the item is void. The single-node reference must report
`cached 0`. **One void makes the run VOID, and a void is a failure (P10).**

## Two arms and a control

- **f32 arm, the known-good configuration (P14).** A BAROST01 state restores byte-exactly.
- **int8 arm, the plan's format** (`BARO_STATE_INT8=1`), the gate's own arm. The KV pages are
  quantized per block, so the restored prefix is not bit-identical to the one that was saved.
- **Control S, no state moved.** Node A answers the same fork twice; the second time it restores
  its OWN checkpoint. Gate 4 taught me that a restored run and a cold run need not agree even when
  every byte is right (llama.cpp misses its own cold run on a fifth of these prompts), because the
  last prompt token is evaluated by a different path. Control S measures that on our engine, so a
  miss can be attributed instead of guessed at.

## The gate is fed two known-bad states (P11)

Both go through dedicated forwarder ports, so nothing about a request decides how it is treated.

- **flip:** one payload byte inverted in flight. Node B must answer 409 `state_identity`, field
  `payload_sha`, and node A must relay that status and body. This is the brief's "identity check on
  import stays the 409 path", tested live. Required: 5 of 5.
- **swapkv:** K and V exchanged and `payload_sha` RE-SIGNED (the header's hmac is zero until P0b
  pairing, so it is forgeable). Node B will accept and restore it. Gate 4 showed llama.cpp accepts
  and reuses exactly such a state, so only the ids can catch it. Required: 5 of 5 accepted and
  restored, and at least 4 of 5 with wrong ids. If a wrong state that node B restored still gives
  the right ids, the gate cannot see a wrong state and nothing below means anything.

Verified on the CPU before freezing: on a real 61 MB LAT1 export the sha offset is right, `flip`
changes one byte and breaks the hash, and `swapkv` yields a correctly re-signed wrong body.

## Already seen before this freeze

`bench/fork-live-smoke.sh`, 3 prompts, f32, loopback: 3 of 3 identical with `cached == pos`. The
link test (`PHASE=linktest`, no GPU): 96, 957 and 9594 Mbit/s at the receiver. I have seen no int8
fork, no control S result, no shaped fork and no falsifier result.

## Frozen predictions

| | f32 arm | int8 arm |
|---|---|---|
| control S identical to cold, of 20 | 19 to 20 (point 20) | the same prompts as in the f32 run |
| fork identical to cold, each rate | equal to control S's count | 17 to 20 (point 19) |
| fork identical to control S, each rate | **20 of 20** | 17 to 20 |
| prompts whose result depends on the link rate | **0** | **0** |
| flip refused with 409 `payload_sha` | 5 of 5 | 5 of 5 |
| swapkv accepted and restored / wrong ids | 5 of 5 / 4 to 5 | 5 of 5 / 4 to 5 |
| measured rates | within 0.9 to 1.0 of nominal | the same |
| voids | 0 | 0 |
| state bytes, short prompts | about 61.08 MB | about 54.8 MB |

The two sharp ones are the f32 fork against control S (a byte-exact state on the same engine and
the same GPU must reproduce a local restore exactly) and the zero rate dependence. Either failing
is a transport or import defect, whatever the counts against cold look like.

## The verdict, and how a miss is attributed, decided now

PASS as written: 20 of 20 identical to node A's cold single-node ids at all three rates, zero
voids, both falsifiers failing as they must. The gate's own arm is int8; the f32 arm is reported
beside it with the same rule.

A miss is attributed in this order, and the order is part of the freeze:

1. **Engine restore path:** control S misses the same prompt at the same index. No state crossed
   anything. This is the documented E14 class (reduction order) and is not the cross-node move.
2. **int8 quantization:** the arm is int8 and the f32 arm matched that prompt at that rate.
3. **Cross-node defect:** anything else. Control S reproduced the cold ids and the forked state
   did not. **This fires the lane's kill line** ("an identity miss outside the documented E14 one").

A run with only class 1 and class 2 misses is reported NOT MET against the plan's wording, with the
attribution beside it, and whether that wording should change is the coordinator's call, as for
gate 4. I do not get to call it a pass.

## What this cannot show

Two hosts or two GPUs. Any context beyond a few dozen tokens: the states here are dominated by the
52 MB of conv and SSM slots, so the int8 arm exercises little quantized KV, and a long prefix could
behave differently. The 32k timing claim. Any model but the 9B dense q4 pack. Router placement,
which waits for P0b.

## Amendment 1, 2026-09-17, after the full f32 and int8 runs: the ids are a weak detector here

Results as the frozen scorer printed them (`.work/fork/g2-gate/{f32,int8}`, 20 prompts, 0 voids,
rates read back at 96, 957 and 9610 Mbit/s, both engines `tmax 4096`, node B cold at every rate):

| | control S | 100 Mbit | 1 Gbit | 10 Gbit | fork equals control S | rate-dependent prompts | flip refused | swapkv restored / wrong ids |
|---|---|---|---|---|---|---|---|---|
| f32 | 20 | 20 | 20 | 20 | 20, 20, 20 | 0 | 5 of 5 | 5 of 5 / **3 of 5** |
| int8 | 20 | 16 | 16 | 16 | 16, 16, 16 | 0 | 5 of 5 | 5 of 5 / **3 of 5** |

**Both runs are `RESULT: FAIL, the harness did not prove itself`, and I am leaving that verdict in
place.** I froze "at least 4 of 5 swapped states give wrong ids". Two did not: `p02-python-fib` and
`p04-list-planets`, the same two in both formats. Their continuations are strongly determined (a
fibonacci function, a list of planets), the prompts are 15 and 16 tokens, and 24 of the model's 32
layers are recurrent, so a state whose attention K and V are fully exchanged still yields the same
32 ids. No threshold is changed. A gate that lets two grossly wrong states through has not shown
that 20 of 20 means the state is right, and I will not report the f32 row as a pass.

What held, stated without the word pass: both sharp predictions (an f32 fork equals a local restore
on 20 of 20 at every rate; zero prompts depend on the link rate); every one of 65 imports per run
was accepted only after node B recomputed sha256 over the received body, so the TRANSPORT is
byte-exact by a check that does not depend on ids; a flipped byte is refused 5 of 5 with 409
`state_identity` `payload_sha`, relayed by node A. Control S is 20 of 20, so on OUR engine a
restored run reproduces the cold run on these prompts, unlike llama.cpp in gate 4.

What did not hold: my int8 prediction was 17 to 20, measured 16. All four misses (p07 at 1, p12 at
3, p15 at 14, p16 at 9) are attributed by the frozen order to int8 quantization, because the f32 arm
matched each of them at each rate, and they are the same four at every rate. So the plan's own
export format does NOT meet the plan's identity bar even on prompts of 16 to 32 tokens.

What the falsifier result means, which I had the evidence for and did not use: gate 4 had already
shown the K/V swap is gentle on the hybrid (one preflight state held until index 3). Here it is
gentler still. The same detector is flipped by int8 rounding on 4 of 20 near-tie prompts and is
blind to a full K/V exchange on 2 of 5 strongly determined ones. Its sensitivity is a property of
the prompt, not of the defect. **Untested entirely: corruption of the conv and SSM state**, which is
52 of the 61 MB; I have no evidence the ids would or would not catch it.

A detector with real power exists and is not part of this freeze: have node B RE-EXPORT the
imported prefix and compare the bytes with node A's export (f32: header, tokens, conv, SSM, and K/V
for positions below `pos`). That tests where every byte landed, not what 32 tokens happened to say.
