# B4 stage 2: expert locality on the 35B MoE

Preregistered 2026-09-15, before the trace mode was written, at commit
`ae4c5c5`. Binds `bench/PROTOCOL-RULES.md`. Stage 1 (PCIe bandwidth and
expert bytes) is `exchange/2026-09-15-p3-b4-stage1-report.md` at `b7f7db5`.

## The question, and why the tier is not built first

`docs/NEXT-PLAN.md` B4 claims a 100B-class MoE interactive on one 24 GB card
by streaming experts from host RAM. Stage 1 measured the two numbers that
bound it: **28.5 to 28.8 GB/s sustained H2D** on this box (link read back at
x16, 16 GT/s) and **0.573 GB of routed expert bytes per token** for our 35B
(0.791 GB with the shared expert and router), which scales to 1.64 to 2.26 GB
per token for a 100B-class model at the same expert width.

Everything after that rests on one unmeasured quantity: **what fraction of a
token's selected experts is already resident.** MoE-Infinity found under 5% of
experts repeat within a request for a roughly 100-expert model; ours has 256
per layer and picks 8. If locality is that low here, a host-resident tier
cannot pay for itself and B4's build order changes.

So stage 2 measures locality first, from the real engine on the real prompt
set, and only then decides whether the tier is worth building. The
measurement needs the router's choices, not a tier: a tier built before this
number exists would be an engine change resting on an assumption.

## Method

1. **Trace.** `BARO_EXPERTS=<path>` in `serve/window.mojo` accumulates the
   top-8 expert ids `moe_ffn` already computes, per layer, per token, into one
   device buffer, copied to the host once per request. No per-layer sync, so
   the trace run is not a timed run and does not pretend to be one. The 20
   prompts of `bench/mtp-prompts/`, 64 tokens each, `BARO_SPEC=0 BARO_MEGA=0`.
2. **Replay.** `bench/moe-locality.py` replays an LRU per layer over the trace
   at a sweep of capacities (experts resident per layer, from 8 to 256), and
   reports for each: hit rate, expert bytes moved per token at that capacity,
   the resident VRAM those experts occupy, and the implied milliseconds per
   token at stage 1's measured 28.5 GB/s.
3. **Cross-check** that the trace is the engine's real behaviour: the token
   ids the traced run emits must equal the untraced run's on all 20 prompts.
   A trace that changed what the model did would describe a different engine.

## P1 read-back

Engine sha256 built in the same command as the run; `BARO_EXPERTS` echoed by
the run; `prompt tokens`, `tokens: 64` per prompt; the trace's own row count
(20 prompts x 64 tokens x 40 layers x 8 ids) checked against what the replay
reads, so a truncated trace fails loudly rather than producing a hit rate over
fewer tokens than claimed.

## Frozen predictions

1. **Hit rate at a resident half (128 of 256 experts per layer) is below
   70%.** The router is trained to spread load; if this comes back above 90%,
   the routing is far more repetitive than the literature suggests and that
   result is itself the finding.
2. **Sequential repeat rate (the same expert chosen for the same layer on two
   consecutive tokens) is between 5% and 40%.** This is the number a prefetch
   from the current layer's router would exploit, and MoE-Infinity's under 5%
   for a 100-expert model is the low anchor.
3. **No capacity below 256 per layer makes the 100B-class 30 tok/s target fit
   on transfer time alone.** Stage 1 put the required residency at 41 to 58%
   for that target; prediction 1 says a resident half does not deliver a
   hit rate high enough to cover it.

A prediction that fails here is a finding, not a defect, as long as the trace
cross-check passed: this stage measures the model's behaviour, it does not
change it.

## Kill line

If the hit rate at a resident half is below 30%, the host-resident tier is not
built: the streaming claim then depends on prefetch alone and B4's build order
is rewritten around prediction 2 rather than around a cache.

## Result

Filled in when the trace and the replay run. Nothing here is a claim yet.
