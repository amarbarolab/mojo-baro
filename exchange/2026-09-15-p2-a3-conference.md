# A3 conference: what the continuous-batching plan should look like

`briefs/2026-09-15-p2-a3-conference.md`. Read before writing this:
`docs/NEXT-PLAN.md` A2 and A3, `serve/src/main.rs`, `serve/src/engine.rs`
(full), `serve/PROTOCOL.md`, `bench/PROTOCOL-RULES.md` P5, and my own
`bench/fork-protocol.md`. Everything below is grounded in those, not in
the plan's own summary of them; where a claim is a direct quote or a line
number, that is the check for it.

## 1. Shape

One item per **wire-protocol invariant that must hold**, not one item per
endpoint. The HTTP surface is the cheap part: `/v1/fork` was under 100
lines of Rust and reused every existing piece (`Gen`, `check_and_submit`,
`collect`). Continuous batching's hard part is entirely on the engine
side and the contract between the two processes (`serve/PROTOCOL.md`
calls itself "the seam where the process boundary is later replaced by a
C ABI" -- it already expects to change). An item sized around an endpoint
hides that; an item sized around an invariant does not let me start
writing Rust before the invariant it depends on is proven on the Mojo
side.

A good item, concretely:
- States which invariant it proves (not which feature it ships).
- Names the exact files it touches, and says explicitly what it does
  not touch, the way this brief itself says `serve/window.mojo` and
  `engine.mojo`'s spec config are A1's this leg. Three lanes have shared
  this tree today; every item needs that line or it costs someone a
  stash.
- Comes with its own protocol note before any code, the way
  `bench/fork-protocol.md` states the endpoint shape and why before gate
  1 runs. For A3 that note has to include the wire-format delta (today:
  one request line in, `Tok`/`Done` lines out, strictly serial,
  `serve/PROTOCOL.md` line 30-32 says so in those words) -- not a "the
  scheduler will handle it" wave.
- Has a real-request check, not a unit test, per the standing rule --
  but see question 4 for why "N clients, T=0 identity" alone is not
  enough of one for this specific plan.
- Has its own `ENGINE:`-shaped stopping point already written in, not
  discovered mid-build. B3 and A5 both hit a real boundary this session
  (a kernel ABI question, a device-mask interface); A3 will hit several,
  because the batching decision has to live in the decode loop itself,
  not in Rust.

Roughly four items in this order, each landable and gated on its own:
(a) the wire-protocol extension itself, proven with a fake multi-id echo,
no real batching yet; (b) two requests' KV/conv/ssm state coexisting in
VRAM without corruption, still decoded one at a time (proves the memory
layout, not the scheduling); (c) one decode launch advancing two
requests together, gated on identity against each running alone; (d) the
admission/preemption policy. (c) is the one that needs fable or A1's
direct involvement, not because of ceremony, but because it is a decode
loop change, the same class of change this brief's own kernel-file rule
exists for, even though nothing in it is literally a `kernels/*.mojo`
edit.

## 2. What would make me fail

**The plan must not describe the Rust server as "the scheduler."**
`serve/src/engine.rs`'s `EnginePool::submit` does not schedule anything
today: it picks whichever pool member has the shortest queue and hands
it the whole request (`engine.rs:236-252`). The worker loop then does
`while let Some(job) = rx.recv().await { ... run_one ... }` -- one job,
start to finish, before the next (`engine.rs:281-284`). There is no
point in the current Rust code where a batching *decision* could be
inserted, because there is no place where two requests are ever in
flight on the same engine at once. A plan that says "extend the queue to
batch requests" sends me looking for a batching seam in Rust that does
not exist; the seam has to be built into the Mojo decode loop, which
today processes exactly one `prompt: List[Int]` per request line with no
concept of a second concurrent one.

**`BARO_POOL` is not a step toward this, on this box.** A3's own text
lists it next to the request queue as existing infrastructure. It is
built and typed (`bench/chat-protocol.md` C4), but `BARO_POOL=2`'s
second engine failed to load its pack with `hipErrorOutOfMemory` every
time it was tried, because one engine's MAX runtime reserves 23.5-24.8
GB on a 25.75 GB card (`docs/BASELINE.md`, `bench/chat-protocol.md`
1531-1545, P-J2/P-J3, three attempts, same failure). `BARO_POOL` gives
real concurrency only across GPUs, which this box does not have more
than one of. A3's actual concurrency has to come from batching rows
inside the one engine process already running, not from more processes.

**"mrows up to 8" is not request batching, and treating it as a running
start would be wrong.** `serve/PROTOCOL.md`'s own ready line defines
`mrows` as "prefill chunk" and `kmax`/`spec_k` as draft width -- both are
properties of *one request's own* prefill or speculative window, not
rows contributed by different concurrent requests. A3's phrasing ("the
engine already runs m up to 8 rows per launch") reads as if the
multi-row machinery Orca wants is already half-built. It is not: it is
multi-row *within one sequence*, and reusing it for multi-*request* rows
means threading a request id through every one of those rows, which
today's buffers, KV pool and sampler call sites do not carry.

**The checkpoint mechanism I just built B5 on is sequential-reuse, not
concurrent-hold, and a plan would likely assume otherwise.**
`serve/prefix.mojo`'s `Chain.restore` and `Chain.save` both take a fixed
`slot` parameter, and every call site in `engine.mojo` passes `0` or
`wst.ring` for one active sequence's own ring index -- there is one
conv/ssm region in flight, reused request after request, never two at
once. B5's forking works precisely because branches run sequentially
through that one region. Continuous batching needs N *simultaneously
live* conv/ssm regions (one per in-flight request), which is a memory
layout change, not a scheduling change, and it is squarely inside A2's
scope (block tables), not a detail A3 can defer past it the way the
"needs A2" line implies it might only be a soft dependency.

## 3. What the wishlist gets wrong

**Right, and correctly ordered:** needing A2 first. Not because block
tables are merely convenient for "different lengths sharing a batch" (the
plan's own phrasing), but because the KV pool today is a single
position-addressed region for one sequence (read directly in `prefix.mojo`
for B5); N concurrent sequences cannot share it without either paging or
a hard cap that defeats the point. This is a real blocker, not a
nice-to-have, and the plan should say so more strongly than "needs A2."

**Understated, not wrong:** preemption by recompute. The plan cites
vLLM's own ablation favoring it at small block sizes and leaves it as a
design decision for later. Given prefix checkpoints already exist and
restore is O(1) in prefix length (measured this session, B5: 2.2-4.3 ms
flat from 869 to 32636 tokens, `exchange/2026-09-15-p2-b5-report.md`),
recompute-by-checkpoint-restore is very likely cheaper here than in a
system without that mechanism -- this is a case where the plan should
name a concrete prediction (restore-based preemption beats swap-based
preemption on this engine) and gate it, not leave it as an open design
question with no expected answer.

**Wrong, or at least unmeasured enough that it should not be stated as a
number in a gate:** "aggregate tok/s at m=4 against the P5 row-scaling
receipt." P5's receipt is for the ffn GEMM shape; this model's own
delta/SSM phase does not scale like the ffn GEMM does --
`bench/ssm-mrow-protocol.md`'s own measured table has the delta stage at
1.46x cost for m=2 (0.7074 to 1.0331 ms/window), the worst-scaling stage
in the sub-block, while conv/l2/gated all sit near 1.05-1.06x and the
profiled total comes in at 1.15x. A3's gate should predict a *specific*
sub-linear number for m=4 grounded in that table (delta itself is the
stage likely to dominate any further row increase, not the total), or
say plainly that the m=4 number needs a fresh row-scaling receipt on the
delta phase specifically before it can be predicted at all. Right now
the plan's gate compares a future measurement against a receipt from a
different kernel family (ffn GEMM, bandwidth-bound; delta is not).

**Not actually needed for a first useful version:** per-request sampler
state as a *design decision*. It already exists -- C3 (this repo's own
sampler wiring) already landed per-request `temperature`/`seed`/`top_p`/etc.,
and B5 exercised it live today at T=0.7 with independent seeds through a
real fork. Nothing about continuous batching
needs to design this; it needs to *thread the existing SampleParams
correctly through a now-multi-request decode step* so token i still gets
row i's own params, which is an implementation detail of item (c) above,
not a separate open design question the plan needs to resolve first.

## 4. The first gate

The plan's own proposal, "N=4 concurrent clients through `baro-serve`,
each output identical to its single-request run at T=0," is close but
tests the wrong thing as a *first* gate: it already passes today, with
zero batching. `tools/test_server.sh`'s existing "queue" case sends 2
concurrent requests and checks both match ref; that is this exact claim
at N=2, unchanged code, already green. A gate that a no-op
implementation already satisfies is not a gate for this feature -- it
would pass a version of A3 that never batches anything and just keeps
today's serial worker loop.

Smaller, and the one that actually tests the thing being built: **two
requests advanced by one decode step together, both reading back the
token their own single-request run would produce at that same position,
with a receipt that the step actually ran with both request ids present**
(not two separate launches serialized inside one Rust call, which is
indistinguishable from today's behavior from the outside). The check
that proves it happened, not just that the output was correct: the
engine's own per-step log line names both request ids for that launch
(or an equivalent explicit receipt), the same way `mega fail word` and
`tok_s_gen` are already printed and read back rather than assumed
(`bench/PROTOCOL-RULES.md` P1's own rule, general but exactly on point
here). Identity against each request's single-run continuation is still
the correctness half of this gate; the receipt is the half that
proves batching, not serialization wearing a batching label, actually
ran.

## Where I'd stop and say "measure this first"

One number the plan should not guess at: how much of MAX's 23.5-24.8 GB
per-process reservation is a **fixed cost of one running process**
versus **proportional to how many logical sequences that one process is
tracking**. If it is fixed (the "runtime reserves" phrasing in
`docs/BASELINE.md` reads that way, not "runtime uses"), then A2 + A3
inside one process could have far more concurrency headroom than
`BARO_POOL`'s per-process duplication suggested when it hit
`hipErrorOutOfMemory` at a second copy. If it scales with tracked
sequences, the real ceiling on N is a VRAM budget nobody has measured
yet, and it bounds every later item in the plan (how big a batch can
ever be on this card). This is a small, cheap, GPU-minutes-scale
measurement (allocate a second, third, fourth idle KV/conv/ssm region
inside one already-loaded engine process and read the free VRAM back
each time) and it should happen before the plan fixes a target N
anywhere in its gates, not after.
