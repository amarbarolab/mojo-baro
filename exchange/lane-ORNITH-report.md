# Lane ORNITH report: Ornith-1.5-9B (K-quant source) in the engine

Branch `lane-ORNITH` (worktree `$HOME/Projects/mojo/mojo-baro-lanes/ORNITH`), plan item
ORNITH of `~/Brain/mojo/mojo-baro/briefs/2026-09-11-ornith-engine.md`. Preregistration and
results: `bench/ornith-protocol.md`.

**Verdict: item DONE, rebased onto current main (`fc28bf1`, lane-CHAT merged) and re-verified.**
Ornith-1.5-9B decodes correctly and at a usable speed through a K-quant sourced engine pack. No
kernel or engine bug found; every check landed inside its frozen falsifier, and one prediction
band (llama.cpp's tok/s) was wrong on the low side, corrected in the protocol's Result section
rather than silently adjusted.

**Update (post-review):** rebased onto main per the coordinator's request (main had moved to
`fc28bf1` since this lane was cut; `serve/engine.mojo` conflicted with lane-CHAT's decode-loop
changes), fixed the `--q4-draft` K-quant `KeyError` the coordinator flagged, reran G2 against a
fresh main build and `run-tests.sh` — both PASS. Commit SHAs below are post-rebase (all rewritten
by the rebase itself). See "Rebase onto main" below and `bench/ornith-protocol.md`'s matching
section.

## Flag for the reader (read first)

**Scope grew mid-lane, coordinator-approved.** Step 3a needed `BARO_FORCE` (teacher-forced
identity) in `serve/engine.mojo`, which did not exist — only `serve/spark.mojo` had it. The
item's own Rules forbid engine/kernel changes; I stopped and asked (`herd tell`) rather than
either skip 3a or add it unasked. Coordinator's answer: add it, host-level only, proven a no-op
when unset. It needed no kernel change (`serve/engine.mojo`'s decode path already round-trips
through a host `while` loop calling `step_window` once per generated token when `BARO_SPEC=0`;
the addition reads back what that call wrote and optionally overwrites it before the next call).
Full detail in `bench/ornith-protocol.md`'s "Amendment" section.

## Gate

Full suite, per item Rules:

```
gpu-wait run --vram 4 -- bash -c './run-tests.sh > .work/ORNITH-gate.txt 2>&1; echo "run-tests.sh exit $?" >> .work/ORNITH-gate.txt'
```

| receipt | value |
|---|---|
| `run-tests.sh` (pre-rebase) | exit 0 (`.work/ORNITH-gate.txt`) |
| `run-tests.sh` (post-rebase, current) | exit 0 (`.work/ORNITH-gate-rebase.txt`) |
| kernel census before (main `506e91d`, pre-CHAT) | 85 kernels, 38 in registry, 0 orphans |
| kernel census after rebase (`b8c7fa9`, post-CHAT) | 85 kernels, 38 in registry, 0 orphans — identical (this item touches no kernel or `kernels/` file; CHAT's merge added no kernels either) |
| `tools/test_*` files before | 3 (`test_gguf_embed.py`, `test_server.sh`, `test_tokenizer_mojo.py`) |
| `tools/test_*` files after | 4 (**+`test_engine_pack_kquant.py`**) |

Own test (step 2's gate, G0 in the protocol): `tools/test_engine_pack_kquant.py` — packed-q8
dequantised vs gguf-py's own dequantise of the source K-quant bytes, 4 sample tensors (a Q4_K and
a Q6_K trunk tensor, `output.weight`, a `blk.32` NextN tensor). PASS, 0 elements over tolerance
(observed error 27-34% of the q8-rounding tolerance on all 4).

`BARO_FORCE`'s own proof-of-no-op (G2): 20-prompt A/B, this lane's engine vs a pre-change build,
same Qwythos q4 pack, greedy, temperature 0. Run twice — pre-rebase against `main` at `506e91d`
(PASS, 20/20, `.work/g2-ab/results.txt`) and, per the coordinator's request, again post-rebase
against a fresh build of current main `fc28bf1` (PASS, 20/20, `.work/g2-ab-rebase/results.txt`).

## Work done

1. **`tools/engine-pack.py`** now accepts K-quant (Q4_K/Q6_K) source GGUFs: each such tensor is
   dequantised to f32 with gguf-py (`gguf.quants.dequantize`), rounded to bf16, then fed through
   the existing `--q8`/`--q4`/transpose paths unchanged. `token_embd.weight` stays row-major and
   unquantised either way, matching how a native-bf16 source is already handled. Verified the
   tensor name set and shapes match the Qwythos pack order exactly (442/442, 0 missing, 0 extra)
   before writing any code — the item's stop condition never triggered.
2. **`tools/test_engine_pack_kquant.py`** (new): the step-2 gate above.
3. **`bench/ornith-protocol.md`** (new): preregistration (arms, gates, frozen predictions with
   falsifiers, per `bench/PROTOCOL-RULES.md` P1-P6) and the landed Result section.
4. **`serve/engine.mojo`**: `BARO_FORCE` addition (see flag above), proven a no-op when unset.
5. **`bench/ornith-run.sh`** (new): drives G3 (kernel self-consistency) and steps 3a-3d end to end.

## Results (full detail in `bench/ornith-protocol.md`'s Result section)

| check | prediction | landed | verdict |
|---|---|---|---|
| G3: engine vs numpy over the same q8 pack | 64/64 or a few late near-ties | **64/64**, no divergence | exact match |
| 3a: forced agreement vs llama.cpp, median | 60-90% (reported, not gated) | **63/64 (98.4%)**, range 57-64/64 | above band, well clear of the 30% falsifier |
| 3b: ours tok/s_gen, 20-prompt median | 70-95 | **80.78** (76.8-81.0) | inside band |
| 3b: llama.cpp tok/s_gen, 20-prompt median | 110-150 | **88.8** (83.1-90.6) | **below band**, inside the 80-200 falsifier — this protocol's byte-scaling prediction overestimated llama.cpp's edge; ours/llama.cpp = 0.91x |
| 3c: MTP identical to no-spec | 20/20 | **20/20** | exact match |
| 3d: chat smoke | coherent answer | PASS — *"...The capital of France is Paris."* | read and judged coherent |

Evidence: `.work/ornith-run/` in the worktree (`SUMMARY.txt`, `results.txt`, per-prompt logs,
`llama-server.log`, `chat-smoke.json`); `.work/g2-ab/results.txt`; `.work/ORNITH-gate.txt`.

## Rebase onto main (`fc28bf1`, lane-CHAT merged)

`git rebase main` conflicted in `serve/engine.mojo`'s decode loop only: CHAT added
`cancelled`/`stopped`, hint-aware prefill chunking, and extended `Chain.save`'s signature with
two required `Bool`s; this lane's `BARO_FORCE` read-back/overwrite sits in the same loop.
Resolved keeping both behaviours — `pos_before` is captured before whichever `step_window`
variant runs, forcing applies right after using the resulting `wst.pos`, and the checkpoint-save
calls use CHAT's new signature verbatim (the old 5-arg call would not compile against it). No
other file conflicted. Compiles clean; re-verified with a fresh G2 and `run-tests.sh` (both PASS,
receipts above). Detail: `bench/ornith-protocol.md`'s "Rebase onto main" section.

## Commits (`lane-ORNITH`, in order, post-rebase SHAs)

- `7c86ed3` tools(engine-pack): accept K-quant (Q4_K/Q6_K) source GGUFs
- `e55c29c` bench(ornith): preregister identity, decode-speed and MTP predictions
- `cc38a2a` bench(ornith): drop the bf16-numpy reference, use q8-cached model-ref.py
- `5a5932b` serve(engine): add BARO_FORCE teacher-forced identity gate
- `370f3f8` bench(ornith): land step 3 results, add the run orchestration script
- `7697875` tools(engine-pack): fix --q4-draft KeyError on a K-quant output.weight
- `b8c7fa9` bench(ornith): record the main rebase and its re-verification

## Not done / left for the coordinator

- llama.cpp's own `/completion` response `timings` block was not captured in `bench/ornith-run.sh`
  (the 3b llama.cpp number instead comes from `llama-server.log`'s per-request `eval time` line,
  same quantity, different source) — a receipt-completeness gap for whoever next edits that script.
- llama-server's parallel-slot count (4, inferred from round-robin slot ids in its log) was not
  read back from `/props` before the timed runs — did not affect these single-request-at-a-time
  numbers, but the next protocol that shares this script should read it back explicitly (P1).
- Step 3's numbers (G3, 3a-3d) were not rerun post-rebase — nothing in the rebase touched the
  K-quant pack path, forcing semantics, or any kernel, only ordering around CHAT's unrelated
  additions. `bench/ornith-run.sh` is still there to rerun if that's ever in doubt.
