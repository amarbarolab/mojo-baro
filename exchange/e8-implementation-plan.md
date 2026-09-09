# E8 — Training-Free `HIDDEN` on Qwythos & Small Model Co-Residency (The Thesis Test)

Empirical evaluation of whether continuous latent handoff (LatentMAS protocol: feeding last-layer $h_t$ or realigned expected embedding $e_t$ back into the transformer input) matches or exceeds text handoff on checkable tasks without training, comparing **Topology 1 (Two Qwythos-9B instances)** and **Topology 2 (Two small model instances)** co-resident in 24 GB VRAM.

## User Review Required

> [!IMPORTANT]
> **Frozen Preregistration Gate (06 §E8):**
> - **PASS**: $L8\text{-soft} \ge T - 3\text{ pp}$ accuracy $\implies$ `HIDDEN` admitted training-free.
> - **SPEED-ONLY (Wenzel Outcome)**: $T - 3 > L8\text{-soft} \ge \text{arm0} + 3\text{ pp}$ $\implies$ admitted for speed only, quality flagged.
> - **KILL**: $L8\text{-soft} < \text{arm0} + 3\text{ pp}$ and $L32\text{-soft}$ no better $\implies$ untrained model cannot read its own hidden states as input; `HIDDEN` moves to Phase 2 / $\Sigma$ v1 (Coconut training), and Phase 1 latent primitive shrinks to KV/SSM handoff only.

> [!NOTE]
> Per E9 receipt, all `HIDDEN` vectors are stored and transferred in **$f32$** ($128\text{ KiB}$ for 8 steps), as $bf16$ rounding caused $25\%$ sequence divergence.

---

## Experimental Topologies

Testing both co-residency configurations in the 24 GB VRAM envelope on `hercule`:

| Topology | Agent A (Thinker / Producer) | Agent B (Actor / Receiver) | Total VRAM | IPC Mechanism |
|---|---|---|---|---|
| **Topology 1 (9B Co-Residency)** | Qwythos-9B Q4 ($6.64\text{ GB}$) | Qwythos-9B Q4 ($6.64\text{ GB}$) | **$13.28\text{ GB}$** | SCM_RIGHTS sealed `memfd` ($128\text{ KiB}$) |
| **Topology 2 (Small Co-Residency)** | Spark-4B Q4 ($2.5\text{ GB}$) | Spark-4B Q4 ($2.5\text{ GB}$) | **$5.0\text{ GB}$** | SCM_RIGHTS sealed `memfd` ($80\text{ KiB}$) |

Both topologies eliminate PCIe weight swapping between turns (zero-swap co-residency).

---

## The 5 Evaluated Arms

For each task item:

| Arm | Agent A Action | Agent B Action | Cost / Latency Target |
|---|---|---|---|
| **`0`** | None (baseline control) | Answers prompt directly with greedy decode | Single-agent baseline |
| **`T`** | Generates $\le 300$ tokens of text chain-of-thought | Receives prompt + reasoning text, generates answer | $\approx 3.7\text{ s}$ producer time |
| **`L8-raw`** | 8 latent steps with raw $h_L$ feedback | Receives 8 raw $f32$ latent vectors as prefix embeddings, answers | $\approx 0.08\text{ s}$ producer time |
| **`L8-soft`** | 8 latent steps with **realigned** expected embedding: $e = \text{softmax}(W_{head} \cdot h_L) \cdot W_{emb}$ | Receives 8 realigned $f32$ vectors, answers | $\approx 0.09\text{ s}$ producer time |
| **`L32-soft`** | 32 realigned latent steps | Receives 32 realigned $f32$ vectors, answers | $\approx 0.35\text{ s}$ producer time |

---

## Evaluation Benchmark Suite

1. **Structured JSON Extraction (20 items, grammar-constrained)**:
   - Prompts requiring extracting entities, tool call arguments, and nested data into strict JSON schemas from `grammar/corpus/`.
   - Verified with the native `Automaton` / `Matcher` grammar engine.
   - Metric: `schema-valid AND exact-match correct`.
2. **Deterministic Multi-Step Arithmetic (20 items)**:
   - Checkable grade-school word problems with integer solutions.
   - Metric: Exact numerical answer extraction.

Total: 40 checkable items per arm $\times$ 5 arms $\times$ 2 topologies.

---

## Proposed Changes

### 1. Realigned Expected Embedding Kernel
Implement $e = \text{softmax}(W_{head} \cdot h_L) \cdot W_{emb}$ in `mojo-baro`:
- Take post-norm $h_L \in \mathbb{R}^H$.
- Apply LM head dot products to get top-K / full logits.
- Softmax over logits to get probabilities $p$.
- Gather and accumulate weighted embeddings into $e \in \mathbb{R}^H$.

### 2. Multi-Agent Benchmark Harness
- [NEW] [bench/bench_latent_handoff.mojo](file://$HOME/Projects/mojo-baro/bench/bench_latent_handoff.mojo): Full 5-arm evaluator supporting dual-engine co-residency, raw/soft continuous thought generation, and checkable scoring.
- [NEW] [bench/latent-handoff.sh](file://$HOME/Projects/mojo-baro/bench/latent-handoff.sh): Command line wrapper for `gpu-wait run --priority 30 -- bench/latent-handoff.sh`.
- [NEW] [bench/data/e8_tasks.json](file://$HOME/Projects/mojo-baro/bench/data/e8_tasks.json): 40 checkable evaluation items (20 structured JSON + 20 arithmetic).

### 3. Run Receipt
- [NEW] [runs/latent-os/E8-2026-09-09.md](file://$HOME/AMDHQ/runs/latent-os/E8-2026-09-09.md): Formal receipt recording accuracies, wall-clock latencies, speedups, and final verdict on the thesis.

---

## Verification Plan

### Automated Execution
```sh
cd ~/Projects/mojo-baro
gpu-wait run --priority 30 --vram 14 -- bench/latent-handoff.sh
```

### Checks
1. VRAM residency verification (both models loaded concurrently without out-of-memory or PCIe thrashing).
2. Exact-match accuracy comparison across arms `0`, `T`, `L8-raw`, `L8-soft`, `L32-soft`.
3. Wall-clock timing verification (producer time $L8 \approx 0.1\text{ s}$ vs $T \approx 3.7\text{ s}$).
4. Formal gate resolution: determine if `HIDDEN` is admitted training-free, speed-only, or killed.
