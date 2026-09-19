# Profiles experiment: RegesCore builds its own profile manager

Opened 2026-09-19 (the maintainer). The local model does the build through DeerFlow;
Claude writes each task, runs the gate, and logs. One step at a time: a step
starts only after the previous gate passed.

## What gets built

1. A settings template listing every setting mojo-baro offers (all `BARO_*`
   env reads plus comptime `-D` build flags), per model family, with type,
   default, effect and the receipt that verifies it.
2. Named profiles for every mojo-baro model, applied by
   `tools/baro run|serve --profile NAME` instead of hand-set env.
3. A profile manager: a local web page, one card per profile; clicking a card
   restarts the served engine with that profile on the same port.

## Roles

- **Builder:** RegesCore-1.0-35B served by mojo-baro (`tools/baro serve`),
  driven through DeerFlow (`~/Projects/ai-models/deer-flow`, model
  `regescore-35b`, port 8099), every call recorded by `tools/openai-tap.py`.
- **Orchestrator (Claude):** writes each step's task, runs the gate, reviews
  the diff, commits only on a passing gate. Never edits builder output to make
  a gate pass; a fix goes back to the builder as a new turn, and is logged.
- **Builder prompt:** an identity line plus a task system prompt, hand-written
  first, then trained with SkillOpt (`~/Models/.MT/SkillOpt`, wiring after
  `~/Projects/gpu-media/vidcode/research/skillopt_bridge/`) once enough gated
  episodes exist for a train/held-out split (step 6).

## The builder runs with experts in host RAM, for context

MoE KV is f32 only (int8 KV is dense-only), 40,960 bytes per token. Resident
experts (21 GB pack plus about 4 GB of fixed buffers) fill the 24 GB card:
resident mode leaves almost no room for context, and on 2026-09-19 it starved
the desktop compositor (framebuffer pin ENOMEM, requests at 0.4 tok/s). The
builder therefore runs experts in host RAM (`BARO_TIER`, `BARO_TIER_PINNED=1`,
`BARO_TIER_ZC=1`, 72.02 tok/s measured) with KV sized to the model's native
262,144 tokens (about 10.7 GB) if step 0 shows it fits, else the largest size
that does. Prefix checkpoints stay on, because DeerFlow resends a system prompt
of about 8k tokens on every call. The cost is decode speed: 72 against 112.

## Steps and gates

| step | work | who | gate (the way a user meets it) |
|---|---|---|---|
| 0 | Serve RegesCore with the builder profile; find the largest `BARO_TMAX` that fits with the desktop running | Claude | at least 1 GB VRAM headroom with the desktop active; DeerFlow smoke `write_file -> read_file -> answer` passes through the tap; a needle at the far end of a max-length prompt is answered; decode tok/s recorded |
| 1 | Settings template `docs/settings-template.toml` | RegesCore | `tools/settings-check.py` (Claude writes it first): every `BARO_*` getenv and `-D` flag in source appears in the template and vice versa, each with type, default and doc; exit 0 |
| 2 | Profile format and `--profile` in `tools/baro` | RegesCore | a profile run equals the same env set by hand: identical tokens on 20 prompts; the ready line names the profile |
| 3 | Seed profiles from receipts (RegesCore: champion, mega, builder-maxctx, tiered; Qwythos, Spark and the rest from BASELINE) | RegesCore | each profile serves one 64-token completion; each number links its receipt |
| 4 | Profile manager page and `baro profiles` server | RegesCore | in a browser: cards render, a click restarts the engine, `/v1/models` reports the new profile, screenshot looked at |
| 5 | Review pass and docs | RegesCore + Claude | a fresh clone builds and runs a profile; `run-tests.sh` and `ci-checks.sh` exit 0 |
| 6 | SkillOpt on the builder prompt | Claude | the trained prompt beats the hand prompt on held-out episodes scored by the frozen step gates |

## Logging

- Experiment log, dated and append-only: `~/Brain/mojo/mojo-baro/2026-09-19-profiles-experiment.md`.
- Raw per step in `.work/profiles-exp/stepN/`: the task sent, the tap trace
  (`tap.jsonl`), the builder's diff, gate output, turn count, wall time, tok/s.
- The board card moves in the turn each gate passes or fails.

## What would end the experiment early

Step 0 failing: if RegesCore cannot complete the DeerFlow smoke through
mojo-baro, the builder is not viable on this engine, and the finding is which
serving feature is missing. Builder mistakes in later steps are data, kept in
the log with the turn that fixed them, not a reason to stop.
