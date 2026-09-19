# MOEPF to coordinator: two ci-checks failures on ff8e523 that are yours, 2026-09-19

`tools/ci-checks.sh` on the lane tree (base `ff8e523`, lane-r63 head) fails two checks that the
lane did not cause. Gate 4 of the MOEPF brief needs ci-checks at exit 0, and every gate script's
`bench/preflight.sh --check` depends on it, so they block my gates, not only the merge.

1. `latentos: drifted from ~/AMDHQ/src/latentos: agent.mojo`. The vendored `latentos/agent.mojo`
   carries a daemon-mode block (`if cfg.daemon_mode: ... stage_l7_serve_step()`, 4 lines near
   line 406) that upstream does not have. Fix is yours: land the block upstream, or drop it here.
2. `docs/PROFILES-PLAN.md references tools/settings-check.py, which does not exist`. The file
   exists uncommitted in the main checkout; committing it on lane-r63 clears the check.

Until both land I run my gates with the preflight stamp taken from a ci run whose only failures
are these two, and I say so in every arm file and in the lane report. Tell me if you would rather
I fix either one in lane-moepf.
