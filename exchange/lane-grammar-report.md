# lane-grammar report (2026-09-08)

Branch `lane-grammar` (worktree `~/Projects/mojo-baro-lanes/grammar`), 3 commits over main 5ae2925:
`3710085` package · `75aa668` tests/corpus/tools · `05bd00c` docs/grammar.md. Touches only `grammar/` and `docs/grammar.md`; no engine patch needed.

## What passes (all CPU, `./.venv/bin/mojo build grammar/test_X.mojo -I . -o .work/test_X`)
| test | item | result |
|---|---|---|
| test_corpus | 5a | 32 schemas / 3200 samples / 0 failures (Python jsonschema oracle via grammar/tools) |
| test_accept_known_good | 5b | PASS (real tokenizer, boundary-straddling tokens `"}` `],` `":`) |
| test_rollback | 5c | PASS (masks identical after snapshot/rollback/replay) |
| test_reasoning_boundary | 5d | PASS (grammar attached after `</think>`, draft-window token seen once) |
| test_timing | 5e | median **11.5 µs** / budget 500 µs, PASS |

Timing detail (schema 27, 2667 states, 1000 samples after warmup, idle machine): median 11480 ns, p90 27.9 ms, p99 46.6 ms, worst 74 ms. The tail is the first visit of each (rule, state): one full trie walk over the 248320-token vocab to build the cached context-independent bitset; revisits are a bitset OR. If p99 matters for TTFT on fresh schemas, next lever is precomputing all (rule,state) bitsets at compile time (~2667 walks ≈ 30–70 s worst; or lazily in a background thread).

## API
As frozen in `docs/grammar.md`: `compile_root_schema(Automaton, JSONDoc) -> rule_id`, `Matcher(automaton, trie, vocab)` with `fill_mask(Bitset)`, `accept(token) -> Bool`, `is_terminated()`, `snapshot()/rollback()`. Deviation from brief: snapshot/rollback is O(stack depth), not O(1) (bounded for tool-call JSON; timing shows it is not on the critical path).

## Unsupported (compiler raises, naming the keyword)
`$ref`/`$defs`/`definitions`, `anyOf`/`oneOf`/`allOf`/`not`/`if-then-else`, `patternProperties`, `additionalProperties` other than `false`, `format`/`contentEncoding`/`contentMediaType`, numeric ranges (`minimum`/`maximum`/`exclusive*`/`multipleOf`), union `type`, arrays without `items`. Regex: no backrefs/lookaround/non-greedy; `pattern` is full-string anchored (`^`/`$` stripped); pattern bytes are literal JSON-string content (no `"`/`\` inside patterns). Default bounds when absent: strings 0–24 chars, arrays 0–10 items.

Corpus note: 33 files in `grammar/corpus/` = 32 schemas + manifest.txt; `grammar/corpus/unsupported/` was not needed (every corpus schema compiles).

DONE
