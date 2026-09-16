# mojo-baro as a ComfyUI engine (plan, 2026-09-17)

An LLM node inside ComfyUI workflows (prompt expansion, captioning of text inputs, schema-enforced
JSON for downstream nodes), backed by our engine, sharing the one 24 GB card with the diffusion
models ComfyUI holds. ComfyUI: `~/Projects/imports/ComfyUI` (custom nodes: controlnet_aux,
IPAdapter_plus, Manager; no LLM node today). Our side: `baro-serve` already speaks OpenAI
`/v1/chat/completions` and `/v1/completions` with sampling, logprobs and `response_format`
(JSON schema enforced on the dense/MoE engine; spark profiles refuse it with 400), 5.5 ms of
serving overhead per request (A0.1).

## The one fact that decides the shape

BASELINE: one dense engine's MAX runtime reserves about 23.5 to 24.8 GB, so a second engine does
not fit; the dense q4 pack itself is 6.8 GB plus 2 GB of KV at 32k. Whether that reservation is
the MAX allocator taking the card or our own allocation is not on record, and it decides whether
the engine can be RESIDENT next to ComfyUI's models or must be TIME-SLICED (ComfyUI unloads, the
engine runs, exits). The queue already has both primitives: `gpu-wait run --shared --vram N` for
residents and leases, `gpu-wait gpu` for `budget_bytes` and `declared_bytes`
(`~/Projects/gpu-media/gpuwaitingroom/README.md`). ComfyUI has `POST /free {unload_models,
free_memory}`.

## Item 0: VRAM receipt of a small engine (S, one GPU job, no code)

Start `serve/spark.mojo` on Qwen2.5-7B q4 (`~/Models/qwen2.5-7b-instruct-gguf`, the pack from the
sampling lane) and on Llama-3.2-1B under `gpu-wait run --shared --vram 8`, serve one request, and
read back: `rocm-smi --showmeminfo vram` used bytes while resident, `gpu-wait gpu` declared vs
budget, and the same for the dense q4 engine. Done: a table of engine, pack bytes, KV bytes at
its TMAX, VRAM actually held. If a 7B spark engine holds under 8 GB, residency is the design; if
MAX takes the card regardless, time-slicing is.

## Item 1: `comfyui-baro` custom node pack (S, sonnet; Python because ComfyUI is Python)

Directory `~/Projects/imports/ComfyUI/custom_nodes/comfyui-baro/` (own repo, local git), stdlib
`urllib` only, three nodes:
- `BaroChat`: system, prompt, temperature, top_p, seed, max_tokens, endpoint (default
  `http://127.0.0.1:8080`); returns STRING. Streams nothing; one request per execution.
- `BaroJSON`: same plus a JSON schema string; sends `response_format` and returns the parsed
  object's fields as STRINGs (a downstream node picks `prompt`, `negative`, `style`). Dense/MoE
  engine only; the node surfaces the 400 from a spark engine as a node error.
- `BaroTokens` (optional): logprobs of the chosen text, for scoring prompts; only if item 2 needs it.
Every node writes the request and the engine's done line (tok/s, prefill, seed) into the node's
output metadata, so a saved workflow carries its own receipt.
Done: a workflow `BaroChat -> CLIPTextEncode -> KSampler -> SaveImage` run through `comfy-run`
(the way a user meets it) produces an image whose sidecar names the expanded prompt; the node's
T=0 output on the engine's reference prompt equals `ref-tokens-64` decoded (`tools/check-tokens.sh`),
so ComfyUI adds no drift.

## Item 2: lifecycle (M, sonnet host code; opus only if the queue needs a change)

`BaroEngine` node (or a sidecar script the pack calls): starts the engine on demand through
`gpu-wait run --shared --vram <item 0's number> -- baro-serve --engine ... --port 0`, reads
`listening on`, keeps the handle for the ComfyUI process, stops it after an idle timeout. In
time-sliced mode it calls ComfyUI's `/free` first and runs the engine without `--shared`. Done: a
workflow that starts cold, runs an image job and the LLM node in one graph, and `gpu-wait gpu`
never shows declared bytes above budget; three repeated runs, identical images at fixed seeds.

## Gates that close the plan

1. Item 0's table, before any node code (P14: the design rests on a measured number).
2. Item 1's end-to-end workflow through `comfy-run`, image produced, sidecar carries the prompt.
3. `BaroJSON`: 20 schema requests from the grammar corpus through the node, 20/20 valid
   (`bench/grammar-gate.py` shape).
4. Item 2's concurrency run, plus the engine's own `tools/test_server.sh` unchanged.
5. Latency line in the report: node round trip for a 64-token expansion on the chosen model.

## Decisions for the maintainer

- Default model for the node: Qwen2.5-7B coder (JSON and prompts, spark profile, no
  `response_format`), Spark-X2.5-4B, or the dense Qwen3.5-9B (schema enforcement, 22 GB, ComfyUI
  must unload). Item 0 tells which of these can be resident.
- Resident or time-sliced, after item 0.
- Whether the node pack lives in `~/Projects/imports/ComfyUI/custom_nodes/` (ComfyUI-Manager can
  see it) or in mojo-baro under `serve/comfy/` with a symlink.
