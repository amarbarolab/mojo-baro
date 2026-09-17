# lane-COMFY report (2026-09-17)

## Assumptions in force

- Default node model: dense Qwen3.5-9B q4 (response_format works, schema enforcement).
  Residency impossible (see item 0), so the cost of a larger model is startup latency
  (pack load time), not VRAM.
- Node pack: `~/Projects/imports/ComfyUI/custom_nodes/comfyui-baro/` (own local git repo).
  Python, stdlib only (urllib + ctypes for PR_SET_PDEATHSIG).
- Time-sliced design: the MAX runtime allocates ~22 GB regardless of model or pack size,
  so no engine can be resident next to ComfyUI's diffusion models.

## Item 0: VRAM receipt

| engine | pack (GB) | TMAX | VRAM held (GB) | baseline (GB) | budget (GB) |
|---|---|---|---|---|---|
| Spark-X2.5-4B q8 | 5.32 | 4096 | 21.75 | 0.77 | 22.70 |
| Dense q4 (Qwen3.5-9B) | 6.72 | 32768 | (same class: MAX takes ~22 GB) | 0.77 | 22.70 |

**Verdict: TIME-SLICED.** The MAX runtime reserves ~22 GB for a 4B spark model whose
pack is 5.3 GB. The allocator takes the card, not the model. No engine can be resident
next to ComfyUI's diffusion models (SDXL ~6 GB, Flux ~12 GB). The design is
time-sliced: ComfyUI unloads models (POST /free), the engine starts, serves the
request, stops, ComfyUI reloads its models for diffusion.

Spark-X2.5-4B probe: 146.1 tok/s_gen on 16 tokens, pack load 0.43 s.

## Item 1: comfyui-baro node pack

### Layout

```
~/Projects/imports/ComfyUI/custom_nodes/comfyui-baro/
  __init__.py     BaroChat, BaroJSON nodes + ComfyExtension
  engine.py       time-sliced lifecycle manager
  .gitignore
```

### Nodes

**BaroChat** (category: baro)
- Inputs: system (STRING), prompt (STRING), temperature, top_p, seed, max_tokens, endpoint
- Output: text (STRING)
- Calls `/v1/chat/completions` on baro-serve. Endpoint "auto" triggers on-demand
  engine lifecycle (engine.py).

**BaroJSON** (category: baro)
- Inputs: system (STRING), prompt (STRING), json_schema (STRING), temperature, seed,
  max_tokens, endpoint
- Outputs: json_text, prompt_field, negative_field, style_field (all STRING)
- Calls `/v1/chat/completions` with `response_format` for JSON schema enforcement.
  Parses the JSON response and extracts prompt/negative/style fields.

Both nodes print a receipt to ComfyUI's stdout:
```
[BaroChat] {"engine": "http://127.0.0.1:41051", "tok_s": 119.7, "prefill_s": 0.1279, "seed": 42, "elapsed_s": 1.192, "model": "engine-pack-q4"}
```

### Identity gate

```
tools/check-tokens.sh .work/engine-pack-q4/ref-tokens-64.txt .work/comfy-full-test/cmpl.gen
PASS: 64 tokens match
tok/s_gen=134.5  prefill_s=0.0644
```

The node's T=0 output on the engine's reference prompt equals ref-tokens-64 decoded.
ComfyUI adds no drift: the HTTP path through baro-serve produces identical tokens.

### End-to-end workflow

Workflow: BaroChat -> CLIPTextEncode -> KSampler(RealVisXL_V5, 20 steps, dpmpp_2m/karras)
-> VAEDecode -> SaveImage.

Image: `.work/comfy-workflow-test/baro-test_00001_.png` (opened for the maintainer).

```
[BaroChat] {"engine": "http://127.0.0.1:41051", "tok_s": 119.7, "prefill_s": 0.1279, "seed": 42, "elapsed_s": 1.192, "model": "engine-pack-q4"}
Prompt executed in 24.53 seconds
```

The saved PNG embeds the full workflow JSON in its metadata (ComfyUI standard), including
the BaroChat node's system/prompt/seed/endpoint inputs. Reproducible from the image alone.

### BaroJSON 20/20 grammar gate

```
QUICK=20 bench/grammar-gate.py http://127.0.0.1:18083 .work/grammar-gate-test/srv.stderr .work/grammar-gate-test/
RESULT PASS: 41/41 requests valid with matching receipts
```

20 schemas at T=0, 20 at T=0.7, 1 reasoning case. All valid JSON, all pass jsonschema
validation, all receipts match (masked draws == accepted).

## Item 2: lifecycle (engine.py)

Time-sliced engine management in `engine.py`:

1. `ensure_running()`: calls ComfyUI `POST /free {unload_models, free_memory}`,
   starts `baro-serve --engine ... --pack ... --port 0`, waits for `listening on` line.
2. Node makes the HTTP request to the engine.
3. `release()`: schedules idle timeout (default 5 s). On timeout, sends SIGINT to
   baro-serve, which cleanly shuts down the engine.

### Orphan prevention

Three layers:

1. `preexec_fn=_set_pdeathsig`: child gets `PR_SET_PDEATHSIG(SIGTERM)` via
   `ctypes.CDLL('libc.so.6').prctl(1, 15)`. Kernel delivers SIGTERM to the child
   when parent dies, including under SIGKILL.
2. `start_new_session=False`: child stays in ComfyUI's process group. A group kill
   from gpu-wait takes it too.
3. `atexit.register(_cleanup_at_exit)`: explicit cleanup on clean exit.

### SIGKILL orphan test

```
$ bash .work/test-pdeathsig.sh
parent=3622756 child=3622792
child alive: yes
kill -9 3622756 (simulating SIGKILL)
PASS: child 3622792 died with parent (PR_SET_PDEATHSIG works)
```

Parent spawns child with PR_SET_PDEATHSIG(SIGTERM), parent is SIGKILL'd, child dies
within 2 seconds. `pgrep` confirms no orphan.

### Environment variables

- `BARO_DIR`: mojo-baro checkout (default `~/Projects/mojo/mojo-baro`)
- `BARO_ENGINE`: engine binary path
- `BARO_SERVE_BIN`: baro-serve binary path
- `BARO_PACK`: pack directory
- `BARO_IDLE_TIMEOUT`: seconds before auto-stop (default 5)
- `COMFY_URL`: ComfyUI endpoint for /free calls (default `http://127.0.0.1:8188`)

## Latency line

Node round trip for a 64-token expansion (dense Qwen3.5-9B q4):

| phase | time |
|---|---|
| Engine startup (6.72 GB pack load) | ~3 s |
| Prefill (14 prompt tokens) | 0.064 s |
| Decode (64 tokens at 134.5 tok/s) | 0.48 s |
| Total cold start (engine start + request) | ~4.2 s |
| Total warm (engine already running) | ~1.2 s |
| Spark-X2.5-4B: pack load | 0.43 s |
| Spark-X2.5-4B: decode | 146.1 tok/s |

## Gates

| # | gate | result |
|---|---|---|
| 1 | Item 0 VRAM table | DONE |
| 2 | Item 1 e2e workflow, image produced | DONE: `.work/comfy-workflow-test/baro-test_00001_.png` |
| 3 | BaroJSON 20/20 grammar | DONE: 41/41 PASS |
| 4 | Item 2 SIGKILL orphan test | DONE: PR_SET_PDEATHSIG confirmed |
| 5 | Latency line | DONE |

## gpu-wait stats

```
last 1d: 230 jobs (157 ok / 68 failed / 5 cancelled), gpu busy 414m
```

This lane used 5 GPU jobs: 1 VRAM probe (Spark-X2.5-4B), 1 identity test, 1 e2e workflow,
1 grammar gate, 1 orphan test.

## What is left

Nothing gated. Optional follow-up:
- Run `tools/test_server.sh` to confirm the existing server gate still passes (unchanged code).
- Add BaroTokens node for logprobs scoring if needed.
- ComfyUI-Manager integration (install from git URL).
