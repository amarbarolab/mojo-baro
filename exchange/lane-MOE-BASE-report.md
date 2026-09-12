# W4 RegesCore-35B llama.cpp reference-arm receipt

Status: measured reference arm. The protocol was preregistered in lane commit `0293498` and amended before the retry in `3712efc`. The measurement artifacts are under the lane worktree at `.work/moe-base/run-20260912-03/`.

## Arm receipt

| field | read-back |
|---|---|
| llama.cpp | commit `ca3d5a3e10d53f7ea672cb9b6178faca3e2807bc`; version `0.3.0-dev`, build 10665, GCC 16.2.1 |
| backend | ROCm 7.2.4, AMD HIP/ROCm device, gfx1100 |
| GPU | AMD Navi 31, PCI ID `1002:744c`, RX 7900 XTX class |
| driver | `7.2.4-arch1-2` |
| CPU / OS | x86_64 Linux, EndeavourOS, kernel `7.2.4-arch1-2`, 8 llama.cpp threads |
| model | RegesCore-1.0-35B-UD-Q4_K_S-BARO.gguf, 20,893,070,752 bytes, GGUF Q4_K - Small |
| model sha256 prefix | `02d1fa2e4d35c038` |
| architecture | `qwen35moe` |
| context | 4096 tokens, server log `n_ctx_slot = 4096` |
| batch / ubatch | 2048 / 512 |
| GPU layers | 99 requested and loaded on the single gfx1100 device |
| KV cache | f16 K, f16 V |
| slots / batching | 1 slot, continuous batching disabled |
| sampling | temperature 0, top_k 1, seed 1, ignore_eos true |
| generation | 64 tokens per response, verified by response `predicted_n = 64` |
| cache state | warm only: one server load, then page cache and VRAM resident; `cache_n = 0` on measured requests |

The model hash prefix was captured by the run in `.work/moe-base/run-20260912-03/model.sha256.prefix`; the full model remains read-only and was not copied or converted.

## Verbatim invocation

```text
gpu-wait run --priority 20 --preemptible --vram 22 -- bash bench/moe-baseline.sh .work/moe-base/run-20260912-03 18082
```

The script's server invocation was:

```text
$HOME/llama.cpp/build/bin/llama-server -m $HOME/Models/RegesCore-1.0-35/RegesCore-1.0-35B-UD-Q4_K_S-BARO.gguf -c 4096 -ngl 99 -fa on -ctk f16 -ctv f16 -b 2048 -ub 512 -t 8 -np 1 --no-cont-batching --host 127.0.0.1 --port 18082
```

Each prompt came from `bench/mtp-prompts/p*.txt`, not the sibling `.tokens` file. The request used `n_predict=64`, `temperature=0`, `top_k=1`, `seed=1`, `ignore_eos=true`, `cache_prompt=false`, and `return_tokens=true`.

One untimed warmup and five measured sequential requests were made per prompt. Decode and prefill values below are medians of the five measured responses for that prompt. Decode is llama.cpp `timings.predicted_per_second`; prefill is `timings.prompt_per_second`.

## Per-prompt results

| prompt | decode tok/s | prefill tok/s |
|---|---:|---:|
| p01-water | 109.413 | 268.346 |
| p02-python-fib | 109.322 | 301.392 |
| p03-story | 109.402 | 418.682 |
| p04-list-planets | 109.556 | 344.086 |
| p05-math | 109.707 | 307.730 |
| p06-translate | 109.847 | 391.222 |
| p07-json | 109.465 | 420.142 |
| p08-sql | 109.562 | 459.380 |
| p09-explain-gpu | 109.537 | 391.720 |
| p10-recipe | 109.699 | 273.407 |
| p11-email | 109.367 | 298.424 |
| p12-rust | 109.444 | 546.541 |
| p13-haiku | 109.262 | 227.078 |
| p14-history | 109.248 | 187.041 |
| p15-bash | 109.320 | 368.324 |
| p16-chat | 109.388 | 335.951 |
| p17-summarize | 109.463 | 745.816 |
| p18-regex | 109.446 | 266.150 |
| p19-numbers | 109.113 | 440.610 |
| p20-dialog | 109.623 | 290.627 |

## Aggregate

- Decode: **109.445 tok/s median across the 20 prompt medians**, range **109.113 to 109.847 tok/s**.
- Prefill: **340.018 tok/s median across the 20 prompt medians**, range **187.041 to 745.816 tok/s**.
- Stability: decode prompt-median spread is 0.67% from min to max.
- Raw receipt: 100 measured responses in `raw.json`; summarized values in `results.json`.

## Void attempt and fairness notes

The first launch, `.work/moe-base/run-20260912-02/`, was void. `p13-haiku` ended after one generated token and reported zero decode tok/s. No number from that launch is used. The protocol was amended before retry to set `ignore_eos=true` and require `predicted_n=64`.

The successful arm is warm-cache only. A cold-cache comparison must drop relevant page caches and use a separately preregistered protocol; these numbers must not be compared to a cold engine run without that amendment. The engine-side comparison must use this exact model, the same text prompts, 64 generated tokens, greedy sampling, f16 K/V if intended as the same arm, and the same explicit warm or cold cache state. Feeding the `.tokens` files directly would not be the same reference arm because llama.cpp tokenizes the `.txt` input itself.

No engine-side number or prediction is included here.
