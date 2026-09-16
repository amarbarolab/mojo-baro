#!/usr/bin/env python3
"""serve/spark.mojo logprobs gate: identity, oracle, forced dump.

  bench/spark-logprob-check.py ENGINE PACK GGUF OUT_DIR

1. T=0 plain request on the bake's baro.run.prompt.tokens reproduces
   baro.run.ref.tokens exactly (the no-logprob path is unchanged).
2. T=0.8 top_k 40 top_p 0.95 seed 11 top_logprobs 8, 24 tokens, rows dumped
   (BARO_DUMP_LOGITS_DIR): each step's chosen and top-8 logprobs match an
   independent numpy nucleus oracle on the raw row, |d| < 1e-4.
3. T=0 top_logprobs 1 with "force": every emitted token equals the forced id
   and one row per step is dumped.
"""
import json, os, subprocess, sys
import numpy as np

eng, pack, gguf, out = sys.argv[1:5]
os.makedirs(out, exist_ok=True)
meta = json.loads(subprocess.run([sys.executable, "tools/gguf-extract.py", gguf, "--meta"], capture_output=True, text=True, check=True).stdout)
prompt = [int(x) for x in meta["baro.run.prompt.tokens"].split()]
ref = [int(x) for x in meta["baro.run.ref.tokens"].split()]


def nucleus(x, T, k, p):
    o = np.argsort(-x, kind="stable"); xs = x[o]; lmax = xs[0]; e = np.exp(xs - lmax)
    mk = k if 0 < k < x.size else x.size
    if p < 1.0:
        zc = float(e[:mk].sum()); w = max(1.0, min(p * zc, zc))
        mk = min(int(np.searchsorted(np.cumsum(e[:mk]), w) + 1), mk)
    ev = np.exp((xs[:mk] - lmax) / T)
    return {int(i): float(np.log(v)) for i, v in zip(o[:mk], ev / ev.sum())}


def run(reqs, dump):
    env = dict(os.environ, BARO_SERVE="1", BARO_PACK=pack)
    if dump:
        env["BARO_DUMP_LOGITS_DIR"] = dump
    p = subprocess.Popen([eng], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, env=env, text=True)
    lines, _ = p.communicate("".join(json.dumps(r) + "\n" for r in reqs), timeout=900)
    res = {}
    for ln in lines.splitlines():
        try:
            d = json.loads(ln)
        except json.JSONDecodeError:
            continue
        if "tok" in d:
            res.setdefault(d["id"], []).append(d)
        elif "error" in d:
            res[d["id"]] = d
    return res


fails = 0
r1 = run([{"id": 1, "prompt": prompt, "n": len(ref), "spec": False}], None)[1]
got = [d["tok"] for d in r1]
ok = got == ref
fails += not ok
print(("PASS" if ok else "FAIL"), f"1 identity: {sum(a == b for a, b in zip(got, ref))}/{len(ref)} T=0 tokens equal baro.run.ref.tokens")

d2 = os.path.join(out, "rows-sampled"); os.makedirs(d2, exist_ok=True)
r2 = run([{"id": 2, "prompt": prompt, "n": 24, "spec": False, "temperature": 0.8, "top_k": 40, "top_p": 0.95, "seed": 11, "top_logprobs": 8}], d2)[2]
json.dump(r2, open(os.path.join(out, "sampled-lines.json"), "w"))
worst = 0.0; bad = 0
for step, d in enumerate(r2):
    x = np.fromfile(os.path.join(d2, f"row-{step}.bin"), dtype=np.float32).astype(np.float64)
    orc = nucleus(x, 0.8, 40, 0.95)
    diffs = [abs(d["logprob"] - orc.get(d["tok"], float("inf")))]
    diffs += [abs(t["logprob"] - orc.get(t["id"], float("inf"))) for t in d["top_logprobs"] if t["logprob"] > -1e29]
    m = max(diffs); worst = max(worst, m); bad += m >= 1e-4
fails += bad > 0 or len(r2) != 24
print(("PASS" if bad == 0 and len(r2) == 24 else "FAIL"), f"2 oracle: {len(r2) - bad}/{len(r2)} steps within 1e-4, max |d| {worst:.2e}")

d3 = os.path.join(out, "rows-forced"); os.makedirs(d3, exist_ok=True)
force = ref[:16]
r3 = run([{"id": 3, "prompt": prompt, "n": 16, "spec": False, "top_logprobs": 1, "force": force}], d3)[3]
rows = len([f for f in os.listdir(d3) if f.endswith(".bin")])
ok = [d["tok"] for d in r3] == force and rows == 16
fails += not ok
print(("PASS" if ok else "FAIL"), f"3 forced: tokens equal force {[d['tok'] for d in r3] == force}, rows dumped {rows}/16")
print("RESULT", "PASS" if fails == 0 else f"FAIL {fails}")
sys.exit(1 if fails else 0)
