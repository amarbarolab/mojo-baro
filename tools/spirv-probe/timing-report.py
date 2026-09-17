"""Tables and tok/s ceilings from a `time.sh run` log. usage: timing-report.py <run.log>"""
import re, sys

log = open(sys.argv[1]).read()
ours, shape = {}, None
for l in log.splitlines():
    m = re.search(r"shapes R=(\d+) H=\d+ V=(\d+) GN=(\d+) GK=(\d+)", l)
    if m:
        shape = dict(R=int(m.group(1)), V=int(m.group(2)), N=int(m.group(3)), K=int(m.group(4)))
    m = re.match(r"TIME (\w+) .*?sync_us med ([\d.]+) min ([\d.]+) max ([\d.]+) \| batch_us(?:\(B=(\d+)\) med ([\d.]+) min ([\d.]+) max ([\d.]+))?", l)
    if m:
        k = m.group(1)
        key = (k, shape["R"]) if "rmsnorm" in k else (k,) if "softmax" in k else (k, shape["N"], shape["K"])
        ours[key] = dict(sync=float(m.group(2)), smin=float(m.group(3)), smax=float(m.group(4)),
                         batch=float(m.group(6)) if m.group(6) else None, B=m.group(5))
ref = {1: {}, 0: {}}
for m in re.finditer(r"G0 (\w+)\((.*?)\) ops_per_graph=(\d+) graphs=(\d+) us_per_op med ([\d.]+) min ([\d.]+) max ([\d.]+)", log):
    op, v, per, graphs, med = m.group(1), m.group(2), int(m.group(3)), int(m.group(4)), float(m.group(5))
    if op == "MUL_MAT":
        key = ("gemv", int(re.search(r"\bm=(\d+)", v).group(1)), int(re.search(r"\bk=(\d+)", v).group(1)))
    else:
        ne = re.search(r"ne=\[(\d+),(\d+)", v)
        key = ("rms", int(ne.group(2))) if op == "RMS_NORM" else ("softmax",)
    ref[1 if per == 1 else 0][key] = dict(med=med, per=per, graphs=graphs)

def gemv(n, k):
    return ours[("amar_matmul_skinny_q4rowb", n, k)]["batch"] + ours[("amar_skinny_reduce", n, k)]["batch"]

print("### rmsnorm, H = 4096 (us)\n")
print("| rows | ours sync med (min to max) | ours batch | host-reduction sync med (min to max) | host-reduction / lowered, med | same, min | reference sync | reference batched | ours batch / reference batched |")
print("|---|---|---|---|---|---|---|---|---|")
for r in (1, 2, 4, 8, 16, 32, 64):
    a, h = ours[("amar_rmsnorm", r)], ours[("amar_rmsnorm_hostreduce", r)]
    print(f"| {r} | {a['sync']:.0f} ({a['smin']:.0f} to {a['smax']:.0f}) | {a['batch']:.1f} | {h['sync']:.0f} ({h['smin']:.0f} to {h['smax']:.0f}) | {h['sync'] / a['sync']:.2f} | {h['smin'] / a['smin']:.2f} | {ref[1][('rms', r)]['med']:.0f} | {ref[0][('rms', r)]['med']:.1f} | {a['batch'] / ref[0][('rms', r)]['med']:.2f} |")
s = ours[("amar_softmax_rows",)]
print(f"\n### softmax, one row of 151936 (us)\n\n| ours sync | ours batch | reference sync | reference batched | ours batch / reference batched |\n|---|---|---|---|---|")
print(f"| {s['sync']:.0f} | {s['batch']:.0f} | {ref[1][('softmax',)]['med']:.0f} | {ref[0][('softmax',)]['med']:.0f} | {s['batch'] / ref[0][('softmax',)]['med']:.2f} |")
print("\n### q4 GEMV, one token (us)\n")
print("| N x K | ours q4rowb sync | ours q4rowb batch | ours reduce batch | ours pair batch | reference sync | reference batched | pair / reference | q4rowb alone / reference | our weight GB/s |")
print("|---|---|---|---|---|---|---|---|---|---|")
for key in [k for k in ours if k[0] == "amar_matmul_skinny_q4rowb"]:
    _, n, k = key
    q, rd, rf = ours[key], ours[("amar_skinny_reduce", n, k)], ref[0][("gemv", n, k)]
    gbs = n * (k // 2 + k // 32 * 2) / (q["batch"] / 1e6) / 1e9
    print(f"| {n} x {k} | {q['sync']:.0f} | {q['batch']:.1f} | {rd['batch']:.1f} | {gemv(n, k):.1f} | {ref[1][('gemv', n, k)]['med']:.0f} | {rf['med']:.1f} | {gemv(n, k) / rf['med']:.2f} | {q['batch'] / rf['med']:.2f} | {gbs:.1f} |")
print("\nReference only, Qwen2.5-0.5B true widths (ours cannot run them): " + ", ".join(
    f"{k[1]} x {k[2]} {v['med']:.1f}" for k, v in ref[0].items() if k[0] == "gemv" and k[2] in (896, 4864)) + " us batched.")

MODELS = {  # layers, q, kv, o, gate_up, down, head as (N, K)
    "Qwen2.5-0.5B, ours at padded widths": (24, (896, 1024), (128, 1024), (896, 1024), (4864, 1024), (896, 5120), (151936, 1024)),
    "Qwen2.5-0.5B, true widths (reference only)": (24, (896, 896), (128, 896), (896, 896), (4864, 896), (896, 4864), (151936, 896)),
    "Qwen3-0.6B": (28, (2048, 1024), (1024, 1024), (1024, 2048), (3072, 1024), (1024, 3072), (151936, 1024)),
    "Qwen3-1.7B": (28, (2048, 2048), (1024, 2048), (2048, 2048), (6144, 2048), (2048, 6144), (151936, 2048)),
}
print("\n### tok/s ceiling: L x (2 rmsnorm + q + 2 kv + o + 2 gate_up + down) + rmsnorm + head, batched medians\n")
print("| model | ours us/token | ours ceiling tok/s | head share of ours | reference us/token | reference op ceiling tok/s | ours / reference |")
print("|---|---|---|---|---|---|---|")
for name, (L, q, kv, o, gu, dn, hd) in MODELS.items():
    def total(g, rms):
        return L * (2 * rms + g(*q) + 2 * g(*kv) + g(*o) + 2 * g(*gu) + g(*dn)) + rms + g(*hd)
    rt = total(lambda n, k: ref[0][("gemv", n, k)]["med"], ref[0][("rms", 1)]["med"])
    if ("amar_matmul_skinny_q4rowb",) + q in ours:
        ot = total(gemv, ours[("amar_rmsnorm", 1)]["batch"])
        print(f"| {name} | {ot:.0f} | {1e6 / ot:.1f} | {100 * gemv(*hd) / ot:.0f}% | {rt:.0f} | {1e6 / rt:.1f} | {rt / ot:.2f} |")
    else:
        print(f"| {name} | n/a | n/a | n/a | {rt:.0f} | {1e6 / rt:.1f} | n/a |")
