#!/usr/bin/env python3
"""LatentOS E12 verdict from a scored E8-harness file (arms 0, T, KV).

Applies the thresholds frozen in ~/AMDHQ/docs/design/latent-os/06-experiments.md
section E12, in order: void conditions, paired quality gate on exact-match
correctness (Wald CI on KV - T), receiver-time gate. Prints the numbers the
receipt needs; decides nothing the preregistration did not.

usage: bench/e12_verdict.py results/e8/<prefix>-merged.json [bench/data/e8_tasks.json]
"""
import json
import math
import statistics as st
import sys


def binom_tail_ge(k, n):
    """P(X >= k) for X ~ Binomial(n, 0.5)."""
    return sum(math.comb(n, i) for i in range(k, n + 1)) / 2 ** n if n else 1.0


def main():
    scored = json.load(open(sys.argv[1]))
    tasks = json.load(open(sys.argv[2] if len(sys.argv) > 2 else "bench/data/e8_tasks.json"))
    plen = {t["id"]: len(t["tokens"]) for t in tasks}

    rows = []
    for it in scored["items"]:
        arms = {a["arm"]: a for a in it["arms"]}
        missing = [a for a in ("0", "T", "KV") if a not in arms]
        if missing:
            sys.exit(f"{it['id']}: missing arms {missing}")
        errs = [f"{k}: {a['error']}" for k, a in arms.items() if a.get("error")]
        if errs:
            sys.exit(f"{it['id']}: arm errors {errs}")
        rows.append((it["id"], arms))
    n = len(rows)

    acc = {a: sum(bool(r[1][a]["correct_exact"]) for r in rows) for a in ("0", "T", "KV")}
    print(f"items {n}  correct: 0 {acc['0']}  T {acc['T']}  KV {acc['KV']}")

    mismatch = [i for i, a in rows if a["T"]["handoff_hash"] != a["KV"]["handoff_hash"]]
    same_ids = sum(a["T"]["generated_ids"] == a["KV"]["generated_ids"] for _, a in rows)
    print(f"handoff_hash T != KV: {len(mismatch)} {mismatch[:5]}")
    print(f"B generated ids identical T vs KV: {same_ids}/{n} ({100 * same_ids / n:.1f}%)")

    lift = 100 * (acc["T"] - acc["0"]) / n
    print(f"precondition T - arm0: {lift:+.1f} pp (void below +3)")

    b = sum(a["KV"]["correct_exact"] and not a["T"]["correct_exact"] for _, a in rows)
    c = sum(a["T"]["correct_exact"] and not a["KV"]["correct_exact"] for _, a in rows)
    d = (b - c) / n
    se = math.sqrt(max((b + c) - (b - c) ** 2 / n, 0.0)) / n
    lo, hi = d - 1.645 * se, d + 1.645 * se
    p_worse = binom_tail_ge(c, b + c)
    print(f"KV - T: {100 * d:+.2f} pp  discordant b={b} c={c}  90% CI [{100 * lo:+.2f}, {100 * hi:+.2f}] pp  "
          f"one-sided p(KV<T)={p_worse:.3g}")

    rt = [a["T"]["receiver_s"] for _, a in rows]
    rk = [a["KV"]["receiver_s"] for _, a in rows]
    r0 = [a["0"]["receiver_s"] for _, a in rows]
    pt = [a["T"]["producer_s"] for _, a in rows]
    pk = [a["KV"]["producer_s"] for _, a in rows]
    ing = [a["KV"]["ingest_s"] for _, a in rows]
    mnt = [a["KV"]["mint_s"] for _, a in rows]
    cot = [a["T"]["handoff_pos"] - plen[i] for i, a in rows]
    mrt, mrk = st.median(rt), st.median(rk)
    print(f"receiver_s median: 0 {st.median(r0):.3f}  T {mrt:.3f}  KV {mrk:.3f}  "
          f"saving {100 * (1 - mrk / mrt):.1f}%  KV faster on {sum(k < t for k, t in zip(rk, rt))}/{n}")
    print(f"producer_s median: T {st.median(pt):.3f}  KV {st.median(pk):.3f}  ratio KV/T {st.median(pk) / st.median(pt):.3f}")
    print(f"ingest_s median {st.median(ing) * 1e3:.1f} ms max {max(ing) * 1e3:.1f} ms  "
          f"mint_s median {st.median(mnt) * 1e3:.1f} ms")
    print(f"T CoT tokens: median {st.median(cot)}  min {min(cot)}  max {max(cot)}  at 300: {sum(x >= 300 for x in cot)}")

    if mismatch or lift < 3:
        print("VERDICT: VOID (" + ("handoff_hash mismatch" if mismatch else "T does not beat arm0 by 3 pp") + ")")
        return
    quality = "PASS" if -0.05 < lo and hi < 0.05 else ("FAIL" if p_worse < 0.05 and c > b else "INCONCLUSIVE")
    speed = "PASS" if mrk < mrt else "FAIL"
    print(f"quality {quality}  speed {speed}")
    if quality == "INCONCLUSIVE" and abs(d) < 0.05:
        r = (b + c) / n
        need = math.ceil((1.645 * math.sqrt(max(r - d * d, 1e-12)) / (0.05 - abs(d))) ** 2)
        print(f"n needed for the CI to fit inside +/-5 pp at this discordance: {need}")
    verdict = "FAIL" if "FAIL" in (quality, speed) else ("PASS" if quality == speed == "PASS" else "INCONCLUSIVE")
    print(f"VERDICT: {verdict}")


if __name__ == "__main__":
    main()
