#!/usr/bin/env python3
"""Task eval, both arms. bench/quality-protocol.md. Sends bench/data/e8_tasks.json
items as chat messages to two already-running servers (baro-serve and
llama-server, both OpenAI-compatible /v1/chat/completions), T=0, and scores
with bench/e8_score.py's own exact-match functions (imported, not
redefined) so this eval path and E8's raw-dump harness never diverge on what
"correct" means.
"""
import argparse
import json
import re
import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).parent))
from e8_score import extract_last_int, parse_json_obj, subset_match  # noqa: E402

THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
FENCE_RE = re.compile(r"^```[a-zA-Z]*\n(.*)\n```$", re.DOTALL)
OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)


def strip_for_scoring(text, is_json):
    t = THINK_RE.sub("", text).strip()
    m = FENCE_RE.match(t)
    if m:
        t = m.group(1).strip()
    if is_json:
        m2 = OBJ_RE.search(t)
        if m2:
            t = m2.group(0)
    return t


def sys_prompts(tasks):
    out = {}
    for t in tasks:
        if t["type"] in out:
            continue
        fp = t["full_prompt"]
        m = re.search(r"<\|im_start\|>system\n(.*?)<\|im_end\|>", fp, re.DOTALL)
        if m:
            out[t["type"]] = m.group(1)
    return out


def ask(url, sysmsg, user, max_tokens):
    body = {
        "model": "local",
        "messages": [{"role": "system", "content": sysmsg}, {"role": "user", "content": user}],
        "temperature": 0,
        "max_tokens": max_tokens,
    }
    r = requests.post(url.rstrip("/") + "/v1/chat/completions", json=body, timeout=180)
    r.raise_for_status()
    d = r.json()
    text = d["choices"][0]["message"]["content"]
    usage = d.get("usage", {})
    return text, usage


def score_one(task, text):
    scored = strip_for_scoring(text, task["type"] == "json")
    if task["type"] == "math":
        got = extract_last_int(scored)
        want = task["expected"]
        ok = got is not None and got == want
        return ok, ok, ("" if ok else f"got {got} want {want}")
    if task["type"] == "json":
        parsed = parse_json_obj(scored)
        if parsed is None:
            return False, False, "did not parse as JSON"
        exact = parsed == task["expected"]
        subset = subset_match(parsed, task["expected"])
        return exact, subset, ("" if exact else f"parsed {parsed!r} vs {task['expected']!r}")
    raise ValueError(task["type"])


def run_arm(name, url, tasks, sys_by_type, out_dir, max_tokens):
    items = []
    exact = subset = 0
    first_usage = None
    for i, t in enumerate(tasks):
        try:
            text, usage = ask(url, sys_by_type[t["type"]], t["prompt_text"], max_tokens)
        except Exception as e:  # noqa: BLE001
            items.append({"id": t["id"], "type": t["type"], "error": str(e)})
            continue
        if i == 0:
            first_usage = usage
        ok, sub, reason = score_one(t, text)
        exact += ok
        subset += sub
        items.append({"id": t["id"], "type": t["type"], "text": text, "correct_exact": ok,
                       "correct_subset": sub, "reason": reason, "usage": usage})
    result = {"arm": name, "url": url, "n": len(tasks), "exact": exact, "subset": subset,
              "exact_pct": 100.0 * exact / len(tasks), "first_usage": first_usage, "items": items}
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(out_dir) / f"{name}.json", "w") as f:
        json.dump(result, f, indent=2)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ours-url", required=True)
    ap.add_argument("--llama-url", required=True)
    ap.add_argument("--tasks", default="bench/data/e8_tasks.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-tokens", type=int, default=300)
    args = ap.parse_args()

    tasks = json.loads(Path(args.tasks).read_text())
    sys_by_type = sys_prompts(tasks)

    ours = run_arm("ours", args.ours_url, tasks, sys_by_type, args.out, args.max_tokens)
    llama = run_arm("llama", args.llama_url, tasks, sys_by_type, args.out, args.max_tokens)

    delta = ours["exact_pct"] - llama["exact_pct"]
    print(f"ours: {ours['exact']}/{ours['n']} ({ours['exact_pct']:.1f}%)  "
          f"llama: {llama['exact']}/{llama['n']} ({llama['exact_pct']:.1f}%)  delta_pp={delta:.1f}")
    if ours["first_usage"] and llama["first_usage"]:
        ou = ours["first_usage"].get("prompt_tokens")
        lu = llama["first_usage"].get("prompt_tokens")
        if ou is not None and lu is not None and ou != lu:
            print(f"NOTE: first-task prompt_tokens differ: ours={ou} llama={lu} (template mismatch, read back not assumed)")


if __name__ == "__main__":
    main()
