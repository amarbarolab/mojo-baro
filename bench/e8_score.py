#!/usr/bin/env python3
"""E8 HARNESS scoring oracle (exchange/e8-lane-plan-2026-09-09.md, item HARNESS).

Reads the raw dump the Mojo binary writes (bench_latent_handoff.mojo: per item,
per arm -- generated ids, decoded answer_text, and, for json tasks, the
schema-valid flag computed in Mojo via grammar/'s Automaton/Matcher) plus
bench/data/e8_tasks.json (expected answers), and fills in `correct`:
  math -> last integer in answer_text == expected
  json -> schema_valid AND parsed answer_text == expected (exact match)
Never launches the GPU; never recomputes the Mojo side's schema check.
Writes the final scored JSON and a markdown summary table.
"""
import argparse
import json
import re
import statistics
import sys
from pathlib import Path

LAST_INT_RE = re.compile(r"-?\d+")


def extract_last_int(text):
    matches = LAST_INT_RE.findall(text)
    if not matches:
        return None
    return int(matches[-1])


def extract_json_obj(text):
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    lo = text.find("{")
    hi = text.rfind("}")
    if lo < 0 or hi <= lo:
        lo = text.find("[")
        hi = text.rfind("]")
    if lo < 0 or hi <= lo:
        return None
    try:
        return json.loads(text[lo : hi + 1])
    except json.JSONDecodeError:
        return None


def score_arm(task, arm):
    if arm.get("error"):
        return False, "arm error: " + arm["error"]
    text = arm.get("answer_text", "")
    if task["type"] == "math":
        got = extract_last_int(text)
        want = task["expected"]
        if got is None:
            return False, "no integer found in answer_text"
        ok = got == want
        return ok, ("" if ok else f"got {got} want {want}")
    if task["type"] == "json":
        if not arm.get("schema_valid"):
            return False, "schema_valid=false (or null) from Mojo"
        parsed = extract_json_obj(text)
        if parsed is None:
            return False, "answer_text did not parse as JSON"
        ok = parsed == task["expected"]
        return ok, ("" if ok else f"parsed {parsed!r} != expected {task['expected']!r}")
    return False, "unknown task type " + task["type"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("raw_json")
    ap.add_argument("tasks_json")
    ap.add_argument("out_json")
    ap.add_argument("out_md")
    ap.add_argument("--vram-before", type=int, default=None)
    ap.add_argument("--vram-after", type=int, default=None)
    ap.add_argument("--topology2-status", default=None)
    args = ap.parse_args()

    raw = json.loads(Path(args.raw_json).read_text())
    tasks = {t["id"]: t for t in json.loads(Path(args.tasks_json).read_text())}

    arm_names = []
    per_arm_correct = {}
    per_arm_total = {}
    per_arm_producer_s = {}
    per_type_arm_correct = {}
    per_type_arm_total = {}

    for item in raw["items"]:
        task = tasks.get(item["id"])
        if task is None:
            print("WARNING: item", item["id"], "not found in", args.tasks_json, file=sys.stderr)
            continue
        for arm in item["arms"]:
            name = arm["arm"]
            if name not in arm_names:
                arm_names.append(name)
                per_arm_correct[name] = 0
                per_arm_total[name] = 0
                per_arm_producer_s[name] = []
            ok, reason = score_arm(task, arm)
            arm["correct"] = ok
            arm["score_reason"] = reason
            per_arm_total[name] += 1
            if ok:
                per_arm_correct[name] += 1
            if arm.get("producer_s"):
                per_arm_producer_s[name].append(arm["producer_s"])
            key = (task["type"], name)
            per_type_arm_total[key] = per_type_arm_total.get(key, 0) + 1
            if ok:
                per_type_arm_correct[key] = per_type_arm_correct.get(key, 0) + 1

    raw["vram_before_bytes"] = args.vram_before
    raw["vram_after_bytes"] = args.vram_after
    raw["topology2_status"] = args.topology2_status

    Path(args.out_json).write_text(json.dumps(raw, indent=2))

    types = sorted({t["type"] for t in tasks.values()})
    lines = []
    lines.append(f"# E8 HARNESS results -- {raw.get('topology')}")
    lines.append("")
    lines.append(f"pack: `{raw.get('pack')}`  transport: {raw.get('transport')}")
    if args.vram_before is not None and args.vram_after is not None:
        lines.append(
            f"VRAM before both loads: {args.vram_before / 1e9:.2f} GB, "
            f"after both loads: {args.vram_after / 1e9:.2f} GB "
            f"(delta {(args.vram_after - args.vram_before) / 1e9:.2f} GB)"
        )
    if args.topology2_status:
        lines.append(f"Topology 2: {args.topology2_status}")
    lines.append("")
    lines.append("## Accuracy per arm per task type")
    lines.append("")
    header = "| arm | " + " | ".join(types) + " | overall |"
    lines.append(header)
    lines.append("|---|" + "---|" * (len(types) + 1))
    for name in arm_names:
        row = [name]
        for t in types:
            key = (t, name)
            c = per_type_arm_correct.get(key, 0)
            n = per_type_arm_total.get(key, 0)
            row.append(f"{c}/{n}" if n else "-")
        oc = per_arm_correct[name]
        on = per_arm_total[name]
        row.append(f"{oc}/{on}" if on else "-")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    lines.append("## Producer time (median, s)")
    lines.append("")
    lines.append("| arm | median producer_s | n |")
    lines.append("|---|---|---|")
    for name in arm_names:
        vals = per_arm_producer_s[name]
        med = f"{statistics.median(vals):.3f}" if vals else "-"
        lines.append(f"| {name} | {med} | {len(vals)} |")
    Path(args.out_md).write_text("\n".join(lines) + "\n")
    print("wrote", args.out_json, "and", args.out_md)


if __name__ == "__main__":
    main()
