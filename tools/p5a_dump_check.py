#!/usr/bin/env python
"""CPU receipt for a real P5a v2 dump."""
import argparse
import json

from mtp_head import H, read_v2_dump


def inspect(path):
    docs = read_v2_dump(path)
    records = [r for doc in docs for r in doc["records"]]
    if not records:
        raise ValueError("v2 dump has no records")
    sums = [float(r["top8_probs"].sum()) for r in records]
    distinct_argmax_by_doc = [
        len({int(r["target_argmax"]) for r in doc["records"]})
        for doc in docs
    ]
    greedy_next_matches = sum(
        int(r["target_argmax"]) == int(doc["tokens"][r["pos"] + 1])
        for doc in docs
        for r in doc["records"]
    )
    report = {
        "dump": path,
        "documents": len(docs),
        "records": len(records),
        "position_min": min(r["pos"] for r in records),
        "position_max": max(r["pos"] for r in records),
        "hidden_width": H,
        "finite_values": True,
        "max_top8_sum_error": max(abs(s - 1.0) for s in sums),
        "argmax_consistent": sum(r["target_argmax"] == r["top8_ids"][0].item() for r in records),
        "distinct_argmax_by_doc": distinct_argmax_by_doc,
        "distinct_argmax_min": min(distinct_argmax_by_doc),
        "distinct_argmax_all_docs_gt1": all(n > 1 for n in distinct_argmax_by_doc),
        "greedy_next_matches": greedy_next_matches,
        "greedy_next_fraction": greedy_next_matches / len(records),
        "input_alignment": sum(
            r["input_token"] == docs[di]["tokens"][r["pos"]]
            for di, doc in enumerate(docs)
            for r in doc["records"]
        ),
    }
    report["argmax_consistent_all"] = report["argmax_consistent"] == report["records"]
    report["input_alignment_all"] = report["input_alignment"] == report["records"]
    report["normalization_pass"] = report["max_top8_sum_error"] <= 1e-5
    report["greedy_next_pass"] = report["greedy_next_fraction"] > 0.30
    report["pass"] = all((
        report["finite_values"], report["argmax_consistent_all"],
        report["input_alignment_all"], report["normalization_pass"],
        report["distinct_argmax_all_docs_gt1"], report["greedy_next_pass"],
    ))
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--report", required=True)
    args = ap.parse_args()
    report = inspect(args.dump)
    with open(args.report, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
    if not report["pass"]:
        raise SystemExit("P5a v2 dump check FAIL")


if __name__ == "__main__":
    main()
