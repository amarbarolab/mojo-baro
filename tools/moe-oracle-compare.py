#!/usr/bin/env python3
"""Compare the first engine token's layer taps with eval-callback samples."""
import argparse
import array
import importlib.util
import math
import re


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("oracle")
    parser.add_argument("dump")
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location("oracle", "tools/llama-oracle.py")
    oracle = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(oracle)
    _, rows = oracle.parse(args.oracle)
    values = array.array("f")
    with open(args.dump, "rb") as stream:
        values.frombytes(stream.read(80 * 2048 * 4))
    assert len(values) == 80 * 2048, "missing first token"
    print("layer tensor sampled_rel_l2 max_abs ours_first3 oracle_first3")
    for layer in range(40):
        for half, name in enumerate(("attn_residual", "l_out")):
            row = rows[f"{name}-{layer}"][0]
            expected = [float(x) for x in re.findall(r"[-+]?\d+\.\d+(?:[eE][-+]?\d+)?", row)]
            assert len(expected) == 6, (name, layer, row)
            offset = (2 * layer + half) * 2048
            actual = [values[offset + i] for i in (0, 1, 2, 2045, 2046, 2047)]
            assert all(math.isfinite(x) for x in actual), (name, layer)
            delta = [a - b for a, b in zip(actual, expected)]
            rel = math.sqrt(sum(x*x for x in delta) / sum(x*x for x in expected))
            print(layer, name, f"{rel:.6f}", f"{max(map(abs, delta)):.6f}",
                  [round(x, 6) for x in actual[:3]], expected[:3])


if __name__ == "__main__":
    main()
