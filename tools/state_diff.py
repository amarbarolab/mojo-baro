#!/usr/bin/env python3
"""Compare two BAROST01 engine state files section by section (oracle).

Usage: tools/state_diff.py REFERENCE.state OTHER.state

Prints, for conv, SSM state, K and V (positions below pos only), the relative L2
error ||other - ref|| / ||ref|| and the max absolute difference. Used by
bench/llama-handoff.sh to check a llama.cpp-converted state against the engine's
own state at the same position: a layout error (transpose, head or layer order)
shows up as an O(1) relative error, rounding and kernel differences as small ones.
Exit 1 if the headers (pos, sizes, tokens) disagree.
"""
import sys

import numpy as np

KVPAGE, N_ATT, NKVH, HD = 128, 8, 4, 256


def load(path):
    b = open(path, "rb").read()
    assert b[:8] == b"BAROST01", path
    pos, conv_n, ssm_n, kvn = np.frombuffer(b[8:40], dtype="<i8")
    salt = b[40:72]
    o = 72
    toks = np.frombuffer(b[o:o + 4 * pos], dtype="<i4"); o += 4 * pos
    f = np.frombuffer(b[o:], dtype="<f4")
    conv = f[:conv_n]; ssm = f[conv_n:conv_n + ssm_n]
    k = f[conv_n + ssm_n:conv_n + ssm_n + kvn]; v = f[conv_n + ssm_n + kvn:conv_n + ssm_n + 2 * kvn]
    return dict(pos=int(pos), salt=salt, toks=toks, conv=conv, ssm=ssm, k=k, v=v)


def kv_valid(x, pos):
    pages = x.reshape(-1, N_ATT, NKVH, KVPAGE, HD)
    t = np.arange(pages.shape[0] * KVPAGE).reshape(-1, KVPAGE)
    mask = (t < pos)[:, None, None, :, None]
    return pages[np.broadcast_to(mask, pages.shape)]


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    a, b = load(sys.argv[1]), load(sys.argv[2])
    ok = a["pos"] == b["pos"] and a["conv"].size == b["conv"].size and a["ssm"].size == b["ssm"].size \
        and a["k"].size == b["k"].size and np.array_equal(a["toks"], b["toks"]) and a["salt"] == b["salt"]
    print(f"pos {a['pos']} vs {b['pos']}; tokens equal {np.array_equal(a['toks'], b['toks'])}; salt equal {a['salt'] == b['salt']}")
    if not ok:
        print("FAIL: headers disagree")
        sys.exit(1)
    for name in ("conv", "ssm", "k", "v"):
        x, y = a[name], b[name]
        if name in ("k", "v"):
            x, y = kv_valid(x, a["pos"]), kv_valid(y, a["pos"])
        rel = float(np.linalg.norm(y - x) / max(np.linalg.norm(x), 1e-30))
        print(f"{name:>4}: rel L2 {rel:.3e}  max abs {float(np.max(np.abs(y - x))):.3e}  ref norm {float(np.linalg.norm(x)):.3e}")


if __name__ == "__main__":
    main()
